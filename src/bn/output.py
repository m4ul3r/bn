from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .paths import ensure_private_dir, spill_root
from .transport import BridgeError


class OutputWriteError(BridgeError):
    """Raised when an explicitly requested --out file cannot be written.

    Subclasses ``BridgeError`` so ``cli.main()`` prints it as a clean one-line
    message instead of a traceback (that is the only exception type the CLI
    layer turns into clean output).
    """


# Rendered payload size (in estimated tokens) past which the caller prints its
# slicing note instead of staying silent. NOT a spill threshold: spilling is
# opt-in via BN_SPILL_TOKENS (#409), so by default a payload this large goes to
# stdout whole and the consuming agent's harness bounds what it reads.
DEFAULT_SLICE_NOTE_TOKENS = 10_000
# Offline token estimate: ~3 bytes of UTF-8 per token. Deliberately
# conservative for the decompiled-code/JSON output this tool produces (which
# tokenizes denser than prose), so a large payload is flagged a little early
# rather than flooding the consuming agent's context. This replaces a
# tiktoken dependency that downloaded the OpenAI BPE at runtime and crashed
# every command on offline machines.
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3


@dataclass(frozen=True)
class OutputWriteResult:
    rendered: str
    artifact: dict[str, Any] | None = None
    spilled: bool = False
    # #409: True when output did NOT spill but is within 20% of the CONFIGURED spill
    # threshold -- a cheap preflight signal that a slightly larger read (next page /
    # bigger fn) will spill, so an agent can pre-emptively slice. Surfaced by the
    # caller on stderr, and only ever set when a threshold is configured at all.
    near_spill: bool = False
    # Estimated tokens of the RENDERED payload, always populated. Lets the caller
    # name a size in the slicing note without re-encoding the string.
    token_count: int = 0
    # No spill threshold configured and the payload exceeds DEFAULT_SLICE_NOTE_TOKENS:
    # nothing was written to disk, but a consuming agent truncates tool output this
    # large. Mutually exclusive with `near_spill` (which only exists under opt-in).
    truncation_risk: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def render_value(value: Any, fmt: str) -> str:
    if fmt == "json":
        # COMPACT json (no indent): pretty-printing inflated structured/list output
        # ~3x, tripping the spill threshold so early that `function list --format
        # json | jq` of a few-hundred-function binary already read the spill
        # envelope instead of the data (#215). Compact ~3x's the pre-spill ceiling;
        # `| jq` re-pretties for humans. sort_keys keeps output deterministic.
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n"

    if fmt == "ndjson":
        if isinstance(value, list):
            lines = [
                json.dumps(item, sort_keys=True, default=_json_default) for item in value
            ]
            return "\n".join(lines) + ("\n" if lines else "")
        # A paged-list envelope ({items|functions:[...], total, offset, ...})
        # is the common ndjson target; emit ONE record per item per line, then a
        # trailing {"_meta": true, ...paging...} line -- actual newline-delimited
        # streaming, not the whole envelope collapsed onto a single line (which
        # was identical to compact --format json and defeated the point). (J5)
        # `_meta` is a SENTINEL this fan-out invents, so a payload that already
        # carries that key cannot be represented: copying the non-page keys into
        # the trailing record would silently overwrite the caller's own value. An
        # artifact is the caller's data -- write it whole as one record rather
        # than a stream that lost a field. Mirrors the bridge-side --out writer,
        # `_shared.py::_write_json_artifact`; the two must stay interchangeable.
        if isinstance(value, dict) and "_meta" not in value:
            for page_key in ("items", "functions"):
                page = value.get(page_key)
                if isinstance(page, list):
                    lines = [
                        json.dumps(item, sort_keys=True, default=_json_default)
                        for item in page
                    ]
                    meta = {
                        k: v for k, v in value.items()
                        if k not in ("items", "functions")
                    }
                    meta["_meta"] = True
                    lines.append(json.dumps(meta, sort_keys=True, default=_json_default))
                    return "\n".join(lines) + "\n"
        return json.dumps(value, sort_keys=True, default=_json_default) + "\n"

    if isinstance(value, str):
        return value if value.endswith("\n") else value + "\n"
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        # A paged-list envelope ({items|functions:[...], total, ...}) must report
        # the array's element count (and the logical total), NOT the count of
        # envelope keys -- the latter misled callers about result size (e.g. a
        # 486-item page summarized as count=6).
        for page_key in ("items", "functions"):
            page = value.get(page_key)
            if isinstance(page, list):
                out: dict[str, Any] = {
                    "kind": "object",
                    "keys": sorted(value.keys())[:10],
                    "page_key": page_key,
                    "count": len(page),
                }
                total = value.get("total")
                if isinstance(total, int):
                    out["total"] = total
                return out
        return {"kind": "object", "keys": sorted(value.keys())[:10], "count": len(value)}
    if isinstance(value, list):
        return {"kind": "array", "count": len(value)}
    if isinstance(value, str):
        return {"kind": "string", "chars": len(value)}
    return {"kind": type(value).__name__}


DEFAULT_SPILL_RETENTION_DAYS = 14
# Set once a process has swept, so a command that spills fifty pages pays for
# the `iterdir()` once rather than fifty times.
_spill_pruned = False


def resolve_spill_retention_days(default: int = DEFAULT_SPILL_RETENTION_DAYS) -> int:
    """How many days of spill artifacts to keep (#591).

    ``BN_SPILL_RETENTION_DAYS=0`` disables pruning for an engagement that must
    keep every artifact. A negative or non-numeric value falls back to the
    default: the failure mode of a typo must not be silently unbounded growth,
    which is the bug being fixed.
    """
    raw = os.environ.get("BN_SPILL_RETENTION_DAYS")
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip(), 0)
    except ValueError:
        return default
    return value if value >= 0 else default


def _prune_old_spill_days(root: Path, today: date) -> list[Path]:
    """Remove whole spill day-directories older than the retention window.

    Spills already live in ``spills/<YYYYMMDD>/``, so retention costs one
    ``iterdir()`` over a few dozen names and zero per-file stats -- the
    measured 1.0 GB / 4187-file spill root is reclaimed a day at a time.

    Only an entry that is a directory AND whose name is the CANONICAL
    ``%Y%m%d`` rendering of a date is eligible. Anything else in the spill root
    -- a user's notes, another tool's file, a partially-named dir -- is
    evidence of something this function did not create, so it is left alone
    (#618: act on evidence, never on its absence).

    ``strptime`` alone is NOT that check: its numeric fields accept unpadded
    input, so ``202611`` parses happily as 2026-01-01 and a directory by that
    name -- which this writer, which always calls ``strftime``, could not have
    produced -- was recursively deleted. The round-trip is the check: a name is
    eligible only if formatting the parsed date reproduces the name byte for
    byte.
    """
    retention = resolve_spill_retention_days()
    if retention == 0:
        return []
    try:
        cutoff = today - timedelta(days=retention)
    except OverflowError:
        # A window wider than the calendar is a request to keep everything, and
        # it must not become an exception on the output path: the spill write's
        # fallback catches OSError, so an OverflowError here would deny the
        # caller its result AND its artifact over a config value.
        return []
    removed: list[Path] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        try:
            day = datetime.strptime(entry.name, "%Y%m%d").date()
        except ValueError:
            continue
        if day.strftime("%Y%m%d") != entry.name:
            continue
        if day >= cutoff or not entry.is_dir():
            continue
        # A racing peer may remove the same stale day; that is the outcome we
        # wanted either way, so a failure here is never the command's problem.
        with contextlib.suppress(OSError):
            shutil.rmtree(entry)
            removed.append(entry)
    return removed


def _spill_path(stem: str, suffix: str) -> Path:
    global _spill_pruned
    now = datetime.now(timezone.utc)
    if not _spill_pruned:
        _spill_pruned = True
        removed = _prune_old_spill_days(spill_root(), now.date())
        if removed:
            print(
                f"note: pruned {len(removed)} spill day(s) older than "
                f"{resolve_spill_retention_days()} days from {spill_root()} "
                "(set BN_SPILL_RETENTION_DAYS=0 to keep everything)",
                file=sys.stderr,
            )
    # spill_root() is already private (0o700); tighten the per-day subdir too so a
    # permissive umask can't leave decompiled artifacts group/world-readable (#612).
    directory = ensure_private_dir(spill_root() / now.strftime("%Y%m%d"))
    # pid + random component: parallel agents spilling in the same second
    # must not clobber each other's artifacts.
    unique = f"{os.getpid()}-{secrets.token_hex(2)}"
    return directory / f"{stem}-{now.strftime('%H%M%S')}-{unique}{suffix}"


def _write_private_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* created owner-only (``0o600``) regardless of umask.

    Spill artifacts hold decompiled output from real targets, so they must never
    be born group/world-readable under a permissive umask -- ``Path.write_bytes``
    would open at ``0o666 & ~umask``. Opening with an explicit ``0o600`` creation
    mode (umask can only clear bits, never add them) closes that window without a
    create-then-chmod race (#612). Spill paths are freshly randomized names, so
    ``O_CREAT`` always makes a new file and the mode is authoritative.
    """
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def estimate_tokens(encoded: bytes) -> int:
    return -(-len(encoded) // TOKEN_ESTIMATE_BYTES_PER_TOKEN)


def resolve_spill_limit() -> int | None:
    """The opt-in spill threshold in estimated tokens (#409), or ``None``.

    Spilling is OFF unless ``BN_SPILL_TOKENS`` names a positive integer: the
    default is to write the full rendered payload to stdout and let the
    consuming agent's harness bound what it reads. Unset, empty, whitespace,
    non-numeric, zero and negative all mean "no spill" -- a typo can never
    silently re-arm disk output.

    A caller that passes ``spill_token_limit=`` explicitly bypasses this
    resolver entirely and forces the threshold it named (test and programmatic
    use).
    """
    raw = os.environ.get("BN_SPILL_TOKENS")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip(), 0)
    except ValueError:
        return None
    return value if value > 0 else None


# Per-command rerun/slicing knob named in a spill envelope so an agent bounds the
# next read instead of guessing (#409). Keyed by the command's output `stem`.
_LIST_SLICE_STEMS = frozenset({
    "functions", "function-search", "imports", "exports", "strings", "sections",
    "class-list", "comments", "callsites", "evidence-xrefs", "go-functions", "xrefs",
    "types", "taint-models", "field-xrefs",
})


def _rerun_hint(stem: str | None, fmt: str = "text") -> str:
    s = (stem or "").lower()
    # The pointer key the consumer actually sees: the text envelope renders
    # `path:` (render_artifact_envelope maps artifact_path -> path), json/ndjson
    # carry `artifact_path`. Naming the wrong key sends a model hunting for a
    # field that is not there.
    pointer = "path" if fmt == "text" else "artifact_path"
    if s in _LIST_SLICE_STEMS or s.endswith("-list"):
        return "bound the next read with --limit N (and --offset K to page), or read the spilled file"
    if s in ("disasm", "il", "structured-il"):
        if fmt != "text":
            # --lines is text-only; suggesting it to a json/ndjson consumer is a
            # dead end (#120).
            return f"read the spilled file at `{pointer}`, or re-run with --out FILE"
        return ("rerun with --lines START:END (function/CFG order) or "
                f"--linear N at an address, or read the spilled file at `{pointer}`")
    if s == "function-evidence":
        return "bound the next read with --limit N / --address-window A:B, or read the spilled file"
    if s == "decompile":
        if fmt == "text":
            return f"rerun with --lines START:END to fetch a slice, or read the spilled file at `{pointer}`"
        return f"read the spilled file at `{pointer}`, or re-run with --out FILE"
    if s == "function-bundle":
        return f"narrow the scope or read the spilled file at `{pointer}`"
    return (f"read the spilled file at `{pointer}`, or re-run with a slicing flag "
            "(--limit/--offset/--lines) or --out")


def _artifact_payload(
    *,
    artifact_path: Path,
    fmt: str,
    encoded: bytes,
    token_count: int,
    value: Any,
    spilled: bool,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "spilled": spilled,
        "artifact_path": str(artifact_path),
        "format": fmt,
        "bytes": len(encoded),
        "tokens": token_count,
        "tokenizer": "estimate",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "summary": _summary(value),
    }
    # #653.8: WHICH target/instance produced this artifact. Two agents sharing a
    # scratchpad both wrote `fns.json`; one silently read the other's list -- a
    # different target, a different binary -- and concluded its own name recovery
    # covered 6 of 1006 functions. Nothing in the artifact made that detectable.
    # (`sha256` above is the digest of THIS artifact's bytes, not of the binary.)
    for key, val in (provenance or {}).items():
        if val is not None:
            payload.setdefault(key, val)
    # Hoist the canonical logical total to the TOP LEVEL so `jq '.total'` returns
    # the real count whether or not the read spilled (#311). On a spilled
    # envelope the items live on disk at artifact_path, so `jq '.items'` reads
    # null and a sink with 120 callers misreads as "0"; `.total` (and the
    # `spilled: true` flag) are the canonical, spill-stable signals.
    summary = payload["summary"]
    if isinstance(summary, dict) and isinstance(summary.get("total"), int):
        payload["total"] = summary["total"]
    return payload


def _format_envelope_value(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return str(value)


def render_artifact_envelope(payload: dict[str, Any]) -> str:
    lines = []
    if "ok" in payload:
        lines.append(f"ok: {str(bool(payload.get('ok'))).lower()}")
    if "spilled" in payload:
        lines.append(f"spilled: {str(bool(payload.get('spilled'))).lower()}")
    if "artifact_path" in payload:
        lines.append(f"path: {payload['artifact_path']}")
    for key in ("format", "bytes", "tokens", "tokenizer", "sha256"):
        if key in payload:
            lines.append(f"{key}: {payload[key]}")
    # #409: surface the token threshold + the rerun/slicing hint so an agent bounds
    # the next read (only present on a spilled envelope).
    if "spill_token_limit" in payload:
        lines.append(f"spill_token_limit: {payload['spill_token_limit']}")
    if payload.get("rerun"):
        lines.append(f"rerun: {payload['rerun']}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_parts = []
        kind = summary.get("kind")
        if kind is not None:
            summary_parts.append(f"kind={_format_envelope_value(kind)}")
        for key in sorted(summary):
            if key == "kind":
                continue
            summary_parts.append(f"{key}={_format_envelope_value(summary[key])}")
        if summary_parts:
            lines.append(f"summary: {' '.join(summary_parts)}")
    return "\n".join(lines) + "\n"


def render_envelope(payload: dict[str, Any], fmt: str) -> str:
    """Render a spill/``--out`` artifact envelope honoring the requested format.

    The envelope must itself be valid JSON under ``json``/``ndjson`` so that
    ``bn <cmd> --format json | jq`` keeps working when output spills to disk or
    is redirected with ``--out`` (the spilled/written file already holds the
    real payload in the requested format). Only ``text`` gets the human-readable
    ``key: value`` form.
    """
    if fmt in ("json", "ndjson"):
        # payload is always a dict here, so render_value's json/ndjson branches
        # produce byte-identical output -- share them rather than keep a second
        # copy of the indent/sort_keys/default settings that could drift.
        return render_value(payload, fmt)
    return render_artifact_envelope(payload)


def render_error(
    message: str,
    fmt: str,
    *,
    status: str | None = None,
    requested: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
) -> str:
    """Render an error as a machine-readable envelope under json/ndjson.

    Routes through :func:`render_value` so error envelopes match successful
    JSON output (compact, ``sort_keys=True`` since #215) instead of a divergent
    hand-rolled ``json.dumps``. Lets ``bn ... --format json | jq`` parse an error
    object rather than an empty stream. *status*/*requested*/*observed* mirror
    the structured fields an escaped bridge ``OperationFailure`` carries; they
    are omitted from the envelope when ``None`` (unstructured/transport errors).
    """
    payload: dict[str, Any] = {"ok": False, "error": message}
    if status is not None:
        payload["status"] = status
    if requested is not None:
        payload["requested"] = requested
    if observed is not None:
        payload["observed"] = observed
    return render_value(payload, fmt)


def write_output_result(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_token_limit: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> OutputWriteResult:
    # #409: resolve the spill threshold from BN_SPILL_TOKENS when not explicitly set.
    if spill_token_limit is None:
        spill_token_limit = resolve_spill_limit()
    rendered = render_value(value, fmt)
    encoded = rendered.encode("utf-8")
    token_count = estimate_tokens(encoded)

    if out_path is not None:
        try:
            # A user-chosen --out destination, NOT a private cache dir: honor the
            # caller's own umask/permissions here rather than forcing 0o700 on a
            # directory they explicitly named (may be shared/intentionally group-
            # readable). ensure_private_dir is only for our cache/spill tree.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(encoded)
        except OSError as exc:
            # The user explicitly asked for a file; silently falling back to
            # stdout would be wrong. Fail with a clean message instead.
            raise OutputWriteError(f"Failed to write --out file {out_path}: {exc}") from exc
        artifact = _artifact_payload(
            artifact_path=out_path,
            fmt=fmt,
            encoded=encoded,
            token_count=token_count,
            value=value,
            spilled=False,
            provenance=provenance,
        )
        return OutputWriteResult(
            rendered=render_envelope(artifact, fmt),
            artifact=artifact,
            spilled=False,
            token_count=token_count,
        )

    if spill_token_limit is None or token_count <= spill_token_limit:
        if spill_token_limit is None:
            # Opt-in spill is off: the full payload is on stdout, so a read this
            # large is bounded by the consumer's harness, not by us.
            return OutputWriteResult(
                rendered=rendered,
                token_count=token_count,
                truncation_risk=token_count >= DEFAULT_SLICE_NOTE_TOKENS,
            )
        # #409: fit, but within 20% of the CONFIGURED threshold, so the caller can warn
        # that a slightly larger next read (next page / bigger function) will spill --
        # a preflight signal without a second run.
        near = token_count >= (spill_token_limit * 4) // 5
        return OutputWriteResult(rendered=rendered, near_spill=near, token_count=token_count)

    suffix = ".ndjson" if fmt == "ndjson" else ".txt" if fmt == "text" else ".json"
    try:
        spill_path = _spill_path(stem, suffix)
        _write_private_bytes(spill_path, encoded)
    except OSError as exc:
        # The rendered output is already in memory; losing it over a failed
        # spill write (disk full, permissions) would punish the user twice.
        print(
            f"warning: failed to write spill artifact ({exc}); printing full output",
            file=sys.stderr,
        )
        return OutputWriteResult(rendered=rendered)
    artifact = _artifact_payload(
        artifact_path=spill_path,
        fmt=fmt,
        encoded=encoded,
        token_count=token_count,
        value=value,
        spilled=True,
        provenance=provenance,
    )
    # #409: name the command-specific slicing knob + the threshold that tripped, so
    # the agent bounds the next read instead of re-running blind. BN_SPILL_TOKENS
    # raises/lowers the threshold.
    artifact["rerun"] = _rerun_hint(stem, fmt)
    artifact["spill_token_limit"] = spill_token_limit
    return OutputWriteResult(
        rendered=render_envelope(artifact, fmt),
        artifact=artifact,
        spilled=True,
    )


def write_bytes_result(
    data: bytes,
    *,
    out_path: Path | None,
    fmt: str,
    summary: dict[str, Any] | None = None,
) -> OutputWriteResult:
    """Write raw *data* to *out_path* with the same guarantees as
    :func:`write_output_result`: create parent dirs, wrap OSError in a clean
    OutputWriteError (not a raw traceback), and return an artifact envelope
    (path/sha256/size, ``format: bytes``). The previous raw-bytes path did
    ``out_path.write_bytes(data)`` directly -- no mkdir, no error wrap, no
    envelope (#96). With no out_path the caller writes the raw bytes to stdout.
    """
    if out_path is None:
        return OutputWriteResult(rendered="")
    try:
        # User-chosen --out destination (see write_output_result): honor the
        # caller's permissions, don't force 0o700 as ensure_private_dir would.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    except OSError as exc:
        raise OutputWriteError(f"Failed to write --out file {out_path}: {exc}") from exc
    artifact = {
        "ok": True,
        "spilled": False,
        "artifact_path": str(out_path),
        "format": "bytes",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if summary is not None:
        artifact["summary"] = summary
    return OutputWriteResult(
        rendered=render_envelope(artifact, fmt),
        artifact=artifact,
        spilled=False,
    )


def write_output(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_token_limit: int | None = None,  # #409: None resolves BN_SPILL_TOKENS
) -> str:
    return write_output_result(
        value,
        fmt=fmt,
        out_path=out_path,
        stem=stem,
        spill_token_limit=spill_token_limit,
    ).rendered
