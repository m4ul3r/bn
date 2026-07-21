from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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


DEFAULT_SPILL_TOKEN_LIMIT = 10_000
# Offline token estimate: ~3 bytes of UTF-8 per token. Deliberately
# conservative for the decompiled-code/JSON output this tool produces (which
# tokenizes denser than prose), so oversized output spills to disk a little
# early rather than flooding the consuming agent's context. This replaces a
# tiktoken dependency that downloaded the OpenAI BPE at runtime and crashed
# every command on offline machines.
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3


@dataclass(frozen=True)
class OutputWriteResult:
    rendered: str
    artifact: dict[str, Any] | None = None
    spilled: bool = False
    # #409: True when output did NOT spill but is within 20% of the threshold -- a
    # cheap preflight signal that a slightly larger read (next page / bigger fn) will
    # spill, so an agent can pre-emptively slice. Surfaced by the caller on stderr.
    near_spill: bool = False


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
        if isinstance(value, dict):
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


def _spill_path(stem: str, suffix: str) -> Path:
    now = datetime.now(timezone.utc)
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


def resolve_spill_limit(default: int = DEFAULT_SPILL_TOKEN_LIMIT) -> int:
    """The spill threshold in estimated tokens (#409). Overridable via the
    ``BN_SPILL_TOKENS`` env var so an agent with a bigger/smaller context window can
    raise or lower the on-disk-spill point (e.g. ``BN_SPILL_TOKENS=40000``). A
    non-positive / non-numeric value falls back to the default rather than disabling
    spill silently (an unbounded read could flood the context)."""
    raw = os.environ.get("BN_SPILL_TOKENS")
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip(), 0)
    except ValueError:
        return default
    return value if value > 0 else default


# Per-command rerun/slicing knob named in a spill envelope so an agent bounds the
# next read instead of guessing (#409). Keyed by the command's output `stem`.
_LIST_SLICE_STEMS = frozenset({
    "functions", "function-search", "imports", "exports", "strings", "sections",
    "class-list", "comments", "callsites", "evidence-xrefs", "go-functions", "xrefs",
    "types", "taint-models", "field-xrefs",
})


def _rerun_hint(stem: str | None) -> str:
    s = (stem or "").lower()
    if s in _LIST_SLICE_STEMS or s.endswith("-list"):
        return "bound the next read with --limit N (and --offset K to page), or read the spilled file"
    if s in ("disasm", "il", "structured-il"):
        return ("bound the next read with --lines START:END (function/CFG order) or "
                "--linear N at an address, or read the spilled file")
    if s == "function-evidence":
        return "bound the next read with --limit N / --address-window A:B, or read the spilled file"
    if s in ("decompile", "function-bundle"):
        return "narrow the scope or read the spilled file at artifact_path (no in-line slicing knob)"
    return "read the spilled file at artifact_path, or re-run with a slicing flag (--limit/--offset/--lines) or --out"


def _artifact_payload(
    *,
    artifact_path: Path,
    fmt: str,
    encoded: bytes,
    token_count: int,
    value: Any,
    spilled: bool,
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


def render_error(message: str, fmt: str) -> str:
    """Render an error as a machine-readable envelope under json/ndjson.

    Routes through :func:`render_value` so error envelopes match successful
    JSON output (compact, ``sort_keys=True`` since #215) instead of a divergent
    hand-rolled ``json.dumps``. Lets ``bn ... --format json | jq`` parse an error
    object rather than an empty stream.
    """
    return render_value({"ok": False, "error": message}, fmt)


def write_output_result(
    value: Any,
    *,
    fmt: str,
    out_path: Path | None,
    stem: str,
    spill_token_limit: int | None = None,
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
        )
        return OutputWriteResult(
            rendered=render_envelope(artifact, fmt),
            artifact=artifact,
            spilled=False,
        )

    if token_count <= spill_token_limit:
        # #409: flag a read that fit but is within 20% of the threshold, so the caller
        # can warn that a slightly larger next read (next page / bigger function) will
        # spill -- a preflight signal without a second run.
        near = token_count >= (spill_token_limit * 4) // 5
        return OutputWriteResult(rendered=rendered, near_spill=near)

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
    )
    # #409: name the command-specific slicing knob + the threshold that tripped, so
    # the agent bounds the next read instead of re-running blind. BN_SPILL_TOKENS
    # raises/lowers the threshold.
    artifact["rerun"] = _rerun_hint(stem)
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
