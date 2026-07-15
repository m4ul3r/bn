"""Module-free helper layer shared across the bridge and its domain modules.

These are the small, state-free helpers (and ``OperationFailure``) that the
bridge dispatch, the read-op domains, and the mutation engine all need. They
were hoisted out of ``bridge.py`` so that ``seam.py`` and the later domain
modules can import them WITHOUT importing ``bridge.py`` (which would create an
import cycle). This module imports ONLY stdlib + binaryninja -- never bridge.
"""
from __future__ import annotations

import hashlib
import json
import re
import traceback
from pathlib import Path
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]
try:
    from binaryninja.mainthread import execute_on_main_thread_and_wait, is_main_thread
except ModuleNotFoundError:  # no BN GUI runtime (tests, headless tooling): run inline
    def execute_on_main_thread_and_wait(func):
        return func()

    def is_main_thread() -> bool:
        return True


def _json_response(*, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "result": result, "error": error}


def _symbol_type_name(fn: Any) -> str | None:
    """The symbol-type enum-member name of a function (e.g. ``FunctionSymbol``,
    ``ImportedFunctionSymbol``), or None. Guards the WHOLE access: BN's
    ``Function.symbol`` property asserts its core symbol is non-None, so a rare
    invariant-violating None would raise an ``AssertionError`` that plain
    ``getattr(..., None)`` does NOT swallow -- catch it so name resolution never
    crashes on a weird binary (#122)."""
    try:
        sym = getattr(fn, "symbol", None)
        return getattr(getattr(sym, "type", None), "name", None)
    except Exception:
        return None


def _format_ambiguous_function_error(identifier: Any, matches: list[Any]) -> str:
    lines = [f"Ambiguous function identifier: {identifier} matches {len(matches)} functions:"]
    for fn in sorted(matches, key=lambda f: int(f.start)):
        # Show each candidate's symbol kind (e.g. [FunctionSymbol] /
        # [ImportedFunctionSymbol]) so the collision self-documents -- the stub
        # vs implementation is then obvious (#122). Mirrors the symbol variant.
        kind = _symbol_type_name(fn) or ""
        suffix = f"  [{kind}]" if kind else ""
        lines.append(f"  {int(fn.start):#010x}  {str(fn.name)}{suffix}")
    lines.append("retry with one of the addresses above (e.g. `bn function info 0x…`)")
    return "\n".join(lines)


def _format_ambiguous_symbol_error(identifier: Any, matches: list[Any]) -> str:
    lines = [f"Ambiguous symbol identifier: {identifier} matches {len(matches)} symbols:"]
    for sym in sorted(matches, key=lambda s: int(s.address)):
        kind = getattr(getattr(sym, "type", None), "name", "") or str(getattr(sym, "type", ""))
        lines.append(f"  {int(sym.address):#010x}  {str(sym.name)}  [{kind}]")
    lines.append("retry with one of the addresses above")
    return "\n".join(lines)


def _serialize_error(exc: BaseException) -> str:
    """Render an exception for the user-facing error field.

    User-facing errors (``OperationFailure`` and the plain ``RuntimeError`` /
    ``ValueError`` instances raised throughout the bridge to report bad input or
    missing targets) carry an already-actionable message, so we emit ``str(exc)``
    verbatim. Anything outside that whitelist is treated as an unexpected bug and
    prefixed with ``internal error:`` plus its class name so callers can tell the
    difference without us leaking raw Python class names into normal errors.
    """

    # OperationFailure subclasses RuntimeError, so it is already covered by the
    # tuple below; it is listed explicitly to document the intent.
    if isinstance(exc, USER_FACING_ERRORS):
        return str(exc)
    return f"internal error: {type(exc).__name__}: {exc}"


class OperationFailure(RuntimeError):
    def __init__(
        self,
        status: str,
        message: str,
        *,
        requested: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.requested = requested or {}
        self.observed = observed or {}


# Exception classes whose messages are already user-facing and actionable.
# OperationFailure is a RuntimeError subclass; RuntimeError covers the bulk of
# "Function not found"/"Type not found"/"Unknown target selector" style errors,
# and ValueError covers argument-validation errors like "Unknown operation".
USER_FACING_ERRORS: tuple[type[BaseException], ...] = (OperationFailure, RuntimeError, ValueError)


def _validate_count(value: Any, *, label: str, minimum: int, allow_none: bool = False) -> int | None:
    """Coerce a limit/offset param to int and enforce *minimum*.

    The CLI argparse layer already rejects out-of-range count/limit/offset
    flags, but non-CLI callers (``bn py exec``, a raw socket client) reach the
    op handlers directly, so the bridge re-enforces the same contract: a
    negative limit must not silently drop the tail via Python slice math, and a
    zero limit must not return a degenerate empty-but-"truncated" result.
    """
    if value is None:
        if allow_none:
            return None
        raise OperationFailure("invalid_request", f"{label} is required")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise OperationFailure("invalid_request", f"{label} must be an integer, got {value!r}")
    if n < minimum:
        raise OperationFailure("invalid_request", f"{label} must be >= {minimum}, got {n}")
    return n


def _require_mapped_address(bv, address: int) -> None:
    """Raise if *address* is affirmatively unmapped in *bv*.

    Address-taking reads (``read``/``decompile``) reject an unmapped address, but
    ``xrefs``/``comment get`` historically accepted one as a benign empty result
    (exit 0), so a typo'd or stale address silently misread as a real
    "0 callers" / "no comment" (#374). This restores parity.

    A no-op when the view cannot answer (no ``is_valid_offset``, or it raises):
    callers must NOT reject on an indeterminate result, so a *mapped* address with
    zero refs stays a clean ``0`` / exit 0 -- only the genuinely-unmapped case is
    rejected. Real BN always provides ``is_valid_offset``.
    """
    is_valid = getattr(bv, "is_valid_offset", None)
    if not callable(is_valid):
        return
    try:
        mapped = bool(is_valid(int(address)))
    except Exception:
        return
    if not mapped:
        raise RuntimeError(f"Address {hex(int(address))} is not mapped in this binary")


def _validate_bool(value: Any, *, label: str, default: bool) -> bool:
    """Require a known boolean param to be an actual JSON boolean.

    The CLI always sends real booleans, but raw socket clients, ``py exec``
    callers and batch manifests can send strings. Plain truthiness coercion is
    destructive here: ``bool("false")`` is ``True``, so ``"all": "false"`` would
    close every target and ``"quick": "false"`` would enable quick load. Reject
    anything that isn't a JSON ``true``/``false`` with a clean invalid_request
    (#91). ``None`` (param absent) takes *default*.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise OperationFailure(
        "invalid_request",
        f"{label} must be a JSON boolean (true or false), got {value!r}",
    )


def _run_on_main_thread(func):
    if is_main_thread():
        return func()

    holder: dict[str, Any] = {}

    def wrapper():
        try:
            holder["result"] = func()
        except Exception as exc:  # pragma: no cover - exercised inside GUI
            holder["error"] = exc
            holder["traceback"] = traceback.format_exc()

    execute_on_main_thread_and_wait(wrapper)
    if "error" in holder:
        exc = holder["error"]
        if "traceback" in holder:
            bn.log_error(holder["traceback"])
        raise exc
    return holder.get("result")


_CONVENTION_RE = re.compile(r'\s*__convention\("[^"]*"\)\s*')
# BN-inferred function-type attributes, rendered as a trailing token (e.g.
# ``int64_t() __pure``). Matched as whole words so a parameter/type that merely
# contains the substring is untouched (#199).
_ATTRIBUTE_RE = re.compile(r'\s*\b__(?:pure|noreturn)\b')


def _normalize_prototype(proto: str) -> str:
    """Normalize a prototype for verification comparison.

    Strips BN-inferred annotations that a requested prototype need not carry but
    that analysis re-adds on readback -- the ``__convention("...")`` calling
    convention and the ``__pure`` / ``__noreturn`` function attributes -- and
    collapses whitespace. Without this, a valid edit on a function BN tags
    ``__pure`` (common on accessors) reads back as a textual mismatch and is
    reported ``verification_failed`` + reverted, even though the requested type
    landed (#199)."""
    stripped = _CONVENTION_RE.sub("", proto)
    stripped = _ATTRIBUTE_RE.sub("", stripped)
    return " ".join(stripped.split())


def _parse_address(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    except ValueError:
        raise ValueError(
            f"{value!r} is not a valid address; expected a decimal or 0x-prefixed hex value"
        ) from None


def _artifact_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"kind": "object", "keys": sorted(value.keys())[:10], "count": len(value)}
    if isinstance(value, list):
        return {"kind": "array", "count": len(value)}
    if isinstance(value, str):
        return {"kind": "string", "chars": len(value)}
    return {"kind": type(value).__name__}


def _write_json_artifact(path_text: str | None, payload: Any) -> dict[str, Any] | None:
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return {
        "ok": True,
        "artifact_path": str(path),
        "format": "json",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "summary": _artifact_summary(payload),
    }


# BN's built-in tag-type names (probed on BN 5.4). BN itself offers NO protection
# against removing these -- `remove_tag_type("Bugs")` succeeds silently -- so the
# tag-type removal op guards against them, and `tag types` marks them is_builtin.
# There is no programmatic "builtin" flag on TagType (both built-in and
# user-created report TagTypeType.UserTagType), so this name set is the signal.
_BUILTIN_TAG_TYPE_NAMES: frozenset[str] = frozenset({
    "Bookmarks", "Bugs", "Crashes", "Important", "Library", "Needs Analysis",
    "Unresolved Stack Adjustment", "Unresolved Indirect Control Flow",
    "Unresolved Stack Pointer Value", "Invalid Instruction",
    "Could not Generate Flag IL", "Unimplemented Instruction (LLIL)",
    "Unimplemented Instruction (MLIL)", "Unimplemented Instruction (HLIL)",
    "Non-code Branch", "Function too Large",
    "Function Exceeded Max Analysis Time", "Jump to Unhandled Relocation",
    "Jump to Malformed Target", "WARP", "WARP: Ignored Function",
})
