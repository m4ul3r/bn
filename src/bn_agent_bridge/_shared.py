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


# Row keys per collection `kind`. Collection rows differ ON PURPOSE -- functions
# key on address/size, sections on start/end/length, callsites nest
# callee/containing_function -- and an agent filtering rows locally had no
# in-band way to learn which schema it was holding: `brief(rows, "address")` on
# sections was the only feedback, after the read had already been paid for.
#
# This map exists ONLY for the empty-page case, where there is no runtime row to
# read the schema off. Populated pages derive the hint from their own rows, so a
# renamed or added key can never leave a populated read advertising a stale
# schema; `test_declared_row_fields_match_the_rows_the_bridge_builds` holds the
# declarations to the same standard for the empty case.
#
# A key that a producer sets only CONDITIONALLY (not on every row of a kind --
# e.g. a relocation-recovered import's `provenance`, a segment-backed section's
# permission fields, a fn-pointer-scan hit's `function_pointer`/`thumb_pointer`,
# a variadic callee's `callee_variadic`) MUST still be declared here, not just
# left for a populated read to surface (#694 item 9). A zero-hit page is
# EXACTLY the case where there is no row to infer the schema from, so leaving
# an optional key out here means the one time in-band schema discovery is the
# only source of truth, it lies.
_DECLARED_ROW_FIELDS: dict[str, tuple[str, ...]] = {
    "functions": (
        "address", "name", "raw_name", "display_name", "size", "size_known",
        "imported", "auto_named", "basic_block_count",
    ),
    "strings": (
        "address", "value", "chars", "length", "type",
        "format_directives", "directive_count", "code_refs",
    ),
    "imports": (
        "address", "name", "raw_name", "kind", "library", "namespace",
        "provenance",
    ),
    "exports": (
        "address", "name", "raw_name", "display_name", "kind", "binding",
        "ordinal",
    ),
    "sections": (
        "name", "start", "end", "length", "semantics",
        "readable", "writable", "executable", "writable_executable",
        "permission_source",
    ),
    "xrefs": (
        "address", "kind", "function", "caller_function", "context",
        "function_pointer", "thumb_pointer",
    ),
    "callsites": (
        "callee", "containing_function", "call_addr", "call_kind",
        "caller_static", "instruction_length", "call_instruction",
        "previous_instructions", "next_instructions", "hlil_statement",
        "hlil_statement_reason", "pre_branch_condition", "callee_variadic",
        "call_index", "within_query",
    ),
}


def _annotate_row_fields(result: Any) -> Any:
    """Attach `row_fields` to a collection envelope, in place.

    Applied once at the dispatch boundary so every collection read -- current and
    future, CLI and native -- carries the hint without a second catalog for the
    CLI or the kernel to re-declare. Declared keys lead (a stable documented
    order), then any key an actual row carries, so an undeclared `kind` still
    self-describes from its rows. Non-collection results are returned untouched:
    a row hint on `decompile` text would be a lie.
    """
    if not isinstance(result, dict) or "row_fields" in result:
        return result
    kind = result.get("kind")
    items = result.get("items")
    if not isinstance(kind, str) or not isinstance(items, list):
        return result
    fields = list(_DECLARED_ROW_FIELDS.get(kind, ()))
    seen = set(fields)
    for row in items:
        if not isinstance(row, dict):
            # Not a row collection (e.g. a list of strings): no field hint.
            return result
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if fields:
        result["row_fields"] = fields
    return result


# Symbol types whose NAME comes from relocations/imports, not from analysis or a
# human. Counting these as "named" badly overstates how much real code is named
# on a stripped binary (PLT import trampolines dominate it), so they get their
# own bucket (#122). Compared by enum-member name to avoid importing the enum.
IMPORT_SYMBOL_TYPE_NAMES = frozenset(
    {"ImportedFunctionSymbol", "ImportAddressSymbol", "ExternalSymbol"}
)


def is_imported_function(fn) -> bool:
    sym = getattr(fn, "symbol", None)
    sym_type = getattr(sym, "type", None)
    return getattr(sym_type, "name", None) in IMPORT_SYMBOL_TYPE_NAMES


def is_auto_function_name(name: str) -> bool:
    """True for BN's auto-generated function names -- ``sub_<hex>`` and the
    ``j_sub_<hex>`` thunk variant. Everything else counts as a meaningful name.

    Lives here, not in ``bridge.py``, so `target info`'s named/auto-named summary
    and `function list --named/--unnamed` share ONE predicate: the two numbers
    disagreeing (with neither queryable) is exactly what #653.4 reported.
    """
    core = name[2:] if name.startswith("j_") else name
    if not core.startswith("sub_"):
        return False
    suffix = core[4:]
    return bool(suffix) and all(c in "0123456789abcdefABCDEF" for c in suffix)


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


def _residue_error_disclosure(exc: BaseException) -> dict[str, Any] | None:
    """Structured disclosure for the ``result`` field of an ``ok:false`` response
    when the raised exception carries an unclearable ``has_user_type`` residue.

    A mutation can raise AFTER an applied ``set_prototype`` pinned
    ``BNFunctionHasUserType`` on an AUTO function -- BN exposes no API to clear it,
    so the value revert leaves the view MODIFIED. The exception is annotated with
    ``prototype_user_type_residue = True`` (see ``mutation_engine._mutation``), but
    ``str(exc)`` alone does not disclose that residue. An unattended control loop
    reads the raw response, so surface the residue there -- ``success:false`` /
    ``rolled_back:false`` / ``prototype_user_type_residue:true`` plus an
    explanation -- rather than forcing the caller to infer it from a later
    ``close`` (#630 round 3). Returns None when there is no residue, so a plain
    error response keeps ``result: null``."""
    if not getattr(exc, "prototype_user_type_residue", False):
        return None
    return {
        "success": False,
        "rolled_back": False,
        "prototype_user_type_residue": True,
        "message": (
            "An applied set_prototype pinned the function's has_user_type override "
            "and Binary Ninja exposes no API to clear it, so the view is left "
            "modified even though the operation did not complete. The unsaved "
            "state must be reverted or discarded (e.g. reload the target)."
        ),
    }


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


def _require_nonempty_name(value: Any, *, label: str = "new name",
                            requested: dict[str, Any] | None = None) -> str:
    """Reject an empty/whitespace-only/null name before any view mutation.

    A JSON ``null`` must not ``str()`` into the literal "None" and slip the
    emptiness check, and an empty name leaves a degenerate unnamed
    function/local/field that then "verifies" against itself (#363/#605)."""
    if value is None or not str(value).strip():
        raise OperationFailure("invalid_request", f"{label} must be non-empty", requested=requested)
    return str(value)


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
