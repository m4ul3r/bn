"""The batch mutation engine: apply / verify / op / snapshot / restore.

The ~50-method mutation cluster that used to live on ``BinaryNinjaBridge`` moves
here as module-level free functions, each taking the ``BridgeContext`` seam
(``ctx``) in place of ``self``. The single public entry is ``_mutation`` (the
former ``BinaryNinjaBridge._mutation``); ``BinaryNinjaBridge`` keeps a thin
delegating shim for every name the test suite / op binders reference.

Outbound calls resolve through:
  * ``ctx`` -- resolution / type helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_functions_containing``,
    ``_find_type``, ``_resolve_rename_target``, ``_render_type_layout``,
    ``_current_type_entry``);
  * ``il_format`` -- the state-free HLIL renderer (``_function_text``);
  * ``vars_mod`` -- the state-free variable helpers (``_iter_canonical_variables``,
    ``_variable_identifier``, ``_find_variable_by_storage``,
    ``_find_variable_selector``, ``_local_id``, ``_variable_source_name``);
  * ``_shared`` -- module-free helpers (``_parse_address``,
    ``_normalize_prototype``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``, and
``_shared`` (plus stdlib + binaryninja). It NEVER imports ``bridge`` (the bridge
imports it), and ``read_types`` no longer reaches into it -- the cycle-breakers
``_find_type``/``_render_type_layout`` live on the seam (design spec §3.2).
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

import binaryninja as bn

from . import il_format
from . import vars as vars_mod
from ._shared import (
    USER_FACING_ERRORS,
    OperationFailure,
    _normalize_prototype,
    _parse_address,
    _serialize_error,
    _validate_bool,
)

# Required request fields per mutation op kind, validated before dispatch so a
# missing field is reported as a malformed REQUEST (invalid_request, naming the
# field) rather than letting a raw KeyError surface -- and so a KeyError raised
# by BN internals inside a handler is NOT mislabeled as a missing request field.
# Optional fields (read via op.get(...)) are intentionally omitted.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "rename_symbol": ("identifier", "new_name"),
    "set_comment": ("comment",),
    "delete_comment": (),
    "set_prototype": ("identifier", "prototype"),
    "local_rename": ("function", "variable", "new_name"),
    "local_retype": ("function", "variable", "new_type"),
    "struct_field_set": ("struct_name", "field_type", "offset", "field_name"),
    "struct_field_rename": ("struct_name", "old_name", "new_name"),
    "struct_field_delete": ("struct_name", "field_name"),
    "types_declare": ("declaration",),
}

# Ops that accept one of several alternative locator fields. set_comment /
# delete_comment target EITHER a function (`function`) OR an address (`address`):
# listing `address` as unconditionally required wrongly rejected the documented
# function-only form (#67). Each group requires at least one of its fields.
REQUIRED_ONE_OF: dict[str, tuple[tuple[str, ...], ...]] = {
    "set_comment": (("function", "address"),),
    "delete_comment": (("function", "address"),),
}

# Fields restricted to a fixed value set, mirroring an interactive command's
# argparse `choices=`. The batch path has no argparse layer, so without this an
# out-of-set value reaches a handler and silently mis-resolves -- e.g. an
# unknown rename `kind` skips the function branch in _resolve_rename_target and
# falls through to data-symbol resolution, surfacing a misleading "Symbol not
# found" instead of being rejected the way `bn rename --kind` rejects it. Only a
# PRESENT field is checked; absence is governed by REQUIRED_FIELDS (e.g. `kind`
# is optional, the handler defaults it to "auto"). (#173)
ENUM_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "rename_symbol": {"kind": ("auto", "function", "data")},
}


def _normalize_struct_alias(op: dict[str, Any]) -> dict[str, Any]:
    """Accept ``type_name`` as an alias for ``struct_name`` on struct_field_* ops.

    The output surfaces a struct's identifier under ``type_name`` (affected_types
    entries, ``types show``), so a batch manifest inferred from what the tool
    *shows* naturally uses ``type_name`` and would otherwise fail validation with
    "missing required field 'struct_name'". Normalize to the canonical
    ``struct_name`` in place (an explicit ``struct_name`` always wins) so the
    pre-apply snapshot pass and the apply both resolve the struct. (M12)
    """
    if not isinstance(op, dict):
        return op
    kind = op.get("op")
    if isinstance(kind, str) and kind.startswith("struct_field_"):
        if "struct_name" not in op and "type_name" in op:
            op["struct_name"] = op["type_name"]
    return op

# Op kinds that mutate a function's local variables via create_user_var /
# set_user_type, which BN may PROPAGATE onto aliased siblings. Batches with any
# of these get a full per-function local snapshot so revert paths can undo the
# propagation, not just the targeted variable (see _capture_local_var_snapshots).
_VAR_DRIFT_OPS = {"local_rename", "local_retype", "set_prototype"}


def _guess_type_affected_functions(ctx, bv, type_name: str, limit: int | None = 10):
    matches = []
    needle = type_name.lower()
    for fn in list(bv.functions):
        text = str(fn.type).lower()
        if needle in text:
            matches.append(fn)
            if limit is not None and len(matches) >= limit:
                break
    return matches



def _parse_declaration_source(ctx, bv, declaration: str, *, source_path: str | None = None):
    parse_result = None
    source_error: Exception | None = None
    platform = getattr(bv, "platform", None)
    if platform is not None and hasattr(platform, "parse_types_from_source"):
        kwargs: dict[str, Any] = {}
        if source_path:
            kwargs["filename"] = source_path
            kwargs["include_dirs"] = [str(Path(source_path).expanduser().resolve().parent)]
        try:
            parse_result = platform.parse_types_from_source(declaration, **kwargs)
        except Exception as exc:
            source_error = exc

    if parse_result is None:
        try:
            parse_result = bv.parse_types_from_string(declaration)
        except Exception:
            if source_error is not None:
                raise source_error
            raise

    return {
        "types": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "types", {}).items())],
        "variables": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "variables", {}).items())],
        "functions": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "functions", {}).items())],
    }



def _operation_type_names(ctx, bv, op: dict[str, Any]) -> list[str]:
    kind = op.get("op") or "rename_symbol"
    if kind.startswith("struct_") and op.get("struct_name"):
        # Struct ops resolve the name canonically (case-insensitively) via
        # _find_type and commit under the RESOLVED name. Snapshot under
        # that same name, or a request like "mystruct" against an actual
        # "MyStruct" misses both pre- and post-snapshots and the layout
        # diff silently drops out of affected_types. Fall back to the raw
        # name when resolution fails: this pre-apply pass must not raise --
        # _apply_operation surfaces the precise error.
        raw_name = str(op["struct_name"])
        try:
            resolved_name, type_obj = ctx._find_type(bv, raw_name)
            # Follow a typedef (NamedTypeReference) to the underlying struct tag,
            # the SAME name _struct_builder commits under (#246). Without this the
            # snapshot/diff keys on the alias -- which has no members -- so a
            # field rename through a typedef renders an identical header line and
            # is falsely reported "No effective change detected", and affected_types
            # names the alias while results[].struct_name names the tag. Guard so
            # this pre-apply pass still can't raise (a non-struct typedef returns
            # the raw name; _apply_operation surfaces the precise error).
            resolved_name, _ = _resolve_to_structure(bv, resolved_name, type_obj, raw_name)
        except Exception:
            return [raw_name]
        return [resolved_name]
    if kind == "types_declare":
        # Tolerate a malformed op here (the pre-apply snapshot pass): a
        # missing `declaration` must surface as the precise invalid_request
        # from _apply_operation's field validation, not a raw KeyError that
        # escapes _mutation before the apply loop.
        declaration = op.get("declaration")
        if not declaration:
            return []
        try:
            parsed = _parse_declaration_source(
                ctx, bv, str(declaration), source_path=op.get("source_path"),
            )
        except Exception:
            # A malformed declaration must NOT raise from this pre-apply pass --
            # _op_types_declare surfaces the precise, clean parse error instead
            # (otherwise the SyntaxError escapes _mutation entirely) (#122).
            return []
        return [name for name, _ in parsed["types"]]
    return []



def _functions_for_op(ctx, bv, op: dict[str, Any], *, type_limit: int | None):
    """Resolve the functions a single op touches. ``type_limit`` caps how many
    type-referencing functions come back (None = uncapped, for a true count).
    Shared by _guess_affected_functions (the capped snapshot set) and
    _count_referenced_functions (the uncapped blast-radius total) so both agree
    on what "affected" means."""
    kind = op.get("op") or "rename_symbol"
    functions = []
    try:
        if kind == "rename_symbol" and op.get("kind") != "data":
            functions = [ctx._find_function(bv, op["identifier"])]
        elif kind in {"set_prototype", "local_rename", "local_retype"}:
            ident = op.get("identifier") or op.get("function")
            functions = [ctx._find_function(bv, ident)]
        elif kind in {"set_comment", "delete_comment"}:
            if op.get("function"):
                functions = [ctx._find_function(bv, op["function"])]
            elif op.get("address"):
                functions = ctx._functions_containing(bv, _parse_address(op["address"]))
        elif kind.startswith("struct_") or kind == "types_declare":
            for type_name in _operation_type_names(ctx, bv, op):
                functions.extend(_guess_type_affected_functions(ctx, bv, type_name, limit=type_limit))
    except Exception:
        functions = []
    return [fn for fn in functions if fn is not None]


def _guess_affected_functions(ctx, bv, operations: list[dict[str, Any]]):
    affected = []
    seen = set()
    for op in operations:
        if not isinstance(op, dict):
            continue  # a non-dict op is rejected cleanly in _apply_operation (#48)
        for fn in _functions_for_op(ctx, bv, op, type_limit=10):
            marker = int(fn.start)
            if marker not in seen:
                seen.add(marker)
                affected.append(fn)
    return affected


def _is_type_op(op: Any) -> bool:
    """A type-shape op (type (re)declaration or struct field edit), whose blast
    radius is "functions referencing the type" -- as opposed to a direct op
    (rename/prototype/comment) that targets one specific function."""
    kind = str((op or {}).get("op") or "") if isinstance(op, dict) else ""
    return kind == "types_declare" or kind.startswith("struct_")


def _operation_function_starts(ctx, bv, operations: list[dict[str, Any]]) -> set[int]:
    """Uncapped set of distinct function start addresses in the affected set of
    ``operations`` (type-referencing functions for type ops; the targeted
    function for direct ops). The basis for both the blast-radius count and the
    direct/type origin tag on each affected-function diff."""
    starts: set[int] = set()
    for op in operations:
        if not isinstance(op, dict):
            continue
        for fn in _functions_for_op(ctx, bv, op, type_limit=None):
            starts.add(int(fn.start))
    return starts


def _count_referenced_functions(ctx, bv, operations: list[dict[str, Any]], *, fallback: int) -> int:
    """Uncapped distinct count of functions in the affected set, so a mutation can
    report its true blast radius even though affected_functions is capped at 10
    for snapshotting. A struct referenced by 200 functions previously surfaced as
    "10" with no hint of the real scope. Falls back to the capped length when the
    scan can't run (e.g. a stubbed view in tests)."""
    try:
        return max(len(_operation_function_starts(ctx, bv, operations)), fallback)
    except Exception:
        return fallback



def _affected_type_names(ctx, bv, operations: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for op in operations:
        if not isinstance(op, dict):
            continue  # a non-dict op is rejected cleanly in _apply_operation (#48)
        for type_name in _operation_type_names(ctx, bv, op):
            if type_name not in seen:
                seen.add(type_name)
                names.append(type_name)
    return names



def _capture_type_snapshots(ctx, bv, operations: list[dict[str, Any]]):
    snapshots: dict[str, dict[str, Any]] = {}
    for type_name in _affected_type_names(ctx, bv, operations):
        type_obj = bv.get_type_by_name(type_name)
        if type_obj is None:
            continue
        snapshots[type_name] = {
            "type_name": type_name,
            "decl": str(type_obj),
            "layout": ctx._render_type_layout(type_obj),
        }
    return snapshots



def _diff_type_snapshots(ctx, before: dict[str, Any], after: dict[str, Any]):
    diffs = []
    for type_name in sorted(set(before) | set(after)):
        old = before.get(type_name, {"decl": "", "layout": ""})
        new = after.get(type_name, {"decl": "", "layout": ""})
        layout_diff = "\n".join(
            difflib.unified_diff(
                old["layout"].splitlines(),
                new["layout"].splitlines(),
                fromfile=f"before:{type_name}",
                tofile=f"after:{type_name}",
                lineterm="",
            )
        )
        changed = old["decl"] != new["decl"] or old["layout"] != new["layout"]
        entry = {
            # `name` mirrors `type_name` (the qualified type name): an agent keying
            # off affected_types[].name previously read null because only
            # `type_name` was emitted, making a real change look anonymous (#211).
            "name": type_name,
            "type_name": type_name,
            "before_decl": old["decl"],
            "after_decl": new["decl"],
            "before_layout": old["layout"],
            "after_layout": new["layout"],
            "layout_diff": layout_diff,
            "changed": changed,
        }
        if not changed:
            entry["message"] = "No effective change detected"
        diffs.append(entry)
    return diffs



def _slim_type_result_for_output(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the heavy per-type layout dump from a verified types_declare result.
    The canonical before/after layout already rides on ``affected_types``; echoing
    it again under ``defined_type_layouts`` and ``observed.defined_type_layouts``
    stamped the same layout up to four times per response for zero new signal.
    Only applied on the success path -- on a failure ``affected_types`` is empty
    and the observed layout is the only evidence, so it stays put there."""
    if result.get("op") != "types_declare":
        return result
    slim = dict(result)
    slim.pop("defined_type_layouts", None)
    observed = slim.get("observed")
    if isinstance(observed, dict) and "defined_type_layouts" in observed:
        observed = dict(observed)
        observed.pop("defined_type_layouts", None)
        slim["observed"] = observed
    return slim


def _annotate_operation_results(ctx, results: list[dict[str, Any]], type_diffs: list[dict[str, Any]]):
    type_changes = {item["type_name"]: item for item in type_diffs}
    annotated = []
    for result in results:
        item = dict(result)
        type_name = item.get("struct_name")
        if type_name and type_name in type_changes:
            change = type_changes[type_name]
            item["changed"] = bool(change["changed"])
            if not change["changed"]:
                item["message"] = change["message"]
                if item.get("status") == "verified":
                    item["status"] = "noop"
        defined_types = dict(item.get("defined_types") or {})
        if defined_types:
            changed_types = {name: bool(type_changes.get(name, {}).get("changed")) for name in defined_types}
            item["changed_types"] = changed_types
            # The authoritative change signal is the before/after layout diff
            # (changed_types), not the verify step's decl-string compare -- a
            # redeclaration of an existing NAME renders the same `struct QA`
            # decl before and after, so that compare wrongly reported `noop`
            # on a real layout change (#57). Reclassify from changed_types for
            # the success statuses only (never override a *_failed status).
            if item.get("status") in ("verified", "noop"):
                if any(changed_types.values()):
                    item["status"] = "verified"
                else:
                    item["status"] = "noop"
                    item["message"] = "No effective change detected"
        annotated.append(item)
    return annotated



def _function_comment_state(ctx, bv, fn) -> dict[str, str]:
    """View-level comments at addresses inside this function. The comment ops
    write to BN's GLOBAL comment store (bv.set_comment_at ->
    BNSetGlobalCommentForAddress, surfaced as bv.address_comments), which is a
    DIFFERENT store from Function.comments (the function-local
    BNSetCommentForAddress) -- the two never share data. So the snapshot must
    read the global store, filtered to this function, or a verified comment
    set/delete --preview still shows changed:false / empty diff (#121)."""
    try:
        # bv.address_comments is a property over BN core; guard the access
        # itself (not just a missing attr) so a raising getter degrades to an
        # empty comment signal instead of aborting the whole mutation snapshot.
        all_comments = bv.address_comments
        items = list(dict(all_comments).items()) if all_comments else []
    except Exception:
        return {}
    if not items:
        return {}
    target = int(fn.start)
    state: dict[str, str] = {}
    for addr, text in items:
        if not text:
            continue
        try:
            a = int(addr)
        except Exception:
            continue  # skip one malformed key, don't zero out the whole signal
        try:
            containing = bv.get_functions_containing(a) or []
        except Exception:
            continue
        if any(int(getattr(f, "start", -1)) == target for f in containing):
            state[hex(a)] = str(text)
    return state


def _function_local_state(ctx, fn) -> dict[str, str]:
    """Map of local identifier -> 'name:type' for a function. A local
    rename/retype of a variable not rendered in the HLIL body leaves the body
    text identical, so the diff/changed signal must reflect local name/type
    state too (#121). Reuses the canonical-variable walk the restore paths rely
    on; guarded so functions without resolvable locals just yield {}."""
    state: dict[str, str] = {}
    try:
        for var, _ in vars_mod._iter_canonical_variables(fn):
            identifier = vars_mod._variable_identifier(var)
            if identifier is None:
                continue
            state[str(identifier)] = f"{var.name}:{var.type}"
    except Exception:
        return state
    return state


def _capture_function_snapshots(ctx, bv, functions):
    snapshots = {}
    for fn in functions:
        snapshots[int(fn.start)] = {
            "name": fn.name,
            "address": hex(fn.start),
            "text": il_format._function_text(bv, fn, view="hlil"),
            "comments": _function_comment_state(ctx, bv, fn),
            "locals": _function_local_state(ctx, fn),
        }
    return snapshots


def _format_metadata_change(old: dict[str, Any], new: dict[str, Any], address: int) -> str:
    """A compact diff for changes invisible in the HLIL body text -- a function
    rename, comment set/delete, or local rename/retype -- so a verified
    --preview of them shows a real before/after instead of an empty diff (#121).
    For a name-only change this reduces to the original two-line header."""
    lines = [
        f"--- before:{old.get('name', hex(address))}",
        f"+++ after:{new.get('name', hex(address))}",
    ]
    ob, nb = old.get("comments") or {}, new.get("comments") or {}
    for addr in sorted(set(ob) | set(nb)):
        if ob.get(addr) != nb.get(addr):
            lines.append(f"comment @ {addr}: {ob.get(addr, '')!r} -> {nb.get(addr, '')!r}")
    ol, nl = old.get("locals") or {}, new.get("locals") or {}
    for ident in sorted(set(ol) | set(nl)):
        if ol.get(ident) != nl.get(ident):
            lines.append(f"local {ident}: {ol.get(ident, '')} -> {nl.get(ident, '')}")
    return "\n".join(lines)



def _snippet_for_change(ctx, before_text: str, after_text: str, *, context_lines: int = 3, max_lines: int = 10):
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    line_count = max(len(before_lines), len(after_lines))

    changed_line = None
    for index in range(line_count):
        before_line = before_lines[index] if index < len(before_lines) else None
        after_line = after_lines[index] if index < len(after_lines) else None
        if before_line != after_line:
            changed_line = index
            break

    if changed_line is None:
        return None

    start = max(0, changed_line - context_lines)
    end = min(line_count, start + max_lines)
    return {
        "start_line": start + 1,
        "before_excerpt": "\n".join(before_lines[start:end]),
        "after_excerpt": "\n".join(after_lines[start:end]),
    }



# A previewed mutation's per-function `diff` is a full unified diff of the HLIL
# body. A rename/prototype change ripples through every call site, so for a large
# function the diff can be the whole body (~85 KB / ~1.5k lines) -- enough that a
# single-op preview trips the 10k-token spill threshold and the analyst must read
# an artifact off disk just to confirm one rename landed. The focused
# before/after_excerpt already captures the change for a glance; cap the full
# diff so the common single-op preview stays inline, and point at --out / the
# spill artifact for the complete diff. (M14)
PREVIEW_DIFF_MAX_LINES = 120


def _truncate_preview_diff(diff: str, max_lines: int = PREVIEW_DIFF_MAX_LINES) -> str:
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    hidden = len(lines) - max_lines
    kept = "\n".join(lines[:max_lines])
    return (
        f"{kept}\n... (diff truncated: {hidden} more line(s); "
        f"re-run with --out <path> for the full diff)"
    )


def _diff_snapshots(ctx, before: dict[int, Any], after: dict[int, Any]):
    diffs = []
    snippets_added = 0
    for address in sorted(set(before) | set(after)):
        old = before.get(address, {"text": ""})
        new = after.get(address, {"text": ""})
        text_changed = old.get("text", "") != new.get("text", "")
        name_changed = old.get("name") != new.get("name")
        # Comment set/delete and local rename/retype of non-body variables do
        # not alter the HLIL body text, so a text-only compare reports them as
        # changed:false / empty diff even when verified. Fold them into the
        # change signal and synthesize a diff for them (#121).
        comments_changed = old.get("comments") != new.get("comments")
        locals_changed = old.get("locals") != new.get("locals")
        diff = "\n".join(
            difflib.unified_diff(
                old["text"].splitlines(),
                new["text"].splitlines(),
                fromfile=f"before:{old.get('name', hex(address))}",
                tofile=f"after:{new.get('name', hex(address))}",
                lineterm="",
            )
        )
        if not diff and (name_changed or comments_changed or locals_changed):
            diff = _format_metadata_change(old, new, address)
        diffs.append(
            {
                "address": hex(address),
                "before_name": old.get("name"),
                "after_name": new.get("name"),
                "changed": bool(text_changed or name_changed or comments_changed or locals_changed),
                "diff": _truncate_preview_diff(diff),
            }
        )
        # The focused excerpt only earns its tokens when the full diff was
        # truncated; otherwise the inline `diff` already shows the whole change and
        # the excerpt just duplicates it (the common small-function case, which is
        # where the redundant before/after_excerpt bloat came from). (M14)
        if text_changed and snippets_added < 3 and len(diff.splitlines()) > PREVIEW_DIFF_MAX_LINES:
            snippet = _snippet_for_change(ctx, old.get("text", ""), new.get("text", ""))
            if snippet is not None:
                diffs[-1].update(snippet)
                snippets_added += 1
    return diffs



def _diff_function_start(diff: dict[str, Any]) -> int | None:
    """The int start address of an affected-function diff (its ``address`` is a
    hex string), or None when it can't be parsed."""
    try:
        return int(str(diff.get("address")), 16)
    except (TypeError, ValueError):
        return None


def _operation_requested(ctx, op: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(op, dict):
        return {}
    return {key: value for key, value in op.items() if key != "preview"}



def _operation_failure_result(ctx, op: dict[str, Any], exc: OperationFailure) -> dict[str, Any]:
    op_label = str(op.get("op") or "<missing>") if isinstance(op, dict) else "<non-object>"
    result = {
        "op": op_label,
        "status": exc.status,
        "message": exc.message,
        "requested": exc.requested or _operation_requested(ctx, op),
    }
    if exc.observed:
        result["observed"] = exc.observed
    return result



def _mark_unverified_results(
    ctx, results: list[dict[str, Any]], message: str, status: str = "reverted"
) -> list[dict[str, Any]]:
    """Stamp an honest *status* on ops that ran before a sibling op failed.

    These ops WERE supported and applied -- they were rolled back because a
    later op failed, so the old generic ``unsupported`` was a lie (#118).
    ``reverted`` = cleanly rolled back (not a failure); ``rollback_failed`` =
    the revert itself failed, so the op may still be applied (a real failure
    state, so it stays in ``FAILED_MUTATION_STATUSES``)."""
    annotated = []
    for result in results:
        item = dict(result)
        item["status"] = status
        item["message"] = message
        annotated.append(item)
    return annotated



def _has_failed_results(ctx, results: list[dict[str, Any]]) -> bool:
    return any(item.get("status") in {"unsupported", "verification_failed"} for item in results)



def _member_byte_width(member) -> int:
    """The declared byte width of a struct member's type, or 0 when unknown."""
    try:
        return int(getattr(getattr(member, "type", None), "width", 0) or 0)
    except Exception:
        return 0


def _first_overlapping_member(ctx, type_obj, offset: int, width: int):
    """The first existing member whose byte range intersects the range a new
    field of *width* bytes at *offset* would occupy, else None. Width 0/unknown
    is treated as 1 byte so an exact start-offset collision is always caught.
    Used to enforce struct field set --no-overwrite (#56)."""
    members = getattr(type_obj, "members", None)
    if members is None:
        return None
    new_start = int(offset)
    new_end = new_start + max(int(width or 0), 1)
    for member in list(members):
        m_start = int(getattr(member, "offset", 0))
        try:
            m_width = int(getattr(getattr(member, "type", None), "width", 0) or 0)
        except Exception:
            m_width = 0
        m_end = m_start + max(m_width, 1)
        if new_start < m_end and m_start < new_end:
            return member
    return None



def _find_member(ctx, type_obj, *, offset: int | None = None, name: str | None = None):
    members = getattr(type_obj, "members", None)
    if members is None:
        return None
    for member in list(members):
        member_offset = int(getattr(member, "offset", 0))
        member_name = str(getattr(member, "name", ""))
        if offset is not None and member_offset != int(offset):
            continue
        if name is not None and member_name != name:
            continue
        return member
    return None



def _verify_operation(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    op = result.get("op")
    try:
        if op == "rename_symbol":
            return _verify_rename_symbol(ctx, bv, result)
        if op == "set_comment":
            return _verify_set_comment(ctx, bv, result)
        if op == "delete_comment":
            return _verify_delete_comment(ctx, bv, result)
        if op == "set_prototype":
            return _verify_set_prototype(ctx, bv, result)
        if op == "local_rename":
            return _verify_local_rename(ctx, bv, result)
        if op == "local_retype":
            return _verify_local_retype(ctx, bv, result)
        if op == "struct_field_set":
            return _verify_struct_field_set(ctx, bv, result)
        if op == "struct_field_rename":
            return _verify_struct_field_rename(ctx, bv, result)
        if op == "struct_field_delete":
            return _verify_struct_field_delete(ctx, bv, result)
        if op == "types_declare":
            return _verify_declared_types(ctx, bv, result)
        raise OperationFailure("unsupported", f"Unsupported verification path: {op}", requested=result.get("requested"))
    except OperationFailure as exc:
        item = dict(result)
        item["status"] = exc.status
        item["message"] = exc.message
        if exc.requested:
            item["requested"] = exc.requested
        if exc.observed:
            item["observed"] = exc.observed
        return item
    except Exception as exc:
        item = dict(result)
        item["status"] = "verification_failed"
        item["message"] = f"{type(exc).__name__}: {exc}"
        if item.get("requested") is None:
            item["requested"] = {}
        return item



def _verify_rename_symbol(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    requested_name = str(item["new_name"])
    before_name = item.get("before_name")
    observed_name = None
    if item.get("kind") == "function":
        fn = bv.get_function_at(address)
        if fn is None:
            raise OperationFailure(
                "verification_failed",
                f"Function missing after rename at {item['address']}",
                requested=item.get("requested"),
                observed={"address": item["address"], "name": None},
            )
        observed_name = str(fn.name)
    else:
        symbol = bv.get_symbol_at(address)
        observed_name = str(symbol.name) if symbol is not None else None
    item["observed"] = {"address": item["address"], "name": observed_name}
    if observed_name != requested_name:
        raise OperationFailure(
            "verification_failed",
            f"Live rename verification failed at {item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if before_name == requested_name else "verified"
    return item



def _verify_set_comment(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    expected = str(item["requested"]["comment"])
    observed = bv.get_comment_at(address) or ""
    item["observed"] = {"address": item["address"], "comment": observed}
    if observed != expected:
        raise OperationFailure(
            "verification_failed",
            f"Live comment verification failed at {item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if item.get("before_comment", "") == expected else "verified"
    return item



def _verify_delete_comment(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    observed = bv.get_comment_at(address) or ""
    item["observed"] = {"address": item["address"], "comment": observed}
    if observed:
        raise OperationFailure(
            "verification_failed",
            f"Live comment deletion verification failed at {item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if not item.get("before_comment") else "verified"
    return item



def _prototype_matches_ignoring_param_names(bv, observed_type, expected_str: str) -> bool:
    """True if *observed_type* matches the requested prototype string when
    parameter NAMES are ignored -- only the return type and the per-parameter
    types are compared.

    BN auto-names unnamed parameters on readback (``arg1``, ``arg2``, ...), so a
    valid ``proto set`` whose requested string omits names reads back carrying
    names it never asked for. That is a pure textual mismatch which wrongly trips
    ``verification_failed`` + revert -- the same false-negative class #199 fixed
    for ``__convention``/``__pure``, here for parameter names (#254). Comparing the
    parsed parameter *types* (each rendered canonically by BN, so function
    pointers / arrays survive) is name-insensitive and robust.

    Best-effort: returns False (so the caller falls back to the string compare and
    its existing behavior) when the expected string won't parse or either type
    lacks the BN function-type shape -- e.g. the mocked unit suite, where
    ``fn.type`` is a plain string."""
    try:
        expected_type, _ = bv.parse_type_string(str(expected_str))
    except Exception:  # noqa: BLE001 - unparseable expected -> fall back, never raise
        return False
    try:
        if str(observed_type.return_value) != str(expected_type.return_value):
            return False
        # Varargs are part of the signature but not a named parameter, so don't let
        # the name-insensitive compare silently accept a `...` mismatch. getattr
        # default keeps this a no-op for the mocked unit types that lack the field.
        if bool(getattr(observed_type, "has_variable_arguments", False)) != bool(
            getattr(expected_type, "has_variable_arguments", False)
        ):
            return False
        observed_params = list(observed_type.parameters or [])
        expected_params = list(expected_type.parameters or [])
    except AttributeError:
        return False
    if len(observed_params) != len(expected_params):
        return False
    for o, e in zip(observed_params, expected_params):
        if str(o.type) != str(e.type):
            return False
        # Only ignore the name when the request OMITTED it: BN auto-names an
        # unnamed param arg1/arg2/... on readback, which is the #254 false-negative
        # we tolerate. But when the request EXPLICITLY named a param and it reads
        # back under a different name, the name did NOT land -- that's a partial
        # application, not a verified one, so reject it (#263 review).
        expected_name = str(getattr(e, "name", "") or "")
        if expected_name and str(getattr(o, "name", "") or "") != expected_name:
            return False
    return True


def _verify_set_prototype(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    fn = bv.get_function_at(address)
    if fn is None:
        raise OperationFailure(
            "verification_failed",
            f"Function missing after prototype change at {item['address']}",
            requested=item.get("requested"),
            observed={"address": item["address"], "prototype": None},
        )
    observed = str(fn.type)
    item["observed"] = {"address": item["address"], "prototype": observed}
    expected = item["expected_prototype"]
    if observed != expected:
        # BN analysis may add an implicit calling convention (e.g.
        # __convention("cdecl")) that wasn't present in the parsed
        # expected type.  Normalize both before rejecting.
        if _normalize_prototype(observed) != _normalize_prototype(expected):
            # ...and it auto-names unnamed parameters on readback, so a prototype
            # requested without parameter names reads back with arg1/arg2/...; accept
            # when the return type and all parameter TYPES still match (#254).
            if not _prototype_matches_ignoring_param_names(bv, fn.type, expected):
                raise OperationFailure(
                    "verification_failed",
                    f"Live prototype verification failed at {item['address']}",
                    requested=item.get("requested"),
                    observed=item["observed"],
                )
    item["status"] = "noop" if item.get("before_prototype") == expected else "verified"
    return item



def _verify_local_rename(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    fn = bv.get_function_at(address)
    if fn is None:
        raise OperationFailure(
            "verification_failed",
            f"Function missing after local rename at {item['address']}",
            requested=item.get("requested"),
            observed={"address": item["address"], "variable": None},
        )
    # After analysis, variable objects may be reconstructed.  Try
    # identifier-based lookup first (stable across analysis passes),
    # then fall back to storage.  Check all variables at the storage
    # offset because BN may keep both auto and user-named entries.
    expected_name = item["new_name"]
    storage = int(item["storage"])
    identifier = item.get("identifier")
    var = None
    if identifier is not None:
        for v, _ in vars_mod._iter_canonical_variables(fn):
            if vars_mod._variable_identifier(v) == identifier:
                var = v
                break
    if var is None:
        var, _ = vars_mod._find_variable_by_storage(
            fn, storage, is_parameter=bool(item["is_parameter"]),
        )
    observed_name = str(var.name)
    # If the primary variable still shows the auto name, scan the
    # raw variable lists (bypassing dedup) because BN may keep both
    # auto-named and user-named entries at the same storage offset
    # after analysis. The alternate entry must still be the *same*
    # variable (matching identifier); an unrelated neighbor that
    # happens to carry the requested name must not count as success.
    if observed_name != expected_name:
        is_param = bool(item["is_parameter"])
        collections = [fn.parameter_vars] if is_param else [fn.stack_layout]
        for collection in collections:
            for v in list(collection):
                if (
                    int(getattr(v, "storage", -1)) == storage
                    and str(v.name) == expected_name
                    and (identifier is None or vars_mod._variable_identifier(v) == identifier)
                ):
                    observed_name = expected_name
                    var = v
                    break
            if observed_name == expected_name:
                break
    item["observed"] = {"address": item["address"], "variable": observed_name, "storage": storage}
    if observed_name != expected_name:
        raise OperationFailure(
            "verification_failed",
            f"Live local rename verification failed at {item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if item.get("before_name") == expected_name else "verified"
    return item



def _verify_local_retype(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    fn = bv.get_function_at(address)
    if fn is None:
        raise OperationFailure(
            "verification_failed",
            f"Function missing after local retype at {item['address']}",
            requested=item.get("requested"),
            observed={"address": item["address"], "type": None},
        )
    # Mirror _verify_local_rename: identifier-based lookup first (stable
    # across analysis passes, and it covers register/HLIL-visible locals
    # that _find_variable_by_storage cannot see because that helper scans
    # only parameter_vars/stack_layout). Fall back to storage ONLY when no
    # identifier was recorded -- with an identifier in hand, a same-storage
    # variable with a different identifier is a different variable, and
    # types collide far more often than names (every other local is an
    # int32_t), so a storage-only match could verify the wrong one.
    expected_type = item["expected_type"]
    storage = int(item["storage"])
    identifier = item.get("identifier")
    var = None
    if identifier is not None:
        for v, _ in vars_mod._iter_canonical_variables(fn):
            if vars_mod._variable_identifier(v) == identifier:
                var = v
                break
        # A width-narrowing retype can drop a register-backed local out of
        # hlil.vars (it never lived in parameter_vars/stack_layout), so the
        # canonical scan misses it even though the change landed. The complete
        # func.vars set keeps it; relocate by identifier there. (#156)
        if var is None or str(var.type) != expected_type:
            relocated = vars_mod._find_variable_by_identifier(fn, identifier)
            if relocated is not None:
                var = relocated
    else:
        var, _ = vars_mod._find_variable_by_storage(
            fn,
            storage,
            is_parameter=bool(item["is_parameter"]),
        )
    observed_type = None if var is None else str(var.type)
    # If the primary variable still shows the old type, scan the raw
    # variable lists (bypassing dedup) because BN may keep both auto and
    # user entries at the same storage offset after analysis. The alternate
    # entry must still be the *same* variable (matching identifier); an
    # unrelated neighbor that happens to carry the expected type must not
    # count as success (see _verify_local_rename).
    if var is not None and observed_type != expected_type:
        is_param = bool(item["is_parameter"])
        collections = [fn.parameter_vars] if is_param else [fn.stack_layout]
        for collection in collections:
            for v in list(collection):
                if (
                    int(getattr(v, "storage", -1)) == storage
                    and str(v.type) == expected_type
                    and (identifier is None or vars_mod._variable_identifier(v) == identifier)
                ):
                    observed_type = expected_type
                    var = v
                    break
            if observed_type == expected_type:
                break
    item["observed"] = {
        "address": item["address"],
        "variable": None if var is None else str(var.name),
        "type": observed_type,
    }
    if observed_type != expected_type:
        raise OperationFailure(
            "verification_failed",
            f"Live local retype verification failed at {item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if item.get("before_type") == expected_type else "verified"
    return item



def _verify_struct_field_set(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    type_obj = bv.get_type_by_name(item["struct_name"])
    if type_obj is None:
        raise OperationFailure(
            "verification_failed",
            f"Struct missing after field set: {item['struct_name']}",
            requested=item.get("requested"),
            observed={"type_name": item["struct_name"]},
        )
    member = _find_member(ctx, type_obj, offset=int(item["member_offset"]), name=item["field_name"])
    observed = {
        "type_name": item["struct_name"],
        "offset": item["offset"],
        "field_name": getattr(member, "name", None),
        "field_type": str(getattr(member, "type", "")) if member is not None else None,
    }
    item["observed"] = observed
    if member is None or observed["field_type"] != item["field_type"]:
        raise OperationFailure(
            "verification_failed",
            f"Live struct field verification failed for {item['struct_name']} at {item['offset']}",
            requested=item.get("requested"),
            observed=observed,
        )
    previous = item.get("before_member")
    if previous and previous.get("field_name") == item["field_name"] and previous.get("field_type") == item["field_type"]:
        item["status"] = "noop"
    else:
        item["status"] = "verified"
    return item



def _verify_struct_field_rename(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    type_obj = bv.get_type_by_name(item["struct_name"])
    if type_obj is None:
        raise OperationFailure(
            "verification_failed",
            f"Struct missing after field rename: {item['struct_name']}",
            requested=item.get("requested"),
            observed={"type_name": item["struct_name"]},
        )
    # Verify by OFFSET, not by name: with duplicate member names a global
    # name lookup would see the OTHER same-named member and falsely report
    # failure (the #25 duplicate-name case). The member at the renamed
    # offset must now carry new_name.
    offset = int(item.get("member_offset", -1))
    member = _find_member(ctx, type_obj, offset=offset, name=item["new_name"])
    observed = {
        "type_name": item["struct_name"],
        "offset": hex(offset) if offset >= 0 else None,
        "new_name": getattr(member, "name", None),
    }
    item["observed"] = observed
    if member is None:
        raise OperationFailure(
            "verification_failed",
            f"Live struct field rename verification failed for {item['struct_name']}",
            requested=item.get("requested"),
            observed=observed,
        )
    item["status"] = "noop" if item["old_name"] == item["new_name"] else "verified"
    return item



def _verify_struct_field_delete(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    type_obj = bv.get_type_by_name(item["struct_name"])
    if type_obj is None:
        raise OperationFailure(
            "verification_failed",
            f"Struct missing after field delete: {item['struct_name']}",
            requested=item.get("requested"),
            observed={"type_name": item["struct_name"]},
        )
    # Verify by (offset, name), not by name alone: with duplicate member
    # names a global name lookup would see the OTHER same-named member at a
    # different offset and falsely report the delete failed (#25). The
    # specific member that was removed must be gone from its offset.
    offset = int(item.get("member_offset", -1))
    member = _find_member(ctx, type_obj, offset=offset, name=item["field_name"])
    live_width = None
    try:
        live_width = int(getattr(type_obj, "width", -1))
    except Exception:
        live_width = None
    item["observed"] = {
        "type_name": item["struct_name"],
        "offset": hex(offset) if offset >= 0 else None,
        "field_present": member is not None,
        "width": live_width,
    }
    if member is not None:
        raise OperationFailure(
            "verification_failed",
            f"Live struct field delete verification failed for {item['struct_name']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    # The width must actually have shrunk to the intended value: a delete of the
    # trailing field that left the width stale would otherwise still report
    # `verified` (the original #320 false-positive). Only assert when the op
    # intended a shrink and BN reported a width back.
    expected_width = item.get("expected_width")
    if expected_width is not None and live_width is not None and live_width >= 0 and live_width != int(expected_width):
        raise OperationFailure(
            "verification_failed",
            f"Struct {item['struct_name']} width is {live_width} after deleting the "
            f"trailing field {item['field_name']!r}; expected {int(expected_width)} "
            f"(the end of the new last field). The delete left phantom trailing bytes.",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "verified"
    return item



def _verify_declared_types(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    defined_types = dict(item.get("defined_types") or {})
    defined_type_layouts = dict(item.get("defined_type_layouts") or {})
    if not defined_types:
        item["observed"] = {
            "defined_types": {},
            "parsed_functions": list(item.get("parsed_functions") or []),
            "parsed_variables": list(item.get("parsed_variables") or []),
        }
        item["status"] = "noop"
        item["message"] = "Parsed declarations but no named types were defined."
        return item
    observed_types: dict[str, str | None] = {}
    observed_type_layouts: dict[str, str | None] = {}
    for name, expected in defined_types.items():
        type_obj = bv.get_type_by_name(name)
        observed_types[name] = str(type_obj) if type_obj is not None else None
        observed_type_layouts[name] = ctx._render_type_layout(type_obj) if type_obj is not None else None
        if observed_types[name] != expected:
            if defined_type_layouts.get(name) and observed_type_layouts[name] == defined_type_layouts[name]:
                continue
            raise OperationFailure(
                "verification_failed",
                f"Live type verification failed for {name}",
                requested=item.get("requested"),
                observed={
                    "defined_types": observed_types,
                    "defined_type_layouts": observed_type_layouts,
                },
            )
    item["observed"] = {
        "defined_types": observed_types,
        "defined_type_layouts": observed_type_layouts,
    }
    before = dict(item.get("before_defined_types") or {})
    item["status"] = "noop" if before and all(before.get(name) == expected for name, expected in defined_types.items()) else "verified"
    return item



def _apply_operation(ctx, bv, op: dict[str, Any], restores: list | None = None):
    # A manifest op must be a JSON object; a non-object element (e.g.
    # "ops": ["foo"]) gets a clean invalid_request, not an AttributeError (#48).
    if not isinstance(op, dict):
        raise OperationFailure(
            "invalid_request",
            f"each manifest operation must be a JSON object, got {type(op).__name__}",
        )
    _normalize_struct_alias(op)  # type_name -> struct_name alias for struct ops (M12)
    # A missing `op` key must be its own invalid_request, NOT silently assumed
    # to be a rename_symbol -- a typo'd/absent op kind would otherwise apply
    # the wrong mutation (#48). Internal single-op callers always set `op`.
    kind = op.get("op")
    if not kind:
        raise OperationFailure(
            "invalid_request",
            "operation is missing required field 'op' (the mutation kind, e.g. "
            "'rename_symbol', 'set_comment', 'types_declare')",
            requested=_operation_requested(ctx, op),
        )
    # Validate required request fields up front so a malformed request is
    # reported precisely (invalid_request, naming the field) and a KeyError
    # raised deeper -- e.g. by BN internals inside a handler -- is no longer
    # misreported as a missing request field.
    for field in REQUIRED_FIELDS.get(kind, ()):
        if field not in op:
            raise OperationFailure(
                "invalid_request",
                f"operation {kind!r} is missing required field {field!r}",
                requested=_operation_requested(ctx, op),
            )
    for group in REQUIRED_ONE_OF.get(kind, ()):
        if not any(field in op for field in group):
            raise OperationFailure(
                "invalid_request",
                f"operation {kind!r} requires one of "
                f"{' / '.join(repr(f) for f in group)}",
                requested=_operation_requested(ctx, op),
            )
    for field, allowed in ENUM_FIELDS.get(kind, {}).items():
        if field in op and str(op[field]) not in allowed:
            raise OperationFailure(
                "invalid_request",
                f"operation {kind!r} field {field!r} must be one of "
                f"{' / '.join(repr(v) for v in allowed)}, got {str(op[field])!r}",
                requested=_operation_requested(ctx, op),
            )
    try:
        if kind == "rename_symbol":
            return _op_rename_symbol(ctx, bv, op)
        if kind == "set_comment":
            return _op_set_comment(ctx, bv, op)
        if kind == "delete_comment":
            return _op_delete_comment(ctx, bv, op)
        if kind == "set_prototype":
            return _op_set_prototype(ctx, bv, op, restores)
        if kind == "local_rename":
            return _op_local_rename(ctx, bv, op, restores)
        if kind == "local_retype":
            return _op_local_retype(ctx, bv, op, restores)
        if kind == "struct_field_set":
            return _op_struct_field_set(ctx, bv, op)
        if kind == "struct_field_rename":
            return _op_struct_field_rename(ctx, bv, op)
        if kind == "struct_field_delete":
            return _op_struct_field_delete(ctx, bv, op)
        if kind == "types_declare":
            return _op_types_declare(ctx, bv, op)
        raise OperationFailure("unsupported", f"Unsupported operation: {kind}", requested=_operation_requested(ctx, op))
    except OperationFailure:
        raise
    except Exception as exc:
        # Expected user errors (a mistyped function/struct/variable name -> a
        # RuntimeError "X not found", a bad address -> ValueError) carry an
        # already-actionable message: surface it verbatim with the usual
        # `unsupported` status. A genuinely UNEXPECTED error is a bug, not an
        # unsupported request -- give it a distinct `internal_error` status and
        # keep the class name for debugging. Reuses the bridge-wide taxonomy so
        # mutation errors read the same as every other op (#122).
        user_facing = isinstance(exc, USER_FACING_ERRORS)
        raise OperationFailure(
            "unsupported" if user_facing else "internal_error",
            _serialize_error(exc),
            requested=_operation_requested(ctx, op),
        ) from exc



def _revert_undo_safely(ctx, bv, state) -> bool:
    """Best-effort rollback. Returns False when the revert itself failed,
    meaning partially-applied changes may still be live in the view."""
    try:
        bv.revert_undo_actions(state)
        return True
    except Exception as exc:
        bn.log_error(f"BN Agent Bridge: rollback failed, view may be partially modified: {exc!r}")
        return False



def _find_var_for_restore(ctx, fn, identifier, storage, is_parameter):
    """Re-resolve a local for restore the way verification does: identifier
    first (stable across analysis passes and covers register vars, which
    stack_layout omits), then storage. Returns None if it can't be found."""
    if identifier is not None:
        for var, _ in vars_mod._iter_canonical_variables(fn):
            if vars_mod._variable_identifier(var) == identifier:
                return var
        # Register local dropped from hlil.vars by a width-narrowing retype:
        # the canonical scan misses it, but func.vars still has it. Without
        # this, the restore closure raises and a clean preview falsely warns
        # "the view may be left modified". (#156)
        relocated = vars_mod._find_variable_by_identifier(fn, identifier)
        if relocated is not None:
            return relocated
    try:
        var, _ = vars_mod._find_variable_by_storage(fn, int(storage), is_parameter=is_parameter)
        return var
    except RuntimeError:
        return None



def _capture_local_var_snapshots(ctx, bv, functions) -> dict[int, dict[int, tuple[str, Any]]]:
    """Snapshot every identifiable local's (name, type) per affected function.

    BN's create_user_var/set_user_type PROPAGATE a user name/type onto
    aliased variables -- naming a stack var also names the register that
    copies it (e.g. naming var_8 also renames the aliased r2 to var_8_1).
    Those propagated siblings are NOT the variable an op targeted, so the
    per-op explicit restore (_run_local_restores) never reverts them, and a
    --preview or a rolled-back batch would leave them modified. Snapshotting
    the whole canonical set lets _restore_local_var_drift put every drifted
    local back, not just the targeted one."""
    snapshots: dict[int, dict[int, tuple[str, Any]]] = {}
    for fn in functions:
        entries: dict[int, tuple[str, Any]] = {}
        try:
            for var, _ in vars_mod._iter_canonical_variables(fn):
                identifier = vars_mod._variable_identifier(var)
                if identifier is None:
                    continue  # can't re-resolve it after reanalysis
                entries[identifier] = (str(var.name), var.type)
        except Exception as exc:
            bn.log_error(f"BN Agent Bridge: local var snapshot failed for {hex(int(fn.start))}: {exc!r}")
        snapshots[int(fn.start)] = entries
    return snapshots



def _restore_local_var_drift(ctx, bv, snapshots) -> bool:
    """Put any local whose (name, type) drifted from *snapshots* back, then
    reanalyze. Covers BN's name/type propagation onto aliased siblings that
    the targeted per-op restores miss (see _capture_local_var_snapshots).
    Returns False if any restore raised."""
    if not snapshots:
        return True
    ok = True
    touched = False
    for fn_start, entries in snapshots.items():
        if not entries:
            continue
        rfn = bv.get_function_at(int(fn_start))
        if rfn is None:
            ok = False
            bn.log_error(f"BN Agent Bridge: function {hex(int(fn_start))} missing on var-drift restore")
            continue
        try:
            current = list(vars_mod._iter_canonical_variables(rfn))
        except Exception as exc:
            ok = False
            bn.log_error(f"BN Agent Bridge: var-drift re-enumeration failed for {hex(int(fn_start))}: {exc!r}")
            continue
        for var, _ in current:
            identifier = vars_mod._variable_identifier(var)
            if identifier is None or identifier not in entries:
                continue
            snap_name, snap_type = entries[identifier]
            if str(var.name) == snap_name and str(var.type) == str(snap_type):
                continue
            try:
                rfn.create_user_var(var, snap_type, snap_name)
                touched = True
            except Exception as exc:
                ok = False
                bn.log_error(
                    "BN Agent Bridge: failed to restore a propagated local "
                    f"(identifier {identifier}) in {hex(int(fn_start))}: {exc!r}"
                )
    if touched:
        try:
            bv.update_analysis_and_wait()
        except Exception as exc:
            ok = False
            bn.log_error(f"BN Agent Bridge: reanalysis after var-drift restore failed: {exc!r}")
    return ok



def _run_local_restores(ctx, bv, restores) -> bool:
    """Explicitly undo changes BN's undo buffer does NOT journal -- local var
    rename/retype (Function.create_user_var) and function prototypes
    (Function.set_user_type). For these, revert_undo_actions is a silent no-op,
    so without replaying these restores --preview and rollback-on-failure would
    leave the change permanently applied. Runs in reverse apply order, then
    reanalyzes. Returns False if any restore failed."""
    if not restores:
        return True
    ok = True
    for restore in reversed(restores):
        try:
            restore()
        except Exception as exc:
            ok = False
            bn.log_error(
                "BN Agent Bridge: failed to restore a non-journaled change "
                "(local var create_user_var / prototype set_user_type) on revert: "
                f"{exc!r}"
            )
    # create_user_var only materializes once analysis runs again (same as the
    # forward apply path), so settle the view before returning -- otherwise the
    # restore is queued but the old name/type is still showing.
    try:
        bv.update_analysis_and_wait()
    except Exception as exc:
        ok = False
        bn.log_error(f"BN Agent Bridge: reanalysis after local restore failed: {exc!r}")
    return ok



def _mutation(ctx, selector: str | None, preview: bool, operations: list[dict[str, Any]]):
    if not operations:
        raise ValueError("Batch operation list is empty")

    # Normalize struct field op aliases up front so the pre-apply snapshot pass
    # (_guess_affected_functions / _capture_type_snapshots) resolves the struct
    # under the same key the apply will, not just _apply_operation. (M12)
    for op in operations:
        _normalize_struct_alias(op)

    bv = ctx._resolve_view(selector)
    affected = _guess_affected_functions(ctx, bv, operations)
    # Partition for blast-radius attribution: a type op's reach is "functions
    # referencing the type"; a direct op targets one function. direct_starts are
    # the direct ops' targets, so they can be excluded from a type's blast radius.
    type_ops = [op for op in operations if _is_type_op(op)]
    direct_starts = _operation_function_starts(
        ctx, bv, [op for op in operations if isinstance(op, dict) and not _is_type_op(op)]
    )
    before = _capture_function_snapshots(ctx, bv, affected)
    type_before = _capture_type_snapshots(ctx, bv, operations)
    # Snapshot affected-function locals up front when the batch mutates any
    # (rename/retype/prototype), so the revert paths can undo BN's name/type
    # propagation onto aliased siblings (see _capture_local_var_snapshots).
    var_before = (
        _capture_local_var_snapshots(ctx, bv, affected)
        if any(
            isinstance(op, dict)
            and (op.get("op") or "rename_symbol") in _VAR_DRIFT_OPS
            for op in operations
        )
        else {}
    )
    state = bv.begin_undo_actions()
    results = []
    # Explicit restores for local var ops, which BN's undo buffer can't
    # revert (see _run_local_restores). Replayed on every revert path.
    restores: list = []
    try:
        for op in operations:
            results.append(_apply_operation(ctx, bv, op, restores))
    except OperationFailure as exc:
        # Run BOTH revert steps unconditionally: an `and` would short-circuit
        # past the explicit restores when the undo revert fails, leaving
        # non-journaled local/prototype changes applied.
        undo_ok = _revert_undo_safely(ctx, bv, state)
        restore_ok = _run_local_restores(ctx, bv, restores)
        drift_ok = _restore_local_var_drift(ctx, bv, var_before)
        reverted = undo_ok and restore_ok and drift_ok
        if reverted:
            message = "Rolled back before post-state verification because an operation failed to apply."
            result_note = "Rolled back before post-state verification."
            result_status = "reverted"
        else:
            message = (
                "An operation failed to apply AND the rollback itself failed; "
                "the view may be left partially modified."
            )
            result_note = "Rollback failed; this operation may still be applied."
            result_status = "rollback_failed"
        return {
            "preview": preview,
            "success": False,
            "committed": False,
            "rolled_back": reverted,
            "message": message,
            "results": _mark_unverified_results(ctx, results, result_note, status=result_status)
            + [_operation_failure_result(ctx, operations[len(results)], exc)],
            "affected_functions": [],
            "affected_types": [],
            "affected_summary": {
                "referenced": _count_referenced_functions(ctx, bv, type_ops, fallback=0),
                "reflowed": 0,
            },
        }

    try:
        bv.update_analysis_and_wait()
        after = _capture_function_snapshots(ctx, bv, affected)
        type_after = _capture_type_snapshots(ctx, bv, operations)
        diffs = _diff_snapshots(ctx, before, after)
        type_diffs = _diff_type_snapshots(ctx, type_before, type_after)
        verified_results = [_verify_operation(ctx, bv, result) for result in results]
        annotated_results = _annotate_operation_results(ctx, verified_results, type_diffs)
        failed = _has_failed_results(ctx, annotated_results)
        # Origin tag: a direct op (rename/prototype/comment) targets a specific
        # function; mark its affected-function diffs `direct` so a mixed batch
        # never attributes that function to a type's "referenced by" set.
        for d in diffs:
            d["direct"] = _diff_function_start(d) in direct_starts
        # Blast radius is a TYPE concept, so scope it to the type ops only:
        # reflowed = type-referencing functions whose body text actually moved;
        # referenced = uncapped distinct functions referencing the type(s) (so a
        # struct used by 200 functions reads as 200, not the 10-fn snapshot cap).
        # A batch with no type op reports referenced=0 (no blast line).
        reflowed = sum(1 for d in diffs if d.get("changed") and not d.get("direct"))
        referenced = (
            _count_referenced_functions(ctx, bv, type_ops,
                                        fallback=sum(1 for d in diffs if not d.get("direct")))
            if type_ops else 0
        )
        # On success the canonical type layout lives in affected_types; slim the
        # redundant copies out of results. On failure keep them as the evidence.
        output_results = annotated_results if failed else [
            _slim_type_result_for_output(item) for item in annotated_results
        ]
        restored = True
        if preview or failed:
            bv.revert_undo_actions(state)
            # Targeted/prototype restores first, then mop up BN's propagation
            # onto aliased siblings; both must succeed for a clean revert.
            restored = _run_local_restores(ctx, bv, restores)
            restored = _restore_local_var_drift(ctx, bv, var_before) and restored
        else:
            bv.commit_undo_actions(state)
        message = None
        if preview:
            message = "Preview verified and reverted." if restored else (
                "Preview verified, but reverting a non-journaled change "
                "(local variable or prototype) failed; the view may be left modified."
            )
        elif failed:
            message = "Rolled back because live-session verification failed." if restored else (
                "Live-session verification failed AND reverting a non-journaled change "
                "(local variable or prototype) failed; the view may be left modified."
            )
        else:
            message = "Applied and verified in the live Binary Ninja session."
        # A preview whose non-journaled restore failed left the view
        # modified -- that is not a success, even if every operation
        # verified. Automation keys off `success`; `restored` is only
        # False on the preview/failed paths.
        result = {
            "preview": preview,
            "success": (not failed) and restored,
            "committed": bool((not preview) and (not failed)),
            "message": message,
            "results": output_results,
            "affected_functions": diffs,
            "affected_types": type_diffs,
            "affected_summary": {"referenced": referenced, "reflowed": reflowed},
        }
        if preview or failed:
            result["rolled_back"] = restored
        return result
    except Exception as exc:
        undo_ok = _revert_undo_safely(ctx, bv, state)
        restore_ok = _run_local_restores(ctx, bv, restores)
        drift_ok = _restore_local_var_drift(ctx, bv, var_before)
        if not (undo_ok and restore_ok and drift_ok):
            raise RuntimeError(
                f"{exc} (additionally, rollback failed; the view may be left partially modified)"
            ) from exc
        raise



def _op_rename_symbol(ctx, bv, op: dict[str, Any]):
    kind = str(op.get("kind", "auto"))
    identifier = op["identifier"]
    new_name = str(op["new_name"])
    target = ctx._resolve_rename_target(bv, identifier, kind)
    requested = _operation_requested(ctx, op)
    if target["kind"] == "function":
        fn = bv.get_function_at(target["address"])
        if fn is None:
            raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)
        if target["before_name"] != new_name:
            fn.name = new_name
        return {
            "op": "rename_symbol",
            "kind": "function",
            "address": hex(target["address"]),
            "before_name": target["before_name"],
            "new_name": new_name,
            "requested": requested,
        }
    address = int(target["address"])
    if target["before_name"] != new_name:
        bv.define_user_symbol(bn.Symbol(bn.SymbolType.DataSymbol, address, new_name))
    return {
        "op": "rename_symbol",
        "kind": "data",
        "address": hex(address),
        "before_name": target["before_name"],
        "new_name": new_name,
        "requested": requested,
    }



def _op_set_comment(ctx, bv, op: dict[str, Any]):
    comment = str(op["comment"])
    if op.get("function") and op.get("address"):
        raise OperationFailure(
            "invalid_request",
            "Pass function OR address, not both: they target different locations.",
            requested=_operation_requested(ctx, op),
        )
    if op.get("function"):
        fn = ctx._find_function(bv, op["function"])
        before_comment = bv.get_comment_at(fn.start) or ""
        if before_comment != comment:
            bv.set_comment_at(fn.start, comment)
        return {
            "op": "set_comment",
            "address": hex(fn.start),
            "function": fn.name,
            "before_comment": before_comment,
            "requested": _operation_requested(ctx, op),
        }
    address = _parse_address(op["address"])
    before_comment = bv.get_comment_at(address) or ""
    if before_comment != comment:
        bv.set_comment_at(address, comment)
    return {
        "op": "set_comment",
        "address": hex(address),
        "before_comment": before_comment,
        "requested": _operation_requested(ctx, op),
    }



def _op_delete_comment(ctx, bv, op: dict[str, Any]):
    if op.get("function") and op.get("address"):
        raise OperationFailure(
            "invalid_request",
            "Pass function OR address, not both: they target different locations.",
            requested=_operation_requested(ctx, op),
        )
    if op.get("function"):
        fn = ctx._find_function(bv, op["function"])
        before_comment = bv.get_comment_at(fn.start) or ""
        if before_comment:
            bv.set_comment_at(fn.start, None)
        return {
            "op": "delete_comment",
            "address": hex(fn.start),
            "function": fn.name,
            "before_comment": before_comment,
            "requested": _operation_requested(ctx, op),
        }
    address = _parse_address(op["address"])
    before_comment = bv.get_comment_at(address) or ""
    if before_comment:
        bv.set_comment_at(address, None)
    return {
        "op": "delete_comment",
        "address": hex(address),
        "before_comment": before_comment,
        "requested": _operation_requested(ctx, op),
    }



def _split_qualified_name(name: str) -> list[str]:
    """Split a ``::``-qualified C++ name into its components, splitting ONLY at
    bracket depth 0 so template arguments are not torn apart -- e.g.
    ``__alloc_traits<std::allocator<char> >::pointer`` ->
    ``['__alloc_traits<std::allocator<char> >', 'pointer']``, not three pieces
    (#200)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(name):
        ch = name[i]
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif depth <= 0 and name[i:i + 2] == "::":
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _lookup_named_type(bv, base: str):
    """Resolve a (possibly ``::``-qualified) type name to a BN Type, or None.

    BN keys namespaced types by a multi-component ``QualifiedName``, NOT the raw
    ``::``-joined string -- ``get_type_by_name("ns::Foo")`` coerces to a SINGLE
    component and misses a type registered as ``['ns','Foo']``. So try the raw
    string first (flat / single-component names), then a depth-aware
    ``QualifiedName`` split for the namespaced case (#200, verified live on real
    recovered C++ types)."""
    try:
        named = bv.get_type_by_name(base)
    except Exception:
        named = None
    if named is not None:
        return named
    if "::" not in base:
        return None
    try:
        return bv.get_type_by_name(bn.QualifiedName(_split_qualified_name(base)))
    except Exception:
        return None


def _resolve_named_type_string(bv, text: str):
    """Build a Type for a string that names an existing user/named type,
    optionally with trailing ``*`` pointer levels, or None when the base name is
    not a known type.

    BN's C type-string parser rejects ``::``-qualified names ("use of undeclared
    identifier 'ns'"), so a recovered C++ class type can't be applied by name
    even when it's defined. This rebuilds the type directly from
    ``get_type_by_name`` + a named-type reference, so ``ns::demo::Foo`` /
    ``ns::demo::Foo*`` resolve without a flat-name alias workaround (#200)."""
    base = str(text).strip()
    depth = 0
    while base.endswith("*"):
        base = base[:-1].rstrip()
        depth += 1
    if not base:
        return None
    named = _lookup_named_type(bv, base)
    if named is None:
        return None
    try:
        # Pointer directly to the named type: BN preserves the registered name
        # (rendering `struct ns::demo::Foo*`), which is what the live readback
        # reports -- so verification matches. Building a fresh typedef reference
        # instead renders without the `struct` tag and fails verification (#200).
        resolved = named
        for _ in range(depth):
            resolved = bn.Type.pointer(bv.arch, resolved)
    except Exception:
        return None
    return resolved


def _parse_type_or_hint(ctx, bv, op: dict[str, Any], type_text: Any, *, label: str):
    """``bv.parse_type_string`` with a friendly, actionable error.

    A prototype/type that references a struct or type not defined yet fails to
    parse. Surface a clear ``invalid_request`` with a next step instead of the
    generic ``unsupported: <ExceptionClass>: ...`` the catch-all in
    ``_apply_operation`` would otherwise emit -- which leaks the Python
    exception class name and gives no way forward (#122)."""
    try:
        return bv.parse_type_string(str(type_text))
    except OperationFailure:
        raise
    except Exception as exc:
        # BN's C type-string parser rejects a ::-qualified user type even when it
        # is defined and resolvable (every recovered C++ class type is
        # namespaced). Before erroring, try to build the type from an existing
        # named type, optionally wrapped in trailing '*' pointer levels (#200).
        resolved = _resolve_named_type_string(bv, str(type_text))
        if resolved is not None:
            return resolved, None
        # Collapse BN's multi-line parser output ("error: ...\n1 error
        # generated.") into a single clean clause.
        detail = " ".join(str(exc).split())
        raise OperationFailure(
            "invalid_request",
            f"could not parse {label} {str(type_text)!r}"
            + (f": {detail}" if detail else "")
            + ". If it references a struct or type that isn't defined yet, "
            "declare it first (e.g. with `bn types declare`), then retry.",
            requested=_operation_requested(ctx, op),
        ) from exc


def _op_set_prototype(ctx, bv, op: dict[str, Any], restores: list | None = None):
    fn = ctx._find_function(bv, op["identifier"])
    expected_type, _ = _parse_type_or_hint(ctx, bv, op, op["prototype"], label="prototype")
    before_prototype = str(fn.type)
    before_type_obj = fn.type
    expected_prototype = str(expected_type)
    if before_prototype != expected_prototype:
        # Function.set_user_type is NOT journaled by BN's undo buffer (same
        # class as create_user_var for locals), so revert_undo_actions is a
        # silent no-op for prototypes -- without an explicit restore, --preview
        # and rollback-on-failure would leave the previewed prototype committed
        # to the view (#51). Register the restore before mutating.
        _register_prototype_restore(ctx,
            bv, restores, fn=fn,
            before_prototype=before_prototype, before_type_obj=before_type_obj,
        )
        try:
            fn.set_user_type(expected_prototype)
        except TypeError:
            fn.set_user_type(expected_type)
    return {
        "op": "set_prototype",
        "function": fn.name,
        "address": hex(fn.start),
        "before_prototype": before_prototype,
        "expected_prototype": expected_prototype,
        "requested": _operation_requested(ctx, op),
    }



def _register_prototype_restore(ctx, bv, restores, *, fn, before_prototype, before_type_obj):
    """Capture how to put a function prototype back on revert. Mirrors
    :meth:`_register_local_restore`: ``set_user_type`` is not journaled by BN's
    undo buffer, so the preview/rollback paths replay this explicitly. Re-resolves
    the function fresh at restore time by its start address."""
    if restores is None:
        return
    fn_start = int(fn.start)

    def _restore():
        rfn = bv.get_function_at(fn_start)
        if rfn is None:
            raise RuntimeError(f"function {hex(fn_start)} missing on prototype restore")
        # Restore via the .type property setter with the captured Type object:
        # it puts back the EXACT prior prototype, whereas set_user_type would
        # re-pin the calling convention explicitly (turning an implicit-default
        # auto prototype into `... __convention("cdecl") ...`), which is not a
        # true no-op. Fall back to the type string only if the object path fails.
        try:
            rfn.type = before_type_obj
        except Exception:
            rfn.set_user_type(before_prototype)

    restores.append(_restore)



def _register_local_restore(ctx, bv, restores, *, fn, var, name, type_obj, is_parameter):
    """Capture how to put a local back to (name, type_obj) on revert.
    Re-resolves the variable fresh at restore time by identifier/storage,
    because the captured Variable's index can shift across reanalysis."""
    if restores is None:
        return
    fn_start = int(fn.start)
    identifier = vars_mod._variable_identifier(var)
    storage = int(var.storage)

    def _restore():
        rfn = bv.get_function_at(fn_start)
        if rfn is None:
            raise RuntimeError(f"function {hex(fn_start)} missing on restore")
        rvar = _find_var_for_restore(ctx, rfn, identifier, storage, is_parameter)
        if rvar is None:
            raise RuntimeError(f"local at storage {storage} missing on restore in {hex(fn_start)}")
        rfn.create_user_var(rvar, type_obj, name)

    restores.append(_restore)



def _op_local_rename(ctx, bv, op: dict[str, Any], restores: list | None = None):
    fn = ctx._find_function(bv, op["function"])
    var, is_parameter = vars_mod._find_variable_selector(fn, str(op["variable"]))
    new_name = str(op["new_name"])
    # Variable.name is a live property backed by the core: snapshot it
    # before mutating, or before_name reads back the new name and
    # verification misclassifies a real change as a noop.
    before_name = str(var.name)
    if before_name != new_name:
        # create_user_var isn't journaled by BN's undo buffer, so register
        # an explicit restore for the preview/rollback paths.
        _register_local_restore(ctx,
            bv, restores, fn=fn, var=var, name=before_name, type_obj=var.type, is_parameter=is_parameter
        )
        fn.create_user_var(var, var.type, new_name)
    return {
        "op": "local_rename",
        "function": fn.name,
        "address": hex(fn.start),
        "variable": str(op["variable"]),
        "local_id": vars_mod._local_id(fn, var, is_parameter=is_parameter),
        "storage": int(var.storage),
        "identifier": vars_mod._variable_identifier(var),
        "source_type": vars_mod._variable_source_name(var),
        "is_parameter": is_parameter,
        "before_name": before_name,
        "new_name": new_name,
        "requested": _operation_requested(ctx, op),
    }



def _op_local_retype(ctx, bv, op: dict[str, Any], restores: list | None = None):
    fn = ctx._find_function(bv, op["function"])
    var, is_parameter = vars_mod._find_variable_selector(fn, str(op["variable"]))
    expected_type, _ = _parse_type_or_hint(ctx, bv, op, op["new_type"], label="type")
    # Variable.type is a live property backed by the core: snapshot it
    # before mutating (see _op_local_rename).
    before_type_obj = var.type
    before_type = str(before_type_obj)
    if before_type != str(expected_type):
        # create_user_var isn't journaled by BN's undo buffer, so register
        # an explicit restore for the preview/rollback paths.
        _register_local_restore(ctx,
            bv, restores, fn=fn, var=var, name=str(var.name), type_obj=before_type_obj, is_parameter=is_parameter
        )
        fn.create_user_var(var, expected_type, var.name)
    return {
        "op": "local_retype",
        "function": fn.name,
        "address": hex(fn.start),
        "variable": str(op["variable"]),
        "local_id": vars_mod._local_id(fn, var, is_parameter=is_parameter),
        "storage": int(var.storage),
        "identifier": vars_mod._variable_identifier(var),
        "source_type": vars_mod._variable_source_name(var),
        "is_parameter": is_parameter,
        "before_type": before_type,
        "expected_type": str(expected_type),
        "requested": _operation_requested(ctx, op),
    }



def _type_class_name(t: Any) -> str:
    """String name of a BN ``Type.type_class`` (enum on real BN, plain string on
    the unit fakes), so callers can duck-type without importing ``TypeClass``."""
    tc = getattr(t, "type_class", None)
    return str(getattr(tc, "name", None) or (str(tc) if tc is not None else ""))


def _resolve_to_structure(bv, resolved_name: str, type_obj, requested: str):
    """Follow a typedef (a ``NamedTypeReference``) to the underlying registered
    struct so field edits land on the body, not the alias.

    ``typedef struct { ... } Foo;`` is the idiomatic C struct declaration, so the
    alias path is the common case, not an edge: BN registers the body under a tag
    (`Foo` -> `InnerRec`, or an auto-named `_Foo` for an anonymous struct) and
    makes ``Foo`` a ``NamedTypeReference`` to it. ``mutable_copy()`` on that
    reference returns a ``NamedTypeReferenceBuilder`` with no
    ``add_member_at_offset`` -- so we must resolve to the tag first and edit/commit
    THAT; the typedef keeps pointing at it. Raises a clean ``invalid_request`` when
    the name resolves to a non-aggregate (e.g. ``typedef uint32_t Foo;``) instead
    of crashing in ``add_member_at_offset`` (#246)."""
    seen: set[str] = set()
    while "NamedTypeReference" in _type_class_name(type_obj):
        if resolved_name in seen:        # defend against a pathological self-cycle
            break
        seen.add(resolved_name)
        try:
            target = type_obj.target(bv)
        except Exception:
            target = None
        if target is None or target is type_obj:
            break
        reg_name = getattr(getattr(target, "registered_name", None), "name", None)
        if reg_name is not None:
            resolved_name = str(reg_name)
        type_obj = target
    if "Structure" not in _type_class_name(type_obj):
        raise OperationFailure(
            "invalid_request",
            f"{requested!r} resolves to a non-struct type "
            f"({_type_class_name(type_obj) or 'unknown'}); struct field ops require "
            f"a struct or union. Declare it as a struct, or target the right type.",
        )
    return resolved_name, type_obj


def _struct_builder(ctx, bv, struct_name: str):
    try:
        resolved_name, type_obj = ctx._find_type(bv, struct_name)
    except RuntimeError:
        raise RuntimeError(f"Struct not found: {struct_name}")
    resolved_name, type_obj = _resolve_to_structure(bv, resolved_name, type_obj, struct_name)
    return resolved_name, type_obj.mutable_copy()



def _commit_struct_builder(ctx, bv, struct_name: str, builder):
    bv.define_user_type(struct_name, builder)



def _op_struct_field_set(ctx, bv, op: dict[str, Any]):
    struct_name = str(op["struct_name"])
    resolved_name, builder = _struct_builder(ctx, bv, struct_name)
    field_type, _ = _parse_type_or_hint(ctx, bv, op, op["field_type"], label="field type")
    offset = _parse_address(op["offset"])
    overwrite = _validate_bool(op.get("overwrite_existing"), label="overwrite_existing", default=True)
    before_type = bv.get_type_by_name(resolved_name)
    before_member = None
    if before_type is not None:
        member = _find_member(ctx, before_type, offset=offset)
        if member is not None:
            before_member = {
                "field_name": str(getattr(member, "name", "")),
                "field_type": str(getattr(member, "type", "")),
                "offset": hex(int(getattr(member, "offset", offset))),
            }
    # --no-overwrite must REFUSE when the new field would clobber existing
    # data, not add an overlapping member: BN's add_member_at_offset(...,
    # overwrite=False) silently appends an overlapping member (worse than the
    # overwrite path). Guard the whole BYTE RANGE the new field occupies, not
    # just an exact start-offset collision -- an offset that lands *inside* a
    # wider member (e.g. 0x4 within an int64_t at 0x0) overlaps just as much
    # (#56).
    if not overwrite and before_type is not None:
        new_width = 0
        try:
            new_width = int(field_type.width)
        except Exception:
            new_width = 0
        clash = _first_overlapping_member(ctx, before_type, offset, new_width)
        if clash is not None:
            raise OperationFailure(
                "invalid_request",
                f"a {new_width or 1}-byte field at offset {hex(offset)} in struct "
                f"{resolved_name} would overlap existing member "
                f"{str(getattr(clash, 'name', ''))!r} "
                f"({str(getattr(clash, 'type', ''))}) at "
                f"{hex(int(getattr(clash, 'offset', offset)))}; --no-overwrite refuses "
                f"to clobber it. Re-run without --no-overwrite to replace it, or choose "
                f"a free range.",
                requested=_operation_requested(ctx, op),
            )
    builder.add_member_at_offset(str(op["field_name"]), field_type, offset, overwrite)
    try:
        builder.width = max(int(builder.width), int(offset) + int(field_type.width))
    except Exception:
        pass
    _commit_struct_builder(ctx, bv, resolved_name, builder)
    return {
        "op": "struct_field_set",
        "struct_name": resolved_name,
        "offset": hex(offset),
        "field_name": str(op["field_name"]),
        "field_type": str(field_type),
        "member_offset": int(offset),
        "before_member": before_member,
        "requested": _operation_requested(ctx, op),
    }



def _resolve_struct_field(ctx, builder, resolved_name: str, locator: Any):
    """Resolve a struct-field locator (a field NAME, or an OFFSET like 0x8 /
    8) to its ``(index, member)`` in *builder* from a SINGLE scan.

    Returning the index+member directly -- instead of a name that the caller
    re-looks-up -- is what makes ``rename``/``delete`` hit the right field
    when two members share a name at different offsets: a name round-trip
    went through ``index_by_name`` (first-match), silently mutating the
    wrong field. The offset is parsed with ``_parse_address`` so the grammar
    is identical to ``struct field set`` (a zero-padded ``0008`` resolves the
    same in all three). Raises invalid_request when nothing matches."""
    text = str(locator)
    members = list(getattr(builder, "members", []) or [])
    for index, member in enumerate(members):
        if str(getattr(member, "name", "")) == text:
            return index, member
    try:
        offset = _parse_address(text)
    except ValueError:
        offset = None
    if offset is not None:
        for index, member in enumerate(members):
            if int(getattr(member, "offset", -1)) == offset:
                return index, member
    raise OperationFailure(
        "invalid_request",
        f"no field named or at offset {text!r} in struct {resolved_name}",
    )



def _op_struct_field_rename(ctx, bv, op: dict[str, Any]):
    struct_name = str(op["struct_name"])
    resolved_name, builder = _struct_builder(ctx, bv, struct_name)
    index, member = _resolve_struct_field(ctx, builder, resolved_name, op["old_name"])
    old_name = str(getattr(member, "name", ""))
    member_offset = int(getattr(member, "offset", -1))
    builder.replace(index, member.type, str(op["new_name"]), True)
    _commit_struct_builder(ctx, bv, resolved_name, builder)
    return {
        "op": "struct_field_rename",
        "struct_name": resolved_name,
        "old_name": old_name,
        "new_name": str(op["new_name"]),
        "member_offset": member_offset,
        "requested": _operation_requested(ctx, op),
    }



def _op_struct_field_delete(ctx, bv, op: dict[str, Any]):
    struct_name = str(op["struct_name"])
    resolved_name, builder = _struct_builder(ctx, bv, struct_name)
    index, member = _resolve_struct_field(ctx, builder, resolved_name, op["field_name"])
    field_name = str(getattr(member, "name", ""))
    member_offset = int(getattr(member, "offset", -1))
    old_width = int(getattr(builder, "width", 0) or 0)
    deleted_end = member_offset + _member_byte_width(member)
    builder.remove(index)
    # BN's StructureBuilder.remove() leaves the struct width untouched, so
    # deleting the field that reached the struct end would keep phantom trailing
    # bytes (a stale width that still verifies) (#320). Shrink the width to the
    # end of the new last field ONLY when the deleted field actually defined the
    # struct's end -- a field that stops short of the width (e.g. an early field
    # in a partially-recovered struct sized larger than its mapped members) must
    # leave the intentional width alone.
    expected_width = None
    if old_width > 0 and deleted_end >= old_width:
        new_end = 0
        for remaining in getattr(builder, "members", []) or []:
            new_end = max(new_end, int(getattr(remaining, "offset", 0)) + _member_byte_width(remaining))
        if new_end < old_width:
            # The intended post-delete width. Recorded so the verifier can
            # confirm the shrink actually LANDED -- if the assignment below
            # silently fails (the except: pass) or BN ignores it, a width-blind
            # verifier would still report `verified` on a stale width, the very
            # false-positive #320 is about.
            expected_width = new_end
            try:
                builder.width = new_end
            except Exception:
                pass
    _commit_struct_builder(ctx, bv, resolved_name, builder)
    return {
        "op": "struct_field_delete",
        "struct_name": resolved_name,
        "field_name": field_name,
        "member_offset": member_offset,
        "expected_width": expected_width,
        "requested": _operation_requested(ctx, op),
    }



# A C bitfield member: an identifier, a colon, an integer width (decimal OR hex),
# then a member terminator (`;`/`,`/`}`). BN's headless C parser silently DROPS
# the `:N` width and lays each bitfield out as a full-width integer at the BYTE
# offset of its bit position, producing overlapping/oversized members -- and
# reports it `verified` (#322). We reject such declarations instead. Anchoring on
# an integer literal after the colon avoids C++ inheritance (`: public Base`),
# and the literal must be immediately followed by a terminator so a stray
# `width:32` in a comment (stripped first) or other text doesn't match.
_BITFIELD_MEMBER_RE = re.compile(r"[A-Za-z_]\w*\s*:\s*(?:0[xX][0-9A-Fa-f]+|\d+)\s*[;,}]")
# Match a block OR line comment in a SINGLE left-to-right pass: a `/*` inside a
# `//` line is consumed as part of the line comment (and vice versa), so
# stripping can't span across a comment of the other kind and swallow a real
# bitfield between them.
_C_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
# Member/segment separators: scanning back from a candidate colon to the nearest
# of these bounds the expression it belongs to, so a `?` in that window marks a
# ternary's `:` rather than a bitfield width.
_SEGMENT_BOUNDS = "{};,"


def _strip_c_comments(source: str) -> str:
    """Remove /* block */ and // line comments so a colon-number inside a comment
    isn't mistaken for a bitfield."""
    return _C_COMMENT_RE.sub(" ", source)


def _declaration_has_bitfield(declaration: str) -> bool:
    text = _strip_c_comments(declaration)
    for match in _BITFIELD_MEMBER_RE.finditer(text):
        colon = text.find(":", match.start(), match.end())
        if colon < 0:
            continue
        # A `?` between the enclosing segment boundary and the colon means this
        # colon is the `:` of a ternary expression (e.g. an enum value
        # `B = cond ? A : 3` or a default initializer), which BN parses fine --
        # not a bitfield. Don't reject those.
        seg_start = max((text.rfind(ch, 0, colon) for ch in _SEGMENT_BOUNDS), default=-1)
        if "?" in text[seg_start + 1:colon]:
            continue
        return True
    return False


def _bitfield_member_width(member) -> int:
    """The declared byte width of a struct member's type, or 0 when unknown."""
    try:
        return int(getattr(getattr(member, "type", None), "width", 0) or 0)
    except Exception:
        return 0


def _struct_overflow_member(type_obj):
    """The first member of a structure whose declared byte range extends past the
    struct's own width (a corrupt layout BN should never produce for a valid
    declaration), else None. Only the OVERFLOW direction is checked: unions are
    parsed as a struct variant with members overlapping at offset 0, which is
    legitimate and must not be flagged, but no member of a valid struct or union
    ever extends beyond the type width."""
    if "Structure" not in _type_class_name(type_obj):
        return None
    try:
        width = int(getattr(type_obj, "width", 0) or 0)
    except Exception:
        return None
    if width <= 0:  # opaque / forward-declared: nothing to check
        return None
    for member in getattr(type_obj, "members", None) or []:
        end = int(getattr(member, "offset", 0)) + _bitfield_member_width(member)
        if end > width:
            return member
    return None


def _op_types_declare(ctx, bv, op: dict[str, Any]):
    declaration = str(op["declaration"])
    # Reject C bitfield syntax up front: BN's parser corrupts it (see
    # _BITFIELD_MEMBER_RE) and would otherwise return a malformed, overlapping
    # layout reported as `verified` (#322). Nothing is applied.
    if _declaration_has_bitfield(declaration):
        raise OperationFailure(
            "invalid_request",
            "C bitfield syntax (`name : N`) is not supported: the type parser "
            "silently drops the bit width and lays each bitfield out as a "
            "full-width integer at the byte offset of its bit position, producing "
            "an overlapping, oversized layout. Declare the storage unit as a plain "
            "integer (e.g. `uint32_t flags;`) and extract the bits in your "
            "analysis, or split it into separate byte/half-word fields.",
            requested=_operation_requested(ctx, op),
        )
    try:
        parsed = _parse_declaration_source(ctx,
            bv,
            str(op["declaration"]),
            source_path=op.get("source_path"),
        )
    except OperationFailure:
        raise
    except Exception as exc:
        # A malformed C declaration is a user mistake, not an engine bug: surface
        # a clean invalid_request carrying BN's parser message, instead of
        # leaking the raw SyntaxError class / mislabeling it internal_error.
        # Mirrors _parse_type_or_hint for the other type-taking ops (#122).
        detail = " ".join(str(exc).split())
        raise OperationFailure(
            "invalid_request",
            "could not parse declaration" + (f": {detail}" if detail else "") + ".",
            requested=_operation_requested(ctx, op),
        ) from exc
    named_types = list(parsed["types"])
    # Backstop for any OTHER malformed layout the parser might emit (beyond the
    # bitfield case caught above): a member extending past the struct width is a
    # corrupt layout that must not be applied and reported `verified` (#322).
    # Checked before defining anything, so nothing is applied on rejection.
    for name, type_obj in named_types:
        bad = _struct_overflow_member(type_obj)
        if bad is not None:
            end = int(getattr(bad, "offset", 0)) + _bitfield_member_width(bad)
            raise OperationFailure(
                "invalid_request",
                f"declared type {str(name)!r} has a corrupt layout: member "
                f"{str(getattr(bad, 'name', '')) or '<unnamed>'!r} extends to "
                f"{hex(end)}, past the type width {hex(int(getattr(type_obj, 'width', 0) or 0))}. "
                f"The parser produced an inconsistent struct; refusing to apply it.",
                requested=_operation_requested(ctx, op),
            )
    defined_types = {}
    defined_type_layouts = {}
    before_defined_types = {}
    for name, type_obj in named_types:
        existing = ctx._current_type_entry(bv, str(name))
        before_defined_types[str(name)] = existing["decl"] if existing is not None else None
        bv.define_user_type(name, type_obj)
        current = ctx._current_type_entry(bv, str(name))
        defined_types[str(name)] = current["decl"] if current is not None else str(type_obj)
        defined_type_layouts[str(name)] = current["layout"] if current is not None else ctx._render_type_layout(type_obj)
    return {
        "op": "types_declare",
        "defined_types": defined_types,
        "defined_type_layouts": defined_type_layouts,
        "before_defined_types": before_defined_types,
        "count": len(defined_types),
        "parsed_functions": [name for name, _ in parsed["functions"]],
        "parsed_variables": [name for name, _ in parsed["variables"]],
        "parsed_type_count": len(named_types),
        "parsed_function_count": len(parsed["functions"]),
        "parsed_variable_count": len(parsed["variables"]),
        "requested": _operation_requested(ctx, op),
    }
