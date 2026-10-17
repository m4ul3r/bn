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
    ``_normalize_prototype``, ``OperationFailure``, and the type-name helpers
    ``_lookup_named_type`` / ``_named_type_references`` that ``read_decompile``
    also uses).

Import direction is one-way: this module imports ``il_format``, ``vars``, and
``_shared`` (plus stdlib + binaryninja). It NEVER imports ``bridge`` (the bridge
imports it), and ``read_types`` no longer reaches into it -- the cycle-breakers
``_find_type``/``_render_type_layout`` live on the seam (design spec §3.2).
"""
from __future__ import annotations

import contextlib
import contextvars
import difflib
import functools
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any, NamedTuple

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import vars as vars_mod
from .bridge_state import _quick_loaded_views
# #777: the ONE definition of the failure-status vocabulary, reached through the
# same symlinked-leaf arrangement paths/version/target_hint already use, because
# the bridge must not import the `bn` CLI package.
from .mutation_statuses import FAILED_MUTATION_STATUSES
from ._shared import (
    USER_FACING_ERRORS,
    OperationFailure,
    _BUILTIN_TAG_TYPE_NAMES,
    _lookup_named_type,
    _named_type_references,
    _normalize_prototype,
    _parse_address,
    _require_nonempty_name,
    _serialize_error,
    _split_qualified_name,
    _type_class_name,
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
    "data_retype": ("address", "new_type"),
    "struct_field_set": ("struct_name", "field_type", "offset", "field_name"),
    "struct_field_rename": ("struct_name", "old_name", "new_name"),
    "struct_field_delete": ("struct_name", "field_name"),
    "types_declare": ("declaration",),
    "function_create": ("address",),
    "function_delete": ("address",),
    "tag_add": ("type",),
    "tag_remove": (),
    "tag_type_create": ("name", "icon"),
    "tag_type_remove": ("name",),
}

# Ops that accept one of several alternative locator fields. set_comment /
# delete_comment target EITHER a function (`function`) OR an address (`address`):
# listing `address` as unconditionally required wrongly rejected the documented
# function-only form (#67). Each group requires at least one of its fields.
REQUIRED_ONE_OF: dict[str, tuple[tuple[str, ...], ...]] = {
    "set_comment": (("function", "address"),),
    "delete_comment": (("function", "address"),),
    "tag_add": (("function", "address"),),
    "tag_remove": (("tag_id", "address", "function"),),
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
    ``struct_name`` so the pre-apply snapshot pass and the apply both resolve the
    struct. (M12)

    #825: returns a normalized COPY instead of editing the caller's dict. The
    in-place version handed back the same object with an extra `struct_name` key
    written into it, so a caller's own manifest silently gained a field it never
    wrote -- externally observable, and a trap for any caller that re-reads,
    re-sends, hashes or diffs its request after a batch. Callers take the return
    value; an op needing no normalization is returned unchanged, so the common
    path still allocates nothing.
    """
    if not isinstance(op, dict):
        return op
    kind = op.get("op")
    if (isinstance(kind, str) and kind.startswith("struct_field_")
            and "struct_name" not in op and "type_name" in op):
        return {**op, "struct_name": op["type_name"]}
    return op

# Op kinds that mutate a function's local variables via create_user_var /
# set_user_type, which BN may PROPAGATE onto aliased siblings. Batches with any
# of these get a full per-function local snapshot so revert paths can undo the
# propagation, not just the targeted variable (see _capture_local_var_snapshots).
_VAR_DRIFT_OPS = {"local_rename", "local_retype", "set_prototype"}

# Valid batch op `kind` names, in dispatch order (kept in sync with the if-chain
# in _apply_operation). Used to suggest a close match when an agent reuses a CLI
# verb spelling like `proto_set`/`rename_local` (#361).
_BATCH_OP_NAMES = (
    "rename_symbol", "set_comment", "delete_comment", "set_prototype",
    "local_rename", "local_retype", "data_retype", "struct_field_set", "struct_field_rename",
    "struct_field_delete", "types_declare", "function_create", "function_delete", "tag_add", "tag_remove",
    "tag_type_create", "tag_type_remove",
)


def _suggest_batch_op(kind: str) -> str | None:
    """Closest valid batch op name for an unknown *kind*, or None (#361).

    The common agent error is a word-SWAPPED spelling (`local rename` -> typed
    `rename_local` instead of the op `local_rename`). A raw difflib match misleads
    here (it ranks `rename_local` closer to `rename_symbol` than to `local_rename`),
    so first map each op's token-reversed form to the op for an EXACT, non-misleading
    hit, then fall back to difflib for everything else."""
    reversed_aliases = {
        "_".join(reversed(op.split("_"))): op
        for op in _BATCH_OP_NAMES if "_" in op
    }
    if kind in reversed_aliases:
        return reversed_aliases[kind]
    matches = difflib.get_close_matches(kind, _BATCH_OP_NAMES, n=1, cutoff=0.4)
    return matches[0] if matches else None

# Soft upper bound on a struct field offset. A real struct is at most a few KiB;
# anything past this (256 MiB) is a mistyped/garbage offset that would explode
# the struct into a multi-GB sparse type, so struct field set refuses it (#369).
_MAX_STRUCT_FIELD_OFFSET = 0x10000000


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



def _declaration_include_root(source_path: str | None) -> str | None:
    """The implicit include root a `--file` declaration parses under.

    `parse_types_from_source` is handed the header's PARENT DIRECTORY as an
    include root, which is what lets `#include "sibling.h"` resolve. That is
    deliberate and load-bearing -- but it means a declare can succeed because
    of a directory the user never named, and if a sibling header shadows
    another of the same name, which one was used is invisible (#825 item 3).

    One function so the disclosure cannot state a different root than the
    parse used: the value reported and the value passed come from one call.
    """
    if not source_path:
        return None
    return str(Path(source_path).expanduser().resolve().parent)


# #993 item 2 / #1012 item 1: True while a `--preview` mutation runs. BN's own
# parser imports every type-library type a parse uses, its undo does not revert
# that, and REMOVING an import is worse (BN keeps its imported-record and never
# imports that type again). So every parse a preview runs passes BN's own
# `import_dependencies=False` (`_parse_import_kwargs`): BN then refuses a parse
# that needs a library type instead of importing it, and the preview stops on
# that op (`_preview_import_refusal`). The judge stays the outcome: the batch
# compares BN's type ids before and after (`_new_library_imports`) and fails a
# preview that gained any, whatever path brought it in.
_PREVIEW_PARSE: contextvars.ContextVar[bool] = contextvars.ContextVar("bn_preview_parse", default=False)

# Request-local attribution includes snapshot and preflight parses, not only apply.
_IMPORT_TRACE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "bn_mutation_import_trace", default=None)


def _track_operation(handler: Any) -> Any:
    """Record the active operation without enumerating the type container."""
    @functools.wraps(handler)
    def tracked(ctx: Any, bv: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
        trace = _IMPORT_TRACE.get()
        previous = trace.get("owner") if trace is not None else None
        if trace is not None and previous is None:
            trace["owner"] = (id(op), threading.get_ident())
        try:
            return handler(ctx, bv, op, *args, **kwargs)
        except Exception:
            if trace is not None:
                trace["failed_op"] = trace["owner"][0]
            raise
        finally:
            if trace is not None and previous is None:
                trace.pop("owner", None)
    return tracked


def _start_import_observer(bv: Any, trace: dict[str, Any]) -> None:
    """Journal SDK definition events; only the final ID diff judges imports."""
    if not (hasattr(bn, "BinaryDataNotification") and hasattr(bn, "NotificationType")
            and callable(getattr(bv, "register_notification", None))
            and callable(getattr(bv, "unregister_notification", None))):
        return

    class Definitions(bn.BinaryDataNotification):
        def __init__(self) -> None:
            super().__init__(bn.NotificationType.TypeDefined)

        def type_defined(self, view: Any, name: Any, type_obj: Any) -> None:
            type_id = view.get_type_id(name)
            if not type_id:
                return
            owner = trace.get("owner")
            # An event outside the operation's own SDK thread is not evidence
            # that this operation caused it. Keep it unowned rather than guess.
            operation = owner[0] if owner and owner[1] == threading.get_ident() else None
            trace["definitions"].setdefault(str(type_id), operation)

    observer = Definitions()
    bv.register_notification(observer)
    trace["observer"] = observer


def _stop_import_observer(trace: dict[str, Any]) -> None:
    observer = trace.pop("observer", None)
    if observer is not None:
        trace["bv"].unregister_notification(observer)


def _collect_import_outcome(trace: dict[str, Any]) -> None:
    """Take the one end-of-request snapshot, then attribute only actual imports."""
    if trace.get("collected") or trace.get("baseline") is None:
        return
    _stop_import_observer(trace)
    bv = trace["bv"]
    imported = _new_library_imports(bv, trace["baseline"])
    trace["library_imports"] = imported
    trace["library_sources"] = {
        name: source for name, type_id in imported.items()
        if (source := _import_library_name(bv, name, type_id)) is not None}
    owners = {id(op) for op in trace["operations"]} if imported else set()
    for name, type_id in imported.items():
        owner = trace["definitions"].get(type_id)
        if owner not in owners:
            trace["unattributed"].add(name)
        else:
            trace["imports"].setdefault(owner, set()).add(name)
    trace["collected"] = True


def _parse_import_kwargs() -> dict[str, bool]:
    """Keyword arguments for a BinaryView parse: under --preview, BN's own
    `import_dependencies=False`, so the parse imports nothing; a live parse
    keeps BN's default (import)."""
    return {"import_dependencies": False} if _PREVIEW_PARSE.get() else {}


# How BN's parser refuses a parse that `import_dependencies=False` kept from
# importing (a warning promoted to the parse error; it names no type).
_UNIMPORTED_LIBRARY_DIAGNOSTIC = "reference Library or Archive types that were not imported"

# A name a type string may use: plain, `::`-qualified, or backtick-quoted.
_TYPE_NAME_TOKEN_RE = re.compile(r"`([^`]+)`|([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)")


def _library_name_matches(bv, text: str) -> list[tuple[str, str]]:
    """``(name, library)`` for every name in *text* that the view lacks and an
    attached type library defines, in first-seen order. A lookup by name that
    imports nothing, used only to NAME what BN's parser already refused: a
    member that happens to share a library type's name is listed too."""
    stripped = _C_LITERAL_RE.sub('""', _strip_c_comments(text))
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for quoted, plain in _TYPE_NAME_TOKEN_RE.findall(stripped):
        name = re.sub(r"\s*::\s*", "::", quoted or plain)
        if name in seen:
            continue
        seen.add(name)
        if _lookup_named_type(bv, name) is not None:
            continue
        match = _library_type(bv, name)
        if match is not None:
            found.append((name, str(getattr(match[0], "name", "") or "<unnamed library>")))
    return found


def _library_match_text(matches: list[tuple[str, str]]) -> str:
    """``pid_t, FILE (libc_x86_64.so.6)`` for name matches grouped by library."""
    by_library: dict[str, list[str]] = {}
    for name, library in matches:
        by_library.setdefault(library, []).append(name)
    return "; ".join(f"{', '.join(names)} ({library})" for library, names in by_library.items())


def _preview_import_refusal(bv, text: str, exc: BaseException, *,
                            requested: dict[str, Any] | None = None) -> OperationFailure | None:
    """The preview refusal for a parse BN declined because it would import, or
    None for any other parse error (#1012 item 1). Nothing was imported: the
    parse ran with `import_dependencies=False`.

    BN's refusal names no type, so the message names none as required: it lists
    the by-name matches (`library_name_matches`) as matches only -- a member
    named `FILE` matches libc's `FILE` without being a type use (#1012 review r1)."""
    if not _PREVIEW_PARSE.get() or _UNIMPORTED_LIBRARY_DIAGNOSTIC not in str(exc):
        return None
    matches = _library_name_matches(bv, text)
    if matches:
        named = (" Names in the text that the view lacks and a loaded type library defines: "
                 f"{_library_match_text(matches)} -- a by-name match, so an identifier that "
                 "only shares a library type's name is listed too.")
    else:
        named = " No name in the text matched a loaded type library by lookup."
    return OperationFailure(
        "unsupported",
        "not previewed: Binary Ninja refused to parse this without importing a type-library "
        f"or type-archive type the view does not have yet.{named} Nothing was imported: a "
        "preview parses with import_dependencies=False, because no undo removes an import. "
        "Run without --preview to apply it; the import stays in the view.",
        requested=requested,
        observed={"library_name_matches": [name for name, _ in matches],
                  "libraries": {name: library for name, library in matches},
                  "imported": False},
    )




def _parse_declaration_source(ctx, bv, declaration: str, *, source_path: str | None = None):
    parse_result = None
    platform = getattr(bv, "platform", None)
    if platform is not None and hasattr(platform, "parse_types_from_source"):
        kwargs: dict[str, Any] = {}
        if source_path:
            kwargs["filename"] = source_path
            kwargs["include_dirs"] = [_declaration_include_root(source_path)]
        try:
            parse_result = platform.parse_types_from_source(declaration, **kwargs)
        except Exception:
            # The platform parser has no view type-library context. Its error
            # may call a valid library type unknown; the view fallback below
            # supplies the authoritative result, including any parse failure.
            pass

    if parse_result is None:
        try:
            if source_path:
                # BinaryView's library-aware parser accepts include_dirs but no
                # filename argument. A line directive retains the main filename
                # (and its original line numbers) in diagnostics from that parser.
                source = f"#line 1 {json.dumps(source_path)}\n{declaration}"
                parse_result = bv.parse_types_from_string(
                    source, include_dirs=[_declaration_include_root(source_path)],
                    **_parse_import_kwargs(),
                )
            else:
                parse_result = bv.parse_types_from_string(
                    declaration, **_parse_import_kwargs())
        except Exception as exc:
            refusal = _preview_import_refusal(bv, declaration, exc)
            if refusal is not None:
                raise refusal from exc
            raise

    return {
        "types": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "types", {}).items())],
        "variables": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "variables", {}).items())],
        "functions": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "functions", {}).items())],
    }



@_track_operation
def _parse_operation_declaration(ctx, bv, op: dict[str, Any]):
    trace = _IMPORT_TRACE.get()
    if trace is not None:
        trace["parsed"].add(id(op))
    return _parse_declaration_source(
        ctx, bv, str(op["declaration"]), source_path=op.get("source_path"),
    )


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
            parsed = _parse_operation_declaration(ctx, bv, op)
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
        elif kind == "function_create" and op.get("address"):
            # The function(s) a deliberate split cuts are affected too (#927
            # review): a gap address has none, an interior one without
            # `allow_overlap` is refused before the snapshot matters.
            functions = ctx._functions_containing(bv, _parse_address(op["address"]))
        elif kind == "data_retype":
            # A global's type reflows every function that reads it, so the blast
            # radius is the code xrefs to the address, not a single function (#649).
            address, _ = ctx._resolve_address_operand(bv, op["address"])
            functions = list(ctx._functions_referencing(bv, address, limit=type_limit))
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
            "layout": ctx._render_type_layout(type_obj, bv),
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
            # BN's whole-function documentation property (Function.comment),
            # DISTINCT from the address-comment store `comments` above --
            # `comment set/delete --function` (Task 8) writes here. Without
            # it in the snapshot, a --preview of that op showed changed:false
            # / an empty diff even though apply/verify/revert all landed
            # (final-review Fix 1, #121 pattern).
            "function_doc": str(getattr(fn, "comment", "") or ""),
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
    old_doc, new_doc = old.get("function_doc") or "", new.get("function_doc") or ""
    if old_doc != new_doc:
        lines.append(f"function_doc: {old_doc!r} -> {new_doc!r}")
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
        doc_changed = (old.get("function_doc") or "") != (new.get("function_doc") or "")
        diff = "\n".join(
            difflib.unified_diff(
                old["text"].splitlines(),
                new["text"].splitlines(),
                fromfile=f"before:{old.get('name', hex(address))}",
                tofile=f"after:{new.get('name', hex(address))}",
                lineterm="",
            )
        )
        if not diff and (name_changed or comments_changed or locals_changed or doc_changed):
            diff = _format_metadata_change(old, new, address)
        diffs.append(
            {
                "address": hex(address),
                "before_name": old.get("name"),
                "after_name": new.get("name"),
                "changed": bool(
                    text_changed or name_changed or comments_changed or locals_changed or doc_changed
                ),
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


def _refuse_unmapped_mutation_address(
    ctx, bv, op: dict[str, Any], address: int, consequence: str
) -> None:
    """Refuse an affirmatively-unmapped mutation target, and ONLY that.

    A typo'd address used to be accepted by the address-taking mutations,
    writing into nowhere and then reporting `verified` against its own
    readback -- while `comment get` rejected the identical address. An
    unmapped target is a bad REQUEST, not an unsupported operation, hence
    `invalid_request`.

    The guard is deliberately TRI-STATE and the middle state is the whole
    reason this is a helper rather than an inlined `if` (#890, the #777 drift
    shape): `is_valid_offset` answers mapped, unmapped, or nothing at all.
    Only an AFFIRMATIVE "unmapped" refuses. A view that does not implement
    the probe, or whose probe raises, is INDETERMINATE and must keep working
    -- reduced views and test fakes live there, and hardening the middle
    state into a refusal would reject perfectly good requests. That is the
    same distinction that kept `_require_mapped_address` and
    `_address_is_mapped` separate under #827 item 3c, so it is stated once
    here instead of being re-derived at each callsite.

    *consequence* completes "Address 0x… is not mapped in this binary, so …",
    because the actionable half of the message is what the caller was trying
    to do, which the callsite knows and this helper does not.
    """
    is_valid = getattr(bv, "is_valid_offset", None)
    if not callable(is_valid):
        return
    try:
        mapped = bool(is_valid(address))
    except Exception:  # noqa: BLE001 - a raising probe is indeterminate, not invalid
        mapped = True
    if not mapped:
        raise OperationFailure(
            "invalid_request",
            f"Address {hex(address)} is not mapped in this binary, so {consequence}",
            requested=_operation_requested(ctx, op),
        )



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


_RESOLVER_MISS_PREFIXES = ("Function not found: ", "Symbol not found: ")


def _is_resolver_miss(message: str, name: str) -> bool:
    """Whether *message* is the name resolver's own miss for exactly *name*
    (`Function not found: <name>` / `Symbol not found: <name>`, optionally
    followed by its `.`-led suggestions) -- never a different lookup's miss
    (`Variable not found: ...`) nor a longer name sharing *name* as a prefix."""
    for prefix in _RESOLVER_MISS_PREFIXES:
        head = prefix + name
        if message == head or message.startswith(head + "."):
            return True
    return False


def _explain_renamed_reference(op: Any, results: list[dict[str, Any]],
                               exc: OperationFailure) -> OperationFailure:
    """#975: a batch resolves each op's names against the state the earlier ops
    left, so naming a function by the spelling an earlier ``rename_symbol`` just
    replaced fails lookup -- and the resolver's "Did you mean" list then offers
    lexically close but unrelated functions, which reads as a wrong address rather
    than an ordering problem. When the failure IS the resolver's miss for the op's
    own name, and that name was renamed away earlier in this batch and never
    assigned again, replace the message with the rename that explains it.

    Both conditions matter. A name a later rename gave back (a round trip) or
    handed to another function (reuse) still resolves, so the op failed for its
    own reason -- a missing variable, say -- and a rename hint would bury that
    cause and point the agent at the function the name USED to denote. Any other
    failure is returned unchanged."""
    if not isinstance(op, dict):
        return exc
    reassigned = {row.get("new_name") for row in results if row.get("op") == "rename_symbol"}
    refs = {op[key] for key in ("identifier", "function")
            if isinstance(op.get(key), str) and op[key] not in reassigned
            and _is_resolver_miss(exc.message, op[key])}
    for index, row in enumerate(results):
        before = row.get("before_name")
        if row.get("op") != "rename_symbol" or before not in refs:
            continue
        address = row.get("address")
        renamed_to = row.get("new_name")
        # A second rename of the same symbol never reaches here: a rename chain
        # fails at the second rename's apply (`_refuse_rename_chain`, #995).
        return OperationFailure(
            exc.status,
            f"{before} not found: operation {index} of this batch renamed {before} "
            f"(at {address}) to {renamed_to}. Each operation resolves "
            f"names against the state the earlier operations left, so refer to it "
            f"as {renamed_to} or by its address {address}.",
            requested=exc.requested,
            observed=exc.observed,
        )
    return exc



def _unattempted_results(ctx, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A row per op the batch never reached, after a sibling op failed (#675
    item 15).

    The failure path used to return rows only up to and including the op that
    failed, so a two-op batch whose first op failed answered with ONE row. The
    response then could not express "not attempted": a consumer diffing the
    manifest against the results reads the missing op as absent from the
    request, and a consumer counting reads 1 where it submitted 2. That is the
    absence-versus-silence shape, in the batch envelope rather than a renderer
    -- and unlike the rendering cases it is not a wording problem, because
    there was no field carrying the fact at all.

    `not_attempted` is deliberately NOT in `FAILED_MUTATION_STATUSES`: these
    ops did not fail, they never ran, and stamping them as failures would turn
    one bad op into N and inflate the failure count a control loop reads. It
    sits beside `reverted` (#118) as the other honest non-failure status a
    doomed batch produces.

    Each row echoes its `requested` op, so the reconciliation a consumer needs
    -- one row per submitted op, in submission order -- holds without the
    consumer having to re-read the manifest it sent.
    """
    rows: list[dict[str, Any]] = []
    for op in operations:
        rows.append({
            "op": str(op.get("op") or "<missing>") if isinstance(op, dict) else "<non-object>",
            "status": "not_attempted",
            "message": (
                "not attempted: an earlier operation in this batch failed and the "
                "batch was rolled back, so this operation never ran"
            ),
            "requested": _operation_requested(ctx, op),
        })
    return rows


# #999 item 5: an op a --preview left out because it cannot be previewed
# reversibly, or depends on one that cannot. Like `not_attempted` it is NOT in
# `FAILED_MUTATION_STATUSES`: the op never ran, and the preview of the rest of
# the batch is still a real one.
NOT_PREVIEWED = "not_previewed"


def _not_previewed_result(ctx, op: Any, reason: str) -> dict[str, Any]:
    """The row for an op the preview skipped. It carries no `address`, so the
    has_user_type residue scan (`_prototype_user_type_residue`) never reads a
    skipped set_prototype as one that ran."""
    return {
        "op": str(op.get("op") or "<missing>") if isinstance(op, dict) else "<non-object>",
        "status": NOT_PREVIEWED,
        "message": reason,
        "requested": _operation_requested(ctx, op),
    }


def _disclose_not_previewed(result: dict[str, Any], skipped: dict[int, str],
                            operations: list) -> dict[str, Any]:
    """Say on the batch result itself that a preview did not cover every op:
    `preview_skipped` lists them by index, and the message leads with the gap so
    a reader of the message alone cannot take the preview for the whole batch."""
    if not skipped:
        return result
    result["preview_skipped"] = [
        {"index": index,
         "op": str(operations[index].get("op") or "<missing>")
         if isinstance(operations[index], dict) else "<non-object>",
         "reason": reason}
        for index, reason in sorted(skipped.items())
    ]
    result["message"] = (
        f"{len(skipped)} of {len(operations)} operations were not previewed (see "
        f"preview_skipped); the preview covers the rest only. {result.get('message') or ''}"
    ).rstrip()
    return result



def _mark_unverified_results(
    ctx, results: list[dict[str, Any]], message: str, status: str = "reverted"
) -> list[dict[str, Any]]:
    """Stamp an honest *status* on ops that ran before a sibling op failed.

    These ops WERE supported and applied -- they were rolled back because a
    later op failed, so the old generic ``unsupported`` was a lie (#118).
    ``reverted`` = cleanly rolled back (not a failure); ``rollback_failed`` =
    the revert itself failed, so the op may still be applied (a real failure
    state, so it stays in ``FAILED_MUTATION_STATUSES``).

    A genuine ``noop`` op (its requested state was already satisfied) changed
    nothing, so a rollback leaves it exactly as observed -- keep it ``noop`` (#630)
    rather than restamping it as reverted/rollback_failed and corrupting counts.
    A ``not_previewed`` op never ran, so it keeps its status too (#999 item 5)."""
    annotated = []
    for result in results:
        item = dict(result)
        if item.get("status") not in ("noop", NOT_PREVIEWED):
            item["status"] = status
            item["message"] = message
        annotated.append(item)
    return annotated



def _has_failed_results(ctx, results: list[dict[str, Any]]) -> bool:
    """True when any per-op result carries a FAILURE status (#777).

    This gates "revert the batch". It used to carry its own inline
    ``{"unsupported", "verification_failed"}`` literal -- a strict subset of the
    canonical five, missing ``invalid_request``, ``rollback_failed`` and
    ``internal_error``. The drift was latent rather than live (no ``_verify_*``
    raises the other three, and apply-phase failures short-circuit before this
    runs), but it was a landmine aimed squarely at the revert path: the next
    verifier to raise one of the missing statuses would leave the batch
    UN-REVERTED while the CLI still, correctly, reported exit 3. Reading the one
    shared definition removes the drift instead of re-syncing two copies.
    """
    return any(item.get("status") in FAILED_MUTATION_STATUSES for item in results)



# Per-op statuses that a full batch revert invalidates: once the batch is rolled
# back these ops are no longer live, so they must be re-stamped honestly (#602).
# ``noop`` is DELIBERATELY excluded (#630): a genuine no-op op (its requested
# state was already satisfied) changed nothing, so a rollback leaves it exactly
# as observed -- restamping it "reverted"/"rollback_failed" would corrupt the
# per-op status and the summary counts. A real failure status
# (verification_failed, unsupported, ...) is likewise left intact -- it is the
# reason for the rollback.
_SUCCESS_CLASS_STATUSES = {"verified", "applied"}


def _restamp_reverted_siblings(
    ctx, results: list[dict[str, Any]], restored: bool
) -> list[dict[str, Any]]:
    """On a rolled-back LIVE batch (a sibling op failed verification), re-stamp
    the ops that verified cleanly so they no longer claim ``verified``/``noop``:
    the whole batch was undone (#602). Mirrors the apply-failure honesty stamp
    (#118). Only success-class statuses are rewritten; the op that actually
    failed keeps its failure status, message and observed evidence."""
    stamp = "reverted" if restored else "rollback_failed"
    note = (
        "Rolled back because the batch did not commit."
        if restored
        else "Rollback failed; this operation may still be applied."
    )
    output = []
    for item in results:
        if item.get("status") in _SUCCESS_CLASS_STATUSES:
            restamped = dict(item)
            restamped["status"] = stamp
            restamped["message"] = note
            output.append(restamped)
        else:
            output.append(item)
    return output


# Batch op types that provably CANNOT change how a function or symbol resolves by
# NAME or ADDRESS: comments, local-variable edits, struct-field edits, type
# declarations and tags all leave the function/symbol namespace untouched, and
# ``set_prototype`` sets a type through ``set_user_type`` WITHOUT renaming or
# (re)defining any symbol (see ``_op_set_prototype``). Everything NOT on this
# whitelist -- ``rename_symbol`` (a function rename, OR a ``define_user_symbol``
# that creates a DataSymbol at an address), ``function_create`` (materializes a
# function/symbol at an address), and any unknown/future op -- is treated as
# possibly resolution-affecting. A WHITELIST (not a blocklist) is deliberate: a
# newly added op defaults to "could move resolution" -> defer its successors'
# preflight checks until that op has actually run inside the preview.
_RESOLUTION_SAFE_OPS = frozenset({
    "set_comment", "delete_comment", "set_prototype",
    "local_rename", "local_retype", "data_retype",
    "struct_field_set", "struct_field_rename", "struct_field_delete",
    "types_declare",
    "tag_add", "tag_remove", "tag_type_create", "tag_type_remove",
})


def _batch_changes_symbol_resolution(operations) -> bool:
    """True if any op in *operations* could change function/symbol NAME-or-ADDRESS
    resolution (a rename, a symbol/function create, or an unrecognized op).

    Whitelist-based: unknown or malformed ops default to possibly affecting
    resolution. This decides only where preflight must stop; preview checks
    after that point use BN's actual current-state lookup, not a name prediction."""
    return any(not isinstance(op, dict) or op.get("op") not in _RESOLUTION_SAFE_OPS
               for op in operations)


def _unrevertible_preview_prototypes(ctx, bv, operations) -> list[tuple[int, str]]:
    """AUTO-changing prototypes in the prefix safe to inspect before applying.

    Setting a prototype pins ``has_user_type``, which BN cannot clear. Preserve
    the pristine refusal for a standalone AUTO-changing prototype, but never
    parse or resolve across an earlier declaration or resolution-changing op:
    those prototypes are checked against the actual preview state instead."""
    flagged: list[tuple[int, str]] = []
    for index, op in enumerate(operations):
        if (not isinstance(op, dict) or op.get("op") == "types_declare"
                or _batch_changes_symbol_resolution([op])):
            break
        if op.get("op") != "set_prototype":
            continue
        label = _unrevertible_label(ctx, bv, op)
        if label is not None:
            flagged.append((index, label))
    return flagged


def _unrevertible_label(ctx, bv, op: dict[str, Any], *, fn=None) -> str | None:
    """Label an AUTO-changing prototype using BN's lookup and parsed Type.

    Call only when preceding ops have already run, or in a prefix whose targets
    and declarations are unchanged. A parse or resolution failure is a real
    error, never evidence that an op can be skipped -- except the preview's own
    refusal to import (#1012 item 1): a prototype that needs a type the view
    lacks differs from the current one, whose types are all in the view."""
    if op.get("identifier") is None:
        return None
    if fn is None:
        fn = _resolve_prototype_target(ctx, bv, op)
    if bool(getattr(fn, "has_user_type", False)):
        return None
    label = str(il_format._display_name(fn) or op["identifier"])
    try:
        expected_type = _parse_prototype(ctx, bv, op)
    except OperationFailure as exc:
        if (exc.observed or {}).get("imported") is not False:
            raise
        matches = (exc.observed or {}).get("library_name_matches") or []
        return (f"{label} (its prototype needs a type-library type the view does not have "
                "yet, which a preview does not import"
                + (f"; name matches: {', '.join(matches)}" if matches else "") + ")")
    if str(fn.type) == str(expected_type):
        return None
    return label


def _resolve_preview_function(ctx, bv, op: dict[str, Any], identifier):
    """Resolve read-only with the live mutation lookup and preserve its error."""
    try:
        fn = ctx._find_function(bv, identifier)
    except OperationFailure:
        raise
    except Exception as exc:
        raise OperationFailure("unsupported", str(exc),
                               requested=_operation_requested(ctx, op)) from exc
    if fn is None:
        raise OperationFailure("unsupported", f"Function not found: {identifier}",
                               requested=_operation_requested(ctx, op))
    return fn


def _resolve_prototype_target(ctx, bv, op: dict[str, Any]):
    return _resolve_preview_function(ctx, bv, op, op.get("identifier"))


# The ops whose meaning depends on a function's prototype: a parameter is a
# local, so renaming or retyping one means nothing until the prototype lands.
_PROTOTYPE_DEPENDENT_OPS = frozenset({"local_rename", "local_retype"})


@_track_operation
def _preview_operation_skip(ctx, bv, op: dict[str, Any], index: int,
                            skipped_prototypes: dict[int, int]) -> str | None:
    """Check one op immediately before applying it to the current preview view.

    ``skipped_prototypes`` maps actual function starts to earlier skipped
    prototype indices. Only read-only target resolution and prototype parsing
    happen here: never set a user type to discover whether it can be reverted.
    A target miss, ambiguity, malformed address or parse failure propagates the
    live error. Names are never matched against earlier rename requests."""
    if not isinstance(op, dict):
        return None
    if op.get("op") == "set_prototype":
        fn = _resolve_prototype_target(ctx, bv, op)
        label = _unrevertible_label(ctx, bv, op, fn=fn)
        if label is not None:
            skipped_prototypes.setdefault(int(fn.start), index)
            return (
                f"not previewed: {label} has no user prototype yet, and setting one "
                "pins its has_user_type flag, which Binary Ninja exposes no API to "
                "clear, so the preview could not be cleanly reverted")
    elif op.get("op") in _PROTOTYPE_DEPENDENT_OPS and skipped_prototypes:
        fn = _resolve_preview_function(ctx, bv, op, op.get("function"))
        source = skipped_prototypes.get(int(fn.start))
        if source is not None:
            return (f"not previewed: depends on operation {source}, a prototype "
                    "change this preview leaves out")
    return None


def _unpreviewable_operations(ctx, bv, operations) -> dict[int, str]:
    """Preflight only the prefix whose resolution/declarations are unchanged.

    This preserves the pristine all-skipped/standalone refusal. Every operation
    is checked again immediately before apply by ``_preview_operation_skip``;
    after an earlier rename, function create or type declaration, only that
    outcome-based check can decide skip versus a real error."""
    skipped: dict[int, str] = {}
    skipped_prototypes: dict[int, int] = {}
    for index, op in enumerate(operations):
        if (not isinstance(op, dict) or op.get("op") == "types_declare"
                or _batch_changes_symbol_resolution([op])):
            break
        reason = _preview_operation_skip(ctx, bv, op, index, skipped_prototypes)
        if reason is not None:
            skipped[index] = reason
    return skipped


def _prototype_user_type_residue(ctx, bv, results: list[dict[str, Any]]) -> bool:
    """True if any ``set_prototype`` op on a function that had NO prior user type
    left it pinned with ``has_user_type`` after the revert (#582). BN's undo and
    ``set_auto_type`` both leave ``BNFunctionHasUserType`` set (verified live on
    BN 5.4), so there is no supported clear. Callers use this only to DISCLOSE the
    residue in the message/result -- not to fail the batch: the prototype VALUE
    round-trips exactly, so a preview still verifies and reverts (exit 0). A
    function with a genuine prior user type is skipped: restoring that override is
    correct, not residue."""
    for item in results:
        if not isinstance(item, dict) or item.get("op") != "set_prototype":
            continue
        if item.get("before_has_user_type"):
            continue
        addr = item.get("address")
        if addr is None:
            continue
        try:
            fn = bv.get_function_at(_parse_address(addr))
            if fn is not None and bool(fn.has_user_type):
                return True
        except Exception:
            continue
    return False



def _member_byte_width(member) -> int:
    """The declared byte width of a struct member's type, or 0 when unknown."""
    try:
        return int(getattr(getattr(member, "type", None), "width", 0) or 0)
    except Exception:
        return 0


def _member_overlaps(member, offset: int, width: int) -> bool:
    """True when *member*'s byte range intersects the range a new field of
    *width* bytes at *offset* would occupy. Width 0/unknown is treated as 1
    byte (on either side) so an exact start-offset collision always counts."""
    new_start = int(offset)
    new_end = new_start + max(int(width or 0), 1)
    m_start = int(getattr(member, "offset", 0))
    m_end = m_start + max(_member_byte_width(member), 1)
    return new_start < m_end and m_start < new_end


def _overlapping_members(type_obj, offset: int, width: int) -> list:
    """Every existing member whose byte range intersects the range a new field
    of *width* bytes at *offset* would occupy (see ``_member_overlaps``). Used
    by the ``--no-overwrite`` clash check (#56)."""
    members = getattr(type_obj, "members", None)
    if members is None:
        return []
    return [member for member in list(members) if _member_overlaps(member, offset, width)]


def _first_overlapping_member(ctx, type_obj, offset: int, width: int):
    """The first existing member whose byte range intersects the range a new
    field of *width* bytes at *offset* would occupy, else None. Width 0/unknown
    is treated as 1 byte so an exact start-offset collision is always caught.
    Used to enforce struct field set --no-overwrite (#56)."""
    overlaps = _overlapping_members(type_obj, offset, width)
    return overlaps[0] if overlaps else None



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



def _function_at_result_address(ctx, bv, item: dict[str, Any]):
    """Re-resolve a function for VERIFY by its stored start address, not its
    stored name (#621). Apply already recorded ``fn.start``; a later op in the
    same batch may have renamed the function, so a name lookup would false-fail
    the whole batch. Address is stable across a rename."""
    addr = _parse_address(item["address"])
    fn = bv.get_function_at(addr)
    if fn is None:
        raise OperationFailure(
            "verification_failed",
            f"Function missing after mutation at {item['address']}",
            requested=item.get("requested"),
            observed={"address": item["address"], "function": None},
        )
    return fn


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
        if op == "data_retype":
            return _verify_data_retype(ctx, bv, result)
        if op == "struct_field_set":
            return _verify_struct_field_set(ctx, bv, result)
        if op == "struct_field_rename":
            return _verify_struct_field_rename(ctx, bv, result)
        if op == "struct_field_delete":
            return _verify_struct_field_delete(ctx, bv, result)
        if op == "types_declare":
            return _verify_declared_types(ctx, bv, result)
        if op == "function_create":
            return _verify_function_create(ctx, bv, result)
        if op == "tag_add":
            return _verify_tag_add(ctx, bv, result)
        if op == "tag_remove":
            return _verify_tag_remove(ctx, bv, result)
        if op == "tag_type_create":
            return _verify_tag_type_create(ctx, bv, result)
        if op == "tag_type_remove":
            return _verify_tag_type_remove(ctx, bv, result)
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
    # A rename that wrote nothing is not invalidated by a later real rename
    # through another locator. Keep its no-op verdict and report the live name.
    if observed_name != requested_name and before_name != requested_name:
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
    expected = str(item["requested"]["comment"])
    if item.get("scope") == "function_doc":
        fn = _function_at_result_address(ctx, bv, item)
        observed = str(getattr(fn, "comment", "") or "")
        item["observed"] = {"function": item["function"], "comment": observed}
    else:
        address = _parse_address(item["address"])
        observed = bv.get_comment_at(address) or ""
        item["observed"] = {"address": item["address"], "comment": observed}
    if observed != expected:
        raise OperationFailure(
            "verification_failed",
            f"Live comment verification failed for {item.get('function') or item['address']}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if item.get("before_comment", "") == expected else "verified"
    return item



def _verify_tag_add(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    type_name = str(item["tag_type"])
    data = str(item["data"])
    scope = item["scope"]
    if scope == "function":
        # Resolve by stored start address, not name: a later same-batch rename
        # would otherwise false-fail this verify (#621).
        fn = _function_at_result_address(ctx, bv, item)
        present = any(t.type.name == type_name and t.data == data
                      for t in fn.get_function_tags(auto=False))
    else:
        addr = _parse_address(item["address"])
        if scope == "data":
            present = any(t.type.name == type_name and t.data == data
                          for t in bv.get_tags_at(addr, auto=False))
        else:
            # address scope: verify against the EXACT function the tag was added
            # to (its recorded stable start address), not "any function containing
            # addr" -- overlapping functions could otherwise let a match on a
            # DIFFERENT function pass (#630). Fall back to the containing-functions
            # sweep only for older result dicts that predate function_start (#621).
            fn_start = item.get("function_start")
            if fn_start is not None:
                fn = bv.get_function_at(_parse_address(fn_start))
                present = fn is not None and any(
                    t.type.name == type_name and t.data == data
                    for t in fn.get_tags_at(addr, auto=False))
            else:
                present = any(t.type.name == type_name and t.data == data
                              for fn in bv.get_functions_containing(addr)
                              for t in fn.get_tags_at(addr, auto=False))
    item["observed"] = {"present": present}
    if not present:
        raise OperationFailure(
            "verification_failed",
            f"Tag {type_name!r} not found after add ({scope} scope).",
            requested=item.get("requested"), observed=item["observed"],
        )
    # #779: the readback above is unchanged -- the tag must still be present or
    # this raises. All that changes is WHICH success status is reported: an add
    # that found its (type, data) pair already there wrote nothing, and `noop`
    # is the status this codebase already uses for "already in the requested
    # state" (it is a success, and it does not affect the exit code).
    item["status"] = "noop" if item.get("pre_existing") else "verified"
    return item


def _verify_tag_remove(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    if not item.get("targets"):
        item["status"] = "noop"
        item["observed"] = {"removed": 0}
        return item
    still_present = []
    for tgt in item["targets"]:
        if tgt["scope"] == "function":
            # Resolve the EXACT function by its stored start address (stable
            # across a same-batch rename, #621). Only fall back to the name for
            # older result dicts that predate the address field -- NOT when the
            # stored address is present but resolves to nothing: a function gone
            # from its start address has no function-scope tags left, so the
            # removal verifies rather than matching a different function by name
            # (#630).
            if tgt.get("address") is not None:
                fn = bv.get_function_at(_parse_address(tgt["address"]))
                if fn is None:
                    continue  # function gone -> its function tag cannot remain
            else:
                fn = ctx._find_function(bv, tgt["function"])
            ids = {str(t.id) for t in fn.get_function_tags(auto=False)}
        elif tgt["scope"] == "data":
            ids = {str(t.id) for t in bv.get_tags_at(_parse_address(tgt["address"]), auto=False)}
        else:
            addr = _parse_address(tgt["address"])
            ids = set()
            for fn in bv.get_functions_containing(addr):
                ids |= {str(t.id) for t in fn.get_tags_at(addr, auto=False)}
        if tgt["id"] in ids:
            still_present.append(tgt["id"])
    item["observed"] = {"still_present": still_present}
    if still_present:
        raise OperationFailure(
            "verification_failed",
            f"{len(still_present)} tag(s) still present after remove.",
            requested=item.get("requested"), observed=item["observed"],
        )
    item["status"] = "verified"
    return item


def _verify_tag_type_create(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    name = str(item["name"])
    tt = bv.get_tag_type(name)
    item["observed"] = {"present": tt is not None}
    if tt is None:
        raise OperationFailure(
            "verification_failed",
            f"Tag type {name!r} not present after create.",
            requested=item.get("requested"), observed=item["observed"],
        )
    item["status"] = "noop" if item.get("existed") else "verified"
    return item


def _verify_tag_type_remove(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    name = str(item["name"])
    tt = bv.get_tag_type(name)
    item["observed"] = {"present": tt is not None}
    if tt is not None:
        raise OperationFailure(
            "verification_failed",
            f"Tag type {name!r} still present after remove.",
            requested=item.get("requested"), observed=item["observed"],
        )
    item["status"] = "noop" if not item.get("existed") else "verified"
    return item


def _verify_delete_comment(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    if item.get("scope") == "function_doc":
        fn = _function_at_result_address(ctx, bv, item)
        observed = str(getattr(fn, "comment", "") or "")
        item["observed"] = {"function": item["function"], "comment": observed}
    else:
        address = _parse_address(item["address"])
        observed = bv.get_comment_at(address) or ""
        item["observed"] = {"address": item["address"], "comment": observed}
    if observed:
        raise OperationFailure(
            "verification_failed",
            f"Live comment deletion verification failed for {item.get('function') or item['address']}",
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
        expected_type, _ = bv.parse_type_string(
            str(expected_str), **_parse_import_kwargs())
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
            # #999 item 1: parse the REQUESTED text, not `expected` -- `str(Type)`
            # drops the backticks a `::`-qualified type needs, so it never parses.
            requested_text = (item.get("requested") or {}).get("prototype")
            source = (_quote_qualified_declarator(str(requested_text))
                      if requested_text is not None else expected)
            if not _prototype_matches_ignoring_param_names(bv, fn.type, source):
                raise OperationFailure(
                    "verification_failed",
                    f"Live prototype verification failed at {item['address']}",
                    requested=item.get("requested"),
                    observed=item["observed"],
                )
    # #966: the prototype comparisons above ignore `__noreturn` (BN may re-infer
    # it on readback, #199), so a requested `can_return` change is judged on its
    # own, by BN's effective `can_return` after reanalysis.
    observed_can_return = _function_can_return(fn, observed)
    item["observed"]["can_return"] = observed_can_return
    expected_can_return = item.get("expected_can_return")
    if (expected_can_return is not None and expected_can_return != item.get("before_can_return")
            and observed_can_return != expected_can_return):
        raise OperationFailure(
            "verification_failed",
            f"Live prototype verification failed at {item['address']}: the prototype "
            + ("requests __noreturn, but the function can still return"
               if expected_can_return is False else
               "drops __noreturn, but the function still cannot return")
            + " after reanalysis (can_return reads "
            + ("unreadable" if observed_can_return is None else str(observed_can_return).lower())
            + ").",
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
    # #758 observes, correctly, that this guard is unreachable from the one
    # production dispatch path: `_op_types_declare` raises `invalid_request` the
    # moment its parse yields no named types, and that raise short-circuits the
    # batch long before verification runs, so a real result always carries at
    # least one entry here. RETAINED anyway, deliberately.
    #
    # This is a verifier -- the function whose entire job is refusing to stamp
    # `verified` on something it did not observe. An empty `defined_types` is a
    # result claiming a declaration landed while naming nothing that landed, and
    # the failure mode of removing the check is the worst one this codebase has:
    # a mutation that declared nothing reported as a success. The guard costs
    # eight lines and one branch, it is pinned by
    # `test_declared_types_verifier_rejects_an_empty_apply_result`, and
    # `_verify_operation` is reachable from batch replay as well as the single
    # op path, so "today's only caller cannot produce it" is a weaker guarantee
    # than it looks. Unreachable-by-construction is a reason to keep a boundary
    # check cheap, not a reason to delete it.
    # #989 review: a declaration of only forward declarations of types a type
    # library defines (`struct timespec;`) changes nothing and has no view type
    # to read back. What CAN be observed is that no user stub now shadows them;
    # that is the whole claim, so it is checked rather than assumed.
    if not defined_types and item.get("reference_only"):
        referenced = [str(name) for name in item.get("referenced_existing_types") or []]
        shadowed = [name for name in referenced if bv.get_type_by_name(name) is not None]
        item["observed"] = {"defined_types": {}, "referenced_existing_types": referenced,
                            "shadowing_view_types": shadowed}
        if referenced and not shadowed:
            item["status"] = "noop"
            return item
    if not defined_types:
        item["observed"] = {
            "defined_types": {},
            "parsed_functions": list(item.get("parsed_functions") or []),
            "parsed_variables": list(item.get("parsed_variables") or []),
        }
        item["status"] = "verification_failed"
        item["message"] = "No declared named types were available for live verification."
        return item
    observed_types: dict[str, str | None] = {}
    observed_type_layouts: dict[str, str | None] = {}
    for name, expected in defined_types.items():
        type_obj = bv.get_type_by_name(name)
        observed_types[name] = str(type_obj) if type_obj is not None else None
        observed_type_layouts[name] = ctx._render_type_layout(type_obj, bv) if type_obj is not None else None
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



def _suggest_field(provided: Any, missing: str) -> str | None:
    """A provided key that looks like the *missing* required field, else None (#650).

    The batch path has no argparse layer, so a guessed field name fails only at
    APPLY time and -- the batch being atomic -- reverts every good op with it. One
    dogfood agent rolled back 12 good ops over a field name; the fields are also
    not mutually consistent across ops (``local_retype`` takes ``variable`` where
    ``rename_symbol`` takes ``identifier``), so the guess is easy to make."""
    if not isinstance(provided, dict):
        return None
    candidates = [str(k) for k in provided if k not in ("op", "preview", "target")]
    matches = difflib.get_close_matches(missing, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _validate_operation_request(ctx, op: dict[str, Any], *, index: int | None = None) -> str:
    """Validate one manifest op's shape and symbol name, and return its kind.

    Split out of ``_apply_operation`` so a batch can check EVERY op before
    applying ANY (#650): the batch is atomic, so a typo in op 13 used to roll back
    12 good ops. Covers a missing/unknown op kind (with a did-you-mean), missing
    required fields (with a did-you-mean drawn from the keys actually provided),
    the one-of locator groups, enum-valued fields, and symbol-name validity.
    """
    where = f"operation {index} " if index is not None else "operation "
    # A missing `op` key must be its own invalid_request, NOT silently assumed
    # to be a rename_symbol -- a typo'd/absent op kind would otherwise apply
    # the wrong mutation (#48). Internal single-op callers always set `op`.
    kind = op.get("op")
    if not kind:
        raise OperationFailure(
            "invalid_request",
            f"{where}is missing required field 'op' (the mutation kind, e.g. "
            "'rename_symbol', 'set_comment', 'types_declare')",
            requested=_operation_requested(ctx, op),
        )
    kind = str(kind)
    if kind not in _BATCH_OP_NAMES:
        # #361's did-you-mean, now reachable BEFORE anything is applied (#650).
        match = _suggest_batch_op(kind)
        hint = f" Did you mean {match!r}?" if match else ""
        raise OperationFailure(
            "unsupported",
            f"Unsupported operation: {kind}.{hint} Valid op names: "
            f"{', '.join(_BATCH_OP_NAMES)}.",
            requested=_operation_requested(ctx, op),
        )
    # Validate required request fields up front so a malformed request is
    # reported precisely (invalid_request, naming the field) and a KeyError
    # raised deeper -- e.g. by BN internals inside a handler -- is no longer
    # misreported as a missing request field.
    for field in REQUIRED_FIELDS.get(kind, ()):
        if field not in op:
            near = _suggest_field(op, field)
            hint = f" (got {near!r}; did you mean {field!r}?)" if near else ""
            raise OperationFailure(
                "invalid_request",
                f"{where}{kind!r} is missing required field {field!r}{hint}",
                requested=_operation_requested(ctx, op),
            )
    for group in REQUIRED_ONE_OF.get(kind, ()):
        if not any(field in op for field in group):
            raise OperationFailure(
                "invalid_request",
                f"{where}{kind!r} requires one of "
                f"{' / '.join(repr(f) for f in group)}",
                requested=_operation_requested(ctx, op),
            )
    for field, allowed in ENUM_FIELDS.get(kind, {}).items():
        if field in op and str(op[field]) not in allowed:
            raise OperationFailure(
                "invalid_request",
                f"{where}{kind!r} field {field!r} must be one of "
                f"{' / '.join(repr(v) for v in allowed)}, got {str(op[field])!r}",
                requested=_operation_requested(ctx, op),
            )
    if kind == "rename_symbol":
        # Validate the entire batch before any sibling can write (#923).
        _require_nonempty_name(
            op["new_name"], requested=_operation_requested(ctx, op), reject_controls=True,
            label=f"{where}{kind!r} new name",
        )
    return kind


@_track_operation
def _apply_operation(ctx, bv, op: dict[str, Any], restores: list | None = None, *,
                     defer_reanalysis: bool = False):
    # A manifest op must be a JSON object; a non-object element (e.g.
    # "ops": ["foo"]) gets a clean invalid_request, not an AttributeError (#48).
    if not isinstance(op, dict):
        raise OperationFailure(
            "invalid_request",
            f"each manifest operation must be a JSON object, got {type(op).__name__}",
        )
    op = _normalize_struct_alias(op)  # type_name -> struct_name alias (M12/#825)
    kind = _validate_operation_request(ctx, op)
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
        if kind == "data_retype":
            return _op_data_retype(ctx, bv, op)
        if kind == "struct_field_set":
            return _op_struct_field_set(ctx, bv, op)
        if kind == "struct_field_rename":
            return _op_struct_field_rename(ctx, bv, op)
        if kind == "struct_field_delete":
            return _op_struct_field_delete(ctx, bv, op)
        if kind == "types_declare":
            return _op_types_declare(ctx, bv, op)
        if kind == "function_create":
            return _op_function_create(ctx, bv, op, restores,
                                       defer_reanalysis=defer_reanalysis)
        if kind == "tag_add":
            return _op_tag_add(ctx, bv, op)
        if kind == "tag_remove":
            return _op_tag_remove(ctx, bv, op)
        if kind == "tag_type_create":
            return _op_tag_type_create(ctx, bv, op)
        if kind == "tag_type_remove":
            return _op_tag_type_remove(ctx, bv, op)
        # The batch op names are a different namespace than the CLI verbs
        # (`bn proto set` -> op `set_prototype`); an agent reusing the CLI verb
        # gets a bare "unsupported" and silently wastes a batch. Suggest the
        # closest valid op name and always list them all (#361).
        match = _suggest_batch_op(str(kind))
        hint = f" Did you mean {match!r}?" if match else ""
        raise OperationFailure(
            "unsupported",
            f"Unsupported operation: {kind}.{hint} Valid op names: "
            f"{', '.join(_BATCH_OP_NAMES)}.",
            requested=_operation_requested(ctx, op),
        )
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
        # A captured identifier that no longer resolves must NOT fall through to
        # the storage lookup: _find_variable_by_storage matches on storage
        # alone, so a DIFFERENT variable now occupying the same storage would
        # match and the restore closure would stamp the old name/type onto the
        # wrong logical variable (#521). Give up instead -- the storage fallback
        # is only safe when BN never gave us an identifier to be exact with.
        return None
    try:
        var, _ = vars_mod._find_variable_by_storage(fn, int(storage), is_parameter=is_parameter)
        return var
    except RuntimeError:
        return None



def _capture_local_var_snapshots(ctx, bv, functions) -> dict[int, dict[int, tuple[str, Any, bool]]]:
    """Snapshot every identifiable local's (name, type, was_user) per function.

    BN's create_user_var/set_user_type PROPAGATE a user name/type onto
    aliased variables -- naming a stack var also names the register that
    copies it (e.g. naming var_8 also renames the aliased r2 to var_8_1).
    Those propagated siblings are NOT the variable an op targeted, so the
    per-op explicit restore (_run_local_restores) never reverts them, and a
    --preview or a rolled-back batch would leave them modified. Snapshotting
    the whole canonical set lets _restore_local_var_drift put every drifted
    local back, not just the targeted one.

    ``was_user`` records each local's AUTO/USER provenance BEFORE apply so the
    drift restore can pick the RIGHT inverse (#630): an AUTO local that BN
    propagated a USER override onto must be un-pinned with ``delete_user_var``,
    NOT re-pinned with ``create_user_var`` -- the latter restores the displayed
    value but permanently stamps the var USER, so it comes back USER after a
    preview/rollback that claimed a clean revert. Absent a way to read
    provenance (exotic doubles), default USER so the value is at least replayed."""
    snapshots: dict[int, dict[int, tuple[str, Any, bool]]] = {}
    for fn in functions:
        entries: dict[int, tuple[str, Any, bool]] = {}
        has_prov = hasattr(fn, "is_var_user_defined")
        try:
            for var, _ in vars_mod._iter_canonical_variables(fn):
                identifier = vars_mod._variable_identifier(var)
                if identifier is None:
                    continue  # can't re-resolve it after reanalysis
                was_user = bool(fn.is_var_user_defined(var)) if has_prov else True
                entries[identifier] = (str(var.name), var.type, was_user)
        except Exception as exc:
            bn.log_error(f"BN Agent Bridge: local var snapshot failed for {hex(int(fn.start))}: {exc!r}")
        snapshots[int(fn.start)] = entries
    return snapshots



def _settle_before_drift_restore(bv, var_before) -> None:
    """Settle analysis before reading var drift (#581/#657).

    ``revert_undo_actions`` restores provenance, but BN only re-derives an AUTO
    variable's name/type on the next analysis pass, and the restores that ran
    before this may have skipped their own reanalysis (they do when the failing
    op registered none -- the blast-radius case). A stale read here would see
    phantom drift and re-pin an AUTO local as USER, so every revert path must
    call this immediately before ``_restore_local_var_drift``. Nothing to
    compare (empty snapshot) means no reanalysis is needed; a failing settle is
    logged and absorbed, never raised -- the drift read after it stays
    best-effort."""
    if not var_before:
        return
    try:
        bv.update_analysis_and_wait()
    except Exception as exc:
        bn.log_error(f"BN Agent Bridge: reanalysis before var-drift check failed: {exc!r}")


def _restore_local_var_drift(ctx, bv, snapshots) -> bool:
    """Put any local whose (name, type) drifted from *snapshots* back, then
    reanalyze. Covers BN's name/type propagation onto aliased siblings that
    the targeted per-op restores miss (see _capture_local_var_snapshots).
    Returns False if any restore raised. Callers must settle analysis after
    ``revert_undo_actions`` before invoking this, so drift is read post-
    reanalysis and BN's re-derived AUTO names are not mistaken for drift (#581)."""
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
        has_prov = hasattr(rfn, "is_var_user_defined")
        for var, _ in current:
            identifier = vars_mod._variable_identifier(var)
            if identifier is None or identifier not in entries:
                continue
            snap = entries[identifier]
            snap_name, snap_type = snap[0], snap[1]
            snap_was_user = snap[2] if len(snap) > 2 else True
            drifted = not (str(var.name) == snap_name and str(var.type) == str(snap_type))
            if snap_was_user:
                # Genuine prior USER local: re-assert it exactly if it drifted.
                if not drifted:
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
                continue
            # Snapshot AUTO. A residual USER override -- BN propagated it onto this
            # aliased sibling and neither the undo journal nor the targeted restore
            # cleared it -- must be DROPPED so analysis re-derives the AUTO value.
            # Re-pinning it with create_user_var (the pre-#630 behavior) would
            # leave the var USER while claiming a clean revert. If the var is still
            # AUTO, any name/type difference is BN's own re-derivation (phantom
            # drift, #581) -- leave it untouched.
            cur_is_user = bool(rfn.is_var_user_defined(var)) if has_prov else False
            if not cur_is_user:
                continue
            delete = getattr(rfn, "delete_user_var", None)
            if delete is None:
                # No inverse to un-pin the propagated override: honest failure
                # rather than a create_user_var re-pin that fakes a clean revert.
                ok = False
                bn.log_error(
                    "BN Agent Bridge: cannot un-pin AUTO local "
                    f"(identifier {identifier}) in {hex(int(fn_start))}: "
                    "delete_user_var unavailable; view may be left modified"
                )
                continue
            try:
                delete(var)
                touched = True
            except Exception as exc:
                ok = False
                bn.log_error(
                    "BN Agent Bridge: failed to un-pin a propagated AUTO local "
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



def _normalized_locator(value: Any) -> Any:
    """An address-or-name locator in a canonical form, so ``"0x401120"`` and
    ``4198688`` are recognized as the same key (#652)."""
    if value is None:
        return None
    try:
        return ("addr", int(_parse_address(value)))
    except Exception:
        return ("name", str(value))


# Per-op key identifying WHAT a manifest op writes, grouped so that only ops
# verified against the SAME piece of state share a family (#652). local_rename and
# local_retype touch one variable but verify different attributes, so they do not
# conflict; set_comment and delete_comment do. Accumulative ops (tag_add /
# tag_remove) and ops with no cheaply-derivable key (types_declare,
# function_create) are deliberately absent -- listing them would reject legitimate
# manifests, and a missed conflict still fails the way it does today.
def _op_conflict_key(op: dict[str, Any]) -> tuple | None:
    kind = op.get("op")
    if kind in ("set_comment", "delete_comment"):
        locator = op.get("function") if op.get("function") is not None else op.get("address")
        return ("comment", _normalized_locator(locator))
    if kind == "rename_symbol":
        return ("symbol_name", _normalized_locator(op.get("identifier")))
    if kind == "set_prototype":
        return ("prototype", _normalized_locator(op.get("identifier")))
    if kind == "local_rename":
        return ("local_name", _normalized_locator(op.get("function")), str(op.get("variable")))
    if kind == "local_retype":
        return ("local_type", _normalized_locator(op.get("function")), str(op.get("variable")))
    if kind == "data_retype":
        return ("data_type", _normalized_locator(op.get("address")))
    if kind == "struct_field_set":
        return ("struct_field_set", str(op.get("struct_name")), _normalized_locator(op.get("offset")))
    if kind == "struct_field_rename":
        return ("struct_field_rename", str(op.get("struct_name")), str(op.get("old_name")))
    if kind == "struct_field_delete":
        return ("struct_field_delete", str(op.get("struct_name")), str(op.get("field_name")))
    return None


_CONFLICT_KEY_LABELS = {
    "comment": "the comment at",
    "symbol_name": "the symbol name of",
    "prototype": "the prototype of",
    "local_name": "the name of the local in",
    "local_type": "the type of the local in",
    "data_type": "the data-variable type at",
    "struct_field_set": "the struct field at offset in",
    "struct_field_rename": "the struct field name in",
    "struct_field_delete": "the struct field in",
}


def _reject_duplicate_write_keys(ctx, operations: list[dict[str, Any]]) -> None:
    """Refuse a manifest that writes the same key twice, up front (#652).

    Every op is verified against the batch's END state, so under last-write-wins
    the earlier op is judged against the later op's value, fails
    ``verification_failed``, and takes the whole (atomic) batch down with it --
    meaning such a manifest is structurally unappliable however correct its intent.
    That is easy to hit when a manifest is machine-generated from two passes, which
    is the recommended bulk workflow. An ``invalid_request`` before anything is
    applied beats rolling back N good ops, and naming BOTH indices says what to fix.
    """
    first_seen: dict[tuple, int] = {}
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            continue
        key = _op_conflict_key(op)
        if key is None:
            continue
        previous = first_seen.get(key)
        if previous is None:
            first_seen[key] = index
            continue
        label = _CONFLICT_KEY_LABELS.get(key[0], "the same state in")
        locator = key[-1]
        target = hex(locator[1]) if isinstance(locator, tuple) and locator[0] == "addr" else str(
            locator[1] if isinstance(locator, tuple) else locator)
        raise OperationFailure(
            "invalid_request",
            f"operations {previous} and {index} both write {label} {target}: every op "
            f"is verified against the batch's END state, so operation {previous} would "
            f"be judged against operation {index}'s value, fail verification, and roll "
            f"the whole batch back. Keep one write per key (last-write-wins is not "
            f"expressible in a single batch) or split them across two batches.",
            requested={
                "op": str(op.get("op")),
                "conflicting_op_indices": [previous, index],
                "conflict_key": target,
            },
        )


def _refuse_rename_chain(row: dict[str, Any], results: list[dict[str, Any]],
                         index: int) -> None:
    """Fail a batch that renames one symbol twice, at the second rename (#995).

    Every op is verified against the batch's END state, so the first rename of
    a chain (`add -> foo`, then `foo -> bar`) was judged against `bar` and
    failed as `Live rename verification failed at <addr>` -- a rollback that read
    as a BN problem at an address, not as a manifest that renames one function
    twice.

    The proof is the apply itself: *row* is what `_op_rename_symbol` did, and
    its address is where Binary Ninja's resolver -- with its own case-folding,
    impl-over-stub, `kind` and raw-symbol rules, against the state the earlier
    ops left -- actually landed the rename. Two renames at one address rename
    one symbol slot (a `kind: data` rename at a function's address renames that
    function), so the batch can never verify unless both chose the same name.
    Two review rounds showed that simulating the resolver up front misses some
    chains and, worse, refuses batches that work while advising a rename of the
    wrong symbol; this check cannot do either, because it never predicts. The
    cost is that the refusal comes after the earlier ops applied, so the batch
    is rolled back like any other apply failure."""
    if row.get("op") != "rename_symbol" or row.get("address") is None:
        return
    for first, earlier in enumerate(results):
        if (earlier.get("op") != "rename_symbol" or earlier.get("address") != row["address"]
                or earlier.get("before_name") == earlier.get("new_name")
                or earlier.get("new_name") == row.get("new_name")):
            continue
        advice = json.dumps({
            "op": "rename_symbol",
            "identifier": row["address"],
            "new_name": row.get("new_name"),
        })
        raise OperationFailure(
            "invalid_request",
            f"operation {index} renames the symbol at {row['address']} again: operation "
            f"{first} already renamed it ({earlier.get('before_name')} -> "
            f"{earlier.get('new_name')}), and operation {index}'s identifier "
            f"{(row.get('requested') or {}).get('identifier')!r} resolved to that same "
            f"symbol, so it would be renamed again to {row.get('new_name')}. Every op is "
            f"verified against the batch's END state, so operation {first} could never "
            f"verify. Drop operation {first} and make operation {index} rename the "
            f"symbol by its address: {advice} -- kept in "
            f"that position, after the renames between them (moved first, it can run into "
            f"an ambiguous name). Dropping operation {first} means its intermediate name "
            f"{earlier.get('new_name')!r} disappears; update any operations that refer to "
            f"it to use the address too -- or split the renames across two batches.",
            requested={
                **(row.get("requested") or {}),
                "conflicting_op_indices": [first, index],
                "conflict_key": row["address"],
            },
        )


def _later_create_needs_ranges(operations: list, index: int) -> bool:
    """Whether op *index* is a `function_create` with another one after it.

    The later create's interior guard (`_op_function_create`) must see the
    earlier one's analyzed ranges (#927 P2), and only a create's guard reads
    them -- so a batch with a single create, or with creates followed only
    by other ops, keeps the one post-apply analysis #780 left it with.
    """
    op = operations[index]
    if not (isinstance(op, dict) and op.get("op") == "function_create"):
        return False
    return any(isinstance(later, dict) and later.get("op") == "function_create"
               for later in operations[index + 1:])



_QUICK_VIEW_REANALYSIS = (
    "Mutations reanalyze under the write lock, which would wedge the instance."
)


def refuse_quick_view(bv, *, requested: dict[str, Any],
                      why: str = _QUICK_VIEW_REANALYSIS) -> None:
    """The mutation eligibility policy for a --quick view (#479): refuse before any
    write, live and --preview alike, and point at `bn refresh`. Shared by the
    generic engine and `go rename`, which bypasses the engine (#952) -- on a quick
    view the pcln walk sees almost no defined functions, so a run would report a
    measured zero-candidate success over a target that was never analysed. *why*
    names the caller's own reason the view is ineligible."""
    if bv in _quick_loaded_views:
        raise OperationFailure(
            "invalid_request",
            "Cannot apply mutations: this target was loaded with --quick (no "
            f"analysis). {why} Run `bn refresh` first.",
            requested=requested,
        )


@contextlib.contextmanager
def _import_outcome(ctx, selector, preview: bool, *, exclusive=None, direct: bool = False):
    """Share one retained-ID outcome across a request, preview or live,
    including the direct paths (standalone `function create`, Go rename, class
    apply-types) that bypass the batch engine (#1012 review r1: a live direct
    path skipped tracking, so its analysis-side imports went unreported).

    The caller holds its write gate; only the two type snapshots take the target
    lock. A nested engine call keeps this request's baseline and finalizer.
    """
    if _IMPORT_TRACE.get() is not None:
        yield lambda result: result
        return
    token = _PREVIEW_PARSE.set(bool(preview))
    trace: dict[str, Any] = {
        "imports": {}, "library_imports": {}, "parsed": set(), "import_errors": {},
        "definitions": {}, "unattributed": set(),
    }
    trace_token = _IMPORT_TRACE.set(trace)
    exclusive_scope = exclusive if exclusive is not None else contextlib.nullcontext
    try:
        if direct:
            with exclusive_scope():
                bv = ctx._resolve_view(selector)
                baseline = _declare_batch_type_names(bv, [])
                trace.update(bv=bv, baseline=baseline, operations=[])
                _start_import_observer(bv, trace)
        try:
            yield lambda result: _finalize_import_outcome(
                ctx, trace, result, preview, exclusive_scope)
        except Exception as exc:
            if trace.get("baseline") is not None:
                with exclusive_scope():
                    _collect_import_outcome(trace)
            if trace["library_imports"]:
                exc.kept_library_imports = sorted(trace["library_imports"])
                if isinstance(exc, OperationFailure):
                    exc.observed["kept_library_imports"] = exc.kept_library_imports
                    if trace.get("library_sources"):
                        exc.observed["library_sources"] = dict(sorted(trace["library_sources"].items()))
            raise
    finally:
        try:
            _stop_import_observer(trace)
        finally:
            _IMPORT_TRACE.reset(trace_token)
            _PREVIEW_PARSE.reset(token)


def _mutation(ctx, selector: str | None, preview: bool, operations: list[dict[str, Any]], *,
              exclusive=None):
    """Judge imports from one request-level ID delta, independent of attribution."""
    with _import_outcome(ctx, selector, preview, exclusive=exclusive) as finish:
        return finish(_apply_mutation(ctx, selector, preview, operations, exclusive=exclusive))


def _apply_mutation(ctx, selector, preview: bool, operations, *, exclusive=None):
    trace = _IMPORT_TRACE.get()
    exclusive_scope = exclusive if exclusive is not None else contextlib.nullcontext
    try:
        result = _apply_batch(ctx, selector, preview, operations, exclusive=exclusive)
    except Exception as exc:
        # A preflight/snapshot can parse before an undo group exists. Even
        # this exit must disclose its irreversible imports and their owner.
        if trace.get("baseline") is not None:
            with exclusive_scope():
                _collect_import_outcome(trace)
        if trace["library_imports"]:
            exc.kept_library_imports = sorted(trace["library_imports"])
            if isinstance(exc, OperationFailure):
                exc.observed["kept_library_imports"] = exc.kept_library_imports
                if trace.get("library_sources"):
                    exc.observed["library_sources"] = dict(sorted(trace["library_sources"].items()))
        if (not trace["library_imports"] or trace.get("transaction_started")
                or not isinstance(exc, OperationFailure)):
            raise
        ops = trace["operations"]
        failed_id = trace.get("failed_op")
        rows = [
            _operation_failure_result(ctx, op, exc)
            if id(op) == failed_id and isinstance(exc, OperationFailure)
            else _unattempted_results(ctx, [op])[0]
            for op in ops
        ]
        _report_library_imports(trace["bv"], trace["baseline"], rows)
        if preview:
            rows = _fail_importing_preview(rows)
        result = {
            "preview": preview, "success": False, "committed": False,
            "rolled_back": True, "message": str(exc), "results": rows,
            "affected_functions": [], "affected_types": [],
            "affected_summary": {"referenced": 0, "reflowed": 0},
            **({"library_sources": dict(sorted(trace["library_sources"].items()))}
               if trace.get("library_sources") else {}),
        }
    return result


def _finalize_import_outcome(ctx, trace, result, preview: bool, exclusive_scope):
    # Irreversible deletion returns before initializing this guard. Its own
    # result/dirty ledger must survive even when no further lock can be taken.
    if trace.get("baseline") is None:
        return result
    with exclusive_scope():
        _collect_import_outcome(trace)
    if trace["parsed"] or trace["library_imports"]:
        _report_library_imports(trace["bv"], trace["baseline"], result["results"])
    if trace["library_imports"]:
        if preview or not result.get("committed"):
            result["kept_library_imports"] = sorted(trace["library_imports"])
        if trace["unattributed"]:
            result["unattributed_library_imports"] = sorted(trace["unattributed"])
        if trace.get("library_sources"):
            # #1012 item 2: the library each import came from, by its type id.
            result["library_sources"] = dict(sorted(trace["library_sources"].items()))
        if preview:
            explanation = (result.get("message") or "") if not result.get("success") else (
                "Preview changes were reverted." if result.get("rolled_back")
                else "Preview completed without committing." if "rolled_back" not in result
                else "Preview restoration did not complete.")
            result["results"] = _fail_importing_preview(result["results"])
            result["results"] = _restamp_reverted_siblings(
                ctx, result["results"], bool(result.get("rolled_back")))
            result["success"] = False
            result["status"] = "unsupported"
            result["message"] = (
                f"{explanation} Cannot --preview retained new types: types "
                "stay in the view (a rollback cannot remove them).").strip()
            if trace["unattributed"]:
                result["message"] += (
                    " No operation attribution is available for: "
                    + ", ".join(sorted(trace["unattributed"])) + ".")
    return result


def _apply_batch(ctx, selector: str | None, preview: bool, operations: list[dict[str, Any]], *,
                 exclusive=None):
    """Apply, verify, and (on failure or --preview) revert a batch of mutations.

    #628 lock split: ``exclusive`` is the caller's EXCLUSIVE target-lock scope -- a
    context-manager FACTORY (``bridge._mutation`` passes ``self._target_lock.write``).
    The caller holds the bridge write GATE for the whole call, which is what
    serializes writers (no mutation / py_exec / close / save / refresh / go_rename
    can interleave); the exclusive lock is taken here only around the phases that
    mutate or snapshot BN state. The post-apply ``update_analysis_and_wait()``
    deliberately runs OUTSIDE it, under the gate alone, so concurrent reads
    (doctor, target info, function list, ...) stay live instead of starving for a
    multi-minute reanalysis -- parity with load_binary (#99), refresh (#321) and
    go_rename (#365). #780 extends that to the per-op reanalysis a function_create
    op used to run inside the apply loop: it is deferred to this post-apply pass, so
    an N-create batch reanalyzes once, never inside the exclusive lock -- except
    that a create FOLLOWED BY another create is analyzed before the next one
    applies (gate-only, between two exclusive apply windows), because the next
    create's interior guard needs its ranges (#927 P2). The
    revert/settle paths stay exclusive on purpose: a reader
    must never observe a half-reverted view. The #479 quick-load refusal is
    unchanged. ``exclusive=None`` means "no exclusive scope" (a no-op), so direct
    engine calls and tests keep working. Every return path produces exactly the
    same result dict as before -- no schema, status, message or ordering change.
    """
    from . import create_comments  # one-way import direction, see _op_function_create

    exclusive_scope = exclusive if exclusive is not None else contextlib.nullcontext
    if not operations:
        raise ValueError("Batch operation list is empty")

    # Normalize struct field op aliases up front so the pre-apply snapshot pass
    # (_guess_affected_functions / _capture_type_snapshots) resolves the struct
    # under the same key the apply will, not just _apply_operation. (M12)
    # #825: rebind rather than mutate -- the normalized list is this function's
    # own, so the caller's manifest dicts are left exactly as they were sent.
    operations = [_normalize_struct_alias(op) for op in operations]
    # #652: reject a same-key-twice manifest BEFORE touching the view.
    _reject_duplicate_write_keys(ctx, operations)
    deletions = [op for op in operations
                 if isinstance(op, dict) and op.get("op") == "function_delete"]
    if deletions:
        with exclusive_scope():
            bv = ctx._resolve_view(selector)
            refuse_quick_view(bv, requested={"op": "function_delete"})
            if preview or len(operations) != 1:
                raise OperationFailure(
                    "unsupported",
                    "function delete has no clean undo: remove_user_function leaves a "
                    "persistent suppression override. --preview and combining deletion "
                    "with other batch operations are refused before writing. Apply one "
                    "function_delete operation without --preview, then read back the outcome.",
                    requested={"op": "function_delete"},
                    observed={"request_applied": False},
                )
            _validate_operation_request(ctx, deletions[0], index=0)
        return create_comments._function_delete(
            ctx, bv, deletions[0]["address"], exclusive=exclusive_scope)


    with exclusive_scope():
        bv = ctx._resolve_view(selector)
        # #479: every write op except the standalone function_create routes through
        # here, and both the success path (bv.update_analysis_and_wait() below) and the
        # revert paths reanalyze. On a --quick view that runs the full deferred analysis
        # as a side effect of a mutation -- a multi-minute stall the caller never asked
        # for -- so refuse the whole batch fast (including --preview, whose promise is a
        # bounded probe) and point at `bn refresh`. #628 shortened the exclusive window
        # the reanalysis runs in; it does NOT change this refusal. Subsumes and
        # generalizes the per-op function_create guard to bn type/comment/rename/proto/
        # struct-field/batch-apply on a quick view.
        refuse_quick_view(bv, requested={"op": "mutation", "operations": len(operations)})
        # #993 item 2: taken before anything parses a declaration -- the
        # pre-apply snapshot below already does -- for the live batch's
        # library-import report and every revert path's dangling-name cleanup.
        library_baseline = _declare_batch_type_names(bv, operations)
        trace = _IMPORT_TRACE.get()
        if trace is not None:
            if "baseline" not in trace:
                trace.update(bv=bv, baseline=library_baseline)
                _start_import_observer(bv, trace)
            trace["operations"] = operations
        # #650: validate EVERY op's shape before applying ANY. The batch is atomic, so a
        # guessed op or field name in op 13 previously rolled back 12 good ops -- and with
        # no argparse layer on this path, guessing is the norm, not the exception. It runs
        # right after the --quick refusal (target state) and BEFORE the #995/#630
        # preflights below, which read op fields: run first, a misspelled field there
        # surfaced as `internal error: KeyError` instead of its field-name hint (#999
        # review). Still strictly before the first mutation, which is what matters.
        for index, op in enumerate(operations):
            if isinstance(op, dict):
                _validate_operation_request(ctx, op, index=index)
        # A changing AUTO prototype would pin has_user_type irreversibly.
        # Preflight checks only the unchanged prefix; after a rename/create or
        # declaration, each target is resolved against actual preview state.
        # Skip irreversible prototype writes and their dependent locals, but
        # retain the pristine refusal when the whole batch would be skipped.
        skipped = _unpreviewable_operations(ctx, bv, operations) if preview else {}
        if skipped and len(skipped) == len(operations):
            unrevertible = [label for _, label in
                            _unrevertible_preview_prototypes(ctx, bv, operations)]
            names = ", ".join(unrevertible)
            if trace is not None:
                trace["failed_op"] = id(operations[min(skipped)])
            raise OperationFailure(
                "unsupported",
                f"Cannot --preview a prototype change on {names}: setting a "
                "prototype pins the function's has_user_type flag and Binary "
                "Ninja exposes no API to clear it, so the preview could not be "
                "cleanly reverted (it would leave the view modified). Re-run "
                "without --preview to apply the change, or preview a function "
                "that already has a user-defined prototype.",
                requested={"op": "set_prototype", "functions": unrevertible},
            )
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
        # #927 round 3: the functions a create may cut, as they are BEFORE op 0
        # -- the state every revert path below must return them to. A create
        # op's own restore repairs their ranges from a snapshot taken just
        # before IT ran, which inside a batch is an intermediate state; their
        # names are the journal's, and a rollback is not clean until both agree
        # with this baseline (`_split_hosts_at_baseline`).
        host_baseline = create_comments._split_host_baseline(ctx, bv, operations)
        state = bv.begin_undo_actions()
        if trace is not None:
            trace["transaction_started"] = True
    results = []
    # Explicit restores for local var ops, which BN's undo buffer can't
    # revert (see _run_local_restores). Replayed on every revert path.
    restores: list = []
    if library_baseline is not None and trace is not None and trace.get("parses_types"):
        # Registered first, so it runs last on every revert path, after the
        # journal has removed the batch's own types.
        restores.append(lambda: _clear_dangling_type_names(bv, library_baseline))
    skipped_prototypes: dict[int, int] = {}
    try:
        for index, op in enumerate(operations):
            # #780: function_create is the one op that reanalyzes while it
            # applies. The batch's own post-apply reanalysis (below, gate-only)
            # already covers every create, so the per-op one only added N full
            # analyses under the exclusive lock to an N-create batch, starving
            # every concurrent read. Defer it -- the same #628 deferral -- and
            # _verify_function_create does the post-analysis readback + #386
            # code guard afterwards.
            if preview:
                with exclusive_scope():
                    reason = _preview_operation_skip(
                        ctx, bv, op, index, skipped_prototypes)
                if reason is None:
                    skipped.pop(index, None)
                else:
                    skipped[index] = reason
            if index in skipped:
                results.append(_not_previewed_result(ctx, op, skipped[index]))
                continue
            with exclusive_scope():
                row = _apply_operation(ctx, bv, op, restores, defer_reanalysis=True)
                _refuse_rename_chain(row, results, index)
                results.append(row)
            if _later_create_needs_ranges(operations, index):
                # #927 P2: a create's interior guard reads the ranges of the
                # functions already there, and an EARLIER create in this batch
                # has none until something analyzes it -- so a second create
                # inside the first sailed past the guard and truncated it
                # behind `verified`. Analyze between two creates, under the
                # gate alone like the post-apply pass (the #780 discipline is
                # "never inside the exclusive lock", which this keeps), so the
                # next create's probe sees what the live view will.
                bv.update_analysis_and_wait()
    except OperationFailure as exc:
        exc = _explain_renamed_reference(operations[len(results)], results, exc)
        with exclusive_scope():
            failed_row = _operation_failure_result(ctx, operations[len(results)], exc)
            # Run BOTH revert steps unconditionally: an `and` would short-circuit
            # past the explicit restores when the undo revert fails, leaving
            # non-journaled local/prototype changes applied.
            undo_ok = _revert_undo_safely(ctx, bv, state)
            restore_ok = _run_local_restores(ctx, bv, restores)
            _settle_before_drift_restore(bv, var_before)
            drift_ok = _restore_local_var_drift(ctx, bv, var_before)
            # #927 round 3: the hosts of any create are back ONLY if they match
            # the pre-transaction baseline -- name included -- after the undo
            # AND the restores; a split host left renamed is a failed rollback.
            hosts_ok = create_comments._split_hosts_at_baseline(
                bv, host_baseline, what="the batch's creates")
            # #630: an already-applied set_prototype on an AUTO function leaves an
            # unclearable has_user_type override after revert. That is real residue,
            # not a clean revert, so it must fail and be disclosed here too -- the
            # apply-failure path previously reported status "reverted"/rolled_back
            # True with no disclosure at all.
            proto_residue = _prototype_user_type_residue(ctx, bv, results)
            values_reverted = undo_ok and restore_ok and drift_ok and hosts_ok
            reverted = values_reverted and not proto_residue
            if values_reverted and not proto_residue:
                message = "Rolled back before post-state verification because an operation failed to apply."
                result_note = "Rolled back before post-state verification."
                result_status = "reverted"
            elif values_reverted and proto_residue:
                message = (
                    "An operation failed to apply; the applied operations were rolled "
                    "back but BN cannot clear the has_user_type override an applied "
                    "prototype set, so the view is left modified."
                )
                result_note = "Rolled back, but an unclearable has_user_type override remains."
                result_status = "rollback_failed"
            else:
                message = (
                    "An operation failed to apply AND the rollback itself failed; "
                    "the view may be left partially modified."
                )
                result_note = "Rollback failed; this operation may still be applied."
                result_status = "rollback_failed"
            result = {
                "preview": preview,
                "success": False,
                "committed": False,
                "rolled_back": reverted,
                "message": message,
                "results": _mark_unverified_results(ctx, results, result_note, status=result_status)
                + [failed_row]
                + [_not_previewed_result(ctx, op, skipped[index]) if index in skipped
                   else _unattempted_results(ctx, [op])[0]
                   for index, op in enumerate(operations)
                   if index > len(results)],
                "affected_functions": [],
                "affected_types": [],
                "affected_summary": {
                    "referenced": _count_referenced_functions(ctx, bv, type_ops, fallback=0),
                    "reflowed": 0,
                },
            }
            if proto_residue:
                result["prototype_user_type_residue"] = True
            return _disclose_not_previewed(result, skipped, operations)

    # #628: the reanalysis after a successful apply is the long pole on a large
    # target and changes nothing we snapshot here -- it runs under the write gate
    # alone (still held by the caller, so no second writer can start), leaving
    # concurrent reads live. The post-analysis snapshot/verify/commit phase below
    # re-takes the exclusive lock: a reader must never observe half-verified or
    # half-reverted state.
    try:
        bv.update_analysis_and_wait()
        with exclusive_scope():
            after = _capture_function_snapshots(ctx, bv, affected)
            type_after = _capture_type_snapshots(ctx, bv, operations)
            diffs = _diff_snapshots(ctx, before, after)
            type_diffs = _diff_type_snapshots(ctx, type_before, type_after)
            verified_results = [result if result.get("status") == NOT_PREVIEWED
                                else _verify_operation(ctx, bv, result) for result in results]
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
            restored = True
            proto_residue = False
            if preview or failed:
                # Use the safe helper, not a bare bv.revert_undo_actions: a raise
                # from the undo revert must be absorbed into a structured result
                # (rolled_back=False), not escape as a generic bridge error, and
                # journaled-undo failure alone must flip `restored` even when the
                # local/drift restores succeed -- fold all three like the
                # apply-failure path does (#598).
                undo_ok = _revert_undo_safely(ctx, bv, state)
                # Targeted/prototype restores first, then mop up BN's propagation
                # onto aliased siblings; all must succeed for a clean revert.
                restore_ok = _run_local_restores(ctx, bv, restores)
                _settle_before_drift_restore(bv, var_before)
                drift_ok = _restore_local_var_drift(ctx, bv, var_before)
                # #927 round 3: a create's hosts must match the pre-transaction
                # baseline (name AND ranges) after the undo and every restore.
                hosts_ok = create_comments._split_hosts_at_baseline(
                    bv, host_baseline, what="the batch's creates")
                # #630: a proto set on a function that had NO prior user type pins
                # has_user_type, and neither BN's undo nor set_auto_type clears the
                # flag (verified live on BN 5.4) -- there is no BN API to clear it.
                # has_user_type is behaviorally meaningful (once set, analysis will
                # not re-derive the signature), NOT value-neutral metadata, so residue
                # means the rollback is INCOMPLETE -- fold it into `restored` so the
                # batch reports success:false / rolled_back:false and is marked dirty,
                # never a false clean rollback. A --preview never reaches here for the
                # AUTO-prototype case (it is refused before any mutation); this covers
                # the INVOLUNTARY rollback of a live batch (a sibling op failed
                # verification) that included a proto-set-on-AUTO op.
                #
                # `values_reverted` (the reversible ops) is tracked separately so the
                # per-op restamp still reports a rename/local that DID revert as
                # "reverted", not "rollback_failed".
                proto_residue = _prototype_user_type_residue(ctx, bv, results)
                values_reverted = undo_ok and restore_ok and drift_ok and hosts_ok
                restored = values_reverted and not proto_residue
            else:
                values_reverted = hosts_ok = True
                bv.commit_undo_actions(state)

            # Build the per-op output only once the revert outcome is known. On
            # success the canonical type layout lives in affected_types, so slim the
            # redundant copies out. On a rolled-back LIVE batch, re-stamp the ops that
            # verified cleanly -- the whole batch was undone, so they are no longer
            # "verified"/"noop" (#602). A preview keeps its verified evidence (its
            # non-commit is already advertised).
            if not failed:
                output_results = [_slim_type_result_for_output(item) for item in annotated_results]
            # Failed previews have reverted siblings just like failed live batches.
            else:
                output_results = _restamp_reverted_siblings(ctx, annotated_results, values_reverted)
            message = None
            if preview:
                # #937: a preview whose readback FAILED is still a preview -- but its
                # message must never claim verification succeeded. The lead names
                # the verification outcome; the clauses after it name the revert's.
                lead = (
                    "Preview verification failed, and" if failed else "Preview verified, but"
                )
                if not hosts_ok:
                    message = (
                        f"{lead} a function a create split is not back as "
                        "it was before the batch; the view may be left modified."
                    )
                elif not values_reverted:
                    message = (
                        f"{lead} reverting a non-journaled change "
                        "(local variable or prototype) failed; the view may be left modified."
                    )
                elif proto_residue:
                    # Value reverted, but has_user_type stuck (BN has no API to clear
                    # it), so the revert is INCOMPLETE and the view is left modified.
                    message = (
                        f"{lead} the function's has_user_type override could NOT be "
                        "cleared even though the prototype value was reverted (Binary "
                        "Ninja exposes no API to clear it); the view is left modified and "
                        "BN will no longer re-derive that signature."
                    )
                elif failed:
                    message = "Preview verification failed; the change was reverted."
                else:
                    message = "Preview verified and reverted."
            elif failed:
                if not hosts_ok:
                    message = (
                        "Live-session verification failed AND a function a create split "
                        "is not back as it was before the batch; the view may be left modified."
                    )
                elif not values_reverted:
                    message = (
                        "Live-session verification failed AND reverting a non-journaled change "
                        "(local variable or prototype) failed; the view may be left modified."
                    )
                elif proto_residue:
                    message = (
                        "Live-session verification failed; the prototype value was rolled "
                        "back but BN cannot clear the has_user_type override the applied "
                        "prototype set, so the view is left modified."
                    )
                else:
                    message = "Rolled back because live-session verification failed."
            else:
                message = "Applied and verified in the live Binary Ninja session."
            # A preview/rollback whose non-journaled restore failed OR that left an
            # unclearable has_user_type override behind did NOT cleanly revert -- that
            # is not a success, even if every operation verified. Automation keys off
            # `success`; `restored` is only False on the preview/failed paths.
            result = {
                "preview": preview,
                "success": (not failed) and restored,
                "committed": bool((not preview) and (not failed)),
                "message": message,
                "results": output_results,
                "affected_functions": diffs,
                "affected_types": type_diffs,
                "affected_summary": {"referenced": referenced, "reflowed": reflowed},
                # #652: `rolled_back` used to be present in every case EXCEPT success, so
                # a parser written against a preview or a failure -- the cases you develop
                # against -- raised KeyError on the happy path. Always emit it (False when
                # committed), matching the read-command contract where paging keys are
                # present regardless of outcome.
                "rolled_back": restored if (preview or failed) else False,
            }
            if proto_residue:
                # Structured disclosure of the unclearable has_user_type override, in
                # addition to success:false / rolled_back:false (#630).
                result["prototype_user_type_residue"] = True
            return _disclose_not_previewed(result, skipped, operations)
    except Exception as exc:
        with exclusive_scope():
            undo_ok = _revert_undo_safely(ctx, bv, state)
            restore_ok = _run_local_restores(ctx, bv, restores)
            _settle_before_drift_restore(bv, var_before)
            drift_ok = _restore_local_var_drift(ctx, bv, var_before)
            hosts_ok = create_comments._split_hosts_at_baseline(
                bv, host_baseline, what="the batch's creates")
            # #630 round 2: even when the value revert succeeds, an already-applied
            # set_prototype on an AUTO function leaves an unclearable has_user_type
            # override -- BN's undo and set_auto_type both restore the value but leave
            # the flag pinned, and there is no API to clear it. That residue leaves the
            # view MODIFIED, so disclose it on the raised error; the bridge shim keys
            # off the attribute to mark the view dirty, or `close` would read
            # bv.file.modified == false and wrongly report no unsaved state.
            proto_residue = _prototype_user_type_residue(ctx, bv, results)
            if not (undo_ok and restore_ok and drift_ok and hosts_ok):
                # #927 round 4: a failed revert on THIS path used to escape as a
                # bare RuntimeError -- no status, so the CLI exited 2 as if the
                # bridge had merely erred, and the bridge shim (which keys off the
                # residue attribute alone) never marked the view dirty, so `close`
                # read bv.file.modified == false over a host left renamed. Raise
                # the structured failure the other two revert paths report:
                # status `rollback_failed` (exit 3, `observed.dirty_after`), and
                # the shim marks the view dirty off the status.
                err = OperationFailure(
                    "rollback_failed",
                    f"{exc} (additionally, rollback failed; the view may be left partially modified)",
                    observed={"rolled_back": False, "dirty_after": True},
                )
                if proto_residue:
                    err.prototype_user_type_residue = True
                raise err from exc
            if proto_residue:
                # Keep the original exception's information; annotate it so the bridge
                # marks the view dirty. Isolate the attribute write from the re-raise so
                # an exception whose type rejects attribute assignment (rare builtins)
                # falls back to a wrapper rather than being mistaken for a set failure.
                try:
                    exc.prototype_user_type_residue = True
                except (AttributeError, TypeError):
                    err = RuntimeError(str(exc))
                    err.prototype_user_type_residue = True
                    raise err from exc
        raise



def _op_rename_symbol(ctx, bv, op: dict[str, Any]):
    kind = str(op.get("kind", "auto"))
    identifier = op["identifier"]
    # Keep the guard on direct handler calls as well as batch preflight (#923).
    # Reject before resolution or writes, without stripping or banning quotes.
    new_name = _require_nonempty_name(
        op["new_name"], requested=_operation_requested(ctx, op), reject_controls=True,
    )
    target = ctx._resolve_rename_target(bv, identifier, kind)
    requested = _operation_requested(ctx, op)
    # #1042 item 1: a rendered `data_<hex>`/`sub_<hex>` name resolved to the
    # address it spells; the row says so, so the rename is not read as having
    # found a symbol by that name.
    resolved = {"resolved_symbol": target["resolved_symbol"]} if target.get("resolved_symbol") else {}
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
            **resolved,
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
        **resolved,
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
        before_comment = str(getattr(fn, "comment", "") or "")
        if before_comment != comment:
            fn.comment = comment
        return {
            "op": "set_comment",
            "scope": "function_doc",
            "address": hex(int(fn.start)),
            "function": fn.name,
            "before_comment": before_comment,
            "requested": _operation_requested(ctx, op),
        }
    address = _parse_address(op["address"])
    _refuse_unmapped_mutation_address(
        ctx, bv, op, address, "no comment can be set there."
    )
    before_comment = bv.get_comment_at(address) or ""
    if before_comment != comment:
        bv.set_comment_at(address, comment)
    return {
        "op": "set_comment",
        "address": hex(address),
        "before_comment": before_comment,
        "requested": _operation_requested(ctx, op),
    }



def _resolve_tag_type_or_fail(ctx, bv, op, type_name: str):
    tt = bv.get_tag_type(type_name)
    if tt is None:
        available = ", ".join(sorted(bv.tag_types)) or "(none)"
        raise OperationFailure(
            "invalid_request",
            f"Unknown tag type {type_name!r}. Available types: {available}. "
            "Create a custom type with `bn tag type create`.",
            requested=_operation_requested(ctx, op),
        )
    return tt


def _find_new_tag(before_ids: set[str], tags) -> Any | None:
    """Both ``Function.add_tag`` and ``BinaryView.add_tag`` return ``None`` on
    real Binary Ninja (only the test fakes return the created Tag, as a test
    convenience) -- so the id needed for ``tag remove --id`` has to be
    recovered by diffing the relevant tag collection before/after the add.
    Discovered live (#see _op_tag_add): the old code called ``.id`` on the
    ``None`` return and crashed every real-BN tag add."""
    for t in tags:
        if str(t.id) not in before_ids:
            return t
    return None


def _op_tag_add(ctx, bv, op: dict[str, Any]):
    type_name = str(op["type"])
    data = str(op.get("data", ""))
    _resolve_tag_type_or_fail(ctx, bv, op, type_name)
    if op.get("function") and op.get("address"):
        raise OperationFailure(
            "invalid_request",
            "Pass a function OR an address, not both: they target different locations.",
            requested=_operation_requested(ctx, op),
        )
    if op.get("function") and op.get("force_data"):
        # --data-scope forces the data-level (bv.add_tag) path, which is
        # inherently address-based; a function tag has no address, so the two
        # are contradictory. Reject rather than silently dropping force_data.
        raise OperationFailure(
            "invalid_request",
            "--data-scope only applies to a tag at an address; it can't be "
            "combined with --function (a function tag has no address).",
            requested=_operation_requested(ctx, op),
        )
    requested = _operation_requested(ctx, op)

    # #779: an add whose (type, data) pair is ALREADY present is a no-op, and
    # BN's add_tag does not dedupe -- `_find_new_tag` exists precisely because
    # every call mints a fresh Tag with its own id, so running `bn tag add`
    # twice genuinely produced two identical tags and still reported `verified`
    # both times. Check before writing, exactly as `_op_set_comment` compares
    # `before_comment` before assigning, and hand the verifier a `pre_existing`
    # flag so it can report `noop` instead of claiming a write that did not
    # happen. Skipping the write is also what keeps revert honest: tag ops roll
    # back through BN's undo journal, so an add that never happened records
    # nothing to undo and a pre-existing tag cannot be destroyed by reverting
    # someone else's preview.
    def _existing(tags):
        return next((t for t in tags if _tag_matches(t, type_name, data)), None)

    if op.get("function"):
        fn = ctx._find_function(bv, op["function"])
        tag = _existing(fn.get_function_tags(auto=False))
        pre_existing = tag is not None
        if not pre_existing:
            before_ids = {str(t.id) for t in fn.get_function_tags(auto=False)}
            fn.add_tag(type_name, data, None)
            tag = _find_new_tag(before_ids, fn.get_function_tags(auto=False))
        return {"op": "tag_add", "tag_type": type_name, "data": data, "scope": "function",
                "function": fn.name, "address": hex(int(fn.start)),
                "tag_id": str(tag.id) if tag is not None else None,
                "pre_existing": pre_existing,
                "message": (f"Function tag {type_name!r} already present on {fn.name}."
                            if pre_existing else
                            f"Added function tag {type_name!r} to {fn.name}."),
                "requested": requested}

    addr = _parse_address(op["address"])
    force_data = bool(op.get("force_data"))
    funcs = bv.get_functions_containing(addr)
    if force_data or not funcs:
        tag = _existing(bv.get_tags_at(addr, auto=False))
        pre_existing = tag is not None
        if not pre_existing:
            before_ids = {str(t.id) for t in bv.get_tags_at(addr, auto=False)}
            bv.add_tag(addr, type_name, data)
            tag = _find_new_tag(before_ids, bv.get_tags_at(addr, auto=False))
        note = "" if force_data else " (no function contains this address)"
        return {"op": "tag_add", "tag_type": type_name, "data": data, "scope": "data",
                "address": hex(addr), "function": None,
                "tag_id": str(tag.id) if tag is not None else None,
                "pre_existing": pre_existing,
                "message": (f"Data tag {type_name!r} already present at {hex(addr)}."
                            if pre_existing else
                            f"Added data tag {type_name!r} at {hex(addr)}{note}."),
                "requested": requested}
    fn = funcs[0]
    tag = _existing(fn.get_tags_at(addr, auto=False))
    pre_existing = tag is not None
    if not pre_existing:
        before_ids = {str(t.id) for t in fn.get_tags_at(addr, auto=False)}
        fn.add_tag(type_name, data, addr)
        tag = _find_new_tag(before_ids, fn.get_tags_at(addr, auto=False))
    return {"op": "tag_add", "tag_type": type_name, "data": data, "scope": "address",
            "address": hex(addr), "function": fn.name,
            # Record the EXACT function the tag was added to (its stable start
            # address) so verification checks that function, not "any function
            # containing addr" -- overlapping functions could otherwise let a
            # match on a DIFFERENT function pass verification (#630).
            "function_start": hex(int(fn.start)),
            "tag_id": str(tag.id) if tag is not None else None,
            "pre_existing": pre_existing,
            "message": (f"Tag {type_name!r} already present at {hex(addr)} in {fn.name}."
                        if pre_existing else
                        f"Added tag {type_name!r} at {hex(addr)} in {fn.name}."),
            "requested": requested}


def _tag_matches(tag, type_name, data) -> bool:
    if type_name is not None and tag.type.name != type_name:
        return False
    if data is not None and tag.data != data:
        return False
    return True


def _op_tag_remove(ctx, bv, op: dict[str, Any]):
    tag_id = op.get("tag_id")
    type_name = op.get("type")
    data = op.get("data")
    requested = _operation_requested(ctx, op)
    if not tag_id and not type_name:
        raise OperationFailure(
            "invalid_request",
            "tag remove needs --id, or a location plus --type.",
            requested=requested,
        )
    # A tag id is a UUID (the `id` column of `bn tag list`). Reject a malformed
    # id up front -- parity with `tag add`'s bad-type/bad-address errors -- so a
    # typo'd id is a clear invalid_request instead of a silent no-op that reads
    # like "there was no such tag". A well-formed but nonexistent id still falls
    # through to the normal no-match `noop` (below).
    if tag_id:
        try:
            uuid.UUID(str(tag_id))
        except (ValueError, AttributeError, TypeError):
            raise OperationFailure(
                "invalid_request",
                f"{tag_id!r} is not a valid tag id; expected a UUID from the "
                "`id` column of `bn tag list`.",
                requested=requested,
            )
    removed: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []  # for verification: (scope, address, id)

    def match(tag) -> bool:
        if tag_id:
            return str(tag.id) == str(tag_id)
        return _tag_matches(tag, type_name, data)

    if op.get("function"):
        fn = ctx._find_function(bv, op["function"])
        for t in list(fn.get_function_tags(auto=False)):
            if match(t):
                fn.remove_user_function_tag(t)
                removed.append({"type": t.type.name, "data": t.data, "scope": "function"})
                targets.append({"scope": "function", "function": fn.name,
                                "address": hex(int(fn.start)), "id": str(t.id)})
    elif op.get("address") is not None:
        addr = _parse_address(op["address"])
        force_data = bool(op.get("force_data"))
        if not force_data:
            for fn in bv.get_functions_containing(addr):
                for t in list(fn.get_tags_at(addr, auto=False)):
                    if match(t):
                        fn.remove_user_address_tag(addr, t)
                        removed.append({"type": t.type.name, "data": t.data, "scope": "address"})
                        targets.append({"scope": "address", "address": hex(addr), "id": str(t.id)})
        for t in list(bv.get_tags_at(addr, auto=False)):
            if match(t):
                bv.remove_user_data_tag(addr, t)
                removed.append({"type": t.type.name, "data": t.data, "scope": "data"})
                targets.append({"scope": "data", "address": hex(addr), "id": str(t.id)})
    elif tag_id:
        # id-only: search every scope. bv.get_tags() only covers view-level
        # DATA tags -- it never surfaces a function's ADDRESS-scope tags
        # (fn.tags), so those need their own sweep, mirroring
        # read_tags._collect_tags's address-tag sweep exactly. Track removed
        # ids so a tag id that (in principle) turns up more than once across
        # sweeps isn't double-removed/double-counted.
        removed_ids: set[str] = set()
        for addr, t in list(bv.get_tags(auto=False)):
            if str(t.id) == str(tag_id) and str(t.id) not in removed_ids:
                bv.remove_user_data_tag(addr, t)
                removed.append({"type": t.type.name, "data": t.data, "scope": "data"})
                targets.append({"scope": "data", "address": hex(addr), "id": str(t.id)})
                removed_ids.add(str(t.id))
        for fn in list(bv.functions):
            for t in list(fn.get_function_tags(auto=False)):
                if str(t.id) == str(tag_id) and str(t.id) not in removed_ids:
                    fn.remove_user_function_tag(t)
                    removed.append({"type": t.type.name, "data": t.data, "scope": "function"})
                    targets.append({"scope": "function", "function": fn.name,
                                    "address": hex(int(fn.start)), "id": str(t.id)})
                    removed_ids.add(str(t.id))
            for _arch, addr, t in list(fn.tags):
                if str(t.id) == str(tag_id) and str(t.id) not in removed_ids:
                    fn.remove_user_address_tag(addr, t)
                    removed.append({"type": t.type.name, "data": t.data, "scope": "address"})
                    targets.append({"scope": "address", "address": hex(addr), "id": str(t.id)})
                    removed_ids.add(str(t.id))
    else:
        raise OperationFailure(
            "invalid_request",
            "tag remove needs --id, an address, or --function.",
            requested=requested,
        )

    return {"op": "tag_remove", "removed": len(removed), "removed_tags": removed,
            "targets": targets, "requested": requested,
            "message": f"Removed {len(removed)} tag(s)."}


def _op_tag_type_create(ctx, bv, op: dict[str, Any]):
    name = str(op["name"])
    icon = str(op["icon"])
    requested = _operation_requested(ctx, op)
    existed = bv.get_tag_type(name) is not None
    bv.create_tag_type(name, icon)  # BN no-ops on an existing name
    return {"op": "tag_type_create", "name": name, "icon": icon,
            "existed": existed, "requested": requested,
            "message": (f"Tag type {name!r} already exists." if existed
                        else f"Created tag type {name!r}.")}


def _op_tag_type_remove(ctx, bv, op: dict[str, Any]):
    name = str(op["name"])
    requested = _operation_requested(ctx, op)
    if name in _BUILTIN_TAG_TYPE_NAMES:
        raise OperationFailure(
            "invalid_request",
            f"Refusing to remove built-in tag type {name!r}. Only custom tag "
            "types can be removed.",
            requested=requested,
        )
    existed = bv.get_tag_type(name) is not None
    if existed:
        bv.remove_tag_type(name)
    return {"op": "tag_type_remove", "name": name, "existed": existed,
            "requested": requested,
            "message": (f"Removed tag type {name!r}." if existed
                        else f"Tag type {name!r} does not exist.")}


def _op_delete_comment(ctx, bv, op: dict[str, Any]):
    if op.get("function") and op.get("address"):
        raise OperationFailure(
            "invalid_request",
            "Pass function OR address, not both: they target different locations.",
            requested=_operation_requested(ctx, op),
        )
    if op.get("function"):
        fn = ctx._find_function(bv, op["function"])
        before_comment = str(getattr(fn, "comment", "") or "")
        if before_comment:
            fn.comment = ""
        return {
            "op": "delete_comment",
            "scope": "function_doc",
            "address": hex(int(fn.start)),
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



# The C tag keywords a type string may carry before a named aggregate
# (`struct ns::Foo*`). `struct` and `class` are interchangeable in C++.
_TAG_KEYWORDS = ("struct", "class", "union", "enum")


def _type_tag_kind(type_obj) -> str | None:
    """The tag keyword a named BN type is spelled with (`struct`/`union`/`enum`;
    `class` for a C++ class structure), or None for a type that is not an
    aggregate tag (a typedef, a primitive, a function type)."""
    tc = _type_class_name(type_obj)
    if "Enumeration" in tc:
        return "enum"
    if "Structure" not in tc:
        return None
    kind = getattr(type_obj, "type", None)
    kind_name = str(getattr(kind, "name", kind) or "")
    if "Union" in kind_name:
        return "union"
    if "Class" in kind_name:
        return "class"
    return "struct"


def _tag_keyword_matches(keyword: str, type_obj) -> bool:
    kind = _type_tag_kind(type_obj)
    if keyword in ("struct", "class"):
        return kind in ("struct", "class")
    return kind == keyword


def _resolve_named_type_string(bv, text: str):
    """Build a Type for a string that names an existing user/named type,
    optionally with trailing ``*`` pointer levels, or None when the base name is
    not a known type.

    BN's C type-string parser rejects ``::``-qualified names ("use of undeclared
    identifier 'ns'"), so a recovered C++ class type can't be applied by name
    even when it's defined. This rebuilds the type directly from
    ``get_type_by_name`` + a named-type reference, so ``ns::demo::Foo`` /
    ``ns::demo::Foo*`` resolve without a flat-name alias workaround (#200).
    A leading tag keyword (``struct ns::Foo*``) and backtick quoting
    (``struct `ns::Foo`*``) are accepted too: they are how BN itself spells
    these types, so an agent copying a decl from ``types show`` reaches here
    with them (#962). A tag keyword must agree with the type it names (``union
    Foo`` never resolves a struct ``Foo``), so the fallback cannot widen what a
    spelling means."""
    base = str(text).strip()
    depth = 0
    while base.endswith("*"):
        base = base[:-1].rstrip()
        depth += 1
    # A const/volatile qualifier (leading or trailing) is common on a
    # verbatim-from-the-decompiler C++ field type (`ns::Class const*`,
    # `const ns::Class*`). The #200 fallback stripped only `*`, so the qualifier
    # made the base-name lookup miss and the whole parse failed-closed (#389).
    # Strip leading/trailing cv-qualifiers (word-boundary only, so a template
    # name's interior spacing is preserved) and resolve the unqualified type:
    # a cv-qualifier is layout- and indirection-preserving, so dropping it is
    # safe -- unlike the bitfield / dropped-pointer coercions #367 rejects.
    # Only the common west-const pointer-to-const forms (`Foo const*`,
    # `const Foo*`) are normalized; the rarer east-const-on-the-pointer forms
    # (`Foo *const`) leave a stray mid-string `*` after the qualifier strip and
    # fail closed (clean "declare it first"), never silently mis-resolve.
    tag_keyword = None
    changed = True
    while changed:
        changed = False
        for q in ("const", "volatile"):
            if base == q:
                base = ""
                changed = True
            elif base.startswith(q + " "):
                base = base[len(q) + 1:].lstrip()
                changed = True
            elif base.endswith(" " + q):
                base = base[: -(len(q) + 1)].rstrip()
                changed = True
        for keyword in _TAG_KEYWORDS:
            if tag_keyword is None and base.startswith(keyword + " "):
                base = base[len(keyword) + 1:].lstrip()
                tag_keyword = keyword
                changed = True
    if len(base) > 2 and base.startswith("`") and base.endswith("`"):
        base = base[1:-1]
    if not base or "`" in base:
        return None
    named = _lookup_named_type(bv, base)
    if named is None:
        return None
    if tag_keyword is not None and not _tag_keyword_matches(tag_keyword, named):
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


# The parser diagnostics that name a type the declaration could not resolve:
# a bare unknown name (`pthread_mutex_t` where the view only has `union
# pthread_mutex_t`), the first component of a `::`-qualified name the C parser
# reads as an identifier (`demo` of `demo::Widget::VTable`), and a tagged
# reference to a structure the view does not have under that spelling.
_UNRESOLVED_TYPE_DIAGNOSTICS = (
    re.compile(r"unknown type name '([^']+)'"),
    re.compile(r"use of undeclared identifier '([^']+)'"),
    re.compile(r"Referencing unknown (?:structure|union|enum|enumeration) type:? `?([^\s`]+)`?"),
)


def _type_spelling(name: str, type_obj) -> str:
    """How a type-string must spell the existing type *name*: its tag keyword
    when it is an aggregate (C does not make a tag name a type name), and
    backticks around a ``::``-qualified name (BN's parser reads a bare one as a
    scope expression)."""
    shown = f"`{name}`" if "::" in name else name
    tag = _type_tag_kind(type_obj)
    return f"{tag} {shown}" if tag else shown


def _type_spelling_hint(bv, text: str, detail: str, *, limit: int = 5) -> str:
    """#962: a ``" Existing type(s): ..."`` sentence naming the exact spelling of
    the view's types that the failed parse most plausibly meant, or "".

    The parser's message names only a token (`unknown type name 'X'`, or for a
    qualified name its FIRST component); the full name is recovered from the
    text that was parsed, then looked up -- exactly, else by its last component,
    else fuzzily -- in ``bv.types``, the same population ``types --query``
    searches, so every suggestion is a spelling that resolves."""
    tokens: list[str] = []
    for pattern in _UNRESOLVED_TYPE_DIAGNOSTICS:
        tokens.extend(pattern.findall(detail))
    if not tokens:
        return ""
    try:
        population = {str(name): type_obj for name, type_obj in list(bv.types.items())}
    except Exception:
        return ""
    if not population:
        return ""
    suggestions: list[str] = []
    # A displayed prototype may name the same namespace in its function name
    # and a parameter type. Only types should participate in spelling hints.
    text = str(text)
    span = _qualified_declarator_span(text)
    if span is None:
        head, sep, _ = text.partition("(")
        quoted = re.search(r"`[^`]*::[^`]*`\s*$", head) if sep else None
        if quoted is not None and head[:quoted.start()].strip():
            span = quoted.span()
    if span is not None:
        start, end = span
        text = text[:start] + " " + text[end:]
    for token in dict.fromkeys(tokens):
        # Widen a leading component back to the qualified name it started.
        full = token
        widened = re.search(r"(?<![\w:])" + re.escape(token) + r"(?:::[A-Za-z_~]\w*)+", str(text))
        if widened:
            full = widened.group(0).strip()
        leaf = _split_qualified_name(full)[-1]
        matches = [name for name in (full, token) if name in population]
        if not matches:
            matches = sorted(name for name in population
                             if _split_qualified_name(name)[-1] == leaf)
        if not matches:
            matches = difflib.get_close_matches(full, list(population), n=3, cutoff=0.75)
        suggestions.extend(_type_spelling(name, population[name]) for name in matches)
    ordered = list(dict.fromkeys(suggestions))
    if not ordered:
        return ""
    shown = ", ".join(ordered[:limit]) + (f" (+{len(ordered) - limit} more)" if len(ordered) > limit else "")
    return (f" Closest existing type(s), spelled the way a type string must name them: {shown}. "
            "A struct/union/enum needs its tag keyword, and a `::`-qualified name "
            "needs backticks (struct `ns::Foo`*).")


@_track_operation
def _parse_type_or_hint(ctx, bv, op: dict[str, Any], type_text: Any, *, label: str,
                        shown: str | None = None, column_of=None):
    """``bv.parse_type_string`` with a friendly, actionable error.

    A prototype/type that references a struct or type not defined yet fails to
    parse. Surface a clear ``invalid_request`` with a next step instead of the
    generic ``unsupported: <ExceptionClass>: ...`` the catch-all in
    ``_apply_operation`` would otherwise emit -- which leaks the Python
    exception class name and gives no way forward (#122). *shown* is the text
    the error quotes when *type_text* is a rewrite of what the caller typed, and
    *column_of* maps a 1-based column of *type_text* to the same spot in it."""
    trace = _IMPORT_TRACE.get()
    if trace is not None:
        trace["parsed"].add(id(op))
    try:
        return bv.parse_type_string(str(type_text), **_parse_import_kwargs())
    except OperationFailure:
        raise
    except Exception as exc:
        # An optional fallback failure must not replace the original parse
        # error with an internal exception (#1013 items 6-7).
        try:
            resolved = _resolve_named_type_string(bv, str(type_text))
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved, None
        refusal = _preview_import_refusal(
            bv, str(type_text), exc, requested=_operation_requested(ctx, op))
        if refusal is not None:
            raise refusal from exc
        # Collapse BN's multi-line parser output ("error: ...\n1 error
        # generated.") into a single clean clause.
        detail = " ".join(str(exc).split()).rstrip(".")
        if column_of is not None:
            # BN's `file:line:col` positions count the parsed text; translate
            # them to the text the error quotes.
            detail = re.sub(r"(\S+?:)(\d+):(\d+)",
                            lambda m: (f"{m.group(1)}{m.group(2)}:"
                                       f"{column_of(int(m.group(2)), int(m.group(3)))}"),
                            detail)
        raise OperationFailure(
            "invalid_request",
            f"could not parse {label} {shown if shown is not None else str(type_text)!r}"
            + (f": {detail}" if detail else "")
            + "."
            + _parse_failure_hint(bv, str(type_text), detail),
            requested=_operation_requested(ctx, op),
        ) from exc


def _parse_failure_hint(bv, text: str, detail: str) -> str:
    """The next step for a type string BN's parser refused, most specific first.

    #999 item 2: two failures are about the spelling, not a missing type, and
    the generic "declare it first" sent agents re-declaring types they already
    had: ``this`` is a C++ keyword BN's parser rejects as a parameter name, and a
    ``::``-qualified function name outside backticks reads as an out-of-line
    member definition (one the declarator quoting cannot reach, such as a
    partly backticked name).
    The #962 existing-type spelling still wins over the generic ``::`` advice
    when it can name the type the string meant."""
    if "'this' is a keyword" in detail:
        identifiers = set(re.findall(r"[A-Za-z_]\w*", text))
        replacement = "self"
        suffix = 0
        while replacement in identifiers:
            suffix += 1
            replacement = f"self_{suffix}"
        return (" `this` is a keyword the parser rejects as a parameter name. "
                f"Rename that parameter to `{replacement}`.")
    spelling = _type_spelling_hint(bv, text, detail)
    if spelling:
        return spelling
    if "out-of-line declaration" in detail:
        return (" A `::`-qualified function name must be quoted with backticks as "
                "a whole (int `ns::Foo::run`(int x)); the name in a prototype is "
                "ignored, so it can also be omitted.")
    return (" If it references a struct or type that isn't defined yet, "
            "declare it first (e.g. with `bn types declare`), then retry.")


# The declarator name of a prototype: the identifier right before its first `(`,
# with something (the return type, ending in a space, `*` or `&`) in front of it.
_PROTOTYPE_DECLARATOR = re.compile(
    r"^(?P<head>.*[\s*&])(?P<name>[A-Za-z_~][\w~]*(?:::[A-Za-z_~][\w~]*)+)\s*$", re.DOTALL)


def _qualified_declarator_span(text: str) -> tuple[int, int] | None:
    """``(start, end)`` of an unquoted ``::``-qualified function name in a
    prototype, or None when there is none to quote (see below)."""
    head, sep, _ = text.partition("(")
    if not sep:
        return None
    match = _PROTOTYPE_DECLARATOR.match(head)
    if match is None or not match.group("head").strip():
        return None
    return match.start("name"), match.end("name")


def _quote_qualified_declarator(text: str) -> str:
    """#999 item 2: backtick-quote a ``::``-qualified function name in a
    prototype (``int ns::Cls::run(int x)``). BN's parser reads it as an
    out-of-line member definition and refuses it, yet the name is ignored --
    ``set_user_type`` sets a type, never a symbol -- so quoting it changes nothing
    but whether the string parses. A name already in backticks, or a prototype
    with no return type in front of the qualified token, is left alone."""
    span = _qualified_declarator_span(text)
    if span is None:
        return text
    start, end = span
    return f"{text[:start]}`{text[start:end]}`{text[end:]}"


def _parse_prototype(ctx, bv, op: dict[str, Any]):
    """The prototype a ``set_prototype`` op requests, parsed to a ``Type``.
    Its parse error quotes the prototype as typed, not the quoted rewrite, and
    BN's column numbers are translated back to that text (#999 review)."""
    trace = _IMPORT_TRACE.get()
    if trace is not None:
        trace["parsed"].add(id(op))
    typed = str(op.get("prototype"))
    span = _qualified_declarator_span(typed)
    column_of = None
    if span is not None:
        start, end = span
        # The name never spans a newline, so both backticks land on its line;
        # positions are taken relative to that line's start.
        name_line = typed.count("\n", 0, start) + 1
        line_start = typed.rfind("\n", 0, start) + 1
        start, end = start - line_start, end - line_start

        def column_of(line: int, column: int) -> int:
            # The rewrite adds one backtick before index `start` and one after
            # index `end - 1` of the name's line; other lines are unchanged.
            position = column - 1
            if line != name_line:
                return column
            if position > end + 1:
                position -= 2
            elif position > start:
                position -= 1
            return position + 1

    expected_type, _ = _parse_type_or_hint(
        ctx, bv, op, _quote_qualified_declarator(typed), label="prototype", shown=typed,
        column_of=column_of)
    return expected_type


def _parse_concrete_type(ctx, bv, op: dict[str, Any], type_text: Any, *, label: str):
    """``_parse_type_or_hint`` plus the two coercion guards a concrete variable /
    field type needs so a SILENTLY-coerced type is rejected (``invalid_request``)
    instead of applied and reported ``verified`` against itself (#367). BN's
    headless C parser, confirmed on real BN:

    * drops a ``:N`` bitfield width and mis-lays-out the members as full-width
      integers (the same defect #322 guards for ``types declare``); and
    * drops the trailing ``*`` of a pointer-to-inline-anonymous-aggregate
      (``struct{...}*`` parses to a struct VALUE, losing pointer-ness).

    The live verify path compares the applied type against itself, so neither is
    caught downstream -- they must be rejected here. Named-type pointers
    (``Foo*``) keep pointer-ness and pass; only the inline-anon drop is caught."""
    text = str(type_text)
    if _declaration_has_bitfield(text):
        raise OperationFailure(
            "invalid_request",
            f"{label} {text!r} uses C bitfield syntax (`:N`), which BN's headless "
            "parser silently drops and mis-lays-out as full-width members (#322/"
            "#367). Declare the type with explicit-width fields instead.",
            requested=_operation_requested(ctx, op),
        )
    parsed, hint = _parse_type_or_hint(ctx, bv, op, type_text, label=label)
    if text.rstrip().endswith("*") and "Pointer" not in _type_class_name(parsed):
        raise OperationFailure(
            "invalid_request",
            f"{label} {text!r} requests a pointer, but BN's parser dropped the "
            "pointer-ness (it does so for inline anonymous aggregates like "
            "`struct{...}*`). Name the type first (`bn types declare`), then use "
            "`<Name>*`.",
            requested=_operation_requested(ctx, op),
        )
    return parsed, hint


def _function_can_return(fn, text: str | None = None) -> bool | None:
    """BN's effective ``can_return`` as a plain bool (the getter returns a
    BoolWithConfidence), else whether *text* lacks the ``__noreturn``
    attribute, else None when neither is readable."""
    value = getattr(fn, "can_return", None)
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        return raw
    if text is not None:
        return re.search(r"\b__noreturn\b", text) is None
    return None


def _op_set_prototype(ctx, bv, op: dict[str, Any], restores: list | None = None):
    fn = ctx._find_function(bv, op["identifier"])
    expected_type = _parse_prototype(ctx, bv, op)
    before_prototype = str(fn.type)
    before_type_obj = fn.type
    # #582: whether the function ALREADY carried a user prototype before we
    # touched it. set_user_type pins has_user_type; on a preview/rollback we
    # must only re-assert that pin when it was genuinely there beforehand.
    before_has_user_type = bool(getattr(fn, "has_user_type", False))
    expected_prototype = str(expected_type)
    # #966: `__noreturn` in a prototype is how `can_return` is set. Both values
    # are kept so verification can judge the attribute the request changes.
    before_can_return = _function_can_return(fn, before_prototype)
    expected_can_return = _function_can_return(expected_type, expected_prototype)
    if before_prototype != expected_prototype:
        # Function.set_user_type is NOT journaled by BN's undo buffer (same
        # class as create_user_var for locals), so revert_undo_actions is a
        # silent no-op for the prototype VALUE -- without an explicit restore,
        # --preview and rollback-on-failure would leave the previewed prototype
        # committed to the view (#51). Register the restore before mutating.
        _register_prototype_restore(ctx,
            bv, restores, fn=fn,
            before_prototype=before_prototype, before_type_obj=before_type_obj,
            had_user_type=before_has_user_type,
        )
        # #999 item 1: hand BN the parsed Type, never its string. `str(Type)`
        # drops the backticks a `::`-qualified type needs (`struct ns::Cls*`),
        # so re-parsing it failed on a prototype that had just parsed fine.
        fn.set_user_type(expected_type)
    return {
        "op": "set_prototype",
        "function": fn.name,
        "address": hex(fn.start),
        "before_prototype": before_prototype,
        "before_has_user_type": before_has_user_type,
        "before_can_return": before_can_return,
        "expected_prototype": expected_prototype,
        "expected_can_return": expected_can_return,
        "requested": _operation_requested(ctx, op),
    }



def _register_prototype_restore(ctx, bv, restores, *, fn, before_prototype, before_type_obj,
                                had_user_type):
    """Capture how to put a function prototype back on revert. Mirrors
    :meth:`_register_local_restore`: ``set_user_type`` is not journaled by BN's
    undo buffer, so the preview/rollback paths replay this explicitly. Re-resolves
    the function fresh at restore time by its start address.

    ``had_user_type`` records whether the function already carried a user
    prototype before apply (#582). When it did NOT, the restore must put the
    value back WITHOUT re-pinning ``has_user_type`` -- ``Function.type``'s setter
    routes unconditionally through ``set_user_type`` (BN function.py:1207-1214),
    so it would pin an AUTO prototype as USER. Use ``set_auto_type`` in that case
    (BN cannot clear the flag; any residue is detected/disclosed separately)."""
    if restores is None:
        return
    fn_start = int(fn.start)

    def _restore():
        rfn = bv.get_function_at(fn_start)
        if rfn is None:
            raise RuntimeError(f"function {hex(fn_start)} missing on prototype restore")
        if had_user_type:
            # Restore the type without routing through the string-sensitive
            # Function.type setter, which can also rename the function.
            try:
                rfn.set_user_type(before_type_obj)
            except Exception:
                rfn.set_user_type(before_prototype)
        else:
            # No prior user prototype: restore the value via set_auto_type so we
            # do not re-pin has_user_type ourselves (#582).
            try:
                rfn.set_auto_type(before_type_obj)
            except Exception:
                rfn.set_auto_type(before_prototype)

    restores.append(_restore)



def _register_local_restore(ctx, bv, restores, *, fn, var, name, type_obj, is_parameter,
                            was_user):
    """Capture how to put a local back to (name, type_obj) on revert.
    Re-resolves the variable fresh at restore time by identifier/storage,
    because the captured Variable's index can shift across reanalysis.

    ``was_user`` records whether the variable carried a USER definition before
    apply (#581). When it did NOT, ``create_user_var`` is the WRONG inverse: it
    permanently pins the AUTO variable as USER (BN will never re-derive its
    name/type again). ``delete_user_var`` is the correct inverse -- it drops the
    override so analysis re-derives the AUTO value. A genuine prior user var is
    replayed with ``create_user_var`` so it survives the preview."""
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
        if was_user:
            rfn.create_user_var(rvar, type_obj, name)
        else:
            delete = getattr(rfn, "delete_user_var", None)
            if delete is not None:
                delete(rvar)
            else:
                # Older BN without delete_user_var: fall back to the replay
                # rather than leaving the previewed change applied.
                rfn.create_user_var(rvar, type_obj, name)

    restores.append(_restore)



def _op_local_rename(ctx, bv, op: dict[str, Any], restores: list | None = None):
    fn = ctx._find_function(bv, op["function"])
    var, is_parameter = vars_mod._find_variable_selector(fn, str(op["variable"]))
    # Reject a degenerate empty/whitespace/null name before touching the view
    # (mirrors _op_rename_symbol's guard, see #363/#605): `str(None)` would
    # otherwise become the literal "None" and slip through, and "" / "   "
    # would create an unnamed/blank-named local. Inspect the RAW value first;
    # done pre-mutation so no view state changes on rejection.
    new_name = _require_nonempty_name(op["new_name"], requested=_operation_requested(ctx, op))
    # Variable.name is a live property backed by the core: snapshot it
    # before mutating, or before_name reads back the new name and
    # verification misclassifies a real change as a noop.
    before_name = str(var.name)
    was_user = bool(fn.is_var_user_defined(var)) if hasattr(fn, "is_var_user_defined") else True
    if before_name != new_name:
        # create_user_var isn't journaled by BN's undo buffer, so register
        # an explicit restore for the preview/rollback paths.
        _register_local_restore(ctx,
            bv, restores, fn=fn, var=var, name=before_name, type_obj=var.type,
            is_parameter=is_parameter, was_user=was_user,
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
    expected_type, _ = _parse_concrete_type(ctx, bv, op, op["new_type"], label="type")
    # Variable.type is a live property backed by the core: snapshot it
    # before mutating (see _op_local_rename).
    before_type_obj = var.type
    before_type = str(before_type_obj)
    was_user = bool(fn.is_var_user_defined(var)) if hasattr(fn, "is_var_user_defined") else True
    if before_type != str(expected_type):
        # create_user_var isn't journaled by BN's undo buffer, so register
        # an explicit restore for the preview/rollback paths.
        _register_local_restore(ctx,
            bv, restores, fn=fn, var=var, name=str(var.name), type_obj=before_type_obj,
            is_parameter=is_parameter, was_user=was_user,
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



def _op_data_retype(ctx, bv, op: dict[str, Any]):
    """Type a DATA variable at an address (#649).

    Struct-typing a recovered global table is a routine RE move, and it had no
    verified mutation path at all: `types declare` defines the struct but cannot
    apply it, `symbol rename --kind data` renames without typing, and `struct field
    set` edits a type rather than a variable's binding -- so the only way through
    was `bn py exec`, i.e. no --preview, no readback verification, no batch
    atomicity, no audit trail. ``define_user_data_var`` IS journaled in BN's undo
    buffer (verified live: revert_undo_actions restores the prior auto type), so the
    standard preview/rollback machinery covers it with no explicit restore."""
    # #1042 item 2: the address operand takes a symbol name (or a rendered
    # `data_<hex>`) like every other address-taking surface, through the one
    # resolver; the row records how the name resolved.
    try:
        address, resolved = ctx._resolve_address_operand(bv, op["address"])
    except RuntimeError as exc:
        raise OperationFailure("unsupported", str(exc),
                               requested=_operation_requested(ctx, op)) from None
    _refuse_unmapped_mutation_address(
        ctx, bv, op, address, "no data variable can be defined there."
    )
    expected_type, _ = _parse_concrete_type(ctx, bv, op, op["new_type"], label="type")
    before = bv.get_data_var_at(address)
    before_type = str(before.type) if before is not None else None
    before_auto = bool(getattr(before, "auto_discovered", True)) if before is not None else None
    if before_type != str(expected_type):
        bv.define_user_data_var(address, expected_type)
    return {
        "op": "data_retype",
        "address": hex(address),
        "before_type": before_type,
        "before_auto_discovered": before_auto,
        "expected_type": str(expected_type),
        "requested": _operation_requested(ctx, op),
        **({"resolved_symbol": resolved} if resolved else {}),
    }


def _verify_data_retype(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    address = _parse_address(item["address"])
    expected = str(item["expected_type"])
    data_var = bv.get_data_var_at(address)
    observed = str(data_var.type) if data_var is not None else None
    item["observed"] = {"address": item["address"], "type": observed}
    if observed != expected:
        raise OperationFailure(
            "verification_failed",
            f"Data variable type at {item['address']} did not land: requested "
            f"{expected!r}, observed {observed!r}",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    item["status"] = "noop" if item.get("before_type") == expected else "verified"
    return item


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
    field_type, _ = _parse_concrete_type(ctx, bv, op, op["field_type"], label="field type")
    # A negative offset silently no-ops in BN's add_member_at_offset, which then
    # surfaced as a misleading "No effective change detected"; an absurd offset
    # (e.g. a mistyped 0xFFFFFFFF) explodes the struct into a multi-GB sparse type
    # that was applied + reported verified. Reject both with an actionable message
    # instead (#369). A negative *hex* offset ("-0x8") doesn't even parse in
    # _parse_address (its hex branch only accepts a leading "0x"), so catch the
    # string sign before parsing so every negative form gets the same message.
    raw_offset = op["offset"]
    if isinstance(raw_offset, str) and raw_offset.strip().startswith("-"):
        raise OperationFailure(
            "invalid_request",
            f"offset must be >= 0; got {raw_offset!r}",
            requested=_operation_requested(ctx, op),
        )
    offset = _parse_address(raw_offset)
    if offset < 0:
        raise OperationFailure(
            "invalid_request",
            f"offset must be >= 0; got {raw_offset!r}",
            requested=_operation_requested(ctx, op),
        )
    if offset > _MAX_STRUCT_FIELD_OFFSET:
        raise OperationFailure(
            "invalid_request",
            f"offset {hex(offset)} would create a struct larger than "
            f"{_MAX_STRUCT_FIELD_OFFSET >> 20} MiB; refused as a likely typo. If "
            "you really need a field this far in, declare the struct with an "
            "explicit large size first.",
            requested=_operation_requested(ctx, op),
        )
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
    new_width = 0
    try:
        new_width = int(field_type.width)
    except Exception:
        new_width = 0
    # --no-overwrite must REFUSE when the new field would clobber existing
    # data, not add an overlapping member: BN's add_member_at_offset(...,
    # overwrite=False) silently appends an overlapping member (worse than the
    # overwrite path). Guard the whole BYTE RANGE the new field occupies, not
    # just an exact start-offset collision -- an offset that lands *inside* a
    # wider member (e.g. 0x4 within an int64_t at 0x0) overlaps just as much
    # (#56).
    if not overwrite and before_type is not None:
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
    # A member the write does NOT remove must not collide by NAME either.
    # add_member_at_offset(overwrite=True) removes every member whose byte
    # range overlaps the new field's, so a retype at a DIFFERENT offset (e.g.
    # narrowing a wide field and re-adding a member under the same name
    # elsewhere in its old range) creates no duplicate and must not be
    # refused -- the guard round 2 rejected did exactly that (#666). Only a
    # member OUTSIDE the range this write removes, that already holds the
    # requested name, is a real duplicate.
    if before_type is not None:
        requested_field_name = str(op["field_name"])
        # One snapshot, judged member by member: BN returns NEW member objects
        # on every `.members` access, so a set of `id()`s collected from one
        # read never matched the members of a second read, and retyping a field
        # under its own name was refused as a duplicate of itself (#961).
        for member in list(getattr(before_type, "members", None) or []):
            if overwrite and _member_overlaps(member, offset, new_width):
                continue
            if str(getattr(member, "name", "")) != requested_field_name:
                continue
            raise OperationFailure(
                "invalid_request",
                f"struct {resolved_name} already has a member named "
                f"{requested_field_name!r} at offset "
                f"{hex(int(getattr(member, 'offset', 0)))}, outside the byte range "
                f"this write replaces ({hex(offset)}..{hex(offset + max(new_width, 1))}); "
                "field names must be unique. Choose a different name, or target "
                "that member's offset instead.",
                requested=_operation_requested(ctx, op),
            )
    # #999 item 4: add_member_at_offset(overwrite=True) removes every member the
    # new field overlaps, and the result named only the member at the exact
    # offset (`before_member`), so `_unk78` at 0x78 vanished under a field set
    # at 0x118 behind `changed=1 verified=1`. Name each one it removes; the
    # member rewritten in place under its own name is `before_member`'s.
    replaced_members = []
    if overwrite and before_type is not None:
        for member in list(getattr(before_type, "members", None) or []):
            if not _member_overlaps(member, offset, new_width):
                continue
            member_offset = int(getattr(member, "offset", 0))
            member_name = str(getattr(member, "name", ""))
            if member_offset == offset and member_name == str(op["field_name"]):
                continue
            replaced_members.append({
                "field_name": member_name,
                "field_type": str(getattr(member, "type", "")),
                "offset": hex(member_offset),
            })
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
        **({"replaced_members": replaced_members} if replaced_members else {}),
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
    by_name = [(i, m) for i, m in enumerate(members) if str(getattr(m, "name", "")) == text]
    if len(by_name) > 1:
        offsets = ", ".join(hex(int(getattr(m, "offset", -1))) for _, m in by_name)
        raise OperationFailure(
            "invalid_request",
            f"struct {resolved_name} has {len(by_name)} members named {text!r} "
            f"(at {offsets}); target one by offset instead of by name",
        )
    if by_name:
        return by_name[0]
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
    # Reject a degenerate empty/whitespace/null name before mutating (mirrors
    # _op_rename_symbol/_op_local_rename, see #363/#605): str(None) would
    # otherwise become the literal "None" and slip through.
    new_name = _require_nonempty_name(op["new_name"], requested=_operation_requested(ctx, op))
    # A rename to a name already held by another member "verifies" cleanly but
    # leaves the struct with a duplicate field name; a later struct_field_delete
    # resolving that name would then pick an arbitrary member (#666).
    for other_index, other_member in enumerate(getattr(builder, "members", []) or []):
        if other_index != index and str(getattr(other_member, "name", "")) == new_name:
            raise OperationFailure(
                "invalid_request",
                f"struct {resolved_name} already has a member named {new_name!r}; "
                "field names must be unique",
                requested=_operation_requested(ctx, op),
            )
    builder.replace(index, member.type, new_name, True)
    _commit_struct_builder(ctx, bv, resolved_name, builder)
    return {
        "op": "struct_field_rename",
        "struct_name": resolved_name,
        "old_name": old_name,
        "new_name": new_name,
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


def _split_top_level_declarations(source: str) -> list[str]:
    """The top-level `;`-separated declarations in *source* (#760).

    Brace/paren/bracket/quote/comment aware, so the `;` inside a struct body, an
    array bound, a string literal or a comment never splits a declaration. A
    trailing fragment with no `;` of its own is kept, so an unterminated
    declaration is still inspected rather than silently ignored.
    """
    fragments: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    index = 0
    while index < len(source):
        char = source[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char == "'" and index > 0 and source[index - 1].isdigit():
            # A C++14 digit separator (`1'000`), not a character literal -- and
            # `isdigit`, not `isalnum`, so the widish/UTF char-literal prefixes
            # (`L';'`, `u'x'`, `U'x'`) still open a literal as they should. Opening a
            # quote where none belongs swallows every later top-level `;` into one
            # fragment and silently stops the guard inspecting the rest of the string
            # -- a missed refusal rather than a false one, but free to avoid.
            pass
        elif char in "\"'":
            quote = char
        elif char == "/" and source[index + 1:index + 2] == "*":
            end = source.find("*/", index + 2)
            index = len(source) if end < 0 else end + 2
            continue
        elif char == "/" and source[index + 1:index + 2] == "/":
            end = source.find("\n", index + 2)
            index = len(source) if end < 0 else end + 1
            continue
        elif char in "{([":
            depth += 1
        elif char in "})]":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            fragments.append(source[start:index + 1])
            start = index + 1
        index += 1
    tail = source[start:]
    if tail.strip():
        fragments.append(tail)
    # A fragment that is only whitespace/comments declares nothing: drop it so the
    # split reflects declarations, not the comment layout around them.
    return [
        fragment for fragment in fragments if _strip_c_comments(fragment).strip()
    ]


_LEADING_QUALIFIER_RE = re.compile(r"^(?:static|extern|const|volatile|register|inline)\b")
_TYPE_TAG_RE = re.compile(r"^(?:typedef\s+)?(?:struct|union|enum|class)\b[^;={]*\{")


def _skip_balanced_parens(text: str) -> int | None:
    """Index just past the balanced `( ... )` group *text* begins with, else None.

    A scanner rather than a regex because the groups nest:
    `__attribute__((aligned(8)))` has an inner `)`, which a `[^)]*` class stops at
    (review of #761 -- the shape that let a drop stay silent).
    """
    if not text.startswith("("):
        return None
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _strip_leading_declaration_noise(text: str) -> str:
    """Consume the prefixes a definition may carry ahead of its tag (#760).

    Storage-class keywords, an `extern "C"` linkage specifier, and
    `__attribute__((...))` / `__declspec(...)` groups -- the attribute groups with a
    balanced-paren scan, since they nest. Anything else (including malformed
    prefixes) is returned unconsumed, so the tag matcher decides.
    """
    remaining = text
    while remaining:
        match = _LEADING_QUALIFIER_RE.match(remaining)
        if match:
            remaining = remaining[match.end():].lstrip()
            continue
        if remaining.startswith('"'):
            end = remaining.find('"', 1)
            if end < 0:
                return remaining
            remaining = remaining[end + 1:].lstrip()
            continue
        for keyword in ("__attribute__", "__attribute", "__declspec"):
            if not remaining.startswith(keyword):
                continue
            rest = remaining[len(keyword):].lstrip()
            end = _skip_balanced_parens(rest)
            if end is None:
                return remaining
            remaining = rest[end:].lstrip()
            break
        else:
            return remaining
    return remaining


def _declares_a_type(fragment: str) -> bool:
    """Whether a top-level fragment intends to DEFINE a type (#760).

    A fragment qualifies when it STARTS with a definition: an optional run of
    storage-class keywords, an `extern "C"` specifier and `__attribute__`/
    `__declspec` groups, an optional `typedef`, then a tag whose body opens before
    any `;` or `=`. The `=` exclusion is load-bearing: a variable declaration with a
    brace initializer (`struct A x = {};`) is a USAGE whose braces are initializer
    braces, and an earlier cut of this check that only looked for `{` refused it --
    breaking working input and diagnosing a variable as a dropped type declaration
    (review of #761).

    A brace-less fragment qualifies only when it is a `typedef`, so a usage
    (`struct A x;`), a forward declaration (`struct A;`) or a function/variable
    declaration is never inspected. A definition whose tag is introduced by a macro
    expansion (`DECL(uint32);`) stays invisible to any text-level check like this.
    """
    text = _strip_c_comments(fragment).strip()
    if "{" not in text:
        return text.startswith("typedef")
    return bool(_TYPE_TAG_RE.match(_strip_leading_declaration_noise(text)))


def _declarations_without_named_types(ctx, bv, declaration: str,
                                      source_path: str | None) -> list[str]:
    """Top-level fragments that define no named type when parsed on their own (#760).

    The platform parser DROPS a declaration whose name collides with a built-in
    type instead of reporting it, so `struct uint32_t { int x; }; struct
    widget_cfg_t { int y; };` applied one type, discarded the other, and reported
    `verified`. Re-parsing each flaggable fragment alone is the available ground
    truth for the discarded one. A fragment that FAILS to parse alone is skipped:
    it may legitimately depend on a type an earlier fragment defines, so silence is
    the honest answer. Single-fragment declarations are left to the all-dropped
    guard in `_op_types_declare`, which already refuses them.
    """
    fragments = _split_top_level_declarations(declaration)
    if len(fragments) < 2:
        return []
    dropped: list[str] = []
    for fragment in fragments:
        if not _declares_a_type(fragment):
            continue
        try:
            parsed = _parse_declaration_source(ctx, bv, fragment, source_path=source_path)
        except Exception:
            continue
        if not list(parsed.get("types") or []):
            dropped.append(" ".join(fragment.split()))
    return dropped


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


def _declares_body_for(declaration: str, name: str) -> bool:
    """Whether *declaration* gives the aggregate *name* a body (`struct name {`,
    `class name : Base {`, backtick-quoted or not), as opposed to only naming
    it (`struct name*`, `struct name;`)."""
    quoted = re.escape(name)
    pattern = (r"\b(?:struct|class|union|enum)\s+(?:`" + quoted + r"`|" + quoted + r"(?![\w:]))"
               r"\s*(?:final\s*)?(?::(?!:)[^{};]*)?\{")
    text = _strip_c_comments(declaration)
    try:
        # `struct alignas(8) name {` / `struct __attribute__((packed)) name {`
        text = _strip_head_attributes(text)
    except ValueError:
        pass
    return re.search(pattern, text) is not None


def _library_type(bv, name: str):
    """``(library, type)`` for the first type library loaded on the view that
    defines *name* (e.g. libc's `struct timespec`), whether or not the view has
    imported it, else None."""
    try:
        libraries = list(getattr(bv, "type_libraries", None) or [])
    except Exception:
        return None
    for library in libraries:
        try:
            found = library.get_named_type(name)
        except Exception:
            continue
        if found is not None:
            return library, found
    return None


def _view_type_names(bv) -> set[str] | None:
    """Every type name the view defines, or None when the view cannot list them."""
    try:
        return {str(name) for name in bv.type_names}
    except Exception:
        return None


def _is_library_import(bv, name: str) -> bool:
    """Whether the view's *name* is an AUTO type imported from a type library
    (never a user type: those are the undo journal's to revert)."""
    try:
        return (bv.lookup_imported_type_library(name) is not None
                and bool(bv.is_type_auto_defined(name)))
    except Exception:
        return False


def _import_referenced_library_types(bv, type_objs, *, apply: bool = True) -> list[str]:
    """#993 item 2: every type a loaded type library defines that *type_objs*
    reference, transitively, and that the view lacks -- imported through BN's
    own ``import_library_type`` when *apply* (so a declaration that points at
    libc's `struct timespec` lands with the type it names, and BN renames a
    dependency that collides with a user type, `__time_t` -> `__time_t_1`), or
    only named when not. Returns the names found missing, in discovery order.

    BN's import is the only path: bn never removes an imported type (BN would
    keep its imported-record and refuse to import it again), so the record and
    the view always agree and the import does what BN's parser would."""
    pending = [name for type_obj in type_objs for name in _named_type_references(type_obj)]
    seen: set[str] = set()
    missing: list[str] = []
    while pending:
        name = pending.pop(0)
        if name in seen or _lookup_named_type(bv, name) is not None:
            continue
        seen.add(name)
        found = _library_type(bv, name)
        if found is None:
            continue
        library, library_type = found
        missing.append(name)
        if apply:
            try:
                bv.import_library_type(name, library)
            except Exception as exc:
                bn.log_error(f"BN Agent Bridge: importing library type {name!r} failed: {exc!r}")
        pending.extend(_named_type_references(library_type))
    return missing


def _preview_library_refusal(bv, names) -> str:
    """Why a `--preview` of a declaration is refused when its parsed types refer
    to type-library types the view lacks. This is the path where BN's parse
    itself succeeded without importing (a tag only pointed at, e.g. `struct
    timespec*`); the names are the parsed types' own references, not text
    matches. It advises nothing but the live run: any declare it printed would
    itself need those types, or overwrite a user type of the name it picked."""
    matches = [(name, str(getattr(found[0], "name", "") or "<unnamed library>"))
               for name in names if (found := _library_type(bv, name)) is not None]
    return (
        "not previewed: the declaration's types refer to type-library type(s) the view "
        "does not have yet. The live declare imports them with the library types they "
        f"depend on: {_library_match_text(matches) or ', '.join(names)}. No undo removes an "
        "import, so the preview stopped before importing. Nothing was imported. Run the "
        "declare without --preview: it imports them and lists them as `imported_from_library`."
    )


class _TypeBaseline(NamedTuple):
    """The view's type names and ids before a type-parsing step.
    Imports are found by ID (#993 review round 4): BN
    renames an import that collides with a user type (`__time_t` comes in as
    `__time_t_1` under id `<guid>:["__time_t"]`), so a name diff can be hidden
    by a same-named user type, while a new id cannot."""
    names: set[str]
    ids: set[str] | None


def _view_type_ids(bv) -> set[str] | None:
    """Every type id the view defines, or None when it cannot list them."""
    try:
        return {str(type_id) for type_id in bv.type_container.type_ids}
    except Exception:
        return None


def _declare_batch_type_names(bv, operations) -> _TypeBaseline | None:
    """One initial baseline for every mutation, even a comment-only preview."""
    trace = _IMPORT_TRACE.get()
    if trace is None:
        return None
    trace["parses_types"] = any(
        isinstance(op, dict) and op.get("op") in {
            "types_declare", "struct_field_set", "set_prototype", "local_retype", "data_retype",
        } for op in operations)
    if "baseline" in trace:
        return trace["baseline"]
    return _TypeBaseline(_view_type_names(bv) or set(), _view_type_ids(bv))


def _library_for_type_id(bv, type_id: str):
    """The type library a type id names, or None. BN gives an imported type
    the id `<library guid>:<name>` (#993 review round 4); the guid is looked up
    among the view's attached libraries, its platform's libraries (an
    explicitly selected platform's `SYSCALLS` is not attached, #1012 item 2)
    and BN's own registry for the architecture. Only a guid BN resolves counts."""
    guid, sep, _ = str(type_id).partition(":")
    if not sep or not guid:
        return None
    candidates: list = []
    for owner in (bv, getattr(bv, "platform", None)):
        try:
            candidates.extend(getattr(owner, "type_libraries", None) or [])
        except Exception:
            continue
    for library in candidates:
        try:
            if str(library.guid) == guid:
                return library
        except Exception:
            continue
    lookup = getattr(getattr(bn, "TypeLibrary", None), "from_guid", None)
    arch = getattr(bv, "arch", None)
    if lookup is None or arch is None:
        return None
    try:
        return lookup(arch, guid)
    except Exception:
        return None


def _import_library_name(bv, name: str, type_id: str | None) -> str | None:
    """The name of the library an imported *name* came from: by its type id's
    guid, else BN's own imported-type record. None when neither names one."""
    library = _library_for_type_id(bv, type_id) if type_id else None
    if library is None:
        try:
            record = bv.lookup_imported_type_library(name)
        except Exception:
            record = None
        library = record[0] if isinstance(record, tuple) and record else record
    library_name = str(getattr(library, "name", "") or "") if library is not None else ""
    return library_name or None


def _new_library_imports(bv, baseline: _TypeBaseline) -> dict[str, str | None]:
    """Retained names/IDs: every new ID in previews, recognized imports live.

    Preview safety is deliberately provenance-independent, including retained
    non-library IDs. Live, an import is recognized by BN's imported-type record
    or by a type id whose guid BN resolves to a type library."""
    current_ids = _view_type_ids(bv) if baseline.ids is not None else None
    if current_ids is None:
        current = _view_type_names(bv) or set()
        return {name: None for name in current - baseline.names
                if _PREVIEW_PARSE.get() or _is_library_import(bv, name)}
    found: dict[str, str | None] = {}
    for type_id in current_ids - baseline.ids:
        try:
            name = str(bv.get_type_name_by_id(type_id) or "")
        except Exception:
            name = ""
        if _PREVIEW_PARSE.get():
            found[name or type_id] = type_id
        elif name and (_is_library_import(bv, name) or _library_for_type_id(bv, type_id) is not None):
            found[name] = type_id
    return found


def _stamp_library_imports(row: dict[str, Any], names) -> None:
    """Record *names* as the row's `imported_from_library`, and take them off
    its `referenced_existing_types`: a library type the view did not have is
    not an existing type the declaration left unchanged."""
    names = sorted(set(row.get("imported_from_library") or []) | set(names))
    if not names:
        return
    row["imported_from_library"] = names
    kept = [name for name in row.get("referenced_existing_types") or [] if name not in names]
    if kept:
        row["referenced_existing_types"] = kept
    else:
        row.pop("referenced_existing_types", None)


def _report_library_imports(bv, baseline: _TypeBaseline | None, results: list,
                            failed_row: dict[str, Any] | None = None) -> None:
    """Attribute imports to the exact op whose parse/apply introduced them."""
    trace = _IMPORT_TRACE.get()
    if trace is None:
        return
    rows = results + ([failed_row] if failed_row is not None else [])
    for op, row in zip(trace["operations"], rows):
        if id(op) in trace["parsed"]:
            row["parse_performed"] = True
        names = trace["imports"].get(id(op), ())
        if names:
            _stamp_library_imports(row, names)


def _fail_importing_preview(results: list) -> list:
    """Fail only the importing steps, including preflight-only parses."""
    out = []
    trace = _IMPORT_TRACE.get()
    for index, row in enumerate(results):
        mine = row.get("imported_from_library") or []
        if mine:
            # Keep the real parse/apply failure as well as the import warning.
            # This helper runs before rollback and again on final disclosure.
            prior = (str(row.get("message") or "")
                     if row.get("status") in FAILED_MUTATION_STATUSES else "")
            if trace is not None:
                prior = trace["import_errors"].setdefault(index, prior)
            message = (
                "Binary Ninja retained new type(s) from this operation "
                f"({', '.join(mine)}); a preview cannot undo an import, so the imported "
                "types stay in the view (a rollback cannot remove them). "
                + ("Run the declare without --preview." if row.get("op") == "types_declare"
                   else "Run the operation without --preview only if the change is intended.")
            )
            row = {**row, "status": ("rollback_failed" if row.get("status") == "rollback_failed"
                                    else "unsupported"),
                   "message": f"{prior} {message}".strip() if prior else message}
        out.append(row)
    return out


def _clear_dangling_type_names(bv, baseline: _TypeBaseline) -> None:
    """Revert step for a batch that declared types (#993 item 2 review).

    Defining a type that refers to a name the view lacks (a pointer to a struct
    nothing defines yet) can register that name with
    no type behind it, and the undo journal does not remove the entry. Left
    there it poisons the next parse: BN takes the name as present, will not
    import it, and `struct timespec ts;` then fails as an incomplete type.
    `undefine_type` alone does not drop such an entry; defining a placeholder
    under its id and undefining that does. Only names added since *baseline*
    with nothing behind them are touched. Raises when one stays, so the
    revert reports itself incomplete instead of clean."""
    leftover = []
    for name in sorted((_view_type_names(bv) or set()) - baseline.names):
        try:
            type_id = bv.get_type_id(name)
            if not type_id or bv.get_type_by_id(type_id) is not None:
                continue
            bv.define_type(type_id, name, bn.Type.void())
            bv.undefine_type(type_id)
        except Exception as exc:
            bn.log_error(f"BN Agent Bridge: clearing dangling type name {name!r} failed: {exc!r}")
        if name in (_view_type_names(bv) or set()):
            leftover.append(name)
    if leftover:
        raise RuntimeError(f"could not clear dangling type name(s): {', '.join(leftover)}")


def _declares_forward(declaration: str, name: str) -> bool:
    """Whether *declaration* forward-declares *name* as a statement of its own
    (`struct name;`), as opposed to only mentioning it in a member type."""
    quoted = re.escape(name)
    pattern = (r"(?:^|[;{}])\s*(?:struct|class|union|enum)\s+(?:`" + quoted + r"`|" + quoted
               + r"(?![\w:]))\s*;")
    return re.search(pattern, _strip_c_comments(declaration)) is not None


# A C string or character literal, escapes included.
_C_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')


def _tag_keyword_mentioning(declaration: str, name: str) -> str | None:
    """The tag keyword (`struct`/`class`/`union`) *declaration* spells *name*
    with (`struct name*`, ``struct `ns::name`*``), or None when it never does.
    Comments, string/character literals and attribute arguments are not code,
    so a `"struct name"` inside one is no mention (#993 review)."""
    text = _C_LITERAL_RE.sub('""', _strip_c_comments(declaration))
    try:
        text = _strip_head_attributes(text)
    except ValueError:
        pass
    quoted = re.escape(name)
    match = re.search(r"\b(struct|class|union)\s+(?:`" + quoted + r"`|" + quoted + r"(?![\w:]))", text)
    return match.group(1) if match else None


def _opaque_tag_stub(ctx, bv, keyword: str, name: str):
    """The empty aggregate BN's parser makes for a forward declaration
    (`struct name;`), or None when it makes none. Parsed rather than built so
    the stub is exactly what a member mention of a NEW name already defines."""
    spelled = name if re.fullmatch(r"[A-Za-z_]\w*", name) else f"`{name}`"
    try:
        parsed = _parse_declaration_source(ctx, bv, f"{keyword} {spelled};")
    except Exception:
        return None
    for parsed_name, type_obj in parsed.get("types") or []:
        if str(parsed_name) == name:
            return type_obj
    return None


def _drop_references_to_existing_types(ctx, bv, declaration: str,
                                       named_types: list) -> tuple[list, list[str], list[str]]:
    """#963: BN's declaration parser does not see the view's types, so a member
    such as ``struct `ns::Widget::VTable`* vtable;`` or ``struct Alpha* a;``
    comes back as a NAMED, EMPTY forward-declared struct -- and defining every
    parsed type overwrote the existing ``ns::Widget::VTable`` (BN's own, with
    its slot members) or ``Alpha`` with a 0-byte stub, reported `verified`.
    That silently destroyed the very type the declaration was pointing at: the
    typed vptr then led to an empty VTable, so virtual calls kept rendering as
    raw offsets.

    A parsed type is a mere reference -- and is NOT applied -- when it is an
    empty aggregate (no members, zero width), the declaration gives it no body,
    and the view already defines a type of that name, or a type library loaded
    on the view does (a 0-byte user `timespec` would shadow libc's). A forward
    declaration of a NEW name still creates its opaque type, as before.

    #993 item 1: a tag mentioned only inside a function-pointer parameter list
    (`typedef void (*cb)(struct A*);`) comes back from the parser as a named
    `void` rather than an empty struct, which redefined an existing `A` as
    `void` and failed verification. Such a `void` is the same reference: left
    alone when the name exists, and replaced by the forward declaration's empty
    aggregate when it does not, so a new name never becomes a `void` typedef.

    Returns the types to apply, the names left untouched, and the new empty
    types created only because a member MENTIONED them (no body, no forward
    declaration of their own) -- typically a misspelled name -- so the result
    can say so instead of passing them off as declared."""
    kept, referenced, by_mention = [], [], []
    for name, type_obj in named_types:
        mention_keyword = (_tag_keyword_mentioning(declaration, str(name))
                           if "Void" in _type_class_name(type_obj) else None)
        empty = (not list(getattr(type_obj, "members", None) or [])
                 and int(getattr(type_obj, "width", 0) or 0) == 0
                 and (_type_tag_kind(type_obj) is not None or mention_keyword is not None))
        if not empty or _declares_body_for(declaration, str(name)):
            kept.append((name, type_obj))
            continue
        if _lookup_named_type(bv, str(name)) is not None or _library_type(bv, str(name)) is not None:
            referenced.append(str(name))
            continue
        if mention_keyword is not None:
            # Not `stub or type_obj`: a 0-byte BN type is falsy.
            stub = _opaque_tag_stub(ctx, bv, mention_keyword, str(name))
            if stub is not None:
                type_obj = stub
        if not _declares_forward(declaration, str(name)):
            by_mention.append(str(name))
        kept.append((name, type_obj))
    return kept, referenced, by_mention


# What may sit between `struct`/`class` and the name, or after the name, in a
# definition head without being part of either: `alignas(N)`,
# `__attribute__((...))`, `__declspec(...)` and `[[...]]`.
_HEAD_ATTRIBUTE_RE = re.compile(r"\b(?:alignas|__attribute__|__attribute|__declspec)\s*\(")
_AGGREGATE_NAME_RE = re.compile(r"`[^`]+`|[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*")
_BASE_SPECIFIER_WORDS = {"public", "protected", "private", "virtual", "struct", "class"}


def _split_base_list(text: str) -> list[str]:
    """Split a base clause at top-level commas only (template arguments carry
    their own)."""
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif ch == "," and depth <= 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _strip_head_attributes(head: str) -> str:
    """*head* without `alignas(...)`, `__attribute__((...))`, `__declspec(...)`
    and `[[...]]`, each removed with its balanced brackets."""
    out = head
    while True:
        match = _HEAD_ATTRIBUTE_RE.search(out)
        opener = "("
        if match is None:
            start = out.find("[[")
            if start < 0:
                return out
            body, opener = start + 1, "["
        else:
            start, body = match.start(), match.end() - 1
        depth, end = 0, None
        closer = ")" if opener == "(" else "]"
        for i in range(body, len(out)):
            if out[i] == opener:
                depth += 1
            elif out[i] == closer:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError(f"unbalanced attribute in {head.strip()!r}")
        if opener == "[":
            end += 1   # the second `]` of `[[...]]`
        out = out[:start] + " " + out[end:]


def _top_level_colon(head: str) -> int:
    """Index of the first `:` in *head* that is not half of `::` and not inside
    brackets, or -1."""
    depth = 0
    i = 0
    while i < len(head):
        ch = head[i]
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif ch == ":" and depth <= 0:
            if head[i:i + 2] == "::":
                i += 2
                continue
            return i
        i += 1
    return -1


def _unrecorded_base_clauses(declaration: str) -> dict[str, str]:
    """#965 (split out of #989): bn does not record C++ bases. BN's type parser
    lays a derived struct's own members out after its bases but keeps no base
    (`base_structures` comes back empty), and three reviews of a reconstruction
    found shapes whose layout it got wrong. So nothing is recorded, and every
    base clause in the declaration -- whatever attributes precede or follow the
    name (`struct alignas(32) D : A`, `struct __attribute__((packed)) D : A`,
    `struct D final : A`) -- is reported as ``{derived name: reason}`` so no
    base is ever dropped silently.

    A definition head with a base clause this cannot read raises
    `OperationFailure` rather than being skipped."""
    text = _strip_c_comments(declaration)
    unrecorded: dict[str, str] = {}
    for match in re.finditer(r"\b(?:struct|class)\b", text):
        before = text[:match.start()].rstrip()
        if before.endswith("enum"):
            continue   # `enum class E : int {` -- an underlying type, not a base
        # The definition head runs to the first top-level `{`; a `;`, `}` or an
        # unbalanced `)` first means this keyword names a type in a member,
        # parameter or forward declaration instead.
        end = match.end()
        stop = _definition_head_end(text, end)
        if stop is None:
            continue
        raw_head = text[end:stop]
        try:
            head = _strip_head_attributes(raw_head)
        except ValueError as exc:
            raise OperationFailure(
                "invalid_request",
                f"could not read the struct/class head {raw_head.strip()!r}: {exc}",
            ) from exc
        colon = _top_level_colon(head)
        if colon < 0:
            continue
        name_part = re.sub(r"\bfinal\b", " ", head[:colon]).strip()
        bases = [" ".join(w for w in spec.split() if w not in _BASE_SPECIFIER_WORDS).strip("`")
                 for spec in _split_base_list(head[colon + 1:])]
        if not _AGGREGATE_NAME_RE.fullmatch(name_part) or not bases or not all(bases):
            raise OperationFailure(
                "invalid_request",
                f"could not read the base clause in {('struct' + raw_head).strip()!r}; bn reports "
                "every C++ base it does not record (#965) and will not drop one it cannot read. "
                "Declare the struct without the base clause, or with a plainer head.",
            )
        name = re.sub(r"\s*::\s*", "::", name_part.strip("`"))
        unrecorded[name] = (
            f"bn does not yet record C++ bases (see #965): declared without its base(s) "
            f"{', '.join(bases)}. Its own members keep the offsets the parser gave them, after "
            f"the bases, but no base or inherited field is recorded. Those offsets can differ "
            f"from the compiler's when a base is empty or non-POD (for `struct E {{}}; struct D "
            f": E {{ int x; }}` the parser puts x at 0x4, g++ at 0x0), so check them"
        )
    return unrecorded


def _definition_head_end(text: str, start: int) -> int | None:
    """Index of the `{` that opens an aggregate body after *start*, or None when
    a `;`, `}` or unbalanced `)` (outside brackets) comes first."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            if ch == "{":
                return i
            if ch in ";}":
                return None
    return None


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


# Reserved C keywords that can occupy a TYPE POSITION and that BN's parser
# does NOT implement. Enumerated rather than inferred, because "reserved" is
# not queryable from the parser and guessing would be the fabrication this
# guard exists to stop.
_RESERVED_UNIMPLEMENTED_TYPE_KEYWORDS = frozenset({
    "_Bool", "_Complex", "_Imaginary", "_Atomic",
})


def _refuse_reserved_keyword_tags(ctx, op: dict[str, Any], named_types) -> None:
    """Refuse a declaration whose TAG is a reserved C keyword BN cannot parse.

    This completes an existing guard rather than adding a second one. For a
    name the parser DOES implement, `_declarations_without_named_types`
    (#760) already refuses: the platform parser silently DROPS a declaration
    colliding with a built-in, so `struct uint32_t {...}` would apply one
    type, discard the other and report `verified`. That guard's coverage is
    exactly the parser's builtin table.

    A reserved keyword the parser does NOT implement reaches the same harm
    by the opposite route: nothing is dropped, because nothing was there --
    the tag is accepted as a fresh name. Measured, `struct bool {...}` is
    refused today while `struct _Bool {...}` is applied and `verified`, and
    a later `struct three { _Bool a; _Bool b; _Bool c; }` then comes back
    0xC with three `struct _Bool` members where C says 3 bytes. The
    asymmetry IS the defect.

    REFUSE rather than warn, for a structural reason: the fabrication is
    generated by the NEXT command -- the one writing a bare `_Bool` field --
    not by this declaration, so a warning here leaves that command's
    `verified` untouched. Same shape as the bitfield guard above, which
    names the consequence rather than the syntax.

    Checked over the PARSED names, not the source text, so the ground truth
    is what BN would actually define: a `_Bool` in a comment or used as a
    FIELD name cannot trip it. Tags are normalised, because a `struct X`
    declaration may report as `struct X` rather than `X`.

    SCOPE, stated because the next reader will otherwise expect more: this
    makes the tool HONEST, not CAPABLE. BN has no `_Bool`, so a field of
    that type still fails to parse and still will. What changes is that the
    tool stops manufacturing a meaning for a reserved name and stamping it
    `verified`.
    """
    offenders = []
    for name, _type_obj in named_types or ():
        tag = str(name).split()[-1] if str(name).strip() else ""
        if tag in _RESERVED_UNIMPLEMENTED_TYPE_KEYWORDS:
            offenders.append(tag)
    if not offenders:
        return
    named = ", ".join(sorted(set(offenders)))
    raise OperationFailure(
        "invalid_request",
        f"{named} is a reserved C keyword that this parser does not implement, "
        f"so declaring a type with that tag does not shadow the builtin -- it "
        f"CREATES one. A later field written as `{offenders[0]} x;` would then "
        f"silently mean this struct rather than the C type, and be reported "
        f"`verified`: three such fields lay out as 0xC bytes where C says 3. "
        f"A tag that collides with a builtin the parser DOES implement is "
        f"already refused (#760); this is the same refusal for the reserved "
        f"names it does not. Rename the tag (e.g. `{offenders[0].lstrip('_')}_t`).",
        requested=_operation_requested(ctx, op),
    )


@_track_operation
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
        parsed = _parse_operation_declaration(ctx, bv, op)
    except OperationFailure:
        raise
    except Exception as exc:
        # A malformed C declaration is a user mistake, not an engine bug: surface
        # a clean invalid_request carrying BN's parser message, instead of
        # leaking the raw SyntaxError class / mislabeling it internal_error.
        # Mirrors _parse_type_or_hint for the other type-taking ops (#122).
        detail = " ".join(str(exc).split()).rstrip(".")
        raise OperationFailure(
            "invalid_request",
            "could not parse declaration" + (f": {detail}" if detail else "") + "."
            + _type_spelling_hint(bv, declaration, detail),
            requested=_operation_requested(ctx, op),
        ) from exc
    named_types = list(parsed["types"])
    _refuse_reserved_keyword_tags(ctx, op, named_types)
    if not named_types:
        raise OperationFailure(
            "invalid_request",
            "Parsed declarations but no named types were defined. "
            "Provide a named type declaration and check for names that collide "
            "with built-in types.",
            requested=_operation_requested(ctx, op),
            observed={
                "defined_types": {},
                "parsed_functions": [name for name, _ in parsed["functions"]],
                "parsed_variables": [name for name, _ in parsed["variables"]],
            },
        )
    # A declaration whose name collides with a built-in type is DROPPED by the
    # parser without an error, so a multi-declaration string could define one type,
    # discard another, and still report `verified` (#760). Refuse rather than apply
    # a partial declaration: the caller sees the whole outcome or none, which is the
    # behaviour the all-dropped guard above already establishes.
    dropped = _declarations_without_named_types(ctx, bv, declaration, op.get("source_path"))
    if dropped:
        raise OperationFailure(
            "invalid_request",
            "these declarations define no named type and would be discarded: "
            + "; ".join(dropped)
            + ". A declaration whose name collides with a built-in type is dropped by "
            "the parser; rename it, or declare the remaining types without it.",
            requested=_operation_requested(ctx, op),
            observed={
                "defined_types": sorted(str(name) for name, _ in named_types),
                "dropped_declarations": dropped,
            },
        )
    # #778: a declaration may ALSO carry function prototypes / variable
    # declarations. The parser returns all three buckets and this op applies
    # only `types`; the other two were echoed as bare `parsed_functions` /
    # `parsed_variables` counts with nothing saying they had NOT been applied,
    # so a caller who declared a struct and its accessors in one header got the
    # struct, a `verified` status, and no signal about the rest.
    #
    # DISCLOSED, not refused. Refusing was the first cut and it is wrong twice
    # over: #760 already settled that a variable declaration with a brace
    # initializer is a USAGE rather than a definition and must not break working
    # input (test_types_declare_allows_a_variable_declaration_with_a_brace_
    # initializer_760), and a header that declares a type alongside the
    # functions taking it is the normal shape -- refusing it would reject the
    # very input this verb exists for.
    #
    # Applying them is not possible here either: a bare `int f(struct S *);`
    # carries no ADDRESS, and BN binds a function type through
    # `Function.set_user_type` and a data var through `define_user_data_var`,
    # both of which need a target this op never receives. Matching prototypes to
    # symbols by name would invent an address-resolution policy nobody asked
    # for. So the honest answer is to say plainly what was and was not applied
    # and name the address-aware verbs that can finish the job.
    _proto_fns = [str(name) for name, _ in parsed["functions"]]
    _proto_vars = [str(name) for name, _ in parsed["variables"]]
    _unapplied_note = None
    if _proto_fns or _proto_vars:
        _named = "; ".join(
            part for part in (
                f"function prototype(s): {', '.join(_proto_fns)}" if _proto_fns else "",
                f"variable declaration(s): {', '.join(_proto_vars)}" if _proto_vars else "",
            ) if part
        )
        _unapplied_note = (
            f"declared the named types only; NOT applied -- {_named}. A bare "
            "prototype carries no address, so bind it with `bn proto set "
            "<function>` or `bn data retype <address>`."
        )
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
    parsed_type_count = len(named_types)
    named_types, referenced_existing, created_by_mention = _drop_references_to_existing_types(
        ctx, bv, declaration, named_types)
    reference_only = not named_types
    if reference_only:
        # #989 review: `struct Alpha;` against an existing Alpha used to apply
        # (as the clobbering stub) and exit 0; a header of forward declarations
        # must stay a success. It is now the no-op it always should have been:
        # report each view type as declared-and-unchanged so verification
        # observes it and stamps `noop`. A type only a type library defines
        # (`struct timespec;`) has nothing in the view to observe; the result's
        # `reference_only` marker lets the verifier stamp that case `noop` too.
        named_types = [(name, bv.get_type_by_name(name)) for name in referenced_existing
                       if bv.get_type_by_name(name) is not None]
    # #965: bn records no C++ base; every base clause is reported instead.
    # Read before anything is defined, so an unreadable clause applies nothing.
    try:
        unattached_bases = _unrecorded_base_clauses(declaration)
    except OperationFailure as exc:
        raise OperationFailure(exc.status, exc.message, requested=_operation_requested(ctx, op)) from exc
    defined_types = {}
    defined_type_layouts = {}
    before_defined_types = {}
    for name, _type_obj in named_types:
        existing = ctx._current_type_entry(bv, str(name))
        before_defined_types[str(name)] = existing["decl"] if existing is not None else None
    if not reference_only:
        for name, type_obj in named_types:
            bv.define_user_type(name, type_obj)
        # #993 item 2: BN imports a library type the parser needed BY VALUE and
        # leaves one only pointed at dangling; import both on purpose (the
        # batch reports what came in). A preview imports nothing: its parse runs
        # with import_dependencies=False, which BN refuses for most library
        # uses; one it parses anyway (a tag only pointed at) is refused here.
        previewing = _PREVIEW_PARSE.get()
        needed = _import_referenced_library_types(
            bv, [type_obj for _, type_obj in named_types], apply=not previewing)
        if previewing and needed:
            raise OperationFailure(
                "unsupported", _preview_library_refusal(bv, needed),
                requested=_operation_requested(ctx, op),
                observed={"library_types": needed, "imported": False})
    # Read back only once everything is defined: a type rendered while a name it
    # uses was still missing (`void (*)(Leaf*)` before `struct Leaf` lands) reads
    # differently afterwards, and verification compares against the final view.
    for name, type_obj in named_types:
        current = ctx._current_type_entry(bv, str(name))
        defined_types[str(name)] = current["decl"] if current is not None else str(type_obj)
        defined_type_layouts[str(name)] = current["layout"] if current is not None else ctx._render_type_layout(type_obj)
    result = {
        "op": "types_declare",
        "defined_types": defined_types,
        "defined_type_layouts": defined_type_layouts,
        "before_defined_types": before_defined_types,
        "count": 0 if reference_only else len(defined_types),
        "parsed_functions": [name for name, _ in parsed["functions"]],
        "parsed_variables": [name for name, _ in parsed["variables"]],
        # What the parser returned, before references to existing types were
        # set aside (#989 review): `count` is what was applied, and
        # `referenced_existing_types` names the rest.
        "parsed_type_count": parsed_type_count,
        "parsed_function_count": len(parsed["functions"]),
        "parsed_variable_count": len(parsed["variables"]),
        # #965: every C++ base clause in the declaration, none of which bn
        # records (BN's parser drops them), named with the reason; absent when
        # the declaration has none, so an ordinary result grows no key.
        **({"unattached_bases": unattached_bases} if unattached_bases else {}),
        # #963: existing types the declaration only REFERENCED (left as they
        # were rather than overwritten by the parser's empty forward stub).
        **({"referenced_existing_types": referenced_existing} if referenced_existing else {}),
        # #989 review: NEW empty types created only because a member mentioned
        # an unknown name (no body, no forward declaration of their own) --
        # usually a misspelling, e.g. ``struct `ns::W::VTabl`*``.
        **({"created_by_reference": created_by_mention} if created_by_mention else {}),
        **({"reference_only": True,
            "note": "the declaration only forward-declares existing types; nothing was changed"}
           if reference_only else {}),
        # #825 item 3: a `--file` declaration parses with the header's PARENT
        # DIRECTORY as an implicit include root, so `#include "sibling.h"`
        # resolves. The root was derivable from `requested.source_path` but
        # never stated, which makes the interesting case invisible: a declare
        # that only succeeded because a sibling header was found, or that
        # picked a shadowing sibling over the header the user expected.
        # Absent for an inline declaration, so an ordinary declare grows no
        # key -- and it comes from the same helper the parse used, so the
        # reported root cannot differ from the one actually searched.
        **({"include_root": _declaration_include_root(op.get("source_path"))}
           if op.get("source_path") else {}),
        # #778: the two buckets this verb parsed but CANNOT bind, stated as
        # not-applied rather than left for the caller to infer from a bare
        # count. Absent entirely when the declaration was types-only, so an
        # ordinary request grows no keys and reads exactly as before.
        **({"unapplied_prototypes": {"functions": _proto_fns,
                                     "variables": _proto_vars},
            "unapplied_note": _unapplied_note} if _unapplied_note else {}),
        "requested": _operation_requested(ctx, op),
    }
    return result


def _function_looks_like_code(bv, created, addr: int) -> str | None:
    """Return a human-readable reason if the just-created function does NOT look
    like real code (so it must be removed and the op failed), else None.

    Forced ``create_user_function`` always lands *a* function -- even on non-code
    -- so the post-create ``get_function_at`` can no longer catch a typo/wrong
    address (#386). Reject the cheaply-detectable non-code cases:

    * an unaligned start on a fixed-width ISA (aarch64/MIPS/ARM) -- impossible for
      a real function start (uses the created function's own arch so Thumb's
      2-byte alignment is honored, not the view default);
    * a start BN can't decode an instruction at (``get_instruction_length`` == 0);
    * an empty body (``total_bytes`` == 0).

    A *valid-but-wrong* decode -- an x86 mid-instruction, or a jump-table/data word
    that happens to decode to a plausible instruction -- is NOT detectable cheaply
    (it is aligned, decodable, non-empty) and stays the caller's responsibility;
    the op's error notes this limitation."""
    arch = getattr(created, "arch", None) or getattr(bv, "arch", None)
    align = int(getattr(arch, "instr_alignment", 0) or 0)
    arch_name = getattr(arch, "name", "?")
    if align > 1 and (addr % align) != 0:
        return (f"address {hex(addr)} is not {align}-byte aligned for {arch_name}, "
                "so it cannot be a function start")
    try:
        ilen = bv.get_instruction_length(addr, arch)
    except Exception:
        ilen = None
    if ilen == 0:
        return f"no decodable instruction at {hex(addr)} (not code)"
    if getattr(created, "total_bytes", None) == 0:
        return f"the created function at {hex(addr)} has an empty body (not code)"
    return None


def _function_create_guard_message(addr: int, reason: str) -> str:
    return (
        f"Refused to keep a fabricated function at {hex(addr)}: {reason}. Forcing "
        "create_user_function past auto-analysis lands a function even on non-code; "
        "this guard catches an unaligned / undecodable / empty start. A data word "
        "that happens to decode to a plausible instruction is not detectable here -- "
        "verify the disassembly before trusting an aligned, decodable result."
    )


def _op_function_create(ctx, bv, op: dict[str, Any], restores: list | None = None, *,
                        defer_reanalysis: bool = False):
    """Create (and analyze) a function at an address inside a BATCH (#308).

    Unlike the standalone ``function create`` op (create_comments._function_create,
    which manages its own undo bracket), this runs inside the batch's single
    transaction: it creates + verifies, and registers a restore that removes the
    created function on revert. Function creation is not reliably undone by
    ``revert_undo_actions`` alone, so the restore uses the non-poisoning
    ``remove_function`` (#304) so a later create still works.
    read_misc / create_comments are imported locally to keep this module's
    one-way import direction (create_comments imports mutation_engine, not the
    reverse).

    #780 lock split: ``defer_reanalysis`` is the batch apply loop's request to run
    this op's reanalysis OUTSIDE the exclusive target lock, exactly as #628 did for
    the batch's post-apply reanalysis and for create_comments._function_create. The
    op then only creates (``create_user_function`` registers the function
    synchronously); the batch's single post-apply ``update_analysis_and_wait()``
    analyzes every created function at once, gate-only, and ``_verify_function_create``
    runs the post-analysis readback + #386 code guard. ``defer_reanalysis=False`` (the
    default) keeps the op self-contained -- create, analyze, read back and guard -- so
    direct engine calls and tests are unaffected, mirroring ``exclusive=None``."""
    from . import create_comments, read_misc

    addr = _parse_address(op["address"])
    requested = _operation_requested(ctx, op)
    # #479: on a --quick-loaded view, create_user_function + update_analysis_and_wait
    # below would run the full (multi-minute) analysis under the exclusive write lock
    # and wedge the instance (even later `target info` blocks). Refuse fast and point
    # at `bn refresh` instead of starting that analysis under the lock.
    if bv in _quick_loaded_views:
        raise OperationFailure(
            "invalid_request",
            "Cannot create a function: this target was loaded with --quick (no "
            "analysis). Creating a function would run the full analysis under the "
            "write lock and wedge the instance. Run `bn refresh` first.",
            requested=requested,
        )
    # M-03: REFUSED unless the op says `allow_overlap: true`, then disclosed
    # -- the same rule, in the same words, as the standalone command
    # (`create_comments._overlap_refusal`), so an agent that learned one
    # surface is not surprised by the other. The deliberate split (BN
    # over-extended the host) stays reachable through the flag. Validated
    # BEFORE the existing-start noop below (#927 review): a malformed field
    # is malformed whether or not a function already starts here, and the
    # standalone binder validates before its handler runs.
    allow_overlap = _validate_bool(op.get("allow_overlap"), label="allow_overlap",
                                   default=False)
    existing = bv.get_function_at(addr)
    if existing is not None:
        return {
            "op": "function_create",
            "status": "noop",
            "address": hex(addr),
            "function": str(existing.name),
            "message": "A function already starts at this address.",
            "requested": requested,
        }
    # #675 item 14: capture the CONTAINING functions before the create, while
    # the answer still exists -- afterwards this address IS a function start
    # and `get_functions_containing` reports the new one. A create at a
    # mid-function address succeeded with `verified` and said nothing, so a
    # caller who mistyped an address, or who did not realise BN had already
    # attributed those bytes, was left with a truncated host and no trace of
    # it in the result. Inside a batch an EARLIER create's function is a
    # possible host too: `_mutation` analyzes between two creates (gate-only)
    # so that function has its ranges by the time this probe runs (#927 P2).
    overlapped = create_comments._containing_function_rows(bv, addr, requested=requested)
    if len(bytes(bv.read(addr, 1))) == 0:
        raise OperationFailure(
            "invalid_request",
            f"Cannot create function: address {hex(addr)} is not mapped",
            requested=requested,
        )
    if not read_misc._is_executable_address(ctx, bv, addr):
        raise OperationFailure(
            "invalid_request",
            f"Cannot create function: address {hex(addr)} is not inside an executable segment",
            requested=requested,
        )
    if overlapped and not allow_overlap:
        raise create_comments._overlap_refusal(
            addr, overlapped, requested, flag='"allow_overlap": true on the op')
    # #927 P1: record the hosts a deliberate split will cut, so the restore
    # below can verify -- and repair -- them rather than assume removing the
    # created function put them back.
    hosts = create_comments._snapshot_split_hosts(bv, overlapped) if overlapped else {"hosts": {}}
    fallthrough_from = create_comments._create_fallthrough_source(bv, addr)
    # create_user_function (FORCED), not add_function (an advisory auto hint that
    # declines exactly the auto-skipped data-table / missed-handler addresses this
    # op exists to recover -- #360). Mirrors the standalone create_comments path.
    # BN registers the user function synchronously, so its return value names the
    # created function even before any analysis runs.
    created = bv.create_user_function(addr)
    if defer_reanalysis:
        # #780: no reanalysis here -- the batch's post-apply
        # update_analysis_and_wait() runs once for the whole batch, outside the
        # exclusive lock, and _verify_function_create then does the post-analysis
        # readback (is the function still there?) and the #386 code guard. Fall
        # back to the view lookup if this BN's create_user_function returns None.
        if created is None:
            created = bv.get_function_at(addr)
    else:
        bv.update_analysis_and_wait()
        created = bv.get_function_at(addr)
        if created is None:
            # BN's analysis declined to keep a function here; drop the stray
            # (non-poisoning remove_function) before failing the op.
            create_comments._remove_created_function(ctx, bv, addr)
            raise OperationFailure(
                "verification_failed",
                f"No function starts at {hex(addr)} after analysis.",
                requested=requested,
                observed={"address": hex(addr), "function": None},
            )
    # Register the removal of this created function as a batch restore BEFORE the
    # code guard runs, so a guard REJECTION is covered by the standard batch
    # rollback accounting rather than a fire-and-forget cleanup whose failure is
    # discarded (#520). create_user_function is not reliably undone by
    # revert_undo_actions, so this explicit removal (non-poisoning
    # remove_function, #304) is the only real revert. The restore RAISES when
    # removal fails so _run_local_restores marks restore_ok=False and the batch
    # reports rolled_back=false instead of claiming a clean rollback while the
    # fabricated function still persists in the view. The same raise covers a
    # host of a deliberate split whose ranges are not back as recorded (#927
    # P1): restores run newest-first, so a later create inside THIS one is
    # already gone when this verifies its hosts. The hosts' NAMES are not this
    # restore's to write -- `hosts` was snapshotted after the earlier siblings
    # applied, and the batch's undo has already put the journaled state back
    # by the time restores run -- so `_mutation` checks them against the
    # pre-transaction baseline instead (#927 round 3).
    def _remove_created_restore():
        if not create_comments._remove_created_function(ctx, bv, addr):
            raise RuntimeError(
                f"failed to remove fabricated function at {hex(addr)} on revert"
            )
        if not create_comments._restore_split_hosts(ctx, bv, addr, hosts):
            raise RuntimeError(
                f"the function(s) a create at {hex(addr)} split were not restored on revert"
            )

    if restores is not None:
        restores.append(_remove_created_restore)

    if not defer_reanalysis:
        reason = _function_looks_like_code(bv, created, addr)
        if reason is not None:
            # The forced create landed a junk function on non-code; drop it
            # (non-poisoning remove_function) and fail honestly instead of reporting
            # the fabricated function verified (#386). The removal boolean is not
            # discarded: the restore registered above re-checks it on batch rollback
            # and RAISES if the function still persists, so a failed cleanup is
            # reported as rolled_back=false rather than a silent clean rollback with
            # the fabricated function still live (#520).
            create_comments._remove_created_function(ctx, bv, addr)
            raise OperationFailure(
                "verification_failed",
                _function_create_guard_message(addr, reason),
                requested=requested,
                observed={"address": hex(addr), "function": str(created.name)},
            )
    result = {
        "op": "function_create",
        "status": "verified",
        "address": hex(addr),
        "function": str(created.name) if created is not None else None,
        "requested": requested,
    }
    if not defer_reanalysis:
        # Analyzed here, so the hosts' fate is observable here; a deferred op
        # is stamped by `_verify_function_create` after the batch's analysis.
        overlapped = create_comments._observe_split(bv, addr, overlapped)
    # ONE builder for both create paths (the other is
    # `create_comments._function_create`), so the two cannot state the same
    # fact differently.
    return create_comments._with_fallthrough_note(
        create_comments._with_overlap_note(result, overlapped), fallthrough_from)


def _verify_function_create(ctx, bv, result: dict[str, Any]) -> dict[str, Any]:
    item = dict(result)
    if item.get("status") == "noop":
        return item
    addr = _parse_address(item["address"])
    fn = bv.get_function_at(addr)
    item["observed"] = {"address": hex(addr), "function": str(fn.name) if fn is not None else None}
    if fn is None:
        # The batch's post-apply reanalysis dropped the function (a borderline
        # address BN won't keep): fail honestly instead of reporting verified.
        raise OperationFailure(
            "verification_failed",
            f"No function starts at {hex(addr)} after analysis.",
            requested=item.get("requested"),
            observed=item["observed"],
        )
    # #780: the op's own reanalysis is deferred out of the exclusive lock to the
    # batch's post-apply one, so the #386 code guard runs HERE -- after that
    # reanalysis, and before the op can be reported verified. This is the same
    # check _op_function_create applies when it reanalyzes itself (``defer_reanalysis``
    # False); running it a second time for a self-analyzing op is a no-op.
    reason = _function_looks_like_code(bv, fn, addr)
    if reason is not None:
        # A junk function on non-code: fail the op. The batch's standard rollback
        # (the removal restore registered by _op_function_create) drops the
        # fabricated function -- same net result as the apply-time guard.
        raise OperationFailure(
            "verification_failed",
            _function_create_guard_message(addr, reason),
            requested=item.get("requested"),
            observed=item["observed"],
        )
    # Report the post-analysis name (the op's readback is post-analysis too when it
    # analyzes itself; a deferred op only saw the pre-analysis name).
    item["function"] = str(fn.name)
    item["status"] = "verified"
    overlaps = item.get("overlaps")
    if isinstance(overlaps, list) and overlaps:
        # A deliberate split's hosts are observable now that the batch has
        # analyzed: say which were truncated and which are intact, in the
        # same words the self-analyzing op and the standalone command use.
        from . import create_comments
        fallthrough_from = item.get("fallthrough_from")
        item = create_comments._with_overlap_note(
            item, create_comments._observe_split(bv, addr, overlaps))
        if isinstance(fallthrough_from, str):
            # The overlap note is rebuilt whole; put #997's clause back on it.
            item = create_comments._with_fallthrough_note(item, int(fallthrough_from, 16))
    return item


def _class_apply_types_plan(ctx, selector: str | None, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """#993 item 3: the batch `class apply-types` runs, and what it covers.

    Typing a class for named virtual calls took one declaration and one
    `local retype` per method, by hand: the instance struct with a vptr typed as
    BN's own ``<Base>::<Class>::VTable*`` (the table type sits on the data
    variable at the vtable's slot array), then `this` on every method. This
    builds exactly those ops for the ordinary engine to apply, verify and, under
    `--preview` or on a failure, revert as one mutation:

    * the instance struct is DECLARED when the view has none (sized from the
      class card's ``size`` when one was recovered, padding after the vptr),
      or its offset-0 member is retyped when it exists, keeping that member's
      name; nothing when it already holds the typed vptr;
    * `this` is retyped on each of the class's own methods that has a parameter
      of that name. A thunk, or a method without one (a static member, or a
      stripped build where BN did not name it), is skipped and listed, never
      guessed at. Each skipped row carries a `bn proto get` check (#1049).

    Refuses when the class has no primary vtable or BN defined no VTable type
    for it: then there is no table type for the vptr to point at."""
    from . import read_class
    card = read_class._class_show(ctx, selector, name)
    requested = {"op": "class_apply_types", "name": name}
    if card.get("ambiguous"):
        names = ", ".join(str(m.get("name")) for m in card.get("matches") or [])
        raise OperationFailure(
            "invalid_request",
            f"{name!r} names more than one class ({names}); pass the full name of one.",
            requested=requested,
        )
    cls = str(card.get("name") or name)
    bv = ctx._resolve_view(selector)
    vtable = card.get("vtable") or {}
    if not vtable.get("address"):
        raise OperationFailure(
            "invalid_request",
            f"class {cls} has no primary vtable bn could locate, so there is no vptr to type. "
            "Declare the instance struct with `bn types declare` and retype `this` with "
            "`bn local retype <function> this '<type>'`.",
            requested=requested,
        )
    ptr = int(getattr(bv, "address_size", 8) or 8)
    slot_start = int(str(vtable["address"]), 16) + 2 * ptr
    vtable_type = read_class._bn_vtable_type_name(bv, slot_start)
    instance = _lookup_named_type(bv, cls)
    if vtable_type is None:
        type_names = sorted(_view_type_names(bv) or ())
        if instance is None:
            # The struct-show note names a derived class's table when BN made
            # only those (an abstract base), with the caveat that borrowing one
            # is a choice this command does not make for the caller.
            note = ctx._class_vtable_note(cls, type_names)
        else:
            # #993 review: that note says the struct does not exist and offers
            # a declare that would replace it; this one is about the vptr only.
            derived = sorted(n for n in type_names
                             if n.startswith(f"{cls}::") and n.endswith("::VTable"))
            note = (f" BN made VTable types only for classes derived from {cls} "
                    f"({', '.join(derived[:3])}); their leading slots are {cls}'s virtuals, but "
                    f"their `this` types belong to the derived class. To borrow one, retype the "
                    f"vptr yourself: bn struct field set {cls} 0x0 <member> 'struct `{derived[0]}`*'."
                    if derived else "")
        # #1035.5: a view where BN made no VTable type for ANY class (its RTTI
        # analysis named nothing -- the non-PIE builds) says so, so the refusal
        # is not read as something about this one class.
        none_at_all = ("" if any(n.endswith("::VTable") for n in type_names) else
                       " This view has no `<Class>::VTable` type for any class: Binary Ninja's "
                       "RTTI analysis created none here.")
        raise OperationFailure(
            "invalid_request",
            f"Binary Ninja defined no `<Class>::VTable` type at {cls}'s slot array "
            f"({hex(slot_start)}), so there is no table type of its own for the vptr to point "
            "at." + none_at_all + (note or " Declare one with `bn types declare`, then type the "
                                   "vptr with `bn struct field set`."),
            requested=requested,
        )
    vptr_type = f"struct `{vtable_type}`*"
    operations: list[dict[str, Any]] = []
    size = card.get("size") or {}
    width = 0
    try:
        width = int(str(size.get("value")), 16) if size.get("value") else 0
    except ValueError:
        width = 0
    size_value = size.get("value") if width else None
    size_source = card.get("size_source")
    vptr_before: dict[str, str] | None = None
    if instance is None:
        body = f"struct `{vtable_type}`* vtable;"
        if width > ptr:
            body += f" uint8_t _unk{ptr:x}[{hex(width - ptr)}];"
        operations.append({"op": "types_declare", "declaration": f"struct `{cls}` {{ {body} }};"})
        struct_action = "declared"
    else:
        if _type_tag_kind(instance) not in ("struct", "class"):
            # #993 review round 2: a union also has a Structure type class, and
            # passed this check only to fail later in `local retype`.
            raise OperationFailure(
                "invalid_request",
                f"{cls} names a {str(instance)!r} type, not a struct, so it cannot carry a vptr.",
                requested=requested,
            )
        # #993 review: the vptr's 8 bytes may only replace ONE pointer-width
        # scalar at offset 0 (a `void* vtable`, a `uint64_t`); an embedded base,
        # a padding array, a union or two narrow fields there are layout the
        # write would destroy, so they are refused, not overwritten.
        occupying = [m for m in getattr(instance, "members", None) or []
                     if int(getattr(m, "offset", 0)) < ptr]
        first = occupying[0] if len(occupying) == 1 and int(getattr(occupying[0], "offset", -1)) == 0 else None
        if first is not None and _is_pointer_to(getattr(first, "type", None), vtable_type):
            struct_action = "unchanged"
        elif occupying and (first is None or not _replaceable_vptr_slot(bv, getattr(first, "type", None), ptr)):
            listed = ", ".join(
                f"{getattr(m, 'name', '') or '<unnamed>'} ({getattr(m, 'type', '')} @{hex(int(getattr(m, 'offset', 0)))})"
                for m in occupying)
            # An embedded base is typed with ITS class's own table, not this
            # class's (that is the borrowing the abstract-base refusal warns of).
            resolved, base = _resolve_typedefs(bv, getattr(first, "type", None) if first is not None else None)
            base_hint = ""
            if base and "Structure" in _type_class_name(resolved or object()):
                # Its own table exists iff BN named one `<base>::VTable` or
                # `<Root>::<base>::VTable`; an abstract base has none.
                own_table = any(n == f"{base}::VTable" or n.endswith(f"::{base}::VTable")
                                for n in _view_type_names(bv) or ())
                base_hint = (f"If {first.name} is an embedded base, type its vptr with "
                             f"`bn class apply-types {base}`. " if own_table else
                             f"If {first.name} is an embedded base, BN made no VTable type of "
                             f"{base}'s own, so type the vptr inside {base} by hand "
                             f"(`bn struct field set {base} 0x0 <member> '<type>'`). ")
            raise OperationFailure(
                "invalid_request",
                f"struct {cls} holds {listed} where the {ptr}-byte vptr goes, and writing the vptr "
                f"would replace it; only a single pointer-width pointer or integer member at 0x0 "
                f"(typedefs of one included) is retyped. {base_hint}To replace it anyway, which "
                f"removes it, run `bn struct field set {cls} 0x0 vtable '{vptr_type}'`.",
                requested=requested,
            )
        else:
            operations.append({
                "op": "struct_field_set", "struct_name": cls, "offset": "0x0",
                "field_name": str(getattr(first, "name", "") or "vtable") if first is not None else "vtable",
                "field_type": vptr_type, "overwrite_existing": True,
            })
            struct_action = "vptr_retyped" if first is not None else "vptr_added"
            if first is not None:
                vptr_before = {"field_name": str(getattr(first, "name", "")),
                               "field_type": str(getattr(first, "type", ""))}
    retyped: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    seen: set[int] = set()

    def skip(address: int, name: str, reason: str) -> None:
        # #1049: every skipped row carries the read to make before typing it
        # by hand; nothing here establishes what its parameters are.
        skipped.append({"address": hex(address), "name": name, "reason": reason,
                        "check": f"bn proto get {hex(address)}"})

    for method in card.get("methods") or []:
        try:
            address = int(str(method.get("address")), 16)
        except ValueError:
            continue
        if address in seen:
            continue
        seen.add(address)
        row = {"address": hex(address), "name": str(method.get("demangled") or method.get("mangled") or "")}
        if (method.get("thunk") or {}).get("is_candidate"):
            skip(address, row["name"], "thunk")
            continue
        fn = bv.get_function_at(address)
        params = list(getattr(fn, "parameter_vars", None) or []) if fn is not None else []
        if not any(str(getattr(var, "name", "")) == "this" for var in params):
            skip(address, row["name"], "no parameter named `this`")
            continue
        operations.append({"op": "local_retype", "function": hex(address), "variable": "this",
                           "new_type": f"struct `{cls}`*"})
        retyped.append(row)
    # #993 review: a vtable slot that is not one of the class's own methods --
    # inherited, or every slot of a stripped build, whose card has no methods
    # -- is not retyped (its `this` may be a base's), but it is listed.
    for slot in vtable.get("slots") or []:
        target = (slot or {}).get("method") or {}
        try:
            address = int(str(target.get("address")), 16)
        except ValueError:
            continue
        if address in seen:
            continue
        seen.add(address)
        skip(address, str(target.get("display_name") or target.get("name") or ""),
             "vtable slot that is not one of the class's own methods "
             "(inherited, or unnamed in a stripped build)")
    plan = {
        "class": cls,
        "vtable_type": vtable_type,
        "instance_struct": struct_action,
        "size": size_value,
        "size_source": size_source,
        "retyped_methods": retyped,
        "skipped_methods": skipped,
        **({"vptr_member_before": vptr_before} if vptr_before else {}),
    }
    return operations, plan


def _is_pointer_to(type_obj, name: str) -> bool:
    """Whether *type_obj* is exactly a pointer to the named type *name* (not a
    pointer to a pointer, an array of them, or the type by value)."""
    if type_obj is None or "Pointer" not in _type_class_name(type_obj):
        return False
    target = getattr(type_obj, "target", None)
    if target is None:
        return False
    if "NamedTypeReference" in _type_class_name(target):
        return str(getattr(target, "name", "") or "") == name
    registered = getattr(target, "registered_name", None)
    return registered is not None and str(getattr(registered, "name", "") or "") == name


def _resolve_typedefs(bv, type_obj):
    """*type_obj* with every typedef (named reference) followed, and the name
    of the last named type on the way (`HB` -> `struct svc::Handler` gives the
    struct and `svc::Handler`). The walk is keyed by name, so a cycle stops it
    and a long chain does not."""
    name = ""
    seen: set[str] = set()
    while type_obj is not None and "NamedTypeReference" in _type_class_name(type_obj):
        name = str(getattr(type_obj, "name", "") or "")
        if not name or name in seen:
            return None, name
        seen.add(name)
        type_obj = _lookup_named_type(bv, name)
    registered = getattr(type_obj, "registered_name", None) if type_obj is not None else None
    registered_name = str(getattr(registered, "name", "") or "") if registered is not None else ""
    return type_obj, registered_name or name


def _replaceable_vptr_slot(bv, type_obj, ptr: int) -> bool:
    """Whether an offset-0 member of this type may be retyped as the vptr: a
    pointer or integer exactly one pointer wide, through any typedefs (a
    `typedef void* opaque_ptr` member counts, #993 review round 2)."""
    type_obj, _ = _resolve_typedefs(bv, type_obj)
    try:
        width = int(getattr(type_obj, "width", 0) or 0)
    except (TypeError, ValueError):
        return False
    kind = _type_class_name(type_obj)
    return width == ptr and ("Pointer" in kind or "Integer" in kind)


def _with_class_apply_types_plan(result: Any, plan: dict[str, Any]) -> Any:
    """Attach *plan* to the batch result, each retyped method stamped with its
    own row's status (`verified`, or `noop` when `this` already had the type),
    so `this retyped:` counts what changed rather than what was planned."""
    if not isinstance(result, dict):
        return result
    rows = [row for row in result.get("results") or []
            if isinstance(row, dict) and row.get("op") == "local_retype"]
    plan = dict(plan)
    plan["retyped_methods"] = [
        {**method, "status": rows[index].get("status")} if index < len(rows) else dict(method)
        for index, method in enumerate(plan.get("retyped_methods") or [])
    ]
    result["class_apply_types"] = plan
    return result


def _class_apply_types_noop(plan: dict[str, Any], *, preview: bool) -> dict[str, Any]:
    """The result of a `class apply-types` with nothing left to write: the
    struct already holds the typed vptr and no method has a `this` to retype
    (a re-run on a stripped build). One `noop` row, like any all-noop batch,
    so it exits 0 rather than being refused (#993 review)."""
    return {
        "preview": preview,
        "success": True,
        "committed": not preview,
        "message": f"Nothing to apply: {plan.get('class')} is already typed.",
        "results": [{"op": "class_apply_types", "status": "noop",
                     "requested": {"op": "class_apply_types", "name": plan.get("class")}}],
        "affected_functions": [],
        "affected_types": [],
        "affected_summary": {"referenced": 0, "reflowed": 0},
        # Same convention as the engine: a preview reports its (empty) revert.
        "rolled_back": bool(preview),
        "class_apply_types": plan,
    }
