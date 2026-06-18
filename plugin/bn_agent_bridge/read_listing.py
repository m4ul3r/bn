"""Function listing / search / paging + callsite analysis read handlers.

The function-listing, search, paging and callsite read-op cluster that used to
live on ``BinaryNinjaBridge`` moves here as module-level free functions, each
taking the ``BridgeContext`` seam (``ctx``) in place of ``self``.
``BinaryNinjaBridge`` keeps a thin delegating shim for every name the test
suite / op binders reference (``_callsites_within_function``, ``_callsites``,
``_parse_function_address_bounds``, ``_filtered_functions``, ``_list_functions``,
``_paged_function_result``, ``_search_functions``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``, ``_resolve_scope_functions``);
  * ``il_format`` -- the pure IL/HLIL/disasm renderers and iteration helpers
    (``_structured_disasm_entries``, ``_iter_llil_instructions``, ``_il_op_name``,
    ``_hlil_statement_text``, ``_hlil_pre_branch_condition``,
    ``_instruction_length``, ``_llil_constant_value``);
  * ``_shared`` -- module-free helpers (``_validate_count``, ``_parse_address``,
    ``OperationFailure``).

Import direction is one-way: this module imports ``il_format`` and ``_shared``
(plus stdlib + binaryninja). It NEVER imports ``bridge`` or ``seam`` -- those
import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

import re
from typing import Any

import binaryninja as bn  # noqa: F401  (kept for parity / future use)

from . import il_format
from . import read_misc
from ._shared import OperationFailure, _parse_address, _validate_count
from .bridge_state import require_analysis


def _callsites_within_function(ctx, bv, callee, func, *, context: int) -> list[dict[str, Any]]:
    func_arch = getattr(func, "arch", None)
    disasm_entries = il_format._structured_disasm_entries(bv, func)
    index_by_addr = {
        int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
    }
    callee_address = int(callee.start)
    # Align callsites' edge set with xrefs. The LLIL `dest` is a literal const
    # only on statically-resolved calls; on stripped/kernel/register-resolved
    # calls BN records the edge in the code-ref DB (the same source xrefs reads)
    # while the LLIL dest is a register/computed value. Union the two so
    # callsites never silently drops an edge xrefs/dataflow-callgraph confirm.
    # A code-ref addr is specific to THIS callee and we only inspect this
    # function's call insns, so matching on it stays correctly scoped.
    _get_code_refs = getattr(bv, "get_code_refs", None)
    code_ref_addrs = (
        {int(getattr(ref, "address", -1)) for ref in _get_code_refs(callee_address)}
        if callable(_get_code_refs)
        else set()
    )
    rows = []
    for insn in il_format._iter_llil_instructions(func):
        op_name = il_format._il_op_name(insn)
        # Count tail-branch references too (a `b`/branch into the sink rendered
        # as `return <addr>(...) __tailcall`), not just bl/blx -- xrefs and
        # taint backward already treat these as calls, so callsites must agree
        # or it silently misses a reachable sink during triage (#47).
        if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL"}:
            continue
        call_addr = int(getattr(insn, "address", 0))
        dest_value = il_format._llil_constant_value(getattr(insn, "dest", None))
        if dest_value != callee_address and call_addr not in code_ref_addrs:
            continue
        call_kind = "tailcall" if "TAILCALL" in op_name else "call"

        instruction_length = il_format._instruction_length(bv, call_addr, arch=func_arch)
        caller_static = call_addr + instruction_length
        disasm_index = index_by_addr.get(call_addr)
        if disasm_index is None:
            continue

        previous = [
            {
                "address": item["address"],
                "text": item["text"],
            }
            for item in disasm_entries[max(0, disasm_index - context) : disasm_index]
        ]
        next_instructions = [
            {
                "address": item["address"],
                "text": item["text"],
            }
            for item in disasm_entries[disasm_index + 1 : disasm_index + 1 + context]
        ]
        call_instruction = {
            "address": disasm_entries[disasm_index]["address"],
            "text": disasm_entries[disasm_index]["text"],
        }
        rows.append(
            {
                "callee": {
                    "name": str(callee.name),
                    "address": hex(callee_address),
                },
                "containing_function": {
                    "name": str(func.name),
                    "address": hex(int(func.start)),
                },
                "call_addr": hex(call_addr),
                "call_kind": call_kind,
                "instruction_length": instruction_length,
                "caller_static": hex(caller_static),
                "call_instruction": call_instruction,
                "previous_instructions": previous,
                "next_instructions": next_instructions,
                "hlil_statement": il_format._hlil_statement_text(insn),
                "pre_branch_condition": il_format._hlil_pre_branch_condition(insn),
            }
        )
    rows.sort(key=lambda item: int(item["call_addr"], 16))
    return rows


def _callsites(
    ctx,
    selector: str | None,
    callee_identifier: str,
    *,
    within_identifiers: list[Any],
    context: int = 3,
) -> list[dict[str, Any]]:
    if context < 0:
        raise OperationFailure("invalid_context", f"Invalid callsite context size: {context}")

    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Callsites")
    callee = ctx._find_function(bv, callee_identifier)
    scope_functions = ctx._resolve_scope_functions(bv, within_identifiers)

    rows = []
    for within_query, func in scope_functions:
        function_rows = _callsites_within_function(ctx, bv, callee, func, context=context)
        for call_index, row in enumerate(function_rows):
            row["call_index"] = call_index
            row["within_query"] = str(within_query)
        rows.extend(function_rows)
    # Honest paging envelope for JSON parity (#131 / item 11): callsites is a
    # flat row list, so wrap it like the sibling list ops. --limit stays a
    # text-only renderer cap (no bridge-side paging), hence offset=0/limit=None.
    return read_misc._paged_list_result(rows, offset=0, limit=None)


def _parse_function_address_bounds(
    ctx,
    min_address: Any = None,
    max_address: Any = None,
) -> tuple[int | None, int | None]:
    lower = _parse_address(min_address) if min_address not in (None, "") else None
    upper = _parse_address(max_address) if max_address not in (None, "") else None
    if lower is not None and upper is not None and lower > upper:
        raise OperationFailure(
            "invalid_address_range",
            f"Invalid function address range: {hex(lower)} is greater than {hex(upper)}",
        )
    return lower, upper


def _filtered_functions(
    ctx,
    bv,
    *,
    min_address: Any = None,
    max_address: Any = None,
) -> list[Any]:
    lower, upper = _parse_function_address_bounds(ctx, min_address, max_address)
    functions = []
    for fn in list(bv.functions):
        address = int(fn.start)
        if lower is not None and address < lower:
            continue
        if upper is not None and address > upper:
            continue
        functions.append(fn)
    functions.sort(key=lambda fn: (int(fn.start), fn.name))
    return functions


_FUNCTION_SORTS = ("address", "size", "name")


def _sort_function_items(items: list[dict[str, Any]], sort: str,
                         reverse: bool = False) -> list[dict[str, Any]]:
    """Order function-listing rows. 'address' (default) keeps the bridge's
    natural start-address order; 'size' ranks largest-first (the common
    'find the biggest function' triage step that otherwise needs a write-locked
    py exec); 'name' is case-insensitive. ``reverse`` flips the NATURAL order of
    the chosen sort, so e.g. ``--sort size --reverse`` surfaces the SMALLEST
    functions and ``--sort address --reverse`` walks high->low (#221)."""
    if sort not in _FUNCTION_SORTS:
        raise OperationFailure(
            "invalid_request",
            f"Invalid sort '{sort}'; choose one of {', '.join(_FUNCTION_SORTS)}",
        )
    if sort == "size":
        # natural = largest first; reverse -> smallest first
        items.sort(key=lambda it: (it.get("size") or 0), reverse=not reverse)
    elif sort == "name":
        items.sort(key=lambda it: str(it.get("name", "")).lower(), reverse=reverse)
    elif reverse:  # 'address': natural is the start-address order; flip it
        items.sort(key=lambda it: int(str(it.get("address", "0x0")), 16), reverse=True)
    return items


def _list_functions(
    ctx,
    selector: str | None,
    *,
    min_address: Any = None,
    max_address: Any = None,
    offset: int = 0,
    limit: int | None = None,
    count_only: bool = False,
    sort: str = "address",
    reverse: bool = False,
):
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    functions = list(_filtered_functions(ctx, bv, min_address=min_address, max_address=max_address))
    if count_only:
        # `total` mirrors the list envelope's key for the same number; `count`
        # kept for back-compat.
        return {"count": len(functions), "total": len(functions)}
    items = [
        {
            "name": fn.name,
            "address": hex(fn.start),
            "raw_name": getattr(fn, "raw_name", fn.name),
            "display_name": il_format._display_name(fn),
            "size": il_format._function_size(fn),
        }
        for fn in functions
    ]
    _sort_function_items(items, sort, reverse)
    return _paged_function_result(ctx, items, offset=offset, limit=limit)


def _paged_function_result(ctx, items: list[dict[str, Any]], *, offset: int,
                           limit: int | None) -> dict[str, Any]:
    """Return a function-listing page WITH paging metadata.

    The CLI can't compute the true total itself -- it fetches a bounded page
    -- so the bridge, which has the full filtered set, returns total/offset/
    limit/returned/has_more alongside the page. This lets `function list`
    state the real total + remainder (text) and expose paging in JSON, the
    same honesty convention as evidence xrefs (#59)."""
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    return {
        # DEPRECATED: `functions` duplicates `items` byte-for-byte, kept only for
        # back-compat since #139. `items` is the universal paged-array key
        # (imports/strings/sections/types/xrefs/comment-list/callsites all use
        # it) -- new consumers should read `items`; `functions` will be dropped on
        # the next breaking (feat(json)!) bump (#165).
        "functions": page,
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + len(page)) < total,
    }


def _search_functions(
    ctx,
    selector: str | None,
    query: str,
    *,
    regex: bool = False,
    exact: bool = False,
    min_address: Any = None,
    max_address: Any = None,
    offset: int = 0,
    limit: int | None = None,
    sort: str = "address",
    reverse: bool = False,
):
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    items = []
    if regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise OperationFailure("invalid_regex", f"Invalid function regex: {exc}") from exc

        def matches(name: str) -> bool:
            return bool(pattern.search(name))

    elif exact:
        needle = query.lower()

        def matches(name: str) -> bool:
            return name.lower() == needle

    else:
        needle = query.lower()

        def matches(name: str) -> bool:
            return needle in name.lower()

    for fn in _filtered_functions(ctx, bv, min_address=min_address, max_address=max_address):
        # Match across name forms (mangled fn.name, demangled display_name, raw)
        # so a demangled C++ query finds a function BN named with the mangled
        # symbol -- the same greppability `--demangle` gives the listing (#196).
        display = il_format._display_name(fn)
        raw = str(getattr(fn, "raw_name", fn.name))
        if any(matches(str(form)) for form in (fn.name, display, raw) if form):
            items.append({
                "name": fn.name,
                "address": hex(fn.start),
                "raw_name": raw,
                "display_name": display,
                "size": il_format._function_size(fn),
            })
    _sort_function_items(items, sort, reverse)
    return _paged_function_result(ctx, items, offset=offset, limit=limit)
