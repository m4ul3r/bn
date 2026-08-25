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
from types import SimpleNamespace
from typing import Any

try:
    import binaryninja as bn  # noqa: F401  (kept for parity / future use)
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format
from . import read_misc
from ._shared import (
    OperationFailure,
    _parse_address,
    _validate_count,
    is_auto_function_name,
    is_imported_function,
)
from .bridge_state import require_analysis, _quick_loaded_views


def _callee_variadic_hint(callee) -> dict[str, Any] | None:
    """A provenance-labeled hint when the callee is an imported variadic
    (printf/scanf-family) function (#558): HLIL callsite text can show only the
    fixed argument, so point at the argument-recovery views. Returns None for a
    non-variadic callee. Never asserts a finding -- it steers, it does not judge."""
    name = str(getattr(callee, "name", "") or "")
    family = il_format._variadic_format_family(name)
    if family is None and not il_format._function_is_variadic(callee):
        return None
    fmt_index = family[0] if family is not None else None
    is_scanf = bool(family[1]) if family is not None else False
    return {
        "name": il_format._normalize_libc_name(name),
        "is_variadic": True,
        "family": "scanf" if is_scanf else ("printf" if family is not None else None),
        "format_arg_index": fmt_index,
        "note": (
            "callee is an imported variadic function; the HLIL statement may show only "
            "the fixed argument(s) even when ABI setup supplied a format string and "
            "additional arguments. Run `bn evidence function <caller>` for the format "
            "string, destination pointers, and raw ABI argument candidates, or inspect "
            "`bn disasm <caller> --linear`."
        ),
    }


def _callsites_within_function(ctx, bv, callee, func, *, context: int,
                               stub_addrs: frozenset[int] = frozenset(),
                               variadic_hint: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    func_arch = getattr(func, "arch", None)
    disasm_entries = il_format._structured_disasm_entries(bv, func)
    index_by_addr = {
        int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
    }
    callee_address = int(callee.start)
    # An exported function's intra-lib callers reach it through a same-name PLT
    # stub; treat a call to the stub as a call to the callee so `callsites
    # --within` sees through it (#286), mirroring the xrefs union.
    callee_addresses = {callee_address} | {int(a) for a in stub_addrs}
    # Align callsites' edge set with xrefs. The LLIL `dest` is a literal const
    # only on statically-resolved calls; on stripped/kernel/register-resolved
    # calls BN records the edge in the code-ref DB (the same source xrefs reads)
    # while the LLIL dest is a register/computed value. Union the two so
    # callsites never silently drops an edge xrefs/dataflow-callgraph confirm.
    # A code-ref addr is specific to THIS callee (and its stub) and we only
    # inspect this function's call insns, so matching on it stays correctly scoped.
    _get_code_refs = getattr(bv, "get_code_refs", None)
    code_ref_addrs: set[int] = set()
    if callable(_get_code_refs):
        for target in callee_addresses:
            code_ref_addrs |= {int(getattr(ref, "address", -1)) for ref in _get_code_refs(target)}
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
        if dest_value not in callee_addresses and call_addr not in code_ref_addrs:
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
        # #557: when the HLIL statement can't be localized, expose a stable
        # machine-readable reason code alongside the null so an agent knows WHY
        # (e.g. an ambiguous BN call-fold) instead of re-running decompile and
        # correlating addresses by hand. Null the reason when a statement is present.
        hlil_statement, hlil_reason = il_format._hlil_statement_localization(insn)
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
                "hlil_statement": hlil_statement,
                "hlil_statement_reason": hlil_reason,
                "pre_branch_condition": il_format._hlil_pre_branch_condition(insn),
            }
        )
        if variadic_hint is not None:
            rows[-1]["callee_variadic"] = variadic_hint
    rows.sort(key=lambda item: int(item["call_addr"], 16))
    return rows


def _all_caller_functions(
    bv,
    callee_addresses: set[int],
) -> list[tuple[str, Any]]:
    callers: dict[int, Any] = {}
    get_code_refs = getattr(bv, "get_code_refs", None)
    if not callable(get_code_refs):
        return []
    for address in sorted(callee_addresses):
        try:
            refs = list(get_code_refs(address) or [])
        except Exception:
            continue
        for ref in refs:
            functions = []
            direct = getattr(ref, "function", None)
            if direct is not None:
                functions = [direct]
            else:
                try:
                    functions = list(
                        bv.get_functions_containing(int(getattr(ref, "address")))
                        or []
                    )
                except Exception:
                    functions = []
            for function in functions:
                start = int(getattr(function, "start", -1))
                if start >= 0:
                    callers.setdefault(start, function)
    return [
        (str(getattr(function, "name", "") or hex(start)), function)
        for start, function in sorted(callers.items())
    ]


def _callsites(
    ctx,
    selector: str | None,
    callee_identifier: str,
    *,
    within_identifiers: list[Any],
    context: int = 3,
    offset: int = 0,
    limit: int | None = 100,
) -> dict[str, Any]:
    if context < 0:
        raise OperationFailure("invalid_context", f"Invalid callsite context size: {context}")
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)

    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Callsites")
    callee_symbol_only = False
    try:
        callee = ctx._find_function(bv, callee_identifier)
    except Exception:
        getter = getattr(bv, "get_symbols_by_name", None)
        symbols = (
            list(getter(str(callee_identifier)) or [])
            if callable(getter) and callee_identifier
            else []
        )
        if not symbols:
            raw_getter = getattr(bv, "get_symbol_by_raw_name", None)
            raw_symbol = (
                raw_getter(str(callee_identifier))
                if callable(raw_getter) and callee_identifier
                else None
            )
            if raw_symbol is not None:
                symbols = [raw_symbol]
        imported = []
        allowed_types = {
            getattr(getattr(bn, "SymbolType", None), name, None)
            for name in (
                "ImportedFunctionSymbol",
                "ImportedDataSymbol",
                "ImportAddressSymbol",
                "ExternalSymbol",
            )
        }
        for symbol in symbols:
            symbol_type = getattr(symbol, "type", None)
            type_name = str(getattr(symbol_type, "name", symbol_type))
            if symbol_type in allowed_types or type_name in {
                "ImportedFunctionSymbol",
                "ImportedDataSymbol",
                "ImportAddressSymbol",
                "ExternalSymbol",
            }:
                imported.append(symbol)
        if not imported:
            raise
        symbol = min(imported, key=lambda item: int(getattr(item, "address", 0)))
        callee = SimpleNamespace(
            name=str(
                getattr(symbol, "short_name", "")
                or getattr(symbol, "name", callee_identifier)
            ),
            start=int(getattr(symbol, "address", 0)),
        )
        callee_symbol_only = True
    # #286: an exported callee's intra-lib callers route through its same-name PLT
    # stub, so a call targeting the stub must count as a call to the callee.
    try:
        stub_addrs = frozenset(int(s.start) for s in ctx._same_name_stub_functions(bv, callee))
    except Exception:
        stub_addrs = frozenset()
    if within_identifiers:
        scope_functions = ctx._resolve_scope_functions(bv, within_identifiers)
    else:
        scope_functions = _all_caller_functions(
            bv, {int(callee.start), *stub_addrs}
        )
    # #558: an imported variadic (scanf/printf-family) callee's HLIL callsite text
    # can show only the fixed argument; attach a steer to the argument-recovery views.
    variadic_hint = _callee_variadic_hint(callee)

    rows = []
    callers_scanned = 0
    scan_truncated = False
    row_scan_target = offset + limit + 1 if limit is not None else None
    for scope_index, (within_query, func) in enumerate(scope_functions):
        function_rows = _callsites_within_function(
            ctx, bv, callee, func, context=context, stub_addrs=stub_addrs,
            variadic_hint=variadic_hint)
        callers_scanned += 1
        for call_index, row in enumerate(function_rows):
            row["call_index"] = call_index
            row["within_query"] = str(within_query)
        rows.extend(function_rows)
        if (
            row_scan_target is not None
            and len(rows) >= row_scan_target
            and scope_index + 1 < len(scope_functions)
        ):
            scan_truncated = True
            break

    if scan_truncated:
        assert limit is not None
        page = rows[offset:offset + limit]
        return {
            "kind": "callsites",
            "items": page,
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "total": None,
            "total_lower_bound": len(rows),
            "has_more": True,
            "scan_truncated": True,
            "callers_scanned": callers_scanned,
            "caller_total": len(scope_functions),
            "callee_symbol_only": callee_symbol_only,
        }

    result = read_misc._paged_list_result(
        rows, offset=offset, limit=limit, kind="callsites"
    )
    result.update(
        {
            "scan_truncated": False,
            "callers_scanned": callers_scanned,
            "caller_total": len(scope_functions),
            "callee_symbol_only": callee_symbol_only,
        }
    )
    return result


def _annotation_summary(ctx, bv) -> dict[str, Any]:
    """Count annotations ALREADY present in the view (#561).

    On a cached/shared BNDB, inherited comments/names can bias analysis and let
    an agent over-credit itself for state a prior run produced. Surface bounded
    counts so orientation discloses the inherited baseline. Cheap: global address
    comments come from the ``address_comments`` map; function-doc comments read one
    string attribute per function (the per-function address-comment map is NOT
    materialized, to keep this a fast triage read)."""
    comments = 0
    comment_locations: list[dict[str, Any]] = []
    try:
        address_comments = getattr(bv, "address_comments", None)
        if address_comments is not None:
            comments = len(address_comments)
            for address, text in list(address_comments.items())[:20]:
                comment_locations.append(
                    {
                        "address": hex(int(address)),
                        "comment": str(text)[:160],
                    }
                )
    except Exception:
        comments = 0
        comment_locations = []

    function_comments = 0
    function_comment_locations: list[dict[str, Any]] = []
    for fn in list(getattr(bv, "functions", []) or []):
        try:
            text = str(getattr(fn, "comment", "") or "").strip()
            if text:
                function_comments += 1
                if len(function_comment_locations) < 20:
                    function_comment_locations.append(
                        {
                            "name": str(getattr(fn, "name", "")),
                            "address": hex(int(getattr(fn, "start", 0))),
                            "comment": text[:160],
                        }
                    )
        except Exception:
            continue

    user_symbols = 0
    user_symbol_locations: list[dict[str, Any]] = []
    try:
        getter = getattr(bv, "get_symbols", None)
        symbols = getter() if callable(getter) else list(getattr(bv, "symbols", []) or [])
        for symbol in symbols:
            if getattr(symbol, "auto", None) is False:
                user_symbols += 1
                if len(user_symbol_locations) < 20:
                    user_symbol_locations.append(
                        {
                            "name": str(
                                getattr(symbol, "raw_name", "")
                                or getattr(symbol, "name", "")
                            ),
                            "address": hex(int(getattr(symbol, "address", 0))),
                        }
                    )
    except Exception:
        user_symbols = 0
        user_symbol_locations = []

    return {
        "comments": comments,
        "comment_locations": comment_locations,
        "function_comments": function_comments,
        "function_comment_locations": function_comment_locations,
        "user_symbols": user_symbols,
        "user_symbol_locations": user_symbol_locations,
        "locations_truncated": any(
            count > len(locations)
            for count, locations in (
                (comments, comment_locations),
                (function_comments, function_comment_locations),
                (user_symbols, user_symbol_locations),
            )
        ),
    }


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


def _analysis_state_fields(bv: Any) -> dict[str, Any]:
    """Envelope fields disclosing whether *bv* is quick-loaded (partial) or fully
    analyzed. A ``--quick`` function count is partial, but the ``functions``
    envelope looked complete ({count, total}); thread the same signal the bridge
    already derives for ``target info`` / the orient digest through the listing
    paths so a partial count is never mistaken for the whole binary (#437)."""
    quick = bv in _quick_loaded_views
    return {"analysis_state": "quick" if quick else "full", "partial": quick}


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
    min_size: Any = None,
    offset: int = 0,
    limit: int | None = None,
    count_only: bool = False,
    sort: str = "address",
    reverse: bool = False,
    named: bool | None = None,
):
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    min_size = _validate_count(min_size, label="min_size", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    functions = list(_filtered_functions(ctx, bv, min_address=min_address, max_address=max_address))
    if min_size is not None:
        # #446: drop tiny PLT/GOT thunk veneers (typically <= 16 bytes) that
        # otherwise list under the same name as the real body.
        functions = [fn for fn in functions if (il_format._function_size(fn) or 0) >= min_size]
    if named is not None:
        # #653.4: "how much of this binary is still sub_*?" is THE sizing question on
        # a stripped target, and `function search --regex '^sub_' --count` has no
        # negation -- three agents dumped the full list and post-processed it with
        # jq/python instead. Partitioned exactly like `target info`'s named /
        # auto-named / imported summary (one shared predicate, so the two numbers
        # cannot disagree): import thunks are in NEITHER bucket, since their names
        # come from relocations rather than from analysis or a human.
        functions = [
            fn for fn in functions
            if not is_imported_function(fn)
            and (not is_auto_function_name(str(getattr(fn, "name", "") or ""))) == named
        ]
    if count_only:
        # `total` mirrors the list envelope's key for the same number; `count`
        # kept for back-compat.
        return {"kind": "functions", "count": len(functions), "total": len(functions),
                **_analysis_state_fields(bv)}
    # #411 established that per-page display projection (basic_block_count) must
    # not be computed for the whole filtered set. display_name (a per-function
    # symbol lookup) and size follow the same rule: neither is needed to build
    # the full set here -- display_name is never a sort key, and size is only
    # needed full-set for `--sort size`. Deferring them to the returned page
    # (via _project_page_fields) is the difference between a `function list
    # --limit 100` that demangles + sizes 100 functions and one that pays it for
    # all 24k. `_fn` carries the live Function to the page projection, then drops.
    include_size = sort == "size"  # sorting the FULL set by size needs it per-row
    items = [
        {
            "name": fn.name,
            "address": hex(fn.start),
            "raw_name": getattr(fn, "raw_name", fn.name),
            **({"size": il_format._function_size(fn)} if include_size else {}),
            # #653.4's `imported`/`auto_named` are page projections, NOT full-set
            # fields: `is_imported_function` is a per-function `fn.symbol` lookup,
            # the same cost #639 moved off the filtered set. Computing them here
            # would hand back most of that win. The --named/--unnamed FILTER above
            # reads the live Function directly, so it is unaffected.
            "_fn": fn,   # transient: page projection reads this, then drops it
        }
        for fn in functions
    ]
    _sort_function_items(items, sort, reverse)
    result = _paged_function_result(ctx, items, offset=offset, limit=limit)
    result.update(_analysis_state_fields(bv))
    return _project_page_fields(result)


def _project_page_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Compute the per-row DISPLAY projections for the returned page ONLY, then
    drop the transient `_fn`.

    #411 first moved basic_block_count here so a 24k-function list didn't
    materialize block lists for every filtered function. display_name (a
    per-function symbol/short_name lookup) and size (a `total_bytes`/basic-block
    read) are the same shape of cost and are moved here too: `_list_functions`
    no longer computes them for the whole filtered set, so a bounded page no
    longer pays a full-set projection (measured ~540ms -> ~70ms for a 100-row
    page on a ~6.5k-function target). Callers that genuinely need a field for
    the FULL set -- `function search` matches on display_name, and both paths
    sort/filter on size -- set it on the item before paging; those values are
    preserved here (this only fills what the page is missing), so no field is
    computed twice.
    """
    for it in result.get("items", []):
        fn = it.pop("_fn", None)
        if fn is None:
            # No live Function retained (defensive): keep the row well-formed
            # with the same keys every consumer expects.
            it.setdefault("display_name", it.get("name", ""))
            size = it.get("size")
            size_known = isinstance(size, int) and not isinstance(size, bool) and size >= 0
            it["size"] = size if size_known else 0
            it["size_known"] = size_known
            it.setdefault("imported", False)
            it.setdefault("auto_named", False)
            it.setdefault("basic_block_count", None)
            continue
        if "display_name" not in it:
            it["display_name"] = il_format._display_name(fn)
        size = it.get("size") if "size" in it else il_format._function_size(fn)
        size_known = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        it["size"] = size if size_known else 0
        it["size_known"] = size_known
        # #653.4: label the two partitions `target info` counts, so a listing is
        # self-describing (an import thunk is neither named nor auto-named).
        if "imported" not in it:
            it["imported"] = is_imported_function(fn)
        if "auto_named" not in it:
            it["auto_named"] = is_auto_function_name(str(getattr(fn, "name", "") or ""))
        # BN's Function exposes no basic_block_count attribute -- len(basic_blocks)
        # is the count (materializes the block list, but only for the returned page).
        # Guard the access (mirrors il_format._function_size): one problematic
        # function on the page must not fail the whole list/search request (#411).
        try:
            bbs = getattr(fn, "basic_blocks", None)
            it["basic_block_count"] = len(bbs) if bbs is not None else None
        except Exception:
            it["basic_block_count"] = None
    return result


def _paged_function_result(ctx, items: list[dict[str, Any]], *, offset: int,
                           limit: int | None, kind: str = "functions") -> dict[str, Any]:
    """Return a function-listing page WITH paging metadata.

    The CLI can't compute the true total itself -- it fetches a bounded page
    -- so the bridge, which has the full filtered set, returns total/offset/
    limit/returned/has_more alongside the page. This lets `function list`
    state the real total + remainder (text) and expose paging in JSON, the
    same honesty convention as evidence xrefs (#59). `kind` is the envelope
    discriminator (#275); `items` is the sole data container (the legacy
    `functions` alias was dropped in the #275 clean break)."""
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    return {
        "kind": kind,
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
    word: bool = False,
    min_address: Any = None,
    max_address: Any = None,
    min_size: Any = None,
    offset: int = 0,
    limit: int | None = None,
    count_only: bool = False,
    sort: str = "address",
    reverse: bool = False,
):
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
    min_size = _validate_count(min_size, label="min_size", minimum=1, allow_none=True)
    bv = ctx._resolve_view(selector)
    items = []
    if regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise OperationFailure("invalid_regex", f"Invalid function regex: {exc}") from exc

        def matches(name: str) -> bool:
            return bool(pattern.search(name))

    elif word:
        # #457: match the query as a whole IDENTIFIER TOKEN (word-boundary), so a
        # sink survey for `popen` hits `popen` / `popen@plt` but NOT the substring
        # false positives `zipOpenArchive` / `my_popen_wrapper`. Looser than
        # --exact (still finds `@plt`-decorated and parenthesized forms), tighter
        # than the default substring match.
        pattern = re.compile(r"\b" + re.escape(query) + r"\b", re.IGNORECASE)

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
                "_fn": fn,   # transient: enrich the returned page only (perf), then drop
            })
    if min_size is not None:
        # #446: drop tiny PLT/GOT thunk veneers so a `function search RFCOMM...`
        # doesn't return each export twice (16-byte veneer + real body).
        items = [it for it in items if (it.get("size") or 0) >= min_size]
    if count_only:
        # Mirror `_list_functions` count_only: `total` matches the list envelope
        # key, `count` kept for back-compat (#252). (`_fn` is never serialized
        # here -- only the returned page is enriched/cleaned below.)
        return {"kind": "functions", "count": len(items), "total": len(items),
                **_analysis_state_fields(bv)}
    _sort_function_items(items, sort, reverse)
    result = _paged_function_result(ctx, items, offset=offset, limit=limit)
    result.update(_analysis_state_fields(bv))
    return _project_page_fields(result)
