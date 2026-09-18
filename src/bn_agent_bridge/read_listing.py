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
from . import read_xrefs
from ._shared import (
    OperationFailure,
    _parse_address,
    _validate_count,
    is_auto_function_name,
    is_imported_function,
    is_placeholder_symbol_name,
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
        # #816: a recovered call whose address is absent from this function's
        # structured-disasm index (the block walk never decoded an entry here --
        # a decode hole, or a call the walk's block ranges do not cover) used to
        # drop the row outright, so `callsites` reported fewer sites than the
        # xrefs/dataflow-callgraph evidence it is supposed to agree with and
        # said nothing about why. The IDENTITY of the site (callee, caller,
        # addresses) is known and actionable whatever the disassembly sweep did,
        # so emit the row with a null context plus a machine-readable reason --
        # the `hlil_statement_reason` shape, for the same "localize or say why
        # not" policy.
        disasm_index = index_by_addr.get(call_addr)
        if disasm_index is None:
            previous: list[dict[str, Any]] = []
            next_instructions: list[dict[str, Any]] = []
            call_instruction: dict[str, Any] | None = None
            disasm_context_reason: str | None = "no_structured_disasm_entry"
        else:
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
            disasm_context_reason = None
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
                "disasm_context_reason": disasm_context_reason,
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


def _scan_caller_functions(
    ctx, bv, callee_addresses: set[int]
) -> tuple[list[tuple[str, Any]], bool, str | None]:
    """Callers of *callee_addresses* recovered by scanning function LLIL.

    #816: `_all_caller_functions` enumerates callers from BN's code-ref DB alone.
    For an IMPORTED callee BN recorded no code ref for, `xrefs` covers exactly that
    class with its own bounded LLIL call scan (#622) -- so enumerating from code
    refs alone makes `callsites <import>` report "no callers" for a callee `xrefs`
    reports a confirmed call to, which is the silent drop this op exists to avoid.
    Reuse that scan (same budgets, same reasons) instead of a second walk.

    Returns ``(scope_functions, truncated, note)``. *truncated* True means the
    enumeration is PARTIAL -- the scan stopped on its budget or could not read
    some LLIL -- so the caller set (and therefore every count below it) must
    never be presented as complete; *note* names which happened.
    """
    callers: dict[int, Any] = {}
    # The scan reports its hits as `caller_function` address/name pairs; the row
    # builder needs the function OBJECTS, and it walks the same `bv.functions` the
    # scan did, so index that instead of re-resolving each address.
    by_start = {
        int(getattr(function, "start", -1)): function
        for function in (getattr(bv, "functions", None) or [])
    }
    truncated = False
    notes: list[str] = []
    for address in sorted(callee_addresses):
        refs, scan_truncated, note = read_xrefs._scan_for_calls_to(ctx, bv, address)
        truncated = truncated or scan_truncated
        if note and note not in notes:
            notes.append(note)
        for ref in refs:
            caller = ref.get("caller_function") or {}
            try:
                start = int(str(caller.get("address")), 16)
            except (TypeError, ValueError):
                continue
            function = by_start.get(start)
            if function is not None:
                callers.setdefault(start, function)
    scope_functions = [
        (str(getattr(function, "name", "") or hex(start)), function)
        for start, function in sorted(callers.items())
    ]
    return scope_functions, truncated, "; ".join(notes) if notes else None


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
    caller_scan_truncated, caller_scan_note = False, None
    if within_identifiers:
        scope_functions = ctx._resolve_scope_functions(bv, within_identifiers)
    else:
        scope_functions = _all_caller_functions(
            bv, {int(callee.start), *stub_addrs}
        )
        if not scope_functions and callee_symbol_only:
            # #816: BN recorded no code ref to this imported callee, so the
            # code-ref enumeration above can only answer "no callers" -- which is
            # exactly the false certainty `xrefs` refuses to report for the same
            # callee (it falls back to its #622 LLIL call scan). Enumerate the same
            # way here, and carry the scan's truncation up so a partial caller list
            # can never read as "not called".
            scope_functions, caller_scan_truncated, caller_scan_note = (
                _scan_caller_functions(ctx, bv, {int(callee.start), *stub_addrs})
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

    # Producer side of the monotone `total` contract (#694 item 3): `total`
    # stays `null` here while the caller scan is capped, becomes the exact
    # count once a later page's scan completes without truncation, and never
    # regresses from an int back to `null` or to a different int. The client
    # page validators (`src/bn/client.py` and
    # `skills/bn-kernel/src/bn_kernel/__init__.py`) enforce that monotonicity
    # across pages of one collection.
    if caller_scan_truncated:
        # #816: the caller ENUMERATION itself is partial (the #622-style LLIL call
        # scan ran out of budget, or some IL could not be read), so every count
        # derived from the caller set is a LOWER BOUND and none of it may be
        # presented as complete -- same monotone-`total` contract as the row-scan
        # cap below (#694 item 3), with the reason named so text and JSON alike
        # disclose it. `has_more` stays a fact about THIS page (paging advances
        # through what was found) rather than a promise about the missing tail.
        result = read_misc._paged_list_result(
            rows, offset=offset, limit=limit, kind="callsites"
        )
        result.update(
            {
                "total": None,
                "total_lower_bound": len(rows),
                "scan_truncated": scan_truncated,
                "caller_scan_truncated": True,
                "caller_scan_note": caller_scan_note,
                "callers_scanned": callers_scanned,
                # How many callers EXIST is what the scan was deciding; when it
                # stopped early that number is unknown, not `len(scope_functions)`.
                "caller_total": None,
                "callee_symbol_only": callee_symbol_only,
            }
        )
        return result

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
            "caller_scan_truncated": caller_scan_truncated,
            "caller_scan_note": caller_scan_note,
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
            "caller_scan_truncated": caller_scan_truncated,
            "caller_scan_note": caller_scan_note,
            "callers_scanned": callers_scanned,
            "caller_total": len(scope_functions),
            "callee_symbol_only": callee_symbol_only,
        }
    )
    return result


#: Every collection of named entries `bv.debug_info` exposes that can surface as
#: a non-auto SYMBOL, with the name attributes each of them carries. Functions
#: were not enough: a `cc -g` program holding one `static volatile int` reported
#: that variable as inherited analyst work, because the exclusion walked
#: `functions` while the counting walks every non-auto symbol (#733 F2 review).
#: `DebugFunctionInfo` carries `short_name`/`full_name`/`raw_name` (on a C++
#: target they legitimately disagree -- qualified vs mangled -- so a symbol may
#: match any one of them); `DataVariableAndName` carries only `name`. Asking
#: every attribute of both is cheaper than remembering which shape has which.
_DEBUG_INFO_COLLECTIONS = ("functions", "data_variables")
_DEBUG_INFO_NAME_FIELDS = ("short_name", "full_name", "raw_name", "name")


def _debug_info_symbols(bv) -> frozenset[tuple[str, str]]:
    """``(name, address)`` for everything the view's IMPORTED DEBUG INFO named.

    Provenance, not a name shape. BN marks a plain ELF symtab/dynsym name
    ``auto=True`` -- the summary never sees those -- but a name its DWARF
    importer recovered arrives ``auto=False``, so a `cc -g` build of a
    three-function program reported two "analyst symbols" on a view nobody had
    touched and `bn_kernel.assert_unannotated` refused it (#733 F2 review).
    Those names are the binary's own, so they are classified as placeholders.

    Keyed on the name AND the address the importer reported, because a bare
    name is not provenance: an analyst who renames a second parser copy to
    ``parse_header`` -- a name the debug info supplied for a DIFFERENT function
    -- was credited to the loader and the gate certified the view clean, which
    is the fail-OPEN direction. Measured on a real `-g` build, BN's symbol
    address and the importer's address agree exactly for every recovered
    function and data variable (an imported entry such as ``printf`` reports
    address 0 and simply matches no symbol), so the pair costs nothing in
    recall.

    Empty for a view with no imported debug info, and for one that cannot be
    asked: an unreadable source must not reclassify analyst work as a
    placeholder, so failure here counts symbols as analyst work, the fail-closed
    direction for the contamination gate. Each collection is read
    independently, so one that raises cannot cost the other.

    Cost, measured: these are generators that materialize a type (and, for a
    function, a platform and its local variables) per entry, so a 4000-function
    `-g` build spends ~0.3s cold / ~0.1s warm building this set -- the bulk of
    this otherwise-fast triage read, and roughly linear beyond that. A stripped
    or non-`-g` target reports nothing here and pays nothing.
    """
    debug_info = None
    try:
        debug_info = getattr(bv, "debug_info", None)
    except Exception:
        return frozenset()
    if debug_info is None:
        return frozenset()
    named: set[tuple[str, str]] = set()
    for collection in _DEBUG_INFO_COLLECTIONS:
        try:
            entries = list(getattr(debug_info, collection, []) or [])
        except Exception:
            continue
        for entry in entries:
            address = _symbol_address_text(entry)
            if address is None:
                continue
            for attribute in _DEBUG_INFO_NAME_FIELDS:
                try:
                    value = getattr(entry, attribute, None)
                except Exception:
                    continue
                if isinstance(value, str) and value:
                    named.add((value, address))
    return frozenset(named)


def _symbol_address_text(symbol) -> str | None:
    """A symbol's address as ``0x`` text, or None when it cannot be read.

    Separate and guarded so a sample ROW degrades on an unreadable address
    while the COUNTS beside it stay exact -- those counts drive the
    contamination refusal in `bn_kernel.assert_unannotated`, and a count that
    collapsed to zero because one symbol's `address` raised reads as "this view
    is pristine" (#733 F2).
    """
    try:
        return hex(int(getattr(symbol, "address", 0)))
    except Exception:
        return None


def _annotation_summary(ctx, bv) -> dict[str, Any]:
    """Count annotations ALREADY present in the view (#561).

    On a cached/shared BNDB, inherited comments/names can bias analysis and let
    an agent over-credit itself for state a prior run produced. Surface counts
    and bounded annotation samples; symbol exclusions are uncapped so each has
    a reason. Address-comment counts include both the global map and each
    function's local map; function-doc comments have their own count."""
    comments = 0
    comment_locations: list[dict[str, Any]] = []
    try:
        address_comments = getattr(bv, "address_comments", None)
        if address_comments is not None:
            comments = len(address_comments)
            # #861: `list(address_comments.items())` materialises a VIEW of the
            # live global comment map, so it walks the collection BN may still be
            # annotating. Snapshot the whole map first, then walk the snapshot.
            for address, text in list(dict(address_comments).items())[:20]:
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
        # #861: the per-function map is live too -- `dict(getattr(func, "comments",
        # {}))` is the same snapshot the decompile lane takes of the identical
        # collection -- so walking the attribute directly raised
        # `RuntimeError: dictionary changed size during iteration` on a view
        # analysis was still annotating, and the enclosing handler turned that
        # into an `unavailable` marker on both `target info` and `evidence orient`.
        local_comments = dict(getattr(fn, "comments", {}) or {})
        comments += len(local_comments)
        for address, text in local_comments.items():
            if len(comment_locations) >= 20:
                break
            comment_locations.append(
                {
                    "name": str(getattr(fn, "name", "")),
                    "address": hex(int(address)),
                    "comment": str(text)[:160],
                }
            )
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
    # #733 F2: the raw non-auto count above includes the names BN's LOADERS
    # synthesize, so it is not a measure of analyst work. Split, losslessly:
    # `user_symbols` keeps its meaning and sampling order.
    analyst_symbols = 0
    placeholder_symbols = 0
    analyst_symbol_locations: list[dict[str, Any]] = []
    symbol_exclusions: list[dict[str, Any]] = []
    # The functions the binary's own debug info named. BN marks those
    # `auto=False`, so without this a `cc -g` build reported its own function
    # names as inherited analyst work (#733 F2 review).
    debug_info_symbols = _debug_info_symbols(bv)
    # A failure to ENUMERATE the symbols propagates: these counts drive
    # `bn_kernel.assert_unannotated`'s refusal, so a summary nobody could
    # measure, published as `analyst_symbols: 0`, certifies contaminated
    # benchmark data clean. `_orient_digest` already degrades an unreadable
    # view to an `unavailable` marker that the kernel's `_require_orient_digest`
    # refuses as a contract violation; swallowing here into zeros was what kept
    # that path from ever being reached (#733 F2 review). A PER-SYMBOL failure
    # is absorbed below instead, so one odd symbol cannot cost the whole read.
    getter = getattr(bv, "get_symbols", None)
    symbols = getter() if callable(getter) else list(getattr(bv, "symbols", []) or [])
    for symbol in symbols:
        try:
            is_auto = getattr(symbol, "auto", None)
        except Exception:
            # Unreadable provenance is counted, never skipped: a symbol dropped
            # here is a fabricated shortfall in the count the gate refuses on.
            is_auto = False
        if is_auto is not False:
            continue
        user_symbols += 1
        try:
            name = str(
                getattr(symbol, "raw_name", "")
                or getattr(symbol, "name", "")
                or ""
            )
        except Exception:
            name = ""     # not a placeholder shape -> counted as analyst work
        # The address is read for every non-auto symbol -- guarded, so an
        # unreadable one costs its own sample ROW and nothing else. It is
        # load-bearing for the classification, not just for the row: the
        # debug-info exclusion is keyed on `(name, address)` so an analyst
        # rename that reuses a name the debug info gave a DIFFERENT function
        # still counts as analyst work.
        address = _symbol_address_text(symbol)
        exclusion_reason = None
        if address is not None and (name, address) in debug_info_symbols:
            exclusion_reason = "debug_info"
        elif is_placeholder_symbol_name(name):
            exclusion_reason = "name_shape"
        if len(user_symbol_locations) < 20 and address is not None:
            user_symbol_locations.append({"name": name, "address": address})
        if exclusion_reason is not None:
            placeholder_symbols += 1
            symbol_exclusions.append(
                {"name": name, "address": address, "reason": exclusion_reason}
            )
        else:
            analyst_symbols += 1
            if len(analyst_symbol_locations) < 20 and address is not None:
                analyst_symbol_locations.append({"name": name, "address": address})

    return {
        "comments": comments,
        "comment_locations": comment_locations,
        "function_comments": function_comments,
        "function_comment_locations": function_comment_locations,
        "user_symbols": user_symbols,
        "user_symbol_locations": user_symbol_locations,
        "analyst_symbols": analyst_symbols,
        "placeholder_symbols": placeholder_symbols,
        "analyst_symbol_locations": analyst_symbol_locations,
        "symbol_exclusions": symbol_exclusions,
        "symbol_exclusion_limitations": (
            "name_shape is a heuristic, not provenance: analyst renames matching "
            "excluded name families may remain undetected. Internal symbol "
            "namespaces also occur on user renames and are not proof of origin. "
            "debug_info requires the imported name and address to match."
        ),
        # No fourth pair: `analyst_symbols <= user_symbols` and both samples cap
        # at 20, so whenever the analyst pair could report truncation the user
        # pair already does (#733 F2). Not because one sample contains the
        # other -- it does not, once more than 20 placeholders precede an
        # analyst row.
        "locations_truncated": any(
            count > len(locations)
            for count, locations in (
                (comments, comment_locations),
                (function_comments, function_comment_locations),
                (user_symbols, user_symbol_locations),
            )
        ),
    }


def _annotations_unavailable(exc: BaseException, *, filename: str = "") -> dict[str, Any]:
    """The degrade marker for annotation counts nobody could read (#733 F2/#793).

    ONE spelling, used both when the counts themselves fail and when the view
    cannot be resolved at all: an unreadable summary published as
    ``comments: 0`` certifies a contaminated view clean, and this marker is what
    the kernel's ``assert_unannotated`` refuses instead. No counts are claimed --
    a reader of this marker must not find an absent ``analyst_symbols`` and
    assume zero.
    """
    return {
        "unavailable": f"annotation counts unavailable: {exc}",
        "analysis_cache_restored": str(filename or "").endswith(".bndb"),
    }


def _existing_annotations(ctx, bv, *, filename: str = "") -> dict[str, Any]:
    """Counts + provenance hint for annotations ALREADY present in *bv* (#561).

    ONE builder for the two surfaces that answer "can I trust this view as
    pristine?" -- `target info` (#793) and the orient digest. #793 was filed on
    the two of them DISAGREEING: the digest published ``existing_annotations``
    (with ``analysis_cache_restored`` and ``provenance_hint``) while `target
    info` -- the command every agent runs first -- had no annotation key at all,
    so the same cached target read annotated on one surface and clean on the
    other. Both now publish this block under the same key, from here.

    The caller resolves *bv* itself, the way its own read path resolves it (the
    digest through the bridge shim its unit doubles patch, `target info` from the
    view it already holds); a resolution failure degrades to
    ``_annotations_unavailable`` in the caller. ``analysis_cache_restored`` is
    derived from *filename*: a ``.bndb`` carries the analysis cache, which is
    where inherited comments/names come from. ``provenance_hint`` is keyed on
    ANALYST work, not the raw non-auto count -- the loader's own placeholders
    made a pristine view hint that its entirely-current-run analysis may predate
    the run (#733 F2).
    """
    analysis_cache_restored = str(filename or "").endswith(".bndb")
    try:
        annotations = _annotation_summary(ctx, bv)
    except Exception as exc:
        return _annotations_unavailable(exc, filename=filename)
    total_annotations = (
        annotations["comments"] + annotations["function_comments"]
        + annotations["analyst_symbols"]
    )
    hint = None
    if analysis_cache_restored or total_annotations:
        hint = (
            f"existing BNDB annotations may predate this run: "
            f"{annotations['comments']} comment(s), "
            f"{annotations['function_comments']} function doc(s), "
            f"{annotations['analyst_symbols']} analyst symbol(s) already present "
            f"({annotations['placeholder_symbols']} loader placeholder(s) excluded)"
            + (" (analysis cache restored from a .bndb)" if analysis_cache_restored else "")
            + " -- do not over-credit current-run analysis"
        )
    return {
        **annotations,
        "analysis_cache_restored": analysis_cache_restored,
        "provenance_hint": hint,
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
    paths so a partial count is never mistaken for the whole binary (#437).

    The SAME shape is attached by the other read ops that answer on a quick view
    (decompile, evidence function, types, class list -- #820), so a consumer sees
    one contract for "this answer may be incomplete" across commands instead of a
    per-op spelling. Import it from here; never re-derive the fields."""
    quick = bv in _quick_loaded_views
    return {"analysis_state": "quick" if quick else "full", "partial": quick}


_FUNCTION_SORTS = ("address", "size", "name")


def _filtered_functions(
    ctx,
    bv,
    *,
    min_address: Any = None,
    max_address: Any = None,
) -> list[Any]:
    lower, upper = _parse_function_address_bounds(
        ctx, min_address, max_address
    )
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


def _extent_known(fn) -> bool:
    """Whether this record's extent can be read at all (#757 review)."""
    size = il_format._function_size(fn)
    return isinstance(size, int) and not isinstance(size, bool) and size >= 0


def _duplicate_extent_key(fn) -> tuple[int, int]:
    """Order two records that claim the SAME start address by extent (#757).

    Only consulted for a group whose every member has a readable extent
    (see `_collapse_duplicate_starts`), so the readable-size preference here is
    a tiebreak among comparable records, not a substitute for comparison.
    """
    size = il_format._function_size(fn)
    known = isinstance(size, int) and not isinstance(size, bool) and size >= 0
    return (1 if known else 0, size if known else -1)


def _collapse_duplicate_starts(functions: list[Any]) -> tuple[list[Any], int, int]:
    """Keep ONE record per start address, and count the addresses that had more.

    BN can hold several Function records for a single start address (an
    overlapping or duplicated definition), and their sizes DISAGREE while both
    rows assert ``size_known: true`` -- so a size-sorted triage or a "small
    function = stub" heuristic reads whichever record sorted first as fact, per
    address, with no round trip that could tell the two apart (#757). One
    address is one function here: the record with the LARGER extent is retained
    (the real body; the phantom is the smaller, stub-shaped one), and every
    address that had more than one record is reported so the collapse is
    disclosed rather than silent.

    Returns ``(kept, collapsed, unresolved)``. A group whose members are all
    sized collapses, and its larger extent wins. A group where any extent is
    UNREADABLE cannot be ordered by that rule at all, so it is left standing
    and counted in ``unresolved`` -- the issue's own second answer ("or report
    the conflict"), and the only option that cannot promote a phantom.

    Cheap by construction: addresses with a single record (every address on a
    well-formed target) are never sized -- the extent read happens only inside a
    group that actually collided. Ordering is preserved (the population arrives
    ``(start, name)``-ordered, so first-seen grouping is address order).
    """
    grouped: dict[object, list[Any]] = {}
    for fn in functions:
        key: object
        try:
            key = int(fn.start)
        except (AttributeError, TypeError, ValueError):
            # A record whose start cannot be read cannot be shown to be a
            # duplicate of ANYTHING, so it is keyed by identity and forms its own
            # group (unit fakes model no `start`; passing the record through
            # unchanged is the only answer that never invents a collapse).
            key = fn
        grouped.setdefault(key, []).append(fn)
    if len(grouped) == len(functions):
        return functions, 0, 0
    collapsed = 0
    unresolved = 0
    kept: list[Any] = []
    for group in grouped.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        if all(_extent_known(fn) for fn in group):
            collapsed += 1
            kept.append(max(group, key=_duplicate_extent_key))
            continue
        # "Keep the larger extent" is undefined when a record's extent cannot be
        # read at all: choosing the record that happens to state a size lets a
        # stub-shaped phantom outvote a real body the view would not size -- the
        # exact confusion #757 was filed for. The issue's other accepted answer
        # is "report the conflict", so the group is left intact and disclosed.
        unresolved += 1
        kept.extend(group)
    return kept, collapsed, unresolved


def _disclose_collapsed_starts(result: dict[str, Any], collapsed: int,
                               unresolved: int = 0) -> dict[str, Any]:
    """Attach the #757 duplicate-start counts, when there were any.

    ``duplicate_starts_collapsed`` counts addresses where the larger extent was
    kept; ``duplicate_starts_unresolved`` counts addresses left with MORE than
    one record because at least one extent could not be read, so the issue's
    rule could not be applied (see `_collapse_duplicate_starts`). Both are
    present only when non-zero, so the common envelope keeps the key set every
    consumer already parses (the ``got_collapsed`` / ``self_defined_excluded``
    convention in ``read_misc._imports``). A caller whose ``total`` lands below
    its own count of raw BN records can then tell why -- and whether the
    retained row carries the LARGER extent or the conflict was left standing.
    """
    if collapsed:
        result["duplicate_starts_collapsed"] = collapsed
    if unresolved:
        result["duplicate_starts_unresolved"] = unresolved
    return result


def _function_population_key(fn, sort: str, sizes: dict[int, Any]) -> Any:
    """The order key for *sort* read off the LIVE Function (#814) -- the same
    keys ``--sort`` used on a materialized row before it (``size`` or 0 for
    ``size``, the lowercased name for ``name``, the address otherwise).

    Ordering the population rather than the rows is sound because every base
    sort was a STABLE sort over rows built in ``_filtered_functions`` order
    (``(start, name)``): keying on the sort key alone reproduces that order,
    tie-break included. ``size`` is the one key that costs a per-function read
    (``il_format._function_size``: ``total_bytes`` or a basic-block walk); it is
    recorded in *sizes* so the page rows reuse it instead of paying twice, the
    way the full build handed each row its size. ``address``/``name`` read a
    field the filter pass already touched.
    """
    if sort == "size":
        size = il_format._function_size(fn)
        sizes[id(fn)] = size
        return size or 0
    if sort == "name":
        return str(getattr(fn, "name", "")).lower()
    return int(fn.start)


def _order_function_population(
    population: list[Any],
    sort: str,
    reverse: bool = False,
    function_of=None,
) -> dict[int, Any]:
    """Order address, size, or name ascending unless ``reverse`` is set.

    Sorts the filtered POPULATION in place -- live Functions, or tuples whose
    first element is one (*function_of* extracts it) -- and returns the sizes it
    had to read (``{id(fn): size}``; empty unless *sort* is ``size``). #814:
    ordering the population instead of the rows is what lets the row build, the
    display projection and the xref enrichment happen for the returned page
    only."""
    if sort not in _FUNCTION_SORTS:
        raise OperationFailure(
            "invalid_request",
            f"Invalid sort '{sort}'; choose one of {', '.join(_FUNCTION_SORTS)}",
        )
    sizes: dict[int, Any] = {}
    if sort == "address" and not reverse:
        # The population arrives in ``(start, name)`` order -- already the
        # address order, tie-break included -- so this is not a skip.
        return sizes
    get_fn = function_of or (lambda item: item)
    population.sort(
        key=lambda item: _function_population_key(get_fn(item), sort, sizes),
        reverse=reverse,
    )
    return sizes


_NO_SIZE = object()


def _function_list_row(fn, *, display_name: str | None = None, size: Any = _NO_SIZE) -> dict[str, Any]:
    """Build ONE ``function list`` / ``function search`` row (#814).

    Called once per RETURNED row, never per filtered function. Everything else a
    consumer sees on the row (``display_name``, ``size``, ``basic_block_count``,
    the #653.4 ``imported``/``auto_named`` labels) is filled by
    ``_project_page_fields`` for the page, so a bounded page never pays a
    population-wide projection. ``search`` passes the ``display_name`` it already
    computed while matching (it is a match key there), and ``--sort size`` passes
    the size the ordering pass already read; both otherwise stay deferred.
    """
    row = {
        "name": fn.name,
        "address": hex(fn.start),
        "raw_name": getattr(fn, "raw_name", fn.name),
        "_fn": fn,   # transient: page projection reads this, then drops it
    }
    if display_name is not None:
        row["display_name"] = display_name
    if size is not _NO_SIZE:
        row["size"] = size
    return row


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
    functions, collapsed_starts, unresolved_starts = _collapse_duplicate_starts(
        list(_filtered_functions(ctx, bv, min_address=min_address, max_address=max_address))
    )
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
        result = {"kind": "functions", "count": len(functions), "total": len(functions),
                  **_analysis_state_fields(bv)}
        return _disclose_collapsed_starts(result, collapsed_starts, unresolved_starts)
    # #411 established that per-page display projection (basic_block_count) must
    # not be computed for the whole filtered set. display_name (a per-function
    # symbol lookup) and size follow the same rule, and #814 extends it to the
    # row dicts themselves: the filtered population is ORDERED as live Functions
    # and a row is built only for the returned window, so a `function list
    # --limit 100` over a 50k-function target no longer materializes (and sorts)
    # 50k rows to hand back 100. `_fn` carries the live Function to the page
    # projection, then drops.
    sizes = _order_function_population(functions, sort, reverse)
    start, stop = read_misc._page_window(len(functions), offset=offset, limit=limit)
    items = [
        # #653.4's `imported`/`auto_named` are page projections, NOT full-set
        # fields: `is_imported_function` is a per-function `fn.symbol` lookup,
        # the same cost #639 moved off the filtered set. Computing them here
        # would hand back most of that win. The --named/--unnamed FILTER above
        # reads the live Function directly, so it is unaffected.
        _function_list_row(fn, size=sizes[id(fn)] if sort == "size" else _NO_SIZE)
        for fn in functions[start:stop]
    ]
    result = read_misc._paged_envelope(
        kind="functions", items=items, total=len(functions), offset=offset, limit=limit,
    )
    result.update(_analysis_state_fields(bv))
    return _project_page_fields(_disclose_collapsed_starts(result, collapsed_starts, unresolved_starts))


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

    Slices an already-materialized *items* list to the requested window; the
    #814 paths in ``_list_functions`` / ``_search_functions`` build rows for the
    window up front instead and share ``read_misc._paged_envelope`` directly,
    which keeps ONE envelope shape for both. The CLI can't compute the true
    total itself -- it fetches a bounded page -- so the bridge, which has the
    filtered population, returns total/offset/limit/returned/has_more alongside
    the page. This lets `function list` state the real total + remainder (text)
    and expose paging in JSON, the same honesty convention as evidence xrefs
    (#59). `kind` is the envelope discriminator (#275); `items` is the sole data
    container (the legacy `functions` alias was dropped in the #275 clean
    break)."""
    start, stop = read_misc._page_window(len(items), offset=offset, limit=limit)
    return read_misc._paged_envelope(
        kind=kind, items=items[start:stop], total=len(items), offset=offset, limit=limit,
    )


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

    matched: list[tuple[Any, str]] = []
    # #757: collapse the duplicate records BN can hold for one start address
    # BEFORE matching, so a phantom twin cannot match twice (under two conflicting
    # sizes) and reach the page.
    population, collapsed_starts, unresolved_starts = _collapse_duplicate_starts(
        list(_filtered_functions(ctx, bv, min_address=min_address, max_address=max_address))
    )
    for fn in population:
        # Match across name forms (mangled fn.name, demangled display_name, raw)
        # so a demangled C++ query finds a function BN named with the mangled
        # symbol -- the same greppability `--demangle` gives the listing (#196).
        display = il_format._display_name(fn)
        raw = str(getattr(fn, "raw_name", fn.name))
        if any(matches(str(form)) for form in (fn.name, display, raw) if form):
            # #814: retain the live Function plus the display name this match
            # already computed -- NOT a row. The row (and the size/block/label
            # projections behind it) is built for the returned page only, so a
            # `function search --limit 20` over a 50k-function target no longer
            # sizes and materializes every match.
            matched.append((fn, display))
    if min_size is not None:
        # #446: drop tiny PLT/GOT thunk veneers so a `function search RFCOMM...`
        # doesn't return each export twice (16-byte veneer + real body). size IS
        # the filter key here, so this pass still reads it per match (as it did
        # before #814); it is only deferred when nothing filters or sorts on it.
        matched = [
            (fn, display) for fn, display in matched
            if (il_format._function_size(fn) or 0) >= min_size
        ]
    if count_only:
        # Mirror `_list_functions` count_only: `total` matches the list envelope
        # key, `count` kept for back-compat (#252). (`_fn` is never serialized
        # here -- only the returned page is enriched/cleaned below.)
        result = {"kind": "functions", "count": len(matched), "total": len(matched),
                  **_analysis_state_fields(bv)}
        return _disclose_collapsed_starts(result, collapsed_starts, unresolved_starts)
    sizes = _order_function_population(matched, sort, reverse, function_of=lambda pair: pair[0])
    start, stop = read_misc._page_window(len(matched), offset=offset, limit=limit)
    items = [
        _function_list_row(
            fn,
            display_name=display,
            size=sizes[id(fn)] if sort == "size" else _NO_SIZE,
        )
        for fn, display in matched[start:stop]
    ]
    result = read_misc._paged_envelope(
        kind="functions", items=items, total=len(matched), offset=offset, limit=limit,
    )
    result.update(_analysis_state_fields(bv))
    return _project_page_fields(_disclose_collapsed_starts(result, collapsed_starts, unresolved_starts))
