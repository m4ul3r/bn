"""Decompile / IL / disasm / prototype / locals / dataflow read handlers.

The decompile/IL/disasm read-op cluster that used to live on
``BinaryNinjaBridge`` moves here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_decompile``, ``_function_info``, ``_get_prototype``,
``_list_locals_for_function``, ``_il``, ``_disasm``, ``_structured_il``,
``_defuse``, ``_resolved_calls``, ``_possible_values``, ``_pvs_targets``,
``_force_function_analysis``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``, ``_functions_containing``);
  * ``il_format`` -- the pure IL/HLIL/disasm renderers and SSA/value-set
    helpers (``_decompile_text``, ``_render_warnings``, ``_analysis_stub_warning``,
    ``_comment_map``, ``_function_metadata``, ``_function_text``, ``_disasm_text``,
    ``_il_function_for``, ``_il_op_name``, ``_ssa_var_entry``,
    ``_resolve_ssa_variable``, ``_serialize_pvs``, ``_pvs_determined``);
  * ``vars`` -- variable discovery (``_list_locals``, ``_find_variable_selector``);
  * ``taint_engine`` -- call-target / function resolution used by the dataflow
    ops (imported as the SAME ``from . import taint_engine as _taint`` alias the
    bridge uses);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

from typing import Any

import binaryninja as bn  # noqa: F401  (kept for parity / future use)

from . import il_format
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address  # noqa: F401
from .bridge_state import require_analysis


def _force_function_analysis(ctx, bv, func):
    """Override a skipped function's analysis and reanalyze it in place.

    Mutates the BinaryView (skip override + reanalysis), so callers must hold
    the exclusive write lock. Returns the (possibly rebuilt) function object.
    """
    start = int(func.start)
    try:
        func.analysis_skipped = False  # setter installs NeverSkipFunctionAnalysis
    except Exception:
        pass
    try:
        func.reanalyze()
    except Exception:
        pass
    bv.update_analysis_and_wait()
    return bv.get_function_at(start) or func


def _decompile(ctx, selector: str | None, identifier, *, addresses: bool = False, force_analysis: bool = False):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    forced = False
    if force_analysis and bool(getattr(func, "analysis_skipped", False)):
        func = _force_function_analysis(ctx, bv, func)
        forced = True
    text = il_format._decompile_text(bv, func, addresses=addresses)
    warnings = il_format._render_warnings(text)
    stub = il_format._analysis_stub_warning(func, text, forced=forced)
    if stub:
        warnings.append(stub)
    comments = il_format._comment_map(bv, func)
    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": text,
        "comments": comments,
        "warnings": warnings,
        "analysis_skipped": bool(getattr(func, "analysis_skipped", False)),
        # `analysis_force_requested` echoes the --force-analysis flag; `analysis_forced`
        # is True only when a reanalysis actually ran this call. On a second forced
        # decompile the function is no longer skipped, so nothing reruns and
        # analysis_forced is False -- the echo tells callers the flag wasn't ignored.
        "analysis_force_requested": bool(force_analysis),
        "analysis_forced": forced,
    }


def _function_info(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Function info")
    func = ctx._find_function(bv, identifier)
    metadata = il_format._function_metadata(func)
    variables = vars_mod._list_locals(func)
    parameters = [item for item in variables if item["is_parameter"]]
    locals_only = [item for item in variables if not item["is_parameter"]]
    code_ref_count = len(list(bv.get_code_refs(func.start)))
    return {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
            "display_name": il_format._display_name(func),
        },
        **metadata,
        "parameters": parameters,
        "locals": locals_only,
        "xref_count": code_ref_count,
        # Count (+ addresses) of instructions BN's lifter could not model, so a
        # function with unlifted computation isn't mistaken for fully analyzed
        # (#206). count:0 means no unlifted instruction was found.
        "unimplemented_instructions": il_format._unimplemented_instructions(func),
    }


def _get_prototype(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    return {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        **il_format._function_metadata(func),
    }


def _list_locals_for_function(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    variables = vars_mod._list_locals(func)
    return {
        "function": {
            "name": func.name,
            "address": hex(func.start),
            "raw_name": getattr(func, "raw_name", func.name),
        },
        "locals": variables,
    }


def _il(ctx, selector: str | None, identifier, view: str, ssa: bool):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    text = il_format._function_text(bv, func, view=view, ssa=ssa)
    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "text": text,
        "warnings": il_format._render_warnings(text),
    }


def _disasm(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "text": il_format._disasm_text(bv, func),
    }


def _structured_il(ctx, selector, identifier, *, view: str = "mlil", ssa: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    il = il_format._il_function_for(func, view, ssa)
    instructions = []
    try:
        items = list(il.instructions)
    except Exception:
        items = []
    for ins in items:
        opn = il_format._il_op_name(ins)
        instructions.append({
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": opn,
            "text": str(ins),
            "vars_read": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_read", None) or [])],
            "vars_written": [il_format._ssa_var_entry(v) for v in (getattr(ins, "vars_written", None) or [])],
            "operands_summary": [str(o) for o in (getattr(ins, "operands", None) or [])],
            "is_call": "CALL" in opn,
        })
    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "view": view,
        "ssa": ssa,
        "instructions": instructions,
    }


def _defuse(ctx, selector, identifier, var_selector: str):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    il = il_format._il_function_for(func, "mlil", True)
    ssa_var, other_versions = il_format._resolve_ssa_variable(func, il, var_selector)

    try:
        definition = il.get_ssa_var_definition(ssa_var)
    except Exception:
        definition = None
    try:
        uses = list(il.get_ssa_var_uses(ssa_var) or [])
    except Exception:
        uses = []

    def _ref(ins):
        if ins is None:
            return None
        return {
            "il_index": int(getattr(ins, "instr_index", -1)),
            "address": hex(int(getattr(ins, "address", func.start))),
            "op": il_format._il_op_name(ins),
            "text": str(ins),
        }

    is_phi = definition is not None and "PHI" in il_format._il_op_name(definition)
    phi_sources = []
    if is_phi:
        for s in (getattr(definition, "src", None) or []):
            phi_sources.append(il_format._ssa_var_entry(s))

    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "variable": il_format._ssa_var_entry(ssa_var),
        "definition": _ref(definition),
        "uses": [_ref(u) for u in uses],
        "is_phi": is_phi,
        "phi_sources": phi_sources,
        "other_versions": other_versions or [],
    }


def _pvs_targets(ctx, bv, pvs) -> list[dict[str, Any]]:
    if pvs is None:
        return []
    type_name = getattr(getattr(pvs, "type", None), "name", "") or ""
    addrs: list[int] = []
    if type_name in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
        value = getattr(pvs, "value", None)
        if value is not None:
            try:
                addrs.append(int(value))
            except Exception:
                pass
    elif type_name == "InSetOfValues":
        for v in (getattr(pvs, "values", None) or []):
            try:
                addrs.append(int(v))
            except Exception:
                pass
    elif type_name == "LookupTableValue":
        mapping = getattr(pvs, "mapping", None)
        if isinstance(mapping, dict):
            for v in mapping.values():
                try:
                    addrs.append(int(v))
                except Exception:
                    pass
        else:
            for entry in (getattr(pvs, "table", None) or []):
                target = getattr(entry, "to", None)
                if target is not None:
                    try:
                        addrs.append(int(target))
                    except Exception:
                        pass
    targets = []
    for addr in addrs:
        # function_at normalizes the Thumb low bit so an odd value-set target
        # resolves to its function (#89 Problem B). The raw address is kept
        # in the reported "address".
        fn = _taint.function_at(bv, addr)
        name = None
        if fn is not None:
            name = str(getattr(fn, "name", None))
        else:
            sym = bv.get_symbol_at(addr) if hasattr(bv, "get_symbol_at") else None
            if sym is None and (addr & 1) and hasattr(bv, "get_symbol_at"):
                sym = bv.get_symbol_at(addr & ~1)
            name = str(getattr(sym, "name", None)) if sym is not None else None
        targets.append({"address": hex(addr), "name": name})
    return targets


def _resolved_calls(ctx, selector, identifier, *, direction: str = "both", resolve_indirect: bool = True):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    result: dict[str, Any] = {"function": {"name": func.name, "address": hex(func.start)}}

    if direction in ("callees", "both"):
        il = il_format._il_function_for(func, "mlil", True)
        callees = []
        try:
            items = list(il.instructions)
        except Exception:
            items = []
        for ins in items:
            opn = il_format._il_op_name(ins)
            if "CALL" not in opn and "TAILCALL" not in opn:
                continue
            # resolve_call_target (no thunk following) resolves MLIL_IMPORT
            # calls and Thumb-tagged targets that const_target/exact-address
            # lookup miss (#89). Constant direct calls resolve identically.
            resolved = _taint.resolve_call_target(bv, ins, follow_thunks=False)
            target = resolved.address
            row = {
                "call_addr": hex(int(getattr(ins, "address", func.start))),
                "il_index": int(getattr(ins, "instr_index", -1)),
            }
            if target is not None:
                fn = resolved.function or (bv.get_function_at(target) if hasattr(bv, "get_function_at") else None)
                name = str(fn.name) if fn is not None else None
                if name is None:
                    sym = bv.get_symbol_at(target) if hasattr(bv, "get_symbol_at") else None
                    name = str(sym.name) if sym is not None else None
                row.update({"kind": "direct", "target": {"address": hex(target), "name": name}})
            else:
                row.update({"kind": "indirect", "dest_expr": str(getattr(ins, "dest", ""))})
                if resolve_indirect:
                    pvs = getattr(getattr(ins, "dest", None), "possible_values", None)
                    resolved = _pvs_targets(ctx, bv, pvs)
                    row["resolved"] = resolved
                    type_name = getattr(getattr(pvs, "type", None), "name", "") or "none"
                    row["resolution"] = "possible_values" if resolved else "unresolved"
                    row["resolution_detail"] = type_name
            callees.append(row)
        result["callees"] = callees

    if direction in ("callers", "both"):
        callers = []
        seen: set[int] = set()
        for site in (list(getattr(func, "caller_sites", None) or [])):
            addr = int(getattr(site, "address", 0))
            fn = getattr(site, "function", None)
            if fn is None:
                functions = ctx._functions_containing(bv, addr)
                fn = functions[0] if functions else None
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)} if fn is not None else None
            )
            callers.append({"call_addr": hex(addr), "caller": caller})
        if not callers:
            for fn in (list(getattr(func, "callers", None) or [])):
                marker = int(getattr(fn, "start", 0))
                if marker in seen:
                    continue
                seen.add(marker)
                callers.append({"caller": {"address": hex(marker), "name": str(fn.name)}})
        result["callers"] = callers

    return result


def _possible_values(ctx, selector, identifier, at):
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    address = _parse_address(at)
    il = il_format._il_function_for(func, "mlil", True)
    target_ins = None
    try:
        for ins in list(il.instructions):
            if int(getattr(ins, "address", -1)) == address:
                target_ins = ins
                break
    except Exception:
        target_ins = None
    instr_pvs = getattr(target_ins, "possible_values", None) if target_ins is not None else None
    src_expr = getattr(target_ins, "src", None) if target_ins is not None else None
    src_pvs = getattr(src_expr, "possible_values", None) if src_expr is not None else None
    # BN leaves a SET_VAR/STORE *instruction* value-set undetermined while the
    # SOURCE expression (the value being assigned) carries the real value-set
    # -- const / range / lookup-table. Report the source's value-set for an
    # assignment so values surface as the help promises, instead of always
    # printing UndeterminedValue; fall back to the instruction-level set when
    # there is no source or only the instruction-level set is determined (#52).
    if il_format._pvs_determined(src_pvs) and not il_format._pvs_determined(instr_pvs):
        chosen, basis = src_pvs, "source_expression"
    elif il_format._pvs_determined(instr_pvs):
        chosen, basis = instr_pvs, "instruction"
    elif src_pvs is not None:
        chosen, basis = src_pvs, "source_expression"
    else:
        chosen, basis = instr_pvs, "instruction"
    return {
        "function": {"name": func.name, "address": hex(func.start)},
        "at": hex(address),
        "expression": str(target_ins) if target_ins is not None else None,
        "value_basis": basis,
        "source_expression": str(src_expr) if src_expr is not None else None,
        "possible_values": il_format._serialize_pvs(chosen),
    }
