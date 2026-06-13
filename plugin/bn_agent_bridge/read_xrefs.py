"""Cross-reference resolution: code/data xrefs, import-symbol xrefs, field xrefs.

The xref-resolution methods that used to live on ``BinaryNinjaBridge`` move here
as module-level free functions, each taking the ``BridgeContext`` seam (``ctx``)
in place of ``self``. ``BinaryNinjaBridge`` keeps a thin delegating shim for
every name the test suite / op binders reference (``_xrefs``,
``_xrefs_to_address``, ``_scan_for_calls_to``, ``_resolve_type_field``, ...).

Outbound calls resolve through:
  * ``ctx`` -- resolution / address-context / type helpers relocated to the seam
    (``_resolve_view``, ``_find_function``, ``_functions_containing``,
    ``_address_context``, ``_find_type``);
  * ``il_format`` -- the state-free IL helpers used by the call scan
    (``_iter_llil_instructions``, ``_il_op_name``, ``_llil_constant_value``);
  * ``_shared`` -- module-free helpers (``_parse_address``).

Import direction is one-way: this module imports ``il_format`` and ``_shared``
(plus stdlib + binaryninja). It NEVER imports ``bridge``, ``seam``,
``read_evidence``, ``read_misc``, or ``create_comments`` -- those import THIS
module one-way (design spec §3.2).
"""
from __future__ import annotations

import difflib
from typing import Any

import binaryninja as bn

from . import il_format
from ._shared import _parse_address
from .bridge_state import require_analysis

# Import symbol kinds, in resolution-preference order. Mirrors the literal that
# also lives as ``BinaryNinjaBridge._IMPORT_SYMBOL_TYPES`` (used by the imports
# op, which stays on the bridge); kept here verbatim so the xref free functions
# need no callback into the class.
_IMPORT_SYMBOL_TYPES: list[tuple[str, str]] = [
    ("ImportedFunctionSymbol", "function"),
    ("ImportedDataSymbol", "data"),
    ("ImportAddressSymbol", "address"),
]


def _xrefs(ctx, selector: str | None, identifier):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Cross-references")
    try:
        address = _parse_address(identifier)
    except Exception:
        try:
            address = ctx._find_function(bv, identifier).start
        except RuntimeError as exc:
            # An ambiguous identifier is actionable as-is; replacing it
            # with "not found / not an import symbol" would be misleading.
            # Only fall back to import-symbol lookup for genuine misses.
            if "Ambiguous" in str(exc):
                raise
            return _xrefs_import_symbol(ctx, bv, identifier)
    return _xrefs_to_address(ctx, bv, address)


def _xrefs_to_address(ctx, bv, address: int) -> dict[str, Any]:
    code_refs = []
    data_refs = []
    get_code_refs = getattr(bv, "get_code_refs", None)
    raw_code_refs = list(get_code_refs(address)) if callable(get_code_refs) else []
    for ref in sorted(raw_code_refs, key=lambda item: int(item.address)):
        fn = getattr(ref, "function", None)
        caller = (
            {"address": hex(int(fn.start)), "name": str(fn.name)}
            if fn is not None
            else None
        )
        ref_addr = int(ref.address)
        ref_arch = getattr(ref, "arch", None) or getattr(fn, "arch", None)
        code_refs.append(
            {
                "function": fn.name if fn is not None else None,
                "address": hex(ref_addr),
                "caller_function": caller,
                "kind": "code",
                "context": ctx._address_context(
                    bv, ref_addr, include_disasm=True, arch=ref_arch, assume_code=True
                ),
            }
        )
    get_data_refs = getattr(bv, "get_data_refs", None)
    raw_data_refs = list(get_data_refs(address)) if callable(get_data_refs) else []
    for ref_addr in sorted(raw_data_refs):
        ref_addr = int(ref_addr)
        functions = ctx._functions_containing(bv, ref_addr)
        fn = functions[0] if functions else None
        caller = (
            {"address": hex(int(fn.start)), "name": str(fn.name)}
            if fn is not None
            else None
        )
        data_refs.append(
            {
                "function": fn.name if fn is not None else None,
                "address": hex(ref_addr),
                "caller_function": caller,
                "kind": "data",
                "context": ctx._address_context(bv, ref_addr),
            }
        )
    return {
        "address": hex(address),
        "target_context": ctx._address_context(bv, address, include_disasm=True),
        "code_refs": code_refs,
        "data_refs": data_refs,
    }


def _import_symbol_name(sym) -> str:
    """Preferred display name for an import symbol."""
    return str(
        getattr(sym, "short_name", None)
        or getattr(sym, "full_name", None)
        or sym.name
    )


def _find_import_symbol(ctx, bv, name: str):
    needle = name.lower()
    for attr_name, kind in _IMPORT_SYMBOL_TYPES:
        sym_type = getattr(bn.SymbolType, attr_name, None)
        if sym_type is None:
            continue
        for sym in list(bv.get_symbols_of_type(sym_type)):
            if _import_symbol_name(sym).lower() == needle:
                return sym
    return None


def _xrefs_import_symbol(ctx, bv, identifier: str) -> dict[str, Any]:
    sym = _find_import_symbol(ctx, bv, identifier)
    if sym is None:
        available: list[str] = []
        for attr_name, kind in _IMPORT_SYMBOL_TYPES:
            sym_type = getattr(bn.SymbolType, attr_name, None)
            if sym_type is None:
                continue
            for s in list(bv.get_symbols_of_type(sym_type)):
                available.append(_import_symbol_name(s))
        suggestions = difflib.get_close_matches(identifier, sorted(set(available)), n=5, cutoff=0.5)
        msg = f"Function not found: {identifier}."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}"
        msg += " Not found as an import symbol either. Use 'bn imports' to see available imports."
        raise RuntimeError(msg)

    sym_address = int(sym.address)
    result = _xrefs_to_address(ctx, bv, sym_address)
    result["import_resolved"] = True
    result["import_name"] = str(identifier)

    if not result.get("code_refs"):
        manual = _scan_for_calls_to(ctx, bv, sym_address)
        if manual:
            result["code_refs"] = manual
            result["code_refs_scanned"] = True

    return result


def _scan_for_calls_to(ctx, bv, target_address: int) -> list[dict[str, Any]]:
    code_refs = []
    seen: set[int] = set()
    for fn in list(bv.functions):
        for insn in il_format._iter_llil_instructions(fn):
            op_name = il_format._il_op_name(insn)
            if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL"}:
                continue
            dest_value = il_format._llil_constant_value(getattr(insn, "dest", None))
            if dest_value != target_address:
                continue
            ref_addr = int(getattr(insn, "address", 0))
            if ref_addr in seen:
                continue
            seen.add(ref_addr)
            fn_arch = getattr(fn, "arch", None)
            code_refs.append({
                "function": str(fn.name),
                "address": hex(ref_addr),
                "caller_function": {
                    "address": hex(int(fn.start)),
                    "name": str(fn.name),
                },
                "kind": "code",
                "context": ctx._address_context(
                    bv, ref_addr, include_disasm=True, arch=fn_arch, assume_code=True
                ),
            })
    code_refs.sort(key=lambda item: int(item["address"], 16))
    return code_refs


def _resolve_type_field(ctx, bv, field_spec: str):
    type_name, sep, field_name = str(field_spec).rpartition(".")
    if not sep or not type_name or not field_name:
        raise RuntimeError("Field selector must be in the form Struct.field")

    resolved_name, type_obj = ctx._find_type(bv, type_name)
    members = getattr(type_obj, "members", None)
    if members is None:
        raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")

    member_list = list(members)

    def field_info(member, index: int):
        return {
            "type_name": resolved_name,
            "field_name": str(getattr(member, "name", "")) or field_name,
            "offset": int(getattr(member, "offset", 0)),
            "member_index": index,
            "field_type": str(getattr(member, "type", "")),
        }

    for index, member in enumerate(member_list):
        if str(getattr(member, "name", "")) != field_name:
            continue
        return field_info(member, index)

    folded_matches = [
        (index, member)
        for index, member in enumerate(member_list)
        if str(getattr(member, "name", "")).lower() == field_name.lower()
    ]
    if len(folded_matches) == 1:
        index, member = folded_matches[0]
        return field_info(member, index)

    try:
        requested_offset = _parse_address(field_name)
    except Exception:
        requested_offset = None
    if requested_offset is not None:
        for index, member in enumerate(member_list):
            if int(getattr(member, "offset", 0)) != requested_offset:
                continue
            return field_info(member, index)
        raise RuntimeError(f"Field not found: {resolved_name}.0x{requested_offset:x}")

    available = [str(getattr(member, "name", "")) for member in member_list if str(getattr(member, "name", ""))]
    suggestions = difflib.get_close_matches(field_name, available, n=5, cutoff=0.5)
    if suggestions:
        raise RuntimeError(
            f"Field not found: {resolved_name}.{field_name}. Did you mean: {', '.join(suggestions)}"
        )
    raise RuntimeError(f"Field not found: {resolved_name}.{field_name}")


def _field_xrefs(ctx, selector: str | None, field_spec: str):
    bv = ctx._resolve_view(selector)
    field = _resolve_type_field(ctx, bv, field_spec)

    code_refs = []
    for ref in sorted(
        list(bv.get_code_refs_for_type_field(field["type_name"], field["offset"])),
        key=lambda item: int(getattr(item, "address", 0)),
    ):
        func = getattr(ref, "func", None)
        address = int(getattr(ref, "address", 0))
        code_refs.append(
            {
                "function": func.name if func is not None else None,
                "address": hex(address),
                "size": int(getattr(ref, "size", 0)),
                "incoming_type": str(getattr(ref, "incomingType", "")) or None,
                "disasm": bv.get_disassembly(address) or "",
            }
        )

    data_refs = []
    for address in sorted(list(bv.get_data_refs_for_type_field(field["type_name"], field["offset"]))):
        symbol = bv.get_symbol_at(address)
        # BinaryView has no get_type_at(); the data variable defined at the
        # address carries the type. The old call raised AttributeError and
        # took the whole --field query down whenever a field had data refs.
        data_var = bv.get_data_var_at(address)
        type_obj = getattr(data_var, "type", None) if data_var is not None else None
        data_refs.append(
            {
                "address": hex(address),
                "symbol": symbol.name if symbol is not None else None,
                "type": str(type_obj) if type_obj is not None else None,
            }
        )

    return {
        "field": field,
        "code_refs": code_refs,
        "data_refs": data_refs,
    }
