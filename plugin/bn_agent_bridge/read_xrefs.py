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
from ._shared import _parse_address, _validate_count
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


def _xrefs(ctx, selector: str | None, identifier, *, offset: int = 0, limit: int | None = None):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Cross-references")
    offset = _validate_count(offset, label="offset", minimum=0)
    limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
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
            return _drop_legacy_ref_arrays(
                _xrefs_import_symbol(ctx, bv, identifier, offset=offset, limit=limit)
            )
    return _drop_legacy_ref_arrays(
        _xrefs_to_address(ctx, bv, address, offset=offset, limit=limit)
    )


def _drop_legacy_ref_arrays(envelope: dict[str, Any]) -> dict[str, Any]:
    """Strip the deprecated full ``code_refs``/``data_refs`` arrays from the
    ``xrefs`` OP response so ``--offset``/``--limit`` bound the entire serialized
    payload, not just ``items`` (#184). On a high-fanout symbol the arrays rode
    full regardless of paging and spilled the JSON even at ``--limit 1``. The
    full-set summary counts (``code_ref_count``/``data_ref_count``/
    ``caller_function_count``) and the paged ``items`` (each carrying its
    ``kind``) are everything a "who references X" triage needs. The lower-level
    builders (``_xrefs_to_address``/``_xrefs_import_symbol`` via
    ``_xref_envelope``) still produce the dual shape, which ``function info`` and
    evidence message-lensing embed by calling them directly."""
    envelope.pop("code_refs", None)
    envelope.pop("data_refs", None)
    return envelope


def _xref_envelope(address, target_context, code_refs, data_refs, *,
                   offset: int = 0, limit: int | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap xref results in the canonical paging envelope (#164).

    ``items`` is the unified list (code refs first, then data refs), each row
    carrying its ``kind`` (code|data). ``code_refs``/``data_refs`` are kept as a
    deprecated dual shape for back-compat, the text renderer, and ``function
    info`` (which embeds the full set, unpaged). Summary counts (#140) reflect
    the FULL set regardless of paging."""
    caller_addrs = {
        ref["caller_function"]["address"]
        for ref in code_refs
        if isinstance(ref.get("caller_function"), dict) and ref["caller_function"].get("address")
    }
    items = code_refs + data_refs
    total = len(items)
    page = items[offset:]
    if limit is not None:
        page = page[:limit]
    out: dict[str, Any] = {
        "address": hex(address) if isinstance(address, int) else address,
        "target_context": target_context,
        "code_ref_count": len(code_refs),
        "data_ref_count": len(data_refs),
        "caller_function_count": len(caller_addrs),
        # Deprecated dual shape (kept for back-compat + function-info embedding).
        "code_refs": code_refs,
        "data_refs": data_refs,
        # Canonical paging envelope.
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + len(page)) < total,
    }
    if extra:
        out.update(extra)
    return out


def _xrefs_to_address(ctx, bv, address: int, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
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
    return _xref_envelope(
        address,
        ctx._address_context(bv, address, include_disasm=True),
        code_refs,
        data_refs,
        offset=offset,
        limit=limit,
    )


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


def _xrefs_import_symbol(ctx, bv, identifier: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
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

    # #201: a demangled C++ name matches an import veneer (PLT stub) via its
    # short_name, but the same symbol may also be DEFINED in this module (a PIC
    # self-reference). Resolving xrefs to the stub gives the wrong call-graph, so
    # redirect to the real definition -- reusing the same impl-over-stub resolver
    # `_find_function` uses for a name collision. `xrefs <mangled>` and decompile
    # already reach the definition; this makes `xrefs <demangled>` consistent.
    raw_name = str(getattr(sym, "raw_name", sym.name))
    bodies = ctx._find_functions_by_name(bv, raw_name, case_sensitive=True)
    impl = ctx._resolve_impl_over_stub(bodies) if bodies else None
    if impl is not None and int(impl.start) != int(sym.address):
        result = _xrefs_to_address(ctx, bv, int(impl.start), offset=offset, limit=limit)
        result["import_resolved"] = True
        result["import_name"] = str(identifier)
        result["resolved_to_definition"] = hex(int(impl.start))
        return result

    sym_address = int(sym.address)
    result = _xrefs_to_address(ctx, bv, sym_address, offset=offset, limit=limit)
    result["import_resolved"] = True
    result["import_name"] = str(identifier)

    if not result.get("code_refs"):
        manual = _scan_for_calls_to(ctx, bv, sym_address)
        if manual:
            # Rebuild the envelope so the manually-discovered code refs land in
            # both the deprecated `code_refs` and the canonical `items` page.
            result = _xref_envelope(
                sym_address, result["target_context"], manual, result["data_refs"],
                offset=offset, limit=limit,
                extra={"import_resolved": True, "import_name": str(identifier),
                       "code_refs_scanned": True},
            )

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
