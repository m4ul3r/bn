"""State-free variable discovery and serialization helpers.

These are the read-only variable helpers shared by decompile, bundle, taint, and
the mutation verify/restore paths. They operate purely on Binary Ninja ``func``/
``var`` objects (no bridge instance state), so they live here as module-level
free functions. The bridge keeps a thin delegating shim for each one.

This module is a LEAF: it imports ONLY stdlib + binaryninja. It must NOT import
``bridge``, ``seam``, or ``il_format``.
"""
from __future__ import annotations

from typing import Any

_SOURCE_TYPE_SHORT: dict[str, str] = {
    "RegisterVariableSourceType": "reg",
    "StackVariableSourceType": "stack",
    "FlagVariableSourceType": "flag",
}


def _find_variable_by_storage(func, storage: int, *, is_parameter: bool | None = None):
    collections = []
    if is_parameter is True:
        collections = [(func.parameter_vars, True)]
    elif is_parameter is False:
        collections = [(func.stack_layout, False)]
    else:
        collections = [(func.parameter_vars, True), (func.stack_layout, False)]

    for collection, marker in collections:
        for var in list(collection):
            if int(var.storage) == int(storage):
                return var, marker
    raise RuntimeError(f"Variable not found at storage {storage}")


def _find_variable_by_identifier(func, identifier):
    """Locate a variable by its BN identifier across the COMPLETE set
    (``func.vars``), not just the canonical param/stack/HLIL union.

    Narrowing a register-backed local (e.g. uint32_t -> uint8_t) can drop it
    out of ``hlil.vars`` even though the retype landed and ``func.vars`` still
    carries it correctly typed -- so ``_iter_canonical_variables`` (which
    surfaces register locals only via HLIL) no longer yields it. The BN
    identifier packs source_type|index|storage and is unique per variable, so
    a match here is exact (no same-storage type-collision risk). Returns None
    if ``identifier`` is None, ``func.vars`` is unavailable, or nothing
    matches. (#156)"""
    if identifier is None:
        return None
    try:
        variables = list(func.vars)
    except Exception:
        return None
    for var in variables:
        if _variable_identifier(var) == identifier:
            return var
    return None


def _variable_source_name(var) -> str:
    source_type = getattr(var, "source_type", None)
    if source_type is None:
        return "unknown"
    return str(getattr(source_type, "name", source_type))


def _variable_identifier(var) -> int | None:
    try:
        return int(getattr(var, "identifier"))
    except Exception:
        return None


def _local_id(func, var, *, is_parameter: bool) -> str:
    role = "param" if is_parameter else "local"
    storage = int(getattr(var, "storage", 0))
    index = int(getattr(var, "index", 0))
    identifier = _variable_identifier(var)
    source_name = _variable_source_name(var)
    short_source = _SOURCE_TYPE_SHORT.get(source_name, source_name)
    return ":".join(
        [
            hex(int(func.start)),
            role,
            short_source,
            str(storage),
            str(index),
            str(identifier if identifier is not None else "none"),
        ]
    )


def _variable_entry(func, var, *, is_parameter: bool) -> dict[str, Any]:
    return {
        "name": str(var.name),
        "storage": int(var.storage),
        "type": str(var.type),
        "is_parameter": is_parameter,
        "index": int(getattr(var, "index", 0)),
        "identifier": _variable_identifier(var),
        "source_type": _variable_source_name(var),
        "local_id": _local_id(func, var, is_parameter=is_parameter),
    }


def _variable_marker(var) -> tuple[int | None, int]:
    return (_variable_identifier(var), int(getattr(var, "storage", 0)))


def _iter_canonical_variables(func):
    seen: set[tuple[int | None, int]] = set()

    for var in list(func.parameter_vars):
        marker = _variable_marker(var)
        if marker in seen:
            continue
        seen.add(marker)
        yield var, True

    for var in list(func.stack_layout):
        marker = _variable_marker(var)
        if marker in seen:
            continue
        seen.add(marker)
        yield var, False

    # Register/flag locals that HLIL renders (e.g. rsi_1, rdx_3, loop
    # counters, the success flag) are real, nameable Variables that live in
    # neither parameter_vars nor stack_layout, so without this they are
    # invisible to `local list` and unresolvable by `local rename`/`retype`
    # (-> "Variable not found", which rolls back the whole batch). Surface
    # the HLIL-visible ones; func.vars would also drag in dataflow
    # temporaries (temp0, cond intermediates) that never appear in output.
    for var in _iter_hlil_variables(func):
        marker = _variable_marker(var)
        if marker in seen:
            continue
        seen.add(marker)
        yield var, False


def _iter_hlil_variables(func):
    """HLIL-rendered variables, or empty when HLIL is unavailable.

    Large or non-decompilable functions may have no HLIL; fall back to the
    parameter/stack set rather than failing the whole listing.
    """
    try:
        hlil = func.hlil
    except Exception:
        return []
    if hlil is None:
        return []
    try:
        return list(hlil.vars)
    except Exception:
        return []


def _sort_variable_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            0 if item.get("is_parameter") else 1,
            str(item.get("source_type", "")),
            int(item.get("storage", 0)),
            int(item.get("identifier") or 0),
            str(item.get("name", "")),
        ),
    )


def _annotate_stack_spans(entries: list[dict[str, Any]]) -> None:
    """Add ``distance_to_next_slot`` to each stack variable.

    BN gives a variable's frame offset (``storage``, negative for a stack slot)
    but not its size. ``distance_to_next_slot`` is the byte distance from this
    offset to the next-higher DISTINCT recovered stack offset; the top-most slot
    measures to 0 (the entry stack pointer). Register/flag locals (non-negative
    storage) get none. (F20)

    It is slot PITCH, not capacity. BN recovers one object as several adjacent
    slots -- a memset'd 48-byte buffer can surface as a run of 8-byte slots -- so
    an object's real extent can run past the next slot. For overflow headroom use
    ``list_frame`` (``bn local list --frame``), whose ``to_canary`` measures to
    the canary slot.

    Several variables can share one offset (SSA versions, HLIL-rendered aliases);
    distance is measured between distinct offsets, so every alias of a slot
    carries the same value rather than 0 for all but the last.

    ``span_to_next`` repeats the value under its original F20 name, retained for
    bn-tui's stack viewer (``LocalVariable.span_to_next``, labelled "slot"); drop
    it once that consumer reads ``distance_to_next_slot``.
    """
    stack = [
        e for e in entries
        if isinstance(e.get("storage"), int) and not isinstance(e["storage"], bool) and e["storage"] < 0
    ]
    offsets = sorted({e["storage"] for e in stack})
    next_offset = {offset: offsets[i + 1] if i + 1 < len(offsets) else 0 for i, offset in enumerate(offsets)}
    for entry in stack:
        distance = next_offset[entry["storage"]] - entry["storage"]
        entry["distance_to_next_slot"] = distance
        entry["span_to_next"] = distance


def _list_locals(func) -> list[dict[str, Any]]:
    variables = [
        _variable_entry(func, var, is_parameter=is_parameter)
        for var, is_parameter in _iter_canonical_variables(func)
    ]
    _annotate_stack_spans(variables)
    return _sort_variable_entries(variables)


_CANARY_SYMBOL = "__stack_chk_guard"
# TLS canary slots of the x86 Linux ABIs: fs:0x28 (x86_64) and gs:0x14 (i386).
_TLS_CANARY_SLOTS = {("x86_64", "fsbase", 0x28), ("x86", "gsbase", 0x14)}
_CONSTANT_OPS = ("MLIL_CONST", "MLIL_CONST_PTR", "MLIL_IMPORT")


def _is_canary_load(func, expr) -> bool:
    """True when ``expr`` loads the stack-protector guard value: a load of the
    ``__stack_chk_guard`` symbol (directly, or through one pointer such as a GOT
    slot) or of the x86 TLS canary slot, identified by its architectural register
    rather than the variable's mutable display name."""
    if expr.operation.name != "MLIL_LOAD" or expr.size != func.arch.address_size:
        return False
    address = expr.src
    if address.operation.name == "MLIL_LOAD":
        if (address.size != func.arch.address_size
                or address.src.operation.name not in _CONSTANT_OPS):
            return False
        address = address.src
    op = address.operation.name
    if op in _CONSTANT_OPS:
        symbol = func.view.get_symbol_at(int(address.constant))
        return symbol is not None and str(symbol.name).split("@")[0] == _CANARY_SYMBOL
    if op == "MLIL_ADD":
        for base, offset in ((address.left, address.right), (address.right, address.left)):
            if (
                base.operation.name == "MLIL_VAR"
                and offset.operation.name in _CONSTANT_OPS
                and _variable_source_name(base.src) == "RegisterVariableSourceType"
                and (func.arch.name, func.arch.get_reg_name(base.src.storage),
                     int(offset.constant)) in _TLS_CANARY_SLOTS
            ):
                return True
    return False


def _vars_written(ins) -> list:
    """Every variable a non-SSA MLIL instruction writes. BN's ``vars_written``
    returns nothing for a partial field write (``rax.al = ...``) and for a typed
    syscall, so those two name their written variables here."""
    op = ins.operation.name
    if op == "MLIL_SET_VAR_FIELD":
        return [ins.dest]
    if op == "MLIL_SYSCALL":
        return list(ins.output)
    return list(ins.vars_written)


def _find_canary_storage(func) -> int | None:
    """Storage of the stack slot the prologue copies the stack-protector guard
    into, or None when no such copy is found.

    Only an observed full-ABI-word ``slot = guard`` (or ``reg = guard; slot = reg``
    within one basic block) counts; partial loads or copies are not guard
    evidence. Any write to a register ends its guard copy, whatever
    instruction makes it (a call output, a split or partial-field write as much
    as a plain assignment). Any failure reading MLIL is treated as no evidence."""
    try:
        mlil = func.mlil
        if mlil is None:
            return None
        guard_width = func.arch.address_size
        for block in list(mlil.basic_blocks):
            guard_registers: set[int] = set()
            for ins in block:
                from_guard = (
                    ins.operation.name == "MLIL_SET_VAR"
                    and ins.size == guard_width
                    and (
                        _is_canary_load(func, ins.src)
                        or (ins.src.operation.name == "MLIL_VAR"
                            and ins.src.size == guard_width
                            and _variable_identifier(ins.src.src) in guard_registers)
                    )
                )
                for written in _vars_written(ins):
                    written_id = _variable_identifier(written)
                    if written_id is None:
                        guard_registers.clear()
                    else:
                        guard_registers.discard(written_id)
                if not from_guard:
                    continue
                if _variable_source_name(ins.dest) == "StackVariableSourceType":
                    return int(ins.dest.storage)
                identifier = _variable_identifier(ins.dest)
                if identifier is not None:
                    guard_registers.add(identifier)
    except Exception:
        return None
    return None


def _frame_slot_kind(storage: int, names: list[str], canary: int | None) -> str:
    if storage == canary:
        return "canary"
    if any(name.startswith("__saved_") for name in names):
        return "saved_register"
    if "__return_addr" in names:
        return "return_address"
    if storage >= 0:
        return "caller_frame"
    return "local"


def _list_frame(func) -> dict[str, Any]:
    """One row per DISTINCT ``func.stack_layout`` offset, ascending.

    ``bn local list`` has one row per variable, so a frame whose slots carry
    several SSA versions or aliases each reads as dozens of rows. Here the
    aliases collapse into their slot, which is what a layout question ("how far
    is this buffer from the canary") needs. Each row carries:

    * ``distance_to_next_slot`` -- bytes to the next distinct offset (slot pitch,
      NOT capacity; see ``_annotate_stack_spans``); the top-most negative slot
      measures to 0, a non-negative top slot has none;
    * ``to_canary`` -- bytes from this offset to the canary slot, for slots
      below it: the headroom a write starting here has before it reaches the
      canary, however BN split the object into slots;
    * ``kind`` -- ``canary``, ``saved_register``, ``return_address``,
      ``caller_frame`` (non-negative storage) or ``local``.

    Variables keep their ``local_id`` so a row can drive ``local rename`` /
    ``local retype`` directly. Collapsing happens here only: the canonical
    iteration that resolves ``local_id`` selectors is untouched.
    """
    parameter_markers = {_variable_marker(var) for var in list(func.parameter_vars)}
    by_storage: dict[int, list[Any]] = {}
    seen: set[tuple[int | None, int]] = set()
    for var in list(func.stack_layout):
        marker = _variable_marker(var)
        if marker in seen:
            continue
        seen.add(marker)
        by_storage.setdefault(int(var.storage), []).append(var)

    canary = _find_canary_storage(func)
    offsets = sorted(by_storage)
    slots: list[dict[str, Any]] = []
    for i, storage in enumerate(offsets):
        if storage < 0:
            next_offset = min(offsets[i + 1], 0) if i + 1 < len(offsets) else 0
            distance: int | None = next_offset - storage
        elif i + 1 < len(offsets):
            distance = offsets[i + 1] - storage
        else:
            distance = None
        variables = []
        for var in by_storage[storage]:
            is_parameter = _variable_marker(var) in parameter_markers
            variables.append(
                {
                    "name": str(var.name),
                    "type": str(var.type),
                    "is_parameter": is_parameter,
                    "local_id": _local_id(func, var, is_parameter=is_parameter),
                }
            )
        names = [item["name"] for item in variables]
        slots.append(
            {
                "storage": storage,
                "kind": _frame_slot_kind(storage, names, canary),
                "names": names,
                "types": list(dict.fromkeys(item["type"] for item in variables)),
                "distance_to_next_slot": distance,
                "to_canary": canary - storage if canary is not None and storage < canary else None,
                "variables": variables,
            }
        )
    return {
        "slots": slots,
        "slot_count": len(slots),
        "variable_count": len(seen),
        "canary_storage": canary,
    }


def _find_variables_by_name(func, name: str) -> list[tuple[Any, bool]]:
    matches = []
    for var, is_parameter in _iter_canonical_variables(func):
        if str(var.name) == name:
            matches.append((var, is_parameter))
    return matches


def _find_variable_selector(func, selector: str) -> tuple[Any, bool]:
    locals_by_id: dict[str, tuple[Any, bool]] = {}
    legacy_by_id: dict[str, tuple[Any, bool]] = {}
    for var, is_parameter in _iter_canonical_variables(func):
        local_id = _local_id(func, var, is_parameter=is_parameter)
        locals_by_id[local_id] = (var, is_parameter)
        # Build legacy (long-form) ID for backward compat
        role = "param" if is_parameter else "local"
        source_name = _variable_source_name(var)
        storage = int(getattr(var, "storage", 0))
        index = int(getattr(var, "index", 0))
        identifier = _variable_identifier(var)
        legacy_id = ":".join([
            hex(int(func.start)), role, source_name,
            str(storage), str(index),
            str(identifier if identifier is not None else "none"),
        ])
        if legacy_id != local_id:
            legacy_by_id[legacy_id] = (var, is_parameter)
    if selector in locals_by_id:
        return locals_by_id[selector]
    if selector in legacy_by_id:
        return legacy_by_id[selector]

    matches = _find_variables_by_name(func, selector)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        lines = [f"Ambiguous variable selector: {selector} matches {len(matches)} variables:"]
        for var, is_parameter in matches:
            role = "param" if is_parameter else "local"
            source_name = _variable_source_name(var)
            storage = int(getattr(var, "storage", 0))
            lines.append(f"  {str(getattr(var, 'name', '<unknown>'))}  [{role}; storage={storage}; source={source_name}]")
        lines.append("retry with the full local_id from `bn local list --format json`")
        raise RuntimeError("\n".join(lines))
    raise RuntimeError(f"Variable not found: {selector}")
