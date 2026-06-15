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
    """Add ``span_to_next`` (bytes to the next stack slot) to each stack variable.

    BN gives a variable's frame offset (``storage``, negative for a stack slot)
    but not its size, so judging a buffer overflow meant dropping to a ``py exec``
    over ``func.stack_layout`` to subtract adjacent offsets. The distance to the
    next-higher stack slot IS the slot's capacity; compute it once here so every
    stack var carries it. The top-most slot spans to 0 (the saved frame base).
    Register/flag locals (non-negative storage) get no span. (F20)
    """
    stack = sorted(
        (e for e in entries if isinstance(e.get("storage"), int) and e["storage"] < 0),
        key=lambda e: e["storage"],
    )
    for i, entry in enumerate(stack):
        next_storage = stack[i + 1]["storage"] if i + 1 < len(stack) else 0
        entry["span_to_next"] = next_storage - entry["storage"]


def _list_locals(func) -> list[dict[str, Any]]:
    variables = [
        _variable_entry(func, var, is_parameter=is_parameter)
        for var, is_parameter in _iter_canonical_variables(func)
    ]
    _annotate_stack_spans(variables)
    return _sort_variable_entries(variables)


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
