"""State-free IL/HLIL/disassembly rendering and IL-instruction iteration.

These helpers operate purely on Binary Ninja ``bv``/``func``/``insn`` objects
(no bridge instance state), so they live here as module-level free functions.
The bridge keeps a thin delegating shim for each one.

This module is a LEAF below the read modules: it imports ONLY stdlib +
binaryninja + ``._shared`` + the ``vars`` module. It must NOT import ``bridge``,
``seam``, or any ``read_*`` module.
"""
from __future__ import annotations

import re
from typing import Any

try:
    import binaryninja as bn
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import vars as vars_mod
from ._shared import OperationFailure, is_imported_function


def _function_size(func) -> int | None:
    try:
        total = getattr(func, "total_bytes", None)
        if total is not None:
            return int(total)
    except Exception:
        pass
    try:
        end = max(int(block.end) for block in list(func.basic_blocks))
        return end - int(func.start)
    except Exception:
        return None


def _function_metadata(func) -> dict[str, Any]:
    func_type = getattr(func, "type", None)
    calling_convention = getattr(func, "calling_convention", None)
    if calling_convention is None and func_type is not None:
        calling_convention = getattr(func_type, "calling_convention", None)
    return_type = getattr(func, "return_type", None)
    if return_type is None and func_type is not None:
        return_type = getattr(func_type, "return_value", None)
    size = _function_size(func)
    return {
        "prototype": str(func_type),
        "return_type": str(return_type) if return_type is not None else None,
        "calling_convention": (
            str(calling_convention)
            if calling_convention is not None
            else None
        ),
        "size": size,
        "size_known": size is not None,
        "imported": is_imported_function(func),
    }


def _display_name(func) -> str:
    """Demangled display name for a function: the symbol's ``short_name`` when BN
    has one (it keeps ``fn.name`` mangled for C++), else ``fn.name``. Lets
    function list/search/info expose a greppable/clusterable C++ name without
    shelling out to c++filt -- `decompile` already demangles (#196)."""
    sym = getattr(func, "symbol", None)
    short = getattr(sym, "short_name", None) if sym is not None else None
    if short:
        return str(short)
    return str(getattr(func, "name", ""))


def _unimplemented_instructions(func, *, cap: int = 64) -> dict[str, Any]:
    """Aggregate signal for instructions Binary Ninja's lifter could not model.

    When BN can't lift an instruction it emits an ``*_UNIMPL`` / ``*_UNIMPL_MEM``
    IL op and renders it inline in HLIL as ``🚫 /* unimplemented {...} */`` -- the
    *only* signal today. A function whose core computation is unlifted (e.g. the
    AArch64 FP fused-multiply-add family ``fnmsub``/``fmadd``, which never reach
    MLIL while the surrounding integer ops do) otherwise reads as fully analyzed,
    and a dataflow/taint pass flows through it as a silent hole (#206).

    We scan LLIL first (closest to decode, so it catches FP/SIMD ops that never
    surface in MLIL) and fall back to MLIL. Returns ``{count, addresses,
    truncated}``; ``addresses`` is capped so a pathological function can't bloat
    the response.
    """
    addrs: list[int] = []
    seen: set[int] = set()
    for attr in ("low_level_il", "llil", "mlil", "medium_level_il"):
        il = getattr(func, attr, None)
        if il is None:
            continue
        try:
            blocks = list(il)
        except Exception:
            blocks = list(getattr(il, "basic_blocks", []) or [])
        found = False
        for block in blocks:
            try:
                items = list(block)
            except Exception:
                continue
            for ins in items:
                if "UNIMPL" in _il_op_name(ins):
                    found = True
                    a = int(getattr(ins, "address", 0))
                    if a not in seen:
                        seen.add(a)
                        addrs.append(a)
        if found:
            break  # first IL level that exposes them is authoritative
    addrs.sort()
    return {
        "count": len(addrs),
        "addresses": [hex(a) for a in addrs[:cap]],
        "truncated": len(addrs) > cap,
    }


def _comment_map(bv, func) -> dict[str, str]:
    arch = getattr(func, "arch", None)
    comments: dict[str, str] = {}
    for block in list(func.basic_blocks):
        addr = block.start
        while addr < block.end:
            text = bv.get_comment_at(addr)
            if text:
                comments[hex(addr)] = text
            addr += max(1, _instruction_length(bv, int(addr), arch=arch))
    return comments


def _il_op_name(item) -> str:
    operation = getattr(item, "operation", None)
    name = getattr(operation, "name", None)
    if name:
        return str(name)
    return str(operation)


def _llil_constant_value(expr) -> int | None:
    if expr is None:
        return None
    if _il_op_name(expr) not in {"LLIL_CONST", "LLIL_CONST_PTR"}:
        return None
    constant = getattr(expr, "constant", None)
    if constant is not None:
        return int(constant)
    value = getattr(expr, "value", None)
    if value is None:
        return None
    nested_value = getattr(value, "value", None)
    if nested_value is not None:
        return int(nested_value)
    try:
        return int(value)
    except Exception:
        return None


def _coerce_il_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _iter_llil_instructions(func) -> list[Any]:
    il = getattr(func, "low_level_il", None)
    if il is None:
        il = getattr(func, "llil", None)
    if il is None:
        return []

    instructions = []
    try:
        blocks = list(il)
    except Exception:
        blocks = list(getattr(il, "basic_blocks", []) or [])
    for block in blocks:
        try:
            instructions.extend(list(block))
        except Exception:
            continue
    instructions.sort(key=lambda item: int(getattr(item, "address", 0)))
    return instructions


def _hlil_candidates_for_llil(insn) -> list[Any]:
    candidates = []
    seen: set[tuple[str, int]] = set()

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        expr_index = getattr(candidate, "expr_index", None)
        marker = (type(candidate).__name__, int(expr_index) if expr_index is not None else id(candidate))
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(candidate)

    for attr in ("hlils", "hlil"):
        for candidate in _coerce_il_list(getattr(insn, attr, None)):
            add(candidate)

    mapped_mlil = getattr(insn, "mapped_medium_level_il", None)
    if mapped_mlil is not None:
        for attr in ("hlils", "hlil"):
            for candidate in _coerce_il_list(getattr(mapped_mlil, attr, None)):
                add(candidate)

    for mlil in _coerce_il_list(getattr(insn, "mlils", None)):
        for attr in ("hlils", "hlil"):
            for candidate in _coerce_il_list(getattr(mlil, attr, None)):
                add(candidate)

    return candidates


def _il_parent(instruction) -> Any | None:
    for attr in ("parent", "parent_instruction"):
        parent = getattr(instruction, attr, None)
        if parent is not None and parent is not instruction:
            return parent
    return None


def _hlil_marker(instruction) -> tuple[str, int]:
    expr_index = getattr(instruction, "expr_index", None)
    return (
        type(instruction).__name__,
        int(expr_index) if expr_index is not None else id(instruction),
    )


def _hlil_type_name(instruction) -> str:
    return type(instruction).__name__


def _hlil_text_is_local(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 240:
        return False
    if stripped.count("\n") > 1:
        return False
    return True


def _hlil_condition_is_meaningful(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return False
    if re.search(r"\bcond:\d", stripped):
        return False
    return True


def _is_hlil_assignment_like(instruction) -> bool:
    return _hlil_type_name(instruction) in {
        "HighLevelILAssign",
        "HighLevelILVarAssign",
        "HighLevelILVarInit",
        "HighLevelILAssignMem",
        "HighLevelILAssignUnpack",
        "HighLevelILVarDeclare",
    }


def _is_hlil_control_flow(instruction) -> bool:
    return _hlil_type_name(instruction) in {
        "HighLevelILIf",
        "HighLevelILWhile",
        "HighLevelILDoWhile",
        "HighLevelILFor",
        "HighLevelILSwitch",
        "HighLevelILCase",
    }


def _is_hlil_hard_boundary(instruction) -> bool:
    if _is_hlil_assignment_like(instruction) or _is_hlil_control_flow(instruction):
        return True
    return _hlil_type_name(instruction) in {
        "HighLevelILRet",
        "HighLevelILBlock",
        "HighLevelILCall",
        "HighLevelILTailcall",
    }


def _is_hlil_trivial_wrapper(instruction) -> bool:
    return _hlil_type_name(instruction) in {
        "HighLevelILCall",
        "HighLevelILSx",
        "HighLevelILZx",
        "HighLevelILLowPart",
        "HighLevelILIntToFloat",
        "HighLevelILFloatToInt",
        "HighLevelILBoolToInt",
        "HighLevelILFloatConv",
        "HighLevelILAddressOf",
        "HighLevelILAddressOfField",
        "HighLevelILArrayIndex",
    }


def _hlil_call_roots(insn) -> list[Any]:
    roots = []
    seen: set[tuple[str, int]] = set()
    for candidate in _hlil_candidates_for_llil(insn):
        current = candidate
        while current is not None:
            if _hlil_type_name(current) == "HighLevelILCall":
                marker = _hlil_marker(current)
                if marker not in seen:
                    seen.add(marker)
                    roots.append(current)
                break
            current = _il_parent(current)
    return roots


def _select_local_hlil_node(insn) -> Any | None:
    roots = _hlil_call_roots(insn)
    if not roots:
        return None

    # #475/#476: BN folds adjacent/nested calls (e.g. `p = f(g(x))`) so one LLIL call
    # maps to MULTIPLE HighLevelILCall roots -- the neighbor's included. Scope to the
    # roots at THIS call's address; if none match and the fold is ambiguous (>1 root),
    # return None so the caller emits a null statement rather than describing another
    # call. A single unmatched root (the common, unambiguous case) is kept as-is. The
    # sibling argument path already scopes this way (read_evidence `_call_arguments`).
    call_addr = int(getattr(insn, "address", 0) or 0)
    matched = [r for r in roots if int(getattr(r, "address", -1) or -1) == call_addr]
    roots = matched or ([] if len(roots) > 1 else roots)
    if not roots:
        return None

    root_fallback = None
    for root in roots:
        current = root
        best_expression = None
        assignment_candidate = None
        enclosing_call = None
        seen: set[tuple[str, int]] = set()
        while current is not None:
            marker = _hlil_marker(current)
            if marker in seen:
                break
            seen.add(marker)

            parent = _il_parent(current)
            if parent is None:
                break
            if _is_hlil_control_flow(parent):
                break
            if _is_hlil_assignment_like(parent):
                text = str(parent)
                if _hlil_text_is_local(text):
                    assignment_candidate = parent
                break
            if _is_hlil_hard_boundary(parent):
                # #490: a folded ANCESTOR call across a trivial cast (`outer(cast(inner()))`)
                # is a hard boundary but -- unlike a Ret/Block -- it PROVABLY contains this
                # call: after #475's address filter every ancestor of the matched root
                # encloses it, so describing its statement can't leak a neighbor. Walk
                # THROUGH the ancestor call to its enclosing statement (the pre-#475
                # behavior), keeping the outermost such call as a fallback for bare
                # `foo(bar(inner()))` expression statements. Ret/Block still stop the walk.
                if _hlil_type_name(parent) in ("HighLevelILCall", "HighLevelILTailcall"):
                    parent_text = str(parent)
                    if _hlil_text_is_local(parent_text):
                        enclosing_call = parent
                    current = parent
                    continue
                break

            parent_text = str(parent)
            if not _is_hlil_trivial_wrapper(parent) and _hlil_text_is_local(parent_text):
                best_expression = parent
            current = parent

        if best_expression is not None:
            return best_expression
        if assignment_candidate is not None:
            return assignment_candidate
        if enclosing_call is not None:
            return enclosing_call
        # #644: a call whose return value is DISCARDED is itself the statement -- its
        # parent is the enclosing HighLevelILBlock, a hard boundary, so the ancestor
        # walk above finds nothing and every bare `memcpy(...)`/`free(...)` call
        # statement reported no_local_statement. Fall back to the matched root: #475's
        # address filter already proved this root belongs to THIS call, so it cannot
        # leak a neighbor's statement. A non-local root is still returned here and
        # rejected by the localization layer as `statement_not_local`, keeping #557's
        # reason codes distinguishable.
        if root_fallback is None or (
            not _hlil_text_is_local(str(root_fallback)) and _hlil_text_is_local(str(root))
        ):
            root_fallback = root
    return root_fallback


def _hlil_statement_localization(insn) -> tuple[str | None, str | None]:
    """Localize a call's HLIL statement AND report why it failed (#557).

    Returns ``(text, reason)``. On success ``text`` is the local HLIL statement
    for THIS call site and ``reason`` is None. On failure ``text`` is None and
    ``reason`` is a stable machine-readable code so a caller emitting a null
    ``hlil_statement`` can say WHY it is null instead of leaving an agent to
    re-run decompile and correlate addresses by hand:

    * ``no_hlil_mapping`` -- BN produced no HLIL instruction for this LLIL call
      at all (nothing to render).
    * ``hlil_not_call_shaped`` -- HLIL exists for the LLIL call but folded into a
      non-call statement (a coarse assignment/return/block), so no
      ``HighLevelILCall`` root is reachable to describe.
    * ``ambiguous_fold`` -- BN folded several calls into this one LLIL
      instruction (#475/#476) and none of the resulting call roots sit at this
      call's address, so attributing a statement would risk describing a
      neighbor's call.
    * ``statement_not_local`` -- a statement was selected but its rendered text
      is non-local (too long / multi-line, e.g. a whole-function blob), which
      the localness filter rejects.
    * ``no_local_statement`` -- roots matched this address but neither the
      ancestor walk nor the #644 root fallback produced a node at all (only
      reachable if the root list is empty after filtering, so genuinely rare).
    """
    roots = _hlil_call_roots(insn)
    if not roots:
        if not _hlil_candidates_for_llil(insn):
            return None, "no_hlil_mapping"
        return None, "hlil_not_call_shaped"
    call_addr = int(getattr(insn, "address", 0) or 0)
    matched = [r for r in roots if int(getattr(r, "address", -1) or -1) == call_addr]
    selected_roots = matched or ([] if len(roots) > 1 else roots)
    if not selected_roots:
        return None, "ambiguous_fold"
    node = _select_local_hlil_node(insn)
    if node is None:
        return None, "no_local_statement"
    text = str(node)
    if not _hlil_text_is_local(text):
        return None, "statement_not_local"
    return text, None


def _hlil_statement_text(insn) -> str | None:
    text, _reason = _hlil_statement_localization(insn)
    return text


def _hlil_pre_branch_condition(insn) -> str | None:
    current = _select_local_hlil_node(insn)
    if current is None:
        return None

    seen: set[tuple[str, int]] = set()
    while current is not None:
        marker = _hlil_marker(current)
        if marker in seen:
            break
        seen.add(marker)
        parent = _il_parent(current)
        if parent is None:
            break
        if _is_hlil_control_flow(parent):
            condition = getattr(parent, "condition", None)
            if condition is None:
                return None
            text = str(condition).strip()
            return text if _hlil_condition_is_meaningful(text) else None
        current = parent
    return None


# --- variadic (printf/scanf-family) format-argument recovery (#558) ----------
#
# Maps a libc variadic format function to (format_arg_index, is_scanf). The
# format arg is 0-based; is_scanf marks the families whose variadic arguments
# are DESTINATION POINTERS (writes into caller storage) rather than values, so
# under-recovery there silently hides parser field writes.
_VARIADIC_FORMAT_FAMILY: dict[str, tuple[int, bool]] = {
    # scanf family: variadic args are destination pointers
    "scanf": (0, True),
    "fscanf": (1, True),
    "sscanf": (1, True),
    # printf family: variadic args are values (%s is a source pointer)
    "printf": (0, False),
    "fprintf": (1, False),
    "sprintf": (1, False),
    "snprintf": (2, False),
    "dprintf": (1, False),
    "syslog": (1, False),
}

# Scanf/printf length modifiers to skip before the conversion specifier.
_FORMAT_LENGTH_MODS = set("hlLqjzt")


def _normalize_libc_name(name: str) -> str:
    """Strip PLT/GOT/import decorations and glibc wrappers from a callee name so
    ``__isoc99_sscanf`` / ``sscanf@plt`` / ``_sscanf`` all resolve to ``sscanf``."""
    n = str(name or "").strip()
    n = n.split("@", 1)[0]           # drop @plt / @got / @GLIBC_* decorations
    if n.startswith("__isoc99_"):
        n = n[len("__isoc99_"):]
    return n.lstrip("_")


def _variadic_format_family(name: str) -> tuple[int, bool] | None:
    """(format_arg_index, is_scanf) for a known printf/scanf-family callee, else None."""
    return _VARIADIC_FORMAT_FAMILY.get(_normalize_libc_name(name))


def _function_is_variadic(func) -> bool:
    """Whether BN's recovered prototype for *func* is variadic (``...``). Tolerates
    a BoolWithConfidence, a plain bool, or a type that lacks the attribute."""
    func_type = getattr(func, "type", None)
    if func_type is None:
        return False
    has_varargs = getattr(func_type, "has_variable_arguments", None)
    if has_varargs is None:
        return False
    value = getattr(has_varargs, "value", has_varargs)
    try:
        return bool(value)
    except Exception:
        return False


def _extract_format_literal(text: str) -> str | None:
    """Pull the C string body out of a rendered format argument like ``"%d %s"``
    (BN renders resolved string constants quoted). Returns None when *text* is not
    a simple double-quoted literal."""
    stripped = str(text or "").strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return None


def _count_format_conversions(fmt: str, *, is_scanf: bool) -> int:
    """Count the argument-consuming conversion specifiers in a printf/scanf format
    string. ``%%`` is skipped; a scanf ``%*`` (assignment suppression) consumes no
    pointer. Heuristic (width ``%*`` in printf and exotic specifiers are not modeled)
    -- used only to estimate an expected argument count, never asserted as fact."""
    count = 0
    i, n = 0, len(fmt)
    while i < n:
        if fmt[i] != "%":
            i += 1
            continue
        i += 1
        if i >= n:
            break
        if fmt[i] == "%":          # literal %%
            i += 1
            continue
        suppressed = is_scanf and fmt[i] == "*"
        if suppressed:
            i += 1
        while i < n and (fmt[i].isdigit() or fmt[i] in _FORMAT_LENGTH_MODS):
            i += 1
        if i >= n:
            break
        conv = fmt[i]
        if conv == "[":            # scanf scanset %[...] / %[^...]
            i += 1
            if i < n and fmt[i] == "^":
                i += 1
            if i < n and fmt[i] == "]":   # ] as the first member is literal
                i += 1
            while i < n and fmt[i] != "]":
                i += 1
            i += 1
        else:
            i += 1
        if not suppressed:
            count += 1
    return count


def _format_hlil_tree(ins, indent=0, *, _else_prefix=False, addresses: bool = True):
    """Recursively format HLIL tree with proper indentation."""
    lines = []
    pad = "    " * indent
    op = ins.operation.name

    BODY_INDENT = "    "
    if addresses:
        def _prefix(i):
            a = getattr(i, "address", None)
            return f"{hex(int(a))}        " if a is not None else "                "

        NO_PREFIX = "                "
    else:
        def _prefix(i):
            return BODY_INDENT

        NO_PREFIX = BODY_INDENT

    if op == "HLIL_NOP":
        pass

    elif op == "HLIL_BLOCK":
        for stmt in ins:
            lines.extend(_format_hlil_tree(stmt, indent, addresses=addresses))

    elif op == "HLIL_IF":
        if _else_prefix:
            lines.append(f"{_prefix(ins)}{pad}}} else if ({ins.condition})")
        else:
            lines.append(f"{_prefix(ins)}{pad}if ({ins.condition})")
        lines.append(f"{NO_PREFIX}{pad}{{")
        lines.extend(_format_hlil_tree(ins.true, indent + 1, addresses=addresses))
        false_branch = ins.false
        false_op = false_branch.operation.name
        if false_op == "HLIL_NOP":
            lines.append(f"{NO_PREFIX}{pad}}}")
        elif false_op == "HLIL_IF":
            lines.extend(_format_hlil_tree(false_branch, indent, _else_prefix=True, addresses=addresses))
        else:
            lines.append(f"{NO_PREFIX}{pad}}} else {{")
            lines.extend(_format_hlil_tree(false_branch, indent + 1, addresses=addresses))
            lines.append(f"{NO_PREFIX}{pad}}}")

    elif op in ("HLIL_WHILE", "HLIL_WHILE_SSA"):
        lines.append(f"{_prefix(ins)}{pad}while ({ins.condition})")
        lines.append(f"{NO_PREFIX}{pad}{{")
        lines.extend(_format_hlil_tree(ins.body, indent + 1, addresses=addresses))
        lines.append(f"{NO_PREFIX}{pad}}}")

    elif op in ("HLIL_DO_WHILE", "HLIL_DO_WHILE_SSA"):
        lines.append(f"{_prefix(ins)}{pad}do")
        lines.append(f"{NO_PREFIX}{pad}{{")
        lines.extend(_format_hlil_tree(ins.body, indent + 1, addresses=addresses))
        lines.append(f"{NO_PREFIX}{pad}}} while ({ins.condition})")

    elif op in ("HLIL_FOR", "HLIL_FOR_SSA"):
        lines.append(f"{_prefix(ins)}{pad}for ({ins.init}; {ins.condition}; {ins.update})")
        lines.append(f"{NO_PREFIX}{pad}{{")
        lines.extend(_format_hlil_tree(ins.body, indent + 1, addresses=addresses))
        lines.append(f"{NO_PREFIX}{pad}}}")

    elif op == "HLIL_SWITCH":
        lines.append(f"{_prefix(ins)}{pad}switch ({ins.condition})")
        lines.append(f"{NO_PREFIX}{pad}{{")
        for case in ins.cases:
            lines.extend(_format_hlil_tree(case, indent + 1, addresses=addresses))
        default = getattr(ins, "default", None)
        if default is not None and default.operation.name != "HLIL_NOP":
            lines.append(f"{NO_PREFIX}{pad}    default:")
            lines.extend(_format_hlil_tree(default, indent + 2, addresses=addresses))
        lines.append(f"{NO_PREFIX}{pad}}}")

    elif op == "HLIL_CASE":
        for val in ins.values:
            lines.append(f"{_prefix(ins)}{pad}case {val}:")
        lines.extend(_format_hlil_tree(ins.body, indent + 1, addresses=addresses))

    else:
        lines.append(f"{_prefix(ins)}{pad}{ins}")

    return lines


_VALID_IL_VIEWS = ("hlil", "mlil", "llil")


def _function_text(bv, func, *, view: str = "hlil", ssa: bool = False, addresses: bool = True) -> str:
    if view not in _VALID_IL_VIEWS:
        # Refuse an unrecognized view rather than silently substituting HLIL: the
        # caller stamps the result with the raw requested view string, so a fallback
        # would mislabel another IL layer's content as the requested view (#527).
        raise OperationFailure(
            "unsupported",
            f"unknown IL view {view!r}; expected one of {', '.join(_VALID_IL_VIEWS)}",
        )
    il_name = view
    try:
        il = getattr(func, il_name)
        if ssa and hasattr(il, "ssa_form") and il.ssa_form is not None:
            il = il.ssa_form
        if il_name == "hlil" and hasattr(il, "root"):
            try:
                lines = _format_hlil_tree(il.root, addresses=addresses)
                if lines:
                    return "\n".join(lines)
            except Exception:
                pass
        lines = []
        for ins in il.instructions:
            if addresses:
                address = getattr(ins, "address", func.start)
                lines.append(f"{hex(int(address))}        {ins}")
            else:
                lines.append(f"    {ins}")
        if lines:
            return "\n".join(lines)
    except Exception as exc:
        # Degrade to the prototype, but say so: a silent prototype-only
        # body with ok:true reads like a successful (empty) render.
        bn.log_warn(
            f"BN Agent Bridge: {view} rendering failed for {getattr(func, 'name', func)}: "
            f"{type(exc).__name__}: {exc}"
        )
        return (
            f"// bn: IL rendering failed ({type(exc).__name__}: {exc}); "
            f"showing prototype only\n{func}"
        )
    return str(func)


def _instruction_length(bv, address: int, *, arch=None, strict: bool = False) -> int:
    if arch is None:
        arch = getattr(bv, "arch", None)
    try:
        max_length = int(getattr(arch, "max_instr_length", 16) or 16)
    except Exception:
        max_length = 16

    if arch is not None and hasattr(arch, "get_instruction_info"):
        try:
            data = bv.read(address, max_length)
            info = arch.get_instruction_info(data, address)
            length = int(getattr(info, "length", 0))
            if length > 0:
                return length
        except Exception:
            pass

    # In strict mode the decode is FORCED to *arch* (e.g. linear --mode), so a
    # failed decode must NOT fall back to bv.get_instruction_length(): that uses
    # the BV-default arch and would return a wrong-mode length, contradicting the
    # forced-mode note (#382). Return 1 so the caller advances a single byte and
    # surfaces the honest `.byte` form instead of a mislabeled decode.
    if strict:
        return 1

    try:
        length = int(bv.get_instruction_length(address))
        if length > 0:
            return length
    except Exception:
        pass
    return 1


def _disasm_entry(bv, address: int, *, arch=None, strict: bool = False) -> dict[str, Any]:
    text = ""
    if arch is not None:
        try:
            max_length = int(getattr(arch, "max_instr_length", 16) or 16)
            data = bv.read(address, max_length)
            tokens, _length = arch.get_instruction_text(data, address)
            if tokens:
                text = "".join(str(t) for t in tokens)
        except Exception:
            pass
    # In strict mode (forced linear --mode) a failed forced-arch decode must NOT
    # fall back to bv.get_disassembly(): that decodes in the BV-default arch and
    # would print a wrong-mode instruction under a "forced via --mode" note
    # (#382). Leave text empty so the caller emits the honest `.byte` form.
    if not text and not strict:
        text = bv.get_disassembly(address) or ""
    return {
        "address": hex(int(address)),
        "text": text,
    }


def _structured_disasm_entries(bv, func) -> list[dict[str, Any]]:
    arch = getattr(func, "arch", None)
    entries = []
    for block in list(func.basic_blocks):
        addr = int(block.start)
        end = int(block.end)
        while addr < end:
            entry = _disasm_entry(bv, addr, arch=arch)
            if entry["text"]:
                entry["_address_int"] = addr
                entries.append(entry)
            addr += max(1, _instruction_length(bv, addr, arch=arch))
    entries.sort(key=lambda item: int(item["_address_int"]))
    return entries


def _disasm_text(bv, func) -> str:
    arch = getattr(func, "arch", None)
    entries = []
    for block in list(func.basic_blocks):
        addr = int(block.start)
        end = int(block.end)
        while addr < end:
            length = max(1, _instruction_length(bv, addr, arch=arch))
            entry = _disasm_entry(bv, addr, arch=arch)
            raw = bv.read(addr, length)
            text = entry["text"]
            if not text:
                if raw:
                    text = ".byte " + ", ".join(f"0x{byte:02x}" for byte in raw)
                else:
                    # No decode AND no bytes: a genuinely unreadable address (a
                    # gap inside an otherwise-mapped block, or a stale/partial
                    # mapping). Say so instead of emitting a byte directive with
                    # no byte -- syntactically broken and would misreport an
                    # unreadable range as "decoded to an empty instruction".
                    text = "<unreadable: no bytes available at this address>"
            entries.append((addr, raw, text))
            addr += length
    entries.sort(key=lambda item: item[0])
    return "\n".join(
        f"{hex(addr)}  {(raw.hex(' ') if raw else ''):<16} {text}"
        for addr, raw, text in entries
    )


def _function_signature(func) -> str:
    """Build a C-style function signature from Binary Ninja metadata."""
    func_type = getattr(func, "type", None)
    if func_type is None:
        return func.name
    return_type = getattr(func_type, "return_value", getattr(func, "return_type", None))
    ret = str(return_type) if return_type is not None else "void"
    params = []
    for var in list(func.parameter_vars):
        params.append(f"{var.type} {var.name}")
    return f"{ret} {func.name}({', '.join(params)})"


def _pseudo_c_text(func, *, addresses: bool = False) -> str:
    """Render Binary Ninja's Pseudo C for one function (GUI-equivalent).

    Walks the language-representation linear view a batch of lines at a
    time. Each line carries its own address, so the optional gutter matches
    what Binary Ninja shows in its UI. Comments are rendered inline by BN.
    """
    settings = bn.DisassemblySettings()
    # Suppress BN's built-in address column so `--addresses` controls the
    # gutter (and its format) on our side rather than doubling it. Keep the
    # explicit type casts (e.g. `*(uint8_t*)x`) that make the access width
    # legible — they are off in a default DisassemblySettings.
    settings.set_option(bn.DisassemblyOption.ShowAddress, False)
    settings.set_option(bn.DisassemblyOption.ShowTypeCasts, True)
    settings.set_option(bn.DisassemblyOption.WaitForIL, True)
    # Keep long statements (and string literals) on one line instead of
    # wrapping them into adjacent fragments — one statement per line is
    # easier to read, slice (--lines), and grep without splitting strings.
    settings.set_option(bn.DisassemblyOption.DisableLineFormatting, True)
    view_obj = bn.lineardisassembly.LinearViewObject.single_function_language_representation(func, settings)
    cursor = bn.lineardisassembly.LinearViewCursor(view_obj)
    cursor.seek_to_begin()
    out: list[str] = []
    seen_content = False
    while True:
        for line in cursor.lines:
            text = str(line.contents)
            if not text.strip():
                # Blank separator line: keep the spacing but never emit a
                # lone address in the gutter (decide blankness on content,
                # not on the prefixed string).
                out.append("")
                continue
            if not seen_content:
                # BN indents the function-header (signature) line by two
                # spaces; the braces and body don't share that indent, so
                # left-justify the header to line up with them.
                text = text.lstrip()
                seen_content = True
            if addresses:
                addr = getattr(line.contents, "address", None)
                prefix = f"{hex(int(addr))}        " if addr is not None else " " * 16
                out.append(f"{prefix}{text}")
            else:
                out.append(text)
        if not cursor.next():
            break
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def _decompile_text(bv, func, *, addresses: bool = False) -> str:
    """Pseudo C for a function, degrading to wrapped HLIL if it is unavailable."""
    marker = ""
    try:
        text = _pseudo_c_text(func, addresses=addresses)
    except Exception as exc:
        # Make the failure visible instead of silently returning the HLIL
        # fallback (or worse, an empty body) with ok:true.
        bn.log_warn(
            f"BN Agent Bridge: pseudo-C decompilation failed for "
            f"{getattr(func, 'name', func)}: {type(exc).__name__}: {exc}"
        )
        marker = (
            f"// bn: decompilation failed ({type(exc).__name__}: {exc}); "
            "showing HLIL fallback\n"
        )
        text = ""
    if text.strip():
        return text
    sig = _function_signature(func)
    body = _function_text(bv, func, view="hlil", addresses=addresses)
    if addresses:
        return f"{marker}{hex(int(func.start))}        {sig}\n{body}"
    return f"{marker}{sig}\n{{\n{body}\n}}"


def _analysis_stub_warning(func, text: str, *, forced: bool = False) -> str | None:
    """Warn when a decompile body is a Binary Ninja analysis stub, not a real body.

    BN skips analysis for oversized functions and renders a placeholder
    instead of a body. The authoritative signal is ``func.analysis_skipped``;
    a distinctive-phrase text match is kept as a fallback.
    """
    skipped = bool(getattr(func, "analysis_skipped", False))
    lowered = text.lower()
    placeholder = (
        len(text) <= 512
        and "this function is taking too long to analyze" in lowered
        and ("loading..." in lowered or "loading…" in lowered)
    )
    if not (skipped or placeholder):
        return None
    reason = None
    try:
        raw = func.analysis_skip_reason
        # BN's AnalysisSkipReason is an IntEnum; on Python 3.11+ str() yields
        # the bare number, so prefer the member name.
        reason = getattr(raw, "name", None) or str(raw)
    except Exception:
        reason = None
    detail = f" (skip reason: {reason})" if reason else ""
    if forced:
        return (
            f"{func.name}: Binary Ninja could not complete analysis even after --force-analysis{detail}; "
            f"this decompile is still an incomplete stub, not the real function body."
        )
    return (
        f"Binary Ninja skipped analysis for {func.name}{detail}; this decompile is an incomplete stub, "
        f"not the real function body. Re-run with --force-analysis to analyze it (may be slow on large functions)."
    )


def _il_function_for(func, view: str, ssa: bool):
    if view not in _VALID_IL_VIEWS:
        # Refuse an unrecognized structured-IL view rather than silently falling
        # back to MLIL: the caller labels the result with the raw view string, so a
        # fallback would mislabel another IL layer's instructions as requested (#527).
        raise OperationFailure(
            "unsupported",
            f"unknown IL view {view!r}; expected one of {', '.join(_VALID_IL_VIEWS)}",
        )
    attr = view
    il = getattr(func, attr, None)
    if il is None:
        raise OperationFailure("unsupported", f"function has no {view.upper()}")
    if ssa:
        ssa_form = getattr(il, "ssa_form", None)
        if ssa_form is not None:
            il = ssa_form
    return il


def _ssa_var_entry(v) -> dict[str, Any]:
    """Serialize an SSAVariable or plain Variable consistently.

    SSA vars expose ``.var`` (-> Variable) and ``.version``; AddressOf
    targets surface as plain Variables (no version) in ``vars_read``.
    """
    base = getattr(v, "var", v)
    version = getattr(v, "version", None)
    name = str(getattr(base, "name", base))
    entry: dict[str, Any] = {
        "name": name,
        "version": int(version) if version is not None else None,
        "ssa": f"{name}#{version}" if version is not None else name,
        "type": str(getattr(base, "type", "")) or None,
        "identifier": vars_mod._variable_identifier(base),
    }
    return entry


def _collect_ssa_vars(il) -> dict[tuple[str, int], Any]:
    found: dict[tuple[str, int], Any] = {}
    try:
        items = list(il.instructions)
    except Exception:
        items = []
    for ins in items:
        for v in list(getattr(ins, "vars_read", None) or []) + list(getattr(ins, "vars_written", None) or []):
            version = getattr(v, "version", None)
            if version is None:
                continue
            base = getattr(v, "var", v)
            found[(str(getattr(base, "name", base)), int(version))] = v
    return found


def _resolve_ssa_variable(func, il, selector: str):
    index = _collect_ssa_vars(il)
    name, sep, version = str(selector).partition("#")
    if sep and version:
        key = (name, int(version))
        if key in index:
            return index[key], None
        raise OperationFailure("unsupported", f"SSA variable not found: {selector}")
    # bare name: return the lowest-version instance plus the list of versions
    versions = sorted(v for (n, v) in index if n == name)
    if not versions:
        raise OperationFailure("unsupported", f"SSA variable not found: {selector}")
    return index[(name, versions[0])], versions


def _serialize_pvs(pvs) -> dict[str, Any] | None:
    if pvs is None:
        return None
    out: dict[str, Any] = {"raw": str(pvs)}
    t = getattr(pvs, "type", None)
    out["type"] = getattr(t, "name", None) or (str(t) if t is not None else None)

    def _coerce(v):
        try:
            return int(v)
        except Exception:
            return str(v)

    value = getattr(pvs, "value", None)
    if value is not None:
        out["value"] = _coerce(value)
    values = getattr(pvs, "values", None)
    if values:
        try:
            out["values"] = sorted(_coerce(v) for v in values)
        except Exception:
            out["values"] = [_coerce(v) for v in values]
    ranges = getattr(pvs, "ranges", None)
    if ranges:
        out["ranges"] = [
            {
                "start": _coerce(getattr(r, "start", 0)),
                "end": _coerce(getattr(r, "end", 0)),
                "step": _coerce(getattr(r, "step", 1)),
            }
            for r in ranges
        ]
    return out


def _pvs_determined(pvs) -> bool:
    """True if a PossibleValueSet carries an actual value (not BN's
    UndeterminedValue). Used to prefer a determined source-expression
    value-set over an undetermined instruction-level one (#52)."""
    if pvs is None:
        return False
    tname = str(getattr(getattr(pvs, "type", None), "name", "") or "")
    if tname:
        return tname != "UndeterminedValue"
    return "undetermined" not in str(pvs).lower()


def _iter_il_instructions(il_func):
    if il_func is None:
        return []
    instructions = []
    try:
        blocks = list(il_func)
    except Exception:
        blocks = list(getattr(il_func, "basic_blocks", []) or [])
    for block in blocks:
        try:
            instructions.extend(list(block))
        except Exception:
            continue
    return instructions


def _render_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    if "__offset(" in text:
        warnings.append(
            "Decompile still contains raw __offset(...) expressions; use `bn types show` or `bn struct show` as the authoritative layout until Binary Ninja refreshes the presentation."
        )
    return warnings
