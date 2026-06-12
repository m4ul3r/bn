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

import binaryninja as bn

from . import vars as vars_mod
from ._shared import OperationFailure


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
    return {
        "prototype": str(func_type),
        "return_type": str(return_type) if return_type is not None else None,
        "calling_convention": str(calling_convention) if calling_convention is not None else None,
        "size": _function_size(func),
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

    for root in roots:
        current = root
        best_expression = None
        assignment_candidate = None
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
                break

            parent_text = str(parent)
            if not _is_hlil_trivial_wrapper(parent) and _hlil_text_is_local(parent_text):
                best_expression = parent
            current = parent

        if best_expression is not None:
            return best_expression
        if assignment_candidate is not None:
            return assignment_candidate
    return None


def _hlil_statement_text(insn) -> str | None:
    node = _select_local_hlil_node(insn)
    if node is None:
        return None
    text = str(node)
    return text if _hlil_text_is_local(text) else None


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


def _format_hlil_tree(ins, indent=0, *, _else_prefix=False, addresses: bool = True):
    """Recursively format HLIL tree with proper indentation."""
    lines = []
    pad = "    " * indent
    op = ins.operation.name

    BODY_INDENT = "    "
    if addresses:
        def _prefix(i):
            a = getattr(i, "address", None)
            return f"{int(a):08x}        " if a is not None else "                "

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


def _function_text(bv, func, *, view: str = "hlil", ssa: bool = False, addresses: bool = True) -> str:
    il_name = {"hlil": "hlil", "mlil": "mlil", "llil": "llil"}.get(view, "hlil")
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
                lines.append(f"{int(address):08x}        {ins}")
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


def _instruction_length(bv, address: int, *, arch=None) -> int:
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

    try:
        length = int(bv.get_instruction_length(address))
        if length > 0:
            return length
    except Exception:
        pass
    return 1


def _disasm_entry(bv, address: int, *, arch=None) -> dict[str, Any]:
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
    if not text:
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
    lines = []
    for block in list(func.basic_blocks):
        addr = block.start
        while addr < block.end:
            length = max(1, _instruction_length(bv, int(addr), arch=arch))
            entry = _disasm_entry(bv, addr, arch=arch)
            disasm = entry["text"]
            raw = bv.read(addr, length)
            hex_bytes = raw.hex(" ") if raw else ""
            lines.append(f"{addr:08x}  {hex_bytes:<16} {disasm}")
            addr += length
    return "\n".join(lines)


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
                prefix = f"{int(addr):08x}        " if addr is not None else " " * 16
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
        return f"{marker}{int(func.start):08x}        {sig}\n{body}"
    return f"{marker}{sig}\n{{\n{body}\n}}"


def _analysis_stub_warning(func, text: str, *, forced: bool = False) -> str | None:
    """Warn when a decompile body is a Binary Ninja analysis stub, not a real body.

    BN skips analysis for oversized functions and renders a placeholder
    instead of a body. The authoritative signal is ``func.analysis_skipped``;
    a distinctive-phrase text match is kept as a fallback.
    """
    skipped = bool(getattr(func, "analysis_skipped", False))
    placeholder = "taking too long to analyze" in text
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
    attr = {"hlil": "hlil", "mlil": "mlil", "llil": "llil"}.get(view, "mlil")
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
