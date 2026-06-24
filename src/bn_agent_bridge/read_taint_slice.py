"""Taint entry point + interprocedural backward-slice machinery.

The ``taint`` op and the SSA backward-slice walker that used to live on
``BinaryNinjaBridge`` move here as module-level free functions, each taking the
``BridgeContext`` seam (``ctx``) in place of ``self``. ``BinaryNinjaBridge``
keeps a thin delegating shim for every name the test suite / op binders
reference (``_taint``, ``_backward_slice``, ``_build_backward_trace``,
``_is_parameter_ssa_var``, ``_resolve_callee``, ``_resolve_thunk``,
``_extract_dest_address``, ``_find_return_vars``, ``_ssa_vars_from``).

Outbound calls resolve through:
  * ``ctx`` -- resolution helpers relocated to the seam (``_resolve_view``,
    ``_find_function``);
  * ``il_format`` -- the state-free IL helpers (``_il_op_name``,
    ``_iter_il_instructions``);
  * ``vars`` -- variable discovery (``_find_variable_selector``);
  * ``taint_engine`` (imported as ``_taint``, matching ``bridge.py``) -- the
    taint model loader + engine + locator/call-target/thunk resolution
    (``load_models``, ``TaintEngine``, ``parse_locator``, ``resolve_call_target``,
    ``follow_thunk``, ``extract_dest_address``, ``TaintError``);
  * ``_shared`` -- module-free helpers (``_parse_address``, ``OperationFailure``).

Import direction is one-way: this module imports ``il_format``, ``vars``,
``taint_engine``, and ``_shared`` (plus stdlib + binaryninja). It NEVER imports
``bridge`` or ``seam`` -- those import THIS module one-way (design spec §3.2).
"""
from __future__ import annotations

from typing import Any

import binaryninja as bn  # noqa: F401  (kept for parity with sibling read_* modules)
from binaryninja import SSAVariable

from . import il_format
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import OperationFailure, _parse_address
from .bridge_state import require_analysis


def _taint_op(ctx, selector, params: dict[str, Any]):
    bv = ctx._resolve_view(selector)
    require_analysis(bv, "Taint analysis")
    direction = str(params.get("direction", "forward"))
    func = ctx._find_function(bv, params["function"])
    try:
        # `user_models` carries project-internal wrapper models supplied via
        # `taint --models <file>` (#317); merged over the builtin/override DB.
        models = _taint.load_models(extra=params.get("user_models"))
    except _taint.TaintError as exc:
        # A broken builtin DB or BN_TAINT_MODELS override is now loud instead
        # of silently producing false negatives (#97).
        raise OperationFailure("unsupported", str(exc)) from exc

    def _find_variable(fn, sel):
        var, _is_param = vars_mod._find_variable_selector(fn, sel)
        return var

    engine = _taint.TaintEngine(
        bv,
        models,
        find_variable=_find_variable,
        unknown_call_policy=str(params.get("unknown_call", "conservative")),
        resolve_map=params.get("resolve_map") or {},
    )
    try:
        if direction == "forward":
            locators = [_taint.parse_locator(s) for s in (params.get("sources") or [])]
            if not locators:
                raise _taint.TaintError("forward taint needs at least one --source")
            result = engine.forward(
                func, locators,
                max_depth=int(params.get("max_depth", 8)),
                enabled_sink_classes=set(params.get("enabled_sink_classes") or []),
            )
        elif direction == "backward":
            locators = [_taint.parse_locator(s) for s in (params.get("sinks") or [])]
            if not locators:
                raise _taint.TaintError("backward taint needs at least one --sink")
            result = engine.backward(func, locators, max_depth=int(params.get("max_depth", 8)))
        else:
            raise OperationFailure("unsupported", f"unknown taint direction: {direction}")
    except _taint.TaintError as exc:
        raise OperationFailure("unsupported", str(exc)) from exc
    # #415: disclose which taint-model overlays were in effect for this run, so an
    # agent can confirm a project-local --models / BN_TAINT_MODELS file landed
    # (load_models reads them per request -- no bridge restart needed).
    if isinstance(result, dict):
        result["model_sources"] = _taint.model_overlay_sources(params.get("user_models"))
    return result


def _ssa_vars_from(vars_list: list) -> list[SSAVariable]:
    return [v for v in vars_list if isinstance(v, SSAVariable)]


def _ssa_label(ssa_var) -> str:
    """Stable analyst-friendly label for an SSA var, e.g. ``var_10#3`` -- avoids
    the verbose ``<SSAVariable ...>`` repr some BN builds emit for ``str()`` and
    keeps trace JSON machine-consumable (#162)."""
    base = getattr(ssa_var, "var", ssa_var)
    name = getattr(base, "name", None)
    version = getattr(ssa_var, "version", None)
    if name and version is not None:
        return f"{name}#{version}"
    return str(ssa_var)


def _const_int(expr) -> int | None:
    if expr is None or "CONST" not in il_format._il_op_name(expr):
        return None
    c = getattr(expr, "constant", None)
    try:
        return int(c) if c is not None else None
    except (TypeError, ValueError):
        return None


def _addr_base_offset_label(expr, depth: int = 0) -> tuple[str | None, int | None]:
    """Best-effort (base_label, offset) for a load address expression: a base
    pointer var plus a constant field offset. Either may be None when the address
    isn't a simple ``base [+ const]`` shape."""
    if expr is None or depth > 6:
        return None, None
    op = il_format._il_op_name(expr)
    if "ADD" in op or "SUB" in op:
        left = getattr(expr, "left", None)
        right = getattr(expr, "right", None)
        rc = _const_int(right)
        if rc is not None:
            bl, _ = _addr_base_offset_label(left, depth + 1)
            return bl, (rc if "ADD" in op else -rc)
        lc = _const_int(left)
        if lc is not None and "ADD" in op:
            br, _ = _addr_base_offset_label(right, depth + 1)
            return br, lc
        return None, None
    reads = _ssa_vars_from(getattr(expr, "vars_read", []) or [])
    if len(reads) == 1:
        return _ssa_label(reads[0]), 0
    return None, None


def _load_expr_of(def_insn):
    """The load expression a definition reads from, or None. Handles both MLIL
    forms: a top-level load def, and the common ``x = [addr]`` SET_VAR whose
    ``.src`` is the load."""
    if "LOAD" in il_format._il_op_name(def_insn):
        return def_insn
    src = getattr(def_insn, "src", None)
    if src is not None and "LOAD" in il_format._il_op_name(src):
        return src
    return None


def _field_load_meta(load_expr) -> tuple[str | None, int | None, int | None]:
    """(base_label, offset, width) for a load expression, best-effort. Lets a
    field load (`*(obj + off)`) carry structured metadata in the trace instead of
    just a `memory_load` reason (#162)."""
    width = getattr(load_expr, "size", None)
    base, offset = _addr_base_offset_label(getattr(load_expr, "src", None))
    return base, offset, (int(width) if isinstance(width, int) else None)


def _is_address_of(expr) -> bool:
    return "ADDRESS_OF" in il_format._il_op_name(expr)


def _arg_register(caller_func, arg_index: int) -> str | None:
    """The calling-convention integer-arg register for *arg_index* (e.g. `x1`,
    `rsi`, `r1`), or None when the convention isn't recoverable (#166)."""
    try:
        cc = getattr(caller_func, "calling_convention", None)
        regs = list(getattr(cc, "int_arg_regs", []) or [])
        if 0 <= arg_index < len(regs):
            return str(regs[arg_index])
    except Exception:
        return None
    return None


def _int_arg_reg_count(caller_func) -> int | None:
    """Number of integer-argument registers in the function's calling convention
    (6 on x86-64 SysV, 8 on AArch64, 4 on ARM), or None if unrecoverable. An arg
    at or beyond this index is passed on the STACK, where BN's MLIL/HLIL call
    model often omits it (#324)."""
    try:
        cc = getattr(caller_func, "calling_convention", None)
        regs = list(getattr(cc, "int_arg_regs", []) or [])
        return len(regs) if regs else None
    except Exception:
        return None


def _arg_label(ctx, bv, call_insn, arg_index: int, caller_func) -> dict[str, Any]:
    """{index, [register], [name]} for the traced call argument: its
    calling-convention register plus the callee's C parameter name when the
    callee resolves (#166)."""
    label: dict[str, Any] = {"index": arg_index}
    reg = _arg_register(caller_func, arg_index)
    if reg:
        label["register"] = reg
    try:
        rt = _taint.resolve_call_target(bv, call_insn, follow_thunks=True)
        callee = getattr(rt, "function", None)
        pvars = list(getattr(callee, "parameter_vars", []) or []) if callee else []
        if 0 <= arg_index < len(pvars):
            nm = getattr(pvars[arg_index], "name", None)
            if nm:
                label["name"] = str(nm)
    except Exception:
        pass
    return label


def _build_backward_trace(
    ctx,
    bv,
    ssa_func,
    initial_vars: list,
    max_depth: int,
    *,
    interprocedural: bool = False,
    ip_depth: int = 2,
    view: str = "mlil",
    _call_depth: int = 0,
    base_depth: int = 0,
) -> list[dict[str, Any]]:
    """Recursively walk SSA use-def chains backward, optionally crossing call boundaries."""
    if _call_depth > 10:
        return []  # Safety: prevent runaway recursion
    trace: list[dict[str, Any]] = []
    # Each worklist item carries its def-use distance from the seed so the
    # reported "depth" is the real graph depth (operands of one definition
    # share a depth) rather than a sequential append index. base_depth
    # offsets a callee sub-walk so its depths continue from the call site.
    worklist: list[tuple[Any, int]] = [(v, 0) for v in initial_vars]
    visited: set[Any] = set()

    while worklist and len(trace) < max_depth:
        ssa_var, node_depth = worklist.pop(0)
        depth = base_depth + node_depth
        if not isinstance(ssa_var, SSAVariable):
            trace.append({
                "ssa_var": str(ssa_var),
                "ssa_label": str(ssa_var),
                "depth": depth,
                "terminates": True,
                "reason": "undefined_or_global",
            })
            continue
        if ssa_var in visited:
            continue
        visited.add(ssa_var)

        try:
            def_insn = ssa_func.get_ssa_var_definition(ssa_var)
        except AttributeError:
            if interprocedural:
                continue
            raise OperationFailure(
                "no_ssa_trace",
                f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
            )
        entry: dict[str, Any] = {
            "ssa_var": str(ssa_var),
            "ssa_label": _ssa_label(ssa_var),
            "depth": depth,
        }

        if def_insn is None:
            # No reaching definition: a real parameter, or an undefined
            # local / global. Only claim "function parameter" when it
            # actually is one; otherwise stay neutral (don't mislead
            # provenance slices).
            entry["terminates"] = True
            entry["reason"] = (
                "function_parameter"
                if _is_parameter_ssa_var(ctx, ssa_func, ssa_var)
                else "undefined_or_global"
            )
            trace.append(entry)
            continue

        entry["address"] = hex(int(getattr(def_insn, "address", 0)))
        entry["il_text"] = str(def_insn)
        entry["operation"] = il_format._il_op_name(def_insn)

        def_op = entry["operation"]
        if "CALL" in def_op or "JUMP" in def_op:
            if interprocedural and ip_depth > 0:
                callee = _resolve_callee(ctx, bv, def_insn)
                if callee is not None:
                    callee_mlil = getattr(callee, "medium_level_il", None)
                    if callee_mlil is not None and callee_mlil.ssa_form is not None:
                        if hasattr(callee_mlil.ssa_form, "get_ssa_var_definition"):
                            callee_ret_vars = _find_return_vars(ctx, callee_mlil.ssa_form, bv)
                            if callee_ret_vars:
                                entry["cross_function"] = True
                                entry["callee"] = callee.name
                                entry["terminates"] = False
                                entry["reason"] = "cross_function"
                                trace.append(entry)
                                callee_trace = _build_backward_trace(
                                    ctx, bv, callee_mlil.ssa_form, callee_ret_vars,
                                    max_depth - len(trace),
                                    interprocedural=True,
                                    ip_depth=ip_depth - 1,
                                    view=view,
                                    _call_depth=_call_depth + 1,
                                    base_depth=depth + 1,
                                )
                                for ct in callee_trace:
                                    ct.setdefault("function_context", callee.name)
                                trace.extend(callee_trace)
                                continue
            entry["terminates"] = True
            entry["reason"] = "call_or_jump_boundary"
            # Resolve the call target to its symbol so a library-call origin reads
            # as e.g. `call strlen` rather than a bare PLT address (#193).
            callee_nm = _callee_display_name(ctx, bv, def_insn)
            if callee_nm:
                entry["callee"] = callee_nm
            trace.append(entry)
            continue

        load_expr = _load_expr_of(def_insn)
        if load_expr is not None:
            base, offset, width = _field_load_meta(load_expr)
            # A resolvable base+offset means this is a struct/field load, not an
            # opaque pointer deref -- label it `field_load` and carry the
            # structured fields so the slice is machine-consumable (#162).
            entry["reason"] = "field_load" if base is not None else "memory_load"
            if base is not None:
                entry["base"] = base
            if offset is not None:
                entry["offset"] = hex(offset) if isinstance(offset, int) else offset
            if width is not None:
                entry["width"] = width
            # Preserve prior per-form walk behavior: a top-level load def
            # terminated; a `x = [addr]` SET_VAR continued through its base
            # pointer (so provenance reaches where the struct came from).
            entry["terminates"] = "LOAD" in def_op
            trace.append(entry)
            for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
                if rv not in visited:
                    worklist.append((rv, node_depth + 1))
            continue

        entry["terminates"] = False
        # Populate `reason` from a controlled vocabulary instead of leaving it
        # null on ordinary steps: a phi merge is `phi_source`, everything else is
        # a plain `definition` (#162).
        entry["reason"] = "phi_source" if "PHI" in def_op else "definition"
        trace.append(entry)

        for rv in _ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
            if rv not in visited:
                worklist.append((rv, node_depth + 1))

    return trace


def _is_parameter_ssa_var(ctx, ssa_func, ssa_var) -> bool:
    """True if *ssa_var* (an SSA variable with no reaching definition) is a
    formal parameter of the function, vs. an undefined local or a global.

    Matched against the source function's ``parameter_vars`` by identifier
    first, then by base name. Returns False whenever parameter information
    is unavailable, so the caller falls back to a neutral label rather than
    guessing 'parameter'."""
    source = getattr(ssa_func, "source_function", None)
    params = list(getattr(source, "parameter_vars", []) or [])
    if not params:
        return False
    base = getattr(ssa_var, "var", ssa_var)
    ident = getattr(base, "identifier", None)
    base_name = str(ssa_var).split("#")[0]
    for param in params:
        if ident is not None and getattr(param, "identifier", None) == ident:
            return True
        if base_name and str(getattr(param, "name", param)) == base_name:
            return True
    return False


def _resolve_callee(ctx, bv, call_insn):
    """Resolve a call instruction's callee to a BN function, or None.

    Thin wrapper over the canonical resolver in ``taint_engine``; follows
    thunks/veneers (single-instruction tailcalls) to their real target so
    interprocedural tracing works through PLT stubs and GCC thunks.
    """
    return _taint.resolve_call_target(bv, call_insn, follow_thunks=True).function


def _callee_display_name(ctx, bv, def_insn) -> str | None:
    """Best-effort symbol name for a call/jump target, so a value that
    originates at a call boundary names the callee instead of just showing the
    raw PLT target address -- the same resolution ``taint backward`` already
    performs (#193). Returns None for a genuinely indirect/unresolved target.
    """
    try:
        fn = _resolve_callee(ctx, bv, def_insn)
    except Exception:
        fn = None
    if fn is not None and getattr(fn, "name", None):
        return str(fn.name)
    # No resolvable function (a PLT stub with only a symbol, or an extern): fall
    # back to a symbol at the raw dest address, Thumb-bit normalized.
    dest = getattr(def_insn, "dest", None)
    if dest is None:
        return None
    try:
        addr = _extract_dest_address(bv, dest)
    except Exception:
        addr = None
    if addr is None:
        return None
    for cand in (addr, addr & ~1):
        try:
            sym = bv.get_symbol_at(cand)
        except Exception:
            sym = None
        if sym is not None and getattr(sym, "name", None):
            return str(sym.name)
    return None


def _resolve_thunk(ctx, bv, fn):
    """If fn is a single-instruction tailcall thunk, return its real target
    (delegates to ``taint_engine.follow_thunk``)."""
    return _taint.follow_thunk(bv, fn)


def _extract_dest_address(bv, dest):
    """Numeric address of a call/tailcall destination expression (delegates
    to ``taint_engine.extract_dest_address``)."""
    return _taint.extract_dest_address(bv, dest)


def _find_return_vars(ctx, ssa_func, bv=None, _visited=None) -> list[SSAVariable]:
    """Find SSA variables that feed into RET instructions in a function.

    For functions that only contain a TAILCALL (PLT stubs, thunks), follows
    the tailcall to the real implementation and returns its return vars.
    """
    ret_vars: list[SSAVariable] = []
    has_ret = False
    for block in getattr(ssa_func, "basic_blocks", []) or []:
        for insn in block:
            op_name = il_format._il_op_name(insn)
            if op_name == "MLIL_RET":
                has_ret = True
                found = _ssa_vars_from(getattr(insn, "vars_read", []) or [])
                if not found:
                    src = getattr(insn, "src", []) or []
                    for s in src:
                        var = getattr(s, "var", None)
                        if var is not None and isinstance(var, SSAVariable):
                            found.append(var)
                if not found:
                    dest = getattr(insn, "dest", None)
                    if dest is not None:
                        found = _ssa_vars_from([dest] if not isinstance(dest, list) else dest)
                if not found:
                    non_ssa = getattr(insn, "non_ssa_form", None)
                    if non_ssa is not None:
                        found = _ssa_vars_from(getattr(non_ssa, "vars_read", []) or [])
                ret_vars.extend(found)
    # No MLIL_RET found — try to follow TAILCALL to real implementation
    if not has_ret and bv is not None:
        if _visited is None:
            _visited = set()
        fn_key = id(ssa_func)
        if fn_key in _visited:
            return ret_vars  # Cycle detected
        _visited.add(fn_key)
        for block in getattr(ssa_func, "basic_blocks", []) or []:
            for insn in block:
                if "TAILCALL" not in il_format._il_op_name(insn):
                    continue
                dest = getattr(insn, "dest", None)
                if dest is None:
                    break
                fn_source = getattr(ssa_func, "source_function", None)
                source_start = getattr(fn_source, "start", None)
                addr = _extract_dest_address(bv, dest)
                if addr is not None:
                    if source_start is not None and (addr == source_start or addr & ~1 == source_start):
                        break  # Self-loop (PLT stub → itself)
                    target = bv.get_function_at(addr)
                    if target is None:
                        target = bv.get_function_at(addr & ~1)
                    if target is not None and source_start is not None and target.start == source_start:
                        break  # Self-loop via different resolution path
                    if target is not None:
                        callee_mlil = getattr(target, "medium_level_il", None)
                        if callee_mlil and callee_mlil.ssa_form:
                            return _find_return_vars(ctx, callee_mlil.ssa_form, bv, _visited)
                break  # Only try the first instruction
    return ret_vars


def _backward_slice(
    ctx,
    selector: str | None,
    identifier: str,
    address: str,
    *,
    arg_index: int = 0,
    view: str = "mlil",
    max_depth: int = 50,
    interprocedural: bool = False,
    ip_depth: int = 2,
) -> dict[str, Any]:
    if max_depth < 1:
        raise OperationFailure("invalid_max_depth", f"Invalid max_depth: {max_depth}")
    bv = ctx._resolve_view(selector)
    func = ctx._find_function(bv, identifier)
    target_addr = _parse_address(address)

    il_name = {"mlil": "medium_level_il", "llil": "low_level_il", "hlil": "high_level_il"}.get(view)
    if il_name is None:
        raise OperationFailure("invalid_view", f"Unsupported IL view: {view}")

    il_func = getattr(func, il_name, None)
    if il_func is None:
        raise OperationFailure("no_il", f"Function {func.name} has no {view}")
    ssa_func = il_func.ssa_form
    if ssa_func is None:
        raise OperationFailure("no_ssa", f"Function {func.name} has no {view} SSA form")

    if not hasattr(ssa_func, "get_ssa_var_definition"):
        raise OperationFailure(
            "no_ssa_trace",
            f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
        )

    ssa_instructions = il_format._iter_il_instructions(ssa_func)
    call_insn = None
    for insn in ssa_instructions:
        if int(getattr(insn, "address", 0)) != target_addr:
            continue
        # "CALL" also matches TAILCALL/SYSCALL op names.
        if "CALL" in il_format._il_op_name(insn):
            call_insn = insn
            break

    if call_insn is None:
        hint = ""
        if view != "mlil":
            hint = " (try --view mlil, which has the broadest call coverage)"
        raise OperationFailure(
            "instruction_not_found",
            f"No call instruction at {address} in {func.name}{hint}",
        )

    params = list(getattr(call_insn, "params", []) or [])
    if not params:
        raise OperationFailure(
            "no_params",
            f"Call at {address} has no exposed parameters in {view}",
        )
    if arg_index < 0 or arg_index >= len(params):
        n = len(params)
        only = " (index 0)" if n == 1 else f" (indices 0..{n - 1})"
        # --arg is 0-based against the MLIL call's recovered parameters, which
        # can differ from the argument positions the decompiler renders (an
        # implicit/struct-return or coalesced arg shifts the count). State the
        # convention so a user reading pseudo-C doesn't reach for the wrong
        # index (#226).
        msg = (
            f"Argument index {arg_index} out of range: this call has {n} "
            f"MLIL argument(s){only}. --arg is 0-based and indexes the MLIL call "
            f"parameters, which may differ from the decompiler's displayed args."
        )
        # #324: an index at/beyond the calling convention's integer-arg registers
        # is passed on the STACK, which BN's MLIL/HLIL call model frequently omits
        # (the stores are visible only in LLIL). Don't silently treat such an arg
        # as absent -- say it is likely stack-passed and point at the LLIL view.
        reg_count = _int_arg_reg_count(func)
        if reg_count is not None and arg_index >= reg_count:
            msg += (
                f" Note: arg {arg_index} is at/beyond the {reg_count} integer-arg "
                f"register(s) of this calling convention, so it is likely passed on "
                f"the STACK; BN's MLIL/HLIL call model often omits stack-passed "
                f"(e.g. variadic) args -- inspect `bn il {func.name} --view llil "
                f"--ssa` for the stack stores feeding the call (#324)."
            )
        raise OperationFailure("invalid_arg_index", msg)

    param_expr = params[arg_index]
    initial_vars: list[Any] = _ssa_vars_from(getattr(param_expr, "vars_read", []) or [])

    arg_label = _arg_label(ctx, bv, call_insn, arg_index, func)

    # An address-of arg with no SSA value reads is an output-pointer dead-end:
    # tracing it would follow where the *pointer* came from (a local buffer),
    # not the data the callee writes through it. Surface that instead of the
    # misleading "constant or immediate -- no SSA trace" (#166).
    hints: list[str] = []
    if not initial_vars and _is_address_of(param_expr):
        callee_nm = arg_label.get("name") or "the callee"
        hints.append(
            f"arg {arg_index} is a pointer (address-of); this traces where the "
            f"pointer came from, not the data written through it. To follow data "
            f"{callee_nm} writes into the pointee, run a forward taint from the "
            f"call site (e.g. `taint forward --source call:<callee>`) or trace the "
            f"buffer's later consumers."
        )
    elif not initial_vars:
        # A constant/immediate arg (e.g. a literal length/flag) has no SSA
        # definition to trace. Surface the value as a structured hint so both
        # text and JSON consumers see *which* constant -- not just the renderer's
        # generic "constant or immediate" line (which carried no value and left
        # JSON `hints` empty).
        cval = _const_int(param_expr)
        if cval is not None:
            hints.append(
                f"arg {arg_index} is the constant {hex(cval)} -- a compile-time "
                f"immediate with no SSA definition to trace back."
            )

    trace = _build_backward_trace(
        ctx, bv, ssa_func, initial_vars, max_depth,
        interprocedural=interprocedural,
        ip_depth=ip_depth,
        view=view,
    )

    return {
        "function": func.name,
        "function_address": hex(func.start),
        "target_address": hex(target_addr),
        "arg_index": arg_index,
        "arg_label": arg_label,
        "view": view,
        "interprocedural": interprocedural,
        "ip_depth": ip_depth if interprocedural else 0,
        "truncated": len(trace) >= max_depth,
        "step_count": len(trace),
        "trace": trace,
        "hints": hints,
    }
