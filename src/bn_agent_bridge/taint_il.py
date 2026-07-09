"""MLIL-SSA / call-target helpers for the taint engine.

Import-free of Binary Ninja: duck-typed over whatever objects the bridge
passes (and the synthetic fakes in unit tests).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

def reaching_reg_def(call_ins: Any, reg: str, bn: Any) -> Any | None:
    """Nearest dominating LLIL-SSA definition of register ``reg`` reaching
    ``call_ins``: walk the immediate-dominator chain from the call's block,
    taking the latest def of ``reg`` that precedes the call. This is the
    dominance-correct reaching def (it includes a ``REG_PHI`` when the value
    joins from multiple paths), so the recovered argument is never seeded from a
    non-reaching definition. Returns the defining LLIL instruction or None."""
    Op = bn.LowLevelILOperation
    call_ops = {getattr(Op, n, None) for n in (
        "LLIL_CALL_SSA", "LLIL_SYSCALL_SSA", "LLIL_INTRINSIC_SSA",
        "LLIL_TAILCALL_SSA")} - {None}

    def defines_reg(ins: Any) -> bool:
        op = getattr(ins, "operation", None)
        if op in (Op.LLIL_SET_REG_SSA, Op.LLIL_REG_PHI):
            return str(getattr(getattr(ins, "dest", None), "reg", "")) == reg
        if op in call_ops:
            return any(str(getattr(o, "reg", "")) == reg
                       for o in (getattr(ins, "output", None) or []))
        return False

    call_bb = getattr(call_ins, "il_basic_block", None)
    call_idx = int(getattr(call_ins, "instr_index", -1))
    bb = call_bb
    seen: set[int] = set()
    while bb is not None and id(bb) not in seen:
        seen.add(id(bb))
        best = None
        for ins in bb:
            ii = int(getattr(ins, "instr_index", -1))
            if bb is call_bb and ii >= call_idx:
                continue
            if defines_reg(ins) and (best is None or ii > int(getattr(best, "instr_index", -1))):
                best = ins
        if best is not None:
            return best
        bb = getattr(bb, "immediate_dominator", None)
    return None


def reaching_arg_seed_vars(func: Any, addr: int, reg: str,
                           bn: Any) -> list[tuple[Any, Any]]:
    """Find the call at ``addr`` in *func*'s LLIL-SSA and recover ``reg``'s
    reaching definition (:func:`reaching_reg_def`), bridged to the MLIL SSA
    variable(s) that hold it -- the seeds a backward walk / trace can slice.
    Returns ``[(mlil_ssa_var, mlil_instr), ...]``; ``[]`` when not recoverable
    (e.g. no LLIL, as in the unit fakes)."""
    lssa = getattr(getattr(func, "llil", None), "ssa_form", None)
    if lssa is None:
        return []
    seeds: list[tuple[Any, Any]] = []
    try:
        for block in lssa:
            for ins in block:
                if int(getattr(ins, "address", -1)) != int(addr):
                    continue
                if getattr(ins, "operation", None) != bn.LowLevelILOperation.LLIL_CALL_SSA:
                    continue
                definition = reaching_reg_def(ins, reg, bn)
                if definition is None:
                    continue
                mapped = getattr(definition, "mlil", None)
                mssa = getattr(mapped, "ssa_form", None) if mapped is not None else None
                if mssa is None:
                    continue
                for w in (getattr(mssa, "vars_written", None) or []):
                    if is_ssa_var(w):
                        seeds.append((w, mssa))
    except Exception:
        return seeds
    return seeds


# --------------------------------------------------------------------------

def op_name(item: Any) -> str:
    operation = getattr(item, "operation", None)
    name = getattr(operation, "name", None)
    return str(name) if name else str(operation)


def is_ssa_var(v: Any) -> bool:
    return hasattr(v, "version") and hasattr(v, "var")


# printf-family conversion specifier: optional flags, width, precision, length
# modifier, then a conversion char. `%%` is a literal (consumes nothing); a `*`
# width or precision each consume one extra int arg.
_FORMAT_SPEC_RE = re.compile(
    r"%([-+ 0#]*)(\*|[0-9]+)?(?:\.(\*|[0-9]+))?(?:hh|h|ll|l|L|q|j|z|t)?([diouxXeEfFgGaAcspn%])")


def _count_format_args(fmt: str) -> int | None:
    """Number of varargs a printf-style *constant* format string consumes, so a
    tainted vararg beyond it can be recognized as a provably-dead flow (#45).

    Returns None when the consumed count can't be determined reliably -- a POSIX
    positional ``%n$`` specifier reorders/reuses args, so a positional format is
    not the linear consumption this regex pass assumes (#69). The caller stays
    conservative (every vararg live) on None, avoiding a false-negative sink."""
    if re.search(r"%\d+\$", fmt):
        return None
    n = 0
    for m in _FORMAT_SPEC_RE.finditer(fmt):
        if m.group(4) == "%":
            continue
        if m.group(2) == "*":
            n += 1
        if m.group(3) == "*":
            n += 1
        n += 1
    return n


def var_key(v: Any) -> tuple[str, Any]:
    """Stable identity for a Variable or the base of an SSAVariable."""
    base = getattr(v, "var", v)
    ident = getattr(base, "identifier", None)
    if ident is not None:
        try:
            return ("id", int(ident))
        except Exception:
            pass
    return ("name", str(getattr(base, "name", base)))


def var_label(v: Any) -> str:
    base = getattr(v, "var", v)
    name = str(getattr(base, "name", base))
    version = getattr(v, "version", None)
    return f"{name}#{version}" if version is not None else name


def ssa_reads(ins: Any) -> list[Any]:
    """SSAVariables an instruction reads by value (excludes AddressOf targets,
    which appear in vars_read as plain Variables)."""
    return [v for v in (getattr(ins, "vars_read", None) or []) if is_ssa_var(v)]


def ssa_writes(ins: Any) -> list[Any]:
    return [v for v in (getattr(ins, "vars_written", None) or []) if is_ssa_var(v)]


def expr_reads(expr: Any) -> list[Any]:
    return [v for v in (getattr(expr, "vars_read", None) or []) if is_ssa_var(v)]


def function_at(bv: Any, addr: int | None) -> Any | None:
    """``get_function_at`` with ARM/Thumb low-bit normalization.

    A code pointer on ARM carries the Thumb tag in its LSB, so an indirect/
    value-set target address can be odd while the function lives at ``addr & ~1``.
    The exact lookup is tried first (raw address preserved for diagnostics), then
    the normalized one (#89). Returns None when neither resolves or *bv* lacks
    ``get_function_at`` (the synthetic test fakes).
    """
    if addr is None or not hasattr(bv, "get_function_at"):
        return None
    try:
        fn = bv.get_function_at(addr)
    except Exception:
        fn = None
    if fn is None and (addr & 1):
        try:
            fn = bv.get_function_at(addr & ~1)
        except Exception:
            fn = None
    return fn


def const_target(expr: Any) -> int | None:
    """Constant call destination (direct call) or None (indirect).

    Accepts MLIL_CONST_PTR *and* MLIL_EXTERN_PTR. On a statically-linked object
    (e.g. a kernel .ko) a direct `bl` to an external helper (strlen / sscanf /
    memcpy / copy_from_user) renders as an EXTERN_PTR whose `.constant` IS the
    stub address that `get_symbol_at` resolves to the real name -- so without
    this the callee name (and its sink/source model) is never recovered and the
    call is misreported as an unresolved indirect call, all-clearing every .ko
    sink reached through a stub. (T2; MLIL_IMPORT stays out: its `.constant` is
    a GOT slot, resolved by name via resolve_call_target/extract_dest_address.)
    """
    if expr is None:
        return None
    name = op_name(expr)
    if "CONST" not in name and "EXTERN_PTR" not in name:
        return None
    c = getattr(expr, "constant", None)
    if c is None:
        c = getattr(expr, "value", None)
        c = getattr(c, "value", c)
    try:
        return int(c)
    except Exception:
        return None

def _mlil_ssa(fn: Any) -> Any:
    """MLIL SSA form of a function, tolerating the ``mlil`` vs ``medium_level_il``
    attribute alias (real BN exposes both; the test fakes expose only ``mlil``)."""
    mlil = getattr(fn, "mlil", None) or getattr(fn, "medium_level_il", None)
    if mlil is None:
        return None
    return getattr(mlil, "ssa_form", None)


def _ssa_instructions(ssaf: Any) -> list[Any]:
    """All SSA instructions, preferring ``.instructions`` and falling back to
    flattening ``.basic_blocks`` (both are valid on real BN)."""
    try:
        return list(ssaf.instructions)
    except Exception:
        out: list[Any] = []
        for block in (getattr(ssaf, "basic_blocks", None) or []):
            try:
                out.extend(list(block))
            except Exception:
                continue
        return out


def _symbols_by_name(bv: Any, name: str) -> list[Any]:
    fn = getattr(bv, "get_symbols_by_name", None)
    if fn is None:
        return []
    try:
        return list(fn(name) or [])
    except Exception:
        return []


def _symbol_by_raw_name(bv: Any, name: str) -> Any:
    fn = getattr(bv, "get_symbol_by_raw_name", None)
    if fn is None:
        return None
    try:
        return fn(name)
    except Exception:
        return None


def extract_dest_address(bv: Any, dest: Any) -> int | None:
    """Numeric address of a call/tailcall destination expression, or None.

    Handles a raw int, MLIL_CONST_PTR (``.constant``), and MLIL_IMPORT. For
    imports the symbol name is resolved *before* ``.constant`` because
    ``.constant`` is the GOT slot, not the function entry point.
    """
    try:
        return int(dest)
    except (ValueError, TypeError):
        pass
    name = getattr(dest, "name", None) or str(dest)
    if name:
        for sym in _symbols_by_name(bv, name):
            fn = bv.get_function_at(sym.address)
            if fn is not None:
                return int(fn.start)
        sym = _symbol_by_raw_name(bv, name)
        if sym is not None:
            fn = bv.get_function_at(sym.address)
            if fn is not None:
                return int(fn.start)
    addr = getattr(dest, "constant", None)
    if addr is not None:
        return int(addr)
    return None


def targets_from_pvs(pvs: Any) -> list[int]:
    """Extract concrete call-target addresses from a PossibleValueSet.

    Handles constants, in-set values, and lookup tables (function-pointer
    tables expose ``.mapping`` {idx: addr} / ``.table``, not ``.values``).
    """
    if pvs is None:
        return []
    tname = str(getattr(getattr(pvs, "type", None), "name", "") or "")
    out: list[int] = []

    def _add(v):
        try:
            out.append(int(v))
        except Exception:
            pass

    if tname in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
        _add(getattr(pvs, "value", None))
    elif tname == "InSetOfValues":
        for v in (getattr(pvs, "values", None) or []):
            _add(v)
    elif tname == "LookupTableValue":
        mapping = getattr(pvs, "mapping", None)
        if isinstance(mapping, dict):
            for v in mapping.values():
                _add(v)
        else:
            for entry in (getattr(pvs, "table", None) or []):
                _add(getattr(entry, "to", None))
    return sorted({a for a in out if a})


def follow_thunk(bv: Any, fn: Any) -> Any | None:
    """If *fn* is a single-instruction tailcall thunk, return its real target.

    Follows thunk chains (PLT stubs, GCC veneers) iteratively, guarding against
    cycles of *any* length via a visited-set of function starts. The old
    recursive form only rejected a direct A->A self-loop, so a multi-step
    tailcall cycle in the binary (A->B->A, or longer) recursed without bound and
    raised ``RecursionError``. Returns the deepest resolved target, or None when
    *fn* is not a thunk.
    """
    seen: set[int] = {int(getattr(fn, "start", -1))}
    cur = fn
    result: Any | None = None
    while True:
        ssa = _mlil_ssa(cur)
        if ssa is None:
            break
        instructions = _ssa_instructions(ssa)
        if len(instructions) != 1:
            break
        insn = instructions[0]
        if "TAILCALL" not in op_name(insn):
            break
        dest = getattr(insn, "dest", None)
        if dest is None:
            break
        addr = extract_dest_address(bv, dest)
        if addr is None:
            break
        target = bv.get_function_at(addr)
        if target is None:
            target = bv.get_function_at(addr & ~1)
        if target is None or int(getattr(target, "start", -2)) in seen:
            break
        seen.add(int(getattr(target, "start", -1)))
        result = target
        cur = target
    return result


@dataclass
class ResolvedTarget:
    """Result of resolving a call instruction's callee."""
    address: int | None         # entry of the resolved function (post-thunk if followed)
    function: Any | None        # bv function object, or None
    via: str | None = None      # "direct" | "import" | "thunk" | None
    thunk_chain: list[int] = field(default_factory=list)  # addresses traversed while following thunks


def resolve_call_target(bv: Any, call_insn: Any, *, follow_thunks: bool = False) -> ResolvedTarget:
    """Resolve a call instruction's callee to a function.

    Mirrors the trace resolver: direct numeric/``.constant`` dest first, then
    import-name resolution (the primary path for PLT stubs, where ``.constant``
    is a GOT slot), then optional thunk following. Value-sets and agent-supplied
    resolve-maps are deliberately *not* consulted here — those are forward-taint
    concerns (see ``targets_from_pvs`` and the engine's resolve_map handling).
    """
    dest = getattr(call_insn, "dest", None)
    if dest is None:
        return ResolvedTarget(None, None)

    fn = None
    via: str | None = None
    try:
        addr = int(dest)
    except (ValueError, TypeError):
        addr = getattr(dest, "constant", None)
    if addr is not None and addr != 0:
        fn = bv.get_function_at(int(addr))
        if fn is None:
            fn = bv.get_function_at(int(addr) & ~1)
        if fn is not None:
            via = "direct"

    if fn is None:
        name = getattr(dest, "name", None)
        if name:
            for sym in _symbols_by_name(bv, name):
                fn = bv.get_function_at(sym.address)
                if fn is not None:
                    via = "import"
                    break
            if fn is None:
                sym = _symbol_by_raw_name(bv, name)
                if sym is not None:
                    fn = bv.get_function_at(sym.address)
                    if fn is not None:
                        via = "import"

    if fn is None:
        return ResolvedTarget(None, None)

    thunk_chain: list[int] = []
    if follow_thunks:
        resolved = follow_thunk(bv, fn)
        if resolved is not None and resolved is not fn:
            thunk_chain = [int(getattr(fn, "start", 0))]
            fn = resolved
            via = "thunk"

    return ResolvedTarget(int(getattr(fn, "start", 0)), fn, via, thunk_chain)


# --------------------------------------------------------------------------
