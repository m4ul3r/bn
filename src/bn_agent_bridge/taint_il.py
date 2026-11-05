"""MLIL-SSA IL helpers + call/thunk resolution.

Split out of ``taint_engine`` (pure structural move, #562). Import-free of
``binaryninja``: all BN access is via duck-typed attributes on passed objects.
The engine re-exports these names for back-compat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import il_format


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


def _call_like_ops(bn: Any) -> set:
    Op = bn.LowLevelILOperation
    return {getattr(Op, n, None) for n in (
        "LLIL_CALL_SSA", "LLIL_SYSCALL_SSA", "LLIL_INTRINSIC_SSA",
        "LLIL_TAILCALL_SSA")} - {None}


# A backwards CFG closure visits at most this many blocks. The key below makes
# the walk terminate on its own; the cap is the belt to that braces, so a view
# whose blocks answer no stable key at all still cannot spin.
_CFG_CLOSURE_BLOCKS = 4096


def _block_key(block: Any) -> tuple:
    """Stable identity for a basic block ACROSS the wrappers BN hands out.

    Binary Ninja materialises a fresh Python object for the same native basic
    block on every access, so `id()` identifies the wrapper, not the block: a
    closure that dedupes on it revisits one loop header forever (review F6).
    The native identity is the owning function plus the block's own index /
    instruction range; `id()` is the last resort, for the synthetic test fakes
    that have neither.
    """
    owner = getattr(block, "function", None)
    fn = getattr(owner, "source_function", None) or owner
    home = getattr(fn, "start", None)
    home = int(home) if isinstance(home, int) else None
    index = getattr(block, "index", None)
    if isinstance(index, int):
        return ("index", home, index)
    start, end = getattr(block, "start", None), getattr(block, "end", None)
    if isinstance(start, int) and isinstance(end, int):
        return ("range", home, start, end)
    return ("id", id(block))


def _reaching_blocks(call_bb: Any) -> tuple[list, bool, bool]:
    """``(blocks, looped, complete)`` -- every block that can reach *call_bb*,
    whether *call_bb* can reach ITSELF (it lies on a cycle, so its own later
    instructions may also have run first), and whether the closure FINISHED.

    ``complete`` is False when the block budget stopped the walk with work
    still pending. A caller asserting what ran before this call must refuse on
    that: an unexamined predecessor is exactly where the clobbering call would
    be, so a capped scan proves nothing (review F10).
    """
    call_key = _block_key(call_bb)
    out: list = []
    seen: set = set()
    frontier = [getattr(e, "source", None)
                for e in (getattr(call_bb, "incoming_edges", None) or [])]
    looped = False
    while frontier:
        if len(seen) >= _CFG_CLOSURE_BLOCKS:
            return out, looped, False
        block = frontier.pop()
        if block is None:
            continue
        key = _block_key(block)
        if key == call_key:
            looped = True
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
        frontier += [getattr(e, "source", None)
                     for e in (getattr(block, "incoming_edges", None) or [])]
    return out, looped, True


def _entry_value_state(call_ins: Any, bn: Any) -> str:
    """``"entry"`` / ``"clobbered"`` / ``"budget"`` for the value an untouched
    register holds where *call_ins* runs.

    Used only after :func:`reaching_reg_def` has established that the register
    has no definition at all reaching the call, which means it still holds the
    value the function was entered with -- *unless* something ran in between
    that the IL does not name as a definition. The one such thing is another
    call: a callee may clobber any caller-saved register, and no definition in
    this function records it.

    So ``"clobbered"`` is answered whenever any call/syscall/intrinsic can
    execute before this one. "Can execute before" is taken from the CFG, not
    from instruction order: every block that REACHES the call's block may run
    first, and a block on a cycle through the call reaches itself, so a loop
    body's later call is caught too. Nothing is inferred from a calling
    convention's clobber list.

    ``"budget"`` is the third state (review F10): the predecessor closure ran
    out of its block budget with work pending. What it did not look at is
    exactly where a clobbering call would be, so the scan establishes nothing
    either way -- and the caller is told that, rather than being handed a
    positive it could not earn.
    """
    call_ops = _call_like_ops(bn)
    call_bb = getattr(call_ins, "il_basic_block", None)
    if call_bb is None:
        return "clobbered"
    call_idx = int(getattr(call_ins, "instr_index", -1))
    earlier, looped, complete = _reaching_blocks(call_bb)
    for block in earlier:
        for ins in block:
            if getattr(ins, "operation", None) in call_ops:
                return "clobbered"
    if not complete:
        return "budget"
    for ins in call_bb:
        if getattr(ins, "operation", None) not in call_ops:
            continue
        idx = int(getattr(ins, "instr_index", -1))
        if idx == call_idx:
            continue
        if idx < call_idx or looped:
            return "clobbered"
    return "entry"


def _entry_value_survives(call_ins: Any, bn: Any) -> bool:
    """Whether the register provably still carries the function's entry value.

    The boolean face of :func:`_entry_value_state`: an exhausted scan is not a
    survival, so it answers False for that too.
    """
    return _entry_value_state(call_ins, bn) == "entry"


def argument_register_provenance(func: Any, addr: int, reg: str, bn: Any, *,
                                 calls_at: dict[int, Any] | None = None,
                                 ) -> tuple[str, list[tuple[Any, Any]]]:
    """What the ABI argument register *reg* carries into the call at *addr*.

    ``("defined", seeds)`` -- a dominating LLIL-SSA definition reaches the
    call; ``seeds`` are its ``(mlil_ssa_var, mlil_instr)`` bridges, the values
    a slice can follow. ``("entry", [])`` -- nothing defines the register and
    nothing that can run first may have clobbered it, so it still holds the
    value the function was ENTERED with (the ``rdi#0``-unchanged shape an
    optimised build leaves when the callee has no prototype to recover
    arguments from). ``("budget", [])`` -- the predecessor closure ran out of
    its block budget, so whether anything clobbered the register was never
    established (review F10). ``("unproven", [])`` -- there is no LLIL to read,
    the call is not in it, or a call may have clobbered the register: nothing
    is claimed.

    ``calls_at`` is :func:`llil_call_sites` for *func*, when the caller keeps
    one: finding the call is then a lookup instead of a scan of the whole
    function's LLIL, which a walk asking about many calls in one body paid
    once per question.
    """
    if calls_at is None:
        calls_at = llil_call_sites(func, bn)
    if calls_at is None:
        return ("unproven", [])
    # A TAIL call is a call: an optimised `return f(p)` leaves the
    # arguments exactly where they already are, which is precisely
    # the shape this answers for. Matching only `LLIL_CALL_SSA`
    # left every tail call unrecovered (review F4).
    ins = calls_at.get(int(addr))
    if ins is None:
        return ("unproven", [])
    try:
        definition = reaching_reg_def(ins, reg, bn)
        if definition is None:
            state = _entry_value_state(ins, bn)
            return (state if state in ("entry", "budget") else "unproven", [])
        if getattr(definition, "operation", None) in _call_like_ops(bn):
            # The reaching "definition" is another CALL that lists this
            # register among its outputs -- i.e. it CLOBBERED it. Its
            # MLIL mapping writes the call's RESULT variable, which is
            # the return register's value and says nothing about an
            # ARGUMENT register. Bridging to it would hand the walk the
            # wrong value entirely; a clobbered register is unproven.
            return ("unproven", [])
        defs = [definition]
        if _mlil_seeds(definition) == [] and getattr(definition, "operation", None) \
                == getattr(bn.LowLevelILOperation, "LLIL_REG_PHI", object()):
            # A phi of the register's definitions on each incoming path --
            # armv7 conditional execution (`itet hi; adds r2, ..; mov r2, ..`)
            # leaves one -- has no MLIL instruction of its own: the value
            # the call receives is one of its arms', so each arm bridges.
            defs = _reg_phi_arms(definition, bn) or []
        seeds: list[tuple[Any, Any]] = []
        for d in defs:
            arm = _mlil_seeds(d)
            if not arm:
                # One arm that bridges to nothing (an entry value, a call's
                # clobber) leaves the argument's other values unknown: claim
                # none rather than a subset.
                return ("defined", [])
            seeds.extend(arm)
        return ("defined", seeds)
    except Exception:
        return ("unproven", [])


def _mlil_seeds(definition: Any) -> list[tuple[Any, Any]]:
    """``[(mlil_ssa_var, mlil_instr), ...]`` an LLIL-SSA register definition
    maps to: the SSA variables its MLIL instruction writes."""
    mapped = getattr(definition, "mlil", None)
    mssa = getattr(mapped, "ssa_form", None) if mapped is not None else None
    if mssa is None:
        return []
    return [(w, mssa) for w in (getattr(mssa, "vars_written", None) or []) if is_ssa_var(w)]


def _reg_phi_arms(phi: Any, bn: Any, depth: int = 0, seen: set | None = None) -> list[Any] | None:
    """The non-phi LLIL-SSA definitions an ``LLIL_REG_PHI`` merges, nested
    phis flattened; None when an arm has no definition (the register's entry
    value) or is a call that clobbered it -- the merged value is then not
    only what the definitions wrote."""
    seen = set() if seen is None else seen
    lssa = getattr(phi, "function", None)
    if lssa is None or depth > 8:
        return None
    Op = bn.LowLevelILOperation
    out: list[Any] = []
    for src in (getattr(phi, "src", None) or []):
        key = (str(getattr(src, "reg", "")), getattr(src, "version", None))
        if key in seen:
            continue            # a loop back edge brings nothing new
        seen.add(key)
        d = lssa.get_ssa_reg_definition(src)
        if d is None or getattr(d, "operation", None) in _call_like_ops(bn):
            return None
        if getattr(d, "operation", None) == Op.LLIL_REG_PHI:
            nested = _reg_phi_arms(d, bn, depth + 1, seen)
            if nested is None:
                return None
            out.extend(nested)
        else:
            out.append(d)
    return out


def llil_call_sites(func: Any, bn: Any) -> dict[int, Any] | None:
    """``{address: the first call-like LLIL-SSA instruction there}`` for
    *func*, from one scan; None when it has no readable LLIL-SSA."""
    lssa = getattr(getattr(func, "llil", None), "ssa_form", None)
    if lssa is None:
        return None
    out: dict[int, Any] = {}
    try:
        site_ops = _call_like_ops(bn)
        for block in lssa:
            for ins in block:
                if getattr(ins, "operation", None) in site_ops:
                    out.setdefault(int(getattr(ins, "address", -1)), ins)
    except Exception:
        return None
    return out


def reaching_arg_seed_vars(func: Any, addr: int, reg: str,
                           bn: Any) -> list[tuple[Any, Any]]:
    """Find the call at ``addr`` in *func*'s LLIL-SSA and recover ``reg``'s
    reaching definition (:func:`reaching_reg_def`), bridged to the MLIL SSA
    variable(s) that hold it -- the seeds a backward walk / trace can slice.
    Returns ``[(mlil_ssa_var, mlil_instr), ...]``; ``[]`` when not recoverable
    (e.g. no LLIL, as in the unit fakes). A register carrying the function's
    own ENTRY value has no definition to slice, so it is not one of these --
    :func:`argument_register_provenance` is the caller that distinguishes it."""
    return argument_register_provenance(func, addr, reg, bn)[1]


# --------------------------------------------------------------------------
# small IL helpers (defensive getattr style, matching bridge.py)
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


# --------------------------------------------------------------------------
# unified call-target / thunk resolver
# --------------------------------------------------------------------------
# Canonical home (issue #7) for "what does this call target, through thunks?".
# Faithful ports of the trace resolver that lived in bridge.py; the bridge now
# delegates here. Kept import-free of binaryninja: all BN access is via
# duck-typed methods on the passed objects, and symbol-table lookups are guarded
# so the resolver degrades gracefully against the synthetic fakes the tests use.


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


def follow_thunk(bv: Any, fn: Any, *, ssa_of: Any = _mlil_ssa) -> Any | None:
    """If *fn* is a single-instruction tailcall thunk, return its real target.

    Follows thunk chains (PLT stubs, GCC veneers) iteratively, guarding against
    cycles of *any* length via a visited-set of function starts. The old
    recursive form only rejected a direct A->A self-loop, so a multi-step
    tailcall cycle in the binary (A->B->A, or longer) recursed without bound and
    raised ``RecursionError``. Returns the deepest resolved target, or None when
    *fn* is not a thunk. *ssa_of* reads a body's MLIL SSA form; the taint
    engine passes its per-request cache, so a hop it already lifted is not
    lifted again.
    """
    seen: set[int] = {int(getattr(fn, "start", -1))}
    cur = fn
    result: Any | None = None
    while True:
        ssa = ssa_of(cur)
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



# An alignment clears at most this many low bits: `p & ~0xfff` aligns a pointer
# to a page, a wider mask is not an alignment an inline expansion emits.
_ALIGN_MASK_BITS = 12

# How many nested operations/definitions the expansion-length walk follows; an
# expansion's count is a handful of instructions from its pointer.
_RESIDUE_DEPTH = 24


def inline_expansion_length(ssaf: Any, addr: int, length: Any,
                            ssa_def: Any) -> dict[str, Any] | None:
    """``{"intrinsic", "length_bound"}`` when the call at *addr* is an inline
    string expansion (:func:`il_format.inline_expansion_intrinsics`) whose
    *length* operand (MLIL SSA) is a constant total split by pointer alignment
    (:func:`expansion_length_bound`); else None. A real call is never one,
    whatever it is named. *ssa_def* is ``(ssaf, var) -> definition``."""
    intrinsics = il_format.inline_expansion_intrinsics(getattr(ssaf, "source_function", None), addr)
    if not intrinsics:
        return None
    bound = expansion_length_bound(ssaf, length, ssa_def)
    if bound is None:
        return None
    return {"intrinsic": il_format.intrinsic_name(intrinsics), "length_bound": hex(bound)}


def expansion_length_bound(ssaf: Any, expr: Any, ssa_def: Any) -> int | None:
    """The largest count an inline string expansion's length operand can take
    when it is built ONLY from constants and pointer-alignment residues; None
    when any other value contributes.

    gcc expands a constant-size ``memset``/``memcpy`` into a head store, a tail
    store and a ``rep stos``/``rep movs`` over the aligned middle, whose count is
    the constant total corrected by how far the destination sits from its
    alignment: ``(p - ((p + 8) & ~7) + TOTAL) >> 3``. The pointer cancels, so the
    count is the constant, whatever the pointer holds -- taint on the POINTER
    says nothing about it. The same expansion of a variable size adds the size
    (``n + (p - align(p + 8)) >> 3``), and that operand stays unbounded here, so
    a tainted total still reports. Recognised from the arithmetic itself: an
    aligned value minus the pointer it was aligned from, a low-bit mask
    (``n & 7``), shifts and constant offsets."""
    val = _residue_interval(ssaf, expr, ssa_def, {}, 0)
    if val is None:
        return None
    base, lo, hi = val
    return hi if base is None and lo >= 0 else None


def _residue_interval(ssaf: Any, expr: Any, ssa_def: Any, memo: dict,
                      depth: int) -> tuple[Any, int, int] | None:
    """``(base, lo, hi)``: *expr* is ``base + k`` with ``lo <= k <= hi``, *base*
    an opaque value's identity (None for a pure constant). None when *expr* is
    not built from one base, constants, alignment masks and shifts (see
    :func:`expansion_length_bound`). *memo* holds each variable's answer for
    one evaluation."""
    if expr is None or depth > _RESIDUE_DEPTH:
        return None
    op = op_name(expr)
    if "CONST" in op:
        c = getattr(expr, "constant", None)
        return (None, int(c), int(c)) if isinstance(c, int) else None
    if op in ("MLIL_ZX", "MLIL_SX", "MLIL_LOW_PART"):
        return _residue_interval(ssaf, getattr(expr, "src", None), ssa_def, memo, depth + 1)
    if op in ("MLIL_VAR_SSA", "MLIL_VAR_SSA_FIELD"):
        if op == "MLIL_VAR_SSA_FIELD" and int(getattr(expr, "offset", 0) or 0) != 0:
            return None
        return _residue_of_var(ssaf, getattr(expr, "src", None), ssa_def, memo, depth + 1)
    left, right = getattr(expr, "left", None), getattr(expr, "right", None)
    if op == "MLIL_AND":
        width = 8 * int(getattr(expr, "size", 0) or 8)
        for mask_side, other in ((right, left), (left, right)):
            m = getattr(mask_side, "constant", None) if "CONST" in op_name(mask_side) else None
            if not isinstance(m, int):
                continue
            # BN hands an alignment mask over signed (`& -8` for `& ~7`).
            m &= (1 << width) - 1
            limit = 1 << _ALIGN_MASK_BITS
            if 0 < m < limit and m & (m + 1) == 0:
                return (None, 0, m)  # a low-bit residue: `n & 7`, `-p & 7`
            low = ~m & ((1 << width) - 1)
            if 0 < low < limit and low & (low + 1) == 0:
                val = _residue_interval(ssaf, other, ssa_def, memo, depth + 1)
                # aligning down moves the value back by at most `low`
                return None if val is None else (val[0], val[1] - low, val[2])
        return None
    if op in ("MLIL_ADD", "MLIL_SUB"):
        a = _residue_interval(ssaf, left, ssa_def, memo, depth + 1)
        b = _residue_interval(ssaf, right, ssa_def, memo, depth + 1)
        if a is None or b is None:
            return None
        if op == "MLIL_ADD":
            if a[0] is not None and b[0] is not None:
                return None
            return (a[0] if a[0] is not None else b[0], a[1] + b[1], a[2] + b[2])
        if a[0] == b[0]:
            base = None
        elif b[0] is None:
            base = a[0]
        else:
            return None
        return (base, a[1] - b[2], a[2] - b[1])
    if op in ("MLIL_LSR", "MLIL_ASR", "MLIL_LSL"):
        k = getattr(right, "constant", None) if "CONST" in op_name(right) else None
        val = _residue_interval(ssaf, left, ssa_def, memo, depth + 1)
        if not isinstance(k, int) or val is None or val[0] is not None or val[1] < 0:
            return None
        if op == "MLIL_LSL":
            return (None, val[1] << k, val[2] << k)
        return (None, val[1] >> k, val[2] >> k)
    return None


def _residue_of_var(ssaf: Any, var: Any, ssa_def: Any, memo: dict,
                    depth: int) -> tuple[Any, int, int] | None:
    """:func:`_residue_interval` of an SSA variable: its definition when that
    is arithmetic the interval follows, every arm of a phi whose arms share one
    base, else the variable itself as an opaque base (a parameter, a load, a
    call result, a phi of unrelated values, a loop-carried value, a definition
    past the depth bound). Memoised per evaluation, so every read of a variable
    gets one answer -- two reads of one opaque value cancel, reads of different
    ones never do -- and a variable on a loop is opaque, never revisited."""
    if not is_ssa_var(var):
        return None
    key = (var_key(var), getattr(var, "version", None))
    if key in memo:
        return memo[key]
    opaque = (key, 0, 0)
    memo[key] = opaque  # a cycle back to this variable sees it opaque
    result = opaque
    defn = ssa_def(ssaf, var) if depth <= _RESIDUE_DEPTH else None
    dop = op_name(defn) if defn is not None else ""
    if dop == "MLIL_VAR_PHI":
        arms = [_residue_of_var(ssaf, v, ssa_def, memo, depth + 1)
                for v in (getattr(defn, "src", None) or [])]
        if arms and all(a is not None for a in arms) and len({a[0] for a in arms}) == 1:
            result = (arms[0][0], min(a[1] for a in arms), max(a[2] for a in arms))
    elif dop == "MLIL_SET_VAR_SSA":
        val = _residue_interval(ssaf, getattr(defn, "src", None), ssa_def, memo, depth + 1)
        if val is not None:
            result = val
    memo[key] = result
    return result
