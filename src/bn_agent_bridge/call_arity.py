"""Call-arity findings shared by `decompile` and `callsites` (#955, #1000, #1038).

`read_decompile` renders them as the `arity` warning and inline notes;
`read_listing` attaches them to callsite rows (#1038 item 6). Both import this
module; it imports neither (#1045 review: the two read modules must not
depend on each other), only ``il_format`` and binaryninja.
"""
from __future__ import annotations

import re
from typing import Any

try:
    import binaryninja as bn  # bn.demangle_gnu3 for mangled callee signatures
except ModuleNotFoundError:  # importable without the Binary Ninja runtime (tests, tooling)
    bn = None  # type: ignore[assignment]

from . import il_format


# Architectures whose ABI returns in a register WIDER than a 32-bit write,
# where that narrower write ZERO-EXTENDS the full register -- so BN infers a
# 64-bit return for a function that only ever writes 32 bits.
#
# Confirmed per arch rather than inherited from x86 (#675 item 5): on x86_64
# the `eax` write zero-extends `rax`, and AArch64 has the SAME relation
# (`w0` zero-extends `x0`) -- measured, a cross-built AArch64 binary shows
# the identical `uint64_t(int32_t, int32_t)` shape. ARM32/thumb2 does NOT
# belong: its registers are natively 32-bit, there is no widening, and the
# same measurement finds ZERO over-wide functions there.
#
# Emitting nothing outside this set is the point. A note that appeared on
# ARM firmware while explaining an x86/AArch64 register relation would be
# worse than no note: it would be a confident claim about a mechanism that
# target does not have.
_ZERO_EXTENDING_RETURN_ARCHS = frozenset({"x86_64", "aarch64"})

# Symbol kinds whose body is not in this binary: an import's PLT stub, its GOT
# slot, or an extern. A call to one of these cannot be IPRA-optimized (#955), so
# an argument register written right before it is an argument.
_IMPORT_SYMBOL_KINDS = {
    "ImportedFunctionSymbol", "ImportAddressSymbol", "ExternalSymbol",
    "LibraryFunctionSymbol",
}


def _full_reg(arch, name: str) -> str:
    """The full-width register *name* is part of (`w2` -> `x2`, `esi` -> `rsi`)."""
    try:
        return str(arch.regs[name].full_width_reg)
    except Exception:  # noqa: BLE001 - an unknown register stands for itself
        return str(name)


def _llil_regs_read(arch, expr, acc: set[str]) -> None:
    if expr is None or not hasattr(expr, "operation"):
        return
    if il_format._il_op_name(expr) in ("LLIL_REG", "LLIL_REG_SSA", "LLIL_REG_SSA_PARTIAL"):
        src = getattr(expr, "src", None)
        name = getattr(src, "name", None)
        if name:
            acc.add(_full_reg(arch, str(name)))
    for operand in getattr(expr, "operands", None) or []:
        for item in operand if isinstance(operand, list) else [operand]:
            _llil_regs_read(arch, item, acc)


def _argument_setup_count(arch, window: list, arg_regs: list[str]) -> tuple[int, dict[str, Any]]:
    """How many leading argument registers *window* writes for the call after it.

    *window* is the LLIL between the previous call (or the block start) and the
    call, in order. LLIL blocks do NOT end at calls, so that bound is the
    caller's job, and it bounds everything here: nothing before an earlier call
    can set up this one, nor explain one of its writes (#985 review).

    A write to an argument register counts only when nothing reads that register
    between the write and the call: such a value has no consumer but the callee,
    since every argument register is caller-saved. Two writes are never counted:
    one that is read before the call (a temporary), and GCC's stack-protector
    scrub -- `ldr xN, [guard]; str xN, [sp, ..]; mov xN, #0` -- which zeroes a
    register it just stored, and is where the issue's invented `__stack_chk_guard`
    argument came from (#955). Returns (count, {written register: its LLIL write}),
    in argument order: the count is the highest written argument register's
    position + 1, because the ABI passes every lower register too.
    """
    read_after: set[str] = set()
    written: dict[str, Any] = {}
    stored_before: list[set[str]] = []
    stored: set[str] = set()
    for ins in window:
        stored_before.append(set(stored))
        if il_format._il_op_name(ins) == "LLIL_STORE":
            src = getattr(ins, "src", None)
            if il_format._il_op_name(src) == "LLIL_REG":
                stored.add(_full_reg(arch, str(getattr(src.src, "name", ""))))
    for pos in range(len(window) - 1, -1, -1):
        ins = window[pos]
        if il_format._il_op_name(ins) == "LLIL_SET_REG":
            name = _full_reg(arch, str(getattr(getattr(ins, "dest", None), "name", "")))
            if name in arg_regs and name not in written and name not in read_after:
                scrub = (il_format._llil_constant_value(getattr(ins, "src", None)) == 0
                         and name in stored_before[pos])
                if not scrub:
                    written[name] = ins
            _llil_regs_read(arch, getattr(ins, "src", None), read_after)
        else:
            _llil_regs_read(arch, ins, read_after)
    if not written:
        return 0, {}
    names = sorted(written, key=arg_regs.index)
    return arg_regs.index(names[-1]) + 1, {name: written[name] for name in names}


_CALL_OPS = ("LLIL_CALL", "LLIL_TAILCALL", "LLIL_CALL_STACK_ADJUST")
# A chain of selects (`ccmp`, then `csel`) lifts to one diamond per select.
_SELECT_DIAMOND_DEPTH = 4


def _block_key(block) -> int | None:
    try:
        return int(block.start)
    except Exception:  # noqa: BLE001 - a block without an index cannot be matched
        return None


def _lifted_select_prefix(block, depth: int = 0) -> tuple[list, set[int]]:
    """The set-up that precedes *block* when *block* is the join of a lifted
    branch-free select, and ``([], set())`` otherwise.

    BN lifts one `csel w2, w2, w3, ls` into a diamond: the block before it
    ends in `if (w2 u<= w3)`, and each arm is a one-instruction block that sets
    w2 and jumps to the join. A call right after the `csel` therefore starts
    its own LLIL block, and the window `_calls_with_setup_windows` hands
    `_argument_setup_count` held nothing -- so `ostream::write(this, buf, n)`,
    whose length is that `csel`, rendered two arguments with no warning
    (#1000 item 2). Returns (the head's tail after its last call + every arm,
    the ids of the arm instructions). Only that exact shape is followed: two
    or more arms, every arm lifted from ONE instruction at the head's `if`
    address and entered only from the head."""
    if depth >= _SELECT_DIAMOND_DEPTH:
        return [], set()
    try:
        arms = [edge.source for edge in list(block.incoming_edges)]
        if len(arms) < 2:
            return [], set()
        arm_bodies = [list(arm) for arm in arms]
        sites = {int(ins.address) for body in arm_bodies for ins in body}
        heads = [[edge.source for edge in list(arm.incoming_edges)] for arm in arms]
        if len(sites) != 1 or any(len(h) != 1 for h in heads):
            return [], set()
        head = heads[0][0]
        if any(_block_key(h[0]) != _block_key(head) for h in heads) or _block_key(head) is None:
            return [], set()
        head_body = list(head)
        if (not head_body or il_format._il_op_name(head_body[-1]) != "LLIL_IF"
                or int(head_body[-1].address) not in sites):
            return [], set()
    except Exception:  # noqa: BLE001 - an unreadable CFG is simply not a diamond
        return [], set()
    tail: list = []
    reset = False
    # The `if` itself stays in: a register only its condition reads (`cmp x6,
    # x7` before the `csel`) is consumed there, not handed to the call.
    for ins in head_body:
        op = il_format._il_op_name(ins)
        if "CALL" in op or op == "LLIL_SYSCALL":
            tail, reset = [], True
        else:
            tail.append(ins)
    prefix, arm_ids = ([], set()) if reset else _lifted_select_prefix(head, depth + 1)
    # Every arm, not one: an arm reads its other operand (`w2 = w3`), and a
    # register only one arm reads must not count as written-and-unread.
    arm = [ins for body in arm_bodies for ins in body
           if il_format._il_op_name(ins) != "LLIL_GOTO"]
    return prefix + tail + arm, arm_ids | {id(ins) for ins in arm}


def _calls_with_setup_windows(llil):
    """Yield ``(call, window, select_ids)`` for each LLIL call, *window* being
    the instructions since the previous call or the start of the call's block.
    A block that is the join of a lifted select continues the window back
    through the select (`_lifted_select_prefix`), whose arm instructions are
    *select_ids*. One linear pass, so the cost is the function's LLIL size,
    not its square."""
    for block in list(getattr(llil, "basic_blocks", None) or []):
        window, select_ids = _lifted_select_prefix(block)
        for ins in block:
            op = il_format._il_op_name(ins)
            if op in _CALL_OPS:
                yield ins, window, select_ids
                window, select_ids = [], set()
            elif "CALL" in op or op == "LLIL_SYSCALL":
                window, select_ids = [], set()
            else:
                window.append(ins)


# Sized deallocation (`operator delete(void*, size_t)` and the array / aligned
# forms). BN's type library declares only the pointer, so every sized delete in
# a C++ binary renders one argument short -- 370 of 383 findings on one stock
# binary (#985 review). The dropped value is the object size the allocator is
# told, not data any analysis follows, so reporting it buried the findings that
# matter.
_SIZED_DELETE_RE = re.compile(r"^_Zd[la]Pv[mj]")


def _demangled_arity(bv, raw_name: str) -> tuple[int, int, bool] | None:
    """(total, explicit, variadic) parameter counts from a mangled name, or None.

    BN's GNU3 demangler includes `this` for every nested name -- including a
    static member or a namespaced free function, which have none -- so *total*
    may overcount by one; *explicit* excludes a `this` parameter and never does.
    """
    demangle = getattr(bn, "demangle_gnu3", None)
    if demangle is None or not raw_name.startswith("_Z"):
        return None
    try:
        demangled_type, _ = demangle(bv.arch, raw_name)
        params = list(getattr(demangled_type, "parameters", None) or [])
    except Exception:  # noqa: BLE001 - an undemanglable name gives no signature
        return None
    if demangled_type is None:
        return None
    has_this = bool(params) and str(getattr(params[0], "name", "")) == "this"
    variadic = bool(getattr(demangled_type, "has_variable_arguments", False))
    return len(params), len(params) - int(has_this), variadic


def _call_arity_findings(bv, func) -> list[dict[str, Any]]:
    """One row per call whose rendered argument list is shorter than the call
    site's (#955), or -- for a mangled callee -- longer than its signature, or
    whose callee's prototype declares no parameters at all while HLIL renders
    some (#1038 item 1).

    HLIL renders exactly the callee prototype's parameter count. When that
    prototype is short -- BN's bundled C++ type libraries declare many library
    methods without `this` and drop trailing parameters -- the extra arguments
    vanish from the output with no trace: `std::filebuf::open(&f)` for a call
    whose disassembly sets up `this`, the path and the open mode.

    Two witnesses, so neither can fire alone where it is known to be wrong:
      * the demangled signature of a mangled callee. It counts `this` for a
        static member too, so a shortfall against it is only reported when the
        explicit parameters alone already exceed the rendered count, or when
        the call site itself writes a register past the rendered arguments;
      * the argument registers the call's block writes and nothing else reads
        (`_argument_setup_count`). Used alone only for a callee with no body
        here -- an import cannot be IPRA-compiled, so a register it is handed
        cannot be a value the caller keeps across the call.
    An excess (more rendered than the signature takes, the invented argument
    the issue saw) is reported against the mangled signature only, and never
    for a variadic one.

    The third shape claims no arity: an import BN's type library has no entry
    for gets `int64_t socket()`, and HLIL then renders whatever argument
    registers BN's call-site analysis picked (`socket(1, 2, 0,
    __stack_chk_guard, 0)`). A prototype with no parameters cannot be where
    rendered arguments came from, so they are BN's guess -- unless BN or the
    analyst gave this call its own type (`get_call_type_adjustment`), or BN
    determined the callee variadic. The row says that and nothing about how
    many arguments the callee really takes (`shape: "no_parameters"`).

    Each row keeps what the inline note needs (#1000 item 2): the argument
    registers past the rendered ones, with the value each holds at the call
    when it is known -- the value the call site wrote, else BN's own dataflow
    value there (#1038 item 2: `adrp x1; add x1, x1, #off` is a constant
    pointer BN knows, and its name is the argument a reader needs). Empty
    when every call agrees; never raises.
    """
    try:
        llil = getattr(func, "llil", None) or getattr(func, "low_level_il", None)
        calls = list(_calls_with_setup_windows(llil)) if llil is not None else []
        cc = getattr(func, "calling_convention", None)
        arg_regs = [_full_reg(bv.arch, str(r)) for r in (getattr(cc, "int_arg_regs", None) or [])]
    except Exception:  # noqa: BLE001 - no IL means nothing to compare
        return []
    rows: list[dict[str, Any]] = []
    for call, window, select_ids in calls:
        try:
            target = il_format._llil_constant_value(getattr(call, "dest", None))
            address = int(getattr(call, "address", 0))
            if target is None:
                continue
            mlil = getattr(call, "mlil", None)
            params = getattr(mlil, "params", None)
            if params is None:
                continue
            rendered = len(list(params))
            symbol = bv.get_symbol_at(target)
            callee = bv.get_function_at(target)
            kind = str(getattr(getattr(symbol, "type", None), "name", ""))
            raw_name = str(getattr(symbol, "raw_name", "") or getattr(callee, "name", "") or "")
            if _SIZED_DELETE_RE.match(raw_name):
                continue
            no_body = kind in _IMPORT_SYMBOL_KINDS or bool(getattr(callee, "is_thunk", False))
            signature = _demangled_arity(bv, raw_name)
            setup, writes = _argument_setup_count(bv.arch, window, arg_regs) if arg_regs else (0, {})
            unprototyped = (signature is None and rendered > 0
                            and _declares_no_parameters(func, callee, address))
        except Exception:  # noqa: BLE001 - one odd call must not hide the rest
            continue
        setup_text = f"the call site writes {', '.join(writes)} just before it"
        shape = "shortfall"
        if signature is not None:
            # The demangled signature is the arity whenever there is one: the
            # register witness only corroborates it, never overrides it.
            total, explicit, variadic = signature
            if rendered > total and not variadic:
                expected = total
                shape = "excess"
                evidence = [f"the demangled signature takes only {total}"]
            elif rendered >= total or not (explicit > rendered or setup > rendered):
                continue
            else:
                expected = total
                evidence = [f"the demangled signature takes {total}"
                            + (" with `this`" if total != explicit else "")]
                if setup > rendered:
                    evidence.append(setup_text)
        elif unprototyped:
            # No expected count: nothing here knows how many the callee takes.
            expected, shape = rendered, "no_parameters"
            evidence = [("its prototype declares no parameters, so the rendered arguments "
                         "are BN's guess from the registers at the call site")]
            if writes:
                evidence.append(setup_text)
        elif no_body and setup > rendered:
            expected, evidence = setup, [setup_text]
        else:
            continue
        values = _window_values(bv, func, window, select_ids)
        missing = [(reg, _missing_value(bv, func, address, reg,
                                        values.get(reg) if reg in writes else None))
                   for reg in arg_regs[rendered:expected]]
        rows.append({
            "address": address, "target": target,
            "label": _callee_label(symbol, callee, target),
            "rendered": rendered, "expected": expected, "shape": shape,
            "evidence": tuple(evidence), "missing": missing,
        })
    return rows


def _declares_no_parameters(func, callee, address: int) -> bool:
    """True when *callee*'s prototype has no parameters, is not one BN
    determined variadic, and the call at *address* has no call-type
    adjustment of its own (which, not the prototype, is then what HLIL
    renders)."""
    if callee is None:
        return False
    params = getattr(getattr(callee, "type", None), "parameters", None)
    if params is None or len(list(params)) != 0:
        return False
    if il_format._variadic_determination(callee) is True:
        return False
    try:
        return func.get_call_type_adjustment(address) is None
    except Exception:  # noqa: BLE001 - an unreadable adjustment is not "none"
        return False


def _missing_value(bv, func, address: int, reg: str, written: str | None) -> str | None:
    """What an unrendered argument register holds at the call: the value the
    call site wrote (*written*), else BN's own dataflow value at the call
    when that is a constant (#1038 item 2). A value BN's dataflow calls a
    constant POINTER carries the name the view gives that address (`0x4008e0
    (on_packet)`); a plain constant is never named, since a length of 0x30
    can equal a symbol's address. None when neither knows."""
    if written is not None:
        return written
    try:
        value = func.get_reg_value_at(address, reg)
        kind = str(getattr(getattr(value, "type", None), "name", ""))
        if kind not in ("ConstantValue", "ConstantPointerValue"):
            return None
        text = hex(int(value.value))
    except Exception:  # noqa: BLE001 - no dataflow: the register stands alone
        return None
    name = il_format._view_name_at(bv, int(value.value)) if kind == "ConstantPointerValue" else None
    return f"{text} ({name})" if name else text


_WRITE_TEXT_MAX = 40
_IL_VAR_ANNOTATION_RE = re.compile(r"\s*\{[^{}]*\}")


def _reg_bits(arch, name: str) -> int | None:
    try:
        size = int(arch.regs[name].size)
    except Exception:  # noqa: BLE001 - an unknown register has no known width
        return None
    return size * 8 if size > 0 else None


def _narrowed(arch, spelled: str, text: str, bits: int | None) -> tuple[str, int | None]:
    """*text*, a value *bits* wide (None: as wide as the full register), as
    read through the register spelling *spelled*, with the width it then
    has: `mov w0, w1` copies only the low 32 bits of a 64-bit `arg2`, which
    is `(uint32_t)arg2`, not `arg2`."""
    narrow = _reg_bits(arch, spelled)
    full = _reg_bits(arch, _full_reg(arch, spelled))
    if narrow is None or full is None or narrow >= full or (bits is not None and bits <= narrow):
        return text, bits
    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return hex(int(text, 16) & ((1 << narrow) - 1)), narrow
    cast = f"(uint{narrow}_t){text}" if re.fullmatch(r"\w+", text) else f"(uint{narrow}_t)({text})"
    return cast, narrow


def _entry_name(bv, func, address: int | None, reg: str) -> tuple[str, int | None] | None:
    """What *reg* holds at *address* by BN's own dataflow, as (text, bits): a
    constant, or the function's parameter (with its width) when the register
    still holds its entry value. None otherwise."""
    if address is None:
        return None
    try:
        value = func.get_reg_value_at(address, reg)
        kind = str(getattr(getattr(value, "type", None), "name", ""))
        if kind in ("ConstantValue", "ConstantPointerValue"):
            return hex(int(value.value)), None
        if kind != "EntryValue":
            return None
        entry = _full_reg(bv.arch, str(getattr(value, "reg", "") or ""))
        for var in list(getattr(func, "parameter_vars", None) or []):
            if _register_name(bv.arch, var) == entry:
                width = getattr(getattr(var, "type", None), "width", None)
                return str(var.name), int(width) * 8 if isinstance(width, int) and width > 0 else None
    except Exception:  # noqa: BLE001 - no dataflow: say nothing rather than guess
        return None
    return None


def _llil_writes(arch, ins) -> list[str]:
    """Every register *ins* writes, full width: a plain set, a split pair
    (`mul`'s rdx:rax), any other `LLIL_SET_REG*` form, and an intrinsic's
    outputs (`mrs x5, tpidr_el0`). A write this misses would let a note name
    a value the register no longer holds."""
    op = il_format._il_op_name(ins)
    if op == "LLIL_INTRINSIC":
        targets = list(getattr(ins, "output", None) or [])
    elif op.startswith("LLIL_SET_REG"):
        targets = [getattr(ins, key, None) for key in ("dest", "hi", "lo")]
    else:
        return []
    names = [str(getattr(t, "name", "")) for t in targets if getattr(t, "name", None)]
    return [_full_reg(arch, name) for name in names]


def _window_values(bv, func, window: list, select_ids: set[int]) -> dict[str, str | None]:
    """The value each register holds after *window*, in words a note can show.

    Taking a write's LLIL source text as its value was wrong whenever a
    register that source reads changed before the call: `mov x2, x0` copies
    `strlen`'s result, then `x0 = &s`, and the note said `x2 = x0`. So the
    window is evaluated forward: a register copy carries the copied value
    along (`w4 = w2; w2 = w3; w3 = w4` leaves x3 holding the old x2), and a
    register read at the window's start is named by BN's dataflow there (a
    parameter or a constant) or left as itself only if nothing in the window
    writes it. Any other source is shown with the same naming applied to each
    register it reads (`w1 + 1` -> `arg2 + 1` when x1 is overwritten later),
    and not at all when one it reads was already rewritten in the window or
    cannot be named. A value read through a narrower register keeps that
    width (`(uint32_t)arg2`), a write narrower than 32 bits (or any partial
    write on an architecture that does not zero-extend) leaves the value
    unknown, and a write this does not model (a split pair, an intrinsic)
    clobbers the register. A lifted select arm is its mnemonic and address
    (`csel @ 0x4015cc`): one arm is not the value. None means unknown, never
    a guess."""
    arch = bv.arch
    written = {reg for ins in window for reg in _llil_writes(arch, ins)}
    start = None
    if window:
        try:
            start = int(window[0].address)
        except Exception:  # noqa: BLE001
            start = None
    # Each value is (text, bits): bits is how wide the value is when narrower
    # than its register, so a later copy through a narrower spelling knows
    # whether it truncates.
    values: dict[str, tuple[str, int | None] | None] = {}
    written_so_far: set[str] = set()
    regs = getattr(arch, "regs", None) or {}
    zero_extends = str(getattr(arch, "name", "")) in _ZERO_EXTENDING_RETURN_ARCHS

    def spelled_bits(spelled: str) -> int | None:
        narrow, full = _reg_bits(arch, spelled), _reg_bits(arch, _full_reg(arch, spelled))
        return narrow if narrow is not None and full is not None and narrow < full else None

    def at_start(reg: str, spelled: str) -> tuple[str, int | None] | None:
        named = _entry_name(bv, func, start, reg)
        if named is not None:
            return _narrowed(arch, spelled, *named)
        return (spelled, spelled_bits(spelled)) if reg not in written else None

    def named_source(src, bits: int | None) -> tuple[str, int | None] | None:
        read: set[str] = set()
        _llil_regs_read(arch, src, read)
        if read & written_so_far:
            return None
        text = _IL_VAR_ANNOTATION_RE.sub("", str(src or "")).strip()
        unknown = False

        def rename(match: re.Match[str]) -> str:
            nonlocal unknown
            token = match.group(0)
            if token not in regs:
                return token
            named = _entry_name(bv, func, start, _full_reg(arch, token))
            if named is not None:
                return _narrowed(arch, token, *named)[0]
            if _full_reg(arch, token) in written:
                unknown = True
            return token

        text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rename, text)
        if unknown or not text or len(text) > _WRITE_TEXT_MAX:
            return None
        return text, bits

    for ins in window:
        if il_format._il_op_name(ins) != "LLIL_SET_REG":
            for reg in _llil_writes(arch, ins):
                values[reg] = None
                written_so_far.add(reg)
            continue
        spelled_dest = str(getattr(getattr(ins, "dest", None), "name", ""))
        dest = _full_reg(arch, spelled_dest)
        src = getattr(ins, "src", None)
        try:
            dest_bits = spelled_bits(spelled_dest)
            if dest_bits is not None and not (dest_bits == 32 and zero_extends):
                values[dest] = None  # the rest of the register keeps an older value
            elif id(ins) in select_ids:
                words = str(bv.get_disassembly(int(ins.address)) or "").split()
                values[dest] = (f"{words[0]} @ {hex(int(ins.address))}", dest_bits) if words else None
            elif il_format._il_op_name(src) == "LLIL_REG":
                name = str(getattr(getattr(src, "src", None), "name", ""))
                full = _full_reg(arch, name)
                if full in values:
                    held = values[full]
                    values[dest] = None if held is None else _narrowed(arch, name, *held)
                else:
                    values[dest] = at_start(full, name)
            else:
                values[dest] = named_source(src, dest_bits)
        except Exception:  # noqa: BLE001 - a value is a nicety; the register stands alone
            values[dest] = None
        written_so_far.add(dest)
    return {reg: (value[0] if value is not None else None) for reg, value in values.items()}


def _arity_text(row: dict[str, Any]) -> str:
    """What one arity finding says about its call, in a few words: the body
    of the inline `/* bn: ... */` note and of a `callsites` row's `arity`."""
    if row.get("shape") == "no_parameters":
        return (f"prototype declares no params; the {row['rendered']} "
                f"arg{'s' if row['rendered'] != 1 else ''} shown are BN's guess")
    if row["rendered"] > row["expected"]:
        return f"renders {row['rendered']} args, signature takes {row['expected']}"
    regs = ", ".join(f"{reg} = {value}" if value else reg for reg, value in row["missing"])
    count = row["expected"] - row["rendered"]
    return f"+{count} arg{'s' if count != 1 else ''} not rendered: {regs}"


def _callee_label(symbol, callee, target: int) -> str:
    name = str(getattr(symbol, "short_name", "") or "") if symbol is not None else ""
    if not name and callee is not None:
        name = il_format._display_name(callee)
    name = name.split("(", 1)[0] if name else ""
    return f"{name} @ {hex(target)}" if name else hex(target)


def _register_name(arch, var) -> str | None:
    """The register *var* lives in, or None for a stack/flag variable."""
    source = getattr(getattr(var, "source_type", None), "name", "")
    if source != "RegisterVariableSourceType":
        return None
    try:
        return _full_reg(arch, str(arch.get_reg_name(var.storage)))
    except Exception:  # noqa: BLE001
        return None
