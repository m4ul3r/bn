"""Argument-register accounting across BN's prototype mapping (P2-02, P2-07).

Which argument registers a call's recovered arguments occupy, whether a
register written before the call still holds THIS call's argument, whether a
callee uses a register's entry value or only saves it, what a declared-variadic
callee's extra registers are, and what a backward slice that ends at an
unmapped entry register is."""
from __future__ import annotations

import types

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te

_REGS = ["r0", "r1", "r2", "r3"]
_REG_INDEX = {r: i for i, r in enumerate(_REGS)}
CALLER, CALLEE, OTHER, CALL = 0x1000, 0x2000, 0x3000, 0x1040


def _reg_var(name, reg, ident):
    """A register variable, as BN describes one: register source, storage."""
    v = FVar(name, ident=ident)
    v.source_type = 1
    v.storage = _REG_INDEX[reg]
    return v


def _stack_var(name, ident, offset):
    v = FVar(name, ident=ident)
    v.source_type = 0
    v.storage = offset
    return v


def _arch():
    return types.SimpleNamespace(
        get_reg_index=lambda r: _REG_INDEX[r],
        get_reg_name=lambda i: _REGS[i],
        regs={r: types.SimpleNamespace(full_width_reg=r) for r in _REGS})


def _cc():
    return types.SimpleNamespace(int_arg_regs=list(_REGS))


def _program(*, callee_formals, call_params, block_before, variadic=False,
             callee_body=None):
    """``caller(p) { <block_before>; callee(<call_params>); }``.

    ``callee_formals`` are the callee's recovered parameters (register or
    stack variables), ``block_before`` the caller's instructions in the call's
    block ahead of the call."""
    p = _reg_var("p", "r0", 40)
    callee = FFunc("consumer", CALLEE, FSSAFunc(callee_body or [
        FInstr(0, CALLEE + 4, "MLIL_RET", "return", reads=[], src=[])]),
        params=list(callee_formals))
    callee.calling_convention = _cc()
    callee.type = types.SimpleNamespace(has_variable_arguments=variadic)
    call = FInstr(len(block_before), CALL, "MLIL_CALL_SSA", "consumer(...)",
                  reads=[r for c in call_params for r in c.vars_read], writes=[],
                  dest=FExpr("MLIL_CONST_PTR", hex(CALLEE), constant=CALLEE),
                  params=list(call_params))
    block = list(block_before) + [call]
    for i, ins in enumerate(block):
        ins.instr_index = i
        ins.il_basic_block = block
    caller = FFunc("caller", CALLER, FSSAFunc(block), params=[p])
    caller.calling_convention = _cc()
    other = FFunc("prior", OTHER, FSSAFunc([FInstr(0, OTHER, "MLIL_RET", "return")]))
    bv = FBV({CALLEE: "consumer", OTHER: "prior"}, funcs={CALLEE: callee, OTHER: other})
    bv.arch = _arch()
    bv.platform = types.SimpleNamespace(default_calling_convention=_cc())
    return caller, bv, FSSA(p, 0)


def _set(index, dest, src_var):
    return FInstr(index, CALL - 0x10 + index * 2, "MLIL_SET_VAR_SSA", f"{dest} = {src_var}",
                  reads=[src_var], writes=[dest], dest=dest,
                  src=FExpr("MLIL_VAR_SSA", str(src_var), reads=[src_var], src=src_var))


def _prior_call(index, args=()):
    return FInstr(index, CALL - 0x10 + index * 2, "MLIL_CALL_SSA", "prior(...)",
                  reads=[r for a in args for r in a.vars_read], writes=[],
                  dest=FExpr("MLIL_CONST_PTR", hex(OTHER), constant=OTHER), params=list(args))


def _engine(bv, monkeypatch, reads=("r0", "r1", "r2", "r3")):
    engine = te.TaintEngine(bv, te.load_models())
    monkeypatch.setattr(engine, "_reg_reads_as_input", lambda c, r: r in reads)
    # No LLIL in these fakes: nothing beyond the call's own block is proven.
    monkeypatch.setattr(engine, "_abi_arg_operands", lambda *a, **k: [])
    return engine


def _register_frontiers(result, callee="consumer"):
    return [lf for lf in result["leaves"] if lf.get("kind") == "unmodeled_callee"
            and lf.get("cause") in ("arg_under_recovered", "varargs_not_followed")
            and lf.get("evidence") and lf["callee"]["name"] == callee]


# --------------------------------------------------------------------------
# P2-07.1: a register written for an EARLIER call is not this call's argument
# --------------------------------------------------------------------------

@pytest.mark.parametrize("intervening_call", [False, True])
def test_a_register_written_before_an_intervening_call_is_not_this_calls(
        monkeypatch, intervening_call):
    # `r3 = p; prior(.., r3); consumer(x)`: the tainted r3 was the earlier
    # call's argument, and that call clobbered it (argument registers are
    # caller-saved). Without the intervening call it IS still in r3.
    r3 = FSSA(_reg_var("r3_1", "r3", 50), 1)
    caller, bv, _p0 = _program(
        callee_formals=[_reg_var("x", "r0", 60)],
        call_params=[FExpr("MLIL_CONST", "0", constant=0)],
        block_before=[_set(0, r3, FSSA(_reg_var("p", "r0", 40), 0))]
        + ([_prior_call(1)] if intervening_call else []))
    result = _engine(bv, monkeypatch).forward(caller, [te.parse_locator("param:0")])
    frontiers = _register_frontiers(result)
    if intervening_call:
        assert frontiers == [], frontiers
    else:
        [leaf] = frontiers
        assert (leaf["cause"], leaf["registers"]) == ("arg_under_recovered", ["r3"])


# --------------------------------------------------------------------------
# P2-07.1: a declared-variadic callee's extra registers are variadic slots
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variadic", [True, False])
def test_a_variadic_callees_extra_register_is_a_vararg_never_under_recovery(
        monkeypatch, variadic):
    # `log_line(fmt, ...)` called with a tainted value set up in r2 that the
    # MLIL call does not carry. Declared variadic, r2 is a variadic slot --
    # `varargs_not_followed`, never a signature to repair. Declared
    # nonvariadic, and the body uses r2, it is an under-recovered argument.
    p = _reg_var("p", "r0", 40)
    r2 = FSSA(_reg_var("r2_1", "r2", 51), 1)
    caller, bv, _p0 = _program(
        callee_formals=[_reg_var("fmt", "r0", 60)],
        call_params=[FExpr("MLIL_CONST_PTR", "0x5000", constant=0x5000)],
        block_before=[_set(0, r2, FSSA(p, 0))], variadic=variadic)
    # The variadic body's va_start spill is not read as a use: no register
    # read is needed for a variadic slot.
    reads = () if variadic else ("r2",)
    result = _engine(bv, monkeypatch, reads=reads).forward(
        caller, [te.parse_locator("param:0")])
    [leaf] = _register_frontiers(result)
    assert leaf["registers"] == ["r2"]
    assert leaf["cause"] == ("varargs_not_followed" if variadic else "arg_under_recovered")
    if variadic:
        assert leaf["callee_variadic"] is True and "proto set" not in leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# P2-02(b): which registers a call's recovered arguments occupy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("x_bound", ["stack", "r1"])
def test_a_formal_bound_to_the_stack_occupies_no_register(monkeypatch, x_bound):
    # armv7 `put64(char *dst, unsigned long long x, int n)`: EABI passes x in
    # r2:r3, but BN binds x to a stack slot and n to r1. The call carries
    # three arguments, yet r2 -- tainted, and used by the body -- is in none
    # of them; counting the convention's registers in order called it
    # consumed. A tainted RECOVERED argument beside it does not hide it.
    p = _reg_var("p", "r0", 40)
    r2 = FSSA(_reg_var("r2_1", "r2", 51), 1)
    x = _stack_var("x", 61, 0) if x_bound == "stack" else _reg_var("x", "r1", 61)
    caller, bv, _p0 = _program(
        callee_formals=[_reg_var("dst", "r0", 60), x,
                        _reg_var("n", "r1" if x_bound == "stack" else "r2", 62)],
        call_params=[FExpr("MLIL_VAR_SSA", "p#0", reads=[FSSA(p, 0)], src=FSSA(p, 0)),
                     FExpr("MLIL_CONST", "2", constant=2),
                     FExpr("MLIL_CONST", "3", constant=3)],
        block_before=[_set(0, r2, FSSA(p, 0))])
    result = _engine(bv, monkeypatch).forward(caller, [te.parse_locator("param:0")])
    frontiers = _register_frontiers(result)
    if x_bound == "stack":
        [leaf] = frontiers
        assert (leaf["cause"], leaf["registers"], leaf["argument_index"]) == (
            "arg_under_recovered", ["r2"], 2)
    else:
        # Every formal in a register: r2 IS argument 2, carried by the call.
        assert frontiers == [], frontiers


# --------------------------------------------------------------------------
# P2-07.2: a prologue SAVE of an argument register is not an input
# --------------------------------------------------------------------------

def _callee_reading(*instructions, params=()):
    callee = FFunc("noarg", CALLEE, FSSAFunc(list(instructions)), params=list(params))
    bv = FBV({CALLEE: "noarg"}, funcs={CALLEE: callee})
    bv.arch = _arch()
    return te.TaintEngine(bv, te.load_models()), callee


class SSARegister:
    """LLIL-SSA's register operand, matched by type name like BN's own."""

    def __init__(self, reg, version):
        self.reg, self.version = reg, version


def test_a_thumb_alignment_push_is_not_a_read_of_the_register():
    # Thumb `noarg: push {r3, lr}` keeps the stack 8-byte aligned. Its LLIL
    # stores `r3#0` to the stack -- a version-0 read -- and BN's MLIL makes
    # the slot a stack variable (`var_8 = r3`) that nothing reads.
    r3_0 = FSSA(_reg_var("r3", "r3", 70), 0)
    var_8 = FSSA(_stack_var("var_8", 71, -8), 1)
    engine, callee = _callee_reading(
        _set(0, var_8, r3_0),
        FInstr(1, CALLEE + 8, "MLIL_RET", "return", reads=[], src=[]))
    push = types.SimpleNamespace(operands=[
        types.SimpleNamespace(operands=[SSARegister("sp", 1)]), SSARegister("r3", 0)])
    callee.llil = types.SimpleNamespace(ssa_form=types.SimpleNamespace(instructions=[push]))
    assert engine._reg_reads_as_input(callee, "r3") is False


def test_a_spilled_argument_that_is_read_back_is_an_input():
    # -O0 `str r3, [sp, #x]` then a reload: the spill slot IS used.
    r3_0 = FSSA(_reg_var("r3", "r3", 70), 0)
    var_8 = FSSA(_stack_var("var_8", 71, -8), 1)
    out = FSSA(_reg_var("r0_1", "r0", 72), 1)
    engine, callee = _callee_reading(
        _set(0, var_8, r3_0), _set(1, out, var_8),
        FInstr(2, CALLEE + 8, "MLIL_RET", "return", reads=[out], src=[]))
    assert engine._reg_reads_as_input(callee, "r3") is True


def test_a_loop_phi_nothing_reads_is_not_an_input():
    # `rcx_1 = phi(rcx#0, rcx_1#1)` at a loop header whose value is
    # redefined before every read: the entry value is never used.
    rcx0 = FSSA(_reg_var("r3", "r3", 70), 0)
    phi_out, redef = FSSA(_reg_var("r3_1", "r3", 73), 2), FSSA(_reg_var("r3_1", "r3", 73), 1)
    k = FSSA(_reg_var("r0", "r0", 74), 0)
    phi = FInstr(0, CALLEE, "MLIL_VAR_PHI", "r3_1#2 = phi(r3#0, r3_1#1)",
                 reads=[rcx0, redef], writes=[phi_out])
    phi.dest = phi_out
    engine, callee = _callee_reading(
        phi, _set(1, redef, k),
        FInstr(2, CALLEE + 8, "MLIL_RET", "return", reads=[redef], src=[]))
    assert engine._reg_reads_as_input(callee, "r3") is False
    assert engine._reg_reads_as_input(callee, "r0") is True


# --------------------------------------------------------------------------
# P2-02(a): a slice that ends at an unmapped entry register is never complete
# --------------------------------------------------------------------------

def test_a_slice_ending_at_an_unbound_entry_register_is_a_frontier():
    # armv7 `put64`: `r0 = entry_r2; malloc(r0)` -- the real low half of x,
    # in a register the prototype binds no formal to. Reported as a complete
    # slice with origin `entry`.
    entry_r2 = FSSA(_reg_var("entry_r2", "r2", 80), 0)
    r0 = FSSA(_reg_var("r0_1", "r0", 81), 1)
    malloc = 0x9000
    func = FFunc("put64", CALLER, FSSAFunc([
        _set(0, r0, entry_r2),
        FInstr(1, CALL, "MLIL_CALL_SSA", "malloc(r0_1#1)", reads=[r0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", hex(malloc), constant=malloc),
               params=[FExpr("MLIL_VAR_SSA", "r0_1#1", reads=[r0], src=r0)]),
    ]), params=[_reg_var("dst", "r0", 82), _stack_var("x", 83, 0)])
    func.type = "void put64(char* dst, long long unsigned int x, int n)"
    bv = FBV({malloc: "malloc"})
    bv.arch = _arch()
    result = te.TaintEngine(bv, te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "abi_mapping_unbound"]
    assert (leaf["register"], leaf["var"], leaf["function"]["address"]) == (
        "r2", "entry_r2#0", hex(CALLER))
    assert "bn proto set" in leaf["detail"]
    assert [o["kind"] for sl in result["slices"] for o in sl["origins"]] == ["abi_mapping_unbound"]
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_an_entry_value_that_is_not_a_register_keeps_its_entry_origin():
    # Control: an uninitialised STACK local read has no caller location to
    # map; it stays an `entry` origin.
    var_8 = FSSA(_stack_var("var_8", 84, -8), 0)
    r0 = FSSA(_reg_var("r0_1", "r0", 81), 1)
    malloc = 0x9000
    func = FFunc("put64", CALLER, FSSAFunc([
        _set(0, r0, var_8),
        FInstr(1, CALL, "MLIL_CALL_SSA", "malloc(r0_1#1)", reads=[r0], writes=[],
               dest=FExpr("MLIL_CONST_PTR", hex(malloc), constant=malloc),
               params=[FExpr("MLIL_VAR_SSA", "r0_1#1", reads=[r0], src=r0)]),
    ]), params=[])
    bv = FBV({malloc: "malloc"})
    bv.arch = _arch()
    result = te.TaintEngine(bv, te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])
    assert [o["kind"] for sl in result["slices"] for o in sl["origins"]] == ["entry"]
    assert not [lf for lf in result["leaves"] if lf["kind"] == "abi_mapping_unbound"]


_FRAME_REGS = ["r0", "r1", "r2", "r3", "r11", "sp", "lr"]


def _entry_value_slice(reg):
    """`loop: x = phi(<reg>#0, x#1)`; `malloc(x)` in a function whose
    convention passes arguments in r0-r3 -- the shape of a loop phi BN builds
    over a register a `push {..., r11, lr}` prologue saved."""
    idx = {r: i for i, r in enumerate(_FRAME_REGS)}
    v = FVar(reg, ident=90)
    v.source_type, v.storage = 1, idx[reg]
    entry = FSSA(v, 0)
    merged, again = FSSA(FVar("x", ident=91), 2), FSSA(FVar("x", ident=91), 1)
    phi = FInstr(0, CALLER, "MLIL_VAR_PHI", f"x#2 = phi({reg}#0, x#1)", reads=[entry, again],
                 writes=[merged])
    phi.dest, phi.src = merged, [entry, again]
    malloc = 0x9000
    func = FFunc("decode", CALLER, FSSAFunc([
        phi,
        FInstr(1, CALL, "MLIL_CALL_SSA", "malloc(x#2)", reads=[merged], writes=[],
               dest=FExpr("MLIL_CONST_PTR", hex(malloc), constant=malloc),
               params=[FExpr("MLIL_VAR_SSA", "x#2", reads=[merged], src=merged)]),
    ]), params=[_reg_var("src", "r0", 92)])
    func.calling_convention = types.SimpleNamespace(int_arg_regs=["r0", "r1", "r2", "r3"])
    func.type = "void* decode(char* src)"
    bv = FBV({malloc: "malloc"})
    bv.arch = types.SimpleNamespace(
        get_reg_index=lambda r: idx[r], get_reg_name=lambda i: _FRAME_REGS[i],
        regs={r: types.SimpleNamespace(full_width_reg=r) for r in _FRAME_REGS})
    return te.TaintEngine(bv, te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])


@pytest.mark.parametrize("reg", ["lr", "sp", "r11"])
def test_a_frame_or_return_register_entry_value_is_no_unbound_argument(reg):
    # The link register, the stack pointer and the frame pointer carry the
    # return address and the frame, never an argument the prototype failed
    # to map: an `entry` origin, no `abi_mapping_unbound` frontier.
    result = _entry_value_slice(reg)
    assert not [lf for lf in result["leaves"] if lf["kind"] == "abi_mapping_unbound"]
    assert "entry" in {o["kind"] for sl in result["slices"] for o in sl["origins"]}


def test_an_unmapped_argument_register_entry_value_stays_a_frontier():
    # Control on the same shape: r2 is an argument register no formal binds.
    result = _entry_value_slice("r2")
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "abi_mapping_unbound"]
    assert leaf["register"] == "r2"
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


# --------------------------------------------------------------------------
# P2-02(c): a PLT stub typed with its IFUNC resolver's prototype
# --------------------------------------------------------------------------

def _resolver_type():
    fn_type = types.SimpleNamespace(type_class=types.SimpleNamespace(name="FunctionTypeClass"))
    ret = types.SimpleNamespace(type_class=types.SimpleNamespace(name="PointerTypeClass"),
                                target=fn_type)
    return types.SimpleNamespace(return_value=ret)


@pytest.mark.parametrize("resolver", [True, False])
def test_a_plt_stub_with_a_resolver_prototype_is_routed_through_its_model(resolver):
    # armv7 glibc: `memcpy` -> `void* (*memcpy(int32_t hwcap))(...)`, so the
    # MLIL call carries one argument and the tainted length was read only
    # through the ABI register, `class_from: model`, unproven. The stub's
    # prototype is not memcpy's: the call is routed through memcpy's model
    # by name and the length is an argument like any other.
    p = _reg_var("p", "r0", 40)
    stub_at = 0x112bc
    stub = FFunc("memcpy", stub_at, FSSAFunc([]), symbol_type="ImportedFunctionSymbol")
    stub.type = _resolver_type() if resolver else types.SimpleNamespace(return_value=None)
    call = FInstr(0, CALL, "MLIL_CALL_SSA", "memcpy(0x5000)", reads=[], writes=[],
                  dest=FExpr("MLIL_CONST_PTR", hex(stub_at), constant=stub_at),
                  params=[FExpr("MLIL_CONST_PTR", "0x5000", constant=0x5000)])
    call.il_basic_block = [call]
    caller = FFunc("caller", CALLER, FSSAFunc([call]), params=[p])
    caller.calling_convention = _cc()
    bv = FBV({stub_at: "memcpy"}, funcs={stub_at: stub})
    bv.arch = _arch()
    bv.platform = types.SimpleNamespace(default_calling_convention=_cc())
    engine = te.TaintEngine(bv, te.load_models())
    src = FSSA(_reg_var("r1_1", "r1", 90), 1)
    regs = {1: (src, "r1"), 2: (FSSA(p, 0), "r2")}
    engine._abi_arg_operands = lambda f, i, idx, **k: (
        [te._RegisterOperand(regs[idx][0], regs[idx][1])] if idx in regs else [])
    result = engine.forward(caller, [te.parse_locator("param:0")])
    [flow] = result["reached_sinks"]
    assert flow["sink"]["callee"] == "memcpy" and flow["sink"]["tainted_arg_index"] == 2
    if resolver:
        assert "arity_mismatch" not in flow["sink"] and flow["sink"].get("via") != "abi_register"
        assert any("IFUNC resolver" in a and "r1, r2" in a for a in result["assumptions"])
    else:
        # An ordinary under-recovered prototype keeps the reach-only path.
        assert flow["sink"]["via"] == "abi_register"
        assert flow["sink"]["arity_mismatch"]["class_from"] == "model"
