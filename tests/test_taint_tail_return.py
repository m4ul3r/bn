"""Forward return summaries must include tail calls, not their arguments alone."""
from __future__ import annotations

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te


def _const(value):
    return FExpr("MLIL_CONST_PTR", hex(value), constant=value)


def _value(var):
    return FExpr("MLIL_VAR_SSA", str(var), reads=[var], src=var)


def _program(kind, *, tail=True, late=False, wrapper_returns_value=False):
    n, p, result, temporary = [FVar(s) for s in ("n", "p", "result", "temporary")]
    ns, ps, rs, ts = FSSA(n, 0), FSSA(p, 0), FSSA(result, 1), FSSA(temporary, 1)
    argument = ts if late else ps
    wrap_call = FInstr(0, 0x2010, "MLIL_TAILCALL_SSA" if tail else "MLIL_CALL_SSA",
                       "callee(p)", reads=[argument], writes=[] if tail else [ts],
                       params=[_value(argument)], dest=_const(0x3000))
    instructions = [wrap_call]
    if late:
        instructions.append(FInstr(1, 0x2014, "MLIL_SET_VAR_SSA", "temporary = p",
                                   reads=[ps], writes=[ts], src=_value(ps), dest=ts))
    if not tail:
        instructions.append(FInstr(2, 0x2018, "MLIL_RET", "return temporary",
                                   reads=[], src=[_value(ts)]))
    elif wrapper_returns_value:
        # `if (!take) return 0;` beside the tail call: the wrapper returns a value.
        instructions.append(FInstr(3, 0x201c, "MLIL_RET", "return 0", reads=[],
                                   src=[FExpr("MLIL_CONST", "0", constant=0)]))
    wrapper = FFunc("wrap", 0x2000, FSSAFunc(instructions), params=[p])
    main = FFunc("main", 0x1000, FSSAFunc([
        FInstr(0, 0x1010, "MLIL_CALL_SSA", "result = wrap(n)", reads=[ns], writes=[rs],
               params=[_value(ns)], dest=_const(0x2000)),
        FInstr(1, 0x1014, "MLIL_CALL_SSA", "memcpy(dst, src, result)", reads=[rs],
               params=[_const(0x6000), _const(0x7000), _value(rs)], dest=_const(0x4000)),
    ]), params=[n])
    funcs = {0x1000: main, 0x2000: wrapper}
    overlay = None
    if kind == "valueless":
        # AArch64 `ident(v) { return v; }`: a bare `ret`, recovered with no operand.
        funcs[0x3000] = FFunc("callee", 0x3000, FSSAFunc([
            FInstr(0, 0x3010, "MLIL_RET", "return", reads=[], src=[])]), params=[FVar("q")])
    if kind in ("tainted", "clean"):
        q = FVar("q")
        funcs[0x3000] = FFunc("callee", 0x3000, FSSAFunc([
            FInstr(0, 0x3010, "MLIL_RET", "return q" if kind == "tainted" else "return 0",
                   reads=[], src=[_value(FSSA(q, 0))] if kind == "tainted"
                   else [FExpr("MLIL_CONST", "0", constant=0)])]), params=[q])
    elif kind == "skipped":
        funcs[0x3000] = FFunc("callee", 0x3000, FSSAFunc([]), params=[FVar("q")])
        funcs[0x3000].analysis_skipped = True
        funcs[0x3000].analysis_skip_reason = type(
            "SkipReason", (), {"name": "ExceedFunctionSizeSkipReason"})()
    elif kind in ("model", "model_buffer", "model_clean"):
        overlay = {"callee": {} if kind == "model_clean" else {"propagates": [
            {"from": "arg:0", "to": "*ret" if kind == "model_buffer" else "ret"}]}}
    return main, FBV({0x3000: "callee", 0x4000: "memcpy"}, funcs), te.load_models(overlay)


def _walk(kind, *, tail=True, late=False, policy="conservative"):
    main, bv, models = _program(kind, tail=tail, late=late)
    return te.TaintEngine(bv, models, unknown_call_policy=policy).forward(
        main, [te.parse_locator("param:0")])


@pytest.mark.parametrize("kind", ["external", "skipped", "tainted", "model", "model_buffer"])
@pytest.mark.parametrize("tail", [True, False], ids=["tail", "ordinary-call-return"])
def test_callee_return_reaches_caller_sink(kind, tail):
    result = _walk(kind, tail=tail)
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"])


@pytest.mark.parametrize("kind", ["clean", "model_clean"])
def test_tainted_tail_arguments_do_not_taint_clean_return(kind):
    assert not _walk(kind)["reached_sinks"]


@pytest.mark.parametrize("kind", ["external", "skipped"])
def test_tail_unknown_stop_preserves_frontier_without_return(kind):
    result = _walk(kind, policy="stop")
    assert not result["reached_sinks"]
    assert any(l["kind"] == "unmodeled_callee" for l in result["leaves"])


@pytest.mark.parametrize("kind", ["external", "tainted", "model"])
def test_tail_return_fixpoint_observes_late_argument_taint(kind):
    result = _walk(kind, late=True)
    assert any(s["sink"]["class"] == "overflow_len" for s in result["reached_sinks"])


def _valueless_leaves(result):
    return [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]


@pytest.mark.parametrize(("tail", "wrapper_returns_value"), [
    (True, True),     # the #1034 item 5 shape: conditional tail wrapper
    (False, False),   # an ordinary call whose result BN recovered as used
])
def test_valueless_return_on_a_consumed_call_is_a_frontier_1034(tail, wrapper_returns_value):
    main, bv, models = _program("valueless", tail=tail,
                                wrapper_returns_value=wrapper_returns_value)
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])

    assert not result["reached_sinks"]
    [leaf] = _valueless_leaves(result)
    assert leaf["callee"]["address"] == "0x3000"
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize(("kind", "wrapper_returns_value"), [
    ("valueless", False),   # a void wrapper tail-calls a void callee: nothing consumed
    ("clean", True),        # BN recovered a value; it is simply untainted
])
def test_recovered_or_unconsumed_return_is_no_frontier_1034(kind, wrapper_returns_value):
    main, bv, models = _program(kind, wrapper_returns_value=wrapper_returns_value)
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])

    assert _valueless_leaves(result) == []


# --------------------------------------------------------------------------
# #1034 item 5 at the call site (#1050 review round 9): every descended call
# with tainted arguments to which BN's MLIL gives NO result is a blocking
# frontier. Whether the caller uses the return register is not checked.
# --------------------------------------------------------------------------

def _dropped_result_program(*, callee_returns_value):
    """main(n): `wrap(n)` -- an MLIL call with NO result. Nothing about the
    caller's machine code is given: no LLIL is consulted."""
    n, p = FVar("n"), FVar("p")
    ns, ps = FSSA(n, 0), FSSA(p, 0)
    ret_src = [_value(ps)] if callee_returns_value else []
    callee = FFunc("wrap", 0x2000, FSSAFunc([
        FInstr(0, 0x2004, "MLIL_RET", "return", reads=[ps] if ret_src else [],
               src=ret_src)]), params=[p])
    main = FFunc("main", 0x1000, FSSAFunc([
        FInstr(0, 0x1010, "MLIL_CALL_SSA", "wrap(n)", reads=[ns], writes=[],
               params=[_value(ns)], dest=_const(0x2000))]), params=[n])
    return main, FBV({}, {0x1000: main, 0x2000: callee}), te.load_models()


def _dropped(result):
    return [lf for lf in result["leaves"]
            if lf.get("cause") == "return_value_unrecovered" and lf["address"] == "0x1010"]


@pytest.mark.parametrize("callee_returns_value", [True, False],
                         ids=["value-callee", "void-callee"])
def test_a_no_result_call_with_tainted_arguments_is_always_a_frontier_1034(callee_returns_value):
    """The void callee is the accepted precision cost: it withholds the
    all-clear too, since the caller's use of the register is not checked."""
    main, bv, models = _dropped_result_program(callee_returns_value=callee_returns_value)
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])

    [leaf] = _dropped(result)
    assert leaf["kind"] == "unmodeled_callee"
    assert "return_register" not in leaf and "return_register_use" not in leaf
    assert "bn disasm 0x1000" in str(leaf)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False



# --------------------------------------------------------------------------
# BN25-META-01: one call cannot be both modeled and unmodeled, and the
# verdict must not depend on what ran before it.
# --------------------------------------------------------------------------

def _modeled_body_program(name):
    """main(n) calls an IN-BINARY body named *name* with no recovered result.

    The static shape: a `-static` build has a real body for the library
    function a model also covers, so the body is descended (#807) while the
    model's rules run at the same call site."""
    n, p = FVar("n"), FVar("p")
    ns, ps = FSSA(n, 0), FSSA(p, 0)
    body = FFunc(name, 0x2000, FSSAFunc([
        FInstr(0, 0x2004, "MLIL_RET", "return p", reads=[ps], src=[_value(ps)])]),
        params=[p])
    main = FFunc("main", 0x1000, FSSAFunc([
        FInstr(0, 0x1010, "MLIL_CALL_SSA", f"{name}(n)", reads=[ns], writes=[],
               params=[_value(ns)], dest=_const(0x2000))]), params=[n])
    return main, FBV({0x2000: name}, {0x1000: main, 0x2000: body}), te.load_models()


def test_a_modeled_callee_is_not_also_reported_unmodeled():
    """The model states what the call hands back, so the frontier that
    exists for an UNKNOWN return channel does not apply to it. Reporting
    both said the same call was modeled and unmodeled at once."""
    main, bv, models = _modeled_body_program("wmemcpy")
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])
    assert _dropped(result) == [], result["leaves"]


def test_an_unmodeled_callee_still_raises_the_dropped_result_frontier():
    main, bv, models = _modeled_body_program("project_helper")
    result = te.TaintEngine(bv, models).forward(main, [te.parse_locator("param:0")])
    assert _dropped(result), result["leaves"]


class _LateThunk:
    """A function whose ``is_thunk`` flag only becomes True once Binary
    Ninja has lifted it -- which is how BN really behaves on a freshly
    analysed view."""

    def __init__(self, inner):
        self._inner = inner
        self._lifted = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def is_thunk(self):
        return self._lifted

    @property
    def mlil(self):
        self._lifted = True
        return self._inner.mlil


def test_a_thunk_is_recognised_before_anything_else_lifted_it():
    """BN25-META-01: asking the timing-dependent flag BEFORE the body was
    lifted answered False for a stub that IS one, so the first request
    descended into it and every later one did not -- two identical
    requests, two different results."""
    stub = _LateThunk(FFunc("__wmemcpy", 0x2000, FSSAFunc([
        FInstr(0, 0x2004, "MLIL_TAILCALL_SSA", "tailcall memcpy", reads=[],
               params=[], dest=_const(0x3000))]), params=[FVar("p")]))
    engine = te.TaintEngine(FBV({0x2000: "__wmemcpy"}, {0x2000: stub}),
                            te.load_models())
    assert engine._not_internal_reason(stub) == "thunk"
    assert engine._not_internal_reason(stub) == "thunk"
