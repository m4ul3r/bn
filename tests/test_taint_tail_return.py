"""Forward return summaries must include tail calls, not their arguments alone."""
from __future__ import annotations

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te


def _const(value):
    return FExpr("MLIL_CONST_PTR", hex(value), constant=value)


def _value(var):
    return FExpr("MLIL_VAR_SSA", str(var), reads=[var], src=var)


def _program(kind, *, tail=True, late=False):
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
    wrapper = FFunc("wrap", 0x2000, FSSAFunc(instructions), params=[p])
    main = FFunc("main", 0x1000, FSSAFunc([
        FInstr(0, 0x1010, "MLIL_CALL_SSA", "result = wrap(n)", reads=[ns], writes=[rs],
               params=[_value(ns)], dest=_const(0x2000)),
        FInstr(1, 0x1014, "MLIL_CALL_SSA", "memcpy(dst, src, result)", reads=[rs],
               params=[_const(0x6000), _const(0x7000), _value(rs)], dest=_const(0x4000)),
    ]), params=[n])
    funcs = {0x1000: main, 0x2000: wrapper}
    overlay = None
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
