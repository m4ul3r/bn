"""Fixpoint contracts of the forward walk.

* A call is revisited when the CONTEXT it hands the callee grows, not only
  when its set of tainted argument indexes does: a struct argument whose
  tainted fields grow on a later pass (a field stored after the call, reached
  again over a back edge) is descended again with the larger field set, and
  the revisit reports nothing twice.
* `_settle_access` answers a cyclic dependency between accesses the way a
  fresh question would: an answer computed while a node it asked about was
  still unfinished is not cached as final -- the cycle is iterated until
  every answer in it is stable -- and the resolver stays an explicit stack,
  so a chain thousands of accesses long needs no Python recursion.

All names and addresses are invented.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_taint_claim_gaps import _field_store, _ret
from test_taint_engine import FExpr, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import MEMCPY, FGlobalBV, GFunc, _const, _forward, _gptr, _var

from bn_agent_bridge.taint_engine import _Use

USE_BOTH = 0x401600


@pytest.fixture
def models():
    return te.load_models()


def _two_field_consumer(name, start, offsets):
    """`consume(struct rec *o)`: for each offset, `n = o-><offset>;
    memcpy(&dst, o, n)` -- one length sink per field, in order."""
    o0 = FSSA(FVar("o", typ="struct rec*"), 0)
    dst = FExpr("MLIL_ADDRESS_OF", "&dst", src=FVar("dst", typ="char[0x100]"))
    instrs = []
    for k, off in enumerate(offsets):
        n = FSSA(FVar(f"n{k}"), 1)
        addr = FExpr("MLIL_ADD", f"o#0 + {off:#x}", reads=[o0], left=_var(o0), right=_const(off))
        load = FExpr("MLIL_LOAD_SSA", f"[{addr}].q", reads=[o0], src=addr)
        load.size = 8
        base = start + 4 + 8 * k
        instrs.append(FInstr(2 * k, base, "MLIL_SET_VAR_SSA", f"n{k}#1 = [{addr}].q",
                             reads=[o0], writes=[n], src=load))
        instrs.append(FInstr(2 * k + 1, base + 4, "MLIL_CALL_SSA", f"memcpy(&dst, o#0, n{k}#1)",
                             reads=[o0, n], dest=_gptr(MEMCPY), params=[dst, _var(o0), _var(n)]))
    instrs.append(FInstr(len(instrs), start + 4 + 8 * len(offsets), "MLIL_RET", "return 0",
                         src=[_const(0)]))
    return GFunc(name, start, FSSAFunc(instrs), params=[o0.var])


def _loop_program():
    """`main(arg)`: `rec.len0 = arg` (+0); loop { `use_both(&rec)`;
    `rec.len1 = arg` (+0x10) } -- the +0x10 store reaches the call only over
    the loop's back edge, so the call's tainted ARGUMENT set ({0}) is the same
    on both passes while the fields it carries grow."""
    arg0 = FSSA(FVar("arg"), 0)
    rec = FVar("rec", typ="struct rec")
    first, _ = _field_store(0, 0x401104, rec, 1, arg0, offset=0)
    later, _ = _field_store(2, 0x40110c, rec, 2, arg0, offset=0x10)
    consumer = _two_field_consumer("use_both", USE_BOTH, (0, 0x10))
    r = FSSA(FVar("r"), 1)
    call = FInstr(1, 0x401108, "MLIL_CALL_SSA", "r#1 = use_both(&rec)", writes=[r],
                  dest=_gptr(USE_BOTH),
                  params=[FExpr("MLIL_ADDRESS_OF", "&rec", src=rec)])
    func = GFunc("main", 0x401100, FSSAFunc([first, call, later, _ret(3, 0x401110)]),
                 params=[arg0.var])
    return func, consumer


def _bv(*funcs):
    return FGlobalBV({MEMCPY: "memcpy", **{fn.start: fn.name for fn in funcs}},
                     funcs={fn.start: fn for fn in funcs})


def test_a_call_is_revisited_when_the_fields_it_carries_grow(models):
    # Pass 1 hands `use_both` only +0 (its first sink); the +0x10 store lands
    # after the call, and the next pass must hand it +0 and +0x10. Keyed on
    # the argument indexes alone, the second pass skipped the call: one sink
    # and an all-clear for the field read at the second memcpy.
    func, consumer = _loop_program()
    result = _forward(_bv(consumer), func, models)
    assert [(s["sink"]["class"], s["sink"]["address"]) for s in result["reached_sinks"]] == [
        ("overflow_len", hex(USE_BOTH + 8)), ("overflow_len", hex(USE_BOTH + 16))]
    assert result["stats"]["sinks"] == 2


def test_revisiting_a_call_reports_nothing_twice(models):
    # The first context's sink is reached again by the second; it is still
    # one finding, and no leaf is listed twice.
    func, consumer = _loop_program()
    result = _forward(_bv(consumer), func, models)
    sites = [s["sink"]["address"] for s in result["reached_sinks"]]
    assert len(sites) == len(set(sites)) == 2
    rows = [json.dumps(lf, sort_keys=True) for lf in result["leaves"]]
    assert len(rows) == len(set(rows))
    assert result["stats"]["leaves"] == len(result["leaves"])


def test_a_called_frame_that_revisits_a_call_carries_each_sink_once(models):
    # The same loop one call down: `entry(arg)` calls `main(arg)`. Each
    # pass that re-descends `use_both` re-reaches the first context's sink;
    # the summary `main` hands its callers names it once, so a frame's
    # findings do not multiply with every context any frame below revisits.
    main, consumer = _loop_program()
    arg0 = FSSA(FVar("arg"), 0)
    r = FSSA(FVar("r"), 1)
    entry = GFunc("entry", 0x401000, FSSAFunc([
        FInstr(0, 0x401004, "MLIL_CALL_SSA", "r#1 = main(arg#0)", reads=[arg0], writes=[r],
               dest=_gptr(main.start), params=[_var(arg0)]),
        _ret(1, 0x401008),
    ]), params=[arg0.var])
    engine = te.TaintEngine(_bv(consumer, main), models)
    result = engine.forward(entry, [te.parse_locator("param:0")])
    assert [s["sink"]["address"] for s in result["reached_sinks"]] == [
        hex(USE_BOTH + 8), hex(USE_BOTH + 16)]
    [summary] = [v for k, v in engine._cache.items() if k[0] == main.start]
    assert [f["sink"]["address"] for f in summary["findings"]] == [
        hex(USE_BOTH + 8), hex(USE_BOTH + 16)]
