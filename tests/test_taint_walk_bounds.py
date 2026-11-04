"""Forward/backward taint walks stay bounded on ordinary binaries.

The global-memory continuation (IC03) asked its reachability question at
every fixpoint visit and scanned neighbouring globals by mutual recursion, so
a `main` that names a few hundred data symbols ran for minutes or unwound on
Python's recursion limit and lost the sinks it had not reached yet. A float
constant could crash the whole op, a cancelled request kept running, and the
under-recovered-argument gate scanned every callee register before asking
whether anything tainted was there. Each test here pins the consumer-visible
contract that broke.
"""
from __future__ import annotations

import bisect
import types

import pytest
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import (
    G_DIR,
    KIND,
    STRCPY,
    SYSTEM,
    FDataVar,
    FExtentBV,
    FGlobalBV,
    FRef,
    GFunc,
    _const,
    _copy_then_call,
    _forward,
    _gptr,
    _leaves,
    _quiet,
    _reader,
    _var,
)


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# P1-01: the continuation question is asked once per (site, global)
# --------------------------------------------------------------------------

def test_the_continuation_question_is_asked_once_per_site_and_global(models, monkeypatch):
    # Two calls cross a tainted global; the fixpoint needs a second pass to
    # confirm convergence. The question used to be re-asked at every pass for
    # every pair it had not yet disclosed -- the reachability search behind it
    # was nearly all of a forward run on a real binary.
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    tidy = _quiet("tidy", 0x401300)
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), []), (0x40110c, _gptr(0x401300), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook",
                    0x401300: "tidy"},
                   funcs={0x401200: hook, 0x401300: tidy},
                   refs={G_DIR: [FRef(hook, 0x401210)]})
    asked: list[tuple[int, tuple]] = []
    real = te.TaintEngine._call_sees_global

    def spy(self, ssaf, call, sites):
        asked.append((int(call.address), tuple(sorted(sites or ()))))
        return real(self, ssaf, call, sites)

    monkeypatch.setattr(te.TaintEngine, "_call_sees_global", spy)
    result = _forward(bv, func, models)

    # Every call the fixpoint crossed with the global live -- the copy that
    # wrote it included -- is asked about exactly once.
    assert sorted(a for a, _s in asked) == [0x401104, 0x401108, 0x40110c]
    # ...and the answer is the one the per-visit search gave: the reader is
    # disclosed, the call that reaches no reader is not.
    assert [(lf["address"], lf["global"]["address"]) for lf in _leaves(result)] == [
        (hex(0x401108), hex(G_DIR))]


def test_a_call_that_gains_a_tainted_argument_later_is_judged_on_the_final_pass(models):
    # The pass that first crosses the global sees no tainted argument; a later
    # pass does, and the call is then descended -- so its unresolved target is
    # `indirect_call_unresolved`'s to report, not a second boundary.
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    tmp = FSSA(FVar("p"), 1)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "strcpy(g, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "cb(p#1)", reads=[tmp],
               dest=FExpr("MLIL_VAR_SSA", "cb", reads=[]), params=[_var(tmp)]),
        FInstr(2, 0x40110c, "MLIL_SET_VAR_SSA", "p#1 = argv#0", reads=[argv0],
               writes=[tmp], src=_var(argv0)),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    kinds = sorted(lf["kind"] for lf in result["leaves"]
                   if lf.get("address") == hex(0x401108))
    assert kinds == ["indirect_call_unresolved"], result["leaves"]


# --------------------------------------------------------------------------
# P1-02: the neighbouring-object scan is linear, not one frame per global
# --------------------------------------------------------------------------

SECTION = (0x500000, 0x540000)
G_FIRST = 0x500000
N_GLOBALS = 2000
G_TAINT = G_FIRST + 8 * N_GLOBALS     # the tainted object, after the run


class FWideSectionBV(FExtentBV):
    """``FExtentBV`` with indexed layout queries, so a section holding
    thousands of symbols answers in O(log n) like a real view does."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._starts = sorted(set(self._objects) | set(self._interior))
        self._ref_addrs = sorted(self._refs)

    def get_data_var_at(self, addr):
        addr = int(addr)
        at = bisect.bisect_right(self._starts, addr) - 1
        if at < 0:
            return None
        base = self._starts[at]
        width = self._objects.get(base)
        if width is None:
            return FDataVar(addr, 0) if addr == base else None
        if (width and addr < base + width) or (not width and addr == base):
            return FDataVar(base, width, base not in self._authoritative)
        return None

    def get_next_data_var_after(self, addr):
        at = bisect.bisect_right(self._starts, int(addr))
        if at >= len(self._starts):
            return None
        a = self._starts[at]
        return FDataVar(a, self._objects.get(a, 0))

    def get_previous_data_var_before(self, addr):
        at = bisect.bisect_left(self._starts, int(addr)) - 1
        if at < 0:
            return None
        a = self._starts[at]
        return FDataVar(a, self._objects.get(a, 0))

    def get_code_refs(self, addr, length=None):
        lo = int(addr)
        hi = lo + (int(length) if length else 1)
        out = []
        for ref_addr in self._ref_addrs[bisect.bisect_left(self._ref_addrs, lo):
                                        bisect.bisect_left(self._ref_addrs, hi)]:
            out.extend(self._refs[ref_addr])
        return out


def _wide_section_program(consumer):
    """``main(argv)``: copy argv into G_TAINT, call *consumer*, then hand argv
    to ``system``. G_TAINT sits after N_GLOBALS symbolled objects in one
    section, so the search for an enclosing object runs out of hops."""
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "strcpy(g, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_TAINT), _var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "consumer()",
               dest=_gptr(int(consumer.start)), params=[]),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "system(argv#0)", reads=[argv0],
               dest=_gptr(SYSTEM), params=[_var(argv0)]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    globals_ = [G_FIRST + 8 * i for i in range(N_GLOBALS)]
    refs = {G_TAINT: [FRef(func, 0x401104)]}
    for ins in consumer.mlil.ssa_form.instructions:
        target = getattr(getattr(ins, "dest", None), "constant", None)
        if target in globals_:
            refs.setdefault(target, []).append(FRef(consumer, ins.address))
    objects = {g: 8 for g in globals_}
    objects[G_TAINT] = 0x40
    bv = FWideSectionBV({STRCPY: "strcpy", SYSTEM: "system",
                         int(consumer.start): consumer.name},
                        funcs={int(consumer.start): consumer}, refs=refs,
                        objects=objects, section=SECTION)
    return func, bv


def _assert_walk_finished(result):
    # The copy itself is a sink; the one that matters is the `system(argv)`
    # AFTER the call, which an unwind at the call never reached.
    assert hex(0x40110c) in [f["sink"]["address"] for f in result["reached_sinks"]]
    assert "recursion" not in result["stats"]["truncation_cause"], result["stats"]
    assert not [lf for lf in result["leaves"] if lf["kind"] == "analysis_truncated"]


def test_a_body_naming_two_thousand_globals_is_classified_without_recursion(models):
    # `reset_all()` zeroes 2000 neighbouring globals, highest first. Asking
    # whether it touches the tainted object's region re-entered the access
    # scan once per global it names -- the walk unwound on Python's frame
    # limit and the `system(argv)` AFTER the call was never reached.
    stores = [FInstr(k, 0x402000 + 4 * k, "MLIL_STORE_SSA", f"[{hex(g)}].q = 0",
                     dest=_gptr(g), src=_const(0))
              for k, g in enumerate(G_FIRST + 8 * i for i in reversed(range(N_GLOBALS)))]
    consumer = GFunc("reset_all", 0x402000, FSSAFunc(stores), params=[])
    func, bv = _wide_section_program(consumer)
    result = _forward(bv, func, models)
    _assert_walk_finished(result)
    # The bound the enclosing-object search hit is still disclosed.
    [leaf] = _leaves(result)
    assert leaf["address"] == hex(0x401108)
    assert "object_search_budget" in leaf["reachability"]["causes"], leaf
    assert "global_object_search" in result["stats"]["truncation_cause"]


def test_a_section_of_two_thousand_unnamed_neighbours_is_scanned_without_recursion(models):
    # A body that names none of the tainted object's predecessors asks, of
    # each, whether it reaches it -- and that object's answer asks the same
    # of ITS predecessors, one frame chain down the whole section.
    consumer = _quiet("tidy", 0x402000)
    func, bv = _wide_section_program(consumer)
    result = _forward(bv, func, models)
    _assert_walk_finished(result)
    assert _leaves(result) == []


# --------------------------------------------------------------------------
# P1-02: a recursion unwind names a remedy that exists
# --------------------------------------------------------------------------

HELPER = 0x401300


def _seed_then_descend():
    """``main(argv)``: system(argv), then helper(argv) -- a descended call."""
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    q = FVar("q")
    q0 = FSSA(q, 0)
    helper = GFunc("helper", HELPER, FSSAFunc([
        FInstr(0, HELPER + 4, "MLIL_RET", "return", reads=[q0], src=[])]), params=[q])
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "system(argv#0)", reads=[argv0],
               dest=_gptr(SYSTEM), params=[_var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "helper(argv#0)", reads=[argv0],
               dest=_gptr(HELPER), params=[_var(argv0)]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    bv = FGlobalBV({SYSTEM: "system", HELPER: "helper"}, funcs={HELPER: helper})
    return func, bv


@pytest.mark.parametrize("cut_at, depth", [(0x401100, 0), (HELPER, 1)])
def test_a_recursion_unwind_reports_where_it_was_cut(models, monkeypatch, cut_at, depth):
    real = te.TaintEngine._unimplemented_addrs

    def unwind_at(self, func, instrs):
        if int(func.start) == cut_at:
            raise RecursionError("synthetic")
        return real(self, func, instrs)

    monkeypatch.setattr(te.TaintEngine, "_unimplemented_addrs", unwind_at)
    func, bv = _seed_then_descend()
    result = _forward(bv, func, models)

    [cut] = [lf for lf in result["leaves"] if lf["kind"] == "analysis_truncated"]
    assert (cut["cause"], cut["depth"], cut["address"]) == ("recursion", depth, hex(cut_at))
    assert "recursion" in result["stats"]["truncation_cause"]
    if depth:
        # The sink seen before the descent survives the unwind, and the bound
        # advised is one under which that descent does not happen.
        assert [f["sink"]["address"] for f in result["reached_sinks"]] == [hex(0x401104)]
        assert "--max-depth 0" in cut["detail"]
    else:
        # Cut in the seeded function itself: no depth bound avoids it, so
        # none is offered -- `--max-depth 0` fails the same way.
        assert result["reached_sinks"] == []
        assert "--max-depth" not in result["diagnostics"]["next_action"].replace(
            "no `--max-depth` avoids it", "")


# --------------------------------------------------------------------------
# P1-03: a float constant names no address
# --------------------------------------------------------------------------

def test_an_infinite_float_constant_does_not_escape_the_op(models):
    # An MLIL float constant can be `inf`; `int(inf)` raised OverflowError,
    # which escaped the whole op after minutes of work.
    inf = FExpr("MLIL_FLOAT_CONST", "inf", constant=float("inf"))
    x1 = FSSA(FVar("x"), 1)
    reader = GFunc("run_hook", 0x401200, FSSAFunc([
        FInstr(0, 0x401204, "MLIL_SET_VAR_SSA", "x#1 = inf", writes=[x1], src=inf),
        FInstr(1, 0x401208, "MLIL_CALL_SSA", "scale(x#1, g)", reads=[x1],
               dest=_gptr(0x401400), params=[inf, _gptr(G_DIR)]),
        FInstr(2, 0x401210, "MLIL_CALL_SSA", f"system({hex(G_DIR)})",
               dest=_gptr(SYSTEM), params=[_gptr(G_DIR)]),
    ]), params=[])
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401200), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook",
                    0x401400: "scale"},
                   funcs={0x401200: reader},
                   refs={G_DIR: [FRef(reader, 0x401208), FRef(reader, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["address"] == hex(0x401108)
    assert [r["display_name"] for r in leaf["references"]] == ["run_hook"], leaf


# --------------------------------------------------------------------------
# P2-06: the tainted-write gate comes before the callee scan
# --------------------------------------------------------------------------

def _reg_var(name, index):
    var = FVar(name)
    var.source_type = 1          # RegisterVariableSourceType
    var.storage = index
    return var


def test_only_a_register_carrying_taint_costs_a_callee_scan(models, monkeypatch):
    # `consumer()` was recovered with no arguments; the call's block wrote a
    # constant into rdi and the tainted parameter into rsi. Only rsi can be a
    # dropped tainted argument, so only rsi is worth asking the callee about.
    p = FVar("p")
    p0 = FSSA(p, 0)
    rdi1 = FSSA(_reg_var("rdi", 0), 1)
    rsi1 = FSSA(_reg_var("rsi", 1), 1)
    consumer = FFunc("consumer", 0x405000, FSSAFunc([
        FInstr(0, 0x405004, "MLIL_RET", "return", src=[])]), params=[])
    consumer.calling_convention = types.SimpleNamespace(int_arg_regs=["rdi", "rsi"])
    set_rdi = FInstr(0, 0x401000, "MLIL_SET_VAR_SSA", "rdi#1 = 5", writes=[rdi1],
                     src=_const(5))
    set_rsi = FInstr(1, 0x401004, "MLIL_SET_VAR_SSA", "rsi#1 = p#0", reads=[p0],
                     writes=[rsi1], src=_var(p0))
    call = FInstr(2, 0x401008, "MLIL_CALL_SSA", "consumer()", dest=_gptr(0x405000),
                  params=[])
    block = [set_rdi, set_rsi, call]
    for ins in block:
        ins.il_basic_block = block
    entry = FFunc("caller", 0x401000, FSSAFunc(block), params=[p])
    bv = FBV({0x405000: "consumer"}, funcs={0x405000: consumer})
    bv.arch = types.SimpleNamespace(get_reg_index=lambda r: {"rdi": 0, "rsi": 1}[r])
    bv.platform = types.SimpleNamespace(
        default_calling_convention=types.SimpleNamespace(int_arg_regs=["rdi", "rsi"]))
    asked: list[str] = []
    monkeypatch.setattr(te.TaintEngine, "_reg_reads_as_input",
                        lambda self, callee, reg: asked.append(reg) or True)
    result = te.TaintEngine(bv, models).forward(entry, [te.parse_locator("param:0")])

    assert set(asked) == {"rsi"}, asked
    [leaf] = [lf for lf in result["leaves"] if lf.get("cause") == "arg_under_recovered"]
    assert (leaf["registers"], leaf["via"]) == (["rsi"], "block_write"), leaf


# --------------------------------------------------------------------------
# P1-07: a cancelled request stops the walk
# --------------------------------------------------------------------------

def test_a_cancelled_walk_stops_between_function_visits(models, monkeypatch):
    func, bv = _seed_then_descend()
    entered: list[int] = []
    real = te.TaintEngine._unimplemented_addrs

    def record(self, fn, instrs):
        entered.append(int(fn.start))
        return real(self, fn, instrs)

    monkeypatch.setattr(te.TaintEngine, "_unimplemented_addrs", record)
    # The request is cancelled once the first function's walk has started.
    engine = te.TaintEngine(bv, models, cancel_check=lambda: bool(entered))
    with pytest.raises(te.TaintCancelled) as caught:
        engine.forward(func, [te.parse_locator("param:0")])
    assert caught.value.functions_visited == 1
    assert entered == [0x401100]          # the descent into helper never ran


def test_a_cancelled_backward_slice_stops(models):
    func, bv = _seed_then_descend()
    engine = te.TaintEngine(bv, models, cancel_check=lambda: True)
    with pytest.raises(te.TaintCancelled):
        engine.backward(func, [te.parse_locator("arg:system:0")])
