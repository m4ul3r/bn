"""Forward taint's global-memory continuation frontier.

Tainted bytes a modeled copy writes into a *keyable writable global* are
program state, not frame state. The forward walk keeps its taint set per
function and descends a call only when an ARGUMENT at that call carries taint,
so a callee that reads the global -- and hands those bytes to a modeled sink --
used to be reached by nothing at all: zero sinks, zero leaves and
``safe_to_report_all_clear: true`` (the lane-5 ``globalflow`` and lane-6
``global_frontier`` shapes).

The fix does not propagate global state across the program (no whole-program
solver). It DISCLOSES the boundary: when BN's own code-reference database shows
a function that references the tainted global is reachable from a call the walk
makes, the call emits a blocking ``global_memory_continuation`` leaf naming the
global, the call and where the global is referenced. Without that evidence --
a clean global, a callee that touches a different global, a target that does
not resolve, a view with no code-reference API -- nothing is emitted, so the
frontier is never a blanket "a global was written" marker.
"""
from __future__ import annotations

import pytest
from test_taint_engine import (
    FBV,
    FExpr,
    FFunc,
    FInstr,
    FPVS,
    FSSA,
    FSSAFunc,
    FVar,
    te,
)

from bn import formatters
from bn_agent_bridge.taint_result import BLOCKING_LEAF_KINDS

KIND = "global_memory_continuation"

G_DIR = 0x404060        # the global the copy fills
G_OTHER = 0x404160      # a second, unrelated global
STRNCPY, STRCPY, SPRINTF, MEMCPY, SYSTEM = 0x401030, 0x401040, 0x401050, 0x401060, 0x401070
MEMSET = 0x401080
STRCAT = 0x401090


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# fakes: a BinaryView that can answer "who references this address", and
# functions that expose BN's `callees` edge so reachability can be walked.
# --------------------------------------------------------------------------

class FRef:
    def __init__(self, function, address):
        self.function = function
        self.address = address


class FGlobalBV(FBV):
    """``FBV`` plus BN's code-reference DB and writable-segment introspection."""

    def __init__(self, addr_names, funcs=None, refs=None, writable_from=0x404000):
        super().__init__(addr_names, funcs)
        self._refs = refs or {}
        self._writable_from = writable_from

    def get_code_refs(self, addr):
        return list(self._refs.get(int(addr), []))

    def is_offset_writable(self, addr):
        return int(addr) >= self._writable_from


class GFunc(FFunc):
    """A function that also exposes its direct callees (BN's ``Function.callees``)."""

    def __init__(self, *args, callees=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.callees = list(callees)


def _var(ssa, text=None):
    return FExpr("MLIL_VAR_SSA", text or f"{ssa.var.name}#{ssa.version}", reads=[ssa])


def _const(value):
    return FExpr("MLIL_CONST", hex(value), constant=value)


def _gptr(addr):
    return FExpr("MLIL_CONST_PTR", hex(addr), constant=addr)


def _reader(name, start, global_addr, read_at, *, callees=()):
    """A callee whose body reads *global_addr* and hands it to ``system``."""
    instrs = [
        FInstr(0, read_at, "MLIL_CALL_SSA", f"system({hex(global_addr)})", writes=[],
               dest=_gptr(SYSTEM), params=[_gptr(global_addr)]),
    ]
    return GFunc(name, start, FSSAFunc(instrs), params=[], callees=callees)


def _quiet(name, start, *, callees=()):
    """A callee that touches no global at all."""
    instrs = [FInstr(0, start, "MLIL_RET", "return 0")]
    return GFunc(name, start, FSSAFunc(instrs), params=[], callees=callees)


def _copy_then_call(copy_addr, copy_target, dest, *, call_sites, params_extra=None,
                    main_start=0x401100, call_dest=None):
    """``main(argv)``: copy the tainted parameter into *dest*, then make each
    call in *call_sites* (``(address, dest_expr, params)``)."""
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    instrs = [
        FInstr(0, copy_addr, "MLIL_CALL_SSA",
               f"{hex(copy_target)}({dest}, argv#0)", reads=[argv0], writes=[],
               dest=_gptr(copy_target),
               params=[dest if not isinstance(dest, int) else _gptr(dest),
                       _var(argv0), *(params_extra or [])]),
    ]
    for i, (addr, dexpr, params) in enumerate(call_sites, start=1):
        instrs.append(FInstr(i, addr, "MLIL_CALL_SSA", f"call @ {hex(addr)}",
                             writes=[], dest=dexpr, params=list(params)))
    return GFunc("main", main_start, FSSAFunc(instrs), params=[argv]), argv


def _globalflow(*, copy_target=STRNCPY, dest_global=G_DIR, hook_reads=G_DIR,
                extra_copy_params=None, hook_callees=(), hook_start=0x401200,
                hook_read_at=0x401210, refs_extra=None, bv_cls=FGlobalBV,
                hook_params=(), hook_call_params=()):
    """The lane-5 shape: ``main`` copies argv into a global, then calls a hook
    that references ``hook_reads``."""
    hook = _reader("run_hook", hook_start, hook_reads, hook_read_at, callees=hook_callees)
    func, argv = _copy_then_call(
        0x401104, copy_target, dest_global,
        call_sites=[(0x401108, _gptr(hook_start), list(hook_call_params))],
        params_extra=extra_copy_params)
    hook.parameter_vars = list(hook_params)
    refs = {hook_reads: [FRef(hook, hook_read_at)]}
    for addr, rows in (refs_extra or {}).items():
        refs.setdefault(addr, []).extend(rows)
    bv = bv_cls({STRNCPY: "strncpy", STRCPY: "strcpy", SPRINTF: "sprintf",
                 MEMCPY: "memcpy", SYSTEM: "system", hook_start: "run_hook"},
                funcs={hook_start: hook}, refs=refs)
    return func, hook, bv


def _forward(bv, func, models, source="param:0", **kw):
    return te.TaintEngine(bv, models).forward(func, [te.parse_locator(source)], **kw)


def _leaves(result, kind=KIND):
    return [lf for lf in result["leaves"] if lf["kind"] == kind]


# --------------------------------------------------------------------------
# the defect: a keyable global destination and a zero-argument consumer
# --------------------------------------------------------------------------

def test_keyable_global_copy_into_zero_arg_consumer_emits_continuation(models):
    # strncpy(g_dir, argv[1], n) then run_hook(): no argument of run_hook()
    # carries taint, so the walk never descends -- and the bytes it would find
    # are sitting in g_dir. Base: reached_sinks=[], leaves=[], all-clear true.
    func, hook, bv = _globalflow(extra_copy_params=[_const(0xff)])
    result = _forward(bv, func, models)

    [leaf] = _leaves(result)
    assert leaf["address"] == hex(0x401108)                       # the call boundary
    assert leaf["global"] == {"address": hex(G_DIR), "label": f"glob_{hex(G_DIR)}"}
    assert leaf["callee"]["address"] == hex(hook.start)
    assert leaf["references"] == [{"name": "run_hook", "display_name": "run_hook",
                                   "address": hex(hook.start), "at": hex(0x401210),
                                   "access": "call_arg", "offset": 0}]
    assert leaf["reachability"]["status"] == "complete"
    assert leaf["call_descended"] is False
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert KIND in BLOCKING_LEAF_KINDS


def test_continuation_survives_a_copy_that_itself_fires_as_a_sink(models):
    # strcpy flags its SOURCE operand as a sink, which suppresses the "copied
    # into the destination" caveat -- the run that finds a sink used to
    # disclose LESS about where the bytes went than the run that finds none.
    # The frontier is derived from the tainted global, not from that caveat.
    func, hook, bv = _globalflow(copy_target=STRCPY)
    result = _forward(bv, func, models)

    assert any(s["sink"]["callee"] == "strcpy" for s in result["reached_sinks"]), result
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G_DIR)
    # A positive sink row is not a substitute for the frontier, and the sink
    # does not silence the leaf channel.
    assert leaf["references"][0]["at"] == hex(0x401210)


@pytest.mark.parametrize("target,extra", [
    (MEMCPY, [_const(8)]),
    (STRCPY, []),
    (SPRINTF, []),
])
def test_every_keyable_global_copy_destination_is_accounted(models, target, extra):
    # memcpy / strcpy / sprintf all declare `*arg:1 -> *arg:0`; none may be
    # less honestly accounted for because its destination was keyable.
    func, hook, bv = _globalflow(copy_target=target, extra_copy_params=extra)
    result = _forward(bv, func, models)
    assert [lf["global"]["address"] for lf in _leaves(result)] == [hex(G_DIR)]


# --------------------------------------------------------------------------
# controls: the frontier needs evidence, and must not fire without it
# --------------------------------------------------------------------------

def test_consumer_that_reads_a_different_global_is_not_a_continuation(models):
    # The hook references G_OTHER; G_DIR is the tainted one. Reachability is
    # asked per global, so the unrelated reference proves nothing.
    func, hook, bv = _globalflow(hook_reads=G_OTHER)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]


def test_clean_global_is_not_a_continuation(models):
    # The copy's destination is a LOCAL, so no global is ever tainted -- even
    # though the hook reads G_DIR on every run.
    dst = FVar("dst")
    local_dest = FExpr("MLIL_ADDRESS_OF", "&dst", src=dst)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(0x401104, STRCPY, local_dest,
                                  call_sites=[(0x401108, _gptr(0x401200), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]


def test_a_clean_sibling_global_is_not_disclosed(models):
    # Only the global the run actually tainted is a boundary. The second
    # global is filled from a read-only literal and its own consumer stays
    # silent, so the disclosure is per global, not "a global was written".
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    hot = _reader("hot_hook", 0x401200, G_DIR, 0x401210)
    cool = _reader("cool_hook", 0x401300, G_OTHER, 0x401310)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "strcpy(g_dir, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "strcpy(g_other, 0x402000)",
               dest=_gptr(STRCPY), params=[_gptr(G_OTHER), _gptr(0x402000)]),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "cool_hook()", dest=_gptr(0x401300), params=[]),
        FInstr(3, 0x401110, "MLIL_CALL_SSA", "hot_hook()", dest=_gptr(0x401200), params=[]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system",
                    0x401200: "hot_hook", 0x401300: "cool_hook"},
                   funcs={0x401200: hot, 0x401300: cool},
                   refs={G_DIR: [FRef(hot, 0x401210)],
                         G_OTHER: [FRef(cool, 0x401310)]})
    result = _forward(bv, func, models)
    assert [(lf["address"], lf["global"]["address"]) for lf in _leaves(result)] == [
        (hex(0x401110), hex(G_DIR))]


def test_a_later_clean_copy_does_not_retract_the_boundary(models):
    # A clean copy over the SAME global afterwards does not clear its taint:
    # this engine's global model is coarse and version-agnostic (global_approx)
    # and taint only ever grows, so the boundary is still disclosed. Pinned
    # deliberately -- over-reporting costs a reviewable frontier, the other
    # direction loses the bug on any path where the overwrite does not run.
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "strcpy(g_dir, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "strcpy(g_dir, 0x402000)",
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _gptr(0x402000)]),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "run_hook()", dest=_gptr(0x401200), params=[]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    assert [lf["address"] for lf in _leaves(result)] == [hex(0x40110c)]


def test_callee_that_touches_no_global_is_not_a_continuation(models):
    # "No requirement to walk unrelated no-argument callees": a quiet callee
    # after a global write stays silent -- no blanket frontier.
    quiet = _quiet("bookkeeping", 0x401300)
    hook_refs = {}
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401300: "bookkeeping"},
                   funcs={0x401300: quiet}, refs=hook_refs)
    result = _forward(bv, func, models)
    assert result["leaves"] == [], result["leaves"]


def test_no_code_ref_api_fabricates_no_frontier(models):
    # A view that cannot enumerate code references (the unit fakes, a partially
    # loaded view) has no evidence either way: stay silent rather than invent a
    # boundary on every call after a global write.
    class NoRefsBV(FGlobalBV):
        get_code_refs = None

    func, hook, bv = _globalflow(bv_cls=NoRefsBV)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]


def test_an_unresolved_zero_argument_callback_is_a_blocking_boundary(models):
    # Round-3 review F10: control leaves for somewhere bn could not name,
    # with tainted global bytes live and code in this binary that references
    # them. Nothing proves the target is not one of those readers, so the
    # boundary is disclosed -- silence here is the false all-clear.
    fp = FVar("fp")
    fp1 = FSSA(fp, 1)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _var(fp1), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["callee"] is None, leaf
    assert leaf["dest_expr"] == "fp#1", leaf
    assert leaf["reachability"]["cause"] == "unresolved_call_target"
    assert [s["address"] for s in leaf["reachability"]["unexamined_reference_sites"]] \
        == [hex(0x401200)], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_an_unresolved_callback_without_a_live_global_stays_silent(models):
    # The control: no tainted global, so there is nothing for the unknown
    # target to read and no global boundary is invented for it.
    fp = FVar("fp")
    fp1 = FSSA(fp, 1)
    dst = FVar("dst")
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
        call_sites=[(0x401108, _var(fp1), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    assert _leaves(_forward(bv, func, models)) == []


def test_a_resolved_map_target_replaces_the_unresolved_boundary(models):
    # --resolve-map pins the callback, so the walk answers with the real
    # target instead of the unresolved-target boundary.
    fp = FVar("fp")
    fp1 = FSSA(fp, 1)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _var(fp1), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    engine = te.TaintEngine(bv, models, resolve_map={hex(0x401108): [hex(0x401200)]})
    result = engine.forward(func, [te.parse_locator("param:0")])
    [leaf] = _leaves(result)
    assert leaf["callee"]["address"] == hex(0x401200), leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf


# --------------------------------------------------------------------------
# reach: descended calls, resolved indirect targets, transitivity, depth
# --------------------------------------------------------------------------

def test_consumer_with_explicit_tainted_args_still_discloses_the_global(models):
    # The call DOES carry a tainted argument, so the walk descends it -- but it
    # descends with the argument's taint only. The global the callee also reads
    # is not carried across, so the boundary is disclosed all the same.
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    hook.parameter_vars = [FVar("p")]
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [_var(argv0)])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["call_descended"] is True


def test_resolved_indirect_candidate_is_followed_for_the_global(models):
    # A value-set-resolved target is a resolved body like any other: the same
    # reachability question is asked of it.
    fp = FVar("fp")
    fp1 = FSSA(fp, 1)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    dest = FExpr("MLIL_VAR_SSA", "fp#1", reads=[fp1],
                 possible_values=FPVS("ConstantPointerValue", value=0x401200))
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR, call_sites=[(0x401108, dest, [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    assert [lf["callee"]["address"] for lf in _leaves(result)] == [hex(0x401200)]


def test_transitive_consumer_within_the_depth_budget(models):
    # main -> wrapper -> deep_hook, and only deep_hook references the global.
    deep = _reader("deep_hook", 0x401400, G_DIR, 0x401410)
    wrapper = _quiet("wrapper", 0x401300, callees=[deep])
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system",
                    0x401300: "wrapper", 0x401400: "deep_hook"},
                   funcs={0x401300: wrapper, 0x401400: deep},
                   refs={G_DIR: [FRef(deep, 0x401410)]})
    reached = _forward(bv, func, models)
    [leaf] = _leaves(reached)
    assert leaf["callee"]["address"] == hex(0x401300)        # the call boundary
    assert leaf["references"][0]["address"] == hex(0x401400)  # where it is read

    # ... and with no descent budget left the walk cannot reach deep_hook, so
    # it must NOT answer "nothing reads it": the boundary is disclosed as a
    # TRUNCATED reachability answer naming the bound and what it never
    # examined, and the run is marked truncated with the matching cause
    # (round-3 review F2 -- the same rule the function budget follows).
    capped = _forward(bv, func, models, max_depth=0)
    [cut] = _leaves(capped)
    assert cut["references"] == [], cut
    assert cut["reachability"]["status"] == "truncated"
    assert cut["reachability"]["cause"] == "depth_budget"
    assert cut["reachability"]["unexamined_references"] == [hex(0x401400)]
    assert capped["stats"]["truncated"] is True
    assert "max_depth" in capped["stats"]["truncation_cause"]


def test_recursive_call_graph_terminates(models):
    # a <-> b with the reader behind the cycle: reachability must converge.
    reader = _reader("reader", 0x401500, G_DIR, 0x401510)
    a = _quiet("a", 0x401300)
    b = _quiet("b", 0x401400, callees=[a, reader])
    a.callees = [b]
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401300: "a",
                    0x401400: "b", 0x401500: "reader"},
                   funcs={0x401300: a, 0x401400: b, 0x401500: reader},
                   refs={G_DIR: [FRef(reader, 0x401510)]})
    result = _forward(bv, func, models, max_depth=8)
    assert [lf["references"][0]["address"] for lf in _leaves(result)] == [hex(0x401500)]


def test_two_tainted_globals_are_disclosed_separately(models):
    # Two copies, two globals, one consumer each: one leaf per (call, global),
    # never one merged "a global escaped" marker.
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    hook_a = _reader("hook_a", 0x401200, G_DIR, 0x401210)
    hook_b = _reader("hook_b", 0x401300, G_OTHER, 0x401310)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "strcpy(g_dir, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(argv0)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "strcpy(g_other, argv#0)", reads=[argv0],
               dest=_gptr(STRCPY), params=[_gptr(G_OTHER), _var(argv0)]),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "hook_a()", dest=_gptr(0x401200), params=[]),
        FInstr(3, 0x401110, "MLIL_CALL_SSA", "hook_b()", dest=_gptr(0x401300), params=[]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[argv])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system",
                    0x401200: "hook_a", 0x401300: "hook_b"},
                   funcs={0x401200: hook_a, 0x401300: hook_b},
                   refs={G_DIR: [FRef(hook_a, 0x401210)],
                         G_OTHER: [FRef(hook_b, 0x401310)]})
    result = _forward(bv, func, models)
    assert sorted((lf["address"], lf["global"]["address"]) for lf in _leaves(result)) == [
        (hex(0x40110c), hex(G_DIR)), (hex(0x401110), hex(G_OTHER))]


def test_one_leaf_per_call_and_global_across_fixpoint_passes(models):
    # The fixpoint revisits every instruction; the disclosure is per (call,
    # global), not per pass, and a second call to the SAME consumer is its own
    # boundary.
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), []), (0x40110c, _gptr(0x401200), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    assert sorted(lf["address"] for lf in _leaves(result)) == [hex(0x401108), hex(0x40110c)]


# --------------------------------------------------------------------------
# result shaping: diagnostics, per-source union, text
# --------------------------------------------------------------------------

def test_zero_sink_diagnostics_count_and_explain_the_continuation(models):
    func, hook, bv = _globalflow()
    diag = _forward(bv, func, models)["diagnostics"]
    assert diag["frontier"]["by_kind"][KIND] == 1
    assert diag["frontier"]["coarse_memory"] == 1
    assert diag["safe_to_report_all_clear"] is False
    assert KIND in diag["all_clear_reason"]
    assert "global" in diag["next_action"] and hex(G_DIR) in diag["next_action"]


def test_per_source_attribution_unions_the_continuation(models):
    # Two call sites of the seeded callee are analysed independently; the
    # top-level leaves are the union and each callsite keeps its own count.
    rax = FVar("rax")
    rax1, rax2 = FSSA(rax, 1), FSSA(rax, 2)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    instrs = [
        FInstr(0, 0x401104, "MLIL_CALL_SSA", "rax#1 = getenv(0x402000)", writes=[rax1],
               dest=_gptr(0x401080), params=[_gptr(0x402000)]),
        FInstr(1, 0x401108, "MLIL_CALL_SSA", "rax#2 = getenv(0x402010)", writes=[rax2],
               dest=_gptr(0x401080), params=[_gptr(0x402010)]),
        FInstr(2, 0x40110c, "MLIL_CALL_SSA", "strcpy(g_dir, rax#1)", reads=[rax1],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(rax1)]),
        FInstr(3, 0x401110, "MLIL_CALL_SSA", "strcpy(g_dir, rax#2)", reads=[rax2],
               dest=_gptr(STRCPY), params=[_gptr(G_DIR), _var(rax2)]),
        FInstr(4, 0x401114, "MLIL_CALL_SSA", "run_hook()", dest=_gptr(0x401200), params=[]),
    ]
    func = GFunc("main", 0x401100, FSSAFunc(instrs), params=[])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401080: "getenv",
                    0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models, source="ret:getenv")
    assert len(_leaves(result)) == 1, result["leaves"]
    assert set(result["by_source"]) == {hex(0x401104), hex(0x401108)}
    for site in result["by_source"].values():
        assert [lf["kind"] for lf in site["leaves"]] == [KIND]
        assert site["frontier"] == 1


def test_text_rendering_names_the_global_the_call_and_the_reference():
    leaf = {
        "kind": KIND,
        "address": "0x401108",
        "global": {"address": "0x404060", "label": "glob_0x404060"},
        "callee": {"name": "run_hook", "display_name": "run_hook", "address": "0x401200"},
        "references": [{"name": "run_hook", "display_name": "run_hook",
                        "address": "0x401200", "at": "0x401210", "access": "call_arg"}],
        "reachability": {"status": "complete", "cause": None,
                         "functions_examined": 1, "function_budget": 512,
                         "unexamined_references_total": 0, "unexamined_references": []},
        "call_descended": False,
        "detail": "the bytes are not followed past this call",
    }
    line = formatters._render_leaf_line(leaf)
    assert KIND in line and "0x401108" in line and "0x404060" in line
    assert "run_hook" in line and "0x401210" in line and "call_arg" in line
    assert "the bytes are not followed past this call" in line

    # --mangled prints the raw symbol for the callee AND for the reference, the
    # same name pair every other leaf row carries (#978).
    cpp = {**leaf,
           "callee": {"name": "_Z8run_hookv", "display_name": "run_hook()",
                      "address": "0x401200"},
           "references": [{"name": "_Z8run_hookv", "display_name": "run_hook()",
                           "address": "0x401200", "at": "0x401210"}]}
    assert "run_hook()" in formatters._render_leaf_line(cpp)
    raw = formatters._render_leaf_line(cpp, mangled=True)
    assert "_Z8run_hookv" in raw and "run_hook()" not in raw, raw

    # Two boundaries over distinct globals are distinct rows, not one
    # "[2 leaves]" group keyed on the kind alone.
    other = {**leaf, "address": "0x401110",
             "global": {"address": "0x404160", "label": "glob_0x404160"}}
    assert formatters._leaf_group_key(leaf) != formatters._leaf_group_key(other)
    rows = [r for r in formatters._render_grouped_leaves([leaf, other]) if KIND in r]
    assert len(rows) == 2 and not any("leaves]" in r for r in rows), rows


def test_text_rendering_of_a_truncated_reachability_row():
    # A row that names no reader must say WHY, with the bound and what it
    # never examined -- otherwise it reads as "a boundary with nothing behind
    # it", which is the all-clear this frontier exists to withhold.
    leaf = {
        "kind": KIND,
        "address": "0x401108",
        "global": {"address": "0x404060", "label": "glob_0x404060"},
        "callee": {"name": "fanout", "display_name": "fanout", "address": "0x401200"},
        "references": [],
        "reachability": {"status": "truncated", "cause": "function_budget",
                         "functions_examined": 512, "function_budget": 512,
                         "unexamined_references_total": 1,
                         "unexamined_references": ["0x401900"]},
        "call_descended": False,
        "detail": "the search was truncated",
    }
    line = formatters._render_leaf_line(leaf)
    assert "truncated" in line and "function_budget" in line
    assert "512/512" in line and "1 reference(s) unexamined" in line


# --------------------------------------------------------------------------
# F1 (round-3 review): a resolvable in-binary thunk is not a dead end
# --------------------------------------------------------------------------

def _thunk(name, start, target_start, *, is_thunk=True, resolvable=True):
    """A single-instruction tail-call veneer, the shape `follow_thunk` takes.

    ``resolvable=False`` keeps BN's thunk flag but gives the body a second
    instruction, so the shared resolver declines it."""
    instrs = [FInstr(0, start, "MLIL_TAILCALL_SSA", f"jump({hex(target_start)})",
                     dest=_gptr(target_start), params=[])]
    if not resolvable:
        instrs.append(FInstr(1, start + 4, "MLIL_RET", "return"))
    return GFunc(name, start, FSSAFunc(instrs), params=[], is_thunk=is_thunk)


def _behind(chain, reader_start=0x401500, reader_at=0x401510):
    """`main` copies into G_DIR then calls the head of a veneer *chain* that
    ends at a reader of G_DIR. Returns (func, bv)."""
    reader = _reader("consumer", reader_start, G_DIR, reader_at)
    funcs = {reader_start: reader}
    names = {STRNCPY: "strncpy", SYSTEM: "system", reader_start: "consumer"}
    nxt = reader_start
    for i, start in enumerate(reversed(chain)):
        veneer = _thunk(f"veneer{len(chain) - i}", start, nxt)
        funcs[start] = veneer
        names[start] = veneer.name
        nxt = start
    head = chain[0]
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR, call_sites=[(0x401108, _gptr(head), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV(names, funcs=funcs, refs={G_DIR: [FRef(reader, reader_at)]})
    return func, bv


def test_a_direct_thunk_to_the_consumer_is_followed(models):
    # main -> veneer (a one-instruction tail call) -> consumer, which hands
    # the tainted global to system. The veneer is not a BODY, so the check
    # used to drop it and report a clean run; it must resolve the veneer with
    # the same helper the descent uses.
    func, bv = _behind([0x401300])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["callee"]["address"] == hex(0x401500)        # the resolved body
    assert leaf["references"][0]["at"] == hex(0x401510)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_multi_hop_thunk_chain_to_the_consumer_is_followed(models):
    # veneer1 -> veneer2 -> consumer: `follow_thunk` walks the whole chain.
    func, bv = _behind([0x401300, 0x401400])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["callee"]["address"] == hex(0x401500)
    assert leaf["references"][0]["at"] == hex(0x401510)


def test_an_unresolved_in_binary_thunk_fails_closed(models):
    # BN marks it a thunk but the shared resolver cannot take its body, so
    # its references cannot be read: "nothing reads the global" is not an
    # answer here, and the boundary is disclosed as truncated.
    reader = _reader("consumer", 0x401500, G_DIR, 0x401510)
    veneer = _thunk("veneer", 0x401300, 0x401500, resolvable=False)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system",
                    0x401300: "veneer", 0x401500: "consumer"},
                   funcs={0x401300: veneer, 0x401500: reader},
                   refs={G_DIR: [FRef(reader, 0x401510)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"] == [], leaf
    assert leaf["reachability"]["cause"] == "unresolved_thunk"
    assert leaf["reachability"]["unexamined_references"] == [hex(0x401500)]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # An unreadable callee is not a BUDGET: the leaf carries the whole story,
    # and the run is not mislabelled as depth- or budget-truncated.
    assert result["stats"]["truncation_cause"] == [], result["stats"]


def test_a_thunk_onto_an_import_stays_external(models):
    # A PLT stub resolving to an import is not a body this walk could read,
    # and the unmodeled/modeled-callee machinery owns that call as before --
    # no global boundary is invented for it.
    imported = GFunc("system", 0x401500, FSSAFunc([]), params=[],
                     symbol_type="ImportedFunctionSymbol")
    veneer = _thunk("system@plt", 0x401300, 0x401500)
    hook = _reader("run_hook", 0x401600, G_DIR, 0x401610)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", 0x401300: "system@plt", 0x401500: "system",
                    0x401600: "run_hook"},
                   funcs={0x401300: veneer, 0x401500: imported, 0x401600: hook},
                   refs={G_DIR: [FRef(hook, 0x401610)]})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]


# --------------------------------------------------------------------------
# F2 (round-3 review): the reachability bound is tri-state, never a clear
# --------------------------------------------------------------------------

def _fanout(n_callees, *, reader_last=True):
    """`main` copies into G_DIR then calls `fanout`, which has *n_callees*
    direct callees; the reader of G_DIR is the last of them."""
    reader = _reader("consumer", 0x402000, G_DIR, 0x402010)
    filler = [_quiet(f"q{i}", 0x410000 + i * 0x10) for i in range(n_callees - 1)]
    callees = [*filler, reader] if reader_last else [reader, *filler]
    fan = _quiet("fanout", 0x401300, callees=callees)
    names = {STRNCPY: "strncpy", SYSTEM: "system", 0x401300: "fanout",
             0x402000: "consumer"}
    funcs = {0x401300: fan, 0x402000: reader}
    for q in filler:
        names[q.start] = q.name
        funcs[q.start] = q
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR, call_sites=[(0x401108, _gptr(0x401300), [])],
        params_extra=[_const(0xff)])
    return func, FGlobalBV(names, funcs=funcs, refs={G_DIR: [FRef(reader, 0x402010)]})


def test_fanout_inside_the_function_budget_finds_the_consumer(models):
    # 511 callees + the call's own target = exactly the 512-function bound.
    func, bv = _fanout(511)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["at"] == hex(0x402010)
    assert leaf["reachability"]["status"] == "complete"
    assert result["stats"]["truncated"] is False


@pytest.mark.parametrize("n", [512, 513])
def test_fanout_past_the_function_budget_discloses_the_truncation(models, n):
    # One function over the bound and the reader is never examined. The old
    # behaviour was silence -- zero sinks, zero leaves, a true all-clear.
    func, bv = _fanout(n)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"] == [], leaf
    reach = leaf["reachability"]
    assert reach["status"] == "truncated" and reach["cause"] == "function_budget"
    assert reach["functions_examined"] == 512 and reach["function_budget"] == 512
    assert reach["unexamined_references"] == [hex(0x402000)]
    assert reach["unexamined_references_total"] == 1
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert result["stats"]["truncated"] is True
    assert "global_reference_budget" in result["stats"]["truncation_cause"]
    assert "function budget" in result["diagnostics"]["next_action"] \
        or "TRUNCATED" in result["diagnostics"]["next_action"]


def test_a_bound_that_hides_nothing_is_still_complete(models):
    # The bound is hit, but every function BN records a reference from was
    # already examined -- so it cannot have hidden a reader and the answer
    # stands. Without this the frontier would fire on any wide call graph.
    func, bv = _fanout(600, reader_last=False)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["at"] == hex(0x402010)
    assert leaf["reachability"]["status"] == "complete"
    assert result["stats"]["truncated"] is False


# --------------------------------------------------------------------------
# F3 (round-3 review): a reference is a consumer only if it can READ
# --------------------------------------------------------------------------

def _store_only(name, start, global_addr, at):
    """`g[0] = 0` and nothing else: a reference that cannot read the bytes."""
    instrs = [FInstr(0, at, "MLIL_STORE_SSA", f"[{hex(global_addr)}].b = 0",
                     dest=_gptr(global_addr), src=_const(0))]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def _load_only(name, start, global_addr, at):
    """`x = g[0]`: a plain load of the global's bytes."""
    v = FSSA(FVar("x"), 1)
    instrs = [FInstr(0, at, "MLIL_SET_VAR_SSA", f"x#1 = [{hex(global_addr)}]",
                     writes=[v],
                     src=FExpr("MLIL_LOAD_SSA", f"[{hex(global_addr)}]",
                               src=_gptr(global_addr)))]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def _modify(name, start, global_addr, at):
    """`g[0] += 1`: a read-modify-write still reads."""
    load = FExpr("MLIL_LOAD_SSA", f"[{hex(global_addr)}]", src=_gptr(global_addr))
    instrs = [FInstr(0, at, "MLIL_STORE_SSA", f"[{hex(global_addr)}].b = .. + 1",
                     dest=_gptr(global_addr),
                     src=FExpr("MLIL_ADD", "load + 1", left=load, right=_const(1)))]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def _mixed(name, start, global_addr, store_at, read_at):
    """A store into the global followed by a call that is handed its address."""
    instrs = [
        FInstr(0, store_at, "MLIL_STORE_SSA", f"[{hex(global_addr)}].b = 0",
               dest=_gptr(global_addr), src=_const(0)),
        FInstr(1, read_at, "MLIL_CALL_SSA", f"system({hex(global_addr)})",
               dest=_gptr(SYSTEM), params=[_gptr(global_addr)]),
    ]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def _after_copy(callee):
    """`main`: strcpy(G_DIR, argv[0]) then call *callee* with no arguments."""
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(callee.start)), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", int(callee.start): callee.name},
                   funcs={int(callee.start): callee},
                   refs={G_DIR: [FRef(callee, 0x401210)]})
    return func, bv


def test_a_write_only_reference_is_not_a_consumer(models):
    # `g[0] = 0` consumes nothing. Naming it a consumer manufactures a
    # blocking frontier the base engine never had, on a program where the
    # bytes genuinely go nowhere.
    func, bv = _after_copy(_store_only("write_only", 0x401200, G_DIR, 0x401210))
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


@pytest.mark.parametrize("build,access", [
    (_load_only, "load"),
    (_modify, "load"),
])
def test_a_reading_reference_is_a_consumer(models, build, access):
    func, bv = _after_copy(build("reader", 0x401200, G_DIR, 0x401210))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == access
    assert leaf["references"][0]["at"] == hex(0x401210)


def test_a_mixed_write_then_read_reference_is_a_consumer(models):
    # The store alone would be write-only; the call that is handed the
    # address is what makes it a consumer, and the leaf cites THAT site.
    func, bv = _after_copy(_mixed("overwrite_then_run", 0x401200, G_DIR,
                                  0x401210, 0x401218))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0] == {
        "name": "overwrite_then_run", "display_name": "overwrite_then_run",
        "address": hex(0x401200), "at": hex(0x401218), "access": "call_arg",
        "offset": 0}



# --------------------------------------------------------------------------
# round-2 review: the classifier defaults to CONSUMER, and only a proven
# write -- a store, or a modeled write-only destination argument -- clears it
# --------------------------------------------------------------------------

def _phi_reader(name, start, global_addr, at):
    """`p = cond ? g : other; system(p)`: the pointer reaches the call through
    an SSA PHI, which no single-reader copy chain resolves."""
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    instrs = [
        FInstr(0, start, "MLIL_SET_VAR_SSA", f"p#1 = {hex(global_addr)}",
               writes=[p1], src=_gptr(global_addr)),
        FInstr(1, start + 4, "MLIL_SET_VAR_SSA", f"p#2 = {hex(G_OTHER)}",
               writes=[p2], src=_gptr(G_OTHER)),
        FInstr(2, start + 8, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
               writes=[p3], src=[p1, p2]),
        FInstr(3, at, "MLIL_CALL_SSA", "system(p#3)", reads=[p3],
               dest=_gptr(SYSTEM), params=[_var(p3)]),
    ]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def test_a_phi_selected_global_pointer_is_a_consumer(models):
    # Round-2 review R1: a branch-selected pointer is still that pointer. The
    # classifier used to find no `load` and no directly-resolvable call
    # argument and call the whole body write-only -- a false all-clear on an
    # ordinary `cond ? g : other` consumer.
    func, bv = _after_copy(_phi_reader("conditional_reader", 0x401200, G_DIR, 0x401210))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] != "write_only", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and the mechanism is the resolver itself: the call argument reaches
    # the global only through the PHI, and `_may_be_global` follows it. A
    # single-reader copy chain (what `_global_addr` walks) stops at the merge.
    eng = te.TaintEngine(bv, models)
    phi_fn = bv.get_function_at(0x401200)
    ssaf = phi_fn.mlil.ssa_form
    call = ssaf.instructions[3]
    assert eng._global_addr(ssaf, call.params[0]) is None
    assert eng._may_be_global(ssaf, call.params[0], G_DIR) is True


def test_a_materialised_address_with_no_terminal_use_is_a_consumer(models):
    # The address is materialised and the pointer then goes nowhere this
    # scan can account for: no terminal use at all, so no write was proven.
    # Defaulting to "it only writes" is what produced the false all-clear;
    # the conservative answer is a disclosed boundary.
    v = FSSA(FVar("p"), 1)
    body = GFunc("opaque_use", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_SET_VAR_SSA", f"p#1 = {hex(G_DIR)}",
               writes=[v], src=_gptr(G_DIR)),
    ]), params=[])
    func, bv = _after_copy(body)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "unclassified", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def _modeled_call(name, start, callee_addr, params, at=0x401210):
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"{hex(callee_addr)}(...)",
               dest=_gptr(callee_addr), params=list(params)),
    ]), params=[])


@pytest.mark.parametrize("callee,params", [
    (MEMSET, [_gptr(G_DIR), _const(0), _const(64)]),          # memset(g, 0, 64)
    (MEMCPY, [_gptr(G_DIR), _gptr(0x402000), _const(5)]),     # memcpy(g, "safe", 5)
])
def test_a_modeled_write_only_destination_is_not_a_reader(models, callee, params):
    # Round-2 review R2: the model says these callees only WRITE through arg0.
    # Treating the argument as a read invented a blocking frontier on a helper
    # that overwrites the global with constants.
    helper = _modeled_call("reset_helper", 0x401200, callee, params)
    func, bv = _after_copy(helper)
    bv._names[callee] = {MEMSET: "memset", MEMCPY: "memcpy"}[callee]
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_modeled_source_argument_is_still_a_reader(models):
    # The mirror: the same model reads through arg1, so a helper that copies
    # the tainted global OUT of it is a consumer.
    helper = _modeled_call("leak_helper", 0x401200, MEMCPY,
                           [_gptr(G_OTHER), _gptr(G_DIR), _const(16)])
    func, bv = _after_copy(helper)
    bv._names[MEMCPY] = "memcpy"
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf


def test_an_unmodeled_call_handed_the_global_is_an_honest_unknown(models):
    # No model, no promise. The boundary is disclosed -- but as an UNKNOWN,
    # not as a claimed read: nothing here saw the callee consume the bytes
    # (round-3 review F5), and the prose says "may read".
    helper = _modeled_call("unknown_helper", 0x401200, 0x401900, [_gptr(G_DIR)])
    func, bv = _after_copy(helper)
    bv._names[0x401900] = "app_blob"
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "unknown_call_arg", leaf
    assert "may read" in leaf["detail"] and "reachable from this call reads" not in leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_call_argument_binary_ninja_did_not_recover_is_an_honest_unknown(models):
    # The global is in the call instruction but in no RECOVERED argument:
    # BN under-recovered the arity, so what the callee does with the bytes is
    # unknown. Silence here is the false all-clear; a claimed read is a lie.
    helper = GFunc("short_arity", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_CALL_SSA", f"app_blob() /* {hex(G_DIR)} */",
               dest=_gptr(0x401900), params=[],
               src=_gptr(G_DIR)),
    ]), params=[])
    func, bv = _after_copy(helper)
    bv._names[0x401900] = "app_blob"
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "unknown_call_arg", leaf


def test_a_modeled_write_destination_beside_a_read_is_still_a_reader(models):
    # memcpy(g, g, 8): arg0 is the model's write destination, arg1 its source.
    # One write-only position does not clear the call.
    helper = _modeled_call("churn", 0x401200, MEMCPY,
                           [_gptr(G_DIR), _gptr(G_DIR), _const(8)])
    func, bv = _after_copy(helper)
    bv._names[MEMCPY] = "memcpy"
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf


def test_a_scalar_store_to_another_global_is_silent(models):
    # The control for the write-only rule: a helper that writes an UNRELATED
    # global is not a reference to the tainted one at all.
    helper = _store_only("scalar_helper", 0x401200, G_OTHER, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "scalar_helper"},
                   funcs={0x401200: helper},
                   refs={G_OTHER: [FRef(helper, 0x401210)]})
    result = _forward(bv, func, models)
    assert result["leaves"] == [], result["leaves"]


# --------------------------------------------------------------------------
# round-2 review R3: an in-binary consumer BN never analysed
# --------------------------------------------------------------------------

def test_an_unanalysed_in_binary_consumer_is_disclosed(models):
    # BN measured the body and skipped it (oversized past
    # analysis.limits.maxFunctionSize). Its references cannot be read, so
    # "nothing reads the global" is not an answer -- and the run says its
    # analysis was incomplete, which it was.
    skipped = GFunc("huge_reader", 0x401200, FSSAFunc([]), params=[])
    skipped.analysis_skipped = True
    skipped.analysis_skip_reason = type("R", (), {"name": "ExceedFunctionSizeSkipReason"})()
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "huge_reader"},
                   funcs={0x401200: skipped},
                   refs={G_DIR: [FRef(skipped, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    # Native evidence: BN's code-reference DB already places a reference to
    # the global INSIDE the body that could not be read, so the leaf cites it
    # -- marked `unreadable`, so the prose says "may read", never "reads".
    assert leaf["references"] == [{"name": "huge_reader", "display_name": "huge_reader",
                                   "address": hex(0x401200), "at": hex(0x401210),
                                   "access": "unreadable", "offset": 0}], leaf
    assert "may read" in leaf["detail"]
    assert leaf["reachability"]["cause"] == "unread_callee_body"
    assert leaf["reachability"]["status"] == "truncated"
    assert "force-analysis" in leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert result["stats"]["analysis_incomplete"] is True
    assert "huge_reader" in result["stats"]["analysis_incomplete_functions"]


def test_an_in_binary_callee_with_no_il_at_all_is_disclosed(models):
    # Round-3 review F6: "skipped" and "no MLIL at all" are the same thing to
    # this question -- the body's use of the global cannot be read either way,
    # so neither may continue to an all-clear.
    stub = GFunc("app_handler", 0x401200, FSSAFunc([]), params=[])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "app_handler"},
                   funcs={0x401200: stub}, refs={G_DIR: [FRef(stub, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] == "unread_callee_body"
    assert result["stats"]["analysis_incomplete"] is True


def test_a_modeled_callee_with_no_body_stays_the_model_s_story(models):
    # The control that keeps F6 from firing on every libc call: a modeled
    # name behind an import stub has no body by construction, and its model
    # owns what it does -- exactly as the descent treats it.
    stub = GFunc("memset", 0x401080, FSSAFunc([]), params=[])
    hook = _reader("run_hook", 0x401300, G_DIR, 0x401310)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401080), [_gptr(G_OTHER), _const(0), _const(8)])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401080: "memset",
                    0x401300: "run_hook"},
                   funcs={0x401080: stub, 0x401300: hook},
                   refs={G_DIR: [FRef(hook, 0x401310)]})
    assert _leaves(_forward(bv, func, models)) == []


# --------------------------------------------------------------------------
# round-3 review F4: the pointer is followed through copies, PHIs and
# arithmetic, in either operand order
# --------------------------------------------------------------------------

def _derived_reader(name, start, build, at=0x401210):
    """A consumer that hands `system` a pointer *build* derives from a global.

    ``build(gptr, off)`` returns the argument expression; the body always
    copies the constant into an SSA var first, so the resolution really has
    to walk the definition chain rather than read a constant operand."""
    base, off = FVar("base"), FVar("off")
    b1, o1 = FSSA(base, 1), FSSA(off, 1)
    # The call comes FIRST so the classifier has to resolve the argument,
    # rather than stopping at the address-taking copy that defines it.
    instrs = [
        FInstr(0, at, "MLIL_CALL_SSA", "system(<derived>)", reads=[b1, o1],
               dest=_gptr(SYSTEM), params=[build(_var(b1), _var(o1))]),
        FInstr(1, start, "MLIL_SET_VAR_SSA", f"base#1 = {hex(G_DIR)}",
               writes=[b1], src=_gptr(G_DIR)),
        FInstr(2, start + 4, "MLIL_SET_VAR_SSA", "off#1 = 4", writes=[o1], src=_const(4)),
    ]
    return GFunc(name, start, FSSAFunc(instrs), params=[])


@pytest.mark.parametrize("name,build", [
    ("copy", lambda g, o: g),
    ("base_plus_offset", lambda g, o: FExpr("MLIL_ADD", "base#1 + off#1", left=g, right=o)),
    ("offset_plus_base", lambda g, o: FExpr("MLIL_ADD", "off#1 + base#1", left=o, right=g)),
    ("base_minus_const", lambda g, o: FExpr("MLIL_SUB", "base#1 - 4", left=g,
                                            right=_const(4))),
    ("zero_extended", lambda g, o: FExpr("MLIL_ZX", "zx.q(base#1)", src=g)),
])
def test_a_pointer_derived_from_the_global_is_a_consumer(models, name, build):
    # Round-3 review F4: `g`, `g + off`, `off + g`, `g - 4` and a widened copy
    # all still point into `g`. Resolving only a constant operand -- or only
    # the left one -- read the consumer as write-only and cleared the run.
    func, bv = _after_copy(_derived_reader(f"reader_{name}", 0x401200, build))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", (name, leaf)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("arm", [0, 1])
def test_either_arm_of_a_phi_makes_the_consumer_read_the_global(models, arm):
    # Both incoming versions are tried, so the global reaches the call
    # whichever side of the branch selected it.
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    addrs = [_gptr(G_OTHER), _gptr(G_OTHER)]
    addrs[arm] = _gptr(G_DIR)
    body = GFunc("branchy", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_CALL_SSA", "system(p#3)", reads=[p3],
               dest=_gptr(SYSTEM), params=[_var(p3)]),
        FInstr(1, 0x401200, "MLIL_SET_VAR_SSA", "p#1 = ..", writes=[p1], src=addrs[0]),
        FInstr(2, 0x401204, "MLIL_SET_VAR_SSA", "p#2 = ..", writes=[p2], src=addrs[1]),
        FInstr(3, 0x401208, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
               writes=[p3], src=[p1, p2]),
    ]), params=[])
    eng = te.TaintEngine(FGlobalBV({SYSTEM: "system"}), models)
    ssaf = body.mlil.ssa_form
    assert eng._may_be_global(ssaf, ssaf.instructions[0].params[0], G_DIR) is True

    func, bv = _after_copy(body)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf


def test_a_phi_of_unrelated_pointers_is_not_the_global(models):
    # The control for the PHI walk: neither arm is the tainted global, so the
    # body is not a consumer of it and nothing is disclosed.
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    body = GFunc("branchy", 0x401200, FSSAFunc([
        FInstr(0, 0x401200, "MLIL_SET_VAR_SSA", "p#1 = g_other",
               writes=[p1], src=_gptr(G_OTHER)),
        FInstr(1, 0x401204, "MLIL_SET_VAR_SSA", "p#2 = g_other+8",
               writes=[p2], src=_gptr(G_OTHER + 8)),
        FInstr(2, 0x401208, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
               writes=[p3], src=[p1, p2]),
        FInstr(3, 0x401210, "MLIL_CALL_SSA", "system(p#3)", reads=[p3],
               dest=_gptr(SYSTEM), params=[_var(p3)]),
    ]), params=[])
    eng = te.TaintEngine(FGlobalBV({SYSTEM: "system"}), models)
    ssaf = body.mlil.ssa_form
    assert eng._may_be_global(ssaf, ssaf.instructions[3].params[0], G_DIR) is False


def test_a_self_referential_phi_terminates(models):
    # A loop-carried pointer phi refers to itself; the walk must converge
    # rather than recurse until the depth guard.
    p = FVar("p")
    p1, p2 = FSSA(p, 1), FSSA(p, 2)
    body = GFunc("loopy", 0x401200, FSSAFunc([
        FInstr(0, 0x401200, "MLIL_VAR_PHI", "p#2 = phi(p#1, p#2)",
               writes=[p2], src=[p1, p2]),
        FInstr(1, 0x401204, "MLIL_VAR_PHI", "p#1 = phi(p#2)", writes=[p1], src=[p2]),
    ]), params=[])
    eng = te.TaintEngine(FGlobalBV({SYSTEM: "system"}), models)
    ssaf = body.mlil.ssa_form
    assert eng._may_be_global(ssaf, p2, G_DIR) is False


# --------------------------------------------------------------------------
# round-3 review F7: a model is additive flow evidence, not a no-read
# guarantee -- the in-binary BODY is the authority
# --------------------------------------------------------------------------

def _body_helper(name, start, instrs, params):
    return GFunc(name, start, FSSAFunc(instrs), params=params)


def _calls_helper(caller_name, caller_start, helper_start, args, at=0x401210):
    return GFunc(caller_name, caller_start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"{hex(helper_start)}(..)",
               dest=_gptr(helper_start), params=list(args)),
    ]), params=[])


def test_a_partial_overlay_on_an_in_binary_body_is_not_a_write_guarantee(models):
    # The overlay says `copy_and_use` fills arg0 from arg1. Its BODY also
    # hands arg0 to system() -- which the overlay never describes. Reading
    # the model as "and it reads nothing else" suppressed a real consumer.
    dst, src = FVar("dst"), FVar("src")
    d0, s0 = FSSA(dst, 0), FSSA(src, 0)
    helper = _body_helper("copy_and_use", 0x401400, [
        FInstr(0, 0x401404, "MLIL_CALL_SSA", "system(dst#0)", reads=[d0],
               dest=_gptr(SYSTEM), params=[_var(d0)]),
        FInstr(1, 0x401408, "MLIL_CALL_SSA", "memcpy(dst#0, src#0, 16)", reads=[d0, s0],
               dest=_gptr(MEMCPY), params=[_var(d0), _var(s0), _const(16)]),
    ], params=[dst, src])
    consumer = _calls_helper("consumer", 0x401200, 0x401400,
                             [_gptr(G_DIR), _gptr(0x402000)])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", MEMCPY: "memcpy",
                    0x401200: "consumer", 0x401400: "copy_and_use"},
                   funcs={0x401200: consumer, 0x401400: helper},
                   refs={G_DIR: [FRef(consumer, 0x401210)]})

    overlay = dict(models)
    overlay["copy_and_use"] = {"propagates": [{"from": "*arg:1", "to": "*arg:0"}]}
    for catalog in (models, overlay):
        result = _forward(bv, func, catalog)
        [leaf] = _leaves(result)
        assert leaf["references"][0]["access"] == "unknown_call_arg", leaf
        assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_an_in_binary_body_that_only_writes_its_parameter_is_write_only(models):
    # The mirror: the helper's body stores through the parameter and does
    # nothing else with it, so the call really is a write destination -- the
    # verdict comes from the BODY, with no model involved at all.
    dst = FVar("dst")
    d0 = FSSA(dst, 0)
    helper = _body_helper("fill", 0x401400, [
        FInstr(0, 0x401404, "MLIL_STORE_SSA", "[dst#0].b = 0", reads=[d0],
               dest=_var(d0), src=_const(0)),
    ], params=[dst])
    consumer = _calls_helper("reset", 0x401200, 0x401400, [_gptr(G_DIR)])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system",
                    0x401200: "reset", 0x401400: "fill"},
                   funcs={0x401200: consumer, 0x401400: helper},
                   refs={G_DIR: [FRef(consumer, 0x401210)]})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_an_in_binary_body_that_reads_its_parameter_blocks(models):
    dst = FVar("dst")
    d0 = FSSA(dst, 0)
    x = FSSA(FVar("x"), 1)
    helper = _body_helper("peek", 0x401400, [
        FInstr(0, 0x401404, "MLIL_SET_VAR_SSA", "x#1 = [dst#0]", reads=[d0], writes=[x],
               src=FExpr("MLIL_LOAD_SSA", "[dst#0]", src=_var(d0))),
    ], params=[dst])
    consumer = _calls_helper("peeker", 0x401200, 0x401400, [_gptr(G_DIR)])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system",
                    0x401200: "peeker", 0x401400: "peek"},
                   funcs={0x401200: consumer, 0x401400: helper},
                   refs={G_DIR: [FRef(consumer, 0x401210)]})
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "unknown_call_arg", leaf


def test_an_unread_in_binary_helper_body_blocks_rather_than_inheriting_a_model(models):
    # The helper has a body BN could not read. A model on its name is still
    # only an overlay, so the call stays unknown instead of inheriting the
    # model's write-destination claim.
    helper = GFunc("memcpy", 0x401400, FSSAFunc([]), params=[FVar("dst")])
    helper.analysis_skipped = True
    helper.analysis_skip_reason = type("R", (), {"name": "ExceedFunctionSizeSkipReason"})()
    consumer = _calls_helper("wrapper", 0x401200, 0x401400,
                             [_gptr(G_DIR), _gptr(0x402000), _const(16)])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", 0x401200: "wrapper", 0x401400: "memcpy"},
                   funcs={0x401200: consumer, 0x401400: helper},
                   refs={G_DIR: [FRef(consumer, 0x401210)]})
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "unknown_call_arg", leaf


# --------------------------------------------------------------------------
# round-3 review F5 (remaining): a PHI-selected WRITE destination
# --------------------------------------------------------------------------

def _phi_dest_helper(name, start, callee, extra, at=0x401218):
    """`p = flag ? g : other; callee(p, ..)` -- the destination reaches the
    call only through a branch merge."""
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, start, "MLIL_SET_VAR_SSA", "p#1 = g", writes=[p1], src=_gptr(G_DIR)),
        FInstr(1, start + 4, "MLIL_SET_VAR_SSA", "p#2 = other",
               writes=[p2], src=_gptr(G_OTHER)),
        FInstr(2, start + 8, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
               writes=[p3], src=[p1, p2]),
        FInstr(3, at, "MLIL_CALL_SSA", f"{hex(callee)}(p#3, ..)", reads=[p3],
               dest=_gptr(callee), params=[_var(p3), *extra]),
    ]), params=[])


@pytest.mark.parametrize("callee,extra,name", [
    (MEMSET, [_const(0), _const(64)], "memset"),
    (MEMCPY, [_gptr(0x402000), _const(5)], "memcpy"),
])
def test_a_phi_selected_write_destination_is_not_a_consumer(models, callee, extra, name):
    # Round-3 review F5: the `lea` that materialises the destination is not
    # an access. Following the pointer to its TERMINAL use shows the only
    # thing done with it is the modeled write, so nothing is disclosed.
    helper = _phi_dest_helper("phi_reset", 0x401200, callee, extra)
    func, bv = _after_copy(helper)
    bv._names[callee] = name
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_phi_selected_pointer_with_one_reading_use_still_blocks(models):
    # Mixed arms: the same merged pointer is written through once and read
    # through once. Any read among the terminal uses blocks.
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    helper = GFunc("phi_mixed", 0x401200, FSSAFunc([
        FInstr(0, 0x401200, "MLIL_SET_VAR_SSA", "p#1 = g", writes=[p1], src=_gptr(G_DIR)),
        FInstr(1, 0x401204, "MLIL_SET_VAR_SSA", "p#2 = other",
               writes=[p2], src=_gptr(G_OTHER)),
        FInstr(2, 0x401208, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
               writes=[p3], src=[p1, p2]),
        FInstr(3, 0x401218, "MLIL_CALL_SSA", "memset(p#3, 0, 64)", reads=[p3],
               dest=_gptr(MEMSET), params=[_var(p3), _const(0), _const(64)]),
        FInstr(4, 0x40121c, "MLIL_CALL_SSA", "system(p#3)", reads=[p3],
               dest=_gptr(SYSTEM), params=[_var(p3)]),
    ]), params=[])
    func, bv = _after_copy(helper)
    bv._names[MEMSET] = "memset"
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x40121c), leaf


# --------------------------------------------------------------------------
# round-3 review F8: an interior reference is a reference to the object
# --------------------------------------------------------------------------

class FDataVar:
    def __init__(self, address, width):
        self.address = address
        self.type = type("T", (), {"width": width})()


class FExtentBV(FGlobalBV):
    """``FGlobalBV`` plus data variables, so a global has an EXTENT and an
    interior address canonicalizes back to its base.

    ``objects`` maps base -> width; a width of 0 is the STRIPPED shape (BN
    reports `void`), where the extent has to come from the layout: the next
    data variable that carries a symbol, or the section end. ``interior``
    lists bare data variables BN invented at referenced offsets, which carry
    no symbol and must not cut the object short.
    """

    def __init__(self, addr_names, funcs=None, refs=None, objects=None,
                 interior=(), section=None, **kw):
        super().__init__(addr_names, funcs, refs, **kw)
        self._objects = objects or {}
        self._interior = sorted(interior)
        self._section = section

    def get_data_var_at(self, addr):
        addr = int(addr)
        if addr in self._interior:
            return FDataVar(addr, 0)
        for base, width in self._objects.items():
            if width and base <= addr < base + width:
                return FDataVar(base, width)
            if not width and base == addr:
                return FDataVar(base, 0)
        return None

    def get_next_data_var_after(self, addr):
        starts = sorted(set(self._objects) | set(self._interior))
        for a in starts:
            if a > int(addr):
                return FDataVar(a, self._objects.get(a, 0))
        return None

    def get_symbol_at(self, addr):
        if int(addr) in self._interior or int(addr) not in self._objects:
            return None
        return type("Sym", (), {"name": f"obj_{int(addr):x}", "address": int(addr)})()

    def get_sections_at(self, addr):
        if self._section and self._section[0] <= int(addr) < self._section[1]:
            return [type("S", (), {"start": self._section[0],
                                   "end": self._section[1]})()]
        return []

    def get_code_refs(self, addr, length=None):
        lo, hi = int(addr), int(addr) + (int(length) if length else 1)
        out = []
        for ref_addr, rows in self._refs.items():
            if lo <= int(ref_addr) < hi:
                out.extend(rows)
        return out


def _interior_reader(name, start, addr, at):
    """`system(g + 4)` -- the constant pointer names the object's interior."""
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"system({hex(addr)})",
               dest=_gptr(SYSTEM), params=[_gptr(addr)]),
    ]), params=[])


def _interior_program(consumer, refs, objects, **kw):
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(consumer.start)), [])],
        params_extra=[_const(0xff)])
    bv = FExtentBV({STRNCPY: "strncpy", SYSTEM: "system",
                    int(consumer.start): consumer.name},
                   funcs={int(consumer.start): consumer},
                   refs=refs, objects=objects, **kw)
    return func, bv


def test_a_consumer_of_an_interior_address_blocks_with_its_offset(models):
    # Round-3 review F8: the copy fills g[0..63] and the consumer reads
    # `g + 4`. BN records that code reference at the INTERIOR address, so
    # asking only about the base found nothing and cleared the run.
    consumer = _interior_reader("offset_consumer", 0x401200, G_DIR + 4, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR + 4: [FRef(consumer, 0x401210)]},
        {G_DIR: 64, G_OTHER: 64})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G_DIR), leaf
    assert leaf["references"][0]["offset"] == 4, leaf
    assert leaf["references"][0]["at"] == hex(0x401210), leaf
    assert hex(4) in leaf["detail"], leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_consumer_of_the_base_address_reports_offset_zero(models):
    consumer = _interior_reader("base_consumer", 0x401200, G_DIR, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR: [FRef(consumer, 0x401210)]}, {G_DIR: 64, G_OTHER: 64})
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["offset"] == 0, leaf


def test_a_disjoint_adjacent_object_is_not_the_tainted_global(models):
    # The neighbouring object starts one byte past the end of the tainted
    # one. Sweeping refs by extent must not reach into it.
    adjacent = G_DIR + 64
    consumer = _interior_reader("neighbour_consumer", 0x401200, adjacent + 8, 0x401210)
    func, bv = _interior_program(
        consumer, {adjacent + 8: [FRef(consumer, 0x401210)]},
        {G_DIR: 64, adjacent: 64})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# round-3 review F9: an unread consumer behind a wrapper, at any depth
# --------------------------------------------------------------------------

def _skipped(name, start):
    fn = GFunc(name, start, FSSAFunc([]), params=[])
    fn.analysis_skipped = True
    fn.analysis_skip_reason = type("R", (), {"name": "ExceedFunctionSizeSkipReason"})()
    return fn


@pytest.mark.parametrize("hops", [1, 2])
def test_an_unread_consumer_behind_a_wrapper_is_disclosed(models, hops):
    # Round-3 review F9: the immediate callee is an ordinary analysed
    # wrapper; the body nothing could read sits one or two calls further on.
    # Walking THROUGH it and answering "no reader" is the same hole as
    # calling it directly.
    unread_fn = _skipped("huge_reader", 0x401500)
    chain = [unread_fn]
    for i in range(hops):
        chain.append(_quiet(f"wrapper{i}", 0x401300 + i * 0x20, callees=[chain[-1]]))
    head = chain[-1]
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(head.start)), [])], params_extra=[_const(0xff)])
    names = {STRNCPY: "strncpy", SYSTEM: "system"}
    funcs = {}
    for fn in chain:
        names[int(fn.start)] = fn.name
        funcs[int(fn.start)] = fn
    hook = _reader("other_reader", 0x401600, G_DIR, 0x401610)
    names[0x401600] = "other_reader"
    funcs[0x401600] = hook
    bv = FGlobalBV(names, funcs=funcs, refs={G_DIR: [FRef(hook, 0x401610)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] == "unread_callee_body", leaf
    assert [b["name"] for b in leaf["reachability"]["unread_bodies"]] == ["huge_reader"]
    assert "huge_reader" in leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert result["stats"]["analysis_incomplete"] is True
    assert "huge_reader" in result["stats"]["analysis_incomplete_functions"]


def test_a_wrapper_over_fully_analysed_callees_stays_silent(models):
    # The control: every descendant was read and none of them touches the
    # tainted global, so the answer stands.
    leafy = _quiet("leafy", 0x401500)
    wrapper = _quiet("wrapper", 0x401300, callees=[leafy])
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401300), [])], params_extra=[_const(0xff)])
    hook = _reader("other_reader", 0x401600, G_DIR, 0x401610)
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401300: "wrapper",
                    0x401500: "leafy", 0x401600: "other_reader"},
                   funcs={0x401300: wrapper, 0x401500: leafy, 0x401600: hook},
                   refs={G_DIR: [FRef(hook, 0x401610)]})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["stats"]["analysis_incomplete"] is False


# --------------------------------------------------------------------------
# round-3 review I1: an UNTYPED global still has an extent
# --------------------------------------------------------------------------

def test_an_interior_reader_of_an_untyped_global_blocks(models):
    # Round-3 review I1: a stripped binary gives BN no type for the object
    # (`void`, width 0), and BN then invents a bare data variable at the very
    # offset the consumer references. Taking the type's word for it made the
    # extent one byte -- so the reference sweep missed the consumer and the
    # interior pointer folded to a different "object" -- and the run cleared.
    consumer = _interior_reader("offset_consumer", 0x401200, G_DIR + 4, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR + 4: [FRef(consumer, 0x401210)]},
        {G_DIR: 0, G_DIR + 64: 0},            # both untyped
        interior=[G_DIR + 4], section=(G_DIR - 0x20, G_DIR + 0x80))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G_DIR), leaf
    assert leaf["references"][0]["offset"] == 4, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_an_untyped_neighbour_object_is_still_disjoint(models):
    # The layout bound stops at the next data variable that carries a SYMBOL,
    # which is exactly where the neighbouring object begins.
    neighbour = G_DIR + 64
    consumer = _interior_reader("neighbour_consumer", 0x401200, neighbour + 8, 0x401210)
    func, bv = _interior_program(
        consumer, {neighbour + 8: [FRef(consumer, 0x401210)]},
        {G_DIR: 0, neighbour: 0},
        interior=[neighbour + 8], section=(G_DIR - 0x20, G_DIR + 0x80))
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# round-3 review I2: a callback behind a wrapper
# --------------------------------------------------------------------------

def _dispatcher(name, start, call_at=0x401310):
    """`int dispatch(cb) { return cb(); }` -- BN lists no callees for it."""
    cb = FVar("cb")
    cb0 = FSSA(cb, 0)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, call_at, "MLIL_CALL_SSA", "cb#0()", reads=[cb0],
               dest=_var(cb0), params=[]),
    ]), params=[cb])


def test_a_callback_behind_a_wrapper_is_a_boundary(models):
    # Round-3 review I2: the reachability walk follows `Function.callees`,
    # which lists resolved edges only -- so a dispatcher whose whole job is
    # `cb()` looked like a leaf and the walk answered "nothing reads it".
    reader = _reader("consumer", 0x401500, G_DIR, 0x401510)
    dispatch = _dispatcher("dispatch", 0x401300)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401300), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system",
                    0x401300: "dispatch", 0x401500: "consumer"},
                   funcs={0x401300: dispatch, 0x401500: reader},
                   refs={G_DIR: [FRef(reader, 0x401510)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] == "unresolved_call_target", leaf
    assert leaf["reachability"]["unresolved_calls"] == [hex(0x401310)], leaf
    assert hex(0x401310) in leaf["detail"], leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # --resolve-map pins the dispatcher's callsite: the walk follows it and
    # reports the real consumer instead of the unresolved boundary.
    engine = te.TaintEngine(bv, models,
                            resolve_map={hex(0x401310): [hex(0x401500)]})
    pinned = engine.forward(func, [te.parse_locator("param:0")])
    [leaf] = _leaves(pinned)
    assert leaf["reachability"]["cause"] is None, leaf
    assert leaf["references"][0]["display_name"] == "consumer", leaf


def test_a_wrapper_with_no_indirect_call_is_not_a_boundary(models):
    # The control: an ordinary analysed wrapper with only resolved edges
    # leaves the answer standing.
    quiet = _quiet("wrapper", 0x401300, callees=[_quiet("leafy", 0x401500)])
    reader = _reader("consumer", 0x401600, G_DIR, 0x401610)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401300), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401300: "wrapper",
                    0x401600: "consumer"},
                   funcs={0x401300: quiet, 0x401600: reader},
                   refs={G_DIR: [FRef(reader, 0x401610)]})
    assert _leaves(_forward(bv, func, models)) == []


# --------------------------------------------------------------------------
# round-3 review I3/I4: returned destination aliases and append destinations
# --------------------------------------------------------------------------

def test_a_returned_destination_alias_is_the_same_object(models):
    # Round-3 review I3: `p = memcpy(g, "x", 1); system(p + 1)`. The model
    # says `arg:0 -> ret`, so the result IS the destination and `p + 1`
    # still points into the tainted bytes -- the copy overwrote only byte 0.
    p = FVar("p")
    p1 = FSSA(p, 1)
    consumer = GFunc("alias_consumer", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_CALL_SSA", "p#1 = memcpy(g, 'x', 1)", writes=[p1],
               dest=_gptr(MEMCPY),
               params=[_gptr(G_DIR), _gptr(0x402000), _const(1)]),
        FInstr(1, 0x401218, "MLIL_CALL_SSA", "system(p#1 + 1)", reads=[p1],
               dest=_gptr(SYSTEM),
               params=[FExpr("MLIL_ADD", "p#1 + 1", left=_var(p1), right=_const(1))]),
    ]), params=[])
    func, bv = _after_copy(consumer)
    bv._names[MEMCPY] = "memcpy"
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401218), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_an_append_destination_is_a_reader(models):
    # Round-3 review I4: strcat walks the destination's existing bytes to
    # find the terminator, so its destination is never write-only. The
    # model says so with `reads_dest`.
    consumer = _modeled_call("concat_consumer", 0x401200, STRCAT,
                             [_gptr(G_DIR), _gptr(0x402000)])
    func, bv = _after_copy(consumer)
    bv._names[STRCAT] = "strcat"
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and the model carries the claim, so a user overlay can state it too
    assert te.TaintEngine._model_arg_role(models["strcat"], 0) == "read"
    assert te.TaintEngine._model_arg_role(models["memset"], 0) == "write"


def test_a_true_reset_destination_stays_write_only(models):
    # The control for I4: memset genuinely writes without reading, so a
    # helper whose only use of the global is `memset(g, 0, 64)` is silent.
    consumer = _modeled_call("reset_consumer", 0x401200, MEMSET,
                             [_gptr(G_DIR), _const(0), _const(64)])
    func, bv = _after_copy(consumer)
    bv._names[MEMSET] = "memset"
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
