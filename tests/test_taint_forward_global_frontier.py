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
STRCAT_CHK = 0x4010a0


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
    assert leaf["global"] == {"address": hex(G_DIR), "label": f"glob_{hex(G_DIR)}",
                              "written_offsets": []}
    assert leaf["callee"]["address"] == hex(hook.start)
    assert leaf["references"] == [{"name": "run_hook", "display_name": "run_hook",
                                   "address": hex(hook.start), "at": hex(0x401210),
                                   "access": "call_arg", "offset": 0,
                                   "offset_known": True,
                                   "seed": "arg:system:0"}]
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
        "offset": 0, "offset_known": True, "seed": "arg:system:0"}



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
                                   "access": "unreadable", "offset": 0,
                                   "offset_known": True, "seed": None}], leaf
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

    def get_previous_data_var_before(self, addr):
        starts = sorted(set(self._objects) | set(self._interior))
        for a in reversed(starts):
            if a < int(addr):
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


def _interior_program(consumer, refs, objects, copy_len=0x20, **kw):
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(consumer.start)), [])],
        params_extra=[_const(copy_len)])
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


# --------------------------------------------------------------------------
# round-3 review F11: a recovered type is a LOWER bound on the object
# --------------------------------------------------------------------------

def test_an_inferred_string_width_does_not_shrink_the_object(models):
    # `char g[64] = "seed"` with no debug info comes back `char[0x5]`: BN
    # sized it from the bytes it could see. Taking that as the object's
    # extent stopped five bytes in, so a consumer reading `g + 8` was never
    # discovered and the run cleared (round-3 review F11). The layout bound
    # -- to the next data variable that carries a symbol -- is the wider and
    # therefore the answer.
    consumer = _interior_reader("interior_reader", 0x401200, G_DIR + 8, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR + 8: [FRef(consumer, 0x401210)]},
        {G_DIR: 5, G_DIR + 64: 5},               # BN's guess from the literal
        interior=[G_DIR + 8], section=(G_DIR - 0x20, G_DIR + 0x80))
    assert te.TaintEngine(bv, models)._global_extent(G_DIR) == 64
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["offset"] == 8, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_the_neighbour_still_bounds_an_under_typed_object(models):
    # The same under-typed pair: widening stops where the symbolled
    # neighbour begins, so its reader is still a different object.
    neighbour = G_DIR + 64
    consumer = _interior_reader("neighbour_reader", 0x401200, neighbour + 8, 0x401210)
    func, bv = _interior_program(
        consumer, {neighbour + 8: [FRef(consumer, 0x401210)]},
        {G_DIR: 5, neighbour: 5},
        interior=[neighbour + 8], section=(G_DIR - 0x20, G_DIR + 0x80))
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_type_wider_than_the_layout_keeps_its_own_width(models):
    # The other direction: the recovered type is a lower bound, so a typed
    # object wider than the next symbol is not cut down to it.
    bv = FExtentBV({}, objects={G_DIR: 256, G_DIR + 64: 8},
                   section=(G_DIR, G_DIR + 0x80))
    assert te.TaintEngine(bv, te.load_models())._global_extent(G_DIR) == 256


# --------------------------------------------------------------------------
# round-3 review F14: fortified and bounds-checked appends read too
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "strcat", "strncat", "strlcat",
    "strcat_chk", "strncat_chk", "strlcat_chk",
    "strcat_s", "strncat_s", "g_strlcat",
])
def test_every_append_spelling_reads_its_destination(models, name):
    # A fortified (`__*_chk`) or bounds-checked (`*_s`) wrapper changes WHERE
    # an append stops, never that it walks the destination's existing bytes
    # first -- so none of these destinations is write-only.
    assert te.TaintEngine._model_arg_role(models[name], 0) == "read", name


@pytest.mark.parametrize("name", ["memset", "memset_chk", "strcpy_chk", "strlcpy_chk"])
def test_a_plain_write_destination_stays_write_only(models, name):
    assert te.TaintEngine._model_arg_role(models[name], 0) == "write", name


def test_a_fortified_append_consumer_blocks(models):
    # End to end: the helper's only use of the tainted global is
    # `__strcat_chk(g, "suffix", 64)`, and that reads g before it appends.
    consumer = _modeled_call("fort_append_consumer", 0x401200, STRCAT_CHK,
                             [_gptr(G_DIR), _gptr(0x402000), _const(64)])
    func, bv = _after_copy(consumer)
    bv._names[STRCAT_CHK] = "__strcat_chk"
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# round-3 review F15: no silent cap on how much of an object is swept
# --------------------------------------------------------------------------

BIG = 1 << 17          # 128 KiB, past the old 64 KiB cap
FAR = 100000           # the offset the consumer reads


def test_a_reader_past_the_old_cap_is_found(models):
    # The sweep used to clip every object to 64 KiB, so a reader of
    # `g + 100000` in a 128 KiB buffer was never enumerated and the run
    # cleared (round-3 review F15). The object's own extent governs now.
    consumer = _interior_reader("large_reader", 0x401200, G_DIR + FAR, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR + FAR: [FRef(consumer, 0x401210)]},
        {G_DIR: BIG}, section=(G_DIR, G_DIR + BIG))
    eng = te.TaintEngine(bv, models)
    assert eng._global_extent(G_DIR) == BIG
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["offset"] == FAR, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_the_sweep_asks_in_range_chunks_not_per_byte(models):
    # Range-aware queries, one per chunk, never one per byte.
    asked: list[tuple] = []

    class CountingBV(FExtentBV):
        def get_code_refs(self, addr, length=None):
            asked.append((int(addr), int(length or 1)))
            return super().get_code_refs(addr, length)

    consumer = _interior_reader("large_reader", 0x401200, G_DIR + FAR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = CountingBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "large_reader"},
                    funcs={0x401200: consumer},
                    refs={G_DIR + FAR: [FRef(consumer, 0x401210)]},
                    objects={G_DIR: BIG}, section=(G_DIR, G_DIR + BIG))
    assert _leaves(_forward(bv, func, models)), "the far reader must be found"
    assert asked, asked
    assert len(asked) <= 1 + BIG // te.TaintEngine._GLOBAL_REF_CHUNK, asked
    assert all(span > 1 for _at, span in asked), asked


def test_an_object_past_the_sweep_budget_is_disclosed(models, monkeypatch):
    # An object too large for the chunk budget leaves "these are all the
    # references" unestablished -- a truncated sweep, never an all-clear.
    monkeypatch.setattr(te.TaintEngine, "_GLOBAL_REF_CHUNKS", 1)
    consumer = _interior_reader("far_reader", 0x401200, G_DIR + FAR, 0x401210)
    func, bv = _interior_program(
        consumer, {G_DIR + FAR: [FRef(consumer, 0x401210)]},
        {G_DIR: BIG}, section=(G_DIR, G_DIR + BIG))
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] == "reference_sweep_budget", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# round-3 review F16: a veneer onto an unread body
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hops", [1, 2])
def test_a_veneer_onto_an_unread_body_is_disclosed(models, hops):
    # The veneer resolves, but its target is an in-binary body BN measured
    # and skipped. Calling the resolved target "external" lost both the
    # boundary and the analysis-incomplete signal the SAME body gets when it
    # is called directly (round-3 review F16).
    unread_fn = _skipped("huge_reader", 0x401500)
    funcs = {0x401500: unread_fn}
    names = {STRNCPY: "strncpy", SYSTEM: "system", 0x401500: "huge_reader"}
    nxt = 0x401500
    for i in range(hops):
        start = 0x401300 + i * 0x20
        veneer = _thunk(f"veneer{i}", start, nxt)
        funcs[start] = veneer
        names[start] = veneer.name
        nxt = start
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(nxt), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV(names, funcs=funcs, refs={G_DIR: [FRef(unread_fn, 0x401510)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["callee"]["display_name"] == "huge_reader", leaf
    assert leaf["reachability"]["cause"] == "unread_callee_body", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert result["stats"]["analysis_incomplete"] is True
    assert "huge_reader" in result["stats"]["analysis_incomplete_functions"]


# --------------------------------------------------------------------------
# round-3 review F17: returning a pointer is not reading its bytes
# --------------------------------------------------------------------------

def _ret_helper(name, start, at=0x401210):
    """`char *reset_ret(void) { return memset(g, 0, 64); }`."""
    r = FSSA(FVar("r"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", "r#1 = memset(g, 0, 64)", writes=[r],
               dest=_gptr(MEMSET), params=[_gptr(G_DIR), _const(0), _const(64)]),
        FInstr(1, at + 8, "MLIL_RET", "return r#1", reads=[r], src=[_var(r)]),
    ]), params=[])


def _ignores_result(name, start, helper_start, at=0x401310):
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"{hex(helper_start)}()",
               dest=_gptr(helper_start), params=[]),
        FInstr(1, at + 8, "MLIL_RET", "return 0", src=[_const(0)]),
    ]), params=[])


def _uses_result(name, start, helper_start, at=0x401310, use_at=0x401318, phi=False):
    p = FVar("p")
    p1, p2, p3 = FSSA(p, 1), FSSA(p, 2), FSSA(p, 3)
    instrs = [
        FInstr(0, at, "MLIL_CALL_SSA", f"p#1 = {hex(helper_start)}()", writes=[p1],
               dest=_gptr(helper_start), params=[]),
    ]
    if phi:
        instrs += [
            FInstr(1, at + 4, "MLIL_SET_VAR_SSA", "p#2 = other",
                   writes=[p2], src=_gptr(G_OTHER)),
            FInstr(2, at + 8, "MLIL_VAR_PHI", "p#3 = phi(p#1, p#2)",
                   writes=[p3], src=[p1, p2]),
            FInstr(3, use_at, "MLIL_CALL_SSA", "system(p#3)", reads=[p3],
                   dest=_gptr(SYSTEM), params=[_var(p3)]),
        ]
    else:
        instrs.append(
            FInstr(1, use_at, "MLIL_CALL_SSA", "system(p#1)", reads=[p1],
                   dest=_gptr(SYSTEM), params=[_var(p1)]))
    return GFunc(name, start, FSSAFunc(instrs), params=[])


def _ret_program(caller, helper):
    caller.callees = [helper]
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(caller.start)), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", MEMSET: "memset",
                    int(caller.start): caller.name, int(helper.start): helper.name},
                   funcs={int(caller.start): caller, int(helper.start): helper},
                   refs={G_DIR: [FRef(helper, 0x401210)]})
    return func, bv


def test_a_write_only_helper_whose_return_is_ignored_is_silent(models):
    # Round-3 review F17: BN recovers `reset` as `return memset(g, 0, 64)`,
    # but returning the POINTER is not reading the bytes -- and the caller
    # discards it. The old classifier called the RET a read and raised a
    # blocking frontier on a helper that only writes.
    helper = _ret_helper("reset_ret", 0x401200)
    caller = _ignores_result("reset_ignore", 0x401300, 0x401200)
    func, bv = _ret_program(caller, helper)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    assert te.TaintEngine(bv, models)._global_access(helper, G_DIR)[1] == "write_only"


@pytest.mark.parametrize("phi", [False, True], ids=["assigned", "phi"])
def test_a_returned_pointer_the_caller_consumes_blocks_at_the_consumer(models, phi):
    # ...and when the caller DOES consume the returned pointer -- directly or
    # through a branch merge -- the boundary is reported at that consumer,
    # not at the helper that merely handed the address back.
    helper = _ret_helper("reset_ret", 0x401200)
    caller = _uses_result("use_reset_ret", 0x401300, 0x401200, phi=phi)
    func, bv = _ret_program(caller, helper)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["display_name"] == "use_reset_ret", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401318), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_returning_the_global_itself_is_still_an_escape(models):
    # The control for the RET rule: a helper that returns the ADDRESS (not a
    # call's residual alias) still escapes it, and that is disclosed.
    helper = GFunc("expose", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_RET", "return g", src=[_gptr(G_DIR)]),
    ]), params=[])
    caller = _ignores_result("expose_caller", 0x401300, 0x401200)
    func, bv = _ret_program(caller, helper)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "return", leaf
    assert "may read" in leaf["detail"], leaf["detail"]


# --------------------------------------------------------------------------
# round-3 review F18/F19: provenance through chains of returning wrappers
# --------------------------------------------------------------------------

READ = 0x4010b0


def _relay(name, start, inner_start, at=0x401210):
    """`char *relay(void) { return inner(); }` -- it names no global itself."""
    r = FSSA(FVar("r"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"r#1 = {hex(inner_start)}()", writes=[r],
               dest=_gptr(inner_start), params=[]),
        FInstr(1, at + 8, "MLIL_RET", "return r#1", reads=[r], src=[_var(r)]),
    ]), params=[])


def _arg_reset(name, start, at=0x401210):
    """`char *reset_arg(char *p) { return memset(p, 0, 1); }`."""
    p = FVar("p")
    p0, r = FSSA(p, 0), FSSA(FVar("r"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", "r#1 = memset(p#0, 0, 1)", reads=[p0], writes=[r],
               dest=_gptr(MEMSET), params=[_var(p0), _const(0), _const(1)]),
        FInstr(1, at + 8, "MLIL_RET", "return r#1", reads=[r], src=[_var(r)]),
    ]), params=[p])


def _reads_offset(name, start, helper_start, args=(), at=0x401310, use_at=0x401318):
    """`system(helper(..) + 8)` -- the read is at an offset the reset missed."""
    q = FSSA(FVar("q"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"q#1 = {hex(helper_start)}(..)", writes=[q],
               dest=_gptr(helper_start), params=list(args)),
        FInstr(1, use_at, "MLIL_CALL_SSA", "system(q#1 + 8)", reads=[q],
               dest=_gptr(SYSTEM),
               params=[FExpr("MLIL_ADD", "q#1 + 8", left=_var(q), right=_const(8))]),
    ]), params=[])


def _chain_program(chain, consumer, refs_from):
    """`main` copies into G_DIR, then calls *consumer*; *chain* are the
    helpers it reaches. Only *refs_from* names the global."""
    names = {STRNCPY: "strncpy", SYSTEM: "system", MEMSET: "memset", READ: "read"}
    funcs = {}
    for fn in [*chain, consumer]:
        names[int(fn.start)] = fn.name
        funcs[int(fn.start)] = fn
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(int(consumer.start)), [])],
        params_extra=[_const(0xff)])
    bv = FGlobalBV(names, funcs=funcs,
                   refs={G_DIR: [FRef(refs_from, 0x401210)]})
    return func, bv


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_a_chain_of_returning_wrappers_keeps_the_provenance(models, hops):
    # Round-3 review F18: `reset_ret` returns `memset(g, 0, 1)`, a relay
    # returns that, and the consumer reads `+ 8` -- bytes the one-byte reset
    # never touched. Only `reset_ret` names the global, so a walk that
    # classified just the naming functions reported a clean run.
    inner = _ret_helper("reset_ret", 0x401200)
    chain = [inner]
    nxt = 0x401200
    for i in range(hops - 1):
        relay = _relay(f"relay{i}", 0x401400 + i * 0x20, nxt)
        chain.append(relay)
        nxt = int(relay.start)
    consumer = _reads_offset("use_ret", 0x401300, nxt)
    consumer.callees = [chain[-1]]
    for i in range(len(chain) - 1, 0, -1):
        chain[i].callees = [chain[i - 1]]
    func, bv = _chain_program(chain, consumer, inner)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["display_name"] == "use_ret", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401318), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_helper_returning_its_own_parameter_remaps_to_the_actual(models):
    # Round-3 review F19: `reset_arg(char *p) { return memset(p, 0, 1); }`
    # names no global -- inside it the object is a FORMAL. The caller passes
    # the global and reads `+ 8` through the result, so the actual has to be
    # mapped onto the formal and back.
    helper = _arg_reset("reset_arg", 0x401200)
    consumer = _reads_offset("use_arg_ret", 0x401300, 0x401200, args=[_gptr(G_DIR)])
    consumer.callees = [helper]
    func, bv = _chain_program([helper], consumer, consumer)
    eng = te.TaintEngine(bv, models)
    assert eng._fn_returned_params(helper) == (0,)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401318), leaf


def test_a_chain_whose_result_is_ignored_stays_silent(models):
    # The control: the same chain, but nothing consumes what comes back.
    inner = _ret_helper("reset_ret", 0x401200)
    relay = _relay("relay", 0x401400, 0x401200)
    relay.callees = [inner]
    consumer = _ignores_result("ignore_ret", 0x401300, 0x401400)
    consumer.callees = [relay]
    func, bv = _chain_program([inner, relay], consumer, inner)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_partial_reset_does_not_sanitize_the_whole_object(models):
    # A one-byte `memset` is a write to byte 0, not a proof that the object
    # holds nothing: the read at `+8` is still a read of tainted bytes.
    inner = _ret_helper("reset_ret", 0x401200)
    consumer = _reads_offset("use_ret", 0x401300, 0x401200)
    consumer.callees = [inner]
    func, bv = _chain_program([inner], consumer, inner)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf


def test_the_scan_budget_is_disclosed_not_silently_cleared(models, monkeypatch):
    # The walk's node pool is per function scan; running out must leave an
    # honest "could not finish classifying", never a clean answer.
    monkeypatch.setattr(te.TaintEngine, "_PTR_SCAN_NODES", 0)
    func, hook, bv = _globalflow(extra_copy_params=[_const(0xff)])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "scan_budget_exhausted", leaf
    # BN21-BOUND-02: the bound is named, the search is NOT called complete,
    # and the run's aggregate statistics carry it.
    assert leaf["reachability"]["status"] == "truncated", leaf
    assert leaf["reachability"]["cause"] == "pointer_scan_budget", leaf
    assert result["stats"]["truncated"] is True
    assert "pointer_scan_budget" in result["stats"]["truncation_cause"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# round-3 review F20: a source model's output buffer is a WRITE
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,idx,role", [
    ("read", 1, "write"), ("recv", 1, "write"), ("fgets", 0, "write"),
    ("memset", 0, "write"), ("system", 0, "read"), ("strcat", 0, "read"),
    ("memcpy", 1, "read"),
])
def test_model_argument_roles(models, name, idx, role):
    assert te.TaintEngine._model_arg_role(models[name], idx) == role, (name, idx)


def test_a_read_output_destination_is_not_a_consumer(models):
    # Round-3 review F20: `read(0, g, 64)` FILLS g. The role lives in the
    # model's `sources`, which the role test never consulted, so the call
    # came back "unknown" and raised a blocking frontier on a refill.
    helper = _modeled_call("refill", 0x401200, READ,
                           [_const(0), _gptr(G_DIR), _const(64)])
    func, bv = _after_copy(helper)
    bv._names[READ] = "read"
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_read_followed_by_a_use_of_the_global_still_blocks(models):
    # ...and a refill followed by a genuine read of the bytes is a consumer.
    helper = GFunc("refill_then_read", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_CALL_SSA", "read(0, g, 64)",
               dest=_gptr(READ), params=[_const(0), _gptr(G_DIR), _const(64)]),
        FInstr(1, 0x401218, "MLIL_CALL_SSA", "system(g)",
               dest=_gptr(SYSTEM), params=[_gptr(G_DIR)]),
    ]), params=[])
    func, bv = _after_copy(helper)
    bv._names[READ] = "read"
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401218), leaf


def test_an_in_binary_body_still_outranks_a_source_model(models):
    # F7 stands: a model never speaks for a callee this binary has a body
    # for. A local `read`-named body that hands the buffer to system is a
    # consumer, whatever its name's model says.
    body = GFunc("read", 0x401400, FSSAFunc([
        FInstr(0, 0x401410, "MLIL_CALL_SSA", "system(p#0)",
               reads=[FSSA(FVar("p"), 0)], dest=_gptr(SYSTEM),
               params=[_var(FSSA(FVar("p"), 0))]),
    ]), params=[FVar("fd"), FVar("p"), FVar("n")])
    helper = _modeled_call("refill", 0x401200, 0x401400,
                           [_const(0), _gptr(G_DIR), _const(64)])
    func, bv = _after_copy(helper)
    bv._names[0x401400] = "read"
    bv._funcs[0x401400] = body
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["access"] == "unknown_call_arg", leaf


# --------------------------------------------------------------------------
# round-3 review R1: an INTERIOR destination is the same object
# --------------------------------------------------------------------------

def _copy_into(dest_addr, consumer_start, *, copy_at=0x401104, call_at=0x401108):
    """`main(argv)`: `memcpy(dest_addr, argv, 64)` then `consumer()`."""
    return _copy_then_call(copy_at, MEMCPY, _gptr(dest_addr),
                           call_sites=[(call_at, _gptr(consumer_start), [])],
                           params_extra=[_const(0x40)])


def _interior_dest_program(dest_addr, *, objects, interior, refs_at=G_DIR):
    reader = _reader("reader", 0x401200, refs_at, 0x401210)
    func, _argv = _copy_into(dest_addr, 0x401200)
    bv = FExtentBV({STRNCPY: "strncpy", MEMCPY: "memcpy", SYSTEM: "system",
                    0x401200: "reader"},
                   funcs={0x401200: reader},
                   refs={refs_at: [FRef(reader, 0x401210)]},
                   objects=objects, interior=interior,
                   section=(0x404000, 0x405000))
    return func, bv


def test_an_interior_write_is_keyed_to_the_containing_object(models):
    # Round-3 review R1: `memcpy(g + 8, s, 64)` makes BN invent a nameless
    # data variable at `g + 8`. Keying the write THERE and the reader's
    # `system(g)` at the base left two unrelated locations, and a real flow
    # answered all-clear.
    func, bv = _interior_dest_program(G_DIR + 8,
                                 objects={G_DIR: 0, G_OTHER: 0},
                                 interior=[G_DIR + 8])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["global"]["address"] == hex(G_DIR)
    # ...and WHICH bytes were written is kept: a write to part of an object
    # never reads as coverage of the whole of it.
    assert leaf["global"]["written_offsets"] == ["0x8"]
    assert leaf["references"][0]["display_name"] == "reader"
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_write_into_a_disjoint_object_is_not_folded(models):
    # The control: `G_OTHER` is its own symbolled object, so a write there is
    # NOT the object `reader` reads, whatever lies between them.
    func, bv = _interior_dest_program(G_OTHER,
                                 objects={G_DIR: 0, G_OTHER: 0},
                                 interior=[])
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_an_interior_write_past_the_object_extent_is_not_folded(models):
    # An address inside a NEIGHBOUR's extent belongs to the neighbour: the
    # fold stops where the next symbolled object begins.
    func, bv = _interior_dest_program(G_OTHER + 8,
                                 objects={G_DIR: 0, G_OTHER: 0},
                                 interior=[G_OTHER + 8])
    assert _leaves(_forward(bv, func, models)) == []


def test_a_typed_object_folds_an_interior_address_within_its_width(models):
    func, bv = _interior_dest_program(G_DIR + 8,
                                 objects={G_DIR: 0x80, G_OTHER: 0x80},
                                 interior=[G_DIR + 8])
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["global"]["address"] == hex(G_DIR)
    assert leaf["global"]["written_offsets"] == ["0x8"]


def test_the_text_row_names_the_written_offset(models):
    func, bv = _interior_dest_program(G_DIR + 8,
                                 objects={G_DIR: 0, G_OTHER: 0},
                                 interior=[G_DIR + 8])
    [leaf] = _leaves(_forward(bv, func, models))
    row = formatters._render_leaf_line(leaf)
    assert f"global={hex(G_DIR)}(+0x8)" in row, row


# --------------------------------------------------------------------------
# round-3 review R2: a --resolve-map pin binds the returned pointer too
# --------------------------------------------------------------------------

def _indirect_consumer(name, start, call_at=0x401310, use_at=0x401318):
    """`void f(cb) { char *p = cb(); system(p + 8); }` -- the call is indirect."""
    cb = FVar("cb")
    cb0, q = FSSA(cb, 0), FSSA(FVar("q"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, call_at, "MLIL_CALL_SSA", "q#1 = cb#0()", reads=[cb0], writes=[q],
               dest=_var(cb0), params=[]),
        FInstr(1, use_at, "MLIL_CALL_SSA", "system(q#1 + 8)", reads=[q],
               dest=_gptr(SYSTEM),
               params=[FExpr("MLIL_ADD", "q#1 + 8", left=_var(q), right=_const(8))]),
    ]), params=[cb])


def _resolve_map_program():
    inner = _ret_helper("reset_ret", 0x401200)        # returns memset(g, 0, 1)
    consumer = _indirect_consumer("indirect_reader", 0x401300)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_DIR),
        call_sites=[(0x401108, _gptr(0x401300), [_gptr(0x401200)])],
        params_extra=[_const(0x40)])
    bv = FGlobalBV({MEMCPY: "memcpy", SYSTEM: "system", MEMSET: "memset",
                    0x401200: "reset_ret", 0x401300: "indirect_reader"},
                   funcs={0x401200: inner, 0x401300: consumer},
                   refs={G_DIR: [FRef(inner, 0x401210)]})
    return func, bv


def test_an_unpinned_indirect_site_is_disclosed(models):
    func, bv = _resolve_map_program()
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["reachability"]["cause"] == "unresolved_call_target", leaf


def test_a_pinned_indirect_site_keeps_the_returned_global_provenance(models):
    # Round-3 review R2: supplying --resolve-map RESOLVED the site, which
    # removed the `unresolved_call_target` disclosure -- and the alias walk
    # did not honour the same pin, so the consumer that reads `p + 8` was
    # never classified and the run went all-clear. Resolving a site must
    # never lose a boundary.
    func, bv = _resolve_map_program()
    result = te.TaintEngine(
        bv, models, resolve_map={"0x401310": ["0x401200"]}).forward(
            func, [te.parse_locator("param:0")])
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] is None, leaf
    assert leaf["references"][0]["display_name"] == "indirect_reader", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# round-3 review R3: alias inference is cycle-safe, and a recursion unwind
# never discards what the walk already observed
# --------------------------------------------------------------------------

def _self_recursive_alias(name, start):
    """`char *f(char *p, int n) { return n ? f(p, n - 1) : p; }`."""
    p, n = FVar("p"), FVar("n")
    p0, n0, r = FSSA(p, 0), FSSA(n, 0), FSSA(FVar("r"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, start, "MLIL_CALL_SSA", f"r#1 = {hex(start)}(p#0, n#0)",
               reads=[p0, n0], writes=[r], dest=_gptr(start),
               params=[_var(p0), _var(n0)]),
        FInstr(1, start + 8, "MLIL_RET", "return r#1", reads=[r], src=[_var(r)]),
        FInstr(2, start + 16, "MLIL_RET", "return p#0", reads=[p0], src=[_var(p0)]),
    ]), params=[p, n])


def test_returned_parameter_inference_terminates_on_a_cycle(models):
    # Round-3 review R3: unrolling `f(p, n - 1)` is what reached Python's
    # recursion limit. The cycle answers conservatively -- every formal may
    # come back -- and never by guessing "nothing does".
    helper = _self_recursive_alias("recursive_alias", 0x401200)
    bv = FGlobalBV({0x401200: "recursive_alias"}, funcs={0x401200: helper})
    eng = te.TaintEngine(bv, models)
    assert eng._fn_returned_params(helper) == (0, 1)


def test_a_recursive_alias_chain_is_classified_without_unwinding(models):
    helper = _self_recursive_alias("recursive_alias", 0x401200)
    consumer = _reads_offset("recursive_reader", 0x401300, 0x401200,
                             args=[_gptr(G_DIR), _const(3)])
    consumer.callees = [helper]
    func, bv = _chain_program([helper], consumer, helper)
    [leaf] = _leaves(_forward(bv, func, models))
    assert [r["display_name"] for r in leaf["references"]] == [
        "recursive_alias", "recursive_reader"], leaf


def test_a_recursion_unwind_keeps_what_was_already_found(models, monkeypatch):
    # Round-3 review R3: a sink observed BEFORE the cycle was discarded by
    # the unwind, so the run reported `reached_sinks: []` for a flow it had
    # already proven. The walk is lost; the findings are not.
    func, hook, bv = _globalflow(extra_copy_params=[_const(0xff)])
    eng = te.TaintEngine(bv, models)
    real = eng._run_forward

    def explode(fn, *a, **kw):
        real(fn, *a, **kw)
        raise RecursionError("synthetic cycle")

    monkeypatch.setattr(eng, "_run_forward", explode)
    result = eng.forward(func, [te.parse_locator("param:0")])
    assert result["stats"]["truncation_cause"] == ["recursion"]
    kinds = [lf["kind"] for lf in result["leaves"]]
    assert "analysis_truncated" in kinds, result["leaves"]
    assert KIND in kinds, result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert "recursion limit" in result["diagnostics"]["next_action"]


def test_the_truncation_leaf_blocks_the_all_clear():
    assert "analysis_truncated" in BLOCKING_LEAF_KINDS


# --------------------------------------------------------------------------
# round-3 review R4: the re-seed advice is runnable, or absent
# --------------------------------------------------------------------------

def test_the_reseed_advice_names_a_runnable_locator(models):
    # `var:VAR` was a template nothing could fill in: the consumer reads the
    # object by CONSTANT ADDRESS, so it has no variable to seed. The
    # argument it hands those bytes to does name one.
    func, hook, bv = _globalflow(extra_copy_params=[_const(0xff)])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["seed"] == "arg:system:0"
    assert leaf["reseed"] == (
        f"`bn taint forward -f {hex(hook.start)} --source arg:system:0`")
    assert "var:VAR" not in leaf["detail"]
    assert "--source arg:system:0" in leaf["detail"]
    assert "var:VAR" not in result["diagnostics"]["next_action"]
    assert "--source arg:system:0" in result["diagnostics"]["next_action"]


def test_an_unnameable_consumer_omits_the_reseed_instead_of_faking_one(models):
    # An unread body names no locator at all: the advice keeps the xref
    # audit and says to do it by hand, rather than printing a command that
    # cannot be run.
    skipped = GFunc("huge_reader", 0x401200, FSSAFunc([]), params=[])
    skipped.analysis_skipped = True
    skipped.analysis_skip_reason = type(
        "R", (), {"name": "ExceedFunctionSizeSkipReason"})()
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(0x401200), [])], params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "huge_reader"},
                   funcs={0x401200: skipped},
                   refs={G_DIR: [FRef(skipped, 0x401210)]})
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reseed"] is None, leaf
    assert "var:VAR" not in leaf["detail"]
    assert "audit the references by hand" in leaf["detail"]
    assert "var:VAR" not in result["diagnostics"]["next_action"]


# --------------------------------------------------------------------------
# BN21-OBJ-01: an alias symbol inside an untyped object is not its end
# --------------------------------------------------------------------------

G_MID = G_DIR + 64      # `.set alias_middle, alias_global + 64`


def _alias_program(copy_len, read_at, objects=None):
    """`memcpy(g, p, copy_len)` then a reader of `g + read_at`, where a SECOND
    symbol sits at `g + 64` -- the shape an alias/subobject symbol makes."""
    consumer = _interior_reader("alias_reader", 0x401200, G_DIR + read_at, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_DIR),
        call_sites=[(0x401108, _gptr(0x401200), [])],
        params_extra=[_const(copy_len)])
    bv = FExtentBV({MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "alias_reader"},
                   funcs={0x401200: consumer},
                   refs={G_DIR + read_at: [FRef(consumer, 0x401210)]},
                   objects=objects if objects is not None else {G_DIR: 0, G_MID: 0},
                   section=(0x404000, 0x405000))
    return func, bv


def test_a_copy_past_an_untyped_objects_layout_bound_is_disclosed(models):
    # BN21-OBJ-01: `alias_global` has no recovered type, so its end was taken
    # from the LAYOUT -- and `.set alias_middle, alias_global + 64` puts a
    # symbol inside it. The object was cut in half and the reader at +80 fell
    # outside, clearing a real flow. Writing 96 bytes proves otherwise.
    func, bv = _alias_program(96, 80)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["display_name"] == "alias_reader", leaf
    assert leaf["references"][0]["offset"] == 80, leaf
    # ...and the uncertainty itself is disclosed, not quietly resolved.
    assert leaf["reachability"]["cause"] == "object_extent_unproven", leaf
    assert "global_object_extent" in result["stats"]["truncation_cause"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_copy_within_the_layout_bound_keeps_the_neighbour_disjoint(models):
    # The control: the copy stays inside the object the layout describes, so
    # the next symbol is still a different object and its reader is not this
    # object's business.
    func, bv = _alias_program(32, 80)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_typed_object_is_not_second_guessed_by_an_overflowing_copy(models):
    # A recovered type IS an answer about the object's extent. A copy longer
    # than it is an overflow -- the sink machinery's business, not evidence
    # that the neighbour is part of this object.
    func, bv = _alias_program(96, 80, objects={G_DIR: 64, G_MID: 64})
    assert _leaves(_forward(bv, func, models)) == []


# --------------------------------------------------------------------------
# BN21-ALIAS-01: a receive call that hands its destination back
# --------------------------------------------------------------------------

FGETS = 0x4010c0


@pytest.mark.parametrize("name,idx", [("fgets", 0), ("gets", 0)])
def test_the_receive_family_declares_its_destination_return(models, name, idx):
    # The relation is POINTER identity, spelled the way memcpy/strcpy spell
    # it: `arg:N -> ret` carries the pointer, never the bytes.
    _key, model = te.lookup_model(models, name)
    assert {"from": f"arg:{idx}", "to": "ret"} in model["propagates"]
    assert te.TaintEngine._model_arg_role(model, idx) == "write"


def _fgets_program(consume):
    """`char *p = fgets(g, 2, stdin);` and, optionally, `system(p + 8)`."""
    q = FSSA(FVar("q"), 1)
    body = [FInstr(0, 0x401310, "MLIL_CALL_SSA", "q#1 = fgets(g, 2, stdin)",
                   writes=[q], dest=_gptr(FGETS),
                   params=[_gptr(G_DIR), _const(2), _const(0)])]
    if consume:
        body.append(FInstr(1, 0x401318, "MLIL_CALL_SSA", "system(q#1 + 8)",
                           reads=[q], dest=_gptr(SYSTEM),
                           params=[FExpr("MLIL_ADD", "q#1 + 8",
                                         left=_var(q), right=_const(8))]))
    consumer = GFunc("use_fgets", 0x401300, FSSAFunc(body), params=[])
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_DIR),
        call_sites=[(0x401108, _gptr(0x401300), [])],
        params_extra=[_const(0x40)])
    bv = FGlobalBV({MEMCPY: "memcpy", SYSTEM: "system", FGETS: "fgets",
                    0x401300: "use_fgets"},
                   funcs={0x401300: consumer},
                   refs={G_DIR: [FRef(consumer, 0x401310)]})
    return func, bv


def test_a_partial_refill_through_a_returned_destination_blocks(models):
    # BN21-ALIAS-01: `fgets(g, 2, ..)` overwrites at most two bytes and hands
    # `g` back; `system(p + 8)` reads bytes it never touched. Forgetting the
    # returned destination cleared that flow.
    func, bv = _fgets_program(consume=True)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["at"] == hex(0x401318), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_refill_whose_return_is_ignored_stays_silent(models):
    func, bv = _fgets_program(consume=False)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


# --------------------------------------------------------------------------
# BN21-REACH-01: a partial model never buys off an unread body
# --------------------------------------------------------------------------

def _unread_program(hops=0):
    """A call onto a body BN measured and skipped, optionally behind hops."""
    skipped = _skipped("opaque_stage", 0x401200)
    funcs = {0x401200: skipped}
    names = {STRNCPY: "strncpy", SYSTEM: "system", 0x401200: "opaque_stage"}
    entry = 0x401200
    for i in range(hops):
        start = 0x401400 + i * 0x20
        wrapper = GFunc(f"wrap{i}", start, FSSAFunc([
            FInstr(0, start, "MLIL_CALL_SSA", f"{hex(entry)}()",
                   dest=_gptr(entry), params=[]),
        ]), params=[], callees=[funcs[entry]])
        funcs[start] = wrapper
        names[start] = wrapper.name
        entry = start
    func, _argv = _copy_then_call(
        0x401104, STRNCPY, G_DIR,
        call_sites=[(0x401108, _gptr(entry), [])], params_extra=[_const(0x20)])
    bv = FGlobalBV(names, funcs=funcs, refs={G_DIR: [FRef(skipped, 0x401210)]})
    return func, bv


@pytest.mark.parametrize("hops", [0, 1, 2])
def test_a_partial_model_does_not_erase_an_unread_body(models, hops):
    # BN21-REACH-01: adding a TRUTHFUL partial overlay (`arg:0 -> ret`) made
    # the result LESS conservative -- the unread-body frontier and the
    # analysis-incomplete signal both disappeared. A model is additive
    # evidence over a body; it is never a substitute for reading one.
    func, bv = _unread_program(hops)
    overlaid = dict(models)
    overlaid["opaque_stage"] = {"propagates": [{"from": "arg:0", "to": "ret"}]}
    for catalog in (models, overlaid):
        result = _forward(bv, func, catalog)
        [leaf] = _leaves(result)
        assert leaf["reachability"]["cause"] == "unread_callee_body", leaf
        assert result["diagnostics"]["analysis_incomplete"] is True
        assert result["diagnostics"]["safe_to_report_all_clear"] is False


# --------------------------------------------------------------------------
# BN21-BOUND-01/02/03: every reached bound is in the aggregate statistics
# --------------------------------------------------------------------------

def test_a_truncated_reference_sweep_reaches_the_run_statistics(models,
                                                                monkeypatch):
    # BN21-BOUND-01: the leaf said `reference_sweep_budget` while
    # `stats.truncated` stayed false, so a consumer reading the aggregate saw
    # a complete run.
    monkeypatch.setattr(te.TaintEngine, "_GLOBAL_REF_CHUNKS", 0)
    func, bv = _alias_program(32, 8)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reachability"]["cause"] == "reference_sweep_budget", leaf
    assert result["stats"]["truncated"] is True
    assert "global_reference_budget" in result["stats"]["truncation_cause"]


def test_pointer_return_depth_exhaustion_is_its_own_blocking_bound(models,
                                                                   monkeypatch):
    # BN21-BOUND-03: a returning-wrapper chain longer than the INTERNAL
    # pointer-return depth answered "not an alias" and cleared the run, with
    # CLI `--max-depth 64` powerless over it.
    monkeypatch.setattr(te.TaintEngine, "_PTR_RESOLVE_DEPTH", 1)
    inner = _ret_helper("reset_ret", 0x401200)
    relay = _relay("relay", 0x401400, 0x401200)
    relay.callees = [inner]
    consumer = _reads_offset("use_ret", 0x401300, 0x401400)
    consumer.callees = [relay]
    func, bv = _chain_program([inner, relay], consumer, inner)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert any(r["access"] == "pointer_depth_exhausted"
               for r in leaf["references"]), leaf
    assert leaf["reachability"]["cause"] == "pointer_return_depth", leaf
    assert "pointer_return_depth" in result["stats"]["truncation_cause"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # ...and the advice does not promise a knob that cannot move this bound.
    assert "--max-depth" not in result["diagnostics"]["next_action"] or \
        "does not move it" in result["diagnostics"]["next_action"]


def test_an_unfinished_scan_is_not_rendered_as_a_reader(models, monkeypatch):
    # BN21-BOUND-02: "read by noisy" for a body whose classification never
    # finished states a fact the walk did not establish.
    monkeypatch.setattr(te.TaintEngine, "_PTR_SCAN_NODES", 0)
    func, hook, bv = _globalflow(extra_copy_params=[_const(0x20)])
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["reads_proven"] is False, leaf
    row = formatters._render_leaf_line(leaf)
    assert "may be read by" in row, row
    assert "(read by" not in row, row
    assert " reads (" not in result["diagnostics"]["next_action"]


# --------------------------------------------------------------------------
# round-10 review C1: a named symbol can be the INTERIOR of one object
# --------------------------------------------------------------------------

G_TAIL = G_DIR + 64     # `.set packet_tail, packet + 64`


def _spanning_reader(name, start, from_addr, length, at=0x401210):
    """`memcpy(local, from_addr, length)` -- a read whose SPAN is known."""
    local = FSSA(FVar("local"), 1)
    return GFunc(name, start, FSSAFunc([
        FInstr(0, at, "MLIL_CALL_SSA", f"memcpy(&local, {hex(from_addr)}, {length})",
               writes=[local], dest=_gptr(MEMCPY),
               params=[_var(local), _gptr(from_addr), _const(length)]),
    ]), params=[])


def _overlap_program(read_len, read_from=G_DIR, objects=None):
    """Write 32 bytes into the tail symbol, then call a reader of the base."""
    consumer = _spanning_reader("consumer", 0x401200, read_from, read_len)
    func, _argv = _copy_then_call(
        0x401104, MEMCPY, _gptr(G_TAIL),
        call_sites=[(0x401108, _gptr(0x401200), [])],
        params_extra=[_const(32)])
    bv = FExtentBV({MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "consumer"},
                   funcs={0x401200: consumer},
                   # BN records the writer's own reference to the tail
                   # symbol, exactly as it does for the real fixture.
                   refs={read_from: [FRef(consumer, 0x401210)],
                         # BN records the writer's own reference to the tail
                         # symbol, exactly as it does for the real fixture.
                         G_TAIL: [FRef(func, 0x401104)]},
                   objects=objects if objects is not None else {
                       G_DIR: 0, G_TAIL: 0, G_OTHER: 0},
                   section=(0x404000, 0x405000))
    return func, bv


def test_an_overlapping_read_of_the_enclosing_symbol_blocks(models):
    # Round-10 review C1: `.set packet_tail, packet + 64` gives one buffer
    # two symbols. A copy of `packet[0..112)` reads the bytes a write to
    # `packet_tail` put there, but it names the OTHER symbol -- so neither
    # the reference sweep nor the classifier saw it and the run cleared.
    func, bv = _overlap_program(112)
    result = _forward(bv, func, models)
    [leaf] = _leaves(result)
    assert leaf["references"][0]["display_name"] == "consumer", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["references"][0]["offset"] == 0, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def test_a_read_of_the_neighbour_that_stops_short_is_silent(models):
    # The discriminator is the access SPAN, not the adjacency: a read of the
    # first 16 bytes of the preceding symbol never reaches these bytes.
    func, bv = _overlap_program(16)
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_reader_of_an_unrelated_object_is_not_an_overlap(models):
    func, bv = _overlap_program(112, read_from=G_OTHER)
    assert _leaves(_forward(bv, func, models)) == []


# --------------------------------------------------------------------------
# round-10 review M1: the displacement through a returned pointer
# --------------------------------------------------------------------------

def test_a_returned_destination_reports_the_offset_it_is_read_at(models):
    # `p = fgets(g, 2, ..); system(p + 8)` is a read at +8. The literal scan
    # looks at the INSTRUCTION, where that displacement is not written down,
    # and reported 0 -- which reads as "the base".
    func, bv = _fgets_program(consume=True)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["offset"] == 8, leaf
    assert leaf["references"][0]["offset_known"] is True, leaf


def test_an_offset_through_a_wrapper_chain_is_resolved(models):
    # ...and through a chain of returning wrappers, where the pointer and
    # its displacement are two hops apart.
    inner = _ret_helper("reset_ret", 0x401200)
    relay = _relay("relay", 0x401400, 0x401200)
    relay.callees = [inner]
    consumer = _reads_offset("use_ret", 0x401300, 0x401400)
    consumer.callees = [relay]
    func, bv = _chain_program([inner, relay], consumer, inner)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["offset"] == 8, leaf
    assert leaf["references"][0]["offset_known"] is True, leaf


def test_an_unresolved_displacement_is_not_reported_as_the_base(models,
                                                                monkeypatch):
    # A displacement this scan cannot establish is UNKNOWN. Rendering it as
    # 0 conflates it with a read of the object's first byte.
    monkeypatch.setattr(te.TaintEngine, "_global_ptr_offset",
                        lambda *a, **k: None)
    func, bv = _fgets_program(consume=True)
    [leaf] = _leaves(_forward(bv, func, models))
    assert leaf["references"][0]["offset_known"] is False, leaf
    assert "(+?)" in formatters._render_leaf_line(leaf)


# --------------------------------------------------------------------------
# round-10 review M2: a call BEFORE the write carries nothing
# --------------------------------------------------------------------------

class FBlock:
    def __init__(self, start, end, edges=()):
        self.start, self.end = start, end
        self.outgoing_edges = [type("E", (), {"target": t})() for t in edges]


def _before_after_program(order, blocks=None):
    """`reader()` and the tainting copy, in the requested order."""
    hook = _reader("reader", 0x401200, G_DIR, 0x401210)
    argv = FVar("argv")
    argv0 = FSSA(argv, 0)
    copy = FInstr(0, 0x401104, "MLIL_CALL_SSA", "memcpy(g, argv#0, 0x40)",
                  reads=[argv0], dest=_gptr(MEMCPY),
                  params=[_gptr(G_DIR), _var(argv0), _const(0x40)])
    call = FInstr(1, 0x401108, "MLIL_CALL_SSA", "reader()",
                  dest=_gptr(0x401200), params=[])
    instrs = [copy, call] if order == "after" else [call, copy]
    for i, ins in enumerate(instrs):
        ins.instr_index = i
    ssaf = FSSAFunc(instrs)
    if blocks is not None:
        ssaf.basic_blocks = blocks
    func = GFunc("main", 0x401100, ssaf, params=[argv])
    bv = FGlobalBV({MEMCPY: "memcpy", SYSTEM: "system", 0x401200: "reader"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    return func, bv


def test_a_call_before_the_only_write_carries_nothing(models):
    # Round-10 review M2: `reader(); memcpy(g, s, 64);` has no path from the
    # write back to the reader, yet the fixpoint kept the global live across
    # passes and attached a proven-read continuation to the EARLIER call.
    func, bv = _before_after_program("before", blocks=[FBlock(0, 2)])
    result = _forward(bv, func, models, source="param:0")
    assert _leaves(result) == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_a_call_after_the_write_still_blocks(models):
    func, bv = _before_after_program("after", blocks=[FBlock(0, 2)])
    [leaf] = _leaves(_forward(bv, func, models, source="param:0"))
    assert leaf["references"][0]["display_name"] == "reader", leaf


def test_a_loop_back_edge_reaches_the_earlier_call(models):
    # The write is textually later, but the loop's back edge carries those
    # bytes round to the reader on the next iteration.
    block = FBlock(0, 2)
    block.outgoing_edges = [type("E", (), {"target": block})()]
    func, bv = _before_after_program("before", blocks=[block])
    [leaf] = _leaves(_forward(bv, func, models, source="param:0"))
    assert leaf["references"][0]["display_name"] == "reader", leaf


def test_without_a_cfg_the_walk_claims_nothing(models):
    # A view that cannot answer "is there a path" must not be read as "no".
    func, bv = _before_after_program("before")
    assert _leaves(_forward(bv, func, models, source="param:0")) != []
