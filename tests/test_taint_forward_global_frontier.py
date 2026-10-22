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
                                   "address": hex(hook.start), "at": hex(0x401210)}]
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


def test_unresolved_indirect_call_is_not_a_global_continuation(models):
    # The target does not resolve, so there is no body whose references could
    # be read: the engine's existing indirect frontier owns that call, and no
    # global boundary is invented for it.
    fp = FVar("fp")
    fp1 = FSSA(fp, 1)
    hook = _reader("run_hook", 0x401200, G_DIR, 0x401210)
    func, _argv = _copy_then_call(
        0x401104, STRCPY, G_DIR,
        call_sites=[(0x401108, _var(fp1), [])])
    bv = FGlobalBV({STRCPY: "strcpy", SYSTEM: "system", 0x401200: "run_hook"},
                   funcs={0x401200: hook}, refs={G_DIR: [FRef(hook, 0x401210)]})
    result = _forward(bv, func, models)
    assert _leaves(result) == [], result["leaves"]


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

    # ... and the same program with no descent budget left reaches nothing, so
    # the frontier is not claimed beyond what the walk itself would cover.
    capped = _forward(bv, func, models, max_depth=0)
    assert _leaves(capped) == [], capped["leaves"]


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
                        "address": "0x401200", "at": "0x401210"}],
        "call_descended": False,
        "detail": "the bytes are not followed past this call",
    }
    line = formatters._render_leaf_line(leaf)
    assert KIND in line and "0x401108" in line and "0x404060" in line
    assert "run_hook" in line and "0x401210" in line
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
