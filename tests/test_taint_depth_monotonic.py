"""A deeper ``--max-depth`` never loses a sink a shallower one reached (V5-01).

A call the depth bound stops at is assumed to return taint. When the same call
IS descended, the callee's own summary must be at least as tainting wherever
its body does what that assumption stood for. The shape that broke it: a
helper copies its tainted argument into a static buffer at a runtime offset
(``strcpy(buf + strlen(buf), name)``) and returns the buffer's address. The
write through ``buf + n`` was left unkeyed, so the descended helper handed back
a clean pointer and the fortified copy the caller makes from it -- reached
whenever the bound cut the helper off -- vanished at every deeper budget.
"""
from __future__ import annotations

import pytest
from test_taint_engine import FExpr, FInstr, FSSA, FSSAFunc, FVar, te
from test_taint_forward_global_frontier import FExtentBV, FGlobalBV, GFunc

NAME_BUF = 0x404800      # static name buffer the helper fills
OUT_BUF = 0x404c00       # the caller's fixed-size destination
STRLEN, STRCPY, STRCPY_CHK = 0x401030, 0x401040, 0x401050
MAIN, WALK, FORMAT = 0x401000, 0x401100, 0x401200


@pytest.fixture
def models():
    return te.load_models()


def _ptr(addr):
    return FExpr("MLIL_CONST_PTR", hex(addr), constant=addr)


def _var(ssa):
    return FExpr("MLIL_VAR_SSA", f"{ssa.var.name}#{ssa.version}", reads=[ssa])


def _call(index, addr, target, params, *, writes=()):
    return FInstr(index, addr, "MLIL_CALL_SSA", f"call {hex(target)}",
                  reads=[r for p in params for r in getattr(p, "vars_read", [])],
                  writes=list(writes), dest=_ptr(target), params=list(params))


def _program():
    """``main(arg)`` -> recursive ``walk(p, n)`` -> ``format_name(name)``.

    ``format_name`` appends ``name`` to a static buffer at ``strlen`` of it
    and returns the buffer; ``walk`` recurses, then copies what
    ``format_name`` returned into a fixed-size global with ``__strcpy_chk``."""
    name, n, d = FSSA(FVar("name"), 0), FSSA(FVar("n"), 1), FSSA(FVar("d"), 1)
    fmt = GFunc("format_name", FORMAT, FSSAFunc([
        _call(0, FORMAT + 0x04, STRLEN, [_ptr(NAME_BUF)], writes=[n]),
        FInstr(1, FORMAT + 0x09, "MLIL_SET_VAR_SSA", "d#1 = 0x404800 + n#1",
               reads=[n], writes=[d],
               src=FExpr("MLIL_ADD", "0x404800 + n#1", reads=[n],
                         left=_ptr(NAME_BUF), right=_var(n))),
        _call(2, FORMAT + 0x10, STRCPY, [_var(d), _var(name)]),
        FInstr(3, FORMAT + 0x15, "MLIL_RET", "return 0x404800", src=[_ptr(NAME_BUF)]),
    ]), params=[name.var])

    p, depth, depth1, r = (FSSA(FVar("p"), 0), FSSA(FVar("depth"), 0),
                           FSSA(FVar("depth"), 1), FSSA(FVar("r"), 1))
    walk = GFunc("walk", WALK, FSSAFunc([
        FInstr(0, WALK + 0x04, "MLIL_SET_VAR_SSA", "depth#1 = depth#0 - 1",
               reads=[depth], writes=[depth1],
               src=FExpr("MLIL_SUB", "depth#0 - 1", reads=[depth])),
        _call(1, WALK + 0x08, WALK, [_var(p), _var(depth1)]),
        _call(2, WALK + 0x10, FORMAT, [_var(p)], writes=[r]),
        _call(3, WALK + 0x18, STRCPY_CHK,
              [_ptr(OUT_BUF), _var(r), FExpr("MLIL_CONST", "0x401", constant=0x401)]),
        FInstr(4, WALK + 0x20, "MLIL_RET", "return", src=[]),
    ]), params=[p.var, depth.var], callees=[fmt])
    walk.callees.insert(0, walk)

    arg = FSSA(FVar("arg"), 0)
    main = GFunc("main", MAIN, FSSAFunc([
        _call(0, MAIN + 0x04, WALK, [_var(arg), FExpr("MLIL_CONST", "3", constant=3)]),
        FInstr(1, MAIN + 0x10, "MLIL_RET", "return", src=[]),
    ]), params=[arg.var], callees=[walk])
    bv = FGlobalBV({STRLEN: "strlen", STRCPY: "strcpy", STRCPY_CHK: "__strcpy_chk"},
                   funcs={MAIN: main, WALK: walk, FORMAT: fmt})
    return bv, main


def _reached(models, max_depth):
    bv, main = _program()
    result = te.TaintEngine(bv, models).forward(
        main, [te.parse_locator("param:0")], max_depth=max_depth)
    return {(f["sink"]["address"], f["sink"]["class"]) for f in result["reached_sinks"]}


def test_a_deeper_budget_reaches_every_sink_a_shallower_one_did(models):
    copy_out = (hex(WALK + 0x18), "fortified_overflow")
    reached = {depth: _reached(models, depth) for depth in range(1, 6)}
    # The bound cuts `format_name` off at depth 1, so its return is assumed
    # tainted there; every deeper run descends it and must agree.
    assert copy_out in reached[1]
    for shallow, deep in zip(range(1, 5), range(2, 6)):
        assert reached[shallow] <= reached[deep], (shallow, deep, reached)
    assert copy_out in reached[5]


def test_a_byte_stored_at_a_runtime_index_taints_the_static_buffer(models):
    # `buf[i] = c; system(buf)`: the store lands somewhere in `buf`, so the
    # command `system` runs is tainted -- not an uncorrelated store.
    c, i, d = FSSA(FVar("c"), 0), FSSA(FVar("i"), 0), FSSA(FVar("d"), 1)
    system = 0x401060
    stash = GFunc("stash", 0x401300, FSSAFunc([
        FInstr(0, 0x401304, "MLIL_SET_VAR_SSA", "d#1 = 0x404800 + i#0",
               reads=[i], writes=[d],
               src=FExpr("MLIL_ADD", "0x404800 + i#0", reads=[i],
                         left=_ptr(NAME_BUF), right=_var(i))),
        FInstr(1, 0x401308, "MLIL_STORE_SSA", "[d#1].b = c#0", reads=[c],
               dest=_var(d), src=_var(c)),
        _call(2, 0x401310, system, [_ptr(NAME_BUF)]),
    ]), params=[c.var, i.var])
    bv = FGlobalBV({system: "system"}, funcs={0x401300: stash})
    result = te.TaintEngine(bv, models).forward(stash, [te.parse_locator("param:0")])
    assert [(f["sink"]["callee"], f["sink"]["address"]) for f in result["reached_sinks"]] \
        == [("system", hex(0x401310))]
    # Nothing bounds `i` to the buffer: the byte may also have landed past it.
    assert [lf["global"]["address"] for lf in result["leaves"]
            if lf["kind"] == "coarse_memory_store" and lf.get("index_bounded") is False] \
        == [hex(NAME_BUF)]


# --------------------------------------------------------------------------
# D1-01: an index nothing bounds keeps the out-of-object frontier
# --------------------------------------------------------------------------

FIRST, SECOND = 0x404800, 0x404840     # two adjacent DECLARED 64-byte globals
SYSTEM, MEMCPY = 0x401060, 0x401070


class _Range:
    """BN's value set for an address: every value in [start, end]."""

    def __init__(self, start, end):
        self.type = type("T", (), {"name": "UnsignedRangeValue"})()
        self.ranges = [type("R", (), {"start": start, "end": end})()]


def _adjacent_globals_bv(func):
    return FExtentBV({SYSTEM: "system", MEMCPY: "memcpy", STRCPY: "strcpy"},
                     funcs={func.start: func}, objects={FIRST: 64, SECOND: 64},
                     authoritative=[FIRST, SECOND], section=(0x404000, 0x405000))


def _indexed_program(write, index_range=None):
    """`f(v, i)`: write *v* at `first + i`, then `system(second)`.

    *write* is ``"store"`` (`first[i] = v`), ``"memcpy"`` (an 8-byte copy)
    or ``"strcpy"`` (a copy no length bounds). *index_range* is BN's value
    set for `first + i`, or None when nothing bounds it."""
    v, i, d = FSSA(FVar("v"), 0), FSSA(FVar("i"), 0), FSSA(FVar("d"), 1)
    dest = _var(d)
    if index_range is not None:
        dest.possible_values = _Range(*index_range)
    instrs = [FInstr(0, 0x401304, "MLIL_SET_VAR_SSA", "d#1 = first + i#0",
                     reads=[i], writes=[d],
                     src=FExpr("MLIL_ADD", "first + i#0", reads=[i],
                               left=_ptr(FIRST), right=_var(i)))]
    if write == "store":
        store = FInstr(1, 0x401308, "MLIL_STORE_SSA", "[d#1].b = v#0", reads=[v],
                       dest=dest, src=_var(v))
        store.size = 1
        instrs.append(store)
    elif write == "memcpy":
        instrs.append(_call(1, 0x401308, MEMCPY,
                            [dest, _var(v), FExpr("MLIL_CONST", "8", constant=8)]))
    else:
        instrs.append(_call(1, 0x401308, STRCPY, [dest, _var(v)]))
    instrs.append(_call(2, 0x401310, SYSTEM, [_ptr(SECOND)]))
    func = GFunc("f", 0x401300, FSSAFunc(instrs), params=[v.var, i.var])
    return func, _adjacent_globals_bv(func)


def _index_frontiers(result):
    return [lf for lf in result["leaves"]
            if lf["kind"] == "coarse_memory_store" and lf.get("index_bounded") is False]


@pytest.mark.parametrize("write", ["store", "memcpy", "strcpy"])
def test_an_unbounded_index_into_a_declared_global_is_a_blocking_frontier(models, write):
    # The reviewer's shape: `first[i] = v; system(second)`. The object taint
    # cannot widen into `second` (its declared extent ends there), so without
    # a frontier the possibility that the write landed in `second` vanished
    # into an all-clear.
    func, bv = _indexed_program(write)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    # `second` is not tainted by the write (strcpy's own copy is a sink).
    assert not [f for f in result["reached_sinks"] if f["sink"]["callee"] == "system"]
    [leaf] = _index_frontiers(result)
    assert (leaf["global"]["address"], leaf["address"]) == (hex(FIRST), hex(0x401308))
    if not result["reached_sinks"]:
        assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("write, reach", [("store", 1), ("memcpy", 8)])
def test_an_index_bn_proves_inside_the_object_raises_no_frontier(models, write, reach):
    # BN bounds the address to [first, first + 64 - reach]: the write stays in
    # `first`, whose own taint is the whole answer.
    func, bv = _indexed_program(write, (FIRST, FIRST + 64 - reach))
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    assert _index_frontiers(result) == []


@pytest.mark.parametrize("write, reach", [("store", 1), ("memcpy", 8)])
def test_an_index_bound_that_reaches_the_neighbour_keeps_the_frontier(models, write, reach):
    # One byte too far: the last write may cover `second`'s first byte.
    func, bv = _indexed_program(write, (FIRST, FIRST + 64 - reach + 1))
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    assert [lf["global"]["address"] for lf in _index_frontiers(result)] == [hex(FIRST)]
