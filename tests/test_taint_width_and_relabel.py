"""Forward sink disclosures: a length computed in a narrower integer than the
sink's parameter (W10-04), and the ``overread_len`` relabel gate for a length
that is the return of the modeled fill of the copy's own source (W01-10)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FPVS, FSSA, FSSAFunc, FVar, te


@pytest.fixture
def models():
    return te.load_models()


def _sized(expr, size):
    expr.size = size
    return expr


def _var(ssa, size):
    return _sized(FExpr("MLIL_VAR_SSA", f"{ssa.var.name}#{ssa.version}", reads=[ssa]), size)


# ---------------------------------------------------------------------------
# W10-04: sink.width_narrowing
# ---------------------------------------------------------------------------

def _alloc_after(arith_op, arith_size, *, widen="MLIL_ZX"):
    """``t = n <op> 1 (arith_size bytes); w = zx.q(t); p = calloc(1, w)``."""
    n = FVar("n"); n0 = FSSA(n, 0)
    t1 = FSSA(FVar("t"), 1); w1 = FSSA(FVar("w"), 1); p1 = FSSA(FVar("p"), 1)
    arith = _sized(FExpr(arith_op, "n#0 + 1", reads=[n0], left=_var(n0, 4),
                         right=_sized(FExpr("MLIL_CONST", "1", constant=1), arith_size)),
                   arith_size)
    widened = _sized(FExpr(widen, "zx.q(t#1)", reads=[t1], src=_var(t1, arith_size)), 8)
    instrs = [
        FInstr(0, 0x1008, "MLIL_SET_VAR_SSA", "t#1 = n#0 + 1", reads=[n0], writes=[t1], src=arith),
        FInstr(1, 0x100c, "MLIL_SET_VAR_SSA", "w#1 = zx.q(t#1)", reads=[t1], writes=[w1], src=widened),
        FInstr(2, 0x1014, "MLIL_CALL_SSA", "p#1 = calloc(1, w#1)", reads=[w1], writes=[p1],
               dest=FExpr("MLIL_CONST_PTR", "0x3000", constant=0x3000),
               params=[_sized(FExpr("MLIL_CONST", "1", constant=1), 8), _var(w1, 8)]),
    ]
    func = FFunc("frame_alloc", 0x1000, FSSAFunc(instrs), params=[n])
    return func, FBV({0x3000: "calloc"})


def _calloc_sink(models, func, bv):
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    sinks = [s["sink"] for s in result["reached_sinks"] if s["sink"]["callee"] == "calloc"]
    assert len(sinks) == 1, result["reached_sinks"]
    return sinks[0]


def test_a_32_bit_add_widened_into_calloc_discloses_width_narrowing(models):
    sink = _calloc_sink(models, *_alloc_after("MLIL_ADD", 4))
    assert sink["class"] == "alloc_size"
    assert sink["tainted_arg_index"] == 1
    assert sink["width_narrowing"] == {"computed_width": 4, "parameter_width": 8, "at": "0x1008"}


@pytest.mark.parametrize("op", ["MLIL_MUL", "MLIL_LSL", "MLIL_SUB"])
def test_every_wrapping_op_is_a_narrowing(models, op):
    sink = _calloc_sink(models, *_alloc_after(op, 4, widen="MLIL_SX"))
    assert sink["width_narrowing"]["computed_width"] == 4
    assert sink["width_narrowing"]["at"] == "0x1008"


def test_an_add_in_the_parameter_width_is_not_a_narrowing(models):
    sink = _calloc_sink(models, *_alloc_after("MLIL_ADD", 8))
    assert sink["class"] == "alloc_size"
    assert "width_narrowing" not in sink


def test_a_non_wrapping_last_op_is_not_a_narrowing(models):
    # `zx.q(n & 0xffff)`: a mask cannot wrap, so there is nothing to disclose.
    sink = _calloc_sink(models, *_alloc_after("MLIL_AND", 4))
    assert "width_narrowing" not in sink


def test_width_narrowing_renders_one_line_under_the_flow():
    from bn.formatters import _render_taint_text
    value = {"direction": "forward", "function": {"name": "frame_alloc", "address": "0x1000"},
             "sources": [], "reached_sinks": [{"sink": {
                 "callee": "calloc", "address": "0x1014", "tainted_arg_index": 1,
                 "class": "alloc_size",
                 "width_narrowing": {"computed_width": 4, "parameter_width": 8, "at": "0x1008"}}}]}
    lines = [ln.strip() for ln in _render_taint_text(value).splitlines()]
    narrow = [ln for ln in lines if ln.startswith("width narrowing:")]
    assert len(narrow) == 1
    assert "0x1008" in narrow[0] and " 4 " in narrow[0] and "8-byte" in narrow[0]
    flow = next(i for i, ln in enumerate(lines) if ln.startswith("[alloc_size] calloc @ 0x1014"))
    assert lines[flow + 1] == narrow[0]


# ---------------------------------------------------------------------------
# W01-10: overread_len gate for a fill's own return
# ---------------------------------------------------------------------------

def _value_range(start, end):
    """BN's value set for a length its dominating branches confine to
    ``start..end``: `if (n <= 0) return;` gives ``1..0x7fffffff``."""
    pvs = FPVS("SignedRangeValue")
    pvs.ranges = [SimpleNamespace(start=start, end=end, step=1)]
    return pvs


#: The copy's length after `if (n <= 0) return;` on a 32-bit return.
GUARDED = _value_range(1, 0x7fffffff)


def _recv_then_dup(*, count, length, src, at_copy=GUARDED):
    """``p = &buf; n = recv(fd, p, <count>, 0); m = <length>(n);
    d = malloc(m); memcpy(d, <src>, m)`` with ``char buf[0x40]``.

    ``length``: ``"sx"`` (``m = sx.q(n)``) or ``"add"`` (``m = n + 4``).
    ``src``: ``"same"`` (``&buf``), ``"offset"`` (``&buf + 4``) or ``"other"``
    (``&spare``, another 64-byte local). ``at_copy``: BN's value set for the
    copy's length operand, which carries the guards dominating the copy
    (``GUARDED`` by default; None when nothing is stated)."""
    buf = FVar("buf", typ="char[0x40]"); spare = FVar("spare", typ="char[0x40]")
    fd0 = FSSA(FVar("fd"), 0); cap0 = FSSA(FVar("cap"), 0)
    p1 = FSSA(FVar("p"), 1); n1 = FSSA(FVar("n"), 1); m2 = FSSA(FVar("m"), 2)
    d1 = FSSA(FVar("d"), 1)
    count_expr = (FExpr("MLIL_CONST", "0x40", constant=0x40) if count == "const"
                  else _var(cap0, 8))
    m_src = (FExpr("MLIL_SX", "sx.q(n#1)", reads=[n1], src=_var(n1, 4)) if length == "sx"
             else FExpr("MLIL_ADD", "n#1 + 4", reads=[n1], left=_var(n1, 4),
                        right=FExpr("MLIL_CONST", "4", constant=4)))
    src_expr = {
        "same": FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
        "offset": FExpr("MLIL_ADD", "&buf + 4", left=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                        right=FExpr("MLIL_CONST", "4", constant=4)),
        "other": FExpr("MLIL_ADDRESS_OF", "&spare", src=spare),
    }[src]
    instrs = [
        FInstr(0, 0x1004, "MLIL_SET_VAR_SSA", "p#1 = &buf", writes=[p1],
               src=FExpr("MLIL_ADDRESS_OF", "&buf", src=buf)),
        FInstr(1, 0x1010, "MLIL_CALL_SSA", "n#1 = recv(fd#0, p#1, count, 0)",
               reads=[fd0, p1], writes=[n1],
               dest=FExpr("MLIL_CONST_PTR", "0x3000", constant=0x3000),
               params=[_var(fd0, 4), _var(p1, 8), count_expr,
                       FExpr("MLIL_CONST", "0", constant=0)]),
        FInstr(2, 0x1018, "MLIL_SET_VAR_SSA", f"m#2 = {m_src}", reads=[n1], writes=[m2], src=m_src),
        FInstr(3, 0x1020, "MLIL_CALL_SSA", "d#1 = malloc(m#2)", reads=[m2], writes=[d1],
               dest=FExpr("MLIL_CONST_PTR", "0x3010", constant=0x3010), params=[_var(m2, 8)]),
        FInstr(4, 0x1030, "MLIL_CALL_SSA", "memcpy(d#1, src, m#2)", reads=[d1, m2],
               dest=FExpr("MLIL_CONST_PTR", "0x3020", constant=0x3020),
               params=[_var(d1, 8), src_expr, _at_copy(_var(m2, 8), at_copy)]),
    ]
    func = FFunc("pull_reply", 0x1000, FSSAFunc(instrs), params=[FVar("fd"), FVar("cap")])
    return func, FBV({0x3000: "recv", 0x3010: "malloc", 0x3020: "memcpy"})


def _at_copy(expr, pvs):
    if pvs is not None:
        expr.possible_values = pvs
    return expr


def _memcpy_sink(models, func, bv):
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("call:recv")])
    sinks = [s["sink"] for s in result["reached_sinks"] if s["sink"]["callee"] == "memcpy"]
    assert len(sinks) == 1, result["reached_sinks"]
    return sinks[0]


@pytest.mark.parametrize("count", ["const", "param"])
def test_a_fill_return_copied_back_out_of_the_same_buffer_is_not_an_overread(models, count):
    # R3-01: `if (n <= 0) return;` dominates the copy, so the length is a
    # byte count the fill wrote, not a failure return.
    sink = _memcpy_sink(models, *_recv_then_dup(count=count, length="sx", src="same"))
    assert sink["class"] == "bounded_len"
    assert sink["catalog_class"] == "overflow_len"
    assert sink["src_fill"] == {"callee": "recv", "at": "0x1010"}
    assert sink["src_capacity"] == 0x40


@pytest.mark.parametrize("length,src", [
    ("sx", "other"),     # a different buffer than the one recv filled
    ("sx", "offset"),    # the same buffer, read from a later offset
    ("add", "same"),     # arithmetic on the return: not the fill's own count
])
def test_the_overread_still_fires_when_the_length_is_not_that_fills_return(models, length, src):
    sink = _memcpy_sink(models, *_recv_then_dup(count="param", length=length, src=src))
    assert sink["class"] == "overread_len", sink
    assert "src_fill" not in sink
