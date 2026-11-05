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


# ---------------------------------------------------------------------------
# V7-01: a modeled receive's return_bound bounds a copy only once proven
# ---------------------------------------------------------------------------

#: No branch narrows the return: recv's -1 failure is a possible length.
UNCHECKED = _value_range(-0x8000000000000000, 0x7fffffffffffffff)


def _recv_then_copy(*, width, at_copy):
    """``n = recv(fd, buf, 0x40, 0); memcpy(dst, buf, <n>)`` with a fixed
    destination: an 8-byte ``ssize_t n`` read as is, or a 4-byte ``int n``
    read as ``sx.q(n)``. ``at_copy``: BN's value set for the length."""
    buf = FVar("buf", typ="char[0x40]")
    fd0, dst0, n1 = FSSA(FVar("fd"), 0), FSSA(FVar("dst"), 0), FSSA(FVar("n"), 1)
    length = (_var(n1, 8) if width == 8 else
              _sized(FExpr("MLIL_SX", "sx.q(n#1)", reads=[n1], src=_var(n1, 4)), 8))
    instrs = [
        FInstr(0, 0x1010, "MLIL_CALL_SSA", "n#1 = recv(fd#0, &buf, 0x40, 0)",
               reads=[fd0], writes=[n1],
               dest=FExpr("MLIL_CONST_PTR", "0x3000", constant=0x3000),
               params=[_var(fd0, 4), FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                       FExpr("MLIL_CONST", "0x40", constant=0x40),
                       FExpr("MLIL_CONST", "0", constant=0)]),
        FInstr(1, 0x1020, "MLIL_CALL_SSA", "memcpy(dst#0, &buf, n)", reads=[dst0, n1],
               dest=FExpr("MLIL_CONST_PTR", "0x3020", constant=0x3020),
               params=[_var(dst0, 8), FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                       _at_copy(length, at_copy)]),
    ]
    func = FFunc("pull", 0x1000, FSSAFunc(instrs), params=[FVar("fd"), FVar("dst")])
    return func, FBV({0x3000: "recv", 0x3020: "memcpy"})


@pytest.mark.parametrize("width", [8, 4], ids=["ssize_t", "int"])
@pytest.mark.parametrize("at_copy, bounded", [
    (GUARDED, True),        # `if (n <= 0) return;` dominates the copy
    (UNCHECKED, False),     # nothing checks the return
    (None, False),          # BN states nothing
], ids=["guarded", "unchecked", "no-value-set"])
def test_a_receive_return_bounds_the_copy_only_when_proven_a_count(models, width, at_copy,
                                                                   bounded):
    # V7-01: the 8-byte unchecked return was `bounded_len` by recv's count
    # argument, though its -1 failure is SIZE_MAX at the copy.
    sink = _memcpy_sink(models, *_recv_then_copy(width=width, at_copy=at_copy))
    if bounded:
        assert (sink["class"], sink.get("source_bound")) == ("bounded_len", "0x40"), sink
        assert "return_unchecked" not in sink
    else:
        assert (sink["class"], sink.get("return_unchecked")) == ("overflow_len", True), sink
        assert "source_bound" not in sink


def test_an_unchecked_fill_return_is_no_bound_on_its_source_either(models):
    # The destination is sized by the length, the source is a 64-byte buffer
    # the length is the unchecked fill return of: it no longer "provably
    # holds" the copied length.
    sink = _memcpy_sink(models, *_recv_then_dup(count="const", length="sx", src="other",
                                                at_copy=UNCHECKED))
    assert (sink["class"], sink.get("return_unchecked"), sink["src_capacity"]) == (
        "overread_len", True, 0x40), sink


# ---------------------------------------------------------------------------
# G3-01/02: a guard on the return that excludes -1, as the CFG proves it
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.incoming_edges, self.outgoing_edges = [], []


def _edge(source, target):
    e = SimpleNamespace(source=source, target=target)
    source.outgoing_edges.append(e)
    target.incoming_edges.append(e)


_CMP = {"<": "MLIL_CMP_SLT", ">": "MLIL_CMP_SGT", "!=": "MLIL_CMP_NE",
        "u>": "MLIL_CMP_UGT", "u<": "MLIL_CMP_ULT"}


def _recv_guard_then_copy(*, guard, copy_on, copy_first=True, count="const"):
    """``n = recv(fd, &buf, <count>, 0); if (<guard>) ...`` with the copy
    ``memcpy(dst, &buf, sx.q(n))`` heading the block the branch's *copy_on*
    (``"true"``/``"false"``) side enters, the other side returning. ``int n``
    (4 bytes). *guard* is ``(left, test, const)``: *left* ``"n"`` (the return
    as is), ``"sx"`` (``(size_t)n``) or ``"zx"`` (``(size_t)(unsigned)n``);
    *test* a key of ``_CMP``. *copy_first*: the copy is the first instruction
    of its block, else a store precedes it there. BN's value set at the copy
    is unnarrowed, so only the branch can prove the return a count.
    *count*: ``"const"`` (0x40) or ``"param"``."""
    left, test, const = guard
    buf = FVar("buf", typ="char[0x40]")
    fd0, dst0, cap0, n1 = (FSSA(FVar("fd"), 0), FSSA(FVar("dst"), 0), FSSA(FVar("cap"), 0),
                           FSSA(FVar("n"), 1))
    left_expr = {
        "n": _var(n1, 4),
        "sx": _sized(FExpr("MLIL_SX", "sx.q(n#1)", reads=[n1], src=_var(n1, 4)), 8),
        "zx": _sized(FExpr("MLIL_ZX", "zx.q(n#1)", reads=[n1], src=_var(n1, 4)), 8),
        "sign": _sized(FExpr("MLIL_AND", "n#1 & 0x80000000", reads=[n1], left=_var(n1, 4),
                             right=_sized(FExpr("MLIL_CONST", "0x80000000",
                                                constant=0x80000000), 4)), 4),
    }[left]
    cond = FExpr(_CMP[test], f"{left_expr} {test} {const:#x}", reads=[n1], left=left_expr,
                 right=_sized(FExpr("MLIL_CONST", hex(const), constant=const), left_expr.size))
    fill = FInstr(0, 0x1010, "MLIL_CALL_SSA", "n#1 = recv(fd#0, &buf, count, 0)",
                  reads=[fd0], writes=[n1],
                  dest=FExpr("MLIL_CONST_PTR", "0x3000", constant=0x3000),
                  params=[_var(fd0, 4), FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                          FExpr("MLIL_CONST", "0x40", constant=0x40) if count == "const"
                          else _var(cap0, 8), FExpr("MLIL_CONST", "0", constant=0)])
    branch = FInstr(1, 0x1018, "MLIL_IF", f"if ({cond})", reads=[n1])
    branch.condition = cond
    pre = [] if copy_first else [
        FInstr(2, 0x1020, "MLIL_SET_VAR_SSA", "t#1 = 0", writes=[FSSA(FVar("t"), 1)],
               src=FExpr("MLIL_CONST", "0", constant=0))]
    k = len(pre)
    length = _at_copy(_sized(FExpr("MLIL_SX", "sx.q(n#1)", reads=[n1], src=_var(n1, 4)), 8),
                      UNCHECKED)
    copy = FInstr(2 + k, 0x1024, "MLIL_CALL_SSA", "memcpy(dst#0, &buf, sx.q(n#1))",
                  reads=[dst0, n1], dest=FExpr("MLIL_CONST_PTR", "0x3020", constant=0x3020),
                  params=[_var(dst0, 8), FExpr("MLIL_ADDRESS_OF", "&buf", src=buf), length])
    length.instr_index = copy.instr_index
    err = FInstr(3 + k, 0x1030, "MLIL_RET", "return")
    head, body, tail = _Block(0, 2), _Block(2, 3 + k), _Block(3 + k, 4 + k)
    _edge(head, body)
    _edge(head, tail)
    _edge(body, tail)
    for ins, block in [(fill, head), (branch, head), *[(p, body) for p in pre],
                       (copy, body), (err, tail)]:
        ins.il_basic_block = block
    other = "false" if copy_on == "true" else "true"
    setattr(branch, copy_on, body.start)
    setattr(branch, other, tail.start)
    func = FFunc("pull", 0x1000, FSSAFunc([fill, branch, *pre, copy, err]),
                 params=[FVar("fd"), FVar("dst"), FVar("cap")])
    return func, FBV({0x3000: "recv", 0x3020: "memcpy"})


@pytest.mark.parametrize("copy_first", [True, False], ids=["copy-heads-block", "copy-later"])
@pytest.mark.parametrize("guard, copy_on", [
    (("n", "<", 0), "false"),                  # if (n < 0) goto err; memcpy(...)
    (("n", "!=", -1), "true"),                 # if (n != -1) memcpy(...)
    (("sx", "u>", 0x40), "false"),             # if ((size_t)n > 0x40) return; memcpy(...)
    (("n", "u>", 0x40), "false"),              # if ((unsigned)n > 0x40) return; ...
    (("sx", "u<", 0x41), "true"),              # if ((size_t)n < 0x41) memcpy(...)
    (("zx", "u>", 0x40), "false"),             # (size_t)(unsigned)n: -1 is 0xffffffff
    (("sign", "!=", 0), "false"),              # AArch64's `n < 0`: (n & 1 << 31) != 0
], ids=["signed-lt-0", "ne-minus-1", "size_t-ugt", "unsigned-ugt", "size_t-ult", "zx-ugt",
        "sign-bit"])
def test_a_guard_that_excludes_the_failure_return_bounds_the_copy(models, guard, copy_on,
                                                                  copy_first):
    # G3-01: with the copy heading the guarded block the search for a path
    # avoiding that block's entry started AT the copy and found one. G3-02:
    # only signed and equality tests were read, though -1 is the unsigned
    # maximum an unsigned bound excludes.
    sink = _memcpy_sink(models, *_recv_guard_then_copy(guard=guard, copy_on=copy_on,
                                                       copy_first=copy_first))
    assert (sink["class"], sink.get("source_bound")) == ("bounded_len", "0x40"), sink
    assert "return_unchecked" not in sink


@pytest.mark.parametrize("guard, copy_on", [
    (("n", ">", 0x40), "false"),               # if (n > 0x40) return: -1 passes it
    (("zx", "u>", 0xffffffff), "false"),       # zero-extended -1 is 0xffffffff, not above it
    (("sx", "u<", 0x41), "false"),             # the copy is where -1 (SIZE_MAX) goes
    (("n", "<", 0), "true"),                   # the copy is on the failure side
    (("sign", "!=", 0), "true"),               # ... as the sign bit says
], ids=["signed-gt", "zx-ugt-wide", "size_t-ult-else", "failure-side", "sign-bit-failure-side"])
def test_a_guard_the_failure_return_passes_leaves_it_unchecked(models, guard, copy_on):
    sink = _memcpy_sink(models, *_recv_guard_then_copy(guard=guard, copy_on=copy_on))
    assert (sink["class"], sink.get("return_unchecked")) == ("overflow_len", True), sink
    assert "source_bound" not in sink


# ---------------------------------------------------------------------------
# G3-03: a non-constant count still leaves an unchecked return unchecked
# ---------------------------------------------------------------------------

def test_an_unchecked_return_of_a_fill_with_a_variable_count_is_return_unchecked(models):
    # `recv(fd, buf, cap, 0)` copied unchecked: no constant bounds it, and the
    # qualifier was dropped with the bound.
    sink = _memcpy_sink(models, *_recv_guard_then_copy(guard=("n", ">", 0x40), copy_on="false",
                                                       count="param"))
    assert (sink["class"], sink.get("return_unchecked")) == ("overflow_len", True), sink
    assert "source_bound" not in sink


def test_a_proven_return_with_a_variable_count_keeps_its_class_unqualified(models):
    sink = _memcpy_sink(models, *_recv_guard_then_copy(guard=("n", "<", 0), copy_on="false",
                                                       count="param"))
    assert sink["class"] == "overflow_len" and "return_unchecked" not in sink, sink
    assert "source_bound" not in sink


def test_an_unchecked_variable_count_fill_return_is_no_bound_on_its_source(models):
    # The source side of the same fact: `memcpy(malloc(m), &spare, m)` with
    # `m` an unchecked `recv(fd, &buf, cap, 0)` return.
    sink = _memcpy_sink(models, *_recv_then_dup(count="param", length="sx", src="other",
                                                at_copy=UNCHECKED))
    assert (sink["class"], sink.get("return_unchecked")) == ("overread_len", True), sink

