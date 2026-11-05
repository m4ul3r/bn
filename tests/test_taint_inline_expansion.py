"""Inline string expansions (V8-03): BN lifts gcc's inline `rep stos`/`rep movs`
to an `LLIL_INTRINSIC` that MLIL renders as a synthesised `__builtin_memset`/
`__builtin_memcpy` call. When the count is a constant total split by pointer
alignment, taint on the POINTER does not steer it: the sink is `bounded_len`
with an `intrinsic_expansion` qualifier. A tainted total, or the same length
handed to a real call, still reports `overflow_len`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te

MEMSET = 0x3000
SITE = 0x1040


@pytest.fixture
def models():
    return te.load_models()


def _const(value):
    return FExpr("MLIL_CONST", hex(value), constant=value)


def _v(ssa):
    return FExpr("MLIL_VAR_SSA", f"{ssa.var.name}#{ssa.version}", reads=[ssa], src=ssa)


def _bin(op, left, right, text):
    return FExpr(op, text, reads=list(left.vars_read) + list(right.vars_read),
                 left=left, right=right)


def _intrinsic(name):
    return SimpleNamespace(operation=SimpleNamespace(name="LLIL_INTRINSIC"),
                           intrinsic=SimpleNamespace(name=name), params=[], address=SITE)


def _fill(length_of, *, intrinsic=True):
    """``fill(p, n)``: ``len#1 = length_of(p#0, n#0); __builtin_memset(p#0, 0, len#1)``
    at SITE, an inline `rep stos` expansion unless *intrinsic* is False (then
    the instruction there is a real call)."""
    p, n = FVar("p"), FVar("n")
    p0, n0 = FSSA(p, 0), FSSA(n, 0)
    a1, len1 = FSSA(FVar("a"), 1), FSSA(FVar("len"), 1)
    aligned = _bin("MLIL_AND", _bin("MLIL_ADD", _v(p0), _const(8), "p#0 + 8"),
                   _const(-8), "(p#0 + 8) & -8")
    length = length_of(p0, n0, a1)
    instrs = [
        FInstr(0, 0x1010, "MLIL_SET_VAR_SSA", "a#1 = (p#0 + 8) & -8", reads=[p0],
               writes=[a1], src=aligned),
        FInstr(1, 0x1020, "MLIL_SET_VAR_SSA", f"len#1 = {length}",
               reads=list(length.vars_read), writes=[len1], src=length),
        FInstr(2, SITE, "MLIL_CALL_SSA", "__builtin_memset(p#0, 0, len#1)", reads=[p0, len1],
               dest=FExpr("MLIL_CONST_PTR", hex(MEMSET), constant=MEMSET),
               params=[_v(p0), _const(0), _v(len1)]),
    ]
    ssa = FSSAFunc(instrs)
    func = FFunc("fill", 0x1000, ssa, params=[p, n])
    lifted = ([_intrinsic("__memfill_u64"), _intrinsic("__memfill_u64")] if intrinsic
              else [SimpleNamespace(operation=SimpleNamespace(name="LLIL_CALL"), address=SITE)])
    func.get_llils_at = lambda addr: lifted if addr == SITE else []
    ssa.source_function = func
    return func, FBV({MEMSET: "__builtin_memset"})


def _aligned_total(p0, _n0, a1):
    # ((p - align8(p + 8)) + 0x140) >> 3 << 3: a 0x140-byte fill's aligned middle
    middle = _bin("MLIL_ADD", _bin("MLIL_SUB", _v(p0), _v(a1), "p#0 - a#1"), _const(0x140),
                  "p#0 - a#1 + 0x140")
    return _bin("MLIL_LSL", _bin("MLIL_LSR", middle, _const(3), "(...) u>> 3"), _const(3),
                "(...) << 3")


def _masked_distance(p0, n0, _a1):
    # (p - n) & 7: the misalignment between two pointers
    return _bin("MLIL_AND", _bin("MLIL_SUB", _v(p0), _v(n0), "p#0 - n#0"), _const(7),
                "(p#0 - n#0) & 7")


def _negated_residue(p0, _n0, _a1):
    # -p & 7: the bytes up to p's next 8-byte boundary
    neg = FExpr("MLIL_NEG", "-p#0", reads=[p0], src=_v(p0))
    return _bin("MLIL_AND", neg, _const(7), "-p#0 & 7")


def _tainted_total(p0, n0, a1):
    # (n + (p - align8(p + 8))) >> 3 << 3: the same expansion of a variable size
    middle = _bin("MLIL_ADD", _v(n0), _bin("MLIL_SUB", _v(p0), _v(a1), "p#0 - a#1"),
                  "n#0 + (p#0 - a#1)")
    return _bin("MLIL_LSL", _bin("MLIL_LSR", middle, _const(3), "(...) u>> 3"), _const(3),
                "(...) << 3")


def _memset_sink(models, func, bv, source):
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator(source)])
    sinks = [s["sink"] for s in result["reached_sinks"] if s["sink"]["address"] == hex(SITE)]
    assert len(sinks) == 1, result["reached_sinks"]
    return sinks[0]


@pytest.mark.parametrize("length_of, bound", [
    (_aligned_total, 0x138), (_masked_distance, 7), (_negated_residue, 7)])
def test_an_alignment_split_length_over_a_tainted_pointer_is_bounded(models, length_of, bound):
    sink = _memset_sink(models, *_fill(length_of), "param:0")
    assert sink["class"] == "bounded_len"
    assert sink["catalog_class"] == "overflow_len"
    assert sink["intrinsic_expansion"] == {"intrinsic": "__memfill_u64",
                                           "length_bound": hex(bound)}


def test_a_tainted_total_in_the_same_expansion_still_overflows(models):
    sink = _memset_sink(models, *_fill(_tainted_total), "param:1")
    assert sink["class"] == "overflow_len"
    assert "intrinsic_expansion" not in sink


def test_a_real_call_with_the_same_length_is_not_an_expansion(models):
    sink = _memset_sink(models, *_fill(_aligned_total, intrinsic=False), "param:0")
    assert sink["class"] == "overflow_len"
    assert "intrinsic_expansion" not in sink
