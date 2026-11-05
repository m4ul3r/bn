"""Seed advice that names the callers to seed instead.

W12-06: a `call:read`/`arg:read:1` seed inside a read wrapper whose buffer is
the wrapper's own parameter cannot key the buffer -- its callers own it. The
`source_seed_misanchored` leaf names those callers (first ones by callsite
address, plus the total), each with a ready `arg:<wrapper>:<n>` reseed, and
`next_action` prints them.
W10-03: a callee seeded in a function that does not call it directly is
refused with the functions that DO call it, the ones this function calls first.
"""
from __future__ import annotations

import re
import shlex

import pytest
from test_taint_engine import (
    FBV,
    FSSA,
    FExpr,
    FFunc,
    FInstr,
    FSite,
    FSSAFunc,
    FVar,
    _ext_call,
    te,
)

READ, PULL = 0x900, 0x3000


@pytest.fixture
def models():
    return te.load_models()


def _var(ssa):
    return FExpr("MLIL_VAR_SSA", f"{ssa.var.name}#{ssa.version}", reads=[ssa])


def _caller(name, start):
    return FFunc(name, start, FSSAFunc([]))


def _read_wrapper(*, pointer="param", callers=()):
    """`fill(fd, buf, want)`: `got = read(fd, <pointer>, want)` where the pointer
    is the incoming `buf`, `buf + have`, or a field of a context."""
    fd, buf, want, got, have, ctx = (FVar(n, ident=i) for i, n in enumerate(
        ("fd", "buf", "want", "got", "have", "ctx")))
    fd0, buf0, want0, got1, have1 = FSSA(fd, 0), FSSA(buf, 0), FSSA(want, 0), FSSA(got, 1), FSSA(have, 1)
    setup = []
    if pointer == "param":
        dest = _var(buf0)
    elif pointer == "offset":
        setup = [FInstr(0, 0x1000, "MLIL_SET_VAR_SSA", "have#1 = 0x10", writes=[have1],
                        src=FExpr("MLIL_CONST", "0x10", constant=0x10))]
        dest = FExpr("MLIL_ADD", "buf#0 + have#1", reads=[buf0, have1],
                     left=_var(buf0), right=_var(have1))
    else:   # a field of an incoming context: not the parameter itself
        ctx0 = FSSA(ctx, 0)
        dest = FExpr("MLIL_LOAD_SSA", "[ctx#0 + 8]", reads=[ctx0],
                     src=FExpr("MLIL_ADD", "ctx#0 + 8", reads=[ctx0], left=_var(ctx0),
                               right=FExpr("MLIL_CONST", "0x8", constant=8)))
    instrs = [
        *setup,
        _ext_call(1, 0x1004, "got#1 = read(fd#0, buf, want#0)", READ,
                  [_var(fd0), dest, _var(want0)], reads=[fd0, buf0, want0], writes=[got1]),
    ]
    params = [ctx, buf, want] if pointer == "field" else [fd, buf, want]
    return FFunc("fill", 0x1000, FSSAFunc(instrs), params=params, caller_sites=callers)


# Seven sites from six callers, listed out of address order; `beta` calls twice.
_SITES = [
    (_caller("zeta", 0x9000), 0x9010),
    (_caller("beta", 0x2000), 0x2040),
    (_caller("eta", 0xa000), 0xa020),
    (_caller("alpha", 0x1800), 0x1830),
    (_caller("delta", 0x5000), 0x5008),
    (_caller("gamma", 0x4000), 0x4010),
]
_BETA_AGAIN = 0x2010


def _sites():
    sites = [FSite(fn, at) for fn, at in _SITES]
    sites.append(FSite(_SITES[1][0], _BETA_AGAIN))
    return sites


def _misanchored(result):
    return [lf for lf in result["leaves"] if lf.get("kind") == "source_seed_misanchored"]


def _engine(models):
    return te.TaintEngine(FBV({READ: "read"}), models)


@pytest.mark.parametrize("source", ["call:read", "arg:read:1"])
@pytest.mark.parametrize("pointer", ["param", "offset"])
def test_wrapper_buffer_leaf_names_the_first_callers_to_seed(models, source, pointer):
    result = _engine(models).forward(_read_wrapper(pointer=pointer, callers=_sites()),
                                     [te.parse_locator(source)])

    [leaf] = _misanchored(result)
    assert leaf["reason"] == "buffer_not_keyed"
    assert leaf["buffer_param"] == {"function": "fill", "index": 1}
    assert leaf["callers_total"] == 6
    # First five callers by callsite address, each at its lowest callsite.
    assert [(r["function"]["name"], r["address"]) for r in leaf["caller_seeds"]] == [
        ("alpha", "0x1830"), ("beta", "0x2010"), ("gamma", "0x4010"),
        ("delta", "0x5008"), ("zeta", "0x9010")]
    for row in leaf["caller_seeds"]:
        [command] = re.findall(r"`(bn [^`]+)`", row["reseed"])
        assert shlex.split(command) == [
            "bn", "taint", "forward", "-f", row["function"]["address"],
            "--source", "arg:fill:1"]
    advice = result["diagnostics"]["next_action"]
    assert [c for c in re.findall(r"`(bn [^`]+)`", advice) if "arg:fill:1" in c] == [
        re.findall(r"`(bn [^`]+)`", row["reseed"])[0] for row in leaf["caller_seeds"]]


def test_wrapper_buffer_with_no_caller_reports_zero_and_no_reseed(models):
    result = _engine(models).forward(_read_wrapper(), [te.parse_locator("call:read")])

    [leaf] = _misanchored(result)
    assert (leaf["buffer_param"], leaf["callers_total"], leaf["caller_seeds"]) == (
        {"function": "fill", "index": 1}, 0, [])
    assert "arg:fill:" not in result["diagnostics"]["next_action"]


def test_buffer_that_is_not_a_parameter_names_no_callers(models):
    result = _engine(models).forward(_read_wrapper(pointer="field", callers=_sites()),
                                     [te.parse_locator("call:read")])

    [leaf] = _misanchored(result)
    assert not {"buffer_param", "caller_seeds", "callers_total"} & set(leaf)


class _NamedBV(FBV):
    """A view that resolves symbols by name, as Binary Ninja's does."""

    def __init__(self, names, funcs, symbols):
        super().__init__(names, funcs)
        self._symbols = symbols

    def get_symbols_by_name(self, name):
        return [type("S", (), {"address": a})() for a in self._symbols.get(name, [])]


def _outer_program():
    """`outer` calls `mid` directly and dispatches one indirect call; `pull` is
    called by `mid` (0x2000) and by `side` (0x1500, a lower address)."""
    mid, side = _caller("mid", 0x2000), _caller("side", 0x1500)
    pull = FFunc("pull", PULL, FSSAFunc([]),
                 caller_sites=[FSite(side, 0x1510), FSite(mid, 0x2010)])
    slot = FSSA(FVar("slot"), 1)
    instrs = [
        _ext_call(0, 0x6000, "mid()", 0x2000, []),
        FInstr(1, 0x6004, "MLIL_CALL_SSA", "slot#1()", reads=[slot],
               dest=FExpr("MLIL_VAR_SSA", "slot#1", reads=[slot]), params=[]),
    ]
    outer = FFunc("outer", 0x6000, FSSAFunc(instrs))
    bv = _NamedBV({0x2000: "mid", 0x1500: "side", PULL: "pull"},
                  {0x2000: mid, 0x1500: side, PULL: pull, 0x6000: outer},
                  {"pull": [PULL]})
    return bv, outer


@pytest.mark.parametrize("source", ["call:pull", "ret:pull"])
def test_seed_above_the_direct_caller_is_refused_with_its_callers(models, source):
    bv, outer = _outer_program()
    models = dict(models, pull={"sources": [{"to": "*arg:1"}, {"to": "ret"}]})

    with pytest.raises(te.TaintError) as err:
        te.TaintEngine(bv, models).forward(outer, [te.parse_locator(source)])

    offered = re.findall(r"-f (0x[0-9a-f]+)", str(err.value))
    # The caller `outer` itself calls leads, then the rest by callsite address.
    assert offered == ["0x2000", "0x1500"]
