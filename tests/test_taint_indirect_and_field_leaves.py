"""Frontier leaves an agent acts on: an unpinned indirect call and a field load
nothing accounts for.

An ``indirect_call_unresolved`` leaf is one row per call site, whatever
argument contexts reached it, and carries the ``--resolve-map`` document that
pins it (W01-11, W11-08). A ``field_load_unresolved`` leaf names every call on
the load's memory chain that may have written the bytes, by symbol and display
name rather than the raw MLIL spelling (W02-09).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te


@pytest.fixture
def models():
    return te.load_models()


# --------------------------------------------------------------------------
# indirect_call_unresolved: one row per site, with its --resolve-map template
# --------------------------------------------------------------------------

_HELPER = 0x500
_SITE = 0x50c          # helper's unpinned `fp(x, y)`


def _helper() -> FFunc:
    """`relay(x, y)`: `fp#1(x#0, y#0)` through a pointer no value set pins."""
    x, y = FVar("x"), FVar("y")
    x0, y0 = FSSA(x, 0), FSSA(y, 0)
    return FFunc("relay", _HELPER, FSSAFunc([
        FInstr(0, _SITE, "MLIL_CALL_SSA", "fp#1(x#0, y#0)", reads=[x0, y0], writes=[],
               dest=FExpr("MLIL_VAR_SSA", "fp#1", reads=[]),
               params=[FExpr("MLIL_VAR_SSA", "x#0", reads=[x0]),
                       FExpr("MLIL_VAR_SSA", "y#0", reads=[y0])]),
    ]), params=[x, y])


def _relay_twice() -> FFunc:
    """`fan(t)`: `relay(t, 0); relay(0, t)` -- the one indirect call inside
    `relay` is reached from two argument contexts."""
    t = FVar("t")
    t0 = FSSA(t, 0)

    def tv():
        return FExpr("MLIL_VAR_SSA", "t#0", reads=[t0])

    def zero():
        return FExpr("MLIL_CONST", "0", constant=0)

    def call(i, addr, params):
        return FInstr(i, addr, "MLIL_CALL_SSA", f"relay({', '.join(map(str, params))})",
                      reads=[t0], writes=[],
                      dest=FExpr("MLIL_CONST_PTR", hex(_HELPER), constant=_HELPER),
                      params=params)

    return FFunc("fan", 0x100, FSSAFunc([
        call(0, 0x104, [tv(), zero()]),
        call(1, 0x108, [zero(), tv()]),
    ]), params=[t])


def _indirect(result):
    return [lf for lf in result["leaves"] if lf["kind"] == "indirect_call_unresolved"]


def test_one_indirect_leaf_per_site_with_every_tainted_argument(models):
    # W11-08: the site is reached with arg 0 tainted from the first call and
    # arg 1 from the second; that is ONE frontier, its argument indexes merged.
    bv = FBV({}, funcs={_HELPER: _helper()})
    result = te.TaintEngine(bv, models).forward(_relay_twice(), [te.parse_locator("param:0")])
    rows = _indirect(result)
    assert [lf["address"] for lf in rows] == [hex(_SITE)], result["leaves"]
    assert rows[0]["tainted_args"] == [0, 1], rows


def test_indirect_leaf_carries_its_resolve_map_template(models):
    # W01-11: the leaf names the --resolve-map document that pins this site.
    bv = FBV({}, funcs={_HELPER: _helper()})
    result = te.TaintEngine(bv, models).forward(_helper(), [te.parse_locator("param:1")])
    (leaf,) = _indirect(result)
    assert leaf["address"] == hex(_SITE)
    assert leaf["tainted_args"] == [1]
    assert leaf["resolve_map_template"] == {hex(_SITE): ["TARGET"]}


def test_next_action_pins_the_only_indirect_frontier(models):
    # W01-11: an unpinned indirect call is the only frontier, so the next step
    # is the --resolve-map run, with the site's template document in it.
    bv = FBV({}, funcs={_HELPER: _helper()})
    result = te.TaintEngine(bv, models).forward(_helper(), [te.parse_locator("param:0")])
    action = result["diagnostics"]["next_action"]
    assert "--resolve-map FILE" in action, action
    assert json.dumps({hex(_SITE): ["TARGET"]}) in action, action
    assert f"bn taint forward -f {hex(_HELPER)} --source SOURCE --resolve-map FILE" in action


def test_next_action_template_names_at_most_eight_sites():
    leaves = [{"kind": "indirect_call_unresolved", "address": hex(0x1000 + 0x10 * i),
               "tainted_args": [0], "resolve_map_template": {hex(0x1000 + 0x10 * i): ["TARGET"]}}
              for i in range(10)]
    diag = te.forward_zero_diagnostics(
        {"leaves": leaves, "assumptions": [], "diag": {"tainted_values": 3}},
        seed_callsites=1, function="0x1000")
    action = diag["next_action"]
    doc = json.dumps({hex(0x1000 + 0x10 * i): ["TARGET"] for i in range(8)})
    assert doc in action, action
    assert hex(0x1000 + 0x10 * 9) not in action, action


def test_an_unmodeled_callee_beside_it_keeps_the_general_advice():
    # Not the only frontier: the resolve-map step is no longer the whole remedy.
    leaves = [{"kind": "indirect_call_unresolved", "address": "0x1010", "tainted_args": [0]},
              {"kind": "unmodeled_callee", "address": "0x1020", "tainted_args": [0],
               "callee": {"name": "parse", "address": "0x2000"}}]
    diag = te.forward_zero_diagnostics(
        {"leaves": leaves, "assumptions": [], "diag": {"tainted_values": 3}},
        seed_callsites=1, function="0x1000")
    assert "--resolve-map FILE" not in diag["next_action"], diag["next_action"]
    assert "bn taint forward -f 0x2000 --source param:0" in diag["next_action"]


# --------------------------------------------------------------------------
# field_load_unresolved: every candidate writer call, by name
# --------------------------------------------------------------------------

_FILL, _PEEK, _WORD, _STAMP = 0xa90, 0xaa0, 0xab0, 0xad0
_MEMCPY = 0xac0
_RAW = {_FILL: "fill_header", _PEEK: "peek_flags", _WORD: "_ZN6Reader8readWordEv",
        _STAMP: "stamp_header", _MEMCPY: "memcpy"}
_SHOWN = {"_ZN6Reader8readWordEv": "Reader::readWord"}


class _Chain:
    """`f(m, src)`: memory-defining stores and calls numbered like BN's memory
    SSA, then ``x#1 = [m + 8].d`` and ``memcpy(d, s, x#1)``."""

    def __init__(self):
        self.m, self.src = FVar("m", ident=1), FVar("src", ident=2)
        self.m0, self.src0 = FSSA(self.m, 0), FSSA(self.src, 0)
        self.instrs: list = []
        self.mem_defs: dict = {}
        self.mem = 0

    def _at(self):
        return len(self.instrs), 0x10 + 4 * len(self.instrs)

    def _field(self):
        a = FExpr("MLIL_ADD", "m#0 + 0x8", reads=[self.m0])
        a.left = FExpr("MLIL_VAR_SSA", "m#0", reads=[self.m0])
        a.right = FExpr("MLIL_CONST", "0x8", constant=8)
        return a

    def _define(self, ins):
        ins.src_memory = self.mem
        self.mem += 1
        ins.dest_memory = self.mem
        self.mem_defs[self.mem] = ins
        self.instrs.append(ins)
        return ins

    def store_field(self):
        i, a = self._at()
        value = FExpr("MLIL_VAR_SSA", "src#0", reads=[self.src0])
        dest = self._field()
        ins = FInstr(i, a, "MLIL_STORE_SSA", f"[{dest}].d = src#0",
                     reads=[self.m0, self.src0], dest=dest, src=value)
        ins.size = 4
        return self._define(ins)

    def call(self, target, arg="m"):
        """`<target>(m#0)` (handed the object) or `<target>(src#0)`, spelled
        as BN's SSA text does: by raw address."""
        var = self.m0 if arg == "m" else self.src0
        i, a = self._at()
        return self._define(FInstr(
            i, a, "MLIL_CALL_SSA", f"{target:#x}({arg}#0) @ mem#{self.mem}", reads=[var],
            dest=FExpr("MLIL_CONST_PTR", hex(target), constant=target),
            params=[FExpr("MLIL_VAR_SSA", f"{arg}#0", reads=[var])]))

    def run(self):
        ld = FExpr("MLIL_LOAD_SSA", "[m#0 + 0x8].d", reads=[self.m0], src=self._field(),
                   src_memory=self.mem)
        ld.size = 4
        x1 = FSSA(FVar("x", ident=3), 1)
        i, a = self._at()
        self.instrs.append(FInstr(i, a, "MLIL_SET_VAR_SSA", "x#1 = [m#0 + 0x8].d",
                                  reads=[self.m0], writes=[x1], src=ld))
        i, a = self._at()
        self.instrs.append(FInstr(i, a, "MLIL_CALL_SSA", "memcpy(d, s, x#1)", reads=[x1],
                                  dest=FExpr("MLIL_CONST_PTR", hex(_MEMCPY), constant=_MEMCPY),
                                  params=[FExpr("MLIL_VAR_SSA", "d", reads=[]),
                                          FExpr("MLIL_VAR_SSA", "s", reads=[]),
                                          FExpr("MLIL_VAR_SSA", "x#1", reads=[x1])]))
        func = FFunc("f", 0x10, FSSAFunc(self.instrs, mem_defs=self.mem_defs),
                     params=[self.m, self.src])
        bv = FBV(_RAW)
        bv.get_symbol_by_raw_name = lambda raw: (
            SimpleNamespace(name=raw, short_name=_SHOWN.get(raw, raw))
            if raw in _RAW.values() else None)
        result = te.TaintEngine(bv, te.load_models()).backward(
            func, [te.parse_locator("arg:memcpy:2")])
        return [lf for lf in result["leaves"] if lf["kind"] == "field_load_unresolved"]


def _site(ins):
    return hex(ins.address)


def test_field_frontier_names_every_call_on_the_chain_by_display_name():
    # W02-09: `fill_header(m)` is the call that really wrote `m->len`, but the
    # nearest call -- the one that stopped the writer walk -- was the only one
    # the leaf named, and by its raw MLIL text. Every call back to the covering
    # store is a candidate; the call before that store is not. The one handed
    # the object leads, then the others nearest first.
    p = _Chain()
    p.call(_STAMP)
    p.store_field()
    fill = p.call(_FILL)
    peek = p.call(_PEEK, arg="src")
    word = p.call(_WORD, arg="src")
    (leaf,) = p.run()
    assert [(w["address"], (w["callee"] or {}).get("display_name"), w["handed_object"])
            for w in leaf["candidate_writers"]] == [
        (_site(fill), "fill_header", True), (_site(word), "Reader::readWord", False),
        (_site(peek), "peek_flags", False)], leaf
    assert leaf["candidate_writers"][1]["callee"] == {
        "name": "_ZN6Reader8readWordEv", "display_name": "Reader::readWord",
        "address": hex(_WORD)}
    assert leaf["candidate_writers_total"] == 3
    assert leaf["candidate_writers_complete"] is True
    assert f"{_WORD:#x}(src#0)" not in leaf["detail"], leaf["detail"]


def test_field_frontier_lists_the_first_eight_writers_and_counts_all():
    p = _Chain()
    calls = [p.call(_FILL) for _ in range(10)]
    (leaf,) = p.run()
    assert [w["address"] for w in leaf["candidate_writers"]] == [
        _site(c) for c in reversed(calls)][:8], leaf
    assert leaf["candidate_writers_total"] == 10
    assert leaf["candidate_writers_complete"] is True
