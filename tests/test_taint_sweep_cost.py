"""What a forward run's global-continuation sweep costs (P1-01).

Binary Ninja regenerates a function's IL each time a fresh wrapper asks for it
while nothing holds the last one, so on a real view every repeated lift of one
body is paid again -- the request lifts each body once and keeps it. And a
body that cannot obtain the tainted object's address, and whose every call BN
resolved, is walked on its call edges alone whatever lies below it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_taint_claim_gaps import FEdgeBV, G, G_SIZE, _copy_then_calls
from test_taint_engine import FInstr, FSSAFunc, te
from test_taint_forward_global_frontier import _behind, GFunc, _gptr, _reader

_INT = SimpleNamespace(width=4)      # a declared return no address fits in


@pytest.fixture
def models():
    return te.load_models()


def _relay_chain(depth):
    """`main` fills G, then calls `relay_0`; each relay calls the next, and the
    last calls `consumer`, the one body that reads G. No relay names G or can
    hand an address back (each returns an `int`), and BN resolved every call."""
    consumer = _reader("consumer", 0x402000, G, 0x402010)
    consumer.return_type = _INT
    chain, nxt = [], consumer
    for i in reversed(range(depth)):
        start = 0x401300 + 0x40 * i
        relay = GFunc(f"relay_{i}", start, FSSAFunc([
            FInstr(0, start + 4, "MLIL_CALL_SSA", f"{nxt.name}()", dest=_gptr(nxt.start),
                   params=[])]), params=[], callees=[nxt])
        relay.return_type = _INT
        chain.insert(0, relay)
        nxt = relay
    func = _copy_then_calls(chain[0])
    edges = {}
    for body, callee in ((func, chain[0]), *zip(chain, chain[1:] + [consumer])):
        site = next(int(i.address) for i in body.mlil.ssa_form.instructions
                    if "CALL" in i.operation.name
                    and int(getattr(i.dest, "constant", -1)) == callee.start)
        body.call_sites = [*getattr(body, "call_sites", []), SimpleNamespace(address=site)]
        edges[site] = [callee.start]
        callee.callers = [*getattr(callee, "callers", []), body]
    consumer.call_sites = []
    func.callers = []
    names = {0x401060: "memcpy", 0x401070: "system", consumer.start: "consumer",
             **{r.start: r.name for r in chain}}
    funcs = {consumer.start: consumer, **{r.start: r for r in chain}}
    bv = FEdgeBV(names, funcs=funcs,
                 refs={G: [SimpleNamespace(function=consumer, address=0x402010)]},
                 objects={G: G_SIZE}, edges=edges)
    return func, bv, chain


def test_bodies_on_the_way_to_a_reader_are_walked_on_their_edges(models, monkeypatch):
    # Each relay is an ANCESTOR of the reader but cannot see the object
    # itself: the walk goes through it on BN's edges without lifting it.
    func, bv, chain = _relay_chain(6)
    lifted = []
    original = te.TaintEngine._not_internal_reason

    def spy(self, fn):
        lifted.append(getattr(fn, "name", None))
        return original(self, fn)

    monkeypatch.setattr(te.TaintEngine, "_not_internal_reason", spy)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert [r["display_name"] for r in leaf["references"]] == ["consumer"]
    assert leaf["reachability"]["status"] == "complete", leaf["reachability"]
    # The call's own target is classified where the walk meets it; the
    # relays past it are not.
    assert not {r.name for r in chain[1:]} & set(lifted), lifted


class CountedMLIL:
    """A function's MLIL whose SSA form counts how often it is fetched."""

    def __init__(self, ssa, counts, name):
        self._ssa, self._counts, self._name = ssa, counts, name
        self.instructions = ssa.instructions

    @property
    def ssa_form(self):
        self._counts[self._name] = self._counts.get(self._name, 0) + 1
        return self._ssa


def test_each_body_is_lifted_once_per_request(models):
    # main -> veneer1 -> veneer2 -> consumer: each veneer is classified (is
    # it a pure tail call?), resolved as a thunk chain, and read again when
    # the reader search walks through it -- one lift each.
    func, bv = _behind([0x401300, 0x401400])
    counts: dict[str, int] = {}
    for body in (func, *(bv.get_function_at(a) for a in (0x401300, 0x401400, 0x401500))):
        body.mlil = CountedMLIL(body.mlil.ssa_form, counts, body.name)
    result = te.TaintEngine(bv, models).forward(func, [te.parse_locator("param:0")])
    assert [lf["kind"] for lf in result["leaves"]] == ["global_memory_continuation"]
    assert counts and max(counts.values()) == 1, counts
