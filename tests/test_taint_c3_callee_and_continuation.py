"""Taint naming residuals (C3): one `callee` rule, and the global-continuation
leaf's label and reseed.

* A fortified `__*_chk` call named its finding by the model key
  (`callee: "strcpy_chk"`, no `model`), while an ISO scanf alias named the
  symbol it calls (`callee: "__isoc99_scanf"`, `model: "scanf"`), so a
  consumer could not know which one `callee` held (V4-03). `callee` is now
  always the called symbol, `model` the model key when it differs -- for
  reached sinks, frontier leaves and backward origins alike.
* A `global_memory_continuation` leaf labelled a global that has a data
  symbol `glob_0x...`, and its `reseed` seeded the reader's own sink
  argument, a run that re-finds that one call in one step (W07-13). The
  label is the symbol's name; a seed is the reader's input or nothing.
"""
from __future__ import annotations

import pytest

from test_taint_engine import FBVStr, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, _ext_call, te
from test_taint_forward_global_frontier import (
    G_DIR, MEMCPY, STRNCPY, SYSTEM, FGlobalBV, FRef, GFunc, _const, _copy_then_call,
    _globalflow, _gptr, _leaves, _var,
)

from bn.formatters import _render_grouped_leaves

CALL, MALLOC, FMT = 0x900, 0x901, 0x700


class _SymBV(FBVStr):
    """Import symbols resolve by raw name, as a real view's do."""

    def get_symbol_by_raw_name(self, name):
        if name in self._names.values():
            return type("Sym", (), {"name": name, "short_name": name})()
        return None


def _ptr(v, text):
    return FExpr("MLIL_VAR_SSA", text, reads=[v])


def _strcpy_shape(callee):
    """copy(s): <callee>(&d, s#0[, 0x10]) -- the tainted source is the sink operand."""
    s, d = FVar("s"), FVar("d", typ="char[0x10]")
    s0 = FSSA(s, 0)
    params = [FExpr("MLIL_ADDRESS_OF", "&d", src=d), _ptr(s0, "s#0")]
    if callee.endswith("_chk"):
        params.append(FExpr("MLIL_CONST", "0x10", constant=0x10))
    instrs = [_ext_call(0, 0x10, f"{callee}(&d, s#0)", CALL, params, reads=[s0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    return FFunc("copy", 0x10, FSSAFunc(instrs), params=[s]), _SymBV({CALL: callee})


def _sscanf_shape(callee):
    """parse(s): <callee>(s#0, "%s", &d) -- a width-less %s is unbounded."""
    s, d = FVar("s"), FVar("d", typ="char[0x10]")
    s0 = FSSA(s, 0)
    instrs = [_ext_call(0, 0x10, f"{callee}(s#0, fmt, &d)", CALL,
                        [_ptr(s0, "s#0"), FExpr("MLIL_CONST_PTR", "fmt", constant=FMT),
                         FExpr("MLIL_ADDRESS_OF", "&d", src=d)], reads=[s0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    return (FFunc("parse", 0x10, FSSAFunc(instrs), params=[s]),
            _SymBV({CALL: callee}, strings={FMT: "%s"}))


@pytest.mark.parametrize(("shape", "callee", "model"), [
    (_strcpy_shape, "__strcpy_chk", "strcpy_chk"),   # fortified: underscores stripped
    (_sscanf_shape, "__isoc99_sscanf", "sscanf"),    # ISO alias: folded onto the plain model
    (_strcpy_shape, "strcpy", None),                 # the key IS the symbol: no `model`
])
def test_a_reached_sink_names_the_called_symbol_and_the_model_beside_it(shape, callee, model):
    func, bv = shape(callee)
    result = te.TaintEngine(bv, te.load_models()).forward(func, [te.parse_locator("param:0")])
    [finding] = result["reached_sinks"]
    sink = finding["sink"]
    assert (sink["callee"], sink["callee_symbol"], sink.get("model")) == (callee, callee, model)
    assert ("model" in sink) is (model is not None)
    # The path's sink step and the signature name the same symbol.
    assert finding["path"][-1]["callee"] == callee
    assert finding["signature"]["sink_callee"] == callee


def test_a_gated_sink_leaf_names_the_called_symbol_and_its_model():
    # `__read_chk(fd, &b, n, 16)` arms the opt-in `recv_overflow` class through
    # its length: without `--sink-class` the leaf discloses it.
    n, b = FVar("n"), FVar("b", typ="char[0x10]")
    n0 = FSSA(n, 0)
    instrs = [_ext_call(0, 0x10, "__read_chk(0, &b, n#0, 0x10)", CALL,
                        [_const(0), FExpr("MLIL_ADDRESS_OF", "&b", src=b), _ptr(n0, "n#0"),
                         _const(0x10)], reads=[n0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    func = FFunc("fill", 0x10, FSSAFunc(instrs), params=[n])
    result = te.TaintEngine(_SymBV({CALL: "__read_chk"}), te.load_models()).forward(
        func, [te.parse_locator("param:0")])
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "gated_sink_reached"]
    assert (leaf["callee"], leaf.get("model")) == ("__read_chk", "read_chk"), leaf


def test_a_backward_origin_names_the_called_symbol_and_its_model():
    # `n = __read_chk(0, &b, 16, 16); malloc(n)`: the length's origin is the
    # fortified source call.
    b = FVar("b", typ="char[0x10]")
    n1 = FSSA(FVar("n"), 1)
    instrs = [_ext_call(0, 0x10, "n#1 = __read_chk(0, &b, 0x10, 0x10)", CALL,
                        [_const(0), FExpr("MLIL_ADDRESS_OF", "&b", src=b), _const(0x10),
                         _const(0x10)], writes=[n1]),
              _ext_call(1, 0x14, "malloc(n#1)", MALLOC, [_ptr(n1, "n#1")], reads=[n1]),
              FInstr(2, 0x18, "MLIL_RET", "return")]
    func = FFunc("size", 0x10, FSSAFunc(instrs), params=[])
    bv = _SymBV({CALL: "__read_chk", MALLOC: "malloc"})
    result = te.TaintEngine(bv, te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])
    origins = [(sl["origin"].get("kind"), sl["origin"].get("callee"), sl["origin"].get("model"))
               for sl in result["slices"]]
    assert origins == [("source", "__read_chk", "read_chk")], result["slices"]


@pytest.mark.parametrize(("shape", "callee", "locator", "model", "status"), [
    # The ISO alias queried by its plain model name: seeded at the call.
    (_sscanf_shape, "__isoc99_sscanf", "arg:sscanf:2", "sscanf", "seeded"),
    # The fortified symbol queried by its own name; its size is a constant.
    (_strcpy_shape, "__strcpy_chk", "arg:__strcpy_chk:2", "strcpy_chk", "constant"),
    # The model key IS the symbol: no `model`.
    (_strcpy_shape, "strcpy", "arg:strcpy:1", None, "seeded"),
])
def test_a_backward_call_site_row_names_the_called_symbol_and_its_model(
        shape, callee, locator, model, status):
    """R3 residual: `slices[].sink` and `sink_status[]` describe one call site
    but kept the locator's spelling -- `callee: "sscanf"`, no `model` -- for a
    call to `__isoc99_sscanf`. They follow the forward rule now; only the
    top-level `sinks[]` echoes the query as written."""
    func, bv = shape(callee)
    query = te.parse_locator(locator)
    result = te.TaintEngine(bv, te.load_models()).backward(func, [query])
    assert [s["callee"] for s in result["sinks"]] == [query["callee"]]
    [row] = result["sink_status"]
    assert (row["status"], row["callee"], row.get("model")) == (status, callee, model)
    assert [(sl["sink"]["callee"], sl["sink"].get("model")) for sl in result["slices"]] == (
        [(callee, model)] if status == "seeded" else [])


@pytest.mark.parametrize(("symbol", "callee", "model"), [
    ("__strdup", "__strdup", "strdup"),     # the model key differs: named beside it
    ("malloc@plt", "malloc", None),         # a decorated import: the bare symbol
])
def test_a_backward_call_origin_names_the_called_symbol_and_its_model(symbol, callee, model):
    """`n = <call>(0x10); malloc(n)`: the returned value's origin is the call,
    named like every other origin (R3 residual)."""
    n1 = FSSA(FVar("n"), 1)
    instrs = [_ext_call(0, 0x10, f"n#1 = {symbol}(0x10)", CALL, [_const(0x10)], writes=[n1]),
              _ext_call(1, 0x14, "malloc(n#1)", MALLOC, [_ptr(n1, "n#1")], reads=[n1]),
              FInstr(2, 0x18, "MLIL_RET", "return")]
    func = FFunc("size", 0x10, FSSAFunc(instrs), params=[])
    result = te.TaintEngine(_SymBV({CALL: symbol, MALLOC: "malloc"}), te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])
    assert [(sl["origin"]["kind"], sl["origin"]["callee"], sl["origin"].get("model"))
            for sl in result["slices"]] == [("call", callee, model)]


# --------------------------------------------------------------------------
# W07-13: the continuation leaf's label and reseed
# --------------------------------------------------------------------------

class _NamedGlobalBV(FGlobalBV):
    """A view whose tainted global has a data symbol."""

    def get_symbol_at(self, addr):
        if int(addr) == G_DIR:
            return type("Sym", (), {"name": "g_name", "short_name": "g_name"})()
        return None


def _forward(bv, func, **kw):
    return te.TaintEngine(bv, te.load_models(), **kw).forward(func, [te.parse_locator("param:0")])


@pytest.mark.parametrize(("bv_cls", "label"), [
    (_NamedGlobalBV, "g_name"), (FGlobalBV, f"glob_{hex(G_DIR)}")])
def test_the_continuation_label_is_the_globals_data_symbol(bv_cls, label):
    func, _hook, bv = _globalflow(extra_copy_params=[_const(0xff)], bv_cls=bv_cls)
    [leaf] = _leaves(_forward(bv, func))
    assert leaf["global"]["address"] == hex(G_DIR)
    assert leaf["global"]["label"] == label


def test_a_reader_that_hands_the_global_straight_to_a_sink_gets_no_reseed():
    # run_hook(): system(g) -- the constant address IS the sink's argument, so
    # `arg:system:0` would re-find that one call in one step. No seed, no
    # reseed, and the text follow-up is the xrefs audit.
    func, _hook, bv = _globalflow(extra_copy_params=[_const(0xff)])
    [leaf] = _leaves(_forward(bv, func))
    assert [r["seed"] for r in leaf["references"]] == [None], leaf["references"]
    assert "reseed" not in leaf, leaf
    nexts = [row.strip() for row in _render_grouped_leaves([leaf]) if "next:" in row]
    assert len(nexts) == 1 and f"xrefs {hex(G_DIR)}" in nexts[0], nexts


def _named_locals(*names):
    """The seeder's variable resolver: only *names* are nameable locals."""
    def find(_func, selector):
        if selector in names:
            return FVar(selector)
        raise RuntimeError(f"Variable not found: {selector}")
    return find


@pytest.mark.parametrize("nameable", [True, False])
def test_a_reader_that_loads_the_global_is_reseeded_on_the_loaded_value(nameable):
    # reader(): rax = *(int *)g; n = rax; malloc(n) -- the loaded value is the
    # reader's input. The register temporary is no variable `--source` can
    # name, so the seed follows the plain copy to `n`; a value no resolvable
    # variable holds gets no seed and no reseed.
    rax, n = FVar("rax"), FVar("n")
    rax1, n1 = FSSA(rax, 1), FSSA(n, 1)
    reader = GFunc("reader", 0x401200, FSSAFunc([
        FInstr(0, 0x401210, "MLIL_SET_VAR_SSA", f"rax#1 = [{hex(G_DIR)}]", writes=[rax1],
               dest=rax1, src=FExpr("MLIL_LOAD_SSA", f"[{hex(G_DIR)}]", src=_gptr(G_DIR))),
        FInstr(1, 0x401214, "MLIL_SET_VAR_SSA", "n#1 = rax#1", reads=[rax1], writes=[n1],
               dest=n1, src=_var(rax1)),
        FInstr(2, 0x401218, "MLIL_CALL_SSA", "malloc(n#1)", reads=[n1],
               dest=_gptr(MALLOC), params=[_var(n1)]),
    ]), params=[])
    func, _argv = _copy_then_call(0x401104, STRNCPY, G_DIR,
                                  call_sites=[(0x401108, _gptr(0x401200), [])],
                                  params_extra=[_const(0xff)])
    bv = FGlobalBV({STRNCPY: "strncpy", MALLOC: "malloc", 0x401200: "reader"},
                   funcs={0x401200: reader}, refs={G_DIR: [FRef(reader, 0x401210)]})
    find = _named_locals(*(["n"] if nameable else []))
    [leaf] = _leaves(_forward(bv, func, find_variable=find))
    [row] = leaf["references"]
    if not nameable:
        assert (row["access"], row["seed"]) == ("load", None), row
        assert "reseed" not in leaf, leaf
        return
    assert (row["access"], row["seed"]) == ("load", "var:n"), row
    assert leaf["reseed"] == f"`bn taint forward -f {hex(0x401200)} --source var:n`"

    inside = te.TaintEngine(bv, te.load_models(),
                            find_variable=lambda _f, sel: n if sel == "n" else None
                            ).forward(reader, [te.parse_locator("var:n")])
    [hit] = inside["reached_sinks"]
    assert (hit["sink"]["callee"], hit["sink"]["address"]) == ("malloc", hex(0x401218)), hit
