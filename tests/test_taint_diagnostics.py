from __future__ import annotations

import types

import pytest

from test_taint_engine import (
    FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar,
    _popen_of_local_program, te,
)
from test_read_taint_slice import _load_bridge


class NamedBV(FBV):
    def get_symbol_by_raw_name(self, name):
        if name in self._names.values() and name.startswith("_Z"):
            return types.SimpleNamespace(name=name, short_name="app::fill")
        return None


@pytest.mark.parametrize("mangled", [False, True])
def test_external_caveat_names_callee_in_selected_mode(mangled):
    raw = "_ZN3app4fillEPc"
    p = FVar("p", ident=1)
    p0 = FSSA(p, 0)
    call = FInstr(0, 0x1010, "MLIL_CALL_SSA", "callee(p)", reads=[p0],
                  params=[FExpr("MLIL_VAR_SSA", "p", reads=[p0])],
                  dest=FExpr("MLIL_CONST_PTR", "0x2000", constant=0x2000))
    function = FFunc("handler", 0x1000, FSSAFunc([call]), params=[p])
    bv = NamedBV({0x2000: raw})
    result = te.TaintEngine(bv, te.load_models(), mangled=mangled).forward(
        function, [te.parse_locator("param:0")])
    caveat = next(a for a in result["assumptions"] if "has no model" in a)
    assert (raw in caveat) is mangled
    assert ("app::fill" in caveat) is not mangled
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("mangled", [False, True])
def test_backward_opaque_writer_names_prose_without_changing_key(mangled):
    raw = "_ZN3app4fillEPc"
    function, original = _popen_of_local_program(fill=raw)
    bv = NamedBV(original._names)
    result = te.TaintEngine(bv, te.load_models(), mangled=mangled).backward(
        function, [te.parse_locator("arg:popen:0")])
    leaf = next(l for l in result["leaves"] if l["kind"] == "address_of_local_unresolved")
    assert leaf["callee"] == raw
    assert (raw in leaf["detail"]) is mangled
    assert ("app::fill" in leaf["detail"]) is not mangled
    reasons = "\n".join(step.get("reason", "") for row in result["slices"] for step in row["slice"])
    assert (raw in reasons) is mangled
    assert ("app::fill" in reasons) is not mangled
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("mangled", [False, True])
def test_trace_out_parameter_caveat_names_selected_mode(monkeypatch, mangled):
    rts = _load_bridge(monkeypatch).read_taint_slice
    raw = "_ZN3app4fillEPc"
    caveats = rts._trace_assumptions(
        [{"reason": "out_param_not_followed", "out_param_callee": raw,
          "out_param_callee_display_name": "app::fill"}],
        truncated=False, max_depth=10, mangled=mangled)
    assert (raw in caveats[0]) is mangled
    assert ("app::fill" in caveats[0]) is not mangled


@pytest.mark.parametrize("variadic", [True, False, None])
def test_modeled_sink_recovery_advice_respects_variadic_status(monkeypatch, variadic):
    from bn.formatters import _render_taint_text
    from test_taint_engine import _underrecovered_forward_sink_program

    function, length = _underrecovered_forward_sink_program()
    call = function.mlil.ssa_form.instructions[2]
    callee = FFunc("app_copy", 0x900, FSSAFunc([]),
                   params=[FVar("destination")], symbol_type="ImportedFunctionSymbol")
    callee.type = types.SimpleNamespace(has_variable_arguments=variadic)
    bv = FBV({0x900: "app_copy"}, {0x900: callee})
    models = te.load_models({"app_copy": {
        "sink": {"class": "overflow_len", "len_arg": 2, "buf_arg": 0}}})
    engine = te.TaintEngine(bv, models)
    monkeypatch.setattr(engine, "_reaching_arg_seeds_via_reg",
                        lambda f, sites, idx: ("rdx", [(length, call)]))
    result = engine.forward(function, [te.parse_locator("param:0")])
    sink = result["reached_sinks"][0]["sink"]
    assert sink["class"] == "overflow_len" and sink["via"] == "abi_register"
    assert sink["arity_mismatch"]["callee_variadic"] is variadic
    assert sink["arity_mismatch"]["dropped"] == [2]
    for text in (sink["detail"], _render_taint_text(result),
                 _render_taint_text(result, full=True)):
        assert ("proto set" in text) is (variadic is False)
        assert "unproven" in text
        if variadic is True:
            assert "already declared variadic" in text
        elif variadic is None:
            assert "undetermined" in text


def _unwound_forward(monkeypatch, salvaged: list[dict]) -> dict:
    """A forward run whose walk hits the recursion limit after salvaging
    *salvaged* -- the shape a deep alias scan produced on large entry points."""
    p = FVar("p", ident=1)
    function = FFunc("dispatch", 0x1000,
                     FSSAFunc([FInstr(0, 0x1004, "MLIL_RET", "return", reads=[])]),
                     params=[p])
    engine = te.TaintEngine(FBV({}), te.load_models())

    def unwind(*_a, **_k):
        engine._salvage["leaves"].extend(salvaged)
        raise RecursionError

    monkeypatch.setattr(engine, "_run_forward", unwind)
    return engine.forward(function, [te.parse_locator("param:0")])


def test_unwound_walk_seed_reach_is_unmeasured_and_propagation_read_off_the_result(monkeypatch):
    continuation = {"kind": "global_memory_continuation", "address": "0x1010",
                    "callee": {"name": "flush_log", "address": "0x2000"},
                    "global": {"address": "0x6010", "label": "log_buf"}}
    result = _unwound_forward(monkeypatch, [continuation])
    assert result["stats"]["truncation_cause"] == ["recursion"]
    diag = result["diagnostics"]
    # Nothing was measured, so no count is claimed ...
    assert diag["tainted_values"] is None
    # ... and the continuation leaf in the same result is proof of propagation.
    assert diag["propagated"] is True
    assert diag["safe_to_report_all_clear"] is False


def test_unwound_walk_with_only_its_marker_does_not_claim_propagation(monkeypatch):
    result = _unwound_forward(monkeypatch, [])
    diag = result["diagnostics"]
    assert diag["tainted_values"] is None
    assert diag["propagated"] is False


def test_finished_dead_seed_is_measured_and_not_propagated():
    # The control: a walk that FINISHED and whose seed reached nothing keeps
    # its measured count, and the result says it did not propagate.
    diag = te.forward_zero_diagnostics(
        {"leaves": [], "assumptions": [], "diag": {"tainted_values": 1, "last_use": None}},
        seed_callsites=0)
    assert diag["tainted_values"] == 1
    assert diag["propagated"] is False


def test_forward_text_never_reports_a_dead_seed_beside_propagation_proof(monkeypatch):
    from bn.formatters import _render_taint_text

    continuation = {"kind": "global_memory_continuation", "address": "0x1010",
                    "callee": {"name": "flush_log", "address": "0x2000"},
                    "global": {"address": "0x6010", "label": "log_buf"}}
    proven = _render_taint_text(_unwound_forward(monkeypatch, [continuation])).splitlines()
    dead = _render_taint_text({
        "direction": "forward", "function": {"name": "dispatch", "address": "0x1000"},
        "sources": ["param:0"], "reached_sinks": [], "leaves": [], "stats": {},
        "diagnostics": te.forward_zero_diagnostics(
            {"leaves": [], "assumptions": [], "diag": {"tainted_values": 1, "last_use": None}},
            seed_callsites=0)}).splitlines()

    def line(lines: list[str], head: str) -> str:
        return next(x for x in lines if x.strip().startswith(head))

    # The seed and last-use lines of a result that proves propagation differ
    # from those of a finished dead seed: the claim follows the result.
    assert line(proven, "seed:") != line(dead, "seed:")
    assert line(proven, "last propagated use:") != line(dead, "last propagated use:")


@pytest.mark.parametrize("cause, widens", [("depth_budget", True), ("function_budget", False),
                                           ("object_extent_unproven", False)])
def test_continuation_advice_offers_max_depth_only_for_the_depth_bound(cause, widens):
    # `--max-depth` moves the call-graph bound and nothing else: the
    # reachability search's own function budget is internal, and advising the
    # flag for it was advice that could not change the next run.
    leaf = {"kind": "global_memory_continuation", "address": "0x1010",
            "callee": {"name": "flush_log", "address": "0x2000"},
            "global": {"address": "0x6010", "label": "log_buf"}, "references": [],
            "reachability": {"status": "truncated", "cause": cause}}
    diag = te.forward_zero_diagnostics(
        {"leaves": [leaf], "assumptions": [], "diag": {"tainted_values": 3, "last_use": None}},
        seed_callsites=0, function="0x1000")
    assert ("raise --max-depth" in diag["next_action"]) is widens
