"""Confidence-aware prototype advice must preserve incomplete-flow disclosure."""
from __future__ import annotations

import types

import pytest

from _bridge_fakes import _load_bridge
from test_read_taint_slice import _BV, _FMT_STR, _add, _call, _const, _fmt_ptr, _func, _reg, _store
from test_taint_engine import _variadic_logger_program, models, te


@pytest.mark.parametrize("variadic", [True, False, None])
def test_forward_total_handoff_confidence_advice(models, variadic):
    handler, bv, _, _ = _variadic_logger_program(variadic)
    result = te.TaintEngine(bv, models).forward(handler, [te.parse_locator("param:0")])
    leaf = next(l for l in result["leaves"] if l["kind"] == "unmodeled_callee")
    assert leaf["callee_variadic"] is variadic
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    for text in (leaf["note"], result["diagnostics"]["next_action"]):
        assert ("proto set" in text) is (variadic is False)
        if variadic is None:
            assert "inspect" in text.lower()


@pytest.mark.parametrize("variadic", [True, False, None])
def test_forward_partial_handoff_confidence_advice(models, monkeypatch, variadic):
    _, bv, callee, call = _variadic_logger_program(variadic)
    callee.calling_convention = types.SimpleNamespace(int_arg_regs=["x0", "x1"])
    engine = te.TaintEngine(bv, models)
    monkeypatch.setattr(engine, "_reg_reads_as_input", lambda fn, reg: True)
    leaf = engine._arg_under_recovered_leaf(call, callee, [1, 3], 1)
    assert leaf["dropped_args"] == [1, 3]
    assert leaf["callee_variadic"] is variadic
    assert ("proto set" in leaf["note"]) is (variadic is False)
    diagnostics = te.forward_zero_diagnostics(
        {"leaves": [leaf], "assumptions": [], "diag": {"tainted_values": 2}},
        seed_callsites=1)
    assert diagnostics["safe_to_report_all_clear"] is False
    assert ("proto set" in diagnostics["next_action"]) is (variadic is False)


@pytest.mark.parametrize("variadic", [True, False, None])
def test_stack_candidates_do_not_prove_prototype_discrepancy(monkeypatch, variadic):
    rts = _load_bridge(monkeypatch).read_taint_slice
    callee = types.SimpleNamespace(type=types.SimpleNamespace(has_variable_arguments=variadic))
    monkeypatch.setattr(rts._taint, "resolve_call_target", lambda *a, **k: types.SimpleNamespace(function=callee))
    block = [_store(_reg("sp"), 0x10), _store(_add(_reg("sp"), _const(4)), 0x12), _call(0x1c)]
    note = rts._call_model_truncation_note(_BV(_FMT_STR), _func(block), _call(0x1c), 0x1c, [_fmt_ptr(0x5000)], "app_log")
    assert note is not None
    assert "candidate" in note and "not proven to feed the call" in note
    assert "declare it variadic" not in note
    assert "cannot see them until the prototype" not in note
    if variadic is None:
        assert "proto set" not in note

@pytest.mark.parametrize("variadic", [True, False, None])
def test_backward_out_of_range_confidence_advice(models, monkeypatch, variadic):
    handler, bv, callee, call = _variadic_logger_program(variadic)
    engine = te.TaintEngine(bv, models)
    monkeypatch.setattr(engine, "_find_callsites", lambda *a, **k: [call])
    monkeypatch.setattr(engine, "_note_indirect_anchors", lambda *a: None)
    monkeypatch.setattr(te, "resolve_call_target", lambda *a, **k: types.SimpleNamespace(function=callee))
    with pytest.raises(te.TaintError) as error:
        engine._seed_backward(handler, None, [call], te.parse_locator("arg:app_log:4"))
    text = str(error.value)
    assert "out of range" in text
    assert ("proto set" in text) is (variadic is False)
    if variadic is True:
        assert "already declared variadic" in text
    elif variadic is None:
        assert "undetermined" in text


@pytest.mark.parametrize("variadic", [True, False, None])
def test_shared_out_of_range_confidence_advice(monkeypatch, variadic):
    rts = _load_bridge(monkeypatch).read_taint_slice
    text = rts._out_of_range_arg_msg(
        _func([], name="handler"), 4, 1, callee_variadic=variadic,
        variadic_callee="app_log" if variadic is True else None)
    assert "out of range" in text and "1 MLIL argument" in text
    assert "declare it variadic" not in text
    if variadic is True:
        assert "not a prototype defect" in text
    elif variadic is False:
        assert "independent argument provenance" in text
    else:
        assert "undetermined" in text


def test_correct_varargs_partial_handoff_retains_unread_pure_stack_frontier(models, monkeypatch):
    _, bv, callee, call = _variadic_logger_program(True)
    callee.calling_convention = types.SimpleNamespace(int_arg_regs=[])
    engine = te.TaintEngine(bv, models)
    monkeypatch.setattr(engine, "_reg_reads_as_input", lambda *a: False)
    leaf = engine._arg_under_recovered_leaf(call, callee, [1], 1)
    assert leaf["cause"] == "varargs_not_followed"
    assert leaf["dropped_args"] == [1]


def test_aggregate_mixed_frontiers_do_not_prescribe_prototype_repair(models):
    sub = {"leaves": [
        {"kind": "arg_under_recovered", "cause": "varargs_not_followed"},
        {"kind": "unmodeled_callee", "cause": "prototype_status_unknown"},
    ], "assumptions": [], "findings": []}
    diagnostics = te.forward_zero_diagnostics(sub, seed_callsites=1)
    assert diagnostics["safe_to_report_all_clear"] is False
    assert "proto set" not in diagnostics["next_action"]

@pytest.mark.parametrize("value", [True, False])
def test_unsettled_variadic_flag_keeps_unknown_forward_advice(models, value):
    handler, bv, callee, _ = _variadic_logger_program(
        types.SimpleNamespace(value=value, confidence=0))
    result = te.TaintEngine(bv, models).forward(handler, [te.parse_locator("param:0")])
    leaf = next(l for l in result["leaves"] if l["kind"] == "unmodeled_callee")
    assert leaf["callee_variadic"] is None
    assert leaf["cause"] == "prototype_status_unknown"
    assert "proto set" not in leaf["note"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("variadic", [True, False, None])
def test_missing_named_parameter_confidence_advice(models, direction, variadic):
    _, bv, callee, _ = _variadic_logger_program(variadic)
    engine = te.TaintEngine(bv, models)
    with pytest.raises(te.TaintError) as error:
        getattr(engine, direction)(callee, [te.parse_locator("param:4")])
    text = str(error.value)
    assert "parameter 4 not found" in text
    assert "1 parameter(s)" in text
    assert ("proto set" in text) is (variadic is False)
    if variadic is True:
        assert "already declared variadic" in text
        assert "varargs limitation" in text
    elif variadic is False:
        assert "Only if independent argument evidence" in text
        assert "declaration discrepancy" in text
    else:
        assert "undetermined" in text and "inspect" in text


@pytest.mark.parametrize("cause", [
    "varargs_not_followed", "signature_under_recovered", "prototype_status_unknown",
])
def test_backward_aggregate_preserves_prototype_confidence(cause):
    from bn_agent_bridge import taint_result as tr

    diagnostics = tr.backward_diagnostics(
        [{"kind": "arg_under_recovered", "cause": cause}], [],
        sinks_seeded=1, slices=1)
    assert diagnostics["safe_to_report_complete_slice"] is False
    assert diagnostics["frontier"]["unresolved"] == 1
    text = diagnostics["next_action"]
    assert ("proto set" in text) is (cause == "signature_under_recovered")
    if cause == "varargs_not_followed":
        assert "already declared variadic" in text
        assert "does not follow" in text
    elif cause == "signature_under_recovered":
        assert "only where independent argument evidence" in text
    else:
        assert "inspect" in text


def test_backward_known_buffer_writer_aggregate_recommends_model():
    from bn_agent_bridge import taint_result as tr

    diagnostics = tr.backward_diagnostics(
        [{"kind": "address_of_local_unresolved", "callee": "fill_buffer",
          "buffer": "local_buf", "note": "model the call or seed inside it"}],
        [], sinks_seeded=1, slices=1, unresolved_origins=1)
    assert diagnostics["safe_to_report_complete_slice"] is False
    assert "model" in diagnostics["next_action"]
    assert "proto set" not in diagnostics["next_action"]


def test_backward_mixed_and_unknown_frontiers_keep_neutral_advice():
    from bn_agent_bridge import taint_result as tr

    for leaves in (
        [{"kind": "field_load_unresolved"}],
        [{"kind": "arg_under_recovered", "cause": "varargs_not_followed"},
         {"kind": "unmodeled_callee", "cause": "prototype_status_unknown"}],
    ):
        diagnostics = tr.backward_diagnostics(
            leaves, [], sinks_seeded=1, slices=1)
        assert diagnostics["safe_to_report_complete_slice"] is False
        assert "inspect" in diagnostics["next_action"]
        assert "proto set" not in diagnostics["next_action"]


@pytest.mark.parametrize("value", [True, False])
def test_unsettled_variadic_flag_keeps_missing_parameter_advice_neutral(models, value):
    _, bv, callee, _ = _variadic_logger_program(
        types.SimpleNamespace(value=value, confidence=0))
    text = str(te.TaintEngine(bv, models)._param_not_found_error(callee, 4))
    assert "undetermined" in text
    assert "proto set" not in text
