"""IC02: every selected sink site and origin has its own truthful accounting."""
from __future__ import annotations

import pytest
from bn.formatters import _render_taint_text
from bn_agent_bridge.taint_result import backward_diagnostics
from test_taint_backward_coverage import (
    _NAMES, _MEMCPY, _RECV, _backward, _call, _const, _length_sink, _var,
)
from test_taint_engine import FBV, FExpr, FFunc, FSite, FSSA, FSSAFunc, FVar, te


def _sites(operands):
    n = FVar("n", ident=101)
    n0 = FSSA(n, 0)
    calls = []
    for i, operand in enumerate(operands):
        arg = _var(n0) if operand == "variable" else _const(operand) if isinstance(operand, int) else operand
        params = [_const(0x5000), _const(0x6000)]
        if arg != "missing":
            params.append(arg)
        calls.append(_call(i, 0x810 + 0x10 * i, _MEMCPY, params))
    return FFunc("copy_sites", 0x800, FSSAFunc(calls), params=[n])


@pytest.mark.parametrize("operands, statuses, complete", [
    (["variable", "variable"], ["seeded", "seeded"], True),
    ([16, 4096], ["constant", "constant"], True),
    (["variable", 4096], ["seeded", "constant"], True),
    (["missing", "variable"], ["unseeded", "seeded"], False),
    (["missing", 4096], ["unseeded", "constant"], False),
    ([FExpr("MLIL_CONST_PTR", "fixed address", constant=0x7000), "variable"],
     ["unseeded", "seeded"], False),
    ([FExpr("MLIL_ADD", "fixed expression"), "variable"], ["unseeded", "seeded"], False),
])
def test_arg_locator_accounts_for_every_matching_site(operands, statuses, complete):
    result = _backward(_sites(operands), _NAMES)
    rows = result["sink_status"]
    assert [(r["address"], r["status"]) for r in rows] == [
        (hex(0x810 + 0x10 * i), status) for i, status in enumerate(statuses)]
    assert result["diagnostics"]["sinks_seeded"] == statuses.count("seeded")
    assert result["diagnostics"]["safe_to_report_complete_slice"] is complete
    constants = [r for r in rows if r["status"] == "constant"]
    assert [r["scalar"]["value"] for r in constants] == [v for v in operands if isinstance(v, int)]
    assert all(r["scalar"]["kind"] == "constant" and r["destination_fit"] == "unproved"
               and not r["seeded"] and "bounded" not in r for r in constants)
    text = _render_taint_text(result)
    for row in rows:
        assert row["address"] in text
        assert row["status"] in text
    for row in constants:
        assert hex(row["scalar"]["value"]) in text
    assert "provably bounded" not in text
    if constants:
        assert "capacity" in text
        assert "check that the --sink" not in result["diagnostics"]["next_action"]
    if not complete:
        assert [lf["address"] for lf in result["leaves"] if lf["kind"] == "sink_unseeded"] == ["0x810"]


def test_absent_locator_and_all_missing_still_refuse():
    engine = te.TaintEngine(FBV(_NAMES), te.load_models())
    with pytest.raises(te.TaintError, match="no call"):
        engine.backward(_sites(["variable"]), [te.parse_locator("arg:malloc:0")])
    with pytest.raises(te.TaintError, match="out of range"):
        engine.backward(_sites(["missing", "missing"]), [te.parse_locator("arg:memcpy:2")])


@pytest.mark.parametrize("opcode", ["MLIL_CONST", "MLIL_CONST_PTR"])
def test_constant_pointer_demand_is_not_a_scalar_failure(opcode):
    pointer = FExpr(opcode, "0x7000", constant=0x7000)
    fn = FFunc("run_hook", 0x800, FSSAFunc([_call(0, 0x810, 0x990, [pointer])]))
    result = te.TaintEngine(FBV({0x990: "popen"}), te.load_models()).backward(
        fn, [te.parse_locator("arg:popen:0")])
    [row] = result["sink_status"]
    assert (row["address"], row["status"], row["seeded"]) == ("0x810", "demand", False)
    assert "scalar" not in row and "bounded" not in row
    assert result["diagnostics"]["sinks_seeded"] == 0
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert result["leaves"][0]["kind"] == "pointer_demand_not_certified"
    assert result["leaves"][0]["objects"][0]["address"] == "0x7000"
    text = _render_taint_text(result)
    assert "demand" in text and "0x810" in text
    # These are contradictory scalar failure/capacity claims, not wording pins.
    for claim in ("reads no variable", "no def-chain", "def-chain was not followed", "UNSEEDED SINKS", "provably bounded"):
        assert claim not in text


@pytest.mark.parametrize("origin_kind", ["parameter", "source", "entry"])
@pytest.mark.parametrize("same_function", [False, True])
def test_origin_identity_survives_public_slice_dedup(origin_kind, same_function):
    sink = _length_sink("copy_len", 0x800, 10)
    callers = []
    for i in range(1 if same_function else 2):
        start = 0xa00 + i * 0x100
        # Deliberately identical names: addresses, not presentation, identify functions.
        v = FVar("input", ident=50)
        value = FSSA(v, 1 if origin_kind == "source" else 0)
        ins = []
        if origin_kind == "source":
            ins.append(_call(0, start + 4, _RECV, [], writes=[value]))
        ins.append(_call(len(ins), start + 0x10, sink.start, [_const(0), _const(0), _var(value)]))
        if same_function:
            ins.append(_call(len(ins), start + 0x20, sink.start, [_const(0), _const(0), _var(value)]))
        caller = FFunc("caller", start, FSSAFunc(ins), params=[v] if origin_kind == "parameter" else [])
        callers.append(caller)
    sink.caller_sites = [FSite(c, ins.address) for c in callers for ins in c.mlil.ssa_form.instructions
                         if ins.dest.constant == sink.start]
    result = _backward(sink, {**_NAMES, sink.start: sink.name})
    rows = result["slices"]
    assert len(rows) == len(callers)
    assert {r["origin"]["kind"] for r in rows} == {origin_kind}
    assert {r["origin"]["function"] for r in rows} == {hex(c.start) for c in callers}
    assert [r.get("reached_via_call_sites", 1) for r in rows] == ([2] if same_function else [1, 1])
    assert result["diagnostics"]["safe_to_report_complete_slice"] is True
    text = _render_taint_text(result)
    assert all(hex(c.start) in text for c in callers)


@pytest.mark.parametrize("counts", [(1,), (4,), (12,), (1, 4, 12)])
def test_dropped_callers_sums_sites_not_frontier_rows(counts):
    leaves = [{"kind": "caller_sites_truncated", "callers_dropped": count,
               "callers_total": count + 16, "callers_followed": 16,
               "function": {"name": f"handler_{i}", "address": hex(0x800 + i * 0x100)},
               "reasons": {"caller_cap": count}} for i, count in enumerate(counts)]
    diag = backward_diagnostics(leaves, [], sinks_seeded=1, slices=1)
    assert diag["frontier"]["dropped_callers"] == sum(counts)
    assert diag["frontier"]["by_kind"]["caller_sites_truncated"] == len(counts)
    assert diag["safe_to_report_complete_slice"] is False
    text = _render_taint_text({"direction": "backward", "leaves": leaves, "diagnostics": diag})
    assert f"{sum(counts)} dropped-caller-site(s)" in text


@pytest.mark.parametrize("resolution", ["value-set", "agent-map"])
def test_constant_site_retains_indirect_callee_attribution(resolution):
    from test_taint_engine import FPVS

    fn = _sites([64])
    call = fn.mlil.ssa_form.instructions[0]
    call.dest = FExpr("MLIL_VAR_SSA", "callback", possible_values=(
        FPVS("InSetOfValues", values=[_MEMCPY, 0x990, 0x991])
        if resolution == "value-set" else None))
    engine = te.TaintEngine(
        FBV(_NAMES), te.load_models(),
        resolve_map={"0x810": [hex(_MEMCPY)]} if resolution == "agent-map" else {})
    result = engine.backward(fn, [te.parse_locator("arg:memcpy:2")])
    # The qualification is public provenance, not a definite call to memcpy.
    assert any("0x810" in a and resolution in a for a in result["assumptions"])
    if resolution == "value-set":
        assert any("1 of 3 candidate targets" in a for a in result["assumptions"])
