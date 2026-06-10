"""Tier-1 tests for the dataflow/taint CLI handlers and text formatters.

No Binary Ninja required: the transport is monkeypatched, so these assert the
forwarded op/params and the text rendering of representative bridge results.
"""
from __future__ import annotations

import json

import bn.cli
import pytest

from bn.formatters import (
    _render_callgraph_text,
    _render_defuse_text,
    _render_structured_il_text,
    _render_taint_text,
    _render_values_text,
)


def _fake(op_results):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "sample"}]}
        if op in op_results:
            return {"ok": True, "result": op_results[op]}
        raise AssertionError(f"unexpected op: {op}")

    return fake_send_request, calls


# --------------------------------------------------------------------------
# forwarding: each subcommand sends the right op + params
# --------------------------------------------------------------------------

def test_dataflow_defuse_forwards(monkeypatch, capsys):
    fake, calls = _fake({"defuse": {"function": {"name": "f", "address": "0x1"},
                                     "variable": {"ssa": "len#2", "name": "len", "type": "int"},
                                     "definition": {"address": "0x10", "op": "MLIL_SET_VAR_SSA", "text": "len#2 = len#1 + 4"},
                                     "uses": [], "is_phi": False, "phi_sources": [], "other_versions": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["dataflow", "defuse", "f", "--var", "len#2", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "defuse"][0]
    assert call["params"] == {"identifier": "f", "var": "len#2"}
    assert "len#2 = len#1 + 4" in capsys.readouterr().out


def test_dataflow_callgraph_forwards_flags(monkeypatch, capsys):
    fake, calls = _fake({"resolved_calls": {"function": {"name": "f", "address": "0x1"}, "callees": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["dataflow", "callgraph", "f", "--direction", "callees",
                      "--no-resolve-indirect", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "resolved_calls"][0]
    assert call["params"] == {"identifier": "f", "direction": "callees", "resolve_indirect": False}


def test_dataflow_values_forwards(monkeypatch, capsys):
    fake, calls = _fake({"possible_values": {"function": {"name": "f", "address": "0x1"},
                                             "at": "0x10", "expression": "x", "possible_values": None}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["dataflow", "values", "f", "--at", "0x10", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "possible_values"][0]
    assert call["params"] == {"identifier": "f", "at": "0x10"}


def test_structured_il_forwards_defaults(monkeypatch, capsys):
    fake, calls = _fake({"structured_il": {"function": {"name": "f", "address": "0x1"},
                                           "view": "mlil", "ssa": True, "instructions": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["function", "structured-il", "f", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "structured_il"][0]
    assert call["params"] == {"identifier": "f", "view": "mlil", "ssa": True}


def test_taint_forward_forwards_sources(monkeypatch, capsys):
    fake, calls = _fake({"taint": {"direction": "forward", "function": {"name": "p", "address": "0x1"},
                                   "sources": [], "reached_sinks": [], "leaves": [],
                                   "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "forward", "-f", "process", "--source", "arg:read:1",
                      "--source", "param:0", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["direction"] == "forward"
    assert call["params"]["function"] == "process"
    assert call["params"]["sources"] == ["arg:read:1", "param:0"]
    assert call["params"]["unknown_call"] == "conservative"


def test_taint_forward_reads_and_forwards_resolve_map(monkeypatch, capsys, tmp_path):
    rmap = tmp_path / "rmap.json"
    rmap.write_text(json.dumps({"0x4011f0": ["0x401176", "0x401195"]}))
    fake, calls = _fake({"taint": {"direction": "forward", "function": {"name": "p", "address": "0x1"},
                                   "sources": [], "reached_sinks": [], "leaves": [],
                                   "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "forward", "-f", "dispatch", "--source", "param:1",
                      "--resolve-map", str(rmap), "--max-depth", "4", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["resolve_map"] == {"0x4011f0": ["0x401176", "0x401195"]}
    assert call["params"]["max_depth"] == 4


def test_taint_backward_forwards_sinks(monkeypatch, capsys):
    fake, calls = _fake({"taint": {"direction": "backward", "function": {"name": "p", "address": "0x1"},
                                   "sinks": [], "slices": [], "leaves": [], "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "backward", "-f", "process", "--sink", "arg:memcpy:2", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["direction"] == "backward"
    assert call["params"]["sinks"] == ["arg:memcpy:2"]


def test_taint_forward_requires_source(monkeypatch, capsys):
    fake, _ = _fake({})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "forward", "-f", "process", "--target", "active"])
    assert rc != 0


# --------------------------------------------------------------------------
# formatter rendering
# --------------------------------------------------------------------------

def test_render_taint_forward_text():
    value = {
        "direction": "forward",
        "function": {"name": "process", "address": "0x401189"},
        "sources": [{"kind": "arg", "callee": "read", "index": 1}],
        "reached_sinks": [{
            "sink": {"callee": "memcpy", "address": "0x4011db", "tainted_arg_index": 2,
                     "class": "overflow_len", "detail": "attacker-controlled length to memcpy"},
            "path": [
                {"address": "0x4011a9", "op": "MLIL_CALL_SSA", "il_text": "read(...)", "reason": "source: read fills arg1 buffer"},
                {"address": "0x4011db", "op": "MLIL_CALL_SSA", "il_text": "memcpy(...)", "reason": "tainted arg2 reaches memcpy"},
            ],
        }],
        "leaves": [{"kind": "indirect_call_unresolved", "address": "0x401200", "dest_expr": "rax#3", "detail": "unresolved"}],
        "assumptions": ["memory aliasing modeled coarsely (memory_approx)"],
        "soundness": "may-analysis (intraprocedural, MVP)",
    }
    text = _render_taint_text(value)
    assert "forward taint in process @ 0x401189" in text
    assert "[overflow_len] memcpy @ 0x4011db (arg 2)" in text
    assert "source: read fills arg1 buffer" in text
    assert "UNRESOLVED LEAVES (1)" in text
    assert "ASSUMPTIONS:" in text
    assert "soundness:" in text


def test_render_taint_backward_text():
    value = {
        "direction": "backward",
        "function": {"name": "process", "address": "0x401189"},
        "sinks": [{"kind": "arg", "callee": "memcpy", "index": 2}],
        "slices": [{
            "sink": {"callee": "memcpy", "address": "0x4011db", "seed": "rdx_1#1"},
            "origin": {"kind": "parameter_or_entry", "var": "buf#1"},
            "slice": [{"address": "0x4011bc", "op": "MLIL_SET_VAR_SSA", "il_text": "len#2 = len#1 + 4"}],
        }],
        "leaves": [], "assumptions": [], "soundness": "x",
    }
    text = _render_taint_text(value)
    assert "slice for memcpy @ 0x4011db (seed rdx_1#1)" in text
    assert "origin: parameter_or_entry buf#1" in text
    assert "len#2 = len#1 + 4" in text


def test_render_callgraph_indirect_unresolved():
    value = {
        "function": {"name": "dispatch", "address": "0x401200"},
        "callees": [{"call_addr": "0x40121a", "kind": "indirect", "dest_expr": "rdx#1",
                     "resolved": [], "resolution": "unresolved", "resolution_detail": "UndeterminedValue"}],
    }
    text = _render_callgraph_text(value)
    assert "indirect [rdx#1]" in text
    assert "UNRESOLVED (UndeterminedValue)" in text


def test_render_structured_il_text():
    value = {"function": {"name": "f", "address": "0x1"}, "view": "mlil", "ssa": True,
             "instructions": [{"il_index": 0, "address": "0x1", "op": "MLIL_SET_VAR_SSA",
                               "text": "a#1 = b#0", "vars_read": [{"ssa": "b#0"}], "vars_written": [{"ssa": "a#1"}]}]}
    text = _render_structured_il_text(value)
    assert "mlil ssa" in text
    assert "a#1 = b#0" in text
    assert "r:[b#0]  w:[a#1]" in text


def test_render_values_text_constant():
    value = {"function": {"name": "f", "address": "0x1"}, "at": "0x10", "expression": "x",
             "possible_values": {"type": "ConstantValue", "value": 0x40, "raw": "<const 0x40>"}}
    text = _render_values_text(value)
    assert "ConstantValue" in text
    assert "0x40" in text
