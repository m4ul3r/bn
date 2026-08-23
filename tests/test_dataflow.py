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
    _taint_forward_verdict,
    _taint_via_trail,
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


def test_taint_backward_reads_and_forwards_resolve_map(monkeypatch, capsys, tmp_path):
    rmap = tmp_path / "rmap.json"
    rmap.write_text(json.dumps({"0x401200": ["0x401050"]}))
    fake, calls = _fake({"taint": {"direction": "backward", "function": {"name": "p", "address": "0x1"},
                                   "sinks": [], "slices": [], "leaves": [],
                                   "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "backward", "-f", "emit", "--sink", "arg:send:1",
                      "--resolve-map", str(rmap), "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["resolve_map"] == {"0x401200": ["0x401050"]}


def test_taint_forward_requires_source(monkeypatch, capsys):
    # --source is argparse-required, so the usage is honest (`--source LOCATOR`,
    # not `[--source LOCATOR]`) and a missing source errors before dispatch (#375).
    fake, _ = _fake({})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    with pytest.raises(SystemExit):
        bn.cli.main(["taint", "forward", "-f", "process", "--target", "active"])


def test_taint_backward_requires_sink(monkeypatch, capsys):
    # #375: backward `--sink` was bracketed (optional) in usage but required at
    # runtime. Now argparse-required, so the usage no longer misleads.
    fake, _ = _fake({})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    with pytest.raises(SystemExit):
        bn.cli.main(["taint", "backward", "-f", "process", "--target", "active"])
    err = capsys.readouterr().err
    assert "--sink" in err  # argparse names the missing required arg


def test_taint_backward_required_sink_usage_not_bracketed(capsys):
    """The backward usage must present `--sink` as required (no `[ ]`), so an
    agent doesn't read it as optional (#375)."""
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["taint", "backward", "--help"])
    usage = capsys.readouterr().out
    assert "[--sink" not in usage      # not optional
    assert "--sink LOCATOR" in usage   # shown as a required arg


def test_taint_forward_source_help_documents_call_locator(capsys):
    # call:<callee> is a first-class source kind (#157) but was missing from the
    # --source help, so it only surfaced in parse-error text. The locator help
    # must list it alongside param:/var:/ret:/arg:.
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["taint", "forward", "--help"])
    help_text = capsys.readouterr().out
    assert "call:" in help_text


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
    text = _render_taint_text(value, full=True)
    assert "forward taint in process @ 0x401189" in text
    assert "[overflow_len] memcpy @ 0x4011db (arg 2)" in text
    assert "source: read fills arg1 buffer" in text
    assert "frontiers (1)" in text
    assert "caveats (" in text
    assert "soundness:" in text


def test_render_taint_discloses_model_sources_in_text():
    # #415 review: TEXT is the default taint output, so the active-overlay
    # disclosure must appear there -- not only in --format json. A run that loaded
    # a --models file shows the file and its (correctly unwrapped) model count.
    value = {
        "direction": "forward",
        "function": {"name": "process", "address": "0x401189"},
        "sources": [{"kind": "param", "index": 0}],
        "reached_sinks": [],
        "model_sources": [
            {"kind": "builtin", "path": "/builtin/models.json"},
            {"kind": "user", "via": "--models", "count": 2, "path": "proj.json"},
        ],
        "soundness": "may-analysis (intraprocedural, MVP)",
    }
    text = _render_taint_text(value)
    assert "models: builtin + --models proj.json (2 model(s))" in text


def test_render_taint_forward_frontier_message():
    # reached_sinks empty BUT an unmodeled_callee frontier leaf -> the terminal
    # line must be frontier-aware, not the bare "no sinks reached" (#8).
    value = {
        "direction": "forward",
        "function": {"name": "ipc_read", "address": "0x1000"},
        "sources": [{"kind": "arg", "callee": "recv", "index": 1}],
        "reached_sinks": [],
        "leaves": [{"kind": "unmodeled_callee", "address": "0x1008",
                    "callee": {"name": "parse_event", "address": "0x3000"},
                    "tainted_args": [0],
                    "note": "tainted data passed to in-binary callee with no model; investigate"}],
        "assumptions": [],
        "soundness": "may-analysis",
    }
    text = _render_taint_text(value)
    assert "no sinks reached by tainted data" not in text
    assert "NO modeled sink reached" in text
    assert "1 tainted frontier" in text and "NOT an all-clear" in text
    # the call -> callee hand-off line for the frontier leaf
    assert "parse_event @ 0x3000" in text
    assert "0x1008" in text


def test_render_taint_forward_by_source():
    # by_source present -> a per-source attribution section keyed by call addr.
    sink = {"callee": "strcpy", "address": "0x24", "tainted_arg_index": 1,
            "class": "overflow_unbounded", "detail": "unbounded copy"}
    value = {
        "direction": "forward",
        "function": {"name": "server", "address": "0x10"},
        "sources": [{"kind": "arg", "callee": "recv", "index": 1}],
        "reached_sinks": [{"sink": sink, "path": []}],
        "leaves": [],
        "assumptions": [],
        "by_source": {
            "0x14": {"reached_sinks": [{"sink": sink, "path": []}], "leaves": []},
            "0x1c": {"reached_sinks": [], "leaves": []},
        },
        "soundness": "may-analysis",
    }
    text = _render_taint_text(value)
    assert "PER-SOURCE" in text
    assert "0x14" in text and "0x1c" in text
    assert "strcpy" in text


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
    text = _render_taint_text(value, full=True)
    assert "slice for memcpy @ 0x4011db (seed rdx_1#1)" in text
    assert "origin: parameter_or_entry buf#1" in text
    assert "len#2 = len#1 + 4" in text


def test_render_taint_backward_bounded_sink_is_success_not_unseeded():
    # #310: a provably-bounded (constant-length) sink renders under "provably
    # bounded", NOT "UNSEEDED SINKS" -- it's a successful conclusion.
    value = {
        "direction": "backward",
        "function": {"name": "f", "address": "0x401189"},
        "sinks": [{"kind": "arg", "callee": "memcpy", "index": 2}],
        "slices": [],
        "sink_status": [{"kind": "arg", "callee": "memcpy", "index": 2,
                         "seeded": False, "bounded": True,
                         "note": "arg 2 of memcpy is a compile-time constant -- provably bounded"}],
        "leaves": [], "assumptions": [], "soundness": "x",
    }
    text = _render_taint_text(value)
    assert "provably bounded (1)" in text
    assert "UNSEEDED SINKS" not in text


def test_render_taint_groups_repeated_leaves():
    # many near-identical frontier leaves must collapse to one grouped line with
    # a count, not a wall of per-leaf lines (#160).
    leaves = [{"kind": "unmodeled_callee", "address": hex(0x1000 + i),
               "callee": {"name": "g_free", "address": "0x3000"}, "tainted_args": [0]}
              for i in range(37)]
    leaves.append({"kind": "unmodeled_callee", "address": "0x2000",
                   "callee": {"name": "g_slist_append", "address": "0x3100"}, "tainted_args": [1]})
    value = {"direction": "forward", "function": {"name": "f", "address": "0x10"},
             "sources": [], "reached_sinks": [], "leaves": leaves,
             "assumptions": [], "soundness": "x"}
    text = _render_taint_text(value)
    assert "frontiers (38 in 2 group(s))" in text
    assert "g_free" in text and "(x37)" in text
    assert "g_slist_append" in text
    # the 38 leaves collapse to exactly two rendered leaf lines
    assert text.count("unmodeled_callee @") == 2


def test_render_taint_grouped_leaves_top_n_cap():
    # more distinct groups than the cap -> a "... and N more group(s)" summary.
    leaves = [{"kind": "unmodeled_callee", "address": hex(0x1000 + i),
               "callee": {"name": f"fn_{i}", "address": "0x3000"}, "tainted_args": [0]}
              for i in range(20)]
    value = {"direction": "forward", "function": {"name": "f", "address": "0x10"},
             "sources": [], "reached_sinks": [], "leaves": leaves,
             "assumptions": [], "soundness": "x"}
    text = _render_taint_text(value)
    assert "frontiers (20)" in text
    assert "... and 8 more group(s)" in text
    assert "see --format json" in text


def test_render_taint_field_load_unresolved_leaf():
    # the #158 field_load_unresolved leaf renders base/offset/width, not a bare kind.
    value = {"direction": "backward", "function": {"name": "parse", "address": "0x10"},
             "sinks": [{"kind": "arg", "callee": "memcpy", "index": 2}],
             "slices": [{"sink": {"callee": "memcpy", "address": "0x40", "seed": "x#1"},
                         "origin": {"kind": "field_load_unresolved", "base": "obj#1",
                                    "offset": "0x8", "width": 4},
                         "slice": []}],
             "leaves": [{"kind": "field_load_unresolved", "address": "0x30", "base": "obj#1",
                         "offset": "0x8", "width": 4, "il_text": "x#1 = [obj#1 + 8]"}],
             "assumptions": [], "soundness": "x"}
    text = _render_taint_text(value, full=True)
    assert "frontiers (" in text
    assert "field_load_unresolved @ 0x30" in text
    assert "base=obj#1" in text and "offset=0x8" in text and "width=4" in text
    assert "origin: field_load_unresolved" in text


def test_render_taint_forward_sink_without_arg_index_has_no_arg_none():
    value = {"direction": "forward", "function": {"name": "f", "address": "0x1"},
             "sources": [], "reached_sinks": [{
                 "sink": {"callee": "<memory store>", "address": "0x14",
                          "tainted_arg_index": None, "class": "tainted_index",
                          "detail": "d"}, "path": []}],
             "leaves": [], "assumptions": [], "stats": {}, "soundness": "x"}
    text = _render_taint_text(value)
    assert "(arg None)" not in text
    assert "[tainted_index] <memory store> @ 0x14" in text


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


def test_taint_forward_verdict_sinks():
    v = {"reached_sinks": [{"sink": {"class": "overflow_len"}}],
         "leaves": [], "stats": {"functions_visited": 8, "truncated": True, "max_depth": 8}}
    assert _taint_forward_verdict(v) == (
        "verdict: 1 sink(s) reached (overflow_len) · taint crossed 8 fn(s) · truncated @depth 8")


def test_taint_forward_verdict_frontiers_not_all_clear():
    v = {"reached_sinks": [], "leaves": [{"kind": "coarse_memory_store"}] * 12,
         "stats": {"functions_visited": 1}}
    out = _taint_forward_verdict(v)
    assert "NO modeled sink reached" in out and "12 tainted frontier" in out and "NOT an all-clear" in out


def test_taint_forward_verdict_empty_is_loudly_caveated():
    # #310: a genuinely empty forward result (no sink, no frontier) is the MOST
    # caveated case -- it must carry the same "NOT an all-clear" qualifier as the
    # partial-coverage paths and note the coverage, since it's also how a bug the
    # engine can't structurally see appears.
    v = {"reached_sinks": [], "leaves": [], "stats": {"functions_visited": 3}}
    out = _taint_forward_verdict(v)
    assert "no taint reached any sink or frontier" in out
    assert "NOT an all-clear" in out
    assert "visited 3 fn(s)" in out
    assert "can't structurally see" in out


def test_taint_via_trail_from_path_reasons():
    value = {"function": {"name": "ip_input"}}
    finding = {"sink": {"callee": "memcpy"}, "path": [
        {"reason": "[agent-map-resolved] calls tcp_input with tainted arg(s) [0, 1]"},
        {"reason": "calls m_pullup with tainted arg(s) [0]"},
        {"reason": "tainted arg2 reaches memcpy"},
    ]}
    assert _taint_via_trail(value, finding) == "via: ip_input → tcp_input → m_pullup → memcpy"


def test_taint_via_trail_none_when_no_callees():
    assert _taint_via_trail({"function": {"name": "f"}},
                            {"sink": {"callee": "x"}, "path": [{"reason": "assignment/copy of tainted value"}]}) is None


def test_locator_help_states_zero_based_indices():
    # #291.4: the arg:<callee>:<n> / param:<n> grammar is 0-based; make that
    # explicit in the locator help so a user reading pseudo-C doesn't reach for
    # the 1-based index.
    from bn.commands import dataflow
    assert "0-based" in dataflow._LOCATOR_HELP
    assert "0-based" in dataflow._SINK_LOCATOR_HELP


def test_taint_forward_reads_and_forwards_user_models(monkeypatch, capsys, tmp_path):
    # #317: --models loads project-internal wrapper models and forwards them as
    # `user_models` so the bridge merges them over the builtin DB.
    mfile = tmp_path / "models.json"
    mfile.write_text(json.dumps({"app_copy": {"sink": {"class": "overflow_len", "tainted_args": [1]},
                                              "propagates": [{"from": "*arg:1", "to": "*arg:0"}]}}))
    fake, calls = _fake({"taint": {"direction": "forward", "function": {"name": "p", "address": "0x1"},
                                   "sources": [], "reached_sinks": [], "leaves": [],
                                   "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "forward", "-f", "h", "--source", "param:0",
                      "--models", str(mfile), "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["user_models"]["app_copy"]["sink"]["class"] == "overflow_len"


def test_taint_backward_reads_and_forwards_user_models(monkeypatch, capsys, tmp_path):
    mfile = tmp_path / "models.json"
    mfile.write_text(json.dumps({"app_fmt": {"sink": {"class": "format_string", "tainted_args": [0]}}}))
    fake, calls = _fake({"taint": {"direction": "backward", "function": {"name": "p", "address": "0x1"},
                                   "sinks": [], "slices": [], "leaves": [], "assumptions": [], "soundness": "x"}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "backward", "-f", "h", "--sink", "arg:app_fmt:0",
                      "--models", str(mfile), "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint"][0]
    assert call["params"]["user_models"]["app_fmt"]["sink"]["class"] == "format_string"


def test_taint_models_bad_file_is_loud(monkeypatch, capsys, tmp_path):
    # A broken --models file is a clean error, not a silent skip (#97/#317).
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    fake, _ = _fake({"taint": {}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "forward", "-f", "h", "--source", "param:0",
                      "--models", str(bad), "--target", "active"])
    assert rc != 0   # BridgeError -> non-zero
    # a nonexistent file is likewise loud
    rc2 = bn.cli.main(["taint", "forward", "-f", "h", "--source", "param:0",
                       "--models", str(tmp_path / "nope.json"), "--target", "active"])
    assert rc2 != 0


def test_render_backward_origin_via_spill():
    value = {"direction": "backward", "function": {"name": "use_len", "address": "0x800"},
             "sinks": [{"kind": "arg", "callee": "memcpy", "index": 2}],
             "slices": [{"sink": {"callee": "memcpy", "address": "0x804", "seed": "var_n#5"},
                         "origin": {"kind": "parameter", "index": 2, "var": "var_n#5", "via_spill": True},
                         "slice": []}],
             "leaves": [], "assumptions": [], "soundness": "x"}
    # #463 moved the origin line behind --full; via_spill still renders there.
    text = _render_taint_text(value, full=True)
    assert "origin: parameter var_n#5 (via spill)" in text


def test_taint_models_command_dumps_catalog(monkeypatch, capsys):
    fake, calls = _fake({"taint_models": {
        "sources": [{"symbol": "recv", "to": "*arg:1, ret"}],
        "sinks_by_class": {"overflow_len": [{"symbol": "memcpy", "tainted_args": [2],
                                             "class": "overflow_len", "detail": "len"}]},
        "propagators": [{"symbol": "strlen", "from_to": "*arg:0->ret"}],
        "overlays": [{"kind": "builtin", "path": "/x/taint_models.json"}], "items": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recv" in out and "memcpy" in out and "overflow_len" in out
    call = [c for c in calls if c["op"] == "taint_models"][0]
    assert call["params"] == {}                          # no filters -> empty params


def test_taint_models_command_forwards_filters(monkeypatch, capsys):
    fake, calls = _fake({"taint_models": {"sources": [], "sinks_by_class": {}, "propagators": [],
                                          "overlays": [], "items": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "models", "--role", "sink", "--class", "overflow_len",
                      "--present", "--callsites", "--target", "active"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint_models"][0]
    assert call["params"] == {"role": "sink", "class": "overflow_len",
                              "present": True, "callsites": True}


def test_taint_models_present_auto_selects_single_target(monkeypatch, capsys):
    # #473: --present needs a target, and with exactly one open it must be
    # auto-selected (like every other read command) rather than forwarding
    # target=None and hitting the bridge's "need a target" error. No --target here.
    fake, calls = _fake({"taint_models": {"sources": [], "sinks_by_class": {}, "propagators": [],
                                          "overlays": [], "items": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "models", "--present"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint_models"][0]
    assert call["target"] == "1:1:1"                     # single open target pinned by target_id (#690 R3)
    assert call["params"] == {"present": True}


def test_taint_models_catalog_only_needs_no_target(monkeypatch, capsys):
    # Regression: catalog-only mode (no --present/--callsites) must NOT require a
    # target -- it forwards target=None so the whole model DB dumps with none open.
    fake, calls = _fake({"taint_models": {"sources": [], "sinks_by_class": {}, "propagators": [],
                                          "overlays": [], "items": []}})
    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["taint", "models", "--role", "sink"])
    assert rc == 0
    call = [c for c in calls if c["op"] == "taint_models"][0]
    assert call["target"] is None
