from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_render_name_address_rows_escapes_control_chars():
    """A symbol name containing a newline must not break the row across two lines
    in --format text; control chars are escaped (#370.1). JSON keeps the raw name."""
    from bn.formatters import _render_name_address_rows
    out = _render_name_address_rows([
        {"address": "0x1000", "name": "good_name"},
        {"address": "0x2000", "name": "evil\nname\twith\x07ctrl"},
    ])
    # the malicious row stays on ONE physical line (no raw newline injected)
    rows = out.splitlines()
    assert len(rows) == 2, rows
    assert "evil\\nname\\twith" in out          # escaped, visible
    assert "\x07" not in out                     # raw control byte gone


def test_render_name_address_rows_shows_basic_block_count():
    """#411: text is the DEFAULT read output, so the real complexity metric
    (basic_block_count) must be visible there, not only in JSON. A row carrying
    basic_block_count renders it alongside the byte span; a row whose count is
    None/absent omits the blocks clause and still shows the byte span."""
    from bn.formatters import _render_name_address_rows
    out = _render_name_address_rows([
        {"address": "0x401000", "name": "parse_loop", "size": 256,
         "basic_block_count": 42},
        {"address": "0x402000", "name": "tiny", "size": 8,
         "basic_block_count": None},
        {"address": "0x403000", "name": "legacy", "size": 16},  # field absent
    ])
    rows = out.splitlines()
    assert rows[0] == "0x401000  parse_loop  (256 bytes, 42 blocks)"
    assert rows[1] == "0x402000  tiny  (8 bytes)"          # None -> no blocks clause
    assert rows[2] == "0x403000  legacy  (16 bytes)"       # absent -> no blocks clause


def test_render_function_bundle_text_pretty_prints_not_escaped():
    """`bundle function --format text` must render readable multi-line JSON with a
    note, not a single line of escaped JSON like the default fallback (#362)."""
    from bn.formatters import _render_function_bundle_text
    value = {"kind": "function_bundle",
             "function": {"name": "login", "address": "0x401000"}}
    out = _render_function_bundle_text(value)
    assert out.count("\n") >= 2              # multi-line, not one escaped blob
    assert '"name": "login"' in out          # readable, indented JSON
    assert "function bundle" in out.lower()  # the note


def test_parser_default_formats():
    parser = bn.cli.build_parser()

    # Read commands default to text.
    assert parser.parse_args(["function", "list"]).format == "text"
    assert parser.parse_args(["function", "list"]).target is None
    assert parser.parse_args(["callsites", "crt_rand", "--within", "bonus_pick_random_type"]).format == "text"
    assert parser.parse_args(["decompile", "sub_401000"]).target is None
    assert parser.parse_args(["decompile", "sub_401000"]).format == "text"

    # Setup-style commands keep JSON for structured envelopes; skill install is human-friendly.
    assert parser.parse_args(["plugin", "install"]).format == "json"
    assert parser.parse_args(["skill", "install"]).format == "text"
    assert parser.parse_args(["skill", "install"]).mode == "symlink"
    assert parser.parse_args(["bundle", "function", "sub_401000"]).format == "json"

    # Mutations default to JSON per the documented convention; the text
    # summary is one --format text away when needed.
    assert parser.parse_args(["symbol", "rename", "sub_401000", "player_update"]).format == "json"
    assert parser.parse_args(["rename", "sub_401000", "player_update"]).format == "json"
    assert parser.parse_args(["types", "declare", "typedef struct Player { int hp; } Player;"]).format == "json"
    assert parser.parse_args(["comment", "set", "--address", "0x401000", "msg"]).format == "json"
    assert parser.parse_args(["comment", "delete", "--address", "0x401000"]).format == "json"
    assert parser.parse_args(["proto", "set", "sub_401000", "void()"]).format == "json"
    assert parser.parse_args(["local", "rename", "fn", "var", "new"]).format == "json"
    assert parser.parse_args(["local", "retype", "fn", "var", "int"]).format == "json"
    assert parser.parse_args(["struct", "field", "set", "S", "0", "f", "uint32_t"]).format == "json"
    assert parser.parse_args(["struct", "field", "rename", "S", "old", "new"]).format == "json"
    assert parser.parse_args(["struct", "field", "delete", "S", "f"]).format == "json"
    assert parser.parse_args(["batch", "apply", "manifest.json"]).format == "json"
    assert parser.parse_args(["function", "create", "0x401000"]).format == "json"


def test_argparse_error_emits_json_envelope_under_format_json(capsys):
    # An argparse usage/type error must emit a parseable {"ok": false, ...}
    # object on stdout under --format json (not an empty stream), while the
    # human-readable usage still goes to stderr at exit code 2 (#29).
    import json as _json
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["function", "list", "--target", "active", "--limit", "-1", "--format", "json"])
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    payload = _json.loads(out)
    assert payload["ok"] is False
    assert "error" in payload and payload["error"]
    assert err  # usage text still on stderr


def test_argparse_error_text_format_keeps_stdout_empty(capsys):
    # Default text format keeps the prior contract: usage on stderr, no stdout.
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["function", "list", "--target", "active", "--limit", "-1"])
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err


def test_lines_range_rejects_zero_index_with_helpful_error(monkeypatch, capsys):
    # argparse type errors exit via SystemExit(2)
    with pytest.raises(SystemExit):
        bn.cli.main(["disasm", "0x1000", "--target", "active", "--lines", "0:3"])
    err = capsys.readouterr().err
    assert "1-indexed" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["decompile", "sub_401000", "--format", "json", "--lines", "1:5"],
        ["decompile", "sub_401000", "--format", "ndjson", "--lines", "1:5"],
        ["il", "sub_401000", "--format", "json", "--lines", "1:5"],
    ],
)
def test_lines_flag_rejected_outside_text_mode(monkeypatch, capsys, argv):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(argv + ["--target", "active"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--lines only applies to --format text" in err
    assert "Traceback" not in err


def test_disasm_json_lines_are_sliced_by_bridge(fake_transport, capsys):
    calls = fake_transport(
        {
            "disasm": {
                "ok": True,
                "result": {
                    "text": "line two",
                    "total_lines": 4,
                    "returned_lines": 1,
                    "line_range": {"start": 2, "end": 2},
                },
            }
        }
    )

    rc = bn.cli.main(
        [
            "disasm",
            "sub_401000",
            "--format",
            "json",
            "--lines",
            "2:2",
            "--target",
            "active",
        ]
    )

    assert rc == 0
    assert calls[-1]["params"]["line_start"] == 2
    assert calls[-1]["params"]["line_end"] == 2
    assert calls[-1]["params"]["strict_range"] is True
    assert json.loads(capsys.readouterr().out)["text"] == "line two"


def test_json_format_error_emits_json_to_stdout(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise BridgeError("Type not found: Foo")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["types", "show", "Foo", "--format", "json", "--target", "active"])
    assert rc == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)              # stdout is valid JSON under --format json
    assert payload["ok"] is False
    assert "Type not found" in payload["error"]
    assert "Type not found" in err          # human-readable line still on stderr


def test_text_format_error_stays_on_stderr(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise BridgeError("Type not found: Foo")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["types", "show", "Foo", "--target", "active"])
    assert rc == 2
    out, err = capsys.readouterr()
    assert out == ""                        # nothing on stdout in text mode
    assert "Type not found" in err


def test_text_renderer_failure_becomes_clean_error(fake_transport, capsys):
    # #101: a malformed bridge result that trips a text renderer must surface a
    # clean BridgeError (exit 2) pointing at --format json, not a raw traceback.
    # 'function' present but a STRING, not a dict -> .get() would crash.
    fake_transport({"function_info": {"ok": True, "result": {"function": "not-a-dict"}}})
    # function info should now render with placeholders (defensive _as_dict), not crash.
    rc = bn.cli.main(["function", "info", "main", "--target", "active", "--format", "text"])
    assert rc == 0
    assert "<unknown>" in capsys.readouterr().out


# --- arg_under_recovered frontier rendering (Thread A) -----------------------

def test_render_arg_under_recovered_leaf():
    from bn.formatters import _render_grouped_leaves
    leaf = {"kind": "arg_under_recovered", "address": "0x40130a",
            "callee": {"name": "_M_create", "address": "0x3000"},
            "recovered_params": 1, "dropped_args": [1],
            "note": 'tainted arg(s) [1] ... apply `bn proto set _M_create "<prototype>"` ...'}
    out = "\n".join(_render_grouped_leaves([leaf]))
    assert "arg_under_recovered @ 0x40130a" in out
    assert "_M_create" in out
    assert "recovered 1 param" in out
    assert "proto set _M_create" in out


def test_arg_under_recovered_leaves_group_per_callee():
    from bn.formatters import _render_grouped_leaves
    mk = lambda addr: {"kind": "arg_under_recovered", "address": addr,
                       "callee": {"name": "f", "address": "0x3000"},
                       "recovered_params": 1, "dropped_args": [1], "note": "n"}
    out = "\n".join(_render_grouped_leaves([mk("0x10"), mk("0x20")]))
    assert "(x2)" in out                                   # two call sites of f collapse
# --- compact-default taint output (Thread C) ---------------------------------

_FWD_FLOW = {
    "direction": "forward",
    "function": {"name": "parse_request", "address": "0x401000"},
    "sources": ["arg:recv:1"],
    "reached_sinks": [{
        "sink": {"class": "overflow_len", "callee": "memcpy", "address": "0x401f30",
                 "tainted_arg_index": 2, "detail": "attacker-controlled length"},
        "path": [{"address": "0x401f30", "op": "MLIL_CALL_SSA", "il_text": "memcpy(...)"}],
        "metrics": {"steps": 11, "fns_spanned": 3, "traverses_unresolved": False},
        "signature": {"source": "arg:recv:1", "chain": ["parse_hdr", "copy_field"],
                      "sink_class": "overflow_len", "sink_callee": "memcpy",
                      "rendered": "arg:recv:1 → parse_hdr → copy_field → [overflow_len] memcpy"},
    }],
    "leaves": [], "assumptions": [], "soundness": "may-analysis",
    "stats": {"functions_visited": 3},
}


def test_taint_compact_default_one_line_per_flow():
    from bn.formatters import _render_taint_text
    out = _render_taint_text(_FWD_FLOW)                  # full defaults False
    assert "arg:recv:1 → parse_hdr → copy_field → [overflow_len] memcpy" in out
    assert "steps=11" in out and "fns=3" in out and "unresolved=n" in out
    assert "memcpy(...)" not in out                      # SSA path suppressed by default
    assert "soundness" in out                            # honesty guard kept


_FWD_ZERO_WITH_DIAG = {
    "direction": "forward",
    "function": {"name": "ipc_read", "address": "0x1000"},
    "sources": [{"kind": "arg", "callee": "recv", "index": 1}],
    "reached_sinks": [],
    "leaves": [{"kind": "unmodeled_callee", "address": "0x1008",
                "callee": {"name": "parse_event", "address": "0x3000"}}],
    "assumptions": [],
    "soundness": "may-analysis",
    "stats": {"functions_visited": 1, "leaves": 1, "truncated": False, "max_depth": 0},
    "diagnostics": {
        "source_callsites": 1,
        "tainted_values": 2,
        "last_use": {"label": "rsi#1", "address": "0x1008", "reason": "arg to parse_event"},
        "unmodeled_calls_reached": True,
        "frontier": {"unresolved": 1, "coarse_memory": 0,
                     "by_kind": {"unmodeled_callee": 1}},
        "next_action": "recover the callee prototype with `bn proto set`",
    },
}


def test_taint_zero_result_renders_frontier_diagnostics_559():
    """#559: a zero-result forward run surfaces its diagnostic block in text mode
    so an agent doesn't misread the empty result as a clean breadth check."""
    from bn.formatters import _render_taint_text
    out = _render_taint_text(_FWD_ZERO_WITH_DIAG)
    assert "diagnostics:" in out
    assert "matched 1 source callsite(s)" in out
    assert "produced 2 tainted value(s)" in out
    assert "last propagated use: rsi#1 @ 0x1008" in out
    assert "unmodeled call(s) reached: yes" in out
    assert "1 unresolved" in out
    assert "bn proto set" in out


def test_taint_zero_result_without_diagnostics_is_unchanged():
    """A zero-result payload with no diagnostics block renders no diagnostics line."""
    from bn.formatters import _render_taint_text
    bare = {k: v for k, v in _FWD_ZERO_WITH_DIAG.items() if k != "diagnostics"}
    out = _render_taint_text(bare)
    assert "diagnostics:" not in out


def test_taint_zero_result_renders_folded_claim_gate_562():
    """#562: the honesty claim gate is FOLDED INTO the single diagnostics block
    (#571 renderer), never a competing block. A false gate withholds the
    all-clear and names the seed-misanchored frontier."""
    from bn.formatters import _render_taint_text
    payload = {k: v for k, v in _FWD_ZERO_WITH_DIAG.items()}
    payload["diagnostics"] = {
        **_FWD_ZERO_WITH_DIAG["diagnostics"],
        "frontier": {"unresolved": 0, "coarse_memory": 0, "seed_misanchored": 1,
                     "by_kind": {"source_seed_misanchored": 1}},
        "safe_to_report_all_clear": False,
        "all_clear_reason": "no modeled sink reached, but 1 blocking frontier "
                            "leaf(s) (source_seed_misanchored) remain -- NOT an all-clear",
    }
    out = _render_taint_text(payload)
    assert "safe_to_report_all_clear: false" in out
    assert "NOT an all-clear" in out
    assert "1 seed-misanchored" in out


def test_taint_zero_result_renders_true_gate_as_may_analysis_562():
    """A true gate must render as may-analysis, not a proof of safety."""
    from bn.formatters import _render_taint_text
    payload = {k: v for k, v in _FWD_ZERO_WITH_DIAG.items()}
    payload["diagnostics"] = {
        **_FWD_ZERO_WITH_DIAG["diagnostics"],
        "safe_to_report_all_clear": True,
        "all_clear_reason": "no modeled sink and no tainted frontier; still a "
                            "may-analysis -- not a proof of safety",
    }
    out = _render_taint_text(payload)
    assert "safe_to_report_all_clear: true (may-analysis, not a proof)" in out


def test_taint_full_restores_ssa_path():
    from bn.formatters import _render_taint_text
    out = _render_taint_text(_FWD_FLOW, full=True)
    assert "memcpy(...)" in out                          # SSA path shown


def test_taint_two_distinct_sink_addresses_never_fold():
    from bn.formatters import _render_taint_text
    two = {**_FWD_FLOW, "reached_sinks": [
        _FWD_FLOW["reached_sinks"][0],
        {**_FWD_FLOW["reached_sinks"][0],
         "sink": {**_FWD_FLOW["reached_sinks"][0]["sink"], "address": "0x402a10"}},
    ]}
    out = _render_taint_text(two)
    assert "0x401f30" in out and "0x402a10" in out       # both sinks visible, not folded


def test_render_field_xrefs_text_paging_note_532():
    from bn.formatters import _render_field_xrefs_text
    base_field = {"type_name": "Hot", "field_name": "f", "offset": 8, "field_type": "int"}
    # Full set (offset 0, returned == total): no paging note.
    full = {"kind": "field_xrefs", "field": base_field,
            "items": [{"kind": "code", "address": "0x1000"}],
            "total": 1, "offset": 0, "limit": None, "returned": 1, "has_more": False}
    assert "showing" not in _render_field_xrefs_text(full)
    # More pages remain: note + "more available".
    more = {**full, "total": 12, "returned": 5, "limit": 5, "has_more": True}
    out_more = _render_field_xrefs_text(more)
    assert "showing 5 of 12" in out_more and "more available" in out_more
    # Last page of an --offset run (has_more False but returned != total): still noted,
    # so the skipped refs aren't silently dropped.
    tail = {**full, "total": 12, "offset": 10, "returned": 2, "limit": 5, "has_more": False}
    out_tail = _render_field_xrefs_text(tail)
    assert "showing 2 of 12" in out_tail and "offset 10" in out_tail


def test_render_virtual_call_text_includes_method_address_533():
    # #533: the text output must show the concrete jump target (method_address) --
    # the pointer's VALUE, distinct from vtable_entry (the slot's address).
    from bn.formatters import _render_virtual_call_text
    value = {
        "callsite": "0x40115d", "caller": "consumer",
        "slot_offset": "0x18", "slot_index": 3, "factory": "makeProvider",
        "candidates": [{
            "provider": "libprov.so", "class": "Provider",
            "vtable": "0x9000", "vtable_entry": "0x9020",
            "method": "doWork", "method_address": "0x4100",
        }],
        "ambiguous": False, "resolved": True,
    }
    out = _render_virtual_call_text(value)
    assert "0x4100" in out            # method_address rendered
    assert "0x9020" in out            # vtable_entry still shown, distinct
    assert "doWork" in out


def test_render_virtual_call_text_handles_int_and_missing_method_address_533():
    from bn.formatters import _render_virtual_call_text
    # int method_address is hex-formatted; a None one renders without crashing.
    value = {
        "callsite": "0x1000", "caller": "c", "slot_offset": "0x8", "slot_index": 1,
        "factory": None,
        "candidates": [
            {"class": "A", "method": "m", "vtable": "0x1", "vtable_entry": "0x2",
             "provider": "p", "method_address": 0x4200},
            {"class": "B", "method": "n", "vtable": "0x3", "vtable_entry": "0x4",
             "provider": "p", "method_address": None},
        ],
        "ambiguous": True, "resolved": False,
    }
    out = _render_virtual_call_text(value)
    assert "0x4200" in out            # int -> hex
    assert "B" in out and "n" in out  # missing method_address still renders the line


def test_render_virtual_call_text_surfaces_warnings_alongside_candidates():
    # #706 follow-up (round-2 finding 9): a `resolved: true` result can still
    # carry a `warnings` entry (a DIFFERENT provider's vtable scan was
    # capped before it reached this slot) -- must render alongside the
    # resolved candidate, not only in the empty-candidates branch.
    from bn.formatters import _render_virtual_call_text
    value = {
        "callsite": "0x1000", "caller": "consumer", "slot_offset": "0x10",
        "slot_index": 2, "factory": None,
        "candidates": [{
            "provider": "self", "class": "Provider", "vtable": "0x9000",
            "vtable_entry": "0x9010", "method": "doWork", "method_address": "0x4100",
        }],
        "ambiguous": False, "resolved": True,
        "warnings": ["resolution may be incomplete: slot 2 is beyond the recovered "
                     "vtable window (scan capped at 2 slots) in at least one OTHER "
                     "provider that was not fully scanned for this slot -- it could "
                     "supply an additional candidate not reflected in "
                     "`resolved`/`ambiguous`"],
    }
    out = _render_virtual_call_text(value)
    assert "doWork" in out
    assert "warning:" in out
    assert "resolution may be incomplete" in out


def test_render_callsites_shows_null_hlil_reason_and_variadic_hint():
    # #557 + #558: text output surfaces the null-hlil reason code and the
    # variadic-callee steer.
    from bn.formatters import _render_callsites_text
    value = {"items": [{
        "callee": {"name": "sscanf", "address": "0x461746"},
        "containing_function": {"name": "parse_line", "address": "0x500000"},
        "call_addr": "0x500010", "caller_static": "0x500014",
        "hlil_statement": None, "hlil_statement_reason": "no_hlil_mapping",
        "call_instruction": {"address": "0x500010", "text": "bl sscanf"},
        "previous_instructions": [], "next_instructions": [],
        "callee_variadic": {"name": "sscanf", "is_variadic": True, "family": "scanf",
                            "format_arg_index": 1, "note": "..."},
    }], "total": 1, "has_more": False}
    out = _render_callsites_text(value)
    assert "hlil: null (no_hlil_mapping)" in out
    assert "variadic-callee: sscanf" in out
    assert "bn evidence function parse_line" in out


def test_render_evidence_shows_argument_confidence_and_variadic():
    # #549 + #558: evidence text tags argument confidence and the variadic warning.
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "parse_line", "address": "0x500000"},
        "prototype": "void parse_line()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [{
            "address": "0x500010", "operation": "LLIL_CALL", "direct": True,
            "hlil_statement": None, "hlil_statement_reason": "hlil_not_call_shaped",
            "argument_source": "hlil", "argument_confidence": "authoritative",
            "arguments": [{"text": "input"}],
            "argument_candidates": [{"source": "llil", "index": 0, "text": "r0", "confidence": "low"}],
            "variadic": {"is_variadic": True, "family": "scanf", "callee": "sscanf",
                         "under_recovered": True,
                         "warning": "imported variadic call `sscanf` under-recovered in HLIL: ..."},
        }],
    }
    out = _render_function_evidence_text(value)
    assert "hlil: null (hlil_not_call_shaped)" in out
    assert "arguments: (hlil authoritative)" in out
    assert "variadic: UNDER-RECOVERED" in out


def test_render_evidence_function_shows_recorded_local_tailcall_target_704():
    # #704 round 4: `_function_thunk_summary` records a resolved LOCAL branch
    # target (`is_candidate: False`, `target` populated) so a genuine
    # `j_`-style veneer lifted as LLIL_JUMP is not silently invisible in text
    # output -- the target must be rendered as a plain fact, without
    # asserting the thunk/veneer verdict the tool never established for a
    # local destination.
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "init_array_0", "address": "0x500000"},
        "prototype": "void init_array_0()", "calling_convention": "__cdecl",
        "thunk": {
            "is_candidate": False, "reason": None,
            "target": {"function": {"name": "init_helper", "address": "0x461746",
                                     "exact_start": True}},
        },
        "total_calls": 0, "matched_calls": 0, "offset": 0, "limit": None,
        "calls": [],
    }
    out = _render_function_evidence_text(value)
    assert "thunk: no" in out
    assert "candidate" not in out
    assert "init_helper @ 0x461746" in out


def test_render_evidence_function_notes_arity_mismatch_704():
    # #648/#704: a call demoted via `arity_mismatch` must state the reason in
    # text mode, not just print `arguments: (hlil inferred)` with no note.
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "parse_line", "address": "0x500000"},
        "prototype": "void parse_line()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [{
            "address": "0x500010", "operation": "LLIL_CALL", "direct": True,
            "argument_source": "hlil", "argument_confidence": "inferred",
            "arguments": [{"text": "1"}, {"text": "2"}],
            "argument_candidates": [],
            "arity_unknown": False, "arity_mismatch": True, "declared_arity": 3,
        }],
    }
    out = _render_function_evidence_text(value)
    assert "arity: MISMATCH" in out
    assert "2 argument(s)" in out and "declares 3" in out


def test_render_evidence_function_notes_callee_unresolved_704():
    # #648/#704: a call demoted via `callee_unresolved` (genuinely indirect,
    # or a resolved-but-unmatched direct destination) must state the reason,
    # even when it rendered NO arguments (so the note lives outside `if args:`).
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "dispatch", "address": "0x401800"},
        "prototype": "void dispatch()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [{
            "address": "0x401800", "operation": "LLIL_CALL", "direct": False,
            "argument_source": "hlil", "argument_confidence": "heuristic",
            "arguments": [], "argument_candidates": [],
            "arity_unknown": False, "indirect_call": True, "callee_unresolved": True,
        }],
    }
    out = _render_function_evidence_text(value)
    assert "arity: UNKNOWN — call target could not be resolved" in out


def test_render_orient_shows_existing_annotations():
    # #561: the orient card surfaces inherited-annotation counts + provenance hint.
    from bn.formatters import _render_orient_text
    value = {"kind": "orient_digest", "target": {"basename": "shared.bndb"},
             "analyzed": True, "analysis_state": "full", "function_count": 10,
             "existing_annotations": {"comments": 8, "function_comments": 3, "user_symbols": 12,
                                      "analysis_cache_restored": True,
                                      "provenance_hint": "existing BNDB annotations may predate this run"}}
    out = _render_orient_text(value)
    assert "existing annotations: comments=8" in out
    assert "user-symbols=12" in out and "cache-restored=True" in out
    assert "predate this run" in out


def test_render_session_status_single_job_names_the_poll_command():
    # Text mode is the DEFAULT for `session status`, so the human/agent driving a
    # detached load must be able to re-poll without going and re-reading docs.
    from bn.formatters import _render_session_status_text
    value = {
        "kind": "load_job",
        "job_id": "abc123",
        "state": "running",
        "terminal": False,
        "succeeded": None,
        "job": {"job_id": "abc123", "state": "running", "path": "/tmp/s.bndb"},
        "items": [{"job_id": "abc123", "state": "running", "path": "/tmp/s.bndb"}],
        "count": 1,
        "status_command": "bn -i worker session status abc123",
    }
    out = _render_session_status_text(value)
    assert "abc123  running  /tmp/s.bndb" in out
    assert "poll: bn -i worker session status abc123" in out
    # Concise: the poll hint plus the row, nothing else.
    assert len(out.splitlines()) == 2


def test_render_session_status_terminal_job_drops_the_poll_command():
    # Re-polling a finished job is pure waste; the note must disappear once the
    # job is terminal so text mode never contradicts `terminal: true`.
    from bn.formatters import _render_session_status_text
    value = {
        "kind": "load_job",
        "job_id": "abc123",
        "state": "complete",
        "terminal": True,
        "succeeded": True,
        "job": {
            "job_id": "abc123",
            "state": "complete",
            "path": "/tmp/s.bndb",
            "result": {"targets": [{"selector": "s.bndb"}]},
        },
        "items": [{
            "job_id": "abc123",
            "state": "complete",
            "path": "/tmp/s.bndb",
            "result": {"targets": [{"selector": "s.bndb"}]},
        }],
        "count": 1,
        "status_command": "bn -i worker session status abc123",
    }
    out = _render_session_status_text(value)
    assert "poll:" not in out
    assert "target: s.bndb" in out


def test_render_session_start_text_surfaces_reload_capture_failure():
    # Follow-up to PR #703 round 3: `reload_capture_failed` was invisible in
    # text mode -- only stderr and the exit code carried the signal. Default
    # stdout must now show an in-band line, mirroring the
    # `project_association_error` precedent.
    from bn.formatters import _render_session_start_text
    value = {
        "instance_id": "worker-1",
        "pid": 4242,
        "socket_path": "/tmp/worker-1.sock",
        "restarted": True,
        "loaded": [],
        "reload_capture_failed": True,
        "reload_capture_error": "OSError: connection refused",
    }
    out = _render_session_start_text(value)
    assert "target capture error: OSError: connection refused" in out
    assert "open targets could not be listed before the restart" in out


def test_render_session_start_text_omits_capture_error_when_absent():
    from bn.formatters import _render_session_start_text
    value = {
        "instance_id": "worker-1",
        "pid": 4242,
        "socket_path": "/tmp/worker-1.sock",
        "restarted": True,
        "loaded": [],
    }
    out = _render_session_start_text(value)
    assert "target capture error" not in out


def test_resolution_note_does_not_claim_containment_for_an_exact_start():
    # Bare-decimal input is disclosed with offset +0x0 when it names the exact
    # function start. The old text said "<addr> is inside <fn> @ <addr> (+0x0);
    # showing the containing function" -- which contradicts the JSON (offset 0 ==
    # exact start) and reads as if the read answered for a different function.
    from bn.formatters import _resolution_note
    value = {
        "function": {"name": "parse_packet", "address": "0x401000"},
        "resolved_from": {
            "requested_address": "0x401000",
            "offset": "+0x0",
            "input_format": "decimal",
        },
    }
    note = _resolution_note(value)
    assert "is inside" not in note
    assert "showing the containing function" not in note
    # It still discloses that a digit-only token was read as an address, which is
    # the whole point of the +0x0 disclosure.
    assert "decimal" in note
    assert "0x401000" in note


def test_resolution_note_discloses_decimal_input_on_an_interior_address():
    # JSON says input_format=decimal; text must say so too, or an agent working
    # in the default text mode never sees the documented disclosure.
    from bn.formatters import _resolution_note
    value = {
        "function": {"name": "parse_packet", "address": "0x401000"},
        "resolved_from": {
            "requested_address": "0x401010",
            "offset": "+0x10",
            "input_format": "decimal",
        },
    }
    note = _resolution_note(value)
    assert "0x401010" in note and "is inside parse_packet @ 0x401000 (+0x10)" in note
    assert "decimal" in note


def test_resolution_note_hex_interior_address_is_unchanged():
    from bn.formatters import _resolution_note
    value = {
        "function": {"name": "parse_packet", "address": "0x401000"},
        "resolved_from": {"requested_address": "0x401010", "offset": "+0x10"},
    }
    note = _resolution_note(value)
    assert "0x401010 is inside parse_packet @ 0x401000 (+0x10)" in note
    assert "showing the containing function" in note
    assert "decimal" not in note


def test_disasm_linear_steer_note_suppressed_for_an_exact_decimal_start():
    # The steer exists because `--count` slices from the PROLOGUE, not the
    # requested interior address. At offset +0x0 they are the same address, so
    # the advice is false and sends the agent to `--linear` for no reason.
    from bn.formatters import _disasm_linear_steer_note
    value = {
        "function": {"name": "parse_packet", "address": "0x401000"},
        "resolved_from": {
            "requested_address": "0x401000",
            "offset": "+0x0",
            "input_format": "decimal",
        },
    }
    assert _disasm_linear_steer_note(value, sliced=True) == ""
    interior = {
        "function": {"name": "parse_packet", "address": "0x401000"},
        "resolved_from": {"requested_address": "0x401010", "offset": "+0x10"},
    }
    assert "--linear" in _disasm_linear_steer_note(interior, sliced=True)
