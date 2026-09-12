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


def _callsite_row(call_addr: str) -> dict:
    return {
        "callee": {"name": "rotl8", "address": "0x401156"},
        "containing_function": {"name": "encrypt", "address": "0x40118b"},
        "call_addr": call_addr, "caller_static": call_addr,
        "call_instruction": {"address": call_addr, "text": "call rotl8"},
        "previous_instructions": [], "next_instructions": [],
    }


def test_render_callsites_footer_on_partial_last_page():
    # #611: a partial LAST page (offset > 0, has_more False, fewer rows than the
    # total) must still say so -- it used to read as the complete result.
    from bn.formatters import _render_callsites_text
    value = {
        "items": [_callsite_row("0x500010")],
        "total": 3, "offset": 2, "has_more": False,
    }
    out = _render_callsites_text(value)
    assert "showing 1 of 3 callsites" in out
    assert "offset 2" in out


def test_render_callsites_last_page_footer_omits_page_forward_hint():
    # finding 3 (round 2): a true last page (has_more False) must not repeat
    # the page-forward hint that only makes sense mid-page.
    from bn.formatters import _render_callsites_text
    value = {
        "items": [_callsite_row("0x500010")],
        "total": 3, "offset": 2, "has_more": False,
    }
    out = _render_callsites_text(value)
    assert "--offset/--limit" not in out


def test_render_callsites_mid_page_footer_keeps_page_forward_hint():
    from bn.formatters import _render_callsites_text
    value = {
        "items": [_callsite_row("0x500010")],
        "total": 47, "offset": 0, "has_more": True,
    }
    out = _render_callsites_text(value)
    assert "--offset/--limit" in out


def test_render_callsites_over_shot_page_still_footers():
    # finding 3 (round 2): an --offset past the end must report the true
    # total, not "no callsites found" (which reads as "never called"), and
    # must not begin with leading blank lines.
    from bn.formatters import _render_callsites_text
    value = {"items": [], "total": 47, "offset": 60, "has_more": False}
    out = _render_callsites_text(value)
    assert "showing 0 of 47 callsites (offset 60)" in out
    assert not out.startswith("\n")


def test_render_callsites_no_footer_on_complete_single_page():
    from bn.formatters import _render_callsites_text
    value = {
        "items": [_callsite_row("0x500010"), _callsite_row("0x500020")],
        "total": 2, "offset": 0, "has_more": False,
    }
    out = _render_callsites_text(value)
    assert "showing" not in out


def test_render_callsites_empty_string_row_does_not_read_as_no_callsites():
    # A row that renders to a literal empty string (fallback text for a raw ""
    # item) must not be silently dropped into the "no callsites found" fallback
    # -- that misrepresents a one-row page as a zero-result page.
    from bn.formatters import _render_callsites_text
    value = {"items": [""]}
    out = _render_callsites_text(value)
    assert out != "no callsites found"


def test_render_callsites_string_offset_footer_renders_unknown_not_fabricated():
    # #619: a string offset ("60") must not coerce to the same footer text as
    # the int 60 -- _fmt_offset must disclose it as unknown instead of
    # fabricating a specific page position the payload never stated as an int.
    from bn.formatters import _render_callsites_text
    value = {"items": [], "total": 47, "offset": "60", "has_more": False}
    out = _render_callsites_text(value)
    assert "(offset <unknown>)" in out
    assert "(offset 60)" not in out


def test_render_callsites_non_int_total_empty_page_does_not_assert_zero():
    # #619 follow-up: an empty page whose `total` is not an int is not a
    # confirmed zero-result page -- claiming "no callsites found" here is
    # the same confidently-wrong shape the over-shot-page fix (F3) stopped
    # fabricating for a bad offset. The total is unusable, so say so instead
    # of asserting zero.
    from bn.formatters import _render_callsites_text
    value = {"items": [], "total": "47", "offset": 60}
    out = _render_callsites_text(value)
    assert out != "no callsites found"
    assert "total count is not a number" in out
    assert "(offset 60)" in out


def test_render_capabilities_malformed_element_renders_placeholder():
    # #619: a malformed (non-dict) list element must degrade to a placeholder
    # line, not raise inside the renderer.
    from bn.formatters import _render_capabilities_text
    value = {"items": [
        {"group": "read", "command": "bn read", "help": "read bytes"},
        None,
    ]}
    out = _render_capabilities_text(value)
    assert "bn read" in out
    assert "None" in out


def test_render_capabilities_malformed_string_element_renders_repr():
    # finding 5: a string element must not read as a real row -- repr()
    # visibly distinguishes it (quoted) from a real "  cmd  --  help" row.
    from bn.formatters import _render_capabilities_text
    value = {"items": [
        {"group": "read", "command": "bn read", "help": "read bytes"},
        "disabled",
    ]}
    out = _render_capabilities_text(value)
    assert "  'disabled'" in out
    assert "\n  disabled\n" not in out + "\n"


def test_render_callgraph_malformed_element_renders_placeholder():
    from bn.formatters import _render_callgraph_text
    value = {
        "function": {"name": "main", "address": "0x401000"},
        "callees": [{"kind": "direct", "call_addr": "0x401010",
                     "target": {"name": "helper", "address": "0x401200"}}, "bad"],
        "callers": ["bad", {"caller": {"name": "start", "address": "0x400ff0"},
                            "call_addr": "0x401000"}],
    }
    out = _render_callgraph_text(value)
    assert "helper" in out
    assert "start" in out
    assert "'bad'" in out


def test_render_callgraph_malformed_function_field_does_not_raise():
    # #619: a string `function` field must degrade, not crash the header.
    from bn.formatters import _render_callgraph_text
    value = {"function": "main", "callees": []}
    out = _render_callgraph_text(value)
    assert "<unknown> @ <unknown>" in out


def test_render_taint_models_malformed_element_renders_placeholder():
    from bn.formatters import _render_taint_models_text
    value = {
        "sources": [{"symbol": "gets", "to": "*arg:0"}, "bad"],
        "sinks_by_class": {"unbounded_input": ["bad", {
            "symbol": "gets", "tainted_args": [0], "present": True,
        }]},
        "propagators": ["bad", {"symbol": "strcpy", "from_to": "arg1 -> arg0"}],
        "overlays": [None, {"path": "extra_models.json"}],
    }
    out = _render_taint_models_text(value)
    assert "gets" in out
    assert "strcpy" in out
    assert "extra_models.json" in out
    assert "'bad'" in out


def test_render_taint_models_sink_entry_missing_symbol_does_not_raise():
    # #619: a sink entry dict missing "symbol" must degrade, not KeyError.
    from bn.formatters import _render_taint_models_text
    value = {"sinks_by_class": {"unbounded_input": [{"tainted_args": [0], "present": True}]}}
    out = _render_taint_models_text(value)
    assert "<unknown>" in out


def test_render_field_xrefs_offset_none_renders_unknown_not_zero():
    # #619: a None/uncoercible offset must render as an explicit placeholder,
    # never the fabricated "+0x0" (indistinguishable from a real offset-0 field).
    from bn.formatters import _render_field_xrefs_text
    value = {"field": {"type_name": "Widget", "field_name": "flags", "offset": None,
                        "field_type": "uint32_t"}, "items": []}
    out = _render_field_xrefs_text(value)
    assert "Widget.flags @ +<unknown: None>" in out
    assert "+0x0" not in out


def test_render_field_xrefs_offset_hex_string_renders_unknown_not_zero():
    from bn.formatters import _render_field_xrefs_text
    value = {"field": {"type_name": "Widget", "field_name": "flags", "offset": "0x40",
                        "field_type": "uint32_t"}, "items": []}
    out = _render_field_xrefs_text(value)
    assert "+<unknown: '0x40'>" in out


def test_render_field_xrefs_missing_offset_key_renders_unknown_not_zero():
    # F2: the call site must not default a missing "offset" key to 0 -- that
    # fabricates a real offset-0 member for a struct field the payload never
    # gave an offset for.
    from bn.formatters import _render_field_xrefs_text
    value = {"field": {"type_name": "Widget", "field_name": "flags",
                        "field_type": "uint32_t"}, "items": []}
    out = _render_field_xrefs_text(value)
    assert "Widget.flags @ +<unknown: None>" in out
    assert "+0x0" not in out


def test_render_field_xrefs_non_dict_field_renders_unknown_not_zero():
    # F2: a malformed (non-dict) "field" degrades to placeholder names, and
    # the offset must degrade alongside them instead of impersonating +0x0.
    from bn.formatters import _render_field_xrefs_text
    value = {"field": "not-a-dict", "items": []}
    out = _render_field_xrefs_text(value)
    assert "<unknown>.<unknown> @ +<unknown: None>" in out
    assert "+0x0" not in out


# --- #619: finish _as_dict adoption / soft-degrade malformed list elements ----
# Same class as the #101 function-info fix: a nested field or list element that
# is not the shape the renderer assumed (version skew, partial bridge payload)
# must degrade to placeholder text, not raise -- the CLI turns the raise into a
# BridgeError that costs the agent the whole partial text view.
#
# Degrading must stay LOUD. Coercing a malformed container to empty renders the
# same row a genuinely empty result does, so the caller reads a confident
# "nothing here" and cannot tell the payload was unusable -- a wrong answer
# nobody can detect, which is worse than the AttributeError it replaced. Every
# test below therefore pins the DISCLOSURE, not just the absence of a raise.

def test_render_field_xrefs_non_dict_item_degrades():
    from bn.formatters import _render_field_xrefs_text
    value = {"field": {"type_name": "T", "field_name": "n", "offset": 0},
             "items": [{"kind": "code", "address": "0x1"}, "bad"]}
    out = _render_field_xrefs_text(value)
    assert "0x1" in out
    # The malformed ref is neither code nor data, so both kind filters drop it;
    # it must still be disclosed instead of vanishing from a ref inventory.
    assert "'bad'" in out


def test_render_local_list_non_dict_function_degrades():
    from bn.formatters import _render_local_list_text
    out = _render_local_list_text({"function": "main", "items": []})
    assert "<unknown> @ <unknown>" in out


def test_render_local_list_non_dict_element_renders_placeholder():
    from bn.formatters import _render_local_list_text
    out = _render_local_list_text({
        "function": {"name": "f", "address": "0x1"},
        "items": [{"name": "count", "type": "int"}, "bad"],
    })
    assert "count" in out
    assert "'bad'" in out


def test_render_function_info_verbose_non_dict_local_element_degrades():
    from bn.formatters import _render_function_info_text
    out = _render_function_info_text(
        {"function": {"name": "f"}, "parameters": ["bad"]}, verbose=True)
    assert "'bad'" in out


def test_render_defuse_non_dict_nested_fields_degrade():
    from bn.formatters import _render_defuse_text
    out = _render_defuse_text({"function": "main", "variable": "v",
                               "is_phi": True, "phi_sources": ["bad"],
                               "uses": ["bad"]})
    assert "<unknown> @ <unknown>" in out
    assert "'bad'" in out


def test_render_values_non_dict_function_and_values_degrade():
    from bn.formatters import _render_values_text
    out = _render_values_text({"function": "main", "possible_values": "bad"})
    assert "<unknown> @ <unknown>" in out
    # `<unavailable>` is what an ABSENT possible_values prints: a present but
    # malformed one must not impersonate "the analysis had no answer".
    assert "possible values: <malformed: 'bad'>" in out


def test_render_class_show_non_dict_members_degrade():
    from bn.formatters import _render_class_show_text
    out = _render_class_show_text({
        "name": "Widget", "bases": ["bad"],
        "methods": ["bad"],
        "vtable": {"address": "0x1", "slots": ["bad"]},
        "instances": {"construction_sites": ["bad"], "stored_globals": ["bad"]},
    })
    assert "class Widget" in out
    assert "'bad'" in out


def test_render_class_list_dict_base_rows_render_names():
    # "bases" arrives as name-dicts (what class show already renders); the list
    # renderer must not blow up joining them as strings.
    from bn.formatters import _render_class_list_text
    out = _render_class_list_text(
        {"items": [{"name": "Widget", "bases": [{"name": "Base"}]}]})
    assert "Base" in out


def test_render_taint_non_dict_nested_fields_degrade():
    from bn.formatters import _render_taint_text
    out = _render_taint_text({"function": "main", "direction": "forward",
                              "reached_sinks": ["bad"], "leaves": ["bad"],
                              "by_source": {"0x1": "bad"}})
    assert "forward taint in <unknown> @ <unknown>" in out
    assert "'bad'" in out


def test_render_taint_backward_non_dict_slices_degrade():
    from bn.formatters import _render_taint_text
    out = _render_taint_text({"function": {"name": "f", "address": "0x1"},
                              "direction": "backward", "slices": ["bad"],
                              "sink_status": ["bad"]})
    assert "backward taint in f @ 0x1" in out
    assert "'bad'" in out


def test_render_taint_models_non_dict_sinks_by_class_degrades():
    from bn.formatters import _render_taint_models_text
    out = _render_taint_models_text({"sources": [{"symbol": "gets"}],
                                     "sinks_by_class": ["bad"]})
    assert "gets" in out
    # Dropping the whole sink section silently would read as "no sinks modeled".
    assert "malformed sinks_by_class field" in out


def test_render_taint_models_sink_entry_non_dict_callsite_degrades():
    from bn.formatters import _render_taint_models_text
    value = {"sinks_by_class": {"unbounded_input": [
        {"symbol": "gets", "callsites": ["bad"]}]}}
    out = _render_taint_models_text(value)
    assert "gets" in out
    assert "'bad'" in out


def test_render_surface_non_dict_summary_degrades():
    from bn.formatters import _render_surface_text
    out = _render_surface_text({"summary": "bad"})
    # A clean scan with nothing to report prints all zeros; a malformed summary
    # must not render byte-identically to it.
    assert "hidden surface: ? init section(s)" in out
    assert "malformed summary field" in out
    assert out != _render_surface_text({})


def test_render_orient_non_dict_target_and_kind_breakdown_degrade():
    from bn.formatters import _render_orient_text
    out = _render_orient_text({"target": "bad",
                               "imports_summary": {"total": 3, "by_kind": ["bad"]}})
    assert "orientation: <target>" in out
    assert "imports: 3" in out


def test_render_cfg_non_dict_nested_fields_degrade():
    from bn.formatters import _render_cfg_text
    out = _render_cfg_text({"function": "bad", "blocks": ["bad"]})
    assert "? @ ? (?)" in out
    assert "'bad'" in out


def test_render_data_vars_non_dict_row_degrades():
    from bn.formatters import _render_data_vars_text
    out = _render_data_vars_text({"items": ["bad"], "total": 1})
    assert "'bad'" in out


def test_render_imports_summary_non_dict_breakdowns_degrade():
    from bn.formatters import _render_imports_summary_text
    out = _render_imports_summary_text({"total_symbols": 2, "namespaces": ["bad"],
                                        "by_kind": ["bad"]})
    assert "total symbols: 2" in out
    # Both breakdowns are unusable; skipping them silently reads as "no imports
    # in any namespace", which the non-zero total contradicts.
    assert "malformed namespaces field" in out
    assert "malformed by_kind field" in out


def test_render_imports_summary_uncoercible_count_renders_and_orders():
    # The sort key coerces the count, but the column interpolated the RAW value
    # one line later: `f"{None:>5}"` is a TypeError, so a single malformed count
    # still cost the whole text view (#619).
    from bn.formatters import _render_imports_summary_text
    out = _render_imports_summary_text(
        {"total_symbols": 7, "namespaces": {"lo": 1, "hi": 5, "broken": None}})
    rows = [ln for ln in out.splitlines() if ln.startswith("  ")]
    assert [r.split()[-1] for r in rows] == ["hi", "lo", "broken"]
    assert "<unknown>" in out


def test_render_local_list_malformed_items_alias_keeps_the_retained_alias_rows():
    # `items` is canonical, `locals` the retained alias (#651). A truthy but
    # malformed `items` short-circuited the alias away, so the real rows
    # vanished behind a confident "no locals".
    from bn.formatters import _render_local_list_text
    out = _render_local_list_text({
        "function": {"name": "f", "address": "0x1"},
        "items": "bad",
        "locals": [{"name": "count", "type": "int"}],
    })
    assert "count" in out
    assert "no locals" not in out
    assert "malformed items field" in out


def test_render_class_list_malformed_items_alias_is_disclosed_not_zero():
    from bn.formatters import _render_class_list_text
    out = _render_class_list_text({"items": "bad", "classes": [{"name": "Widget"}],
                                   "total": 1})
    assert "Widget" in out
    assert "classes: 0 shown of 1" not in out
    assert "malformed items field" in out


def test_render_orient_non_string_sample_sections_do_not_raise():
    from bn.formatters import _render_orient_text
    out = _render_orient_text({"strings_sample": {"items": [], "sample_sections": [1, 2]}})
    assert "from 1, 2" in out


def test_render_orient_non_list_strings_sample_items_is_disclosed_not_counted():
    from bn.formatters import _render_orient_text
    out = _render_orient_text({"strings_sample": {"items": "abc"}})
    assert "sample 3 of 3" not in out          # 3 characters are not 3 strings
    assert "malformed items field" in out


def test_render_orient_non_dict_imports_summary_is_disclosed_not_dropped():
    from bn.formatters import _render_orient_text
    out = _render_orient_text({"imports_summary": "bad"})
    assert "malformed imports_summary field" in out


def test_render_grouped_leaves_distinct_malformed_leaves_each_render():
    # Grouping malformed leaves by TYPE collapsed every distinct one behind an
    # `(xN)` count on the first, so all but one were unreadable.
    from bn.formatters import _render_grouped_leaves
    out = "\n".join(_render_grouped_leaves(["alpha", "beta", "alpha"]))
    assert "'alpha'" in out and "'beta'" in out
    assert "(x2)" in out          # two IDENTICAL malformed leaves still collapse


def test_render_data_symbols_non_dict_row_degrades():
    from bn.formatters import _render_data_symbols_text
    out = _render_data_symbols_text({"items": [{"a": "0x1", "n": "sym"}, "bad"]})
    assert "sym" in out
    assert "'bad'" in out


def test_render_data_symbols_malformed_items_is_disclosed_not_none():
    from bn.formatters import _render_data_symbols_text
    out = _render_data_symbols_text({"items": "bad", "total": 3})
    assert out != "none"
    assert "malformed items field" in out


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


def test_render_trace_text_header_names_callee_not_parameter():
    # #662: the header names the resolved CALLEE, not a callee parameter name
    # -- the value being sliced is the caller's operand at the callsite, not
    # necessarily anything the callee's parameter is called.
    from bn.formatters import _render_trace_text
    value = {
        "function": "encrypt",
        "function_address": "0x401180",
        "target_address": "0x4011f2",
        "arg_index": 1,
        "arg_label": {"index": 1, "register": "rsi", "name": "count", "callee": "rotl8"},
        "trace": [
            {"ssa_var": "arg1#1", "ssa_label": "arg1#1", "depth": 0,
             "terminates": True, "reason": "function_parameter"},
        ],
    }
    out = _render_trace_text(value)
    assert "backward trace of arg[1] of rotl8 (rsi) in encrypt @ 0x4011f2" in out
    assert '"count"' not in out
    assert "count" not in out.splitlines()[0]


def test_render_trace_text_header_omits_missing_callee_or_register():
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {"index": 0}, "trace": [],
    }
    out = _render_trace_text(value)
    assert "backward trace of arg[0] in f @ 0x1010" in out


def test_render_trace_text_renders_caveats_from_assumptions():
    # #671: a non-empty `assumptions` list renders a `caveats (N):` block,
    # mirroring `_render_taint_text`.
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {}, "truncated": True,
        "trace": [
            {"ssa_var": "v#1", "ssa_label": "v#1", "depth": 0,
             "terminates": True, "reason": "undefined_or_global"},
        ],
        "assumptions": ["depth cap reached (1); slice is incomplete -- raise --max-depth to continue"],
    }
    out = _render_trace_text(value)
    assert "caveats (1):" in out
    assert "depth cap reached (1)" in out


def test_render_trace_text_empty_trace_still_renders_caveats():
    # Defensive hardening (round-2 finding F): an empty `trace` short-circuited
    # before the caveats block, which would silently drop any `assumptions` if
    # a future producer ever populated `assumptions` on a zero-step result.
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {}, "trace": [],
        "assumptions": ["crossing was not attempted at one or more call boundaries "
                        "because the --ip-depth budget was spent before reaching "
                        "them; the slice may be incomplete beyond those boundaries"],
    }
    out = _render_trace_text(value)
    assert "caveats (1):" in out
    assert "--ip-depth budget was spent" in out


def test_render_trace_text_omits_caveats_when_assumptions_empty():
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {},
        "trace": [
            {"ssa_var": "v#1", "ssa_label": "v#1", "depth": 0,
             "terminates": True, "reason": "function_parameter"},
        ],
        "assumptions": [],
    }
    out = _render_trace_text(value)
    assert "caveats" not in out


def test_render_trace_text_intra_out_param_reason_shows_callee():
    # #672: the intra-mode reason renders the same "(via <callee>)" suffix as
    # the interprocedural one.
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {},
        "trace": [
            {"ssa_var": "local#1", "ssa_label": "local#1", "depth": 0,
             "terminates": True, "reason": "out_param_not_followed",
             "out_param_callee": "parse_input"},
        ],
    }
    out = _render_trace_text(value)
    assert "out-param fill not followed (via parse_input)" in out


def test_go_rename_summary_shares_one_builder_with_the_mutation_summary(monkeypatch):
    """#685: `go rename` reports through its own go_* counters but emits the SAME
    compact-status schema as every other mutation, so both summaries must be
    produced by ONE builder. They used to be separate dict literals, and the
    copy drifted twice into defects the shared path already handled: a failure
    row's `first_error`, and a revert that fails after every rename verified.
    Pin the sharing (a change to the builder reaches both) and the shared rules
    (the same OUTCOME reaches the same keys through either path)."""
    from bn import formatters

    real_builder = formatters._build_mutation_summary
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(formatters, "_build_mutation_summary", spy)

    measured = formatters._mutation_summary({
        "success": True, "committed": True, "preview": False, "rolled_back": False,
        "results": [{"status": "verified"}]})
    go = formatters._go_rename_summary({
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "rolled_back": False, "results": [], "go_renamed_candidates": 3,
        "go_committed_count": 3, "go_verified_count": 3, "go_failed_count": 0,
        "skipped_user_named": 1})

    # One builder produced both -- neither path can drift from the other again.
    assert len(calls) == 2
    assert measured["changed_count"] == 1 and go["changed_count"] == 3
    assert set(measured) == set(go)

    # A preview whose revert failed: zero failure rows, explanation only in the
    # top-level message. Both paths must call it dirty AND carry the error --
    # reporting failed=0 with no error while the view is partially renamed is
    # the defect the copy shipped once.
    message = "preview rollback failed; the view is partially renamed"
    mutation_stuck = formatters._mutation_summary({
        "success": False, "committed": False, "preview": True, "rolled_back": False,
        "message": message, "results": [{"status": "verified"}]})
    go_stuck = formatters._go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": True,
        "rolled_back": False, "message": message, "results": [],
        "go_renamed_candidates": 5, "go_committed_count": 0,
        "go_verified_count": 5, "go_failed_count": 0, "skipped_user_named": 0})
    for key in ("kind", "ok", "success", "committed", "preview", "measured",
                "rolled_back", "dirty_after", "first_error"):
        assert mutation_stuck[key] == go_stuck[key]
    assert go_stuck["first_error"] == message

    # A failure whose ONLY explanation is a results[] row: `first_error` must be
    # read off the row on both paths, never dropped to a bare failed count.
    reason = "rename is unsupported on this view"
    mutation_row = formatters._mutation_summary({
        "success": False, "committed": False, "rolled_back": True,
        "results": [{"status": "unsupported", "message": reason}]})
    go_row = formatters._go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": False,
        "rolled_back": True,
        "results": [{"status": "unsupported", "message": reason}],
        "go_renamed_candidates": 5, "go_verified_count": 0, "go_committed_count": 0,
        "go_failed_count": 1, "skipped_user_named": 0})
    assert mutation_row["failed_count"] == go_row["failed_count"] == 1
    assert mutation_row["first_error"] == go_row["first_error"] == reason
