from __future__ import annotations

import copy
import functools
import inspect
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
    # A malformed breakdown NESTED inside a well-formed imports_summary renders
    # the same line the breakdown simply being absent does, so it is recorded at
    # the same choke point and named in the same note as the top-level skew.
    assert "malformed by_kind, target fields" in out


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
    assert "malformed by_kind, namespaces fields" in out


def test_render_imports_summary_uncoercible_count_renders_and_orders():
    # The sort key coerces the count, but the column interpolated the RAW value
    # one line later: `f"{None:>5}"` is a TypeError, so a single malformed count
    # still cost the whole text view (#619). The placeholder also has to fit the
    # right-aligned column, or one bad row skews the whole table.
    from bn.formatters import _render_imports_summary_text
    out = _render_imports_summary_text(
        {"total_symbols": 7, "namespaces": {"lo": 1, "hi": 5, "broken": None}})
    assert [ln for ln in out.splitlines() if ln.startswith("  ")] == [
        "      5  hi", "      1  lo", "      ?  broken"]


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


def test_the_disclosure_reaches_an_early_return_path():
    # The whole point of declaring the coerced keys per renderer instead of
    # appending a line per branch: several renderers bail out BEFORE their
    # normal tail -- "none", "no instance has a binary matching ...", the
    # no-possible-values return -- and a per-branch line misses exactly those.
    from bn.formatters import (_render_data_symbols_text,
                               _render_instance_find_text, _render_values_text)
    listing = _render_data_symbols_text({"items": "bad", "total": 3})
    assert listing != "none" and "malformed items field" in listing

    found = _render_instance_find_text({"query": "q", "items": "bad"})
    assert found.startswith("no instance has a binary matching")
    assert "malformed items field" in found

    # Here the field that bails out (an absent possible_values) is NOT the field
    # that is malformed, so only a wrapper around every return path discloses it.
    values = _render_values_text({"function": "bad"})
    assert "possible values: <unavailable>" in values
    assert "malformed function field" in values


def _coercion_sites(source: str | None = None):
    """Every named-field container coercion in the formatter module, and every
    coercion that BYPASSES the recording helpers, read out of the module's AST.

    Derived, never hand-listed. The first attempt at this guard was a table of
    the renderers' own `@_discloses(lists=..., dicts=...)` declarations, which is
    a restatement of the implementation: a declaration list and a table of that
    same list agreeing with each other proves nothing.

    `source` parses a module given as TEXT instead, so both what this recognises
    and what it is BLIND to can be asserted as data rather than claimed in prose.
    Every round of this PR that described the guard's reach in prose described it
    wrongly; `_GUARD_CATCHES` and `_GUARD_BLIND` below are the actual statement.

    Scope is every function, method, lambda and module-level statement -- not
    just top-level `def`s -- because "inside a class" and "at module level" were
    two of the evasions review found. A payload lookup is `.get(k)` with a
    literal or variable key, or a `[k]` subscript, reached directly or through
    one local alias. Five rules then fire on it:

      R1 `_as_dict(<lookup>)` / `_as_list(<lookup>)` -- the raw coercers.
      R2 `<lookup> or []` / `{}` / `()` / `set()` / a module-level empty constant.
      R3 `.get/.pop/.setdefault(k, <empty container>)` -- a defaulted lookup.
      R4 the `isinstance` ternary in BOTH orientations, defaulting to an empty
         container, or to `None` when the test decides the shape of the value
         being bound (`None` is what the helpers read as ABSENT, so it buries a
         malformed value exactly as completely as `[]` does).
      R5 the STATEMENT family: a name bound BOTH from an expression containing a
         payload lookup AND to an empty container -- which covers if/else, the
         negated rebind, pre-initialise-then-assign, `try/except TypeError`, a
         walrus, an annotated assign and tuple unpacking without enumerating any
         of them -- plus the same rebind to `None` when a container-shape test on
         that same expression is what gates it.

    R5 deliberately does NOT require the assigned name to come from a BARE
    lookup: `rows = list(v.get(k))` inside the shape branch is the spelling a
    maintainer adding a defensive copy would reach for, and requiring bareness is
    how the round-6 version of this rule stayed blind to it.

    The remaining blind spots are asserted, not described -- see `_GUARD_BLIND`.
    They are why the ENUMERATED differential below RUNS the renderers instead of
    reading them, and why no claim of completeness is made here."""
    import ast
    import inspect

    from bn import formatters

    tree = ast.parse(source if source is not None else inspect.getsource(formatters))
    RECORDERS = ("_field_list", "_field_dict")
    COERCERS = ("_as_list", "_as_dict")
    CONTAINERS = ("list", "dict", "tuple", "set", "frozenset")

    def is_lookup(node):
        """A read of a NAMED field off some mapping, by `.get(k)` or `[k]`.

        A subscript counts only when the key is a string literal or a variable:
        `rows[-1]` is an index into a list, which carries no field name to
        disclose and whose element-level coercion is already rendered inline."""
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args):
            return True
        if not isinstance(node, ast.Subscript):
            return False
        key = node.slice
        return isinstance(key, ast.Name) or (
            isinstance(key, ast.Constant) and isinstance(key.value, str))

    def bound_names(target):
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for e in target.elts for n in bound_names(e)]
        if isinstance(target, ast.Starred):
            return bound_names(target.value)
        return []                       # a subscript/attribute store rebinds nothing

    def bindings(node):
        """(names, value) pairs this statement binds, distributing a tuple assign
        element-wise so `rows, n = [], 0` is seen as binding `rows` to `[]`."""
        if isinstance(node, ast.Assign):
            pairs = []
            for target in node.targets:
                if (isinstance(target, (ast.Tuple, ast.List))
                        and isinstance(node.value, (ast.Tuple, ast.List))
                        and len(target.elts) == len(node.value.elts)):
                    pairs += [(bound_names(t), v)
                              for t, v in zip(target.elts, node.value.elts)]
                else:
                    pairs.append((bound_names(target), node.value))
            return pairs
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return [(bound_names(node.target), node.value)] if node.value else []
        return []

    empty_consts: set[str] = set()

    def literal_empty(node):
        """An empty container written out, with no name indirection."""
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts:
            return True
        if isinstance(node, ast.Dict) and not node.keys:
            return True
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in CONTAINERS and not node.args and not node.keywords)

    def empty_container(node):
        """...or a NAME bound to one, at module or function scope. A default
        taken from `_EMPTY = []` coerces exactly as the literal does."""
        return literal_empty(node) or (isinstance(node, ast.Name)
                                       and node.id in empty_consts)

    def is_none(node):
        return isinstance(node, ast.Constant) and node.value is None

    def container_isinstance(node, negated=None):
        """The expression whose CONTAINER shape `node` tests, else None.
        `negated=True` asks only for the `not isinstance(...)` spelling, which is
        what distinguishes a SKIP from a dispatch."""
        neg = False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node, neg = node.operand, True
        if negated is not None and neg is not negated:
            return None
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance" and len(node.args) == 2):
            return None
        wanted = node.args[1]
        names = wanted.elts if isinstance(wanted, ast.Tuple) else [wanted]
        if not any(isinstance(n, ast.Name) and n.id in CONTAINERS for n in names):
            return None
        return node.args[0]

    # A module-level `_EMPTY = []` is as good a coercion default as the literal.
    module_empties = {n for stmt in tree.body for names, val in bindings(stmt)
                      if literal_empty(val) for n in names}
    empty_consts = module_empties

    def scope_nodes(scope):
        """Every node belonging to this scope, not descending into a nested
        function/class -- those are separate scopes with their own aliases."""
        out, stack = [], list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            out.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(node))
        return out

    scopes = [("<module>", tree)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append((node.name, node))
        elif isinstance(node, ast.Lambda):
            scopes.append(("<lambda>", node))
        elif isinstance(node, ast.ClassDef):
            scopes.append((f"class {node.name}", node))

    sites, raw = [], []
    for fn_name, scope in scopes:
        # The recording helpers ARE the choke point: they necessarily test the
        # shape of a lookup and fall back, which is the very shape being hunted
        # everywhere else. Scanning them would report the fix as the defect.
        if fn_name in RECORDERS + COERCERS:
            continue
        nodes = scope_nodes(scope)
        # A FUNCTION-LOCAL `_empty = []` defaults exactly as a module-level one.
        local_empties = {n for node in nodes for names, val in bindings(node)
                         if literal_empty(val) for n in names}
        empty_consts = module_empties | local_empties
        aliases = {n for node in nodes for names, val in bindings(node)
                   if is_lookup(val) for n in names}
        # `g = value.get` then `g("k")`: a bound-method alias is still a lookup.
        getters = {n for node in nodes for names, val in bindings(node)
                   if isinstance(val, ast.Attribute) and val.attr == "get" for n in names}

        def is_lookup_here(node):
            if is_lookup(node):
                return True
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in getters and bool(node.args))

        def suspect(node):
            return is_lookup_here(node) or (isinstance(node, ast.Name) and node.id in aliases)

        def suspect_within(node):
            return any(suspect(n) for n in ast.walk(node))

        lookup_bound: dict[str, list] = {}
        container_default: set[str] = set()
        none_default: set[str] = set()
        for node in nodes:
            for names, val in bindings(node):
                if suspect_within(val):
                    for n in names:
                        lookup_bound.setdefault(n, []).append(val)
                if empty_container(val):
                    container_default |= set(names)
                elif is_none(val):
                    none_default |= set(names)

        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in RECORDERS:
                    base = node.args[0] if node.args else None
                    top = isinstance(base, ast.Name) and base.id == "value"
                    keys = [a.value for a in node.args[1:] if isinstance(a, ast.Constant)]
                    kind = "list" if node.func.id == "_field_list" else "dict"
                    sites.append((fn_name, tuple(keys), kind, top))
                elif node.func.id in COERCERS and node.args and suspect_within(node.args[0]):
                    raw.append((fn_name, "R1 " + ast.unparse(node)))          # R1
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):  # R2
                if empty_container(node.values[-1]):
                    for c in node.values[:-1]:
                        if suspect_within(c):
                            raw.append((fn_name, "R2 " + ast.unparse(node)))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop", "setdefault")
                    and len(node.args) == 2 and empty_container(node.args[1])):
                raw.append((fn_name, "R3 " + ast.unparse(node)))              # R3
            if isinstance(node, ast.IfExp):                                   # R4
                tested = container_isinstance(node.test)
                for real, default in ((node.body, node.orelse), (node.orelse, node.body)):
                    if not suspect_within(real):
                        continue
                    # The tested expression must BE the value being bound, not
                    # merely appear inside it: `fn.get("name") if isinstance(fn,
                    # dict) else None` guards the SOURCE and binds a string, and
                    # reporting that would make the guard fire on correct code.
                    shaped = tested is not None and ast.dump(tested) == ast.dump(real)
                    # ...unless the test is itself a LOOKUP that the chosen branch
                    # re-reads, which is a container coercion whatever the default
                    # is: `len(v.get(k)) if isinstance(v.get(k), list) else 0`.
                    reread = (tested is not None and is_lookup_here(tested)
                              and any(ast.dump(tested) == ast.dump(n) for n in ast.walk(real)))
                    if empty_container(default) or (shaped and is_none(default)) or reread:
                        raw.append((fn_name, "R4 " + ast.unparse(node)))
                        break
            if isinstance(node, ast.If):                                      # R6
                # The SKIP form: a wrong shape silently leaves the function or
                # the loop. Only the NEGATED spelling -- `if isinstance(x, list):
                # ... return` is a dispatch, not a skip -- and not when the exit
                # hands back the raw payload echo, which discloses everything.
                tested = container_isinstance(node.test, negated=True)
                if tested is not None and suspect(tested):
                    exits = [s for s in node.body
                             if isinstance(s, (ast.Return, ast.Continue, ast.Break))]
                    echoes = any(isinstance(s, ast.Return) and isinstance(s.value, ast.Call)
                                 and isinstance(s.value.func, ast.Name)
                                 and "fallback" in s.value.func.id for s in node.body)
                    if exits and not echoes:
                        raw.append((fn_name, "R6 skip on " + ast.unparse(node.test)))
            if isinstance(node, ast.Match) and suspect_within(node.subject):  # R7
                if any(isinstance(c.pattern, (ast.MatchValue, ast.MatchSequence,
                                              ast.MatchMapping)) for c in node.cases):
                    raw.append((fn_name, "R7 match on " + ast.unparse(node.subject)))
            if isinstance(node, ast.comprehension):                           # R8
                for cond in node.ifs:
                    tested = container_isinstance(cond)
                    if (tested is not None and suspect(tested)
                            and any(ast.dump(tested) == ast.dump(n)
                                    for n in ast.walk(node.iter))):
                        raw.append((fn_name, "R8 comprehension filter " + ast.unparse(cond)))

        for name in sorted(set(lookup_bound) & container_default):            # R5
            raw.append((fn_name, f"R5 {name!r} is bound from a payload lookup and "
                                 f"also to an empty container"))
        shape_tests = [container_isinstance(n.test)
                       for n in nodes if isinstance(n, (ast.If, ast.IfExp))]
        shape_tests = [t for t in shape_tests if t is not None and suspect(t)]
        for name in sorted(set(lookup_bound) & none_default):
            # EXACT match, never a substring of the dump: a scalar read out of a
            # container inside a container-shape branch is correct code, and
            # reporting it would train a maintainer to switch the guard off.
            if any(ast.dump(t) == ast.dump(v)
                   for t in shape_tests for v in lookup_bound[name]):
                raw.append((fn_name, f"R5 {name!r} is shape-tested and defaulted to None"))
        empty_consts = module_empties
    return sites, raw


def test_no_payload_container_is_coerced_outside_the_recording_helpers():
    """THE load-bearing assertion. `_field_list`/`_field_dict` disclose by
    construction, so "every _field_* call discloses" is close to a tautology.
    What actually makes the choke point a choke point is that nothing coerces a
    payload container any other way: the earlier attempts at this fix each
    enumerated the sites they could see and left the rest silent, and a single
    `x.get("k") or []` -- or `_as_list(vt["slots"])`, which is how a live blocker
    hid from the first version of this check -- puts the defect straight back."""
    sites, raw = _coercion_sites()
    assert sites, "AST walk found no coercion sites at all -- the guard is blind"
    assert not raw, (
        f"{len(raw)} container coercion(s) bypass the recording helpers, so a "
        f"malformed value there renders as empty with no disclosure: {raw[:6]}")


def _container_key_decisions():
    """Every place the module DECIDES something about a container field, split
    into the ones that delegate to the choke point and the ones that re-derive
    the answer from the raw payload.

    A container key is one the module itself reads through `_field_list` /
    `_field_dict` somewhere. Two returns: `membership` is a raw `"k" in mapping`
    test on such a key, and `before_read` is a branch that tests a container key
    raw EARLIER in the same function than that key's recorded read."""
    import ast
    import inspect

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    RECORDERS = ("_field_list", "_field_dict")
    HELPERS = RECORDERS + ("_field_present", "_field_declared")
    container_keys = {
        a.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in RECORDERS
        for a in n.args[1:]
        if isinstance(a, ast.Constant) and isinstance(a.value, str)}

    def raw_key(node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)):
            return node.args[0].value
        if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            return node.slice.value
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
                and isinstance(node.left, ast.Constant)):
            return node.left.value
        return None

    def answers_presence(node):
        """Is this expression an ANSWER to "is this field there?" -- as opposed to
        a value, or a classification of a value?

        `bool(<lookup>)`, `<lookup> is None`, `<lookup> is not None` and
        `"<key>" in <mapping>` all are. `kind in ("ctor", "dtor")` is NOT: the
        right-hand side is a literal set of values, so it classifies a value
        rather than testing a mapping for a field."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return answers_presence(node.operand)
        if isinstance(node, ast.BoolOp):
            return any(answers_presence(v) for v in node.values)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "bool" and node.args):
            return raw_key(node.args[0]) is not None or answers_presence(node.args[0])
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op, right = node.ops[0], node.comparators[0]
            if (isinstance(op, (ast.In, ast.NotIn))
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and not isinstance(right, (ast.Tuple, ast.List, ast.Set))):
                return True
            if (isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant)
                    and right.value is None):
                return raw_key(node.left) is not None
        return False

    membership, before_read, delegated = [], [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in HELPERS:
            continue
        # A second definition moved ONE FUNCTION AWAY. The two rules below are
        # both intra-function and lexical, so a renderer that gates its recorded
        # read on `_has_callees(value)` -- a helper that returns a raw presence
        # answer -- passed both while being exactly the drift they exist to
        # stop. The question, not the call site, is what may not be duplicated.
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None \
                    and answers_presence(node.value):
                delegated.append((fn.name, ast.unparse(node)))
        first_read: dict[str, int] = {}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in RECORDERS):
                for a in node.args[1:]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        first_read[a.value] = min(first_read.get(a.value, 1 << 30),
                                                  node.lineno)
        for node in ast.walk(fn):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and isinstance(node.left, ast.Constant)
                    and node.left.value in container_keys):
                membership.append((fn.name, ast.unparse(node)))
            tests = ([node.test] if isinstance(node, (ast.If, ast.IfExp))
                     else node.ifs if isinstance(node, ast.comprehension) else [])
            for test in tests:
                for sub in ast.walk(test):
                    key = raw_key(sub)
                    if key in first_read and getattr(sub, "lineno", 1 << 30) < first_read[key]:
                        before_read.append((fn.name, ast.unparse(sub)))
    assert container_keys, "no container key found -- the scan is blind"
    return membership, before_read, delegated


def test_exactly_the_choke_point_decides_what_present_means():
    """The count-the-deciders property, and the one this PR has failed FIVE
    rounds running -- every time by growing a SECOND place that answers a
    question the choke point already answers, which then drifts from it.

    Two questions exist and they are different: `_field_present` (did the payload
    CLAIM anything -- an explicit null did not) and `_field_declared` (does the
    envelope carry the key at all -- null counts, which is how you tell a paged
    envelope from a bare list). Both live in the choke point. A renderer that
    spells either one itself is a second definition, and it is a finding even
    while it still agrees: round 6's blocker was `"items" in secs` agreeing with
    nothing about an explicit null, written in the same commit as the helper.

    The second half is ordering: a renderer may branch on a container key only
    AFTER that key's recorded read, because a branch that returns or skips first
    means the skew was never recorded and a malformed value renders as a
    confident result. That is precisely how the class-listing count-only
    envelope stayed silent on a falsy wrong-shaped listing.

    The third half -- and the reason the first two were not enough -- is that
    both of those rules are intra-function and lexical, so the SAME second
    definition simply moved one function away and passed: a helper
    `def _has_callees(value): return bool(_as_dict(value).get("callees"))`, used
    to gate the recorded read, is a second definition of PRESENT with a name on
    it. What may not be duplicated is the QUESTION, not the call site, so no
    function outside the choke point may RETURN a presence answer at all."""
    membership, before_read, delegated = _container_key_decisions()
    assert not membership, (
        f"{len(membership)} raw `key in mapping` test(s) on a container key -- a "
        f"second definition of PRESENT that will drift from the helpers' one; "
        f"call _field_declared or _field_present instead: {membership[:6]}")
    assert not before_read, (
        f"{len(before_read)} branch(es) decide on a container key BEFORE its "
        f"recorded read, so a malformed value there never reaches the "
        f"disclosure: {before_read[:6]}")
    assert not delegated, (
        f"{len(delegated)} function(s) outside the choke point RETURN a presence "
        f"answer about a payload field, which is a second definition of PRESENT "
        f"with a name on it -- the intra-function rules above cannot see it "
        f"because it is one hop away; return _field_present/_field_declared's "
        f"answer instead of re-deriving it: {delegated[:6]}")


# Every spelling the guard sees, one probe module each, enumerated as DATA so the
# claim is ASSERTED rather than described. Rounds 5 and 6 each evaded the guard
# with a shape its prose had called covered; the table is the answer to that.
_GUARD_CATCHES = {
    "R2-bound-method-alias": 'g = value.get\n    rows = g("k") or []\n    return str(rows)',
    "R2-function-local-empty-constant": ('_EMPTY = []\n    rows = value.get("k") or _EMPTY\n'
                                         '    return str(rows)'),
    "R4-len-guarded-ternary": ('n = len(value.get("k")) if isinstance(value.get("k"), list)'
                               ' else 0\n    return str(n)'),
    "R6-skip-form-return": ('rows = value.get("k")\n'
                            '    if not isinstance(rows, list):\n        return ""\n'
                            '    return str(rows)'),
    "R7-match-statement": ('match value.get("k"):\n        case []:\n            rows = []\n'
                           '        case _:\n            rows = []\n    return str(rows)'),
    "R8-comprehension-shape-filter": ('rows = [r for r in value.get("k")'
                                      ' if isinstance(value.get("k"), list)]\n'
                                      '    return str(rows)'),
    "R1-coercer-on-get": 'return str(_as_dict(value.get("k")))',
    "R1-coercer-on-subscript": 'return str(_as_dict(value["k"]))',
    "R1-coercer-variable-key": 'return str(_as_dict(value.get(key)))',
    "R1-coercer-one-alias-hop": 'raw = value.get("k")\n    return str(_as_dict(raw))',
    "R1-coercer-list-name-still-guarded": 'return str(_as_list(value.get("k")))',
    "R2-or-empty-list": 'rows = value.get("k") or []\n    return str(rows)',
    "R2-or-empty-dict": 'rows = value.get("k") or {}\n    return str(rows)',
    "R2-or-empty-tuple": 'rows = value.get("k") or ()\n    return str(rows)',
    "R2-or-empty-set-call": 'rows = value.get("k") or set()\n    return str(rows)',
    "R2-or-empty-subscript": 'rows = value["k"] or []\n    return str(rows)',
    "R2-and-or-idiom": ('rows = isinstance(value.get("k"), list) and value.get("k") or []\n'
                        '    return str(rows)'),
    "R3-defaulted-get": 'return str(value.get("k", []))',
    "R3-defaulted-get-dict": 'return str(value.get("k", {}))',
    "R3-defaulted-get-call": 'return str(value.get("k", list()))',
    "R3-defaulted-pop": 'return str(value.pop("k", []))',
    "R3-defaulted-setdefault": 'return str(value.setdefault("k", []))',
    "R4-ternary-else-empty": ('rows = value.get("k") if isinstance(value, dict) else []\n'
                              '    return str(rows)'),
    "R4-ternary-else-none": ('rows = value.get("k") if isinstance(value.get("k"), list)'
                             ' else None\n    return str(rows)'),
    "R4-ternary-else-none-alias": ('raw = value.get("k")\n'
                                   '    rows = raw if isinstance(raw, dict) else None\n'
                                   '    return str(rows)'),
    "R4-ternary-inverted": ('rows = [] if not isinstance(value.get("k"), list)'
                            ' else value.get("k")\n    return str(rows)'),
    "R4-ternary-comprehension-body": ('rows = [r for r in value.get("k")]'
                                      ' if isinstance(value.get("k"), list) else []\n'
                                      '    return str(rows)'),
    "R5-if-else-empty": ('if isinstance(value.get("k"), list):\n'
                         '        rows = value.get("k")\n'
                         '    else:\n        rows = []\n    return str(rows)'),
    "R5-if-else-none": ('if isinstance(value.get("k"), dict):\n'
                        '        rows = value.get("k")\n'
                        '    else:\n        rows = None\n    return str(rows)'),
    "R5-negated-rebind": ('rows = value.get("k")\n'
                          '    if not isinstance(rows, list):\n        rows = []\n'
                          '    return str(rows)'),
    "R5-defensive-copy-branch": ('if isinstance(value.get("k"), list):\n'
                                 '        rows = list(value.get("k"))\n'
                                 '    else:\n        rows = []\n    return str(rows)'),
    "R5-pre-initialise-then-assign": ('rows = []\n'
                                      '    if isinstance(value.get("k"), list):\n'
                                      '        rows = value.get("k")\n    return str(rows)'),
    "R5-pre-initialise-none": ('rows = None\n'
                               '    if isinstance(value.get("k"), list):\n'
                               '        rows = value.get("k")\n    return str(rows)'),
    "R5-annotated-rebind": ('rows = value.get("k")\n'
                            '    if not isinstance(rows, list):\n        rows: list = []\n'
                            '    return str(rows)'),
    "R5-try-except": ('try:\n        rows = list(value.get("k"))\n'
                      '    except TypeError:\n        rows = []\n    return str(rows)'),
    "R5-walrus": ('if not isinstance(rows := value.get("k"), list):\n        rows = []\n'
                  '    return str(rows)'),
    "R5-tuple-unpack": ('rows, n = value.get("k"), 1\n'
                        '    if not isinstance(rows, list):\n        rows, n = [], 1\n'
                        '    return str(rows)'),
    "R5-module-level-constant-default": None,       # needs its own module text
    "scope-inside-a-class": None,
    "scope-async-def": None,
    "scope-module-level": None,
}

_GUARD_CATCH_MODULES = {
    "R5-module-level-constant-default":
        '_EMPTY = []\n\n\ndef _render_probe(value):\n'
        '    rows = value.get("k") or _EMPTY\n    return str(rows)\n',
    "scope-inside-a-class":
        'class Renderer:\n    def render(self, value):\n'
        '        rows = value.get("k") or []\n        return str(rows)\n',
    "scope-async-def":
        'async def _render_probe(value):\n    rows = value.get("k") or []\n'
        '    return str(rows)\n',
    "scope-module-level":
        '_PAYLOAD = {}\n_ROWS = _PAYLOAD.get("k") or []\n',
}

# The mirror: shapes that must NOT be reported, or the guard degenerates into
# "this module contains an isinstance" and stops discriminating at all.
_GUARD_IGNORES = {
    "recorded-lookup": 'return str(_field_list(value, "k"))',
    "recorded-dict-lookup": 'return str(_field_dict(value, "k"))',
    "source-shape-guard": ('name = value.get("k") if isinstance(value, dict) else None\n'
                           '    return str(name)'),
    "element-shape-check": ('out = []\n    for s in _field_list(value, "k"):\n'
                            '        if not isinstance(s, dict):\n            continue\n'
                            '        out.append(s)\n    return str(out)'),
    "presence-test": ('rows = _field_list(value, "k")\n'
                      '    if rows or _field_present(value, "k"):\n'
                      '        return str(len(rows))\n    return ""'),
    "index-into-a-list": 'rows = _field_list(value, "k")\n    return str(rows[-1:])',
    "accumulator-inside-a-shape-branch": ('if isinstance(value.get("k"), list):\n'
                                          '        lines = []\n'
                                          '        lines.append("x")\n'
                                          '        return str(lines)\n'
                                          '    return ""'),
    "scalar-defaulted-to-none-in-a-shape-branch":
        ('addr = None\n    holder = _field_dict(value, "k")\n'
         '    if isinstance(value.get("k"), dict):\n        addr = holder.get("a")\n'
         '    return str(addr)'),
}

# The blind spots, asserted rather than listed in prose. Every round of this PR
# that DESCRIBED the guard's limits described them wrongly -- either claiming a
# shape was covered when it was not, or calling the list complete when it was
# not. A shape here is one the guard provably cannot see; if a future change
# happens to cover one, this test fails and the claim gets updated with it.
_GUARD_BLIND = {
    "two-alias-hops": ('a = value.get("k")\n    b = a\n    rows = b or []\n'
                       '    return str(rows)'),
    "lookup-across-a-function-boundary": ('rows = _grab(value) or []\n    return str(rows)'),
    "deferred-call": ('import functools\n'
                      '    f = functools.partial(value.get, "k")\n'
                      '    rows = f() or []\n    return str(rows)'),
}


@pytest.mark.parametrize("name,body", sorted(_GUARD_CATCHES.items()))
def test_the_coercion_guard_sees_every_form_it_claims_to_see(name, body):
    text = (_GUARD_CATCH_MODULES[name] if body is None
            else f"def _render_probe(value, key='k'):\n    {body}\n")
    _, raw = _coercion_sites(text)
    assert raw, f"the guard is blind to the {name} spelling of a container coercion"


@pytest.mark.parametrize("name,body", sorted(_GUARD_IGNORES.items()))
def test_the_coercion_guard_does_not_fire_on_a_recorded_read(name, body):
    _, raw = _coercion_sites(f"def _render_probe(value, key='k'):\n    {body}\n")
    assert not raw, f"the guard mis-reports {name} as a bypass: {raw}"


@pytest.mark.parametrize("name,body", sorted(_GUARD_BLIND.items()))
def test_the_coercion_guards_blind_spots_are_the_ones_it_declares(name, body):
    """The guard is a structural proxy, not a proof. Pinning what it CANNOT see
    keeps its docstring honest -- an under-stated limit is what let two live
    bypasses sit behind a "the class is closed" claim for two rounds."""
    _, raw = _coercion_sites(f"def _render_probe(value, key='k'):\n    {body}\n")
    assert not raw, (
        f"the guard now SEES {name}; that is good news, but it is declared as a "
        f"blind spot -- move it to _GUARD_CATCHES so the claim matches: {raw}")


# Malformed payloads for a field whose well-formed shape is a list / a dict.
# Includes the FALSY wrong shapes: `0`, `""`, `False` and the opposite empty
# container are PRESENT values of the wrong type, not absent ones, and reading
# them as absent is the same confident-empty answer this change exists to stop.
_MALFORMED = {
    "list": ("bad", {"a": 1}, 0, "", False, {}),
    "dict": ("bad", ["bad"], 0, "", False, []),
}


def _probe_renderers():
    """Every renderer in the module, crossed with every combination of its
    boolean flags. Discovered by INSPECTING THE MODULE, not by asking the
    coercion guard which functions it found a site in.

    The arity rule is ONE REQUIRED positional parameter -- the payload. An
    earlier version required exactly one positional parameter FULL STOP, which
    silently dropped six renderers that take a defaulted flag beside their
    payload (`full`, `verbose`, `demangle`, `limit`, `prefer_caller_static`,
    `inner_renderer`), five of them live top-level CLI text renderers. Their
    top-level reads were then mis-filed as unreachable-nested. A signature
    filter that quietly removes a renderer from the population is the same
    defect as a population taken from the guard: it reports success over the
    part it never examined.

    The flags are probed at BOTH values rather than only their defaults,
    because a flag gates whole blocks of reads (`--verbose` locals, `--full`
    SSA paths) that the default call never reaches.

    Exactly one function is excluded, by name and with its reason:
    `_render_paged_list_text` takes its page key and its item renderer as
    REQUIRED parameters, so the field it reads is an argument rather than a
    property of the module, and every caller reaches it through a renderer that
    is itself probed.

    A renderer that returns lines instead of a string is rendered the way its
    caller renders it, and one that is not itself a `@_discloses` boundary is
    wrapped in one -- in production its caller's boundary is what appends the
    note, so probing it without a boundary would report every helper as silent.
    Wrapping is not a shortcut past the property: the boundary drains what the
    CHOKE POINT recorded, so a coercion that bypasses the choke point still
    produces no note and still fails below."""
    import itertools

    from bn import formatters

    out = []
    for name in sorted(dir(formatters)):
        if not name.startswith("_render"):
            continue
        fn = getattr(formatters, name)
        if not callable(fn):
            continue
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):                    # pragma: no cover
            continue
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len([p for p in positional if p.default is p.empty]) != 1:
            continue                       # not a payload renderer; see docstring
        flags = [p.name for p in params
                 if p.default is not p.empty and isinstance(p.default, bool)]
        for values in itertools.product((False, True), repeat=len(flags)):
            kwargs = dict(zip(flags, values))
            label = name + ("" if not kwargs else
                            "(" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")")

            def call(payload, _fn=fn, _kwargs=kwargs):
                rendered = _fn(payload, **_kwargs)
                if isinstance(rendered, str):
                    return rendered
                if isinstance(rendered, (list, tuple)):
                    return "\n".join(str(line) for line in rendered)
                return str(rendered)

            # An already-decorated renderer appends its own note from inside
            # `call`; an undecorated helper needs the boundary its caller
            # supplies in production.
            out.append((label, call if hasattr(fn, "__wrapped__")
                        else formatters._discloses(call)))
    return out


class _KeyProbe(dict):
    """A payload that records which top-level keys the renderer ASKS FOR.

    This is the population's origin, and the reason it is not a restatement of
    the guard: the probe sees the read itself, so a renderer that coerces a
    container WITHOUT the choke point is still in the population -- and, having
    recorded nothing to disclose, fails. Deriving the population from the
    guard's site list instead dropped exactly that renderer OUT of it: a guard
    whose population comes from the thing it guards cannot fail."""

    def __init__(self, data, seen):
        super().__init__(data)
        self.seen = seen

    def _note(self, key):
        if isinstance(key, str):
            self.seen.add(key)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._note(key)
        return super().__contains__(key)

    def get(self, key, *default):
        self._note(key)
        return super().get(key, *default)

    def pop(self, key, *default):
        self._note(key)
        return super().pop(key, *default)

    def setdefault(self, key, *default):
        self._note(key)
        return super().setdefault(key, *default)


# Serialization is the OPPOSITE of consumption: `json.dumps` walks a container
# to show it verbatim, which hides nothing and so needs no disclosure. Counting
# the encoder's walk as a container read reported 19 string-typed fields as
# containers, so the flag suppresses hits raised underneath a dumps().
_SERIALIZING: list[int] = []


class _QuietJson:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def dumps(self, *args, **kwargs):
        _SERIALIZING.append(1)
        try:
            return self._real.dumps(*args, **kwargs)
        finally:
            _SERIALIZING.pop()


class _WatchedList(list):
    """A list that records being USED as a container. `__bool__` is defined so a
    truthiness test is distinguishable from `len()` -- without it, `if x:` on a
    list falls through to `__len__` and every scalar field tested for truth read
    as a container."""

    def __init__(self, items=()):
        super().__init__(items)
        self.hits: set[str] = set()

    def _hit(self, name):
        if not _SERIALIZING:
            self.hits.add(name)

    def __bool__(self):
        self._hit("bool")
        return list.__len__(self) > 0

    def __len__(self):
        self._hit("len")
        return list.__len__(self)

    def __iter__(self):
        self._hit("iter")
        return list.__iter__(self)

    def __getitem__(self, index):
        self._hit("item")
        return list.__getitem__(self, index)

    def __contains__(self, value):
        self._hit("contains")
        return list.__contains__(self, value)


class _WatchedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hits: set[str] = set()

    def _hit(self, name):
        if not _SERIALIZING:
            self.hits.add(name)

    def __bool__(self):
        self._hit("bool")
        return dict.__len__(self) > 0

    def __len__(self):
        self._hit("len")
        return dict.__len__(self)

    def __iter__(self):
        self._hit("iter")
        return dict.__iter__(self)

    def __getitem__(self, key):
        self._hit("item")
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._hit("contains")
        return dict.__contains__(self, key)

    def get(self, key, *default):
        self._hit("item")
        return dict.get(self, key, *default)

    def keys(self):
        self._hit("keys")
        return dict.keys(self)

    def items(self):
        self._hit("items")
        return dict.items(self)

    def values(self):
        self._hit("values")
        return dict.values(self)


# Truthiness alone is NOT container use -- see _WatchedList.__bool__.
_CONTAINER_USE = frozenset({"len", "iter", "item", "keys", "items", "values", "contains"})
_PROBE_ELEMENT = {"name": "probe", "address": "0x1", "kind": "code", "symbol": "probe",
                  "type": "int", "offset": 0, "count": 1, "op": "probe", "status": "ok"}
# Keyed by the observed kind; `None` (no container use observed) gets a plain
# string, so filling a renderer's OTHER keys does not shove a container into a
# scalar field and send it down a branch it would never take in production.
_PROBE_WELL_FORMED = {"list": [_PROBE_ELEMENT], "dict": dict(_PROBE_ELEMENT), None: "probe"}


def _render_or_exception(render, payload):
    try:
        return render(payload)
    except Exception as exc:                   # noqa: BLE001 - the sweep's subject
        return exc


@functools.lru_cache(maxsize=1)
def _comparison_constants():
    """Per function, every constant the module COMPARES a value against.

    Test inputs, not a population. A read behind `if rec.get("confidence") ==
    "rtti":` only happens when that exact string is in the payload, so the probe
    needs the string -- and the module itself is where the string lives, which
    means a NEW value-gated branch brings its own opener with it instead of
    waiting for someone to notice and hand-add a filler. `==`, `!=`, `is`,
    `in` and `not in` all count, and a tuple/list/set on either side is
    unpacked, so `in ("ctor", "dtor")` yields both."""
    import ast
    import inspect

    from bn import formatters

    out: dict[str, tuple] = {}
    for fn in ast.walk(ast.parse(inspect.getsource(formatters))):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        found: list = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            for side in (node.left, *node.comparators):
                parts = (side.elts if isinstance(side, (ast.Tuple, ast.List, ast.Set))
                         else [side])
                for part in parts:
                    if (isinstance(part, ast.Constant)
                            and isinstance(part.value, (str, int, bool))
                            and part.value not in found):
                        found.append(part.value)
        out[fn.name] = tuple(found)
    return out


def _comparison_literals(fn_name):
    return _comparison_constants().get(fn_name, ())


@functools.lru_cache(maxsize=1)
def _runtime_population():
    """THE differential's population: every `(renderer, key)` the module reads at
    RUNTIME, the container kind observed at that read, and the payload context
    the read happens in.

    Four things it deliberately does not do:

    * It does not ask `_coercion_sites()` anything. The AST guard has declared
      blind spots (`_GUARD_BLIND`), and a population taken from it cannot fail
      on a coercion it cannot see -- the bypass leaves the population instead of
      failing in it. Round 5 deleted an 89-row table for this; the round-7
      population repeated it one level up. Derived here by RUNNING renderers, it
      found 16 live bypasses the guard is blind to, and widening the discovery
      itself at round 8 found 8 more in renderers the probe had been skipping.
    * It does not infer the container kind from the source. A key is a container
      position only if a probe container placed there was actually WALKED --
      iterated, indexed, len'd, or asked for keys/items/values.
    * It does not assume a key is readable in an empty payload. The context is
      the one the read was OBSERVED in, so a key behind a mutually exclusive
      branch (a resolved `function` hides the `context` fallback) is probed in
      the payload that reaches it rather than being silently skipped.
    * It does not assume a branch opens for a container. A read behind
      `if rec.get("confidence") == "rtti":` needs that exact STRING to be
      present, so the fillers include every constant the module compares
      against, harvested per function from the module's own AST
      (`_comparison_literals`). That harvest supplies test INPUTS, not the
      population -- the population is still only what a renderer was observed
      reading -- and it grows by itself: a new value-gated branch brings its own
      opener with it.

    What it still cannot open, stated rather than implied: a branch gated on a
    value that appears nowhere as a literal in the module (a computed threshold,
    a value copied out of another field, a length test)."""
    from bn import formatters

    real_json = formatters.json
    formatters.json = _QuietJson(real_json)
    try:
        population = []
        for name, render in _probe_renderers():
            literals = _comparison_literals(name.split("(")[0])
            seen: set[str] = set()
            for _ in range(6):                        # fixed point: gated branches open
                before = frozenset(seen)
                fillers = [None, "list", "dict"]
                for filler in fillers:
                    ctx = {k: copy.deepcopy(_PROBE_WELL_FORMED[filler]) for k in sorted(seen)}
                    _render_or_exception(render, _KeyProbe(ctx, seen))
                # Value-gated branches: one pass per harvested constant, that
                # constant in every slot, so a read behind an equality test on
                # it is reached and enters the population.
                for literal in literals:
                    ctx = {k: literal for k in sorted(seen)}
                    _render_or_exception(render, _KeyProbe(ctx, seen))
                for filler in fillers:
                    ctx = {k: copy.deepcopy(_PROBE_WELL_FORMED[filler]) for k in sorted(seen)}
                    _render_or_exception(render, _KeyProbe(ctx, seen))
                if frozenset(seen) == before:
                    break
            keys = sorted(seen)

            # Contexts are tried least-perturbing first: the BARE payload, then
            # every other key filled at the kind that key was itself classified
            # as, then cruder fills. Order matters twice over -- a retained alias
            # (#651) is only read when the canonical key is ABSENT, so probing it
            # in a filled context classified it off a payload where the canonical
            # key was malformed, and that context then made the MIRROR fire on
            # data it had built wrong.
            kinds = {}
            records = {}
            for _round in (0, 1):
                records = {}
                for key in keys:
                    contexts = [{},
                                {k: copy.deepcopy(_PROBE_WELL_FORMED[kinds.get(k)])
                                 for k in keys if k != key},
                                {k: copy.deepcopy(_PROBE_WELL_FORMED["list"])
                                 for k in keys if k != key},
                                {k: copy.deepcopy(_PROBE_WELL_FORMED["dict"])
                                 for k in keys if k != key},
                                {k: "probe" for k in keys if k != key},
                                *({k: literal for k in keys if k != key}
                                  for literal in literals)]
                    observed = None
                    for ctx in contexts:
                        for kind in ("list", "dict"):
                            probe = _watched(kind)
                            _render_or_exception(render, {**copy.deepcopy(ctx), key: probe})
                            if probe.hits & _CONTAINER_USE:
                                observed = (ctx, kind)
                                break
                        if observed:
                            break
                    records[key] = observed if observed else (contexts[0], None)
                kinds = {key: rec[1] for key, rec in records.items()}
            for key in keys:
                ctx, kind = records[key]
                population.append((name, render, key, kind, ctx))
        return population
    finally:
        formatters.json = real_json


def _watched(kind):
    return (_WatchedList([dict(_PROBE_ELEMENT)]) if kind == "list"
            else _WatchedDict(_PROBE_ELEMENT))


def test_the_runtime_population_is_exactly_this_big():
    """The LOAD-BEARING half of the differential below, and the half every
    earlier round left out.

    Without an exact size, a site that VANISHES from the population is
    indistinguishable from a site that passed -- which is how a `>= 520` floor
    over 552 cases tolerated five renderers quietly leaving the population. A
    floor cannot tell a fix from a disappearance. These numbers are therefore
    exact, and a deliberate change to the module updates them in the same
    commit; that update is visible in review, a shrinking floor is not."""
    population = _runtime_population()
    probed = len(_probe_renderers())
    reading = {name for name, _, _, _, _ in population}
    containers = [rec for rec in population if rec[3] is not None]
    assert (probed, len(reading), len(population), len(containers)) == (98, 84, 507, 190), (
        "the runtime-discovered population changed size: "
        f"{probed} renderers probed / {len(reading)} of them read a named field / "
        f"{len(population)} (renderer, key) pairs / {len(containers)} of those "
        "pairs read as a container. If you ADDED a renderer or a field, update "
        "these four numbers. If you did not, a renderer stopped reading a field "
        "it used to read, and the differential below just stopped covering it -- "
        "which is the failure this assertion exists to make visible.")


def test_the_container_probe_misses_exactly_three_top_level_reads():
    """What the runtime probe CANNOT classify, named rather than left as a
    number. The round-7 differential said "consumed under-detects at 18 of 92
    sites" and stopped there, which is an unexplained hole; this is the same
    question answered.

    The module's own `_field_list`/`_field_dict` literal arguments are an
    INDEPENDENT inventory of the choke-point reads -- independent because the
    probe never consults it, and it is used here only to measure the probe's
    coverage, never to build the population (building the population from it is
    the defect this whole rework removed).

    Two gaps, both structural and both stated exactly:

    * Three top-level reads are not CLASSIFIED, and all three for one reason:
      the container's contents reach the output only by INTERPOLATION --
      `f"(tainted arg(s) {args})"`, `f"... {others}"` -- never by a walk.
      `__repr__` does not go through `__iter__`, and truthiness is deliberately
      not counted either, because `bool()` cannot tell a list from a scalar and
      counting it reported 936 scalar fields as containers. They are
      `_render_defuse_text.other_versions`, `_render_leaf_line.dropped_args`
      and `_render_leaf_line.tainted_args`. All three are still swept for
      raises, and all three still DISCLOSE a skew, because they read through the
      choke point; they are only outside the disclosure differential.
    * The remaining declared reads sit where a TOP-LEVEL key probe cannot reach
      them: a key of a callee ROW, a per-block `insns`, a flow's `leaves`, or a
      read inside a helper that is handed a nested object rather than the
      renderer's payload. Those are covered by the named nested tests above.

    Both counts are exact, so a read that silently leaves top-level coverage
    fails here instead of quietly shrinking the differential."""
    import ast
    import inspect

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    declared = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("_field_list", "_field_dict")):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        declared.add((fn.name, arg.value))

    population = _runtime_population()
    # A population label carries the flag combination it was probed under
    # (`_render_taint_text(full=True)`); the module's AST knows only the
    # function name, so compare on that.
    def fn_of(label):
        return label.split("(")[0]
    classified = {(fn_of(name), key) for name, _, key, kind, _ in population
                  if kind is not None}
    read = {(fn_of(name), key) for name, _, key, _, _ in population}
    missed = sorted(f"{name}.{key}" for name, key in declared - classified
                    if (name, key) in read)
    nested = [pair for pair in declared - classified if pair not in read]
    assert len(declared) == 202, f"the module declares {len(declared)} choke-point reads, not 202"
    assert missed == ["_render_defuse_text.other_versions",
                      "_render_leaf_line.dropped_args",
                      "_render_leaf_line.tainted_args"], (
        "a choke-point read the probe reaches at top level is no longer "
        f"classified as a container, so the differential stopped covering it: {missed}")
    assert len(nested) == 72, (
        f"{len(nested)} declared reads sit where the top-level probe cannot "
        "reach them, not 72")


def test_a_present_container_is_never_absorbed_into_the_empty_rendering():
    """THE positive differential, over a population discovered by RUNNING the
    renderers (see `_runtime_population`) rather than by asking the coercion
    guard which sites it found.

    The property is the defect, stated directly: a container field that is
    PRESENT but holds the wrong shape must never render byte-identically to that
    field being ABSENT or EMPTY. That is the whole harm of #619 -- the caller
    reads a confident "nothing here" out of a payload the renderer could not
    use, and cannot tell. Disclosing satisfies it; so does rendering the value
    visibly; absorbing it silently does not.

    Stated this way it needs no expected-output table and no "renders identically
    to absent is acceptable" arm -- that arm was the round-7 escape hatch, and it
    is now the failure condition.

    What it does NOT catch, named rather than implied: a renderer that answers a
    malformed container with some OTHER confident value (a count fabricated from
    a string's characters) renders differently from empty and passes here. That
    direction is pinned by name -- see the `sample 3 of 3`, class-count and
    ambiguous-count tests -- because it needs a per-site expectation this
    property cannot derive.

    Measured by replaying THIS population against the base module (same keys,
    same contexts, same `_MALFORMED` values; base's renderers are called bare
    because base has no disclosure boundary to wrap them in): base absorbs 876
    of these 1140 cases at 187 of the 190 container positions, and raises in
    100 more; this commit absorbs 0 and raises 0."""
    from bn import formatters

    echoes = formatters._render_fallback_text              # the raw-payload dump
    absorbed, visible, checked = [], set(), 0
    for fn_name, render, key, kind, ctx in _runtime_population():
        if kind is None:
            continue
        absent = _render_or_exception(render, copy.deepcopy(ctx))
        empty = _render_or_exception(
            render, {**copy.deepcopy(ctx), key: [] if kind == "list" else {}})
        for bogus in _MALFORMED[kind]:
            payload = {**copy.deepcopy(ctx), key: bogus}
            out = _render_or_exception(render, payload)
            checked += 1
            if isinstance(out, Exception):
                continue                   # the raise sweep owns this case
            if "malformed" in out:
                continue
            # The raw dump hides nothing, so there is nothing to disclose.
            if out == echoes(payload) or echoes(payload) in out:
                continue
            if out == absent or out == empty:
                absorbed.append(
                    f"{fn_name}({key}={bogus!r}) renders byte-identically to that "
                    f"field being {'absent' if out == absent else 'empty'}, with no "
                    f"disclosure -- an unusable payload reading as a confident result")
            else:
                visible.add(f"{fn_name}.{key}")
    assert not absorbed, absorbed[:8]
    # The residue: PRESENT, rendered visibly, not disclosed. Legitimate only for
    # a union-typed field, where a scalar is a real shape and not a skew. Pinned
    # by name so a container field cannot join them quietly.
    assert sorted(visible) == ["_render_class_show_text.size", "_render_one_class.size"], (
        "a field renders a wrong-shaped container visibly but without disclosing; "
        f"that is only correct for a scalar-or-envelope union: {sorted(visible)}")
    # Last, so a real absorption reports itself rather than being masked by the
    # anti-vacuity count it also changes.
    assert checked == 1140, f"the differential ran {checked} cases, not 1140"


def test_no_renderer_raises_on_a_field_the_absent_payload_survived():
    """The soft-degrade half of #619, kind-free, so it covers all 507 read keys
    rather than the 190 the container probe classifies as containers: a renderer
    that renders an absent field cleanly and DIES on a present wrong-shaped one
    has regressed to the crash this change replaced.

    Over these same 4056 renders base raises 153 times across 58 (renderer, key)
    positions; this commit raises 0. Two of those renderers
    (`_render_function_info_text`, `_render_taint_text`) are only in the
    population at all because round 8 fixed the arity rule to admit a renderer
    that takes a defaulted flag beside its payload -- which is the argument for
    deriving the population instead of listing it."""
    swept, raised = 0, []
    for fn_name, render, key, _kind, ctx in _runtime_population():
        absent = _render_or_exception(render, copy.deepcopy(ctx))
        for bogus in ("bad", {"a": 1}, ["bad"], 0, "", False, {}, []):
            swept += 1
            out = _render_or_exception(render, {**copy.deepcopy(ctx), key: bogus})
            if isinstance(out, Exception) and not isinstance(absent, Exception):
                raised.append(f"{fn_name}({key}={bogus!r}) raised "
                              f"{type(out).__name__} where the absent payload "
                              f"rendered cleanly")
    assert swept == 4056, f"the raise sweep ran {swept} renders, not 4056"
    assert not raised, raised[:8]


def test_the_malformed_disclosure_never_fires_on_a_well_formed_payload():
    """The mirror property, and the more dangerous direction: crying "malformed"
    at a genuinely empty result would teach a caller to ignore the signal, which
    destroys it while appearing to fix it.

    Over the same runtime population, but on payloads that are benign BY
    CONSTRUCTION rather than by the probe's guess: the key absent, the key an
    explicit null, and -- only where a container kind was actually observed --
    the empty container of that kind. Two deliberate exclusions, both because
    the probe would be asserting its own shape guess rather than the renderer's
    behaviour:

    * A field the container probe could not classify gets no substituted value.
      Filling it with a string guesses its type, and a guessed type on a list
      field is a real skew, so the mirror fired on a payload it built wrong.
    * A POPULATED container is excluded because there is no single well-formed
      dict for every dict field -- `sinks_by_class` is a dict of LISTS, and a
      dict of scalars is a genuine skew one level down. Populated well-formed
      payloads are covered by name instead, in
      `test_well_formed_realistic_payloads_carry_no_disclosure`."""
    noisy, checked = [], 0
    for fn_name, render, key, kind, _ctx in _runtime_population():
        benign: list[object] = [None]
        if kind is not None:
            benign.append([] if kind == "list" else {})
        for payload in [{}] + [{key: value} for value in benign]:
            out = _render_or_exception(render, payload)
            if isinstance(out, Exception):
                continue          # the renderer's own shape requirement
            checked += 1
            if "malformed" in out:
                noisy.append(f"{fn_name}({key}) on {payload!r}")
    assert checked == 1204, f"the mirror ran {checked} renders, not 1204"
    assert not noisy, f"disclosure fired on well-formed data: {noisy}"


def test_well_formed_realistic_payloads_carry_no_disclosure():
    # The mirror again, on data shaped like reality rather than one key at a
    # time: an empty listing, a 0-import target, a zero-class result and a taint
    # run with nothing to report are all CORRECT empty answers.
    from bn import formatters
    clean = [
        formatters._render_imports_summary_text(
            {"total_symbols": 0, "needed_libraries": [], "namespaces": {}, "by_kind": {}}),
        formatters._render_class_list_text({"items": [], "total": 0}),
        formatters._render_local_list_text(
            {"function": {"name": "f", "address": "0x1"}, "items": []}),
        formatters._render_taint_text(
            {"function": {"name": "f", "address": "0x1"}, "direction": "forward",
             "sources": [{"address": "0x2"}], "reached_sinks": [], "leaves": [],
             "stats": {"functions_visited": 3}}),
        formatters._render_surface_text(
            {"summary": {"init_sections": 0, "candidate_tables": 0,
                         "missing_function_candidates": 0}}),
        formatters._render_cfg_text(
            {"function": {"name": "f", "address": "0x1"}, "view": "mlil",
             "blocks": [{"start": "0x1", "insns": [{"a": "0x1", "t": "nop"}], "edges": []}]}),
        formatters._render_class_show_text(
            {"name": "Widget", "bases": [], "methods": [], "confidence": "rtti"}),
    ]
    assert not [out for out in clean if "malformed" in out]


def test_a_truncated_taint_run_with_a_skewed_stats_field_says_so():
    # The worst case in this class: `stats` is coerced inside the verdict helper
    # the renderer hands its WHOLE payload to, so a declaration on the renderer
    # never saw it. A truncated run whose stats arrive skewed silently loses the
    # "truncated @depth" clause and the visited-function count, and an
    # INCOMPLETE analysis then reads byte-identically to a complete one.
    from bn.formatters import _render_taint_text
    payload = {"function": {"name": "f", "address": "0x1"}, "direction": "forward",
               "sources": [{"address": "0x2"}], "reached_sinks": [], "leaves": []}
    skewed = _render_taint_text({**payload, "stats": "bad"})
    assert "malformed stats field" in skewed
    assert skewed != _render_taint_text(payload)


def test_class_show_skewed_primary_vtable_slots_say_more_than_the_empty_case():
    # This one was introduced BY this PR: converting `for s in vt["slots"]` to a
    # bare coercion turned base's loud failure into a class card that printed a
    # vtable address and then nothing at all -- strictly LESS than the
    # genuinely-empty case, which at least explains why no slots resolved. A
    # subscript lookup also hid it from the first version of the coercion guard.
    from bn.formatters import _render_class_show_text
    base = {"name": "Widget", "confidence": "rtti", "vtable": {"address": "0x10"}}
    skewed = _render_class_show_text({**base, "vtable": {"address": "0x10", "slots": "bad"}})
    empty = _render_class_show_text({**base, "vtable": {"address": "0x10", "slots": []}})
    assert "no slots resolved here" in skewed        # the empty case's explanation
    assert "malformed slots field" in skewed         # plus what the empty case cannot say
    assert all(line in skewed for line in empty.splitlines())
    assert skewed != empty


def test_class_show_discloses_a_skewed_method_list_on_both_paths():
    # `methods`/`bases` are coerced in the single-class helper, which the
    # renderer calls with its own payload on the non-ambiguous path and with a
    # match ROW on the ambiguous one. A class whose method list arrives skewed
    # must not render as a class that simply has no methods.
    from bn.formatters import _render_class_show_text
    direct = _render_class_show_text(
        {"name": "Widget", "confidence": "rtti", "methods": "bad", "bases": "bad"})
    assert "class Widget" in direct
    assert "malformed bases, methods fields" in direct
    assert direct != _render_class_show_text(
        {"name": "Widget", "confidence": "rtti", "methods": [], "bases": []})

    ambiguous = _render_class_show_text(
        {"ambiguous": True, "query": "W",
         "matches": [{"name": "Widget", "confidence": "rtti", "methods": "bad"}]})
    assert "malformed methods field" in ambiguous


def test_class_show_ambiguous_count_is_the_rows_rendered_not_the_raw_field():
    # A skewed `matches` counted one match per CHARACTER of the string: invented
    # data in the one line a reader uses to judge how ambiguous the query was.
    from bn.formatters import _render_class_show_text
    out = _render_class_show_text({"ambiguous": True, "query": "W", "matches": "bad"})
    assert "0 matches" in out
    assert "3 matches" not in out
    assert "malformed matches field" in out


def test_class_show_discloses_a_skewed_secondary_vtable_slot_list():
    from bn.formatters import _render_class_show_text
    out = _render_class_show_text(
        {"name": "Widget", "confidence": "rtti",
         "secondary_vtables": [{"address": "0x10", "slots": "bad"}]})
    assert "malformed slots field" in out


def test_cfg_discloses_skewed_per_block_containers():
    # Nested one level below the renderer's own payload: per-block `insns` and
    # `edges`. Base raised here; a coerced [] renders a block that looks empty.
    from bn.formatters import _render_cfg_text
    out = _render_cfg_text({"function": {"name": "f", "address": "0x1"},
                            "view": "mlil",
                            "blocks": [{"start": "0x1", "insns": "bad", "edges": "bad"}]})
    assert "block 0x1" in out
    assert "malformed edges, insns fields" in out


def test_callgraph_discloses_a_skewed_per_callee_resolution_list():
    from bn.formatters import _render_callgraph_text
    out = _render_callgraph_text({
        "function": {"name": "f", "address": "0x1"},
        "callees": [{"call_addr": "0x5", "direct": False, "dest_expr": "rax",
                     "resolved": "bad"}]})
    assert "UNRESOLVED" in out          # an unusable list is not "no targets"
    assert "malformed resolved field" in out


def test_structured_il_discloses_skewed_per_instruction_var_lists():
    from bn.formatters import _render_structured_il_text
    out = _render_structured_il_text({
        "function": {"name": "f", "address": "0x1"}, "view": "mlil",
        "instructions": [{"il_index": 0, "address": "0x1", "op": "MLIL_SET_VAR",
                          "text": "x = 1", "vars_read": "bad", "vars_written": "bad"}]})
    assert "MLIL_SET_VAR" in out
    assert "malformed vars_read, vars_written fields" in out


def test_class_list_discloses_a_skewed_per_row_base_list():
    from bn.formatters import _render_class_list_text
    out = _render_class_list_text({"items": [{"name": "Widget", "bases": "bad"}], "total": 1})
    assert "Widget" in out
    assert "malformed bases field" in out


def test_orient_discloses_a_skewed_sample_attribution_list():
    # A round-1 major fixed the RAISE here by coercing the list; coercing it
    # silently traded the crash for a silent drop, which is the trade this whole
    # change exists to refuse.
    from bn.formatters import _render_orient_text
    out = _render_orient_text({"strings_sample": {"items": [], "sample_sections": "bad"}})
    assert "malformed sample_sections field" in out


def test_xref_renderers_disclose_skewed_ref_containers():
    from bn.formatters import _render_xrefs_text
    out = _render_xrefs_text({"symbol": "gets", "code_refs": "bad", "data_refs": "bad"})
    assert "malformed code_refs, data_refs fields" in out


def test_mutation_renderer_discloses_a_skewed_affected_types_list():
    from bn.formatters import _render_mutation_text
    out = _render_mutation_text({"success": True, "committed": True,
                                 "results": [{"op": "types_declare", "status": "verified"}],
                                 "affected_types": "bad"})
    assert "malformed affected_types field" in out


def test_taint_sink_entry_discloses_a_skewed_callsite_list():
    from bn.formatters import _render_taint_models_text
    out = _render_taint_models_text({"sinks_by_class": {"unbounded_input": [
        {"symbol": "gets", "callsites": "bad"}]}})
    assert "gets" in out
    assert "malformed callsites field" in out


def test_render_instance_find_non_dict_item_degrades():
    from bn.formatters import _render_instance_find_text
    out = _render_instance_find_text({"query": "q", "items": [
        {"selector": "s", "instance_id": "i", "binary": "b"}, "bad"]})
    assert "s  (instance i)" in out
    assert "'bad'" in out


def test_render_taint_models_class_rows_are_counted_by_what_was_examined():
    # Two directions, both of them a mis-count of an inventory an auditor reads
    # as ground truth. Wrapping a malformed entry list as a one-element list
    # counted it as one modeled sink (a fabricated row); filtering the class out
    # when its entry list was falsy then DROPPED a class that was examined from
    # "in N class(es)" (a claim that fewer classes were looked at than were).
    # The class row is what "examined" means, the entries are what "found"
    # means, and the two counts are independent.
    from bn.formatters import _render_taint_models_text

    # PRESENT AND EMPTY: a real result -- we looked at this class and found no
    # modeled sink. Still listed, still counted, and not called malformed.
    empty = _render_taint_models_text(
        {"sinks_by_class": {"exec": [{"symbol": "system"}], "empty_cls": []}})
    assert "sinks (1 in 2 class(es))" in empty
    assert "[empty_cls]" in empty
    assert "malformed" not in empty

    # ABSENT (explicit null): nothing was claimed for the class, so there is no
    # skew to disclose -- but the class key is still there and still counted.
    nulled = _render_taint_models_text({"sinks_by_class": {"unbounded_input": None}})
    assert "sinks (0 in 1 class(es))" in nulled
    assert "[unbounded_input]" in nulled
    assert "malformed" not in nulled

    # PRESENT BUT WRONG SHAPE: zero sink rows (never one per character), the
    # class still listed and counted, and the skew disclosed by name.
    skewed = _render_taint_models_text({"sinks_by_class": {"unbounded_input": "gets"}})
    assert "sinks (0 in 1 class(es))" in skewed
    assert "[unbounded_input]" in skewed
    assert "malformed unbounded_input field" in skewed

    # And the FALSY wrong shape, which read as absent until presence replaced
    # truthiness inside the recording helpers.
    falsy = _render_taint_models_text({"sinks_by_class": {"unbounded_input": {}}})
    assert "sinks (0 in 1 class(es))" in falsy
    assert "malformed unbounded_input field" in falsy


def test_a_falsy_wrong_shaped_container_is_disclosed_not_read_as_absent():
    # `0`, `""`, `False` and a `{}` where a list belongs are PRESENT values of
    # the wrong type. Testing the raw value for truth instead of presence read
    # every one of them as absent and rendered a confident empty listing from a
    # payload the renderer could not use (#619).
    from bn.formatters import (_render_class_list_text, _render_callsites_text,
                               _render_orient_text)
    for bogus in ({}, 0, "", False):
        out = _render_class_list_text({"items": bogus, "total": 1})
        assert "malformed items field" in out, bogus
        out = _render_callsites_text({"items": bogus, "total": 1})
        assert "malformed items field" in out, bogus
    # Nested one level down, and for a dict-shaped field.
    assert "malformed imports_summary field" in _render_orient_text({"imports_summary": []})


def test_class_show_discloses_a_skewed_vtable_container():
    # The vtable container itself, not its slots: a malformed one rendered a
    # class card byte-identical to a class that genuinely has no vtable, inside
    # the very cluster #619 names.
    from bn.formatters import _render_class_show_text
    rec = {"name": "Widget", "confidence": "rtti"}
    absent = _render_class_show_text(rec)
    skewed = _render_class_show_text({**rec, "vtable": "bad"})
    assert "malformed vtable field" in skewed
    assert skewed.startswith(absent)                 # strict superset of absent
    # An explicit null still means "no vtable", and must stay quiet.
    assert "malformed" not in _render_class_show_text({**rec, "vtable": None})


def test_xref_grouping_discloses_a_skewed_caller_function():
    # Coerced through a ternary the choke-point guard could not see, so a
    # malformed caller_function silently grouped the ref as if it had no
    # containing function at all.
    from bn.formatters import _render_xrefs_text
    out = _render_xrefs_text({"symbol": "gets", "code_refs": [
        {"address": "0x1", "caller_function": "main"}]})
    assert "malformed caller_function field" in out


def test_orient_skewed_section_listing_is_a_superset_of_the_empty_one():
    # The same superset invariant the vtable case established, applied to the
    # nested section listing: a malformed listing must say everything the empty
    # one says and then disclose, never render strictly LESS than empty.
    from bn.formatters import _render_orient_text
    empty = _render_orient_text({"sections": {"items": []}})
    skewed = _render_orient_text({"sections": {"items": "bad"}})
    absent = _render_orient_text({})
    assert "sections: 0" in empty
    assert "sections: 0" in skewed
    assert "malformed items field" in skewed
    assert skewed.startswith(empty)
    assert "sections" not in absent                  # absent claims nothing


def test_an_explicit_null_listing_claims_nothing_and_renders_no_count_row():
    # The other half of the same row, and the trap the superset fix fell into:
    # the renderer must use the CHOKE POINT's definition of PRESENT. Spelling it
    # `"items" in secs` disagreed with `_field_list` about an explicit null and
    # printed a confident `sections: 0` -- and, with a sibling total, printed
    # that total -- for a payload that had claimed nothing at all.
    from bn.formatters import _render_orient_text
    nulled = _render_orient_text({"sections": {"items": None}})
    assert "sections" not in nulled
    assert "malformed" not in nulled
    assert nulled == _render_orient_text({})
    # A null alongside a total is the loudest version: the total must not be
    # rendered as if a listing had come back.
    assert "sections" not in _render_orient_text({"sections": {"items": None, "total": 7}})


def test_the_class_count_only_envelope_still_discloses_a_falsy_wrong_listing():
    # #484's count-only branch decided on the raw containers' truth and returned
    # BEFORE the recorded read ran, so a falsy wrong-shaped listing rendered the
    # count line byte-identically to the listing simply being absent. Reading
    # through the choke point first leaves the branch alone and still discloses.
    from bn.formatters import _render_class_list_text
    bare = _render_class_list_text({"count": 3})
    assert bare == "classes: 3"
    for bogus in ({}, 0, "", False):
        out = _render_class_list_text({"count": 3, "items": bogus})
        assert out.startswith("classes: 3"), bogus     # superset of the bare row
        assert "malformed items field" in out, bogus
    # A genuinely empty listing is a real count-only envelope and stays quiet.
    assert _render_class_list_text({"count": 3, "items": []}) == bare


def test_class_show_renders_a_declared_but_unnamed_base_instead_of_dropping_it():
    # An element filter on the base list turned "has an unnamed base" into "has
    # no such base" in a hierarchy view -- a silent drop, in the direction that
    # understates what the payload declared.
    from bn.formatters import _render_class_show_text
    rec = {"name": "Widget", "confidence": "rtti"}
    assert "base: ?, A" in _render_class_show_text({**rec, "bases": [{}, {"name": "A"}]})
    assert "base: ?" in _render_class_show_text({**rec, "bases": [{}]})
    assert "base: ?" in _render_class_show_text({**rec, "bases": [None]})
    # A genuinely empty base list still means "no bases", so no clause at all.
    assert "base:" not in _render_class_show_text({**rec, "bases": []})


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
    measured_payload = {
        "success": True, "committed": True, "preview": False, "rolled_back": False,
        "results": [{"status": "verified"}]}
    go_payload = {
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "rolled_back": False, "results": [], "go_renamed_candidates": 3,
        "go_committed_count": 3, "go_verified_count": 3, "go_failed_count": 0,
        "skipped_user_named": 1}

    measured = formatters._mutation_summary(dict(measured_payload))
    go = formatters._go_rename_summary(dict(go_payload))
    assert measured["changed_count"] == 1 and go["changed_count"] == 3
    assert set(measured) == set(go)

    # One builder produced both, proven by CONSUMER-OBSERVABLE output: a change
    # made in the builder has to appear in both summaries. A call count would
    # only say each path called something; this says neither has a literal of
    # its own that a builder change would leave behind.
    monkeypatch.setattr(formatters, "_build_mutation_summary",
                        lambda **kw: {**real_builder(**kw), "reached_via_builder": True})
    assert formatters._mutation_summary(dict(measured_payload))["reached_via_builder"]
    assert formatters._go_rename_summary(dict(go_payload))["reached_via_builder"]
    monkeypatch.setattr(formatters, "_build_mutation_summary", real_builder)

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


def test_the_compact_summary_ok_key_mirrors_success_on_every_outcome():
    """#447: `.ok` exists so one `jq '.ok'` works across reads AND mutations. It
    is only worth having if it TRACKS the outcome -- a hardcoded true is strictly
    worse than the null it replaced, because a failed or rolled-back mutation
    would then report success to the control loop that was told to trust it.

    Enumerated over the outcome space rather than one example: the previous cover
    was a single `assert out["ok"] is True`, which a literal `True` satisfies,
    and the same literal survived the whole mocked suite."""
    import itertools

    from bn import formatters

    seen = set()
    for reported, committed, preview, rolled_back, failures in itertools.product(
            (True, False), (True, False), (True, False), (True, False, None), (0, 1)):
        rows = ([{"status": "failed", "message": "boom"}] if failures
                else [{"status": "verified"}])
        base = {"success": reported, "committed": committed, "preview": preview,
                "rolled_back": rolled_back, "results": rows}
        generic = formatters._mutation_summary(dict(base))
        go = formatters._go_rename_summary({
            **base, "kind": "go_rename", "results": rows,
            "go_renamed_candidates": 2, "go_committed_count": 2,
            "go_verified_count": 2, "go_failed_count": failures,
            "skipped_user_named": 0})
        for name, summary in (("mutation", generic), ("go rename", go)):
            assert summary["ok"] is summary["success"], (
                f"{name} summary reported ok={summary['ok']!r} for "
                f"success={summary['success']!r} on {base!r}")
            seen.add(summary["ok"])
    # Without this the mirror is satisfiable by a constant: the population has to
    # actually produce BOTH outcomes for the assertion above to have bitten.
    assert seen == {True, False}, f"the outcome population only produced ok={seen}"


def test_no_renderer_mutates_the_payload_it_was_handed():
    """The recording helpers hand back the payload's OWN list, so a renderer that
    appended to one would silently edit the caller's result -- and the JSON path
    would then emit rows the text path invented. No defensive copy is taken (that
    would allocate on all ~190 reads to defend against nothing), so the no-mutate
    rule is asserted instead.

    Asserted by RUNNING the renderers and comparing the payload against a
    snapshot taken before the render, not by matching mutation syntax in the AST.
    The AST form saw only a `.append` on a name bound directly from a helper, so
    an alias hop, handing the list to a helper, mutating an ELEMENT and a walrus
    binding all evaded it -- a shape list only ever sees the shapes its author
    imagined, which is the same defect as deriving the differential's population
    from the guard that the differential exists to check.

    What a runtime check cannot see, stated plainly: a mutation on a branch these
    payloads do not reach. It covers every (renderer, key) the population
    discovered, in the context that key is read in, which is strictly more of
    the module than any spelling list reached."""
    mutated = []
    for fn_name, render, key, kind, ctx in _runtime_population():
        payload = {**copy.deepcopy(ctx), key: copy.deepcopy(_PROBE_WELL_FORMED[kind])}
        before = copy.deepcopy(payload)
        _render_or_exception(render, payload)
        if payload != before:
            mutated.append(f"{fn_name}({key}) left the payload changed: "
                           f"{before!r} -> {payload!r}")
    assert not mutated, (
        "a renderer mutated the payload it was handed, which is the CALLER's "
        f"object and is what the JSON path emits: {mutated[:6]}")
