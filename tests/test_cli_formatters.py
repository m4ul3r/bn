from __future__ import annotations

import collections
import copy
import functools
import inspect
import json
import re
import types

from decimal import Decimal
from fractions import Fraction

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


def test_render_comment_list_row_escapes_control_chars():
    """#771: the comment cell of a `comment list` row is operator-supplied free
    text (`comment set <addr> $'a\\nb'`), so a control char in it used to split
    the row across two lines for every consumer of the text view -- including
    `--format text --out` files. It now goes through the same escaper as the
    symbol-name cell (#370.1). The JSON path is a rendering-independent concern
    and must stay raw."""
    from bn.formatters import _render_comment_list_text
    from bn.output import render_value
    payload = [{"address": "0x1000", "function": "parse_one",
                "comment": "line1\nline2\twith\rctrl\x07char"}]
    out = _render_comment_list_text(payload)
    assert out.splitlines() == [
        "0x1000  parse_one  line1\\nline2\\twith\\rctrl\\x07char"], out
    # --format json is untouched: the raw comment round-trips byte-for-byte.
    assert json.loads(render_value(payload, "json")) == payload


def test_render_comment_function_view_escapes_doc_and_address_rows():
    """#771: `comment get --function` lists the same two stores as `comment
    list` (the function doc + the in-function address comments), so both row
    kinds stay on one line. The standalone single-comment form is the whole
    payload -- a document, not a row -- so its newlines stay content."""
    from bn.formatters import _render_comment_text
    out = _render_comment_text({
        "function": {"name": "parse_one", "address": "0x1000"},
        "has_function_doc": True,
        "function_doc": "doc line1\ndoc line2",
        "comments": [{"address": "0x1010", "comment": "a\nb"}],
    })
    assert out.splitlines() == ["[doc] doc line1\\ndoc line2", "0x1010  a\\nb"], out
    assert _render_comment_text({"comment": "a\nb"}) == "a\nb"


def test_comment_list_text_is_one_line_end_to_end(fake_transport, capsys):
    """#771 at the CLI level: `comment list --format text` must render a comment
    carrying a newline on ONE line (a `--out` file's rows are read line-wise),
    while `--format json` still round-trips the raw comment untouched."""
    fake_transport({"list_comments": {"ok": True, "result": {
        "items": [{"address": "0x401000", "function": "parse_one",
                   "comment": "line1\nline2"}],
        "total": 1, "offset": 0, "limit": 50, "returned": 1, "has_more": False}}})
    rc = bn.cli.main(["comment", "list", "--target", "active", "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["0x401000  parse_one  line1\\nline2"], out

    rc = bn.cli.main(["comment", "list", "--target", "active", "--format", "json"])
    assert rc == 0
    row = json.loads(capsys.readouterr().out)["items"][0]
    assert row["comment"] == "line1\nline2"


def test_render_tag_row_escapes_control_chars():
    """#771: a tag's `data` is operator-supplied text (`tag add --data`); a
    multi-line tag must not split the tag row (`tag get` / `tag list`)."""
    from bn.formatters import _render_tag_list_text, _render_tag_row
    tag = {"address": "0x2000", "scope": "address", "icon": "🎯",
           "type": "Bookmarks", "data": "first\nsecond\x1b[0m"}
    row = _render_tag_row(tag)
    assert row.splitlines() == [row], row
    assert "first\\nsecond\\x1b[0m" in row
    listed = _render_tag_list_text([tag])
    assert listed.splitlines() == [row], listed


def test_render_local_list_escapes_local_name_control_chars():
    """#771: `local rename` can put a newline in a local's name; the `locals:`
    row must stay one physical line instead of becoming two."""
    from bn.formatters import _render_local_list_text
    out = _render_local_list_text({
        "function": {"name": "parse_one", "address": "0x1000"},
        "items": [{"name": "ev\nil", "type": "int", "local_id": "L1"}],
    })
    # header + "" + "locals:" + the one row
    assert len(out.splitlines()) == 4, out
    assert "  ev\\nil" in out
    assert "L1" in out


def test_render_name_address_rows_escapes_library_and_raw_name():
    """#771: :1565 escaped the name cell of a name/address row but left its
    `library` and `raw_name` columns raw -- either one splits the row on its
    own, so they take the same escaper."""
    from bn.formatters import _render_name_address_rows
    out = _render_name_address_rows([
        {"address": "0x3000", "name": "alloc_thing", "kind": "import",
         "library": "lib\nc.so", "raw_name": "raw\rname"},
    ])
    assert out.splitlines() == [
        "0x3000  alloc_thing (import) [lib\\nc.so] (raw: raw\\rname)"], out


def test_render_comment_list_row_escapes_func_cell():
    """#771 round 2: escaping the comment cell alone did not make the row
    one-line-safe -- the same row interpolates the containing function's symbol
    name, which `rename` accepts with a newline (`_require_nonempty_name` only
    rejects empty/whitespace). The row must stay one physical line whichever
    cell carries the control char."""
    from bn.formatters import _render_comment_list_text
    out = _render_comment_list_text([{"address": "0x1000", "function": "fn\nname",
                                      "comment": "plain"}])
    assert out.splitlines() == ["0x1000  fn\\nname  plain"], out


def test_render_tag_row_escapes_loc_type_and_icon_cells():
    """#771 round 2: the tag row's other three settable cells split it just as
    well as `data` did -- `tag type create <name> [--icon]` passes both through
    to the view with no charset check, and the function-scope `loc` is a symbol
    name. Every cell of the row goes through the same escaper."""
    from bn.formatters import _render_tag_row
    row = _render_tag_row({"scope": "function", "function": "fn\nname",
                           "icon": "i\nc", "type": "Book\nmarks", "data": "plain"})
    assert row.splitlines() == [row], row
    assert row == "fn\\nname  [function]  i\\nc Book\\nmarks  plain", row


def test_render_tag_types_text_escapes_name_and_icon():
    """#771 round 2: `tag types` lists the same settable name/icon pair as the
    tag row, so it stays one line per type too."""
    from bn.formatters import _render_tag_types_text
    out = _render_tag_types_text({"tag_types": [{"name": "Book\nmarks",
                                                 "icon": "i\nc"}]})
    assert out.splitlines() == ["i\\nc  Book\\nmarks"], out


def test_render_local_list_header_escapes_function_name():
    """#771 round 2: `local list` prints the function name in its header raw,
    while `function info` escapes the same value in its own header (#370.1) --
    one of the two could still be split by a renamed function."""
    from bn.formatters import _render_local_list_text
    out = _render_local_list_text({
        "function": {"name": "fn\nname", "address": "0x1000"},
        "items": [{"name": "ok", "type": "int", "local_id": "L1"}]})
    assert len(out.splitlines()) == 4, out
    assert out.splitlines()[0].startswith("fn\\nname @ 0x1000"), out


def test_format_local_entry_escapes_type_cell():
    """#771 round 2: the local entry's neighbouring `type` cell is interpolated
    into the same row as the name the PR escaped, so it takes the escaper too --
    one raw cell splits the row just as well."""
    from bn.formatters import _format_local_entry
    line = _format_local_entry({"name": "ok", "type": "in\nt", "local_id": "L1"})
    assert line.splitlines() == [line], line
    assert "in\\nt" in line


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
    # #101: a malformed bridge result that trips a text renderer must SOFT-DEGRADE
    # to placeholders at exit 0; the BridgeError wrapper (exit 2, pointing at
    # --format json) remains the last resort for truly unhandleable shapes, and
    # `test_text_format_error_stays_on_stderr` above is what covers it.
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
    # More pages remain: the SHARED footer, with the real resume offset (#770 --
    # this renderer used to print its own "more available -- raise --limit or use
    # --offset" wording, which named neither the remainder nor the next offset).
    more = {**full, "total": 12, "returned": 5, "limit": 5, "has_more": True}
    out_more = _render_field_xrefs_text(more)
    assert "// showing 5 of 12 (7 more); rerun with --offset 5" in out_more
    # Last page of an --offset run (has_more False but returned != total): still noted,
    # so the skipped refs aren't silently dropped.
    tail = {**full, "total": 12, "offset": 10, "returned": 2, "limit": 5, "has_more": False}
    out_tail = _render_field_xrefs_text(tail)
    assert "// showing 2 of 12" in out_tail and "--offset" not in out_tail
    # A self-contradicting window is REFUSED by name instead of rendered as a
    # partial page (the shared footer's rule, which the bespoke one had no way to
    # state) -- and the render still says so.
    impossible = {**full, "total": 2, "offset": 0, "returned": 9, "has_more": False}
    out_impossible = _render_field_xrefs_text(impossible)
    assert "page position not stated" in out_impossible and "offset + returned exceeds total" in out_impossible


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


def test_render_virtual_call_text_shows_typed_reason_code_822():
    # #822: the prose alone cannot be branched on, so the typed discriminator
    # renders alongside it in parens -- the `hlil: null (reason_code)` shape.
    from bn.formatters import _render_virtual_call_text
    truncated = {
        "callsite": "0x1000", "caller": "consumer", "slot_offset": "0x230",
        "slot_index": 70, "factory": None, "candidates": [], "resolved": False,
        "unresolved_reason": "slot 70 is beyond the recovered vtable window (scan capped at "
                            "64 slots) in at least one provider -- the target method may "
                            "exist past the cap rather than being genuinely unresolvable",
        "unresolved_reason_code": "vtable_scan_truncated",
    }
    out = _render_virtual_call_text(truncated)
    assert "unresolved:" in out
    assert "(vtable_scan_truncated)" in out

    # An absent slot carries the code with the generic hint (no prose), so a
    # reader still sees WHY nothing resolved.
    absent = {
        "callsite": "0x1000", "caller": "consumer", "slot_offset": "0x28",
        "slot_index": 5, "factory": None, "candidates": [], "resolved": False,
        "unresolved_reason_code": "slot_not_present",
    }
    out = _render_virtual_call_text(absent)
    assert "(slot_not_present)" in out
    assert "no provider class implements this slot" in out


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


def test_render_callsites_shows_missing_context_reason():
    # #816: a call whose address the disassembly sweep never decoded keeps its row
    # (identity fields intact) with a null context -- text mode must say WHY the
    # context is unavailable instead of printing a bare "<unknown>".
    from bn.formatters import _render_callsites_text
    value = {"items": [{
        "callee": {"name": "target_fn", "address": "0x461746"},
        "containing_function": {"name": "caller_fn", "address": "0x412470"},
        "call_addr": "0x4124a6", "caller_static": "0x4124ab",
        "call_instruction": None, "disasm_context_reason": "no_structured_disasm_entry",
        "previous_instructions": [], "next_instructions": [],
    }], "total": 1, "has_more": False}
    out = _render_callsites_text(value)
    assert "call 0x4124a6 | caller_static 0x4124ab" in out
    assert "> unavailable (no_structured_disasm_entry)" in out
    assert "<unknown>" not in out


def test_render_callsites_flags_an_incomplete_caller_scan():
    # #816: a partial caller ENUMERATION makes the rows a lower bound, not the
    # whole set -- text mode must name the reason, or a short caller list reads
    # as the complete answer.
    from bn.formatters import _render_callsites_text
    note = (
        "call scan stopped at its budget (256 functions / 4096 LLIL instructions "
        "examined); the caller list may be incomplete"
    )
    value = {
        "items": [{
            "callee": {"name": "target_fn", "address": "0x461746"},
            "containing_function": {"name": "caller_fn", "address": "0x412470"},
            "call_addr": "0x4124a6", "caller_static": "0x4124ab",
            "call_instruction": {"address": "0x4124a6", "text": "call target_fn"},
            "previous_instructions": [], "next_instructions": [],
            "disasm_context_reason": None,
        }],
        "total": None, "total_lower_bound": 1, "has_more": False,
        "scan_truncated": False, "caller_scan_truncated": True,
        "callers_scanned": 1, "caller_total": None, "caller_scan_note": note,
    }
    out = _render_callsites_text(value)
    assert "call 0x4124a6 | caller_static 0x4124ab" in out
    assert "note: the caller scan was incomplete" in out
    assert note in out


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


# The choke-point helpers, and the bare coercion helpers that BYPASS them.
# Hoisted out of `_coercion_sites` so `test_the_coercion_guard_names_no_deleted
# _helper` can check them against the live module: the list carried `_as_list`
# for four rounds after this PR deleted it, and a guard row naming a helper that
# does not exist tests a spelling nobody can write. Harmless is not the same as
# checked.
_RECORDERS = ("_field_list", "_field_dict", "_row_list")
_COERCERS = ("_as_dict",)


def test_the_coercion_guard_names_no_deleted_helper():
    """Every helper name the AST guard matches on must still exist.

    A stale name is a silently dead guard row, and this file's whole subject is
    that a justification nobody re-checks does not expire when it stops being
    true."""
    from bn import formatters

    missing = [name for name in _RECORDERS + _COERCERS
               if not hasattr(formatters, name)]
    assert not missing, (
        f"the coercion guard matches on {missing}, which the module no longer "
        "defines -- those rows can never fire again")


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
    RECORDERS = _RECORDERS
    COERCERS = _COERCERS
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
                    kind = "dict" if node.func.id == "_field_dict" else "list"
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
    RECORDERS = _RECORDERS
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

# The ELEMENT axis of the same idea: a list that arrives with the wrong CONTENTS
# rather than as the wrong container. Shared by the top-level and the nested
# element sweeps, so the two cannot drift into covering different shapes -- two
# sweeps of the same property with two junk sets is one of them silently
# narrower than the other.
_ELEMENT_JUNK = (None, 0, 1, True, "s", [], {}, [{"a": 1}], {"a": 1})


# Excluded from the probe BY NAME, each with the reason it is not a payload TEXT
# renderer. An exclusion that is not named here does not exist: a silent one is
# how five live renderers left the population at round 8 and two more at round 9.
#
# Each entry is `(category, prose)`, and the CATEGORY IS EXECUTABLE: the test
# below re-derives it from the live module and fails when it stops holding. That
# is round 11's second lesson. `_go_rename_summary` sat here under "the summary
# it builds is rendered by a renderer that is itself probed" -- a sentence that
# is false for EVERY transform, because a transform runs before any renderer
# exists -- and prose cannot go stale loudly. An exemption whose justification
# is only prose is an exemption nobody re-checks.
_EXCLUSION_CATEGORIES = ("not-callable", "takes-no-payload",
                         "takes-more-than-a-payload",
                         "is-a-renderer-factory", "returns-data-not-text")
_PROBE_EXCLUSIONS = {
    "_render_paged_list_text": (
        "takes-more-than-a-payload",
        "takes its page key and its item renderer as REQUIRED parameters, so the "
        "field it reads is an argument rather than a property of the module, and "
        "every caller reaches it through a renderer that is itself probed"),
    "_slice_text_lines": (
        "takes-more-than-a-payload",
        "its first argument is ALREADY-RENDERED TEXT plus a required line range, "
        "not a payload; it reads no payload key at all"),
    "_text_field": (
        "is-a-renderer-factory",
        "its argument is the KEY, not a payload. The renderer it returns reads "
        "that one key and returns it only when it is already a STRING, falling "
        "back to the raw dump otherwise -- no container walk, so there is nothing "
        "for either differential"),
    "_xref_buckets": (
        "returns-data-not-text",
        "returns DATA (the split ref buckets), not text, so there is no rendering "
        "to absorb; the CLI counts groups with it for a pipe note and hands the "
        "SAME RAW payload to the body renderer, which discloses the skew. The "
        "pipe-note consumer additionally opens its own boundary via "
        "`formatters.disclosure_boundary` and states the unreadable case itself, "
        "so neither consumer of this payload answers a confident count off a "
        "bucket it could not read. That is what makes this legitimate where a "
        "transform's is not"),
    "_group_refs_by_caller": (
        "returns-data-not-text",
        "returns DATA (grouped rows), not text -- same pipe-note path as the "
        "bucket splitter: the body renderer likewise still sees the raw payload, "
        "and the pipe note opens `formatters.disclosure_boundary` so a skewed "
        "bucket reports as unreadable rather than as zero hidden groups"),
    "FAILED_MUTATION_STATUSES": (
        "not-callable",
        "a set of status STRINGS the CLI compares an op row against, not a "
        "payload consumer -- there is no render to absorb. Named because "
        "`_probe_renderers` skipping every non-callable SILENTLY is the defect "
        "this list exists to stop"),
    "disclosure_boundary": (
        "takes-no-payload",
        "takes NO argument at all: it is `@_discloses` opened as a context "
        "manager for a consumer that is not a renderer (the xrefs pipe note, "
        "which answers a note-or-None on stderr rather than returning a body). "
        "It reads no payload key, so there is nothing for a payload differential "
        "to probe -- what it does is make the skew its caller's choke-point "
        "reads record reach a boundary, which is the property the pipe-note test "
        "asserts behaviourally"),
    # The CHOKE POINT itself, now referenced from a command module: the
    # `strings --count` line reads its two numbers through the same helpers
    # `_render_strings_text` uses one surface over, so the two `strings`
    # surfaces cannot answer "how many did the filter drop" differently (#795
    # round-2 review). Excluded for the reason the list exists to record: these
    # are what the differential MEASURES, not something it can measure.
    "_count_field": (
        "takes-more-than-a-payload",
        "the field it reads is an ARGUMENT, not a property of the module: it "
        "takes the payload AND the key, returns an int rather than a rendering, "
        "and is the choke point every probed renderer's disclosure is derived "
        "from -- a differential over it would be measuring the oracle"),
    "_stated_count": (
        "takes-more-than-a-payload",
        "`_count_field` for a line that STATES the number, so same shape and "
        "same reason: payload plus key in, a count-or-`?` string out, with the "
        "skew recorded for the ENCLOSING boundary to disclose"),
    "_nonnegative_count": (
        "takes-more-than-a-payload",
        "`_count_field` for a key whose count is a CARDINALITY, so same shape "
        "and same reason as its two siblings: payload plus key in, an int out, "
        "with the skew recorded for the ENCLOSING boundary to disclose. It is "
        "referenced from a command module because the `imports --count` line "
        "reads the excluded count through the very helper the paged listing "
        "and the `--summary` card read it through -- one decision, so the "
        "three surfaces cannot hold three opinions about one payload"),
    "_discloses": (
        "takes-no-payload",
        "it IS the boundary, not a consumer of one: a decorator taking the "
        "renderer (or nothing, under `prefix=`), with no required payload "
        "argument at all. Same class as `disclosure_boundary` above -- what it "
        "does is make the skew its wrapped renderer recorded reach a note, "
        "which is the property every probe below asserts behaviourally"),
}


def _exclusion_category_holds(name, category):
    """Re-derive one exclusion's stated category from the LIVE module.

    The whole point: a reason that stops being true fails a test instead of
    sitting in a comment. Each category is a property of the symbol itself, so
    nothing here consults the exclusion list to decide whether the exclusion is
    warranted."""
    from bn import formatters

    obj = getattr(formatters, name, None)
    if category == "not-callable":
        return not callable(obj)
    if not callable(obj):
        return False
    positional = [p for p in inspect.signature(obj).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    if category == "takes-more-than-a-payload":
        return len(required) > 1
    if category == "takes-no-payload":
        return not required
    if len(required) != 1:
        return False           # the two categories below are about the RETURN
    if category == "is-a-renderer-factory":
        return callable(_render_or_exception(obj, "probe"))
    if category == "returns-data-not-text":
        out = _render_or_exception(obj, dict(_PROBE_ELEMENT))
        # BOTH halves of the stated reason, because the second was the
        # load-bearing one and was still prose: returning data is what makes
        # there be no rendering to absorb, and the CLI handing the SAME RAW
        # payload to a probed renderer is what makes the skew this helper
        # records reach a boundary at all. Drop the second and the exclusion
        # becomes the very thing round 11 blocked on -- a reader excused
        # because "something else renders its output" when nothing does.
        return (not isinstance(out, (str, Exception))
                and _raw_payload_also_rendered(name))
    return False


def _raw_payload_also_rendered(name):
    """Does the CLI hand the payload this data helper reads to a PROBED text
    renderer in the same call?

    Re-derived from the CLI's own AST: find the functions that reference the
    helper, then the call that installs one of them as a keyword, and require
    that call to install a `*renderer=` whose formatters symbols are all probed.
    So if the pipe-note path ever stopped sharing its payload with the body
    renderer, this exclusion fails instead of continuing to assert it."""
    import ast
    import pathlib

    from bn import formatters

    module = pathlib.Path(formatters.__file__).resolve()
    probed = {label.split("(")[0] for label, _ in _probe_renderers()}
    for path in sorted(module.parent.rglob("*.py")):
        if path.resolve() == module:
            continue
        tree = ast.parse(path.read_text())
        symbols, modules = _formatters_bindings(tree, _package_modules().get(path.resolve()))
        users = {fn.name for fn in ast.walk(tree)
                 if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and name in _formatters_refs(fn, symbols, modules)}
        if not users:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not any(isinstance(inner, ast.Name) and inner.id in users
                       for kw in node.keywords for inner in ast.walk(kw.value)):
                continue
            rendered = {r for kw in node.keywords
                        if kw.arg and kw.arg.endswith("renderer")
                        for r in _formatters_refs(kw.value, symbols, modules)}
            if rendered and rendered <= probed:
                return True
    return False


# A transform the CLI installs is NOT a renderer, and probing it as one measures
# the harness instead of the code: it returns a DICT, so `str()` of its result
# differs between a malformed key and an absent one for free, and a differential
# over that passes while the user still sees nothing. The CLI runs each of these
# as a `result_transform`/`summary_transform` and then renders the
# ALREADY-TRANSFORMED value, so the live entry point is the COMPOSITION -- which
# is exactly where the skew was being dropped, because `@_discloses` installs its
# recorder inside the renderer, long after the transform has consumed and
# discarded the payload.
#
# The renderer each transform is paired with is DECLARED here; that this table is
# COMPLETE is not. `_cli_installed_transforms()` reads the transform names out of
# the CLI's own AST and the inventory test fails if one of them is missing, so a
# new transform cannot enter the codebase unclassified. Naming one in
# `_PROBE_EXCLUSIONS` instead is rejected outright: "the summary it builds is
# rendered by a renderer that is itself probed" was the reason `_go_rename_summary`
# carried, and it is FALSE for a transform -- the renderer is handed the OUTPUT,
# never the payload, so nothing downstream can re-read what the transform
# absorbed. That false premise was round 11's blocker.
_COMPOSED_ENTRY_POINTS = {
    "_mutation_summary": "_render_mutation_summary_text",
    "_go_rename_summary": "_render_mutation_summary_text",
    "_add_mutation_ok": "_render_mutation_text",
}


@functools.lru_cache(maxsize=1)
def _package_modules():
    """Every module of the formatters' own package, IMPORTED, keyed by file.

    The identity sweep needs OBJECTS, not source text: a binding is only
    discoverable by `id()` once the module that holds it exists."""
    import importlib
    import pathlib
    import pkgutil

    from bn import formatters

    package = formatters.__package__
    root = pathlib.Path(formatters.__file__).resolve().parent
    names = [package] + [info.name for info in
                         pkgutil.walk_packages([str(root)], prefix=package + ".")]
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        if getattr(module, "__file__", None):
            modules[pathlib.Path(module.__file__).resolve()] = module
    return modules


@functools.lru_cache(maxsize=1)
def _formatters_identities():
    """`id()` -> canonical name, for every symbol the formatters module OWNS.

    Ownership is `__module__`, not membership of `vars()`: `formatters.json` is
    the json module, and a command module's own `import json` binds the very
    same object, so an identity rule without an ownership filter would report
    `json` as a formatters entry point. Sorted so an in-module alias resolves to
    one canonical name deterministically rather than to whichever binding the
    dict happened to yield last."""
    from bn import formatters

    return {id(obj): name for name, obj in sorted(vars(formatters).items(), reverse=True)
            if callable(obj)
            and getattr(obj, "__module__", None) == formatters.__name__}


@functools.lru_cache(maxsize=1)
def _formatters_identity_inventory():
    """Every formatters symbol a package module BINDS AT MODULE LEVEL, found by
    `id()`.

    THE answer to "is this inventory a spelling rule", and the reason it is not
    widened a sixth time. Rounds 8-13 each closed one more spelling -- a bare
    name, a module attribute, a non-`_render` prefix, a directory glob, a
    keyword other than `*_renderer=` -- and each widening exposed live payload
    consumers the previous one had certified. `id()` does not care how the
    binding was WRITTEN: a symbol re-exported through a shim (`from .shim import
    _render_x`), one renamed on the way in, one bound by a module-level
    `getattr(formatters, "_render_x")`, and one reached through an alias of an
    alias are all the SAME OBJECT in `vars(module)`.

    The live case this found: `bn.cli` re-exports `_format_operation_result` so
    tests and scripts can monkeypatch it, and then never mentions the name
    again. No AST reference walk can see that -- there is no reference -- so a
    payload consumer that renders every mutation's op rows sat in no inventory,
    no population and no exclusion, and its malformed `defined_types` rendered
    byte-identically to the field being absent.

    What `id()` CANNOT see, stated rather than implied, because the previous
    version of this docstring claimed the opposite and a reviewer had to prove
    it: a symbol resolved INSIDE A FUNCTION by a computed name
    (`getattr(formatters, name)`) is bound to no module global, so it is in
    neither half of the inventory. That is not left as a caveat --
    `test_no_consumer_reaches_the_formatters_module_by_a_computed_name` asserts
    the package contains no such spelling, so the gap cannot open quietly. An
    unexecutable promise is exactly what this file spent five rounds deleting."""
    from bn import formatters

    owned = _formatters_identities()
    found: set[str] = set()
    for module in _package_modules().values():
        if module is formatters:
            continue
        for obj in list(vars(module).values()):
            if id(obj) in owned:
                found.add(owned[id(obj)])
    return frozenset(found)


def _formatters_bindings(tree, module=None):
    """Every name in one module that is bound to a formatters SYMBOL, and every
    name bound to the formatters MODULE.

    Identity first, import syntax second. The identity half resolves a local
    name to its CANONICAL formatters name however the binding was spelled (see
    `_formatters_identity_inventory`), including a module alias of an alias --
    `import bn.formatters as f; g = f` leaves `g` the same module object, which
    no name-matching walk can follow. The syntax half is kept for what identity
    cannot attribute: a NON-CALLABLE like `FAILED_MUTATION_STATUSES` carries no
    `__module__`, so ownership of it is only visible in the import statement.
    The union is strictly stronger than either."""
    import ast

    from bn import formatters

    owned = _formatters_identities()
    symbols: dict[str, str] = {}
    modules: set[str] = set()
    if module is not None:
        for local, obj in list(vars(module).items()):
            if id(obj) in owned:
                symbols[local] = owned[id(obj)]
            elif obj is formatters:
                modules.add(local)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "formatters":          # from . import formatters
                    modules.add(alias.asname or alias.name)
                elif (node.module or "").endswith("formatters"):
                    symbols.setdefault(alias.asname or alias.name, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("formatters"):   # import bn.formatters as F
                    modules.add(alias.asname or alias.name.split(".")[0])
    return symbols, modules


def _formatters_refs(node, symbols, modules):
    """Every formatters symbol REFERENCED anywhere under `node`, by any of three
    spellings: a bare name bound from the module, an attribute taken off the
    module itself (`_fmt._x`, `bn.formatters._x`), or a `getattr` on the module
    with a LITERAL name. An attribute is only counted when the live module
    really has it, so `_fmt.json` is not mistaken for an entry point. A bare
    name resolves through `symbols` to its CANONICAL name, so a renamed import
    is not reported under the local spelling.

    The `getattr` spelling is here because the docstring of
    `_formatters_identity_inventory` used to claim `id()` covered it and it does
    not: a module-level `getattr` binding is in `vars(module)` and so IS seen by
    identity, but one inside a function is bound to nothing. A literal name is
    readable from the AST, so it is read here; the computed form is asserted not
    to exist at all, by
    `test_no_consumer_reaches_the_formatters_module_by_a_computed_name`."""
    import ast

    from bn import formatters

    found: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in symbols:
            found.add(symbols[inner.id])
        elif isinstance(inner, ast.Attribute):
            base = inner.value
            if ((isinstance(base, ast.Name) and base.id in modules)
                    or (isinstance(base, ast.Attribute)
                        and base.attr == "formatters")):
                if hasattr(formatters, inner.attr):
                    found.add(inner.attr)
        elif (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id == "getattr" and len(inner.args) >= 2
                and _is_formatters_module(inner.args[0], modules)
                and isinstance(inner.args[1], ast.Constant)
                and isinstance(inner.args[1].value, str)
                and hasattr(formatters, inner.args[1].value)):
            found.add(inner.args[1].value)
    return found


def _is_formatters_module(node, modules):
    """Is this expression the formatters MODULE, by either spelling a package
    module can bind it under?"""
    import ast

    return ((isinstance(node, ast.Name) and node.id in modules)
            or (isinstance(node, ast.Attribute) and node.attr == "formatters"))


def _computed_formatters_getattrs(tree, modules):
    """Line numbers of every `getattr(<the formatters module>, <not a literal>)`
    under `tree`. The detector, separated from the walk of the package so the
    test below can prove it fires on a source that contains one."""
    import ast

    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and _is_formatters_module(node.args[0], modules)
            and not (isinstance(node.args[1], ast.Constant)
                     and isinstance(node.args[1].value, str))]


# Every package module that binds the formatters MODULE OBJECT, as opposed to
# importing symbols out of it. MEASURED, and it is not empty: the package's own
# `__init__` carries it, because importing a submodule sets it as an attribute
# of the package whether anyone wrote that binding or not. Every command module
# spells `from ..formatters import (...)` instead, so the object is in scope
# nowhere else.
#
# Declared because this set is the entire surface on which the inventory's one
# real blind spot can open -- a `getattr(formatters, name)` with a runtime name
# can only be written where that object is in scope. If another module starts
# binding it, the test below fails, the name goes in here, and the
# computed-name rule starts applying to it.
_MODULES_BINDING_THE_FORMATTERS_MODULE = frozenset({"__init__.py"})


def test_no_consumer_reaches_the_formatters_module_by_a_computed_name():
    """The one gap `id()` cannot close, made executable instead of caveated.

    A symbol resolved inside a function by a COMPUTED name -- `getattr(
    formatters, name)` off a table, a prefix, or a subcommand string -- is bound
    to no module global, so the identity inventory cannot see it and the AST
    reference walk cannot read it. It would be a live payload consumer in
    NEITHER half of the inventory, and inventory membership is exactly what
    decides whether a helper is probed on the surface a caller receives.

    Round 14 left this as a docstring caveat and round 15 left it as a minor.
    A caveat does not expire when it stops being true, so it is measured here in
    two halves, because the first half alone would be vacuous today:

    * WHERE it could be written. No package module binds the formatters module
      object at all, so today there is no expression to `getattr` on. That is
      not a rule imposed on the codebase -- it is a measurement, declared, and
      the failure asks you to update the declaration.
    * THAT THE DETECTOR WORKS. Asserted against synthetic sources rather than
      against the package, for the reason every guard in this file is: an
      emptiness claim over a detector nobody exercised is free. This is the same
      discipline as the coercion guard's own probe modules."""
    import ast
    import pathlib

    from bn import formatters

    # Half two first, so a broken detector fails HERE rather than silently
    # certifying the package below.
    caught = _computed_formatters_getattrs(
        *_probe_tree_and_modules('from . import formatters\n'
                                 'def use(name):\n'
                                 '    return formatters.getattr_probe\n'
                                 'def reach(name):\n'
                                 '    return getattr(formatters, name)\n'))
    assert caught == [5], (
        "the computed-name detector no longer fires on a module that binds the "
        f"formatters module and getattrs it by a runtime name: {caught}")
    missed = _computed_formatters_getattrs(
        *_probe_tree_and_modules('import bn.formatters as F\n'
                                 'def fine():\n'
                                 '    return getattr(F, "_render_fallback_text")\n'))
    assert missed == [], (
        "the detector fires on a LITERAL name, which is readable from the AST "
        f"and counted by `_formatters_refs`: {missed}")
    aliased = _computed_formatters_getattrs(
        *_probe_tree_and_modules('import bn.formatters as F\n'
                                 'def reach(name):\n'
                                 '    return getattr(F, name)\n'))
    assert aliased == [3], (
        f"the detector misses the `import ... as` spelling of the module: {aliased}")

    # Half one: the package itself.
    module = pathlib.Path(formatters.__file__).resolve()
    binding, computed = set(), []
    for path in sorted(module.parent.rglob("*.py")):
        if path.resolve() == module:
            continue
        tree = ast.parse(path.read_text())
        _symbols, modules = _formatters_bindings(
            tree, _package_modules().get(path.resolve()))
        if not modules:
            continue
        binding.add(path.name)
        computed += [f"{path.name}:{line}"
                     for line in _computed_formatters_getattrs(tree, modules)]
    assert binding == set(_MODULES_BINDING_THE_FORMATTERS_MODULE), (
        f"{sorted(binding)} binds the formatters MODULE OBJECT and "
        f"{sorted(_MODULES_BINDING_THE_FORMATTERS_MODULE)} is declared. That is "
        "the surface a computed-name lookup can be written on, so put the new "
        "one in _MODULES_BINDING_THE_FORMATTERS_MODULE -- the rule below then "
        "applies to it.")
    assert not computed, (
        f"{computed} resolves a formatters symbol by a name computed at "
        "runtime, which puts a live payload consumer in NEITHER half of the "
        "inventory: `id()` sees only module-level bindings and the AST walk can "
        "only read a literal. Bind it at module level, spell the name as a "
        "literal, or add the resolved names to the inventory explicitly -- do "
        "not leave it to a docstring.")


def _probe_tree_and_modules(source):
    """Parse a synthetic consumer module and resolve which of its names hold the
    formatters MODULE, using the same binding walk the package walk uses."""
    import ast

    tree = ast.parse(source)
    _symbols, modules = _formatters_bindings(tree)
    return tree, modules


@functools.lru_cache(maxsize=1)
def _cli_installed_transforms():
    """Every `bn.formatters` symbol the CLI installs as a RESULT TRANSFORM, read
    out of the CLI's own AST.

    A transform's output REPLACES the payload the text renderer is handed, which
    makes it the one entry-point kind whose choke-point reads can reach no
    `@_discloses` boundary at all. Derived rather than listed for the usual
    reason: a table of transforms maintained beside the transforms is a
    population taken from the thing it guards."""
    import ast
    import pathlib

    from bn import formatters

    module = pathlib.Path(formatters.__file__).resolve()
    names: set[str] = set()
    for path in sorted(module.parent.rglob("*.py")):
        if path.resolve() == module:
            continue
        tree = ast.parse(path.read_text())
        symbols, modules = _formatters_bindings(tree, _package_modules().get(path.resolve()))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                # `spill_status` is the transform; `spill_status_renderer` beside
                # it is a RENDERER and is probed as one, so match the transform
                # keyword exactly rather than by prefix.
                if not kw.arg or ("transform" not in kw.arg
                                  and kw.arg != "spill_status"):
                    continue
                names |= _formatters_refs(kw.value, symbols, modules)
    return frozenset(names)


@functools.lru_cache(maxsize=1)
def _cli_invoked_formatters():
    """Every `bn.formatters` symbol the package MENTIONS, read out of its AST.

    An inventory of this module's LIVE ENTRY POINTS that is independent of how
    the module spells its function names and of which functions carry the
    `@_discloses` decorator. Both of those are properties of the thing under
    guard, and both have already dropped live renderers out of the population:
    round 8's arity rule dropped six, and the `_render*` NAME prefix beside it
    dropped `_resolution_note` and `_disasm_linear_steer_note` -- two renderers
    the CLI concatenates ahead of six subcommands' output -- whose top-level
    reads were then certified as part of the unreachable-nested bucket.

    Any REFERENCE counts, not just a `*_renderer=` keyword. Scoping the walk to
    that keyword was round 10's minor: a symbol installed under a differently
    named keyword, or called directly inside a pipe-note helper, sat outside the
    inventory -- and an inventory with a spelling rule of its own is the same
    defect one level up.

    The walk covers the WHOLE package, not the command modules. Scoping it to
    `commands/*.py` was round 11's blocker and the same defect a third time: the
    CLI ENTRY module installs the mutation transforms directly, so three live
    payload consumers were in neither population and in no named exclusion, and
    a coercion added to any of them kept every guard in this file green. A
    directory glob is a spelling rule like any other.

    Every symbol in this inventory is probed BARE (see `_probe_renderers`).
    Splitting it into "invoked" and "merely bound" and wrapping the second class
    in a manufactured boundary was round 14's own defect: `bn.cli` re-exports
    `_format_operation_result` precisely so tests and scripts CAN call it, an
    existing test does, and a direct caller installs no boundary -- so the
    wrapper hid a live absorption the widening had just uncovered. An entry
    point that cannot disclose on its own surface must be FIXED, not wrapped."""
    import ast
    import pathlib

    from bn import formatters

    module = pathlib.Path(formatters.__file__).resolve()
    referenced: set[str] = set()
    for path in sorted(module.parent.rglob("*.py")):
        if path.resolve() == module:
            continue                   # the module under guard is not its own caller
        tree = ast.parse(path.read_text())
        symbols, modules = _formatters_bindings(tree, _package_modules().get(path.resolve()))
        referenced |= _formatters_refs(tree, symbols, modules)
    return frozenset(referenced)


@functools.lru_cache(maxsize=1)
def _cli_referenced_formatters():
    """THE inventory: every formatters symbol the package reaches, however.

    The AST half is only HALF, because every one of the five earlier widenings
    was another spelling. This is the union of what the package MENTIONS and
    what it BINDS BY IDENTITY, so a consumer reachable through a re-export, a
    `getattr`, a rename or an alias of an alias cannot be outside it -- see
    `_formatters_identity_inventory`, which is what finally caught `bn.cli`'s
    re-exported `_format_operation_result`."""
    return _cli_invoked_formatters() | _formatters_identity_inventory()


def test_every_renderer_the_cli_installs_is_in_the_population():
    """The population's entry points come from the CLI, not from this module's
    naming convention.

    `_probe_renderers` scans `dir(formatters)` for a `_render` prefix, which is a
    property of the thing being guarded: a live renderer spelled `_resolution_note`
    was invisible to it, so the differential, the raise sweep and the no-mutation
    invariant could not fail on it, and its top-level reads were mis-filed as
    unreachable-nested. This asserts the two inventories agree, so a renderer the
    CLI installs cannot leave the population by being named differently."""
    from bn import formatters

    installed = _cli_referenced_formatters()
    # Anti-vacuity: a broken walk returning nothing would satisfy the subset
    # check below without examining anything. These names are the ones the
    # spelling rules this test exists to replace would each have missed: a note
    # (no `_render` prefix), a steer (same, plus a REQUIRED keyword-only flag),
    # a text slicer, a bucket splitter the CLI calls directly rather than
    # installing under a `*_renderer=` keyword, two transforms that live in the
    # CLI ENTRY module rather than under `commands/`, and -- the one no AST
    # reference walk can reach at all -- `_format_operation_result`, which
    # `bn.cli` re-exports for monkeypatching and never references by name.
    # Only the identity sweep sees that one, and it absorbed a malformed
    # `defined_types` silently for the whole time it was invisible.
    assert {"_resolution_note", "_disasm_linear_steer_note", "_slice_text_lines",
            "_xref_buckets", "_render_defuse_text", "_mutation_summary",
            "_add_mutation_ok", "_format_operation_result"} <= installed, sorted(installed)
    probed = {label.split("(")[0] for label, _ in _probe_renderers()}
    missing = sorted(installed - probed - set(_PROBE_EXCLUSIONS))
    assert not missing, (
        f"the CLI references {missing} as payload consumer(s) the probe never "
        "runs, so no guard below can fail on them. Probe them, compose them in "
        "_COMPOSED_ENTRY_POINTS, or name each one in _PROBE_EXCLUSIONS with the "
        "reason it is not a payload text renderer.")
    # The identity sweep's residue: names the package BINDS but never mentions
    # again. Round 14 probed them WRAPPED, on the claim that production reaches
    # them only as a fragment under the boundary of the renderer that composes
    # them -- and that claim was FALSE for the one name it applied to, because
    # `bn.cli` re-exports `_format_operation_result` so callers can reach it and
    # one already does. The manufactured boundary then hid the absorption the
    # widening had just uncovered: an exemption that suppresses the finding it
    # was invented for. There is no such class any more, and this asserts it
    # STAYS gone -- every name in the inventory is probed on the surface a
    # caller actually gets.
    bare = {label.split("(")[0] for label, probe in _probe_renderers()
            if getattr(probe, "__wrapped__", None) is None}
    wrapped = {label.split("(")[0] for label, probe in _probe_renderers()
               if getattr(probe, "__wrapped__", None) is not None}
    # THE property, and it is no longer asked of the set that decides it. Round
    # 15 compared `wrapped` against `installed` -- the very set the wrap rule
    # keys off -- so the answer was empty by construction while 19 helpers were
    # still probed under a manufactured boundary. Round 15's repair re-pointed
    # it at `_boundary_free_helpers`, and round 16 showed that is the SAME
    # defect one level along: the wrap rule keys off that walk too, so
    # `wrapped & exposed` is empty by DEFINITION and narrowing the walk to the
    # two names below still passed.
    #
    # The question is therefore put to an independent oracle -- what the module
    # ACTUALLY does at runtime (`_helpers_observed_without_a_recorder`), which
    # no wrap rule consults. A helper the probe wrapped that is nonetheless
    # observed running with no recorder is a manufactured disclosure, and the
    # static walk having missed it is exactly the failure mode.
    exposed = set(_boundary_free_helpers())
    live_bare = set(_helpers_observed_without_a_recorder())
    manufactured = sorted(wrapped & live_bare)
    assert not manufactured, (
        f"{manufactured} is probed under a boundary, and the running module "
        "reaches it from a live entry point with NO recorder installed -- so a "
        "skew it cannot disclose on its own surface reads as disclosed here, "
        "and the static walk that decided to wrap it is wrong. Probe it bare "
        "and fix the renderer.")
    # The two oracles must agree in the direction that matters: anything seen
    # running bare must be something the conservative walk predicted. A witness
    # the walk did not name means the walk under-approximates, which is the only
    # direction that lets a manufactured boundary through.
    unpredicted = sorted(live_bare - exposed)
    assert not unpredicted, (
        f"{unpredicted} runs with no recorder on a live path and the static "
        "boundary-free walk does not name it, so the walk is missing an edge "
        "and the wrap rule above is deciding on bad information.")
    # Anti-vacuity, because both assertions above are emptiness claims, and an
    # emptiness claim over an oracle that witnessed nothing is free. The
    # WITNESSES are pinned exactly -- they are what the module actually did, so
    # a driving loop that stops reaching a live surface goes red here instead of
    # quietly making the two checks above trivially true. Nine of the walk's 17
    # boundary-free names are witnessed; the other eight sit on branches these
    # payloads do not open, which is the stated limit of a live witness and the
    # reason the conservative static walk is kept beside it rather than
    # replaced by it.
    assert sorted(live_bare) == [
        "_as_dict", "_field_dict", "_field_list", "_operation_row",
        "_record_skew", "_render_fallback_text", "_render_target_choice",
        "_skew_note", "_unknown_ref_label"], (
        "the runtime oracle no longer witnesses the same set running bare, so "
        "its emptiness claims above are over different evidence than the ones "
        f"that were checked: {sorted(live_bare)}")
    assert {"_render_target_choice", "_render_fallback_text"} <= bare, (
        "a helper with a boundary-free production path must be probed on that "
        "surface, which is the whole point of the assertion above")
    # A transform's output REPLACES the payload, so no downstream renderer can
    # re-read what it absorbed: it must be COMPOSED, and it may never be waved
    # through as an exclusion. Both halves are derived from the CLI's AST, so a
    # new transform arrives here already unclassified rather than silently
    # uncovered -- round 11's blocker was a transform sitting in the exclusion
    # list under a reason that was false for its whole category.
    transforms = set(_cli_installed_transforms())
    assert transforms == set(_COMPOSED_ENTRY_POINTS), (
        "the CLI installs these result transforms and _COMPOSED_ENTRY_POINTS "
        f"declares those: {sorted(transforms)} != "
        f"{sorted(_COMPOSED_ENTRY_POINTS)}. A transform outside the table is "
        "probed by nothing.")
    assert not transforms & set(_PROBE_EXCLUSIONS), (
        f"{sorted(transforms & set(_PROBE_EXCLUSIONS))} is installed as a result "
        "transform, so the renderer downstream of it never sees the payload it "
        "read. It cannot be excluded on the grounds that something else renders "
        "its output -- compose it.")
    for transform, renderer in _COMPOSED_ENTRY_POINTS.items():
        assert callable(getattr(formatters, transform, None)), transform
        paired = getattr(formatters, renderer, None)
        assert callable(paired), f"{transform} is paired with a missing {renderer}"
        assert hasattr(paired, "__wrapped__"), (
            f"{renderer} renders a transform's output but is no @_discloses "
            "boundary, so the composition has no drain at all")
    # A stale exclusion is a silent one, and prose goes stale silently by
    # definition -- that was round 11's second finding. Every exclusion's stated
    # CATEGORY is therefore re-derived from the live module here, so a
    # justification that stops being true fails instead of sitting in a comment.
    for name, (category, reason) in _PROBE_EXCLUSIONS.items():
        assert hasattr(formatters, name), f"{name} no longer exists"
        assert reason, f"{name} is excluded without a reason"
        assert category in _EXCLUSION_CATEGORIES, (
            f"{name} is excluded under {category!r}, which is not one of the "
            f"checkable categories {_EXCLUSION_CATEGORIES}. An exclusion whose "
            "reason cannot be re-derived is prose, and prose does not expire.")
        assert _exclusion_category_holds(name, category), (
            f"{name} is excluded as {category!r} and the module no longer agrees: "
            f"{reason}. Either it became a payload text renderer and must be "
            "probed, or its category changed and the entry must say which.")


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

    Two candidate sets, unioned: every `_render*` name in the module, and every
    symbol the CLI installs as a text renderer (`_cli_installed_renderers`,
    read out of the command modules rather than out of this one). The name
    prefix alone is a property of the thing under guard, and it dropped
    `_resolution_note` and `_disasm_linear_steer_note` -- both concatenated
    ahead of six live subcommands -- out of the population entirely.

    Exclusions are named with their reason in `_PROBE_EXCLUSIONS` and asserted
    by `test_every_renderer_the_cli_installs_is_in_the_population`; a silent
    exclusion is the defect, not the exclusion itself.

    A renderer that returns lines instead of a string is rendered the way its
    caller renders it, and a NESTED helper that is not itself a `@_discloses`
    boundary is wrapped in one -- in production its caller's boundary is what
    appends the note, so probing it without a boundary would report every helper
    as silent. Wrapping is not a shortcut past the property: the boundary drains
    what the CHOKE POINT recorded, so a coercion that bypasses the choke point
    still produces no note and still fails below.

    A renderer the CLI installs DIRECTLY is never wrapped, because production
    does not wrap it: the command modules call `_resolution_note(value) +
    _render_defuse_text(value)`, so an undecorated one has no boundary at all and
    the skew it records is dropped. Wrapping it here would have manufactured the
    disclosure the user never sees -- the probe must call a live entry point
    exactly as its caller does, or it measures the harness instead of the code."""

    import itertools

    from bn import formatters

    out = []
    candidates = ({name for name in dir(formatters) if name.startswith("_render")}
                  | set(_cli_referenced_formatters()))
    for name in sorted(candidates):
        if (name in _PROBE_EXCLUSIONS or name in _COMPOSED_ENTRY_POINTS
                or not hasattr(formatters, name)):
            continue                   # named with its reason, or composed below
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
        # A REQUIRED keyword-only bool is a flag too: `_disasm_linear_steer_note`
        # gates its whole body on `sliced`, and calling it without one is a
        # TypeError, so a rule that only saw defaulted flags could not run it.
        flags = [p.name for p in params
                 if (p.default is not p.empty and isinstance(p.default, bool))
                 or (p.kind is p.KEYWORD_ONLY and p.default is p.empty
                     and p.annotation in (bool, "bool"))]
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
            # `call`; a nested helper needs the boundary its caller supplies in
            # production. A renderer the package INVOKES has no such caller, so
            # it is probed exactly as bare as production leaves it.
            #
            # INVOKED, not merely in the inventory: a symbol that is only BOUND
            # (`bn.cli` re-exports `_format_operation_result` and never mentions
            # it again) is called by nothing at that spelling, and every
            # production path reaches it as the FRAGMENT BUILDER it is, under the
            # boundary of the renderer that composes its row. Probing such a one
            # bare would report an absorption no caller can see, and wrapping an
            # invoked one would manufacture a disclosure the user never gets --
            # the probe must call a live entry point exactly as its caller does,
            # or it measures the harness instead of the code. Which side a name
            # falls on is asserted in
            # `test_every_renderer_the_cli_installs_is_in_the_population`.
            # Wrapped only when NO production path reaches this helper without
            # crossing a boundary (`_boundary_free_helpers`). Keying it on
            # inventory membership was round 15's residue: it made the assertion
            # that checks it a tautology, and it left two helpers the CLI reaches
            # through undecorated callers probed under a disclosure production
            # never installs.
            out.append((label, call
                        if (hasattr(fn, "__wrapped__")
                            or name in _cli_referenced_formatters()
                            or name in _boundary_free_helpers())
                        else formatters._discloses(call)))
    # The composed entry points, appended LAST and never wrapped: production
    # does not wrap them either -- the paired renderer carries its own boundary,
    # and that boundary is precisely what the transform's reads happen too early
    # to reach.
    for transform_name in sorted(_COMPOSED_ENTRY_POINTS):
        transform = getattr(formatters, transform_name)
        renderer = getattr(formatters, _COMPOSED_ENTRY_POINTS[transform_name])

        def composed(payload, _t=transform, _r=renderer):
            rendered = _r(_t(payload))
            return rendered if isinstance(rendered, str) else str(rendered)

        out.append((f"{transform_name}(via "
                    f"{_COMPOSED_ENTRY_POINTS[transform_name]})", composed))
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
# A second, DIFFERENT well-formed element. Two copies of the same object make an
# adjacent-difference gate (`if rows[i] != rows[i+1]`, a de-duplicating pass, an
# `(xN)` collapse) permanently False, so a read behind one was reachable by no
# payload this file built -- proven by injection at round 16. Every list this
# harness builds at cardinality 3 puts this between two copies of the element
# under test, which also makes index 2 reachable.
_PROBE_SIBLING_ELEMENT = {**_PROBE_ELEMENT, "name": "sibling", "address": "0x2",
                          "kind": "data", "symbol": "sibling", "op": "sibling"}
# Keyed by the observed kind; `None` (no container use observed) gets a plain
# string, so filling a renderer's OTHER keys does not shove a container into a
# scalar field and send it down a branch it would never take in production.
# The list filler carries THREE rows, two of them equal and one different, for
# the cardinality reason `_payload_for` states: a one-element filler cannot open
# a branch gated on a SECOND row (`len(rows) > 1`, a "... and N more" tail, a
# separator), two EQUAL elements cannot open one gated on adjacent rows
# differing, and neither can reach index 2.
_PROBE_WELL_FORMED = {"list": [_PROBE_ELEMENT, _PROBE_SIBLING_ELEMENT, _PROBE_ELEMENT],
                      "dict": dict(_PROBE_ELEMENT), None: "probe"}


def _render_or_exception(render, payload):
    try:
        return render(payload)
    except Exception as exc:                   # noqa: BLE001 - the sweep's subject
        return exc


# Four values that differ in TYPE and in TRUTH, so a position gated on either
# can be told apart. Deliberately all well-formed-ish: the question here is
# whether the read reaches the output at all, not whether it degrades.
_DEPENDENCE_PROBES = ("probe-a", "probe-b", {"a": 1}, 0)


def _value_reaches_the_output(render, build):
    """Does the value at this position reach the RENDER, or is it merely ASKED?

    "The key was asked here" is a weaker property than the sweeps need; "a
    different value here renders differently" is the one they need, and it is a
    measurement rather than a guess.

    An exception counts as a rendering, because a raise is exactly the observable
    the raise sweeps hunt: a context in which a wrong shape can raise is a
    context in which the read is live."""
    seen = set()
    for value in _DEPENDENCE_PROBES:
        out = _render_or_exception(render, build(copy.deepcopy(value)))
        seen.add(f"{type(out).__name__}:{out}" if isinstance(out, BaseException)
                 else str(out))
        if len(seen) > 1:
            return True
    return False


def _most_failable_context(render, contexts, reaches, with_value):
    """Of the contexts in which this key is READ, the one a sweep can FAIL in.

    Taking the FIRST such context -- the round-15 ladder -- is the defect this
    replaces, and it was still live in two different ways.

    * The first context may read the key and do nothing with the answer, so no
      value placed there can change any observable. Measured by
      `_value_reaches_the_output`, which ranks above everything else here.
    * The first context may read the key and USE it only in a shape the sweeps
      cannot make fail. `possible_values.get("type")` renders into the summary
      in the bare leaf -- value-dependent, genuinely read -- but the `summary +=`
      that a wrong-shaped type kills only runs when a `value` sits BESIDE it, so
      reverting the coercion that fixed it left 924 tests green with four live
      TypeErrors restored. The leaf has to be the SIBLING-COMPLETE one, when the
      read survives there.

    So: among the contexts the key is actually read in, prefer a value-dependent
    one, and among those the RICHEST -- the most siblings present. Richness is a
    proxy for "the most of the renderer's own branches are open", and it is
    bounded by reachability rather than by a ladder position: a context that
    hides the read (a retained alias only consulted when the canonical key is
    ABSENT, an `elif` after a sibling gate) never enters `reached` at all, which
    is why filling siblings cannot repeat round 12's classification defect here.
    Ties keep the caller's order, so the least-perturbing of two equals wins."""
    reached = [ctx for ctx in contexts if reaches(ctx)]
    if not reached:
        return None
    ranked = sorted(enumerate(reached), key=lambda pair: (-len(pair[1]), pair[0]))
    for _rank, ctx in ranked:
        if _value_reaches_the_output(render, lambda v, _c=ctx: with_value(_c, v)):
            return ctx
    return ranked[0][1]


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


def _lookup_key(node, aliases=None):
    """The string key a node reads off a mapping, or None. `x.get("k")`,
    `x["k"]`, and a LOCAL bound from one of those.

    The alias hop is not a nicety: `op = item.get("op")` followed by
    `op == "types_declare"` is the module's DOMINANT gate spelling, and a rule
    that matched only the inline form harvested no key at all for such a
    function -- so its gates fell back to "one literal in EVERY slot", which
    opens the branch and simultaneously fills the earlier alternative of every
    `or` chain inside it. A key read only on the LATER alternative
    (`item.get("local_id") or item.get("variable")`) was then read in no context
    the discovery ever built, which is round 13's conjunction hole reached
    through a variable instead of a second key."""
    import ast

    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "pop", "setdefault") and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return node.args[0].value
    if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)):
        return node.slice.value
    if isinstance(node, ast.Name) and aliases:
        return aliases.get(node.id)
    return None


def _lookup_aliases(fn):
    """Every local in one function bound to a keyed read, `name -> key`. Last
    binding wins: `op = item.get("op", "?")` followed by `op = str(op)` keeps
    the key, because the second binding names none."""
    import ast

    aliases: dict[str, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            key = _lookup_key(node.value)
            if key:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = key
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            key = _lookup_key(node.value)
            if key:
                aliases[node.target.id] = key
    return aliases


@functools.lru_cache(maxsize=1)
def _keyed_comparison_constants():
    """Per function, the constants each KEY is compared against BY NAME.

    `_comparison_constants` harvests the constants and loses which key wanted
    them, so the probe could only ever put ONE value in EVERY slot at a time.
    A read behind a CONJUNCTION of two different key values -- `kind ==
    "go_rename"` AND `phase == "apply"` -- is then never DISCOVERED at all, so
    no population pair exists for any later guard to check: proven by injecting
    exactly that gate and watching every behavioural assertion stay green while
    a malformed counter fabricated a 0 into a decision key.

    Pairing each key with its own literal is what makes a conjunction
    satisfiable, and it comes from the module's own AST, so a new two-key gate
    brings both its openers with it."""
    import ast
    import inspect

    from bn import formatters

    out: dict[str, dict[str, list]] = {}
    for fn in ast.walk(ast.parse(inspect.getsource(formatters))):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        keyed: dict[str, list] = {}
        aliases = _lookup_aliases(fn)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            sides = [node.left, *node.comparators]
            keys = [k for k in (_lookup_key(side, aliases) for side in sides) if k]
            if not keys:
                continue
            consts = []
            for side in sides:
                parts = (side.elts if isinstance(side, (ast.Tuple, ast.List, ast.Set))
                         else [side])
                for part in parts:
                    if (isinstance(part, ast.Constant)
                            and isinstance(part.value, (str, int, bool))):
                        consts.append(part.value)
            for key in keys:
                slot = keyed.setdefault(key, [])
                for const in consts:
                    if const not in slot:
                        slot.append(const)
        out[fn.name] = keyed
    return out


def _keyed_literals(fn_name):
    """`_keyed_comparison_constants` for one renderer, widened through the call
    graph the way `_comparison_literals` is -- a nested gate's constant lives in
    the helper, not in the renderer."""
    merged: dict[str, list] = {}
    constants = _keyed_comparison_constants()
    for name in (fn_name, *sorted(_module_reach().get(fn_name, ()))):
        for key, values in constants.get(name, {}).items():
            slot = merged.setdefault(key, [])
            for value in values:
                if value not in slot:
                    slot.append(value)
    return merged


def _gated_contexts(keys, keyed):
    """Contexts that satisfy SEVERAL key gates at once: each key holds a
    constant IT is compared against, simultaneously. One context per round of
    the longest literal list, so a key with several openers gets each of them
    while its neighbours keep theirs."""
    relevant = {k: keyed[k] for k in keys if keyed.get(k)}
    if not relevant:
        return []
    rounds = max(len(v) for v in relevant.values())
    return [{k: v[i % len(v)] for k, v in relevant.items()}
            for i in range(rounds)]


@functools.lru_cache(maxsize=1)
def _module_reach():
    """Per function, every function in the module it can reach -- by CALLING it
    or by handing it somewhere as a callback (`_render_paged_list_text(value,
    "items", _render_strings_rows)`), which a call-only graph misses.

    Two uses, both of them repairs to a rule that matched on NAMES: the fillers
    below need the constants of the helpers a renderer reaches, because a branch
    gated on `kind == "unmodeled_callee"` is opened by a constant that lives in
    the helper and not in the renderer; and the nested-coverage assertion needs
    to know which renderer's runtime reads can stand for a read the AST
    attributes to a helper."""
    import ast

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    edges: dict[str, set[str]] = {name: set() for name in names}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in names and node.id != fn.name:
                edges[fn.name].add(node.id)
    reach: dict[str, frozenset] = {}
    for name in names:
        seen: set[str] = set()
        stack = [name]
        while stack:
            for callee in edges.get(stack.pop(), ()):
                if callee not in seen:
                    seen.add(callee)
                    stack.append(callee)
        reach[name] = frozenset(seen)
    return reach


_BOUNDARY_DECORATORS = ("_discloses", "_discloses_in_summary")


@functools.lru_cache(maxsize=1)
def _boundary_free_helpers():
    """Every module function a live ENTRY POINT reaches without crossing a
    disclosure boundary.

    THE question `_probe_renderers` has to answer before it may wrap a helper in
    a manufactured `@_discloses`, and the question round 15's assertion could
    not ask. That assertion compared the wrapped probes against the INVENTORY,
    which is the same set the wrap rule keys off -- so it was satisfied by
    construction and reported zero over the 19 helpers that are still wrapped.
    A guard whose population comes from the thing it guards cannot fail; this
    file's own rule, applied one level up from where it was last applied.

    The population here comes from the MODULE's call graph instead. A helper the
    package can reach from an entry point through only undecorated functions has
    no boundary in production, so a skew it records drains nowhere and wrapping
    it in the probe manufactures a disclosure the user never gets -- exactly the
    class round 14 deleted for `_format_operation_result`, which this found
    surviving for two more helpers. Such a helper must be probed BARE and fixed.

    Edges are `_module_reach`'s -- any NAME reference, so a callback handed to a
    paged renderer counts as reachable -- and the walk stops AT a boundary: a
    decorated function drains what everything under it recorded, so nothing
    below it is boundary-free through that path."""
    import ast

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    funcs = {node.name: node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def is_boundary(name):
        node = funcs.get(name)
        if node is None:
            return False
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            spelling = getattr(target, "attr", getattr(target, "id", None))
            if spelling in _BOUNDARY_DECORATORS:
                return True
        return False

    edges: dict[str, set[str]] = {name: set() for name in funcs}
    for name, fn in funcs.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in funcs and node.id != name:
                edges[name].add(node.id)

    exposed: set[str] = set()
    for entry in sorted(_cli_referenced_formatters()):
        if entry not in funcs or is_boundary(entry):
            continue
        seen, stack = {entry}, [entry]
        while stack:
            for callee in sorted(edges[stack.pop()]):
                if callee in seen:
                    continue
                seen.add(callee)
                exposed.add(callee)
                if not is_boundary(callee):
                    stack.append(callee)
    return frozenset(exposed)


@functools.lru_cache(maxsize=1)
def _helpers_observed_without_a_recorder():
    """Which module helpers actually RUN with no skew recorder installed.

    The second, INDEPENDENT oracle for the same question `_boundary_free_helpers`
    answers, and the reason the assertion that uses it is no longer empty by
    construction. That walk is static -- decorator names and call-graph edges
    read out of the AST -- and the probe's wrap rule KEYS OFF IT, so comparing
    the wrapped set against it can only ever return the empty set. Round 16's
    major: narrowing the walk to the two names its own anti-vacuity assertion
    pins left everything passing.

    This one asks the running module instead. Every module function is spied on
    (the spy records the value of `_SKEWED_FIELDS` AT ENTRY, before the callee
    can install its own), then every live CLI entry point is driven BARE with
    the payloads the population discovered for it. A helper that arrives with
    `None` there has no boundary above it on that path, whatever the AST thinks
    -- so if the static walk silently stops naming it, the probe wraps it, and
    this catches the manufactured disclosure the walk was supposed to prevent.

    Two names are not recorded. A BOUNDARY running with no outer recorder is the
    normal case, and so is the entry point currently being DRIVEN -- the static
    walk records callees only, and counting the entry against itself would be a
    disagreement the harness manufactured.

    What it cannot see, stated: a helper on a branch these payloads do not
    reach. That is why it is a CROSS-CHECK on the walk and not a replacement for
    it -- the walk is the conservative over-approximation, this is the live
    witness, and they are asserted to agree in the direction that matters."""
    from bn import formatters

    # Every payload is built BEFORE a spy exists, so discovery cannot be
    # mistaken for a live call.
    entries = sorted(_cli_referenced_formatters())
    probes = {label.split("(")[0]: probe for label, probe in _probe_renderers()}
    drives = []
    for name in entries:
        fn = getattr(formatters, name, None)
        if not inspect.isfunction(fn):
            continue                           # a re-exported constant
        # The probe callable when there is one: it already supplies the
        # renderer's flags, and the wrap rule leaves an ENTRY bare, which is
        # precisely the surface this oracle has to drive. Otherwise call it
        # directly -- an entry the probe excludes is still an entry the static
        # walk starts from, and dropping it here would let the two oracles
        # compare different populations.
        call = probes.get(name, fn)
        asked: set[str] = set()
        for _ in range(4):                     # fixed point: gated keys open
            before = frozenset(asked)
            for filler in (None, "list", "dict"):
                ctx = {k: copy.deepcopy(_PROBE_WELL_FORMED[filler])
                       for k in sorted(asked)}
                _render_or_exception(call, _KeyProbe(ctx, asked))
            if frozenset(asked) == before:
                break
        for filler in ("dict", "list", None):
            drives.append((name, call, {k: copy.deepcopy(_PROBE_WELL_FORMED[filler])
                                        for k in sorted(asked)}))
        # Not every entry consumes a MAPPING: the target chooser is handed the
        # choice list itself. Driving only dict-shaped payloads left the helper
        # under it unwitnessed, which is the shape of hole this oracle exists to
        # refuse.
        drives.append((name, call, {}))
        drives.append((name, call, _PROBE_WELL_FORMED["list"]))
        drives.append((name, call, "probe"))

    observed: set[str] = set()
    driving = [""]
    originals = {name: getattr(formatters, name) for name in dir(formatters)
                 if inspect.isfunction(getattr(formatters, name, None))
                 and getattr(formatters, name).__module__ == formatters.__name__}

    def spy(name, fn):
        @functools.wraps(fn)
        def watched(*args, **kwargs):
            if (formatters._SKEWED_FIELDS.get() is None
                    and getattr(fn, "__wrapped__", None) is None
                    and name != driving[0]):
                observed.add(name)
            return fn(*args, **kwargs)
        return watched

    real_json = formatters.json
    formatters.json = _QuietJson(real_json)
    try:
        for name, fn in originals.items():
            setattr(formatters, name, spy(name, fn))
        for name, call, payload in drives:
            driving[0] = name
            _render_or_exception(call, payload)
            driving[0] = ""
    finally:
        for name, fn in originals.items():
            setattr(formatters, name, fn)
        formatters.json = real_json
    return frozenset(observed)


def _comparison_literals(fn_name):
    """The fillers for one renderer: the constants it compares against, plus the
    constants of every function it reaches. Scoping this to the renderer's OWN
    function was round 10's residue -- a nested read behind `kind ==
    "unmodeled_callee"` is gated by a constant that lives in the grouping
    helper, so no filler ever opened it."""
    constants = _comparison_constants()
    found: list = list(constants.get(fn_name, ()))
    for reached in sorted(_module_reach().get(fn_name, ())):
        for value in constants.get(reached, ()):
            if value not in found:
                found.append(value)
    return tuple(found)


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
            keyed = _keyed_literals(name.split("(")[0])
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
                # Value-gated branches whose gates are a CONJUNCTION: every key
                # at a constant IT is compared against, all at once. The pass
                # above puts ONE value in EVERY slot, which can never satisfy
                # `kind == "go_rename" and phase == "apply"` -- so such a read
                # was not merely swept in the wrong context, it was never
                # DISCOVERED, and no population pair existed for any guard to
                # check. Round 13's blocker.
                for gate in _gated_contexts(sorted(seen), keyed):
                    for filler in fillers:
                        ctx = {**{k: copy.deepcopy(_PROBE_WELL_FORMED[filler])
                                  for k in sorted(seen)}, **gate}
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
                                  for literal in literals),
                                # The gate ALONE, before the filled variants: a
                                # filler is a VALUE, and a value satisfies the
                                # earlier alternative of an `or` fallback --
                                # `item.get("local_id") or item.get("variable")`
                                # never reaches `variable` in any context whose
                                # other keys are filled, so that read was
                                # recorded against the bare payload (where the
                                # op gate is absent and it is not read either)
                                # and its sweep entered no branch at all. Same
                                # for a dict read only through `.get()` on a
                                # fallback path: unfilled, `requested` is
                                # walked and classifies as the container it is.
                                *({g: v for g, v in gate.items() if g != key}
                                  for gate in _gated_contexts(keys, keyed)),
                                *({**{k: copy.deepcopy(_PROBE_WELL_FORMED["dict"])
                                      for k in keys if k != key},
                                   **{g: v for g, v in gate.items() if g != key}}
                                  for gate in _gated_contexts(keys, keyed))]
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
                    if observed:
                        records[key] = observed
                        continue
                    # Not a container position anywhere -- so WHERE is this key
                    # read? Falling straight back to the BARE payload for every
                    # unclassified key was a live bypass and not a detail: a read
                    # behind a value-gated branch (`kind == "go_rename"`, and all
                    # six of that op's counters behind it) happens in exactly one
                    # of these contexts and in none of the others, so the sweeps
                    # below swept a branch they never entered and the only
                    # tripwire left for a read added there was a re-baselineable
                    # size counter.
                    #
                    # And the FIRST context that reads it is not good enough
                    # either -- that was round 15's residue. See
                    # `_most_failable_context`.
                    def _reaches(ctx, _key=key, _render=render):
                        asked: set[str] = set()
                        _render_or_exception(
                            _render,
                            _KeyProbe({**copy.deepcopy(ctx), _key: "probe"}, asked))
                        return _key in asked

                    chosen = _most_failable_context(
                        render, contexts, _reaches,
                        lambda ctx, v, _key=key: {**copy.deepcopy(ctx), _key: v})
                    records[key] = (contexts[0] if chosen is None else chosen, None)
                kinds = {key: rec[1] for key, rec in records.items()}
            for key in keys:
                ctx, kind = records[key]
                population.append((name, render, key, kind, ctx))
        return population
    finally:
        formatters.json = real_json


def _watched(kind):
    """The container placed at a candidate position to see whether it is WALKED.

    THREE elements, and the middle one DIFFERENT, for the reason `_payload_for`
    gives. A one-element list makes every element both the first and the last,
    so a read reached only at a non-terminal or non-initial position was
    discovered by nothing. Two identical elements closed that instance and left
    the axis open twice over: index 2 was still unreachable, and an
    adjacent-elements-DIFFER gate (a de-duplicating pass, an `(xN)` collapse, an
    `if row != previous` separator) is permanently False when the same object is
    at both positions, so a read behind one was discovered by no payload this
    file built. Round 16 proved both live by injection."""
    return (_WatchedList([dict(_PROBE_ELEMENT), dict(_PROBE_SIBLING_ELEMENT),
                          dict(_PROBE_ELEMENT)]) if kind == "list"
            else _WatchedDict(_PROBE_ELEMENT))


def _disclosed(out, key):
    """Did this render disclose THIS key -- not merely SOME key?

    `"malformed" in out` was the pass condition, and any sibling field's note
    satisfied it: at 29 of the 366 container positions the baseline render
    already carries a note, so the property could not fail there and a newly
    added raw-coerced read shipped silent and green. The note names the fields
    it covers (`! malformed code_refs, data_refs fields: ...`), so the honest
    test is whether the key under test is one of them."""
    pattern = re.compile(rf"\bmalformed\b.*\b{re.escape(key)}\b")
    return any(pattern.search(line) for line in out.splitlines())


def _payload_for(ctx, path, leaf):
    """The renderer's payload with `leaf` placed at the end of `path`.

    A path step is `(key, kind)` or `(key, kind, siblings)`: a list-kind step
    puts the node in the list at that key, a dict-kind step puts the node AT the
    key, because that is what the renderer walks in each case. `siblings` is the
    context the node CARRYING that key was observed with, so an intermediate
    element is rebuilt as it was SEEN rather than as a one-key dict -- a read
    gated on a sibling of an intermediate key is otherwise never reached at the
    level below it. The first step needs none: the node carrying it is the
    renderer's own payload, which is `ctx`.

    A list step carries the node at index 0 AND index 2, with a DIFFERENT
    well-formed element between them, and none of that is a detail. Every
    population and every sweep in this file used to build lists of cardinality
    exactly ONE, so an element was always simultaneously the first and the last
    and a read gated on POSITION -- `if i != last`, `if idx`, `rows[1:]`, a
    separator between elements, `len(rows) > 1` -- was outside all six sweeps.
    That is the eighth axis: not depth, not siblings, CARDINALITY. Cardinality
    two closed that instance and left two more shapes of the same axis open,
    both proven live by injection at round 16: a read reached only at index >= 2
    (a top-N slice's "... and N more" tail is exactly this shape), and a read
    behind an adjacent-elements-DIFFER gate, which cannot open at all while the
    same object sits at every position. `[node, other, node]` puts the node
    under test at a first, a last and a >= 2 position, and puts a difference on
    both sides of it, in one render.

    The SAME object is placed at both of the node's positions rather than a
    copy, because the leaf may be a recording probe whose `hits`/`asked` are
    read back by identity afterwards -- a deep copy would silently drop a read
    that only happens at the last element, which is the very thing this covers.
    A renderer that mutates it is caught by
    `test_no_renderer_mutates_the_payload_it_was_handed`."""
    node = leaf
    for step in reversed(path):
        key, kind = step[0], step[1]
        siblings = step[2] if len(step) > 2 else {}
        node = {**copy.deepcopy(siblings),
                key: [node, dict(_PROBE_SIBLING_ELEMENT), node]
                if kind == "list" else node}
    return {**copy.deepcopy(ctx), **node}


# A RUNAWAY GUARD, never the stopping condition. The descent stops when a level
# discovers no further container, and `test_the_nested_population_converges_before_the_depth_cap`
# asserts the deepest path is strictly shallower than this cap -- which is the
# proof it converged rather than being cut off. A fixed depth of 2 was the
# round-10 blocker: it read as a stated limit and behaved as a live bypass,
# because a guard-blind coercion one container below the cap kept the whole file
# green. A cap that the data never reaches cannot hide a read behind itself.
_NEST_DEPTH_CAP = 12


@functools.lru_cache(maxsize=1)
def _nested_population():
    """The same discovery, every container BELOW the payload.

    The top-level probe records what a renderer asks of its own payload, so a
    container read out of a callee row or a per-match ref bucket is invisible to
    it: those reads were counted (the `nested` bucket) and then never
    differentially tested, so a silent-empty coercion at a nested position
    passed every guard in this file -- proven by injecting a one-hop helper at a
    nested ref bucket with all 176 tests green.

    Counting is not covering. This descends instead: for every container
    position already observed, it hands that container a RECORDING element,
    collects the keys the renderer asks of it, classifies each by whether a
    watched container placed there is actually walked, and pushes the ones that
    are onto the next level's frontier. The population is still only what a
    renderer was OBSERVED reading -- never what the module declares."""
    from bn import formatters

    real_json = formatters.json
    formatters.json = _QuietJson(real_json)
    try:
        nested = []
        frontier = [(name, render, (key, kind), ctx)
                    for name, render, key, kind, ctx in _runtime_population()
                    if kind is not None]
        for _level in range(_NEST_DEPTH_CAP):
            next_frontier = []
            for name, render, step, ctx in frontier:
                path = step if isinstance(step[0], tuple) else (step,)
                literals = _comparison_literals(name.split("(")[0])
                keyed = _keyed_literals(name.split("(")[0])
                seen: set[str] = set()
                # The leaf each key was FIRST asked in. The discovery's fixed
                # point grows `seen` one pass at a time, so it necessarily
                # passes through intermediate leaves -- and some reads happen
                # ONLY there: `variadic.get("format_string")` is an `elif` after
                # `under_recovered and warning`, so a leaf with every sibling
                # filled takes the other branch and no all-or-nothing ladder can
                # reach it. Keeping the leaf the read was observed in makes the
                # recorded context reachable BY CONSTRUCTION rather than by a
                # ladder that happens to be long enough.
                first_leaf: dict[str, dict] = {}

                def _discover(base, _first=first_leaf):
                    asked: set[str] = set()
                    _render_or_exception(
                        render, _payload_for(ctx, path, _KeyProbe(base, asked)))
                    for key in asked:
                        _first.setdefault(
                            key, {k: v for k, v in base.items() if k != key})
                    return asked

                for _ in range(4):                    # fixed point, as at top level
                    before = frozenset(seen)
                    for filler in (None, "list", "dict"):
                        seen |= _discover(
                            {**_PROBE_ELEMENT,
                             **{k: copy.deepcopy(_PROBE_WELL_FORMED[filler])
                                for k in sorted(seen)}})
                    for literal in literals:
                        seen |= _discover(
                            {**_PROBE_ELEMENT, **{k: literal for k in sorted(seen)}})
                    for gate in _gated_contexts(sorted(seen), keyed):
                        for filler in (None, "list", "dict"):
                            seen |= _discover(
                                {**_PROBE_ELEMENT,
                                 **{k: copy.deepcopy(_PROBE_WELL_FORMED[filler])
                                    for k in sorted(seen)}, **gate})
                    if frozenset(seen) == before:
                        break
                # The leaf context ladder, least-perturbing first, and for the
                # same reason the top level has one: the sweeps below REBUILD
                # the leaf from what is recorded here, so a key read only when a
                # SIBLING is present (`if args:` before the argument tag, a
                # `value` beside the `type`) has to be recorded in a leaf that
                # HAS that sibling. Rebuilding every leaf from a bare probe
                # element left 187 of the nested rows swept in a branch they
                # never entered -- round 13's top-level defect one level down,
                # and it hid three live raises that this ladder surfaces.
                leaf_kinds: dict[str, str | None] = {}
                records: dict[str, tuple] = {}
                for _round in (0, 1):
                    records = {}
                    for nkey in sorted(seen):
                        bare = {k: v for k, v in _PROBE_ELEMENT.items() if k != nkey}
                        siblings = [s for s in sorted(seen) if s != nkey]
                        leaves = [
                            bare,
                            {**bare, **{s: copy.deepcopy(_PROBE_WELL_FORMED[leaf_kinds.get(s)])
                                        for s in siblings}},
                            {**bare, **{s: copy.deepcopy(_PROBE_WELL_FORMED["dict"])
                                        for s in siblings}},
                            {**bare, **{s: copy.deepcopy(_PROBE_WELL_FORMED["list"])
                                        for s in siblings}},
                            {**bare, **{s: "probe" for s in siblings}},
                            *({**bare, **{s: literal for s in siblings}}
                              for literal in literals),
                            # Conjunction gates, as at top level: a nested read
                            # behind two DIFFERENT sibling values (a format
                            # string beside a conversion list, a variadic callee
                            # beside its kind) is opened by no single filler, so
                            # each key gets a constant IT is compared against.
                            *({**bare, **{g: v for g, v in gate.items() if g != nkey}}
                              for gate in _gated_contexts(sorted(seen), keyed)),
                            *({**bare,
                               **{s: copy.deepcopy(_PROBE_WELL_FORMED["dict"])
                                  for s in siblings},
                               **{g: v for g, v in gate.items() if g != nkey}}
                              for gate in _gated_contexts(sorted(seen), keyed)),
                            # LAST, and the rung that makes the recorded context
                            # reachable by construction: the leaf this key was
                            # observed being asked in. Every rung above is a
                            # guess that happens to be less perturbing; this one
                            # is a measurement.
                            first_leaf.get(nkey, bare)]
                        observed = None
                        for leaf_ctx in leaves:
                            for kind in ("list", "dict"):
                                probe = _watched(kind)
                                _render_or_exception(render, _payload_for(
                                    ctx, path, {**copy.deepcopy(leaf_ctx), nkey: probe}))
                                if probe.hits & _CONTAINER_USE:
                                    observed = (leaf_ctx, kind)
                                    break
                            if observed:
                                break
                        if observed:
                            records[nkey] = observed
                            continue
                        # Same question as at top level, and the same answer:
                        # the first leaf that merely ASKS is not a leaf the
                        # sweeps can fail in. See `_most_failable_context` --
                        # `possible_values.get("type")` is the case it was
                        # written for.
                        def _reaches(leaf_ctx, _nkey=nkey, _render=render,
                                     _ctx=ctx, _path=path):
                            asked: set[str] = set()
                            _render_or_exception(_render, _payload_for(
                                _ctx, _path,
                                _KeyProbe({**copy.deepcopy(leaf_ctx), _nkey: "probe"},
                                          asked)))
                            return _nkey in asked

                        chosen = _most_failable_context(
                            render, leaves, _reaches,
                            lambda leaf_ctx, v, _nkey=nkey, _ctx=ctx, _path=path:
                                _payload_for(_ctx, _path,
                                             {**copy.deepcopy(leaf_ctx), _nkey: v}))
                        records[nkey] = (leaves[0] if chosen is None else chosen, None)
                    leaf_kinds = {k: rec[1] for k, rec in records.items()}
                for nkey in sorted(seen):
                    leaf_ctx, kind = records[nkey]
                    # Kind-free, exactly like the top level: an UNCLASSIFIED
                    # nested key still gets swept for raises, because a scalar
                    # read that dies on a wrong shape is the same #619 defect --
                    # a sampled string sliced as `(s.get("value") or "")[:80]`
                    # raised on a dict and cost the whole card. Only a classified
                    # container is descended into and differentiated.
                    nested.append((name, render, path, nkey, kind, ctx, leaf_ctx))
                    if kind is not None:
                        # The step carries the leaf context it was observed in, so
                        # the level below rebuilds this element as it was SEEN and
                        # not as a bare probe element.
                        next_frontier.append(
                            (name, render, path + ((nkey, kind, leaf_ctx),), ctx))
            frontier = next_frontier
            if not frontier:                  # converged: nothing left to descend into
                break
        return nested
    finally:
        formatters.json = real_json


def test_every_population_context_actually_reaches_the_read_it_was_recorded_for():
    """The population's third promise, made executable.

    `_runtime_population` states that "the context is the one the read was
    OBSERVED in". That was FALSE for 367 of 564 pairs: any key no watched
    container was walked at fell back to the BARE payload, so a read behind a
    value-gated branch was swept in a payload that never entered the branch.
    The guard could not fail there -- proven by adding a seventh `go rename`
    counter, which left every behavioural assertion green and moved only three
    size counters.

    A promise in a docstring is prose, and prose does not expire. This re-runs
    every recorded (renderer, key, context) triple and asserts the renderer
    ACTUALLY ASKS for that key in that context, which is the property the
    sweeps depend on and the one nobody could see was broken."""
    unreached = []
    for label, render, key, kind, ctx in _runtime_population():
        asked: set[str] = set()
        _render_or_exception(
            render, _KeyProbe({**copy.deepcopy(ctx), key: "probe"}, asked))
        if key in asked:
            continue
        # A container position may only be read when a container is what sits
        # there (`if isinstance(x, list)` gates the read itself), so give it the
        # kind it was classified as before calling the context unreachable.
        payload = _KeyProbe(copy.deepcopy(ctx), asked)
        payload[key] = _watched(kind or "list")
        _render_or_exception(render, payload)
        if key not in asked:
            unreached.append(f"{label}.{key}")
    assert not unreached, (
        "these population entries record a context the renderer never reads the "
        "key in, so every sweep below runs a branch it does not enter and "
        f"cannot fail there: {unreached[:8]}")


# The keys `_go_rename_summary` reads that are NOT counts: a kind discriminator,
# three flags, a message and the failure listing, plus the seven keys the paired
# renderer reads back off the summary it produced. Declared here so the runtime
# read set below can be split, and so a NEW key has to arrive in one bucket or
# the other deliberately.
_GO_RENAME_NON_COUNT_READS = {
    "kind", "committed", "preview", "success", "rolled_back", "message",
    "results", "prototype_user_type_residue",
    "changed_count", "verified_count", "noop_count", "failed_count",
    "measured", "dirty_after", "first_error",
}


def test_every_count_the_go_rename_summary_decides_on_is_covered_by_measured():
    """`measured` is derived from the counter reads, so a counter read OUTSIDE
    that loop is a counter `measured` does not cover -- and a fabricated 0 from
    such a read is this op's "nothing changed, do not save" verdict (#683).

    A SET, not a number, and discovered by RUNNING the transform rather than by
    matching its source: `_KeyProbe` reports the keys the composed entry point
    actually asked for, so a seventh counter read appears here however it is
    spelled -- `int(value.get(...))`, an isinstance-guarded coercion, an alias
    hop -- and lands in neither bucket. Both injection forms round 12 used moved
    only re-baselineable size counters before this existed.

    Deliberately NOT derived from `_count_field` call sites: a population taken
    from the choke point cannot fail on a read that skipped the choke point,
    which is the whole defect class this file keeps re-finding."""
    from bn import formatters

    read = {key for label, _, key, _, _ in _runtime_population()
            if label.split("(")[0] == "_go_rename_summary"}
    counters = set(formatters._GO_RENAME_COUNTERS)
    assert counters <= read, (
        f"{sorted(counters - read)} is declared a `go rename` counter and the "
        "composed transform never reads it -- the loop that decides `measured` "
        "has stopped covering it")
    stray = sorted(read - counters - _GO_RENAME_NON_COUNT_READS)
    assert not stray, (
        f"the go-rename summary reads {stray}, which is in neither "
        "_GO_RENAME_COUNTERS nor the declared non-count reads. If it is a "
        "counter it must be read through the loop that decides `measured`, or "
        "an unreadable one fabricates a 0 into a DECISION key; if it is not, "
        "add it to _GO_RENAME_NON_COUNT_READS and say why in the commit.")


def _count_helper_sites():
    """Every `(function, literal key)` the module reads through a COUNT helper.

    Harvested from the module's own AST, like the other guards here, and keyed
    on the enclosing function so the differential below can look the renderer up
    in the discovered population. A dynamic key is not harvested: there is no
    payload this file could build for it -- which is also why the count family
    itself stays out, since each member passes its caller's `key` PARAMETER
    down to the next.
    """
    import ast
    import inspect

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    funcs = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def owner(node):
        enclosing = [f for f in funcs if f.lineno <= node.lineno <= f.end_lineno]
        return (min(enclosing, key=lambda f: f.end_lineno - f.lineno).name
                if enclosing else None)

    sites = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("_count_field", "_stated_count",
                                     "_nonnegative_count")):
            continue
        if len(node.args) != 2:
            continue
        key = node.args[1]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        name = owner(node)
        if name is not None:
            sites.add((name, key.value))
    return sites


def test_no_renderer_states_a_count_it_could_not_read_as_a_real_number():
    """THE differential for #619's count half, stated as a runtime DIFFERENCE
    rather than as a spelling rule.

    A count read through the choke point answers 0 for a counter it could not
    use, and the `@_discloses` note lands on a LATER line -- so an unreadable
    `go_verified_count` rendered "go rename: 0 renamed, 0 failed, 0 skipped",
    byte-identically to a genuine all-noop run, on the line a caller acts on.
    That is the #683 fabricated-zero harm with a footnote attached.

    Three renders per (renderer, key), and the third is what makes the property
    non-vacuous: the BODY (everything before the disclosure note the boundary
    appends) is taken with the key at 0, at 7, and at a shape no count reads.

      * body(0) == body(7) means this renderer does not STATE that count in the
        context the population recorded -- nothing to impersonate, so the pair
        is skipped, and the skipped set is asserted by name below so a renderer
        that stops stating a count cannot silently leave the differential.
      * otherwise the unreadable body must match NEITHER, because either match
        is an unreadable counter wearing a real number's rendering. A trailing
        note cannot satisfy this, which is the point: the note arrives after the
        line a caller acts on.

    Sibling count keys are set readable in the context, because a renderer's
    whole-line refusal for one unreadable counter (`go rename: cannot say what
    this run did`) otherwise hides every other counter's rendering behind it.
    """
    sites = _count_helper_sites()
    # 18 -> 19 (#858 review r5): `_render_trace_text` now reads `arg_index`
    # through `_stated_count`, which is a read this differential covers.
    # 19 -> 20 (#795): `_render_strings_text` now reads `filtered` through
    # `_count_field` to state how many strings the active filters dropped.
    # 20 -> 22 (#795 round-3 review): the imports LISTING and `--summary`
    # renderers now read `self_defined_excluded` through `_count_field` too --
    # they tested it with `isinstance(int)` while the `--count` line beside
    # them went through the choke point, so one payload got three different
    # descriptions. Two more (renderer, key) pairs, measured.
    # 22 -> 23 (#795 round-4 review): `_render_function_count_text` -- the
    # renderer behind `function list --count`, `function search --count` and
    # `types --count`, so the WIDEST count line in the CLI -- now reads its
    # count through `_stated_count` instead of interpolating it raw.
    # 23 -> 29 (#795 round-6 review): `_render_class_list_text` stated SIX
    # numbers of its own raw -- the count-only headline through `_stated_count`
    # now, and the non-class artifact share plus the three suppressed shares
    # through `_nonnegative_count`, because a cardinality cannot be negative.
    # Six more (renderer, key) pairs this differential now covers, measured.
    # 29 -> 30 (#795 round-7 review): the per-ROW method count in the same
    # renderer -- the one number the round-6 repair walked past -- now reads
    # through `_stated_count` too. One more (renderer, key) pair, measured.
    assert len(sites) == 30, (
        f"the module reads {len(sites)} (renderer, literal key) pairs through a "
        "count helper, not 30. The number is the size of the covered set: a "
        "read that vanishes is a read this differential stops running, so move "
        "it only with the read you deliberately added or removed.")

    keys_by_renderer = collections.defaultdict(set)
    for name, key in sites:
        keys_by_renderer[name].add(key)
    merged = collections.defaultdict(dict)
    for label, _render, key, _kind, ctx in _runtime_population():
        merged[label.split("(")[0]].update(copy.deepcopy(ctx))
    renderers = {}
    for label, render in _probe_renderers():
        renderers.setdefault(label.split("(")[0], render)

    def body(out):
        # Everything the renderer said BEFORE the boundary's disclosure note.
        return out.split("\n! malformed")[0]

    fabricated, not_stated, checked = [], [], 0
    for name, key in sorted(sites):
        render = renderers.get(name)
        if render is None:
            # Not a payload renderer: a fragment helper (`_paging_footer`,
            # `_blast_radius_line`, `_operation_row_text`) whose count refusal
            # is pinned by its own named test.
            not_stated.append(f"{name}({key}) [not a payload renderer]")
            continue
        ctx = {**merged.get(name, {}), **{k: 3 for k in keys_by_renderer[name]}}
        rendered = {}
        for probe in (0, 7, "many"):
            out = _render_or_exception(render, {**copy.deepcopy(ctx), key: probe})
            rendered[probe] = None if isinstance(out, Exception) else body(out)
        if any(out is None for out in rendered.values()):
            not_stated.append(f"{name}({key}) [the renderer's own shape refusal]")
            continue
        if rendered[0] == rendered[7]:
            not_stated.append(f"{name}({key}) [count not stated in this context]")
            continue
        checked += 1
        impersonated = [str(probe) for probe in (0, 7)
                        if rendered["many"] == rendered[probe]]
        if impersonated:
            fabricated.append(
                f"{name}({key}) renders an unreadable counter exactly like "
                f"{'/'.join(impersonated)}: {rendered['many']!r}")
    assert checked, "no harvested pair states a count, so this proves nothing"
    assert not fabricated, (
        "these render an unreadable counter byte-identically to a real number, "
        f"so the disclosure arrives after the decision: {fabricated}")
    # The skipped set, by name: a pair leaves the differential only for a reason
    # stated here, so a renderer that quietly stops stating a count fails.
    assert sorted(not_stated) == [
        "_blast_radius_line(referenced) [not a payload renderer]",
        "_blast_radius_line(reflowed) [not a payload renderer]",
        "_operation_row_text(count) [not a payload renderer]",
        "_paging_footer(offset) [not a payload renderer]",
        "_paging_footer(returned) [not a payload renderer]",
        "_paging_footer(total) [not a payload renderer]",
        # #795 round-6 review: the class lens states these two on its
        # COUNT-ONLY branch, which this differential's context never takes --
        # it fills every harvested key, so the listing branch always wins.
        # Their honesty is pinned directly by
        # `test_the_class_listing_reads_its_count_and_every_cardinality_through_the_choke_point_795`.
        # The renderer's four `hidden:` shares ARE stated in this context and
        # are checked above.
        "_render_class_list_text(artifact_count) [count not stated in this context]",
        "_render_class_list_text(count) [count not stated in this context]",
        # #795 round-7 review: a ROW-level key. This differential fills every
        # harvested key on the top-level payload, and nothing there reaches a
        # row, so the pair is skipped by the HARNESS rather than by the
        # renderer -- which DOES state it, pinned in the named test above.
        "_render_class_list_text(method_count) [count not stated in this context]",
        "_render_function_evidence_text(offset) [count not stated in this context]",
        "_render_go_rename_text(defined_count) [count not stated in this context]",
        # #795 round-3 review: same harness cut as the strings pair below --
        # this renderer's inner `_render_paged_list_text` boundary discloses
        # first and the differential's body split stops at that note, so the
        # pair is skipped by the HARNESS, not by the renderer. The renderer DOES
        # state it: covered by name in
        # `tests/test_cli_misc.py::test_the_three_imports_surfaces_agree_about_the_excluded_count_795`,
        # which drives all three imports surfaces over the same payload.
        # Round-4 review: that cover used to be untrue. Every assertion it made
        # was satisfied by the trailing `@_discloses` note, which the renderer
        # gets whether or not it states the row -- so replacing this renderer's
        # unreadable branch with `pass` left the named test GREEN. It now cuts
        # the boundary note off and requires each surface's own body to differ
        # from the body it renders for a payload that claimed nothing, so the
        # exemption fails with the branch it exempts.
        "_render_name_address_list_text(self_defined_excluded) [count not "
        "stated in this context]",
        # Both live NESTED under `existing_annotations`, so a top-level probe
        # cannot open the presence gate that states them. Covered by name in
        # `test_render_orient_states_the_analyst_split_without_fabricating_it`,
        # which drives the real nested payload and asserts `placeholders=?` for
        # a count no value reads out of (#733 F2 review).
        "_render_orient_text(analyst_symbols) [count not stated in this context]",
        "_render_orient_text(placeholder_symbols) [count not stated in this "
        "context]",
        # #795: this pair DOES state the count -- the recorded context makes the
        # inner `_render_paged_list_text` boundary disclose first, and the
        # differential's body split cuts at that note, so the pair is skipped by
        # the HARNESS rather than by the renderer. Covered by name in
        # `test_strings_discloses_the_dropped_count_795` (page + count line) and
        # `test_strings_count_text_states_the_dropped_count_795`.
        "_render_strings_text(filtered) [count not stated in this context]",
    ], sorted(not_stated)


def test_the_call_window_never_states_a_resume_offset_it_could_not_derive():
    """The evidence card's own resume hint, which the differential above skips
    because the `has_more` branch is not open in the recorded context.

    Same harm as the paging footer's invented offset: `--offset 3` derived from
    an unreadable page position sends an agent paging from a window the payload
    never stated -- it re-reads what it has, or loops on the first page. The
    refusal states the position as unreadable instead."""
    from bn import formatters

    def card(offset):
        return formatters._render_function_evidence_text({
            "function": {"name": "log_printf", "address": "0x401000"},
            "calls": [{"address": "0x401010", "operation": "LLIL_CALL",
                       "direct": True}],
            "total_calls": 9, "matched_calls": 9, "offset": offset,
            "has_more": True,
        })

    readable = card(4)
    assert "rerun with --offset 5" in readable, readable

    refused = card("bad")
    assert "page position unreadable" in refused, refused
    assert "--offset" not in refused.split("\n! malformed")[0], refused


def test_the_go_rename_nothing_to_do_line_never_states_a_count_it_could_not_read():
    """The one `_stated_count` site the differential above cannot reach: this
    branch renders only when `go_renamed_candidates` is a readable zero, and the
    recorded population context opens the detail branch instead.

    The line is the actionable one on this op -- "nothing to do" is what a
    caller stops on -- so the two numbers it offers as the REASON must not be
    fabricated. A skewed `defined_count` used to render "(0 defined at pcln
    addresses)", which reads as "this binary has no Go symbol table" rather
    than "I could not read that count"."""
    from bn import formatters

    def nothing_to_do(**over):
        return formatters._render_go_rename_text({
            "kind": "go_rename", "success": True, "committed": True,
            "go_renamed_candidates": 0, "defined_count": 7,
            "skipped_user_named": 2, **over,
        })

    readable = nothing_to_do()
    assert "(7 defined at pcln addresses, 2 already user-named)" in readable, readable

    for key in ("defined_count", "skipped_user_named"):
        refused = nothing_to_do(**{key: "lots"})
        body = refused.split("\n! malformed")[0]
        assert "? " in body or "(? " in body, (key, refused)
        assert body != readable, (key, refused)
        assert f"malformed {key} field" in refused, refused


def test_the_function_count_line_never_states_a_count_it_could_not_read():
    """The widest `--count` line in the CLI, and the last one reading raw.

    This renderer is the `text_renderer` for `function list --count`,
    `function search --count` and `types --count`, and it interpolated
    `value.get("count", 0)` straight into the line -- so a bool rendered
    "Total functions: True" (a flag stated as a quantity), a container
    rendered a raw Python repr, an explicit null rendered "None", and a
    text-spelled count that IS a number was dropped to 0. That last one is the
    #683 harm on the loudest surface there is: "Total functions: 0" from an
    unreadable counter reads byte-identically to a binary with no functions,
    which is exactly the answer an agent stops on.

    Round-4 review: the commit that closed this class across the command
    module left this renderer out, so the claim was wider than the change.
    """
    from bn import formatters

    readable = formatters._render_function_count_text({"count": 175})
    assert readable == "Total functions: 175", readable
    # A count spelled as text IS a count, and states the same line.
    assert formatters._render_function_count_text({"count": "175"}) == readable

    for value in (True, {"n": 3}, [1, 2, 3], "lots", 1.5):
        refused = formatters._render_function_count_text({"count": value})
        body = refused.split("\n! malformed")[0]
        assert body == "Total functions: ?", (value, refused)
        assert f"{value}" not in body, (value, refused)
        assert "malformed count field" in refused, (value, refused)

    # An ABSENT or null count claimed nothing, so the honest zero is unchanged
    # and nothing is disclosed.
    for payload in ({}, {"count": None}):
        quiet = formatters._render_function_count_text(payload)
        assert quiet == "Total functions: 0", (payload, quiet)



# The raw numeric spellings this module still carries, MEASURED rather than
# described. Each is a count read that does not go through `_count_field` --
# `<payload lookup> or 0`, `.get(<literal>, 0)`, `int(<payload lookup>)` -- so
# an unreadable value there is a fabricated number with nothing disclosed. The
# routed audit converted the surfaces that state an ACTIONABLE number (the go
# rename headline, the evidence card's resume offset); the rest are descriptive
# counts on rows and summaries, and they are inventoried here so a NEW raw count
# read arrives red instead of joining a prose claim.
#
# `_render_mutation_summary_text`'s `value.get('changed_count', 0)` is
# deliberately in the residue: it reads a value the ALREADY-guarded summary
# transform put there (`None` for a refused counter, which is why `changed=None`
# prints), so a blanket zero-assertion would be wrong where an inventory is
# right.
# 49 -> 48 (#858 review r5): `_render_trace_text`'s `arg_index` was the last
# raw numeric spelling in that renderer and now goes through `_stated_count`.
# 48 -> 46 (#770): `_render_field_xrefs_text`'s bespoke paging footer read
# `value.get('offset', 0)` twice (bare and `or 0`) to build its own note; the
# renderer now delegates to `_paging_footer`, which reads all three counts
# through the choke point, so both spellings are deliberately GONE.
# 46 -> 45 (#795 round-4 review): `_render_function_count_text`'s
# `value.get('count', 0)` was the widest raw count read left in the module --
# three CLI surfaces install that renderer -- and now goes through
# `_stated_count`.
# 45 -> 39 (#795 round-6 review): `_render_class_list_text` stated SIX numbers
# of its own raw -- the count-only headline, the non-class artifact share
# beside it, and the three suppressed shares in the `hidden:` tail. The
# headline now reads through `_stated_count` and the four cardinalities
# through `_nonnegative_count`, so all six spellings are deliberately GONE.
# 39 -> 38 (#795 round-7 review): the SEVENTH number in that renderer, the
# per-ROW method count, which the round-6 repair walked past -- it rendered a
# flag as a quantity and a container as a Python repr, undisclosed, in the
# very renderer the round-6 major was filed against. Now `_stated_count`.
_RAW_COUNT_SPELLINGS = 38


def test_the_raw_count_residue_is_exactly_this_big():
    """The inventory, so the routed item above stops being a prose claim."""
    import ast
    import inspect

    from bn import formatters

    def is_lookup(node):
        if isinstance(node, ast.Subscript):
            return True
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop"))

    tree = ast.parse(inspect.getsource(formatters))
    get_zero, or_zero, int_lookup = [], [], []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 0):
            get_zero.append(f"{ast.unparse(node)} at line {node.lineno}")
        if (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and len(node.values) == 2
                and isinstance(node.values[1], ast.Constant)
                and node.values[1].value == 0
                and is_lookup(node.values[0])):
            or_zero.append(f"{ast.unparse(node)} at line {node.lineno}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "int" and node.args
                and is_lookup(node.args[0])):
            int_lookup.append(f"{ast.unparse(node)} at line {node.lineno}")
    residue = get_zero + or_zero + int_lookup
    assert len(residue) == _RAW_COUNT_SPELLINGS, (
        f"the module carries {len(residue)} raw count spellings "
        f"({len(get_zero)} `get(k, 0)`, {len(or_zero)} `or 0`, "
        f"{len(int_lookup)} `int(lookup)`), not {_RAW_COUNT_SPELLINGS}. A NEW "
        "one is a fabricated number with nothing disclosed -- route it through "
        "`_count_field`/`_stated_count`; one you deliberately REMOVED moves "
        f"this constant in the same commit. Current: {sorted(residue)}")
    # `int(<payload lookup>)` is the spelling that RAISES rather than
    # fabricating, and the module carries none: an arriving one costs a whole
    # render, which is the harm `_count_field` exists to end.
    assert not int_lookup, (
        f"a payload lookup reaches bare `int()`, which raises: {int_lookup}")


# The declared choke-point reads neither differential reaches, each with a
# payload that DOES reach it. These are helpers behind a branch gated on a value
# no probe filler supplies at the position that needs it: a specific op name
# paired with a requested shape, a truncation cause, a frontier-leaf kind on the
# nested ELEMENT rather than on the payload, a callee row's resolved target.
#
# The point of the table is that the exemption stops being prose. Each builder
# places the value under test at the read's own position, and
# `test_every_uncovered_read_is_covered_directly` asserts the choke point
# actually RECORDED the skew there -- which is only possible if the payload
# reached the read. A builder that stops reaching it fails, instead of the list
# quietly certifying a read nothing runs.
# One builder PER CALL SITE, not per (function, key) pair -- see the uniqueness
# half of the test below. `_leaf_group_key` reads `callee` behind two different
# leaf kinds, and a single builder would have exempted both while exercising
# one.
_RESIDUE_DIRECT_COVER = {
    "_blast_radius_line.affected_summary":
        ("dict", [lambda v: {"affected_summary": v}]),
    "_types_affected_lines.affected_types":
        ("list", [lambda v: {"affected_types": v}]),
}


def test_every_uncovered_read_is_covered_directly():
    """The residue's exemption, executed.

    Five of the seven entries occurred exactly ONCE in this file -- inside the
    assertion list that exempted them -- under a comment claiming "each with its
    own named test above". There was no such test, so a guard-blind coercion at
    an exempted read restored the base crash with the whole file green. This is
    the third time in this PR that a stated reason turned out to be false, and
    the answer is the same one that worked for `_PROBE_EXCLUSIONS`: make the
    reason run.

    The property is the choke point's own record, not the rendered text: these
    are helpers whose output is a tuple, a list, a fragment or a line, and
    disclosing a skew is the enclosing renderer's job. What must hold is that
    the read went THROUGH the choke point, which is observable exactly here --
    and it also proves the payload reached the read, since an unreached read
    records nothing.

    Both directions: a well-formed container at the same position records
    nothing, so an unconditional `_record_skew` could not satisfy this.

    An exemption must also be UNIQUE, not a pattern. A cover keyed by
    `(function, key)` matches every CALL SITE of that pair, so a second read of
    the same key added behind a different gate in the same function would
    silently INHERIT this one's cover -- the same defect a sibling PR found in
    its own named exemption, matched by pattern where it should be matched by
    identity. Each entry therefore has to cover exactly ONE site, and the total
    is pinned: every exemption is a hole someone promised not to look through,
    so the promise must be both executable and unique."""
    import ast
    import inspect

    from bn import formatters

    sites: list[tuple] = []
    tree = ast.parse(inspect.getsource(formatters))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in _RECORDERS):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        sites.append((fn.name, arg.value, node.lineno))
    covered = 0
    for label in sorted(_RESIDUE_DIRECT_COVER):
        fn_name, key = label.rsplit(".", 1)
        builders = _RESIDUE_DIRECT_COVER[label][1]
        matching = [s for s in sites if s[0] == fn_name and s[1] == key]
        assert len(matching) == len(builders), (
            f"{label} exempts {len(matching)} call sites and this entry "
            f"exercises {len(builders)}: {matching}. An exemption that matches "
            "more reads than it runs is a PATTERN, and a read behind a "
            "different gate inherits it silently. Give the new site its own "
            "builder, or get it into a population.")
        covered += len(matching)
    assert covered == 2, (
        f"the residue cover accounts for {covered} reads, not 2 -- the size of "
        "the exempted set, which moves only with an entry you deliberately "
        "added or removed. It has shrunk from eight to two in this round alone: "
        "the identity sweep put `_format_operation_result` in the top-level "
        "population, and the nested descent -- once it rebuilt each leaf from "
        "the context the read was OBSERVED in -- started exercising three more "
        "of these directly. An exempted read that gets a real differential "
        "must LEAVE this table, or the exemption outlives the hole it was "
        "written for.")

    for label, (kind, builders) in sorted(_RESIDUE_DIRECT_COVER.items()):
        fn_name, key = label.rsplit(".", 1)
        entry = getattr(formatters, fn_name)
        # Unwrap a `@_discloses` boundary: it installs its OWN recorder on
        # entry, which would swallow the record this test measures. The property
        # here is that the read goes through the CHOKE POINT at all; what the
        # enclosing boundary then does with it is the differentials' business.
        fn = getattr(entry, "__wrapped__", entry)
        for build in builders:
            for bogus in _MALFORMED[kind]:
                token = formatters._SKEWED_FIELDS.set([])
                try:
                    out = _render_or_exception(fn, build(copy.deepcopy(bogus)))
                    recorded = list(formatters._SKEWED_FIELDS.get() or ())
                finally:
                    formatters._SKEWED_FIELDS.reset(token)
                assert not isinstance(out, BaseException), (
                    f"{label} raised {type(out).__name__} on {bogus!r}: {out}")
                assert key in recorded, (
                    f"{label} did not record a skew for {bogus!r}. Either the "
                    "read no longer goes through the choke point, or this "
                    f"entry's payload stopped reaching it -- got {recorded}")
            # Populated, genuinely EMPTY, and an explicit null -- the three
            # states the choke point distinguishes, none of them a skew.
            for clean in (_PROBE_WELL_FORMED[kind],
                          [] if kind == "list" else {},
                          None):
                token = formatters._SKEWED_FIELDS.set([])
                try:
                    _render_or_exception(fn, build(copy.deepcopy(clean)))
                    recorded = list(formatters._SKEWED_FIELDS.get() or ())
                finally:
                    formatters._SKEWED_FIELDS.reset(token)
                assert key not in recorded, (
                    f"{label} cried skew for the well-formed {clean!r}")


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

    Measured by replaying this differential against the base module (same keys,
    same contexts, same `_MALFORMED` values, base's own discovered population
    -- 568 pairs, 199 of them containers; base's renderers are called bare
    because base has no disclosure boundary to wrap them in): base absorbs 904
    of its 1194 cases at 191 of those 199 container positions, and raises in
    130 more; this commit absorbs 0 and raises 0."""
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
            if _disclosed(out, key):
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
    # #857 r4: `_render_save_text` now reads the `collides_with_open_target` container and the session-start `loaded` rows read `attempted_path`, both discovered reads, so these derived populations grow with them. Measured.
    # 1212 -> 1218 (#797): `_render_defuse_text` now reads the `hints` list (the
    # #489 call-model-truncation disclosure), 1 list position x 6 malformed
    # container shapes, measured.
    assert checked == 1218, f"the differential ran {checked} cases, not 1218"


def test_no_renderer_raises_on_a_field_the_absent_payload_survived():
    """The soft-degrade half of #619, kind-free, so it covers all 610 read keys
    rather than the 201 the container probe classifies as containers: a renderer
    that renders an absent field cleanly and DIES on a present wrong-shaped one
    has regressed to the crash this change replaced.

    Base, swept the same way over its own population, raises 166 times across
    67 (renderer, key) positions in 4544 renders; this commit raises 0 in 4880.
    (#822 taught `_render_virtual_call_text` to render the typed
    `unresolved_reason_code`, which is the renderer read that takes this
    population from its base's 4840 to 4848 -- the population follows the
    renderers, which is the point of deriving it. #820's quick-load warning adds
    the four scalar `partial` reads that take it from 4848 to 4880 the same way.)
    Two of those renderers
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
    assert not raised, raised[:8]
    # Last, so a real raise reports itself instead of being masked by the count
    # it also moves (the round-8 rule, applied to the sweeps too).
    # 4848 -> 4880 (#820): four more `(renderer, ctx)` pairs, because
    # `_render_type_list_text`, `_render_function_evidence_text` and
    # `_render_class_list_text` now READ `partial` (the quick-load warning), and a
    # discovered read is a discovered pair -- 4 pairs x 8 bogus values, MEASURED
    # on the rebased tree rather than carried over from the pre-merge branch. The
    # assertion still exists to catch the population SHRINKING silently.
    #
    # 4880 -> 4888 (#755): ONE more pair, because `_render_trace_text` now reads
    # the top-level `callee` to tell an unresolved callee from a callee nothing
    # was computed for -- 1 pair x 8 bogus values, measured the same way.
    #
    # #857 r4: `_render_save_text` now reads the `collides_with_open_target` container and the session-start `loaded` rows read `attempted_path`, both discovered reads, so these derived populations grow with them. Measured on the rebased tree as the sum of BOTH contributions -- neither branch's own number survives the merge (#857 r8 rebase). 4880 + 8 (#755) + 8 (#857 r4) = 4896.
    # 4920 -> 4928 (#795): `filtered` is a read `_render_strings_text` did not
    # make (1 pair x 8 bogus values), measured the same way.
    # 4896 -> 4920 (#770): `_render_class_list_text` now delegates its paging to
    # the SHARED `_paging_footer`, which reads `returned`, `offset` and
    # `has_more` -- three keys that renderer did not ask for while it built its
    # own footer (it already read `total`). 3 pairs x 8 bogus values = 24,
    # MEASURED by diffing `_runtime_population()` rather than carried over.
    # 4928 -> 4936 (#797): `hints` is one more discovered read on
    # `_render_defuse_text` (1 pair x 8 bogus values), measured.
    # 4936 -> 4944 (#795 round-6 review): the class listing's non-class artifact
    # share was read `or 0` inside its own conditional and was therefore
    # discovered by nothing; routing it through `_nonnegative_count` makes it
    # ONE more discovered read (1 pair x 8 bogus values), measured by diffing
    # `_runtime_population()`. The listing's other five newly-choked numbers
    # were already discovered reads, so they add nothing here.
    assert swept == 4944, f"the raise sweep ran {swept} renders, not 4944"


def test_the_nested_population_converges_before_the_depth_cap():
    """The nested descent must exhaust its frontier before the runaway cap."""
    deepest = max(len(path) for _, _, path, _, _, _, _ in _nested_population())
    # THE convergence proof, and the answer to round 10's second blocker: the
    # descent stopped because a level found no further container, not because it
    # hit the cap. A cap the data reaches would be a live bypass one level down.
    assert deepest < _NEST_DEPTH_CAP, (
        f"the nested descent reached the runaway cap ({deepest} of "
        f"{_NEST_DEPTH_CAP}), so it was CUT OFF rather than converging, and a "
        "read below it is outside every differential")


def test_every_NESTED_population_context_actually_reaches_the_read_it_was_recorded_for():
    """The nested population's third promise, made executable -- the same guard
    the top level has, which the nested half went five rounds without.

    It is not a formality: the nested sweeps REBUILD their leaf from what the
    population recorded, and rebuilding it from a bare probe element left 187 of
    1333 rows swept in a payload that never entered the read. A read gated on a
    SIBLING of its own key (`if args:` before the argument tag, a `value` beside
    the `type`) could not fail there, and three live raises sat behind exactly
    that -- found by replaying the identical sweep with the siblings the descent
    had already discovered.

    So every recorded (path, key, leaf) triple is re-run and the renderer must
    ACTUALLY ASK for that key in that leaf, which is the property the sweeps
    depend on and the one nobody could see was broken."""
    unreached = []
    for label, render, path, key, kind, ctx, leaf in _nested_population():
        asked: set[str] = set()
        _render_or_exception(render, _payload_for(
            ctx, path, _KeyProbe({**copy.deepcopy(leaf), key: "probe"}, asked)))
        if key in asked:
            continue
        # A container position may only be read when a container is what sits
        # there, exactly as at top level: give it the kind it was classified as
        # before calling the context unreachable.
        probe = _KeyProbe(copy.deepcopy(leaf), asked)
        probe[key] = _watched(kind or "list")
        _render_or_exception(render, _payload_for(ctx, path, probe))
        if key not in asked:
            where = ".".join(k for k, *_ in path)
            unreached.append(f"{label}.{where}[].{key}")
    assert not unreached, (
        f"{len(unreached)} nested population entries record a leaf the renderer "
        "never reads the key in, so every nested sweep runs a branch it does "
        f"not enter and cannot fail there: {unreached[:8]}")


def test_a_nested_container_is_never_absorbed_into_the_empty_rendering():
    """THE differential, at the positions the top-level probe cannot reach: a key
    of a callee ROW, a per-block `insns`, a ref bucket inside a match row.

    Same property as at top level -- a container that is PRESENT but holds the
    wrong shape must never render byte-identically to that field being ABSENT or
    EMPTY -- and the same reason: the caller reads a confident "nothing here" out
    of a payload the renderer could not use.

    A one-hop helper coercing a nested ref bucket must not let a falsy wrong
    shape render byte-identically to the key being absent."""
    from bn import formatters

    echoes = formatters._render_fallback_text
    absorbed = []
    for fn_name, render, path, key, kind, ctx, leaf in _nested_population():
        if kind is None:
            continue
        # The baseline drops the key from the ELEMENT the read was OBSERVED in,
        # so "absent" really is absent even for a key the probe element carries
        # by default, and the siblings that OPEN the read are still there.
        base = {k: v for k, v in leaf.items() if k != key}
        absent = _render_or_exception(render, _payload_for(ctx, path, dict(base)))
        empty = _render_or_exception(
            render, _payload_for(ctx, path, {**base, key: [] if kind == "list" else {}}))
        for bogus in _MALFORMED[kind]:
            payload = _payload_for(ctx, path, {**base, key: bogus})
            out = _render_or_exception(render, payload)
            if isinstance(out, Exception):
                continue                   # the nested raise sweep owns this case
            if _disclosed(out, key):
                continue
            if out == echoes(payload) or echoes(payload) in out:
                continue
            if out == absent or out == empty:
                where = ".".join(k for k, *_ in path)
                absorbed.append(
                    f"{fn_name}({where}[].{key}={bogus!r}) renders byte-identically "
                    f"to that nested field being "
                    f"{'absent' if out == absent else 'empty'}, with no disclosure")
    assert not absorbed, absorbed[:8]


def test_no_renderer_raises_on_a_nested_field_the_absent_payload_survived():
    """The soft-degrade half of #619 at nested positions, kind-free: a renderer
    that renders a nested field's absence cleanly and DIES on a present
    wrong-shaped one has regressed to the crash this change replaced.

    Kind-free because the crash does not need a container: a sampled string
    sliced as `(s.get("value") or "")[:80]`, a block index in a `:<4` format
    spec, an unhashable `kind` used as a grouping key."""
    raised = []
    for fn_name, render, path, key, _kind, ctx, leaf in _nested_population():
        base = {k: v for k, v in leaf.items() if k != key}
        absent = _render_or_exception(render, _payload_for(ctx, path, dict(base)))
        for bogus in ("bad", {"a": 1}, ["bad"], 0, "", False, {}, []):
            out = _render_or_exception(render, _payload_for(ctx, path, {**base, key: bogus}))
            if isinstance(out, Exception) and not isinstance(absent, Exception):
                where = ".".join(k for k, *_ in path)
                raised.append(f"{fn_name}({where}[].{key}={bogus!r}) raised "
                              f"{type(out).__name__} where the absent payload "
                              "rendered cleanly")
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
    # 1413 -> 1421 with the same four pairs the raise sweep gained (#820): the
    # quick-load warning reads `partial` in three more renderers, and the mirror
    # contributes 2 benign payloads per pair. Measured on the rebased tree.
    # 1421 -> 1423 (#755): the one `_render_trace_text`/`callee` pair the raise
    # sweep also gained, x 2 benign payloads.
    #
    # #857 r4: `_render_save_text` now reads the `collides_with_open_target` container and the session-start `loaded` rows read `attempted_path`, both discovered reads, so these derived populations grow with them. Measured on the rebased tree as the sum of BOTH contributions (#857 r8 rebase). 1421 + 2 (#755) + 3 (#857 r4) = 1426.
    # 1426 -> 1432 (#770): the same three `_render_class_list_text` pairs above,
    # x 2 benign payloads each. Measured, not carried over.
    # 1432 -> 1434 (#795): the one `_render_strings_text`/`filtered` pair, x 2.
    # 1434 -> 1437 (#797): the one `_render_defuse_text`/`hints` pair -- a LIST,
    # so its benign half is 3 payloads (None/[]/{}), measured.
    # 1437 -> 1439 (#795 round-6 review): the one new `_render_class_list_text`
    # /`artifact_count` pair the raise sweep also gained, x 2 benign payloads.
    assert checked == 1439, f"the mirror ran {checked} renders, not 1439"
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


def test_xrefs_text_says_so_when_the_caller_scan_was_truncated():
    """#622 put `truncated` + `scan_note` on the xrefs envelope when the
    budgeted LLIL caller scan stopped early, and only the JSON envelope carried
    them: a capped page rendered byte-identically to a complete one, so an empty
    caller list read as proof of no callers."""
    from bn.formatters import _render_xrefs_text
    out = _render_xrefs_text({
        "address": "0x401000", "code_refs": [], "data_refs": [],
        "code_ref_count": 0, "data_ref_count": 0,
        "truncated": True, "scan_note": "scan stopped at its budget",
    })
    assert "TRUNCATED" in out, out
    assert "scan stopped at its budget" in out, out
    assert "NOT proof there are no callers" in out, out

    # A complete page makes no such claim.
    clean = _render_xrefs_text({"address": "0x401000", "code_refs": [],
                                "data_refs": [], "code_ref_count": 0,
                                "data_ref_count": 0})
    assert "TRUNCATED" not in clean, clean


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


def test_the_class_listing_reads_its_count_and_every_cardinality_through_the_choke_point_795():
    """One count contract on the class lens's own numbers (#795 round-6 review).

    Besides its rows this renderer states five numbers: the count-only
    headline, the non-class artifact share beside it, and the three suppressed
    shares in the `hidden:` tail. Every one was read raw, so the class lens
    described a payload exactly the way the imports trio did before the
    round-3/4/5 repairs -- a flag as a quantity, a container as a Python repr,
    a text-spelled count silently dropping its qualifier -- and on the four
    keys that are CARDINALITIES (how many rows the lens folded OUT) it stated
    an impossible negative at rc 0 with nothing disclosed. Same payload, two
    descriptions, depending on which surface the caller hit.
    """
    from bn.formatters import _render_class_list_text as render

    # (a) The HEADLINE states the count, so it follows `_stated_count`: a bool
    # is not a quantity, a numeric string states the line the integer states,
    # and a container is disclosed instead of interpolated as a repr.
    assert render({"count": 3}) == "classes: 3"
    assert render({"count": "3"}) == render({"count": 3})
    flagged = render({"count": True})
    assert "classes: True" not in flagged and "malformed count field" in flagged
    boxed = render({"count": {"n": 1}})
    assert "{" not in boxed and "}" not in boxed and "malformed count field" in boxed

    # (b) The four CARDINALITIES cannot be negative -- a survey cannot have
    # folded out -2 rows -- and are refused the way the imports trio refuses
    # it: the share is not restated as a quantity and the skew reaches the
    # `@_discloses` boundary note.
    for key, rest in (("artifact_count", {"count": 3}),
                      ("construction_vtables_suppressed", {"items": []}),
                      ("thunks_suppressed", {"items": []}),
                      ("library_suppressed", {"items": [], "no_stl": True}),
                      ("vendor_suppressed", {"items": [], "no_vendor": True})):
        negative = render({**rest, key: -2})
        assert "-2" not in negative, (key, negative)
        assert f"malformed {key} field" in negative, (key, negative)
        flag = render({**rest, key: True})
        assert "True" not in flag, (key, flag)
        assert f"malformed {key} field" in flag, (key, flag)
        # A text-spelled share states the same line the integer spelling does.
        assert render({**rest, key: "2"}) == render({**rest, key: 2}), key
        # ...and the share is STATED as unknown, not dropped. A dropped share
        # renders byte-identically to a survey that folded out nothing, so the
        # reader has already decided by the time the boundary note arrives
        # (#619/#683) -- the harm the trailing note alone cannot repair.
        body = negative.split("\n! malformed")[0]
        assert body != render({**rest, key: 0}), (key, body)
        assert "?" in body, (key, body)

    # (c) The per-ROW method count is a stated number too, and it was the one
    # number in this renderer the round-6 repair walked past -- in the very
    # renderer that repair was filed against.
    def row(method_count):
        return render({"items": [{"name": "Probe", "method_count": method_count}],
                       "total": 1})

    assert "methods=3" in row(3) and row("3") == row(3)
    assert "methods=True" not in row(True), row(True)
    assert "malformed method_count field" in row(True), row(True)
    boxed_row = row({"n": 1})
    assert "{" not in boxed_row and "}" not in boxed_row, boxed_row

    # (d) A share the run never asked to fold out stays silent even when the
    # payload spells it wrong. Reading it BEFORE the gate that decides whether
    # to state it recorded a skew on a run with nothing in the `hidden:` tail
    # to act on, so the reader got a malformed-field note about a number the
    # command was never going to print.
    for gate, key in (("no_stl", "library_suppressed"),
                      ("no_vendor", "vendor_suppressed")):
        ungated = render({"items": [], key: {"n": 1}})
        assert "malformed" not in ungated, (key, ungated)
        assert ungated == render({"items": [], key: 7}), (key, ungated)
        # ...and with the gate on, the same payload IS disclosed.
        assert "?" in render({"items": [], gate: True, key: {"n": 1}}), key


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


def test_render_evidence_text_states_the_decompile_deferral():
    """A sliced evidence read skips the Pseudo-C decompile, and the bridge says
    so twice -- a sentence in `warnings` and a top-level `decompile_deferred`
    flag. TEXT mode printed NEITHER (it dropped the whole `warnings` list), so a
    sliced card read like a full-fidelity one."""
    from bn.formatters import _render_function_evidence_text
    base = {
        "function": {"name": "parse_line", "address": "0x500000"},
        "prototype": "void parse_line()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 0, "matched_calls": 0, "offset": 0, "limit": None,
        "calls": [],
    }

    # The sentence the bridge wrote is what gets printed ...
    out = _render_function_evidence_text({
        **base, "decompile_deferred": True,
        "warnings": ["Pseudo-C decompile deferred for this sliced read"],
    })
    assert "warning: Pseudo-C decompile deferred for this sliced read" in out, out
    # ... once, not alongside the fallback restating the same claim.
    assert out.count("deferred") == 1, out

    # ... and the flag alone still states it, for a payload whose `warnings`
    # arrived absent, empty or skewed.
    flag_only = _render_function_evidence_text({**base, "decompile_deferred": True})
    assert "decompile deferred" in flag_only, flag_only
    assert "re-read unsliced" in flag_only, flag_only

    # A full-fidelity read claims nothing.
    assert "deferred" not in _render_function_evidence_text(base)


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


def test_render_evidence_function_states_the_library_contradiction_759():
    """#862 review: `prototype_unverified` is the THIRD demotion cause, and the
    only one that rendered no reason -- the row showed a bare
    `arguments: (hlil inferred)` on the very surface the issue quotes, which is
    the silent demotion this issue family exists to stop. Its siblings
    (`arity_unknown`, `arity_mismatch`, `callee_unresolved`) each explain
    themselves here."""
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "decode_block", "address": "0x401800"},
        "prototype": "void decode_block()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [{
            "address": "0x401800", "operation": "LLIL_CALL", "direct": True,
            "argument_source": "hlil", "argument_confidence": "inferred",
            "arguments": [], "argument_candidates": [],
            "arity_unknown": False, "prototype_unverified": True,
            "declared_arity": 0, "library_arity": 1,
            "library_source": "libgcc_s_x86_64.so.1",
        }],
    }
    out = _render_function_evidence_text(value)
    assert "arity: UNVERIFIED" in out
    # The two counts and the library that supplied the contradiction, so the
    # reader can check it rather than take the demotion on trust.
    assert "declares 0 but libgcc_s_x86_64.so.1 declares 1" in out
    # The precedence PROMISE is gone (the dogfood showed suppressing on
    # has_user_type disabled the check on every saved database), so the line now
    # states what it can defend: the contradiction is reported, not judged.
    # The card must not tell the reader the disagreement is harmless when it IS
    # a demotion trigger (#862 review round 6), nor claim it is the ONLY one:
    # `prototype_unverified` can co-occur with `arity_mismatch`/`arity_unknown`,
    # whose own `arity:` line prints directly above (round 7).
    assert "one reason this row is not fully corroborated" in out
    assert "any other `arity:` line above names another" in out
    # Neither overclaim may come back: not "the whole basis", and not a named
    # confidence the row may not even have -- an MLIL/LLIL-sourced list is
    # `heuristic`, never `inferred` (round 7).
    assert "the whole basis" not in out
    assert "this row's `inferred` confidence" not in out
    assert "this row disagrees with your statement" in out
    assert "takes precedence" not in out
    assert "is not evidence against it" not in out


def test_render_evidence_function_library_contradiction_without_counts():
    """The same row with unusable counts still states the cause: a disclosure
    that can only be rendered when every field is well formed is a disclosure
    that vanishes exactly when the payload is degraded."""
    from bn.formatters import _render_function_evidence_text
    value = {
        "function": {"name": "decode_block", "address": "0x401800"},
        "prototype": "void decode_block()", "calling_convention": "__cdecl",
        "thunk": {"is_candidate": False},
        "total_calls": 1, "matched_calls": 1, "offset": 0, "limit": None,
        "calls": [{
            "address": "0x401800", "operation": "LLIL_CALL", "direct": True,
            "argument_source": "hlil", "argument_confidence": "inferred",
            "arguments": [], "argument_candidates": [],
            "prototype_unverified": True, "declared_arity": "bad",
        }],
    }
    out = _render_function_evidence_text(value)
    assert "arity: UNVERIFIED" in out
    assert "disagrees with an attached type library" in out


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


def test_render_orient_states_the_analyst_split_without_fabricating_it():
    """#733 F2: the split prints when reported, is absent when not, and a
    half-reported pair prints `?` rather than a bare `None` or a confident 0 --
    the same rule every other stated count in this module follows."""
    from bn.formatters import _render_orient_text

    base = {"kind": "orient_digest", "target": {"basename": "netsvcd"},
            "analyzed": True, "analysis_state": "full", "function_count": 10}

    full = _render_orient_text({**base, "existing_annotations": {
        "comments": 0, "function_comments": 0, "user_symbols": 612,
        "analyst_symbols": 0, "placeholder_symbols": 612,
        "analysis_cache_restored": False}})
    assert "analyst-symbols=0, placeholders=612" in full

    older = _render_orient_text({**base, "existing_annotations": {
        "comments": 0, "function_comments": 0, "user_symbols": 540,
        "analysis_cache_restored": False}})
    assert "analyst-symbols" not in older
    assert "user-symbols=540" in older

    half = _render_orient_text({**base, "existing_annotations": {
        "comments": 0, "function_comments": 0, "user_symbols": 612,
        "analyst_symbols": 3, "analysis_cache_restored": False}})
    # The sibling was never claimed, so it is omitted -- not printed as a
    # confident `placeholders=0`, and not as a bare `None`.
    assert "analyst-symbols=3," in half and "placeholders" not in half

    unreadable_analyst = _render_orient_text({**base, "existing_annotations": {
        "comments": 0, "function_comments": 0, "user_symbols": 612,
        "analyst_symbols": "many", "placeholder_symbols": 612,
        "analysis_cache_restored": False}})
    assert "analyst-symbols=?" in unreadable_analyst

    unreadable = _render_orient_text({**base, "existing_annotations": {
        "comments": 0, "function_comments": 0, "user_symbols": 612,
        "analyst_symbols": 3, "placeholder_symbols": "many",
        "analysis_cache_restored": False}})
    assert "placeholders=?" in unreadable


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


def test_render_trace_text_header_discloses_an_unresolved_callee_755():
    """#755: the header used to OMIT the callee when the payload said it did not
    resolve, so the output claimed nothing about which call it answered -- two
    indirect calls in one function then rendered headers differing only by
    address, and an analyst who copied a nearby address got an equally confident
    slice about a different call with no signal at all.

    The callee slot is never silent when the producer computed one; the register
    stays optional."""
    from bn.formatters import _render_trace_text
    value = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {"index": 0}, "trace": [],
    }
    out = _render_trace_text(value)
    assert "backward trace of arg[0] of <unresolved callee> in f @ 0x1010" in out
    # No register in the label, so the header must not invent one: the arg
    # descriptor ends at the callee and carries no parenthesised register.
    assert "<unresolved callee> in f" in out and "(" not in out.splitlines()[0]
    resolved = _render_trace_text(dict(value, arg_label={"index": 0, "callee": "memcpy"}))
    assert "backward trace of arg[0] of memcpy in f @ 0x1010" in resolved
    assert "(" not in resolved.splitlines()[0]


def test_render_trace_text_makes_no_callee_claim_when_none_was_computed_755():
    """#755 review: the disclosure must not become its own absent-vs-null
    conflation. Only an `arg_label` that arrived as an OBJECT means the producer
    computed a callee slot; a missing key and an explicit null both claim
    nothing, which is `_field_present`'s stated rule, how `_field_list` and
    `_field_dict` already read a nulled field, and the reading the mirror test
    relies on when it feeds `{key: None}` as a benign payload for every
    discovered read.

    Round 1 used `_field_declared` -- whose docstring says it answers the
    DIFFERENT question of which envelope shape arrived -- so an explicit
    `"arg_label": null` still rendered the affirmative finding. All three states
    are pinned here so neither direction can drift again."""
    from bn.formatters import _render_trace_text
    bare = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "trace": [],
    }

    # 1. Absent: nothing was computed, so nothing is claimed.
    out = _render_trace_text(bare)
    assert "backward trace of arg[0] in f @ 0x1010" in out
    assert "unresolved callee" not in out

    # 2. Explicit null, on either key: still claims nothing (round-2 finding).
    for nulled in ({"arg_label": None}, {"callee": None},
                   {"arg_label": None, "callee": None}):
        nulled_out = _render_trace_text({**bare, **nulled})
        assert "backward trace of arg[0] in f @ 0x1010" in nulled_out, nulled
        assert "unresolved callee" not in nulled_out, nulled

    # 3. An arg_label OBJECT is the positive signal: computed, and if its callee
    #    is missing or null the row says so rather than going silent.
    for computed in ({"index": 0}, {"index": 0, "callee": None}):
        assert "of <unresolved callee>" in _render_trace_text(
            dict(bare, arg_label=computed)), computed

    # A name still wins from either key.
    assert "of parse_header" in _render_trace_text(
        dict(bare, arg_label={"index": 0, "callee": "parse_header"}))
    assert "of parse_header" in _render_trace_text(dict(bare, callee="parse_header"))


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
    # `first_error` is the one key an agent contract tells callers to read, so
    # it is TEXT or nothing on both callers -- never a container whose Python
    # repr the compact renderer prints. The explanation goes through the text
    # choke point, so an unreadable one means "this row explained nothing" and
    # the next fallback answers, with the boundary naming the field.
    for unreadable in ({"code": 7}, ["boom"], 7, 1.5, ()):
        generic = formatters._mutation_summary({
            "success": False, "committed": False,
            "results": [{"op": "rename", "status": "verification_failed",
                         "message": unreadable}]})
        go = formatters._go_rename_summary({
            "kind": "go_rename", "success": False, "committed": False,
            "go_renamed_candidates": 1, "go_committed_count": 0,
            "go_verified_count": 0, "go_failed_count": 1,
            "skipped_user_named": 0, "message": unreadable,
            "results": [{"status": "verification_failed"}]})
        for name, summary in (("mutation", generic), ("go rename", go)):
            assert isinstance(summary["first_error"], str), (
                f"{name} summary put a {type(summary['first_error']).__name__} in "
                f"first_error for message={unreadable!r}: {summary!r}")
            assert "malformed message" in summary["first_error"], (
                f"{name} summary dropped an unreadable message with no note "
                f"naming it: {summary!r}")
        # And the message still READS when it is text, on both.
        assert formatters._mutation_summary({
            "success": False, "committed": False, "message": "revert failed",
            "results": [{"status": "verified"}]})["first_error"] == "revert failed"


def test_the_compact_mutation_status_discloses_a_malformed_results_listing():
    """The DEFAULT mutation text path, and the one place a choke-point read has
    no renderer to record into.

    The CLI runs the compact status as a `result_transform` and hands the text
    renderer its OUTPUT, so `@_discloses` installs its recorder after the raw
    payload is already gone. A present-but-malformed `results[]` therefore
    produced a status byte-identical to one built from no `results[]` at all, on
    every mutation subcommand's default view: a confident unmeasured verdict
    from a payload the code could not read, with nothing saying so. Both
    transforms, all six wrong shapes -- including the FALSY ones, which is where
    "present" and "absent" collapse into each other."""
    from bn import formatters

    def rendered(transform, payload):
        return formatters._render_mutation_summary_text(transform(payload))

    cases = [
        (formatters._mutation_summary, {"success": True, "committed": True}),
        (formatters._go_rename_summary, {"kind": "go_rename", "success": False,
                                         "go_failed_count": 2,
                                         "go_renamed_candidates": 5}),
    ]
    for transform, base in cases:
        absent = rendered(transform, dict(base))
        assert "malformed" not in absent, (
            f"{transform.__name__} cries malformed on a payload that simply has "
            f"no results[]: {absent!r}")
        for bogus in _MALFORMED["list"]:
            out = rendered(transform, {**base, "results": bogus})
            assert out != absent, (
                f"{transform.__name__} renders results={bogus!r} byte-identically "
                "to that key being absent, with no disclosure -- an unusable "
                "payload reading as a confident status")
            assert _disclosed(out, "results"), (
                f"{transform.__name__} on results={bogus!r} discloses nothing "
                f"that NAMES the key it could not read: {out!r}")


def test_the_verbose_mutation_transform_never_claims_ok_from_unreadable_rows():
    """`ok` is "the bridge reported success AND no op row failed" (#447), and the
    second half is derived from `results[]`.

    This transform also runs before any renderer, so a present-but-malformed
    listing left it reading ZERO failures and answering a confident `ok: true` --
    the JSON half of the defect the text path discloses above, and the key an
    agent contract tells a control loop to close on. With the rows unreadable
    the second half is not established, so it must fail safe the way
    `dirty_after` already does."""
    from bn import formatters

    ok_of = lambda payload: formatters._add_mutation_ok(payload)["ok"]
    # Unchanged where the rows ARE readable: absent, empty, clean, and failed.
    assert ok_of({"success": True}) is True
    assert ok_of({"success": True, "results": []}) is True
    assert ok_of({"success": True, "results": [{"status": "verified"}]}) is True
    assert ok_of({"success": True,
                  "results": [{"status": "verification_failed"}]}) is False
    for bogus in _MALFORMED["list"]:
        assert ok_of({"success": True, "results": bogus}) is False, (
            f"ok claimed success from results={bogus!r}, which no op row could be "
            "read out of -- a fabricated all-clear for a batch nobody checked")


def test_the_go_rename_counters_degrade_instead_of_costing_the_whole_summary():
    """`go rename` reports through its OWN counters, and every one of the six
    reached a bare `int(...)`: a counter arriving as a string or a container
    raised ValueError/TypeError and cost the ENTIRE summary, where the same
    payload with that counter ABSENT rendered cleanly. Quietly reading it as 0
    is the other half of the same bug -- `changed=0 ... dirty_after=False` is
    exactly the "nothing happened, don't save" verdict that discards a completed
    rename batch -- so `_count_field` records the skew and the summary discloses
    it by name.

    Named as well as swept, and the STATED LIMIT it used to carry is gone: the
    probe recorded an unclassified key's context as the bare payload, which
    delegates to the generic summary and never opens `kind == "go_rename"`, so
    these six reads were outside every sweep. The population records the
    context the read was OBSERVED in now, and
    `test_every_population_context_actually_reaches_the_read_it_was_recorded_for`
    asserts that of every entry -- so the aggregate sweep covers these too, and
    this test's job is the per-counter disclosure the aggregate cannot state."""
    from bn import formatters

    base = {"kind": "go_rename", "success": True, "committed": True}
    absent = formatters._render_mutation_summary_text(
        formatters._go_rename_summary(dict(base)))
    assert "malformed" not in absent, absent
    for counter in ("go_renamed_candidates", "go_committed_count",
                    "go_verified_count", "go_failed_count", "skipped_user_named",
                    "skipped_changed_during_apply"):
        for bogus in ("bad", {"a": 1}, ["x"], "", False, {}, []):
            out = formatters._render_mutation_summary_text(
                formatters._go_rename_summary({**base, counter: bogus}))
            assert out != absent, (
                f"{counter}={bogus!r} renders byte-identically to that counter "
                "being absent")
            assert _disclosed(out, counter), (
                f"{counter}={bogus!r} left no note naming the counter that could "
                f"not be read: {out!r}")
        # A real count, and the numeric string the bridge has always been allowed
        # to send, still read as themselves and cry nothing.
        for genuine in (3, "3"):
            out = formatters._render_mutation_summary_text(
                formatters._go_rename_summary({**base, counter: genuine}))
            assert "malformed" not in out, (
                f"{counter}={genuine!r} is a readable count and must not disclose: "
                f"{out!r}")


def test_an_unreadable_go_rename_counter_can_never_read_as_a_finished_run():
    """`measured` must be DERIVED from the reads, never asserted.

    `go rename` is the op whose entire reason for existing is that a fabricated
    `changed=0` discarded a completed rename batch (#683) -- and `measured=True`
    was hardcoded here, on the grounds that this function measures through its
    own counters "by design". That made the shared builder's unmeasured
    fail-safe unreachable from the one path that needs it most: an unreadable
    commit counter fabricated `changed_count: 0` and flipped `dirty_after` to
    `False`, so every DECISION key matched a genuine all-noop run byte for byte
    on the default text path.

    Disclosure is not enough here and that is the whole point. Every other
    finding on this file is "an unusable payload renders as clean"; this one is
    "an unusable payload renders as an actionable instruction to throw work
    away". A note in `first_error` does not reach a control loop that branches
    on `dirty_after` -- `if not summary["dirty_after"]: close()`, the idiom the
    builder's own #684 comment enumerates -- so the DECISION KEYS themselves
    must differ from a real nothing-changed run.
    """
    from bn import formatters

    decision = lambda s: tuple(s[k] for k in ("dirty_after", "changed_count",
                                              "verified_count", "noop_count",
                                              "failed_count", "measured"))
    base = {"kind": "go_rename", "success": True, "committed": True}
    # A REAL all-noop commit: every counter readable and zero. This is the
    # verdict a skewed run must never be able to impersonate.
    noop = formatters._go_rename_summary({**base, "go_renamed_candidates": 0,
                                          "go_committed_count": 0,
                                          "go_verified_count": 0,
                                          "go_failed_count": 0,
                                          "skipped_user_named": 0})
    assert decision(noop) == (False, 0, 0, 0, 0, True), noop
    for counter in ("go_renamed_candidates", "go_committed_count",
                    "go_verified_count", "go_failed_count", "skipped_user_named",
                    "skipped_changed_during_apply"):
        for bogus in ("bad", "1783 renamed", {"a": 1}, ["x"], "", False, {}, []):
            skewed = formatters._go_rename_summary(
                {**base, "go_renamed_candidates": 1783, "go_verified_count": 1783,
                 counter: bogus})
            assert skewed["measured"] is False, (
                f"{counter}={bogus!r} could not be read, and the summary still "
                f"claims measured=True: {skewed!r}")
            assert skewed["dirty_after"] is True, (
                f"{counter}={bogus!r} left dirty_after={skewed['dirty_after']!r} "
                "-- a control loop branching on it discards the batch")
            for count in ("changed_count", "verified_count", "noop_count",
                          "failed_count"):
                assert skewed[count] is None, (
                    f"{counter}={bogus!r} fabricated {count}="
                    f"{skewed[count]!r} out of a payload it could not read")
            assert decision(skewed) != decision(noop), (
                f"{counter}={bogus!r} produces the decision keys of a genuine "
                f"all-noop run: {decision(skewed)}")
        # Anti-vacuity, both directions: `measured` has to be able to be True,
        # or a hardcoded False would satisfy every assertion above. A real
        # count and the numeric string the bridge has always been allowed to
        # send both stay measured, with the count read as itself. The
        # measurement source for a COMMITTED run (`go_committed_count`) is
        # supplied here because its ABSENCE is itself unmeasured -- see
        # `test_a_go_rename_summary_with_no_counters_is_not_a_measured_noop`.
        # `go_failed_count` is the one counter the payload also carries the
        # EVIDENCE for: the bridge builds it as `len(failed_rows)` beside
        # `"results": failed_rows`, so a readable 3 with no failure rows is a
        # self-contradicting envelope rather than a readable measurement, and
        # is refused -- the refusal itself is pinned in
        # `test_the_go_rename_status_withholds_ok_on_rows_it_could_not_read`.
        for genuine, expected in ((3, 3), ("3", 3)):
            payload = {**base, "go_committed_count": 7, counter: genuine}
            if counter == "go_failed_count":
                payload["results"] = [{"status": "verification_failed"}] * 3
            read = formatters._go_rename_summary(payload)
            assert read["measured"] is True, (
                f"{counter}={genuine!r} is a readable count and the summary "
                f"reports it unmeasured: {read!r}")
            if counter == "go_committed_count":
                assert read["changed_count"] == expected, read


def test_a_go_rename_summary_with_no_counters_is_not_a_measured_noop():
    """ABSENT is not zero, for a count exactly as for a container.

    `_count_field` answers 0 for a key that was never there -- and on this op a
    0 IS the "nothing changed, do not save" verdict. So a partial or
    version-skewed envelope that carried `kind: go_rename` and none of its
    counters produced `changed=0 / dirty_after=False / measured=True`,
    byte-identical in every decision key to a genuine all-noop commit: #683's
    harm reached by ABSENCE rather than by a wrong shape. `_mutation_summary`
    has always called its own missing measurement source unmeasured; the two
    callers of the one shared builder must answer that question the same way.

    The source required is the counter `changed` is actually READ FROM, which
    differs by branch -- so all three branches are checked, including the
    reverted one, where "nothing landed" is established by the revert and needs
    no counter at all."""
    from bn import formatters

    committed = formatters._go_rename_summary(
        {"kind": "go_rename", "success": True, "committed": True})
    assert committed["measured"] is False, committed
    assert committed["dirty_after"] is True, committed
    assert committed["changed_count"] is None, committed
    # A REAL all-noop states its zero, and stays measured.
    real = formatters._go_rename_summary(
        {"kind": "go_rename", "success": True, "committed": True,
         "go_committed_count": 0})
    assert real["measured"] is True and real["dirty_after"] is False, real
    # Preview measures through `go_verified_count`, so that is its source.
    preview = formatters._go_rename_summary(
        {"kind": "go_rename", "success": True, "preview": True})
    assert preview["measured"] is False, preview
    assert formatters._go_rename_summary(
        {"kind": "go_rename", "success": True, "preview": True,
         "go_verified_count": 0})["measured"] is True
    # A live run that was reverted: nothing landed, established by the revert.
    reverted = formatters._go_rename_summary(
        {"kind": "go_rename", "success": False, "committed": False,
         "rolled_back": True})
    assert reverted["measured"] is True, reverted
    assert reverted["changed_count"] == 0, reverted


def test_the_compact_summary_withholds_ok_on_a_payload_it_could_not_read():
    """One answer to "can success be claimed off a payload we could not read".

    `_add_mutation_ok` withholds `ok` when `results[]` is unreadable, because
    "no op row failed" is not established. The compact summary claimed
    `ok: true` on the SAME payload, so a uniform `jq '.ok'` -- the entire point
    of #447 -- flipped depending on whether `--summary` was passed. Both paths
    are checked here against the same inputs so they cannot drift apart again.

    Merely EMPTY rows are NOT unusable, and both paths still report ok there:
    that boundary is what makes this parity rather than a behaviour change."""
    from bn import formatters

    for bogus in _MALFORMED["list"]:
        payload = {"success": True, "committed": True, "results": bogus}
        summary = formatters._mutation_summary(dict(payload))
        verbose = formatters._add_mutation_ok(dict(payload))
        assert summary["ok"] is False and summary["success"] is False, (
            f"the compact summary claimed success from results={bogus!r}: {summary!r}")
        assert verbose["ok"] is False, verbose
        assert summary["ok"] is verbose["ok"], (
            f"`jq '.ok'` flips with --summary on results={bogus!r}: "
            f"{summary['ok']!r} vs {verbose['ok']!r}")
    # The boundary: absent and empty rows are readable, and both paths agree.
    for rows in ({}, {"results": []}):
        payload = {"success": True, "committed": True, **rows}
        assert formatters._mutation_summary(dict(payload))["ok"] is True
        assert formatters._add_mutation_ok(dict(payload))["ok"] is True


def test_the_unmeasured_cause_round_trips_for_every_cause():
    """The renderer names the CAUSE, and there is one definition of the format.

    The cause cannot travel in its own summary key -- the key set IS the
    documented #685 contract -- so it rides in `first_error` and the renderer
    reads it back by splitting on the same template that wrote it. That is only
    safe if the round trip is asserted, for every cause the module can produce,
    including the "no note at all" case where the renderer must name nothing
    rather than guess."""
    from bn import formatters

    causes = {
        "this op reported no results[] rows":
            formatters._mutation_summary({"success": True, "committed": True}),
        "this op's own counters could not be read":
            formatters._go_rename_summary({"kind": "go_rename", "success": True,
                                           "committed": True,
                                           "go_committed_count": "bad"}),
        "this op reported none of its own counters":
            formatters._go_rename_summary({"kind": "go_rename", "success": True,
                                           "committed": True}),
        # The states this round added, each named for the field a reader would
        # go look at: a populated `results[]` one row of which could not be
        # read is NOT "no rows", and go rename's rows being unreadable is not
        # its counters being unreadable. The results[] phrase covers BOTH
        # granularities it can be unreadable at -- the whole field and one
        # element of it -- because naming a ROW for a field that arrived as a
        # string would send the reader looking for the wrong thing.
        "this op's results[] could not be read":
            formatters._mutation_summary({"success": True, "committed": True,
                                          "results": [7, {"status": "verified"}]}),
        "this op's failure rows could not be read":
            formatters._go_rename_summary({"kind": "go_rename", "success": True,
                                           "committed": True,
                                           "go_committed_count": 3,
                                           "results": ["not a row"]}),
        "this op's failure rows contradict its own counters":
            formatters._go_rename_summary({"kind": "go_rename", "success": True,
                                           "committed": True,
                                           "go_committed_count": 3,
                                           "go_failed_count": 0,
                                           "results": [{"status": "verification_failed"}]}),
        "an op row's status could not be read":
            formatters._mutation_summary({"success": True, "committed": True,
                                          "results": [{"op": "rename",
                                                       "status": {"code": 7}}]}),
    }
    for cause, summary in causes.items():
        assert summary["measured"] is False, summary
        assert formatters._unmeasured_cause(summary["first_error"]) == cause, (
            f"the cause did not round-trip out of first_error: "
            f"{summary['first_error']!r}")
        warning = [ln for ln in formatters._render_mutation_summary_text(summary)
                   .splitlines() if ln.startswith("warning: unmeasured")]
        assert warning and cause in warning[0], warning
    # The same phrase for the FIELD granularity, which is the half a
    # row-naming phrase got wrong: `results` arriving as a string carries no
    # rows to name.
    field = formatters._mutation_summary({"success": True, "committed": True,
                                          "results": "not a listing"})
    assert (formatters._unmeasured_cause(field["first_error"])
            == "this op's results[] could not be read"), field
    # The documented sample output in the public reference quotes this line
    # verbatim for the generic cause, so it is a contract, not wording.
    generic = formatters._render_mutation_summary_text(
        causes["this op reported no results[] rows"]).splitlines()[1]
    assert generic.startswith(
        "warning: unmeasured -- this op reported no results[] rows; the "
        "changed/verified/noop/failed counts above are UNKNOWN."), generic
    # No note, no guess.
    assert formatters._unmeasured_cause(None) == ""
    assert formatters._unmeasured_cause("something else entirely") == ""


def test_the_verbose_go_rename_view_never_claims_nothing_to_do_from_an_unreadable_count():
    """The `--verbose` go-rename view -- the op's DETAIL renderer.

    Not its default: the CLI has installed the compact status for every
    mutation since #645, and `_render_go_rename_text` is the `detail_renderer`
    it installs instead when `--verbose`, `--out` or an explicit machine
    `--format` asks for detail.

    The compact summary was fixed for #683 and this SIBLING renderer was still
    reading its three counters (`skipped_user_named`,
    `go_renamed_candidates`, `go_verified_count`) through a helper that
    silently defaults an unreadable one to 0 and records nothing. A candidate
    counter arriving in the wrong shape therefore printed "nothing to do -- no
    auto-named Go functions to rename" for a batch that had just committed
    1783 renames.

    "nothing to do" is an ACTIONABLE claim: a caller reads it and stops. So the
    property here is not merely that the output differs from the absent case --
    it must not make the claim at all, and it must name the field it could not
    read."""
    from bn import formatters

    committed = {"kind": "go_rename", "committed": True,
                 "go_committed_count": 1783, "go_verified_count": 1783,
                 "skipped_user_named": 12}
    for bogus in ("bad", {"total": 1783}, ["x"], True, ""):
        out = formatters._render_go_rename_text(
            {**committed, "go_renamed_candidates": bogus})
        assert "nothing to do" not in out, (
            f"go_renamed_candidates={bogus!r} could not be read and the view "
            f"still claims there was nothing to do: {out!r}")
        assert _disclosed(out, "go_renamed_candidates"), (
            f"go_renamed_candidates={bogus!r} left no note naming it: {out!r}")
    # Every other counter this view reads goes through the choke point too, and
    # none of them may be fabricated silently. (`defined_count` is read only on
    # the nothing-to-do path above, which is covered there.)
    for counter in ("skipped_user_named", "go_verified_count"):
        for bogus in ("bad", ["x"], {"a": 1}):
            out = formatters._render_go_rename_text(
                {**committed, "go_renamed_candidates": 5, counter: bogus})
            assert _disclosed(out, counter), (
                f"{counter}={bogus!r} was read as a count with no note: {out!r}")
    # Anti-vacuity: the real thing still renders, and cries nothing.
    clean = formatters._render_go_rename_text(
        {**committed, "go_renamed_candidates": 1795})
    assert "1783 renamed" in clean and "malformed" not in clean, clean
    # A genuine nothing-to-do still says so.
    idle = formatters._render_go_rename_text(
        {"kind": "go_rename", "committed": True, "go_renamed_candidates": 0,
         "defined_count": 40, "skipped_user_named": 40})
    assert "nothing to do" in idle and "malformed" not in idle, idle


def test_a_fanout_row_that_cannot_be_trusted_does_not_render_as_a_clean_row():
    """`--all-instances` is a SURVEY, which is what makes this the worst place
    for a silent fallback: a fallback row is indistinguishable from a row that
    genuinely had little to say.

    A `BridgeError` is not a render failure. It is this CLI declaring it cannot
    TRUST the reply, and the documented contract for that is exit 2 out of
    `main()`. A bare `except Exception` around the inner renderer swallowed it
    and produced a fallback render at exit 0.

    BOTH directions, because fixing one destroys the other: the trust failure
    must propagate, and an ordinary renderer crash must STILL fall back and
    still render the remaining rows. The fallback intent is right -- one
    instance's odd-but-parseable payload must not cost the other nine."""
    import pytest

    from bn import formatters
    from bn.transport import BridgeError

    rows = [{"instance": "a", "ok": True, "result": {"n": 1}},
            {"instance": "b", "ok": True, "result": {"n": 2}}]
    payload = {"instances": rows}

    def refuses(inner):
        if inner.get("n") == 1:
            raise BridgeError("cannot trust this reply")
        return "second row"

    with pytest.raises(BridgeError):
        formatters._render_fanout_text(payload, refuses)

    def crashes(inner):
        if inner.get("n") == 1:
            raise TypeError("odd but parseable")
        return "second row rendered"

    out = formatters._render_fanout_text(payload, crashes)
    assert "second row rendered" in out, (
        f"a renderer crash on one instance cost the rest of the survey: {out!r}")
    assert "instance a" in out and '"n": 1' in out, (
        f"the crashed row lost its fallback render: {out!r}")


# Every broad `except` in the module that wraps a call which can raise
# `BridgeError`, with the reason it is allowed to be broad. A bare
# `except Exception` around a renderer call is a POPULATION question, not a
# one-site fix: the one this run found had made a whole boundary inert.
_BROAD_EXCEPT_ALLOWED = {
    "_render_fanout_text": "re-raises BridgeError first, so only a render "
                           "failure is absorbed and the survey continues",
}


def test_no_broad_except_swallows_a_trust_failure():
    """A `BridgeError` must never be absorbed by a catch-all.

    Asserted over EVERY broad handler in the module rather than the one that
    was found: a handler catching `Exception` around a call must re-raise
    `BridgeError` first, or be named here with the reason it need not. A new
    catch-all arrives unclassified instead of silently inert."""
    import ast
    import inspect

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Try):
                continue
            broad = [h for h in node.handlers
                     if h.type is None
                     or (isinstance(h.type, ast.Name) and h.type.id == "Exception")]
            if not broad:
                continue
            reraises = any(isinstance(h.type, ast.Name) and h.type.id == "BridgeError"
                           and any(isinstance(s, ast.Raise) for s in h.body)
                           for h in node.handlers)
            if reraises or fn.name in _BROAD_EXCEPT_ALLOWED:
                continue
            offenders.append(f"{fn.name}:{node.lineno}")
    assert not offenders, (
        f"these broad handlers can absorb a BridgeError: {offenders}. A "
        "BridgeError is the CLI refusing to trust a reply and must reach "
        "main() as exit 2 -- put `except BridgeError: raise` ahead of the "
        "catch-all, or name the function in _BROAD_EXCEPT_ALLOWED with why.")
    # The allow-list is not a place to park a new one: each entry must still
    # exist AND must actually re-raise, so the exemption is executable.
    for name, reason in _BROAD_EXCEPT_ALLOWED.items():
        assert reason, f"{name} is allowed a broad except without a reason"
        fn = next((f for f in ast.walk(tree)
                   if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and f.name == name), None)
        assert fn is not None, f"_BROAD_EXCEPT_ALLOWED names {name}, which is gone"
        assert any(isinstance(h.type, ast.Name) and h.type.id == "BridgeError"
                   and any(isinstance(s, ast.Raise) for s in h.body)
                   for t in ast.walk(fn) if isinstance(t, ast.Try)
                   for h in t.handlers), (
            f"{name} is exempted on the grounds that it re-raises BridgeError, "
            "and it no longer does")


def test_a_well_formed_empty_container_is_never_reported_as_unusable():
    """"PRESENT but the wrong shape" is a THIRD question, and only the choke
    point can answer it.

    `_field_dict` hands back `{}` for a genuinely empty dict AND for one it
    could not use, and `_field_present` is True for both -- so a renderer that
    spelled "unusable" as `_field_present` was a SECOND decider over a question
    already decided, and it answered wrong on the well-formed input: an empty
    `definition` object printed a raw Python repr `def: {}`, with nothing
    disclosed because there was nothing wrong, where the diagnostic
    `<none (parameter/entry/aliased)>` belonged. Ask `_field_skewed`, which is
    the choke point's own record of what it could not read.

    Both directions, because either alone is satisfiable by a constant: a
    well-formed empty container renders the diagnostic and discloses nothing,
    and every FALSY wrong shape still renders what arrived and discloses."""
    from bn import formatters

    absent = formatters._render_defuse_text({})
    assert "def: <none (parameter/entry/aliased)>" in absent, absent
    empty = formatters._render_defuse_text({"definition": {}})
    assert "def: <none (parameter/entry/aliased)>" in empty, (
        "a well-formed EMPTY definition object is a real result -- we looked "
        f"and found none -- and must not render as an unusable payload: {empty!r}")
    assert "malformed" not in empty, (
        f"a well-formed empty container cried malformed: {empty!r}")
    # Non-dict shapes only: a dict with unexpected KEYS is a usable object the
    # choke point has no complaint about, and it renders through the normal
    # branch with placeholders. The skew is about the container's SHAPE.
    for bogus in (0, "", False, [], "x", 3, ["a"]):
        out = formatters._render_defuse_text({"definition": bogus})
        assert "<none (parameter/entry/aliased)>" not in out, (
            f"definition={bogus!r} is not a usable definition object and must "
            f"not render the confident 'why there is none' diagnosis: {out!r}")
        assert _disclosed(out, "definition"), (
            f"definition={bogus!r} left no note naming the field: {out!r}")


# THE population for the text choke point -- and it is no longer the choke
# point's own call sites.
#
# Round 16's major, and this file's own earned rule turned on itself: a guard
# whose population comes from the thing it guards cannot fail. `_TEXT_VALUE_SITES`
# was a table of `_text_value` CALL SITES, so it could only ever cover a read
# somebody had ALREADY routed. It reported a covered population of ONE while
# eight live inline `isinstance(..., str)` filters each dropped a line with no
# note, and one of them (`after_layout` on an unchanged type entry) did not
# even drop it -- it raised AttributeError and cost the whole mutation card.
#
# The population here is the DEFECT'S SIGNATURE instead: every inline
# `isinstance(<x>, str)` test the module performs, harvested from its AST. A new
# one arrives UNCLASSIFIED and red whether or not its author ever heard of the
# choke point, and none of the eight could have been added silently.
#
# Each guard must fall in one of four categories, and the category is RE-DERIVED
# from the module rather than asserted in prose (the `_PROBE_EXCLUSIONS`
# discipline, applied here):
#
#   chokepoint          the guard IS the choke point (`_text_value` itself);
#   routed              the guarded value was produced by `_text_value` in this
#                       same function, so the skew is already recorded;
#   routed_in           the guarded value is a PARAMETER every module call site
#                       hands a `_text_value` result -- directly, through a local
#                       bound from one, or through another routed parameter
#                       (a fixed point, so `_layout_size(before_layout)` inside
#                       `_size_delta` counts);
#   reaches_the_output  the value still reaches the render ON THE PATH THAT
#                       REFUSED IT -- re-rendered (`repr`/`str`/`json.dumps`/
#                       `_render_fallback_text`), interpolated into an f-string,
#                       returned as-is, or covered because the whole payload it
#                       was read off is dumped there. Nothing is absorbed, so
#                       nothing needs disclosing.
#
# Anything else DROPS a value the payload carried and says nothing, which is
# #619 exactly.
#
# "On the path that refused it" is the whole of the rule, and getting it wrong
# is how the FIRST cut of this classifier was itself empty by construction:
# scanning the whole function for any render of the name grants
# `x = value.get(k); if isinstance(x, str) and x: <render x>` -- the module's
# dominant #619 shape -- a clean bill, because the only render of `x` sits in
# the branch where `x` WAS a string. Three of the eight live drops this round
# repaired were invisible under that reading. The drop region is now the other
# branch plus what follows the conditional, and
# `test_every_string_shape_guard_either_shows_the_value_or_routes_it` puts both
# shapes to the rule on a synthetic module so the rule itself can fail.
#
# STATED LIMITATION, because this is a syntactic property over the module's own
# source and an exhaustive answer for arbitrary payload shapes is not attainable
# statically. Three things it does not do, named rather than implied:
#
#   * it sees a shape test spelled `isinstance(x, str)`, including one whose
#     class is a module-level ALIAS of `str` (`_TEXT_TYPES = (str,)`, and an
#     alias of an alias), which the classifier resolves the way
#     `_coercion_sites` resolves a module-level `_EMPTY = []`. A test spelled
#     `type(x) is str` or as a duck-typed `try: x.strip()` is outside it: the
#     first is asserted absent from the module below, the second is not
#     detectable;
#   * the drop region is an APPROXIMATION of control flow, not a CFG: a render
#     that follows the conditional counts as reachable from the refusing path
#     even when some unrelated earlier branch would have returned first. That
#     errs toward granting `reaches_the_output`, so it can over-forgive -- never
#     over-accuse;
#   * it decides whether the value REACHES the output, not whether what reaches
#     it is intelligible.
#
# What DOES hold unconditionally is the runtime half: the container differential
# and the raise sweeps cover every CONTAINER position the probes discover,
# whatever spelling guards it.
_STRING_SHAPE_GUARD_CATEGORIES = (
    "chokepoint", "routed", "routed_in", "reaches_the_output")

_STRING_SHAPE_GUARDS = {
    ("_text_value", "raw", 0): "chokepoint",
    ("_clean_prototype", "proto", 0): "routed_in",
    ("_layout_field_count", "layout", 0): "routed_in",
    ("_layout_field_deltas", "layout_diff", 0): "routed_in",
    ("_layout_size", "layout", 0): "routed_in",
    ("_unmeasured_cause", "first_error", 0): "routed_in",
    ("_leaf_group_key", "kind", 0): "reaches_the_output",
    ("_operation_row_text", "op", 0): "reaches_the_output",
    ("_render_comment_text", "comment", 0): "reaches_the_output",
    ("_render_fallback_text", "value", 0): "reaches_the_output",
    ("_render_mutation_text", "msg", 0): "reaches_the_output",
    ("_render_mutation_text", "msg", 1): "reaches_the_output",
    ("_render_orient_text", "raw", 0): "reaches_the_output",
    ("_render_proto_text", "prototype", 0): "reaches_the_output",
    ("_render_py_exec_text", "result", 0): "reaches_the_output",
    ("_render_read_text", "hex_str", 0): "reaches_the_output",
    ("_render_sections_text", "n", 0): "reaches_the_output",
    ("_render_trace_frontiers", "reason", 0): "reaches_the_output",
    ("_render_type_info_text", "decl", 0): "reaches_the_output",
    ("_render_type_info_text", "layout", 0): "reaches_the_output",
    ("render", "text", 0): "reaches_the_output",
    ("rendered", "out", 0): "reaches_the_output",
}


def _negations_to(root, target, seen=0):
    """How many `not`s wrap `target` on its path down from `root`, or None when
    `target` is not in this subtree.

    The polarity of a shape test decides WHICH branch is the one that refused
    the value: `if isinstance(x, str):` hands the usable value to the body,
    `if not isinstance(x, str):` hands it to everything after. Getting that
    backwards inverts the whole question."""
    import ast

    if root is target:
        return seen
    for child in ast.iter_child_nodes(root):
        deeper = seen + 1 if (isinstance(root, ast.UnaryOp)
                              and isinstance(root.op, ast.Not)) else seen
        found = _negations_to(child, target, deeper)
        if found is not None:
            return found
    return None


@functools.lru_cache(maxsize=1)
def _string_shape_guards():
    """Every inline `isinstance(<x>, str)` test in the MODULE, classified."""
    import inspect

    from bn import formatters

    return _classify_string_shape_guards(inspect.getsource(formatters))


@functools.lru_cache(maxsize=4)
def _classify_string_shape_guards(source):
    """Every inline `isinstance(<x>, str)` test in `source`, classified.

    See `_STRING_SHAPE_GUARDS` for what the four categories mean and why the
    population is harvested here rather than taken from the choke point's own
    call sites. Returns `{(function, tested expression): category}`, with
    `"DROPS"` for a guard that discards a value the payload carried without
    routing it through `_text_value` -- the #619 defect, and the only outcome
    the test below refuses.

    Takes the SOURCE rather than reading the module itself, so the rule can be
    put to a synthetic module carrying both shapes on purpose. A classifier that
    cannot tell a dropping filter from a showing one reports any module clean,
    which is how the first cut of this passed while three of the eight live
    drops it was written for sat in front of it."""
    import ast

    shows = {"repr", "str", "json.dumps", "_render_fallback_text"}
    tree = ast.parse(source)
    # A class held in a MODULE-LEVEL name is the same guard spelled once
    # removed (`_TEXT_TYPES = (str,)`, then `isinstance(x, _TEXT_TYPES)`), and
    # requiring the literal `str` put it outside the population AND outside the
    # anti-drift assertion -- it would drop a text field with nothing in this
    # file able to see it. Resolved the way `_coercion_sites` resolves a
    # module-level `_EMPTY = []`: collect the names bound at module scope to
    # `str`, or to a tuple/list containing it, and treat them as naming `str`.
    str_aliases: set[str] = set()
    for _ in range(4):                       # an alias of an alias
        before = set(str_aliases)
        for stmt in tree.body:
            targets = (stmt.targets if isinstance(stmt, ast.Assign)
                       else [stmt.target] if isinstance(stmt, ast.AnnAssign)
                       else [])
            if not (targets and getattr(stmt, "value", None) is not None):
                continue
            bound = (list(stmt.value.elts)
                     if isinstance(stmt.value, (ast.Tuple, ast.List))
                     else [stmt.value])
            if not any(isinstance(c, ast.Name)
                       and (c.id == "str" or c.id in str_aliases) for c in bound):
                continue
            str_aliases |= {t.id for t in targets if isinstance(t, ast.Name)}
        if before == str_aliases:
            break

    def _names_str(cls, aliases):
        return isinstance(cls, ast.Name) and (cls.id == "str" or cls.id in aliases)

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    calls: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)

    def owner(node):
        # The INNERMOST enclosing function: a guard inside `_discloses`'s
        # `rendered` closure belongs to the closure, not to the decorator.
        # `None` for a call outside every function body -- a decorator
        # expression (`@_discloses(prefix=True)`) is one.
        enclosing = [f for f in funcs if f.lineno <= node.lineno <= f.end_lineno]
        return min(enclosing, key=lambda f: f.end_lineno - f.lineno) if enclosing else None

    owner_of = {id(node): owner(node) for name in calls for node in calls[name]}
    params = {id(f): [p.arg for p in f.args.posonlyargs + f.args.args] for f in funcs}

    binds: dict[int, dict[str, list]] = {}
    for fn in funcs:
        table: dict[str, list] = {}
        for node in ast.walk(fn):
            target = value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and value is not None:
                table.setdefault(target.id, []).append(value)
        binds[id(fn)] = table

    routed_params: set[tuple[str, int]] = set()

    def routed(node, fn):
        """Did this expression come out of the text choke point?"""
        if fn is None:                          # a decorator expression
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_text_value":
                return True
            return any(routed(a, fn) for a in node.args)
        if isinstance(node, ast.BoolOp):        # `after_layout or ""`
            return routed(node.values[0], fn)
        if isinstance(node, ast.Name):
            bound = binds[id(fn)].get(node.id)
            if bound:
                return all(routed(v, fn) for v in bound)
            names = params[id(fn)]
            return (node.id in names
                    and (fn.name, names.index(node.id)) in routed_params)
        return False

    # Fixed point, because routing is transitive: `_layout_size(after)` inside
    # `_size_delta` is routed only once `_size_delta`'s own parameter is known
    # to be. One pass would have left the layout helpers looking unrouted and
    # this whole table unprovable.
    for _ in range(len(funcs)):
        before = set(routed_params)
        for fn in funcs:
            sites = calls.get(fn.name, ())
            if not sites:
                continue
            for idx in range(len(params[id(fn)])):
                if all(len(c.args) > idx and routed(c.args[idx], owner_of[id(c)])
                       for c in sites):
                    routed_params.add((fn.name, idx))
        if routed_params == before:
            break

    def drop_region(fn, guard):
        """The code reached on the path where the value was NOT a usable string.

        THE question the first cut of this classifier did not ask, and the reason
        it could not fail for the module's dominant #619 shape. Scanning the
        WHOLE function for any render of the name files
        `if isinstance(x, str) and x: lines.append(f"...{x}")` as safe -- the
        only render of `x` is in the branch where it IS a string, and the path
        that dropped it renders nothing at all. Under that rule, three of the
        eight live drops this round repaired classified `reaches_the_output`,
        and a fresh one added beside them classified safe too.

        The region is the OTHER branch plus everything that follows the
        conditional, climbing out of each enclosing block up to the function
        body -- falling through is how a refusing path reaches a later
        `return _render_fallback_text(value)`. For a ternary or a comprehension
        filter there is no fall-through to add: the other arm is the whole
        region, and a filtered-out element reaches nothing at all.
        """
        parents = {id(child): node
                   for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}

        def following(node):
            out, current = [], node
            while True:
                parent = parents.get(id(current))
                if parent is None:
                    return out
                for _field, value in ast.iter_fields(parent):
                    if isinstance(value, list) and any(v is current for v in value):
                        index = next(i for i, v in enumerate(value) if v is current)
                        out.extend(value[index + 1:])
                if parent is fn:
                    return out
                current = parent

        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                arms = [(node.test, node.body, node.orelse, True)]
            elif isinstance(node, ast.IfExp):
                arms = [(node.test, [node.body], [node.orelse], False)]
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                arms = [(cond, [node.elt], [], False)
                        for gen in node.generators for cond in gen.ifs]
            else:
                continue
            for test, body, orelse, falls_through in arms:
                negations = _negations_to(test, guard)
                if negations is None:
                    continue
                refused = orelse if negations % 2 == 0 else body
                return list(refused) + (following(node) if falls_through else [])
        return None

    def classify(fn, guard, expr):
        if fn.name == "_text_value":
            return "chokepoint"
        bound = binds[id(fn)].get(expr)
        if bound and all(routed(v, fn) for v in bound):
            return "routed"
        names = params[id(fn)]
        if expr in names and (fn.name, names.index(expr)) in routed_params:
            return "routed_in"
        refused = drop_region(fn, guard)
        if refused is None:
            # No conditional consumes this shape test, so which path drops the
            # value cannot be decided. Fail CLOSED rather than grant it.
            return "unstructured"
        # The value still reaches the render ON THE PATH THAT REFUSED IT: itself,
        # or the payload it was read off being dumped whole.
        carriers = {expr}
        for value in bound or ():
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get"):
                carriers.add(ast.unparse(value.func.value))
            elif isinstance(value, ast.Subscript):
                carriers.add(ast.unparse(value.value))
        for root in refused:
            for node in ast.walk(root):
                if (isinstance(node, ast.Call) and ast.unparse(node.func) in shows
                        and any(ast.unparse(a) in carriers for a in node.args)):
                    return "reaches_the_output"
                if (isinstance(node, ast.FormattedValue)
                        and ast.unparse(node.value).split("!")[0] in carriers):
                    return "reaches_the_output"
                if (isinstance(node, ast.Return) and node.value is not None
                        and ast.unparse(node.value) == expr):
                    return "reaches_the_output"
        return "DROPS"

    # EVERY guard NODE, not one per (function, expression). Keying the
    # population by name alone silently discarded the second and later guards
    # over the same expression -- `_render_mutation_text` tests `msg` twice --
    # so a dropping filter placed at the discarded site left the refusal green.
    # That is the same "cannot fail" defect as the whole-function scan it just
    # replaced, one disguise along, so the occurrence INDEX is part of the key
    # and the count of nodes is the count of rows.
    guards = []
    for node in calls.get("isinstance", ()):
        if len(node.args) != 2:
            continue
        classes = node.args[1]
        elements = (list(classes.elts)
                    if isinstance(classes, (ast.Tuple, ast.List)) else [classes])
        if not any(_names_str(c, str_aliases) for c in elements):
            continue
        guards.append((node.lineno, node.col_offset, node))
    found: dict[tuple[str, str, int], str] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for _line, _col, node in sorted(guards, key=lambda g: (g[0], g[1])):
        fn = owner_of[id(node)]
        expr = ast.unparse(node.args[0])
        index = occurrences.get((fn.name, expr), 0)
        occurrences[(fn.name, expr)] = index + 1
        found[(fn.name, expr, index)] = classify(fn, node, expr)
    return found


def test_every_string_shape_guard_either_shows_the_value_or_routes_it():
    """The text choke point's population, and the round-16 major it replaces.

    The old table listed the module's `_text_value` CALL SITES -- the set the
    choke point already covers -- so it was a population of size one derived
    from the thing it guards, and it could not fail. Eight live inline
    `isinstance(..., str)` filters were dropping a line each with no note while
    it reported full coverage.

    The population is the defect's own shape now: every inline string-shape test
    in the module. Each must either keep the value visible on its failing path
    or take it through the choke point, and both are re-derived from the module
    (see `_STRING_SHAPE_GUARDS`), so a new filter that silently drops a line
    arrives here red."""
    import ast
    import inspect

    from bn import formatters

    guards = _string_shape_guards()
    dropping = sorted(k for k, v in guards.items() if v == "DROPS")
    assert not dropping, (
        f"{dropping} test a value's shape and, when it is not a string, drop it "
        "with nothing rendered and nothing recorded -- so the line comes out "
        "byte-identical to a payload that carried no such field at all. Read it "
        "through `_text_value` instead, or render the value you refused.")
    assert dict(guards) == dict(_STRING_SHAPE_GUARDS), (
        f"the module's string-shape guards are {sorted(guards.items())} and this "
        f"table declares {sorted(_STRING_SHAPE_GUARDS.items())}. Every one is "
        "classified by re-derivation, so a guard that changed category changed "
        "behaviour -- say which it is now.")
    assert set(guards.values()) <= set(_STRING_SHAPE_GUARD_CATEGORIES)
    # ANTI-VACUITY ON THE RULE, not on the module. An emptiness claim over a
    # classifier that grants `reaches_the_output` to everything is free, and
    # that is precisely how the first cut of this shipped: it scanned the WHOLE
    # function for any render of the name, so the module's dominant shape --
    # `x = value.get(k); if isinstance(x, str) and x: <render x>` -- came out
    # SAFE even though the path that refused `x` renders nothing at all. Three
    # of the eight live drops this round repaired were invisible to it.
    #
    # Both shapes, in one synthetic module, so the rule is asked the question
    # directly rather than inferred from the module happening to be clean.
    probe = "\n".join((
        "def shows(value):",
        "    if not isinstance(value, dict):",
        "        return _render_fallback_text(value)",
        "    text = value.get('k')",
        "    if isinstance(text, str) and text:",
        "        return 'k=' + text",
        "    return _render_fallback_text(value)",
        "",
        "def drops(value):",
        "    if not isinstance(value, dict):",
        "        return _render_fallback_text(value)",
        "    lines = []",
        "    note = value.get('note')",
        "    if isinstance(note, str) and note:",
        "        lines.append(f'note: {note}')",
        "    return chr(10).join(lines)",
        "",
        "def routes(value):",
        "    note = _text_value(value, 'note')",
        "    if isinstance(note, str):",
        "        return note",
        "    return ''",
        ""))
    verdicts = _classify_string_shape_guards(probe)
    assert verdicts[("drops", "note", 0)] == "DROPS", (
        "the classifier cannot see the module's dominant #619 shape -- a value "
        "rendered ONLY in the branch where it was usable, and dropped in "
        f"silence otherwise: {verdicts}")
    assert verdicts[("shows", "text", 0)] == "reaches_the_output", (
        f"a value the refusing path still dumps is not a drop: {verdicts}")
    assert verdicts[("routes", "note", 0)] == "routed", (
        f"a value taken from the choke point is already disclosed: {verdicts}")
    # And the population itself, since the categories a rule collapse would
    # empty are the ones doing the work here.
    counts = collections.Counter(guards.values())
    assert counts["routed_in"] >= 5 and counts["reaches_the_output"] >= 10, counts
    # The stated limitation, made checkable for the two spellings that ARE
    # detectable: this harvest sees `isinstance(x, str)` with a literal class,
    # so a module that started testing shape another way would leave the
    # population silently. A duck-typed `try: x.strip()` remains outside it and
    # is disclosed rather than claimed.
    tree = ast.parse(inspect.getsource(formatters))
    other = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Call)
                and ast.unparse(node.left.func) == "type"):
            other.append(ast.unparse(node))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance" and len(node.args) == 2
                and not isinstance(node.args[1], (ast.Name, ast.Tuple, ast.List))):
            other.append(ast.unparse(node))
    assert not other, (
        f"{other} tests a value's type in a spelling this harvest cannot see, so "
        "the population above is no longer the module's whole string-shape "
        "surface. Spell it `isinstance(x, str)` or widen the harvest.")


def test_a_wrong_shaped_text_field_never_renders_as_if_nothing_was_observed():
    """The leaf under the container round 15 said it had closed.

    `set_prototype`'s "what landed" line IS the renderer's subject: an agent
    reads it to confirm the prototype it set is live. `_field_dict(item,
    "observed")` discloses an unusable OBSERVATION, but the `prototype` INSIDE a
    well-formed observation was read inline, so a non-string one dropped the
    line and the row came out byte-identical to an op that reported no
    observation at all -- the #619 defect exactly, one level in, and undisclosed.

    Three states, as everywhere else in this module. A usable string renders the
    line; ABSENT, an explicit null and an empty string are "nothing was
    observed" and render as such WITHOUT a note (over-disclosing would destroy
    the signal); anything present that is not a string is a skew that must be
    both visibly different and named."""
    from bn import formatters

    def row(observed):
        item = {"op": "set_prototype", "status": "verified",
                "function": "fn", "address": "0x1"}
        if observed is not None:
            item["observed"] = observed
        return formatters._render_mutation_text({"results": [item]})

    nothing_observed = row(None)
    assert "int fn()" in row({"prototype": "int fn()"}), (
        "a usable observed prototype must render the what-landed line")
    # The mirror: the three shapes that really do claim nothing.
    for quiet in ({}, {"prototype": None}, {"prototype": ""}):
        out = row(quiet)
        assert out == nothing_observed, (
            f"observed={quiet!r} claims no prototype and must render exactly "
            f"like an op that reported no observation: {out!r}")
        assert "malformed" not in out, (
            f"observed={quiet!r} is not a skew and must not disclose: {out!r}")
    for bogus in (7, 0, True, ["int fn()"], {"decl": "int fn()"}, (), 1.5):
        out = row({"prototype": bogus})
        assert out != nothing_observed, (
            f"observed carrying prototype={bogus!r} is an observation the "
            "renderer could not read, and it rendered byte-identically to an op "
            f"that observed nothing: {out!r}")
        assert _disclosed(out, "prototype"), (
            f"prototype={bogus!r} was dropped with no note naming it: {out!r}")


# The nine positions round 16's falsification lens proved LIVE, each executed
# here. The AST test above owns the POPULATION -- it is what makes a tenth
# arrive red -- and this owns the PROOF: for each one, a container renders
# differently from absent and names the field, a string renders, and the three
# quiet shapes stay silent.
#
# Each row is `(what, renderer, build, key, good)`: `build(value)` places
# `value` at the position and `build(_ABSENT)` leaves it out entirely.
_ABSENT = object()


def _without(mapping, key):
    return {k: v for k, v in mapping.items() if k != key}


def _ref_context(value):
    ctx = {"section": ".text", "disasm": "call rax"}
    return {"address": "0x1000",
            "target_context": _without(ctx, "disasm") if value is _ABSENT
            else {**ctx, "disasm": value}}


def _read_note(value):
    out = {"address": "0x1000", "hex": "deadbeef", "note": "truncated at 4 bytes"}
    return _without(out, "note") if value is _ABSENT else {**out, "note": value}


def _py_exec_stdout(value):
    out = {"stdout": "hello"}
    return _without(out, "stdout") if value is _ABSENT else {**out, "stdout": value}


def _type_entry(key, changed, value):
    entry = {"type_name": "widget_t", "changed": changed,
             "before_layout": "struct widget_t size=0x10\n    0x0 int a",
             "after_layout": "struct widget_t size=0x20\n    0x0 int a\n    0x8 int b",
             "layout_diff": "--- before\n+++ after\n+    0x8 int b"}
    entry = _without(entry, key) if value is _ABSENT else {**entry, key: value}
    return {"success": True, "committed": True, "affected_types": [entry],
            "results": [{"op": "types_declare", "status": "verified", "count": 1}]}


def _data_vars_resume(value):
    row = {"a": "0x1000", "t": "int", "w": 4}
    return {"has_more": True,
            "items": [{"a": "0x0ff0", "t": "int", "w": 4},
                      _without(row, "a") if value is _ABSENT else {**row, "a": value}]}


def _unmeasured_first_error(value):
    out = {"kind": "mutation", "measured": False, "committed": True,
           "preview": False, "dirty_after": True, "changed_count": None}
    return _without(out, "first_error") if value is _ABSENT else {**out, "first_error": value}


_ROUTED_TEXT_POSITIONS = (
    ("a ref row's disassembly", "_render_evidence_xrefs_text", _ref_context,
     "disasm", "call rax"),
    ("a partial read's note", "_render_read_text", _read_note,
     "note", "truncated at 1 byte"),
    ("a script's stdout", "_render_py_exec_text", _py_exec_stdout,
     "stdout", "hello"),
    ("a changed type's before size", "_render_mutation_text",
     functools.partial(_type_entry, "before_layout", True),
     "before_layout", "struct widget_t size=0x08\n    0x0 int a"),
    ("a changed type's after size", "_render_mutation_text",
     functools.partial(_type_entry, "after_layout", True),
     "after_layout", "struct widget_t size=0x40\n    0x0 int a"),
    ("a changed type's field deltas", "_render_mutation_text",
     functools.partial(_type_entry, "layout_diff", True),
     "layout_diff", "--- before\n+++ after\n+    0x8 int c"),
    ("an unchanged type's layout", "_render_mutation_text",
     functools.partial(_type_entry, "after_layout", False),
     "after_layout", "struct widget_t size=0x10\n    0x0 int a\n    0x4 int b"),
    ("an unmeasured summary's cause", "_render_mutation_summary_text",
     _unmeasured_first_error, "first_error",
     "unmeasured: an op reported no results[] rows, so "
     "changed/verified/noop/failed counts could not be derived (None, not a "
     "confirmed 0) and dirty_after defaults to True as a fail-safe -- do not "
     "assume nothing changed"),
)


def test_no_unreadable_text_field_renders_as_if_the_payload_never_carried_it():
    """The eight live absorptions round 16 found behind inline shape filters.

    Every one of them dropped its clause and left the render byte-identical to a
    payload that carried no such field -- a ref row with no disassembly, a
    partial read with no truncation note, a script that printed nothing, a type
    change with no size or field delta, an unmeasured summary that does not say
    WHY. One was worse than silent: a container `after_layout` on an unchanged
    type entry reached `.strip()` and raised AttributeError, costing the whole
    mutation card.

    Three states each, as everywhere else here. A usable string renders; ABSENT,
    an explicit null and an empty string claim nothing and must render exactly
    like absence, silently; anything else is a skew that must be visibly
    different AND named, because "not disclosed" is the half that makes an
    absorption dangerous rather than merely lossy."""
    from bn import formatters

    for what, renderer, build, key, good in _ROUTED_TEXT_POSITIONS:
        render = getattr(formatters, renderer)
        absent = render(build(_ABSENT))
        assert render(build(good)) != absent, (
            f"{what}: a usable {key} renders nothing extra, so this position "
            "proves nothing either way")
        for quiet in (None, ""):
            out = render(build(quiet))
            assert out == absent, (
                f"{what}: {key}={quiet!r} claims nothing and must render exactly "
                f"like the field being absent: {out!r} != {absent!r}")
            assert "malformed" not in out, (
                f"{what}: {key}={quiet!r} is not a skew and must not disclose")
        for bogus in ({"a": 1}, ["a"], 7, True, 1.5, ()):
            out = render(build(bogus))
            assert out != absent, (
                f"{what}: {key}={bogus!r} was absorbed -- the render is "
                f"byte-identical to the payload never carrying {key}: {out!r}")
            assert _disclosed(out, key), (
                f"{what}: {key}={bogus!r} was dropped with no note naming it: {out!r}")


def test_a_paged_window_never_loses_its_resume_hint_in_silence():
    """The ninth position, and the one that is NOT a byte-identical absorption.

    A container-shaped address on the LAST row of a truncated `data_vars` window
    dropped the `resume with --start ...` hint. The value itself still reaches
    the output -- the row cell renders its repr -- so the render is not
    identical to one that never carried it, and the table above would have
    mis-stated the property. What vanished is the HINT: a paged window that says
    more rows remain and gives no way to reach them, with nothing to say the
    payload was the reason.

    Kept separate rather than flagged in the table, because a per-row exception
    is how a table stops meaning one thing."""
    from bn import formatters

    render = formatters._render_data_vars_text
    assert "resume with --start 0x2001" in render(_data_vars_resume("0x2000")), (
        "a usable address on the last row must still produce the resume hint")
    for quiet in (_ABSENT, None, ""):
        out = render(_data_vars_resume(quiet))
        assert "resume with" not in out, (
            f"a={quiet!r} carries no address to resume from: {out!r}")
        assert "malformed" not in out, (
            f"a={quiet!r} is not a skew and must not disclose: {out!r}")
    for bogus in ({"a": 1}, ["a"], 7, True, 1.5, ()):
        out = render(_data_vars_resume(bogus))
        assert "resume with" not in out, (
            f"a={bogus!r} is not an address and must not be turned into one")
        assert _disclosed(out, "a"), (
            f"a={bogus!r} cost the window its resume hint with no note naming "
            f"the field: {out!r}")


def test_an_unreadable_op_status_is_neither_a_crash_nor_a_pass():
    """One decider for "did this op fail", and it answers in THREE states.

    Four sites asked the question: three spelled it
    `str(row.get("status")) in FAILED_MUTATION_STATUSES` and the fourth tested
    the RAW value against the set. The two answers did not merely differ -- the
    raw one RAISED. An unhashable status (a dict or list, which is what a
    malformed or newer bridge result carries) is `TypeError: cannot use 'dict'
    as a set element`, and it cost the WHOLE mutation card, where the same row
    with `status` absent rendered cleanly.

    Coercing all four with `str()` is the other half of the trap and is what
    this test refuses: `str({...})` is in no failure set, so an UNREADABLE
    status would answer "not a failure" -- identical to a row that genuinely
    passed -- and `ok` would come out True over a row nobody could classify.
    That is #683's fabricated zero in a different coat. Unreadable must stay
    distinguishable from not-failed: the status goes through the text choke
    point, so the third state survives as a recorded skew, the card names it,
    and `ok` is withheld."""
    from bn import formatters

    def card(status):
        item = {"op": "set_prototype", "function": "fn", "address": "0x1",
                "observed": {"prototype": "int fn()"}}
        if status is not _ABSENT:
            item["status"] = status
        return {"success": True, "committed": True, "results": [item]}

    clean = formatters._render_mutation_text(card(_ABSENT))
    assert "int fn()" in clean
    assert formatters._add_mutation_ok(card(_ABSENT))["ok"] is True
    assert formatters._add_mutation_ok(card("verified"))["ok"] is True
    # A named failure is a failure, unchanged, and says nothing about shape.
    for failing in sorted(formatters.FAILED_MUTATION_STATUSES):
        assert formatters._add_mutation_ok(card(failing))["ok"] is False
        assert "malformed" not in formatters._render_mutation_text(card(failing))
    # #447 parity, on EVERY shape and not only the unreadable ones: a uniform
    # `jq '.ok'` must give the same answer whether or not `--summary` was
    # passed. Half a parity is how the first cut of this shipped -- `ok` was
    # withheld on the full result and claimed on the compact one built from the
    # identical payload, because only one of the two derivations read the row
    # status inside its recorder capture.
    for status in (_ABSENT, None, "verified", "noop", "verification_failed",
                   {"code": 7}, ["verification_failed"], 7, True, 1.5, ()):
        payload = card(status)
        assert (formatters._mutation_summary(payload)["ok"]
                is formatters._add_mutation_ok(payload)["ok"]), (
            f"`jq '.ok'` flips with --summary on status={status!r}: "
            f"{formatters._mutation_summary(payload)['ok']} vs "
            f"{formatters._add_mutation_ok(payload)['ok']}")
    for unreadable in ({"code": 7}, ["verification_failed"], 7, True, 1.5, ()):
        out = formatters._render_mutation_text(card(unreadable))
        assert _disclosed(out, "status"), (
            f"status={unreadable!r} left the card with no note naming it, so an "
            f"unclassifiable row reads exactly like one that passed: {out!r}")
        assert formatters._add_mutation_ok(card(unreadable))["ok"] is False, (
            f"status={unreadable!r} cannot be read, so 'no op row failed' is not "
            "established and ok must not be claimed")
        summary = formatters._mutation_summary(card(unreadable))
        assert summary["ok"] is False and summary["success"] is False, (
            f"status={unreadable!r} is a row nobody could classify, so the "
            f"compact status must withhold ok too, not claim it: {summary!r}")
        assert "malformed" in str(summary.get("first_error")), (
            f"status={unreadable!r} must reach the compact summary a control "
            f"loop reads, not just the text card: {summary!r}")
        # And the count keys, not only `ok`: a row nobody could classify is not
        # a row that passed, so `failed_count: 0` over it is the fabricated
        # zero in the SECONDARY keys -- and the sibling caller of the one
        # builder already refused it there, which made the two answer
        # `measured` differently for the identical defect.
        assert summary["measured"] is False and summary["failed_count"] is None, (
            f"status={unreadable!r} left the compact summary stating counts "
            f"derived from a row it could not classify: {summary!r}")
        go = formatters._go_rename_summary({
            "kind": "go_rename", "success": True, "committed": True,
            "go_committed_count": 1, "go_failed_count": 0,
            "results": [{"status": unreadable}]})
        assert (go["measured"], go["failed_count"]) == (summary["measured"],
                                                        summary["failed_count"]), (
            f"the two callers of the one builder answer `measured`/"
            f"`failed_count` differently for status={unreadable!r}: "
            f"{go!r} vs {summary!r}")
    # Anti-vacuity for the pair above: a readable batch is still measured and
    # still states its zero.
    clean_summary = formatters._mutation_summary(card("verified"))
    assert (clean_summary["measured"], clean_summary["failed_count"]) == (True, 0), (
        clean_summary)


def test_a_type_row_never_states_a_field_count_it_could_not_measure():
    """#683's harm, one row over from where the op-row property guards it.

    `struct widget_t, 0 fields` reads as "this type is empty" -- the same
    "nothing landed" a control loop acts on -- and the unchanged-type row
    derived that zero from `after_layout` WITHOUT asking whether it could read
    it. A skew note went out beside the number, and the note is not what the
    loop reads: that is exactly the reasoning
    `test_an_op_row_never_states_a_count_the_payload_did_not` applies to the op
    row, so the module held two different count contracts. It holds one now.

    ABSENT and EMPTY are still MEASURED zeroes and still state one -- refusing
    those would be a silent cap on every honest row."""
    from bn import formatters

    def card(after_layout):
        entry = {"type_name": "widget_t", "changed": False}
        if after_layout is not _ABSENT:
            entry["after_layout"] = after_layout
        return formatters._render_mutation_text(
            {"success": True, "committed": True, "affected_types": [entry],
             "results": [{"op": "types_declare", "status": "verified", "count": 1}]})

    assert ", 2 fields" in card("struct widget_t size=0x8\n    0x0 int a\n"
                                "    0x4 int b"), "a readable layout must be counted"
    for measured in (_ABSENT, None, ""):
        assert ", 0 fields" in card(measured), (
            f"after_layout={measured!r} claims nothing and 'we looked and found "
            "none' is a real measurement -- it must keep stating its zero")
    for unreadable in ({"fields": [1, 2, 3]}, ["0x0 int a"], 7, True, 1.5, ()):
        out = card(unreadable)
        assert "0 fields" not in out, (
            f"after_layout={unreadable!r} could not be read, so a field count "
            f"derived from it is fabricated -- and 0 is the dangerous one: {out!r}")
        assert _disclosed(out, "after_layout"), (
            f"after_layout={unreadable!r} was refused with no note naming it: {out!r}")
    # The refusal is attributed to the ENTRY it came from, not to the render.
    # `_field_skewed` answers for the whole card, so a batch whose FIRST type
    # arrived unreadable would otherwise suppress the count of every later type
    # in the same batch -- the mirror of the fabrication, and just as wrong:
    # a row that WAS measured must still state its zero or its two.
    batch = formatters._render_mutation_text({
        "success": True, "committed": True,
        "results": [{"op": "types_declare", "status": "verified", "count": 2}],
        "affected_types": [
            {"type_name": "broken_t", "changed": False, "after_layout": {"f": [1]}},
            {"type_name": "widget_t", "changed": False,
             "after_layout": "struct widget_t size=0x8\n    0x0 int a\n    0x4 int b"},
        ]})
    assert "struct widget_t size=0x8, 2 fields" in batch, (
        "a readable entry beside an unreadable one must still state its own "
        f"measured count: {batch!r}")
    assert "struct broken_t, field count not stated" in batch, (
        f"the unreadable entry must be the one refused: {batch!r}")


def test_the_go_rename_status_withholds_ok_on_rows_it_could_not_read():
    """#447 parity on the OTHER caller of the one builder, at BOTH granularities
    of the failure list.

    `go rename` reports through its own counters, and its `results[]` holds only
    the FAILURE rows. Read in the builder's argument list -- outside the
    recorder capture -- an unreadable failure list still disclosed through
    `first_error` but never reached `unusable`, so the compact status claimed
    `ok: true` on the payload `_add_mutation_ok` refuses. The compact path is
    this op's DEFAULT, so `jq '.ok'` flipped on whether the caller asked for
    detail -- the same half-parity round 18 closed on the generic summary,
    surviving one caller along.

    Closing it on the LIST was still only half, and that is what this round
    adds: the ROW STATUSES were never classified here at all. `ok` and
    `failed_count` came off `go_failed_count` alone, so a row that NAMES a
    failure (or one whose status nobody could read) beside a counter saying
    zero produced `ok: true, failed_count: 0, first_error: null` with no
    disclosure, while `_add_mutation_ok` -- which classifies the rows --
    refused the identical payload. The counter and the rows answer the SAME
    question on this op, so they may not disagree: a `go_failed_count` of 0
    beside a failure row is not a measurement of the run.

    This test is about the COMPACT status alone. `ok` is now decided in one
    place for this op -- `_add_mutation_ok` delegates to `_go_rename_summary`
    for a `go_rename` envelope -- so asserting the two functions agree would be
    true by construction and could not fail. What a CALLER observes across the
    CLI's two paths is asserted instead, end to end, in
    `test_the_go_rename_ok_key_reads_the_same_on_both_cli_paths`."""
    from bn import formatters

    def envelope(results):
        value = {"kind": "go_rename", "success": True, "committed": True,
                 "go_renamed_candidates": 3, "go_committed_count": 3,
                 "go_verified_count": 3, "go_failed_count": 0,
                 "skipped_user_named": 0, "skipped_changed_during_apply": 0}
        if results is not _ABSENT:
            value["results"] = results
        return value

    assert formatters._go_rename_summary(envelope(_ABSENT))["ok"] is True
    assert formatters._go_rename_summary(envelope([]))["ok"] is True
    for unreadable in ({"row": 1}, "boom", 7, True):
        summary = formatters._go_rename_summary(envelope(unreadable))
        assert summary["ok"] is False and summary["success"] is False, (
            f"results={unreadable!r} is a failure list nobody could read, so "
            f"'nothing failed' is not established and ok must not be: {summary!r}")
        assert "malformed" in str(summary.get("first_error")), summary
    # A readable row that NAMES a failure, and a row whose status nobody could
    # read, both beside `go_failed_count: 0` -- the counter and the rows
    # disagreeing about the one question this summary exists to answer.
    for row in ({"status": "verification_failed"}, {"status": "rollback_failed"},
                {"status": {"code": 7}}, {"status": ["verification_failed"]},
                {"status": 7}, {"status": True}):
        summary = formatters._go_rename_summary(envelope([row]))
        assert summary["ok"] is False and summary["success"] is False, (
            f"results=[{row!r}] with go_failed_count 0: the rows and the "
            f"counter disagree, so 'nothing failed' is not established and ok "
            f"must not be claimed: {summary!r}")
        assert summary["measured"] is False and summary["failed_count"] is None, (
            f"a counter contradicted by its own rows is not a measurement: "
            f"{summary!r}")
        assert summary["first_error"], (
            f"results=[{row!r}] left first_error empty, which is the one key a "
            f"control loop is told to check: {summary!r}")
        assert "warning: unmeasured" in formatters._render_mutation_summary_text(summary), (
            f"the compact TEXT -- this op's default view -- said nothing: {summary!r}")
    # Anti-vacuity, and the honest agreement: the same failure row with the
    # counter that AGREES is measured, states its 1, and is not disclosed as a
    # shape problem. A blanket refusal would satisfy every assertion above.
    agreed = formatters._go_rename_summary(
        {**envelope([{"status": "verification_failed", "message": "readback"}]),
         "go_failed_count": 1, "success": False, "committed": False})
    assert agreed["measured"] is True and agreed["failed_count"] == 1, agreed
    assert agreed["ok"] is False and agreed["first_error"] == "readback", agreed
    assert "malformed" not in str(agreed["first_error"]), agreed
    # A NON-failure row (the bridge sends only failures, but a `noop` row is
    # readable and names no failure) does not contradict a zero counter.
    quiet = formatters._go_rename_summary(envelope([{"status": "noop"}]))
    assert quiet["ok"] is True and quiet["measured"] is True, quiet
    # The counter need not be ZERO to be contradicted, and that was the half a
    # `not failed` test could never see: the bridge builds this counter AS
    # `len(failed_rows)`, so a counter naming FEWER failures than the payload
    # carries rows for is not a measurement either -- and the op's `--verbose`
    # text view states the ROW count, so the two surfaces would disagree out
    # loud.
    understated = formatters._go_rename_summary(
        {**envelope([{"status": "verification_failed"}] * 5),
         "go_failed_count": 2, "success": False, "committed": False})
    assert understated["measured"] is False, (
        f"5 failure rows beside go_failed_count 2 is not a measured run: "
        f"{understated!r}")
    assert understated["failed_count"] is None and understated["ok"] is False, understated
    assert "warning: unmeasured" in formatters._render_mutation_summary_text(understated)
    # The OTHER direction is a contradiction too, and the round-22 comment that
    # excused it was false about the payload: the bridge builds `results` AS
    # `failed_rows` and `go_failed_count` AS `len(failed_rows)`, so the listing
    # is not a sample of a larger population. Only the `--verbose` view's
    # DISPLAY is capped -- it prints 50 rows and then "... and N more" -- and
    # the count it states beside them is the FULL row count. Believing the
    # larger counter therefore let this op's two views -- its `--verbose` text
    # view and its compact summary -- state
    # DIFFERENT failure counts while both read `measured`, which is the drift
    # #685 exists to close rather than a wording problem.
    overstated_value = {**envelope([{"status": "verification_failed", "message": "m"}] * 5),
                        "go_failed_count": 60, "success": False, "committed": False}
    overstated = formatters._go_rename_summary(overstated_value)
    assert overstated["measured"] is False, (
        f"go_failed_count 60 beside the 5 failure rows it is built from is not "
        f"a measurement of this run: {overstated!r}")
    assert overstated["failed_count"] is None and overstated["ok"] is False, overstated
    assert "warning: unmeasured" in formatters._render_mutation_summary_text(overstated)
    # The two surfaces, on the same payload: the text view states the rows it
    # holds, so no measured summary may state a different number beside it.
    assert "5 failed" in formatters._render_go_rename_text(overstated_value), (
        formatters._render_go_rename_text(overstated_value))
    # The extreme of that direction, and the one an absent listing reaches: a
    # counter naming failures beside NO failure rows at all. The text view
    # states "0 failed" for it.
    bare = {**envelope(_ABSENT), "go_failed_count": 3,
            "success": False, "committed": False}
    assert formatters._go_rename_summary(bare)["measured"] is False, (
        f"go_failed_count 3 with no failure rows anywhere is not a measured "
        f"run: {formatters._go_rename_summary(bare)!r}")
    assert "0 failed" in formatters._render_go_rename_text(bare)


def test_the_blast_radius_line_never_states_a_count_it_could_not_read():
    """The THIRD count surface, and the one still spelled the way `_count_field`
    exists to replace.

    `int(summary.get("referenced") or 0)` had both of that expression's failure
    modes live at once: a string, dict or list RAISED out of the whole mutation
    card (the CLI degrades that to exit 2 with empty stdout, so a mutation that
    COMMITTED reported no card at all), and a bool silently fabricated
    "referenced by 1 fn, 2 reflowed" with no note -- in the same render where
    the op row prints `<count not stated>` and the type row refuses its own
    fabricated zero. One count contract, on every surface that states one."""
    from bn import formatters

    def card(referenced, reflowed=2):
        return formatters._render_mutation_text({
            "success": True, "committed": True,
            "results": [{"op": "types_declare", "status": "verified", "count": 1}],
            "affected_summary": {"referenced": referenced, "reflowed": reflowed}})

    assert "referenced by 4 fns, 2 reflowed" in card(4)
    assert "referenced by" not in card(0), "a measured zero states no blast radius"
    for unreadable in ("many", {"n": 4}, [4], True):
        out = card(unreadable)
        assert "referenced by" not in out, (
            f"referenced={unreadable!r} could not be read, so a blast radius "
            f"derived from it is fabricated: {out!r}")
        assert _disclosed(out, "referenced"), (
            f"referenced={unreadable!r} was refused with no note naming it: {out!r}")


def test_an_op_row_never_states_a_count_the_payload_did_not():
    """#683's harm, stated as the property instead of as one row's wording, and
    the assertion three rounds of this PR shipped a fix without.

    A count is the one thing in these rows a CONTROL LOOP reads: "0 types
    defined" is "nothing landed, do not save", and that reading is what #683
    discarded a committed rename batch to. So a numeral in an op row must be
    traceable to the payload -- either the payload STATED the count, or it
    handed over a readable listing, which is a measurement even when it is
    empty ("we looked, and defined none"). A count derived from a field the
    renderer could not READ is a fabrication, and disclosing it beside the
    number is not enough, because the note is not what the loop reads.

    Enumerated over every op the row handler names -- harvested from the
    module's own AST, so an op that starts stating a count arrives covered --
    crossed with every malformed shape at every key those rows read, on EVERY
    builder that makes a row. Before this, reverting the refusal left all 198
    tests in this file and 827 across the types/mutation/core files green while
    the fabricated zero came back.

    Both halves of the population are harvested, and round 16 proved why each
    has to be. The BUILDERS: `_operation_row` has two callers, and only the
    direct one was swept -- so a fabricated count on the op SUMMARY row, which
    is the row the mutation card prints for every failed op and every multi-op
    batch (#683's own scenario), shipped green. The KEYS: they were a hand
    written tuple of five, two of which no op row reads at all, so a fabrication
    reading any other key was outside the sweep by construction."""
    import ast
    import inspect
    import re

    from bn import formatters

    tree = ast.parse(inspect.getsource(formatters))
    funcs = {fn.name: fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)}
    ops: set[str] = set()
    for node in ast.walk(funcs["_operation_row_text"]):
        if not isinstance(node, ast.Compare):
            continue
        for cmp in node.comparators:
            if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                ops.add(cmp.value)
            elif isinstance(cmp, (ast.Set, ast.Tuple, ast.List)):
                ops.update(elt.value for elt in cmp.elts
                           if isinstance(elt, ast.Constant)
                           and isinstance(elt.value, str))
    assert len(ops) == 11, (
        f"the op row handles {len(ops)} ops, not 11: {sorted(ops)}. The count is "
        "the size of the covered set -- an op that leaves this list is an op no "
        "case below runs.")

    # Every module function that BUILDS an op row: the callers of the shared
    # row helper. Harvested, not listed, so a third wrapper cannot arrive
    # uncovered the way the second one did.
    builders = sorted(
        name for name, fn in funcs.items()
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "_operation_row" for node in ast.walk(fn)))
    assert builders == ["_format_op_summary", "_format_operation_result"], builders

    # Every key those four functions read off the op item, by any spelling --
    # and every key the functions THEY REACH read too. Scoped to the four
    # syntactically, a fabricated count read inside a helper they call was
    # outside the sweep: the key never entered the swept set, so the case was
    # never built. The reach set costs nothing here (it adds three helpers and
    # no new key today) and it means a count hidden one call down arrives
    # covered rather than uncovered.
    readers = {"_field_list", "_field_dict", "_field_present", "_field_declared",
               "_count_field", "_text_value"}
    roots = ("_operation_row", "_operation_row_text", *builders)
    reached = set(roots)
    for root in roots:
        reached |= set(_module_reach().get(root, ()))
    keys: set[str] = set()
    for name in sorted(reached):
        if name not in funcs:
            continue
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "get"
                        and node.args and isinstance(node.args[0], ast.Constant)):
                    keys.add(node.args[0].value)
                elif isinstance(func, ast.Name) and func.id in readers:
                    keys.update(a.value for a in node.args[1:]
                                if isinstance(a, ast.Constant)
                                and isinstance(a.value, str))
            elif (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
    assert len(keys) == 10, f"the op row reads {sorted(keys)}, not 10 keys"

    digits = re.compile(r"\d+")
    fabricated, rendered, checked = [], 0, 0
    for builder_name in builders:
        builder = getattr(formatters, builder_name)
        for op in sorted(ops):
            for key in sorted(keys):
                for bogus in ("bad", ["bad"], {"a": 1}, 0, "", False, {}, [], True):
                    item = {"op": op, key: bogus}
                    rendered += 1
                    out = builder(item)
                    # A readable container IS the measurement, empty or not, so a
                    # count beside one is the payload's own and not a fabrication.
                    if isinstance(bogus, (dict, list)) and key != "count":
                        continue
                    checked += 1
                    stated = set()
                    for value in item.values():
                        stated |= set(digits.findall(str(value)))
                    invented = [n for n in digits.findall(out) if n not in stated]
                    if invented:
                        fabricated.append(
                            f"{builder_name}: {op} with {key}={bogus!r} rendered "
                            f"{out!r}, which states {invented} -- a count the "
                            "payload never did")
    assert not fabricated, fabricated[:6]
    assert (rendered, checked) == (1980, 1188), (
        f"the op-row count sweep RENDERED {rendered} cases, not 1980, and "
        f"CHECKED {checked} of them, not 1188. Two sizes, because they are two "
        "different claims: the carve-out for a readable container skips 792 "
        "renders before any assertion, and pinning only the larger number "
        "overstated the covered set by 40%.")
    # The other half, and the reason this is not a blanket "never print a
    # number": a count the payload DID state must still be stated, or the
    # refusal would be a silent cap on every honest row.
    assert formatters._format_operation_result(
        {"op": "types_declare", "defined_types": "bad", "count": 3}
    ).startswith("types_declare 3 types"), "a stated count must still be stated"
    assert formatters._format_operation_result(
        {"op": "types_declare", "defined_types": {}}) == "types_declare 0 types", (
        "a readable EMPTY listing is a measured zero and must keep rendering as one")
    assert formatters._format_operation_result(
        {"op": "types_declare", "defined_types": {"widget_t": "struct widget_t"}}
    ) == "types_declare widget_t", "a readable listing must name what it defined"


def test_no_list_ELEMENT_costs_the_whole_render():
    """The granularity at which #619 names half its defects, and the one no
    other population here varies.

    Every differential and every sweep in this file varies the FIELD: it puts a
    wrong-shaped value AT a key. A list that arrives as a list of the WRONG
    ELEMENTS is the other half -- `', '.join(names)` on a list of rows, a
    `row["symbol"]` on a list of strings -- and it is what an older bridge
    version actually sends. Those positions were outside every population by
    construction, which left a live raise family the guards could not see: one
    malformed element in a section listing raised TypeError and cost the whole
    listing PLUS the W+X security verdict the renderer prefixes to it.

    The property is the sweep's, not the differential's: an element the
    renderer cannot use may legitimately render as a placeholder or be skipped,
    but it may never RAISE where the same payload with that list absent
    renders cleanly. Base raises 855 times at 30 of its list positions over the
    4536 renders of its own population; this commit raises 0. Sizes are exact,
    for the reason every size here is.

    Swept at FOUR shapes, and that is the eighth axis (see `_payload_for`). A
    one-element list makes its element both the first and the last, so a read
    reached only at a non-terminal or non-initial position -- a separator, an
    `if i != last` tail, a `rows[1:]` slice -- was outside every sweep in this
    file. `[junk, junk]` puts the junk on both sides of every position gate;
    `[junk]` is kept beside it because a single-element list is a real shape too
    and a renderer may only mishandle THAT one. Cardinality two is still not the
    axis: `[junk, junk, junk]` is the first shape that reaches index 2 at all
    (a top-N slice with an "... and N more" tail is exactly that read), and
    `[junk, well_formed, junk]` is the first whose ADJACENT elements differ --
    a gate that is permanently False while every position holds the same value.
    Round 16 proved both live by injection.

    STATED LIMITATION, and the point at which this axis stops being chased.
    Four shapes are four shapes, not a proof over all lists. What they cover is
    exactly: cardinality 1, 2 and 3; the node under test at a first, a last and
    an index >= 2 position; and one adjacent-pair difference on each side of it.
    What they do NOT cover, named rather than implied:

      * a read that opens only at cardinality 4 or more, or at one specific
        length (`len(rows) == 7`, a column-wrapping width);
      * a difference that is NOT between adjacent elements (a first-vs-last
        comparison over a longer list, a "all rows share a prefix" test);
      * a list mixing SEVERAL junk kinds at once -- each shape here is one junk
        kind repeated, so a read gated on two differently-broken neighbours is
        outside it;
      * ordering: the junk is always at index 0 and the tail, never only in the
        middle of a longer list.

    There is no exhaustive static answer here, and pretending otherwise is the
    failure this file keeps correcting: the space is every list shape a bridge
    could send, which is unbounded, so each further shape is another instance
    and not the class. What DOES hold unconditionally is the property, over the
    population the probes discover: no element shape in the covered set costs a
    render that survives the list being absent, on any list position any
    renderer was OBSERVED walking. A reader who needs a guarantee for a shape
    outside that list has to add the shape -- the sizes above are exact so that
    adding one is visible."""
    raised = []
    swept = 0
    for label, render, key, kind, ctx in _runtime_population():
        if kind != "list":
            continue
        absent = _render_or_exception(render, copy.deepcopy(ctx))
        for element in _ELEMENT_JUNK:
            for rows in ([copy.deepcopy(element)],
                         [copy.deepcopy(element), copy.deepcopy(element)],
                         [copy.deepcopy(element), copy.deepcopy(element),
                          copy.deepcopy(element)],
                         [copy.deepcopy(element), dict(_PROBE_SIBLING_ELEMENT),
                          copy.deepcopy(element)]):
                payload = {**copy.deepcopy(ctx), key: rows}
                swept += 1
                out = _render_or_exception(render, payload)
                if isinstance(out, BaseException) and not isinstance(absent, BaseException):
                    raised.append(f"{label}({key}={rows!r}) raised "
                                  f"{type(out).__name__}: {out}")
    assert not raised, (
        "a wrong-shaped list ELEMENT cost the whole render where the same "
        f"payload with the list absent rendered cleanly: {raised[:6]}")
    # 4572 -> 4608 (#797): `_render_defuse_text` now reads the `hints` list, so
    # the element sweep gained one position (1 x 9 junk elements x 4 shapes = 36),
    # measured. The sweep is what proves a wrong-shaped hint element cannot cost
    # the whole def-use card.
    assert swept == 4608, (
        f"the element sweep ran {swept} renders, not 4608 -- the size of the "
        "covered set (every list position the population discovered x every "
        "junk element kind x all four element shapes), so move it only with a "
        "position you deliberately added or removed")


def test_no_NESTED_list_ELEMENT_costs_the_whole_render():
    """The element sweep at the positions the top-level one cannot reach, and the
    last population in this file that covered only its own first level.

    Exactly the repair round 9 made to the container differential, one axis
    over: the top-level element sweep varies the elements of a list AT the
    payload, so a list of wrong elements one or six containers DOWN -- a block's
    `outgoing` edges, a slice's `crossed_functions`, a record row's `fields` --
    was outside every population by construction. Measured at the head that
    added the top-level sweep: 74 live raises across 4 code sites and 10
    (renderer, path) positions, every one of them a `", ".join(...)` or a format
    spec over an element the payload never promised was a string. One such
    element cost the WHOLE function card, where the same payload with that
    nested list absent rendered cleanly.

    Same property as at top level -- an unusable element may render as a
    placeholder or be skipped, never RAISE -- and the same junk set and the same
    FOUR element shapes, shared with it so the two sweeps cannot drift into
    covering different cardinalities."""
    raised = []
    for label, render, path, key, kind, ctx, leaf in _nested_population():
        if kind != "list":
            continue
        base = {k: v for k, v in leaf.items() if k != key}
        absent = _render_or_exception(render, _payload_for(ctx, path, dict(base)))
        for element in _ELEMENT_JUNK:
            for rows in ([copy.deepcopy(element)],
                         [copy.deepcopy(element), copy.deepcopy(element)],
                         [copy.deepcopy(element), copy.deepcopy(element),
                          copy.deepcopy(element)],
                         [copy.deepcopy(element), dict(_PROBE_SIBLING_ELEMENT),
                          copy.deepcopy(element)]):
                out = _render_or_exception(
                    render, _payload_for(ctx, path, {**base, key: rows}))
                if isinstance(out, BaseException) and not isinstance(absent, BaseException):
                    where = ".".join(k for k, *_ in path)
                    raised.append(f"{label}({where}[].{key}={rows!r}) raised "
                                  f"{type(out).__name__}: {out}")
    assert not raised, (
        "a wrong-shaped ELEMENT of a NESTED list cost the whole render where "
        "the same payload with that list absent rendered cleanly: "
        f"{raised[:6]}")



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


# --- ROUTED OUT OF #619/#685, NAMED RATHER THAN LEFT IMPLIED ------------------
#
# TWO findings on this module survive by RULING, not by oversight. Both are
# pre-existing at base, and both are what the follow-up audit deliberately left
# after landing the exhaustive count-and-shape answer #619/#685 routed to it.
#
# What that follow-up landed, so the routing above is a record and not an open
# claim:
#
#   * THE STATED-COUNT CONTRACT. `_stated_count` is `_count_field` for a line
#     that PRINTS the number: the count when it read, `?` when the key arrived
#     in a shape no count reads out of. Applied to every surface that states an
#     ACTIONABLE number -- `_render_go_rename_text`'s three interpolated
#     counters, `_render_instance_gc_text`'s three, and the two resume hints
#     (`_render_data_symbols_text` and the evidence card's call window) that
#     were building `--offset N` out of an unreadable page position.
#   * ITS DIFFERENTIAL, which is the part that cannot go stale:
#     `test_no_renderer_states_a_count_it_could_not_read_as_a_real_number`
#     harvests every `(renderer, literal key)` the module reads through a count
#     helper, asserts that population's EXACT size, and requires the rendered
#     BODY of an unreadable counter to match neither a real 0 nor a real 7.
#     A trailing disclosure note cannot satisfy it, and every pair that leaves
#     the differential is named with the reason it left.
#   * THE RESIDUE INVENTORY. `test_the_raw_count_residue_is_exactly_this_big`
#     counts the raw numeric spellings that remain -- `<lookup> or 0`,
#     `.get(<literal>, 0)`, `int(<lookup>)` -- so a NEW raw count read arrives
#     red instead of joining a prose claim. `_render_class_list_text` (seven)
#     and `_render_data_symbols_text` are its largest members; these are
#     DESCRIPTIVE row/summary counts, not numbers a caller acts on, which is
#     why the contract above reaches the actionable ones and the inventory
#     bounds the rest.
#   * THE STRING-SHAPE POPULATION'S ALIAS BLINDNESS, which was the second
#     routed item: `_classify_string_shape_guards` now resolves a module-level
#     name bound to `str` (or to a tuple containing it), so a guard spelled
#     `isinstance(x, _TEXT_TYPES)` is harvested and classified like any other.
#     `test_the_string_shape_population_harvests_a_guard_named_through_an_alias`
#     proves it on a synthetic module carrying that spelling; the old
#     "the module may not CONTAIN this spelling" assertion is retired BY the
#     fix. `type(x) is str` is still outside the classifier and still asserted
#     absent; a duck-typed `try: x.strip()` remains undetectable.
#
# What it deliberately left, and why -- these are the two live items:
#
#   1. FLAG AND ARITHMETIC SURFACES have the shape contract only where #619's
#      harm reached them. `_flag_field` exists because `has_more` decides
#      whether the paging footer states a resume instruction, and a raw
#      truthiness test on a flag reads `"false"` as True; every OTHER flag read
#      in this module (`preview`, `committed`, `truncated`, `changed`,
#      `direct`, `has_more` on the renderers that do their own paging) is still
#      `bool(value.get(k))` or a bare `value.get(k)`, which is the same class
#      one field over. Likewise the arithmetic: the footer refuses counts
#      that contradict each other, while `_go_rename_summary`'s
#      `op_count = candidates + skipped - skipped_changed_during_apply` can
#      still go negative on an inconsistent envelope. Both were left by the
#      count-and-shape follow-up on purpose: its subject was the numbers a
#      caller ACTS on, and the honest version of these two is one contract over
#      every flag read and every derived count, discovered by a population
#      rather than listed by hand.
#   2. THE PAGING FOOTER'S IMPOSSIBILITY REFUSAL IS PINNED ON THE COUNTS'
#      VALUES, NOT ON WHICH KEYS CARRY THEM. This one is not pre-existing at
#      base -- it is #619's own mechanism -- and it is routed for the
#      other reason: three consecutive rounds widened this pin, each closing
#      the axis the last lens named and leaving the next, with HEAD
#      behaviourally CORRECT on every one of them. What is routed is the PIN,
#      stated here as a limit rather than pinned a fourth time.
#      `test_the_paging_footer_refuses_every_impossible_page_it_names`
#      exercises four count triples across three paging shapes, and every one
#      of those payloads states `total`, `returned` AND `offset` as literal
#      keys, none with a total of 0. So each of these narrowings of
#      `if impossible:` in `_paging_footer` leaves this file GREEN -- and
#      `tests/test_cli_mutation.py` with it -- while restoring the
#      fabrication. No pass COUNT is quoted here on purpose: a number would go
#      stale the next time a test is added and become one more false claim in
#      a file that has corrected several. A reader may run them:
#        * `if impossible and _field_present(value, "returned"):` --
#          `{functions: [row], total: 10, offset: 50, has_more: True}` omits
#          `returned` (which then falls back to the item count, the default
#          path the helper itself codes) and renders
#          `// showing 1 of 10 (-41 more); rerun with --offset 51 or a larger
#          --limit` again, where HEAD refuses with `// page position not
#          stated: offset + returned exceeds total (total 10, returned 1,
#          offset 50)`.
#        * `if impossible and total:` -- `{functions: [row], total: 0,
#          returned: 1, offset: 0, has_more: False}` renders
#          `// showing 1 of 0`, where HEAD refuses with `// page position not
#          stated: offset + returned exceeds total (total 0, returned 1,
#          offset 0)`.
#        * swapping the `returned is negative` and `offset is negative`
#          members reorders the rendered line -- cosmetic, both names still
#          appear, and nothing pins the order.
#      Closing these by enumeration makes the trigger set a product of count
#      VALUES x key PRESENCE x paging SHAPE, which is the hand-written table
#      this file has already deleted twice; a discovered population is the
#      routed audit's answer, and a limit that names the exact mutation to try
#      is worth more than a fourth pin that closes one more corner of it.


def test_the_string_shape_population_harvests_a_guard_named_through_an_alias():
    """The classifier used to require a LITERAL `str` in the class argument, so
    a guard spelled through a module-level alias (`_TEXT_TYPES = (str,)`;
    `isinstance(x, _TEXT_TYPES)`) was outside the population AND outside its
    anti-drift assertion -- it could drop a text field with nothing in this file
    able to see it. The limitation was papered over by asserting the module does
    not CONTAIN the spelling, which is bookkeeping for a blindness rather than a
    fix.

    Now the classifier resolves the alias, so the claim is about the RULE and is
    put to a synthetic module that carries the spelling on purpose: the guard
    must appear in the population, and a dropping one must classify as `DROPS`.
    A classifier that skipped it would report that module clean.

    `type(x) is str` keeps its own absence assertion below; it is still outside
    the classifier, and a duck-typed `try: x.strip()` is not detectable at all.
    """
    source = (
        "_TEXT_TYPES = (str,)\n"
        "_ALSO_TEXT = _TEXT_TYPES\n"
        "\n"
        "def _render_probe(value):\n"
        "    note = value.get('note')\n"
        "    if isinstance(note, _TEXT_TYPES):\n"
        "        return note\n"
        "    return ''\n"
        "\n"
        "def _render_probe_alias_of_alias(value):\n"
        "    label = value.get('label')\n"
        "    if isinstance(label, _ALSO_TEXT):\n"
        "        return label\n"
        "    return ''\n"
    )
    guards = _classify_string_shape_guards(source)

    assert ("_render_probe", "note", 0) in guards, (
        "a guard whose class is a module-level alias of `str` is outside the "
        f"population, so a drop behind it is invisible: {sorted(guards)}")
    assert guards[("_render_probe", "note", 0)] == "DROPS", guards
    assert guards[("_render_probe_alias_of_alias", "label", 0)] == "DROPS", (
        "an alias of an alias is the same guard one hop further out")

    # The literal spelling still classifies identically, so resolving aliases
    # did not move the rule for the module's own guards.
    literal = _classify_string_shape_guards(
        source.replace("_TEXT_TYPES)", "str)").replace("_ALSO_TEXT)", "str)"))
    assert literal == guards


def test_the_paging_footer_never_states_a_resume_offset_it_could_not_derive():
    """The footer names three counts and refused on only one of them.

    `total` arriving unreadable drops the footer and discloses. `returned` and
    `offset` reach ARITHMETIC through the same choke point, and a skew there
    was absorbed into the fabricated 0 the choke point returns -- so the footer
    went on to state a resume instruction DERIVED from the fabrication. Two
    shapes of harm, both worse than the dropped footer:

      * `offset` unreadable -> "showing 1 of 100 (99 more); rerun with --offset
        1" on a page that actually started at 50: a pager re-reads the window
        it already has and never reaches the tail;
      * `returned` unreadable -> "--offset 0" with `has_more` true, which does
        not ADVANCE: an unattended pager loops forever on page one;
      * and the FOURTH count, which is not one of the three and is the reason
        the first cut of this refusal was still incomplete: the PAGE ITSELF.
        `returned` falls back to the length of the item list, which is a
        measurement only while the page was readable -- the choke point hands
        back an empty list for a page it could not use, so an unreadable
        `functions`/`items` key produced "showing 0 of 100 (50 more); rerun
        with --offset 50" from an offset of 50: the same non-advancing loop,
        reached without any of the three counts being wrong.

    A number in this footer is the one thing in it a loop acts on, so the
    refusal has to be the same one the count claims already make -- state
    nothing rather than state a count nobody could derive, and let the boundary
    name the field. ONE count contract for all three, which is what the
    helper's own comment already claimed, plus the page the third of them is
    measured off."""
    from bn import formatters

    def page(**over):
        value = {"functions": [{"name": "a", "address": "0x1"}],
                 "total": 100, "returned": 1, "offset": 50, "has_more": True}
        for key, val in over.items():
            if val is _ABSENT:
                value.pop(key, None)
            else:
                value[key] = val
        return formatters._render_function_list_text(value)

    # Anti-vacuity: the honest page must still page, and the resume offset must
    # still be the NEXT window -- a footer that always refused would satisfy
    # every refusal below.
    assert "// showing 1 of 100 (49 more); rerun with --offset 51" in page(), page()
    # `returned` absent is a MEASUREMENT off the rows, not a fabrication, and
    # stays exactly as it was.
    assert "rerun with --offset 51" in page(returned=_ABSENT), page(returned=_ABSENT)
    # A last page states its position without a resume instruction.
    assert "// showing 1 of 100" in page(has_more=_ABSENT)
    for unreadable in ("many", {"n": 1}, [1], True, ()):
        # `functions` is the PAGE key here: the fourth count, reached through
        # `returned`'s item-count default rather than through any of the three.
        # A LIST is a readable page whatever its elements hold, so `[1]` is not
        # one of its unreadable shapes -- see the element case below.
        keys = ("total", "returned", "offset")
        for key in (keys if isinstance(unreadable, list) else keys + ("functions",)):
            out = page(**{key: unreadable})
            assert "--offset" not in out, (
                f"{key}={unreadable!r} could not be read, so the resume offset "
                f"is derived from a fabricated count: {out!r}")
            assert "showing" not in out, (
                f"{key}={unreadable!r} could not be read, so the page's own "
                f"counts are fabricated too: {out!r}")
            assert _disclosed(out, key), (
                f"{key}={unreadable!r} cost the footer with no note naming "
                f"it: {out!r}")
            # The same refusal on the LAST page, where the footer states counts
            # with no resume instruction to hide behind.
            tail = page(has_more=_ABSENT, **{key: unreadable})
            assert "showing" not in tail, (
                f"{key}={unreadable!r} on a last page still states a count it "
                f"could not read: {tail!r}")
            assert _disclosed(tail, key), tail
    # The rows themselves are untouched by the refusal: dropping the footer is
    # not dropping the page.
    assert "0x1  a" in page(offset={"n": 1})
    # The ELEMENT case, and the line between the two: a list of one junk
    # element IS a readable page of one row -- the element renders visibly, so
    # "showing 1 of 100" is a measurement and not a fabrication. Refusing here
    # would be the mirror error, a page silently losing its resume hint over a
    # row the reader can see.
    element = page(functions=[1])
    assert "// showing 1 of 100 (49 more); rerun with --offset 51" in element, element
    assert "1" in element.splitlines()[0], element
    # The flag that decides whether a resume instruction exists at all, which
    # had no shape contract while the three counts beside it had one -- and it
    # is the one field a raw truthiness test reads BACKWARDS: `"false"` is True
    # to Python, so a LAST page printed "rerun with --offset 51" and sent a
    # pager after a window that does not exist.
    for bogus in ("false", "no", {"more": False}, ["x"], 1.5, ()):
        out = page(has_more=bogus)
        assert "--offset" not in out and "showing" not in out, (
            f"has_more={bogus!r} could not be read, so whether more rows exist "
            f"is not established and no resume instruction follows from it: {out!r}")
        assert _disclosed(out, "has_more"), out
    # A real bool, and the 0/1 a wire format may number its booleans with, both
    # still READ -- refusing those would drop the footer on every honest page.
    assert "rerun with --offset 51" in page(has_more=1)
    assert "// showing 1 of 100" in page(has_more=0)
    assert "malformed" not in page(has_more=0)
    # Counts that are all readable and mutually impossible: a window starting
    # past the end of the set derived "(-41 more)" and "--offset -4", which is
    # an actionable instruction built out of arithmetic on a self-contradicting
    # envelope. The payload's own numbers are stated; nothing is derived.
    clash = page(total=10)
    assert ("// page position not stated: offset + returned exceeds total "
            "(total 10, returned 1, offset 50)") in clash, clash
    assert "--offset" not in clash and "more)" not in clash, clash
    # ... and the boundary case is NOT a clash: the last window ends exactly at
    # the total.
    assert "// showing 1 of 100" in page(offset=99, has_more=_ABSENT)


def test_an_unreadable_results_ROW_can_never_read_as_ok():
    """#683's harm in the `ok` field, one granularity in from where round 18
    fixed it.

    `results` arriving as the WRONG FIELD (a string, a dict) is a skew, and
    both `ok` derivations withhold on it. An ELEMENT of a well-formed `results`
    list that is not a row was silently DROPPED by an `isinstance(r, dict)`
    filter at every site that reads it -- so a batch carrying one row nobody
    could classify beside one that verified reported `ok: true`,
    `failed_count: 0`, `first_error: null` and a text card showing only the
    readable row, with nothing anywhere saying a row had been discarded. A
    control loop reads that and closes.

    This is the same three-state question `_is_failed_status` answers for a
    row's STATUS -- FAILED / not-failed / UNREADABLE -- asked about the ROW
    itself, so it gets the same answer and the two must not disagree: an
    unreadable row is not a row that passed. Dropping the element is still the
    right RENDER (there is nothing in it to show), but it is a skew, and the
    counts derived from the surviving rows are no longer a measurement of the
    batch -- `measured` goes False for exactly the reason an unreadable
    `results` FIELD already makes it False."""
    from bn import formatters

    good = {"op": "rename", "function": "fn", "status": "verified"}

    def batch(rows):
        return {"success": True, "committed": True, "results": rows}

    def go(rows):
        value = {"kind": "go_rename", "success": True, "committed": True,
                 "go_renamed_candidates": 3, "go_committed_count": 3,
                 "go_verified_count": 3, "go_failed_count": 0,
                 "skipped_user_named": 0, "skipped_changed_during_apply": 0}
        if rows is not _ABSENT:
            value["results"] = rows
        return value

    # Anti-vacuity: readable rows still pass, and an EMPTY list is still the
    # measured zero it always was -- the refusal must not become a blanket
    # "never claim ok".
    assert formatters._add_mutation_ok(batch([good]))["ok"] is True
    ok_summary = formatters._mutation_summary(batch([good]))
    assert (ok_summary["ok"], ok_summary["measured"], ok_summary["failed_count"]) \
        == (True, True, 0), ok_summary
    assert formatters._go_rename_summary(go(_ABSENT))["ok"] is True
    assert formatters._go_rename_summary(go([]))["ok"] is True
    # `None` is in this set deliberately: an explicit null FIELD claimed
    # nothing, but a null INSIDE the row list is a row position the sender
    # filled with no row -- an op of the batch this summary cannot account for.
    for junk in ("<unreadable row>", ["rename"], 7, True, 1.5, (), None):
        payload = batch([junk, dict(good)])
        summary = formatters._mutation_summary(payload)
        verbose = formatters._add_mutation_ok(payload)
        assert summary["ok"] is False and summary["success"] is False, (
            f"results=[{junk!r}, <verified row>] carries a row nobody could "
            f"classify, so 'nothing failed' is not established: {summary!r}")
        assert verbose["ok"] is False, (
            f"results=[{junk!r}, ...] claimed ok on the verbose path: {verbose!r}")
        assert summary["ok"] is verbose["ok"], (
            f"`jq '.ok'` flips with --summary on results=[{junk!r}, ...]")
        assert summary["measured"] is False and summary["failed_count"] is None, (
            f"the surviving rows are not a measurement of a batch one of whose "
            f"rows was discarded: {summary!r}")
        assert "malformed" in str(summary.get("first_error")), summary
        card = formatters._render_mutation_text(payload)
        assert "fn" in card, f"the READABLE row must still render: {card!r}"
        assert _disclosed(card, "results"), (
            f"a discarded row left the card with no note naming the field it "
            f"was discarded from: {card!r}")
        # The other caller of the one builder, on the key it reads for its
        # FAILURE rows.
        gr = formatters._go_rename_summary(go([junk]))
        assert gr["ok"] is False and gr["measured"] is False, (
            f"go rename's failure list held {junk!r}, which is not a row, so "
            f"'nothing failed' is not established either: {gr!r}")
        text = formatters._render_go_rename_text(go([junk]))
        assert _disclosed(text, "results"), (
            f"the verbose go-rename view discarded a row silently: {text!r}")
    # The row-level decider and the element-level one must agree rather than
    # cancel: a batch with BOTH an unreadable row and an unreadable status on a
    # surviving row withholds ok once, names both fields, and never reads as a
    # pass.
    mixed = batch([7, {"op": "rename", "function": "fn", "status": {"code": 7}}])
    assert formatters._mutation_summary(mixed)["ok"] is False
    assert formatters._add_mutation_ok(mixed)["ok"] is False
    card = formatters._render_mutation_text(mixed)
    assert _disclosed(card, "results") and _disclosed(card, "status"), card


def test_the_paging_footer_refuses_every_impossible_page_it_names():
    """The mutual-impossibility refusal was incomplete on its own stated terms.

    Round 22 added it and spelled it as ONE comparison, `offset + returned >
    total`, while the comment beside it quoted "a negative remaining count and
    a negative `--offset -4`" as the harm it exists to stop. A readable
    NEGATIVE count walks straight past a single comparison: `returned: -5`
    against a total of 100 stated "showing -5 of 100 (105 more); rerun with
    --offset -5" -- 105 more rows in a set of 100, and a resume offset no pager
    can use. Every count on that line is readable, so no SHAPE refusal fires
    and nothing is disclosed either.

    The refused conditions are a SET, so the module states them as one
    (`_IMPOSSIBLE_PAGE`) and this test enumerates that set rather than
    re-listing it: a member that stops refusing, or one deleted as redundant
    because another member happens to catch the same payload, fails here BY
    NAME."""
    from bn import formatters

    def page(**over):
        value = {"functions": [{"name": "a", "address": "0x1"}],
                 "total": 100, "returned": 1, "offset": 50, "has_more": True}
        for key, val in over.items():
            if val is _ABSENT:
                value.pop(key, None)
            else:
                value[key] = val
        return formatters._render_function_list_text(value)

    # The two shapes a single `offset + returned > total` comparison lets
    # through, with every count readable in both.
    negative_returned = page(returned=-5, offset=0)
    assert "more)" not in negative_returned and "--offset" not in negative_returned, (
        f"a readable returned=-5 still derived a remaining count exceeding the "
        f"total and a resume offset out of it: {negative_returned!r}")
    assert "showing" not in negative_returned, negative_returned
    negative_offset = page(offset=-9)
    assert "--offset" not in negative_offset, (
        f"a readable offset=-9 still derived a resume offset: {negative_offset!r}")

    # One payload per member of the set, each making that member's condition
    # hold. The refusal must NAME the member, so weakening or deleting it fails
    # here even when a different member still refuses the same payload.
    triggers = {
        "total is negative": dict(total=-1, returned=0, offset=0),
        "returned is negative": dict(total=100, returned=-5, offset=0),
        "offset is negative": dict(total=100, returned=1, offset=-9),
        "offset + returned exceeds total": dict(total=10, returned=1, offset=50),
    }
    assert set(triggers) == {name for name, _ in formatters._IMPOSSIBLE_PAGE}, (
        f"the refused set and the payloads that exercise it have drifted: "
        f"{[name for name, _ in formatters._IMPOSSIBLE_PAGE]}")
    # ... across every paging SHAPE, because the refusal is a property of the
    # COUNTS and not of the paging state. Every payload above carries
    # `has_more: True` and one item, so a refusal narrowed to
    # `if impossible and more:` -- or to `and items:` -- would state the
    # fabricated footer again on a last page or an empty one with the pin
    # still green.
    shapes = {"paging": {},
              "last page": {"has_more": _ABSENT},
              "empty page": {"has_more": _ABSENT, "functions": []}}
    for name, over in triggers.items():
        for shape, extra in shapes.items():
            out = page(**over, **extra)
            assert "// page position not stated:" in out, (
                f"{over} on a {shape} is not a window any page can have, and "
                f"the footer stated a position for it anyway: {out!r}")
            assert name in out, (
                f"{over} on a {shape} was refused without naming `{name}`: {out!r}")
            assert "--offset" not in out and "more)" not in out and "showing" not in out, out
            # The payload's own numbers are stated; nothing is derived from them.
            assert (f"total {over['total']}" in out
                    and f"returned {over['returned']}" in out
                    and f"offset {over['offset']}" in out), out
            # Refusing the footer is not refusing the page.
            if shape != "empty page":
                assert "0x1  a" in out, out
    # TWO conditions at once, which is what the set exists for: a member is not
    # excused because a sibling catches the same payload, so the line names
    # BOTH. Collapsing the join to the first name alone fails here.
    both = page(total=-1, returned=0, offset=0)
    assert ("// page position not stated: total is negative; offset + returned "
            "exceeds total (total -1, returned 0, offset 0)") in both, both
    # Anti-vacuity: a refusal that fired on every page would satisfy all of the
    # above. The honest page still pages, and each condition's BOUNDARY is a
    # page rather than a clash.
    assert "// showing 1 of 100 (49 more); rerun with --offset 51" in page()
    assert "// showing 1 of 100" in page(offset=99, has_more=_ABSENT)
    empty = page(functions=[], total=0, returned=0, offset=0, has_more=_ABSENT)
    assert "page position not stated" not in empty, (
        f"a genuinely empty set is a readable page, not an impossible one: {empty!r}")


def test_a_rows_own_skew_never_costs_the_page_its_footer():
    """The footer's nested capture, which nothing in this file executed.

    `_paging_footer` reads its four fields inside a FRESH `_SKEWED_FIELDS`
    capture so that only ITS OWN reads can suppress the footer. The ambient
    list holds every skew the whole render recorded -- including one from a
    ROW, on a key with nothing to do with paging -- and a footer dropped over
    an unrelated field's skew is the mirror of the fabrication the refusal
    exists to stop: the page silently loses the resume instruction that is the
    only actionable thing in it, and a reader takes one window for the whole
    set.

    Seeding that capture from the ambient list instead of an empty one left
    the entire file green, so the mechanism shipped with no discriminating
    test at all -- this PR's own recurring defect, in the hunk added to close
    the last generation of it."""
    from bn import formatters

    row = {"address": "0x1", "value": "hi", "length": 2,
           "format_directives": "oops"}

    def strings(**over):
        return formatters._render_strings_text(
            {"items": [row], "total": 100, "returned": 1, "offset": 50,
             "has_more": True, **over})

    out = strings()
    assert "// showing 1 of 100 (49 more); rerun with --offset 51" in out, (
        f"a row's own malformed `format_directives` reached the footer's read "
        f"and cost the page its resume instruction: {out!r}")
    # Keeping the footer is not swallowing the row's problem.
    assert _disclosed(out, "format_directives"), out
    # The counterpart the nesting must NOT swallow: a skew on one of the
    # footer's OWN fields still refuses, from inside the same capture, with the
    # unrelated row skew sitting in the ambient list beside it.
    skewed = strings(total={"n": 1})
    assert "showing" not in skewed and "--offset" not in skewed, skewed
    assert _disclosed(skewed, "total"), skewed
    assert _disclosed(skewed, "format_directives"), skewed


def test_the_compact_first_error_never_states_a_container_where_text_belongs():
    """`first_error` is the one summary key an agent contract tells a control
    loop to read, and the top-level `message` is what lands in it whenever the
    payload carried no failure ROW -- a revert that failed after every op
    verified, an op that explained itself only at the envelope.

    A raw `value.get("message")` put a dict/list straight into the documented
    schema and `_render_mutation_summary_text` printed its Python repr there.
    The failure ROW's message already went through the text choke point; the
    envelope's did not, and the asymmetry is invisible until the envelope is
    the only explanation available. Both callers of the one builder are pinned
    here, because "one builder" means neither caller may answer differently."""
    from bn import formatters

    def mutation(message):
        return {"success": False, "committed": False, "message": message,
                "results": [{"op": "rename", "function": "fn",
                             "status": "verified"}]}

    def go(message):
        return {"kind": "go_rename", "success": False, "committed": False,
                "message": message, "go_renamed_candidates": 1,
                "go_committed_count": 0, "go_verified_count": 0,
                "go_failed_count": 0, "skipped_user_named": 0,
                "skipped_changed_during_apply": 0}

    for bogus in ({"code": 7}, ["boom"], 7, True, 1.5, ()):
        for payload, default in ((mutation(bogus), "mutation failed"),
                                 (go(bogus), "go rename failed")):
            summary = (formatters._go_rename_summary(payload)
                       if payload.get("kind") == "go_rename"
                       else formatters._mutation_summary(payload))
            first_error = summary["first_error"]
            assert isinstance(first_error, str), (
                f"message={bogus!r} landed in the documented first_error "
                f"unchanged: {first_error!r}")
            assert repr(bogus) not in first_error, (
                f"message={bogus!r} reached first_error as a Python repr: "
                f"{first_error!r}")
            assert default in first_error, (
                f"message={bogus!r} explained nothing, so the op's own default "
                f"explanation had to answer instead: {first_error!r}")
            assert _disclosed(first_error, "message"), (
                f"message={bogus!r} was dropped with no note naming it: "
                f"{first_error!r}")
            text = formatters._render_mutation_summary_text(summary)
            assert repr(bogus) not in text, text
    # Anti-vacuity: a readable envelope message is still the explanation, and
    # is not disclosed as a shape problem.
    readable = formatters._mutation_summary(mutation("revert failed"))
    assert readable["first_error"] == "revert failed", readable
    assert "malformed" not in str(readable["first_error"]), readable
    # An EMPTY message is a real answer that explains nothing, so the default
    # answers and nothing is disclosed.
    blank = formatters._mutation_summary(mutation(""))
    assert blank["first_error"] == "mutation failed", blank


def test_the_go_rename_ok_key_reads_the_same_on_both_cli_paths(fake_transport, capsys):
    """`jq '.ok'` is #447's whole promise, and on this op it was one flag deep.

    The CLI picks the result transform from the DETAIL flags: an explicit
    machine `--format` (or `--verbose`, or `--out`) selects `_add_mutation_ok`,
    anything else selects the op's compact status. `_add_mutation_ok` derives
    `ok` from `success` and `results[]` -- and `go rename`'s `results[]` holds
    the FAILURE rows alone, so an unreadable `go_*` counter never reached it
    and the detail path answered `ok: true` on the payload the compact status
    -- the CLI's default for this op since #645 -- refuses. An unattended loop
    that pipes `--format json` into `jq '.ok'` reads true and closes without
    saving: #683's discard, reached through the one key #447 added to prevent
    it.

    Asserted on what a CALLER OBSERVES rather than on two functions returning
    the same value. `ok` now has one decider for this op, so comparing the two
    transforms directly would be true by construction and could not fail;
    running the two CLI invocations still can, because the CLI is free to
    re-split which transform each path gets."""
    envelope = {"kind": "go_rename", "success": True, "committed": True,
                "preview": False, "results": [],
                "go_renamed_candidates": 3, "go_committed_count": 3,
                "go_verified_count": 3, "go_failed_count": 0,
                "skipped_user_named": 0, "skipped_changed_during_apply": 0}

    def observed(result, extra_argv):
        fake_transport({"go_rename": {"ok": True, "result": dict(result)}})
        argv = ["go", "rename", "--target", "active", "--format", "json"]
        rc = bn.cli.main(argv + extra_argv)
        return rc, json.loads(capsys.readouterr().out)["ok"]

    cases = {
        "healthy": (envelope, True),
        # The counter channel, which the detail path could not see at all.
        "unreadable counter": ({**envelope, "go_verified_count": {"n": 1}}, False),
        "unreadable candidates": ({**envelope, "go_renamed_candidates": "many"}, False),
        # The counter and its own rows disagreeing, in both directions.
        "counter under-states": ({**envelope, "success": False, "committed": False,
                                  "results": [{"status": "verification_failed"}] * 2},
                                 False),
        "counter over-states": ({**envelope, "go_failed_count": 2}, False),
        # A ROW whose status nobody could classify: not a row that passed, and
        # the shapes the two deleted same-function parity assertions covered.
        # A container status is deliberately NOT among them: `_mutation_exit_code`
        # (src/bn/cli.py, outside this PR's fence, byte-identical at base) does
        # `item.get("status") in FAILED_MUTATION_STATUSES` on the raw value and
        # raises `TypeError: unhashable type: 'dict'` before any output, so that
        # shape cannot be observed through the CLI at all. Reported upward as an
        # out-of-fence finding rather than worked around here.
        "unreadable row status": ({**envelope, "results": [{"status": 7}]}, False),
        "non-dict row": ({**envelope, "results": [7]}, False),
        # The failure list itself unreadable.
        "unreadable results": ({**envelope, "results": "boom"}, False),
        # A STALE `ok` already on the envelope. The compact path overrides it
        # unconditionally, so the detail path must too -- left behind the
        # already-has-`ok` short-circuit this was the one input on which the
        # one decider was still two.
        "stale ok beside an unreadable counter": (
            {**envelope, "ok": True, "go_failed_count": {"n": 1}}, False),
        "stale ok on a healthy run": ({**envelope, "ok": False}, True),
    }
    for name, (payload, expected) in cases.items():
        detail_rc, detail_ok = observed(payload, [])
        compact_rc, compact_ok = observed(payload, ["--summary"])
        assert detail_ok is compact_ok, (
            f"{name}: `jq '.ok'` reads {detail_ok!r} on `--format json` and "
            f"{compact_ok!r} on `--format json --summary`, so a control loop's "
            f"verdict depends on whether it asked for detail")
        assert detail_ok is expected, (
            f"{name}: both CLI paths agree on {detail_ok!r}, but the payload "
            f"establishes {expected!r} -- agreeing on the wrong answer is not "
            f"the property")
        assert detail_rc == compact_rc, (
            f"{name}: exit code differs across the detail flag: "
            f"{detail_rc} vs {compact_rc}")


@pytest.mark.parametrize(
    "bad", [{"name": "x"}, 0, 1, True, ["x"]],
    ids=["dict", "zero", "int", "bool", "list"],
)
def test_render_trace_text_discloses_a_wrong_shaped_callee_755(bad):
    """#858 review round-2 minor: the top-level `callee` read was a bare `.get`,
    so a wrong-shaped value rendered straight into the header -- a dict became
    the raw Python repr `arg[0] of {'name': 'x'}`, and a number read as
    "unresolved" in silence. Routed through `_text_value`, the module's string
    sibling of `_field_list`/`_field_dict`/`_count_field`, so a key present in a
    shape no name reads out of is DISCLOSED rather than interpolated."""
    from bn.formatters import _render_trace_text
    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {"index": 0}, "callee": bad, "trace": [],
    })
    assert "malformed callee field" in out
    # Never the value itself, and never a silent claim of a resolved name.
    assert "{'name'" not in out and "['x']" not in out
    assert "of <unresolved callee>" in out


@pytest.mark.parametrize(
    "bad", [{"reg": "rdi"}, 0, 7, True, ["rdi"]],
    ids=["dict", "zero", "int", "bool", "list"],
)
def test_render_trace_text_discloses_a_wrong_shaped_register_755(bad):
    """#858 review round 3 MAJOR: the `register` read one line below the repaired
    `callee` read was still a bare `.get`, interpolating a wrong-shaped value
    straight into the header (`arg[0] ({'reg': 'rdi'})`) with no disclosure. Same
    reader, same rule -- and the PR body's claim that `arg_label` had exactly one
    consumer, so there was no sibling to fix, was wrong: this was the sibling."""
    from bn.formatters import _render_trace_text
    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {"index": 0, "callee": "memcpy", "register": bad},
        "trace": [],
    })
    assert "malformed register field" in out
    assert "{'reg'" not in out and "['rdi']" not in out
    # The rest of the header still renders: one unusable field must not cost the
    # callee name it sits beside.
    assert "backward trace of arg[0] of memcpy in f @ 0x1010" in out


def test_render_trace_text_records_a_skew_on_either_callee_key_755():
    """#858 review round 3 minor: `_text_value(a) or _text_value(b)` short-circuits,
    so a skew on the SECOND key went unrecorded whenever the first was truthy --
    the alias trap `_field_list`'s docstring names. Both keys are read."""
    from bn.formatters import _render_trace_text
    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "arg_label": {"index": 0, "callee": "memcpy"},
        "callee": {"name": "shadow"},          # skewed, and second in precedence
        "trace": [],
    })
    assert "malformed callee field" in out
    # The usable name still wins, and the unusable one never reaches the header.
    assert "of memcpy in f" in out and "shadow" not in out


@pytest.mark.parametrize(
    "bad", [{"a": 1}, ["x"], True, float("nan"), "abc"],
    ids=["dict", "list", "bool", "nan", "non-numeric-string"],
)
def test_render_trace_text_discloses_a_wrong_shaped_arg_index_755(bad):
    """#858 review round 5: `arg_index` was the THIRD component of this one arg
    descriptor still read with a bare `.get`, after `callee` (round 2) and
    `register` (round 4). A wrong-shaped value was interpolated raw --
    `backward trace of arg[{'a': 1}] of memcpy in f @ 0x1010` -- with no
    disclosure. Routed through `_stated_count`, the int sibling of `_text_value`:
    the skew is recorded for the enclosing boundary and the slot states `?`."""
    from bn.formatters import _render_trace_text
    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": bad, "arg_label": {"index": 0, "callee": "memcpy"}, "trace": [],
    })
    assert "malformed arg_index field" in out
    assert "backward trace of arg[?] of memcpy in f @ 0x1010" in out
    # The value itself never reaches the header.
    for leak in ("{'a'", "['x']", "nan", "abc"):
        assert f"arg[{leak}]" not in out


def test_render_trace_text_states_a_readable_arg_index_755():
    """The readable side, so the reader swap cannot quietly turn every index into
    `?`: an int renders as itself, and a numeric string still reads (the
    `_count_field` contract) rather than being disclosed as malformed."""
    from bn.formatters import _render_trace_text
    base = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_label": {"index": 2, "callee": "memcpy"}, "trace": [],
    }
    assert "arg[2] of memcpy" in _render_trace_text(dict(base, arg_index=2))
    assert "arg[2] of memcpy" in _render_trace_text(dict(base, arg_index="2"))
    assert "malformed arg_index" not in _render_trace_text(dict(base, arg_index=2))
    # Absent is a real default, not a skew: the op always sends one, and a
    # missing key must not manufacture a disclosure.
    assert "arg[0] of memcpy" in _render_trace_text(base)
    assert "malformed arg_index" not in _render_trace_text(base)


# --- #866: a count is stated only when the value IS that integer -------------
# `_count_field` read with a bare `int(raw)`, so every shape carrying a fraction
# was TRUNCATED into a confident integer the payload never stated (`arg[1]` for
# `1.5`) with no disclosure -- the undisclosed-raw-repr half of the same seam was
# closed by routing `arg_index` through the choke point, this is the other half.
# The values below are the shapes a forward-compat / hand-built / third-party
# payload can spell; `_MALFORMED`-style junk is already covered elsewhere.
_NON_INTEGRAL_COUNTS = [
    (1.5, "1", "float"),
    (Decimal("2.5"), "2", "decimal"),
    (Fraction(7, 2), "3", "fraction"),
    (b"1", "1", "bytes"),
    (bytearray(b"2"), "2", "bytearray"),
]


@pytest.mark.parametrize("stated,truncated,label", _NON_INTEGRAL_COUNTS,
                         ids=[c[2] for c in _NON_INTEGRAL_COUNTS])
def test_count_field_refuses_a_value_that_is_not_that_integer_866(stated, truncated, label):
    """#866: a count read must not REWRITE the number it was handed.

    `1.5` -> `arg[1]` is worse than the `?` the same reader gives a dict: the
    truncated value is a plausible index nobody stated, so an agent cannot tell
    the payload disagreed with the header. `_count_field`'s own docstring calls
    anything that is not a plain integer a skew, and the header must say so.
    """
    from bn.formatters import _count_field, _render_trace_text

    reviewed = _count_field({"count": stated}, "count")
    assert reviewed == 0, f"{label}: {stated!r} read as the count {reviewed}"

    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": stated, "arg_label": {"index": 0, "callee": "memcpy"},
        "trace": [],
    })
    assert "malformed arg_index field" in out, out
    assert "backward trace of arg[?] of memcpy in f" in out, out
    # Neither the truncated index nor the value itself reaches the header.
    assert f"arg[{truncated}]" not in out, out
    assert f"arg[{stated!r}]" not in out, out


def test_count_field_reads_an_integral_number_866():
    """The other direction, so the strictness cannot silently turn every number
    into a skew: a value that IS the integer reads as it, and EVERY honest way a
    producer can spell one as text reads too.

    #866 review: the first cut compared against the canonical rendering of the
    integer, which re-rejected spellings base read correctly (`"+2"`, `"02"`,
    `"0002"`, `"-02"`, `"+0"`, `"1_0"`) and a text spelling of the integral VALUE
    (`"2.0"`). The contract is "the payload stated this integer", not "the payload
    rendered it the way Python would" -- a producer that pads, signs or zero-fills
    a count is stating it, and refusal here costs a line of real output."""
    from bn.formatters import _count_field, _field_skewed, _render_trace_text

    for stated in (2, 2.0, Decimal("2"), Decimal("2.0"), Fraction(4, 2), "2",
                   "+2", "02", "0002", "2.0", "2e0", " 2 "):
        assert _count_field({"count": stated}, "count") == 2, repr(stated)
    # A ZERO spelled with a sign is still a zero, not a refusal -- the only
    # spelling where a wrong answer would be invisible.
    assert _count_field({"count": "+0"}, "count") == 0
    assert not _field_skewed("count")
    # Python's own digit-separator spelling reads too (base read it): the rule is
    # "the text parses as this integer", not "the text is the canonical digits".
    assert _count_field({"count": "1_0"}, "count") == 10
    # ...and the SIGN is read, not stripped: a negative count stays negative
    # (base read `"-02"` as -2, and a headline must not gain 4 out of nowhere).
    for stated in (-2, "-02", " -2 "):
        assert _count_field({"count": stated}, "count") == -2, repr(stated)
    out = _render_trace_text({
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": Decimal("2"), "arg_label": {"index": 0, "callee": "memcpy"},
        "trace": [],
    })
    assert "arg[2] of memcpy" in out and "malformed arg_index" not in out, out


def test_the_three_divergent_paging_footers_converge_770():
    """#770: three renderers stated their page position their own way.

    (a) `field xrefs` built a bespoke footer -- "showing 5 of 12 refs (offset 0);
    more available -- raise --limit or use --offset" -- with no `//` and no next
    offset, while every other paged list used `_paging_footer`. (b) `class list`
    asserted "classes: N shown of TOTAL" UNCONDITIONALLY, so a page that WAS the
    whole set claimed a paging comparison the --count line for the same op never
    makes, and a partial page still stated no resume instruction. (c) `evidence
    message` printed at most 3 code + 3 data refs per match under a header that
    stated the true counts, so an 8-ref match lost five rows with nothing said.
    All three now read the same way: the count line states the page, the shared
    footer states the total/remainder/resume, and a display cap says what it kept.
    """
    from bn.formatters import (_render_class_list_text, _render_field_xrefs_text,
                               _render_message_lens_text)

    field = {"type_name": "Hot", "field_name": "f", "offset": 8, "field_type": "int"}
    partial = {"field": field, "items": [{"kind": "code", "address": "0x1000"}],
               "total": 12, "returned": 5, "offset": 0, "limit": 5, "has_more": True}
    out = _render_field_xrefs_text(partial)
    assert "// showing 5 of 12 (7 more); rerun with --offset 5" in out, out
    assert "more available" not in out, out    # the bespoke second wording is gone

    whole = {"items": [{"name": "Widget", "method_count": 1, "has_vtable": True,
                        "size": None, "bases": [], "confidence": "rtti"}],
             "total": 1, "offset": 0, "limit": None, "returned": 1, "has_more": False}
    whole_out = _render_class_list_text(whole)
    assert "classes: 1" in whole_out and "shown of" not in whole_out, whole_out
    # A partial page states the same footer every other paged list does.
    paged = {**whole, "total": 30, "returned": 1, "limit": 1, "has_more": True}
    paged_out = _render_class_list_text(paged)
    assert "classes: 1" in paged_out and "shown of" not in paged_out, paged_out
    assert "// showing 1 of 30 (29 more); rerun with --offset 1" in paged_out, paged_out

    lens = {"query": "Codec", "count": 1, "total": 1, "items": [{
        "type_string": {"address": "0x5000", "value": "CodecInfo"},
        "xrefs": {"code_refs": [{"address": f"0x40{i:04x}", "function": f"f{i}"}
                                for i in range(8)],
                  "data_refs": [{"address": f"0x50{i:04x}"} for i in range(4)]},
    }]}
    lens_out = _render_message_lens_text(lens)
    assert "xrefs: 8 code, 4 data" in lens_out, lens_out
    assert lens_out.count("    code 0x") == 3 and lens_out.count("    data 0x") == 3, lens_out
    assert "code refs: 8 total, showing first 3" in lens_out, lens_out
    assert "data refs: 4 total, showing first 3" in lens_out, lens_out
    # A match whose refs fit is not made noisy by the disclosure.
    small = {"query": "Codec", "count": 1, "total": 1, "items": [{
        "type_string": {"address": "0x5000", "value": "CodecInfo"},
        "xrefs": {"code_refs": [{"address": "0x401000", "function": "parse"}],
                  "data_refs": []}}]}
    small_out = _render_message_lens_text(small)
    assert "total, showing first" not in small_out, small_out
