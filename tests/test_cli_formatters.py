from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


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
        ["disasm", "sub_401000", "--format", "json", "--lines", "1:5"],
    ],
)
def test_lines_flag_rejected_outside_text_mode(monkeypatch, capsys, argv):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(argv + ["--target", "active"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--lines only applies to --format text" in err
    assert "Traceback" not in err


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
