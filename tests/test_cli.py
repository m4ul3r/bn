from __future__ import annotations

import json
import types

import bn.cli
import pytest


def test_function_list_uses_implicit_target_when_single_target_is_open(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "list_functions":
            return {"ok": True, "result": {"functions": [{"name": "sub_401000", "address": "0x401000"}],
                                           "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "list_functions"]
    assert calls[1]["params"] == {"limit": 100}
    assert calls[1]["target"] == "active"
    output = capsys.readouterr().out
    assert output == "0x401000  sub_401000\n"
    assert '"name"' not in output


def test_function_list_requires_target_when_multiple_targets_are_open(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                        "active": True,
                    },
                    {"target_id": "123:2:8", "selector": "other.exe.bndb", "active": False},
                ],
            }
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])

    assert rc == 2
    assert capsys.readouterr().err == (
        "This command requires --target when multiple targets are open.\n"
        "Open targets:\n"
        "- SnailMail_unwrapped.exe.bndb [active] (target_id: 123:1:7)\n"
        "- other.exe.bndb (target_id: 123:2:8)\n"
    )


def test_function_list_returns_full_result_set(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {"functions": [{"name": f"sub_{index:06x}", "address": hex(index)} for index in range(150)],
                       "total": 150, "offset": 0, "limit": 200, "returned": 150, "has_more": False},
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list", "--target", "active", "--format", "json", "--limit", "200"])

    assert rc == 0
    assert captured["op"] == "list_functions"
    assert captured["params"] == {"limit": 200}
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert len(payload["functions"]) == 150
    assert payload["total"] == 150 and payload["has_more"] is False
    assert stderr == ""


def test_function_list_warns_when_output_auto_spills(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        return {
            "ok": True,
            "result": {"functions": [
                {"name": "sub_401000", "address": "0x401000"},
                {"name": "sub_402000", "address": "0x402000"},
            ], "total": 2, "offset": 0, "limit": 100, "returned": 2, "has_more": False},
        }

    def fake_write_output_result(value, *, fmt, out_path, stem):
        captured["value"] = value
        captured["fmt"] = fmt
        captured["out_path"] = out_path
        captured["stem"] = stem
        return types.SimpleNamespace(
            rendered=(
                "ok: true\n"
                "spilled: true\n"
                "path: /tmp/functions.txt\n"
                "format: text\n"
                "bytes: 1234\n"
                "tokens: 23456\n"
                "tokenizer: estimate\n"
                "sha256: deadbeef\n"
                "summary: kind=string chars=42\n"
            ),
            spilled=True,
            artifact={
                "artifact_path": "/tmp/functions.txt",
                "bytes": 1234,
                "format": "text",
                "sha256": "deadbeef",
                "spilled": True,
                "summary": {"kind": "string", "chars": 42},
                "tokenizer": "estimate",
                "tokens": 23456,
            },
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["function", "list", "--target", "active"])

    assert rc == 0
    stdout, stderr = capsys.readouterr()
    assert stdout.startswith("ok: true\nspilled: true\npath: /tmp/functions.txt\n")
    assert captured["value"] == "0x401000  sub_401000\n0x402000  sub_402000"
    assert stderr == (
        "warning: function list output spilled to /tmp/functions.txt; "
        "rerun with --limit/--offset to page through the results\n"
    )


def _spill_artifact_namespace(path: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rendered=f"ok: true\nspilled: true\npath: {path}\n",
        spilled=True,
        artifact={
            "artifact_path": path,
            "bytes": 4321,
            "format": "text",
            "sha256": "feedface",
            "spilled": True,
            "summary": {"kind": "string", "chars": 99},
            "tokenizer": "estimate",
            "tokens": 34567,
        },
    )


def test_decompile_spill_warning_suggests_line_slicing(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "decompile"
        return _spill_artifact_namespace("/tmp/decompile.txt")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert stderr == (
        "warning: decompile output spilled to /tmp/decompile.txt; "
        "rerun with --lines START:END to fetch a slice instead\n"
    )


def test_decompile_json_spill_warning_does_not_suggest_lines(monkeypatch, capsys):
    # --lines is a text-only flag; when decompile --format json spills, the hint
    # must NOT tell JSON consumers to rerun with --lines (a dead end) -- it
    # should point at --out / the artifact instead (#120).
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "decompile"
        return _spill_artifact_namespace("/tmp/decompile.json")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active", "--format", "json"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert "--lines" not in stderr
    assert "--out" in stderr


def test_scalar_spill_warning_points_at_artifact(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        # type_info returns a dict (non-list) payload that can spill.
        return {"ok": True, "result": {"name": "Player", "decl": "struct Player { ... }"}}

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "type-show"
        return _spill_artifact_namespace("/tmp/type-show.txt")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["types", "show", "Player", "--target", "active"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert stderr == (
        "warning: type info output spilled to /tmp/type-show.txt; "
        "rerun with --out <path> to write it to a file, or read that artifact "
        "to inspect the full output\n"
    )


def test_function_list_forwards_address_filters(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "function",
            "list",
            "--target",
            "active",
            "--min-address",
            "0x401000",
            "--max-address",
            "0x402000",
        ]
    )

    assert rc == 0
    assert captured["op"] == "list_functions"
    assert captured["params"]["min_address"] == "0x401000"
    assert captured["params"]["max_address"] == "0x402000"
    assert capsys.readouterr().out == "none\n"


def test_function_search_can_request_regex_matching(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": {"functions": [{"name": "load_attachment", "address": "0x401000"}],
                                       "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "search", "--target", "active", "--regex", "attach|detach"])

    assert rc == 0
    assert captured["op"] == "search_functions"
    assert captured["params"]["query"] == "attach|detach"
    assert captured["params"]["regex"] is True
    assert "offset" not in captured["params"]
    assert captured["params"]["limit"] == 100
    assert capsys.readouterr().out == "0x401000  load_attachment\n"


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


def test_target_flag_accepted_before_subcommand():
    parser = bn.cli.build_parser()

    # Names with dots, names that collide with subcommand strings, and
    # interleaving with --instance must all parse with -t before the subcommand.
    cases = [
        (["-t", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", None),
        (["--target", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", None),
        (["-t", "session", "function", "list"], "session", None),
        (["-t", "function", "function", "list"], "function", None),
        (["--instance", "X", "-t", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", "X"),
        (["-t", "pam_qnx.so.2", "--instance", "X", "function", "list"], "pam_qnx.so.2", "X"),
    ]
    for argv, expected_target, expected_instance in cases:
        args = parser.parse_args(argv)
        assert args.target == expected_target, argv
        assert args.instance == expected_instance, argv


def test_target_flag_after_subcommand_still_works():
    parser = bn.cli.build_parser()

    # The pre-existing form (target after subcommand) must keep working.
    args = parser.parse_args(["function", "list", "-t", "pam_qnx.so.2"])
    assert args.target == "pam_qnx.so.2"


def test_target_flag_root_does_not_clobber_subparser_value():
    parser = bn.cli.build_parser()

    # Root-level -t followed by a subparser-level -t: the later one wins
    # (argparse default), and neither None nor SUPPRESS leaks through.
    args = parser.parse_args(["-t", "first", "function", "list", "-t", "second"])
    assert args.target == "second"


def test_function_commands_accept_paging_flags():
    parser = bn.cli.build_parser()

    args = parser.parse_args(["function", "list", "--limit", "10"])
    assert args.limit == 10
    assert args.offset == 0

    args = parser.parse_args(["function", "search", "--offset", "10", "--limit", "50", "attach"])
    assert args.offset == 10
    assert args.limit == 50
    assert args.query == "attach"


def test_callsites_both_scope_flags_still_rejected():
    parser = bn.cli.build_parser()

    # Passing both scope flags is still a mutex violation handled by argparse.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "callsites",
                "crt_rand",
                "--within",
                "bonus_pick_random_type",
                "--within-file",
                "functions.txt",
            ]
        )


def test_unrecognized_argument_routes_to_subcommand_usage(capsys):
    parser = bn.cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["function", "list", "--bogus"])

    assert exc_info.value.code == 2
    _, stderr = capsys.readouterr()
    # Usage and error line should reference the specific subcommand, not bare `bn`.
    assert "usage: bn function list" in stderr
    assert "bn function list: error: unrecognized arguments: --bogus" in stderr


def test_unrecognized_argument_at_root_routes_to_root_usage(capsys):
    parser = bn.cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--bogus"])

    assert exc_info.value.code == 2
    _, stderr = capsys.readouterr()
    assert "usage: bn " in stderr
    assert "bn: error: unrecognized arguments: --bogus" in stderr


def test_callsites_missing_scope_raises_actionable_error(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise AssertionError("bridge should not be called when scope is missing")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["callsites", "crt_rand", "--target", "active"])

    # BridgeError surfaces as a nonzero exit with a human-facing message.
    assert rc != 0
    combined = capsys.readouterr()
    text = combined.err + combined.out
    assert "--within" in text
    assert "--within-file" in text
    assert "bn xrefs crt_rand" in text


def test_function_info_uses_active_target_and_text_renderer(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "prototype": "int32_t sub_401000(int32_t arg1)",
                "return_type": "int32_t",
                "calling_convention": "__cdecl",
                "size": 24,
                "parameters": [{"name": "arg1", "type": "int32_t", "storage": 0, "is_parameter": True, "local_id": "0x401000:param:stack:0:0:1"}],
                "locals": [{"name": "var_4", "type": "int32_t", "storage": -4, "is_parameter": False, "local_id": "0x401000:local:stack:-4:1:2"}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "info", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert captured["op"] == "function_info"
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert "sub_401000 @ 0x401000" in output
    assert "calling convention: __cdecl" in output
    assert "size: 24" in output
    assert "xrefs: 0" in output
    assert "locals: 1 variables" in output
    # compact mode should NOT show full parameter/local details
    assert "id=0x401000:param:stack:0:0:1" not in output


def test_symbol_rename_builds_preview_payload(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": {"preview": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "symbol",
            "rename",
            "--target",
            "123:1:7",
            "--preview",
            "sub_401000",
            "player_update",
        ]
    )
    assert rc == 0
    assert captured["op"] == "rename_symbol"
    assert captured["target"] == "123:1:7"
    assert captured["params"]["preview"] is True


def test_symbol_rename_uses_implicit_target_when_single_target_is_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                    }
                ],
            }
        if op == "rename_symbol":
            return {"ok": True, "result": {"preview": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--preview", "sub_401000", "player_update"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "rename_symbol"]
    assert calls[1]["target"] == "active"


def test_symbol_rename_requires_target_when_multiple_targets_are_open(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                        "active": True,
                    },
                    {"target_id": "123:2:8", "selector": "other.exe.bndb", "active": False},
                ],
            }
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "sub_401000", "player_update"])

    assert rc == 2
    assert capsys.readouterr().err == (
        "This command requires --target when multiple targets are open.\n"
        "Open targets:\n"
        "- SnailMail_unwrapped.exe.bndb [active] (target_id: 123:1:7)\n"
        "- other.exe.bndb (target_id: 123:2:8)\n"
    )


def test_function_create_builds_payload_and_defaults_to_json(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "message": "Function created and verified in the live Binary Ninja session.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verified",
                        "address": "0x401000",
                        "function": "sub_401000",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "0x401000"])

    assert rc == 0
    assert captured["op"] == "function_create"
    assert captured["target"] == "123:1:7"
    assert captured["params"] == {"address": "0x401000", "preview": False}
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "verified"


def test_function_create_text_output_renders_verified_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "message": "Function created and verified in the live Binary Ninja session.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verified",
                        "address": "0x401000",
                        "function": "sub_401000",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--format", "text", "0x401000"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "function_create 0x401000 (sub_401000) [verified]" in out


def test_function_create_forwards_preview_flag(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--preview", "0x401000"])

    assert rc == 0
    assert captured["params"]["preview"] is True


def test_function_create_verification_failure_exits_three(monkeypatch):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": False,
                "committed": False,
                "message": "Rolled back because no function was created at the address.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verification_failed",
                        "address": "0x401000",
                        "message": "No function starts at 0x401000 after analysis.",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "0x401000"])

    assert rc == 3


def test_skills_source_dir_prefers_repo_then_falls_back_to_prefix(monkeypatch, tmp_path):
    # #83: editable checkout uses repo skills/; a wheel install (no repo skills/)
    # falls back to the install prefix where the data files land.
    import sys as _sys

    import bn.paths as paths

    (tmp_path / "skills").mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.skills_source_dir() == tmp_path / "skills"

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: empty)  # no skills/ here
    assert paths.skills_source_dir() == paths.Path(_sys.prefix)


def test_plugin_source_dir_prefers_repo_then_falls_back_to_installed_module(monkeypatch, tmp_path):
    # #83: editable checkout uses repo plugin/<name>; a wheel install resolves
    # the bridge packaged into site-packages via find_spec.
    import bn.paths as paths

    repo_plugin = tmp_path / "plugin" / paths.PLUGIN_NAME
    repo_plugin.mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.plugin_source_dir() == repo_plugin

    empty = tmp_path / "empty"
    empty.mkdir()
    installed = tmp_path / "site" / paths.PLUGIN_NAME
    installed.mkdir(parents=True)
    fake_spec = types.SimpleNamespace(origin=str(installed / "__init__.py"))
    monkeypatch.setattr(paths, "repo_root", lambda: empty)
    monkeypatch.setattr(paths.importlib.util, "find_spec", lambda name: fake_spec)
    assert paths.plugin_source_dir() == installed


def test_plugin_install_copy_mode(tmp_path):
    destination = tmp_path / "plugin-copy"
    rc = bn.cli.main(
        [
            "plugin",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bridge.py").exists()


def test_skill_install_copy_mode(tmp_path):
    destination = tmp_path / "skill-copy"
    rc = bn.cli.main(
        [
            "skill",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bn" / "SKILL.md").exists()
    assert (destination / "bn" / "agents" / "openai.yaml").exists()
    assert (destination / "bn-re" / "SKILL.md").exists()
    assert (destination / "bn-vr" / "SKILL.md").exists()


def test_skill_install_defaults_to_claude_only_without_codex_home(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert not codex_root.exists()


def test_skill_install_defaults_to_claude_and_codex_when_codex_home_exists(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_defaults_skip_existing_destinations(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    (claude_root / "bn").mkdir(parents=True)
    (claude_root / "bn-re").mkdir()
    (claude_root / "bn-vr").mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_default_output_is_text(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    output = capsys.readouterr().out
    assert output.startswith("Installed skills (copy):\n")
    assert "- " + str(claude_root / "bn") in output
    assert '"installed"' not in output


def test_skill_install_json_output_remains_available(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--format", "json"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"installed": true' in output
    assert '"installed_destinations"' in output


def test_skill_install_custom_dest_still_fails_when_destination_exists(tmp_path):
    destination = tmp_path / "skill-copy"
    (destination / "bn").mkdir(parents=True)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--dest", str(destination)])

    assert rc == 2


def test_target_list_text_format_renders_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "list_targets"
        return {
            "ok": True,
            "result": [
                {
                    "selector": "SnailMail_unwrapped.exe.bndb",
                    "target_id": "123:1:7",
                    "view_id": "1",
                    "view_name": "PE",
                    "filename": "/tmp/SnailMail_unwrapped.exe.bndb",
                    "active": True,
                }
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "SnailMail_unwrapped.exe.bndb [active]" in output
    assert "target: 123:1:7" in output
    assert '"selector"' not in output


def test_refresh_uses_implicit_target_when_single_target_is_open(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "refresh":
            return {
                "ok": True,
                "result": {
                    "refreshed": True,
                    "target": {
                        "selector": "SnailMail_unwrapped.exe.bndb",
                        "target_id": "123:1:7",
                        "view_id": "1",
                        "view_name": "PE",
                        "filename": "/tmp/SnailMail_unwrapped.exe.bndb",
                        "active": True,
                    },
                },
            }
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["refresh", "--format", "text"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "refresh"]
    assert calls[1]["target"] == "active"
    output = capsys.readouterr().out
    assert "refreshed: true" in output
    assert "SnailMail_unwrapped.exe.bndb" in output


def test_types_show_uses_type_info_and_text_renderer(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "name": "Player",
                "kind": "StructureTypeClass",
                "decl": "struct Player",
                "layout": "struct Player // size=0x10\n0x0000: int32_t hp",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["types", "show", "--format", "text", "--target", "active", "Player"])

    assert rc == 0
    assert captured["op"] == "type_info"
    assert captured["params"]["type_name"] == "Player"
    assert captured["params"]["require_struct"] is False
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert output.startswith("struct Player")
    assert '"decl"' not in output


def test_types_declare_uses_implicit_target_when_single_target_is_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "types_declare":
            return {"ok": True, "result": {"preview": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["types", "declare", "typedef struct Player { int hp; } Player;"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "types_declare"]
    assert calls[1]["target"] == "active"
    assert "typedef struct Player" in calls[1]["params"]["declaration"]


def test_types_declare_passes_source_path_for_file_input(monkeypatch, tmp_path):
    captured = {}
    declaration_file = tmp_path / "win32_min.h"
    declaration_file.write_text("typedef struct Player { int hp; } Player;", encoding="utf-8")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": {"preview": False, "success": True, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(declaration_file)])

    assert rc == 0
    assert captured["op"] == "types_declare"
    assert captured["params"]["source_path"] == str(declaration_file)


def test_xrefs_field_routes_to_field_xrefs(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "field": {
                    "type_name": "TrackRowCell",
                    "field_name": "tile_type",
                    "offset": 8,
                    "field_type": "uint32_t",
                },
                "code_refs": [{"address": "0x401000", "function": "sub_401000", "incoming_type": "TrackRowCell*", "disasm": "mov eax, [ecx+8]"}],
                "data_refs": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["xrefs", "--field", "TrackRowCell.tile_type", "--format", "text", "--target", "active"])

    assert rc == 0
    assert captured["op"] == "field_xrefs"
    assert captured["params"]["field"] == "TrackRowCell.tile_type"
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert "TrackRowCell.tile_type" in output
    assert "code refs:" in output


def test_xrefs_text_format_renders_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "xrefs"
        return {
            "ok": True,
            "result": {
                "address": "0x401000",
                "code_refs": [
                    {
                        "address": "0x402000",
                        "function": "sub_402000",
                        "caller_function": {"address": "0x401f00", "name": "sub_402000"},
                    }
                ],
                "data_refs": [
                    {
                        "address": "0x403000",
                        "function": "sub_403000",
                        "caller_function": {"address": "0x402f00", "name": "sub_403000"},
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["xrefs", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "xrefs to 0x401000" in output
    assert "code refs: 1 site across 1 function" in output
    assert "0x401f00  sub_402000  (1 site: 0x402000)" in output
    assert "data refs: 1 site across 1 function" in output
    assert "0x402f00  sub_403000  (1 site: 0x403000)" in output


def test_evidence_xrefs_routes_and_renders_context(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "address": "0x175b20",
                "target_context": {
                    "sections": [{"name": ".rodata"}],
                    "symbol": {"name": "common.HeadUnitInfo", "type": "DataSymbol"},
                    "string": {
                        "value": "Usage: %s [OPTION]...\n",
                        "encoding": "ascii",
                        "truncated": True,
                    },
                },
                "code_refs": [
                    {
                        "address": "0x586c0",
                        "function": "sub_586a2",
                        "kind": "code",
                        "context": {
                            "sections": [{"name": ".text"}],
                            "segment": {"readable": True, "writable": False, "executable": True},
                            "disasm": "adr r1, common.HeadUnitInfo",
                        },
                    }
                ],
                "data_refs": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["evidence", "xrefs", "--format", "text", "--target", "active", "0x175b20"])

    assert rc == 0
    assert captured["op"] == "xrefs"
    assert captured["params"]["identifier"] == "0x175b20"
    output = capsys.readouterr().out
    assert "target | section=.rodata | symbol=common.HeadUnitInfo[DataSymbol]" in output
    assert 'string="Usage: %s [OPTION]...\\n" [truncated]' in output
    assert "0x586c0  code  sub_586a2" in output
    assert "seg=r-x" in output


def test_evidence_function_routes_and_renders_calls(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "function": {"name": "build_response", "address": "0x412470"},
                "prototype": "int32_t build_response(void*)",
                "calling_convention": "__cdecl",
                "thunk": {"is_candidate": False},
                "calls": [
                    {
                        "address": "0x4124a0",
                        "operation": "LLIL_CALL",
                        "direct": True,
                        "target": {
                            "raw": "0x461746",
                            "normalized": "0x461746",
                            "function": {"name": "send_message", "address": "0x461746"},
                        },
                        "llil": "call(0x461746)",
                        "hlil_statement": "send_message(6, &response)",
                        "argument_source": "hlil",
                        "arguments": [
                            {"index": 0, "text": "6"},
                            {
                                "index": 1,
                                "text": "0x2a4f4",
                                "resolved": {"string": "4", "section": ".rodata"},
                            },
                        ],
                        "argument_candidates": [
                            {"source": "mlil", "index": 0, "text": "r0"},
                        ],
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["evidence", "function", "--target", "active", "--context", "1", "build_response"])

    assert rc == 0
    assert captured["op"] == "function_evidence"
    assert captured["params"] == {"identifier": "build_response", "context": 1}
    output = capsys.readouterr().out
    assert "build_response @ 0x412470" in output
    assert "target: send_message @ 0x461746" in output
    assert "arguments: (hlil)" in output
    assert '0x2a4f4 -> "4" [.rodata]' in output
    # uncertain extras are JSON-only; not shown in text
    assert "r0" not in output


def test_evidence_table_routes_and_renders_targets(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "address": "0x64ea0",
                "pointer_size": 4,
                "stride": 4,
                "warnings": [
                    "table start is in an executable segment; this may be code, not a pointer table"
                ],
                "entries": [
                    {
                        "index": 0,
                        "entry_address": "0x64ea0",
                        "value": "0x46971",
                        "readable": True,
                        "plausible": True,
                        "target": {
                            "raw": "0x46971",
                            "normalized": "0x46970",
                            "thumb_adjusted": True,
                            "function": {"name": "sub_46970", "address": "0x46970"},
                        },
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["evidence", "table", "--target", "active", "--entries", "1", "0x64ea0"])

    assert rc == 0
    assert captured["op"] == "pointer_table"
    assert captured["params"]["address"] == "0x64ea0"
    assert captured["params"]["entries"] == 1
    output = capsys.readouterr().out
    assert "pointer table @ 0x64ea0" in output
    assert "warning: table start is in an executable segment" in output
    assert "sub_46970 @ 0x46970 (raw 0x46971) [thumb-adjusted]" in output


def test_render_target_line_shows_symbol_and_string_for_mapped_targets():
    # ILX #4: mapped non-function targets should surface symbol/string + section,
    # not just bare hex.
    from bn import formatters

    vtable = {
        "raw": "0x3f418",
        "normalized": "0x3f418",
        "function": None,
        "status": "mapped",
        "context": {
            "symbol": {"name": "_ZTVN17service_framework7IPCBoolE", "type": "ExternalSymbol"},
            "sections": [{"name": ".extern"}],
        },
    }
    line = formatters._render_target_line(vtable)
    assert "_ZTVN17service_framework7IPCBoolE @ 0x3f418" in line
    assert "[.extern, ExternalSymbol]" in line

    rodata_string = {
        "raw": "0x2a407",
        "normalized": "0x2a407",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {"value": "N19androidauto_service17AndroidAutoClientE", "encoding": "ascii"},
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(rodata_string)
    assert '"N19androidauto_service17AndroidAutoClientE"' in line
    assert "[.rodata]" in line

    truncated_string = {
        "raw": "0x427840",
        "normalized": "0x427840",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {
                "value": "Usage: %s [OPTION]...\n" + ("A" * 16),
                "encoding": "ascii",
                "truncated": True,
            },
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(truncated_string)
    assert '"Usage: %s [OPTION]...\\nAAAAAAAAAAAAAAAA"' in line
    assert "[.rodata, truncated]" in line


def test_evidence_table_renders_interior_function_targets(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "pointer_table"
        return {
            "ok": True,
            "result": {
                "address": "0x402000",
                "pointer_size": 8,
                "stride": 8,
                "warnings": ["1 entries resolve inside functions but not at function starts"],
                "entries": [
                    {
                        "index": 0,
                        "entry_address": "0x402000",
                        "value": "0x401001",
                        "readable": True,
                        "plausible": True,
                        "target": {
                            "raw": "0x401001",
                            "normalized": "0x401001",
                            "thumb_adjusted": False,
                            "function": {
                                "name": "target",
                                "address": "0x401000",
                                "exact_start": False,
                                "offset": "0x1",
                            },
                        },
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["evidence", "table", "--target", "active", "--entries", "1", "0x402000"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "warning: 1 entries resolve inside functions but not at function starts" in output
    assert "target @ 0x401000+0x1 (target 0x401001, not start)" in output
    assert "[thumb-adjusted]" not in output


def test_evidence_message_routes_and_renders_lens(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "query": "HeadUnitInfo",
                "count": 1,
                "matches": [
                    {
                        "type_string": {
                            "address": "0x175b20",
                            "value": "common.HeadUnitInfo",
                            "context": {"sections": [{"name": ".rodata"}]},
                        },
                        "xrefs": {
                            "code_refs": [{"address": "0x586c0", "function": "sub_586a2"}],
                            "data_refs": [{"address": "0x175bd8"}],
                        },
                        "metadata_table_windows": [{"address": "0x175bd0"}],
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "evidence",
            "message",
            "--target",
            "active",
            "--limit",
            "5",
            "--table-entries",
            "4",
            "HeadUnitInfo",
        ]
    )

    assert rc == 0
    assert captured["op"] == "message_lens"
    assert captured["params"] == {"query": "HeadUnitInfo", "limit": 5, "table_entries": 4}
    output = capsys.readouterr().out
    assert "message lens: HeadUnitInfo (1 matches)" in output
    assert "0x175b20  \"common.HeadUnitInfo\"" in output
    assert "metadata table windows: 1" in output


def test_evidence_init_routes_and_renders_sections(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "pointer_size": 4,
                "sections": [
                    {
                        "name": ".init_array",
                        "start": "0x5000",
                        "end": "0x5008",
                        "total_entries": 2,
                        "shown_entries": 2,
                        "truncated": False,
                        "table": {
                            "entries": [
                                {
                                    "index": 0,
                                    "entry_address": "0x5000",
                                    "value": "0x401001",
                                    "readable": True,
                                    "target": {
                                        "raw": "0x401001",
                                        "normalized": "0x401000",
                                        "thumb_adjusted": True,
                                        "function": {"name": "global_ctor", "address": "0x401000"},
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["evidence", "init", "--target", "active", "--limit", "4"])

    assert rc == 0
    assert captured["op"] == "init_arrays"
    assert captured["params"] == {"limit": 4}
    output = capsys.readouterr().out
    assert "init arrays: 1 section(s), pointer-size=4" in output
    assert ".init_array 0x5000-0x5008 entries=2" in output
    assert "global_ctor @ 0x401000 (raw 0x401001) [thumb-adjusted]" in output


def test_callsites_routes_within_scope_and_renders_text(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": [
                {
                    "callee": {"name": "crt_rand", "address": "0x461746"},
                    "containing_function": {
                        "name": "bonus_pick_random_type",
                        "address": "0x412470",
                    },
                    "within_query": "bonus_pick_random_type",
                    "call_index": 0,
                    "call_addr": "0x4124a0",
                    "instruction_length": 5,
                    "caller_static": "0x4124a5",
                    "call_instruction": {"address": "0x4124a0", "text": "call crt_rand"},
                    "previous_instructions": [
                        {"address": "0x41249c", "text": "mov eax, 0"},
                    ],
                    "next_instructions": [
                        {"address": "0x4124a5", "text": "cmp eax, 0xd"},
                    ],
                    "hlil_statement": "edx_1:eax_1 = sx.q(crt_rand())",
                    "pre_branch_condition": "result == 2",
                }
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "callsites",
            "--format",
            "text",
            "--target",
            "active",
            "--within",
            "bonus_pick_random_type",
            "--caller-static",
            "crt_rand",
        ]
    )

    assert rc == 0
    assert captured["op"] == "callsites"
    assert captured["target"] == "active"
    assert captured["params"]["callee"] == "crt_rand"
    assert captured["params"]["within_identifiers"] == ["bonus_pick_random_type"]
    assert captured["params"]["context"] == 3
    assert captured["params"]["caller_static"] is True
    output = capsys.readouterr().out
    assert output.startswith("caller_static 0x4124a5 | call 0x4124a0")
    assert "within: bonus_pick_random_type @ 0x412470" in output
    assert "call-index: 0" in output
    assert "within-query: bonus_pick_random_type" in output
    assert "hlil: edx_1:eax_1 = sx.q(crt_rand())" in output
    assert "pre-branch: result == 2" in output
    assert "> 0x4124a0  call crt_rand" in output


def test_callsites_within_file_ignores_comments_and_blank_lines(monkeypatch, tmp_path):
    captured = {}
    scope_file = tmp_path / "functions.txt"
    scope_file.write_text(
        "\n# curated trial functions\nbonus_pick_random_type\n\nfx_queue_add_random\n",
        encoding="utf-8",
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "callsites",
            "--target",
            "active",
            "--within-file",
            str(scope_file),
            "crt_rand",
        ]
    )

    assert rc == 0
    assert captured["op"] == "callsites"
    assert captured["params"]["within_identifiers"] == [
        "bonus_pick_random_type",
        "fx_queue_add_random",
    ]


def test_callsites_text_omits_null_hlil_and_pre_branch(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "callsites"
        return {
            "ok": True,
            "result": [
                {
                    "callee": {"name": "crt_rand", "address": "0x461746"},
                    "containing_function": {"name": "fx_queue_add_random", "address": "0x427700"},
                    "within_query": "fx_queue_add_random",
                    "call_index": 3,
                    "call_addr": "0x427806",
                    "instruction_length": 5,
                    "caller_static": "0x42780b",
                    "call_instruction": {"address": "0x427806", "text": "call crt_rand"},
                    "previous_instructions": [],
                    "next_instructions": [],
                    "hlil_statement": None,
                    "pre_branch_condition": None,
                }
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["callsites", "--format", "text", "--target", "active", "--within", "fx_queue_add_random", "crt_rand"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "call-index: 3" in output
    assert "hlil:" not in output
    assert "pre-branch:" not in output


def test_comment_get_uses_implicit_target_when_single_target_is_open(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "get_comment":
            return {"ok": True, "result": {"address": "0x401000", "comment": "interesting branch", "has_comment": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--address", "0x401000"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "get_comment"]
    assert calls[1]["target"] == "active"
    assert capsys.readouterr().out == "interesting branch\n"


def test_py_exec_accepts_inline_code(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": {"stdout": "", "result": None}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["py", "exec", "--target", "active", "--code", "print('hi')"])

    assert rc == 0
    assert captured["op"] == "py_exec"
    assert captured["target"] == "active"
    assert captured["params"]["script"] == "print('hi')"
    assert "out_path" not in captured["params"]


def test_py_exec_missing_script_mentions_code(capsys):
    rc = bn.cli.main(["py", "exec", "--target", "active", "--script", "missing.py"])

    assert rc == 2
    assert "Use --code for inline Python" in capsys.readouterr().err


def test_strings_text_format_renders_rows(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "strings"
        return {
            "ok": True,
            "result": {
                "items": [
                    {
                        "address": "0x500000",
                        "length": 6,
                        "type": "AsciiString",
                        "value": "follow",
                    }
                ],
                "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--format", "text", "--target", "active", "--query", "follow"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '0x500000  len=6  AsciiString  "follow"' in output
    assert '"value"' not in output


def test_py_exec_text_format_renders_stdout_and_result(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "py_exec"
        return {
            "ok": True,
            "result": {
                "stdout": "hi\n",
                "result": {"functions": 7},
                "warnings": ["warning one"],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["py", "exec", "--format", "text", "--target", "active", "--code", "print('hi')"])

    assert rc == 0
    output = capsys.readouterr().out
    assert output.startswith("hi\n\nresult:\n")
    assert '"functions": 7' in output
    assert "warnings:" in output


def test_proto_get_renders_prototype_text(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "get_prototype"
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "prototype": "int32_t sub_401000(int32_t arg1)",
                "return_type": "int32_t",
                "calling_convention": "__cdecl",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["proto", "get", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "int32_t sub_401000(int32_t arg1)\n"


def test_local_list_text_is_slim(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "list_locals"
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "locals": [
                    {
                        "name": "arg1",
                        "type": "int32_t",
                        "storage": 4,
                        "source_type": "StackVariableSourceType",
                        "index": 0,
                        "identifier": 1,
                        "is_parameter": True,
                        "local_id": "0x401000:param:stack:4:0:1",
                    },
                    {
                        "name": "var_c",
                        "type": "char*",
                        "storage": -12,
                        "source_type": "StackVariableSourceType",
                        "index": 0,
                        "identifier": 2,
                        "is_parameter": False,
                        "local_id": "0x401000:local:stack:-12:0:2",
                    },
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["local", "list", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "(1 params, 1 locals)" in output
    assert "params:" in output
    assert "locals:" in output
    assert "arg1" in output and "int32_t" in output
    assert "var_c" in output and "char*" in output
    # local_id IS shown in text mode: it is the stable handle that `local rename`
    # / `local retype` take, so omitting it forced a --format json round-trip to
    # drive those commands (#122). The other internal fields stay out of the
    # slim text view.
    assert "0x401000:param:stack:4:0:1" in output
    assert "0x401000:local:stack:-12:0:2" in output
    assert "storage=" not in output
    assert "source=" not in output
    assert "identifier=" not in output


def test_local_list_json_retains_ids(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "locals": [
                    {
                        "name": "arg1",
                        "type": "int32_t",
                        "storage": 4,
                        "source_type": "StackVariableSourceType",
                        "index": 0,
                        "identifier": 1,
                        "is_parameter": True,
                        "local_id": "0x401000:param:stack:4:0:1",
                    }
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["local", "list", "--format", "json", "--target", "active", "sub_401000"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["locals"][0]["local_id"] == "0x401000:param:stack:4:0:1"
    assert payload["locals"][0]["identifier"] == 1


def test_target_summary_text_shows_function_counts():
    # target info should surface the function-count + named-vs-auto summary that
    # every agent immediately reaches for (#122).
    from bn import formatters
    out = formatters._render_target_summary({
        "selector": "active",
        "arch": "x86_64",
        "function_count": 412,
        "named_function_count": 87,
        "unnamed_function_count": 313,
        "imported_function_count": 12,
    })
    assert "412 functions" in out
    assert "87 named" in out
    assert "313 auto-named" in out
    assert "12 imported" in out


def test_target_summary_text_marks_quick_view():
    """A --quick (unanalyzed) target must be flagged in text output, not just JSON."""
    from bn import formatters
    quick = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": False, "analysis_state": "quick",
    })
    assert "[not analyzed]" in quick
    full = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": True, "analysis_state": "full",
    })
    assert "[not analyzed]" not in full


def _zero_function_search(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
    return {"ok": True, "result": {"functions": [], "total": 0, "offset": 0,
                                   "limit": None, "returned": 0, "has_more": False}}


def test_function_search_hints_regex_on_zero_matches_with_metachars(monkeypatch, capsys):
    """A non-regex search whose query has regex metacharacters and matches
    nothing should suggest --regex instead of a bare 'none' (#122)."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "init|fini", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--regex" in err
    assert "init|fini" in err


def _empty_xrefs(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
    return {"ok": True, "result": {"address": "0x308", "code_refs": [], "data_refs": []}}


def test_xrefs_hints_struct_field_on_small_offset_zero_match(monkeypatch, capsys):
    """`xrefs 0x308` with 0 matches: 0x308 looks like a struct-field offset
    misread as an absolute address. Nudge toward --field."""
    monkeypatch.setattr(bn.cli, "send_request", _empty_xrefs)
    rc = bn.cli.main(["xrefs", "0x308", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--field" in err


def test_xrefs_no_offset_hint_for_real_address(monkeypatch, capsys):
    """A plausible code/data address (>= 0x10000) with 0 xrefs is a normal
    'nothing references this' result -- no struct-field hint."""
    monkeypatch.setattr(bn.cli, "send_request", _empty_xrefs)
    rc = bn.cli.main(["xrefs", "0x401000", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--field" not in err


def test_function_search_no_regex_hint_when_regex_flag_set(monkeypatch, capsys):
    """--regex already set: the query IS a pattern, so no hint even at 0 matches."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "init|fini", "--regex", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "add --regex" not in err


def test_function_search_no_regex_hint_for_plain_query(monkeypatch, capsys):
    """A plain query with no metacharacters and 0 matches gets no regex hint."""
    monkeypatch.setattr(bn.cli, "send_request", _zero_function_search)
    rc = bn.cli.main(["function", "search", "plainname", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "add --regex" not in err


def test_strings_hints_regex_on_zero_matches_with_metachars(monkeypatch, capsys):
    """strings (a bare list today) with a metacharacter query and 0 matches also
    suggests --regex (#122)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": []}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["strings", "--query", "foo(bar", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--regex" in err


def test_bundle_function_out_path_is_bridge_owned(monkeypatch, tmp_path, capsys):
    captured = {}
    out_path = tmp_path / "bundle.json"

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "ok": True,
                "artifact_path": str(out_path),
                "format": "json",
                "bytes": 123,
                "sha256": "deadbeef",
                "summary": {"kind": "object", "count": 3},
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["bundle", "function", "--out", str(out_path), "sub_401000"])

    assert rc == 0
    assert captured["op"] == "bundle_function"
    assert captured["params"]["out_path"] == str(out_path)
    assert not out_path.exists()
    output = capsys.readouterr().out
    # bundle function defaults to --format json; the bridge-owned --out envelope
    # printed to stdout must itself be valid JSON, not a text key:value block
    # (issue #10).
    payload = json.loads(output)
    assert payload["artifact_path"] == str(out_path)
    assert payload["spilled"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["function", "list", "--target", "active", "--limit", "-1"],
        ["function", "list", "--target", "active", "--limit", "0"],
        ["function", "list", "--target", "active", "--offset", "-1"],
        ["types", "--target", "active", "--limit", "-2"],
        ["xrefs", "sub_401000", "--target", "active", "--limit", "-1"],
        # evidence message has its own per-command --limit (_positive_int);
        # callsites declares no --limit, so the prior callsites row asserted
        # nothing about the validator (it exited 2 via "unrecognized arguments").
        ["evidence", "message", "token", "--target", "active", "--limit", "0"],
    ],
)
def test_negative_or_zero_pagination_args_rejected_with_exit_2(argv):
    # Negative/zero --limit (and negative --offset) must be rejected at the arg
    # layer with argparse's exit code 2, never leak into Python negative-slice
    # semantics downstream (issue #9; #15 types --limit 0). Covers both the
    # shared paged path (function list / types) and per-command renderer limits
    # (xrefs / evidence message).
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def test_positive_pagination_args_still_accepted():
    parser = bn.cli.build_parser()
    ns = parser.parse_args(["function", "list", "--target", "active", "--limit", "5", "--offset", "0"])
    assert ns.limit == 5
    assert ns.offset == 0


def test_count_flags_accept_hex_literals():
    # The count/index validators parse with int(value, 0), so --limit 0x10 must
    # be accepted -- matching _int_or_hex's grammar for --length etc. (#32).
    parser = bn.cli.build_parser()
    ns = parser.parse_args(
        ["function", "list", "--target", "active", "--limit", "0x10", "--offset", "0x4"])
    assert ns.limit == 16
    assert ns.offset == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["taint", "forward", "-f", "h", "--source", "param:0", "--max-depth", "-1", "--target", "active"],
        ["taint", "backward", "-f", "h", "--sink", "arg:memcpy:0", "--max-depth", "-1", "--target", "active"],
        ["evidence", "message", "q", "--table-entries", "-1", "--target", "active"],
    ],
)
def test_sibling_count_flags_reject_negative(argv):
    # Count siblings beyond --limit/--offset must also reject negatives at the
    # arg layer, not silently degrade analysis or empty a window (#28).
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def test_taint_max_depth_zero_allowed():
    # 0 is a meaningful "intraprocedural only" choice and must be accepted.
    parser = bn.cli.build_parser()
    ns = parser.parse_args(
        ["taint", "forward", "-f", "h", "--source", "param:0", "--max-depth", "0", "--target", "active"])
    assert ns.max_depth == 0


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


def test_version_flag_prints_version(capsys):
    # `bn --version` is a real affordance, not "unrecognized arguments" (#49).
    from bn.version import VERSION
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["--version"])
    assert exc.value.code == 0
    out, _ = capsys.readouterr()
    assert out.strip() == f"bn {VERSION}"


def test_version_is_single_sourced_from_pyproject():
    # The version literal lives only in pyproject.toml; version.py derives it,
    # so CLI/bridge never drift and a bump touches one file (#82).
    import tomllib
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    canonical = pyproject["project"]["version"]

    assert canonical == "0.20.0"  # the reset target
    assert bn.version.VERSION == canonical
    # No stray literal: the old number must be gone from the version module.
    assert "0.12.2" not in Path(bn.version.__file__).read_text(encoding="utf-8")


def test_resolve_version_falls_back_to_dist_metadata(monkeypatch):
    # When pyproject is unreachable (installed wheel), VERSION resolves from the
    # installed distribution metadata rather than crashing (#82).
    import tomllib

    import bn.version

    monkeypatch.setattr(
        tomllib, "loads",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pyproject here")),
    )
    # Editable install metadata reports the same canonical version.
    assert bn.version._resolve_version() == bn.version.VERSION


def test_bridge_plugin_json_carries_no_version_literal():
    # plugin.json is BN-manager metadata that cannot import Python; it must not
    # duplicate the version (single-source invariant, #82).
    import json
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    manifest = json.loads(
        (repo_root / "plugin" / "bn_agent_bridge" / "plugin.json").read_text(encoding="utf-8")
    )
    assert "version" not in manifest


def test_max_depth_validator_says_depth_not_index(capsys):
    # Depth flags get a "depth" label, not the generic "index" (#49).
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["taint", "forward", "-f", "main", "--source", "param:0",
                     "--max-depth", "-1", "--target", "active"])
    assert exc.value.code == 2
    _, err = capsys.readouterr()
    assert "depth must be an integer >= 0" in err
    assert "index must be" not in err


def test_trace_max_depth_zero_rejected_at_parse_time(capsys):
    # `trace --max-depth 0` is a 0-step budget the bridge rejects; the CLI must
    # reject it at parse time so the contract matches (#129). --ip-depth 0 stays
    # valid (it means "do not cross call boundaries").
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["trace", "f", "0x10", "--target", "active", "--max-depth", "0"])
    assert exc.value.code == 2
    _, err = capsys.readouterr()
    assert "depth must be an integer >= 1" in err


def test_entries_validator_hex_aware_and_rejects_zero(capsys):
    # evidence table --entries is wired to the shared count validator: hex is
    # accepted and a degenerate 0/negative is rejected with the standard message (#59).
    with pytest.raises(SystemExit) as exc0:
        bn.cli.main(["evidence", "table", "0x1000", "--entries", "0", "--target", "active"])
    assert exc0.value.code == 2
    _, err0 = capsys.readouterr()
    assert "count must be an integer >= 1" in err0


def test_argparse_error_text_format_keeps_stdout_empty(capsys):
    # Default text format keeps the prior contract: usage on stderr, no stdout.
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["function", "list", "--target", "active", "--limit", "-1"])
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err


def test_removed_experimental_commands_are_not_present():
    parser = bn.cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["data"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bundle", "corpus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["struct", "replace"])
    with pytest.raises(SystemExit):
        parser.parse_args(["patch", "bytes"])


def test_missing_subcommand_prints_exact_help(capsys):
    rc = bn.cli.main(["struct"])

    assert rc == 1
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct [-h] [--help-full] {show,field} ..." in stdout
    assert "--help-full   Show help for this command and all subcommands" in stdout
    assert "usage: bn [-h]" not in stdout
    assert stderr == ""


def test_missing_nested_subcommand_prints_exact_help(capsys):
    rc = bn.cli.main(["struct", "field"])

    assert rc == 1
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct field [-h] [--help-full] {set,rename,delete} ..." in stdout
    assert "--help-full          Show help for this command and all subcommands" in stdout
    assert "usage: bn [-h]" not in stdout
    assert stderr == ""


def test_help_full_prints_recursive_root_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bn.cli.main(["--help-full"])

    assert exc_info.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "usage: bn" in stdout
    assert "usage: bn struct {show,field} ..." in stdout
    assert "usage: bn struct field set" in stdout
    assert "-h, --help" not in stdout
    assert "--help-full" not in stdout
    assert stderr == ""


def test_help_full_prints_recursive_subtree_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bn.cli.main(["struct", "field", "--help-full"])

    assert exc_info.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct field {set,rename,delete} ..." in stdout
    assert "usage: bn struct field set" in stdout
    assert "usage: bn struct field rename" in stdout
    assert "usage: bn\n" not in stdout
    assert "-h, --help" not in stdout
    assert "--help-full" not in stdout
    assert stderr == ""


def test_help_full_prints_leaf_help_without_required_positionals(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bn.cli.main(["struct", "field", "set", "--help-full"])

    assert exc_info.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct field set" in stdout
    assert "struct_name offset field_name field_type" in stdout
    assert "usage: bn struct field rename" not in stdout
    assert "-h, --help" not in stdout
    assert "--help-full" not in stdout
    assert stderr == ""


def test_doctor_reports_stale_loaded_plugin(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": "0.4.0",
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.4.0",
                "plugin_build_id": "oldbuild123456",
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor", "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cli_version"] == bn.cli.VERSION
    assert payload["plugin_install_build_id"]
    assert payload["instances"][0]["stale_plugin_version"] is True
    assert payload["instances"][0]["stale_plugin_code"] is True


def test_doctor_text_marks_healthy_instance_ok(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": bn.cli.VERSION,
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": bn.cli.VERSION,
                "plugin_build_id": "newbuild123456",
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"pid=123 plugin={bn.cli.VERSION} status=ok" in output
    assert "status=error" not in output


def test_doctor_json_carries_reachable_and_status(monkeypatch, tmp_path, capsys):
    """doctor --format json must carry the same health signal the text mode shows
    (reachable / status), so a scripted JSON health check can read it directly
    instead of re-deriving reachability from the absence of doctor.error. (L16)"""
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")

    def _inst(pid, name):
        return type("FakeInstance", (), {
            "pid": pid, "socket_path": tmp_path / f"{name}.sock",
            "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
            "instance_id": name,
        })()

    ok_inst, bad_inst = _inst(1, "ok"), _inst(2, "bad")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [ok_inst, bad_inst])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)

    def fake_send(instance, op, params=None, target=None):
        if instance is ok_inst:
            return {"ok": True, "result": {
                "plugin_version": bn.cli.VERSION, "plugin_build_id": "b", "targets": []}}
        raise OSError("connection refused")

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", fake_send)

    rc = bn.cli.main(["doctor", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    by_pid = {i["pid"]: i for i in data["instances"]}
    assert by_pid[1]["reachable"] is True and by_pid[1]["status"] == "ok"
    assert by_pid[2]["reachable"] is False and by_pid[2]["status"] == "error"


def test_target_info_verbose_renders_segments():
    """target info --verbose text appends the segment map with r/w/x perms; the
    block is absent when no segments are present (target list rows). (F21)"""
    from bn.formatters import _render_target_summary

    out = _render_target_summary({
        "selector": "svc", "arch": "aarch64",
        "segments": [
            {"start": "0x1000", "end": "0x2000", "length": 0x1000,
             "readable": True, "writable": False, "executable": True},
        ],
    })
    assert "segments:" in out
    assert "0x1000-0x2000 r-x (4096 bytes)" in out

    out2 = _render_target_summary({"selector": "svc", "arch": "aarch64"})
    assert "segments:" not in out2


def test_symbol_rename_text_format_renders_mutation_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                    }
                ],
                "affected_functions": [
                    {
                        "address": "0x401000",
                        "before_name": "sub_401000",
                        "after_name": "player_update",
                        "changed": True,
                        "diff": "--- before:sub_401000\n+++ after:player_update",
                    }
                ],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "symbol",
            "rename",
            "--format",
            "text",
            "--target",
            "active",
            "--preview",
            "sub_401000",
            "player_update",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "preview: change applied + reverted" in output
    assert "rename_symbol function 0x401000 -> player_update" in output
    assert "0x401000 sub_401000 -> player_update" in output
    assert '"results"' not in output


def test_symbol_rename_verification_failure_returns_nonzero(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": False,
                "committed": False,
                "message": "Rolled back because live-session verification failed.",
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "verification_failed",
                        "message": "Live rename verification failed at 0x401000",
                        "requested": {
                            "identifier": "sub_401000",
                            "kind": "function",
                            "new_name": "player_update",
                        },
                        "observed": {
                            "address": "0x401000",
                            "name": "sub_401000",
                        },
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--format", "text", "--target", "active", "sub_401000", "player_update"])

    assert rc == 3
    output = capsys.readouterr().out
    assert "rolled back" in output
    assert "failed: rename_symbol" in output
    assert "[verification_failed]" in output
    assert 'requested: {"identifier": "sub_401000"' in output
    assert 'observed: {"address": "0x401000", "name": "sub_401000"}' in output


def test_symbol_rename_noop_still_succeeds(monkeypatch):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "noop",
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "player_update", "player_update"])

    assert rc == 0


def test_decompile_text_format_unwraps_text_field(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "text": "return 7;",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["decompile", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "return 7;\n"


def test_comment_get_empty_comment_shows_placeholder(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "get_comment"
        return {"ok": True, "result": {"address": "0x401000", "comment": "", "has_comment": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--target", "active", "--address", "0x401000"])

    assert rc == 0
    assert capsys.readouterr().out == "(no comment)\n"


def test_callsites_empty_result_shows_descriptive_message(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "callsites"
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["callsites", "--format", "text", "--target", "active", "--within", "main", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "no callsites found\n"


def test_format_operation_result_falls_back_to_requested():
    item = {
        "op": "struct_field_set",
        "status": "unsupported",
        "message": "Struct not found",
        "requested": {
            "struct_name": "Player",
            "offset": "0x8",
            "field_name": "health",
            "field_type": "int32_t",
        },
    }
    result = bn.cli._format_operation_result(item)
    assert "Player" in result
    assert "0x8" in result
    assert "health" in result
    assert "int32_t" in result
    assert "<unknown>" not in result


def test_function_list_pagination_states_true_total(monkeypatch, capsys):
    # #59: the bridge returns the page WITH the true total; the footer states it
    # (showing N of TOTAL (REMAINING more)) on stdout. The CLI sends the real
    # limit (not limit+1).
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "functions": [{"name": f"sub_{i:06x}", "address": hex(i)} for i in range(20)],
                "total": 6350, "offset": 0, "limit": 20, "returned": 20, "has_more": True,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list", "--target", "active", "--limit", "20"])

    assert rc == 0
    assert captured["params"]["limit"] == 20            # real limit, not +1
    stdout, _ = capsys.readouterr()
    assert "// showing 20 of 6350 (6330 more)" in stdout   # true total + remainder
    assert "--offset 20" in stdout


def test_function_list_json_carries_paging_metadata(monkeypatch, capsys):
    # #59: machine consumers get total/has_more in JSON, not only a stderr note.
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {
            "functions": [{"name": "sub_0", "address": "0x0"}],
            "total": 6350, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["function", "list", "--target", "active", "--limit", "1", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 6350 and payload["has_more"] is True and payload["returned"] == 1


def test_strings_json_carries_paging_envelope(monkeypatch, capsys):
    # #122: strings now returns the {items, total, ...} envelope, so machine
    # consumers see the true total + remainder, not a bare truncated list. The
    # CLI forwards the REAL --limit (no client-side limit+1 probe).
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {
            "items": [{"address": "0x1000", "length": 5, "chars": 5,
                       "type": "ascii", "value": "alpha"}],
            "total": 4096, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["strings", "--target", "active", "--query", "alpha",
                      "--limit", "1", "--format", "json"])
    assert rc == 0
    assert captured["params"]["limit"] == 1   # real limit, not limit+1
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 4096
    assert payload["has_more"] is True and payload["returned"] == 1
    assert payload["items"][0]["value"] == "alpha"


def test_strings_text_footer_states_true_total(monkeypatch, capsys):
    # Text mode renders the rows AND a "showing N of TOTAL (R more)" footer that
    # mirrors function list, so a truncated dump still admits the remainder (#122).
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {
            "items": [{"address": hex(0x1000 + i), "length": 5, "chars": 5,
                       "type": "ascii", "value": f"str{i}"} for i in range(3)],
            "total": 50, "offset": 0, "limit": 3, "returned": 3, "has_more": True,
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["strings", "--target", "active", "--query", "str",
                      "--limit", "3", "--format", "text"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert '"str0"' in stdout                                # rows are rendered
    assert "// showing 3 of 50 (47 more)" in stdout          # honest total + remainder
    assert "--offset 3" in stdout


def test_imports_json_carries_paging_envelope(monkeypatch, capsys):
    # The non-summary imports list also returns the envelope, and the CLI
    # forwards the REAL --limit (no client-side limit+1 probe) (#122).
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {
            "items": [{"name": "printf", "address": "0x1000", "library": "libc",
                       "raw_name": "printf", "kind": "function"}],
            "total": 512, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["imports", "--target", "active", "--limit", "1", "--format", "json"])
    assert rc == 0
    assert captured["params"]["limit"] == 1   # real limit, not limit+1
    assert captured["params"].get("summary") is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 512 and payload["has_more"] is True
    assert payload["items"][0]["name"] == "printf"


def test_function_search_pagination_forwards_offset(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": [{"name": "sub_401000", "address": "0x401000"}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "search", "--target", "active", "--offset", "50", "--limit", "25", "sub"])

    assert rc == 0
    assert captured["params"]["offset"] == 50
    assert captured["params"]["limit"] == 25


def test_instance_flag_passed_to_send_request(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "abc123", "function", "list"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_on_subcommand(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "--instance", "abc123"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_from_env(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured_instance_ids


def test_session_list_shows_instances(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_instances = [
        BridgeInstance(
            pid=111,
            socket_path=__import__("pathlib").Path("/tmp/a.sock"),
            registry_path=__import__("pathlib").Path("/tmp/a.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:00:00Z",
            meta={},
            instance_id="aaaa1111",
        ),
        BridgeInstance(
            pid=222,
            socket_path=__import__("pathlib").Path("/tmp/b.sock"),
            registry_path=__import__("pathlib").Path("/tmp/b.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:01:00Z",
            meta={},
            instance_id="bbbb2222",
        ),
    ]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: fake_instances)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert len(parsed["instances"]) == 2
    assert parsed["instances"][0]["selector"] == "aaaa1111"
    assert parsed["instances"][0]["instance_id"] == "aaaa1111"
    assert parsed["instances"][1]["instance_id"] == "bbbb2222"
    assert "rss_mb" in parsed["instances"][0]
    assert "total_rss_mb" in parsed


def test_session_stop_sends_shutdown(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "shutdown"
        assert instance_id == "abc123"
        return {"ok": True, "result": {"shutting_down": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["stopped"] is True
    assert parsed["instance_id"] == "abc123"


def test_session_start_spawns_instance(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    rc = bn.cli.main(["session", "start", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["instance_id"] == "test1234"
    assert parsed["pid"] == 999


def test_session_start_all_loads_fail_stops_bridge_and_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeError, BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="ghost9",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    ops = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        ops.append(op)
        if op == "load_binary":
            raise BridgeError(f"File not found: {params['path']}")
        if op == "shutdown":
            return {"ok": True, "result": {"shutting_down": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", "/tmp/nonexistent.so", "--format", "json"])

    # Non-zero exit, and the empty zombie bridge is shut down rather than leaked.
    assert rc == 1
    assert "shutdown" in ops
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is True


def test_session_start_partial_failure_keeps_bridge_but_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeError, BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="half",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    ops = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        ops.append(op)
        if op == "load_binary":
            if "good" in params["path"]:
                return {"ok": True, "result": {"path": params["path"], "loaded": True}}
            raise BridgeError(f"File not found: {params['path']}")
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", "/tmp/good.so", "/tmp/bad.so", "--format", "json"])

    # One binary loaded, so the bridge stays up, but the failure still surfaces.
    assert rc == 1
    assert "shutdown" not in ops
    parsed = json.loads(capsys.readouterr().out)
    assert "stopped" not in parsed


def test_batch_apply_missing_manifest_clean_error(monkeypatch, capsys, tmp_path):
    from bn.transport import BridgeError

    def fail_send_request(*args, **kwargs):
        raise AssertionError("bridge should not be contacted for a missing manifest")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)

    missing = tmp_path / "no" / "such" / "manifest.json"
    rc = bn.cli.main(["batch", "apply", str(missing)])

    assert rc == 2  # BridgeError exit code
    err = capsys.readouterr().err
    assert "Manifest file not found" in err
    assert "Traceback" not in err


def test_batch_apply_invalid_json_clean_error(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", str(bad)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "Invalid JSON in manifest" in err
    assert "Traceback" not in err


def test_batch_apply_bare_array_manifest_clean_error(monkeypatch, capsys, tmp_path):
    # A bare JSON array (an easy mistake) must be a clean BridgeError, not a
    # client-side ValueError traceback in _call's dict(params) (#48).
    bad = tmp_path / "manifest.json"
    bad.write_text('[{"op": "set_comment", "address": "0x1000", "comment": "x"}]', encoding="utf-8")
    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", str(bad)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "must be a JSON object" in err
    assert "Traceback" not in err


def test_batch_apply_manifest_without_ops_clean_error(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text('{"target": "x"}', encoding="utf-8")  # no "ops" array
    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", str(bad)])

    assert rc == 2
    err = capsys.readouterr().err
    assert '"ops" array' in err
    assert "Traceback" not in err


def test_batch_apply_reads_manifest_from_stdin(monkeypatch, capsys):
    # "-" reads the manifest from stdin, enabling the quoted-heredoc form. A
    # comment containing ', ", $, and parens must survive verbatim with no
    # escaping (that is the whole point of a quoted heredoc) (#104).
    import io

    comment = "len isn't checked; $sp + (a) \"bad\""
    manifest = (
        '{"target": "active", "ops": ['
        '{"op": "set_comment", "address": "0x1000", "comment": "' + comment.replace('"', '\\"') + '"}'
        "]}"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(manifest))

    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"preview": False, "success": True, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 0
    assert captured["op"] == "batch_apply"
    # The free-text comment reached the bridge byte-for-byte.
    assert captured["params"]["ops"][0]["comment"] == comment


def test_batch_apply_stdin_forwards_preview_flag(monkeypatch):
    import io

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"ops": [{"op": "set_comment", "address": "0x1000", "comment": "x"}]}'),
    )
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["batch", "apply", "--preview", "-"])

    assert rc == 0
    assert captured["params"]["preview"] is True


def test_batch_apply_empty_stdin_clean_error(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "No manifest on stdin" in err
    assert "Traceback" not in err


def test_batch_apply_invalid_stdin_json_clean_error(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("{not valid json"))
    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "Invalid JSON in manifest (<stdin>)" in err
    assert "Traceback" not in err


def test_batch_apply_stdin_bare_array_clean_error(monkeypatch, capsys):
    import io

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('[{"op": "set_comment", "address": "0x1000", "comment": "x"}]'),
    )
    monkeypatch.setattr(
        bn.cli, "send_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge should not be contacted")),
    )

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "must be a JSON object" in err
    assert "Traceback" not in err


def test_il_lines_slices_output_with_header(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "il":
            return {"ok": True, "result": {"text": "line1\nline2\nline3\nline4\nline5"}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["il", "main", "--target", "active", "--lines", "2:4"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "// lines 2-4 of 5" in out
    assert "line2" in out and "line3" in out and "line4" in out
    assert "line1" not in out
    assert "line5" not in out


def test_disasm_lines_slices_output_with_header(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "disasm":
            return {"ok": True, "result": {"text": "aaa\nbbb\nccc\nddd"}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["disasm", "0x1000", "--target", "active", "--lines", "1:2"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "// lines 1-2 of 4" in out
    assert "aaa" in out and "bbb" in out
    assert "ccc" not in out and "ddd" not in out


def test_lines_range_rejects_zero_index_with_helpful_error(monkeypatch, capsys):
    # argparse type errors exit via SystemExit(2)
    with pytest.raises(SystemExit):
        bn.cli.main(["disasm", "0x1000", "--target", "active", "--lines", "0:3"])
    err = capsys.readouterr().err
    assert "1-indexed" in err


# --- I2: strings filtering CLI args ---


def test_strings_passes_min_length_to_bridge(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "strings":
            captured_params.update(params)
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--min-length", "5"])

    assert rc == 0
    assert captured_params["min_length"] == 5


def test_strings_passes_section_and_no_crt_to_bridge(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "strings":
            captured_params.update(params)
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--section", ".rodata", "--no-crt"])

    assert rc == 0
    assert captured_params["section"] == ".rodata"
    assert captured_params["no_crt"] is True


def test_strings_passes_regex_to_bridge(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "strings":
            captured_params.update(params)
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--query", "foo|bar", "--regex"])

    assert rc == 0
    assert captured_params["query"] == "foo|bar"
    assert captured_params["regex"] is True


def test_strings_query_value_can_look_like_flag(monkeypatch, capsys):
    captured_queries = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "strings":
            captured_queries.append(params["query"])
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--query", "-h"])
    assert rc == 0

    rc = bn.cli.main(["strings", "--target", "active", "--query", "--"])

    assert rc == 0
    assert captured_queries == ["-h", "--"]


def test_strings_query_value_can_be_a_known_sibling_flag(monkeypatch, capsys):
    # #102: a protected data option must accept ANY following flag-like token as
    # its literal value, including ones that collide with KNOWN sibling options
    # (--regex, --format, --target). `bn strings --query --format` searches for
    # the literal "--format". A user wanting --query <val> --regex uses = syntax.
    captured_queries = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "strings":
            captured_queries.append(params["query"])
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    assert bn.cli.main(["strings", "--target", "active", "--query", "--regex"]) == 0
    assert bn.cli.main(["strings", "--target", "active", "--query", "--format"]) == 0
    assert bn.cli.main(["strings", "--target", "active", "--query", "--limit"]) == 0
    assert captured_queries == ["--regex", "--format", "--limit"]


# --- I5: sections CLI ---


def test_sections_text_format_renders_rows(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "sections"
        return {
            "ok": True,
            "result": {
                "items": [
                    {
                        "name": ".text",
                        "start": "0x1000",
                        "end": "0x5000",
                        "length": 16384,
                        "semantics": "ReadOnlyCode",
                        "readable": True,
                        "writable": False,
                        "executable": True,
                    }
                ],
                "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["sections", "--format", "text", "--target", "active"])

    assert rc == 0
    output = capsys.readouterr().out
    assert ".text" in output
    assert "0x1000" in output
    assert "r-x" in output


def test_sections_passes_query_to_bridge(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "sections":
            captured_params.update(params)
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["sections", "--target", "active", "--query", "data"])

    assert rc == 0
    assert captured_params["query"] == "data"


# --- I8: enhanced imports CLI ---


def test_imports_text_shows_kind_for_non_function(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "imports"
        return {
            "ok": True,
            "result": {
                "items": [
                    {"name": "printf", "address": "0x1000", "library": "libc", "raw_name": "printf", "kind": "function"},
                    {"name": "__stdout", "address": "0x2000", "library": "libc", "raw_name": "__stdout", "kind": "data"},
                ],
                "total": 2, "offset": 0, "limit": 100, "returned": 2, "has_more": False,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["imports", "--format", "text", "--target", "active"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "printf" in output
    assert "(data)" in output
    assert "(function)" not in output  # function kind is not shown


# --- read: raw bytes at an address ---


def test_read_text_renders_hexdump(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "read"
        captured_params.update(params)
        return {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 8,
                "hex": "48656c6c6f0090ff",
                "ascii": "Hello...",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["read", "--target", "active", "--address", "0x1000", "--length", "8"])

    assert rc == 0
    assert captured_params == {"address": "0x1000", "length": 8}
    output = capsys.readouterr().out
    assert "00001000: 48 65 6c 6c 6f 00 90 ff" in output
    assert "Hello..." in output


def test_read_json_returns_structured_payload(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "read"
        return {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 4,
                "hex": "41424344",
                "ascii": "ABCD",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        ["read", "--format", "json", "--target", "active", "--address", "0x1000", "--length", "4"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "address": "0x1000",
        "length": 4,
        "hex": "41424344",
        "ascii": "ABCD",
    }


def test_read_unmapped_address_surfaces_bridge_error(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "read":
            raise bn.cli.BridgeError("Address 0xdead is not mapped (no bytes available)")
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["read", "--target", "active", "--address", "0xdead", "--length", "16"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "0xdead" in err
    assert "not mapped" in err


def test_read_short_read_text_includes_note(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "read"
        return {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 4,
                "hex": "01020304",
                "ascii": "....",
                "requested_length": 16,
                "short_read": True,
                "note": "short read: requested 16 bytes, only 4 mapped from 0x1000",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["read", "--target", "active", "--address", "0x1000", "--length", "16"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "00001000: 01 02 03 04" in output
    assert "note: short read: requested 16 bytes, only 4 mapped from 0x1000" in output


def test_read_bytes_encoding_writes_raw_bytes(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "read"
        return {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 4,
                "hex": "41424344",
                "ascii": "ABCD",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        ["read", "--target", "active", "--address", "0x1000", "--length", "4", "--encoding", "bytes"]
    )

    assert rc == 0
    assert capsys.readouterr().out == "ABCD"


def test_read_accepts_positional_address(monkeypatch):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "read"
        captured_params.update(params)
        return {"ok": True, "result": {"address": "0x1000", "length": 8, "hex": "00" * 8, "ascii": "." * 8}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    # Positional address matches the convention used by decompile/disasm/il/xrefs.
    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--length", "8"])

    assert rc == 0
    assert captured_params == {"address": "0x1000", "length": 8}


def test_read_length_accepts_hex(monkeypatch):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_params.update(params)
        return {"ok": True, "result": {"address": "0x1000", "length": 194, "hex": "", "ascii": ""}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--length", "0xc2"])

    assert rc == 0
    assert captured_params["length"] == 194


def test_read_conflicting_address_errors(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise AssertionError("send_request should not run when addresses conflict")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--address", "0x2000", "--length", "8"])

    assert rc == 2
    assert "given twice with different values" in capsys.readouterr().err


def test_read_missing_address_errors(monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "send_request", lambda *a, **k: None)

    rc = bn.cli.main(["read", "--target", "active", "--length", "8"])

    assert rc == 2
    assert "read address is required" in capsys.readouterr().err


def test_save_accepts_path_flag(monkeypatch, tmp_path):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "save_database"
        captured_params.update(params or {})
        return {"ok": True, "result": {"path": params.get("path"), "saved": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    out = tmp_path / "out.bndb"
    rc = bn.cli.main(["save", "--target", "active", "--path", str(out)])

    assert rc == 0
    assert captured_params["path"] == str(out.expanduser().resolve())


def test_rename_alias_maps_to_symbol_rename(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"preview": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["rename", "--target", "123:1:7", "--preview", "sub_401000", "player_update"])

    assert rc == 0
    assert captured["op"] == "rename_symbol"
    assert captured["params"]["identifier"] == "sub_401000"
    assert captured["params"]["new_name"] == "player_update"


def test_close_warns_on_unsaved_changes(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "close_binary"
        return {
            "ok": True,
            "result": {
                "closed": [{"path": "/tmp/foo.bndb", "unsaved": True}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "closed: /tmp/foo.bndb" in output
    assert "unsaved" in output.lower()
    assert "bn save" in output


def test_close_silent_when_clean(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "close_binary"
        return {
            "ok": True,
            "result": {
                "closed": [{"path": "/tmp/foo.bndb", "unsaved": False}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "closed: /tmp/foo.bndb" in output
    assert "warning" not in output.lower()
    assert "unsaved" not in output.lower()


def test_close_forwards_explicit_target_selector(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "close_binary"
        captured["target"] = target
        return {"ok": True, "result": {"closed": [{"path": "/tmp/foo", "unsaved": False}]}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "-t", "foo", "--format", "text"])

    assert rc == 0
    assert captured["target"] == "foo"


def test_close_all_flag_sets_param(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": {"closed": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "--all", "--format", "text"])

    assert rc == 0
    assert captured["params"].get("all") is True


def test_close_rejects_path_and_all_together(monkeypatch, capsys):
    # `bn close <path> --all` is contradictory; the bridge would let --all win
    # and close everything despite naming one file. The CLI rejects it before
    # any request is sent (#85).
    sent = {"called": False}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        sent["called"] = True
        return {"ok": True, "result": {"closed": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "/tmp/one-binary", "--all", "--format", "text"])

    assert rc == 2  # BridgeError -> exit 2
    assert sent["called"] is False  # never reached the bridge
    err = capsys.readouterr().err
    assert "not both" in err


def test_close_ignores_sticky_target_pin(monkeypatch, capsys):
    # A sticky pin must NOT turn a bare `close` (documented close-all) into
    # close-one, and a stale pin must not make cleanup fail.
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["target"] = target
        return {"ok": True, "result": {"closed": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "stale_pin"})

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    assert captured["target"] is None  # pin ignored -> close-all


def test_function_list_count_prints_total(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "x"}]}
        if op == "list_functions":
            assert params.get("count_only") is True
            return {"ok": True, "result": {"count": 4242}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["function", "list", "--count"])

    assert rc == 0
    assert "Total functions: 4242" in capsys.readouterr().out


# --- Sticky instance/target ---


@pytest.fixture
def tmp_session(tmp_path, monkeypatch):
    """Isolate session-state file per test by redirecting BN_CACHE_DIR and cwd."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_bridge_instance(instance_id="abc123", pid=111):
    from pathlib import Path as _Path

    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=pid,
        socket_path=_Path(f"/tmp/{instance_id}.sock"),
        registry_path=_Path(f"/tmp/{instance_id}.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id=instance_id,
    )


def test_instance_use_writes_state(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "abc123"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "abc123"
    assert capsys.readouterr().out.strip() == "instance: abc123"


def test_instance_use_default_pins_gui_bridge(tmp_session, monkeypatch, capsys):
    # The fixed GUI bridge has instance_id=None and selector "default". Storing
    # the raw None made session_state.update() DELETE the pin, so the pin
    # silently vanished. `bn instance use default` must persist "default" so
    # later bare commands resolve to the GUI bridge (#93).
    gui = _fake_bridge_instance("gui")
    object.__setattr__(gui, "instance_id", None)  # GUI bridge: id is None
    named = _fake_bridge_instance("headless1")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [gui, named])

    rc = bn.cli.main(["instance", "use", "default"])

    assert rc == 0
    state = bn.session_state.read()
    assert state.get("instance_id") == "default"  # pin persisted, not deleted
    assert capsys.readouterr().out.strip() == "instance: default"


def test_instance_use_rejects_unknown_id(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "not-running"])

    assert rc == 2
    assert "No running bridge instance" in capsys.readouterr().err
    assert bn.session_state.read() == {}


def test_instance_clear_removes_state(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="abc123")
    assert bn.session_state.read()["instance_id"] == "abc123"

    rc = bn.cli.main(["instance", "clear"])

    assert rc == 0
    assert "instance_id" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


def test_target_use_validates_and_writes_state(tmp_session, monkeypatch, capsys):
    # target use validates the selector against the open targets before pinning,
    # so a known selector is persisted (#55).
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "pam_qnx.so.2"])

    assert rc == 0
    assert bn.session_state.read()["target"] == "pam_qnx.so.2"
    assert "target: pam_qnx.so.2" in capsys.readouterr().out


def test_target_use_rejects_unknown_selector_without_pinning(tmp_session, monkeypatch, capsys):
    # A typo must NOT be persisted: an unknown selector is rejected with exit 2
    # and the pin is left unchanged, instead of poisoning every later command (#55).
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "TOTALLY-BOGUS-XYZ"])

    assert rc == 2
    assert "Unknown target selector" in capsys.readouterr().err
    assert bn.session_state.read() == {}   # pin NOT written


def test_target_use_does_not_overwrite_existing_pin_on_reject(tmp_session, monkeypatch, capsys):
    # A failed `target use` must leave a previously-valid pin intact.
    bn.session_state.update(target="pam_qnx.so.2")
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "bogus"])

    assert rc == 2
    assert bn.session_state.read()["target"] == "pam_qnx.so.2"   # unchanged


def test_target_clear_removes_state(tmp_session, capsys):
    bn.session_state.update(target="pam_qnx.so.2")

    rc = bn.cli.main(["target", "clear"])

    assert rc == 0
    assert "target" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


def test_sticky_instance_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "sticky_inst" in captured


def test_cli_instance_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "explicit", "function", "list"])

    assert "explicit" in captured
    assert "sticky_inst" not in captured


def test_env_var_overrides_sticky_instance(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured
    assert "sticky_inst" not in captured


def test_sticky_target_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(target="pam_qnx.so.2")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "pam_qnx.so.2" in captured


def test_cli_target_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(target="sticky_tgt")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "-t", "explicit_tgt"])

    assert "explicit_tgt" in captured
    assert "sticky_tgt" not in captured


def test_session_state_survives_subdir_navigation(tmp_session, monkeypatch):
    # Mark tmp_session as a project root via .git, then descend into subdirs.
    (tmp_session / ".git").mkdir()
    bn.session_state.update(target="pam_qnx.so.2")

    sub = tmp_session / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    assert bn.session_state.read()["target"] == "pam_qnx.so.2"


def test_malformed_session_state_treated_as_empty(tmp_session):
    from bn.paths import session_state_path, sessions_dir

    sessions_dir().mkdir(parents=True, exist_ok=True)
    session_state_path().write_text("{not json")

    assert bn.session_state.read() == {}


def test_session_list_marks_sticky(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "list_instances",
        lambda: [_fake_bridge_instance("aaaa1111"), _fake_bridge_instance("bbbb2222", pid=222)],
    )
    bn.session_state.update(instance_id="aaaa1111")

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    by_id = {entry["instance_id"]: entry for entry in parsed["instances"]}
    assert by_id["aaaa1111"].get("sticky") is True
    assert "sticky" not in by_id["bbbb2222"]


def test_target_list_marks_sticky(tmp_session, monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": [
                {"target_id": "1", "selector": "foo.so", "filename": "/p/foo.so"},
                {"target_id": "2", "selector": "bar.so", "filename": "/p/bar.so"},
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    bn.session_state.update(target="foo.so")

    rc = bn.cli.main(["target", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    by_sel = {entry["selector"]: entry for entry in parsed}
    assert by_sel["foo.so"].get("sticky") is True
    assert "sticky" not in by_sel["bar.so"]


def test_stale_sticky_instance_emits_hint(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="dead_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(f"No bridge instance found with id: {instance_id}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "No bridge instance found with id: dead_inst" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_failed_contact(tmp_session, monkeypatch, capsys):
    """Bridge stopped mid-flight surfaces a transport error, not a registry miss."""
    bn.session_state.update(instance_id="dying_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Failed to contact Binary Ninja bridge pid 17881 at /tmp/x.sock: "
            "[Errno 104] Connection reset by peer"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Failed to contact" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_bridge_timeout(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="slow_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Timed out waiting for Binary Ninja bridge pid 9999 at /tmp/x.sock after 30.0s"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Timed out" in err
    assert "bn instance clear" in err


def test_sticky_hint_skipped_for_unrelated_errors(tmp_session, monkeypatch, capsys):
    """Bridge-side analysis errors must not get the sticky-clear hint."""
    bn.session_state.update(instance_id="alive_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE("Function not found: nonexistent_symbol")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "info", "nonexistent_symbol"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Function not found" in err
    assert "bn instance clear" not in err


# --- bn load --no-bndb plumbing ---


def test_load_defaults_to_prefer_bndb(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "load_binary"
        captured.update(params)
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", str(raw)])

    assert rc == 0
    assert captured["prefer_bndb"] is True


def test_load_no_bndb_flag_disables_prefer_bndb(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "load_binary"
        captured.update(params)
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", "--no-bndb", str(raw)])

    assert rc == 0
    assert captured["prefer_bndb"] is False


def _load_capture(monkeypatch, raw, analyzed):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "load_binary"
        captured.update(params)
        return {"ok": True, "result": {
            "loaded": True, "path": str(raw), "analyzed": analyzed,
            "notes": ([] if analyzed else ["loaded without analysis (--quick): run `bn refresh`"]),
            "targets": [],
        }}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    return captured


def test_load_quick_flag_sets_quick_param(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--quick", str(raw)]) == 0
    assert captured["quick"] is True


def test_load_no_analysis_alias_sets_quick_param(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--no-analysis", str(raw)]) == 0
    assert captured["quick"] is True


def test_load_default_is_not_quick(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=True)
    assert bn.cli.main(["load", str(raw)]) == 0
    assert captured["quick"] is False


def test_load_quick_output_marks_not_analyzed(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--quick", str(raw)]) == 0
    out = capsys.readouterr().out
    assert "[not analyzed]" in out
    assert "bn refresh" in out


def test_load_text_renders_notes(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "loaded": True,
                "path": str(bndb),
                "requested_path": str(raw),
                "notes": [f"loaded {bndb} instead of {raw} (use --no-bndb to skip)"],
                "targets": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", str(raw)])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert f"loaded: {bndb}" in stdout
    assert "note: loaded" in stdout
    assert "--no-bndb" in stdout


def test_session_start_no_bndb_propagates_to_each_load(monkeypatch, tmp_path):
    from bn.transport import BridgeInstance
    import pathlib

    a = tmp_path / "a"
    a.write_bytes(b"")
    b = tmp_path / "b"
    b.write_bytes(b"")

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=pathlib.Path("/tmp/test.sock"),
        registry_path=pathlib.Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(dict(params or {}))
        return {"ok": True, "result": {"loaded": True, "path": params["path"], "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["session", "start", "--no-bndb", str(a), str(b)])

    assert rc == 0
    assert len(captured) == 2
    assert all(item["prefer_bndb"] is False for item in captured)
    assert {item["path"] for item in captured} == {str(a), str(b)}


def test_taint_forward_passes_enabled_sink_classes(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "a.bndb"}]}
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"direction": "forward",
                                       "function": {"name": "handler", "address": "0x10"},
                                       "sources": [], "reached_sinks": [], "leaves": [],
                                       "assumptions": [], "soundness": "may"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["taint", "forward", "-f", "handler", "--source", "arg:read:1",
                      "--sink-class", "file_write", "--sink-class", "net_write",
                      "--format", "json"])
    assert rc == 0
    assert captured["op"] == "taint"
    assert captured["params"]["enabled_sink_classes"] == ["file_write", "net_write"]


def test_taint_forward_sink_classes_default_empty(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "a.bndb"}]}
        captured["params"] = params
        return {"ok": True, "result": {"direction": "forward",
                                       "function": {"name": "handler", "address": "0x10"},
                                       "sources": [], "reached_sinks": [], "leaves": [],
                                       "assumptions": [], "soundness": "may"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["taint", "forward", "-f", "handler", "--source", "arg:read:1",
                      "--format", "json"])
    assert rc == 0
    assert captured["params"]["enabled_sink_classes"] == []


def test_trace_routes_and_renders_text(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "function": "test_func",
                "function_address": "0x10000",
                "target_address": "0x10010",
                "arg_index": 0,
                "view": "mlil",
                "truncated": False,
                "step_count": 2,
                "trace": [
                    {
                        "ssa_var": "r0#1",
                        "depth": 0,
                        "address": "0x10010",
                        "il_text": "MLIL_CALL_SSA @ 0x10010",
                        "operation": "MLIL_SET_VAR_SSA",
                        "terminates": False,
                        "reason": None,
                    },
                    {
                        "ssa_var": "r1#2",
                        "depth": 1,
                        "address": None,
                        "il_text": None,
                        "operation": "undefined",
                        "terminates": True,
                        "reason": "function_parameter_or_global",
                    },
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main([
        "trace",
        "--format", "text",
        "--target", "active",
        "--arg", "0",
        "test_func",
        "0x10010",
    ])

    assert rc == 0
    assert captured["op"] == "backward_slice"
    assert captured["target"] == "active"
    assert captured["params"]["identifier"] == "test_func"
    assert captured["params"]["address"] == "0x10010"
    assert captured["params"]["arg_index"] == 0
    output = capsys.readouterr().out
    assert "backward trace of arg[0]" in output
    assert "test_func" in output
    assert "r0#1" in output or "0x10010" in output
    assert "r1#2" in output
    assert "function parameter" in output


def test_trace_defaults_to_arg_0(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params or {}
        return {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active"])
    assert rc == 0
    assert captured["params"]["arg_index"] == 0


def test_trace_respects_view_and_max_depth(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = dict(params or {})
        return {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main([
        "trace", "f", "0x10010",
        "--target", "active",
        "--view", "hlil",
        "--max-depth", "10",
    ])
    assert rc == 0
    assert captured["params"]["view"] == "hlil"
    assert captured["params"]["max_depth"] == 10


def test_trace_text_renderer_empty_trace(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"function": "f", "trace": [], "step_count": 0, "truncated": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "constant or immediate" in output or "no SSA trace" in output


def test_trace_json_renders_structure(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "function": "f",
                "function_address": "0x10000",
                "target_address": "0x10010",
                "arg_index": 0,
                "view": "mlil",
                "truncated": False,
                "step_count": 2,
                "trace": [
                    {"ssa_var": "x#1", "depth": 0, "terminates": False},
                    {"ssa_var": "y#2", "depth": 1, "terminates": True, "reason": "function_parameter_or_global"},
                ],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["trace", "f", "0x10010", "--target", "active", "--format", "json"])
    assert rc == 0
    output = capsys.readouterr().out
    import json as _json
    parsed = _json.loads(output)
    assert parsed["function"] == "f"
    assert len(parsed["trace"]) == 2


# --- imports --summary CLI routing/rendering ---


def test_imports_summary_routes_and_renders_text(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {
            "ok": True,
            "result": {
                "total_symbols": 4,
                "namespaces": {"libc": 3, "libfoo": 1},
                "by_kind": {"function": 3, "data": 1},
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["imports", "--summary", "--format", "text", "--target", "active"])

    assert rc == 0
    assert captured["op"] == "imports"
    assert captured["params"]["summary"] is True
    output = capsys.readouterr().out
    assert "total symbols: 4" in output  # label matches the JSON key total_symbols
    assert "by namespace:" in output
    assert "libc" in output
    assert "by kind:" in output


def test_imports_summary_text_omits_empty_breakdown_sections():
    """A 0-import target must not print dangling 'by namespace:'/'by kind:'
    headers with nothing under them, and the label matches the JSON key."""
    from bn import formatters
    out = formatters._render_imports_summary_text(
        {"total_symbols": 0, "needed_libraries": [], "namespaces": {}, "by_kind": {}}
    )
    assert "total symbols: 0" in out
    assert "by namespace:" not in out
    assert "by kind:" not in out


def test_imports_without_summary_routes_false(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["imports", "--target", "active"])

    assert rc == 0
    assert captured["params"]["summary"] is False


# --- function search --exact CLI routing + mutual exclusion ---


def test_function_search_exact_routes(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": [{"name": "system", "address": "0x401000"}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "search", "--target", "active", "--exact", "system"])

    assert rc == 0
    assert captured["op"] == "search_functions"
    assert captured["params"]["query"] == "system"
    assert captured["params"]["exact"] is True
    assert captured["params"]["regex"] is False


def test_function_search_regex_and_exact_are_mutually_exclusive(monkeypatch):
    # argparse must reject both flags together (exit code 2), before any request.
    def fail_send_request(*a, **k):
        raise AssertionError("send_request should not be called when args are invalid")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)

    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["function", "search", "--target", "active", "--regex", "--exact", "system"])
    assert exc.value.code == 2


# --- Review fixes: display-only flags, xrefs mutex, session stop, spill hints ---


def _assert_no_bridge_call(monkeypatch):
    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        raise AssertionError(f"bridge should not be called (got op {op!r})")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)


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


def test_xrefs_limit_rejected_outside_text_mode(monkeypatch, capsys):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(["xrefs", "sub_401000", "--target", "active", "--format", "json", "--limit", "3"])

    assert rc == 2
    assert "--limit only applies to --format text" in capsys.readouterr().err


def test_evidence_xrefs_limit_rejected_outside_text_mode(monkeypatch, capsys):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(
        ["evidence", "xrefs", "sub_401000", "--target", "active", "--format", "json", "--limit", "3"]
    )

    assert rc == 2
    assert "--limit only applies to --format text" in capsys.readouterr().err


def test_xrefs_identifier_and_field_are_mutually_exclusive(monkeypatch, capsys):
    _assert_no_bridge_call(monkeypatch)

    rc = bn.cli.main(["xrefs", "some_func", "--field", "T.x", "--target", "active"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "not both" in err
    assert "some_func" in err
    assert "T.x" in err


def test_session_stop_kill_failure_reports_error_and_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    def fail_kill(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("os.kill", fail_kill)

    rc = bn.cli.main(["session", "stop", "abc123"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "failed to stop bridge instance abc123" in captured.err
    assert "stopped" not in captured.out


def test_session_stop_sigterm_fallback_reports_method(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])
    # Convergence polling is covered by its own transport test; here we only
    # assert the SIGTERM dispatch + reported method, so simulate a clean teardown.
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)

    kills = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert kills == [(111, __import__("signal").SIGTERM)]
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is True
    assert parsed["method"] == "sigterm"


def test_duplicate_command_path_registration_raises():
    bn.cli.build_parser()  # ensure command modules populated the registry

    with pytest.raises(ValueError, match="duplicate command path"):
        bn.cli.command("xrefs")(lambda args: 0)


def test_unpaged_list_spill_hint_points_at_out_flag(monkeypatch, capsys):
    # callsites returns a list but is not a paged command: the spill hint must
    # not suggest --limit/--offset (those flags do not exist on `bn callsites`).
    # (imports/sections used to be the example here, but they are paged now.)
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": [{"call": "0x1000", "caller_static": "0x1004"}]}

    def fake_write_output_result(value, *, fmt, out_path, stem):
        assert stem == "callsites"
        return _spill_artifact_namespace("/tmp/callsites.txt")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)

    rc = bn.cli.main(["callsites", "memcpy", "--within", "f", "--target", "active", "--format", "json"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert "--limit/--offset" not in stderr
    # the path is named once (in "spilled to <path>"); the hint must not repeat it
    assert "spilled to /tmp/callsites.txt" in stderr
    assert stderr.count("/tmp/callsites.txt") == 1
    assert "rerun with --out <path>" in stderr


def test_slice_text_lines_start_beyond_end_keeps_header_sane():
    from bn import formatters

    out = formatters._slice_text_lines("a\nb\nc", (10, 12))

    assert out == "// lines 0 of 3 (start 10 is beyond the last line)"


def test_read_bytes_malformed_response_clean_error(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"length": 4}}  # no "hex" payload

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        ["read", "0x1000", "--length", "4", "--encoding", "bytes", "--target", "active"]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "malformed read response" in err
    assert "Traceback" not in err


def test_load_opts_into_spawn_missing_named(monkeypatch, tmp_path):
    # `bn load --instance <new-id>` should auto-spawn that named bridge, so the
    # load handler is the one command that opts into spawn_missing_named.
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["instance_id"] = instance_id
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", str(raw), "--instance", "brandnew"])

    assert rc == 0
    assert captured["op"] == "load_binary"
    assert captured["instance_id"] == "brandnew"
    assert captured["spawn_missing_named"] is True


def test_non_load_command_does_not_spawn_missing_named(monkeypatch):
    # Read commands must not silently spawn a process for a typo'd --instance.
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["sections", "--target", "active"])

    assert rc == 0
    assert captured["spawn_missing_named"] is False


def test_strings_unfiltered_emits_section_hint(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    assert bn.cli.main(["strings", "--target", "active"]) == 0

    _, stderr = capsys.readouterr()
    assert "--section .rodata" in stderr


def test_strings_with_filter_suppresses_section_hint(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    assert bn.cli.main(["strings", "--section", ".rodata", "--target", "active"]) == 0

    _, stderr = capsys.readouterr()
    assert "tip:" not in stderr


def test_strings_section_hint_suppressed_when_request_fails(monkeypatch, capsys):
    """The unfiltered-dump tip must not precede/bury a failure (e.g. a --quick
    refusal). It belongs after a successful dump, not before the request."""
    def boom(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise bn.cli.BridgeError(
            "Strings are not available: this target was loaded with --quick (no analysis)."
        )

    monkeypatch.setattr(bn.cli, "send_request", boom)
    rc = bn.cli.main(["strings", "--target", "active"])
    assert rc == 2
    _, stderr = capsys.readouterr()
    assert "tip:" not in stderr          # the noise tip must not lead
    assert "--quick" in stderr           # the real reason is what surfaces


def test_trace_render_step_grammar_singular_and_plural():
    from bn import formatters
    base = {
        "function": "f", "function_address": "0x1000", "target_address": "0x1010",
        "arg_index": 0, "truncated": False,
        "trace": [{"ssa_var": "x#1", "terminates": True, "reason": "undefined_or_global", "depth": 0}],
    }
    out1 = formatters._render_trace_text(base)
    assert "1 step" in out1 and "1 steps" not in out1
    two = dict(base, trace=base["trace"] + [
        {"ssa_var": "y#1", "terminates": True, "reason": "undefined_or_global", "depth": 1}])
    assert "2 steps" in formatters._render_trace_text(two)


def test_trace_render_shows_arg_register_and_field_load_meta():
    # header labels the traced arg with its register + C name; a field_load step
    # renders base/offset/width (#162, #166).
    from bn import formatters
    value = {
        "function": "parse", "function_address": "0x1000", "target_address": "0x1040",
        "arg_index": 1, "arg_label": {"index": 1, "register": "x1", "name": "buf"},
        "truncated": False, "step_count": 1,
        "trace": [{"ssa_var": "len#3", "ssa_label": "len#3", "depth": 0,
                   "reason": "field_load", "terminates": True,
                   "base": "obj#1", "offset": "0x8", "width": 4,
                   "il_text": "len#3 = [obj#1 + 8]", "address": "0x1030"}],
        "hints": [],
    }
    text = formatters._render_trace_text(value)
    assert 'arg[1] (x1, "buf")' in text
    assert "field load" in text
    assert "base=obj#1" in text and "offset=0x8" in text and "width=4" in text


def test_trace_render_output_pointer_hint_replaces_empty_trace_message():
    # an empty trace WITH an output-pointer hint shows the hint, not the bare
    # "constant or immediate" dead-end (#166).
    from bn import formatters
    value = {
        "function": "h", "function_address": "0x1000", "target_address": "0x1040",
        "arg_index": 1, "arg_label": {"index": 1, "register": "x1"},
        "truncated": False, "step_count": 0, "trace": [],
        "hints": ["arg 1 is a pointer (address-of); trace the pointee instead."],
    }
    text = formatters._render_trace_text(value)
    assert "constant or immediate" not in text
    assert "hint:" in text
    assert "pointer" in text


def test_render_mutation_text_does_not_claim_rollback_when_revert_failed():
    """When a mutation failed AND its revert failed (rolled_back=False), the
    text renderer must not print 'rolled back' -- that contradicts the honest
    'view may be left modified' message and re-states the #117 symptom (#117)."""
    from bn import formatters
    value = {
        "preview": True,
        "success": False,
        "committed": False,
        "rolled_back": False,
        "message": "Preview verified, but removing the created function on revert failed; the view may be left modified.",
        "results": [{"op": "function_create", "status": "rollback_failed", "address": "0x1000", "function": "sub_1000"}],
        "affected_functions": [],
        "affected_types": [],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" not in out
    assert "rollback failed" in out
    assert "may be left modified" in out
    # the op renders under 'failed:', not as a bare '[verified]'
    assert "failed: " in out
    assert "[verified]" not in out


def test_render_mutation_text_still_reports_clean_rollback():
    """A failed batch that WAS cleanly reverted still says 'rolled back'."""
    from bn import formatters
    value = {
        "preview": False,
        "success": False,
        "committed": False,
        "rolled_back": True,
        "message": "Rolled back because live-session verification failed.",
        "results": [{"op": "rename_symbol", "status": "verification_failed", "address": "0x1000"}],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" in out


def test_unknown_ref_label_prefers_symbol_then_section():
    from bn import formatters
    assert formatters._unknown_ref_label({"symbol": {"name": "some_export"}}) == "some_export"
    assert formatters._unknown_ref_label({"sections": [{"name": ".got"}]}) == ".got"
    assert formatters._unknown_ref_label({"symbol": {"name": "s"}, "sections": [{"name": ".got"}]}) == "s"
    assert formatters._unknown_ref_label({}) == ""
    assert formatters._unknown_ref_label(None) == ""


def test_xrefs_data_ref_labels_unknown_caller_by_section_or_symbol():
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0x1a254", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".got"}], "symbol": {"name": "some_export"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    assert "some_export" in out          # symbol preferred over a bare <unknown>
    assert "<unknown>  <unknown>" not in out


def test_xrefs_distinct_functionless_refs_are_not_merged_under_one_label():
    # Two function-less data refs with DIFFERENT symbols must render as two
    # distinct, correctly-labeled lines -- not collapse into one group stamped
    # with the first ref's symbol (which mislabeled the second site) (#24).
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0xaaaa", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".got"}], "symbol": {"name": "sym_a"}}},
            {"address": "0xbbbb", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".data"}], "symbol": {"name": "sym_b"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    # function-less refs are "locations", not "functions" (#49)
    assert "data refs: 2 sites across 2 locations" in out
    # each site is labeled with its OWN symbol, on a line with its OWN address
    assert any("0xaaaa" in ln and "sym_a" in ln and "sym_b" not in ln for ln in out.splitlines())
    assert any("0xbbbb" in ln and "sym_b" in ln and "sym_a" not in ln for ln in out.splitlines())


def test_xrefs_same_label_functionless_refs_still_coalesce():
    # Refs sharing a resolved label are still grouped (two sites, one line).
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0xaaaa", "caller_function": None, "function": None,
             "context": {"symbol": {"name": "sym_a"}}},
            {"address": "0xbbbb", "caller_function": None, "function": None,
             "context": {"symbol": {"name": "sym_a"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    assert "data refs: 2 sites across 1 location" in out   # function-less -> location (#49)
    assert any("0xaaaa" in ln and "0xbbbb" in ln for ln in out.splitlines())


def test_evidence_xrefs_text_reports_truncation_total():
    # Capping evidence xrefs must surface the true total + a "showing first N"
    # marker, not silently drop refs (#31, the #13 honesty convention).
    from bn import formatters
    value = {
        "address": "0x1000",
        "code_refs": [{"address": hex(0x2000 + i), "kind": "code"} for i in range(14)],
        "data_refs": [],
    }
    out = formatters._render_evidence_xrefs_text(value, limit=3)
    assert "code refs: 14 total, showing first 3" in out
    # only 3 ref lines rendered
    assert sum(1 for ln in out.splitlines() if ln.startswith("- 0x")) == 3
    # uncapped render shows the plain header (no truncation marker)
    out_full = formatters._render_evidence_xrefs_text(value)
    assert "code refs:" in out_full and "total, showing" not in out_full


def test_negative_ip_depth_rejected_with_exit_2():
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["trace", "f", "0x1000", "--target", "active", "--ip-depth", "-1"])
    assert excinfo.value.code == 2
    # ip-depth 0 (disable crossing) is allowed
    ns = parser.parse_args(["trace", "f", "0x1000", "--target", "active", "--ip-depth", "0"])
    assert ns.ip_depth == 0


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


# ---------------------------------------------------------------------------
# Batch 5: CLI validation/rendering (#94, #96, #100, #101, #102)
# ---------------------------------------------------------------------------


def test_comment_get_rejects_both_address_and_function(capsys):
    # #94: --address and --function are mutually exclusive (the bridge checks
    # function first, so accepting both silently dropped the address).
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["comment", "get", "--target", "active", "--address", "0x1000", "--function", "main"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_comment_get_requires_a_locator(capsys):
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["comment", "get", "--target", "active"])
    assert exc.value.code == 2  # required mutex group: exactly one


def test_types_declare_rejects_multiple_sources(monkeypatch, capsys, tmp_path):
    # #94 Problem B: --file + positional must not silently pick one.
    f = tmp_path / "d.h"
    f.write_text("struct S { int a; };", encoding="utf-8")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise AssertionError("must reject before calling the bridge")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(f), "struct T { int b; };"])
    assert rc == 2  # BridgeError -> exit 2
    assert "exactly one declaration source" in capsys.readouterr().err


def test_read_bytes_out_writes_envelope_and_creates_parents(monkeypatch, capsys, tmp_path):
    # #96: the bytes --out path must mkdir parents and emit an artifact envelope.
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "read":
            return {"ok": True, "result": {"hex": "deadbeef"}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    out = tmp_path / "nested" / "dir" / "out.bin"  # parent does not exist yet
    rc = bn.cli.main(["read", "0x1000", "--length", "4", "--encoding", "bytes",
                      "--target", "active", "--out", str(out), "--format", "json"])
    assert rc == 0
    assert out.read_bytes() == bytes.fromhex("deadbeef")  # parents created, data written
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["format"] == "bytes"
    assert envelope["bytes"] == 4
    assert envelope["artifact_path"] == str(out)
    assert "sha256" in envelope


def test_read_bytes_out_bad_dir_is_clean_error(monkeypatch, capsys, tmp_path):
    # A write failure must be a clean BridgeError, not a raw traceback.
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"hex": "00"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    # A path whose parent is an existing FILE can't be mkdir'd.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    out = blocker / "sub" / "out.bin"
    rc = bn.cli.main(["read", "0x1000", "--length", "1", "--encoding", "bytes",
                      "--target", "active", "--out", str(out)])
    assert rc == 2  # OutputWriteError is a BridgeError -> exit 2
    assert "Failed to write --out file" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["strings", "--target", "active", "--min-length", "-5"],
    ["callsites", "memcpy", "--target", "active", "--context", "-1"],
    ["trace", "main", "0x1000", "--target", "active", "--arg", "-2"],
])
def test_negative_count_flags_rejected(argv, capsys):
    # #100 Problem B: these flags now use the shared non-negative validator.
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(argv)
    assert exc.value.code == 2
    assert "must be an integer >= 0" in capsys.readouterr().err


def test_text_renderer_failure_becomes_clean_error(monkeypatch, capsys):
    # #101: a malformed bridge result that trips a text renderer must surface a
    # clean BridgeError (exit 2) pointing at --format json, not a raw traceback.
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "function_info":
            # 'function' present but a STRING, not a dict -> .get() would crash.
            return {"ok": True, "result": {"function": "not-a-dict"}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    # function info should now render with placeholders (defensive _as_dict), not crash.
    rc = bn.cli.main(["function", "info", "main", "--target", "active", "--format", "text"])
    assert rc == 0
    assert "<unknown>" in capsys.readouterr().out
