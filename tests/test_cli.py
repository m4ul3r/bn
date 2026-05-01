from __future__ import annotations

import json
import types

import bn.cli
import pytest


def test_function_list_uses_implicit_target_when_single_target_is_open(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "list_functions":
            return {"ok": True, "result": [{"name": "sub_401000", "address": "0x401000"}]}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "list_functions"]
    assert calls[1]["params"] == {"limit": 101}
    assert calls[1]["target"] == "active"
    output = capsys.readouterr().out
    assert output == "0x401000  sub_401000\n"
    assert '"name"' not in output


def test_function_list_requires_target_when_multiple_targets_are_open(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": [{"name": f"sub_{index:06x}", "address": hex(index)} for index in range(150)],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list", "--target", "active", "--format", "json", "--limit", "200"])

    assert rc == 0
    assert captured["op"] == "list_functions"
    assert captured["params"] == {"limit": 201}
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert len(payload) == 150
    assert stderr == ""


def test_function_list_warns_when_output_auto_spills(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured["op"] = op
        return {
            "ok": True,
            "result": [
                {"name": "sub_401000", "address": "0x401000"},
                {"name": "sub_402000", "address": "0x402000"},
            ],
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
                "tokenizer: o200k_base\n"
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
                "tokenizer": "o200k_base",
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
    assert stderr == "warning: function list output spilled to /tmp/functions.txt\n"


def test_function_list_forwards_address_filters(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": [{"name": "load_attachment", "address": "0x401000"}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "search", "--target", "active", "--regex", "attach|detach"])

    assert rc == 0
    assert captured["op"] == "search_functions"
    assert captured["params"]["query"] == "attach|detach"
    assert captured["params"]["regex"] is True
    assert "offset" not in captured["params"]
    assert captured["params"]["limit"] == 101
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

    # Mutations now default to text — the rendered summary is enough for an agent
    # and JSON is one --format json away when needed.
    assert parser.parse_args(["symbol", "rename", "sub_401000", "player_update"]).format == "text"
    assert parser.parse_args(["types", "declare", "typedef struct Player { int hp; } Player;"]).format == "text"
    assert parser.parse_args(["comment", "set", "--address", "0x401000", "msg"]).format == "text"
    assert parser.parse_args(["proto", "set", "sub_401000", "void()"]).format == "text"
    assert parser.parse_args(["local", "rename", "fn", "var", "new"]).format == "text"
    assert parser.parse_args(["struct", "field", "set", "S", "0", "f", "uint32_t"]).format == "text"
    assert parser.parse_args(["batch", "apply", "manifest.json"]).format == "text"


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


def test_callsites_missing_scope_raises_actionable_error(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert output.startswith("struct Player")
    assert '"decl"' not in output


def test_types_declare_uses_implicit_target_when_single_target_is_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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


def test_callsites_routes_within_scope_and_renders_text(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        assert op == "strings"
        return {
            "ok": True,
            "result": [
                {
                    "address": "0x500000",
                    "length": 6,
                    "type": "AsciiString",
                    "value": "follow",
                }
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--format", "text", "--target", "active", "--query", "follow"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '0x500000  len=6  AsciiString  "follow"' in output
    assert '"value"' not in output


def test_py_exec_text_format_renders_stdout_and_result(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    # internal IDs must not leak into text mode
    assert "0x401000:param:stack:4:0:1" not in output
    assert "storage=" not in output
    assert "source=" not in output
    assert "identifier=" not in output


def test_local_list_json_retains_ids(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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


def test_bundle_function_out_path_is_bridge_owned(monkeypatch, tmp_path, capsys):
    captured = {}
    out_path = tmp_path / "bundle.json"

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    assert f"path: {out_path}" in output
    assert "spilled: false" in output


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


def test_symbol_rename_text_format_renders_mutation_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        assert op == "get_comment"
        return {"ok": True, "result": {"address": "0x401000", "comment": "", "has_comment": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--target", "active", "--address", "0x401000"])

    assert rc == 0
    assert capsys.readouterr().out == "(no comment)\n"


def test_callsites_empty_result_shows_descriptive_message(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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


def test_function_list_pagination_truncates_and_warns(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured["params"] = params
        return {
            "ok": True,
            "result": [{"name": f"sub_{i:06x}", "address": hex(i)} for i in range(21)],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list", "--target", "active", "--limit", "20"])

    assert rc == 0
    assert captured["params"]["limit"] == 21
    stdout, stderr = capsys.readouterr()
    assert stdout.count("\n") == 20
    assert "truncated to 20 items" in stderr
    assert "--offset 20" in stderr


def test_function_search_pagination_forwards_offset(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured["params"] = params
        return {"ok": True, "result": [{"name": "sub_401000", "address": "0x401000"}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "search", "--target", "active", "--offset", "50", "--limit", "25", "sub"])

    assert rc == 0
    assert captured["params"]["offset"] == 50
    assert captured["params"]["limit"] == 26


def test_instance_flag_passed_to_send_request(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "abc123", "function", "list"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_on_subcommand(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "--instance", "abc123"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_from_env(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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


# --- I2: strings filtering CLI args ---


def test_strings_passes_min_length_to_bridge(monkeypatch, capsys):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        if op == "strings":
            captured_params.update(params)
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--section", ".rodata", "--no-crt"])

    assert rc == 0
    assert captured_params["section"] == ".rodata"
    assert captured_params["no_crt"] is True


# --- I5: sections CLI ---


def test_sections_text_format_renders_rows(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        assert op == "sections"
        return {
            "ok": True,
            "result": [
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        assert op == "imports"
        return {
            "ok": True,
            "result": [
                {"name": "printf", "address": "0x1000", "library": "libc", "raw_name": "printf", "kind": "function"},
                {"name": "__stdout", "address": "0x2000", "library": "libc", "raw_name": "__stdout", "kind": "data"},
            ],
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["imports", "--format", "text", "--target", "active"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "printf" in output
    assert "(data)" in output
    assert "(function)" not in output  # function kind is not shown


def test_close_warns_on_unsaved_changes(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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


def test_target_use_writes_state_without_bridge_call(tmp_session, monkeypatch, capsys):
    def fail(*_a, **_kw):
        raise AssertionError("target use must not call the bridge")

    monkeypatch.setattr(bn.cli, "send_request", fail)

    rc = bn.cli.main(["target", "use", "pam_qnx.so.2"])

    assert rc == 0
    assert bn.session_state.read()["target"] == "pam_qnx.so.2"
    assert "target: pam_qnx.so.2" in capsys.readouterr().out


def test_target_clear_removes_state(tmp_session, capsys):
    bn.session_state.update(target="pam_qnx.so.2")

    rc = bn.cli.main(["target", "clear"])

    assert rc == 0
    assert "target" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


def test_sticky_instance_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "pam_qnx.so.2" in captured


def test_cli_target_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(target="sticky_tgt")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        assert op == "load_binary"
        captured.update(params)
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", "--no-bndb", str(raw)])

    assert rc == 0
    assert captured["prefer_bndb"] is False


def test_load_text_renders_notes(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None):
        captured.append(dict(params or {}))
        return {"ok": True, "result": {"loaded": True, "path": params["path"], "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["session", "start", "--no-bndb", str(a), str(b)])

    assert rc == 0
    assert len(captured) == 2
    assert all(item["prefer_bndb"] is False for item in captured)
    assert {item["path"] for item in captured} == {str(a), str(b)}
