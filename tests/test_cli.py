from __future__ import annotations

from pathlib import Path

import bn.cli


def test_function_list_defaults_to_active_target(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": [{"name": "sub_401000", "address": "0x401000"}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    assert rc == 0
    assert captured["op"] == "list_functions"
    assert captured["target"] == "active"
    assert "sub_401000" in capsys.readouterr().out


def test_function_info_uses_active_target_and_text_renderer(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "prototype": "int32_t sub_401000(int32_t arg1)",
                "parameters": [{"name": "arg1", "type": "int32_t", "storage": 0, "is_parameter": True}],
                "locals": [{"name": "var_4", "type": "int32_t", "storage": -4, "is_parameter": False}],
                "stack_vars": [{"name": "var_4", "type": "int32_t", "storage": -4, "is_parameter": False}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "info", "--format", "text", "sub_401000"])

    assert rc == 0
    assert captured["op"] == "function_info"
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert "sub_401000 @ 0x401000" in output
    assert "parameters:" in output
    assert "locals:" in output


def test_symbol_rename_builds_preview_payload(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
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


def test_symbol_rename_defaults_to_active_when_single_target_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
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
    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
        if op == "list_targets":
            return {
                "ok": True,
                "result": [
                    {"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"},
                    {"target_id": "123:2:8", "selector": "other.exe.bndb"},
                ],
            }
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "sub_401000", "player_update"])

    assert rc == 2
    assert "requires --target when multiple targets are open" in capsys.readouterr().err


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


def test_types_show_uses_type_info_and_text_renderer(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
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

    rc = bn.cli.main(["types", "show", "--format", "text", "Player"])

    assert rc == 0
    assert captured["op"] == "type_info"
    assert captured["params"]["type_name"] == "Player"
    assert captured["target"] == "active"
    output = capsys.readouterr().out
    assert output.startswith("struct Player")
    assert '"decl"' not in output


def test_types_declare_defaults_to_active_when_single_target_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
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


def test_py_exec_accepts_inline_code(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
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


def test_py_exec_missing_script_mentions_code(capsys):
    rc = bn.cli.main(["py", "exec", "--target", "active", "--script", "missing.py"])

    assert rc == 2
    assert "Use --code for inline Python" in capsys.readouterr().err


def test_decompile_text_format_unwraps_text_field(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, instance_pid=None, timeout=30.0):
        return {
            "ok": True,
            "result": {
                "function": {"name": "sub_401000", "address": "0x401000"},
                "text": "return 7;",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["decompile", "--format", "text", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "return 7;\n"
