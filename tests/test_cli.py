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
