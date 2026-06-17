from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


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
