from __future__ import annotations

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_types_show_uses_type_info_and_text_renderer(fake_transport, capsys):
    calls = fake_transport({"type_info": {
        "ok": True,
        "result": {
            "name": "Player",
            "kind": "StructureTypeClass",
            "decl": "struct Player",
            "layout": "struct Player // size=0x10\n0x0000: int32_t hp",
        },
    }})

    rc = bn.cli.main(["types", "show", "--format", "text", "--target", "active", "Player"])

    assert rc == 0
    assert calls[-1]["op"] == "type_info"
    assert calls[-1]["params"]["type_name"] == "Player"
    assert calls[-1]["params"]["require_struct"] is False
    assert calls[-1]["target"] == "active"
    output = capsys.readouterr().out
    assert output.startswith("struct Player")
    assert '"decl"' not in output


def test_types_declare_uses_implicit_target_when_single_target_is_open(fake_transport):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}]},
        "types_declare": {"ok": True, "result": {"preview": True}},
    })

    rc = bn.cli.main(["types", "declare", "typedef struct Player { int hp; } Player;"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "types_declare"]
    assert calls[1]["target"] == "active"
    assert "typedef struct Player" in calls[1]["params"]["declaration"]


def test_types_declare_passes_source_path_for_file_input(fake_transport, tmp_path):
    declaration_file = tmp_path / "win32_min.h"
    declaration_file.write_text("typedef struct Player { int hp; } Player;", encoding="utf-8")
    calls = fake_transport({"types_declare": {"ok": True, "result": {"preview": False, "success": True, "results": []}}})

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(declaration_file)])

    assert rc == 0
    assert calls[-1]["op"] == "types_declare"
    assert calls[-1]["params"]["source_path"] == str(declaration_file)


def test_proto_get_renders_prototype_text(fake_transport, capsys):
    fake_transport({"get_prototype": {
        "ok": True,
        "result": {
            "function": {"name": "sub_401000", "address": "0x401000"},
            "prototype": "int32_t sub_401000(int32_t arg1)",
            "return_type": "int32_t",
            "calling_convention": "__cdecl",
        },
    }})

    rc = bn.cli.main(["proto", "get", "--format", "text", "--target", "active", "sub_401000"])

    assert rc == 0
    assert capsys.readouterr().out == "int32_t sub_401000(int32_t arg1)\n"


def test_types_declare_rejects_multiple_sources(fake_transport, capsys, tmp_path):
    # #94 Problem B: --file + positional must not silently pick one.
    f = tmp_path / "d.h"
    f.write_text("struct S { int a; };", encoding="utf-8")
    fake_transport()  # empty results -> any bridge call raises; rejection must precede it

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(f), "struct T { int b; };"])
    assert rc == 2  # BridgeError -> exit 2
    assert "exactly one declaration source" in capsys.readouterr().err
