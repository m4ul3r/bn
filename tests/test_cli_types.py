from __future__ import annotations

import os
import threading
import time

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
    assert calls[-1]["target"] == "active"  # explicit -t active passes through for reads
    output = capsys.readouterr().out
    assert output.startswith("struct Player")
    assert '"decl"' not in output


def test_types_declare_uses_implicit_target_when_single_target_is_open(fake_transport):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}]},
        "types_declare": {"ok": True, "result": {"preview": True, "results": [{"status": "verified"}]}},
    })

    rc = bn.cli.main(["types", "declare", "typedef struct Player { int hp; } Player;"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "types_declare"]
    assert calls[1]["target"] == "123:1:7"  # implicit resolution pins the target_id (#690 R3)
    assert "typedef struct Player" in calls[1]["params"]["declaration"]


def test_types_declare_passes_source_path_for_file_input(fake_transport, tmp_path):
    declaration_file = tmp_path / "win32_min.h"
    declaration_file.write_text("typedef struct Player { int hp; } Player;", encoding="utf-8")
    calls = fake_transport({"types_declare": {"ok": True, "result": {"preview": False, "success": True, "results": [{"status": "verified"}]}}})

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(declaration_file)])

    assert rc == 0
    assert calls[-1]["op"] == "types_declare"
    assert calls[-1]["params"]["source_path"] == str(declaration_file)


def test_declaration_missing_file_is_not_a_semantic_refusal(fake_transport, capsys, tmp_path):
    calls = fake_transport()
    assert bn.cli.main(["types", "declare", "--file", str(tmp_path / "missing.h"),
                        "--format", "json"]) == 2
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "status" not in payload
    assert calls == []


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
    assert rc == 3
    assert "exactly one declaration source" in capsys.readouterr().err


def test_proto_get_splices_function_name_into_anonymous_prototype(monkeypatch, capsys):
    """BN renders the prototype anonymously; the renderer splices in the function
    name so it's a copy-pasteable C declaration (#222)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {
            "function": {"name": "parse_image", "address": "0x401000"},
            "prototype": "uint64_t (int32_t arg1, char* arg2)",
        }}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["proto", "get", "--format", "text", "--target", "active", "parse_image"])
    assert rc == 0
    assert capsys.readouterr().out == "uint64_t parse_image(int32_t arg1, char* arg2)\n"


def test_proto_get_splices_name_that_is_substring_of_return_type(monkeypatch, capsys):
    """A function name that is a substring of the return type must still be
    spliced (the naive `name in head` guard wrongly skipped it) (#222 review)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {
            "function": {"name": "t", "address": "0x1000"},
            "prototype": "uint64_t (int32_t a)"}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["proto", "get", "--format", "text", "--target", "active", "t"])
    assert rc == 0
    assert capsys.readouterr().out == "uint64_t t(int32_t a)\n"


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_struct_field_rename_rejects_empty_new_name(fake_transport, capsys, bad_name):
    """An empty/whitespace-only new name is refused client-side (exit 3) before
    any struct_field_rename op is sent -- mirrors _symbol_rename's guard (#605)."""
    calls = fake_transport({"struct_field_rename": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(["struct", "field", "rename", "--target", "123:1:7", "S", "old", bad_name])

    assert rc == 3
    assert "new name must be non-empty" in capsys.readouterr().err
    assert [call["op"] for call in calls] == []


@pytest.mark.parametrize("mode", ["directory", "undecodable"])
def test_types_declare_file_failures_are_structured_refusals_754(
    fake_transport, capsys, tmp_path, mode
):
    """#754: `--file` guarded only `exists()`, so a directory (true for `exists()`)
    reached `read_text` and died with a raw IsADirectoryError -- exit 1 and 0 bytes
    of stdout, the one shape in this family that was not a structured refusal. A
    non-UTF-8 file failed the same way through UnicodeDecodeError. Both must refuse
    like the missing-file case: exit 2, a JSON envelope, no traceback, no op sent."""
    if mode == "directory":
        target = tmp_path
    else:
        target = tmp_path / "decls.h"
        target.write_bytes(b"struct S { int a; };\n\xff\xfe")
    calls = fake_transport()  # empty results -> any bridge call raises

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(target)])

    assert rc == 2
    assert [call["op"] for call in calls] == []
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    assert '"ok":false' in captured.out.replace(" ", "")
    assert str(target) in captured.out


def test_types_declare_refuses_a_fifo_instead_of_hanging(fake_transport, capsys, tmp_path):
    """#864: `--file <fifo>` blocked forever with no output and no envelope; the
    shared reader refuses the one FIFO shape that cannot terminate -- nobody is
    writing and nothing is buffered -- and says which of the two it is, because
    "not a regular file" would also condemn the process substitutions below."""
    fifo = tmp_path / "decl.h"
    os.mkfifo(fifo)
    calls = fake_transport()

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(fifo)])

    assert rc == 2
    assert [call["op"] for call in calls] == []
    captured = capsys.readouterr()
    assert "FIFO" in captured.err
    assert "no writer" in captured.err
    assert str(fifo) in captured.err
    assert "Traceback" not in captured.err


def _declare_ok():
    return {"types_declare": {"ok": True, "result": {"preview": False, "success": True,
                                                     "results": [{"status": "verified"}]}}}


def test_types_declare_reads_a_buffered_process_substitution(fake_transport):
    """#864 asked the decision to cover process substitution; a blanket
    non-regular refusal answers it by deleting it. A shell hands `<(printf ...)`
    over as /dev/fd/N -- a FIFO whose bytes are already buffered and whose
    writer has exited -- and that read terminated on base, so refusing it is a
    regression, not a guardrail."""
    calls = fake_transport(_declare_ok())
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"struct P { int hp; };")
    os.close(write_fd)
    try:
        rc = bn.cli.main(
            ["types", "declare", "--target", "active", "--file", f"/dev/fd/{read_fd}"])
    finally:
        os.close(read_fd)

    assert rc == 0
    assert calls[-1]["op"] == "types_declare"
    assert calls[-1]["params"]["declaration"] == "struct P { int hp; };"


def test_types_declare_waits_for_a_running_process_substitution_writer(fake_transport):
    """The other half of the same shape: `<(sleep 1; gen)` has a writer attached
    but no first byte yet, so the non-blocking probe sees EAGAIN rather than
    EOF. Refusing on "nothing buffered yet" would break every generator that is
    not instantaneous; the read waits for the writer it can see."""
    calls = fake_transport(_declare_ok())
    read_fd, write_fd = os.pipe()

    def _write_late():
        time.sleep(0.05)
        os.write(write_fd, b"struct Q { int a; };")
        os.close(write_fd)

    writer = threading.Thread(target=_write_late)
    writer.start()
    try:
        rc = bn.cli.main(
            ["types", "declare", "--target", "active", "--file", f"/dev/fd/{read_fd}"])
    finally:
        writer.join(timeout=10)
        os.close(read_fd)

    assert rc == 0
    assert calls[-1]["params"]["declaration"] == "struct Q { int a; };"


def test_types_declare_dev_null_still_reaches_the_op(fake_transport):
    """The #864 rule must not swallow /dev/null, which #754/#855 deliberately
    keep working as an empty declaration."""
    calls = fake_transport({
        "types_declare": {"ok": True, "result": {"preview": False, "success": True,
                                                 "results": [{"status": "verified"}]}},
    })

    rc = bn.cli.main(["types", "declare", "--target", "active", "--file", "/dev/null"])

    assert rc == 0
    assert calls[-1]["op"] == "types_declare"
    assert calls[-1]["params"]["declaration"] == ""
