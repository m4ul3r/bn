from __future__ import annotations

import contextlib
import os
import resource
import signal
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


@contextlib.contextmanager
def _must_not_hang(seconds: float = 10.0):
    """Turn a hang into a failure for the #864 tests below.

    Every one of them asserts that some FIFO shape TERMINATES, and a read that
    never returns is the exact defect #864 reports -- but an unbounded read
    makes pytest stall rather than fail, so a regression would score as "still
    running" instead of as a red test, and would take the whole suite with it.
    """
    def _fire(signum, frame):
        raise AssertionError(f"the call never returned ({seconds:g}s) -- it hung")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_types_declare_refuses_a_fifo_instead_of_hanging(fake_transport, capsys, tmp_path):
    """#864: `--file <fifo>` blocked forever with no output and no envelope. The
    shared reader refuses on what the stream DELIVERED, so the refusal names
    that rather than "not a regular file", which would also condemn the process
    substitutions below."""
    fifo = tmp_path / "decl.h"
    os.mkfifo(fifo)
    calls = fake_transport()

    with _must_not_hang():
        rc = bn.cli.main(["types", "declare", "--target", "active", "--file", str(fifo)])

    assert rc == 2
    assert [call["op"] for call in calls] == []
    captured = capsys.readouterr()
    assert "FIFO" in captured.err
    assert "delivered no data" in captured.err
    assert str(fifo) in captured.err
    assert "Traceback" not in captured.err


def _declare_ok():
    return {"types_declare": {"ok": True, "result": {"preview": False, "success": True,
                                                     "results": [{"status": "verified"}]}}}


def _declare(path: str) -> list[str]:
    return ["types", "declare", "--target", "active", "--file", path]


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
        with _must_not_hang():
            rc = bn.cli.main(_declare(f"/dev/fd/{read_fd}"))
    finally:
        os.close(read_fd)

    assert rc == 0
    assert calls[-1]["op"] == "types_declare"
    assert calls[-1]["params"]["declaration"] == "struct P { int hp; };"


def test_types_declare_waits_for_a_running_process_substitution_writer(
        fake_transport, monkeypatch):
    """The other half of the same shape: `<(sleep 1; gen)` has a writer attached
    but no first byte yet, so the non-blocking probe sees EAGAIN rather than
    data. Refusing on "nothing buffered yet" would break every generator that
    is not instantaneous; the read waits for the writer it can see.

    The writer is released only once the reader has OPENED the pipe, and the
    wait is measured from that open -- otherwise the writer wins the race on a
    slow interpreter start and the test silently degenerates into the buffered
    case above, scoring a read that waited and a read that did not identically.
    """
    calls = fake_transport(_declare_ok())
    read_fd, write_fd = os.pipe()
    path = f"/dev/fd/{read_fd}"
    opened = threading.Event()
    opened_at: list[float] = []
    real_open = os.open

    def spy_open(target, *args, **kwargs):
        fd = real_open(target, *args, **kwargs)
        if str(target) == path:
            opened_at.append(time.monotonic())
            opened.set()
        return fd

    monkeypatch.setattr(os, "open", spy_open)

    def _write_once_the_reader_is_waiting():
        opened.wait(timeout=10)
        time.sleep(0.2)
        os.write(write_fd, b"struct Q { int a; };")
        os.close(write_fd)

    writer = threading.Thread(target=_write_once_the_reader_is_waiting)
    writer.start()
    try:
        with _must_not_hang():
            rc = bn.cli.main(_declare(path))
    finally:
        writer.join(timeout=10)
        os.close(read_fd)
    waited = time.monotonic() - opened_at[0]

    assert rc == 0
    assert calls[-1]["params"]["declaration"] == "struct Q { int a; };"
    assert waited >= 0.19, f"the read did not wait for the writer ({waited:.3f}s)"


def test_a_zero_output_process_substitution_is_refused_the_same_way_either_way(
        fake_transport, capsys):
    """`--file <(cmd)` where cmd writes nothing races the open: sometimes the
    writer has already exited (the probe sees EOF), sometimes it is still
    attached (the probe sees EAGAIN). It is the same input, so it must get the
    same answer -- deciding on whichever state the probe caught means the same
    command is refused or silently accepted as an empty declaration depending
    on machine load."""
    outcomes = []
    for writer_exits_after in (0.0, 0.2):
        calls = fake_transport(_declare_ok())
        read_fd, write_fd = os.pipe()
        closer = None
        if writer_exits_after:
            closer = threading.Thread(
                target=lambda: (time.sleep(writer_exits_after), os.close(write_fd)))
            closer.start()
        else:
            os.close(write_fd)
        try:
            with _must_not_hang():
                rc = bn.cli.main(_declare(f"/dev/fd/{read_fd}"))
        finally:
            if closer is not None:
                closer.join(timeout=10)
            os.close(read_fd)
        err = capsys.readouterr().err.replace(f"/dev/fd/{read_fd}", "<pipe>")
        outcomes.append((rc, err, [call["op"] for call in calls]))

    assert outcomes[0] == outcomes[1], outcomes
    rc, err, ops = outcomes[0]
    assert rc == 2
    assert ops == []
    assert "FIFO" in err and "delivered no data" in err
    assert "Traceback" not in err


def test_types_declare_refuses_a_fifo_whose_writer_is_attached_but_silent(
        fake_transport, capsys, monkeypatch, tmp_path):
    """The reported hang from the other side of the pipe: `mkfifo f; sleep 60 >
    f &` attaches a writer, so the non-blocking open succeeds and EAGAIN says
    "a writer is there". Waiting for it unconditionally reproduces #864's exact
    measured symptom -- rc=124, zero bytes, no envelope -- so the wait for the
    next byte is bounded and the timeout is a structured refusal."""
    monkeypatch.setattr(bn.cli, "_FIFO_IDLE_TIMEOUT", 0.3)
    fifo = tmp_path / "decl.h"
    os.mkfifo(fifo)
    calls = fake_transport()
    # A write-only open of a FIFO fails with ENXIO while no reader is attached,
    # so the test holds one open for the duration. It changes nothing the
    # reader under test observes: the writer it finds is attached and silent.
    keep_reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    silent_writer = os.open(fifo, os.O_WRONLY)
    try:
        with _must_not_hang():
            rc = bn.cli.main(_declare(str(fifo)))
    finally:
        os.close(silent_writer)
        os.close(keep_reader)

    assert rc == 2
    assert [call["op"] for call in calls] == []
    captured = capsys.readouterr()
    assert "went quiet" in captured.err
    assert str(fifo) in captured.err
    assert "Traceback" not in captured.err


def test_a_slow_but_steady_producer_is_not_cut_off_by_the_idle_bound(
        fake_transport, monkeypatch):
    """The bound is on IDLE time, not on total duration. A generator that takes
    longer overall than the bound but never goes quiet for that long must read
    in full -- a total-duration cap would turn a correct slow producer into a
    new wrong answer."""
    monkeypatch.setattr(bn.cli, "_FIFO_IDLE_TIMEOUT", 0.3)
    calls = fake_transport(_declare_ok())
    read_fd, write_fd = os.pipe()

    def _dribble():
        for piece in (b"struct ", b"R { ", b"int a; ", b"};"):
            os.write(write_fd, piece)
            time.sleep(0.15)
        os.close(write_fd)

    writer = threading.Thread(target=_dribble)
    writer.start()
    try:
        with _must_not_hang():
            rc = bn.cli.main(_declare(f"/dev/fd/{read_fd}"))
    finally:
        writer.join(timeout=10)
        os.close(read_fd)

    assert rc == 0
    assert calls[-1]["params"]["declaration"] == "struct R { int a; };"


_FD_SETSIZE = 1024


@pytest.mark.skipif(
    resource.getrlimit(resource.RLIMIT_NOFILE)[0] < _FD_SETSIZE + 400,
    reason="needs headroom to hold more than FD_SETSIZE descriptors open")
def test_a_fifo_above_fd_setsize_still_answers_with_an_envelope(
        fake_transport, capsys, monkeypatch):
    """`select()`'s fd_set stops at FD_SETSIZE and raises a bare `ValueError`
    past it -- not an `OSError`, so it escapes the reader's own handler. A `bn`
    launched from a supervisor or CI runner that leaks a large descriptor table
    would then get a Python traceback at exit 1 out of the one code path whose
    entire job is to answer in a structured envelope. Ballast pushes the pipe
    past the ceiling so the wait is exercised with a high descriptor."""
    monkeypatch.setattr(bn.cli, "_FIFO_IDLE_TIMEOUT", 0.3)
    calls = fake_transport()
    ballast = [os.open(os.devnull, os.O_RDONLY) for _ in range(_FD_SETSIZE + 100)]
    read_fd, write_fd = os.pipe()
    try:
        assert read_fd > _FD_SETSIZE, f"ballast did not clear FD_SETSIZE ({read_fd})"
        # The write end stays open and silent, so the read reaches the bounded
        # wait -- the only place a descriptor is handed to the readiness call.
        with _must_not_hang():
            rc = bn.cli.main(_declare(f"/dev/fd/{read_fd}"))
    finally:
        os.close(write_fd)
        os.close(read_fd)
        for fd in ballast:
            os.close(fd)

    assert rc == 2
    assert [call["op"] for call in calls] == []
    captured = capsys.readouterr()
    assert "went quiet" in captured.err
    assert "Traceback" not in captured.err


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
