from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import bn.cli
import pytest

from bn_agent_bridge._shared import OperationFailure, _serialize_error, _write_json_artifact
from bn.commands.misc import _resolved_out_format
from bn.output import OutputWriteError, render_value, write_output_result
from _cli_helpers import *  # noqa: F401,F403


def test_evidence_init_routes_and_renders_sections(fake_transport, capsys):
    calls = fake_transport({
        "init_arrays": {
            "ok": True,
            "result": {
                "kind": "init_arrays",
                "pointer_size": 4,
                "items": [
                    {
                        "name": ".init_array",
                        "start": "0x5000",
                        "end": "0x5008",
                        "total_entries": 2,
                        "shown_entries": 2,
                        "truncated": False,
                        "table": {
                            "kind": "pointer_table",
                            "items": [
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
        },
    })

    rc = bn.cli.main(["evidence", "init", "--target", "active", "--limit", "4"])

    assert rc == 0
    assert calls[-1]["op"] == "init_arrays"
    assert calls[-1]["params"] == {"limit": 4}
    output = capsys.readouterr().out
    assert "init arrays: 1 section(s), pointer-size=4" in output
    assert ".init_array 0x5000-0x5008 entries=2" in output
    assert "global_ctor @ 0x401000 (raw 0x401001) [thumb-adjusted]" in output


def test_py_exec_accepts_inline_code(fake_transport):
    calls = fake_transport({"py_exec": {"ok": True, "result": {"stdout": "", "result": None}}})

    rc = bn.cli.main(["py", "exec", "--target", "active", "--code", "print('hi')"])

    assert rc == 0
    assert calls[-1]["op"] == "py_exec"
    assert calls[-1]["target"] == "active"
    assert calls[-1]["params"]["script"] == "print('hi')"
    assert "out_path" not in calls[-1]["params"]


def test_py_exec_dash_reads_stdin(monkeypatch, fake_transport):
    # #312: standardize the stdin idiom with `batch apply -`: a positional "-"
    # reads the script from stdin (alongside the explicit --stdin flag).
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("print('from stdin')"))
    calls = fake_transport({"py_exec": {"ok": True, "result": {"stdout": "", "result": None}}})
    rc = bn.cli.main(["py", "exec", "-", "--target", "active"])
    assert rc == 0
    assert calls[-1]["params"]["script"] == "print('from stdin')"


def test_py_exec_stdin_flag_still_works(monkeypatch, fake_transport):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("print('via flag')"))
    calls = fake_transport({"py_exec": {"ok": True, "result": {"stdout": "", "result": None}}})
    rc = bn.cli.main(["py", "exec", "--stdin", "--target", "active"])
    assert rc == 0
    assert calls[-1]["params"]["script"] == "print('via flag')"


def test_py_exec_missing_script_mentions_code(capsys):
    rc = bn.cli.main(["py", "exec", "--target", "active", "--script", "missing.py"])

    assert rc == 2
    assert "Use --code for inline Python" in capsys.readouterr().err


def test_strings_text_format_renders_rows(fake_transport, capsys):
    calls = fake_transport({
        "strings": {
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
        },
    })

    rc = bn.cli.main(["strings", "--format", "text", "--target", "active", "--query", "follow"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '0x500000  len=6  AsciiString  "follow"' in output
    assert '"value"' not in output


def test_py_exec_text_format_renders_stdout_and_result(fake_transport, capsys):
    fake_transport({
        "py_exec": {
            "ok": True,
            "result": {
                "stdout": "hi\n",
                "result": {"functions": 7},
                "warnings": ["warning one"],
            },
        },
    })

    rc = bn.cli.main(["py", "exec", "--format", "text", "--target", "active", "--code", "print('hi')"])

    assert rc == 0
    output = capsys.readouterr().out
    assert output.startswith("hi\n\nresult:\n")
    assert '"functions": 7' in output
    assert "warnings:" in output


def test_strings_hints_regex_on_zero_matches_with_metachars(fake_transport, capsys):
    """strings with a metacharacter query and 0 matches suggests --regex (#122);
    the empty canonical envelope (total 0) drives the nudge (#275)."""
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "items": [], "total": 0, "offset": 0,
        "limit": None, "returned": 0, "has_more": False}}})
    rc = bn.cli.main(["strings", "--query", "foo(bar", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--regex" in err


def test_bundle_function_out_path_is_bridge_owned(fake_transport, tmp_path, capsys):
    out_path = tmp_path / "bundle.json"

    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
        },
        "bundle_function": {
            "ok": True,
            "result": {
                "ok": True,
                "artifact_path": str(out_path),
                "format": "json",
                "bytes": 123,
                "sha256": "deadbeef",
                "summary": {"kind": "object", "count": 3},
            },
        },
    })

    rc = bn.cli.main(["bundle", "function", "--out", str(out_path), "sub_401000"])

    assert rc == 0
    assert calls[-1]["op"] == "bundle_function"
    assert calls[-1]["params"]["out_path"] == str(out_path)
    assert not out_path.exists()
    output = capsys.readouterr().out
    # bundle function defaults to --format json; the bridge-owned --out envelope
    # printed to stdout must itself be valid JSON, not a text key:value block
    # (issue #10).
    payload = json.loads(output)
    assert payload["artifact_path"] == str(out_path)
    assert payload["spilled"] is False


def test_bundle_function_relative_out_resolves_to_cli_cwd(fake_transport, monkeypatch, tmp_path, capsys):
    # #665: bundle is bridge-owned (the bridge process, not this CLI process,
    # writes the file), so a relative --out must be resolved to an absolute
    # path BEFORE it is threaded into the request params -- otherwise a
    # long-lived bridge spawned from a different directory writes the
    # artifact next to itself instead of next to the invoking shell.
    work = tmp_path / "shell-cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    expected = (work / "bundle.json").resolve()

    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "alpha.bndb"}],
        },
        "bundle_function": {
            "ok": True,
            "result": {
                "ok": True,
                "artifact_path": str(expected),
                "format": "json",
                "bytes": 123,
                "sha256": "deadbeef",
                "summary": {"kind": "object", "count": 3},
            },
        },
    })

    rc = bn.cli.main(["bundle", "function", "--out", "bundle.json", "sub_401000"])

    assert rc == 0
    assert calls[-1]["op"] == "bundle_function"
    # The bug: this used to be the literal relative string "bundle.json".
    assert calls[-1]["params"]["out_path"] == str(expected)
    assert Path(calls[-1]["params"]["out_path"]).is_absolute()


def test_bundle_function_process_local_out_is_written_cli_side(fake_transport, capsys):
    # #665/#708: `bundle function` normally lets the BRIDGE write --out. A
    # process-local fd path must be the exception: `--out /proc/self/fd/<n>`
    # (the bn-kernel CLI backend's artifact contract) and `--out /dev/stdout`
    # resolve in whichever process opens them, so forwarding the literal path
    # makes the bridge write into ITS OWN fd <n> -- the caller then reads zero
    # bytes behind an ok/bytes/sha256 envelope. Keep the write in this process:
    # out_path must be None on the wire and the bundle body must reach the fd.
    bundle = {
        "target": {"selector": "alpha.bndb"},
        "function": {"name": "sub_401000", "address": "0x401000"},
        "decompile": "int64_t sub_401000()\n{\n    return 0;\n}\n",
        "warnings": [],
    }
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "alpha.bndb"}],
        },
        # out_path=None makes the bridge return the bundle itself, not an artifact.
        "bundle_function": {"ok": True, "result": bundle},
    })

    read_fd, write_fd = os.pipe()
    fd_path = f"/proc/self/fd/{write_fd}"
    try:
        rc = bn.cli.main(["bundle", "function", "--out", fd_path, "sub_401000"])
        assert rc == 0
        assert calls[-1]["op"] == "bundle_function"
        # The bug: this used to be the literal "/proc/self/fd/<n>" string, which
        # the BRIDGE would resolve against its own fd table.
        assert calls[-1]["params"]["out_path"] is None
        os.close(write_fd)
        write_fd = -1
        body = os.read(read_fd, 1 << 20)
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)

    # Full body recovered through the caller's own fd, not zero bytes.
    assert json.loads(body.decode()) == bundle
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["artifact_path"] == fd_path
    assert envelope["ok"] is True
    assert envelope["bytes"] == len(body)


@pytest.mark.parametrize("suffix, explicit_format", [
    pytest.param(".ndjson", "json", id="json-format-onto-ndjson-path"),
    pytest.param(".json", "ndjson", id="ndjson-format-onto-json-path"),
])
def test_bundle_explicit_format_beats_the_out_suffix(
        fake_transport, tmp_path, capsys, suffix, explicit_format):
    # #670 (blocker): the bridge-side writer can only key off the --out suffix, so
    # handing it a path whose suffix disagrees with the resolved --format makes it
    # write the OTHER format while the CLI prints "writing <format>". Both
    # directions must keep the note and the artifact in agreement: delegate only
    # when the suffix-derived bridge format IS the resolved one, else write here.
    bundle = {
        "target": {"selector": "alpha.bndb"},
        "function": {"name": "sub_401000", "address": "0x401000"},
        "decompile": "int64_t sub_401000()\n{\n    return 0;\n}\n",
        "warnings": [],
    }
    out_path = tmp_path / f"bundle{suffix}"
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "alpha.bndb"}],
        },
        # out_path=None means the bridge returns the bundle itself: the CLI writes.
        "bundle_function": {"ok": True, "result": bundle},
    })

    rc = bn.cli.main(["bundle", "function", "--format", explicit_format,
                      "--out", str(out_path), "sub_401000"])

    assert rc == 0
    assert calls[-1]["op"] == "bundle_function"
    # (a) The CLI must own the write: the bridge would have produced the other
    # format for this suffix.
    assert calls[-1]["params"]["out_path"] is None
    # (b) The artifact is exactly what the CLI resolved the format to be.
    assert out_path.read_text(encoding="utf-8") == render_value(bundle, explicit_format)
    captured = capsys.readouterr()
    # (c) The envelope's stated format matches the bytes on disk.
    assert json.loads(captured.out)["format"] == explicit_format
    # (d) ... and matches what the CLI told the user it was writing (#670).
    assert f"writing {explicit_format}" in captured.err


@pytest.mark.parametrize("suffix, inferred_note", [
    pytest.param(".ndjson", True, id="ndjson-path-inferred"),
    pytest.param(".json", False, id="json-path-is-default"),
])
def test_bundle_default_out_suffix_still_lets_the_bridge_write(
        fake_transport, tmp_path, capsys, suffix, inferred_note):
    # Regression guard for the #670 fix itself: with NO explicit --format the
    # suffix IS the resolved format, so the bridge keeps owning the write (the
    # original #670 path must survive end-to-end).
    out_path = tmp_path / f"bundle{suffix}"
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "alpha.bndb"}],
        },
        "bundle_function": {
            "ok": True,
            "result": {
                "ok": True,
                "artifact_path": str(out_path),
                "format": suffix.lstrip("."),
                "bytes": 123,
                "sha256": "deadbeef",
                "summary": {"kind": "object", "count": 3},
            },
        },
    })

    rc = bn.cli.main(["bundle", "function", "--out", str(out_path), "sub_401000"])

    assert rc == 0
    assert calls[-1]["params"]["out_path"] == str(out_path)
    err = capsys.readouterr().err
    assert ("inferring --format ndjson" in err) is inferred_note


def test_write_json_artifact_ndjson_out_is_line_delimited(tmp_path):
    # #670: `bundle function --out foo.ndjson` prints "note: inferring --format
    # ndjson from the .ndjson --out path" while the bridge-owned writer hard-coded
    # pretty single-document JSON behind an envelope claiming format json. The
    # suffix is the only format signal the bridge side has, so the writer must
    # honor it byte-for-byte as the CLI's own render_value("ndjson") does.
    payload = {"alpha": 1, "nested": {"beta": [1, 2]}, "name": "sample"}
    out_path = tmp_path / "bundle.ndjson"

    envelope = _write_json_artifact(str(out_path), payload)

    file_bytes = out_path.read_bytes()
    text = file_bytes.decode("utf-8")
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload
    assert text == render_value(payload, "ndjson")
    assert envelope["format"] == "ndjson"
    assert envelope["bytes"] == len(file_bytes)
    assert envelope["sha256"] == hashlib.sha256(file_bytes).hexdigest()


def test_write_json_artifact_json_out_stays_pretty_single_document(tmp_path):
    # Regression guard for the unchanged .json path (this passes at base by
    # design; it pins that the #670 fix did not churn it).
    payload = {"alpha": 1, "nested": {"beta": [1, 2]}}
    out_path = tmp_path / "bundle.json"

    envelope = _write_json_artifact(str(out_path), payload)

    text = out_path.read_text(encoding="utf-8")
    assert json.loads(text) == payload
    assert text == json.dumps(payload, indent=2, sort_keys=True)
    assert envelope["format"] == "json"
    assert envelope["bytes"] == len(text.encode("utf-8"))


@pytest.mark.parametrize("payload", [
    pytest.param({"alpha": 1, "nested": {"beta": [1, 2]}, "name": "sample"},
                 id="plain-dict"),
    pytest.param({"items": [{"name": "a", "address": "0x1"},
                            {"name": "b", "address": "0x2"}],
                  "total": 2, "offset": 0, "returned": 2, "has_more": False},
                 id="paged-dict"),
    pytest.param([{"name": "a"}, {"name": "b"}, {"name": "c"}],
                 id="top-level-list"),
])
def test_write_json_artifact_ndjson_matches_render_value_for_paged_and_plain_payloads(
        tmp_path, payload):
    # #670 (major b): the bridge writer emitted exactly one line for ANY dict,
    # while render_value fans a paged dict (a list under `items`/`functions`) into
    # one record per item plus a trailing `_meta` record. Interchangeable writers
    # mean byte-identity for both shapes, not only today's non-paged bundle.
    out_path = tmp_path / "bundle.ndjson"

    _write_json_artifact(str(out_path), payload)

    text = out_path.read_text(encoding="utf-8")
    assert text == render_value(payload, "ndjson")
    lines = [line for line in text.splitlines() if line]
    assert lines  # a non-empty payload is never an empty stream
    for line in lines:
        json.loads(line)  # every record stands alone as NDJSON
    if isinstance(payload, list):
        assert len(lines) == len(payload)      # one record per element, not one line
    if isinstance(payload, dict) and "items" in payload:
        assert len(lines) == len(payload["items"]) + 1
        meta = json.loads(lines[-1])
        assert meta["_meta"] is True
        assert meta["total"] == payload["total"]


def test_write_json_artifact_ndjson_writes_an_empty_payload_as_an_empty_stream(tmp_path):
    # #670: every line of an NDJSON artifact must be a JSON document, so a
    # payload with no records is ZERO bytes -- not a file holding a bare
    # newline, which is a blank line no reader can parse. `render_value` answers
    # "" for an empty list and the two --out writers must stay interchangeable
    # at that boundary too, which is the one place the trailing-newline term can
    # be observed at all.
    out_path = tmp_path / "bundle.ndjson"

    envelope = _write_json_artifact(str(out_path), [])

    assert out_path.read_bytes() == b""
    assert out_path.read_text(encoding="utf-8") == render_value([], "ndjson")
    assert envelope["bytes"] == 0


def test_bundle_out_error_matches_the_cli_side_writer(tmp_path):
    # #719: the CLI-side and bridge-side --out writers must be indistinguishable
    # to a caller. The bridge one used to leak the raw exception class name
    # behind an `internal error:` prefix for the same user mistake.
    payload = {"alpha": 1}
    adir = tmp_path / "adir"
    adir.mkdir()

    with pytest.raises(OutputWriteError) as cli_exc:
        write_output_result(payload, fmt="json", out_path=adir, stem="function-bundle")
    with pytest.raises(OperationFailure) as bridge_exc:
        _write_json_artifact(str(adir), payload)

    assert str(bridge_exc.value) == str(cli_exc.value)
    assert "internal error:" not in str(bridge_exc.value)
    assert bridge_exc.value.status == "output_write_failed"


def test_bundle_out_unwritable_path_uses_the_same_shape(tmp_path):
    # The parent path is a regular FILE, so mkdir fails with an OSError on any
    # uid (no chmod-based unwritability, which is a no-op when tests run as root).
    afile = tmp_path / "afile"
    afile.write_text("not a directory", encoding="utf-8")
    out_path = afile / "x.json"

    with pytest.raises(OperationFailure) as exc:
        _write_json_artifact(str(out_path), {"alpha": 1})

    message = str(exc.value)
    assert message.startswith(f"Failed to write --out file {out_path}: ")
    assert "internal error:" not in message
    assert exc.value.status == "output_write_failed"


def test_serialize_error_distinguishes_user_and_internal_failures(tmp_path):
    # Acceptance criterion 4 of #719: a genuine bridge-internal failure must keep
    # reporting as an internal error, not be laundered into a user-facing one.
    user = OperationFailure("output_write_failed", "Failed to write --out file /x: nope")
    assert _serialize_error(user) == "Failed to write --out file /x: nope"
    assert _serialize_error(TypeError("boom")) == "internal error: TypeError: boom"
    # ...and that holds for a failure raised out of the CHANGED writer itself:
    # the OSError guard must not swallow a genuinely unserializable payload into
    # a user-facing OperationFailure. A non-str mapping KEY is the case no
    # `default=` hook can rescue -- json.dumps never consults it for keys -- so
    # it stays internal on both writers.
    with pytest.raises(TypeError) as exc:
        _write_json_artifact(str(tmp_path / "bundle.json"), {(1, 2): "tuple key"})
    assert not isinstance(exc.value, OperationFailure)
    assert _serialize_error(exc.value).startswith("internal error: TypeError")
    with pytest.raises(TypeError):
        render_value({(1, 2): "tuple key"}, "json")


@pytest.mark.parametrize("suffix", [".ndjson", ".json"])
def test_write_json_artifact_renders_non_json_native_values_like_the_cli(
        tmp_path, suffix):
    # #670 (major): render_value passes default=_json_default, so the CLI-side
    # --out writer renders a Path (or any exotic value) instead of dying. The
    # bridge-side writer omitted the hook, so the SAME payload through the SAME
    # flag came back as `internal error: TypeError` from one writer and an
    # artifact from the other.
    payload = {"path": Path("/x/y"), "items": [Path("/a"), 1], "total": 2}
    out_path = tmp_path / f"bundle{suffix}"

    envelope = _write_json_artifact(str(out_path), payload)

    text = out_path.read_text(encoding="utf-8")
    if suffix == ".ndjson":
        assert text == render_value(payload, "ndjson")
    else:
        assert json.loads(text) == {"path": "/x/y", "items": ["/a", 1], "total": 2}
    assert envelope["bytes"] == len(text.encode("utf-8"))


def test_write_json_artifact_ndjson_keeps_a_payloads_own_meta_key(tmp_path):
    # #670 (major): the trailing paging record is marked with a synthetic
    # `_meta` key. Copying the payload's other keys into it silently DESTROYS a
    # real `_meta` value. The bridge-owned artifact is the caller's data, so the
    # fan-out is skipped for the one payload shape it cannot represent, rather
    # than writing a stream that has quietly lost a field.
    payload = {
        "items": [{"name": "a"}, {"name": "b"}],
        "total": 2,
        "_meta": {"origin": "an upstream ndjson stream"},
    }
    out_path = tmp_path / "bundle.ndjson"

    _write_json_artifact(str(out_path), payload)

    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload      # nothing dropped, nothing rewritten


@pytest.mark.parametrize("explicit_format", [None, "json", "ndjson", "text"])
@pytest.mark.parametrize("out", ["x.json", "x.ndjson", "x.txt", "x", None])
def test_bundle_delegation_format_tracks_the_cli_resolver(
        capsys, explicit_format, out):
    # #670 (major): the delegation predicate re-derives cli._resolve_output_format's
    # precedence because the real resolver PRINTS the note/warning and `_call`
    # invokes it again. Two copies of one precedence rule drift silently, so the
    # equivalence is pinned over every --format x --out-suffix combination.
    args = types.SimpleNamespace(out=out, format=explicit_format or "text")
    if explicit_format is not None:
        args._format_explicit = True

    quiet = _resolved_out_format(args)
    loud = bn.cli._resolve_output_format(args)
    capsys.readouterr()      # the real resolver's note/warning is a side effect

    assert quiet == loud


def test_bundle_does_not_delegate_a_suffix_the_bridge_writes_as_json(
        fake_transport, tmp_path, capsys, monkeypatch):
    # #670 (major): the bridge writer keys off the literal `.ndjson` suffix, the
    # CLI off a suffix MAP. Deriving the bridge's format from the map reinstates
    # the note/bytes contradiction the moment a second ndjson suffix is mapped:
    # the CLI announces ndjson, delegates, and the bridge writes pretty JSON.
    monkeypatch.setitem(bn.cli._OUT_FORMAT_BY_SUFFIX, ".jsonl", "ndjson")
    bundle = {"function": {"name": "sub_401000"}, "warnings": []}
    out_path = tmp_path / "bundle.jsonl"
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "alpha.bndb"}],
        },
        "bundle_function": {"ok": True, "result": bundle},
    })

    rc = bn.cli.main(["bundle", "function", "--out", str(out_path), "sub_401000"])

    assert rc == 0
    assert calls[-1]["params"]["out_path"] is None     # the CLI keeps the write
    captured = capsys.readouterr()
    assert "inferring --format ndjson" in captured.err
    assert json.loads(captured.out)["format"] == "ndjson"
    assert out_path.read_text(encoding="utf-8") == render_value(bundle, "ndjson")


def test_strings_json_carries_paging_envelope(fake_transport, capsys):
    # #122: strings now returns the {items, total, ...} envelope, so machine
    # consumers see the true total + remainder, not a bare truncated list. The
    # CLI forwards the REAL --limit (no client-side limit+1 probe).
    calls = fake_transport({"strings": {"ok": True, "result": {
        "items": [{"address": "0x1000", "length": 5, "chars": 5,
                   "type": "ascii", "value": "alpha"}],
        "total": 4096, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
    }}})
    rc = bn.cli.main(["strings", "--target", "active", "--query", "alpha",
                      "--limit", "1", "--format", "json"])
    assert rc == 0
    assert calls[-1]["params"]["limit"] == 1   # real limit, not limit+1
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 4096
    assert payload["has_more"] is True and payload["returned"] == 1
    assert payload["items"][0]["value"] == "alpha"


def test_strings_text_footer_states_true_total(fake_transport, capsys):
    # Text mode renders the rows AND a "showing N of TOTAL (R more)" footer that
    # mirrors function list, so a truncated dump still admits the remainder (#122).
    fake_transport({"strings": {"ok": True, "result": {
        "items": [{"address": hex(0x1000 + i), "length": 5, "chars": 5,
                   "type": "ascii", "value": f"str{i}"} for i in range(3)],
        "total": 50, "offset": 0, "limit": 3, "returned": 3, "has_more": True,
    }}})
    rc = bn.cli.main(["strings", "--target", "active", "--query", "str",
                      "--limit", "3", "--format", "text"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert '"str0"' in stdout                                # rows are rendered
    assert "// showing 3 of 50 (47 more)" in stdout          # honest total + remainder
    assert "--offset 3" in stdout


def test_imports_json_carries_paging_envelope(fake_transport, capsys):
    # The non-summary imports list also returns the envelope, and the CLI
    # forwards the REAL --limit (no client-side limit+1 probe) (#122).
    calls = fake_transport({"imports": {"ok": True, "result": {
        "items": [{"name": "printf", "address": "0x1000", "library": "libc",
                   "raw_name": "printf", "kind": "function"}],
        "total": 512, "offset": 0, "limit": 1, "returned": 1, "has_more": True,
    }}})
    rc = bn.cli.main(["imports", "--target", "active", "--limit", "1", "--format", "json"])
    assert rc == 0
    assert calls[-1]["params"]["limit"] == 1   # real limit, not limit+1
    assert calls[-1]["params"].get("summary") is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 512 and payload["has_more"] is True
    assert payload["items"][0]["name"] == "printf"


def test_exports_list_alias_routes_to_export_enumerator(fake_transport, capsys):
    calls = fake_transport(
        {
            "list_exports": {
                "ok": True,
                "result": {
                    "items": [],
                    "offset": 0,
                    "returned": 0,
                    "total": 0,
                    "has_more": False,
                },
            }
        }
    )

    rc = bn.cli.main(
        ["exports", "list", "--target", "active", "--format", "json"]
    )

    assert rc == 0
    assert calls[-1]["op"] == "list_exports"
@pytest.mark.parametrize("manifest, expected", [
    pytest.param(None, "Manifest file not found", id="missing-file"),
    pytest.param("{not valid json", "Invalid JSON in manifest", id="invalid-json"),
    pytest.param('[{"op": "set_comment", "address": "0x1000", "comment": "x"}]', "must be a JSON object", id="bare-array"),
])
def test_batch_apply_file_clean_error(fake_transport, capsys, tmp_path, manifest, expected):
    # Bad manifest files must surface a clean BridgeError (exit 2), never a
    # client-side traceback (e.g. a bare array hitting _call's dict(params), #48).
    calls = fake_transport()
    if manifest is None:
        path = tmp_path / "no" / "such" / "manifest.json"
    else:
        path = tmp_path / "manifest.json"
        path.write_text(manifest, encoding="utf-8")

    rc = bn.cli.main(["batch", "apply", str(path)])

    assert rc == 2  # BridgeError exit code
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err
    assert calls == []


@pytest.mark.parametrize("mode", ["directory", "undecodable"])
@pytest.mark.parametrize("command", ["batch-apply", "py-exec"])
def test_file_argument_failures_never_traceback_754(
    fake_transport, capsys, tmp_path, command, mode
):
    """#754 consistency: `types declare --file` was not the only unguarded
    `exists()`-then-`read_text` in this family. `py exec --script` tracebacked at
    exit 1 on BOTH a directory and a non-UTF-8 file; `batch apply`'s manifest read
    caught OSError only, so a non-UTF-8 manifest tracebacked while its directory
    case was already wrapped (that row is the negative control).

    The property the issue names is that every --file failure mode returns a JSON
    ENVELOPE, so that is what this asserts. Checking only rc/stderr does not
    discriminate it: a refusal rewritten as `print(msg, file=sys.stderr); return 2`
    satisfies rc == 2, leaves no traceback and names the path, while stdout is
    empty and a JSON consumer gets nothing (review of #855).

    `batch apply` is `fmt="json"`, so its envelope is the default; `py exec`
    renders text by default and is asked for JSON explicitly."""
    if mode == "directory":
        path = tmp_path
    else:
        path = tmp_path / "payload"
        path.write_bytes(b"{}\xff\xfe")
    calls = fake_transport()

    argv = (
        ["batch", "apply", str(path)] if command == "batch-apply"
        else ["py", "exec", "--target", "active", "--script", str(path),
              "--format", "json"]
    )
    rc = bn.cli.main(argv)

    assert rc == 2
    assert calls == []
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    # The envelope, parsed -- not a substring match, so a truncated or
    # double-encoded payload cannot pass.
    envelope = json.loads(captured.out)
    assert envelope["ok"] is False
    assert str(path) in envelope["error"]


@pytest.mark.parametrize("stdin, expected", [
    pytest.param("   \n", "No manifest on stdin", id="empty"),
    pytest.param("{not valid json", "Invalid JSON in manifest (<stdin>)", id="invalid-json"),
    pytest.param('[{"op": "set_comment", "address": "0x1000", "comment": "x"}]', "must be a JSON object", id="bare-array"),
])
def test_batch_apply_stdin_clean_error(monkeypatch, fake_transport, capsys, stdin, expected):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    calls = fake_transport()

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 2
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err
    assert calls == []


@pytest.mark.parametrize("manifest", [{}, {"ops": None}, {"ops": {}}])
@pytest.mark.parametrize("from_stdin", [False, True])
def test_batch_operation_shape_refusal_is_presend_invalid_request(
        monkeypatch, fake_transport, capsys, tmp_path, manifest, from_stdin):
    import io

    raw = json.dumps(manifest)
    if from_stdin:
        monkeypatch.setattr("sys.stdin", io.StringIO(raw))
        source = "-"
    else:
        path = tmp_path / "manifest.json"
        path.write_text(raw, encoding="utf-8")
        source = str(path)
    calls = fake_transport()
    assert bn.cli.main(["batch", "apply", source, "--format", "json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["status"] == "invalid_request"
    assert "ops" in payload["error"]
    assert payload["observed"]["request_sent"] is False
    assert calls == []




def test_batch_apply_reads_manifest_from_stdin(monkeypatch, fake_transport, capsys):
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

    calls = fake_transport({"batch_apply": {"ok": True, "result": {"preview": False, "success": True, "results": [{"status": "verified"}]}}})

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 0
    assert calls[-1]["op"] == "batch_apply"
    # The free-text comment reached the bridge byte-for-byte.
    assert calls[-1]["params"]["ops"][0]["comment"] == comment


def test_batch_apply_full_result_carries_top_level_ok(monkeypatch, fake_transport, capsys):
    # #447: mutation/batch JSON used only success/committed, so `jq '.ok'` read
    # null. Add a top-level `ok` mirroring the read-command envelope.
    import io, json as _json
    monkeypatch.setattr("sys.stdin", io.StringIO(
        '{"target": "active", "ops": [{"op": "rename_function", "address": "0x1000", "name": "f"}]}'))
    fake_transport({"batch_apply": {"ok": True, "result": {
        "preview": False, "success": True, "committed": True,
        "results": [{"status": "verified"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "--format", "json"])
    assert rc == 0
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True
    assert parsed["success"] is True  # unchanged, additive


def test_batch_apply_ok_false_on_failed_op(monkeypatch, fake_transport, capsys):
    import io, json as _json
    monkeypatch.setattr("sys.stdin", io.StringIO(
        '{"target": "active", "ops": [{"op": "rename_function", "address": "0x1000", "name": "f"}]}'))
    fake_transport({"batch_apply": {"ok": True, "result": {
        "preview": False, "success": False, "committed": False,
        "results": [{"status": "verification_failed"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "--format", "json"])
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed["ok"] is False


def test_batch_apply_summary_carries_ok(monkeypatch, fake_transport, capsys):
    import io, json as _json
    monkeypatch.setattr("sys.stdin", io.StringIO(
        '{"target": "active", "ops": [{"op": "rename_function", "address": "0x1000", "name": "f"}]}'))
    fake_transport({"batch_apply": {"ok": True, "result": {
        "preview": False, "success": True, "committed": True,
        "results": [{"status": "verified"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "--summary", "--format", "json"])
    assert rc == 0
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed["kind"] == "mutation_summary"
    assert parsed["ok"] is True


def test_batch_apply_accepts_target_flag(monkeypatch, fake_transport):
    # #308: batch apply now accepts -t like every other mutate command; the flag
    # supplies the manifest target when the manifest itself omits one.
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"ops": [{"op": "set_comment", "address": "0x1", "comment": "c"}]}'))
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": [{"status": "verified"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "-t", "foo.bndb", "-i", "inst"])
    assert rc == 0
    assert calls[-1]["params"].get("target") == "foo.bndb"


def test_batch_apply_cli_target_overrides_manifest(monkeypatch, fake_transport):
    # CLI -t is the explicit per-invocation selector and WINS over a manifest
    # "target" (#366): a fan-out agent that copies the documented {"target":"active"}
    # example but passes a correct -t must not be sabotaged by the in-payload value.
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"target": "active", "ops": []}'))
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": [{"status": "verified"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "-t", "fromflag", "-i", "inst"])
    assert rc == 0
    assert calls[-1]["params"].get("target") == "fromflag"


def test_batch_apply_manifest_target_used_when_no_flag(monkeypatch, fake_transport):
    # Without -t, the manifest "target" is still honored.
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"target": "explicit", "ops": []}'))
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": [{"status": "verified"}]}}})
    rc = bn.cli.main(["batch", "apply", "-", "-i", "inst"])
    assert rc == 0
    assert calls[-1]["params"].get("target") == "explicit"
@pytest.mark.parametrize("argv, expected", [
    pytest.param(["--min-length", "5"], {"min_length": 5}, id="min-length"),
    pytest.param(["--max-length", "80"], {"max_length": 80}, id="max-length"),
    pytest.param(["--section", ".rodata", "--no-crt"], {"section": ".rodata", "no_crt": True}, id="section-no-crt"),
    pytest.param(["--query", "foo|bar", "--regex"], {"query": "foo|bar", "regex": True}, id="regex"),
    pytest.param(["--probable-format-strings"], {"probable_format_strings": True}, id="probable-format"),
])
def test_strings_passes_args_to_bridge(fake_transport, argv, expected):
    # The CLI's job is argv -> bridge request; assert the params it forwarded.
    calls = fake_transport({"strings": {"ok": True, "result": []}})
    rc = bn.cli.main(["strings", "--target", "active", *argv])
    assert rc == 0
    params = calls[-1]["params"]
    for key, val in expected.items():
        assert params[key] == val


def test_strings_probable_format_text_shows_directives_and_refs(fake_transport, capsys):
    # --probable-format-strings enrichment renders in text mode: the recovered
    # directives and the code-xref count appear on the row.
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings",
        "items": [{
            "address": "0x1000", "length": 7, "chars": 7, "type": "ascii",
            "value": "%s: %d\n", "format_directives": ["%s", "%d"],
            "directive_count": 2, "code_refs": 3,
        }],
        "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False,
    }}})
    rc = bn.cli.main(["strings", "--target", "active", "--probable-format-strings",
                      "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fmt: %s %d" in out
    assert "code_refs=3" in out




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


def test_strings_query_value_rejects_known_sibling_flag(monkeypatch, capsys):
    """#694 item 14: a KNOWN sibling flag right after --query is a usage error,
    not a silent literal search for the flag text -- a `--query --regex` typo
    used to return a confident, wrong "no matches for '--regex'" instead of
    telling the caller they forgot to reorder their flags."""
    def fake_send_request(op, **kwargs):
        raise AssertionError(f"send_request must not be called: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["strings", "--target", "active", "--query", "--regex"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "argument --query: expected a value but found the known flag '--regex'" in err
    assert "--query=--regex" in err


def test_strings_query_explicit_equals_still_searches_literal_flag_text(monkeypatch, capsys):
    """The explicit `--query=<value>` spelling is the documented escape hatch:
    it must still search for the literal flag text even when that text
    collides with a known sibling option (unaffected by the new sibling-flag
    rejection, which only applies to the SPACE-separated form)."""
    captured_queries = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None,
                          spawn_missing_named=False):
        if op == "strings":
            captured_queries.append(params["query"])
            return {"ok": True, "result": []}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["strings", "--target", "active", "--query=--format"])

    assert rc == 0
    assert captured_queries == ["--format"]


def test_py_exec_code_value_rejects_known_sibling_flag(monkeypatch, capsys):
    """--code (`bn py exec`) is protected the same way --query is (#694 item 14):
    a known sibling flag right after it is a usage error, not a literal-value
    guess."""
    def fake_send_request(op, **kwargs):
        raise AssertionError(f"send_request must not be called: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["py", "exec", "--target", "active", "--code", "--format"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "argument --code: expected a value but found the known flag '--format'" in err


# --- I5: sections CLI ---


def test_sections_text_format_renders_rows(fake_transport, capsys):
    fake_transport({
        "sections": {
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
        },
    })

    rc = bn.cli.main(["sections", "--format", "text", "--target", "active"])

    assert rc == 0
    output = capsys.readouterr().out
    assert ".text" in output
    assert "0x1000" in output
    assert "r-x" in output


def test_sections_passes_query_to_bridge(fake_transport, capsys):
    calls = fake_transport({"sections": {"ok": True, "result": []}})

    rc = bn.cli.main(["sections", "--target", "active", "--query", "data"])

    assert rc == 0
    assert calls[-1]["params"]["query"] == "data"


# --- I8: enhanced imports CLI ---


def test_imports_text_shows_kind_for_non_function(fake_transport, capsys):
    fake_transport({
        "imports": {
            "ok": True,
            "result": {
                "items": [
                    {"name": "printf", "address": "0x1000", "library": "libc", "raw_name": "printf", "kind": "function"},
                    {"name": "__stdout", "address": "0x2000", "library": "libc", "raw_name": "__stdout", "kind": "data"},
                ],
                "total": 2, "offset": 0, "limit": 100, "returned": 2, "has_more": False,
            },
        },
    })

    rc = bn.cli.main(["imports", "--format", "text", "--target", "active"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "printf" in output
    assert "(data)" in output
    assert "(function)" not in output  # function kind is not shown


# --- read: raw bytes at an address ---


def test_read_text_renders_hexdump(fake_transport, capsys):
    calls = fake_transport({
        "read": {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 8,
                "hex": "48656c6c6f0090ff",
                "ascii": "Hello...",
            },
        },
    })

    rc = bn.cli.main(["read", "--target", "active", "--address", "0x1000", "--length", "8"])

    assert rc == 0
    assert calls[-1]["params"] == {"address": "0x1000", "length": 8}
    output = capsys.readouterr().out
    assert "00001000: 48 65 6c 6c 6f 00 90 ff" in output
    assert "Hello..." in output


def test_read_json_returns_structured_payload(fake_transport, capsys):
    fake_transport({
        "read": {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 4,
                "hex": "41424344",
                "ascii": "ABCD",
            },
        },
    })

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


def test_read_short_read_text_includes_note(fake_transport, capsys):
    fake_transport({
        "read": {
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
        },
    })

    rc = bn.cli.main(["read", "--target", "active", "--address", "0x1000", "--length", "16"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "00001000: 01 02 03 04" in output
    assert "note: short read: requested 16 bytes, only 4 mapped from 0x1000" in output


def test_read_bytes_encoding_writes_raw_bytes(fake_transport, capsys):
    fake_transport({
        "read": {
            "ok": True,
            "result": {
                "address": "0x1000",
                "length": 4,
                "hex": "41424344",
                "ascii": "ABCD",
            },
        },
    })

    rc = bn.cli.main(
        ["read", "--target", "active", "--address", "0x1000", "--length", "4", "--encoding", "bytes"]
    )

    assert rc == 0
    assert capsys.readouterr().out == "ABCD"


def test_read_accepts_positional_address(fake_transport):
    calls = fake_transport({
        "read": {"ok": True, "result": {"address": "0x1000", "length": 8, "hex": "00" * 8, "ascii": "." * 8}},
    })

    # Positional address matches the convention used by decompile/disasm/il/xrefs.
    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--length", "8"])

    assert rc == 0
    assert calls[-1]["params"] == {"address": "0x1000", "length": 8}


def test_read_accepts_size_alias_for_length(fake_transport):
    # #410: --size is an alias for --length.
    calls = fake_transport({
        "read": {"ok": True, "result": {"address": "0x1000", "length": 32, "hex": "", "ascii": ""}},
    })
    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--size", "32"])
    assert rc == 0
    assert calls[-1]["params"]["length"] == 32


def test_read_length_accepts_hex(fake_transport):
    calls = fake_transport({
        "read": {"ok": True, "result": {"address": "0x1000", "length": 194, "hex": "", "ascii": ""}},
    })

    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--length", "0xc2"])

    assert rc == 0
    assert calls[-1]["params"]["length"] == 194


def test_read_defaults_length_when_omitted(fake_transport):
    # #312: a bare `bn read <addr>` reads a small default window instead of
    # erroring "the following arguments are required: --length".
    calls = fake_transport({
        "read": {"ok": True, "result": {"address": "0x1000", "length": 16, "hex": "00" * 16, "ascii": "." * 16}},
    })
    rc = bn.cli.main(["read", "--target", "active", "0x1000"])
    assert rc == 0
    assert calls[-1]["params"] == {"address": "0x1000", "length": 16}


def test_read_conflicting_address_errors(fake_transport, capsys):
    fake_transport()

    rc = bn.cli.main(["read", "--target", "active", "0x1000", "--address", "0x2000", "--length", "8"])

    assert rc == 2
    assert "given twice with different values" in capsys.readouterr().err


def test_read_missing_address_errors(monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "send_request", lambda *a, **k: None)

    rc = bn.cli.main(["read", "--target", "active", "--length", "8"])

    assert rc == 2
    assert "read address is required" in capsys.readouterr().err


# --- imports --summary CLI routing/rendering ---


def test_imports_summary_routes_and_renders_text(fake_transport, capsys):
    calls = fake_transport({
        "imports": {
            "ok": True,
            "result": {
                "total_symbols": 4,
                "namespaces": {"libc": 3, "libfoo": 1},
                "by_kind": {"function": 3, "data": 1},
            },
        },
    })

    rc = bn.cli.main(["imports", "--summary", "--format", "text", "--target", "active"])

    assert rc == 0
    assert calls[-1]["op"] == "imports"
    assert calls[-1]["params"]["summary"] is True
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


def test_imports_summary_and_count_text_surface_self_defined_excluded():
    """The PIC self-export exclusion count (#202) must show in the summary AND
    count text renderers, not just the default list footer -- the reviewer noted
    those two paths silently omitted it (#209 follow-up)."""
    from bn import formatters
    from bn.commands.misc import _imports_count_text
    summary = formatters._render_imports_summary_text(
        {"total_symbols": 3, "needed_libraries": [], "namespaces": {}, "by_kind": {},
         "self_defined_excluded": 9}
    )
    assert "self-defined excluded: 9" in summary
    # absent when zero / missing
    assert "self-defined" not in formatters._render_imports_summary_text(
        {"total_symbols": 3, "needed_libraries": [], "namespaces": {}, "by_kind": {}}
    )
    assert _imports_count_text({"count": 3, "self_defined_excluded": 9}) == (
        "Total imports: 3 (9 self-defined excluded)"
    )
    assert _imports_count_text({"count": 3}) == "Total imports: 3"


def test_imports_without_summary_routes_false(fake_transport, capsys):
    calls = fake_transport({"imports": {"ok": True, "result": []}})

    rc = bn.cli.main(["imports", "--target", "active"])

    assert rc == 0
    assert calls[-1]["params"]["summary"] is False


def test_read_bytes_malformed_response_clean_error(fake_transport, capsys):
    fake_transport({"read": {"ok": True, "result": {"length": 4}}})  # no "hex" payload

    rc = bn.cli.main(
        ["read", "0x1000", "--length", "4", "--encoding", "bytes", "--target", "active"]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "malformed read response" in err
    assert "Traceback" not in err


def test_strings_unfiltered_emits_section_hint(fake_transport, capsys):
    fake_transport({"strings": {"ok": True, "result": []}})
    assert bn.cli.main(["strings", "--target", "active"]) == 0

    _, stderr = capsys.readouterr()
    assert "--section .rodata" in stderr


def test_strings_with_filter_suppresses_section_hint(fake_transport, capsys):
    fake_transport({"strings": {"ok": True, "result": []}})
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


def test_read_bytes_out_writes_envelope_and_creates_parents(fake_transport, capsys, tmp_path):
    # #96: the bytes --out path must mkdir parents and emit an artifact envelope.
    fake_transport({"read": {"ok": True, "result": {"hex": "deadbeef"}}})
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


def test_read_bytes_out_bad_dir_is_clean_error(fake_transport, capsys, tmp_path):
    # A write failure must be a clean BridgeError, not a raw traceback.
    fake_transport({"read": {"ok": True, "result": {"hex": "00"}}})
    # A path whose parent is an existing FILE can't be mkdir'd.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    out = blocker / "sub" / "out.bin"
    rc = bn.cli.main(["read", "0x1000", "--length", "1", "--encoding", "bytes",
                      "--target", "active", "--out", str(out)])
    assert rc == 2  # OutputWriteError is a BridgeError -> exit 2
    assert "Failed to write --out file" in capsys.readouterr().err


def test_py_exec_accepts_positional_code(monkeypatch):
    """`bn py exec '<code>'` positional works, matching the skill examples (#197)."""
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"stdout": "", "result": None}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["py", "exec", "--target", "active", "print('hi')"])
    assert rc == 0
    assert captured["params"]["script"] == "print('hi')"


def test_batch_apply_drops_instance_id_target(monkeypatch):
    """A fan-out agent putting the --instance id in the manifest target has it
    dropped, so the instance's single open target resolves (#227)."""
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        captured["instance_id"] = instance_id
        return {"ok": True, "result": {"results": [{"status": "verified"}], "status": "verified"}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    import io
    manifest = '{"target": "my_inst", "ops": []}'
    monkeypatch.setattr("sys.stdin", io.StringIO(manifest))
    rc = bn.cli.main(["--instance", "my_inst", "batch", "apply", "-"])
    assert rc == 0
    assert "target" not in captured["params"]      # instance-id target was dropped




def test_bare_py_arg_hints_at_exec_subcommand(capsys):
    # `bn py '<code>'` is the natural (wrong) shape; argparse rejects the code as
    # an invalid subcommand choice. A single-subcommand group must point at the
    # real form (`bn py exec ...`) instead of echoing the code as a bad choice.
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["py", "print(1+1)", "--target", "active"])
    assert exc.value.code == 2
    _, err = capsys.readouterr()
    assert "bn py exec" in err  # actionable hint, not just "invalid choice"


def test_valid_py_exec_still_parses(capsys):
    # The hint must not disturb the correct form: `bn py exec '<code>'` still
    # reaches the handler (fails only at transport, exit 2, not an arg error).
    parser = bn.cli.build_parser()
    ns = parser.parse_args(["py", "exec", "1+1", "--target", "active"])
    assert getattr(ns, "code_pos", None) == "1+1"


# --- data vars / data symbols -------------------------------------------------


def test_data_vars_builds_request_and_renders_rows(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "demo_app.bndb"}]},
        "data_vars": {
            "ok": True,
            "result": {"kind": "data_vars", "has_more": False, "items": [
                {"a": "0x2000", "n": "", "t": "int32_t", "w": 4, "v": 42, "sec": ".data"},
                {"a": "0x2004", "n": "g_handler", "t": "char*", "w": 4,
                 "p": "0x5000", "ps": "on_message", "sec": ".data"},
                {"a": "0x2008", "n": "", "t": "char*", "w": 4,
                 "p": "0x6000", "pstr": "hello", "sec": ".data"},
            ]},
        },
    })

    rc = bn.cli.main(["data", "vars", "--start", "0x2000", "--end", "0x3000"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "data_vars"]
    assert calls[1]["params"] == {"start": "0x2000", "end": "0x3000", "limit": None}
    out = capsys.readouterr().out
    assert "0x2000" in out and "= 42" in out
    assert "g_handler" in out and "-> 0x5000 on_message" in out
    assert '-> 0x6000 "hello"' in out


def test_data_vars_forwards_limit_and_notes_truncation(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "demo_app.bndb"}]},
        "data_vars": {
            "ok": True,
            "result": {"kind": "data_vars", "has_more": True, "items": [
                {"a": "0x2000", "n": "", "t": "int32_t", "w": 4, "v": 1},
            ]},
        },
    })

    rc = bn.cli.main(["data", "vars", "--start", "0x2000", "--end", "0x3000", "--limit", "1"])

    assert rc == 0
    assert calls[1]["params"] == {"start": "0x2000", "end": "0x3000", "limit": 1}
    out = capsys.readouterr().out
    assert "more data vars remain" in out
    assert "0x2001" in out  # resume hint: last address + 1


def test_data_symbols_lists_address_name_pairs(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "demo_app.bndb"}]},
        "data_symbols": {
            "ok": True,
            "result": {"kind": "data_symbols", "total": 2, "offset": 0, "limit": None,
                       "returned": 2, "has_more": False, "items": [
                           {"a": "0x2000", "n": "g_state"},
                           {"a": "0x2010", "n": "g_table"},
                       ]},
        },
    })

    rc = bn.cli.main(["data", "symbols"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "data_symbols"]
    # Unbounded by default: the index build wants every data global in one call.
    assert calls[1]["params"] == {"offset": 0, "limit": None}
    out = capsys.readouterr().out
    assert "0x2000  g_state" in out
    assert "0x2010  g_table" in out
    assert "showing" not in out  # nothing truncated: no paging footer


def test_data_symbols_pages_and_prints_a_resume_footer(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {"ok": True, "result": [{"target_id": "1:1:1", "selector": "demo_app.bndb"}]},
        "data_symbols": {
            "ok": True,
            "result": {"kind": "data_symbols", "total": 900, "offset": 0, "limit": 2,
                       "returned": 2, "has_more": True, "items": [
                           {"a": "0x2000", "n": "g_state"},
                           {"a": "0x2010", "n": "g_table"},
                       ]},
        },
    })

    rc = bn.cli.main(["data", "symbols", "--limit", "2"])

    assert rc == 0
    assert calls[1]["params"] == {"offset": 0, "limit": 2}
    out = capsys.readouterr().out
    assert "showing 2 of 900" in out
    assert "--offset 2" in out


def test_batch_rejects_explicit_empty_manifest_target(fake_transport, monkeypatch, capsys, tmp_path):
    # #690 r4: a manifest {"target": ""} (an unset shell variable templated
    # into the file) must error like `-t ""` does -- not ride the focused-tab
    # convenience, and not be silently overwritten by a sticky pin.
    import json as _json
    manifest = tmp_path / "batch.json"
    manifest.write_text(_json.dumps({"target": "", "ops": [{"op": "rename_symbol"}]}))
    for sticky in ({}, {"target": "beta.so"}):
        calls = fake_transport({})
        monkeypatch.setattr(bn.cli.session_state, "read", lambda s=sticky: s)

        rc = bn.cli.main(["batch", "apply", str(manifest)])

        assert rc == 2, sticky
        assert calls == [], sticky
        err = capsys.readouterr().err
        assert "Manifest" in err and "target is empty" in err, sticky


def test_fanout_all_instances_rejects_explicit_empty_target(fake_transport, monkeypatch, capsys):
    # #690 r4: the --all-instances fan-out branch returns before
    # _resolve_target, so it needs the same explicit-empty rejection -- the
    # empty selector must not be silently discarded into an auto-survey.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport({})

    rc = bn.cli.main(["function", "list", "--all-instances", "-t", ""])

    assert rc == 2
    assert calls == []
    assert "--target is empty" in capsys.readouterr().err


def test_strings_discloses_the_dropped_count_795(fake_transport, capsys):
    """#795: the filter's denominator cost a SECOND invocation.

    `strings --count --format json` reported 1359 and the same command with
    `--probable-format-strings` reported 30, and NOTHING in either answer said
    what happened to the 1329 in between -- so an agent had to spend a second
    unfiltered call to learn the filter's denominator. The bridge now reports the
    dropped count on both the count result and the list envelope, and text mode
    states it next to the page (the same shape `imports` uses for the exports its
    own filter excludes, #202).
    """
    envelope = {"items": [{"address": "0x401000", "length": 6, "chars": 6,
                           "type": "ascii", "value": "%s%s"}],
                "total": 30, "offset": 0, "limit": 1, "returned": 1,
                "has_more": True, "filtered": 1329}
    calls = fake_transport({"strings": {"ok": True, "result": envelope}})

    rc = bn.cli.main(["strings", "--target", "active", "--probable-format-strings",
                      "--limit", "1", "--format", "text"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert "%s%s" in stdout
    assert "// showing 1 of 30 (29 more)" in stdout
    assert "// 1329 string(s) filtered out by the active filters" in stdout

    rc = bn.cli.main(["strings", "--target", "active", "--probable-format-strings",
                      "--count", "--format", "json"])
    assert rc == 0
    assert calls[-1]["params"]["count_only"] is True
    capsys.readouterr()          # the list envelope this fake still answers with
    # The count result carries the same denominator (`filtered`), so the JSON
    # consumer reads 30 and 1329 from ONE invocation.
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": 30, "total": 30, "filtered": 1329}}})
    rc = bn.cli.main(["strings", "--target", "active", "--probable-format-strings",
                      "--count", "--format", "json"])
    assert rc == 0
    counted = json.loads(capsys.readouterr().out)
    assert counted["count"] == 30 and counted["filtered"] == 1329


def test_strings_count_text_states_the_dropped_count_795(fake_transport, capsys):
    """The `--count` line is the one an agent stops on, so it carries the
    denominator itself: `Total strings: 30 (1329 filtered out ...)`. An
    unfiltered dump is unchanged (nothing was dropped, nothing to disclose)."""
    calls = fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": 30, "total": 30, "filtered": 1329}}})
    rc = bn.cli.main(["strings", "--target", "active", "--probable-format-strings",
                      "--count", "--format", "text"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == (
        "Total strings: 30 (1329 filtered out by the active filters)")

    calls = fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": 1359, "total": 1359, "filtered": 0}}})
    rc = bn.cli.main(["strings", "--target", "active", "--count", "--format", "text"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "Total strings: 1359"


def test_strings_count_line_reads_the_denominator_through_the_choke_point_795(
        fake_transport, capsys):
    """One count contract, on BOTH surfaces that state this number (#619/#795).

    The listing renderer reads `filtered` through `_count_field`; the `--count`
    line tested it with `isinstance(int)` -- a SECOND decider over a question
    this codebase already decided, and it answers differently in both
    directions. A producer that spells counts as text (`"1329"`) dropped the
    disclosure entirely, which is the second unfiltered invocation #795 exists
    to remove; and `bool` IS an `int` in Python, so `filtered: true` rendered
    "(True filtered out by the active filters)" -- a flag printed as a quantity,
    with nothing saying the number was unreadable.
    """
    # (a) A numeric-string count states the same denominator an int one does.
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": "30", "total": "30", "filtered": "1329"}}})
    assert bn.cli.main(["strings", "--target", "active", "--count",
                        "--probable-format-strings", "--format", "text"]) == 0
    assert capsys.readouterr().out.strip() == (
        "Total strings: 30 (1329 filtered out by the active filters)")

    # (b) A bool is not a count. It is disclosed as unreadable -- the same
    # reading the listing renderer already gives it -- never rendered as one.
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": 30, "total": 30, "filtered": True}}})
    assert bn.cli.main(["strings", "--target", "active", "--count",
                        "--format", "text"]) == 0
    flagged = capsys.readouterr().out
    assert "True filtered out" not in flagged
    assert flagged.startswith("Total strings: 30")
    assert "filtered-string count is not a number that can be read" in flagged

    # (c) An unreadable HEADLINE count is `?`, not a fabricated 0: "Total
    # strings: 0" from a container reads byte-identically to a real empty
    # binary, which is the #683 harm the choke point's stated sibling exists for.
    fake_transport({"strings": {"ok": True, "result": {
        "kind": "strings", "count": {"n": 30}, "total": 30}}})
    assert bn.cli.main(["strings", "--target", "active", "--count",
                        "--format", "text"]) == 0
    unreadable = capsys.readouterr().out
    assert unreadable.startswith("Total strings: ?")
    assert "Total strings: 0" not in unreadable

    # (d) The filter cannot have dropped a NEGATIVE number of strings, and
    # both surfaces that state this number say so rather than restating the
    # impossible quantity. `-2 filtered out by the active filters` is the
    # confident-wrong-number harm the choke point exists to end, and it
    # survived the imports repair because the cardinality rule was wired to
    # the excluded key only (#795 round-5 review).
    from bn import formatters
    from bn.commands.misc import _strings_count_text

    for dropped in (-2, "-02"):
        line = _strings_count_text({"count": 30, "filtered": dropped})
        listing = formatters._render_strings_text(
            {"items": [], "total": 0, "count": 30, "filtered": dropped})
        for rendered in (line, listing):
            assert "-2" not in rendered, (dropped, rendered)
            assert "malformed filtered field" in rendered, (dropped, rendered)
        # ...and neither renders byte-identically to the unfiltered dump, which
        # is the reading a silent drop would have given it.
        quiet_line = _strings_count_text({"count": 30})
        quiet_listing = formatters._render_strings_text(
            {"items": [], "total": 0, "count": 30})
        assert line.split("\n! malformed")[0] != quiet_line, (dropped, line)
        assert listing.split("\n! malformed")[0] != quiet_listing, (dropped, listing)


class _ProbedValue(int):
    """An integer that also answers mapping reads, recording the path taken.

    Being an `int` subclass is what lets a renderer format it, compare it and
    hand it to a count helper while the probe is still watching; answering
    `.get`/`[]`/`in`/`len` is what lets a NESTED read be recorded too."""

    def __new__(cls, seen, path, number):
        value = int.__new__(cls, number)
        value._seen, value._path = seen, path
        return value

    def _child(self, key):
        self._seen.add(self._path + (key,))
        return _ProbedValue(self._seen, self._path + (key,), int(self))

    def get(self, key, default=None):
        return self._child(key)

    def __getitem__(self, key):
        return self._child(key)

    def __contains__(self, key):
        return True

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class _ProbedPayload(dict):
    """A payload that records every key path a renderer actually reads.

    A `dict` subclass, because every renderer here opens with
    `isinstance(value, dict)`."""

    def __init__(self, seen, number):
        super().__init__()
        self._seen, self._number = seen, number

    def _child(self, key):
        self._seen.add((key,))
        return _ProbedValue(self._seen, (key,), self._number)

    def get(self, key, default=None):
        return self._child(key)

    def __getitem__(self, key):
        return self._child(key)

    def __contains__(self, key):
        return True

    def keys(self):
        return ()

    def items(self):
        return ()


def _observed_key_paths(renderer):
    """The payload key paths a renderer READS, observed by driving it.

    Every previous version of this harvest parsed the renderer's source for
    a `.get("literal")`-shaped read, and every review round found the next
    count line spelled just outside whatever the parser matched: a key held
    in a module constant, a key one level down, a key reached through an
    imported helper. Asking the renderer instead of reading it ends that
    class -- a key it looks at is recorded however the lookup is spelled,
    and a key it never looks at could not be probed anyway."""
    seen: set = set()
    try:
        renderer(_ProbedPayload(seen, 4242))
    except Exception:
        # Partial observation is still observation: the paths reached before
        # the renderer gave up are real reads, and they are probed below.
        pass
    return seen


def _payload_at(path, value):
    """`value` placed at `path` in an otherwise empty payload."""
    placed = value
    for key in reversed(path):
        placed = {key: placed}
    return placed


def _defining_module(renderer):
    """The module a renderer belongs to, or None when it cannot be told.

    `functools.partial` and friends carry no `__module__` of their own, and
    a renderer whose home cannot be established has to fail CLOSED rather
    than drop out of the population unnoticed."""
    module = getattr(renderer, "__module__", None)
    if module is None:
        module = getattr(getattr(renderer, "func", None), "__module__", None)
    return module


def _installed_text_renderers(module):
    """Every `text_renderer=` a module installs, resolved to a CALLABLE.

    The population has to be the renderers themselves rather than their
    spellings, which is the whole lesson of #795's review history: this guard
    selected first on two hand-named functions, then on a `*_count_text` name
    suffix, then on that suffix plus a refusal for any lambda whose source
    contained the substring `count`. Each of those is a spelling test, and
    each was walked past by the next count line that spelled itself
    differently.

    A `lambda` is not actually unprobeable -- it is only unNAMEable.
    Compiling the expression in its own module's namespace yields the same
    callable the registry installs, so it is probed like any other, and a
    lambda that correctly DELEGATES to a choke-point renderer passes instead
    of being refused for its shape. Returns `(label, callable)` pairs for the
    renderers this module DEFINES; one imported from `bn.formatters` is
    already covered by that module's own differential, mirror and raise
    sweep."""
    import ast
    import inspect
    import pathlib

    tree = ast.parse(pathlib.Path(inspect.getfile(module)).read_text(encoding="utf-8"))

    def resolve(node):
        if isinstance(node, ast.Name):
            return [(node.id, getattr(module, node.id, None))]
        if isinstance(node, ast.IfExp):        # `A if flag else B` installs both
            return resolve(node.body) + resolve(node.orelse)
        if isinstance(node, ast.Lambda):
            expression = ast.Expression(body=node)
            ast.fix_missing_locations(expression)
            try:
                fn = eval(compile(expression, "<text_renderer>", "eval"),
                          vars(module))
            except Exception:                  # closes over a local: unprobeable
                fn = None
            return [(ast.unparse(node), fn)]
        return [(ast.unparse(node), None)]

    installed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "text_renderer":
            installed.extend(resolve(node.value))
    # Unprobeable means "no callable came back" or "a callable whose home
    # module cannot be established". One that resolves fine and simply lives
    # in `bn.formatters` is neither, and is covered there.
    unresolved = [label for label, fn in installed
                  if not callable(fn) or _defining_module(fn) is None]
    mine = [(label, fn) for label, fn in installed
            if callable(fn) and _defining_module(fn) == module.__name__]
    return mine, unresolved


def _count_baseline(renderer, path):
    """The well-formed shape this renderer states a count FROM, if it does.

    `int` for a line that states a counter, `list` for one that states
    `len(<rows>)`. Decided by driving the renderer, so a number formatted,
    truncated or mapped through a lookup before printing still counts: the
    rendering merely has to CHANGE with the value."""
    try:
        stated, nudged = (renderer(_payload_at(path, 4242)),
                          renderer(_payload_at(path, 4243)))
        if isinstance(stated, str) and ("4242" in stated or stated != nudged):
            return "int"
    except Exception:
        pass
    try:
        three, four = (renderer(_payload_at(path, [None] * 3)),
                       renderer(_payload_at(path, [None] * 4)))
        if isinstance(three, str) and three != four:
            return "list"
    except Exception:
        pass
    return None


def test_every_count_line_this_module_installs_reads_through_the_choke_point_795():
    """One count contract, on every `--count` line this module renders (#619).

    `bn.formatters` keeps the reading honest for the renderers that live
    there, and the probe population in `tests/test_cli_formatters.py` is
    derived from that module -- so a `text_renderer=` DEFINED in a command
    module is reached by no differential, no mirror and no raise sweep. This
    module's count lines were all written that way.

    Neither the population, the selection, nor the key harvest is a spelling
    any more, because every spelling-shaped version of this guard was walked
    past by the next count line. Every installed renderer is resolved to the
    callable the registry installs; the keys it reads are OBSERVED by driving
    it with a payload that records each lookup, at any depth and however the
    key is written; and it is a count line if its rendering is sensitive to
    that number. Whatever answers yes is driven over the shapes a raw read
    gets wrong -- and a renderer that RAISES on one of them fails here too,
    because costing the caller a whole render is the other half of #619.
    """
    from bn.commands import misc

    renderers, unresolved = _installed_text_renderers(misc)
    # The honest remainder of the old lambda refusal: something this harness
    # cannot turn into a callable it can place, so no probe can reach it.
    assert not unresolved, (
        "these installed renderers cannot be resolved to a probeable "
        f"callable, so no guard can see what they render: {unresolved}")

    stated = []
    for label, renderer in renderers:
        for path in sorted(_observed_key_paths(renderer)):
            kind = _count_baseline(renderer, path)
            if kind is None:
                continue
            where = (label, ".".join(path))
            stated.append(where)

            def render(value, _r=renderer, _p=path):
                return _r(_payload_at(_p, value))

            if kind == "list":
                # A count stated as `len(<rows>)` cannot be probed with an
                # integer at all, and the shape that harms it is a payload
                # that is COUNTABLE but wrong: `len()` of a mapping is its
                # key count and `len()` of a string is its length, so either
                # renders as a confident row count nobody has.
                rows = render([None] * 3)
                assert render({"a": 1, "b": 2, "c": 3}) != rows, where
                assert render("abc") != rows, where
                continue

            zero = render(0)

            # (a) A bool is not a count. `bool` IS an `int`, so a raw read
            # prints the flag as a quantity -- and a read that coerces with
            # `int()` prints it as the quantity `1`, which is why this asks
            # whether the flag renders like a NUMBER rather than whether the
            # word "True" appears.
            flagged = render(True)
            assert "True" not in flagged, where + (flagged,)
            assert flagged != render(1), where + (flagged,)
            assert render(False) != zero, where

            # (b) A numeric string IS a count, and states the same line the
            # integer spelling does. A raw `isinstance(int)` silently drops
            # the whole qualifier for a producer that spells counts as text.
            assert render("7") == render(7), where

            # (c) A non-integral number is not a count, and must not be
            # quietly truncated into one.
            assert render(1.5) != render(1), where

            # (d) A container is disclosed, never interpolated into the line
            # as a raw Python repr -- and never rendered as the real zero it
            # is not. Compared against the well-formed zero rather than
            # pattern-matched on `": 0"`, which a renderer with a different
            # separator walks past.
            unreadable = render({"n": 1})
            assert "{" not in unreadable and "}" not in unreadable, where + (unreadable,)
            assert unreadable != zero, where + (unreadable,)

    # ...and the derivation cannot quietly degrade into probing nothing. The
    # five historical count lines are a FLOOR, not the filter: one of them
    # ceasing to state its number is a surface that changed behaviour, and a
    # SIXTH line is caught by the probe above whatever it is called.
    assert stated, "no installed renderer states a count, so this proves nothing"
    assert {label for label, _path in stated} >= {
        "_exports_count_text", "_go_functions_count_text", "_imports_count_text",
        "_sections_count_text", "_strings_count_text"}, sorted(stated)


def test_the_three_imports_surfaces_agree_about_the_excluded_count_795():
    """One payload, one answer -- across all three surfaces that state it.

    Making only the `--count` line strict was a regression dressed as a fix:
    the paged listing and the `--summary` card still tested the same key with
    `isinstance(int)`, so a bridge reporting it as text got the denominator
    from one surface and silence from the other two, and a bool got
    "(True self-defined excluded)", "// True self-defined export(s) excluded"
    and "self-defined excluded: True" -- three descriptions of one payload,
    which is worse than the single wrong answer they agreed on before.

    Round-4 review found the repair had left the DECISION duplicated three
    times and the copies disagreeing on the one shape this matrix did not
    probe: a NEGATIVE count was stated by the `--count` line and dropped
    silently by the other two, where base had agreed. It also found the
    listing's unreadable branch measured by nothing -- every assertion here
    was satisfied by the trailing `@_discloses` boundary note, which the
    renderer gets whether or not it states the row itself. So the property is
    now asserted per surface on the renderer's OWN body, with the boundary
    note cut off.
    """
    from bn import formatters
    from bn.commands.misc import _imports_count_text

    def surfaces(excluded):
        return (_imports_count_text({"count": 9, "self_defined_excluded": excluded}),
                formatters._render_name_address_list_text(
                    {"items": [], "total": 0, "self_defined_excluded": excluded}),
                formatters._render_imports_summary_text(
                    {"total_symbols": 9, "self_defined_excluded": excluded}))

    def bodies(rendered):
        """Each surface's own text, with the shared boundary note removed.

        `@_discloses` appends `! malformed <key> field: ...` to EVERY renderer
        that recorded a skew, so an assertion over the whole string is
        satisfied by the boundary even when the renderer states nothing --
        which is how the listing's `elif _field_skewed` branch shipped
        unmeasured. Cutting the note is what makes each surface answer for
        itself."""
        return tuple(r.split("\n! malformed")[0] for r in rendered)

    silent = bodies((_imports_count_text({"count": 9}),
                     formatters._render_name_address_list_text({"items": [], "total": 0}),
                     formatters._render_imports_summary_text({"total_symbols": 9})))

    # A text-spelled count IS a count, and every surface states it exactly as
    # it states the integer spelling.
    assert surfaces("3") == surfaces(3)
    for rendered in surfaces("3"):
        assert "3" in rendered and "excluded" in rendered, rendered

    # Every shape no count reads out of is stated AS unreadable, by each
    # surface in its own words -- so the three agree, and none of them renders
    # byte-identically to the page where the key claimed nothing.
    #
    # A NEGATIVE count is in this set on purpose: `_count_field` reads `-02`
    # back as -2 by the #866 contract, but a survey cannot exclude a negative
    # number of symbols, so it is not a count either. Base agreed (all three
    # silent); the round-3 repair made the `--count` line state `(-2
    # self-defined excluded)` while the other two stayed silent.
    for excluded in (True, {"n": 3}, [1, 2, 3], "lots", 1.5, -2, "-02"):
        rendered = surfaces(excluded)
        for one in rendered:
            assert f"{excluded}" not in one, (excluded, one)
            assert "malformed self_defined_excluded field" in one, (excluded, one)
        for surface, body, quiet in zip(("count", "listing", "summary"),
                                        bodies(rendered), silent):
            assert body != quiet, (
                f"the imports {surface} surface renders an unreadable "
                f"{excluded!r} byte-identically to a payload that claimed "
                "nothing, so the whole disclosure is the shared boundary "
                "note -- state it on the surface itself")

    # ...and an ABSENT key claimed nothing, so every surface stays silent.
    for rendered in (_imports_count_text({"count": 9}),
                     formatters._render_name_address_list_text({"items": [], "total": 0}),
                     formatters._render_imports_summary_text({"total_symbols": 9})):
        assert "excluded" not in rendered and "malformed" not in rendered, rendered


def test_estimate_output_preflights_the_raw_bytes_read_796(fake_transport, capsys):
    """#796: `read --encoding bytes` is a SECOND emit path, and it must preflight
    like the first one.

    `read` is marked estimable, but its `--encoding bytes` branch returns before
    `_call` and wrote the payload straight to `sys.stdout.buffer` -- so
    `--estimate-output` was accepted and ignored at rc 0, dumping the very bytes
    the flag's own help promises it prints INSTEAD of ("the read still runs;
    nothing is written"). The same command with `--encoding hex` printed the
    estimate, so one command's two halves disagreed about what the flag means.
    """
    payload = "41" * 64                      # 64 bytes of 'A'
    fake_transport({"read": {"ok": True, "result": {
        "kind": "bytes", "address": "0x401000", "length": 64, "hex": payload}}})

    assert bn.cli.main(["read", "0x401000", "--length", "64", "--encoding", "bytes",
                        "--estimate-output", "--target", "active"]) == 0
    text = capsys.readouterr().out
    assert "estimated: true" in text
    assert "tokens: " in text and "tokenizer: estimate" in text
    assert "bytes: 64" in text                   # the raw payload's real size
    assert "--length" in text                    # this command's own slicing knob
    assert "AAAA" not in text                    # ...and NOT the payload itself

    # Machine-readable under --format json, and it measures the RAW byte payload
    # (64 bytes), not a hex rendering of it.
    assert bn.cli.main(["read", "0x401000", "--length", "64", "--encoding", "bytes",
                        "--estimate-output", "--format", "json",
                        "--target", "active"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["estimated"] is True and envelope["format"] == "bytes"
    assert envelope["bytes"] == 64 and envelope["tokens"] > 0
    assert envelope["summary"] == {"kind": "bytes", "address": "0x401000", "length": 64}
    assert "hex" not in envelope

    # Without the flag the raw bytes still reach stdout unchanged: the preflight
    # is opt-in and replaces nothing when it is not asked for.
    assert bn.cli.main(["read", "0x401000", "--length", "64", "--encoding", "bytes",
                        "--target", "active"]) == 0
    assert capsys.readouterr().out == "A" * 64


def test_estimate_output_preflights_a_large_read_796(fake_transport, capsys):
    """#796: `--estimate-output` on the reads whose cost you want to know FIRST.

    The issue's own repro was `bn function list --limit 5 --estimate-output` ->
    `error: unrecognized arguments` (rc 2), with no way to learn a read's size
    before paying for it in context. The flag is accepted by every command that
    renders a payload; under text it prints the size and this command's own
    slicing knob, and it does NOT print the rows.
    """
    calls = fake_transport({"list_functions": {"ok": True, "result": {
        "kind": "functions",
        "items": [{"name": f"sub_{i:06d}", "address": hex(0x401000 + i * 0x10)}
                  for i in range(200)],
        "total": 200, "offset": 0, "limit": 5, "returned": 200, "has_more": True}}})

    rc = bn.cli.main(["function", "list", "--limit", "5", "--estimate-output",
                      "--target", "active"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "estimated: true" in out
    assert "tokens: " in out and "tokenizer: estimate" in out
    assert "--limit" in out                      # the slicing knob this command takes
    assert "sub_000000" not in out               # ...and NOT the payload itself
    assert calls[-1]["op"] == "list_functions"

    # Under --format json the same run is machine-readable, so a caller can branch
    # on the cost without parsing prose.
    rc = bn.cli.main(["function", "list", "--limit", "5", "--estimate-output",
                      "--format", "json", "--target", "active"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["estimated"] is True and envelope["tokens"] > 0
    assert envelope["summary"]["total"] == 200
    assert "items" not in envelope


def test_estimate_output_covers_per_function_reads_796(fake_transport, capsys):
    """The same preflight for the large PER-FUNCTION reads (`decompile`, `il`,
    `strings`), whose cost is the one an agent most often misjudges: the flag
    reports the size of the rendering the caller would have received, and names
    that command's own slicing flag (`decompile`/`il` slice with --lines)."""
    fake_transport({
        "decompile": {"ok": True, "result": {"text": "int parse_hdr(char *p)\n{\n" + "  *p++;\n" * 500 + "}\n"}},
        "strings": {"ok": True, "result": {
            "items": [{"address": hex(0x402000 + i), "length": 5, "chars": 5,
                       "type": "ascii", "value": f"str{i}"} for i in range(50)],
            "total": 50, "offset": 0, "limit": 50, "returned": 50, "has_more": False,
            "filtered": 0}},
    })

    assert bn.cli.main(["decompile", "parse_hdr", "--estimate-output",
                        "--target", "active"]) == 0
    decompile_out = capsys.readouterr().out
    assert "estimated: true" in decompile_out
    assert "--lines START:END" in decompile_out
    assert "*p++" not in decompile_out            # the decompilation is NOT printed
    assert "tokens: " in decompile_out

    assert bn.cli.main(["strings", "--estimate-output", "--target", "active"]) == 0
    strings_out = capsys.readouterr().out
    assert "estimated: true" in strings_out
    assert "--limit" in strings_out
    assert '"str0"' not in strings_out            # the rows are NOT printed


# Argument values that make a read command's parser accept an invocation, keyed
# by the registry's OWN argument name (positionals) or the parser's `dest`
# (required options). One vocabulary, not a per-command argv list: a command
# that grows a required argument this table cannot fill fails the sweep loudly
# instead of silently dropping out of the coverage claim.
_ESTIMATE_ARG_VALUES = {
    "identifier": "main", "callee": "main", "function": "main",
    "name": "Widget", "query": "a", "address": "0x401000",
    "type_name": "int32_t", "struct_name": "hdr",
    "at": "0x401000", "var": "x#1", "start": "0x401000", "end": "0x402000",
    "arg_index": "1", "sinks": "arg:memcpy:0", "sources": "call:recv",
}
# One permissive reply for every op. The sweep is about which EMIT PATH a
# handler takes, not about any single op's payload shape.
_ESTIMATE_STUB = {"ok": True, "result": {
    "kind": "probe", "items": [], "total": 0, "offset": 0, "limit": 10,
    "returned": 0, "has_more": False, "count": 0, "hex": "41414141",
    "text": "probe\n", "name": "probe", "address": "0x401000",
}}


def _estimate_emit_modes(leaf_parser):
    """Every alternate emit mode a command's OWN parser offers, derived from it.

    A handler is free to branch on its own flags and take a different way out --
    `read --encoding bytes` returns before `_call` and writes to
    `sys.stdout.buffer` -- so sweeping the boolean flags and every value of
    every choice flag is what reaches the second path. Deriving the modes from
    the parser (rather than listing them) is what makes the sweep grow with the
    surface instead of going stale the next time a handler grows a branch.
    """
    import argparse

    modes = [[]]
    for action in leaf_parser._actions:
        if not action.option_strings:
            continue
        flag = action.option_strings[0]
        # --out is mutually exclusive with the flag under test (argparse rc 2),
        # --stdin would block on a tty, and --help exits before dispatch.
        if flag in ("--estimate-output", "--out", "--help", "--stdin"):
            continue
        if isinstance(action, argparse._StoreTrueAction):
            modes.append([flag])
        elif action.choices:
            modes.extend([flag, str(choice)] for choice in action.choices)
    return modes


def _estimate_required_flags(leaf_parser):
    """The command's own REQUIRED options, filled from the shared vocabulary.

    Derived from the parser rather than listed per command: `taint forward`
    needs `--source`, `evidence virtual-call` needs `--at`, and a command that
    grows a new required option must either be fillable or say so."""
    argv, unfillable = [], []
    for action in leaf_parser._actions:
        if not action.option_strings or not action.required:
            continue
        if action.dest not in _ESTIMATE_ARG_VALUES:
            unfillable.append(action.dest)
            continue
        argv += [action.option_strings[0], _ESTIMATE_ARG_VALUES[action.dest]]
    return argv, unfillable


def _is_error_envelope(out: str) -> bool:
    """A refusal reported as a machine-readable envelope is not a payload."""
    try:
        return json.loads(out).get("ok") is False
    except (ValueError, AttributeError):
        return False


def _is_estimate_envelope(out: str) -> bool:
    """Did this invocation print the preflight instead of the payload?"""
    try:
        return json.loads(out).get("estimated") is True
    except (ValueError, AttributeError):
        return any(line.strip() == "estimated: true" for line in out.splitlines())


def test_estimate_output_is_advertised_only_where_it_is_implemented_796(
        fake_transport, capsys, monkeypatch):
    """#796 review: the flag lives on the code path that implements it, not on
    every command that happens to share an output-option group.

    The wide placement advertised `--estimate-output` on all 90 leaf parsers while
    only the `_call` -> `_render_result` path honors it, so `bn close`, `bn save`,
    `bn load`, `bn refresh` and `bn py exec` executed their side effect and then
    printed a byte count over the outcome (`save` wrote a BNDB while the flag's own
    help promised nothing was written), and 18 `_emit_result` commands ignored it
    outright -- `bn capabilities --estimate-output` printed the very payload the
    flag exists to avoid.

    The rule is the `fanout=True` precedent (#169 L1 review): an EXPLICIT
    allow-list on the registry. What proves the allow-list is a coverage claim
    rather than a wish is the sweep below, which RUNS every marked command in
    every emit mode its parser offers and reads what landed on stdout.

    That is deliberately not a source-text derivation. The previous version of
    this test derived the set by grepping each handler for `_call(` with no
    `_mutate`/`_emit_result` beside it, and `read` satisfies that grep while its
    `--encoding bytes` branch returns above the `_call` and writes the payload
    straight to `sys.stdout.buffer` -- a second emit path no reading of the
    source's call names can see, which shipped the flag accepted-and-ignored at
    rc 0 on exactly the read whose size a caller most wants to preflight.
    """
    import argparse

    parser = bn.cli.build_parser()
    # `main()` rebuilds the whole argparse tree per call (~110 ms x 285 probes),
    # and it only ever PARSES with it -- nothing below mutates parser state --
    # so the sweep hands it the one tree it already built.
    monkeypatch.setattr(bn.cli, "build_parser", lambda: parser)

    def leaf(path):
        current = parser
        for name in path:
            action = next(a for a in current._actions
                          if isinstance(a, argparse._SubParsersAction))
            current = action.choices[name]
        return current

    advertised = {tuple(spec["path"]) for spec in bn.cli._COMMANDS
                  if "--estimate-output" in bn.cli._known_option_strings(leaf(spec["path"]))}
    marked = {tuple(spec["path"]) for spec in bn.cli._COMMANDS if spec.get("estimable")}

    # A DUAL-ROLE node (`types`, `exports`) is a leaf AND a group: argparse builds
    # ONE parser for both roles, so a flag attached there is accepted BEFORE the
    # subcommand is dispatched -- `bn types --estimate-output declare ...` ran the
    # declaration and printed a byte count over it, and `bn types --estimate-output
    # show X` had the flag clobbered back to False by the leaf default (#251's
    # hazard, on the one node class where an "intermediate" parser and a leaf are
    # the same object). The builder therefore declines to advertise it on any
    # group node, derived from the registry rather than a second hand-kept list --
    # and that derivation is what the assertions below check in both
    # directions, so neither a new command nor a new subcommand can reopen it.
    groups = {tuple(spec["path"])[:i] for spec in bn.cli._COMMANDS
              for i in range(1, len(tuple(spec["path"])))}
    assert groups, "no group paths at all -- the derivation is broken"
    assert all("--estimate-output" not in bn.cli._known_option_strings(leaf(path))
               for path in groups), (
        "a GROUP parser carries --estimate-output, so it is accepted before the "
        "subcommand is dispatched: a mutation behind it runs while its outcome is "
        f"replaced by a size ({sorted(p for p in groups if '--estimate-output' in bn.cli._known_option_strings(leaf(p)))})")

    assert advertised == marked - groups, (
        f"advertised {sorted(advertised - (marked - groups))} without being marked, "
        f"or marked-and-advertisable but not advertised "
        f"{sorted((marked - groups) - advertised)}")
    assert len(marked) == 49, (
        f"{len(marked)} commands are marked estimable, not 49 -- a command that "
        "joins or leaves this set is a deliberate change to the coverage claim, so "
        "move the number in the same commit")
    # ...of which the two dual-role leaves cannot carry the flag (see above), so
    # 47 advertise it and 45 refuse it by absence.
    assert len(advertised) == 47 and len(bn.cli._COMMANDS) - len(advertised) == 45
    # ...and everything else refuses it BY ABSENCE (argparse's own rc 2), which is
    # stronger than a bespoke refusal: there is no path on which the flag is
    # accepted and ignored, because the parser never builds it.
    assert all(not spec.get("estimable") for spec in bn.cli._COMMANDS
               if tuple(spec["path"]) not in marked)
    assert len(bn.cli._COMMANDS) - len(marked) == 43

    # THE COVERAGE CLAIM, observed. One invariant over every marked command in
    # every emit mode its parser offers: whatever lands on stdout is the
    # ESTIMATE ENVELOPE or a machine-readable refusal -- never a payload -- and
    # a run that exits 0 produced the envelope. A handler that reaches an emit
    # path the flag does not gate lands in `leaked` with the argv that got there.
    positionals = {tuple(spec["path"]): [a[0][0] for a in spec["args"]
                                         if not a[0][0].startswith("-")]
                   for spec in bn.cli._COMMANDS}
    leaked, probed, emitted, covered = [], 0, 0, set()
    for path in sorted(advertised):
        required, unfillable = _estimate_required_flags(leaf(path))
        unfillable += [p for p in positionals[path]
                       if p not in _ESTIMATE_ARG_VALUES]
        assert not unfillable, (
            f"{' '.join(path)} takes arguments this sweep cannot synthesize "
            f"({unfillable}); add them to _ESTIMATE_ARG_VALUES rather than "
            "letting the command drop out of the coverage claim")
        # A NONEXISTENT instance id on every probe, never a bare `--target
        # active`. The transport is stubbed here so nothing can leave the
        # process either way, but `active` resolves to whatever target some
        # other session has focused, and a sweep this wide is exactly the shape
        # that must not depend on a fixture to stay harmless: if a change ever
        # re-routes one of these 285 invocations past the stub, it must find
        # nothing on the other end rather than a live view.
        base = (list(path)
                + [_ESTIMATE_ARG_VALUES[p] for p in positionals[path]]
                + required
                + ["--instance", "prfleet-nonexistent-879", "--target", "active"])
        for mode in _estimate_emit_modes(leaf(path)):
            argv = base + mode + ["--estimate-output"]
            fake_transport(default=_ESTIMATE_STUB)
            try:
                rc = bn.cli.main(argv)
            except SystemExit as exc:                 # argparse refusals
                rc = exc.code
            out = capsys.readouterr().out
            probed += 1
            estimated = _is_estimate_envelope(out)
            if rc == 0:
                emitted += 1
                covered.add(path)
            if estimated:
                continue
            if rc == 0 or (out.strip() and not _is_error_envelope(out)):
                leaked.append((" ".join(argv), out[:160]))
    assert not leaked, (
        "these invocations of --estimate-output put something other than the "
        f"estimate envelope on stdout: {leaked[:6]}")
    # ...and the sweep cannot quietly degrade into "every probe errored out":
    # every advertised command reached an emitting path at least once, and the
    # probe count is pinned so a mode class cannot silently stop being swept.
    assert covered == advertised, (
        f"never reached an emitting path: {sorted(advertised - covered)}")
    assert probed == 285, (
        f"the sweep ran {probed} probes, not 285 -- a flag class joining or "
        "leaving the derivation changes the coverage claim, so move the number "
        "in the same commit")
    assert emitted, "no probe reached an emit path at all"


def test_the_output_reference_documents_the_estimate_preflight_796(monkeypatch):
    """#796: the preflight has to be discoverable where an agent looks for it.

    The issue's own repro grepped `src/`, `skills/`, `README.md` and
    `CLAUDE.md`, and `--estimate-output` had zero hits outside `src/` -- so a
    flag whose whole purpose is letting an agent bound a read before paying for
    it shipped invisible to the agent surface it was filed for. Two places
    carry it: the runtime reference that enumerates the output flags and the
    envelope keys, and the reading reference's "bound the read" guidance, which
    is where a reader is already being told to slice.

    The envelope half is derived from the envelopes this CLI actually emits, so
    the reference cannot drift from the payload the next time a key is added.
    All THREE envelopes are drained, because a key that only one of them
    carries is exactly the one a hand-kept list forgets: the spill threshold is
    conditional (`spill_token_limit` appears only when `BN_SPILL_TOKENS` is
    armed), and the raw-byte preflight is a second payload kind.
    """
    from bn.output import estimate_bytes_result, estimate_output_result

    root = Path(bn.cli.__file__).resolve().parents[2]
    runtime = (root / "skills" / "bn" / "reference" / "runtime.md").read_text(encoding="utf-8")
    reading = (root / "skills" / "bn" / "reference" / "reading.md").read_text(encoding="utf-8")

    assert "--estimate-output" in runtime, (
        "the reference that enumerates every output flag does not name "
        "--estimate-output, so the preflight is undiscoverable from the skill")
    assert "--estimate-output" in reading, (
        "the reading reference tells an agent to bound a large read but never "
        "names the flag that measures one first")

    envelopes = [estimate_output_result({"items": [], "total": 0}, fmt="json",
                                        rerun_hint="rerun with --limit").artifact]
    monkeypatch.setenv("BN_SPILL_TOKENS", "40000")
    envelopes.append(estimate_output_result({"items": [], "total": 0}, fmt="json",
                                            rerun_hint="rerun with --limit").artifact)
    envelopes.append(estimate_bytes_result(b"AAAA", fmt="json",
                                           summary={"kind": "bytes"},
                                           rerun_hint="rerun with --length").artifact)
    assert "spill_token_limit" in envelopes[1], (
        "the armed envelope no longer carries the conditional key this guard "
        "exists to reach")
    undocumented = sorted({key for envelope in envelopes for key in envelope
                           if f"`{key}`" not in runtime})
    assert not undocumented, (
        f"the estimate envelope carries keys the reference never states: "
        f"{undocumented}")


def test_estimate_output_is_not_advertised_on_mutations_or_side_effecting_commands_796(
        fake_transport, capsys):
    """The other half of the scoping: the commands that must never take it.

    A mutation prints a status line because that line IS the answer (#645), and a
    side-effecting `_call` command (`save`/`close`/`load`/`refresh`/`py exec`)
    performs its work and then reports it -- for both, `--estimate-output` would
    replace an outcome with a byte count. Neither advertises the flag, so both
    refuse it the way argparse refuses any unknown option (rc 2), before any
    request is sent.
    """
    import argparse

    a_mutation = bn.cli._selected_parser_for_argv(
        bn.cli.build_parser(), ["comment", "set", "0x401000", "note"])
    assert "--estimate-output" not in bn.cli._known_option_strings(a_mutation)

    parser = bn.cli.build_parser()
    calls = fake_transport()

    # An argparse refusal exits 2 the same way a usage error always does (the
    # text-format path raises SystemExit; see `test_argparse_error_text_format_
    # keeps_stdout_empty`), so the flag is refused by the PARSER, not by a gate
    # somebody has to remember to write.
    with pytest.raises(SystemExit) as refused:
        bn.cli.main(["comment", "set", "0x401000", "note", "--estimate-output",
                     "--target", "active"])
    assert refused.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not calls                       # refused by the parser, before the request

    # THE DUAL-ROLE NODES, behaviourally. The flag must not be accepted BEFORE the
    # subcommand, because that occurrence belongs to the group parser: the
    # mutation behind it (`types declare`) executed and reported a size in R2 of
    # this review, and the read behind it (`types show`) ran with the flag
    # silently dropped by the leaf default. Both refuse now, before any request.
    for argv in (["types", "--estimate-output", "declare",
                  "struct DF879Leak { int a; int b; };"],
                 ["types", "--estimate-output", "show", "DF879Leak"],
                 ["exports", "--estimate-output", "list"]):
        calls = fake_transport()
        with pytest.raises(SystemExit) as hijacked:
            bn.cli.main(argv + ["--target", "active"])
        assert hijacked.value.code == 2, argv
        captured = capsys.readouterr()
        assert "unrecognized arguments: --estimate-output" in captured.err, argv
        assert captured.out == "", argv        # NO payload
        assert not calls, argv                 # and NO side effect

    for path in (["save"], ["close"], ["target", "close"], ["load"], ["refresh"],
                 ["py", "exec"], ["go", "rename"], ["batch", "apply"]):
        current = parser
        for name in path:
            action = next(a for a in current._actions
                          if isinstance(a, argparse._SubParsersAction))
            current = action.choices[name]
        assert "--estimate-output" not in bn.cli._known_option_strings(current), path

    # Where it IS advertised, the two answers to "where does this go" are refused
    # by argparse's own mutually exclusive group rather than one silently winning:
    # a caller who asked for a size AND a file asked for two different things.
    calls = fake_transport({"list_functions": {"ok": True, "result": {
        "items": [], "total": 0, "offset": 0, "limit": 5, "returned": 0,
        "has_more": False}}})
    with pytest.raises(SystemExit) as conflicting:
        bn.cli.main(["function", "list", "--estimate-output", "--out", "/tmp/bn-est.json",
                     "--target", "active"])
    assert conflicting.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert not calls
