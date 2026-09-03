from __future__ import annotations

import json
import types
from pathlib import Path

import bn.cli
import pytest

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
    pytest.param('{"target": "x"}', '"ops" array', id="without-ops"),
])
def test_batch_apply_file_clean_error(fake_transport, capsys, tmp_path, manifest, expected):
    # Bad manifest files must surface a clean BridgeError (exit 2), never a
    # client-side traceback (e.g. a bare array hitting _call's dict(params), #48).
    fake_transport()
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


@pytest.mark.parametrize("stdin, expected", [
    pytest.param("   \n", "No manifest on stdin", id="empty"),
    pytest.param("{not valid json", "Invalid JSON in manifest (<stdin>)", id="invalid-json"),
    pytest.param('[{"op": "set_comment", "address": "0x1000", "comment": "x"}]', "must be a JSON object", id="bare-array"),
])
def test_batch_apply_stdin_clean_error(monkeypatch, fake_transport, capsys, stdin, expected):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    fake_transport()

    rc = bn.cli.main(["batch", "apply", "-"])

    assert rc == 2
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err




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

    calls = fake_transport({"batch_apply": {"ok": True, "result": {"preview": False, "success": True, "results": []}}})

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
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": []}}})
    rc = bn.cli.main(["batch", "apply", "-", "-t", "foo.bndb", "-i", "inst"])
    assert rc == 0
    assert calls[-1]["params"].get("target") == "foo.bndb"


def test_batch_apply_cli_target_overrides_manifest(monkeypatch, fake_transport):
    # CLI -t is the explicit per-invocation selector and WINS over a manifest
    # "target" (#366): a fan-out agent that copies the documented {"target":"active"}
    # example but passes a correct -t must not be sabotaged by the in-payload value.
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"target": "active", "ops": []}'))
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": []}}})
    rc = bn.cli.main(["batch", "apply", "-", "-t", "fromflag", "-i", "inst"])
    assert rc == 0
    assert calls[-1]["params"].get("target") == "fromflag"


def test_batch_apply_manifest_target_used_when_no_flag(monkeypatch, fake_transport):
    # Without -t, the manifest "target" is still honored.
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"target": "explicit", "ops": []}'))
    calls = fake_transport({"batch_apply": {"ok": True, "result": {"success": True, "results": []}}})
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
        return {"ok": True, "result": {"results": [], "status": "verified"}}
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
