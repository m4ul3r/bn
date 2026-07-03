from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_spill_warns_about_pipe_trap_when_stdout_is_a_pipe(monkeypatch, capsys):
    """When spilled output is piped (FIFO) into grep/jq, the consumer sees only
    the envelope, not the data -- a no-match then misreads as "absent". Emit an
    explicit caution in that case so the trap isn't silent (#195). The caution
    fires only for a real pipe (not a terminal / capsys), so the other spill
    tests are unaffected."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    def fake_write_output_result(value, *, fmt, out_path, stem):
        return _spill_artifact_namespace("/tmp/decompile.txt")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result", fake_write_output_result)
    monkeypatch.setattr(bn.cli, "_stdout_is_pipe", lambda: True, raising=False)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active"])

    assert rc == 0
    _, stderr = capsys.readouterr()
    assert "spilled to /tmp/decompile.txt" in stderr
    # the pipe-specific caution: names the trap and the --out remedy
    assert "grep" in stderr and "jq" in stderr
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


def test_local_list_text_is_slim(fake_transport, capsys):
    fake_transport({
        "list_locals": {
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
        },
    })

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


def test_local_list_json_retains_ids(fake_transport, capsys):
    fake_transport({
        "list_locals": {
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
        },
    })
    rc = bn.cli.main(["local", "list", "--format", "json", "--target", "active", "sub_401000"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["locals"][0]["local_id"] == "0x401000:param:stack:4:0:1"
    assert payload["locals"][0]["identifier"] == 1


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


def test_max_depth_validator_says_depth_not_index(capsys):
    # Depth flags get a "depth" label, not the generic "index" (#49).
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["taint", "forward", "-f", "main", "--source", "param:0",
                     "--max-depth", "-1", "--target", "active"])
    assert exc.value.code == 2
    _, err = capsys.readouterr()
    assert "depth must be an integer >= 0" in err
    assert "index must be" not in err


def test_entries_validator_hex_aware_and_rejects_zero(capsys):
    # evidence table --entries is wired to the shared count validator: hex is
    # accepted and a degenerate 0/negative is rejected with the standard message (#59).
    with pytest.raises(SystemExit) as exc0:
        bn.cli.main(["evidence", "table", "0x1000", "--entries", "0", "--target", "active"])
    assert exc0.value.code == 2
    _, err0 = capsys.readouterr()
    assert "count must be an integer >= 1" in err0


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


def test_parse_line_range_accepts_colon_and_dash():
    """--lines accepts both START:END and the natural START-END (which matches the
    hyphenated range the output header prints), not just the colon form (#359)."""
    import argparse
    from bn.cli import _parse_line_range
    assert _parse_line_range("1:15") == (1, 15)
    assert _parse_line_range("1-15") == (1, 15)
    for bad in ("0-5", "5-1", "abc", "1:2:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_line_range(bad)


def test_parse_line_range_accepts_bare_count():
    """A bare line count `--lines N` means the first N lines (1..N), matching
    `--count`/`--limit` N, so `disasm --lines 5` no longer errors asking for
    START:END (#371.4)."""
    import argparse
    from bn.cli import _parse_line_range
    assert _parse_line_range("5") == (1, 5)
    assert _parse_line_range("1") == (1, 1)
    # base-0 parsing matches --count/--limit (_positive_int), so a hex bare
    # count works too -- `disasm --lines 0x10` == first 16 lines.
    assert _parse_line_range("0x10") == (1, 16)
    for bad in ("0", "-3", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_line_range(bad)


def test_class_and_go_subcommand_groups_have_help():
    """The `class` and `go` top-level groups must show a one-line help in
    `bn --help`, like every sibling group -- they were blank (#359)."""
    from bn.cli import _GROUP_HELP
    assert _GROUP_HELP.get(("class",))
    assert _GROUP_HELP.get(("go",))
    help_text = bn.cli.build_parser().format_help()
    assert "C++ object-model lens" in help_text
    assert "Go binary symbol recovery" in help_text


def test_class_show_ambiguous_exits_nonzero_keeping_matches(fake_transport, capsys):
    # #413: an ambiguous leaf still returns the informative matches, but with a
    # NON-zero exit so a shell/agent can't read it as a clean single-class result.
    fake_transport({"class_show": {"ok": True, "result": {
        "ambiguous": True, "query": "Foo",
        "matches": [{"name": "a::Foo"}, {"name": "b::Foo"}]}}})
    rc = bn.cli.main(["class", "show", "--target", "active", "Foo", "--format", "json"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "a::Foo" in out and "b::Foo" in out


def test_missing_subcommand_prints_exact_help(capsys):
    rc = bn.cli.main(["struct"])

    assert rc == 1
    stdout, stderr = capsys.readouterr()
    # #251: intermediate group parsers now carry -t/--instance so they can be
    # passed before the leaf; the usage advertises them (and wraps).
    assert "usage: bn struct [-h] [--help-full] [-i INSTANCE] [-t TARGET]" in stdout
    assert "{show,field} ..." in stdout
    assert "--help-full" in stdout
    assert "Show help for this command and all subcommands" in stdout
    assert "usage: bn [-h]" not in stdout
    assert stderr == ""


def test_missing_nested_subcommand_prints_exact_help(capsys):
    rc = bn.cli.main(["struct", "field"])

    assert rc == 1
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct field [-h] [--help-full] [-i INSTANCE] [-t TARGET]" in stdout
    assert "{set,rename,delete} ..." in stdout
    assert "--help-full" in stdout
    assert "Show help for this command and all subcommands" in stdout
    assert "usage: bn [-h]" not in stdout
    assert stderr == ""


def test_root_help_includes_capability_map():
    # #276 Option 1: `bn --help` carries a "which command when" map for the
    # overlapping groups, so an agent routes deterministically instead of
    # re-deriving it. Assert on map-only phrases (the bare verb names already
    # appear in the subcommand listing).
    help_text = bn.cli.build_parser().format_help()

    assert "Picking between overlapping commands:" in help_text
    assert "exact caller -> callsite address mapping" in help_text   # callsites vs xrefs
    assert "follow data source -> sink across calls" in help_text    # taint vs dataflow vs evidence
    assert "backward-slice one call argument" in help_text           # trace vs taint backward
    assert "C++ class hierarchy" in help_text                        # class (RTTI lens) vs evidence
    assert "match by name or regex" in help_text                     # function list vs search
    # The pre-existing spill-envelope note must survive (RawDescription must not
    # drop it).
    assert "spills to disk" in help_text


def test_help_full_prints_recursive_root_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bn.cli.main(["--help-full"])

    assert exc_info.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "usage: bn" in stdout
    # The recursive formatter strips -h/--help-full but keeps the now-present
    # -t/--instance on intermediate group nodes (#251).
    assert "usage: bn struct [-i INSTANCE] [-t TARGET] {show,field} ..." in stdout
    assert "usage: bn struct field set" in stdout
    assert "-h, --help" not in stdout
    assert "--help-full" not in stdout
    assert stderr == ""


def test_help_full_prints_recursive_subtree_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        bn.cli.main(["struct", "field", "--help-full"])

    assert exc_info.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "usage: bn struct field [-i INSTANCE] [-t TARGET]" in stdout
    assert "{set,rename,delete} ..." in stdout
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


def test_build_id_for_package_detects_engine_edit(tmp_path):
    # #161: the whole-package fingerprint changes when ANY .py / model .json in
    # the package changes -- not just bridge.py.
    from bn.version import build_id_for_package
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "bridge.py").write_text("print('a')\n", encoding="utf-8")
    (pkg / "taint_engine.py").write_text("X = 1\n", encoding="utf-8")
    (pkg / "taint_models.json").write_text("{}\n", encoding="utf-8")
    before = build_id_for_package(pkg)
    assert before
    # Editing a sibling module (NOT bridge.py) must change the package id.
    (pkg / "taint_engine.py").write_text("X = 2\n", encoding="utf-8")
    after = build_id_for_package(pkg)
    assert after and after != before


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


@pytest.mark.parametrize(
    "cmd,op,total_label",
    [
        ("strings", "strings", "Total strings:"),
        ("imports", "imports", "Total imports:"),
        ("sections", "sections", "Total sections:"),
        ("types", "types", "Total types:"),
    ],
)
def test_list_count_flag_forwards_count_only(fake_transport, capsys, cmd, op, total_label):
    # #165: --count on strings/imports/sections/types forwards count_only and
    # renders the total (mirrors `function list --count`).
    calls = fake_transport(default={"ok": True, "result": {"count": 42, "total": 42}})
    rc = bn.cli.main([cmd, "--count", "--target", "active"])
    assert rc == 0
    assert calls[-1]["op"] == op
    assert (calls[-1]["params"] or {}).get("count_only") is True
    assert total_label in capsys.readouterr().out


def test_effective_limit_uncaps_for_out_export():
    # #165: default page limit is 100, but --out uncaps it (full-body export);
    # an explicit --limit always wins.
    import argparse as _argparse
    from bn.cli import _effective_limit
    assert _effective_limit(_argparse.Namespace(limit=None, out=None)) == 100
    assert _effective_limit(_argparse.Namespace(limit=None, out="fns.json")) is None
    assert _effective_limit(_argparse.Namespace(limit=25, out="fns.json")) == 25
    assert _effective_limit(_argparse.Namespace(limit=25, out=None)) == 25


def test_duplicate_command_path_registration_raises():
    bn.cli.build_parser()  # ensure command modules populated the registry

    with pytest.raises(ValueError, match="duplicate command path"):
        bn.cli.command("xrefs")(lambda args: 0)


def test_paged_callsites_spill_hint_suggests_limit_offset(monkeypatch, capsys):
    # #454: callsites is now a paged command, so its spill hint suggests
    # --limit/--offset (the unpaged --out-hint branch is covered by types show above).
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
    assert "--limit/--offset" in stderr
    # the path is named once (in "spilled to <path>"); the hint must not repeat it
    assert "spilled to /tmp/callsites.txt" in stderr
    assert stderr.count("/tmp/callsites.txt") == 1


def test_negative_ip_depth_rejected_with_exit_2():
    parser = bn.cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["trace", "f", "0x1000", "--target", "active", "--ip-depth", "-1"])
    assert excinfo.value.code == 2
    # ip-depth 0 (disable crossing) is allowed
    ns = parser.parse_args(["trace", "f", "0x1000", "--target", "active", "--ip-depth", "0"])
    assert ns.ip_depth == 0


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


def test_spill_emits_greppable_marker_on_piped_text(monkeypatch, capsys):
    """A piped TEXT spill prepends a loud __BN_SPILLED__ marker as the FIRST
    stdout line, so a downstream grep/awk can't mistake a no-match for absence
    (#216) -- the stderr note alone is invisible to the pipe consumer."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result",
                        lambda value, *, fmt, out_path, stem: _spill_artifact_namespace("/tmp/decompile.txt"))
    monkeypatch.setattr(bn.cli, "_stdout_is_pipe", lambda: True, raising=False)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert stdout.splitlines()[0] == "__BN_SPILLED__ /tmp/decompile.txt"


def test_spill_json_pipe_has_no_marker(monkeypatch, capsys):
    """A piped JSON spill must NOT get the text marker -- it would corrupt the
    jq-parseable envelope, whose spilled:true field is already machine-checkable
    (#216)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result",
                        lambda value, *, fmt, out_path, stem: _spill_artifact_namespace("/tmp/decompile.txt"))
    monkeypatch.setattr(bn.cli, "_stdout_is_pipe", lambda: True, raising=False)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active", "--format", "json"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert "__BN_SPILLED__" not in stdout


def test_spill_no_marker_when_not_piped(monkeypatch, capsys):
    """A non-pipe (terminal / capture) text spill stays quiet -- no marker line
    pollutes interactive output (#216)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"text": "long decompiled text"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "write_output_result",
                        lambda value, *, fmt, out_path, stem: _spill_artifact_namespace("/tmp/decompile.txt"))
    monkeypatch.setattr(bn.cli, "_stdout_is_pipe", lambda: False, raising=False)

    rc = bn.cli.main(["decompile", "sub_401000", "--target", "active"])
    assert rc == 0
    stdout, _ = capsys.readouterr()
    assert "__BN_SPILLED__" not in stdout




def test_decompile_lines_out_of_range_exits_nonzero(monkeypatch, capsys):
    # End-to-end: an out-of-range --lines start exits non-zero with a stderr
    # diagnostic and NO code-like stdout, so a scripted consumer can tell it
    # apart from a real slice (#253).
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "decompile":
            return {"ok": True, "result": {"text": "int main() {\n  return 0;\n}"}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["decompile", "main", "--target", "active", "--lines", "99999:100000"])

    assert rc == 2
    out, err = capsys.readouterr()
    assert out.strip() == ""           # nothing that looks like decompiled output
    assert "beyond the last line" in err
    assert "Traceback" not in err

def test_slice_text_lines_in_range_still_slices():
    # In-range slicing (incl. end past the last line, which clamps) is unchanged.
    from bn import formatters

    assert formatters._slice_text_lines("a\nb\nc", (2, 99)) == "// lines 2-3 of 3\nb\nc"

def test_slice_text_lines_start_beyond_end_raises(monkeypatch):
    # #253: an out-of-range --lines start is a user error, not a result. It must
    # NOT return a `//` line (mistakable for code) with exit 0 -- it raises a
    # BridgeError so the CLI exits non-zero with a stderr diagnostic.
    from bn import formatters
    from bn.transport import BridgeError

    with pytest.raises(BridgeError, match="beyond the last line"):
        formatters._slice_text_lines("a\nb\nc", (10, 12))

def test_instance_accepted_before_two_level_subcommand():
    # --instance is likewise carried by intermediate group parsers (#251).
    parser = bn.cli.build_parser()
    ns = parser.parse_args(["bundle", "--instance", "inst9", "function", "_init"])
    assert ns.instance == "inst9"

def test_target_accepted_before_two_level_subcommand():
    # #251: -t/--target works BEFORE the leaf of a two-level command (after the
    # group name), not only after the leaf -- parity with single-level commands
    # and root-level -t. The intermediate group parser must carry -t.
    parser = bn.cli.build_parser()

    pre = parser.parse_args(["bundle", "-t", "mytarget", "function", "_init"])
    assert pre.target == "mytarget"
    assert pre.identifier == "_init"

    # Post-leaf form must still resolve identically (SUPPRESS default = no clobber).
    post = parser.parse_args(["bundle", "function", "_init", "-t", "mytarget"])
    assert post.target == "mytarget"
    assert post.identifier == "_init"

def test_target_before_three_level_subcommand():
    # Three-level commands (struct field set) accept -t at any intermediate level.
    parser = bn.cli.build_parser()
    ns = parser.parse_args(
        ["struct", "-t", "tgt", "field", "set", "MyStruct", "0", "field0", "int32_t"])
    assert ns.target == "tgt"


# --- #315: infer --format from the --out extension ---

from pathlib import Path as _Path
from bn.cli import _resolve_output_format as _rof


def _fmt_ns(**kw):
    base = {"format": "text", "out": None, "_format_explicit": False}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_resolve_output_format_infers_json_from_extension(capsys):
    assert _rof(_fmt_ns(out=_Path("x.json"))) == "json"
    assert "inferring --format json" in capsys.readouterr().err


def test_resolve_output_format_infers_ndjson_from_extension(capsys):
    assert _rof(_fmt_ns(out=_Path("x.ndjson"))) == "ndjson"
    assert "inferring --format ndjson" in capsys.readouterr().err


def test_resolve_output_format_uppercase_extension(capsys):
    assert _rof(_fmt_ns(out=_Path("REPORT.JSON"))) == "json"


def test_resolve_output_format_no_inference_for_plain_extension(capsys):
    assert _rof(_fmt_ns(out=_Path("x.txt"))) == "text"
    assert capsys.readouterr().err == ""


def test_resolve_output_format_no_out_is_unchanged(capsys):
    assert _rof(_fmt_ns(out=None)) == "text"
    assert capsys.readouterr().err == ""


def test_resolve_output_format_explicit_format_wins_with_warning(capsys):
    # An explicit --format text to a .json file is honored, but warned about.
    assert _rof(_fmt_ns(out=_Path("x.json"), format="text", _format_explicit=True)) == "text"
    err = capsys.readouterr().err.lower()
    assert "warning" in err and ".json" in err


def test_resolve_output_format_matching_explicit_json_is_silent(capsys):
    assert _rof(_fmt_ns(out=_Path("x.json"), format="json", _format_explicit=True)) == "json"
    assert capsys.readouterr().err == ""


def test_out_json_extension_writes_valid_json_end_to_end(fake_transport, tmp_path, capsys):
    # The real footgun: `--out x.json` without --format json used to write the
    # human TEXT renderer into the .json file, breaking a downstream json.load.
    fake_transport({"list_functions": {"ok": True, "result": {
        "kind": "functions",
        "items": [{"address": "0x1000", "name": "f", "size": 10}],
        "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False,
    }}})
    out = tmp_path / "fns.json"
    rc = bn.cli.main(["function", "list", "--target", "active", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())  # must parse as JSON, not text
    assert data["items"][0]["name"] == "f"


def test_out_json_with_explicit_text_writes_text_and_warns(fake_transport, tmp_path, capsys):
    fake_transport({"list_functions": {"ok": True, "result": {
        "kind": "functions",
        "items": [{"address": "0x1000", "name": "f", "size": 10}],
        "total": 1, "offset": 0, "limit": 100, "returned": 1, "has_more": False,
    }}})
    out = tmp_path / "fns.json"
    rc = bn.cli.main(["function", "list", "--target", "active", "--out", str(out), "--format", "text"])
    assert rc == 0
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.read_text())  # honored explicit text
    assert "warning" in capsys.readouterr().err.lower()


def test_fanout_all_instances_aggregates_and_isolates_errors(monkeypatch, capsys):
    # #169 L1: --all-instances runs the read across every instance and aggregates;
    # a per-instance failure (e.g. ambiguous target) is an ok:false row, not a
    # hard failure of the whole command.
    import json as _json
    import types as _types
    import bn.cli as cli
    from bn.transport import BridgeError

    insts = [_types.SimpleNamespace(instance_id="a"), _types.SimpleNamespace(instance_id="b")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")
    seen = []

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        seen.append(instance_id)
        if instance_id == "b":
            raise BridgeError("multiple targets open")
        return {"result": {"kind": "functions", "items": [{"name": "f", "address": "0x1"}],
                           "total": 1, "count": 1}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["function", "list", "--all-instances", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out["kind"] == "fanout" and out["count"] == 2
    by = {r["instance"]: r for r in out["instances"]}
    assert by["a"]["ok"] is True and by["a"]["result"]["total"] == 1
    assert by["b"]["ok"] is False and "multiple targets" in by["b"]["error"]
    assert set(seen) == {"a", "b"}


def test_fanout_all_instances_text_renders_per_instance(monkeypatch, capsys):
    # #169 L1: text mode renders each instance with the command's own renderer
    # and an error line for failed instances.
    import types as _types
    import bn.cli as cli
    from bn.transport import BridgeError

    insts = [_types.SimpleNamespace(instance_id="a"), _types.SimpleNamespace(instance_id="b")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if instance_id == "b":
            raise BridgeError("no targets open")
        return {"result": {"kind": "functions", "items": [{"name": "alpha", "address": "0x1000"}],
                           "total": 1, "count": 1}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["function", "list", "--all-instances"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fan-out:" in out and "== instance a" in out and "== instance b" in out
    # Assert the function-list RENDERER ran (its `0xADDR  name` row form), not just
    # that "alpha" appears (the JSON fallback would also contain it) -- real teeth
    # on the "command's own renderer" claim (#169 L1 review).
    assert "0x1000  alpha" in out
    assert "error: no targets open" in out      # per-instance failure surfaced


def test_fanout_all_instances_exits_nonzero_when_all_fail(monkeypatch, capsys):
    # #169 L1 review: a fan-out where EVERY instance fails must exit non-zero, so a
    # scripted consumer doesn't read total failure as success.
    import types as _types
    import bn.cli as cli
    from bn.transport import BridgeError
    insts = [_types.SimpleNamespace(instance_id="a"), _types.SimpleNamespace(instance_id="b")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")
    def fail_send(op, *, params=None, target=None, instance_id=None, **k):
        raise BridgeError("down")
    monkeypatch.setattr(cli, "send_request", fail_send)
    rc = cli.main(["function", "list", "--all-instances", "--format", "json"])
    assert rc == 2   # all failed -> non-zero


def test_fanout_flag_not_on_write_or_per_function_commands():
    # #169 L1 review (CRITICAL guard): --all-instances must NOT be attached to any
    # write/side-effecting command (it could fan a write across every instance) or
    # to per-function reads (their identifier wouldn't resolve elsewhere).
    import argparse, contextlib, io
    import bn.cli as cli
    for argv in (["save", "--help"], ["close", "--help"], ["refresh", "--help"],
                 ["py", "exec", "--help"], ["decompile", "--help"], ["xrefs", "--help"],
                 ["rename", "--help"]):
        buf = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            cli.main(argv)
        assert "--all-instances" not in buf.getvalue(), f"{argv[0]} must not be fannable"


def test_fanout_no_instances_is_clean_error(monkeypatch):
    import bn.cli as cli
    monkeypatch.setattr(cli, "list_instances", lambda: [])
    rc = cli.main(["function", "list", "--all-instances"])
    assert rc == 2   # BridgeError -> exit 2


def test_fanout_all_targets_within_instance(monkeypatch, capsys):
    # #169 L1: --all-targets fans every open target IN one instance (no
    # --all-instances), one row per target, via list_targets.
    import json as _json
    import bn.cli as cli

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            # production list_targets returns a BARE list (no {"items":...} envelope)
            return {"result": [{"target_id": "t1"}, {"target_id": "t2"}]}
        return {"result": {"kind": "sections", "items": [{"name": ".text"}], "total": 1,
                           "_t": target}}
    monkeypatch.setattr(cli, "send_request", fake_send)
    # no --all-instances -> single (default) instance, fan its targets
    rc = cli.main(["sections", "--all-targets", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out["kind"] == "fanout" and out["count"] == 2
    seen_targets = {r["target"] for r in out["instances"]}
    assert seen_targets == {"t1", "t2"}
    assert all(r["ok"] for r in out["instances"])


def test_fanout_all_targets_no_targets_open_is_row_not_crash(monkeypatch, capsys):
    import json as _json
    import bn.cli as cli

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            return {"result": []}                # instance with nothing open (bare list, production shape)
        raise AssertionError("op should not run when no targets")
    monkeypatch.setattr(cli, "send_request", fake_send)
    rc = cli.main(["imports", "--all-targets", "--format", "json"])
    assert rc == 2   # the only "instance" produced an ok:false row -> all failed
    out = _json.loads(capsys.readouterr().out)
    assert out["instances"][0]["ok"] is False and "no targets" in out["instances"][0]["error"]


def test_fanout_instances_x_targets_matrix(monkeypatch, capsys):
    # #169 L1: --all-instances --all-targets fans every (instance, target) pair.
    import json as _json
    import types as _types
    import bn.cli as cli
    insts = [_types.SimpleNamespace(instance_id="A"), _types.SimpleNamespace(instance_id="B")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            return {"result": [{"target_id": instance_id + "-t1"},
                                {"target_id": instance_id + "-t2"}]}
        return {"result": {"kind": "sections", "items": [], "total": 0}}
    monkeypatch.setattr(cli, "send_request", fake_send)
    rc = cli.main(["sections", "--all-instances", "--all-targets", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out["count"] == 4   # 2 instances x 2 targets
    pairs = {(r["instance"], r["target"]) for r in out["instances"]}
    assert pairs == {("A", "A-t1"), ("A", "A-t2"), ("B", "B-t1"), ("B", "B-t2")}


def test_fanout_all_instances_auto_surveys_multi_target_instance(monkeypatch, capsys):
    # #368 facet 1: --all-instances (no --all-targets) must NOT drop a multi-target
    # instance to an "ambiguous target" error row -- it surveys ALL of its targets,
    # so coverage is complete, and discloses the expansion.
    import json as _json
    import types as _types
    import bn.cli as cli
    insts = [_types.SimpleNamespace(instance_id="solo"), _types.SimpleNamespace(instance_id="multi")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            if instance_id == "multi":
                return {"result": [{"target_id": "m-t1"}, {"target_id": "m-t2"}]}
            return {"result": [{"target_id": "solo-t1"}]}   # single target
        return {"result": {"kind": "sections", "items": [], "total": 0, "_t": target}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["sections", "--all-instances", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    # solo -> 1 row (its single peeked target id reused directly), multi -> 2 rows
    # (both targets surveyed).
    assert out["count"] == 3
    pairs = {(r["instance"], r.get("target")) for r in out["instances"]}
    # The single-target peek path reuses the peeked id directly (no _resolve_target
    # for it) -- assert the concrete id so that path actually has teeth.
    assert ("solo", "solo-t1") in pairs
    assert ("multi", "m-t1") in pairs and ("multi", "m-t2") in pairs
    assert out["auto_expanded_instances"] == ["multi"]


def test_fanout_reports_per_row_duration_and_slow_rows(monkeypatch, capsys):
    # #417: a fan-out runs per-instance reads concurrently and reports per-row
    # duration_ms plus a top-level slow_rows summary, so an agent can see WHERE a
    # broad survey spent its time (and a slow instance does not serialize the rest).
    import json as _json
    import time as _time
    import types as _types
    import bn.cli as cli
    from bn.transport import BridgeError

    insts = [_types.SimpleNamespace(instance_id=x) for x in ("a", "b", "c")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if instance_id == "b":
            _time.sleep(0.05)  # the slow instance
        if instance_id == "c":
            raise BridgeError("no targets open")  # error rows still isolated + timed
        return {"result": {"kind": "functions", "items": [], "total": 0, "count": 0}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["function", "list", "--all-instances", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out["count"] == 3
    by = {r["instance"]: r for r in out["instances"]}
    # every row (including the error row) carries a duration.
    assert all(isinstance(by[i]["duration_ms"], int | float) for i in ("a", "b", "c"))
    assert by["c"]["ok"] is False  # per-row error isolation preserved
    # slow_rows surfaces the slowest instance ('b' slept) first.
    assert out["slow_rows"][0]["instance"] == "b"
    assert by["b"]["duration_ms"] >= 50  # ~the 50ms sleep


def test_fanout_preserves_instance_order_with_enumeration_error(monkeypatch, capsys):
    # #417 review: an enumeration error (list_targets fails for one instance) must
    # keep its slot so the rows stay in instance order, even though the successful
    # reads complete out of order under the concurrent pool.
    import json as _json
    import types as _types
    import bn.cli as cli
    from bn.transport import BridgeError

    insts = [_types.SimpleNamespace(instance_id=x) for x in ("a", "b", "c")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            if instance_id == "b":
                raise BridgeError("down")  # enumeration fails for b
            return {"result": [{"target_id": instance_id + "-t"}]}
        return {"result": {"kind": "sections", "items": [], "total": 0}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["sections", "--all-instances", "--all-targets", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    order = [r["instance"] for r in out["instances"]]
    assert order == ["a", "b", "c"]           # b's enumeration error kept its slot
    by = {r["instance"]: r for r in out["instances"]}
    assert by["b"]["ok"] is False and "down" in by["b"]["error"]
    assert by["a"]["ok"] is True and by["c"]["ok"] is True


def test_fanout_all_instances_auto_surveys_despite_sticky_target_pin(monkeypatch, capsys):
    # #368 review (HIGH): a STICKY target pin must NOT count as an explicit -t.
    # _apply_sticky_defaults fills args.target from session state and marks it
    # _sticky_target=True; that must still let --all-instances auto-survey a
    # multi-target instance (only a -t passed on the CLI applies to every instance).
    import json as _json
    import types as _types
    import bn.cli as cli
    insts = [_types.SimpleNamespace(instance_id="solo"), _types.SimpleNamespace(instance_id="multi")]
    monkeypatch.setattr(cli, "list_instances", lambda: insts)
    monkeypatch.setattr(cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(cli, "_resolve_target", lambda args, **k: "active")
    # A sticky target pin is present -> _apply_sticky_defaults sets args.target +
    # args._sticky_target=True before the fan-out handler runs.
    monkeypatch.setattr(cli.session_state, "read", lambda: {"target": "pinned_tgt"})

    def fake_send(op, *, params=None, target=None, instance_id=None, **k):
        if op == "list_targets":
            if instance_id == "multi":
                return {"result": [{"target_id": "m-t1"}, {"target_id": "m-t2"}]}
            return {"result": [{"target_id": "solo-t1"}]}
        return {"result": {"kind": "sections", "items": [], "total": 0, "_t": target}}
    monkeypatch.setattr(cli, "send_request", fake_send)

    rc = cli.main(["sections", "--all-instances", "--format", "json"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    # The multi-target instance is still surveyed in full despite the sticky pin.
    assert out["count"] == 3
    pairs = {(r["instance"], r.get("target")) for r in out["instances"]}
    assert ("multi", "m-t1") in pairs and ("multi", "m-t2") in pairs
    assert out["auto_expanded_instances"] == ["multi"]
    # The sticky pin was NOT applied to every instance.
    assert ("multi", "pinned_tgt") not in pairs


def test_all_targets_flag_only_on_fanout_commands():
    # --all-targets, like --all-instances, is allow-listed (not on writes/per-fn).
    import contextlib, io
    import bn.cli as cli
    for argv, present in (
        (["imports", "--help"], True),
        (["sections", "--help"], True),
        (["save", "--help"], False),
        (["decompile", "--help"], False),
        (["rename", "--help"], False),
    ):
        buf = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            cli.main(argv)
        assert ("--all-targets" in buf.getvalue()) is present, argv
