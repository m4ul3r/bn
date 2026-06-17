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


def test_local_list_text_is_slim(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
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
    # local_id IS shown in text mode: it is the stable handle that `local rename`
    # / `local retype` take, so omitting it forced a --format json round-trip to
    # drive those commands (#122). The other internal fields stay out of the
    # slim text view.
    assert "0x401000:param:stack:4:0:1" in output
    assert "0x401000:local:stack:-12:0:2" in output
    assert "storage=" not in output
    assert "source=" not in output
    assert "identifier=" not in output


def test_local_list_json_retains_ids(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
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
def test_list_count_flag_forwards_count_only(monkeypatch, capsys, cmd, op, total_label):
    # #165: --count on strings/imports/sections/types forwards count_only and
    # renders the total (mirrors `function list --count`).
    captured = {}

    def fake_send_request(o, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = o
        captured["params"] = params or {}
        return {"ok": True, "result": {"count": 42, "total": 42}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main([cmd, "--count", "--target", "active"])
    assert rc == 0
    assert captured["op"] == op
    assert captured["params"].get("count_only") is True
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


def test_unpaged_list_spill_hint_points_at_out_flag(monkeypatch, capsys):
    # callsites returns a list but is not a paged command: the spill hint must
    # not suggest --limit/--offset (those flags do not exist on `bn callsites`).
    # (imports/sections used to be the example here, but they are paged now.)
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
    assert "--limit/--offset" not in stderr
    # the path is named once (in "spilled to <path>"); the hint must not repeat it
    assert "spilled to /tmp/callsites.txt" in stderr
    assert stderr.count("/tmp/callsites.txt") == 1
    assert "rerun with --out <path>" in stderr


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
