from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_symbol_rename_builds_preview_payload(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
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


def test_symbol_rename_uses_implicit_target_when_single_target_is_open(monkeypatch):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
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
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        if op == "list_targets":
            return {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                        "active": True,
                    },
                    {"target_id": "123:2:8", "selector": "other.exe.bndb", "active": False},
                ],
            }
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "sub_401000", "player_update"])

    assert rc == 2
    assert capsys.readouterr().err == (
        "This command requires --target when multiple targets are open.\n"
        "Open targets:\n"
        "- SnailMail_unwrapped.exe.bndb [active] (target_id: 123:1:7)\n"
        "- other.exe.bndb (target_id: 123:2:8)\n"
    )


def test_function_create_builds_payload_and_defaults_to_json(monkeypatch, capsys):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "message": "Function created and verified in the live Binary Ninja session.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verified",
                        "address": "0x401000",
                        "function": "sub_401000",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "0x401000"])

    assert rc == 0
    assert captured["op"] == "function_create"
    assert captured["target"] == "123:1:7"
    assert captured["params"] == {"address": "0x401000", "preview": False}
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "verified"


def test_function_create_text_output_renders_verified_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "message": "Function created and verified in the live Binary Ninja session.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verified",
                        "address": "0x401000",
                        "function": "sub_401000",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--format", "text", "0x401000"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "function_create 0x401000 (sub_401000) [verified]" in out


def test_function_create_forwards_preview_flag(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--preview", "0x401000"])

    assert rc == 0
    assert captured["params"]["preview"] is True


def test_function_create_verification_failure_exits_three(monkeypatch):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": False,
                "committed": False,
                "message": "Rolled back because no function was created at the address.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "verification_failed",
                        "address": "0x401000",
                        "message": "No function starts at 0x401000 after analysis.",
                        "requested": {"op": "function_create", "address": "0x401000"},
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "0x401000"])

    assert rc == 3


def test_render_target_line_shows_symbol_and_string_for_mapped_targets():
    # ILX #4: mapped non-function targets should surface symbol/string + section,
    # not just bare hex.
    from bn import formatters

    vtable = {
        "raw": "0x3f418",
        "normalized": "0x3f418",
        "function": None,
        "status": "mapped",
        "context": {
            "symbol": {"name": "_ZTVN17service_framework7IPCBoolE", "type": "ExternalSymbol"},
            "sections": [{"name": ".extern"}],
        },
    }
    line = formatters._render_target_line(vtable)
    assert "_ZTVN17service_framework7IPCBoolE @ 0x3f418" in line
    assert "[.extern, ExternalSymbol]" in line

    rodata_string = {
        "raw": "0x2a407",
        "normalized": "0x2a407",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {"value": "N19androidauto_service17AndroidAutoClientE", "encoding": "ascii"},
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(rodata_string)
    assert '"N19androidauto_service17AndroidAutoClientE"' in line
    assert "[.rodata]" in line

    truncated_string = {
        "raw": "0x427840",
        "normalized": "0x427840",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {
                "value": "Usage: %s [OPTION]...\n" + ("A" * 16),
                "encoding": "ascii",
                "truncated": True,
            },
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(truncated_string)
    assert '"Usage: %s [OPTION]...\\nAAAAAAAAAAAAAAAA"' in line
    assert "[.rodata, truncated]" in line


def test_callsites_within_file_ignores_comments_and_blank_lines(monkeypatch, tmp_path):
    captured = {}
    scope_file = tmp_path / "functions.txt"
    scope_file.write_text(
        "\n# curated trial functions\nbonus_pick_random_type\n\nfx_queue_add_random\n",
        encoding="utf-8",
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        captured["target"] = target
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "callsites",
            "--target",
            "active",
            "--within-file",
            str(scope_file),
            "crt_rand",
        ]
    )

    assert rc == 0
    assert captured["op"] == "callsites"
    assert captured["params"]["within_identifiers"] == [
        "bonus_pick_random_type",
        "fx_queue_add_random",
    ]


def test_comment_get_uses_implicit_target_when_single_target_is_open(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append({"op": op, "params": params, "target": target})
        if op == "list_targets":
            return {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            }
        if op == "get_comment":
            return {"ok": True, "result": {"address": "0x401000", "comment": "interesting branch", "has_comment": True}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--address", "0x401000"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "get_comment"]
    assert calls[1]["target"] == "active"
    assert capsys.readouterr().out == "interesting branch\n"


def test_xrefs_hints_struct_field_on_small_offset_zero_match(monkeypatch, capsys):
    """`xrefs 0x308` with 0 matches: 0x308 looks like a struct-field offset
    misread as an absolute address. Nudge toward --field."""
    monkeypatch.setattr(bn.cli, "send_request", _empty_xrefs)
    rc = bn.cli.main(["xrefs", "0x308", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--field" in err


def test_symbol_rename_text_format_renders_mutation_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                    }
                ],
                "affected_functions": [
                    {
                        "address": "0x401000",
                        "before_name": "sub_401000",
                        "after_name": "player_update",
                        "changed": True,
                        "diff": "--- before:sub_401000\n+++ after:player_update",
                    }
                ],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "symbol",
            "rename",
            "--format",
            "text",
            "--target",
            "active",
            "--preview",
            "sub_401000",
            "player_update",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "preview: change applied + reverted" in output
    assert "rename_symbol function 0x401000 -> player_update" in output
    assert "0x401000 sub_401000 -> player_update" in output
    assert '"results"' not in output


def test_symbol_rename_verification_failure_returns_nonzero(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": False,
                "committed": False,
                "message": "Rolled back because live-session verification failed.",
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "verification_failed",
                        "message": "Live rename verification failed at 0x401000",
                        "requested": {
                            "identifier": "sub_401000",
                            "kind": "function",
                            "new_name": "player_update",
                        },
                        "observed": {
                            "address": "0x401000",
                            "name": "sub_401000",
                        },
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--format", "text", "--target", "active", "sub_401000", "player_update"])

    assert rc == 3
    output = capsys.readouterr().out
    assert "rolled back" in output
    assert "failed: rename_symbol" in output
    assert "[verification_failed]" in output
    assert 'requested: {"identifier": "sub_401000"' in output
    assert 'observed: {"address": "0x401000", "name": "sub_401000"}' in output


def test_symbol_rename_noop_still_succeeds(monkeypatch):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "noop",
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "player_update", "player_update"])

    assert rc == 0


def test_comment_get_empty_comment_shows_placeholder(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "get_comment"
        return {"ok": True, "result": {"address": "0x401000", "comment": "", "has_comment": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--target", "active", "--address", "0x401000"])

    assert rc == 0
    assert capsys.readouterr().out == "(no comment)\n"


def test_batch_apply_stdin_forwards_preview_flag(monkeypatch):
    import io

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"ops": [{"op": "set_comment", "address": "0x1000", "comment": "x"}]}'),
    )
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["params"] = params
        return {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["batch", "apply", "--preview", "-"])

    assert rc == 0
    assert captured["params"]["preview"] is True


def test_rename_alias_maps_to_symbol_rename(monkeypatch):
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"preview": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["rename", "--target", "123:1:7", "--preview", "sub_401000", "player_update"])

    assert rc == 0
    assert captured["op"] == "rename_symbol"
    assert captured["params"]["identifier"] == "sub_401000"
    assert captured["params"]["new_name"] == "player_update"


def test_render_mutation_text_does_not_claim_rollback_when_revert_failed():
    """When a mutation failed AND its revert failed (rolled_back=False), the
    text renderer must not print 'rolled back' -- that contradicts the honest
    'view may be left modified' message and re-states the #117 symptom (#117)."""
    from bn import formatters
    value = {
        "preview": True,
        "success": False,
        "committed": False,
        "rolled_back": False,
        "message": "Preview verified, but removing the created function on revert failed; the view may be left modified.",
        "results": [{"op": "function_create", "status": "rollback_failed", "address": "0x1000", "function": "sub_1000"}],
        "affected_functions": [],
        "affected_types": [],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" not in out
    assert "rollback failed" in out
    assert "may be left modified" in out
    # the op renders under 'failed:', not as a bare '[verified]'
    assert "failed: " in out
    assert "[verified]" not in out


def test_render_mutation_text_still_reports_clean_rollback():
    """A failed batch that WAS cleanly reverted still says 'rolled back'."""
    from bn import formatters
    value = {
        "preview": False,
        "success": False,
        "committed": False,
        "rolled_back": True,
        "message": "Rolled back because live-session verification failed.",
        "results": [{"op": "rename_symbol", "status": "verification_failed", "address": "0x1000"}],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" in out


def test_unknown_ref_label_prefers_symbol_then_section():
    from bn import formatters
    assert formatters._unknown_ref_label({"symbol": {"name": "some_export"}}) == "some_export"
    assert formatters._unknown_ref_label({"sections": [{"name": ".got"}]}) == ".got"
    assert formatters._unknown_ref_label({"symbol": {"name": "s"}, "sections": [{"name": ".got"}]}) == "s"
    assert formatters._unknown_ref_label({}) == ""
    assert formatters._unknown_ref_label(None) == ""


def test_xrefs_data_ref_labels_unknown_caller_by_section_or_symbol():
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0x1a254", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".got"}], "symbol": {"name": "some_export"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    assert "some_export" in out          # symbol preferred over a bare <unknown>
    assert "<unknown>  <unknown>" not in out


# ---------------------------------------------------------------------------
# Batch 5: CLI validation/rendering (#94, #96, #100, #101, #102)
# ---------------------------------------------------------------------------


def test_comment_get_rejects_both_address_and_function(capsys):
    # #94: --address and --function are mutually exclusive (the bridge checks
    # function first, so accepting both silently dropped the address).
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["comment", "get", "--target", "active", "--address", "0x1000", "--function", "main"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_comment_get_requires_a_locator(capsys):
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["comment", "get", "--target", "active"])
    assert exc.value.code == 2  # required mutex group: exactly one
