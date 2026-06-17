from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_target_flag_accepted_before_subcommand():
    parser = bn.cli.build_parser()

    # Names with dots, names that collide with subcommand strings, and
    # interleaving with --instance must all parse with -t before the subcommand.
    cases = [
        (["-t", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", None),
        (["--target", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", None),
        (["-t", "session", "function", "list"], "session", None),
        (["-t", "function", "function", "list"], "function", None),
        (["--instance", "X", "-t", "pam_qnx.so.2", "function", "list"], "pam_qnx.so.2", "X"),
        (["-t", "pam_qnx.so.2", "--instance", "X", "function", "list"], "pam_qnx.so.2", "X"),
    ]
    for argv, expected_target, expected_instance in cases:
        args = parser.parse_args(argv)
        assert args.target == expected_target, argv
        assert args.instance == expected_instance, argv


def test_target_flag_after_subcommand_still_works():
    parser = bn.cli.build_parser()

    # The pre-existing form (target after subcommand) must keep working.
    args = parser.parse_args(["function", "list", "-t", "pam_qnx.so.2"])
    assert args.target == "pam_qnx.so.2"


def test_target_flag_root_does_not_clobber_subparser_value():
    parser = bn.cli.build_parser()

    # Root-level -t followed by a subparser-level -t: the later one wins
    # (argparse default), and neither None nor SUPPRESS leaks through.
    args = parser.parse_args(["-t", "first", "function", "list", "-t", "second"])
    assert args.target == "second"


def test_target_list_text_format_renders_summary(fake_transport, capsys):
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {
                    "selector": "SnailMail_unwrapped.exe.bndb",
                    "target_id": "123:1:7",
                    "view_id": "1",
                    "view_name": "PE",
                    "filename": "/tmp/SnailMail_unwrapped.exe.bndb",
                    "active": True,
                }
            ],
        }
    })

    rc = bn.cli.main(["target", "list", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "SnailMail_unwrapped.exe.bndb [active]" in output
    assert "target: 123:1:7" in output
    assert '"selector"' not in output


def test_refresh_uses_implicit_target_when_single_target_is_open(fake_transport, capsys):
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
        },
        "refresh": {
            "ok": True,
            "result": {
                "refreshed": True,
                "target": {
                    "selector": "SnailMail_unwrapped.exe.bndb",
                    "target_id": "123:1:7",
                    "view_id": "1",
                    "view_name": "PE",
                    "filename": "/tmp/SnailMail_unwrapped.exe.bndb",
                    "active": True,
                },
            },
        },
    })

    rc = bn.cli.main(["refresh", "--format", "text"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "refresh"]
    assert calls[1]["target"] == "active"
    output = capsys.readouterr().out
    assert "refreshed: true" in output
    assert "SnailMail_unwrapped.exe.bndb" in output


def test_target_summary_text_marks_quick_view():
    """A --quick (unanalyzed) target must be flagged in text output, not just JSON."""
    from bn import formatters
    quick = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": False, "analysis_state": "quick",
    })
    assert "[not analyzed]" in quick
    full = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": True, "analysis_state": "full",
    })
    assert "[not analyzed]" not in full


def test_target_info_verbose_renders_segments():
    """target info --verbose text appends the segment map with r/w/x perms; the
    block is absent when no segments are present (target list rows). (F21)"""
    from bn.formatters import _render_target_summary

    out = _render_target_summary({
        "selector": "svc", "arch": "aarch64",
        "segments": [
            {"start": "0x1000", "end": "0x2000", "length": 0x1000,
             "readable": True, "writable": False, "executable": True},
        ],
    })
    assert "segments:" in out
    assert "0x1000-0x2000 r-x (4096 bytes)" in out

    out2 = _render_target_summary({"selector": "svc", "arch": "aarch64"})
    assert "segments:" not in out2


def test_save_accepts_path_flag(monkeypatch, tmp_path):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "save_database"
        captured_params.update(params or {})
        return {"ok": True, "result": {"path": params.get("path"), "saved": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    out = tmp_path / "out.bndb"
    rc = bn.cli.main(["save", "--target", "active", "--path", str(out)])

    assert rc == 0
    assert captured_params["path"] == str(out.expanduser().resolve())


def test_close_warns_on_unsaved_changes(fake_transport, capsys):
    fake_transport({
        "close_binary": {
            "ok": True,
            "result": {
                "closed": [{"path": "/tmp/foo.bndb", "unsaved": True}],
            },
        }
    })

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "closed: /tmp/foo.bndb" in output
    assert "unsaved" in output.lower()
    assert "bn save" in output


def test_close_silent_when_clean(fake_transport, capsys):
    fake_transport({
        "close_binary": {
            "ok": True,
            "result": {
                "closed": [{"path": "/tmp/foo.bndb", "unsaved": False}],
            },
        }
    })

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "closed: /tmp/foo.bndb" in output
    assert "warning" not in output.lower()
    assert "unsaved" not in output.lower()


def test_close_forwards_explicit_target_selector(fake_transport, monkeypatch, capsys):
    calls = fake_transport({
        "close_binary": {"ok": True, "result": {"closed": [{"path": "/tmp/foo", "unsaved": False}]}}
    })
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "-t", "foo", "--format", "text"])

    assert rc == 0
    assert calls[-1]["target"] == "foo"


def test_close_all_flag_sets_param(fake_transport, monkeypatch, capsys):
    calls = fake_transport({"close_binary": {"ok": True, "result": {"closed": []}}})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "--all", "--format", "text"])

    assert rc == 0
    assert calls[-1]["params"].get("all") is True


def test_close_rejects_path_and_all_together(monkeypatch, capsys):
    # `bn close <path> --all` is contradictory; the bridge would let --all win
    # and close everything despite naming one file. The CLI rejects it before
    # any request is sent (#85).
    sent = {"called": False}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        sent["called"] = True
        return {"ok": True, "result": {"closed": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "/tmp/one-binary", "--all", "--format", "text"])

    assert rc == 2  # BridgeError -> exit 2
    assert sent["called"] is False  # never reached the bridge
    err = capsys.readouterr().err
    assert "not both" in err


def test_target_use_validates_and_writes_state(tmp_session, monkeypatch, capsys):
    # target use validates the selector against the open targets before pinning,
    # so a known selector is persisted (#55).
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "pam_qnx.so.2"])

    assert rc == 0
    assert bn.session_state.read()["target"] == "pam_qnx.so.2"
    assert "target: pam_qnx.so.2" in capsys.readouterr().out


def test_target_use_rejects_unknown_selector_without_pinning(tmp_session, monkeypatch, capsys):
    # A typo must NOT be persisted: an unknown selector is rejected with exit 2
    # and the pin is left unchanged, instead of poisoning every later command (#55).
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "TOTALLY-BOGUS-XYZ"])

    assert rc == 2
    assert "Unknown target selector" in capsys.readouterr().err
    assert bn.session_state.read() == {}   # pin NOT written


def test_target_use_does_not_overwrite_existing_pin_on_reject(tmp_session, monkeypatch, capsys):
    # A failed `target use` must leave a previously-valid pin intact.
    bn.session_state.update(target="pam_qnx.so.2")
    targets = [{"selector": "pam_qnx.so.2", "filename": "/x/pam_qnx.so.2"}]
    monkeypatch.setattr(bn.cli, "send_request", lambda *_a, **_kw: {"result": targets})

    rc = bn.cli.main(["target", "use", "bogus"])

    assert rc == 2
    assert bn.session_state.read()["target"] == "pam_qnx.so.2"   # unchanged


def test_target_clear_removes_state(tmp_session, capsys):
    bn.session_state.update(target="pam_qnx.so.2")

    rc = bn.cli.main(["target", "clear"])

    assert rc == 0
    assert "target" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


# --- bn load --no-bndb plumbing ---


def test_load_defaults_to_prefer_bndb(fake_transport, tmp_path):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    calls = fake_transport({
        "load_binary": {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}
    })
    rc = bn.cli.main(["load", str(raw)])

    assert rc == 0
    assert calls[-1]["params"]["prefer_bndb"] is True


def test_load_no_bndb_flag_disables_prefer_bndb(fake_transport, tmp_path):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    calls = fake_transport({
        "load_binary": {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}
    })
    rc = bn.cli.main(["load", "--no-bndb", str(raw)])

    assert rc == 0
    assert calls[-1]["params"]["prefer_bndb"] is False


def test_load_quick_flag_sets_quick_param(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--quick", str(raw)]) == 0
    assert captured["quick"] is True


def test_load_no_analysis_alias_sets_quick_param(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--no-analysis", str(raw)]) == 0
    assert captured["quick"] is True


def test_load_default_is_not_quick(monkeypatch, tmp_path):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    captured = _load_capture(monkeypatch, raw, analyzed=True)
    assert bn.cli.main(["load", str(raw)]) == 0
    assert captured["quick"] is False


def test_load_quick_output_marks_not_analyzed(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "foo.so"; raw.write_bytes(b"")
    _load_capture(monkeypatch, raw, analyzed=False)
    assert bn.cli.main(["load", "--quick", str(raw)]) == 0
    out = capsys.readouterr().out
    assert "[not analyzed]" in out
    assert "bn refresh" in out


def test_load_text_renders_notes(fake_transport, tmp_path, capsys):
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"

    fake_transport({
        "load_binary": {
            "ok": True,
            "result": {
                "loaded": True,
                "path": str(bndb),
                "requested_path": str(raw),
                "notes": [f"loaded {bndb} instead of {raw} (use --no-bndb to skip)"],
                "targets": [],
            },
        }
    })
    rc = bn.cli.main(["load", str(raw)])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert f"loaded: {bndb}" in stdout
    assert "note: loaded" in stdout
    assert "--no-bndb" in stdout


def test_load_opts_into_spawn_missing_named(monkeypatch, tmp_path):
    # `bn load --instance <new-id>` should auto-spawn that named bridge, so the
    # load handler is the one command that opts into spawn_missing_named.
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["instance_id"] = instance_id
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", str(raw), "--instance", "brandnew"])

    assert rc == 0
    assert captured["op"] == "load_binary"
    assert captured["instance_id"] == "brandnew"
    assert captured["spawn_missing_named"] is True


def test_non_load_command_does_not_spawn_missing_named(monkeypatch):
    # Read commands must not silently spawn a process for a typo'd --instance.
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["sections", "--target", "active"])

    assert rc == 0
    assert captured["spawn_missing_named"] is False
