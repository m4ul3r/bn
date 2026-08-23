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


def test_instance_short_flag_accepted():
    parser = bn.cli.build_parser()

    # -i is the short alias for --instance, symmetric with -t/--target. It must
    # parse to args.instance before and after the subcommand, at root and leaf,
    # interleaved with -t, and the long --instance must keep working.
    cases = [
        (["-i", "abc123", "function", "list"], "abc123"),
        (["function", "list", "-i", "abc123"], "abc123"),
        (["--instance", "abc123", "function", "list"], "abc123"),
        (["-t", "libfoo.so", "-i", "abc123", "function", "list"], "abc123"),
        (["-i", "abc123", "-t", "libfoo.so", "function", "list"], "abc123"),
    ]
    for argv, expected_instance in cases:
        args = parser.parse_args(argv)
        assert args.instance == expected_instance, argv


def test_instance_short_flag_does_not_collide_with_instance_id():
    parser = bn.cli.build_parser()

    # `session start --instance-id` (auto-spawn naming) and the `instance use`
    # positional both bind to args.instance_id and must be unaffected by -i.
    start = parser.parse_args(["session", "start", "/bin/ls", "--instance-id", "named"])
    assert start.instance_id == "named"

    use = parser.parse_args(["instance", "use", "pinme"])
    assert use.instance_id == "pinme"

    # -i still binds only to the global selector (args.instance), distinct from
    # --instance-id; argparse short-flag matching is exact, no abbreviation clash.
    both = parser.parse_args(["session", "start", "/bin/ls", "-i", "sel", "--instance-id", "named"])
    assert both.instance == "sel"
    assert both.instance_id == "named"


def test_instance_short_flag_shown_in_help(capsys):
    parser = bn.cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["function", "list", "--help"])
    out = capsys.readouterr().out
    # argparse formats this as either "-i, --instance …" or
    # "-i INSTANCE, --instance INSTANCE" depending on Python version.
    assert "-i" in out and "--instance" in out
    assert "BN_INSTANCE" in out


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


def test_target_summary_text_shows_analysis_state():
    """target info text surfaces analysis_state (full/quick) -- the field the
    bn-re methodology tells agents to gate their survey on, previously visible
    only in JSON (#378)."""
    from bn import formatters
    full = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": True, "analysis_state": "full",
    })
    assert "analysis: full" in full
    quick = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analyzed": False, "analysis_state": "quick",
    })
    assert "analysis: quick" in quick


def test_target_summary_text_shows_analysis_progress_when_active():
    """#321: a long `bn refresh` surfaces a live `analysis progress:` line in text so
    the analysis reads as movement, not a wedge -- but only when actively progressing
    (total > 0); an idle view omits it."""
    from bn import formatters
    active = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analysis_state": "quick",
        "analysis_progress": {"state": "AnalyzeState", "count": 1112, "total": 1939},
    })
    assert "analysis progress: AnalyzeState 1112/1939" in active
    idle = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analysis_state": "full",
        "analysis_progress": {"state": "IdleState", "count": 0, "total": 0},
    })
    assert "analysis progress:" not in idle
    # An active phase that legitimately reports 0/0 (Discovery/ExtendedAnalyze) still
    # shows the line -- just without counts -- so it doesn't flicker away mid-analysis.
    discovery = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analysis_state": "quick",
        "analysis_progress": {"state": "DiscoveryState", "count": 0, "total": 0},
    })
    assert "analysis progress: DiscoveryState" in discovery
    assert "DiscoveryState 0/0" not in discovery  # no counts when total is 0
    # Not-yet-started phase is hidden like idle.
    initial = formatters._render_target_summary({
        "selector": "active", "view_id": 1, "analysis_state": "quick",
        "analysis_progress": {"state": "InitialState", "count": 0, "total": 0},
    })
    assert "analysis progress:" not in initial


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


def test_target_info_renders_image_base():
    """#564: target info text surfaces the preferred/image base (bv.start) so a
    debugger handoff can rebase BN addresses; absent when not reported."""
    from bn.formatters import _render_target_summary

    out = _render_target_summary({
        "selector": "svc", "arch": "x86_64",
        "image_base": "0x400000", "entry_point": "0x40b180",
    })
    assert "image base: 0x400000" in out

    out2 = _render_target_summary({"selector": "svc", "arch": "x86_64"})
    assert "image base" not in out2


def test_target_info_renders_the_pointer_format():
    """target info text surfaces pointer width + byte order so a reader decoding a
    raw dump doesn't infer them from the arch name; absent when a bridge predating
    the fields doesn't report them, and never rendered from a bogus width."""
    from bn.formatters import _render_target_summary

    out = _render_target_summary({
        "selector": "svc", "arch": "ppc64",
        "address_size": 8, "endianness": "big",
    })
    assert "pointer size: 8 bytes" in out
    assert "endianness: big" in out

    # Singular, because "1 bytes" reads as a formatting bug.
    assert "pointer size: 1 byte\n" in _render_target_summary({
        "selector": "svc", "address_size": 1, "endianness": "little",
    })

    # An older bridge reports neither field: the lines are simply omitted.
    out2 = _render_target_summary({"selector": "svc", "arch": "ppc64"})
    assert "pointer size" not in out2
    assert "endianness" not in out2

    # A width outside 1..8 is a payload surprise, not a target fact -- it must not
    # render as a confident "0 bytes".
    for bogus in (0, 9, -4, "8", True, None):
        rendered = _render_target_summary({"selector": "svc", "address_size": bogus})
        assert "pointer size" not in rendered, f"address_size={bogus!r}"


def test_save_accepts_path_flag(monkeypatch, tmp_path):
    captured_params = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
        assert op == "save_database"
        captured_params.update(params or {})
        return {"ok": True, "result": {"path": params.get("path"), "saved": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    out = tmp_path / "out.bndb"
    rc = bn.cli.main(["save", "--target", "active", "--path", str(out)])

    assert rc == 0
    assert captured_params["path"] == str(out.expanduser().resolve())


_TWO_TARGETS = {
    "ok": True,
    "result": [
        {"target_id": "123:1:7", "selector": "alpha.so", "view_id": "1"},
        {"target_id": "123:2:9", "selector": "beta.so", "view_id": "2"},
    ],
}


def test_close_warns_on_unsaved_changes(fake_transport, capsys):
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "foo.bndb"}],
        },
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
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "foo.bndb"}],
        },
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


def test_close_bare_single_target_pins_peeked_target_id(fake_transport, monkeypatch, capsys):
    # #664 round 2 (B2): with ONE target open, bare `bn close` (and `-t active`,
    # the same volatile literal spelled explicitly) peeks list_targets and then
    # sends the target_id it OBSERVED -- never the literal "active", which the
    # bridge would re-resolve at close time. If the open target changes between
    # the peek and the close (concurrent close/load), the pinned id matches
    # nothing and the bridge returns a safe unknown-selector error instead of
    # closing a different binary.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    for argv in (["close"], ["close", "-t", "active"]):
        calls = fake_transport({
            "list_targets": {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "foo.bndb", "view_id": "1"}],
            },
            "close_binary": {"ok": True, "result": {"closed": [{"path": "/tmp/foo", "unsaved": False}]}},
        })

        rc = bn.cli.main([*argv, "--format", "text"])

        assert rc == 0, argv
        assert [c["op"] for c in calls] == ["list_targets", "close_binary"], argv
        assert calls[-1]["target"] == "123:1:7", argv
        assert calls[-1]["params"] == {}, argv
        assert "closed: /tmp/foo" in capsys.readouterr().out


def test_close_empty_selector_errors_without_closing(fake_transport, monkeypatch, capsys):
    # `bn close -t ""` is neither an explicit selector nor a bare close; it must
    # ERROR -- with one target open as much as with several -- and never send
    # close_binary. (It used to resolve like a bare close and tear the single
    # target down.)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    one = {"ok": True, "result": [{"target_id": "123:1:7", "selector": "foo.bndb", "view_id": "1"}]}
    for listing in (one, _TWO_TARGETS):
        for selector in ("", "   "):
            calls = fake_transport({
                "list_targets": listing,
                "close_binary": {"ok": True, "result": {"closed": []}},
            })

            rc = bn.cli.main(["close", "-t", selector, "--format", "text"])

            assert rc == 2, (selector, listing)
            assert "close_binary" not in [c["op"] for c in calls], (selector, listing)
            err = capsys.readouterr().err
            assert "empty" in err and "--target" in err, (selector, listing)


def test_close_empty_selector_errors_even_with_sticky_pin(fake_transport, monkeypatch, capsys):
    # A sticky target pin must not paper over an explicit `-t ""`: the empty
    # selector is still an error (and the pin is not used either).
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "foo.bndb"})
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "foo.bndb", "view_id": "1"}],
        },
        "close_binary": {"ok": True, "result": {"closed": []}},
    })

    rc = bn.cli.main(["close", "-t", "", "--format", "text"])

    assert rc == 2
    assert "close_binary" not in [c["op"] for c in calls]
    assert "empty" in capsys.readouterr().err


def test_close_rejects_target_with_path(fake_transport, monkeypatch, capsys):
    # `bn close -t alpha.so /proj/beta.so`: the bridge gives the target priority
    # and the path was silently ignored. Contradictory -> error before any
    # request, mirroring the path+--all guard.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport({
        "list_targets": _TWO_TARGETS,
        "close_binary": {"ok": True, "result": {"closed": []}},
    })

    rc = bn.cli.main(["close", "-t", "alpha.so", "/proj/beta.so", "--format", "text"])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "not both" in err and "--target" in err


def test_close_rejects_target_with_all(fake_transport, monkeypatch, capsys):
    # `bn close -t alpha.so --all`: the target won and --all was silently
    # discarded. Contradictory -> error before any request.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport({
        "list_targets": _TWO_TARGETS,
        "close_binary": {"ok": True, "result": {"closed": []}},
    })

    rc = bn.cli.main(["close", "-t", "alpha.so", "--all", "--format", "text"])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "not both" in err and "--all" in err


def test_close_sticky_pin_with_path_or_all_is_not_a_conflict(fake_transport, monkeypatch, capsys):
    # Only an EXPLICIT -t conflicts with a path / --all. A sticky pin is dropped
    # for close (it must never pick what gets torn down), so `bn close <path>`
    # and `bn close --all` keep working under a pin.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "alpha.so"})
    for argv, expect_params in (
        (["close", "/tmp/one-binary"], {"path": str(__import__("pathlib").Path("/tmp/one-binary").resolve())}),
        (["close", "--all"], {"all": True}),
    ):
        calls = fake_transport({
            "list_targets": _TWO_TARGETS,
            "close_binary": {"ok": True, "result": {"closed": []}},
        })

        rc = bn.cli.main([*argv, "--format", "text"])

        assert rc == 0, argv
        assert [c["op"] for c in calls] == ["close_binary"], argv
        assert calls[-1]["target"] is None, argv
        assert calls[-1]["params"] == expect_params, argv


def test_close_bare_multiple_targets_requires_target(fake_transport, monkeypatch, capsys):
    # #664: bare `bn close` under multiple open targets used to close ALL of
    # them (the bridge treated "no target" as close-all) while `bn save`
    # refused. It now refuses with the same actionable hint + open-target list
    # as every other target-required command, and never sends close_binary.
    # `-t active` is the same volatile literal spelled explicitly and must
    # behave the same (it is never forwarded for the bridge to re-resolve).
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    for argv in (["close"], ["close", "-t", "active"]):
        calls = fake_transport({
            "list_targets": _TWO_TARGETS,
            "close_binary": {"ok": True, "result": {"closed": []}},
        })

        rc = bn.cli.main([*argv, "--format", "text"])

        assert rc == 2, argv
        assert [c["op"] for c in calls] == ["list_targets"], argv
        err = capsys.readouterr().err
        assert "requires --target when multiple targets are open" in err, argv
        assert "alpha.so" in err and "beta.so" in err, argv


def test_close_never_forwards_the_active_literal(fake_transport, monkeypatch, capsys):
    # #664 round 2 (B2): `active` is volatile -- the bridge re-resolves it at
    # close time, so a concurrent close/load could land the close on a
    # different binary. The CLI must never send it for close: it is resolved
    # client-side to the observed target_id (single target) or refused
    # (multiple), exactly like a bare close. Nothing else reaches the bridge.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport({
        "list_targets": _TWO_TARGETS,
        "close_binary": {"ok": True, "result": {"closed": []}},
    })

    rc = bn.cli.main(["close", "-t", "active", "--format", "text"])

    assert rc == 2
    assert [c["op"] for c in calls] == ["list_targets"]
    assert not any(c["target"] == "active" for c in calls)
    assert "multiple targets are open" in capsys.readouterr().err


def test_close_explicit_target_multiple_targets_skips_implicit_resolution(fake_transport, monkeypatch, capsys):
    # `bn close -t <target>` keeps working under multiple targets: an explicit
    # selector is forwarded as-is with no list_targets round-trip.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport({
        "list_targets": _TWO_TARGETS,
        "close_binary": {"ok": True, "result": {"closed": [{"path": "/proj/beta.so", "unsaved": False}]}},
    })

    rc = bn.cli.main(["close", "-t", "beta.so", "--format", "text"])

    assert rc == 0
    assert [c["op"] for c in calls] == ["close_binary"]
    assert calls[-1]["target"] == "beta.so"


def test_close_path_and_all_skip_target_resolution(fake_transport, monkeypatch, capsys):
    # `bn close <path>` and `bn close --all` keep their explicit semantics under
    # multiple targets: neither consults list_targets nor requires -t.
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    for argv, expect_params in (
        (["close", "/tmp/one-binary"], {"path": str(__import__("pathlib").Path("/tmp/one-binary").resolve())}),
        (["close", "--all"], {"all": True}),
    ):
        calls = fake_transport({
            "list_targets": _TWO_TARGETS,
            "close_binary": {"ok": True, "result": {"closed": []}},
        })

        rc = bn.cli.main([*argv, "--format", "text"])

        assert rc == 0, argv
        assert [c["op"] for c in calls] == ["close_binary"], argv
        assert calls[-1]["target"] is None, argv
        assert calls[-1]["params"] == expect_params, argv


def test_close_rejects_path_and_all_together(monkeypatch, capsys):
    # `bn close <path> --all` is contradictory; the bridge would let --all win
    # and close everything despite naming one file. The CLI rejects it before
    # any request is sent (#85).
    sent = {"called": False}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
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

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["sections", "--target", "active"])

    assert rc == 0
    assert captured["spawn_missing_named"] is False


def test_load_accepts_instance_id_alias_for_spawn_name(monkeypatch, tmp_path):
    # #258: `bn load --instance-id <new-id>` is an alias for `--instance <new-id>`,
    # so the spawn-name flag is consistent with `bn session start --instance-id`.
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
        captured["op"] = op
        captured["instance_id"] = instance_id
        captured["spawn_missing_named"] = spawn_missing_named
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["load", str(raw), "--instance-id", "brandnew"])

    assert rc == 0
    assert captured["op"] == "load_binary"
    assert captured["instance_id"] == "brandnew"
    assert captured["spawn_missing_named"] is True

def test_load_instance_id_does_not_clobber_env_instance(monkeypatch, tmp_path):
    # The --instance-id alias defaults to SUPPRESS, so when it is NOT passed it
    # must not overwrite a root-level --instance / BN_INSTANCE selection.
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False, **kwargs):
        captured["instance_id"] = instance_id
        return {"ok": True, "result": {"loaded": True, "path": str(raw), "notes": [], "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["--instance", "fromroot", "load", str(raw)])

    assert rc == 0
    assert captured["instance_id"] == "fromroot"


def test_load_forwards_workdir_and_marker_optout(monkeypatch, tmp_path):
    # #80: `bn load` forwards cwd as `workdir` (the marker anchor) and `no_marker`
    # from --no-marker / BN_NO_MARKERS.
    captured = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None,
                          spawn_missing_named=False, **kwargs):
        assert op == "load_binary"
        captured.update(params or {})
        return {"ok": True, "result": {"loaded": True, "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.delenv("BN_NO_MARKERS", raising=False)
    binpath = tmp_path / "prog"
    binpath.write_bytes(b"\x7fELF")

    rc = bn.cli.main(["load", str(binpath)])
    assert rc == 0
    assert "workdir" in captured and captured["no_marker"] is False

    # --no-marker opts out
    captured.clear()
    rc = bn.cli.main(["load", str(binpath), "--no-marker"])
    assert rc == 0 and captured["no_marker"] is True

    # BN_NO_MARKERS env opts out too
    captured.clear()
    monkeypatch.setenv("BN_NO_MARKERS", "1")
    rc = bn.cli.main(["load", str(binpath)])
    assert rc == 0 and captured["no_marker"] is True

    # BN_NO_MARKERS=0 does NOT opt out
    captured.clear()
    monkeypatch.setenv("BN_NO_MARKERS", "0")
    rc = bn.cli.main(["load", str(binpath)])
    assert rc == 0 and captured["no_marker"] is False
