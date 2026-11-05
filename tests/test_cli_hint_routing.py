from __future__ import annotations

import shlex

import pytest

import bn.cli
from bn.transport import (
    BridgeError,
    BridgeInstance,
    BridgeUnreachableError,
    _bridge_recovery_advice,
)


@pytest.mark.parametrize("status", [None, "invalid_arg_index"])
@pytest.mark.parametrize("error_type", [BridgeError, BridgeUnreachableError])
def test_call_routes_inspection_but_preserves_transport_recovery(
    monkeypatch, tmp_path, status, error_type,
):
    instance = BridgeInstance(
        pid=123,
        socket_path=tmp_path / "X1.sock",
        registry_path=tmp_path / "X1.json",
        plugin_name="test",
        plugin_version="test",
        started_at=None,
        meta={},
        instance_id="X1",
    )
    recovery = " ".join(_bridge_recovery_advice(instance))
    inspection = "il 'handler with $space' --view llil --ssa"
    already_routed = "`bn -i X1 target info`"
    requested = {"arg_index": 6}
    observed = {"argument_count": 2}
    error = error_type(
        f"Request failed. {recovery} Poll {already_routed}. Inspect `bn {inspection}`.",
        status=status,
        requested=requested,
        observed=observed,
    )

    def fail(op, **kwargs):
        raise error

    monkeypatch.setattr(bn.cli, "send_request", fail)
    args = bn.cli.build_parser().parse_args([
        "-i", "X1", "-t", "my tgt", "trace", "0x401000", "0x401020", "--arg", "6",
    ])
    with pytest.raises(error_type) as raised:
        bn.cli._call(
            args, "trace", require_target=True, stem="trace",
        )

    assert raised.value is error
    assert error.status is status
    assert error.requested is requested
    assert error.observed is observed
    assert str(error) == (
        f"Request failed. {recovery} Poll `bn -t 'my tgt' -i X1 target info`. "
        f"Inspect `bn -i X1 -t 'my tgt' {inspection}`."
    )


@pytest.mark.parametrize("command", [
    "session start",
    "session start --instance-id fresh --format json",
    "session list --format json",
    "session stop 'other session' --force",
    "session status 'job with spaces' --format json",
    "session restart other --discard-unsaved",
    "-i other session stop --force",
    "session --instance other stop --force",
    "-i il session stop other --force",
    "--instance=il session restart other --discard-unsaved",
    "session --instance il stop other --force",
    "instance list",
    "instance use other",
    "load '/tmp/my target' --instance-id fresh",
    "doctor",
    "not-a-command anything",
])
def test_lifecycle_and_non_target_hints_keep_their_own_arguments(command):
    original = f"Recover with `bn {command}`."
    value = {"error": original, "hints": [original]}
    bn.cli._contextualize_diagnostic_hints(
        value, target="my tgt", instance="X1",
    )
    assert value == {"error": original, "hints": [original]}


@pytest.mark.parametrize(("command", "expected"), [
    ("il 'handler with $space' --view llil --ssa",
     "-i X1 '--target=-local sample' il 'handler with $space' --view llil --ssa"),
    ('proto set 0x401000 "void f(char *p)"',
     '-i X1 \'--target=-local sample\' proto set 0x401000 "void f(char *p)"'),
    ("taint forward -f 'handler with space' --source 'var:a b'",
     "-i X1 '--target=-local sample' taint forward -f 'handler with space' --source 'var:a b'"),
    ("il 0x401000 --target='other target' --ssa",
     "-i X1 il 0x401000 --target='other target' --ssa"),
    ("il 0x401000 -i other --ssa",
     "'--target=-local sample' il 0x401000 -i other --ssa"),
    ("il 0x401000 -i other -t 'other target' --ssa",
     "il 0x401000 -i other -t 'other target' --ssa"),
    ("-i other target info", "'--target=-local sample' -i other target info"),
    ("target -i other info", "'--target=-local sample' target -i other info"),
    ("--instance=other target info", "'--target=-local sample' --instance=other target info"),
    ("-i session target info", "'--target=-local sample' -i session target info"),
    ("-iother target info", "'--target=-local sample' -iother target info"),
    ("--target='other target' il 0x401000", "-i X1 --target='other target' il 0x401000"),
    ("-i other -t 'other target' target info", "-i other -t 'other target' target info"),
])
def test_target_hints_route_without_rebuilding_shell_arguments(command, expected):
    value = {
        "hints": [f"Inspect `bn {command}`."],
        "nested": {"note": f"Inspect `bn {command}`."},
        "assumptions": [{"detail": f"Inspect `bn {command}`."}],
        "name": f"`bn {command}`",
        "il_text": f"`bn {command}`",
    }
    bn.cli._contextualize_diagnostic_hints(value, target="-local sample", instance="X1")
    assert value["hints"] == [f"Inspect `bn {expected}`."]
    assert value["nested"]["note"] == f"Inspect `bn {expected}`."
    assert value["assumptions"][0]["detail"] == f"Inspect `bn {expected}`."
    assert value["name"] == f"`bn {command}`"
    assert value["il_text"] == f"`bn {command}`"


def test_unparseable_or_unclosed_hint_is_not_rewritten():
    value = {"hints": ["Inspect `bn il 'unterminated`.", "Inspect `bn il 0x401000."]}
    original = list(value["hints"])
    bn.cli._contextualize_diagnostic_hints(value, target="my tgt", instance="X1")
    assert value["hints"] == original


# ---------------------------------------------------------------------------
# V7-02: every printed quick-view `bn refresh` carries this call's -i/-t
# ---------------------------------------------------------------------------

def _printed_refreshes(text: str) -> list[list[str]]:
    """Every `bn ... refresh` command *text* prints, as shell words."""
    from bn.formatters import CODE_SPAN_RE
    commands = [shlex.split(m.group(2)) for m in CODE_SPAN_RE.finditer(text)
                if m.group(2).startswith("bn ")]
    return [words for words in commands if words[-1] == "refresh"]


_ROUTED_REFRESH = ["bn", "-i", "X1", "-t", "my tgt", "refresh"]


def _answering(monkeypatch, result=None, error=None):
    def fake(op, **kwargs):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "my tgt"}]}
        if error is not None:
            raise error
        return {"ok": True, "result": result}
    monkeypatch.setattr(bn.cli, "send_request", fake)


@pytest.mark.parametrize(("argv", "result"), [
    (["function", "list"],
     {"analysis_state": "quick", "partial": True, "has_more": False, "total": 0, "items": []}),
    (["decompile", "0x401100"],
     {"analysis_state": "quick", "partial": True, "function": {"address": "0x401100",
      "name": "copy_in"}, "text": "void copy_in(void)\n{\n}", "warnings": [],
      "warning_rows": [], "comments": {}}),
    (["trace", "copy_in", "0x401120", "--arg", "1"],
     {"analysis_state": "quick", "partial": True, "function": "copy_in",
      "function_address": "0x401100", "function_display_name": "copy_in",
      "callee": "memcpy", "arg_index": 1, "target_address": "0x401120", "step_count": 0,
      "trace": [], "frontiers": [], "hints": [], "assumptions": [],
      "interprocedural": False, "ip_depth": 0}),
], ids=["function-list", "decompile", "trace"])
def test_a_quick_view_warning_prints_a_refresh_routed_to_this_call(monkeypatch, capsys,
                                                                  argv, result):
    # The shared warning printed a bare `bn refresh`, which fails with several
    # instances or targets open; only commands that opted in were routed.
    _answering(monkeypatch, result=result)
    assert bn.cli.main(["-i", "X1", "-t", "my tgt", *argv]) == 0
    assert _printed_refreshes(capsys.readouterr().out) == [_ROUTED_REFRESH]


def test_a_quick_view_refusal_prints_a_refresh_routed_to_this_call(monkeypatch, capsys):
    _answering(monkeypatch, error=BridgeError(
        "Strings are not available: this target was loaded with --quick (no analysis). "
        "Run `bn refresh` to build the full string set first."))
    assert bn.cli.main(["-i", "X1", "-t", "my tgt", "strings"]) == 2
    assert _printed_refreshes(capsys.readouterr().err) == [_ROUTED_REFRESH]


def test_a_quick_load_note_routes_its_refresh_to_the_target_it_loaded():
    # A load has no target of its own: each loaded row's note names its own.
    value = {"loaded": [{"target": {"selector": "second copy"}, "notes": [
        "loaded without analysis (--quick): the full function set needs `bn refresh`"]}]}
    bn.cli._contextualize_diagnostic_hints(value, target=None, instance="X1")
    assert _printed_refreshes(value["loaded"][0]["notes"][0]) == [
        ["bn", "-i", "X1", "-t", "second copy", "refresh"]]


def test_a_fanout_routes_each_rows_bridge_commands_to_that_row(monkeypatch, capsys):
    # G3-05: a fan-out never passed a row's reply through the routing pass, so
    # a command the bridge printed in it -- a result's note, a row's refusal
    # -- stayed bare, though it names that one row's target. Text renders the
    # rows stored here.
    import json

    rows = {"1:1:1": "first copy", "2:2:2": "second copy"}

    def fake(op, **kwargs):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": t, "selector": s}
                                           for t, s in rows.items()]}
        if kwargs["target"] == "2:2:2":
            raise BridgeError("Strings are not available: this target was loaded with "
                              "--quick (no analysis). Run `bn refresh` first.")
        return {"ok": True, "result": {
            "items": [], "total": 0, "offset": 0, "limit": 100, "returned": 0,
            "has_more": False, "note": "the string set is partial: run `bn refresh`"}}

    monkeypatch.setattr(bn.cli, "send_request", fake)
    assert bn.cli.main(["-i", "X1", "strings", "--all-targets", "--format", "json"]) == 0
    printed = {row["target"]: _printed_refreshes(
        row["result"]["note"] if row["ok"] else row["error"])
        for row in json.loads(capsys.readouterr().out)["instances"]}
    assert printed == {t: [["bn", "-i", "X1", "-t", t, "refresh"]] for t in rows}
