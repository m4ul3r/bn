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


# V8-07: every command that resolves a function name prints the bridge's
# ambiguity refusal with its retry command routed to this call. `proto get`
# printed a bare `bn function info 0x…` while `decompile` printed it routed.
def _ambiguous_name_refusal() -> BridgeError:
    from types import SimpleNamespace

    from bn_agent_bridge._shared import _format_ambiguous_function_error
    kind = SimpleNamespace(type=SimpleNamespace(name="FunctionSymbol"))
    rows = [SimpleNamespace(start=a, name="helper", symbol=kind) for a in (0x401130, 0x401150)]
    return BridgeError(_format_ambiguous_function_error("helper", rows))


def _printed_commands(text: str) -> list[list[str]]:
    from bn.formatters import CODE_SPAN_RE
    return [shlex.split(m.group(2)) for m in CODE_SPAN_RE.finditer(text)
            if m.group(2).startswith("bn ")]


# The arguments that name a function, by argparse dest, across the registry:
# the positional `identifier`/`function`, `--function` / `-f`, `--within`.
_FUNCTION_DESTS = frozenset({"identifier", "function", "within"})
# A value for every OTHER required argument a function-naming command takes,
# by dest. A command whose required argument is missing here fails the base
# case below, so a new one cannot slip past.
_REQUIRED_VALUES = {
    "callee": "memcpy", "comment": "note", "var": "x", "at": "0x401140",
    "arg_index": "0", "variable": "var_8", "new_name": "scale", "new_type": "int",
    "prototype": "int helper(int x)", "type": "Bookmarks", "sinks": "arg:memcpy:2",
    "sources": "param:0", "address": "0x401140",
}
# Options the handler requires together with a function locator though argparse
# does not (`tag remove --function` wants a `--type`), filled from the table above.
_COMPANION_DESTS = frozenset({"type"})
# Flags that do not select another code path of the command itself.
_NEUTRAL_FLAGS = frozenset({"-h", "--help", "--help-full", "--all-instances", "--estimate-output"})


def _function_naming_commands() -> list[tuple[tuple[str, ...], list[list[str]]]]:
    """Every registered command that resolves a function name, with its base
    invocation (`helper` in the function slot, a value in every other
    required one) and one variant per mode flag (`local list --frame`)."""
    import argparse

    import bn.commands  # noqa: F401 -- populates the registry
    parser = bn.cli.build_parser()
    out = []
    for spec in sorted(bn.cli._COMMANDS, key=lambda s: s["path"]):
        sub = bn.cli._selected_parser_for_argv(parser, list(spec["path"]))
        actions = [a for a in sub._actions if not isinstance(a, argparse._HelpAction)]
        named = [a for a in actions if a.dest in _FUNCTION_DESTS]
        if not named:
            continue
        fn = named[0]
        argv = list(spec["path"])
        for a in actions:
            if a.option_strings:
                continue
            if a is fn:
                argv.append("helper")
            elif a.nargs not in ("?", "*"):
                argv.append(_REQUIRED_VALUES.get(a.dest, f"<{a.dest}>"))
        if fn.option_strings:
            argv += [fn.option_strings[-1], "helper"]
        for a in actions:
            if a.option_strings and a is not fn and (a.required or a.dest in _COMPANION_DESTS):
                argv += [a.option_strings[-1], _REQUIRED_VALUES.get(a.dest, f"<{a.dest}>")]
        flags = [a.option_strings[-1] for a in actions if a.option_strings and a.nargs == 0
                 and not set(a.option_strings) & _NEUTRAL_FLAGS]
        out.append((spec["path"], [argv] + [argv + [flag] for flag in flags]))
    return out


_FUNCTION_NAMING = _function_naming_commands()


def test_the_function_naming_population_comes_from_the_registry():
    # The review's omissions are in it because the registry has them, not
    # because a list names them.
    paths = {" ".join(path) for path, _ in _FUNCTION_NAMING}
    assert {"evidence calls", "local retype", "local list", "comment set", "comment delete",
            "tag get", "tag add", "tag remove", "proto get", "decompile"} <= paths


@pytest.mark.parametrize(("path", "invocations"), _FUNCTION_NAMING,
                         ids=[" ".join(p) for p, _ in _FUNCTION_NAMING])
def test_an_ambiguous_name_refusal_prints_its_retry_routed(monkeypatch, capsys, path,
                                                          invocations):
    # V8-07 (closed by G3's global routing; this is its registry-derived
    # coverage): every invocation that reaches the bridge and is refused for an
    # ambiguous name prints its retry routed to this call's -i/-t.
    sent: list[str] = []

    def fake(op, **kwargs):
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "my tgt"}]}
        sent.append(op)
        # A fresh refusal per request: the CLI routes the message in place, so
        # a shared one would arrive pre-routed at every later invocation.
        raise _ambiguous_name_refusal()
    monkeypatch.setattr(bn.cli, "send_request", fake)
    for k, argv in enumerate(invocations):
        sent.clear()
        try:
            rc = bn.cli.main(["-i", "X1", "-t", "my tgt", *argv])
        except SystemExit as stop:
            rc = stop.code
        err = capsys.readouterr().err
        if k == 0:
            # The base invocation must reach the bridge, or this is no coverage.
            assert sent, (argv, err)
        if not sent:
            continue
        assert rc == 2, (argv, err)
        printed = _printed_commands(err)
        assert printed and all(words[1:5] == ["-i", "X1", "-t", "my tgt"]
                               for words in printed), (argv, printed)
