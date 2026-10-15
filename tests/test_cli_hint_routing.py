from __future__ import annotations

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
            args, "trace", require_target=True, stem="trace", contextual_hints=True,
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
