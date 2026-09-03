from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable

import pytest

import bn.cli as cli

from _bridge_fakes import _load_bridge


def _real_commands() -> list[dict[str, Any]]:
    """The live `@command` registry (`bn.cli._COMMANDS`), populated lazily by
    `build_parser()` -- mirrors how every other CLI test reaches it."""
    if not cli._COMMANDS:
        cli.build_parser()
    return cli._COMMANDS


def _mutate_call_sites(handler: Callable[..., int]) -> list[tuple[str, bool]]:
    """``[(op_name, has_summary_transform), ...]`` for every literal
    ``_mutate(args, "<op>", ...)`` call site found in *handler*'s source.

    This is how the op<->CLI-command mapping is DERIVED (#684): rather than
    hand-listing which command drives which bridge op, statically scan each
    registered command handler's own source for its `_mutate()` call(s) and
    read the op name straight off the literal second argument.
    """
    try:
        source = textwrap.dedent(inspect.getsource(handler))
    except (OSError, TypeError):
        return []
    tree = ast.parse(source)
    sites: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name != "_mutate":
            continue
        if len(node.args) < 2 or not (
            isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
        ):
            continue
        has_transform = any(kw.arg == "summary_transform" for kw in node.keywords)
        sites.append((node.args[1].value, has_transform))
    return sites


def _cli_summary_wiring(commands: list[dict[str, Any]]) -> dict[str, bool]:
    """``op name -> True`` iff EVERY `_mutate()` call site for that op (derived
    from *commands* via `_mutate_call_sites`, never hand-listed) registers a
    `summary_transform` escape hatch. An op absent from the returned dict is
    never routed through `_mutate()` by any command at all."""
    wiring: dict[str, bool] = {}
    for spec in commands:
        for op_name, has_transform in _mutate_call_sites(spec["handler"]):
            wiring[op_name] = wiring.get(op_name, True) and has_transform
    return wiring


def _binder_populates_results(binder: Callable[..., Any]) -> bool:
    """True iff *binder* delegates directly to `bridge._mutation(` -- the one
    bridge helper that structurally guarantees a `results[]` row per requested
    operation: `mutation_engine._mutation()` refuses an empty operation list
    outright, and appends exactly one result row per requested op on every
    return path (including the mid-batch failure path)."""
    try:
        source = inspect.getsource(binder)
    except (OSError, TypeError):
        return False
    return "bridge._mutation(" in source


# Ops whose bridge implementation is bespoke -- it does NOT delegate to the
# shared `bridge._mutation()` helper -- but has been manually audited to
# populate a `results[]` row on every return path, so it does not need a
# `summary_transform` escape hatch either. Frozen like
# EXPECTED_READ/EXPECTED_WRITE in test_op_registry.py: update this set, with a
# comment proving the audit, in the SAME commit as any change to the bespoke
# op's result shape.
AUDITED_BESPOKE_SAFE_OPS = {
    # create_comments._function_create builds its own "results" list (with
    # exactly one row) on every return path -- the already-exists noop, the
    # non-code guard rejection, the post-analysis verification failure, and the
    # success/preview row -- instead of calling bridge._mutation() (verified by
    # reading src/bn_agent_bridge/create_comments.py:_function_create).
    "function_create",
}


def _assert_op_is_summary_safe(
    op_name: str, *, binder: Callable[..., Any], cli_wiring: dict[str, bool]
) -> None:
    """The #684 contract for one op: it is safe for the GENERIC
    `_mutation_summary` compact path iff its bridge binder is known to
    populate `results[]` (delegates to `bridge._mutation()`, or is an audited
    bespoke exception), OR every CLI command that invokes it via `_mutate()`
    registers a `summary_transform`. An op the CLI never routes through
    `_mutate()` at all (rendered by its own dedicated formatter, e.g.
    close/save/py-exec) never reaches the compact mutation summary and is
    outside this bug's failure surface, so it is not checked."""
    if _binder_populates_results(binder) or op_name in AUDITED_BESPOKE_SAFE_OPS:
        return
    if op_name not in cli_wiring:
        return
    assert cli_wiring[op_name], (
        f"write-locked op {op_name!r} neither delegates to bridge._mutation() "
        "(so its results[] population is unverified) nor has a summary_transform "
        "registered on its CLI command -- the default _mutation_summary compact "
        "path would render it as changed=0 verified=0 noop=0 failed=0 "
        "dirty_after=False even if it actually did real work (#684, the class of "
        "bug behind #683's go_rename regression). Register summary_transform=... "
        "on its _mutate() call, or add it to AUDITED_BESPOKE_SAFE_OPS with proof "
        "results[] is populated on every return path."
    )


def test_every_write_locked_op_has_safe_summary_wiring(monkeypatch):
    """#684 primary guard: every write-locked bridge op is either provably safe
    for the shared compact mutation summary, or explicitly opts out via
    `summary_transform`. A FUTURE op that reports through its own counters and
    forgets the wiring must fail here -- not render a plausible all-zero status
    at an agent that then discards real work."""
    bridge = _load_bridge(monkeypatch)
    cli_wiring = _cli_summary_wiring(_real_commands())
    write_ops = bridge.REGISTRY.write_locked_ops()
    assert write_ops, "sanity: the write-locked op set must not be empty"
    for op_name in sorted(write_ops):
        _assert_op_is_summary_safe(
            op_name, binder=bridge.REGISTRY.spec(op_name).binder, cli_wiring=cli_wiring,
        )


def test_hypothetical_counter_reporting_op_without_wiring_is_flagged():
    """Construct a SYNTHETIC op -- never registered in the production REGISTRY
    or `_COMMANDS` -- shaped exactly like the class of bug #684 describes: a
    bridge binder that reports through its OWN counter field instead of
    `results[]`, invoked by a CLI command with no `summary_transform`. The
    checker must reject that combination, and accept it once either half of
    the wiring (bridge-side `_mutation()` delegation, or a CLI
    `summary_transform`) is present.
    """

    def _bind_hypothetical_counter_op(bridge, params, target):
        # Bespoke: reports via its own counter, never touches results[] -- the
        # exact shape go_rename had before #683's summary_transform fix.
        return {"kind": "hypothetical_counter_op", "success": True, "committed": True,
                "hypothetical_verified_count": 42}

    def _fake_cli_handler_no_transform(args):
        return cli._mutate(args, "hypothetical_counter_op", {}, stem="hypothetical")

    def _fake_cli_handler_with_transform(args):
        return cli._mutate(args, "hypothetical_counter_op", {}, stem="hypothetical",
                            summary_transform=lambda v: v)

    unsafe_commands = [{"handler": _fake_cli_handler_no_transform}]
    safe_commands = [{"handler": _fake_cli_handler_with_transform}]

    # Neither wiring present -> flagged.
    with pytest.raises(AssertionError, match="hypothetical_counter_op"):
        _assert_op_is_summary_safe(
            "hypothetical_counter_op",
            binder=_bind_hypothetical_counter_op,
            cli_wiring=_cli_summary_wiring(unsafe_commands),
        )

    # Registering summary_transform on the CLI side clears it...
    _assert_op_is_summary_safe(
        "hypothetical_counter_op",
        binder=_bind_hypothetical_counter_op,
        cli_wiring=_cli_summary_wiring(safe_commands),
    )

    # ...and so does routing the bridge binder through the shared helper
    # instead, even with no CLI-side summary_transform.
    def _bind_via_mutation_engine(bridge, params, target):
        return bridge._mutation(target, False, [{**params, "op": "hypothetical_counter_op"}])

    _assert_op_is_summary_safe(
        "hypothetical_counter_op",
        binder=_bind_via_mutation_engine,
        cli_wiring=_cli_summary_wiring(unsafe_commands),
    )


def test_go_rename_is_not_write_locked_but_has_its_own_escape_hatch(monkeypatch):
    """`go_rename` self-manages locking (lock="none", #365 -- it releases the
    write lock between chunks), so it is intentionally absent from
    `write_locked_ops()` and the primary registry sweep above never visits it.
    It still needs -- and has -- the same #684 protection via its dedicated
    `summary_transform`; this pins that it isn't accidentally dropped."""
    bridge = _load_bridge(monkeypatch)
    assert "go_rename" not in bridge.REGISTRY.write_locked_ops()
    cli_wiring = _cli_summary_wiring(_real_commands())
    assert cli_wiring.get("go_rename") is True
