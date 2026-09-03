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

    A `_mutate()` call whose op name is NOT a string literal raises instead of
    being skipped. Skipping would drop that op out of the swept population
    silently -- the same "narrower than the real failure surface" hole that let
    the first version of this sweep miss `go_rename`. Every call site in the
    tree today passes the op positionally as a literal; a future one that does
    not must make this guard fail loudly and get an extractor that understands
    it, not vanish from the sweep.
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
        op_arg: ast.expr | None = node.args[1] if len(node.args) >= 2 else next(
            (kw.value for kw in node.keywords if kw.arg == "op"), None
        )
        assert isinstance(op_arg, ast.Constant) and isinstance(op_arg.value, str), (
            f"{getattr(handler, '__qualname__', handler)!r} calls _mutate() with a "
            "non-literal op name, so this sweep cannot tell WHICH bridge op it "
            "routes and would silently drop it from the #684 population. Pass the "
            "op as a string literal, or teach _mutate_call_sites to resolve it."
        )
        has_transform = any(kw.arg == "summary_transform" for kw in node.keywords)
        sites.append((op_arg.value, has_transform))
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
    """True iff *binder*'s own source contains a call shaped `bridge._mutation(
    ... )` -- the one bridge helper that structurally guarantees a `results[]`
    row per requested operation: `mutation_engine._mutation()` refuses an
    empty operation list outright, and appends exactly one result row per
    requested op on every return path (including the mid-batch failure path).

    Matched via AST (an `ast.Call` whose `func` is an `ast.Attribute` with
    `attr == "_mutation"`), not a source substring: a substring match is
    fooled by a binder that only MENTIONS `bridge._mutation(` in a comment
    while actually reporting through its own bespoke counters -- exactly the
    counter-reporting shape this whole file exists to catch.
    """
    try:
        source = textwrap.dedent(inspect.getsource(binder))
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_mutation"
        for node in ast.walk(tree)
    )


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
    registers a `summary_transform`.

    *op_name* MUST be present in *cli_wiring* -- i.e. some command really does
    route it through `_mutate()`. There is deliberately no "not routed, so skip"
    branch: the swept population IS `cli_wiring.keys()`, so such a branch would
    be unreachable dead code whose only possible effect is a SILENT pass, the
    exact failure mode that let the previous `write_locked_ops()`-based sweep
    drop `go_rename`. An op the CLI never routes through `_mutate()` (rendered
    by its own dedicated formatter, e.g. close/save/py-exec) cannot reach the
    compact mutation summary at all and is simply never in the population."""
    if _binder_populates_results(binder) or op_name in AUDITED_BESPOKE_SAFE_OPS:
        return
    assert cli_wiring[op_name], (
        f"mutating op {op_name!r} (routed through cli._mutate) neither delegates "
        "to bridge._mutation() (so its results[] population is unverified) nor "
        "has a summary_transform registered on its CLI command. If it reports "
        "through its own counters it will reach the generic _mutation_summary "
        "with an EMPTY results[], which cannot measure anything: the compact "
        "status renders changed=None verified=None noop=None failed=None with a "
        "fail-safe dirty_after=True and an `unmeasured` warning, instead of the "
        "real counts the op did produce (#684, the class of bug behind #683's "
        "go_rename regression -- note go_rename is lock=\"none\", so being "
        "outside REGISTRY.write_locked_ops() is no excuse). Register "
        "summary_transform=... on its _mutate() call, or add it to "
        "AUDITED_BESPOKE_SAFE_OPS with proof results[] is populated on every "
        "return path."
    )


def _sweep(cli_wiring: dict[str, bool], spec: Callable[[str], Any]) -> None:
    """The #684 sweep body, factored out so both the production guard below
    and the regression test that proves `go_rename` coverage run the SAME
    code, not a bespoke re-implementation. *spec* is `REGISTRY.spec` (or a
    stand-in with the same signature)."""
    assert cli_wiring, "sanity: no _mutate() call site was found at all"
    for op_name in sorted(cli_wiring):
        _assert_op_is_summary_safe(
            op_name, binder=spec(op_name).binder, cli_wiring=cli_wiring,
        )


def test_every_mutating_op_has_safe_summary_wiring(monkeypatch):
    """#684 primary guard: every op the CLI routes through `_mutate()` -- the
    only path that can reach the generic `_mutation_summary` (`cli.py`
    `_mutate`'s `result_transform`/`spill_status`) -- is either provably safe
    for it, or explicitly opts out via `summary_transform`.

    The population swept is `cli_wiring.keys()`, derived straight from
    `_mutate()` call sites via `_cli_summary_wiring` -- NOT
    `REGISTRY.write_locked_ops()`. `write_locked_ops()` is the WRONG
    population: it excludes any op that self-manages its own locking
    (`lock="none"`), which is exactly the shape that caused #683's
    `go_rename` regression. The OLD version of this sweep iterated
    `write_locked_ops() ∩ cli_wiring` -- a strict subset of `cli_wiring` --
    and so could never see `go_rename` at all; it had to be patched with a
    hand-written exception test instead (the "same remembered exception #684
    complains about"). `cli_wiring.keys()` is complete, minimal, needs no
    registry knowledge, and covers `go_rename` automatically -- see
    `test_go_rename_summary_transform_removal_is_caught_by_the_sweep` below.
    A FUTURE `lock="none"` op that reports through its own counters and
    forgets the wiring must fail HERE."""
    bridge = _load_bridge(monkeypatch)
    cli_wiring = _cli_summary_wiring(_real_commands())
    _sweep(cli_wiring, bridge.REGISTRY.spec)


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


def test_go_rename_is_not_write_locked(monkeypatch):
    """`go_rename` self-manages locking (lock="none", #365 -- it releases the
    write lock between chunks), so it is intentionally absent from
    `write_locked_ops()`. That fact is WHY the old `write_locked_ops()`-based
    sweep could never see `go_rename` (#683) -- it is not itself something the
    new `cli_wiring.keys()`-based sweep needs to know, since it does not
    consult `write_locked_ops()` at all, but it stays worth pinning on its own
    so `write_locked_ops()` is never mistaken for a safe sweep population
    again. The wiring check itself now lives in
    `test_go_rename_summary_transform_removal_is_caught_by_the_sweep` below,
    which exercises the real sweep instead of re-deriving `cli_wiring` here."""
    bridge = _load_bridge(monkeypatch)
    assert "go_rename" not in bridge.REGISTRY.write_locked_ops()


def test_go_rename_summary_transform_removal_is_caught_by_the_sweep(monkeypatch):
    """Major-2 fix verification: prove the NEW sweep actually covers
    `go_rename`, by running the SAME sweep code
    (`test_every_mutating_op_has_safe_summary_wiring` calls `_sweep`) with
    `go_rename`'s `summary_transform` wiring simulated as dropped -- the
    actual #683 regression -- and showing it fails. Not a bespoke re-check:
    `_sweep` is the identical function the production guard runs.

    `go_rename`'s bridge binder is bespoke (does not delegate to
    `bridge._mutation()`) and `go_rename` is not in `AUDITED_BESPOKE_SAFE_OPS`,
    so it depends entirely on the CLI-side `summary_transform` this test
    strips."""
    bridge = _load_bridge(monkeypatch)
    cli_wiring = dict(_cli_summary_wiring(_real_commands()))
    assert "go_rename" in cli_wiring          # now inside the swept population at all
    assert cli_wiring["go_rename"] is True    # currently wired safely
    cli_wiring["go_rename"] = False           # simulate the #683 regression
    with pytest.raises(AssertionError, match="go_rename"):
        _sweep(cli_wiring, bridge.REGISTRY.spec)
