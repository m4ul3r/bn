from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
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


def _mutate_calls_in_source(
    source: str, *, line_offset: int = 0, where: str = "source"
) -> list[tuple[int, str, bool]]:
    """``[(lineno, op_name, has_summary_transform), ...]`` for every literal
    ``_mutate(args, "<op>", ...)`` call site in *source*; *line_offset* maps the
    parsed fragment's line numbers back onto the file it was sliced out of.

    The ONE extractor behind every population in this file -- #684's
    per-handler walk and #720's cross-check of the handler-derived population
    against a package-wide scan -- so no call site can be visible to one side
    and invisible to the other.

    A `_mutate()` call whose op name is NOT a string literal raises instead of
    being skipped. Skipping would drop that op out of the swept population
    silently -- the same "narrower than the real failure surface" hole that let
    the first version of this sweep miss `go_rename`. Every call site in the
    tree today passes the op positionally as a literal; a future one that does
    not must make this guard fail loudly and get an extractor that understands
    it, not vanish from the sweep.
    """
    calls: list[tuple[int, str, bool]] = []
    for node in ast.walk(ast.parse(source)):
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
            f"{where} calls _mutate() at line {node.lineno + line_offset} with a "
            "non-literal op name, so this sweep cannot tell WHICH bridge op it "
            "routes and would silently drop it from the #684 population. Pass the "
            "op as a string literal, or teach the extractor to resolve it."
        )
        calls.append((node.lineno + line_offset, op_arg.value,
                      any(kw.arg == "summary_transform" for kw in node.keywords)))
    return calls


def _mutate_sites_in_source(
    source: str, *, line_offset: int = 0, where: str = "source"
) -> set[tuple[int, str]]:
    """``{(lineno, op_name), ...}`` -- `_mutate_calls_in_source` without the
    per-site wiring detail, the shape both #720 populations are keyed on. A
    multiset of op names is NOT enough here: moving one call site into a helper
    leaves the module-wide op multiset unchanged, so the cross-check has to
    compare exact (file, lineno, op) triples."""
    return {(lineno, op) for lineno, op, _ in _mutate_calls_in_source(
        source, line_offset=line_offset, where=where)}


def _mutate_call_sites(handler: Callable[..., int]) -> list[tuple[str, bool]]:
    """``[(op_name, has_summary_transform), ...]`` for every literal
    ``_mutate(args, "<op>", ...)`` call site found in *handler*'s OWN source.

    This is how the op<->CLI-command mapping is DERIVED (#684): rather than
    hand-listing which command drives which bridge op, statically scan each
    registered command handler's own source for its `_mutate()` call(s) and
    read the op name straight off the literal second argument. A site only a
    module-level helper reaches is deliberately NOT in here -- #720's
    `_assert_scan_accounted_for` cross-check exists to catch exactly that, since
    otherwise the op it routes would drop out of the sweep unseen.
    """
    try:
        source = textwrap.dedent(inspect.getsource(handler))
    except (OSError, TypeError):
        return []
    return [(op, has_transform) for _, op, has_transform in _mutate_calls_in_source(
        source, where=f"{getattr(handler, '__qualname__', handler)!r}")]


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


def _module_mutate_sites(package_root: Path) -> set[tuple[str, int, str]]:
    """The scan half of the #720 cross-check: every literal-op `_mutate()` call
    site ANYWHERE in the `bn` package -- the CLI layer (`bn.cli`, which defines
    `_mutate`, plus `bn.commands`) and anything else that may grow a call site
    -- keyed by (package-relative file, absolute lineno, op name).

    The region is the WHOLE package, never just `commands/` + `cli.py`: a
    handler that delegates to a module-level helper in a sibling module (e.g.
    `bn/<helper>.py`) moves its site outside the narrower region, so BOTH halves
    of the comparison shrink together and the op drops out of the sweep with the
    cross-check silent. Scanning everything `bn` owns keeps the two populations
    comparable wherever a call site lands.

    The region is anchored on where `bn.cli` actually lives, never on the
    process cwd, so an unrelated checkout as cwd cannot make this read another
    tree."""
    sites: set[tuple[str, int, str]] = set()
    for path in sorted(package_root.rglob("*.py")):
        if not path.is_file():
            # Narrowing the region is loud, not silent: any handler owning a
            # call site the scan cannot see lands in the handler-only half of
            # `_assert_scan_accounted_for`.
            continue
        relpath = path.relative_to(package_root).as_posix()
        for lineno, op in _mutate_sites_in_source(path.read_text(), where=relpath):
            sites.add((relpath, lineno, op))
    return sites


def _handler_mutate_sites(
    commands: list[dict[str, Any]], *, package_root: Path
) -> set[tuple[str, int, str]]:
    """The handler-derived half of the #720 cross-check: every literal-op
    `_mutate()` call site inside a registered handler's OWN source, keyed the
    same way as `_module_mutate_sites` so the two populations are directly
    comparable.

    A handler registered under several command paths appears several times in
    `_COMMANDS`, which needs no special handling: the population is a SET of
    (file, lineno, op) triples, so one source block parsed once per
    registration collapses onto the same triples."""
    sites: set[tuple[str, int, str]] = set()
    for spec in commands:
        handler = spec["handler"]
        try:
            lines, start = inspect.getsourcelines(handler)
        except (OSError, TypeError):
            # Uninspectable handler: any site it owns then shows up as
            # module-only below, so this cannot hide a call site.
            continue
        source_file = inspect.getsourcefile(handler)
        if source_file is None:
            continue
        source_path = Path(source_file).resolve()
        try:
            relpath = source_path.relative_to(package_root).as_posix()
        except ValueError:
            # A handler defined outside the scanned package (e.g. a plugin)
            # owns no site INSIDE the scanned region, so it contributes none.
            continue
        for lineno, op in _mutate_sites_in_source(
            textwrap.dedent("".join(lines)), line_offset=start - 1, where=relpath
        ):
            sites.add((relpath, lineno, op))
    return sites


def _assert_scan_accounted_for(
    handler_sites: set[tuple[str, int, str]], module_sites: set[tuple[str, int, str]]
) -> None:
    """#720 cross-check: the #684 population is DERIVED from each registered
    handler's own source, so a `_mutate()` call site reached through a
    module-level helper -- the handler delegates, the helper calls `_mutate()` --
    is invisible to it and its op silently drops out of the sweep. `assert
    cli_wiring` does not catch that either: the other ops are still found. So
    compare the handler-derived population against an independent AST scan of
    the whole `bn` package, and fail on a difference in EITHER direction: a
    module-only site is a blind spot in the population, a handler-only site
    means the scan region is narrower than the population it is checked
    against."""
    def _render(sites: set[tuple[str, int, str]]) -> str:
        return "\n".join(f"  {relpath}:{lineno}  {op}"
                         for relpath, lineno, op in sorted(sites)) or "  (none)"

    module_only = module_sites - handler_sites
    handler_only = handler_sites - module_sites
    assert not module_only and not handler_only, (
        "the #684 sweep population (derived from each registered command "
        "handler's OWN source) does not account for every _mutate() call site in "
        "the bn package (#720).\n"
        f"_mutate() call sites NO registered handler's own source contains "
        f"({len(module_only)}) -- the helper-mediated delegation shape: a handler "
        "calls a module-level helper that calls _mutate(), so the op it routes is "
        "missing from the swept population and its summary wiring is never "
        f"checked:\n{_render(module_only)}\n"
        f"handler-owned call sites the scan region does NOT contain "
        f"({len(handler_only)}) -- the scanned file set is narrower than the "
        f"population checked against it, so widen the scan:\n{_render(handler_only)}"
    )


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


def _sweep(cli_wiring: dict[str, bool], spec: Callable[[str], Any],
           commands: list[dict[str, Any]]) -> None:
    """The #684 sweep body, factored out so both the production guard below
    and the regression test that proves `go_rename` coverage run the SAME
    code, not a bespoke re-implementation. *spec* is `REGISTRY.spec` (or a
    stand-in with the same signature); *commands* is the `@command` registry
    the swept population is derived from -- `_real_commands()` -- which also
    supplies the handler-side half of the #720 cross-check that keeps a
    helper-mediated `_mutate()` call site from dropping out of the population
    unseen."""
    assert cli_wiring, "sanity: no _mutate() call site was found at all"
    package_root = Path(cli.__file__).resolve().parent
    _assert_scan_accounted_for(
        _handler_mutate_sites(commands, package_root=package_root),
        _module_mutate_sites(package_root),
    )
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
    commands = _real_commands()
    cli_wiring = _cli_summary_wiring(commands)
    _sweep(cli_wiring, bridge.REGISTRY.spec, commands)


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
    commands = _real_commands()
    cli_wiring = dict(_cli_summary_wiring(commands))
    assert "go_rename" in cli_wiring          # now inside the swept population at all
    assert cli_wiring["go_rename"] is True    # currently wired safely
    cli_wiring["go_rename"] = False           # simulate the #683 regression
    with pytest.raises(AssertionError, match="go_rename"):
        _sweep(cli_wiring, bridge.REGISTRY.spec, commands)


def test_helper_mediated_mutate_call_site_is_flagged(tmp_path, monkeypatch):
    """#720: the #684 population is DERIVED from each registered handler's OWN
    source, so a handler that delegates to a module-level helper which calls
    `_mutate()` drops its op out of the sweep entirely -- and invisibly, because
    `assert cli_wiring` only notices TOTAL extractor breakage.

    Proven on a scratch MIRROR of a real commands module (`commands/tags.py`):
    byte-identical first as a control, then with its `_tag_add` `_mutate()` call
    MOVED into an appended module-level helper and the vacated lines padded so
    every other line number is unchanged. Both are compared through
    `_assert_scan_accounted_for`, the same comparison the sweep runs. The
    module-wide op multiset is identical before and after the move, so only the
    (file, lineno, op) triples can catch it. The scratch module is never
    imported or registered -- its handlers are found by AST -- while the
    production population still comes from the live registry.

    The move is then repeated with the helper OUTSIDE `commands/` -- a sibling
    module at the scratch package root. That is the region the original
    commands/+cli.py scan did not cover, so under it the site was invisible to
    both halves and the cross-check stayed silent while the op dropped out of
    the population; the scan now covers the whole package and names the sibling
    module's file:line.
    """
    bridge = _load_bridge(monkeypatch)
    package_root = Path(cli.__file__).resolve().parent
    scratch_module = tmp_path / "commands" / "tags.py"
    scratch_module.parent.mkdir(parents=True)
    scratch_module.write_text((package_root / "commands" / "tags.py").read_text())

    def _is_command_handler(node: ast.AST) -> bool:
        if not isinstance(node, ast.FunctionDef):
            return False
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = (target.id if isinstance(target, ast.Name)
                    else target.attr if isinstance(target, ast.Attribute) else None)
            if name == "command":
                return True
        return False

    def _scratch_handler_sites() -> set[tuple[str, int, str]]:
        """The handler-side population of the scratch mirror: every `_mutate()`
        call site inside a `@command`-decorated function's own source, found by
        AST rather than by importing the module."""
        sites: set[tuple[str, int, str]] = set()
        for path in sorted((tmp_path / "commands").rglob("*.py")):
            relpath = path.relative_to(tmp_path).as_posix()
            source = path.read_text()
            lines = source.splitlines(keepends=True)
            for node in ast.walk(ast.parse(source)):
                if not _is_command_handler(node):
                    continue
                block = textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))
                sites |= {(relpath, lineno, op) for lineno, op in
                          _mutate_sites_in_source(block, line_offset=node.lineno - 1)}
        return sites

    # Control: the byte-identical mirror is fully accounted for, so what follows
    # measures the transformation, not the mirroring.
    control_handler_sites = _scratch_handler_sites()
    control_module_sites = _module_mutate_sites(tmp_path)
    _assert_scan_accounted_for(control_handler_sites, control_module_sites)

    # The #720 shape: move `_tag_add`'s `_mutate()` call out of the handler into
    # an appended module-level helper, padding the vacated lines so every other
    # line number in the module is unchanged.
    source = scratch_module.read_text()
    lines = source.splitlines(keepends=True)
    tag_add_line = next(
        lineno for lineno, op in _mutate_sites_in_source(source) if op == "tag_add"
    )
    call = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_mutate" and node.lineno == tag_add_line)
    moved = "".join(lines[call.lineno - 1:call.end_lineno])
    vacated = call.end_lineno - call.lineno + 1
    scratch_module.write_text(
        "".join(lines[:call.lineno - 1])
        + "    return _tag_add_via_helper(args)\n" + "\n" * (vacated - 1)
        + "".join(lines[call.end_lineno:])
        + "\n\ndef _tag_add_via_helper(args):\n" + moved
    )

    handler_sites = _scratch_handler_sites()
    module_sites = _module_mutate_sites(tmp_path)
    # The move changes no op name and no site count, only where the site lives,
    # so a bare op multiset comparison could not tell the two apart...
    assert (sorted(op for _, _, op in module_sites)
            == sorted(op for _, _, op in control_module_sites))
    # ...and, because the vacated lines are padded, the ONLY handler site that
    # went away is the moved one: every other line number is unchanged.
    assert handler_sites == control_handler_sites - {("commands/tags.py", tag_add_line,
                                                      "tag_add")}
    module_only = module_sites - handler_sites
    assert len(module_only) == 1, module_only
    relpath, lineno, op = next(iter(module_only))
    assert (relpath, op) == ("commands/tags.py", "tag_add")
    with pytest.raises(AssertionError) as excinfo:
        _assert_scan_accounted_for(handler_sites, module_sites)
    assert f"{relpath}:{lineno}" in str(excinfo.value)
    assert "tag_add" in str(excinfo.value)

    # Same transformation, helper OUTSIDE `commands/`: rewrite the mirror
    # WITHOUT the helper the move above appended (same delegated body, same
    # padding, so still byte-comparable), and land that helper in a sibling
    # module at the scratch package root instead. A scan restricted to
    # `commands/` + `cli.py` sees neither the handler site (moved away) nor the
    # sibling one (outside the region), so both populations shrink together and
    # the cross-check stays silent -- the #720 hole the whole-package region
    # closes.
    sibling_module = tmp_path / "mutation_helpers.py"
    scratch_module.write_text(
        "".join(lines[:call.lineno - 1])
        + "    return _tag_add_via_helper(args)\n" + "\n" * (vacated - 1)
        + "".join(lines[call.end_lineno:])
    )
    sibling_module.write_text(
        "\n\ndef _tag_add_via_helper(args):\n" + moved
    )

    handler_sites = _scratch_handler_sites()
    module_sites = _module_mutate_sites(tmp_path)
    assert (sorted(op for _, _, op in module_sites)
            == sorted(op for _, _, op in control_module_sites))
    assert handler_sites == control_handler_sites - {("commands/tags.py", tag_add_line,
                                                      "tag_add")}
    module_only = module_sites - handler_sites
    assert len(module_only) == 1, module_only
    relpath, lineno, op = next(iter(module_only))
    assert (relpath, op) == ("mutation_helpers.py", "tag_add")
    with pytest.raises(AssertionError) as excinfo:
        _assert_scan_accounted_for(handler_sites, module_sites)
    assert f"{relpath}:{lineno}" in str(excinfo.value)
    assert "tag_add" in str(excinfo.value)

    # The sweep must actually RUN that comparison: hand `_sweep` the live
    # registry minus the `tag add` registration -- the in-package call site then
    # has no registered handler whose own source contains it -- and it has to
    # fail naming the orphaned site.
    commands = _real_commands()
    unaccounted = [spec for spec in commands if spec["path"] != ("tag", "add")]
    assert len(unaccounted) == len(commands) - 1
    orphan = next(site for site in _module_mutate_sites(package_root) if site[2] == "tag_add")
    with pytest.raises(AssertionError) as excinfo:
        _sweep(_cli_summary_wiring(unaccounted), bridge.REGISTRY.spec, unaccounted)
    assert f"{orphan[0]}:{orphan[1]}" in str(excinfo.value)
