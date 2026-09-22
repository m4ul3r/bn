"""Guards that keep the agent-instruction docs honest.

`CLAUDE.md` is the canonical agent-instruction file and root `AGENTS.md` is a
tracked symlink to it, so an agent reading either one sees the same tree layout
(#607). These tests pin the invariants that actually misled agents before:
a second physical copy that drifts, a bridge path that no longer exists, a
`uv run pytest` line naming a module or test id that was renamed away (#614),
and CLI failure classifications that callers rely on.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest

# NB: no module-level `binaryninja` stub here. Injecting one at import time
# poisons `sys.modules` for every module collected AFTER this one, and the
# bridge's taint modules import real symbols from it -- under a randomized
# collection order that turned into a suite-wide collection error. These two
# imports are CLI-side and need no engine.
from bn.formatters import (FAILED_MUTATION_STATUSES, _go_rename_summary,
                           _mutation_summary)

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO / "CLAUDE.md"
AGENTS_MD = REPO / "AGENTS.md"

# Agent-facing docs an agent reads after CLAUDE.md; all must stay off the ghost
# `plugin/` tree (#607 acceptance criterion 3).
AGENT_FACING_DOCS = (CLAUDE_MD, REPO / "README.md", *sorted(REPO.glob("skills/**/*.md")))

# A doc may legitimately *name* a path in prose (e.g. "test_cli.py does not
# exist"); only lines that tell an agent to RUN something must resolve. The path
# body allows `/` and `*` so a subdirectory or glob form cannot pass vacuously.
_PYTEST_PATH = re.compile(r"tests/[\w*/-]*test_[\w*-]+\.py(?:::(\w+))?")

# A test module cited in PROSE as the thing that enforces a claim, and the
# retirement phrasing that is the opposite claim about one. The phrasing is the
# only one the tree uses, and it does not need to be exhaustive: a retirement a
# reader states some other way is not recognised as a denial, so the module it
# names stays a positive citation and must resolve. The unknown case fails
# CLOSED.
_MODULE_CITATION = re.compile(r"`((?:tests/)?test_\w+\.py)`")
_MODULE_RETIRED = re.compile(r"do(?:es)? not exist", re.I)


def _doc_text(path: Path = CLAUDE_MD) -> str:
    """Never let a missing doc abort collection -- `test_claude_md_exists` owns that."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _runnable_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "pytest" in line]


def _short_id(line: str) -> str:
    stripped = line.strip()
    return stripped[:48] + "..." if len(stripped) > 48 else stripped


def test_claude_md_exists():
    """The other guards degrade to no-ops without this one failing loudly."""
    assert CLAUDE_MD.is_file(), f"{CLAUDE_MD} is missing"


def test_agents_md_mirrors_claude_md():
    """A second physical copy is how AGENTS.md drifted onto a ghost tree (#607).

    A checkout without symlink support (Windows without `core.symlinks`, or an
    archive that flattens them) materialises the `120000` entry as a small plain
    file holding the link text. That is git's doing, not a drifted copy, and the
    anti-drift guarantee still holds: a real copy carries the whole document and
    never the bare target name.
    """
    assert AGENTS_MD.exists(), "root AGENTS.md is missing"
    if AGENTS_MD.is_symlink():
        assert os.readlink(AGENTS_MD) == "CLAUDE.md"
        assert AGENTS_MD.resolve() == CLAUDE_MD.resolve()
        return
    # A bounded head keeps a drifted copy's failure to one short line instead of
    # dumping the whole canonical document into the assertion diff.
    head = AGENTS_MD.read_text(encoding="utf-8").strip()[:64]
    assert head == "CLAUDE.md", (
        "AGENTS.md must be a symlink to CLAUDE.md (or its unexpanded link text on a "
        "checkout without symlink support), never a second copy -- a copy drifts"
    )


@pytest.mark.parametrize("doc", AGENT_FACING_DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_agent_docs_do_not_point_at_the_ghost_plugin_tree(doc: Path):
    """`plugin/bn_agent_bridge/` holds only stale bytecode; sources are in src/."""
    assert "plugin/bn_agent_bridge" not in _doc_text(doc), doc


@pytest.mark.parametrize("line", _runnable_lines(_doc_text()), ids=_short_id)
def test_documented_pytest_invocations_resolve(line: str):
    """Every `uv run pytest <path>` in the docs must name a real module/test."""
    for match in _PYTEST_PATH.finditer(line):
        rel = match.group(0).split("::", 1)[0]
        if "*" in rel:
            assert any(REPO.glob(rel)), f"{rel} matches nothing: {line!r}"
            continue
        module = REPO / rel
        assert module.is_file(), f"{rel} does not exist: {line!r}"
        test_id = match.group(1)
        if test_id is not None:
            source = module.read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(test_id)}\(", source, re.M), (
                f"{test_id} is not defined in {rel}: {line!r}"
            )


def _bullet(prefix: str) -> str:
    """The single `prefix` bullet with its indented continuation lines folded in.

    Rewrapping a bullet must not change what it claims, so a wrapped-away status
    or module name is neither a false alarm nor a silent pass.
    """
    lines = _doc_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(starts) == 1, f"expected one {prefix!r} bullet, found {len(starts)}"
    bullet = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith((" ", "\t")) or not line.strip():
            break
        bullet.append(line)
    # Collapse the fold seams too: a bullet wrapped between a name and its
    # parenthetical must read the same as the one-line form.
    return " ".join(" ".join(bullet).split())


# Shared with the kernel exit-error coverage in test_bn_kernel.py.
_CLI_PACKAGE = REPO / "src" / "bn"
# The two ways this package ends a process with a status of its own.
_EXIT_CALLS = frozenset({"exit", "_exit", "SystemExit"})


def _result_positions(expr: ast.expr) -> list[ast.expr]:
    """The expressions *expr* can itself evaluate to.

    A conditional and a boolean operator both return one of their operands, so
    `return 7 if cond else 0` returns the literal 7 -- a widening the round-17
    lens smuggled past a sweep that only looked at `return <constant>`. Nothing
    else is descended into: an int inside a call argument or a subscript is not
    a value the function returns, and treating it as one flagged a page limit
    and a rounding precision.
    """
    if isinstance(expr, ast.IfExp):
        return [*_result_positions(expr.body), *_result_positions(expr.orelse)]
    if isinstance(expr, ast.BoolOp):
        return [pos for value in expr.values for pos in _result_positions(value)]
    return [expr]


def _int_literal(expr: ast.expr | None) -> int | None:
    # `type(...) is int` and not isinstance: `True` is an int subclass, and
    # `return True` is not a claim about an exit code.
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return expr.value
    return None


def _codes_the_cli_can_return() -> dict[int, list[str]]:
    """Every status this package can hand the shell as a LITERAL.

    A total population -- every module under `src/bn`, no entry-point list and
    no exemption -- because a sweep that reads only the functions someone
    remembered is how a widened contract got past this file in the first place.

    Two mechanisms, because the package uses both: a `return`, whose value
    reaches the shell through `main()`; and a status handed to `sys.exit` /
    `os._exit` / `SystemExit`, which is how argparse's refusal and
    `__main__.py` deliver theirs. A return counts only when EVERY position it
    can evaluate to is an int literal, so `return None if spilling else 100` --
    a page limit, not a code -- is not one, and `return 7 if cond else 0` is.

    What a literal sweep cannot see, stated rather than implied: a status
    computed at run time (`sys.exit(main())`, `parser.exit(status)`) is not a
    literal anywhere, and the code it carries is the one its callee returns,
    which this population already holds. A code-less `exit()` is 0 by
    definition, and 0 is documented.
    """
    codes: dict[int, list[str]] = {}

    def record(code: int, path: Path, node: ast.AST) -> None:
        codes.setdefault(code, []).append(f"{path.name}:{node.lineno}")

    for path in sorted(_CLI_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Return) and node.value is not None:
                literals = [_int_literal(pos) for pos in _result_positions(node.value)]
                if literals and all(code is not None for code in literals):
                    for code in literals:
                        record(code, path, node)
                continue
            call = node.exc if isinstance(node, ast.Raise) else node
            if not isinstance(call, ast.Call):
                continue
            name = (call.func.attr if isinstance(call.func, ast.Attribute)
                    else getattr(call.func, "id", None))
            if name not in _EXIT_CALLS:
                continue
            for argument in call.args[:1]:
                code = _int_literal(argument)
                if code is not None:
                    record(code, path, node)
    return codes


def test_mutation_exit_classifications():
    """Failures outrank unmeasured results, and measured successes stay zero."""
    from bn.cli import _mutation_exit_code

    verified = {"success": True, "committed": True, "results": [{"status": "verified"}]}
    all_noop = {"success": True, "committed": True, "results": [{"status": "noop"}]}
    unmeasured = {"success": True, "committed": True, "results": []}
    failing_unmeasured = {"success": False, "committed": False, "results": []}
    refused_up_front = {"success": False, "committed": False,
                        "results": [{"status": "invalid_request"}]}
    failed_at_apply = {"success": False, "committed": False,
                       "results": [{"status": "verification_failed"}]}
    assert _mutation_exit_code(verified, _mutation_summary) == 0
    # "a measured all-`noop` is 0"
    assert _mutation_exit_code(all_noop, _mutation_summary) == 0
    # "4 = ... `measured: false`"
    assert _mutation_exit_code(unmeasured, _mutation_summary) == 4
    # "A failure still wins over 4 (3 before 4)"
    assert _mutation_exit_code(failing_unmeasured, _mutation_summary) == 3
    # "the refusal is exit 3 whether it was raised up front or during apply"
    assert _mutation_exit_code(refused_up_front, _mutation_summary) == 3
    assert _mutation_exit_code(failed_at_apply, _mutation_summary) == 3
    # "an op that counts through its own registered summary (`go rename`) is 0
    #  only while those counters read" -- the REAL summary, because it DERIVES
    #  `measured` from those reads and a stub asserting `measured: True` makes
    #  the claim unfalsifiable at the point it stops being true.
    go = {"kind": "go_rename", "preview": False, "success": True,
          "committed": True, "results": []}
    assert _mutation_exit_code({**go, "go_renamed_candidates": 7,
                                "go_committed_count": 7, "go_verified_count": 7,
                                "go_failed_count": 0}, _go_rename_summary) == 0
    # "...one whose counter arrives unreadable is `measured: false` and 4"
    assert _mutation_exit_code({**go, "go_renamed_candidates": "many"},
                               _go_rename_summary) == 4
    # "So a single refused field is never 2: it still yields a verdict, and that
    #  verdict is 3 or 4." -- the failing half of the same arrangement.
    assert _mutation_exit_code(
        {**go, "success": False, "committed": False,
         "go_renamed_candidates": "many",
         "results": [{"status": "verification_failed"}]}, _go_rename_summary) == 3
    # "a mutation result this CLI cannot classify AT ALL ... so no verdict could
    #  be derived" -- and that really is what 2 is reserved for.
    from bn.transport import BridgeError
    with pytest.raises(BridgeError, match="could not classify"):
        _mutation_exit_code(["verified"], _mutation_summary)


def test_unmeasured_summaries_disclose_the_missing_measurement():
    from bn.formatters import _render_mutation_summary_text

    nothing_to_count = _mutation_summary({"success": True, "committed": True,
                                          "results": []})
    counter_refused = _go_rename_summary(
        {"kind": "go_rename", "preview": False, "success": True,
         "committed": True, "go_renamed_candidates": "many", "results": []})
    for field, summary in (("results[]", nothing_to_count),
                           ("go_renamed_candidates", counter_refused)):
        assert summary["measured"] is False
        assert summary["dirty_after"] is True
        assert field in summary["first_error"]
        assert summary["first_error"] in _render_mutation_summary_text(summary)


def test_every_test_module_an_agent_doc_names_exists():
    """A doc that cites a guard by filename is telling an agent where the rule
    lives; a filename that resolves to nothing sends them hunting.

    `_PYTEST_PATH` already covers lines that tell an agent to RUN something.
    This is the other half -- a module named in PROSE as the thing that enforces
    a claim -- which a round-21 lens falsified by replacing the cited filename
    with one that does not exist, with every guard green.
    """
    # A doc may also cite a module to say it is GONE ("`tests/test_cli.py` and
    # `tests/test_bridge.py` do not exist"), which is the opposite claim and is
    # asserted by test_test_layout_bullet_retired_modules_stay_gone. Read the
    # negation off the document rather than hardcoding the two names, so a third
    # retirement needs no edit here.
    #
    # The negation binds to the NAME it denies, through the one implementation
    # of that rule (shared with the flag guards in
    # tests/test_skill_reference_drift.py -- round 23 found this same defect in
    # both, because the repair had been applied to one file and not its
    # sibling). Read per DOCUMENT one retirement sentence excused every citation
    # in the file; read per SENTENCE, a single sentence carrying both a
    # retirement and a POSITIVE citation of an invented module was skipped
    # whole -- "enforced by `tests/test_invented_guard.py` because
    # `tests/test_cli.py` does not exist" stayed green. Bound to the name, the
    # subordinator ends the denied run and the invented module is still a claim.
    from test_skill_reference_drift import bound_absence_claims

    cited = set()
    for doc in AGENT_FACING_DOCS:
        where = str(doc.relative_to(REPO))
        for sentence in re.split(r"(?<=[.;:])\s+|\n", _doc_text(doc)):
            names = {match[1] for match in _MODULE_CITATION.finditer(sentence)}
            if not names:
                continue
            retired = bound_absence_claims(sentence, _MODULE_CITATION,
                                           _MODULE_RETIRED)
            cited.update((where, name) for name in names - retired)
    assert cited, "no agent-facing doc cites a test module, so this cell proves nothing"
    missing = sorted(f"{where}: {name}" for where, name in cited
                     if not (REPO / name).is_file()
                     and not (REPO / "tests" / name).is_file())
    assert not missing, (
        "these agent-facing docs cite a test module that does not exist, so a "
        f"reader sent to the rule finds nothing: {missing}"
    )


def test_cli_read_and_mutation_failure_boundaries(monkeypatch, tmp_path):
    """Exercise slice, reachability, fan-out and mutation failure boundaries."""
    import bn.cli
    from bn.cli import _mutation_exit_code
    from bn.transport import BridgeError

    # "A START past the last line ... exits non-zero with a stderr diagnostic
    #  (not a `//` comment on stdout)"
    body = {"name": "fn", "address": 0x401000, "hlil": "int fn() {\n  return 0;\n}\n"}
    monkeypatch.setattr(bn.cli, "send_request",
                        lambda op, **kwargs: {"ok": True, "result": body})
    argv = ["decompile", "fn", "--target", "active", "--lines"]
    with contextlib.redirect_stdout(io.StringIO()) as out, \
            contextlib.redirect_stderr(io.StringIO()) as err:
        past_end = bn.cli.main([*argv, "999:1000"])
    assert past_end != 0, "an out-of-range slice must not read as a result"
    assert "beyond the last line" in err.getvalue(), err.getvalue()
    assert "//" not in out.getvalue(), out.getvalue()
    # ...and the claim is only meaningful because an IN-range slice is 0.
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        assert bn.cli.main([*argv, "1:2"]) == 0

    # "Exit code is reachability-only: nonzero if any probed instance is
    #  unreachable, zero otherwise (staleness ... never affect the exit code;
    #  zero registered instances is not a failure)."
    install_dir, source_dir = tmp_path / "install", tmp_path / "source"
    for directory in (install_dir, source_dir):
        directory.mkdir()
        (directory / "bridge.py").write_text("print('b')\n", encoding="utf-8")
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)

    def instance(pid: int, name: str):
        return type("FakeInstance", (), {
            "pid": pid, "socket_path": tmp_path / f"{name}.sock",
            "plugin_version": bn.cli.VERSION,
            "started_at": "2026-03-09T00:00:00+00:00", "instance_id": name,
        })()

    live, dead, stale = instance(1, "live"), instance(2, "dead"), instance(3, "stale")

    def reply(inst, op, params=None, target=None, **_kwargs):
        if inst is dead:
            raise OSError("connection refused")
        version = "0.0.0-ancient" if inst is stale else bn.cli.VERSION
        return {"ok": True, "result": {"plugin_version": version,
                                       "plugin_build_id": "b", "targets": []}}

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", reply)

    def doctor(instances: list) -> int:
        monkeypatch.setattr(bn.cli, "list_instances", lambda: instances)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return bn.cli.main(["doctor", "--format", "json"])

    assert doctor([live, dead]) != 0, "an unreachable instance must be nonzero"
    assert doctor([live]) == 0, "every instance reachable must be zero"
    # "staleness fields are informational and never affect the exit code"
    assert doctor([live, stale]) == 0, "a stale but reachable bridge is not a failure"
    # "zero registered instances is not a failure"
    assert doctor([]) == 0, "no registered instance must not read as unreachable"

    # "the command exits non-zero only if **every** result failed" -- a fan-out
    #  whose rows PARTLY failed is still 0, which is the half a scripted
    #  consumer depends on and the half a "nonzero on failure" reading gets
    #  backwards.
    import types

    fanned = [types.SimpleNamespace(instance_id="a"),
              types.SimpleNamespace(instance_id="b")]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: fanned)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda inst: inst.instance_id)
    monkeypatch.setattr(bn.cli, "_resolve_target", lambda args, **kwargs: "active")

    def fanout(failing: set[str]) -> int:
        def send(op, *, params=None, target=None, instance_id=None, **kwargs):
            if instance_id in failing:
                raise BridgeError("down")
            return {"ok": True, "result": {"kind": "sections", "items": [], "total": 0}}
        monkeypatch.setattr(bn.cli, "send_request", send)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return bn.cli.main(["sections", "--all-instances", "--format", "json"])

    assert fanout({"a", "b"}) != 0, "a fan-out where every row failed must be nonzero"
    assert fanout({"a"}) == 0, "a partly-failed fan-out must not read as a failure"
    assert fanout(set()) == 0

    # "`reverted`, which is **not** a failure and does not affect the exit code"
    reverted_sibling = {"success": True, "committed": True,
                        "results": [{"status": "verified"}, {"status": "reverted"}]}
    assert _mutation_exit_code(reverted_sibling, _mutation_summary) == 0, (
        "CLAUDE.md says a `reverted` sibling does not affect the exit code"
    )

    # "If verification fails, the CLI returns a nonzero exit code"
    verification_failed = {"success": False, "committed": False,
                           "results": [{"status": "verification_failed"}]}
    assert _mutation_exit_code(verification_failed, _mutation_summary) != 0, (
        "README says a failed verification is nonzero"
    )

def _unserializable_reply() -> dict[str, object]:
    """An ok, classifiable mutation reply that `json.dumps` cannot encode.

    Depth is a property of the RESPONSE, and it is what makes the divergence
    observable: the default status line prints named fields and never walks
    this, while a machine format has to encode all of it."""
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(100_000):
        child: dict[str, object] = {}
        cursor["next"] = child
        cursor = child
    return {"success": True, "committed": True,
            "results": [{"status": "verified"}], "detail": nested}


def _exit_code_of_a_run(argv: list[str], result: dict[str, object] | None) -> int:
    """What `bn` really EXITS with for *argv* against a fixed bridge reply.

    Output failures happen after classification; unreachable bridges never
    reach it. Run the CLI so these cases exercise their actual boundaries.

    *result* of None means the bridge could not be reached at all.
    """
    import bn.cli
    from bn.transport import BridgeError

    def unreachable(op, **kwargs):
        raise BridgeError(
            "Failed to contact Binary Ninja bridge; no instance is listening")

    reply = {"ok": True, "result": result}
    original = bn.cli.send_request
    bn.cli.send_request = (unreachable if result is None
                           else (lambda op, **kwargs: reply))
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return bn.cli.main(argv)
    finally:
        bn.cli.send_request = original


_RENAME_ARGV = ["symbol", "rename", "--target", "active", "sub_401000", "x"]


def _exit_code_for(scenario: str) -> int:
    """What the CLI really returns for each runtime scenario."""
    from bn.cli import _mutation_exit_code

    if scenario == "undeliverable-output":
        # A destination that cannot exist: the parent path is a FILE. Chosen over
        # a read-only directory because it needs no mode change to set up and
        # none to clean up.
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp, "not-a-directory")
            blocker.write_text("", encoding="utf-8")
            return _exit_code_of_a_run(
                [*_RENAME_ARGV, "--out", str(blocker / "detail.json")],
                {"success": True, "committed": True,
                 "results": [{"status": "verified"}]})
    if scenario == "unserializable-reply-as-text":
        return _exit_code_of_a_run(_RENAME_ARGV, _unserializable_reply())
    if scenario == "unreachable-bridge":
        return _exit_code_of_a_run(_RENAME_ARGV, None)

    # The `own-summary` scenarios run the REAL `_go_rename_summary`, not a stub
    # that hardcodes `measured: True`. A stub made the doc's claim unfalsifiable
    # at exactly the point it stopped being true: the summary DERIVES `measured`
    # from whether its six counters read, so "an op that counts through its own
    # summary stays measured" is a claim about that derivation, and a fake that
    # asserts the answer quotes the document back at itself. Both sides of the
    # derivation are scenarios, because the doc now states both.
    go = {"kind": "go_rename", "preview": False, "success": True,
          "committed": True, "results": []}
    shapes = {
        "verified": ({"success": True, "committed": True,
                      "results": [{"status": "verified"}]}, _mutation_summary),
        "failing": ({"success": False, "committed": False,
                     "results": [{"status": "invalid_request"}]}, _mutation_summary),
        "unmeasured": ({"success": True, "committed": True, "results": []},
                       _mutation_summary),
        # The same unmeasured shape on the other kind of call: a PREVIEW, so it
        # reverted and never committed. The claim is that the verdict keys on
        # `measured: false` and not on the kind of call, so the preview flags
        # are really set rather than reusing the live scenario's reply.
        "unmeasured-preview": ({"success": True, "committed": False,
                                "preview": True, "results": []},
                               _mutation_summary),
        "own-summary": ({**go, "go_renamed_candidates": 7, "go_committed_count": 7,
                         "go_verified_count": 7, "go_failed_count": 0},
                        _go_rename_summary),
        # The same op, the same summary, one counter in a shape no count reads
        # out of: refused and disclosed by name, so a verdict still exists and
        # it is an unmeasured one.
        "own-summary-refused": ({**go, "go_renamed_candidates": "many"},
                                _go_rename_summary),
    }
    result, summary = shapes[scenario]
    return _mutation_exit_code(result, summary)


@pytest.mark.parametrize("scenario,expected", [
    ("verified", 0),
    ("failing", 3),
    ("unmeasured", 4),
    ("unmeasured-preview", 4),
    ("own-summary", 0),
    ("own-summary-refused", 4),
    ("undeliverable-output", 2),
    ("unserializable-reply-as-text", 0),
    ("unreachable-bridge", 2),
])
def test_mutation_scenario_exit_codes(scenario: str, expected: int):
    assert _exit_code_for(scenario) == expected


@pytest.mark.parametrize("status", ["verified", "noop", *sorted(FAILED_MUTATION_STATUSES)])
def test_mutation_row_status_controls_exit_code(status: str):
    from bn.cli import _mutation_exit_code

    result = {"success": True, "committed": True, "results": [{"status": status}]}
    expected = 0 if status in {"verified", "noop"} else 3
    assert _mutation_exit_code(result, _mutation_summary) == expected


@pytest.mark.parametrize("group", ("cli", "bridge"))
def test_test_layout_bullet_concerns_still_resolve(group: str):
    """#614's defect was the bullet naming modules a split had renamed away, so
    every concern it lists must still resolve to a real module.

    Deliberately one-directional: the ticket asked for the *pattern* rather than
    an exhaustive list, so a newly added `test_cli_*.py` is covered by the glob
    and does not have to be enumerated here.
    """
    bullet = _bullet("- Test files mirror source")
    listed = re.search(rf"`test_{group}_\*\.py` \(([^)]+)\)", bullet)
    assert listed, f"the bullet no longer lists the test_{group}_* concerns"
    # A concern token never contains a space, so dropping every space inside the
    # parenthetical survives a hard wrap that broke one across two lines.
    concerns = listed.group(1).replace(" ", "").split("/")
    missing = [c for c in concerns if not (REPO / "tests" / f"test_{group}_{c}.py").is_file()]
    assert not missing, f"test_{group}_* concerns naming no module: {missing}"


def test_test_layout_bullet_retired_modules_stay_gone():
    """The bullet closes by naming the two monolith modules that no longer exist
    (#614). If one comes back, that sentence is the stale claim."""
    bullet = _bullet("- Test files mirror source")
    retired = re.search(r"((?:`tests/test_\w+\.py`(?:,| and )?)+) do not exist", bullet)
    assert retired, "the bullet must keep naming the monolith modules the split retired"
    names = re.findall(r"`(tests/test_\w+\.py)`", retired.group(1))
    assert len(names) == 2, f"expected the two monolith names, got {names}"
    resurrected = [n for n in names if (REPO / n).exists()]
    assert not resurrected, f"the bullet says these do not exist, but they do: {resurrected}"


def test_cli_layout_names_every_command_module():
    """The CLI Layout list reads as an inventory, so a handler module absent from
    it sends an agent adding a command to the wrong file (or to a new one).
    """
    modules = sorted(
        path.name
        for path in (REPO / "src" / "bn" / "commands").glob("*.py")
        if path.name != "__init__.py"
    )
    assert modules, "no src/bn/commands/*.py modules found"
    text = _doc_text()
    missing = [name for name in modules if f"`{name}`" not in text]
    assert not missing, f"command modules missing from the CLI Layout list: {missing}"


CLI_LAYOUT_HEADING = "### CLI Layout (`src/bn/`)"

# Top-level modules no agent reaches for by name: the package initializer and
# the `python -m bn` entry point. Named explicitly so a *new* module fails the
# guard until it is listed (or deliberately added here), never silently omitted.
CLI_LAYOUT_INTERNAL_MODULES = frozenset({"__init__.py", "__main__.py"})


def _claude_md_section(heading: str) -> str:
    """One `### ` section body, from `heading` to the next `### ` heading."""
    lines = _doc_text().splitlines()
    assert heading in lines, f"{heading!r} heading is missing from CLAUDE.md"
    body = lines[lines.index(heading) + 1:]
    for index, line in enumerate(body):
        if line.startswith("### "):
            return "\n".join(body[:index])
    return "\n".join(body)


def test_cli_layout_names_every_top_level_module():
    """#721: `client.py`, `proc_identity.py` and `version.py` were missing from
    the CLI Layout inventory, so an agent adding a paged read command had no
    pointer to the page-aggregation helper and re-implemented paging in the
    handler.

    Only an INVENTORY ENTRY counts: a `- ` list item, or the section's lead
    sentence (which is how `cli.py` is introduced). Two looser readings both
    let a module leave the inventory with the guard still green, so neither is
    used: "named anywhere in the section" is satisfied by the `version.py`
    bullet's aside about `paths.py` and by the symlink sentence below the list,
    and "leading backticked token of any line" is satisfied by replacing a
    bullet with a bare prose mention that documents nothing. One directory glob,
    and any deliberate omission named in `CLI_LAYOUT_INTERNAL_MODULES`.
    """
    lines = _claude_md_section(CLI_LAYOUT_HEADING).splitlines()
    lead = next((line for line in lines if line.strip()), "")
    entries = [line for line in lines if line.startswith("- ")]
    assert entries, f"the {CLI_LAYOUT_HEADING} section has no `- ` inventory entries"
    introduced = {
        match.group(1)
        for line in [lead, *entries]
        if (match := re.match(r"(?:- )?`([^`]+)`", line))
    }
    modules = sorted(
        path.name
        for path in (REPO / "src" / "bn").glob("*.py")
        if path.name not in CLI_LAYOUT_INTERNAL_MODULES
    )
    assert modules, "no src/bn/*.py modules found"
    missing = [name for name in modules if name not in introduced]
    assert not missing, f"top-level modules missing from the CLI Layout list: {missing}"
    stale = sorted(name for name in CLI_LAYOUT_INTERNAL_MODULES
                   if not (REPO / "src" / "bn" / name).is_file())
    assert not stale, (
        "CLI_LAYOUT_INTERNAL_MODULES skips modules src/bn no longer has: "
        f"{stale}. Every other exemption in this file stale-fails; a skip list "
        "that outlives its subject silently shrinks what this guard covers"
    )


# The Conventions bullet in CLAUDE.md is the rule this guard enforces, so the
# guard READS it instead of restating it: a rule mirrored into Python here is a
# rule the doc can quietly contradict, which is the exact defect #826 reports.
# Three properties make the reading load-bearing. The bullet is located by
# SECTION, so a correct copy elsewhere cannot vouch for a stale one where an
# agent actually looks. Each clause is matched on the RULE it states rather
# than on a keyword, so a bullet that names a different naming scheme cannot
# stand in for the one the tree follows. And each clause is required exactly
# when the registry holds a command of that shape, so a clause can neither be
# struck from the doc nor outlive its subject. The reach ends at polarity: a
# clause carrying its own wording intact but negated around it still reads as
# stated, which no rewrite of a rule into a different rule does.
_CONVENTIONS_HEADING = "## Conventions"
_HANDLER_BULLET_PREFIX = "- Command handlers are named"
_HANDLER_BULLET_PATTERN = "`_<group>_<subcommand>()`"
_DOC_TOPLEVEL_CLAUSE = re.compile(r"top-level command keeps its bare verb")
_DOC_ALIAS_CLAUSE = re.compile(r"alias keeps the name of the path it aliases")
_DOC_HANDLER_EXCEPTION = re.compile(r"`([a-z][\w -]*)` is `(_\w+)`")
# An example handler name is a claim about the tree exactly like the rule it
# illustrates, so the citations are checked against the registry too.
_DOC_CITED_HANDLER = re.compile(r"`(_\w+)`")


class _DocumentedRules(NamedTuple):
    grouped: bool
    top_level: bool
    alias: bool
    exceptions: dict[str, str]
    cited: set[str]


def _prefixed(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(_HANDLER_BULLET_PREFIX)]


def _handler_convention_bullets(text: str) -> tuple[list[str], list[str]]:
    """Handler-naming bullets under `## Conventions`, and the same anywhere in *text*.

    Returning both is what lets the guard reject a SECOND copy: taking the
    first prefixed line in the file would let a correct bullet inserted
    anywhere else stand in for a stale one in the section an agent reads.
    """
    lines = text.splitlines()
    if _CONVENTIONS_HEADING not in lines:
        return [], _prefixed(lines)
    section: list[str] = []
    for line in lines[lines.index(_CONVENTIONS_HEADING) + 1:]:
        if line.startswith("## "):
            break
        section.append(line)
    return _prefixed(section), _prefixed(lines)


def _documented_rules(bullet: str) -> _DocumentedRules:
    """What the bullet states, clause by clause."""
    return _DocumentedRules(
        grouped=_HANDLER_BULLET_PATTERN in bullet,
        top_level=bool(_DOC_TOPLEVEL_CLAUSE.search(bullet)),
        alias=bool(_DOC_ALIAS_CLAUSE.search(bullet)),
        exceptions=dict(_DOC_HANDLER_EXCEPTION.findall(bullet)),
        cited=set(_DOC_CITED_HANDLER.findall(bullet)),
    )


def _clause_coverage(
    handlers: dict[str, str], rules: _DocumentedRules
) -> tuple[list[str], list[str]]:
    """(command shapes the registry has that the bullet no longer documents,
    rules the bullet documents that the registry holds no subject for).

    A module-level helper rather than a table inside the test, so dropping a
    row is itself pinnable: a row silently deleted would leave that command
    shape unchecked with every other guard green, which is the reported defect
    one level up.
    """
    clauses = (
        ("grouped `_<group>_<subcommand>` commands",
         any(" " in path for path in handlers), rules.grouped),
        ("top-level bare-verb commands",
         any(" " not in path for path in handlers), rules.top_level),
        ("aliased commands",
         len(set(handlers.values())) < len(handlers), rules.alias),
    )
    return (
        [label for label, present, stated in clauses if present and not stated],
        [label for label, present, stated in clauses if stated and not present],
    )


def _expected_handler_name(path: str) -> str:
    """`_<group>_<subcommand>` for a grouped path, the bare verb for a top-level one."""
    return "_" + "_".join(word.replace("-", "_") for word in path.split())


def _naming_violations(
    handlers: dict[str, str], documented: dict[str, str], alias_rule: bool
) -> list[str]:
    """Registered paths whose handler name the documented rule does not allow.

    The alias exemption is granted to the ALIASING path only -- a path whose
    handler is named after a DIFFERENT path of the same handler, which is what
    "keeps the name of the path it aliases" means. Exempting every path of a
    multi-path handler instead would let an aliased handler be renamed to
    anything at all and stay green.
    """
    paths_by_handler: dict[str, set[str]] = {}
    for path, name in handlers.items():
        paths_by_handler.setdefault(name, set()).add(path)
    violations = []
    for path, name in sorted(handlers.items()):
        expected = _expected_handler_name(path)
        if name == expected:
            continue
        if documented.get(path) == name:
            continue
        if alias_rule and any(
            _expected_handler_name(other) == name
            for other in paths_by_handler[name] - {path}
        ):
            continue
        violations.append(f"{path!r} -> {name!r}, expected {expected!r}")
    return violations


def _registered_handler_names() -> dict[str, str]:
    """Registered command path -> handler function name, from the live registry.

    Importing `bn.commands` is what POPULATES `_COMMANDS` -- the `@command`
    decorators run at import -- so the registry is empty without it and a sweep
    over it would check nothing.
    """
    import bn.cli

    import bn.commands  # noqa: F401 -- importing the package registers its commands

    return {
        " ".join(spec["path"]): spec["handler"].__name__ for spec in bn.cli._COMMANDS
    }


def test_command_handlers_follow_the_documented_naming_convention():
    """The Conventions bullet names the handler of each command, so a handler
    that does not follow it is an agent's grep for the implementation coming
    back empty.

    Every clause is read out of the bullet and applied to the registry, so the
    two halves of the claim fail together: strike a clause and the command
    shape it covers is left undocumented; rename a handler and the registry
    stops matching what the bullet says.
    """
    handlers = _registered_handler_names()
    assert handlers, "the @command registry is empty, so nothing was checked"
    in_section, anywhere = _handler_convention_bullets(_doc_text())
    assert len(in_section) == 1, (
        f"CLAUDE.md's {_CONVENTIONS_HEADING} section must carry exactly one "
        f"handler-naming bullet -- this guard reads the rule from it, so "
        f"without it nothing is enforced. Found: {in_section}"
    )
    assert anywhere == in_section, (
        "a second handler-naming bullet sits outside "
        f"{_CONVENTIONS_HEADING}: {[line for line in anywhere if line not in in_section]}. "
        "Two copies is how the section an agent reads goes stale while a guard "
        "reads the other one"
    )
    strays = {
        path.relative_to(REPO).as_posix(): copies
        for path in AGENT_FACING_DOCS
        if path != CLAUDE_MD and (copies := _prefixed(_doc_text(path).splitlines()))
    }
    assert not strays, (
        f"another agent-facing doc states the handler-naming rule: {strays}. The "
        "rule lives in one place so an agent cannot read a contradicting second "
        "copy, and only CLAUDE.md's copy is checked against the registry"
    )
    bullet = in_section[0]
    rules = _documented_rules(bullet)
    undocumented, stale = _clause_coverage(handlers, rules)
    assert not undocumented, (
        f"the registry has {undocumented} but the bullet no longer states the "
        f"rule for them, so nothing checks their handler names. The bullet "
        f"reads: {bullet!r}"
    )
    assert not stale, (
        f"the bullet documents {stale} while the registry holds none: a rule is "
        "only true while it has a subject"
    )
    violations = _naming_violations(handlers, rules.exceptions, rules.alias)
    assert not violations, (
        "these registered commands break the naming rule CLAUDE.md's Conventions "
        f"bullet documents: {violations}. The bullet reads: {bullet!r}"
    )
    for path, name in sorted(rules.exceptions.items()):
        assert handlers.get(path) == name, (
            f"the bullet's naming exception {path!r} is `{name}`, but the registry "
            f"has {handlers.get(path)!r}: an exemption is only true while it has "
            "a subject"
        )
    unknown = sorted(rules.cited - set(handlers.values()))
    assert not unknown, (
        f"the bullet cites handler names the registry does not have: {unknown}. "
        "An example is a claim about the tree exactly like the rule it illustrates"
    )


def test_the_alias_exemption_covers_only_the_aliasing_path():
    """#826: the exemption belongs to an alias PATH, not to its handler.

    Pinned against a synthetic registry because the live one holds no
    counter-example: while every alias is well-named, "exempt the whole handler"
    and "exempt only the aliasing path" agree. They disagree the moment an
    aliased handler is renamed, and that is the case the wider form waved
    through for both of the registry's aliases.
    """
    aliased = {"symbol rename": "_symbol_rename", "rename": "_symbol_rename"}
    assert _naming_violations(aliased, {}, True) == []
    renamed = {"symbol rename": "_rename_symbol", "rename": "_rename_symbol"}
    assert _naming_violations(renamed, {}, True) == [
        "'rename' -> '_rename_symbol', expected '_rename'",
        "'symbol rename' -> '_rename_symbol', expected '_symbol_rename'",
    ], "a multi-path handler that follows the rule on NO path must not be exempt"
    assert _naming_violations(aliased, {}, False) == [
        "'rename' -> '_symbol_rename', expected '_rename'"
    ], "the alias exemption must come from the documented bullet, not the guard"


# A synthetic bullet stating all three rules, and for each clause the exact
# text to strike plus the DIFFERENT rules it is rewritten into, each written
# in the clause's OWN vocabulary. The rewrites are the load-bearing half: the
# clause's words survive them, so only a matcher reading the RULE reports the
# clause absent. A matcher loosened to any word the rewrites carry reports the
# clause present and reds -- which is why a clause needs one rewrite per word
# form it can be reduced to (`alias` and `aliases` are two). Words that occur
# ONLY inside the operative phrase cannot be covered this way: a rewrite
# carrying them would state the rule again. Loosening a matcher to one of
# those stays green, which is bounded -- the doc must still spell the rule out
# for the guard to read it as stated.
_PARSER_BULLET = (
    "- Command handlers are named `_<group>_<subcommand>()` (e.g., "
    "`_function_list`); a top-level command keeps its bare verb "
    "(`_decompile`) — `help` is `_help_index`, and a command that is an "
    "alias keeps the name of the path it aliases (`rename` is `_symbol_rename`)"
)
_CLAUSE_PINS = (
    ("grouped", "`_<group>_<subcommand>()`",
     ("`<group>::<subcommand>()`", "`_<group>-<subcommand>()`")),
    ("top_level",
     "; a top-level command keeps its bare verb (`_decompile`)",
     ("; a top-level command is prefixed `_top_` (`_top_decompile`)",
      "; a top-level command keeps its group prefix, never its bare verb"
      " (`_top_decompile`)")),
    ("alias",
     ", and a command that is an alias keeps the name of the path it aliases "
     "(`rename` is `_symbol_rename`)",
     (", and an alias command is named `_alias_<path>` (`_alias_rename`)",
      ", and command aliases are forbidden in this CLI")),
)


def test_the_bullet_parser_reads_each_rule_not_a_keyword():
    """#826: a clause must be recognised by the RULE it states.

    `alias` appearing in a clause that says something else about aliases used
    to license the alias exemption, and the top-level clause was not read at
    all, so striking it from the doc was invisible to every guard. Every
    clause is pinned by striking it and by rewriting it into other rules that
    keep its vocabulary, so a matcher cannot loosen towards a keyword -- in
    either the clause's singular or its plural form -- and stay green.
    """
    rules = _documented_rules(_PARSER_BULLET)
    assert (rules.grouped, rules.top_level, rules.alias) == (True, True, True)
    assert rules.exceptions == {"help": "_help_index", "rename": "_symbol_rename"}
    assert rules.cited == {
        "_function_list", "_decompile", "_help_index", "_symbol_rename"
    }
    assert {field for field, _, _ in _CLAUSE_PINS} == {
        field for field in _DocumentedRules._fields
        if field not in ("exceptions", "cited")
    }, (
        "every clause `_documented_rules` reports needs an anti-loosening pin; "
        "a row dropped from _CLAUSE_PINS silently removes one"
    )
    for field, clause, other_rules in _CLAUSE_PINS:
        assert clause in _PARSER_BULLET, (
            f"the {field} pin no longer quotes its own clause, so all of its "
            "mutations are no-ops"
        )
        struck = _documented_rules(_PARSER_BULLET.replace(clause, ""))
        assert getattr(struck, field) is False, (
            f"striking the {field} clause must stop it counting as documented"
        )
        for other_rule in other_rules:
            reworded = _documented_rules(_PARSER_BULLET.replace(clause, other_rule))
            assert getattr(reworded, field) is False, (
                f"a bullet stating a DIFFERENT {field} rule must not read as "
                f"stating this one just by reusing its words: {other_rule!r}"
            )


def test_a_naming_exception_is_a_path_handler_pair_not_any_backticked_pair():
    """#826: the Conventions list is dense with backticked prose.

    A matcher that harvested every `x` is `y` pair would grant an exemption
    the bullet never gave, and the exempted path's handler could then be named
    anything at all.
    """
    with_prose = _PARSER_BULLET + "; the default `--format` is `text` for reads"
    assert _documented_rules(with_prose).exceptions == {
        "help": "_help_index", "rename": "_symbol_rename"
    }, "only a `path` is `_handler` pair is a naming exception"


def test_every_command_shape_in_the_registry_must_be_a_documented_rule():
    """#826: the coverage wiring, not just the parser that feeds it.

    A clause row dropped from `_clause_coverage` would leave that command
    shape unchecked while every other guard stayed green -- the same defect
    one level up from an unread clause.
    """
    every_shape = {"function list": "_function_list", "decompile": "_decompile",
                   "symbol rename": "_symbol_rename", "rename": "_symbol_rename"}
    all_stated = _DocumentedRules(True, True, True, {}, set())
    assert _clause_coverage(every_shape, all_stated) == ([], [])
    for field in ("grouped", "top_level", "alias"):
        undocumented, stale = _clause_coverage(
            every_shape, all_stated._replace(**{field: False})
        )
        assert len(undocumented) == 1 and not stale, (
            f"a registry holding every command shape must report the {field} "
            "rule undocumented the moment the bullet stops stating it"
        )
    undocumented, stale = _clause_coverage(
        {"function list": "_function_list"}, all_stated
    )
    assert undocumented == [] and len(stale) == 2, (
        "a rule the registry has no subject for must be reported stale"
    )


def test_the_guard_reads_the_conventions_section_not_the_first_match():
    """#826: a correct copy elsewhere must not vouch for a stale section bullet."""
    doc = (
        "# Title\n\n- Command handlers are named `correct`\n\n"
        "## Conventions\n\n- Command handlers are named `stale`\n\n"
        "## Next\n\n- Command handlers are named `other`\n"
    )
    in_section, anywhere = _handler_convention_bullets(doc)
    assert in_section == ["- Command handlers are named `stale`"], (
        "the bullet must be taken from the section an agent reads"
    )
    assert len(anywhere) == 3, "the doc-wide sweep must see every copy"
    assert _handler_convention_bullets(
        "# Title\n\n- Command handlers are named `x`\n"
    ) == ([], ["- Command handlers are named `x`"]), (
        "a bullet with no Conventions section at all is not in the section"
    )


# The lock class is declared at the `@op` decorator, so the declarations are the
# ground truth for what `lock="none"` actually covers.
_NONE_LOCK_OP = re.compile(r'@op\(\s*"([^"]+)"\s*,\s*lock="none"')

# The `none` semantics an agent reads before choosing a lock class. Four cuts
# were escaped before this one. Three were proxies for meaning rather than pins
# on the text: "some op name anywhere in the paragraph" (satisfied by an
# incidental parenthetical after the refutation was deleted), `"not" in lead`
# (satisfied by "as a general note"), and a word-boundary `\bnot\b` at every
# mention (satisfied by "is not merely ...", which AFFIRMS the false reading).
# The fourth was a pin whose ACCOUNTING had a boundary: it covered the paragraph
# from the first `none` claim onward, so a sentence inserted BEFORE that marker,
# in the same paragraph, taught the falsehood with every pin green.
#
# What replaced the five clause-by-clause pins and the character sweep over them
# (#732) is one presence check plus the two cells that execute code. A proxy for
# meaning is escapable by construction -- but so is a transcription of the
# sentence, which reds on every rephrase while saying nothing about whether the
# paragraph is TRUE, and the claim underneath it is proven by
# test_the_lock_model_counterexample_is_a_really_stateful_none_op. What is worth
# keeping is the claim-deletion tripwire, over the whole paragraph.
#
# The two `none` ops that really are pure signals: they set an event and must
# stay deliverable while a write op holds the lock. Everything else declared
# `none` runs real work on the view and takes whatever lock it needs itself.
SIGNAL_ONLY_NONE_OPS = frozenset({"shutdown", "cancel_request"})

LOCK_MODEL_SENTENCE_PREFIX = "`op_registry.py` is the single source of truth"

# The claim this paragraph exists to refute, as WORDS rather than as a quoted
# string. Accounted for in every doc: moving it somewhere else -- or dropping
# the quotation marks -- is the same defect as affirming it here.
LOCK_MODEL_FALSE_CLAIM = "touches no BN state"

# The refutation as a pattern, because two other cells read it: the
# cross-document check below, and
# test_the_lock_model_counterexample_is_a_really_stateful_none_op, which pulls
# the counterexample ops out of the match. Deliberately loose on either side of
# the negation -- a rewrap or a rephrase must not red -- and tight ON the
# negation, because "is *not* merely <claim>" affirms the claim and a cut that
# accepted any nearby `not` was escaped exactly that way. Bounded by the
# clause's own semicolon, so the match cannot run past the sentence it quotes.
_LOCK_MODEL_REFUTATION = (
    rf"`lock=\"none\"`[^;]*\*not\* \"{LOCK_MODEL_FALSE_CLAIM}\"[^;]*")


def _lock_model_region() -> str:
    """The WHOLE lock-model paragraph -- every line of it, to the blank line.

    Two earlier cuts each stopped short of the paragraph's real edge: one began
    at the first `none` claim, leaving the registry prose on the same line
    unaccounted; the next took that whole LINE and called it the paragraph, so
    an affirmation on the NEXT line -- same paragraph, no blank line between --
    sat outside the accounting again. A markdown paragraph ends at a blank
    line, so that is where this region ends.
    """
    lines = _doc_text().splitlines()
    start = next((at for at, line in enumerate(lines)
                  if line.startswith(LOCK_MODEL_SENTENCE_PREFIX)), None)
    assert start is not None, (
        f"the lock-model paragraph ({LOCK_MODEL_SENTENCE_PREFIX}...) is gone")
    end = start
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    return "\n".join(lines[start:end + 1])


def test_the_lock_model_region_still_states_the_lock_model():
    """The claim-deletion tripwire, which is the only thing the five wording
    pins added over the cells that execute code (#732): the paragraph still
    names the three lock classes an agent chooses between, and still refutes the
    reading that makes them dangerous.

    `none` means the DISPATCHER holds no lock and the op body self-manages;
    reading it as "touches no BN state" makes an agent either write-lock an op
    that must not be, or ship a stateful op taking no lock. Scoped to the WHOLE
    paragraph, because two earlier cuts stopped short of its real edge and an
    affirmation sat just outside the accounting both times.
    """
    region = _lock_model_region()
    missing = [lock_class for lock_class in ("read", "write", "none")
               if f'"{lock_class}"' not in region]
    assert not missing, (
        f"the lock-model paragraph no longer names the {missing} lock class(es) "
        f"an agent has to choose between: {region}"
    )
    assert re.search(_LOCK_MODEL_REFUTATION, region), (
        'the lock-model paragraph no longer refutes the "touches no BN state" '
        f'reading of `lock="none"`, so an agent can read it that way: {region}'
    )


@pytest.mark.parametrize("doc", AGENT_FACING_DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_agent_doc_states_the_false_lock_reading_unrefuted(doc: Path):
    """...and the paragraph is not the only place the false reading could be
    taught. Every occurrence of the claim in every agent-facing doc must sit
    inside the refutation cell's own span, so moving it to another document --
    or to another paragraph of this one -- fails the same way as affirming it
    here.

    Matched as WORDS, not as the quoted string round 8 pinned: the same reading
    taught without the quotation marks, or across a line wrap, is the same
    reading. And over EVERY agent-facing doc, not the six that state an exit
    code: the docstring said "every agent-facing doc" while the
    parametrization said six of eleven, and the lock model is not an
    exit-code concern -- an agent reads the other five the same way.
    """
    text = _doc_text(doc)
    claimed = bytearray(len(text))
    for match in re.finditer(_LOCK_MODEL_REFUTATION, text):
        claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    claim = r"\s+".join(re.escape(word) for word in LOCK_MODEL_FALSE_CLAIM.split())
    unrefuted = [
        f"line {text.count(chr(10), 0, match.start()) + 1}: "
        f"...{text[max(0, match.start() - 70):match.end() + 20]}..."
        for match in re.finditer(claim, text, re.I)
        if not claimed[match.start()]
    ]
    assert not unrefuted, (
        f"{doc} states {LOCK_MODEL_FALSE_CLAIM!r} outside the refutation that "
        "exists to correct it, so an agent reading it there ships a stateful op "
        f"taking no lock: {unrefuted}"
    )


def test_the_lock_model_counterexample_is_a_really_stateful_none_op():
    """...and the pinned text is only worth pinning while it is TRUE: the
    counterexample ops must really be declared `lock="none"` and really do work
    on the view, so the guard cannot outlive a re-classing. The moment every
    `none` op is a pure signal, the premise fails and this must be rewritten
    rather than quietly kept."""
    bridge = (REPO / "src" / "bn_agent_bridge" / "bridge.py").read_text(encoding="utf-8")
    none_ops = set(_NONE_LOCK_OP.findall(bridge))
    assert none_ops, 'no lock="none" op declarations found; this guard is unchecked'
    # SIGNAL_ONLY_NONE_OPS is an exemption, so it is asserted live like the
    # others: an op renamed away would silently shrink `stateful` and make the
    # counterexample requirement easier to satisfy rather than failing.
    retired = sorted(SIGNAL_ONLY_NONE_OPS - none_ops)
    assert not retired, (
        'SIGNAL_ONLY_NONE_OPS exempts ops that are no longer declared '
        f'lock="none": {retired}; a stale exemption weakens this guard silently'
    )
    stateful = sorted(none_ops - SIGNAL_ONLY_NONE_OPS)
    assert stateful, (
        'every lock="none" op is now a pure signal, so the "touches no BN state" '
        "reading would be true again -- rewrite this guard instead of deleting it"
    )
    region = _lock_model_region()
    match = re.search(_LOCK_MODEL_REFUTATION, region)
    assert match, f"the presence check above owns this; region reads: {region}"
    named = [op for op in stateful if f"`{op}`" in match.group(0)]
    assert named, (
        'the refutation must name a stateful lock="none" op as its '
        f"counterexample; it reads {match.group(0)!r} and the stateful ops are "
        f"{stateful}"
    )


def test_documented_python_requirement_matches_pyproject():
    """`Requires Python >= X.Y` is the first claim an agent acts on, and a stale
    floor sends it to install the wrong interpreter."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject)
    assert declared, "requires-python is missing from pyproject.toml"
    spec = declared.group(1).replace(" ", "")
    floor = re.fullmatch(r">=(\d+\.\d+)", spec)
    assert floor, f"unexpected requires-python form {spec!r}; update this guard"
    assert f"Python >= {floor.group(1)}" in _doc_text(), (
        f"CLAUDE.md must document Python >= {floor.group(1)} (pyproject: {spec})"
    )
