"""Guards that keep the agent-instruction docs honest.

`CLAUDE.md` is the canonical agent-instruction file and root `AGENTS.md` is a
tracked symlink to it, so an agent reading either one sees the same tree layout
(#607). These tests pin the invariants that actually misled agents before:
a second physical copy that drifts, a bridge path that no longer exists, a
`uv run pytest` line naming a module or test id that was renamed away (#614),
and an exit-code list that silently falls behind `FAILED_MUTATION_STATUSES`.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

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


def test_exit_code_3_lists_every_failed_mutation_status():
    """Adding a failure status must not silently leave the docs understating exit 3.

    Scoped to the exit-code bullet itself: a status named somewhere else in the
    file (the Mutation Verification prose) does not tell an agent reading the
    exit-code contract that the status maps to 3, which was exactly #614's gap.
    """
    text = _bullet("- Exit codes:")
    missing = sorted(s for s in FAILED_MUTATION_STATUSES if f"`{s}`" not in text)
    assert not missing, f"statuses missing from the exit-code bullet: {missing}"


def test_exit_code_bullet_documents_every_code_the_cli_can_return():
    """#715 widened the contract from 0/1/2/3 to 0/1/2/3/4 and added the
    unclassifiable-result case to exit 2. Both new clauses arrived UNGUARDED:
    deleting either left this module green, in the very PR whose purpose is
    making the docs provably match the code -- the same shape as the defect #721
    exists to fix.

    Each claim is tied to something executable rather than pinned as prose: the
    `measured` key really is what the compact summary emits for a result with no
    rows to count, and the CLI really does have a malformed-result rule that
    turns an unreadable response into a `BridgeError`.
    """
    bullet = _bullet("- Exit codes:")
    missing = [code for code in ("0", "1", "2", "3", "4")
               if not re.search(rf"(?:^|[ ,]){code} = ", bullet)]
    assert not missing, f"exit codes missing from the exit-code bullet: {missing}"

    # exit 4's stated subject: the summary key an unmeasurable result really sets.
    unmeasured = _mutation_summary({"success": True, "committed": True})
    assert unmeasured["measured"] is False, unmeasured
    assert "`measured: false`" in bullet.lower(), (
        "the exit-4 clause must name the `measured: false` summary key it is "
        f"keyed on, not merely the number: {bullet}"
    )

    # exit 2's stated subject: the malformed-result rule really exists.
    from bn.cli import _MALFORMED_RESULT_ERRORS
    assert _MALFORMED_RESULT_ERRORS, "the CLI has no malformed-result rule to document"
    assert "classify" in bullet, (
        "the exit-2 clause must say a result this CLI cannot classify is 2, "
        f"which is what the malformed-result rule does: {bullet}"
    )


# The CLI package, which IS the population below: the bridge lives outside it,
# so an integer this package hands the shell is either a process exit code or a
# bare magic number that ought to be a named constant.
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


def test_no_code_the_cli_can_return_is_undocumented():
    """The direction the bullet's inventory was missing.

    `test_exit_code_bullet_documents_every_code_the_cli_can_return` reds when
    the DOCUMENT narrows -- a deleted clause -- and stayed green when the CODE
    widened: round 16's falsification lens injected `return 7` into a registered
    handler and left this module, `tests/test_cli_mutation.py` and
    `tests/test_skill_reference_drift.py` all green; round 17's got past the
    repair twice more, with `return 7 if False else 0` and with `sys.exit(7)`.
    A contract that documents the codes someone wrote down rather than the codes
    the CLI can produce is exactly the #721 defect this file exists to catch, so
    the two sets are asserted EQUAL over both exit mechanisms. The population's
    own limit is stated on the helper rather than implied away.
    """
    documented = {int(code) for code
                  in re.findall(r"(?:^|[ ,—-]) ?(\d+) = ", _bullet("- Exit codes:"))}
    assert documented, "the exit-code bullet no longer lists a single code"
    returned = _codes_the_cli_can_return()
    undocumented = {code: returned[code] for code in sorted(set(returned) - documented)}
    assert set(returned) == documented, (
        "the exit-code bullet and the CLI package disagree about which codes "
        f"exist. Documented but produced nowhere: {sorted(documented - set(returned))}. "
        f"Produced but undocumented: {undocumented}. An undocumented status here "
        "is either an exit code the contract has not caught up with, or a magic "
        "number that should be a named constant rather than a bare literal"
    )


# EVERY clause of the exit-code bullet, each with a cell of its own. An earlier
# cut asserted the five code numbers, the literal `measured: false` and the word
# `classify`, so four clauses this contract rests on could be deleted with the
# module still green -- including #716's only documented rule. Deleting any
# clause below reds exactly the cell that names it.
#
# Each pattern spans its WHOLE clause, not a distinctive prefix, because the
# coverage half underneath measures the characters these patterns claim: a
# prefix leaves the rest of its own clause unclaimed, which is where a second
# claim can be smuggled in beside a guarded one.
_EXIT_CODE_CLAUSES = (
    # The bullet's own label. A cell of its own so the coverage half underneath
    # can demand that NOTHING is unclaimed: the old "three claim-words or more"
    # floor existed only to let the label through, and `; 5 = reserved` and
    # `is 0 or 2` both shipped under it.
    ("bullet-label", r"^- Exit codes:"),
    ("0-success", r"\b0 = success"),
    ("1-cli-side-handler-error",
     r"\b1 = CLI-side handler error \(e\.g\. partial `session start` failure\)"),
    ("2-bridge-error",
     r"\b2 = `BridgeError` \(transport failures and bridge-side errors,"),
    ("2-read-or-resolver-status",
     r"including a status in `FAILED_MUTATION_STATUSES` on a read/resolver call,"),
    ("2-unclassifiable-mutation",
     r"and a mutation result this CLI cannot classify AT ALL — malformed or newer "
     r"than the CLI, so no verdict could be derived\)"),
    ("3-mutation-status",
     r"\b3 = a `_mutate`-marked call whose status is `verification_failed`, `unsupported`, "
     r"`invalid_request`, `rollback_failed`, or `internal_error`"),
    ("3-refused-up-front-or-at-apply",
     r"— the refusal is exit 3 whether it was raised up front or during apply, "
     r"never 2 for a mutation —"),
    ("4-unmeasured",
     r"\b4 = a `_mutate`-marked call whose compact summary reports `measured: false`, "
     r"i\.e\. the counts could not be derived:"),
    ("4-nothing-to-count",
     r"the op returned no `results\[\]` rows to derive counts from AND registered no "
     r"summary of its own to count with,"),
    ("4-a-counted-field-was-refused",
     r"or a field the summary counts FROM arrived in a shape no value reads out of, "
     r"so it was refused and disclosed by name rather than fabricated as a zero "
     r"\(#715/#619\);"),
    ("4-own-summary-is-0-only-while-its-counters-read",
     r"an op that counts through its own registered summary \(`go rename`\) is 0 only "
     r"while those counters read — one whose counter arrives unreadable is "
     r"`measured: false` and 4 like any other unmeasured run\."),
    ("2-one-refused-field-still-classifies",
     r"So a single refused field is never 2: it still yields a verdict, and that "
     r"verdict is 3 or 4\."),
    ("4-failure-wins-and-all-noop-is-zero",
     r"A failure still wins over 4 \(3 before 4\), and a measured all-`noop` is 0"),
)


@pytest.mark.parametrize("clause,pattern", _EXIT_CODE_CLAUSES,
                         ids=[name for name, _ in _EXIT_CODE_CLAUSES])
def test_every_clause_of_the_exit_code_bullet_is_guarded(clause: str, pattern: str):
    """One cell per clause: this is the test that goes red when a clause is
    deleted, which is the whole reason the bullet is worth writing."""
    bullet = _bullet("- Exit codes:")
    assert re.search(pattern, bullet), (
        f"the exit-code bullet no longer states the {clause!r} clause, so the "
        f"documented contract and the code can now disagree silently: {bullet}"
    )


def _unclaimed_runs(text: str, patterns: list[str]) -> list[str]:
    """The runs of *text* no pattern matches.

    The one implementation of "what is accounted for", used by the exit-code
    bullet and the lock-model paragraph: two copies of an accounting rule are
    two rules, and they drift.
    """
    claimed = bytearray(len(text))
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.M):
            claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    runs: list[str] = []
    run: list[str] = []
    for index, char in enumerate(text):
        if claimed[index]:
            if run:
                runs.append("".join(run))
                run = []
        else:
            run.append(char)
    if run:
        runs.append("".join(run))
    return runs


def test_the_exit_code_bullet_carries_no_unguarded_clause():
    """The other half: the cells above must account for the WHOLE bullet.

    Presence alone is satisfiable by a list that has stopped keeping up, which
    is exactly how the exit-4 body, the own-summary clause and #716's refusal
    rule shipped unguarded. The first cut split the bullet on its clause
    punctuation -- and a claim smuggled INSIDE a parenthetical, or comma-joined
    to a guarded clause, is not a clause boundary, so it stayed invisible. The
    second measured characters but only reported an unclaimed run of three
    claim-words or more, so `; 5 = reserved` (publishing a code the CLI cannot
    return) and `or 2` appended to the all-`noop` clause both stayed green.

    So the accounting is now total: every character of the bullet is claimed by
    a cell, and an unclaimed run carrying ANY word character is a claim nothing
    guards -- however short, and wherever in the bullet it was put.
    """
    bullet = _bullet("- Exit codes:")
    prose = [text for text in _unclaimed_runs(bullet, [p for _, p in _EXIT_CODE_CLAUSES])
             if re.search(r"\w", text)]
    assert not prose, (
        "these runs of the exit-code bullet are claimed by no cell in "
        f"_EXIT_CODE_CLAUSES, so they could be changed or deleted with this "
        f"module green: {prose}"
    )


def test_the_exit_code_bullet_clauses_are_what_the_cli_actually_does():
    """...and the clauses are pinned as TEXT above, so this is the half that
    makes them true rather than merely quoted: each behavioural claim is executed
    against the helper the bullet describes."""
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


# The exit-code contract has ONE implementation and several places that state
# it. Round 7 counted three deciders -- CLAUDE.md's bullet, README.md's status
# list, and the mutation reference an agent opens to do mutation work -- and
# only the bullet was guarded, so flipping this file's exit 4 to 0 and
# README's exit 3 to 2 left every doc test green while the two documents an
# agent actually reads stated the opposite of the shipped code.
#
# So the code decides and the documents echo: each echo is pinned with its
# NUMBER captured, and the number is compared against what
# `_mutation_exit_code` returns for that scenario. A flipped digit fails here;
# a deleted sentence fails here; and neither can be argued about, because the
# expected value is executed rather than quoted.
#
# Pinning the statements someone NOTICED is the same fail-open list one radius
# out, so the count is not hand-maintained either: every exit-code claim in
# every agent-facing doc is detected mechanically below, and a claim no cell
# accounts for fails. That audit found four more deciders of this one contract
# -- a fourth full 0/1/2/3/4 list in the mutation reference, the failure rule
# restated in CLAUDE.md outside its own bullet, the read-vs-mutate split stated
# twice more, and the session-restart codes in the runtime reference.
_EXIT_CODE_ECHOES = (
    ("README.md", "mutation-failure",
     r"puts a mutation at exit code `(?P<code>\d)`", "failing"),
    ("README.md", "unmeasured-is-separate",
     r"Exit `(?P<code>\d)` is the separate \"unverifiable\" case", "unmeasured"),
    ("skills/bn/reference/mutating.md", "refusal-up-front-or-at-apply",
     r"it is a mutation failure: exit (?P<code>\d) on any mutation command", "failing"),
    ("skills/bn/reference/mutating.md", "unmeasured-live-success",
     r"An unmeasured \*\*live\*\* success also changes the exit code: it is \*\*`(?P<code>\d)`\*\*",
     "unmeasured"),
    # Both `own-summary` echoes use `\s+` between words: these references are
    # hand-wrapped prose, and a claim that reds on a re-wrap is bookkeeping
    # rather than a guard.
    ("skills/bn/reference/mutating.md", "own-summary-counters-read",
     r"so\s+a\s+clean\s+run\s+whose\s+counters\s+read\s+is\s+exit\s+`(?P<code>\d)`",
     "own-summary"),
    # The other side of that derivation, stated by the same file. Its own
    # summary is not a measurement GUARANTEE -- it is a different measurement
    # SOURCE, and an unreadable counter leaves this op exactly as unmeasured as
    # an empty `results[]` leaves every other one.
    ("skills/bn/reference/mutating.md", "own-summary-counter-unreadable",
     r"a\s+counter\s+that\s+arrives\s+unreadable\s+is\s+disclosed\s+by\s+name\s+and\s+"
     r"the\s+run\s+is\s+the\s+unmeasured\s+`(?P<code>\d)`", "own-summary-refused"),
    ("skills/bn/reference/mutating.md", "unsupported-op-kind",
     r"either way exit (?P<code>\d), and a", "failing"),
    # Round 15 measured the "No combination changes the exit code" sentence FALSE
    # (#716): the same reply exits 0 on the default status line and 2 when the
    # requested output cannot be delivered. The sentence now states the
    # divergence, and these two echoes measure BOTH of its digits by running the
    # command -- the divergence lives strictly after the classification, so
    # asking `_mutation_exit_code` for it would quote the doc back at itself.
    ("skills/bn/reference/mutating.md", "undeliverable-output-is-2",
     r"reports the documented `(?P<code>\d)` instead", "undeliverable-output"),
    ("skills/bn/reference/mutating.md", "the-status-line-never-walks-it",
     r"verified reply exits `(?P<code>\d)` there", "unserializable-reply-as-text"),
    # Round 17: the two clauses that replaced a refuted sentence, measured rather
    # than pinned as present. The refusal one deliberately reuses the `failing`
    # scenario, so the doc's "a refusal is exit 3" is compared against what the
    # classifier really returns for a refused mutation.
    ("skills/bn/reference/mutating.md", "an-unreachable-bridge-is-2",
     r"path a (?P<code>\d) is also the code for", "unreachable-bridge"),
    ("skills/bn/reference/mutating.md", "a-refusal-is-not-one-of-those",
     r"on a mutation a refusal is exit (?P<code>\d)", "failing"),
)


# Exit-code claims whose expected value is not a mutation outcome the helper can
# be asked for (a read-path refusal, a session-restart code, the reference's own
# full list). Pinned as LITERAL text INCLUDING the digit, so a flip stops
# matching and reds -- there is nothing to capture and compare against.
_EXIT_CODE_PINS = (
    ("CLAUDE.md", "failure-statuses-put-the-run-at-3",
     "and any of them puts the run at exit 3."),
    ("README.md", "own-counters-success-is-0",
     "so it exits `0` on success like any other."),
    ("README.md", "measured-all-noop-is-0",
     "A measured all-`noop` is `0` too."),
    # Found by the per-number accounting: both sit on the same line as two
    # already-pinned statements, so a line-level sweep read them as covered.
    ("README.md", "unmeasured-does-not-outrank-a-failure",
     "`4` is distinct from `3` (a failure, which still wins when both apply) "
     "and from `0`,"),
    ("README.md", "an-unmeasured-preview-is-4-too",
     "a `--preview` that comes back unmeasured is `4` as well"),
    ("skills/bn/reference/mutating.md", "unknown-op-kind-is-3",
     "An unknown op kind is `unsupported` and likewise exit 3."),
    ("skills/bn/reference/mutating.md", "mutation-3-read-or-resolver-2",
     "a status in `FAILED_MUTATION_STATUSES` on a mutation call is exit 3 "
     "(#625/#716); only the same status escaping a read/resolver op is exit 2."),
    ("skills/bn/reference/mutating.md", "exit-2-covers-the-rest",
     "Exit 2 still covers everything on this path that is *not* one of those "
     "statuses:"),
    ("skills/bn/reference/mutating.md", "the-references-own-full-list",
     "0 ok / 1 a CLI-side handler error / 2 bridge or request error (including a\n"
     "response this CLI cannot classify at all) / 3 a mutation status `verification_failed`,\n"
     "`unsupported`, `invalid_request`, `rollback_failed`, or `internal_error` / 4 an\n"
     "unmeasured success"),
    # The boundary of exit 2 on this path, stated in the file an agent opens to
    # do mutation work. Measured, not merely present, by
    # tests/test_cli_mutation.py::test_the_cases_the_reference_lists_for_exit_2_really_are_2,
    # which runs a reply carrying one refused field and asserts it is not 2.
    ("skills/bn/reference/mutating.md", "one-refused-field-is-not-a-2",
     "And a reply carrying ONE field this CLI cannot\n"
     "read: that field is refused and disclosed by name, a verdict is still derived\n"
     "from the rest, and the run exits 3 or 4 accordingly."),
    # The direction of the one divergence, which is the part a $?-only consumer
    # depends on: an output failure can only ever replace the code with 2.
    # Behaviourally pinned by
    # tests/test_cli_mutation.py::test_no_output_flag_can_turn_a_nonzero_classification_into_zero.
    ("skills/bn/reference/mutating.md", "an-output-failure-only-moves-toward-2",
     "undeliverable output replaces the code with 2 and can never turn a failed or an\n"
     "unmeasured mutation into a clean zero."),
    # Round 16's falsification lens refuted the sentence that stood here ("a 2 on
    # a mutation says the requested output did not arrive, not that the write did
    # not land": an invalid flag VALUE exits 2 at parse time with no request
    # sent), and round 17's refuted the replacement's first clause ("2 is also
    # this path's code for a refused request": on a mutation a refusal is 3,
    # which this same file states 43 lines earlier). The two clauses that state a
    # code are now ECHOES, measured below, because a clause parked in a pin is
    # only asserted to be PRESENT; this pin holds the part with nothing to
    # measure, and tests/test_cli_mutation.py::
    # test_the_cases_the_reference_lists_for_exit_2_really_are_2 measures each
    # case it names.
    ("skills/bn/reference/mutating.md", "a-2-alone-does-not-say-the-write-landed",
     "So a 2 alone does not tell you whether the write landed"),
    ("skills/bn/reference/reading.md", "bounded-slice-is-a-success",
     "a provably-bounded constant length (a success, exit 0)"),
    # Found by widening the sweep to the whole agent-facing set: this doc states
    # a `bn` exit code and was outside the six docs the sweep used to read, so
    # the claim could have drifted from the code with every guard green.
    ("skills/bn-kernel/SKILL.md", "limit-zero-is-refused-at-parse-time",
     "rejected at parse time (exit 2), because the bridge would reject a zero "
     "limit and"),
    ("skills/bn/reference/runtime.md", "restart-of-an-unreachable-bridge-is-1",
     "this way exits **1** rather than 0 whenever the teardown and respawn succeed"),
    ("skills/bn/reference/runtime.md", "restart-that-cannot-signal-is-2",
     "the restart refuses to signal and exits **2** instead"),
)

# The DOCUMENT SET was the last enumerated population left in this accounting,
# and it failed the same way every enumerated population in this PR failed: six
# docs were swept because six were the ones a reviewer had found stating an exit
# code, and the reason was PROSE -- "every doc an agent reads for the contract".
# It was not even a theoretical gap: `skills/bn-kernel/SKILL.md` states a real
# `bn` exit code (a `--limit 0` refusal) and sat outside the sweep, and a false
# contract appended to `skills/bn-vr/SKILL.md` and `skills/bn/agents/bn-re.md`
# left all 101 tests in this module and the drift module green. So the sweep is
# the whole agent-facing set, DERIVED from the same glob the ghost-tree and
# lock-reading properties quantify over: a doc joins the population by EXISTING,
# not by being listed.
EXIT_CODE_DOCS = tuple(str(doc.relative_to(REPO)) for doc in AGENT_FACING_DOCS)

# Round 8 replaced a hand-listed echo table with a sweep, and the sweep was a
# RECOGNISER: a bare 0-4 within 120 characters after the word `exit`. Both
# round 9 lenses walked past it in one line -- "a failed mutation returns `2`
# to the shell", "a refused op returns three", "Exit 5 is reserved ... exit 9
# for a bridge that refuses to start" -- because the phrasing avoided the word,
# or the digit was outside the range, or the number was spelled out.
#
# A recogniser for natural language is escapable by construction, so there is
# almost no vocabulary here: the population is every line of every agent-facing
# doc that carries a NUMBER or names an EXIT, and each is accounted for exactly
# one of two ways -- a cell above pins it to what the CLI really returns, or the
# ledger below records that it is not a claim.
#
# What this does NOT catch, stated in full because the previous wording claimed
# otherwise ("in ANY phrasing"):
#
#   * a contract with no numeral, no number word and no form of the word "exit"
#     -- "a mutation the bridge refuses before apply exits with the bridge-error
#     code" was caught only once `exit\w*` joined the alphabet, and "the shell
#     status of a refused mutation is the bridge-error status" still is not;
#   * a code written in hex (`exit 0x4`) or as a number word past `hundred`.
#
# Both are implausible phrasings for a doc, which is the only reason they are
# accepted rather than paid for: admitting `0x<hex>` would put every example
# address in the population. The population is therefore large and STATED,
# rather than total and claimed -- every widening of it so far came from someone
# writing the escape down, which is why the escapes are written down here.
#
# A decimal is one number, not two (`3.11` must not read as a `3` and an `11`),
# and an issue reference is not a code (`#625`). Everything else counts: a
# trailing period is sentence punctuation, not a decimal point (excluding it
# outright let "a refused op returns three." straight through), a hyphen or
# slash neighbour is still a number (`utf-8`, "exit 2/3", "twenty-five"), and
# the words run past nine so that spelling one out is not a way around the
# digits. A numeral is matched as a RUN of digits: matching one digit at a
# time with a word-character guard on both sides meant "Exit 10" produced no
# token at all, and a claim the population cannot see is a claim the
# accounting cannot account for.
_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
)
# `exit`/`exits`/`exited`/`exit-code`, `$?`, and the two spellings of the same
# idea a shell script uses. A line naming an exit is in the population whether
# or not it carries a digit: the falsification lens stated a whole false
# contract with neither a numeral nor a number word.
_EXIT_WORDS = r"exit\w*|status code|return code|returncode|\$\?"
_NUMBER_TOKEN = re.compile(
    rf"(?<![\w#$])(?<!\d\.)(?:\d+|{_NUMBER_WORDS}|{_EXIT_WORDS})(?![\w])(?!\.\d)",
    re.I)
# Generated, not authored: the fingerprint of every prose line in a fenced doc
# that carries a number and is not pinned by a cell above. Regenerate with
# `_fingerprint(" ".join(line.split()))` over the docs. An entry is a claim
# that this line's number is NOT an exit code; adding one is a deliberate
# statement in a diff, which is the point -- the previous accounting made that
# statement silently, by not matching a pattern.
NON_CLAIM_NUMBER_LINES: dict[str, frozenset[str]] = {
    "CLAUDE.md": frozenset({
        "019b39ab",  # ... uv run pytest ... ... # strict: fail (not skip) if BN is mis
        "0535459a",  # ... On the bridge, register the op with `@op("name", lock="read"
        "06935f28",  # `bn` is an agent-friendly CLI for Binary ... It has two parts: a
        "08df77ee",  # ... is the single source of truth: `@op(name, lock="read"|"write
        "0b9e7173",  # When only one target is open, target-required commands can omit 
        "3f1d7f43",  # ... Add a handler in the appropriate ... module, decorated with 
        "49146e24",  # This file provides guidance to Claude Code ... when working with
        "5369b91c",  # ... Add tests in ... (mirror the source ...
        "5c14e27a",  # `observed` are `{}` when the failure supplied no ... All three k
        "9a7f2e2c",  # ### Two-Process Model
        "9bcf19ed",  # - Test files mirror source, split by concern rather than one mod
        "ad00bd37",  # uv run pytest ... # one module
        "b65d6dff",  # **Line count is not a split criterion ... Split a module only on
        "c25f3873",  # Tests mock the `binaryninja` module — no BN license needed excep
        "d95c6e1f",  # All mutations support `--preview` (apply → capture diffs → rever
    }),
    "README.md": frozenset({
        "0e9e0137",  # Any status above other than ... puts a mutation at exit code ...
        "1e0f43b6",  # bn local retype ... ... float --preview
        "35d15f75",  # bn local rename ... ... speed --preview
        "3d833038",  # `bn function list` and `bn function search` return the full matc
        "66b37d2e",  # Omitting `--target` ... works when exactly one target is ... If 
        "76157733",  # - The CLI discovers a bridge, connects to it, and forwards ... W
        "7d089991",  # When you need counts from BN iterators such as ... materialize t
        "8b8c9c6a",  # You can run several sessions in ... When exactly one live sessio
        "9a80299a",  # Use `--stdin` or `--script` for multiline Python ... Use `--code
        "9cd96091",  # bytes: ...
        "b135db65",  # `bn session start` spawns a `bn-agent` process, registers it und
        "b35752e8",  # `bn callsites` is the direct-call lane for exact return-address 
        "b4a59231",  # Non-preview writes only report success after reading the live BN
        "b78dcc70",  # summary: kind=object ...
        "c5fcc64c",  # - `bn` has two parts:
        "ca1fc685",  # - ... zero-based ordinal for matching callsites in the containin
        "d4db75c0",  # - **Peer-credential enforcement ... Every connection to the brid
        "d55c4454",  # If exactly one ... is open, target-specific commands can omit `-
        "e15482b8",  # Run Python inside the Binary Ninja process for one-off inspectio
        "ee507a32",  # Single-function escape hatch — analyze just one function without
        "f50ed8c8",  # tokens: ...
        "f6f3e5cb",  # ... ...
    }),
    "skills/bn/SKILL.md": frozenset({
        "0580e23a",  # > bn -i ... -t ... xrefs main
        "12038a20",  # > **Parallel ... fan-out agents — HARD ... Sticky pins (`instanc
        "2b531c12",  # - **accumulator ... shift structure** — a size-parse loop that s
        "34ea1441",  # - **loop-invariant bound pointers** — a hoisted fixed limit (`ad
        "5e15be9b",  # - **HLIL can mislead beyond access width — trust `bn ... Pseudo-
        "600daf80",  # > bn -i ... -t ... decompile main
        "6efe0859",  # ## Two gotchas that cause wrong answers
        "8bccdffa",  # - **conditional-compare ... `csel` ... `ccmn` guards** — ... fla
        "94ae359d",  # The full command catalog lives in three files **in this skill's 
        "98945b00",  # One open target: omit ... ... open: pass `-t ... (from `bn targe
        "b5f3f247",  # - **access width** — a byte compare can render full-width, and a
        "c9765e33",  # > OMP sibling task agents also share one retained eval ... They 
        "e8374ea0",  # > bn session start ... --instance-id ... # spawn name (not globa
    }),
    "skills/bn/agents/bn-re.md": frozenset({
        "05ac791d",  # ... **Cheap signature check first ... ... `bn class list --no-st
        "16912b15",  # - **Map:** functions you ... (old → new, one line each, with the
        "3fd2c53a",  # ... table, OR ... non-stub ... constructors
        "6f2dfbee",  # ... **Hidden-code-surface sweep — CONDITIONAL, triggered by evid
        "7247d687",  # one-phrase purpose), structs recovered (name + key fields), and 
        "899545ee",  # ... **Confirm the target is live before anything ... Run `bn tar
        "b9626673",  # ... **Persist, don't ... Apply every rename ... retype ... struc
        "e7d6caa7",  # ... ... You may be dispatched alongside sibling RE agents over
    }),
    "skills/bn/agents/bn-vr.md": frozenset({
        "3b240f3b",  # ... **Parallel-safety ... ... Reads fan out safely; writes seria
        "510c255d",  # ... **MANDATORY sink enumeration + source→sink tracing — this is
        "519384bb",  # ... **Confirm the target is live and pick the ... Run `bn target
        "79cda4e3",  # ... **Adversarially verify every finding before you report ... H
        "800a7391",  # on a stripped ...
        "87c67dbd",  # static dispatch table, or ... non-stub constructors), since stri
        "c251ca40",  # Confirm each bug against `bn disasm` ... `ldrb` vs `ldr` for off
        "db90b2f2",  # ... **Persist context, don't ... Leave your reasoning in the BND
        "eb154184",  # - **Bug class** — buffer overflow ... format string ... integer 
    }),
    "skills/bn/reference/mutating.md": frozenset({
        "0169bb89",  # on a bulk rename a zero is the "nothing changed, don't save" ver
        "0501df9a",  # distinct from ... (a failure — a status in ... which
        "107f9fec",  # | ... | the first failure's explanation, or the unmeasured expla
        "10f5e35f",  # against op ... ... Such a manifest is rejected up front, naming 
        "139a9f88",  # in a write-heavy session (a `proto set` cost ... KB; a ... previ
        "18b47674",  # or — like `go rename`, the one op that reports through its own c
        "207a375e",  # That is ... ... The full audit payload — every per-op diff, `req
        "23f868f0",  # ## ... Bundles
        "2bb6b7aa",  # ### Step ... — save before close
        "2dbbc6bc",  # mutation: committed ... ... ... ... ...
        "3c817542",  # bn data retype ... ... [--preview]
        "3dda57c0",  # Every shipped mutation is measurable one of two ways: it populat
        "3f5376fa",  # which fails safe on its own) before trusting a ... status line a
        "41408e3c",  # line and exit codes are therefore the same as every other mutati
        "441109f0",  # typo in op ... no longer rolls back ... good ...
        "46d085e3",  # `comment ... take the address either positionally (`bn comment s
        "474d7d87",  # | ... ... ... ... ... | derived from `results[]`; ... are `null`
        "48b3a493",  # each call takes exactly one location: an address (positional or 
        "531bd535",  # ### Step ... — live writes are verified
        "560d20bb",  # ("applied but unverifiable"), so a script that only checks `$?` 
        "5a40c289",  # Mutations print a **one-line status summary** by default:
        "5b35cedf",  # ... plus a ... can never verify: op ... would be judged
        "61a586ea",  # | `measured` | **false** when the counts below could not be deri
        "6f8e7c3f",  # - ... — the request was refused: a bad field *value*, a missing 
        "77f8dfa0",  # Two things are NOT in that ... A refusal: on a mutation a refusa
        "82ffca8f",  # pairing, since the two are almost always applied ...
        "87b64a1d",  # It is the one mutation whose bridge result reports the work thro
        "8d2b1a2d",  # Annotations live in the ... Always save before closing — `bn clo
        "9cc9bb4f",  # ### Step ... — preview first
        "9e742d4b",  # | `op` | required fields | one of | interactive equivalent |
        "a29b6d0b",  # Split them across two batches — last-write-wins is not expressib
        "bd21de45",  # The mutation surface is built around a four-step safety loop: **
        "bfb392d4",  # ... KB ... ... tokens), so it is **opt-in**:
        "ca67eb7e",  # measurement source: one that arrived ... A count is only read th
        "cca7f348",  # - **One write per ... Every op is verified against the batch's E
        "cdf328d4",  # ## ... Mutation flow
        "d4b55e60",  # kind of call, so an unmeasured `--preview` is ... as well — ther
        "dbcf469b",  # ### Step ... — read back
        "e1e1f0e1",  # write could not be confirmed instead of reading it as a clean ..
        "f6312d90",  # than answering ... — a fabricated zero is indistinguishable from
        "f83e45cf",  # manifest that writes the same key twice (two ... on one address,
        "fecd543b",  # `measured: true` when those six counters read and agree with the
        "fefcdd8b",  # still wins if both apply) and from ... (a verified or measured a
    }),
    "skills/bn/reference/reading.md": frozenset({
        "0644faaa",  # bn trace main ... --arg ... --interprocedural # IP: follows into
        "06b091d5",  # All three dispatch under the **shared read lock**, so they stay 
        "06f979cb",  # - **"Pointers-to-code" means the target's SECTION is code, not m
        "0c2b8572",  # ## ... Caller-static mapping
        "0c699f6f",  # If you call `bn callsites <callee>` without `--within` ... `--wi
        "163c410d",  # - `bn xrefs` accepts a function name *or* a ... ... Text groups 
        "1d2d1df3",  # - ... in-band row-key ... Row schemas differ **on purpose** — `f
        "1e4db595",  # - `bn data vars --start <addr> --end <addr>` lists the **typed d
        "1e4fc043",  # - **JSON list-command field map (the ... idiom and its ... Most 
        "2004d8ed",  # - `bn evidence ... is a read-locked family that surfaces the **r
        "20900821",  # bn function list [--sort {address|size|name}] [--reverse] [--min
        "224aa0bc",  # bn evidence table <addr> --record-size N --field ... --field ...
        "23940cf6",  # bn disasm <fn> [--lines ... | --count ...
        "29c52aa0",  # bn strings [--query <q>] [--regex] [--min-length ... [--section 
        "2d2e5021",  # struct ... desc = ...
        "3a2dd5dc",  # `--within-file` accepts one identifier (name or hex address) per
        "3e7166e2",  # - `bn class` is the **C++ object-model lens** ... a correlation 
        "3f16152c",  # bn evidence calls <reg-fn> --arg-struct N --field ... --field ..
        "48853422",  # bn class show <Name> # one class: methods, vtable, size, bases, 
        "491d1280",  # - **`bn evidence surface`** enumerates the **hidden code surface
        "4a9d61dc",  # bn evidence function <fn> [--context ... # per-call ABI args ...
        "4acd3a8b",  # `{"kind": <discriminator>, "items": ... "total": N, "offset": ..
        "4c17ee45",  # - ... decompiles carry dead ... ... ... Each call site is preced
        "4cd6e220",  # --field ... --field ... --field ... ...
        "52c4dfff",  # - **JSON envelope contract ... Every collection-returning read e
        "67ce174e",  # bn tag get ... | --function <fn> # tags at one address, or the w
        "69200094",  # bn trace ... ... --arg ... --format json # structured JSON outpu
        "6f2cba1d",  # bn evidence calls ... --arg-struct ... ...
        "71ad85de",  # - **Unconditional (always-unsafe) sinks — no ... no ... A sink w
        "74a30711",  # - ... resolves for a call whose **return value is discarded** — 
        "78b987cf",  # **High-fan-in `total` is monotone, not ... A callsites read stop
        "829c1ed6",  # bn decompile <fn> [--addresses] [--lines ... [--force-analysis] 
        "84b0522c",  # bn taint backward -f <fn> --sink ... # slice a sink's args back 
        "8ee528f5",  # - **Bounded-WRITE sinks — wrapped ... overflows ... A length-pre
        "926c7cae",  # - **`evidence calls <reg-fn> --arg-struct N --field …`** recover
        "93d922fb",  # - `xrefs` → ... each row carrying ... (`code` | `data`), ... (co
        "9625a12f",  # ## ... Read flow
        "a7f1eea5",  # bn dataflow defuse <fn> --var ... # SSA def site + use sites of 
        "a9412200",  # - `bn disasm <function> --lines N:M` is a ... slice of the bridg
        "ab8129a3",  # - `bn trace <fn> <addr> [--arg N] [--interprocedural]` walks **M
        "abd1941b",  # bn xrefs <fn-or-addr> [--limit ...
        "b06ea8ac",  # bn disasm <addr> --linear [N] # linear disasm of N (default ... 
        "b15c995b",  # - **Nested tables are canonical too:** a pointer table embedded 
        "b3400709",  # - **Project-internal wrappers — model them so taint follows them
        "c5e2eebe",  # - `bn function create <address> [--preview]` forces Binary Ninja
        "ceb51492",  # --field ... --field ... --field ...
        "d3e7db45",  # - Two more signals from the same demotion logic: ... true` when 
        "dd81c5c0",  # - **Width-sensitive reads — trust `bn disasm`, not the ... Pseud
        "dfa3a380",  # - **Nothing-found vs incomplete (don't confuse them):** `items: 
        "e190b0f0",  # bn evidence orient # one-shot triage digest under a single read 
        "e2f2e169",  # bn trace main ... --arg ... # intra: stops at call boundary
        "ec1c6720",  # bn taint forward -f <fn> --source ... [--sink-class ... # untrus
        "f0e63d8f",  # - **Addresses in JSON are hex STRINGS**, not integers: `{"addres
        "f760dc5e",  # - **Unpaged ... fixed-window ... presence reads** carry `{kind, 
        "fb0af0ab",  # - **Spilled output is NOT the data ... A heavy `--format json` r
    }),
    "skills/bn/reference/runtime.md": frozenset({
        "0580e23a",  # > bn -i ... -t ... xrefs main
        "0d658e61",  # **Predicting spill ... Two signals let you avoid a wasted full r
        "10113c3e",  # **Spill ... When output exceeds ... ... estimated tokens** ... .
        "178a8928",  # - **Instance:** CLI ... > env ... > sticky > sole live instance 
        "17deb2e4",  # - **No targets ⇒ no `py ... `bn py exec` requires at least one o
        "201b6c48",  # ## ... Output & context
        "20e2e605",  # ... sign=False), ... ...
        "2449caa9",  # > **Global BNDB cache (read-only ... Auto-prefer isn't limited t
        "27da9925",  # ## ... Skill install
        "2b10a25d",  # **Stopping is identity-checked and atomically signalled ... `ses
        "2e17cf8a",  # bn decompile main -i ... -t ... # after the leaf
        "30c54757",  # ## ... Workflow & target selection
        "393df258",  # > bn session start ... --instance-id ... # spawn naming
        "3ecb3444",  # bn session list [-i ... # all running instances, or filter one
        "42663a3b",  # bn xrefs <fn-or-addr> --limit ... # cap text output
        "42c9764f",  # Requests time out after ... by default; override with ... ... ..
        "472980ae",  # bn decompile <fn> --lines ... # ... inclusive; prints ... lines 
        "4c0dc63e",  # ... Pick a target:
        "4c3276d9",  # bn bundle -i ... -t ... function main # between group and leaf (
        "56550fe3",  # ## ... Known quirks
        "5ad3c6c3",  # - `--script <file>` for code on disk; `--code` for true ...
        "5d6983de",  # `bn load <raw>` and `bn session start <raw> ... auto-prefer a si
        "5dea8f09",  # shape, so a polling agent never has to index ... or re-derive te
        "600daf80",  # > bn -i ... -t ... decompile main
        "63414412",  # The `[N]` prefix is the view id; you can pass `-t ... If no brid
        "64bf870d",  # ## ... Python escape hatch
        "6785f03a",  # State lives at ... Project root walks up to the nearest ... (cwd
        "6a2f4c3b",  # ... sign=False), ...
        "6aaa0c2e",  # It checks CLI version, plugin staleness ... ... the Binary Ninja
        "6b121aa2",  # **Private project ... `bn session start` associates the new brid
        "6bcea80f",  # "count": ...
        "78a7e943",  # **Fan-out (`--all-instances` ... ... Whole-target **read survey*
        "7ae58359",  # Identity is `(boot id, pid, process start ... Start times count 
        "7f52ba89",  # verdict, because one verdict over many jobs would be a lie, and 
        "8c11c1cb",  # ... (Optional) Pin sticky defaults — useful for a **single** ...
        "9308d393",  # **Unreachable bridges are hidden, not ... The bridge binds its s
        "959dac82",  # - ... and ... work **before or after** the subcommand, and for t
        "9954584e",  # > **HARD rule for parallel ... fan-out ... Sticky pins are **one
        "a7fc01f4",  # ## ... Sessions & headless
        "aa616d1b",  # ... Discover targets:
        "ac6214d0",  # Blast radius: a bare, path, or `--all` close resolves against **
        "b2734c06",  # `--lines START:END` works on `decompile`, `il`, `disasm`, and `f
        "b35e947b",  # ## ... Troubleshooting
        "b54e57c1",  # - **Threshold override** — set ... ... ... to ... the spill poin
        "bddeeded",  # **Quick-mode capability ... Per-command behavior on a `--quick` 
        "c114407a",  # | `evidence function` | **partial** — reads one function's call 
        "c1803b96",  # - **Near-spill note** — when a read *fits* but lands within ... 
        "ca93b1b0",  # > **`xrefs` text is display-capped (not just ... For a hot symbo
        "cb024404",  # - **`types declare` verification ... The source-parser path hand
        "cfcd9558",  # | `decompile`, `il` | **partial** — render only already-analyzed
        "d1ce60b3",  # bn close [<path>] [-t ... [--all] # close one or explicitly --al
        "d6cfb85b",  # - `target`, `instance` — **provenance**: which target and bridge
        "dd5822d0",  # ... ... sign=False)), ...
        "e0237277",  # When multiple bridge instances exist, flagless `bn load <path>` 
        "e3475983",  # ... sign=False), ...
        "f2a53a19",  # bn -i ... -t ... decompile main # at root (preferred for agents)
    }),
    "skills/bn-kernel/SKILL.md": frozenset({
        "031591a9",  # reports ... and ... keeps the bridge-owned `kind`, `total` and
        "0ab592f9",  # wire this is an internal one-row **probe** at your requested `of
        "1016956a",  # On every reachable exit, close only the exact selector returned 
        "13d5a877",  # ... bn session start ... --instance-id ... --detach
        "19e6cec0",  # ... asks for the schema, not the ... Passing ... to a curated
        "1a60fe9a",  # run, every start and load succeeded with that budget, but two st
        "1b9cffb0",  # band, including on a **zero-hit** page for any pre-declared kind
        "26b229c0",  # Python process safe for sibling task ... A sibling exit ... can 
        "27e9ecca",  # returns only after attempting that exact teardown on every reach
        "292ab6e0",  # ... on the same agent-owned spawn, never a substitute for ...
        "3804218c",  # ... bn session start ... --instance-id ...
        "3a2677e0",  # ... ...
        "3bd6c497",  # spawn budget (for example ... and give the surrounding tool a
        "3ff3109a",  # HLIL and decompilation can distort access width, conditional gua
        "4333a9bd",  # Full loads can take many minutes and each bridge can consume hun
        "501a0d7d",  # This is a programmatic-only ... Wire-level `bn <paged command> -
        "5217ba8e",  # ONE real request and returns no ... The bridge enforces `limit >
        "5b90ca26",  # ... may legitimately be **absent** from a ... envelope: the brid
        "615d3c92",  # polling the exact job ... In a ... dogfood
        "62271b7e",  # applied exactly once as one end-to-end deadline: every page of a
        "69aebef4",  # ... Use `int(row["address"], ... for arithmetic; do not call `he
        "6bb82d86",  # return len(rows), ... "name", "address", "size", ...
        "6dd5e055",  # - `await ... ... `await ... ... ... always return row lists; eve
        "71820845",  # Use this skill for high-volume reads that benefit from OMP's ret
        "742c5bd1",  # more than ... seconds (maximum ... ...
        "74a85e20",  # ... is always normalized to ... A non-zero `offset` means the
        "7e6433f3",  # bn -i ... session status "$JOB" --format json # one job: machine
        "7ed9f73a",  # - `await ... reports offending comment locations; ... is the exp
        "90230d5e",  # Paged reads also require each page to publish an integer `offset
        "92bdec53",  # - `await ... exposes ... ... (the exact `imports` row count), an
        "9e322829",  # The bootstrap is idempotent: rerun it after an eval-kernel ... E
        "9f6fa0eb",  # s = ... target="<target-selector>")
        "a32d2524",  # "you never acquired ownership, on every reachable exit close its
        "a36439c7",  # with ... bn session start <target> --instance-id ...
        "a46e3990",  # - `await ... count=N)` ... `lines=(START, END)` returns an addre
        "b5ecda74",  # - `await ... ... defaults to ... rows to avoid latency cliffs; p
        "b817911f",  # rows = await ...
        "be1fbfeb",  # diagnostic channel for a failure: read the raised ... for ... Th
        "c197cd77",  # A deliberate alternative timeout must be positive; never use ...
        "c515afa2",  # - `await ... ... ... defaults to ... ... A bounded high-fan-in p
        "ccd87225",  # When two or more concurrent children will use bn-kernel, launch 
        "cd848d36",  # await ... ...
        "ce70200b",  # ... — one verdict over many jobs would be a lie — and its items
        "d7159bc2",  # The skill can detect and contain foreign bindings, but it cannot
        "d7d5f3f9",  # ...
        "dadd509e",  # ... ... ...
        "db9deb20",  # zero-row position, that row alone proves more exists at this ...
        "e22c411b",  # bridge, or treat the command as unavailable — do not synthesize 
        "e5563d00",  # "Start your unique headless bridge with the exact ... "
        "e6fb072e",  # large = [row for row in rows if ... ... >= ...
        "e9265826",  # **Bare-decimal addresses, one disclosure ... Every containment-e
        "f23ef97b",  # > **Concurrent sibling task agents:** OMP currently shares one r
        "f9d5322a",  # ... # ...
        "fd3e3a3d",  # the collection already ... The documented ... spelling disables
        "fede3564",  # inherit one eval session and can overwrite ... or kill sibling
        "ffc45d17",  # Native reads are bounded to ... seconds by ... Every curated exp
    }),
    "skills/bn-re/SKILL.md": frozenset({
        "02a15c24",  # ... **Map the C++ type lattice (RTTI ... symbolicated C++ target
        "0a6d25f3",  # bn class show ... # one class: methods, vtable slots, bases, con
        "131433a5",  # > **One-shot sweep: `bn evidence ... It composes this whole sect
        "133b9367",  # ### Phase ... Struct reconstruction
        "13d6d847",  # ### Phase ... Retype locals and parameters
        "2cc133cf",  # ### Phase ... Rename functions
        "2dae86aa",  # - **Build a mental call tree** — for key functions, trace both u
        "39e278be",  # ... Skip the toolchain stub ... — it's the first slot on most GC
        "4226b58c",  # ... **Survey imports and strings** — these reveal libraries, API
        "45acc326",  # > **Quick-loaded target?** If the binary was opened with `bn loa
        "5b9065e6",  # Binary Ninja's auto-analysis follows direct ... Two important ca
        "5bc6b916",  # ... **Scan the function list** — get a sense of scope:
        "98b2f8c2",  # ... `bn evidence init` finds every ... section ... ... ... …), w
        "991a98f8",  # ... Decompile each remaining ... Anything that writes to BSS ...
        "a0373b7c",  # Note the total count, address range, and whether symbols are ...
        "b9a586c2",  # ... `bn evidence table <table-addr> --entries N` reads the ... t
        "cc459ea3",  # ... **Orient** — get architecture, platform, and entry point:
        "ce59e5c8",  # When this comes up most: VM opcode handler tables, FSA predicate
        "d4ecc83d",  # - **First, if the binary still has demangled C++ symbols, use th
        "d5b78ae4",  # ... `bn function create <target> --preview` creates and verifies
        "e20ac70e",  # narrows when you want only ...
        "fea416ec",  # > **One-shot triage (steps ... in a single consistent ... `bn ev
    }),
    "skills/bn-vr/SKILL.md": frozenset({
        "03b7cda5",  # bn evidence function ... --context ... # raw ABI args at each ..
        "04f10e8a",  # > `bn taint forward -f ... --source ... — which surfaces the
        "0b2bd3cc",  # bn strings --regex --query ... --no-crt --min-length ...
        "0c92c9f8",  # strcat(out, decrypt(chunk)); ... bound = Σ decrypted-chunk lengt
        "0e36f595",  # ... p + ... ... ... handler runs before the header is proven com
        "0fef6d54",  # ... **Interesting strings** — format strings, SQL fragments, she
        "132f4dad",  # **Worked example — ... ... is an applet multiplexer: `main` disp
        "145b57e4",  # bn disasm ... --linear ... # address-linear: confirm ... widths 
        "152a5d55",  # p += ... + ... ... advances by an ... length
        "1961f9cb",  # > (backward); reading backward ... as results misreports a real 
        "1ad7941d",  # ... **For ... APIs, audit EVERY caller** — a ... src)` wrapper i
        "2c53b547",  # > **`file`=stripped ≠ static ... ... reports "stripped" whenever
        "2ce56b81",  # code = ... ... ... type
        "2d6c57a2",  # bn taint backward -f <handler> --sink ... # where does the lengt
        "35cc74cd",  # > **Shortcut (step ... of sink enumeration):** `bn taint models 
        "36c9380b",  # Reports **one compact line per flow** by default: the bug class 
        "43223b71",  # ... **Walk constructor and dispatch ... Static firmware hides en
        "4344b6fb",  # bn strings --regex --query ... --no-crt --min-length ...
        "45e9a61b",  # bn disasm ... --linear ... # confirm frame size + register args
        "48b6ca21",  # bn taint backward -f <handler> --sink ...
        "4be13115",  # bn taint forward -f <handler> --source ...
        "526e3eac",  # > ... taint through an object parser or a raw decoder is NOT an 
        "539306d6",  # ... **Memory layout** — understand which regions are writable, e
        "545c4dd4",  # > **Quick-loaded target?** If the binary was opened with `bn loa
        "54717824",  # char ...
        "59061229",  # bn trace ... ... --arg ... # dest -> its allocation + capacity
        "5f09cda6",  # > bn trace ... ... --arg ... # what payload the decoder reads
        "627fcd2b",  # ... **Dangerous imports** — scan for functions with known vulner
        "637387df",  # ... = ... *)(p + ... ... ... length read PAST a ... tail
        "654433c2",  # ... **Confirm the bound in disasm — not just ... Stripped + ARM 
        "69cff84d",  # ... **Trace dest ... to its allocation** — `bn trace <fn> <call>
        "7097e6ec",  # - **Unknown-option skip** — does an unrecognized `code` advance 
        "77f417d4",  # remaining -= ... + ...
        "7e63683e",  # ... **Recover the libc-like sinks by ... You can't `bn xrefs str
        "7ebbe54b",  # > Sanitized shape: ... → ... → ... → ... decoded)` — taint dies 
        "82cd26ce",  # bn trace handler ... --arg ... --interprocedural --ip-depth ... 
        "85e13861",  # > **C ... firmware dispatch registered via a stack descriptor? M
        "86329388",  # strcpy(tmp, ... ... overflow iff ... >= ...
        "894e5c7f",  # bn trace ... ... --arg ... # where ... came from (attacker vs co
        "8a5d70e1",  # Confirm the guard and the field-load widths in `bn disasm` (not 
        "90ef13c5",  # Then audit each reachable applet (httpd request parsing, telnetd
        "99782a22",  # bn trace ... ... --arg ... # source -> provenance + max length
        "9e38d640",  # ... **Confirm the ...
        "a54806ec",  # ... **Enter from ... Strings are the surviving attack-surface ma
        "a9619a55",  # Reminder: HLIL misleads beyond ... width — besides field-load si
        "ac3452de",  # Plain `bn ... thin out when dispatch is indirect or the decompil
        "b10b433a",  # char ...
        "b138da91",  # - Decompile the candidate, recognize the idiom (byte-copy loop, 
        "b28e77be",  # ... **Confirm ABI + alloc sizes in disassembly** — `bn disasm` f
        "b5627bda",  # ... **Input sources** — identify where external data enters:
        "b7f8b9bd",  # ... **Trace source ... to provenance + max NUL-terminated length
        "bf3acbeb",  # ... Identify the sink callsite and its arguments
        "c0ddc155",  # ... Use `bn xrefs` on the caller to find *its* callers
        "c80d68ee",  # ... **Treat taint output as frontier guidance, not proof** — a .
        "c89d94c8",  # - **Loop guard vs fixed-header width** — is the continue conditi
        "cadf552c",  # libc sink typed by its resolver as ... args, which silently unde
        "cb01190b",  # Source→sink taint proves *attacker data reaches a modeled ... It
        "cb9b1402",  # bn trace handler ... --arg ... --interprocedural # follow throug
        "cfbe0f50",  # ... {"sink": {"class": ... ... ... ... ...
        "d1c3ac2a",  # ... **For `strcat`, bound dest length + total appends** — the ov
        "d5c8e51d",  # ... Repeat until you reach an input source or lose the trail
        "da3874be",  # ... Trace each argument back through the caller's locals and par
        "dfa07040",  # > bn disasm ... --linear ... # is `decoded` a fixed buffer? boun
        "e0a2d963",  # bn taint forward -f <handler> --source ...
        "e6803cea",  # while (remaining > ... { ... BUG: must be `remaining >= ... (the
        "f847521c",  # > `--source ... (or ... can report **zero propagation** —
    }),
}



def _doc_lines(doc: str) -> list[tuple[int, int, str, str]]:
    """(line number, start offset, raw text, normalized text) for EVERY line
    of *doc*, fenced code blocks included.

    They used to be excluded as "transcripts and command syntax", and a
    contract stated as a `# Exit 5 means ...` comment inside a fenced block is
    read by an agent exactly like prose. An exclusion is a boundary, and this
    accounting has run out of patience with boundaries. The normalized form
    collapses whitespace, so a re-indent is not a change.
    """
    lines: list[tuple[int, int, str, str]] = []
    offset = 0
    for at, line in enumerate(_doc_text(REPO / doc).splitlines(), start=1):
        start, offset = offset, offset + len(line) + 1
        lines.append((at, start, line, " ".join(line.split())))
    return lines


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


# A bare fingerprint is unreviewable -- a diff shows that a hash changed and not
# WHICH line was re-fingerprinted -- so every ledger entry carries its line as a
# comment. But an agent-facing doc demonstrates commands against example
# targets, and copying `-i <id> -t <lib>.so`, a `sub_<hex>` or a `0x<hex>` into
# a test file is a disclosure wherever it is copied, not a nit. So the comment
# is a REDACTED rendering: any token carrying a digit, an underscore, a dot or a
# slash is elided, which covers an address, a symbol, a path, a filename and an
# instance id without a vocabulary of target names to keep up to date.
# Deliberately broader than the requirement -- eliding `op_registry.py` costs
# nothing and missing one name is a disclosure.
_ELIDED = re.compile(r"\S*[0-9_./\\]\S*")
# A symbol or a target name need carry none of those: `PlayerUpdate` is a
# CamelCase identifier and nothing else. An inner lower-to-upper transition is
# what makes it one, and eliding `BinaryView` as collateral costs nothing.
_CAMEL = re.compile(r"[a-z][A-Z]")
# ...and an instance id or target selector is identified by the flag in front of
# it rather than by its own spelling, so the token AFTER one of these is elided
# whatever it looks like.
_TARGET_FLAGS = frozenset({"-i", "--instance", "--instance-id", "-t", "--target"})


def _ledger_comment(normalized: str) -> str:
    rendered, elide_next = [], False
    for token in normalized.split():
        rendered.append("..." if (elide_next or _ELIDED.fullmatch(token)
                                  or _CAMEL.search(token)) else token)
        elide_next = token.strip("`\"'<>()[],.") in _TARGET_FLAGS
    return " ".join(rendered)[:64]


def _bullet_span(prefix: str, text: str) -> tuple[int, int]:
    """The character range of the `prefix` bullet, continuation lines folded in.

    `_bullet` returns the bullet's TEXT; the sweep needs its EXTENT, because a
    cell that pins a clause of that bullet pins it THERE and nowhere else.
    """
    offset, span = 0, None
    for line in text.splitlines(keepends=True):
        if span is None and line.startswith(prefix):
            span = [offset, offset + len(line)]
        elif span is not None:
            if not line.startswith((" ", "\t")) or not line.strip():
                break
            span[1] = offset + len(line)
        offset += len(line)
    assert span is not None, f"{prefix!r} bullet is gone, so its clauses pin nothing"
    return span[0], span[1]


def _exit_code_claimed(doc: str, text: str) -> bytearray:
    """The characters of *text* that some cell pins.

    A clause pattern of the exit-code bullet is applied to THAT BULLET only.
    Applied document-wide it pre-claimed every other line that reused a guarded
    clause's wording, so appending "0 = success even when every instance
    failed" to `CLAUDE.md` -- a false contract -- produced no residual token and
    the sweep never saw the line (round 12). A cell pins one statement in one
    place; letting its wording immunise the rest of the document is the
    fail-open echo this accounting exists to retire.
    """
    patterns = [pattern for cell_doc, _, pattern, _ in _EXIT_CODE_ECHOES
                if cell_doc == doc]
    patterns += [re.escape(literal) for cell_doc, _, literal in _EXIT_CODE_PINS
                 if cell_doc == doc]
    claimed = bytearray(len(text))
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.M):
            claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    if doc == "CLAUDE.md":
        start, end = _bullet_span("- Exit codes:", text)
        for _, pattern in _EXIT_CODE_CLAUSES:
            for match in re.finditer(pattern, text[start:end], re.M):
                at = start + match.start()
                claimed[at:start + match.end()] = b"\x01" * (match.end() - match.start())
    return claimed


def _number_lines(doc: str) -> list[tuple[str, str, list[str]]]:
    """(where, normalized line, residual tokens) for every line carrying a
    number that no cell pins.

    The RESIDUAL is what the ledger has to excuse, and it is part of the
    ledger's key. Keyed on the line alone, one ledger entry excused every
    number on it: `README.md`'s contract paragraph carries eight cell-pinned
    digits and the prose word "one", so its fingerprint sat in the ledger and
    DELETING the cell that ties that paragraph to `_mutation_exit_code` left
    the module green. An entry now excuses one exact set of leftovers, so
    removing a pin changes the leftovers and reds.
    """
    text = _doc_text(REPO / doc)
    claimed = _exit_code_claimed(doc, text)
    rows = []
    for at, start, raw, normalized in _doc_lines(doc):
        residual = [match.group(0).lower() for match in _NUMBER_TOKEN.finditer(raw)
                    if not claimed[start + match.start()]]
        if residual:
            rows.append((f"line {at}", normalized, residual))
    return rows


def _ledger_key(normalized: str, residual: list[str]) -> str:
    return _fingerprint("\x00".join([normalized, *residual]))


def _unaccounted_number_lines(doc: str) -> list[str]:
    """Lines carrying a number that NO cell pins and the ledger does not record."""
    ledger = NON_CLAIM_NUMBER_LINES.get(doc, frozenset())
    return [
        f"{where} ({', '.join(residual)}): {normalized[:140]}"
        for where, normalized, residual in _number_lines(doc)
        if _ledger_key(normalized, residual) not in ledger
    ]


@pytest.mark.parametrize("doc,claim,literal", _EXIT_CODE_PINS,
                         ids=[f"{doc}:{claim}" for doc, claim, _ in _EXIT_CODE_PINS])
def test_every_pinned_exit_code_statement_still_reads_as_pinned(
        doc: str, claim: str, literal: str):
    """The deletion half for the statements with nothing to compute: the text is
    the pin, so a flipped digit or a deleted sentence reds exactly this cell."""
    assert literal in _doc_text(REPO / doc), (
        f"{doc} no longer states the {claim!r} exit code as pinned, so this echo "
        f"of the contract can drift from the code: expected {literal!r}"
    )


@pytest.mark.parametrize("doc", EXIT_CODE_DOCS)
def test_no_agent_doc_carries_a_number_nothing_accounts_for(doc: str):
    """The coverage half, over every NUMBER instead of over a phrasing.

    #716's defect was two documents stating the contract while one was guarded.
    Pinning the statements a reviewer happened to find is the same defect with a
    longer list, and pinning the statements a REGEX happened to find is the same
    defect with a longer reach. So no line of these documents may carry a number
    unless a cell pins it or the ledger records why it is not a claim -- and a
    new sentence stating an exit code is a new line with a number in it,
    whatever words it uses to say so.
    """
    unaccounted = _unaccounted_number_lines(doc)
    assert not unaccounted, (
        f"{doc} carries numbers no cell pins and the ledger does not record, so "
        "an exit code stated here can drift from the code with every other "
        "guard green. If it states one, add a cell to _EXIT_CODE_ECHOES or "
        "_EXIT_CODE_PINS; if it does not, add its fingerprint to "
        f"NON_CLAIM_NUMBER_LINES: {unaccounted}"
    )


def test_the_non_claim_number_ledger_has_no_stale_entry():
    """...and the ledger stale-fails, so it cannot outlive the lines it excuses
    and quietly become a place to park a claim."""
    stale = {
        doc: sorted(ledger - {_ledger_key(normalized, residual)
                              for _, normalized, residual in _number_lines(doc)})
        for doc, ledger in NON_CLAIM_NUMBER_LINES.items()
    }
    stale = {doc: entries for doc, entries in stale.items() if entries}
    assert not stale, (
        "these ledger entries excuse lines that no longer exist, so the ledger "
        "is bookkeeping for a document that has moved on; regenerate it against "
        f"the current docs: {stale}"
    )


_LEDGER_ROW = re.compile(r'^        "(?P<key>[0-9a-f]{8})",  # (?P<comment>.*)$')

# The requirement, stated HERE and not in the redactor: a guard that asks the
# redactor what provenance is cannot fail when the redactor loosens. It names
# the four classes that are machine-recognisable in isolation or by position --
# an address, an underscored or CamelCase symbol, a target filename, and a value
# introduced by an instance/target flag.
#
# What it CANNOT recognise, stated rather than claimed away: a bare lowercase
# word that happens to be a symbol or an instance id (`prfleetseven`) is not
# distinguishable from prose by any rule, so the redactor is deliberately
# broader than this check and a reviewer reads the diff for that one case.
_TARGET_PROVENANCE = re.compile(
    r"0x[0-9a-fA-F]+|[A-Za-z]\w*_\w+|\w+\.(?:so|bndb|bin|elf|exe|dll|dylib)\b"
    r"|[a-z][A-Z]|(?:-i|--instance|--instance-id|-t|--target) +[^ .]+")


_LEDGER_DOC = re.compile(r'^    "(?P<doc>[^"]+)": frozenset\({$')


def _ledger_comments() -> dict[tuple[str, str], str]:
    """The comment beside each ledger entry, keyed by (document, fingerprint).

    Keyed by fingerprint alone, two docs sharing an identical line and residual
    shared one dict entry, so the earlier row's comment was SHADOWED and read by
    neither guard -- a provenance string could be committed in it and both cells
    stayed green (round 12). The ledger is per-document and so is this.
    """
    source = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = source.index("NON_CLAIM_NUMBER_LINES: dict[str, frozenset[str]] = {")
    rows: dict[tuple[str, str], str] = {}
    doc = None
    for line in source[start + 1:]:
        if line == "}":
            break
        heading = _LEDGER_DOC.match(line)
        if heading:
            doc = heading["doc"]
            continue
        match = _LEDGER_ROW.match(line)
        if match:
            assert doc is not None, f"a ledger row sits outside any document: {line}"
            rows[(doc, match["key"])] = match["comment"]
    return rows


def test_every_ledger_entry_carries_the_line_it_excuses():
    """The half a reviewer can check: the comment renders the live document's
    line, so an entry re-fingerprinted against a changed line shows WHICH line
    changed instead of only that a hash did."""
    comments = _ledger_comments()
    keyed = {(doc, key) for doc, ledger in NON_CLAIM_NUMBER_LINES.items()
             for key in ledger}
    assert set(comments) == keyed, (
        "every ledger entry must carry its line as a comment and name a key the "
        f"ledger holds, per document: {sorted(set(comments) ^ keyed)}"
    )
    expected = {(doc, _ledger_key(normalized, residual)): _ledger_comment(normalized)
                for doc in EXIT_CODE_DOCS
                for _, normalized, residual in _number_lines(doc)}
    wrong = {key: (comment, expected.get(key))
             for key, comment in comments.items() if expected.get(key) != comment}
    assert not wrong, (
        "these ledger comments do not render the line their key excuses, so the "
        f"comment is bookkeeping a reviewer cannot trust: {wrong}"
    )


def test_no_ledger_comment_names_a_target():
    """...and rendering the line must not carry the line's PROVENANCE into a
    committed file: an address, a symbol, a target filename or an instance id
    quoted by a doc's example command is a disclosure wherever it is copied."""
    leaked = {key: _TARGET_PROVENANCE.findall(comment)
              for key, comment in _ledger_comments().items()
              if _TARGET_PROVENANCE.search(comment)}
    assert not leaked, (
        "these ledger comments carry target provenance; the comment is a "
        f"redacted rendering of the line, not the raw line: {leaked}"
    )


def test_the_non_claim_number_ledger_only_names_fenced_docs():
    """A ledger entry for a document outside the sweep excuses nothing and
    hides the fact that the document is unswept."""
    assert set(NON_CLAIM_NUMBER_LINES) <= set(EXIT_CODE_DOCS), (
        "the ledger names documents the exit-code sweep does not read: "
        f"{sorted(set(NON_CLAIM_NUMBER_LINES) - set(EXIT_CODE_DOCS))}"
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

    Three echoes state a code no classifier call can produce: the output
    divergences live strictly after the classification, in the step that
    delivers the output, and an unreachable bridge never reaches it. Asking
    `_mutation_exit_code` for any of them would quote the document back at
    itself, so the command is run.

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
    """What the CLI really returns for the scenario an echo describes."""
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


@pytest.mark.parametrize("doc,claim,pattern,scenario", _EXIT_CODE_ECHOES,
                         ids=[f"{doc}:{claim}" for doc, claim, _, _ in _EXIT_CODE_ECHOES])
def test_every_doc_that_states_an_exit_code_states_the_one_the_cli_returns(
        doc: str, claim: str, pattern: str, scenario: str):
    """One cell per statement, so the document that drifts is the one that
    fails, and the expected code comes from the helper rather than from a second
    copy of the contract."""
    text = _doc_text(REPO / doc)
    match = re.search(pattern, text)
    assert match, (
        f"{doc} no longer states the {claim!r} exit code, so this echo of the "
        "contract can drift from the code with every other guard green"
    )
    assert int(match.group("code")) == _exit_code_for(scenario), (
        f"{doc} tells an agent the {claim!r} case is exit {match.group('code')}, "
        f"but the CLI returns {_exit_code_for(scenario)} for it"
    )


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


# The lock class is declared at the `@op` decorator, so the declarations are the
# ground truth for what `lock="none"` actually covers.
_NONE_LOCK_OP = re.compile(r'@op\(\s*"([^"]+)"\s*,\s*lock="none"')

# The `none` semantics an agent reads before choosing a lock class, pinned the
# same way the exit-code bullet is: one cell per load-bearing claim, plus a
# coverage half that leaves no prose in the paragraph unaccounted for.
#
# Four cuts were escaped before this one. Three were proxies for meaning rather
# than pins on the text: "some op name anywhere in the paragraph" (satisfied by
# an incidental parenthetical after the refutation was deleted), `"not" in lead`
# (satisfied by "as a general note"), and a word-boundary `\bnot\b` at every
# mention (satisfied by "is not merely ...", which AFFIRMS the false reading).
# The fourth was a pin whose ACCOUNTING had a boundary: it covered the paragraph
# from the first `none` claim onward, so a sentence inserted BEFORE that marker,
# in the same paragraph, taught the falsehood with every pin green.
#
# A proxy for meaning is escapable by construction; an accounting with a
# boundary is escapable just outside it. Round 8 made the region "the whole
# paragraph" and got the paragraph wrong -- it took the one LINE the sentence
# starts on, and a markdown paragraph is every line up to the blank one, so an
# affirmation on the next line sat inside the same paragraph and outside the
# accounting. The cross-document half had the mirror-image hole: it matched the
# claim only in its QUOTED form, so the same words without the quotes were
# invisible.
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

_LOCK_MODEL_CLAIMS = (
    ("registry-is-the-source-of-truth",
     r"`op_registry\.py` is the single source of truth: "
     r"`@op\(name, lock=\"read\"\|\"write\"\|\"none\"\)` declares each op once, and "
     r"both the lock sets and dispatch routing are derived from it "
     r"\(`REGISTRY\.read_locked_ops\(\)` / `write_locked_ops\(\)`\)\."),
    ("read-and-write-ops-dispatch-locked",
     r"Read ops dispatch under a shared writer-priority `_ReadWriteLock`; "
     r"write ops under an exclusive lock;"),
    ("none-ops-self-manage",
     r"`none` ops run outside the dispatcher's lock and \*\*self-manage locking\*\*: "
     r"the op body takes the write gate and/or the exclusive target lock itself\."),
    ("not-touches-no-bn-state",
     r"`lock=\"none\"` is emphatically \*not\* \"touches no BN state\" — "
     r"`load_binary`, `refresh` and `go_rename` all mutate the view"),
    ("why-the-dispatcher-holds-nothing",
     r"it means the dispatcher must not hold a lock for them, because they take "
     r"their own \(`refresh` runs analysis holding the analysis lock\) or because "
     r"holding one would deadlock them \(`shutdown` and `cancel_request` must stay "
     r"deliverable while a write op is wedged\)\."),
)


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


@pytest.mark.parametrize("claim,pattern", _LOCK_MODEL_CLAIMS,
                         ids=[name for name, _ in _LOCK_MODEL_CLAIMS])
def test_every_claim_of_the_lock_model_region_is_guarded(claim: str, pattern: str):
    """One cell per claim: deleting or rewording any of them reds exactly this
    cell. `none` means the DISPATCHER holds no lock and the op body
    self-manages; reading it as "touches no BN state" makes an agent either
    write-lock an op that must not be, or ship a stateful op taking no lock."""
    region = _lock_model_region()
    assert re.search(pattern, region), (
        f"the lock-model paragraph no longer states the {claim!r} claim: {region}"
    )


def test_the_lock_model_region_carries_no_unguarded_claim():
    """The half that stops the NEXT claim: every character of the paragraph is
    claimed by a cell above, so a clause inserted anywhere in it -- before the
    `none` claims, between them, or after -- is unclaimed prose and fails here.
    An affirmation reusing an earlier mention's negation escaped the proxy cuts;
    an affirmation inserted before the region marker escaped the first pin."""
    region = _lock_model_region()
    prose = [text for text in _unclaimed_runs(region, [p for _, p in _LOCK_MODEL_CLAIMS])
             if re.search(r"\w", text)]
    assert not prose, (
        "these runs of the lock-model paragraph are claimed by no cell in "
        f"_LOCK_MODEL_CLAIMS, so they could teach anything with this module "
        f"green: {prose}"
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
    refutation = next(pattern for name, pattern in _LOCK_MODEL_CLAIMS
                      if name == "not-touches-no-bn-state")
    claimed = bytearray(len(text))
    for match in re.finditer(refutation, text):
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
    refutation = next(pattern for name, pattern in _LOCK_MODEL_CLAIMS
                      if name == "not-touches-no-bn-state")
    match = re.search(refutation, region)
    assert match, f"the refutation cell above owns this; region reads: {region}"
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
