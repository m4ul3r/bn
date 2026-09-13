"""Guards that keep the agent-instruction docs honest.

`CLAUDE.md` is the canonical agent-instruction file and root `AGENTS.md` is a
tracked symlink to it, so an agent reading either one sees the same tree layout
(#607). These tests pin the invariants that actually misled agents before:
a second physical copy that drifts, a bridge path that no longer exists, a
`uv run pytest` line naming a module or test id that was renamed away (#614),
and an exit-code list that silently falls behind `FAILED_MUTATION_STATUSES`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# NB: no module-level `binaryninja` stub here. Injecting one at import time
# poisons `sys.modules` for every module collected AFTER this one, and the
# bridge's taint modules import real symbols from it -- under a randomized
# collection order that turned into a suite-wide collection error. These two
# imports are CLI-side and need no engine.
from bn.formatters import FAILED_MUTATION_STATUSES, _mutation_summary

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
     r"and a mutation result this CLI cannot classify — malformed or newer than the CLI\)"),
    ("3-mutation-status",
     r"\b3 = a `_mutate`-marked call whose status is `verification_failed`, `unsupported`, "
     r"`invalid_request`, `rollback_failed`, or `internal_error`"),
    ("3-refused-up-front-or-at-apply",
     r"— the refusal is exit 3 whether it was raised up front or during apply, "
     r"never 2 for a mutation —"),
    ("4-unmeasured",
     r"\b4 = a `_mutate`-marked call whose compact summary reports `measured: false`, i\.e\."),
    ("4-no-rows-and-no-own-summary",
     r"the op returned no `results\[\]` rows to derive counts from AND registered no "
     r"summary of its own to count with, so the outcome could not be verified \(#715\);"),
    ("4-own-summary-stays-measured",
     r"an op that counts through its own registered summary \(`go rename`\) stays "
     r"measured and exits 0\."),
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
    claimed = bytearray(len(bullet))
    for _, pattern in _EXIT_CODE_CLAUSES:
        for match in re.finditer(pattern, bullet):
            claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    unclaimed: list[str] = []
    run: list[str] = []
    for index, char in enumerate(bullet):
        if claimed[index]:
            if run:
                unclaimed.append("".join(run))
                run = []
        else:
            run.append(char)
    if run:
        unclaimed.append("".join(run))
    prose = [text for text in unclaimed if re.search(r"\w", text)]
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
    # "an op that counts through its own registered summary stays measured and exits 0"
    own_counters = lambda result: {"kind": "mutation_summary", "measured": True}
    assert _mutation_exit_code(unmeasured, own_counters) == 0


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
    ("skills/bn/reference/mutating.md", "own-summary-stays-measured",
     r"so a clean run is exit `(?P<code>\d)`, not the unmeasured `4`", "own-summary"),
    ("skills/bn/reference/mutating.md", "unsupported-op-kind",
     r"either way exit (?P<code>\d), and a", "failing"),
)


def _exit_code_for(scenario: str) -> int:
    """What the CLI really returns for the scenario an echo describes."""
    from bn.cli import _mutation_exit_code

    shapes = {
        "verified": ({"success": True, "committed": True,
                      "results": [{"status": "verified"}]}, _mutation_summary),
        "failing": ({"success": False, "committed": False,
                     "results": [{"status": "invalid_request"}]}, _mutation_summary),
        "unmeasured": ({"success": True, "committed": True, "results": []},
                       _mutation_summary),
        "own-summary": ({"success": True, "committed": True, "results": []},
                        lambda result: {"kind": "mutation_summary", "measured": True}),
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


# The lock class is declared at the `@op` decorator, so the declarations are the
# ground truth for what `lock="none"` actually covers.
_NONE_LOCK_OP = re.compile(r'@op\(\s*"([^"]+)"\s*,\s*lock="none"')

# The `none` semantics an agent reads before choosing a lock class, pinned the
# same way the exit-code bullet is: one cell per load-bearing claim, plus a
# coverage half that leaves no prose in the region unaccounted for.
#
# Three cuts were escaped before this one, each a proxy for meaning rather than
# a pin on the text: "some op name anywhere in the paragraph" (satisfied by an
# incidental parenthetical after the refutation was deleted), `"not" in lead`
# (satisfied by "as a general note"), and a word-boundary `\bnot\b` at every
# mention (satisfied by "is not merely ...", which AFFIRMS the false reading,
# and by an appended clause that inherited an earlier mention's negation).
#
# A proxy for meaning is escapable by construction. A pin is not: changing this
# region reds its cell, which is the point -- someone has to re-read the claim
# and move the pin with it.
#
# The two `none` ops that really are pure signals: they set an event and must
# stay deliverable while a write op holds the lock. Everything else declared
# `none` runs real work on the view and takes whatever lock it needs itself.
SIGNAL_ONLY_NONE_OPS = frozenset({"shutdown", "cancel_request"})

LOCK_MODEL_SENTENCE_PREFIX = "`op_registry.py` is the single source of truth"

# The region starts at the first claim about `none` ops; everything before it is
# the registry/dispatch prose the other guards own.
LOCK_MODEL_REGION_START = "`none` ops run outside"

_LOCK_MODEL_CLAIMS = (
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
    sentence = next(
        (line for line in _doc_text().splitlines()
         if line.startswith(LOCK_MODEL_SENTENCE_PREFIX)),
        None,
    )
    assert sentence, f"the lock-model paragraph ({LOCK_MODEL_SENTENCE_PREFIX}...) is gone"
    at = sentence.find(LOCK_MODEL_REGION_START)
    assert at >= 0, (
        f"the lock-model paragraph no longer states {LOCK_MODEL_REGION_START!r}, so "
        f"this guard cannot find the region it checks: {sentence}"
    )
    return sentence[at:]


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
    """The half that stops the NEXT claim: every character of the region is
    claimed by a cell above, so a clause appended anywhere in it -- refuting,
    affirming, or merely stale -- is unclaimed prose and fails here. An
    affirmation that reused an earlier mention's negation was how the previous
    two cuts of this guard were escaped."""
    region = _lock_model_region()
    claimed = bytearray(len(region))
    for _, pattern in _LOCK_MODEL_CLAIMS:
        for match in re.finditer(pattern, region):
            claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    runs: list[str] = []
    run: list[str] = []
    for index, char in enumerate(region):
        if claimed[index]:
            if run:
                runs.append("".join(run))
                run = []
        else:
            run.append(char)
    if run:
        runs.append("".join(run))
    prose = [text for text in runs if re.search(r"\w", text)]
    assert not prose, (
        "these runs of the lock-model region are claimed by no cell in "
        f"_LOCK_MODEL_CLAIMS, so they could teach anything with this module "
        f"green: {prose}"
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
