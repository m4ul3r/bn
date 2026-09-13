"""Guards that keep the agent-instruction docs honest.

`CLAUDE.md` is the canonical agent-instruction file and root `AGENTS.md` is a
tracked symlink to it, so an agent reading either one sees the same tree layout
(#607). These tests pin the invariants that actually misled agents before:
a second physical copy that drifts, a bridge path that no longer exists, a
`uv run pytest` line naming a module or test id that was renamed away (#614),
and an exit-code list that silently falls behind `FAILED_MUTATION_STATUSES`.
"""

from __future__ import annotations

import hashlib
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
    ("skills/bn/reference/mutating.md", "own-summary-stays-measured",
     r"so a clean run is exit `(?P<code>\d)`, not the unmeasured `4`", "own-summary"),
    ("skills/bn/reference/mutating.md", "unsupported-op-kind",
     r"either way exit (?P<code>\d), and a", "failing"),
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
     "response this CLI cannot parse) / 3 a mutation status `verification_failed`,\n"
     "`unsupported`, `invalid_request`, `rollback_failed`, or `internal_error` / 4 an\n"
     "unmeasured success"),
    ("skills/bn/reference/reading.md", "bounded-slice-is-a-success",
     "a provably-bounded constant length (a success, exit 0)"),
    ("skills/bn/reference/runtime.md", "restart-of-an-unreachable-bridge-is-1",
     "this way exits **1** rather than 0 whenever the teardown and respawn succeed"),
    ("skills/bn/reference/runtime.md", "restart-that-cannot-signal-is-2",
     "the restart refuses to signal and exits **2** instead"),
)

# Every doc an agent reads for the contract. `skills/bn/SKILL.md` states no exit
# code today and is in the sweep so that adding one there fails until pinned.
EXIT_CODE_DOCS = ("CLAUDE.md", "README.md", "skills/bn/SKILL.md",
                  "skills/bn/reference/mutating.md",
                  "skills/bn/reference/reading.md",
                  "skills/bn/reference/runtime.md")

# Round 8 replaced a hand-listed echo table with a sweep, and the sweep was a
# RECOGNISER: a bare 0-4 within 120 characters after the word `exit`. Both
# round 9 lenses walked past it in one line -- "a failed mutation returns `2`
# to the shell", "a refused op returns three", "Exit 5 is reserved ... exit 9
# for a bridge that refuses to start" -- because the phrasing avoided the word,
# or the digit was outside the range, or the number was spelled out.
#
# A recogniser for natural language is escapable by construction, so there is
# no vocabulary here at all. The population is EVERY line of every fenced doc
# that carries a number, and each is accounted for exactly one of two ways: a
# cell above pins it to what the CLI really returns, or it is in the ledger
# below of lines that carry a number for some other reason. A new sentence
# about an exit code fails here in ANY phrasing, because it is a new line with
# a number in it and nothing accounts for it yet.
#
# A decimal is one number, not two (`3.11` must not read as a `3` and an `11`),
# and an issue reference is not a code (`#625`). Everything else counts: a
# trailing period is sentence punctuation, not a decimal point (excluding it
# outright let "a refused op returns three." straight through), a hyphen
# neighbour is still a number ("twenty-five", `utf-8`), and the words run past
# nine so that spelling one out is not a way around the digits. A multi-digit
# numeral yields one token, because its later digits follow a word character.
_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
)
_NUMBER_TOKEN = re.compile(
    rf"(?<![\w#$/])(?<!\d\.)(?:\d|{_NUMBER_WORDS})(?![\w/])(?!\.\d)", re.I)
# Generated, not authored: the fingerprint of every prose line in a fenced doc
# that carries a number and is not pinned by a cell above. Regenerate with
# `_fingerprint(" ".join(line.split()))` over the docs. An entry is a claim
# that this line's number is NOT an exit code; adding one is a deliberate
# statement in a diff, which is the point -- the previous accounting made that
# statement silently, by not matching a pattern.
NON_CLAIM_NUMBER_LINES: dict[str, frozenset[str]] = {
    "CLAUDE.md": frozenset({
        "2813ca55", "4177be03", "78629073", "7d6b2776", "80180431", "8597bc7e", "8e4a9610",
        "bb36a4be", "dadf1f99", "db981a08", "ee61fade", "fa70ea09",
    }),
    "README.md": frozenset({
        "0a768f64", "18aefee8", "1946ee65", "39297f76", "4a59a3b8", "4f15b254", "589b6b1c",
        "62c7882b", "70fe4ca5", "767b9f9b", "ae3959b5", "b3987998", "b70bd14f", "d765c702",
    }),
    "skills/bn/SKILL.md": frozenset({
        "3435274a", "351cea9b", "3d99e01e", "476c699a", "61785515", "649eaeac", "66190c94",
        "718cbccf", "7f2b6f7d", "8909cfbd", "8919c69b", "c094744f", "cc207c7d",
    }),
    "skills/bn/reference/mutating.md": frozenset({
        "004be911", "023ef17d", "073c32cb", "21ab0f26", "357b33d7", "3a31edd0", "3ab0e800",
        "3b705aa6", "4d2bfe51", "4ea5a88f", "73915809", "757a2ce7", "7a963eb3", "7b90ea1f",
        "7d08894a", "7d87431b", "99a32d48", "abcbb98d", "b01a5a75", "b29746c8", "b570f08e",
        "c11a3930", "ca2ff185", "dfafc784", "e0108fe1", "e26fd985", "e7bc29dc", "f73aab60",
        "fe025816",
    }),
    "skills/bn/reference/reading.md": frozenset({
        "075c33cd", "0c9d943c", "1428ab30", "1949d287", "1a70e75d", "1c15c2d2", "1dd5bce8",
        "25419b43", "2ea1ed0f", "33fd9640", "3716a90b", "43ca5e2b", "4f5394e5", "4fe41c72",
        "505bed3f", "5221e5b6", "6a4f918d", "6a58a9dc", "6ab53c02", "6ed98be4", "7bc2e390",
        "86526afb", "91995abd", "965d7b1e", "9971ee4b", "9a98f5ff", "9dbbd6d8", "a916839d",
        "b71d1544", "b92a2be1", "be1545e3", "be4ef095", "c0610400", "c4b1b451", "d418748d",
        "dad2603f", "df48aed6", "e9b95184", "eca3534e",
    }),
    "skills/bn/reference/runtime.md": frozenset({
        "06aad85c", "079c846e", "0a8cddd1", "0ddc02d0", "1fbd64eb", "347a4a13", "34c79efb",
        "44497fad", "53f0229b", "55a24561", "589fc7e3", "596d0a92", "61785515", "65df27e3",
        "66190c94", "6a70bcd2", "6f94b989", "7272fe33", "767bdfe8", "8c218bae", "8f382f50",
        "926f4920", "947fffc7", "98cfec52", "9a1ac7d1", "9cb452f2", "a8fb6baa", "af1d6d85",
        "b4306499", "b5febfe7", "be5e5584", "bec2b018", "c22c0ad1", "c2335aac", "ca2c1267",
        "cb4d6e9e", "d20853ec", "d6c30113", "d7f57f32", "e1e09dec", "e45a920a", "e6c69801",
        "e8bc9423", "eddb9cfa",
    }),
}



def _prose_lines(doc: str) -> list[tuple[int, int, str, str]]:
    """(line number, start offset, raw text, normalized text) for every PROSE
    line of *doc*.

    Fenced code blocks are excluded: they are transcripts and command syntax,
    not statements to an agent about what the CLI returns. The normalized form
    collapses whitespace, so a re-indent or a re-wrap is not a change.
    """
    lines: list[tuple[int, int, str, str]] = []
    offset = 0
    fenced = False
    for at, line in enumerate(_doc_text(REPO / doc).splitlines(), start=1):
        start, offset = offset, offset + len(line) + 1
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            lines.append((at, start, line, " ".join(line.split())))
    return lines


def _number_lines(doc: str) -> list[tuple[int, int, str, str]]:
    return [row for row in _prose_lines(doc) if row[3] and _NUMBER_TOKEN.search(row[2])]


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _exit_code_claimed(doc: str, text: str) -> bytearray:
    """The characters of *text* that some cell pins."""
    patterns = [pattern for cell_doc, _, pattern, _ in _EXIT_CODE_ECHOES
                if cell_doc == doc]
    patterns += [re.escape(literal) for cell_doc, _, literal in _EXIT_CODE_PINS
                 if cell_doc == doc]
    if doc == "CLAUDE.md":
        patterns += [pattern for _, pattern in _EXIT_CODE_CLAUSES]
    claimed = bytearray(len(text))
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.M):
            claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    return claimed


def _unaccounted_number_lines(doc: str) -> list[str]:
    """Lines carrying a number that NO cell pins and the ledger does not record.

    Accounting is per NUMBER, not per line: a cell that pins one sentence of a
    line does not account for a second claim appended to the same line, which
    is how an unpinned claim rode into a pinned paragraph.
    """
    text = _doc_text(REPO / doc)
    claimed = _exit_code_claimed(doc, text)
    ledger = NON_CLAIM_NUMBER_LINES.get(doc, frozenset())
    unaccounted = []
    for at, start, raw, normalized in _number_lines(doc):
        loose = [match.group(0) for match in _NUMBER_TOKEN.finditer(raw)
                 if not claimed[start + match.start()]]
        if not loose or _fingerprint(normalized) in ledger:
            continue
        unaccounted.append(f"line {at} ({', '.join(loose)}): {normalized[:140]}")
    return unaccounted


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
        doc: sorted(ledger - {_fingerprint(text) for _, _, _raw, text in _number_lines(doc)})
        for doc, ledger in NON_CLAIM_NUMBER_LINES.items()
    }
    stale = {doc: entries for doc, entries in stale.items() if entries}
    assert not stale, (
        "these ledger entries excuse lines that no longer exist, so the ledger "
        "is bookkeeping for a document that has moved on; regenerate it against "
        f"the current docs: {stale}"
    )


def test_the_non_claim_number_ledger_only_names_fenced_docs():
    """A ledger entry for a document outside the sweep excuses nothing and
    hides the fact that the document is unswept."""
    assert set(NON_CLAIM_NUMBER_LINES) <= set(EXIT_CODE_DOCS), (
        "the ledger names documents the exit-code sweep does not read: "
        f"{sorted(set(NON_CLAIM_NUMBER_LINES) - set(EXIT_CODE_DOCS))}"
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


@pytest.mark.parametrize("doc", EXIT_CODE_DOCS)
def test_no_agent_doc_states_the_false_lock_reading_unrefuted(doc: str):
    """...and the paragraph is not the only place the false reading could be
    taught. Every occurrence of the claim in every agent-facing doc must sit
    inside the refutation cell's own span, so moving it to another document --
    or to another paragraph of this one -- fails the same way as affirming it
    here.

    Matched as WORDS, not as the quoted string round 8 pinned: the same reading
    taught without the quotation marks, or across a line wrap, is the same
    reading.
    """
    text = _doc_text(REPO / doc)
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
        f"{doc} states {LOCK_MODEL_FALSE_CLAIM} outside the refutation that "
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
