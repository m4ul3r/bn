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
    """The bullet's presence check: it names every code the CLI can return, and
    the two clauses that carry no numeral of their own.

    #715 widened the contract from 0/1/2/3 to 0/1/2/3/4 and added the
    unclassifiable-result case to exit 2. Both new clauses arrived UNGUARDED:
    deleting either left this module green, in the very PR whose purpose is
    making the docs provably match the code -- the same shape as the defect #721
    exists to fix.

    The code set is DERIVED from `_codes_the_cli_can_return`, the population
    `test_no_code_the_cli_can_return_is_undocumented` compares the bullet
    against, so this cell cannot drift from the CLI. It replaced fourteen
    near-verbatim clause patterns and a character-accounting sweep over them
    (#732): what those asserted was that the paragraph still says what the
    paragraph says, while every behavioural claim underneath them is executed by
    `test_the_exit_code_bullet_clauses_are_what_the_cli_actually_does`. What is
    worth keeping from them is the clause-deletion tripwire, and that is this
    cell.

    Scoped to the bullet through `_bullet_span`, never document-wide: a clause's
    wording found somewhere else in the file tells an agent reading the
    exit-code contract nothing, and letting it count was round 12's fail-open.

    Each of the two wordless clauses is tied to something executable rather than
    pinned as prose: the `measured` key really is what the compact summary emits
    for a result with no rows to count, and the CLI really does have a
    malformed-result rule that turns an unreadable response into a
    `BridgeError`.
    """
    text = _doc_text()
    start, end = _bullet_span("- Exit codes:", text)
    bullet = " ".join(text[start:end].split())
    missing = [code for code in sorted(_codes_the_cli_can_return())
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


def test_the_exit_code_bullet_clauses_are_what_the_cli_actually_does():
    """...and the presence check above only asserts that the bullet STATES its
    clauses, so this is the half that makes them true rather than merely
    quoted: each behavioural claim is executed against the helper the bullet
    describes."""
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


def test_the_unmeasured_causes_the_reference_names_are_the_ones_emitted():
    """The mutation reference quotes three cause strings VERBATIM, so an agent
    greps its session output for them. They are pinned by running the summaries
    and reading the strings back out, not by a text pin: a doc that quotes a
    phrase the code stopped emitting sends a reader looking for a line that is
    never printed, and the round-20 falsification lens deleted this paragraph
    whole with 520 tests still green.

    Both directions for the two causes an agent actually meets, and the PAIRING
    as well as the phrases. Deliberately not a quantifier over the cause
    population -- `tests/test_cli_formatters.py` holds that, and it fails when a
    seventh cause string appears or an existing one drifts. This cell's subject
    is the reference: that it quotes the live strings and attaches each to the
    condition that produces it.
    """
    from bn.formatters import _render_mutation_summary_text

    doc = _doc_text(REPO / "skills/bn/reference/mutating.md")
    # Nothing to count: the generic summary over a result with no rows.
    nothing_to_count = _mutation_summary({"success": True, "committed": True,
                                          "results": []})
    # A counter refused: the op that counts through its own counters, with one
    # of them in a shape no count reads out of.
    counter_refused = _go_rename_summary(
        {"kind": "go_rename", "preview": False, "success": True,
         "committed": True, "go_renamed_candidates": "many", "results": []})
    # (phrase, the condition the reference pairs it with, what really emits it).
    # The PAIRING is asserted, not merely the presence of both phrases: with
    # each half checked alone, swapping which phrase describes which condition
    # left every guard green, and an agent sent to look for a missing
    # `results[]` on a REFUSED COUNTER finds one, populated, and stops -- the
    # exact harm `formatters.py` says this wording exists to prevent.
    causes = (
        ("this op reported no results[] rows", "there was nothing to count",
         nothing_to_count),
        ("this op's own counters could not be read", "a counter was refused",
         counter_refused),
    )
    prose = " ".join(doc.split())
    for phrase, condition, summary in causes:
        emitted = str(summary["first_error"])
        assert phrase in emitted, (
            f"the reference quotes {phrase!r} for the case {condition!r}, but "
            f"the summary emits {emitted!r}"
        )
        assert f"`{phrase}` when {condition}" in prose, (
            f"the reference must pair {phrase!r} with the condition that really "
            f"produces it ({condition!r}); naming both causes without binding "
            "each to its case lets the two be swapped, which tells an agent to "
            "look at the wrong field"
        )
    # ...and the pairing must be EXCLUSIVE. Presence alone is satisfied by a
    # document that states both correct pairings AND both swapped ones, which
    # restores the whole harm: a reader who meets the wrong sentence first goes
    # to the wrong field. So every WRONG pairing of these phrases and conditions
    # must be absent, derived from the same table rather than listed.
    wrong = [(phrase, condition)
             for phrase, _, _ in causes
             for _, condition, _ in causes
             if (phrase, condition) not in {(p, c) for p, c, _ in causes}]
    assert wrong, "the cause table has one row, so exclusivity proves nothing"
    stated = [f"`{phrase}` when {condition}" for phrase, condition in wrong
              if f"`{phrase}` when {condition}" in prose]
    assert not stated, (
        "the reference states these cause-to-condition pairings, which the code "
        f"does not produce; each sends a reader to the wrong field: {stated}"
    )
    # ...and the per-field disclosure line, whose SHAPE is what a reader matches
    # on: the marker, the word, and the field name in between. A plain substring
    # check accepts a WIDENED marker -- `!! malformed ...` still contains
    # `! malformed ...` -- which would leave the doc's quoted shape wrong with
    # this cell green, so the marker's left edge is asserted too.
    rendered = _render_mutation_summary_text(counter_refused)
    assert re.search(r"(?<!!)! malformed go_renamed_candidates field", rendered), rendered
    assert "`! malformed <field> field`" in doc, (
        "the reference no longer states the shape of the per-field disclosure "
        "line the compact status really prints"
    )


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


def test_the_word_form_exit_claims_are_what_the_cli_does(monkeypatch, tmp_path):
    """The exit-code contracts the sweep could SEE but no cell was measuring.

    All of these are stated in WORDS -- "exits non-zero", "nonzero ... zero
    otherwise", "does not affect the exit code" -- so there is no digit for an
    echo to capture and compare against the code. The sweep sees only their
    `exits`/`exit`/`zero` tokens, and an exit word no cell pins is ruled on by
    `DECLARED_NON_EXIT_CODE_WORDS` rather than measured -- which is how these
    came to be parked as non-claims when they are precisely the word-form
    contracts the sweep exists to catch. Round 20 found two, round 21 found a
    third one document over, which is what a per-instance repair looks like --
    so the CLASS is closed by
    `test_no_parked_line_carries_an_undeclared_exit_word` below, and every
    member is pinned in `_EXIT_CODE_PINS` and executed here.
    """
    import bn.cli
    from bn.cli import _mutation_exit_code
    from bn.transport import BridgeError

    doc = _doc_text(REPO / "skills/bn/reference/runtime.md")

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

    def reply(inst, op, params=None, target=None):
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

    for where, phrase in (
        ("skills/bn/reference/runtime.md",
         "the command exits non-zero with a stderr diagnostic"),
        ("skills/bn/reference/runtime.md", "Exit code is reachability-only"),
        ("skills/bn/reference/runtime.md",
         "the command exits non-zero only if **every** result failed"),
        ("CLAUDE.md", "does not affect the exit code (#118)"),
        ("README.md", "the CLI returns a nonzero exit code"),
    ):
        assert phrase in _doc_text(REPO / where), (
            f"{where} no longer states {phrase!r}, so this word-form exit "
            "contract can drift from the code with every other guard green"
        )


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
    # README states the same derivation the mutation reference does, so it gets
    # the same measured echo: the digit is compared against what the REAL
    # `_go_rename_summary` makes of an unreadable counter.
    ("README.md", "own-counter-unreadable-is-4",
     r"which leaves that run `measured: false` and exit `(?P<code>\d)` like any "
     r"other unmeasured one", "own-summary-refused"),
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
    # Claims stated in WORDS rather than digits, which is how they came to be
    # parked as non-claims: the sweep saw their `exits`/`exit`/`zero` tokens and
    # nothing distinguished "we read this line and it states no code" from
    # "nobody looked", while they are exactly the word-form contracts the sweep
    # exists to catch. There is no digit to capture, so they are pinned here and
    # MEASURED by test_the_word_form_exit_claims_are_what_the_cli_does. The
    # population itself is closed by
    # test_no_parked_line_carries_an_undeclared_exit_word -- round 20 pinned the
    # first two and round 21 found a third one document over, which is what
    # pinning the instances instead of the class buys.
    ("skills/bn/reference/runtime.md", "an-out-of-range-line-slice-is-nonzero",
     "the command exits non-zero with a stderr diagnostic (not a `//` comment "
     "on stdout), so a scripted consumer can tell an out-of-range slice apart "
     "from a real result."),
    ("skills/bn/reference/runtime.md", "doctor-is-reachability-only",
     "Exit code is reachability-only: nonzero if any probed instance is "
     "unreachable, zero otherwise (staleness fields are informational and "
     "never affect the exit code; zero registered instances is not a failure)."),
    # The exit-4 clause's ACTION, which is the whole reason 4 is distinct from
    # 2: a caller that reads 2 concludes nothing was written, and this sentence
    # is what tells the reader of a 4 that something was. Deleting it left the
    # suite green.
    ("skills/bn/reference/mutating.md", "an-unmeasured-run-must-be-read-back",
     "It is not a new failure mode: the mutation did apply, so read the view back and\n"
     "`bn save` before closing."),
    ("skills/bn/reference/runtime.md", "a-fanout-is-nonzero-only-if-every-row-failed",
     "the command exits non-zero only if **every** result failed"),
    ("CLAUDE.md", "a-reverted-sibling-does-not-affect-the-code",
     "does not affect the exit code (#118)"),
    ("README.md", "a-failed-verification-is-nonzero",
     "the CLI returns a nonzero exit code and reverts the whole mutation or batch."),
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
# one of two ways -- a cell above pins it to what the CLI really returns, or one
# of the number KINDS below excuses it as something that cannot be a code.
#
# What this does NOT catch, stated in full because the previous wording claimed
# otherwise ("in ANY phrasing"):
#
#   * a contract with no numeral, no number word and no form of the word "exit"
#     -- "a mutation the bridge refuses before apply exits with the bridge-error
#     code" was caught only once `exit\w*` joined the alphabet, and "the shell
#     status of a refused mutation is the bridge-error status" still is not;
#   * a code written in hex (`exit 0x4`) or as a number word past `hundred`;
#   * a code written as an inline-code literal in a sentence that names no exit
#     -- "a failed mutation returns `2` to the shell", which is what round 9
#     wrote. The `inline-code-span` kind below excuses it, because every one of
#     these documents spells offsets, defaults, field widths and JSON values in
#     backticks, and reading those as claims put a page limit and a struct
#     offset in the population. The same sentence without the backticks is
#     still caught, and so is any phrasing that names an exit.
#
# All three are implausible phrasings for a doc, which is the only reason they
# are accepted rather than paid for: admitting `0x<hex>` would put every example
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
# The words this tree puts in front of a number when it is ASSERTING that
# number as a value -- the shape `0 = success`, `a refused op returns 3`,
# `status 2 is a bridge error` and `Exit 5 is reserved` all share. A digit
# introduced by one of these is never excused as a quantity. `exit` itself is
# deliberately absent: a line saying it is in the population for the WORD,
# whatever its digits do, so blacklisting it here would only be decoration.
_ASSERTS_A_VALUE = (r"codes?|status|statuses|returns?|returned|is|are|be|was|were|"
                    r"means?|says?|reserved|yields?|gives?|equals?")
# The units and counted nouns these documents measure in. A closed list,
# because the point of the kind is that the number is a MEASUREMENT; the
# open-ended half of the same idea is `quantity-beside-a-word` below, which
# takes any ordinary word as the thing being counted.
_UNITS = (r"bytes?|KB|MB|GB|kB|tokens?|ms|seconds?|minutes?|hours?|days?|rows?|chars?|"
          r"characters?|functions?|lines?|insns?|instructions?|commands?|layers?|callers?|"
          r"callsites?|sites?|args?|ops?|entries|files?|children|constructors?|slots?|"
          r"digest|directories|spills?|targets?|pointers?|symbols?")

# The KINDS of number an agent-facing doc carries that cannot be an exit code.
# This replaced a ledger of 366 sha1 fingerprints -- one per doc LINE, across 11
# documents -- plus the machinery that rendered and stale-checked it (#731).
# Every entry there was a hash of a line compared against a stored hash of that
# same line, so no product bug could trip it and every prose edit did, twice:
# the failure did not hand you the entry, it had to be re-derived.
#
# Each entry below is a statement about a kind of number rather than about one
# sentence, so a new sentence of a known kind -- another measurement, another
# `--limit` example -- needs no edit here. Anything no kind excuses stays loud,
# exactly as the ledger left it, and the escape hatch for a genuinely novel
# numbered sentence is a cell in `_EXIT_CODE_PINS` or a rewording, never a new
# entry here.
#
# An EXIT WORD is never excused by any of these -- only numbers are, see
# `_number_lines` -- so a line that names an exit stays in the population
# however it is phrased, and `DECLARED_NON_EXIT_CODE_WORDS` still has to rule on
# it one line at a time.
_NON_CLAIM_NUMBERS = (
    # A fenced block is program text: transcripts, command syntax, struct
    # offsets, JSON envelopes. Its exit words are still swept, so a contract
    # stated as a `# Exit 5 means ...` comment inside a fence is still loud.
    ("fenced-block", r"(?ms)^ {0,3}```.*?^ {0,3}```"),
    # ...and so is an inline code span, for the same reason: `--limit 50`,
    # `items[0]`, `total: 0`, `0x401000`. Spans are paired left to right, the
    # way markdown pairs them.
    ("inline-code-span", r"`[^`\n]*`"),
    # A section heading's own ordinal or count ("## 6. Mutation flow",
    # "### Step 1 --- preview first", "### Two-Process Model"): a title.
    ("heading-ordinal", r"^#{1,6} .*"),
    # The marker of an ordered list: the "3." of a numbered procedure step.
    ("ordered-list-marker", r"^[ \t>]*\d+\.(?= )"),
    # A command invocation and its arguments, in a fence or in a blockquote:
    # `bn xrefs <fn-or-addr> --limit 20`, `uv run pytest -n 8`.
    ("command-invocation",
     r"^[ \t>]*(?:[A-Z][A-Z0-9_]*=\S+ +)*(?:bn|bn-agent|uv|python|make|pytest)\b.*"),
    # A spelled-out number in front of the thing it counts ("two parts", "three
    # files") or inside a compound ("zero-based", "one-off"), plus a bare `one`
    # or `zero`, which are ordinary English words as well as numbers ("a
    # stripped one", "read as a zero"). A bare `two`..`hundred` is a VALUE, not
    # a quantity, and stays loud: "a refused op returns three" is how round 9
    # stated a false contract with no digit in it at all.
    ("spelled-out-quantity",
     rf"(?i)(?:\b\w+-(?:{_NUMBER_WORDS})\b|\b(?:{_NUMBER_WORDS})[- ](?=[\w`*])\w*"
     r"|\b(?:one|zero)\b)"),
    # A digit with an ordinary word in front of it: "median 0", "op 13",
    # "default 2", "2000 functions", "last **14** day-directories". The word
    # must not be one of `_ASSERTS_A_VALUE`, because that is the shape a stated
    # code has.
    ("quantity-beside-a-word",
     rf"\b(?!(?:{_ASSERTS_A_VALUE})\b)[A-Za-z][\w.]*[ \u00a0\u2013-]\*{{0,2}}~?\d+"),
    # A digit in front of its unit, which is a measurement however the sentence
    # around it reads: "~225 bytes", "120 seconds", "20 %", "514 pointers".
    ("measurement-with-a-unit",
     rf"(?<![\w.])~?\d+(?:[.,]\d+)?\*{{0,2}}[ \u00a0\u2013-]?(?:%|(?:{_UNITS})\b)"),
    # A number bounded by a comparison, which states a threshold: ">= 2", "> 0".
    ("bounded-by-a-comparison", r"[\u2264\u2265<>]=?[ \u00a0]?~?\d+"),
    # A proportion: "0 of 387 callsites", "127 of which".
    ("proportion", r"(?<![\w.])~?\d+ of (?:~?\d+|which|them)\b"),
    # A range or an alternative pair, which names no single value: "a 3-5 line
    # summary", "2-3 layers", "typed as 0/1 args", "--lines 40:80".
    ("range-or-pair", r"\b\d+(?:[\u2013/:-]\d+)+\b"),
    # A thousands group written with a space: "10 000 estimated tokens".
    ("grouped-thousands", r"\b\d{1,3}(?: \d{3})+\b"),
    # A number in a parenthetical annotation, or one closing it: "(median ~24)",
    # "(default 2)", the enumerators "(1)"/"(2)", "zero-extended to 4)".
    ("parenthetical-aside", r"\(\s*~?\d+\s*\)|~?\d+(?=\s*\))"),
    # A quoted literal, which is program text with quotes instead of backticks.
    ("quoted-literal", r'"\d+"?'),
    # A cross-reference to a numbered section: "(see section 2)".
    ("section-cross-reference", r"\u00a7 ?\d+"),
)


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

    An echo or a pin is applied to the document that states it, and the
    exit-code bullet is claimed by its EXTENT rather than by its wording.
    Applying a clause pattern document-wide pre-claimed every other line that
    reused a guarded clause's words, so appending "0 = success even when every
    instance failed" to `CLAUDE.md` -- a false contract -- produced no residual
    token and the sweep never saw the line (round 12). A cell pins one statement
    in one place; letting its wording immunise the rest of the document is the
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
        # The bullet IS the contract rather than an echo of it, and three cells
        # hold it: test_exit_code_bullet_documents_every_code_the_cli_can_return
        # derives the code set from the CLI package,
        # test_no_code_the_cli_can_return_is_undocumented asserts the two sets
        # EQUAL (so `; 5 = reserved` reds), and
        # test_the_exit_code_bullet_clauses_are_what_the_cli_actually_does runs
        # `_mutation_exit_code` over every case it states. So its digits are
        # accounted for THERE, and the sweep claims the bullet's span.
        start, end = _bullet_span("- Exit codes:", text)
        claimed[start:end] = b"\x01" * (end - start)
    return claimed


def _non_claim_numbers(text: str) -> bytearray:
    """The characters of *text* whose number one of the kinds above excuses."""
    excused = bytearray(len(text))
    for _, pattern in _NON_CLAIM_NUMBERS:
        for match in re.finditer(pattern, text, re.M):
            excused[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    return excused


def _number_lines(doc: str) -> list[tuple[str, str, list[str]]]:
    """(where, normalized line, residual tokens) for every line carrying a
    number that no cell pins and no kind excuses.

    The residual is per TOKEN, not per line. Excused per line, one entry covered
    every number on it: `README.md`'s contract paragraph carries eight
    cell-pinned digits and one prose word, so DELETING the cell that ties that
    paragraph to `_mutation_exit_code` left the module green. A kind excuses the
    number it matches and nothing else, so removing a pin leaves its digit loud.

    An exit WORD is never excused by a kind: the line is in the population for
    NAMING an exit, and no statement about a kind of number rules on that. It
    stays in the residual until a cell pins it or `DECLARED_NON_EXIT_CODE_WORDS`
    rules on it.
    """
    text = _doc_text(REPO / doc)
    claimed = _exit_code_claimed(doc, text)
    excused = _non_claim_numbers(text)
    rows = []
    for at, start, raw, normalized in _doc_lines(doc):
        residual = []
        for match in _NUMBER_TOKEN.finditer(raw):
            offset = start + match.start()
            if claimed[offset]:
                continue
            if excused[offset] and not _EXIT_WORD_TOKEN.match(match.group(0)):
                continue
            residual.append(match.group(0).lower())
        if residual:
            rows.append((f"line {at}", normalized, residual))
    return rows


def _declared_reasons(doc: str) -> list[str]:
    """The phrases `DECLARED_NON_EXIT_CODE_WORDS` rules on in *doc*."""
    return [phrase for declared_doc, phrase in DECLARED_NON_EXIT_CODE_WORDS
            if declared_doc == doc]


def _unaccounted_number_lines(doc: str) -> list[str]:
    """Lines carrying a number that no cell pins, no kind excuses and no
    declaration rules on.

    A declaration rules on the line's EXIT WORDS and nothing else, so a numbered
    claim appended to a declared line is still unaccounted -- excusing the whole
    line is how a paragraph carrying eight pinned digits became exempt from this
    half in the first place.
    """
    declared = _declared_reasons(doc)
    unaccounted = []
    for where, normalized, residual in _number_lines(doc):
        if any(phrase in normalized for phrase in declared):
            residual = [token for token in residual
                        if not _EXIT_WORD_TOKEN.match(token)]
        if residual:
            unaccounted.append(f"{where} ({', '.join(residual)}): {normalized[:140]}")
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
    unless a cell pins it, one of the `_NON_CLAIM_NUMBERS` kinds excuses it, or a
    declaration rules on its exit word -- and a new sentence stating an exit code
    is a new line with a number in it, whatever words it uses to say so.
    """
    unaccounted = _unaccounted_number_lines(doc)
    assert not unaccounted, (
        f"{doc} carries numbers no cell pins and no kind excuses, so an exit "
        "code stated here can drift from the code with every other guard green. "
        "If it states one, add a cell to _EXIT_CODE_ECHOES or _EXIT_CODE_PINS; "
        "if it is a kind of number this accounting has not met yet, add that "
        f"KIND to _NON_CLAIM_NUMBERS: {unaccounted}"
    )


# The sweep's remaining fail-open edge, and the one both of the last two rounds
# walked straight through. `_EXIT_WORDS` puts a line in the population for
# saying `exit` at all -- but `exit` ALSO means "leave the process", and `$?`
# can be named without stating a code, so a line carrying an exit word is
# sometimes a genuine non-claim. "This one is not a claim" is then a judgement
# no regex makes, and the accounting used to record it by OMISSION: nothing
# distinguished "we read this line and it states no code" from "nobody looked".
# Round 20 found two such lines that WERE claims and pinned them; round 21 found
# a third, one document over, because pinning the two that were named is a
# per-instance repair of a population defect.
#
# So the judgement is DECLARED, with its reason, and keyed by a PHRASE of the
# line it rules on: a line whose residual carries an exit word must be pinned by
# a cell or match one of these, and a NEW one fails until someone states which
# it is -- in a diff, where it can be argued with. Keyed by line NUMBER a reason
# followed the position rather than the text: swapping the contents of two
# declared lines left both cells green with each reason excusing the other line,
# and inserting a line above one re-pointed its declaration at a stranger.
# Keyed by a FINGERPRINT of the line it was unreviewable, and re-deriving the
# hash after a rewrap was the tax #731 retired. A phrase is content-keyed and
# readable, and one phrase rules on every line that repeats the same sentence,
# which is what "not a claim" means for a recurring obligation.
DECLARED_NON_EXIT_CODE_WORDS: dict[tuple[str, str], str] = {
    ("README.md", "only checks `$?` cannot read an unconfirmed write"):
        '`$?` names the variable a consumer reads; every digit on this line is '
        'pinned or echoed',
    ("skills/bn-kernel/SKILL.md", "after an eval-kernel exit/reset"):
        'an eval-kernel exit/reset, not an exit code',
    ("skills/bn-kernel/SKILL.md", "every reachable exit"):
        "'on every reachable exit' is a teardown obligation, not an exit code",
    ("skills/bn-kernel/SKILL.md", "A sibling exit 130 can still destroy"):
        "a SIGINT-killed sibling process's 130, not a code `bn` itself returns",
    ("skills/bn/reference/mutating.md", "does **not** change the exit code"):
        'the lead-in to the pinned sentence on the same line, which states both '
        'digits',
    ("skills/bn/reference/mutating.md", "a script that only checks `$?` sees"):
        '`$?` names the variable a consumer reads; the digit is echoed',
    ("skills/bn/reference/mutating.md", "exit codes are therefore the same"):
        'the lead-in to the two echoed digits on the following lines',
    ("skills/bn/reference/runtime.md",
     "the verified process can exit and its pid be reused"):
        'a process EXITING and its pid being reused, not an exit code',
    ("skills/bn/reference/runtime.md", "purged as soon as a proven owner exits"):
        'a proven owner EXITING, plus a warning not to key on the code at all; '
        'the restart digits are pinned two paragraphs down',
}

# The vocabulary that puts a line in the population for naming an EXIT rather
# than a number. Read off `_NUMBER_TOKEN`'s own alphabet so the two cannot drift.
_EXIT_WORD_TOKEN = re.compile(rf"^(?:{_EXIT_WORDS})$", re.I)


def _parked_exit_word_lines() -> set[tuple[str, str]]:
    """Every line whose residual NAMES an exit rather than a number, keyed by
    the document and the line's own normalized text.

    Keyed by LINE NUMBER, a reason followed the position rather than the text:
    swapping the contents of two declared lines left both cells green with each
    reason excusing the other line, and inserting a line above one re-pointed
    its declaration at a stranger. So this keys on content, and so does the
    declaration that rules on it.
    """
    return {(doc, normalized)
            for doc in EXIT_CODE_DOCS
            for _, normalized, residual in _number_lines(doc)
            if any(_EXIT_WORD_TOKEN.match(token) for token in residual)}


def test_no_parked_line_carries_an_undeclared_exit_word():
    """Every parked line that NAMES an exit must have been read and ruled on.

    The reason is the whole point: an entry says "this line's `exit` is not an
    exit code", which is a claim a reviewer can check and disagree with. Before
    this cell the same statement was made by writing nothing.
    """
    undeclared = sorted(
        f"{doc}: {normalized[:120]}"
        for doc, normalized in _parked_exit_word_lines()
        if not any(phrase in normalized for phrase in _declared_reasons(doc))
    )
    assert not undeclared, (
        "these parked lines name an exit and no cell pins them, so whether they "
        "state an exit code has been decided by omission. Pin the line if it "
        "states one; add a phrase of it to DECLARED_NON_EXIT_CODE_WORDS with "
        f"the reason if it does not: {undeclared}"
    )


def test_no_declared_exit_word_entry_is_stale():
    """...and the declarations stale-fail too, so the list cannot outlive the
    lines it rules on and become the next place a claim can park."""
    parked = _parked_exit_word_lines()
    stale = sorted(
        (doc, phrase) for doc, phrase in DECLARED_NON_EXIT_CODE_WORDS
        if not any(parked_doc == doc and phrase in normalized
                   for parked_doc, normalized in parked)
    )
    assert not stale, (
        "these declarations rule on a parked exit word that is no longer there "
        f"(the line moved, was pinned, or was rewritten): {stale}"
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
