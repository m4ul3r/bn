"""Docs-drift guards for `skills/bn/reference/` (#650, #651).

Agents read the skill reference as the API surface, not `--help`. Where the
reference omits a flag, agents conclude it does not exist: in one 5-agent dogfood
run **three** independently re-filed shipped features as missing capabilities, and
two wrote the same ad-hoc python to sort functions by size -- the workaround the
`--sort` flag was built to prevent.

So these are not style checks. Each assertion below encodes a documented claim that
the code must keep true (or a flag whose omission provably cost real work), and they
fail when either side drifts:

* every batch op kind and its required fields appear in `mutating.md` -- the batch
  path has no argparse layer, so an undocumented field name is discovered only by an
  atomic apply failing and reverting N good ops;
* the shipped `function list` / mutation flags agents re-invented stay documented;
* the compact mutation-status key table names exactly the keys the formatters emit
  -- #684 made that schema the signal that stops an agent discarding real work, so
  an omitted key is a consumer that never learns to check it;
* the JSON-envelope leaf keys the reference promises match what the handlers emit
  (`local list` is the exception that was documented in #248, lost in the SKILL.md ->
  reference/ split, and is now fixed at the source instead).
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parent.parent / "skills" / "bn" / "reference"
READING = REFERENCE / "reading.md"
MUTATING = REFERENCE / "mutating.md"
SKILL = REFERENCE.parent / "SKILL.md"

COMMAND_INDEX_HEADING = "## Command index"

# Top-level command groups an agent must be able to discover from the skill's
# Command index. Each is also asserted to exist in the live `_COMMANDS`
# registry, so the allow-list cannot outlive a command rename (#627).
REQUIRED_INDEX_GROUPS = ("capabilities", "dataflow", "exports", "go", "tag", "taint")


@pytest.fixture(scope="module")
def stub_engine():
    """A `binaryninja` stub that CANNOT outlive the module that asked for it.

    These fixtures used `sys.modules.setdefault`, which has no teardown: the
    stub stays in `sys.modules` for the rest of the session, so a module
    collected later that imports real symbols from the engine fails. That is not
    hypothetical -- the same pattern at import scope deterministically broke four
    unrelated modules in this suite. `monkeypatch.setitem` restores (and deletes
    a key that was absent), so the stub is gone when the module finishes.

    Still `setdefault` SEMANTICS: a real engine, if one is installed, is left
    alone rather than shadowed by a stub.
    """
    with pytest.MonkeyPatch.context() as patch:
        if "binaryninja" not in sys.modules:
            try:
                importlib.import_module("binaryninja")
            except ImportError:
                patch.setitem(sys.modules, "binaryninja", types.ModuleType("binaryninja"))
        yield


@pytest.fixture(scope="module")
def mutation_engine(stub_engine):
    """The engine module, imported against a stub `binaryninja` (no BN needed)."""
    return importlib.import_module("bn_agent_bridge.mutation_engine")


@pytest.fixture(scope="module")
def command_groups(stub_engine) -> set[str]:
    """The live top-level groups, from the @command registry (no BN needed)."""
    importlib.import_module("bn.commands")          # populates bn.cli._COMMANDS
    cli = importlib.import_module("bn.cli")
    groups = {spec["path"][0] for spec in cli._COMMANDS}
    assert groups, "bn.commands registered no commands; the allow-list is unchecked"
    return groups


@pytest.fixture(scope="module")
def command_paths(stub_engine) -> set[str]:
    """Every live command path (`"tag add"`, `"types show"`), from the registry.

    The index is only a map if what it advertises exists; pinning names against
    the registry is also what stops an allow-list outliving a rename (#627).
    """
    importlib.import_module("bn.commands")          # populates bn.cli._COMMANDS
    cli = importlib.import_module("bn.cli")
    paths = {" ".join(spec["path"]) for spec in cli._COMMANDS}
    assert paths, "bn.commands registered no commands; the index is unchecked"
    return paths


# The test modules in this PR's fence that stub the engine to read a registry
# without Binary Ninja installed.
ENGINE_STUBBING_TESTS = ("test_skill_reference_drift.py", "test_agent_docs.py")


def test_an_engine_stub_is_installed_only_in_a_form_that_restores():
    """`sys.modules.setdefault` has NO teardown, so a stub installed that way
    outlives the fixture -- and, at import scope, outlives the module: the same
    pattern deterministically broke four unrelated modules in this suite by
    shadowing the real engine for everything collected afterwards.

    Two halves, because either alone is satisfiable without the other: the
    mechanism really does restore, and nothing here still uses the form that
    does not. The second half is the one that goes red against the fixtures this
    replaced.
    """
    probe = "bn_absent_engine_probe"
    assert probe not in sys.modules, "pick a name nothing has imported"

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, probe, types.ModuleType(probe))
        assert sys.modules[probe] is not None
    assert probe not in sys.modules, "the patched stub outlived its context"

    leaking = sorted(
        f"{name}:{node.lineno}"
        for name in ENGINE_STUBBING_TESTS
        for node in ast.walk(ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault"
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "modules"
    )
    assert not leaking, (
        "these install a module stub with no teardown, so it outlives the test "
        f"that asked for it and can break a module collected later: {leaking}"
    )


def test_skill_command_index_names_every_required_group(command_groups):
    """#627: `skills/bn/SKILL.md` is the first thing an agent reads, and its
    Command index silently omitted `tag`, `taint`, `dataflow`, `exports`, `go`
    and `capabilities` -- so an agent told to bookmark findings and run
    source->sink analysis never reached `bn tag add --type Bookmarks` or
    `bn taint forward`.

    Scoped to the index section (the Reference block names files, not groups),
    and paired with the registry so the requirement fails if a group is renamed
    away rather than pinning a name that no longer exists.
    """
    text = SKILL.read_text(encoding="utf-8")
    assert COMMAND_INDEX_HEADING in text, f"{COMMAND_INDEX_HEADING!r} is gone from SKILL.md"
    index = text.split(COMMAND_INDEX_HEADING, 1)[1].split("\n## ", 1)[0]
    missing = [group for group in REQUIRED_INDEX_GROUPS
               if not re.search(rf"`{group}\b", index)]
    assert not missing, f"groups missing from the SKILL.md Command index: {missing}"
    renamed = sorted(set(REQUIRED_INDEX_GROUPS) - command_groups)
    assert not renamed, f"the index requires groups the CLI registry does not have: {renamed}"


def _index_section() -> str:
    """The Command index body, WITHOUT the remainder of its heading line."""
    text = SKILL.read_text(encoding="utf-8")
    sections = text.count(COMMAND_INDEX_HEADING)
    assert sections == 1, (
        f"expected exactly one {COMMAND_INDEX_HEADING!r} section, found {sections}: "
        "this guard reads the FIRST one, so a second would sit outside the sweep "
        "entirely and every command it advertises would stop being checked"
    )
    after = text.split(COMMAND_INDEX_HEADING, 1)[1]
    body = after.split("\n", 1)[1] if "\n" in after else ""
    return body.split("\n## ", 1)[0]


# One index entry, WHOLE: a backticked command whose last word may carry
# `/`-separated alternatives and an optional `[subcommand]` marker, plus an
# optional parenthetical gloss. Matched with `fullmatch` against the whole
# entry, never searched for inside it -- a `re.findall` over the line silently
# drops every entry it cannot read, which is the same "advertised but
# undocumented" hole this guard exists to close.
_INDEX_ENTRY = re.compile(
    r"`(?P<cmd>[a-z][a-z ]*(?:/[a-z]+)*)(?: \[(?P<optional>[a-z]+)\])?`(?: \([^)]*\))?"
)

# Any list item introducing a group, whatever the marker or indentation. The
# index guard must REFUSE a line it cannot read rather than drop it from the
# sweep: a `- ` -> `* ` marker swap renders identically in Markdown and used to
# take a whole line's commands out of the check silently.
_INDEX_LINE = re.compile(r"\s*[-*+] +\*\*(?P<group>[^*]+)\*\*")

# ```bash blocks are how every reference presents a command an agent can run.
_BASH_BLOCK = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.S)

# A runnable invocation: a line whose first word is `bn`.
_RUNNABLE = re.compile(r"^[ \t]*bn +(?P<rest>[a-z].*)$", re.M)

# The index line's TERMINAL pointer -- what follows the arrow. Searching the
# whole line instead returns the first `reference/*.md` it happens to mention,
# so a reference named inside an entry's parenthetical gloss silently stood in
# for the file the line actually points an agent at.
_INDEX_REFERENCE = re.compile(r"`(reference/[a-z_]+\.md)`")


def _index_pointer(line: str) -> str:
    assert "\u2192" in line, f"index line has no `->` pointer: {line}"
    tail = line.split("\u2192", 1)[1]
    pointers = _INDEX_REFERENCE.findall(tail)
    assert len(pointers) == 1, (
        f"an index line must point at exactly one reference file, found {pointers}: {line}"
    )
    return pointers[0]


def _expand(entry: re.Match[str]) -> list[str]:
    """`struct field set/rename/delete` -> the three full command strings.

    An `[optional]` subcommand yields BOTH forms: `types [show]` advertises
    `types` and `types show`, and both have to be real and documented. Parsing
    the marker and then discarding it is how `tag [frobnicate]` passed on the
    strength of `bn tag` alone.
    """
    words = entry.group("cmd").split()
    prefix, last = words[:-1], words[-1]
    commands = [" ".join([*prefix, alt]) for alt in last.split("/")]
    if entry.group("optional"):
        commands += [f"{cmd} {entry.group('optional')}" for cmd in commands]
    return commands


def _index_entries(line: str) -> list[str]:
    """Every comma-separated entry between the index line's em dash and its
    `->` pointer. Returns them RAW so the caller can reject what it cannot read
    rather than skipping it."""
    assert "—" in line and "→" in line, f"unrecognised index line shape: {line}"
    body = line.split("—", 1)[1].split("→", 1)[0]
    return [entry.strip() for entry in body.split(",") if entry.strip()]


def _documented_commands(text: str, command_paths: set[str]) -> set[str]:
    """The command paths a reference actually SHOWS as runnable.

    Each `bn ...` line resolves to its longest registered command path, so a
    match is exact rather than a prefix: `bn types show <name>` documents
    `types show` and NOT `types`, which a prefix check would let stand in for it.
    """
    documented: set[str] = set()
    for match in _RUNNABLE.finditer("\n".join(_BASH_BLOCK.findall(text))):
        words = match.group("rest").split()
        for size in range(min(3, len(words)), 0, -1):
            candidate = " ".join(words[:size])
            if candidate in command_paths:
                documented.add(candidate)
                break
    return documented


# The groups the Command index must carry. Pinned as a SET, not a floor: a floor
# is satisfied by "still at least four lines", which is exactly how a line that
# stopped matching disappeared unnoticed. `_INDEX_LINE` refusing what it cannot
# read covers the other half -- nothing silently leaves the sweep.
EXPECTED_INDEX_GROUPS = frozenset({"Read", "Mutate", "Discover", "Session", "Escape hatch"})


def _index_lines() -> list[str]:
    lines = [line for line in _index_section().splitlines() if line.strip()]
    unreadable = [line for line in lines if not _INDEX_LINE.match(line)]
    assert not unreadable, (
        "every line in the Command index must be a readable `- **Group** ...` "
        f"entry, or this guard silently stops checking it: {unreadable}"
    )
    groups = {_INDEX_LINE.match(line).group("group").strip() for line in lines}
    assert groups == EXPECTED_INDEX_GROUPS, (
        f"the Command index groups changed: expected {sorted(EXPECTED_INDEX_GROUPS)}, "
        f"parsed {sorted(groups)}"
    )
    return lines


@pytest.mark.parametrize("line", _index_lines(),
                         ids=lambda line: line.strip()[:24])
def test_skill_index_entries_are_documented_where_the_index_points(line, command_paths):
    """#627: naming a group in the index is only half the map -- each line ends
    in `-> reference/<file>.md`, so every command it advertises must actually be
    documented in THAT file. The first cut of the widened index advertised a
    mutation whose reference never mentioned it, and pointed the read line at a
    reference that catalogued one of its commands in a different file; both send
    an agent that followed the pointer to a file with no entry for the command
    it came for.

    Three things this guard must not do, because each restores the defect while
    staying green, and each was caught doing it: DROP a line it does not
    recognise (a `- ` -> `* ` marker swap renders identically and took a whole
    line out of the sweep), SKIP an entry it cannot parse, and accept a passing
    mention in prose as documentation. So every line in the section must be a
    readable index line, every entry must parse, every advertised command must
    exist in the live `@command` registry, and the documentation bar is a
    runnable `bn <command>` line inside a ```bash block, resolved to its exact
    command path rather than matched as a prefix.
    """
    assert _INDEX_LINE.match(line), (
        "every line in the Command index must be a readable `- **Group** ...` "
        f"entry, or this guard silently stops checking it: {line!r}"
    )
    reference = _index_pointer(line)
    entries = _index_entries(line)
    assert entries, f"index line advertises nothing: {line}"
    unreadable = [entry for entry in entries if not _INDEX_ENTRY.fullmatch(entry)]
    assert not unreadable, (
        "these index entries are not a plain backticked command, so this guard "
        f"cannot check them and must not pretend it did: {unreadable}"
    )
    commands = [full for entry in entries
                for full in _expand(_INDEX_ENTRY.fullmatch(entry))]
    unregistered = sorted(set(commands) - command_paths)
    assert not unregistered, (
        f"the SKILL.md index advertises commands the CLI registry does not have: "
        f"{unregistered}"
    )
    text = (REFERENCE.parent / reference).read_text(encoding="utf-8")
    documented = _documented_commands(text, command_paths)
    missing = [cmd for cmd in commands if cmd not in documented]
    assert not missing, (
        f"the SKILL.md index points at {reference} for commands that "
        f"file never shows as a runnable command: {missing}"
    )


def test_mutating_reference_documents_every_batch_op(mutation_engine):
    """#650: `mutation_engine.REQUIRED_FIELDS` defines 16 batch ops; `mutating.md`
    used to name TWO. Every op AND every required field name must be documented --
    a guessed field name fails at apply time and, the batch being atomic, takes
    every good op with it (one agent lost 12 that way)."""
    text = MUTATING.read_text(encoding="utf-8")
    missing_ops = [op for op in mutation_engine.REQUIRED_FIELDS if f"`{op}`" not in text]
    assert not missing_ops, f"batch ops undocumented in mutating.md: {missing_ops}"

    missing_fields = []
    for op, fields in mutation_engine.REQUIRED_FIELDS.items():
        for field in fields:
            if f"`{field}`" not in text:
                missing_fields.append(f"{op}.{field}")
    assert not missing_fields, (
        f"required batch fields undocumented in mutating.md: {missing_fields}")

    for op, groups in mutation_engine.REQUIRED_ONE_OF.items():
        for group in groups:
            for field in group:
                assert f"`{field}`" in text, f"{op} one-of field {field!r} undocumented"


def test_mutating_reference_documents_the_output_flags(mutation_engine):
    """#650/#645: three agents reported `batch apply` as having no summary mode and a
    `proto set` as unavoidably flooding context. Both flags shipped; only the docs
    were missing."""
    text = MUTATING.read_text(encoding="utf-8")
    for flag in ("--summary", "--quiet", "--verbose", "--format json", "--out"):
        assert flag in text, f"mutation output flag {flag!r} undocumented"
    # The compact default itself has to be stated, or an agent still expects the
    # old full-JSON default and parses the wrong thing.
    assert "status line" in text or "status summary" in text


def _documented_compact_status_keys() -> set[str]:
    """The keys named in the first column of `mutating.md`'s "Compact status
    keys" table. One row may name several keys (the counts share a row), so
    collect every backticked identifier in that cell."""
    section = MUTATING.read_text(encoding="utf-8").split("### Compact status keys", 1)
    assert len(section) == 2, "the compact-status key table section is gone"
    keys: set[str] = set()
    for line in section[1].splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| key "):
            if keys:
                break                       # past the end of the table
            continue
        keys.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line.split("|")[1]))
    return keys


def test_mutating_reference_documents_every_compact_status_key():
    """#684 made `measured` -- and the null-vs-zero count semantics around it --
    load-bearing for a JSON control loop, and the answer to "a value nobody is
    told to read is not loud enough" is that the compact schema is DOCUMENTED.
    A table that silently drifts from the emitted keys is the same
    remembered-exception failure mode as a forgotten `summary_transform`: it
    shipped omitting `kind` and `prototype_user_type_residue` (#630's residue
    flag, which a control loop must see). Pin the documented key set to the
    union of what `_mutation_summary` and `_go_rename_summary` actually emit
    across the measured, unmeasured and residue shapes."""
    from bn.formatters import _go_rename_summary, _mutation_summary

    measured = _mutation_summary({"success": True, "committed": True,
                                  "rolled_back": False, "results": [{"status": "noop"}]})
    unmeasured = _mutation_summary({"success": True, "committed": True, "results": []})
    residue = _mutation_summary({"success": False, "committed": False,
                                 "rolled_back": True, "message": "override stuck",
                                 "prototype_user_type_residue": True,
                                 "results": [{"status": "rollback_failed"}]})
    go = _go_rename_summary({"kind": "go_rename", "success": True, "committed": True,
                             "rolled_back": False, "go_renamed_candidates": 2,
                             "go_committed_count": 2, "go_verified_count": 2,
                             "go_failed_count": 0, "skipped_user_named": 1})
    emitted = set(measured) | set(unmeasured) | set(residue) | set(go)
    assert emitted == _documented_compact_status_keys()
    # ...and the table's "always present except prototype_user_type_residue"
    # caveat is itself true.
    assert "prototype_user_type_residue" in residue
    assert "prototype_user_type_residue" not in measured
    # #685: the two summary builders must keep emitting the SAME key set -- the
    # nulled unmeasured counts must not widen one side only.
    assert set(measured) == set(unmeasured) == set(go)


def test_reading_reference_documents_the_function_list_flags():
    """#650: `--sort` / `--reverse` / `--min-size` / `--demangle` all ship, and all
    were absent from `reading.md` -- so two agents wrote the same python to sort
    functions by size."""
    text = READING.read_text(encoding="utf-8")
    for flag in ("--sort", "--reverse", "--min-size", "--demangle", "--named", "--unnamed"):
        assert flag in text, f"`function list` flag {flag!r} undocumented in reading.md"


def test_reading_reference_leaf_keys_match_the_handlers(mutation_engine):
    """#651: `reading.md` claimed `items` is "always" the container while
    `local list` emitted only `locals` -- so `jq '.items[]'` on a function with 40
    recovered locals reported nothing, which reads like "no recovered variables".
    The exception was documented once (#248) and lost in a file split, so assert the
    CODE now satisfies the claim rather than re-documenting the exception."""
    read_decompile = importlib.import_module("bn_agent_bridge.read_decompile")

    class _Fn:
        start = 0x401120
        name = "handle_request"
        raw_name = "handle_request"

    class _Ctx:
        def _resolve_view(self, selector):
            return object()

        def _find_function(self, bv, identifier, contained=False):
            return _Fn()

        def _containment_meta(self, identifier, func):
            return None

    vars_mod = importlib.import_module("bn_agent_bridge.vars")
    original = vars_mod._list_locals
    try:
        vars_mod._list_locals = lambda fn: [{"name": "var_c", "is_parameter": False}]
        result = read_decompile._list_locals_for_function(_Ctx(), None, "handle_request")
    finally:
        vars_mod._list_locals = original

    assert result["items"] == result["locals"], "`items` must be the same list as `locals`"
    assert result["kind"] == "locals"
    text = READING.read_text(encoding="utf-8")
    assert "`local list` → `.items[]" in text or "local list` → `.items[]" in text
    # The old absolute claim must not have been restored verbatim.
    assert "`items` is **always** the data container" not in text


def test_reading_reference_documents_hex_string_addresses():
    """#653.7: `{"address": "0x401ed8"}` is a hex STRING everywhere; arithmetic needs
    `int(x, 16)`. Consistent, but undocumented -- one traceback to discover."""
    text = READING.read_text(encoding="utf-8")
    assert "hex STRING" in text or "hex strings" in text.lower()
