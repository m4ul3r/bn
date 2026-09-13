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
# Command index, mapped to the index LINE each must be discoverable from. Each
# group is also asserted to exist in the live `_COMMANDS` registry, so the
# allow-list cannot outlive a command rename (#627).
#
# The placement is the point: asking only whether a group appears SOMEWHERE in
# the index let the whole read-side entry for `tag` and `go` be deleted while
# their mutate-side entries kept the guard green -- and an agent that cannot
# find `tag list` on the Read line does not run it.
REQUIRED_INDEX_GROUPS = {
    "capabilities": ("Discover",),
    "dataflow": ("Read",),
    "exports": ("Read",),
    "taint": ("Read",),
    "tag": ("Read", "Mutate"),
    "go": ("Read", "Mutate"),
}

# Commands deliberately NOT advertised in the Command index. This is the
# complement of the population below: every registered command path must appear
# in the index unless it is named here, so a new command is a documentation
# failure until someone classifies it. The previous cut listed the six groups
# the bug report happened to name, which is fail-OPEN -- deleting the index
# entries for `evidence`, `trace` and `class` left every doc test green.
#
# Each name is asserted live, so an exemption cannot outlive a rename -- and
# each group's REASON is asserted too, not merely written down. Round 8 found a
# real documentation gap parked in the "catalogued in a reference" group, which
# is exactly what an unasserted reason is for.

# Sticky per-repo pins. Advertising these to an agent is actively harmful: they
# are one shared file and clobber a concurrent session. ASSERTED: not
# advertised anywhere in the index.
INDEX_EXEMPT_STICKY = frozenset({
    "instance use", "instance clear", "target use", "target clear",
})

# Reached through an advertised command instead. ASSERTED: the command named as
# the way in IS advertised in the index.
INDEX_EXEMPT_REACHED_VIA = {
    "instance list": "session list",
    "instance find": "session list",
    "instance gc": "session stop",
    "session restart": "session start",
    "session status": "session list",
    "target close": "target list",
    "exports list": "exports",
    "rename": "symbol rename",
}

# Host-side tooling, not analysis surface. ASSERTED: live only -- there is no
# stronger claim to make about a command an agent is not meant to reach for.
INDEX_EXEMPT_TOOLING = frozenset({
    "doctor", "help", "plugin install", "skill install",
})

# Deeper read/write surface the index reaches through its group entry.
# ASSERTED: each is catalogued somewhere under `skills/`, which is the stated
# reason. `evidence virtual-call` was in this group while appearing in no
# reference at all.
INDEX_EXEMPT_CATALOGUED = frozenset({
    "data retype", "data symbols", "data vars",
    "evidence calls", "evidence orient", "evidence surface", "evidence virtual-call",
    "function cfg", "function structured-il",
})

INDEX_EXEMPT_COMMANDS = (INDEX_EXEMPT_STICKY | frozenset(INDEX_EXEMPT_REACHED_VIA)
                         | INDEX_EXEMPT_TOOLING | INDEX_EXEMPT_CATALOGUED)


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


TESTS = Path(__file__).resolve().parent

# Writing to `sys.modules` without restoring it. Both write shapes AND every
# route to the mapping, because each narrower cut was escaped by the next one:
# first only `setdefault`, then only `sys.modules` spelled literally, while
# `m = sys.modules; m[name] = stub`, `from sys import modules`,
# `vars(sys)["modules"]` and `getattr(sys, "modules")` all reach the same dict.
# Round 7 escaped it twice more: `sys.__dict__["modules"][k] = v` was a route
# the scan did not know, and `sys.modules |= {...}` writes the mapping WHOLE
# rather than a subscript of it, which the write side did not look for.
#
# So the route test accepts any `[...]["modules"]` lookup (a namespace dict by
# any spelling), and the write test accepts the mapping itself as a target, not
# only a subscript of it.
#
# `monkeypatch.setitem(sys.modules, ...)` and `MonkeyPatch.context()` pass the
# mapping as an ARGUMENT and put it back, so they are not writes ON it and are
# correctly invisible here.
_SYS_MODULES_MUTATORS = frozenset({
    "setdefault", "update", "pop", "popitem", "clear", "__setitem__", "__delitem__",
})


def _modules_mapping_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to the module table anywhere in *tree*."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sys":
            aliases |= {alias.asname or alias.name
                        for alias in node.names if alias.name == "modules"}
    for _ in range(4):                       # `a = sys.modules; b = a; c = b`
        grew = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not _is_modules_mapping(node.value, aliases):
                continue
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and name.id not in aliases:
                        aliases.add(name.id)
                        grew = True
        if not grew:
            break
    return aliases


def _is_modules_mapping(node: ast.expr, aliases: frozenset[str] | set[str] = frozenset()) -> bool:
    """Every expression that evaluates to the interpreter's module table."""
    if isinstance(node, ast.Attribute) and node.attr == "modules":
        return True                                   # sys.modules, s.modules
    if isinstance(node, ast.Name) and node.id in aliases:
        return True                                   # m = sys.modules
    if (isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant) and node.slice.value == "modules"):
        # vars(sys)["modules"], sys.__dict__["modules"], globals-style lookups:
        # a namespace dict by any spelling. Naming the spellings one at a time
        # is what let `sys.__dict__` through, and nothing under tests/ indexes
        # anything else by "modules".
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "modules")       # getattr(sys, "modules")


def _unrestored_sys_modules_writes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _modules_mapping_aliases(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SYS_MODULES_MUTATORS
                and _is_modules_mapping(node.func.value, aliases)):
            yield f"{path.name}:{node.lineno} the module table .{node.func.attr}()"
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if _is_modules_mapping(target, aliases):
                # The mapping itself rather than a key of it: `sys.modules |=
                # {...}`, `sys.modules = {...}`, `del sys.modules`. `m =
                # sys.modules` BINDS the alias and writes nothing, so a plain
                # name target is only a write when it is augmented in place.
                if isinstance(node, ast.AugAssign) or not isinstance(target, ast.Name):
                    yield f"{path.name}:{node.lineno} the module table written whole"
            elif isinstance(target, ast.Subscript) and _is_modules_mapping(target.value, aliases):
                yield f"{path.name}:{node.lineno} the module table written by subscript"


def test_an_engine_stub_is_installed_only_in_a_form_that_restores():
    """A module stub that outlives the module that installed it shadows the real
    engine for everything collected afterwards -- four unrelated modules in this
    suite went down that way, deterministically, as a collection error.

    Two halves, because either alone is satisfiable without the other: the
    mechanism really does restore, and no test module anywhere still writes
    `sys.modules` in a form that does not.

    The population is EVERY module under `tests/`, and every write shape. The
    previous cut named the two modules this change happened to touch and the one
    call shape it happened to use, so a third module's stub and a plain
    `sys.modules[name] = ...` inside one of those two both stayed green while
    reproducing the original defect exactly.
    """
    probe = "bn_absent_engine_probe"
    assert probe not in sys.modules, "pick a name nothing has imported"

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, probe, types.ModuleType(probe))
        assert sys.modules[probe] is not None
    assert probe not in sys.modules, "the patched stub outlived its context"

    modules = sorted(TESTS.rglob("*.py"))
    assert len(modules) > 20, f"only {len(modules)} test modules found; check the glob"
    leaking = sorted(write for path in modules
                     for write in _unrestored_sys_modules_writes(path))
    assert not leaking, (
        "these write sys.modules with no teardown, so the entry outlives the "
        f"module that asked for it and can break one collected later: {leaking}"
    )


def _advertised_commands(line: str) -> list[str]:
    """Every command path an index LINE advertises, through the one parser.

    The placement check used to `re.search` the raw line, which includes the
    tail after the `->` pointer that `_index_entries` never reads -- so a whole
    read-side entry could be deleted and re-satisfied by a bare backticked
    token in the tail. Two deciders of "where the index advertises a group",
    disagreeing. This is the only one.
    """
    entries = _index_entries(line)
    assert entries, f"index line advertises nothing: {line}"
    unreadable = [entry for entry in entries if not _INDEX_ENTRY.fullmatch(entry)]
    assert not unreadable, (
        "these index entries are not a plain backticked command, so this guard "
        f"cannot check them and must not pretend it did: {unreadable}"
    )
    return [full for entry in entries
            for full in _expand(_INDEX_ENTRY.fullmatch(entry))]


def _index_lines_by_group() -> dict[str, str]:
    lines: dict[str, str] = {}
    for line in _index_section().splitlines():
        heading = _INDEX_LINE.match(line)
        if heading:
            lines[heading.group("group").strip()] = line
    return lines


def test_skill_command_index_names_every_required_group(command_groups):
    """#627: `skills/bn/SKILL.md` is the first thing an agent reads, and its
    Command index silently omitted `tag`, `taint`, `dataflow`, `exports`, `go`
    and `capabilities` -- so an agent told to bookmark findings and run
    source->sink analysis never reached `bn tag add --type Bookmarks` or
    `bn taint forward`.

    This half is PLACEMENT: a read-side entry deleted while the mutate-side one
    remains is still a group an agent cannot find where it looks. Decided from
    the entries the index parser reads, never from the raw line.
    """
    lines = _index_lines_by_group()
    missing = sorted(
        f"{group} under **{where}**"
        for group, placements in REQUIRED_INDEX_GROUPS.items()
        for where in placements
        if where not in lines
        or group not in {command.split()[0] for command in _advertised_commands(lines[where])}
    )
    assert not missing, (
        f"the SKILL.md Command index does not advertise these where an agent "
        f"looks for them: {missing}; index lines are {sorted(lines)}"
    )
    renamed = sorted(set(REQUIRED_INDEX_GROUPS) - command_groups)
    assert not renamed, f"the index requires groups the CLI registry does not have: {renamed}"


def test_skill_command_index_advertises_every_registered_command(command_paths):
    """...and this half is POPULATION, derived from the registry rather than from
    the names one bug report happened to list.

    Requiring only those six names is fail-open: `evidence`, `trace`, `class`,
    `struct show` and `struct field set/rename/delete` were all deleted from the
    index with every doc test green. So every registered command path must be
    advertised unless it is declared exempt, and a new command is a
    documentation failure until someone classifies it either way.
    """
    advertised = {command for line in _index_lines()
                  for command in _advertised_commands(line)}
    stale = sorted(INDEX_EXEMPT_COMMANDS - command_paths)
    assert not stale, (
        f"these exemptions name commands the registry does not have: {stale}; a "
        "stale exemption silently drops a real command from the requirement"
    )
    unadvertised = sorted(command_paths - advertised - INDEX_EXEMPT_COMMANDS)
    assert not unadvertised, (
        "the SKILL.md Command index -- the map an agent reads first -- does not "
        f"advertise these registered commands: {unadvertised}. Add an index "
        "entry, or declare the exemption in INDEX_EXEMPT_COMMANDS with a reason"
    )


def test_every_index_exemption_states_a_reason_that_is_true(command_paths):
    """An exemption is a claim, and an unasserted claim is where a real gap
    hides: `evidence virtual-call` sat in the "catalogued in a reference" group
    while appearing in no reference at all. So each group's stated reason is
    checked here, and the groups together must be exactly the exemption set."""
    groups = (INDEX_EXEMPT_STICKY | frozenset(INDEX_EXEMPT_REACHED_VIA)
              | INDEX_EXEMPT_TOOLING | INDEX_EXEMPT_CATALOGUED)
    assert groups == INDEX_EXEMPT_COMMANDS, (
        "every exemption must sit in exactly one reason group, or its reason is "
        f"unchecked: {sorted(groups ^ INDEX_EXEMPT_COMMANDS)}"
    )
    stale = sorted(INDEX_EXEMPT_COMMANDS - command_paths)
    assert not stale, (
        f"these exemptions name commands the registry does not have: {stale}; a "
        "stale exemption silently drops a real command from the requirement"
    )
    advertised = {command for line in _index_lines()
                  for command in _advertised_commands(line)}
    leaked = sorted(INDEX_EXEMPT_STICKY & advertised)
    assert not leaked, (
        "these sticky per-repo pins are exempt BECAUSE advertising them is "
        f"harmful, yet the index advertises them: {leaked}"
    )
    missing_way_in = sorted(
        f"{command} -> {via}" for command, via in INDEX_EXEMPT_REACHED_VIA.items()
        if via not in advertised
    )
    assert not missing_way_in, (
        "these are exempt because an agent reaches them through another "
        f"command, but that command is not advertised either: {missing_way_in}"
    )
    unresolved = sorted(set(INDEX_EXEMPT_REACHED_VIA.values()) - command_paths)
    assert not unresolved, (
        f"these ways in are not registered commands: {unresolved}"
    )
    skills = "\n".join(path.read_text(encoding="utf-8")
                       for path in sorted(SKILL.parent.parent.rglob("*.md")))
    uncatalogued = sorted(command for command in INDEX_EXEMPT_CATALOGUED
                          if f"bn {command}" not in skills and f"`{command}`" not in skills)
    assert not uncatalogued, (
        "these are exempt from the index BECAUSE they are catalogued in a "
        f"reference, and they are catalogued nowhere under skills/: {uncatalogued}"
    )


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
    """Every non-blank line of the Command index, unfiltered.

    Unfiltered on purpose: filtering to the lines this guard can READ is how a
    `- ` -> `* ` marker swap took a whole line out of the sweep silently, and
    the per-line cell below asserts readability itself.

    This used to assert here, at parametrize time -- so a doc slip became
    `Interrupted: 1 error during collection`, which aborts the whole pytest
    session and silences every other module's result. One red cell is the
    correct blast radius; `test_the_command_index_is_shaped_as_this_guard_reads_it`
    owns the section-level claims.
    """
    return [line for line in _index_section().splitlines() if line.strip()]


def test_the_command_index_is_shaped_as_this_guard_reads_it():
    """The section-level claims, as a failing TEST rather than a collection
    error: every line is readable, and the group set is pinned as a SET rather
    than a floor ("still at least four lines" is how a line that stopped
    matching disappeared unnoticed)."""
    lines = _index_lines()
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
