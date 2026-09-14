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

# Sticky per-repo pins. Advertising these is actively harmful: they write one
# shared per-project file and clobber a concurrent session.
#
# The old reason was "not advertised anywhere in the index", which is satisfied
# by the very act of removing an entry -- so ANY command could be dropped into
# this group and the guard stayed green (round 13). The reason is now POSITIVE
# and derived from the code in BOTH directions: an exempt command's handler must
# write the sticky state, and every registered command whose handler writes it
# must be here. A command that does not touch the pin cannot be parked in this
# group, which is what makes the group an exemption rather than a free pass.
INDEX_EXEMPT_STICKY = frozenset({
    "instance use", "instance clear", "target use", "target clear",
})

# A true ALIAS: the same registered handler under a second path, so advertising
# the other path advertises this behaviour.
#
# This group used to hold eight commands whose stated way-in was a DIFFERENT
# command, asserted only to be advertised itself -- never to actually reach the
# exempt one, so `"<any read command>": "session list"` passed (round 13). The
# six that were not aliases are now advertised in the index like everything
# else; the two that remain are aliases the registry itself proves, because
# both paths resolve to one handler function.
INDEX_EXEMPT_ALIASES = {
    "exports list": "exports",
    "rename": "symbol rename",
}

# `doctor`, `help`, `plugin install` and `skill install` used to sit in an
# INDEX_EXEMPT_TOOLING group whose stated reason was "live only -- there is no
# stronger claim to make", i.e. nothing. They are advertised now.

# There used to be a fourth group here: nine deeper read/write commands exempt
# BECAUSE they were "catalogued in a reference". Its reason failed three times.
# Round 8 found `evidence virtual-call` in the group while it appeared in no
# reference at all. Round 11 found the repair accepting any backticked mention
# of the bare name -- which a sentence saying the command is UNDOCUMENTED also
# satisfies. Round 12 found the second repair, which required that sentence to
# be a structural entry, satisfied by writing the denial as a `> ` callout.
#
# A reason that keeps being satisfiable is not a reason that needs hardening
# again: an exemption is a hole someone promised not to look through, and the
# only reliable way to close it is to stop making the promise. All nine are now
# advertised in the Command index like every other registered command, which is
# what #627 asked for in the first place -- the exemption was papering over the
# very issue it was meant to serve.
INDEX_EXEMPT_COMMANDS = INDEX_EXEMPT_STICKY | frozenset(INDEX_EXEMPT_ALIASES)


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


def _registered_handlers() -> dict[str, object]:
    """Command path -> the handler function the registry holds for it."""
    importlib.import_module("bn.commands")
    cli = importlib.import_module("bn.cli")
    return {" ".join(spec["path"]): spec["handler"] for spec in cli._COMMANDS}


# The sticky pin has exactly ONE writer, and the exemption's reason is a fact
# about the CODE: "this command writes the shared per-project pin". So the check
# has to be about THAT FUNCTION. Two earlier cuts answered a different question:
#
#   * `"session_state.update" in ast.unparse(call.func)` is a substring over
#     rendered source. It accepts any attribute chain that merely ENDS in those
#     two names (`namespace.session_state.update(...)` on a parameter), and it
#     misses the same function reached under another name
#     (`from bn.session_state import update as pin; pin(...)`).
#   * matching a registered handler to that source by `handler.__name__` is not
#     unique across modules: a dead same-named function in any module under
#     `src/bn` reclassified an unrelated registered command (round 14).
#
# So a call's callee is RESOLVED through the importing module's own bindings to
# a module-qualified name, a handler is matched by `__module__` + `__qualname__`
# rather than by bare name, and a handler that reaches the writer through a
# chain of resolvable calls counts too -- a wrapper is not an escape.
#
# LIMIT, stated rather than implied -- and stated in BOTH directions, because
# round 15 measured the earlier one-directional statement ("never an unrelated
# command being parked here") false in the direction it excluded:
#
#   NOT DETECTED. This resolves REFERENCES, so a call made through a value
#   instead of a name -- a variable holding the function, `getattr(module,
#   "update")`, a dispatch table looked up at run time -- is not resolvable from
#   source. And a route to the pin FILE that never passes through the seed below
#   is invisible too; the seed is the function that performs the write rather
#   than the public wrapper, and
#   `test_nothing_outside_the_pin_module_can_reach_the_pin_file` is what stops
#   that second direction from being merely asserted.
#
#   NOT CLAIMED. Over-detection is not the benign direction: the exemption set
#   is asserted to be EXACTLY the detected writers, so a command wrongly
#   detected has to be parked in INDEX_EXEMPT_STICKY, which drops a real command
#   from the index requirement (#627) -- the precise harm the exemption exists to
#   prevent. So a nested `def` is a scope of its own, folded in only when the
#   enclosing body invokes it, and a name a function BINDS itself (a parameter,
#   an assignment target) is not the module of the same spelling.
#
# Both directions are pinned by
# `test_the_pin_writer_resolution_states_its_limit_in_both_directions`, against
# a corpus carrying the shapes the real tree happens not to have.
#
# Two over-approximations remain, and they are listed as the ones found rather
# than as all there are: two nested `def`s of the SAME name in one function share
# an entry, so invoking either folds in both; and a call written in a DEFAULT
# ARGUMENT is attributed to the function whose signature carries it, though it
# runs once at definition time and never when that function is invoked. Both
# over-detect, which is the direction that costs a real command its index
# requirement -- so a further one found later is a finding, not a footnote, and
# round 17 found one (a nested CLASS method sharing the flat nested-name space),
# which is closed above rather than added to this list.
_PIN_WRITER = "bn.session_state._atomic_write"


def _module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("bn", *parts))


def _module_bindings(tree: ast.Module, module: str) -> dict[str, str]:
    """Local name -> the dotted object it names, for one module.

    Function-local imports are read too: `from bn.session_state import update as
    pin` inside a handler body binds `pin` just as a module-level import does.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                bindings[alias.asname or head] = alias.name if alias.asname else head
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                owner = module.split(".")[:-1]
                if node.level > 1:
                    owner = owner[:max(0, len(owner) - (node.level - 1))]
                base = ".".join([*owner, *([base] if base else [])])
            for alias in node.names:
                bindings[alias.asname or alias.name] = (
                    f"{base}.{alias.name}" if base else alias.name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] = f"{module}.{node.name}"
    return bindings


def _dotted_callee(expr: ast.expr) -> str | None:
    """`a.b.c` as a dotted string; None for anything that is not a name path."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        head = _dotted_callee(expr.value)
        return f"{head}.{expr.attr}" if head else None
    return None


def _canonical(dotted: str, module: str, bindings: dict[str, dict[str, str]]) -> str:
    """*dotted*, as written inside *module*, rewritten to its defining module.

    Each step replaces the longest prefix that names a module in the corpus plus
    a name that module binds. `cli.session_state.update` written in
    `bn.commands.admin` becomes `bn.cli.session_state.update`, then
    `bn.session_state.update`. A name the corpus does not bind is left alone, so
    an attribute chain on a local value never collides with a real function.
    """
    parts = [*module.split("."), *dotted.split(".")]
    for _ in range(8):
        for cut in range(len(parts) - 1, 0, -1):
            target = bindings.get(".".join(parts[:cut]), {}).get(parts[cut])
            if target:
                following = [*target.split("."), *parts[cut + 1:]]
                break
        else:
            break
        if following == parts:
            break
        parts = following
    return ".".join(parts)


def _corpus_functions(tree: ast.Module):
    """(qualname, node) for every function a registered handler can BE: module
    level, and one class level down."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{member.name}", member


def _bound_names(scope: ast.AST) -> frozenset[str]:
    """The names *scope* binds itself: parameters and assignment targets.

    A bound name is NOT the module-level object of the same spelling, so
    `def h(session_state): session_state.update(1)` reaches nothing this corpus
    can resolve -- reading it as the pin writer mis-attributed a command that
    cannot touch the pin. Import-bound names are deliberately excluded: a
    function-local `from bn.session_state import update as pin` is exactly the
    aliased write `_module_bindings` exists to resolve.
    """
    bound: set[str] = set()
    args = getattr(scope, "args", None)
    if args is not None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs,
                    args.vararg, args.kwarg):
            if arg is not None:
                bound.add(arg.arg)
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    imported = {alias.asname or alias.name.split(".")[0]
                for node in ast.walk(scope)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    return frozenset(bound - imported)


def _invoked_callees(fn: ast.AST, module: str,
                     bindings: dict[str, dict[str, str]]) -> set[str]:
    """The canonical callees *fn* can actually invoke.

    A nested `def`/`lambda` is a scope of its OWN, folded in only when some
    reached scope calls it by name (transitively, so a chain of nested helpers
    still counts). Attributing an UNCALLED nested writer to its enclosing
    function classified a handler that cannot write the pin, and over-detection
    is not the benign direction here -- see the LIMIT above. An immediately
    applied lambda IS invoked, so it is folded in; a lambda stored and called
    through the value is a call through a value, the stated limit. A nested
    CLASS body is skipped for the same reason: its methods are reached through
    an INSTANCE, which is a value, and registering them under their bare names
    in this flat space made a bare call to an unrelated module-level function of
    the same name fold a method's callees into the caller (round 17).
    """
    nested: dict[str, ast.AST] = {}
    raw: dict[str, set[str]] = {}

    def collect(scope: ast.AST, key: str) -> None:
        found: set[str] = set()
        pending = list(ast.iter_child_nodes(scope))
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested[node.name] = node
                collect(node, node.name)
                continue
            if isinstance(node, ast.ClassDef):
                continue
            if isinstance(node, ast.Call):
                dotted = _dotted_callee(node.func)
                if dotted:
                    found.add(dotted)
                if isinstance(node.func, ast.Lambda):
                    pending.extend(ast.iter_child_nodes(node.func))
            elif isinstance(node, ast.Lambda):
                continue
            pending.extend(ast.iter_child_nodes(node))
        raw[key] = found

    collect(fn, "")
    reached, frontier = {""}, [""]
    while frontier:
        for name in raw[frontier.pop()] & nested.keys():
            if name not in reached:
                reached.add(name)
                frontier.append(name)
    # A nested `def` also SHADOWS a module-level name it repeats, so a bare call
    # to it must not resolve to the module-level function of that name.
    shadowed = _bound_names(fn) | nested.keys()
    return {_canonical(dotted, module, bindings)
            for key in reached for dotted in raw[key]
            if dotted.split(".")[0] not in shadowed}


def _pin_writing_functions(sources: dict[str, str]) -> frozenset[str]:
    """Module-qualified names of every function in *sources* that reaches the
    pin writer -- directly, or through a chain of calls the corpus resolves.

    Takes the corpus as an argument so the resolution itself is testable against
    a corpus that contains the collisions the real tree does not currently have.
    """
    trees = {module: ast.parse(text) for module, text in sources.items()}
    bindings = {module: _module_bindings(tree, module)
                for module, tree in trees.items()}
    calls: dict[str, set[str]] = {}
    for module, tree in trees.items():
        for qualname, fn in _corpus_functions(tree):
            calls[f"{module}.{qualname}"] = _invoked_callees(fn, module, bindings)
    writers = {name for name, targets in calls.items() if _PIN_WRITER in targets}
    while True:
        reaching = {name for name, targets in calls.items()
                    if name not in writers and targets & writers}
        if not reaching:
            return frozenset(writers)
        writers |= reaching


def _sticky_state_writers() -> frozenset[str]:
    """Every registered command whose handler writes the sticky pin.

    Derived from the handler's own identity -- the function the registry holds,
    located by module and qualified name -- so the group cannot be used to park a
    command that does not touch the pin, and a new pin-writing command is a
    documentation decision that fails here until it is made.
    """
    root = SKILL.parents[2] / "src" / "bn"
    writers = _pin_writing_functions({
        _module_name(path, root): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
    })
    assert writers, (
        f"no function under src/bn reaches {_PIN_WRITER}; the sticky exemption's "
        "reason is unchecked, so the pin writer has been renamed"
    )
    return frozenset(
        command for command, handler in _registered_handlers().items()
        if f"{handler.__module__}.{handler.__qualname__}" in writers
    )


def test_the_sticky_pin_writer_is_matched_by_identity_not_by_name():
    """The exemption's reason is "THIS handler writes the shared pin", and the
    two cuts before this one answered a different question -- a substring over
    rendered source, then a bare function name shared across modules. Asserted
    against a corpus carrying the collisions the real tree happens not to have,
    because the real tree not having them today is why both cuts looked right.
    """
    writers = _pin_writing_functions({
        # The real module's shape: a public wrapper over the function that
        # performs the write, which is what `_PIN_WRITER` seeds on.
        "bn.session_state": ("def update(**fields):\n"
                             "    _atomic_write(fields)\n"
                             "def _atomic_write(state):\n"
                             "    pass\n"),
        "bn.commands.admin": (
            "from bn import session_state\n"
            "def pin(selector):\n"
            "    session_state.update(target=selector)\n"
            "def wrapper(selector):\n"
            "    pin(selector)\n"
            "def unrelated(namespace):\n"
            "    return namespace.session_state.update(namespace)\n"
        ),
        "bn.commands.decoy": (
            "def pin(selector):\n"
            "    return selector\n"
            "def aliased(selector):\n"
            "    from bn.session_state import update as _set\n"
            "    _set(target=selector)\n"
        ),
    })
    assert writers == {
        # the wrapper in the pin module itself, which is what makes reaching
        # `update` reach the write...
        "bn.session_state.update",
        # ...the direct write...
        "bn.commands.admin.pin",
        # ...the wrapper that reaches it...
        "bn.commands.admin.wrapper",
        # ...and the same function under an import alias in another module.
        "bn.commands.decoy.aliased",
        # NOT bn.commands.decoy.pin (same bare name, writes nothing) and NOT
        # bn.commands.admin.unrelated (an attribute chain on a parameter that
        # merely ends in the writer's two names).
    }, sorted(writers)


# The pin module as the real one is shaped: the public entry point and the
# function that performs the write. Shared by the cells below so a corpus cannot
# quietly disagree with the seed about which of the two is the writer.
_PIN_MODULE = ("def update(**fields):\n"
               "    _atomic_write(fields)\n"
               "def _atomic_write(state):\n"
               "    pass\n")


@pytest.mark.parametrize("detected,body", [
    # --- DETECTED ---
    pytest.param(True, "def h(a):\n    session_state.update(target=a)\n",
                 id="a-plain-dotted-write"),
    # The direction round 15 measured the old one-line residual wrong about: a
    # call straight to the function that writes the file, made through a plain
    # NAME, which the old seed (`update`, the wrapper) could not see at all.
    pytest.param(True, "def h(a):\n    session_state._atomic_write({'target': a})\n",
                 id="a-write-that-skips-the-wrapper"),
    pytest.param(True, ("def h(a):\n"
                        "    def inner():\n"
                        "        session_state.update(target=a)\n"
                        "    inner()\n"),
                 id="a-nested-scope-the-body-invokes"),
    pytest.param(True, ("def h(a):\n"
                        "    def outer():\n"
                        "        def deeper():\n"
                        "            session_state.update(target=a)\n"
                        "        deeper()\n"
                        "    outer()\n"),
                 id="a-chain-of-invoked-nested-scopes"),
    pytest.param(True, "def h(a):\n    (lambda: session_state.update(target=a))()\n",
                 id="an-immediately-applied-lambda"),
    # --- NOT DETECTED ---
    # The other direction it was wrong about: two shapes that CANNOT write the
    # pin and were classified as writers, which forces an unrelated command into
    # INDEX_EXEMPT_STICKY and drops it from the index requirement.
    pytest.param(False, ("def h(a):\n"
                         "    def never():\n"
                         "        session_state.update(target=a)\n"
                         "    return 0\n"),
                 id="a-nested-scope-nothing-invokes"),
    pytest.param(False, "def h(session_state):\n    return session_state.update(1)\n",
                 id="a-parameter-shadowing-the-module"),
    # Round 17: a nested CLASS method used to share the flat nested-name space,
    # so a bare call to an unrelated module-level function of the same name
    # folded the method's callees into the caller. `h` here calls the
    # module-level `write`, which writes nothing.
    pytest.param(False, ("def write(a):\n"
                         "    return a\n"
                         "def h(a):\n"
                         "    class _Holder:\n"
                         "        def write(self):\n"
                         "            session_state.update(target=a)\n"
                         "    return write(a)\n"),
                 id="a-nested-class-method-nothing-instantiates"),
    # ...and the limit that remains stated: resolution is by REFERENCE, so a
    # call made through a value is not visible in the source.
    pytest.param(False, ("def h(a):\n"
                         "    writer = session_state.update\n"
                         "    writer(target=a)\n"),
                 id="a-call-through-a-value"),
    pytest.param(False, "def h(a):\n    getattr(session_state, 'update')(target=a)\n",
                 id="a-call-through-getattr"),
    pytest.param(False, ("def h(a):\n"
                         "    class _Holder:\n"
                         "        def write(self):\n"
                         "            session_state.update(target=a)\n"
                         "    return _Holder().write()\n"),
                 id="a-method-call-through-an-instance"),
])
def test_the_pin_writer_resolution_states_its_limit_in_both_directions(
        detected: bool, body: str):
    """The residual limit above, executed in both directions.

    Round 15 measured the previous one-line residual ("a call made through a
    value instead of a name ... never an unrelated command being parked here")
    false BOTH ways: it missed a write made through a plain dotted name, and it
    classified two shapes that cannot write the pin at all. A residual wrong in
    both directions is worse than an unstated one, because a reader trusts it to
    bound the failure mode -- so each side of the restated limit is a parameter
    here, and a resolver that drifts either way reds the parameter that names
    the drift.
    """
    writers = _pin_writing_functions({
        "bn.session_state": _PIN_MODULE,
        "bn.commands.admin": f"from bn import session_state\n{body}",
    })
    assert ("bn.commands.admin.h" in writers) is detected, sorted(writers)


def test_nothing_outside_the_pin_module_can_reach_the_pin_file():
    """The half of the restated limit a corpus cannot check.

    The resolver seeds on ONE function, so a second route to the pin file --
    another module resolving `session_state_path()` and writing it itself --
    would be invisible to every cell above while a new sticky-pin command went
    unexempted. That route does not exist, and this is what makes "not detected"
    a fact about the tree rather than a hope: the path helper is referenced only
    by the module that defines it and the module that owns the pin.
    """
    root = SKILL.parents[2] / "src" / "bn"
    reached = sorted(
        _module_name(path, root) for path in sorted(root.rglob("*.py"))
        if "session_state_path" in path.read_text(encoding="utf-8")
    )
    assert reached == ["bn.paths", "bn.session_state"], (
        "the sticky pin's path helper is referenced outside the module that "
        f"defines it and the module that owns the pin: {reached}. Either route "
        f"the write through {_PIN_WRITER} or widen the seed -- as it stands, a "
        "pin-writing command there is invisible to the sticky exemption check"
    )


def test_every_index_exemption_states_a_reason_that_is_true(command_paths):
    """An exemption is a claim, and an unasserted claim is where a real gap
    hides: `evidence virtual-call` sat in the "catalogued in a reference" group
    while appearing in no reference at all. So each group's stated reason is
    checked here, and the groups together must be exactly the exemption set."""
    groups = INDEX_EXEMPT_STICKY | frozenset(INDEX_EXEMPT_ALIASES)
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
        f"{command} -> {via}" for command, via in INDEX_EXEMPT_ALIASES.items()
        if via not in advertised
    )
    assert not missing_way_in, (
        "these are exempt because they are a second path to an advertised "
        f"command, but that command is not advertised either: {missing_way_in}"
    )
    unresolved = sorted(set(INDEX_EXEMPT_ALIASES.values()) - command_paths)
    assert not unresolved, (
        f"these alias targets are not registered commands: {unresolved}"
    )
    handlers = _registered_handlers()
    not_an_alias = sorted(
        f"{command} -> {via}" for command, via in INDEX_EXEMPT_ALIASES.items()
        if handlers.get(command) is not handlers.get(via)
    )
    assert not not_an_alias, (
        "these are exempt BECAUSE they are a second path to the same handler, "
        "and the registry says they resolve to different functions -- the "
        f"exemption's reason is false: {not_an_alias}"
    )
    writers = _sticky_state_writers()
    assert INDEX_EXEMPT_STICKY == writers, (
        "the sticky-pin exemption must be exactly the registered commands whose "
        "handler writes the shared per-project pin: a command that does not "
        "write it cannot be exempt for being harmful to advertise, and one that "
        f"does must be: {sorted(INDEX_EXEMPT_STICKY ^ writers)}"
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
    r"`(?P<cmd>[a-z][a-z -]*(?:/[a-z-]+)*)(?: \[(?P<optional>[a-z]+)\])?`(?: \([^)]*\))?"
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
EXPECTED_INDEX_GROUPS = frozenset({"Read", "Mutate", "Discover", "Session", "Tooling",
                                   "Escape hatch"})


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


def test_reading_reference_binds_the_absence_claim_to_the_cap_flag_the_cli_prints():
    """The reference taught absence from a CAPPED page: "`items: []` + `total: 0`
    means clean -- nothing found", qualified only for the taint shapes. An
    `xrefs` page on an import can be empty AND `truncated: true` with a
    `scan_note` (the budgeted LLIL caller scan), so the flat rule teaches the
    exact error the CLI's own text output refuses.

    Pinned from both ends: the sentence must bind absence to the cap flags, and
    the note it QUOTES must be the line the renderer really prints -- delete
    either half and the doc is quoting text the CLI does not emit."""
    from bn.formatters import _render_xrefs_text

    quoted = "note: the caller scan was TRUNCATED"
    text = READING.read_text(encoding="utf-8")
    assert quoted in text, (
        "the reading reference no longer quotes the truncation note, so nothing "
        "tells a reader an empty capped page is unknown rather than absent")
    assert "truncated: true" in text and "scan_note" in text, (
        "the absence rule must name the cap flags it is conditional on")

    rendered = _render_xrefs_text({
        "address": "0x401000", "code_refs": [], "data_refs": [],
        "code_ref_count": 0, "data_ref_count": 0,
        "truncated": True, "scan_note": "scan stopped at its budget",
    })
    assert quoted in rendered, (
        f"the reference quotes {quoted!r} but the renderer prints: {rendered}")


# ---------------------------------------------------------------------------
# The mirror of this module's founding defect. Its header records that an
# OMITTED flag makes agents conclude a shipped feature does not exist -- three
# re-filed it in one dogfood run. The inverse costs the same and was guarded by
# nothing: a reference that ADVERTISES a flag the parser has no idea about sends
# an agent to write `--dry-run` and read the argparse refusal as a broken CLI.
# A round-21 lens added a nonexistent `--dry-run` to the mutation reference and
# a nonexistent `--max` to the reading reference, and the suite stayed green.
#
# So every long flag these documents NAME is checked against the parser -- in
# prose and in fenced examples alike, because an agent copies both. A document
# may also state that a flag does NOT exist (the mutation reference says so of
# `--comment`, whose natural spelling fails with an argparse error), which is
# the opposite claim and is asserted in the opposite direction.
# The population is a GLOB, not a list. Round 22's lens put an invented long
# flag in `CLAUDE.md` -- outside the four references that were parametrized --
# and it stayed green, which is the same enumerated-population defect this PR
# spent its rounds deleting everywhere else. A document joins by EXISTING.
AGENT_FACING_DOCS = (
    REFERENCE.parent.parent.parent / "CLAUDE.md",
    REFERENCE.parent.parent.parent / "README.md",
    *sorted((REFERENCE.parent.parent).glob("**/*.md")),
)

# Any long flag, not just a lowercase one: `--Force`, `--dry_run` and `--O2` are
# all things a doc can invent and an agent can copy, and the first cut's
# `[a-z][a-z0-9-]+` saw none of them.
_DOC_FLAG = re.compile(r"--[A-Za-z0-9][\w-]*")

# A doc may also claim a flag does NOT exist -- the mutation reference says so
# of `--comment`, whose natural spelling fails with an argparse error. That is
# the opposite claim and is asserted in the opposite direction. The forms below
# are ASSERTIONS OF ABSENCE; "do not pass `--preview` here" is advice about a
# flag that exists and deliberately does not match, while round 22's exploit
# ("There is no `--preview` option") does.
_ABSENCE_CLAIM = re.compile(
    r"`(--[\w-]+)`[^.`]{0,30}do(?:es)? not exist"
    r"|there is no `(--[\w-]+)`"
    r"|no such `?(--[\w-]+)`?"
    r"|no `(--[\w-]+)` (?:flag|option|switch|argument)"
    r"|`(--[\w-]+)` is not a (?:real )?(?:flag|option)",
    re.I)


# One rule, one implementation: a denial binds to the NAME it denies.
#
# Rounds 20-22 each closed one instance of the same class -- a guard scoped to a
# population someone had listed -- and round 23 found the class again one
# granularity in: the negation was still bound to a text WINDOW rather than to
# the name it negates. Subtracted DOCUMENT-wide, one denial sentence excused an
# invented flag inside a runnable fenced example elsewhere in the same file; in
# `tests/test_agent_docs.py` the same defect read per SENTENCE, so one sentence
# carrying both a retirement and a positive citation was skipped whole.
#
# So a denial is bound by ADJACENCY: the names it covers are the ones its own
# pattern captures, plus the run of name tokens immediately before it joined by
# nothing but whitespace, list punctuation, markdown emphasis and a coordinator.
# A verb or a subordinator ("because") ends the run, so a name on the far side
# of one is a POSITIVE claim about that name -- and a phrasing the recogniser
# does not know denies nothing at all, which fails CLOSED for every guard that
# asks what a document may be excused from naming.
_SENTENCE = re.compile(r"(?<=[.;:])\s+|\n")
_COORDINATED = re.compile(r"^[\s,;*_`]*(?:and|or|nor)?[\s,;*_`]*$", re.I)


def bound_absence_claims(sentence: str, token: re.Pattern[str],
                         absence: re.Pattern[str]) -> set[str]:
    """The names `sentence` denies, bound to the phrase that denies them.

    Shared with `tests/test_agent_docs.py`, which binds retirement claims about
    test modules through this same function: round 23 found the identical defect
    in both because the earlier repair had been applied to one and not to its
    sibling.
    """
    names = [(match.start(), match.end(), match[match.lastindex or 0])
             for match in token.finditer(sentence)]
    denied: set[str] = set()
    for claim in absence.finditer(sentence):
        denied.update(group for group in claim.groups() if group)
        edge = claim.start()
        for start, end, name in reversed(names):
            if end > edge:
                continue
            if not _COORDINATED.fullmatch(sentence[end:edge]):
                break
            denied.add(name)
            edge = start
    return denied


def _prose_and_fenced(text: str) -> tuple[str, str]:
    """A document split into its prose and its fenced examples.

    A fenced example is a runnable instruction, never a denial, so a flag named
    inside one is a positive claim whatever the prose around it says.
    """
    prose: list[str] = []
    fenced: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        (fenced if inside else prose).append(line)
    return "\n".join(prose), "\n".join(fenced)


def _parser_long_flags() -> set[str]:
    """Every long option the CLI accepts, at any depth of the subcommand tree."""
    import argparse

    import bn.cli

    flags: set[str] = set()

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            flags.update(opt for opt in action.option_strings if opt.startswith("--"))
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    walk(sub)

    walk(bn.cli.build_parser())
    return flags


@pytest.mark.parametrize("doc", AGENT_FACING_DOCS, ids=lambda p: p.name)
def test_every_flag_the_reference_names_exists(doc: Path):
    """A flag an agent reads in any agent-facing doc must be one the CLI accepts.

    The mirror of this module's founding defect. Its header records that an
    OMITTED flag makes agents conclude a shipped feature does not exist -- three
    re-filed it in one dogfood run. The inverse costs the same and was guarded
    by nothing: a document that ADVERTISES a flag the parser has never heard of
    sends an agent to write `--dry-run` and read the argparse refusal as a
    broken CLI.

    The absence exemption is bound to the OCCURRENCE, not to the document. Read
    document-wide it let a round-23 lens ship a runnable `bn go rename
    --dry-run` example and have it excused by an unrelated denial sentence
    further down the same file -- certifying the exact false instruction this
    cell exists to catch.
    """
    known = _parser_long_flags()
    assert known, "the parser exposes no long flags, so this cell proves nothing"
    prose, fenced = _prose_and_fenced(doc.read_text(encoding="utf-8"))
    invented = {match[0] for match in _DOC_FLAG.finditer(fenced)} - known
    for sentence in _SENTENCE.split(prose):
        named = {match[0] for match in _DOC_FLAG.finditer(sentence)}
        invented |= (named - known
                     - bound_absence_claims(sentence, _DOC_FLAG, _ABSENCE_CLAIM))
    assert not invented, (
        f"{doc.name} names flags the parser does not accept, so an agent copying "
        f"them gets an argparse refusal and reads it as a broken CLI: "
        f"{sorted(invented)}"
    )


# The opposite direction is the one claim in this accounting that is NOT
# closable, and saying so is the point of this block rather than a sixth
# phrasing in `_ABSENCE_CLAIM`.
#
# Recognising that a document DENIES a shipped flag means recognising negation
# in arbitrary English. Round 23 walked five unlisted forms straight past the
# recogniser ("doesn't exist", "There's no X here", "is unsupported", "was
# removed", "This CLI accepts no X"), and widening the alternation only moves
# the boundary rather than closing it: `skills/bn-kernel/SKILL.md` already says
# "It takes no `--all`" -- a true statement about ONE command, in the same shape
# as a false global denial -- so a recogniser complete enough to catch "accepts
# no `--preview`" also reds that line, and the two differ only in which subject
# the sentence is about. An exhaustive natural-language absence recogniser is
# not attainable here.
#
# What IS closed, and why the residual is bounded:
#   * an INVENTED flag can never be laundered: the cell above fails CLOSED, so a
#     phrasing the recogniser does not parse denies nothing and the flag stays a
#     positive claim. No document can advertise a flag the CLI lacks.
#   * a command whose reference carries an EXHAUSTIVE flag list is compared
#     against the parser as a SET (`test_go_rename_reference_lists_exactly_the_
#     flags_it_takes`), which catches a denial by omission completely.
#   * what survives is a doc denying a SHIPPED flag in a form this pattern does
#     not parse. The cost is an agent avoiding a flag that works, not writing a
#     command that fails; no document in the tree does it; and the cell below
#     catches every form the tree's own denials actually use.
@pytest.mark.parametrize("doc", AGENT_FACING_DOCS, ids=lambda p: p.name)
def test_every_flag_the_reference_denies_really_does_not_exist(doc: Path):
    """...and the other direction, as far as it is attainable: a flag a document
    denies in a form this module can PARSE must stay unreachable, or the doc is
    steering an agent away from something that now works.

    Deliberately NOT a universal claim -- the block above records which shapes
    survive and why completeness is unattainable.
    """
    denied: set[str] = set()
    for sentence in _SENTENCE.split(doc.read_text(encoding="utf-8")):
        denied |= bound_absence_claims(sentence, _DOC_FLAG, _ABSENCE_CLAIM)
    shipped = sorted(denied & _parser_long_flags())
    assert not shipped, (
        f"{doc.name} says these flags do not exist, but the parser accepts them: "
        f"{shipped}"
    )


def test_an_absence_claim_binds_to_the_name_it_denies():
    """The binding rule itself, which both flag cells above and the
    module-citation guard in `tests/test_agent_docs.py` rest on.

    A run of names joined by a coordinator is denied together; a name on the far
    side of a clause is a positive claim, which is the round-23 bypass that put
    an invented module and a retirement in one sentence; and an unrecognised
    phrasing denies nothing, so the name it names stays asserted.
    """
    assert bound_absence_claims("`--alpha` and `--beta` do not exist.",
                                _DOC_FLAG, _ABSENCE_CLAIM) == {"--alpha", "--beta"}
    assert bound_absence_claims(
        "The rule is enforced by `--alpha` because `--beta` does not exist.",
        _DOC_FLAG, _ABSENCE_CLAIM) == {"--beta"}
    assert bound_absence_claims("`--alpha` was retired years ago.",
                                _DOC_FLAG, _ABSENCE_CLAIM) == set()


def test_go_rename_reference_states_the_scope_the_bridge_enforces():
    """`go rename` is the one bulk mutation, and its safety claim is its SCOPE:
    auto-named functions only, so a manual name is never overwritten and the op
    is idempotent. A round-21 lens inverted that sentence -- "renames every
    function ... NOT idempotent" -- with every guard green, which is a doc an
    agent could act on to destroy its own naming work.

    Executed against the bridge predicate the claim is about, not quoted.
    """
    from bn_agent_bridge.bridge import _is_go_rename_auto_name

    text = MUTATING.read_text(encoding="utf-8")
    prose = " ".join(text.split())
    assert "renames **auto-named `sub_*`/`nullsub_*` functions only**" in prose, (
        "the reference no longer states `go rename`'s scope, which is its only "
        "safety property"
    )
    assert "idempotent and safe to re-run" in prose, (
        "the reference no longer states that `go rename` is idempotent"
    )
    # The predicate really is auto-names-only, in both directions.
    assert _is_go_rename_auto_name("sub_401000", 0x401000)
    assert _is_go_rename_auto_name("nullsub_12", 0x401000)
    assert not _is_go_rename_auto_name("player_update", 0x401000), (
        "a manual name must never be replaceable, which is what the reference "
        "promises and what makes a re-run safe"
    )
    # ...and "auto" is judged against THIS address, not any sub_ name.
    assert not _is_go_rename_auto_name("sub_401000", 0x402000)


def test_go_rename_reference_lists_exactly_the_flags_it_takes():
    """The reference says `go rename` takes the standard mutation flags "and
    nothing else", which is an EXHAUSTIVE claim -- so it is compared against the
    parser as a set. Presence-only checking let a lens append a nonexistent
    `--dry-run` to that very list."""
    import argparse

    import bn.cli

    parser = bn.cli.build_parser()
    args = parser.parse_args(["go", "rename"])
    del args
    prose = " ".join(MUTATING.read_text(encoding="utf-8").split())
    listed = re.search(
        r"It takes the standard mutation flags \(([^)]*)\) and nothing else\.", prose)
    assert listed, (
        "the reference no longer states `go rename`'s flag list, so it can grow "
        "a flag the CLI does not have"
    )
    # The same token the doc-wide sweep uses, so a `--Force` invented inside
    # this EXHAUSTIVE list is not invisible to the cell that owns the list.
    claimed = {match[0] for match in _DOC_FLAG.finditer(listed.group(1))}

    def subparser(path: list[str]) -> argparse.ArgumentParser:
        current = parser
        for name in path:
            action = next(a for a in current._actions
                          if isinstance(a, argparse._SubParsersAction))
            current = action.choices[name]
        return current

    real = {opt for action in subparser(["go", "rename"])._actions
            for opt in action.option_strings if opt.startswith("--")}
    # The flags every command carries are not this command's own surface.
    shared = {opt for action in parser._actions
              for opt in action.option_strings if opt.startswith("--")}
    # ...nor is an ALIAS of a flag already listed. The same document declares
    # them (`--verbose` (alias `--diffs`)), so the mapping is read off the doc
    # rather than hardcoded here: dropping an alias declaration makes this claim
    # stop adding up, which is the right direction to fail in.
    # The declaration reads `<flags>` (alias `--x`), and <flags> is sometimes a
    # PHRASE (`--format json --summary` (alias `--quiet`)), so the alias is tied
    # to every flag in the run just before it.
    aliases: dict[str, set[str]] = {}
    for match in re.finditer(r"(?P<canonical>(?:`[^`]+`[ ]?)+)"
                             r"\(alias `(?P<alias>--[\w-]+)`\)", prose):
        aliases.setdefault(match["alias"], set()).update(
            re.findall(r"--[\w-]+", match["canonical"]))
    # An alias declaration is itself a claim about the parser, and until round 22
    # it was taken on the document's word: declaring a FALSE alias dropped a real
    # flag out of an EXHAUSTIVE list with the suite green. So each declared pair
    # is checked to be a real alias -- both spellings known, and both carried by
    # the SAME argparse action, which is what "alias" means.
    by_option = {opt: action
                 for action in subparser(["go", "rename"])._actions
                 for opt in action.option_strings}
    # The canonical side may be a PHRASE naming several flags
    # (`--format json --summary` (alias `--quiet`)), so the alias must share its
    # action with at LEAST one of them -- that is what makes it an alias -- and
    # only those genuine partners may excuse it from the exhaustive list.
    real_aliases: dict[str, set[str]] = {}
    for alias, canonicals in aliases.items():
        if alias not in by_option:
            continue                  # not one of this command's options
        partners = {canonical for canonical in canonicals
                    if by_option.get(canonical) is by_option[alias]}
        assert partners, (
            f"the reference declares `{alias}` an alias of "
            f"{sorted(canonicals)}, but the parser carries it on none of their "
            "options, so the exhaustive flag list below is excusing a flag it "
            "should name"
        )
        real_aliases[alias] = partners
    own = {opt for opt in real - shared
           if not (real_aliases.get(opt, set()) & claimed)}
    assert claimed == own, (
        f"the reference lists {sorted(claimed)} for `go rename` and says that is "
        f"all of them; the parser defines {sorted(own)} beyond the shared flags "
        f"and the aliases the document declares ({aliases})"
    )
