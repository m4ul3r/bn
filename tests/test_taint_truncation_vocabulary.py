"""Every truncation cause the engine can record renders as itself (P1-08).

The verdict line and the bridge's all-clear reason read one table,
``taint_truncation.TRUNCATION_CAUSES``. A cause missing from it used to fall
through to ``truncated @depth N`` and send the reader to ``--max-depth``; the
completeness test below reads the engine's own source, so the next cause added
there fails here until it is given a meaning and a remedy.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from bn.formatters import _render_taint_text, _taint_truncation_note
from bn.taint_truncation import TRUNCATION_CAUSES

BRIDGE = Path(__file__).resolve().parents[1] / "src" / "bn_agent_bridge"


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)


class _Unresolved(Exception):
    """A producer argument this crawler cannot reduce to string literals."""


def _bound_names(target: ast.expr) -> set[str]:
    """The names an assignment target BINDS: a name, or the names of a
    (nested) tuple/list -- never one used inside a subscript or attribute
    target (`state[key] = v` binds nothing called `key`)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_bound_names(e) for e in target.elts))
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return set()


class _Crawler:
    """Reduce the argument of every ``*truncation_causes.add(...)`` call in one
    module to the set of string literals it can hold.

    Followed: a literal; a conditional expression; a name bound by assignment
    (also tuple-unpacked from a literal tuple) or by a ``for`` loop over a
    literal collection or a dict literal's ``keys()``/``values()``/``items()``;
    ``D.get(...)`` and ``D[...]`` on a dict literal (every value). A name is
    looked up in its own function, then in each enclosing scope out to the
    module. Anything else -- a parameter, a computed value -- raises
    :class:`_Unresolved`, which the test reports with the call site: a
    producer the crawler silently skipped is exactly how a cause escapes."""

    def __init__(self, tree: ast.AST) -> None:
        self.parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node

    def scopes(self, node: ast.AST) -> list[ast.AST]:
        out = []
        while node in self.parent:
            node = self.parent[node]
            if isinstance(node, _SCOPES):
                out.append(node)
        return out

    def owner(self, node: ast.AST) -> ast.AST:
        return self.scopes(node)[0]

    def bindings(self, name: str, at: ast.AST) -> tuple[ast.AST, list[ast.AST]]:
        """The nearest scope binding *name* and its binding nodes."""
        for scope in self.scopes(at):
            args = getattr(scope, "args", None)
            params = ({a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
                      if args is not None else set())
            if name in params:
                raise _Unresolved(f"{name!r} is a parameter of {getattr(scope, 'name', '<lambda>')}")
            found = [n for n in ast.walk(scope)
                     if isinstance(n, (ast.Assign, ast.For, ast.comprehension))
                     and self.owner(n) is scope
                     and name in _bound_names(n.targets[0] if isinstance(n, ast.Assign)
                                              else n.target)]
            if found:
                return scope, found
        raise _Unresolved(f"no binding for {name!r}")

    def dict_literal(self, expr: ast.expr) -> ast.Dict:
        if isinstance(expr, ast.Dict):
            return expr
        if isinstance(expr, ast.Name):
            _scope, found = self.bindings(expr.id, expr)
            tables = [n.value for n in found if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Dict)]
            if len(tables) == len(found):
                return tables[0] if len(tables) == 1 else ast.Dict(
                    keys=[k for t in tables for k in t.keys],
                    values=[v for t in tables for v in t.values])
        raise _Unresolved(f"{ast.unparse(expr)} is not a dict literal")

    def strings(self, expr: ast.expr) -> set[str]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.IfExp):
            return self.strings(expr.body) | self.strings(expr.orelse)
        if isinstance(expr, ast.Name):
            out: set[str] = set()
            _scope, found = self.bindings(expr.id, expr)
            for node in found:
                out |= self.bound(node, expr.id)
            return out
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "get"):
            return self.values(self.dict_literal(expr.func.value))
        if isinstance(expr, ast.Subscript):
            return self.values(self.dict_literal(expr.value))
        raise _Unresolved(ast.unparse(expr))

    def values(self, table: ast.Dict) -> set[str]:
        return set().union(*(self.strings(v) for v in table.values))

    def keys(self, table: ast.Dict) -> set[str]:
        return set().union(*(self.strings(k) for k in table.keys if k is not None))

    def bound(self, node: ast.AST, name: str) -> set[str]:
        """What binding *node* puts in *name*."""
        if isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
            if isinstance(target, ast.Name):
                return self.strings(value)
            if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                i = [isinstance(t, ast.Name) and t.id == name for t in target.elts].index(True)
                return self.strings(value.elts[i])
            raise _Unresolved(ast.unparse(node))
        target, it = node.target, node.iter
        if isinstance(target, ast.Name):
            return self.elements(it)
        if (isinstance(target, ast.Tuple) and isinstance(it, ast.Call)
                and isinstance(it.func, ast.Attribute) and it.func.attr == "items"):
            i = [isinstance(t, ast.Name) and t.id == name for t in target.elts].index(True)
            table = self.dict_literal(it.func.value)
            return self.keys(table) if i == 0 else self.values(table)
        if isinstance(target, ast.Tuple) and isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            i = [isinstance(t, ast.Name) and t.id == name for t in target.elts].index(True)
            return set().union(*(self.strings(e.elts[i]) for e in it.elts))
        raise _Unresolved(f"for {ast.unparse(target)} in {ast.unparse(it)}")

    def elements(self, it: ast.expr) -> set[str]:
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            return set().union(*(self.strings(e) for e in it.elts))
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
            if it.func.attr == "keys":
                return self.keys(self.dict_literal(it.func.value))
            if it.func.attr == "values":
                return self.values(self.dict_literal(it.func.value))
        if isinstance(it, (ast.Name, ast.Dict)):
            return self.keys(self.dict_literal(it))
        raise _Unresolved(f"iteration over {ast.unparse(it)}")


def _cause_literals(path: Path) -> set[str]:
    """Every string a ``*truncation_causes.add(...)`` call in *path* can add.

    A producer shape :class:`_Crawler` cannot follow fails with its call site
    named rather than being skipped."""
    tree = ast.parse(path.read_text())
    crawler = _Crawler(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr.endswith("truncation_causes")):
            continue
        (arg,) = node.args
        try:
            found |= crawler.strings(arg)
        except _Unresolved as exc:
            pytest.fail(f"{path.name}:{node.lineno}: cannot tell which truncation cause "
                        f"`{ast.unparse(node)}` records ({exc}); pass a literal or a "
                        "name bound to literals, or teach this crawler the shape")
    return found


def test_every_engine_truncation_cause_is_in_the_shared_vocabulary():
    causes: set[str] = set()
    for path in sorted(BRIDGE.glob("*.py")):
        if not path.is_symlink():
            causes |= _cause_literals(path)
    # The four bounds the walk itself enforces must be among them, or the
    # extraction above silently stopped seeing producers.
    assert {"max_depth", "fixpoint_exhausted", "recursion", "caller_cap"} <= causes
    assert "global_object_extent" in causes
    missing = sorted(causes - set(TRUNCATION_CAUSES))
    assert not missing, f"engine truncation causes with no verdict entry: {missing}"

_PRODUCERS = textwrap.dedent('''
    TABLE = {"x": "from_module_table"}
    class Engine:
        def run(self, reach):
            names = {"a": "from_get"}
            self._truncation_causes.add("from_literal")
            aggregate = names.get(reach)
            self._truncation_causes.add(aggregate)
            self._truncation_causes.add(TABLE[reach])
            bounds = {"from_items": "prose"}
            for cause, _text in bounds.items():
                self._bw_truncation_causes.add(cause)
            for c in ("from_loop",):
                self._truncation_causes.add(c)
            first, _second = "from_unpack", "unused"
            self._truncation_causes.add(first)
            def inner():
                self._truncation_causes.add(aggregate)
            self._truncation_causes.add("brand_new_budget" if reach else "from_literal")
''')


def test_the_crawler_follows_every_producer_shape_and_surfaces_a_new_cause(tmp_path):
    module = tmp_path / "engine.py"
    module.write_text(_PRODUCERS)
    causes = _cause_literals(module)
    assert causes == {"from_literal", "from_get", "from_module_table", "from_items",
                      "from_loop", "from_unpack", "brand_new_budget"}
    # A cause the table does not know is exactly what the main test reports.
    assert "brand_new_budget" in causes - set(TRUNCATION_CAUSES)


def test_the_crawler_refuses_a_cause_it_cannot_follow_naming_the_call_site(tmp_path):
    module = tmp_path / "engine.py"
    module.write_text("class E:\n    def run(self, cause):\n"
                      "        self._truncation_causes.add(cause)\n")
    with pytest.raises(pytest.fail.Exception, match=r"engine\.py:3: .*parameter"):
        _cause_literals(module)



@pytest.mark.parametrize("cause", sorted(set(TRUNCATION_CAUSES) - {"max_depth"}))
def test_each_known_cause_renders_as_itself_not_a_depth_cutoff(cause):
    note = _taint_truncation_note(
        {"truncated": True, "max_depth": 6, "truncation_cause": [cause]})
    assert "@depth" not in note
    assert TRUNCATION_CAUSES[cause].what in note
    assert TRUNCATION_CAUSES[cause].remedy in note
    # No cause but a depth cut names the depth knob as its fix.
    assert "raise --max-depth" not in note


def test_unknown_cause_renders_verbatim_never_as_depth():
    note = _taint_truncation_note(
        {"truncated": True, "max_depth": 6, "truncation_cause": ["future_budget"]})
    assert "future_budget" in note
    assert "@depth" not in note and "--max-depth" not in note


def test_each_cause_of_a_mixed_run_is_named_once():
    note = _taint_truncation_note({
        "truncated": True, "max_depth": 8,
        "truncation_cause": ["global_object_extent", "max_depth", "pointer_scan_budget"]})
    assert note.count("@depth 8") == 1
    for cause in ("global_object_extent", "max_depth", "pointer_scan_budget"):
        assert TRUNCATION_CAUSES[cause].what in note


def test_forward_verdict_for_a_continuation_budget_names_that_budget():
    # The shape every lane met: a run whose only cause is a continuation
    # budget, with a reached depth below the configured limit.
    text = _render_taint_text({
        "direction": "forward", "function": {"name": "dispatch", "address": "0x401000"},
        "sources": ["param:1"], "reached_sinks": [], "leaves": [],
        "stats": {"functions_visited": 4, "max_depth": 6, "truncated": True,
                  "truncation_cause": ["global_object_extent"]},
        "run_params": {"max_depth": 8},
    })
    verdict = next(line for line in text.splitlines() if line.startswith("verdict:"))
    assert "@depth" not in verdict
    assert TRUNCATION_CAUSES["global_object_extent"].what in verdict


def test_bridge_all_clear_reason_reads_the_same_vocabulary():
    from bn_agent_bridge import taint_result as tr

    for cause, entry in TRUNCATION_CAUSES.items():
        hint = tr._truncation_hint([cause])
        assert entry.what in hint and entry.remedy in hint
    assert "future_budget" in tr._truncation_hint(["future_budget"])
    assert "--max-depth" not in tr._truncation_hint(["future_budget"])
