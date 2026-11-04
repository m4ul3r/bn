"""Every truncation cause the engine can record renders as itself (P1-08).

The verdict line and the bridge's all-clear reason read one table,
``taint_truncation.TRUNCATION_CAUSES``. A cause missing from it used to fall
through to ``truncated @depth N`` and send the reader to ``--max-depth``; the
completeness test below reads the engine's own source, so the next cause added
there fails here until it is given a meaning and a remedy.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bn.formatters import _render_taint_text, _taint_truncation_note
from bn.taint_truncation import TRUNCATION_CAUSES

BRIDGE = Path(__file__).resolve().parents[1] / "src" / "bn_agent_bridge"


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _assigned(scope: ast.AST, name: str) -> list[ast.expr]:
    return [n.value for n in ast.walk(scope)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def _cause_literals(path: Path) -> set[str]:
    """Every string a ``*truncation_causes.add(...)`` call in *path* can add.

    A literal argument is taken as is; a name is followed to the mapping it is
    looked up in (``aggregate = names.get(cause)``) and every value of that
    mapping counts. A producer shape this cannot follow fails loudly rather
    than being skipped -- a skipped producer is exactly how a cause escapes."""
    tree = ast.parse(path.read_text())
    parent = _enclosing_functions(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr.endswith("truncation_causes")):
            continue
        (arg,) = node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            found.add(arg.value)
            continue
        scope = node
        while not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parent[scope]
        assert isinstance(arg, ast.Name), (path.name, node.lineno, ast.unparse(arg))
        (value,) = _assigned(scope, arg.id)
        assert (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get" and isinstance(value.func.value, ast.Name)), \
            (path.name, node.lineno, ast.unparse(value))
        (table,) = _assigned(scope, value.func.value.id)
        assert isinstance(table, ast.Dict), (path.name, node.lineno)
        for v in table.values:
            assert isinstance(v, ast.Constant) and isinstance(v.value, str), ast.unparse(v)
            found.add(v.value)
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
