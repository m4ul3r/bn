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

from bn.formatters import FAILED_MUTATION_STATUSES

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
    assert AGENTS_MD.read_text(encoding="utf-8").strip() == "CLAUDE.md", (
        "AGENTS.md must be a symlink to CLAUDE.md (or its unexpanded link text on a "
        "checkout without symlink support), never a second copy -- a copy drifts"
    )


@pytest.mark.parametrize("doc", AGENT_FACING_DOCS, ids=lambda p: p.name)
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


def test_exit_code_3_lists_every_failed_mutation_status():
    """Adding a failure status must not silently leave the docs understating exit 3.

    Scoped to the exit-code bullet itself: a status named somewhere else in the
    file (the Mutation Verification prose) does not tell an agent reading the
    exit-code contract that the status maps to 3, which was exactly #614's gap.
    Indented continuation lines are folded in so rewrapping the bullet reports
    what actually went missing instead of all five statuses.
    """
    lines = _doc_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("- Exit codes:")]
    assert len(starts) == 1, f"expected one exit-code bullet, found {len(starts)}"
    bullet = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith((" ", "\t")) or not line.strip():
            break
        bullet.append(line)
    text = " ".join(bullet)
    missing = sorted(s for s in FAILED_MUTATION_STATUSES if f"`{s}`" not in text)
    assert not missing, f"statuses missing from the exit-code bullet: {missing}"
