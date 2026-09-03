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

# A doc may legitimately *name* a path in prose (e.g. "test_cli.py does not
# exist"); only lines that tell an agent to RUN something must resolve.
_PYTEST_PATH = re.compile(r"tests/test_\w+\.py(?:::(\w+))?")


def _runnable_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "pytest" in line]


def test_agents_md_is_a_symlink_to_claude_md():
    """A second physical copy is how AGENTS.md drifted onto a ghost tree (#607)."""
    assert AGENTS_MD.is_symlink(), (
        "AGENTS.md must be a symlink to CLAUDE.md, not a copy -- a copy drifts"
    )
    assert os.readlink(AGENTS_MD) == "CLAUDE.md"
    assert AGENTS_MD.resolve() == CLAUDE_MD.resolve()


def test_agent_docs_do_not_point_at_the_ghost_plugin_tree():
    """`plugin/bn_agent_bridge/` holds only stale bytecode; sources are in src/."""
    assert "plugin/bn_agent_bridge" not in CLAUDE_MD.read_text(encoding="utf-8")


@pytest.mark.parametrize("line", _runnable_lines(CLAUDE_MD.read_text(encoding="utf-8")))
def test_documented_pytest_invocations_resolve(line: str):
    """Every `uv run pytest <path>` in the docs must name a real module/test."""
    for match in _PYTEST_PATH.finditer(line):
        module = REPO / match.group(0).split("::", 1)[0]
        assert module.is_file(), f"{module.relative_to(REPO)} does not exist: {line!r}"
        test_id = match.group(1)
        if test_id is not None:
            source = module.read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(test_id)}\(", source, re.M), (
                f"{test_id} is not defined in {module.relative_to(REPO)}: {line!r}"
            )


def test_exit_code_3_lists_every_failed_mutation_status():
    """Adding a failure status must not silently leave the docs understating exit 3."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    missing = sorted(s for s in FAILED_MUTATION_STATUSES if f"`{s}`" not in text)
    assert not missing, f"statuses missing from CLAUDE.md exit-code docs: {missing}"
