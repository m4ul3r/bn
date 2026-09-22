"""Tracked-tree hygiene: local scratch artifacts must not reach `git status`.

#789's durable half -- the half a fresh clone can see -- is that the root review
and dogfood notes the audits write were untracked but NOT ignored, so every one
of them showed up as an untracked file: noise that at best gets skimmed past and
at worst gets committed by accident. The finding's other half is about an
untracked scratch script of the same class, which no test can see at all.

The claim is asserted by ASKING GIT rather than by matching the `.gitignore`
text: the property that matters is the one `git status` acts on, and a pattern
that reads correctly while matching nothing passes a text check.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# One name per family in .gitignore's two scratch blocks, in the order they are
# listed there. A family is one row because a pattern is one line: a `GOAL_*`
# line that does not match and a `GROK_REVIEW_*` line that does not match are
# separate ways for an artifact to reach `git status`, and each gets its own
# failure. Deliberately absent are the block's non-family entries -- `.dogfood/`
# (a directory), and `docs` / `AGENTS.md`, whose negation is a structural
# exception rather than a local artifact.
ROOT_SCRATCH_ARTIFACTS = (
    "DEEPSEEK_aria.md",
    "RE-VIBE.json",
    "vibe_check.py",
    "vibe_report.json",
    "CODEX_CODE_REVIEW_06070809.md",
    "DOGFOOD_flywheel.md",
    "FORWARD_TAINT_DESIGN.md",
    "GOAL_taint_sweep.md",
    "GROK_REVIEW_07172026.md",
    "IDEA_001.md",
    "TESTING_PLAN_aria.md",
    "WORK_POOLS.md",
    "WATCHDOG.yml",
    "omp_ideas.md",
    "hermes.md",
)


def _git_usable() -> bool:
    """`git check-ignore` must be able to answer for this checkout."""
    return shutil.which("git") is not None and subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=REPO, capture_output=True, text=True, check=False,
    ).returncode == 0


@pytest.mark.parametrize("name", ROOT_SCRATCH_ARTIFACTS)
def test_root_scratch_artifacts_are_ignored(name: str):
    if not _git_usable():
        pytest.skip("git is required to ask what `git status` would hide")

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", name],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert ignored.returncode == 0, (
        f"{name!r} at the repo root is not ignored, so a local scratch artifact "
        f"shows up as an untracked file in `git status`. The root-notes block of "
        f".gitignore is where that is decided."
    )
