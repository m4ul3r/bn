"""Regression tests for the suite-wide hermeticity fixture (#589).

The only quality gate in this repo is a green `uv run pytest`, so the gate
itself has to be deterministic: it must not depend on the developer's shell
(``FORCE_COLOR`` makes stdlib argparse colorize usage text and breaks eight
assertions), and it must never read or write the developer's real
``~/.cache/bn`` state (sticky session pins, instance registries).

`tests/conftest.py` installs an autouse `_hermetic_env` fixture; these tests
assert that it is actually in force, that it can still be deliberately
overridden by a test that needs to (no over-correction), and that two tests
cannot observe each other's cache state.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import bn.cli
import pytest
from bn import paths, session_state


# --- 1. color determinism -------------------------------------------------

def test_color_env_is_pinned_regardless_of_developer_shell():
    assert os.environ.get("NO_COLOR") == "1"
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE", "PYTHON_COLORS"):
        assert var not in os.environ, f"{var} leaked into the test environment"


def test_argparse_usage_text_is_never_colorized():
    """Positive control: the bug (#589) was ANSI codes in usage/help text."""
    help_text = bn.cli.build_parser().format_help()
    assert "\x1b[" not in help_text
    assert "usage: bn" in help_text


def test_a_test_may_still_opt_into_color(monkeypatch):
    """Negative control: the fixture pins the default, it does not hard-disable
    argparse colorization for a test that deliberately asks for it."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "3")
    assert "\x1b[" in bn.cli.build_parser().format_help()


def test_color_override_is_restored_to_the_isolated_state():
    """Runs after the override test above: monkeypatch put the pin back."""
    assert os.environ.get("NO_COLOR") == "1"
    assert "FORCE_COLOR" not in os.environ


# --- 2. cache / session isolation ----------------------------------------

def test_cache_root_is_isolated_from_real_user_state():
    root = paths.cache_home()
    assert os.environ.get("BN_CACHE_DIR") == str(root)
    assert root.is_dir()
    real = Path.home() / ".cache" / "bn"
    assert real not in (root, *root.parents)


def _real_cache_home(monkeypatch) -> Path:
    """The cache root the fixture is shielding us from, computed the same way
    `paths.cache_home()` would on this platform with the pin absent."""
    with monkeypatch.context() as m:
        m.delenv("BN_CACHE_DIR", raising=False)
        return paths.cache_home()


def test_session_state_cannot_read_real_sticky_pins(monkeypatch):
    """The real repo may carry a sticky `bn target use` pin; a test must not see
    it, and a test's own pin must not land in the developer's real cache."""
    isolated = paths.session_state_path()
    real = _real_cache_home(monkeypatch) / "sessions" / isolated.name

    # Discriminating: with the fixture reverted these two are the same file.
    assert isolated != real
    assert Path.home() not in isolated.parents
    assert not isolated.exists(), "a sticky pin leaked into the isolated cache"

    session_state.update(target="isolation-probe")
    assert isolated.exists()
    assert session_state.read()["target"] == "isolation-probe"
    assert not real.exists(), "a test's sticky pin landed in the real user cache"


_SEEN_ROOTS: list[Path] = []


@pytest.mark.parametrize("name", ["first", "second"])
def test_two_tests_cannot_observe_each_others_cache_state(monkeypatch, name):
    root = paths.cache_home()
    # Assert isolation BEFORE mutating: if the fixture regresses, `root` is the
    # developer's real ~/.cache/bn and this test would otherwise write into it.
    assert root != _real_cache_home(monkeypatch)
    assert Path.home() not in root.parents
    assert root not in _SEEN_ROOTS, "cache root shared between tests"
    _SEEN_ROOTS.append(root)
    leaked = list(root.rglob("leaked-*"))
    assert leaked == [], f"state leaked from a previous test: {leaked}"
    (root / f"leaked-{name}").write_text(name)


@pytest.mark.no_cache_isolation
def test_a_test_may_opt_out_of_the_cache_pin():
    """Negative control for the opt-out branch in conftest's `_hermetic_env`:
    a marked test sees no `BN_CACHE_DIR` at all, so it can exercise
    `paths.cache_home()`'s platform-default selection (which the pin
    short-circuits). Unmarked tests get the pin -- see the tests above."""
    assert "BN_CACHE_DIR" not in os.environ
    assert paths.cache_home() == _platform_default_cache_home()


def _platform_default_cache_home() -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Caches" / "bn"
    if system == "Windows" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "bn"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "bn"
    return home / ".cache" / "bn"


def test_a_test_may_still_point_bn_cache_dir_elsewhere(monkeypatch, tmp_path):
    """Negative control: the fixture must not defeat an explicit per-test pin
    (tests/test_transport.py does this dozens of times)."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("BN_CACHE_DIR", str(other))
    assert paths.cache_home() == other


def test_cache_override_is_restored_to_the_isolated_state():
    assert os.environ.get("BN_CACHE_DIR") == str(paths.cache_home())
    assert paths.cache_home().is_dir()
