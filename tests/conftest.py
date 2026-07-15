"""Shared pytest fixtures for the bn test suite.

`fake_transport` removes the per-test boilerplate of redeclaring a
`fake_send_request` closure: it installs a fake `bn.cli.send_request` that
records every call and returns canned results keyed by op, and hands back the
recorded-calls list so a test can assert on the request the CLI built (the
CLI's contract is argv -> bridge request, so this is a real assertion, not a
tautology). Bridge-side tests keep using the `_bridge_fakes._load_bridge` seam.

`_hermetic_env` is autouse: it pins the process environment every test (and
every subprocess a test spawns) runs under, so `uv run pytest` green means the
same thing on every machine. See `tests/test_suite_isolation.py`.
"""
from __future__ import annotations

import sys

import bn.cli
import pytest

sys.dont_write_bytecode = True

# Variables that make Python 3.14's stdlib argparse colorize usage/help text.
# `bn` has no color code of its own, but ~8 assertions on usage strings break
# when the developer's shell exports FORCE_COLOR (#589). requires-python is
# >=3.14, so this is permanent -- pin it rather than rewrite the assertions,
# which are testing the right thing.
_COLOR_FORCING_VARS = ("FORCE_COLOR", "CLICOLOR_FORCE", "PYTHON_COLORS")


@pytest.fixture(autouse=True)
def _hermetic_env(request, monkeypatch, tmp_path_factory):
    """Make every test's environment deterministic and free of real user state.

    - plain argparse output regardless of the developer's shell;
    - `BN_CACHE_DIR` pointed at a fresh per-test directory, so nothing can read
      or write the developer's real `~/.cache/bn` (instance registries, sticky
      `bn target use` / `bn instance use` pins) and no two tests can observe
      each other's cache state.

    A test may still override any of these with `monkeypatch` (many do); the
    override is restored to this isolated state at teardown. Tests that
    genuinely exercise platform-default path selection can opt out of the cache
    pin with `@pytest.mark.no_cache_isolation` -- narrow, and never for tests
    that merely touch the cache.
    """
    for var in _COLOR_FORCING_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    if request.node.get_closest_marker("no_cache_isolation") is None:
        cache_root = tmp_path_factory.mktemp("bn-cache")
        monkeypatch.setenv("BN_CACHE_DIR", str(cache_root))
    yield


@pytest.fixture
def fake_transport(monkeypatch):
    def install(results=None, *, default=None):
        results = results or {}
        calls = []

        def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                              instance_id=None, spawn_missing_named=False, **kwargs):
            calls.append({"op": op, "params": params, "target": target})
            if op in results:
                return results[op]
            if default is not None:
                return default
            raise AssertionError(f"unexpected op: {op}")

        monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
        return calls

    return install
