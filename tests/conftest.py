"""Shared pytest fixtures for the bn test suite.

`fake_transport` removes the per-test boilerplate of redeclaring a
`fake_send_request` closure: it installs a fake `bn.cli.send_request` that
records every call and returns canned results keyed by op, and hands back the
recorded-calls list so a test can assert on the request the CLI built (the
CLI's contract is argv -> bridge request, so this is a real assertion, not a
tautology). Bridge-side tests keep using the `_bridge_fakes._load_bridge` seam.
"""
from __future__ import annotations

import bn.cli
import pytest


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
