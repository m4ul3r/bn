"""Tests for the ``bn-agent`` headless entry point (``src/bn/headless.py``)."""
from __future__ import annotations

import sys
import types

from bn import headless


def _capture_start_headless(monkeypatch) -> dict:
    """Stub ``bn_agent_bridge.bridge.start_headless`` and capture its kwargs.

    ``headless.main`` imports ``start_headless`` lazily after manipulating
    sys.path, so inject a fake bridge module into sys.modules to intercept the
    call without needing the real bridge or a Binary Ninja install.
    """
    captured: dict = {}

    def fake_start_headless(binaries=None, instance_id=None, quick=False, prefer_bndb=True):
        captured["binaries"] = binaries
        captured["instance_id"] = instance_id
        captured["quick"] = quick
        captured["prefer_bndb"] = prefer_bndb

    fake_pkg = types.ModuleType("bn_agent_bridge")
    fake_bridge = types.ModuleType("bn_agent_bridge.bridge")
    fake_bridge.start_headless = fake_start_headless
    monkeypatch.setitem(sys.modules, "bn_agent_bridge", fake_pkg)
    monkeypatch.setitem(sys.modules, "bn_agent_bridge.bridge", fake_bridge)
    return captured


def test_bn_agent_prefers_bndb_by_default(monkeypatch):
    captured = _capture_start_headless(monkeypatch)

    rc = headless.main(["foo.bin"])

    assert rc == 0
    assert captured["prefer_bndb"] is True


def test_bn_agent_no_bndb_opt_out(monkeypatch):
    captured = _capture_start_headless(monkeypatch)

    rc = headless.main(["foo.bin", "--no-bndb"])

    assert rc == 0
    assert captured["prefer_bndb"] is False
