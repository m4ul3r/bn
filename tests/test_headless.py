"""Tests for the headless entry points.

Covers the two CLI front-ends that bring a binary into a headless bridge:
``bn-agent`` (``src/bn/headless.py``) and ``python -m bn_agent_bridge``
(``plugin/bn_agent_bridge/__main__.py``). Both must thread ``--no-bndb`` through
to ``start_headless`` so the sidecar-``.bndb`` opt-out matches ``bn load`` (#178).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from bn import headless

# Sentinel default for the start_headless stub: NOT True/False, so a caller that
# forgets to thread prefer_bndb is caught instead of silently passing on the
# stub's own default (which would make the default-prefer test tautological).
_SENTINEL = object()

_PLUGIN_PKG_DIR = Path(__file__).resolve().parents[1] / "plugin" / "bn_agent_bridge"


def _fake_start_headless(captured: dict):
    def fake(binaries=None, instance_id=None, quick=False, prefer_bndb=_SENTINEL):
        captured["binaries"] = binaries
        captured["instance_id"] = instance_id
        captured["quick"] = quick
        captured["prefer_bndb"] = prefer_bndb

    return fake


def _install_fake_bridge(monkeypatch, captured: dict) -> types.ModuleType:
    """Put a fake ``bn_agent_bridge`` + ``.bridge`` in sys.modules.

    Lets the entry points resolve ``start_headless`` to a capturing stub without
    importing the real package (whose ``__init__`` pulls in ``binaryninja``).
    Returns the fake package module so callers can set ``__path__`` if needed.
    """
    fake_pkg = types.ModuleType("bn_agent_bridge")
    fake_bridge = types.ModuleType("bn_agent_bridge.bridge")
    fake_bridge.start_headless = _fake_start_headless(captured)
    monkeypatch.setitem(sys.modules, "bn_agent_bridge", fake_pkg)
    monkeypatch.setitem(sys.modules, "bn_agent_bridge.bridge", fake_bridge)
    return fake_pkg


# --- bn-agent (src/bn/headless.py) -----------------------------------------


def test_bn_agent_prefers_bndb_by_default(monkeypatch):
    captured: dict = {"prefer_bndb": _SENTINEL}
    _install_fake_bridge(monkeypatch, captured)

    rc = headless.main(["foo.bin"])

    assert rc == 0
    assert captured["prefer_bndb"] is True


def test_bn_agent_no_bndb_opt_out(monkeypatch):
    captured: dict = {"prefer_bndb": _SENTINEL}
    _install_fake_bridge(monkeypatch, captured)

    rc = headless.main(["foo.bin", "--no-bndb"])

    assert rc == 0
    assert captured["prefer_bndb"] is False


# --- python -m bn_agent_bridge (plugin/bn_agent_bridge/__main__.py) ---------


def _load_module_main(monkeypatch, captured: dict):
    """Load the plugin ``__main__.py`` as ``bn_agent_bridge.__main__``.

    Loads from file with a fake parent package so the relative
    ``from .bridge import start_headless`` resolves to the capturing stub and the
    real ``__init__`` (which imports binaryninja) is never executed.
    """
    fake_pkg = _install_fake_bridge(monkeypatch, captured)
    fake_pkg.__path__ = [str(_PLUGIN_PKG_DIR)]

    spec = importlib.util.spec_from_file_location(
        "bn_agent_bridge.__main__", _PLUGIN_PKG_DIR / "__main__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "bn_agent_bridge.__main__", mod)
    spec.loader.exec_module(mod)
    return mod


def test_module_main_prefers_bndb_by_default(monkeypatch):
    captured: dict = {"prefer_bndb": _SENTINEL}
    mod = _load_module_main(monkeypatch, captured)

    mod.main(["foo.bin"])

    assert captured["prefer_bndb"] is True


def test_module_main_no_bndb_opt_out(monkeypatch):
    captured: dict = {"prefer_bndb": _SENTINEL}
    mod = _load_module_main(monkeypatch, captured)

    mod.main(["foo.bin", "--no-bndb"])

    assert captured["prefer_bndb"] is False
