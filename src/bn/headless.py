"""Headless entry point: ``bn-agent [binary ...]``"""
from __future__ import annotations

import os
import platform
import argparse
import sys
from pathlib import Path

_DEFAULT_BN_DIRS = {
    "Linux": ["/opt/binaryninja"],
    "Darwin": ["/Applications/Binary Ninja.app/Contents/Resources"],
    "Windows": [],
}


def _find_bn_python() -> Path | None:
    """Return the ``python/`` directory inside a Binary Ninja installation."""
    env = os.environ.get("BN_INSTALL_DIR")
    if env:
        candidate = Path(env).expanduser() / "python"
        if candidate.is_dir():
            return candidate

    for d in _DEFAULT_BN_DIRS.get(platform.system(), []):
        candidate = Path(d) / "python"
        if candidate.is_dir():
            return candidate

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bn-agent",
        description="Run the BN Agent Bridge in headless mode",
    )
    parser.add_argument(
        "binaries",
        nargs="*",
        help="Binary file paths to open at startup",
    )
    args = parser.parse_args(argv)

    # Make the binaryninja package importable.
    bn_python = _find_bn_python()
    if bn_python is not None and str(bn_python) not in sys.path:
        sys.path.insert(0, str(bn_python))

    # The bridge plugin lives outside the installed package.  Resolve it
    # relative to the repo so ``uv run bn-agent`` works from a dev install.
    plugin_dir = Path(__file__).resolve().parents[2] / "plugin"
    if plugin_dir.is_dir() and str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))

    from bn_agent_bridge.bridge import start_headless

    start_headless(args.binaries)
    return 0
