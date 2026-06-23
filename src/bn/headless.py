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
    parser.add_argument(
        "--instance-id",
        help="Instance ID for this bridge session (default: random)",
        default=None,
    )
    parser.add_argument(
        "--quick",
        "--no-analysis",
        dest="quick",
        action="store_true",
        help="Preload binaries without full analysis (fast); run `bn refresh` for full analysis",
    )
    parser.add_argument(
        "--no-bndb",
        dest="no_bndb",
        action="store_true",
        help="Open the raw binary even if an adjacent <binary>.bndb exists "
        "(default: load the saved sidecar database, like `bn load`)",
    )
    args = parser.parse_args(argv)

    # Make the binaryninja package importable.
    bn_python = _find_bn_python()
    if bn_python is not None and str(bn_python) not in sys.path:
        sys.path.insert(0, str(bn_python))

    # When this file is executed directly from a checkout, add the src root so
    # the sibling bn_agent_bridge package is importable. Console scripts and
    # wheel installs already have it on sys.path.
    src_dir = Path(__file__).resolve().parents[1]
    if (src_dir / "bn_agent_bridge").is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bn_agent_bridge.bridge import start_headless

    start_headless(
        args.binaries,
        instance_id=args.instance_id,
        quick=args.quick,
        prefer_bndb=not args.no_bndb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
