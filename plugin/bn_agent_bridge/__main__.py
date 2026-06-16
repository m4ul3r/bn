"""Headless entry point: python -m bn_agent_bridge [binary ...]"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="bn_agent_bridge",
        description="Run the BN Agent Bridge in headless mode",
    )
    parser.add_argument(
        "binaries",
        nargs="*",
        help="Binary file paths to open at startup",
    )
    parser.add_argument(
        "--no-bndb",
        dest="no_bndb",
        action="store_true",
        help="Open the raw binary even if an adjacent <binary>.bndb exists "
        "(default: load the saved sidecar database, like `bn load`)",
    )
    args = parser.parse_args(argv)

    from .bridge import start_headless

    start_headless(args.binaries, prefer_bndb=not args.no_bndb)


if __name__ == "__main__":
    sys.exit(main() or 0)
