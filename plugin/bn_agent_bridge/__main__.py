"""Headless entry point: python -m bn_agent_bridge [binary ...]"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="bn_agent_bridge",
        description="Run the BN Agent Bridge in headless mode",
    )
    parser.add_argument(
        "binaries",
        nargs="*",
        help="Binary file paths to open at startup",
    )
    args = parser.parse_args()

    from .bridge import start_headless

    start_headless(args.binaries)


if __name__ == "__main__":
    sys.exit(main() or 0)
