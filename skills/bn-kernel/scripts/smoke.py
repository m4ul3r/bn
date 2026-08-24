#!/usr/bin/env python3
"""Exercise bn_kernel without printing target-derived data."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR / "src"))

_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import bn_kernel  # noqa: E402
finally:
    sys.dont_write_bytecode = _previous_dont_write_bytecode


def _check(label: str, condition: bool, *, count: int | None = None) -> None:
    if not condition:
        raise AssertionError(label)
    suffix = "" if count is None else f" count={count}"
    print(f"PASS {label}{suffix}")


async def _exercise(instance: str, backend: str) -> None:
    session = bn_kernel.session(instance=instance, backend=backend)
    print(f"PASS backend={session.backend}")

    info = await session.info()
    _check("info", isinstance(info, dict))

    functions = await session.functions(limit=7)
    _check("functions", 1 <= len(functions) <= 7, count=len(functions))
    _check(
        "function metadata",
        session.last is not None
        and isinstance(session.last.payload, dict)
        and session.last.payload.get("returned") == len(functions),
        count=len(functions),
    )

    identifier = functions[0].get("address") or functions[0].get("name")
    decompilation = await session.decompile(str(identifier))
    _check("decompile", isinstance(decompilation, str) and bool(decompilation.strip()))

    strings = await session.strings(limit=5)
    _check("strings", isinstance(strings, list), count=len(strings))

    imports = await session.imports(limit=5)
    _check("imports", isinstance(imports, list), count=len(imports))

    filtered = [row for row in functions if row.get("size", 0) >= 0]
    _check("local filter", isinstance(filtered, list), count=len(filtered))

    generic = await session.run("target", "info", unwrap=False)
    _check("generic CLI", isinstance(generic, dict))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default="/bin/ls")
    parser.add_argument("--instance", default="bn-kernel-smoke")
    parser.add_argument(
        "--backend", choices=("auto", "cli", "native"), default="auto"
    )
    parser.add_argument("--keep", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    executable = shutil.which("bn")
    if executable is None:
        print("FAIL bn executable unavailable")
        return 1

    started = subprocess.run(
        [
            executable,
            "session",
            "start",
            args.binary,
            "--instance-id",
            args.instance,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if started.returncode:
        print(f"FAIL session start returncode={started.returncode}")
        return 1

    succeeded = False
    try:
        asyncio.run(_exercise(args.instance, args.backend))
        succeeded = True
    except Exception as exc:
        print(f"FAIL smoke error={type(exc).__name__}")
    finally:
        if not args.keep:
            stopped = subprocess.run(
                [executable, "session", "stop", args.instance],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if stopped.returncode:
                print(f"FAIL session stop returncode={stopped.returncode}")
                succeeded = False
            else:
                print("PASS session stop")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
