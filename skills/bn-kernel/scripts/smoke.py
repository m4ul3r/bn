#!/usr/bin/env python3
"""Exercise bn_kernel without printing target-derived data."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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


def _loaded_selector(stdout: str) -> str:
    payload = json.loads(stdout)
    loaded = payload.get("loaded") if isinstance(payload, dict) else None
    selectors = [
        target.get("selector")
        for item in loaded or []
        if isinstance(item, dict)
        for target in item.get("targets") or []
        if isinstance(target, dict) and isinstance(target.get("selector"), str)
    ]
    if len(selectors) != 1:
        raise RuntimeError(
            f"session start returned {len(selectors)} target selectors; expected 1"
        )
    return selectors[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default="/bin/ls")
    parser.add_argument("--instance", default=f"bn-kernel-smoke-{os.getpid()}")
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

    start_env = os.environ.copy()
    start_env["BN_IDLE_TIMEOUT"] = "3600"
    succeeded = False
    target_selector: str | None = None
    try:
        started = subprocess.run(
            [
                executable,
                "session",
                "start",
                args.binary,
                "--instance-id",
                args.instance,
                "--format",
                "json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=start_env,
        )
        if started.returncode:
            print(f"FAIL session start returncode={started.returncode}")
        else:
            try:
                target_selector = _loaded_selector(started.stdout)
                asyncio.run(_exercise(args.instance, args.backend))
                succeeded = True
            except Exception as exc:
                print(f"FAIL smoke error={type(exc).__name__}")
    finally:
        if not args.keep:
            try:
                if target_selector is not None:
                    try:
                        closed = subprocess.run(
                            [
                                executable,
                                "-i",
                                args.instance,
                                "target",
                                "close",
                                target_selector,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                        )
                    except Exception as exc:
                        print(f"FAIL target close error={type(exc).__name__}")
                        succeeded = False
                    else:
                        if closed.returncode:
                            print(
                                f"FAIL target close returncode={closed.returncode}"
                            )
                            succeeded = False
                        else:
                            print("PASS target close")
            finally:
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
