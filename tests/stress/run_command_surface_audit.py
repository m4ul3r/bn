#!/usr/bin/env python3
"""Command-surface smoke audit for bn.

Enumerates every registered CLI command (``bn.cli._COMMANDS``) and:

  Phase 1 -- runs ``bn <command> --help`` for ALL commands, asserting the
             argparse tree is wired and nothing crashes building it.
  Phase 2 -- loads a tiny synthetic fixture and actually runs every target
             *read* command against it with best-effort synthesized arguments,
             asserting no command crashes (uncaught exception / segfault).

A "crash" is a Python traceback on stderr or an abnormal exit (segfault /
timeout). A clean handled error (BridgeError, invalid_request, exit 1/2/3) is
fine -- the audit only fails on crashes. Mutations and global-state commands
are help-only here (covered by the pytest suite and run_stress.sh); they are
listed explicitly so coverage gaps are never silent.

Fixtures are license-clean synthetic binaries built from tests/fixtures/src;
no target/proprietary data is involved. CI-free: run locally with
``uv run python tests/stress/run_command_surface_audit.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures"
FIXTURE = FIXTURE_DIR / "add_x86_64"

# Commands that mutate the view or global/session state: help-only here.
MUTATION_PATHS = {
    ("batch", "apply"), ("comment", "delete"), ("comment", "set"),
    ("function", "create"), ("local", "rename"), ("local", "retype"),
    ("proto", "set"), ("rename",), ("struct", "field", "delete"),
    ("struct", "field", "rename"), ("struct", "field", "set"),
    ("symbol", "rename"), ("types", "declare"),
}
GLOBAL_STATE_PATHS = {
    ("load",), ("close",), ("save",), ("refresh",), ("py", "exec"),
    ("session", "start"), ("session", "stop"), ("session", "list"),
    ("instance", "use"), ("instance", "clear"), ("instance", "list"),
    ("target", "use"), ("target", "clear"), ("target", "list"),
    ("plugin", "install"), ("skill", "install"), ("doctor",),
}


def _load_commands():
    sys.path.insert(0, str(REPO / "src"))
    from bn import cli
    import bn.commands  # noqa: F401 -- populates _COMMANDS via @command
    return cli._COMMANDS


def _run(args, timeout=120):
    return subprocess.run(
        ["uv", "run", "bn", *args],
        capture_output=True, text=True, timeout=timeout, cwd=REPO,
    )


def _is_crash(proc) -> bool:
    if "Traceback (most recent call last)" in proc.stderr:
        return True
    # negative rc == killed by signal (segfault); >=124 == timeout/abnormal
    return proc.returncode < 0 or proc.returncode >= 124


def main() -> int:
    commands = _load_commands()

    # Build fixtures if missing (clean checkout).
    if not FIXTURE.exists():
        print("Building synthetic fixtures (make -C tests/fixtures)...")
        if subprocess.run(["make", "-C", str(FIXTURE_DIR)],
                          capture_output=True).returncode != 0:
            print("FATAL: fixture build failed (need cc/gcc + make)")
            return 2

    crashes: list[str] = []
    help_only: list[str] = []

    # ---- Phase 1: --help wiring for every command ----
    print("=== Phase 1: --help for every command ===")
    for spec in sorted(commands, key=lambda s: s["path"]):
        path = list(spec["path"])
        proc = _run([*path, "--help"], timeout=60)
        name = " ".join(path)
        if proc.returncode != 0 or "Traceback (most recent call last)" in proc.stderr:
            crashes.append(f"--help {name} (rc={proc.returncode})")
            print(f"  CRASH  {name} --help")
    print(f"  {len(commands)} commands checked")

    # ---- Phase 2: run target read commands against a fixture ----
    print("=== Phase 2: exercise read commands against a fixture ===")
    start = _run(["session", "start", str(FIXTURE), "--format", "json"], timeout=180)
    try:
        iid = json.loads(start.stdout)["instance_id"]
    except Exception:
        print(f"FATAL: could not start session: {start.stdout}\n{start.stderr}")
        return 2

    def bn_inst(args, timeout=120):
        return _run(["--instance", iid, *args], timeout=timeout)

    # Resolve a concrete function + address from the fixture.
    func = "main"
    addr = None
    info = bn_inst(["function", "info", func, "--format", "json"])
    try:
        data = json.loads(info.stdout)
        addr = data.get("address") or data.get("start")
    except Exception:
        pass
    addr = addr or "0x401156"

    # positional arg name -> synthesized value (read commands only)
    synth = {
        "identifier": func, "callee": func, "function": func,
        "query": "a", "address": addr, "at": addr,
        "type_name": "int32_t", "struct_name": "x",
    }
    # commands with no positional synth + need explicit options
    special = {
        ("taint", "forward"): ["taint", "forward", "--function", func],
        ("taint", "backward"): ["taint", "backward", "--function", func, "--sink", func],
        ("trace",): ["trace", func, addr],
    }

    ran = handled = 0
    for spec in sorted(commands, key=lambda s: s["path"]):
        path = tuple(spec["path"])
        name = " ".join(path)
        if not spec["target"] or path in MUTATION_PATHS or path in GLOBAL_STATE_PATHS:
            help_only.append(name)
            continue
        if path in special:
            argv = special[path]
        else:
            positionals = [a[0][0] for a in spec["args"] if not a[0][0].startswith("-")]
            unsynth = [p for p in positionals if p not in synth]
            if unsynth:
                help_only.append(f"{name} (unsynthesizable: {unsynth})")
                continue
            argv = list(path) + [synth[p] for p in positionals]
        proc = bn_inst(argv, timeout=180)
        ran += 1
        if _is_crash(proc):
            crashes.append(f"{name} (rc={proc.returncode})")
            print(f"  CRASH  {name}\n{proc.stderr[-400:]}")
        elif proc.returncode != 0:
            handled += 1
            print(f"  ok(handled rc={proc.returncode})  {name}")
        else:
            print(f"  ok  {name}")

    _run(["session", "stop", iid], timeout=60)

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"commands: {len(commands)}   phase2 ran: {ran} ({handled} handled-error)")
    print(f"help-only (mutations / global-state / unsynthesizable): {len(help_only)}")
    for h in help_only:
        print(f"    - {h}")
    if crashes:
        print(f"\nCRASHES ({len(crashes)}):")
        for c in crashes:
            print(f"    !! {c}")
        return 1
    print("\nNo crashes. OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
