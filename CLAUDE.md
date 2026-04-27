# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`bn` is an agent-friendly CLI for Binary Ninja. It has two parts: a Python CLI (`src/bn/`) and a Binary Ninja bridge plugin (`plugin/bn_agent_bridge/`). They communicate over a Unix socket using a JSON request/response protocol.

## Build & Run

```bash
uv tool install -e .          # Install CLI on PATH
bn plugin install              # Symlink bridge into BN plugins dir
bn skill install               # Symlink skills into ~/.claude/skills/ and, when present, ~/.codex/skills/

uv run bn --help               # Run CLI from repo without installing
```

Requires Python >= 3.14 and uv.

## Testing

```bash
uv run pytest                              # All tests
uv run pytest tests/test_cli.py            # CLI tests only
uv run pytest tests/test_cli.py::test_foo  # Single test
uv run pytest -v                           # Verbose output
```

Tests mock the `binaryninja` module — no BN license needed except for `test_integration.py` which requires a real BN install at `/opt/binaryninja`.

## Architecture

### Two-Process Model

CLI (no BN dependency) → Unix socket → Bridge (owns all BN API access)

The bridge runs either as a **GUI plugin** (auto-starts when BN loads) or as a **headless process** (`bn-agent` / `python -m bn_agent_bridge`). The CLI discovers the bridge via a registry file + socket probe, auto-spawning headless if needed.

### Key Files

- `src/bn/cli.py` — All CLI commands, argument parsing, output formatting (~2.2k LOC)
- `src/bn/transport.py` — Socket communication, bridge discovery, auto-spawn
- `src/bn/output.py` — Token-aware rendering, artifact spillover (>10k tokens → disk)
- `plugin/bn_agent_bridge/bridge.py` — Core bridge: target manager, all operation handlers, mutation engine (~3.2k LOC)
- `plugin/bn_agent_bridge/paths.py` / `version.py` — Symlinks to `src/bn/` shared modules

### Declarative Command Registration

Commands are registered via `@command()` decorator in `cli.py`. The decorator declares help text, output format, whether a target is needed, pagination, address filtering, and argument specs. A single `build_parser()` function walks `_COMMANDS` to construct the argparse tree — no manual parser wiring.

To add a new command: decorate a handler function with `@command(...)`, add the corresponding operation in `bridge.py`'s `dispatch()`, and write tests.

### Target Management

The bridge tracks open BinaryViews via `TargetManager` using weak references. When only one target is open, CLI commands can omit `--target`. Multiple open targets require an explicit selector.

### Mutation Verification

All mutations support `--preview` (apply → capture diffs → revert) and live verification (readback confirms requested state landed). Statuses: `verified`, `noop`, `unsupported`, `verification_failed`. Failed batches are fully reverted.

### Read/Write Locking

Bridge operations are categorized as `READ_LOCKED_OPS` (concurrent) or `WRITE_LOCKED_OPS` (exclusive). This is enforced in `bridge.py`'s dispatch path.

### JSON Protocol

Request: `{"op": "decompile", "params": {...}, "target": "selector", "id": "uuid"}`
Response: `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`

## Conventions

- Command handlers are named `_<group>_<subcommand>()` (e.g., `_function_list`)
- Exit codes: 0 = success, 1 = handler/mutation error, 2 = BridgeError, 3 = verification failed
- `BridgeError` for user-facing errors, `OperationFailure` for bridge-side mutation failures with structured fields
- Read commands default to `--format text`, mutations default to `--format json`
- Type hints everywhere, `from __future__ import annotations` in all modules
- Test files mirror source: `test_cli.py`, `test_bridge.py`, `test_transport.py`, `test_output.py`
- Tests use `monkeypatch` fixtures and fake `binaryninja` module stubs
