# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`bn` is an agent-friendly CLI for Binary Ninja. It has two parts: a Python CLI (`src/bn/`) and a Binary Ninja bridge plugin package (`src/bn_agent_bridge/`). They communicate over a Unix socket using a JSON request/response protocol.

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

### CLI Layout (`src/bn/`)

`cli.py` is the entry point and shared infrastructure only — argparse plumbing, the `@command` decorator + `_COMMANDS` registry, target/instance resolution, the `_call` request wrapper, and `main()`. It does **not** define command handlers or text rendering anymore.

- `commands/` — handler modules grouped by concern: `binary.py` (load/close/save/refresh/target info), `function.py` (list/search/info/decompile/il/disasm/xrefs/callsites), `types.py`, `mutation.py`, `misc.py` (strings/imports/sections/bundle/py exec/batch), `admin.py` (doctor/plugin/skill install/session/instance/target pins). Importing the package via `commands/__init__.py` triggers `@command` decorators that populate `_COMMANDS`. Registering the same command path twice raises at import time.
- `formatters.py` — all text-mode rendering (`_render_*`, `_format_operation_result`). Add new text output here, not in `cli.py`.
- `transport.py` — socket I/O, bridge discovery, multi-instance registry, auto-spawn.
- `output.py` — token-aware rendering and artifact spillover (>10k tokens → disk).
- `session_state.py` — sticky per-project pins (`instance_id`, `target`) read by `bn instance use` / `bn target use`.
- `paths.py` — all on-disk locations (cache, instances, sessions, spills, plugin/skills install dirs).
- `headless.py` — `bn-agent` entry point.

`src/bn_agent_bridge/paths.py` and `version.py` are symlinks to `src/bn/`, so the bridge and CLI agree on filesystem layout and version without duplication.

### Adding a New Command

1. Add a handler in the appropriate `src/bn/commands/*.py` module, decorated with `@command(...)` (declares help, output format, target requirement, pagination, address filter, args).
2. On the bridge, register the op with `@op("name", lock="read"|"write"|"none")` (from `op_registry.py`) and bind it to a handler. Put the handler logic as a free function in the relevant sibling module (`read_*.py`, `mutation_engine.py`, `taint_engine.py`, `vars.py`, `create_comments.py`), taking the `BridgeContext` seam (`ctx`) instead of `self`. The lock sets and dispatch routing are *derived* from the registry — don't hand-edit `READ_LOCKED_OPS` / `WRITE_LOCKED_OPS`.
3. Add tests in `tests/` (mirror the source layout).

`build_parser()` in `cli.py` walks `_COMMANDS` to construct the full argparse tree — no manual parser wiring needed.

### Bridge (`src/bn_agent_bridge/`)

The bridge is a package, not a monolith. `bridge.py` (~2k LOC) is the coordinator: it owns `TargetManager` (weak-reffed `BinaryView`s, selector resolution), the `BinaryNinjaBridge` facade + `dispatch()`, and the block of `@op` binders. Op *handler logic* lives in sibling modules as free functions that take the `BridgeContext` seam (`ctx`, in `seam.py`) instead of `self`: `read_*.py` (decompile/listing/xrefs/types/evidence/taint-slice/misc), `mutation_engine.py`, `taint_engine.py`, `vars.py`, `create_comments.py`. `BinaryNinjaBridge` keeps thin delegating shims for every handler the op binders and test suite reference, and the seam exists to break import cycles (read modules never import `bridge`/`mutation_engine`).

`op_registry.py` is the single source of truth: `@op(name, lock="read"|"write"|"none")` declares each op once, and both the lock sets and dispatch routing are derived from it (`REGISTRY.read_locked_ops()` / `write_locked_ops()`). Read ops dispatch under a shared writer-priority `_ReadWriteLock`; write ops under an exclusive lock; `none` ops run unlocked — only for ops that touch no BN state (e.g. `shutdown`).

### Target Selection

When only one target is open, target-required commands can omit `--target`. Multiple open targets require an explicit selector or a sticky pin via `bn target use`.

### Multi-Instance Bridges

The CLI supports several headless bridges concurrently. Each instance gets its own files under `~/.cache/bn/instances/<id>.{json,sock}`; the GUI plugin uses the legacy fixed pair (`~/.cache/bn/bn_agent_bridge.{json,sock}`). Sticky per-project state (selected instance, selected target) lives under `~/.cache/bn/sessions/<sha>.json`, keyed by the project's git root so parallel agents in different repos don't collide.

### Mutation Verification

All mutations support `--preview` (apply → capture diffs → revert) and live verification (readback confirms requested state landed). Statuses: `verified`, `noop`, `unsupported`, `verification_failed`. Failed batches are fully reverted.

### JSON Protocol

Request: `{"op": "decompile", "params": {...}, "target": "selector", "id": "uuid"}`
Response: `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`

## Conventions

- Command handlers are named `_<group>_<subcommand>()` (e.g., `_function_list`)
- Exit codes: 0 = success, 1 = CLI-side handler error (e.g. partial `session start` failure), 2 = `BridgeError` (transport failures and bridge-side errors, including mutation apply failures), 3 = mutation status `verification_failed` or `unsupported`
- `BridgeError` for user-facing errors, `OperationFailure` for bridge-side mutation failures with structured fields
- Read commands default to `--format text`, mutations default to `--format json`
- Type hints everywhere, `from __future__ import annotations` in all modules
- Test files mirror source: `test_cli.py`, `test_bridge.py`, `test_transport.py`, `test_output.py`
- Tests use `monkeypatch` fixtures and fake `binaryninja` module stubs

## Issues, PRs & Commits — Sanitize Test Data

This tool is dogfooded against real binaries (firmware, proprietary apps). **Never disclose data from those targets in anything shared or committed** — GitHub issues, PR descriptions, commit messages, review notes, or checked-in fixtures. Treat as sensitive: binary/target names, instance IDs, subsystem or product names, paths that reveal them, real function/symbol names, concrete addresses, and decompiled output lifted verbatim from a target.

Instead, **reproduce the bug or demonstrate the fix with realistic mock data that stands on its own.** Invent plausible function names, addresses, and structures that exhibit the same behavior, and keep them internally consistent so the example reads like a real session. A reader should understand the defect or the change from the example alone, without access to — or knowledge of — the original binary.
