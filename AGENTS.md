# Repository Guidelines

## Project Structure & Module Organization

`bn` is a Python 3.14+ CLI for Binary Ninja plus a companion bridge plugin.
CLI code lives in `src/bn/`; command handlers are grouped under
`src/bn/commands/`, shared rendering is in `src/bn/formatters.py`, and socket
discovery/transport code is in `src/bn/transport.py`. The Binary Ninja plugin
and bridge handlers live in `plugin/bn_agent_bridge/`. Bundled agent skills are
under `skills/`. Tests live in `tests/`, with C/C++ taint fixtures in
`tests/taint_corpus/` and stress harnesses in `tests/stress/`.

## Build, Test, and Development Commands

- `uv run bn --help`: run the CLI from the checkout.
- `uv tool install -e .`: install the editable `bn` CLI on `PATH`.
- `bn plugin install`: symlink `plugin/bn_agent_bridge` into Binary Ninja.
- `bn skill install`: install bundled skills into supported agent directories.
- `uv run pytest`: run the main test suite.
- `uv run pytest tests/test_cli.py::test_name`: run one focused test.
- `uv run python tests/stress/run_command_surface_audit.py`: smoke-test command
  registration and read-only command behavior against synthetic fixtures.

`tests/stress/run_stress.sh` needs a real Binary Ninja install and stops all
registered `bn` instances during preflight.

## Coding Style & Naming Conventions

Use type hints and `from __future__ import annotations` in Python modules. Keep
CLI command handlers in the appropriate `src/bn/commands/*.py` file and name
them with a leading underscore, for example `_taint_forward`. Add text output in
`src/bn/formatters.py`, not inside command handlers. Bridge operations should be
registered once with `@op(..., lock="read"|"write"|"none")`; handler logic
belongs in focused bridge modules such as `read_xrefs.py` or
`mutation_engine.py`.

## Testing Guidelines

Pytest is the primary framework. Most tests use mocked Binary Ninja APIs; real
BN-dependent checks are limited to integration/stress paths. Mirror the source
area being changed: CLI behavior in `tests/test_cli.py`, bridge behavior in
`tests/test_bridge.py`, transport in `tests/test_transport.py`, and taint logic
in `tests/test_taint_engine.py` or `tests/test_taint_integration.py`. Add
synthetic fixtures instead of real target data.

## Commit & Pull Request Guidelines

Follow the existing commit style: `feat(scope): ...`, `fix(scope): ...`,
`test(scope): ...`, or `chore(scope): ...`; use `!` for breaking changes when
needed, as in `feat(json)!: ...`. Include issue or PR references when relevant
(`(#123)`). Pull requests should describe behavior changes, list tests run, and
call out user-facing output or protocol changes.

## Security & Data Hygiene

This repo is dogfooded on real firmware and proprietary binaries. Do not commit
target names, paths, addresses, decompiled output, instance IDs, or symbol names
from real targets. Reproduce bugs with license-clean synthetic fixtures and
invented, internally consistent examples.
