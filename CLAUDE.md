# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is the canonical agent-instruction file: root `AGENTS.md` is a tracked symlink to it, so the two can never drift (#607).

## What This Is

`bn` is an agent-friendly CLI for Binary Ninja. It has two parts: a Python CLI (`src/bn/`) and a Binary Ninja bridge plugin package (`src/bn_agent_bridge/`). They communicate over a Unix socket using a JSON request/response protocol.

## Build & Run

```bash
uv tool install -e .          # Install CLI on PATH
bn plugin install              # Symlink bridge into BN plugins dir
bn skill install               # Symlink skills into ~/.claude/skills/, plus each other agent skill root whose home exists (~/.codex/skills/, the omp agent's skills/)

uv run bn --help               # Run CLI from repo without installing
```

Requires Python >= 3.14 and uv.

## Testing

```bash
uv run pytest                              # All tests
uv run pytest -n 8                         # All tests, in parallel (pytest-xdist)
uv run pytest tests/test_cli_core.py       # one module
uv run pytest tests/test_cli_mutation.py::test_mutation_summary_committed_noop_is_not_dirty  # single test
uv run pytest -v                           # Verbose output
uv run pytest -m "not real_bn"             # skip the lanes that drive a real BN
```

The suite is xdist-clean, and parallel is where the wall time is: `-n 8` takes the full run from roughly seven and a half minutes to two and a half on a six-core laptop. It is not in `addopts` because a serial run is the one that gives a readable failure and a usable `-x`.

Tests mock the `binaryninja` module — no BN license needed except for the two real-BN lanes, `test_integration.py` and `test_taint_integration.py`, which need a real BN install plus a C toolchain (`cc`/`gcc` + `make`). BN is discovered the same way the CLI discovers it (platform defaults), except that `BN_INSTALL_DIR` is authoritative here: if it points at a non-install, the lane treats BN as absent rather than falling back to another install.

The `tests/fixtures/*_x86_64` binaries those tests run against are compiler output and stay untracked. A session-scoped fixture in `tests/conftest.py` builds them automatically when BN is present; if the build fails — missing compiler, make error, or timeout — the run errors out with `FixtureBuildError` and the diagnostics rather than skipping. Manual fallback: `make -C tests/fixtures`.

```bash
BN_REQUIRE_REAL_TESTS=1 uv run pytest tests/test_integration.py tests/test_taint_integration.py   # strict: fail (not skip) if BN is missing
```

Without BN the `real_bn`-marked tests skip visibly. Use the strict invocation on any lane that is *supposed* to have BN, so an absent install can't report green. Both real-BN modules carry `pytest.mark.real_bn` — never a bare module-level `skipif`, which bypasses the strict gate entirely. The same flag refuses any skip the machine cannot un-skip, not just an absent BN: a test whose precondition is environmental rather than installable — running as root, where the DAC denial a permission test asserts is bypassed — calls `refuse_silent_skip` (in `tests/conftest.py`), which fails under strict mode instead of disappearing.

### The real-BN lane's shared bridge

`shared_bn` (in `tests/conftest.py`) is one headless bridge per pytest session, and every real-BN test that is not *about* process lifecycle uses it. The lane's cost was lifecycle, not the work under test: `bn session start` measures ~2.9s warm (fork + BN import + analysis) and `session stop` ~0.6s, against ~0.2s to `bn load` into a live bridge and ~0.4s for a read command — so a bridge per test spent ~3.5s of startup to run a handful of sub-second commands.

What the fixture keeps is the isolation that mattered: `shared_bn.load(binary)` copies the binary into the test's own scratch directory first, so a retype/rename/tag/comment/create — or a `bn save`, which writes its `.bndb` beside the copy — cannot be observed by another test. Exactly one target is open while a test runs, which is what the lane's commands assume when they omit `--target`; `begin()` refuses a dirty bridge and the teardown closes everything, so that invariant is enforced rather than hoped for, and a leak fails the test that leaked.

Keep spawning a private session when the spawn IS the subject: `session start/stop/restart`, multi-instance selection, and anything proving a BNDB survives a reload. Under `-n` each worker gets its own bridge.

An expensive binary is analysed once and reloaded from its saved database: `_build_and_prime_aarch64_probe` cross-builds the `-static` AArch64 probe, loads it (~20s of analysis), saves the `.bndb` and closes it; each test then loads a copy in under a second, because the bridge prefers an adjacent `<binary>.bndb` (#717) and `SharedBridge.load` copies that sidecar along with the binary.

## Architecture

### Two-Process Model

CLI (no BN dependency) → Unix socket → Bridge (owns all BN API access)

The bridge runs either as a **GUI plugin** (auto-starts when BN loads) or as a **headless process** (`bn-agent` / `python -m bn_agent_bridge`). The CLI discovers the bridge via a registry file + socket probe, auto-spawning headless if needed.

### CLI Layout (`src/bn/`)

`cli.py` is the entry point and shared infrastructure only — argparse plumbing, the `@command` decorator + `_COMMANDS` registry, target/instance resolution, the `_call` request wrapper, and `main()`. It does **not** define command handlers or text rendering anymore.

- `commands/` — handler modules grouped by concern: `binary.py` (load/close/save/refresh/target info), `function.py` (list/search/info/decompile/il/disasm/xrefs/callsites), `types.py`, `mutation.py`, `cpp_class.py` (`class list/show`), `dataflow.py` (`dataflow defuse/callgraph/values` and the `taint forward/backward/models` surface), `tags.py` (`tag list/get/add/remove`, `tag types`, `tag type create/remove`), `misc.py` (strings/imports/sections/bundle/py exec/batch), `admin.py` (doctor/plugin/skill install/session/instance/target pins). Importing the package via `commands/__init__.py` triggers `@command` decorators that populate `_COMMANDS`. Registering the same command path twice raises at import time.
- `formatters.py` — all text-mode rendering (`_render_*`, `_format_operation_result`). Add new text output here, not in `cli.py`.
- `transport.py` — socket I/O, bridge discovery, multi-instance registry, auto-spawn.
- `client.py` — the public `Client` surface (`request`, and `collect`, the page-aggregation loop that walks `{items, has_more}` and validates each page so a stale bridge cannot silently duplicate rows) for non-CLI consumers such as the `bn-kernel` backend.
- `output.py` — token-aware rendering and artifact spillover (>10k tokens → disk).
- `session_state.py` — sticky per-project pins (`instance_id`, `target`) read by `bn instance use` / `bn target use`.
- `paths.py` — all on-disk locations (cache, instances, sessions, spills, plugin/skills install dirs).
- `proc_identity.py` — durable process identity (boot id + pid + start ticks) and pidfd-pinned signalling, so the instance registry only ever signals a bridge it actually proved.
- `socket_evidence.py` — `path_has_bound_socket`: the kernel's own answer (from `/proc/net/unix`) to "is anything BOUND to this path", with `None` for every shape that listing cannot represent. Both processes destroy socket files — the CLI's `gc` sweep and the bridge's own `start()` — and a failed `connect()` cannot answer the question (a socket bound but not yet past `listen` refuses exactly like a leftover file), so a wrong negative unlinks a live bridge's endpoint. Symlinked into `src/bn_agent_bridge/` like `paths.py`, so the unlinking process and the process being unlinked share one rule.
- `target_hint.py` — the one multi-target hint grammar: the `-t`-prefixed, shell-quoted `Open targets:` listing and the headline/instruction wrapper around it. The CLI pre-flight (`_implicit_target`) and the bridge resolver refuse the same condition — several targets open, no selector — and used to render two different listings, so an agent that learned one misparsed the other and every message fix had to be made twice (#688). Rows carry `-t` because they are a copy-paste contract: `bn save`'s positional is an OUTPUT PATH, so a bare selector invites a wrong-file write. Symlinked into `src/bn_agent_bridge/` like `paths.py`.
- `version.py` — the single canonical `VERSION` (derived from `pyproject.toml`, else installed metadata) and the `build_id_*` fingerprints `doctor` compares; symlinked into `src/bn_agent_bridge/` like `paths.py`.
- `wire_limits.py` — the request-size ceiling (`MAX_REQUEST_BYTES`) and the `batch apply` preflight derived from it. The bridge has always refused an oversized request on arrival with a bare `request too large`; `bn batch apply` had no guard at all, so a runaway manifest was read, serialized and sent only to fail late with a message naming neither the limit nor a remedy (#769). The CLI now refuses first and says what to do, and `BN_BATCH_APPLY_MAX_OPS` / `BN_BATCH_APPLY_MAX_BYTES` raise that ceiling (`0` disables). Symlinked into `src/bn_agent_bridge/` like `paths.py`, so the process that refuses on arrival and the process that warns before sending read ONE number — a client-side copy guessed low would reject requests the bridge accepts, which is the duplicated-constant drift of #777/#890.
- `headless.py` — `bn-agent` entry point.

`src/bn_agent_bridge/paths.py`, `version.py`, `proc_identity.py`, `socket_evidence.py`, `target_hint.py` and `wire_limits.py` are symlinks to `src/bn/`, so the bridge and CLI agree on filesystem layout, version, process identity, socket evidence, the multi-target hint grammar and the request-size ceiling without duplication.

### Adding a New Command

1. Add a handler in the appropriate `src/bn/commands/*.py` module, decorated with `@command(...)` (declares help, output format, target requirement, pagination, address filter, args).
2. On the bridge, register the op with `@op("name", lock="read"|"write"|"none")` (from `op_registry.py`) and bind it to a handler. Put the handler logic as a free function in the relevant sibling module (`read_*.py`, `mutation_engine.py`, `taint_engine.py`, `vars.py`, `create_comments.py`), taking the `BridgeContext` seam (`ctx`) instead of `self`. The lock sets and dispatch routing are *derived* from the registry — don't hand-edit `READ_LOCKED_OPS` / `WRITE_LOCKED_OPS`.
3. Add tests in `tests/` (mirror the source layout).

`build_parser()` in `cli.py` walks `_COMMANDS` to construct the full argparse tree — no manual parser wiring needed.

### Bridge (`src/bn_agent_bridge/`)

The bridge is a package, not a monolith. `bridge.py` (by far the largest module — check it with `wc -l` rather than trusting a figure in prose) is the coordinator: it owns `TargetManager` (open `BinaryView`s held by strong reference and keyed by BN's handle-based identity — BN interns no Python wrapper for a non-focused view, so a weakref would die the moment the collecting call returns; selector resolution), the `BinaryNinjaBridge` facade + `dispatch()`, and the block of `@op` binders. Op *handler logic* lives in sibling modules as free functions that take the `BridgeContext` seam (`ctx`, in `seam.py`) instead of `self`: `read_*.py` (`read_class`, `read_decompile`, `read_evidence`, `read_go`, `read_listing`, `read_misc`, `read_tags`, `read_taint_models`, `read_taint_slice`, `read_types`, `read_xrefs`), `mutation_engine.py`, `taint_engine.py`, `vars.py`, `create_comments.py`. `BinaryNinjaBridge` keeps thin delegating shims for every handler the op binders and test suite reference, and the seam exists to break import cycles (read modules never import `bridge`/`mutation_engine`).

`op_registry.py` is the single source of truth: `@op(name, lock="read"|"write"|"none")` declares each op once, and both the lock sets and dispatch routing are derived from it (`REGISTRY.read_locked_ops()` / `write_locked_ops()`), as is the destructive-op gate (`destructive=True`, plus an optional `selector` reader for an op that resolves a selector from its params rather than the request's top level). Read ops dispatch under a shared writer-priority `_ReadWriteLock`; write ops under an exclusive lock; `none` ops run outside the dispatcher's lock and **self-manage locking**: the op body takes the write gate and/or the exclusive target lock itself. `lock="none"` is emphatically *not* "touches no BN state" — `load_binary`, `refresh` and `go_rename` all mutate the view; it means the dispatcher must not hold a lock for them, because they take their own (`refresh` runs analysis holding the analysis lock) or because holding one would deadlock them (`shutdown` and `cancel_request` must stay deliverable while a write op is wedged).

**Line count is not a split criterion (#629).** Split a module only on measured incohesion — disjoint call graphs and zero shared helpers, the test #592 applies to the incohesive `read_evidence` clusters — never on size: the R34-style splits of `taint_engine.py`, `mutation_engine.py` and `formatters.py` are rejected because each is a single connected component with cross-cutting helpers (taint's forward/backward share ~28 helpers, `mutation_engine`'s `_op_*`/`_verify_*` pairs and journal path change together, `formatters`' row/fallback helpers serve every command group), so a size-driven peel would export a large private surface and churn on every correctness fix. `bridge.py`'s residual bulk is the intentional thin-shim façade from #33, not a second business-logic monolith — peel only with a new measured seam.

### Target Selection

When only one target is open, target-required commands can omit `--target`. Multiple open targets require an explicit selector or a sticky pin via `bn target use`.

The bridge resolver keeps a count-free convenience for a bare, empty or `"active"` selector: it returns the focused GUI tab (headless has none, so it falls back to the sole open view). That convenience is refused for an op declared `destructive=True` in the op registry — `save_database`, `py_exec`, `batch_apply`, `go_rename` — which refuse on the COUNT while several targets are open, so a raw protocol client cannot overwrite whichever tab happened to have focus (#688). The policy lives in `dispatch()` and the registry flag, not in each handler; `close_binary` is the exception, guarding locally because its refusal must also name its `all=true` escape hatch. Counting is only half of it: the gate judges the selector the HANDLER will resolve (`batch_apply` prefers a `target` inside its manifest, so the registry's `selector` reader is the one decider both read), and nothing implicit is ever forwarded unbound: when the gate lets a bare request through it substitutes a `PinnedTarget` for the volatile selector -- an in-process `str` subclass, so nothing off the wire can carry pin semantics and no filename-derived selector can collide with one -- which `_matches_record` resolves by exact `target_id` and by none of the human-selector fallbacks (basename, `.bndb`-stripped core, path tail), because the handler resolves it later -- for a `lock="none"` op outside the gate's lock -- and a close/load in between must fail as an unknown selector rather than land the write on whatever replaced the view. An explicit-but-empty selector and an empty snapshot are refused by the gate itself rather than left to each handler: `py_exec` and `go_rename` have no such check, so theirs fell through to the focused tab. The view a bare selector resolves to comes from the same `refresh()` snapshot the listing was built from, so the targets a user is shown and the target a request lands on are one sample of GUI state.

### Multi-Instance Bridges

The CLI supports several headless bridges concurrently. Each instance gets its own files under `~/.cache/bn/instances/<id>.{json,sock}`; the GUI plugin uses the legacy fixed pair (`~/.cache/bn/bn_agent_bridge.{json,sock}`). Sticky per-project state (selected instance, selected target) lives under `~/.cache/bn/sessions/<sha>.json`, keyed by the project's git root so parallel agents in different repos don't collide.

### Mutation Verification

All mutations support `--preview` (apply → capture diffs → revert) and live verification (readback confirms requested state landed). Success-ish statuses: `verified` (requested state observed) and `noop` (already in the requested state). Failure statuses are exactly `FAILED_MUTATION_STATUSES` in `src/bn/formatters.py` — `unsupported`, `verification_failed`, `invalid_request`, `rollback_failed`, `internal_error` — and any of them puts the run at exit 3. A failed batch is fully reverted, and its cleanly rolled-back siblings are stamped `reverted`, which is **not** a failure and does not affect the exit code (#118); a sibling whose restore itself failed is stamped `rollback_failed` and is.

### Tags & function docs

- `bn tag list/get/add/remove` manage Binary Ninja tags at data/address/function
  scope; "bookmarks" are just `--type Bookmarks`. `bn tag types` and
  `bn tag type create/remove` manage tag types (built-ins cannot be removed).
- `bn comment --function` targets the function's documentation comment
  (`fn.comment`), shown atop the function — NOT an address comment. Use the
  function's entry address for an entry-line note.

### JSON Protocol

Request: `{"op": "decompile", "params": {...}, "target": "selector", "id": "uuid"}`
Response: `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`. On an
`OperationFailure` the `ok: false` envelope also carries `status`, `requested`
and `observed` describing the structured mutation failure; `requested` and
`observed` are `{}` when the failure supplied no detail. All three keys are
omitted entirely for other errors.

## Conventions

- Command handlers are named `_<group>_<subcommand>()` (e.g., `_function_list`)
- Exit codes: 0 = success, 1 = CLI-side handler error (e.g. partial `session start` failure), 2 = CLI argument-parser errors, handled `BridgeError` file/document/routing failures, read/resolver refusals, transport faults and other `BridgeError` cases without a failed mutation status, including a mutation result this CLI cannot classify AT ALL — malformed or newer than the CLI, so no verdict could be derived; 3 = a mutation call or local operation preflight whose status is `verification_failed`, `unsupported`, `invalid_request`, `rollback_failed`, or `internal_error`. Semantic pre-send mutation refusals use `_mutation_preflight` and report `invalid_request` with `observed.request_sent: false`; CLI argument-parser errors and handled file/document/routing/read/transport failures remain separate at exit 2, not arbitrary unhandled I/O exceptions. A bridge mutation refusal is exit 3 whether raised before apply or during apply. 4 = a `_mutate`-marked call whose compact summary reports `measured: false`, i.e. the counts could not be derived: the op returned no `results[]` rows to derive counts from AND registered no summary of its own to count with, or a field the summary counts FROM arrived in a shape no value reads out of, so it was refused and disclosed by name rather than fabricated as a zero (#715/#619); an op that counts through its own registered summary (`go rename`) is 0 only while those counters read — one whose counter arrives unreadable is `measured: false` and 4 like any other unmeasured run. So a single refused field is never 2: it still yields a verdict, and that verdict is 3 or 4. A failure still wins over 4 (3 before 4), and a measured all-`noop` is 0
- `BridgeError` for user-facing errors, `OperationFailure` for bridge-side mutation failures with structured fields
- Read commands default to `--format text`, mutations default to `--format json`
- Type hints everywhere, `from __future__ import annotations` in all modules
- Test files mirror source, split by concern rather than one module per package: `test_cli_*.py` (core/admin/binary/function/mutation/types/misc/formatters), `test_bridge_*.py` (dispatch/lifecycle/mutation/idle_reaper), `test_read_*.py`, `test_transport.py`, `test_output.py`, plus domain modules (`test_taint_*.py`, `test_op_registry.py`, `test_paths.py`, …). `tests/test_cli.py` and `tests/test_bridge.py` do not exist.
- Tests use `monkeypatch` fixtures and fake `binaryninja` module stubs

## Issues, PRs & Commits — Sanitize Test Data

This tool is dogfooded against real binaries (firmware, proprietary apps). **Never disclose data from those targets in anything shared or committed** — GitHub issues, PR descriptions, commit messages, review notes, or checked-in fixtures. Treat as sensitive: binary/target names, instance IDs, subsystem or product names, paths that reveal them, real function/symbol names, concrete addresses, and decompiled output lifted verbatim from a target.

Instead, **reproduce the bug or demonstrate the fix with realistic mock data that stands on its own.** Invent plausible function names, addresses, and structures that exhibit the same behavior, and keep them internally consistent so the example reads like a real session. A reader should understand the defect or the change from the example alone, without access to — or knowledge of — the original binary.
