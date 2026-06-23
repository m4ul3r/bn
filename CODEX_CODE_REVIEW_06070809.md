# Codex Code Review 06070809

Date: 2026-06-11

Scope: current `/opt/bn` workspace, including `src/bn`, `plugin/bn_agent_bridge`, tests, README, bundled skills, packaging metadata, and runtime entry points.

Method: parallel read-only review with scoped subagents for CLI, transport/session, bridge lifecycle, bridge mutations, taint/dataflow, and output formatting, plus local packaging/docs validation. Existing tests were run once before this deeper pass: `uv run pytest` -> `551 passed, 3 skipped`.

## Executive Summary

The codebase has strong unit coverage and many recent defensive fixes, but several correctness problems remain around install packaging, lifecycle commands, GUI/headless behavior, mutation rollback, and taint resolution. The most serious risks are:

- A normal wheel install omits the bridge plugin and skills, so core setup commands fail outside editable/checkout installs.
- User-supplied instance IDs are used as filesystem paths without validation.
- Some command combinations can close or mutate a broader target than the command text implies.
- GUI `load`/`close` behavior is inconsistent with target discovery.
- Mutation preview/rollback paths can leave non-journaled local/prototype changes behind while reporting success or skipping restores.
- Forward taint and callgraph paths miss imports and Thumb-tagged indirect targets that other resolver code already knows how to handle.

## P1 Findings

### 1. Wheel installs omit the plugin and skills, breaking core setup commands

References:

- `pyproject.toml:12`
- `src/bn/paths.py:13`
- `src/bn/paths.py:104`
- `src/bn/commands/admin.py:173`
- `README.md:18`
- `README.md:25`
- `README.md:33`
- `README.md:74`

The project builds a wheel containing only `bn/*`. It does not include `plugin/bn_agent_bridge` or `skills/*`. In an editable checkout this is masked because `repo_root()` resolves back to `/opt/bn`, but in a clean wheel install `repo_root()` becomes the Python library directory and `plugin_source_dir()` points at a nonexistent `.../lib/python3.14/plugin/bn_agent_bridge`.

Validated clean-install behavior with a temporary Python 3.14 venv:

- `bn plugin install --dest /tmp/bn-review-plugin` fails with `Source directory is missing: .../plugin/bn_agent_bridge`.
- `bn skill install --dest /tmp/bn-review-skills` crashes with `FileNotFoundError: .../skills`.
- `python -m bn_agent_bridge --help` fails with `No module named bn_agent_bridge`.

Impact: users who install from a wheel or non-editable source cannot install the companion plugin or bundled skills, and the README-advertised module entry point is absent.

Suggested fix: package the bridge and skills as package data or separate packages, resolve install sources via `importlib.resources`, and either make `bn_agent_bridge` importable from the installed distribution or remove/update the documented `python -m bn_agent_bridge` path. Add a wheel-install smoke test for `bn plugin install --dest`, `bn skill install --dest`, and the advertised entry points.

### 2. Instance IDs can escape the instances directory

References:

- `src/bn/paths.py:76`
- `src/bn/paths.py:82`
- `src/bn/transport.py:385`
- `src/bn/transport.py:388`
- `src/bn/transport.py:398`

`bridge_registry_path(instance_id)` and `bridge_socket_path(instance_id)` append the supplied ID directly into a `Path`. IDs such as `../evil` or absolute paths can move registry, socket, and log files outside `instances_dir()`. `list_instances()` only scans `instances_dir/*.json`, so an escaped bridge can be started but later not listed or stopped by normal session commands.

Impact: bad instance IDs can create orphaned bridge processes and files outside the expected cache tree. This is also a path traversal primitive inside the user's cache/process environment.

Suggested fix: validate instance IDs at the CLI and transport boundary. Accept a strict basename grammar such as `^[A-Za-z0-9_.-]+$`, reject separators, absolute paths, `.` and `..`, and continue reserving `default`.

### 3. `bn close <path> --all` accepts conflicting selectors and closes everything

References:

- `src/bn/commands/binary.py:44`
- `src/bn/commands/binary.py:57`
- `src/bn/commands/binary.py:59`
- `plugin/bn_agent_bridge/bridge.py:1146`

The CLI accepts both a path and `--all`, forwards both, and the bridge gives `all_` priority. A command that visibly names one file can close every loaded target.

Repro shape:

```bash
bn close /tmp/one-binary --all
```

Impact: destructive surprise in multi-target sessions.

Suggested fix: make path and `--all` mutually exclusive, or reject `args.path and args.all` before sending a bridge request.

### 4. GUI `bn load` succeeds but creates invisible targets

References:

- `plugin/bn_agent_bridge/bridge.py:377`
- `plugin/bn_agent_bridge/bridge.py:378`
- `plugin/bn_agent_bridge/bridge.py:382`
- `plugin/bn_agent_bridge/bridge.py:1111`
- `plugin/bn_agent_bridge/bridge.py:1120`

In GUI mode, `_collect_open_views()` ignores `_headless_views` entirely and discovers only UI contexts/tabs. `_load_binary()` always appends to `_headless_views`, then returns `targets.refresh()`. A `bn load` sent to a GUI bridge can report success while the loaded `BinaryView` is omitted from `target list` and cannot be selected by later commands.

Impact: the user gets a successful load response for a target the CLI cannot subsequently see or operate on. The view is also retained outside normal GUI target management.

Suggested fix: either reject `load_binary` in GUI bridges with a clear error, or publish GUI loads through the Binary Ninja UI/open-file path. If headless-loaded views are supported in GUI bridges, merge `_headless_views` into `_collect_open_views()` even when `ui is not None`.

### 5. GUI `bn close -t ...` fails before closing a GUI target

References:

- `plugin/bn_agent_bridge/bridge.py:1133`
- `plugin/bn_agent_bridge/bridge.py:1135`
- `plugin/bn_agent_bridge/bridge.py:1136`
- `plugin/bn_agent_bridge/bridge.py:1140`

`_close_binary()` resolves the requested target first, but then enters `_headless_views_lock` and raises `No binaries are currently loaded` when `_headless_views` is empty. In a normal GUI bridge, GUI-opened targets are not in `_headless_views`, so target-based close fails before reaching the `target_bv is not None` branch.

Impact: `bn close -t active` cannot close normal GUI targets even though target resolution succeeded.

Suggested fix: handle `target_bv is not None` before checking `_headless_views`, and use the proper GUI/main-thread close API. Only mutate `_headless_views` when the closed view is actually tracked there.

### 6. `local_retype` verification can reject valid changes or verify the wrong local

References:

- `plugin/bn_agent_bridge/bridge.py:1889`
- `plugin/bn_agent_bridge/bridge.py:5404`
- `plugin/bn_agent_bridge/bridge.py:5414`

Local selection can target canonical/HLIL-visible variables, including register locals outside `stack_layout`. `_verify_local_retype()` re-resolves only by storage through `_find_variable_by_storage()`. It does not first match the stable identifier recorded in the operation result.

Impact: retyping an HLIL-visible register local from `bn local list` can apply successfully, then fail verification and roll back. If multiple variables share storage, verification can also inspect the wrong variable.

Suggested fix: mirror `_verify_local_rename()`: first resolve by `result["identifier"]` over `_iter_canonical_variables()`, then fall back to storage only when the identifier is absent, and do not accept a same-storage variable with a different identifier.

### 7. Rollback can skip explicit local/prototype restores when BN undo fails

References:

- `plugin/bn_agent_bridge/bridge.py:5698`
- `plugin/bn_agent_bridge/bridge.py:5705`

The apply-failure path computes:

```python
reverted = self._revert_undo_safely(bv, state) and self._run_local_restores(bv, restores)
```

Because `and` short-circuits, if `revert_undo_actions()` fails, the explicit restores for non-journaled `create_user_var()` / `set_user_type()` changes are never attempted.

Impact: a batch that applies a local/prototype mutation and then fails later can leave those non-journaled changes applied even when the explicit restore callback could have fixed them.

Suggested fix: always run both paths:

```python
undo_ok = self._revert_undo_safely(bv, state)
restore_ok = self._run_local_restores(bv, restores)
reverted = undo_ok and restore_ok
```

### 8. Preview can leave changes applied while returning `success: true`

References:

- `plugin/bn_agent_bridge/bridge.py:5737`
- `plugin/bn_agent_bridge/bridge.py:5755`
- `plugin/bn_agent_bridge/bridge.py:5757`
- `plugin/bn_agent_bridge/bridge.py:5765`

In the preview path, restore failure changes only `message` and `rolled_back`; `success` remains `not failed`. For non-journaled local/prototype mutations, a preview can therefore report success while also saying the view may be left modified.

Impact: automation that keys off `success` treats a dirty preview as clean.

Suggested fix: make preview success require successful rollback/restoration, or add a failed result status when `preview and not restored`.

### 9. Forward taint and callgraph miss modeled import calls

References:

- `plugin/bn_agent_bridge/taint_engine.py:256`
- `plugin/bn_agent_bridge/taint_engine.py:267`
- `plugin/bn_agent_bridge/taint_engine.py:1212`
- `plugin/bn_agent_bridge/taint_engine.py:1213`
- `plugin/bn_agent_bridge/taint_engine.py:1215`
- `plugin/bn_agent_bridge/bridge.py:3175`

The shared resolver handles `MLIL_IMPORT` by resolving the import name before falling back to the GOT-slot constant. Forward taint's main call loop still uses `const_target(dest)` and `_callee_name(target)`, and callgraph uses `_taint.const_target()` directly. Imported modeled calls such as `memcpy` or `system` can be treated as unresolved/unmodeled instead of matching the model database.

Impact: false negatives for common dynamically linked libc sources/sinks on binaries where Binary Ninja emits `MLIL_IMPORT`.

Suggested fix: use `resolve_call_target(..., follow_thunks=True)` in forward taint and callgraph direct callee reporting. Fall back to possible-value resolution only when the shared resolver cannot resolve.

### 10. Thumb-tagged indirect targets are not normalized

References:

- `plugin/bn_agent_bridge/taint_engine.py:284`
- `plugin/bn_agent_bridge/taint_engine.py:1271`
- `plugin/bn_agent_bridge/taint_engine.py:1272`
- `plugin/bn_agent_bridge/bridge.py:3148`

`targets_from_pvs()` returns raw possible-value addresses, and forward taint looks up `bv.get_function_at(taddr)` without also trying `taddr & ~1`. The bridge callgraph helper has the same exact-address lookup. Other resolver paths already account for Thumb low-bit tagging.

Impact: ARM/Thumb indirect dispatches can be reported as unmodeled externals even when the function exists at the normalized address, causing false negatives and misleading assumptions.

Suggested fix: centralize code-pointer normalization for all candidate call targets. Preserve raw addresses in diagnostics, but use the normalized function entry for lookup, thunk following, and model matching.

## P2 Findings

### 11. `start_headless(..., quick=True)` does not mark preloaded views as quick

References:

- `plugin/bn_agent_bridge/bridge.py:1099`
- `plugin/bn_agent_bridge/bridge.py:1106`
- `plugin/bn_agent_bridge/bridge.py:1213`
- `plugin/bn_agent_bridge/bridge.py:4339`
- `plugin/bn_agent_bridge/bridge.py:6220`
- `plugin/bn_agent_bridge/bridge.py:6223`

`bn load --quick` records the view in `_quick_loaded_views`; direct `bn-agent --quick <binary>` skips analysis but only appends the view to `_headless_views`. Later `target_info` reports full analysis, and `strings` can return a misleading empty result instead of the intended "run `bn refresh`" error.

Suggested fix: share the same load helper between preload and `load_binary`, or explicitly add quick preloads to `_quick_loaded_views` and discard full-analysis preloads.

### 12. Raw JSON booleans are coerced by truthiness

References:

- `plugin/bn_agent_bridge/bridge.py:819`
- `plugin/bn_agent_bridge/bridge.py:853`
- `plugin/bn_agent_bridge/bridge.py:854`
- `plugin/bn_agent_bridge/bridge.py:857`
- `plugin/bn_agent_bridge/bridge.py:1043`

The dispatcher accepts raw `params` without schema/type validation and uses `bool(...)` on request values. A raw client sending `"quick": "false"` enables quick load, and `"all": "false"` closes all because non-empty strings are truthy.

Impact: non-CLI clients or scripts can trigger destructive lifecycle behavior with malformed JSON that looks visually false.

Suggested fix: validate `params` as an object and validate known booleans as actual JSON booleans. Return a clean `invalid_request` error for invalid types.

### 13. Named auto-spawn is not serialized and can attach callers to the wrong process

References:

- `src/bn/transport.py:239`
- `src/bn/transport.py:369`
- `src/bn/transport.py:388`
- `src/bn/transport.py:398`
- `src/bn/transport.py:401`

Unnamed auto-spawn uses `_auto_spawn_locked()`, but explicit named spawns from `bn session start --instance-id X` or `bn load --instance X` call `spawn_instance(X)` without a lock. Two callers can pass the duplicate check, spawn two children, and both accept whichever registry file appears first without verifying it belongs to their child PID.

Impact: concurrent callers can load into the wrong bridge or leave an extra child racing/exiting.

Suggested fix: serialize all spawns by instance ID and verify the loaded registry PID matches the child process. If another process won, terminate/wait for the losing child and return a clear duplicate/race error.

### 14. `session stop` reports success before teardown converges

References:

- `src/bn/commands/admin.py:282`
- `src/bn/commands/admin.py:283`
- `src/bn/commands/admin.py:284`
- `src/bn/commands/admin.py:287`
- `src/bn/commands/admin.py:290`

`_session_stop()` treats a shutdown ACK or `SIGTERM` delivery as stopped. It does not wait for process exit, socket removal, or registry removal.

Impact: `bn session stop id && bn session start --instance-id id` can race against the old live registry/socket and fail as a duplicate. SIGTERM fallback also relies on later discovery to clean stale registry state.

Suggested fix: after graceful shutdown or signal fallback, poll until the registry/socket is gone or `_load_instance()` no longer reports the instance. Escalate after a timeout and report failure if teardown does not converge.

### 15. `bn instance use default` clears the pin

References:

- `src/bn/transport.py:65`
- `src/bn/commands/admin.py:366`
- `src/bn/commands/admin.py:374`
- `src/bn/session_state.py:37`

The fixed GUI bridge is selectable as `default` via `instance_selector()`, but its `instance_id` is `None`. `_instance_use()` stores `matches[0].instance_id`, so selecting `default` writes `None`, which removes the session-state key.

Impact: with a GUI bridge plus a named headless bridge, `bn instance use default` can report success but leave no pin; later bare commands still hit "Multiple instances".

Suggested fix: store `cli.instance_selector(matches[0])` instead of `matches[0].instance_id`, or special-case fixed GUI instances to persist `"default"`.

### 16. Comment locator options are not mutually exclusive

References:

- `src/bn/commands/mutation.py:65`
- `src/bn/commands/mutation.py:77`
- `src/bn/commands/mutation.py:91`
- `src/bn/commands/mutation.py:108`
- `plugin/bn_agent_bridge/bridge.py:4724`
- `plugin/bn_agent_bridge/bridge.py:5810`
- `plugin/bn_agent_bridge/bridge.py:5833`

`comment set/get/delete` accept both `--address` and `--function`. The bridge checks the function branch first, silently ignoring the explicit address.

Impact: comments can be read, set, or deleted at a function entry when the command also supplied a different address.

Suggested fix: use a mutually exclusive locator group for `--address` / `--function`. Make it required for `get`/`delete`; for `set`, require exactly one locator plus the comment text.

### 17. `types declare` accepts multiple declaration sources and silently chooses one

References:

- `src/bn/commands/types.py:52`
- `src/bn/commands/types.py:56`
- `src/bn/commands/types.py:57`
- `src/bn/commands/types.py:58`
- `src/bn/commands/types.py:62`

`--file`, `--stdin`, and positional declaration are all accepted together. The handler chooses `--file` first, then `--stdin`, then positional text, silently ignoring the rest.

Impact: a user or script can apply different declarations than intended.

Suggested fix: validate exactly one declaration source before reading input, ideally via argparse mutual exclusion plus explicit validation for the positional form.

### 18. Affected type diffs can be lost for resolved struct names

References:

- `plugin/bn_agent_bridge/bridge.py:4896`
- `plugin/bn_agent_bridge/bridge.py:4898`
- `plugin/bn_agent_bridge/bridge.py:5000`
- `plugin/bn_agent_bridge/bridge.py:5003`
- `plugin/bn_agent_bridge/bridge.py:6049`

Struct mutations accept canonical/case-insensitive type resolution through `_find_type()` and commit under the resolved name, but pre/post snapshots use the raw `op["struct_name"]` and direct `bv.get_type_by_name()`. If the requested name only works via fallback resolution, the mutation can verify while `affected_types` is empty or lacks the layout diff.

Impact: preview/apply output can omit the type diff that users rely on to audit struct edits.

Suggested fix: canonicalize struct type names in `_operation_type_names()` using `_find_type()`, or capture affected type names from the resolved operation results after apply.

### 19. Raw-byte `read --encoding bytes --out` bypasses output handling

References:

- `src/bn/commands/misc.py:146`
- `src/bn/commands/misc.py:162`
- `src/bn/output.py:186`
- `src/bn/output.py:188`
- `src/bn/output.py:190`

The raw-byte path writes directly with `args.out.write_bytes(data)`. It does not create parent directories, wrap `OSError` as a clean `BridgeError`, or emit an artifact envelope. The normal output path does all three.

Impact: `bn read ... --encoding bytes --out missing/dir/out.bin` can raise an uncaught `FileNotFoundError`, and successful writes produce empty stdout with no path/hash/size confirmation for automation.

Suggested fix: add a binary-aware output helper that mirrors `write_output_result()`: create parents, catch write errors, and emit a text/JSON/NDJSON artifact envelope with `format: bytes`.

### 20. Taint model loading silently ignores broken model files

References:

- `plugin/bn_agent_bridge/taint_engine.py:56`
- `plugin/bn_agent_bridge/taint_engine.py:59`
- `plugin/bn_agent_bridge/taint_engine.py:61`
- `plugin/bn_agent_bridge/taint_engine.py:68`

Both builtin and user override model loading catch all exceptions and continue. Malformed `BN_TAINT_MODELS`, unreadable overrides, invalid top-level shapes, or a corrupt builtin model DB produce no warning in the taint result.

Impact: user-added source/sink/propagation models can fail to load silently, causing false negatives that look like analysis limitations.

Suggested fix: make builtin model load failure a hard `TaintError`; for user overrides, either hard-fail or surface the failure in taint result metadata/assumptions. Validate top-level type and model shape before updating the model map.

### 21. `arg:` taint locators reject C++ qualified names

References:

- `plugin/bn_agent_bridge/taint_engine.py:1938`
- `plugin/bn_agent_bridge/taint_engine.py:1943`
- `plugin/bn_agent_bridge/taint_engine.py:1944`
- `plugin/bn_agent_bridge/taint_engine.py:1947`

`arg:<callee>:<n>` uses `rest.partition(":")`, so `arg:Namespace::method:1` parses `callee="Namespace"` and `n=":method:1"`. `ret:` locators preserve the full callee string and do not have this problem.

Impact: C++ methods/namespaced callees cannot be used by name as `arg:` sources/sinks; users must know and use an address locator.

Suggested fix: parse `arg:` with `rsplit(":", 1)` so only the final colon separates the index.

### 22. `load_binary` holds the write lock through full analysis

References:

- `plugin/bn_agent_bridge/bridge.py:823`
- `plugin/bn_agent_bridge/bridge.py:829`
- `plugin/bn_agent_bridge/bridge.py:850`
- `plugin/bn_agent_bridge/bridge.py:1108`

`load_binary` is write-locked for the entire load and `update_analysis_and_wait()` path. While a long analysis runs, read-locked ops such as `doctor`, `list_targets`, and `target_info` block behind it.

Impact: an existing bridge can become unresponsive to status/target reads for minutes during a large load.

Suggested fix: open/analyze into an unpublished local view, then take the exclusive target lock only while publishing/removing target state. If Binary Ninja global state truly requires serialization, split the target registry lock from the broader BN mutation lock so status commands can still respond.

## P3 Findings

### 23. Some raw list endpoints still accept unchecked paging values

References:

- `plugin/bn_agent_bridge/bridge.py:4535`
- `plugin/bn_agent_bridge/bridge.py:4572`
- `plugin/bn_agent_bridge/bridge.py:4747`
- `plugin/bn_agent_bridge/bridge.py:4771`

Several list-style bridge methods still slice with raw `offset`/`limit` values instead of `_validate_count()`. CLI argparse protects normal invocations, but raw socket clients or `py exec` callers can pass negative values and get Python negative-slice behavior.

Impact: malformed raw requests can silently return the wrong subset rather than a clean invalid-request error.

Suggested fix: apply `_validate_count()` consistently to every endpoint that supports `offset` or `limit`.

### 24. Text renderers can traceback on malformed nested bridge results

References:

- `src/bn/cli.py:510`
- `src/bn/formatters.py:64`
- `src/bn/formatters.py:107`
- `src/bn/formatters.py:945`

`_call()` invokes text renderers directly, and `main()` only catches `BridgeError`. Some renderers call `.get()` on nested fields without checking that those fields are dicts. A malformed bridge result can therefore produce a local traceback instead of a clean error or fallback rendering.

Impact: bridge/plugin version skew or unexpected result shapes can crash the CLI text path.

Suggested fix: make nested renderers defensive before `.get()` calls, render non-dict list elements with fallback text, and consider wrapping renderer failures in `BridgeError`.

### 25. `strings --min-length` accepts negative values

References:

- `src/bn/commands/misc.py:21`
- `src/bn/commands/misc.py:26`

`--min-length` uses raw `int`, so `bn strings --min-length -5` parses and effectively disables the filter because every real string length is greater than `-5`.

Suggested fix: use `_positive_int` or `_non_negative_int`.

### 26. CLI accepts negative context/argument values that are semantically invalid

References:

- `src/bn/commands/function.py:323`
- `src/bn/commands/function.py:375`
- `src/bn/commands/function.py:494`

`callsites --context`, `evidence function --context`, and `trace --arg` accept negative integers at parse time. The bridge rejects some later, but users get avoidable bridge-side errors.

Suggested fix: use `_non_negative_int` for these flags.

### 27. Protected free-form values still fail when they match known options

References:

- `src/bn/cli.py:679`
- `src/bn/cli.py:688`
- `src/bn/cli.py:707`
- `src/bn/cli.py:710`

The protected `--query` / `--code` rewrite only protects option-looking values that are unknown options or help flags. Values such as `--format` or `--target` are still treated as options instead of data.

Impact: `bn strings --query --format` or `bn py exec --code --target` cannot search/execute those literal strings despite the options being documented as free-form.

Suggested fix: for protected data options, rewrite any following token beginning with `-` to `--opt=value`.

## Additional Notes

- The existing test suite passed before report generation: `551 passed, 3 skipped`.
- A temporary clean wheel install was used to validate packaging behavior; temporary files were removed afterward.
- The review was intentionally bug/risk focused. It did not attempt to redesign architecture or verify every Binary Ninja API edge case against real large binaries.
