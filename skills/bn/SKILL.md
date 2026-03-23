---
name: bn
description: Use the local bn CLI for Binary Ninja reversing work through the bn bridge. Works with both a live GUI session and headless mode. Prefer this skill for decompilation, function search, callsite recovery, IL/disassembly, xrefs, type inspection, struct field edits, previewed mutations, and inline Python execution.
---

# bn

Use this skill when the user wants reverse-engineering work against a Binary Ninja database and the local `bn` CLI is available. The bridge can run in GUI mode (attached to an open Binary Ninja window) or headless mode (no GUI required).

## Workflow

1. Start with target discovery:

```bash
bn target list
bn doctor
```

Use `bn doctor` when bridge state is unclear or `bn target list` does not show what you expect.

2. Pick a target:
- If there is exactly one open BinaryView, target-scoped commands can omit `--target` entirely.
- If multiple targets are open, commands that omit `--target` fail; pass `--target <selector>` from `bn target list`.
- Use `--target active` only when you explicitly mean the GUI-selected target.

3. Pick the right output mode:
- Read commands default to `text`.
- Mutation, preview, setup, and export commands default to `json`.
- Other options: `--format json`, `--format ndjson`, `--out <path>`.

Outputs above `10_000` `o200k_base` tokens auto-spill to disk. When that happens, stdout is empty and stderr carries the spill metadata as plain text, so do not chain `bn ... | rg ...` and expect to search the real output. Use `--out <path>` when you want the full body written to a known file.

## Headless Mode

When no GUI is available, use `bn session start` to launch a headless bridge:

```bash
bn session start /path/to/binary.bndb
```

Or start with no binaries and load them later:

```bash
bn session start
bn load /path/to/binary.bndb
```

If no bridge is running, `bn` auto-starts one on first use — no manual setup needed.

Close binaries with `bn close [path]` (omit path to close all). All other commands work identically in headless and GUI modes.

## High-Value Read Commands

```bash
bn target list
bn target info
bn function list
bn function list --min-address 0x401000 --max-address 0x40ffff
bn function search attachment
bn function search --regex 'attach|detach|follow'
bn function info sample_track_floor_height_at_position
bn callsites crt_rand --within bonus_pick_random_type
bn callsites crt_rand --within-file /tmp/rng-functions.txt --format ndjson
bn proto get sample_track_floor_height_at_position
bn local list sample_track_floor_height_at_position
bn decompile sample_track_floor_height_at_position
bn decompile sample_track_floor_height_at_position --addresses
bn il sample_track_floor_height_at_position
bn disasm sample_track_floor_height_at_position
bn xrefs sample_track_floor_height_at_position
bn xrefs field TrackRowCell.tile_type
bn comment get --address 0x401000
bn types --query Player
bn types show Player
bn struct show Player
bn strings --query follow
bn imports
```

`bn function search` is case-insensitive substring matching by default. Add `--regex` when you need regular expressions. `bn function list` and `bn function search` both accept `--min-address` and `--max-address`.

`bn decompile` omits address prefixes by default for cleaner output. Add `--addresses` when you need per-line addresses (e.g., for `bn comment set --address`).

## Caller-Static Mapping

Prefer `bn callsites` over ad hoc `py exec` when the task is "find exact native RNG return-address callers" or any similar direct-call mapping workflow.

`bn callsites` reports both:
- `call_addr`: the native `call ...` instruction address
- `caller_static`: the exact post-call return address

The key rule is:
- `caller_static = call_addr + instruction_length`

Use it like this:

```bash
bn callsites crt_rand --within bonus_pick_random_type --caller-static
bn callsites crt_rand --within fx_queue_add_random --caller-static
bn callsites crt_rand --within-file /tmp/rng-functions.txt --format json
```

The `--within-file` format is one function identifier per non-empty line. Lines beginning with `#` are ignored.

For close-together callsites, `bn callsites` also returns:
- previous instructions
- next instructions
- `call_index` within the containing function
- `within_query` with the original unresolved scope token
- a local-or-null HLIL statement
- a best-effort `pre_branch_condition`

`hlil_statement` is intentionally local-or-null. If Binary Ninja only exposes a coarse enclosing region instead of the smallest call-containing expression or statement, expect `hlil_statement: null` rather than a noisy whole-function blob.

`pre_branch_condition` means the nearest enclosing pre-call HLIL condition when it can be recovered confidently. It is not a generic "related branch" field, so `null` is normal when the condition cannot be derived cleanly.

Use `bn xrefs` when you only need inbound references. Use `bn callsites` when you need exact return-address recovery and local context around the call.

## Bundles

Use bundles when you want a reusable artifact instead of pasting long output into context:

```bash
bn bundle function sample_track_floor_height_at_position --out /tmp/floor.json
```

With `--out`, the CLI returns a JSON envelope for the written artifact instead of dumping the whole bundle to stdout.

## Python Escape Hatch

Use inline Python as a normal lane for one-off Binary Ninja inspection that is awkward to express as a built-in command:

```bash
bn py exec --code "print(hex(bv.entry_point)); result = {'functions': len(list(bv.functions))}"
```

Use `--stdin` with a quoted heredoc for multiline Python snippets:

Shell details matter here:
- Quote the heredoc delimiter as `<<'PY'` so the shell does not expand `$vars`, backticks, or backslashes before Binary Ninja sees the Python.
- Keep the closing `PY` on its own line with no indentation or trailing spaces.
- Use `--script <file>` only for real files you want to keep on disk.
- Use `--code` for true one-liners only.
- If you are counting or collecting BN iterators such as `f.hlil.instructions`, materialize them explicitly with `list(...)` or a generator consumption pattern instead of assuming random-access behavior.

Use this pattern for larger inspection snippets:

```bash
bn py exec --stdin <<'PY'
out = []
for f in bv.functions:
    if 0x416000 <= f.start < 0x41C000:
        out.append((f.start, f.symbol.short_name))
out.sort()
print("\n".join(f"{addr:#x} {name}" for addr, name in out))
PY
```

The `py exec` environment includes:`bn`, `binaryninja`, `bv`, `result`.

`py exec` always returns `stdout` and `result`. If `result` is not JSON-serializable, the CLI returns `repr(result)` plus a warning instead of silently flattening it.

## Additional Read Commands

```bash
bn il sub_401000 --view mlil              # MLIL instead of default HLIL
bn il sub_401000 --view llil --ssa        # LLIL in SSA form
bn local retype func_name var_name "uint32_t*"  # retype a local variable
bn comment delete --address 0x401000      # remove a comment
bn struct field rename Player 0x308 new_name    # rename a field
bn struct field delete Player 0x308             # delete a field
```

## Batch Apply

For bulk mutations (renames, retypes, comments), use `bn batch apply` with a JSON manifest. This is significantly faster than individual commands.

Manifest format:

```json
{
  "target": "active",
  "ops": [
    {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "player_update"},
    {"op": "rename_symbol", "identifier": "sub_402000", "new_name": "player_init"},
    {"op": "rename_symbol", "identifier": "sub_403000", "new_name": "player_destroy"}
  ]
}
```

```bash
bn batch apply /tmp/manifest.json            # apply live
bn batch apply /tmp/manifest.json --preview  # preview diffs first
```

Key details:
- The manifest MUST be a dict with an `"ops"` key (not a bare list)
- Include `"target"` in the manifest or it will fail with "Unknown target selector: None"
- All ops are verified and the entire batch is reverted if any op fails
- Supports `--preview` to see diffs without committing

## Mutation Workflow

Prefer preview first:

```bash
bn types declare "typedef struct Player { int hp; } Player;" --preview
bn types declare --file /path/to/win32_min.h --preview
bn struct field set Player 0x308 movement_flag_selector uint32_t --preview
bn symbol rename sub_401000 player_update --preview
bn proto get sub_401000
bn local list sub_401000
bn proto set sub_401000 "int __cdecl player_update(Player* self)" --preview
```

Preview mode applies the change, refreshes analysis, captures affected decompile diffs, and then reverts the mutation.

For struct previews, inspect:`results`, `affected_types`, `affected_functions`.

For the first few changed functions, `affected_functions` may also include `before_excerpt` and `after_excerpt` HLIL snippets around the first changed lines.

If a struct edit is already identical, preview may report `changed: false` with `No effective change detected`.

`bn types declare` uses Binary Ninja's source parser when available. With `--file`, it forwards the real source path so relative includes work like GUI header import.

If a declaration only introduces functions or extern variables and no named types, `types declare` now reports a no-op instead of failing with `No named types found in declaration`.

Non-preview writes are live-verified by default. If the requested state does not read back from Binary Ninja, the command exits nonzero and the whole mutation or batch is reverted.

After any live type or prototype mutation, do an explicit readback:

```bash
bn proto get sub_401000
bn struct show Player
bn types show Player
bn decompile sub_401000
```

Key result statuses:
- `verified`
- `noop`
- `unsupported`
- `verification_failed`

When verification fails, JSON output also includes the requested and observed state for the failed operation.

If you need to force BN to recalculate presentation after a type change, run:

```bash
bn refresh
```

## Saving and Loading Databases

**Always save work as `.bndb`** before closing a target. Annotations (renames, types, comments) are lost if a raw binary is closed without saving.

```bash
bn save                          # saves to <filename>.bndb
bn save /path/to/output.bndb    # saves to explicit path
```

**Loading behavior**: `bn load foo.so` does NOT auto-detect a sibling `foo.so.bndb`. It re-analyzes from scratch, discarding all prior annotations. Always load the `.bndb` path explicitly:

```bash
# Wrong — ignores existing .bndb, re-analyzes from scratch
bn load /path/to/foo.so

# Correct — loads saved annotations
bn load /path/to/foo.so.bndb
```

Before loading a binary, check if a `.bndb` already exists and prefer it.

## Load Timeout

`bn load` has a ~30-second default timeout. Large binaries may exceed this — the command exits with a timeout error but analysis continues in the background. Check back with:

```bash
bn target list
```

If the target appears, it loaded successfully despite the timeout.

## Multi-Instance / Parallel Agents

When multiple agents need to analyze different binaries concurrently, each agent should start its own dedicated bridge session to avoid conflicts:

```bash
# Start a session (optionally preload a binary)
bn session start /path/to/binary.bndb
# → {"instance_id": "a3f1bc09", "pid": 12345, "socket_path": "..."}

# Route all subsequent commands to that session
bn --instance a3f1bc09 target list
bn --instance a3f1bc09 decompile main
bn --instance a3f1bc09 save

# Or set the env var once
export BN_INSTANCE=a3f1bc09
bn decompile main

# Stop when done
bn session stop a3f1bc09
```

Session management commands:
- `bn session start [binary...]` — start a new headless bridge, return instance ID
- `bn session list` — show all running bridge instances
- `bn session stop <id>` — shut down a specific instance

If no bridge is running and no `--instance` is specified, `bn` auto-starts a headless bridge. For parallel workflows, prefer explicit `bn session start` so each agent controls its own session lifecycle.

## Known Quirks

- **`bn local rename` and `bn comment set` return null status**: The operations succeed but the JSON response shows `null` for status instead of `verified`. Verify by re-decompiling.
- **`types declare` verification failures**: `bn types declare` may fail with `verification_failed` and roll back, even for correct declarations. Workaround: define structs directly via `bn py exec` using `bntypes.StructureBuilder`:

```bash
bn py exec --stdin <<'PY'
from binaryninja import types as bntypes
s = bntypes.StructureBuilder.create()
s.append(bntypes.Type.pointer(bv.arch, bntypes.Type.void()), "vtable")
s.append(bntypes.Type.array(bntypes.Type.int(1, sign=False), 0x20), "pad_04")
s.append(bntypes.Type.int(4, sign=False), "m_bLoad")
s.append(bntypes.Type.pointer(bv.arch, bntypes.Type.int(1, sign=False)), "m_fileBuf")
s.append(bntypes.Type.int(4, sign=False), "m_fileBufSize")
bv.define_user_type("MyStruct", bntypes.Type.structure_type(s))
print("defined MyStruct")
PY
```

- **Stale bridge**: If `bn doctor` reports `stale: loaded plugin code does not match installed plugin file`, restart Binary Ninja (GUI or headless) to pick up the updated bridge code. Commands will behave unpredictably with stale code.
- **No targets = no `py exec`**: `bn py exec` requires at least one open BinaryView. If `bn load` timed out and the target isn't ready yet, `py exec` will fail with "No BinaryView targets are open".
