---
name: bn
description: Use the local bn CLI for Binary Ninja reversing through the bn bridge, in a live GUI session or headless mode. Triggers include decompilation, function search, xrefs, callsites and exact caller_static mapping, IL/disassembly, type recovery, struct field edits, previewed mutations, stable local IDs, batch apply, BNDB save/load, and inline BN Python.
---

# bn

Use this skill when the user wants reverse-engineering work against a Binary Ninja database and the local `bn` CLI is available. The bridge runs as a GUI plugin (attached to an open Binary Ninja window) or as a headless process. The CLI auto-spawns a headless instance on first use if none is running.

> **Methodology skills:** for structured RE workflows see `bn-re`; for vulnerability research see `bn-vr`. Both delegate command syntax back here.

## 1. Workflow & target selection

1. Discover targets:

   ```bash
   bn target list
   ```

   The `[N]` prefix is the view id; you can pass `-t N`. If no bridge is running, any command auto-starts one.

2. Pick a target:
   - Single open BinaryView: omit `-t`.
   - Multiple open: pass `-t <selector>` from `bn target list`. Selectors match against `selector`, `target_id`, `view_id`, full filename, or basename.
   - `-t` works **before or after** the subcommand. Use the pre-subcommand form to disambiguate selectors that collide with subcommand names like `session` or `pam_qnx.so.2`:

     ```bash
     bn -t pam_qnx.so.2 decompile main
     bn decompile main -t pam_qnx.so.2
     ```

   - Use `-t active` only when you explicitly want to follow the GUI selection.

3. (Optional) Pin sticky defaults — useful when you'll run many commands against the same instance/target:

   ```bash
   bn instance use <id>          # pin --instance for this project
   bn target use <selector>       # pin -t for this project
   bn instance clear              # clear pinned instance
   bn target clear                # clear pinned target
   ```

   Resolution order:
   - **Instance:** CLI `--instance` > env `BN_INSTANCE` > sticky > auto-pick / auto-spawn.
   - **Target:** CLI `-t/--target` > sticky > single-open auto-pick. **`BN_TARGET` does not exist** — target selection is the CLI flag or `bn target use`, nothing else.

   State lives at `~/.cache/bn/sessions/<sha256(project_root)[:16]>.json`. Project root walks up to the nearest `.git` (cwd as fallback). `bn session list` and `bn target list` mark matching entries with `[sticky]`. When a sticky instance points at a dead bridge, errors append `Clear it with bn instance clear`.

## 2. Sessions & headless

The bridge runs as a GUI plugin or as a headless process; both speak the same protocol.

```bash
bn load /path/to/binary.bndb           # auto-spawns a headless bridge if none is running
bn session start /path/to/binary [--instance-id <id>]
bn session list                         # running instances + RSS + sticky marker
bn session stop <id>                    # shut one down
bn close [<path>]                       # close one (omit path → close all)
```

`bn close` reports each closed view as `{path, unsaved}`. If a view had unsaved mutations, stdout warns — run `bn save` *first* if you care about annotations:

```bash
bn save                                  # saves to <filename>.bndb
bn save /path/to/output.bndb             # explicit path
```

`bn load <raw>` and `bn session start <raw> [...]` auto-prefer a sibling `<raw>.bndb` when one exists, so saved annotations come back without you having to retype the `.bndb` suffix. The CLI prints which file was actually opened:

```bash
$ bn load /path/to/foo.so
loaded: /path/to/foo.so.bndb
note: loaded /path/to/foo.so.bndb instead of /path/to/foo.so (use --no-bndb to skip)
```

Pass `--no-bndb` to force loading the raw binary even when a sibling `.bndb` exists. Passing a path that already ends in `.bndb` skips the lookup. The same `--no-bndb` flag works on `bn session start`.

`bn load` blocks until analysis completes (the bridge runs `update_analysis_and_wait()` and the CLI socket has no timeout). Plan for it on large binaries.

`--instance` is accepted on every subcommand (env `BN_INSTANCE`).

## 3. Output & context

Defaults:

- Read commands → `--format text`.
- Mutation, preview, setup, and export commands → `--format json`.
- `--format ndjson` is available where it makes sense.
- `--out <path>` writes the full body to disk and returns an envelope on stdout.

**Spill envelopes.** When output exceeds **10 000 `o200k_base` tokens**, the body is written to disk and stdout carries a compact envelope; stderr carries a one-line warning. Envelope keys:

- `ok` — request status.
- `spilled` — `true` when the body was written to disk because of the threshold; `false` when `--out` was used.
- `path` (text envelope) / `artifact_path` (JSON) — location on disk: `<tempdir>/bn-spills/YYYYMMDD/<stem>-HHMMSS.<json|ndjson|txt>`.
- `format` — `json`, `ndjson`, or `text`.
- `bytes`, `tokens`, `tokenizer` (`o200k_base`), `sha256` — size + integrity.
- `summary` — shape hint with `kind` and `count` / `chars` / `keys`.

Don't pipe `bn ... | rg ...` and expect `rg` to see real content when output spills — read the artifact path instead.

Slicing knobs to avoid spilling in the first place:

```bash
bn decompile <fn> --lines 40:80         # 1-indexed inclusive; prints "// lines 40-80 of N"
bn xrefs <fn-or-addr> --limit 20        # cap text output
bn function info <fn>                    # compact by default
bn function info <fn> --verbose          # full params + locals
```

Pagination: `--limit` / `--offset` on list commands.

## 4. Read flow

```bash
bn target info
bn function list [--min-address 0x401000 --max-address 0x40ffff]
bn function search attachment
bn function search --regex 'attach|detach|follow'
bn function info <fn> [--verbose]
bn decompile <fn> [--addresses] [--lines 40:80]
bn il <fn> [--view {hlil|mlil|llil}] [--ssa]
bn disasm <fn>
bn xrefs <fn-or-addr> [--limit 20]
bn xrefs --field <Struct.field>
bn callsites <callee> --within <fn>
bn callsites <callee> --within-file <path>
bn proto get <fn>
bn local list <fn>
bn types [--query <q>]
bn types show <name>
bn struct show <name>
bn strings [--query <q>] [--min-length 5] [--section .rodata] [--no-crt]
bn imports
bn sections [--query <q>]
bn comment list [--query <q>]
bn comment get   --address 0x... | --function <fn>
```

Notes:

- `bn function search` is case-insensitive substring; add `--regex` for regular expressions. `function list` and `function search` both accept `--min-address` / `--max-address`.
- `bn xrefs` accepts a function name *or* a hex/decimal address. Text groups refs by caller (`code refs: 12 sites across 4 functions`); JSON adds `caller_function: {address, name}` so an `xrefs → --within-file` pipeline survives duplicate symbol names. Use `bn xrefs` for inbound references; reach for `bn callsites` when you need exact return-address recovery and local context.
- `bn decompile` omits address prefixes by default. Add `--addresses` when you need them (e.g. for `bn comment set --address`).
- `bn imports` JSON tags each entry with `kind` (`function`, `data`, `address`) and includes `library` + `raw_name`. Text marks data/address imports with `(data)` / `(address)`.
- `bn sections` exposes start/end, length, semantics, and segment-derived `r/w/x` permission flags.
- `bn strings`: `--no-crt` is a heuristic — drops single-character repetitions and strings sitting in `.text`. Combine with `--min-length` and `--section`.

## 5. Caller-static mapping

Prefer `bn callsites` over ad-hoc `py exec` whenever the task is "find the exact native return-address callers" or any direct-call mapping workflow.

`bn callsites` reports:

- `call_addr` — the native `call ...` instruction address.
- `caller_static` — the post-call return address (`call_addr + instruction_length`).
- `call_index` within the containing function, `within_query`, previous/next instructions, a local-or-null `hlil_statement`, and a best-effort `pre_branch_condition`.

```bash
bn callsites crt_rand --within bonus_pick_random_type --caller-static
bn callsites crt_rand --within fx_queue_add_random   --caller-static
bn callsites crt_rand --within-file /tmp/rng-functions.txt --format json
```

`--within-file` accepts one identifier (name or hex address) per non-empty line; lines beginning with `#` are ignored.

`hlil_statement` is intentionally local-or-null — when Binary Ninja only exposes a coarse enclosing region, expect `null` instead of a noisy whole-function blob. `pre_branch_condition` is the nearest enclosing pre-call HLIL condition when it can be recovered confidently; `null` is normal.

If you call `bn callsites <callee>` without `--within` / `--within-file`, the CLI prints a 3-option help block (`single caller`, `many callers`, `list callers`) instead of erroring.

## 6. Mutation flow

The mutation surface is built around a four-step safety loop: **preview → live-verify → read back → save**.

### Step 1 — preview first

```bash
bn types declare "typedef struct Player { int hp; } Player;" --preview
bn types declare --file /path/to/win32_min.h --preview
bn struct field set Player 0x308 movement_flag_selector uint32_t --preview
bn symbol rename sub_401000 player_update --preview
bn proto set sub_401000 "int __cdecl player_update(Player* self)" --preview
bn comment set --address 0x401000 "explain this" --preview
```

Preview applies → refreshes analysis → captures decompile diffs → reverts. Inspect:

- `results` — per-op outcome and observed state.
- `affected_types` — type-level layout diffs.
- `affected_functions` — for the first few changed functions, also includes `before_excerpt` / `after_excerpt` HLIL snippets near the first change.

A no-op edit reports `changed: false` ("No effective change detected").

### Step 2 — live writes are verified

Per-op statuses:

- `verified` — change applied and read back as requested.
- `noop` — already in the requested state.
- `unsupported` — operation not supported on this object.
- `verification_failed` — readback disagrees; the whole mutation/batch is reverted, and JSON also returns the requested vs observed state.

### Step 3 — read back

```bash
bn proto get <fn>
bn struct show <name>
bn types show <name>
bn decompile <fn>
bn refresh                                # if BN still shows stale presentation
```

### Locals — prefer `local_id` over names

```bash
bn local list <fn>
bn local rename <fn> <local_id|name> <new_name>
bn local retype <fn> <local_id|name> <new_type>
```

`bn local list` text output splits params and locals into compact `name  type` rows. JSON entries carry `name`, `type`, `storage`, `index`, `identifier`, `source_type`, `is_parameter`, and **`local_id`** — a stable handle that survives re-analysis. Reach for `local_id` whenever Binary Ninja might rebuild the variable list.

### Comments

```bash
bn comment set --address 0x401000 "explain this"
bn comment set --function player_update "explain this"
bn comment delete --address 0x401000
bn comment delete --function player_update
```

### Struct field edits

```bash
bn struct field set Player 0x308 flags uint32_t [--no-overwrite]
bn struct field rename Player old_name new_name
bn struct field delete Player <field_name>     # NOTE: takes the field name, not an offset
```

### Bulk mutations — batch manifest

For large rename/retype/comment runs, use `bn batch apply` with a JSON manifest. Significantly faster than firing individual commands.

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
bn batch apply /tmp/manifest.json
bn batch apply /tmp/manifest.json --preview
```

Rules:

- The manifest must be a dict with an `"ops"` key (not a bare list).
- Include `"target"` in the manifest or it fails with `Unknown target selector: None`.
- All ops are verified — a single failure reverts the entire batch.
- `--preview` shows diffs without committing.

### Step 4 — save before close

Annotations live in the `.bndb`. Always save before closing — `bn close` warns when unsaved mutations are about to be discarded (see §2).

## 7. Bundles

Use bundles when you want a reusable artifact instead of pasting long output into context:

```bash
bn bundle function sample_track_floor_height_at_position --out /tmp/floor.json
```

With `--out`, the CLI returns a JSON envelope for the written artifact instead of dumping the bundle to stdout.

## 8. Python escape hatch

Reach for `bn py exec` only when built-in commands are awkward — arbitrary BinaryView introspection or operations the bridge does not expose. Built-ins are preferred because they are verified, cache-friendly, and integrate with the preview/verify loop.

```bash
bn py exec --code "print(hex(bv.entry_point)); result = {'functions': len(list(bv.functions))}"
```

Multiline snippets via stdin with a quoted heredoc:

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

Shell rules:

- Quote the delimiter as `<<'PY'` so the shell does not expand `$vars`, backticks, or backslashes before Binary Ninja sees the Python.
- Keep the closing `PY` on its own line with no indentation or trailing whitespace.
- `--script <file>` for code on disk; `--code` for true one-liners.
- Materialize Binary Ninja iterators (`f.hlil.instructions`, etc.) with `list(...)` instead of assuming random-access behavior.

The exec environment includes `bn`, `binaryninja`, `bv`, and `result`.

`py exec` always returns `stdout` and `result`. `result` is JSON-serialized when possible; if not, the CLI returns `repr(result)` and a non-fatal entry in `warnings`. If your script writes a JSON artifact, it is surfaced under `artifact`.

## 9. Troubleshooting

Run `bn doctor` only when something is wrong — commands fail unexpectedly, targets don't appear, or the bridge seems unresponsive:

```bash
bn doctor
```

It checks CLI version, plugin staleness (`stale_plugin_version`, `stale_plugin_code`), and instance connectivity. Don't run it as part of normal workflow.

## 10. Known quirks

- **`types declare` verification failures.** The source-parser path handles most declarations, but a stubborn one may roll back with `verification_failed`. Workaround: define the struct directly via `bn py exec` using `StructureBuilder`, then re-run `bn types show`:

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

- **Stale bridge.** If `bn doctor` reports `stale: loaded plugin code does not match installed plugin file`, restart Binary Ninja (GUI or headless) to pick up the updated bridge. Commands behave unpredictably with stale code.

- **No targets ⇒ no `py exec`.** `bn py exec` requires at least one open BinaryView. If `bn load` is still running or the target isn't ready yet, `py exec` errors with "No BinaryView targets are open".

## 11. Skill install

`bn skill install` is idempotent. It links/copies the bundled skills into `~/.claude/skills/` and, when `~/.codex/` exists, also into `~/.codex/skills/`. Honors `CLAUDE_HOME` / `CODEX_HOME`. Use `--mode copy` for standalone copies, `--dest <path>` for a single explicit destination, and `--force` to overwrite. Restart your agent to pick up renamed or newly added skills.
