# bn reference — runtime

Target selection, sticky pins, instance/target resolution order, sessions/headless, output & spill, py exec, troubleshooting, quirks, install for the `bn` skill. See `../SKILL.md` for the map.

## 1. Workflow & target selection

1. Discover targets:

   ```bash
   bn target list
   ```

   The `[N]` prefix is the view id; you can pass `-t N`. If no bridge is running, any command auto-starts one.

2. Pick a target:
   - Single open BinaryView: omit `-t`.
   - Multiple open: pass `-t <selector>` from `bn target list`. Selectors match against `selector`, `target_id`, `view_id`, full filename, or basename.
   - `-t` / `--instance` work **before or after** the subcommand, and for two-level commands they are also accepted **between the group and the leaf**. Use a pre-subcommand form to disambiguate selectors that collide with subcommand names like `session` or `pam_qnx.so.2`:

     ```bash
     bn -t pam_qnx.so.2 decompile main      # at root
     bn decompile main -t pam_qnx.so.2      # after the leaf
     bn bundle -t pam_qnx.so.2 function main  # between group and leaf (two-level)
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

   > **HARD rule for parallel / fan-out agents.** Sticky pins are **one shared file per git repo** — every agent rooted in the same repo reads and writes the same `instance_id` / `target`. If multiple agents run concurrently against that repo, one agent's `bn instance use` / `bn target use` / `bn instance clear` / `bn target clear` silently changes the target for *all* of them, causing cross-talk and commands hitting the wrong binary. Parallel/fan-out agents **MUST** pass `-t/--target` and `--instance` explicitly on **every** command and **MUST NOT** call `instance use` / `target use` / `instance clear` / `target clear`. Prefer one dedicated headless instance per agent: `bn session start <binary> --instance-id <unique-id>`, then thread that `--instance <unique-id>` (and an explicit `-t`) through every subsequent call.

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
bn save /path/to/output.bndb             # explicit path (positional)
bn save --path /path/to/output.bndb      # --path is an accepted alias for the positional
```

> **Selector rebind after save.** `bn save` / `bn save <path>` rebinds the in-memory view's filename, so its basename / filename selector changes (e.g. `foo` becomes `foo.bndb`). A `-t foo` that worked before the save can stop resolving afterward. Post-save commands should target the **stable** `view_id` / `target_id` (the `[N]` prefix from `bn target list`), not the basename, to avoid `Unknown target selector` after a save.

`bn load <raw>` and `bn session start <raw> [...]` auto-prefer a sibling `<raw>.bndb` when one exists, so saved annotations come back without you having to retype the `.bndb` suffix. The CLI prints which file was actually opened:

```bash
$ bn load /path/to/foo.so
loaded: /path/to/foo.so.bndb
note: loaded /path/to/foo.so.bndb instead of /path/to/foo.so (use --no-bndb to skip)
```

Pass `--no-bndb` to force loading the raw binary even when a sibling `.bndb` exists. Passing a path that already ends in `.bndb` skips the lookup. The same `--no-bndb` flag works on `bn session start`.

`bn load` blocks until analysis completes (the bridge runs `update_analysis_and_wait()` and the CLI socket has no timeout). Plan for it on large binaries.

**Quick load (`--quick` / `--no-analysis`).** `bn load --quick` and `bn session start --quick` skip that analysis pass (~1s instead of waiting for the full function set), at the cost of a **capability boundary** — the container is parsed but the code is not yet analyzed:

- Ready immediately: `bn sections`, `bn imports`, the symbol table, `bn target list` / `bn target info` (flagged `[not analyzed]`, JSON `analysis_state: "quick"`).
- `bn strings` **errors** until `bn refresh` (it refuses with a "Strings are not available … Run `bn refresh`" directive rather than return an empty list that reads as "no strings").
- **Partial** until `bn refresh`: `bn function list` / `bn function search` (only entry-point + symbol functions exist pre-analysis; the count grows after refresh), and `bn decompile` / `bn il` / `bn disasm` / `bn xrefs` across the binary.

Run `bn refresh` once to promote the view to full analysis (`analysis_state` flips to `"full"`), or `bn decompile <fn> --force-analysis` to analyze a single function without the full pass. Branch on `analysis_state` rather than guessing from empty results. Loading a `.bndb` ignores `--quick` (the database already carries its analysis).

`--instance` is accepted on every subcommand (env `BN_INSTANCE`).

Requests time out after 600s by default so a wedged bridge can't hang the CLI; override with `BN_REQUEST_TIMEOUT=<seconds>` (`0` disables).

## 3. Output & context

Defaults:

- Read commands → `--format text`.
- Mutation, preview, setup, and export commands → `--format json`.
- `--format ndjson` is available where it makes sense.
- `--out <path>` writes the full body to disk and returns an envelope on stdout.

**Spill envelopes.** When output exceeds **10 000 estimated tokens** (~3 bytes/token heuristic), the body is written to disk and stdout carries a compact envelope; stderr carries a one-line warning. Envelope keys:

- `ok` — request status.
- `spilled` — `true` when the body was written to disk because of the threshold; `false` when `--out` was used.
- `path` (text envelope) / `artifact_path` (JSON) — location on disk: `<cache>/spills/YYYYMMDD/<stem>-HHMMSS-<pid>-<rand>.<json|ndjson|txt>` (cache dir defaults to `~/.cache/bn`, override with `BN_CACHE_DIR`).
- `format` — `json`, `ndjson`, or `text`.
- `bytes`, `tokens` (estimate), `tokenizer` (`estimate`), `sha256` — size + integrity.
- `summary` — shape hint with `kind` and `count` / `chars` / `keys`.

**Pipe trap (correctness).** When output spills, a downstream `grep`/`jq`/`awk`/`rg` reads only the small envelope, **not** the data — so a no-match silently reads as "absent" (e.g. `bn decompile <fn> | grep memcpy` finding nothing does *not* mean there's no `memcpy`). `bn` now prints an extra `note:` on stderr when stdout is a pipe and output spilled, but don't rely on noticing it. Instead, write to a file first and process that: `bn decompile <fn> --out /tmp/f.txt && grep memcpy /tmp/f.txt`, or slice with `--lines`/`--limit` so it doesn't spill.

Slicing knobs to avoid spilling in the first place:

```bash
bn decompile <fn> --lines 40:80         # 1-indexed inclusive; prints "// lines 40-80 of N"
bn xrefs <fn-or-addr> --limit 20        # cap text output
bn function info <fn>                    # compact by default
bn function info <fn> --verbose          # full params + locals
```

Pagination: `--limit` / `--offset` on list commands.


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

> **Exclusive write lock + unsandboxed.** `py exec` runs under the bridge's **exclusive write lock**, so a long-running snippet blocks **every** other op (reads and writes) on a shared bridge until it returns — don't park slow scripts on a bridge other agents are using. It also runs **unsandboxed** with full `bv` / `binaryninja` access (it can mutate or write to disk). Keep snippets short on shared bridges, and for raw byte reads prefer the dedicated `bn read` (read-locked, parallel-safe) instead of a `py exec` that calls `bv.read(...)`.

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
