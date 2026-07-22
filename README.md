# bn

`bn` is a coding agent-first CLI for Binary Ninja. It gives a shell session or tool-calling agent stable commands, structured output, and access to the same live Binary Ninja database you already have open in the GUI.

## Headline Features

- Query live Binary Ninja state from the shell: targets, functions, callsites, decompile text, IL, disassembly, xrefs, types, strings, imports, and reusable bundles.
- Execute Python inside the Binary Ninja process instead of maintaining a separate headless workflow.
- Apply mutations with `--preview`, capture decompile diffs, and verify the live post-state before reporting success.
- Emit structured `json` or `ndjson` output, auto-spill large results to files, and return token counts so agents can budget context intelligently.

## Install

Recommended setup: install the CLI, the Binary Ninja companion plugin, and the bundled agent skills.

Install the CLI on your PATH:

```bash
uv tool install -e .
```

Install the Binary Ninja companion plugin:

```bash
bn plugin install
```

That links [`src/bn_agent_bridge`](src/bn_agent_bridge) into your Binary Ninja plugins directory.

Install the bundled Claude Code/Codex skills:

```bash
bn skill install
```

That symlinks the bundled skills into `~/.claude/skills/` by default. If `~/.codex/` exists, it also installs them into `~/.codex/skills/`. Use `--mode copy` if you want standalone copies instead. Restart your agent to pick up a new or renamed skill.

If the plugin code changes, reload Binary Ninja Python plugins or restart Binary Ninja.

## How It Works

- `bn` has two parts:
  - a normal Python CLI that you can run from your shell or agent tool harness
  - a Binary Ninja bridge that owns all Binary Ninja API access
- The bridge runs either as a GUI plugin or as a standalone headless process.
- The GUI plugin uses a single fixed bridge socket and registry file. Headless bridges each get their own socket and registry under `~/.cache/bn/instances/`, so you can run multiple in parallel.
- The CLI discovers a bridge, connects to it, and forwards commands. With multiple bridges open, pick one with `-i/--instance <id>` or pin it with `bn instance use` (single shell only — parallel agents must pass `-i` every time).
- In GUI mode, the bridge runs as a plugin and works with a personal license.
- In headless mode, the bridge runs standalone and requires the `binaryninja` Python package on `sys.path` (commercial headless license or the headless API).

## Quick Start (GUI)

Open a binary or `.bndb` in Binary Ninja, then run:

```bash
bn doctor
bn target list
bn refresh
bn function list
bn decompile sub_401000
```

## Quick Start (Headless)

Start a managed headless session and optionally preload binaries:

```bash
bn session start /path/to/binary.bndb
bn function list
bn session list
bn session stop <instance>
```

`bn session start` spawns a `bn-agent` process, registers it under `~/.cache/bn/instances/<id>.{json,sock}`, and prints the instance ID. Use `bn session list` to see what's running and `bn session stop <id>` to shut one down. You can also start a bridge directly with `python -m bn_agent_bridge /path/to/binary.bndb` if you want to manage the process yourself.

You can run several sessions in parallel. With more than one instance up, either pass **`-i/--instance <id>`** per call (prefer the short form) or pin one for a single shell with:

```bash
bn -i <id> decompile main          # explicit per call (required for parallel agents)
bn instance use <id>               # sticky pin — single shell only; not for multi-agent
bn instance clear
```

Spawn naming is separate from routing: `bn session start /path/to/bin --instance-id myid` names the new bridge; subsequent commands use `bn -i myid …` to talk to it.

`bn load` and `bn close` are available for dynamic binary management (useful in headless, but also work with the GUI bridge). All commands work identically in GUI and headless mode.

If exactly one BinaryView is open, target-specific commands can omit `--target` entirely. If multiple targets are open, pass `--target <selector>` from `bn target list`, or pin one with `bn target use <selector>` (and `bn target clear` to undo).

## Quick Load

`bn load --quick` (alias `--no-analysis`) and `bn session start --quick` skip Binary Ninja's full analysis pass (`update_analysis_and_wait`), so a binary is ready in ~1s instead of after the whole function set has been recovered. The trade-off is a **capability boundary**: a quick-loaded view has its container parsed but its code not yet analyzed, so some commands refuse outright and others return a partial result until you refresh.

Ready immediately after `--quick`:

- `bn sections`
- `bn imports`
- the symbol table (and `bn rename` against named symbols)
- `bn target list` / `bn target info` (the view is flagged `[not analyzed]`)

Need `bn refresh` first — these depend on the analysis pass:

- `bn strings` **errors** until you refresh — it refuses with "Strings are not available: this target was loaded with --quick (no analysis). Run bn refresh …" rather than return an empty list that looks like "this binary has no strings".
- `bn function list` / `bn function search` return a **partial** set — only entry-point and symbol functions exist before analysis; the count grows once the analysis pass recovers the rest.
- `bn decompile` / `bn il` / `bn disasm` / `bn xrefs` across the binary are partial for the same reason (functions BN hasn't discovered yet aren't there to read).

Run `bn refresh` once to promote a quick view to full analysis:

```bash
bn load /path/to/firmware.bin --quick   # fast: container ready, code not analyzed yet
bn sections                             # works now
bn strings                              # errors: "Strings are not available … Run bn refresh"
bn refresh                              # runs update_analysis_and_wait()
bn strings                              # now returns the full string set
```

Single-function escape hatch — analyze just one function without a full refresh:

```bash
bn decompile handle_request --force-analysis
```

Loading a `.bndb` ignores `--quick`: a database already carries its saved analysis, so the flag is a no-op there.

### Detecting a quick view programmatically

`bn target list` / `bn target info --format json` carry an `analysis_state` field (and the matching `analyzed` boolean), so automation can branch on it instead of inferring the state from a `strings` error or a short function list:

```json
{
  "target_id": "firmware.bin@0",
  "selector": "firmware.bin",
  "basename": "firmware.bin",
  "analyzed": false,
  "analysis_state": "quick"
}
```

`analysis_state` is `"quick"` before analysis and `"full"` after `bn refresh`. A consumer that sees `"quick"` should `bn refresh` before relying on `strings` or the full function set.

## Target Selection

Use `bn target list` to see available targets:

```bash
bn target list
```

Targets can be selected with:

- the `selector` field from `bn target list`
- the full `target_id`
- the BinaryView basename
- the full filename
- the view id
- `active` when you explicitly want the GUI-selected target

In normal use, prefer the `selector` field. For a single open database, this is usually just the `.bndb` basename:

```bash
bn decompile update_player_movement_flags --target SnailMail_unwrapped.exe.bndb
```

Omitting `--target` only works when exactly one target is open. If multiple targets are open, the CLI rejects the command instead of silently falling back to `active`.

## Output Behavior

Every command supports:

- `--format json`
- `--format text`
- `--format ndjson`
- `--out <path>`

Interactive read commands default to `text`. Mutation, setup, and export commands default to `json`.
Add `--format json` when you need stable fields for automation or piping into structured tooling.

Examples:

```bash
bn function list --format ndjson
bn function list --min-address 0x401000 --max-address 0x40ffff
bn function search --regex 'attach|detach'
bn decompile sample_track_floor_height_at_position --out /tmp/floor.json
```

If `--out` is set, the command writes the rendered result to that path and prints a compact text envelope with the artifact path, byte size, token count, tokenizer, hash, and summary. Agents can use that envelope to decide whether to read the full artifact, keep a summary, or defer loading it into context.

Example artifact envelope:

```text
ok: true
spilled: false
path: /tmp/floor.json
format: json
bytes: 1234
tokens: 456
tokenizer: estimate
sha256: deadbeef...
summary: kind=object count=3
```

The only exception is `bn bundle function`, which writes the bundle artifact from inside the bridge and prints the envelope back to the CLI.

`bn function list` and `bn function search` return the full matching set for the selected target or address range. Large results auto-spill to an artifact instead of forcing manual pagination. Spill is token-based and currently triggers above 10,000 tokens. When that happens, stdout contains the compact artifact envelope and stderr carries a short warning with the artifact path.

## Extraction Commands

Common read-only commands:

```bash
bn target list
bn target info

bn function list
bn function list --min-address 0x401000 --max-address 0x40ffff
bn function search attachment
bn function search --regex 'attach|detach|follow'
bn function info end_track_attachment_follow_state
bn callsites crt_rand --within bonus_pick_random_type
bn callsites crt_rand --within-file /tmp/rng-functions.txt --format ndjson
bn proto get end_track_attachment_follow_state
bn local list end_track_attachment_follow_state
bn refresh

bn decompile end_track_attachment_follow_state
bn il end_track_attachment_follow_state
bn disasm end_track_attachment_follow_state
bn xrefs end_track_attachment_follow_state
bn xrefs field TrackRowCell.tile_type
bn comment get --address 0x401000

bn types --query Player
bn types show Player
bn struct show Player
bn types declare --file /path/to/win32_min.h --preview
bn strings --query follow
bn imports
```

`bn function search` stays case-insensitive substring matching by default. Add `--regex` when you need regular expressions. `bn function list` and `bn function search` both accept `--min-address` and `--max-address` to filter by function start address.

`bn callsites` is the direct-call lane for exact return-address recovery. It reports both the native `call_addr` and the post-call `caller_static`, where `caller_static = call_addr + instruction_length`. Scope it with `--within <function>` or `--within-file <path>`; the file format is one function identifier per non-empty line, with `#` comments ignored.

Each callsite row also includes:

- `call_index`: zero-based ordinal for matching callsites in the containing function, ordered by `call_addr`
- `within_query`: the original unresolved scope token from `--within` or `--within-file`
- `hlil_statement`: the smallest recoverable HLIL expression or statement containing the call, or `null` when Binary Ninja only exposes a coarse enclosing region
- `pre_branch_condition`: the nearest enclosing pre-call HLIL condition when it can be recovered confidently, otherwise `null`

`hlil_statement` is intentionally local-or-null. If the best available HLIL mapping expands to a broad whole-function or multi-statement blob, `bn callsites` suppresses it instead of returning noisy context.

## Bundles And Python

`bn decompile` is the HLIL-text convenience lane. It is useful for quick function reading, but typed layouts remain authoritative in `bn types show` and `bn struct show`.

Export a reusable function bundle:

```bash
bn bundle function end_track_attachment_follow_state --out /tmp/end_track_attachment_follow_state.json
```

Run Python inside the Binary Ninja process for one-off inspection and BN-native scripting:

```bash
bn py exec --code "print(hex(bv.entry_point)); result = {'functions': len(list(bv.functions))}"

bn py exec --stdin <<'PY'
print(hex(bv.entry_point))
result = {"functions": len(list(bv.functions))}
PY
```

Use `--stdin` or `--script` for multiline Python snippets. Use `--code` for true one-liners only.

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

Use a quoted heredoc for multiline Python snippets.

When you need counts from BN iterators such as `f.hlil.instructions`, materialize them explicitly with `list(...)` or consume them with `sum(1 for ...)` instead of assuming sequence semantics.

The `py exec` environment includes:

- `bn`
- `binaryninja`
- `bv`
- `result`

Stdout and `result` are both returned. If `result` is not JSON-serializable, `bn` returns `repr(result)` and includes a warning instead of silently stringifying the whole response.

## Mutation Commands

Mutations follow the same target-selection rules as other target-specific commands.

Examples:

```bash
bn symbol rename sub_401000 player_update --preview
bn comment set --address 0x401000 "interesting branch" --preview
bn comment get --address 0x401000
bn proto get sub_401000
bn proto set sub_401000 "int __cdecl player_update(Player* self)" --preview
bn local list sub_401000
bn local rename sub_401000 0x401000:local:StackVariableSourceType:-20:2:12345 speed --preview
bn local retype sub_401000 0x401000:local:StackVariableSourceType:-20:2:12345 float --preview
bn types declare "typedef struct Player { int hp; } Player;" --preview
bn struct field set Player 0x308 movement_flag_selector uint32_t --preview
```

Preview mode applies the change, refreshes analysis, captures affected decompile diffs, and then reverts the mutation.

Non-preview writes only report success after reading the live BN session back and verifying that the requested post-state actually landed. If verification fails, the CLI returns a nonzero exit code and reverts the whole mutation or batch.

After any live type or prototype mutation, do an explicit readback:

```bash
bn proto get sub_401000
bn struct show Player
bn types show Player
bn decompile sub_401000
```

For declaration and struct mutations, preview results also include `affected_types` with before/after layouts and a unified diff. If a field edit is already identical, the result is marked with `changed: false` and a `No effective change detected` message.

For the first few changed functions, `affected_functions` also includes short `before_excerpt` and `after_excerpt` snippets around the first changed HLIL lines.

Mutation results now distinguish:

- `verified`
- `noop`
- `unsupported`
- `verification_failed`

When verification fails, JSON output also includes `requested` and `observed` state for the failed op.

`bn types declare` now uses Binary Ninja's source parser when available. When you pass `--file`, the CLI also forwards the source path so relative includes resolve the same way they would during header import in the GUI.

If a declaration only parses functions or extern variables and introduces no named types to persist, `types declare` returns a verified no-op instead of failing with `No named types found in declaration`.

`bn local list` and `bn function info` return stable `local_id` values for parameters and locals. Prefer those IDs for `bn local rename`, `bn local retype`, and batch manifests; legacy name-based targeting still works for compatibility.

## Batch Manifests

`bn batch apply` accepts a JSON manifest:

```json
{
  "target": "SnailMail_unwrapped.exe.bndb",
  "preview": true,
  "ops": [
    {
      "op": "rename_symbol",
      "kind": "function",
      "identifier": "sub_401000",
      "new_name": "player_update"
    },
    {
      "op": "set_prototype",
      "identifier": "player_update",
      "prototype": "int __cdecl player_update(Player* self)"
    }
  ]
}
```

Apply it with:

```bash
bn batch apply manifest.json
```

Batch apply verifies the live session by default. If any op fails to apply or fails post-state verification, the entire batch is reverted.

## Security / Trust Boundary

`bn` is built for a **same-user, single-host** trust model: the local agent running as your uid is trusted, and the bridge is **not** a multi-tenant API. Defense in depth is enforced so that boundary is defended, not just assumed:

- **Peer-credential enforcement (Linux).** Every connection to the bridge socket is authenticated with `SO_PEERCRED` before any operation is dispatched; a peer whose uid does not match the bridge's own uid is rejected with an `ok: false` error and never reaches an op. The check is Linux-only (documented platform guarantee), fails closed if peer credentials can't be read, and is on by default. The single documented opt-out is `BN_SOCKET_ALLOW_ANY_UID=1` — set it only when you deliberately accept connections from other uids.
- **Directory and socket mode enforcement.** The cache tree (`instances/`, `sessions/`, `spills/` under `BN_CACHE_DIR` / `~/.cache/bn`) is created `0o700`, and an already-existing directory is tightened to `0o700`. The bound Unix socket is `chmod`'d to `0o600`, and registry / session-pin / spill artifact files are written `0o600` regardless of your umask, so decompiled output and bridge metadata are never born group/world-readable. Directory- and socket-mode tightening is enforced-with-warning: on a filesystem or ACL that permits creation but rejects a mode change, the process keeps running but logs a warning naming the path rather than silently leaving it group/world-accessible. **`BN_CACHE_DIR` must point at a private, non-shared directory** — do not place it on a multi-user or networked filesystem.
- **`py_exec`.** `bn py exec` runs arbitrary Python inside the Binary Ninja process with a live `bv` — that is full compromise of the bridge user. It runs **unconditionally**: whoever can reach the owner-only socket (peercred + `0o600` socket / `0o700` dir) is trusted to run arbitrary code as your uid, so the socket trust boundary above is the only gate. Do not expose the bridge socket beyond that boundary, and do not attempt to sandbox `py_exec` as a substitute (the BN API surface makes in-process sandboxing a false guarantee).

## Troubleshooting

Check bridge state:

```bash
bn doctor
```

If `bn target list` is empty:

- make sure Binary Ninja is open
- make sure a binary or `.bndb` is open
- make sure the plugin is installed with `bn plugin install`
- reload Binary Ninja plugins or restart Binary Ninja after plugin changes

If multiple targets are open, omitted `--target` is rejected. Pass `--target <selector>` from `bn target list`, or use `--target active` only when you intentionally want the GUI-selected target.

If decompile text still looks stale after a type change, run:

```bash
bn refresh
```

That forces an analysis refresh, but it still may not fully eliminate Binary Ninja's stale `__offset(...)` presentation in every case.

## Development

Run tests with:

```bash
uv run pytest
```

Run the CLI from the repo without installing it globally:

```bash
uv run bn --help
```

## License

`bn` is released under the [MIT License](LICENSE).

It is a fork of [`banteg/bn`](https://github.com/banteg/bn); the original
copyright and permission notice are retained in `LICENSE` alongside this
fork's, as required by the MIT terms.
