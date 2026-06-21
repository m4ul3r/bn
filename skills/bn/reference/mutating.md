# bn reference — mutating

Mutation surface (preview→verify→save) + bundles for the `bn` skill. See `../SKILL.md` for the map.

## 6. Mutation flow

The mutation surface is built around a four-step safety loop: **preview → live-verify → read back → save**.

### Step 1 — preview first

```bash
bn types declare "typedef struct Player { int hp; } Player;" --preview
bn types declare --file /path/to/win32_min.h --preview
bn struct field set Player 0x308 movement_flag_selector uint32_t --preview
bn symbol rename sub_401000 player_update --preview   # `bn rename sub_401000 player_update` is a top-level alias (locals: `bn local rename`; struct fields: `bn struct field rename`)
bn proto set sub_401000 "int __cdecl player_update(Player* self)" --preview
bn comment set --address 0x401000 "explain this" --preview
bn function create 0x401000 --preview
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

`bn local list` includes the register/flag locals HLIL actually renders (`rsi_1`, `rdx_3`, loop counters, the success flag), so they can be renamed and retyped like stack vars. Their **auto-generated names drift** across re-analysis — a `proto set` or `local retype` can re-render `rcx` as `result` — while the `local_id` is invariant. So for these especially, capture the `local_id` from `bn local list --format json` and pass **that** (not the on-screen name) to `local rename` / `local retype`; a name you saw earlier may no longer resolve after an intervening re-analysis.

### Comments

```bash
bn comment set 0x401000 "explain this"            # positional address = alias for --address
bn comment set --address 0x401000 "explain this"
bn comment set --function player_update "explain this"
bn comment delete 0x401000
bn comment delete --function player_update
```

`comment set/get/delete` take the address either positionally (`bn comment set 0x401000 "..."`) or via `--address`; `--function` attaches a function-level comment instead. Exactly one of address / `--function` is required.

### Struct field edits

```bash
bn struct field set Player 0x308 flags uint32_t [--no-overwrite]
bn struct field rename Player old_name new_name
bn struct field delete Player <field_name>     # NOTE: takes the field name, not an offset
```

### Bulk mutations — batch manifest

For large rename/retype/comment runs, use `bn batch apply` with a JSON manifest. Significantly faster than firing individual commands.

**Primary form — pipe the manifest on stdin with a quoted heredoc** (`-` means "read stdin"). The quoted delimiter (`<<'BN_EOF'`) makes the whole payload literal, so comments with quotes, apostrophes, `$`, backticks, or parens need no escaping — and there is no temp file to write or clean up:

```bash
bn batch apply - <<'BN_EOF'
{"target": "active", "ops": [
  {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "player_update"},
  {"op": "rename_symbol", "identifier": "sub_402000", "new_name": "player_init"},
  {"op": "set_comment", "address": "0x401040", "comment": "len isn't checked; attacker-controlled (see $r0)"}
]}
BN_EOF
```

Add `--preview` before the `-` to diff without committing: `bn batch apply --preview - <<'BN_EOF' ... BN_EOF`.

The file-path form is also accepted (`bn batch apply /tmp/manifest.json`) — use it when the manifest already exists on disk.

Rules:

- The manifest must be a dict with an `"ops"` key (not a bare list).
- Include `"target"` in the manifest or it fails with `Unknown target selector: None`.
- All ops are verified — a single failure reverts the entire batch.
- `--preview` shows diffs without committing.
- Use a unique heredoc sentinel (`BN_EOF`) so a line in a comment can't accidentally close the payload. Empty or malformed stdin yields a clean error, not a traceback.

### Step 4 — save before close

Annotations live in the `.bndb`. Always save before closing — `bn close` warns when unsaved mutations are about to be discarded (see §2).

## 7. Bundles

Use bundles when you want a reusable artifact instead of pasting long output into context:

```bash
bn bundle function sample_track_floor_height_at_position --out /tmp/floor.json
```

With `--out`, the CLI returns a JSON envelope for the written artifact instead of dumping the bundle to stdout.

