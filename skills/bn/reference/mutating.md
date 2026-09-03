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
- `invalid_request` — the op was refused **during apply** (bad field *value*, an ambiguous target, conflicting options); the whole mutation/batch is reverted and this status shows up in a per-op `results[]` row, exit 3. A manifest that instead fails the **up-front shape check** (an unknown op kind, or a missing required field) never reaches apply — nothing is touched, so there is nothing to revert — and surfaces as a bridge/request error with no `results[]` envelope at all, exit 2.
- `rollback_failed` — an operation failed and the automatic revert of that failure also failed; the view may be left in a mixed state.
- `internal_error` — an unexpected exception during apply; treated like a failure and reverted.

### Output shape — compact by default, detail on request

Mutations print a **one-line status summary** by default:

```
mutation: committed  changed=71  verified=71  noop=0  failed=0  dirty_after=True
```

That is ~225 bytes. The full audit payload — every per-op diff, `requested`,
`observed`, `before_*` field — is the single largest source of avoidable token burn
in a write-heavy session (a `proto set` cost ~7 KB; a 115-op previewed batch cost
261 KB / 87k tokens), so it is **opt-in**:

| You want | Pass |
|---|---|
| the status line (default) | *nothing* |
| full detail, human-readable | `--verbose` (alias `--diffs`) |
| full JSON envelope (`results[]`, `affected_functions[]`) | `--format json` |
| the compact status as JSON | `--format json --summary` (alias `--quiet`) |
| full detail written to a file | `--out detail.json` (stdout keeps a small envelope) |

`--format` picks the medium; `--verbose`/`--summary` pick the detail level. Exit
codes are unchanged in every combination (0 ok / 2 bridge or request error / 3
mutation status `verification_failed`, `unsupported`, `invalid_request`,
`rollback_failed`, or `internal_error`).

### Compact status keys

The compact status — the default text line, and the object returned by
`--format json --summary` — is a stable schema. Every key below is always
present except `prototype_user_type_residue`, which is emitted only when it is
true:

| key | meaning |
|---|---|
| `kind` | always `"mutation_summary"` — how a consumer tells a compact status apart from a full mutation envelope |
| `ok` / `success` | mirrors the read-command envelope (`ok` is always present, unlike the full result) |
| `committed` | true for any non-preview mutation that reached apply — including an all-noop |
| `preview` | true when `--preview` was requested |
| `measured` | **false** when the op reported no `results[]` rows to derive the fields below from; see "Unmeasured mutations" |
| `op_count`, `changed_count`, `verified_count`, `noop_count`, `failed_count` | derived from `results[]`; `changed_count`/`verified_count`/`noop_count`/`failed_count` are `null` (not `0`) when `measured` is `false` — `op_count` stays `0`, which is literally true |
| `rolled_back` | `true`/`false` when a revert was attempted, `null` when none was needed |
| `first_error` | the first failure's explanation, or the unmeasured explanation below when `measured` is `false` — this is the one key every consumer should check regardless of `dirty_after`. It is **not** a failure signal on its own: read `ok`/`success` for that |
| `dirty_after` | `true` iff the BNDB was left modified and needs `bn save` before closing |
| `prototype_user_type_residue` | present and `true` only when a reverted `proto set` on an AUTO function left an unclearable `has_user_type` override behind; the view is modified even though the prototype value round-tripped, so `dirty_after` is `true` too |

#### Unmeasured mutations

A small number of bespoke ops (identified statically by `test_mutation_summary_wiring.py`) report success through their own counters instead of populating `results[]`. When that happens the compact summary cannot derive real counts, and says so:

```
mutation: committed  changed=None  verified=None  noop=None  failed=None  dirty_after=True
warning: unmeasured -- this op reported no results[] rows; the changed/verified/noop/failed
counts above are UNKNOWN. dirty_after is reported True as a fail-safe, not confirmed. Do not
assume nothing changed: read the view back (e.g. `bn target info` or a targeted readback) and
`bn save` before closing.
first_error: unmeasured: this op reported no results[] rows, ...
```

`dirty_after` is deliberately reported `true` here rather than `null`: `null` is
falsy under every truthiness check a control loop actually writes (`jq 'if
.dirty_after then'`, `if summary["dirty_after"]:`, `if (!s.dirty_after)
close()`), so it would read identically to a confirmed clean no-op and a naive
consumer would discard real work. Check `measured` (or just read `dirty_after`,
which fails safe on its own) before trusting a `0`-looking status line as a
confirmed no-op. The exit code does **not** change for an unmeasured result —
it is still `0` on success — so a script that only checks `$?` will not notice;
read the summary object.

**A mutation result never spills.** A read that spills is recoverable (re-read the
artifact); an atomic write whose result is unparseable is not — the agent's model of
the BNDB silently desyncs from the BNDB. However large the detail payload, stdout
keeps the parseable status and the detail goes to an artifact named in
`detail_artifact_path` (plus a stderr note). So `json.loads(stdout)` on a
`batch apply` always works.

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

`comment set/get/delete` take the address either positionally (`bn comment set 0x401000 "..."`) or via `--address`; `--function` attaches a function-level comment instead. Exactly one of address / `--function` is required. The **comment text is a positional argument** — `bn comment set --address 0x.. "text"`; there is **no `--comment` flag** (the natural `--comment "text"` fails with an argparse error).

### Data variables — bind a recovered type to an address

```bash
bn types declare 'struct cmd_help_entry { char* desc; char* usage; };'
bn data retype 0x460000 'cmd_help_entry[257]' [--preview]
```

`bn data retype <addr> <type>` types a **data variable** through the standard
mutation loop — `--preview`, live verification by reading back
`bv.get_data_var_at(addr).type`, and the usual `verified` / `noop` /
`verification_failed` statuses. Before this, struct-typing a recovered global table
(a routine RE move) had no first-class path at all: `types declare` defines the
struct but cannot apply it, `symbol rename --kind data` renames without typing, and
`struct field set` edits a *type*, not a variable's binding — so the only way
through was `bn py exec`, i.e. no preview, no readback, no batch atomicity, no audit
trail.

Declare named types first: an undeclared type name is a clean `invalid_request`
pointing at `types declare`, and an unmapped address is rejected rather than typed
into nowhere. The matching batch op is `data_retype` (`address`, `new_type`), which
composes atomically with the `types_declare` that defines the struct — the natural
pairing, since the two are almost always applied together.

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
bn batch apply -t <selector> - <<'BN_EOF'
{"ops": [
  {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "player_update"},
  {"op": "rename_symbol", "identifier": "sub_402000", "new_name": "player_init"},
  {"op": "set_comment", "address": "0x401040", "comment": "len isn't checked; attacker-controlled (see $r0)"}
]}
BN_EOF
```

Pass the target with `-t <selector>` (the same selector every other command takes). Do **not** put `"target": "active"` in the manifest — `active` does not resolve under multi-target headless (the mode fan-out agents run in). A concrete `"target"` in the manifest is allowed, but a CLI `-t` always wins over it (#366).

Add `--preview` before the `-` to diff without committing: `bn batch apply --preview - <<'BN_EOF' ... BN_EOF`.

The file-path form is also accepted (`bn batch apply /tmp/manifest.json`) — use it when the manifest already exists on disk.

#### Batch op kinds and their required fields

This table is the whole manifest surface. It is asserted against
`mutation_engine.REQUIRED_FIELDS` / `REQUIRED_ONE_OF` by
`test_mutating_reference_documents_every_batch_op`, so it cannot drift from the
code. Field names are **not** mutually consistent across ops (`local_retype` takes
`variable` where `rename_symbol` takes `identifier`) — read the row, don't guess.

| `op` | required fields | one of | interactive equivalent |
|---|---|---|---|
| `rename_symbol` | `identifier`, `new_name` | — | `bn rename` / `bn symbol rename` |
| `set_comment` | `comment` | `function` \| `address` | `bn comment set` |
| `delete_comment` | — | `function` \| `address` | `bn comment delete` |
| `set_prototype` | `identifier`, `prototype` | — | `bn proto set` |
| `local_rename` | `function`, `variable`, `new_name` | — | `bn local rename` |
| `local_retype` | `function`, `variable`, `new_type` | — | `bn local retype` |
| `data_retype` | `address`, `new_type` | — | `bn data retype` |
| `struct_field_set` | `struct_name`, `field_type`, `offset`, `field_name` | — | `bn struct field set` |
| `struct_field_rename` | `struct_name`, `old_name`, `new_name` | — | `bn struct field rename` |
| `struct_field_delete` | `struct_name`, `field_name` | — | `bn struct field delete` |
| `types_declare` | `declaration` | — | `bn types declare` |
| `function_create` | `address` | — | `bn function create` |
| `tag_add` | `type` | `function` \| `address` | `bn tag add` |
| `tag_remove` | — | `tag_id` \| `address` \| `function` | `bn tag remove` |
| `tag_type_create` | `name`, `icon` | — | `bn tag type create` |
| `tag_type_remove` | `name` | — | `bn tag type remove` |

Optional fields read by the handlers: `kind` on `rename_symbol`
(`auto`/`function`/`data`), `overwrite_existing` and `type_name` (an accepted alias
for `struct_name`) on the `struct_field_*` ops, `source_path` on `types_declare`.

Rules:

- The manifest must be a dict with an `"ops"` key (not a bare list).
- **Every op is validated before ANY is applied.** A guessed op name or field name is
  a clean `invalid_request` naming the op *index* — with a "did you mean" hint — so a
  typo in op 13 no longer rolls back 12 good ops.
- **One write per key.** Every op is verified against the batch's END state, so a
  manifest that writes the same key twice (two `set_comment`s on one address, a
  `set_comment` plus a `delete_comment`) can never verify: op 0 would be judged
  against op 1's value. Such a manifest is rejected up front, naming both indices.
  Split them across two batches — last-write-wins is not expressible in one.
- `rolled_back` is **always** present in the result (`false` when committed), so a
  parser written against a preview or a failure doesn't `KeyError` on the happy path.
- Supply the target with `-t <selector>` (recommended), or a concrete `"target"` in the manifest; a CLI `-t` wins over the manifest value (#366). Without either it fails with `Unknown target selector: None`. Do not use `"target": "active"` — it doesn't resolve under multi-target headless.
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

