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
- `noop` — already in the requested state. For `types declare`, the named types must resolve in the live database; parsing no named types is `invalid_request`, with the reason in `first_error` and the default status output, not a successful no-op.
- `unsupported` — operation not supported on this object.
- `verification_failed` — readback disagrees; the whole mutation/batch is reverted, and JSON also returns the requested vs observed state.
- `invalid_request` — the operation was refused: a bad field *value*, a missing required field, an ambiguous operation target, or conflicting options. Local semantic preflights (before sending), the bridge's pre-apply checks, and apply-time refusals all exit 3 on mutation commands. Anything already applied in the mutation/batch is reverted. An unknown op kind is `unsupported` and likewise exit 3. Only these mutation boundaries classify failure statuses as exit 3; read/resolver errors remain exit 2 (#625/#716/#744).
- `rollback_failed` — an operation failed and the automatic revert of that failure also failed; the view may be left in a mixed state.
- `internal_error` — an unexpected exception during apply; treated like a failure and reverted.

#### Refusal versus request/bridge failure

| Boundary | Exit | Meaning |
|---|---|---|
| CLI operation preflight | 3 | Nothing sent. Missing/conflicting comment or tag locations, empty rename names, missing/conflicting declaration sources, and a parsed manifest object missing an `ops` array are `invalid_request`. |
| CLI argument parser or handled file/document/routing error | 2 | Invalid flags/choices, missing declaration files, manifest I/O errors converted to `BridgeError`, empty batch-manifest stdin, invalid manifest JSON, a non-object manifest, or missing/ambiguous instance/target routing. This does not classify unhandled I/O exceptions as exit 2. These are not operation refusals. |
| Bridge mutation refusal | 3 | A failed mutation status; consult the result for rollback and observed state. |
| Read operation or bridge/transport fault | 2 | Read refusals stay read errors. An unreachable bridge or a bridge error without a failed mutation status is not proof that a write did or did not land. |

Pre-send operation refusals reuse the structured error envelope, not a fabricated
apply/rollback result. With `--format json` (or `ndjson`), for example:

```json
{"ok": false, "status": "invalid_request", "error": "comment set needs a location: an address (positional or --address) or --function", "observed": {"request_sent": false}}
```

The explanation is also printed to stderr (including under `--format text`).
`observed.request_sent: false` distinguishes this local refusal from an error
received from the bridge. `comment get` and `tag get` use the same location
checks as their mutation siblings but remain read errors (exit 2, no mutation
status). For batch input, document parsing must succeed first: invalid JSON or
a non-object document stays exit 2; an object whose `ops` field is missing or
not an array is an operation-level `invalid_request` (exit 3), before any send.

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

`--format` picks the medium; `--verbose`/`--summary` pick the detail level. No
combination changes how the outcome is CLASSIFIED (each classifies the same
full result, before anything is rendered):
0 ok / 1 a CLI-side handler error / 2 bridge or request error (including a
response this CLI cannot classify at all) / 3 a mutation status `verification_failed`,
`unsupported`, `invalid_request`, `rollback_failed`, or `internal_error` / 4 an
unmeasured success (`measured: false` — applied but unverifiable; see
"Unmeasured mutations").

An output flag cannot reclassify the outcome, but it CAN fail to deliver, and
that is the divergence. The classification is made before any output is
produced, so a combination asking for output this CLI cannot deliver
reports the documented `2` instead — an `--out` destination it cannot write, or
a reply too deeply nested for `--format json`/`ndjson` to serialize. The
default status line prints named fields and never walks such a reply, so a
verified reply exits `0` there. The divergence moves in a single direction: an
undeliverable output replaces the code with 2 and can never turn a failed or an
unmeasured mutation into a clean zero. But do NOT read that backwards: on this
path a 2 is also the code for a bridge this CLI could not reach, a reply it
could not classify at all, and a flag value rejected before anything was sent.
Two things are NOT in that list. An operation-level refusal: on a mutation it
is exit 3, as the boundary table above says (not an argparse, input-document,
or routing error). And a reply carrying ONE field this CLI cannot read: that
field is refused and disclosed by name, a verdict is still derived from the
rest, and the run exits 3 or 4 accordingly.
So a 2 alone does not tell you whether the write landed; the stderr line names
which of them it was, and when it names the delivery step, re-read the view
rather than re-issuing the mutation.

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
| `measured` | **false** when the counts below could not be derived — the op reported no `results[]` rows to derive them from, or a field they are derived from arrived in a shape no value reads out of and was refused rather than read as a zero; see "Unmeasured mutations" |
| `op_count`, `changed_count`, `verified_count`, `noop_count`, `failed_count` | derived from `results[]`; `changed_count`/`verified_count`/`noop_count`/`failed_count` are `null` (not `0`) when `measured` is `false` — `op_count` stays `0`, which is literally true |
| `rolled_back` | `true`/`false` when a revert was attempted, `null` when none was needed |
| `first_error` | the first failure's explanation, or the unmeasured explanation below when `measured` is `false` — this is the one key every consumer should check regardless of `dirty_after`. It is **not** a failure signal on its own: read `ok`/`success` for that |
| `dirty_after` | `true` iff the BNDB was left modified and needs `bn save` before closing |
| `prototype_user_type_residue` | present and `true` only when a reverted `proto set` on an AUTO function left an unclearable `has_user_type` override behind; the view is modified even though the prototype value round-tripped, so `dirty_after` is `true` too |

#### Unmeasured mutations

Every shipped mutation is measurable one of two ways: it populates `results[]`,
or — like `go rename`, the one op that reports through its own counters — it
registers a compact summary that counts those counters instead.
`test_mutation_summary_wiring.py` statically enforces that pairing for every
`_mutate`-routed op, so a mutation with nothing to count means a bridge older
or newer than this CLI, or a wiring regression that slipped the sweep.

There is a second way to arrive here, and it does not need a missing
measurement source: one that arrived UNREADABLE. A count is only read through a
choke point that refuses a field it cannot use and discloses it by name, rather
than answering `0` — a fabricated zero is indistinguishable from a real one, and
on a bulk rename a zero is the "nothing changed, don't save" verdict that
discards the batch. So a counter or a row status in a shape no value reads out
of leaves the derived counts exactly as unknown as an empty `results[]` does,
and is reported the same way. That applies to `go rename` too: registering its
own summary makes it count from a different SOURCE, not measured by guarantee.

Either way, the compact summary cannot derive real counts, and says so:

```
mutation: committed  changed=None  verified=None  noop=None  failed=None  dirty_after=True
warning: unmeasured -- this op reported no results[] rows; the changed/verified/noop/failed
counts above are UNKNOWN. dirty_after is reported True as a fail-safe, not confirmed. Do not
assume nothing changed: read the view back (e.g. `bn target info` or a targeted readback) and
`bn save` before closing.
first_error: unmeasured: this op reported no results[] rows, ...
```

...with the cause named: `this op reported no results[] rows` when there was
nothing to count, `this op's own counters could not be read` when a counter was
refused, and a `! malformed <field> field` line naming each field that was.

`dirty_after` is deliberately reported `true` here rather than `null`: `null` is
falsy under every truthiness check a control loop actually writes (`jq 'if
.dirty_after then'`, `if summary["dirty_after"]:`, `if (!s.dirty_after)
close()`), so it would read identically to a confirmed clean no-op and a naive
consumer would discard real work. Check `measured` (or just read `dirty_after`,
which fails safe on its own) before trusting a `0`-looking status line as a
confirmed no-op.

An unmeasured **live** success also changes the exit code: it is **`4`**
("applied but unverifiable"), so a script that only checks `$?` sees that the
write could not be confirmed instead of reading it as a clean success. `4` is
distinct from `3` (a failure — a status in `FAILED_MUTATION_STATUSES`, which
still wins if both apply) and from `0` (a verified or measured all-`noop` run).
It is not a new failure mode: the mutation did apply, so read the view back and
`bn save` before closing. The rule is keyed on `measured: false`, not on the
kind of call, so an unmeasured `--preview` is `4` as well — there the write was
reverted, and what could not be confirmed is what *would* have landed.

**A mutation result never spills.** A read that spills is recoverable (re-read the
artifact); an atomic write whose result is unparseable is not — the agent's model of
the BNDB silently desyncs from the BNDB. So even with `BN_SPILL_TOKENS` armed and the
detail payload over it, stdout keeps the parseable status and the detail goes to an
artifact named in `detail_artifact_path` (plus a stderr note).

**Read that status in the format that produces it.** The default mutation output is
TEXT (`mutation: committed changed=200 …`), so `json.loads(stdout)` is only
meaningful under `--format json`/`--verbose`. `--out` is the one mode that REPLACES
the status with an artifact envelope (`artifact_path`, `bytes`, `sha256`, `tokens` —
no `changed_count`), so it is not how you keep a parseable result; `--summary` is —
it forces the compact `kind: mutation_summary` envelope under any format, and it is
also what a spilled mutation falls back to (#645). A `--verbose`/`--format json`
detail large enough for the consuming wrapper to truncate is bounded with `--out`.

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

### Tags

```bash
bn tag add 0x401000 --type Important --data "len unchecked" [--preview]
bn tag add --function player_update --type Bookmarks --data "entry point"
bn tag remove --id <tag-id>                        # ids come from `bn tag list --format json`
bn tag remove 0x401000 --type Important
bn tag type create my_sink --icon <glyph>
bn tag type remove my_sink
```

Tags are the "remember this spot" annotation path — a **bookmark is just
`--type Bookmarks`** — and they run the standard preview→verify loop. A custom
type must exist (`tag type create`, a mutation) before `tag add` can use it, and
each call takes exactly one location: an address (positional or `--address`) or
`--function`, never both. Reads (`bn tag list/get/types`) are in `reading.md`.

### Go names — apply what `.gopclntab` recovered

```bash
bn go functions --summary                          # read side: what would be renamed
bn go rename [--preview]                           # apply; no positional args
```

`bn go rename` is the bulk mutation that writes the names `bn go functions`
recovered from `.gopclntab` into the database. It renames **auto-named
`sub_*`/`nullsub_*` functions only** — an already-named function is left alone
and counted as a `noop` — so it is idempotent and safe to re-run. It takes the
standard mutation flags (`--preview`, `--summary`, `--verbose`, `--format`,
`--out`) and nothing else.

It is the one mutation whose bridge result reports the work through its **own
counters** rather than a `results[]` row per rename (that array carries only the
failure rows), so it registers its own compact summary to count them. The status
line and exit codes are therefore the same as every other mutation: it reports
`measured: true` when those six counters read and agree with the failure rows,
so a clean run whose counters read is exit `0` — and a counter that arrives
unreadable is disclosed by name and the run is the unmeasured `4`, exactly as an
empty `results[]` would be on any other op. Its own summary is a different
measurement SOURCE, not an exemption from measurement.

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
- **Every op is validated before ANY is applied.** A missing required field or a bad
  field *value* is a clean `invalid_request` naming the op *index* — with a "did you
  mean" hint — and an unrecognized op kind is `unsupported`; either way exit 3, and a
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

