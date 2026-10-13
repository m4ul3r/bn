# bn reference — mutating

Use this reference when changing a BNDB. Mutations verify their requested state; a saved `.bndb` is the durable artifact. `bundle function` is a read/export command in `reference/reading.md`.

## Mutation loop and commands

Preview when the operation supports a reversible preview, apply, read back, and save. A first `proto set` on a function without a user prototype **cannot** be previewed: setting it pins Binary Ninja's `has_user_type` flag, which cannot be cleared. Apply that change live only when intended, then check `proto get` and its callers. A later change to an existing user prototype can be previewed.

A preview of an existing USER prototype restores the exact prior prototype and keeps `has_user_type` true. A local rename preview restores the prior name, type, and AUTO/USER provenance; an AUTO local must not become user-defined merely because it was previewed. The same local restoration applies when a later operation fails in a mixed batch.

```bash
bn symbol rename sub_401000 parse_record --preview
bn proto get parse_record
bn proto set parse_record 'int parse_record(char *buf, int len)'
bn local list parse_record --format json
bn local rename parse_record <local_id> record_len --preview
bn local retype parse_record <local_id> uint32_t --preview
bn types declare 'struct Record { uint32_t length; };' --preview
bn data retype 0x404000 'Record[8]' --preview
bn struct field set Record 0x4 flags uint32_t --preview
bn struct field rename Record flags options --preview
bn struct field delete Record options --preview
bn function create 0x402000 --preview
bn comment set --address 0x402010 'length comes from header' --preview
bn comment delete --address 0x402010 --preview
bn tag type create audit_note --icon A --preview
bn tag add 0x402010 --type audit_note --data 'review bound' --preview
bn tag remove 0x402010 --type audit_note --preview
bn tag type remove audit_note --preview
bn go rename --preview
bn batch apply -t <selector> /tmp/changes.json --preview
bn save
```

Use `local_id` from `local list --format json` for local edits; auto variable names can change after analysis. `comment set/get/delete` take an address (positional or `--address`) or `--function` for a function documentation comment, exactly one location per call. The comment body is positional. A bookmark is a tag with `--type Bookmarks`; a custom tag type must exist before use. `struct field delete` accepts a field name or offset. Declare a named type before `data retype` binds it to an address. `go rename` applies names recovered by `bn go functions` and renames **auto-named `sub_*`/`nullsub_*` functions only** as recognized by the name-shape check: exactly `sub_<address>` or any `nullsub_` prefix. A user-chosen name of that shape is also replaced; other names are skipped. After Go names land, it is idempotent and safe to re-run.

`symbol rename` (and its `rename` alias) rejects empty, whitespace-only, null, or control-character-bearing names with `invalid_request` (exit 3), before writing. Controls mean C0 (`U+0000`–`U+001F`), DEL (`U+007F`), and C1 (`U+0080`–`U+009F`); they are rejected, not stripped. Batch validation checks symbol names before **any** sibling operation writes and identifies the failing operation by its zero-based index. Raw bridge calls enforce the same name rule. This symbol-specific control check does not change local/field naming rules.

Printable punctuation, including literal quote characters, remains valid and is preserved. In a shell, `bn rename parse_record ''` delivers an empty argument and is rejected; `bn rename parse_record "''"` delivers two literal single quotes and names the symbol `''`. A readback containing those two characters is not evidence that an empty-string value was accepted.

`function create` is for an address auto-analysis missed. An address **inside** an existing function is refused (`invalid_request`, exit 3) naming the containing function, its start, the offset and whether the address is an instruction boundary there, because Binary Ninja does not simply add a second function beside it: at an instruction boundary it truncates the host at the new start (the host keeps only the bytes before it), and at a mid-instruction address it leaves the host intact beside an overlapping function that decodes a different instruction stream -- the shape of a mistyped address. Pass `--allow-overlap` (batch op field `"allow_overlap": true`) only for a deliberate cut of an over-extended function; the compact status line then carries `overlaps=N` and an `overlapping_functions:` line naming each host as `truncated` or `intact`, `--format json --summary` carries `overlapping_functions`, and the full result lists the host under `affected_functions` too. `--preview` obeys the same rule, and its revert verifies the host is back as it was before the command -- in a batch, as it was before the batch's first op, so a sibling rename of the host is undone with the rest -- a host that cannot be restored is reported as `rollback_failed` with `dirty_after` true, never as a clean revert (when the batch itself errored after applying, the error envelope carries `status: rollback_failed` and `observed.dirty_after: true` instead of a result). An unmapped, non-executable or `--quick`-loaded target is refused the same way on both surfaces.

For `go rename`, `--verbose` (alias `--diffs`) requests detail and `--summary` (alias `--quiet`) forces the compact summary. It takes the standard mutation flags (`--preview`, `--summary`, `--verbose`, `--format`, `--out`) and nothing else.

Per-op statuses:

- `verified` — requested change applied and read back.
- `noop` — the requested state already existed.
- `unsupported` — this operation is unavailable for the object.
- `verification_failed` — readback disagreed; the batch attempts rollback.
- `invalid_request` — semantic request refusal, including an invalid field value.
- `reverted` — an earlier successful sibling was undone after a later batch failure; not itself a failed status.
- `not_attempted` — a later sibling was never run after a batch failure; not itself a failed status.
- `rollback_failed` — restore failed; the view may contain some changes.
- `internal_error` — unexpected apply failure; rollback is attempted.

A failed batch returns one row per submitted op. Failure statuses are `unsupported`, `verification_failed`, `invalid_request`, `rollback_failed`, and `internal_error` (exit 3). CLI parser, file, routing, read, and transport errors normally exit 2, as can a semantic bridge refusal without one of those failure statuses. A local semantic mutation preflight exits 3 with `observed.request_sent: false`, distinguishing a refusal before send from a bridge-side exit 3 (which may carry `observed: {}`). Exit 4 means the mutation result could not be measured; inspect the view rather than interpreting unknown counts as zero. A failed rollback sets `dirty_after: true` and may leave `changed_count: null`.

## Output and compact status

Mutations print a compact **text status line** by default. Use `--format json --summary` for a machine-readable status, `--verbose` for full diffs, or `--out FILE` to write detail to an artifact. An explicit `--format json` requests the full JSON result unless `--summary` is also set. A mutation never swaps its status for a spill envelope: when detail spills, stdout keeps the compact status and points to `detail_artifact_path` for `results[]`. With `--out`, stdout is an artifact envelope whose `ok` describes the file write, not the mutation; read the artifact or exit code for the verdict. A refusal can emit an error envelope without writing an artifact.

### Compact status keys

This table describes the stable object from `--format json --summary`; the default text line shows fewer fields, while full `--format json` is the detailed audit without these summary counts. Every key below is present except `prototype_user_type_residue` (only when true), `overlapping_functions` (only for a create inside an existing function), and `go_rename` (only for Go rename):

| key | meaning |
|---|---|
| `kind` | `mutation_summary` |
| `ok` / `success` | classified operation outcome |
| `committed` | whether live apply reached commit, including all-noop; `go rename` with no candidates reports false |
| `preview` | whether preview was requested |
| `measured` | false when result rows or required counters cannot establish counts |
| `op_count` | For `go rename`, candidates plus scan-time `skipped_user_named`; excludes already-named and interior-PC scan skips. Apply-time skips are not counted twice. This need not equal `defined_count` and can be zero on a repeat run. |
| `changed_count`, `verified_count`, `noop_count`, `failed_count` | measured counts; an unknown derived count is `null`, not zero |
| `rolled_back` | meaningful when `committed` is false; false/null on a committed run does not mean restore ran |
| `first_error` | first failure or measurement explanation; inspect even when `dirty_after` is false |
| `dirty_after` | whether a live change may need saving; true is the safe value when measurement or rollback failed |
| `prototype_user_type_residue` | unclearable user-type override after a failed/reverted prototype change |
| `overlapping_functions` | the `{name, address, offset, instruction_boundary, truncated}` of each function a `function create --allow-overlap` landed inside (`truncated` says whether the host was cut at the new start or left intact); absent for an ordinary create |
| `go_rename` | Go-only compact scan disclosure: `candidates`, `skipped_user_named`, `skipped_already_named`, `skipped_interior_pc`, and `skipped_changed_during_apply`; also rendered as a second text line |

`go rename` counts through its own counters rather than one `results[]` row per rename. Its `noop_count` retains `skipped_user_named`, which includes `skipped_changed_during_apply`; that overlap is subtracted from `candidates + noop_count` to obtain `op_count`. Already-named and interior-PC rows are scan-only skips, disclosed separately in `go_rename` rather than added to operation counts. Those two buckets and the apply-time overlap are optional in the full result; their absence means zero, while an unreadable supplied value is unknown, not zero.

A genuine zero-candidate scan explicitly reports zero verified, committed, and failed renames. Both live and preview runs then report `measured: true`, `changed_count: 0`, `verified_count: 0`, `failed_count: 0`, `dirty_after: false`, and exit 0; `committed` stays false. The additional compact scan disclosure explains an idempotent re-run or containment-only scan without changing the operation population. This describes what this call changed, not whether the view had earlier unsaved edits.

Unknown fields in the compact `go_rename` disclosure remain JSON `null` and render as `?` in text, including spilled status and text output files; a measured zero still renders as `0`.

If a required measurement source is missing, a supplied counter is unreadable, or a failure counter contradicts the failure rows, `measured` is false; do not trust a zero-looking `op_count` as proof nothing changed. Under a successful `--preview`, `changed_count` counts changes that would land even though the view is reverted. After a failed preview or live run whose revert completed it is zero; an incomplete revert leaves it unknown. Exit 3 for a classified failure wins over exit 4 for an unmeasured result. Read back and save before closing whenever `dirty_after` is true or the result cannot establish cleanliness.

## Batch apply

Use `batch apply` for related writes that should verify and revert as one unit. Give it a concrete selector; a CLI `-t` wins over a manifest `"target"`. Do not use `"target": "active"` in a multi-target headless manifest. A quoted heredoc keeps comments literal:

```bash
bn batch apply -t <selector> --preview - <<'BN_EOF'
{"ops": [
  {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "parse_record"},
  {"op": "set_comment", "address": "0x401020", "comment": "length is validated by caller"}
]}
BN_EOF
```

The manifest must be an object with an `ops` array. Validation runs before apply; a missing field or an unknown op refuses the batch. Do not write the same key twice in one batch: each op is verified against the final state, so a later write would invalidate the earlier op's result. Ops apply in order and each resolves names against the state the earlier ops left, so after a `rename_symbol` refer to that function by its new name or its address, not the old `sub_*` name. Large batches hold the exclusive write lock; split a manifest that exceeds the default 5000-op or 32 MiB request limits.

### Batch op kinds and required fields

`bn batch apply --help` lists every accepted kind; use this table for its required fields and interactive equivalent.
Help locates this reference within the installed `bn` skill; a source checkout or a particular working directory is not required.

| `op` | required fields | one of | interactive equivalent |
|---|---|---|---|
| `rename_symbol` | `identifier`, `new_name` | — | `bn symbol rename` |
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

Optional fields include `kind` for symbol rename, `overwrite_existing` or `type_name` for struct fields, `allow_overlap` for `function_create`, and `source_path` for declarations. The table names the operation's real field names; they differ across commands. The same preview, verification, readback, and save rules apply to the batch.
