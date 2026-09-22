# bn reference — mutating

Use this reference when changing a BNDB. Mutations verify their requested state; a saved `.bndb` is the durable artifact. `bundle function` is a read/export command in `reference/reading.md`.

## Mutation loop and commands

Preview when the operation supports a reversible preview, apply, read back, and save. A first `proto set` on a function without a user prototype **cannot** be previewed: setting it pins Binary Ninja's `has_user_type` flag, which cannot be cleared. Apply that change live only when intended, then check `proto get` and its callers. A later change to an existing user prototype can be previewed.

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

Use `local_id` from `local list --format json` for local edits; auto variable names can change after analysis. `comment set/get/delete` take an address (positional or `--address`) or `--function` for a function documentation comment, exactly one location per call. The comment body is positional. A bookmark is a tag with `--type Bookmarks`; a custom tag type must exist before use. `struct field delete` accepts a field name or offset. Declare a named type before `data retype` binds it to an address. `go rename` applies names recovered by `bn go functions` and renames **auto-named `sub_*`/`nullsub_*` functions only**. It is idempotent and safe to re-run.

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

A failed batch returns one row per submitted op. Failure statuses are `unsupported`, `verification_failed`, `invalid_request`, `rollback_failed`, and `internal_error` (exit 3). CLI parser, file, routing, read, and transport errors normally exit 2; a local semantic mutation preflight exits 3 with `observed.request_sent: false`, distinguishing a refusal before send from a bridge-side exit 3 (which may carry `observed: {}`). Exit 4 means the mutation result could not be measured; inspect the view rather than interpreting unknown counts as zero. A failed rollback sets `dirty_after: true` and may leave `changed_count: null`.

## Output and compact status

Mutations print a compact **text status line** by default. Use `--format json --summary` for a machine-readable status, `--verbose` for full diffs, or `--out FILE` to write detail to an artifact. An explicit `--format json` requests the full JSON result unless `--summary` is also set. A mutation result never swaps its status for a spill envelope; with `--out`, stdout is an artifact envelope rather than the mutation status.

### Compact status keys

This table describes the stable object from `--format json --summary`; the default text line shows fewer fields, while full `--format json` is the detailed audit without these summary counts. Every key below is present in the summary object except `prototype_user_type_residue`, emitted only when true:

| key | meaning |
|---|---|
| `kind` | `mutation_summary` |
| `ok` / `success` | classified operation outcome |
| `committed` | whether live apply reached commit, including all-noop |
| `preview` | whether preview was requested |
| `measured` | false when result rows or required counters cannot establish counts |
| `op_count` | For `go rename`, candidates plus scan-time `skipped_user_named`; excludes `skipped_already_named` and `skipped_interior_pc`, so it need not equal `defined_count` and can be zero on a repeat run. |
| `changed_count`, `verified_count`, `noop_count`, `failed_count` | measured counts; an unknown derived count is `null`, not zero |
| `rolled_back` | true/false when restore was attempted, null when none was needed |
| `first_error` | first failure or measurement explanation; inspect even when `dirty_after` is false |
| `dirty_after` | whether a live change may need saving; true is the safe value when measurement or rollback failed |
| `prototype_user_type_residue` | unclearable user-type override after a failed/reverted prototype change |

`go rename` counts through its own counters rather than one `results[]` row per rename. If a counter is missing, unreadable, or inconsistent with failure rows, `measured` is false; do not trust a zero-looking `op_count` as proof nothing changed. A cleanly completed revert establishes `changed_count: 0`; an incomplete revert leaves it unknown. Exit 3 for a classified failure wins over exit 4 for an unmeasured result. Read back and save before closing whenever `dirty_after` is true or the result cannot establish cleanliness.

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

The manifest must be an object with an `ops` array. Validation runs before apply; a missing field or an unknown op refuses the batch. Do not write the same key twice in one batch: each op is verified against the final state, so a later write would invalidate the earlier op's result. Large batches hold the exclusive write lock; split a manifest that exceeds the default 5000-op or 32 MiB request limits.

### Batch op kinds and required fields

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

Optional fields include `kind` for symbol rename, `overwrite_existing` or `type_name` for struct fields, and `source_path` for declarations. The table names the operation's real field names; they differ across commands. The same preview, verification, readback, and save rules apply to the batch.
