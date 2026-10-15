# bn reference — mutating

Use this reference when changing a BNDB. Mutations verify their requested state; a saved `.bndb` is the durable artifact. `bundle function` is a read/export command in `reference/reading.md`.

## Mutation loop and commands

Preview when the operation supports a reversible preview, apply, read back, and save. A first `proto set` on a function without a user prototype **cannot** be previewed: setting it pins Binary Ninja's `has_user_type` flag, which cannot be cleared. Apply that change live only when intended, then check `proto get` and its callers. A later change to an existing user prototype can be previewed. In a `batch apply --preview`, such a prototype op -- and any `set_prototype` that comes after an op that renames or creates symbols -- is left out rather than refusing the whole batch: the rest is previewed, and the skipped ops plus the `local_rename`/`local_retype` ops that depend on them are listed with status `not_previewed`, under `preview_skipped`, and as `not previewed: op N ...` lines. Only ops BEFORE a prototype count: it applies in the view they leave. A skipped prototype is still checked where it can be: a prototype that does not parse fails the preview as it would fail the apply, and so does a target that does not resolve unless an earlier op is of a kind that could supply it -- a function create (or an unknown op) for any target, a rename for a name but never for an address; a data retype supplies nothing. When one could, the target is not checked at all: the op is listed as `not previewed: its target does not resolve yet (...)`, and only the live apply tells whether it lands. A batch with nothing left to preview is still refused.

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
bn class apply-types ns::Widget --preview
bn struct field rename Record flags options --preview
bn struct field delete Record options --preview
bn function create 0x402000 --preview
# Irreversible repair only after inspecting both bodies; --preview refuses.
bn function delete 0x402000
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

Type strings (`types declare`, `struct field set`, `data retype`, `local retype`, `proto set`) follow C spelling: a `::`-qualified C++ name such as a recovered `ns::Widget::VTable` must be backtick-quoted inside a declaration body (``struct `ns::Widget::VTable`* vtable;``), and a struct/union/enum that is not also a typedef may need its tag keyword (`union pthread_mutex_t m;`). A single-type argument (`struct field set`, `data retype`, `local retype`) also accepts the unquoted forms `ns::Widget::VTable*` and `struct ns::Widget::VTable*` for a type that exists; a tag keyword must match the type's kind. When a type string fails to parse on an unknown name, the error lists the closest existing types in the spelling that resolves.

The function name in a `proto set` prototype is ignored (a prototype sets a type, never a symbol), so it may be omitted; a `::`-qualified name (`int ns::Widget::run(...)`) is accepted as written. `this` is a C++ keyword the parser rejects as a parameter name: name the parameter something else (`self`), or keep the existing `this` and type it with `bn local retype <function> this '<type>'`.

A declaration that only *references* an existing type (`struct Alpha* a;`, ``struct `ns::Widget::VTable`* vtable;``, or a function-pointer parameter such as `typedef void (*cb)(struct Alpha*);`) leaves that type as it is; only types given a body are (re)defined, and the result lists the untouched ones as `referenced_existing_types`. A type-library type the declaration uses -- by value or through a pointer, e.g. libc's `struct timespec` -- is imported into the view through Binary Ninja's own import, with the types it depends on (a dependency whose name a user type already holds comes in renamed, `__time_t_1`), and the result lists every one in `imported_from_library` instead. Under `--preview` such a declaration does not succeed (below). A declaration made only of forward declarations of existing types (`struct Alpha;`, `struct timespec;`) is a `noop`. A NEW name that a member or a function-pointer parameter merely mentions (no body, no forward declaration of its own -- usually a misspelling) is still created as an empty struct, listed in `created_by_reference` and not counted as declared; a name spelled with a tag only inside a string literal or an attribute argument is no mention. C++ base clauses (`struct Derived : Base { ... };`) are **not recorded** yet (#965): the derived struct is declared with its own members at the offsets the parser gives them, after the bases, but with no base or inherited field, and every base clause -- with or without `alignas`/`__attribute__`/`final` in its head -- is listed in `unattached_bases`. A base clause bn cannot read fails the declaration rather than being dropped.

`struct field set` replaces (by default) every member the new field's byte range overlaps, not only the one at its offset; each removed member is named in the result's `replaced_members`, on the compact line as `replaced=N` plus a `replaced: Struct::field (type @offset)` line, and in `--format json --summary` as `replaced_members`. Preview first when a struct carries padding arrays. `--no-overwrite` refuses instead of replacing.

`--preview` and type libraries: Binary Ninja imports a library type while it parses a declaration that uses it, its undo does not revert the import, and removing an import stops it from ever importing that type again, so a preview cannot undo one. What a previewed `types declare` -- or a batch that contains one -- guarantees is that it **never reports success while anything was imported**: it compares Binary Ninja's type ids before and after (a renamed import such as `__time_t_1` counts), and a preview that gained any fails (`unsupported`, exit 3), lists every import in `kept_library_imports` (they stay in the view) and says to run the declare live. Other previews make no such check and can still import and report success: `struct field set Ex 0x10 t 'struct tm' --preview` exits 0 and leaves `tm` imported, and the same holds for `proto set`, `local retype` and `data retype`. The declare preview avoids importing only where the text is certain: a tag-keyword mention (`struct`/`union`/`enum X`) of a library type the view lacks is refused before anything parses, naming it. Untagged uses (`rusage r;`, `timespec* p;`, `pthread_mutex_t m;`, `FILE* f;`, `pid_t p;`) and tags hidden by a macro can still import during a declare preview -- and then fail it as above. Either way, run the declare without `--preview`; it imports them and lists them in `imported_from_library`. A declaration that uses only platform types (`uint32_t`, `size_t`) or view types previews as before, and so does one whose member names merely look like library types (`int FILE;`). A live declare that fails keeps what Binary Ninja imported and names it -- on its own row when its own parse imported, else on the declare row that reaches the type -- as `imported_from_library`, plus `kept_library_imports` in `--format json --summary` and a `kept library imports (a rollback cannot remove them):` text line. `dirty_after` does not count them: Binary Ninja re-imports a library type on demand.

Named virtual calls: BN's RTTI analysis defines a class's vtable type (`ns::Widget::VTable`; for a derived class BN names it `ns::Base::ns::Widget::VTable`) but no instance struct. `bn class apply-types <Class>` does the typing as one verified mutation (`--preview` supported): it declares the instance struct with a vptr at offset 0 typed as the VTable type BN put on the class's slot array (padded to the class card's `size` when one was recovered; otherwise only the vptr is declared). On an existing instance struct it leaves a member that is exactly a pointer to that VTable alone (`unchanged`), retypes a single pointer-width pointer or integer member at offset 0 under its own name (`vptr_retyped`, naming what it was), adds the vptr when nothing occupies the first pointer's bytes, and **refuses** anything else there -- an embedded base, a padding array, a union, the table by value, narrower fields -- rather than overwrite it. Then it retypes `this` on each of the class's own methods; `this retyped:` counts the ones that changed and `already typed:` the rest. A thunk, a method with no parameter named `this` (a static member), and a vtable slot that is not one of the class's own methods (inherited, or every slot of a stripped build) are skipped and listed under `skipped:`; retype them with `bn local retype`. A re-run with nothing left to write is a `noop` at exit 0. It refuses a class with no primary vtable, or one BN made no VTable type of its own for (an abstract base whose table type BN only made for derived classes); the refusal names the derived tables. By hand, the same steps are:

```bash
bn types declare 'struct `ns::Widget` { struct `ns::Widget::VTable`* vtable; int32_t field; };'
bn local retype ns::dispatch <this_local_id> 'struct `ns::Widget`*'
```

`symbol rename` (and its `rename` alias) rejects empty, whitespace-only, null, or control-character-bearing names with `invalid_request` (exit 3), before writing. Controls mean C0 (`U+0000`–`U+001F`), DEL (`U+007F`), and C1 (`U+0080`–`U+009F`); they are rejected, not stripped. Batch validation checks symbol names before **any** sibling operation writes and identifies the failing operation by its zero-based index. Raw bridge calls enforce the same name rule. This symbol-specific control check does not change local/field naming rules.

Printable punctuation, including literal quote characters, remains valid and is preserved. In a shell, `bn rename parse_record ''` delivers an empty argument and is rejected; `bn rename parse_record "''"` delivers two literal single quotes and names the symbol `''`. A readback containing those two characters is not evidence that an empty-string value was accepted.

`function create` is for an address auto-analysis missed. An address **inside** an existing function is refused (`invalid_request`, exit 3) naming the containing function, its start, the offset and whether the address is an instruction boundary there, because Binary Ninja does not simply add a second function beside it: at an instruction boundary it truncates the host at the new start (the host keeps only the bytes before it), and at a mid-instruction address it leaves the host intact beside an overlapping function that decodes a different instruction stream -- the shape of a mistyped address. Pass `--allow-overlap` (batch op field `"allow_overlap": true`) only for a deliberate cut of an over-extended function; the compact status line then carries `overlaps=N` and an `overlapping_functions:` line naming each host as `truncated` or `intact`, `--format json --summary` carries `overlapping_functions`, and the full result lists the host under `affected_functions` too. `--preview` obeys the same rule, and its revert verifies the host is back as it was before the command -- in a batch, as it was before the batch's first op, so a sibling rename of the host is undone with the rest -- a host that cannot be restored is reported as `rollback_failed` with `dirty_after` true, never as a clean revert (when the batch itself errored after applying, the error envelope carries `status: rollback_failed` and `observed.dirty_after: true` instead of a result). An unmapped, non-executable or `--quick`-loaded target is refused the same way on both surfaces.

`bn function delete <address>` repairs an erroneous function entry, such as a shared tail identified by the split-body warning. For example, `bn function delete 0x402000` removes the entry at `0x402000` with Binary Ninja's `remove_user_function`, then waits for reanalysis. This mutation is journaled but has **no clean undo**: it leaves a persistent user override suppressing automatic function creation at that address, and recreating an entry does not restore its prior analysis or annotations. `--preview` returns `unsupported` before any writes, without changes. Save a durable database before applying if you need a recoverable baseline.

The mutation's `observed` state reports `function_removed`, `function_at_entry`, `bodies_containing_deleted_entry` (rows with `address` and `name`), `reanalysis_completed`, and `clean_undo_available: false`. Verification requires the entry to be absent and reanalysis to have completed. A body containing the deleted entry proves only that address is contained, not that the full tail merged or that the resulting boundaries are correct. Read the reported bodies with `bn disasm` and `bn decompile` before relying on them. A failure cannot promise automatic restoration.

An attempted live deletion carries `irreversible_change: true` even if removal or reanalysis raises: a partial write cannot be ruled out or cleanly undone. Its compact summary therefore counts that operation as changed/possibly changed and reports `dirty_after: true`, independently of `verified_count`. A failed verification still reports failure (exit 3); changed does not imply verified.

For `go rename`, `--verbose` (alias `--diffs`) requests detail and `--summary` (alias `--quiet`) forces the compact summary. It takes the standard mutation flags (`--preview`, `--summary`, `--verbose`, `--format`, `--out`) and nothing else.

Per-op statuses:

- `verified` — requested change applied and read back.
- `noop` — the requested state already existed.
- `unsupported` — this operation is unavailable for the object.
- `verification_failed` — readback disagreed; the batch attempts rollback.
- `invalid_request` — semantic request refusal, including an invalid field value.
- `reverted` — an earlier successful sibling was undone after a later batch failure; not itself a failed status.
- `not_previewed` — left out of a `--preview` because it cannot be previewed reversibly or depends on an op that cannot; not itself a failed status.
- `not_attempted` — a later sibling was never run after a batch failure; not itself a failed status.
- `rollback_failed` — restore failed; the view may contain some changes.
- `internal_error` — unexpected apply failure; rollback is attempted.

A failed batch returns one row per submitted op. Failure statuses are `unsupported`, `verification_failed`, `invalid_request`, `rollback_failed`, and `internal_error` (exit 3). CLI parser, file, routing, read, and transport errors normally exit 2, as can a semantic bridge refusal without one of those failure statuses. A local semantic mutation preflight exits 3 with `observed.request_sent: false`, distinguishing a refusal before send from a bridge-side exit 3 (which may carry `observed: {}`). Exit 4 means the mutation result could not be measured; inspect the view rather than interpreting unknown counts as zero. A failed rollback sets `dirty_after: true` and may leave `changed_count: null`.

## Output and compact status

Mutations print a compact **text status line** by default. Use `--format json --summary` for a machine-readable status, `--verbose` for full diffs, or `--out FILE` to write detail to an artifact. An explicit `--format json` requests the full JSON result unless `--summary` is also set. A mutation never swaps its status for a spill envelope: when detail spills, stdout keeps the compact status and points to `detail_artifact_path` for `results[]`. With `--out`, stdout is an artifact envelope whose `ok` describes the file write, not the mutation; read the artifact or exit code for the verdict. A refusal can emit an error envelope without writing an artifact.

### Compact status keys

This table describes the stable object from `--format json --summary`; the default text line shows fewer fields, while full `--format json` is the detailed audit without these summary counts. Every key below is present except `prototype_user_type_residue` (only when true), `overlapping_functions` (only for a create inside an existing function), `fallthrough_entries` (only for a create reached by fall-through), `replaced_members` (only when a `struct field set` replaced overlapped members), `declared_types` (only when a `types declare` landed), `kept_library_imports` (only when a failed live declare, or a failed preview, left library imports in the view), `class_apply_types` (only for `class apply-types`), `preview_skipped` (only when a preview left ops out), and `go_rename` (only for Go rename):

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
| `function_deletions` | optional `function delete` outcome rows `{address, message, observed}`; preserves the deletion outcome and no-clean-undo disclosure in compact JSON and default text output rather than reporting counts alone. `observed.bodies_containing_deleted_entry` establishes entry containment only, not a full-tail merge |
| `replaced_members` | the `{struct_name, field_name, field_type, offset}` of each member a `struct field set` removed because the new field overlaps it (the member rewritten in place under its own name is not listed); also rendered as `replaced=N` and a `replaced:` line |
| `kept_library_imports` | the library types Binary Ninja imported that stay in the view although the declare did not land: a failed live declare's, or a preview's (which then fails, since a preview cannot undo an import); also rendered as a `kept library imports (a rollback cannot remove them):` line |
| `class_apply_types` | `class apply-types` only: `class`, `vtable_type` (the vptr points at `struct <vtable_type>*`), `instance_struct` (`declared`, `vptr_retyped`, `vptr_added` or `unchanged`), `vptr_member_before` (`{field_name, field_type}` of a retyped offset-0 member), `size` and `size_source`, `retyped_methods` (`{address, name, status}`; `noop` when `this` already had the type) and `skipped_methods` (`{address, name, reason}`); also rendered as `class:`, `this retyped:` and `skipped:` lines |
| `fallthrough_entries` | the `{address, fallthrough_from}` of each `function create` whose previous instruction (fixed-width ISAs other than ARM/Thumb only; padding `nop`s and calls excluded, and a data variable over the preceding words counts as a literal pool unless the instruction before the variable's start runs into it) runs straight into the address -- likely mid-function; also rendered as a `warning:` line. The create still goes ahead |
| `go_rename` | Go-only compact scan disclosure: `candidates`, `skipped_user_named`, `skipped_already_named`, `skipped_interior_pc`, and `skipped_changed_during_apply`; also rendered as a second text line |
| `declared_types` | `types declare` counts operations, not types (one declaration is one op however many types it holds), so this carries the per-type view: `count`, the first 10 `names`, `truncated`, `unattached_bases` (`{type: reason}` for every C++ base clause, none of which is recorded yet, #965), `referenced_existing` (existing types only referenced and left unchanged), `created_by_reference` (empty types created because a member mentioned an unknown name) and `imported_from_library` (type-library types the declaration brought into the view); also rendered as `types: N declared (...)` (omitted when nothing was declared) plus one line per non-empty list |
| `preview_skipped` | the `{index, op, reason}` of each op a `--preview` left out (a prototype it cannot revert, and the ops that depend on it); the counts cover only the other ops. Also rendered as one `not previewed: op N <op> -- <reason>` line each |

`go rename` counts through its own counters rather than one `results[]` row per rename. Its `noop_count` retains `skipped_user_named`, which includes `skipped_changed_during_apply`; that overlap is subtracted from `candidates + noop_count` to obtain `op_count`. Already-named and interior-PC rows are scan-only skips, disclosed separately in `go_rename` rather than added to operation counts. Those two buckets and the apply-time overlap are optional in the full result; their absence means zero, while an unreadable supplied value is unknown, not zero.

A genuine zero-candidate scan explicitly reports zero verified, committed, and failed renames. Both live and preview runs then report `measured: true`, `changed_count: 0`, `verified_count: 0`, `failed_count: 0`, `dirty_after: false`, and exit 0; `committed` stays false. The additional compact scan disclosure explains an idempotent re-run or containment-only scan without changing the operation population. This describes what this call changed, not whether the view had earlier unsaved edits.

Unknown fields in the compact `go_rename` disclosure remain JSON `null` and render as `?` in text, including spilled status and text output files; a measured zero still renders as `0`.

If a required measurement source is missing, a supplied counter is unreadable, or a failure counter contradicts the failure rows, `measured` is false; do not trust a zero-looking `op_count` as proof nothing changed. Under a successful `--preview`, `changed_count` counts changes that would land even though the view is reverted. After a failed preview or live run whose revert completed it is zero; an incomplete revert leaves it unknown. Exit 3 for a classified failure wins over exit 4 for an unmeasured result. Read back and save before closing whenever `dirty_after` is true or the result cannot establish cleanliness.

## Batch apply

Use `batch apply` for related writes that should verify and revert as one unit. Give it a concrete selector; a CLI `-t` wins over a manifest `"target"`. Do not use `"target": "active"` in a multi-target headless manifest. A quoted heredoc keeps comments literal:

Exception: `function_delete` is non-reversible and must be the **only operation** in its batch. Mixed batches and multiple deletes are refused before any writes; preview of a singleton delete is also refused before writes. A singleton manifest is `{"ops":[{"op":"function_delete","address":"0x402000"}]}`. Do not rely on rollback to restore a deletion.

```bash
bn batch apply -t <selector> --preview - <<'BN_EOF'
{"ops": [
  {"op": "rename_symbol", "identifier": "sub_401000", "new_name": "parse_record"},
  {"op": "set_comment", "address": "0x401020", "comment": "length is validated by caller"}
]}
BN_EOF
```

The manifest must be an object with an `ops` array. Validation runs before apply; a missing field or an unknown op refuses the batch. Do not write the same key twice in one batch: each op is verified against the final state, so a later write would invalidate the earlier op's result. That includes renaming one symbol twice through a chain (`add` → `foo`, then `foo` → `bar`, or the second rename naming it by address, a case variant, or a `kind: data` rename at the function's address): when a rename lands on a symbol an earlier rename in the batch already renamed, the batch stops there with `invalid_request` naming both operations and is rolled back. The message prints the fix: drop the earlier rename and make the later one rename the symbol by its address, in the later one's position (moved first, a swap through a temporary name runs into an ambiguous name). The check uses the address the rename actually resolved to, so two renames of different symbols that happen to pass through the same name are not refused. Ops apply in order and each resolves names against the state the earlier ops left, so after a `rename_symbol` refer to that function by its new name or its address, not the old `sub_*` name. Large batches hold the exclusive write lock; split a manifest that exceeds the default 5000-op or 32 MiB request limits.

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
| `function_delete` | `address` | — | `bn function delete` (singleton batch only; no preview or clean undo) |
| `tag_add` | `type` | `function` \| `address` | `bn tag add` |
| `tag_remove` | — | `tag_id` \| `address` \| `function` | `bn tag remove` |
| `tag_type_create` | `name`, `icon` | — | `bn tag type create` |
| `tag_type_remove` | `name` | — | `bn tag type remove` |

Optional fields include `kind` for symbol rename, `overwrite_existing` or `type_name` for struct fields, `allow_overlap` for `function_create`, and `source_path` for declarations. The table names the operation's real field names; they differ across commands. The same preview, verification, readback, and save rules apply to the batch.
