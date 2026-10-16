# bn reference — mutating

Use this reference when changing a BNDB. Mutations verify their requested state; a saved `.bndb` is the durable artifact. `bundle function` is a read/export command in `reference/reading.md`.

## Mutation loop and commands

Preview when the operation supports a reversible preview, apply, read back, and save. A first `proto set` on a function without a user prototype **cannot** be previewed if it changes the signature: setting it pins Binary Ninja's `has_user_type` flag, which cannot be cleared. Apply that change live only when intended, then check `proto get` and its callers. A later change to an existing user prototype can be previewed. In `batch apply --preview`, each prototype target is resolved read-only against the actual state left by preceding operations. A rename or create is not itself a reason to skip: a missing, ambiguous or malformed target fails with the live resolver error. A changing AUTO prototype is left out, together with later local operations resolving to that same function, while the rest is previewed. These skipped writes appear as `not_previewed` and under `preview_skipped`. Their preflight may still parse the proposed type; an import during that parse is disclosed and makes the importing row fail, not `not_previewed`. A batch with nothing reversible left to preview is refused.

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

`--preview` and retained types: library imports are not reverted by Binary Ninja's undo, and removing an import prevents later re-imports. **A preview never reports success when any new type ID remains in its whole-request start/end difference, regardless of library provenance.** This conservative outcome guard includes collision-renamed imports, types introduced by parsing, another operation or analysis, and retained types whose provenance is unrecognized. It does not prevent imports or promise to remove them. A retained-ID preview fails at exit 3 and lists every retained new type in the existing `kept_library_imports` field; that field's name does not establish library provenance in preview mode. Operations with observed ID/operation evidence are `unsupported`; successfully undone siblings are `reverted`. When no operation can be credited, the failure is batch-level: detailed output lists `unattributed_library_imports` and the message says attribution is unavailable, rather than blaming a sibling. A declaration naming a missing library type with a tag keyword is refused before parsing where possible. Untagged names (`timespec*`, `FILE*`, `pid_t`) and macro-hidden tags can still import and then fail; this reports the outcome, not a promise to prevent imports. Run the intended operation live only if wanted. Do not use `dirty_after` alone to establish that no types remain: it does not count library imports.

The common no-import path takes one initial type-ID snapshot and one final snapshot for the whole request, not scans around individual operations or repeated parses. Lightweight SDK type-definition notifications record ID/operation evidence; preview attribution is resolved for retained IDs in the final difference without requiring recognized library provenance. Events outside an operation's own SDK thread remain unowned. This avoids replaying irreversible imports or predicting parser behavior. `parse_performed: true` separately discloses actual parsing, including a snapshot parse for an unattempted write that imported nothing; `not_attempted` describes the write, not the snapshot parse.

Preview entry-point coverage:

| Entry point | Retained-ID guard |
|---|---|
| Symbol rename (including alias), comments, prototypes, locals, data, structs, declarations, tags and batch apply | Mutation-engine request guard |
| Standalone `function create --preview` | Same guard around the direct create/analysis/revert path |
| `go rename --preview`, including no candidates | Same guard around recovery and direct chunked renaming |
| `class apply-types --preview`, including planning/no-op results | Same guard around planning; engine-routed writes reuse that request scope |
| `spill gc --dry-run` | Filesystem-only inspection; no BinaryView or bridge mutation is performed |


Live import reporting and provenance classification remain limited to recognized attached-library or lookup metadata. An import from an explicitly selected platform's unattached `SYSCALLS` library historically escaped that classifier. The conservative preview ID rule now catches its retained IDs without classifying their provenance; it does not fix the live classifier. Explicit-platform provenance classification and the remaining #1012 items 2–3 work are left to a split follow-up for the coordinator to file.

For an unattributed, batch-level retained-ID failure, per-operation failed counts may be zero because no particular operation is blamed; `success`/`ok` remain false, exit status is 3, and `first_error` plus the retained-type list explain the failure.

Declaration-file fallback preserves the source filename and sibling include directory while using the view's library-aware parser, so real included-header errors are not replaced by false missing-file errors. Import disclosure supplements, rather than replaces, the original AUTO-prototype non-revertibility refusal. Unexpected SDK exceptions retain their original exception and exit-code classification: their text/JSON error message names every kept import, and raw protocol error results also carry `kept_library_imports`.

Named virtual calls: BN's RTTI analysis defines a class's vtable type (`ns::Widget::VTable`; for a derived class BN names it `ns::Base::ns::Widget::VTable`) but no instance struct. `bn class apply-types <Class>` does the typing as one verified mutation (`--preview` supported): it declares the instance struct with a vptr at offset 0 typed as the VTable type BN put on the class's slot array (padded to the class card's `size` when one was recovered; otherwise only the vptr is declared). On an existing instance struct it leaves a member that is exactly a pointer to that VTable alone (`unchanged`), retypes a single pointer-width pointer or integer member at offset 0 under its own name (`vptr_retyped`, naming what it was), adds the vptr when nothing occupies the first pointer's bytes, and **refuses** anything else there -- an embedded base, a padding array, a union, the table by value, narrower fields -- rather than overwrite it. Then it retypes the object parameter: the parameter named `this` on a method the card lists under the class, else -- a stripped build names none -- parameter 0, keeping its name, of a function whose OWN body stores this class's vtable address point at offset 0 of BN's parameter-0 variable (a constructor's or destructor's vptr store, read from its MLIL, through plain copies). That store is the evidence that parameter 0 points at an object of the class, whatever the ABI passes first (a hidden return buffer on x86-64, i386 and 32-bit ARM). Candidates are the card's methods, the vtable's slot targets and every function with a code reference to the address point. A store at a nonzero offset (a secondary base) does not count; a function that also stores another class's address point there (a derived constructor with an inlined base one) is skipped naming both classes; a parameter 0 already typed as a pointer to another struct is left alone; a class known only through the typeinfo byte scan gets no parameter retypes. Every slot target without the store -- an ordinary virtual method, a free function placed in a table, a callback BN's VTable type runs into -- is listed as `slot target, parameter 0 not retyped (no address-point store)` with a `check` command (`bn proto get <address>`) to read before typing it by hand. The command writes none of them, though BN's own analysis may carry a new type on into a function that passes its parameter straight to a retyped one (a deleting destructor calling the complete one). Each retype is verified by readback. `object parameter retyped:` counts the ones that changed, `already typed:` the rest (a parameter 0 already `struct <Class>*` is counted there without a write), and says how many went by parameter 0. Skipped methods are COUNTED by reason on the `skipped:` line, followed by the first `bn proto get` command; `--format json` lists each. A re-run with nothing left to write changes nothing and exits 0. It refuses a class with no primary vtable, or one BN made no VTable type of its own for (an abstract base whose table type BN only made for derived classes); the refusal names the derived tables, and says so when the view has no VTable type for any class (a non-PIE build, where BN's RTTI analysis names nothing). By hand, the same steps are:

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

The irreversible result is established and the view is marked dirty before removal. An ordinary subsequent exception, including journal or readback failure, returns that failed result with the original `observed.error` and `observed.error_stage` rather than a generic read error. Post-state observations not obtained are `null`, not invented absence or empty ownership; `function_at_entry: null` proves absence only when `function_removed: true`. Errors also appear in the mutation's default text and compact `first_error`.

Known limitations: `KeyboardInterrupt`, `SystemExit`, and other direct `BaseException` interruptions propagate rather than returning a mutation response. Dirty tracking is already established before removal, so an interrupted write remains protected by the unsaved-change gate while the bridge survives; reconnect and inspect before saving or discarding. If dirty marking itself fails, removal is not attempted. Its response may conservatively say changed/dirty even though no deletion occurred. No clean undo or interrupted-journal restoration is promised.

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

This table describes `--format json --summary`; default text shows fewer fields, while full JSON is the detailed audit. Conditional keys are `reverted_count` (reverted siblings present), `prototype_user_type_residue` (true only), `overlapping_functions`, `fallthrough_entries`, `replaced_members`, `declared_types`, `kept_library_imports` (recognized kept imports after a failed live mutation, or every retained new type ID after a preview), `class_apply_types`, `preview_skipped`, and `go_rename`; other keys below are always present.

| key | meaning |
|---|---|
| `kind` | `mutation_summary` |
| `ok` / `success` | classified operation outcome |
| `committed` | whether live apply reached commit, including all-noop; `go rename` with no candidates reports false |
| `preview` | whether preview was requested |
| `measured` | false when result rows or required counters cannot establish counts |
| `op_count` | For `go rename`, candidates plus scan-time `skipped_user_named`; excludes already-named and interior-PC scan skips. Apply-time skips are not counted twice. This need not equal `defined_count` and can be zero on a repeat run. |
| `changed_count`, `verified_count`, `noop_count`, `failed_count` | measured counts; an unknown derived count is `null`, not zero |
| `reverted_count` | successfully undone sibling operations after a batch failure; when nonzero, also rendered as `reverted=N` |
| `rolled_back` | meaningful when `committed` is false; false/null on a committed run does not mean restore ran |
| `first_error` | first failure or measurement explanation; inspect even when `dirty_after` is false |
| `dirty_after` | whether a live change may need saving; true is the safe value when measurement or rollback failed |
| `prototype_user_type_residue` | unclearable user-type override after a failed/reverted prototype change |
| `overlapping_functions` | the `{name, address, offset, instruction_boundary, truncated}` of each function a `function create --allow-overlap` landed inside (`truncated` says whether the host was cut at the new start or left intact); absent for an ordinary create |
| `function_deletions` | optional `function delete` outcome rows `{address, message, observed}`; preserves the deletion outcome and no-clean-undo disclosure in compact JSON and default text output rather than reporting counts alone. `observed.bodies_containing_deleted_entry` establishes entry containment only, not a full-tail merge |
| `replaced_members` | the `{struct_name, field_name, field_type, offset}` of each member a `struct field set` removed because the new field overlaps it (the member rewritten in place under its own name is not listed); also rendered as `replaced=N` and a `replaced:` line |
| `kept_library_imports` | in preview, every new type ID retained at request end regardless of library provenance, including snapshot/preflight types whose write did not run; after a failed live mutation, kept imports recognized by the attached-library/lookup classifier. Also rendered as a separate retained-type disclosure; the field name alone does not prove library provenance in preview mode. |
| `class_apply_types` | `class apply-types` only: `class`, `vtable_type` (the vptr points at `struct <vtable_type>*`), `instance_struct` (`declared`, `vptr_retyped`, `vptr_added` or `unchanged`), `vptr_member_before` (`{field_name, field_type}` of a retyped offset-0 member), `size` and `size_source`, `retyped_methods` (`{address, name, variable, basis, status}`; `variable` is `this` or parameter 0's own name, `basis` says which evidence -- a parameter BN named `this`, or the class's vtable address point stored into parameter 0; `noop` when it already had the type), `skipped_methods` (`{address, name, reason}`, plus `check` on a slot target) and `address_point_refs_truncated` when more than 256 functions refer to the address point; also rendered as `class:`, `object parameter retyped:` and a counted `skipped:` line |
| `fallthrough_entries` | the `{address, fallthrough_from}` of each `function create` whose previous decoded instruction (fixed-width ISAs other than ARM/Thumb only; padding `nop`s and calls excluded) runs straight into the address -- it may be mid-function; also rendered as a `warning:` line naming that instruction. The create still goes ahead. Typed data can suppress the advisory, but untyped literal-pool words may decode as instructions: this is not a literal-pool detector or proof of code |
| `go_rename` | Go-only compact scan disclosure: `candidates`, `skipped_user_named`, `skipped_already_named`, `skipped_interior_pc`, and `skipped_changed_during_apply`; also rendered as a second text line |
| `declared_types` | `types declare` counts operations, not types (one declaration is one op however many types it holds), so this carries the per-type view: `count`, the first 10 `names`, `truncated`, `unattached_bases` (`{type: reason}` for every C++ base clause, none of which is recorded yet, #965), `referenced_existing` (existing types only referenced and left unchanged), `created_by_reference` (empty types created because a member mentioned an unknown name) and `imported_from_library` (type-library types the declaration brought into the view); also rendered as `types: N declared (...)` (omitted when nothing was declared) plus one line per non-empty list |
| `preview_skipped` | the `{index, op, reason}` of each write left out of a preview; preflight parsing may still run, and an importing parse is counted as a failure. Also rendered as one `not previewed: op N <op> -- <reason>` line each |

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

Rename-chain advice is JSON-escaped, including quote and backslash characters in names. Dropping the first rename removes its intermediate name; update references to that name to use the symbol address. A rename that changes nothing is not the first rename of a chain and stays `noop` if a later operation renames that symbol through another locator; its `observed.name` still reports the final live name. The up-front same-key duplicate-write rule is unchanged.

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
