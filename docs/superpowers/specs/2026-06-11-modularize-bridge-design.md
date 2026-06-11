# Modularize `bridge.py` — declarative op registry + module split (issue #33)

**Status:** approved design, pre-implementation
**Issue:** #33 (enhancement) — *Modularize bridge.py: BinaryNinjaBridge is a 199-method, ~5.2k-line class; start with a declarative op registry*
**Date:** 2026-06-11
**Scope decision:** all three staged steps (registry + mutation-engine extraction + read-op domain split).
**Delivery:** feature branch off `m4ul3r`, one commit per stage (each behaviour-preserving and green), opened as a single PR into `m4ul3r`.

---

## 1. Problem

`plugin/bn_agent_bridge/bridge.py` is ~6.5k lines. `BinaryNinjaBridge` is a 206-method class that is simultaneously the request dispatcher, the owner of target state, the entire read-op library, and the mutation engine — ~190 private helpers in one flat namespace.

The single most defensible defect is the **triple-maintained op list**: adding/renaming an op means editing three places that nothing keeps in sync —

1. `READ_LOCKED_OPS` (33 ops) — `bridge.py:236`
2. `WRITE_LOCKED_OPS` (16 ops) — `bridge.py:280`
3. the `_dispatch_on_main` if-chain (~50 arms, each also doing per-op param extraction) — `bridge.py:883`–`1101`

A missing/typo'd entry is a silent routing or locking bug. The CLI already solved this with the `@command`/`_COMMANDS` registry (`src/bn/cli.py:278`); the bridge is the half of the seam that stayed imperative.

## 2. Goals / Non-goals

**Goals**
- Single source of truth for ops: derive `READ_LOCKED_OPS`, `WRITE_LOCKED_OPS`, and dispatch from one declarative registry.
- Extract the mutation engine and the read-op domains into focused modules with enforced boundaries.
- **Behaviour preservation is the hard requirement.** Every op routes to the same logic, with the same params and the same lock, producing identical output. No bug fixes, no behaviour changes ride along.

**Non-goals**
- No fixes to #25/#30 or other open bugs bundled in (keep the diff a pure refactor; the issue suggested bundling #30's `REQUIRED_FIELDS` work, but we keep this PR behaviour-only).
- No changes to the JSON protocol, the CLI, or the socket transport.
- No change to `TargetManager`, `_ReadWriteLock`, `BridgeHandler`/`ThreadedUnixServer` — already small and cohesive; left alone.

## 3. Architecture

The `BinaryNinjaBridge` class becomes a **facade of thin delegating methods**, exactly mirroring the existing `_taint` → `taint_engine.py` precedent (`bridge.py:28`, `_taint` at `bridge.py:3378`). Each former handler body moves to a module-level free function that receives a **`BridgeContext` seam** instead of reaching back into the class; the bridge method shrinks to a one-line delegation:

```python
class BinaryNinjaBridge:
    def _decompile(self, *a, **k):
        return read_decompile.decompile(self.ctx, *a, **k)
    def _mutation(self, *a, **k):
        return mutation_engine.run(self.ctx, *a, **k)
```

This keeps all method names alive, so the **238 direct `instance._method(...)` call sites across 65 distinct methods in `test_bridge.py` pass unchanged** — the strongest available proof the refactor preserves behaviour.

### 3.1 Module layout (`plugin/bn_agent_bridge/`)

| Module | Owns | Risk |
|---|---|---|
| `bridge.py` (core) | `__init__`, `start`/`stop`, `_write_registry`, `dispatch`, `_dispatch_on_main`, `_doctor`, `_load_binary`, `_close_binary`, `_save_database`, `_target_info`, `_refresh`; the module-level lifecycle helpers + globals (`_headless_views`, `_quick_loaded_views`). **Only true instance state lives here** (`targets`, `_target_lock`, `_server`, `_thread`, `_shutdown_event`). | low |
| `op_registry.py` | `@op` decorator, `OpSpec`, `_OPS` table, duplicate-registration guard, lock-set derivation. | low |
| `seam.py` (`BridgeContext`) | `resolve_view` (the only state-touching resolver, wraps `self.targets.resolve`) + every bv-explicit resolver (`find_function`, `find_functions_by_name`, `resolve_scope_functions`, `find_symbols_by_name`, `resolve_rename_target`, `functions_containing`, `function_object_at`, `function_entry_for_address`), the address-context family (`address_context`, `sections_at`, `segment_at`, `symbol_at`, `raw_sections_at`, `section_semantics_name`, `address_is_code`, `resolve_data_string`, `safe_disassembly`), the ABI/pointer helpers (`pointer_size`, `byteorder`, `read_pointer_value`, `normalize_code_pointer`, `supports_thumb_pointer_tags`, `pointer_table_for_view`), **plus the relocated `find_type` and `render_type_layout`** (see §3.2). | medium |
| `il_format.py` | pure IL/HLIL/disasm rendering + IL-instruction iteration over `bv`/`func`/`insn`: `_il_op_name`, `_iter_llil_instructions`, `_iter_il_instructions`, the `_hlil_*` predicates, `_format_hlil_tree`, `_function_text`, `_instruction_length`, `_disasm_entry`, `_structured_disasm_entries`, `_disasm_text`, `_pseudo_c_text`, `_function_signature`, `_decompile_text`, `_comment_map`, `_render_warnings`, `_analysis_stub_warning`, `_ssa_var_entry`, `_collect_ssa_vars`, `_resolve_ssa_variable`, `_il_function_for`, `_serialize_pvs`, `_pvs_determined`, `_function_metadata`, `_function_size`. State-free leaf. | low |
| `vars.py` | read-only variable discovery/serialization: `_find_variable_by_storage`, `_variable_source_name`, `_variable_identifier`, `_local_id`, `_variable_entry`, `_variable_marker`, `_iter_canonical_variables`, `_iter_hlil_variables`, `_sort_variable_entries`, `_list_locals`, `_find_variables_by_name`, `_find_variable_selector`. State-free leaf. | low |
| `read_decompile.py` | `_decompile`, `_function_info`, `_get_prototype`, `_list_locals_for_function`, `_il`, `_disasm`, `_structured_il`, `_defuse`, `_resolved_calls`, `_possible_values`, `_pvs_targets`, `_force_function_analysis`. | low |
| `read_listing.py` | `_callsites_within_function`, `_callsites`, `_parse_function_address_bounds`, `_filtered_functions`, `_list_functions`, `_paged_function_result`, `_search_functions`. | low |
| `read_xrefs.py` | `_xrefs`, `_xrefs_to_address`, `_import_symbol_name`, `_find_import_symbol`, `_xrefs_import_symbol`, `_scan_for_calls_to`, `_resolve_type_field`, `_field_xrefs`. | low |
| `read_evidence.py` | `_function_evidence`, `_function_call_evidence`, `_function_thunk_summary`, `_call_arguments`, `_resolve_argument_value`, `_call_destination_value`, `_target_entry_for_call`, `_il_argument_texts`, `_safe_int`, `_pointer_table`, `_message_lens`, `_init_arrays`. | medium |
| `read_taint_slice.py` | `_taint`, `_backward_slice`, `_build_backward_trace`, `_is_parameter_ssa_var`, `_resolve_callee`, `_resolve_thunk`, `_extract_dest_address`, `_find_return_vars`, `_ssa_vars_from` (already leans on `taint_engine`). | low |
| `read_types.py` | `_types`, `_type_entry`, `_current_type_entry`, `_type_info` (the cycle-makers `_find_type` + `_render_type_layout` moved to seam). | low |
| `read_misc.py` | `_strings`, `_imports`, `_imports_build_summary`, `_needed_libraries`, `_sections`, `_read`, `_ascii_render`, `_is_executable_address`. | low |
| `create_comments.py` | `_function_create` (+ its own undo/preview), `_get_comment`, `_list_comments`, `_bundle_function`, `_normalize_py_result`, `_py_exec`. | medium |
| `mutation_engine.py` | the ~50-method batch mutation engine: `_mutation` (single entry), `_apply_operation`, all `_verify_*`, all `_op_*`, snapshot/diff/restore helpers, `_guess_*`, `_operation_*`, `_find_member`, `_struct_builder`. | medium |

`_revert_undo_safely` is shared by `_function_create` and the mutation engine → place in the seam (or `mutation_engine` with a re-export) to avoid a `create_comments ↔ mutation_engine` edge.

### 3.2 The one real cycle and how it is broken

Exactly one hard cycle exists (everything else is one-way, verified):

- `read_types._type_entry` (`bridge.py:4411`) calls `_render_type_layout`, which physically lives in the mutation cluster (`bridge.py:5096`).
- `mutation._operation_type_names` (`:5031`) and `_struct_builder` (`:6282`) call `_find_type`, which lives in read_types (`:4380`).

**Resolution:** relocate **both** `_find_type` and `_render_type_layout` into `seam.py` (both are state-free — `_render_type_layout` is pure formatting over `type_obj.members`, `_find_type` is resolution over `bv`). Then `read_types` and `mutation_engine` each import only the seam; neither imports the other. This relocation is the load-bearing move and **must land before `mutation_engine.py` is extracted**.

### 3.3 The op registry

Mirror the CLI precedent (`src/bn/cli.py:278`): an `@op` decorator populates a module-level `_OPS: dict[str, OpSpec]` with an import-time duplicate-registration error. Importing each domain module registers its ops as a side effect, exactly like `commands/__init__.py` triggers `@command`.

```python
@dataclass(frozen=True)
class OpSpec:
    name: str
    lock: str                                # "read" | "write" | "none"
    binder: Callable                         # (bridge, params, target) -> result
    lock_escalation: Callable | None = None  # (params) -> bool; read->write upgrade
```

- **What the decorator decorates: the binder, not the handler.** The handler methods/functions keep their current *typed* signatures (so the 238 typed `instance._method(...)` test calls forward through the facade shims unchanged — end-state "typed-in"). The thing that varies per op is the **param extraction** the if-arms do today (`int(params.get("offset", 0))`, `params["identifier"]`, the `batch_apply` manifest re-derivation at `:1091`, the 11 mutation wrappers that inject `{"op": "...", **params}`). So `@op("decompile", lock="read")` decorates a small **binder function `(bridge, params, target) -> result`** whose body *is the current if-arm verbatim* — it extracts params and calls `bridge._decompile(target, identifier, addresses=…, …)`. Binders are co-located with their domain module (in stage 1, in `bridge.py`; they migrate with their handlers in stages 3–5), so importing a module registers its ops — exactly like `commands/*.py`. There is **no separate `handler` field**: the binder closes over the handler call. The op-name set for the CLI cross-check test is simply `_OPS.keys()`. Net line count is ~neutral (≈50 binders replace ≈50 if-arms) but now declarative and single-source.
- **`READ_LOCKED_OPS` / `WRITE_LOCKED_OPS` are derived** from `_OPS` (`{name for name, s in _OPS.items() if s.lock == "..."}`) and kept as module-level names so `test_bridge.py:3532-3533` (`bridge.WRITE_LOCKED_OPS` / `bridge.READ_LOCKED_OPS`) keep working.
- **Three lock cases** stay expressible:
  1. normal `read`/`write` — `dispatch()` consults `spec.lock` (replaces the set-membership checks at `:867`/`:875`).
  2. `decompile` read→write **escalation** on `force_analysis=True` — encoded as `spec.lock_escalation`; the decision stays **in `dispatch()`** (which owns `_target_lock`), preserving today's behaviour at `:869-874`. Not buried in the handler.
  3. `load_binary` + `shutdown` **no-lock** — `lock="none"` → `dispatch()` uses `contextlib.nullcontext()` (matches `:866`). Rationale comments preserved: `load_binary` does its own fine-grained locking inside `_load_binary` (#99); `shutdown` only sets an event and must run while a writer is wedged.
- `_dispatch_on_main(self, op, params, target)` is **kept as the entry point** (tests call it directly) — its body becomes `spec = _OPS.get(op); if spec is None: raise ValueError(f"Unknown operation: {op}"); return spec.binder(self, params, target)`, preserving the `Unknown operation` error for unregistered ops. The lock selection moves into `dispatch()` consulting `spec.lock` / `spec.lock_escalation` *before* calling `_dispatch_on_main`, exactly as the set-membership checks do today.

## 4. Staging plan (one commit per stage; each behaviour-preserving and green)

1. **Op registry** (issue step 1). `@op` + `_OPS` + derived lock sets + registry-driven `_dispatch_on_main`. Handlers stay as methods for now; the decorator/binder reference `self._method`. Headline fix; self-contained; could stand alone as a PR.
2. **Seam + cycle-break.** Introduce `BridgeContext`; relocate `_find_type` + `_render_type_layout` into it; bridge methods delegate. No behaviour change.
3. **Leaf extractions.** `il_format.py`, `vars.py` (state-free); bridge methods become delegating shims.
4. **`mutation_engine.py`** (issue step 2). Extract the mutation cluster as free functions taking `ctx`; `_mutation`/`_op_*`/`_verify_*`/`_apply_operation` become shims.
5. **Read-op domain modules** (issue step 3). `read_decompile`, `read_listing`, `read_xrefs`, `read_evidence`, `read_taint_slice`, `read_types`, `read_misc`, `create_comments`. As each lands, the op handler's registration travels to the module; the bridge keeps a delegating shim for any name a test references.

## 5. Testing strategy

### 5.1 Unit / contract tests (no BN license — mock `binaryninja`)
- **Lock-set golden test:** assert the registry-derived `READ_LOCKED_OPS` and `WRITE_LOCKED_OPS` equal the exact frozen sets from before the refactor (the 33 read / 16 write op names). This proves the derivation is behaviour-identical.
- **Registry completeness:** every op name reachable from `_dispatch_on_main` has exactly one `OpSpec`; no op is in both lock classes; `decompile` is the only op with `lock_escalation`; `load_binary`/`shutdown` are `lock="none"`.
- **CLI cross-check:** every op the CLI can emit (walk `_COMMANDS`) has a bridge `OpSpec` (catches a CLI op with no bridge handler).
- **Duplicate guard:** registering a duplicate op name raises at import (mirrors the CLI guard test).
- **Unknown op:** `_dispatch_on_main("nope", …)` still raises `ValueError`.
- **Whole suite green at every stage:** the existing 5,657-line `test_bridge.py` (238 direct method calls) must pass unchanged after each of the 5 commits — this is the per-stage gate.

### 5.2 Firmware regression dogfood (against `/tmp/DMH-2000NEX` — local only)
Real-binary proof that the routing/extraction refactor produced identical behaviour.

- **Targets:** stripped 32-bit ARM ELFs from the extracted firmware, primarily `…/bin/aa_accessory_service` (stripped executable — exercises address/symbol resolution hard), plus 1-2 `.so` libraries for type/section/import coverage.
- **Method:** headless-load each binary via the `bn` CLI and run a broad op sweep — `target info`, `function list/search`, `decompile`, `il`, `disasm`, `xrefs`, `callsites`, `strings`, `types`/`type info`, `sections`, `imports`, `dataflow` (defuse/resolved-calls/possible-values), `taint`, `evidence` (function-evidence/pointer-table/message-lens/init-arrays), `read`, and **previewed mutations** (rename/comment/proto/local/struct-field with `--preview`, which apply→capture→revert so they leave no state). Capture `--format json` for each.
- **Equivalence gate:** run the identical sweep against the **baseline (`master`/pre-refactor)** bridge and the **refactored branch**, and **diff the JSON outputs**. A pure refactor must yield byte-identical results (modulo volatile fields like pid/socket path/timestamps, which the harness normalizes). Any diff is a regression and blocks the stage.
- **Leak hygiene:** this sweep and its artifacts stay **local**; nothing referencing the head-unit firmware path or binary names is committed. Committed integration tests continue to use the existing `tests/fixtures/hello_x86_64` + `add_x86_64`.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Circular imports between domain modules | The seam is the only shared dependency; the single real cycle (`read_types ↔ mutation`) is broken in stage 2 before any extraction; all other edges verified one-way. An import-time smoke test (`import bridge` in headless) gates each stage. |
| A binder mis-extracts a param (silent behaviour change) | Binders are copied verbatim from the current if-arms; the firmware JSON-equivalence diff + the full unit suite catch any drift. |
| Lock class regresses for an op (wrong lock → race/deadlock) | The lock-set golden test pins exact membership; `decompile` escalation and `load_binary`/`shutdown` no-lock have dedicated assertions. |
| Test coupling to private methods breaks mid-extraction | Facade shims keep all 65 referenced method names; suite must be green per stage, not just at the end. |
| Plugin packaging (symlinks, GUI vs headless import) | New modules use the same `from . import` relative-import style as `taint_engine`; `__init__.py`/`__main__.py` import paths verified; headless `bn-agent` start tested. |

## 7. Acceptance criteria

- `READ_LOCKED_OPS`/`WRITE_LOCKED_OPS` are derived from `_OPS`; adding an op is a single `@op` declaration.
- `bridge.py` core no longer contains the read-op library or the mutation engine; those live in the modules in §3.1 with the seam as their only shared dependency.
- `_render_type_layout` and `_find_type` live in the seam; no module imports another domain module in a cycle.
- `uv run pytest` fully green after every stage.
- Firmware dogfood: JSON output identical between baseline and refactored branch across the op sweep on the ARM targets.
- One PR into `m4ul3r` with the 5 staged commits.
