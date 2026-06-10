# Issue #7 — Consolidate backward slicers and call-target resolvers

## Implementation status

- **PR1 (Part 2 — call-target/thunk resolver): DONE.** The canonical resolver now lives in
  `taint_engine.py` as `extract_dest_address`, `follow_thunk`, `targets_from_pvs`,
  `resolve_call_target` (+ `ResolvedTarget`), plus the tolerant `_mlil_ssa`/`_ssa_instructions`
  and guarded symbol-API helpers. The bridge's `_resolve_callee`/`_resolve_thunk`/
  `_extract_dest_address` are thin delegating shims; `TaintEngine._call_targets_from_pvs`
  delegates to the module-level `targets_from_pvs`. Evidence's pointer-normalizer is left
  separate (intentional — it is not call resolution). 15 new import-free unit tests added;
  full `test_taint_engine`/`test_bridge`/`test_cli` suites stay green (367 passed).
- **PR2 (Part 1 — full backward-slicer merge): NOT DONE, by decision.** On inspection the two
  walkers use incompatible variable models (trace: `isinstance(SSAVariable)` + `str()`; engine:
  duck-typed `.var`/`.version` via `var_label`) and different test fakes
  (`_FakeFunction.medium_level_il` / `_FakeSSAFunction.basic_blocks` / `_FakeSSAVariable.name`).
  Routing `bn trace` through `TaintEngine.backward` would require rewriting the bridge trace
  test suite (which the plan forbade) and cannot be byte-verified without a live Binary Ninja.
  Per the plan's documented fallback, Part 1 degrades to "share the resolver, keep two walks" —
  the shared resolution substrate delivered in PR1.

> The remainder of this document is the originally approved plan, preserved for the record.

---

## Context

Issue #7 is the cleanup follow-up to the #6 merge-order decision. After landing #3/#5
on `m4ul3r` and rebasing #4, the tree deliberately carries duplication in two areas:

1. **Two backward MLIL-SSA slicers** — `bn trace` (`_build_backward_trace` in `bridge.py`)
   and `bn taint backward` (`TaintEngine._backward_slice` in `taint_engine.py`).
2. **Call-target/thunk resolution** spread across trace, taint, and the evidence ops.

Exploration refined the ticket's framing in two important ways, and the plan reflects them:

- The "three parallel thunk resolvers" is overstated. **Only the trace path actually follows
  thunks** (`_resolve_callee`/`_resolve_thunk`, `bridge.py:3740-3808`). Evidence's
  `_normalize_code_pointer` (`bridge.py:1715`) is pointer normalization + context decoration,
  not call resolution. Taint's `_call_targets_from_pvs` (`taint_engine.py:642`) is already pure
  and deliberately treats thunks as opaque. The genuinely duplicated primitive is **dest-address
  extraction**: `_extract_dest_address` (`bridge.py:3810`) is a superset of `const_target`
  (`taint_engine.py:141`). So Part 2 unifies the real shared core (dest-address + thunk-follow),
  and leaves evidence's normalizer separate.
- The two slicers share the use-def walk core but are **mirror-imaged**: trace descends into
  callees at CALL defs (`bridge.py:3690-3719`); taint ascends into callers at parameter terminals
  (`taint_engine.py:1252`) and consults the model DB (`1228-1243`). Their outputs differ
  (trace = flat var-centric list; taint = `{steps, origin, crossed}` paths) and both are pinned
  by tests. Unification is via mode flags on the engine + a bridge adapter that keeps
  `bn trace`'s output **byte-stable**.

**Canonical home:** `taint_engine.py` — self-contained, zero `binaryninja` import (duck-typed
against synthetic IL fakes in `tests/test_taint_engine.py`), and the best-tested module.

**Hard invariant:** any code moved into `taint_engine.py` must preserve the zero-`binaryninja`-import
rule. It may only touch BN state through duck-typed objects/methods (`bv.get_function_at`,
`fn.mlil`, `.ssa_form`, `.basic_blocks`, …). The trace resolver already obeys this (it never
constructs a `binaryninja` class), so the move is mechanically feasible.

**Outcome:** one call-target/thunk resolver used by all real consumers, and one backward slicer
that both `bn trace` and `bn taint backward` route through — with no change to CLI output, dispatch
op names, or lock classification.

Two PRs. PR1 (resolvers) lands first; PR2 (slicers) depends on it for callee resolution.

---

## Known porting hazards (apply throughout)

- **`mlil` vs `medium_level_il` accessor gap.** `taint_engine._ssa_func` reads `getattr(func, "mlil")`
  and the test fakes (`FFunc`) expose `.mlil` only; trace's helpers read `func.medium_level_il`.
  Real BN aliases both. Moved code MUST use a tolerant accessor:
  `getattr(fn, "mlil", None) or getattr(fn, "medium_level_il", None)`.
- **Missing symbol APIs on fakes.** `FBV` has `get_function_at` but **not** `get_symbols_by_name` /
  `get_symbol_by_raw_name`. Import-name resolution must `getattr`-guard those so unit tests exercise
  the thunk/value-set paths without faking a PLT symbol table.
- Reuse existing pure helpers in `taint_engine.py`: `op_name` (98), `const_target` (141),
  `_instr_dict` (157), `ssa_reads`/`expr_reads` (127/137). Do **not** reintroduce `_il_op_name`.

---

## PR1 — Unify the call-target / thunk resolver (Part 2)

Canonical resolver lives in `taint_engine.py` as module-level, duck-typed functions (testable with
zero engine state), reused by trace and taint. Evidence's pointer-normalizer stays put.

### Step 1 — Add the unified resolver to `taint_engine.py`
Port verbatim from the bridge/trace logic, adapting to the duck-typed + tolerant-accessor rules:

```python
@dataclass
class ResolvedTarget:
    address: int | None     # entry of resolved function (post-thunk if followed)
    function: Any | None    # bv function object or None
    via: str | None         # "direct" | "import" | "value-set" | "agent-map" | "thunk"
    thunk_chain: list[int]  # addresses traversed while following thunks ([] if none)

def extract_dest_address(bv, dest) -> int | None         # port of bridge _extract_dest_address (3810)
def follow_thunk(bv, fn, *, _seen=None) -> Any | None     # port of bridge _resolve_thunk (3781)
def targets_from_pvs(pvs) -> list[int]                    # move of _call_targets_from_pvs (642), already pure
def resolve_call_target(bv, call_insn, *, follow_thunks=False, resolve_map=None) -> ResolvedTarget
```

`resolve_call_target` precedence (superset of all real behaviors): direct numeric/`.constant` dest →
import-name lookup (guarded) → `resolve_map[hex(addr)]` → value-set via `targets_from_pvs` →
optional `follow_thunk`. Each consumer enables only what it needs. Keep `const_target` (141) as the
direct-const fast path used widely in forward analysis; have it delegate to `extract_dest_address` so
there is a single source of truth.

### Step 2 — Rewire taint's forward analysis
`taint_engine.py:852-862`: replace the inline `resolve_map`-then-PVS branch with
`resolve_call_target(self.bv, ins, resolve_map=self.resolve_map)` and read its candidates + `via`.
Keep `_is_internal` (573) thunk-**skipping** unchanged — forward taint models thunks rather than
unwinding them; do **not** set `follow_thunks=True` here (default off preserves current output).

### Step 3 — Rewire trace's callee resolver in `bridge.py`
Reimplement `_resolve_callee` (3740) as a thin wrapper over
`_taint.resolve_call_target(bv, call_insn, follow_thunks=True).function`, preserving its name/signature
so the slicer and tests are untouched. Convert `_resolve_thunk` (3781) and `_extract_dest_address` (3810)
into thin shims delegating to the engine functions (defer outright deletion to PR2's cleanup tail to
keep this diff reviewable).

### Step 4 — Leave evidence ops alone
`_normalize_code_pointer` (1715), `_call_destination_value` (3202), `_function_thunk_summary` (3377)
are intentionally separate (pointer hygiene + a heuristic thunk *summary*, not call resolution). Note
this explicitly in the PR description. Optionally route `_call_destination_value`'s constant extraction
through the shared helper **only if** output is provably unchanged; otherwise skip to avoid scope creep.

### Step 5 — Tests (all in `tests/test_taint_engine.py`, import-free)
New unit tests for the resolver: direct-const; import-name-before-`.constant`; `resolve_map` precedence;
value-set (constant / in-set / lookup-table — reuse `FPVS`); thunk follow + self-loop guard; graceful
degradation when `bv` lacks symbol APIs. Confirm existing forward-taint tests and `test_bridge.py`
`resolved_calls`/`function_evidence`/`pointer_table`/thunk tests stay green.

---

## PR2 — Unify the backward slicers on `taint_engine.backward` (Part 1)

Depends on PR1 (callee descent calls `resolve_call_target`). The engine's `backward()` stays the base;
it gains trace's two behaviors behind flags, and `bn trace` becomes a thin frontend whose output the
bridge reshapes to today's exact flat format.

### Step 1 — Address-pinned seeding (`_seed_backward`, `taint_engine.py:1310`)
Add a `kind == "call_arg"` branch: `{"kind":"call_arg","address":int,"index":int}`. Find the call insn
whose `int(address) == addr` (reuse the `_instrs`/`_is_call` scan), take `params[index]`, seed from
`expr_reads(param_expr)`. Mirrors the bridge's current seeding at `bridge.py:3943-3975` but inside the
engine.

### Step 2 — Composable interprocedural direction + model gating
Add keyword flags to `backward()` (1160) and `_backward_slice()` (1195), threaded through:
- `ascend_callers: bool = True` — gates the existing `_continue_into_callers` call (1252).
- `descend_callees: bool = False` — new branch at the CALL-def site (1228).
- `consult_models: bool = True` — gates the `lookup_model` source/origin classification (1231-1242).
- `ip_depth: int` — callee-descent depth budget (trace uses 2).

`bn taint backward` keeps today's defaults. `bn trace` calls with
`descend_callees=True, ascend_callers=False, consult_models=False, ip_depth=ip_depth`.

### Step 3 — Callee descent inside the walk
At the CALL-def branch, when `descend_callees and ip_depth > 0`: resolve the callee via
`resolve_call_target(self.bv, defn, follow_thunks=True).function`, get its return vars via a new
`_callee_return_vars(self, callee)` (port of `bridge.py:_find_return_vars` 3846, using the tolerant
mlil accessor + `op_name` + `follow_thunk`), seed a sub-walk in the callee with `ip_depth-1`, and record
a cross-function boundary. Add a `_call_depth`/visited guard mirroring the bridge's runaway protection.

### Step 4 — Make `bn trace` a thin frontend with a byte-stable adapter
The engine's native output is instruction-centric (`_instr_dict`: `il_index/address/op/il_text/reason`),
while `_render_trace_text` (`formatters.py:1425`) consumes a **variable-centric** flat list keyed on
`ssa_var`, `terminates`, `reason`, `cross_function`, `callee`, `function_context`, plus the result dict
keys `function`, `function_address`, `target_address`, `arg_index`, `view`, `interprocedural`,
`ip_depth`, `truncated`, `step_count`, `trace` (`bridge.py:3984-3995`).

To reproduce that exactly: when called in trace mode, the engine tags each walked step with the
`ssa_var` it resolved and a terminal classification (small additive fields the taint renderer ignores).
The bridge's `_backward_slice` op handler (3908) then:
1. keeps all existing validation/errors (`invalid_max_depth`, `invalid_view`, `no_ssa`, `no_ssa_trace`,
   `instruction_not_found`, `no_params`, `invalid_arg_index`) — these are pinned by `test_bridge.py`
   and `test_cli.py`;
2. builds a `call_arg` locator and calls `engine.backward(...)` with the trace-mode flags;
3. flattens the result into today's result dict, mapping engine origin kinds → trace terminal reasons
   (parameter/entry → `function_parameter_or_global`; call/source/indirect → `call_or_jump_boundary`;
   load → `memory_load`; callee entry → `cross_function` with `callee`/`function_context`).

Then delete `_build_backward_trace` (3632). **Fallback if the var-centric reshape proves too lossy:**
keep a slim trace-specific walker in `bridge.py` that reuses PR1's shared resolver + the new
`_callee_return_vars` (i.e. Part 1 degrades to "share the resolver, keep two walks"). Decide during
implementation based on whether the byte-diff in Step 6 is clean.

### Step 5 — Keep dispatch + locks unchanged
`bn trace` keeps routing to op `backward_slice`; `bn taint backward` to op `taint`. **Do not** collapse
trace into the `taint` op (`test_cli.py` asserts `op == "backward_slice"`). Both stay in
`READ_LOCKED_OPS` (`bridge.py:186,194`).

### Step 6 — Tests
- New import-free unit tests in `tests/test_taint_engine.py`: `call_arg` seeding; callee descent reaching
  a return source; `descend_callees=False` stops at the call boundary; `consult_models=False` skips
  source classification; `ip_depth` bound. Reuse the `use_len`/`handler`/`recv` fixture pattern
  (`test_taint_engine.py:828-931`).
- Existing oracles must stay green **unchanged**: `test_bridge.py` backward_slice tests (incl.
  interprocedural callee follow + llil rejection, ~3322-3522), `test_cli.py` trace render tests
  (~3374-3517), `test_taint_engine.py` backward tests (828-947).

### Step 7 — Cleanup tail
Remove now-dead bridge helpers (`_resolve_thunk`, `_extract_dest_address`, `_find_return_vars`, and
`_build_backward_trace`) once the engine reproduces them; `grep` first — `_ssa_vars_from` is used at
3725/3734/3859/3869/3873 and may stay. Keep `_resolve_callee` as a delegating shim if any other caller
references it (grep shows only trace uses it today).

---

## Verification

Run after each PR:

```bash
uv run pytest tests/test_taint_engine.py tests/test_bridge.py tests/test_cli.py -q
uv run pytest tests/test_taint_integration.py -q   # corpus/BN-gated; run if a fixture binary is available
```

**Equivalence (the real proof `bn trace` is unchanged):** on a real binary, before vs. after, diff stdout
for both text and `--format json`:

```bash
bn trace <fn> <call_addr> --arg N
bn trace <fn> <call_addr> --arg N --interprocedural --ip-depth 2
bn taint backward -f <fn> --sink arg:<callee>:<n>
```

The interprocedural trace case is the key oracle for callee-descent equivalence; expect byte-identical
output. The taint backward case proves the added flags didn't regress the default path.

---

## Risks & PR split

- **Highest risk: `bn trace` output shape.** `_render_trace_text` reads exact per-step fields and reason
  strings. Mitigated by trace-mode step decoration + the byte-diff in Verification, with the Step 4
  fallback if reshape is lossy.
- **Zero-import invariant** for `taint_engine.py` — enforced by the tolerant accessor + symbol-API guards;
  the import-free `test_taint_engine.py` suite is the tripwire.
- **Don't over-build the resolver:** evidence's `_normalize_code_pointer`/`_function_thunk_summary` are not
  call resolution and stay separate.
- **Don't merge dispatch ops or change lock sets.**
- **PR1** (resolver) is self-contained with no output changes and lands first. **PR2** (slicers) depends on
  it. Keep PR2's dead-code removal as its tail (or a tiny PR3) to keep migration diffs reviewable.
