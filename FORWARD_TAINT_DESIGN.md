# FORWARD_TAINT_DESIGN — honesty + precision for `bn taint forward` (#8 + #5)

**Status:** ✅ IMPLEMENTED (#146/#147 — frontier-leaf honesty, additive `by_source`, per-callsite
re-run, T1/T2/T3; extended by #157/#159 — `call:`/`model:` source presets + read-bounded length
class). Retained as the original design record; the shipped behavior lives in
`plugin/bn_agent_bridge/taint_engine.py`, exercised by `tests/taint_corpus/`.
**Source:** Cluster D of the dogfood flywheel (see [[DOGFOOD_FLYWHEEL.md]] §12/§13). The two findings
are real on current `m4ul3r` (HEAD), blind-verified at disasm level on a real firmware target.
**Sanitization:** examples use generic names (`recv`/`ipc_read`/`parse_event`, mock addresses). The
real firmware identifiers stay in-session only.

---

## 1. Scope (locked)

| | Decision |
|--|----------|
| **In scope** | `bn taint forward` only — #8 (frontier honesty) + #5 (per-source attribution). Both extend the forward-taint **result contract**, so one coherent change. |
| **#5 mechanism** | **Per-callsite re-run** (reuse the single-source propagation N times), NOT taint-provenance labels. |
| **#5 result** | **Additive `by_source`** — top-level `reached_sinks`/`leaves` stay as the union (back-compat); a new `by_source` map adds the per-callsite breakdown. |
| **CLI surface** | **None added.** Both fixes are invisible precision (user memory: "prefer invisible precision over new CLI surface"). |
| **Out of scope** | Trace items #6 (output-pointer hint) / #15 (`--arg` labeling) — separable, smaller, a later pass. Provenance-label propagation. Any new flags/locator syntax. |

## 2. Problem

Two ways `bn taint forward` currently misleads, both verified on a real aarch64 firmware service:

- **#8 — false "no sinks reached" (dangerous false negative).** Forward taint from a `recvfrom`
  buffer printed *"no sinks reached by tainted data"* — a clean bill of health — although the buffer
  (and an underflowable length derived from `recvfrom`'s return) flows directly into an **in-binary
  parser that isn't in the sink-model DB**. The flow vanished silently (`reached_sinks:[]`,
  `leaves:[]`, `max_depth:0`). An analyst trusting that "all clear" misses a live bug.
- **#5 — source conflation.** `--source arg:recv:1` on a function with N callsites of `recv` seeds
  from **all** of them into one merged propagation (`assumptions:["N callsites of recv; seeded from
  all"]`), so N distinct buffers collapse into one verdict — and the one interesting buffer's flow
  can't be isolated. The workaround today is manual `bn trace <addr>` + reading decompilation.

## 3. Design — make the result tell the truth about where taint went

Forward taint already maintains a `leaves`/`assumptions` honesty channel and **already records
indirect calls and unmodeled _external_ calls as leaves** (`taint_engine.py:1486`, `:1558`). The
design extends that existing, proven pattern; it does not invent a new subsystem.

### 3a. #8 — frontier leaves for unmodeled in-binary callees

When tainted data reaches a `call` to an **in-binary callee that has no sink model and is not
recursed into** (depth bound, recursion cycle, or unanalyzable body), record a leaf instead of
dropping the flow:

```jsonc
// appended to leaves[]
{ "kind": "unmodeled_callee",
  "address": "0x43e9a0",                  // the call site
  "callee": {"name": "parse_event", "address": "0x43cac0"},
  "tainted_args": [1, 2],                  // which arg positions carried taint
  "note": "tainted data passed to in-binary callee with no model; investigate or raise --depth" }
```

Terminal-message change: when `reached_sinks` is empty but `leaves` is non-empty, the text/JSON no
longer says a bare *"no sinks reached"* — it says *"no modeled sink reached; N tainted frontier(s) —
see leaves"*. (The "honest leaf" channel that backward taint and the indirect/external cases already
use, now applied to the in-binary frontier case.)

### 3b. #5 — per-source attribution via per-callsite re-run

Give the engine an **internal** "seed only this callsite" capability (no CLI flag). The forward
orchestration then:

1. Resolves the source locator's callsites as today (`_find_callsites`).
2. If there is **one** callsite → behaves exactly as now (no `by_source`, no extra cost).
3. If there are **N>1** → runs the existing single-source propagation **once per callsite**, each
   seeded from exactly that call address, and tags the result with the seeding address.

Results merge into the existing top-level fields as the **union** (back-compat) plus a new additive
breakdown:

```jsonc
{
  "reached_sinks": [ /* union across callsites — unchanged shape */ ],
  "leaves":        [ /* union across callsites — unchanged shape */ ],
  "assumptions":   [ /* incl. a per-callsite note instead of "seeded from all" */ ],
  "by_source": {
    "0x440b98": { "reached_sinks": [ … ], "leaves": [ … ] },
    "0x440bf0": { "reached_sinks": [],    "leaves": [ {unmodeled_callee …} ] },
    "0x440cf4": { … }, "0x440d30": { … }
  },
  "stats": { /* functions_visited/max_depth = max across runs; see §5 */ }
}
```

The merged *"no sinks reached"* failure mode disappears: each callsite's flow is reported on its own,
so the one buffer that flows into a parser (a `#8` frontier leaf) is no longer masked by three that
don't.

**Cost:** N propagations instead of 1 when the source callee has N callsites. Bounded by callsite
count × `max_depth`; only triggered for multi-callsite sources. Acceptable for the precision gain;
single-callsite (the common case) is unchanged. Note the multiplier in `assumptions`.

## 4. New/changed result contract (summary)

- `leaves[]` gains `kind:"unmodeled_callee"` entries (additive — existing leaf kinds unchanged).
- New top-level `by_source: {<call_addr>: {reached_sinks, leaves}}` (present only when N>1 callsites).
- Terminal message distinguishes "no flow at all" from "flow stopped at unmodeled frontier(s)".
- `reached_sinks`/`leaves`/`assumptions` top-level semantics (union) unchanged → **non-breaking**
  for existing consumers; `by_source` is purely additive.

## 5. Implementation map (`plugin/bn_agent_bridge/taint_engine.py`)

- `forward()` (~`:964`) — orchestration: after `_find_callsites`, branch on N; loop per callsite for
  N>1; assemble `by_source` + union. **`stats` merge rule (pinned):** `max_depth` = max across the
  per-callsite runs; `functions_visited` = size of the **union** of functions visited across runs
  (a function visited from two callsites counts once); any other counters sum unless they are sets,
  in which case union.
- `_seed_forward()` (~`:1732`, the `"seeded from all"` assumption at `:1755`) — add an optional
  `only_callsite_addr` param so a run seeds from exactly one callsite; the `"seeded from all"`
  assumption becomes a per-callsite note (or drops when attributed).
- Frontier-leaf recording (#8) — at the in-binary `call` handling (near the external-leaf path
  `:1558` and the indirect-leaf path `:1486`): when a tainted arg flows into an in-binary callee that
  is not recursed into / has no model, append the `unmodeled_callee` leaf. Reuse the existing
  `leaves`/`add_assumption` plumbing; do not duplicate it.
- CLI: **no change** to `src/bn/commands/*` (no new args). The bridge op params are unchanged.

## 6. Renderers (`src/bn/formatters.py`)

- Forward-taint text renderer: render `by_source` as per-callsite sections when present; render
  `unmodeled_callee` leaves with the call→callee hand-off line; replace the bare "no sinks reached"
  with the frontier-aware message. Keep the bare/legacy shape working (back-compat fallback).

## 7. Verification

1. **TDD (mocked BN), engine-level:**
   - #8: a fixture where a tainted arg flows into an in-binary callee with no model and depth is
     bounded → assert a `kind:"unmodeled_callee"` leaf with the right `address`/`callee`/`tainted_args`
     and that the terminal message is frontier-aware (not "no sinks reached").
   - #5: a fixture function with ≥2 callsites of the source callee where only one buffer reaches a
     sink/frontier → assert `by_source` has a per-callsite entry for each and the interesting callsite
     is attributed; single-callsite fixture → assert NO `by_source` and unchanged behavior.
   - Back-compat: top-level `reached_sinks`/`leaves` remain the union; existing forward-taint tests
     stay green.
2. **Live re-use gate (`dfwa3`, current-code analyzed bridge):** re-run the exact flow that exposed
   #8 (a `recvfrom`/`recv` buffer → unmodeled in-binary parser) and confirm it now reports a frontier
   leaf instead of "no sinks reached"; run a known multi-callsite source and confirm per-callsite
   attribution. (Must confirm `stale_plugin_code == False` first — see [[stale-bridge-dogfood-gotcha]].)
3. No CLI/arg changes → CLI test surface only grows by renderer tests.

## 8. Out of scope (and why)

- **Trace #6 / #15** — different op (`bn trace`), separable; a later small pass.
- **Provenance-label propagation** for #5 — more precise on shared-downstream flows but invasive
  across the whole propagation; rejected in favor of the per-callsite re-run (simpler, low-risk,
  reuses the propagation unchanged).
- **Any new CLI flag or locator syntax** — explicitly avoided (invisible precision).
- **Soundness rework** — bn-taint stays a may-analysis, not a proof; this only fixes *honesty*
  (report frontiers) and *attribution* (don't conflate sources), not propagation completeness.
