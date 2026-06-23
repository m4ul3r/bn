# DOGFOOD_FLYWHEEL — improvement-led dogfooding of `bn` against real firmware

**Status:** Design spec, approved interactively. Cycle 1 not yet run.
**Goal type:** Standing improvement-led flywheel. Real VR work is the *means*; a measurably better `bn` is the *end*.
**Sensitivity:** Cycle 1 drives a **real firmware** target. All names/addresses/paths/decompiled
output from that target are sensitive. This doc and every shared artifact use the alias **svc-A**
and mock addresses. In-session reasoning may use the real identifiers; nothing else may.

---

## 1. Locked decisions

| Axis | Decision |
|------|----------|
| **Objective** | Improvement-led flywheel — success = real use got smoother, measured on the binary; not "issues closed". |
| **Governance** | Discover → triage (the one human gate) → autonomous execution. Calibrate the "worth fixing" bar once, then run hands-off under the verify contract. |
| **Target** | A real firmware service daemon (**svc-A**), driven **both** as its analyzed `.master.bndb` *and* as the raw (unanalyzed) binary. The BNDB exercises the mastered-DB workflow; the raw load exercises the cold-load / `--quick` / `refresh` / `orient` surface. |
| **Discovery method** | Multi-agent dogfood workflow (fan out `bn-vr` agents across attack surfaces). User opted in explicitly. |
| **Live bridge** | **Dedicated, current-code instance** (`dfwa1`), loading the same `.master.bndb`. The pre-existing campaign instance is **stale** (its process predates current `m4ul3r` code) and must **not** be the dogfood target — see §11. Do not restart the campaign bridge (it holds in-progress analyzed state). |

---

## 2. The loop

```
USE  →  CAPTURE  →  TRIAGE (you)  →  IMPROVE  →  RE-USE
```

- **USE** — run a genuine VR pass on svc-A using the `bn-vr` methodology: attack-surface map →
  sink enumeration → source→sink taint. Real work, not synthetic pokes.
- **CAPTURE** — log every friction point the moment it occurs: shell-loop fallbacks, inconsistent
  JSON shapes, confusing/empty errors, missing visibility, anything that required skill-doc
  knowledge the tool itself should have surfaced.
- **TRIAGE (you)** — I return a ranked, proven friction inventory; we pick the worth-it subset.
  This is the only human gate.
- **IMPROVE** — each agreed item → fix root cause (no NOP/stub/hardcode) → `uv run pytest` green →
  verified **on the live target that exposed it** → PR-per-fix to `m4ul3r` → merge on green.
- **RE-USE** — re-run the affected step with the improved `bn` and confirm the friction is gone.
  This closes the loop; it is what makes the work a flywheel rather than a backlog.

## 3. Cycle-1 discovery pass (the multi-agent dogfood)

**Warm-up (solo, me):** a short orientation pass on svc-A so I understand the binary before fan-out
— target info, imports summary, sections, entry surface, a couple of decompiles. Both the BNDB and
the raw load, so the cold-load/orient friction is captured first-hand.

**Fan-out (`bn-vr` subagents, one per attack surface):** each agent is told to (a) do real VR on its
surface and (b) keep a friction log in a fixed schema. Candidate surfaces (finalized after warm-up):
network/IPC input parsing, command/dispatch handling, file/config parsing, auth/privilege
boundaries, memory-management sinks (the `memcpy`/`strcpy`/`sprintf`/`recv` family). Each agent works
the source→sink chain and records where `bn` helped vs. fought it.

**Dual-load coverage:** at least one agent runs against the **raw** binary (load → `--quick` →
`refresh` → `orient` arc) and at least one against the **`.master.bndb`** (mastered-DB arc), so both
friction surfaces are represented.

**Orchestration:** I run the fan-out as a Workflow, then dedupe and rank the merged friction logs.
Decompiler/xref/taint output stays in the agents' windows; only the distilled friction rows return.

## 4. The "proven real" bar (applied before triage)

Every inventory row must clear **both** gates or it is dropped / explicitly flagged unverified:

1. **Base-tree repro** — reproduce against the base `m4ul3r` tree. (This gate is why the batch-comment
   complaint from a recent external review would have been caught: it was already fixed in the tree.)
2. **Blind verify** — a second agent, given only the claim, independently confirms it is a real tool
   defect and not misuse of the CLI.

## 5. Friction inventory schema (the triage artifact)

A ranked table, fully scrubbed (svc-A, mock addresses). One row per friction:

| field | meaning |
|-------|---------|
| `severity` | how much it taxes *real* use (blocks / detours / annoys) |
| `friction` | one-line description |
| `expected` | what a clean tool would have done |
| `actual` | what `bn` did |
| `repro` | minimal sanitized command sequence |
| `proposed_fix` | the change |
| `kind` | bug-fix vs. feature |
| `effort` | rough size |
| `verified` | base-repro ✓ + blind-verify ✓ |

Severity ranks by real-use tax, not by how easy the fix is.

## 6. Autonomous execution contract (post-triage)

For each agreed item: branch off `m4ul3r` → fix root cause → **green** → **re-use gate**
(confirm the friction is gone on the live svc-A target) → small, reviewable, **sanitized** PR-per-fix
→ merge on green. Progress reported in batches. I stop only for a genuine fork — a feature that needs
a design decision, or ambiguous severity. Mirrors the established `/goal` contract.

**"Green" is defined precisely** (no CI in this repo): local `uv run pytest` passes **and** the live
re-use gate passes against svc-A. Unit tests run against mocked `binaryninja`, which can give false
assurance about real BN API behaviour — so a fix is not "done" on pytest alone; the live re-use gate
is mandatory, not optional.

## 7. Sensitivity discipline (enforced on every artifact)

Scrub before anything leaves the session — inventory, issues, PRs, commits, this doc:
binary/service names → **svc-A** (etc.); product/subsystem path tokens; real symbol names;
concrete addresses → mock; verbatim decompiled output → realistic mock that stands alone.
In-session reasoning is the only place real identifiers may appear.

## 8. Success metric (cycle 1)

Cycle 1 succeeds when **all** hold:

1. A ranked, proven (base-repro + blind-verify) friction inventory exists.
2. The triaged subset is merged to `m4ul3r` and green.
3. The top friction items are **demonstrably gone** on re-use against the live svc-A target.

Not "N issues closed" — "the next stretch of real VR on svc-A is measurably smoother."

## 9. Out of scope (YAGNI)

- Cross-binary / cross-instance automation.
- Speculative feature builds (e.g. `IDEA_001`'s fan-out / composite ops) **unless** the pass
  independently surfaces that exact friction with a repro.
- The campaign's actual firmware findings. The *bug in the firmware* is the user's; this goal only
  cares about `bn`'s friction while analyzing it.
- Auto-committing this doc or auto-filing issues without the triage gate.

---

## 10. Known seeds (already observed this session — must still clear the proven-real bar)

These are starting hypotheses, **not** confirmed inventory rows. Each re-runs through §4:

- **JSON shape inconsistency** — `function list`/`search` return an object envelope
  (`{functions,total,offset,limit,returned,has_more}`); `callsites` returns a bare top-level array.
  Recent paging-envelope work (#122/#130) covered strings/imports/sections + function list/search but
  not callsites; issue **#131** tracks the same gap for `types` and `comment list`. Fix should extend
  the existing envelope convention (domain key, not a generic `items`).
- **Analysis-state visibility** — `target info` already exposes `analyzed` + `analysis_state`
  ("quick"/"full"); `target list` rows do **not**. Per-capability map (`strings_available`,
  `needs_refresh_for`) does not exist. Partial gap, not a total one.
- **Stress harness reproducibility** — `tests/stress/run_stress.sh` needs `tests/fixtures/`, which is
  `.gitignore`d with no committed Makefile to regenerate it. Deliberate "don't commit binaries"
  choice + missing build recipe.

---

## 11. Operating constraint — dogfood only against a CURRENT-code bridge

**Lesson from the warm-up (two phantom findings caught):** a long-running bridge process keeps
executing the code it imported at start-up. If `m4ul3r` advances after the process started, the
process serves **stale behaviour** silently — e.g. a pre-envelope `imports` returning a bare list —
while the on-disk plugin is current. Any tool-behaviour observation from such a bridge is invalid.

**`bn doctor` detects this correctly**, but the flag is **per-instance**, not top-level:
`instances[].stale_plugin_code` compares the running process's embedded `plugin_build_id` against the
installed-file build_id. Text output says `stale: loaded plugin code does not match installed plugin
file`. (Two would-be findings — "imports lost its envelope" and "doctor has a staleness blind spot" —
were both phantoms: the first was stale-bridge output, the second was me reading only top-level keys.)

**Rule:** before trusting *any* friction observation, confirm the target bridge has
`stale_plugin_code == False`. The structural facts about the binary (arch, imports set, sections,
call graph) are bridge-version-independent and safe to read anywhere; **output shaping, errors, and
envelopes are not** — read those only from a current-code bridge.

---

## 12. Cycle-1 results (run 2026-06-13, dfwa1 = HEAD `7ec…`)

5 `bn-vr` agents · 28 raw friction → 17 confirmed / 11 rejected by blind-verify. Verify gate caught
1 `stale_or_fixed` (the imports "100 shown" phantom, independently re-caught), 4 `tool_misuse`, and
held 6 as `needs_design`. Triage: **A+B+C autonomous, D design-first.** Full raw result lived at the
session task artifact; firmware vuln leads kept out of this doc (sensitive — held for the user).

**Cluster A — quick-load honesty (8, mostly bug, S–M) — AUTONOMOUS.** In `--quick` mode (analysis
not run) analysis-dependent ops mislead with exit 0:
1. `xrefs` → silent empty `0/0` (reads as "no refs", not "not analyzed")
2. `function info` → `size:0` + bogus signature, no quick marker
3. `taint backward` → "no call to <sink> found; check the name" (misdiagnosis)
4. `callsites` → "Function not found: <name>. Did you mean…?" (blames a typo)
5. `target list` → no `analysis_state` per-target (text+JSON)
6. `target info` text → omits quick/unanalyzed state (only JSON has it)
7. `strings` → leads with generic `.dynsym noise` tip *before* the quick refusal
8. quick-blocked commands exit 0 → scripted cold agent can't detect the unfulfilled request
→ Plan: shared quick-state guard across the read ops; propagate `analysis_state` to `target list` +
text renderers; non-zero exit / explicit marker for quick-blocked. ~2–3 PRs.

**Cluster B — JSON consistency (2, bug, S) — AUTONOMOUS.** `callsites` is a bare array (no envelope);
paged key is `items` (imports/strings/sections) vs `functions` (function list/search). Ties #131.

**Cluster C — `xrefs <hex>` offset hint (1, S) — AUTONOMOUS.** 0-match on a struct-field offset given
as a raw address → add a hint. (#9 `callsites hlil_statement` null-for-wrappers: dropped, BN limit.)

**Cluster D — taint/trace precision (4, feature, M) — DESIGN-FIRST (not auto-merge):**
- forward taint seeds from ALL callsites of a callee (no per-callsite/address scoping)
- `trace --arg` dead-ends on output-pointer-written args with no callee hint
- forward taint false "no sinks reached" on recv→in-binary-parser (likely entangled with the above)
- `--arg N` index→C-arg/register mapping is unlabeled

## 13. Execution log

- **PR1 — `quick-state-honesty` — MERGED (#132, squash → `m4ul3r`).** Cluster A core (items 1–4):
  `xrefs`/`function info`/`callsites`/`taint` now refuse on a `--quick` view via a shared
  `bridge_state.require_analysis(bv, what)` guard (honest "loaded with --quick … run `bn refresh`"
  + non-zero exit) instead of misleading empties/typo-blame. TDD (4 RED→GREEN), full suite
  679 passed/3 skipped, live re-use gate on a fresh `--quick` raw load (exit 2; `sections`/`imports`
  unaffected). Note: the non-zero exit also resolves the exit-0 concern (item 8) **for these ops** —
  PR3 shrinks to any quick-blocked op that still returns empty+exit-0 rather than raising.
- **PR2 — `quick-state-visibility` — MERGED (#133, squash → `m4ul3r`).** Items 10 + 12:
  `TargetManager.refresh()` sets `analyzed`/`analysis_state` on every `target list` row (text+JSON);
  `_render_target_summary` appends `[not analyzed]` so `target list`/`target info` text flag a quick
  view. TDD (2 RED→GREEN), full suite 681 passed/3 skipped, live re-use gate on fresh `--quick` load.
- **PR5 — `xrefs-offset-hint` — MERGED (#134, squash → `m4ul3r`).** Item 7: `_maybe_offset_hint`
  nudges toward `--field` when `xrefs <bare-number < 0x10000>` matches nothing (offset misread as
  address). TDD (positive+negative), full suite 683 passed, live gate (0x308 hints, 0x999999 silent).
- **PR3 — `strings-tip-order` — MERGED (#135, squash → `m4ul3r`).** Item 13: the unfiltered
  noise-tip now prints AFTER a successful dump, not before the request — so a `--quick` refusal is
  no longer preceded/buried by irrelevant advice. TDD, full suite 684 passed, live gate (quick view
  shows refusal only). **Item 8 needed no work** — audit showed `decompile`/`il`/`disasm`/`proto`/
  `local` correctly work on-demand in quick mode (exit 0 is right); the only truly-blocked ops were
  the 4 PR1 already guards (exit 2). So Cluster A is fully closed except item 11 (→ PR4).
- **PR4 — `json-envelope-parity` — MERGED (#136, squash → `m4ul3r`; closes #131).** `feat(json)!`:
  `types`/`comment list`/`callsites` now return the `{items,total,…}` envelope (was bare lists), via
  `_paged_list_result` + shared paged renderer/footer; CLI switched to `paged_spill=True`. Item 11 +
  the #131 backlog (types/comment-list) done in one breaking PR. TDD (8 callsites tests updated to
  unwrap+assert envelope; new comment-list envelope test), full suite 685 passed, live gate
  (types/comment-list/callsites all enveloped; types footer `// showing 3 of 157 (154 more)`).
- **Design-first (needs user input):** Cluster D — taint/trace precision (inventory items 5,6,8,15);
  item 14 (items-vs-`functions` paged-key unification — a breaking cross-API key decision; this PR
  deliberately did NOT touch `function list/search`'s `functions` key).
  - **Cluster D / forward-taint (#8+#5): DESIGNED — see [[FORWARD_TAINT_DESIGN.md]]** (approved:
    per-callsite re-run + additive `by_source`; #8 = `unmodeled_callee` frontier leaves; zero CLI
    surface; non-breaking). Ready for writing-plans → implementation in a focused session. Trace
    #6/#15 + item 14 still un-designed.

### Cycle-1 scorecard — autonomous queue COMPLETE
17 confirmed → **6 merged PRs** (#132 quick-state honesty · #133 analysis_state visibility · #134
xrefs offset hint · #135 strings tip order · #136 JSON envelope parity / closes #131; item 8 closed
by audit) closing **all autonomously-triaged items (A+B+C)**. Loop proven end-to-end; every merge
passed pytest + a live re-use gate on the real `--quick`/analyzed bridge. Only **design-first** work
remains (Cluster D taint/trace, item 14 key-unify). Dogfood instances still running: `dfwa1`/`dfwa3`
(analyzed), `dfwq`/`dfwq2` (quick) — safe to stop; campaign bridge `c111d` untouched.
