# ultracode — design spec

**Date:** 2026-07-08
**Artifact:** `.claude/workflows/ultracode.js` (Workflow-tool script; committed; carries zero target data)
**Status:** design approved pending user review

## 1. Purpose

A reusable, committed workflow that drives the `bn` / `bn-re` / `bn-vr` skills through a
disciplined RE→VR pipeline, with an independent reviewer after each working phase that
audits whether the previous agent used the bn tooling *well* (methodology + soundness,
correctness-weighted) and — for the VR phase — adversarially verifies each finding.

The `.bndb` is the hand-off medium: the RE agent **enriches** a shared database (renames,
prototypes, struct fields, comments) and the VR agent audits that enriched version. The
reviewer diffs the BNDB to confirm the RE agent actually left recovered state behind, not
just a chat report.

## 2. Goals / Non-goals

**Goals**
- One-command, repeatable "reverse it, then hunt bugs in it, and check the tool was used
  well" over an arbitrary binary.
- Enforce the skills' own hard-won gotchas (pipe/spill trap, disasm-confirm bounds, right
  lane, one-shot digests, fan-out instance hygiene) via the reviewer rubric.
- Persist recovered state (saved BNDB) + a durable local report for follow-up.

**Non-goals**
- Not fan-out breadth (chosen shape is a **linear pipeline**; each agent may fan out its
  own reads internally, but the pipeline is a single chain).
- Not committing any target-specific output (reports are git-ignored — real firmware data).
- Not a general vuln scanner: scope is one binary + optional focus per run.

## 3. Invocation contract

```
Workflow({ name: 'ultracode', args: {
  binary: '<path>',      // required
  focus?: '<free-form hint>',  // e.g. 'httpd request path', 'IPC handlers', a fn name; empty = broad
  instance?: '<id>',     // optional override; default derived from binary basename
  runid?: '<string>'     // optional label for the report filename; default = binary basename
}})
```

`Date.now()` / `Math.random()` are unavailable in workflow scripts, so uniqueness comes
from `args.instance` / `args.runid` when supplied; otherwise the sanitized binary basename
is used (documented collision caveat: two concurrent runs on the same binary must pass
distinct `instance`).

## 4. Pipeline (linear chain)

| Stage | agentType | Responsibility | Returns (schema) |
|---|---|---|---|
| 0 · Setup | general-purpose | `bn session start <binary> --instance-id <id>`; `bn evidence orient`; detect arch + **lane** (import-first vs stripped/static by the real tell, not `file`); capture **baseline** counts. | `{arch, lane, baseline:{functions,symbols,comments}, orientation}` |
| 1 · RE | **bn-re** | bn-re methodology scoped to `focus` (or broad). **Enrich the BNDB** (rename handlers, set protos/structs, comment assumptions). Recover hidden surface where warranted. | `{map:[{entry,dispatch,handler,sinks}], enriched:{renames,protos,structs,comments}, hidden_surface, notes}` |
| 2 · review-RE | general-purpose (fresh) | Score RE vs §5 rubric. **Diff BNDB** vs baseline (did symbols/comments grow? high-value moves made?). | `{verdict:'pass'\|'gaps', redo:[...], efficiency_notes}` |
| — | | if `gaps` and not already redone → re-run Stage 1 with `redo` appended (ONCE), then proceed unconditionally | |
| 3 · VR | **bn-vr** | Audit enriched BNDB. Exhaustive sink enum (`taint models --present`), source→sink trace, **manual lanes** (parser-invariant, dest-capacity). | `{findings:[{class,fn,addr,source,sink,path,prelim_confidence,soundness}]}` |
| 4 · review-VR | general-purpose (fresh) | Score VR vs §5 rubric **and adversarially verify each finding** — disasm-confirm the bound, prove attacker control, demand soundness caveat; demote false all-clears. | `{verdict, redo:[...], efficiency_notes, verified:[...], demoted:[...]}` |
| — | | if `gaps` and not already redone → re-run Stage 3 with `redo` (ONCE), then proceed | |
| 5 · Synthesize | general-purpose | `bn save`; assemble + write the report to `.dogfood/audits/<runid>.md`; `bn session stop`. | `{report_path, summary}` |

The workflow returns `{report_path, summary}` to the orchestrator (me) to relay.

## 5. Reviewer rubric (correctness-weighted "both")

Primary (methodology + soundness):
- **Right lane** chosen by the real tell (`imports` non-empty / mostly-named ⇒ import-first;
  empty-imports + overwhelmingly `sub_*` ⇒ stripped/static), not by `file`.
- **Bounds/field-widths disasm-confirmed** before any overflow/off-by-one/truncation claim
  (HLIL flattens `ccmp`/`csel`, aliases hoisted bounds, drops `<<4`).
- **BNDB enriched** (RE): symbol/comment counts materially grew vs baseline; handlers named;
  key structs/protos set — a chat-only report with an untouched BNDB is a fail.
- **Sinks enumerated exhaustively** (VR): `taint models --present`; empty taint is **not**
  an all-clear — parser-invariant + destination-capacity manual lanes were checked.
- **Findings verified** (VR): each has a disasm-confirmed bound, proven attacker control,
  and the `bn taint` soundness caveat.

Secondary (token/process efficiency — flag, don't gate):
- No pipe/spill trap (`bn … | grep` reading the envelope).
- One-shot digests (`evidence orient`/`surface`) used instead of many redundant small reads.
- No repeated re-decompiles of the same function.

Reviewer verdict: `pass` (proceed) or `gaps` (emit a *targeted* redo list; the phase re-runs
once, then the pipeline proceeds regardless — bounded cost).

## 6. Instance / BNDB seam & safety

- **One dedicated instance per run**; every stage's agent is told: pass `--instance <id> -t
  <sel>` on **every** `bn` command, and **never** call `instance use` / `target use`
  (sticky pins are one shared file per repo — fan-out agents clobber each other).
- BNDB state persists in the instance across stages (RE writes → VR reads). `bn save` in
  Stage 5 before `bn session stop`.
- **Sensitivity:** the report contains real target names/addresses → written to the
  git-ignored `.dogfood/audits/` only; **never committed**. The workflow *script* is generic
  and committed. Agents are reminded to keep target data out of any shared artifact.

## 7. Failure handling

- A stage agent that dies → the workflow surfaces it in the summary and stops cleanly
  (still runs Stage 5 `bn session stop` for the instance if reachable).
- Reviewer `gaps` is not a failure — it triggers the single bounded redo.
- Quick-loaded / cache-restored target caveats (from bn-re/bn-vr) are baked into the Setup
  agent prompt (`bn refresh` if `analysis_state != full`; `--no-bndb` note if cache-restored
  and a pristine baseline is wanted).

## 8. Budget / scoping

- `focus` narrows RE + VR; absent, RE covers the surfaced high-value set, VR the top sinks.
- The script guards each phase against `budget.remaining()` so a large firmware target
  cannot run away; caps are logged (`log()`) so silent truncation is visible.

## 9. Files

- **Create:** `.claude/workflows/ultracode.js` (committed), this spec (committed).
- **Runtime-only (git-ignored):** `.dogfood/audits/<runid>.md`, the per-run bn instance,
  the saved `.bndb` (global cache).
