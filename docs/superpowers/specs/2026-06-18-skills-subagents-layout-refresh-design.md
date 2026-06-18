# Skills / Subagents Layout — Refresh + Route

- **Date:** 2026-06-18
- **Status:** Approved (design) — ready for implementation plan
- **Scope (locked):** Keep the existing structure. Skill/agent **text only** — no `src/bn`
  code changes. All routing/guidance is advisory and trigger-based, never a mandatory gate.

## Motivation

A two-arm "layout vibe" dogfood was run against a single rich C++ aarch64 service library
(~4.8k functions, dense RTTI/vtables, PIE) to decide how the skills and subagents should be
laid out. Arm A drove the skills directly inline; Arm B dispatched the `bn-re` and `bn-vr`
subagents read-only into their own dedicated headless instances. (Target identity is
deliberately omitted — see the repo's sanitize rule.)

Findings:

1. **The current cut is correct.** One capability skill (`bn`) + two methodology skills
   (`bn-re`, `bn-vr`) + two harness subagents. Neither arm wanted a merge or a finer split.
2. **Subagent isolation is the always-pays benefit.** The two subagents burned ~54k and ~50k
   tokens internally and returned ~1.5k-token distilled artifacts; the decompiler/class/xref
   flood never reached the orchestrator.
3. **The methodology earns its place via two specific, reconfirmed mechanisms:**
   - `bn-re`: the conditional hidden-surface sweep's **skip** decision — it recognised
     virtual-dispatch C++ with no data-only dispatch table and deliberately did *not* hunt one,
     conserving budget.
   - `bn-vr`: forced **sink-enumeration-by-behavior + source→sink tracing** — the import table
     exposed a single bounded sink, which naive triage would call clean; the methodology forced
     the audit onto the locally-defined copy callsites and serializer and several taint slices,
     turning a probable false all-clear into a verified (bounded-coverage) clean result.

The gaps are **not structural**:

| Gap | Evidence | Fix class |
|-----|----------|-----------|
| **Freshness** | `class` appears 1× in `bn-re/SKILL.md` vs 5× in `bn/SKILL.md`; the methodology predates the C++ class lens (#205) and the matured `evidence` family. The subagents used the class lens anyway — via the `bn` skill + competence, not the methodology. | text |
| **Passive routing** | `bn` only points at the methodology skills with a passive blockquote; the RE→VR handoff (class-lens → directive parser → "this is an audit target") exists only in the operator's head. | text |
| **Product polish** | `bn class list --no-stl` leaves template-explosion library types unfolded and renders them raw-mangled while siblings demangle. | out of scope — separate ticket |
| **Budget tension (VR)** | ~15 commands could not do both exhaustive sink triage and vtable-entry-surface enumeration. | guidance, not new surface |

## Decision

Adopt the **refresh + route** approach: keep the structure unchanged; update the methodology
skills and agent bodies to (a) absorb the newer command surface and (b) make routing active.
Rejected alternatives: adding a specialist skill/subagent (no evidence finer cuts help; added
enforcement is a measured tax) and merging the methodology skills (the RE/VR split is correctly
drawn — `taint` 0× in `bn-re` vs 24× in `bn-vr`; "sweep-skip" and "sink-enum" are distinct
mandates).

## Non-goals

- No new skills or subagents.
- No merging or splitting of existing skills/agents.
- No `src/bn` / bridge code changes (no CLI hand-off hint in this spec; deferred follow-up).
- No wholesale methodology rewrite — targeted edits only.

## Changes

### 1. `skills/bn-re/SKILL.md`

1. **Reframe the `description`** to lead with the differentiated payoff — *conditional
   hidden-surface sweep (and its skip decision) + C++ class-lens triage* — rather than the
   generic "triage / function identification / type recovery" wording that overlaps with `bn`
   and reads as optional background.
2. **Add a first-class class-lens orientation step** in the triage/identification section: on
   any target with RTTI or demangled C++ symbols, run `bn class list --no-stl` *first* (clusters
   functions by class, separates public API from implementation, surfaces vtables and
   inheritance), then `bn class show <Name>` to drill into a class. This supersedes hand-rolled
   symbol grepping for orientation.
3. **Fold `bn class list` into the conditional-sweep cheap-signature check** — the lens already
   reports the RTTI/ctor cluster count, which *is* the signal that decides whether to run or
   skip the full sweep.

### 2. `skills/bn-vr/SKILL.md`

1. **Reframe the `description`** to lead with the proven payoff: *forced
   sink-enumeration-by-behavior and source→sink tracing are the steps that stop a bare agent's
   false all-clear when the import table looks empty.*
2. **Add the class lens as an attack-surface accelerator**: on a C++ target, use
   `bn class list` / `bn class show` to locate directive/parse/dispatch handlers quickly before
   sink tracing.

### 3. `skills/bn/SKILL.md` — active routing

Replace the passive "for RE see `bn-re`; for VR see `bn-vr`" blockquote with an **active routing
decision block** near the top: open-ended understand/map → invoke the `bn-re` skill, or
**dispatch the `bn-re` subagent for a long survey** (isolation is the proven win); find
bugs/audit → `bn-vr` likewise. Framing: *the methodology tells you WHAT to do and why; this
skill is the HOW.* Keep it short and trigger-based.

### 4. Agent bodies — `bn-re.md`, `bn-vr.md`

These currently live at `~/.claude/agents/bn-re.md` and `bn-vr.md`.

1. **Swap the cheap-signature check** from the raw `bn strings --regex '_ZTV|_ZTI|_ZN[0-9]'`
   scan to `bn class list --no-stl` (purpose-built, cleaner signal for the same decision).
2. **`bn-re` return contract gains a "VR handoff" line:** attacker-facing parse/dispatch
   handlers the survey surfaces are flagged explicitly as a `bn-vr` worklist, so the RE pass
   actively *produces* the audit targets instead of leaving the handoff implicit.
3. The subagent `description`s already lead with isolation ("keeps the token-flood out of the
   orchestrator") and the conditional sweep is already softened — leave those as-is.

### Agent-definition versioning (open item for the plan)

The two agent `.md` files currently live only under `~/.claude/agents/` (hand-installed), not in
the repo, while a sibling distribution card already exists at `skills/bn/agents/openai.yaml`.
The plan must decide whether to **canonicalize the agent defs in-repo** (e.g. under
`skills/bn/agents/`) so they are versioned alongside the skills, and how they are installed.
Wiring `bn skill install` to deploy them is a `src/bn` change and therefore **deferred** —
this spec edits the live agent files and records the versioning gap; the plan resolves the
canonical-location question without adding install code.

## Cross-cutting principle

Every addition is **advisory and trigger-based** ("reach for X when signature Y"), matching the
existing conditional-sweep philosophy. No step becomes a mandatory gate — the deep A/B measured
unconditional enforcement as a budget tax that grows as real surface shrinks.

## Verification

No code changes, so "green" is the existing suite staying green plus a behavioral re-check:

1. **Regression sanity:** `uv run pytest` stays green (nothing here touches code; this only
   guards against an accidental edit).
2. **Coverage re-grep:** after the edits, `class` appears as a first-class step in
   `bn-re/SKILL.md` (not a lone cross-link), and both methodology `description`s lead with their
   differentiated payoff.
3. **Behavioral re-dogfood (the real check):** dispatch both subagents read-only at a *fresh*
   rich C++ target and confirm (a) the cheap-signature check now uses the class lens, (b) the
   `bn-re` return includes a populated VR-handoff worklist, and (c) the `bn` skill's routing
   block points an operator at the right methodology/subagent without being asked. Compare the
   distilled artifacts against this dogfood's baseline (`.dogfood/vibe/`).

## Spun-off follow-up (not in this spec)

File a separate product ticket against `src/bn` for the `bn class list --no-stl` rendering bug:
template-instantiation library types (e.g. deeply-nested standard-library wrappers) are not
folded by `--no-stl` and are printed raw-mangled while sibling domain classes demangle.
