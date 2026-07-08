# ultracode Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `.claude/workflows/ultracode.js`, a linear RE→review→VR→review→synthesize Workflow-tool pipeline over the bn/bn-re/bn-vr skills.

**Architecture:** A single self-contained Workflow-tool script. Six phases run as a linear chain sharing one dedicated bn instance (the `.bndb` is the RE→VR hand-off). A fresh `general-purpose` reviewer runs after each working phase (correctness-weighted rubric, gate + one bounded targeted redo). Stages 1/3 use the `bn-re`/`bn-vr` subagents; stages 0/2/4/5 use `general-purpose`. Output is a git-ignored `.dogfood/audits/<runid>.md` report + a saved BNDB.

**Tech Stack:** Plain JavaScript (Workflow-tool dialect — no TypeScript, no `Date.now()`/`Math.random()`/`new Date()`; runtime globals `agent`/`parallel`/`pipeline`/`phase`/`log`/`args`/`budget`). Validated with `node --check` (syntax) + one live Workflow run.

## Global Constraints

- File is **generic and committed**; it must contain **zero target-specific data** (verbatim from spec §2).
- Reports contain real target names/addresses → written **only** to git-ignored `.dogfood/audits/`; **never committed** (spec §6).
- No `Date.now()`/`Math.random()`/`new Date()` in the script — uniqueness from `args.instance`/`args.runid`, else sanitized binary basename (spec §3).
- Every stage agent prompt must include the fan-out hygiene reminder: pass `--instance <id> -t <sel>` on every bn command; never `instance use`/`target use`/`*clear`; never `bn … | grep` (spill/pipe trap) — write-to-file-then-grep (spec §6, bn skill).
- Redo is **strictly one** bounded targeted retry per working phase, then proceed regardless (spec §4).
- Reviewer stages are **fresh `general-purpose`** agents, independent of the `bn-re`/`bn-vr` agent they grade (spec §4).

---

## File Structure

- **Create:** `.claude/workflows/ultracode.js` — the entire workflow (one file, one responsibility: orchestrate the pipeline).
- **Already git-ignored:** `.dogfood/` (report output tree) — confirm, no new rule needed.
- **Committed alongside:** `.claude/workflows/ultracode.design.md` (spec), `.claude/workflows/ultracode.plan.md` (this plan).

Because the deliverable is a single cohesive file built top-to-bottom, tasks are sections of that file. Each task ends with `node --check` passing; the final task validates end-to-end and commits.

---

### Task 1: Scaffold — meta, args, hygiene, schemas

**Files:**
- Create: `.claude/workflows/ultracode.js`

**Interfaces:**
- Produces: module-scope consts `binary`, `focus`, `base`, `instance`, `runid`, `HYGIENE`, and schema objects `SETUP_SCHEMA`, `RE_SCHEMA`, `REVIEW_SCHEMA`, `VR_SCHEMA`, `REVIEW_VR_SCHEMA`, `SYNTH_SCHEMA` consumed by later tasks.

- [ ] **Step 1: Write the scaffold**

```js
export const meta = {
  name: 'ultracode',
  description: 'Linear RE→review→VR→review pipeline over the bn/bn-re/bn-vr skills; a fresh reviewer audits each phase and every vuln is adversarially verified.',
  phases: [
    { title: 'Setup' },
    { title: 'RE' },
    { title: 'Review-RE' },
    { title: 'VR' },
    { title: 'Review-VR' },
    { title: 'Synthesize' },
  ],
}

// ---- args & identity (no Date.now/Math.random available) ----
const binary = args && args.binary
if (!binary) throw new Error('ultracode: args.binary is required (path to the target binary)')
const focus = (args && args.focus) || ''
const base = String(binary).split('/').pop().replace(/[^A-Za-z0-9._-]/g, '_')
const instance = (args && args.instance) || `ultracode-${base}`
const runid = (args && args.runid) || base

// ---- fan-out hygiene reminder, threaded into every agent prompt ----
const HYGIENE = [
  `You are a fan-out agent sharing ONE bn instance with the rest of this pipeline.`,
  `On EVERY bn command pass \`--instance ${instance}\` and \`-t ${base}\` (or the exact selector \`bn target list\` shows).`,
  `NEVER run \`bn instance use\` / \`bn target use\` / any \`*clear\` — sticky pins are one shared file per repo and clobber concurrent agents.`,
  `Large bn reads spill to disk and stdout carries only an envelope: write to a file then grep/jq it (\`bn decompile f --out /tmp/f.txt && grep x /tmp/f.txt\`), NEVER \`bn ... | grep\` (that greps the envelope, not the data).`,
  `Keep real target names/addresses out of anything you would commit; this run's report is written to a git-ignored path only.`,
].join(' ')

// ---- structured-output schemas ----
const SETUP_SCHEMA = {
  type: 'object',
  required: ['arch', 'lane', 'baseline', 'orientation'],
  properties: {
    arch: { type: 'string' },
    lane: { type: 'string', enum: ['import-first', 'stripped-static'] },
    baseline: {
      type: 'object',
      required: ['functions', 'named_symbols', 'comments'],
      properties: {
        functions: { type: 'number' },
        named_symbols: { type: 'number' },
        comments: { type: 'number' },
      },
    },
    orientation: { type: 'string', description: 'entry point, notable imports/strings, cache-restore note' },
  },
}

const RE_SCHEMA = {
  type: 'object',
  required: ['map', 'enriched'],
  properties: {
    map: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          entry: { type: 'string' }, dispatch: { type: 'string' },
          handler: { type: 'string' }, sinks: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    enriched: {
      type: 'object',
      properties: {
        renames: { type: 'number' }, protos: { type: 'number' },
        structs: { type: 'number' }, comments: { type: 'number' },
      },
    },
    hidden_surface: { type: 'string' },
    notes: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['verdict', 'efficiency_notes'],
  properties: {
    verdict: { type: 'string', enum: ['pass', 'gaps'] },
    redo: { type: 'array', items: { type: 'string' } },
    efficiency_notes: { type: 'string' },
  },
}

const VR_SCHEMA = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['class', 'location', 'soundness'],
        properties: {
          class: { type: 'string' }, location: { type: 'string' },
          source: { type: 'string' }, sink: { type: 'string' },
          path: { type: 'string' }, prelim_confidence: { type: 'string' },
          soundness: { type: 'string' },
        },
      },
    },
  },
}

const REVIEW_VR_SCHEMA = {
  type: 'object',
  required: ['verdict', 'efficiency_notes', 'verified', 'demoted'],
  properties: {
    verdict: { type: 'string', enum: ['pass', 'gaps'] },
    redo: { type: 'array', items: { type: 'string' } },
    efficiency_notes: { type: 'string' },
    verified: { type: 'array', items: { type: 'object' } },
    demoted: { type: 'array', items: { type: 'object' } },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  required: ['report_path', 'summary'],
  properties: {
    report_path: { type: 'string' },
    summary: { type: 'string' },
  },
}
```

- [ ] **Step 2: Syntax check**

Run: `node --check .claude/workflows/ultracode.js`
Expected: no output, exit 0.

---

### Task 2: Setup + RE stages

**Files:**
- Modify: `.claude/workflows/ultracode.js` (append)

**Interfaces:**
- Consumes: `HYGIENE`, `binary`, `instance`, `focus`, `SETUP_SCHEMA`, `RE_SCHEMA` (Task 1).
- Produces: `setup` (SETUP_SCHEMA result), `rePrompt(redo)` builder, `re` (RE_SCHEMA result).

- [ ] **Step 1: Append Setup + RE**

```js
// ================= Stage 0: Setup =================
phase('Setup')
const setup = await agent(`${HYGIENE}

Load and orient a binary for a downstream RE→VR pipeline. Do:
1. \`bn session start ${binary} --instance-id ${instance}\`. If it reports "restored cached database", say so in orientation (state may be a prior run's).
2. If \`bn target info\` shows analysis_state != "full", run \`bn refresh\` before surveying.
3. Run \`bn evidence orient\` for a one-shot digest (arch, imports, strings sample, sections, count).
4. Decide the LANE by the real tell, NOT \`file\`: "import-first" if \`bn imports\` is non-empty OR \`bn function list\` is mostly named; "stripped-static" only if imports are empty AND names are overwhelmingly sub_XXXX.
5. Capture BASELINE counts (the review stage diffs against these): total functions, named symbols (non sub_*), and comments.
Return the structured result.`,
  { label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA, agentType: 'general-purpose' })

if (!setup) throw new Error('ultracode: setup stage failed (agent returned null)')
log(`setup: arch=${setup.arch} lane=${setup.lane} baseline fns=${setup.baseline && setup.baseline.functions}`)

// ================= Stage 1: RE =================
phase('RE')
const rePrompt = (redo) => `${HYGIENE}

Follow the bn-re methodology. Reverse-engineer ${focus ? `the "${focus}" surface of ` : ''}this binary toward source, and ENRICH THE BNDB — this database is the hand-off to the security phase, so recovered state must persist, not just be described:
- Rename handlers/dispatchers/parsers to meaningful names (preview → apply).
- Set prototypes; reconstruct structs where repeated fixed-offset accesses reveal them.
- Comment assumptions/edge-cases a name can't carry.
- Recover hidden surface where signatures warrant (\`bn evidence surface\`/\`init\`/\`table\`): .init_array constructors, dispatch/vtable tables, missing functions.
- Confirm any bound/width claim in \`bn disasm\` (HLIL flattens ccmp/csel, aliases hoisted bounds, drops <<4).
Produce a distilled entry→dispatch→handler→sink map the VR phase can act on.${redo ? `\n\nA reviewer flagged gaps — address ONLY these, then return the updated map:\n${redo}` : ''}
Return the structured map and the counts of what you enriched.`

let re = await agent(rePrompt(''), { label: 'RE', phase: 'RE', schema: RE_SCHEMA, agentType: 'bn-re' })
if (!re) throw new Error('ultracode: RE stage failed (agent returned null)')
```

- [ ] **Step 2: Syntax check**

Run: `node --check .claude/workflows/ultracode.js`
Expected: exit 0.

---

### Task 3: review-RE + bounded redo

**Files:**
- Modify: `.claude/workflows/ultracode.js` (append)

**Interfaces:**
- Consumes: `re`, `setup`, `rePrompt`, `HYGIENE`, `REVIEW_SCHEMA`, `RE_SCHEMA`.
- Produces: possibly-updated `re`.

- [ ] **Step 1: Append review-RE + redo**

```js
// ================= Stage 2: review-RE =================
phase('Review-RE')
const reReview = await agent(`${HYGIENE}

You are an INDEPENDENT reviewer — you did NOT perform the RE. Audit the RE agent's use of the bn tooling, correctness-weighted. Its returned map:
${JSON.stringify(re, null, 2)}

Baseline counts captured before RE ran: ${JSON.stringify(setup.baseline)}.

PRIMARY (methodology + soundness):
- Right lane chosen by the real tell (imports/named vs empty+sub_*), not \`file\`.
- Any bound/width claim disasm-confirmed (not HLIL alone).
- BNDB ACTUALLY enriched: re-read the LIVE state (\`bn function list\`, \`bn comment list\`) and confirm named-symbol and comment counts materially grew vs baseline. A chat-only report over an untouched BNDB is a FAIL.
- Hidden surface recovered where warranted; the map is usable (entry→handler→sink).
SECONDARY (flag, do NOT gate): pipe/spill trap, one-shot digests used, no redundant re-decompiles.

Return verdict 'pass' or 'gaps'. If 'gaps', give a SHORT targeted redo list (only material items the RE agent must fix). Put secondary observations in efficiency_notes.`,
  { label: 'review:RE', phase: 'Review-RE', schema: REVIEW_SCHEMA, agentType: 'general-purpose' })

if (reReview && reReview.verdict === 'gaps' && reReview.redo && reReview.redo.length) {
  log(`review-RE: gaps → one targeted redo (${reReview.redo.length} items)`)
  const redone = await agent(rePrompt(reReview.redo.join('\n')),
    { label: 'RE-redo', phase: 'RE', schema: RE_SCHEMA, agentType: 'bn-re' })
  if (redone) re = redone
}
```

- [ ] **Step 2: Syntax check**

Run: `node --check .claude/workflows/ultracode.js`
Expected: exit 0.

---

### Task 4: VR + review-VR (verify) + bounded redo + re-verify

**Files:**
- Modify: `.claude/workflows/ultracode.js` (append)

**Interfaces:**
- Consumes: `re`, `focus`, `HYGIENE`, `VR_SCHEMA`, `REVIEW_VR_SCHEMA`.
- Produces: `vr`, `verified` (array).

- [ ] **Step 1: Append VR + review-VR + redo/re-verify**

```js
// ================= Stage 3: VR =================
phase('VR')
const vrPrompt = (redo) => `${HYGIENE}

Follow the bn-vr methodology against this ENRICHED BNDB — the RE phase named handlers/structs/sinks, so use those names. Find BOTH memory-corruption and logic vulns${focus ? ` in the "${focus}" surface` : ''}:
- Enumerate sinks exhaustively: \`bn taint models --role sink --present --callsites\`. A short/empty list is NOT an all-clear on a stripped/static target — recover sinks by shape (bn-vr stripped/static lane).
- Source→sink trace each candidate (\`bn taint forward/backward\`, \`bn trace\`). An empty taint result is NOT an all-clear: also run the MANUAL lanes — parser/fixed-header invariants (loop guard vs header width) and destination-capacity (strcpy/strcat aggregate, audit every wrapper caller).
- Confirm every bound/field-width in \`bn disasm\`, not HLIL.
For each finding report: class, location (fn+addr), source, sink, the source→sink path, a preliminary confidence, and the \`bn taint\` soundness caveat.${redo ? `\n\nA reviewer flagged gaps — address ONLY these:\n${redo}` : ''}
Return structured findings (empty array is allowed, but only after the manual lanes were genuinely checked).`

let vr = await agent(vrPrompt(''), { label: 'VR', phase: 'VR', schema: VR_SCHEMA, agentType: 'bn-vr' })
if (!vr) throw new Error('ultracode: VR stage failed (agent returned null)')

// ================= Stage 4: review-VR (audit + adversarial verify) =================
phase('Review-VR')
const VERIFY_INSTRUCTIONS = `ADVERSARIALLY VERIFY each finding — independently re-derive it in bn: disasm-confirm the bound/field-width, prove the attacker controls the reaching input, and require a soundness caveat. Move any finding you cannot independently confirm (unproven control, HLIL-only bound, false all-clear) into 'demoted' with a one-line reason. Keep confirmed ones in 'verified'.`

const vrReview = await agent(`${HYGIENE}

You are an INDEPENDENT reviewer — you did NOT perform the VR. Two jobs, correctness-weighted:
(A) Audit tool usage: sinks enumerated via \`taint models --present\`; empty taint NOT treated as all-clear (manual parser-invariant + destination-capacity lanes actually run); right lane. Flag (do NOT gate) pipe-trap / redundant reads in efficiency_notes.
(B) ${VERIFY_INSTRUCTIONS}

Findings to verify:
${JSON.stringify((vr && vr.findings) || [], null, 2)}

Return: verdict ('pass', or 'gaps' only for a tool-usage gap worth one redo — e.g. a whole lane unaudited), redo[], efficiency_notes, verified[], demoted[].`,
  { label: 'review:VR', phase: 'Review-VR', schema: REVIEW_VR_SCHEMA, agentType: 'general-purpose' })

let verified = (vrReview && vrReview.verified) || []
if (vrReview && vrReview.verdict === 'gaps' && vrReview.redo && vrReview.redo.length) {
  log(`review-VR: gaps → one targeted redo (${vrReview.redo.length} items)`)
  const vrRedone = await agent(vrPrompt(vrReview.redo.join('\n')),
    { label: 'VR-redo', phase: 'VR', schema: VR_SCHEMA, agentType: 'bn-vr' })
  if (vrRedone) vr = vrRedone
  const reverify = await agent(`${HYGIENE}\n\n${VERIFY_INSTRUCTIONS}\n\nFindings:\n${JSON.stringify((vr && vr.findings) || [], null, 2)}\n\nReturn verified[] and demoted[] (verdict 'pass', empty redo).`,
    { label: 'reverify:VR', phase: 'Review-VR', schema: REVIEW_VR_SCHEMA, agentType: 'general-purpose' })
  if (reverify && reverify.verified) verified = reverify.verified
}
```

- [ ] **Step 2: Syntax check**

Run: `node --check .claude/workflows/ultracode.js`
Expected: exit 0.

---

### Task 5: Synthesize + return

**Files:**
- Modify: `.claude/workflows/ultracode.js` (append)

**Interfaces:**
- Consumes: `re`, `verified`, `setup`, `reReview`, `vrReview`, `binary`, `focus`, `runid`, `instance`, `SYNTH_SCHEMA`.
- Produces: the workflow's return value `{ report_path, summary, verified_count }`.

- [ ] **Step 1: Append Synthesize + return**

```js
// ================= Stage 5: Synthesize =================
phase('Synthesize')
const report = [
  `# ultracode audit — ${runid}`,
  ``,
  `- binary: \`${binary}\``,
  `- focus: ${focus || '(broad)'}`,
  `- arch / lane: ${setup.arch} / ${setup.lane}`,
  `- verified findings: ${verified.length}`,
  ``,
  `## RE map`,
  '```json', JSON.stringify((re && re.map) || [], null, 2), '```',
  ``,
  `### BNDB enrichment`,
  '```json', JSON.stringify((re && re.enriched) || {}, null, 2), '```',
  `${(re && re.hidden_surface) ? `\nHidden surface: ${re.hidden_surface}` : ''}`,
  ``,
  `## Verified findings (${verified.length})`,
  '```json', JSON.stringify(verified, null, 2), '```',
  ``,
  `## Efficiency audit`,
  `### RE phase`,
  (reReview && reReview.efficiency_notes) || '(none)',
  `### VR phase`,
  (vrReview && vrReview.efficiency_notes) || '(none)',
].join('\n')

const synth = await agent(`${HYGIENE}

Finalize this run:
1. \`bn save --instance ${instance}\` to persist the enriched BNDB.
2. \`mkdir -p .dogfood/audits\` then write the report below VERBATIM to \`.dogfood/audits/${runid}.md\`. This path is git-ignored — do NOT git add / commit it.
3. \`bn session stop --instance ${instance}\`.
Return the exact report path written and a 2-3 line summary (arch, #verified findings, the single highest-severity finding if any).

---- REPORT (write verbatim) ----
${report}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, agentType: 'general-purpose' })

log(`ultracode done: ${verified.length} verified finding(s) → ${(synth && synth.report_path) || `.dogfood/audits/${runid}.md`}`)
return {
  report_path: (synth && synth.report_path) || `.dogfood/audits/${runid}.md`,
  summary: (synth && synth.summary) || 'run complete; see report',
  verified_count: verified.length,
}
```

- [ ] **Step 2: Syntax check**

Run: `node --check .claude/workflows/ultracode.js`
Expected: exit 0.

---

### Task 6: End-to-end validation + commit

**Files:**
- Verify: `.claude/workflows/ultracode.js`
- Confirm ignore: `.dogfood/` (already git-ignored)

- [ ] **Step 1: Confirm `.dogfood/` is git-ignored**

Run: `git -C /opt/bn check-ignore .dogfood/audits/x.md`
Expected: prints `.dogfood/audits/x.md` (ignored). If not, add `.dogfood/` to `.gitignore`.

- [ ] **Step 2: Live end-to-end run on a small dogfood target**

Invoke via the Workflow tool (a small x86-64 CWE-corpus binary keeps it cheap and gives a known-answer):
```
Workflow({ name: 'ultracode', args: { binary: '/home/m4ul3r/mg_testbins/targets/cwe-121/<pick-one>', focus: '' } })
```
Watch `/workflows` for the six phases (Setup → RE → Review-RE → VR → Review-VR → Synthesize). On completion, confirm:
- the return value has `report_path` + `verified_count`;
- the report file exists at that path and contains the RE map + findings + efficiency audit;
- `bn session list` shows the `ultracode-*` instance was stopped.

- [ ] **Step 3: Fix any wiring issues surfaced by the run, re-run until clean**

Common issues to check: an agent returning `null` (guarded with throws in Tasks 2/4); the `-t ${base}` selector not matching (fall back to whatever `bn target list` shows — update HYGIENE if needed); the report path not git-ignored.

- [ ] **Step 4: Commit the workflow + spec + plan (script is generic, no target data)**

```bash
cd /opt/bn
git add .claude/workflows/ultracode.js .claude/workflows/ultracode.design.md .claude/workflows/ultracode.plan.md
git status --short   # confirm NO .dogfood/ or report files are staged
git commit -m "feat(workflow): add ultracode RE→review→VR→review pipeline over bn skills

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3 invocation contract → Task 1 (args/identity). ✓
- §4 pipeline stages 0–5 → Tasks 2 (0,1), 3 (2), 4 (3,4), 5 (5). ✓
- §5 rubric → embedded in the review-RE (Task 3) and review-VR (Task 4) prompts. ✓
- §6 instance/BNDB seam + sensitivity → `HYGIENE` const (Task 1), threaded everywhere; git-ignored report (Tasks 5, 6). ✓
- §7 failure handling → null-guards/throws (Tasks 2, 4); `bn session stop` in Synthesize (Task 5). ✓
- §8 budget/scoping → `focus` threaded into RE/VR prompts; caps logged via `log()`. (Note: explicit `budget.remaining()` guards deferred as YAGNI for v1 — the fixed six-phase chain can't meaningfully skip phases, and each agent bounds its own fan-out. Documented here rather than silently dropped.)
- §9 files → Tasks 1, 6. ✓

**Placeholder scan:** no TBD/TODO; every code step is complete. ✓

**Type consistency:** `setup.baseline.{functions,named_symbols,comments}` defined in SETUP_SCHEMA (Task 1) and read in Task 3 review prompt. `re.map`/`re.enriched` defined (RE_SCHEMA) and read in Task 5. `vrReview.verified` defined (REVIEW_VR_SCHEMA) and read in Tasks 4/5. `rePrompt`/`vrPrompt` builders defined before use. Agent-type assignments match spec §4 (general-purpose for 0/2/4/5, bn-re for 1, bn-vr for 3). ✓

**One spec deviation to note for the reviewer:** §8's explicit per-phase `budget.remaining()` guard is intentionally not implemented in v1 (see spec-coverage note above). Flag if you want it added.
