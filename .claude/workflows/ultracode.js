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

/*
 * ultracode — dev-only harness for exercising the bn / bn-re / bn-vr skills.
 * NOT part of the bn package; reports go to the git-ignored .dogfood/ tree.
 *
 * INVOKE via scriptPath — the `name:` registry caches the FIRST version, so to run
 * the current file always pass its path:
 *   Workflow({ scriptPath: '<repo>/.claude/workflows/ultracode.js',
 *              args: { binary: '<path>', focus?: '<hint>', depth?: 'deep'|'smoke' } })
 *   • args may be an object OR a JSON string (both are handled).
 *   • depth 'deep' (default): full pipeline + one bounded targeted redo per working
 *     phase, normal agent effort.
 *   • depth 'smoke': single pass (no redo loops) at low agent effort — a faster,
 *     cheaper triage. Every stage still runs (incl. the independent verify), so the
 *     "no false all-clear" guarantee holds; only the corrective retry is skipped.
 * Output: a git-ignored .dogfood/audits/<runid>.md report + a saved BNDB. The report
 * carries real target data — never commit it.
 */

// ---- args & identity (no Date.now/Math.random available) ----
// The Workflow tool may hand `args` through as a JSON string rather than an
// object; accept either so invocation is robust.
let _args = args
if (typeof _args === 'string') {
  try { _args = JSON.parse(_args) } catch (_e) { _args = {} }
}
const binary = _args && _args.binary
if (!binary) throw new Error(`ultracode: args.binary is required (path to the target binary); got typeof args=${typeof args}, value=${String(JSON.stringify(args)).slice(0, 160)}`)
const focus = (_args && _args.focus) || ''
const depth = String((_args && _args.depth) || 'deep').toLowerCase()
const SMOKE = depth === 'smoke'
// Low agent effort in smoke mode; inherit the session default in deep mode.
const EX = SMOKE ? { effort: 'low' } : {}
const base = String(binary).split('/').pop().replace(/[^A-Za-z0-9._-]/g, '_')
const instance = (_args && _args.instance) || `ultracode-${base}`
const runid = (_args && _args.runid) || base

// ---- fan-out hygiene reminder, threaded into every agent prompt ----
// A function of the resolved selector: Setup discovers the exact `bn target
// list` selector (a read-only-mount target restores a cache DB named
// `<stem>.<hash>.bndb`, not `<stem>`) and threads it to every later stage, so no
// agent has to re-discover it. `base` is only the bootstrap guess for Setup.
const makeHygiene = (sel) => [
  `You are a fan-out agent sharing ONE bn instance with the rest of this pipeline.`,
  `On EVERY bn command pass \`--instance ${instance}\` and \`-t ${sel}\` — this is the exact selector \`bn target list\` shows; use it verbatim (do NOT abbreviate or re-derive it).`,
  `NEVER run \`bn instance use\` / \`bn target use\` / any \`*clear\` — sticky pins are one shared file per repo and clobber concurrent agents.`,
  `The shell is zsh, which does NOT word-split an unquoted \`$var\`: NEVER stash bn flags in a shell variable and expand it (\`D="--instance x -t y"; bn cmd $D\` sends the whole string as ONE argv token → "unrecognized arguments"/"Invalid instance id"). Write the flags inline on each command.`,
  `Large bn reads spill to disk and stdout carries only an envelope: write to a file then grep/jq it (\`bn decompile f --out /tmp/f.txt && grep x /tmp/f.txt\`), NEVER \`bn ... | grep\` (that greps the envelope, not the data).`,
  `Keep real target names/addresses out of anything you would commit; this run's report is written to a git-ignored path only.`,
].join(' ')
// Bootstrap for Setup only; reassigned to the resolved selector once Setup runs.
let HYGIENE = makeHygiene(base)

// ---- severity ranking (for the return payload + prose ordering) ----
const SEV_RANK = { critical: 3, high: 2, medium: 1, low: 0 }
const sevRank = (s) => {
  const r = SEV_RANK[String(s || '').toLowerCase()]
  return r === undefined ? -1 : r
}

// ---- structured-output schemas ----
const SETUP_SCHEMA = {
  type: 'object',
  required: ['arch', 'lane', 'baseline', 'orientation', 'selector'],
  properties: {
    arch: { type: 'string' },
    selector: { type: 'string', description: 'the exact `bn target list` selector for this target (used verbatim by every later stage)' },
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

// A verified/candidate finding. severity + headline drive the return payload and the
// prose report; the rest is the evidence.
const FINDING_PROPS = {
  severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'info'] },
  headline: { type: 'string', description: 'one-line summary, no addresses' },
  class: { type: 'string' }, location: { type: 'string' },
  source: { type: 'string' }, sink: { type: 'string' },
  path: { type: 'string' }, prelim_confidence: { type: 'string' },
  soundness: { type: 'string' },
}

const VR_SCHEMA = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: { type: 'object', required: ['class', 'location', 'severity', 'headline', 'soundness'], properties: FINDING_PROPS },
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
    verified: { type: 'array', items: { type: 'object', properties: FINDING_PROPS } },
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

// ================= Stage 0: Setup =================
phase('Setup')
const setup = await agent(`${HYGIENE}

Load and orient a binary for a downstream RE→VR pipeline. Do:
1. \`bn session start ${binary} --instance-id ${instance}\`. If it reports "restored cached database", say so in orientation (state may be a prior run's).
2. Run \`bn target list --instance ${instance}\` and capture the EXACT selector it prints for this target as \`selector\` — on a read-only mount it will be a cache name like \`<stem>.<hash>.bndb\`, not the bare stem. Every later stage uses this verbatim, so it must be the literal selector, not a guess.
3. If \`bn target info\` shows analysis_state != "full", run \`bn refresh\` before surveying.
4. Run \`bn evidence orient\` for a one-shot digest (arch, imports, strings sample, sections, count).
5. Decide the LANE by the real tell, NOT \`file\`: "import-first" if \`bn imports\` is non-empty OR \`bn function list\` is mostly named; "stripped-static" only if imports are empty AND names are overwhelmingly sub_XXXX.
6. Capture BASELINE counts (the review stage diffs against these): total functions, named symbols (non sub_*), and comments.
Return the structured result (including \`selector\`).`,
  { ...EX, label: 'setup', phase: 'Setup', schema: SETUP_SCHEMA, agentType: 'general-purpose' })

if (!setup) throw new Error('ultracode: setup stage failed (agent returned null)')
// Thread the exact selector Setup resolved into every later stage's HYGIENE, so
// no agent re-derives it (a read-only-mount cache DB is `<stem>.<hash>.bndb`).
const SELECTOR = (setup.selector && String(setup.selector).trim()) || base
HYGIENE = makeHygiene(SELECTOR)
log(`setup: arch=${setup.arch} lane=${setup.lane} depth=${depth} selector=${SELECTOR} baseline fns=${setup.baseline && setup.baseline.functions}`)

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

let re = await agent(rePrompt(''), { ...EX, label: 'RE', phase: 'RE', schema: RE_SCHEMA, agentType: 'bn-re' })
if (!re) throw new Error('ultracode: RE stage failed (agent returned null)')

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
  { ...EX, label: 'review:RE', phase: 'Review-RE', schema: REVIEW_SCHEMA, agentType: 'general-purpose' })

// deep mode gates a single bounded targeted redo; smoke mode logs the gaps and proceeds.
if (!SMOKE && reReview && reReview.verdict === 'gaps' && reReview.redo && reReview.redo.length) {
  log(`review-RE: gaps → one targeted redo (${reReview.redo.length} items)`)
  const redone = await agent(rePrompt(reReview.redo.join('\n')),
    { ...EX, label: 'RE-redo', phase: 'RE', schema: RE_SCHEMA, agentType: 'bn-re' })
  if (redone) re = redone
} else if (SMOKE && reReview && reReview.verdict === 'gaps') {
  log(`review-RE: gaps (smoke mode — no redo)`)
}

// ================= Stage 3: VR =================
phase('VR')
const vrPrompt = (redo) => `${HYGIENE}

Follow the bn-vr methodology against this ENRICHED BNDB — the RE phase named handlers/structs/sinks, so use those names. Find BOTH memory-corruption and logic vulns${focus ? ` in the "${focus}" surface` : ''}:
- Enumerate sinks exhaustively: \`bn taint models --role sink --present --callsites\`. A short/empty list is NOT an all-clear on a stripped/static target — recover sinks by shape (bn-vr stripped/static lane).
- Source→sink trace each candidate (\`bn taint forward/backward\`, \`bn trace\`). An empty taint result is NOT an all-clear: also run the MANUAL lanes — parser/fixed-header invariants (loop guard vs header width) and destination-capacity (strcpy/strcat aggregate, audit every wrapper caller).
- Confirm every bound/field-width in \`bn disasm\`, not HLIL.
For each finding report: severity (critical/high/medium/low/info), a one-line headline (no addresses), class, location (fn+addr), source, sink, the source→sink path, a preliminary confidence, and the \`bn taint\` soundness caveat.${redo ? `\n\nA reviewer flagged gaps — address ONLY these:\n${redo}` : ''}
Return structured findings (empty array is allowed, but only after the manual lanes were genuinely checked).`

let vr = await agent(vrPrompt(''), { ...EX, label: 'VR', phase: 'VR', schema: VR_SCHEMA, agentType: 'bn-vr' })
if (!vr) throw new Error('ultracode: VR stage failed (agent returned null)')

// ================= Stage 4: review-VR (audit + adversarial verify) =================
phase('Review-VR')
const VERIFY_INSTRUCTIONS = `ADVERSARIALLY VERIFY each finding — independently re-derive it in bn: disasm-confirm the bound/field-width, prove the attacker controls the reaching input, and require a soundness caveat. Move any finding you cannot independently confirm (unproven control, HLIL-only bound, false all-clear) into 'demoted' with a one-line reason. Keep confirmed ones in 'verified'. Each verified item MUST carry: severity (critical/high/medium/low/info), a one-line headline (no addresses), class, location, source, sink, path, soundness — set severity from real impact + attacker control, downgrading anything not fully proven.`

const vrReview = await agent(`${HYGIENE}

You are an INDEPENDENT reviewer — you did NOT perform the VR. Two jobs, correctness-weighted:
(A) Audit tool usage: sinks enumerated via \`taint models --present\`; empty taint NOT treated as all-clear (manual parser-invariant + destination-capacity lanes actually run); right lane. Flag (do NOT gate) pipe-trap / redundant reads in efficiency_notes.
(B) ${VERIFY_INSTRUCTIONS}

Findings to verify:
${JSON.stringify((vr && vr.findings) || [], null, 2)}

Return: verdict ('pass', or 'gaps' only for a tool-usage gap worth one redo — e.g. a whole lane unaudited), redo[], efficiency_notes, verified[], demoted[].`,
  { ...EX, label: 'review:VR', phase: 'Review-VR', schema: REVIEW_VR_SCHEMA, agentType: 'general-purpose' })

let verified = (vrReview && vrReview.verified) || []
if (!SMOKE && vrReview && vrReview.verdict === 'gaps' && vrReview.redo && vrReview.redo.length) {
  log(`review-VR: gaps → one targeted redo (${vrReview.redo.length} items)`)
  const vrRedone = await agent(vrPrompt(vrReview.redo.join('\n')),
    { ...EX, label: 'VR-redo', phase: 'VR', schema: VR_SCHEMA, agentType: 'bn-vr' })
  if (vrRedone) vr = vrRedone
  const reverify = await agent(`${HYGIENE}\n\n${VERIFY_INSTRUCTIONS}\n\nFindings:\n${JSON.stringify((vr && vr.findings) || [], null, 2)}\n\nReturn verified[] and demoted[] (verdict 'pass', empty redo).`,
    { ...EX, label: 'reverify:VR', phase: 'Review-VR', schema: REVIEW_VR_SCHEMA, agentType: 'general-purpose' })
  if (reverify && reverify.verified) verified = reverify.verified
} else if (SMOKE && vrReview && vrReview.verdict === 'gaps') {
  log(`review-VR: gaps (smoke mode — no redo)`)
}

// rank verified findings by severity (highest first) for the report + return payload
const ranked = verified.slice().sort((a, b) => sevRank(b && b.severity) - sevRank(a && a.severity))

// ================= Stage 5: Synthesize =================
phase('Synthesize')
const sevBadge = (s) => `[${String(s || 'UNRATED').toUpperCase()}]`
const findingsProse = ranked.length
  ? ranked.map((f, i) => [
      `### ${i + 1}. ${sevBadge(f && f.severity)} ${(f && f.headline) || (f && f.class) || 'finding'}`,
      (f && f.class) ? `- **class:** ${f.class}` : null,
      (f && f.location) ? `- **location:** ${f.location}` : null,
      (f && f.source) ? `- **source:** ${f.source}` : null,
      (f && f.sink) ? `- **sink:** ${f.sink}` : null,
      (f && f.path) ? `- **path:** ${f.path}` : null,
      (f && f.soundness) ? `- **soundness:** ${f.soundness}` : null,
    ].filter(Boolean).join('\n')).join('\n\n')
  : '_No verified findings._ (An empty result here means the pipeline could not confirm a bug — check the Efficiency audit + RE map for disclosed frontiers, which are NOT all-clears.)'

const report = [
  `# ultracode audit — ${runid}`,
  ``,
  `- binary: \`${binary}\``,
  `- focus: ${focus || '(broad)'}`,
  `- depth: ${depth}${SMOKE ? ' (single-pass, low-effort triage)' : ''}`,
  `- arch / lane: ${setup.arch} / ${setup.lane}`,
  `- verified findings: ${ranked.length}`,
  ``,
  `## Verified findings (${ranked.length})`,
  findingsProse,
  ``,
  `## RE map`,
  '```json', JSON.stringify((re && re.map) || [], null, 2), '```',
  ``,
  `### BNDB enrichment`,
  '```json', JSON.stringify((re && re.enriched) || {}, null, 2), '```',
  `${(re && re.hidden_surface) ? `\nHidden surface: ${re.hidden_surface}` : ''}`,
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
  { ...EX, label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, agentType: 'general-purpose' })

const top = ranked[0]
  ? { severity: (ranked[0].severity) || 'unrated', headline: ranked[0].headline || ranked[0].class || 'finding' }
  : null
log(`ultracode done (${depth}): ${ranked.length} verified finding(s) → ${(synth && synth.report_path) || `.dogfood/audits/${runid}.md`}`)
return {
  report_path: (synth && synth.report_path) || `.dogfood/audits/${runid}.md`,
  summary: (synth && synth.summary) || 'run complete; see report',
  verified_count: ranked.length,
  top,
  findings: ranked.map((f) => ({
    severity: (f && f.severity) || 'unrated',
    headline: (f && f.headline) || (f && f.class) || 'finding',
    location: (f && f.location) || '?',
  })),
}
