# IDEA_001 — Wrapping repeatable agent patterns for the `bn` skills

**Status:** Design spec (no code). Implementation deferred to a follow-up session.
**Author:** generated from a 4-agent analysis of the `bn` command surface, the three skills,
and dogfood memory notes.
**Scope question being answered:** *Which repeatable patterns that agents run belong as
external shell scripts, and which belong as internal `bn` tools?*

---

## 1. Problem & goal

Agents driving `bn` re-run the same multi-step command sequences in every reverse-engineering
and vuln-research session, and they fall back to ad-hoc shell loops for anything spanning
multiple open bridges. The motivating example, verbatim from the user:

```bash
cd /home/m4ul3r/motiongraph
for inst in c107copc c107vcop c108gate c108ui c108url; do
  echo "=== $inst ==="
  timeout 25 bn --instance $inst target list 2>&1 | head -5
done
```

The three skills (`skills/bn`, `skills/bn-re`, `skills/bn-vr`) document ~14 recurring recipes
in prose but ship **no wrappers** for any of them. Two consequences:

1. Agents retype the same sequences, with per-call bridge-discovery overhead and no aggregation.
2. Some methodology steps get **skipped entirely** because `bn` only *passively* points at them.
   The memory note **[[bn-re-vr-skill-value]]** records the A/B finding: the real value of the
   methodology skills is the hidden-code-surface and sink-enumeration steps, and agents skip them
   precisely because nothing actively hands them off. A wrapper turns a passive hint into a command.

Goal: define a durable rule for where each pattern belongs, then specify the concrete first
implementations so a future session can build them without re-deriving the architecture.

---

## 2. Decision framework — classify by the state each pattern touches

The script-vs-tool question is ambiguous until you stop asking "which is fancier" and start
asking **what state the pattern reads or mutates**. That yields three layers, not two:

| Layer | What it is | Lives in | State it touches | Why it can't live elsewhere |
|-------|-----------|----------|------------------|------------------------------|
| **1. CLI orchestration** | Fan a single `bn` read across many instances/targets, aggregate | `src/bn/` only (no bridge change) | CLI process state: the instance/target registry | A bash loop can't reuse instance-resolution errors, sticky-state rules, or the JSON spillover envelope |
| **2. Bridge composite op** | One round-trip composite of several BN-state reads | `bridge.py` + a command handler + a formatter | BinaryView state, held under a single lock | A multi-call script pays N round-trips, N lock acquisitions, and can interleave with a concurrent writer |
| **3. Skill script** | A recipe composing `bn` with `jq`/`grep`/other tools, or branching logic | `skills/<name>/scripts/*.sh`, referenced from `SKILL.md` | Nothing bn-internal — pure outside orchestration | Putting `jq`-pipelines or RE branching logic in the bridge would bloat it with non-BN concerns |

**The rule (apply in this order):**

- Does it *loop one `bn` op over instances/targets and aggregate the rows*? → **Layer 1.**
- Is it an *atomic composite of BN-state reads that wants internal consistency*? → **Layer 2.**
- Does it *compose `bn` with non-`bn` tools, or encode multi-step branching*? → **Layer 3.**

The user's own motivating example is **Layer 1, not a script.** Cross-instance fan-out needs
`list_instances()` (`src/bn/transport.py:178`), the instance-resolution error messages, the
sticky-pin semantics, and the >10k-token disk-spillover envelope — none of which a `for` loop
can reuse. That single reclassification is the spine of this design: *the thing the user reached
for a shell loop to do is exactly the thing that should be built into the CLI.*

---

## 3. Inventory of recurring patterns (tagged by layer)

Extracted from the three `SKILL.md` files and corroborated against the dogfood notes
([[fullstack-sweep-jun10]], [[dmh-dogfood-jun10]], [[goal-progress]]). "Skipped?" marks steps
the skills say agents tend to miss.

| # | Pattern | Typical sequence | Layer | Skipped? |
|---|---------|------------------|-------|----------|
| 1 | **Orientation / triage** | `target info` → `imports` → `strings` → `function list --count` → `sections` | **2** (`evidence orient`) | — |
| 2 | **Pre-main / constructor recovery** | `evidence init` → decompile each → rename `stage1_*` | **2** (folds into `evidence surface`) | **yes** |
| 3 | **Dispatch / vtable table recovery** | `evidence table` → `function create` per missing slot | **2** read part → `evidence surface`; create stays a separate write | **yes** |
| 4 | **Type-recovery loop** | `symbol rename` → `proto set` → `local retype` → `struct field set` (preview each) | n/a (already first-class; batch via `batch apply`) | — |
| 5 | **Call-graph traversal** | `decompile` → `xrefs` → `callsites --within` → `trace --interprocedural` | n/a (first-class ops) | — |
| 6 | **Sink → source tracing** | `xrefs <sink>` → `callsites` → `decompile` → `trace --arg` | n/a (first-class) | — |
| 7 | **Stripped-binary data-first entry** | `target info` → `function list --count` → `strings --regex` → `xrefs` → `evidence init` | **3** (recipe) + leans on Layer 2 | partial |
| 8 | **Forward / backward taint + fallback** | `taint forward` → (`dataflow defuse`/`values` + `callsites` stitch) | n/a (first-class) | — |
| 9 | **Struct reconstruction iteration** | `decompile --addresses` → `struct field set` ×N → `local retype` → re-decompile | n/a (first-class; batch) | — |
| 10 | **Evidence gathering (C++/proto/RTTI)** | `evidence message`/`table`/`function`/`xrefs` → `disasm` confirm | n/a (first-class evidence ops) | — |
| 11 | **TODO-driven analysis** | `comment set "TODO: …"` → `comment list --query TODO` | n/a (first-class) | — |
| 12 | **Mutation safety loop** | `<mut> --preview` → `<mut>` → readback → `save` | n/a (built into every mutation) | — |
| 13 | **Batch mutations** | manifest JSON → `batch apply --preview` → `batch apply` | n/a (`batch apply` exists) | — |
| 14 | **Dangerous-sink enumeration** | `imports` → filter memcpy/strcpy/system/recv → `xrefs` each | **3** (`sink-sweep.sh`) | **yes** |
| 15 | **Cross-instance / cross-target fan-out** | `for inst in …; do bn --instance $inst <op>; done` | **1** (`--all-instances`/`--all-targets`) | — |

Most patterns (4, 5, 6, 8–13) are *already first-class* — they need no wrapper, only the skills
already reference them. The actionable gaps are **#1, #2/#3, #14, #15**, which map to the three
layers below.

---

## 4. Layer 1 — CLI-side fan-out (`--all-instances` / `--all-targets`)

**Decision: a modifier flag, not a `bn fanout <subcommand>` wrapper.** Grounded in the code:
`_build_from_commands` (`src/bn/cli.py:310`) walks `_COMMANDS` and already attaches the
`--instance`/`--target` options to *every* leaf parser via `_instance_option`
(`src/bn/cli.py:211`). A `bn fanout <subcommand>` form would have to re-parse an arbitrary
nested argv and duplicate the whole arg tree; a modifier reuses the single resolution chokepoint
that every read already funnels through.

**Design:**

- Add `--all-instances` and `--all-targets` (both `store_true`) next to `_instance_option`
  (`src/bn/cli.py:211`), attached in `_build_from_commands` (`src/bn/cli.py:310`) **only to read
  commands** (gate on the command spec's default format being text, or add an explicit
  `fanout=False` to the `@command` decorator for mutations).
- Branch inside `_call()` (`src/bn/cli.py:465`) — the single point where one instance/target is
  resolved today. When a fan-out flag is set, enumerate instances via `list_instances()`
  (`src/bn/transport.py:178`) and/or per-instance targets via the existing `list_targets` op, then
  loop the existing `send_request` per `(instance, target)` pair and collect
  `{instance, target, ok, result | error}` rows.
- Render the aggregate through the existing `_render_result` path so `--format json` and the
  disk-spillover envelope keep working; add `_render_fanout_text` in `src/bn/formatters.py` for a
  per-section text view (header per instance/target, then the inner result).

**Semantics & edge cases:**

- **Precedence over sticky pins.** `_apply_sticky_defaults` (`src/bn/cli.py:716`) fills
  `args.instance`/`args.target` from per-project pins. Fan-out flags must win — check them before
  sticky fill, or have sticky-fill skip when a fan-out flag is set.
- **Per-instance target ambiguity.** `--all-instances` alone fans instances but each needs a
  target. Apply the normal `_implicit_target` rule (`src/bn/cli.py:422`) per instance; an
  instance whose target is ambiguous returns a row with `ok:false` and the existing multi-target
  message — it does **not** abort the whole run.
- **Partial failure → exit code.** Collect-don't-abort. A dead bridge (already purged by
  `list_instances`) just yields a failed row. Exit `0` if all rows ok, `2` if any failed (matches
  the `BridgeError` → exit-2 convention).
- **No auto-spawn on empty.** Fan-out over zero instances must return an empty result, **not**
  trigger `_auto_spawn_locked` (`src/bn/transport.py:207`) — that would spawn an empty bridge.

**Result:** `bn --all-instances target list`, `bn imports --all-targets`,
`bn --all-instances --all-targets strings --query foo` replace the user's hand-written loop and
return one aggregated, spill-aware, jq-able artifact.

---

## 5. Layer 2 — Bridge composite read ops

Two new ops, each wired by cloning the handler shape of `_evidence_init`
(`src/bn/commands/function.py:476`, decorator `@command("evidence","init")` at line 469) and the
compose-internal-methods body of `_bundle_function`
(`plugin/bn_agent_bridge/bridge.py:4753`), which already calls several private read methods and
returns one structured artifact under a single lock.

**Lock advantage (the reason these are ops, not scripts):** a composite read holds the
writer-priority read lock for its *entire* duration, so no writer interleaves between sub-reads.
The digest is internally consistent — a guarantee a multi-call shell script cannot get.

### 5a. `bn evidence orient` → op `orient_digest`

Composite of methods that already exist in `bridge.py`:

- `_target_info(selector)` (`plugin/bn_agent_bridge/bridge.py:1203`) — and **surface its
  `analyzed`/`analysis_state` fields** (set at ~`bridge.py:1213–1222`) at the top of the digest,
  so an agent doesn't trust an empty strings/function set from a `--quick`-loaded view.
- imports summary via the existing imports handler with `summary=True`.
- a **bounded** strings sample (small `limit`, `min_length>=6`) — never the full set.
- function **count only** (`count_only`), never the full list.
- sections list.

Returns `{target, analyzed, imports_summary, strings_sample, function_count, sections}`.

**Wiring:** new `_orient_digest` method in `bridge.py`; dispatch entry alongside the other
evidence ops; add `"orient_digest"` to `READ_LOCKED_OPS` (`plugin/bn_agent_bridge/bridge.py:216`)
— verified no collision with existing entries. New `@command("evidence","orient")` handler in
`src/bn/commands/function.py` (clone `_evidence_init`), new `_render_orient_text` in
`src/bn/formatters.py`.

### 5b. `bn evidence surface` → op `hidden_surface`

The "hidden code surface" discovery that skills say agents skip (patterns #2 and #3). Composite of:

- `_init_arrays(selector, limit=…)` (`plugin/bn_agent_bridge/bridge.py:4384`) — `.init_array` /
  `.ctors` / `.fini_array` evidence.
- a dispatch/vtable scan reusing `_pointer_table_for_view`
  (`plugin/bn_agent_bridge/bridge.py:3530`) over candidate `.data`/`.rodata` sections.
- a **data-only function candidates** report: pointer targets whose bytes look like a prologue but
  have no `bv` function. Reuse the existing instruction-evidence/prologue helper rather than
  hand-rolling arch detection; tag each candidate `plausible:true|false` exactly as
  `evidence table` already does — **do not assert they are functions**.

Returns `{init_sections, candidate_tables, missing_function_candidates}`.

**Critical:** `hidden_surface` is **read-only** — it reports addresses BN missed but does **not**
call `function_create` (that is the agent's next, separate write op). It therefore stays in
`READ_LOCKED_OPS`; mixing a write would force the exclusive lock and change the op's contract.

**Wiring:** same pattern — `_hidden_surface` method, dispatch entry, add `"hidden_surface"` to
`READ_LOCKED_OPS` (no collision), `@command("evidence","surface")` handler, `_render_surface_text`.

---

## 6. Layer 3 — Skill methodology scripts

**Location:** `skills/<name>/scripts/*.sh`. These ride the existing whole-directory install with
**zero changes to the iteration logic**: `_skill_install` (`src/bn/commands/admin.py:167`)
installs each skill dir wholesale via `_install_tree` (`src/bn/commands/admin.py:134`), which
does a directory-level `os.symlink` / `shutil.copytree`. There is already a non-`SKILL.md` subdir
precedent — `skills/bn/agents/` ships today.

**Discoverability:** do **not** put scripts on `$PATH`. In the default symlink-install mode that
would leak repo internals, and the scripts are invoked *by the agent following `SKILL.md`*, which
can reference them by their installed relative path (e.g. `bash scripts/sink-sweep.sh <target>`).
The scripts call `bn`, which is already on `PATH` via `uv tool install`.

**Only code change needed:** set the executable bit on `scripts/*.sh` after a **copy**-mode
install (symlink mode follows the source file's bit). A ~3-line guarded loop at the end of
`_skill_install` (`src/bn/commands/admin.py:167`).

**First scripts to ship:**

- `skills/bn-re/scripts/orient.sh` — wraps `bn evidence orient --format json | jq` into a
  human-readable triage card. (Bridges Layer 2 → Layer 3; ship *after* `evidence orient` exists.)
- `skills/bn-vr/scripts/sink-sweep.sh` — `bn imports --format json | jq` to enumerate
  dangerous-sink imports (memcpy/strcpy/system/recv/sprintf/…), then `bn xrefs` each. This
  encodes the sink-enumeration step the memory note [[bn-re-vr-skill-value]] says agents skip.

---

## 7. Recommended build phasing (for the implementation session)

- **Phase 1 (cleanest wins):**
  - `bn evidence orient` — most-reused pattern, has a direct clone target in `_bundle_function`,
    pure-read composite with trivial lock placement.
  - Layer 1 `--all-instances` / `--all-targets` on read commands — directly retires the user's loop.
- **Phase 2:** `bn evidence surface` — higher value for vuln work but more bridge logic; the
  "looks like a prologue but isn't a function" detector is arch-sensitive (x86 `endbr64`/`push
  rbp`, ARM/Thumb `push {lr}`/`stp x29,x30`). Defer until `orient` validates the composite pattern.
- **Phase 3:** Layer 3 scripts — `orient.sh` (wraps the Phase-1 op) and `sink-sweep.sh`.

**Out of scope (and why):**

- Fan-out over *mutations* (`--all-targets bn symbol rename`) — cross-instance locking and
  partial-revert is a separate hard problem; the memory note [[taint-scope-keep-simple]] warns
  against new cross-binary CLI surface. **Reads only.**
- Scripts on `$PATH`.
- A general `bn fanout <subcommand>` wrapper (rejected — see §4).
- Any cross-binary automation.

---

## 8. Risks & edge cases

**Layer 1 fan-out:** sticky-pin precedence (§4); per-instance target ambiguity returns `ok:false`
rather than aborting; one dead bridge yields a failed row, exit 2 if any fail; large
N-targets × output can exceed the 10k-token spill threshold but the aggregate already flows
through `_render_result`, so spillover works — preserve the per-row labels into the artifact;
must not auto-spawn on an empty instance list.

**Layer 2 composite ops:** a read composite holds the read lock for the whole digest — cap the
strings sample, use `count_only` for the function count, and **never decompile** inside `orient`
(per-function, slow). `_target_info` already reports `analysis_state: "quick"`
(`bridge.py:1213–1222`); propagate it so an empty result from a quick-load isn't trusted.
`hidden_surface` false positives are arch-specific — reuse the existing instruction-evidence
helper and tag `plausible:true|false`; never create functions inside the op.

**Layer 3 scripts:** copy-mode needs the `+x` bit set post-copy; symlink mode follows the source
bit (set it once in the repo). Scripts assume `bn` is on `PATH` — document the dependency in
`SKILL.md`. `_skill_install` already skips existing destinations unless `--force`, so adding a
`scripts/` subdir changes nothing about idempotency.

---

## 9. Verification (for the implementation, when built)

1. **Citation drift:** re-confirm the anchors before editing — `_bundle_function`
   (`bridge.py:4753`), `_evidence_init` (`function.py:476`), `READ_LOCKED_OPS` (`bridge.py:216`),
   `_call` (`cli.py:465`), `_skill_install` (`admin.py:167`). (Verified current as of this spec.)
2. **No op-name collision:** `orient_digest` and `hidden_surface` are absent from `dispatch()`,
   `READ_LOCKED_OPS`, and `WRITE_LOCKED_OPS` today — re-check at build time.
3. **End-to-end, live:** against an open instance, run `bn evidence orient` and confirm the digest
   contents match the individual `target info` / `imports --summary` / `strings` / `function list
   --count` / `sections` commands; run `bn --all-instances target list` against the motiongraph
   instances (`c107copc`, `c107vcop`, …) and confirm it reproduces the user's loop output as one
   aggregate; confirm a stopped instance yields a failed row + exit 2, not a crash.
4. **Tests** mirror source layout: `tests/test_cli.py` (fan-out flag + aggregation + exit codes),
   `tests/test_bridge.py` (`orient_digest` / `hidden_surface` composites, lock membership,
   quick-load honesty), `tests/test_cli.py` or a skill test for the copy-mode `+x` bit.
