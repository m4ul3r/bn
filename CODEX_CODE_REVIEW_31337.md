# Codex Code Review 31337

Date: 2026-06-12

Scope: usability and live-command coverage audit for the local `bn` CLI and bridge workflow in `/opt/bn`.

Redaction policy: this report intentionally omits supplied target paths, filenames, raw strings, target symbols, decompiler output, disassembly, hashes, build IDs, socket paths, cache paths, and saved artifact contents. Supplied binaries are referenced only by anonymous labels.

## Executive Summary

I exercised the full registered CLI surface: 63 command paths were enumerated from the parser registry and every command path was live-tested, unit-tested, or safely tested through a temp destination. The audit used isolated `bn-agent` instances, explicit `--instance` routing, temporary install destinations, temporary BNDB outputs, preview mutations, and in-memory scratch mutations that were closed without saving to the supplied corpora.

Overall result: no unresolved live failures remain after follow-up runs. The first pass exposed several expected guardrails and two harness mistakes; the follow-up pass verified the intended workflows with full-analysis loads and corrected inputs.

High-signal outcomes:

- `uv run pytest -q` completed with `667 passed, 3 skipped`.
- 166 live audit steps were executed across three passes.
- All 63 registered command paths were covered.
- All supplied target groups loaded successfully through isolated sessions.
- Full-analysis string extraction, spill behavior, taint, dataflow, trace, mutation preview, save, close, auto-spawn, preload, and multi-instance routing were verified.
- No supplied target content is included in this report.

## Anonymous Targets

| Label | Origin | Purpose |
| --- | --- | --- |
| U1 | Supplied firmware corpus | Small ARM ELF load/list/import/section/string smoke coverage |
| U2 | Supplied rootfs corpus | ARM executable load/list/import/section/string smoke coverage |
| U3 | Supplied firmware corpus | PE32 load/list/import/section/string smoke coverage |
| U4 | Supplied target corpus | x86-64 ELF load/list/import/section/string/spill coverage |
| M1 | Local compiled micro-corpus | Known calls, taint, dataflow, trace, mutation, type, and output coverage |
| M2 | Local compiled micro-corpus | C++/indirect-call callgraph coverage |

## Verification Summary

| Area | Result | Notes |
| --- | --- | --- |
| Unit and integration tests | PASS | `667 passed, 3 skipped` |
| CLI registry enumeration | PASS | 63 command paths found |
| Live command audit pass 1 | REVIEWED | 129 steps, 123 direct passes, 6 adjudicated |
| Live command audit pass 2 | REVIEWED | 31 steps, 30 direct passes, 1 harness input error |
| Batch-file correction pass | PASS | 6 steps, 6 passes |
| Supplied target loading | PASS | U1-U4 loaded in isolated sessions |
| Full-analysis supplied target strings | PASS | U1-U4 verified after full-analysis loads |
| Large output spill probe | PASS | Verified after full-analysis load |
| Sticky state cleanup | PASS | Project sticky state was restored after pin tests |
| Session cleanup | PASS | Dedicated audit instances were stopped |

## Command Coverage Matrix

| Command path | Coverage | Notes |
| --- | --- | --- |
| `batch apply` | PASS | Stdin and file manifest forms tested with preview |
| `bundle function` | PASS | Function bundle written to temp output |
| `callsites` | PASS | Scoped JSON/text paths and missing-scope error path tested |
| `close` | PASS | Selected target, named path, `--all`, and conflicting path plus `--all` guard tested |
| `comment delete` | PASS | Live scratch delete and preview flow tested |
| `comment get` | PASS | Address and function forms tested |
| `comment list` | PASS | Query filtering tested |
| `comment set` | PASS | Live scratch and preview forms tested |
| `dataflow callgraph` | PASS | Direct, indirect, both directions, and no-resolve flag tested |
| `dataflow defuse` | PASS | SSA variable selector tested |
| `dataflow values` | PASS | Instruction-address value-set lookup tested |
| `decompile` | PASS | Text, JSON, address gutter, line slicing, force-analysis, and invalid JSON line-slice guard tested |
| `disasm` | PASS | Line slicing tested |
| `doctor` | PASS | Unscoped and instance-scoped forms tested |
| `evidence function` | PASS | Function evidence helper tested |
| `evidence init` | PASS | Constructor/destructor section helper tested |
| `evidence message` | PASS | Message/type-name lens tested without embedding matches |
| `evidence table` | PASS | Pointer-table helper tested |
| `evidence xrefs` | PASS | Contextual xrefs tested |
| `function create` | PASS | Existing-function preview/no-op behavior tested |
| `function info` | PASS | Compact and verbose forms tested |
| `function list` | PASS | Count, paged text, JSON, NDJSON, `--out`, and multi-target guard tested |
| `function search` | PASS | Substring, regex, exact, and address-filter forms tested |
| `function structured-il` | PASS | SSA/default and non-SSA HLIL forms tested |
| `il` | PASS | HLIL, MLIL, LLIL, SSA, JSON, and line slicing tested |
| `imports` | PASS | Summary and paged list forms tested |
| `instance clear` | PASS | Sticky pin cleanup tested and restored |
| `instance list` | PASS | Alias coverage tested |
| `instance use` | PASS | Sticky pin set tested and restored |
| `load` | PASS | Normal, quick, `--no-bndb`, named auto-spawn, and multi-target loads tested |
| `local list` | PASS | Stable local identifiers listed |
| `local rename` | PASS | Preview mutation tested |
| `local retype` | PASS | Preview mutation tested |
| `plugin install` | PASS | Copy-mode install to temp destination tested |
| `proto get` | PASS | Prototype read tested |
| `proto set` | PASS | Preview mutation tested |
| `py exec` | PASS | `--code`, `--script`, and `--stdin` forms tested |
| `read` | PASS | Positional address, `--address`, hex output, raw bytes to file, and raw bytes stdout tested |
| `refresh` | PASS | Full analysis refresh tested on micro targets |
| `rename` | PASS | Top-level alias preview tested |
| `save` | PASS | Positional path and `--path` alias tested to temp BNDB outputs |
| `sections` | PASS | Paged and query forms tested |
| `session list` | PASS | Active instance visibility tested |
| `session start` | PASS | Empty start, explicit ID, preload, and second-instance forms tested |
| `session stop` | PASS | Dedicated audit instances stopped |
| `skill install` | PASS | Copy-mode install to temp destination tested |
| `strings` | PASS | Quick-load guard, full-analysis extraction, regex, no-CRT, min-length, and spill probe tested |
| `struct field delete` | PASS | Preview mutation tested |
| `struct field rename` | PASS | Preview mutation tested |
| `struct field set` | PASS | Preview mutation tested |
| `struct show` | PASS | Declared scratch struct displayed |
| `symbol rename` | PASS | Preview mutation tested |
| `taint backward` | PASS | Known sink slice tested |
| `taint forward` | PASS | Known source-to-sink flow and missing-source error path tested |
| `target clear` | PASS | Sticky pin cleanup tested and restored |
| `target info` | PASS | Selected target info tested |
| `target list` | PASS | Multi-target and single-target forms tested |
| `target use` | PASS | Sticky target pin tested and restored |
| `trace` | PASS | MLIL and HLIL trace forms tested from a resolved callsite |
| `types` | PASS | Listing and query forms tested |
| `types declare` | PASS | Positional, `--file`, `--stdin`, preview, and scratch live declare tested |
| `types show` | PASS | Declared type and listed existing type tested |
| `xrefs` | PASS | JSON and text-limit forms tested |

## Usability Findings

### 1. Stress harness references missing fixtures

The repository contains `tests/stress/run_stress.sh`, but the referenced `tests/fixtures` directory is absent in this checkout. The normal unit suite still passes with skips, but the documented stress script cannot be run as written.

Impact: a user trying to reproduce broad multi-instance stress coverage from the included prompt will hit fixture setup friction immediately.

Recommendation: either restore/check in the fixture build directory, generate those binaries from sources already in the repo, or update the stress script to compile and use the existing local micro-corpus.

### 2. Quick-load state is partly visible, but discoverability can improve

On quick-loaded targets, `sections` and `imports` worked immediately, while `strings` returned a clear error telling the user to run `bn refresh`. Full-analysis follow-up loads verified `strings` on all supplied target labels.

Current code exposes coarse quick/full state through `target info` via `analysis_state`, so the state is not absent. The remaining usability gap is that `target list` does not expose that state and there is no capability map showing which operations require refresh.

Impact: the behavior is correct, but users may reasonably expect quick-loaded targets to support limited strings because several other listing commands already work.

Recommendation: keep the error, surface quick/full state in `target list`, and consider a capability-oriented hint such as which analysis-dependent commands need `refresh`.

### 3. `callsites` JSON shape differs from many paged commands

The audit harness initially expected a dictionary envelope, but `callsites --format json` returned an array. That is valid, and the corrected trace flow passed, but it is a scripting ergonomics mismatch compared with commands such as function search/list.

Impact: automation has to special-case this command shape.

Recommendation: document the array shape explicitly, or normalize around the existing domain-key envelope convention while preserving backward compatibility.

### 4. Batch operation parity should be audited, not assumed broken

Interactive `comment set` supports `--address` and `--function`. Current code also allows batch `set_comment` to target either a function or an address, so the specific address-only concern from the original live harness is stale.

Impact: the remaining question is broader parity: each batch operation should be checked against its closest interactive command so agents can move between single commands and manifests without learning a separate field model.

Recommendation: audit and document the accepted fields for every batch operation. Keep tests for batch `set_comment` function/address parity and invalid mixed locators.

### 5. Common primitive type names are not guaranteed in `types show`

`types show int32_t` failed on the scratch target, while showing a type first obtained from `types --limit` passed. This is not a correctness bug, but it is a common user expectation.

Impact: scripts that assume C typedef aliases exist will be brittle across targets/platforms.

Recommendation: examples should query/list types first or use a declared scratch type for mutation examples. Consider friendlier error text suggesting `bn types --query`.

## Guardrails Verified

- Multiple open targets without `--target` are rejected instead of guessing.
- `decompile --lines` with JSON output is rejected instead of silently ignoring a text-only flag.
- `taint forward` without a source is rejected.
- `callsites` without a scope prints actionable guidance.
- `close <path> --all` is rejected.
- Invalid batch operations roll back before committing.
- Preview mutations do not persist by default.
- Sticky instance/target pins can be set and cleared, and audit state was restored afterward.

## Output Behavior Verified

- `--format text`, `--format json`, and `--format ndjson`.
- `--out` envelopes.
- Raw byte output to file and stdout.
- Large output spill envelope.
- Pagination and truncation warnings.
- Regex-hint stderr for literal queries containing regex metacharacters.

## Mutation Safety

Mutation coverage used either `--preview` or in-memory scratch writes against a temporary analysis session. Temporary live scratch changes included a declared type and a comment, both closed without saving to supplied targets. Saved BNDB outputs were written only to temporary audit paths.

Tested mutation families:

- Symbol rename and top-level rename alias.
- Prototype set/get.
- Local rename/retype.
- Comment set/get/list/delete.
- Type declaration from positional, file, and stdin sources.
- Struct field set/rename/delete.
- Function create preview/no-op.
- Batch apply from stdin and file.

## Residual Risk

This was a broad usability audit, not an exhaustive semantic proof of Binary Ninja analysis correctness. The tests verify that each CLI surface is reachable, returns structured results or clean errors, and behaves safely under common workflows. They do not assert that every decompiler or taint result is semantically complete for every architecture in the supplied corpora.

The most important remaining gap is long-duration stress coverage over very large binaries and fixture-based stress tests, because the repository stress fixture directory is missing.

## Recommended Next Actions

1. Restore or regenerate `tests/fixtures` so `tests/stress/run_stress.sh` can run as documented.
2. Add a sanitized smoke-audit script to the repo that enumerates `_COMMANDS` and verifies each command path with safe temp targets.
3. Document quick-load limitations in command examples that use `strings`.
4. Normalize or document JSON shapes for commands that return bare arrays.
5. Audit remaining batch operation parity against the interactive commands and document accepted fields.
