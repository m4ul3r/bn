---
name: bn-kernel
description: "Use OMP's retained Python kernel for high-volume Binary Ninja reads through bn. Trigger for list-shaped, multi-function, grep-like, or locally filtered analysis where full rows or decompilation should remain in Python variables instead of the transcript."
---

# bn-kernel

Use this skill for high-volume reads that benefit from OMP's retained Python state. Use the `bn` skill for bridge/session lifecycle, small one-off reads, command discovery, and mutations.

## First cell: import from this installed skill

Resolve `<skill-dir>` to the **absolute directory shown when this skill loads**, then run:

```python
from pathlib import Path

skill_dir = Path("<absolute-installed-skill-dir>")
exec((skill_dir / "bootstrap.py").read_text(encoding="utf-8"))
```

The bootstrap is idempotent: rerun it after an eval-kernel exit/reset. Every run
prints `reused` or `reloaded`, the absolute source path, and its source hash. It
removes foreign `bn_kernel` source roots from `sys.path`, puts this source first,
evicts stale modules/bytecode, and validates required API signatures. Always set
the absolute `skill_dir`; an exec context with neither `skill_dir` nor `__file__`
fails with that exact recovery instruction instead of a `NameError`. It cannot
preserve an in-flight cell when a sibling kills the shared Python process; true
crash isolation requires a per-subagent OMP kernel.

## Bind explicitly

Never rely on sticky instance or target pins. `<target-selector>` is the exact
selector shown by `bn -i <instance> target list` (normally a basename, or the
bridge's disambiguated selector when names collide). Bind both and assert the
observed target before trusting any rows:

```python
s = bn_kernel.session(instance="analysis-1", target="<target-selector>")
await s.assert_target("<stem-or-basename-or-absolute-loaded-path>", timeout=30)
s.backend
```

A stem-only check such as `assert_target("sample")` accepts either
`sample.bin` or `sample.bndb`. A basename check such as
`assert_target("sample.bndb")` must match the loaded basename. An absolute path
is strict and must name the actual loaded file: if sibling-BNDB preference
loaded `/x/sample.bndb` for `/x/sample.bin`, assert the `.bndb` path.

> **Concurrent sibling task agents:** OMP currently shares one retained eval
> namespace across those siblings. Module globals can be rebound between cells.
> Ordinary inactive `Session` objects may be retained for inspecting `.last` and
> still emit cross-binding warnings. `scoped()` fails closed only on a foreign
> concurrently active scope, callback reuse for another target, or overlapping
> use of the same callback. Give each target a fresh, uniquely named `async def`.

```python
async def analyze_cell(bound):
    await bound.assert_target("<expected-basename-or-absolute-path>")
    await bound.assert_unannotated()  # required for clean benchmark/dogfood inputs
    rows = await bound.functions(limit=5000)
    large = [row for row in rows if row.get("size", 0) >= 4096]
    return len(rows), bn_kernel.brief(large, "name", "address", "size", n=10)

count, preview = await bn_kernel.scoped(
    analyze_cell,
    instance="analysis-1",
    target="<target-selector>",
)
print(count)
print(preview)
```

`scoped()` keeps the session and bulk rows function-local, refuses returning its
Session, and fails closed on foreign active scopes or callback-binding reuse.
Sequential scopes may coexist with inactive retained Sessions. Do not return bulk
collections into shared globals; process isolation remains a harness requirement.

## Parallel bn-kernel subagents

When two or more concurrent children will use bn-kernel, launch them from an
Eval cell with `agent()` inside `parallel()`. Eval-agent children receive
independent retained kernels on current OMP releases; ordinary `task` children
inherit one eval session and can overwrite globals/modules or kill sibling
cells. A direct-CLI-only fleet may still use an ordinary task batch.

Give every child a self-contained assignment, a unique bridge instance and
target, and require a bounded summary rather than returning its Session or bulk
rows:

```python
results = await parallel([
    lambda: agent(
        "Use bn-kernel. Analyze target A via instance bnk-a and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary.",
        label="A",
    ),
    lambda: agent(
        "Use bn-kernel. Analyze target B via instance bnk-b and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary.",
        label="B",
    ),
    lambda: agent(
        "Use bn-kernel. Analyze target C via instance bnk-c and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary.",
        label="C",
    ),
])
```

This is the required current workaround for parallel retained-kernel analysis;
it does not replace explicit binding, `assert_target()`, `assert_unannotated()`,
or cleanup. If ordinary task children gain per-agent retained-kernel isolation
in a future OMP release, either launch path becomes safe.

`BN_BACKEND=auto|cli|native` selects the default backend; invalid values fail
before client construction. An explicit non-`auto` `backend=` argument wins over
a valid environment default.

Native reads are bounded to 120 seconds by default. Every curated expensive read
accepts `timeout=`: `info`, `assert_target`, `assert_unannotated`, `functions`,
`search`, `function_info`, `decompile`, `disasm`, `il`, `xrefs`, `callsites`,
`strings`, `imports`, and `sections`. Collection timeouts apply to the whole
multi-page/fallback operation. Unknown keywords raise `TypeError` rather than
becoming silent bridge filters. Timeout errors report the requested end-to-end
budget and retain the `bn -i NAME target info` analysis-progress guidance.

Prefer these curated helpers for list-shaped and common reads:

- `await s.info(verbose=False)` exposes `function_count`, `import_symbol_count` (the exact `imports` row count), and `imported_function_count` (callable imported targets); do not compare the latter two as if they were the same population.
- `await s.functions(timeout=..., ...)`, `await s.search(query, timeout=..., ...)` always return row lists; every row has integer `size` plus `size_known`, and `s.last.payload` is the paged envelope. Address/name/size sorts are ascending; pass `reverse=True` for descending/largest-first. Search matches function names/display names only. Regex-shaped zero hits disclose `regex_fallback=True|False` on both backends; `"."` is treated as the all-names regex even when literal dots exist, invalid regex-like input raises, and `exact=True` forces a literal.
- `await s.function_info(identifier, blocks=False)` returns flattened `name`, `address`, `size`, `size_known`, and `imported`; `blocks=True` adds `blocks`. Raw identity remains under `s.last.payload['function']`.
- `await s.decompile(identifier)` returns the non-empty text string with inherited annotation bodies redacted. Use `include_annotations=True` only after an explicit contamination decision. Skipped placeholders raise and direct you to `force_analysis=True`.
- `await s.disasm(identifier, count=N)` / `lines=(START, END)` returns an address-ordered, bridge-sliced string with canonical `0x` addresses. `lines` is a 1-indexed inclusive text-line range, never an address range; out-of-range windows raise.
- `await s.il(identifier)`, `await s.xrefs(identifier, timeout=..., ...)`
- `await s.callsites(callee, timeout=..., ...)` defaults to 100 rows. A bounded high-fan-in payload may have `total=None`; read `total_lower_bound`, `callers_scanned`, `caller_total`, and `scan_truncated` instead of treating null as zero.
- `await s.strings(timeout=..., ...)` defaults to 100 rows to avoid latency cliffs; pass `limit=None` explicitly for a full collection. `imports` and `sections` retain explicit `limit=` control.
- `await s.assert_unannotated()` reports offending comment locations; `allow_contaminated=True` is the explicit bypass and returns the full orientation digest.

Collection and text helpers reject malformed, nested, or silently truncated
payloads instead of returning an empty/list/dict/`None` shape that can be
misread as “no findings.”

**Row keys differ per collection, and you never have to guess.** `functions`
rows key on `address`/`size`, `sections` on `start`/`end`/`length`, `callsites`
nest `callee`/`containing_function`. Every collection payload carries the key
list in band, including on a **zero-hit** page (exactly when there is no row to
read the schema off):

```python
rows = await s.sections(limit=None)
s.last.row_fields          # ['name', 'start', 'end', 'length', 'semantics']
print(bn_kernel.brief(rows, *s.last.row_fields[:3]))
```

`brief()` accepts only a sequence of row mappings; pass
`s.last.payload['items']` (or the returned row list), never the payload dict or
plain text. A missing key raises a `KeyError` that **lists the row's actual
top-level keys**, and adds dotted-path guidance (`brief(rows, "callee.name",
"call_addr")`) only when that row really does nest mappings.

**Bare-decimal addresses, one disclosure shape.** Every containment-enabled read
(`decompile`, `function_info`, `il`, `disasm`, `cfg`, `proto get`, `local list`,
`structured_il`, `defuse`, `resolved_calls`, `possible_values`, `evidence
function`) accepts a decimal address and discloses it identically under
`s.last.payload['resolved_from']`:

```json
{"requested_address": "0x401010", "offset": "+0x10", "input_format": "decimal"}
```

`requested_address` is always normalized to hex. A non-zero `offset` means the
address landed **inside** a function and the read answered for the container; an
exact bare-decimal start is still disclosed, with `offset: "+0x0"`, so a
digit-only token can never be silently mistaken for a symbol name. Exact `0x`
starts and function names carry no `resolved_from` at all, and text mode says
exactly what the JSON says. Prefer `0x`.

**`s.last is None` after a failed operation.** A raise never leaves the previous
read's rows behind as `last`: request failures, mid-pagination failures, function
row-contract and callsite-attribution rejections, invalid regex-like queries, and
timeouts all clear it on both backends. So `s.last` is only ever the result of
the operation that just succeeded — but for the same reason it is *not* a
diagnostic channel for a failure: read the raised `BnError` for that. The one
deliberate exception is a policy refusal over a payload that genuinely
succeeded: `assert_unannotated()` raises on contamination and **keeps** the
orientation digest in `s.last`, which is what you need to decide whether to pass
`allow_contaminated=True`.

## Generic commands and mutations

Discover command-family grammar in-band before guessing arguments:

```python
print(await s.help("evidence"))          # concise by default
print(await s.help("evidence", full=True))
print(await s.help("search"))            # maps to `function search`
catalog = await s.run("capabilities", unwrap=False)
payload = await s.run("evidence", "orient", unwrap=False)
```

`help()` makes no bridge request. Use `full=True` only for the expanded grammar;
use `capabilities` when code needs a machine-readable catalog. `Session.run()`
always uses the CLI artifact path, even when `s.backend == 'native'`. Before any
mutation/save escape hatch, call `await s.assert_target("<expected>")` and follow
the `bn` skill's preview, verification, readback, and save loop.

Use the CLI for lifecycle — bridge start/stop, load polling, and target close all
live there, not in the kernel. For large BNDBs, queue loading instead of tying it
to a single command budget:

```bash
bn session start /path/to/large.bndb --instance-id worker --detach
JOB=<job_id from the start output>
bn -i worker session status "$JOB" --format json   # one job: machine verdict
bn -i worker session status                        # every job: collection
bn -i worker target list                           # selector once terminal
```

**Poll on the job, not on the collection.** `session status <job-id>` returns
`kind: "load_job"` with top-level `job_id`, `state`
(`queued|running|complete|failed`), `terminal`, `succeeded`, the canonical record
under `job`, and the exact `status_command` to re-run. Loop on
`terminal == false`; do **not** re-derive terminality from the state string, and
do not read `succeeded` as a failure while it is `null` (non-terminal means
unknown, not failed). `succeeded` is `true` for `complete` and `false` for
`failed`. An unknown job id is a loud error, not an empty result. Omitting the id
returns the population collection (`kind: "load_jobs"`, `items`, `count`) with no
`terminal`/`succeeded` — one verdict over many jobs would be a lie.

Poll from **bash**, never from an eval cell wait loop: an eval cell that blocks on
a load burns the harness cell timeout and can take the retained kernel with it.

```bash
bn -i worker target close <selector>   # close exactly that target
bn session stop worker                 # then drop the bridge
```

`bn target close <selector>` is the explicit single-target close (the same
implementation as `bn close -t <selector>`, including the unsaved-analysis
warning). It takes no path and no `--all`, so it cannot widen into closing
everything; `bn close --all` is the deliberate spelling for that. Close the
target, then stop the instance.

## Load cost and memory

Full loads can take many minutes and each bridge can consume hundreds of MB. Detached start registers the bridge first and exposes queued/running/complete/failed load state through `session status`; it is the recovery path when a synchronous cold load would exceed 120 seconds. Bound fan-out concurrency, watch RSS with `bn session list`, use `--quick` for raw/container triage (it cannot skip analysis already stored inside a BNDB), and stop instances promptly.

## OMP harness escalation

The skill can detect and contain foreign bindings, but it cannot make one shared
Python process safe for sibling task agents. A sibling exit 130 can still destroy
other agents' in-flight state. OMP owners must provide per-subagent kernel
processes or namespaces; do not weaken `scoped()`/`assert_target()` to work around
that harness boundary.

## Accuracy boundary

HLIL and decompilation can distort access width, conditional guards, loop-invariant bounds, and shift/accumulator structure. Before making any bounds, overflow, or off-by-one claim, confirm the relevant instructions with `await s.disasm(identifier)`; pseudo-C alone is insufficient.
