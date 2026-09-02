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

## Own and reap every headless bridge

A workflow that starts a headless bridge owns that exact instance until it stops,
unless the exact duplicate-ID rejection below proves the bridge already existed.
Every agent-owned spawn command must begin exactly as shown:

```bash
BN_IDLE_TIMEOUT=3600 bn session start /path/to/binary --instance-id worker
```

For an agent-owned or ambiguously started bridge, begin every lifecycle response
with `BN_IDLE_TIMEOUT=3600 bn session start <target> --instance-id <instance-id>`.
This includes cleanup-only questions after success, failure, or timeout; use the
known path and ID when supplied and keep `<target>` only when the path is unknown.
Omit the spawn line only for the confirmed non-ownership collision below or when
the user explicitly selected another positive idle timeout; merely describing an
idle timeout or setting `BN_SPAWN_TIMEOUT` does not arm this fallback.

A deliberate alternative timeout must be positive; never use `0`, `none`, or
`off` for an agent-owned bridge. The reaper starts after preload, resets after
completed requests, and never fires during an in-flight request or active load
job. It covers hard agent/process death; it does not replace normal cleanup.

On every reachable exit, close only the exact selector returned by the bridge
when a target opened. Never infer it from a path or basename; a basename is valid
when the bridge returned it as the selector. Then always stop the exact owned
instance even if start, load, analysis, or target close failed:

```bash
bn -i worker target close <target-selector>  # when a target opened
bn session stop worker                       # always attempt this exact owned ID
```

Run `session stop` even when the close command fails; a linear close/stop list
must state this guarantee or use a `finally`/trap equivalent. Only the exact
`Bridge instance already exists with id: <instance-id>` start error proves the
workflow never acquired ownership: do not close a target or stop that pre-existing
instance. A timed-out or otherwise failed start is uncertain ownership because
its child may have registered after the harness stopped waiting, so attempt to
stop the unique ID unconditionally. Do not first poll, list, or test whether it
registered, and do not assume absence means no process exists. Never compensate
with `bn close --all`, sticky pins, or another agent's instance.

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
lifecycle = (
    "Start your unique headless bridge with the exact BN_IDLE_TIMEOUT=3600 "
    "assignment on its spawn command. Unless the exact duplicate-ID error proves "
    "you never acquired ownership, on every reachable exit close its exact target "
    "if opened, then always stop its exact instance even if start, load, analysis, "
    "or target close fails; every other failed or timed-out start is ambiguous. "
)
results = await parallel([
    lambda: agent(
        "Use bn-kernel. Analyze target A via instance bnk-a and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary. " + lifecycle,
        label="A",
    ),
    lambda: agent(
        "Use bn-kernel. Analyze target B via instance bnk-b and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary. " + lifecycle,
        label="B",
    ),
    lambda: agent(
        "Use bn-kernel. Analyze target C via instance bnk-c and its exact "
        "selector. Use direct bn only for lifecycle, keep state function-local "
        "with scoped(), and return a bounded summary. " + lifecycle,
        label="C",
    ),
])
```

Each child retains cleanup responsibility for its exact target and instance and
returns only after attempting that exact teardown on every reachable exit. This
is the required current workaround for parallel retained-kernel analysis; it does
not replace explicit binding, `assert_target()`, or `assert_unannotated()`. If
ordinary task children gain per-agent retained-kernel isolation in a future OMP
release, either launch path becomes safe.

`BN_BACKEND=auto|cli|native` selects the default backend; invalid values fail
before client construction. An explicit non-`auto` `backend=` argument wins over
a valid environment default.

Both backends are POSIX-only and Linux-first, like `bn` itself: the bridge speaks
`AF_UNIX`, its peer check needs `SO_PEERCRED`, and the CLI backend hands the child
`bn` its artifact through `/proc/self/fd` (falling back to `/dev/fd`) plus
`pass_fds`. There is no Windows path.

Native reads are bounded to 120 seconds by default. Every curated expensive read
accepts `timeout=`: `info`, `assert_target`, `assert_unannotated`, `functions`,
`search`, `function_info`, `decompile`, `disasm`, `il`, `xrefs`, `callsites`,
`strings`, `imports`, and `sections`. Collection timeouts apply to the whole
multi-page/fallback operation. `BN_REQUEST_TIMEOUT` overrides that budget and is
applied exactly once as one end-to-end deadline: every page of a collection gets
the shrinking remainder, never a fresh copy of the full value, and the child `bn`
is told the remainder so bridge-side cancellation is not scheduled off a budget
the collection already spent. The documented `0`/`none`/`off` spelling disables
the deadline; a collection with no `limit=`, no deadline, and a `total=null` page
that still claims `has_more` is then refused as intrinsically unbounded rather
than paged forever. Unknown keywords raise `TypeError` rather than
becoming silent bridge filters. Timeout errors report the requested end-to-end
budget and retain the `bn -i NAME target info` analysis-progress guidance.

Prefer these curated helpers for list-shaped and common reads:

- `await s.info(verbose=False)` exposes `function_count`, `import_symbol_count` (the exact `imports` row count), and `imported_function_count` (callable imported targets); do not compare the latter two as if they were the same population. It requires the canonical `target_info` shape rather than "some mapping", so another payload cannot answer every one of those questions with a silent "absent": `filename` and `basename` must be present as strings or null; `function_count`, `named_function_count`, `unnamed_function_count` and `imported_function_count` must be present non-negative integers; `import_symbol_count` must be present and either a non-negative integer or `null` (the bridge uses `null` when the imports count fails).
- `await s.functions(timeout=..., ...)`, `await s.search(query, timeout=..., ...)` always return row lists; every row has integer `size` plus `size_known`, and `s.last.payload` is the paged envelope. Address/name/size sorts are ascending; pass `reverse=True` for descending/largest-first. Search matches function names/display names only. Regex-shaped zero hits disclose `regex_fallback=True|False` on both backends; `"."` is treated as the all-names regex even when literal dots exist, invalid regex-like input raises, and `exact=True` forces a literal.
- `await s.function_info(identifier, blocks=False)` returns flattened `name`, `address`, `size`, `size_known`, and `imported`; `blocks=True` adds `blocks`. Raw identity remains under `s.last.payload['function']`.
- `await s.decompile(identifier)` returns the non-empty text string with inherited annotation bodies redacted. Use `include_annotations=True` only after an explicit contamination decision. Skipped placeholders raise and direct you to `force_analysis=True`.
- `await s.disasm(identifier, count=N)` / `lines=(START, END)` returns an address-ordered, bridge-sliced string with canonical `0x` addresses. `lines` is a 1-indexed inclusive text-line range, never an address range; out-of-range windows raise.
- `await s.il(identifier)` returns the non-empty text string, `await s.xrefs(identifier, timeout=..., ...)` a validated row collection.
- `await s.callsites(callee, timeout=..., ...)` defaults to 100 rows. A bounded high-fan-in payload may have `total=None`; read `total_lower_bound`, `callers_scanned`, `caller_total`, and `scan_truncated` instead of treating null as zero. `total` is monotone across a collection's pages: a `None` page can be followed by a page with the exact integer once the caller scan completes, so a long `callsites` collection can legitimately end with a determined total after starting with null ones -- but an already-determined total never reverts to null or changes to a different int.
- `await s.strings(timeout=..., ...)` defaults to 100 rows to avoid latency cliffs; pass `limit=None` explicitly for a full collection. `imports` and `sections` retain explicit `limit=` control.
- `await s.assert_unannotated()` reports offending comment locations; `allow_contaminated=True` is the explicit bypass and returns the full orientation digest. It fails **closed**: the digest must be a mapping whose `existing_annotations` is a mapping carrying non-negative integer `comments`, `function_comments` and `user_symbols`. An unreadable digest raises instead of collapsing to "zero comments", and `allow_contaminated=True` waives the contamination *policy*, never that payload contract.

Every collection and text helper validates **after** the backend branch, so `cli`
and `native` enforce the same shape: malformed, nested, or silently truncated
payloads raise instead of returning an empty/list/dict/`None` shape that can be
misread as “no findings.”

Paged reads also require each page to publish an integer `offset` equal to the one
requested, and hold `total` to a **monotone** contract: `null` means "not
determined yet" (a capped scan, e.g. high-fan-in `callsites`) and an integer means
"determined", so `null`→int is a legal refinement, while int→`null` and a changed
int are rejected as bridge drift. Progress is otherwise tracked by the caller's own
arithmetic, so a bridge that ignores pagination would silently return duplicate rows.

**`limit=0` asks for the schema, not the rows.** Passing `limit=0` to a curated
collection helper (or to `Session.all()` / `bn.Client.collect()`) performs exactly
ONE real request and returns no rows. The bridge enforces `limit >= 1`, so on the
wire this is an internal one-row **probe** at your requested `offset`; the probed
row is validated through the normal page contract and then discarded. It is never
returned to you and never lands in `s.last.value`. The envelope you get back
reports `returned=0` and `limit=0`, keeps the bridge-owned `kind`, `total` and
`row_fields`, and reports `has_more=true` whenever the probe found a row — from a
zero-row position, that row alone proves more exists at this offset.

`Result` projects the common probe verdicts directly:

```python
s.last.returned   # 0
s.last.has_more   # True when the probe found a row
```

`s.last.payload` remains the complete envelope and is authoritative for fields
without a `Result` convenience property.

`row_fields` may legitimately be **absent** from a `limit=0` envelope: the bridge
derives it from a declared schema or from an actual row, so an *empty* collection
of a kind it does not pre-declare has neither source. The pre-declared kinds are
`functions`, `strings`, `imports`, `exports`, `sections`, `xrefs` and `callsites`;
an empty `types`, `tags`, `comments` or similar read can come back without it.
A `row_fields` that IS present is always validated as a list of strings. Treat
absence as "no schema available yet", not as an error — ask for a real row instead.

This is a programmatic-only shape. Wire-level `bn <paged command> --limit 0` stays
rejected at parse time (exit 2), because the bridge would reject a zero limit and
the CLI declines to round-trip to that error.

**Row keys differ per collection, and you never have to guess.** `functions`
rows key on `address`/`size`, `sections` on `start`/`end`/`length`, `callsites`
nest `callee`/`containing_function`. Collection payloads carry the key list in
band, including on a **zero-hit** page for any pre-declared kind — exactly when
there is no row to read the schema off (the caveat above applies: an empty page of
a kind the bridge does not pre-declare has no schema to publish):

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

Curated address fields are canonical hexadecimal strings (`"0x401000"`), not
integers. Use `int(row["address"], 0)` for arithmetic; do not call `hex()` on
an address returned by bn-kernel. This holds for `functions.address`,
`sections.start`/`end`, `callsites.call_addr`, and containment's
`requested_address` alike.

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
BN_IDLE_TIMEOUT=3600 bn session start /path/to/large.bndb --instance-id worker --detach
JOB=<job_id from the start output>
bn -i worker session status "$JOB" --format json   # one job: machine verdict
bn -i worker session status                        # every job: collection
bn -i worker target list                           # selector once terminal
```

**Poll on the job, not on the collection.** `session status <job-id>` returns
`kind: "load_job"` with top-level `job_id`, `state`
(`queued|running|complete|failed`), `terminal`, `succeeded`, the canonical record
under `job`, and `status_command`. Loop on
`terminal == false`; do **not** re-derive terminality from the state string, and
do not read `succeeded` as a failure while it is `null` (non-terminal means
unknown, not failed). `succeeded` is `true` for `complete` and `false` for
`failed`. An unknown job id is a loud error, not an empty result. Omitting the id
returns the population collection (`kind: "load_jobs"`, `items`, `count`) with no
`terminal`/`succeeded` — one verdict over many jobs would be a lie — and its items
are the raw job records, which carry no `status_command`.

`status_command` is the exact command to re-run, but only a bridge that has an
instance id can name itself on a fresh command line. Any headless bridge does
(that is what `--instance-id` sets, and the detached-start flow above always
sets it). A GUI-loaded bridge does not: it has no unambiguous CLI selector, so it
publishes `status_command: null` instead of a command that cannot address it. The
key is always present, so `null` is distinguishable from a missing field. On
`null`, poll through the client or connection you already have bound to that
bridge, or treat the command as unavailable — do not synthesize one.

Poll from **bash**, never from an eval cell wait loop: an eval cell that blocks on
a load burns the harness cell timeout and can take the retained kernel with it.

```bash
bn -i worker target close <selector>   # close exactly that target
bn session stop worker                 # then drop the bridge
```

The stop attempt is unconditional for an owned instance: run it even if target
close fails. The exact duplicate-ID start error proves non-ownership, so do not
close or stop that pre-existing instance. Any other failed or timed-out start
still triggers an exact stop attempt because registration may have completed
after the caller stopped waiting.

`bn target close <selector>` is the explicit single-target close (the same
implementation as `bn close -t <selector>`, including the unsaved-analysis
warning). It takes no path and no `--all`, so it cannot widen into closing
everything; `bn close --all` is the deliberate spelling for that. Close the
target, then stop the instance.

## Load cost and memory

Full loads can take many minutes and each bridge can consume hundreds of MB. Detached start registers the bridge first and exposes queued/running/complete/failed load state through `session status`; it is the recovery path when a synchronous cold load would exceed 120 seconds. Bound fan-out concurrency, watch RSS with `bn session list`, use `--quick` for raw/container triage (it cannot skip analysis already stored inside a BNDB), stop every owned instance deterministically as soon as its work ends, and rely on one-hour idle reaping only as the crash fallback.

For high-fanout cold starts, the orchestration tool's command timeout must exceed
`BN_SPAWN_TIMEOUT`; otherwise the harness can kill `bn session start` while its
new-session child continues registering. On a heavily loaded host, set a larger
spawn budget (for example `BN_SPAWN_TIMEOUT=180`) and give the surrounding tool a
strictly larger timeout. That assignment is additive to the required
`BN_IDLE_TIMEOUT=3600` on the same agent-owned spawn, never a substitute for it.
It changes the registration budget, not the detached load-job budget; continue
polling the exact job separately. In a 16-way dogfood
run, every start and load succeeded with that budget, but two start commands took
more than 30 seconds (maximum 34.3 seconds).

## OMP harness escalation

The skill can detect and contain foreign bindings, but it cannot make one shared
Python process safe for sibling task agents. A sibling exit 130 can still destroy
other agents' in-flight state. OMP owners must provide per-subagent kernel
processes or namespaces; do not weaken `scoped()`/`assert_target()` to work around
that harness boundary.

## Accuracy boundary

HLIL and decompilation can distort access width, conditional guards, loop-invariant bounds, and shift/accumulator structure. Before making any bounds, overflow, or off-by-one claim, confirm the relevant instructions with `await s.disasm(identifier)`; pseudo-C alone is insufficient.
