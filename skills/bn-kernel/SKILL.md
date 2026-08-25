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

The bootstrap is idempotent: rerun it after an eval-kernel exit/reset. It hashes
the installed source, evicts stale `bn_kernel*` modules, removes stale
`__pycache__`, reimports, and validates required API signatures. A signature
mismatch fails loudly. It cannot preserve an in-flight cell when a sibling kills
the shared Python process; true crash isolation requires a per-subagent OMP
kernel.

## Bind explicitly

Never rely on sticky instance or target pins. In a single-agent OMP session,
bind both and assert the observed target before trusting any rows:

```python
s = bn_kernel.session(instance="analysis-1", target="<target-selector>")
await s.assert_target("<name-or-name.bndb-or-absolute-path>", timeout=30)
s.backend
```

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

Native reads are bounded to 120 seconds by default. Collection helpers accept an
explicit `timeout=` that applies to the whole multi-page operation; unknown
keywords raise `TypeError` rather than becoming silent bridge filters. Timeout
errors point to `bn -i NAME target info` for analysis progress.

Prefer these curated helpers for list-shaped and common reads:

- `await s.info(verbose=False)`
- `await s.functions(timeout=..., ...)`, `await s.search(query, timeout=..., ...)` always return row lists; every row has integer `size` plus `size_known`, and `s.last.payload` is the paged envelope.
- `await s.function_info(identifier, blocks=False)` flattens identity fields at top level while preserving the raw nested payload.
- `await s.decompile(identifier)` returns non-empty `payload['text']`; function identity remains under `s.last.payload['function']`. Skipped/“taking too long” placeholders raise and direct you to `force_analysis=True`.
- `await s.disasm(identifier, count=N)` / `lines=(START, END)` returns an address-ordered, bounded string with `0x`-prefixed addresses.
- `await s.il(identifier)`, `await s.xrefs(identifier, timeout=..., ...)`
- `await s.callsites(callee, timeout=..., ...)` always returns attributed rows.
- `await s.strings(timeout=..., ...)`, `await s.imports(timeout=..., ...)`, `await s.sections(timeout=..., ...)`

Collection and text helpers reject malformed, nested, or silently truncated
payloads instead of returning an empty/list/dict/`None` shape that can be
misread as “no findings.” `brief()` accepts only a sequence of row mappings;
pass `s.last.payload['items']`, never the payload dict or plain text.

## Generic commands and mutations

Discover command-family grammar in-band before guessing arguments:

```python
print(await s.help("evidence"))
payload = await s.run("evidence", "orient", unwrap=False)
```

`help()` captures argparse help directly and makes no bridge request. `Session.run()` always uses the CLI artifact path, even when `s.backend == 'native'`, so it preserves the complete command surface without duplicating the command registry. Before any mutation/save escape hatch, call `await s.assert_target("<expected>")`. Continue to follow the `bn` skill's preview, verification, readback, and save loop for every mutation. Use the CLI directly for session start/list/stop, load, close, refresh, and save.
 
## Load cost and memory

Full loads run Binary Ninja analysis to completion. They can take seconds to minutes under contention, not a fixed few seconds. Each headless bridge can consume hundreds of MB; a large or complex BNDB may be OOM-killed. Bound fan-out concurrency, watch RSS with `bn session list`, use `--quick` for container-level triage, and stop instances promptly.

## OMP harness escalation

The skill can detect and contain foreign bindings, but it cannot make one shared
Python process safe for sibling task agents. A sibling exit 130 can still destroy
other agents' in-flight state. OMP owners must provide per-subagent kernel
processes or namespaces; do not weaken `scoped()`/`assert_target()` to work around
that harness boundary.

## Accuracy boundary

HLIL and decompilation can distort access width, conditional guards, loop-invariant bounds, and shift/accumulator structure. Before making any bounds, overflow, or off-by-one claim, confirm the relevant instructions with `await s.disasm(identifier)`; pseudo-C alone is insufficient.
