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
import sys

skill_dir = Path("<absolute-installed-skill-dir>")
sys.path.insert(0, str(skill_dir / "src"))
previous = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import bn_kernel
finally:
    sys.dont_write_bytecode = previous
```

Do not assume the skill's Python module is installed into OMP's interpreter. The adapter requires only a working `bn` executable on `PATH`; when the interpreter can also import `bn.Client`, `backend='auto'` selects the native transport.

## Bind explicitly

Never rely on sticky instance or target pins. In a single-agent OMP session,
bind both and assert the observed target before trusting any rows:

```python
s = bn_kernel.session(instance="analysis-1", target="<target-selector>")
await s.assert_target("<expected-basename-or-absolute-path>")
s.backend
```

> **Concurrent sibling task agents:** OMP currently shares one retained eval
> namespace across those siblings. Module globals such as `s`, `rows`, `lines`,
> and `decomp` can be rebound by another agent between cells. Distinct live
> bindings emit a runtime warning, but a warning cannot isolate Python globals.
> Use the direct `bn -i/-t` CLI for parallel agents. If kernel use is required,
> complete the whole operation in one function-local callback:

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

`scoped()` keeps the session and bulk rows function-local; do not return the
session or bulk collections into shared globals. True per-subagent namespaces
must be provided by the OMP harness.

Use `backend='cli'` to force the zero-extra-install executable path for reproducibility. Use `backend='native'` to diagnose whether the OMP interpreter can import the supported `bn.Client`; its error explains how to fall back.

Prefer these curated helpers for list-shaped and common reads:

- `await s.info(verbose=False)`
- `await s.functions(...)`, `await s.search(query, ...)` (case-insensitive substring; regex-like zero-hit queries retry as regex like the CLI)
- `await s.function_info(identifier, blocks=False)` (identity fields are flattened at top level; `s.last.payload` keeps the raw nested bridge shape)
- `await s.decompile(identifier)`, `await s.disasm(identifier)`, `await s.il(identifier)`
- `await s.xrefs(identifier, ...)`
- `await s.callsites(callee, ...)` for all callers, or restrict with `within="caller"` / `within=[...]`
- `await s.strings(...)`, `await s.imports(...)`, `await s.sections(...)`

`Session.last.payload` retains the complete response and pagination metadata. `brief()` formats bounded output. Large variables stay out of the transcript only while you do **not** print or display them; inspect counts, selected fields, and bounded slices instead.

Normal Python filtering, joins, counts, and regexes over retained variables make no additional bridge request. Keep the rows in variables across cells rather than fetching them again.

## Generic commands and mutations

Discover command-family grammar in-band before guessing arguments:

```python
print(await s.help("evidence"))
payload = await s.run("evidence", "orient", unwrap=False)
```

`help()` captures argparse help directly and makes no bridge request. `Session.run()` always uses the CLI artifact path, even when `s.backend == 'native'`, so it preserves the complete command surface without duplicating the command registry. Before any mutation/save escape hatch, call `await s.assert_target("<expected>")`. Continue to follow the `bn` skill's preview, verification, readback, and save loop for every mutation. Use the CLI directly for session start/list/stop, load, close, refresh, and save.
 
## Load cost and memory

Full loads run Binary Ninja analysis to completion. They can take seconds to minutes under contention, not a fixed few seconds. Each headless bridge can consume hundreds of MB; a large or complex BNDB may be OOM-killed. Bound fan-out concurrency, watch RSS with `bn session list`, use `--quick` for container-level triage, and stop instances promptly.

## Accuracy boundary

HLIL and decompilation can distort access width, conditional guards, loop-invariant bounds, and shift/accumulator structure. Before making any bounds, overflow, or off-by-one claim, confirm the relevant instructions with `await s.disasm(identifier)`; pseudo-C alone is insufficient.
