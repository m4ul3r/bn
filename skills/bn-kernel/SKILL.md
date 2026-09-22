---
name: bn-kernel
description: "Use OMP's retained Python kernel for list-shaped, multi-function, grep-like, or locally filtered Binary Ninja reads where rows should stay in Python rather than the transcript. Bind a bridge instance and target explicitly; use the bn CLI for lifecycle and mutations."
---

# bn-kernel

Use this skill for list-shaped, multi-function, or locally filtered reads in an OMP retained Python kernel. Use `bn` for bridge lifecycle, one-off reads, command discovery, and verified mutations. If no retained Python eval runtime is available, use the direct CLI and bound output with `--limit`, `--lines`, or `--out`.

## Bootstrap and bind

In the first eval cell, use the absolute directory of this installed skill:

```python
from pathlib import Path
skill_dir = Path("<absolute-installed-skill-dir>")
exec((skill_dir / "bootstrap.py").read_text(encoding="utf-8"))
```

Rerun bootstrap after a kernel reset. It prints whether the source was reused or reloaded, plus its path and hash. Bind both selectors and confirm the observed loaded file before trusting data:

```python
s = bn_kernel.session(instance="analysis-1", target="<selector-from-target-list>")
_ = await s.assert_target("<expected-loaded-basename-or-absolute-path>")
```

`assert_target` accepts a stem, exact basename, or strict absolute loaded path. A sidecar or cached `.bndb` may be the actual loaded file, so use `bn -i analysis-1 target list` rather than guessing from the original raw path. A retained kernel does not inherit later shell exports: `session(cache_dir=...)` and `scoped(cache_dir=...)` set `BN_CACHE_DIR` process-wide (last explicit value wins), including for sibling sessions. In a shared kernel, launch it with the bridge's cache directory; use `cache_dir=` only in an isolated kernel, or use the CLI for another directory. For a clean benchmark or from-scratch dogfood input, run `_ = await s.assert_unannotated()` before interpreting rows; assigning the returned digest keeps eval output bounded. `allow_contaminated=True` explicitly permits known annotations; symbol-name exclusions are heuristic, not proof of an untouched database.

Keep large rows inside a function and return a bounded summary:

```python
async def inspect(bound):
    await bound.assert_target("<expected-loaded-basename>")
    rows = await bound.functions(limit=5000)
    return len(rows), bn_kernel.brief(rows, "name", "address", n=8)

count, sample = await bn_kernel.scoped(
    inspect, instance="analysis-1", target="<selector-from-target-list>")
```

In OMP setups where ordinary sibling task agents share one retained eval namespace, globals can be rebound between cells. `bn_kernel.scoped(callback, instance=..., target=...)` protects bindings and refuses overlapping foreign scopes; it does not isolate `os.environ` or an in-flight cell from a sibling killing the shared process. Use isolated kernel processes for concurrent retained reads; otherwise use direct `bn -i ... -t ...` CLI commands. Check the current harness's isolation before relying on a particular agent-launch API.

## Reads and result shape

Curated helpers include `info`, `functions`, `search`, `function_info`, `decompile`, `il`, `disasm`, `xrefs`, `callsites`, `strings`, `imports`, and `sections`. They validate response shapes on both native and CLI backends; malformed or truncated payloads raise rather than becoming empty results. List helpers return rows; inside a scoped callback, `bound.last.payload` holds the complete envelope, including `total`, `has_more`, and `row_fields` when available. `last` is cleared after a failed request. A policy refusal from `assert_unannotated()` is the exception: it retains the successful orientation digest for inspection.

`brief()` takes a list of row mappings, not the envelope or text. Row fields vary by collection; use `bound.last.row_fields` or inspect a row before naming columns. Addresses are hex strings, so use `int(row["address"], 0)` for arithmetic. Prefer `0x` for address inputs: a bare decimal address can resolve to a containing function and is disclosed under `resolved_from` with its offset.

`limit=0` on a curated collection is a one-request schema probe that returns no rows. `row_fields` can be absent for an empty, undeclared kind; ask for a real row instead. A page with `total=None` is not zero findings. For high-fan-in `callsites`, check `total_lower_bound`, `scan_truncated`, `caller_scan_truncated`, and `caller_scan_note` before claiming coverage; either truncation flag makes the rows a lower bound. `strings()` defaults to 100 rows; pass `limit=None` for all rows. `callsites()` defaults to 100 rows. `decompile()` omits stored annotation bodies by default; use `include_annotations=True` only after a contamination decision. Skipped function analysis requires `force_analysis=True` and may be expensive.

A read's `timeout=` is one end-to-end budget, including pagination. Native reads default to 120 seconds. `BN_REQUEST_TIMEOUT` changes the default; `0`, `none`, or `off` disables it. Avoid unlimited collections when the bridge cannot determine a total. `BN_BACKEND=auto|cli|native` selects the backend; an explicit non-auto `backend=` wins. `await s.help("evidence")` is local CLI help, while `await s.run("capabilities", unwrap=False)` returns the registry catalog. `Session.run()` uses the CLI artifact path even if curated reads use the native backend. Use it for generic commands and the `bn` skill's verification loop for any mutation.

## Own only what you start

A workflow that starts a headless bridge should name a unique instance, arm a positive idle timeout as a crash fallback, and stop that exact instance on every reachable exit:

```bash
BN_IDLE_TIMEOUT=3600 bn session start /path/to/sample.bin --instance-id analysis-1
bn -i analysis-1 target close <selector-returned-by-bridge>  # if a target opened
bn session stop analysis-1                               # attempt even if close failed
```

The exact duplicate-ID error means you did not acquire ownership; leave that existing bridge alone. A timed-out or otherwise failed start has uncertain ownership because its child may have registered later, so attempt `bn session stop <your-unique-id>`. Never use `bn close --all` for cleanup of an owned target; it can close unrelated GUI tabs. Cleanup-only work begins with the exact close/stop commands, not with a new start. `BN_IDLE_TIMEOUT` is optional for an existing user-owned bridge and does not replace normal cleanup for one you own.

For large loads, use `bn session start ... --detach` and poll the returned job with `bn -i <id> session status <job-id> --format json`. Poll the named job's `terminal` field from a shell; do not block an eval cell waiting for analysis. Once terminal, take the returned selector, perform reads, close that target, and stop the owned instance.

## Accuracy boundary

HLIL can distort access width, guards, and loop bounds. Before making an overflow or off-by-one claim, confirm the relevant instructions with `await s.disasm(identifier)` and the `bn-vr` methodology.
