# Modularize `bridge.py` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the triple-maintained op list and the 206-method `BinaryNinjaBridge` god-class with a declarative `@op` registry plus focused domain modules behind a `BridgeContext` seam — with zero behaviour change.

**Architecture:** The bridge class becomes a facade of thin delegating methods (the existing `_taint` → `taint_engine.py` precedent). Op logic moves to modules that receive a `BridgeContext` seam instead of reaching into the class. `READ_LOCKED_OPS`/`WRITE_LOCKED_OPS` and `_dispatch_on_main` are derived from one `@op`-populated registry. Delivered as 5 staged, individually-green commits on `feat/33-modularize-bridge` → one PR into `m4ul3r`.

**Tech Stack:** Python ≥3.14, `uv`, pytest (mocked `binaryninja`), Binary Ninja headless API at `/opt/binaryninja`, the `bn` CLI for the firmware regression dogfood.

**Design spec:** `docs/superpowers/specs/2026-06-11-modularize-bridge-design.md` — read it first.

**Governing invariant (every task):** `uv run pytest` stays fully green (638 tests), and `python -c "import bn_agent_bridge.bridge"` (import smoke) succeeds. Behaviour is preserved; if a change makes a *correctness* improvement or alters output, STOP — that is out of scope for this refactor.

---

## File Structure

**Created:**
- `plugin/bn_agent_bridge/op_registry.py` — `@op` decorator, `OpSpec`, `_OPS` table, duplicate guard, lock-set derivation.
- `plugin/bn_agent_bridge/seam.py` — `BridgeContext` + resolution/ABI/address-context helpers + relocated `_find_type`/`_render_type_layout`.
- `plugin/bn_agent_bridge/il_format.py` — pure IL/HLIL/disasm rendering (state-free).
- `plugin/bn_agent_bridge/vars.py` — variable discovery/serialization (state-free).
- `plugin/bn_agent_bridge/read_decompile.py`, `read_listing.py`, `read_xrefs.py`, `read_evidence.py`, `read_taint_slice.py`, `read_types.py`, `read_misc.py` — read-op domains.
- `plugin/bn_agent_bridge/create_comments.py` — `function_create`, comments, bundle, py_exec.
- `plugin/bn_agent_bridge/mutation_engine.py` — the batch mutation engine.
- `tests/test_op_registry.py` — registry contract + lock-set golden tests.

**Modified:**
- `plugin/bn_agent_bridge/bridge.py` — dispatch becomes registry-driven; handler bodies move out; methods become delegating shims; keeps exporting `start_bridge`, `start_headless`, `ui`, `READ_LOCKED_OPS`, `WRITE_LOCKED_OPS`, `BinaryNinjaBridge`, `_dispatch_on_main`.

**Local-only (NOT committed — leak hygiene, see spec §5.2):**
- `/tmp/bn33_dogfood/sweep.py`, `/tmp/bn33_dogfood/baseline.json`, `/tmp/bn33_dogfood/*.json` — firmware regression harness + captures.

---

## Stage 0 — Pre-flight: capture the characterization baseline

The refactor's correctness oracle is twofold: the full unit suite, and a byte-identical firmware op-sweep. Capture both on the **pre-refactor tip** before touching code.

### Task 0.1: Confirm green baseline + record exact op membership

**Files:** none (read-only).

- [ ] **Step 1: Run the full suite, record the count.**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: all pass (≈638 passed). If anything fails on the clean tip, STOP and report — the baseline must be green.

- [ ] **Step 2: Snapshot the exact current lock-set membership** (this becomes the golden test's frozen literals).

Run: `uv run python -c "import sys; sys.path.insert(0,'plugin'); import bn_agent_bridge.bridge as b; print('READ', sorted(b.READ_LOCKED_OPS)); print('WRITE', sorted(b.WRITE_LOCKED_OPS))"`
Expected: prints the 33 read ops and 16 write ops. Copy this output verbatim — Task 1.1 hard-codes it.

- [ ] **Step 3: Enumerate every op the dispatch handles** (the registry must cover all of these).

Run: `uv run python -c "import re; s=open('plugin/bn_agent_bridge/bridge.py').read(); m=re.findall(r'op == \"([a-z_]+)\"', s); print(len(set(m)), sorted(set(m)))"`
Expected: the full op list incl. `doctor, list_targets, target_info, refresh, shutdown, load_binary, …` (the 51 ops). Save it — Task 1.5 asserts the registry covers exactly this set.

### Task 0.2: Build the firmware dogfood harness and capture the baseline

**Files:** Create `/tmp/bn33_dogfood/sweep.py` (local only).

- [ ] **Step 1: Verify BN + a real ARM target are present.**

Run:
```bash
ls /opt/binaryninja/python >/dev/null && echo BN_OK
ls "/tmp/DMH-2000NEX/update/CUST_PIO.BIN_extract/12754944-67129344.squashfs_v4_le_extract/bin/aa_accessory_service" && echo TARGET_OK
```
Expected: `BN_OK` and `TARGET_OK`. If BN is missing, the firmware gate is skipped and the unit suite is the sole oracle (note it in the PR).

- [ ] **Step 2: Write the sweep harness.** It loads one binary headlessly, runs a fixed op sweep via the `bn` CLI, normalizes volatile fields, and writes one JSON blob.

Create `/tmp/bn33_dogfood/sweep.py`:
```python
"""Local firmware regression harness for issue #33. NOT committed."""
import json, re, subprocess, sys, hashlib
from pathlib import Path

BN = "bn"
TARGET = sys.argv[1]
OUT = Path(sys.argv[2])

# A pure routing/extraction refactor must produce identical output for every
# one of these. Mutations use --preview (apply->capture->revert: no state left).
def run(instance, *args):
    p = subprocess.run([BN, "--instance", instance, *args, "--format", "json"],
                       capture_output=True, text=True, timeout=300)
    return {"args": list(args), "rc": p.returncode, "out": p.stdout, "err": p.stderr}

def normalize(blob: str) -> str:
    # Strip volatile fields so only semantic content is compared.
    for pat in [r'"pid":\s*\d+', r'"socket_path":\s*"[^"]*"', r'"instance_id":\s*"[^"]*"',
                r'"started_at":\s*"[^"]*"', r'"plugin_build_id":\s*"[^"]*"',
                r'/tmp/[^"]*\.sock', r'"path":\s*"[^"]*"']:
        blob = re.sub(pat, '<v>', blob)
    return blob

def main():
    start = subprocess.run([BN, "session", "start", TARGET, "--format", "json"],
                           capture_output=True, text=True, timeout=600)
    sess = json.loads(start.stdout)
    inst = sess["instances"][0]["instance_id"] if "instances" in sess else sess["instance_id"]
    try:
        # Pick a deterministic function to exercise function-scoped ops.
        fns = json.loads(run(inst, "function", "list", "--limit", "5")["out"])
        fid = None
        items = fns.get("functions") or fns.get("items") or []
        if items:
            fid = items[0].get("address") or items[0].get("identifier") or items[0].get("name")
        ops = [
            ("target", "info"),
            ("function", "list", "--limit", "20"),
            ("function", "search", "init"),
            ("strings", "--limit", "30"),
            ("section", "list") if False else ("sections",),  # adjust to real CLI verb
            ("imports",),
            ("types",),
        ]
        results = {}
        for op in ops:
            results[" ".join(op)] = normalize(run(inst, *op)["out"])
        if fid is not None:
            for op in [("decompile", str(fid)), ("il", str(fid)), ("disasm", str(fid)),
                       ("xrefs", str(fid)), ("function", "info", str(fid))]:
                results[" ".join(op)] = normalize(run(inst, *op)["out"])
        OUT.write_text(json.dumps(results, indent=1, sort_keys=True))
        print("WROTE", OUT, "ops:", len(results))
    finally:
        subprocess.run([BN, "session", "stop", inst], capture_output=True, text=True, timeout=60)

if __name__ == "__main__":
    main()
```

> **Note for the executor:** the exact `bn` subcommand verbs/flags above are a starting list — before relying on it, run `bn --help` and each `bn <group> --help` once, fix any verb/flag mismatches (e.g. `sections` vs `section list`, the function-id field name), and re-run until every op returns rc 0 on the baseline. The harness is only useful once it runs clean on the baseline.

- [ ] **Step 3: Capture the baseline** (still on the pre-refactor tip — the spec commit doesn't change behaviour, so capturing on `feat/33-modularize-bridge` HEAD is fine).

Run:
```bash
mkdir -p /tmp/bn33_dogfood
uv run python /tmp/bn33_dogfood/sweep.py \
  "/tmp/DMH-2000NEX/update/CUST_PIO.BIN_extract/12754944-67129344.squashfs_v4_le_extract/bin/aa_accessory_service" \
  /tmp/bn33_dogfood/baseline.json
```
Expected: `WROTE …/baseline.json ops: N` with N ≥ 10 and every op rc 0. This file is the regression oracle for every later stage.

---

## Stage 1 — Declarative op registry (issue step 1)

The headline fix. Handlers stay as methods; binders (verbatim if-arm bodies) register them; dispatch + lock-sets derive from the registry.

### Task 1.1: Characterization test — pin the current lock-set membership

**Files:** Create `tests/test_op_registry.py`.

- [ ] **Step 1: Write the golden test** using the exact sets recorded in Task 0.1 Step 2. (Substitute the real members you copied; the lists below are the expected content.)

```python
from __future__ import annotations
import importlib

import bn_agent_bridge.bridge as bridge

# Frozen from the pre-refactor tip (Task 0.1). If a future PR legitimately adds
# an op, update these two sets in the SAME commit — that is the single point of
# truth this test enforces.
EXPECTED_READ = {
    "doctor", "list_targets", "target_info", "function_info", "get_prototype",
    "list_functions", "list_locals", "search_functions", "callsites", "decompile",
    "il", "structured_il", "defuse", "resolved_calls", "possible_values", "taint",
    "disasm", "function_evidence", "xrefs", "field_xrefs", "pointer_table",
    "message_lens", "init_arrays", "backward_slice", "types", "type_info",
    "strings", "imports", "bundle_function", "get_comment", "list_comments",
    "sections", "read",
}
EXPECTED_WRITE = {
    "py_exec", "function_create", "rename_symbol", "set_comment", "delete_comment",
    "set_prototype", "local_rename", "local_retype", "struct_field_set",
    "struct_field_rename", "struct_field_delete", "types_declare", "batch_apply",
    "refresh", "close_binary", "save_database",
}

def test_read_locked_ops_membership_unchanged():
    assert set(bridge.READ_LOCKED_OPS) == EXPECTED_READ

def test_write_locked_ops_membership_unchanged():
    assert set(bridge.WRITE_LOCKED_OPS) == EXPECTED_WRITE

def test_no_op_is_both_read_and_write():
    assert set(bridge.READ_LOCKED_OPS).isdisjoint(bridge.WRITE_LOCKED_OPS)

def test_load_binary_and_shutdown_are_unlocked():
    assert "load_binary" not in bridge.READ_LOCKED_OPS
    assert "load_binary" not in bridge.WRITE_LOCKED_OPS
    assert "shutdown" not in bridge.READ_LOCKED_OPS
    assert "shutdown" not in bridge.WRITE_LOCKED_OPS
```

- [ ] **Step 2: Run it against the CURRENT code** (registry not built yet — these pass against the existing literals, proving they pin today's behaviour).

Run: `uv run pytest tests/test_op_registry.py -v`
Expected: 4 passed. (If any fail, your EXPECTED_* sets don't match Task 0.1 — fix them, don't touch bridge.py.)

- [ ] **Step 3: Commit the characterization net** (it guards the refactor that follows).

```bash
git add tests/test_op_registry.py
git commit -m "test(#33): pin bridge lock-set membership before registry refactor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 1.2: Create the registry module

**Files:** Create `plugin/bn_agent_bridge/op_registry.py`; Test `tests/test_op_registry.py`.

- [ ] **Step 1: Write failing tests for the registry primitives.**

Append to `tests/test_op_registry.py`:
```python
from bn_agent_bridge import op_registry

def test_op_decorator_registers_and_derives_locks():
    reg = op_registry.OpRegistry()

    @reg.op("alpha", lock="read")
    def _bind_alpha(bridge, params, target):
        return ("alpha", target)

    @reg.op("beta", lock="write")
    def _bind_beta(bridge, params, target):
        return "beta"

    assert reg.read_locked_ops() == {"alpha"}
    assert reg.write_locked_ops() == {"beta"}
    assert reg.spec("alpha").binder(None, {}, "t") == ("alpha", "t")

def test_duplicate_op_registration_raises():
    reg = op_registry.OpRegistry()

    @reg.op("dup", lock="read")
    def _a(bridge, params, target): return 1

    with pytest.raises(ValueError, match="duplicate op registration"):
        @reg.op("dup", lock="read")
        def _b(bridge, params, target): return 2

def test_invalid_lock_class_raises():
    reg = op_registry.OpRegistry()
    with pytest.raises(ValueError, match="invalid lock class"):
        @reg.op("x", lock="sometimes")
        def _x(bridge, params, target): return 1

def test_escalation_is_stored():
    reg = op_registry.OpRegistry()

    @reg.op("e", lock="read", escalation=lambda p: bool(p.get("force")))
    def _e(bridge, params, target): return 1

    assert reg.spec("e").lock_escalation({"force": True}) is True
```
Add `import pytest` to the test file's imports.

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_op_registry.py -k "registry or duplicate or invalid_lock or escalation" -v`
Expected: FAIL — `ModuleNotFoundError: bn_agent_bridge.op_registry`.

- [ ] **Step 3: Implement `op_registry.py`.**

Create `plugin/bn_agent_bridge/op_registry.py`:
```python
"""Declarative op registry for the bridge dispatch.

Mirrors the CLI's @command/_COMMANDS registry (src/bn/cli.py). Each op is
declared once via @op; the read/write lock sets and the dispatch routing are
both DERIVED from this single source, replacing the triple-maintained list
(READ_LOCKED_OPS / WRITE_LOCKED_OPS / the _dispatch_on_main if-chain).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Binder = Callable[[Any, dict[str, Any], "str | None"], Any]
Escalation = Callable[[dict[str, Any]], bool]

_LOCK_CLASSES = ("read", "write", "none")


@dataclass(frozen=True)
class OpSpec:
    name: str
    lock: str
    binder: Binder
    lock_escalation: Escalation | None = None


class OpRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}

    def op(self, name: str, *, lock: str, escalation: Escalation | None = None) -> Callable[[Binder], Binder]:
        if lock not in _LOCK_CLASSES:
            raise ValueError(f"invalid lock class {lock!r} for op {name!r}; expected one of {_LOCK_CLASSES}")

        def decorator(binder: Binder) -> Binder:
            if name in self._ops:
                raise ValueError(f"duplicate op registration: {name!r}")
            self._ops[name] = OpSpec(name=name, lock=lock, binder=binder, lock_escalation=escalation)
            return binder

        return decorator

    def spec(self, name: str) -> OpSpec | None:
        return self._ops.get(name)

    def names(self) -> set[str]:
        return set(self._ops)

    def read_locked_ops(self) -> set[str]:
        return {n for n, s in self._ops.items() if s.lock == "read"}

    def write_locked_ops(self) -> set[str]:
        return {n for n, s in self._ops.items() if s.lock == "write"}


# The bridge's single global registry. bridge.py registers all ops against it.
REGISTRY = OpRegistry()
op = REGISTRY.op
```

- [ ] **Step 4: Run to verify pass.**

Run: `uv run pytest tests/test_op_registry.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add plugin/bn_agent_bridge/op_registry.py tests/test_op_registry.py
git commit -m "feat(#33): add declarative bridge op registry primitives

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 1.3: Register all 51 ops as binders in `bridge.py`

**Files:** Modify `plugin/bn_agent_bridge/bridge.py` (add a binder block after the `BinaryNinjaBridge` class; the if-chain is removed in Task 1.4).

**The transformation:** every `if op == "X": return <BODY>` arm in `_dispatch_on_main` (`bridge.py:883`–`1101`) becomes a module-level binder:
```python
@op("X", lock=<class>)
def _bind_X(bridge, params, target):
    return <BODY, with `self` rewritten to `bridge`>
```
The lock `<class>` comes from the golden sets (read/write) or is `none` for `load_binary`/`shutdown`. Copy each `<BODY>` **verbatim** — do not "improve" extraction.

- [ ] **Step 1: Add the binder block.** Place after the class definition. Worked examples covering every shape (transcribe the rest the same way from the if-chain):

```python
from .op_registry import op  # add near the existing `from . import taint_engine as _taint`

# ---- op binders: each reproduces one former _dispatch_on_main if-arm ----

# trivial (no params)
@op("doctor", lock="read")
def _bind_doctor(bridge, params, target):
    return bridge._doctor()

# inline arm (no handler method existed — body was inline in the if-chain)
@op("list_targets", lock="read")
def _bind_list_targets(bridge, params, target):
    return bridge.targets.refresh()

@op("shutdown", lock="none")
def _bind_shutdown(bridge, params, target):
    bridge._shutdown_event.set()
    return {"shutting_down": True}

# selector/target arm
@op("target_info", lock="read")
def _bind_target_info(bridge, params, target):
    return bridge._target_info(params.get("selector") or target)

@op("refresh", lock="write")
def _bind_refresh(bridge, params, target):
    return bridge._refresh(target)

# no-lock, self-locking handler
@op("load_binary", lock="none")
def _bind_load_binary(bridge, params, target):
    return bridge._load_binary(
        str(params["path"]),
        prefer_bndb=_validate_bool(params.get("prefer_bndb"), label="prefer_bndb", default=True),
        quick=_validate_bool(params.get("quick"), label="quick", default=False),
    )

# typed-kwargs read op WITH lock escalation
@op("decompile", lock="read",
    escalation=lambda p: _validate_bool(p.get("force_analysis"), label="force_analysis", default=False))
def _bind_decompile(bridge, params, target):
    return bridge._decompile(
        target,
        params["identifier"],
        addresses=bool(params.get("addresses")),
        force_analysis=bool(params.get("force_analysis")),
    )

# paged read op
@op("list_functions", lock="read")
def _bind_list_functions(bridge, params, target):
    return bridge._list_functions(
        target,
        min_address=params.get("min_address"),
        max_address=params.get("max_address"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if "limit" in params else None,
        count_only=bool(params.get("count_only", False)),
    )

# single-op mutation wrapper (pattern shared by rename_symbol/set_comment/
# delete_comment/set_prototype/local_rename/local_retype/struct_field_set/
# struct_field_rename/struct_field_delete/types_declare)
@op("rename_symbol", lock="write")
def _bind_rename_symbol(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "rename_symbol", **params}])

# batch_apply (manifest re-derivation — copy verbatim from bridge.py:1091-1099)
@op("batch_apply", lock="write")
def _bind_batch_apply(bridge, params, target):
    manifest = dict(params)
    preview = bool(manifest.get("preview"))
    chosen = manifest.get("target") or target
    selector = str(chosen) if chosen is not None else None
    operations = list(manifest.get("ops") or [])
    return bridge._mutation(selector, preview, operations)
```
Transcribe the remaining ops the same way (full list, with lock class):
`read`: `target_info✓, get_prototype, list_locals, il, structured_il, defuse, resolved_calls, possible_values, taint, disasm, function_evidence, xrefs, field_xrefs, pointer_table, message_lens, init_arrays, backward_slice, types, type_info, strings, imports, sections, read, search_functions, callsites, function_info, bundle_function, get_comment, list_comments`.
`write`: `close_binary, save_database, function_create, py_exec, set_comment, delete_comment, set_prototype, local_rename, local_retype, struct_field_set, struct_field_rename, struct_field_delete, types_declare`.
Each body is the verbatim `<BODY>` from the matching if-arm with `self`→`bridge`.

- [ ] **Step 2: Add a completeness test** to `tests/test_op_registry.py`:
```python
def test_registry_covers_every_dispatch_op():
    from bn_agent_bridge.op_registry import REGISTRY
    expected = EXPECTED_READ | EXPECTED_WRITE | {"load_binary", "shutdown"}
    assert REGISTRY.names() == expected

def test_decompile_is_the_only_escalating_op():
    from bn_agent_bridge.op_registry import REGISTRY
    escalating = {n for n in REGISTRY.names() if REGISTRY.spec(n).lock_escalation is not None}
    assert escalating == {"decompile"}
```

- [ ] **Step 3: Run.** (Registry now populated at import; dispatch still uses the old if-chain — both coexist this task.)

Run: `uv run pytest tests/test_op_registry.py -v`
Expected: all pass, including the two new tests (51 ops registered).

- [ ] **Step 4: Full suite still green** (nothing wired to the registry yet, so behaviour is identical).

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: 638+ passed.

- [ ] **Step 5: Commit.**

```bash
git add plugin/bn_agent_bridge/bridge.py tests/test_op_registry.py
git commit -m "feat(#33): register all bridge ops as @op binders (dispatch still legacy)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 1.4: Drive dispatch + lock-sets from the registry; delete the if-chain

**Files:** Modify `plugin/bn_agent_bridge/bridge.py` — `dispatch` (`:861`), `_dispatch_on_main` (`:883`–`1101`), and the literal `READ_LOCKED_OPS`/`WRITE_LOCKED_OPS` (`:236`,`:280`).

- [ ] **Step 1: Replace `_dispatch_on_main`'s body** (delete the entire 216-line if-chain, keep the method + signature):
```python
def _dispatch_on_main(self, op, params, target):  # pragma: no cover - GUI runtime
    spec = REGISTRY.spec(op)
    if spec is None:
        raise ValueError(f"Unknown operation: {op}")
    return spec.binder(self, params, target)
```
Add `from .op_registry import REGISTRY` to the imports.

- [ ] **Step 2: Replace the lock selection in `dispatch`** (`:866`–`:876`) with registry-driven selection:
```python
spec = REGISTRY.spec(op)
lock = contextlib.nullcontext()
if spec is not None:
    lock_class = spec.lock
    if spec.lock_escalation is not None and spec.lock_escalation(params):
        lock_class = "write"
    if lock_class == "write":
        lock = self._target_lock.write()
    elif lock_class == "read":
        lock = self._target_lock.read()
```
(The `decompile`/`force_analysis` escalation now comes from `spec.lock_escalation`, preserving today's behaviour at `:869`. Keep the surrounding `try/except` and `_json_response` exactly as-is.)

- [ ] **Step 3: Replace the literal `READ_LOCKED_OPS`/`WRITE_LOCKED_OPS` definitions** (`:236` and `:280`). Delete both literal sets. After the binder block (so all ops are registered), add:
```python
# Derived from the op registry — single source of truth. Kept as module-level
# names because callers/tests reference bridge.READ_LOCKED_OPS / WRITE_LOCKED_OPS.
READ_LOCKED_OPS = frozenset(REGISTRY.read_locked_ops())
WRITE_LOCKED_OPS = frozenset(REGISTRY.write_locked_ops())
```
Preserve the explanatory comments from the originals as a docstring/comment above this block (the `#99` load_binary rationale, the `shutdown` unlocked rationale).

> **Ordering:** the derived assignment must run AFTER the binder block executes. Place the binder block + these two lines together near the end of the module, before `start_bridge`/`start_headless` definitions if those don't depend on the sets (they don't). Verify with the import smoke test in Step 5.

- [ ] **Step 4: Run the registry + golden tests.**

Run: `uv run pytest tests/test_op_registry.py -v`
Expected: all pass — the golden `test_*_membership_unchanged` now validate the *derived* sets equal the frozen literals (the proof the derivation is behaviour-identical).

- [ ] **Step 5: Import smoke + full suite.**

Run:
```bash
PYTHONPATH=plugin uv run python -c "import bn_agent_bridge.bridge as b; print(len(b.READ_LOCKED_OPS), len(b.WRITE_LOCKED_OPS)); b.start_bridge; b.start_headless; b.ui"
uv run pytest -q 2>&1 | tail -3
```
Expected: prints `33 16`; no AttributeError (exports intact); 638+ passed.

- [ ] **Step 6: Firmware equivalence gate** (if BN available).

Run:
```bash
uv run python /tmp/bn33_dogfood/sweep.py \
  "/tmp/DMH-2000NEX/update/CUST_PIO.BIN_extract/12754944-67129344.squashfs_v4_le_extract/bin/aa_accessory_service" \
  /tmp/bn33_dogfood/stage1.json
diff <(jq -S . /tmp/bn33_dogfood/baseline.json) <(jq -S . /tmp/bn33_dogfood/stage1.json) && echo IDENTICAL
```
Expected: `IDENTICAL` (no diff). Any diff is a regression — STOP and debug before committing.

- [ ] **Step 7: Commit Stage 1.**

```bash
git add plugin/bn_agent_bridge/bridge.py
git commit -m "feat(#33): derive dispatch + lock sets from the op registry

Kills the triple-maintained op list (READ_LOCKED_OPS / WRITE_LOCKED_OPS /
the 216-line _dispatch_on_main if-chain). Adding an op is now one @op line.
Behaviour-preserving: lock-set golden tests + full suite + firmware sweep
identical to baseline.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Stages 2–5 — Extractions (characterization-protected mechanical moves)

> **Why these stages are structured differently:** moving ~190 method bodies verbatim into modules cannot be transcribed in full here without reproducing the 6.5k-line file. These are **mechanical relocations**, and the oracle is the characterization suite, not new per-method tests: the existing `test_bridge.py` (238 direct `instance._method(...)` calls) plus the import smoke test plus the firmware diff fully cover behaviour. Each stage therefore follows the same loop, with a worked pattern example. **Do not add behaviour; do not fix bugs; only relocate.**

**The extraction pattern (applies to every module):** a former bridge method `def _foo(self, a, b): <body using self._bar(...) and self.targets>` becomes a free function in its module taking a `ctx` (the `BridgeContext` seam) in place of `self`:
```python
# in e.g. read_decompile.py
def decompile(ctx, target, identifier, *, addresses, force_analysis):
    bv = ctx.resolve_view(target)
    ...  # body, with self._bar(...) -> ctx.bar(...) for SEAM helpers,
         # and self._baz(...) -> module_baz(...) for same-module / imported helpers
```
and the bridge keeps a delegating shim so the 238 test call sites and the binders keep working:
```python
# in bridge.py
def _decompile(self, *a, **k):
    return read_decompile.decompile(self.ctx, *a, **k)
```
`ctx` is constructed once in `BinaryNinjaBridge.__init__` (Task 2.1).

### Task 2.1: Introduce the `BridgeContext` seam + break the one cycle

**Files:** Create `plugin/bn_agent_bridge/seam.py`; Modify `bridge.py`.

- [ ] **Step 1: Create `seam.py` with `BridgeContext`.** Move the resolution/ABI/address-context helpers listed in spec §3.1 (`seam.py` row) out of `BinaryNinjaBridge` into a `BridgeContext` class whose only state is `targets` (passed in). `resolve_view` wraps `self.targets.resolve`; every other helper takes `bv` explicitly. **Also relocate `_find_type` and `_render_type_layout` here** (the cycle-breakers — both state-free). Method bodies move verbatim with `self.`→ internal `self.` (they remain methods of `BridgeContext`).

```python
# seam.py (skeleton — fill with the verbatim relocated bodies)
from __future__ import annotations
class BridgeContext:
    def __init__(self, targets):
        self.targets = targets
    def resolve_view(self, selector):
        return self.targets.resolve(selector)
    def find_function(self, bv, identifier): ...
    # ... all seam helpers from spec §3.1, bodies moved verbatim ...
    def find_type(self, bv, type_name): ...          # relocated from read_types
    def render_type_layout(self, type_obj, ...): ...  # relocated from mutation
```

- [ ] **Step 2: Construct `ctx` in the bridge and make the moved methods delegate.** In `BinaryNinjaBridge.__init__` add `self.ctx = BridgeContext(self.targets)`. Replace each relocated bridge method with a shim, e.g. `def _resolve_view(self, selector): return self.ctx.resolve_view(selector)`, `def _find_type(self, *a, **k): return self.ctx.find_type(*a, **k)`, etc. (Tests call `instance._find_function`, `instance._render_type_layout`, etc. — shims keep them working.)

- [ ] **Step 3: Import smoke + full suite + firmware diff.**

Run:
```bash
PYTHONPATH=plugin uv run python -c "import bn_agent_bridge.bridge"
uv run pytest -q 2>&1 | tail -3
uv run python /tmp/bn33_dogfood/sweep.py "<TARGET>" /tmp/bn33_dogfood/stage2.json
diff <(jq -S . /tmp/bn33_dogfood/baseline.json) <(jq -S . /tmp/bn33_dogfood/stage2.json) && echo IDENTICAL
```
Expected: import OK; 638+ passed; `IDENTICAL`.

- [ ] **Step 4: Commit.**

```bash
git add plugin/bn_agent_bridge/seam.py plugin/bn_agent_bridge/bridge.py
git commit -m "refactor(#33): extract BridgeContext seam; relocate find_type + render_type_layout to break read_types<->mutation cycle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3.1: Extract `il_format.py` and `vars.py` (state-free leaves)

**Files:** Create `il_format.py`, `vars.py`; Modify `bridge.py`.

- [ ] **Step 1:** Move the methods listed in spec §3.1 for `il_format.py` and `vars.py` into those modules as free functions (they take `func`/`bv`/`insn`/`var` — no `ctx` needed except where they call a seam helper; pass `ctx` only to the few that do). Replace each in `bridge.py` with a delegating shim. `il_format`'s `_ssa_var_entry` imports `vars.variable_identifier`.
- [ ] **Step 2:** Import smoke + full suite + firmware diff (same commands as Task 2.1 Step 3, output `stage3.json`). Expected: import OK; 638+ passed; `IDENTICAL`.
- [ ] **Step 3:** Commit: `refactor(#33): extract il_format.py + vars.py (state-free rendering/variable helpers)`.

### Task 4.1: Extract `mutation_engine.py` (issue step 2)

**Files:** Create `mutation_engine.py`; Modify `bridge.py`.

- [ ] **Step 1:** Move the ~50 mutation-cluster methods (spec §3.1 `mutation_engine.py` row) into `mutation_engine.py` as free functions taking `ctx`. The single public entry is `run(ctx, selector, preview, operations)` (former `_mutation`). Outbound calls resolve via `ctx` (`resolve_view`, `find_function`, `functions_containing`, `find_type`, `render_type_layout`), `il_format.function_text`, `vars.*`, and module-free `_parse_address`/`_normalize_prototype`. `_revert_undo_safely` lives here; `create_comments` imports it (or it goes to the seam — choose to avoid a `create↔mutation` cycle; seam is cleaner).
- [ ] **Step 2:** Keep bridge shims for every mutation method tests reference: `_mutation`, `_apply_operation`, `_verify_operation`, `_op_struct_field_set`, `_op_struct_field_rename`, `_op_set_prototype`, `_op_types_declare`, `_affected_type_names`, `_restore_local_var_drift`, `_run_local_restores`, `_render_type_layout` (already in seam), etc. — i.e. each `instance._*` mutation call in `test_bridge.py`.
- [ ] **Step 3:** Import smoke + full suite + firmware diff (`stage4.json`). The firmware sweep MUST include previewed mutations (rename/comment/proto/local/struct-field `--preview`) so this stage is exercised. Expected: import OK; 638+ passed; `IDENTICAL`.
- [ ] **Step 4:** Commit: `refactor(#33): extract mutation_engine.py (apply/verify/op/snapshot/restore)`.

### Task 5.1: Extract the read-op domain modules (issue step 3)

**Files:** Create `read_decompile.py`, `read_listing.py`, `read_xrefs.py`, `read_evidence.py`, `read_taint_slice.py`, `read_types.py`, `read_misc.py`, `create_comments.py`; Modify `bridge.py`.

Do these **one module per commit** (8 commits) so each is independently green and reviewable. For each module:

- [ ] **Step 1:** Move that module's methods (spec §3.1) into it as free functions taking `ctx`; rewrite `self._seamhelper`→`ctx.helper`, `self._samemodulehelper`→ local call, `self._othermodulefn`→ `import othermodule`. Keep one-way import direction per spec §3.2 (e.g. `read_evidence` imports `read_xrefs`, never the reverse).
- [ ] **Step 2:** Replace the corresponding bridge methods with delegating shims (keep every name `test_bridge.py` calls; the Stage-1 binders call these shims unchanged). When a module owns an op handler, its `@op` binder may move into the module too (mirrors `commands/*.py`), with `bridge.py` importing the module so registration fires — but moving binders is optional polish; leaving them in `bridge.py` calling shims is equally correct. Prefer leaving them to minimize churn.
- [ ] **Step 3:** After each module: import smoke + full suite + firmware diff (`stage5_<module>.json`). Expected: import OK; 638+ passed; `IDENTICAL`.
- [ ] **Step 4:** Commit per module: `refactor(#33): extract <module>.py`.

---

## Stage 6 — Finalize

### Task 6.1: Whole-refactor verification + PR

- [ ] **Step 1: Confirm `bridge.py` shrank and responsibilities moved.**

Run: `wc -l plugin/bn_agent_bridge/*.py | sort -n`
Expected: `bridge.py` is now a few hundred lines of lifecycle + dispatch + shims; the read-op library and mutation engine live in their modules.

- [ ] **Step 2: Full green + import smoke + final firmware diff.**

Run:
```bash
uv run pytest -q 2>&1 | tail -3
PYTHONPATH=plugin uv run python -c "import bn_agent_bridge.bridge as b; [getattr(b,n) for n in ('start_bridge','start_headless','ui','READ_LOCKED_OPS','WRITE_LOCKED_OPS','BinaryNinjaBridge','_dispatch_on_main')]; print('exports OK')"
uv run python /tmp/bn33_dogfood/sweep.py "<TARGET>" /tmp/bn33_dogfood/final.json
diff <(jq -S . /tmp/bn33_dogfood/baseline.json) <(jq -S . /tmp/bn33_dogfood/final.json) && echo IDENTICAL
```
Expected: 638+ passed; `exports OK`; `IDENTICAL`.

- [ ] **Step 3: Headless start sanity** (the GUI/headless entry the suite mocks).

Run: `bn session start "<TARGET>" --format json` then `bn session list --format json` then `bn session stop <id>` — confirm a real bridge boots, serves an op, and stops cleanly against the refactored code.

- [ ] **Step 4: Push branch + open one PR into `m4ul3r`.**

```bash
git push -u origin feat/33-modularize-bridge
gh pr create --base m4ul3r --title "Modularize bridge.py: declarative op registry + domain-module split (#33)" --body "<see template below>"
```
PR body covers: the registry (kills the triple-maintained list), the seam + cycle-break, the 5 staged commits, and the verification evidence (lock-set golden test, full suite green per stage, firmware JSON sweep identical to baseline). **Scrub the head-unit name** — refer to the firmware generically (e.g. "a stripped 32-bit ARM firmware executable"), per the no-leak-targets rule. Closes #33.

---

## Self-Review (completed)

**Spec coverage:** registry → Tasks 1.1–1.4; derived lock sets → Task 1.4 Step 3; `decompile` escalation + `load_binary`/`shutdown` no-lock → Task 1.3 examples + 1.4 Step 2; seam + cycle-break → Task 2.1; il_format/vars → Task 3.1; mutation_engine → Task 4.1; read-op modules → Task 5.1; facade shims preserving 238 test calls → extraction pattern + each stage's shim step; firmware dogfood → Task 0.2 + every stage's diff gate; leak hygiene → Task 0.2 (local only) + 6.1 Step 4. All spec sections mapped.

**Placeholder scan:** Stage 1 + the harness are fully coded. Stages 2–5 deliberately use the documented mechanical-move pattern with a worked example rather than 190 inlined method bodies (reproducing them is infeasible and the characterization suite is the oracle); the `<TARGET>` placeholder is the firmware path defined in Task 0.2 Step 1 — substitute it.

**Type consistency:** `OpRegistry.op/spec/names/read_locked_ops/write_locked_ops`, `OpSpec(name, lock, binder, lock_escalation)`, `REGISTRY`, `BridgeContext.resolve_view/find_function/find_type/render_type_layout`, binder signature `(bridge, params, target)`, shim form `def _x(self, *a, **k): return module.fn(self.ctx, *a, **k)` — consistent across all tasks.
