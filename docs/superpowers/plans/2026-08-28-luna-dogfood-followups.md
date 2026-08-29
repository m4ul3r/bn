# Luna Dogfood Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make collection-envelope metadata discoverable and validate/document 16-way detached bridge startup with realistic orchestration budgets.

**Architecture:** Add two read-only projections on the existing immutable `bn_kernel.Result`; the backend payload and validation paths remain unchanged. Keep lifecycle locking unchanged, measure the real 16-target workload with explicit spawn/tool budgets, and document the observed orchestration requirement.

**Tech Stack:** Python 3.14, pytest, asyncio subprocess orchestration, Binary Ninja bridge CLI, Markdown skill documentation.

## Global Constraints

- Preserve the safety-critical spawn lock, registry ownership checks, and process-identity protocol.
- `Result.payload` remains the complete backend envelope.
- Malformed convenience-property values return `None`; successful collection helpers retain their existing strict payload validation.
- The live measurement uses only unique `luna-followup-01` through `luna-followup-16` instances and must clean all of them up.
- Generated timing output remains a session artifact, not a repository file.
- Commit without pushing.

---

### Task 1: Result collection metadata properties

**Files:**
- Modify: `skills/bn-kernel/src/bn_kernel/__init__.py:749-784`
- Test: `tests/test_bn_kernel.py:295-305`
- Test: `tests/test_bn_kernel.py:506-548`

**Interfaces:**
- Consumes: `Result.payload: Any`
- Produces: `Result.returned: int | None` and `Result.has_more: bool | None`

- [ ] **Step 1: Add failing property contract tests**

Extend `test_result_is_immutable_and_uses_tuples` with a valid collection payload and assertions:

```python
result = bn_kernel.Result(
    value=[],
    payload={"total": 4, "returned": 0, "has_more": True},
    notes=("note",),
    argv=("strings",),
    backend="cli",
)
assert result.returned == 0
assert result.has_more is True
```

Add parameterized malformed-shape coverage:

```python
@pytest.mark.parametrize(
    ("payload", "returned", "has_more"),
    [
        ("text", None, None),
        ({}, None, None),
        ({"returned": -1, "has_more": 1}, None, None),
        ({"returned": True, "has_more": "yes"}, None, None),
        ({"returned": 3, "has_more": False}, 3, False),
    ],
)
def test_result_collection_metadata_properties_are_typed(payload, returned, has_more):
    result = bn_kernel.Result(
        value=[], payload=payload, notes=(), argv=("strings",), backend="cli"
    )
    assert result.returned == returned
    assert result.has_more is has_more
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
uv run pytest -q tests/test_bn_kernel.py -k 'result_is_immutable or result_collection_metadata'
```

Expected: failures because `Result` has no `returned` or `has_more` properties.

- [ ] **Step 3: Implement the minimal immutable properties**

Insert after `Result.total`:

```python
@property
def returned(self) -> int | None:
    if isinstance(self.payload, Mapping):
        returned = self.payload.get("returned")
        if (
            isinstance(returned, int)
            and not isinstance(returned, bool)
            and returned >= 0
        ):
            return returned
    return None

@property
def has_more(self) -> bool | None:
    if isinstance(self.payload, Mapping):
        has_more = self.payload.get("has_more")
        if isinstance(has_more, bool):
            return has_more
    return None
```

- [ ] **Step 4: Pin the real zero-limit result surface**

In the existing CLI `limit=0` test, retain the payload assertions and add:

```python
assert session.last.returned == 0
assert session.last.has_more is True
```

This verifies the convenience properties project the actual synthesized probe envelope and that no row leaks.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest -q tests/test_bn_kernel.py -k 'result or zero_limit'
```

Expected: all selected tests pass.

---

### Task 2: High-fanout lifecycle measurement and skill guidance

**Files:**
- Modify: `skills/bn-kernel/SKILL.md` in the parallel-agent and load-cost sections
- Test: `tests/test_skill_reference_drift.py`

**Interfaces:**
- Consumes: `bn session start PATH --instance-id ID --detach --format json`, `session status`, `target list`, `target close`, and `session stop`
- Produces: measured start/queue latency for 16 instances and committed high-fanout timeout guidance

- [ ] **Step 1: Confirm the measurement starts clean**

Run:

```bash
uv run bn session list
```

Expected: `no sessions`.

- [ ] **Step 2: Launch the same 16 targets concurrently with explicit budgets**

Use one retained Python Eval cell with `asyncio.create_subprocess_exec`. For each target, launch:

```text
env BN_SPAWN_TIMEOUT=180 uv run bn session start <path> --instance-id luna-followup-NN --detach --format json
```

Wrap each `communicate()` in `asyncio.wait_for(..., timeout=210)`. Record monotonic elapsed seconds, return code, stdout, and stderr. Launch all 16 subprocesses before awaiting any individual result.

Use these paths in order:

```python
[
    "/home/m4ul3r/mg_testbins/targets/cwe-121/pnm2png.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-122/gif2rgb-5.2.2.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-125/file-5.42.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-190/minizip-1.3.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-193/nginx-1.20.0.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-22/lighttpd-1.4.45.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-252/ftp-2.4.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-415/patch-2.7.6.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-416/tinyproxy-1.11.1.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-444/haproxy-2.6.6.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-476/opusinfo-0.12.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-59/nano-7.2.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-78/jhead-3.06.0.1.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-787/lz4-1.9.3.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-806/cwe806-loop.bndb",
    "/home/m4ul3r/mg_testbins/targets/cwe-908/7za-16.02.bndb",
]
```

- [ ] **Step 3: Poll each exact load job to terminal state**

Parse each successful start response for `instance_id` and its `loaded[0].job_id`. Poll:

```text
uv run bn -i <instance> session status <job-id> --format json
```

until top-level `terminal` is true. Record `succeeded`, final state, and elapsed load time. Cap each polling operation at 15 minutes and report exact stderr for failures.

- [ ] **Step 4: Clean every measurement instance in a finally path**

For each registered instance, obtain the exact selector from `target list`, close only that selector, then stop only that instance. Finally run:

```bash
uv run bn session list
```

Expected: `no sessions`.

Write bounded JSON timing evidence to `local://luna-followup-lifecycle-measurement.json`; do not add it to git.

- [ ] **Step 5: Update high-fanout guidance**

Add guidance to `SKILL.md` stating:

```markdown
For high-fanout cold starts, the orchestration tool's command timeout must exceed
`BN_SPAWN_TIMEOUT`; otherwise the harness can kill `bn session start` while its
new-session child continues registering. On a heavily loaded host, set a larger
spawn budget (for example `BN_SPAWN_TIMEOUT=180`) and give the surrounding tool a
strictly larger timeout. This changes the registration budget, not the detached
load-job budget; continue polling the exact job separately.
```

Also update the `limit=0` section to show:

```python
s.last.returned   # 0
s.last.has_more   # True when the probe found a row
```

and state that `s.last.payload` remains the complete envelope.

- [ ] **Step 6: Run documentation tests**

Run:

```bash
uv run pytest -q tests/test_skill_reference_drift.py tests/test_bn_kernel.py
```

Expected: all tests pass.

---

### Task 3: Integrated verification and implementation commit

**Files:**
- Verify all modified and untracked files in the existing reviewed worktree
- Commit the implementation and existing review remediation; do not push

**Interfaces:**
- Consumes: completed Tasks 1 and 2
- Produces: one verified implementation commit on `feat/omp-kernel-integration`

- [ ] **Step 1: Run lifecycle-focused tests**

Run:

```bash
uv run pytest -q tests/test_transport.py tests/test_cli_admin.py tests/test_bridge_lifecycle.py
```

Expected: all selected tests pass; only environment-declared skips are allowed.

- [ ] **Step 2: Compile the changed Python source**

Run:

```bash
uv run python -m compileall -q src skills/bn-kernel/src
```

Expected: exit 0 with no output.

- [ ] **Step 3: Run the full suite**

Run:

```bash
uv run pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Check diff integrity and state**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the previously reviewed remediation files, follow-up source/tests/docs, and plan file are changed or untracked.

- [ ] **Step 5: Commit without pushing**

Stage all reviewed implementation changes and the implementation plan, then commit:

```bash
git add -A
git add -f docs/superpowers/plans/2026-08-28-luna-dogfood-followups.md
git commit -m "fix: close Luna bn-kernel dogfood follow-ups"
```

Expected: commit succeeds on `feat/omp-kernel-integration`; working tree is clean afterward.