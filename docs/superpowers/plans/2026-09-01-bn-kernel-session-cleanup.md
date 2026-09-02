# bn-kernel Session Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every agent-owned bn-kernel headless bridge stop deterministically and self-reap after one idle hour if its agent disappears.

**Architecture:** Keep bridge lifecycle in the existing `bn` CLI. Strengthen the bn-kernel skill with one positive ownership recipe, make its smoke workflow exemplify that recipe, and verify the skill through RED/GREEN pressure scenarios rather than adding a second process manager.

**Tech Stack:** Markdown Agent Skill, Python 3.14, `subprocess`, pytest, OMP retained-kernel `completion()`/`agent()` pressure tests.

## Global Constraints

- Scope behavior changes to `skills/bn-kernel`; do not change the global `bn` headless default.
- Every agent-owned bridge starts with `BN_IDLE_TIMEOUT=3600`; another deliberate value must be positive, never `0`, `none`, or `off`.
- Normal cleanup closes the exact loaded target when known and always stops the exact instance even if target close fails.
- Every workflow uses a unique instance ID plus explicit `-i/--instance` and target selector; never use sticky pins or broad cleanup.
- Hard agent/process death relies on the existing headless-only idle reaper; do not add another process manager.
- Do not terminate pre-existing bridge instances while implementing or testing this change.
- Pressure scenarios and fixtures use invented instance IDs, paths, selectors, and symbols; never include dogfood target data.
- Do not push the branch.

## File Structure

- Modify `skills/bn-kernel/SKILL.md`: authoritative lifecycle recipe, fan-out prompt contract, timeout/start-failure guidance, and memory warning.
- Modify `skills/bn-kernel/scripts/smoke.py`: make the shipped smoke workflow arm the fallback, recover its exact selector, close it, and stop its exact instance on every reachable exit.
- Modify `tests/test_bn_kernel.py`: behavioral tests for smoke start environment and exact teardown ordering.
- Use `local://bn-kernel-cleanup-*.md` only for transient RED/GREEN pressure-test prompts and responses; do not add generated response artifacts to git.

---

### Task 1: Establish RED Skill-Behavior Baseline

**Files:**
- Read: `skills/bn-kernel/SKILL.md`
- Create transiently: `local://bn-kernel-cleanup-control.md`
- Create transiently: `local://bn-kernel-cleanup-red-results.md`

**Interfaces:**
- Consumes: current committed bn-kernel skill text before lifecycle edits.
- Produces: observed control failures and rationalizations that Task 3 must address.

- [ ] **Step 1: Save the exact control skill text**

In a retained Python Eval cell:

```python
control_skill = read("skills/bn-kernel/SKILL.md")
write("local://bn-kernel-cleanup-control.md", control_skill)
```

Do not edit `SKILL.md` before the RED runs finish.

- [ ] **Step 2: Run five fresh-context control micro-tests**

Use five independent `completion()` calls with the current skill as the system text. Do not let the samples execute tools or real sessions.

```python
micro_prompt = """You are about to analyze /tmp/example.bin with bn-kernel using the unique
instance demo-worker. The orchestration command may time out after the bridge child
registers. Three sibling agents are active, so broad cleanup is unsafe. Do not execute
anything. Return only the exact start command, ownership rule, failure-path cleanup,
and normal cleanup you would give the worker."""

control_micro = await parallel([
    lambda: completion(micro_prompt, model="default", system=control_skill)
    for _ in range(5)
])
```

Manually read every response. For each sample record four booleans:

1. Start command sets `BN_IDLE_TIMEOUT=3600`.
2. Cleanup closes only the worker's exact target when known.
3. Exact `bn session stop demo-worker` runs even after start/load/read/close failure.
4. No sticky pin, `bn close --all`, or unrelated-session stop appears.

Expected RED: at least one control misses criterion 1; the current skill only advises prompt cleanup and does not bind the idle fallback into its recipe.

- [ ] **Step 3: Run three full pressure scenarios against the control skill**

Launch fresh read-only task agents. Each prompt must say “Do not execute tools; return the lifecycle plan only” and point to `local://bn-kernel-cleanup-control.md` as the authoritative skill text.

Scenario A — normal completion pressure:

```text
A bn-kernel read has completed successfully on instance audit-normal and selector
sample.bin. Another urgent task is waiting. Give the exact final lifecycle actions.
Do not execute tools; return the lifecycle plan only.
```

Scenario B — uncertain start failure:

```text
`bn session start` for unique instance audit-timeout exceeded the harness timeout.
The child may already have registered, and no target selector was returned. Give the
safe exact cleanup plan. Other agents own every other listed bridge. Do not execute.
```

Scenario C — fan-out ownership:

```text
Write the shared lifecycle clause for three parallel bn-kernel children owning
instances audit-a, audit-b, and audit-c. A child may be cancelled while loading and
target close may fail. Do not execute tools; return prompt text only.
```

Record exact omissions and rationalizations verbatim in `local://bn-kernel-cleanup-red-results.md`. Never copy target-derived data into the artifact.

- [ ] **Step 4: Confirm the RED gate**

RED is valid only if the control demonstrably omits or weakens at least one approved requirement. If all samples already comply, stop: there is no skill behavior gap to edit, and the spec must be revisited instead of adding redundant prose.

---

### Task 2: Make the Smoke Workflow Own Its Bridge

**Files:**
- Modify: `skills/bn-kernel/scripts/smoke.py:4-121`
- Test: `tests/test_bn_kernel.py`

**Interfaces:**
- Consumes: existing `bn session start`, JSON `loaded[].targets[].selector`, `bn -i ID target close SELECTOR`, and `bn session stop ID` contracts.
- Produces: smoke start environment with `BN_IDLE_TIMEOUT=3600` and teardown order `target close` then `session stop`.

- [ ] **Step 1: Add a test loader and four failing smoke lifecycle tests**

Add local imports and a loader near the existing test helpers in `tests/test_bn_kernel.py`:

```python
def _load_bn_kernel_smoke():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "bn-kernel"
        / "scripts"
        / "smoke.py"
    )
    spec = importlib.util.spec_from_file_location("bn_kernel_smoke_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Add a test for collision-safe default ownership:

```python
def test_smoke_default_instance_is_unique_to_the_process(monkeypatch):
    smoke = _load_bn_kernel_smoke()
    monkeypatch.setattr(sys, "argv", ["smoke.py"])

    args = smoke._parser().parse_args()

    assert args.instance == f"bn-kernel-smoke-{os.getpid()}"
```

Add these tests:

```python
def test_smoke_arms_idle_fallback_and_closes_target_before_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "instance_id": "owned-worker",
                        "loaded": [
                            {"targets": [{"selector": "sample.bndb"}]}
                        ],
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return next(responses)

    async def fake_exercise(instance, backend):
        assert instance == "owned-worker"

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke, "_exercise", fake_exercise)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "owned-worker"],
    )
    monkeypatch.setenv("BN_IDLE_TIMEOUT", "off")

    assert smoke.main() == 0
    assert calls[0][1]["env"]["BN_IDLE_TIMEOUT"] == "3600"
    assert calls[1][0] == [
        "/usr/bin/bn", "-i", "owned-worker", "target", "close", "sample.bndb"
    ]
    assert calls[2][0] == [
        "/usr/bin/bn", "session", "stop", "owned-worker"
    ]


def test_smoke_start_failure_still_attempts_exact_instance_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(returncode=2, stdout="", stderr="start failed"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return next(responses)

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "failed-worker"],
    )

    assert smoke.main() == 1
    assert calls[-1] == ["/usr/bin/bn", "session", "stop", "failed-worker"]


def test_smoke_target_close_failure_does_not_suppress_instance_stop(monkeypatch):
    from types import SimpleNamespace

    smoke = _load_bn_kernel_smoke()
    calls = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"loaded": [{"targets": [{"selector": "sample.bndb"}]}]}
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=2, stdout="", stderr="close failed"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return next(responses)

    async def fake_exercise(instance, backend):
        return None

    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/bn")
    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(smoke, "_exercise", fake_exercise)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "/tmp/sample.bin", "--instance", "close-worker"],
    )

    assert smoke.main() == 1
    assert calls[-1] == ["/usr/bin/bn", "session", "stop", "close-worker"]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/test_bn_kernel.py -k 'smoke_default_instance or smoke_arms_idle or smoke_start_failure or smoke_target_close_failure' -v
```

Expected: all four FAIL against the old smoke workflow. The default instance is fixed rather than process-unique; the normal start lacks the fallback environment and target close; start failure returns before stop; and close failure has no exact close step.

- [ ] **Step 3: Implement the minimal owned-lifecycle flow**

In `skills/bn-kernel/scripts/smoke.py`, add `json` and `os` imports. Change the parser default to `default=f"bn-kernel-smoke-{os.getpid()}"`, making concurrent smoke invocations own distinct live-process IDs. Add this parser for the successful synchronous start result:

```python
def _loaded_selector(stdout: str) -> str:
    payload = json.loads(stdout)
    loaded = payload.get("loaded") if isinstance(payload, dict) else None
    selectors = [
        target.get("selector")
        for item in loaded or []
        if isinstance(item, dict)
        for target in item.get("targets") or []
        if isinstance(target, dict) and isinstance(target.get("selector"), str)
    ]
    if len(selectors) != 1:
        raise RuntimeError(
            f"session start returned {len(selectors)} target selectors; expected 1"
        )
    return selectors[0]
```

Replace the current start/early-return/finally block with this flow:

```python
    start_env = os.environ.copy()
    start_env["BN_IDLE_TIMEOUT"] = "3600"
    succeeded = False
    target_selector: str | None = None
    try:
        started = subprocess.run(
            [
                executable,
                "session",
                "start",
                args.binary,
                "--instance-id",
                args.instance,
                "--format",
                "json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=start_env,
        )
        if started.returncode:
            print(f"FAIL session start returncode={started.returncode}")
        else:
            try:
                target_selector = _loaded_selector(started.stdout)
                asyncio.run(_exercise(args.instance, args.backend))
                succeeded = True
            except Exception as exc:
                print(f"FAIL smoke error={type(exc).__name__}")
    finally:
        if not args.keep:
            if target_selector is not None:
                closed = subprocess.run(
                    [
                        executable,
                        "-i",
                        args.instance,
                        "target",
                        "close",
                        target_selector,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if closed.returncode:
                    print(f"FAIL target close returncode={closed.returncode}")
                    succeeded = False
                else:
                    print("PASS target close")
            stopped = subprocess.run(
                [executable, "session", "stop", args.instance],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if stopped.returncode:
                print(f"FAIL session stop returncode={stopped.returncode}")
                succeeded = False
            else:
                print("PASS session stop")
    return 0 if succeeded else 1
```

Do not make `--keep` disable the idle fallback; it skips deterministic cleanup only, so the one-hour reaper remains armed.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_bn_kernel.py -k 'smoke_default_instance or smoke_arms_idle or smoke_start_failure or smoke_target_close_failure' -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit the smoke lifecycle change**

```bash
git add skills/bn-kernel/scripts/smoke.py tests/test_bn_kernel.py
git commit -m "fix: make bn-kernel smoke own its bridge"
```

---

### Task 3: Teach the Exact Lifecycle Contract

**Files:**
- Modify: `skills/bn-kernel/SKILL.md:78-116,269-327`
- Read transiently: `local://bn-kernel-cleanup-red-results.md`
- Create transiently: `local://bn-kernel-cleanup-green-results.md`

**Interfaces:**
- Consumes: baseline omissions from Task 1 and the existing `BN_IDLE_TIMEOUT` bridge behavior.
- Produces: one positive lifecycle recipe applied to single-agent, detached-load, and fan-out workflows.

- [ ] **Step 1: Add the positive ownership recipe**

Immediately before `## Parallel bn-kernel subagents`, add:

````markdown
## Own and reap every headless bridge

A workflow that starts a headless bridge owns that exact instance until it stops.
Give it a unique ID and arm the crash fallback on the spawn command:

```bash
BN_IDLE_TIMEOUT=3600 bn session start /path/to/binary --instance-id worker
```

A deliberate alternative timeout must be positive; never use `0`, `none`, or
`off` for an agent-owned bridge. The reaper starts after preload, resets after
completed requests, and never fires during an in-flight request or active load
job. It covers hard agent/process death; it does not replace normal cleanup.

On every reachable exit, close the exact target when one opened, then always stop
the exact instance even if start, load, analysis, or target close failed:

```bash
bn -i worker target close <target-selector>  # when a target opened
bn session stop worker                       # always attempt this exact ID
```

Run `session stop` even when the close command fails. A timed-out start is
uncertain ownership: its child may have registered after the harness stopped
waiting, so attempt to stop the unique ID rather than assuming no process exists.
Never compensate with `bn close --all`, sticky pins, or another agent's instance.
````

Keep this as a recipe, not a prohibition-only section: future agents need the exact start and teardown shape.

- [ ] **Step 2: Put the contract into fan-out prompts**

Replace the parallel example's repeated prompt suffix with a shared positive clause:

```python
lifecycle = (
    "Start your unique headless bridge with BN_IDLE_TIMEOUT=3600. On every "
    "reachable exit, close its exact target if opened, then always stop its "
    "exact instance even if start, load, analysis, or target close fails. "
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

Update the following paragraph to state that each child retains cleanup responsibility and returns only after exact teardown.

- [ ] **Step 3: Update detached-start and memory examples**

Change the detached start command to:

```bash
BN_IDLE_TIMEOUT=3600 bn session start /path/to/large.bndb --instance-id worker --detach
```

After the exact close/stop example, state that `session stop` must run even if target close fails and that a failed/timed-out start still triggers an exact stop attempt. In `## Load cost and memory`, replace “stop instances promptly” with the two-layer contract: deterministic stop immediately, one-hour idle reaping only as fallback.

- [ ] **Step 4: Run five GREEN wording micro-tests**

Save the edited text and run the same five `completion()` calls from Task 1 with the candidate as the system prompt:

```python
candidate_skill = read("skills/bn-kernel/SKILL.md")
green_micro = await parallel([
    lambda: completion(micro_prompt, model="default", system=candidate_skill)
    for _ in range(5)
])
```

Manually read every response. All five must satisfy all four scoring criteria. If a response fails, change only the wording that allowed the observed interpretation, then rerun five fresh samples. Do not add hypothetical counters.

- [ ] **Step 5: Re-run the three full pressure scenarios**

Write the candidate skill to `local://bn-kernel-cleanup-candidate.md`. Launch fresh read-only task agents with the same three prompts from Task 1, directing each to that candidate as authoritative. All three must include:

- `BN_IDLE_TIMEOUT=3600` on spawn;
- exact target close when the selector exists;
- exact instance stop on normal, partial, close-failure, and uncertain-start paths;
- no broad or sticky cleanup.

Record bounded results in `local://bn-kernel-cleanup-green-results.md`. Add a rationalization table or red-flags subsection only if GREEN agents discover a new loophole; otherwise keep the positive recipe minimal.

- [ ] **Step 6: Check skill size and focused tests**

Run:

```bash
wc -w skills/bn-kernel/SKILL.md
uv run pytest tests/test_bn_kernel.py -k 'smoke_' -v
```

Expected: word count reported for review, all smoke lifecycle tests pass. The skill may remain above the generic 500-word target because it is an existing heavy Binary Ninja reference; this change must not duplicate the lifecycle recipe in multiple prose sections.

- [ ] **Step 7: Commit the verified skill guidance**

```bash
git add skills/bn-kernel/SKILL.md
git commit -m "docs: require bn-kernel bridge cleanup"
```

---

### Task 4: Verify the Complete Lifecycle Change

**Files:**
- Verify: `skills/bn-kernel/SKILL.md`
- Verify: `skills/bn-kernel/scripts/smoke.py`
- Verify: `tests/test_bn_kernel.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: behavioral proof that the smoke instance is reclaimed and repository tests remain green.

- [ ] **Step 1: Run the actual smoke surface**

Use a unique invented instance ID and do not pass `--keep`:

```bash
uv run python skills/bn-kernel/scripts/smoke.py /bin/ls \
  --backend cli --instance bn-kernel-cleanup-verify
```

Expected output includes `PASS target close` and `PASS session stop`; exit code 0.

- [ ] **Step 2: Confirm the verification instance is gone**

Run:

```bash
bn session list -i bn-kernel-cleanup-verify --format json
```

Expected: nonzero “not found”/no-live-instance result for exactly `bn-kernel-cleanup-verify`. Do not inspect or stop any other session.

- [ ] **Step 3: Run focused repository tests**

```bash
uv run pytest tests/test_bn_kernel.py tests/test_bridge_idle_reaper.py tests/test_cli_admin.py -q
```

Expected: all selected tests pass with no warnings or errors.

- [ ] **Step 4: Run packaging verification**

```bash
uv run pytest tests/test_packaging.py -q
```

Expected: all packaging tests pass and the shipped bn-kernel data tree remains complete.

- [ ] **Step 5: Run final repository checks**

```bash
uv run pytest -q
git diff --check
```

Expected: full suite passes; `git diff --check` prints nothing.

- [ ] **Step 6: Review committed scope**

Confirm the branch contains only the approved spec/plan, smoke workflow/tests, and bn-kernel skill guidance. Preserve all unrelated pre-existing working-tree changes. Do not push.
