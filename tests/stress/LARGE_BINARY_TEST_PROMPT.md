# bn Large Binary Stress Test

You are testing the `bn` CLI tool's behavior with large system binaries that exceed the default 30-second load timeout. The code is at `/opt/bn`.

## Background

`bn load` has a 30s default socket timeout. For binaries above ~500KB, `binaryninja.load()` + analysis takes longer than 30s. The CLI reports a timeout error, but the bridge continues loading in the background. The binary should eventually appear in `bn target list`.

**Known timing baselines** (your results may vary):
- `/usr/bin/bash` (1.4MB): ~60s total
- `/usr/lib/x86_64-linux-gnu/libc.so.6` (2.1MB): ~150s total

## Step 1: Build fixtures (if not already done)

```bash
cd /opt/bn/tests/fixtures && make
```

## Step 2: Background load recovery

Test that a timed-out load still completes in the background.

1. Start a session (no binary):
```bash
bn session start
```
2. Attempt to load bash (will timeout at ~30s):
```bash
bn --instance <id> load /usr/bin/bash --format json
```
3. Expect a timeout error. Now poll `target list` every 10s until the target appears:
```bash
bn --instance <id> target list --format json
```
4. Once the target appears, run some read commands to verify it's usable:
```bash
bn --instance <id> function list --limit 10
bn --instance <id> function search main
bn --instance <id> strings --limit 5
```
5. Stop the session.

**Pass criteria**: Binary eventually appears (within ~90s of the initial load), read commands work after it does.

## Step 3: Two large binaries in parallel sessions

Test concurrent large loads in separate sessions.

1. Start two sessions (no binaries):
```bash
bn session start   # → session A
bn session start   # → session B
```
2. Load a different binary into each — fire both loads without waiting:
```bash
bn --instance <A> load /usr/bin/bash --format json &
bn --instance <B> load /usr/lib/x86_64-linux-gnu/libc.so.6 --format json &
wait
```
Both will timeout, that's expected.
3. Poll both sessions every 15s. Report when each target appears:
```bash
bn --instance <A> target list --format json
bn --instance <B> target list --format json
```
4. Once both are loaded, decompile a function from each:
```bash
bn --instance <A> function search main
bn --instance <A> decompile <some_function>
bn --instance <B> function search printf
bn --instance <B> decompile <some_function>
```
5. Stop both sessions.

**Pass criteria**: Both binaries eventually load. Decompile works on each. Sessions don't interfere with each other.

## Step 4: Session start with large binary preload

Test `bn session start <binary>` with a binary that exceeds the timeout.

```bash
bn session start /usr/bin/bash
```

This should return JSON with the instance_id (the session started) and a load error in the `loaded` array (timeout). Verify:
1. The session is running: `bn session list`
2. Poll `bn --instance <id> target list` until bash appears
3. Run a command against it: `bn --instance <id> function list --limit 5`
4. Stop the session.

**Pass criteria**: Session starts successfully even though the preloaded binary times out. Binary eventually appears.

## Step 5: Rapid operations during background load

Test that the bridge remains responsive while a large binary is still loading.

1. Start a session, load bash (will timeout):
```bash
bn session start
bn --instance <id> load /usr/bin/bash --format json
```
2. Immediately after the timeout, try `doctor` and `target list` (these should NOT hang):
```bash
time bn --instance <id> doctor --format json
time bn --instance <id> target list --format json
```
3. Load a small fixture binary while bash is still loading:
```bash
bn --instance <id> load /opt/bn/tests/fixtures/add_x86_64 --format json
```
4. Check targets — the small binary should appear immediately even if bash is still loading:
```bash
bn --instance <id> target list --format json
```
5. If add_x86_64 appeared, try decompiling from it while bash loads in the background:
```bash
bn --instance <id> decompile --target add_x86_64 _start
```
6. Wait for bash to finish, then verify both are present.

**Pass criteria**: `doctor` and `target list` respond quickly (<2s) during background load. Small binary can be loaded and used while large one is still analyzing.

## Step 6: Memory pressure — 3 concurrent large loads

Start 3 sessions, each loading a large binary simultaneously:

```bash
bn session start  # → A
bn session start  # → B
bn session start  # → C

bn --instance <A> load /usr/bin/bash --format json &
bn --instance <B> load /usr/lib/x86_64-linux-gnu/libc.so.6 --format json &
bn --instance <C> load /usr/bin/vim.basic --format json &
wait
```

Poll all three. Report:
- How long each takes to appear
- Whether any fail completely
- Memory usage: `ps aux | grep bn-agent | grep -v grep`

Stop all three sessions when done.

**Pass criteria**: All three eventually load (even if it takes several minutes). No crashes.

## Step 7: Report

| Test | Result | Load time | Notes |
|------|--------|-----------|-------|
| Background load recovery (bash) | PASS/FAIL | Xs | |
| Two parallel large loads | PASS/FAIL | A: Xs, B: Xs | |
| Session start with preload | PASS/FAIL | Xs | |
| Responsiveness during load | PASS/FAIL | | doctor/target list response time |
| Small binary during large load | PASS/FAIL | | |
| 3 concurrent large loads | PASS/FAIL | A: Xs, B: Xs, C: Xs | memory notes |

List any failures, hangs, or unexpected behavior.
