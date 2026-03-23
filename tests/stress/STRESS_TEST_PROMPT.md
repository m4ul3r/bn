# bn Multi-Instance Stress Test

You are testing the `bn` CLI tool's multi-instance session support. The code is at `/opt/bn`.

## Step 1: Build fixtures

```bash
cd /opt/bn/tests/fixtures && make
```

Verify you have 5 binaries: `hello_x86_64`, `add_x86_64`, `crypto_x86_64`, `statemachine_x86_64`, `parser_x86_64`.

## Step 2: Run the automated stress test

```bash
cd /opt/bn && bash tests/stress/run_stress.sh
```

This runs 12 test groups covering:

1. **Session lifecycle** — start, list, stop, verify cleanup
2. **Binary preload** — `session start <binary>` loads it
3. **--instance routing** — commands hit the right bridge
4. **BN_INSTANCE env var** — alternative routing
5. **3 concurrent sessions** — all listed, all respond
6. **Mutation isolation** — rename in session A doesn't affect session B
7. **Load/save/close lifecycle** — full CRUD on targets
8. **Error handling** — bad instance IDs error cleanly
9. **Rapid start/stop cycling** — 5 cycles, verify all cleaned up
10. **Decompile + IL across sessions** — read ops in parallel
11. **Parallel command storm** — concurrent searches against 2 sessions
12. **Multi-target in one session** — 2 binaries, --target enforcement

Report the exact output. If any tests fail, investigate:
- Read the test code in `tests/stress/run_stress.sh` to understand what it checks
- Check `~/.cache/bn/instances/` for stale files
- Check instance logs: `~/.cache/bn/instances/*.log`
- Try the failing command manually with `--format json` and report the output

## Step 3: Run the unit tests

```bash
cd /opt/bn && uv run pytest tests/ -v
```

All 106 tests should pass. Report any failures with full tracebacks.

## Step 4: Manual edge-case probing

After the automated tests, try these manually and report results:

### 4a: Auto-start (no prior session)
Kill any running sessions first:
```bash
for f in ~/.cache/bn/instances/*.json; do
  id=$(python3 -c "import json; print(json.load(open('$f')).get('instance_id',''))" 2>/dev/null)
  [ -n "$id" ] && bn session stop "$id" 2>/dev/null
done
sleep 2
```
Then run a command with no session running:
```bash
bn session list
```
It should return an empty list (or just the GUI bridge if one is running). Now try:
```bash
bn load /opt/bn/tests/fixtures/hello_x86_64
```
This should auto-start a headless bridge and load the binary. Verify with `bn session list` and `bn target list`.

### 4b: Double stop
```bash
bn session start
# capture the instance_id
bn session stop <id>
sleep 2
bn session stop <id>
```
The second stop should fail gracefully with "No bridge instance found".

### 4c: Instance ID collision
```bash
bn session start --instance-id fixed_test_id
bn session start --instance-id fixed_test_id
```
Report what happens — does the second start fail, or create a conflict?

### 4d: Session survives client disconnect
```bash
id=$(bn session start | python3 -c "import sys,json; print(json.load(sys.stdin)['instance_id'])")
echo "Started: $id"
# Wait, then verify it's still alive
sleep 3
bn --instance "$id" target list
bn session stop "$id"
```

## Step 5: Report

Provide a summary table:

| Category | Result | Notes |
|----------|--------|-------|
| Stress test (12 groups) | PASS/FAIL | count |
| Unit tests (106) | PASS/FAIL | count |
| Auto-start | PASS/FAIL | |
| Double stop | PASS/FAIL | |
| ID collision | PASS/FAIL | behavior |
| Session persistence | PASS/FAIL | |

List any failures with reproduction steps.
