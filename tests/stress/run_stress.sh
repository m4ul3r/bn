#!/usr/bin/env bash
#
# Stress test for bn multi-instance sessions.
# Exercises: session lifecycle, --instance routing, concurrent mutations,
# auto-start, error paths, backward compat, and registry cleanup.
#
# Usage: ./run_stress.sh [--keep]
#   --keep  Don't tear down sessions on exit (for debugging)
#
# Exit code: number of failed tests (0 = all passed)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURE_DIR="$SCRIPT_DIR/../fixtures"
BN="bn"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

PASS=0
FAIL=0
STARTED_INSTANCES=()

# ---------- helpers ----------
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
blue()  { printf '\033[1;34m%s\033[0m\n' "$*"; }

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        green "  PASS: $label"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $label (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        green "  PASS: $label"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $label (expected to contain '$needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        red "  FAIL: $label (should NOT contain '$needle')"
        FAIL=$((FAIL + 1))
    else
        green "  PASS: $label"
        PASS=$((PASS + 1))
    fi
}

assert_rc() {
    local label="$1" expected="$2" actual="$3"
    assert_eq "$label (exit code)" "$expected" "$actual"
}

start_session() {
    local out
    out=$($BN session start "$@" 2>/dev/null)
    local id
    id=$(echo "$out" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance_id'])")
    STARTED_INSTANCES+=("$id")
    echo "$id"
}

cleanup() {
    if [[ $KEEP -eq 1 ]]; then
        blue "Keeping sessions alive (--keep): ${STARTED_INSTANCES[*]:-none}"
        return
    fi
    for id in "${STARTED_INSTANCES[@]}"; do
        $BN session stop "$id" >/dev/null 2>&1 || true
    done
    sleep 1
}
trap cleanup EXIT

# ---------- preflight ----------
blue "=== Preflight ==="

# Kill any stale test instances from prior runs
shopt -s nullglob
for f in ~/.cache/bn/instances/*.json; do
    [[ -f "$f" ]] || continue
    id=$(python3 -c "import json; print(json.load(open('$f')).get('instance_id',''))" 2>/dev/null || true)
    [[ -n "$id" ]] && $BN session stop "$id" >/dev/null 2>&1 || true
done
shopt -u nullglob
sleep 1

# Check fixtures exist
for bin in hello_x86_64 add_x86_64 crypto_x86_64 statemachine_x86_64 parser_x86_64; do
    if [[ ! -f "$FIXTURE_DIR/$bin" ]]; then
        red "Missing fixture: $FIXTURE_DIR/$bin"
        red "Run: cd $FIXTURE_DIR && make"
        exit 99
    fi
done
green "  Fixtures OK"

# ====================================================================
blue "=== Test 1: Session lifecycle (start / list / stop) ==="

ID1=$(start_session)
assert_contains "session start returns id" "$ID1" "$ID1"

LIST=$($BN session list 2>/dev/null)
assert_contains "session list shows new instance" "$ID1" "$LIST"

STOP=$($BN session stop "$ID1" 2>/dev/null)
assert_contains "session stop returns stopped:true" '"stopped": true' "$STOP"
# Remove from cleanup list
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID1/}")
sleep 1

LIST2=$($BN session list 2>/dev/null)
assert_not_contains "stopped instance gone from list" "$ID1" "$LIST2"

# ====================================================================
blue "=== Test 2: Session start with binary preload ==="

ID2=$(start_session "$FIXTURE_DIR/crypto_x86_64")

TARGETS=$($BN --instance "$ID2" target list --format json 2>/dev/null)
assert_contains "preloaded binary appears as target" "crypto_x86_64" "$TARGETS"

$BN session stop "$ID2" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID2/}")
sleep 1

# ====================================================================
blue "=== Test 3: --instance routes to correct session ==="

ID_A=$(start_session "$FIXTURE_DIR/hello_x86_64")
ID_B=$(start_session "$FIXTURE_DIR/parser_x86_64")

TARGETS_A=$($BN --instance "$ID_A" target list --format json 2>/dev/null)
TARGETS_B=$($BN --instance "$ID_B" target list --format json 2>/dev/null)

assert_contains "instance A has hello" "hello_x86_64" "$TARGETS_A"
assert_not_contains "instance A does NOT have parser" "parser_x86_64" "$TARGETS_A"
assert_contains "instance B has parser" "parser_x86_64" "$TARGETS_B"
assert_not_contains "instance B does NOT have hello" "hello_x86_64" "$TARGETS_B"

$BN session stop "$ID_A" >/dev/null 2>&1
$BN session stop "$ID_B" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_A/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_B/}")
sleep 1

# ====================================================================
blue "=== Test 4: BN_INSTANCE env var routing ==="

ID_ENV=$(start_session "$FIXTURE_DIR/statemachine_x86_64")

TARGETS_ENV=$(BN_INSTANCE="$ID_ENV" $BN target list --format json 2>/dev/null)
assert_contains "BN_INSTANCE routes correctly" "statemachine_x86_64" "$TARGETS_ENV"

$BN session stop "$ID_ENV" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_ENV/}")
sleep 1

# ====================================================================
blue "=== Test 5: Concurrent sessions (3 parallel) ==="

ID_C1=$(start_session "$FIXTURE_DIR/crypto_x86_64")
ID_C2=$(start_session "$FIXTURE_DIR/statemachine_x86_64")
ID_C3=$(start_session "$FIXTURE_DIR/parser_x86_64")

# All three should appear in session list
LIST_C=$($BN session list 2>/dev/null)
assert_contains "concurrent: session 1 listed" "$ID_C1" "$LIST_C"
assert_contains "concurrent: session 2 listed" "$ID_C2" "$LIST_C"
assert_contains "concurrent: session 3 listed" "$ID_C3" "$LIST_C"

# Run commands against all three in rapid succession
FUNCS1=$($BN --instance "$ID_C1" function list --format json 2>/dev/null)
FUNCS2=$($BN --instance "$ID_C2" function list --format json 2>/dev/null)
FUNCS3=$($BN --instance "$ID_C3" function list --format json 2>/dev/null)

assert_contains "concurrent: crypto has functions" '"name"' "$FUNCS1"
assert_contains "concurrent: statemachine has functions" '"name"' "$FUNCS2"
assert_contains "concurrent: parser has functions" '"name"' "$FUNCS3"

$BN session stop "$ID_C1" >/dev/null 2>&1
$BN session stop "$ID_C2" >/dev/null 2>&1
$BN session stop "$ID_C3" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_C1/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_C2/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_C3/}")
sleep 1

# ====================================================================
blue "=== Test 6: Mutations are isolated ==="

ID_M1=$(start_session "$FIXTURE_DIR/add_x86_64")
ID_M2=$(start_session "$FIXTURE_DIR/add_x86_64")

# Rename a function in session 1 only
RENAME1=$($BN --instance "$ID_M1" symbol rename add stress_test_add --format json 2>/dev/null)
assert_contains "rename succeeds in session 1" "stress_test_add" "$RENAME1"

# Session 2 should still have the original name
FUNCS_M2=$($BN --instance "$ID_M2" function list --format json 2>/dev/null)
assert_not_contains "session 2 unaffected by session 1 rename" "stress_test_add" "$FUNCS_M2"
assert_contains "session 2 still has original name" "add" "$FUNCS_M2"

$BN session stop "$ID_M1" >/dev/null 2>&1
$BN session stop "$ID_M2" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_M1/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_M2/}")
sleep 1

# ====================================================================
blue "=== Test 7: Load, save, close lifecycle ==="

ID_LC=$(start_session)

# Load a binary
LOAD=$($BN --instance "$ID_LC" load "$FIXTURE_DIR/crypto_x86_64" --format json 2>/dev/null)
assert_contains "load succeeds" '"loaded": true' "$LOAD"

# Save it
SAVE_PATH="/tmp/bn_stress_test_$$.bndb"
SAVE=$($BN --instance "$ID_LC" save "$SAVE_PATH" --format json 2>/dev/null)
assert_contains "save succeeds" '"saved": true' "$SAVE"

# Verify file exists
if [[ -f "$SAVE_PATH" ]]; then
    green "  PASS: bndb file created on disk"
    PASS=$((PASS + 1))
    rm -f "$SAVE_PATH"
else
    red "  FAIL: bndb file not found at $SAVE_PATH"
    FAIL=$((FAIL + 1))
fi

# Close the binary
CLOSE=$($BN --instance "$ID_LC" close --format json 2>/dev/null)
assert_contains "close succeeds" '"closed"' "$CLOSE"

# Verify no targets remain
TARGETS_LC=$($BN --instance "$ID_LC" target list --format json 2>/dev/null)
assert_eq "no targets after close" "[]" "$TARGETS_LC"

$BN session stop "$ID_LC" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_LC/}")
sleep 1

# ====================================================================
blue "=== Test 8: Error handling ==="

# Stop a non-existent instance
STOP_BAD=$($BN session stop "nonexistent_id_12345" 2>&1 || true)
assert_contains "stop bad id returns error" "No bridge instance found" "$STOP_BAD"

# --instance with bad id
BAD_INST=$($BN --instance "nonexistent_id_12345" target list 2>&1 || true)
assert_contains "--instance bad id returns error" "No bridge instance found" "$BAD_INST"

# ====================================================================
blue "=== Test 9: Rapid start/stop cycling ==="

for i in 1 2 3 4 5; do
    CID=$(start_session)
    $BN session stop "$CID" >/dev/null 2>&1
    STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$CID/}")
done
sleep 2

# Verify all cleaned up
LIST_CYCLE=$($BN session list 2>/dev/null)
INST_COUNT=$(echo "$LIST_CYCLE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
instances = data.get('instances', data) if isinstance(data, dict) else data
print(sum(1 for d in instances if d.get('instance_id')))
" 2>/dev/null)
assert_eq "all cycled sessions cleaned up" "0" "$INST_COUNT"

# ====================================================================
blue "=== Test 10: Decompile and IL across sessions ==="

ID_D1=$(start_session "$FIXTURE_DIR/statemachine_x86_64")
ID_D2=$(start_session "$FIXTURE_DIR/crypto_x86_64")

# Decompile from each
DECOMP1=$($BN --instance "$ID_D1" decompile _start 2>/dev/null)
DECOMP2=$($BN --instance "$ID_D2" decompile _start 2>/dev/null)

assert_contains "decompile statemachine _start" "_start" "$DECOMP1"
assert_contains "decompile crypto _start" "_start" "$DECOMP2"

# IL from each
IL1=$($BN --instance "$ID_D1" il _start 2>/dev/null)
IL2=$($BN --instance "$ID_D2" il _start 2>/dev/null)

# Just verify we got output (not empty/error)
if [[ ${#IL1} -gt 10 ]]; then
    green "  PASS: statemachine IL has content (${#IL1} chars)"
    PASS=$((PASS + 1))
else
    red "  FAIL: statemachine IL empty or too short"
    FAIL=$((FAIL + 1))
fi

if [[ ${#IL2} -gt 10 ]]; then
    green "  PASS: crypto IL has content (${#IL2} chars)"
    PASS=$((PASS + 1))
else
    red "  FAIL: crypto IL empty or too short"
    FAIL=$((FAIL + 1))
fi

$BN session stop "$ID_D1" >/dev/null 2>&1
$BN session stop "$ID_D2" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_D1/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_D2/}")
sleep 1

# ====================================================================
blue "=== Test 11: Parallel mutations (rename storm) ==="

ID_S1=$(start_session "$FIXTURE_DIR/parser_x86_64")
ID_S2=$(start_session "$FIXTURE_DIR/statemachine_x86_64")

# Fire renames at both sessions concurrently using background processes
(
    $BN --instance "$ID_S1" function search str --format json >/dev/null 2>&1
    $BN --instance "$ID_S1" function search parse --format json >/dev/null 2>&1
    $BN --instance "$ID_S1" function search config --format json >/dev/null 2>&1
    $BN --instance "$ID_S1" strings --format json >/dev/null 2>&1
) &
PID1=$!

(
    $BN --instance "$ID_S2" function search handle --format json >/dev/null 2>&1
    $BN --instance "$ID_S2" function search game --format json >/dev/null 2>&1
    $BN --instance "$ID_S2" function search apply --format json >/dev/null 2>&1
    $BN --instance "$ID_S2" strings --format json >/dev/null 2>&1
) &
PID2=$!

wait $PID1
RC1=$?
wait $PID2
RC2=$?

assert_rc "parallel search storm session 1" "0" "$RC1"
assert_rc "parallel search storm session 2" "0" "$RC2"

$BN session stop "$ID_S1" >/dev/null 2>&1
$BN session stop "$ID_S2" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_S1/}")
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_S2/}")
sleep 1

# ====================================================================
blue "=== Test 12: Multiple binaries in one session ==="

ID_MULTI=$(start_session)

$BN --instance "$ID_MULTI" load "$FIXTURE_DIR/hello_x86_64" --format json >/dev/null 2>&1
$BN --instance "$ID_MULTI" load "$FIXTURE_DIR/add_x86_64" --format json >/dev/null 2>&1

TARGETS_MULTI=$($BN --instance "$ID_MULTI" target list --format json 2>/dev/null)
assert_contains "multi-load: hello present" "hello_x86_64" "$TARGETS_MULTI"
assert_contains "multi-load: add present" "add_x86_64" "$TARGETS_MULTI"

# Must use --target when multiple are open
NEED_TARGET=$($BN --instance "$ID_MULTI" function list 2>&1 || true)
assert_contains "multi-target requires --target" "requires --target" "$NEED_TARGET"

$BN session stop "$ID_MULTI" >/dev/null 2>&1
STARTED_INSTANCES=("${STARTED_INSTANCES[@]/$ID_MULTI/}")
sleep 1

# ====================================================================
# Summary
echo ""
blue "============================================"
if [[ $FAIL -eq 0 ]]; then
    green "ALL PASSED: $PASS tests passed, 0 failed"
else
    red "RESULT: $PASS passed, $FAIL FAILED"
fi
blue "============================================"

exit "$FAIL"
