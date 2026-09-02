# bn-kernel Session Cleanup Design

## Goal

Prevent agent-created headless Binary Ninja bridges from retaining memory after a bn-kernel task ends or its agent disappears.

The observed host has 40 live headless bridges consuming 33.1 GB RSS. Twenty-three have no loaded target and still consume 15.7 GB. Their processes have been reparented to PID 1, so the agents that created them no longer own a cleanup path.

The change is scoped to `skills/bn-kernel`. It will not alter the default lifecycle of general `bn` headless sessions or terminate any bridge that predates the updated workflow.

## Lifecycle Contract

Every bn-kernel workflow that starts a headless bridge owns that bridge and must
apply both cleanup layers, unless the stable exact duplicate-ID error proves the
bridge already existed before spawning:

1. **Deterministic cleanup:** after the final read, close the exact loaded target when one exists, then stop the exact owned instance. Instance stop must still run if target close fails, loading fails, analysis raises, cooperative cancellation runs cleanup, or the result is partial. A timed-out or otherwise failed start remains ambiguous and requires exact stop; only `Bridge instance already exists with id: <instance-id>` proves non-ownership and requires leaving that pre-existing bridge untouched. Hard agent/process death is handled by the idle fallback.
2. **Crash fallback:** start the bridge with `BN_IDLE_TIMEOUT=3600`. A workflow may deliberately select another positive timeout, but it must not disable the reaper for an agent-owned bridge.

The existing bridge implementation supplies the fallback semantics: the idle window starts after preload, request completion resets activity, in-flight requests and active load jobs prevent shutdown, and the reaper is headless-only. The skill will consume that mechanism rather than add another process manager.

Cleanup remains exact and bounded. Agents must use unique instance IDs, explicit `-i/--instance`, and explicit target selectors. They must never use sticky pins or widen cleanup to `bn close --all` or an unrelated instance.

## Single-Agent Flow

The skill will present one complete lifecycle recipe:

1. Choose a unique instance ID.
2. Start through `BN_IDLE_TIMEOUT=3600 bn session start ... --instance-id ID`, including detached start when appropriate.
3. Poll the exact detached load job and bind bn-kernel to the exact instance and target.
4. Perform reads through `scoped()` with function-local bulk state.
5. In unconditional cleanup, close the exact target if it was opened and always
   run `bn session stop ID` afterward for an owned or ambiguously started instance.

The exact `Bridge instance already exists with id: <instance-id>` start error is
a pre-spawn rejection and proves the workflow never acquired ownership, so it must
not close or stop that pre-existing bridge. Any other start failure or timeout is
ambiguous because the bridge may have registered; attempt the same exact instance
stop, where a clean “instance not found” result means nothing remains registered
under that ID. Never broaden cleanup beyond the unique ID.

## Fan-Out Flow

Every parallel child owns a different bridge instance unless the exact duplicate-ID
start error proves non-ownership. Parent prompts must state the lifecycle contract
explicitly, including the one-hour fallback, unconditional exact teardown for owned
or ambiguous starts, and no teardown after that confirmed pre-spawn collision. Each
child returns only a bounded analysis summary; it never returns a live `Session`,
bridge ownership, or cleanup responsibility to the parent.

The parent may report child cleanup failures, but must not compensate with broad session termination because other agents can legitimately own concurrently listed instances.

## Skill Verification

Because this is a behavior-enforcing skill edit, verification follows the writing-skills RED/GREEN process:

1. Run pressure scenarios against the current skill text without the new lifecycle guidance. Scenarios must combine completion pressure, an ambiguous failed or timed-out start, and fan-out ownership. Record whether agents omit explicit stop, omit the idle fallback, or widen cleanup.
2. Add the smallest positive lifecycle recipe and fan-out prompt contract that address the observed failures, including the exact duplicate-ID non-ownership exception.
3. Re-run the same scenarios with the edited skill. Passing behavior requires both `BN_IDLE_TIMEOUT=3600` at start and exact instance stop on every owned or ambiguous failure path, while the confirmed pre-spawn duplicate leaves the existing bridge untouched.
4. Add counters only for rationalizations actually observed in testing.
5. Run the existing bn-kernel documentation/smoke checks and focused repository tests covering the skill installation surface. No bridge idle-reaper implementation change is expected.

## Error Handling and Compatibility

- Explicit stop reclaims memory immediately during normal completion; the one-hour timeout is only the safety net for agent death, cancellation, or missed cleanup.
- A target-close error cannot suppress instance stop.
- A start timeout or any non-duplicate failure is treated as uncertain ownership; only the exact duplicate-ID error proves non-ownership and suppresses stop.
- GUI bridges are unaffected because the existing reaper is headless-only.
- General `bn` sessions retain their current default: no idle timeout unless their caller sets one.
- Existing live bridges are not retroactively changed or mass-stopped.
