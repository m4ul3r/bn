# Luna Dogfood Follow-Ups Design

## Goal

Close the two actionable usability findings from the 16-agent Luna-backed bn-kernel dogfood run without weakening process-lifecycle safety:

1. Make collection-envelope metadata discoverable on `bn_kernel.Result`.
2. Measure high-fanout detached-start behavior with realistic command budgets and document the operational requirement.

The run completed 14 PASS, 2 PARTIAL, and 0 FAIL. No bn-kernel helper correctness defect was confirmed.

## Result Envelope Ergonomics

`Result` will gain two immutable convenience properties derived from its existing payload:

- `returned: int | None`
- `has_more: bool | None`

`returned` accepts only a non-negative integer that is not a boolean. `has_more` accepts only a real boolean. A non-mapping payload, missing key, or malformed value returns `None`. This matches the fail-soft introspection behavior of the existing `total` and `row_fields` properties; it does not weaken collection payload validation performed by `Session`.

The complete backend envelope remains available through `Result.payload`. No payload keys, collection shapes, or backend protocols change.

The bn-kernel skill will show the convenience properties in the `limit=0` schema-probe guidance and state that `payload` remains authoritative for fields without a convenience property.

## High-Fanout Lifecycle Measurement

The safety-critical spawn lock, registry ownership checks, and process-identity protocol remain unchanged. The dogfood timeouts occurred during 16 concurrent starts with 30-second orchestration command limits; retry or restart succeeded for every target.

A one-off measurement will repeat the same 16 detached BNDB starts concurrently with:

- unique instance IDs;
- `BN_SPAWN_TIMEOUT=180` for each CLI process;
- an orchestration timeout greater than the spawn budget;
- exact load-job polling to terminal state;
- exact target close and instance stop cleanup.

The report will capture per-instance start/queue latency, terminal load outcome, failures, and final session cleanup. Generated measurement output will remain a session artifact rather than a repository file.

`SKILL.md` will document that high-fanout bridge startup can exceed a harness's default 30-second command timeout. It will require the lifecycle command timeout to exceed `BN_SPAWN_TIMEOUT` and recommend a larger spawn budget on heavily loaded hosts. This guidance changes orchestration only; it does not promise unlimited parallel bridge startup.

## Error Handling and Compatibility

- Existing callers using `Result.payload`, `Result.total`, or `Result.row_fields` remain unchanged.
- Malformed convenience-property inputs return `None`; validated collection helpers continue to reject malformed wire payloads before constructing a successful result.
- Lifecycle measurement failures are reported with exact CLI stderr and do not trigger production lock changes.
- Cleanup is mandatory even after a failed start or load. Only the 16 unique measurement instances may be stopped.

## Verification

Result-property tests will cover:

1. Valid `returned` and `has_more` values.
2. Missing keys and non-mapping payloads.
3. Negative, boolean, and wrong-type `returned` values.
4. Non-boolean `has_more` values.
5. Immutability remains intact.
6. A real `limit=0` helper result exposes zero through `last.returned` and the probe verdict through `last.has_more` without leaking rows.

After the live 16-way measurement, run focused bn-kernel and documentation-drift tests, the lifecycle-focused transport/admin tests, and the full repository suite. Review `git diff --check`, then commit the existing reviewed fixes and these follow-ups without pushing.