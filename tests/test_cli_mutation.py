from __future__ import annotations

import json
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


def test_mutation_summary_transform_compacts_result():
    # #408: the compact status object an unattended loop reads instead of the full
    # audit payload.
    from bn.formatters import _mutation_summary
    ok = _mutation_summary({"success": True, "committed": True, "preview": False,
                            # post-#652: the success path emits rolled_back=False
                            "rolled_back": False,
                            "results": [{"status": "verified"}, {"status": "noop"}]})
    assert ok["kind"] == "mutation_summary"
    assert ok["success"] is True and ok["committed"] is True
    assert ok["changed_count"] == 1 and ok["noop_count"] == 1 and ok["failed_count"] == 0
    assert ok["dirty_after"] is True
    bad = _mutation_summary({"success": False, "committed": False, "rolled_back": True,
                             "results": [{"status": "verification_failed", "message": "proto mismatch"}]})
    assert bad["success"] is False and bad["failed_count"] == 1
    assert bad["first_error"] == "proto mismatch" and bad["dirty_after"] is False


def test_mutation_summary_committed_noop_is_not_dirty():
    # #408 review: `committed` is True for ANY successful non-preview mutation,
    # including an all-noop (e.g. rename to the same name). A no-op changes nothing,
    # so dirty_after must be False -- not True just because it committed.
    #
    # The fixture carries `rolled_back: False`, the shape #652 introduced on the
    # SUCCESS path. Pinning the pre-#652 `None` here is what let dirty_after
    # regress unnoticed: `rolled_back is False` alone reads True on every
    # all-noop commit, and an idempotent re-run is the common trigger.
    from bn.formatters import _mutation_summary
    noop = _mutation_summary({"success": True, "committed": True, "rolled_back": False,
                              "results": [{"status": "noop"}]})
    assert noop["committed"] is True and noop["changed_count"] == 0
    assert noop["dirty_after"] is False

    # Absent (pre-#652 bridge) must stay equivalent -- a mixed-version CLI/bridge
    # pair should not disagree about whether the DB is dirty.
    legacy = _mutation_summary({"success": True, "committed": True,
                                "results": [{"status": "noop"}]})
    assert legacy["dirty_after"] is False


def test_mutation_summary_failed_revert_is_still_dirty():
    # The other side of the #652 interaction: `rolled_back: False` on a NON-committed
    # result means the revert itself failed, so state is left behind and dirty_after
    # must stay True. Gating on `not committed` must not blunt this.
    from bn.formatters import _mutation_summary
    stuck = _mutation_summary({"success": False, "committed": False, "rolled_back": False,
                               "results": [{"status": "verification_failed",
                                            "message": "readback mismatch"}]})
    assert stuck["dirty_after"] is True
    assert stuck["first_error"] == "readback mismatch"

    # A preview whose revert failed is equally dirty.
    preview = _mutation_summary({"success": False, "committed": False, "preview": True,
                                 "rolled_back": False,
                                 "results": [{"status": "verified"}]})
    assert preview["dirty_after"] is True


def test_mutation_summary_surfaces_prototype_user_type_residue():
    # #630: an unclearable has_user_type override left behind is behaviorally
    # meaningful residue an unattended control loop must see even in the compact
    # summary -- surface it and mark dirty_after.
    from bn.formatters import _mutation_summary
    out = _mutation_summary({
        "success": False, "committed": False, "rolled_back": False,
        "preview": False, "prototype_user_type_residue": True,
        "message": "the has_user_type override could not be cleared",
        "results": [{"status": "rollback_failed"}],
    })
    assert out["prototype_user_type_residue"] is True
    assert out["dirty_after"] is True
    assert out["success"] is False
    assert "has_user_type" in out["first_error"]


def test_mutation_summary_surfaces_top_level_message_error():
    # #408 review: a failure whose only explanation is the top-level `message`
    # (no result row in FAILED_MUTATION_STATUSES -- e.g. revert cleanup failed
    # after every op verified) must still surface first_error, not drop it.
    from bn.formatters import _mutation_summary
    out = _mutation_summary({"success": False, "committed": False, "rolled_back": None,
                             "message": "revert failed: database is read-only",
                             "results": [{"status": "verified"}]})
    assert out["success"] is False and out["failed_count"] == 0
    assert out["first_error"] == "revert failed: database is read-only"


def test_go_rename_summary_emits_compact_status(fake_transport, capsys):
    # #408 review: go rename is a bulk mutation, so --summary is accepted and emits
    # the same compact status object as the single-op mutations.
    # A REAL go_rename envelope: `kind` set, `results` empty, counts in go_*.
    # The old fixture here had no `kind` and a results[] full of verified rows --
    # a shape the bridge never emits -- so it exercised the fall-through branch
    # and would have passed against a sabotaged transform.
    fake_transport({"go_rename": {"ok": True, "result": {
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [], "go_renamed_candidates": 12, "go_committed_count": 12,
        "go_verified_count": 12, "go_failed_count": 0, "skipped_user_named": 0,
        "affected_functions": [{"name": "sub_401000"}] * 12}}})
    rc = bn.cli.main(["go", "rename", "--target", "active", "--summary", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "mutation_summary"
    assert out["changed_count"] == 12 and out["committed"] is True
    assert "results" not in out and "affected_functions" not in out   # compacted


def test_go_rename_summary_reads_its_own_counters():
    # REGRESSION: `go_rename` reports through go_* counters and leaves `results`
    # EMPTY. Routing it through the generic `_mutation_summary` -- which derives
    # every count from `results[]` -- rendered a run that renamed 1783 functions
    # as `changed=0 verified=0 noop=0 failed=0 dirty_after=False`.
    #
    # `dirty_after=False` is the dangerous half: a caller reads "nothing changed",
    # closes without saving, and silently discards every recovered name. Verified
    # live against a real Go binary before the fix.
    from bn.formatters import _go_rename_summary
    live = _go_rename_summary({
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [],                      # empty on success; FAILURE rows land here
        "go_renamed_candidates": 1783, "go_committed_count": 1783,
        "go_verified_count": 1783, "go_failed_count": 0,
        "skipped_user_named": 1, "defined_count": 1784,
    })
    assert live["changed_count"] == 1783 and live["verified_count"] == 1783
    assert live["noop_count"] == 1          # already-user-named: skipped, not failed
    assert live["failed_count"] == 0
    assert live["dirty_after"] is True      # 1783 renames ARE unsaved state

    # A preview commits nothing, so the actionable count is what WOULD change and
    # the DB stays clean.
    prev = _go_rename_summary({
        "kind": "go_rename", "success": True, "committed": False, "preview": True,
        "results": [], "rolled_back": True,
        "go_renamed_candidates": 1783, "go_committed_count": 0,
        "go_verified_count": 1783, "go_failed_count": 0, "skipped_user_named": 1,
    })
    assert prev["changed_count"] == 1783 and prev["dirty_after"] is False

    # A preview whose revert FAILED leaves state behind. Real bridge shape:
    # every rename verified (zero failure rows -- the bridge's results[] always
    # equals the failure rows), the revert then failed, and the ONLY
    # explanation is the top-level message (the exact fallback the gate-on-
    # not-success comment in _go_rename_summary defends).
    stuck = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": True,
        "results": [], "rolled_back": False,
        "message": "Preview rollback failed; the view may be partially renamed",
        "go_renamed_candidates": 5, "go_committed_count": 0,
        "go_verified_count": 5, "go_failed_count": 0, "skipped_user_named": 0,
    })
    assert stuck["dirty_after"] is True and stuck["failed_count"] == 0
    assert stuck["first_error"] == (
        "Preview rollback failed; the view may be partially renamed")

    # `results` is NOT always empty here -- it carries the failure rows -- and the
    # `unsupported` early return puts its ONLY explanation in results[0].message
    # with no top-level `message`. A `failed=1` summary must never lose the reason.
    unsupported = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": False,
        "rolled_back": True,
        "results": [{"op": "rename_symbol", "status": "unsupported",
                     "message": "BinaryView does not support get_function_at"}],
        "go_renamed_candidates": 5, "go_verified_count": 0,
        "go_failed_count": 1, "go_committed_count": 0,
    })
    assert unsupported["first_error"] == "BinaryView does not support get_function_at"
    assert unsupported["failed_count"] == 1

    # Partial failure whose rollback SUCCEEDED: everything reverted, so clean --
    # but the reason still has to survive into the compact line.
    reverted = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": False,
        "rolled_back": True,
        "results": [{"op": "rename_symbol", "status": "verification_failed",
                     "message": "Live rename readback disagreed"}],
        "go_renamed_candidates": 9, "go_verified_count": 4,
        "go_failed_count": 1, "go_committed_count": 0, "skipped_user_named": 0,
    })
    assert reverted["dirty_after"] is False
    assert reverted["first_error"] == "Live rename readback disagreed"

    # Partial failure whose rollback FAILED: renames are live in the view. This is
    # the shape that must never read clean (cf. #656 on the bridge side).
    stuck_partial = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": False,
        "rolled_back": False, "message": "Rollback failed; the view may be partially renamed",
        "results": [{"op": "rename_symbol", "status": "verification_failed",
                     "message": "Live rename failed"}],
        "go_renamed_candidates": 9, "go_verified_count": 4,
        "go_failed_count": 1, "go_committed_count": 0,
    })
    assert stuck_partial["dirty_after"] is True

    # Anything that is not a go_rename envelope falls through untouched.
    other = _go_rename_summary({"success": True, "committed": True,
                                "results": [{"status": "verified"}]})
    assert other["changed_count"] == 1


def test_go_rename_summary_never_claims_reverted_renames_landed():
    # The MIRROR IMAGE of the bug this transform fixes: a LIVE run that failed and
    # was fully reverted has committed=False, so reporting the candidate count as
    # `changed` claims 1783 renames landed when nothing did. `changed` must always
    # describe what is live in the view on return, never the plan.
    from bn.formatters import _go_rename_summary
    reverted = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": False,
        "rolled_back": True,
        "results": [{"op": "rename_symbol", "status": "verification_failed",
                     "message": "readback disagreed"}],
        "go_renamed_candidates": 1783, "go_verified_count": 499,
        "go_failed_count": 1, "go_committed_count": 0, "skipped_user_named": 1,
    })
    assert reverted["changed_count"] == 0        # NOT 1783 -- nothing landed
    assert reverted["dirty_after"] is False
    assert reverted["first_error"] == "readback disagreed"


def test_go_rename_summary_reports_failure_with_no_failure_rows():
    # A revert that fails AFTER every rename verified produces ZERO failure rows,
    # so go_failed_count is 0 and its only explanation is the top-level message.
    # Gating first_error on `failed` would drop it while the view is left
    # partially renamed -- the worst combination.
    from bn.formatters import _go_rename_summary
    stuck = _go_rename_summary({
        "kind": "go_rename", "success": False, "committed": False, "preview": True,
        "rolled_back": False, "results": [],
        "message": "Preview rollback failed; the view may be partially renamed",
        "go_renamed_candidates": 1783, "go_verified_count": 1783,
        "go_failed_count": 0, "go_committed_count": 0, "skipped_user_named": 1,
    })
    assert stuck["failed_count"] == 0
    assert stuck["dirty_after"] is True
    assert stuck["first_error"] == "Preview rollback failed; the view may be partially renamed"


def test_go_rename_preview_counts_match_the_detail_renderer():
    # A preview reports what WOULD land (the verified rows), not the candidate
    # count -- candidates over-report every entry the apply skipped because the
    # function changed underneath it, and would disagree with the detail
    # renderer's own "N would rename".
    from bn.formatters import _go_rename_summary, _render_go_rename_text
    envelope = {
        "kind": "go_rename", "success": True, "committed": False, "preview": True,
        "rolled_back": True, "results": [], "go_renamed_candidates": 10,
        "go_verified_count": 7, "go_failed_count": 0, "go_committed_count": 0,
        "skipped_user_named": 3, "skipped_changed_during_apply": 3,
    }
    assert _go_rename_summary(envelope)["changed_count"] == 7
    assert "7 would rename" in _render_go_rename_text(envelope)


def test_mutation_summary_transforms_are_idempotent():
    # `_call` evaluates spill_status against the ALREADY-transformed result, so a
    # second pass must not re-zero the counts. Harmless today only because a
    # ~200-byte summary never crosses the spill threshold.
    from bn.formatters import _go_rename_summary, _mutation_summary
    go = {"kind": "go_rename", "success": True, "committed": True, "preview": False,
          "results": [], "go_renamed_candidates": 1783, "go_committed_count": 1783,
          "go_verified_count": 1783, "go_failed_count": 0, "skipped_user_named": 1}
    assert _go_rename_summary(_go_rename_summary(go)) == _go_rename_summary(go)
    plain = {"success": True, "committed": True, "results": [{"status": "verified"}]}
    assert _mutation_summary(_mutation_summary(plain)) == _mutation_summary(plain)
    assert _mutation_summary(_mutation_summary(plain))["changed_count"] == 1


def test_go_rename_default_text_reports_real_counts(fake_transport, capsys):
    # End-to-end through the CLI: the DEFAULT (compact) render must not zero out.
    fake_transport({"go_rename": {"ok": True, "result": {
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [], "go_renamed_candidates": 1783, "go_committed_count": 1783,
        "go_verified_count": 1783, "go_failed_count": 0, "skipped_user_named": 1}}})
    assert bn.cli.main(["go", "rename", "--target", "active"]) == 0
    out = capsys.readouterr().out
    assert "changed=1783" in out and "dirty_after=True" in out
    assert "changed=0" not in out


def test_go_rename_full_json_carries_top_level_ok(fake_transport, capsys):
    # #604: `go rename` hand-rolled its _call tail with `result_transform=None`, so
    # the full (--verbose) JSON came back WITHOUT the top-level `ok` every other
    # mutation emits under the #447 envelope contract. Routing it through _mutate
    # -- which applies _add_mutation_ok on the detail path -- lands that key.
    fake_transport({"go_rename": {"ok": True, "result": {
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [], "go_renamed_candidates": 3, "go_committed_count": 3,
        "go_verified_count": 3, "go_failed_count": 0, "skipped_user_named": 0,
        "affected_functions": [{"name": "sub_401000"}] * 3}}})
    rc = bn.cli.main(["go", "rename", "--target", "active", "--verbose", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True                      # the #447 key that was missing
    assert out["committed"] is True
    assert out["go_committed_count"] == 3         # --verbose keeps the full payload


def test_go_rename_defaults_to_compact_status(fake_transport, capsys):
    # #645 applies to go rename too: the compact status is the DEFAULT, detail is
    # opt-in. It hand-rolled its own tail before, so --verbose/--diffs parsed but
    # did nothing -- and go rename is the mutation most likely to emit a huge
    # payload, since it renames every candidate in the binary.
    fake_transport({"go_rename": {"ok": True, "result": {
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [], "go_renamed_candidates": 40, "go_committed_count": 40,
        "go_verified_count": 40, "go_failed_count": 0, "skipped_user_named": 0,
        "affected_functions": [{"name": "sub_401000"}] * 40}}})
    # No flags at all: renders the compact TEXT status line, not a payload dump.
    # (An explicit --format json is itself an opt-in to detail under #645, so it
    # is deliberately not the way to observe the default.)
    rc = bn.cli.main(["go", "rename", "--target", "active"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mutation:" in out and "changed=40" in out
    assert "affected_functions" not in out and "sub_401000" not in out


def test_symbol_rename_summary_emits_compact_status(fake_transport, capsys):
    # #408: --summary returns the compact status object, not the full payload; the
    # verification-aware exit code is unchanged.
    fake_transport({"rename_symbol": {"ok": True, "result": {
        "success": True, "committed": True,
        "results": [{"status": "verified", "op": "rename_symbol"}],
        "affected_functions": [{"name": "sub_401000"}] * 20}}})
    rc = bn.cli.main(["symbol", "rename", "--target", "active", "--summary",
                      "sub_401000", "player_update", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "mutation_summary"
    assert out["changed_count"] == 1 and out["committed"] is True
    assert "affected_functions" not in out and "results" not in out   # compacted


def test_symbol_rename_summary_preserves_failure_exit_code(fake_transport, capsys):
    # the compact view must NOT mask a verification failure's non-zero exit (3).
    fake_transport({"rename_symbol": {"ok": True, "result": {
        "success": False, "committed": False, "rolled_back": True,
        "results": [{"status": "verification_failed", "message": "name did not land"}]}}})
    rc = bn.cli.main(["symbol", "rename", "--target", "active", "--summary",
                      "sub_401000", "x", "--format", "json"])
    assert rc == 3
    out = json.loads(capsys.readouterr().out)
    assert out["failed_count"] == 1 and out["first_error"] == "name did not land"


def _big_batch_result(ops=200, comment_len=400):
    """A batch result whose full audit payload is far past the spill threshold --
    every op echoes its comment body three times (requested / observed /
    before_comment), the shape #645 measured at 261 KB for 115 ops."""
    body = "x" * comment_len
    return {"ok": True, "result": {
        "success": True, "committed": True, "preview": False,
        "results": [{"op": "set_comment", "status": "verified",
                     "address": hex(0x401000 + i * 4),
                     "requested": {"comment": body},
                     "observed": {"comment": body},
                     "before_comment": body} for i in range(ops)],
        "affected_functions": [{"name": f"sub_{0x401000 + i * 4:x}", "diff": body}
                               for i in range(ops)],
        "affected_types": []}}


def test_mutation_defaults_to_compact_status_line_645(fake_transport, capsys):
    """#645: mutations defaulted to --format json and echoed every per-op diff,
    `requested`, `observed`, and `before_*` field -- the largest avoidable token
    burn in a write-heavy session (a `proto set` cost ~7 KB where the status line
    costs 225 bytes). The compact status is now the default."""
    fake_transport({"set_comment": {"ok": True, "result": {
        "success": True, "committed": True, "preview": False,
        "results": [{"op": "set_comment", "status": "verified", "address": "0x401120",
                     "requested": {"comment": "x" * 500},
                     "observed": {"comment": "x" * 500},
                     "before_comment": "y" * 500}],
        "affected_functions": [{"name": "handle_request", "diff": "z" * 2000}],
        "affected_types": []}}})
    rc = bn.cli.main(["comment", "set", "--target", "active", "0x401120", "note"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("mutation: committed")
    assert "verified=1" in out and "failed=0" in out
    # None of the bulky audit fields reach stdout.
    assert "requested" not in out and "before_comment" not in out and "diff" not in out
    assert len(out) < 400


def test_mutation_verbose_restores_full_payload_645(fake_transport, capsys):
    """#645: the detail is opt-in, not gone."""
    fake_transport({"set_comment": {"ok": True, "result": {
        "success": True, "committed": True, "preview": False,
        "results": [{"op": "set_comment", "status": "verified", "address": "0x401120"}],
        "affected_functions": [{"address": "0x401120", "before_name": "a",
                                "after_name": "b", "changed": True, "diff": "--- a\n+++ b"}],
        "affected_types": []}}})
    rc = bn.cli.main(["comment", "set", "--target", "active", "--verbose",
                      "0x401120", "note"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "set_comment" in out and "[verified]" in out
    assert not out.startswith("mutation: committed")


def test_mutation_explicit_json_still_full_envelope_645(fake_transport, capsys):
    """#645: `--format json` (explicit) is the documented full-envelope contract."""
    fake_transport({"set_comment": {"ok": True, "result": {
        "success": True, "committed": True, "preview": False,
        "results": [{"op": "set_comment", "status": "verified", "address": "0x401120"}],
        "affected_functions": [], "affected_types": []}}})
    rc = bn.cli.main(["comment", "set", "--target", "active", "--format", "json",
                      "0x401120", "note"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["status"] == "verified"
    assert out["kind"] != "mutation_summary" if "kind" in out else True


def test_mutation_summary_flag_still_accepted_645(fake_transport, capsys):
    """#645: --summary/--quiet stay accepted for compatibility -- and still force
    compactness under an explicit --format json."""
    fake_transport({"set_comment": {"ok": True, "result": {
        "success": True, "committed": True,
        "results": [{"op": "set_comment", "status": "verified"}],
        "affected_functions": [{"name": "x"}] * 40}}})
    rc = bn.cli.main(["comment", "set", "--target", "active", "--quiet",
                      "--format", "json", "0x401120", "note"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "mutation_summary" and "results" not in out


def test_mutation_result_never_spills_to_an_envelope_645(fake_transport, capsys, monkeypatch):
    """#645: a spilled 38-op batch put the SPILL ENVELOPE on stdout, so `json.loads`
    raised and the agent could not confirm a batch that HAD committed. That is a
    correctness problem, not a cost one: an atomic write whose outcome is unreadable
    desyncs the agent's model of the BNDB from the BNDB. stdout must always carry the
    parseable status; the detail goes to the artifact."""
    import io

    fake_transport({"batch_apply": _big_batch_result()})
    monkeypatch.setenv("BN_SPILL_TOKENS", "500")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"ops": [{"op": "set_comment", "address": hex(0x401000 + i * 4), "comment": "x"}
                 for i in range(200)]})))
    rc = bn.cli.main(["batch", "apply", "--target", "active", "--format", "json", "-"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)          # would raise on a spill envelope
    assert payload["kind"] == "mutation_summary"
    assert payload["committed"] is True and payload["verified_count"] == 200
    assert not payload.get("spilled")
    # the detail is still reachable
    assert payload["detail_artifact_path"].endswith((".json", ".ndjson", ".txt"))
    assert "full mutation detail" in captured.err


def test_symbol_rename_builds_preview_payload(fake_transport):
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(
        [
            "symbol",
            "rename",
            "--target",
            "123:1:7",
            "--preview",
            "sub_401000",
            "player_update",
        ]
    )
    assert rc == 0
    assert calls[-1]["op"] == "rename_symbol"
    assert calls[-1]["target"] == "123:1:7"
    assert calls[-1]["params"]["preview"] is True


def test_symbol_rename_rejects_empty_new_name(fake_transport, capsys):
    """An empty/whitespace-only new name is rejected client-side (exit 2) before
    any rename_symbol op is sent -- never accepted as a 'verified' degenerate
    rename that leaves the function unnamed (#363)."""
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(["symbol", "rename", "--target", "123:1:7", "mput", ""])

    assert rc == 2
    assert "new name must be non-empty" in capsys.readouterr().err
    assert [call["op"] for call in calls] == []


def test_symbol_rename_rejects_whitespace_new_name(fake_transport, capsys):
    """Whitespace-only is as degenerate as empty -- also rejected, no op sent."""
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(["symbol", "rename", "--target", "123:1:7", "mput", "   "])

    assert rc == 2
    assert "new name must be non-empty" in capsys.readouterr().err
    assert [call["op"] for call in calls] == []


def test_symbol_rename_uses_implicit_target_when_single_target_is_open(fake_transport):
    calls = fake_transport(
        {
            "list_targets": {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                    }
                ],
            },
            "rename_symbol": {"ok": True, "result": {"preview": True}},
        }
    )

    rc = bn.cli.main(["symbol", "rename", "--preview", "sub_401000", "player_update"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "rename_symbol"]
    assert calls[1]["target"] == "123:1:7"  # implicit resolution pins the target_id (#690 R3)


def test_symbol_rename_requires_target_when_multiple_targets_are_open(fake_transport, capsys):
    fake_transport(
        {
            "list_targets": {
                "ok": True,
                "result": [
                    {
                        "target_id": "123:1:7",
                        "selector": "SnailMail_unwrapped.exe.bndb",
                        "active": True,
                    },
                    {"target_id": "123:2:8", "selector": "other.exe.bndb", "active": False},
                ],
            }
        }
    )

    rc = bn.cli.main(["symbol", "rename", "sub_401000", "player_update"])

    assert rc == 2
    assert capsys.readouterr().err == (
        "This command requires --target when multiple targets are open.\n"
        "Open targets:\n"
        "- SnailMail_unwrapped.exe.bndb [active] (target_id: 123:1:7)\n"
        "- other.exe.bndb (target_id: 123:2:8)\n"
    )


def test_function_create_builds_payload_explicit_json(fake_transport, capsys):
    calls = fake_transport(
        {
            "function_create": {
                "ok": True,
                "result": {
                    "preview": False,
                    "success": True,
                    "committed": True,
                    "message": "Function created and verified in the live Binary Ninja session.",
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verified",
                            "address": "0x401000",
                            "function": "sub_401000",
                            "requested": {"op": "function_create", "address": "0x401000"},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                },
            }
        }
    )

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--format", "json",
                      "0x401000"])

    assert rc == 0
    assert calls[-1]["op"] == "function_create"
    assert calls[-1]["target"] == "123:1:7"
    assert calls[-1]["params"] == {"address": "0x401000", "preview": False}
    # #645: an EXPLICIT --format json still returns the full audit envelope.
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "verified"


def test_function_create_text_output_renders_verified_summary(fake_transport, capsys):
    fake_transport(
        {
            "function_create": {
                "ok": True,
                "result": {
                    "preview": False,
                    "success": True,
                    "committed": True,
                    "message": "Function created and verified in the live Binary Ninja session.",
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verified",
                            "address": "0x401000",
                            "function": "sub_401000",
                            "requested": {"op": "function_create", "address": "0x401000"},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                },
            }
        }
    )

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--format", "text",
                      "--verbose", "0x401000"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "function_create 0x401000 (sub_401000) [verified]" in out


def test_function_create_forwards_preview_flag(fake_transport):
    calls = fake_transport(
        {"function_create": {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}}
    )

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "--preview", "0x401000"])

    assert rc == 0
    assert calls[-1]["params"]["preview"] is True


def test_function_create_verification_failure_exits_three(fake_transport):
    fake_transport(
        {
            "function_create": {
                "ok": True,
                "result": {
                    "preview": False,
                    "success": False,
                    "committed": False,
                    "message": "Rolled back because no function was created at the address.",
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verification_failed",
                            "address": "0x401000",
                            "message": "No function starts at 0x401000 after analysis.",
                            "requested": {"op": "function_create", "address": "0x401000"},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                },
            }
        }
    )

    rc = bn.cli.main(["function", "create", "--target", "123:1:7", "0x401000"])

    assert rc == 3


def test_render_target_line_shows_symbol_and_string_for_mapped_targets():
    # ILX #4: mapped non-function targets should surface symbol/string + section,
    # not just bare hex.
    from bn import formatters

    vtable = {
        "raw": "0x3f418",
        "normalized": "0x3f418",
        "function": None,
        "status": "mapped",
        "context": {
            "symbol": {"name": "_ZTVN17service_framework7IPCBoolE", "type": "ExternalSymbol"},
            "sections": [{"name": ".extern"}],
        },
    }
    line = formatters._render_target_line(vtable)
    assert "_ZTVN17service_framework7IPCBoolE @ 0x3f418" in line
    assert "[.extern, ExternalSymbol]" in line

    rodata_string = {
        "raw": "0x2a407",
        "normalized": "0x2a407",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {"value": "N19androidauto_service17AndroidAutoClientE", "encoding": "ascii"},
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(rodata_string)
    assert '"N19androidauto_service17AndroidAutoClientE"' in line
    assert "[.rodata]" in line

    truncated_string = {
        "raw": "0x427840",
        "normalized": "0x427840",
        "function": None,
        "status": "mapped",
        "context": {
            "string": {
                "value": "Usage: %s [OPTION]...\n" + ("A" * 16),
                "encoding": "ascii",
                "truncated": True,
            },
            "sections": [{"name": ".rodata"}],
        },
    }
    line = formatters._render_target_line(truncated_string)
    assert '"Usage: %s [OPTION]...\\nAAAAAAAAAAAAAAAA"' in line
    assert "[.rodata, truncated]" in line


def test_callsites_threads_limit_and_offset(fake_transport):
    # #454: high-fan-in sink surveys page bridge-side like xrefs.
    calls = fake_transport({"callsites": {"ok": True, "result": {
        "kind": "callsites", "items": [], "total": 0,
        "offset": 10, "limit": 5, "returned": 0, "has_more": False}}})
    rc = bn.cli.main(["callsites", "--target", "active", "--within", "main",
                      "--limit", "5", "--offset", "10", "memcpy"])
    assert rc == 0
    assert calls[-1]["op"] == "callsites"
    assert calls[-1]["params"]["limit"] == 5
    assert calls[-1]["params"]["offset"] == 10


def test_callsites_within_file_ignores_comments_and_blank_lines(fake_transport, tmp_path):
    scope_file = tmp_path / "functions.txt"
    scope_file.write_text(
        "\n# curated trial functions\nbonus_pick_random_type\n\nfx_queue_add_random\n",
        encoding="utf-8",
    )

    calls = fake_transport({"callsites": {"ok": True, "result": []}})

    rc = bn.cli.main(
        [
            "callsites",
            "--target",
            "active",
            "--within-file",
            str(scope_file),
            "crt_rand",
        ]
    )

    assert rc == 0
    assert calls[-1]["op"] == "callsites"
    assert calls[-1]["params"]["within_identifiers"] == [
        "bonus_pick_random_type",
        "fx_queue_add_random",
    ]


def test_callsites_within_file_binary_gives_clean_error(tmp_path, capsys):
    # The --within-file flag invites passing a binary path by mistake. A
    # non-UTF-8 file must surface a clean BridgeError (exit 2), not a raw
    # UnicodeDecodeError traceback (exit 1). See issue #353.
    scope_file = tmp_path / "looks_like_a_list.bin"
    scope_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\xff\xfe\xfd\x00binary")

    rc = bn.cli.main(
        ["callsites", "--target", "active", "--within-file", str(scope_file), "strcpy"]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "UTF-8 text file" in err
    assert str(scope_file) in err


def test_comment_get_uses_implicit_target_when_single_target_is_open(fake_transport, capsys):
    calls = fake_transport(
        {
            "list_targets": {
                "ok": True,
                "result": [{"target_id": "123:1:7", "selector": "SnailMail_unwrapped.exe.bndb"}],
            },
            "get_comment": {"ok": True, "result": {"address": "0x401000", "comment": "interesting branch", "has_comment": True}},
        }
    )

    rc = bn.cli.main(["comment", "get", "--format", "text", "--address", "0x401000"])

    assert rc == 0
    assert [call["op"] for call in calls] == ["list_targets", "get_comment"]
    assert calls[1]["target"] == "123:1:7"  # implicit resolution pins the target_id (#690 R3)
    assert capsys.readouterr().out == "interesting branch\n"


def test_xrefs_hints_struct_field_on_small_offset_zero_match(monkeypatch, capsys):
    """`xrefs 0x308` with 0 matches: 0x308 looks like a struct-field offset
    misread as an absolute address. Nudge toward --field."""
    monkeypatch.setattr(bn.cli, "send_request", _empty_xrefs)
    rc = bn.cli.main(["xrefs", "0x308", "--target", "active"])
    assert rc == 0
    _, err = capsys.readouterr()
    assert "--field" in err


def test_symbol_rename_text_format_renders_mutation_summary(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                    }
                ],
                "affected_functions": [
                    {
                        "address": "0x401000",
                        "before_name": "sub_401000",
                        "after_name": "player_update",
                        "changed": True,
                        "diff": "--- before:sub_401000\n+++ after:player_update",
                    }
                ],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "symbol",
            "rename",
            "--format",
            "text",
            "--target",
            "active",
            "--preview",
            "--verbose",
            "sub_401000",
            "player_update",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "preview: change applied + reverted" in output
    assert "rename_symbol function 0x401000 -> player_update" in output
    assert "0x401000 sub_401000 -> player_update" in output
    assert '"results"' not in output


def test_symbol_rename_verification_failure_returns_nonzero(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": False,
                "committed": False,
                "message": "Rolled back because live-session verification failed.",
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "verification_failed",
                        "message": "Live rename verification failed at 0x401000",
                        "requested": {
                            "identifier": "sub_401000",
                            "kind": "function",
                            "new_name": "player_update",
                        },
                        "observed": {
                            "address": "0x401000",
                            "name": "sub_401000",
                        },
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--format", "text", "--verbose", "--target",
                      "active", "sub_401000", "player_update"])

    assert rc == 3
    output = capsys.readouterr().out
    assert "rolled back" in output
    assert "failed: rename_symbol" in output
    assert "[verification_failed]" in output
    assert 'requested: {"identifier": "sub_401000"' in output
    assert 'observed: {"address": "0x401000", "name": "sub_401000"}' in output


def test_symbol_rename_noop_still_succeeds(monkeypatch):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "results": [
                    {
                        "op": "rename_symbol",
                        "kind": "function",
                        "address": "0x401000",
                        "new_name": "player_update",
                        "status": "noop",
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "player_update", "player_update"])

    assert rc == 0


def test_comment_get_empty_comment_shows_placeholder(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "get_comment"
        return {"ok": True, "result": {"address": "0x401000", "comment": "", "has_comment": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["comment", "get", "--format", "text", "--target", "active", "--address", "0x401000"])

    assert rc == 0
    assert capsys.readouterr().out == "(no comment)\n"


def test_batch_apply_stdin_forwards_preview_flag(monkeypatch, fake_transport):
    import io

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"ops": [{"op": "set_comment", "address": "0x1000", "comment": "x"}]}'),
    )
    calls = fake_transport(
        {"batch_apply": {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": []}}}
    )

    rc = bn.cli.main(["batch", "apply", "--preview", "-"])

    assert rc == 0
    assert calls[-1]["params"]["preview"] is True


def test_rename_alias_maps_to_symbol_rename(fake_transport):
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(["rename", "--target", "123:1:7", "--preview", "sub_401000", "player_update"])

    assert rc == 0
    assert calls[-1]["op"] == "rename_symbol"
    assert calls[-1]["params"]["identifier"] == "sub_401000"
    assert calls[-1]["params"]["new_name"] == "player_update"


def test_render_mutation_text_does_not_claim_rollback_when_revert_failed():
    """When a mutation failed AND its revert failed (rolled_back=False), the
    text renderer must not print 'rolled back' -- that contradicts the honest
    'view may be left modified' message and re-states the #117 symptom (#117)."""
    from bn import formatters
    value = {
        "preview": True,
        "success": False,
        "committed": False,
        "rolled_back": False,
        "message": "Preview verified, but removing the created function on revert failed; the view may be left modified.",
        "results": [{"op": "function_create", "status": "rollback_failed", "address": "0x1000", "function": "sub_1000"}],
        "affected_functions": [],
        "affected_types": [],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" not in out
    assert "rollback failed" in out
    assert "may be left modified" in out
    # the op renders under 'failed:', not as a bare '[verified]'
    assert "failed: " in out
    assert "[verified]" not in out


def test_render_mutation_text_still_reports_clean_rollback():
    """A failed batch that WAS cleanly reverted still says 'rolled back'."""
    from bn import formatters
    value = {
        "preview": False,
        "success": False,
        "committed": False,
        "rolled_back": True,
        "message": "Rolled back because live-session verification failed.",
        "results": [{"op": "rename_symbol", "status": "verification_failed", "address": "0x1000"}],
    }
    out = formatters._render_mutation_text(value)
    assert "rolled back: live verification failed" in out


def test_unknown_ref_label_prefers_symbol_then_section():
    from bn import formatters
    assert formatters._unknown_ref_label({"symbol": {"name": "some_export"}}) == "some_export"
    assert formatters._unknown_ref_label({"sections": [{"name": ".got"}]}) == ".got"
    assert formatters._unknown_ref_label({"symbol": {"name": "s"}, "sections": [{"name": ".got"}]}) == "s"
    assert formatters._unknown_ref_label({}) == ""
    assert formatters._unknown_ref_label(None) == ""


def test_xrefs_data_ref_labels_unknown_caller_by_section_or_symbol():
    from bn import formatters
    value = {
        "address": "0x18d58", "code_refs": [],
        "data_refs": [
            {"address": "0x1a254", "caller_function": None, "function": None,
             "context": {"sections": [{"name": ".got"}], "symbol": {"name": "some_export"}}},
        ],
    }
    out = formatters._render_xrefs_text(value)
    assert "some_export" in out          # symbol preferred over a bare <unknown>
    assert "<unknown>  <unknown>" not in out


# ---------------------------------------------------------------------------
# Batch 5: CLI validation/rendering (#94, #96, #100, #101, #102)
# ---------------------------------------------------------------------------


def test_comment_get_rejects_both_address_and_function(capsys):
    # #94: an address and --function are mutually exclusive (the bridge checks
    # function first, so accepting both silently dropped the address). Since the
    # positional-address alias (#291.1) replaced the argparse mutex with a handler
    # check, this is now a BridgeError (exit 2), not an argparse usage error.
    rc = bn.cli.main(["comment", "get", "--target", "active", "--address", "0x1000", "--function", "main"])
    assert rc == 2
    assert "not both" in capsys.readouterr().err


def test_comment_get_requires_a_locator(capsys):
    rc = bn.cli.main(["comment", "get", "--target", "active"])
    assert rc == 2  # neither address nor --function -> a clear error
    assert "needs a location" in capsys.readouterr().err


def test_tag_add_rejects_function_with_data_scope(capsys):
    # --data-scope is address-based and can't be combined with --function; the
    # CLI rejects the contradiction up front (BridgeError, exit 2) before any
    # bridge round-trip, parity with the function/address "not both" check.
    rc = bn.cli.main(["tag", "add", "--target", "active", "--function", "main",
                      "--type", "Important", "--data-scope"])
    assert rc == 2
    assert "data-scope" in capsys.readouterr().err.lower()


def test_render_mutation_text_set_prototype_shows_landed_signature():
    """A verified set_prototype confirms itself with the live signature (convention
    cleaned) so no follow-up `proto get` is needed."""
    from bn import formatters
    value = {
        "preview": False, "success": True, "committed": True,
        "results": [{
            "op": "set_prototype", "function": "session_read", "address": "0x401000",
            "status": "verified",
            "observed": {"address": "0x401000", "prototype": 'void __convention("cdecl")(struct Ep* ep, uint32_t flags)'},
        }],
        "affected_functions": [{"address": "0x401000", "before_name": "session_read", "after_name": "session_read", "changed": True}],
        "affected_types": [],
        "affected_summary": {"referenced": 1, "reflowed": 1},
    }
    out = formatters._render_mutation_text(value)
    assert "set_prototype session_read @ 0x401000 [verified]" in out
    assert "void __cdecl(struct Ep* ep, uint32_t flags)" in out


def test_render_mutation_text_types_declare_shows_size_and_field_delta():
    from bn import formatters
    value = {
        "preview": True, "success": True, "committed": False,
        "results": [{"op": "types_declare", "status": "verified", "defined_types": {"Ep": "struct Ep"}}],
        "affected_functions": [
            {"address": "0x10", "before_name": "a", "after_name": "a", "changed": False},
            {"address": "0x20", "before_name": "b", "after_name": "b", "changed": False},
        ],
        "affected_types": [{
            "type_name": "Ep", "name": "Ep", "changed": True,
            "before_layout": "struct Ep // size=0x214\n0x0000: int32_t x",
            "after_layout": "struct Ep // size=0x218\n0x0000: int32_t x\n0x0214: uint32_t seq",
            "layout_diff": "--- before:Ep\n+++ after:Ep\n@@ -1,2 +1,3 @@\n-struct Ep // size=0x214\n+struct Ep // size=0x218\n 0x0000: int32_t x\n+0x0214: uint32_t seq",
        }],
        "affected_summary": {"referenced": 2, "reflowed": 0},
    }
    out = formatters._render_mutation_text(value)
    assert "types_declare Ep [verified]" in out
    assert "size 0x214 -> 0x218 (+4)" in out  # single type: no redundant 'Ep:' prefix
    assert "Ep:" not in out.split("[verified]", 1)[1]  # name not repeated after the header
    assert "+ 0x0214: uint32_t seq" in out
    assert "referenced by 2 fns, 0 reflowed: a, b" in out
    assert "affected functions" not in out  # the per-function dump is not used for type ops


def test_render_mutation_text_types_declare_noop_shows_shape_and_blast_radius():
    from bn import formatters
    value = {
        "preview": False, "success": True, "committed": True,
        "results": [{"op": "types_declare", "status": "noop", "defined_types": {"Ep": "struct Ep"}, "message": "No effective change detected"}],
        "affected_functions": [{"address": "0x10", "before_name": "a", "after_name": "a", "changed": False}],
        "affected_types": [{
            "type_name": "Ep", "name": "Ep", "changed": False,
            "after_layout": "struct Ep // size=0x8\n0x0000: int32_t x\n0x0004: int32_t y",
            "message": "No effective change detected",
        }],
        "affected_summary": {"referenced": 1, "reflowed": 0},
    }
    out = formatters._render_mutation_text(value)
    assert "types_declare Ep" in out
    assert "struct Ep // size=0x8, 2 fields" in out
    assert "referenced by 1 fn, 0 reflowed: a" in out


def test_render_mutation_text_field_rename_omits_unchanged_size_line():
    """A field rename moves no bytes, so a 'size 0xNN' line would be pure noise --
    the +/- field lines carry the change."""
    from bn import formatters
    value = {
        "preview": True, "success": True, "committed": False,
        "results": [{"op": "struct_field_rename", "struct_name": "Ep", "status": "verified",
                     "old_name": "flag", "new_name": "ready"}],
        "affected_functions": [{"address": "0x10", "before_name": "user", "after_name": "user", "changed": True}],
        "affected_types": [{
            "type_name": "Ep", "name": "Ep", "changed": True,
            "before_layout": "struct Ep // size=0x4\n0x0000: uint8_t flag",
            "after_layout": "struct Ep // size=0x4\n0x0000: uint8_t ready",
            "layout_diff": "--- before:Ep\n+++ after:Ep\n@@ -1,2 +1,2 @@\n struct Ep // size=0x4\n-0x0000: uint8_t flag\n+0x0000: uint8_t ready",
        }],
        "affected_summary": {"referenced": 3, "reflowed": 1},
    }
    out = formatters._render_mutation_text(value)
    assert "- 0x0000: uint8_t flag" in out
    assert "+ 0x0000: uint8_t ready" in out
    assert "size 0x4" not in out  # size unchanged + fields moved -> no size line
    assert "referenced by 3 fns, 1 reflowed: user" in out


def test_render_mutation_text_mixed_batch_splits_type_and_direct_detail():
    """A mixed batch (types_declare + set_prototype) must show BOTH the type's
    blast radius AND the direct op's prototype/affected-function detail, and must
    not list the directly-mutated function under the type 'referenced by' line
    (Codex review on #240)."""
    from bn import formatters
    value = {
        "preview": True, "success": True, "committed": False,
        "results": [
            {"op": "types_declare", "status": "verified", "defined_types": {"Ep": "struct Ep"}},
            {"op": "set_prototype", "function": "handler", "address": "0x401000",
             "status": "verified",
             "observed": {"address": "0x401000",
                          "prototype": 'void __convention("cdecl")(struct Ep* ep)'}},
        ],
        "affected_functions": [
            # a type-referencing function that reflowed (NOT a direct-op target)
            {"address": "0x10", "before_name": "uses_ep", "after_name": "uses_ep",
             "changed": True, "direct": False},
            # the directly-mutated function (the set_prototype target)
            {"address": "0x401000", "before_name": "handler", "after_name": "handler",
             "changed": True, "direct": True},
        ],
        "affected_types": [{
            "type_name": "Ep", "name": "Ep", "changed": True,
            "before_layout": "struct Ep // size=0x4\n0x0000: int32_t x",
            "after_layout": "struct Ep // size=0x8\n0x0000: int32_t x\n0x0004: uint32_t seq",
            "layout_diff": "--- before:Ep\n+++ after:Ep\n@@ -1,2 +1,3 @@\n-struct Ep // size=0x4\n+struct Ep // size=0x8\n 0x0000: int32_t x\n+0x0004: uint32_t seq",
        }],
        "affected_summary": {"referenced": 1, "reflowed": 1},
    }
    out = formatters._render_mutation_text(value)
    # Type detail still renders.
    assert "size 0x4 -> 0x8 (+4)" in out
    # Blast radius excludes the directly-mutated function, names the type user.
    blast = [l for l in out.splitlines() if "referenced by" in l]
    assert blast, out
    assert "uses_ep" in blast[0] and "handler" not in blast[0]
    # Direct op detail is no longer hidden: the landed prototype shows...
    assert "void __cdecl(struct Ep* ep)" in out
    # ...and the directly-mutated function appears in its own affected block.
    assert "affected functions" in out
    assert "handler" in out.split("affected functions", 1)[1]


def test_blast_radius_line_caps_names_and_orders_reflowed_first():
    from bn import formatters
    value = {
        "affected_summary": {"referenced": 12, "reflowed": 1},
        "affected_functions": [
            {"address": hex(i), "before_name": f"f{i}", "after_name": f"f{i}", "changed": (i == 0)}
            for i in range(8)
        ],
    }
    line = formatters._blast_radius_line(value)
    assert line.strip().startswith("referenced by 12 fns, 1 reflowed: f0")  # reflowed name first
    assert "(+7 more)" in line  # 12 referenced - 5 shown




# --- #291.1: comment set accepts a positional address (alias for --address) ---


_COMMENT_SET_OK = {"ok": True, "result": {
    "success": True, "committed": True,
    "results": [{"op": "set_comment", "status": "verified", "address": "0x1234"}],
    "affected_functions": [], "affected_types": [],
}}


def test_comment_set_accepts_positional_address(fake_transport):
    # The natural first guess `bn comment set 0x1234 "note"` should work as an
    # alias for `--address 0x1234`, mirroring `bn read 0x.. ` (#291.1).
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "0x1234", "a note", "--target", "active"])
    assert rc == 0
    assert calls[0]["op"] == "set_comment"
    assert calls[0]["params"]["address"] == "0x1234"
    assert calls[0]["params"]["function"] is None
    assert calls[0]["params"]["comment"] == "a note"


def test_comment_set_address_flag_still_works(fake_transport):
    # The original `--address` form must keep working unchanged.
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "--address", "0x1234", "a note", "--target", "active"])
    assert rc == 0
    assert calls[0]["params"]["address"] == "0x1234"
    assert calls[0]["params"]["comment"] == "a note"


def test_comment_set_function_form_still_works(fake_transport):
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "--function", "main", "a note", "--target", "active"])
    assert rc == 0
    assert calls[0]["params"]["function"] == "main"
    assert calls[0]["params"]["address"] is None


def test_comment_set_positional_address_conflicts_with_function(fake_transport, capsys):
    # A positional address AND --function name two different locations -- reject,
    # don't silently drop one.
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "0x1234", "a note", "--function", "main", "--target", "active"])
    assert rc == 2
    assert not calls  # errored before reaching the bridge


def test_comment_set_too_many_positionals_gives_clear_error(fake_transport, capsys):
    # #312: `comment set <fn> <addr> "text"` (3 positionals) used to error on the
    # comment text as "unrecognized arguments". Now it's a clear arity error that
    # names the right form, and never reaches the bridge.
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "DoCommand", "0x403b69", "test note", "--target", "active"])
    assert rc == 2
    assert not calls
    err = capsys.readouterr().err.lower()
    assert "comment set" in err and ("single address" in err or "--function" in err)
    assert "test note" in err  # the message echoes the extra argument(s)


def test_comment_set_positional_and_flag_address_differ_conflicts(fake_transport):
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "0x1", "a note", "--address", "0x2", "--target", "active"])
    assert rc == 2
    assert not calls


def test_comment_set_requires_address_or_function(fake_transport):
    # Neither a positional/`--address` nor `--function` -> a clear error, not a
    # silently dropped value.
    calls = fake_transport({"set_comment": _COMMENT_SET_OK})
    rc = bn.cli.main(["comment", "set", "a note", "--target", "active"])
    assert rc == 2
    assert not calls


# --- #291.1 review (m1): comment get/delete also accept a positional address ---


def test_comment_get_accepts_positional_address(monkeypatch):
    captured = {}

    def fake(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"address": "0x1234", "comment": "x", "has_comment": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["comment", "get", "0x1234", "--target", "active"])
    assert rc == 0
    assert captured["op"] == "get_comment"
    assert captured["params"]["address"] == "0x1234"
    assert captured["params"]["function"] is None


def test_comment_get_positional_conflicts_with_function(monkeypatch):
    def fake(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise AssertionError("should not reach the bridge")

    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["comment", "get", "0x1234", "--function", "main", "--target", "active"])
    assert rc == 2


def test_comment_get_requires_a_locator_after_positional_alias(monkeypatch):
    def fake(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        raise AssertionError("should not reach the bridge")

    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["comment", "get", "--target", "active"])
    assert rc == 2


def test_comment_delete_accepts_positional_address(monkeypatch):
    captured = {}

    def fake(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured["op"] = op
        captured["params"] = params
        return {"ok": True, "result": {"success": True, "committed": True,
                                       "results": [{"op": "delete_comment", "status": "verified",
                                                    "address": "0x1234"}],
                                       "affected_functions": [], "affected_types": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake)
    rc = bn.cli.main(["comment", "delete", "0x1234", "--target", "active"])
    assert rc == 0
    assert captured["op"] == "delete_comment"
    assert captured["params"]["address"] == "0x1234"
    assert captured["params"]["function"] is None


def test_data_retype_builds_payload_and_previews_649(fake_transport, capsys):
    """#649: `bn data retype` drives the standard mutation loop, so a recovered
    global table can be typed through --preview + verification instead of
    `bn py exec` (which has no preview, readback, atomicity, or audit trail)."""
    calls = fake_transport({"data_retype": {"ok": True, "result": {
        "success": True, "committed": True, "preview": False,
        "results": [{"op": "data_retype", "status": "verified", "address": "0x460000",
                     "before_type": "void", "expected_type": "cmd_help_entry[257]"}],
        "affected_functions": [], "affected_types": []}}})

    rc = bn.cli.main(["data", "retype", "--target", "active", "0x460000",
                      "cmd_help_entry[257]"])
    assert rc == 0
    assert calls[-1]["op"] == "data_retype"
    assert calls[-1]["params"] == {"address": "0x460000",
                                   "new_type": "cmd_help_entry[257]", "preview": False}
    assert capsys.readouterr().out.startswith("mutation: committed")

    rc = bn.cli.main(["data", "retype", "--target", "active", "--preview", "0x460000",
                      "cmd_help_entry[257]"])
    assert rc == 0
    assert calls[-1]["params"]["preview"] is True


def test_data_retype_verification_failure_exits_3_649(fake_transport):
    fake_transport({"data_retype": {"ok": True, "result": {
        "success": False, "committed": False, "rolled_back": True,
        "results": [{"op": "data_retype", "status": "verification_failed",
                     "message": "type did not land"}]}}})
    rc = bn.cli.main(["data", "retype", "--target", "active", "0x460000", "uint32_t"])
    assert rc == 3


def test_go_rename_op_count_does_not_double_count_apply_time_skips():
    # The wire `skipped_user_named` FOLDS apply-time "changed underneath us"
    # skips in (bridge: skipped_total = skipped_user_named + skipped_during_apply)
    # while those same rows stay inside go_renamed_candidates -- summing the two
    # wire counters therefore counted every apply-time skip twice: 10 candidates
    # + 2 scan-time user-named functions is 12 distinct functions, not 15.
    from bn.formatters import _go_rename_summary
    summary = _go_rename_summary({
        "kind": "go_rename", "success": True, "committed": True, "preview": False,
        "results": [], "go_renamed_candidates": 10, "go_committed_count": 7,
        "go_verified_count": 7, "go_failed_count": 0,
        "skipped_user_named": 5, "skipped_changed_during_apply": 3,
    })
    assert summary["op_count"] == 12
    assert summary["noop_count"] == 5
    assert summary["changed_count"] == 7
