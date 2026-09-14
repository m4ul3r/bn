from __future__ import annotations

import ast
import collections
import functools
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import NamedTuple

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403

REPO = Path(__file__).resolve().parents[1]


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


def test_mutation_summary_flags_empty_results_as_unmeasured_not_zero():
    # #684: an op that reports through its OWN counters instead of `results[]`
    # and forgets to register a `summary_transform` reaches the GENERIC summary
    # with an empty `results[]` on an otherwise-successful envelope. Rendering
    # that as `changed=0 verified=0 noop=0 failed=0 dirty_after=False` reads as
    # a CONFIRMED no-op to an agent, which closes without saving and silently
    # discards whatever the op actually did (the #683 `go_rename` regression).
    # The generic summary must flag this as unmeasured instead of asserting a
    # zero-change measurement it never actually took.
    #
    # #684 review round 2: `dirty_after: None` was FALSY under every
    # truthiness check a control loop actually writes (`if not dirty_after:`),
    # so it read IDENTICALLY to a confirmed clean no-op and the fix was a
    # no-op on the JSON path. Fail safe instead: unmeasured reports
    # `dirty_after: True` (a spurious save is cheap) and nulls the derived
    # counts (None, not a confident 0) while surfacing an explanation on
    # `first_error` -- the one key an agent contract already tells callers to
    # check.
    from bn.formatters import _mutation_summary, _render_mutation_summary_text
    out = _mutation_summary({"success": True, "committed": True, "results": []})
    assert out["measured"] is False
    assert out["dirty_after"] is True          # fail-safe, NOT a confirmed clean state
    assert out["op_count"] == 0                # literally true: zero results[] rows
    assert out["changed_count"] is None        # unknown, NOT a confirmed 0
    assert out["verified_count"] is None
    assert out["noop_count"] is None
    assert out["failed_count"] is None
    assert out["first_error"] and "unmeasured" in out["first_error"].lower()
    text = _render_mutation_summary_text(out)
    assert "dirty_after=True" in text
    assert "unmeasured" in text.lower()

    # A genuine zero-change result (a real `noop` STATUS ROW inside a non-empty
    # `results[]`) must stay measured and distinct from the unmeasured case above
    # -- the whole point is telling "verified nothing changed" apart from "we
    # never actually counted."
    genuine_noop = _mutation_summary({"success": True, "committed": True,
                                      "rolled_back": False,
                                      "results": [{"status": "noop"}]})
    assert genuine_noop["measured"] is True
    assert genuine_noop["dirty_after"] is False
    assert genuine_noop["noop_count"] == 1 and genuine_noop["changed_count"] == 0
    assert genuine_noop["first_error"] is None
    noop_text = _render_mutation_summary_text(genuine_noop)
    assert "unmeasured" not in noop_text.lower()


def test_mutation_summary_unmeasured_flag_covers_failure_envelopes_too():
    # The same empty-results ambiguity applies on the failure side: a bespoke op
    # that claims failure without ever populating `results[]` must not have its
    # dirty_after silently resolve to a confident False either -- it must fail
    # safe to True like the success-side case above, and `first_error` must
    # carry the unmeasured explanation ALONGSIDE the existing failure message,
    # not overwrite it.
    from bn.formatters import _mutation_summary
    out = _mutation_summary({"success": False, "committed": False, "results": []})
    assert out["measured"] is False
    assert out["dirty_after"] is True
    assert out["changed_count"] is None
    assert "mutation failed" in out["first_error"]
    assert "unmeasured" in out["first_error"].lower()


def test_unmeasured_envelope_is_truthy_dirty_after_unlike_confirmed_noop():
    # #684 review: a JSON consumer doing `if not summary["dirty_after"]: close()`
    # -- the naive pattern a control loop actually writes -- must now SAVE
    # (i.e. NOT take the close-without-saving branch) on an unmeasured
    # envelope, while a genuine measured all-noop must still take it. Before
    # this fix `dirty_after: None` and `dirty_after: False` were
    # indistinguishable under that check; this pins the behavioural
    # difference, not just the presence of the new `measured` key.
    from bn.formatters import _mutation_summary

    def closes_without_saving(summary: dict) -> bool:
        return not summary["dirty_after"]

    unmeasured = _mutation_summary({"success": True, "committed": True, "results": []})
    assert unmeasured["measured"] is False
    assert closes_without_saving(unmeasured) is False   # now falls through to save

    clean_noop = _mutation_summary({"success": True, "committed": True,
                                     "rolled_back": False,
                                     "results": [{"status": "noop"}]})
    assert clean_noop["measured"] is True
    assert closes_without_saving(clean_noop) is True    # unchanged: still skips

    assert closes_without_saving(unmeasured) != closes_without_saving(clean_noop)


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
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True, "results": [{"status": "verified"}]}}})

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


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_local_rename_rejects_empty_new_name(fake_transport, capsys, bad_name):
    """An empty/whitespace-only new name is rejected client-side (exit 2) before
    any local_rename op is sent -- mirrors _symbol_rename's guard (#605)."""
    calls = fake_transport({"local_rename": {"ok": True, "result": {"preview": True}}})

    rc = bn.cli.main(["local", "rename", "--target", "123:1:7", "sub_401000", "var_8", bad_name])

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
            "rename_symbol": {"ok": True, "result": {"preview": True, "results": [{"status": "verified"}]}},
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
        {"function_create": {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": [{"status": "verified"}]}}}
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


def test_unmeasured_mutation_success_exits_four(monkeypatch, capsys):
    """#715: a successful mutation whose result carries no `results[]` rows is
    `measured: false` in its compact summary, so the outcome could not be
    verified. That is neither a failure (exit 3) nor a confirmed success (exit
    0): it is exit 4 "applied but unverifiable", so a script that only checks
    `$?` cannot read it as a clean success."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "rename_symbol"
        return {
            "ok": True,
            "result": {
                "preview": False,
                "success": True,
                "committed": True,
                "rolled_back": False,
                # No `results[]`: the op measures through its own counters.
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "player_update"])

    assert rc == 4
    # The exit code agrees with the status line the same call printed.
    assert "warning: unmeasured" in capsys.readouterr().out


def test_unmeasured_mutation_still_exits_four_in_verbose_mode(monkeypatch):
    """The exit code must not depend on the detail level: `--verbose` still
    renders the full payload, but an unverifiable write is exit 4 either way."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"preview": False, "success": True, "committed": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "--verbose",
                      "sub_401000", "player_update"])

    assert rc == 4


# The three classifications the mutation contract distinguishes, and the output
# combinations the mutation reference tabulates. Both are the reference's own
# lists, and this cell is the measurement behind the sentence there.
_CLASSIFICATIONS = {
    0: {"success": True, "committed": True, "results": [{"status": "verified"}]},
    3: {"success": False, "committed": False,
        "results": [{"status": "invalid_request"}]},
    4: {"success": True, "committed": True, "results": []},
}
_OUTPUT_COMBINATIONS = {
    "text": [],
    "summary": ["--summary"],
    "verbose": ["--verbose"],
    "json": ["--format", "json"],
    "ndjson": ["--format", "ndjson"],
}


@pytest.mark.parametrize("expected", sorted(_CLASSIFICATIONS))
def test_no_output_flag_can_turn_a_nonzero_classification_into_zero(
        monkeypatch, tmp_path, capsys, expected):
    """The sentence `skills/bn/reference/mutating.md` states about output flags,
    measured rather than asserted in prose.

    Round 15 measured the old sentence -- "No combination changes the exit code"
    -- FALSE: the same reply exits 0 under the default status line and 2 under
    `--out` on a destination the CLI cannot write. The divergence is legitimate
    (a command asked for a file it could not produce did not do what was asked,
    and reporting 0 there would be the lie), so the DOC was the defect. What is
    true, and what a `$?`-only consumer actually depends on, is the direction:
    every deliverable combination reports the same classification, and an
    undeliverable output can only replace it with the documented 2 -- never with
    0.

    Red under the repair that would make the doc's first sentence false again:
    handing `_mutation_exit_code` its summary only on the compact path (so
    `--verbose`/`--format json` lose the `measured` verdict) reds the `4` row.
    """
    result = _CLASSIFICATIONS[expected]

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, **kwargs):
        return {"ok": True, "result": result}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    argv = ["symbol", "rename", "--target", "active", "sub_401000", "x"]

    deliverable = {
        name: bn.cli.main([*argv, *flags])
        for name, flags in _OUTPUT_COMBINATIONS.items()
    }
    deliverable["out"] = bn.cli.main(
        [*argv, "--out", str(tmp_path / "detail.json")])
    capsys.readouterr()
    assert set(deliverable.values()) == {expected}, deliverable

    # ...and the one divergence, in the one direction the doc allows: the parent
    # of this destination is a FILE, so the write cannot land however the
    # mutation itself was classified.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    undeliverable = bn.cli.main([*argv, "--out", str(blocker / "detail.json")])
    assert "Failed to write --out file" in capsys.readouterr().err
    assert undeliverable == 2, (
        f"an undeliverable output must report the documented 2, not "
        f"{undeliverable}"
    )


def test_a_rejected_flag_value_is_a_2_with_nothing_sent(monkeypatch):
    """The counterexample the mutation reference now names, measured.

    Round 16 wrote "a 2 on a mutation says the requested output did not arrive,
    not that the write did not land" and pinned it as text. The falsification
    lens refuted it in one line: an invalid flag VALUE is rejected by the
    parser, which exits 2 before a request is sent -- so the write did not land,
    and a consumer following that sentence would re-read the view instead of
    re-issuing a mutation that never happened. The reference now says what a 2
    does NOT tell you, and this is the measurement behind it: a 2 with an empty
    wire log, next to the same command exiting 0 with one request sent.
    """
    sent = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, **kwargs):
        sent.append(op)
        return {"ok": True, "result": {"success": True, "committed": True,
                                       "results": [{"status": "verified"}]}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    argv = ["symbol", "rename", "--target", "active", "sub_401000", "x"]

    with pytest.raises(SystemExit) as refused:
        bn.cli.main([*argv, "--format", "jsonn"])
    assert refused.value.code == 2
    assert sent == [], f"the parser sent a request before refusing: {sent}"

    # ...and the same command with an accepted value does send and does exit 0,
    # so the 2 above is the flag value and not the reply behind it.
    assert bn.cli.main([*argv, "--format", "json"]) == 0
    assert len(sent) == 1, sent


def test_the_cases_the_reference_lists_for_exit_2_really_are_2(monkeypatch, capsys):
    """Every case the mutation reference names for a 2 on this path, measured --
    and the one it deliberately excludes.

    Round 17's falsification lens refuted the sentence this replaces: it said
    "2 is also this path's code for a refused request", and a refused mutation
    is exit 3, which the same file's status table states 43 lines earlier. The
    lens's second point was procedural and is the reason this cell exists: the
    clause was parked in `_EXIT_CODE_PINS`, which asserts only that the literal
    string is PRESENT, so a false clause could sit there indefinitely. Each case
    the corrected sentence lists is executed here, and so is the exclusion.
    """
    from bn.transport import BridgeError

    argv = ["symbol", "rename", "--target", "active", "sub_401000", "x"]

    def reply(result):
        def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                              instance_id=None, **kwargs):
            return {"ok": True, "result": result}
        return fake_send_request

    def raises(exc):
        def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                              instance_id=None, **kwargs):
            raise exc
        return fake_send_request

    # "a bridge this CLI could not reach"
    monkeypatch.setattr(bn.cli, "send_request",
                        raises(BridgeError("Failed to contact Binary Ninja bridge")))
    assert bn.cli.main(argv) == 2

    # "a reply it could not classify at all" -- a result that is not an object
    monkeypatch.setattr(bn.cli, "send_request", reply(["verified"]))
    assert bn.cli.main(argv) == 2

    # "a flag value rejected before anything was sent" is measured on its own in
    # test_a_rejected_flag_value_is_a_2_with_nothing_sent, which also proves the
    # wire log stays empty.

    # ...and the exclusions. The first is what the refuted clause got wrong: on
    # a mutation a refusal is 3, whether it arrives as a row or as a status.
    monkeypatch.setattr(bn.cli, "send_request", reply(
        {"success": False, "committed": False,
         "results": [{"status": "invalid_request"}]}))
    assert bn.cli.main(argv) == 3
    monkeypatch.setattr(bn.cli, "send_request",
                        raises(BridgeError("refused", status="invalid_request")))
    assert bn.cli.main(argv) == 3

    # The second: "a reply carrying ONE field this CLI cannot read" is refused
    # and disclosed, a verdict is still derived from the rest, and the run exits
    # by that verdict. Both branches, because the sentence names both -- a row
    # status no one could read beside a clean success is the unmeasured 4, and
    # the same beside a reported failure is still 3.
    monkeypatch.setattr(bn.cli, "send_request", reply(
        {"success": True, "committed": True, "results": [{"status": 5}]}))
    assert bn.cli.main(argv) == 4
    monkeypatch.setattr(bn.cli, "send_request", reply(
        {"success": False, "committed": False,
         "results": [{"status": "verification_failed"}, {"status": 5}]}))
    assert bn.cli.main(argv) == 3
    capsys.readouterr()


def test_a_locally_built_result_survives_a_renderer_that_cannot_read_it(
        monkeypatch, capsys):
    """The receiving function the behavioural half was not covering.

    `_emit_result` is the rendering tail the admin commands share -- they
    assemble their result from several bridge replies instead of making one
    `_call`, so they never pass through `_call`'s bindings -- and it binds its
    own `text_renderer` to the malformed-result rule. Round 16's falsification
    lens removed that binding and 287 behavioural tests stayed green: the site
    was load-bearing (a renderer raising there escapes `main()`, which catches
    only BridgeError, for exit 1 and a traceback) and had no cell anywhere. A
    guard nothing exercises is the thing this PR is about.

    `capabilities` is the cheapest command on that path: it builds its result
    from the command registry and needs no bridge at all, so what is measured
    here is the rule and nothing else.
    """
    def renderer_that_cannot_read_it(value):
        raise TypeError("unhashable type: 'dict'")

    # String form: it imports the handler module, so this cell does not depend
    # on some earlier import having bound the submodule attribute.
    monkeypatch.setattr("bn.commands.admin._render_capabilities_text",
                        renderer_that_cannot_read_it)

    rc = bn.cli.main(["capabilities", "--format", "text"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "could not render the capabilities result as text" in err, err
    assert "Rerun with --format json" in err, err


def test_invariant_guard_unmeasured_mutation_failure_still_exits_three(monkeypatch):
    """Ordering: an unmeasured envelope that also reports failure is a failure
    (exit 3), not "applied but unverifiable" (exit 4)."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": False, "result": {"preview": False, "success": False,
                                        "committed": False,
                                        "message": "revert failed after apply"}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "player_update"])

    assert rc == 3


def test_invariant_guard_op_with_its_own_summary_is_measured_and_exits_zero(monkeypatch, capsys):
    """#715: "no `results[]` rows" is NOT the exit-4 rule -- `measured: false`
    is. `go rename` reports its work through its own counters and registers a
    compact summary that counts them, so an empty `results[]` (that array holds
    only its FAILURE rows) is a fully measured clean run: exit 0.

    The exit code must therefore be derived with the transform THIS call renders
    with, not with a generic recompute; a recompute would see no rows, call the
    run unmeasured and exit 4 on a successful bulk rename.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "go_rename"
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": True, "committed": True,
                                       "rolled_back": False,
                                       "go_renamed_candidates": 1783,
                                       "go_committed_count": 1783,
                                       "go_verified_count": 1783,
                                       "go_failed_count": 0,
                                       "skipped_user_named": 4,
                                       "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["go", "rename", "--target", "active"])

    assert rc == 0
    # The exit code agrees with the status line: this run WAS measured.
    assert "warning: unmeasured" not in capsys.readouterr().out


def test_unclassifiable_mutation_result_covers_overflowed_wire_numbers(monkeypatch, capsys):
    """The transform this guard wraps AGGREGATES wire numbers, so its failure
    surface includes arithmetic, not just parsing. JSON has no bound on a numeric
    literal: `1e999` decodes to `float("inf")`, and `int(inf)` raises
    `OverflowError` -- an `ArithmeticError`, outside the exception set a text
    renderer can throw. It must still be the documented exit 2, not a traceback.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": True, "committed": True,
                                       # what `1e999` on the wire decodes to
                                       "go_renamed_candidates": float("inf"),
                                       "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["go", "rename", "--target", "active"])

    assert rc == 2
    assert "malformed or newer than this CLI" in capsys.readouterr().err


# The two shapes a counter can arrive in that no count reads out of -- and they
# take DIFFERENT documented exits, which is precisely why the contract is keyed
# on "did a classification survive" rather than on "was some field unreadable":
#
#   * `"many"` is REFUSED by the read: the count is left unknown and the field
#     is disclosed by name, so the summary, the status line and the spill status
#     are all built. The classification stands, and it is a FAILURE -- read from
#     `success: false` and from a `results[]` row whose status parsed cleanly,
#     neither of which is the skewed field. Exit 3.
#   * `float("inf")` -- what `1e999` decodes to -- makes the read RAISE
#     (`int(inf)` is an `OverflowError`), so no status could be built on any
#     format. That is the documented one-directional output override: an
#     undeliverable status replaces the code with 2 and can never turn a failed
#     or an unmeasured mutation into a clean zero.
@pytest.mark.parametrize("counter,expected", [("many", 3), (float("inf"), 2)],
                         ids=["unparseable", "non-finite"])
@pytest.mark.parametrize("extra", [[], ["--verbose"], ["--summary"],
                                   ["--format", "json"], ["--format", "ndjson"],
                                   ["--out"]],
                         ids=["default", "verbose", "summary", "json", "ndjson", "out"])
def test_a_failing_mutation_with_an_unreadable_counter_is_still_a_clean_exit(
        monkeypatch, capsys, tmp_path, extra, counter, expected):
    """A FAILING result short-circuits before the summary transform is ever run,
    so the exit-code guard never sees it -- and `_call` then feeds that same
    transform to the renderer AND to the spill-status builder. Every one of
    those steps must survive an unreadable counter on every format: the process
    must leave with a documented code, never a traceback (exit 1) after the exit
    code was already decided.

    Parametrized over every output path, and asserting the exact code, on
    purpose: the first cut covered only the default format and accepted `2 or
    3`, so it stayed green while three machine formats still crashed.

    The refused counter is **3**, not 2. The earlier expectation rested on "a
    status parsed out of a response that does not parse", and that premise is
    false here: the response parses, one counter does not, and the counter is
    not what the failure verdict rests on. `_mutation_reports_failure` reads
    `success` and the row statuses -- both clean -- so the CLI KNOWS this
    mutation failed and was reverted. Reporting that as "I could not determine
    the outcome" would discard a hard fact, and would hand a `$?`-only consumer
    the code it also gets for an unreachable bridge, inviting a retry of a
    request that deterministically fails. It is also the precedence this
    contract already documents: a failure wins over an unmeasured run, 3 before
    4.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": False, "committed": False,
                                       "go_renamed_candidates": counter,
                                       "results": [{"status": "verification_failed"}]}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    argv = [*extra, str(tmp_path / "detail.json")] if extra == ["--out"] else extra

    rc = bn.cli.main(["go", "rename", "--target", "active", *argv])

    assert rc == expected, rc
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err, captured.err
    if expected == 2:
        assert "malformed or newer than this CLI" in captured.err, captured.err
    else:
        # The classification survived the delivery step, so the code the
        # classifier chose is the code the process leaves with.
        assert "malformed or newer than this CLI" not in captured.err, captured.err


@pytest.mark.parametrize("rows", [5, True, "verified", {"a": 1},
                                  [{"status": ["verification_failed"]}]],
                         ids=["int", "bool", "string", "mapping", "unhashable-status"])
def test_malformed_results_field_is_a_clean_bridge_error(monkeypatch, capsys, rows):
    """The exit-code helper reads `results[]` BEFORE it runs any transform, and
    that preamble parses the bridge response too: it iterates the field and looks
    each row's `status` up in a set. A `results` that is not a list of dicts with
    hashable statuses is a malformed response, so it owes the same documented
    exit 2 -- not a `TypeError` out of `main()`, and not the exit 4 an
    iterable-but-meaningless shape used to fall through to, which would report an
    unreadable response as a real (if unmeasured) mutation.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"preview": False, "success": True,
                                       "committed": True, "results": rows}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "player_update"])

    assert rc == 2, rc
    assert "malformed or newer than this CLI" in capsys.readouterr().err


_OUTPUT_PATHS = pytest.mark.parametrize(
    "extra", [[], ["--verbose"], ["--summary"], ["--format", "json"],
              ["--format", "ndjson"], ["--out"]],
    ids=["default", "verbose", "summary", "json", "ndjson", "out"])


def _argv_for(extra: list[str], tmp_path) -> list[str]:
    return [*extra, str(tmp_path / "detail.json")] if extra == ["--out"] else extra


@pytest.mark.parametrize("result", [[], [{"status": "verified"}], "committed", 7, True],
                         ids=["empty-list", "list", "string", "int", "bool"])
@_OUTPUT_PATHS
def test_a_non_object_mutation_result_is_a_clean_bridge_error(monkeypatch, capsys, tmp_path,
                                                              result, extra):
    """The most malformed shape of all used to be the one that exited 0.

    `_mutation_exit_code` opened with `if not isinstance(result, dict): return 0`,
    so a version-skewed bridge that answered a WRITE with a list, a string or a
    number had its unconfirmed mutation reported to a `$?`-only consumer as a
    clean success -- the exact #715 failure mode, on a shape strictly more
    broken than the non-object compact summary the same helper already rejects.
    Nothing about this result is classifiable, which is the documented exit 2.

    Over every output path because the exit code is decided before the format is
    applied: a claim like that is only worth anything if it was checked on the
    machine formats too.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": result}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000",
                      "player_update", *_argv_for(extra, tmp_path)])

    assert rc == 2, rc
    assert "not an object" in capsys.readouterr().err


@pytest.mark.parametrize("compact", [{"kind": "mutation_summary"},
                                     {"kind": "mutation_summary", "measured": None},
                                     {"kind": "mutation_summary", "measured": "true"}],
                         ids=["absent", "null", "string"])
def test_a_summary_with_no_measured_verdict_is_a_clean_bridge_error(monkeypatch, capsys,
                                                                    compact):
    """The exit-4 test was `compact.get("measured") is False`, so SILENCE read as
    measured and the mutation exited 0.

    A summary that carries no `measured` -- one from a bridge predating #684, or
    an already-compact result short-circuited through the idempotence path --
    said nothing about whether the write was verified. "Nothing said" is the
    unclassifiable case, which is the documented 2, and is the same rule the
    helper already applies to a summary that is not an object at all.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"preview": False, "success": True,
                                       "committed": True, "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "_mutation_summary", lambda result: compact)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000",
                      "player_update"])

    assert rc == 2, rc
    assert "not a boolean" in capsys.readouterr().err


@_OUTPUT_PATHS
def test_an_ok_reply_carrying_no_result_is_a_clean_bridge_error(monkeypatch, capsys,
                                                                tmp_path, extra):
    """`result = response["result"]` sat outside every guard, so a reply of
    `{"ok": true}` -- a bridge one protocol version ahead, or a reply truncated
    on the wire -- left `main()` as a bare `KeyError` (exit 1 and a traceback) on
    every output format. main() catches only `BridgeError`, so an unreadable
    envelope owes the same documented 2 as an unreadable result.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000",
                      "player_update", *_argv_for(extra, tmp_path)])

    assert rc == 2, rc
    assert "carries no `result`" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--format", "json"], ["--format", "ndjson"], ["--out"]],
                         ids=["json", "ndjson", "out"])
def test_a_deeply_nested_result_is_a_clean_bridge_error(monkeypatch, capsys, tmp_path, extra):
    """`RecursionError` was added to the malformed-result set with a comment
    claiming it covered `json.dumps` -- but the serializer was not inside the
    rule, so a response deep enough to exhaust the encoder still left `main()`
    as a raw `RecursionError` (exit 1) on every machine format. Depth is a
    property of the RESPONSE, so it owes the documented 2.

    Scoped to the paths that actually serialize the payload. Under `--format
    text` this command's renderer prints known fields and never walks the deep
    one, so there is nothing there to guard -- asserting a code for it would
    pin a formatter's internals rather than this rule.
    """
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(100_000):
        child: dict[str, object] = {}
        cursor["next"] = child
        cursor = child
    # The premise, executed: this payload really does break the serializer. If a
    # future encoder survives it, this fails instead of passing vacuously.
    with pytest.raises(RecursionError):
        json.dumps({"deep": nested})

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None,
                          spawn_missing_named=False, resolved=False, **kwargs):
        return {"ok": True, "result": {"items": [{"name": "f", "deep": nested}],
                                       "total": 1, "offset": 0, "limit": None,
                                       "returned": 1, "has_more": False}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list", "--target", "active", *_argv_for(extra, tmp_path)])

    assert rc == 2, rc
    err = capsys.readouterr().err
    assert "could not serialize the functions result" in err, err
    assert "malformed or newer than this CLI" in err, err


# --- The transform-guard property, restated over a population it cannot miss ---
#
# Five review rounds each found ONE more place a caller-supplied bridge-result
# transform ran outside the malformed-result rule: three `try` blocks, then a
# handler module, then `truncation_note`, then an aliased call
# (`_t = result_transform; _t(result)`). Every one of those guards asserted
# something true about the sites its author was looking at.
#
# So the population is read off the SOURCE instead of listed, and the
# requirement is moved from the call site to the BOUNDARY: `cli.py` binds each
# transform it receives to the rule before calling it or handing it on, which
# makes the call shape (alias, attribute dispatch, another module) irrelevant.
_CLI = REPO / "src" / "bn" / "cli.py"
_GUARD = "_apply_result_transform"      # the rule
_BOUNDARY = "_guarded_transform"        # binds a transform TO the rule
# The rule's own implementation invokes a transform directly at exactly ONE
# place -- the call inside the `try` that catches `_MALFORMED_RESULT_ERRORS` --
# so the exemption is that CALL, and not the function containing it. Exempting
# the whole body was the same mistake one scope out: a second raw
# `transform(result)` placed anywhere else in `_apply_result_transform` (before
# the `try`, in the handler, in a `finally`) was unreported while every guard
# stayed green and a malformed result left `main()` as a raw traceback
# (round 14). The boundary used to be skipped too, and that exemption was true
# of nothing: `_guarded_transform` invokes no transform -- it closes over one
# and delegates to the rule -- so it needed no exemption, and having one meant
# a raw `transform(probe)` injected into the boundary itself was unreported, in
# the one function whose stated purpose is that this cannot happen (round 13).
# The ONE name cli.py may bind the string `"result"` to outside the unwrap
# helper: the key it writes into a fan-out row. Round 8 made that name the
# exemption, and the name itself then became the dodge -- `response.get(
# _RESULT_ROW_KEY)` reads the key carrying no `"result"` token at all. So the
# exemption is no longer the NAME: it is the one WRITE that pairs this key with
# the helper's return value, and every read of the key, by constant or by
# alias, is a finding.
_RESULT_ROW_KEY_NAME = "_RESULT_ROW_KEY"
# A bridge-result transform is a callable the CALLER hands the CLI, which is
# then applied to a bridge result. Round 6 defined that population by SHAPE --
# an annotation, a direct invocation, or a nine-token name vocabulary -- and
# round 7 escaped all three at once: a parameter typed `Any`, named outside the
# vocabulary, forwarded to a helper and dispatched there out of a dict.
#
# Round 7 replaced the recogniser with a DECLARATION -- every `_call` parameter
# not declared data -- and round 8 escaped THAT, because `_call` is not the only
# door: `_emit_result`, `_render_result` and `_mutation_exit_code` each bind a
# caller-supplied transform of their own, and none of them was seeded. Removing
# their boundary bindings left the module green and a raising summary left
# `main()` as a raw `KeyError`.
#
# A declaration with a doorway is a recogniser again. So the static population
# below is TOTAL: every parameter of every function in `cli.py`, with no
# annotation, no name, no entry point and no exemption. Precision comes from
# what is PROVEN GUARDED, not from what was recognised as a transform, and the
# dispatch shapes are the ones through which a value can be CALLED.
#
# `_CALL_DATA_PARAMS` survives for one narrower job: the behavioural
# parametrization at the bottom of this file, which has to know which of
# `_call`'s parameters to hand a raising transform to.
_CALL_DATA_PARAMS = frozenset({
    "args", "op", "params", "require_target", "allow_implicit_target",
    "page_limit", "page_offset", "page_label", "paged_spill", "stem",
    "bridge_writes_output", "spawn_missing_named", "regex_hint_query",
    "regex_fallback_query", "offset_hint_identifier", "op_default_timeout",
})


def _declared_transform_params() -> frozenset[str]:
    """`_call`'s transform parameters: every parameter not declared data.

    Read from the `cli.py` the static properties parse, so the population they
    quantify over and the population the behavioural cells exercise are one
    declaration and not two. `test_the_malformed_result_rule_is_implemented...`
    pins the parsed signature to the imported one.

    Deliberately assertion-free: this runs at COLLECTION time to build the
    parametrization, and an assertion there is an `Interrupted: 1 error during
    collection` that runs no test in the session and reports the whole tree as
    neither passed nor failed. Every invariant it used to raise is owned by
    `test_the_declared_data_parameters_still_describe_call` below, which fails
    as one red cell.
    """
    call = _named("_call")
    if call is None:
        return frozenset()
    return frozenset({arg.arg for arg in _all_args(call)} - _CALL_DATA_PARAMS)


def test_the_declared_data_parameters_still_describe_call():
    """Fail-closed both ways -- a new parameter is a transform whatever it is
    called and however it is annotated, and a data parameter that leaves the
    signature stale-fails HERE rather than quietly shrinking the population."""
    call = _named("_call")
    assert call is not None, "_call is gone; this whole property is unchecked"
    stale = sorted(_CALL_DATA_PARAMS - {arg.arg for arg in _all_args(call)})
    assert not stale, (
        f"_CALL_DATA_PARAMS declares parameters _call no longer has: {stale}; "
        "a stale entry silently removes a real parameter from the population"
    )
    params = _declared_transform_params()
    assert len(params) >= 6, f"_call's transform parameters vanished: {sorted(params)}"
    orphans = sorted(set(_TRANSFORM_ARRANGEMENTS) - params)
    assert not orphans, (
        f"these arrangements name parameters _call no longer has: {orphans}; a "
        "stale entry makes the table look complete while a real parameter is "
        "unexercised"
    )
    # Reached at collection time through the parametrization, so it belongs
    # here and not in `_cli_functions`, where it would abort the session.
    assert len(_cli_functions()) > 50, (
        f"cli.py shrank to {len(_cli_functions())} functions; the static "
        "properties below quantify over a module that is no longer there")


@functools.lru_cache(maxsize=1)
def _cli_tree() -> ast.Module:
    """`cli.py`, parsed ONCE.

    Cached so every property quantifies over the same node objects: guardedness
    is keyed on a node's position, and two parses produce two sets of nodes.
    """
    return ast.parse(_CLI.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _cli_functions() -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """Every function in `cli.py`, as NODES rather than keyed by name.

    Round 8 keyed the population on the function NAME, and `cli.py` defines two
    `__init__`s and two `__call__`s -- so four of its functions were two, and a
    dispatch method (`__call__` on an argparse action) was silently outside a
    population whose whole claim is that it is total.
    """
    return tuple(node for node in ast.walk(_cli_tree())
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _named(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next((fn for fn in _cli_functions() if fn.name == name), None)


def _all_args(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Every parameter, INCLUDING `*args`/`**kwargs` -- a transform forwarded
    through a catch-all is still the same transform."""
    a = fn.args
    return [*a.posonlyargs, *a.args, *a.kwonlyargs,
            *(arg for arg in (a.vararg, a.kwarg) if arg is not None)]


def _bindings(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    """(targets, value, node, header) for every binding form in *fn*.

    Not just `ast.Assign`: round 7 invoked a transform through a loop variable
    and round 8 laundered one through a walrus, and an assignment-only walk
    taints neither. The NODE is yielded rather than its line, because where a
    binding SITS decides what it dominates and its line does not.

    `header` marks a binding made by a compound statement's own header -- a
    `for t in ...:` target or a `with ... as t:` name. Those bind before the
    statement's body rather than beside it, so they dominate everything inside
    it; scored at the statement's own position they would dominate nothing
    there, which is a false positive waiting for the first such shape.
    """
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            # An ANNOTATION is not a different binding form, and it was the last
            # door out of this population: `inner: Callable[[Any], str] | None =
            # text_renderer` moved a caller's transform to a name no property
            # tracked, and both static guards stayed green (round 11).
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None:
                yield targets, node.value, node, False
        elif isinstance(node, ast.NamedExpr):
            yield [node.target], node.value, node, False
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            yield [node.target], node.iter, node, True
        elif isinstance(node, ast.comprehension):
            yield [node.target], node.iter, node.target, False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    yield [item.optional_vars], item.context_expr, node, True
        elif isinstance(node, ast.MatchValue | ast.MatchSingleton | ast.MatchClass
                        | ast.MatchSequence | ast.MatchOr | ast.MatchAs | ast.MatchStar
                        | ast.MatchMapping):
            # A capture pattern binds whatever the subject held, and a bare
            # `case renderer:` (MatchAs with no sub-pattern) yielded nothing at
            # all. The subject is not reachable from the pattern node, so the
            # binding carries the pattern itself: conservative, since a captured
            # name is then only ever guarded by an explicit later rebinding.
            captured = [ast.Name(id=name, ctx=ast.Store()) for name in (
                *(p.name for p in ast.walk(node)
                  if isinstance(p, ast.MatchAs | ast.MatchStar) and p.name),
                *(p.rest for p in ast.walk(node)
                  if isinstance(p, ast.MatchMapping) and p.rest),
            )]
            if captured:
                yield captured, node.pattern if isinstance(node, ast.MatchAs) \
                    and node.pattern is not None else ast.Constant(value=None), node, False


def _carried(value: ast.expr, holders: set[str]) -> set[str]:
    """The holders an expression carries onward.

    A holder the expression merely CALLS is excluded: `x = t(result)` binds the
    transform's OUTPUT, not the transform.
    """
    called = {inner.func.id for inner in ast.walk(value)
              if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)}
    return ({inner.id for inner in ast.walk(value) if isinstance(inner, ast.Name)}
            & holders) - called


def _bound_names(targets: list[ast.expr]) -> set[str]:
    return {name.id for target in targets
            for name in ast.walk(target) if isinstance(name, ast.Name)}


def _block_paths(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[int, tuple]:
    """Every node of *fn*, mapped to the BLOCK PATH of the statement holding it.

    A path names a statement by the chain of (block field, index) steps taken
    from the function body to reach it, so `(("body", 3), ("orelse", 0))` is the
    first statement of the fourth statement's `else`. Unlike a line number it
    says which CONTROL-FLOW BLOCK a statement is in, which is what decides
    whether one statement certainly runs before another.
    """
    paths: dict[int, tuple] = {}

    def descend(body: list[ast.stmt], field: str, prefix: tuple) -> None:
        for index, stmt in enumerate(body):
            path = (*prefix, (field, index))
            # Outer to inner, so a nested statement overwrites the path it
            # inherited from its parent with its own: the innermost write wins.
            for node in ast.walk(stmt):
                paths[id(node)] = path
            for inner in ("body", "orelse", "finalbody"):
                block = getattr(stmt, inner, None)
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    descend(block, inner, path)
            for at, handler in enumerate(getattr(stmt, "handlers", [])):
                descend(handler.body, f"handler{at}", path)
            for at, case in enumerate(getattr(stmt, "cases", [])):
                descend(case.body, f"case{at}", path)

    descend(fn.body, "body", ())
    return paths


# Appended to a block path to mark a binding made by a compound statement's
# HEADER rather than beside it: it runs before the statement's body.
_HEADER = ("header", -1)


def _dominates(binding: tuple, use: tuple) -> bool:
    """Does a rebinding at *binding* certainly run before a use at *use*?

    Round 8 answered this with line order, and line order is not execution
    order: a boundary rebinding moved inside an `if` stays textually above every
    call site while running on only one path, and the calls on the other path
    read the caller's raw transform. So the rebinding must DOMINATE the use --
    sit in a block that encloses it, earlier in that same block. The empty path
    means "bound on entry", which dominates the whole function.
    """
    if not binding:
        return True
    if binding[-1] == _HEADER:
        # A `for`/`with` header binds before its BODY -- and only its body. A
        # `for ... else:` clause runs exactly when the iterable was EMPTY, so
        # the target was never rebound there and the name still holds whatever
        # it held before the loop. Scoring the header as dominating the whole
        # compound statement laundered a caller's raw transform through the
        # `else`, with all 243 properties green (round 11).
        scope = binding[:-1]
        return (len(use) > len(scope) and use[:len(scope)] == scope
                and use[len(scope)][0] == "body")
    depth = len(binding) - 1
    if len(use) <= depth or binding[:depth] != use[:depth]:
        return False
    field, index = binding[depth]
    use_field, use_index = use[depth]
    return field == use_field and index < use_index


# A function whose only purpose is to exercise the header rule's two sides.
# Parsed here rather than looked for in `cli.py`, because the point of the rule
# is to be right about shapes `cli.py` does not contain YET.
_HEADER_SHAPES = """
def sample(transform, result):
    for transform in _guarded_transform(transform, "why"):
        in_the_loop_body = transform(result)
    else:
        in_the_for_else = transform(result)
    with _guarded_transform(transform, "why") as bound:
        in_the_with_body = bound(result)
    after_the_statements = transform(result)
    return in_the_loop_body, in_the_for_else, in_the_with_body, after_the_statements
"""


def test_a_header_binding_dominates_its_body_and_not_its_for_else():
    """A `for ... else:` clause runs exactly when the iterable was EMPTY, so the
    loop target was never rebound and the name still holds what it held before
    the loop -- the caller's RAW transform. The header rule scored the binding
    as dominating the whole compound statement, so an invocation moved into the
    `else` was reported as guarded: a real `for`/`break`/`else` shape in
    `cli.py` invoked an unbound transform with 243 properties green. The `with`
    body is the sound half and must stay dominated, or the rule cries wolf
    instead of catching anything.
    """
    fn = ast.parse(_HEADER_SHAPES).body[0]
    where = _block_paths(fn)
    headers = {name: (*where[id(node)], _HEADER)
               for targets, _value, node, header in _bindings(fn) if header
               for name in _bound_names(targets)}
    assert set(headers) == {"transform", "bound"}, headers
    uses = {target.targets[0].id: where[id(target)]
            for target in ast.walk(fn) if isinstance(target, ast.Assign)
            and isinstance(target.targets[0], ast.Name)}

    assert _dominates(headers["transform"], uses["in_the_loop_body"])
    assert _dominates(headers["bound"], uses["in_the_with_body"])
    assert not _dominates(headers["transform"], uses["in_the_for_else"])
    assert not _dominates(headers["transform"], uses["after_the_statements"])
    assert not _dominates(headers["bound"], uses["after_the_statements"])


def test_every_name_cli_py_stores_is_a_name_the_population_binds():
    """The population's totality, stated over `cli.py` itself rather than over a
    list of binding FORMS -- which is what let an annotation out.

    `_bindings` enumerated forms, so each new syntactic form was a new door:
    round 7 a loop variable, round 8 a walrus, round 10 a module global, round
    11 an annotated assignment (`inner: Callable[...] | None = text_renderer`,
    two green properties and a green suite). A form cannot be missing from a
    list that is derived instead: every name the module STORES must be a name
    some binding yields, so the next form fails here on the commit that
    introduces it rather than in the round after.
    """
    unseen = {}
    for fn in _cli_functions():
        stored = {node.id for node in ast.walk(fn)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        stored |= {node.name for node in ast.walk(fn)
                   if isinstance(node, ast.MatchAs | ast.MatchStar) and node.name}
        stored |= {node.rest for node in ast.walk(fn)
                   if isinstance(node, ast.MatchMapping) and node.rest}
        # The three forms that bind a name WITHOUT an `ast.Name` store, listed
        # here so the claim in this test's name is literally true rather than
        # true of the nodes it happened to walk.
        stored |= {node.name for node in ast.walk(fn)
                   if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef
                                 | ast.ClassDef)} - {fn.name}
        stored |= {(alias.asname or alias.name).split(".")[0]
                   for node in ast.walk(fn)
                   if isinstance(node, ast.Import | ast.ImportFrom)
                   for alias in node.names}
        stored |= {handler.name for handler in ast.walk(fn)
                   if isinstance(handler, ast.ExceptHandler) and handler.name}
        bound = {name for targets, *_ in _bindings(fn) for name in _bound_names(targets)}
        # The carve-outs, each with a reason that is a fact about the value the
        # name receives rather than about the syntax that binds it:
        #   * `except ... as name` receives an EXCEPTION, never a transform;
        #   * a nested `def`/`class` receives an object `cli.py` defines, and its
        #     own body is a member of `_cli_functions()` scanned in its own right;
        #   * an `import` receives a module.
        # Each can only make a later use look UNguarded, never guarded.
        bound |= {handler.name for handler in ast.walk(fn)
                  if isinstance(handler, ast.ExceptHandler) and handler.name}
        bound |= {node.name for node in ast.walk(fn)
                  if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef
                                | ast.ClassDef)}
        bound |= {(alias.asname or alias.name).split(".")[0]
                  for node in ast.walk(fn)
                  if isinstance(node, ast.Import | ast.ImportFrom)
                  for alias in node.names}
        if stored - bound:
            unseen[fn.name] = sorted(stored - bound)
    assert not unseen, (
        "`cli.py` stores these names through a form `_bindings` does not yield, "
        "so a caller-supplied transform parked on one is outside the population "
        f"every guard below quantifies over: {unseen}"
    )


def _stores_holder(value: ast.expr, holders: set[str]) -> bool:
    """Can *value* evaluate TO one of *holders* -- not merely mention one?

    `_carried` is deliberately loose, because a transform tucked inside a
    wrapper is still reachable. A SLOT is a stricter question: parking
    `str(exc)` under `"error"` does not make `.error` a way to call `exc`,
    and treating it as one flagged two `parser.error(...)` calls -- exactly the
    crying-wolf that got the attribute step dropped in the first place.
    """
    if isinstance(value, ast.Name):
        return value.id in holders
    if isinstance(value, ast.Starred):
        return _stores_holder(value.value, holders)
    if isinstance(value, ast.IfExp):
        return (_stores_holder(value.body, holders)
                or _stores_holder(value.orelse, holders))
    if isinstance(value, ast.BoolOp):
        return any(_stores_holder(operand, holders) for operand in value.values)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return any(_stores_holder(element, holders) for element in value.elts)
    if isinstance(value, ast.Dict):
        return any(element is not None and _stores_holder(element, holders)
                   for element in value.values)
    return False


# --- What the SLOT rules below do and do not answer, stated in full ---
#
# The three rules that follow -- `_slot_names` (what a holder is stored under),
# `_slot_dispatch_sites` (what is invoked through such a name) and
# `_slot_traffic` (which statements handle an exempt slot) -- recognise LITERAL
# stores, LITERAL dispatch sites and LITERAL traffic. A slot name has to appear
# in the source as a string constant, an attribute or a keyword; a dispatch has
# to be an attribute access or a constant-string subscript; and a statement is
# traffic in an exempt slot if it spells that slot out.
#
# These forms are NOT recognised, and each was reached by an injection in
# review rather than found in `cli.py`:
#
#   1. a store whose slot name is COMPUTED -- `setattr(obj, name, t)`,
#      `setattr(obj, "han" + "dler", t)`, `setattr(obj, f"hand{x}ler", t)`;
#   2. a dispatch whose subscript key is computed -- `ns.inner[key](r)`,
#      `ns.inner[f"par{y}ked"](r)`;
#   3. a dispatch assembled by a library -- `operator.attrgetter("parked")(ns)(r)`;
#   4. a COMPREHENSION, in either role: as a callee display
#      (`[t for _ in (0,)][0](r)`, `next(t for _ in (0,))(r)`) or as a store
#      (`ns.inner.update({k: t for k in ("parked",)})`);
#   5. an immediately-applied lambda -- `(lambda: t)()(r)`;
#   6. a store in a NESTED class body -- `class _Reg: handler = t`;
#   7. a subscript store under a non-constant key -- `ns.inner[key] = t`.
#
# There is no sound static answer here, and the reason is not that this sweep is
# young. `cli.py` is PARSED and never run by these properties, so a name has no
# runtime object behind it; deciding whether an arbitrary Python expression
# evaluates to a particular function is the halting problem wearing a hat. Each
# round of review that closed one of these forms was answered by the next form
# out -- rounds 12, 13 and 14 are three consecutive instances -- because an
# adversary writing the escape always has one more syntactic form than the
# recogniser has cases. Nine rounds bought nine forms and left the tenth open.
# So the escapes are WRITTEN DOWN, exactly as the doc number sweep's two escapes
# are written down in tests/test_agent_docs.py, instead of being implied away.
#
# What is actually load-bearing does not depend on any of this:
#
#   * Every transform `cli.py` receives is bound to the malformed-result rule by
#     the function that RECEIVES it, before anything can invoke it -- so the
#     shape of the later call site is irrelevant to whether it is guarded. It is
#     one RULE applied at every door, NOT one door: round 15 measured the
#     earlier "ONE guarded boundary" phrasing false (the module carries ten
#     `_guarded_transform` bindings across five receiving functions), and a
#     reader auditing this disclosure against the source has to find what is
#     actually there. The population that establishes it is not a list of those
#     doors either -- `test_no_bridge_result_transform_leaves_cli_py_unguarded`
#     quantifies over every parameter of every function in the module, so a new
#     door that forgets to bind is a finding by existing. The static properties
#     police that; they are not the guarantee.
#   * `test_every_transform_parameter_of_call_survives_a_malformed_result` RUNS
#     `main()` once per transform parameter with a transform that raises on a
#     malformed result, and asserts the documented BridgeError and exit code
#     rather than a traceback. That cell is behavioural: it cannot be satisfied
#     by a syntactic form, and it is what proved one of the nine boundary
#     bindings inert (an unmeasured `formatters.py` guard, routed to #722).
#
# An injection that defeats a BEHAVIOURAL cell, or that bypasses a receiving
# function's binding at RUNTIME, is a real finding. An injection that merely
# finds form eight of the static sweep is this paragraph.


def _slot_names(fn: ast.FunctionDef | ast.AsyncFunctionDef, holders: set[str]) -> set[str]:
    """The attribute/key names a caller-supplied transform is STORED under in *fn*.

    `_dispatch_roots` deliberately does not follow every attribute step: with a
    total population `p.method(x)` would flag every `parser.add_argument` in the
    module, and a guard that cries wolf gets loosened instead of fixed. But a
    transform parked on an object and dispatched as `holder.render(result)`
    escaped the property entirely. A holder stored under a name can be invoked
    through that name, so the names holders are stored under are exactly the
    attribute steps worth following -- DERIVED from the module, not listed.

    A slot name does not have to be a keyword or a dict key. `setattr(obj,
    "render", t)` names it with a positional string, and so do
    `d.setdefault("render", t)` and any registry helper. So the rule is not a
    list of storing FORMS: any call that is handed a holder puts every string
    constant in that same call into the slot set. Over-matching is harmless --
    a slot only ever matters when something is dispatched through an attribute
    of that exact name.
    """
    slots: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            slots |= {keyword.arg for keyword in node.keywords
                      if keyword.arg and _stores_holder(keyword.value, holders)}
            if (any(_stores_holder(argument, holders) for argument in node.args)
                    or any(_stores_holder(keyword.value, holders)
                           for keyword in node.keywords)):
                slots |= {argument.value for argument in node.args
                          if isinstance(argument, ast.Constant)
                          and isinstance(argument.value, str)}
        elif isinstance(node, ast.Dict):
            slots |= {key.value for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and isinstance(key.value, str)
                      if _stores_holder(value, holders)}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and _stores_holder(node.value, holders):
                slots |= {target.attr for target in targets
                          if isinstance(target, ast.Attribute)}
                # `d["render"] = t` names a slot exactly as `d.render = t` does.
                # Missing it is how `slots = globals(); slots["parked"] = t`
                # paired with `globals()["parked"](result)` escaped: the two
                # sides name the same container with different expressions, so
                # no container-rooted rule can connect them -- the SLOT can.
                slots |= {target.slice.value for target in targets
                          if isinstance(target, ast.Subscript)
                          and isinstance(target.slice, ast.Constant)
                          and isinstance(target.slice.value, str)}
    return slots


def _slot_dispatch_sites(fn, slots: frozenset[str], guarded_slots: frozenset[str]):
    """Calls in *fn* dispatched through a slot name, whatever holds the slot.

    The container-rooted rule needs the container to be in the population, and
    a module namespace fetched by `globals()` in one function and by a local
    alias in another is the same container under two expressions that no such
    rule can relate. A SLOT is the relation: a name a holder was stored under
    can be invoked through that name, so an invocation through an
    unguarded-slot name is reported wherever it appears and whatever it hangs
    off. Over-matching costs nothing -- a slot only exists because something
    stored a holder under it.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif (isinstance(node.func, ast.Subscript)
              and isinstance(node.func.slice, ast.Constant)
              and isinstance(node.func.slice.value, str)):
            name = node.func.slice.value
        if name in slots and name not in guarded_slots:
            yield node, name


def _guarded_slots(transforms: "_Transforms") -> frozenset[str]:
    """Slots every store of which parks a value already bound to the rule.

    A slot is only as safe as its worst store, so one unguarded store makes
    every dispatch through that name a finding.
    """
    unguarded: set[str] = set()
    for fn in _cli_functions():
        holders = transforms.holders[id(fn)]
        bound_at = transforms.guarded[id(fn)]
        where = transforms.paths[id(fn)]
        for targets, value, node, header in _bindings(fn):
            if not _stores_holder(value, holders):
                continue
            names = {target.attr for target in targets
                     if isinstance(target, ast.Attribute)}
            names |= {target.slice.value for target in targets
                      if isinstance(target, ast.Subscript)
                      and isinstance(target.slice, ast.Constant)
                      and isinstance(target.slice.value, str)}
            if not names:
                continue
            use = where.get(id(node), ())
            stored = _carried(value, holders)
            if not stored or not all(
                    any(_dominates(binding, use) for binding in bound_at.get(name, ()))
                    for name in stored):
                unguarded |= names
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            use = where.get(id(node), ())
            for keyword in node.keywords:
                if keyword.arg and _stores_holder(keyword.value, holders):
                    stored = _carried(keyword.value, holders)
                    if not stored or not all(
                            any(_dominates(binding, use)
                                for binding in bound_at.get(name, ()))
                            for name in stored):
                        unguarded.add(keyword.arg)
            # ...and the POSITIONAL string, which `_slot_names` already reads as
            # naming a slot: `setattr(obj, "render", t)`. The two halves of one
            # rule have to read the same store forms -- the slot was created and
            # then classed GUARDED because no store of it was ever seen, which
            # silenced every dispatch through that name (round 14).
            constants = {argument.value for argument in node.args
                         if isinstance(argument, ast.Constant)
                         and isinstance(argument.value, str)}
            for carried in (*node.args, *(keyword.value for keyword in node.keywords)):
                if not constants or not _stores_holder(carried, holders):
                    continue
                stored = _carried(carried, holders)
                if not stored or not all(
                        any(_dominates(binding, use)
                            for binding in bound_at.get(name, ()))
                        for name in stored):
                    unguarded |= constants
        for node in ast.walk(fn):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                            and _stores_holder(value, holders)):
                        stored = _carried(value, holders)
                        use = where.get(id(node), ())
                        if not stored or not all(
                                any(_dominates(binding, use)
                                    for binding in bound_at.get(name, ()))
                                for name in stored):
                            unguarded.add(key.value)
    return frozenset(transforms.slots) - frozenset(unguarded)


def _module_scope_names() -> frozenset[str]:
    """Every name in `cli.py`'s module namespace, plus every name a function
    declares `global`.

    A module-scope name is the third door into the population, after the
    parameter and the call argument: a transform assigned to one is readable
    from every function in the file. Round 9's population had a parameter and a
    call argument and not this, so parking a caller-supplied renderer in a
    module global and dispatching it from a helper left every property green.

    The first cut of THIS function enumerated the binding kinds that reach
    module scope -- `Assign`, `AnnAssign`, `global` -- and a `class` at module
    scope is none of them, so `_RenderSlot.render = text_renderer` in one
    function and `_RenderSlot.render(result)` in another escaped the population
    entirely and `main()` raised a raw `ValueError` (round 12). That is the
    third time a STORAGE LOCATION rather than a call shape defeated this sweep,
    after the module global and the dict value, so the kinds are no longer
    enumerated: the module namespace is asked directly, which is total over
    every storage form the language has -- `class`, `def`, `import`, assignment,
    annotation, or one added to Python next year. `global` names are unioned in
    because a name a function creates at runtime need not exist at import.
    """
    return frozenset(vars(bn.cli)) | {
        name for node in ast.walk(_cli_tree()) if isinstance(node, ast.Global)
        for name in node.names
    }


class _Transforms(NamedTuple):
    holders: dict[int, set[str]]                # id(fn) -> names that may hold a transform
    guarded: dict[int, dict[str, set[tuple]]]   # id(fn) -> name -> paths binding it to the rule
    paths: dict[int, dict[int, tuple]]          # id(fn) -> id(node) -> block path
    slots: frozenset[str]                       # names a holder is stored under


def _transform_holders() -> _Transforms:
    """Per `cli.py` function: the names that may hold a caller-supplied
    transform, and WHERE each is bound to the malformed-result rule.

    The population is every parameter of every function in the module -- no
    declaration, no doorway -- and is propagated to a fixpoint two ways, because
    each was escaped alone:

    * CONTAINMENT over every binding form, so `b = a`, `b = (a, x)`,
      `b = {"k": a}`, `for b in (a,)`, `[b for b in ...]` and `with ... as b`
      all keep the value in the population and `slots["k"](result)` cannot
      launder it out.
    * CALL SITES inside `cli.py`, so an argument taints the callee's
      corresponding parameter: a transform forwarded to a helper is still that
      transform there, under whatever name the helper gave it.
    * MODULE SCOPE, so a transform parked in a module-level name -- the third
      door, after the parameter and the call argument -- is a holder in every
      function that could read it. Nothing orders an assignment in one function
      against a read in another, so such a name is provably bound NOWHERE
      outside the function that bound it: parking a transform in a global and
      dispatching it from a helper is reported, which is what it should be.

    Guardedness does NOT propagate across a call site -- only the population
    does. A transform bound at one caller's boundary and forwarded to a helper
    is reported again IN the helper unless the helper re-binds, because a second
    caller handing that same helper a raw transform would otherwise be laundered
    by the first (round 10). The cost is over-strictness, which `cli.py` pays by
    binding at each boundary; the cost of the alternative was a real escape.
    Over-approximating the POPULATION likewise costs little: a data parameter is
    only ever reported if the module CALLS it.
    """
    functions = _cli_functions()
    by_name: dict[str, list] = {}
    for fn in functions:
        by_name.setdefault(fn.name, []).append(fn)
    module_names = _module_scope_names()
    paths = {id(fn): _block_paths(fn) for fn in functions}
    holders: dict[int, set[str]] = {id(fn): {arg.arg for arg in _all_args(fn)}
                                    for fn in functions}
    guarded: dict[int, dict[str, set[tuple]]] = {id(fn): {} for fn in functions}
    slots: set[str] = set()
    assert _named("_call") is not None, "_call is gone; this whole property is unchecked"

    def share(names: set[str]) -> None:
        """A module-scope name holds its value for the whole module."""
        for shared in names & module_names:
            for other in functions:
                holders[id(other)].add(shared)

    def bound_by(fn, names: set[str], use: tuple) -> bool:
        return all(any(_dominates(binding, use) for binding in guarded[id(fn)].get(name, ()))
                   for name in names)

    def bind(fn, names: set[str], at: tuple) -> None:
        for name in names:
            guarded[id(fn)].setdefault(name, set()).add(at)

    def snapshot() -> tuple:
        return (tuple(sorted((key, tuple(sorted(names))) for key, names in holders.items())),
                tuple(sorted((key, tuple(sorted((name, len(at)) for name, at in bound.items())))
                             for key, bound in guarded.items())),
                tuple(sorted(slots)))

    converged = False
    for _ in range(len(functions) + 8):
        before = snapshot()
        for fn in functions:
            where = paths[id(fn)]
            for targets, value, node, header in _bindings(fn):
                names = _bound_names(targets)
                at = where.get(id(node), ())
                # The `use` position of the binding's own value is the
                # statement it sits in; what it BINDS covers the body.
                use_at, at = at, ((*at, _HEADER) if header else at)
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == _BOUNDARY):
                    holders[id(fn)] |= names
                    bind(fn, names, at)
                    share(names)
                    continue
                carried = _carried(value, holders[id(fn)])
                if carried:
                    holders[id(fn)] |= names
                    if bound_by(fn, carried, use_at):
                        bind(fn, names, at)
                    share(names)
            slots |= _slot_names(fn, holders[id(fn)])
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                at = where.get(id(node), ())
                # A call HANDED a holder can PARK it in another of its
                # arguments: `setattr(obj, "render", t)` puts the transform on
                # `obj` without any binding form this walk would see, and the
                # slot name alone is useless if `obj` is not in the population.
                # A store NAMES the slot, so the string constant is what tells
                # this apart from an ordinary call that merely takes a
                # callable -- `f(t, dict)` parks nothing.
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
                if (any(_stores_holder(argument, holders[id(fn)]) for argument in arguments)
                        and any(isinstance(argument, ast.Constant)
                                and isinstance(argument.value, str)
                                for argument in arguments)):
                    holders[id(fn)] |= {argument.id for argument in node.args
                                        if isinstance(argument, ast.Name)}
                    if isinstance(node.func, ast.Attribute):
                        holders[id(fn)] |= _dispatch_roots(node.func.value, frozenset(slots))
                for name in _callee_roots(node, frozenset(slots)):
                    for callee in by_name.get(name, ()):
                        pairs = [*zip([arg.arg for arg in _all_args(callee)], node.args)]
                        pairs += [(keyword.arg, keyword.value)
                                  for keyword in node.keywords if keyword.arg]
                        for param, expr in pairs:
                            carried = _carried(expr, holders[id(fn)])
                            if not carried:
                                continue
                            holders[id(callee)].add(param)
        if before == snapshot():
            converged = True
            break
    assert converged, "the transform population did not converge; widen the bound"
    return _Transforms(holders, guarded, paths, frozenset(slots))


def _callee_names(node: ast.Call) -> set[str]:
    """Every name that identifies what a call invokes.

    `f(x)`, `obj.f(x)`, `slots["f"](x)`, `registry.slots["f"](x)`: the callee is
    reached by a chain of attributes, subscripts and calls, and each step roots
    in a name. Used where over-matching is harmless -- asking whether a call
    stays inside `cli.py`, and whether the boundary delegates to the rule.
    """
    return ({name.id for name in ast.walk(node.func) if isinstance(name, ast.Name)}
            | {attr.attr for attr in ast.walk(node.func) if isinstance(attr, ast.Attribute)})


def _dispatch_roots(expr: ast.expr, slots: frozenset[str]) -> set[str]:
    """The names a callee expression DISPATCHES THROUGH.

    `f(x)`, `slots["f"](x)`, `factory()(x)` and `handlers[i][j](x)` root in one
    name. `(tr := t)(x)` dispatches through both the walrus target and its
    value. An attribute step is followed only when the attribute is a name some
    holder is STORED under, so `holder.render(x)` is a dispatch and
    `parser.add_argument(x)` is not.
    """
    if isinstance(expr, ast.Name):
        return {expr.id}
    if isinstance(expr, (ast.Subscript, ast.Starred)):
        return _dispatch_roots(expr.value, slots)
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        # `[t][0](result)` and `(t,)[0](result)` root in the DISPLAY, which used
        # to root in nothing: a dict LOOKUP on a name was covered and a
        # container built inline and subscripted was invisible (round 13).
        return set().union(*(_dispatch_roots(element, slots)
                             for element in expr.elts), set())
    if isinstance(expr, ast.Dict):
        return set().union(*(_dispatch_roots(value, slots)
                             for value in expr.values if value is not None), set())
    if isinstance(expr, ast.Call):
        # `factory()(x)` dispatches through whatever the factory returns, and a
        # factory handed a holder can return it wrapped:
        # `functools.partial(text_renderer)(result)` rooted in `partial` alone,
        # so the holder in the ARGUMENTS was invisible and the call was not a
        # dispatch at all (round 12). Every argument is a root too.
        return _dispatch_roots(expr.func, slots).union(*(
            _dispatch_roots(argument, slots)
            for argument in (*expr.args, *(kw.value for kw in expr.keywords))
        ), set())
    if isinstance(expr, ast.NamedExpr):
        target = {expr.target.id} if isinstance(expr.target, ast.Name) else set()
        return target | _dispatch_roots(expr.value, slots)
    if isinstance(expr, ast.Attribute) and expr.attr in slots:
        return _dispatch_roots(expr.value, slots)
    if isinstance(expr, ast.BoolOp):
        # `(t or fallback)(x)` dispatches through EITHER operand, and
        # `_stores_holder` already reads this shape as carrying a holder -- so
        # reading it as a value but not as a callee was a hole between two
        # halves of the same rule.
        return set().union(*(_dispatch_roots(value, slots) for value in expr.values))
    if isinstance(expr, ast.IfExp):
        return _dispatch_roots(expr.body, slots) | _dispatch_roots(expr.orelse, slots)
    return set()


def _callee_roots(node: ast.Call, slots: frozenset[str] = frozenset()) -> set[str]:
    return _dispatch_roots(node.func, slots)


def _callee(node: ast.Call) -> str:
    return ast.unparse(node.func)


def test_the_malformed_result_rule_is_implemented_where_this_property_says_it_is():
    """Both halves of the mechanism, checked -- otherwise the two properties
    below could be satisfied by renaming the rule out of existence."""
    rule = _named(_GUARD)
    assert rule is not None, f"{_GUARD} is gone; this whole property is unchecked"
    handlers = [handler for node in ast.walk(rule) if isinstance(node, ast.Try)
                for handler in node.handlers]
    assert any(isinstance(handler.type, ast.Name)
               and handler.type.id == "_MALFORMED_RESULT_ERRORS" for handler in handlers), (
        f"{_GUARD} no longer catches _MALFORMED_RESULT_ERRORS, so binding a "
        "transform to it guards nothing")
    boundary = _named(_BOUNDARY)
    assert boundary is not None, f"{_BOUNDARY} is gone; nothing binds a transform to the rule"
    assert any(_callee_names(node) & {_GUARD} for node in ast.walk(boundary)
               if isinstance(node, ast.Call)), (
        f"{_BOUNDARY} must delegate to {_GUARD}; otherwise it returns an "
        "unguarded transform under a reassuring name")
    call = _named("_call")
    assert call is not None, "_call is gone; this whole property is unchecked"
    assert ({arg.arg for arg in _all_args(call)}
            == set(inspect.signature(bn.cli._call).parameters)), (
        "the cli.py this module PARSES and the bn.cli it IMPORTS disagree about "
        "_call's parameters, so the static population and the behavioural "
        "population are two different lists"
    )


# The population is every parameter of every function, so `main`'s own `argv`
# is a holder and the parsed namespace built from it is one by containment --
# which makes the command handler `main` reads off that namespace a holder too.
# It is not a bridge-result transform: it takes the namespace, returns an exit
# code, and raises `BridgeError`, which `main` already converts. Binding it to
# the malformed-result rule would be actively wrong -- a genuine `TypeError`
# from a handler would be reported as "the bridge response was malformed".
#
# So it is an EXEMPT SITE, named, with a reason that is executable and not
# prose: nothing in `cli.py` ever STORES a caller-supplied value under this
# slot, which is why the name cannot carry one. The site is line-number-free so
# an unrelated edit does not invalidate it, both halves stale-fail (the site
# must still be reported, and the slot must still receive nothing), and it was
# reported only once the annotated-assignment door closed -- before that the
# annotation hid the binding entirely.
# site -> (slot, the sites `cli.py` may store a holder under that slot).
# The store set is part of the reason, not an afterthought: `@_command` parks the
# function it decorates under `handler`, and by a total population a decorator's
# own parameter is a holder, so "nothing is stored here" is simply false. What is
# true is that the slot is written at ONE site, the command registry, and it
# stale-fails in both directions -- a new store reds, and the registry store
# disappearing reds too, because then the exemption is describing a module that
# has moved on.
_NOT_A_RESULT_TRANSFORM = {
    "handler() in main()": ("handler", {
        # The command registry parks the function `@_command` decorates...
        "handler @ Expr in decorator()": 1,
        # ...and the parser build hands each registered spec's own handler to
        # argparse, which is the registry again, one layer out.
        "handler @ Expr in _build_from_commands()": 1,
        # ...and `main` READS it back off the parsed namespace. A read is in the
        # coarse population too, because telling a read from a store is exactly
        # the kind-specific reasoning that kept failing; declaring it is cheaper
        # and it stale-fails the same way.
        "handler @ AnnAssign in main()": 1,
    }),
}


@functools.lru_cache(maxsize=1)
def _rule_exempt_calls() -> frozenset[int]:
    """The node ids of the calls the malformed-result rule performs UNDER it.

    Exactly the calls inside a `try` of `_apply_result_transform` whose handler
    catches `_MALFORMED_RESULT_ERRORS` -- the one place a transform is invoked
    raw because that invocation IS the rule. Every other call in the rule
    function is swept like any other.

    Fail-closed by construction: if the rule loses its guarded `try` (or the
    rule is renamed out of existence) this set is empty and the rule's own
    `transform(result)` becomes a reported unguarded site, rather than the
    exemption silently widening to the whole module.
    """
    rule = _named(_GUARD)
    if rule is None:
        return frozenset()
    return frozenset(
        id(inner)
        for node in ast.walk(rule) if isinstance(node, ast.Try)
        if any(isinstance(handler.type, ast.Name)
               and handler.type.id == "_MALFORMED_RESULT_ERRORS"
               for handler in node.handlers)
        for statement in node.body
        for inner in ast.walk(statement) if isinstance(inner, ast.Call)
    )


def _dispatch_sites() -> list[str]:
    """Every `cli.py` call that dispatches a holder the boundary does not
    dominate, as `<callee>() in <function>()` plus its line."""
    transforms = _transform_holders()
    guarded_slots = _guarded_slots(transforms)
    sites = []
    exempt = _rule_exempt_calls()
    for fn in _cli_functions():
        where, bound_at = transforms.paths[id(fn)], transforms.guarded[id(fn)]
        sites += [
            f"{_callee(node)}() in {fn.name}()|cli.py:{node.lineno}"
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and id(node) not in exempt
            for use in [where.get(id(node), ())]
            for dispatched in [_callee_roots(node, transforms.slots)
                               & transforms.holders[id(fn)]]
            if any(not any(_dominates(binding, use) for binding in bound_at.get(name, ()))
                   for name in dispatched)
        ]
        # ...and the same question asked of the SLOT rather than the container,
        # which is what relates two expressions naming one namespace.
        sites += [f"{_callee(node)}() in {fn.name}()|cli.py:{node.lineno}"
                  for node, _slot in _slot_dispatch_sites(
                      fn, transforms.slots, guarded_slots)
                  if id(node) not in exempt]
    return sorted(set(sites))


def _own_statements(fn) -> list[ast.stmt]:
    """Every statement *fn* itself contains, not those of a function nested in
    it -- `_cli_functions()` lists a nested definition separately, so walking
    the outer one attributed the same statement to both."""
    owned: list[ast.stmt] = []
    stack = list(fn.body)
    while stack:
        statement = stack.pop()
        owned.append(statement)
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            stack += getattr(statement, field, []) or []
        for handler in getattr(statement, "handlers", []):
            stack += handler.body
        for case in getattr(statement, "cases", []):
            stack += case.body
    return owned


def _own_nodes(statement: ast.stmt):
    """*statement* and the expressions that belong to IT, not to a statement
    nested inside it -- `ast.walk` on a `def` drags in the whole body."""
    yield statement
    stack = [child for child in ast.iter_child_nodes(statement)
             if not isinstance(child, ast.stmt)]
    while stack:
        node = stack.pop()
        yield node
        stack += [child for child in ast.iter_child_nodes(node)
                  if not isinstance(child, ast.stmt)]


def _slot_traffic(slot: str) -> collections.Counter:
    """Every STATEMENT of `cli.py` that both names *slot* and handles a holder.

    Four storage kinds have defeated this sweep -- a module global, a dict
    value, a class attribute, a subscript store -- and enumerating the kinds is
    losing to whoever writes the fifth. Identity would end the class, but this
    property is static: `cli.py` is parsed and never run, so a name has no
    runtime object to follow. So the POPULATION is deliberately coarse and
    kind-blind -- a statement that mentions the slot (as a string constant or an
    attribute) and stores a holder anywhere inside it -- and the precision comes
    from the declared set the exemption carries. A fifth storage kind lands in
    the coarse population and reds for want of a declaration, instead of
    vanishing.

    Counted, not collected into a set: two statements whose descriptors render
    identically are two statements, and a set silently collapsed the second
    onto the first (round 13).
    """
    transforms = _transform_holders()
    traffic: collections.Counter = collections.Counter()
    for fn in _cli_functions():
        holders = transforms.holders[id(fn)]
        for statement in _own_statements(fn):
            inner = list(_own_nodes(statement))
            names = any(
                (isinstance(node, ast.Constant) and node.value == slot)
                or (isinstance(node, ast.Attribute) and node.attr == slot)
                or (isinstance(node, ast.keyword) and node.arg == slot)
                for node in inner)
            if not names:
                continue
            if any(_stores_holder(node, holders) for node in inner
                   if isinstance(node, ast.expr)):
                traffic[f"{slot} @ {type(statement).__name__} in {fn.name}()"] += 1
    return traffic


def test_the_exempt_dispatch_site_is_still_reported_and_still_takes_no_holder():
    """The exemption's three halves, each able to fail on its own.

    IDENTITY: the exemption must match EXACTLY ONE reported site. Keyed on the
    rendered callee plus the enclosing function name, it was matched by PATTERN,
    so a second `def main(handler, result): return handler(result)` -- or one
    more `handler(...)` call anywhere in the real `main` -- inherited it
    silently with every property green. An exemption is a hole you promised not
    to look through, and the promise has to be unique as well as executable.
    NAME: the exempt name may not be BOUND to a holder inside the function that
    dispatches it, which is how a caller-supplied transform was renamed onto it.
    SLOT: nothing in `cli.py` may store a holder under the exempted slot, in any
    binding form -- the claim that makes the site safe in the first place.
    """
    reported = [site.split("|", 1)[0] for site in _dispatch_sites()]
    wrong = {site: reported.count(site) for site in _NOT_A_RESULT_TRANSFORM
             if reported.count(site) != 1}
    assert not wrong, (
        "each exemption must name exactly ONE reported dispatch site: a count "
        "of 0 means the exemption is dead and outliving the call it excuses, "
        "and a count above 1 means a second site is inheriting it under the "
        f"same rendered name: {wrong}"
    )
    transforms = _transform_holders()
    renamed = []
    for site, (slot, _allowed) in _NOT_A_RESULT_TRANSFORM.items():
        owner = site.rsplit(" in ", 1)[1].removesuffix("()")
        for fn in _cli_functions():
            if fn.name != owner:
                continue
            holders = transforms.holders[id(fn)]
            renamed += [
                f"{slot} <- {ast.unparse(value)} in {fn.name}() at cli.py:{node.lineno}"
                for targets, value, node, _header in _bindings(fn)
                if slot in _bound_names(targets) and _stores_holder(value, holders)
            ]
    assert not renamed, (
        "the exempt name is bound to a caller-supplied value inside the very "
        "function that dispatches it, so the exempt site is now a real "
        f"unguarded dispatch wearing the exemption's name: {renamed}"
    )
    for _site, (slot, allowed) in _NOT_A_RESULT_TRANSFORM.items():
        found = _slot_traffic(slot)
        assert found == collections.Counter(allowed), (
            f"the statements that name the exempt slot {slot!r} and handle a "
            "caller-supplied value are not the ones the exemption's reason "
            "names, so the reason no longer describes this module (a COUNT "
            "change means a second statement of a kind already declared): "
            f"found {dict(found)}, declared {dict(allowed)}"
        )


def test_no_bridge_result_transform_is_invoked_before_it_is_bound_to_the_rule():
    """Property: in `cli.py`, a transform is CALLED only where the boundary
    rebinding DOMINATES the call -- whatever the dispatch shape (a name, a dict
    lookup, an attribute a transform was parked on, a loop variable, a walrus, a
    returned callable), in whatever function it was forwarded to, because it is
    the OBJECT that is guarded and not the site.

    "After the rebinding" used to mean "on a later line", and a rebinding moved
    into an `if` is on an earlier line than every call while running on only one
    path. Dominance is the question line order was standing in for.
    """
    unguarded = [site.replace("|", " at ") for site in _dispatch_sites()
                 if site.split("|", 1)[0] not in _NOT_A_RESULT_TRANSFORM]
    assert not unguarded, (
        "these invoke a bridge-result transform on a path the boundary "
        "rebinding does not dominate, so a malformed result escapes the rule "
        f"there: {sorted(unguarded)}"
    )


def _result_key_reads(tree: ast.Module) -> tuple[list[ast.AST], set[str]]:
    """Every occurrence of the reply's result key in `cli.py`, and the names
    bound to it.

    An occurrence is a `"result"` constant OR a load of a name bound to one:
    the population used to be constants alone, so the exemption's own name read
    the key with no `"result"` token anywhere in the expression.
    """
    aliases: set[str] = set()
    for _ in range(8):
        before = set(aliases)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if ((isinstance(value, ast.Constant) and value.value == "result")
                    or (isinstance(value, ast.Name) and value.id in aliases)):
                aliases |= {target.id for target in node.targets
                            if isinstance(target, ast.Name)}
        if before == aliases:
            break
    occurrences = [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value == "result")
        or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id in aliases)
    ]
    return occurrences, aliases


def test_no_bridge_reply_is_indexed_for_its_result_outside_the_unwrap_helper():
    """The same lesson one level up: `_unwrap_result` was applied to the reads
    its author was looking at, and the fan-out planner's fourth read still
    indexed the reply raw, so `{"ok": true}` with no `result` left `main()` as a
    bare `KeyError`.

    Three cuts were escaped. Matching one read SHAPE let `response.get("result")`
    past. Quantifying over every `"result"` CONSTANT but exempting a syntactic
    class -- any dict-literal key -- let `response.get(*{"result": None})` wear
    the syntax. Exempting ONE NAME let `response.get(_RESULT_ROW_KEY)` read the
    key with no constant to find.

    A name can be borrowed, so the exemption is not a name: it is the one WRITE
    that pairs this key with the helper's own return value, outside any `*`
    unpacking. Everything else that mentions the key -- a subscript, a `.get`, a
    `.pop`, an `in` test, a starred dict, an alias of an alias -- is a read, and
    reads go through the helper.
    """
    tree = _cli_tree()
    unwrap = _named("_unwrap_result")
    assert unwrap is not None, "_unwrap_result is gone; nothing checks the envelope"
    declared = [node for node in tree.body
                if isinstance(node, ast.Assign)
                if any(isinstance(target, ast.Name) and target.id == _RESULT_ROW_KEY_NAME
                       for target in node.targets)]
    assert (len(declared) == 1 and isinstance(declared[0].value, ast.Constant)
            and declared[0].value.value == "result"), (
        f"cli.py must declare {_RESULT_ROW_KEY_NAME} exactly once, at module "
        "scope, as the literal 'result'; it names the one write this property exempts"
    )
    occurrences, aliases = _result_key_reads(tree)
    assert aliases == {_RESULT_ROW_KEY_NAME}, (
        "cli.py binds the reply's result key to more names than the one this "
        f"property knows about, and each is a read it cannot see: {sorted(aliases)}"
    )
    starred = {id(inner) for node in ast.walk(tree) if isinstance(node, ast.Starred)
               for inner in ast.walk(node)}
    exempt = {
        id(key)
        for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if key is not None and id(key) not in starred
        # The write is exempt because its VALUE came from the helper: code that
        # claims this exemption has already put the reply through the envelope
        # check, which is the whole invariant.
        if isinstance(value, ast.Call) and _callee_names(value) & {"_unwrap_result"}
        if id(key) in {id(occurrence) for occurrence in occurrences}
    }
    raw = sorted(
        f"{ast.unparse(node)} at cli.py:{node.lineno}"
        for node in occurrences
        if id(node) not in exempt
        if node is not declared[0].value
        if not unwrap.lineno <= node.lineno <= (unwrap.end_lineno or unwrap.lineno)
    )
    assert not raw, (
        "these decide what a bridge reply's `result` is without the envelope "
        "helper, so a reply that carries none is read as a raw KeyError or as a "
        f"silent None: {raw}"
    )
    assert len(exempt) == 1, (
        "the one exempt write -- the fan-out row keyed to `_unwrap_result`'s "
        f"return value -- is claimed by {len(exempt)} sites; an exemption no "
        "site uses is stale, and two sites is two contracts"
    )


def _repo_functions() -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function defined under `src/bn`, by name."""
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for path in sorted((_CLI.parent).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(node)
    assert len(functions) > 200, f"src/bn shrank to {len(functions)} function names"
    return functions


def _dispatched_parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef,
                           slots: frozenset[str]) -> set[str]:
    """The parameters *fn* CALLS -- directly or through anything derived from
    them. This is what makes handing a value to another module dangerous."""
    params = {arg.arg for arg in _all_args(fn)}
    reachable = set(params)
    for _ in range(len(params) + 8):
        before = frozenset(reachable)
        for targets, value, _node, _header in _bindings(fn):
            if _carried(value, reachable):
                reachable |= _bound_names(targets)
        if before == frozenset(reachable):
            break
    dispatched: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            dispatched |= _callee_roots(node, slots) & reachable
    return dispatched


def test_no_bridge_result_transform_leaves_cli_py_unguarded():
    """...and the other way out: a transform handed to another module is invoked
    THERE, where no scan of `cli.py`'s call sites can see it -- the fan-out text
    renderer already calls its `inner_renderer` directly in `formatters.py`. So
    whatever leaves this module must already be bound to the rule.

    The population being total, "any name passed out of the module" would flag
    every `getattr(args, ...)` in the file. So the receiving side answers
    instead: for each call to a function defined elsewhere under `src/bn`, the
    argument is a finding only if THAT function dispatches the parameter it
    lands on. Calls to `cli.py`'s own functions are exempt because the property
    above covers them, and an unresolvable callee (a builtin, a method) cannot
    be shown to invoke anything.
    """
    transforms = _transform_holders()
    in_module = {fn.name for fn in _cli_functions()} | {_BOUNDARY, _GUARD}
    repo = _repo_functions()
    dispatches = {name: set().union(*(_dispatched_parameters(fn, transforms.slots)
                                      for fn in defs))
                  for name, defs in repo.items()}
    leaked = []
    exempt = _rule_exempt_calls()
    for fn in _cli_functions():
        where, bound_at = transforms.paths[id(fn)], transforms.guarded[id(fn)]
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or _callee_names(node) & in_module:
                continue
            if id(node) in exempt:
                continue
            use = where.get(id(node), ())
            for callee in _callee_roots(node, transforms.slots) - in_module:
                if callee not in repo:
                    continue
                params = [arg.arg for arg in _all_args(repo[callee][0])]
                pairs = [*zip(params, node.args)]
                pairs += [(keyword.arg, keyword.value) for keyword in node.keywords
                          if keyword.arg]
                # NOT `isinstance(argument, ast.Name)`: the argument only had
                # to be WRAPPED to leave the module unguarded --
                # `inner_renderer=functools.partial(text_renderer)`, `[t][0]`,
                # `t if c else u` -- and `_stores_holder`/`_carried` already read
                # every one of those as carrying a holder. Two halves of one
                # rule disagreeing is the same defect round 11 fixed for
                # `_dispatch_roots` and BoolOp, one property over.
                leaked += [
                    f"{name} -> {callee}({param}=) in {fn.name}() "
                    f"at cli.py:{node.lineno}"
                    for param, argument in pairs
                    if param in dispatches[callee]
                    for name in _carried(argument, transforms.holders[id(fn)])
                    if not any(_dominates(binding, use)
                               for binding in bound_at.get(name, ()))
                ]
    assert not leaked, (
        "these hand an unguarded bridge-result transform out of cli.py to a "
        "function that INVOKES it, so a malformed result escapes the rule "
        f"there: {sorted(leaked)}"
    )


def _probe_response(op, *, params=None, target=None, timeout=30.0, instance_id=None,
                    spawn_missing_named=False, resolved=False, **kwargs):
    return {"ok": True, "result": {"items": [{"name": "probe"}], "total": 1,
                                   "success": True, "committed": True,
                                   "results": [{"status": "verified"}]}}


def _arrange_always(monkeypatch, tmp_path) -> dict[str, object]:
    """Sites `_call` reaches on every result."""
    return {}


def _arrange_piped_stdout(monkeypatch, tmp_path) -> dict[str, object]:
    """`truncation_note` fires only when the body did not spill and stdout is a pipe."""
    monkeypatch.setattr(bn.cli, "_stdout_is_pipe", lambda: True)
    return {}


def _arrange_spilled_status(monkeypatch, tmp_path) -> dict[str, object]:
    """`spill_status_renderer` fires only on a result that SPILLED, under text."""
    monkeypatch.setenv("BN_SPILL_TOKENS", "1")
    return {"spill_status": lambda result: {"kind": "mutation_summary", "measured": True}}


# The arrangement for each transform parameter's site. Checked against the live
# signature below: a parameter with no arrangement here FAILS, because an
# unexercised transform parameter is exactly an unguarded one.
_TRANSFORM_ARRANGEMENTS = {
    "result_exit_code": _arrange_always,
    "result_transform": _arrange_always,
    "text_renderer": _arrange_always,
    "spill_status": _arrange_always,
    "truncation_note": _arrange_piped_stdout,
    "spill_status_renderer": _arrange_spilled_status,
}


def _call_transform_params() -> list[str]:
    """`_call`'s transform parameters, from the same one declaration the AST
    properties use.

    The population is `inspect.signature` minus the declared data parameters, so
    a parameter added to `_call` is parametrized here whatever it is called and
    however it is annotated, and fails below on its missing arrangement. The
    previous cut derived it from an annotation-and-name recogniser, which an
    `Any`-typed parameter walked straight past.
    """
    return sorted(_declared_transform_params())


@pytest.mark.parametrize("param", _call_transform_params())
def test_every_transform_parameter_of_call_survives_a_malformed_result(param, monkeypatch,
                                                                      tmp_path, capsys):
    """The behavioural half, over the population `inspect.signature` reports.

    Each parameter is handed a transform that raises on the result -- what a
    version-skewed response does to code that parses or aggregates it -- and the
    outcome must be the documented `BridgeError` (exit 2), never an exception
    `main()` does not catch. The cell also asserts the transform really RAN, so
    a parameter whose site the arrangement fails to reach cannot pass by
    proving nothing.
    """
    from bn.transport import BridgeError

    invoked = []

    def raiser(result):
        invoked.append(result)
        raise ValueError("unparseable counter")

    arrange = _TRANSFORM_ARRANGEMENTS.get(param)
    assert arrange is not None, (
        f"_call grew a transform parameter with no arrangement: {param}. Add one "
        "-- an unexercised transform parameter is an unguarded one."
    )
    kwargs = arrange(monkeypatch, tmp_path)
    kwargs[param] = raiser
    monkeypatch.setattr(bn.cli, "send_request", _probe_response)
    args = bn.cli.build_parser().parse_args(["function", "list", "--target", "active"])

    with pytest.raises(BridgeError) as raised:
        bn.cli._call(args, "list_functions", {}, require_target=True, stem="probe", **kwargs)

    assert invoked, f"{param} was never invoked, so this cell proved nothing"
    assert "malformed or newer than this CLI" in str(raised.value), raised.value


def test_unclassifiable_mutation_result_advice_is_actionable(monkeypatch, capsys):
    """The classification guard runs BEFORE rendering, so `--format json` on the
    same call returns this same error envelope and never the raw result. The
    message must therefore not send the reader there -- following advice that
    returns the identical error is how a version-skew report turns into a
    "the CLI is broken" bug.

    The arrangement is a counter whose read RAISES (`float("inf")`, what `1e999`
    decodes to: `int(inf)` is an `OverflowError`). That is what "unclassifiable"
    now means -- no verdict could be derived at all -- and it is the only input
    class that still reaches this message. A counter the read can REFUSE is
    disclosed and classified instead, and is exit 4; see
    `test_a_refused_counter_is_a_disclosed_unmeasured_run_not_a_bridge_error`.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": True, "committed": True,
                                       "go_renamed_candidates": float("inf"),
                                       "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["go", "rename", "--target", "active", "--format", "json"])
    captured = capsys.readouterr()

    # The advertised escape hatch really is closed on this path ...
    assert rc == 2
    assert json.loads(captured.out).get("ok") is False
    assert "go_renamed_candidates" not in captured.out
    # ... so the message must not advertise it.
    assert "--format json" not in captured.err, captured.err
    assert "bn doctor" in captured.err, captured.err


def test_a_refused_counter_is_a_disclosed_unmeasured_run_not_a_bridge_error(
        monkeypatch, capsys):
    """A result the bridge delivered, that did not fail, and whose own counter
    arrived in a shape no count reads out of is exit **4**, not 2.

    This used to be 2, and only because the read RAISED: the CLI produced no
    verdict, so "I could not determine the outcome" was all it had. The read now
    refuses that counter and discloses it by name, which means a verdict DOES
    exist -- this mutation committed, reported no failure, and could not be
    measured -- and that verdict is exactly what exit 4 says. Calling it 2
    instead would hand a `$?`-only consumer the same code it gets for an
    unreachable bridge, an unresolvable target and a refused flag, every one of
    which means *nothing was written*; the caller would then close without
    saving and discard a rename batch that had committed, which is the #683 harm
    this op exists to prevent. 4's instruction -- read the view back and `bn
    save` -- is the one that matches what happened.

    Nothing diagnostic is lost in the move: the field that could not be read is
    named in the payload, where a machine consumer can act on it, instead of in
    a stderr string.
    """
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": True, "committed": True,
                                       # a counter this CLI can refuse but not read
                                       "go_renamed_candidates": "many",
                                       "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["go", "rename", "--target", "active", "--verbose"])

    assert rc == 4
    captured = capsys.readouterr()
    # No traceback, and no claim that the outcome could not be determined.
    assert "Traceback" not in captured.err, captured.err
    assert "malformed or newer than this CLI" not in captured.err, captured.err
    # The unreadable field is named rather than swallowed.
    assert "go_renamed_candidates" in captured.out, captured.out


def test_a_refused_counter_is_disclosed_and_unmeasured_on_the_compact_path(
        monkeypatch, capsys):
    """Same guarantee on the compact default path, where the transform is also
    the renderer input: one rule for every detail level. Here the status line
    itself has to carry the verdict, so it must state the unknown counts as
    unknown and the fail-safe `dirty_after`, not a fabricated zero."""
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {"kind": "go_rename", "preview": False,
                                       "success": True, "committed": True,
                                       "go_renamed_candidates": "many",
                                       "results": []}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["go", "rename", "--target", "active"])

    assert rc == 4
    out = capsys.readouterr().out
    assert "malformed or newer than this CLI" not in out, out
    assert "warning: unmeasured" in out, out
    assert "changed=None" in out and "dirty_after=True" in out, out


def test_exit_2_is_reserved_for_a_mutation_result_that_yields_no_verdict(
        monkeypatch, capsys):
    """The boundary of the whole contract, stated once and run in both directions.

    The classifier returns 0/3/4 whenever a verdict can be DERIVED from the
    result, and 2 exactly when none can. That is what makes 2 meaningful: it is
    not "some field was unreadable" (a refused counter is still classifiable and
    is 4, a refused counter beside a clean failure verdict is still 3) but "this
    CLI cannot say whether the write failed, succeeded or applied unmeasured".

    Both directions are run, because presence alone is satisfiable by a list
    that has stopped keeping up: each listed shape that yields no verdict must
    be 2, and each listed shape that yields one must NOT be 2.

    The two populations are ENUMERATED, and that is a stated limit rather than a
    claim of totality -- "every possible result shape" is not a set this cell
    can iterate. What it does cover is every *decision point* the classifier
    has: the result's type, the `results[]` field's type, a row's type, a row
    status's readability, a counter's readability (refused and raising, since
    those diverge), the summary's return type, and the summary's `measured`
    verdict. A shape that reaches none of those reaches no new code.

    One shape deliberately outside both lists: a row status the CLI does not
    know (a bridge NEWER than this CLI) classifies as 0, because
    `_mutation_summary` counts it as neither a failure nor an unread field. That
    is inherited behaviour -- identical on this PR's base -- and it is decided
    in `formatters.py`, outside this PR's fence, so it is named here rather than
    silently absent.
    """
    from bn.formatters import _go_rename_summary, _mutation_summary
    from bn.transport import BridgeError

    ok = {"success": True, "committed": True}
    go = {"kind": "go_rename", "preview": False, "success": True, "committed": True}

    # A verdict exists -- so the code is the verdict, never 2.
    classifiable = {
        "verified": ({**ok, "results": [{"status": "verified"}]}, _mutation_summary, 0),
        "all-noop": ({**ok, "results": [{"status": "noop"}]}, _mutation_summary, 0),
        "no-rows-to-count": ({**ok, "results": []}, _mutation_summary, 4),
        "row-status-refused": ({**ok, "results": [{"status": 5}]}, _mutation_summary, 4),
        "counter-refused": ({**go, "go_renamed_candidates": "many", "results": []},
                            _go_rename_summary, 4),
        "counter-refused-while-failing":
            ({**go, "success": False, "committed": False,
              "go_renamed_candidates": "many",
              "results": [{"status": "verification_failed"}]}, _go_rename_summary, 3),
        "counters-read-clean":
            ({**go, "go_renamed_candidates": 7, "go_committed_count": 7,
              "go_verified_count": 7, "go_failed_count": 0, "results": []},
             _go_rename_summary, 0),
    }
    for name, (result, summary, expected) in classifiable.items():
        assert bn.cli._mutation_exit_code(result, summary) == expected, name

    # No verdict exists -- so the code is 2, and nothing else.
    unclassifiable = {
        "result-is-not-an-object": (["verified"], _mutation_summary),
        "rows-are-not-a-list": ({**ok, "results": 5}, _mutation_summary),
        "a-row-is-not-an-object": ({**ok, "results": [5]}, _mutation_summary),
        "a-row-status-is-unhashable":
            ({**ok, "results": [{"status": ["verification_failed"]}]}, _mutation_summary),
        "counter-read-raises":
            ({**go, "go_renamed_candidates": float("inf"), "results": []},
             _go_rename_summary),
        "summary-is-not-an-object": ({**ok, "results": []}, lambda result: "nope"),
        "summary-states-no-verdict": ({**ok, "results": []}, lambda result: {"ok": True}),
    }
    for name, (result, summary) in unclassifiable.items():
        with pytest.raises(BridgeError, match="could not classify"):
            bn.cli._mutation_exit_code(result, summary)
    capsys.readouterr()


def test_operation_failure_status_maps_to_exit_3_for_mutation(monkeypatch, capsys):
    """#625: an OperationFailure that escapes a genuine mutation call (routed
    through `_mutate`) with a status in FAILED_MUTATION_STATUSES maps to exit
    3, and its structured fields are surfaced in the --format json envelope."""
    from bn.transport import BridgeError

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError(
            "Cannot apply mutations: this target was loaded with --quick",
            status="invalid_request",
            requested={"op": "mutation", "operations": 1},
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "x"])
    assert rc == 3

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "x", "--format", "json"])
    assert rc == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid_request"
    assert payload["requested"] == {"op": "mutation", "operations": 1}


def test_operation_failure_status_on_read_op_still_exits_2(monkeypatch):
    """#625 scoping: dispatch() attaches `status` to every escaped
    OperationFailure, including read/resolver ops -- not only mutations. A
    read op (`bn function list`, never routed through `_mutate`) whose
    status happens to be in FAILED_MUTATION_STATUSES (e.g. "unsupported")
    must NOT have its exit code widened from 2 to 3; only a genuine mutation
    call may reach exit 3."""
    from bn.transport import BridgeError

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("no matching op", status="unsupported")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["function", "list"])
    assert rc == 2


def test_non_mutation_status_still_exits_2(monkeypatch):
    """A mutation call whose escaped status is not in FAILED_MUTATION_STATUSES
    (a future/unexpected bridge status string) keeps exit 2."""
    from bn.transport import BridgeError

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("weird bridge-side status", status="some_future_status")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["symbol", "rename", "--target", "active", "sub_401000", "x"])
    assert rc == 2


def test_an_unhashable_escaped_status_exits_2_instead_of_tracebacking(monkeypatch):
    """The last member of the family this PR is about: a bridge value tested
    RAW, in the one handler whose whole job is to turn a failure into a
    documented exit code.

    `main()`'s `except BridgeError` asks whether the escaped `status` is a
    `FAILED_MUTATION_STATUSES` member. A set membership test hashes its left
    operand, so a bridge answering with a structured status (an object or an
    array rather than a string) raised `TypeError: unhashable type` OUT of that
    handler -- a traceback and a process exit of 1, which the documented
    0/1/2/3/4 contract does not list, on the error path, for a read op as much as
    for a mutation.

    Such a status is not one of the five failure statuses, so the documented
    answer is the same 2 a future/unexpected status string already gets.
    """
    from bn.transport import BridgeError

    for status in ({"code": "invalid_request"}, ["invalid_request"]):
        def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                              instance_id=None, _status=status, **kwargs):
            raise BridgeError("structured bridge status", status=_status)

        monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
        assert bn.cli.main(
            ["symbol", "rename", "--target", "active", "sub_401000", "x"]) == 2
        monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
        assert bn.cli.main(["function", "list"]) == 2


# The unreadable row status these cells send, kept next to the reason coercing
# it is the wrong repair: `str()` of a structured status is not any member of
# FAILED_MUTATION_STATUSES, so a row that SAID `verification_failed` and could
# not be read would be classified "not a failure".
_UNREADABLE_ROW_STATUS = {"kind": "verification_failed"}


@pytest.mark.parametrize("argv,result", [
    pytest.param(
        ["go", "rename", "--target", "active"],
        {"kind": "go_rename", "success": True, "committed": True,
         "go_renamed_candidates": 1, "go_committed_count": 1,
         "go_verified_count": 1, "results": [{"status": _UNREADABLE_ROW_STATUS}]},
        id="own-summary-op",
    ),
    pytest.param(
        ["symbol", "rename", "--target", "active", "sub_401000", "x"],
        {"success": True, "committed": True,
         "results": [{"status": _UNREADABLE_ROW_STATUS}]},
        id="generic-summary-op",
    ),
])
def test_an_unhashable_result_row_status_is_a_clean_bridge_error(
        monkeypatch, capsys, argv, result):
    """The SAME shape one boundary in, where the answer is deliberately
    different and must stay that way.

    `_mutation_reports_failure` looks each `results[]` row's status up in the
    same set, but it runs INSIDE the malformed-result rule, and `TypeError` is
    in `_MALFORMED_RESULT_ERRORS` -- so an unreadable status is the documented
    BridgeError and exit 2, never a traceback. Pinned because the obvious
    "repair" is to coerce the status to `str` at both sites, and here that would
    silently reclassify a row this CLI could not read as NOT a failure, letting
    an unreadable mutation response continue toward exit 0 or 4. That is exactly
    what the function's docstring refuses to do.

    `own-summary-op` is the parameter that makes this a pin instead of an
    advertisement, and it exists because the `generic-summary-op` one ALONE was
    vacuous: `symbol rename` renders through `_mutation_summary`, which raises on
    the same unreadable row by itself, so exit 2 arrived with the coercion in
    place as readily as without it. `go rename` registers its own summary over
    its own counters and never reads `results[]` on a success, so
    `_mutation_reports_failure` is the ONLY reader of the row: coercing it made
    this command exit 0 and print `committed changed=1 ... failed=0` over a row
    that said `verification_failed`. Both parameters are kept -- one pins the
    generic path's answer, the other detects the change the docstring forbids.
    """
    # The premise, executed: coercion really would classify this row as clean.
    assert str(_UNREADABLE_ROW_STATUS) not in bn.cli.FAILED_MUTATION_STATUSES

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, **kwargs):
        return {"ok": True, "result": result}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(argv)
    assert rc == 2
    assert "classify the mutation result" in capsys.readouterr().err


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
        {"batch_apply": {"ok": True, "result": {"preview": True, "success": True, "committed": False, "results": [{"status": "verified"}]}}}
    )

    rc = bn.cli.main(["batch", "apply", "--preview", "-"])

    assert rc == 0
    assert calls[-1]["params"]["preview"] is True


def test_rename_alias_maps_to_symbol_rename(fake_transport):
    calls = fake_transport({"rename_symbol": {"ok": True, "result": {"preview": True, "results": [{"status": "verified"}]}}})

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


def test_comment_list_defaults_to_page_limit_100(fake_transport):
    # #599: `comment list` must honor the advertised default page limit of 100
    # instead of forwarding argparse's unbounded `None`.
    calls = fake_transport({"list_comments": {"ok": True, "result": []}})
    rc = bn.cli.main(["comment", "list", "--target", "active"])
    assert rc == 0
    assert calls[-1]["params"]["limit"] == 100


def test_comment_list_uncaps_limit_with_out(fake_transport, tmp_path):
    calls = fake_transport({"list_comments": {"ok": True, "result": []}})
    out_path = tmp_path / "comments.json"
    rc = bn.cli.main(["comment", "list", "--target", "active", "--out", str(out_path)])
    assert rc == 0
    assert calls[-1]["params"]["limit"] is None


def test_comment_list_explicit_limit_wins(fake_transport):
    calls = fake_transport({"list_comments": {"ok": True, "result": []}})
    rc = bn.cli.main(["comment", "list", "--target", "active", "--limit", "17"])
    assert rc == 0
    assert calls[-1]["params"]["limit"] == 17
