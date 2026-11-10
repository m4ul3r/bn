"""Observable project formatter contracts, not source/AST/population pins.

A helper count or registry census cannot prove safe terminal output, honest
unknown counts, or correct mutation exit codes. Exercise those consumer harms
with malformed bridge payloads and firmware-controlled cells instead.
"""

import copy
import unicodedata

import pytest

from bn import cli, formatters as fmt


_POISON = "CELL\x00\x1b[31m\nFORGED\r\t\x7f\x85\x9b\xad\u202e\u2066\U000e0001\u2028café"


def _replace_cells(value, replacement):
    if isinstance(value, str):
        return value.replace("CELL", replacement)
    if isinstance(value, list):
        return [_replace_cells(item, replacement) for item in value]
    if isinstance(value, dict):
        return {_replace_cells(key, replacement): _replace_cells(item, replacement)
                for key, item in value.items()}
    return value


_FILE = {"id": "CELL", "path": "CELL", "open_target_id": "CELL",
         "sibling_database_id": "CELL", "size": 10, "is_database": False}
_SYNC = {"root": "CELL", "platform": "CELL", "lookup_paths": [
    {"path": "CELL", "source": "CELL"}, "CELL"], "imported": [_FILE, "CELL"],
    "changed": [{"rel_realpath": "CELL", "project_file_id": "CELL"}, "CELL"],
    "skipped_existing": 0, "notes": ["CELL"]}
_FINDING = {"id": "CELL", "type": "CELL", "review_state": "CELL",
            "freshness": "CELL", "revision": "CELL", "claim": "CELL",
            "author": "CELL", "evidence": [{"CELL": "CELL"}],
            "assumptions": ["CELL"], "reviews": [{"rationale": "CELL"}],
            "action": {"kind": "CELL", "actor": "CELL"}}


@pytest.mark.parametrize("render,payload", [
    (fmt._render_project_attach_text, {"instance_id": "CELL", "path": "CELL"}),
    (fmt._render_project_info_text, {"name": "CELL", "instance": "CELL", "path": "CELL",
        "id": "CELL", "description": "CELL", "file_count": "CELL", "folder_count": "CELL",
        "open_targets": [{"project_path": "CELL", "target_id": "CELL", "view_id": "CELL",
                          "project_file_id": "CELL"}, "CELL"]}),
    (fmt._render_project_file_text, _FILE),
    (fmt._render_project_file_text, {**_FILE, "source_symlink": "CELL -> CELL"}),
    (fmt._render_project_files_text, {"items": [_FILE, "CELL"], "total": "CELL",
                                    "offset": "CELL", "limit": "CELL", "has_more": "CELL",
                                    "folders": [{"id": "CELL", "path": "CELL", "files": "CELL",
                                                 "files_recursive": "CELL"}, "CELL"]}),
    # The retained `files` alias (#1080 review: a bridge from before the
    # items/paging upgrade), exercised on the same escaping contract.
    (fmt._render_project_files_text, {"files": [_FILE, "CELL"],
                                    "folders": [{"id": "CELL", "path": "CELL"}, "CELL"]}),
    (fmt._render_project_load_text, {"path": "CELL", "project_path": "CELL",
        "project_file_id": "CELL", "notes": ["CELL"],
        "targets": [{"selector": "CELL"}, "CELL"]}),
    (fmt._render_save_text, {"path": "CELL", "project_file_id": "CELL",
        "project_path": "CELL", "mode": "CELL"}),
    (fmt._render_project_sync_text, _SYNC),
    (fmt._render_project_init_text, {"root": "CELL", "platform": "CELL", "markers": ["CELL"],
        "project_path": "CELL", "instance": "CELL", "sync": _SYNC}),
    (fmt._render_project_init_text, {"root": "CELL", "platform": "CELL", "markers": ["CELL"],
        "project_path": "CELL", "dry_run": True, "notes": ["CELL"],
        "lookup_paths": [{"path": "CELL", "source": "CELL"}]}),
    (fmt._render_project_init_text, {"project_path": "CELL", "sync_error": "CELL"}),
    (fmt._render_project_libs_text, {"root": "CELL", "platform": "CELL", "synced_at": "CELL",
        "libs": [{"elf_class": "CELL", "e_machine": "CELL", "names": ["CELL"],
                  "soname": "CELL", "project_file_id": "CELL", "project_path": "CELL",
                  "sibling_database_id": "CELL"}, "CELL"]}),
    (fmt._render_project_link_text, {"status": "CELL", "libraries": [
        {"name": "CELL", "resolved": True, "project_path": "CELL", "backing": "CELL"},
        {"name": "CELL", "resolved": False}], "unresolved": [
        {"symbol": "CELL", "reason": "CELL"}], "scope": ["CELL"],
        "ambiguous_libraries": ["CELL", {"CELL": "CELL"}], "notes": ["CELL"]}),
    (fmt._render_project_export_text, {"dest": "CELL", "project_file_id": "CELL"}),
    (fmt._render_project_delete_text, {"project_file_id": "CELL"}),
    (fmt._render_project_close_text, {"path": "CELL", "targets_closed": [
        {"path": "CELL", "unsaved": True}, "CELL"]}),
    (fmt._render_project_finding_text, _FINDING),
    (fmt._render_project_finding_check_text, _FINDING),
    (fmt._render_project_findings_text, {"total": "CELL", "offset": "CELL", "limit": "CELL",
        "has_more": "CELL", "items": [_FINDING, "CELL"]}),
    (fmt._render_project_finding_context_text, {"groups": {"CELL": [_FINDING, "CELL"]}}),
    (fmt._render_project_finding_context_text, {"groups": {"CELL": "CELL"}}),
    (fmt._render_project_finding_types_text, {"types": [
        {"type": "CELL", "count": "CELL", "description": "CELL", "action": "CELL"}, "CELL"]}),
])
def test_project_cells_cannot_inject_terminal_controls_or_extra_rows(render, payload):
    hostile = _replace_cells(payload, _POISON)
    before = copy.deepcopy(hostile)
    baseline = render(payload)
    text = render(hostile)
    assert hostile == before  # Text rendering must not sanitize the JSON payload in place.
    assert len(text.split("\n")) == len(baseline.split("\n"))
    assert not any(unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")
                   for c in text if c not in "\n\t")
    assert "\nFORGED" not in text
    assert "café" in text


@pytest.mark.parametrize("character,escaped", [
    ("\x00", r"\x00"), ("\x1b", r"\x1b"), ("\x7f", r"\x7f"), ("\x9b", r"\x9b"),
    ("\xad", r"\xad"), ("\u202e", r"\u202e"), ("\u2066", r"\u2066"),
    ("\U000e0001", r"\U000e0001"), ("\n", r"\n"), ("\r", r"\r"), ("\t", r"\t"),
])
def test_control_escape_is_visible_and_preserves_printable_unicode(character, escaped):
    assert fmt._escape_control_chars("café" + character) == "café" + escaped


def _link(**changes):
    result = {"status": "verified", "success": True, "committed": True,
              "linked": 0, "relinked": 0, "rebound": 0, "already_linked": 0,
              "unresolved": [], "libraries": []}
    result.update(changes)
    return result


@pytest.mark.parametrize("preview", [False, True])
def test_rebind_only_link_is_a_measured_change_not_a_clean_noop(preview):
    payload = _link(rebound=2, preview=preview, committed=not preview)
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is True
    assert summary["changed_count"] == summary["verified_count"] == summary["op_count"] == 2
    assert summary["project_link"]["rebound"] == 2
    assert summary["dirty_after"] is not preview
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 0
    assert "rebound 2" in fmt._render_mutation_summary_text(summary)


@pytest.mark.parametrize("raw", [None, "bad", {}, [], True, -1, 0.5])
def test_malformed_rebound_never_fabricates_zero_or_clean_state(raw):
    payload = _link(rebound=raw)
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is False
    assert summary["project_link"]["rebound"] is None
    assert summary["changed_count"] is None and summary["op_count"] is None
    assert summary["dirty_after"] is True
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 4
    assert "rebound ?" in fmt._render_mutation_summary_text(summary)


def test_missing_rebound_is_unknown_not_backwards_compatible_zero():
    payload = _link()
    del payload["rebound"]
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is False and summary["project_link"]["rebound"] is None
    assert summary["dirty_after"] is True and summary["op_count"] is None
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 4


@pytest.mark.parametrize("status", sorted(fmt.FAILED_MUTATION_STATUSES))
def test_link_failure_vocabulary_and_notes_reach_both_text_views(status):
    payload = _link(status=status, success=False, committed=False, rebound=1, measured=False,
                    notes=["provider could not be rebound", "readback failed"])
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is False and summary["ok"] is False
    assert summary["failed_count"] is None and summary["dirty_after"] is True
    assert "provider could not be rebound; readback failed" in summary["first_error"]
    assert "did not verify its planned writes" in summary["first_error"]
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 3
    for text in (fmt._render_project_link_text(payload), fmt._render_mutation_summary_text(summary)):
        assert "provider could not be rebound" in text and "readback failed" in text
        assert "unknown status" not in text


def test_link_failure_message_takes_precedence_and_text_escapes_notes():
    payload = _link(status="internal_error", success=False, committed=False,
                    measured=False, notes=[_POISON])
    summary = fmt._project_link_summary(payload)
    assert _POISON in summary["first_error"]
    text = fmt._render_mutation_summary_text(summary)
    assert "\x1b" not in text and "\u202e" not in text and "\nFORGED" not in text
    payload["message"] = "primary failure"
    assert fmt._project_link_summary(payload)["first_error"].startswith("primary failure")


@pytest.mark.parametrize("key", ["preview", "dry_run", "success", "committed", "measured"])
def test_string_false_flags_cannot_assert_preview_or_measured_success(key):
    payload = _link(**{key: "false"})
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is False and summary["dirty_after"] is True
    assert summary["preview"] is False
    assert key in summary["first_error"]
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 4
    if key in ("preview", "dry_run"):
        text = fmt._render_project_link_text(payload)
        assert "[preview]" not in text and key in text.split("! malformed", 1)[1]


def test_string_false_dry_run_cannot_claim_nothing_was_created():
    text = fmt._render_project_init_text({"dry_run": "false", "project_path": "/sample.bnpr"})
    assert "nothing was created" not in text
    assert "unknown (unreadable dry_run)" in text and "! malformed dry_run" in text


@pytest.mark.parametrize("raw", ["unresolved", {"symbol": "foo"}, ["bad row"], [None]])
def test_unreadable_unresolved_is_unknown_through_summary_and_load(raw):
    payload = _link(unresolved=raw)
    summary = fmt._project_link_summary(payload)
    assert summary["project_link"]["unresolved"] is None
    assert summary["measured"] is False and summary["changed_count"] is None
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 4
    assert "unresolved ?" in fmt._render_mutation_summary_text(summary)
    loaded = fmt._render_project_load_text({"path": "app", "link": payload})
    assert "(? unresolved)" in loaded and "(0 unresolved)" not in loaded
    assert "! malformed unresolved" in loaded


@pytest.mark.parametrize("key", ["libraries", "ambiguous_libraries", "scope", "notes"])
def test_string_instead_of_container_is_disclosed_not_iterated_as_characters(key):
    payload = _link(**{key: "FORGED"})
    text = fmt._render_project_link_text(payload)
    assert "! malformed" in text and key in text.split("! malformed", 1)[1]
    assert "F, O, R, G, E, D" not in text


def test_string_resolution_flag_does_not_claim_a_provider_was_found():
    text = fmt._render_project_link_text(_link(libraries=[{
        "name": "libsample.so", "resolved": "false", "project_path": "/fake/provider"}]))
    assert "resolution unknown" in text and "/fake/provider" not in text
    assert "! malformed resolved" in text


def test_empty_findings_and_init_use_the_shared_empty_marker():
    assert fmt._empty_result("findings") in fmt._render_project_findings_text({"items": []})
    context = fmt._render_project_finding_context_text({"groups": {"accepted": []}})
    assert "accepted:\n  " + fmt._empty_result("findings") in context
    init = fmt._render_project_init_text({"dry_run": True, "markers": [], "lookup_paths": []})
    assert "markers: " + fmt._empty_result() in init
    assert "lookup paths: " + fmt._empty_result() in init


def _receipt(**changes):
    result = {"schema_version": 1, "id": "F-" + "a" * 32, "project_id": "sample-project",
              "revision": 2, "claim_revision": 1, "integrity": "b" * 64,
              "action": {"kind": "review", "actor": "reviewer", "timestamp": "2026-11-08"},
              "review_state": "rejected", "freshness": "stale"}
    result.update(changes)
    return result


def test_persisted_finding_receipt_is_measured_independently_of_claim_acceptance():
    receipt = _receipt()
    summary = fmt._project_finding_summary(receipt)
    assert summary["measured"] is True and summary["ok"] is True
    assert summary["changed_count"] == summary["verified_count"] == summary["op_count"] == 1
    assert summary["dirty_after"] is False
    assert cli._mutation_exit_code(receipt, fmt._project_finding_summary) == 0
    assert summary["project_finding"] == {"id": receipt["id"], "revision": 2}
    assert f"finding: {receipt['id']} r2" in fmt._render_mutation_summary_text(summary)


@pytest.mark.parametrize("changes", [
    {"id": "not-a-finding"}, {"id": {}}, {"schema_version": True}, {"revision": 0},
    {"revision": False}, {"claim_revision": 3}, {"project_id": None}, {"integrity": "bad"},
    {"action": "review"}, {"action": {}}, {"action": {"kind": "delete"}},
])
def test_malformed_finding_receipt_cannot_report_a_verified_append(changes):
    receipt = _receipt(**changes)
    summary = fmt._project_finding_summary(receipt)
    assert summary["measured"] is False and summary["verified_count"] is None
    assert summary["op_count"] is None and summary["ok"] is False
    assert cli._mutation_exit_code(receipt, fmt._project_finding_summary) == 4
    assert "append receipt could not be read" in fmt._render_mutation_summary_text(summary)


def test_empty_project_reopen_failure_is_visible_without_target_rows():
    text = fmt._render_session_start_text({"instance_id": "sample", "loaded": [],
        "project": {"path": "/sample.bnpr", "opened": False, "error": "lease held"}})
    assert "project reopen FAILED: /sample.bnpr: lease held" in text


@pytest.mark.parametrize("census", ["bad", [], ["bad"], 1, True])
def test_callsites_discloses_unreadable_vtable_census_instead_of_silent_absence(census):
    payload = {"items": [], "total": 0, "callee_data_ref_count": 1,
               "callee_data_refs": ["0x401000"]}
    absent = fmt._render_callsites_text(payload)
    malformed = fmt._render_callsites_text({**payload, "callee_data_ref_census": census})
    assert "table identity unknown" in malformed
    assert "data refs do not identify indirect callers" in malformed
    assert "! malformed callee_data_ref_census" in malformed
    assert malformed != absent


@pytest.mark.parametrize("linked,relinked,rebound,already", [
    (2, 1, 1, 3), (0, 0, 0, 0), (0, 0, 0, 2),
])
def test_unresolved_imports_are_visible_but_not_counted_as_attempted_mutations(
    linked, relinked, rebound, already
):
    unresolved = [{"symbol": "missing_symbol", "reason": "provider not found"}]
    payload = _link(linked=linked, relinked=relinked, rebound=rebound,
                    already_linked=already, unresolved=unresolved)
    summary = fmt._project_link_summary(payload)
    assert summary["measured"] is True and summary["ok"] is True
    assert summary["op_count"] == linked + relinked + rebound + already
    assert summary["op_count"] == (
        summary["changed_count"] + summary["noop_count"] + summary["failed_count"])
    assert summary["project_link"]["unresolved"] == unresolved
    text = fmt._render_mutation_summary_text(summary)
    assert "unresolved 1" in text and "missing_symbol" in text
    assert "provider not found" in text
    assert cli._mutation_exit_code(payload, fmt._project_link_summary) == 0


@pytest.mark.parametrize("status", ["internal_error", "verification_failed", "rollback_failed"])
def test_project_load_does_not_hide_failed_auto_link_behind_successful_load(status):
    link = _link(status=status, success=False, committed=False, measured=False,
                 notes=["provider assignment failed", "readback incomplete"])
    text = fmt._render_project_load_text({"path": "/sample/app", "link": link})
    assert "loaded: /sample/app" in text
    assert f"auto-link FAILED: {status}" in text
    assert "provider assignment failed; readback incomplete" in text


def test_project_load_auto_link_failure_notes_are_terminal_safe():
    link = _link(status="internal_error", success=False, committed=False, measured=False,
                 notes=[_POISON])
    text = fmt._render_project_load_text({"path": "/sample/app", "link": link})
    assert "auto-link FAILED: internal_error" in text
    assert "\\nFORGED" in text and "\nFORGED" not in text
    assert "\x1b" not in text and "\u202e" not in text


def test_project_load_caught_link_exception_keeps_failure_visible():
    payload = {"path": "/sample/app", "loaded": True, "target": "app",
               "targets": [{"selector": "app"}],
               "link": {"success": False, "status": "internal_error",
                        "error": {"message": "provider open failed"},
                        "notes": ["provider open failed"]}}
    text = fmt._render_project_load_text(payload)
    assert "loaded: /sample/app" in text and "targets:\n- app" in text
    # #1073 item 3: a failed link has no counters to be unknown about.
    assert "linked: n/a (link failed)" in text and "linked: ?" not in text
    assert "auto-link FAILED: internal_error" in text
    assert "auto-link note: provider open failed" in text
