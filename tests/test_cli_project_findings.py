"""Finding document refusals and truthful user-facing reports."""
from __future__ import annotations

import io
import json

import pytest

import bn.cli
from bn import formatters


def _record(**changes):
    row = {
        "id": "F-example", "revision": 3, "claim_revision": 1,
        "type": "vulnerability", "kind": "hypothesis",
        "claim": "An unchecked length may reach the copy", "author": "analyst",
        "review_state": "accepted", "freshness": "stale",
        "scope": [{"file_id": "file-1", "file_path": "/bin/service"}],
        "assumptions": ["Caller permits untrusted lengths"],
        "limitations": ["No dynamic reproduction"],
        "next_steps": ["Recheck changed bytes"],
        "evidence": [{"check": "bytes", "capture_state": "current", "observed": {"hex": "9090"}}],
        "reviews": [{"reviewer": "second", "verdict": "accept", "rationale": "Independent byte check"}],
        "freshness_reasons": ["Persisted file changed"],
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("body", ["", "{", "{\"claim\":", "[" * 2000])
def test_malformed_stdin_refused_before_bridge(body, monkeypatch, fake_transport, capsys):
    calls = fake_transport()
    monkeypatch.setattr("sys.stdin", io.StringIO(body))
    assert bn.cli.main(["project", "finding", "add", "-", "--author", "a"]) == 2
    captured = capsys.readouterr()
    assert "Invalid JSON" in captured.err
    assert "Traceback" not in captured.err
    assert calls == []


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid-utf8", "malformed"])
def test_bad_document_file_is_safe(kind, tmp_path, fake_transport, capsys):
    path = tmp_path / "claim.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif kind == "malformed":
        path.write_text("{")
    calls = fake_transport()
    assert bn.cli.main(["project", "finding", "add", str(path), "--author", "a"]) == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err
    assert calls == []


def test_stdin_read_failure_is_safe(monkeypatch, fake_transport, capsys):
    class BrokenInput:
        def read(self):
            raise OSError("input unavailable")
    monkeypatch.setattr("sys.stdin", BrokenInput())
    calls = fake_transport()
    assert bn.cli.main(["project", "finding", "add", "-", "--author", "a"]) == 2
    assert "input unavailable" in capsys.readouterr().err
    assert calls == []


def test_report_preserves_qualifications_and_independent_states(fake_transport, capsys):
    fake_transport({"project_finding_show": {"ok": True, "result": _record()}})
    assert bn.cli.main(["project", "finding", "show", "F-example"]) == 0
    text = capsys.readouterr().out
    for expected in ("type=vulnerability", "review=accepted", "freshness=stale",
                     "Caller permits untrusted lengths", "No dynamic reproduction",
                     "Persisted file changed", "Independent byte check", "9090",
                     "not claim truth"):
        assert expected in text


def test_unknown_check_is_successful_read_not_acceptance(fake_transport, capsys):
    fake_transport({"project_finding_check": {"ok": True, "result": {
        "id": "F-example", "revision": 1, "type": "open-question",
        "review_state": "proposed", "freshness": "unknown",
        "reasons": ["No machine-checkable evidence"],
        "checks": [{"check": "note", "freshness": "unknown"}],
    }}})
    assert bn.cli.main(["project", "finding", "check", "F-example"]) == 0
    text = capsys.readouterr().out
    assert "review=proposed" in text and "freshness=unknown" in text
    assert "No machine-checkable evidence" in text
    assert "no review was added" in text


def test_context_keeps_stale_acceptance_in_needs_review():
    result = {"groups": {"accepted": [], "needs_review": [_record()]},
              "total": 1, "offset": 0, "limit": 50, "has_more": False}
    text = formatters._render_project_finding_context_text(result)
    assert "accepted:\n  No findings." in text
    assert "needs_review:\n  F-example" in text
    assert "review=accepted freshness=stale" in text
    assert "No dynamic reproduction" in text


@pytest.mark.parametrize("malformed", [1, 0, False, "", "unreadable rows", {}])
def test_context_discloses_and_preserves_malformed_group(malformed):
    result = {"groups": {"a": malformed, "needs_review": [_record()]}}
    text = formatters._render_project_finding_context_text(result)
    assert f"a:\n  {json.dumps(malformed)}" in text
    assert "malformed a field" in text
    assert "No findings." not in text
    for expected in ("F-example", "type=vulnerability", "review=accepted",
                     "freshness=stale", "No dynamic reproduction", "not claim truth"):
        assert expected in text


@pytest.mark.parametrize("malformed", [1, 0, False, "", ["unreadable groups"]])
def test_context_discloses_and_preserves_malformed_groups_container(malformed):
    text = formatters._render_project_finding_context_text({"groups": malformed})
    assert f"groups: {json.dumps(malformed)}" in text
    assert "malformed groups field" in text
    assert "No findings." not in text
    assert "not claim truth" in text


@pytest.mark.parametrize("render,payload,field", [
    (formatters._render_project_finding_context_text,
     {"groups": {"needs_review": [None, _record()]}}, "needs_review[0]"),
    (formatters._render_project_findings_text,
     {"items": [None, _record()]}, "items[0]"),
])
def test_finding_lists_preserve_and_disclose_malformed_rows(render, payload, field):
    text = render(payload)
    assert "null" in text
    assert f"malformed {field} field" in text
    assert "F-example" in text
    assert "review=accepted freshness=stale" in text
    assert "No dynamic reproduction" in text
    assert "No findings." not in text


@pytest.mark.parametrize("malformed", [1, 0, False, "", {"claim": "retained"}])
def test_findings_preserve_malformed_items_without_claiming_empty(malformed):
    text = formatters._render_project_findings_text({"items": malformed})
    assert f"items: {json.dumps(malformed)}" in text
    assert "malformed items field" in text
    assert "No findings." not in text


def test_history_displays_revision_types_and_page_boundaries():
    result = {"items": [_record(revision=1, type="open-question"), _record()],
              "total": 3, "offset": 0, "limit": 2, "has_more": True}
    text = formatters._render_project_findings_text(result)
    assert "F-example r1 type=open-question" in text
    assert "F-example r3 type=vulnerability" in text
    assert "total=3 offset=0 limit=2 has_more=true" in text


def test_types_displays_custom_type_count_and_recommended_action(fake_transport, capsys):
    fake_transport({"project_finding_types": {"ok": True, "result": {"types": [
        {"type": "custom:allocator-rule", "count": 4,
         "description": "Allocator-specific constraint", "action": "Check arena ownership"},
    ]}}})
    assert bn.cli.main(["project", "finding", "types"]) == 0
    text = capsys.readouterr().out
    assert "custom:allocator-rule count=4" in text
    assert "Allocator-specific constraint" in text and "Check arena ownership" in text


@pytest.mark.parametrize("render,wrap", [
    (formatters._render_project_finding_text, lambda row: row),
    (formatters._render_project_finding_check_text, lambda row: row),
    (formatters._render_project_findings_text, lambda row: {"items": [row]}),
    (formatters._render_project_finding_context_text,
     lambda row: {"groups": {"needs_review": [row]}}),
])
@pytest.mark.parametrize("field,malformed", [
    ("scope", {"file_id": "retained"}),
    ("scope_evidence", False),
    ("assumptions", "retained assumption"),
    ("limitations", 0),
    ("next_steps", ""),
    ("dependencies", {"id": "F-retained"}),
    ("tags", False),
    ("evidence", {"observed": "retained observation"}),
    ("reviews", {"rationale": "retained review"}),
    ("freshness_reasons", "retained reason"),
    ("reasons", 1),
    ("checks", {"freshness": "unknown"}),
    ("action", []),
    ("claim", []),
    ("kind", {}),
    ("author", False),
    ("claim_revision", {}),
    ("replacement", []),
    ("created_at", {}),
    ("updated_at", []),
])
def test_finding_details_disclose_wrong_shapes_without_losing_payload(
        render, wrap, field, malformed):
    text = render(wrap(_record(**{field: malformed})))
    assert f"{field}: {json.dumps(malformed)}" in text
    assert f"malformed {field} field" in text
    assert "type=vulnerability" in text
    assert "review=accepted freshness=stale" in text
    assert "not claim truth" in text


def test_valid_structured_finding_details_are_not_reported_as_malformed():
    row = _record(
        scope_evidence=[{"check": "file", "capture_state": "current"}],
        dependencies=[{"id": "F-other", "revision": 1}], tags=["triage"],
        reasons=["Persisted file changed"],
        checks=[{"check": "bytes", "freshness": "stale"}],
        replacement="F-replacement", created_at="2026-11-08T00:00:00Z",
        updated_at="2026-11-08T00:00:00Z",
        action={"kind": "review", "actor": "second", "timestamp": "2026-11-08T00:00:00Z"},
    )
    text = formatters._render_project_finding_text(row)
    assert "malformed" not in text
    for expected in ("type=vulnerability", "review=accepted", "freshness=stale",
                     "Caller permits untrusted lengths", "No dynamic reproduction",
                     "Recheck changed bytes", "Independent byte check", "9090",
                     "Persisted file changed", "F-other", "F-replacement", "not claim truth"):
        assert expected in text
