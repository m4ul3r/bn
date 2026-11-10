"""Immutable findings behavior against injected project files, without native BN."""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _bridge_fakes import _load_bridge
from _project_fakes import FakeProject


class LedgerProject(FakeProject):
    """Native create_file persists nonempty contents; shared fake models empty DBs."""

    def create_file(self, contents, folder, name, description=""):
        return self.add_file(name, folder, data=contents, description=description)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    _load_bridge(monkeypatch)
    module = importlib.import_module("bn_test_bridge.project_findings")
    po = sys.modules["bn_test_bridge.project_ops"]
    project = LedgerProject(tmp_path / "demo.bnpr")
    manager = po.ProjectManager()
    manager._state = po.ProjectState(project, str(project.root), project.id, project.name)
    bridge = SimpleNamespace(projects=manager, targets=SimpleNamespace(refresh=lambda: []))
    source = project.add_file("firmware", data=b"input bytes")
    attachment = project.add_file("trace", data=b"independent reproduction")
    return SimpleNamespace(api=module, bridge=bridge, project=project, source=source, attachment=attachment)


def claim(env, **changes):
    value = {"type": "behavior", "claim": "Input reaches parser", "kind": "observation",
             "scope": [{"file": env.source.id, "function": "parse_input"}],
             "evidence": [{"check": "file", "file": env.source.id}],
             "assumptions": ["The imported image matches deployment"],
             "limitations": ["Interpretation is not automatically verified"],
             "next_steps": ["Independently reproduce"], "tags": ["parser"]}
    value.update(changes)
    return value


def add(env, **changes):
    return env.api.project_finding_add(env.bridge, claim(env, **changes), "analyst")


def review(env, row, verdict="accept", reviewer="peer", **changes):
    document = {"verdict": verdict, "rationale": "Checked independently",
                "evidence": [{"check": "file", "file": env.attachment.id}]}
    document.update(changes)
    return env.api.project_finding_review(env.bridge, row["id"], document, row["revision"], reviewer)


def snapshots(env):
    return [pf for pf in env.project.files if pf.get_path_in_project().startswith("/.bn-findings/")]


def test_append_only_update_reloads_and_history_discloses_new_revision(ledger):
    first = add(ledger)
    old_file = snapshots(ledger)[0]
    old_bytes = Path(old_file.path_on_disk).read_bytes()
    assert first["review_state"] == "proposed" and first["freshness"] == "current"
    accepted = review(ledger, first)
    assert accepted["review_state"] == "accepted" and accepted["claim_revision"] == 1
    updated = ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger, claim="Corrected claim"), 2, "new author")
    assert updated["claim_revision"] == 3 and updated["reviews"] == []
    assert updated["review_state"] == "proposed"
    assert Path(old_file.path_on_disk).read_bytes() == old_bytes
    persisted = [json.loads(Path(pf.path_on_disk).read_bytes()) for pf in snapshots(ledger)]
    assert [row["revision"] for row in persisted] == [1, 2, 3]
    assert persisted[1]["previous_digest"] == persisted[0]["integrity"]
    assert persisted[2]["previous_digest"] == persisted[1]["integrity"]
    assert all("freshness" not in row for row in persisted)
    shown = ledger.api.project_finding_show(ledger.bridge, first["id"])
    assert shown["claim"] == "Corrected claim" and shown["author"] == "new author"
    page = ledger.api.project_finding_history(ledger.bridge, first["id"], offset=1, limit=1)
    assert page["total"] == 3 and page["has_more"] is True
    assert page["items"][0]["revision"] == 2 and page["items"][0]["freshness"] == "stale"
    assert any("newer revision 3" in reason for reason in page["items"][0]["freshness_reasons"])


def test_compare_and_swap_refuses_before_capturing_or_writing(ledger, monkeypatch):
    first = add(ledger)
    review(ledger, first)
    before = len(ledger.project.files)
    monkeypatch.setattr(ledger.api, "capture_evidence", lambda *_: pytest.fail("capture after CAS conflict"))
    with pytest.raises(RuntimeError, match="current revision 2"):
        ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger), 1, "analyst")
    with pytest.raises(RuntimeError, match="current revision 2"):
        ledger.api.project_finding_review(ledger.bridge, first["id"], {}, 1, "peer")
    assert len(ledger.project.files) == before


@pytest.mark.parametrize("document", [
    {}, {"type": "unknown"}, {"type": "custom:Bad_Slug"}, {"kind": "fact"},
    {"scope": []}, {"scope": [{"file": "missing"}]}, {"scope": [{"file": "firmware", "typo": 1}]},
    {"evidence": []}, {"unexpected": "do not ignore"}, {"tags": [""]},
    {"dependencies": [{"id": "missing", "revision": 1}]}, {"claim": " "},
])
def test_invalid_claims_do_not_write(ledger, document):
    value = {} if document == {} else claim(ledger, **document)
    with pytest.raises(RuntimeError):
        ledger.api.project_finding_add(ledger.bridge, value, "analyst")
    assert snapshots(ledger) == []


@pytest.mark.parametrize("author", ["", "  ", None, 7])
def test_identity_is_required_declared_text(ledger, author):
    with pytest.raises(RuntimeError, match="author"):
        ledger.api.project_finding_add(ledger.bridge, claim(ledger), author)
    assert snapshots(ledger) == []


def test_no_self_accept_and_manual_evidence_cannot_establish_acceptance(ledger):
    first = add(ledger)
    with pytest.raises(RuntimeError, match="Self acceptance"):
        review(ledger, first, reviewer="analyst")
    with pytest.raises(RuntimeError, match="machine-checkable"):
        review(ledger, first, evidence=[{"check": "note", "text": "Looks plausible"}])
    assert len(snapshots(ledger)) == 1
    note = add(ledger, evidence=[{"check": "note", "text": "Never execute: /bin/echo payload"}])
    assert note["freshness"] == "unknown"
    with pytest.raises(RuntimeError, match="current claim"):
        review(ledger, note)
    rejected = review(ledger, note, verdict="reject", evidence=[{"check": "note", "text": "Unsupported interpretation"}])
    assert rejected["review_state"] == "rejected"
    assert rejected["reviews"][0]["evidence"][0]["text"] == "Unsupported interpretation"


@pytest.mark.parametrize("author, reviewer", [
    ("analyst", "Analyst"),
    ("analyst", "ANALYST"),
    ("analyst", "\uff41\uff4e\uff41\uff4c\uff59\uff53\uff54"),   # fullwidth: NFKC-equal
    ("lead analyst", "lead \t  analyst"),                         # whitespace run
    ("lead analyst", "lead\u00a0analyst"),                        # no-break space
])
def test_self_accept_refusal_is_a_typo_guard_over_the_declared_label(ledger, author, reviewer):
    """#1074 item 1: the self-accept check compares labels after NFKC, case
    folding, trimming and whitespace collapse. It is a typo guard, not
    authentication -- a genuinely different label still accepts. The refusal
    names both labels and the comparison, since they do differ as typed."""
    first = ledger.api.project_finding_add(ledger.bridge, claim(ledger), author)
    assert first["author"] == author
    with pytest.raises(RuntimeError, match="Self acceptance") as refused:
        review(ledger, first, reviewer=reviewer)
    message = str(refused.value)
    assert repr(reviewer) in message and repr(author) in message
    assert "ignoring case" in message
    assert len(snapshots(ledger)) == 1
    accepted = review(ledger, first, reviewer="analyst-2")
    assert accepted["review_state"] == "accepted"
    assert accepted["reviews"][-1]["reviewer"] == "analyst-2"
    assert accepted["author"] == author


def test_a_stored_accept_from_before_the_typo_guard_still_loads(ledger, monkeypatch):
    """The guard is at the WRITE. A ledger written by an earlier version may
    hold an accept whose reviewer differs from the author only in case; the
    stored-record validator compares exactly, so that history still loads
    instead of becoming an integrity failure (#1074 review)."""
    first = add(ledger)
    with monkeypatch.context() as earlier:
        earlier.setattr(ledger.api, "_same_label", lambda a, b: a == b)
        accepted = review(ledger, first, reviewer="Analyst")
    assert accepted["review_state"] == "accepted"
    shown = ledger.api.project_finding_show(ledger.bridge, first["id"])
    assert shown["review_state"] == "accepted"
    assert shown["reviews"][-1]["reviewer"] == "Analyst" and shown["author"] == "analyst"
    listed = ledger.api.project_finding_list(ledger.bridge)
    assert [row["id"] for row in listed["items"]] == [first["id"]]


def test_acceptance_uses_declared_author_of_current_claim_revision(ledger):
    first = add(ledger)
    changed = ledger.api.project_finding_update(
        ledger.bridge, first["id"], claim(ledger, claim="Corrected interpretation"),
        first["revision"], "corrector")
    with pytest.raises(RuntimeError, match="Self acceptance"):
        review(ledger, changed, reviewer=" corrector ")
    accepted = review(ledger, changed, reviewer="analyst")
    assert accepted["author"] == "corrector"
    assert accepted["claim_revision"] == changed["revision"]
    assert accepted["reviews"][-1]["reviewer"] == "analyst"
    assert accepted["review_state"] == "accepted"
    history = ledger.api.project_finding_history(ledger.bridge, first["id"])
    assert history["items"][0]["author"] == "analyst"



def test_active_reviewer_votes_and_freshness_are_independent(ledger):
    row = add(ledger)
    row = review(ledger, row, reviewer="one")
    row = review(ledger, row, verdict="reject", reviewer="two")
    assert row["review_state"] == "disputed" and row["freshness"] == "current"
    row = review(ledger, row, verdict="accept", reviewer="two")
    assert row["review_state"] == "accepted" and len(row["reviews"]) == 3
    row = review(ledger, row, verdict="dispute", reviewer="three")
    assert row["review_state"] == "disputed"
    row = review(ledger, row, verdict="reject", reviewer="one")
    row = review(ledger, row, verdict="reject", reviewer="two")
    row = review(ledger, row, verdict="reject", reviewer="three")
    assert row["review_state"] == "rejected"
    refreshed = ledger.api.project_finding_update(ledger.bridge, row["id"], claim(ledger), row["revision"], "analyst")
    assert refreshed["review_state"] == "proposed" and refreshed["reviews"] == []


def test_active_acceptance_evidence_changes_stale_and_can_be_rereviewed(ledger):
    row = review(ledger, add(ledger))
    Path(ledger.attachment.path_on_disk).write_bytes(b"new independent proof")
    checked = ledger.api.project_finding_check(ledger.bridge, row["id"])
    assert checked["review_state"] == "accepted" and checked["freshness"] == "stale"
    assert any("acceptance evidence" in reason for reason in checked["reasons"])
    row = review(ledger, row)
    assert row["freshness"] == "current" and len(row["reviews"]) == 2
    # Only the reviewer's latest vote is active: withdrawn acceptance is not a freshness input.
    row = review(ledger, row, verdict="reject", evidence=[{"check": "note", "text": "Withdrawn"}])
    Path(ledger.attachment.path_on_disk).write_bytes(b"changed again")
    assert ledger.api.project_finding_check(ledger.bridge, row["id"])["freshness"] == "unknown"


@pytest.mark.parametrize("accepted_first", [False, True])
@pytest.mark.parametrize("verdict", ["reject", "dispute"])
def test_active_counterevidence_changes_make_freshness_stale(ledger, accepted_first, verdict):
    row = add(ledger)
    source_evidence = [{"check": "file", "file": ledger.source.id}]
    if accepted_first:
        row = review(ledger, row, reviewer="supporter", evidence=source_evidence)
    row = review(ledger, row, verdict=verdict, reviewer="critic")
    state = "disputed" if accepted_first or verdict == "dispute" else "rejected"
    assert row["review_state"] == state and row["freshness"] == "current"

    Path(ledger.attachment.path_on_disk).write_bytes(b"changed counterevidence")
    checked = ledger.api.project_finding_check(ledger.bridge, row["id"])
    assert checked["review_state"] == state and checked["freshness"] == "stale"
    assert next(check for check in checked["checks"] if check["source"] == "claim evidence")["freshness"] == "current"

    # Prior counterevidence cannot block a new independently current acceptance,
    # but still participates in the disputed record's displayed freshness.
    row = review(ledger, row, reviewer="supporter", evidence=source_evidence)
    assert row["review_state"] == "disputed" and row["freshness"] == "stale"


@pytest.mark.parametrize("verdict", ["reject", "dispute"])
def test_note_only_counterevidence_is_unknown_without_blocking_acceptance(ledger, verdict):
    row = review(ledger, add(ledger), verdict=verdict, reviewer="critic",
                 evidence=[{"check": "note", "text": "Interpretation is unsupported"}])
    assert row["review_state"] == ("rejected" if verdict == "reject" else "disputed")
    assert row["freshness"] == "unknown"
    row = review(ledger, row, reviewer="supporter")
    assert row["review_state"] == "disputed" and row["freshness"] == "unknown"


def test_changed_claim_evidence_stays_stale_without_rebasing_on_review(ledger):
    row = add(ledger)
    baseline = copy.deepcopy(row["evidence"])
    Path(ledger.source.path_on_disk).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="current claim"):
        review(ledger, row)
    rejected = review(ledger, row, verdict="reject")
    assert rejected["evidence"] == baseline
    assert rejected["freshness"] == "stale"
    updated = ledger.api.project_finding_update(ledger.bridge, row["id"], claim(ledger), 2, "analyst")
    assert updated["freshness"] == "current" and updated["evidence"] != baseline


def test_missing_scope_id_does_not_remap_to_same_name(ledger):
    row = add(ledger, evidence=[{"check": "file", "file": ledger.attachment.id}])
    ledger.project.delete_file(ledger.source)
    replacement = ledger.project.add_file("firmware", data=b"input bytes")
    assert replacement.id != ledger.source.id
    checked = ledger.api.project_finding_check(ledger.bridge, row["id"])
    assert checked["freshness"] == "stale"
    assert any(ledger.source.id in reason for reason in checked["reasons"])


def test_dependencies_require_latest_accepted_and_transitive_current(ledger):
    base = add(ledger)
    dependent = add(ledger, dependencies=[{"id": base["id"], "revision": 1}])
    assert dependent["freshness"] == "unknown"
    with pytest.raises(RuntimeError, match="dependencies"):
        review(ledger, dependent)
    base = review(ledger, base)
    assert ledger.api.project_finding_check(ledger.bridge, dependent["id"])["freshness"] == "stale"
    dependent = ledger.api.project_finding_update(ledger.bridge, dependent["id"], claim(ledger, dependencies=[{"id": base["id"], "revision": 2}]), 1, "analyst")
    dependent = review(ledger, dependent)
    top = add(ledger, dependencies=[{"id": dependent["id"], "revision": dependent["revision"]}])
    assert top["freshness"] == "current"
    Path(ledger.attachment.path_on_disk).write_bytes(b"changed accepted reproduction")
    assert ledger.api.project_finding_check(ledger.bridge, top["id"])["freshness"] == "stale"
    with pytest.raises(RuntimeError, match="exact latest"):
        add(ledger, dependencies=[{"id": base["id"], "revision": 1}])


def test_dependency_cycle_update_and_self_dependency_are_refused(ledger):
    first = add(ledger)
    second = add(ledger, dependencies=[{"id": first["id"], "revision": 1}])
    with pytest.raises(RuntimeError, match="cycle"):
        ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger, dependencies=[{"id": second["id"], "revision": 1}]), 1, "analyst")
    with pytest.raises(RuntimeError, match="Self"):
        ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger, dependencies=[{"id": first["id"], "revision": 1}]), 1, "analyst")
    assert len(snapshots(ledger)) == 2


@pytest.mark.parametrize("verdict", ["reject", "dispute"])
def test_unusable_dependency_status_is_stale(ledger, verdict):
    base = review(ledger, add(ledger), verdict=verdict)
    dependent = add(ledger, dependencies=[{"id": base["id"], "revision": base["revision"]}])
    assert dependent["freshness"] == "stale"
    assert any(verdict in reason for reason in dependent["freshness_reasons"])


def test_supersede_is_terminal_and_replacement_cycles_are_refused(ledger):
    first, replacement = add(ledger), add(ledger, claim="Corrected successor")
    superseded = review(ledger, first, verdict="supersede", reviewer="analyst", replacement=replacement["id"], evidence=[])
    assert superseded["review_state"] == "superseded" and superseded["replacement"] == replacement["id"]
    with pytest.raises(RuntimeError, match="Superseded"):
        ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger), 2, "analyst")
    with pytest.raises(RuntimeError, match="Superseded"):
        review(ledger, superseded, verdict="reject")
    with pytest.raises(RuntimeError, match="cycle"):
        review(ledger, replacement, verdict="supersede", replacement=first["id"], evidence=[])
    dependent = add(ledger, dependencies=[{"id": first["id"], "revision": 2}])
    assert dependent["freshness"] == "stale"


def test_supersession_records_correction_without_endorsing_replacement(ledger):
    first = review(ledger, review(ledger, add(ledger)), reviewer="another-peer")
    replacement = review(ledger, add(ledger, claim="Alternative interpretation"), verdict="reject")
    superseded = review(
        ledger, first, verdict="supersede", reviewer="analyst",
        replacement=replacement["id"], evidence=[])
    context = ledger.api.project_finding_context(ledger.bridge)["groups"]
    assert context["accepted"] == []
    assert [row["id"] for row in context["superseded"]] == [first["id"]]
    assert [row["id"] for row in context["rejected"]] == [replacement["id"]]
    assert superseded["reviews"][:2] == first["reviews"]
    assert superseded["replacement"] == replacement["id"]
    assert ledger.api.project_finding_show(
        ledger.bridge, first["id"], first["revision"])["review_state"] == "accepted"


def test_literal_dot_folder_does_not_alias_reserved_finding_history(ledger):
    first = add(ledger)
    native_dot = ledger.project.add_folder(".")
    lookalike = ledger.project.add_folder(".bn-findings", native_dot)
    finding_folder = ledger.project.add_folder(first["id"], lookalike)
    unrelated = ledger.project.add_file("000001.json", finding_folder, data=b"not a snapshot")
    assert unrelated.get_path_in_project() == f"/./.bn-findings/{first['id']}/000001.json"
    assert ledger.api.project_finding_show(ledger.bridge, first["id"])["integrity"] == first["integrity"]
    assert [row["id"] for row in ledger.api.project_finding_list(ledger.bridge)["items"]] == [first["id"]]



@pytest.mark.parametrize("changes", [{"rationale": ""}, {"evidence": []}, {"replacement": "F-" + "0" * 32}, {"unknown": True}])
def test_invalid_review_documents_never_append(ledger, changes):
    row = add(ledger)
    with pytest.raises(RuntimeError):
        review(ledger, row, **changes)
    assert len(snapshots(ledger)) == 1


def test_discovery_filters_scopes_evidence_types_search_and_pagination(ledger):
    behavior = add(ledger)
    vulnerability = add(ledger, type="vulnerability", scope=[{"file": ledger.source.id, "function": 4096}])
    custom = add(ledger, type="custom:packet-shape", scope=[{"file": ledger.attachment.id}],
                 evidence=[{"check": "bytes", "file": ledger.source.id, "offset": 0, "length": 5}])
    result = ledger.api.project_finding_list(ledger.bridge, type=["vulnerability", "custom:packet-shape"])
    assert {r["id"] for r in result["items"]} == {vulnerability["id"], custom["id"]}
    result = ledger.api.project_finding_list(ledger.bridge, file=ledger.source.id, function="0x1000")
    assert [r["id"] for r in result["items"]] == [vulnerability["id"]]
    result = ledger.api.project_finding_list(ledger.bridge, function="4096")
    assert [r["id"] for r in result["items"]] == [vulnerability["id"]]
    result = ledger.api.project_finding_list(ledger.bridge, file=ledger.source.id)
    assert result["total"] == 3  # Includes evidence-file identity, not just scope.
    result = ledger.api.project_finding_list(ledger.bridge, query="INTERPRETATION")
    assert result["total"] == 3
    result = ledger.api.project_finding_list(ledger.bridge, function="PARSE_INPUT")
    assert [r["id"] for r in result["items"]] == [behavior["id"]]
    page = ledger.api.project_finding_list(ledger.bridge, offset=1, limit=1)
    assert page["total"] == 3 and page["has_more"] and len(page["items"]) == 1
    types = {r["type"]: r for r in ledger.api.project_finding_types(ledger.bridge)["types"]}
    assert types["behavior"]["count"] == 1 and types["custom:packet-shape"]["count"] == 1
    assert types["dataflow"]["count"] == 0 and types["vulnerability"]["action"]


def test_context_groups_are_disjoint_and_keep_rejected_labels(ledger):
    accepted = review(ledger, add(ledger))
    proposed = add(ledger)
    unknown = add(ledger, evidence=[{"check": "note", "text": "Investigate later"}])
    rejected = review(ledger, add(ledger, evidence=[{"check": "note", "text": "Unsupported"}]), verdict="reject")
    groups = ledger.api.project_finding_context(ledger.bridge)["groups"]
    assert [r["id"] for r in groups["accepted"]] == [accepted["id"]]
    assert [r["id"] for r in groups["proposed"]] == [proposed["id"]]
    assert [r["id"] for r in groups["needs_review"]] == [unknown["id"]]
    assert [r["id"] for r in groups["rejected"]] == [rejected["id"]]
    Path(ledger.source.path_on_disk).write_bytes(b"changed")
    groups = ledger.api.project_finding_context(ledger.bridge)["groups"]
    assert groups["accepted"] == []
    assert {r["id"] for r in groups["needs_review"]} == {accepted["id"], proposed["id"], unknown["id"]}
    all_ids = [r["id"] for rows in groups.values() for r in rows]
    assert len(all_ids) == len(set(all_ids)) == 4
    assert ledger.api.project_finding_list(ledger.bridge, freshness="stale")["total"] == 4


@pytest.mark.parametrize("kwargs", [{"limit": 201}, {"limit": 0}, {"offset": -1}, {"limit": True},
                                    {"type": "behavior"}, {"status": "invented"}, {"freshness": "verified"}])
def test_invalid_discovery_filters_refuse(ledger, kwargs):
    with pytest.raises(RuntimeError):
        ledger.api.project_finding_list(ledger.bridge, **kwargs)


@pytest.mark.parametrize("damage", ["digest", "invalid-json", "missing", "duplicate", "bad-path", "schema", "chain", "oversized", "duplicate-key", "evidence-schema", "scope-binding"])
def test_corrupt_ledger_is_never_silently_ignored(ledger, damage):
    first = add(ledger)
    ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger, claim="second"), 1, "analyst")
    files = snapshots(ledger)
    target = files[1]
    blob = Path(target.path_on_disk)
    if damage == "digest":
        record = json.loads(blob.read_bytes())
        record["claim"] = "tampered"
        blob.write_text(json.dumps(record))
    elif damage == "invalid-json":
        blob.write_text("{")
    elif damage == "missing":
        ledger.project.delete_file(files[0])
    elif damage == "duplicate":
        ledger.project.add_file(target.name, target.folder, data=blob.read_bytes())
    elif damage == "bad-path":
        target.name = "latest.json"
    elif damage == "schema":
        record = json.loads(blob.read_bytes())
        record["unexpected"] = True
        record["integrity"] = ledger.api._digest(record)
        blob.write_text(json.dumps(record))
    elif damage == "chain":
        record = json.loads(blob.read_bytes())
        record["previous_digest"] = "0" * 64
        record["integrity"] = ledger.api._digest(record)
        blob.write_text(json.dumps(record))
    elif damage == "oversized":
        blob.write_bytes(b" " * (256 * 1024 + 1))
    elif damage == "duplicate-key":
        blob.write_text('{"id":"first","id":"second"}')
    elif damage in {"evidence-schema", "scope-binding"}:
        record = json.loads(blob.read_bytes())
        if damage == "evidence-schema":
            record["evidence"][0]["check"] = "bytes"  # Missing offset/length.
        else:
            record["scope_evidence"][0]["file_id"] = ledger.attachment.id
        record["integrity"] = ledger.api._digest(record)
        blob.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="Corrupt findings ledger"):
        ledger.api.project_finding_list(ledger.bridge)
    before = len(ledger.project.files)
    with pytest.raises(RuntimeError, match="Corrupt findings ledger"):
        add(ledger)
    assert len(ledger.project.files) == before


@pytest.mark.parametrize("baseline", ["evidence", "scope_evidence", "review"])
def test_rehashed_incomplete_observation_refuses_loading_and_acceptance(ledger, baseline):
    row = add(ledger)
    if baseline == "review":
        row = review(ledger, row)
    blob = Path(snapshots(ledger)[-1].path_on_disk)
    record = json.loads(blob.read_bytes())
    evidence = record["reviews"][-1]["evidence"] if baseline == "review" else record[baseline]
    captured = evidence[0]
    assert captured["check"] == "file" and captured["capture_state"] == "current"
    assert captured["observed"] and captured["fingerprint"]
    captured["observed"] = {}
    record["integrity"] = ledger.api._digest(record)
    blob.write_text(json.dumps(record))

    before = len(ledger.project.files)
    with pytest.raises(RuntimeError, match="Corrupt findings ledger"):
        ledger.api.project_finding_show(ledger.bridge, row["id"])
    with pytest.raises(RuntimeError, match="Corrupt findings ledger"):
        ledger.api.project_finding_check(ledger.bridge, row["id"])
    with pytest.raises(RuntimeError, match="Corrupt findings ledger"):
        review(ledger, row, reviewer="another-peer")
    assert len(ledger.project.files) == before


def test_rehashed_review_cannot_rewrite_claim_or_prior_reviews(ledger):
    first = review(ledger, add(ledger))
    target = snapshots(ledger)[-1]
    record = json.loads(Path(target.path_on_disk).read_bytes())
    record["claim"] = "silently rewritten in review"
    record["integrity"] = ledger.api._digest(record)
    Path(target.path_on_disk).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="rewrites the claim"):
        ledger.api.project_finding_show(ledger.bridge, first["id"])


def test_readback_failure_deletes_only_new_snapshot(ledger, monkeypatch):
    first = add(ledger)
    original = snapshots(ledger)[0]
    old_bytes = Path(original.path_on_disk).read_bytes()
    create = ledger.project.create_file

    def broken(contents, folder, name, description=""):
        return create(b"truncated", folder, name, description)

    monkeypatch.setattr(ledger.project, "create_file", broken)
    with pytest.raises(RuntimeError, match="readback.*rolled back"):
        ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger), 1, "analyst")
    assert snapshots(ledger) == [original]
    assert Path(original.path_on_disk).read_bytes() == old_bytes
    assert len(ledger.project.deleted) == 1
    assert ledger.api.project_finding_show(ledger.bridge, first["id"])["revision"] == 1


def test_create_then_raise_rolls_back_registered_file(ledger, monkeypatch):
    create = ledger.project.create_file

    def broken(*args):
        create(*args)
        raise OSError("native completion failure")

    monkeypatch.setattr(ledger.project, "create_file", broken)
    with pytest.raises(RuntimeError, match="native completion failure.*rolled back"):
        add(ledger)
    assert snapshots(ledger) == [] and len(ledger.project.deleted) == 1
    # Empty namespace folders are harmless after rollback.
    assert ledger.api.project_finding_list(ledger.bridge)["items"] == []


@pytest.mark.parametrize("rollback", ["false", "raise", "lies"])
@pytest.mark.parametrize("action", ["add", "update", "review"])
def test_rollback_failure_is_explicit(ledger, monkeypatch, rollback, action):
    first = add(ledger) if action != "add" else None
    originals = [(pf, Path(pf.path_on_disk).read_bytes()) for pf in snapshots(ledger)]
    create = ledger.project.create_file
    monkeypatch.setattr(ledger.project, "create_file", lambda contents, folder, name, description: create(b"bad", folder, name, description))

    def delete(_):
        if rollback == "raise":
            raise OSError("locked")
        return rollback == "lies"

    monkeypatch.setattr(ledger.project, "delete_file", delete)
    with pytest.raises(ledger.api.OperationFailure, match="STORAGE ROLLBACK FAILED") as error:
        if action == "add":
            add(ledger)
        elif action == "update":
            ledger.api.project_finding_update(ledger.bridge, first["id"], claim(ledger), 1, "analyst")
        else:
            review(ledger, first)
    retained = snapshots(ledger)
    assert len(retained) == len(originals) + 1
    for pf, data in originals:
        assert pf in retained and Path(pf.path_on_disk).read_bytes() == data
    leftover = retained[-1]
    assert error.value.status == "rollback_failed"
    assert error.value.requested["snapshot_path"] == leftover.get_path_in_project()
    assert error.value.requested["revision"] == (2 if first else 1)
    assert error.value.requested["finding_id"] in leftover.get_path_in_project()
    assert error.value.observed["rolled_back"] is False
    assert error.value.observed["dirty_after"] is True
    assert error.value.observed["candidate_file_ids"] == [leftover.id]
    assert "readback" in error.value.observed["append_error"]
    assert error.value.observed["rollback_error"]


def test_structural_and_snapshot_limits_do_not_write(ledger):
    with pytest.raises(RuntimeError, match="256"):
        add(ledger, scope=[{"file": ledger.source.id}] * 257)
    with pytest.raises(RuntimeError, match="256 KiB"):
        add(ledger, assumptions=["x" * 32768] * 9)
    nested = []
    for _ in range(30):
        nested = [nested]
    with pytest.raises(RuntimeError, match="structural"):
        add(ledger, assumptions=nested)
    assert snapshots(ledger) == []


def test_check_is_read_only_and_native_recheck_error_is_unknown(ledger, monkeypatch):
    row = add(ledger)
    before = [(pf.id, Path(pf.path_on_disk).read_bytes()) for pf in snapshots(ledger)]

    def unavailable(*_):
        raise OSError("view unavailable")

    monkeypatch.setattr(ledger.api, "check_evidence", unavailable)
    checked = ledger.api.project_finding_check(ledger.bridge, row["id"])
    assert checked["freshness"] == "unknown"
    assert any("view unavailable" in reason for reason in checked["reasons"])
    after = [(pf.id, Path(pf.path_on_disk).read_bytes()) for pf in snapshots(ledger)]
    assert before == after


def test_artifact_claim_remains_bound_to_unchanged_scoped_source(ledger):
    row = add(ledger, evidence=[{"check": "file", "file": ledger.attachment.id}])
    row = review(ledger, row)
    scope_baseline = copy.deepcopy(row["scope_evidence"])
    assert scope_baseline[0]["file_id"] == ledger.source.id
    Path(ledger.source.path_on_disk).write_bytes(b"changed scoped firmware")
    shown = ledger.api.project_finding_show(ledger.bridge, row["id"])
    assert shown["review_state"] == "accepted" and shown["freshness"] == "stale"
    assert shown["scope_evidence"] == scope_baseline
    assert any("scope evidence" in reason for reason in shown["freshness_reasons"])
    with pytest.raises(RuntimeError, match="unknown"):
        add(ledger, scope_evidence=[])


def test_dependency_diamond_rechecks_each_record_once(ledger, monkeypatch):
    base = review(ledger, add(ledger))
    left = review(ledger, add(ledger, dependencies=[{"id": base["id"], "revision": base["revision"]}]))
    right = review(ledger, add(ledger, dependencies=[{"id": base["id"], "revision": base["revision"]}]))
    top = add(ledger, dependencies=[{"id": row["id"], "revision": row["revision"]} for row in (left, right)])
    calls = []
    original = ledger.api.check_evidence

    def counted(bridge, evidence):
        calls.append(evidence)
        return original(bridge, evidence)

    monkeypatch.setattr(ledger.api, "check_evidence", counted)
    checked = ledger.api.project_finding_check(ledger.bridge, top["id"])
    assert checked["freshness"] == "current"
    assert len(calls) == 11  # Three accepted records * 3 checks, proposed top * 2.
    assert len(checked["checks"]) == 4  # Claim, scope, and two bounded dependency summaries.


def test_rehashed_dependency_cycle_is_corrupt_not_recursive(ledger):
    first, second = add(ledger), add(ledger)
    for pf, dependency in zip(snapshots(ledger), (second, first)):
        record = json.loads(Path(pf.path_on_disk).read_bytes())
        record["dependencies"] = [{"id": dependency["id"], "revision": dependency["revision"]}]
        record["integrity"] = ledger.api._digest(record)
        Path(pf.path_on_disk).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="Corrupt findings ledger: Dependency cycle"):
        ledger.api.project_finding_check(ledger.bridge, first["id"])


def test_multiple_scopes_are_deduplicated_without_upgrading_notes(ledger):
    row = add(ledger, scope=[{"file": ledger.source.id}, {"file": ledger.source.id, "function": 4096},
                            {"file": ledger.attachment.id}],
              evidence=[{"check": "note", "text": "Manual interpretation only"}])
    assert row["freshness"] == "unknown"
    assert {e["file_id"] for e in row["scope_evidence"]} == {ledger.source.id, ledger.attachment.id}
    assert len(row["scope_evidence"]) == 2
    Path(ledger.attachment.path_on_disk).write_bytes(b"changed other scoped file")
    assert ledger.api.project_finding_check(ledger.bridge, row["id"])["freshness"] == "stale"
