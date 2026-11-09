"""Immutable, project-owned finding snapshots; freshness is not proof of a claim.

The namespace is the index. Every request reconstructs and validates its hash
chains before reading or appending, and only the newly created blob is eligible
for rollback. Caller/dispatcher project locks serialize mutations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from ._shared import OperationFailure
from .project_finding_evidence import capture_evidence, check_evidence, validate_captured_evidence
from .project_ops import FINDINGS_FOLDER

MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_ENTRIES = 256
NAMESPACE = "/" + FINDINGS_FOLDER
_ID = re.compile(r"F-[0-9a-f]{32}\Z")
_CUSTOM_TYPE = re.compile(r"custom:[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STATES = {"proposed", "accepted", "rejected", "disputed", "superseded"}
_FRESHNESS = {"current", "stale", "unknown"}
_TYPES = {
    "vulnerability": ("Potential security vulnerability", "triage/independently validate"),
    "dataflow": ("Source-to-sink data flow", "reuse/verify source-sink path"),
    "function-role": ("Function responsibility", "reuse or refine role"),
    "data-structure": ("Data structure or layout", "reuse/verify layout"),
    "protocol": ("Protocol behavior or constraints", "reuse/verify protocol constraints"),
    "file-format": ("File format or encoding", "reuse/verify format"),
    "dependency": ("Provider or linking dependency", "verify provider/link assumptions"),
    "behavior": ("Observed program behavior", "reuse/verify observation"),
    "dead-end": ("Investigated path without a useful result", "avoid repeating until assumptions change"),
    "open-question": ("Unresolved investigation question", "investigate"),
}
_CLAIM_KEYS = {"type", "claim", "scope", "evidence", "kind", "assumptions", "limitations", "next_steps", "dependencies", "tags"}
_REQUIRED_CLAIM = {"type", "claim", "scope", "evidence", "kind"}
_RECORD_KEYS = _CLAIM_KEYS | {
    "schema_version", "project_id", "id", "revision", "claim_revision",
    "previous_digest", "integrity", "author", "created_at", "updated_at",
    "action", "reviews", "review_state", "scope_evidence",
}
_REVIEW_KEYS = {"reviewer", "verdict", "rationale", "evidence", "reviewed_revision", "claim_revision", "timestamp"}


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeError(f"Finding must be a finite JSON document: {exc}") from exc


def _bounded(value: Any) -> None:
    # Bound traversal before serialization, including deeply nested hostile JSON.
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 20 or nodes > 20000:
            raise RuntimeError("Finding document exceeds structural limits")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise RuntimeError("Finding JSON object keys must be strings")
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise RuntimeError("Finding document must contain only JSON values")
    if len(_canonical(value)) > MAX_SNAPSHOT_BYTES:
        raise RuntimeError("Finding snapshot exceeds 256 KiB")


def _object(value: Any, allowed: set[str], required: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise RuntimeError(f"Invalid {label} fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return value


def _text(value: Any, label: str, maximum: int = 32768) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise RuntimeError(f"{label} must be a nonempty string of at most {maximum} characters")
    return value.strip()


def _same_label(a: str, b: str) -> bool:
    """Do two declared identity labels spell the same name?

    A typo guard for the self-accept refusal, NOT authentication (#1074 item
    1): `Analyst`, ` analyst ` and a fullwidth spelling are one label for the
    purpose of "the reviewer must differ from the author", because a declared
    label proves nothing about who typed it either way. Compared after NFKC
    normalisation, case folding, trimming and collapsing internal whitespace;
    the labels themselves are stored exactly as declared."""
    def fold(label: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", label).casefold().split())
    return fold(a) == fold(b)


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list) or len(value) > MAX_ENTRIES or (nonempty and not value):
        raise RuntimeError(f"{label} must be {'a nonempty' if nonempty else 'an'} array of at most {MAX_ENTRIES} entries")
    return value


def _finding_id(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise RuntimeError("Finding id must be an exact F-uuidhex identifier")
    return value


def _type(value: Any) -> str:
    name = _text(value, "type", 128)
    if name not in _TYPES and not _CUSTOM_TYPE.fullmatch(name):
        raise RuntimeError("Unknown finding type; use a standard type or custom:<lowercase-kebab-slug>")
    return name


def _function(value: Any) -> str | int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str):
        return _text(value, "function", 1024)
    raise RuntimeError("function must be a nonnegative numeric address or nonempty label")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise RuntimeError("Invalid snapshot timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ValueError("not UTC")
    except ValueError as exc:
        raise RuntimeError("Invalid snapshot UTC timestamp") from exc


def _digest(record: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in record.items() if k != "integrity"})).hexdigest()


def _active_reviews(record: dict) -> list[dict]:
    return list({r["reviewer"]: r for r in record["reviews"]}.values())


def _review_state(reviews: list[dict]) -> str:
    votes = {r["verdict"] for r in {r["reviewer"]: r for r in reviews}.values()}
    if "supersede" in votes:
        return "superseded"
    if "dispute" in votes or {"accept", "reject"} <= votes:
        return "disputed"
    if "reject" in votes:
        return "rejected"
    return "accepted" if "accept" in votes else "proposed"


def _captured_schema(evidence: Any, *, nonempty: bool = True) -> None:
    captured = _array(evidence, "captured evidence", nonempty=nonempty)
    # Supersession may have no evidence; other callers require a nonempty list.
    # Request/observation validation belongs to the evidence module alone.
    if captured:
        validate_captured_evidence(captured)


def _record_schema(record: Any, project_id: str, finding_id: str, revision: int) -> None:
    _bounded(record)
    _object(record, _RECORD_KEYS | {"replacement"}, _RECORD_KEYS, "snapshot")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise RuntimeError("Unsupported finding schema_version")
    if record["project_id"] != project_id or record["id"] != finding_id or record["revision"] != revision:
        raise RuntimeError("Snapshot identity does not match its project/path")
    _integer(record["revision"], "revision")
    if _integer(record["claim_revision"], "claim_revision") > revision:
        raise RuntimeError("Invalid claim_revision")
    _type(record["type"])
    _text(record["claim"], "claim")
    _text(record["author"], "author", 256)
    if record["kind"] not in ("observation", "hypothesis", "conclusion"):
        raise RuntimeError("Invalid finding kind")
    for field in ("assumptions", "limitations", "next_steps", "tags"):
        for item in _array(record[field], field):
            _text(item, field)
    for scope in _array(record["scope"], "scope", nonempty=True):
        _object(scope, {"file_id", "file_path", "function"}, {"file_id", "file_path"}, "scope")
        _text(scope["file_id"], "scope file_id", 1024)
        _text(scope["file_path"], "scope file_path", 4096)
        if "function" in scope:
            _function(scope["function"])
    _captured_schema(record["evidence"])
    _captured_schema(record["scope_evidence"])
    scope_ids = {item["file_id"] for item in record["scope"]}
    source_ids = [item.get("file_id") for item in record["scope_evidence"]]
    if (any(item["check"] != "file" for item in record["scope_evidence"])
            or len(source_ids) != len(set(source_ids)) or set(source_ids) != scope_ids):
        raise RuntimeError("Scope evidence must bind each exact scoped file")
    dependencies = set()
    for dep in _array(record["dependencies"], "dependencies"):
        _object(dep, {"id", "revision"}, {"id", "revision"}, "dependency")
        _finding_id(dep["id"])
        _integer(dep["revision"], "dependency revision")
        if dep["id"] in dependencies or dep["id"] == finding_id:
            raise RuntimeError("Duplicate or self dependency")
        dependencies.add(dep["id"])
    for key in ("created_at", "updated_at"):
        _timestamp(record[key])
    action = _object(record["action"], {"kind", "actor", "timestamp"}, {"kind", "actor", "timestamp"}, "action")
    if action["kind"] not in ("add", "update", "review"):
        raise RuntimeError("Invalid action kind")
    _text(action["actor"], "action actor", 256)
    if action["timestamp"] != record["updated_at"]:
        raise RuntimeError("Action timestamp does not match snapshot")
    for review in _array(record["reviews"], "reviews"):
        _object(review, _REVIEW_KEYS | {"replacement"}, _REVIEW_KEYS, "review")
        reviewer = _text(review["reviewer"], "reviewer", 256)
        verdict = review["verdict"]
        if verdict not in ("accept", "reject", "dispute", "supersede"):
            raise RuntimeError("Invalid review verdict")
        _text(review["rationale"], "rationale")
        _timestamp(review["timestamp"])
        if _integer(review["reviewed_revision"], "reviewed_revision") >= revision or review["claim_revision"] != record["claim_revision"]:
            raise RuntimeError("Invalid review revision")
        _captured_schema(review["evidence"], nonempty=verdict != "supersede")
        # Exact, not `_same_label`: this validates records already on disk, and
        # a ledger written before the typo guard may hold an accept whose
        # labels differ only in case. Tightening here would turn such history
        # into an integrity failure; the guard belongs at the write.
        if verdict == "accept" and reviewer == record["author"]:
            raise RuntimeError("Self acceptance is forbidden")
        if verdict == "supersede":
            if _finding_id(review.get("replacement")) == finding_id:
                raise RuntimeError("Self replacement is forbidden")
        elif "replacement" in review:
            raise RuntimeError("Only supersede may name a replacement")
    if record["review_state"] != _review_state(record["reviews"]):
        raise RuntimeError("Snapshot review_state disagrees with active reviews")
    if record["review_state"] == "superseded":
        if not record["reviews"] or record.get("replacement") != record["reviews"][-1].get("replacement"):
            raise RuntimeError("Invalid replacement snapshot")
    elif "replacement" in record:
        raise RuntimeError("Only superseded findings may name replacements")
    if not isinstance(record["integrity"], str) or record["integrity"] != _digest(record):
        raise RuntimeError("Snapshot integrity digest mismatch")


def _transition(previous: dict | None, record: dict) -> None:
    action = record["action"]
    if previous is None:
        if record["revision"] != 1 or record["claim_revision"] != 1 or record["previous_digest"] is not None or action["kind"] != "add":
            raise RuntimeError("Invalid initial snapshot")
        if record["reviews"] or record["created_at"] != record["updated_at"] or action["actor"] != record["author"]:
            raise RuntimeError("Invalid initial provenance")
        return
    if record["previous_digest"] != previous["integrity"]:
        raise RuntimeError("Snapshot hash chain mismatch")
    if previous["review_state"] == "superseded":
        raise RuntimeError("Snapshot follows a terminal superseded finding")
    if record["created_at"] != previous["created_at"]:
        raise RuntimeError("Snapshot rewrites creation time")
    if action["kind"] == "update":
        if record["claim_revision"] != record["revision"] or record["reviews"] or action["actor"] != record["author"]:
            raise RuntimeError("Invalid claim update provenance")
    elif action["kind"] == "review":
        for field in _CLAIM_KEYS | {"author", "claim_revision", "scope_evidence"}:
            if record[field] != previous[field]:
                raise RuntimeError("Review snapshot rewrites the claim")
        if len(record["reviews"]) != len(previous["reviews"]) + 1 or record["reviews"][:-1] != previous["reviews"]:
            raise RuntimeError("Review snapshot rewrites prior reviews")
        review = record["reviews"][-1]
        if review["reviewed_revision"] != previous["revision"] or review["reviewer"] != action["actor"] or review["timestamp"] != action["timestamp"]:
            raise RuntimeError("Invalid review action provenance")
    else:
        raise RuntimeError("Only the initial snapshot may have add action")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _acyclic(graph: dict[str, list[str]], label: str) -> None:
    done: set[str] = set()
    for root in graph:
        if root in done:
            continue
        active: set[str] = set()
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                active.remove(node)
                done.add(node)
                continue
            if node in active:
                raise RuntimeError(f"{label} cycle involving {node}")
            if node in done:
                continue
            active.add(node)
            stack.append((node, True))
            stack.extend((child, False) for child in graph.get(node, []))


class _Ledger:
    def __init__(self, bridge: Any):
        self.bridge = bridge
        self.manager = bridge.projects
        self.state = self.manager.require()
        self.project = self.state.project
        self.files = self.manager.files()
        self.file_ids = {str(pf.id): pf for pf in self.files}
        self.records: dict[str, list[dict]] = {}
        self.latest: dict[str, dict] = {}
        self.checked: dict[tuple[str, int], dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            paths: set[str] = set()
            for folder in self.project.folders:
                path = self.manager.folder_path(folder)
                if path == NAMESPACE or path.startswith(NAMESPACE + "/"):
                    if path in paths:
                        raise RuntimeError(f"Duplicate ledger folder {path}")
                    paths.add(path)
                    suffix = path[len(NAMESPACE):].strip("/")
                    if suffix and not _ID.fullmatch(suffix):
                        raise RuntimeError(f"Malformed ledger folder {path}")
            revisions: dict[str, dict[int, dict]] = {}
            for pf in self.files:
                path = self.manager.path_in_project(pf)
                if path != NAMESPACE and not path.startswith(NAMESPACE + "/"):
                    continue
                match = re.fullmatch(r"/\.bn-findings/(F-[0-9a-f]{32})/([0-9]{6,})\.json", path)
                if not match:
                    raise RuntimeError(f"Malformed ledger path {path}")
                finding_id, number = match.groups()
                revision = int(number)
                if revision < 1 or number != f"{revision:06d}":
                    raise RuntimeError(f"Malformed revision filename {path}")
                bucket = revisions.setdefault(finding_id, {})
                if revision in bucket:
                    raise RuntimeError(f"Duplicate finding revision {path}")
                with open(str(pf.path_on_disk), "rb") as handle:
                    data = handle.read(MAX_SNAPSHOT_BYTES + 1)
                if len(data) > MAX_SNAPSHOT_BYTES:
                    raise RuntimeError(f"Oversized snapshot {path}")
                record = json.loads(data, object_pairs_hook=_no_duplicate_keys)
                _record_schema(record, str(self.state.project_id), finding_id, revision)
                bucket[revision] = record
            for finding_id, bucket in revisions.items():
                ordered = sorted(bucket)
                if ordered != list(range(1, len(ordered) + 1)):
                    raise RuntimeError(f"Missing revision in {finding_id}")
                chain = [bucket[n] for n in ordered]
                previous = None
                for record in chain:
                    _transition(previous, record)
                    previous = record
                self.records[finding_id] = chain
                self.latest[finding_id] = chain[-1]
            # References must name real snapshots even if they are now stale.
            for chain in self.records.values():
                for record in chain:
                    for dep in record["dependencies"]:
                        if dep["id"] not in self.records or dep["revision"] > len(self.records[dep["id"]]):
                            raise RuntimeError("Dependency refers to an absent snapshot")
                    if "replacement" in record and record["replacement"] not in self.latest:
                        raise RuntimeError("Replacement finding is missing")
            _acyclic({key: [d["id"] for d in row["dependencies"]] for key, row in self.latest.items()}, "Dependency")
            _acyclic({key: [row["replacement"]] if "replacement" in row else [] for key, row in self.latest.items()}, "Replacement")
        except Exception as exc:
            raise RuntimeError(f"Corrupt findings ledger: {exc}") from exc

    def get(self, finding_id: Any, revision: Any = None) -> dict:
        finding_id = _finding_id(finding_id)
        if finding_id not in self.records:
            raise RuntimeError(f"No finding {finding_id}")
        if revision is None:
            return self.latest[finding_id]
        revision = _integer(revision, "revision")
        if revision > len(self.records[finding_id]):
            raise RuntimeError(f"No revision {revision} for {finding_id}")
        return self.records[finding_id][revision - 1]

    def mutation(self, finding_id: Any, expected_revision: Any) -> dict:
        expected_revision = _integer(expected_revision, "expected_revision")
        current = self.get(finding_id)
        if current["revision"] != expected_revision:
            raise RuntimeError(f"Finding revision conflict: expected {expected_revision}, current revision {current['revision']}")
        if current["review_state"] == "superseded":
            raise RuntimeError("Superseded findings are immutable; add a corrected successor")
        return current

    def claim(self, document: Any, finding_id: str) -> dict:
        _bounded(document)
        document = _object(document, _CLAIM_KEYS, _REQUIRED_CLAIM, "finding document")
        result = {"type": _type(document["type"]), "claim": _text(document["claim"], "claim")}
        if document["kind"] not in ("observation", "hypothesis", "conclusion"):
            raise RuntimeError("kind must be observation, hypothesis, or conclusion")
        result["kind"] = document["kind"]
        for field in ("assumptions", "limitations", "next_steps", "tags"):
            result[field] = [_text(item, field) for item in _array(document.get(field, []), field)]
        scope = []
        for item in _array(document["scope"], "scope", nonempty=True):
            _object(item, {"file", "function"}, {"file"}, "scope")
            pf = self.manager.resolve_file(item["file"])
            path = self.manager.path_in_project(pf)
            if path == NAMESPACE or path.startswith(NAMESPACE + "/"):
                raise RuntimeError("Finding scope cannot be an internal ledger snapshot")
            normalized = {"file_id": str(pf.id), "file_path": path}
            if "function" in item:
                normalized["function"] = _function(item["function"])
            scope.append(normalized)
        result["scope"] = scope
        dependencies = []
        seen = set()
        for dep in _array(document.get("dependencies", []), "dependencies"):
            _object(dep, {"id", "revision"}, {"id", "revision"}, "dependency")
            dep_id = _finding_id(dep["id"])
            revision = _integer(dep["revision"], "dependency revision")
            if dep_id == finding_id or dep_id in seen:
                raise RuntimeError("Self or duplicate dependencies are forbidden")
            if self.get(dep_id)["revision"] != revision:
                raise RuntimeError(f"Dependency {dep_id} must name its exact latest revision")
            seen.add(dep_id)
            dependencies.append({"id": dep_id, "revision": revision})
        graph = {key: [d["id"] for d in row["dependencies"]] for key, row in self.latest.items()}
        graph[finding_id] = [d["id"] for d in dependencies]
        _acyclic(graph, "Dependency")
        result["dependencies"] = dependencies
        inputs = _array(document["evidence"], "evidence", nonempty=True)
        result["evidence"] = capture_evidence(self.bridge, inputs)
        _captured_schema(result["evidence"])
        result["scope_evidence"] = capture_evidence(
            self.bridge, [{"check": "file", "file": file_id} for file_id in dict.fromkeys(item["file_id"] for item in scope)]
        )
        return result

    def _evidence(self, evidence: list[dict]) -> dict:
        try:
            result = check_evidence(self.bridge, evidence)
            if not isinstance(result, dict) or result.get("freshness") not in _FRESHNESS:
                raise RuntimeError("Evidence checker returned an invalid result")
            return result
        except Exception as exc:
            return {"freshness": "unknown", "reasons": [f"Evidence unavailable: {exc}"], "checks": []}

    def _evaluate(self, record: dict, *, include_reviews: bool = True) -> dict:
        states = []
        reasons = []
        checks = []

        def include(result: dict, label: str) -> None:
            states.append(result["freshness"])
            reasons.extend(f"{label}: {reason}" for reason in result.get("reasons", []))
            checks.append({"source": label, **result})

        include(self._evidence(record["evidence"]), "claim evidence")
        # Scope binding can invalidate a claim, but cannot make manual claim
        # evidence machine-checkable: its state remains a separate input above.
        include(self._evidence(record["scope_evidence"]), "scope evidence")
        for scope in record["scope"]:
            if scope["file_id"] not in self.file_ids:
                states.append("stale")
                reasons.append(f"Scope file is missing: {scope['file_id']} ({scope['file_path']})")
        latest = self.latest[record["id"]]
        if latest["revision"] != record["revision"]:
            states.append("stale")
            reasons.append(f"Historical revision; newer revision {latest['revision']} exists")
        if include_reviews:
            for review in _active_reviews(record):
                if review["verdict"] in {"accept", "reject", "dispute"}:
                    label = {"accept": "acceptance", "reject": "rejection", "dispute": "dispute"}[review["verdict"]]
                    include(self._evidence(review["evidence"]), f"{label} evidence from {review['reviewer']}")
        for dep in record["dependencies"]:
            current = self.latest[dep["id"]]
            label = f"Dependency {dep['id']} revision {dep['revision']}"
            if current["revision"] != dep["revision"]:
                states.append("stale")
                reasons.append(f"{label}: newer revision {current['revision']} exists")
                continue
            if current["review_state"] in {"rejected", "disputed", "superseded"}:
                states.append("stale")
                reasons.append(f"{label}: {current['review_state']}")
            elif current["review_state"] != "accepted":
                states.append("unknown")
                reasons.append(f"{label}: not independently accepted")
            dependency_check = self.checked[(current["id"], current["revision"])]
            states.append(dependency_check["freshness"])
            if dependency_check["freshness"] != "current":
                reasons.append(f"{label}: evidence is {dependency_check['freshness']}")
            checks.append({"source": label, "id": current["id"], "revision": current["revision"],
                           "review_state": current["review_state"], "freshness": dependency_check["freshness"]})
        freshness = "stale" if "stale" in states else "unknown" if "unknown" in states else "current"
        return {"freshness": freshness, "reasons": reasons, "checks": checks}

    def check(self, record: dict) -> dict:
        key = (record["id"], record["revision"])
        stack = [(record, False)]
        active = set()
        while stack:
            row, leaving = stack.pop()
            row_key = (row["id"], row["revision"])
            if row_key in self.checked:
                continue
            if leaving:
                active.remove(row_key)
                self.checked[row_key] = self._evaluate(row)
                continue
            if row_key in active:
                raise RuntimeError("Corrupt findings ledger: dependency cycle during freshness check")
            active.add(row_key)
            stack.append((row, True))
            for dep in row["dependencies"]:
                current = self.latest[dep["id"]]
                if current["revision"] == dep["revision"]:
                    stack.append((current, False))
        return self.checked[key]

    def rendered(self, record: dict) -> dict:
        result = copy.deepcopy(record)
        checked = self.check(record)
        result["freshness"] = checked["freshness"]
        result["freshness_reasons"] = list(checked["reasons"])
        return result

    def append(self, record: dict) -> dict:
        record["integrity"] = _digest(record)
        _record_schema(record, str(self.state.project_id), record["id"], record["revision"])
        _transition(self.latest.get(record["id"]), record)
        data = _canonical(record)
        path = f"{NAMESPACE}/{record['id']}/{record['revision']:06d}.json"
        prior_ids = {str(pf.id) for pf in self.files}
        created = None
        try:
            folder = self.manager.ensure_folder(f"{NAMESPACE}/{record['id']}")
            created = self.project.create_file(data, folder, f"{record['revision']:06d}.json", "Immutable finding snapshot")
            if created is None or isinstance(created, bool):
                raise RuntimeError("Native create_file returned no ProjectFile")
            if str(created.id) in prior_ids:
                raise RuntimeError("Native create_file returned an existing file")
            if self.manager.path_in_project(created) != path:
                raise RuntimeError("Native create_file returned an unexpected project path")
            with open(str(created.path_on_disk), "rb") as handle:
                if handle.read(MAX_SNAPSHOT_BYTES + 1) != data:
                    raise RuntimeError("Snapshot readback does not match appended bytes")
            registered = [pf for pf in self.manager.files() if self.manager.path_in_project(pf) == path]
            if len(registered) != 1 or str(registered[0].id) != str(created.id):
                raise RuntimeError("Snapshot registration verification failed")
        except Exception as exc:
            rollback_ids = []
            try:
                # A native exception may occur after registering the new row.
                candidates = [pf for pf in self.manager.files() if str(pf.id) not in prior_ids and self.manager.path_in_project(pf) == path]
                if created is not None and not isinstance(created, bool) and str(created.id) not in prior_ids and all(str(pf.id) != str(created.id) for pf in candidates):
                    candidates.append(created)
                rollback_ids = [str(pf.id) for pf in candidates]
                for pf in candidates:
                    if self.project.delete_file(pf) is not True:
                        raise RuntimeError(f"Native delete_file refused {pf.id}")
                remaining = {str(pf.id) for pf in self.manager.files()}
                if any(str(pf.id) in remaining for pf in candidates):
                    raise RuntimeError("Rolled-back snapshot remains registered")
            except Exception as rollback:
                raise OperationFailure(
                    "rollback_failed",
                    f"Finding append failed: {exc}; STORAGE ROLLBACK FAILED: {rollback}",
                    requested={"finding_id": record["id"], "revision": record["revision"],
                               "snapshot_path": path},
                    observed={"rolled_back": False, "dirty_after": True,
                              "candidate_file_ids": rollback_ids,
                              "append_error": str(exc), "rollback_error": str(rollback)},
                ) from exc
            raise RuntimeError(f"Finding append failed: {exc} (new snapshot rolled back)") from exc
        self.records.setdefault(record["id"], []).append(record)
        self.latest[record["id"]] = record
        self.checked.clear()
        return self.rendered(record)


def _snapshot(ledger: _Ledger, claim: dict, finding_id: str, author: str, previous: dict | None) -> dict:
    timestamp = _now()
    revision = previous["revision"] + 1 if previous else 1
    return {
        **claim, "schema_version": 1, "project_id": str(ledger.state.project_id),
        "id": finding_id, "revision": revision, "claim_revision": revision,
        "previous_digest": previous["integrity"] if previous else None,
        "author": author, "created_at": previous["created_at"] if previous else timestamp,
        "updated_at": timestamp,
        "action": {"kind": "update" if previous else "add", "actor": author, "timestamp": timestamp},
        "reviews": [], "review_state": "proposed",
    }


def project_finding_add(bridge: Any, document: Any, author: Any) -> dict:
    ledger = _Ledger(bridge)
    author = _text(author, "author", 256)
    finding_id = "F-" + uuid.uuid4().hex
    claim = ledger.claim(document, finding_id)
    return ledger.append(_snapshot(ledger, claim, finding_id, author, None))


def project_finding_update(bridge: Any, id: Any, document: Any, expected_revision: Any, author: Any) -> dict:
    ledger = _Ledger(bridge)
    previous = ledger.mutation(id, expected_revision)
    author = _text(author, "author", 256)
    claim = ledger.claim(document, previous["id"])
    return ledger.append(_snapshot(ledger, claim, previous["id"], author, previous))


def project_finding_review(bridge: Any, id: Any, document: Any, expected_revision: Any, reviewer: Any) -> dict:
    ledger = _Ledger(bridge)
    previous = ledger.mutation(id, expected_revision)
    reviewer = _text(reviewer, "reviewer", 256)
    _bounded(document)
    _object(document, {"verdict", "rationale", "evidence", "replacement"}, {"verdict", "rationale", "evidence"}, "review document")
    verdict = document["verdict"]
    if verdict not in ("accept", "reject", "dispute", "supersede"):
        raise RuntimeError("verdict must be accept, reject, dispute, or supersede")
    rationale = _text(document["rationale"], "rationale")
    inputs = _array(document["evidence"], "review evidence", nonempty=verdict != "supersede")
    if verdict == "accept" and _same_label(reviewer, previous["author"]):
        raise RuntimeError("Self acceptance is forbidden: reviewer must differ from claim author")
    replacement = None
    if verdict == "supersede":
        replacement = ledger.get(document.get("replacement"))["id"]
        if replacement == previous["id"]:
            raise RuntimeError("Finding cannot replace itself")
        graph = {key: [row["replacement"]] if "replacement" in row else [] for key, row in ledger.latest.items()}
        graph[previous["id"]] = [replacement]
        _acyclic(graph, "Replacement")
    elif "replacement" in document:
        raise RuntimeError("replacement is permitted only for supersede")
    evidence = capture_evidence(bridge, inputs) if inputs else []
    _captured_schema(evidence, nonempty=verdict != "supersede")
    if verdict == "accept":
        ledger.check(previous)  # Populate transitive dependency checks once.
        freshness = ledger._evaluate(previous, include_reviews=False)
        if freshness["freshness"] != "current":
            raise RuntimeError(f"Acceptance requires current claim and accepted/current exact dependencies: {freshness['freshness']}; {'; '.join(freshness['reasons'])}")
        review_check = ledger._evidence(evidence)
        if review_check["freshness"] != "current" or not any(item["check"] != "note" for item in evidence):
            raise RuntimeError("Acceptance requires fresh machine-checkable independent review evidence")
    timestamp = _now()
    review = {
        "reviewer": reviewer, "verdict": verdict, "rationale": rationale,
        "evidence": evidence, "reviewed_revision": previous["revision"],
        "claim_revision": previous["claim_revision"], "timestamp": timestamp,
    }
    record = copy.deepcopy(previous)
    if replacement is not None:
        review["replacement"] = replacement
        record["replacement"] = replacement
    record["reviews"].append(review)
    record.update(revision=previous["revision"] + 1, previous_digest=previous["integrity"], updated_at=timestamp,
                  action={"kind": "review", "actor": reviewer, "timestamp": timestamp},
                  review_state=_review_state(record["reviews"]))
    return ledger.append(record)


def project_finding_show(bridge: Any, id: Any, revision: Any = None) -> dict:
    ledger = _Ledger(bridge)
    return ledger.rendered(ledger.get(id, revision))


def project_finding_check(bridge: Any, id: Any, revision: Any = None) -> dict:
    ledger = _Ledger(bridge)
    record = ledger.get(id, revision)
    return {"id": record["id"], "revision": record["revision"], "type": record["type"],
            "review_state": record["review_state"], **ledger.check(record)}


def _pagination(offset: Any, limit: Any) -> tuple[int, int]:
    offset = _integer(offset, "offset", 0)
    limit = _integer(limit, "limit")
    if limit > 200:
        raise RuntimeError("limit must be <= 200")
    return offset, limit


def _page(items: list, offset: int, limit: int) -> dict:
    return {"items": items[offset:offset + limit], "total": len(items), "offset": offset,
            "limit": limit, "has_more": offset + limit < len(items)}


def project_finding_history(bridge: Any, id: Any, offset: Any = 0, limit: Any = 50) -> dict:
    offset, limit = _pagination(offset, limit)
    ledger = _Ledger(bridge)
    finding_id = ledger.get(id)["id"]
    page = _page(ledger.records[finding_id], offset, limit)
    page["items"] = [ledger.rendered(row) for row in page["items"]]
    return page


def _function_key(value: Any) -> str | int:
    if isinstance(value, str):
        try:
            return int(value, 16 if value.lower().startswith("0x") else 10)
        except ValueError:
            return value.casefold()
    return value


def project_finding_list(bridge: Any, file: Any = None, function: Any = None, query: Any = None,
                         type: Any = None, status: Any = None, freshness: Any = None,
                         offset: Any = 0, limit: Any = 50) -> dict:
    offset, limit = _pagination(offset, limit)
    ledger = _Ledger(bridge)
    file_id = str(ledger.manager.resolve_file(file).id) if file is not None else None
    function_key = _function_key(_function(function)) if function is not None else None
    search = _text(query, "query", 4096).casefold() if query is not None else None
    wanted_types = {_type(item) for item in _array(type, "type filter")} if type is not None else None
    if status is not None and (not isinstance(status, str) or status not in _STATES):
        raise RuntimeError("status must be proposed, accepted, disputed, rejected, or superseded")
    if freshness is not None and (not isinstance(freshness, str) or freshness not in _FRESHNESS):
        raise RuntimeError("freshness must be current, stale, or unknown")
    matches = []
    for row in sorted(ledger.latest.values(), key=lambda item: (item["created_at"], item["id"])):
        if wanted_types is not None and row["type"] not in wanted_types:
            continue
        if status is not None and row["review_state"] != status:
            continue
        identities = row["scope"] + row["evidence"] + [e for r in _active_reviews(row) for e in r["evidence"]]
        if file_id is not None or function_key is not None:
            if not any((file_id is None or item.get("file_id") == file_id) and
                       (function_key is None or ("function" in item and _function_key(item["function"]) == function_key))
                       for item in identities):
                continue
        searchable = [row["claim"], row["type"], *row["tags"], *row["assumptions"], *row["limitations"], *row["next_steps"]]
        if search is not None and search not in "\n".join(searchable).casefold():
            continue
        if freshness is not None and ledger.check(row)["freshness"] != freshness:
            continue
        matches.append(row)
    page = _page(matches, offset, limit)
    page["items"] = [ledger.rendered(row) for row in page["items"]]
    return page


def project_finding_context(bridge: Any, **params: Any) -> dict:
    result = project_finding_list(bridge, **params)
    groups: dict[str, list] = {key: [] for key in ("accepted", "proposed", "disputed", "rejected", "superseded", "needs_review")}
    for row in result.pop("items"):
        state = row["review_state"]
        group = state if state in {"rejected", "superseded"} or row["freshness"] == "current" else "needs_review"
        groups[group].append(row)
    return {**result, "groups": groups}


def project_finding_types(bridge: Any) -> dict:
    ledger = _Ledger(bridge)
    counts = Counter(row["type"] for row in ledger.latest.values())
    types = []
    for name in [*_TYPES, *sorted(set(counts) - set(_TYPES))]:
        description, action = _TYPES.get(name, ("Project-defined finding type", "inspect claim and next_steps; independently verify before reuse"))
        types.append({"type": name, "description": description, "action": action, "count": counts[name]})
    return {"types": types}
