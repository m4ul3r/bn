"""Declarative evidence against persisted project blobs and guarded live views."""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from _bridge_fakes import _load_bridge
from _project_fakes import FakeProject, SQLITE_MAGIC


class Expression:
    def __init__(self, text, operands=(0, 0, 0, 0, 0)):
        self.text = text
        self.operation = NS(name="HLIL_RET")
        self.address = 0x401000
        self.size = 8
        self.core_operands = operands

    def __str__(self):
        return self.text


class HLIL:
    def __init__(self):
        self.expressions = [Expression("return packet_size", (7, 0, 0, 0, 0))]
        self.instructions = 1

    def __len__(self):
        return self.instructions

    def get_expr_count(self):
        return len(self.expressions)

    def get_expr(self, index, as_ast=True):
        assert as_ast is False
        return self.expressions[index]


class View:
    def __init__(self, pf):
        self.project_file = pf
        self.file = NS(filename=pf.path_on_disk, has_database=True,
                       modified=False, analysis_changed=False)
        self.analysis_state = NS(name="IdleState")
        self.function = NS(start=0x401000, name="decode_packet", type="uint64_t decode_packet(void)",
                           address_ranges=[NS(start=0x401000, end=0x401004)], hlil_if_available=HLIL())
        self.code = b"\x48\x89\xf8\xc3"
        self.libraries = []
        self.locations = []
        self.read_hook = None

    def get_function_at(self, address):
        return self.function if address == self.function.start else None

    def read(self, address, length):
        assert address == self.function.start
        if self.read_hook:
            self.read_hook()
        return self.code[:length]

    def get_external_libraries(self):
        return list(self.libraries)

    def get_external_locations(self):
        return list(self.locations)

    @property
    def hlil(self):
        raise AssertionError("Generating HLIL is forbidden")


class Targets:
    def __init__(self):
        self.views = {}
        self.dirty = False
        self.binding = None
        self.database = None

    def view_for_project_file(self, file_id):
        return self.views.get(file_id)

    def project_binding(self, bv):
        return self.binding

    def is_dirty(self, bv):
        return self.dirty

    def recorded_database(self, bv):
        return self.database

    def resolve(self, selector):
        raise AssertionError("Evidence must not use target alias resolution")


@pytest.fixture
def env(monkeypatch, tmp_path):
    _load_bridge(monkeypatch)
    evidence = importlib.import_module("bn_test_bridge.project_finding_evidence")
    po = sys.modules["bn_test_bridge.project_ops"]
    sys.modules["binaryninja"].core_version = lambda: "5.4.9000-test"
    project = FakeProject(tmp_path)
    manager = po.ProjectManager()
    manager._state = po.ProjectState(project, str(tmp_path), project.id, project.name)
    bridge = NS(projects=manager, targets=Targets())
    blob = project.add_file("packet.bin", data=b"HEADER\x01\x02\x03\x04payload")
    database = project.add_file("packet.bin.bndb", data=SQLITE_MAGIC + b"saved-analysis-v1")
    view = View(database)
    bridge.targets.views[database.id] = view
    return NS(api=evidence, bridge=bridge, project=project, blob=blob, db=database, view=view)


def capture(env, kind, **kwargs):
    return env.api.capture_evidence(env.bridge, [{"check": kind, **kwargs}])


def check(env, captured):
    return env.api.check_evidence(env.bridge, captured)


def function(env, **kwargs):
    return capture(env, "function", file=env.db.id, function="0x401000", **kwargs)


def test_persisted_file_digest_normalizes_selector_and_detects_changes(env):
    captured = capture(env, "file", file="packet.bin", description="Imported protocol sample")
    row = captured[0]
    assert row["file_id"] == env.blob.id
    assert row["file_path"] == "/packet.bin"
    assert "file" not in row
    original = Path(env.blob.path_on_disk).read_bytes()
    assert row["observed"] == {"size": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    assert check(env, captured)["freshness"] == "current"
    Path(env.blob.path_on_disk).write_bytes(original + b"changed")
    assert check(env, captured)["freshness"] == "stale"


def test_streaming_file_observation_handles_multiple_chunks(env):
    content = b"\x00\xff" * (1024 * 1024 + 1)
    Path(env.blob.path_on_disk).write_bytes(content)
    row = capture(env, "file", file=env.blob.id)[0]
    assert row["observed"] == {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def test_bytes_are_persisted_file_offsets_not_virtual_addresses(env):
    captured = capture(env, "bytes", file=env.blob.id, offset=6, length=4,
                       expected={"hex": "01020304"})
    assert captured[0]["observed"]["hex"] == "01020304"
    assert check(env, captured)["freshness"] == "current"
    data = bytearray(Path(env.blob.path_on_disk).read_bytes())
    data[0] = 0x7f
    Path(env.blob.path_on_disk).write_bytes(data)
    assert check(env, captured)["freshness"] == "current"  # Only the requested range is claimed.
    data[7] = 0xff
    Path(env.blob.path_on_disk).write_bytes(data)
    assert check(env, captured)["freshness"] == "stale"


def test_short_read_has_no_baseline_and_never_silently_recaptures(env):
    captured = capture(env, "bytes", file=env.blob.id, offset=400, length=4)
    assert captured[0]["capture_state"] == "unknown"
    assert "short read" in captured[0]["reason"]
    Path(env.blob.path_on_disk).write_bytes(b"A" * 500)
    assert check(env, captured)["freshness"] == "unknown"


@pytest.mark.parametrize("evidence_request", [
    {"check": "python", "text": "raise SystemExit"},
    {"check": "file", "file": "packet.bin", "command": "touch marker"},
    {"check": "note", "text": " ", "file": "packet.bin"},
    {"check": "bytes", "file": "packet.bin", "offset": True, "length": 1},
    {"check": "bytes", "file": "packet.bin", "offset": -1, "length": 1},
    {"check": "bytes", "file": "packet.bin", "offset": 0, "length": 65537},
    {"check": "bytes", "file": "packet.bin", "offset": 0, "length": 0},
    {"check": "bytes", "file": "packet.bin", "offset": 0, "length": 2, "expected": {"hex": "00"}},
    {"check": "file", "file": "packet.bin", "expected": {"sha256": "not-a-hash"}},
    {"check": "file", "file": "packet.bin", "expected": {"size": 8}},
    {"check": "function", "file": "packet.bin", "function": "decode_packet"},
    {"check": "function", "file": "packet.bin", "function": False},
    {"check": "links", "file": "packet.bin", "expected": {}},
])
def test_declarative_schema_rejects_unsupported_or_unsafe_input(env, evidence_request):
    with pytest.raises(RuntimeError):
        env.api.capture_evidence(env.bridge, [evidence_request])


def test_external_machine_paths_are_not_project_file_selectors(env, tmp_path):
    external = tmp_path / "not-imported.txt"
    external.write_text("private")
    with pytest.raises(RuntimeError, match="No project file"):
        capture(env, "file", file=str(external))


def test_expected_mismatch_is_stale_even_when_later_unavailable(env):
    captured = capture(env, "file", file=env.blob.id, expected={"sha256": "0" * 64})
    assert captured[0]["capture_state"] == "stale"
    assert check(env, captured)["freshness"] == "stale"
    Path(env.blob.path_on_disk).unlink()
    assert check(env, captured)["freshness"] == "stale"


def test_notes_are_not_executed_and_do_not_establish_freshness(env, tmp_path):
    marker = tmp_path / "executed"
    notes = capture(env, "note", text=f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert check(env, notes)["freshness"] == "unknown"
    assert not marker.exists()
    complete = capture(env, "file", file=env.blob.id)
    assert check(env, notes + complete)["freshness"] == "current"
    incomplete = capture(env, "function", file=env.blob.id, function=0x401000)
    assert check(env, notes + complete + incomplete)["freshness"] == "unknown"
    Path(env.blob.path_on_disk).write_bytes(b"changed")
    assert check(env, notes + complete + incomplete)["freshness"] == "stale"


def test_deleted_project_id_never_rebinds_to_same_name_or_path(env):
    captured = capture(env, "file", file=env.blob.id)
    env.project.files.remove(env.blob)
    env.project.add_file("packet.bin", data=Path(env.blob.path_on_disk).read_bytes())
    result = check(env, captured)
    assert result["freshness"] == "stale"
    assert "ID" in result["checks"][0]["reason"]


def test_clean_durable_function_captures_semantic_facts_and_versions(env):
    captured = function(env, expected={"name": "decode_packet", "prototype": "uint64_t decode_packet(void)"})
    row = captured[0]
    assert row["capture_state"] == "current", row.get("reason")
    assert row["function"] == 0x401000
    observed = row["observed"]
    assert observed["name"] == "decode_packet"
    assert observed["ranges"][0]["sha256"] == hashlib.sha256(env.view.code).hexdigest()
    assert observed["hlil_expressions"] == 1
    assert observed["source"]["sha256"] == hashlib.sha256(Path(env.db.path_on_disk).read_bytes()).hexdigest()
    assert observed["tool_versions"]["core"] == "5.4.9000-test"
    assert observed["tool_versions"]["engine_build_id"]
    assert check(env, captured)["freshness"] == "current"


@pytest.mark.parametrize("field,value", [("modified", True), ("analysis_changed", True),
                                        ("has_database", False), ("modified", None)])
def test_semantic_evidence_requires_native_clean_saved_state(env, field, value):
    setattr(env.view.file, field, value)
    captured = function(env)
    assert captured[0]["capture_state"] == "unknown"
    setattr(env.view.file, field, field == "has_database")
    assert check(env, captured)["freshness"] == "unknown"


def test_semantic_evidence_respects_bridge_dirty_and_native_analysis_progress(env):
    env.bridge.targets.dirty = True
    assert function(env)[0]["capture_state"] == "unknown"
    env.bridge.targets.dirty = False
    env.view.analysis_state.name = "AnalyzeState"
    assert function(env)[0]["capture_state"] == "unknown"


def test_missing_native_fact_is_unknown_not_clean_by_default(env):
    del env.view.file.analysis_changed
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "analysis_changed" in row["reason"]


def test_missing_loaded_view_never_loads_or_uses_raw_sibling_alias(env):
    captured = capture(env, "function", file=env.blob.id, function=0x401000)
    assert captured[0]["capture_state"] == "unknown"
    assert "already-loaded" in captured[0]["reason"]
    env.bridge.targets.views.clear()
    assert function(env)[0]["capture_state"] == "unknown"


def test_wrong_loaded_project_binding_is_rejected(env):
    env.bridge.targets.binding = (env.blob.id, "/packet.bin")
    assert function(env)[0]["capture_state"] == "unknown"


def test_raw_blob_cannot_masquerade_as_durable_database(env):
    env.bridge.targets.views[env.blob.id] = View(env.blob)
    row = capture(env, "function", file=env.blob.id, function=0x401000)[0]
    assert row["capture_state"] == "unknown"
    assert "not a raw artifact" in row["reason"]


def test_function_semantic_name_prototype_and_bytes_changes_invalidate(env):
    captured = function(env)
    env.view.function.hlil_if_available.expressions[0].text = "return packet_size + 1"
    assert check(env, captured)["freshness"] == "stale"
    env.view.function.hlil_if_available.expressions[0].text = "return packet_size"
    env.view.function.type = "int64_t decode_packet(void)"
    assert check(env, captured)["freshness"] == "stale"
    env.view.function.type = "uint64_t decode_packet(void)"
    env.view.code = b"\x48\x31\xc0\xc3"
    assert check(env, captured)["freshness"] == "stale"


def test_any_saved_database_change_invalidates_even_without_live_context(env):
    captured = function(env)
    Path(env.db.path_on_disk).write_bytes(SQLITE_MAGIC + b"different-saved-settings-or-models")
    env.bridge.targets.views.clear()
    result = check(env, captured)
    assert result["freshness"] == "stale"
    assert "source" in result["checks"][0]["reason"]


def test_unsaved_database_change_is_unknown_not_reused_current(env):
    captured = function(env)
    env.view.file.analysis_changed = True
    assert check(env, captured)["freshness"] == "unknown"


def test_tool_version_change_invalidates_semantic_baseline(env):
    captured = function(env)
    sys.modules["binaryninja"].core_version = lambda: "5.5.9100-test"
    assert check(env, captured)["freshness"] == "stale"


def test_unavailable_hlil_and_missing_function_are_unknown(env):
    env.view.function.hlil_if_available = None
    assert function(env)[0]["capture_state"] == "unknown"
    captured = capture(env, "function", file=env.db.id, function=0x402000)
    assert captured[0]["capture_state"] == "unknown"


def test_incomplete_or_oversized_semantic_observation_never_claims_current(env):
    env.view.function.hlil_if_available.expressions = [None]
    assert function(env)[0]["capture_state"] == "unknown"
    env.view.function.hlil_if_available.expressions = [Expression("x" * (1024 * 1024 + 1))]
    assert function(env)[0]["capture_state"] == "unknown"
    env.view.function.address_ranges = [NS(start=0x401000, end=0x601001)]
    assert function(env)[0]["capture_state"] == "unknown"


def test_dirty_state_changing_during_observation_is_unknown(env):
    env.view.read_hook = lambda: setattr(env.view.file, "modified", True)
    assert function(env)[0]["capture_state"] == "unknown"


def test_native_exception_recheck_reports_unknown_with_disclosure(env):
    captured = function(env)

    def failing_read(address):
        raise RuntimeError("native function lookup failed")

    env.view.get_function_at = failing_read
    result = check(env, captured)
    assert result["freshness"] == "unknown"
    assert "native function lookup failed" in result["checks"][0]["reason"]


def binding(env, name="libpacket.so", symbol="packet_read", address=0x403000):
    provider = env.project.add_file(name, data=b"\x7fELFlibrary-v1")
    library = NS(name=name, backing_file=provider)
    location = NS(source_symbol=NS(raw_name=symbol, address=address, type=NS(name="ExternalSymbol")),
                  library=library, target_address=None, target_symbol=symbol)
    env.view.libraries.append(library)
    env.view.locations.append(location)
    return provider, library, location


def test_links_capture_provider_ids_and_stable_sorted_binding_facts(env):
    provider, _, _ = binding(env)
    binding(env, "libcrypto.so", "verify_packet", 0x403010)
    captured = capture(env, "links", file=env.db.id)
    assert captured[0]["capture_state"] == "current", captured[0].get("reason")
    observed = captured[0]["observed"]
    assert provider.id in observed["providers"]
    assert any(row["provider_file_id"] == provider.id for row in observed["libraries"])
    env.view.libraries.reverse()
    env.view.locations.reverse()
    assert check(env, captured)["freshness"] == "current"
    Path(provider.path_on_disk).write_bytes(b"\x7fELFlibrary-v2")
    assert check(env, captured)["freshness"] == "stale"


def test_link_binding_target_or_provider_change_invalidates(env):
    _, library, location = binding(env)
    captured = capture(env, "links", file=env.db.id)
    location.target_address = 0x1230
    assert check(env, captured)["freshness"] == "stale"
    location.target_address = None
    library.backing_file = env.project.add_file("replacement.so", data=b"\x7fELFlibrary-v1")
    assert check(env, captured)["freshness"] == "stale"


def test_unresolved_link_is_explicit_binding_observation_not_loader_proof(env):
    env.view.libraries = [NS(name="libmissing.so", backing_file=None)]
    captured = capture(env, "links", file=env.db.id)
    observed = captured[0]["observed"]
    assert captured[0]["capture_state"] == "current"
    assert observed["libraries"] == [{"name": "libmissing.so", "provider_file_id": None, "resolution": "unresolved"}]
    assert "not dynamic-loader" in observed["limitation"]


def test_dangling_provider_reference_is_unknown_not_guessed_by_name(env):
    provider, _, _ = binding(env)
    env.project.files.remove(provider)
    env.project.add_file("libpacket.so", data=b"\x7fELFlibrary-v1")
    captured = capture(env, "links", file=env.db.id)
    assert captured[0]["capture_state"] == "unknown"
    assert provider.id in captured[0]["reason"]


def test_provider_change_stays_stale_when_consumer_view_is_unavailable(env):
    provider, _, _ = binding(env)
    captured = capture(env, "links", file=env.db.id)
    env.bridge.targets.views.clear()
    Path(provider.path_on_disk).write_bytes(b"\x7fELFchanged-library")
    result = check(env, captured)
    assert result["freshness"] == "stale"
    assert "provider" in result["checks"][0]["reason"]


def test_deleted_provider_stays_stale_without_live_consumer(env):
    provider, _, _ = binding(env)
    captured = capture(env, "links", file=env.db.id)
    env.project.files.remove(provider)
    env.bridge.targets.views.clear()
    assert check(env, captured)["freshness"] == "stale"


def test_function_expected_mismatch_stays_stale_without_live_analysis(env):
    captured = function(env, expected={"name": "validate_packet"})
    assert captured[0]["capture_state"] == "stale"
    env.bridge.targets.views.clear()
    assert check(env, captured)["freshness"] == "stale"


def test_database_path_must_match_exact_project_file(env):
    env.view.file.filename = env.blob.path_on_disk
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "exact persisted BNDB" in row["reason"]


def test_bridge_first_save_binding_uses_recorded_database_not_raw_native_identity(env):
    env.view.project_file = env.blob
    env.view.file.filename = env.blob.path_on_disk
    env.bridge.targets.binding = (env.db.id, "/packet.bin.bndb")
    env.bridge.targets.database = env.db.path_on_disk
    assert function(env)[0]["capture_state"] == "current"


def test_unknown_tool_identity_does_not_claim_current(env):
    sys.modules["binaryninja"].core_version = lambda: ""
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "version identity" in row["reason"]


def test_expression_count_bound_refuses_before_iterating(env):
    env.view.function.hlil_if_available.get_expr_count = lambda: 16385
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "expression bound" in row["reason"]


def test_original_unknown_semantics_are_not_repaired_by_loading_later(env):
    env.bridge.targets.views.clear()
    captured = function(env)
    env.bridge.targets.views[env.db.id] = env.view
    assert check(env, captured)["freshness"] == "unknown"
    assert function(env)[0]["capture_state"] == "current"


def test_changed_hlil_instruction_length_during_capture_is_unknown(env):
    hlil = env.view.function.hlil_if_available
    original = hlil.get_expr

    def changing_instruction_length(index, as_ast=True):
        expression = original(index, as_ast=as_ast)
        hlil.instructions += 1
        return expression

    hlil.get_expr = changing_instruction_length
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "HLIL changed" in row["reason"]


def test_malformed_file_observation_cannot_reuse_retained_fingerprint(env):
    captured = capture(env, "file", file=env.blob.id)
    captured[0]["observed"] = {}
    with pytest.raises(RuntimeError, match="Captured blob observation"):
        env.api.validate_captured_evidence(captured)
    with pytest.raises(RuntimeError, match="Captured blob observation"):
        check(env, captured)


def test_valid_shape_with_inconsistent_fingerprint_is_refused(env):
    captured = capture(env, "file", file=env.blob.id)
    captured[0]["observed"]["size"] += 1
    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        check(env, captured)


@pytest.mark.parametrize("kind,missing", [
    ("file", "sha256"), ("file", "size"),
    ("bytes", "hex"), ("bytes", "sha256"),
    ("function", "source"), ("function", "tool_versions"),
    ("function", "function"), ("function", "name"), ("function", "prototype"),
    ("function", "ranges"), ("function", "hlil_sha256"),
    ("function", "hlil_expressions"), ("function", "hlil_instructions"),
    ("links", "libraries"), ("links", "locations"), ("links", "providers"),
    ("links", "source"), ("links", "tool_versions"), ("links", "limitation"),
])
def test_complete_observation_requires_every_check_specific_fact(env, kind, missing):
    options = {"file": env.db.id if kind in {"function", "links"} else env.blob.id}
    if kind == "bytes":
        options.update(offset=0, length=4)
    elif kind == "function":
        options["function"] = 0x401000
    captured = capture(env, kind, **options)
    assert captured[0]["capture_state"] == "current"
    del captured[0]["observed"][missing]
    captured[0]["fingerprint"] = env.api._digest(captured[0]["observed"])
    with pytest.raises(RuntimeError):
        check(env, captured)


@pytest.mark.parametrize("state", ["stale", "unknown"])
def test_capture_state_cannot_disagree_with_complete_baseline(env, state):
    captured = capture(env, "file", file=env.blob.id)
    captured[0]["capture_state"] = state
    with pytest.raises(RuntimeError):
        check(env, captured)


def test_expected_mismatch_cannot_be_relabelled_current(env):
    captured = capture(env, "file", file=env.blob.id, expected={"sha256": "0" * 64})
    captured[0]["capture_state"] = "current"
    with pytest.raises(RuntimeError, match="capture_state disagrees"):
        check(env, captured)


@pytest.mark.parametrize("kind", ["function", "links"])
def test_unknown_partial_source_is_a_valid_uncheckable_baseline(env, kind):
    env.bridge.targets.views.clear()
    options = {"file": env.db.id}
    if kind == "function":
        options["function"] = 0x401000
    captured = capture(env, kind, **options)
    assert set(captured[0]["observed"]) == {"source"}
    assert captured[0]["fingerprint"] is None
    env.api.validate_captured_evidence(captured)
    assert check(env, captured)["freshness"] == "unknown"
    captured[0]["observed"]["source"] = {}
    with pytest.raises(RuntimeError, match="Captured blob observation"):
        check(env, captured)


def test_unknown_null_observations_and_manual_notes_are_valid(env):
    captured = capture(env, "bytes", file=env.blob.id, offset=400, length=4)
    captured += capture(env, "note", text="Independent review still required")
    env.api.validate_captured_evidence(captured)
    assert check(env, captured)["freshness"] == "unknown"
    captured[-1]["capture_state"] = "current"
    with pytest.raises(RuntimeError, match="Manual notes must remain unknown"):
        check(env, captured)


def test_byte_observation_digest_must_match_its_hex(env):
    captured = capture(env, "bytes", file=env.blob.id, offset=0, length=4)
    captured[0]["observed"]["sha256"] = "0" * 64
    captured[0]["fingerprint"] = env.api._digest(captured[0]["observed"])
    with pytest.raises(RuntimeError, match="hex and sha256 disagree"):
        check(env, captured)


@pytest.mark.parametrize("mutation", ["provider", "location", "versions", "ranges"])
def test_nested_semantic_observations_must_remain_complete(env, mutation):
    provider, _, _ = binding(env)
    captured = (function(env) if mutation in {"versions", "ranges"}
                else capture(env, "links", file=env.db.id))
    observed = captured[0]["observed"]
    if mutation == "provider":
        observed["providers"][provider.id] = {}
    elif mutation == "location":
        del observed["locations"][0]["target_symbol"]
    elif mutation == "versions":
        del observed["tool_versions"]["engine_build_id"]
    else:
        observed["ranges"] = []
    captured[0]["fingerprint"] = env.api._digest(observed)
    with pytest.raises(RuntimeError):
        check(env, captured)


@pytest.mark.parametrize("instructions", [0, 16385])
def test_hlil_instruction_length_bound_uses_native_len(env, instructions):
    env.view.function.hlil_if_available.instructions = instructions
    row = function(env)[0]
    assert row["capture_state"] == "unknown"
    assert "expression bound" in row["reason"]


@pytest.mark.parametrize("field,value", [
    ("fingerprint", None), ("fingerprint", "bad-digest"),
    ("observed", None), ("capture_state", "accepted"),
])
def test_current_baseline_requires_coherent_capture_metadata(env, field, value):
    captured = capture(env, "file", file=env.blob.id)
    captured[0][field] = value
    with pytest.raises(RuntimeError):
        check(env, captured)


def test_unknown_baseline_cannot_retain_complete_file_observation(env):
    captured = capture(env, "file", file=env.blob.id)
    captured[0].update(capture_state="unknown", fingerprint=None)
    with pytest.raises(RuntimeError, match="cannot have a complete observation"):
        check(env, captured)


def test_link_provider_fingerprint_set_must_match_library_bindings(env):
    binding(env)
    captured = capture(env, "links", file=env.db.id)
    captured[0]["observed"]["providers"] = {}
    captured[0]["fingerprint"] = env.api._digest(captured[0]["observed"])
    with pytest.raises(RuntimeError, match="do not match bound libraries"):
        check(env, captured)
