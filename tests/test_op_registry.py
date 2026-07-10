from __future__ import annotations

import sys

import pytest

from _bridge_fakes import _load_bridge

# Frozen from the pre-refactor tip (Task 0.1). If a future PR legitimately adds
# an op, update these two sets in the SAME commit — that is the single point of
# truth this test enforces.
EXPECTED_READ = {
    "doctor", "list_targets", "target_info", "function_info", "get_prototype",
    "list_functions", "list_locals", "search_functions", "callsites", "decompile",
    "il", "structured_il", "defuse", "resolved_calls", "possible_values", "taint", "taint_models",
    "disasm", "function_evidence", "xrefs", "xrefs_any", "field_xrefs", "pointer_table",
    "call_descriptors", "hidden_surface", "resolve_virtual_call",
    "message_lens", "init_arrays", "backward_slice", "types", "type_info",
    "strings", "imports", "list_exports", "bundle_function", "get_comment", "list_comments",
    "sections", "read", "class_list", "class_show", "go_functions", "orient_digest",
    "list_tag_types", "get_tags", "list_tags",
}
EXPECTED_WRITE = {
    "py_exec", "function_create", "rename_symbol", "set_comment", "delete_comment",
    "set_prototype", "local_rename", "local_retype", "struct_field_set",
    "struct_field_rename", "struct_field_delete", "types_declare", "batch_apply",
    "close_binary", "save_database",
    # NB: "refresh" is intentionally NOT here -- #321 made it lock="none" so it
    # self-manages locking (analysis runs under the write GATE only, leaving reads
    # responsive), mirroring load_binary. See the self-managed set below.
}


@pytest.fixture
def bridge(monkeypatch):
    # Load the bridge against the shared fake `binaryninja` seam so collection
    # never pulls real BN (bn_agent_bridge/__init__ eagerly imports bridge,
    # which imports binaryninja). The op registry is a pure-Python submodule,
    # but it is only reachable through that package import.
    return _load_bridge(monkeypatch)


@pytest.fixture
def op_registry(bridge):
    return sys.modules["bn_test_bridge.op_registry"]


def test_read_locked_ops_membership_unchanged(bridge):
    assert set(bridge.READ_LOCKED_OPS) == EXPECTED_READ


def test_write_locked_ops_membership_unchanged(bridge):
    assert set(bridge.WRITE_LOCKED_OPS) == EXPECTED_WRITE


def test_no_op_is_both_read_and_write(bridge):
    assert set(bridge.READ_LOCKED_OPS).isdisjoint(bridge.WRITE_LOCKED_OPS)


def test_self_managed_ops_are_unlocked(bridge):
    assert "cancel_request" not in bridge.READ_LOCKED_OPS
    assert "cancel_request" not in bridge.WRITE_LOCKED_OPS
    assert "load_binary" not in bridge.READ_LOCKED_OPS
    assert "load_binary" not in bridge.WRITE_LOCKED_OPS
    assert "go_rename" not in bridge.READ_LOCKED_OPS
    assert "go_rename" not in bridge.WRITE_LOCKED_OPS
    assert "shutdown" not in bridge.READ_LOCKED_OPS
    assert "shutdown" not in bridge.WRITE_LOCKED_OPS
    # #321: refresh self-manages locking (write gate around analysis, not the
    # exclusive target lock) so reads stay responsive during a long analysis.
    assert "refresh" not in bridge.READ_LOCKED_OPS
    assert "refresh" not in bridge.WRITE_LOCKED_OPS


def test_op_decorator_registers_and_derives_locks(op_registry):
    reg = op_registry.OpRegistry()

    @reg.op("alpha", lock="read")
    def _bind_alpha(bridge, params, target):
        return ("alpha", target)

    @reg.op("beta", lock="write")
    def _bind_beta(bridge, params, target):
        return "beta"

    assert reg.read_locked_ops() == {"alpha"}
    assert reg.write_locked_ops() == {"beta"}
    assert reg.spec("alpha").binder(None, {}, "t") == ("alpha", "t")


def test_duplicate_op_registration_raises(op_registry):
    reg = op_registry.OpRegistry()

    @reg.op("dup", lock="read")
    def _a(bridge, params, target): return 1

    with pytest.raises(ValueError, match="duplicate op registration"):
        @reg.op("dup", lock="read")
        def _b(bridge, params, target): return 2


def test_invalid_lock_class_raises(op_registry):
    reg = op_registry.OpRegistry()
    with pytest.raises(ValueError, match="invalid lock class"):
        @reg.op("x", lock="sometimes")
        def _x(bridge, params, target): return 1


def test_escalation_is_stored(op_registry):
    reg = op_registry.OpRegistry()

    @reg.op("e", lock="read", escalation=lambda p: bool(p.get("force")))
    def _e(bridge, params, target): return 1

    assert reg.spec("e").lock_escalation({"force": True}) is True


def test_registry_covers_every_dispatch_op(op_registry):
    REGISTRY = op_registry.REGISTRY
    expected = EXPECTED_READ | EXPECTED_WRITE | {
        "cancel_request", "load_binary", "go_rename", "shutdown", "refresh",
    }
    assert REGISTRY.names() == expected


def test_decompile_is_the_only_escalating_op(op_registry):
    REGISTRY = op_registry.REGISTRY
    escalating = {n for n in REGISTRY.names() if REGISTRY.spec(n).lock_escalation is not None}
    assert escalating == {"decompile"}
