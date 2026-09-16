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
    # The three read-only py_exec programs bn-lens ran, promoted to
    # first-class read ops so they stop taking the exclusive writer lock.
    "cfg", "data_vars", "data_symbols",
    # Project associations update only the private registry payload, which
    # enumerates open targets; this remains a read of BN state.
    "associate_project_roots",
}
EXPECTED_WRITE = {
    # True short writers: gate + exclusive target lock for the whole op.
    "py_exec", "close_binary", "save_database",
    # NB: the mutation ops and "function_create" are intentionally NOT here --
    # #628 made them lock="none" so they self-manage locking (the write gate
    # serializes writers for the whole op, while the exclusive target lock covers
    # only the BN-mutating/snapshotting phases, leaving the post-apply reanalysis
    # readable), exactly like "refresh"/"load_binary" below.
}
# Ops that are registered lock="none" and therefore belong to NEITHER derived
# set. This is the test-side pin for them; the registry stays the single source
# of truth (`REGISTRY.read_locked_ops()`/`write_locked_ops()`).
EXPECTED_SELF_MANAGED = {
    "cancel_request", "load_binary", "load_binary_async", "load_status",
    "go_rename", "shutdown", "refresh",
    # #628: every binder that routes through bridge._mutation() -- all of them
    # reanalyze inside their own body -- plus the standalone function_create.
    "function_create", "rename_symbol", "set_comment", "delete_comment",
    "set_prototype", "local_rename", "local_retype", "data_retype",
    "struct_field_set", "struct_field_rename", "struct_field_delete",
    "types_declare", "batch_apply", "tag_add", "tag_remove",
    "tag_type_create", "tag_type_remove",
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


def test_self_managed_ops_are_unlocked(bridge, op_registry):
    """Every lock="none" op self-manages locking, so it belongs to NEITHER derived
    set. Being outside both is what keeps concurrent readers live while the op runs
    its own analysis/gate dance (#99 load, #321 refresh, #365 go_rename, #628
    mutation + function_create)."""
    assert EXPECTED_SELF_MANAGED.isdisjoint(EXPECTED_READ)
    assert EXPECTED_SELF_MANAGED.isdisjoint(EXPECTED_WRITE)
    # Every pinned name must be a name the registry actually registers. Without
    # this the loop below is vacuously true for a name that does not exist (an
    # unregistered op is trivially in neither derived set), so the pin could
    # drift to naming anything at all and still pass on its own.
    assert EXPECTED_SELF_MANAGED <= op_registry.REGISTRY.names()
    for op_name in sorted(EXPECTED_SELF_MANAGED):
        assert op_name not in bridge.READ_LOCKED_OPS, op_name
        assert op_name not in bridge.WRITE_LOCKED_OPS, op_name


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
    expected = EXPECTED_READ | EXPECTED_WRITE | EXPECTED_SELF_MANAGED
    assert REGISTRY.names() == expected


def test_decompile_is_the_only_escalating_op(op_registry):
    REGISTRY = op_registry.REGISTRY
    escalating = {n for n in REGISTRY.names() if REGISTRY.spec(n).lock_escalation is not None}
    assert escalating == {"decompile"}


# #688: the ops whose bare/empty/"active" target may NOT fall back to the
# focused GUI tab. Pinned here for the same reason the lock sets are: the flag
# is a policy declaration, so an op gaining or losing it is a deliberate
# change. `close_binary` is absent on purpose -- it refuses the ambiguous case
# in `_resolve_sole_target_for_close`, which also names its `all=true` escape.
EXPECTED_DESTRUCTIVE = {"save_database", "py_exec", "batch_apply", "go_rename"}


def test_the_destructive_ops_are_exactly_these(op_registry):
    REGISTRY = op_registry.REGISTRY
    destructive = {n for n in REGISTRY.names() if REGISTRY.spec(n).destructive}
    assert destructive == EXPECTED_DESTRUCTIVE


def test_destructive_bypass_without_the_flag_is_refused(op_registry):
    """A bypass predicate on a non-destructive op is dead code that reads like
    a guard, so it is rejected at import time rather than silently ignored."""
    reg = op_registry.OpRegistry()
    with pytest.raises(ValueError, match="destructive_bypass without destructive"):
        @reg.op("b", lock="write", destructive_bypass=lambda params: True)
        def _b(bridge, params, target): return 1
