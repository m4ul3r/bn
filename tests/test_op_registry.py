from __future__ import annotations
import importlib

import bn_agent_bridge.bridge as bridge

# Frozen from the pre-refactor tip (Task 0.1). If a future PR legitimately adds
# an op, update these two sets in the SAME commit — that is the single point of
# truth this test enforces.
EXPECTED_READ = {
    "doctor", "list_targets", "target_info", "function_info", "get_prototype",
    "list_functions", "list_locals", "search_functions", "callsites", "decompile",
    "il", "structured_il", "defuse", "resolved_calls", "possible_values", "taint",
    "disasm", "function_evidence", "xrefs", "field_xrefs", "pointer_table",
    "message_lens", "init_arrays", "backward_slice", "types", "type_info",
    "strings", "imports", "bundle_function", "get_comment", "list_comments",
    "sections", "read",
}
EXPECTED_WRITE = {
    "py_exec", "function_create", "rename_symbol", "set_comment", "delete_comment",
    "set_prototype", "local_rename", "local_retype", "struct_field_set",
    "struct_field_rename", "struct_field_delete", "types_declare", "batch_apply",
    "refresh", "close_binary", "save_database",
}


def test_read_locked_ops_membership_unchanged():
    assert set(bridge.READ_LOCKED_OPS) == EXPECTED_READ


def test_write_locked_ops_membership_unchanged():
    assert set(bridge.WRITE_LOCKED_OPS) == EXPECTED_WRITE


def test_no_op_is_both_read_and_write():
    assert set(bridge.READ_LOCKED_OPS).isdisjoint(bridge.WRITE_LOCKED_OPS)


def test_load_binary_and_shutdown_are_unlocked():
    assert "load_binary" not in bridge.READ_LOCKED_OPS
    assert "load_binary" not in bridge.WRITE_LOCKED_OPS
    assert "shutdown" not in bridge.READ_LOCKED_OPS
    assert "shutdown" not in bridge.WRITE_LOCKED_OPS
