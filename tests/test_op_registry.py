from __future__ import annotations

import sys

import pytest

from _bridge_fakes import _load_bridge

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


def test_no_op_is_both_read_and_write(bridge):
    assert set(bridge.READ_LOCKED_OPS).isdisjoint(bridge.WRITE_LOCKED_OPS)


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


def test_escalation_exists_only_for_force_analysis(op_registry):
    """Lock escalation is a read op's licence to take the WRITE lock, so the
    set is pinned: an op gaining it is a deliberate change, not a side effect.

    Both members escalate for the same single reason -- `--force-analysis`
    reanalyses a function BN skipped, which mutates the view. `decompile_batch`
    (#676 item 5) is the multi-identifier form of `decompile` and forwards the
    same flag, so it inherits the same licence; it is a second instance of one
    mechanism, not a second mechanism.
    """
    REGISTRY = op_registry.REGISTRY
    escalating = {n for n in REGISTRY.names() if REGISTRY.spec(n).lock_escalation is not None}
    assert escalating == {"decompile", "decompile_batch"}


# #688: the ops whose bare/empty/"active" target may NOT fall back to the
# focused GUI tab. Pinned here for the same reason the lock sets are: the flag
# is a policy declaration, so an op gaining or losing it is a deliberate
# change. `close_binary` is absent on purpose -- it refuses the ambiguous case
# in `_resolve_sole_target_for_close`, which also names its `all=true` escape.
EXPECTED_DESTRUCTIVE = {
    "save_database", "py_exec", "batch_apply", "go_rename", "function_delete", "project_link",
}


def test_the_destructive_ops_are_exactly_these(op_registry):
    REGISTRY = op_registry.REGISTRY
    destructive = {n for n in REGISTRY.names() if REGISTRY.spec(n).destructive}
    assert destructive == EXPECTED_DESTRUCTIVE


def test_a_selector_reader_without_the_destructive_flag_is_refused(op_registry):
    """A selector reader exists to feed the destructive gate, so declaring one
    on a non-destructive op is dead code that reads like a guard: rejected at
    import time rather than silently ignored."""
    reg = op_registry.OpRegistry()
    with pytest.raises(ValueError, match="selector reader without destructive"):
        @reg.op("b", lock="write", selector=lambda params, target: target)
        def _b(bridge, params, target): return 1


def test_batch_apply_is_the_only_op_declaring_a_selector_reader(op_registry):
    """The reader exists because batch_apply resolves its manifest's `target`
    in preference to the request's, so the gate and the binder must read one
    decider (#736 review). Any other op adding one is a policy change."""
    REGISTRY = op_registry.REGISTRY
    with_reader = {n for n in REGISTRY.names() if REGISTRY.spec(n).selector is not None}
    assert with_reader == {"batch_apply"}
