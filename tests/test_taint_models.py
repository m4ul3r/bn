from __future__ import annotations

import pytest

from bn_agent_bridge.read_taint_models import build_catalog

_MODELS = {
    "_comment": "ignored",
    "recv": {"sources": [{"to": "*arg:1"}, {"to": "ret"}]},
    "memcpy": {"propagates": [{"from": "*arg:1", "to": "*arg:0"}],
               "sink": {"tainted_args": [2], "class": "overflow_len", "detail": "len"}},
    "system": {"sink": {"tainted_args": [0], "class": "command_injection", "detail": "cmd"}},
    "strlen": {"propagates": [{"from": "*arg:0", "to": "ret"}]},
}


def test_build_catalog_groups_by_role_and_class():
    cat = build_catalog(_MODELS)
    assert {s["symbol"] for s in cat["sources"]} == {"recv"}
    assert set(cat["sinks_by_class"]) == {"overflow_len", "command_injection"}
    assert {p["symbol"] for p in cat["propagators"]} == {"memcpy", "strlen"}
    assert "_comment" not in {s["symbol"] for s in cat["sources"]}


def test_build_catalog_role_filter():
    cat = build_catalog(_MODELS, role="sink")
    assert cat["sources"] == [] and cat["propagators"] == []
    assert set(cat["sinks_by_class"]) == {"overflow_len", "command_injection"}


def test_build_catalog_class_filter_implies_sink():
    cat = build_catalog(_MODELS, sink_class="overflow_len")
    assert set(cat["sinks_by_class"]) == {"overflow_len"}
    assert cat["sources"] == [] and cat["propagators"] == []


# --- op handler (catalog + binary-present) -----------------------------------

from bn_agent_bridge import read_taint_slice as rts


class _CtxNoView:
    def _resolve_view(self, sel):  # pragma: no cover - not hit for catalog-only
        raise AssertionError("should not resolve a view for catalog-only dump")


def test_taint_models_op_catalog_only():
    res = rts._taint_models_op(_CtxNoView(), None, {})
    assert "sinks_by_class" in res and "overlays" in res and "items" in res
    assert res["sinks_by_class"]                          # builtin DB has real sinks


def test_taint_models_op_present_without_target_errors():
    with pytest.raises(rts.OperationFailure):
        rts._taint_models_op(_CtxNoView(), None, {"present": True})


class _FakeFn:
    def __init__(self, name):
        self.name = name


class _FakeBV:
    def __init__(self, fn_names):
        self.functions = [_FakeFn(n) for n in fn_names]

    def get_symbols(self):
        return []

    def get_symbols_by_name(self, n):
        return []

    def get_code_refs(self, a):
        return []


class _CtxWithBV:
    def __init__(self, bv):
        self._bv = bv

    def _resolve_view(self, sel):
        return self._bv


def test_taint_models_op_present_intersects_binary():
    bv = _FakeBV(["memcpy", "helper_fn", "system"])
    res = rts._taint_models_op(_CtxWithBV(bv), "active", {"present": True})
    syms = {e["symbol"] for lst in res["sinks_by_class"].values() for e in lst}
    assert "memcpy" in syms and "system" in syms          # modeled + present
    assert "strcpy" not in syms                            # modeled but absent -> filtered
    for lst in res["sinks_by_class"].values():
        for e in lst:
            assert e["present"] is True
