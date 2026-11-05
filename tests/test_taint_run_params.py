"""`run_params` names every opt-in a taint run was given (W07-10, W05-05).

A result file is read long after the command that made it. An extra sink
class, a pinned indirect call and a model overlay each change the answer, so a
run that used one must not read like a default run.
"""
from __future__ import annotations

import json

import pytest

from bn_agent_bridge import read_taint_slice as rts


class _View:
    """Identity only: the engine below is recorded, never run."""


class _Ctx:
    def __init__(self) -> None:
        self._bv = _View()

    def _resolve_view(self, selector):
        return self._bv

    def _find_function(self, bv, name):
        if name == "parse_frame":
            return type("F", (), {"start": 0x401176})()
        return object()


@pytest.fixture
def unanalysed(monkeypatch):
    class _Engine(rts._taint.TaintEngine):
        def forward(self, func, locators, **kw):
            # The walk reaches the indirect call at 0x4011f0 and routes it by its pin.
            self.resolve_map.select(0x4011F0)
            return {"direction": "forward", "reached_sinks": [], "leaves": [],
                    "run_params": {"max_depth": kw.get("max_depth")}}

        def backward(self, func, locators, **kw):
            return {"direction": "backward", "slices": [], "leaves": [],
                    "run_params": {"max_depth": kw.get("max_depth")}}

    monkeypatch.setattr(rts._taint, "TaintEngine", _Engine)


def test_a_default_forward_run_echoes_no_opt_in(unanalysed):
    result = rts._taint_op(_Ctx(), "active", {"function": "handler", "sources": ["param:0"]})
    rp = result["run_params"]
    assert rp["sink_classes"] == []
    assert rp["resolve_map"] is None
    assert rp["models"] == []


def test_forward_run_echoes_sink_classes_resolve_map_and_models(unanalysed, tmp_path):
    overlay = tmp_path / "wrappers.json"
    overlay.write_text(json.dumps({"read_packet": {"sources": [{"to": "*arg:1"}]}}))
    result = rts._taint_op(_Ctx(), "active", {
        "function": "handler", "sources": ["param:0"],
        "enabled_sink_classes": ["recv_overflow", "file_write", "recv_overflow"],
        "resolve_map": {"0x4011f0": ["0x401176", "parse_frame"], "0x401200": ["0x401176"]},
        "resolve_map_path": "maps/dispatch.json",
        "user_models": json.loads(overlay.read_text()),
        "user_models_path": str(overlay), "user_models_via": "--models",
    })
    rp = result["run_params"]
    assert rp["sink_classes"] == ["file_write", "recv_overflow"]
    # A pin the walk consulted is matched; one no walked call names is not.
    assert rp["resolve_map"] == {"path": "maps/dispatch.json", "entries": 2,
                                 "matched": ["0x4011f0"], "unmatched": ["0x401200"]}
    assert [m["path"] for m in rp["models"]] == [str(overlay)]
    # The knobs the engine echoes itself are kept, not replaced.
    assert rp["max_depth"] == 8


def test_backward_run_echoes_resolve_map_but_has_no_sink_classes(unanalysed):
    result = rts._taint_op(_Ctx(), "active", {
        "direction": "backward", "function": "handler", "sinks": ["arg:memcpy:2"],
        "resolve_map": {"0x4011f0": ["0x401176"]}, "resolve_map_path": "m.json",
    })
    rp = result["run_params"]
    assert "sink_classes" not in rp
    assert rp["resolve_map"] == {"path": "m.json", "entries": 1,
                                 "matched": [], "unmatched": ["0x4011f0"]}
