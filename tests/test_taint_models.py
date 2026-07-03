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
