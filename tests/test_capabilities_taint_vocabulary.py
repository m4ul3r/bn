"""`bn capabilities` advertises the taint output vocabulary (V4-05).

An agent reading a taint result meets leaf kinds, truncation causes, origin
kinds, `sink_status` statuses and `run_params` keys; the capability index named
none of them, so the only way to learn the vocabulary was to run taint and read
the docs. The `taint` entry carries each list, read from the one source the
engine and the drift tests share -- never a copy -- so a new value in the
engine is advertised without a second edit.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import bn.cli
import bn.taint_vocabulary as vocab
from bn.taint_truncation import TRUNCATION_CAUSES
from bn.taint_vocabulary import (
    BLOCKING_LEAF_KINDS,
    VocabularyUnresolved,
    kind_rows,
    origin_constructors,
    pointer_object_constructors,
    run_params_keys,
    sink_fields,
)

BRIDGE = Path(__file__).resolve().parents[1] / "src" / "bn_agent_bridge"
FIELDS = ["leaf_kinds", "truncation_causes", "origin_kinds", "pointer_object_kinds",
          "sink_status", "run_params", "sink_fields"]


def _capabilities(capsys, *extra: str) -> dict:
    assert bn.cli.main(["capabilities", "--format", "json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_capabilities_json_carries_the_taint_vocabulary_from_its_sources(capsys):
    data = _capabilities(capsys)
    # The existing catalog shape is untouched.
    assert data["kind"] == "capabilities" and data["count"] == len(data["items"])
    taint = data["taint"]
    assert sorted(taint) == sorted(FIELDS)
    assert taint["leaf_kinds"] == sorted(BLOCKING_LEAF_KINDS)
    assert taint["truncation_causes"] == sorted(TRUNCATION_CAUSES)
    engine = BRIDGE / "taint_engine.py"
    assert taint["origin_kinds"] == sorted(kind_rows(engine, origin_constructors))
    # G5-R5: the pointer-demand object kinds, the cross-boundary value kind among
    # them, read off the one routine that builds them.
    assert taint["pointer_object_kinds"] == sorted(kind_rows(engine, pointer_object_constructors))
    assert {"literal_value", "returned", "global", "parameter"} <= set(
        taint["pointer_object_kinds"])
    assert taint["run_params"] == sorted(
        run_params_keys(engine) | run_params_keys(BRIDGE / "read_taint_slice.py"))
    # An independent read of the sink_status producers: every status literal an
    # appended row spells is advertised.
    text = engine.read_text(encoding="utf-8")
    appended = set(re.findall(r'sink_status\.append\(\{[^}]*?"status": "(\w+)"', text, re.S))
    assert appended and set(taint["sink_status"]) == appended
    assert taint["sink_fields"] == sorted(sink_fields(engine))
    for field in FIELDS:
        assert taint[field], field


def test_every_sink_field_a_forward_run_emits_is_advertised(capsys):
    # G3-04: `return_unchecked` (V7-01) reached consumers with no entry in the
    # vocabulary. Independent of the crawler: the sink objects real engine
    # runs build -- a fill-sized copy, an unchecked receive return, a narrowed
    # allocation size -- carry only advertised keys.
    from test_taint_width_and_relabel import (
        UNCHECKED,
        _alloc_after,
        _calloc_sink,
        _memcpy_sink,
        _recv_then_copy,
        _recv_then_dup,
    )
    from test_taint_engine import te
    advertised = set(_capabilities(capsys)["taint"]["sink_fields"])
    models = te.load_models()
    sinks = [
        _memcpy_sink(models, *_recv_then_dup(count="const", length="sx", src="same")),
        _memcpy_sink(models, *_recv_then_copy(width=8, at_copy=UNCHECKED)),
        _calloc_sink(models, *_alloc_after("MLIL_ADD", 4)),
    ]
    emitted = set().union(*sinks)
    assert {"src_fill", "return_unchecked", "width_narrowing"} <= emitted, emitted
    assert emitted <= advertised, emitted - advertised


_ENGINE = textwrap.dedent('''
    NEW_ORIGIN = "brand_new_origin"

    class TaintEngine:
        def forward(self):
            return {"run_params": {"max_depth": 1, "brand_new_knob": 2}}

        def backward(self, kind="parameter"):
            sink_status = []
            sink_status.append({"status": "brand_new_status", "seeded": False})
            origin = {"kind": NEW_ORIGIN if kind else kind}
            return {"sink_status": sink_status, "origin": origin}

        def _demand_objects(self, expr):
            return [{"kind": "brand_new_object", "object": str(expr)}]

        def _make_finding(self, callee, sink):
            return {"sink": {"callee": callee, **self._extra(sink),
                             **({"brand_new_flag": True} if sink else {})}}

        def _extra(self, sink):
            return {"brand_new_field": sink} if sink else {}
''')
_SLICE = textwrap.dedent('''
    def op(result):
        run_params = result.setdefault("run_params", {})
        run_params["brand_new_echo"] = 1
''')


def _bridge(tmp_path: Path, engine: str = _ENGINE, slice_: str = _SLICE) -> Path:
    (tmp_path / "taint_engine.py").write_text(engine, encoding="utf-8")
    (tmp_path / "read_taint_slice.py").write_text(slice_, encoding="utf-8")
    return tmp_path


def test_a_new_engine_literal_is_advertised_through_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vocab, "plugin_source_dir", lambda: _bridge(tmp_path))
    taint = _capabilities(capsys)["taint"]
    assert taint["origin_kinds"] == ["brand_new_origin", "parameter"]
    assert taint["pointer_object_kinds"] == ["brand_new_object"]
    assert taint["sink_status"] == ["brand_new_status"]
    assert taint["run_params"] == ["brand_new_echo", "brand_new_knob", "max_depth"]
    assert taint["sink_fields"] == ["brand_new_field", "brand_new_flag", "callee"]


@pytest.mark.parametrize(("engine", "slice_"), [
    (_ENGINE.replace('"status": "brand_new_status"', '"status": compute()'), _SLICE),
    (_ENGINE, _SLICE.replace('run_params["brand_new_echo"] = 1', "run_params.update(x=1)")),
    (_ENGINE.replace('{"max_depth": 1, "brand_new_knob": 2}', "knobs"), _SLICE),
    (_ENGINE.replace("**self._extra(sink)", "**sink"), _SLICE),
], ids=["unreadable_status", "run_params_method_write", "non_literal_run_params",
        "unreadable_sink_spread"])
def test_a_producer_the_crawler_cannot_read_is_refused_not_skipped(tmp_path, engine, slice_):
    with pytest.raises(VocabularyUnresolved):
        vocab.taint_vocabulary(_bridge(tmp_path, engine, slice_))


def test_capabilities_text_lists_the_taint_vocabulary(capsys):
    # The vocabulary is the last block: a heading line, then one indented
    # `field: values` row per field, in the JSON's order and with its values.
    assert bn.cli.main(["capabilities"]) == 0
    heading, *rows = capsys.readouterr().out.rstrip("\n").split("\n\n")[-1].splitlines()
    assert heading and not heading.startswith(" ")
    taint = vocab.taint_vocabulary()
    assert len(rows) == len(FIELDS) == len(taint)
    assert all(row.startswith("  ") for row in rows)
    parsed = [row.removeprefix("  ").split(": ", 1) for row in rows]
    assert [field for field, _ in parsed] == list(taint)
    assert {field: values.split(", ") for field, values in parsed} == taint


def test_capabilities_needs_neither_the_bridge_package_nor_binaryninja():
    probe = (
        "import json, sys, bn.cli\n"
        "bn.cli.main(['capabilities', '--format', 'json'])\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m.split('.')[0] in ('binaryninja', 'bn_agent_bridge'))\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["taint"]["leaf_kinds"]
