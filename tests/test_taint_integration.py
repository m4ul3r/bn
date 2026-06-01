"""Tier-2 integration tests for bn taint/dataflow against a real Binary Ninja.

Skipped unless Binary Ninja is importable (``/opt/binaryninja/python``) and a
C/C++ compiler is on PATH. Each target in ``tests/taint_corpus`` is compiled at
test time and checked against its structural ground truth (``*.EXPECTED.json``).
Scoring is structural (callee/class/arg tuples), never exact addresses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CORPUS = Path(__file__).parent / "taint_corpus"
_BN_PY = Path("/opt/binaryninja/python")
_BN_CLI = [str(Path(sys.executable).parent / "bn")]

_has_bn = _BN_PY.is_dir()
_has_cc = shutil.which("gcc") is not None
_has_cxx = shutil.which("g++") is not None

pytestmark = pytest.mark.skipif(not (_has_bn and _has_cc), reason="Binary Ninja or gcc not available")


def _bn(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*_BN_CLI, *args], capture_output=True, text=True, timeout=timeout)


def _bn_json(inst: str, *args: str):
    res = _bn("--instance", inst, *args, "--format", "json")
    assert res.returncode == 0, f"bn {args} failed: {res.stderr}\n{res.stdout}"
    return json.loads(res.stdout)


def _session_start(binary: Path) -> str:
    res = _bn("session", "start", str(binary), "--format", "json")
    assert res.returncode == 0, f"session start failed: {res.stderr}"
    return json.loads(res.stdout)["instance_id"]


def _session_stop(inst: str) -> None:
    _bn("session", "stop", inst, timeout=30.0)


def _compile(src: Path, out: Path) -> None:
    if src.suffix == ".cpp":
        cc = ["g++"]
    else:
        cc = ["gcc"]
    cmd = [*cc, "-O0", "-g", "-fno-stack-protector", "-no-pie", str(src), "-o", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0)
    assert res.returncode == 0, f"compile failed for {src.name}: {res.stderr}"


def _resolve_addr(inst: str, name: str) -> str:
    data = _bn_json(inst, "function", "search", name)
    matches = data if isinstance(data, list) else []
    exact = [m for m in matches if str(m.get("name", "")).split("(")[0] == name]
    chosen = (exact or matches)
    assert chosen, f"function {name!r} not found"
    return chosen[0]["address"]


def _expected_files():
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.EXPECTED.json"))


@pytest.mark.parametrize("expected_path", _expected_files(), ids=lambda p: p.stem)
def test_corpus_target(expected_path, tmp_path):
    spec = json.loads(expected_path.read_text())
    if spec.get("lang") == "cpp" and not _has_cxx:
        pytest.skip("g++ not available")

    stem = expected_path.name[: -len(".EXPECTED.json")]
    src = CORPUS / (stem + (".cpp" if spec.get("lang") == "cpp" else ".c"))
    binary = tmp_path / stem
    _compile(src, binary)

    inst = _session_start(binary)
    try:
        for case in spec.get("forward", []):
            addr = _resolve_addr(inst, case["function"])
            result = _bn_json(inst, "taint", "forward", "-f", addr, "--source", case["source"])
            got = result["reached_sinks"]
            for want in case.get("sinks", []):
                assert any(
                    s["sink"]["callee"] == want["callee"]
                    and s["sink"]["class"] == want["class"]
                    and s["sink"]["tainted_arg_index"] == want["arg"]
                    for s in got
                ), f"{stem}: expected sink {want} not in {[s['sink'] for s in got]}"
            for want_leaf in case.get("leaves", []):
                assert any(l["kind"] == want_leaf["kind"] for l in result["leaves"]), \
                    f"{stem}: expected leaf {want_leaf} not in {result['leaves']}"
            assert result["soundness"]

        for case in spec.get("backward", []):
            addr = _resolve_addr(inst, case["function"])
            result = _bn_json(inst, "taint", "backward", "-f", addr, "--sink", case["sink"])
            assert result["slices"], f"{stem}: backward produced no slices"
            kinds = {sl["origin"]["kind"] for sl in result["slices"]}
            assert kinds & set(case["origin_kinds"]), \
                f"{stem}: backward origin {kinds} not in {case['origin_kinds']}"

        for case in spec.get("negative", []):
            addr = _resolve_addr(inst, case["function"])
            result = _bn_json(inst, "taint", "forward", "-f", addr, "--source", case["source"])
            forbidden = set(case.get("forbid_sink_classes", []))
            classes = {s["sink"]["class"] for s in result["reached_sinks"]}
            assert not (classes & forbidden), \
                f"{stem}: false-positive sink classes {classes & forbidden}"

        for case in spec.get("callgraph", []):
            addr = _resolve_addr(inst, case["function"])
            result = _bn_json(inst, "dataflow", "callgraph", addr, "--direction", "callees")
            callees = result.get("callees", [])
            if case.get("expect_indirect"):
                indirect = [c for c in callees if c.get("kind") == "indirect"]
                assert indirect, f"{stem}: expected an indirect callee, got {callees}"
                # honest degradation: every indirect callee carries a resolution verdict
                assert all("resolution" in c for c in indirect)
    finally:
        _session_stop(inst)
