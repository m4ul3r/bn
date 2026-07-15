"""Tier-2 integration tests for bn taint/dataflow against a real Binary Ninja.

This is the suite's second real-BN lane, and it goes through the same gate as
the first (#590): the ``real_bn`` marker, so BN discovery matches the CLI's own,
``BN_REQUIRE_REAL_TESTS=1`` turns an absent install into a failure instead of a
silent "21 skipped, exit 0", and a missing/broken compiler with BN present is a
loud error rather than another skip. Each target in ``tests/taint_corpus`` is
compiled at test time and checked against its structural ground truth
(``*.EXPECTED.json``). Scoring is structural (callee/class/arg tuples), never
exact addresses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FixtureBuildError

CORPUS = Path(__file__).parent / "taint_corpus"
_BN_CLI = [str(Path(sys.executable).parent / "bn")]

pytestmark = pytest.mark.real_bn


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


def _compile(src: Path, out: Path, extra_cflags: list[str] | None = None) -> None:
    """Compile one corpus target, or raise `FixtureBuildError`.

    An absent compiler is *not* a skip (#590): BN is present by the time we get
    here, so the lane is supposed to run, and swallowing a broken toolchain into
    a skip is how a lane reports green without executing anything.
    """
    cc = "g++" if src.suffix == ".cpp" else "gcc"
    if shutil.which(cc) is None:
        raise FixtureBuildError(
            f"{cc!r} not found; Binary Ninja is installed, so the taint corpus "
            f"lane is meant to run, but {src.name} cannot be built. Install a "
            f"C/C++ toolchain."
        )
    # extra_cflags lands after the defaults so a target can opt into e.g.
    # -O2 -D_FORTIFY_SOURCE=2 (needed to emit __*_chk calls); a later -O wins.
    cmd = [cc, "-O0", "-g", "-fno-stack-protector", "-no-pie",
           *(extra_cflags or []), str(src), "-o", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0)
    if res.returncode != 0:
        raise FixtureBuildError(
            f"compile failed for {src.name} (exit {res.returncode}): {res.stderr.strip()}"
        )


def _resolve_addr(inst: str, name: str) -> str:
    data = _bn_json(inst, "function", "search", name)
    # function search returns a paged envelope {kind, items, total, ...} (#59/#275);
    # tolerate the legacy `functions` key and the older bare-list shape too.
    if isinstance(data, dict):
        matches = data.get("items") or data.get("functions") or []
    else:
        matches = data if isinstance(data, list) else []
    exact = [m for m in matches if str(m.get("name", "")).split("(")[0] == name]
    chosen = (exact or matches)
    assert chosen, f"function {name!r} not found"
    return chosen[0]["address"]


def _indirect_call_addr(inst, function: str, stem: str) -> str:
    """Address of the single indirect call inside *function*, via dataflow
    callgraph. The resolve-map fixtures are built with exactly one indirect call
    so this is unambiguous."""
    fn_addr = _resolve_addr(inst, function)
    cg = _bn_json(inst, "dataflow", "callgraph", fn_addr, "--direction", "callees")
    indirect = [c for c in cg.get("callees", []) if c.get("kind") == "indirect"]
    assert len(indirect) == 1, \
        f"{stem}: expected exactly one indirect call in {function}, got {indirect}"
    return indirect[0]["call_addr"]


def _materialize_resolve_map(inst, directive, tmp_path, stem: str) -> str:
    """Turn a symbolic {"in_function","target"} directive into a concrete
    --resolve-map JSON file: {<indirect_call_addr>: [<target_addr>]}."""
    call_addr = _indirect_call_addr(inst, directive["in_function"], stem)
    target_addr = _resolve_addr(inst, directive["target"])
    rmap = tmp_path / f"{stem}_rmap.json"
    rmap.write_text(json.dumps({call_addr: [target_addr]}))
    return str(rmap)


def _assert_assumptions_contain(result, wanted, stem: str) -> None:
    assumptions = result.get("assumptions", [])
    for sub in wanted or []:
        assert any(sub in a for a in assumptions), \
            f"{stem}: assumption containing {sub!r} not in {assumptions}"


def _expected_files():
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.EXPECTED.json"))


@pytest.mark.parametrize("expected_path", _expected_files(), ids=lambda p: p.stem)
def test_corpus_target(expected_path, tmp_path):
    spec = json.loads(expected_path.read_text())
    stem = expected_path.name[: -len(".EXPECTED.json")]
    src = CORPUS / (stem + (".cpp" if spec.get("lang") == "cpp" else ".c"))
    binary = tmp_path / stem
    _compile(src, binary, spec.get("cflags"))

    inst = _session_start(binary)
    try:
        for case in spec.get("forward", []):
            addr = _resolve_addr(inst, case["function"])
            fwd = ["taint", "forward", "-f", addr, "--source", case["source"]]
            for sc in case.get("sink_classes", []) or []:
                fwd += ["--sink-class", sc]
            if case.get("resolve_map"):
                fwd += ["--resolve-map",
                        _materialize_resolve_map(inst, case["resolve_map"], tmp_path, stem)]
            result = _bn_json(inst, *fwd)
            _assert_assumptions_contain(result, case.get("assumptions_contain"), stem)
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
            bwd = ["taint", "backward", "-f", addr, "--sink", case["sink"]]
            if case.get("resolve_map"):
                bwd += ["--resolve-map",
                        _materialize_resolve_map(inst, case["resolve_map"], tmp_path, stem)]
            result = _bn_json(inst, *bwd)
            _assert_assumptions_contain(result, case.get("assumptions_contain"), stem)
            assert result["slices"], f"{stem}: backward produced no slices"
            kinds = {sl["origin"]["kind"] for sl in result["slices"]}
            assert kinds & set(case["origin_kinds"]), \
                f"{stem}: backward origin {kinds} not in {case['origin_kinds']}"
            for want_leaf in case.get("leaves", []):
                assert any(l["kind"] == want_leaf["kind"] for l in result["leaves"]), \
                    f"{stem}: expected backward leaf {want_leaf} not in {result['leaves']}"
            if case.get("expect_crossed"):
                crossed = [c for sl in result["slices"] for c in (sl.get("crossed_functions") or [])]
                assert case["expect_crossed"] in crossed, \
                    f"{stem}: backward did not cross into caller; crossed={crossed}"

        for case in spec.get("negative", []):
            addr = _resolve_addr(inst, case["function"])
            fwd = ["taint", "forward", "-f", addr, "--source", case["source"]]
            for sc in case.get("sink_classes", []) or []:
                fwd += ["--sink-class", sc]
            result = _bn_json(inst, *fwd)
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
            if case.get("expect_direct_callee"):
                want = case["expect_direct_callee"]
                names = [
                    (c.get("target") or {}).get("name") for c in callees if c.get("kind") == "direct"
                ]
                assert any(n and want in n for n in names), \
                    f"{stem}: expected direct callee {want}, got {names}"
    finally:
        _session_stop(inst)


def test_trace_slices_an_indirect_call_site(tmp_path):
    """`trace` takes an explicit call address, so it already slices an argument at
    an INDIRECT (vtable) call site intraprocedurally -- characterize that the
    conn_send_vtable `conn->ops->send(conn, dst, n)` length arg traces back."""
    src = CORPUS / "conn_send_vtable.c"
    binary = tmp_path / "conn_send_vtable"
    _compile(src, binary)
    inst = _session_start(binary)
    try:
        call_addr = _indirect_call_addr(inst, "emit", "conn_send_vtable")
        result = _bn_json(inst, "trace", "emit", call_addr, "--arg", "2")
        # the length argument (arg 2) has a non-empty backward slice; real key is "trace"
        steps = result["trace"]
        assert steps, f"trace produced no slice at indirect call {call_addr}: {result}"
    finally:
        _session_stop(inst)
