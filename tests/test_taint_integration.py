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
from pathlib import Path
import re
import shlex

import pytest
from conftest import FixtureBuildError, refuse_silent_skip

CORPUS = Path(__file__).parent / "taint_corpus"

pytestmark = pytest.mark.real_bn


def _bn_json(bridge, *args: str, timeout: float = 120.0):
    """One JSON read against the test's target on the shared bridge.

    Delegates to `shared_bn.json` -- which carries the shared `BN_CACHE_DIR`,
    so the call reaches the session bridge and not the per-test cache -- and
    only widens the timeout: the fixture's 60s default is sized for cheap reads,
    while a taint walk over a corpus target is the slowest read in this lane.
    """
    return bridge.json(*args, timeout=timeout)


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


def _resolve_addr(bridge, name: str) -> str:
    data = _bn_json(bridge, "function", "search", name)
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


def _indirect_call_addr(bridge, function: str, stem: str) -> str:
    """Address of the single indirect call inside *function*, via dataflow
    callgraph. The resolve-map fixtures are built with exactly one indirect call
    so this is unambiguous."""
    fn_addr = _resolve_addr(bridge, function)
    cg = _bn_json(bridge, "dataflow", "callgraph", fn_addr, "--direction", "callees")
    indirect = [c for c in cg.get("callees", []) if c.get("kind") == "indirect"]
    assert len(indirect) == 1, \
        f"{stem}: expected exactly one indirect call in {function}, got {indirect}"
    return indirect[0]["call_addr"]


def _materialize_resolve_map(bridge, directive, tmp_path, stem: str) -> str:
    """Turn a symbolic {"in_function","target"} directive into a concrete
    --resolve-map JSON file: {<indirect_call_addr>: [<target_addr>]}."""
    call_addr = _indirect_call_addr(bridge, directive["in_function"], stem)
    target_addr = _resolve_addr(bridge, directive["target"])
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
def test_corpus_target(expected_path, tmp_path, shared_bn):
    spec = json.loads(expected_path.read_text())
    stem = expected_path.name[: -len(".EXPECTED.json")]
    src = CORPUS / (stem + (".cpp" if spec.get("lang") == "cpp" else ".c"))
    binary = tmp_path / stem
    _compile(src, binary, spec.get("cflags"))

    # copy=False: the target was just built into this test's own tmp_path, so
    # the view is already private to the test and a second copy buys nothing.
    # The fixture closes it again, which is what keeps the bare (`--target`-less)
    # commands below unambiguous.
    shared_bn.load(binary, copy=False)

    for case in spec.get("forward", []):
        addr = _resolve_addr(shared_bn, case["function"])
        fwd = ["taint", "forward", "-f", addr, "--source", case["source"]]
        for sc in case.get("sink_classes", []) or []:
            fwd += ["--sink-class", sc]
        if case.get("resolve_map"):
            fwd += ["--resolve-map",
                    _materialize_resolve_map(shared_bn, case["resolve_map"], tmp_path, stem)]
        result = _bn_json(shared_bn, *fwd)
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
        if "all_clear" in case:
            gate = result["diagnostics"]["safe_to_report_all_clear"]
            assert gate is case["all_clear"], \
                f"{stem}: safe_to_report_all_clear={gate}, expected {case['all_clear']}: {result['diagnostics']}"
        assert result["soundness"]

    for case in spec.get("backward", []):
        addr = _resolve_addr(shared_bn, case["function"])
        bwd = ["taint", "backward", "-f", addr, "--sink", case["sink"]]
        if case.get("resolve_map"):
            bwd += ["--resolve-map",
                    _materialize_resolve_map(shared_bn, case["resolve_map"], tmp_path, stem)]
        result = _bn_json(shared_bn, *bwd)
        _assert_assumptions_contain(result, case.get("assumptions_contain"), stem)
        assert result["slices"], f"{stem}: backward produced no slices"
        kinds = {sl["origin"]["kind"] for sl in result["slices"]}
        assert kinds & set(case["origin_kinds"]), \
            f"{stem}: backward origin {kinds} not in {case['origin_kinds']}"
        assert not (kinds & set(case.get("forbid_origin_kinds", []))), \
            f"{stem}: forbidden backward origin kind in {kinds}"
        for want_leaf in case.get("leaves", []):
            assert any(l["kind"] == want_leaf["kind"] for l in result["leaves"]), \
                f"{stem}: expected backward leaf {want_leaf} not in {result['leaves']}"
        for forbid in case.get("forbid_leaves", []):
            assert not any(l["kind"] == forbid["kind"] for l in result["leaves"]), \
                f"{stem}: forbidden backward leaf {forbid} in {result['leaves']}"
        for sub in case.get("leaf_details_contain", []):
            assert any(sub in str(l.get("detail") or "") for l in result["leaves"]), \
                f"{stem}: no backward leaf detail containing {sub!r} in {result['leaves']}"
        for sub in case.get("forbid_leaf_details_contain", []):
            assert not any(sub in str(l.get("detail") or "") for l in result["leaves"]), \
                f"{stem}: forbidden backward leaf detail {sub!r} in {result['leaves']}"
        reasons = [st.get("reason", "") for sl in result["slices"] for st in sl["slice"]]
        for sub in case.get("step_reasons_contain", []):
            assert any(sub in r for r in reasons), \
                f"{stem}: no slice step reason containing {sub!r} in {reasons}"
        for sub in case.get("forbid_step_reasons_contain", []):
            assert not any(sub in r for r in reasons), \
                f"{stem}: forbidden slice step reason {sub!r} in {reasons}"
        if "complete_slice" in case:
            gate = result["diagnostics"]["safe_to_report_complete_slice"]
            assert gate is case["complete_slice"], \
                f"{stem}: safe_to_report_complete_slice={gate}, expected {case['complete_slice']}: {result['diagnostics']}"
        if case.get("expect_crossed"):
            crossed = [c for sl in result["slices"] for c in (sl.get("crossed_functions") or [])]
            assert case["expect_crossed"] in crossed, \
                f"{stem}: backward did not cross into caller; crossed={crossed}"

    for case in spec.get("negative", []):
        addr = _resolve_addr(shared_bn, case["function"])
        fwd = ["taint", "forward", "-f", addr, "--source", case["source"]]
        for sc in case.get("sink_classes", []) or []:
            fwd += ["--sink-class", sc]
        result = _bn_json(shared_bn, *fwd)
        forbidden = set(case.get("forbid_sink_classes", []))
        classes = {s["sink"]["class"] for s in result["reached_sinks"]}
        assert not (classes & forbidden), \
            f"{stem}: false-positive sink classes {classes & forbidden}"
        if "all_clear" in case:
            gate = result["diagnostics"]["safe_to_report_all_clear"]
            assert gate is case["all_clear"], \
                f"{stem}: safe_to_report_all_clear={gate}, expected {case['all_clear']}: {result['diagnostics']}"

    for case in spec.get("callgraph", []):
        addr = _resolve_addr(shared_bn, case["function"])
        result = _bn_json(shared_bn, "dataflow", "callgraph", addr, "--direction", "callees")
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


def test_trace_slices_an_indirect_call_site(tmp_path, shared_bn):
    """`trace` takes an explicit call address, so it already slices an argument at
    an INDIRECT (vtable) call site intraprocedurally -- characterize that the
    conn_send_vtable `conn->ops->send(conn, dst, n)` length arg traces back."""
    src = CORPUS / "conn_send_vtable.c"
    binary = tmp_path / "conn_send_vtable"
    _compile(src, binary)
    # This stem is also a corpus stem, so the same *name* is loaded by
    # test_corpus_target. Only one target is open per test on the shared bridge
    # and the fixture closes it, so the name cannot collide -- but that is also
    # why nothing here selects a target by basename.
    shared_bn.load(binary, copy=False)
    call_addr = _indirect_call_addr(shared_bn, "emit", "conn_send_vtable")
    result = _bn_json(shared_bn, "trace", "emit", call_addr, "--arg", "2")
    # the length argument (arg 2) has a non-empty backward slice; real key is "trace"
    steps = result["trace"]
    assert steps, f"trace produced no slice at indirect call {call_addr}: {result}"



def _run_diagnostic_commands(shared_bn, prose):
    """Execute complete diagnostic commands without supplementing their routing."""
    from conftest import _BN_CLI

    commands = re.findall(r"`(bn [^`]+)`", prose)
    assert commands, prose
    outputs = []
    for command in commands:
        argv = shlex.split(command)
        result = subprocess.run(
            [*_BN_CLI, *argv[1:]], capture_output=True,
            text=True, env=shared_bn.env(), timeout=300.0)
        assert result.returncode == 0, f"{command}: {result.stderr}"
        outputs.append(result.stdout)
    return outputs


def test_trace_local_hints_run_as_printed_with_multiple_targets(tmp_path, shared_bn):
    src = tmp_path / "hint.c"
    src.write_text("""
extern long read(int, void *, unsigned long);
extern int puts(const char *);
__attribute__((noinline)) void handle(void) {
    char line[32]; read(0, line, sizeof(line)); puts(line);
}
int main(void) { handle(); return 0; }
""")
    first, second = tmp_path / "hint one", tmp_path / "hint two"
    _compile(src, first)
    shutil.copy2(first, second)
    target = shared_bn.load(first, copy=False)
    shared_bn.load(second, copy=False)
    name = "handle space$"
    _bn_json(shared_bn, "-t", target, "rename", "handle", name)
    calls = _bn_json(shared_bn, "-t", target, "callsites", "puts", "--within", name)
    site = calls["items"][0]["call_addr"]
    for arg, branch in (("1", "first integer-arg register"), ("6", "STACK")):
        for fmt in ("text", "json"):
            refused = shared_bn.run("-t", target, "trace", name, site,
                                    "--arg", arg, "--format", fmt)
            assert refused.returncode == 2
            prose = refused.stderr + refused.stdout
            if fmt == "json":
                # Decode JSON escapes before extracting commands as printed.
                payload = json.loads(refused.stdout)
                prose = payload["error"]
            assert branch in prose, prose
            commands = re.findall(r"`(bn [^`]+)`", prose)
            assert commands, prose
            assert all(shlex.split(command).count(name) == 1 for command in commands)
            error_commands = _run_diagnostic_commands(shared_bn, prose)
            assert all(re.search(r"\b\w+#\d+", output) for output in error_commands)
    trace = _bn_json(shared_bn, "-t", target, "trace", name, site, "--arg", "0")
    outputs = _run_diagnostic_commands(shared_bn, " ".join(trace["hints"]))
    assert {out.split()[0] for out in outputs} == {"backward", "forward"}
    assert all(f"{name} @ {trace['function_address']}" in out for out in outputs)


def test_backward_frontier_advice_runs_as_printed(tmp_path, shared_bn):
    """#1033 item 6: every command a backward leaf or next_action prints runs
    as printed -- the forward run names its function and buffer, the global
    leaf's inspection names the global, and (#931) the call-fill disclosure's
    `callsites` shows the strcpy, for a local buffer and for a heap object
    (round 1), and the outside-object leaf's `il` (round 2) -- with a second
    target open."""
    src = tmp_path / "advice.c"
    src.write_text("""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char g_host[64];
__attribute__((noinline)) void cond_fill(int c, const char *x) {
    char buf[64];
    if (c)
        strcpy(buf, x);
    popen(buf, "r");
}
__attribute__((noinline)) void from_global(void) {
    char cmd[64];
    sprintf(cmd, "%s", g_host);
    popen(cmd, "r");
}
__attribute__((noinline)) void heap_fill(const char *host) {
    char *b = malloc(64);
    strcpy(b, host);
    popen(b, "r");
}
struct req { int id; char cmd[16]; };
__attribute__((noinline)) void outside(void) {
    struct req s;
    __builtin_memcpy(s.cmd, "ls -la /tmp/xyz", 16);
    s.id = 1;
    char *p = s.cmd;
    popen(p - 12, "r");
}
int main(int argc, char **argv) {
    cond_fill(argc, argv[0]); from_global(); heap_fill(argv[0]); outside(); return 0;
}
""")
    first, second = tmp_path / "advice one", tmp_path / "advice two"
    _compile(src, first, ["-fno-inline"])
    shutil.copy2(first, second)
    target = shared_bn.load(first, copy=False)
    shared_bn.load(second, copy=False)
    for fn, verbs in (("cond_fill", ("forward", "callsites")), ("from_global", ("xrefs",)),
                      ("heap_fill", ("callsites",)), ("outside", ("il",))):
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn,
                          "--sink", "arg:popen:0")
        assert not result["diagnostics"]["safe_to_report_complete_slice"], result
        prose = " ".join([str(lf.get("detail") or "") for lf in result["leaves"]]
                         + [result["diagnostics"]["next_action"]])
        commands = re.findall(r"`(bn [^`]+)`", prose)
        for verb in verbs:
            assert any(verb in shlex.split(c) for c in commands), prose
        outputs = _run_diagnostic_commands(shared_bn, prose)
        if "forward" in verbs:
            assert any(out.startswith("forward taint in cond_fill") for out in outputs), outputs
        if "callsites" in verbs:
            assert any("callee: strcpy" in out for out in outputs), outputs
        if "il" in verbs:
            assert any("- 0xc" in out for out in outputs), outputs


@pytest.mark.parametrize("opt", ["-O0", "-O2"])
def test_renamed_stack_fragment_is_never_a_store_proof(tmp_path, shared_bn, opt):
    """#931/#933 review round 3: without debug info BN splits a stack array
    into fragments; a name-only `bn local rename` makes a fragment
    user-defined without any type claim. Neither that nor anything else turns
    a store proof into a complete slice: `char b[32]` with only 16 bytes
    stored, and a 16-byte array whose upper half `read` overwrites, are
    incomplete before and after the rename, the stores reported as facts."""
    src = tmp_path / "fragment.c"
    src.write_text("""
#include <stdio.h>
#include <unistd.h>
__attribute__((noinline)) void rename_fragment(void) {
    char b[32]; __builtin_memcpy(b, "abcdefghijklmnop", 16); popen(b, "r");
}
__attribute__((noinline)) void upper_half_read(int fd) {
    char b[16]; __builtin_memcpy(b, "abcdefghijklmno", 16); read(fd, b + 8, 8);
    popen(b, "r");
}
int main(void) { return 0; }
""")
    binary = tmp_path / "fragment"
    _compile(src, binary, [opt, "-g0", "-fno-inline", "-U_FORTIFY_SOURCE"])
    target = shared_bn.load(binary, copy=False)

    def gate(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn,
                          "--sink", "arg:popen:0")
        return result["diagnostics"]["safe_to_report_complete_slice"], result["leaves"]

    for fn in ("rename_fragment", "upper_half_read"):
        complete, _ = gate(fn)
        assert complete is False
        locals_ = _bn_json(shared_bn, "-t", target, "local", "list", fn)["items"]
        stack = [lc for lc in locals_ if ":local:stack:" in lc["local_id"]
                 and not lc["name"].startswith("__")]
        assert stack, locals_
        _bn_json(shared_bn, "-t", target, "local", "rename", fn, stack[0]["name"], "cmdbuf")
        complete, leaves = gate(fn)
        assert complete is False, leaves
        assert leaves, leaves



def test_aarch64_upper_half_of_wider_store_is_not_the_allocation(tmp_path, shared_bn):
    """#1041 item 2 on the architecture it was found on: two 4-byte
    sign-extending loads of the upper half of an 8-byte field store feed the
    memcpy lengths; the slice follows the store to its input, never the
    caller's malloc of the object, and claims no complete slice."""
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc is required for the AArch64 wide-load fixture; "
            "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    binary = tmp_path / "bw_wide_load_a64"
    built = subprocess.run(
        [cc, "-O2", "-g", "-fno-stack-protector", "-fno-inline", "-no-pie",
         str(CORPUS / "bw_wide_load.c"), "-o", str(binary)],
        capture_output=True, text=True, timeout=120.0)
    if built.returncode:
        raise FixtureBuildError(f"AArch64 wide-load fixture compile failed: {built.stderr}")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "deser",
                      "--sink", "arg:memcpy:2")
    origins = [sl["origin"] for sl in result["slices"]]
    assert len(origins) == 2, result
    assert all(o["kind"] == "field_load_unresolved" for o in origins), origins
    reasons = [st.get("reason", "") for sl in result["slices"] for st in sl["slice"]]
    assert any("reaching store to field" in r for r in reasons), reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_modeled_write_with_an_unrecovered_pointer_is_a_possible_writer(tmp_path, shared_bn):
    """#1053 rounds 2-3: no call is proof that it did not write a field --
    with full recovery recv is a possible writer named in the slice, and when
    BN recovers `recv(fd)` only, the missing output pointer proves even less.
    Neither may turn the older store's parameter into a complete slice, and
    the stash slot recv may rewrite must never make strcpy a named writer."""
    src = tmp_path / "recovery.c"
    src.write_text("""
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
struct packet { char hdr[8]; int len; };
char *stash;
__attribute__((noinline)) void field_omitted(int fd, struct packet *p, int n,
        char *d, const char *s) {
    p->len = n;
    recv(fd, p, 12, 0);
    memcpy(d, s, ((volatile struct packet *)p)->len);
}
__attribute__((noinline)) void stash_omitted(int fd, const char *host) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    stash = buf;
    recv(fd, &stash, sizeof stash, 0);
    strcpy(stash, host);
    popen(buf, "r");
}
int main(void) { return 0; }
""")
    binary = tmp_path / "recovery"
    _compile(src, binary, ["-O2", "-fno-inline", "-fno-builtin", "-U_FORTIFY_SOURCE"])
    target = shared_bn.load(binary, copy=False)

    def query(fn, sink):
        return _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn, "--sink", sink)

    full = query("field_omitted", "arg:memcpy:2")
    assert full["diagnostics"]["safe_to_report_complete_slice"] is False, full["diagnostics"]
    full_reasons = [st.get("reason", "") for sl in full["slices"] for st in sl["slice"]]
    assert any("field may be filled by source recv" in r for r in full_reasons), full_reasons
    _bn_json(shared_bn, "-t", target, "proto", "set", "recv", "int64_t recv(int fd)")
    field = query("field_omitted", "arg:memcpy:2")
    assert field["diagnostics"]["safe_to_report_complete_slice"] is False, field["diagnostics"]
    assert any(lf["kind"] == "field_load_unresolved" for lf in field["leaves"]), field["leaves"]
    stash = query("stash_omitted", "arg:popen:0")
    assert stash["diagnostics"]["safe_to_report_complete_slice"] is False, stash["diagnostics"]
    reasons = [st.get("reason", "") for sl in stash["slices"] for st in sl["slice"]]
    assert not any("buf written by strcpy" in r for r in reasons), reasons


def test_aarch64_oversized_big_preserves_flow_after_force_analysis(tmp_path, shared_bn):
    """Public 84,004-byte add body; inspect BN's actual analysis, not a split prediction."""
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc is required for the oversized AArch64 taint fixture; "
            "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    src = tmp_path / "oversized_big.c"
    src.write_text(r"""
extern long read(int, void *, unsigned long);
extern void *__memcpy_chk(void *, const void *, unsigned long, unsigned long);
__attribute__((noinline)) long big(const char *p, long n) {
    __asm__ volatile(
        ".rept 21001\n"
        "add %0, %0, #1\n"
        ".endr\n"
        : "+r"(n) : "r"(p));
    return n;
}
int main(void) {
    char input[32], output[16];
    long n = read(0, input, sizeof(input));
    n = big(input, n);
    __memcpy_chk(output, input, (unsigned long)n, sizeof(output));
    return output[0];
}
""")
    binary = tmp_path / "oversized_big"
    built = subprocess.run(
        [cc, "-O1", "-g", "-fno-stack-protector", "-fno-builtin", "-no-pie",
         str(src), "-o", str(binary)],
        capture_output=True, text=True, timeout=120.0)
    if built.returncode:
        raise FixtureBuildError(f"AArch64 oversized fixture compile failed: {built.stderr}")
    shared_bn.load(binary, copy=False)
    main_addr = _resolve_addr(shared_bn, "main")
    # A second live target makes the printed hint's target routing necessary.
    other = tmp_path / "other_big"
    shutil.copy2(binary, other)
    target = str(binary.resolve())
    shared_bn.load(other, copy=False)

    def assert_flow():
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", main_addr,
                          "--source", "call:read", timeout=300.0)
        assert any(s["sink"]["class"] == "fortified_overflow"
                   and s["sink"]["tainted_arg_index"] == 2
                   for s in result["reached_sinks"]), result
        return result

    before = assert_flow()
    force_hint = next(a for a in before["assumptions"] if "--force-analysis" in a)
    _run_diagnostic_commands(shared_bn, force_hint)
    assert_flow()


def test_native_scalar_return_through_conditional_tail_wrapper(tmp_path, shared_bn):
    """Descend into ident's native RET rather than relying on call models."""
    src = tmp_path / "scalar_tail_return.c"
    src.write_text("""
extern long read(int, void *, unsigned long);
extern void *__memcpy_chk(void *, const void *, unsigned long, unsigned long);
__attribute__((noinline, noipa)) long ident(long v) { return v; }
__attribute__((noinline, noipa)) long conditional_tail(long v, int take) {
    if (take)
        return ident(v);
    return 0;
}
int main(int argc, char **argv) {
    char input[32], output[16];
    long n = read(0, input, sizeof(input));
    n = conditional_tail(n, argc & 1);
    __memcpy_chk(output, input, (unsigned long)n, sizeof(output));
    return output[0];
}
""")
    binary = tmp_path / "scalar_tail_return"
    _compile(src, binary, ["-O2", "-fno-builtin", "-foptimize-sibling-calls"])
    target = shared_bn.load(binary, copy=False)
    main_addr = _resolve_addr(shared_bn, "main")
    result = _bn_json(
        shared_bn, "-t", target, "taint", "forward", "-f", main_addr,
        "--source", "call:read")
    assert any(
        s["sink"]["class"] == "fortified_overflow"
        and s["sink"]["tainted_arg_index"] == 2
        for s in result["reached_sinks"]
    ), result
    assert result["stats"]["functions_visited"] == 3, result
    assert not any("has no model" in a for a in result["assumptions"]), result
