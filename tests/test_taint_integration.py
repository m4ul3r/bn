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
from conftest import _BN_CLI, FixtureBuildError, refuse_silent_skip

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



@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("opt", ["-O0", "-O2"])
@pytest.mark.parametrize("bsymbolic", [False, True])
def test_pointer_sinks_over_got_loaded_globals_fail_closed(tmp_path, shared_bn, arch, opt,
                                                           bsymbolic):
    """#931, the re-dogfood of round 3 and the round-5 review: in a PIC
    shared object a preemptible global is reached through the GOT
    (MLIL_IMPORT), bare or nested in pointer arithmetic (`global_buf + i`, a
    struct member), directly or through a returned pointer (fgets, strcpy,
    strchr). Every one was complete. The policy: a sink reading through a
    pointer is complete only for a constant read-only literal, and the
    `pointer_demand_not_certified` leaf lists every object, each on its own
    (the local AND the GOT member of a PHI). `-Wl,-Bsymbolic` (direct
    constant addresses) is the control."""
    cc = "gcc" if arch == "x86_64" else shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc is required for the AArch64 GOT fixture; "
            "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    src = tmp_path / "got.c"
    src.write_text("""
#include <stdio.h>
#include <string.h>
char global_buf[64];
struct rec { int id; char cmd[32]; } global_record;
__attribute__((noinline)) void returned_global(FILE *f) { popen(fgets(global_buf, 0x40, f), "r"); }
__attribute__((noinline)) void got_index(FILE *f, int i) { fgets(global_buf, 64, f); popen(global_buf + i, "r"); }
__attribute__((noinline)) void got_strcpy(const char *s, int i) { popen(strcpy(global_buf + i, s), "r"); }
__attribute__((noinline)) void got_strchr(int i) { popen(strchr(global_buf + i, ' '), "r"); }
__attribute__((noinline)) void got_member(int c) { char *p = strchr(global_record.cmd, ' '); popen(c ? p : "ls", "r"); }
__attribute__((noinline)) void local_or_got(FILE *f, int c) { char buf[32]; char *p = c ? buf : global_record.cmd; popen(fgets(p, 32, f), "r"); }
""")
    binary = tmp_path / f"got_{arch}{opt}{'_sym' if bsymbolic else ''}.so"
    built = subprocess.run(
        [cc, opt, "-g", "-fPIC", "-shared", "-fno-inline", "-fno-stack-protector",
         "-U_FORTIFY_SOURCE", *(["-Wl,-Bsymbolic"] if bsymbolic else []),
         str(src), "-o", str(binary)],
        capture_output=True, text=True, timeout=120.0)
    if built.returncode:
        raise FixtureBuildError(f"GOT fixture compile failed: {built.stderr}")
    target = shared_bn.load(binary, copy=False)

    def objects(o):
        yield o
        for i in o.get("may_point_into", ()):
            yield from objects(i)

    for fn, want in [("returned_global", {"global_buf"}), ("got_index", {"global_buf"}),
                     ("got_strcpy", {"global_buf"}), ("got_strchr", {"global_buf"}),
                     ("got_member", {"global_record"}),
                     ("local_or_got", {"global_record", "buf"})]:
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn,
                          "--sink", "arg:popen:0")
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False, (fn, result)
        [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "pointer_demand_not_certified"]
        named = {o["object"] for top in leaf["objects"] for o in objects(top)
                 if o["kind"] in ("got_or_extern", "local", "global")}
        if bsymbolic:
            assert "buf" in named or fn != "local_or_got", (fn, leaf["objects"])
        else:
            assert want <= named, (fn, leaf["objects"])
            assert all(o.get("slot") for top in leaf["objects"] for o in objects(top)
                       if o["kind"] == "got_or_extern"), (fn, leaf["objects"])


_ROLES_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef int (*system_t)(const char *);
static const char tmpl[64] = "echo placeholder";
__attribute__((noinline)) void direct_system(const char *c) { system(c); }
__attribute__((noinline)) void ind_sys_heap(system_t fn, int fd) { char *p = malloc(64); read(fd, p, 64); fn(p); }
__attribute__((noinline)) void malloc_pointer(char *p) { malloc((size_t)p); }
__attribute__((noinline)) void memcpy_length_pointer(char *d, char *s, char *p) { memcpy(d, s, (size_t)p); }
__attribute__((noinline)) void literal_phi(int fd, int flag) { read(fd, (void *)tmpl, 63); const char *p = flag ? tmpl : "id"; popen(p, "r"); }
__attribute__((noinline)) void offsets(int flag) { char b[16]; memset(b, 'a', sizeof b); b[15] = 0; char *p = b + 8; popen(flag ? p - 8 : p - 4, "r"); }
__attribute__((noinline)) void popen_integer(long p) { popen((char *)p, "r"); }
int main(void) { return 0; }
"""


def _cross_cc(arch: str) -> str:
    cc = "gcc" if arch == "x86_64" else shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc is required for the AArch64 sink-role fixtures; "
            "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    return cc


def _build(cmd: list[str]) -> None:
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0)
    if built.returncode:
        raise FixtureBuildError(f"fixture compile failed: {' '.join(cmd)}: {built.stderr}")


def _make_rodata_segment_writable(src: Path, out: Path) -> None:
    """Set PF_W on the PT_LOAD holding `.rodata`, leaving the section's own
    (read-only) flags alone: the bytes are writable at run time while BN's
    section semantics still say read-only data. ELF64 little-endian."""
    import struct
    data = bytearray(src.read_bytes())
    phoff, shoff = struct.unpack_from("<QQ", data, 0x20)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", data, 0x36)
    strtab = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 0x18)[0]
    rodata = None
    for i in range(shnum):
        o = shoff + i * shentsize
        name = bytes(data[strtab + struct.unpack_from("<I", data, o)[0]:]).split(b"\0", 1)[0]
        if name == b".rodata":
            rodata = struct.unpack_from("<Q", data, o + 0x10)[0]
    assert rodata is not None, "no .rodata section"
    for i in range(phnum):
        o = phoff + i * phentsize
        p_type, p_flags = struct.unpack_from("<II", data, o)
        vaddr, memsz = struct.unpack_from("<Q", data, o + 0x10)[0], struct.unpack_from("<Q", data, o + 0x28)[0]
        if p_type == 1 and vaddr <= rodata < vaddr + memsz:
            struct.pack_into("<I", data, o + 4, p_flags | 2)
    out.write_bytes(data)


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("opt", ["-O0", "-O2"])
def test_sink_model_role_decides_the_pointer_demand(tmp_path, shared_bn, arch, opt):
    """#931 round 6: a sink that reads through a pointer is never complete,
    and the sink MODEL's role for the argument -- not BN's typing -- says
    whether it does. An indirect call pinned to `system` by --resolve-map
    uses system's role (round 5 asked the unresolved call for a type and
    completed it); a pointer-typed value cast into malloc's size or memcpy's
    length is a value (no pointer leaf, complete as on base); a literal is a
    fact in the leaf; an object's offset is the consumer's displacement
    (`p = b + 8; popen(p - 8)` reads b at 0); a command BN recovered only
    from its ABI register (popen's prototype reduced to no parameters) still
    carries the leaf."""
    cc = _cross_cc(arch)
    src = tmp_path / "roles.c"
    src.write_text(_ROLES_SRC)
    binary = tmp_path / f"roles_{arch}{opt}"
    _build([cc, opt, "-g", "-fno-inline", "-fno-builtin", "-U_FORTIFY_SOURCE",
            "-fno-stack-protector", "-fno-optimize-sibling-calls", "-no-pie",
            str(src), "-o", str(binary)])
    target = shared_bn.load(binary, copy=False)

    def query(fn, sink, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn,
                        "--sink", sink, *extra)

    def demand(result):
        return [lf for lf in result["leaves"] if lf["kind"] == "pointer_demand_not_certified"]

    call_addr = _indirect_call_addr(shared_bn, "ind_sys_heap", binary.name)
    rmap = tmp_path / "rmap.json"
    rmap.write_text(json.dumps({call_addr: [_resolve_addr(shared_bn, "system")]}))
    pinned = query("ind_sys_heap", "arg:system:0", "--resolve-map", str(rmap))
    assert pinned["diagnostics"]["safe_to_report_complete_slice"] is False, pinned
    [leaf] = demand(pinned)
    assert [(o["kind"], o.get("callee")) for o in leaf["objects"]] == [("returned", "malloc")], leaf

    for fn, sink in [("malloc_pointer", "arg:malloc:0"), ("memcpy_length_pointer", "arg:memcpy:2")]:
        value = query(fn, sink)
        assert value["leaves"] == [], (fn, value["leaves"])
        assert value["diagnostics"]["safe_to_report_complete_slice"] is True, (fn, value)

    lit = query("literal_phi", "arg:popen:0")
    assert lit["diagnostics"]["safe_to_report_complete_slice"] is False, lit
    [leaf] = demand(lit)
    assert {(o["kind"], o["segment_writable"]) for o in leaf["objects"]} == {("literal", False)}, leaf

    off = query("offsets", "arg:popen:0")
    [leaf] = demand(off)
    assert sorted((o["object"], o["offset"]) for o in leaf["objects"]) == [("b", 0), ("b", 4)], leaf

    if opt == "-O0":
        # O0 keeps the command in the ABI register at the call; at O2 the
        # register's reaching definition is the entry value and the seed is
        # refused outright (never complete either).
        _bn_json(shared_bn, "-t", target, "proto", "set", "popen", "FILE* popen(void)")
        reg = query("popen_integer", "arg:popen:0")
        assert any("recovered from calling-convention register" in a for a in reg["assumptions"]), reg
        assert reg["diagnostics"]["safe_to_report_complete_slice"] is False, reg
        [leaf] = demand(reg)
        assert leaf["recovered_from_register"] is True, leaf


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_literal_in_a_writable_mapping_is_never_complete(tmp_path, shared_bn, arch):
    """#931 round 6: a `.rodata` literal in a PT_LOAD the loader maps
    writable -- `read` overwrites it before `popen` picks it or "id". The
    round-5 literal exception trusted the section semantics and completed
    it. No literal is complete now; the leaf reports the segment's
    writability as a fact."""
    cc = _cross_cc(arch)
    src = tmp_path / "roles.c"
    src.write_text(_ROLES_SRC)
    plain = tmp_path / f"roles_{arch}"
    _build([cc, "-O2", "-g", "-fno-inline", "-fno-builtin", "-U_FORTIFY_SOURCE",
            "-fno-stack-protector", "-fno-optimize-sibling-calls", "-no-pie",
            str(src), "-o", str(plain)])
    binary = tmp_path / f"roles_{arch}_w"
    _make_rodata_segment_writable(plain, binary)
    target = shared_bn.load(binary, copy=False)
    lit = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "literal_phi",
                   "--sink", "arg:popen:0")
    assert lit["diagnostics"]["safe_to_report_complete_slice"] is False, lit
    [leaf] = [lf for lf in lit["leaves"] if lf["kind"] == "pointer_demand_not_certified"]
    assert {(o["kind"], o["segment_writable"]) for o in leaf["objects"]} == {("literal", True)}, leaf


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_untyped_import_sink_is_a_pointer_demand(tmp_path, shared_bn, arch):
    """#931 round 6: popen/system imported from a library with no type
    library prototype, declared integer at the call: the model's role makes
    the command a pointer demand, so the slice is not complete."""
    cc = _cross_cc(arch)
    (tmp_path / "stub.c").write_text(
        "long popen(long a, long b) { return a + b; }\nlong system(long a) { return a; }\n")
    (tmp_path / "untyped.c").write_text("""
extern long popen(long, long);
extern long system(long);
__attribute__((noinline)) void u_param(long cmd) { popen(cmd, (long)"r"); }
__attribute__((noinline)) void u_system(long cmd) { system(cmd); }
void _start(void) { u_param(0); u_system(0); }
""")
    _build([cc, "-shared", "-fPIC", str(tmp_path / "stub.c"), "-o", str(tmp_path / "libstub.so")])
    binary = tmp_path / f"untyped_{arch}"
    _build([cc, "-O0", "-g", "-nostdlib", "-no-pie", str(tmp_path / "untyped.c"),
            f"-L{tmp_path}", "-lstub", "-o", str(binary)])
    target = shared_bn.load(binary, copy=False)
    for fn, sink in [("u_param", "arg:popen:0"), ("u_system", "arg:system:0")]:
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn, "--sink", sink)
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False, (fn, result)
        assert [lf["kind"] for lf in result["leaves"]] == ["pointer_demand_not_certified"], (
            fn, result["leaves"])


_ROLE_FALLBACK_SRC = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
struct pair_carrier { uintptr_t ignored; char *pointer; };
union word_carrier { uintptr_t bits; char *pointer; };
__attribute__((noinline)) int double_sink(double c) { return system((char *)(uintptr_t)c); }
__attribute__((noinline)) int double_caller(double c) { return double_sink(c); }
__attribute__((noinline)) int struct_sink(struct pair_carrier c) { return system(c.pointer); }
__attribute__((noinline)) int struct_caller(struct pair_carrier c) { return struct_sink(c); }
__attribute__((noinline)) int union_sink(union word_carrier c) { return system(c.pointer); }
__attribute__((noinline)) int union_caller(union word_carrier c) { return union_sink(c); }
__attribute__((noinline)) long int_sink(long v) { return v * 3; }
__attribute__((noinline)) long int_caller(long v) {
    void *p = malloc((size_t)v);
    __asm__ volatile("" : : "r"(p) : "memory");
    return int_sink(v) + 1;
}
__attribute__((noinline)) void constant_pointer(long n) {
    FILE *stream = popen((char *)(uintptr_t)1, "r");
    void *allocation = malloc((size_t)n);
    __asm__ volatile("" : : "r"(stream), "r"(allocation) : "memory");
}
__attribute__((noinline)) void direct_literal(void) { popen("ls", "r"); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("opt", ["-O0", "-O2"])
def test_sink_argument_without_a_value_role_is_incomplete(tmp_path, shared_bn, arch, opt):
    """#931 rounds 7-8, the announced fallback: an argument is a value only
    when a model says so. A double, a by-value struct or a union carried into
    an unmodeled helper completed under round 6's type fallback; now each is
    incomplete with `sink_role_unknown`, and so is a plain integer -- until a
    filled-in copy of the leaf's TEMPLATE entry is passed with `--models` to
    the same command. The advice prints no overlay command and no re-run
    (round 8); every `bn` command it does print runs as printed with two
    targets open. The role is decided before the no-def-chain exit: a fixed
    address to popen beside a parameter to malloc's size was refused and
    skipped, letting the malloc slice complete the query; a direct
    `popen("ls")` was a refusal. Both are incomplete with their leaf now."""
    cc = _cross_cc(arch)
    src = tmp_path / "fallback.c"
    src.write_text(_ROLE_FALLBACK_SRC)
    binary = tmp_path / f"fallback_{arch}{opt}"
    _build([cc, opt, "-g", "-fno-inline", "-fno-builtin", "-U_FORTIFY_SOURCE",
            "-fno-stack-protector", "-fno-optimize-sibling-calls", "-no-pie",
            str(src), "-o", str(binary)])
    target = shared_bn.load(binary, copy=False)

    def query(fn, *sinks):
        args = ["-t", target, "taint", "backward", "-f", fn]
        for s in sinks:
            args += ["--sink", s]
        return _bn_json(shared_bn, *args)

    for fn, sink in [("double_caller", "arg:double_sink:0"), ("struct_caller", "arg:struct_sink:0"),
                     ("union_caller", "arg:union_sink:0"), ("int_caller", "arg:int_sink:0")]:
        result = query(fn, sink)
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False, (fn, result)
        assert "sink_role_unknown" in {lf["kind"] for lf in result["leaves"]}, (fn, result["leaves"])

    # Round 8: the advice is guidance. The template, filled in (int_sink has
    # no current model, so no other roles to copy), is passed with --models
    # to the SAME command -- the two-sink query keeps both sinks.
    two = query("int_caller", "arg:int_sink:0", "arg:malloc:0")
    advice = two["diagnostics"]["next_action"]
    assert "bn taint backward" not in advice and ">" not in advice, advice
    [leaf] = [lf for lf in two["leaves"] if lf["kind"] == "sink_role_unknown"]
    entry = leaf["declare_value_role"]["entry"]
    assert leaf["declare_value_role"]["template"] is True, leaf
    entry["int_sink"]["sink"]["class"] = "alloc_size"
    del entry["int_sink"]["OTHER_ROLES"]
    models = tmp_path / "value_roles.json"
    models.write_text(json.dumps(entry))
    declared = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "int_caller",
                        "--sink", "arg:int_sink:0", "--sink", "arg:malloc:0",
                        "--models", str(models))
    assert declared["leaves"] == [], declared["leaves"]
    assert declared["diagnostics"]["safe_to_report_complete_slice"] is True, declared

    # Every `bn` command the advice prints is routed and runs as printed
    # with a second target open.
    shared_bn.load(binary)
    printed = set()
    for text in (advice, leaf["detail"], query("direct_literal", "arg:popen:0")["leaves"][0]["detail"]):
        printed |= set(re.findall(r"`(bn [^`]+)`", text))
    assert any(" taint models " in c for c in printed) and any(" il " in c for c in printed), printed
    for command in printed:
        ran = subprocess.run([*_BN_CLI, *shlex.split(command)[1:]], capture_output=True,
                             text=True, timeout=120.0, env=shared_bn.env())
        assert ran.returncode == 0, (command, ran.stderr)

    both = query("constant_pointer", "arg:popen:0", "arg:malloc:0")
    assert both["diagnostics"]["safe_to_report_complete_slice"] is False, both
    assert [lf["kind"] for lf in both["leaves"]] == ["pointer_demand_not_certified"], both["leaves"]
    assert {s["seeded"] for s in both["sink_status"]} == {True, False}, both["sink_status"]

    lit = query("direct_literal", "arg:popen:0")
    assert lit["diagnostics"]["safe_to_report_complete_slice"] is False, lit
    [leaf] = lit["leaves"]
    assert leaf["kind"] == "pointer_demand_not_certified", leaf
    assert [o["kind"] for o in leaf["objects"]] == ["literal"], leaf



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


_TEMPLATE = "template, fill in the UPPER-CASE words: `"


def _forward_prose(result) -> str:
    """Every prose field a forward result prints: its advice, caveats and notes."""
    parts = [str((result.get("diagnostics") or {}).get("next_action") or "")]
    parts += [str(a) for a in result.get("assumptions") or []]
    for leaf in result.get("leaves") or []:
        parts += [str(leaf.get(k) or "") for k in ("note", "detail")]
    return "\n".join(parts)


def _run_or_check_template(shared_bn, prose: str) -> tuple[int, int]:
    """Run each backticked `bn` command exactly as printed; a command marked as
    a template must instead parse as shell words with no `<...>` placeholder
    and name its blanks in UPPER-CASE (#1033 item 6)."""
    from conftest import _BN_CLI

    ran = templates = 0
    for match in re.finditer(r"`(bn [^`]+)`", prose):
        argv = shlex.split(match.group(1))
        if prose[:match.start()].endswith(_TEMPLATE[:-1]):
            assert not any(c in match.group(1) for c in "<>"), match.group(1)
            assert any(w.isupper() or w.split(":")[-1].isupper() for w in argv), match.group(1)
            templates += 1
            continue
        res = subprocess.run([*_BN_CLI, *argv[1:]], capture_output=True, text=True,
                             env=shared_bn.env(), timeout=300.0)
        assert res.returncode == 0, f"{match.group(1)}: {res.stderr}"
        ran += 1
    return ran, templates


def test_forward_advice_runs_as_printed_or_is_a_marked_template(tmp_path, shared_bn):
    """#1033 item 6 / #1041 item 1: the commands forward taint prints."""
    src = tmp_path / "advice.c"
    src.write_text(r"""
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
struct conn { int fd; char *buf; };
__attribute__((noinline)) int pump(struct conn *c, long have) {
    recv(c->fd, c->buf + have, 64, 0);           /* member buffer: not keyed */
    return c->buf[4];
}
__attribute__((noinline)) int sink_len(long n) {
    char d[8]; memcpy(d, "0123456789abcdef", n); return d[0];
}
__attribute__((noinline)) int deep(int fd) {
    char b[16]; long n = read(fd, b, sizeof b); return sink_len(n);
}
__attribute__((noinline)) int hdr(int fd) {
    char b[32]; struct iovec v = {b, sizeof b};
    struct msghdr m = {0}; m.msg_iov = &v; m.msg_iovlen = 1;
    recvmsg(fd, &m, 0); return b[0];
}
int main(int argc, char **argv) {
    struct conn c = {argc, argv[0]};
    return pump(&c, argc) + deep(argc) + hdr(argc);
}
""")
    binary = tmp_path / "advice"
    _compile(src, binary)
    target = shared_bn.load(binary, copy=False)

    def forward(fn, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn, *extra)

    unkeyed = forward("pump", "--source", "call:recv")
    assert unkeyed["diagnostics"]["safe_to_report_all_clear"] is False
    assert any(lf.get("seed") == "call" for lf in unkeyed["leaves"]), unkeyed["leaves"]
    assert _run_or_check_template(shared_bn, _forward_prose(unkeyed))[1] >= 1

    # Depth 0 stops at sink_len: the advice seeds inside it, filled in.
    stopped = forward("deep", "--source", "call:read", "--max-depth", "0")
    assert _run_or_check_template(shared_bn, _forward_prose(stopped))[0] >= 1

    # arg:recvmsg seeds the header; the nudge names the call: reseed.
    header = forward("hdr", "--source", "arg:recvmsg:1")
    assert _run_or_check_template(shared_bn, _forward_prose(header))[0] >= 1


@pytest.mark.parametrize("flags", [
    # The PR's original build.
    ["-O2", "-g", "-fno-stack-protector", "-fno-builtin", "-foptimize-sibling-calls", "-no-pie"],
    # Compiler defaults (stack protector on): BN gives the wrapper call NO
    # result, so only the call-site check sees `mov x2, x0` (#1050 re-dogfood).
    ["-O2", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0"],
    ["-O2", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0", "-no-pie"],
], ids=["no-stack-protector", "compiler-default", "compiler-default-no-pie"])
def test_aarch64_identity_tail_wrapper_is_a_frontier_not_an_all_clear(tmp_path, shared_bn, flags):
    """#1034 item 5: BN recovers the AArch64 `ident` (a bare `ret`) with a
    valueless return; the flow it carried is a frontier, never an all-clear."""
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc is required for the AArch64 tail-wrapper taint fixture; "
            "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    src = tmp_path / "tail_ident.c"
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
    binary = tmp_path / "tail_ident_a64"
    built = subprocess.run(
        [cc, *flags, str(src), "-o", str(binary)],
        capture_output=True, text=True, timeout=120.0)
    if built.returncode != 0:
        raise FixtureBuildError(f"aarch64 build failed: {built.stderr.strip()}")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "call:read")

    # Judged on what BN recovered: the flow itself, or a frontier naming it.
    lost = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    assert result["reached_sinks"] or lost, result
    if not result["reached_sinks"]:
        assert result["diagnostics"]["safe_to_report_all_clear"] is False
        assert _run_or_check_template(shared_bn, _forward_prose(result))[0] >= 1


def _build_arch(tmp_path, arch: str, name: str, source: str) -> Path:
    """Build *source* at `-O2 -fno-stack-protector -fno-builtin -no-pie` for
    `a64` (the AArch64 cross compiler) or `x64` (host gcc)."""
    cc = shutil.which("aarch64-linux-gnu-gcc" if arch == "a64" else "gcc")
    if cc is None:
        if arch == "a64":
            refuse_silent_skip(
                "aarch64-linux-gnu-gcc is required for this AArch64 taint fixture; "
                "install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
        raise FixtureBuildError("'gcc' not found; the real-BN taint lane is meant to run")
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    binary = tmp_path / f"{name}_{arch}"
    built = subprocess.run(
        [cc, "-O2", "-fno-stack-protector", "-fno-builtin", "-no-pie", str(src), "-o",
         str(binary)], capture_output=True, text=True, timeout=120.0)
    if built.returncode != 0:
        raise FixtureBuildError(f"{arch} build of {name} failed: {built.stderr.strip()}")
    return binary


def test_a_return_register_read_in_a_jump_table_case_is_a_frontier(tmp_path, shared_bn):
    """#1050 review round 8: the wrapper's MLIL call has no result, then a
    dense switch (`br x1`) whose case 1 reads x0 into the copy's length. Since
    round 9 every such call is a frontier, without reading the caller's LLIL."""
    cases = "\n".join(f'    case {i}: puts("c{i}"); break;' for i in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    binary = _build_arch(tmp_path, "a64", "switch_tail", f"""
extern long read(int, void *, unsigned long);
extern void *__memcpy_chk(void *, const void *, unsigned long, unsigned long);
extern int puts(const char *);
__attribute__((noinline, noipa)) long ident(long v) {{ return v; }}
__attribute__((noinline, noipa)) long conditional_tail(long v, int take) {{
    if (take)
        return ident(v);
    return 0;
}}
int main(int argc, char **argv) {{
    char input[32], output[16];
    long n = read(0, input, sizeof(input));
    n = conditional_tail(n, argc & 1);
    switch (argc) {{
{cases}
    case 1: __memcpy_chk(output, input, (unsigned long)n, sizeof(output)); break;
    default: return 1;
    }}
    return output[0];
}}
""")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "call:read")

    lost = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    assert result["reached_sinks"] or lost, result
    if not result["reached_sinks"]:
        assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("arch", ["a64", "x64"])
def test_destinations_bn_did_not_recover_withhold_the_all_clear(tmp_path, shared_bn, arch):
    """#1050 review round 8: three bounded conversions; `b` and `c` are
    incoming pointers already in their argument registers, so the compiler
    does not rewrite them and BN's MLIL lists only the format and `a`. The
    shortfall is disclosed and the all-clear withheld -- or, should BN recover
    `b`, its command sink is found."""
    binary = _build_arch(tmp_path, arch, "fwd_scanf", """
extern int scanf(const char *, ...);
extern int system(const char *);
__attribute__((noinline)) int fwd(char *unused0, char *unused1, char *b, char *c) {
    char a[16];
    int r = scanf("%15s %15s %15s", a, b, c);
    system(b);
    return r + a[0];
}
int main(void) { char b[16], c[16]; return fwd(0, 0, b, c); }
""")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "fwd",
                      "--source", "call:scanf")

    if not result["reached_sinks"]:
        assert [lf["missing"] for lf in result["leaves"]
                if lf["kind"] == "format_destination_unrecovered"], result
        assert result["diagnostics"]["safe_to_report_all_clear"] is False
        assert _run_or_check_template(shared_bn, _forward_prose(result))[0] >= 1


@pytest.mark.parametrize("arch", ["a64", "x64"])
@pytest.mark.parametrize("pad", [600, 5000], ids=["nul-in-read", "nul-beyond-read"])
def test_a_format_prefix_never_trims_a_real_destination(tmp_path, shared_bn, arch, pad):
    """#1050 review round 8: two bounded conversions separated by a non-ASCII
    literal and padding, so BN's ASCII string stops short of the second one.
    Only a read that reached the format's NUL counts as the format; the second
    buffer's command sink is found either way."""
    fmt = '"%15s\\xe9" "' + "A" * pad + '%15s"'
    binary = _build_arch(tmp_path, arch, f"longfmt{pad}", f"""
extern int scanf(const char *, ...);
extern int system(const char *);
__attribute__((noinline)) int two(void) {{
    char a[16], b[16];
    int r = scanf({fmt}, a, b);
    system(b);
    return r + a[0];
}}
int main(void) {{ return two(); }}
""")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "two",
                      "--source", "call:scanf")

    assert "command_injection" in {s["sink"]["class"] for s in result["reached_sinks"]}, result


def test_a_loop_reusing_the_previous_result_is_a_frontier(tmp_path, shared_bn):
    """#1050 review round 9: each iteration copies with the count the wrapper
    returned on the previous one, so the read of x0 sits in the call's own
    block, reached by the back edge. Every descended no-result call with
    tainted arguments is a frontier now, so the loop cannot hide it."""
    binary = _build_arch(tmp_path, "a64", "loop_tail", """
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
    long prev = 0, cur = 0;
    for (int i = 0; i < argc; i++) {
        prev = cur;
        cur = conditional_tail(n, i & 1);
    }
    __memcpy_chk(output, input, (unsigned long)prev, sizeof(output));
    return output[0];
}
""")
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "call:read")

    lost = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    assert result["reached_sinks"] or lost, result
    if not result["reached_sinks"]:
        assert result["diagnostics"]["safe_to_report_all_clear"] is False
        assert _run_or_check_template(shared_bn, _forward_prose(result))[0] >= 1


def test_global_memory_continuation_names_the_global_and_its_consumer(tmp_path, shared_bn):
    """IC03 / TT-PILOT-FW03-01: a bounded copy fills a writable global and a
    later call with no tainted argument reads it into a modeled sink.

    The walk tracks taint per function and hands nothing but arguments across a
    call, so this used to report zero sinks, zero leaves and a TRUE all-clear.
    The boundary is now disclosed -- for the consumer only. A callee that
    references no global, and a global this run never tainted, stay silent, so
    the frontier is evidence-gated rather than a marker on every call after a
    global write."""
    src = tmp_path / "globalcont.c"
    src.write_text(r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char g_dir[256];                 /* filled with attacker bytes */
char g_clean[64];                /* never tainted: the control global */
__attribute__((noinline)) void run_hook(void) {
    char cmd[300];
    snprintf(cmd, sizeof cmd, "%s/hook.sh", g_dir);
    system(cmd);
}
__attribute__((noinline)) void clean_hook(void) { puts(g_clean); }
__attribute__((noinline)) void no_global(void) { putchar('.'); }
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    strncpy(g_dir, argv[1], sizeof g_dir - 1);
    no_global();
    clean_hook();
    run_hook();
    return 0;
}
""")
    binary = tmp_path / "globalcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "param:1")

    frontiers = [lf for lf in result["leaves"]
                 if lf["kind"] == "global_memory_continuation"]
    assert len(frontiers) == 1, result["leaves"]
    [leaf] = frontiers
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]

    # It names the consumer, not the quiet callee and not the clean global's.
    assert leaf["callee"]["display_name"] == "run_hook", leaf
    assert [r["display_name"] for r in leaf["references"]] == ["run_hook"], leaf

    # The address the frontier names is the global `main` WROTE and `run_hook`
    # READS -- proven against the disassembly of both, not against the leaf.
    written = _bn_json(shared_bn, "-t", target, "disasm", "main")
    read_back = _bn_json(shared_bn, "-t", target, "disasm", "run_hook")
    g_dir = int(leaf["global"]["address"], 16)

    def lines(listing):
        return [ln for ln in str(listing["text"]).splitlines() if ln.strip()]

    def mentions(listing, addr):
        return any(f"{addr:x}" in ln.lower() for ln in lines(listing))

    assert mentions(written, g_dir) and mentions(read_back, g_dir), leaf["global"]
    # ...and the cited reference address is an instruction of the consumer that
    # names that very global.
    cited = [ln for ln in lines(read_back)
             if ln.lower().startswith(leaf["references"][0]["at"].lower() + " ")]
    assert cited and f"{g_dir:x}" in cited[0].lower(), (cited, leaf)

    # The reference the frontier cites really loads that global: the consumer
    # seeded on it reaches the modeled sink the forward run stopped short of.
    inside = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", leaf["callee"]["address"], "--source", "arg:snprintf:3")
    assert any(s["sink"]["callee"] == "system" for s in inside["reached_sinks"]), inside

    # Advice is runnable or marked as a template, like every other frontier.
    assert sum(_run_or_check_template(shared_bn, _forward_prose(result))) >= 1
