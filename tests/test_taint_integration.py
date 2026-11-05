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


def _dynamic_symbols(path: Path) -> str:
    """The binary's dynamic symbol table, or a loud build error.

    Used only to prove which libc spelling the compiler actually emitted: an
    environment where the C23 spelling is absent must skip (loudly under
    strict mode), never silently pass as if the alias had been exercised.
    """
    nm = shutil.which("nm")
    if nm is None:
        raise FixtureBuildError(
            f"'nm' not found; Binary Ninja is installed, so this lane is meant "
            f"to run, but {path.name}'s imports cannot be inspected. Install binutils."
        )
    res = subprocess.run([nm, "-D", str(path)], capture_output=True, text=True,
                         timeout=120.0)
    if res.returncode != 0:
        raise FixtureBuildError(
            f"nm -D failed for {path.name} (exit {res.returncode}): {res.stderr.strip()}"
        )
    return res.stdout


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
        # Every origin of every slice path: one path may end at several.
        kinds = {o["kind"] for sl in result["slices"] for o in sl["origins"]}
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
    origins = [o for sl in result["slices"] for o in sl["origins"]]
    assert len(origins) == 2, result
    assert all(o["kind"] == "field_load_unresolved" for o in origins), origins
    reasons = [st.get("reason", "") for sl in result["slices"] for st in sl["slice"]]
    assert any("reaching store to field" in r for r in reasons), reasons
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


@pytest.mark.parametrize("opt", ["-O1", "-O2"])
def test_out_param_local_read_under_a_conversion_is_never_complete(tmp_path, shared_bn, opt):
    """P2-01: a memcpy length read back from a local an in-binary helper
    wrote through `&width`, lifted as `width_1 = zx.q(width @ mem#k)`, came
    back origin `entry`, no leaf and a complete slice. The read is the local's
    contents, so the helper is its unmodeled writer, exactly as for a bare
    by-name read."""
    binary = tmp_path / f"bw_outparam_width{opt}"
    _compile(CORPUS / "bw_outparam_width.c", binary, [opt, "-fno-inline"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "copy_field",
                      "--sink", "arg:memcpy:2")
    leaves = [lf for lf in result["leaves"] if lf["kind"] == "address_of_local_unresolved"]
    # The conversion form this fixture exists for is what BN lifted.
    assert any("zx." in lf["il_text"] and "width @ mem#" in lf["il_text"] for lf in leaves), \
        result["leaves"]
    assert {lf["buffer"] for lf in leaves} == {"width"}, leaves
    origins = [o for sl in result["slices"] for o in sl["origins"]]
    assert ("address_of_local_unresolved", "width", "parse_width") in {
        (o["kind"], o.get("buffer"), o.get("callee")) for o in origins}, origins
    assert not any(o["kind"] == "entry" for o in origins), origins
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False


def test_every_caller_site_is_followed_or_named_and_the_cap_is_address_ordered(tmp_path, shared_bn):
    """P1-06 / P2-14: `quote_arg` has ten call sites, two of which reach one
    `pick_name()` origin in `edit_entry` (directly and through `open_alt`);
    rows keyed on the origin kept one site per origin and named neither of
    the others. `put_word` has twenty: the cap follows the sixteen lowest
    addresses and names the other four, in address order."""
    binary = tmp_path / "bw_caller_sites"
    _compile(CORPUS / "bw_caller_sites.c", binary, ["-fno-inline"])
    target = shared_bn.load(binary, copy=False)

    def call_sites(callee):
        rows = _bn_json(shared_bn, "-t", target, "callsites", callee)["items"]
        return {r["call_addr"]: r["containing_function"]["name"] for r in rows}

    def ascent(fn, sink, param):
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn, "--sink", sink)
        followed = {st["address"] for sl in result["slices"] for st in sl["slice"]
                    if st.get("reason") == f"passed as arg {param} to {fn}"}
        leaves = [lf for lf in result["leaves"]
                  if lf["kind"] == "caller_sites_truncated" and lf["function"]["name"] == fn]
        return result, followed, leaves

    quote_sites = call_sites("quote_arg")
    assert len(quote_sites) == 10, quote_sites
    _, followed, leaves = ascent("quote_arg", "arg:strcpy:1", 0)
    named = {row["address"] for lf in leaves for row in lf["unfollowed"]}
    assert followed | named == set(quote_sites), sorted(set(quote_sites) - followed - named)
    assert {quote_sites[a] for a in followed} >= {"edit_entry", "open_alt"}, followed

    word_sites = sorted(call_sites("put_word"), key=lambda a: int(a, 16))
    assert len(word_sites) == 20, word_sites
    result, followed, leaves = ascent("put_word", "arg:memcpy:2", 1)
    assert followed == set(word_sites[:16]), sorted(followed)
    [leaf] = leaves
    assert [row["address"] for row in leaf["unfollowed"]] == word_sites[16:]
    assert (leaf["unfollowed_total"], leaf["reasons"]) == (4, {"caller_cap": 4})
    assert result["stats"]["truncation_cause"] == ["caller_cap"]


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
        parts.append(str(leaf.get("detail") or ""))
    return "\n".join(parts)


def _printed_commands(prose: str) -> list[tuple[int, str]]:
    """``(offset, command)`` for each `bn` command *prose* prints, read as
    the code span `code_span` writes: a fence one backtick longer than the
    longest run inside, and one CommonMark pad space each side when the
    command starts or ends with a backtick. A one-backtick reading stopped
    at a selector's own backtick and checked only a prefix."""
    from bn.formatters import CODE_SPAN_RE

    out = []
    for match in CODE_SPAN_RE.finditer(prose):
        body = match.group(2)
        if len(body) > 2 and body[0] == body[-1] == " " and body.strip():
            body = body[1:-1]
        if body.startswith("bn "):
            out.append((match.start(), body))
    return out


def _run_or_check_template(shared_bn, prose: str) -> tuple[int, int]:
    """Run each fenced `bn` command exactly as printed; a command marked as
    a template must instead parse as shell words with no `<...>` placeholder
    and name its blanks in UPPER-CASE (#1033 item 6)."""
    from conftest import _BN_CLI

    ran = templates = 0
    for start, command in _printed_commands(prose):
        argv = shlex.split(command)
        if prose[:start].endswith(_TEMPLATE[:-1]):
            assert not any(c in command for c in "<>"), command
            assert any(w.isupper() or w.split(":")[-1].isupper() for w in argv), command
            templates += 1
            continue
        res = subprocess.run([*_BN_CLI, *argv[1:]], capture_output=True, text=True,
                             env=shared_bn.env(), timeout=300.0)
        assert res.returncode == 0, f"{command}: {res.stderr}"
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
    # `c->buf + have` is a loaded buffer plus the integer parameter `have`: no
    # parameter is the buffer, so no caller is offered as its owner.
    assert not [lf for lf in unkeyed["leaves"] if lf.get("buffer_param")], unkeyed["leaves"]
    # The advice either runs as printed (the helper runs each such command and
    # asserts it succeeds -- a wrapper's caller reseed) or is a marked template
    # (seed a value read from the buffer); never neither.
    assert sum(_run_or_check_template(shared_bn, _forward_prose(unkeyed))) >= 1

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


def _build_arch(tmp_path, arch: str, name: str, source: str,
                opt: tuple[str, ...] = ("-O2",)) -> Path:
    """Build *source* at *opt* (default `-O2`) plus `-fno-stack-protector
    -fno-builtin -no-pie` for `a64` (the AArch64 cross compiler) or `x64`
    (host gcc)."""
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
        [cc, *opt, "-fno-stack-protector", "-fno-builtin", "-no-pie", str(src), "-o",
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


_FIVE_CONVERSION_SOURCE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Five conversions, four modeled destinations. `e` is the format-proven fifth
   output; the builtin sscanf model's unrolled run ends at *arg:5, so nothing
   seeds it and `system(e)` is invisible without the format shortfall leaf. */
__attribute__((noinline)) int parse_five(const char *line) {
    int a, b, c, d;
    char e[8];
    int r = sscanf(line, "%d %d %d %d %7s", &a, &b, &c, &d, e);
    system(e);
    return r + a;
}

/* Exactly four conversions: the modeled run covers the call, so this is the
   capped optional-slot shape and must stay silent. */
__attribute__((noinline)) int parse_cap4(const char *line) {
    int a, b, c, d;
    return sscanf(line, "%d %d %d %d", &a, &b, &c, &d);
}

/* The parse input is unrelated to the selected source: `line` is only
   returned. The five-conversion gap is real but cannot hide a flow from this
   source, so it must not become a blocking frontier. */
__attribute__((noinline)) int parse_unrelated(const char *line) {
    int a, b, c, d;
    char e[8];
    int r = sscanf("1 2 3 4 x", "%d %d %d %d %7s", &a, &b, &c, &d, e);
    return r + (line ? *line : 0);
}

/* A source-level scanf under C23 (the `call:` preset path). */
__attribute__((noinline)) int parse_scan_preset(void) {
    int a, b, c, d;
    char e[8];
    return scanf("%d %d %d %d %7s", &a, &b, &c, &d, e) + e[0];
}

int main(void) {
    return parse_five("1 2 3 4 x") + parse_cap4("1 2 3 4")
         + parse_unrelated("z") + parse_scan_preset();
}
"""


def _format_holes(result) -> list[dict]:
    """The format-proven destination leaves: unrecovered arguments or a model
    arity residual, one cause each (P2-12)."""
    return [lf for lf in result["leaves"]
            if lf["kind"] in ("format_destination_unrecovered", "scanf_arity_residual")]


def _lost(leaf) -> int:
    return leaf["missing"] if leaf["kind"] == "format_destination_unrecovered" else leaf["residual"]


@pytest.mark.parametrize(("cflags", "symbol"), [
    (["-std=gnu17"], "__isoc99_sscanf"),
    (["-std=c2x"], "__isoc23_sscanf"),
], ids=["c99", "c23"])
def test_format_proven_fifth_destination_withholds_the_all_clear(tmp_path, shared_bn,
                                                                 cflags, symbol):
    """F6 on a real binary: four modeled destinations, a five-conversion
    constant format. The fifth output (`e`, whose `system()` call is the
    consequence) is lost by the arity-capped model whether BN recovered four
    arguments or all five (the fifth is a stack argument on x86-64). Before
    the independent format comparison no rule named that index, so the run was
    a confident all-clear with the sink invisible."""
    src = tmp_path / "scan5.c"
    src.write_text(_FIVE_CONVERSION_SOURCE)
    binary = tmp_path / ("scan5_" + symbol.lstrip("_"))
    _compile(src, binary, cflags)
    imports = _dynamic_symbols(binary)
    if symbol not in imports:
        refuse_silent_skip(
            f"{symbol} is not emitted by this compiler/glibc; the C23 alias is "
            "exercised by its synthetic symbol control, but this native spelling "
            "needs glibc 2.38+ and a compiler with -std=c2x"
        )
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "parse_five",
                      "--source", "param:0")
    # Which kind names the lost fifth destination depends on what BN
    # recovered: an unrecovered argument, or a recovered one past the model's
    # four-slot run. Exactly one leaf, one cause, counts the one destination.
    [leaf] = _format_holes(result)
    assert leaf["destinations"] == 5 and _lost(leaf) == 1, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # The lost fifth destination is not seeded, so its command sink is not
    # reached -- the leaf is what stops the run from reading as clean.
    assert not result["reached_sinks"], result["reached_sinks"]

    capped = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "parse_cap4",
                      "--source", "param:0")
    assert not _format_holes(capped), capped["leaves"]
    assert capped["diagnostics"]["safe_to_report_all_clear"] is True, capped["diagnostics"]

    unrelated = _bn_json(shared_bn, "-t", target, "taint", "forward",
                         "-f", "parse_unrelated", "--source", "param:0")
    # The constant-input parse is unrelated to `--source param:0`, so the new
    # format shortfall must not become a frontier. (A call that carries more
    # actual destinations than the model run may still hit the pre-existing
    # #851 residual assumption; that is not this check's leaf, and `leaves`
    # stays empty.)
    assert unrelated["leaves"] == [], unrelated["leaves"]

    # Both ISO spellings now resolve a plain-name preset (W07-15).
    preset = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "parse_scan_preset", "--source", "call:scanf")
    [short] = _format_holes(preset)
    # The leaf names the symbol the binary calls; the plain model is `model`.
    called = symbol.replace("sscanf", "scanf")
    assert (short["callee"], short.get("model"), _lost(short)) == (called, "scanf", 1), short
    assert preset["diagnostics"]["safe_to_report_all_clear"] is False

    # Backward, the per-call-site rows follow the same rule (R3 residual):
    # the query `arg:sscanf:0` is echoed in `sinks[]`, but the row for the
    # call names the symbol it calls and the model beside it.
    sliced = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "parse_five",
                      "--sink", "arg:sscanf:0")
    assert [s["callee"] for s in sliced["sinks"]] == ["sscanf"]
    assert {(r["callee"], r.get("model")) for r in sliced["sink_status"]} == {(symbol, "sscanf")}
    assert {(sl["sink"]["callee"], sl["sink"].get("model")) for sl in sliced["slices"]} <= {
        (symbol, "sscanf")}


def test_a_synthetic_c23_alias_symbol_uses_the_same_model(tmp_path, shared_bn):
    """Portable control for the C23 alias: define `__isoc23_sscanf` in the
    binary itself (asm label, no libc 2.38 requirement) and call it with a
    five-conversion constant format. The call resolves to the canonical sscanf
    model, so the same F6 shortfall is disclosed on any glibc."""
    src = tmp_path / "synth_c23.c"
    src.write_text(r"""
#include <stdio.h>
#include <stdlib.h>

int synth_sscanf(const char *, const char *, int *, int *, int *, int *, char *)
    __asm__("__isoc23_sscanf");
__attribute__((noinline)) int synth_sscanf(const char *s, const char *f,
                                           int *a, int *b, int *c, int *d, char *e) {
    return sscanf(s, f, a, b, c, d, e);
}

__attribute__((noinline)) int parse_synth(const char *line) {
    int a, b, c, d;
    char e[8];
    int r = synth_sscanf(line, "%d %d %d %d %7s", &a, &b, &c, &d, e);
    system(e);
    return r + a;
}

int main(void) { return parse_synth("1 2 3 4 x"); }
""")
    binary = tmp_path / "synth_c23"
    _compile(src, binary)
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "parse_synth",
                      "--source", "param:0")
    [leaf] = _format_holes(result)
    assert leaf["destinations"] == 5 and _lost(leaf) == 1, leaf
    # `callee` is the symbol the binary calls; `model` the plain model the
    # alias resolved to.
    assert (leaf["callee"], leaf.get("model")) == ("__isoc23_sscanf", "sscanf"), leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


_FORWARDED_SCAN_SOURCE = r"""
#include <stdio.h>
#define N __attribute__((noinline, noipa))
N int scan_fwd(const char *unused, char *a, char *b, char *c) {
    int r = scanf("%15s %15s %15s", a, b, c);
    return r + 1;
}
N int scan_tail(const char *unused, char *a, char *b, char *c) {
    return scanf("%15s %15s %15s", a, b, c);
}
N int parse_fwd(const char *in, int unused, char *a, char *b, char *c) {
    int r = sscanf(in, "%s %s %s", a, b, c);
    return r + 1;
}
int main(int argc, char **argv) {
    char a[16], b[16], c[16];
    if (argc < 2) return scan_fwd(0, a, b, c) + scan_tail(0, a, b, c);
    return parse_fwd(argv[1], 0, a, b, c);
}
"""


@pytest.mark.parametrize(("std", "spelling"), [("gnu17", "__isoc99_"), ("c2x", "__isoc23_")],
                         ids=["c99", "c23"])
def test_forwarded_scanf_destinations_bn_did_not_recover_are_a_frontier(tmp_path, shared_bn,
                                                                      std, spelling):
    """P2-12 on a real binary: at -O2 each conversion's destination is a
    forwarded parameter, so BN recovers the scanf call with only its format.
    A `call:`/`ret:` preset used to be refused outright ("no taint sources
    resolved") -- or, for `ret:`, answered with an all-clear -- discarding the
    leaf that names the three lost destinations. Either spelling of the ISO
    alias is reached by the plain name."""
    src = tmp_path / "scan_fwd.c"
    src.write_text(_FORWARDED_SCAN_SOURCE)
    binary = tmp_path / f"scan_fwd_{std}"
    _compile(src, binary, ["-O2", "-fno-builtin", f"-std={std}"])
    if f"{spelling}scanf" not in _dynamic_symbols(binary):
        refuse_silent_skip(f"{spelling}scanf is not emitted by this compiler/glibc "
                           "(the C23 spelling needs glibc 2.38+ and -std=c2x)")
    target = shared_bn.load(binary, copy=False)

    for fn, source in (("scan_fwd", "call:scanf"), ("scan_fwd", "ret:scanf"),
                       ("scan_fwd", f"call:{spelling}scanf"), ("scan_tail", "call:scanf")):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", source)
        holes = [lf for lf in result["leaves"] if lf["kind"] == "format_destination_unrecovered"]
        assert [(h["callee"], h.get("model"), h["destinations"], h["missing"]) for h in holes] == \
            [(f"{spelling}scanf", "scanf", 3, 3)], (fn, source, result["leaves"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, source)

    # The param-seeded twin: the format itself is the sink, and the lost
    # destinations are still named.
    parsed = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "parse_fwd",
                      "--source", "param:0")
    assert {(s["sink"]["callee"], s["sink"]["callee_symbol"], s["sink"].get("model"),
             s["sink"]["class"]) for s in parsed["reached_sinks"]} == \
        {(f"{spelling}sscanf", f"{spelling}sscanf", "sscanf", "overflow_unbounded")}, \
        parsed["reached_sinks"]
    assert [lf["missing"] for lf in parsed["leaves"]
            if lf["kind"] == "format_destination_unrecovered"] == [3], parsed["leaves"]


def test_builtin_regex_and_fortify_guard_models_on_a_real_binary(tmp_path, shared_bn):
    """W10-01: a copy sized by a regexec match offset reaches its sink with
    the builtin catalog (no overlay). W02-06: slicing the `__strcpy_chk`
    destination size answers with the compile-time constant, not a pointer
    demand with a role-unknown leaf."""
    src = tmp_path / "rx.c"
    src.write_text(r"""
#include <regex.h>
#include <stdio.h>
#include <string.h>
#define N __attribute__((noinline, noipa))
static regex_t re;
N int get_field(const char *s, char *out) {
    regmatch_t m;
    if (regexec(&re, s, 1, &m, 0)) return -1;
    memcpy(out, s, m.rm_eo);
    return 0;
}
N void save_name(const char *s) {
    char d[0x120];
    strcpy(d, s);
    puts(d);
}
int main(int argc, char **argv) {
    char out[32];
    regcomp(&re, "(a+)", REG_EXTENDED);
    save_name(argv[0]);
    return argc > 1 ? get_field(argv[1], out) : 0;
}
""")
    binary = tmp_path / "rx"
    _compile(src, binary, ["-O2", "-D_FORTIFY_SOURCE=2"])
    assert "__strcpy_chk" in _dynamic_symbols(binary)
    target = shared_bn.load(binary, copy=False)

    fwd = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "get_field",
                   "--source", "param:0")
    assert [(s["sink"]["callee"], s["sink"]["class"], s["sink"]["tainted_arg_index"])
            for s in fwd["reached_sinks"]] == [("memcpy", "overflow_len", 2)], fwd

    bwd = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "save_name",
                   "--sink", "arg:__strcpy_chk:2")
    assert [(r["status"], r["scalar"]["value"]) for r in bwd["sink_status"]] == \
        [("constant", 0x120)], bwd["sink_status"]
    assert "sink_role_unknown" not in {lf["kind"] for lf in bwd["leaves"]}, bwd["leaves"]


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


def test_aarch64_whole_slot_overwrite_has_no_entry_origin(tmp_path, shared_bn):
    """W06-08: an 8-byte store into a slot the debug info types as a 4-byte
    local lifts to `rows#3:0.q = x @ rows#2`, `rows#2 = phi(rows#0, rows#1)`.
    The reload reads only that store, so its origin is the call it stored --
    following the old version invented an `entry` origin (and the earlier
    writer's call) for a value the function defined."""
    binary = _build_arch(tmp_path, "a64", "slot_reuse", """
extern void *malloc(unsigned long);
__attribute__((noinline, noipa)) int count_rows(int k) { return k * 3; }
__attribute__((noinline, noipa)) long count_cols(int k) { return k * 5L; }
__attribute__((noinline, noipa)) void note_rows(int v) { (void)v; }
void *alloc_grid(int argc) {
    int rows;
    if (argc > 1) {
        rows = count_rows(argc);
        note_rows(rows);
    }
    *(long *)&rows = count_cols(argc);
    return malloc(*(long *)&rows);
}
int main(int argc, char **argv) {
    (void)argv;
    return alloc_grid(argc) != 0;
}
""", opt=("-O0", "-g", "-fno-strict-aliasing"))
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "alloc_grid",
                      "--sink", "arg:malloc:0")

    assert len({sl["sink"]["address"] for sl in result["slices"]}) == 1, result["slices"]
    origins = {(o["kind"], o.get("callee"))
               for sl in result["slices"] for o in (sl.get("origins") or [sl["origin"]])}
    assert origins == {("call", "count_cols")}, result["slices"]



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
    # ...and the cited reference is an instruction of the consumer, classified
    # by what it does with the bytes: `snprintf` is handed the address, so the
    # site named is that CALL, not the lea/mov that merely materialises the
    # pointer (round-3 review F3 -- a reference is only a consumer when it can
    # read, and the leaf cites the use that reads).
    assert leaf["references"][0]["access"] == "call_arg", leaf
    cited = [ln for ln in lines(read_back)
             if ln.lower().startswith(leaf["references"][0]["at"].lower() + " ")]
    assert cited and "call" in cited[0].lower(), (cited, leaf)

    # The reference the frontier cites really loads that global: the consumer
    # seeded on it reaches the modeled sink the forward run stopped short of.
    inside = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", leaf["callee"]["address"], "--source", "arg:snprintf:3")
    assert any(s["sink"]["callee"] == "system" for s in inside["reached_sinks"]), inside

    # Advice is runnable or marked as a template, like every other frontier.
    assert sum(_run_or_check_template(shared_bn, _forward_prose(result))) >= 1


def _nm_address(path: Path, name: str) -> int:
    """A defined symbol's address from the binary's own symbol table."""
    nm = shutil.which("nm")
    if nm is None:
        raise FixtureBuildError(
            f"'nm' not found; Binary Ninja is installed, so this lane is meant "
            f"to run, but {path.name}'s symbols cannot be read. Install binutils.")
    res = subprocess.run([nm, str(path)], capture_output=True, text=True, timeout=120.0)
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise FixtureBuildError(f"{name} not defined in {path.name}: {res.stderr.strip()}")


def test_global_store_frontier_and_declared_neighbour(tmp_path, shared_bn):
    """P1-04 / P1-05 on `taint_corpus/fwd_global_store_exit.c`.

    `fill` stores the tainted `read()` count into `g_len` and returns; `use`
    copies with that length, but the walk never ascends past `fill`.
    `fill_then_log` does the same and then calls `log_banner`, whose only
    global is `g_banner` -- laid out immediately before `g_len`, declared 16
    bytes, read by an `fputs` of unstated length.

    Base (632891c9): both seeds end in an unkeyed `coarse_memory_store` at the
    store with `safe_to_report_all_clear: false`, and no continuation.
    Head (b075c82c): the keyed store is only a `global_approx` assumption, so
    `fill` answers `safe_to_report_all_clear: true` with no leaf, while
    `fill_then_log` raises two `global_memory_continuation` leaves at the
    `log_banner` call citing `log_banner` as a proven (`call_arg`) reader of
    `g_len` and `g_buf` -- neither of which it touches.
    Fixed: the store is a blocking `coarse_memory_store` naming `g_len` and
    `use` at the instruction BN records the reference at, and `log_banner` is
    cited by nothing."""
    binary = tmp_path / "fwd_global_store_exit"
    _compile(CORPUS / "fwd_global_store_exit.c", binary)
    target = shared_bn.load(binary, copy=False)
    g_len = _nm_address(binary, "g_len")
    xrefs = _bn_json(shared_bn, "-t", target, "xrefs", hex(g_len))
    use_sites = {r["address"] for r in xrefs["items"]
                 if r.get("kind") == "code" and r.get("function") == "use"}
    assert use_sites, xrefs

    for fn in ("fill", "fill_then_log"):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", "call:read")
        assert result["reached_sinks"] == [], (fn, result["reached_sinks"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, \
            (fn, result["diagnostics"])
        [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"
                   and (lf.get("global") or {}).get("address") == hex(g_len)]
        [row] = store["references"]
        assert (row["display_name"], row["access"]) == ("use", "load"), (fn, store)
        assert row["at"] in use_sites, (fn, row, use_sites)
        # The neighbour's reader is a reader of the neighbour only.
        assert not [lf for lf in result["leaves"]
                    if lf["kind"] == "global_memory_continuation"], (fn, result["leaves"])
        assert not [r for lf in result["leaves"] for r in (lf.get("references") or [])
                    if r.get("display_name") == "log_banner"], (fn, result["leaves"])


@pytest.mark.parametrize("opt", ["-O0", "-O2"], ids=["O0", "O2"])
def test_forward_claims_around_globals_fields_and_gated_lengths(tmp_path, shared_bn, opt):
    """P2-11 / P2-08 / P2-10 on `taint_corpus/fwd_global_claims.c`.

    Phase-1 integration answered all-clear for `exit_case` (a source fill of
    a global, read by a function never walked), for `alias_case` (the reader
    copies through a pointer global initialised to the object) and for both
    receive lengths (the gated `recv_overflow` sink), and reported a false
    `memcpy` length flow for `field_case` (a store into one field of a
    struct tainting the whole of it). Fixed: each claim gap is a blocking
    frontier naming the reader or the gated call, a full clear before the
    return is an all-clear, the global a callee fills through its pointer
    parameter is keyed in the caller, a heap block reloaded through a global
    is followed to its `malloc` length, and another field is clean."""
    binary = tmp_path / f"fwd_global_claims{opt[-1]}"
    _compile(CORPUS / "fwd_global_claims.c", binary,
             [opt, "-fno-builtin", "-fno-inline", "-Wno-unused-result"])
    target = shared_bn.load(binary, copy=False)

    def fwd(fn, source, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                        "--source", source, *extra)

    def refs(result, kind):
        return {r["display_name"] for lf in result["leaves"] if lf["kind"] == kind
                for r in (lf.get("references") or [])}

    def all_clear(result):
        return result["diagnostics"]["safe_to_report_all_clear"]

    exit_res = fwd("exit_case", "call:read")
    assert refs(exit_res, "coarse_memory_store") == {"use_exit"}, exit_res["leaves"]
    assert all_clear(exit_res) is False

    cleared = fwd("cleared_case", "call:read")
    assert cleared["leaves"] == [] and all_clear(cleared) is True, cleared

    alias = fwd("alias_case", "call:read")
    assert "use_alias" in refs(alias, "global_memory_continuation"), alias["leaves"]
    assert all_clear(alias) is False

    named = fwd("param_case", "param:0")
    assert "use_name" in refs(named, "global_memory_continuation"), named["leaves"]
    assert not [lf for lf in named["leaves"] if lf["kind"] == "coarse_memory_store"], \
        named["leaves"]

    heap = fwd("heap_case", "param:0")
    assert [(s["sink"]["callee"], s["sink"]["class"]) for s in heap["reached_sinks"]] \
        == [("malloc", "alloc_size")], heap["reached_sinks"]

    field = fwd("field_case", "param:0")
    assert field["reached_sinks"] == [], field["reached_sinks"]
    assert all_clear(field) is True, field["leaves"]

    for fn in ("recv_case", "recv_loop"):
        gated = fwd(fn, "param:2")
        assert gated["reached_sinks"] == [], (fn, gated["reached_sinks"])
        [leaf] = [lf for lf in gated["leaves"] if lf["kind"] == "gated_sink_reached"]
        assert (leaf["callee"], leaf["class"], leaf["gate"], leaf["tainted_args"]) \
            == ("read", "overflow_len", "recv_overflow", [2]), leaf
        assert all_clear(gated) is False, fn
        armed = fwd(fn, "param:2", "--sink-class", "recv_overflow")
        assert {s["sink"]["class"] for s in armed["reached_sinks"]} == {"overflow_len"}, \
            (fn, armed["reached_sinks"])
        assert not [lf for lf in armed["leaves"] if lf["kind"] == "gated_sink_reached"], fn



def test_global_continuation_follows_a_thunk_to_its_consumer(tmp_path, shared_bn):
    """Round-3 review F1: a veneer is not a body, but it is not a dead end.

    `transfer` copies the source bytes into a global and then calls a
    one-instruction tail-call veneer with no arguments. The veneer jumps to a
    consumer that hands the identical global to `system`. The reachability
    question must be asked of the RESOLVED target -- direct and multi-hop --
    the same way the descent resolves thunks, not dropped because the call's
    immediate target has no body."""
    src = tmp_path / "thunkcont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_buffer[64];
__attribute__((noinline)) int consumer(void) { return system(global_buffer); }
__asm__(".globl veneer\n.type veneer,@function\nveneer:\n jmp consumer\n.size veneer,.-veneer\n");
__asm__(".globl veneer2\n.type veneer2,@function\nveneer2:\n jmp veneer\n.size veneer2,.-veneer2\n");
extern int veneer(void);
extern int veneer2(void);
__attribute__((noinline)) int transfer(const char *src) {
    memcpy(global_buffer, src, 16);
    return veneer();
}
__attribute__((noinline)) int transfer2(const char *src) {
    memcpy(global_buffer, src, 16);
    return veneer2();
}
int main(int argc, char **argv) {
    return argc > 1 ? transfer(argv[1]) + transfer2(argv[1]) : 0;
}
""")
    binary = tmp_path / "thunkcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    consumer_at = _resolve_addr(shared_bn, "consumer")

    for fn in ("transfer", "transfer2"):          # one hop, then two
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        [leaf] = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)
        # the RESOLVED body is named, not the veneer, and it is where the
        # global is read
        assert leaf["callee"]["address"] == consumer_at, (fn, leaf)
        assert [r["display_name"] for r in leaf["references"]] == ["consumer"], leaf
        assert leaf["references"][0]["access"] == "call_arg", leaf

        listing = _bn_json(shared_bn, "-t", target, "disasm", "consumer")["text"]
        assert f"{int(leaf['global']['address'], 16):x}" in listing.lower(), leaf


def test_global_continuation_write_only_reference_is_not_a_consumer(tmp_path, shared_bn):
    """Round-3 review F3: `g[0] = 0` reads nothing.

    Two functions copy into the same global; one then calls a helper whose
    only reference to it is a store, the other a helper that hands it to
    `system`. Only the second is a continuation -- a reference is not a
    consumer unless it can READ the bytes. The first still returns with the
    copied bytes live in the global, so it ends in the P2-11.1 exit
    `coarse_memory_store` citing `reader` -- never `write_only`."""
    src = tmp_path / "writeonlycont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char live_bytes[64];
__attribute__((noinline)) void write_only(void) { live_bytes[0] = 0; }
__attribute__((noinline)) void reader(void) { system(live_bytes); }
__attribute__((noinline)) int writer_case(const char *src) {
    memcpy(live_bytes, src, 16); write_only(); return 0;
}
__attribute__((noinline)) int reader_case(const char *src) {
    memcpy(live_bytes, src, 16); reader(); return 0;
}
int main(int argc, char **argv) {
    return argc > 1 ? writer_case(argv[1]) + reader_case(argv[1]) : 0;
}
""")
    binary = tmp_path / "writeonlycont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    quiet = forward("writer_case")
    assert _gmc(quiet) == [], quiet["leaves"]
    [store] = _exit_stores(quiet)
    assert store["global"]["address"] == _data_symbol_addr(binary, "live_bytes"), store
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert quiet["diagnostics"]["safe_to_report_all_clear"] is False, quiet["diagnostics"]

    # the store really is the only reference in that helper
    listing = _bn_json(shared_bn, "-t", target, "disasm", "write_only")["text"]
    assert "call" not in listing.lower(), listing

    loud = forward("reader_case")
    [leaf] = [lf for lf in loud["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["references"][0]["display_name"] == "reader", leaf
    assert loud["diagnostics"]["safe_to_report_all_clear"] is False, loud["diagnostics"]


def test_global_continuation_discloses_an_exhausted_reachability_budget(tmp_path, shared_bn):
    """Round-3 review F2: the function budget must never read as an all-clear.

    `fanout` has more direct callees than the reachability walk will enqueue,
    and the consumer of the tainted global is the last of them. The walk
    cannot reach it, so the answer is UNKNOWN: a blocking frontier carrying
    the bound, how much it examined and which references it never reached,
    plus a truncated run with the matching cause -- not silence."""
    width = 513
    body = [
        "#include <stdlib.h>", "#include <string.h>",
        "char global_buffer[64];",
        "__attribute__((noinline)) int consumer(void) { return system(global_buffer); }",
    ]
    # Void callees: an int-returning one makes the one scan of `fanout` follow
    # every return value as a possible pointer -- a whole pointer-scan budget
    # spent on a shape this test is not about, which the request's sweep
    # allowance (P1-01) then reports ahead of the function budget.
    body += [f"__attribute__((noinline)) void q{i}(void) {{ }}" for i in range(width - 1)]
    body += ["__attribute__((noinline)) int fanout(int v) {"]
    body += [f"    q{i}();" for i in range(width - 1)]
    body += ["    return consumer() + v;", "}"]
    body += [
        "__attribute__((noinline)) int transfer(const char *src) {",
        "    memcpy(global_buffer, src, 16); return fanout(1);", "}",
        "int main(int argc, char **argv) { return argc > 1 ? transfer(argv[1]) : 0; }",
    ]
    src = tmp_path / "fanoutcont.c"
    src.write_text("\n".join(body) + "\n")
    binary = tmp_path / "fanoutcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "transfer", "--source", "param:0")
    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    reach = leaf["reachability"]
    # No reader is NAMED -- a body whose own scan ran out of nodes may still
    # be cited as `unclassified`, which claims nothing about what it reads.
    assert all(r["access"] not in ("load", "call_arg")
               for r in leaf["references"]), leaf
    assert reach["status"] == "truncated" and reach["cause"] == "function_budget", leaf
    assert reach["functions_examined"] == reach["function_budget"], leaf
    assert reach["unexamined_references_total"] >= 1, leaf
    assert _resolve_addr(shared_bn, "consumer") in reach["unexamined_references"], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]
    assert result["stats"]["truncated"] is True, result["stats"]
    assert "global_reference_budget" in result["stats"]["truncation_cause"], result["stats"]
    assert "TRUNCATED" in result["diagnostics"]["next_action"], result["diagnostics"]

    # text says the same thing, and every command it prints runs or is marked
    rendered = shared_bn.run("-t", target, "taint", "forward", "-f", "transfer",
                             "--source", "param:0", "--verbose", timeout=120.0)
    assert rendered.returncode == 0, rendered.stderr
    assert "reachability truncated" in rendered.stdout, rendered.stdout
    assert "function_budget" in rendered.stdout, rendered.stdout
    assert sum(_run_or_check_template(shared_bn, _forward_prose(result))) >= 1


def _data_symbol_addr(binary: Path, name: str) -> str:
    """A data symbol's address from the binary's own symbol table."""
    res = subprocess.run(["nm", str(binary)], capture_output=True, text=True, timeout=60.0)
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return hex(int(parts[0], 16))
    raise AssertionError(f"{name} not in the symbol table of {binary.name}")


def _wide_globals_binary(tmp_path, count: int = 600) -> Path:
    """`main` copies argv into `g_name`, calls `reset_all()` and then hands
    argv to `system`. `reset_all` zeroes *count* neighbouring globals laid out
    just before `g_name` and then calls `show_name`, the one reader of it.

    The neighbours are assembler symbols with no size, so nothing DECLARES
    where any of them ends: the search for an object enclosing `g_name` runs
    out of hops, and whether `reset_all` touches that storage is asked of
    every global it names. A declared (C) neighbour would end the search at
    once and never reach that question."""
    asm = ["  .bss", "  .balign 8"]
    for i in range(count):
        asm += [f"  .globl g_slot{i}", f"g_slot{i}: .zero 8"]
    asm += ["  .text"]
    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>",
             "__asm__(" + " ".join(f'"{a}\\n"' for a in asm) + ");"]
    lines += [f"extern long g_slot{i};" for i in range(count)]
    lines += ["char g_name[64];",
              "__attribute__((noinline)) void show_name(void) { puts(g_name); }",
              "__attribute__((noinline)) void reset_all(void) {"]
    lines += [f"    g_slot{i} = 0;" for i in reversed(range(count))]
    lines += ["    show_name();", "}",
              "int main(int argc, char **argv) {", "    if (argc < 2) return 1;",
              "    strncpy(g_name, argv[1], sizeof g_name - 1);", "    reset_all();",
              "    system(argv[1]);", "    return 0;", "}"]
    src = tmp_path / "wideglobals.c"
    src.write_text("\n".join(lines) + "\n")
    binary = tmp_path / "wideglobals"
    _compile(src, binary, ["-fno-builtin"])
    return binary


@pytest.mark.parametrize("depth", ["8", "0"])
def test_a_callee_naming_hundreds_of_globals_does_not_unwind_the_walk(tmp_path, shared_bn,
                                                                      depth):
    """P1-02: asking whether `reset_all` touches the storage around the
    tainted `g_name` re-entered the neighbouring-object scan once per global
    it names; the walk unwound on Python's recursion limit and the
    `system(argv)` after the call was never reported. Now the walk finishes:
    the sink is kept and the call carries the continuation -- the reader
    `show_name` one call down, or, at `--max-depth 0`, the depth bound that
    kept the search from reaching it."""
    binary = _wide_globals_binary(tmp_path)
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "param:1", "--max-depth", depth, timeout=300.0)
    assert "system" in {f["sink"]["callee"] for f in result["reached_sinks"]}, result
    assert "recursion" not in result["stats"]["truncation_cause"], result["stats"]
    assert not [lf for lf in result["leaves"] if lf["kind"] == "analysis_truncated"]
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["global"]["address"] == _data_symbol_addr(binary, "g_name"), leaf
    assert leaf["callee"]["display_name"] == "reset_all", leaf
    if depth == "8":
        assert [r["display_name"] for r in leaf["references"]] == ["show_name"], leaf
    else:
        assert "depth_budget" in leaf["reachability"]["causes"], leaf


def test_a_float_infinity_in_a_consumer_does_not_crash_the_walk(tmp_path, shared_bn):
    """P1-03: `g_limit = INFINITY` lifts to an MLIL float constant `inf`;
    reading it as an address raised OverflowError out of the whole op."""
    src = tmp_path / "floatinf.c"
    src.write_text(r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>
char g_path[64];
double g_limit;
__attribute__((noinline)) void apply_limits(void) {
    g_limit = INFINITY;
    if (g_limit > 1.0) system(g_path);
}
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    strncpy(g_path, argv[1], sizeof g_path - 1);
    apply_limits();
    return 0;
}
""")
    binary = tmp_path / "floatinf"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "param:1")
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["global"]["address"] == _data_symbol_addr(binary, "g_path"), leaf
    assert [r["display_name"] for r in leaf["references"]] == ["apply_limits"], leaf


def test_a_timed_out_taint_request_is_cancelled_and_releases_the_bridge(
        tmp_path, shared_bn, monkeypatch):
    """P1-07: a client that times out sends `cancel_request`; the walk polls it
    at its next function visit, stops with a `cancelled` outcome the client
    reports in its error envelope (`status: "cancelled"`, the bridge's
    `observed`), and the bridge is free for the next request instead of
    computing for minutes."""
    count = 2000          # one descended helper each: seconds of walking
    lines = ["#include <string.h>"]
    lines += [f"__attribute__((noinline)) int h{i}(const char *s) "
              f"{{ return (int)strlen(s) + {i}; }}" for i in range(count)]
    lines += ["int main(int argc, char **argv) {", "    int t = 0;",
              "    if (argc < 2) return 1;"]
    lines += [f"    t += h{i}(argv[1]);" for i in range(count)]
    lines += ["    return t;", "}"]
    src = tmp_path / "longwalk.c"
    src.write_text("\n".join(lines) + "\n")
    binary = tmp_path / "longwalk"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "1")
    timed_out = shared_bn.run("-t", target, "taint", "forward", "-f", "main",
                              "--source", "param:1", "--format", "json", timeout=120.0)
    monkeypatch.delenv("BN_REQUEST_TIMEOUT")
    assert timed_out.returncode == 2, (timed_out.stdout, timed_out.stderr)
    envelope = json.loads(timed_out.stdout)
    assert envelope["ok"] is False and envelope["status"] == "cancelled", envelope
    observed = envelope["observed"]
    assert (observed["timed_out"], observed["cancel_sent"], observed["polls_cancel"],
            observed["cancel_outcome"], observed["cancelled"]) == (
        True, True, True, "cancelled", True), observed
    assert observed["functions_visited"] >= 1, observed
    listed = shared_bn.run("target", "list", "--format", "json", timeout=5.0)
    assert listed.returncode == 0, listed.stderr


def test_global_continuation_phi_selected_pointer_is_a_consumer(tmp_path, shared_bn):
    """Round-2 review R1: `p = cond ? g : other; system(p)` is a reader.

    The consumer reaches the tainted global through a branch merge, which no
    single-reader copy chain resolves. Concluding "it only writes" from that
    restores the false all-clear this frontier exists to withhold."""
    src = tmp_path / "condcont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char fallback_bytes[64] = "/bin/true";
__attribute__((noinline)) int conditional_reader(int flag) {
    const char *p = flag ? global_bytes : fallback_bytes;
    return system(p);
}
__attribute__((noinline)) int transfer(const char *src) {
    memcpy(global_bytes, src, 16);
    return conditional_reader(1);
}
int main(int argc, char **argv) { return argc > 1 ? transfer(argv[1]) : 0; }
""")
    binary = tmp_path / "condcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "transfer", "--source", "param:0")

    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]
    assert [r["display_name"] for r in leaf["references"]] == ["conditional_reader"], leaf
    assert leaf["references"][0]["access"] != "write_only", leaf

    # both sides of the merge reference the identical global base
    listing = _bn_json(shared_bn, "-t", target, "disasm", "conditional_reader")["text"]
    assert f"{int(leaf['global']['address'], 16):x}" in listing.lower(), listing


def test_global_continuation_modeled_write_destinations_are_not_readers(tmp_path, shared_bn):
    """Round-2 review R2: a modeled write destination reads nothing.

    `memset(g, 0, 64)` and `memcpy(g, "safe", 5)` hand the global to a call,
    but their models say that argument is written and never read, so neither
    helper is a consumer. The unrelated-global and scalar-store controls are
    not continuations either, and a helper that copies the global OUT is
    still a reader. Only the full `memset` clears every tainted byte before
    the return; the others leave them live, so they end in the P2-11.1 exit
    `coarse_memory_store` citing that real reader -- never their helper."""
    src = tmp_path / "wocalls.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char other_global[64];
int counter;
__attribute__((noinline)) void reset_helper(void) { memset(global_bytes, 0, 64); }
__attribute__((noinline)) void copy_reset_helper(void) { memcpy(global_bytes, "safe", 5); }
__attribute__((noinline)) void scalar_helper(void) { counter = 0; }
__attribute__((noinline)) void other_helper(void) { system(other_global); }
__attribute__((noinline)) void leak_helper(void) { memcpy(other_global, global_bytes, 16); }
__attribute__((noinline)) int reset_case(const char *s) { memcpy(global_bytes, s, 16); reset_helper(); return 0; }
__attribute__((noinline)) int copy_reset_case(const char *s) { memcpy(global_bytes, s, 16); copy_reset_helper(); return 0; }
__attribute__((noinline)) int scalar_case(const char *s) { memcpy(global_bytes, s, 16); scalar_helper(); return 0; }
__attribute__((noinline)) int other_case(const char *s) { memcpy(global_bytes, s, 16); other_helper(); return 0; }
__attribute__((noinline)) int leak_case(const char *s) { memcpy(global_bytes, s, 16); leak_helper(); return 0; }
int main(int argc, char **argv) {
    return argc > 1 ? reset_case(argv[1]) + copy_reset_case(argv[1])
                      + scalar_case(argv[1]) + other_case(argv[1])
                      + leak_case(argv[1]) : 0;
}
""")
    binary = tmp_path / "wocalls"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    for fn in ("reset_case", "copy_reset_case", "scalar_case", "other_case"):
        quiet = forward(fn)
        assert _gmc(quiet) == [], (fn, quiet["leaves"])
        if fn == "reset_case":
            assert quiet["leaves"] == [], (fn, quiet["leaves"])
            assert quiet["diagnostics"]["safe_to_report_all_clear"] is True, (fn, quiet)
            continue
        [store] = _exit_stores(quiet)
        assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["leak_helper"], \
            (fn, store)
        assert quiet["diagnostics"]["safe_to_report_all_clear"] is False, (fn, quiet)

    # the mirror: the same model READS through arg1
    loud = forward("leak_case")
    [leaf] = [lf for lf in loud["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["references"][0]["display_name"] == "leak_helper", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert loud["diagnostics"]["safe_to_report_all_clear"] is False, loud["diagnostics"]


def test_global_continuation_unanalysed_consumer_body_is_disclosed(tmp_path, shared_bn):
    """Round-2 review R3: an in-binary body BN skipped is not a dead end.

    The consumer is past `analysis.limits.maxFunctionSize`, so BN never
    analysed it and its references cannot be read. Skipping the call reported
    zero sinks, zero leaves, a true all-clear and `analysis_incomplete: false`
    on a program whose consumer hands the global to `system`."""
    src = tmp_path / "unreadcont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
__attribute__((noinline)) int huge_reader(void) {
    __asm__ volatile(".rept 70000\n nop\n .endr\n");
    return system(global_bytes);
}
__attribute__((noinline)) int transfer(const char *src) {
    memcpy(global_bytes, src, 16);
    return huge_reader();
}
int main(int argc, char **argv) { return argc > 1 ? transfer(argv[1]) : 0; }
""")
    binary = tmp_path / "unreadcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "transfer", "--source", "param:0")
    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["reachability"]["cause"] == "unread_callee_body", leaf
    assert "force-analysis" in leaf["detail"], leaf
    # Native reference evidence stands behind the boundary: BN records no code
    # reference OUT of a body it never analysed, so the leaf names the
    # reference sites for this global that the walk could not reach, with the
    # function and instruction address of each.
    sites = leaf["reachability"]["unexamined_reference_sites"]
    assert sites, leaf["reachability"]
    assert all({"name", "address", "at"} <= set(site) for site in sites), sites
    for site in sites:
        listing = _bn_json(shared_bn, "-t", target, "disasm", site["address"])["text"]
        assert f"{int(leaf['global']['address'], 16):x}" in listing.lower(), (site, listing)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]
    assert result["stats"]["analysis_incomplete"] is True, result["stats"]
    assert "huge_reader" in result["stats"]["analysis_incomplete_functions"], result["stats"]


def test_global_continuation_follows_derived_pointers_into_the_consumer(tmp_path, shared_bn):
    """Round-3 review F4: `g + i` and `cond ? g : other` still point at `g`.

    A consumer that indexes the global, or selects it on one branch, hands
    the same bytes to `system`. Resolving only a bare constant pointer read
    both as write-only and cleared the run."""
    src = tmp_path / "derivedcont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char other_bytes[64] = "/bin/true";
__attribute__((noinline)) int offset_reader(int i) { return system(global_bytes + i); }
__attribute__((noinline)) int cond_reader(int f) { return system(f ? global_bytes : other_bytes); }
__attribute__((noinline)) int other_reader(int i) { return system(other_bytes + i); }
__attribute__((noinline)) int offset_case(const char *s) { memcpy(global_bytes, s, 16); return offset_reader(4); }
__attribute__((noinline)) int cond_case(const char *s) { memcpy(global_bytes, s, 16); return cond_reader(1); }
__attribute__((noinline)) int other_case(const char *s) { memcpy(global_bytes, s, 16); return other_reader(4); }
int main(int argc, char **argv) {
    return argc > 1 ? offset_case(argv[1]) + cond_case(argv[1]) + other_case(argv[1]) : 0;
}
""")
    binary = tmp_path / "derivedcont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    for fn, reader in (("offset_case", "offset_reader"), ("cond_case", "cond_reader")):
        result = forward(fn)
        [leaf] = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert [r["display_name"] for r in leaf["references"]] == [reader], (fn, leaf)
        assert leaf["references"][0]["access"] != "write_only", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)

    # the control: the same shapes over a DIFFERENT global say nothing
    control = forward("other_case")
    assert [lf for lf in control["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], control["leaves"]


def test_global_continuation_unmodeled_callee_is_an_honest_unknown(tmp_path, shared_bn):
    """Round-3 review F5: an unmodeled callee handed the global is unknown.

    Nothing here saw it consume the bytes, so the boundary is disclosed as
    `unknown_call_arg` and the prose says "may read" -- neither a claimed
    read nor a clear."""
    src = tmp_path / "unknowncont.c"
    src.write_text(r"""
#include <string.h>
char global_bytes[64];
__attribute__((noinline)) void app_blob(char *p) { p[1] = p[0]; }
__attribute__((noinline)) void handoff(void) { app_blob(global_bytes); }
__attribute__((noinline)) int transfer(const char *s) {
    memcpy(global_bytes, s, 16); handoff(); return 0;
}
int main(int argc, char **argv) { return argc > 1 ? transfer(argv[1]) : 0; }
""")
    binary = tmp_path / "unknowncont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "transfer", "--source", "param:0")

    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["references"][0]["access"] in ("unknown_call_arg", "address_taken"), leaf
    assert "may read" in leaf["detail"], leaf["detail"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]


def test_global_continuation_partial_overlay_is_not_a_no_read_guarantee(tmp_path, shared_bn):
    """Round-3 review F7: a model is additive flow evidence, not a promise.

    `copy_and_use(dst, src)` hands `dst` to `system` BEFORE the copy an
    overlay describes. Reading `{"propagates": [{"from": "*arg:1", "to":
    "*arg:0"}]}` as "and it reads nothing else" cleared the run; the body is
    the authority, so the same query answers the same way with and without
    the overlay."""
    src = tmp_path / "additivecont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
__attribute__((noinline)) void copy_and_use(char *dst, const char *src) {
    system(dst);
    memcpy(dst, src, 16);
}
__attribute__((noinline)) void consumer(void) {
    copy_and_use(global_bytes, "safe constant bytes");
}
__attribute__((noinline)) int transfer(const char *s) {
    memcpy(global_bytes, s, 16); consumer(); return 0;
}
int main(int argc, char **argv) { return argc > 1 ? transfer(argv[1]) : 0; }
""")
    binary = tmp_path / "additivecont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps(
        {"copy_and_use": {"propagates": [{"from": "*arg:1", "to": "*arg:0"}]}}))

    for extra in ([], ["--models", str(overlay)]):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", "transfer", "--source", "param:0", *extra)
        [leaf] = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (extra, result)
        assert [r["display_name"] for r in leaf["references"]] == ["consumer"], leaf
        assert leaf["references"][0]["access"] == "unknown_call_arg", leaf


def test_global_continuation_phi_selected_write_destination_is_silent(tmp_path, shared_bn):
    """Round-3 review F5: following the pointer to its terminal uses.

    A helper whose entire effect is `p = flag ? g : other; memset(p, 0, 64)`
    reads nothing: the `lea` that materialises the destination is not an
    access. The reading and read-write twins still block. A reset through a
    phi-selected pointer does not prove every tainted byte cleared, so the
    silent helpers' cases end in the P2-11.1 exit `coarse_memory_store`,
    which cites the real reader `phi_read` -- never the write-only helper."""
    src = tmp_path / "phiwocont.c"
    src.write_text(r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char other_global[64];
__attribute__((noinline)) void phi_reset(int f) { char *p = f ? global_bytes : other_global; memset(p, 0, 64); }
__attribute__((noinline)) void phi_copy_reset(int f) { char *p = f ? global_bytes : other_global; memcpy(p, "safe", 5); }
__attribute__((noinline)) void phi_read(int f) { char *p = f ? global_bytes : other_global; system(p); }
__attribute__((noinline)) void phi_rw(int f) { char *p = f ? global_bytes : other_global; memcpy(p, p, 8); }
__attribute__((noinline)) int phi_reset_case(const char *s) { memcpy(global_bytes, s, 16); phi_reset(1); return 0; }
__attribute__((noinline)) int phi_copy_reset_case(const char *s) { memcpy(global_bytes, s, 16); phi_copy_reset(1); return 0; }
__attribute__((noinline)) int phi_read_case(const char *s) { memcpy(global_bytes, s, 16); phi_read(1); return 0; }
__attribute__((noinline)) int phi_rw_case(const char *s) { memcpy(global_bytes, s, 16); phi_rw(1); return 0; }
int main(int argc, char **argv) {
    return argc > 1 ? phi_reset_case(argv[1]) + phi_copy_reset_case(argv[1])
                      + phi_read_case(argv[1]) + phi_rw_case(argv[1]) : 0;
}
""")
    binary = tmp_path / "phiwocont"
    _compile(src, binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    for fn in ("phi_reset_case", "phi_copy_reset_case"):
        quiet = forward(fn)
        assert _gmc(quiet) == [], (fn, quiet["leaves"])
        [store] = _exit_stores(quiet)
        assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["phi_read"], (fn, store)
        assert quiet["diagnostics"]["safe_to_report_all_clear"] is False, (fn, quiet)

    for fn, reader in (("phi_read_case", "phi_read"), ("phi_rw_case", "phi_rw")):
        loud = forward(fn)
        [leaf] = [lf for lf in loud["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert leaf["references"][0]["display_name"] == reader, (fn, leaf)
        assert leaf["references"][0]["access"] == "call_arg", (fn, leaf)
        assert loud["diagnostics"]["safe_to_report_all_clear"] is False, (fn, loud)


_BOUNDARY_SRC = r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char neighbour_bytes[64] = "/bin/true";
__attribute__((noinline)) int offset_consumer(void) { return system(global_bytes + 4); }
__attribute__((noinline)) int base_consumer(void) { return system(global_bytes); }
__attribute__((noinline)) int neighbour_consumer(void) { return system(neighbour_bytes + 8); }
__attribute__((noinline)) int huge_reader(void) {
    __asm__ volatile(".rept 70000\n nop\n .endr\n");
    return system(global_bytes);
}
__attribute__((noinline)) int wrapper(void) { return huge_reader() + 1; }
__attribute__((noinline)) int wrapper2(void) { return wrapper() + 1; }
__attribute__((noinline)) int offset_case(const char *s) { memcpy(global_bytes, s, 16); return offset_consumer(); }
__attribute__((noinline)) int base_case(const char *s) { memcpy(global_bytes, s, 16); return base_consumer(); }
__attribute__((noinline)) int neighbour_case(const char *s) { memcpy(global_bytes, s, 16); return neighbour_consumer(); }
__attribute__((noinline)) int unread_case(const char *s) { memcpy(global_bytes, s, 16); return huge_reader(); }
__attribute__((noinline)) int wrapped_unread_case(const char *s) { memcpy(global_bytes, s, 16); return wrapper(); }
__attribute__((noinline)) int deep_unread_case(const char *s) { memcpy(global_bytes, s, 16); return wrapper2(); }
__attribute__((noinline)) int indirect_case(const char *s, int (*cb)(void)) { memcpy(global_bytes, s, 16); return cb(); }
__attribute__((noinline)) int clean_indirect_case(const char *s, int (*cb)(void)) {
    char local[64]; memcpy(local, s, 16); return cb() + local[0];
}
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return offset_case(argv[1]) + base_case(argv[1]) + neighbour_case(argv[1])
         + unread_case(argv[1]) + wrapped_unread_case(argv[1]) + deep_unread_case(argv[1])
         + indirect_case(argv[1], base_consumer) + clean_indirect_case(argv[1], base_consumer);
}
"""


def _boundary_target(tmp_path, shared_bn):
    """One binary carrying every round-3 boundary shape (F8/F9/F10).

    Built per test rather than per module: `shared_bn` is function-scoped so
    exactly one target is open while a test runs, which is what lets these
    commands omit `--target`.
    """
    src = tmp_path / "boundaries.c"
    src.write_text(_BOUNDARY_SRC)
    binary = tmp_path / "boundaries"
    _compile(src, binary, ["-fno-builtin"])
    return shared_bn.load(binary, copy=False), tmp_path


def _gmc(result):
    return [lf for lf in result["leaves"] if lf["kind"] == "global_memory_continuation"]


def _exit_stores(result, address: str | None = None):
    """P2-11.1 exit frontiers: blocking `coarse_memory_store` leaves for a
    global still tainted at the seeded return (optionally for one global)."""
    return [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"
            and (address is None or (lf.get("global") or {}).get("address") == address)]


def test_global_continuation_interior_reference_blocks_with_its_offset(tmp_path, shared_bn):
    """Round-3 review F8: `system(g + 4)` reads the copied bytes.

    BN records that code reference at the INTERIOR address, so asking only
    about the object's base found no consumer and cleared the run. The whole
    object is asked for now, and the leaf reports the base plus the offset
    the consumer actually named. A DISJOINT neighbouring object is not swept
    in with it: its reader is never cited, and the bytes still live at the
    return end in the P2-11.1 exit store naming a real reader of `g`."""
    target, _tmp = _boundary_target(tmp_path, shared_bn)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    offset = forward("offset_case")
    [leaf] = _gmc(offset)
    assert leaf["references"][0]["display_name"] == "offset_consumer", leaf
    assert leaf["references"][0]["offset"] == 4, leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert offset["diagnostics"]["safe_to_report_all_clear"] is False, offset

    base = forward("base_case")
    [leaf] = _gmc(base)
    assert leaf["references"][0]["offset"] == 0, leaf

    # the disjoint control: the consumer reads the NEXT object
    neighbour = forward("neighbour_case")
    assert _gmc(neighbour) == [], neighbour["leaves"]
    [store] = _exit_stores(neighbour)
    assert store["global"]["address"] == \
        _data_symbol_addr(_tmp / "boundaries", "global_bytes"), store
    assert [(r["display_name"], r["offset"]) for r in store["references"]] \
        == [("offset_consumer", 4)], store
    assert neighbour["diagnostics"]["safe_to_report_all_clear"] is False, neighbour


def test_global_continuation_unread_consumer_behind_a_wrapper(tmp_path, shared_bn):
    """Round-3 review F9: the hole is the unread body, not its position.

    `huge_reader` is past `analysis.limits.maxFunctionSize`; calling it
    directly already disclosed the boundary, but an ordinary wrapper -- at
    one hop or two -- walked straight through it to a clean all-clear."""
    target, _tmp = _boundary_target(tmp_path, shared_bn)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    for fn in ("unread_case", "wrapped_unread_case", "deep_unread_case"):
        result = forward(fn)
        [leaf] = _gmc(result)
        assert leaf["reachability"]["cause"] == "unread_callee_body", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)
        assert result["stats"]["analysis_incomplete"] is True, (fn, result["stats"])
        assert "huge_reader" in result["stats"]["analysis_incomplete_functions"], \
            (fn, result["stats"])
        if fn != "unread_case":
            assert [b["name"] for b in leaf["reachability"]["unread_bodies"]] \
                == ["huge_reader"], (fn, leaf)
            assert "huge_reader" in leaf["detail"], (fn, leaf["detail"])


def test_global_continuation_unresolved_callback_is_a_boundary(tmp_path, shared_bn):
    """Round-3 review F10: an unresolved zero-argument callback.

    No argument carries taint, so the existing indirect frontier says
    nothing -- but the tainted global is live and this binary has code that
    reads it, and nothing proves the unknown target is not that code.
    Pinning the site with `--resolve-map` replaces the boundary with the real
    target's answer; a run with no tainted global stays silent."""
    target, tmp = _boundary_target(tmp_path, shared_bn)

    def forward(fn, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0", *extra)

    unresolved = forward("indirect_case")
    [leaf] = _gmc(unresolved)
    assert leaf["callee"] is None, leaf
    assert leaf["reachability"]["cause"] == "unresolved_call_target", leaf
    assert leaf["reachability"]["unexamined_reference_sites"], leaf["reachability"]
    assert unresolved["diagnostics"]["safe_to_report_all_clear"] is False, unresolved

    # no tainted global, same unresolved callback: nothing to disclose
    clean = forward("clean_indirect_case")
    assert _gmc(clean) == [], clean["leaves"]

    # --resolve-map refines the boundary into the real target's answer
    rmap = tmp / "rmap.json"
    rmap.write_text(json.dumps({
        _indirect_call_addr(shared_bn, "indirect_case", "boundaries"):
            [_resolve_addr(shared_bn, "base_consumer")]}))
    pinned = forward("indirect_case", "--resolve-map", str(rmap))
    [leaf] = _gmc(pinned)
    assert leaf["callee"]["display_name"] == "base_consumer", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert leaf["reachability"]["cause"] is None, leaf


_ACCESS_SRC = r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char neighbour_bytes[64] = "/bin/true";
__attribute__((noinline)) int alias_consumer(void) { char *p = memcpy(global_bytes, "x", 1); return system(p + 1); }
__attribute__((noinline)) int concat_consumer(void) { strcat(global_bytes, "suffix"); return 0; }
__attribute__((noinline)) int reset_consumer(void) { memset(global_bytes, 0, 64); return 0; }
__attribute__((noinline)) int consumer(void) { return system(global_bytes); }
__attribute__((noinline)) int dispatch(int (*cb)(void)) { return cb(); }
__attribute__((noinline)) int offset_consumer(void) { return system(global_bytes + 4); }
__attribute__((noinline)) int neighbour_consumer(void) { return system(neighbour_bytes + 8); }
__attribute__((noinline)) int alias_case(const char *s) { memcpy(global_bytes, s, 16); return alias_consumer(); }
__attribute__((noinline)) int concat_case(const char *s) { memcpy(global_bytes, s, 16); return concat_consumer(); }
__attribute__((noinline)) int reset_case(const char *s) { memcpy(global_bytes, s, 16); return reset_consumer(); }
__attribute__((noinline)) int wrapped_indirect_case(const char *s, int (*cb)(void)) { memcpy(global_bytes, s, 16); return dispatch(cb); }
__attribute__((noinline)) int clean_wrapped_indirect_case(const char *s, int (*cb)(void)) {
    char local[64]; memcpy(local, s, 16); return dispatch(cb) + local[0];
}
__attribute__((noinline)) int offset_case(const char *s) { memcpy(global_bytes, s, 16); return offset_consumer(); }
__attribute__((noinline)) int neighbour_case(const char *s) { memcpy(global_bytes, s, 16); return neighbour_consumer(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return alias_case(argv[1]) + concat_case(argv[1]) + reset_case(argv[1])
         + wrapped_indirect_case(argv[1], consumer)
         + clean_wrapped_indirect_case(argv[1], consumer)
         + offset_case(argv[1]) + neighbour_case(argv[1]);
}
"""


def _access_target(tmp_path, shared_bn, *, debug: bool):
    """The access-shape fixture, built with or without debug types."""
    name = "accesses" if debug else "accesses_nodbg"
    src = tmp_path / f"{name}.c"
    src.write_text(_ACCESS_SRC)
    binary = tmp_path / name
    cc = ["-fno-builtin"] if debug else ["-fno-builtin", "-g0"]
    _compile(src, binary, cc)
    return shared_bn.load(binary, copy=False)


@pytest.mark.parametrize("debug", [True, False], ids=["typed", "stripped"])
def test_global_continuation_interior_reader_without_debug_types(tmp_path, shared_bn,
                                                                 debug):
    """Round-3 review I1: a stripped global still has an extent.

    With no debug type BN reports the object as `void`, width 0, and invents
    a bare data variable at the offset the consumer references -- so taking
    the type's word for it made the extent one byte, the reference sweep
    missed the consumer and the interior pointer folded to a different
    "object". The layout answers instead: the object runs to the next data
    variable that carries a SYMBOL, which is where the neighbour begins. So
    the neighbour's reader is never cited for it -- the copied bytes still
    live at the return end in the P2-11.1 exit store, citing a reader of the
    object itself."""
    target = _access_target(tmp_path, shared_bn, debug=debug)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    result = forward("offset_case")
    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["references"][0]["display_name"] == "offset_consumer", leaf
    assert leaf["references"][0]["offset"] == 4, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    neighbour = forward("neighbour_case")
    assert [lf for lf in neighbour["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], neighbour["leaves"]
    name = "accesses" if debug else "accesses_nodbg"
    [store] = _exit_stores(neighbour)
    assert store["global"]["address"] == \
        _data_symbol_addr(tmp_path / name, "global_bytes"), store
    assert [r["display_name"] for r in store["references"]] == ["alias_consumer"], store
    assert neighbour["diagnostics"]["safe_to_report_all_clear"] is False, neighbour


def test_global_continuation_callback_behind_a_wrapper(tmp_path, shared_bn):
    """Round-3 review I2: `Function.callees` lists resolved edges only.

    `dispatch(cb)` is a leaf to a call-graph walk, so a callback reached
    through it answered "nothing reads the global". Each indirect site in a
    walked BODY is re-resolved the way the forward walk resolves one; what
    still does not resolve is named."""
    target = _access_target(tmp_path, shared_bn, debug=True)

    def forward(fn, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0", *extra)

    result = forward("wrapped_indirect_case")
    [leaf] = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["reachability"]["cause"] == "unresolved_call_target", leaf
    assert leaf["reachability"]["unresolved_calls"], leaf["reachability"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # the control: no tainted global, same wrapper and callback
    clean = forward("clean_wrapped_indirect_case")
    assert [lf for lf in clean["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], clean["leaves"]

    # --resolve-map pins the dispatcher's own callsite: the walk follows it
    rmap = tmp_path / "wrapped_rmap.json"
    rmap.write_text(json.dumps({
        _indirect_call_addr(shared_bn, "dispatch", "accesses"):
            [_resolve_addr(shared_bn, "consumer")]}))
    pinned = forward("wrapped_indirect_case", "--resolve-map", str(rmap))
    [leaf] = [lf for lf in pinned["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["reachability"]["cause"] is None, leaf
    assert [r["display_name"] for r in leaf["references"]] == ["consumer"], leaf


def test_global_continuation_returned_alias_and_append_destinations(tmp_path, shared_bn):
    """Round-3 review I3/I4.

    I3: `p = memcpy(g, "x", 1); system(p + 1)` -- the model says `arg:0 ->
    ret`, so the result IS the destination and the consumer reads the bytes
    the one-byte copy did not overwrite. I4: `strcat(g, "suffix")` walks g's
    existing bytes to find the terminator, so an append destination is never
    write-only. The genuine `memset` reset stays silent."""
    target = _access_target(tmp_path, shared_bn, debug=True)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    for fn, consumer in (("alias_case", "alias_consumer"),
                         ("concat_case", "concat_consumer")):
        result = forward(fn)
        [leaf] = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert [r["display_name"] for r in leaf["references"]] == [consumer], (fn, leaf)
        assert leaf["references"][0]["access"] == "call_arg", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)

    reset = forward("reset_case")
    assert [lf for lf in reset["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], reset["leaves"]
    assert reset["diagnostics"]["safe_to_report_all_clear"] is True, reset


_INITIALIZED_SRC = r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64] = "seed";
char neighbour_bytes[64] = "/bin/true";
extern char *__strcat_chk(char *, const char *, unsigned long);
extern char *__strncat_chk(char *, const char *, unsigned long, unsigned long);
__attribute__((noinline)) int interior_reader(void) { return system(global_bytes + 8); }
__attribute__((noinline)) int neighbour_reader(void) { return system(neighbour_bytes + 8); }
__attribute__((noinline)) int fort_append_consumer(void) { __strcat_chk(global_bytes, "suffix", 64); return 0; }
__attribute__((noinline)) int fort_nappend_consumer(void) { __strncat_chk(global_bytes, "suffix", 6, 64); return 0; }
__attribute__((noinline)) int append_consumer(void) { strcat(global_bytes, "suffix"); return 0; }
__attribute__((noinline)) int reset_consumer(void) { memset(global_bytes, 0, 64); return 0; }
__attribute__((noinline)) int entry(const char *s) { memcpy(global_bytes, s, 16); return interior_reader(); }
__attribute__((noinline)) int neighbour_case(const char *s) { memcpy(global_bytes, s, 16); return neighbour_reader(); }
__attribute__((noinline)) int fort_append_case(const char *s) { memcpy(global_bytes, s, 16); return fort_append_consumer(); }
__attribute__((noinline)) int fort_nappend_case(const char *s) { memcpy(global_bytes, s, 16); return fort_nappend_consumer(); }
__attribute__((noinline)) int append_case(const char *s) { memcpy(global_bytes, s, 16); return append_consumer(); }
__attribute__((noinline)) int reset_case(const char *s) { memcpy(global_bytes, s, 16); return reset_consumer(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return entry(argv[1]) + neighbour_case(argv[1]) + fort_append_case(argv[1])
         + fort_nappend_case(argv[1]) + append_case(argv[1]) + reset_case(argv[1]);
}
"""


@pytest.mark.parametrize("debug", [True, False], ids=["typed", "stripped"])
def test_global_continuation_initialised_global_and_fortified_appends(tmp_path,
                                                                     shared_bn, debug):
    """Round-3 review F11 and F14.

    F11: `char g[64] = "seed"` with no debug info comes back `char[0x5]` --
    BN sized it from the bytes it could see. Taking that as the object's
    extent stopped five bytes in, so `system(g + 8)` was never discovered.
    The recovered type is a LOWER bound now; the layout (to the next data
    variable that carries a symbol) is the wider answer, and the symbolled
    neighbour still bounds it: its reader is never cited, and the bytes
    still live at the return end in the P2-11.1 exit store, which names the
    object's own interior reader.

    F14: a fortified append stops somewhere else than a bare one, but it
    still walks the destination's existing bytes to find the terminator, so
    `__strcat_chk`/`__strncat_chk` destinations are not write-only. The
    genuine `memset` reset stays silent."""
    name = "initialized" if debug else "initialized_nodbg"
    src = tmp_path / f"{name}.c"
    src.write_text(_INITIALIZED_SRC)
    binary = tmp_path / name
    _compile(src, binary, ["-fno-builtin"] + ([] if debug else ["-g0"]))
    target = shared_bn.load(binary, copy=False)

    def forward(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0")

    # F11: the consumer reads past the width BN inferred from the literal
    interior = forward("entry")
    [leaf] = [lf for lf in interior["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaf["references"][0]["display_name"] == "interior_reader", leaf
    assert leaf["references"][0]["offset"] == 8, leaf
    assert interior["diagnostics"]["safe_to_report_all_clear"] is False, interior

    # ...and the symbolled neighbour is still a different object
    neighbour = forward("neighbour_case")
    assert [lf for lf in neighbour["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], neighbour["leaves"]
    [store] = _exit_stores(neighbour)
    assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), store
    assert [(r["display_name"], r["offset"]) for r in store["references"]] \
        == [("interior_reader", 8)], store
    assert neighbour["diagnostics"]["safe_to_report_all_clear"] is False, neighbour

    # F14: fortified, bounds-shifted and bare appends all read the dest
    for fn, consumer in (("fort_append_case", "fort_append_consumer"),
                         ("fort_nappend_case", "fort_nappend_consumer"),
                         ("append_case", "append_consumer")):
        result = forward(fn)
        [leaf] = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert [r["display_name"] for r in leaf["references"]] == [consumer], (fn, leaf)
        assert leaf["references"][0]["access"] == "call_arg", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)

    # ...and a genuine reset is still not a reader
    reset = forward("reset_case")
    assert [lf for lf in reset["leaves"]
            if lf["kind"] == "global_memory_continuation"] == [], reset["leaves"]
    assert reset["diagnostics"]["safe_to_report_all_clear"] is True, reset


_REVIEW_SRC = r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[64];
char large_global[131072];
char neighbour_bytes[64] = "/bin/true";
__attribute__((noinline)) int large_reader(void) { return system(large_global + 100000); }
__attribute__((noinline)) int huge_reader(void) {
    __asm__ volatile(".rept 70000\n nop\n .endr\n");
    return system(global_bytes);
}
__asm__(".globl veneer\n.type veneer,@function\nveneer:\n jmp huge_reader\n.size veneer,.-veneer\n");
__asm__(".globl veneer2\n.type veneer2,@function\nveneer2:\n jmp veneer\n.size veneer2,.-veneer2\n");
extern int veneer(void);
extern int veneer2(void);
__attribute__((noinline)) void reset(void) { memset(global_bytes, 0, 64); }
__attribute__((noinline)) char *reset_ret(void) { return memset(global_bytes, 0, 64); }
__attribute__((noinline)) int use_reset_ret(void) { char *p = reset_ret(); return system(p); }
__attribute__((noinline)) int large_case(const char *s) { memcpy(large_global, s, 110000); return large_reader(); }
__attribute__((noinline)) int veneer_case(const char *s) { memcpy(global_bytes, s, 16); return veneer(); }
__attribute__((noinline)) int veneer2_case(const char *s) { memcpy(global_bytes, s, 16); return veneer2(); }
__attribute__((noinline)) int plt_case(const char *s) { memcpy(global_bytes, s, 16); return puts(neighbour_bytes); }
__attribute__((noinline)) int reset_ignore_case(const char *s) { memcpy(global_bytes, s, 16); reset(); return 0; }
__attribute__((noinline)) int reset_ret_ignore_case(const char *s) { memcpy(global_bytes, s, 16); reset_ret(); return 0; }
__attribute__((noinline)) int reset_ret_used_case(const char *s) { memcpy(global_bytes, s, 16); return use_reset_ret(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return large_case(argv[1]) + veneer_case(argv[1]) + veneer2_case(argv[1])
         + plt_case(argv[1]) + reset_ignore_case(argv[1])
         + reset_ret_ignore_case(argv[1]) + reset_ret_used_case(argv[1]);
}
"""


@pytest.mark.parametrize("debug", [True, False], ids=["typed", "stripped"])
def test_global_continuation_large_objects_veneers_and_returned_pointers(tmp_path,
                                                                        shared_bn,
                                                                        debug):
    """Round-3 review F15, F16 and F17 on one binary, typed and stripped.

    F15: a 128 KiB buffer whose reader is at `+100000` -- the sweep used to
    clip every object to 64 KiB and never enumerate it.
    F16: a veneer onto an in-binary body BN measured and skipped keeps the
    unread-consumer boundary and the analysis-incomplete signal; a genuine
    PLT thunk onto an import stays external: no continuation at the call,
    and the bytes still live at the return end in the P2-11.1 exit store,
    citing the `system(global_bytes)` inside `huge_reader`'s own extent.
    F17: returning a pointer is not reading its bytes. A write-only helper
    whose return the caller discards is silent; when the caller CONSUMES the
    returned pointer the boundary is reported at that consumer."""
    name = "review" if debug else "review_nodbg"
    src = tmp_path / f"{name}.c"
    src.write_text(_REVIEW_SRC)
    binary = tmp_path / name
    _compile(src, binary, ["-fno-builtin"] + ([] if debug else ["-g0"]))
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F15
    result, leaves = gmc("large_case")
    [leaf] = leaves
    assert leaf["references"][0]["display_name"] == "large_reader", leaf
    assert leaf["references"][0]["offset"] == 100000, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # F16: one hop and two, then the genuine external thunk control
    for fn in ("veneer_case", "veneer2_case"):
        result, leaves = gmc(fn)
        [leaf] = leaves
        assert leaf["reachability"]["cause"] == "unread_callee_body", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)
        assert result["stats"]["analysis_incomplete"] is True, (fn, result["stats"])
        assert "huge_reader" in result["stats"]["analysis_incomplete_functions"], \
            (fn, result["stats"])
    result, leaves = gmc("plt_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), store
    [row] = store["references"]
    assert (row["access"], row["offset"]) == ("call_arg", 0), store
    # BN carves the analysed tail of the skipped body into its own `sub_`;
    # the read it cites lies inside `huge_reader`'s symbol, not in `puts`.
    sized = {p[3]: (int(p[0], 16), int(p[1], 16)) for p in (
        ln.split() for ln in subprocess.run(
            ["nm", "-S", str(binary)], capture_output=True, text=True,
            timeout=60.0).stdout.splitlines()) if len(p) == 4}
    lo, size = sized["huge_reader"]
    assert lo <= int(row["at"], 16) < lo + size, (store, hex(lo), hex(size))
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # F17: ignored returns are silent, a consumed one blocks at the consumer
    for fn in ("reset_ignore_case", "reset_ret_ignore_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is True, (fn, result)
    result, leaves = gmc("reset_ret_used_case")
    [leaf] = leaves
    assert leaf["references"][0]["display_name"] == "use_reset_ret", leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result


_RELAY_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
char global_bytes[64];
__attribute__((noinline)) char *reset_ret(void) { return memset(global_bytes, 0, 1); }
__attribute__((noinline)) char *relay_ret(void) { return reset_ret(); }
__attribute__((noinline)) char *relay2_ret(void) { return relay_ret(); }
__attribute__((noinline)) char *reset_arg(char *p) { return memset(p, 0, 1); }
__attribute__((noinline)) int use_relay_ret(void) { return system(relay_ret() + 8); }
__attribute__((noinline)) int use_relay2_ret(void) { return system(relay2_ret() + 8); }
__attribute__((noinline)) int use_arg_ret(void) { char *p = reset_arg(global_bytes); return system(p + 8); }
__attribute__((noinline)) int ignore_relay_ret(void) { relay_ret(); return 0; }
__attribute__((noinline)) void refill(void) { read(0, global_bytes, 64); }
__attribute__((noinline)) void refill_then_read(void) { read(0, global_bytes, 64); system(global_bytes); }
__attribute__((noinline)) void reset_only(void) { memset(global_bytes, 0, 64); }
__attribute__((noinline)) int relay_ret_case(const char *s) { memcpy(global_bytes, s, 64); return use_relay_ret(); }
__attribute__((noinline)) int relay2_ret_case(const char *s) { memcpy(global_bytes, s, 64); return use_relay2_ret(); }
__attribute__((noinline)) int arg_ret_case(const char *s) { memcpy(global_bytes, s, 64); return use_arg_ret(); }
__attribute__((noinline)) int ignore_case(const char *s) { memcpy(global_bytes, s, 64); return ignore_relay_ret(); }
__attribute__((noinline)) int refill_case(const char *s) { memcpy(global_bytes, s, 64); refill(); return 0; }
__attribute__((noinline)) int refill_read_case(const char *s) { memcpy(global_bytes, s, 64); refill_then_read(); return 0; }
__attribute__((noinline)) int reset_only_case(const char *s) { memcpy(global_bytes, s, 64); reset_only(); return 0; }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return relay_ret_case(argv[1]) + relay2_ret_case(argv[1]) + arg_ret_case(argv[1])
         + ignore_case(argv[1]) + refill_case(argv[1]) + refill_read_case(argv[1])
         + reset_only_case(argv[1]);
}
"""


@pytest.mark.parametrize("debug", [True, False], ids=["typed", "stripped"])
def test_global_continuation_returned_aliases_and_source_destinations(tmp_path,
                                                                     shared_bn, debug):
    """Round-3 review F18, F19 and F20.

    F18: the global's address comes back through a CHAIN of returning
    wrappers -- only the innermost helper names it -- and the consumer reads
    `+ 8`, bytes the one-byte reset never touched. A partial write is not
    whole-object sanitisation.
    F19: `reset_arg(char *p)` returns its own FORMAL, so the actual the
    caller passed has to be mapped onto it and back.
    F20: `read(0, g, 64)` FILLS g; its role lives in the model's `sources`,
    and reading that role is what stops a refill raising a false frontier.
    An ignored return and a refill are not continuations, but neither clears
    the copied bytes, so each ends in the P2-11.1 exit `coarse_memory_store`
    citing a real reader -- never the refill or the relay. A full `memset`
    reset stays silent."""
    name = "relay" if debug else "relay_nodbg"
    src = tmp_path / f"{name}.c"
    src.write_text(_RELAY_SRC)
    binary = tmp_path / name
    _compile(src, binary, ["-fno-builtin", "-fno-inline"] + ([] if debug else ["-g0"]))
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F18 / F19: the boundary is reported at the downstream consumer
    for fn, consumer in (("relay_ret_case", "use_relay_ret"),
                         ("relay2_ret_case", "use_relay2_ret"),
                         ("arg_ret_case", "use_arg_ret")):
        result, leaves = gmc(fn)
        [leaf] = leaves
        assert [r["display_name"] for r in leaf["references"]] == [consumer], (fn, leaf)
        assert leaf["references"][0]["access"] == "call_arg", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)

    # ...an ignored return and a refill are not continuations; the bytes
    # they leave live end in the exit store, which cites a real reader
    for fn in ("ignore_case", "refill_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), \
            (fn, store)
        assert [(r["display_name"], r["offset"]) for r in store["references"]] \
            == [("use_arg_ret", 8)], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, (fn, result)
    # ...and a full reset is silent
    result, leaves = gmc("reset_only_case")
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True, result

    # F20 control: a refill followed by a genuine read of the bytes blocks
    result, leaves = gmc("refill_read_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["refill_then_read"], leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result


_R3OBJECTS_SRC = r"""
#include <stdlib.h>
#include <string.h>
char global_bytes[128] = "safe";
char clean_bytes[128] = "clean";
__attribute__((noinline)) void reader(void) { system(global_bytes); }
__attribute__((noinline)) void quiet(void) { system(clean_bytes); }
__attribute__((noinline)) char *reset_ret(void) { return memset(global_bytes, 0, 1); }
__attribute__((noinline)) void indirect_reader(char *(*cb)(void)) { char *p = cb(); system(p + 8); }
__attribute__((noinline)) char *recursive_alias(char *p, int n) {
    if (n > 0) return recursive_alias(p, n - 1);
    return p;
}
__attribute__((noinline)) void recursive_reader(int n) { system(recursive_alias(global_bytes, n) + 8); }
__attribute__((noinline)) int interior_case(const char *s) { memset(global_bytes, 'x', 8); memcpy(global_bytes + 8, s, 64); reader(); return 0; }
__attribute__((noinline)) int disjoint_case(const char *s) { memcpy(global_bytes, s, 64); quiet(); return 0; }
__attribute__((noinline)) int indirect_case(const char *s, char *(*cb)(void)) { memcpy(global_bytes, s, 64); indirect_reader(cb); return 0; }
__attribute__((noinline)) int recursive_case(const char *s, int n) { system(s); memcpy(global_bytes, s, 64); recursive_reader(n); return 0; }
__attribute__((noinline)) int baseline_case(const char *s) { memcpy(global_bytes, s, 64); reader(); return 0; }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    switch (argc) {
    case 2: return interior_case(argv[1]);
    case 3: return disjoint_case(argv[1]);
    case 4: return indirect_case(argv[1], reset_ret);
    case 5: return baseline_case(argv[1]);
    default: return recursive_case(argv[1], argc);
    }
}
"""


@pytest.mark.parametrize("debug", [True, False], ids=["typed", "stripped"])
def test_global_continuation_interior_writes_pins_and_cycles(tmp_path, shared_bn, debug):
    """Round-3 review R1, R2, R3 and R4.

    R1: `memcpy(g + 8, s, 64)` writes the SAME object `system(g)` reads --
    BN invents a nameless data variable at the interior address, and keying
    the write there cleared a real flow. A disjoint neighbour is still a
    different object: its reader is never cited, and the bytes still live at
    the return end in the P2-11.1 exit store naming `reader`.
    R2: pinning the indirect site with `--resolve-map` must not LOSE the
    boundary it discloses while unpinned.
    R3: a self-recursive alias must not unwind the walk, and a sink already
    observed must survive if it ever does.
    R4: the re-seed advice has to be runnable."""
    # A selector is a BASENAME: this lane's bridge is shared, so a fixture
    # name another test already uses would resolve to the wrong target.
    name = "r3objects" if debug else "r3objects_nodbg"
    src = tmp_path / f"{name}.c"
    src.write_text(_R3OBJECTS_SRC)
    binary = tmp_path / name
    _compile(src, binary, ["-fno-builtin", "-fno-inline"] + ([] if debug else ["-g0"]))
    target = shared_bn.load(binary, copy=False)

    def gmc(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # R1: the interior write is the object the reader reads, and WHICH bytes
    # were written is kept.
    result, leaves = gmc("interior_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["reader"], leaf
    assert leaf["global"]["written_offsets"] == ["0x8"], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # ...and a disjoint object is not folded into it
    result, leaves = gmc("disjoint_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "global_bytes"), store
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # R2: unpinned the indirect site is disclosed; pinned it names the reader
    result, leaves = gmc("indirect_case")
    [leaf] = leaves
    assert leaf["reachability"]["cause"] == "unresolved_call_target", leaf
    site = leaf["reachability"]["unresolved_calls"][0]
    pin = tmp_path / "resolve_map.json"
    pin.write_text(json.dumps({site: ["reset_ret"]}))
    result, leaves = gmc("indirect_case", "--resolve-map", str(pin))
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["indirect_reader"], leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result

    # R3: the cycle is classified, and the sink seen before it survives
    result, leaves = gmc("recursive_case")
    assert [s["sink"]["class"] for s in result["reached_sinks"]] == ["command_injection"]
    assert "recursion" not in (result["stats"]["truncation_cause"] or [])
    assert any(r["display_name"] == "recursive_reader"
               for lf in leaves for r in lf["references"]), leaves

    # R4: the advice names a locator that RUNS, or none. `reader` hands the
    # global's constant address straight to `system`: seeding that argument
    # would only re-find the sink (W07-13), so there is no reseed.
    result, leaves = gmc("baseline_case")
    [leaf] = leaves
    assert "var:VAR" not in leaf["detail"], leaf
    assert [r["seed"] for r in leaf["references"]] == [None], leaf["references"]
    assert "reseed" not in leaf, leaf


_ASTRA_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char alias_global[128];
extern char alias_middle[64];
__asm__(".globl alias_middle\n.set alias_middle, alias_global+64\n"
        ".type alias_middle,@object\n.size alias_middle,64\n");
char neighbour_global[64];
char span_global[96] = "seed";
char tracked[128] = "seed";
volatile unsigned char span_observed;
N unsigned char unused_probe(void) { return span_global[40]; }
N void span_reader(void) { char local[80]; memcpy(local, span_global, 80); span_observed = local[40]; }
N void read_alias_middle(void) { system(alias_middle + 16); }
N void read_neighbour(void) { system(neighbour_global + 8); }
N char *refill_alias(char *a) { return fgets(a, 2, stdin); }
N void use_fgets_alias(void) { char *p = refill_alias(tracked); if (p) system(p + 8); }
N void ignore_fgets(void) { refill_alias(tracked); }
N char *relay0(void) { return memset(tracked, 0, 1); }
N char *relay1(void) { return relay0(); }
N char *relay2(void) { return relay1(); }
N char *relay3(void) { return relay2(); }
N char *relay4(void) { return relay3(); }
N void use_relay4(void) { system(relay4() + 24); }
N void alias_cross_case(char *p) { memcpy(alias_global, p, 96); read_alias_middle(); }
N void alias_inside_case(char *p) { memcpy(alias_global, p, 32); read_alias_middle(); }
N void neighbour_case(char *p) { memcpy(alias_global, p, 32); read_neighbour(); }
N void span_case(char *p) { memcpy(span_global + 40, p, 16); span_reader(); }
N void fgets_case(char *p) { memcpy(tracked, p, 64); use_fgets_alias(); }
N void fgets_ignored_case(char *p) { memcpy(tracked, p, 64); ignore_fgets(); }
N void chain_case(char *p) { memcpy(tracked, p, 64); use_relay4(); }
char deep_tracked[128] = "seed";
N char *deep_relay0(void) { return memset(deep_tracked, 0, 1); }
N char *deep_relay1(void) { return deep_relay0(); }
N char *deep_relay2(void) { return deep_relay1(); }
N char *deep_relay3(void) { return deep_relay2(); }
N char *deep_relay4(void) { return deep_relay3(); }
N char *deep_relay5(void) { return deep_relay4(); }
N char *deep_relay6(void) { return deep_relay5(); }
N void use_deep_relay6(void) { system(deep_relay6() + 24); }
N void deep_chain_case(char *p) { memcpy(deep_tracked, p, 64); use_deep_relay6(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    switch (argc) {
    case 2: alias_cross_case(argv[1]); break;
    case 3: alias_inside_case(argv[1]); break;
    case 4: neighbour_case(argv[1]); break;
    case 5: span_case(argv[1]); break;
    case 6: fgets_case(argv[1]); break;
    case 7: fgets_ignored_case(argv[1]); break;
    case 8: chain_case(argv[1]); break;
    default: deep_chain_case(argv[1]); break;
    }
    return 0;
}
"""


def test_global_continuation_object_extent_aliases_and_internal_bounds(tmp_path,
                                                                      shared_bn):
    """BN21-OBJ-01, OBJ-02, ALIAS-01 and BOUND-03 on a real stripped build.

    OBJ-01: `.set alias_middle, alias_global + 64` puts a symbol INSIDE an
    untyped object, so its end was taken from the layout and a reader at +80
    fell outside it. A copy of 96 bytes proves otherwise -- and the
    uncertainty that remains is disclosed, never resolved by guesswork.
    OBJ-02: an analysis-created interior data variable is not a second
    object; a reader spanning the base still sees the interior write.
    ALIAS-01: `fgets` hands its destination back, so a partial refill
    followed by `system(p + 8)` reads bytes it never touched; an ignored
    refill is no continuation, its live bytes end in the P2-11.1 exit store.
    BOUND-03: a returning-wrapper chain within the INTERNAL pointer-return
    depth is followed to the reader at its end; one past it is unknown, not
    "no alias"."""
    src = tmp_path / "r3astra.c"
    src.write_text(_ASTRA_SRC)
    binary = tmp_path / "r3astra"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # OBJ-01: the copy spans the alias symbol and the reader past it is named.
    # The image DECLARES `alias_global` 128 bytes (and `alias_middle` as 64 of
    # them), so where the object ends is established, not unproven (P1-05);
    # the undeclared shape is covered by the mocked suite.
    result, leaves = gmc("alias_cross_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["read_alias_middle"], leaf
    assert leaf["references"][0]["offset"] == 80, leaf
    assert "object_extent_unproven" not in leaf["reachability"]["causes"], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and a copy that stays inside the layout keeps the neighbour disjoint
    for fn in ("alias_inside_case", "neighbour_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is True, fn

    # OBJ-02: the interior variable BN invented is the same object
    result, leaves = gmc("span_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["span_reader"], leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ALIAS-01: a used returned destination blocks, an ignored one does not
    result, leaves = gmc("fgets_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["use_fgets_alias"], leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    result, leaves = gmc("fgets_ignored_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), store
    assert [(r["display_name"], r["offset"]) for r in store["references"]] \
        == [("use_fgets_alias", 8)], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # BOUND-03: five returning wrappers are within the internal pointer-return
    # depth -- the chain resolves and its consumer is the named reader...
    result, leaves = gmc("chain_case")
    [leaf] = leaves
    assert [(r["display_name"], r["access"], r["offset"]) for r in leaf["references"]] \
        == [("use_relay4", "call_arg", 24)], leaf
    assert (leaf["reachability"]["status"], leaf["reachability"]["causes"]) == ("complete", [])
    assert result["stats"]["truncated"] is False, result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # ...and seven run past it: the bound is disclosed, never read as "no alias".
    result, leaves = gmc("deep_chain_case")
    [leaf] = leaves
    assert leaf["references"] == [], leaf
    assert leaf["reachability"]["status"] == "truncated", leaf
    assert leaf["reachability"]["cause"] == "pointer_return_depth", leaf
    assert result["stats"]["truncation_cause"] == ["pointer_return_depth"], result["stats"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


_UNREAD_OVERLAY_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char request_bytes[128];
N int opaque_stage(int token) {
    /* Past analysis.limits.maxFunctionSize, so BN MEASURES this body and
       refuses to analyse it -- the shape a model must never speak for. */
    __asm__ volatile(".rept 70000\n nop\n .endr\n");
    system(request_bytes + 24);
    return token;
}
N int wrapped_stage(int token) { return opaque_stage(token); }
N void direct_case(char *p) { memcpy(request_bytes, p, 64); opaque_stage(1); }
N void wrapped_case(char *p) { memcpy(request_bytes, p, 64); wrapped_stage(1); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    if (argc == 2) direct_case(argv[1]); else wrapped_case(argv[1]);
    return 0;
}
"""


def test_a_partial_model_never_buys_off_an_unread_body(tmp_path, shared_bn):
    """BN21-REACH-01 on a real build with a body BN refuses to analyse.

    A truthful additive overlay (`arg:0 -> ret`) used to DELETE the
    unread-body frontier and the analysis-incomplete signal: accurate
    positive evidence made the answer less conservative, which is backwards.
    """
    src = tmp_path / "r3unread.c"
    src.write_text(_UNREAD_OVERLAY_SRC)
    binary = tmp_path / "r3unread"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-g0"])
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps(
        {"opaque_stage": {"propagates": [{"from": "arg:0", "to": "ret"}]}}))
    target = shared_bn.load(binary, copy=False)

    for fn in ("direct_case", "wrapped_case"):
        for extra in ((), ("--models", str(overlay))):
            result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                              "-f", fn, "--source", "param:0", *extra)
            leaves = [lf for lf in result["leaves"]
                      if lf["kind"] == "global_memory_continuation"]
            assert [lf["reachability"]["cause"] for lf in leaves] == [
                "unread_callee_body"], (fn, extra, result["leaves"])
            assert result["diagnostics"]["analysis_incomplete"] is True, (fn, extra)
            assert result["diagnostics"]["safe_to_report_all_clear"] is False


_IC05_SRC = r"""
/* IC05 / BN6-C008: public synthetic, analyze only, never execute. */
#include <stddef.h>
#include <string.h>
#define N __attribute__((noinline, noipa))
/* A single tail jump even in this -O0 build, so Binary Ninja marks it a thunk
   the way it marks a real PLT/veneer stub. */
#define TAIL __attribute__((noinline, noipa, optimize("O2")))

extern int system(const char *);
char scratch[256];

/* Padded past analysis.limits.maxFunctionSize, so Binary Ninja MEASURES this
   body and declines to analyse it; its tail is a truthful identity return,
   which is exactly what the overlay below states and all it states. */
N long unread_body(long token, long other) {
    __asm__ volatile(".rept 70000\n nop\n .endr\n");
    return token + other;
}
TAIL long unread_thunk(long token) { return unread_body(token, 0); }
TAIL long unread_multihop(long token) { return unread_thunk(token); }
N long wrap_unread(long token) { long r = unread_body(token, 0); return r + 3; }
N long analysed_body(long token) { system((const char *)token); return token; }

N long call_unread(long token) { long r = unread_body(token, 0); return r + 1; }
N long call_thunk(long token) { long r = unread_thunk(token); return r + 1; }
N long call_multihop(long token) { long r = unread_multihop(token); return r + 1; }
N long call_wrapper(long token) { long r = wrap_unread(token); return r + 1; }
N long call_analysed(long token) { long r = analysed_body(token); return r + 1; }
N long call_import(long token) { memset(scratch, (int)token, 16); return token; }
long (*volatile hook)(long, long) = unread_body;
N long call_indirect(long token) { long r = hook(token, 0); return r + 1; }

/* ONE call to the unread body, reached from two tainted-argument contexts. */
N long nested(long a, long b) { long r = unread_body(a, b); return r + 1; }
N long two_contexts(long token) { return nested(token, 0) + nested(0, token); }

/* Both arguments arrive in the registers the call passes them in, so an
   optimised build writes NEITHER before the call -- and Binary Ninja, with no
   prototype for the body it never analysed, then recovers no arguments at
   all for that call site. The guard exists so BN can still recover this
   function's OWN parameters (it reads them) without writing the registers. */
N long abi_entry(long a, long b) {
    if (a == 5 || b == 7) return 0;
    long r = unread_body(a, b);
    return r + 1;
}

/* An optimised TAIL call: `return unread_body(a, b)` leaves both arguments
   exactly where they already are, so the call writes no argument register at
   all -- and Binary Ninja lifts it as a tail call, not an ordinary one. */
N long tail_entry(long a, long b) {
    if (a == 5 || b == 7) return 0;
    return unread_body(a, b);
}

/* The seeded value rides argument 1; argument 0 is a constant. */
N long other_arg(long token) { long r = unread_body(0, token); return r + 1; }
/* Nothing the seed reaches is passed at all. */
N long constant_only(long token) {
    if (token == 3) return 0;
    long r = unread_body(7, 9);
    return r + 1;
}

/* An ANALYSED callee reached the same way: the argument is already in its
   register, so an optimised call site writes none. The guard keeps this
   function's own parameter recoverable. */
N long calls_analysed(long a) {
    if (a == 5) return 0;
    long r = analysed_body(a);
    return r + 1;
}

/* BN recovers argument 0 (a constant it sees written) and nothing else; the
   seed rides argument 1, which the call site never writes. */
N long prefix_entry(long a, long b) {
    if (a == 5 || b == 7) return 0;
    long r = unread_body(7, b);
    return r + 1;
}

/* A loop before the call: the backwards CFG closure walks a cycle. */
N long loop_entry(long a, long b) {
    if (a == 5 || b == 7) return 0;
    for (volatile int i = 0; i < 4; i++) { }
    long r = unread_body(a, b);
    return r + 1;
}

/* An analysed callee that consumes BOTH argument registers, reached with a
   constant in the first and the seed in the second. */
N long two_arg_consumer(long a, long b) { system((const char *)b); return a + b; }
N long calls_second(long a) {
    if (a == 5) return 0;
    long r = two_arg_consumer(0, a);
    return r + 1;
}

/* The analysed callee behind a veneer, and behind a pinnable indirect site. */
TAIL long analysed_thunk(long a) { return analysed_body(a); }
N long calls_analysed_thunk(long a) {
    if (a == 5) return 0;
    long r = analysed_thunk(a);
    return r + 1;
}

/* An indirect site whose targets the VALUE SET itself pins: a const table of
   function pointers, both entries reaching the skipped body, with the
   arguments already in the registers that pass them. */
static long (*const vs_hooks[2])(long, long) = { unread_body, unread_body };
N long value_set_entry(long a, long b) {
    if (a == 5 || b == 7) return 0;
    return vs_hooks[a & 1](a, b) + 1;
}

/* A pinnable indirect site into an ANALYSED callee whose FORMALS are correct
   (nothing redeclares it), at a call that writes no argument register. */
N long abi_formal_consumer(long a) { system((const char *)a); return a + 1; }
long (*volatile abi_formal_hook)(long) = abi_formal_consumer;
N long calls_formal_indirect(long a) {
    if (a == 5) return 0;
    long r = abi_formal_hook(a);
    return r + 1;
}

int main(int argc, char **argv) {
    (void)argv;
    return (int)(call_unread(argc) + call_thunk(argc) + call_multihop(argc)
                 + call_wrapper(argc) + call_analysed(argc) + call_import(argc)
                 + call_indirect(argc) + two_contexts(argc) + other_arg(argc)
                 + constant_only(argc) + abi_entry(argc, argc)
                 + tail_entry(argc, argc) + calls_analysed(argc)
                 + prefix_entry(argc, argc) + loop_entry(argc, argc)
                 + calls_second(argc) + calls_analysed_thunk(argc)
                 + value_set_entry(argc, argc) + calls_formal_indirect(argc));
}
"""

_IC05_OVERLAYS = {
    "partial": {"unread_body": {"propagates": [{"from": "arg:0", "to": "ret"}]}},
    "empty-entry": {"unread_body": {}},
    # Accepted, but states no coverage and no return relation at all.
    "irrelevant": {"unread_body": {"propagates": [{"from": "arg:0", "to": "arg:0"}]}},
    # A full overlay that also arms a sink: still no claim about the body.
    "full": {"unread_body": {"propagates": [{"from": "arg:0", "to": "ret"},
                                            {"from": "arg:1", "to": "ret"}],
                             "sink": {"class": "command_injection",
                                      "tainted_args": [0],
                                      "detail": "modeled sink"}}},
}


def test_a_matching_model_never_certifies_an_unread_argument_callee(tmp_path, shared_bn):
    """IC05 / BN6-C008 on a real build: the tainted-ARGUMENT descent path.

    BN21-REACH-01 closed this question for tainted GLOBALS (IC03). On the
    argument path the unread-body accounting still sat inside the branch that
    runs only when nothing in the call chain is modeled, so ANY matching entry
    -- a truthful `arg:0 -> ret`, an empty `{}`, a reflexive rule that states
    nothing -- turned `analysis_incomplete` off and the all-clear on, for a
    body Binary Ninja never read. Body coverage is a property of the binary.
    """
    src = tmp_path / "ic05_unread.c"
    src.write_text(_IC05_SRC)
    binary = tmp_path / "ic05_unread"
    _compile(src, binary, ["-fno-builtin"])
    overlays = {
        "partial": {"unread_body": {"propagates": [{"from": "arg:0", "to": "ret"}]}},
        "empty-entry": {"unread_body": {}},
        # Accepted, but states no coverage and no return relation at all.
        "irrelevant": {"unread_body": {"propagates": [{"from": "arg:0", "to": "arg:0"}]}},
    }
    files = {}
    for label, doc in overlays.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(doc))
        files[label] = ("--models", str(path))
    files["no-model"] = ()
    target = shared_bn.load(binary, copy=False)

    body = _resolve_addr(shared_bn, "unread_body")
    skipped = _bn_json(shared_bn, "-t", target, "decompile", body)
    # The falsification bar: a real in-binary FunctionSymbol whose MLIL is
    # absent because BN measured the body and skipped it -- not an import.
    assert skipped["analysis_skipped"] is True, skipped
    assert _bn_json(shared_bn, "-t", target, "function", "info",
                    body)["imported"] is False, body

    pin = tmp_path / "ic05_rmap.json"
    pin.write_text(json.dumps(
        {_indirect_call_addr(shared_bn, "call_indirect", "ic05"): [body]}))
    # `unread_thunk` passes a constant second argument before its jump, so it
    # is a body the walk reads (P2-03) and the unread body is reached from
    # inside it; only a pure veneer would be named as `via`.
    routes = {
        "call_unread": None,
        "call_thunk": None,
        "call_multihop": None,
        "call_wrapper": None,
        "call_indirect": None,
    }

    for fn, via in routes.items():
        pinned = ("--resolve-map", str(pin)) if fn == "call_indirect" else ()
        for label, extra in files.items():
            result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                              "--source", "param:0", "--max-depth", "16",
                              *extra, *pinned)
            where = (fn, label)
            assert result["stats"]["analysis_incomplete"] is True, (where, result["stats"])
            # The BODY is named on every route; a veneer is not the function
            # an agent has to analyse.
            assert result["stats"]["analysis_incomplete_functions"] == ["unread_body"], where
            assert result["diagnostics"]["safe_to_report_all_clear"] is False, where
            [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "unread_callee_body"]
            assert leaf["callee"]["name"] == "unread_body", (where, leaf)
            assert leaf["callee"]["address"] == body, (where, leaf)
            assert leaf["skip_reason"] == "analysis_skipped:ExceedFunctionSizeSkipReason", leaf
            assert leaf["tainted_args"] == [0], (where, leaf)
            assert leaf["modeled"] is (label != "no-model"), (where, leaf)
            assert leaf["via"] == via, (where, leaf)

    # An import has no in-binary implementation, so there is nothing
    # incomplete about not reading one.
    imported = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "call_import",
                        "--source", "param:0", *files["partial"])
    assert imported["stats"]["analysis_incomplete"] is False, imported["stats"]
    assert imported["diagnostics"]["safe_to_report_all_clear"] is True, imported

    # A fully analysed body stays descended and authoritative: only the
    # descent finds the `system` call inside it.
    analysed = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "call_analysed",
                        "--source", "param:0", *files["partial"])
    assert analysed["stats"]["analysis_incomplete"] is False, analysed["stats"]
    assert any(s["sink"]["callee"] == "system" for s in analysed["reached_sinks"]), analysed

    # Text mode carries the same two facts, and the printed advice runs as
    # printed -- then the body IS read, and the boundary is gone. The
    # disclosure is about coverage, not a gate that can never be satisfied.
    modeled = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "call_unread",
                       "--source", "param:0", *files["partial"])
    text = shared_bn.run("-t", target, "taint", "forward", "-f", "call_unread",
                         "--source", "param:0", *files["partial"], "--verbose")
    assert text.returncode == 0, text.stderr
    assert "NOTE: analysis incomplete" in text.stdout, text.stdout
    assert "unread_callee_body @" in text.stdout, text.stdout

    ran, _templates = _run_or_check_template(shared_bn, _forward_prose(modeled))
    assert ran, _forward_prose(modeled)
    reread = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "call_unread",
                      "--source", "param:0", *files["partial"], timeout=300.0)
    assert reread["stats"]["analysis_incomplete"] is False, reread["stats"]
    assert [lf for lf in reread["leaves"] if lf["kind"] == "unread_callee_body"] == []


def test_one_unread_call_is_one_boundary_with_every_argument_context(tmp_path, shared_bn):
    """IC05 review F2 on a real build.

    `two_contexts(p)` calls `nested(p, 0)` and `nested(0, p)`, so the ONE call
    `nested` makes into the unread body is reached from two tainted-argument
    contexts. That is one hole: it used to be counted twice, and the text row
    printed the first context's arguments with the second hidden behind a
    `[2 leaves]` tally.
    """
    src = tmp_path / "ic05_contexts.c"
    src.write_text(_IC05_SRC)
    binary = tmp_path / "ic05_contexts"
    _compile(src, binary, ["-fno-builtin"])
    overlay = tmp_path / "partial.json"
    overlay.write_text(json.dumps(
        {"unread_body": {"propagates": [{"from": "arg:0", "to": "ret"}]}}))
    target = shared_bn.load(binary, copy=False)

    for extra in ((), ("--models", str(overlay))):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f",
                          "two_contexts", "--source", "param:0", "--max-depth", "16",
                          *extra)
        leaves = [lf for lf in result["leaves"] if lf["kind"] == "unread_callee_body"]
        assert len(leaves) == 1, (extra, result["leaves"])
        assert leaves[0]["tainted_args"] == [0, 1], leaves[0]
        frontier = result["diagnostics"]["frontier"]
        assert frontier["by_kind"]["unread_callee_body"] == 1, frontier

    text = shared_bn.run("-t", target, "taint", "forward", "-f", "two_contexts",
                         "--source", "param:0", "--max-depth", "16", "--verbose")
    assert text.returncode == 0, text.stderr
    row = next(line for line in text.stdout.splitlines()
               if line.strip().startswith("unread_callee_body"))
    assert "[0, 1]" in row, row
    assert "leaves]" not in row, row


def test_an_argument_an_optimised_call_drops_is_recovered_from_its_register(
        tmp_path, shared_bn):
    """The ABI-provenance gap, on an optimised real build.

    Binary Ninja recovers a call's arguments at the CALL SITE, from the
    registers it sees written there plus the callee's prototype. A callee it
    never analysed has no prototype, and a call whose arguments already sit in
    the registers that pass them writes none -- so `-O2` leaves
    `unread_body()` with no arguments at all, and the walk reported a clean
    all-clear at a call the machine code plainly hands the seeded value to.

    Only what the IL proves is recovered: a register with no definition that
    nothing could have clobbered still carries the function's entry value. A
    call whose arguments Binary Ninja DID recover is never second-guessed --
    that recovery is its arity determination, and a register past it holds
    dead scratch as often as an argument.
    """
    src = tmp_path / "ic05_abi.c"
    src.write_text(_IC05_SRC)
    binary = tmp_path / "ic05_abi"
    # `-g0`: with DWARF, Binary Ninja takes the prototype from the debug info
    # and recovers the arguments itself. The gap is what a stripped optimised
    # build leaves.
    _compile(src, binary, ["-O2", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    body = _resolve_addr(shared_bn, "unread_body")
    assert _bn_json(shared_bn, "-t", target, "decompile",
                    body)["analysis_skipped"] is True, body
    # The premise: BN recovered NO arguments for this call site.
    il = _bn_json(shared_bn, "-t", target, "il", "abi_entry")
    assert "unread_body()" in json.dumps(il), il

    def forward(fn, *sources):
        args = []
        for s in sources or ("param:0",):
            args += ["--source", s]
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                        *args, "--max-depth", "16")

    def boundary(result):
        [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "unread_callee_body"]
        return leaf

    first = forward("abi_entry")
    leaf = boundary(first)
    assert leaf["callee"]["name"] == "unread_body", leaf
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["abi_recovered_args"] == [0], leaf
    assert first["stats"]["analysis_incomplete_functions"] == ["unread_body"]
    assert first["diagnostics"]["safe_to_report_all_clear"] is False
    assert any("calling convention's register" in a
               for a in first["assumptions"]), first["assumptions"]

    # A later argument register, and both at once: each index is recovered on
    # its own register, and an untainted one is simply not claimed.
    assert boundary(forward("abi_entry", "param:1"))["abi_recovered_args"] == [1]
    assert boundary(forward("abi_entry", "param:0", "param:1")
                    )["abi_recovered_args"] == [0, 1]

    # A TAIL call is a call: `return unread_body(a, b)` leaves both arguments
    # where they are, so BN lifts a tail call carrying no arguments at all.
    # Matching only ordinary calls left it certifying the skipped body (F4).
    tail = forward("tail_entry")
    leaf = boundary(tail)
    assert leaf["callee"]["name"] == "unread_body", leaf
    assert leaf["tainted_args"] == [0], leaf
    assert leaf["abi_recovered_args"] == [0], leaf
    assert tail["stats"]["analysis_incomplete_functions"] == ["unread_body"]
    assert tail["diagnostics"]["safe_to_report_all_clear"] is False
    assert boundary(forward("tail_entry", "param:1"))["abi_recovered_args"] == [1]
    assert boundary(forward("tail_entry", "param:0", "param:1")
                    )["abi_recovered_args"] == [0, 1]
    for overlay in _IC05_OVERLAYS:
        path = tmp_path / f"tail-{overlay}.json"
        path.write_text(json.dumps(_IC05_OVERLAYS[overlay]))
        modeled = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f",
                           "tail_entry", "--source", "param:0", "--max-depth", "16",
                           "--models", str(path))
        assert modeled["stats"]["analysis_incomplete"] is True, (overlay, modeled["stats"])
        assert boundary(modeled)["abi_recovered_args"] == [0], overlay
        # The zero-sink gate exists only on a zero-sink run; the sink-bearing
        # overlay reports its finding instead, and still names the body it
        # did not read.
        if not modeled["reached_sinks"]:
            assert modeled["diagnostics"]["safe_to_report_all_clear"] is False, overlay

    # A recovered PREFIX says nothing about the arguments past it: BN recovers
    # the constant argument 0 it sees written and misses argument 1, which the
    # call site never writes because the seed is already in its register.
    prefix = forward("prefix_entry", "param:1")
    leaf = boundary(prefix)
    assert leaf["tainted_args"] == [1], leaf
    assert leaf["abi_recovered_args"] == [1], leaf
    assert prefix["diagnostics"]["safe_to_report_all_clear"] is False

    # A loop before the call: the backwards CFG closure walks a cycle of blocks
    # Binary Ninja hands out as a fresh Python object each time. It terminates,
    # and the answer is still honest.
    looped = forward("loop_entry")
    assert boundary(looped)["callee"]["name"] == "unread_body", looped
    assert looped["diagnostics"]["safe_to_report_all_clear"] is False

    # Review F11: an indirect site whose targets the VALUE SET itself pins is
    # routed into the same skipped body, and the gate must consider it exactly
    # as the descent does.
    pinned = forward("value_set_entry")
    assert boundary(pinned)["callee"]["name"] == "unread_body", pinned
    assert pinned["stats"]["analysis_incomplete_functions"] == ["unread_body"]
    assert pinned["diagnostics"]["safe_to_report_all_clear"] is False

    # Binary Ninja DID recover this call's arguments (it writes both registers
    # at the call site), so nothing is recovered on top of them -- and its own
    # recovery still carries the seed into the boundary.
    second = forward("other_arg")
    leaf = boundary(second)
    assert leaf["tainted_args"] == [1], leaf
    assert leaf["abi_recovered_args"] == [], leaf

    # The must-not-fire twin: every argument is a constant, so no register
    # carries the seed and nothing is recovered or disclosed.
    clean = forward("constant_only")
    assert [lf for lf in clean["leaves"] if lf["kind"] == "unread_callee_body"] == [], clean
    assert clean["stats"]["analysis_incomplete"] is False, clean["stats"]

    # Text names the recovered index, so a reader can tell an argument Binary
    # Ninja recovered from one the calling convention's register proved.
    text = shared_bn.run("-t", target, "taint", "forward", "-f", "abi_entry",
                         "--source", "param:0", "--max-depth", "16", "--verbose")
    assert text.returncode == 0, text.stderr
    row = next(line for line in text.stdout.splitlines()
               if line.strip().startswith("unread_callee_body"))
    assert "ABI-recovered arg(s) [0]" in row, row


def test_an_under_recovered_argument_into_an_analysed_callee_is_a_frontier(
        tmp_path, shared_bn):
    """Review F5: the same ABI evidence, for a callee that IS analysed.

    `analysed_body` consumes its first argument (it hands it to `system`). Force
    the recovery miss the way the #381 lane already does — declare it 0-arity —
    and the call site, which writes no argument register because the value is
    already in `rdi`, carries no arguments at all. The seeded value provably
    reaches a register the callee READS as an incoming value, so the walk must
    not report an all-clear. No formal is invented for the callee and no sink is
    claimed: the flow stops at a named frontier and says why.
    """
    src = tmp_path / "ic05_underrec.c"
    src.write_text(_IC05_SRC)
    binary = tmp_path / "ic05_underrec"
    _compile(src, binary, ["-O2", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn="calls_analysed"):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f",
                        fn, "--source", "param:0", "--max-depth", "16")

    def frontiers(result):
        return [lf for lf in result["leaves"]
                if lf.get("cause") == "arg_under_recovered"]

    # Baseline control: the argument IS recovered, so taint descends and the
    # `system` sink is found -- no under-recovery frontier at all.
    before = forward()
    assert any(s["sink"]["callee"] == "system" for s in before["reached_sinks"]), before
    assert frontiers(before) == [], before["leaves"]

    proto = shared_bn.run("-t", target, "proto", "set", "analysed_body",
                          "void analysed_body(void)", "--format", "json")
    assert proto.returncode == 0, proto.stderr

    after = forward()
    assert not any(s["sink"]["callee"] == "system" for s in after["reached_sinks"]), after
    [leaf] = frontiers(after)
    assert leaf["callee"]["name"] == "analysed_body", leaf
    assert leaf["register"] == "rdi", leaf
    assert leaf["argument_index"] == 0, leaf
    assert leaf["recovered_params"] == 0, leaf
    # The register was never written at the call site: it still carries the
    # value the analysed function was entered with.
    assert leaf["via"] == "entry_value", leaf
    assert after["diagnostics"]["safe_to_report_all_clear"] is False, after

    # Review F9: a veneer routes to the SAME analysed body, and the check must
    # follow the candidate seam the descent uses rather than the direct target
    # alone.
    routed = forward("calls_analysed_thunk")
    [leaf] = frontiers(routed)
    assert leaf["callee"]["name"] == "analysed_body", leaf
    assert leaf["argument_index"] == 0, leaf
    assert routed["diagnostics"]["safe_to_report_all_clear"] is False, routed

    # Review F8: the first missing argument is a CONSTANT and the seed rides
    # the second. Stopping at the first missing register dropped it silently.
    proto = shared_bn.run("-t", target, "proto", "set", "two_arg_consumer",
                          "void two_arg_consumer(void)", "--format", "json")
    assert proto.returncode == 0, proto.stderr
    later = forward("calls_second")
    [leaf] = frontiers(later)
    assert leaf["callee"]["name"] == "two_arg_consumer", leaf
    assert leaf["tainted_args"] == [1], leaf
    assert leaf["register"] == "rsi", leaf
    assert leaf["argument_index"] == 1, leaf
    assert later["diagnostics"]["safe_to_report_all_clear"] is False, later

    # Review F12: an agent-pinned indirect site into a callee whose FORMALS are
    # correct. The call itself recovered no arguments, so argument 0 is missing
    # and must be measured against the CALL, not against the callee's formals
    # (counting from the formals skipped exactly this index).
    site = _indirect_call_addr(shared_bn, "calls_formal_indirect", "ic05")
    pin = tmp_path / "ic05_underrec_rmap.json"
    pin.write_text(json.dumps(
        {site: [_resolve_addr(shared_bn, "abi_formal_consumer")]}))
    mapped = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f",
                      "calls_formal_indirect", "--source", "param:0",
                      "--max-depth", "16", "--resolve-map", str(pin))
    [leaf] = frontiers(mapped)
    assert leaf["callee"]["name"] == "abi_formal_consumer", leaf
    assert leaf["argument_index"] == 0, leaf
    assert leaf["recovered_params"] == 0, leaf
    assert mapped["diagnostics"]["safe_to_report_all_clear"] is False, mapped


_OVERLAP_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char packet[128];
extern char packet_tail[64];
__asm__(".globl packet_tail\n.set packet_tail, packet+64\n"
        ".type packet_tail,@object\n.size packet_tail,64\n");
char other[128];
char tracked[128] = "seed";
N void consumer(void) { char local[112]; memcpy(local, packet, sizeof(local)); system(local + 80); }
N void shallow(void) { char local[16]; memcpy(local, packet, sizeof(local)); system(local); }
N void other_reader(void) { char local[16]; memcpy(local, other, sizeof(local)); system(local); }
N void reader(void) { system(tracked); }
N char *refill(char *a) { return fgets(a, 2, stdin); }
N void use_refill(void) { char *p = refill(tracked); if (p) system(p + 8); }
N void overlap_case(char *s) { memcpy(packet_tail, s, 32); consumer(); }
N void shallow_case(char *s) { memcpy(packet_tail, s, 32); shallow(); }
N void neighbour_case(char *s) { memcpy(packet_tail, s, 32); other_reader(); }
N void string_reader(void) { system(packet); }
N void variable_span_reader(size_t n) { char local[128]; memcpy(local, packet, n); system(local + 80); }
N void shifted_copy_reader(void) { char local[32]; memcpy(local, packet + 48, 32); system(local); }
N void string_case(char *s) { memcpy(packet_tail, s, 32); string_reader(); }
N void variable_case(char *s, size_t n) { memcpy(packet_tail, s, 32); variable_span_reader(n); }
N void shifted_case(char *s) { memcpy(packet_tail, s, 32); shifted_copy_reader(); }
N void branch_case(char *s, int flag) { if (flag) reader(); else memcpy(tracked, s, 64); }
N void conditional_before_case(char *s, int flag) { if (flag) reader(); memcpy(tracked, s, 64); }
N unsigned char observed;
N void load_reader(void) { char *p = refill(tracked); if (p) observed = (unsigned char)p[8]; }
N char *advance(char *p) { return p + 16; }
N char *producer(void) { return advance(memset(tracked, 0, 1)); }
N void advanced_reader(void) { char *p = producer(); system(p); }
N void advanced_plus_reader(void) { char *p = producer(); system(p + 8); }
N void load_case(char *s) { memcpy(tracked, s, 64); load_reader(); }
N void advanced_case(char *s) { memcpy(tracked, s, 64); advanced_reader(); }
N void advanced_plus_case(char *s) { memcpy(tracked, s, 64); advanced_plus_reader(); }
N void before_case(char *s) { reader(); memcpy(tracked, s, 64); }
N void after_case(char *s) { memcpy(tracked, s, 64); reader(); }
N void loop_case(char *s, int n) { for (int i = 0; i < n; i++) { reader(); memcpy(tracked, s, 64); } }
N void offset_case(char *s) { memcpy(tracked, s, 64); use_refill(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    switch (argc) {
    case 2: overlap_case(argv[1]); break;
    case 3: shallow_case(argv[1]); break;
    case 4: neighbour_case(argv[1]); break;
    case 5: before_case(argv[1]); break;
    case 6: after_case(argv[1]); break;
    case 7: offset_case(argv[1]); break;
    case 8: string_case(argv[1]); break;
    case 9: variable_case(argv[1], 112); break;
    case 10: shifted_case(argv[1]); break;
    case 11: branch_case(argv[1], argc & 1); break;
    case 12: conditional_before_case(argv[1], argc & 1); break;
    case 13: load_case(argv[1]); break;
    case 14: advanced_case(argv[1]); break;
    case 15: advanced_plus_case(argv[1]); break;
    default: loop_case(argv[1], argc); break;
    }
    return 0;
}
"""


def test_global_continuation_symbol_overlap_offsets_and_order(tmp_path, shared_bn):
    """Round-10 review C1, M1 and M2 on a real stripped build.

    C1: `.set packet_tail, packet + 64` gives one buffer two symbols, so a
    copy of `packet[0..112)` reads bytes a write to `packet_tail` put there
    while naming the other symbol. The access SPAN is what decides: a read
    that stops short of them, or one of a different object, is no
    continuation -- the bytes still live at the return end in the P2-11.1
    exit store, which cites the ENCLOSING object's reader `consumer`.
    M1: a pointer handed back by `fgets` carries its displacement in the SSA
    chain, not in the instruction -- `system(p + 8)` is a read at +8.
    M2: a call that runs BEFORE the only write carries nothing; a loop's
    back edge still does. Bytes still live at the return end in the P2-11.1
    exit `coarse_memory_store` at the write, citing `reader` -- never as a
    continuation at the earlier call.
    """
    src = tmp_path / "r3overlap.c"
    src.write_text(_OVERLAP_SRC)
    binary = tmp_path / "r3overlap"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # C1: the overlapping read of the enclosing symbol is a consumer
    result, leaves = gmc("overlap_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["consumer"], leaf
    assert leaf["references"][0]["access"] == "call_arg", leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and a read that stops short, or reads a different object, is not;
    # the exit store sweeps the enclosing `packet` and cites its reader
    for fn in ("shallow_case", "neighbour_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert store["global"]["address"] == _data_symbol_addr(binary, "packet_tail"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["consumer"], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # M1: the displacement through the returned destination
    result, leaves = gmc("offset_case")
    [leaf] = leaves
    assert leaf["references"][0]["offset"] == 8, leaf
    assert leaf["references"][0]["offset_known"] is True, leaf

    # M2: before the write is no continuation; after it, and round a loop, is
    result, leaves = gmc("before_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), store
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    for fn in ("after_case", "loop_case"):
        result, leaves = gmc(fn)
        assert [r["display_name"] for r in leaves[0]["references"]] == ["reader"], fn
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # Round-11 review F21: a read whose length is not stated cannot be shown
    # to stop short of the tainted bytes, so it blocks -- and says the
    # extent is what is unproven. One that starts INSIDE the enclosing
    # object and reaches them blocks on the interval, not on adjacency.
    for fn in ("string_case", "variable_case"):
        result, leaves = gmc(fn)
        [leaf] = leaves
        assert leaf["references"][0]["offset_known"] is False, (fn, leaf)
        assert leaf["reachability"]["cause"] == "object_extent_unproven", (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn
    result, leaves = gmc("shifted_case")
    [leaf] = leaves
    assert [r["display_name"] for r in leaf["references"]] == ["shifted_copy_reader"]

    # F22: a pointer-adjusting wrapper composes its displacement, and a
    # scalar load through a returned pointer reports the byte it reads.
    for fn, want in (("load_case", 8), ("advanced_case", 16),
                     ("advanced_plus_case", 24)):
        result, leaves = gmc(fn)
        row = max((r for lf in leaves for r in lf["references"]),
                  key=lambda r: r["offset"])
        assert row["offset"] == want, (fn, leaves)
        assert row["offset_known"] is True, (fn, leaves)

    # F23: mutually exclusive branches never carry those bytes to each other
    # -- the write still reaches the return, so the exit store cites `reader`
    for fn in ("branch_case", "conditional_before_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["reader"], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn


_VARIANTS_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char packet[256];
extern char mid1[240], mid2[224], mid3[208], tail[192];
__asm__(".globl mid1\n.set mid1,packet+16\n.type mid1,@object\n.size mid1,240\n"
        ".globl mid2\n.set mid2,packet+32\n.type mid2,@object\n.size mid2,224\n"
        ".globl mid3\n.set mid3,packet+48\n.type mid3,@object\n.size mid3,208\n"
        ".globl tail\n.set tail,packet+64\n.type tail,@object\n.size tail,192\n");
char init_packet[128] = "seed";
extern char init_tail[64];
__asm__(".globl init_tail\n.set init_tail,init_packet+64\n.type init_tail,@object\n.size init_tail,64\n");
char tracked[128] = "seed";
char clean[128];
volatile int selector;
N void base_reader(void) { char local[112]; memcpy(local, packet, 112); system(local + 80); }
N void init_reader(void) { char local[112]; memcpy(local, init_packet, 112); system(local + 80); }
N void reader(void) { system(tracked); }
N void different_reader(void) { system(clean); }
N char *refill(void) { return fgets(tracked, 2, stdin); }
N char *choose_return(char *p, int flag) { if (flag) return p + 8; return p + 24; }
N char *producer_choices(void) { return choose_return(refill(), selector); }
N void use_producer_choices(void) { char *p = producer_choices(); system(p); }
N void literal_phi(void) { char *p; if (selector) p = tracked + 8; else p = tracked + selector; system(p); }
N void separate_phi(void) { char *p; if (selector) p = tracked + 8; else p = tracked + 24; system(p); }
N void many_aliases(char *s) { memcpy(tail, s, 32); base_reader(); }
N void initialized_alias(char *s) { memcpy(init_tail, s, 32); init_reader(); }
N void earlier_branch_write(char *s, int flag) { if (flag) { memcpy(tracked, s, 16); reader(); } else { memcpy(tracked, s, 64); } }
N void later_branch_write(char *s, int flag) { if (flag) { memcpy(tracked, s, 16); } else { memcpy(tracked, s, 64); reader(); } }
N void disjoint(char *s, int flag) { if (flag) reader(); else memcpy(tracked, s, 64); }
N void loop_continue(char *s, int n) { for (int i = 0; i < n; i++) { if (i & 1) { reader(); continue; } memcpy(tracked, s, 64); } }
N void different_case(char *s) { memcpy(tracked, s, 64); different_reader(); }
N void producer_choices_case(char *s) { memcpy(tracked, s, 64); use_producer_choices(); }
N void literal_phi_case(char *s) { memcpy(tracked, s, 64); literal_phi(); }
N void separate_phi_case(char *s) { memcpy(tracked, s, 64); separate_phi(); }
int main(int argc, char **argv) {
    if (argc < 2) return 0;
    selector = argc;
    memset(packet, 'A', 64);
    switch (argc) {
    case 2: many_aliases(argv[1]); break;
    case 3: initialized_alias(argv[1]); break;
    case 4: earlier_branch_write(argv[1], argc); break;
    case 5: later_branch_write(argv[1], argc); break;
    case 6: disjoint(argv[1], argc); break;
    case 7: loop_continue(argv[1], argc); break;
    case 8: different_case(argv[1]); break;
    case 9: producer_choices_case(argv[1]); break;
    case 10: literal_phi_case(argv[1]); break;
    default: separate_phi_case(argv[1]); break;
    }
    return 0;
}
"""


def test_global_continuation_alias_chains_write_sites_and_ambiguity(tmp_path,
                                                                    shared_bn):
    """Round-12 review F24, F25, F26 and F27 on a real stripped build.

    F24: every tainting site is kept, so a mutually exclusive EARLIER write
    cannot hide the write on the reader's own branch.
    F25: four `.set` aliases name the inside of one object; the search walks
    the whole chain rather than a fixed couple of predecessors.
    F26: BN sizes `char init_packet[128] = "seed"` as `char[0x5]` -- a LOWER
    bound on the object, never proof of where it ends.
    F27: a displacement is exact only when every feasible path agrees.
    Exclusive branches and another object's reader are no continuation; the
    copy still live at the return is the P2-11.1 exit store citing `reader`."""
    src = tmp_path / "r3variants.c"
    src.write_text(_VARIANTS_SRC)
    binary = tmp_path / "r3variants"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F25 / F26: the enclosing object's reader is found through four aliases,
    # and an initialised prefix type does not disown the object's interior.
    for fn, who in (("many_aliases", "base_reader"),
                    ("initialized_alias", "init_reader")):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == who
                   for lf in leaves for r in lf["references"]), (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F24: a write on the reader's own branch blocks whichever branch it is
    for fn in ("earlier_branch_write", "later_branch_write", "loop_continue"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "reader"
                   for lf in leaves for r in lf["references"]), (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # ...and mutually exclusive branches, or another object, are no
    # continuation -- the live bytes end in the exit store naming `reader`
    for fn in ("disjoint", "different_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["reader"], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F27: ambiguous displacements are reported as unknown, never as 0
    for fn in ("literal_phi_case", "separate_phi_case", "producer_choices_case"):
        result, leaves = gmc(fn)
        rows = [r for lf in leaves for r in lf["references"]
                if r["access"] in ("call_arg", "load")]
        assert rows, (fn, leaves)
        assert all(r["offset_known"] is False for r in rows), (fn, rows)


_COUNTEREXAMPLE_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char initialized[128] = "seed";
extern char initialized_tail[64];
__asm__(".globl initialized_tail\n.set initialized_tail,initialized+64\n"
        ".type initialized_tail,@object\n.size initialized_tail,64\n");
N void interior_reader(void) { system(initialized_tail + 16); }
N void initialized_known(char *s) { memcpy(initialized, s, 112); interior_reader(); }
N void initialized_unknown(char *s, size_t n) { memcpy(initialized, s, n); interior_reader(); }
N void initialized_small(char *s) { memcpy(initialized, s, 16); interior_reader(); }
char large_packet[2097280];
extern char far_tail[64];
__asm__(".globl far_tail\n.set far_tail,large_packet+1048640\n"
        ".type far_tail,@object\n.size far_tail,64\n");
N void far_reader(void) {
    char *p = malloc(sizeof large_packet);
    memcpy(p, large_packet, sizeof large_packet);
    system(p + 1048640);
    free(p);
}
N void far_case(char *s) { memcpy(far_tail, s, 32); far_reader(); }
char tracked[128] = "seed";
volatile int selector;
N char *choose_formals(char *p, char *q, int n) { return n ? p : q; }
N char *formal_producer(void) { return choose_formals(tracked + 8, tracked + 24, selector); }
N void formal_consumer(void) { system(formal_producer()); }
N void formal_case(char *s) { memcpy(tracked, s, 64); formal_consumer(); }
char adjacent_a[64] = "safe";
char adjacent_b[64] = "seed";
N void adjacent_reader(void) { system(adjacent_a); }
N void adjacent_case(char *s) { memcpy(adjacent_b, s, 16); adjacent_reader(); }
N char *phi_producer(void) {
    char *p;
    if (selector) p = tracked + 8; else p = tracked + selector;
    return p;
}
N void phi_consumer(void) { char b[64]; memcpy(b, phi_producer(), sizeof b); system(b); }
N void advice_case(char *s) { memcpy(tracked, s, 64); phi_consumer(); }
int main(int argc, char **argv) { return argc > 100 ? argv[0][0] : 0; }
"""


def test_global_continuation_object_evidence_budgets_and_advice(tmp_path,
                                                                shared_bn):
    """Round-13 review F28, F29, F30, F31 and F32 on a real stripped build.

    F28: an inferred initialiser width is a LOWER bound -- a 112-byte copy
    into `char initialized[128] = "seed"` crosses its interior alias, and a
    runtime length stays unbounded rather than clipped to `char[0x5]`.
    F29: reaching the enclosing-object search window is a BOUND, disclosed
    as `object_search_budget`, never as "nothing encloses it".
    F30: a wrapper that may return either formal has no single displacement.
    F31: a type a HUMAN applied IS where the object ends.
    F32: the re-seed command's function and locator come from one reference,
    and it runs."""
    src = tmp_path / "r3counter.c"
    src.write_text(_COUNTEREXAMPLE_SRC)
    binary = tmp_path / "r3counter"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F28: both the known and the unknown copy reach the interior reader
    for fn in ("initialized_known", "initialized_unknown"):
        result, leaves = gmc(fn)
        [leaf] = leaves
        assert [r["display_name"] for r in leaf["references"]] == [
            "interior_reader"], (fn, leaf)
        assert leaf["references"][0]["offset"] == 80, (fn, leaf)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn
    result, leaves = gmc("initialized_small")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True

    # F29: the window is a bound, and reaching it blocks
    result, leaves = gmc("far_case")
    [leaf] = leaves
    assert leaf["reachability"]["cause"] == "object_search_budget", leaf
    assert "global_object_search" in result["stats"]["truncation_cause"], result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # F30: either formal may come back, so the displacement is unknown
    result, leaves = gmc("formal_case")
    row = next(r for lf in leaves for r in lf["references"]
               if r["display_name"] == "formal_consumer")
    assert row["offset_known"] is False, leaves

    # F31: an INFERRED width does not separate them; a DECLARED one does --
    # the image's own `st_size` (P1-05), and a type a user applies.
    result, leaves = gmc("adjacent_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    for name in ("adjacent_a", "adjacent_b"):
        applied = _bn_json(shared_bn, "-t", target, "data", "retype", name,
                           "char[64]")
        assert applied["results"][0]["status"] == "verified", applied
    result, leaves = gmc("adjacent_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True

    # F32: the advice names the function its locator belongs to, and runs
    result, leaves = gmc("advice_case")
    [leaf] = leaves
    rows = leaf["references"]
    assert rows[0]["seed"] is None, rows
    seeded = next(r for r in rows if r["seed"])
    # ...carrying this call's own routing, so the string RUNS where it was
    # produced instead of picking a bridge of its own (BN23-ADVICE-01).
    assert leaf["reseed"] == (
        f"`bn -i {shared_bn.instance_id} -t {shlex.quote(str(target))} "
        f"taint forward -f {seeded['address']} --source {seeded['seed']}`")
    ran = _bn_json(shared_bn, "-t", target, "taint", "forward",
                   "-f", seeded["address"], "--source", seeded["seed"])
    assert [s["sink"]["class"] for s in ran["reached_sinks"]] == ["command_injection"]


_CHALLENGE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define N __attribute__((noinline))
char chain[256] = "seed";
extern char alias16[240], alias32[224], alias64[192];
__asm__(".globl alias16\n.set alias16,chain+16\n.type alias16,@object\n.size alias16,240\n"
        ".globl alias32\n.set alias32,chain+32\n.type alias32,@object\n.size alias32,224\n"
        ".globl alias64\n.set alias64,chain+64\n.type alias64,@object\n.size alias64,192\n");
char form[128] = "seed";
char clean[128] = "safe";
volatile int selector;
N void chain_reader(void) { system(alias64 + 16); }
N int constant_only(void) { return 7; }
N void chain_known(char *s) { memcpy(chain, s, 112); chain_reader(); }
N void chain_unknown(char *s, size_t n) { memcpy(chain, s, n); chain_reader(); }
N void chain_short(char *s) { memcpy(chain, s, 16); chain_reader(); }
N void chain_strcpy(char *s) { strcpy(chain, s); chain_reader(); }
N void chain_sprintf(char *s) { sprintf(chain, "%s", s); chain_reader(); }
N void chain_no_reader(char *s, size_t n) { memcpy(chain, s, n); constant_only(); }
N char *mixed(char *p, int n) { return n ? p : form + 24; }
N char *producer_mixed(void) { return mixed(form + 8, selector); }
N void consume_mixed(void) { system(producer_mixed()); }
N void mixed_case(char *s) { memcpy(form, s, 64); consume_mixed(); }
N char *identity(char *p) { return p; }
N int ignore_identity(void) { identity(form); return 7; }
N void identity_case(char *s) { memcpy(form, s, 64); ignore_identity(); }
N void use_identity(void) { system(identity(form)); }
N void used_identity_case(char *s) { memcpy(form, s, 64); use_identity(); }
char window[2097408];
extern char exact[64];
__asm__(".globl exact\n.set exact,window+1048576\n.type exact,@object\n.size exact,64\n");
N void window_reader(void) { char *p = malloc(sizeof window); memcpy(p, window, sizeof window); system(p + 1048576); free(p); }
N void exact_case(char *s) { memcpy(exact, s, 32); window_reader(); }
N int exact_no_reader(char *s) { memcpy(exact, s, 32); return constant_only(); }
int main(int argc, char **argv) { return argc > 100 ? argv[0][0] : 0; }
"""


def test_global_continuation_alias_runs_joins_and_pointer_only_callees(
        tmp_path, shared_bn):
    """Round-14 review F33, F34, F35, F36 and F37 on a real stripped build.

    F33: three interior aliases do not stop the walk short of a reader past
    the last one, for a known, an unknown and a sink-firing copy alike.
    F34/F35: the alias search's own bounds are disclosed where a reader
    could hide behind them, and NOT where the callee names no global at all
    -- there the live copy is the P2-11.1 exit store citing the enclosing
    object's reader.
    F36: a body returning either a formal or the object by name has two
    feasible displacements, so it has none.
    F37: a callee that only hands the pointer back reads nothing; a caller
    that consumes the result is still the boundary. Where no call after the
    write is a continuation but the bytes stay live at the return and a
    reader elsewhere can reach them, the P2-11.1 exit store cites it."""
    src = tmp_path / "r3chain.c"
    src.write_text(_CHALLENGE_SRC)
    binary = tmp_path / "r3chain"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F33: every copy that can reach +80 finds the reader past three aliases
    for fn in ("chain_known", "chain_unknown", "chain_strcpy", "chain_sprintf"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "chain_reader" and r["offset"] == 80
                   for lf in leaves for r in lf["references"]), (fn, leaves)
    # ...one that stops short does not; one whose only call reads nothing is
    # no continuation, but its unbounded copy reaches +80 and stays live at
    # the return, so the exit store cites `chain_reader` there
    result, leaves = gmc("chain_short")
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    result, leaves = gmc("chain_no_reader")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "chain"), store
    assert [(r["display_name"], r["offset"]) for r in store["references"]] \
        == [("chain_reader", 80)], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # F34/F35: the search bound is disclosed where it can hide a reader...
    result, leaves = gmc("exact_case")
    assert [lf["reachability"]["cause"] for lf in leaves] == [
        "object_search_budget"], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # ...and not where the callee names no global at all: no bound and no
    # continuation, but `window_reader` reads the enclosing object, so the
    # copy still live at the return is the exit store naming it
    result, leaves = gmc("exact_no_reader")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "exact"), store
    assert [r["display_name"] for r in store["references"]] == ["window_reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # F36: {formal, object-by-name} is two displacements, so neither is exact
    result, leaves = gmc("mixed_case")
    row = next(r for lf in leaves for r in lf["references"]
               if r["display_name"] == "consume_mixed")
    assert row["offset_known"] is False, leaves

    # F37: a discarded pointer-only callee reads nothing; a used one blocks.
    # The discarded case's live bytes end in the exit store, which cites the
    # reader that consumes the result -- never the pointer-only callee.
    result, leaves = gmc("identity_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "form"), store
    assert [r["display_name"] for r in store["references"]] == ["use_identity"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    result, leaves = gmc("used_identity_case")
    assert any(r["display_name"] == "use_identity"
               for lf in leaves for r in lf["references"]), leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


_R16_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define N __attribute__((noinline))
volatile int selector;
volatile unsigned char seen;
char chain[256] __attribute__((section(".chain"))) = "seed";
extern char alias16[240], alias64[192];
extern unsigned char alias32;
__asm__(".globl alias16\n.set alias16,chain+16\n.type alias16,@object\n.size alias16,240\n"
        ".globl alias32\n.set alias32,chain+32\n.type alias32,@object\n.size alias32,1\n"
        ".globl alias64\n.set alias64,chain+64\n.type alias64,@object\n.size alias64,192\n");
N void type_alias(void) { seen = alias32; }
N void chain_reader(void) { system(alias64 + 16); }
N void typed_known(char *s) { memcpy(chain, s, 112); chain_reader(); }
N void typed_unknown(char *s, size_t n) { memcpy(chain, s, n); chain_reader(); }
N void typed_strcpy(char *s) { strcpy(chain, s); chain_reader(); }
N void typed_sprintf(char *s) { sprintf(chain, "%s", s); chain_reader(); }
N void typed_short(char *s) { memcpy(chain, s, 16); chain_reader(); }
char tracked[128] __attribute__((section(".track"))) = "seed";
N char *conditional(char *p, int n) { if (n) return p + 1; system(p); return p + 2; }
N char *conditional_reverse(char *p, int n) { if (n) { system(p); return p + 2; } return p + 1; }
N char *early_likely(char *p, int n) { if (__builtin_expect(n != 0, 0)) return p; system(p); return NULL; }
N char *identity(char *p) { return p; }
N int ignore_conditional(void) { conditional(tracked, selector); return 7; }
N int ignore_reverse(void) { conditional_reverse(tracked, selector); return 7; }
N int ignore_likely(void) { early_likely(tracked, selector); return 7; }
N int ignore_identity(void) { identity(tracked); return 7; }
N char *mixed3(char *p, int n) { if (n == 1) return p; if (n == 2) return tracked + 24; return tracked + 40; }
N char *mixed_dynamic(char *p, int n) { return n ? p : tracked + selector; }
N char *mixed_agree(char *p, int n) { return n ? p : tracked + 8; }
N char *producer3(void) { return mixed3(tracked + 8, selector); }
N char *producer_dynamic(void) { return mixed_dynamic(tracked + 8, selector); }
N char *producer_agree(void) { return mixed_agree(tracked + 8, selector); }
N void consume3(void) { system(producer3()); }
N void consume_dynamic(void) { system(producer_dynamic()); }
N void consume_agree(void) { system(producer_agree()); }
#define C(name,fn) N void name(char *s) { memcpy(tracked, s, 64); fn(); }
C(conditional_case, ignore_conditional)
C(reverse_case, ignore_reverse)
C(likely_case, ignore_likely)
C(identity_case, ignore_identity)
C(mixed3_case, consume3)
C(dynamic_case, consume_dynamic)
C(agree_case, consume_agree)
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1"], ids=["O0", "O1"])
def test_global_continuation_auto_types_branches_and_return_routes(
        tmp_path, shared_bn, opt):
    """Round-16 review F33, F38 and F36 on real stripped builds.

    F33: a type BN INFERRED from an access (`uint8_t` at `chain + 32`) is
    not an object boundary, so an unknown-length or sink-firing copy still
    reaches the reader past it.
    F38: "writes only through this parameter" is a verdict over EVERY
    feasible use, not the first one BN happens to list. A pointer-only
    callee is no continuation; the copy it leaves live at the return is the
    P2-11.1 exit store, citing a wrapper whose callee can `system` it.
    F36: a rooted return route that is present but ambiguous makes the
    displacement unknown, however exact the formal route is."""
    src = tmp_path / f"r3r16{opt[-1]}.c"
    src.write_text(_R16_SRC)
    binary = tmp_path / f"r3r16{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-fno-inline",
                           "-fno-optimize-sibling-calls", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F33: the auto-typed alias does not end the object
    for fn in ("typed_known", "typed_unknown", "typed_strcpy", "typed_sprintf"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "chain_reader" and r["offset"] == 80
                   for lf in leaves for r in lf["references"]), (fn, leaves)
    result, leaves = gmc("typed_short")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True

    # F38: a sink on the other arm outranks a pointer-return arm
    for fn in ("conditional_case", "reverse_case", "likely_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["diagnostics"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn
    result, leaves = gmc("identity_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), store
    [row] = store["references"]
    assert row["display_name"] in ("ignore_conditional", "ignore_reverse",
                                   "ignore_likely"), store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # F36: three arms, and a dynamic arm, are both unknown; agreement is not
    for fn in ("mixed3_case", "dynamic_case"):
        row = next(r for lf in gmc(fn)[1] for r in lf["references"]
                   if r["access"] == "call_arg")
        assert row["offset_known"] is False, fn
    row = next(r for lf in gmc("agree_case")[1] for r in lf["references"]
               if r["access"] == "call_arg")
    assert (row["offset"], row["offset_known"]) == (8, True), row


def _alias_count_source(count: int) -> str:
    decls = ", ".join(f"a{i}[16]" % () for i in range(count))
    sets = "\n".join(
        f'        ".globl a{i}\\n.set a{i},chain+{16 * (i + 1)}\\n'
        f'.type a{i},@object\\n.size a{i},16\\n"' for i in range(count))
    return f"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define N __attribute__((noinline))
char chain[4096] __attribute__((section(".chain"))) = "seed";
extern char {decls};
__asm__(
{sets});
char after[64] __attribute__((section(".after"))) = "safe";
N void far_reader(void) {{ system(chain + 1048); }}
N int quiet(void) {{ return 7; }}
N void known_case(char *s) {{ memcpy(chain, s, 2048); far_reader(); }}
N void unknown_case(char *s, size_t k) {{ memcpy(chain, s, k); far_reader(); }}
N void strcpy_case(char *s) {{ strcpy(chain, s); far_reader(); }}
N void sprintf_case(char *s) {{ sprintf(chain, "%s", s); far_reader(); }}
N void short_case(char *s) {{ memcpy(chain, s, 16); far_reader(); }}
N void no_reader_case(char *s, size_t k) {{ memcpy(chain, s, k); quiet(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("count", [64, 65], ids=["at-the-limit", "one-past"])
def test_global_continuation_survives_any_interior_alias_count(tmp_path,
                                                               shared_bn, count):
    """Round-17 review F39 on a real stripped build.

    Running out of ALIASES bounds the enumeration; it does not establish
    where the object ends. Coverage continues to the credible boundary, the
    bound is named, and the short-write control stays silent. The no-reader
    control is no continuation, but its unbounded copy is still live at the
    return and reaches `far_reader`'s +1048, so it is the P2-11.1 exit store.
    """
    src = tmp_path / f"r3alias{count}.c"
    src.write_text(_alias_count_source(count))
    binary = tmp_path / f"r3alias{count}"
    _compile(src, binary, ["-fno-builtin", "-fno-inline", "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("known_case", "unknown_case", "strcpy_case", "sprintf_case"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "far_reader" and r["offset"] == 1048
                   for lf in leaves for r in lf["references"]), (fn, leaves)
        # A run that reaches a sink carries no all-clear gate at all.
        assert (result.get("diagnostics") or {}).get(
            "safe_to_report_all_clear") is not True, fn
    if count > 64:
        # The image DECLARES `chain`'s size, so running out of aliases bounds
        # nothing: the declaration says where the object ends (P1-05). The
        # undeclared shape keeps its `object_search_budget` in the mocked suite.
        _result, leaves = gmc("unknown_case")
        assert "object_search_budget" not in leaves[0]["reachability"]["causes"], leaves
    result, leaves = gmc("short_case")
    assert result["leaves"] == [], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    result, leaves = gmc("no_reader_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert store["global"]["address"] == _data_symbol_addr(binary, "chain"), store
    assert [(r["display_name"], r["offset"]) for r in store["references"]] \
        == [("far_reader", 1048)], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False



_ATTRIBUTION_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char chain[256] __attribute__((section(".chain"))) = "seed";
extern char mid[192];
__asm__(".globl mid\n.set mid,chain+64\n.type mid,@object\n.size mid,192\n");
N void reader(void) { system(chain + 80); }
N void long_first(void) { char *a = getenv("A"); char *b = getenv("B"); memcpy(chain, a, 112); memcpy(chain, b, 8); reader(); }
N void short_first(void) { char *b = getenv("B"); char *a = getenv("A"); memcpy(chain, a, 112); memcpy(chain, b, 8); reader(); }
N void only_short(void) { char *b = getenv("B"); memcpy(chain, b, 8); reader(); }
N void only_long(void) { char *a = getenv("A"); memcpy(chain, a, 112); reader(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1"], ids=["O0", "O1"])
def test_global_continuation_attribution_is_source_order_invariant(
        tmp_path, shared_bn, opt):
    """Round-17 review F40 on real stripped builds.

    Each per-source run starts from its own object evidence: the extent a
    112-byte write established, and everything derived from it, must not
    survive into the 8-byte source's run. The same two sources therefore
    get the same answer whichever order they are enumerated in, and each
    agrees with running that source alone.
    """
    src = tmp_path / f"r3attrib{opt[-1]}.c"
    src.write_text(_ATTRIBUTION_SRC)
    binary = tmp_path / f"r3attrib{opt[-1]}"
    _compile(src, binary, [opt, "-fno-builtin", "-fno-inline",
                           "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def counts(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "ret:getenv")
        per = result.get("by_source") or {}
        return sorted(
            len([lf for lf in (rows.get("leaves") or [])
                 if lf["kind"] == "global_memory_continuation"])
            for rows in per.values() if isinstance(rows, dict))

    assert counts("long_first") == counts("short_first")
    alone = [counts("only_long"), counts("only_short")]
    assert alone[1] == [] or sum(alone[1]) == 0, alone


_STORE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
char chain[256] __attribute__((section(".chain"))) = "seed";
extern char mid[192];
__asm__(".globl mid\n.set mid,chain+64\n.type mid,@object\n.size mid,192\n");
char other[128] __attribute__((section(".other"))) = "safe";
N void reader(void) { system(chain + 80); }
N void other_reader(void) { system(other); }
N int quiet(void) { return 7; }
N void inlined_case(char *s) { memcpy(chain, s, 112); reader(); }
N void inlined_short_case(char *s) { memcpy(chain, s, 16); reader(); }
N void interior_case(char *s) { memcpy(chain + 64, s, 32); reader(); }
N void multi_store_case(char *s) { memcpy(chain, s, 48); memcpy(chain + 48, s, 48); reader(); }
N void no_reader_case(char *s) { memcpy(chain, s, 112); quiet(); }
N void other_case(char *s) { memcpy(chain, s, 112); other_reader(); }
N void sink_copy_case(char *s) { strcpy(chain, s); reader(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_global_continuation_direct_stores_match_modeled_copies(tmp_path,
                                                                shared_bn, opt):
    """A tainted DIRECT store into a known writable global is global
    provenance, not a coarse-memory hole.

    An inlined `memcpy` is a run of stores: at -O1 and -O2 the compiler
    emits one, and the continuation it produces has to be the one the
    modeled call produces at -O0. The span those stores wrote is what
    decides, so a short run stays silent. A reader of a DIFFERENT object, or
    no reader at all, is no continuation at any level; the copy -- stores or
    modeled call alike (P2-11.1) -- is then still live at the return, so
    the blocking exit `coarse_memory_store` names `reader`. The copy note
    and the sink of a `strcpy` are reported together.
    """
    src = tmp_path / f"r3store{opt[-1]}.c"
    src.write_text(_STORE_SRC)
    binary = tmp_path / f"r3store{opt[-1]}"
    _compile(src, binary, [opt, "-fno-builtin", "-fno-inline",
                           "-fno-stack-protector", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("inlined_case", "interior_case", "multi_store_case"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "reader"
                   for lf in leaves for r in lf["references"]), (fn, result)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn
        assert not [lf for lf in result["leaves"]
                    if lf["kind"] == "coarse_memory_store"], (fn, result["leaves"])
    for fn in ("no_reader_case", "other_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        stores = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
        # P1-04 / P2-11.1: the inlined run of DIRECT stores and the modeled
        # copy are both still live when the case returns, and `reader` --
        # never walked after it -- reads them, so the write is the blocking
        # frontier at every level.
        assert [r["display_name"] for lf in stores for r in lf["references"]] \
            == ["reader"], (fn, stores)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # the sink and the copy into the destination are reported together
    result, leaves = gmc("sink_copy_case")
    assert [s["sink"]["class"] for s in result["reached_sinks"]], result
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), leaves
    assert any("copied into the destination" in a
               for a in result["assumptions"]), result["assumptions"]


_TAIL_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define N __attribute__((noinline,noclone))
char shared[256] __attribute__((section(".shared"))) = "seed";
char plain[256] __attribute__((section(".plain"))) = "safe";
volatile unsigned char seen;
N int invoke(char *p) { int r = puts("marker"); system(p); return r + 1; }
N int relay_reader(void) { return invoke(shared); }
N void tail_case(char *s) { memcpy(shared, s, 112); relay_reader(); }
N void tail_store_case(char *s) { shared[0] = s[0]; relay_reader(); }
N int plt_only(void) { return puts("hello"); }
N void plt_case(char *s) { memcpy(shared, s, 112); plt_only(); }
N void quiet(void) { seen = 7; }
N void no_reader(char *s) { memcpy(shared, s, 112); quiet(); }
N int constant(void) { return 7; }
N int cycle_b(int);
N int cycle_a(int n) { return n ? cycle_b(n - 1) : constant(); }
N int cycle_b(int n) { return n ? cycle_a(n - 1) : constant(); }
N int recursion_case(char *s) { memcpy(shared, s, 64); return cycle_a(3); }
int main(int c, char **v) { if (c > 1) tail_case(v[1]); return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_global_continuation_veneers_quiet_callees_and_scalar_recursion(
        tmp_path, shared_bn, opt):
    """Round-18 review F41, F42 and F43 on real stripped builds.

    F41: a veneer that materialises the tainted global and tail-jumps into
    an internal consumer IS the boundary -- resolving it to the target
    dropped the argument binding and cleared the run.
    F42: a callee whose only global lives elsewhere is not a continuation,
    whatever the alias search did not reach; the copy it leaves live at the
    return is the P2-11.1 exit store citing `relay_reader`, never the callee.
    F43: scalar-only mutual recursion is not a pointer question.
    """
    src = tmp_path / f"r3tail{opt[-1]}.c"
    src.write_text(_TAIL_SRC)
    binary = tmp_path / f"r3tail{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-optimize-sibling-calls", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F41: the modeled copy and the direct store both reach the veneer
    for fn in ("tail_case", "tail_store_case"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "relay_reader"
                   for lf in leaves for r in lf["references"]), (fn, result)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # ...while a thunk-shaped callee that reads nothing of it is no
    # continuation; the live copy ends in the exit store naming the veneer
    for fn in ("plt_case", "no_reader", "recursion_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert store["global"]["address"] == _data_symbol_addr(binary, "shared"), \
            (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["relay_reader"], \
            (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F43: no pointer-depth uncertainty from a scalar cycle
    result, _leaves_ = gmc("recursion_case")
    assert "pointer_return_depth" not in result["stats"]["truncation_cause"]


def _alias_quiet_source(count: int) -> str:
    sets = "".join(
        f'extern unsigned char a{j};\n'
        f'__asm__(".globl a{j}\\n.set a{j},chain+{j * 16}\\n'
        f'.type a{j},@object\\n.size a{j},1\\n");\n' for j in range(1, count + 1))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char chain[8192] __attribute__((section(".chain"))) = "seed";
volatile unsigned char seen;
{sets}N void reader(void) {{ system((char *)&a{count} + 8); }}
N void quiet(void) {{ seen = 1; }}
N void unknown_case(char *s, size_t n) {{ memcpy(chain, s, n); reader(); }}
N void quiet_case(char *s, size_t n) {{ memcpy(chain, s, n); quiet(); }}
N void short_case(char *s) {{ memcpy(chain, s, 8); reader(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("count", [63, 65], ids=["under", "over"])
def test_an_exhausted_alias_search_is_reader_specific(tmp_path, shared_bn,
                                                      count):
    """Round-18 review F42: the alias bound is not an unconditional leaf.

    A callee whose only global is a byte in ANOTHER section reads nothing of
    the written object, at 63 aliases and at 65 alike: no continuation is
    raised at it. The runtime-length copy is still live at the return, so
    the P2-11.1 exit store blocks -- naming the real reader, never the
    quiet callee -- and the real reader past them is still disclosed.
    """
    src = tmp_path / f"r3aliasq{count}.c"
    src.write_text(_alias_quiet_source(count))
    binary = tmp_path / f"r3aliasq{count}"
    _compile(src, binary, ["-O2", "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-optimize-sibling-calls", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("quiet_case")
    assert leaves == [], (count, leaves)
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], \
        (count, store)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    result, leaves = gmc("short_case")
    assert leaves == [], (count, leaves)
    result, leaves = gmc("unknown_case")
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), (count, leaves)


_MULTIHOP_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
N void reader(void) { system(tracked); }
N void relay1(void) { reader(); }
N void relay2(void) { relay1(); }
N void unrelated(void) { system(clean); }
N void direct(char *s) { memcpy(tracked, s, 112); reader(); }
N void onehop(char *s) { memcpy(tracked, s, 112); relay1(); }
N void twohop(char *s) { memcpy(tracked, s, 112); relay2(); }
N void store_copy(char *s) { tracked[0] = s[0]; relay2(); }
N void sink_copy(char *s) { strcpy(tracked, s); relay2(); }
N void unrelated_case(char *s) { memcpy(tracked, s, 112); unrelated(); }
N void plt_case(char *s) { memcpy(tracked, s, 112); abort(); }
int main(int c, char **v) { if (c > 1) twohop(v[1]); return 0; }
"""

_REMAP_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
N void write_read(char *dst, char *command) { memset(dst, 0, 4); system(command); }
N void swap(char *a, char *b) { write_read(b, a); }
N void bound_clean(void) { swap(clean, tracked); }
N void write_only_case(char *s) { memcpy(tracked, s, 112); bound_clean(); }
N void consumer(void) { swap(tracked, clean); }
N void read_case(char *s) { memcpy(tracked, s, 112); consumer(); }
int main(void) { return 0; }
"""

_INDIRECT_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
void (*volatile hook)(void);
N void reader(void) { system(tracked); }
N void quiet(void) { system(clean); }
N void store_case(char *s) { tracked[0] = s[0]; hook(); }
N void copy_case(char *s) { memcpy(tracked, s, 112); hook(); }
int main(void) { hook = reader; return 0; }
"""


def _gmc_leaves(result):
    return [lf for lf in result["leaves"]
            if lf["kind"] == "global_memory_continuation"]


def _build_case(tmp_path, shared_bn, stem, source, opt):
    src = tmp_path / f"{stem}{opt[-1]}.c"
    src.write_text(source)
    binary = tmp_path / f"{stem}{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    return shared_bn.load(binary, copy=False)


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_multi_hop_veneer_chain_keeps_the_body_that_reads(tmp_path,
                                                            shared_bn, opt):
    """Round-19 review F44: `relay2 -> relay1 -> reader -> system` resolves
    to an IMPORT, and answering with that end point discarded `reader` --
    the one body that reads the tainted object -- so the run cleared.

    A chain that reads ANOTHER object raises no continuation at its call;
    the copy is still live at the return, so the P2-11.1 exit store blocks
    naming `reader`, never `unrelated`. A function that never returns
    (`abort()`) has no exit for the tainted bytes to outlive."""
    target = _build_case(tmp_path, shared_bn, "r3hop", _MULTIHOP_SRC, opt)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, _gmc_leaves(result)

    for fn in ("direct", "onehop", "twohop", "store_copy", "sink_copy"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "reader"
                   for lf in leaves for r in lf["references"]), (fn, result)
        assert (result.get("diagnostics") or {}).get(
            "safe_to_report_all_clear") is not True, fn

    # ...and `strcpy` keeps its own sink beside the continuation
    result, _ = gmc("sink_copy")
    assert [s["sink"]["class"] for s in result["reached_sinks"]] == \
        ["overflow_unbounded"], result["reached_sinks"]

    # A chain that reads ANOTHER object is not a continuation; the exit
    # store names the real reader of `tracked`, not the chain's body
    result, leaves = gmc("unrelated_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and a plain import that never returns stays silent
    result, leaves = gmc("plt_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_an_argument_swapping_veneer_binds_by_its_own_signature(tmp_path,
                                                                shared_bn, opt):
    """Round-19 review F47: `swap(a, b)` tail-calls `write_read(b, a)`, so
    the call's argument indices mean what SWAP's signature says. Classifying
    them against the resolved body's unchanged index read parameter 1 of
    `write_read` -- the command -- and a write-only global became a
    blocking consumer at the call. The copy is still live at the return,
    so the P2-11.1 exit store blocks -- naming the real reader `consumer`,
    never the write-only `bound_clean`."""
    target = _build_case(tmp_path, shared_bn, "r3remap", _REMAP_SRC, opt)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, _gmc_leaves(result)

    # tracked reaches only the memset DESTINATION; `clean` is what is read
    result, leaves = gmc("write_only_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["consumer"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # the mirror image -- the swap hands TRACKED to the command slot
    result, leaves = gmc("read_case")
    assert any(r["display_name"] == "consumer"
               for lf in leaves for r in lf["references"]), result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_register_indirect_tail_jump_is_a_call_boundary(tmp_path,
                                                          shared_bn, opt):
    """Round-19 review F45: an optimised `jmp reg` is where control leaves
    the function, but BN lifts it as a JUMP rather than a call, so the
    tainted global crossed it with nothing disclosed."""
    target = _build_case(tmp_path, shared_bn, "r3ind", _INDIRECT_SRC, opt)

    def gmc(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, _gmc_leaves(result)

    for fn in ("store_case", "copy_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result)
        assert any(lf["reachability"]["cause"] == "unresolved_call_target"
                   for lf in leaves), (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # the same boundary takes a pin: naming the real target REFINES it...
    jump_at = int(gmc("store_case")[1][0]["address"], 16)
    pin = tmp_path / f"r3pin{opt[-1]}.json"
    pin.write_text(json.dumps({hex(jump_at): ["reader"]}))
    result, leaves = gmc("store_case", "--resolve-map", str(pin))
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # ...and a target that reads ANOTHER object removes the boundary at the
    # jump. The tainted bytes -- the DIRECT store's and the modeled copy's
    # alike -- are still live when the function returns and `reader` --
    # reached here by nothing the walk follows -- reads them, so each run
    # ends in the blocking P2-11.1 exit store naming `reader`, never `quiet`.
    pin.write_text(json.dumps({hex(jump_at): ["quiet"]}))
    result, leaves = gmc("store_case", "--resolve-map", str(pin))
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    copy_jump = int(gmc("copy_case")[1][0]["address"], 16)
    pin.write_text(json.dumps({hex(copy_jump): ["quiet"]}))
    result, leaves = gmc("copy_case", "--resolve-map", str(pin))
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


def _section_source(count: int) -> str:
    sets = "".join(
        f'extern unsigned char a{j};\n'
        f'__asm__(".globl a{j}\\n.set a{j},chain+{j * 16}\\n'
        f'.type a{j},@object\\n.size a{j},1\\n");\n' for j in range(1, count + 1))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char chain[8192] __attribute__((section(".chain"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
{sets}N void reader(void) {{ system((char *)&a{count} + 8); }}
N void unrelated(void) {{ system(clean); }}
N void bounded_case(char *s, size_t n) {{ memcpy(chain, s, (n & 1023) + 1); reader(); }}
N void unrelated_case(char *s, size_t n) {{ memcpy(chain, s, (n & 1023) + 1); unrelated(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("count", [63, 65], ids=["under", "over"])
def test_an_exhausted_alias_search_stops_at_the_section_boundary(tmp_path,
                                                                 shared_bn,
                                                                 count):
    """Round-19 review F46: a section boundary is a hard object boundary.

    Past the alias limit the forward window ran off the end of `.chain` and
    attributed the FOLLOWING `.clean` global to the written object, so a
    function reading only `clean` became a blocking continuation. The copy
    into `chain` is still live at the return, so the P2-11.1 exit store
    blocks -- naming the reader INSIDE `chain`, never `unrelated`."""
    src = tmp_path / f"r3sec{count}.c"
    src.write_text(_section_source(count))
    binary = tmp_path / f"r3sec{count}"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, _gmc_leaves(result)

    result, leaves = gmc("unrelated_case")
    assert leaves == [], (count, leaves)
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], \
        (count, store)
    assert all(0 <= r["offset"] < 8192 for r in store["references"]), store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # the real reader inside the object is disclosed, at an offset within it
    result, leaves = gmc("bounded_case")
    offsets = [r["offset"] for lf in leaves for r in lf["references"]
               if r["display_name"] == "reader"]
    assert offsets and all(0 <= o < 8192 for o in offsets), (count, leaves)


def _core2_source(relay_hops: int = 32, perm_hops: int = 24) -> str:
    """A chain fixture: long veneer relays, an argument-dropping wrapper, a
    long write-only permutation, and a callee that tail-jumps indirectly."""
    relays = "".join(f"N void relay{i}(void) {{ relay{i - 1}(); }}\n"
                     for i in range(1, relay_hops + 1))
    perm = "".join(f"N void w{i}(char *a, char *b) {{ w{i - 1}(b, a); }}\n"
                   for i in range(1, perm_hops + 1))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
void (*volatile hook)(void);
N void relay0(void) {{ system(tracked); }}
{relays}N void w0(char *a, char *b) {{ memset(a, 0, 4); system(b); }}
{perm}N void write_read(char *dst, char *command) {{ memset(dst, 0, 4); system(command); }}
N void drop(char *a, char *b) {{ (void)a; write_read(b, clean); }}
N void dropbind(void) {{ drop(tracked, clean); }}
N void dispatch(void) {{ hook(); }}
N void quiet(void) {{ system(clean); }}
N void copy7(char *s) {{ memcpy(tracked, s, 112); relay7(); }}
N void copy8(char *s) {{ memcpy(tracked, s, 112); relay8(); }}
N void copy12(char *s) {{ memcpy(tracked, s, 112); relay12(); }}
N void copy32(char *s) {{ memcpy(tracked, s, 112); relay32(); }}
N void store12(char *s) {{ tracked[0] = s[0]; relay12(); }}
N void material12(char *s) {{ char *p = tracked; memcpy(p, s, 112); relay12(); }}
N void sink12(char *s) {{ strcpy(tracked, s); relay12(); }}
N void dropcase(char *s) {{ memcpy(tracked, s, 112); dropbind(); }}
N void wcase8(char *s) {{ memcpy(tracked, s, 112); w8(tracked, clean); }}
N void wcase9(char *s) {{ memcpy(tracked, s, 112); w9(clean, tracked); }}
N void wcase24(char *s) {{ memcpy(tracked, s, 112); w24(tracked, clean); }}
N void copy_nested(char *s) {{ memcpy(tracked, s, 112); dispatch(); }}
N void store_nested(char *s) {{ tracked[0] = s[0]; dispatch(); }}
N void sink_nested(char *s) {{ strcpy(tracked, s); dispatch(); }}
int main(int c, char **v) {{ hook = relay0; if (c > 1) copy12(v[1]); return 0; }}
"""


@pytest.fixture(scope="module")
def _core2_build(tmp_path_factory):
    """One build of the chain fixture per optimisation level, shared by the
    round-20 tests (each build is a few seconds of compile + analysis)."""
    root = tmp_path_factory.mktemp("r4core")
    out = {}
    src = root / "core2.c"
    src.write_text(_core2_source())
    for opt in ("-O0", "-O1", "-O2"):
        binary = root / f"core2{opt[-1]}"
        _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
        out[opt] = binary
    return out


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_long_veneer_chain_is_followed_to_the_reader(shared_bn, _core2_build,
                                                       opt):
    """Round-20 review F48: the chain walk stopped after eight bodies and
    went quiet, so a twelve-hop relay to `system(tracked)` cleared."""
    target = shared_bn.load(_core2_build[opt])

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", "--max-depth", "64")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("copy7", "copy8", "copy12", "copy32", "store12", "material12"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "relay0"
                   for lf in leaves for r in lf["references"]), (fn, result)
        assert (result.get("diagnostics") or {}).get(
            "safe_to_report_all_clear") is not True, fn

    # the sink-firing copy keeps BOTH its sink and the continuation
    result, leaves = gmc("sink12")
    assert [s["sink"]["class"] for s in result["reached_sinks"]] == \
        ["overflow_unbounded"], result["reached_sinks"]
    assert any(r["display_name"] == "relay0"
               for lf in leaves for r in lf["references"]), leaves


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_an_indirect_tail_jump_inside_a_callee_is_a_boundary(tmp_path,
                                                             shared_bn,
                                                             _core2_build, opt):
    """Round-20 review F49: the jump is one level down, inside `dispatch()`,
    where the reachability sweep walked past it as if the body were a leaf."""
    target = shared_bn.load(_core2_build[opt])

    def gmc(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("copy_nested", "store_nested", "sink_nested"):
        result, leaves = gmc(fn)
        assert any(lf["reachability"]["cause"] == "unresolved_call_target"
                   for lf in leaves), (fn, result)
        assert (result.get("diagnostics") or {}).get(
            "safe_to_report_all_clear") is not True, fn

    # the nested site takes a pin exactly like a call: the real target
    # refines the leaf, a target that reads another object removes the
    # boundary -- the copy is then still live at the return, so the run
    # ends in the P2-11.1 exit store naming `relay0`, never `quiet`
    il = shared_bn.run("-t", target, "il", "dispatch", "--view", "mlil",
                       "--ssa")
    assert il.returncode == 0, il.stderr
    site = next(int(line.split()[0], 16) for line in il.stdout.splitlines()
                if "(" in line and line.strip().startswith("0x"))
    pin = tmp_path / f"r4pin{opt[-1]}.json"
    pin.write_text(json.dumps({hex(site): ["relay0"]}))
    result, leaves = gmc("copy_nested", "--resolve-map", str(pin))
    assert any(r["display_name"] == "relay0"
               for lf in leaves for r in lf["references"]), result
    pin.write_text(json.dumps({hex(site): ["quiet"]}))
    result, leaves = gmc("copy_nested", "--resolve-map", str(pin))
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["relay0"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_dropped_or_permuted_write_only_argument_is_not_a_consumer(
        shared_bn, _core2_build, opt):
    """Round-20 review F50 and F51: `drop(a, b)` ignores `a` entirely, and
    the permutation chains hand the tracked object only to a memset
    destination -- neither reads it, at any chain length."""
    target = shared_bn.load(_core2_build[opt])

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", "--max-depth", "64")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("dropcase", "wcase8", "wcase9", "wcase24"):
        _result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)


def _writer_source(count: int) -> str:
    sets = "".join(
        f'extern unsigned char a{j};\n'
        f'__asm__(".globl a{j}\\n.set a{j},chain+{j * 16}\\n'
        f'.type a{j},@object\\n.size a{j},1\\n");\n' for j in range(1, count + 1))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char chain[8192] __attribute__((section(".chain"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
{sets}N void reader(void) {{ system((char *)&a{count} + 8); }}
N void writer(void) {{ memset(chain, 0, 64); }}
N void unrelated(void) {{ system(clean); }}
N void reader_case(char *s, size_t n) {{ memcpy(chain, s, (n & 1023) + 1); reader(); }}
N void writer_case(char *s, size_t n) {{ memcpy(chain, s, (n & 1023) + 1); writer(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("count", [63, 65], ids=["under", "over"])
def test_an_exhausted_alias_search_spares_a_write_only_callee(tmp_path,
                                                              shared_bn, count):
    """Round-20 review F52: `writer()` only runs `memset(chain, 0, 64)`.

    Past the alias limit the fallback asked whether the callee TOUCHES this
    object's storage, which a write does, so the bound alone turned a
    write-only callee into a blocking consumer. Its 64-byte reset does not
    cover the copy, so the bytes are still live at the return: the P2-11.1
    exit store blocks naming `reader`, never the write-only `writer`."""
    src = tmp_path / f"r4w{count}.c"
    src.write_text(_writer_source(count))
    binary = tmp_path / f"r4w{count}"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("writer_case")
    assert leaves == [], (count, leaves)
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == ["reader"], \
        (count, store)
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # the genuine interior reader past the same aliases still blocks
    _result, leaves = gmc("reader_case")
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), (count, leaves)


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_every_command_a_continuation_prints_actually_runs(tmp_path, shared_bn,
                                                           opt):
    """Round-20 review F53/F54/F55: advice is only advice if it RUNS.

    The reseed locator was advertised for constant-address operands that no
    `--source` can resolve, the xrefs claim named one address and promised
    every reference, and the remediation named a `bn retype` command that
    does not exist."""
    src = tmp_path / f"r4advice{opt[-1]}.c"
    src.write_text(_ADVICE_SRC)
    binary = tmp_path / f"r4advice{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    ran = 0
    for fn in ("interior_case", "extent_case"):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        texts = [lf.get("detail") for lf in result["leaves"]]
        texts += [lf.get("reseed") for lf in result["leaves"]]
        texts.append((result.get("diagnostics") or {}).get("next_action"))
        seen: set[str] = set()
        for text in texts:
            for cmd in re.findall(r"`(bn [^`]+)`", text or ""):
                if cmd in seen:
                    continue
                seen.add(cmd)
                argv = shlex.split(cmd)
                assert argv[0] == "bn"
                res = shared_bn.run(*argv[1:])
                assert res.returncode == 0, (cmd, res.returncode, res.stderr)
                ran += 1
    assert ran >= 4, ran


_ADVICE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char chain[8192] __attribute__((section(".chain"))) = "seed";
char tail[256] __attribute__((section(".chain"))) = "next";
N void interior_reader(void) { system(chain + 0x50); }
N void interior_case(char *s) { memcpy(chain, s, 112); interior_reader(); }
N void extent_reader(void) { system(chain + 0x50); }
N void extent_case(char *s) { memcpy(chain, s, 112); extent_reader(); }
int main(void) { return 0; }
"""


_CHILD_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
volatile int status_src;
N void reader(void) { system(tracked); }
N int write_child(char *s) { memcpy(tracked, s, 112); return status_src; }
N int store_child(char *s) { tracked[0] = s[0]; return status_src; }
N int other_child(char *s) { memcpy(clean, s, 112); return status_src; }
N int child_case(char *s) { int st = write_child(s); reader(); return st; }
N int store_case(char *s) { int st = store_child(s); reader(); return st; }
N int other_case(char *s) { int st = other_child(s); reader(); return st; }
int main(int c, char **v) { if (c > 1) return child_case(v[1]); return 0; }
"""

def _remedy_source(declared: bool) -> str:
    """`tracked` (256 bytes) with an alias symbol at +16. Declared, the C
    definition gives it an ELF `st_size`; undeclared, it and the alias are
    assembly symbols with no `.size` -- the only shape whose end still
    comes from the layout once declared sizes end objects (P1-05)."""
    if declared:
        objects = (
            'char tracked[256] __attribute__((section(".tracked"))) = "seed";\n'
            'extern unsigned char mid;\n'
            '__asm__(".globl mid\\n.set mid,tracked+16\\n.type mid,@object\\n'
            '.size mid,1\\n");\n')
    else:
        objects = (
            'extern char tracked[256];\n'
            '__asm__(".section .tracked,\\"aw\\"\\n.globl tracked\\n'
            '.type tracked,@object\\ntracked:\\n.ascii \\"seed\\"\\n.zero 252\\n'
            '.previous\\n");\n'
            'extern unsigned char mid;\n'
            '__asm__(".globl mid\\n.set mid,tracked+16\\n.type mid,@object\\n");\n')
    return (
        "#include <stdlib.h>\n#include <string.h>\n"
        "#define N __attribute__((noinline,noclone))\n" + objects
        + "N void reader(void) { system(tracked + 80); }\n"
        "N void runtime_case(char *s, size_t n) { memcpy(tracked, s, n); reader(); }\n"
        "N void fixed_case(char *s) { memcpy(tracked, s, 112); reader(); }\n"
        "int main(void) { return 0; }\n")


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_global_filled_by_a_descended_helper_reaches_the_caller(tmp_path,
                                                                  shared_bn,
                                                                  opt):
    """Round-21 review F56: `write_child(s)` fills the global and returns an
    unrelated status; the caller then calls a zero-argument reader of it.

    The write is an effect on the PROGRAM, but `live_globals` is per
    function and the callee summary did not carry it, so the caller saw
    nothing live and the run cleared."""
    src = tmp_path / f"r5child{opt[-1]}.c"
    src.write_text(_CHILD_SRC)
    binary = tmp_path / f"r5child{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("child_case", "store_case"):
        result, leaves = gmc(fn)
        assert any(r["display_name"] == "reader"
                   for lf in leaves for r in lf["references"]), (fn, result)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # the control: the helper fills ANOTHER object
    result, leaves = gmc("other_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_the_extent_remediation_never_fabricates_a_width(tmp_path, shared_bn,
                                                         opt):
    """Round-21 review F57: for a runtime-length copy the advice printed
    `char[1]`, and applying that authoritative width erased a real
    continuation. A size is printed only where the evidence proves one, and
    applying THAT one keeps the reader disclosed.

    P1-05: an object the image DECLARES needs no remedy -- its `st_size`
    already ends it -- so the advice path is exercised on an object
    defined in assembly with no `.size`, and the declared twin must keep
    its reader with no extent doubt and no retype advice at all."""
    def build(declared):
        stem = f"r5rem{'d' if declared else 'u'}{opt[-1]}"
        src = tmp_path / f"{stem}.c"
        src.write_text(_remedy_source(declared))
        binary = tmp_path / stem
        _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
        return shared_bn.load(binary, copy=False)

    def gmc(target, fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    def retypes(leaf):
        return [c for c in re.findall(r"`(bn [^`]+)`", leaf["detail"])
                if "retype" in c]

    target = build(declared=False)
    result, leaves = gmc(target, "runtime_case")
    assert leaves, result
    detail = leaves[0]["detail"]
    assert "object_extent_unproven" in leaves[0]["reachability"]["causes"], leaves
    assert "char[1]`" not in detail, detail
    assert not retypes(leaves[0]), detail
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # the proven-extent case prints a command; running it keeps the reader
    result, leaves = gmc(target, "fixed_case")
    cmds = retypes(leaves[0])
    assert cmds, leaves[0]["detail"]
    res = shared_bn.run(*shlex.split(cmds[0])[1:])
    assert res.returncode == 0, res.stderr
    result, leaves = gmc(target, "fixed_case")
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # the declared twin: the reader at +80 is named, nothing to remedy
    target = build(declared=True)
    for fn in ("runtime_case", "fixed_case"):
        result, leaves = gmc(target, fn)
        [leaf] = leaves
        assert [(r["display_name"], r["offset"]) for r in leaf["references"]] == \
            [("reader", 80)], (fn, leaf)
        assert "object_extent_unproven" not in leaf["reachability"]["causes"], (fn, leaf)
        assert not retypes(leaf), (fn, leaf["detail"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn


def _scanpool_source(statements: int = 1500) -> str:
    body = "".join(f"  t[{i % 200}] ^= (char)(t[{(i + 7) % 200}] + {i});\n"
                   for i in range(statements))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
void (*volatile hook)(void);
N void churn(char *t) {{
{body}  t[1] ^= tracked[8];
}}
N void plain(void) {{ churn(tracked); system(tracked); }}
N void mixed(void) {{ churn(tracked); system(tracked); hook(); }}
N void plain_case(char *s) {{ memcpy(tracked, s, 112); plain(); }}
N void mixed_case(char *s) {{ memcpy(tracked, s, 112); mixed(); }}
int main(void) {{ return 0; }}
"""


def test_a_second_frontier_cause_does_not_erase_an_exhausted_bound(tmp_path,
                                                                   shared_bn):
    """Round-21 review F59: adding a trailing unresolved `hook()` replaced
    `pointer_scan_budget` as THE cause, and `stats` then reported a
    complete run with no truncation at all.

    `churn` names `tracked` only in its LAST statement, so classifying it
    runs the pointer scan out first: the reader sweep no longer scans a
    body that never names the object (R3 sweep economy), so the bound has
    to be met in one that does."""
    src = tmp_path / "r5scan.c"
    src.write_text(_scanpool_source())
    binary = tmp_path / "r5scan"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def run(fn):
        return _bn_json(shared_bn, "-t", target, "taint", "forward",
                        "-f", fn, "--source", "param:0", timeout=300.0)

    plain = run("plain_case")
    assert plain["stats"]["truncated"] is True, plain["stats"]
    assert "pointer_scan_budget" in plain["stats"]["truncation_cause"]

    mixed = run("mixed_case")
    leaf = next(lf for lf in mixed["leaves"]
                if lf["kind"] == "global_memory_continuation")
    assert "pointer_scan_budget" in leaf["reachability"]["causes"], leaf
    assert mixed["stats"]["truncated"] is True, mixed["stats"]
    assert "pointer_scan_budget" in mixed["stats"]["truncation_cause"], \
        mixed["stats"]


_EFFECTS_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
volatile int status_src;
volatile int flag;
N void reader(void) { system(tracked); }
N int fill(char *s) { memcpy(tracked, s, 112); return status_src; }
N int fill_other(char *s) { memcpy(clean, s, 112); return status_src; }
N int wrap2(char *s) { return fill(s); }
N int wrap1(char *s) { return wrap2(s); }
N int nested_case(char *s) { int st = wrap1(s); reader(); return st; }
N int other_nested_case(char *s) { int st = fill_other(s); reader(); return st; }
N int recurse(char *s, int n) { if (n) return recurse(s, n - 1); return fill(s); }
N int recursive_case(char *s) { int st = recurse(s, 3); reader(); return st; }
N int branch_case(char *s) { int st = 0; if (flag) st = fill(s); reader(); return st; }
N int before_case(char *s) { reader(); return fill(s); }
int main(int c, char **v) { if (c > 1) return nested_case(v[1]); return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_callee_write_effect_composes_through_the_call_graph(tmp_path,
                                                               shared_bn, opt):
    """Round-21 review F56, composed: the effect a helper has on a global
    crosses back out at the callsite and is visible to everything the
    caller does AFTERWARDS -- through three levels of wrappers, through
    recursion, and on a conditional path -- without leaking BACKWARDS to a
    reader that already ran."""
    src = tmp_path / f"r5eff{opt[-1]}.c"
    src.write_text(_EFFECTS_SRC)
    binary = tmp_path / f"r5eff{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("nested_case", "recursive_case", "branch_case"):
        _result, leaves = gmc(fn)
        assert any(r["display_name"] == "reader"
                   for lf in leaves for r in lf["references"]), (fn, leaves)

    # the helper fills ANOTHER object: `reader` is no continuation of it,
    # and the exit frontier that write leaves (P2-11.1) -- when something
    # else references `clean` -- is about `clean`, never `reader`
    clean = _data_symbol_addr(binary, "clean")
    result, leaves = gmc("other_nested_case")
    assert leaves == [], leaves
    exits = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
    assert all(lf["global"]["address"] == clean for lf in exits), exits
    assert not any(r["display_name"] == "reader"
                   for lf in exits for r in lf["references"]), exits
    assert result["diagnostics"]["safe_to_report_all_clear"] is (not exits)
    # ...and a reader that ran BEFORE the write is not a continuation of it;
    # it can still run again after `before_case` returns, so the write in
    # `fill` stays the exit frontier (P2-11.1) naming it
    result, leaves = gmc("before_case")
    assert leaves == [], leaves
    [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
    assert store["global"]["address"] == _data_symbol_addr(binary, "tracked"), store
    assert [r["display_name"] for r in store["references"]] == ["reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


_FRAME_ORDER_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[128] __attribute__((section(".tracked"))) = "seed";
volatile int flag;
volatile int counter;
N void reader(void) { system(tracked); }
N int fill(char *s) { memcpy(tracked, s, 64); return counter; }
N int store(char *s) { tracked[0] = s[0]; return counter; }
N int padded_before(char *s) { int a = counter; a += counter; a += counter; a += counter; reader(); return fill(s) + a; }
N int long_child(char *s) { int a = counter; a += counter; a += counter; a += counter; a += counter; a += counter; a += counter; a += counter; memcpy(tracked, s, 64); return a; }
N int short_before(char *s) { reader(); return long_child(s); }
N int sibling(char *s) { int st = 0; if (flag) reader(); else st = fill(s); return st; }
N int after(char *s) { int st = fill(s); reader(); return st; }
N int before_store(char *s) { int a = counter; a += counter; a += counter; reader(); return store(s) + a; }
int main(int c, char **v) { return c > 1 ? padded_before(v[1]) : 0; }
"""

_TAIL_RETURN_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline))
/* separate sections: an ABUTTING neighbour read by an unbounded `system`
   is deliberately conservative (round-10 C1), which is not what this
   control is about. */
char tracked[128] __attribute__((section(".tracked"))) = "seed";
char clean[128] __attribute__((section(".clean"))) = "safe";
volatile int choice;
N char *choose(char *p, char *q, int n) { return n ? p : q; }
N char *producer_equal(void) { return choose(tracked + 8, tracked + 8, choice); }
N char *producer_clean(void) { return choose(clean + 8, clean + 8, choice); }
N void consume_equal(void) { system(producer_equal()); }
N void consume_clean(void) { system(producer_clean()); }
N void equal_case(char *s) { memcpy(tracked, s, 64); consume_equal(); }
N void clean_case(char *s) { memcpy(tracked, s, 64); consume_clean(); }
int main(void) { return 0; }
"""

_ZERO_LEN_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
extern unsigned char mid;
__asm__(".globl mid\n.set mid,tracked+16\n.type mid,@object\n.size mid,1\n");
N void chain_reader(void) { system(tracked + 80); }
N void zero_case(char *s) { memcpy(tracked, s, 0); chain_reader(); }
N void known_case(char *s) { memcpy(tracked, s, 96); chain_reader(); }
N void runtime_case(char *s, size_t n) { memcpy(tracked, s, n); chain_reader(); }
int main(void) { return 0; }
"""


def _chain_source(levels: int = 12) -> str:
    """`useN` reads the pointer a chain of N returning relays hands back."""
    relays = "".join(
        f"N char *relay{i}(void) {{ return relay{i - 1}(); }}\n"
        for i in range(1, levels + 1))
    uses = "".join(
        f"N void use{i}(void) {{ system(relay{i}() + 24); }}\n"
        f"N void chain{i}(char *s) {{ memcpy(payload, s, 64); use{i}(); }}\n"
        for i in range(0, levels + 1, 2))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char payload[256] __attribute__((section(".payload"))) = "seed";
N char *relay0(void) {{ return memset(payload, 0, 1); }}
{relays}{uses}int main(void) {{ return 0; }}
"""


def _layout_source(count: int) -> str:
    """A 1024-byte object with *count* anonymous interior data variables,
    followed by a separately NAMED neighbour in the same section."""
    touches = "".join(f"  x += (unsigned char)big[{8 * (i + 1)}];\n"
                      for i in range(count))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char big[1024] __attribute__((section(".pair"))) = "seed";
char neighbour[64] __attribute__((section(".pair"))) = "safe";
N unsigned probe(void) {{
  unsigned x = 0;
{touches}  return x;
}}
N void reader(void) {{ system(neighbour); }}
N void case_root(char *s) {{ memcpy(big, s, 1); reader(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_callee_write_cannot_precede_a_reader_that_ran_first(tmp_path,
                                                               shared_bn, opt):
    """Round-22 review TIME-01: the merge copied the CALLEE's own IL
    indices into the caller, where an unrelated (or out-of-range) index
    answered "this write reaches that read" about a reader the caller had
    already run."""
    src = tmp_path / f"r6order{opt[-1]}.c"
    src.write_text(_FRAME_ORDER_SRC)
    binary = tmp_path / f"r6order{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    tracked = _data_symbol_addr(binary, "tracked")
    for fn in ("padded_before", "short_before", "before_store", "sibling"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        # No continuation at the reader that ran FIRST -- but the write
        # (`store`'s direct store, or a `memcpy` in `fill`/`long_child`) is
        # still live when the seeded function returns, and `reader` can run
        # again after it: the write is the blocking exit frontier (P1-04,
        # P2-11.1), cited at the write inside the callee and naming the
        # reader.
        [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
        assert store["global"]["address"] == tracked, (fn, store)
        assert [r["display_name"] for r in store["references"]] == ["reader"], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    result, leaves = gmc("after")
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_tail_returned_alias_is_still_a_continuation(tmp_path, shared_bn,
                                                       opt):
    """Round-22 review RETURN-01: at -O2 `producer_equal` tail-jumps into
    `choose`, and returned-pointer inference scanned only `RET` ops -- so
    the alias the caller hands to `system` disappeared and the run
    cleared."""
    src = tmp_path / f"r6tail{opt[-1]}.c"
    src.write_text(_TAIL_RETURN_SRC)
    binary = tmp_path / f"r6tail{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("equal_case")
    assert any(r["display_name"] == "consume_equal"
               for lf in leaves for r in lf["references"]), result
    assert result["diagnostics"]["safe_to_report_all_clear"] is False

    # the same shape over ANOTHER object stays silent
    result, leaves = gmc("clean_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


def test_an_exhausted_pointer_return_scan_is_never_unused(tmp_path, shared_bn):
    """Round-22 review DEPTH-01 (reopened BN21-BOUND-03): past four relays
    the scan spent its pointer-return depth, then classified the reference
    `unused` because a later pass saw no CHANGE in the cause."""
    src = tmp_path / "r6chain.c"
    src.write_text(_chain_source(12))
    binary = tmp_path / "r6chain"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    for level in (0, 2, 4, 6, 8, 10, 12):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", f"chain{level}", "--source", "param:0",
                          "--max-depth", "64")
        leaves = [lf for lf in result["leaves"]
                  if lf["kind"] == "global_memory_continuation"]
        assert leaves, (level, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, level
        if level >= 6:
            # past the bound the answer is UNKNOWN, and says so
            assert "pointer_return_depth" in leaves[0]["reachability"]["causes"], \
                (level, leaves[0]["reachability"])
            assert "pointer_return_depth" in result["stats"]["truncation_cause"]


def test_two_scan_bounds_both_reach_the_aggregate(tmp_path, shared_bn):
    """Round-22 review CAUSE-01: one search, two references, two DIFFERENT
    scan bounds -- the aggregate took the first kind and dropped the
    other."""
    # A body that TOUCHES the object and then does enough work to spend the
    # scan's node pool, beside a relay chain that spends the pointer-return
    # depth: one search, two different bounds. Each statement nests one
    # expression: a sub-expression is now visited once, so a flat `n += i`
    # run no longer spends the pool (R3 sweep economy).
    work = "".join(f"  n = (n ^ {i}) + {i};\n" for i in range(1600))
    src = tmp_path / "r6both.c"
    src.write_text(_chain_source(6).replace(
        "int main(void) { return 0; }",
        f"""volatile int count_value;
N int noisy(void) {{
  memset(payload, 0, 1);
  int n = count_value;
{work}  return n;
}}
N void both(void) {{ use6(); noisy(); }}
N void combined_case(char *s) {{ memcpy(payload, s, 64); both(); }}
int main(void) {{ return 0; }}
"""))
    binary = tmp_path / "r6both"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "combined_case", "--source", "param:0",
                      "--max-depth", "64", timeout=300.0)
    causes = {c for lf in result["leaves"]
              for c in (lf.get("reachability") or {}).get("causes") or ()}
    assert {"pointer_return_depth", "pointer_scan_budget"} <= causes, causes
    assert {"pointer_return_depth", "pointer_scan_budget"} <= \
        set(result["stats"]["truncation_cause"]), result["stats"]


def test_a_zero_length_copy_writes_no_tainted_bytes(tmp_path, shared_bn):
    """Round-22 review ZERO-01: `memcpy(g, s, 0)` was read as "no length
    stated", which is the WIDEST write there is -- so an empty copy raised
    a proven-read continuation 80 bytes into the object."""
    src = tmp_path / "r6zero.c"
    src.write_text(_ZERO_LEN_SRC)
    binary = tmp_path / "r6zero"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("zero_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    # a real length still reaches the reader, and an unknown one still blocks
    for fn in ("known_case", "runtime_case"):
        _result, leaves = gmc(fn)
        assert any(r["display_name"] == "chain_reader"
                   for lf in leaves for r in lf["references"]), (fn, leaves)


@pytest.mark.parametrize("count", [63, 65], ids=["under", "over"])
def test_the_layout_walk_keeps_a_named_neighbour(tmp_path, shared_bn, count):
    """Round-22 review LAYOUT-01: the next-neighbour walk stepped over at
    most 64 anonymous interior variables, then took the SECTION end as the
    object's end -- swallowing a separately named neighbour whole, with no
    uncertainty recorded, so a one-byte copy "reached" a reader of that
    neighbour."""
    src = tmp_path / f"r6layout{count}.c"
    src.write_text(_layout_source(count))
    binary = tmp_path / f"r6layout{count}"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-g0",
                           "-fno-toplevel-reorder"])
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                      "-f", "case_root", "--source", "param:0")
    leaves = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    assert leaves == [], (count, leaves)
    assert result["diagnostics"]["safe_to_report_all_clear"] is True


_IMPORT_TAIL_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char clean[256] __attribute__((section(".clean"))) = "safe";
volatile unsigned char seen;
N char *refill(void) { return fgets(tracked, 2, stdin); }
N char *reset(void) { return memset(tracked, 0, 1); }
N char *copy(char *s) { return memcpy(tracked, s, 1); }
N char *loader(void) { return strcpy(tracked, "x"); }
N void use_refill(void) { system(refill() + 8); }
N void use_reset(void) { system(reset() + 8); }
N void use_copy(char *s) { system(copy(s) + 8); }
N void use_load(void) { system(loader() + 8); }
N void ignored(void) { refill(); seen = 1; }
N char *refill_clean(void) { return fgets(clean, 2, stdin); }
N void use_clean(void) { system(refill_clean() + 8); }
N void refill_case(char *s) { memcpy(tracked, s, 64); use_refill(); }
N void reset_case(char *s) { memcpy(tracked, s, 64); use_reset(); }
N void copy_case(char *s) { memcpy(tracked, s, 64); use_copy(s); }
N void load_case(char *s) { memcpy(tracked, s, 64); use_load(); }
N void ignored_case(char *s) { memcpy(tracked, s, 64); ignored(); }
N void clean_case(char *s) { memcpy(tracked, s, 64); use_clean(); }
int main(void) { return 0; }
"""

_MIXED_RETURN_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
volatile int choice;
N char *direct_far(void) { return tracked + 80; }
N char *mixed(void) { if (choice) return tracked + 8; return direct_far(); }
N char *equal(void) { if (choice) return tracked + 8; return tracked + 8; }
N void mixed_reader(void) { system(mixed()); }
N void equal_reader(void) { system(equal()); }
N void mixed_case(char *s) { memcpy(tracked, s, 128); mixed_reader(); }
N void equal_case(char *s) { memcpy(tracked, s, 128); equal_reader(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_tail_return_into_a_modeled_import_keeps_its_alias(tmp_path,
                                                             shared_bn, opt):
    """Round-23 review F60: `return fgets(g, 2, stdin);` compiled to a tail
    jump returns the DESTINATION the model names, and the caller's
    `result + 8` points into that object -- the tail path asked only for an
    in-binary body's returned formals, found none, and the run cleared."""
    src = tmp_path / f"r7imp{opt[-1]}.c"
    src.write_text(_IMPORT_TAIL_SRC)
    binary = tmp_path / f"r7imp{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("refill_case", "reset_case", "copy_case", "load_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # the return is DISCARDED, and the same shape over another object
    for fn in ("ignored_case", "clean_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        assert result["diagnostics"]["safe_to_report_all_clear"] is True, fn


@pytest.mark.parametrize("opt", ["-O0", "-O2"], ids=["O0", "O2"])
def test_a_mixed_ordinary_and_tail_return_joins_both_offsets(tmp_path,
                                                             shared_bn, opt):
    """Round-23 review F61: one arm returns `g + 8`, the other tail-returns
    a body handing back `g + 80`. The displacement join gathered only `RET`
    instructions, so one branch's offset was reported as EXACT."""
    src = tmp_path / f"r7mix{opt[-1]}.c"
    src.write_text(_MIXED_RETURN_SRC)
    binary = tmp_path / f"r7mix{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def rows(fn, name):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return [r for lf in result["leaves"]
                for r in lf.get("references", [])
                if r["display_name"] == name]

    [mixed] = rows("mixed_case", "mixed_reader")
    assert mixed["offset_known"] is False, mixed
    [equal] = rows("equal_case", "equal_reader")
    assert (equal["offset"], equal["offset_known"]) == (8, True), equal


def _layout_bound_source(count: int) -> str:
    # `big` is an assembly object with no `.size`: a declared size would end
    # it before any layout walk (P1-05), and the walk -- with its budget --
    # is what this fixture is about.
    touches = "".join(f"  x += (unsigned char)big[{8 * (i + 1)}];\n"
                      for i in range(count))
    return f"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
extern char big[65536];
__asm__(".section .pair,\\"aw\\"\\n.globl big\\n.type big,@object\\n"
        "big:\\n.ascii \\"seed\\"\\n.zero 65532\\n.previous\\n");
char neighbour[64] __attribute__((section(".pair"))) = "safe";
volatile unsigned char seen;
N unsigned probe(void) {{
  unsigned x = 0;
{touches}  return x;
}}
N int quiet(void) {{ return 7; }}
N void reader(void) {{ system(big + 4096); }}
N void quiet_case(char *s) {{ memcpy(big, s, 1); seen = (unsigned char)quiet(); }}
N void read_case(char *s) {{ memcpy(big, s, 8192); reader(); }}
int main(void) {{ return 0; }}
"""


@pytest.mark.parametrize("count", [4095, 4097], ids=["under", "over"])
def test_the_layout_bound_speaks_only_where_a_reader_could_hide(tmp_path,
                                                                shared_bn,
                                                                count):
    """Round-23 review F62: the new layout bound was attached to every
    frontier question about the object, so a callee whose whole body is
    `return 7;` became a blocking continuation once the walk ran out.

    The bound only exists for an object whose end comes from the layout:
    one the image declares a size for ends there (P1-05). `big` is defined
    in assembly without `.size` so the walk still runs."""
    src = tmp_path / f"r7lay{count}.c"
    src.write_text(_layout_bound_source(count))
    binary = tmp_path / f"r7lay{count}"
    _compile(src, binary, ["-O0", "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-g0", "-fno-toplevel-reorder"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", timeout=300.0)
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("quiet_case")
    assert leaves == [], (count, leaves)
    assert result["diagnostics"]["safe_to_report_all_clear"] is True

    # the genuine interior reader stays blocking at either side of the bound
    result, leaves = gmc("read_case")
    assert any(r["display_name"] == "reader"
               for lf in leaves for r in lf["references"]), (count, leaves)
    if count > 4096:
        assert "object_layout_budget" in leaves[0]["reachability"]["causes"]
        assert "global_object_layout" in result["stats"]["truncation_cause"]


_RETURNED_ALIAS_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char output[256] __attribute__((section(".out"))) = "out";
volatile int choice;
#ifdef INTERIOR_ALIAS
extern char middle[];
__asm__(".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n");
#endif
N char *advance(char *p) { return p + 80; }
N char *tail(void) { return advance(tracked + 8); }
N char *mixed(void) { if (choice) return tracked + 8; return advance(tracked + 8); }
N char *direct(void) { return tracked + 88; }
N char *far_clean(void) { return output + 8; }
N void tail_reader(void) { system(tail()); }
N void mixed_reader(void) { system(mixed()); }
N void direct_reader(void) { system(direct()); }
N void clean_reader(void) { system(far_clean()); }
N void zero_reader(void) { memcpy(output, tracked, 0); }
N void one_reader(void) { memcpy(output, tracked, 1); }
N void n_reader(size_t n) { memcpy(output, tracked, n); }
#ifdef INTERIOR_ALIAS
N void tail_case(char *s) { memcpy(middle, s, 64); tail_reader(); }
N void mixed_case(char *s) { memcpy(middle, s, 64); mixed_reader(); }
N void direct_case(char *s) { memcpy(middle, s, 64); direct_reader(); }
N void clean_case(char *s) { memcpy(middle, s, 64); clean_reader(); }
#else
N void tail_case(char *s) { memcpy(tracked, s, 128); tail_reader(); }
N void mixed_case(char *s) { memcpy(tracked, s, 128); mixed_reader(); }
N void direct_case(char *s) { memcpy(tracked, s, 128); direct_reader(); }
N void clean_case(char *s) { memcpy(tracked, s, 128); clean_reader(); }
#endif
N void zero_case(char *s) { memcpy(tracked, s, 128); zero_reader(); }
N void one_case(char *s) { memcpy(tracked, s, 128); one_reader(); }
N void n_case(char *s, size_t n) { memcpy(tracked, s, 128); n_reader(n); }
int main(void) { return 0; }
"""


def _returned_alias_target(tmp_path, shared_bn, opt, interior):
    stem = f"r8ret{opt[-1]}{'i' if interior else 'p'}"
    src = tmp_path / f"{stem}.c"
    src.write_text(_RETURNED_ALIAS_SRC)
    binary = tmp_path / stem
    flags = [opt, "-U_FORTIFY_SOURCE", "-fno-builtin", "-fno-inline", "-g0"]
    if interior:
        flags.append("-DINTERIOR_ALIAS")
    _compile(src, binary, flags)
    return shared_bn.load(binary, copy=False)


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_returned_pointer_into_a_written_interior_alias_blocks(tmp_path,
                                                                 shared_bn,
                                                                 opt):
    """Round-24 review F63: the write names `middle = tracked + 64` and the
    consumer receives `advance(tracked + 8)` from a CALL, naming nothing
    itself -- so the overlap test's direct-reference prefilter dropped it
    and the run cleared. A consumer of a DISJOINT object raises no
    continuation at its call; the copy is still live at the return, so the
    P2-11.1 exit store blocks naming a real reader of the written bytes
    (`direct_reader`, `tracked + 88`), never `clean_reader`."""
    target = _returned_alias_target(tmp_path, shared_bn, opt, interior=True)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("tail_case", "mixed_case", "direct_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # a returned pointer into a DISJOINT object is still not a continuation
    result, leaves = gmc("clean_case")
    assert leaves == [], leaves
    [store] = _exit_stores(result)
    assert [r["display_name"] for r in store["references"]] == \
        ["direct_reader"], store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_tail_return_composes_the_callee_adjustment(tmp_path, shared_bn,
                                                      opt):
    """Round-24 review F64: `advance(p) = p + 80` tail-called with
    `tracked + 8` reaches +88, but the tail summary kept only the actual's
    displacement and certified +8 as EXACT."""
    target = _returned_alias_target(tmp_path, shared_bn, opt, interior=False)

    def rows(fn, name):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return [r for lf in result["leaves"]
                for r in lf.get("references", [])
                if r["display_name"] == name]

    [tail] = rows("tail_case", "tail_reader")
    assert (tail["offset"], tail["offset_known"]) == (88, True), tail
    [direct] = rows("direct_case", "direct_reader")
    assert (direct["offset"], direct["offset_known"]) == (88, True), direct
    # +8 on one route and +88 on the other is not one displacement
    [mixed] = rows("mixed_case", "mixed_reader")
    assert mixed["offset_known"] is False, mixed


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_zero_length_read_is_not_an_access(tmp_path, shared_bn, opt):
    """Round-24 review F65: `memcpy(out, g, 0)` reads no byte of `g`, yet
    the consumer was reported as a proven blocking read.

    The write is still live when `zero_case` returns and other functions
    read `tracked`, so it is the exit frontier (P2-11.1) -- naming one of
    those genuine readers, never the zero-length one."""
    target = _returned_alias_target(tmp_path, shared_bn, opt, interior=False)

    def gmc(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    result, leaves = gmc("zero_case")
    assert leaves == [], leaves
    [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
    [row] = store["references"]
    assert row["display_name"] in {"tail_reader", "mixed_reader", "direct_reader",
                                   "one_reader", "n_reader"}, store
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    # one byte, and a length nothing states, still block
    for fn in ("one_case", "n_case"):
        _result, leaves = gmc(fn)
        assert leaves, fn


_SPAN_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char output[256] __attribute__((section(".out"))) = "out";
volatile int choice;
extern char middle[];
__asm__(".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n");
N void zero_reader(void) { snprintf(output, 0, tracked); }
N void one_reader(void) { snprintf(output, 1, tracked); }
N void wide_reader(void) { wmemcpy((wchar_t *)output, (wchar_t *)tracked, 32); }
N void byte88_reader(void) { output[0] = tracked[88]; }
N char *choose_pointer(void) { return choice ? tracked + 48 : tracked + 64; }
N char *advance16(char *p) { return p + 16; }
N void phi_reader(void) { system(advance16(choose_pointer())); }
N void zero_case(char *s) { memcpy(tracked, s, 128); zero_reader(); }
N void narrow_case(char *s) { memcpy(middle, s, 64); one_reader(); }
N void wide_read_case(char *s) { memcpy(middle, s, 64); wide_reader(); }
N void wide_write_case(char *s) { wmemcpy((wchar_t *)tracked, (wchar_t *)s, 32); byte88_reader(); }
N void phi_case(char *s) { memcpy(middle, s, 64); phi_reader(); }
int main(void) { return 0; }
"""

_ABUT_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char first[64] __attribute__((section(".both"))) = "one";
char second[64] __attribute__((section(".both"))) = "two";
char output[256] __attribute__((section(".out"))) = "out";
N char *pointer(void) { return first; }
N void bounded(void) { memcpy(output, pointer(), 64); }
N void unbounded(void) { system(pointer()); }
N void over_read(void) { memcpy(output, pointer(), 128); }
N void direct(void) { memcpy(output, first, 64); }
N void bounded_root(char *s) { memcpy(second, s, 64); bounded(); }
N void unbounded_root(char *s) { memcpy(second, s, 64); unbounded(); }
N void over_root(char *s) { memcpy(second, s, 64); over_read(); }
N void direct_root(char *s) { memcpy(second, s, 64); direct(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_modeled_count_bounds_the_argument_it_describes(tmp_path, shared_bn,
                                                          opt):
    """Round-25 review F66 and F67: `snprintf(dst, size, fmt)` states a
    DESTINATION capacity -- it bounds nothing about the format it reads --
    and a wide count is in ELEMENTS, not bytes."""
    src = tmp_path / f"r9span{opt[-1]}.c"
    src.write_text(_SPAN_SRC)
    binary = tmp_path / f"r9span{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # F66: a zero and a one-byte capacity both still read the format
    for fn in ("zero_case", "narrow_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F67: 32 wide characters cover 128 bytes, read side and write side
    for fn in ("wide_read_case", "wide_write_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F68: +0 on one route and +16 on the other is not one displacement
    _result, leaves = gmc("phi_case")
    row = next(r for lf in leaves for r in lf["references"]
               if r["display_name"] == "phi_reader")
    assert row["offset_known"] is False, row


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_bounded_read_of_an_abutting_neighbour_is_disjoint(tmp_path,
                                                             shared_bn, opt):
    """Round-25 review F69: `memcpy(out, pointer(), 64)` reads exactly the
    clean neighbour, ending where the tainted object begins -- but RETURNING
    that neighbour's address was counted as an unbounded use of it.

    P1-05: `first` DECLARES 64 bytes, so `system(first)` -- a read of no
    stated length -- is a read of first's own storage, not of `second`;
    `memcpy(out, first, 128)` states that it reads 64 bytes past that end,
    and is cited as a proven reader of `second`. The bytes the roots copy
    into `second` are still live when they return, so each ends in the
    P2-11.1 exit store -- naming that proven reader, never the bounded,
    direct or unbounded reader the root called."""
    src = tmp_path / f"r9abut{opt[-1]}.c"
    src.write_text(_ABUT_SRC)
    binary = tmp_path / f"r9abut{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0", "-fno-toplevel-reorder"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("bounded_root", "direct_root", "unbounded_root"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = _exit_stores(result)
        assert [(r["display_name"], r["access"]) for r in store["references"]] == \
            [("over_read", "call_arg")], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # a read whose stated length runs into the tainted bytes still blocks
    result, leaves = gmc("over_root")
    [leaf] = leaves
    assert [(r["display_name"], r["access"]) for r in leaf["references"]] == \
        [("over_read", "call_arg")], leaf
    assert leaf["reads_proven"] is True, leaf
    assert result["diagnostics"]["safe_to_report_all_clear"] is False


_SCOPE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#define N __attribute__((noinline,noclone))
/* ONE section for all three objects: a distinct neighbour must be told
   apart by the named boundaries between them, not by the section
   (round-27 review F73). */
char tracked[256] __attribute__((section(".same"))) = "seed";
/* `adjacent` follows `tracked` with NOTHING named in between, so the arm's
   OWN symbol is the only boundary there is (round-28 review F73). */
char adjacent[256] __attribute__((section(".same"))) = "next";
char gap[256] __attribute__((section(".same"))) = "gap";
char other[256] __attribute__((section(".same"))) = "other";
char output[512] __attribute__((section(".out"))) = "out";
volatile int choice;
extern char before[];
extern char middle[];
/* `before` is a NAMED interior alias of this object's own storage -- a
   symbol does not make it separate storage (round-29 review F73). */
__asm__(".globl before\n.set before,tracked+48\n.type before,@object\n.size before,1\n");
__asm__(".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n");
N void bounded_reader(void) { strlcpy(output, tracked, 1); }
N void wide_chk_reader(void) { __wmemcpy_chk((wchar_t *)output, (wchar_t *)tracked, 32, 256); }
N char *choose_pointer(void) { return choice ? middle : other + 8; }
N char *choose_base(void) { return choice ? middle : adjacent; }
N char *advance16(char *p) { return p + 16; }
N void phi_reader(void) { system(advance16(choose_pointer())); }
N void strl_case(char *s) { memcpy(middle, s, 64); bounded_reader(); }
N void wide_chk_case(char *s) { memcpy(middle, s, 64); wide_chk_reader(); }
N void phi_case(char *s) { memcpy(middle, s, 64); phi_reader(); }
N char *choose_inside(void) { return choice ? middle : tracked + 40; }
N char *choose_alias(void) { return choice ? middle : before; }
N void base_reader(void) { system(advance16(choose_base())); }
N void inside_reader(void) { system(advance16(choose_inside())); }
N void alias_reader(void) { system(advance16(choose_alias())); }
N void base_case(char *s) { memcpy(middle, s, 64); base_reader(); }
N void inside_case(char *s) { memcpy(middle, s, 64); inside_reader(); }
N void alias_case(char *s) { memcpy(middle, s, 64); alias_reader(); }
int main(void) { return 0; }
"""

_PE_WIDE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#define N __attribute__((noinline,noclone))
char tracked[256] = "seed";
volatile unsigned short seen;
N void byte88_reader(void) { seen = *(unsigned short *)(tracked + 88); }
N void byte62_reader(void) { seen = *(unsigned short *)(tracked + 62); }
N void pe_case(char *s) { wmemcpy((wchar_t *)tracked, (wchar_t *)s, 32); byte88_reader(); }
N void pe_inside_case(char *s) { wmemcpy((wchar_t *)tracked, (wchar_t *)s, 32); byte62_reader(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_bounded_copy_capacity_does_not_bound_its_source(tmp_path,
                                                           shared_bn, opt):
    """Round-26 review F70, F71 and F73.

    `strlcpy(dst, src, 1)` states a DESTINATION capacity and still reads
    the whole source string; `__wmemcpy_chk` counts WIDE characters like
    its unfortified spelling; and a PHI arm naming a separate object with
    other storage between it and this one is disjoint, not "not there
    yet"."""
    src = tmp_path / f"r10scope{opt[-1]}.c"
    src.write_text(_SCOPE_SRC)
    binary = tmp_path / f"r10scope{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0", "-fno-toplevel-reorder"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    for fn in ("strl_case", "wide_chk_case"):
        result, leaves = gmc(fn)
        assert leaves, (fn, result["stats"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # F73: the other arm names a DISTINCT object in the same section, with
    # named objects between it and this one, so only one arm can be this
    # object and its offset IS established
    _result, leaves = gmc("phi_case")
    row = next(r for lf in leaves for r in lf["references"]
               if r["display_name"] == "phi_reader")
    assert (row["offset"], row["offset_known"]) == (16, True), row
    # the same holds when the distinct arm IS that global's base, with no
    # storage in between -- the arm's own symbol is the boundary
    _result, leaves = gmc("base_case")
    row = next(r for lf in leaves for r in lf["references"]
               if r["display_name"] == "base_reader")
    assert (row["offset"], row["offset_known"]) == (16, True), row
    # ...while an arm inside THIS object's own storage keeps the join
    # open, whether or not that arm carries a SYMBOL of its own
    for fn, name in (("inside_case", "inside_reader"),
                     ("alias_case", "alias_reader")):
        _result, leaves = gmc(fn)
        row = next(r for lf in leaves for r in lf["references"]
                   if r["display_name"] == name)
        assert row["offset_known"] is False, (fn, row)


def test_a_wide_count_follows_the_target_wchar_abi(tmp_path, shared_bn):
    """Round-26 review F72: `wchar_t` is 2 bytes on Windows, so
    `wmemcpy(g, s, 32)` writes 64 bytes there -- scaling from the pointer
    width invented a 128-byte footprint and blocked a read of bytes the
    program never wrote.

    The write is live when `pe_case` returns, so it is the exit frontier
    (P2-11.1); the reader it names is the one inside [0, 64), never the
    one at +88 a 128-byte footprint would have cited."""
    cc = shutil.which("x86_64-w64-mingw32-gcc")
    if cc is None:
        refuse_silent_skip("x86_64-w64-mingw32-gcc is required for the PE64 ABI")
        pytest.skip("no mingw cross compiler")
    src = tmp_path / "r10pe.c"
    src.write_text(_PE_WIDE_SRC)
    binary = tmp_path / "r10pe.exe"
    result = subprocess.run(
        [cc, "-O0", "-g0", "-fno-builtin", "-fno-inline",
         str(src), "-o", str(binary)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        res = _bn_json(shared_bn, "-t", target, "taint", "forward",
                       "-f", fn, "--source", "param:0")
        return res, [lf for lf in res["leaves"]
                     if lf["kind"] == "global_memory_continuation"]

    # 32 wide characters cover bytes [0, 64): +88 is outside, +62 inside.
    # mingw's in-binary `wmemcpy` scales the count before its tail call, so
    # it is a body that is walked (P2-03), not a veneer; its own copy through
    # a parameter pointer is a coarse store of that walk. The question here is
    # only the global continuation's footprint.
    _result, leaves = gmc("pe_case")
    assert leaves == [], leaves

    _result, leaves = gmc("pe_inside_case")
    assert any(r["display_name"] == "byte62_reader"
               for lf in leaves for r in lf["references"]), leaves


_MISS_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char lower[256] __attribute__((section(".same"))) = "low";
char tracked[256] __attribute__((section(".same"))) = "seed";
char upper[256] __attribute__((section(".same"))) = "up";
volatile int choice;
volatile unsigned char seen;
extern char middle[];
__asm__(".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n");
N char *choose_miss(void) { return choice ? middle + 80 : lower + 8; }
N char *choose_hit(void) { return choice ? middle + 8 : lower + 8; }
N void miss_reader(void) { seen = *choose_miss(); }
N void hit_reader(void) { seen = *choose_hit(); }
N void direct_miss_reader(void) { seen = *(middle + 80); }
N void far_string_reader(void) { system(tracked + 80); }
N void inline_miss_case(char *s) { memcpy(middle, s, 64); miss_reader(); }
N void inline_hit_case(char *s) { memcpy(middle, s, 64); hit_reader(); }
N void direct_miss_case(char *s) { memcpy(middle, s, 64); direct_miss_reader(); }
N void short_copy_case(char *s) { memcpy(tracked, s, 16); far_string_reader(); }
N void long_copy_case(char *s) { memcpy(tracked, s, 128); far_string_reader(); }
N void unknown_copy_case(char *s, size_t n) { memcpy(tracked, s, n); far_string_reader(); }
int main(void) { return 0; }
"""


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_read_that_misses_every_tainted_byte_is_not_a_reference(tmp_path,
                                                                  shared_bn,
                                                                  opt):
    """Round-30 review F75 and F76.

    A distinct global further below than this object is wide is not an arm
    a route can land inside, and a FORWARD string read that starts past
    every tainted byte touches none of them -- the end being unknown does
    not make the start unknown.

    A write into `middle` that is live when the case returns is the exit
    frontier (P2-11.1) whenever another function reads those bytes: it
    names that reader (`far_string_reader`, from +80), never the reader
    whose every arm misses."""
    src = tmp_path / f"r11miss{opt[-1]}.c"
    src.write_text(_MISS_SRC)
    binary = tmp_path / f"r11miss{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0", "-fno-toplevel-reorder"])
    target = shared_bn.load(binary, copy=False)

    def gmc(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "global_memory_continuation"]

    # neither arm can reach the written bytes, directly or through the PHI
    middle = _data_symbol_addr(binary, "middle")
    for fn in ("inline_miss_case", "direct_miss_case"):
        result, leaves = gmc(fn)
        assert leaves == [], (fn, leaves)
        [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
        assert store["global"]["address"] == middle, (fn, store)
        assert [r["display_name"] for r in store["references"]] == \
            ["far_string_reader"], (fn, store)
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, fn

    # ...while an arm that lands INSIDE the written bytes still blocks
    _result, leaves = gmc("inline_hit_case")
    assert leaves, leaves

    # a string read starting past a 16-byte copy reads none of it
    result, leaves = gmc("short_copy_case")
    assert leaves == [], leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is True
    # ...while a copy that reaches it, and a length nothing states, block
    for fn in ("long_copy_case", "unknown_copy_case"):
        _result, leaves = gmc(fn)
        assert any(r["display_name"] == "far_string_reader"
                   for lf in leaves for r in lf["references"]), (fn, leaves)


_STORAGE_SRC = r"""
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char previous[256] __attribute__((section(".same"))) = "prev";
char tracked[256] __attribute__((section(".same"))) = "seed";
char adjacent[256] __attribute__((section(".same"))) = "next";
char elsewhere[256] __attribute__((section(".other"))) = "other";
volatile int choice;
extern char before[], middle[], upper[];
__asm__(".globl before\n.set before,tracked+48\n.type before,@object\n.size before,1\n"
        ".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n"
        ".globl upper\n.set upper,tracked+144\n.type upper,@object\n.size upper,1\n");
N char *pick(char *p, char *q) { return choice ? p : q; }
N char *plus16(char *p) { return p + 16; }
N char *back32(char *p) { return p - 32; }
N char *only_upper(void) { return upper; }
N char *only_elsewhere(void) { return elsewhere + 144; }
N char *joined(void) { return choice ? middle : upper; }
N char *formal_below(void) { return pick(before, middle); }
N char *formal_equal(void) { return pick(middle, middle); }
N char *formal_previous(void) { return pick(middle, previous); }
N char *formal_adjacent(void) { return pick(middle, adjacent); }
N void upper_reader(void) { system(back32(only_upper())); }
N void joined_reader(void) { system(back32(joined())); }
N void elsewhere_reader(void) { system(back32(only_elsewhere())); }
N void direct_reader(void) { system(tracked + 112); }
N void below_reader(void) { system(plus16(formal_below())); }
N void equal_reader(void) { system(plus16(formal_equal())); }
N void previous_reader(void) { system(plus16(formal_previous())); }
N void adjacent_reader(void) { system(plus16(formal_adjacent())); }
N void upper_case(char *s) { memcpy(middle, s, 64); upper_reader(); }
N void joined_case(char *s) { memcpy(middle, s, 64); joined_reader(); }
N void elsewhere_case(char *s) { memcpy(middle, s, 64); elsewhere_reader(); }
N void direct_case(char *s) { memcpy(middle, s, 64); direct_reader(); }
N void below_case(char *s) { memcpy(middle, s, 64); below_reader(); }
N void equal_case(char *s) { memcpy(middle, s, 64); equal_reader(); }
N void previous_case(char *s) { memcpy(middle, s, 64); previous_reader(); }
N void adjacent_case(char *s) { memcpy(middle, s, 64); adjacent_reader(); }
int main(void) { return 0; }
"""


_APPEND_END_SRC = r"""
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = {[0 ... 63]='A'};
volatile unsigned char observed;
volatile int choice;
extern char middle[];
__asm__(".globl middle\n.set middle,tracked+64\n.type middle,@object\n.size middle,1\n");
N void read64(void) { observed = tracked[64]; }
N void append_one(char *s) { strncat(tracked, s, 1); read64(); }
N void append_chk_one(char *s) { __strncat_chk(tracked, s, 1, 256); read64(); }
N void copy_one(char *s) { memcpy(tracked + 64, s, 1); read64(); }
N void copy_base(char *s) { memcpy(tracked, s, 1); read64(); }
N void scan_one(void) { strncat(tracked, "x", 1); }
N void scan_zero(void) { strncat(tracked, "x", 0); }
N void scan_one_case(char *s) { memcpy(middle, s, 64); scan_one(); }
N void scan_zero_case(char *s) { memcpy(middle, s, 64); scan_zero(); }
N void empty_copy(void) { memcpy(tracked, "x", 0); }
N void empty_copy_case(char *s) { memcpy(middle, s, 64); empty_copy(); }
N char *endcopy(void) { return mempcpy(tracked, "X", 1); }
N char *endstring(void) { return stpcpy(tracked, "X"); }
N char *endfind(void) { return memccpy(tracked, "X", 'X', 1); }
N char *endunknown(void) { return stpcpy(tracked, choice ? "AB" : "ABCD"); }
N void endcopy_reader(void) { system(endcopy() + 8); }
N void endstring_reader(void) { system(endstring() + 8); }
N void endfind_reader(void) { char *p = endfind(); if (p) system(p + 8); }
N void endunknown_reader(void) { system(endunknown() + 8); }
N void endcopy_case(char *s) { memcpy(tracked, s, 128); endcopy_reader(); }
N void endstring_case(char *s) { memcpy(tracked, s, 128); endstring_reader(); }
N void endfind_case(char *s) { memcpy(tracked, s, 128); endfind_reader(); }
N void endunknown_case(char *s) { memcpy(tracked, s, 128); endunknown_reader(); }
int main(void) { return 0; }
"""


_FORTIFIED_SRC = r"""
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define N __attribute__((noinline,noclone))
extern char *__fgets_chk(char *, size_t, int, FILE *);
char tracked[256] __attribute__((section(".tracked"))) = "seed";
N char *fortified(void) { return __fgets_chk(tracked, 256, 2, stdin); }
N char *fortified_tail(void) { return __fgets_chk(tracked, 256, 2, stdin); }
N char *plain(void) { return fgets(tracked, 2, stdin); }
N void fortified_reader(void) { char *p = fortified(); if (p) system(p + 8); }
N void tail_reader(void) { char *p = fortified_tail(); if (p) system(p + 8); }
N void plain_reader(void) { char *p = plain(); if (p) system(p + 8); }
N void ignored_reader(void) { (void)fortified(); }
N void fortified_case(char *s) { memcpy(tracked, s, 64); fortified_reader(); }
N void tail_case(char *s) { memcpy(tracked, s, 64); tail_reader(); }
N void plain_case(char *s) { memcpy(tracked, s, 64); plain_reader(); }
N void ignored_case(char *s) { memcpy(tracked, s, 64); ignored_reader(); }
int main(void) { return 0; }
"""


def _taint_target(tmp_path, shared_bn, source: str, stem: str, opt: str):
    src = tmp_path / f"{stem}{opt[-1]}.c"
    src.write_text(source)
    binary = tmp_path / f"{stem}{opt[-1]}"
    _compile(src, binary, [opt, "-U_FORTIFY_SOURCE", "-fno-builtin",
                           "-fno-inline", "-g0", "-fno-toplevel-reorder"])
    return shared_bn.load(binary, copy=False)


def _gmc_rows(shared_bn, target, fn, *extra):
    """``(leaves, (display_name, offset, offset_known) rows, all-clear)``."""
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                      "--source", "param:0", *extra)
    leaves = [lf for lf in result["leaves"]
              if lf["kind"] == "global_memory_continuation"]
    rows = [(r["display_name"], r["offset"], r["offset_known"])
            for lf in leaves for r in lf["references"]]
    return leaves, rows, result["diagnostics"]["safe_to_report_all_clear"]


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_one_objects_interior_aliases_are_one_storage(tmp_path, shared_bn, opt):
    """BN23-ALIAS-01 and BN23-JOIN-01.

    `before`, `middle` and `upper` name the inside of ONE declared object,
    so a pointer taken from any of them and adjusted composes back into
    the bytes a write to `middle` tainted. The objects next door -- the
    one before it, the one after it, and one in another section -- are
    separate storage however close their bases: a reader of one of them
    is no continuation, and the live write's exit frontier (P2-11.1)
    names a reader of `tracked`, not it."""
    target = _taint_target(tmp_path, shared_bn, _STORAGE_SRC, "r11store", opt)

    # the upper alias, adjusted back INTO the written bytes
    _leaves, rows, clear = _gmc_rows(shared_bn, target, "upper_case")
    assert ("upper_reader", 48, True) in rows, rows
    assert clear is False
    # ...the same displacement the direct spelling reports
    _leaves, rows, _clear = _gmc_rows(shared_bn, target, "direct_case")
    assert ("direct_reader", 48, True) in rows, rows
    # ...and a join of the two feasible arms is not one of them
    leaves, rows, clear = _gmc_rows(shared_bn, target, "joined_case")
    assert leaves and clear is False, leaves
    assert all(known is False for _name, _off, known in rows
               if _name == "joined_reader"), rows

    # a returned FORMAL composes the same way: 0 and 16 are both feasible
    leaves, rows, _clear = _gmc_rows(shared_bn, target, "below_case",
                                     "--max-depth", "64")
    assert leaves, leaves
    assert ("below_reader", 0, False) in rows, rows
    # ...while arms that agree, and arms in other objects, keep +16 exact
    for case, reader in (("equal_case", "equal_reader"),
                         ("previous_case", "previous_reader"),
                         ("adjacent_case", "adjacent_reader")):
        _leaves, rows, _clear = _gmc_rows(shared_bn, target, case,
                                          "--max-depth", "64")
        assert (reader, 16, True) in rows, (case, rows)

    # another section is another object, adjusted or not
    leaves, _rows, clear = _gmc_rows(shared_bn, target, "elsewhere_case")
    assert leaves == [] and clear is False, leaves
    exits = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f",
                     "elsewhere_case", "--source", "param:0")["leaves"]
    middle = _data_symbol_addr(tmp_path / f"r11store{opt[-1]}", "middle")
    assert [(lf["global"]["address"], [r["display_name"] for r in lf["references"]])
            for lf in exits if lf["kind"] == "coarse_memory_store"] == \
        [(middle, ["upper_reader"])], exits


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_an_append_is_not_a_base_relative_bounded_copy(tmp_path, shared_bn,
                                                       opt):
    """BN23-APPEND-01.

    `strncat(dst, src, n)` copies at most n bytes OF SRC to wherever dst's
    terminator is. The count bounds neither the destination scan -- which
    runs over the existing string whatever n says, including 0 -- nor the
    position the appended bytes land at.

    A write live when its case returns is the exit frontier (P2-11.1)
    naming a function that reads the tainted bytes -- never `read64` for
    a byte at the base, never the zero-length copy."""
    target = _taint_target(tmp_path, shared_bn, _APPEND_END_SRC, "r11span", opt)

    # the appended byte lands at the terminator at +64, not at the base
    for case in ("append_one", "append_chk_one"):
        _leaves, rows, clear = _gmc_rows(shared_bn, target, case)
        assert ("read64", 64, True) in rows, (case, rows)
        assert clear is False, case
    # ...which the direct one-byte copy at +64 has always reported
    _leaves, rows, _clear = _gmc_rows(shared_bn, target, "copy_one")
    assert ("read64", 64, True) in rows, rows
    # ...and a one-byte copy at the BASE still reaches nothing at +64: the
    # exit frontier names the append that scans byte 0, not `read64`
    binary = tmp_path / f"r11span{opt[-1]}"
    tracked = _data_symbol_addr(binary, "tracked")

    def exit_rows(case):
        found = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", case,
                         "--source", "param:0")["leaves"]
        return [(lf["global"]["address"], [r["display_name"] for r in lf["references"]])
                for lf in found if lf["kind"] == "coarse_memory_store"]

    leaves, _rows, clear = _gmc_rows(shared_bn, target, "copy_base")
    assert leaves == [] and clear is False, leaves
    assert exit_rows("copy_base") == [(tracked, ["append_one"])]

    # the destination scan reads the tainted bytes whatever the count says
    for case in ("scan_one_case", "scan_zero_case"):
        leaves, _rows, clear = _gmc_rows(shared_bn, target, case)
        assert leaves and clear is False, (case, leaves)
    # ...while a genuine zero-length COPY still touches nothing
    leaves, _rows, clear = _gmc_rows(shared_bn, target, "empty_copy_case")
    assert leaves == [] and clear is False, leaves
    [(where, names)] = exit_rows("empty_copy_case")
    assert where in (tracked, _data_symbol_addr(binary, "middle")), where
    assert names == ["read64"], names


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_an_end_returning_copy_is_not_the_destination_base(tmp_path, shared_bn,
                                                           opt):
    """BN23-ENDPTR-01.

    `mempcpy` returns dest + n, `stpcpy` the terminator it wrote and
    `memccpy` one past the byte it stopped at. Each of these calls states
    enough to resolve that end exactly, so the consumer's `+ 8` is at +9 --
    and where the call does NOT state it, the displacement is unknown
    rather than the base."""
    target = _taint_target(tmp_path, shared_bn, _APPEND_END_SRC, "r11span", opt)
    for case, reader in (("endcopy_case", "endcopy_reader"),
                         ("endstring_case", "endstring_reader"),
                         ("endfind_case", "endfind_reader")):
        _leaves, rows, clear = _gmc_rows(shared_bn, target, case)
        assert (reader, 9, True) in rows, (case, rows)
        assert clear is False, case
    _leaves, rows, clear = _gmc_rows(shared_bn, target, "endunknown_case")
    assert ("endunknown_reader", 0, False) in rows, rows
    assert clear is False


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_a_fortified_fgets_hands_its_destination_back(tmp_path, shared_bn, opt):
    """BN23-FORTIFY-01.

    `__fgets_chk` returns its destination exactly as `fgets` does, so a
    guarded `p + 8` after a two-byte refill still names byte 8 of the
    object the root copied into."""
    target = _taint_target(tmp_path, shared_bn, _FORTIFIED_SRC, "r11fort", opt)
    for case, reader in (("fortified_case", "fortified_reader"),
                         ("tail_case", "tail_reader"),
                         ("plain_case", "plain_reader")):
        _leaves, rows, clear = _gmc_rows(shared_bn, target, case)
        assert (reader, 8, True) in rows, (case, rows)
        assert clear is False, case
    # a return nothing uses reads nothing
    leaves, _rows, clear = _gmc_rows(shared_bn, target, "ignored_case")
    assert leaves == [] and clear is True, leaves


def test_a_structured_reseed_command_runs_where_it_was_produced(tmp_path,
                                                                shared_bn):
    """BN23-ADVICE-01.

    `leaves[].reseed` is an executable command, not a locator: a caller
    that selected an instance and a target explicitly must be able to run
    the string it is handed. With a second bridge alive the unrouted
    spelling refuses, which is what this proves the routed one does not.
    """
    # A reader that LOADS the global: its reseed seeds the loaded value (a
    # reader that hands the object straight to a sink has none, W07-13).
    target = _taint_target(tmp_path, shared_bn, _NAMED_GLOBAL_SRC, "r11seed", "-O0")
    env = shared_bn.env()
    started = subprocess.run(
        [*_BN_CLI, "session", "start", str(tmp_path / "r11seed0"),
         "--format", "json"],
        capture_output=True, text=True, timeout=180.0, env=env)
    assert started.returncode == 0, started.stderr
    other = json.loads(started.stdout)["instance_id"]
    try:
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", "fill_count", "--source", "call:read")
        reseed = next(lf["reseed"] for lf in result["leaves"] if lf.get("reseed"))
        command = shlex.split(reseed.strip("`"))
        assert command[0] == "bn", reseed
        assert shared_bn.instance_id in command, reseed

        ran = subprocess.run([*_BN_CLI, *command[1:]], capture_output=True,
                             text=True, timeout=180.0, env=env)
        assert ran.returncode == 0, f"{reseed}\n{ran.stderr}\n{ran.stdout}"

        # ...and the same command without this call's routing cannot pick a
        # bridge at all, which is the failure the routing prevents.
        bare = [word for index, word in enumerate(command[1:])
                if word not in (shared_bn.instance_id, target, "-i", "-t")
                and command[1:][index - 1] not in ("-i", "-t")]
        refused = subprocess.run([*_BN_CLI, *bare], capture_output=True,
                                 text=True, timeout=180.0, env=env)
        assert refused.returncode == 2, (bare, refused.stdout, refused.stderr)
    finally:
        subprocess.run(
            [*_BN_CLI, "session", "stop", other, "--discard-unsaved"],
            capture_output=True, text=True, timeout=60.0, env=env)


#: A `-static` cross build is the point of this fixture: a dynamically
#: linked one has no in-binary body for the library function the model also
#: covers, which is the whole shape under test.
_CROSS_ANALYSIS_TIMEOUT = 300.0

_WIDE_AARCH64_SRC = r"""
#include <string.h>
#include <wchar.h>
#define N __attribute__((noinline,noclone))
char tracked[256] __attribute__((section(".tracked"))) = "seed";
char output[256] __attribute__((section(".out"))) = "out";
N void byte88_reader(void) { output[0] = tracked[88]; }
N void wide_write_case(char *s) { wmemcpy((wchar_t *)tracked, (wchar_t *)s, 32);
                                  byte88_reader(); }
N void project_helper(char *d, const char *s) { d[0] = s[0]; d[88] = s[1]; }
N void helper_case(char *s) { project_helper(tracked, s); byte88_reader(); }
int main(void) { return 0; }
"""


def _aarch64_binary(tmp_path, opt: str):
    """Cross-build the wide-copy probe, or refuse to disappear quietly."""
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc not available, so the AArch64 taint determinism "
            "guard did not run; install the cross toolchain (Debian/Ubuntu: "
            "gcc-aarch64-linux-gnu)")
    src = tmp_path / f"r12wide{opt[-1]}.c"
    src.write_text(_WIDE_AARCH64_SRC)
    binary = tmp_path / f"r12wide{opt[-1]}"
    build = subprocess.run(
        [cc, opt, "-g0", "-U_FORTIFY_SOURCE", "-fno-builtin", "-fno-inline",
         "-no-pie", "-static", str(src), "-o", str(binary)],
        capture_output=True, text=True, timeout=180)
    if build.returncode != 0:
        raise FixtureBuildError(
            f"aarch64 cross-compile failed (exit {build.returncode}): {build.stderr}")
    return binary


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_repeated_taint_requests_on_a_fresh_view_agree(tmp_path, shared_bn, opt):
    """BN25-META-01, on the architecture and build shape that found it.

    A `-static` AArch64 build has an in-binary `__wmemcpy` stub, and Binary
    Ninja only decides that it IS a stub while it lifts it. The FIRST
    request on a freshly analysed view therefore used to descend into it
    and raise an `unmodeled_callee` frontier for the very call whose model
    it had just applied, while every later request resolved the thunk and
    did not -- so an identical query answered differently depending on what
    had run before, and the text spelling (asked second) disagreed with the
    JSON one.

    The first request here is the TEXT one, deliberately: on a cold view it
    is the one that would have seen the extra leaf.
    """
    binary = _aarch64_binary(tmp_path, opt)
    target = shared_bn.load(binary, copy=False, timeout=_CROSS_ANALYSIS_TIMEOUT)
    args = ("-t", target, "taint", "forward", "-f", "wide_write_case",
            "--source", "param:0")

    rendered = shared_bn.run(*args, timeout=180.0)
    assert rendered.returncode == 0, rendered.stderr
    raw = [shared_bn.run(*args, "--format", "json", timeout=180.0)
           for _ in range(3)]
    for one in raw:
        assert one.returncode == 0, one.stderr
    # the raw bridge result, byte for byte, for three identical requests
    assert raw[0].stdout == raw[1].stdout == raw[2].stdout, [r.stdout for r in raw]

    result = json.loads(raw[0].stdout)
    kinds = sorted(lf["kind"] for lf in result["leaves"])
    # the modeled copy is recognised AS modeled...
    assert any("wmemcpy" in a and "propagate" in a
               for a in result["assumptions"]), result["assumptions"]
    # ...so it is not ALSO reported as an unmodeled callee
    assert kinds == ["global_memory_continuation"], result["leaves"]
    assert result["diagnostics"]["safe_to_report_all_clear"] is False
    assert result["diagnostics"]["frontier"]["unresolved"] == 0

    # ...and the text spelling asked FIRST says the same thing
    assert f"frontiers ({len(kinds)})" in rendered.stdout, rendered.stdout
    assert f"{len(kinds)} tainted frontier(s)" in rendered.stdout, rendered.stdout
    assert "unmodeled call(s) reached: no" in rendered.stdout, rendered.stdout
    for kind in set(kinds):
        assert kind in rendered.stdout, (kind, rendered.stdout)


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"], ids=["O0", "O1", "O2"])
def test_only_a_models_return_rule_answers_what_a_call_hands_back(tmp_path,
                                                                  shared_bn,
                                                                  opt):
    """BN25-META-01's second half, scoped.

    The `return_value_unrecovered` frontier exists because Binary Ninja
    recovered no result for the call, so the return channel is unknown.
    A model answers that ONLY by writing `ret`/`*ret`: a sink class says
    what is dangerous to pass IN, and an argument-only source or
    propagation describes the buffers -- neither says what comes back, and
    taking either for an answer turns a required frontier into an
    all-clear.
    """
    binary = _aarch64_binary(tmp_path, opt)
    target = shared_bn.load(binary, copy=False, timeout=_CROSS_ANALYSIS_TIMEOUT)

    def overlay(name, body):
        path = tmp_path / f"{name}{opt[-1]}.json"
        path.write_text(json.dumps({"project_helper": body}))
        return ["--models", str(path)]

    def dropped(fn, *extra):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward",
                          "-f", fn, "--source", "param:0", *extra)
        return result, [lf for lf in result["leaves"]
                        if lf.get("cause") == "return_value_unrecovered"]

    # no model, a sink-only model, and argument-only rules all leave the
    # return unknown: the frontier stands and the run is not an all-clear
    for label, extra in (
            ("no model", []),
            ("sink only", overlay("sink", {
                "sink": {"tainted_args": [0], "class": "command_injection",
                         "detail": "tainted command to project_helper"}})),
            ("arg-only source", overlay("argsrc", {
                "sources": [{"to": "*arg:0"}]})),
            ("arg-only propagate", overlay("argprop", {
                "propagates": [{"from": "*arg:1", "to": "*arg:0"}]}))):
        result, leaves = dropped("helper_case", *extra)
        assert leaves, (label, result["leaves"])
        assert result["diagnostics"]["safe_to_report_all_clear"] is False, label

    # ...while a rule that DOES define the return channel answers it
    for label, extra in (
            ("arg -> ret", overlay("argret", {
                "propagates": [{"from": "arg:0", "to": "ret"}]})),
            ("source -> ret", overlay("srcret", {"sources": [{"to": "ret"}]}))):
        _result, leaves = dropped("helper_case", *extra)
        assert leaves == [], (label, leaves)

    # ...which is what the builtin `wmemcpy` rules do for the modeled copy,
    # so its frontier stays suppressed with no overlay in play
    result, leaves = dropped("wide_write_case")
    assert leaves == [], result["leaves"]
    assert any("wmemcpy" in a and "propagate" in a
               for a in result["assumptions"]), result["assumptions"]


def test_reached_model_rule_into_unrecovered_destination_is_a_boundary(tmp_path, shared_bn):
    """BN6-C009: a model rule `arg:0 -> *arg:1` reached at a call for which BN
    recovered ONE argument used to drop its declared output silently -- no
    leaf, all-clear. It is a blocking `model_destination_unrecovered` leaf that
    names the callee, call and destination, guesses no pointer, and prints
    reads that run as printed. A fully recovered keyable output keeps
    propagating to its sink with no such leaf."""
    src = tmp_path / "model_dest.c"
    src.write_text("""
#include <stddef.h>
extern void model_emit(size_t input);
extern void model_fill(size_t input, char *out);
extern int system(const char *);
__attribute__((noinline)) void call_model_emit(size_t input) { model_emit(input); }
__attribute__((noinline)) void call_model_fill(size_t input) {
    char out[32]; model_fill(input, out); system(out);
}
""")
    binary = tmp_path / "model_dest.so"
    _compile(src, binary, ["-shared", "-fPIC", "-fno-inline"])
    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        "model_emit": {"propagates": [{"from": "arg:0", "to": "*arg:1"}]},
        "model_fill": {"propagates": [{"from": "arg:0", "to": "*arg:1"}]},
    }))
    target = shared_bn.load(binary, copy=False)

    def forward(fn, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                        "--source", "param:0", *extra)

    result = forward("call_model_emit", "--models", str(models))
    [leaf] = [lf for lf in result["leaves"] if lf["kind"] == "model_destination_unrecovered"]
    assert leaf["callee"]["name"] == "model_emit", leaf
    assert leaf["destination"] == "*arg:1" and leaf["recovered_params"] == 1, leaf
    assert [s["from"] for s in leaf["sources"]] == ["arg:0"], leaf
    diag = result["diagnostics"]
    assert diag["safe_to_report_all_clear"] is False, diag
    assert diag["frontier"]["unresolved"] == 1 and diag["unmodeled_calls_reached"] is False, diag
    outputs = _run_diagnostic_commands(shared_bn, diag["next_action"])
    assert any("model_emit" in out for out in outputs), outputs
    assert any(re.search(r"\bcall\b", out) for out in outputs), outputs

    # Without the model nothing claims the destination: no such leaf.
    plain = forward("call_model_emit")
    assert not [lf for lf in plain["leaves"] if lf["kind"] == "model_destination_unrecovered"]

    # Control: both arguments recovered, the keyed buffer reaches system().
    filled = forward("call_model_fill", "--models", str(models))
    assert [f["sink"]["callee"] for f in filled["reached_sinks"]] == ["system"], filled
    assert not [lf for lf in filled["leaves"] if lf["kind"] == "model_destination_unrecovered"]


def test_continuation_into_unrecovered_destination_is_a_boundary(tmp_path, shared_bn):
    """IC04 review: a continuation rule reached through carried state used to
    drop a declared output BN did not recover at the NULL call -- leaves=[],
    all-clear -- even though the walk said the state continued there. The
    later clean reset must not erase that boundary; a reset before the NULL
    call still kills the continuation."""
    src = tmp_path / "continuation.c"
    src.write_text("""
extern void stateful();
__attribute__((noinline)) void continuation_reset(const char *src) {
    char out[32];
    stateful(src, out);
    stateful(0);
    stateful("clean", out);
}
__attribute__((noinline)) void reset_first(const char *src) {
    char out[32];
    stateful(src, out);
    stateful("clean", out);
    stateful(0);
}
""")
    binary = tmp_path / "continuation.so"
    _compile(src, binary, ["-std=gnu17", "-shared", "-fPIC", "-fno-inline"])
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"stateful": {
        "propagates": [{"from": "arg:0", "to": "*arg:1"}], "continuation": {"null_arg": 0}}}))
    target = shared_bn.load(binary, copy=False)
    calls = _bn_json(shared_bn, "-t", target, "callsites", "stateful",
                     "--within", "continuation_reset")["items"]
    null_call = sorted(c["call_addr"] for c in calls)[1]
    first_call = sorted(c["call_addr"] for c in calls)[0]

    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "continuation_reset",
                      "--source", "param:0", "--models", str(models))
    leaves = [lf for lf in result["leaves"] if lf["kind"] == "model_destination_unrecovered"]
    assert [(lf["address"], lf["destination"]) for lf in leaves] == [(null_call, "*arg:1")], \
        result["leaves"]
    assert leaves[0]["sources"][0]["continued_from"] == first_call, leaves
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]
    outputs = _run_diagnostic_commands(shared_bn, result["diagnostics"]["next_action"])
    assert any("stateful" in out for out in outputs), outputs

    killed = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "reset_first",
                      "--source", "param:0", "--models", str(models))
    assert not [lf for lf in killed["leaves"] if lf["kind"] == "model_destination_unrecovered"], \
        killed["leaves"]


@pytest.mark.parametrize("command", [
    ("taint", "forward", "-f", "call_model_emit", "--source", "param:0"),
    ("taint", "models", "--role", "propagator"),
])
def test_malformed_model_tokens_are_refused_by_the_cli(tmp_path, shared_bn, command):
    """IC04 review: `*arg:-1` passed validation and a reached rule wrote
    through the call's LAST argument; `*arg:x` raised mid-walk. Every public
    model path now refuses the file, naming the model, field and grammar."""
    src = tmp_path / "emit.c"
    src.write_text("""
#include <stddef.h>
extern void model_emit(size_t input);
__attribute__((noinline)) void call_model_emit(size_t input) { model_emit(input); }
""")
    binary = tmp_path / "emit.so"
    _compile(src, binary, ["-shared", "-fPIC", "-fno-inline"])
    target = shared_bn.load(binary, copy=False)
    for token in ("*arg:-1", "*arg:x", "*arg:01"):
        models = tmp_path / "bad.json"
        models.write_text(json.dumps(
            {"model_emit": {"propagates": [{"from": "arg:0", "to": token}]}}))
        res = shared_bn.run("-t", target, *command, "--models", str(models))
        assert res.returncode not in (0, 1), (token, res.returncode, res.stdout, res.stderr)
        out = res.stderr + res.stdout
        assert "model 'model_emit'" in out and "`propagates.to`" in out, out
        assert "non-negative decimal argument index" in out, out
        assert "Traceback" not in out and "internal error" not in out, out


def test_scanf_optional_slots_and_repeated_contexts_on_real_bn(tmp_path, shared_bn):
    """Round-2 review. (F3) A fully recovered sscanf call has no missing
    output: the builtin model's unrolled `*arg:2..5` slots past the call are
    optional varargs. Only a format proving more destinations than recovered
    is a boundary. (F4) One missing output reached through two argument
    contexts of the same callee site is ONE leaf with both sources."""
    src = tmp_path / "scan_ctx.c"
    src.write_text("""
#include <stdio.h>
extern void split_emit(const char *a, const char *b);
__attribute__((noinline)) int parse_none(const char *s) { return sscanf(s, "token"); }
__attribute__((noinline)) int parse_one(const char *s) { int a; return sscanf(s, "%d", &a); }
__attribute__((noinline)) int parse_two(const char *s) { int a, b; return sscanf(s, "%d %d", &a, &b); }
__attribute__((noinline)) int parse_four(const char *s) {
    int a, b, c, d; return sscanf(s, "%d %d %d %d", &a, &b, &c, &d);
}
__attribute__((noinline)) int child(const char *a, const char *b) { split_emit(a, b); return 0; }
__attribute__((noinline)) int repeated_contexts(const char *s) {
    return child(s, "clean") + child("clean", s);
}
""")
    binary = tmp_path / "scan_ctx.so"
    _compile(src, binary, ["-shared", "-fPIC", "-fno-inline", "-Wno-format"])
    target = shared_bn.load(binary, copy=False)

    def forward(fn, *extra):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                        "--source", "param:0", *extra)

    for fn in ("parse_none", "parse_one", "parse_two", "parse_four"):
        result = forward(fn)
        kinds = [lf["kind"] for lf in result["leaves"]]
        assert "model_destination_unrecovered" not in kinds, (fn, result["leaves"])
        assert "format_destination_unrecovered" not in kinds, (fn, result["leaves"])

    models = tmp_path / "split.json"
    models.write_text(json.dumps({"split_emit": {"propagates": [
        {"from": "arg:0", "to": "*arg:6"}, {"from": "arg:1", "to": "*arg:6"}]}}))
    result = forward("repeated_contexts", "--models", str(models))
    leaves = [lf for lf in result["leaves"] if lf["kind"] == "model_destination_unrecovered"]
    assert len(leaves) == 1, leaves
    assert [s["from"] for s in leaves[0]["sources"]] == ["arg:0", "arg:1"], leaves
    assert result["diagnostics"]["frontier"]["by_kind"]["model_destination_unrecovered"] == 1


def test_null_varargs_index_is_refused_by_the_cli(tmp_path, shared_bn):
    """Round-2 review: `varargs.first_index: null` loaded, then crashed the
    walk. It is refused at load, by forward and by the catalog alike."""
    src = tmp_path / "fmt.c"
    src.write_text("""
#include <stddef.h>
extern void app_fmt(char *out, const char *fmt, ...);
__attribute__((noinline)) void call_fmt(const char *s) { char b[16]; app_fmt(b, "%s", s); }
""")
    binary = tmp_path / "fmt.so"
    _compile(src, binary, ["-shared", "-fPIC", "-fno-inline"])
    target = shared_bn.load(binary, copy=False)
    models = tmp_path / "bad.json"
    models.write_text(json.dumps({"app_fmt": {"varargs": {"first_index": None, "to": "*arg:0"}}}))
    for command in (("taint", "forward", "-f", "call_fmt", "--source", "param:0"),
                    ("taint", "models", "--role", "propagator")):
        res = shared_bn.run("-t", target, *command, "--models", str(models))
        out = res.stderr + res.stdout
        assert res.returncode == 2, (command, res.returncode, out)
        assert "model 'app_fmt'" in out and "`varargs.first_index` is required" in out, out
        assert "Traceback" not in out and "internal error" not in out, out


_LOG_SERVER_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N __attribute__((noinline, noipa))
char log_buf[32];
char log_ring[2097280];
extern char log_tail[64];
/* An interior symbol a megabyte into the ring: the search for the object
   enclosing a write there runs out of its window (an internal budget). */
__asm__(".globl log_tail\n.set log_tail,log_ring+1048640\n"
        ".type log_tail,@object\n.size log_tail,64\n");
N void flush_log(void) { puts(log_buf); }
N void emit_record(int k) { puts(log_buf + k); }
N void drain_ring(void) {
    char *p = malloc(sizeof log_ring);
    memcpy(p, log_ring, sizeof log_ring);
    puts(p + 1048640);
    free(p);
}
N void consume(const char *p) { if (p[0] == 'x') puts("x"); }
N void serve(char *p) {
    memcpy(log_buf, p, 16);
    memcpy(log_tail, p, 32);
    flush_log();                 /* both globals stay live across four calls */
    emit_record(1);
    emit_record(2);
    drain_ring();
    consume(p);                  /* a call BN gives no result */
    puts("done");
}
int main(int argc, char **argv) { if (argc > 1) serve(argv[1]); return 0; }
"""


def _frontier_blocks(frontier: list[str]) -> list[list[str]]:
    """Rendered frontier rows split into one block per row head (2-space
    indent); wrapped and body lines are indented deeper."""
    blocks: list[list[str]] = []
    for line in frontier[1:]:
        if line.startswith("  ") and not line.startswith("   "):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def test_forward_frontier_verdict_advice_and_text_on_a_real_build(tmp_path, shared_bn,
                                                                monkeypatch):
    """P1-08, P2-04 and the W07/W05 papercuts on one real build.

    Two globals written once stay live across four later calls, so the walk
    records ONE continuation leaf per global whose `sites` name every call
    it stayed live across, and the write at an
    interior symbol deep inside a large object runs the enclosing-object
    search out of its internal budget; a void call with a tainted argument
    gets the return frontier. The text names the cause with its own remedy
    (not `@depth N`), prints ONE block per global with one routed command,
    wraps at the terminal width, and every leaf command runs as printed."""
    from bn.taint_truncation import TRUNCATION_CAUSES

    src = tmp_path / "logsrv.c"
    src.write_text(_LOG_SERVER_SRC)
    binary = tmp_path / "logsrv"
    _compile(src, binary, ["-O2", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0",
                           "-fno-builtin", "-fno-inline", "-g0"])
    target = shared_bn.load(binary, copy=False)
    forward = ("-t", target, "taint", "forward", "-f", "serve", "--source", "param:0")

    result = _bn_json(shared_bn, *forward, "--sink-class", "file_write")
    causes = result["stats"]["truncation_cause"]
    # A truncation that is NOT the depth bound, rendered through the table.
    assert set(causes) - {"max_depth"}, result["stats"]
    assert set(causes) <= set(TRUNCATION_CAUSES), causes
    assert result["run_params"]["sink_classes"] == ["file_write"], result["run_params"]
    assert result["run_params"]["resolve_map"] is None
    assert result["run_params"]["models"] == []
    assert all("note" not in leaf for leaf in result["leaves"]), result["leaves"]
    gmc = [lf for lf in result["leaves"] if lf["kind"] == "global_memory_continuation"]
    sites: dict[str, list] = {}
    for lf in gmc:
        assert lf["global"]["address"] not in sites, gmc     # one leaf per global
        sites[lf["global"]["address"]] = lf["sites"]
        first = lf["sites"][0]
        assert (lf["address"], lf["callee"]) == (first["address"], first["callee"]), lf
    assert len(sites) == 2 and min(len(s) for s in sites.values()) >= 2, gmc

    # The return frontier points at a bounded window AT the call, and it runs.
    [lost] = [lf for lf in result["leaves"] if lf.get("cause") == "return_value_unrecovered"]
    [command] = re.findall(r"`bn ([^`]+)`", lost["detail"])
    words = shlex.split(command)
    assert "disasm" in words and lost["address"] in words and "--linear" in words, command
    assert _run_or_check_template(shared_bn, lost["detail"]) == (1, 0)

    monkeypatch.setenv("COLUMNS", "100")
    rendered = shared_bn.run(*forward, timeout=120.0)
    assert rendered.returncode == 0, rendered.stderr
    lines = rendered.stdout.splitlines()
    verdict = next(line for line in lines if line.startswith("verdict:"))
    assert "@depth" not in verdict, verdict
    for cause in causes:
        assert TRUNCATION_CAUSES[cause].what in verdict, (cause, verdict)
        assert TRUNCATION_CAUSES[cause].remedy in verdict, (cause, verdict)
    start = next(i for i, line in enumerate(lines) if line.startswith("frontiers ("))
    end = next((i for i in range(start, len(lines)) if not lines[i]), len(lines))
    frontier = lines[start:end]
    # Only a row's fact head may run past the width; its prose wraps.
    assert all(len(line) <= 100 or "`" in line or len(line.split("  -- ")[0]) > 100
               for line in frontier), frontier
    blocks = [b for b in _frontier_blocks(frontier)
              if "global_memory_continuation @" in b[0]]
    # One block per global -- one leaf, so no `[N leaves]` count -- listing
    # every call that global crossed, with exactly one follow-up command.
    assert len(blocks) == len(sites), frontier
    for address, crossed in sites.items():
        [block] = [b for b in blocks if f"global={address}" in b[0]]
        assert " leaves]" not in block[0], block
        text = " ".join(" ".join(block).split())
        for site in crossed:
            assert f"{site['address']} -> {site['callee']['display_name']}" in text, \
                (site, block)
        assert len(re.findall(r"`bn [^`]+`", "\n".join(block))) == 1, block
    # Every command the text prints carries this invocation's routing, and
    # the frontier block's own command runs as printed.
    printed = re.findall(r"`(bn [^`]+)`", rendered.stdout)
    assert printed
    for line in printed:
        assert f"-i {shared_bn.instance_id}" in line and "-t " in line, line
    assert _run_or_check_template(shared_bn, "\n".join(blocks[0]))[0] == 1

    # A depth cut names the flag that exists.
    capped = _bn_json(shared_bn, *forward, "--max-depth", "0")
    cut = [lf for lf in capped["leaves"] if "depth bound" in str(lf.get("detail"))]
    assert cut, capped["leaves"]
    for leaf in cut:
        assert "--max-depth" in leaf["detail"] and " --depth" not in leaf["detail"], leaf

    # --estimate-output never offers --max-depth as the way to bound the read.
    est = shared_bn.run(*forward, "--estimate-output", "--format", "json", timeout=120.0)
    assert est.returncode == 0, est.stderr
    rerun = json.loads(est.stdout)["rerun"]
    assert "--out" in rerun and "smaller" not in rerun, rerun


def test_models_catalog_on_an_all_unnamed_target_marks_no_synthetic_builtin(tmp_path,
                                                                            shared_bn):
    """W04-08 / W12-05: Binary Ninja adds a `.synthetic_builtins` placeholder
    for every compiler builtin to each view. On a fully stripped freestanding
    build no code references them, so no `__builtin_*` model is present, the
    count says no modeled name matched, and text prints the hint."""
    src = tmp_path / "bare.c"
    src.write_text(r"""
void _start(void) {
    __asm__ volatile("mov $60, %eax\n\txor %edi, %edi\n\tsyscall");
}
""")
    binary = tmp_path / "bare"
    _compile(src, binary, ["-nostdlib", "-static", "-s", "-ffreestanding"])
    target = shared_bn.load(binary, copy=False)
    catalog = _bn_json(shared_bn, "-t", target, "taint", "models", "--present")
    assert catalog["presence_catalog"] is True
    assert catalog["present_model_count"] == 0, catalog
    present = [e for e in catalog["items"] if e.get("present")]
    assert not any(str(e.get("symbol", "")).startswith("__builtin_") for e in present), present
    text = shared_bn.run("-t", target, "taint", "models", "--present")
    assert text.returncode == 0, text.stderr
    assert "no modeled name matched" in text.stdout, text.stdout


def test_a_tail_call_wrapper_flagged_a_thunk_is_walked(tmp_path, shared_bn):
    """P2-03: at -O2 every wrapper below is one block ending in a tail call,
    so Binary Ninja flags it `is_thunk` -- and taint skipped each body with a
    prose caveat, missing the fortified copy in all three and reporting
    nothing from `main`'s `read`."""
    binary = tmp_path / "fwd_tail_wrappers"
    _compile(CORPUS / "fwd_tail_wrappers.c", binary,
             ["-O2", "-fno-inline-small-functions", "-D_FORTIFY_SOURCE=2"])
    target = shared_bn.load(binary, copy=False)
    for fn, sources in (("dispatch_frame", ("param:0", "param:1")), ("main", ("call:read",))):
        args = ["-t", target, "taint", "forward", "-f", fn]
        for source in sources:
            args += ["--source", source]
        result = _bn_json(shared_bn, *args)
        reached = {s["sink"]["callee"] for s in result["reached_sinks"]}
        assert {"__memcpy_chk", "__memmove_chk", "__strncpy_chk"} <= reached, (fn, reached)
        assert not any("thunk" in a for a in result["assumptions"]), result["assumptions"]


def test_a_deeper_budget_keeps_the_copy_a_shallower_one_reached(tmp_path, shared_bn):
    """V5-01: `format_name` appends its tainted argument to a static buffer
    at `strlen` of it and returns the buffer; the recursive `walk` copies the
    result into a fixed-size global with `__strcpy_chk`. At a depth that cuts
    `format_name` off its return is assumed tainted and the copy is reached;
    every deeper budget descended `format_name`, found the write through
    `buf + n` unkeyed, returned a clean pointer and lost the copy."""
    binary = tmp_path / "fwd_depth_static_name"
    _compile(CORPUS / "fwd_depth_static_name.c", binary,
             ["-O2", "-fno-inline", "-D_FORTIFY_SOURCE=2"])
    target = shared_bn.load(binary, copy=False)
    reached = {}
    for depth in (1, 2, 3, 4, 8):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                          "--source", "param:1", "--max-depth", str(depth))
        reached[depth] = {(s["sink"]["callee"], s["sink"]["address"])
                          for s in result["reached_sinks"]}
    depths = sorted(reached)
    for shallow, deep in zip(depths, depths[1:]):
        assert reached[shallow] <= reached[deep], (shallow, deep, reached)
    assert "__strcpy_chk" in {callee for callee, _ in reached[8]}, reached


def test_an_unbounded_index_into_a_global_keeps_the_out_of_object_frontier(tmp_path,
                                                                          shared_bn):
    """D1-01 on `taint_corpus/fwd_indexed_global.c`: `slots[i] = v[0];
    system(command)` with `command` the declared object beside `slots`. Keying
    the write to `slots` alone answered `safe_to_report_all_clear: true` with
    no leaf, although the index may put the byte in `command`. A masked index
    BN bounds inside `slots` raises no such frontier."""
    binary = tmp_path / "fwd_indexed_global"
    _compile(CORPUS / "fwd_indexed_global.c", binary, ["-O1", "-fno-inline"])
    target = shared_bn.load(binary, copy=False)

    def index_frontiers(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", "param:0")
        return result, [lf for lf in result["leaves"]
                        if lf["kind"] == "coarse_memory_store"
                        and lf.get("index_bounded") is False]

    unbounded, frontiers = index_frontiers("stash")
    assert [lf["global"]["label"] for lf in frontiers] == ["slots"], unbounded["leaves"]
    assert unbounded["diagnostics"]["safe_to_report_all_clear"] is False

    masked, frontiers = index_frontiers("stash_masked")
    assert frontiers == [], masked["leaves"]


def test_a_sink_reached_only_through_an_object_tainted_at_an_unknown_offset_says_so(
        tmp_path, shared_bn):
    """V6-01 on `taint_corpus/fwd_object_wide.c`: `rec.tag[i] = v[0]` taints the
    record at an unknown offset, and the copy of `rec.note` that follows was a
    flow indistinguishable from one into `rec.note` itself. It is kept and says
    so: `sink.object_taint` names the store and the field read, and the text
    flow line is tagged. A copy of `rec.note` into another global is no more
    certain than `rec.note` (G2c). Copying the string into `rec.note` and back
    out is field-precise and carries neither."""
    binary = tmp_path / "fwd_object_wide"
    # -fno-builtin: GCC inlines the fixed-size memcpy even at -O0.
    _compile(CORPUS / "fwd_object_wide.c", binary, ["-fno-builtin"])
    target = shared_bn.load(binary, copy=False)
    stamp = _bn_json(shared_bn, "-t", target, "function", "info", "stamp")
    span = range(int(stamp["span_start"], 16), int(stamp["span_end"], 16))

    def sinks(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", "param:0")
        return [hit["sink"] for hit in result["reached_sinks"]]

    for fn in ("object_wide", "object_wide_returned", "object_wide_copied"):
        [sink] = sinks(fn)
        tag = sink["object_taint"]
        assert (tag["kind"], tag["field_read"]) == ("unknown_offset", 32), sink
        assert int(tag["store"], 16) in span, (tag, stamp["span_start"])

    precise = sinks("field_precise")
    assert len(precise) == 2 and not [s for s in precise if "object_taint" in s], precise

    text = shared_bn.run("-t", target, "taint", "forward", "-f", "object_wide",
                         "--source", "param:0")
    assert text.returncode == 0, text.stderr
    flows = [ln.split() for ln in text.stdout.splitlines() if ln.startswith("  [")]
    assert len(flows) == 1 and "[object-wide]" in flows[0], text.stdout



def test_a_loop_count_length_is_a_recurrence_not_a_constant(tmp_path, shared_bn):
    """IC08 on `taint_corpus/bw_loop_count.c`: a length `k` that starts at 0
    and grows on loop trips the comma test picks was a `constant 0` origin
    of a complete slice; it is a `recurrence_initializer` (value 0,
    increments 0 or +1) that withholds completeness. So is one counted in a
    nested loop, one updated through a call, and one a call may reset on
    another back edge (E2c: `constant 0`, or only the call, and complete).
    Two literals chosen by a branch stay `constant` origins of a complete
    slice (control)."""
    binary = tmp_path / "bw_loop_count"
    _compile(CORPUS / "bw_loop_count.c", binary)
    target = shared_bn.load(binary, copy=False)

    def slice_of(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", fn,
                          "--sink", "arg:memcpy:2")
        return result, [o for sl in result["slices"] for o in sl["origins"]]

    counted, origins = slice_of("count_fields")
    assert [(o["kind"], o.get("value"), o.get("increments")) for o in origins] == [
        ("recurrence_initializer", 0, [0, 1])], origins
    assert counted["diagnostics"]["safe_to_report_complete_slice"] is False

    # A nested count, a call-defined update (named) and a resettable count
    # each keep the recurrence beside whatever else the slice reaches.
    for fn, others, calls in (("count_cells", set(), None),
                              ("scale_rounds", {"call"}, ["next_round"]),
                              ("count_or_reset", {"call"}, None)):
        result, origins = slice_of(fn)
        [rec] = [o for o in origins if o["kind"] == "recurrence_initializer"]
        assert rec["value"] == 0 and rec.get("calls") == calls, (fn, rec)
        assert {o["kind"] for o in origins} - {"recurrence_initializer"} == others, (fn, origins)
        assert result["diagnostics"]["safe_to_report_complete_slice"] is False, fn

    fixed, origins = slice_of("fixed_header")
    assert origins and {o["kind"] for o in origins} == {"constant"}, origins
    assert {o["value"] for o in origins} <= {8, 16}
    assert fixed["diagnostics"]["safe_to_report_complete_slice"] is True


def _armv7_build(tmp_path, name: str) -> Path:
    cc = shutil.which("arm-linux-gnueabihf-gcc")
    if cc is None:
        refuse_silent_skip(
            "arm-linux-gnueabihf-gcc is required for the armv7 ABI-mapping fixture; "
            "install the cross toolchain (Debian/Ubuntu: gcc-arm-linux-gnueabihf)")
    binary = tmp_path / name
    _build([cc, "-O2", "-mthumb", "-g", "-fno-stack-protector",
            str(CORPUS / f"{name}.c"), "-o", str(binary)])
    return binary


def test_armv7_argument_mapping_is_disclosed_never_complete(tmp_path, shared_bn):
    """P2-02 / P2-07 on armv7: BN maps a `long long` passed in r2:r3 to a
    stack slot. The backward slice that ends at `entry_r2` was a complete
    slice; the forward flow into the same register vanished; and a Thumb
    `push {r3, lr}` alignment slot was read as an under-recovered argument."""
    target = shared_bn.load(_armv7_build(tmp_path, "abi_armv7"), copy=False)

    backward = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "put_wide",
                        "--sink", "arg:malloc:0")
    [leaf] = [lf for lf in backward["leaves"] if lf["kind"] == "abi_mapping_unbound"]
    assert leaf["register"] == "r2", leaf
    assert "abi_mapping_unbound" in {o["kind"] for sl in backward["slices"] for o in sl["origins"]}
    assert backward["diagnostics"]["safe_to_report_complete_slice"] is False

    forward = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "pass_wide",
                       "--source", "param:1")
    under = [lf for lf in forward["leaves"] if lf.get("cause") == "arg_under_recovered"]
    assert [(lf["callee"]["name"], lf["registers"]) for lf in under] == [("put_wide", ["r2"])], under
    assert forward["diagnostics"]["safe_to_report_all_clear"] is False

    padded = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "check_args",
                      "--source", "param:3")
    assert not [lf for lf in padded["leaves"] if lf.get("cause") == "arg_under_recovered"], \
        padded["leaves"]


def test_armv7_ifunc_copy_with_a_two_path_length_is_routed(tmp_path, shared_bn):
    """W06-05: glibc's armv7 `memcpy` PLT stub carries the IFUNC resolver's
    one-argument prototype, so the copy's length is read from r2 -- which a
    Thumb `itee` block fills on two paths, an LLIL phi with no MLIL of its
    own. That length was neither reached forward nor seeded backward."""
    target = shared_bn.load(_armv7_build(tmp_path, "abi_armv7"), copy=False)

    forward = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "flush_window",
                       "--source", "param:2")
    copies = [s["sink"] for s in forward["reached_sinks"] if s["sink"]["callee"] == "memcpy"]
    assert [(c["tainted_arg_index"], c.get("via")) for c in copies] == [(2, None)], copies

    backward = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "flush_window",
                        "--sink", "arg:memcpy:2")
    assert [s["status"] for s in backward["sink_status"]] == ["seeded"], backward["sink_status"]
    assert not [lf for lf in backward["leaves"] if lf["kind"] == "sink_unseeded"]
    assert "parameter" in {o["kind"] for sl in backward["slices"] for o in sl["origins"]}


_FRONTIER_SRC = r"""
#include <alloca.h>
#include <stdio.h>
#include <string.h>

struct packet {
    unsigned int kind;
    unsigned int len;
    char pad[48];
    const char *data;
};

unsigned int g_cap;

__attribute__((noinline)) void copy_field(char *dst, const struct packet *p) {
    memcpy(dst, p->data, p->len);
}

__attribute__((noinline)) void copy_capped(char *dst, const char *src) {
    memcpy(dst, src, g_cap);
}

__attribute__((noinline)) void stage(const char *where, const struct packet *p) {
    if (strlen(where) < 4)
        return;
    unsigned long n = p->len;
    char *tmp = alloca(n + 1);
    memcpy(tmp, where, n);
    tmp[n] = 0;
    puts(tmp);
}

int main(int argc, char **argv) {
    struct packet pk = {1, 8, {0}, argv[0]};
    char buf[64];
    g_cap = argc > 1 ? 16 : 8;
    copy_field(buf, &pk);
    copy_capped(buf, argv[0]);
    stage(argv[0], &pk);
    return 0;
}
"""


@pytest.fixture
def frontier_target(tmp_path, shared_bn):
    src = tmp_path / "frontiers.c"
    src.write_text(_FRONTIER_SRC)
    binary = tmp_path / "frontiers"
    # Stack-clash probes are what `alloca` emits under this flag (and by
    # default on several distributions' compilers).
    _compile(src, binary, ["-fstack-clash-protection"])
    return shared_bn.load(binary, copy=False)


def _memcpy_site(shared_bn, target, within):
    [row] = _bn_json(shared_bn, "-t", target, "callsites", "memcpy", "--within", within)["items"]
    return row["call_addr"]


def test_trace_reports_a_typed_field_loads_member_offset(frontier_target, shared_bn):
    """W09-01: `memcpy(dst, p->data, p->len)` -- the length is the `len` member
    at offset 4, read by a typed field load whose offset BN keeps on the load
    instruction; it was reported as offset 0x0."""
    site = _memcpy_site(shared_bn, frontier_target, "copy_field")
    result = _bn_json(shared_bn, "-t", frontier_target, "trace", "copy_field", site,
                      "--arg", "2")
    [load] = [s for s in result["trace"] if s.get("reason") == "field_load"]
    assert (load["offset"], load["width"], load["terminates"]) == ("0x4", 4, True), load
    assert "field_load" in {g["reason"] for g in result["frontiers"]}, result["frontiers"]


def test_trace_stops_at_a_global_load_as_a_named_frontier(frontier_target, shared_bn):
    """W09-02: a length read from a global used to end the slice with no
    frontier at all. It is a `global_load` naming the global, rolled up, with
    an xrefs command for its writers that runs as printed."""
    site = _memcpy_site(shared_bn, frontier_target, "copy_capped")
    result = _bn_json(shared_bn, "-t", frontier_target, "trace", "copy_capped", site,
                      "--arg", "2")
    [load] = [s for s in result["trace"] if s.get("reason") == "global_load"]
    assert load["global_name"] == "g_cap", load
    assert [g["reason"] for g in result["frontiers"]] == ["global_load"], result["frontiers"]
    xrefs = _bn_json(shared_bn, "-t", frontier_target, "xrefs", load["global_address"])
    assert any("main" in json.dumps(row) for row in xrefs["items"]), xrefs


def test_backward_slice_ignores_alloca_stack_probes(frontier_target, shared_bn):
    """W09-07: the stack-clash probes `alloca` emits (`or qword [rsp+..], 0`)
    store back the bytes they load. They were reported as stores that MAY
    write the parameter's object and the loaded bytes; they write nothing."""
    result = _bn_json(shared_bn, "-t", frontier_target, "taint", "backward", "-f", "stage",
                      "--sink", "arg:memcpy:1", timeout=300.0)
    il = _bn_json(shared_bn, "-t", frontier_target, "function", "structured-il", "stage")
    probes = {ins["address"] for ins in il["instructions"]
              if ins["op"] == "MLIL_STORE_SSA"
              and ins["text"].split(" = ", 1)[0].split(" @ ", 1)[0]
              == ins["text"].split(" = ", 1)[1].split(" @ ", 1)[0]}
    assert probes, "the build emitted no stack probe; the case is not exercised"
    rows = [(st.get("address"), st.get("reason") or "")
            for sl in result["slices"] for st in sl["slice"]]
    assert not [r for r in rows if r[0] in probes], rows


_NARROW_LENGTH_SOURCE = """
extern long read(int, void *, unsigned long);
extern void *calloc(unsigned long, unsigned long);
__attribute__((noinline)) char *load_frame(int fd) {
    unsigned int len;
    if (read(fd, &len, sizeof(len)) != (long)sizeof(len))
        return 0;
    char *body = calloc(1, len + 1);
    if (!body)
        return 0;
    read(fd, body, len);
    return body;
}
int main(void) { char *p = load_frame(0); return p ? p[0] : 0; }
"""


@pytest.mark.parametrize("arch", ["a64", "x64"])
def test_a_32_bit_length_add_widened_into_an_allocator_is_disclosed(tmp_path, shared_bn, arch):
    """W10-04: `calloc(1, len + 1)` with a 32-bit `len` computes the `+ 1` in
    32 bits and zero-extends it into the 64-bit size, so `len = 0xffffffff`
    allocates 0 bytes. The reached alloc_size sink names that narrowing."""
    binary = _build_arch(tmp_path, arch, "narrow_len", _NARROW_LENGTH_SOURCE)
    target = shared_bn.load(binary, copy=False)
    start = int(_resolve_addr(shared_bn, "load_frame"), 16)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "load_frame",
                      "--source", "call:read")

    sinks = [s["sink"] for s in result["reached_sinks"]
             if s["sink"]["callee"] == "calloc" and s["sink"]["tainted_arg_index"] == 1]
    assert len(sinks) == 1, result["reached_sinks"]
    sink = sinks[0]
    assert sink["class"] == "alloc_size"
    narrow = sink["width_narrowing"]
    assert (narrow["computed_width"], narrow["parameter_width"]) == (4, 8), narrow
    assert start <= int(narrow["at"], 16) < int(sink["address"], 16), (hex(start), narrow, sink)

    text = shared_bn.run("-t", target, "taint", "forward", "-f", "load_frame",
                         "--source", "call:read", timeout=120.0)
    assert text.returncode == 0, text.stderr
    shown = [ln for ln in text.stdout.splitlines()
             if ln.strip().startswith("width narrowing:")]
    assert len(shown) == 1 and narrow["at"] in shown[0], text.stdout


_FILL_COPY_SOURCE = """
extern long recv(int, void *, unsigned long, int);
extern void *calloc(unsigned long, unsigned long);
extern void *memcpy(void *, const void *, unsigned long);
__attribute__((noinline)) char *take_reply(int sock) {
    char scratch[2048];
    int got = recv(sock, scratch, sizeof(scratch), 0);
    GUARD
    char *copy = calloc(1, got);
    if (!copy)
        return 0;
    memcpy(copy, scratch, got);
    return copy;
}
int main(void) { char *p = take_reply(0); return p ? p[0] : 0; }
"""


def _fill_copy_sink(tmp_path, shared_bn, arch, guard):
    source = _FILL_COPY_SOURCE.replace("GUARD", guard)
    binary = _build_arch(tmp_path, arch, "fill_copy", source)
    target = shared_bn.load(binary, copy=False)
    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "take_reply",
                      "--source", "call:recv")
    copies = [s["sink"] for s in result["reached_sinks"] if s["sink"]["callee"] == "memcpy"]
    assert len(copies) == 1, result["reached_sinks"]
    return copies[0]


@pytest.mark.parametrize("arch", ["a64", "x64"])
def test_a_fill_return_copied_back_out_of_its_own_buffer_is_not_an_overread(
        tmp_path, shared_bn, arch):
    """W01-10: `got = recv(s, scratch, sizeof scratch, 0); if (got <= 0)
    return 0;` then `memcpy(calloc(1, got), scratch, got)` reads back only the
    bytes recv wrote into `scratch`: the destination-bounded copy stays
    bounded_len and names the fill, instead of an overread_len of `scratch`."""
    sink = _fill_copy_sink(tmp_path, shared_bn, arch, "if (got <= 0) return 0;")
    assert sink["class"] == "bounded_len", sink
    assert sink["src_fill"]["callee"] == "recv", sink
    assert int(sink["src_fill"]["at"], 16) < int(sink["address"], 16), sink


@pytest.mark.parametrize("arch", ["a64", "x64"])
@pytest.mark.parametrize("guard", ["", "if (got == -1) return 0;"],
                         ids=["unchecked", "minus-one-only"])
def test_an_unchecked_fill_return_copied_back_out_of_its_buffer_overreads_it(
        tmp_path, shared_bn, arch, guard):
    """R3-01: without a check BN narrows to a non-negative range, recv's -1
    reaches the copy as SIZE_MAX, so `scratch` is over-read: no `src_fill`."""
    sink = _fill_copy_sink(tmp_path, shared_bn, arch, guard)
    assert sink["class"] == "overread_len", sink
    assert "src_fill" not in sink, sink
    assert sink["src_capacity_source"], sink


_ARGV_COPY_SOURCE = """
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    char name[32];
    if (argc > 1) {
        memcpy(name, argv[1], (size_t)argc);
        puts(name);
    }
    return 0;
}
"""


@pytest.mark.parametrize("pie", [False, True], ids=["no-pie", "pie"])
def test_backward_into_main_classifies_the_runtime_entry_handoff(tmp_path, shared_bn, pie):
    """W07-05: a backward slice from a sink in `main` whose length is the
    argument count ascends into main's references. The ELF entry stub's
    handoff of `main` to the imported C-runtime start routine is a `crt_entry`
    row -- still withholding completeness -- not an `indirect_reference`, and
    it adds no "slice backward from the dispatching caller" xrefs advice."""
    src = tmp_path / "argv_copy.c"
    src.write_text(_ARGV_COPY_SOURCE)
    binary = tmp_path / ("argv_copy_pie" if pie else "argv_copy")
    _compile(src, binary, ["-fno-builtin", *(["-fPIE", "-pie"] if pie else [])])
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "main",
                      "--sink", "arg:memcpy:2")
    assert [(o["kind"], o.get("index")) for sl in result["slices"]
            for o in sl["origins"]] == [("parameter", 0)], result["slices"]
    callers = [lf for lf in result["leaves"] if lf["kind"] == "caller_sites_truncated"]
    rows = [row for lf in callers for row in lf["unfollowed"]]
    assert rows, result["leaves"]
    assert {row["reason"] for row in rows} == {"crt_entry"}, rows
    assert {row["caller"] for row in rows} == {"_start"}, rows
    assert [lf["reasons"] for lf in callers] == [{"crt_entry": len(rows)}]
    # Still not a complete slice: what the runtime passes is not proven.
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    # No xrefs/callsites command for a caller the binary does not contain.
    commands = [shlex.split(c) for c in re.findall(r"`(bn [^`]+)`",
                                                    result["diagnostics"]["next_action"])]
    assert not [c for c in commands if "xrefs" in c or "callsites" in c], commands


_NAMED_GLOBAL_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N __attribute__((noinline))
char g_text[64];
int g_count;
N void text_reader(void) { char d[16]; strcpy(d, g_text); puts(d); }
N __attribute__((optimize("O0"))) void count_reader(void) {
    int n = g_count;
    char *p = malloc(n);
    puts(p);
}
N void fill_text(int fd) { char buf[64]; read(fd, buf, 63); buf[63] = 0; strcpy(g_text, buf); text_reader(); }
N void fill_count(int fd) { read(fd, &g_count, sizeof g_count); count_reader(); }
N void scan_name(void) { char name[16]; scanf("%s", name); puts(name); }
int main(void) { fill_text(0); fill_count(0); scan_name(); return 0; }
"""


def test_callee_rule_and_continuation_reseed_on_a_real_binary(tmp_path, shared_bn):
    """V4-03: a fortified sink and an ISO scanf alias both name the symbol
    the binary calls as `callee`, the model key as `model`. W07-13: a
    continuation leaf is labelled with the global's data symbol, and its
    reseed seeds the reader's input -- never the reader's sink argument --
    or is absent."""
    src = tmp_path / "named_global.c"
    src.write_text(_NAMED_GLOBAL_SRC)
    binary = tmp_path / "named_global"
    _compile(src, binary, ["-O2", "-D_FORTIFY_SOURCE=2", "-std=gnu17"])
    symbols = _dynamic_symbols(binary)
    if "__strcpy_chk" not in symbols or "__isoc99_scanf" not in symbols:
        refuse_silent_skip("this compiler/glibc emitted no __strcpy_chk / __isoc99_scanf")
    target = shared_bn.load(binary, copy=False)

    def forward(fn, source):
        return _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                        "--source", source)

    text = forward("fill_text", "call:read")
    assert {(s["sink"]["callee"], s["sink"]["callee_symbol"], s["sink"].get("model"))
            for s in text["reached_sinks"]} == {("__strcpy_chk", "__strcpy_chk", "strcpy_chk")}, \
        text["reached_sinks"]
    scanned = forward("scan_name", "call:scanf")
    assert {(s["sink"]["callee"], s["sink"]["callee_symbol"], s["sink"].get("model"))
            for s in scanned["reached_sinks"]} == \
        {("__isoc99_scanf", "__isoc99_scanf", "scanf")}, scanned["reached_sinks"]

    # The reader hands the global's constant address straight to its sink:
    # no input to seed, so no reseed (the text follow-up is the xrefs audit).
    [leaf] = [lf for lf in text["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["global"]["label"] == "g_text", leaf["global"]
    assert [r["display_name"] for r in leaf["references"]] == ["text_reader"], leaf
    assert [r["seed"] for r in leaf["references"]] == [None], leaf["references"]
    assert "reseed" not in leaf, leaf

    # The reader loads the global: the reseed is the loaded value, and it
    # reaches the reader's sink through the reader's own flow.
    count = forward("fill_count", "call:read")
    [leaf] = [lf for lf in count["leaves"] if lf["kind"] == "global_memory_continuation"]
    assert leaf["global"]["label"] == "g_count", leaf["global"]
    [row] = leaf["references"]
    assert (row["display_name"], row["access"]) == ("count_reader", "load"), row
    assert row["seed"] and row["seed"].startswith("var:"), row
    assert leaf["reseed"] and f"--source {row['seed']}" in leaf["reseed"], leaf
    assert "--source arg:malloc:" not in leaf["reseed"], leaf["reseed"]
    inside = forward(row["address"], row["seed"])
    [hit] = inside["reached_sinks"]
    assert (hit["sink"]["callee"], hit["sink"]["class"]) == ("malloc", "alloc_size"), hit
    assert hit["metrics"]["steps"] > 1, hit


@pytest.mark.parametrize("opt", ["-O0", "-O1"], ids=["O0", "O1"])
def test_a_static_the_seeded_function_reads_before_its_store_blocks(tmp_path, shared_bn, opt):
    """V3-01 on `taint_corpus/fwd_static_reread.c`.

    `next_index` loads its static `cursor` to index a table, then stores this
    call's `read()` input into it and returns. Head (b075c82c) and the first
    fixed integration answered `safe_to_report_all_clear: true` with no leaf:
    the exit frontier skipped the seeded function's own references as "already
    walked", but this load ran BEFORE the store and sees it only on the next
    activation. Base (632891c9) kept an unkeyed `coarse_memory_store`."""
    binary = tmp_path / f"fwd_static_reread{opt[-1]}"
    _compile(CORPUS / "fwd_static_reread.c", binary, [opt])
    target = shared_bn.load(binary, copy=False)

    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "next_index",
                      "--source", "call:read")
    assert result["diagnostics"]["safe_to_report_all_clear"] is False, result["diagnostics"]
    [store] = [lf for lf in result["leaves"] if lf["kind"] == "coarse_memory_store"]
    assert store["self_reread"] is True, store
    [row] = store["references"]
    assert (row["display_name"], row["access"]) == ("next_index", "load"), store
    xrefs = _bn_json(shared_bn, "-t", target, "xrefs", store["global"]["address"])
    sites = {r["address"] for r in xrefs["items"]
             if r.get("kind") == "code" and r.get("function") == "next_index"}
    assert row["at"] in sites, (row, sites)
    # The load precedes the tainting store in the body.
    assert int(row["at"], 16) < int(store["address"], 16), store

    control = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "latest_index",
                       "--source", "call:read")
    assert not [lf for lf in control["leaves"] if lf["kind"] == "coarse_memory_store"], \
        control["leaves"]


@pytest.mark.parametrize("opt", ["-O0", "-O2"], ids=["O0", "O2"])
def test_a_member_pointing_into_its_own_struct_reads_the_tainted_field(tmp_path, shared_bn,
                                                                      opt):
    """V1-01 / V2-01 on `taint_corpus/fwd_struct_selfptr.c` (built with `-g`, so BN
    types the struct and keeps it one object).

    `main` taints only `data.slots[0].path` and points `data.active` at
    `data.slots[0]`; `attach_active` opens `s->active->path`. Field-only struct
    taint read that load as a clean member and reported no sink at all; the
    true `fopen` in `open_path` must be reached while `touch_log`'s `fopen` of
    the clean `log` member stays silent. `indexed_entry` points `active` at a
    slot chosen at run time: the load is a `field_load_unresolved` frontier,
    never silence."""
    binary = tmp_path / f"fwd_struct_selfptr{opt[-1]}"
    _compile(CORPUS / "fwd_struct_selfptr.c", binary, [opt, "-g"])
    target = shared_bn.load(binary, copy=False)
    sites = _bn_json(shared_bn, "-t", target, "callsites", "fopen")
    owner = {row["call_addr"]: row["containing_function"]["name"] for row in sites["items"]}
    assert sorted(owner.values()) == ["open_path", "touch_log"], owner

    result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "main",
                      "--source", "param:1")
    fopens = [owner.get(hit["sink"]["address"]) for hit in result["reached_sinks"]
              if hit["sink"]["callee"] == "fopen"]
    assert fopens == ["open_path"], result["reached_sinks"]

    indexed = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", "indexed_entry",
                       "--source", "param:1")
    assert not [hit for hit in indexed["reached_sinks"]
                if owner.get(hit["sink"]["address"]) == "touch_log"], indexed["reached_sinks"]
    assert "field_load_unresolved" in {lf["kind"] for lf in indexed["leaves"]}, \
        indexed["leaves"]
    assert indexed["diagnostics"]["safe_to_report_all_clear"] is False, indexed["diagnostics"]


def test_the_forward_all_clear_names_its_sink_class_scope(tmp_path, shared_bn):
    """IC07 on a real binary. `push_label` sends its seeded argument: a
    default run reports no sink -- `net_write` is opt-in -- but records the
    call, scopes its all-clear to the enabled classes and says so in text and
    JSON alike; `--sink-class net_write` reports exactly the recorded
    arguments. `scale_count`'s seed reaches only its return value: an
    all-clear whose diagnostics name that exit, never a dead end."""
    binary = tmp_path / "opt_in_scope"
    _compile(CORPUS / "opt_in_scope.c", binary)
    target = shared_bn.load(binary, copy=False)

    def forward(fn, source, *extra, fmt="json"):
        args = ("-t", target, "taint", "forward", "-f", fn, "--source", source, *extra)
        if fmt == "json":
            return _bn_json(shared_bn, *args)
        res = shared_bn.run(*args, timeout=120.0)
        assert res.returncode == 0, res.stderr
        return res.stdout

    default = forward("push_label", "param:1")
    assert default["reached_sinks"] == [] and default["run_params"]["sink_classes"] == []
    [row] = default["opt_in_sinks_reached"]
    assert (row["callee"], row["class"]) == ("send", "net_write")
    assert default["diagnostics"]["safe_to_report_all_clear"] is True
    assert default["diagnostics"]["opt_in_sink_classes_reached"] == ["net_write"]

    enabled = forward("push_label", "param:1", "--sink-class", "net_write")
    assert enabled["run_params"]["sink_classes"] == ["net_write"]
    assert enabled["opt_in_sinks_reached"] == []
    assert sorted((f["sink"]["address"], f["sink"]["tainted_arg_index"])
                  for f in enabled["reached_sinks"]) == [
        (row["address"], i) for i in row["tainted_args"]]

    [verdict] = [line for line in forward("push_label", "param:1", fmt="text").splitlines()
                 if line.startswith("verdict:")]
    assert "NOT an all-clear" not in verdict and "net_write" in verdict

    scaled = forward("scale_count", "param:0")
    assert scaled["reached_sinks"] == [] and scaled["opt_in_sinks_reached"] == []
    assert scaled["diagnostics"]["safe_to_report_all_clear"] is True
    assert scaled["diagnostics"]["exits"]["return"] is True


def test_trace_reports_observed_truncation_and_quick_view_state(tmp_path, shared_bn):
    """BN6-C011: the memcpy source argument walks back four steps to the
    function's only parameter. Budgets 4 and 5 are the same complete answer;
    3 leaves the parameter pending. Budget 4 used to say `truncated: true`.
    BN6-C010: the same trace on a quick-loaded copy discloses
    `analysis_state: quick` / `partial: true`; it carried no state at all.
    The quick copy has a saved database beside it, as a primed fixture does:
    the quick view must still be the raw bytes, never that full analysis."""
    built = {}
    for name in ("trace_budget", "trace_budget_quick"):
        built[name] = tmp_path / name
        # -fno-builtin: GCC inlines an 8-byte memcpy even at -O0.
        _compile(CORPUS / "trace_budget.c", built[name], ["-fno-builtin"])
    full = shared_bn.load(built["trace_budget"], copy=False)
    sidecar = tmp_path / "trace_budget_quick.bndb"
    primed = shared_bn.load(built["trace_budget_quick"], copy=False)
    saved = shared_bn.run("save", "--target", primed, "--path", str(sidecar), "--format", "json")
    assert saved.returncode == 0 and sidecar.exists(), saved.stderr
    closed = shared_bn.run("close", primed)
    assert closed.returncode == 0, closed.stderr
    quick = shared_bn.load(built["trace_budget_quick"], quick=True)

    sites = _bn_json(shared_bn, "-t", full, "callsites", "memcpy",
                     "--within", "publish_banner")["items"]
    [call] = [site["call_addr"] for site in sites]

    def trace(target, *extra):
        return _bn_json(shared_bn, "-t", target, "trace", "publish_banner", call,
                        "--arg", "1", *extra)

    runs = {depth: trace(full, "--max-depth", str(depth)) for depth in (3, 4, 5)}
    assert [s["reason"] for s in runs[5]["trace"]][-1] == "function_parameter"
    assert {depth: (r["step_count"], r["truncated"], len(r["assumptions"]))
            for depth, r in runs.items()} == {3: (3, True, 1), 4: (4, False, 0),
                                              5: (4, False, 0)}
    assert runs[4]["trace"] == runs[5]["trace"]
    assert (runs[5]["analysis_state"], runs[5]["partial"]) == ("full", False)

    partial = trace(quick)
    assert (partial["analysis_state"], partial["partial"]) == ("quick", True)
    # The raw binary was opened, not the database beside it.
    assert Path(quick).suffix != ".bndb", quick


def _inline_expansion_target(tmp_path, shared_bn) -> str:
    """`taint_corpus/inline_expansion.c` at -O2: two constant-size `rep stos`/
    `rep movs` expansions and one whose length byte comes from the packet."""
    binary = tmp_path / "inline_expansion"
    _compile(CORPUS / "inline_expansion.c", binary, ["-O2"])
    return shared_bn.load(binary, copy=False)


def test_an_alignment_split_inline_expansion_is_bounded_not_overflow(tmp_path, shared_bn):
    """V8-03: `memset(r, 0, sizeof *r)` and `memcpy(d, s, sizeof *d)` inlined
    as `rep stos`/`rep movs` were `overflow_len` sinks under a tainted pointer
    -- their count is the constant total split by the pointer's alignment. A
    length byte read from the packet, expanded the same way, still overflows."""
    target = _inline_expansion_target(tmp_path, shared_bn)

    def sinks(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", "param:0")
        return [s["sink"] for s in result["reached_sinks"]]

    for fn, callee, intrinsic in (("record_reset", "__builtin_memset", "__memfill_u64"),
                                  ("record_copy", "__builtin_memcpy", "__memcpy_u64")):
        [sink] = sinks(fn)
        assert (sink["callee"], sink["class"], sink["catalog_class"]) == (
            callee, "bounded_len", "overflow_len"), (fn, sink)
        assert sink["intrinsic_expansion"]["intrinsic"] == intrinsic
        assert sink["intrinsic_expansion"]["length_kind"] == "alignment_split"
        assert int(sink["intrinsic_expansion"]["length_bound"], 16) <= 0x140, sink

    [label] = sinks("copy_label")
    assert (label["callee"], label["class"]) == ("__builtin_memcpy", "overflow_len"), label
    assert "intrinsic_expansion" not in label


def test_a_tainted_size_inline_memset_overflows_even_when_masked(tmp_path, shared_bn):
    """G4c P1: `memset(out, 0, (pkt[0] & 0x3ff) + 16)` at -O2 is a `rep stos`
    whose count adds the packet's masked size to the alignment correction. The
    mask erased the size's dependency, so the sink read `bounded_len` with
    `length_bound: 0x408` -- an overflow of any destination shorter than that.
    It is `overflow_len` with the bound as a fact, and a catalog audit row; the
    unmasked `memset(out, 0, pkt[0])` (`rep stos` too) stays `overflow_len`."""
    target = _inline_expansion_target(tmp_path, shared_bn)

    def sink_of(fn):
        result = _bn_json(shared_bn, "-t", target, "taint", "forward", "-f", fn,
                          "--source", "param:1")
        [sink] = [s["sink"] for s in result["reached_sinks"]]
        return sink

    masked = sink_of("clear_masked")
    assert (masked["callee"], masked["class"]) == ("__builtin_memset", "overflow_len"), masked
    assert masked["intrinsic_expansion"] == {
        "intrinsic": "__memfill_u64", "length_bound": "0x408",
        "length_kind": "masked_value", "destination_fit": "unproved"}, masked
    tail = sink_of("clear_tail")
    assert (tail["callee"], tail["class"]) == ("__builtin_memset", "overflow_len"), tail

    catalog = _bn_json(shared_bn, "-t", target, "taint", "models", "--role", "sink",
                       "--present", "--callsites", "--all-sites")
    [entry] = [e for e in catalog["items"] if e["symbol"] == "__builtin_memset"]
    kinds = {row["function"]: row["kind"] for row in entry["callsites"]}
    assert (kinds["clear_masked"], kinds["clear_tail"], kinds["record_reset"]) == (
        "app_caller", "app_caller", "inline_builtin"), kinds


def test_an_inline_copy_is_one_backward_sink_site(tmp_path, shared_bn):
    """V8-04: BN lifts the `rep movs` of an inlined variable-length copy to two
    `__builtin_memcpy` calls at one address; backward seeded both, emitting two
    identical `sink_status` rows and every slice twice."""
    target = _inline_expansion_target(tmp_path, shared_bn)
    result = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "copy_label",
                      "--sink", "arg:__builtin_memcpy:2")
    [row] = result["sink_status"]
    assert row["status"] == "seeded", row
    assert row["slices"] == len(result["slices"]) >= 1
    assert {sl["sink"]["address"] for sl in result["slices"]} == {row["address"]}


def test_inline_expansions_are_listed_by_the_catalog_and_callsites(tmp_path, shared_bn):
    """V8-06: the variable-length inline copy (the tainted-total `rep movs`)
    was missing from `taint models --present --callsites --all-sites`, and
    `callsites __builtin_memcpy` refused the synthetic builtin, so only
    `taint forward` ever surfaced it. Both now list it where backward seeds it;
    a fixed-size expansion stays a non-audit `inline_builtin` row."""
    target = _inline_expansion_target(tmp_path, shared_bn)
    backward = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "copy_label",
                        "--sink", "arg:__builtin_memcpy:2")
    [site] = {row["address"] for row in backward["sink_status"]}

    catalog = _bn_json(shared_bn, "-t", target, "taint", "models", "--role", "sink",
                       "--present", "--callsites", "--all-sites")
    kinds = {}
    for symbol in ("__builtin_memcpy", "__builtin_memset"):
        [entry] = [e for e in catalog["items"] if e["symbol"] == symbol]
        kinds.update({(row["function"], row["address"]): row["kind"]
                      for row in entry["callsites"]})
    assert kinds[("copy_label", site)] == "app_caller", kinds
    assert {kind for (fn, _), kind in kinds.items()
            if fn in ("record_reset", "record_copy")} == {"inline_builtin"}, kinds
    assert {fn for fn, _ in kinds} >= {"record_reset", "record_copy"}, kinds

    listed = _bn_json(shared_bn, "-t", target, "callsites", "__builtin_memcpy",
                      "--within", "copy_label")
    assert [(row["call_addr"], row["call_kind"]) for row in listed["items"]] == [
        (site, "inline_expansion")]


def test_trace_at_an_inline_copy_names_the_builtin_and_its_count_register(tmp_path, shared_bn):
    """V8-05: `trace` at the `rep movs` taint names `__builtin_memcpy` printed
    `<unresolved callee>` and the convention's `rdx` for the length; the
    count of a `rep movs` is `rcx`."""
    target = _inline_expansion_target(tmp_path, shared_bn)
    backward = _bn_json(shared_bn, "-t", target, "taint", "backward", "-f", "copy_label",
                        "--sink", "arg:__builtin_memcpy:2")
    [site] = {row["address"] for row in backward["sink_status"]}
    trace = _bn_json(shared_bn, "-t", target, "trace", "copy_label", site, "--arg", "2")
    assert trace["callee"] == "__builtin_memcpy"
    assert trace["arg_label"]["register"] == "rcx"
