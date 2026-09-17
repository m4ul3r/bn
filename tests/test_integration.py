"""Integration tests for multi-instance bridge sessions.

These tests require Binary Ninja to be importable. They are skipped if
the binaryninja module is not available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bn.paths import instances_dir

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"
ADD_BINARY = FIXTURES_DIR / "add_x86_64"
DISPATCH_BINARY = FIXTURES_DIR / "dispatch_table_x86_64"
PARSER_BINARY = FIXTURES_DIR / "parser_x86_64"

# The gate is BN availability *only* (#590). Gating on whether the generated
# fixtures happen to exist made a fresh checkout report "27 skipped, exit 0"
# with BN installed -- indistinguishable from a pass. `real_bn` skips visibly
# (and fails under BN_REQUIRE_REAL_TESTS=1); the session-scoped
# `integration_fixtures` builds the binaries, and errors loudly if it can't.
pytestmark = [
    pytest.mark.real_bn,
    pytest.mark.usefixtures("integration_fixtures"),
]

# Use the bn console-scripts entry point instead of -m bn.cli
# to avoid Python module shadowing issues with the 'bn' package name.
_BN_CLI = [str(Path(sys.executable).parent / "bn")]
# Built at call time, not import time (#589): conftest's autouse `_hermetic_env`
# fixture pins BN_CACHE_DIR/NO_COLOR per test, and a module-import-time snapshot
# of os.environ would predate every fixture -- so the subprocesses these helpers
# spawn would read the developer's real ~/.cache/bn instead of the isolated one.
def _env() -> dict[str, str]:
    return dict(os.environ)


def _bn(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_BN_CLI, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
    )


# Bringing a binary into a bridge is a superprocess: for `session start` it is
# bridge spawn + BN import + FULL analysis of every binary passed in; for a
# `load` into the shared bridge it is the analysis alone. These budgets are
# ceilings on that whole pipeline, sized from the SLOWEST binary rather than
# tuned to one machine.
#
# Measured on a 6-core laptop, warm: ~3s of bridge startup, and ~20s of
# analysis for the -static aarch64 probe (~1.1k functions). #718 was filed
# while this default was 30s -- a margin thinner than the measurement's own
# noise -- so a cold BN cache or a loaded host manufactured
# `subprocess.TimeoutExpired` failures indistinguishable from a genuine hang.
# The cross-arch lane is the one that analyses that probe, so it gets the
# larger, explicitly named budget -- on whichever call does the analysing.
_SESSION_START_TIMEOUT = 120.0
_CROSS_ARCH_ANALYSIS_TIMEOUT = 300.0

# How much of a bridge log to attach to a timeout: the crash output that matters
# is at the end, and the tail cannot be allowed to swamp the failure report.
_LOG_EXCERPT_CHARS = 4000


def _decode(stream: str | bytes | None) -> str:
    """Partial output arrives as bytes even in text mode on the timeout path."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream


def _bridge_log_excerpts(started_at: float) -> list[str]:
    """Excerpt every bridge log written since *started_at* (#718).

    A timed-out `session start` reports no instance id -- the CLI only prints one
    on success -- but the spawn creates ``<instances_dir>/<id>.log`` before it
    does anything else, so any log newer than the attempt is that attempt's.
    That holds because conftest's autouse ``_hermetic_env`` pins ``BN_CACHE_DIR``
    per test (#589): this instances dir belongs to one test, so "newer than the
    attempt" cannot pick up a concurrent unrelated bridge. An empty list is
    itself a diagnosis: the CLI hung before spawning a bridge.
    """
    try:
        paths = sorted(instances_dir().glob("*.log"))
    except OSError:
        return []
    excerpts = []
    for path in paths:
        try:
            if path.stat().st_mtime < started_at:
                continue  # a log from an earlier instance in this cache dir
            text = path.read_text(errors="replace")
        except OSError as exc:
            excerpts.append(f"--- {path.name}: unreadable ({exc}) ---")
            continue
        excerpts.append(f"--- {path.name} (tail) ---\n{text[-_LOG_EXCERPT_CHARS:]}")
    return excerpts


class _SessionStartTimeout(subprocess.TimeoutExpired):
    """A timed-out `session start`, carrying how far the start got (#718).

    A bare `TimeoutExpired` renders as ``Command [...] timed out after N
    seconds`` and nothing else -- Python 3.14 drops the captured output from
    ``__str__`` entirely -- which is exactly how a genuine hang (#80/#86
    deadlocks, #658 spawn-lock waits) looks. Attach the partial bridge
    stdout/stderr and the bridge log, so slow and wedged stay distinguishable.
    Still a `subprocess.TimeoutExpired`, so nothing can catch it as a pass.
    """

    def __init__(self, exc: subprocess.TimeoutExpired, log_excerpts: list[str]) -> None:
        super().__init__(exc.cmd, exc.timeout, output=exc.stdout, stderr=exc.stderr)
        self.log_excerpts = log_excerpts

    def __str__(self) -> str:
        if self.log_excerpts:
            log = "bridge log:\n" + "\n".join(self.log_excerpts)
        else:
            log = (
                "bridge log: none was written before the timeout -- the CLI hung "
                "before (or while) spawning a bridge process"
            )
        return (
            f"Command {self.cmd!r} timed out after {self.timeout} seconds\n"
            f"stdout:\n{_decode(self.output)}\n"
            f"stderr:\n{_decode(self.stderr)}\n"
            f"{log}"
        )


def _session_start(*binaries: str, timeout: float = _SESSION_START_TIMEOUT) -> dict:
    # session start defaults to text output; this helper parses JSON.
    cmd = [*_BN_CLI, "session", "start", "--format", "json"]
    cmd.extend(str(b) for b in binaries)
    started_at = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env())
    except subprocess.TimeoutExpired as exc:
        # #718: a timeout must never surface as a bare `-9` with no evidence.
        raise _SessionStartTimeout(exc, _bridge_log_excerpts(started_at)) from exc
    assert result.returncode == 0, f"session start failed: {result.stderr}"
    return json.loads(result.stdout)


def _session_stop(instance_id: str, timeout: float = 10.0) -> None:
    subprocess.run(
        [*_BN_CLI, "session", "stop", instance_id],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
    )


class TestMultiInstance:
    """Test running two bridge sessions in parallel."""

    def test_two_sessions_isolated(self):
        """Start two sessions with different binaries, verify command isolation."""
        info_a = _session_start(str(HELLO_BINARY))
        try:
            info_b = _session_start(str(ADD_BINARY))
            try:
                id_a = info_a["instance_id"]
                id_b = info_b["instance_id"]
                assert id_a != id_b

                # Each session should have exactly 1 target
                result_a = _bn("--instance", id_a, "target", "list", "--format", "json")
                targets_a = json.loads(result_a.stdout)["items"]   # #358 {kind, items}
                assert len(targets_a) == 1

                result_b = _bn("--instance", id_b, "target", "list", "--format", "json")
                targets_b = json.loads(result_b.stdout)["items"]
                assert len(targets_b) == 1

                # The basenames should differ
                name_a = targets_a[0].get("selector") or targets_a[0].get("basename", "")
                name_b = targets_b[0].get("selector") or targets_b[0].get("basename", "")
                assert name_a != name_b

            finally:
                _session_stop(id_b)
        finally:
            _session_stop(id_a)

    def test_session_list_shows_both(self):
        """session list should show all running sessions."""
        info_a = _session_start()
        try:
            info_b = _session_start()
            try:
                result = _bn("session", "list", "--format", "json")
                data = json.loads(result.stdout)
                sessions = data["items"]   # #358 {kind, items}
                ids = {s["instance_id"] for s in sessions}
                assert info_a["instance_id"] in ids
                assert info_b["instance_id"] in ids
            finally:
                _session_stop(info_b["instance_id"])
        finally:
            _session_stop(info_a["instance_id"])

    def test_save_and_stop(self, tmp_path):
        """Test saving a database before stopping."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            save_path = str(tmp_path / "hello.bndb")
            result = _bn("--instance", inst_id, "save", save_path, "--format", "json")
            assert result.returncode == 0
            parsed = json.loads(result.stdout)
            assert parsed.get("saved") is True
            assert Path(save_path).exists()
        finally:
            _session_stop(inst_id)


class TestSavePathIdentity:
    """Regression for #256: `save --path` writes a COPY and must not re-home the
    live target, so the original selector keeps resolving afterward. Needs real BN
    -- only `bv.create_database` actually rebinds the view's filename."""

    def test_save_path_keeps_original_selector(self, shared_bn, tmp_path):
        # Two targets in one instance so the selector is REQUIRED and name-based.
        shared_bn.load(HELLO_BINARY)
        shared_bn.load(ADD_BINARY)
        listing = json.loads(
            shared_bn.run("target", "list", "--format", "json").stdout)["items"]
        hello = next(t for t in listing
                     if "hello" in (t.get("filename", "") + t.get("basename", "")))
        sel = hello.get("selector") or hello.get("basename")

        copy = str(tmp_path / "copy.bndb")
        saved = shared_bn.run("save", "--target", sel, "--path", copy,
                              "--format", "json")
        assert saved.returncode == 0, saved.stderr
        assert json.loads(saved.stdout).get("saved") is True
        assert Path(copy).exists()

        # The original selector must STILL resolve -- before the fix the live
        # target was rebound to copy.bndb and `sel` raised "not found".
        after = shared_bn.run("target", "info", "--target", sel,
                              "--format", "json")
        assert after.returncode == 0, (
            f"original selector {sel!r} stopped resolving after save --path: "
            f"{after.stdout} {after.stderr}")

    def test_bare_save_multi_target_gets_target_hint(self, shared_bn):
        # #663 end-to-end: bare `bn save` with two targets open in one headless
        # instance must exit 2 with the -t hint + open-target list -- the live
        # behavior the mocked lanes structurally cannot see (their fakes
        # hard-code the very message under test). Since #688 the refusal comes
        # from the registry's destructive gate rather than the resolver's
        # no-active branch: save overwrites state, so it is refused on the
        # COUNT, which also covers a GUI bridge where a focused tab exists.
        shared_bn.load(HELLO_BINARY)
        shared_bn.load(ADD_BINARY)
        result = shared_bn.run("save")
        assert result.returncode == 2, (result.stdout, result.stderr)
        err = result.stderr
        assert "save_database needs an explicit target when multiple targets are open (2)" in err
        assert "Pass -t <selector> (--target) to choose one." in err
        assert "Open targets:" in err
        # Prefix-matches both the raw and the .bndb-restored selector spelling.
        assert "-t hello_x86_64" in err
        assert "-t add_x86_64" in err

        # `-t active` (the documented copy-paste footgun, #366) collapses
        # bridge-side to the same refusal. Assert the discriminating first
        # line, not just "Open targets:", which the unknown-selector error
        # also prints: if the bridge collapse regressed, `-t active` would
        # fall through to "Unknown target selector" and a looser assert
        # would stay green.
        result = shared_bn.run("save", "--target", "active")
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "save_database needs an explicit target when multiple targets are open (2)" \
            in result.stderr

        # A `require_target=True` command refuses in the CLI pre-flight
        # instead, and must print the SAME grammar (#688) -- the two halves
        # that used to disagree, checked against one live bridge.
        result = shared_bn.run("target", "info")
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "This command requires --target when multiple targets are open." \
            in result.stderr
        assert "Pass -t <selector> (--target) to choose one." in result.stderr
        assert "-t hello_x86_64" in result.stderr
        assert "note: view_id / target_id are stable across `bn save`" in result.stderr

        # `-t ""` is an explicit-but-empty selector (an unset shell
        # variable): #690 r3 rejects it CLI-side for every command -- it is
        # never pin-filled and never forwarded (the bridge would collapse
        # it to the focused view with no count check).
        result = shared_bn.run("save", "--target", "")
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "--target is empty" in result.stderr


class TestProtoSetUnnamedParams:
    """Regression for #254: a `proto set` whose prototype omits parameter names
    must verify, not be reported verification_failed and reverted. BN auto-names
    unnamed params on readback (arg1, arg2, ...), so the readback text never
    matches the requested string -- only a real BN readback reproduces this, so
    the mocked suite can't cover it.

    These apply WITHOUT --preview (a committed proto set): setting a prototype
    pins has_user_type, which BN cannot clear, so a --preview of a proto set on an
    AUTO function is refused (see test_auto_prototype_preview_is_refused, #630) --
    committing is the correct way to prove the unnamed/named acceptance."""

    def _first_fn(self, shared_bn):
        out = shared_bn.run("function", "list", "--format", "json")
        return json.loads(out.stdout)["items"][0]["name"]

    def test_unnamed_params_verify(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        fn = self._first_fn(shared_bn)
        res = shared_bn.run("proto", "set", fn,
                            f"void {fn}(int32_t, char**, char**)", "--format", "json")
        parsed = json.loads(res.stdout)
        statuses = [r.get("status") for r in parsed["results"]]
        assert statuses == ["verified"], parsed
        assert res.returncode == 0, res.stdout

    def test_named_params_also_verify(self, shared_bn):
        """Contrast case: a fully NAMED prototype still verifies through the same
        path -- the name-insensitive acceptance must not perturb the normal,
        string-matching case. (The rejection of a genuine type/arity/return
        mismatch can't be forced through real BN, which applies valid prototypes
        verbatim, so that is covered by the mocked unit test
        test_prototype_matches_ignoring_param_names.)"""
        shared_bn.load(HELLO_BINARY)
        fn = self._first_fn(shared_bn)
        res = shared_bn.run("proto", "set", fn,
                            f"int32_t {fn}(int64_t argc, char** argv)", "--format", "json")
        parsed = json.loads(res.stdout)
        assert [r.get("status") for r in parsed["results"]] == ["verified"], parsed

    def test_auto_prototype_preview_is_refused(self, shared_bn):
        """#630: a --preview of a proto set on an AUTO function (no user type) is
        REFUSED before any mutation, because BN cannot clear the has_user_type it
        would pin, so the preview could not be cleanly reverted. Proves the honest
        contract on live BN: refuse rather than apply and claim a clean rollback.
        The view is left pristine -- the function stays AUTO."""
        shared_bn.load(HELLO_BINARY)
        fn = self._first_fn(shared_bn)  # a fresh-analysis function is AUTO
        res = shared_bn.run("proto", "set", fn,
                            f"void {fn}(int32_t, char**, char**)", "--preview", "--format", "json")
        # A refusal is a structured OperationFailure(status="unsupported")
        # escaping a `_mutate`-marked call, so it lands at exit 3 (#625/#701:
        # a FAILED_MUTATION_STATUSES status on a genuine mutation call). It is
        # NOT exit 2 -- that is reserved for a read/resolver op sharing the
        # status string. `status` is surfaced in the --format json envelope.
        assert res.returncode == 3, (res.returncode, res.stdout, res.stderr)
        payload = json.loads(res.stdout)
        assert payload["status"] == "unsupported", payload
        assert "has_user_type" in (res.stdout + res.stderr), (res.stdout, res.stderr)
        # Pristine, checked against a source that ACTUALLY reflects has_user_type:
        # `function info` never emits the flag, so asserting on its output is
        # vacuous. Instead commit a real prototype set now and read the op's
        # before_has_user_type -- it reports the function's provenance at the
        # moment before this commit. If the refused preview had wrongly pinned
        # has_user_type, before_has_user_type would be true and this fails.
        commit = shared_bn.run("proto", "set", fn,
                               f"void {fn}(int32_t, char**, char**)", "--format", "json")
        commit_parsed = json.loads(commit.stdout)
        proto_result = next(r for r in commit_parsed["results"]
                            if r.get("op") == "set_prototype")
        assert proto_result["before_has_user_type"] is False, commit_parsed


class TestStructFieldTypedef:
    """Regression for #246: field ops on a typedef'd (NamedTypeReference) struct
    must follow the alias to the underlying tag instead of crashing in
    add_member_at_offset. The mocked suite cannot reproduce mutable_copy()
    returning an NTR builder, so this drives the real BN type system end-to-end.
    """

    def _declare_and_set(self, shared_bn, decl, struct_name):
        declared = shared_bn.run("types", "declare", decl, "--format", "json")
        assert declared.returncode == 0, declared.stderr
        return shared_bn.run("struct", "field", "set",
                             struct_name, "0x4", "newfield", "uint32_t", "--format", "json")

    def test_set_field_on_named_typedef_struct(self, shared_bn):
        """typedef of a named struct: `typedef struct InnerRec AliasRec;`. The
        report must key on the underlying TAG, not the alias: affected_types names
        the tag (so it carries members and a real layout diff) and agrees with
        results[].struct_name (#246, incl. the reporting-path follow-up)."""
        shared_bn.load(HELLO_BINARY)
        res = self._declare_and_set(
            shared_bn,
            "struct InnerRec { uint32_t x; }; typedef struct InnerRec AliasRec;",
            "AliasRec")
        assert res.returncode == 0, f"set crashed: {res.stdout}\n{res.stderr}"
        parsed = json.loads(res.stdout)
        affected = parsed["affected_types"]
        assert affected and affected[0]["name"] == "InnerRec", affected
        assert affected[0]["changed"] is True, affected
        # the member-level layout (not just the alias header) is in the diff
        assert "newfield" in affected[0]["after_layout"], affected[0]["after_layout"]
        assert parsed["results"][0]["struct_name"] == "InnerRec"
        # the field landed on the underlying tag, and the typedef sees it
        shown = shared_bn.run("struct", "show", "InnerRec")
        assert "newfield" in shown.stdout, shown.stdout

    def test_rename_field_through_typedef_reports_change(self, shared_bn):
        """Regression for the reporting follow-up: a field rename through a typedef
        must report the real change against the TAG -- before the fix it keyed the
        diff on the members-less alias and falsely said 'No effective change
        detected' even though the op verified (#246)."""
        shared_bn.load(HELLO_BINARY)
        self._declare_and_set(
            shared_bn,
            "struct InnerRec { uint32_t x; }; typedef struct InnerRec AliasRec;",
            "AliasRec")
        res = shared_bn.run("struct", "field", "rename",
                            "AliasRec", "newfield", "renamed", "--format", "json")
        assert res.returncode == 0, f"rename failed: {res.stdout}\n{res.stderr}"
        parsed = json.loads(res.stdout)
        assert parsed["results"][0]["status"] == "verified", parsed["results"]
        affected = parsed["affected_types"]
        assert affected and affected[0]["name"] == "InnerRec", affected
        assert affected[0]["changed"] is True, affected
        assert "No effective change" not in (affected[0].get("message") or "")

    def test_set_field_on_anonymous_typedef_struct(self, shared_bn):
        """The idiomatic `typedef struct { ... } AnonRec;` -- body is registered
        under the auto-named tag `_AnonRec`, alias is an NTR to it."""
        shared_bn.load(HELLO_BINARY)
        res = self._declare_and_set(
            shared_bn,
            "typedef struct { uint32_t m; } AnonRec;",
            "AnonRec")
        assert res.returncode == 0, f"set crashed: {res.stdout}\n{res.stderr}"
        shown = shared_bn.run("struct", "show", "_AnonRec")
        assert "newfield" in shown.stdout, shown.stdout

    def test_set_field_on_typedef_to_nonstruct_is_clean_error(self, shared_bn):
        """`typedef uint32_t Foo;` resolves to a non-aggregate: a field set must
        fail cleanly (not exit 0, not an internal AttributeError crash)."""
        shared_bn.load(HELLO_BINARY)
        res = self._declare_and_set(
            shared_bn, "typedef uint32_t NotAStruct;", "NotAStruct")
        assert res.returncode != 0, f"expected a clean failure, got: {res.stdout}"
        assert "AttributeError" not in res.stdout + res.stderr


class TestTaintIndirectValueSetAnchor:
    """Regression for #282: a recv/read-style source must anchor at an INDIRECT
    call whose target Binary Ninja's *value-set* resolves to the callee. The
    mocked unit suite drives this with a synthetic PossibleValueSet; only a real
    BN run over a const function-pointer dispatch table produces a genuine
    LookupTableValue on the call dest, so this is the sole real-BN coverage of
    the value-set anchoring branch. The dispatch_table fixture is a non-PIE C
    `static const handler_t table[3]; table[cmd](buf, n)` -- the one shape BN VSA
    pins (C++ vtables / data-indexed tables / PIE GOT do not)."""

    def _ensure_fixture(self):
        # The ad hoc builder this replaced (#590) was unreachable: the module
        # gate skipped before it could run. `integration_fixtures` owns the
        # build now, so an absent binary here is a bug, not a skip.
        assert DISPATCH_BINARY.exists(), (
            f"{DISPATCH_BINARY.name} missing -- the integration_fixtures build "
            f"fixture should have produced it"
        )

    def test_value_set_resolved_indirect_call_anchors_source(self, shared_bn):
        self._ensure_fixture()
        shared_bn.load(DISPATCH_BINARY)
        # arg:h_copy:1 with NO --resolve-map: the source must anchor at the
        # indirect `table[cmd](buf, n)` call because value-set resolves it to
        # {h_copy, h_noop, h_log}, and the attacker length must reach h_copy's
        # copy sink.
        res = shared_bn.run("taint", "forward", "-f", "dispatch",
                            "--source", "arg:h_copy:1", "--format", "json")
        assert res.returncode == 0, res.stderr
        out = json.loads(res.stdout)
        result = out.get("result", out)
        assumptions = result.get("assumptions", [])
        # anchored via value-set (not a map), with the multiplicity disclosure
        anchor = [a for a in assumptions
                  if "anchored at indirect callsite" in a and "value-set" in a]
        assert anchor, f"no value-set anchor assumption: {assumptions}"
        assert any("candidate target" in a for a in anchor), anchor
        # the seeded length propagated through the resolved callee to a copy sink
        classes = [s.get("sink", {}).get("class") for s in result.get("reached_sinks", [])]
        assert any(c in ("overflow_len", "fortified_overflow") for c in classes), result


class TestStructFieldDeleteWidth:
    """Regression for #320: deleting the trailing field of a struct must shrink
    the struct width (BN's StructureBuilder.remove() leaves it stale), and a
    --preview of that delete must restore the original width on revert. The
    mocked suite cannot model BN's real width bookkeeping, so this drives it
    end-to-end.
    """

    def _declare(self, shared_bn, decl):
        res = shared_bn.run("types", "declare", decl, "--format", "json")
        assert res.returncode == 0, res.stderr
        return res

    def test_delete_trailing_field_shrinks_width(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        self._declare(shared_bn, "struct WTd320 { unsigned char pad[24]; };")
        setres = shared_bn.run("struct", "field", "set",
                               "WTd320", "0x18", "extra", "int32_t", "--format", "json")
        assert setres.returncode == 0, setres.stderr
        shown = shared_bn.run("struct", "show", "WTd320")
        assert "0x1c" in shown.stdout, shown.stdout  # width grew to 0x1c

        res = shared_bn.run("struct", "field", "delete",
                            "WTd320", "extra", "--format", "json")
        assert res.returncode == 0, f"delete failed: {res.stdout}\n{res.stderr}"
        parsed = json.loads(res.stdout)
        assert parsed["results"][0]["status"] == "verified", parsed["results"]
        after = shared_bn.run("struct", "show", "WTd320")
        assert "0x18" in after.stdout, after.stdout   # shrank back to 0x18
        assert "0x1c" not in after.stdout, after.stdout

    def test_preview_delete_restores_width(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        self._declare(shared_bn, "struct WTp320 { unsigned char pad[24]; };")
        shared_bn.run("struct", "field", "set",
                      "WTp320", "0x18", "extra", "int32_t", "--format", "json")
        res = shared_bn.run("struct", "field", "delete",
                            "WTp320", "extra", "--preview", "--format", "json")
        assert res.returncode == 0, f"preview failed: {res.stdout}\n{res.stderr}"
        # after a preview revert, the struct must be unchanged: extra still
        # present and width still 0x1c (preview restored the shrink too).
        after = shared_bn.run("struct", "show", "WTp320")
        assert "extra" in after.stdout, after.stdout
        assert "0x1c" in after.stdout, after.stdout


class TestTypesDeclareBitfield:
    """Regression for #322: BN's headless C parser silently drops bitfield `:N`
    widths and emits an overlapping, oversized layout reported as `verified`. The
    declaration must instead be rejected cleanly, and the corrupt type must NOT
    be registered. Drives the real BN parser end-to-end.
    """

    def test_bitfield_declaration_is_rejected(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run("types", "declare",
                            "struct BF322 { unsigned a:3; unsigned b:5; unsigned c:1; unsigned d:23; };",
                            "--format", "json")
        assert res.returncode != 0, f"expected rejection, got: {res.stdout}"
        parsed = json.loads(res.stdout)
        results = parsed.get("results") or [parsed]
        assert results[0].get("status") == "invalid_request", parsed
        assert "bitfield" in (results[0].get("message") or "").lower(), parsed
        # the corrupt type must not have been registered
        shown = shared_bn.run("struct", "show", "BF322")
        assert "BF322" not in shown.stdout or "size=0x4" not in shown.stdout, shown.stdout

    def test_plain_struct_still_declares(self, shared_bn):
        # The contrast: a bitfield-free struct (incl. a comment containing a
        # colon-number) still declares cleanly -- no false rejection.
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run("types", "declare",
                            "struct OK322 { int a; /* note:32 */ char b; long c; };",
                            "--format", "json")
        assert res.returncode == 0, f"unexpected rejection: {res.stdout}\n{res.stderr}"
        parsed = json.loads(res.stdout)
        assert parsed["results"][0]["status"] == "verified", parsed["results"]


class TestDisasmLinear:
    """Regression for #314: `disasm` must be able to linearly disassemble an
    arbitrary MAPPED address (a missed handler / vtable slot BN left as data),
    not only addresses already inside a function. Drives real BN.
    """

    @staticmethod
    def _unwrap(payload):
        # The CLI prints the bare result; collection reads put the envelope
        # (items/kind/...) at top level. Tolerate a {result: ...} wrapper too.
        return payload.get("result", payload) if isinstance(payload, dict) else payload

    @classmethod
    def _items(cls, payload):
        d = cls._unwrap(payload)
        return d.get("items", []) if isinstance(d, dict) else (d or [])

    def _a_data_address(self, shared_bn) -> str:
        """The start of a non-executable data section -- a mapped address BN did
        not make part of a function."""
        res = shared_bn.run("sections", "--format", "json")
        assert res.returncode == 0, res.stderr
        for sec in self._items(json.loads(res.stdout)):
            if not sec.get("executable") and sec.get("start"):
                return sec["start"]
        raise AssertionError("no non-executable section found")

    def test_linear_disasm_at_non_function_address(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        addr = self._a_data_address(shared_bn)
        # plain disasm refuses it, but points at --linear
        plain = shared_bn.run("disasm", addr)
        assert plain.returncode != 0, plain.stdout
        assert "--linear" in (plain.stdout + plain.stderr)
        # --linear disassembles N instructions there
        res = shared_bn.run("disasm", addr, "--linear", "4", "--format", "json")
        assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
        result = self._unwrap(json.loads(res.stdout))
        assert result.get("linear") is True, result
        assert result.get("function") is None, result
        assert 1 <= result.get("instruction_count", 0) <= 4, result
        assert result["instructions"], result
        assert result["instructions"][0]["address"].lower().startswith("0x")

    def test_linear_disasm_from_function_name(self, shared_bn):
        # --linear also accepts a function name, anchoring at its start.
        shared_bn.load(HELLO_BINARY)
        listing = shared_bn.run("function", "list", "--format", "json")
        name = self._items(json.loads(listing.stdout))[0]["name"]
        res = shared_bn.run("disasm", name, "--linear", "3", "--format", "json")
        assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
        result = self._unwrap(json.loads(res.stdout))
        assert result.get("linear") is True
        assert result.get("instruction_count", 0) >= 1


class TestDisasmThumbIT741:
    """Real native decoders over synthetic raw bytes, without a cross-compiler."""

    @staticmethod
    def _load_raw(shared_bn, tmp_path, raw, platform):
        path = tmp_path / "disasm741.raw"
        path.write_bytes(raw)
        selector = shared_bn.load(path)
        setup = shared_bn.run(
            "py", "exec",
            "import binaryninja as bn\n"
            f"bv.platform = bn.Platform[{platform!r}]\n"
            "bv.add_function(bv.start)\n"
            "bv.update_analysis_and_wait()\n"
            "assert bv.get_function_at(bv.start) is not None\n"
            "print('START741=' + hex(bv.start))",
            "--target", selector,
        )
        assert setup.returncode == 0, f"{setup.stdout}\n{setup.stderr}"
        start = next(
            line.removeprefix("START741=")
            for line in setup.stdout.splitlines() if line.startswith("START741=")
        )
        return selector, int(start, 16)

    @pytest.mark.parametrize(
        ("hex_bytes", "widths"),
        [
            ("14bf022300237047", (2, 2, 2, 2)),
            ("14bf4ff0020300237047", (2, 4, 2, 2)),
        ],
        ids=["thumb16-it", "thumb-mixed-it"],
    )
    def test_physical_rows_counts_evidence_and_boundaries(self, shared_bn, tmp_path, hex_bytes, widths):
        raw = bytes.fromhex(hex_bytes)
        selector, start = self._load_raw(shared_bn, tmp_path, raw, "linux-thumb2")
        offsets = [sum(widths[:i]) for i in range(len(widths))]
        addresses = [hex(start + offset) for offset in offsets]
        full = shared_bn.run("disasm", hex(start), "--format", "json", "--target", selector)
        assert full.returncode == 0, f"{full.stdout}\n{full.stderr}"
        listing = TestDisasmLinear._unwrap(json.loads(full.stdout))
        lines = listing["text"].splitlines()
        assert listing["total_lines"] == listing["returned_lines"] == len(widths)
        assert len(lines) == len(widths)
        texts = []
        for line, offset, width, mnemonic in zip(lines, offsets, widths, ("ite", "mov", "mov", "bx")):
            parts = line.split()
            assert int(parts[0], 16) == start + offset
            assert bytes.fromhex(" ".join(parts[1:1 + width])) == raw[offset:offset + width]
            assert parts[1 + width].startswith(mnemonic), line
            texts.append(" ".join(parts[1 + width:]))
        assert "r3" in texts[1] and "r3" in texts[2]

        sliced = shared_bn.run(
            "disasm", hex(start), "--lines", "2:3", "--format", "json", "--target", selector
        )
        assert sliced.returncode == 0, f"{sliced.stdout}\n{sliced.stderr}"
        window = TestDisasmLinear._unwrap(json.loads(sliced.stdout))
        assert window["text"].splitlines() == lines[1:3]
        assert window["total_lines"] == 4 and window["returned_lines"] == 2

        counted = shared_bn.run(
            "disasm", hex(start), "--count", "2", "--format", "json", "--target", selector
        )
        assert counted.returncode == 0, f"{counted.stdout}\n{counted.stderr}"
        assert TestDisasmLinear._unwrap(json.loads(counted.stdout))["text"].splitlines() == lines[:2]

        linear = shared_bn.run(
            "disasm", hex(start), "--linear", "2", "--mode", "thumb",
            "--format", "json", "--target", selector,
        )
        assert linear.returncode == 0, f"{linear.stdout}\n{linear.stderr}"
        decoded = TestDisasmLinear._unwrap(json.loads(linear.stdout))
        assert decoded["instruction_count"] == decoded["requested_count"] == 2
        assert [row["address"] for row in decoded["instructions"]] == addresses[:2]
        assert [row["length"] for row in decoded["instructions"]] == list(widths[:2])
        assert bytes.fromhex(" ".join(row["bytes"] for row in decoded["instructions"])) == raw[:sum(widths[:2])]
        assert [" ".join(row["text"].split()) for row in decoded["instructions"]] == texts[:2]

        evidence = shared_bn.run(
            "py", "exec",
            "import json\n"
            "from bn_agent_bridge.il_format import _structured_disasm_entries\n"
            "fn = bv.get_function_at(bv.start)\n"
            # BasicBlock.instruction_count aggregates IT; native disassembly
            # lines, not that analysis-span count, are the physical-row oracle.
            "native_addresses = sorted(line.address for block in fn.basic_blocks "
            "for line in block.disassembly_text)\n"
            "print('ROWS741=' + json.dumps({"
            "'entries': _structured_disasm_entries(bv, fn), "
            "'native_addresses': [hex(address) for address in native_addresses]}))",
            "--target", selector,
        )
        assert evidence.returncode == 0, f"{evidence.stdout}\n{evidence.stderr}"
        native = json.loads(next(
            line.removeprefix("ROWS741=")
            for line in evidence.stdout.splitlines() if line.startswith("ROWS741=")
        ))
        assert native["native_addresses"] == addresses
        assert [line.split()[0] for line in lines] == native["native_addresses"]
        assert [row["address"] for row in native["entries"]] == addresses
        assert [" ".join(row["text"].split()) for row in native["entries"]] == texts

        boundary = shared_bn.run(
            "disasm", addresses[1], "--linear", "1", "--snap-to-instruction",
            "--format", "json", "--target", selector,
        )
        assert boundary.returncode == 0, f"{boundary.stdout}\n{boundary.stderr}"
        at_start = TestDisasmLinear._unwrap(json.loads(boundary.stdout))
        assert at_start["address"] == addresses[1]
        assert at_start["boundary_warning"] is None and at_start["snapped_from"] is None
        if widths[1] == 4:
            snapped = shared_bn.run(
                "disasm", hex(start + 4), "--linear", "1", "--snap-to-instruction",
                "--format", "json", "--target", selector,
            )
            assert snapped.returncode == 0, f"{snapped.stdout}\n{snapped.stderr}"
            at_start = TestDisasmLinear._unwrap(json.loads(snapped.stdout))
            assert at_start["address"] == addresses[1]
            assert at_start["snapped_from"] == hex(start + 4)
            assert at_start["instructions"][0]["length"] == 4

    def test_forced_arm_retains_four_byte_rows(self, shared_bn, tmp_path):
        raw = bytes.fromhex("00f020e31eff2fe1")  # ARM nop; bx lr
        selector, start = self._load_raw(shared_bn, tmp_path, raw, "linux-armv7")
        result = shared_bn.run(
            "disasm", hex(start), "--linear", "2", "--mode", "arm",
            "--format", "json", "--target", selector,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        decoded = TestDisasmLinear._unwrap(json.loads(result.stdout))
        assert decoded["decode_arch"] == "armv7"
        assert [row["address"] for row in decoded["instructions"]] == [hex(start), hex(start + 4)]
        assert [row["length"] for row in decoded["instructions"]] == [4, 4]
        assert [row["text"].split()[0] for row in decoded["instructions"]] == ["nop", "bx"]
        assert bytes.fromhex(" ".join(row["bytes"] for row in decoded["instructions"])) == raw


def _build_and_prime_aarch64_probe(bridge, tmp_path_factory) -> Path:
    """Cross-build the AArch64 probe and pay for its analysis ONCE.

    Full analysis of the ~1.1k-function `-static` probe measures ~20s; saving
    its BNDB costs 0.6s and reloading through that sidecar measures 0.5s,
    because the bridge prefers an adjacent `<binary>.bndb` to the binary
    (#717). Priming here turns a two-test lane of 2 x 20s into 20s once plus a
    sub-second load per test -- and `SharedBridge.load()` copies the sidecar
    along with the binary, so each test still gets its own private view.

    A plain function, not the fixture body, so the budget guard below can call
    it with a recording stand-in instead of spending the 20s for real.
    """
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        pytest.skip("aarch64-linux-gnu-gcc not available")
    workdir = tmp_path_factory.mktemp("aarch64-probe")
    src = workdir / "probe.c"
    src.write_text("int add(int a, int b){return a + b;}\nint main(){return add(1, 2);}\n")
    probe = workdir / "probe_aarch64"
    build = subprocess.run(
        [cc, "-O0", "-no-pie", "-static", str(src), "-o", str(probe)],
        capture_output=True, text=True, timeout=120,
    )
    if build.returncode != 0:
        pytest.skip(f"aarch64 cross-compile failed: {build.stderr}")
    selector = bridge.load(probe, copy=False, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
    saved = bridge.run("save", "--target", selector, "--format", "json")
    assert saved.returncode == 0, f"priming save failed: {saved.stderr}\n{saved.stdout}"
    closed = bridge.run("close", selector)
    assert closed.returncode == 0, f"priming close failed: {closed.stderr}\n{closed.stdout}"
    return probe


@pytest.fixture(scope="session")
def aarch64_probe(_shared_bridge, tmp_path_factory) -> Path:
    return _build_and_prime_aarch64_probe(_shared_bridge, tmp_path_factory)


class TestDisasmLinearAArch64:
    """#600 real-path guard. BN 5.4 registers the AArch64 architecture as
    ``aarch64`` (never ``arm64`` -- it is not in ``Architecture`` at all), so the
    mocked ``arm64``-spelled regression test in tests/test_read_decompile.py
    cannot exercise the arch name the live linear-decode path actually sees. This
    drives the REAL aarch64 path end to end through the bridge: an odd linear
    start must NOT be masked as a Thumb function-pointer tag (AArch64 has no Thumb
    mode), and ``--mode arm|thumb`` must be rejected naming the real arch.

    NOTE on scope: this locks the real-path CONTRACT with the arch name BN
    actually emits. It does not by itself distinguish the #600 fix from its
    reversion, because for the real ``aarch64`` spelling BOTH the fixed gate and
    the pre-fix raw ``startswith("arm")/("thumb")`` gate already classify it as
    "not classic ARM/Thumb" (``"aarch64".startswith("arm")`` is False) -- i.e.
    the fix is a no-op for this spelling and only changes behavior for the
    synthetic ``arm64`` spelling BN never produces. The fix's mutation-sensitive
    guard therefore lives in the mocked ``arm64`` test; this test guards the real
    arch name against a future gate that WOULD mishandle it (e.g. a substring
    ``"arch" in name`` check, since ``"aarch64"`` contains ``"arch"``).
    """

    @staticmethod
    def _func_start(shared_bn, name: str) -> int:
        listing = shared_bn.run("function", "list", "--format", "json")
        assert listing.returncode == 0, listing.stderr
        funcs = TestDisasmLinear._items(json.loads(listing.stdout))
        matches = [f for f in funcs if f["name"] == name]
        assert matches, f"{name} not found among {len(funcs)} functions"
        return int(matches[0]["address"], 16)

    def test_aarch64_odd_linear_start_not_thumb_masked(self, shared_bn, aarch64_probe):
        shared_bn.load(aarch64_probe, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        start = self._func_start(shared_bn, "add")
        odd = start | 1  # poke bit 0 so a Thumb-masking gate would strip it
        res = shared_bn.run("disasm", hex(odd), "--linear", "2", "--format", "json")
        assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
        result = TestDisasmLinear._unwrap(json.loads(res.stdout))
        # Premise: the live arch name really is "aarch64", not "arm64".
        assert result["decode_arch"] == "aarch64", result
        # bit 0 preserved -- NOT masked back to the even address ...
        assert int(result["address"], 16) == odd, result
        # ... and no Thumb function-pointer-tag normalization was applied.
        assert "Thumb" not in result["note"], result["note"]

    def test_aarch64_rejects_arm_thumb_mode(self, shared_bn, aarch64_probe):
        # --mode arm|thumb is only meaningful for classic 32-bit ARM/Thumb. On a
        # real aarch64 target it must be rejected with the ACTUAL arch named.
        shared_bn.load(aarch64_probe, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        res = shared_bn.run("disasm", "add", "--linear", "2",
                            "--mode", "arm", "--format", "json")
        assert res.returncode != 0, res.stdout
        assert "aarch64" in (res.stdout + res.stderr).lower(), (res.stdout, res.stderr)


class TestSessionStartTimeoutDiagnostics:
    """#718: `session start` must not race its own measurement, and a timeout
    must carry the evidence of how far the start got.

    The failure path is driven with a stand-in CLI (a hanging interpreter, not a
    BN target) so the diagnostics can be asserted exactly, without spending
    minutes of real analysis on a deliberate hang.
    """

    # Stand-in for the `bn` console script: writes the per-instance log
    # breadcrumb the real spawn creates, emits a line on each stream, then hangs
    # without ever printing the JSON `_session_start` parses. The three markers
    # are assembled at run time on purpose: a bare `TimeoutExpired` prints the
    # command, and a literal marker in this source would let that repr satisfy
    # the assertions below without a single byte of real output.
    _HANGING_CLI = (
        "import os, pathlib, sys, time\n"
        "logs = pathlib.Path(os.environ['BN_CACHE_DIR']) / 'instances'\n"
        "logs.mkdir(parents=True, exist_ok=True)\n"
        "tag = 'before' + '-hang'\n"
        "(logs / 'stand-in.log').write_text('bridge-log' + '-line')\n"
        "print('partial-stdout-' + tag, flush=True)\n"
        "print('partial-stderr-' + tag, file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n"
    )

    @staticmethod
    def _hang(monkeypatch, program: str) -> None:
        monkeypatch.setattr(sys.modules[__name__], "_BN_CLI", [sys.executable, "-c", program])

    def test_cross_arch_lane_budget_clears_the_warm_cost(self):
        """The cross-arch lane must not run on a budget that IS the measurement.

        Analysing the aarch64 probe measures ~20s warm (a ~1.1k-function
        -static build). #718 was filed against a 30s ceiling -- all the margin
        was the measurement's own noise -- so the lane's budget must clear that
        ceiling and be larger than the general default, not equal to it.
        """
        assert _SESSION_START_TIMEOUT > 30.0
        assert _CROSS_ARCH_ANALYSIS_TIMEOUT > _SESSION_START_TIMEOUT

    def test_cross_arch_lane_actually_analyses_on_its_own_budget(self, monkeypatch,
                                                                 tmp_path_factory):
        """...and the lane must USE it. Asserting only the two constants left the
        delivered behaviour unguarded: dropping the budget from the call that
        analyses the probe silently puts it back on the general default with
        every test still green. Observe the budget the lane really asks for, so
        that regression is RED.

        The analysing call is now the `load` into the shared bridge, so the
        stand-in is a bridge whose `load` records the budget instead of doing
        20s of real analysis.
        """
        seen: dict[str, float] = {}

        class _RecordingBridge:
            def load(self, binary, *, copy=True, timeout=None):
                seen["timeout"] = timeout
                return str(binary)

            def run(self, *args, timeout=60.0):
                return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0] if a else [], 0, stdout="", stderr=""))

        _build_and_prime_aarch64_probe(_RecordingBridge(), tmp_path_factory)

        assert seen["timeout"] == _CROSS_ARCH_ANALYSIS_TIMEOUT, seen

    def test_every_cross_arch_analysis_asks_for_the_lane_budget(self):
        """...and the lane must be pinned to it, not merely have one.

        The test above observes today's helper. Nothing stopped a NEW lane test
        from calling the general `_session_start`, which puts the slow
        cross-built probe straight back on the general budget with every test
        green -- the shape #718 was filed against. The first cut of this
        property named ONE class and matched one call shape, so a second
        cross-arch class, or a start routed through a module-level helper, both
        escaped it. The second recognised a lane only by a toolchain name in a
        DIRECT positional argument -- so a lane whose name sits in a list
        literal (this module's dominant idiom, `subprocess.run([...])`) or in a
        module-level constant was not a lane at all, and ran on the general
        budget with all five #718 guards green.

        So: the population is every class OR module-level function (the probe
        fixture is one) that CROSS-COMPILES, recognised from any string
        constant it reaches -- wherever it sits in the expression, and through
        the module-level constants and helpers it names -- the call shape is
        any expression naming `_session_start` or `load` (the two ways a
        cross-built probe gets analysed: spawning a bridge around it, or
        loading it into the shared one), and reachability follows module-level
        helpers the lane calls.
        """
        import ast

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        helpers = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        CROSS_TOOLCHAIN = "-linux-gnu-"

        def names_of(call: ast.Call) -> set[str]:
            return ({name.id for name in ast.walk(call.func) if isinstance(name, ast.Name)}
                    | {attr.attr for attr in ast.walk(call.func) if isinstance(attr, ast.Attribute)})

        def cross_bound_names(node: ast.AST) -> set[str]:
            """Names bound INSIDE *node* to a value carrying a cross toolchain."""
            return {
                target.id if isinstance(target, ast.Name) else target.attr
                for child in ast.walk(node)
                if isinstance(child, (ast.Assign, ast.AnnAssign))
                for target in (child.targets if isinstance(child, ast.Assign)
                               else [child.target])
                if isinstance(target, (ast.Name, ast.Attribute))
                if child.value is not None and CROSS_TOOLCHAIN in ast.unparse(child.value)
            }

        # Module-scope bindings only; a name bound inside a class is added for
        # that class alone. Collecting every binding module-wide into one flat
        # set made a local name reused by four unrelated classes (`cc`) look
        # cross-compiling everywhere it appeared, and collecting only
        # module-scope bindings let a lane hide its triple in a CLASS attribute
        # reached through `cls.TRIPLE`. Scope is the answer to both.
        module_named = {
            target.id
            for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
            if node.value is not None and CROSS_TOOLCHAIN in ast.unparse(node.value)
        }

        def cross_compiles(node: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
            """Does this class INVOKE a cross toolchain? That -- not the class
            name, and not where in the argument expression the name happens to
            sit -- is what makes its `session start` slow enough to need the
            bigger budget.

            The toolchain has to reach a CALL's arguments, at any depth, so a
            name inside `subprocess.run([...])`'s list literal counts and this
            guard's own mention of the marker does not. The name may be a bare
            name or an attribute (`cls.TRIPLE`, `self.TRIPLE`), and is resolved
            in this scope plus module scope.
            """
            named = module_named | cross_bound_names(node)
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
                    for child in ast.walk(argument):
                        if (isinstance(child, ast.Constant) and isinstance(child.value, str)
                                and CROSS_TOOLCHAIN in child.value):
                            return True
                        if isinstance(child, ast.Name) and child.id in named:
                            return True
                        if isinstance(child, ast.Attribute) and child.attr in named:
                            return True
                for name in (names_of(call) & set(helpers)) - seen:
                    if cross_compiles(helpers[name], seen | {name}):
                        return True
            return False

        # A module-level function counts as a lane too: the fixture builder that
        # cross-compiles and primes the probe is where the 20s of analysis
        # actually happens, and it is not inside any class.
        #
        # ONE exemption, and it is this guard's own host class, named from the
        # running frame rather than spelled so it cannot drift onto a second
        # class: the tests here REFERENCE the builder (with a recording
        # stand-in, exactly to avoid paying for real analysis) and deliberately
        # start stand-in CLIs on a 2s hang budget. Reachability would therefore
        # class it as a cross-compiling lane and flag those 2s budgets --
        # measuring the measurement. Nothing in this class brings a real probe
        # into a real bridge.
        host = type(self).__name__
        lanes = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name != host
                 and cross_compiles(node)]
        assert lanes, "no cross-compiling lane found; update this guard"
        assert any(isinstance(node, ast.FunctionDef) for node in lanes), (
            "the probe builder is a module-level function; a population that "
            "sees only classes would not check the call that does the analysing")

        def unbudgeted(node: ast.AST, seen: frozenset[str]) -> list[str]:
            found: list[str] = []
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                reached = names_of(call)
                if reached & {"_session_start", "load"}:
                    budget = next((keyword.value for keyword in call.keywords
                                   if keyword.arg == "timeout"), None)
                    if not (isinstance(budget, ast.Name)
                            and budget.id == "_CROSS_ARCH_ANALYSIS_TIMEOUT"):
                        found.append(f"tests/test_integration.py:{call.lineno}")
                for name in (reached & set(helpers)) - seen:
                    found += unbudgeted(helpers[name], seen | {name})
            return found

        offenders = sorted({site for lane in lanes for site in unbudgeted(lane, frozenset())})
        assert not offenders, (
            "these analyse a cross-built probe without asking for "
            f"_CROSS_ARCH_ANALYSIS_TIMEOUT, so the slow probe runs on the "
            f"general budget: {offenders}"
        )

    def test_timed_out_start_carries_partial_output_and_bridge_log(self, monkeypatch):
        self._hang(monkeypatch, self._HANGING_CLI)
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _session_start("stand-in-probe", timeout=2.0)
        message = str(excinfo.value)
        assert "partial-stdout-before-hang" in message, message
        assert "partial-stderr-before-hang" in message, message
        assert "bridge-log-line" in message, message

    def test_timed_out_start_without_output_or_log_still_fails_loudly(self, monkeypatch):
        # A real hang: no partial output, no bridge log. It must still fail, and
        # say so, rather than degrade into a silent (or empty-message) pass.
        self._hang(monkeypatch, "import time; time.sleep(60)\n")
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _session_start(timeout=2.0)
        message = str(excinfo.value)
        assert "timed out after 2.0 seconds" in message, message
        assert "bridge log: none" in message, message


class TestFunctionCreatePreviewHonesty:
    """Regression for #304: `function create <addr> --preview` reported `verified`
    while the follow-up live `function create <addr>` reported
    `verification_failed`. The preview's revert used remove_user_function, which
    records a persistent user "no function here" override that poisoned the
    address. The non-poisoning remove_function makes preview and live agree.
    Needs real BN -- only BN's analysis reproduces the suppression behavior.
    """

    def _gap_addresses(self, shared_bn):
        """Candidate executable addresses that are NOT function starts: the byte
        just past a function when a gap precedes the next function."""
        listing = shared_bn.run("function", "list", "--format", "json")
        items = json.loads(listing.stdout)
        items = items.get("items", items) if isinstance(items, dict) else items
        fns = sorted(
            ((int(f["address"], 16), int(f.get("size") or 0)) for f in items),
            key=lambda t: t[0],
        )
        gaps = []
        for (start, size), (nxt, _) in zip(fns, fns[1:]):
            end = start + size
            if size > 0 and end < nxt:
                gaps.append(end)
        return gaps

    def test_preview_then_live_agree(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        chosen = None
        for addr in self._gap_addresses(shared_bn)[:20]:
            hexaddr = hex(addr)
            prev = shared_bn.run("function", "create", hexaddr,
                                 "--preview", "--format", "json")
            if prev.returncode != 0:
                continue
            status = json.loads(prev.stdout)["results"][0]["status"]
            if status == "verified":
                chosen = hexaddr
                break
        if chosen is None:
            pytest.skip("no creatable gap address found in this fixture")

        # The live create at the SAME address must ALSO verify -- before the
        # fix the preview's remove_user_function suppressed it and this
        # returned verification_failed.
        live = shared_bn.run("function", "create", chosen, "--format", "json")
        assert live.returncode == 0, f"{chosen}: {live.stdout}\n{live.stderr}"
        parsed = json.loads(live.stdout)
        assert parsed["results"][0]["status"] == "verified", parsed
        assert parsed["committed"] is True, parsed


class TestLoadCacheBndbRestore:
    """Regression for #318: a binary on a read-only mount has no writable adjacent
    .bndb, so `save` falls back to the writable cache. A later load of the same
    binary must RESTORE that cache copy (annotations preserved) instead of
    re-analyzing blank, which looked like total annotation loss. Only real BN
    exercises create_database's RO failure + the cache fallback round-trip."""

    def test_ro_mount_save_then_reload_restores_annotations(self, tmp_path):
        ro = tmp_path / "romnt"
        ro.mkdir()
        prog = ro / "prog"
        prog.write_bytes(Path(HELLO_BINARY).read_bytes())
        prog.chmod(0o755)
        inst = None
        cache_file = None
        try:
            info = _session_start(str(prog))  # raw load (no adjacent .bndb)
            inst = info["instance_id"]
            fns = json.loads(_bn("--instance", inst, "function", "list", "--format", "json").stdout)
            name = (fns.get("items") if isinstance(fns, dict) else fns)[0]["name"]
            renamed = _bn("--instance", inst, "rename", name, "RO318_MARKER", "--format", "json")
            assert renamed.returncode == 0, renamed.stderr

            ro.chmod(0o500)  # read-only mount: adjacent .bndb write will fail
            saved = _bn("--instance", inst, "save", "--format", "json")
            assert saved.returncode == 0, f"{saved.stdout}\n{saved.stderr}"
            sd = json.loads(saved.stdout)
            assert sd.get("fallback") is True, sd  # landed in the cache
            cache_file = Path(sd["path"])
            assert cache_file.exists()

            _session_stop(inst)
            inst = None

            # Reload from the still-read-only mount: must restore the cache copy.
            info2 = _session_start(str(prog))
            inst = info2["instance_id"]
            search = _bn("--instance", inst, "function", "search", "RO318_MARKER", "--format", "json")
            names = [i["name"] for i in json.loads(search.stdout).get("items", [])]
            assert "RO318_MARKER" in names, f"annotation lost on reload (blank): {names}"
        finally:
            if inst:
                _session_stop(inst)
            ro.chmod(0o700)
            if cache_file is not None and cache_file.exists():
                cache_file.unlink()


class TestBatchFunctionCreate:
    """Regression for #308: function_create works as a batch op -- N missed
    slots can be recovered atomically alongside other mutations, --preview
    reverts the whole batch, and the batch revert doesn't poison the address
    (uses the non-poisoning remove_function). Drives real BN."""

    @staticmethod
    def _gaps(shared_bn):
        listing = shared_bn.run("function", "list", "--format", "json")
        items = json.loads(listing.stdout)
        items = items.get("items", items) if isinstance(items, dict) else items
        fns = sorted(((int(f["address"], 16), int(f.get("size") or 0)) for f in items),
                     key=lambda t: t[0])
        return [start + size for (start, size), (nxt, _) in zip(fns, fns[1:])
                if size > 0 and start + size < nxt]

    def test_batch_function_create_preview_then_live_atomic(self, shared_bn, tmp_path):
        shared_bn.load(HELLO_BINARY)
        addr = None
        for cand in self._gaps(shared_bn)[:20]:
            mf = tmp_path / "probe.json"
            mf.write_text(json.dumps({"ops": [{"op": "function_create", "address": hex(cand)}]}))
            prev = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json")
            if prev.returncode == 0 and json.loads(prev.stdout)["results"][0]["status"] == "verified":
                addr = hex(cand)
                break
        if addr is None:
            pytest.skip("no creatable gap address found in this fixture")

        mf = tmp_path / "batch.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "function_create", "address": addr},
            {"op": "set_comment", "address": addr, "comment": "BATCH308"},
        ]}))

        # --preview: both ops verify, nothing commits, and the function is
        # reverted (not left behind).
        prev = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json")
        assert prev.returncode == 0, f"{prev.stdout}\n{prev.stderr}"
        pj = json.loads(prev.stdout)
        assert [r["status"] for r in pj["results"]] == ["verified", "verified"], pj
        assert pj["committed"] is False
        assert shared_bn.run("function", "info", addr).returncode != 0  # reverted

        # live: the batch commits atomically -- function AND comment persist.
        live = shared_bn.run("batch", "apply", str(mf), "--format", "json")
        assert live.returncode == 0, f"{live.stdout}\n{live.stderr}"
        lj = json.loads(live.stdout)
        assert [r["status"] for r in lj["results"]] == ["verified", "verified"], lj
        assert lj["committed"] is True
        assert shared_bn.run("function", "info", addr).returncode == 0  # now a function


class TestFunctionCreateSkippedAddress:
    """Regression for #360: function create must succeed on an address
    auto-analysis SKIPPED (a data-table / missed-handler entry). The handler
    uses the forced create_user_function; the advisory add_function declines
    exactly those addresses, so the op used to return verification_failed on its
    own documented use-case. Drives real BN."""

    def test_create_on_auto_skipped_address(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        # Find a caller-less function (reachable only indirectly -- the
        # data-table-handler shape), undefine it so the address becomes one
        # auto-analysis declines to recreate, and return its address.
        code = (
            "for f in bv.functions:\n"
            "    if f.start != bv.entry_point and len(list(bv.get_code_refs(f.start))) == 0:\n"
            "        a = f.start\n"
            "        bv.remove_user_function(f); bv.update_analysis_and_wait()\n"
            "        if bv.get_function_at(a) is None:\n"
            "            print('ADDR=' + hex(a)); break\n"
        )
        probe = shared_bn.run("py", "exec", code)
        line = next((l for l in probe.stdout.splitlines() if l.startswith("ADDR=")), None)
        if line is None:
            pytest.skip("no caller-less auto-skipped function in this fixture")
        addr = line.split("=", 1)[1].strip()

        # create on the skipped address: must verify and commit (#360). With
        # the advisory add_function this returned verification_failed.
        out = shared_bn.run("function", "create", addr, "--format", "json")
        assert out.returncode == 0, f"{out.stdout}\n{out.stderr}"
        res = json.loads(out.stdout)
        assert res["results"][0]["status"] == "verified", res
        assert res["committed"] is True, res
        assert shared_bn.run("function", "info", addr).returncode == 0

        # --preview on another skipped address verifies AND reverts cleanly,
        # and a subsequent live create still works (the revert must not poison
        # the address, #304).
        probe2 = shared_bn.run("py", "exec", code)
        line2 = next((l for l in probe2.stdout.splitlines() if l.startswith("ADDR=")), None)
        if line2 is not None:
            addr2 = line2.split("=", 1)[1].strip()
            prev = shared_bn.run("function", "create", addr2,
                                 "--preview", "--format", "json")
            assert json.loads(prev.stdout)["results"][0]["status"] == "verified"
            assert shared_bn.run("function", "info", addr2).returncode != 0
            live = shared_bn.run("function", "create", addr2, "--format", "json")
            assert json.loads(live.stdout)["results"][0]["status"] == "verified"
            assert json.loads(live.stdout)["committed"] is True


class TestTaintEmptyVerdictHonesty:
    """Regression for #310.1: a genuinely empty forward-taint result (no sink, no
    frontier) must carry the same loud 'NOT an all-clear' caveat the
    partial-coverage paths do -- it's exactly the shape a structurally-invisible
    bug produces, so it must be the most caveated case, not the least."""

    def test_empty_forward_verdict_is_caveated(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        fns = json.loads(shared_bn.run("function", "list", "--format", "json").stdout)
        names = [f["name"] for f in (fns.get("items") if isinstance(fns, dict) else fns)]
        saw_empty = False
        for name in names[:30]:
            out = shared_bn.run("taint", "forward", "-f", name, "--source", "param:0")
            # Only reason about a clean, non-spilled text result: a spilled
            # (truncated) render can cut the verdict line mid-string, which is
            # not a real "bare phrase without caveat". (No break: the caveat
            # invariant must hold for EVERY empty verdict, not just the first.)
            if out.returncode != 0 or "__BN_SPILLED__" in out.stdout:
                continue
            if "no taint reached any sink or frontier" in out.stdout:
                saw_empty = True
                assert "NOT an all-clear" in out.stdout, out.stdout
                assert "structurally see" in out.stdout, out.stdout
        if not saw_empty:
            pytest.skip("no empty-verdict function found in this fixture")


class TestTaintUnderRecoveredArgFrontier:
    """Regression for #381: a tainted caller argument flowing into a callee whose
    parameters BN under-recovered (Thumb 0-arity miss / variadic) must surface an
    honest frontier, not silently vanish. Forcing the callee to 0-arity via
    `proto set` deterministically simulates the recovery miss; needs real BN (and
    an ARM cross-compiler for the register-passed-arg shape)."""

    _SRC = (
        "#include <string.h>\n#include <stdio.h>\n"
        "__attribute__((noinline)) void build_cmd(char *arg){\n"
        "  char b1[64], b2[64], b3[128];\n"
        "  memcpy(b1, arg, 48); sprintf(b2, \"%s\", arg); strcpy(b3, arg);\n"
        "  printf(\"%s %s %s\\n\", b1, b2, b3);\n}\n"
        "int main(int argc, char **argv){ if (argc > 1) build_cmd(argv[1]); return 0; }\n"
    )

    def test_under_recovered_callee_arg_emits_frontier(self, shared_bn, tmp_path):
        import shutil
        cc = shutil.which("arm-linux-gnueabihf-gcc")
        if cc is None:
            pytest.skip("arm-linux-gnueabihf-gcc required for the register-arg shape")
        src = tmp_path / "vuln.c"
        src.write_text(self._SRC)
        binp = tmp_path / "vuln_arm"
        build = subprocess.run(
            [cc, "-O1", "-D_FORTIFY_SOURCE=2", "-marm", str(src), "-o", str(binp)],
            capture_output=True, text=True)
        if build.returncode != 0:
            pytest.skip(f"arm build failed: {build.stderr}")

        shared_bn.load(binp, copy=False, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        def _taint_leaves():
            out_file = tmp_path / "taint.json"
            shared_bn.run("taint", "forward", "-f", "main",
                          "--source", "param:1", "--format", "json", "--out", str(out_file))
            return json.loads(out_file.read_text()).get("leaves", [])

        def _frontiers(leaves):
            return [l for l in leaves if "under-recovered" in str(l.get("note", ""))]

        # Baseline: build_cmd recovered with its arg -> no #381 frontier.
        assert _frontiers(_taint_leaves()) == []

        # Force the recovery miss: build_cmd as 0-arity.
        shared_bn.run("proto", "set", "build_cmd",
                      "void build_cmd(void)", "--format", "json")

        # Now the tainted argv arg into the under-recovered callee must be an
        # honest frontier, not a silent drop.
        frontiers = _frontiers(_taint_leaves())
        assert frontiers, "expected a #381 under-recovered-arg frontier"
        assert frontiers[0].get("kind") == "unmodeled_callee"
        assert frontiers[0].get("callee", {}).get("name") == "build_cmd"


class TestTaintArgRegisterFallback:
    """Regression for #433: seeding `arg:memcpy:2` (backward taint) or `trace --arg 2`
    on a copy sink whose MLIL under-recovered its call args -- an ARM-Thumb IFUNC/
    veneer copy sink surfaces only the first register arg -- must recover the length
    from the calling-convention register (r2) instead of a hard "out of range"
    dead-end. Forcing memcpy to 1-arity via `proto set` deterministically simulates
    the under-recovery; needs real BN + an ARM cross-compiler."""

    _SRC = (
        "#include <string.h>\n#include <unistd.h>\n"
        "__attribute__((noinline)) void do_copy(char *dst, char *src, int n){\n"
        "  memcpy(dst, src, n - 4);\n}\n"
        "int main(int argc, char **argv){\n"
        "  char d[256], s[256];\n"
        "  int n = read(0, s, 200);\n"
        "  do_copy(d, s, n);\n  return 0;\n}\n"
    )

    def test_arg_register_fallback_backward_and_trace(self, shared_bn, tmp_path):
        import shutil
        cc = shutil.which("arm-linux-gnueabihf-gcc")
        if cc is None:
            pytest.skip("arm-linux-gnueabihf-gcc required for the register-arg shape")
        src = tmp_path / "argreg.c"
        src.write_text(self._SRC)
        binp = tmp_path / "argreg_arm"
        build = subprocess.run(
            [cc, "-O1", "-marm", str(src), "-o", str(binp)],
            capture_output=True, text=True)
        if build.returncode != 0:
            pytest.skip(f"arm build failed: {build.stderr}")

        shared_bn.load(binp, copy=False, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        # Force the recovery miss: memcpy as 1-arity, so arg:memcpy:2 (the length,
        # in r2) is out of range and only the register fallback can seed it.
        shared_bn.run("proto", "set", "memcpy",
                      "void* memcpy(void* dst)", "--format", "json")

        # Backward: the canonical `arg:memcpy:2` length seed resolves to a slice
        # (register-recovered), not a dead-end, and discloses the #433 caveat.
        bw = tmp_path / "bw.json"
        r = shared_bn.run("taint", "backward", "-f", "do_copy",
                          "--sink", "arg:memcpy:2", "--format", "json", "--out", str(bw))
        assert r.returncode == 0, r.stderr
        res = json.loads(bw.read_text())
        assert res.get("slices"), "backward: expected a #433 register-recovered slice"
        assert any("#433" in a for a in res.get("assumptions", [])), \
            "backward: expected the #433 register-recovery caveat"
        seeds = " ".join(str(sl.get("sink", {}).get("seed")) for sl in res["slices"])
        assert "r2" in seeds, f"expected the r2 length register as the seed, got {seeds!r}"
        call_addr = res["slices"][0]["sink"]["address"]

        # trace --arg 2 at the same call recovers arg 2 from r2 and discloses #433.
        tr = tmp_path / "tr.json"
        r2 = shared_bn.run("trace", "do_copy", str(call_addr), "--arg", "2",
                           "--format", "json", "--out", str(tr))
        assert r2.returncode == 0, r2.stderr
        tres = json.loads(tr.read_text())
        assert tres.get("step_count", 0) > 0, "trace: expected a #433 register-recovered trace"
        assert tres.get("arg_label", {}).get("register") == "r2"
        assert any("#433" in h for h in tres.get("hints", [])), \
            "trace: expected the #433 register-recovery hint"


class TestTagRoundtrip:
    """Real-BN round trip for the `bn tag` group: a custom tag type, a
    FUNCTION-scope tag, and an ADDRESS-scope tag. The address-scope readback
    is the key dogfood check here: our code (read_tags._collect_tags /
    _get_tags) assumes real BN's `Function.tags` / `Function.get_tags_at`
    surface address-scope tags the way the mocked fakes in test_tags.py model
    them -- only a real BN Function object can prove that assumption."""

    def _first_function_address(self, shared_bn) -> str:
        out = shared_bn.run("function", "list", "--limit", "1", "--format", "json")
        assert out.returncode == 0, f"{out.stdout}\n{out.stderr}"
        listing = json.loads(out.stdout)
        items = listing.get("items") if isinstance(listing, dict) else listing
        return items[0]["address"]

    def test_tag_roundtrip_across_scopes(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        created = shared_bn.run("tag", "type", "create", "AgentNote",
                                "--icon", "\U0001F916", "--format", "json")
        assert created.returncode == 0, f"{created.stdout}\n{created.stderr}"
        assert json.loads(created.stdout)["results"][0]["status"] in ("verified", "noop")

        fn_addr = self._first_function_address(shared_bn)

        # FUNCTION-scope tag.
        add_fn = shared_bn.run("tag", "add", "--function", fn_addr,
                              "--type", "AgentNote", "--data", "reviewed by agent",
                              "--format", "json")
        assert add_fn.returncode == 0, f"{add_fn.stdout}\n{add_fn.stderr}"
        assert json.loads(add_fn.stdout)["results"][0]["status"] == "verified"

        # ADDRESS-scope tag at the function's entry address (still an
        # address tag, not a function tag -- distinct scope, distinct id).
        add_addr = shared_bn.run("tag", "add", fn_addr,
                                 "--type", "AgentNote", "--data", "flagged address",
                                 "--format", "json")
        assert add_addr.returncode == 0, f"{add_addr.stdout}\n{add_addr.stderr}"
        assert json.loads(add_addr.stdout)["results"][0]["status"] == "verified"

        # `tag list` sweeps Function.get_function_tags() AND Function.tags --
        # both scopes must be found (the address-scope entry is the key
        # real-BN dogfood check, see class docstring).
        listing = json.loads(shared_bn.run("tag", "list",
                                           "--type", "AgentNote", "--format", "json").stdout)
        items = listing["items"]
        scopes_and_data = {(t["scope"], t["data"]) for t in items}
        assert ("function", "reviewed by agent") in scopes_and_data, items
        assert ("address", "flagged address") in scopes_and_data, items

        # `tag get <addr>` independently surfaces the address-scope tag via
        # Function.get_tags_at -- a second, distinct real-BN code path.
        got = json.loads(shared_bn.run("tag", "get", fn_addr,
                                       "--format", "json").stdout)
        assert any(t["scope"] == "address" and t["data"] == "flagged address"
                  for t in got["tags"]), got

        # Remove both tags (each `tag remove` targets one scope).
        rm_addr = shared_bn.run("tag", "remove", fn_addr,
                                "--type", "AgentNote", "--format", "json")
        assert rm_addr.returncode == 0, f"{rm_addr.stdout}\n{rm_addr.stderr}"
        assert json.loads(rm_addr.stdout)["results"][0]["status"] == "verified"

        rm_fn = shared_bn.run("tag", "remove", "--function", fn_addr,
                             "--type", "AgentNote", "--format", "json")
        assert rm_fn.returncode == 0, f"{rm_fn.stdout}\n{rm_fn.stderr}"
        assert json.loads(rm_fn.stdout)["results"][0]["status"] == "verified"

        # Now that no tags of this type remain, the custom tag type itself
        # can be removed.
        rm_type = shared_bn.run("tag", "type", "remove", "AgentNote",
                                "--format", "json")
        assert rm_type.returncode == 0, f"{rm_type.stdout}\n{rm_type.stderr}"
        assert json.loads(rm_type.stdout)["results"][0]["status"] == "verified"

    def test_tag_type_remove_refuses_builtin(self, shared_bn):
        """A built-in tag type (e.g. Bookmarks) must be refused with a clean
        invalid_request, not removed -- mirrors the bitfield-rejection shape in
        TestTypesDeclareBitfield above."""
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run("tag", "type", "remove", "Bookmarks",
                            "--format", "json")
        assert res.returncode != 0, f"expected rejection, got: {res.stdout}"
        parsed = json.loads(res.stdout)
        results = parsed.get("results") or [parsed]
        assert results[0].get("status") == "invalid_request", parsed
        assert "built-in" in (results[0].get("message") or "").lower(), parsed
        # Must NOT have been removed -- still present and flagged built-in.
        types = json.loads(shared_bn.run("tag", "types",
                                         "--format", "json").stdout)["tag_types"]
        bookmarks = next((t for t in types if t["name"] == "Bookmarks"), None)
        assert bookmarks is not None and bookmarks["is_builtin"] is True, types


class TestFunctionDocRoundtrip:
    """Real-BN round trip for the function-doc surface: `comment --function`
    now targets `fn.comment` (the function's documentation comment shown atop
    the function), not an address comment."""

    def test_function_doc_set_get_delete(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        listing = json.loads(shared_bn.run("function", "list",
                                           "--limit", "1", "--format", "json").stdout)
        items = listing.get("items") if isinstance(listing, dict) else listing
        fn_addr = items[0]["address"]

        doc_text = "AgentNote: reviewed and documented by an integration test"
        set_res = shared_bn.run("comment", "set", "--function", fn_addr,
                                doc_text, "--format", "json")
        assert set_res.returncode == 0, f"{set_res.stdout}\n{set_res.stderr}"
        assert json.loads(set_res.stdout)["results"][0]["status"] == "verified"

        got = json.loads(shared_bn.run("comment", "get", "--function", fn_addr,
                                       "--format", "json").stdout)
        assert got["function_doc"] == doc_text, got
        assert got["has_function_doc"] is True, got

        del_res = shared_bn.run("comment", "delete", "--function", fn_addr,
                                "--format", "json")
        assert del_res.returncode == 0, f"{del_res.stdout}\n{del_res.stderr}"
        assert json.loads(del_res.stdout)["results"][0]["status"] == "verified"

        after = json.loads(shared_bn.run("comment", "get", "--function", fn_addr,
                                         "--format", "json").stdout)
        assert after["function_doc"] == "", after
        assert after["has_function_doc"] is False, after


class TestBareVoidCallHlilStatement:
    """Regression for #644: `hlil_statement` was null with reason
    `no_local_statement` for EVERY call whose return value is discarded -- i.e.
    every bare call STATEMENT, the most common shape in real code (memcpy, strcpy,
    memset, sprintf, free). On one dogfood target that was 0 of 387 callsites
    resolved, sending the agent back to `decompile` + manual address correlation.

    Verified against real BN on this fixture before the fix: the `memcpy` call in
    `parse_record` reported `hlil_statement: None`, reason `no_local_statement`,
    while `bn il --view hlil` rendered the statement at the identical address."""

    def test_discarded_return_call_resolves_its_statement(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        ev = json.loads(shared_bn.run("evidence", "function", "parse_record",
                                      "--format", "json").stdout)
        calls = ev["calls"]
        memcpy_calls = [
            c for c in calls
            if "memcpy" in str(((c.get("target") or {}).get("function") or {}).get("name", ""))
        ]
        assert memcpy_calls, f"no memcpy callsite found: {calls}"
        for c in memcpy_calls:
            # `memcpy(dst, src, n)` discards its return, so its HLIL parent is
            # the enclosing Block -- the shape that used to null out.
            assert c["hlil_statement"], (
                f"bare void call statement still unresolved: "
                f"reason={c['hlil_statement_reason']!r}")
            assert "memcpy" in c["hlil_statement"]
            assert c["hlil_statement_reason"] is None

    def test_return_used_call_still_resolves(self, shared_bn):
        """The #475/#490 shapes must keep working: `int k = parse_record(...)` has a
        real assignment parent, and the ancestor walk (not the new root fallback)
        must still be what answers it."""
        shared_bn.load(PARSER_BINARY)
        ev = json.loads(shared_bn.run("evidence", "function", "main",
                                      "--format", "json").stdout)
        resolved = [c for c in ev["calls"]
                    if "parse_record" in str(((c.get("target") or {}).get("function") or {})
                                             .get("name", ""))]
        assert resolved, f"no parse_record callsite in main: {ev['calls']}"
        assert all(c["hlil_statement"] for c in resolved)


class TestArgumentArityConfidence:
    """Regression for #648: `argument_confidence: authoritative` meant "HLIL
    produced a list", not "the list is right" -- on an unknown-arity callee BN
    assumes every argument register is live and HLIL renders whatever sits in them
    (a neighbouring call's staging, the stack canary). This is the negative
    control that keeps the fix from blanket-demoting everything: `memcpy` has a
    bundled 3-parameter prototype, so its arguments really ARE authoritative."""

    def test_known_prototype_callee_stays_authoritative(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        ev = json.loads(shared_bn.run("evidence", "function", "parse_record",
                                      "--format", "json").stdout)
        memcpy_calls = [
            c for c in ev["calls"]
            if "memcpy" in str(((c.get("target") or {}).get("function") or {}).get("name", ""))
        ]
        assert memcpy_calls
        for c in memcpy_calls:
            assert c["argument_confidence"] == "authoritative", c
            assert c["arity_unknown"] is False, c
            assert "abi_register_saturated" not in c, c

    _VENDOR_SRC = (
        "int vendor_get_status(int code, int flags, int retries,\n"
        "                       int timeout, int mode, int reserved) {\n"
        "  return code + flags + retries + timeout + mode + reserved;\n"
        "}\n"
    )
    _CALLER_SRC = (
        "extern int vendor_get_status(int code, int flags, int retries,\n"
        "                              int timeout, int mode, int reserved);\n"
        "__attribute__((noinline, used)) int probe_device(void) {\n"
        "  return vendor_get_status(1, 2, 3, 4, 5, 6);\n"
        "}\n"
        "int main(void) { return probe_device(); }\n"
    )

    def test_demotion_fires(self, shared_bn, tmp_path):
        """#648 positive control: the demotion must actually FIRE, not just
        decline to fire on a known prototype. A call to a DYNAMICALLY-linked
        "vendor" import BN's bundled type library has no signature for (unlike
        memcpy/printf, which are covered) analyzes with zero declared
        parameters, so its `argument_confidence` must demote off
        `authoritative` and flag `arity_unknown`. Needs real BN + a C compiler
        that can build a small shared library."""
        import shutil
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            pytest.skip("a C compiler is required to build the vendor-import fixture")

        vendor_src = tmp_path / "vendor.c"
        vendor_src.write_text(self._VENDOR_SRC)
        libvendor = tmp_path / "libvendor.so"
        build_lib = subprocess.run(
            [cc, "-O0", "-fPIC", "-shared", "-o", str(libvendor), str(vendor_src)],
            capture_output=True, text=True)
        if build_lib.returncode != 0:
            pytest.skip(f"vendor shared-library build failed: {build_lib.stderr}")

        caller_src = tmp_path / "caller.c"
        caller_src.write_text(self._CALLER_SRC)
        binp = tmp_path / "vendor_caller_x86_64"
        build_bin = subprocess.run(
            [cc, "-O0", "-fno-stack-protector", "-no-pie",
             str(caller_src), "-o", str(binp),
             "-L", str(tmp_path), "-lvendor",
             "-Wl,-rpath," + str(tmp_path)],
            capture_output=True, text=True)
        if build_bin.returncode != 0:
            pytest.skip(f"vendor-import caller build failed: {build_bin.stderr}")

        shared_bn.load(binp, copy=False)
        ev = json.loads(shared_bn.run("evidence", "function", "probe_device",
                                      "--format", "json").stdout)
        vendor_calls = [
            c for c in ev["calls"]
            if "vendor_get_status" in str(((c.get("target") or {}).get("function") or {}).get("name", ""))
        ]
        assert vendor_calls, "expected a call to the unprototyped vendor import"
        for c in vendor_calls:
            assert c["arity_unknown"] is True, c
            assert c["argument_confidence"] != "authoritative", c


class TestUserPrototypeArityConfidence742:
    """Use real call-type adjustments, not an accidental compiler recovery quirk."""

    _SOURCE = (
        "__attribute__((noinline)) int arity_sink(int a, int b, int c) {\n"
        "  return a + b + c;\n"
        "}\n"
        "__attribute__((noinline)) int arity_probe(void) {\n"
        "  return arity_sink(1, 2, 3);\n"
        "}\n"
        "__attribute__((noinline)) int arity_dispatch(int (*fn)(int, int, int)) {\n"
        "  return fn(4, 5, 6);\n"
        "}\n"
        "int main(void) { return arity_probe() + arity_dispatch(arity_sink); }\n"
    )

    def test_real_user_prototype_counts_and_indirect_dispatch(self, shared_bn, tmp_path):
        cc = shutil.which("cc") or shutil.which("gcc")
        assert cc is not None, "a C compiler is required for the issue742 real-BN regression"
        source = tmp_path / "arity_fixture.c"
        source.write_text(self._SOURCE)
        binary = tmp_path / "arity_fixture"
        build = subprocess.run(
            [cc, "-O0", "-fno-inline", "-fno-builtin", "-fno-stack-protector", "-no-pie",
             str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=60)
        assert build.returncode == 0, f"arity fixture build failed:\n{build.stdout}\n{build.stderr}"
        target = shared_bn.load(binary, copy=False)
        shared_bn.json(
            "py", "exec",
            "sink = bv.get_functions_by_name('arity_sink')[0]\n"
            "sink.type = 'int arity_sink(int a, int b, int c)'\n"
            "bv.update_analysis_and_wait()\n",
            "-t", target)

        # Changing only the call-site type gives genuine short/empty HLIL while
        # leaving the callee's known three-parameter prototype intact. Clearing
        # the adjustment is the matching-count negative control.
        for adjustment, expected_count in (
            ("int adjusted(int a)", 1),
            ("int adjusted(void)", 0),
            (None, 3),
        ):
            observed = shared_bn.json(
                "py", "exec",
                "caller = bv.get_functions_by_name('arity_probe')[0]\n"
                "sites = list(caller.call_sites)\n"
                "assert len(sites) == 1, sites\n"
                "call_address = sites[0].address\n"
                f"caller.set_call_type_adjustment(call_address, {adjustment!r})\n"
                "bv.update_analysis_and_wait()\n"
                "caller = bv.get_function_at(caller.start)\n"
                "sink = bv.get_functions_by_name('arity_sink')[0]\n"
                "result = {\n"
                "    'call_address': hex(call_address),\n"
                "    'has_user_type': sink.has_user_type,\n"
                "    'declared_count': len(sink.type.parameters),\n"
                "    'hlil_calls': list(caller.hlil.traverse(lambda node: {\n"
                "        'address': hex(node.address),\n"
                "        'parameter_count': len(node.params)\n"
                "    } if node.operation in (bn.HighLevelILOperation.HLIL_CALL,\n"
                "                             bn.HighLevelILOperation.HLIL_TAILCALL) else None)),\n"
                "}\n",
                "-t", target)["result"]
            # Preconditions come directly from BN's prototype and raw HLIL,
            # independently of the evidence argument-selection implementation.
            assert observed["has_user_type"] is True, observed
            assert observed["declared_count"] == 3, observed
            assert observed["hlil_calls"] == [{
                "address": observed["call_address"], "parameter_count": expected_count,
            }], observed

            evidence = shared_bn.json("evidence", "function", "arity_probe", "-t", target)
            assert len(evidence["calls"]) == 1, evidence
            call = evidence["calls"][0]
            assert call["address"] == observed["call_address"], call
            assert call["direct"] is True, call
            assert call["argument_source"] == "hlil", call
            assert len(call["arguments"]) == expected_count, call
            assert call["arity_unknown"] is False, call
            if expected_count < 3:
                assert call["arity_mismatch"] is True, call
                assert call["declared_arity"] == 3, call
                assert call["argument_confidence"] == "inferred", call
            else:
                assert "arity_mismatch" not in call, call
                assert call["argument_confidence"] == "authoritative", call

        indirect = shared_bn.json("evidence", "function", "arity_dispatch", "-t", target)
        assert len(indirect["calls"]) == 1, indirect
        call = indirect["calls"][0]
        assert call["direct"] is False, call
        assert call["indirect_call"] is True, call
        assert call["callee_unresolved"] is True, call
        assert call["argument_confidence"] == "heuristic", call
        assert "arity_mismatch" not in call, call


class TestDataRetypeRoundtrip:
    """Regression for #649: typing a recovered data variable had NO verified
    mutation path -- `types declare` defines a struct but cannot apply it,
    `symbol rename --kind data` renames without typing, and `struct field set`
    edits a type rather than a variable's binding -- so the only way through was
    `bn py exec`: no --preview, no readback verification, no batch atomicity, no
    audit trail. Only real BN exercises define_user_data_var + BNDB persistence."""

    @staticmethod
    def _writable_data_address(run) -> str:
        """*run* takes CLI args and returns a CompletedProcess: `shared_bn.run`
        on the shared bridge, or a private-session wrapper for the persistence
        test below, which needs a bridge it can stop and restart."""
        secs = json.loads(run("sections", "--format", "json").stdout)
        items = secs.get("items") if isinstance(secs, dict) else secs
        by_name = {s.get("name"): s for s in items if isinstance(s, dict)}
        for name in (".data", ".bss", ".rodata"):
            if name in by_name:
                return by_name[name]["start"]
        raise AssertionError(f"no data section found: {list(by_name)}")

    def test_declare_then_retype_verifies_and_persists(self, tmp_path):
        # Its own bridge, not the shared one: the BNDB round trip this proves IS
        # a stop and a restart, which is exactly what the shared bridge removes.
        prog = tmp_path / "prog"
        prog.write_bytes(Path(DISPATCH_BINARY).read_bytes())
        prog.chmod(0o755)
        inst = None
        try:
            inst = _session_start(str(prog))["instance_id"]
            addr = self._writable_data_address(
                lambda *args: _bn("--instance", inst, *args))

            declared = _bn("--instance", inst, "types", "declare",
                           "struct bn649_entry { char* desc; char* usage; };",
                           "--format", "json")
            assert declared.returncode == 0, f"{declared.stdout}\n{declared.stderr}"

            typed = _bn("--instance", inst, "data", "retype", addr, "bn649_entry[2]",
                        "--format", "json")
            assert typed.returncode == 0, f"{typed.stdout}\n{typed.stderr}"
            row = json.loads(typed.stdout)["results"][0]
            assert row["status"] == "verified", row
            assert row["expected_type"] == "struct bn649_entry[0x2]", row

            # Re-applying the SAME type is a noop -- i.e. a real readback, not a
            # claim. This is also how persistence is checked below.
            again = json.loads(_bn("--instance", inst, "data", "retype", addr,
                                   "bn649_entry[2]", "--format", "json").stdout)
            assert again["results"][0]["status"] == "noop", again

            saved = _bn("--instance", inst, "save", "--format", "json")
            assert saved.returncode == 0, f"{saved.stdout}\n{saved.stderr}"
            _session_stop(inst)
            inst = None

            # Reopen and confirm the type survived the BNDB round trip: the same
            # retype must still report `noop`, not `verified`.
            inst = _session_start(str(prog))["instance_id"]
            after = json.loads(_bn("--instance", inst, "data", "retype", addr,
                                   "bn649_entry[2]", "--format", "json").stdout)
            assert after["results"][0]["status"] == "noop", (
                f"data-var type did not persist in the BNDB: {after}")
        finally:
            if inst:
                _session_stop(inst)

    def test_preview_reverts_the_data_var_type(self, shared_bn):
        shared_bn.load(DISPATCH_BINARY)
        addr = self._writable_data_address(shared_bn.run)

        previewed = shared_bn.run("data", "retype", "--preview", addr,
                                  "uint64_t[4]", "--format", "json")
        assert previewed.returncode == 0, f"{previewed.stdout}\n{previewed.stderr}"
        payload = json.loads(previewed.stdout)
        assert payload["results"][0]["status"] == "verified", payload
        assert payload["committed"] is False and payload["rolled_back"] is True

        # The preview reverted, so applying it live now must CHANGE the view
        # (`verified`, not `noop`) -- proof the preview left nothing behind.
        live = json.loads(shared_bn.run("data", "retype", addr,
                                        "uint64_t[4]", "--format", "json").stdout)
        assert live["results"][0]["status"] == "verified", live


class TestCommentListFunctionDocs:
    """Regression for #643: `comment list` enumerated only bv.address_comments, so
    a function doc written by `comment set --function` was invisible to the only
    discovery command -- the write reported `verified` and `comment list --query
    TODO` reported nothing existed, silently breaking the resume/handoff workflow
    the bn-re skill prescribes."""

    def test_function_doc_is_discoverable_by_query(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        listing = json.loads(shared_bn.run("function", "list",
                                           "--limit", "1", "--format", "json").stdout)
        fn_addr = (listing.get("items") or listing)[0]["address"]
        doc = "Dispatcher; TODO643: confirm shm bounds"
        assert shared_bn.run("comment", "set", "--function", fn_addr, doc,
                             "--format", "json").returncode == 0

        found = json.loads(shared_bn.run("comment", "list",
                                         "--query", "TODO643", "--format", "json").stdout)
        assert found["total"] == 1, found
        assert found["items"][0]["scope"] == "function_doc", found
        assert found["items"][0]["comment"] == doc, found

        # --scope address is the old behaviour, still expressible.
        narrowed = json.loads(shared_bn.run("comment", "list",
                                            "--query", "TODO643", "--scope", "address",
                                            "--format", "json").stdout)
        assert narrowed["total"] == 0, narrowed


class TestTypesDeclarePartialDrop760:
    """#760: the platform parser DROPS a declaration whose name collides with a
    built-in type instead of raising, so `struct uint32_t { … }; struct cfg_t { … };`
    applied one type, discarded the other, and reported `verified` with exit 0.

    That premise is a property of the REAL parser -- the mocked lane has to hardcode
    it -- so this class is the only place the guard is validated against the
    behaviour it exists for, and the only place a change in BN's drop behaviour would
    be caught (the guard would go quiet, and these tests with it).
    """

    MIXED = "struct uint32_t { int shadow_x; }; struct rv_drop_probe_t { int y; };"

    def _exists(self, shared_bn, name: str) -> bool:
        """True when the view resolves *name*.

        `types show <missing>` exits 2 and writes its "Type not found" message to
        STDERR, so stdout is not a usable signal -- the exit code is.
        """
        return shared_bn.run("types", "show", name).returncode == 0

    def test_a_single_named_type_declaration_is_accepted(self, shared_bn):
        """Baseline on the real parser: the non-colliding half of the mixed string is
        accepted on its own, so the refusal below is about the dropped declaration and
        not about the declaration syntax."""
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run("types", "declare", "struct rv_ok_probe_t { int y; };",
                            "--format", "json")
        assert res.returncode == 0, res.stdout
        assert [r.get("status") for r in json.loads(res.stdout)["results"]] == ["verified"]
        assert self._exists(shared_bn, "rv_ok_probe_t")

    def test_mixed_declaration_is_refused_and_nothing_lands(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run("types", "declare", self.MIXED, "--format", "json")
        assert res.returncode == 3, res.stdout
        parsed = json.loads(res.stdout)
        assert parsed["success"] is False, parsed
        result = parsed["results"][0]
        assert result["status"] == "invalid_request", result
        assert result["observed"]["dropped_declarations"] == [
            "struct uint32_t { int shadow_x; };"
        ], result
        # Refused before anything was applied: neither name resolves afterwards.
        assert not self._exists(shared_bn, "rv_drop_probe_t")
        assert not self._exists(shared_bn, "uint32_t")

    def test_attribute_prefixed_drop_is_refused(self, shared_bn):
        """The prefix shape #760's harm also reproduces on: the drop used to stay
        silent behind `__attribute__((packed))`."""
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run(
            "types", "declare",
            "__attribute__((packed)) struct uint32_t { int shadow_x; }; "
            "struct rv_pack_t { int y; };",
            "--format", "json",
        )
        assert res.returncode == 3, res.stdout
        assert json.loads(res.stdout)["results"][0]["status"] == "invalid_request"
        assert not self._exists(shared_bn, "rv_pack_t")

    def test_all_good_multi_declaration_still_applies(self, shared_bn):
        """Negative control on the real parser: a multi-declaration whose fragments
        all define a type is not refused (--preview, so the shared view stays clean)."""
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run(
            "types", "declare",
            "struct rv_a_t { int a; }; struct rv_b_t { int b; };",
            "--preview", "--format", "json",
        )
        assert res.returncode == 0, res.stdout
        parsed = json.loads(res.stdout)
        assert [r.get("status") for r in parsed["results"]] == ["verified"], parsed
        assert parsed["committed"] is False, parsed
