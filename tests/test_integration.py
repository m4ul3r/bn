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

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"
ADD_BINARY = FIXTURES_DIR / "add_x86_64"
DISPATCH_BINARY = FIXTURES_DIR / "dispatch_table_x86_64"

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
# Don't litter the bn repo with `.bn-<id>` project markers (#80) during integration
# runs (`bn load` from the repo cwd would otherwise drop one here + touch
# .git/info/exclude). The marker path has its own unit coverage.
#
# Built at call time, not import time (#589): conftest's autouse `_hermetic_env`
# fixture pins BN_CACHE_DIR/NO_COLOR per test, and a module-import-time snapshot
# of os.environ would predate every fixture -- so the subprocesses these helpers
# spawn would read the developer's real ~/.cache/bn instead of the isolated one.
def _env() -> dict[str, str]:
    return {**os.environ, "BN_NO_MARKERS": "1"}


def _bn(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_BN_CLI, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
    )


def _session_start(*binaries: str, timeout: float = 30.0) -> dict:
    # session start defaults to text output; this helper parses JSON.
    cmd = [*_BN_CLI, "session", "start", "--format", "json"]
    cmd.extend(str(b) for b in binaries)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env())
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

    def test_save_path_keeps_original_selector(self, tmp_path):
        # Two targets in one instance so the selector is REQUIRED and name-based.
        info = _session_start(str(HELLO_BINARY), str(ADD_BINARY))
        inst = info["instance_id"]
        try:
            listing = json.loads(
                _bn("--instance", inst, "target", "list", "--format", "json").stdout)["items"]
            hello = next(t for t in listing
                         if "hello" in (t.get("filename", "") + t.get("basename", "")))
            sel = hello.get("selector") or hello.get("basename")

            copy = str(tmp_path / "copy.bndb")
            saved = _bn("--instance", inst, "save", "--target", sel, "--path", copy,
                        "--format", "json")
            assert saved.returncode == 0, saved.stderr
            assert json.loads(saved.stdout).get("saved") is True
            assert Path(copy).exists()

            # The original selector must STILL resolve -- before the fix the live
            # target was rebound to copy.bndb and `sel` raised "not found".
            after = _bn("--instance", inst, "target", "info", "--target", sel,
                        "--format", "json")
            assert after.returncode == 0, (
                f"original selector {sel!r} stopped resolving after save --path: "
                f"{after.stdout} {after.stderr}")
        finally:
            _session_stop(inst)


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

    def _first_fn(self, inst_id):
        out = _bn("--instance", inst_id, "function", "list", "--format", "json")
        return json.loads(out.stdout)["items"][0]["name"]

    def test_unnamed_params_verify(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            fn = self._first_fn(inst_id)
            res = _bn("--instance", inst_id, "proto", "set", fn,
                      f"void {fn}(int32_t, char**, char**)", "--format", "json")
            parsed = json.loads(res.stdout)
            statuses = [r.get("status") for r in parsed["results"]]
            assert statuses == ["verified"], parsed
            assert res.returncode == 0, res.stdout
        finally:
            _session_stop(inst_id)

    def test_named_params_also_verify(self):
        """Contrast case: a fully NAMED prototype still verifies through the same
        path -- the name-insensitive acceptance must not perturb the normal,
        string-matching case. (The rejection of a genuine type/arity/return
        mismatch can't be forced through real BN, which applies valid prototypes
        verbatim, so that is covered by the mocked unit test
        test_prototype_matches_ignoring_param_names.)"""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            fn = self._first_fn(inst_id)
            res = _bn("--instance", inst_id, "proto", "set", fn,
                      f"int32_t {fn}(int64_t argc, char** argv)", "--format", "json")
            parsed = json.loads(res.stdout)
            assert [r.get("status") for r in parsed["results"]] == ["verified"], parsed
        finally:
            _session_stop(inst_id)

    def test_auto_prototype_preview_is_refused(self):
        """#630: a --preview of a proto set on an AUTO function (no user type) is
        REFUSED before any mutation, because BN cannot clear the has_user_type it
        would pin, so the preview could not be cleanly reverted. Proves the honest
        contract on live BN: refuse rather than apply and claim a clean rollback.
        The view is left pristine -- the function stays AUTO."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            fn = self._first_fn(inst_id)  # a fresh-analysis function is AUTO
            res = _bn("--instance", inst_id, "proto", "set", fn,
                      f"void {fn}(int32_t, char**, char**)", "--preview", "--format", "json")
            # A refusal is a BridgeError -> CLI exit 2, specifically (not merely
            # nonzero): the preflight raised before any mutation (#630 round 3).
            assert res.returncode == 2, (res.returncode, res.stdout, res.stderr)
            assert "has_user_type" in (res.stdout + res.stderr), (res.stdout, res.stderr)
            # Pristine, checked against a source that ACTUALLY reflects has_user_type:
            # `function info` never emits the flag, so asserting on its output is
            # vacuous. Instead commit a real prototype set now and read the op's
            # before_has_user_type -- it reports the function's provenance at the
            # moment before this commit. If the refused preview had wrongly pinned
            # has_user_type, before_has_user_type would be true and this fails.
            commit = _bn("--instance", inst_id, "proto", "set", fn,
                         f"void {fn}(int32_t, char**, char**)", "--format", "json")
            commit_parsed = json.loads(commit.stdout)
            proto_result = next(r for r in commit_parsed["results"]
                                if r.get("op") == "set_prototype")
            assert proto_result["before_has_user_type"] is False, commit_parsed
        finally:
            _session_stop(inst_id)


class TestStructFieldTypedef:
    """Regression for #246: field ops on a typedef'd (NamedTypeReference) struct
    must follow the alias to the underlying tag instead of crashing in
    add_member_at_offset. The mocked suite cannot reproduce mutable_copy()
    returning an NTR builder, so this drives the real BN type system end-to-end.
    """

    def _declare_and_set(self, inst_id, decl, struct_name):
        declared = _bn("--instance", inst_id, "types", "declare", decl, "--format", "json")
        assert declared.returncode == 0, declared.stderr
        return _bn("--instance", inst_id, "struct", "field", "set",
                   struct_name, "0x4", "newfield", "uint32_t", "--format", "json")

    def test_set_field_on_named_typedef_struct(self):
        """typedef of a named struct: `typedef struct InnerRec AliasRec;`. The
        report must key on the underlying TAG, not the alias: affected_types names
        the tag (so it carries members and a real layout diff) and agrees with
        results[].struct_name (#246, incl. the reporting-path follow-up)."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            res = self._declare_and_set(
                inst_id,
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
            shown = _bn("--instance", inst_id, "struct", "show", "InnerRec")
            assert "newfield" in shown.stdout, shown.stdout
        finally:
            _session_stop(inst_id)

    def test_rename_field_through_typedef_reports_change(self):
        """Regression for the reporting follow-up: a field rename through a typedef
        must report the real change against the TAG -- before the fix it keyed the
        diff on the members-less alias and falsely said 'No effective change
        detected' even though the op verified (#246)."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            self._declare_and_set(
                inst_id,
                "struct InnerRec { uint32_t x; }; typedef struct InnerRec AliasRec;",
                "AliasRec")
            res = _bn("--instance", inst_id, "struct", "field", "rename",
                      "AliasRec", "newfield", "renamed", "--format", "json")
            assert res.returncode == 0, f"rename failed: {res.stdout}\n{res.stderr}"
            parsed = json.loads(res.stdout)
            assert parsed["results"][0]["status"] == "verified", parsed["results"]
            affected = parsed["affected_types"]
            assert affected and affected[0]["name"] == "InnerRec", affected
            assert affected[0]["changed"] is True, affected
            assert "No effective change" not in (affected[0].get("message") or "")
        finally:
            _session_stop(inst_id)

    def test_set_field_on_anonymous_typedef_struct(self):
        """The idiomatic `typedef struct { ... } AnonRec;` -- body is registered
        under the auto-named tag `_AnonRec`, alias is an NTR to it."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            res = self._declare_and_set(
                inst_id,
                "typedef struct { uint32_t m; } AnonRec;",
                "AnonRec")
            assert res.returncode == 0, f"set crashed: {res.stdout}\n{res.stderr}"
            shown = _bn("--instance", inst_id, "struct", "show", "_AnonRec")
            assert "newfield" in shown.stdout, shown.stdout
        finally:
            _session_stop(inst_id)

    def test_set_field_on_typedef_to_nonstruct_is_clean_error(self):
        """`typedef uint32_t Foo;` resolves to a non-aggregate: a field set must
        fail cleanly (not exit 0, not an internal AttributeError crash)."""
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            res = self._declare_and_set(
                inst_id, "typedef uint32_t NotAStruct;", "NotAStruct")
            assert res.returncode != 0, f"expected a clean failure, got: {res.stdout}"
            assert "AttributeError" not in res.stdout + res.stderr
        finally:
            _session_stop(inst_id)


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

    def test_value_set_resolved_indirect_call_anchors_source(self):
        self._ensure_fixture()
        info = _session_start(str(DISPATCH_BINARY))
        inst_id = info["instance_id"]
        try:
            # arg:h_copy:1 with NO --resolve-map: the source must anchor at the
            # indirect `table[cmd](buf, n)` call because value-set resolves it to
            # {h_copy, h_noop, h_log}, and the attacker length must reach h_copy's
            # copy sink.
            res = _bn("--instance", inst_id, "taint", "forward", "-f", "dispatch",
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
        finally:
            _session_stop(inst_id)


class TestStructFieldDeleteWidth:
    """Regression for #320: deleting the trailing field of a struct must shrink
    the struct width (BN's StructureBuilder.remove() leaves it stale), and a
    --preview of that delete must restore the original width on revert. The
    mocked suite cannot model BN's real width bookkeeping, so this drives it
    end-to-end.
    """

    def _declare(self, inst_id, decl):
        res = _bn("--instance", inst_id, "types", "declare", decl, "--format", "json")
        assert res.returncode == 0, res.stderr
        return res

    def test_delete_trailing_field_shrinks_width(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            self._declare(inst_id, "struct WTd320 { unsigned char pad[24]; };")
            setres = _bn("--instance", inst_id, "struct", "field", "set",
                         "WTd320", "0x18", "extra", "int32_t", "--format", "json")
            assert setres.returncode == 0, setres.stderr
            shown = _bn("--instance", inst_id, "struct", "show", "WTd320")
            assert "0x1c" in shown.stdout, shown.stdout  # width grew to 0x1c

            res = _bn("--instance", inst_id, "struct", "field", "delete",
                      "WTd320", "extra", "--format", "json")
            assert res.returncode == 0, f"delete failed: {res.stdout}\n{res.stderr}"
            parsed = json.loads(res.stdout)
            assert parsed["results"][0]["status"] == "verified", parsed["results"]
            after = _bn("--instance", inst_id, "struct", "show", "WTd320")
            assert "0x18" in after.stdout, after.stdout   # shrank back to 0x18
            assert "0x1c" not in after.stdout, after.stdout
        finally:
            _session_stop(inst_id)

    def test_preview_delete_restores_width(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            self._declare(inst_id, "struct WTp320 { unsigned char pad[24]; };")
            _bn("--instance", inst_id, "struct", "field", "set",
                "WTp320", "0x18", "extra", "int32_t", "--format", "json")
            res = _bn("--instance", inst_id, "struct", "field", "delete",
                      "WTp320", "extra", "--preview", "--format", "json")
            assert res.returncode == 0, f"preview failed: {res.stdout}\n{res.stderr}"
            # after a preview revert, the struct must be unchanged: extra still
            # present and width still 0x1c (preview restored the shrink too).
            after = _bn("--instance", inst_id, "struct", "show", "WTp320")
            assert "extra" in after.stdout, after.stdout
            assert "0x1c" in after.stdout, after.stdout
        finally:
            _session_stop(inst_id)


class TestTypesDeclareBitfield:
    """Regression for #322: BN's headless C parser silently drops bitfield `:N`
    widths and emits an overlapping, oversized layout reported as `verified`. The
    declaration must instead be rejected cleanly, and the corrupt type must NOT
    be registered. Drives the real BN parser end-to-end.
    """

    def test_bitfield_declaration_is_rejected(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            res = _bn("--instance", inst_id, "types", "declare",
                      "struct BF322 { unsigned a:3; unsigned b:5; unsigned c:1; unsigned d:23; };",
                      "--format", "json")
            assert res.returncode != 0, f"expected rejection, got: {res.stdout}"
            parsed = json.loads(res.stdout)
            results = parsed.get("results") or [parsed]
            assert results[0].get("status") == "invalid_request", parsed
            assert "bitfield" in (results[0].get("message") or "").lower(), parsed
            # the corrupt type must not have been registered
            shown = _bn("--instance", inst_id, "struct", "show", "BF322")
            assert "BF322" not in shown.stdout or "size=0x4" not in shown.stdout, shown.stdout
        finally:
            _session_stop(inst_id)

    def test_plain_struct_still_declares(self):
        # The contrast: a bitfield-free struct (incl. a comment containing a
        # colon-number) still declares cleanly -- no false rejection.
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            res = _bn("--instance", inst_id, "types", "declare",
                      "struct OK322 { int a; /* note:32 */ char b; long c; };",
                      "--format", "json")
            assert res.returncode == 0, f"unexpected rejection: {res.stdout}\n{res.stderr}"
            parsed = json.loads(res.stdout)
            assert parsed["results"][0]["status"] == "verified", parsed["results"]
        finally:
            _session_stop(inst_id)


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

    def _a_data_address(self, inst_id) -> str:
        """The start of a non-executable data section -- a mapped address BN did
        not make part of a function."""
        res = _bn("--instance", inst_id, "sections", "--format", "json")
        assert res.returncode == 0, res.stderr
        for sec in self._items(json.loads(res.stdout)):
            if not sec.get("executable") and sec.get("start"):
                return sec["start"]
        raise AssertionError("no non-executable section found")

    def test_linear_disasm_at_non_function_address(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            addr = self._a_data_address(inst_id)
            # plain disasm refuses it, but points at --linear
            plain = _bn("--instance", inst_id, "disasm", addr)
            assert plain.returncode != 0, plain.stdout
            assert "--linear" in (plain.stdout + plain.stderr)
            # --linear disassembles N instructions there
            res = _bn("--instance", inst_id, "disasm", addr, "--linear", "4", "--format", "json")
            assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
            result = self._unwrap(json.loads(res.stdout))
            assert result.get("linear") is True, result
            assert result.get("function") is None, result
            assert 1 <= result.get("instruction_count", 0) <= 4, result
            assert result["instructions"], result
            assert result["instructions"][0]["address"].lower().startswith("0x")
        finally:
            _session_stop(inst_id)

    def test_linear_disasm_from_function_name(self):
        # --linear also accepts a function name, anchoring at its start.
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            listing = _bn("--instance", inst_id, "function", "list", "--format", "json")
            name = self._items(json.loads(listing.stdout))[0]["name"]
            res = _bn("--instance", inst_id, "disasm", name, "--linear", "3", "--format", "json")
            assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
            result = self._unwrap(json.loads(res.stdout))
            assert result.get("linear") is True
            assert result.get("instruction_count", 0) >= 1
        finally:
            _session_stop(inst_id)


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
    def _build_aarch64(tmp_path) -> Path:
        cc = shutil.which("aarch64-linux-gnu-gcc")
        if cc is None:
            pytest.skip("aarch64-linux-gnu-gcc not available")
        src = tmp_path / "probe.c"
        src.write_text("int add(int a, int b){return a + b;}\nint main(){return add(1, 2);}\n")
        out = tmp_path / "probe_aarch64"
        proc = subprocess.run(
            [cc, "-O0", "-no-pie", "-static", str(src), "-o", str(out)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            pytest.skip(f"aarch64 cross-compile failed: {proc.stderr}")
        return out

    @staticmethod
    def _func_start(inst_id: str, name: str) -> int:
        listing = _bn("--instance", inst_id, "function", "list", "--format", "json")
        assert listing.returncode == 0, listing.stderr
        funcs = TestDisasmLinear._items(json.loads(listing.stdout))
        matches = [f for f in funcs if f["name"] == name]
        assert matches, f"{name} not found among {len(funcs)} functions"
        return int(matches[0]["address"], 16)

    def test_aarch64_odd_linear_start_not_thumb_masked(self, tmp_path):
        binary = self._build_aarch64(tmp_path)
        info = _session_start(str(binary))
        inst_id = info["instance_id"]
        try:
            start = self._func_start(inst_id, "add")
            odd = start | 1  # poke bit 0 so a Thumb-masking gate would strip it
            res = _bn("--instance", inst_id, "disasm", hex(odd), "--linear", "2", "--format", "json")
            assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
            result = TestDisasmLinear._unwrap(json.loads(res.stdout))
            # Premise: the live arch name really is "aarch64", not "arm64".
            assert result["decode_arch"] == "aarch64", result
            # bit 0 preserved -- NOT masked back to the even address ...
            assert int(result["address"], 16) == odd, result
            # ... and no Thumb function-pointer-tag normalization was applied.
            assert "Thumb" not in result["note"], result["note"]
        finally:
            _session_stop(inst_id)

    def test_aarch64_rejects_arm_thumb_mode(self, tmp_path):
        # --mode arm|thumb is only meaningful for classic 32-bit ARM/Thumb. On a
        # real aarch64 target it must be rejected with the ACTUAL arch named.
        binary = self._build_aarch64(tmp_path)
        info = _session_start(str(binary))
        inst_id = info["instance_id"]
        try:
            res = _bn("--instance", inst_id, "disasm", "add", "--linear", "2",
                      "--mode", "arm", "--format", "json")
            assert res.returncode != 0, res.stdout
            assert "aarch64" in (res.stdout + res.stderr).lower(), (res.stdout, res.stderr)
        finally:
            _session_stop(inst_id)


class TestFunctionCreatePreviewHonesty:
    """Regression for #304: `function create <addr> --preview` reported `verified`
    while the follow-up live `function create <addr>` reported
    `verification_failed`. The preview's revert used remove_user_function, which
    records a persistent user "no function here" override that poisoned the
    address. The non-poisoning remove_function makes preview and live agree.
    Needs real BN -- only BN's analysis reproduces the suppression behavior.
    """

    def _gap_addresses(self, inst_id):
        """Candidate executable addresses that are NOT function starts: the byte
        just past a function when a gap precedes the next function."""
        listing = _bn("--instance", inst_id, "function", "list", "--format", "json")
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

    def test_preview_then_live_agree(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            chosen = None
            for addr in self._gap_addresses(inst_id)[:20]:
                hexaddr = hex(addr)
                prev = _bn("--instance", inst_id, "function", "create", hexaddr,
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
            live = _bn("--instance", inst_id, "function", "create", chosen, "--format", "json")
            assert live.returncode == 0, f"{chosen}: {live.stdout}\n{live.stderr}"
            parsed = json.loads(live.stdout)
            assert parsed["results"][0]["status"] == "verified", parsed
            assert parsed["committed"] is True, parsed
        finally:
            _session_stop(inst_id)


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
    def _gaps(inst):
        listing = _bn("--instance", inst, "function", "list", "--format", "json")
        items = json.loads(listing.stdout)
        items = items.get("items", items) if isinstance(items, dict) else items
        fns = sorted(((int(f["address"], 16), int(f.get("size") or 0)) for f in items),
                     key=lambda t: t[0])
        return [start + size for (start, size), (nxt, _) in zip(fns, fns[1:])
                if size > 0 and start + size < nxt]

    def test_batch_function_create_preview_then_live_atomic(self, tmp_path):
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
            addr = None
            for cand in self._gaps(inst)[:20]:
                mf = tmp_path / "probe.json"
                mf.write_text(json.dumps({"ops": [{"op": "function_create", "address": hex(cand)}]}))
                prev = _bn("--instance", inst, "batch", "apply", str(mf), "--preview", "--format", "json")
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
            prev = _bn("--instance", inst, "batch", "apply", str(mf), "--preview", "--format", "json")
            assert prev.returncode == 0, f"{prev.stdout}\n{prev.stderr}"
            pj = json.loads(prev.stdout)
            assert [r["status"] for r in pj["results"]] == ["verified", "verified"], pj
            assert pj["committed"] is False
            assert _bn("--instance", inst, "function", "info", addr).returncode != 0  # reverted

            # live: the batch commits atomically -- function AND comment persist.
            live = _bn("--instance", inst, "batch", "apply", str(mf), "--format", "json")
            assert live.returncode == 0, f"{live.stdout}\n{live.stderr}"
            lj = json.loads(live.stdout)
            assert [r["status"] for r in lj["results"]] == ["verified", "verified"], lj
            assert lj["committed"] is True
            assert _bn("--instance", inst, "function", "info", addr).returncode == 0  # now a function
        finally:
            _session_stop(inst)


class TestFunctionCreateSkippedAddress:
    """Regression for #360: function create must succeed on an address
    auto-analysis SKIPPED (a data-table / missed-handler entry). The handler
    uses the forced create_user_function; the advisory add_function declines
    exactly those addresses, so the op used to return verification_failed on its
    own documented use-case. Drives real BN."""

    def test_create_on_auto_skipped_address(self):
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
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
            probe = _bn("--instance", inst, "py", "exec", code)
            line = next((l for l in probe.stdout.splitlines() if l.startswith("ADDR=")), None)
            if line is None:
                pytest.skip("no caller-less auto-skipped function in this fixture")
            addr = line.split("=", 1)[1].strip()

            # create on the skipped address: must verify and commit (#360). With
            # the advisory add_function this returned verification_failed.
            out = _bn("--instance", inst, "function", "create", addr, "--format", "json")
            assert out.returncode == 0, f"{out.stdout}\n{out.stderr}"
            res = json.loads(out.stdout)
            assert res["results"][0]["status"] == "verified", res
            assert res["committed"] is True, res
            assert _bn("--instance", inst, "function", "info", addr).returncode == 0

            # --preview on another skipped address verifies AND reverts cleanly,
            # and a subsequent live create still works (the revert must not poison
            # the address, #304).
            probe2 = _bn("--instance", inst, "py", "exec", code)
            line2 = next((l for l in probe2.stdout.splitlines() if l.startswith("ADDR=")), None)
            if line2 is not None:
                addr2 = line2.split("=", 1)[1].strip()
                prev = _bn("--instance", inst, "function", "create", addr2,
                           "--preview", "--format", "json")
                assert json.loads(prev.stdout)["results"][0]["status"] == "verified"
                assert _bn("--instance", inst, "function", "info", addr2).returncode != 0
                live = _bn("--instance", inst, "function", "create", addr2, "--format", "json")
                assert json.loads(live.stdout)["results"][0]["status"] == "verified"
                assert json.loads(live.stdout)["committed"] is True
        finally:
            _session_stop(inst)


class TestTaintEmptyVerdictHonesty:
    """Regression for #310.1: a genuinely empty forward-taint result (no sink, no
    frontier) must carry the same loud 'NOT an all-clear' caveat the
    partial-coverage paths do -- it's exactly the shape a structurally-invisible
    bug produces, so it must be the most caveated case, not the least."""

    def test_empty_forward_verdict_is_caveated(self):
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
            fns = json.loads(_bn("--instance", inst, "function", "list", "--format", "json").stdout)
            names = [f["name"] for f in (fns.get("items") if isinstance(fns, dict) else fns)]
            saw_empty = False
            for name in names[:30]:
                out = _bn("--instance", inst, "taint", "forward", "-f", name, "--source", "param:0")
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
        finally:
            _session_stop(inst)


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

    def test_under_recovered_callee_arg_emits_frontier(self, tmp_path):
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

        info = _session_start(str(binp))
        inst = info["instance_id"]
        try:
            def _taint_leaves():
                out_file = tmp_path / "taint.json"
                _bn("--instance", inst, "taint", "forward", "-f", "main",
                    "--source", "param:1", "--format", "json", "--out", str(out_file))
                return json.loads(out_file.read_text()).get("leaves", [])

            def _frontiers(leaves):
                return [l for l in leaves if "under-recovered" in str(l.get("note", ""))]

            # Baseline: build_cmd recovered with its arg -> no #381 frontier.
            assert _frontiers(_taint_leaves()) == []

            # Force the recovery miss: build_cmd as 0-arity.
            _bn("--instance", inst, "proto", "set", "build_cmd",
                "void build_cmd(void)", "--format", "json")

            # Now the tainted argv arg into the under-recovered callee must be an
            # honest frontier, not a silent drop.
            frontiers = _frontiers(_taint_leaves())
            assert frontiers, "expected a #381 under-recovered-arg frontier"
            assert frontiers[0].get("kind") == "unmodeled_callee"
            assert frontiers[0].get("callee", {}).get("name") == "build_cmd"
        finally:
            _session_stop(inst)


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

    def test_arg_register_fallback_backward_and_trace(self, tmp_path):
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

        info = _session_start(str(binp))
        inst = info["instance_id"]
        try:
            # Force the recovery miss: memcpy as 1-arity, so arg:memcpy:2 (the length,
            # in r2) is out of range and only the register fallback can seed it.
            _bn("--instance", inst, "proto", "set", "memcpy",
                "void* memcpy(void* dst)", "--format", "json")

            # Backward: the canonical `arg:memcpy:2` length seed resolves to a slice
            # (register-recovered), not a dead-end, and discloses the #433 caveat.
            bw = tmp_path / "bw.json"
            r = _bn("--instance", inst, "taint", "backward", "-f", "do_copy",
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
            r2 = _bn("--instance", inst, "trace", "do_copy", str(call_addr), "--arg", "2",
                     "--format", "json", "--out", str(tr))
            assert r2.returncode == 0, r2.stderr
            tres = json.loads(tr.read_text())
            assert tres.get("step_count", 0) > 0, "trace: expected a #433 register-recovered trace"
            assert tres.get("arg_label", {}).get("register") == "r2"
            assert any("#433" in h for h in tres.get("hints", [])), \
                "trace: expected the #433 register-recovery hint"
        finally:
            _session_stop(inst)


class TestTagRoundtrip:
    """Real-BN round trip for the `bn tag` group: a custom tag type, a
    FUNCTION-scope tag, and an ADDRESS-scope tag. The address-scope readback
    is the key dogfood check here: our code (read_tags._collect_tags /
    _get_tags) assumes real BN's `Function.tags` / `Function.get_tags_at`
    surface address-scope tags the way the mocked fakes in test_tags.py model
    them -- only a real BN Function object can prove that assumption."""

    def _first_function_address(self, inst) -> str:
        out = _bn("--instance", inst, "function", "list", "--limit", "1", "--format", "json")
        assert out.returncode == 0, f"{out.stdout}\n{out.stderr}"
        listing = json.loads(out.stdout)
        items = listing.get("items") if isinstance(listing, dict) else listing
        return items[0]["address"]

    def test_tag_roundtrip_across_scopes(self):
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
            created = _bn("--instance", inst, "tag", "type", "create", "AgentNote",
                          "--icon", "\U0001F916", "--format", "json")
            assert created.returncode == 0, f"{created.stdout}\n{created.stderr}"
            assert json.loads(created.stdout)["results"][0]["status"] in ("verified", "noop")

            fn_addr = self._first_function_address(inst)

            # FUNCTION-scope tag.
            add_fn = _bn("--instance", inst, "tag", "add", "--function", fn_addr,
                        "--type", "AgentNote", "--data", "reviewed by agent",
                        "--format", "json")
            assert add_fn.returncode == 0, f"{add_fn.stdout}\n{add_fn.stderr}"
            assert json.loads(add_fn.stdout)["results"][0]["status"] == "verified"

            # ADDRESS-scope tag at the function's entry address (still an
            # address tag, not a function tag -- distinct scope, distinct id).
            add_addr = _bn("--instance", inst, "tag", "add", fn_addr,
                           "--type", "AgentNote", "--data", "flagged address",
                           "--format", "json")
            assert add_addr.returncode == 0, f"{add_addr.stdout}\n{add_addr.stderr}"
            assert json.loads(add_addr.stdout)["results"][0]["status"] == "verified"

            # `tag list` sweeps Function.get_function_tags() AND Function.tags --
            # both scopes must be found (the address-scope entry is the key
            # real-BN dogfood check, see class docstring).
            listing = json.loads(_bn("--instance", inst, "tag", "list",
                                     "--type", "AgentNote", "--format", "json").stdout)
            items = listing["items"]
            scopes_and_data = {(t["scope"], t["data"]) for t in items}
            assert ("function", "reviewed by agent") in scopes_and_data, items
            assert ("address", "flagged address") in scopes_and_data, items

            # `tag get <addr>` independently surfaces the address-scope tag via
            # Function.get_tags_at -- a second, distinct real-BN code path.
            got = json.loads(_bn("--instance", inst, "tag", "get", fn_addr,
                                 "--format", "json").stdout)
            assert any(t["scope"] == "address" and t["data"] == "flagged address"
                      for t in got["tags"]), got

            # Remove both tags (each `tag remove` targets one scope).
            rm_addr = _bn("--instance", inst, "tag", "remove", fn_addr,
                          "--type", "AgentNote", "--format", "json")
            assert rm_addr.returncode == 0, f"{rm_addr.stdout}\n{rm_addr.stderr}"
            assert json.loads(rm_addr.stdout)["results"][0]["status"] == "verified"

            rm_fn = _bn("--instance", inst, "tag", "remove", "--function", fn_addr,
                       "--type", "AgentNote", "--format", "json")
            assert rm_fn.returncode == 0, f"{rm_fn.stdout}\n{rm_fn.stderr}"
            assert json.loads(rm_fn.stdout)["results"][0]["status"] == "verified"

            # Now that no tags of this type remain, the custom tag type itself
            # can be removed.
            rm_type = _bn("--instance", inst, "tag", "type", "remove", "AgentNote",
                          "--format", "json")
            assert rm_type.returncode == 0, f"{rm_type.stdout}\n{rm_type.stderr}"
            assert json.loads(rm_type.stdout)["results"][0]["status"] == "verified"
        finally:
            _session_stop(inst)

    def test_tag_type_remove_refuses_builtin(self):
        """A built-in tag type (e.g. Bookmarks) must be refused with a clean
        invalid_request, not removed -- mirrors the bitfield-rejection shape in
        TestTypesDeclareBitfield above."""
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
            res = _bn("--instance", inst, "tag", "type", "remove", "Bookmarks",
                      "--format", "json")
            assert res.returncode != 0, f"expected rejection, got: {res.stdout}"
            parsed = json.loads(res.stdout)
            results = parsed.get("results") or [parsed]
            assert results[0].get("status") == "invalid_request", parsed
            assert "built-in" in (results[0].get("message") or "").lower(), parsed
            # Must NOT have been removed -- still present and flagged built-in.
            types = json.loads(_bn("--instance", inst, "tag", "types",
                                   "--format", "json").stdout)["tag_types"]
            bookmarks = next((t for t in types if t["name"] == "Bookmarks"), None)
            assert bookmarks is not None and bookmarks["is_builtin"] is True, types
        finally:
            _session_stop(inst)


class TestFunctionDocRoundtrip:
    """Real-BN round trip for the function-doc surface: `comment --function`
    now targets `fn.comment` (the function's documentation comment shown atop
    the function), not an address comment."""

    def test_function_doc_set_get_delete(self):
        info = _session_start(str(HELLO_BINARY))
        inst = info["instance_id"]
        try:
            listing = json.loads(_bn("--instance", inst, "function", "list",
                                     "--limit", "1", "--format", "json").stdout)
            items = listing.get("items") if isinstance(listing, dict) else listing
            fn_addr = items[0]["address"]

            doc_text = "AgentNote: reviewed and documented by an integration test"
            set_res = _bn("--instance", inst, "comment", "set", "--function", fn_addr,
                          doc_text, "--format", "json")
            assert set_res.returncode == 0, f"{set_res.stdout}\n{set_res.stderr}"
            assert json.loads(set_res.stdout)["results"][0]["status"] == "verified"

            got = json.loads(_bn("--instance", inst, "comment", "get", "--function", fn_addr,
                                 "--format", "json").stdout)
            assert got["function_doc"] == doc_text, got
            assert got["has_function_doc"] is True, got

            del_res = _bn("--instance", inst, "comment", "delete", "--function", fn_addr,
                          "--format", "json")
            assert del_res.returncode == 0, f"{del_res.stdout}\n{del_res.stderr}"
            assert json.loads(del_res.stdout)["results"][0]["status"] == "verified"

            after = json.loads(_bn("--instance", inst, "comment", "get", "--function", fn_addr,
                                   "--format", "json").stdout)
            assert after["function_doc"] == "", after
            assert after["has_function_doc"] is False, after
        finally:
            _session_stop(inst)
