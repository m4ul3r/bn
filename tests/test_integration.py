"""Integration tests for multi-instance bridge sessions.

These tests require Binary Ninja to be importable. They are skipped if
the binaryninja module is not available.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import struct
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import refuse_silent_skip

from bn.paths import instances_dir

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"
ADD_BINARY = FIXTURES_DIR / "add_x86_64"
DISPATCH_BINARY = FIXTURES_DIR / "dispatch_table_x86_64"
PARSER_BINARY = FIXTURES_DIR / "parser_x86_64"
CANARY_BINARY = FIXTURES_DIR / "canary_x86_64"
CLASSES_BINARY = FIXTURES_DIR / "classes_x86_64"
CLASSES_STRIPPED_BINARY = FIXTURES_DIR / "classes_stripped_x86_64"
VCALLS_BINARY = FIXTURES_DIR / "vcalls_x86_64"
STATICS_BINARY = FIXTURES_DIR / "statics_x86_64"

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
# Stop includes the 30s graceful teardown allowance, a 2s post-KILL wait,
# the unsaved-target probe and outer CLI startup/shutdown overhead.
_SESSION_STOP_TIMEOUT = 60.0

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


def _session_stop(instance_id: str, timeout: float = _SESSION_STOP_TIMEOUT) -> None:
    # Teardown, so the discard is deliberate: a test that mutated its private
    # copy and never saved would otherwise be REFUSED here (#925) and leak the
    # bridge until the idle reaper caught it.
    subprocess.run(
        [*_BN_CLI, "session", "stop", instance_id, "--discard-unsaved"],
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

    def test_concurrent_mutations_in_two_sessions_stay_isolated(self, tmp_path):
        """Two sessions mutating AT THE SAME TIME, each into its own copy (#785).

        The concurrent-mutation harness lives in `tests/stress/` and is run by
        hand; inside pytest the only concurrency was single-view (concurrent reads
        during one mutation), so nothing exercised two bridges writing at once.
        Each session renames the same symbol in its OWN copy of the same fixture,
        with both subprocesses released from a barrier so the writes genuinely
        overlap. Both must land -- no cross-instance serialization stall, no
        routing of one session's mutation into the other -- and neither marker may
        be visible in the other session, which is the isolation claim under
        concurrency rather than under a sequential listing.
        """
        copy_a = tmp_path / "twin_a"
        copy_b = tmp_path / "twin_b"
        for copy in (copy_a, copy_b):
            shutil.copyfile(HELLO_BINARY, copy)
            copy.chmod(0o755)

        info_a = _session_start(str(copy_a))
        try:
            info_b = _session_start(str(copy_b))
            try:
                id_a, id_b = info_a["instance_id"], info_b["instance_id"]
                assert id_a != id_b

                listing = json.loads(
                    _bn("--instance", id_a, "function", "list", "--format", "json").stdout)
                items = listing.get("items") if isinstance(listing, dict) else listing
                symbol = items[0]["name"]
                # The copies are byte-identical, so the same symbol exists in both
                # and each session must rename only its own.
                marker_a, marker_b = "TWO_SESSION_ALPHA", "TWO_SESSION_BETA"

                start = threading.Barrier(2)
                renamed: dict[str, subprocess.CompletedProcess[str]] = {}

                def mutate(instance_id: str, marker: str) -> None:
                    start.wait(timeout=30.0)
                    renamed[marker] = _bn(
                        "--instance", instance_id, "rename", symbol, marker,
                        "--format", "json")

                writers = [threading.Thread(target=mutate, args=(id_a, marker_a)),
                           threading.Thread(target=mutate, args=(id_b, marker_b))]
                for writer in writers:
                    writer.start()
                for writer in writers:
                    writer.join(timeout=_SESSION_START_TIMEOUT)
                    assert not writer.is_alive(), "a concurrent mutation never returned"

                for marker, result in renamed.items():
                    assert result.returncode == 0, f"{marker}: {result.stdout}\n{result.stderr}"

                def names_in(instance_id: str, query: str) -> list[str]:
                    found = _bn("--instance", instance_id, "function", "search", query,
                                "--format", "json")
                    assert found.returncode == 0, f"{query}: {found.stdout}\n{found.stderr}"
                    payload = json.loads(found.stdout)
                    rows = payload.get("items") if isinstance(payload, dict) else payload
                    return [row["name"] for row in rows]

                assert names_in(id_a, marker_a) == [marker_a]
                assert names_in(id_a, marker_b) == []
                assert names_in(id_b, marker_b) == [marker_b]
                assert names_in(id_b, marker_a) == []
            finally:
                _session_stop(id_b)
        finally:
            _session_stop(id_a)


class TestSessionStopUnsavedGuard:
    """`session stop` must not discard unsaved work -- least of all because the
    bridge was busy (#925). A private session, because the stop IS the subject.
    """

    def test_a_busy_bridge_refuses_promptly_and_keeps_the_unsaved_rename(self, tmp_path):
        """Both reviewers reproduced the hole on a real bridge: `list_targets`
        waits on the read lock a `py exec` holds, the probe timed out, the
        timeout read as "no targets", and the stop went through with the
        rename lost. Now a bridge that is reached but does not answer refuses
        within the probe's own budget -- not `BN_REQUEST_TIMEOUT` -- and sends
        no shutdown, so the rename is still there once the bridge is free.
        """
        from bn.commands.admin import _UNSAVED_PROBE_TIMEOUT

        copy = tmp_path / "parser"
        shutil.copyfile(HELLO_BINARY, copy)
        copy.chmod(0o755)
        info = _session_start(str(copy))
        inst = info["instance_id"]
        try:
            listing = json.loads(
                _bn("--instance", inst, "function", "list", "--format", "json").stdout)
            items = listing.get("items") if isinstance(listing, dict) else listing
            symbol = items[0]["name"]
            renamed = _bn("--instance", inst, "rename", symbol, "UNSAVED_GUARD_MARKER",
                          "--format", "json")
            assert renamed.returncode == 0, f"{renamed.stdout}\n{renamed.stderr}"
            targets = json.loads(
                _bn("--instance", inst, "target", "list", "--format", "json").stdout)["items"]
            assert [t["unsaved"] for t in targets] == [True]
            selector = targets[0]["selector"]

            # An idle bridge answers: the refusal names the dirty target.
            refused = _bn("session", "stop", inst)
            assert refused.returncode == 2, f"{refused.stdout}\n{refused.stderr}"
            assert f"-t {selector}" in refused.stderr
            assert f"bn -i {inst} -t {selector} save" in refused.stderr

            # A busy bridge does not: `py exec` holds the write lock for longer
            # than the probe budget, so the stop must refuse on "cannot verify"
            # -- promptly, with the bridge's own 600s request default in force.
            hold = 3 * _UNSAVED_PROBE_TIMEOUT
            busy = subprocess.Popen(
                [*_BN_CLI, "--instance", inst, "py", "exec",
                 f"import time; time.sleep({hold})"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_env(),
            )
            try:
                time.sleep(1.0)
                started = time.monotonic()
                stop = _bn("session", "stop", inst, "--format", "json", timeout=hold)
                elapsed = time.monotonic() - started
                assert stop.returncode == 2, f"{stop.stdout}\n{stop.stderr}"
                assert "cannot verify whether instance" in stop.stderr
                assert "no shutdown was sent" in stop.stderr
                assert "--discard-unsaved" in stop.stderr
                assert elapsed < 2 * _UNSAVED_PROBE_TIMEOUT, (
                    f"the probe waited {elapsed:.1f}s behind the write lock")
            finally:
                busy.wait(timeout=hold + 30)

            # The bridge survived the refusal with the rename it was protecting.
            found = _bn("--instance", inst, "function", "search", "UNSAVED_GUARD_MARKER",
                        "--format", "json")
            assert found.returncode == 0, f"{found.stdout}\n{found.stderr}"
            payload = json.loads(found.stdout)
            rows = payload.get("items") if isinstance(payload, dict) else payload
            assert [row["name"] for row in rows] == ["UNSAVED_GUARD_MARKER"]

            # Explicit consent stops it and discloses exactly the dirty target.
            stopped = _bn("session", "stop", inst, "--discard-unsaved", "--format", "json")
            assert stopped.returncode == 0, f"{stopped.stdout}\n{stopped.stderr}"
            result = json.loads(stopped.stdout)
            assert result["stopped"] is True
            assert result["discarded_unsaved"] == [{"selector": selector, "path": str(copy)}]
        finally:
            _session_stop(inst)


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
        assert "-t hello_x86_64" in result.stderr
        assert "note: view_id / target_id are stable across `bn save`" in result.stderr

        # `-t ""` is an explicit-but-empty selector (an unset shell
        # variable): #690 r3 rejects it CLI-side for every command -- it is
        # never pin-filled and never forwarded (the bridge would collapse
        # it to the focused view with no count check).
        result = shared_bn.run("save", "--target", "")
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "--target is empty" in result.stderr


class TestBatchManifestTarget:
    def test_ambient_target_cannot_redirect_a_preview(self, shared_bn, monkeypatch,
                                                      tmp_path):
        """A two-target preview must run against the manifest's chosen view.

        ``add`` exists only in the second synthetic fixture. If BN_TARGET
        overrides the manifest, the batch hits the first view and fails with
        "Symbol not found". Preview makes both probes reversible.
        """
        other = shared_bn.load(HELLO_BINARY)
        chosen = shared_bn.load(ADD_BINARY)
        manifest = tmp_path / "batch.json"
        manifest.write_text(json.dumps({
            "target": chosen,
            "ops": [{"op": "rename_symbol", "identifier": "add",
                     "new_name": "add_reviewed"}],
        }), encoding="utf-8")

        control = shared_bn.run("batch", "apply", str(manifest), "--preview",
                                "--verbose", "--format", "json")
        assert control.returncode == 0, (control.stdout, control.stderr)
        assert json.loads(control.stdout)["success"] is True

        # This deliberately omits -t to exercise the ambient-versus-manifest
        # decision on a dedicated, test-owned bridge.
        monkeypatch.setenv("BN_TARGET", other)
        result = shared_bn.run("batch", "apply", str(manifest), "--preview",
                               "--verbose", "--format", "json")
        assert result.returncode == 0, (result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        assert payload["preview"] is True
        assert payload["results"][0]["status"] == "verified"

    def test_a_name_an_earlier_op_renamed_names_the_rename_975(self, shared_bn, tmp_path):
        """#975: a later op that names a function by the `sub_<addr>` spelling an
        earlier op in the same batch replaced must say so -- not offer
        close-but-unrelated "Did you mean" functions, which reads as a wrong
        address. Stripped, so the name is BN's synthesized one: a named ELF
        symbol would keep resolving after the rename and hide the defect.

        The hint must stay silent when the name still resolves -- a later rename
        reused it for another function -- and the op failed for its own reason;
        otherwise it buries that reason and points at the function the name used
        to denote. A chain through the name (#995) is refused before apply."""
        stripped = tmp_path / "add_stripped"
        build = subprocess.run(["strip", "-o", str(stripped), str(ADD_BINARY)],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        shared_bn.load(stripped, copy=False)
        autos = [f for f in shared_bn.json("function", "list", "--limit", "500")["items"]
                 if f["name"].startswith("sub_")]
        (old, address), (other, _) = [(f["name"], f["address"]) for f in autos[:2]]

        def failed_row(*ops):
            manifest = tmp_path / "batch.json"
            manifest.write_text(json.dumps({"ops": list(ops)}), encoding="utf-8")
            result = shared_bn.run("batch", "apply", str(manifest), "--preview",
                                   "--format", "json")
            assert result.returncode == 3, (result.stdout, result.stderr)
            payload = json.loads(result.stdout)
            assert payload["rolled_back"] is True
            return payload["results"][len(ops) - 1]

        def rename(src, dst):
            return {"op": "rename_symbol", "identifier": src, "new_name": dst}

        failed = failed_row(rename(old, "handle_request"),
                            {"op": "set_comment", "function": old, "comment": "entry"})
        assert failed["status"] == "unsupported"
        assert "Did you mean" not in failed["message"]
        assert (f"operation 0 of this batch renamed {old} (at {address}) to "
                "handle_request") in failed["message"], failed["message"]

        def bad_local(fn):
            return {"op": "local_rename", "function": fn, "variable": "no_such_var",
                    "new_name": "unused"}

        # Reuse: `old` now names the OTHER function, so it resolves.
        failed = failed_row(rename(old, "alpha"), rename(other, old), bad_local(old))
        assert "no_such_var" in failed["message"], failed["message"]
        assert "of this batch renamed" not in failed["message"], failed["message"]

        # #995: renaming one function twice fails at the second rename, by name
        # or by address, naming both operations; the batch is rolled back.
        for second in (rename("foo", "bar"), rename(address, "bar")):
            failed = failed_row(rename(old, "foo"), second)
            assert failed["status"] == "invalid_request", failed
            assert f"renames the symbol at {address} again" in failed["message"], failed
            assert (f'Drop operation 0 and make operation 1 rename the symbol by its '
                    f'address: {{"op": "rename_symbol", "identifier": "{address}", '
                    f'"new_name": "bar"}}') in failed["message"], failed

    def test_rename_chains_are_judged_by_where_the_rename_lands_995(self, shared_bn, tmp_path):
        """#995 review rounds 1-2, on BN's own resolver: a chain is refused only
        when a later rename LANDS on a symbol an earlier one renamed -- through a
        case variant, an address, a data rename at the function's address, or past
        an unrelated rename to the same name -- and batches whose renames land on
        different symbols (a case-folded function beside a data symbol, a data
        symbol holding a name a function rename took) apply."""
        shared_bn.load(ADD_BINARY)
        address = next(f["address"] for f in
                       shared_bn.json("function", "list", "--limit", "500")["items"]
                       if f["name"] == "add")

        def preview(*ops):
            manifest = tmp_path / "chain.json"
            manifest.write_text(json.dumps({"ops": list(ops)}), encoding="utf-8")
            result = shared_bn.run("batch", "apply", str(manifest), "--preview",
                                   "--format", "json")
            return result.returncode, json.loads(result.stdout)

        def rename(src, dst, kind="auto"):
            return {"op": "rename_symbol", "identifier": src, "new_name": dst, "kind": kind}

        chains = [
            ([rename("add", "Q1"), rename("q1", "zz")], 1),
            ([rename(address, "alias", "data"), rename("alias", "q2", "function")], 1),
            ([rename("add", "q1", "function"), rename(address, "q2", "data")], 1),
            ([rename("add", "both", "function"), rename("__dso_handle", "both", "data"),
              rename("both", "lastadd", "function")], 2),
        ]
        for ops, failed_index in chains:
            code, payload = preview(*ops)
            assert code == 3, (ops, payload)
            row = payload["results"][failed_index]
            assert row["status"] == "invalid_request", (ops, row)
            assert f"renames the symbol at {address} again" in row["message"], (ops, row)
            assert payload["rolled_back"] is True, (ops, payload)

        for ops in ([rename("__dso_handle", "ADD", "data"), rename("ADD", "newadd")],
                    [rename("add", "__dso_handle", "function"),
                     rename("__dso_handle", "marker", "data")]):
            code, payload = preview(*ops)
            assert code == 0, (ops, payload)
            assert [r["status"] for r in payload["results"]] == ["verified", "verified"], payload


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


class TestUserPrototypePreview:
    """#803: a changed USER prototype previews without losing its baseline."""

    def _state(self, shared_bn, start: int) -> dict:
        return json.loads(_py_line(shared_bn, (
            "import json\n"
            f"f = bv.get_function_at({start})\n"
            "print('STATE=' + json.dumps({'name': f.name, 'prototype': str(f.type), "
            "'has_user_type': f.has_user_type}))\n"
        ), "STATE"))

    def test_user_prototype_preview_restores_exact_prior_type_and_provenance(self, shared_bn):
        shared_bn.load(ADD_BINARY)
        start, _ = _function_start_and_size(shared_bn, "add")
        baseline = shared_bn.json(
            "proto", "set", hex(start), "int32_t add(int32_t left, int32_t right)",
        )
        assert baseline["success"] is True
        assert baseline["committed"] is True
        before = self._state(shared_bn, start)
        assert before["has_user_type"] is True
        changed = "uint64_t add(uint64_t wide, uint32_t count, void* context)"
        preview = shared_bn.json("proto", "set", hex(start), changed, "--preview")
        assert preview["success"] is True
        assert preview["preview"] is True
        assert preview["committed"] is False
        assert preview["rolled_back"] is True
        row, = preview["results"]
        assert row["status"] == "verified"
        assert row["before_has_user_type"] is True
        assert row["before_prototype"] == before["prototype"]
        assert row["expected_prototype"] != before["prototype"]
        assert row["observed"]["prototype"] != before["prototype"]
        after = self._state(shared_bn, start)
        assert after["has_user_type"] is True
        assert after == before  # Exact prototype (including names) and provenance.


class TestLocalMutationProvenance:
    """#801: preview and failed batches restore AUTO locals, not USER copies."""

    def _state(self, shared_bn, start: int, identifier: int) -> dict:
        return json.loads(_py_line(shared_bn, (
            "import json\n"
            f"f = bv.get_function_at({start})\n"
            f"v = next(v for v in f.vars if v.identifier == {identifier})\n"
            "print('STATE=' + json.dumps({"
            "'name': v.name, 'type': str(v.type), "
            "'user_defined': f.is_var_user_defined(v), 'comment': f.comment}))\n"
        ), "STATE"))

    def _baseline(self, shared_bn) -> tuple[int, dict, dict]:
        shared_bn.load(ADD_BINARY)
        start, _ = _function_start_and_size(shared_bn, "mul")
        locals_ = shared_bn.json("local", "list", hex(start))["items"]
        # The unoptimised synthetic loop has integer stack locals. Select by
        # structure, not a compiler/BN-version-specific auto variable name.
        local = next(v for v in locals_ if not v["is_parameter"]
                     and v["source_type"] == "StackVariableSourceType"
                     and v["type"] == "int32_t")
        assert local["identifier"] is not None
        before = self._state(shared_bn, start, local["identifier"])
        assert before["user_defined"] is False
        assert (before["name"], before["type"]) == (local["name"], local["type"])
        return start, local, before

    def test_auto_local_rename_preview_restores_name_type_and_provenance(self, shared_bn):
        start, local, before = self._baseline(shared_bn)
        new_name = "preview_loop_count"
        assert before["name"] != new_name
        preview = shared_bn.json(
            "local", "rename", hex(start), local["local_id"], new_name, "--preview",
        )
        assert preview["success"] is True
        assert preview["preview"] is True
        assert preview["committed"] is False
        assert preview["rolled_back"] is True
        row, = preview["results"]
        assert row["status"] == "verified"  # A real rename, never a noop.
        assert row["before_name"] == before["name"]
        assert row["observed"]["variable"] == new_name
        after = self._state(shared_bn, start, local["identifier"])
        assert after["user_defined"] is False
        assert after == before

    def test_auto_local_mixed_batch_failure_restores_name_type_and_provenance(self, shared_bn, tmp_path):
        start, local, before = self._baseline(shared_bn)
        new_name = "rollback_loop_count"
        assert before["name"] != new_name
        comment = "temporary loop audit"
        assert before["comment"] != comment
        unmapped = _py_line(shared_bn, (
            "assert not bv.is_valid_offset(bv.end)\n"
            "print('UNMAPPED=' + hex(bv.end))\n"
        ), "UNMAPPED")
        manifest = tmp_path / "local-rollback.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "local_rename", "function": hex(start),
             "variable": local["local_id"], "new_name": new_name},
            {"op": "set_comment", "function": hex(start), "comment": comment},
            # Valid request shape, but fails during apply AFTER the two writes.
            {"op": "set_comment", "address": unmapped, "comment": "cannot land"},
        ]}), encoding="utf-8")
        result = shared_bn.run("batch", "apply", str(manifest), "--format", "json")
        assert result.returncode == 3, (result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert payload["success"] is False
        assert payload["preview"] is False
        assert payload["committed"] is False
        assert payload["rolled_back"] is True
        assert [r["status"] for r in payload["results"]] == [
            "reverted", "reverted", "invalid_request",
        ], payload
        renamed = payload["results"][0]
        assert renamed["before_name"] == before["name"]
        assert renamed["new_name"] == new_name
        after = self._state(shared_bn, start, local["identifier"])
        assert after["user_defined"] is False
        assert after == before  # Name, exact type, provenance AND sibling comment.


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


class TestTypeStringsAndInheritance:
    """#961 / #962 / #964 / #965 against BN's real type system: member objects
    that are fresh on every access, a `::`-qualified type registered under its
    component list, a parser that drops C++ base clauses, and the per-type
    declare summary. None of these can be modelled faithfully by the fakes."""

    def test_retype_field_under_its_own_name(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct Probe961 { void* vtable; int field; };")
        res = shared_bn.run("struct", "field", "set", "Probe961", "0", "vtable", "void**",
                            "--format", "json")
        assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
        assert json.loads(res.stdout)["results"][0]["status"] == "verified"
        assert "void** vtable" in shared_bn.run("struct", "show", "Probe961").stdout

    def test_tagged_and_backticked_qualified_field_types_resolve(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct Probe962 { void* vtable; int field; };")
        # Registered the way a recovered C++ type is: under its component list.
        shared_bn.json("py", "exec", "--code",
                       "import binaryninja as bn\n"
                       "s = bn.StructureBuilder.create()\n"
                       "s.append(bn.Type.pointer(bv.arch, bn.Type.void()), 'slot0')\n"
                       "bv.define_user_type(bn.QualifiedName(['demo', 'Widget', 'VTable']),"
                       " bn.Type.structure_type(s))\n")
        for spelling in ("struct demo::Widget::VTable*", "struct `demo::Widget::VTable`*",
                         "demo::Widget::VTable*"):
            res = shared_bn.run("struct", "field", "set", "Probe962", "0", "vtable", spelling,
                                "--format", "json")
            assert res.returncode == 0, f"{spelling}: {res.stdout}\n{res.stderr}"
        assert "struct demo::Widget::VTable* vtable" in shared_bn.run(
            "struct", "show", "Probe962").stdout
        # A tag keyword that contradicts the type's kind fails, naming the spelling.
        bad = shared_bn.run("struct", "field", "set", "Probe962", "0", "vtable",
                            "union demo::Widget::VTable*", "--format", "json")
        assert bad.returncode == 3, bad.stdout
        assert "struct `demo::Widget::VTable`" in bad.stdout, bad.stdout
        # An unquoted qualified name in a declaration body gets its spelling.
        res = shared_bn.run("types", "declare", "struct Holder962 { demo::Widget::VTable* v; };",
                            "--format", "json")
        assert res.returncode == 3, res.stdout
        assert "struct `demo::Widget::VTable`" in res.stdout, res.stdout

    # The base shapes from the third #989 review's matrix (every shape but the
    # bit-field ones, which the bitfield guard refuses outright), each as the
    # `types declare` steps it was declared in. Bases from the same declaration
    # and from earlier ones, empty, virtual, `final`, `alignas`/`__attribute__`
    # heads, private/protected members, multiple inheritance.
    _BASE_SHAPES = [
        ['struct e02_A { long long a; long long b; };', 'struct e02_D : e02_A { char e; };'],
        ['struct e03_A { long long a; char c; };', 'struct e03_D : e03_A { char e; };'],
        ['struct e04_E { };', 'struct e04_D : e04_E { int x; };'],
        ['struct e05_A { long long a; }; struct e05_B { long long b; };', 'struct e05_D : e05_A, e05_B { char e; };'],
        ['struct e05b_A { long long a; };', 'struct e05b_B { long long b; }; struct e05b_D : e05b_A, e05b_B { char e; };'],
        ['struct e17_A { long long a; }; struct e17_B : e17_A { long long b; };', 'struct e17_C : e17_B { long long c; };', 'struct e17_D : e17_C { char e; };'],
        ['struct alignas(16) e18_A { long long a; };', 'struct e18_D : e18_A { char e; };'],
        ['struct alignas(16) e18b_A { long long a; long long b; };', 'struct e18b_D : e18b_A { char e; };'],
        ['struct e19_A { long long a; long long b; };', 'struct alignas(32) e19_D : e19_A { char e; };'],
        ['struct e20_A { long long a; long long b; };', 'struct __attribute__((packed)) e20_D : e20_A { char e; int f; };'],
        ['struct e22_B { int b; };', 'struct e22_D : virtual e22_B { int x; };'],
        ['class e25_K { long long a; long long b; };', 'struct e25_D : e25_K { char e; };'],
        ['class e37_Z { long long a; char c; };', 'struct e37_A : e37_Z { char d[8]; };', 'struct e37_D : e37_A { char e; };'],
        ['class e43r_Z { long long a; char c; };', 'struct alignas(8) e43r_A : e43r_Z { char d[8]; };', 'struct e43r_D : e43r_A { char e; };'],
        ['struct e44_Z { long long z; }; struct e44_A : e44_Z { char c; }; struct e44_B : e44_A { char d; };', 'struct e44_D : e44_B { char e; };'],
        ['struct e45_Z { long long z; }; struct e45_A : e45_Z { char c; }; struct e45_B : e45_A { };', 'struct e45_D : e45_B { char e; };'],
        ['struct r2fd_F1 {}; struct r2fd_F2 {};', 'struct r2fd_FD2 : r2fd_F1, r2fd_F2 { int x; };'],
        ['struct r2fdx_F1 {}; struct r2fdx_F2 {};', 'struct r2fdx_FD2 : r2fdx_F1, r2fdx_F2 { int x; };'],
        ['class r2ke_KP { long long a; char c; };', 'struct r2ke_KX { char x; }; struct r2ke_KE : r2ke_KP, r2ke_KX { char e; };'],
        ['class r2kex_KP { long long a; char c; };', 'struct r2kex_KX { char x; }; struct r2kex_KE : r2kex_KP, r2kex_KX { char e; };'],
        ['struct r2oe_OA { long long a; char c; }; struct r2oe_OB : r2oe_OA { char d; }; struct r2oe_OM { r2oe_OB m; char z; };', 'struct r2oe_OX { char x; }; struct r2oe_OE : r2oe_OM, r2oe_OX { char e; };'],
        ['struct r2oex_OA { long long a; char c; }; struct r2oex_OB : r2oex_OA { char d; }; struct r2oex_OM { r2oex_OB m; char z; };', 'struct r2oex_OX { char x; }; struct r2oex_OE : r2oex_OM, r2oex_OX { char e; };'],
        ['struct r2qc_MA { long long a; char c; }; struct r2qc_MB : r2qc_MA { char d; };', 'struct r2qc_QC : r2qc_MB { char e; };'],
        ['struct r2qcx_MA { long long a; char c; }; struct r2qcx_MB : r2qcx_MA { char d; };', 'struct r2qcx_QC : r2qcx_MB { char e; };'],
        ['struct r2t5_A5 { long long a; char c; }; struct r2t5_B5 : r2t5_A5 { char d; };', 'struct r2t5_X5 { char x; }; struct r2t5_Y5 { short y; }; struct r2t5_T5 : r2t5_X5, r2t5_Y5, r2t5_B5 { char e; };'],
        ['struct r2t5x_A5 { long long a; char c; }; struct r2t5x_B5 : r2t5x_A5 { char d; };', 'struct r2t5x_X5 { char x; }; struct r2t5x_Y5 { short y; }; struct r2t5x_T5 : r2t5x_X5, r2t5x_Y5, r2t5x_B5 { char e; };'],
        ['struct s01_A { int a; }; struct s01_D : s01_A { char e; };'],
        ['struct s06_A { long long a; char c; }; struct s06_B : s06_A { char d; }; struct s06_X { char x; }; struct s06_D : s06_B, s06_X { char e; };'],
        ['struct s07_E { }; struct s07_D : s07_E { int x; };'],
        ['struct s08_N { int a; }; struct s08_E { }; struct s08_D : s08_N, s08_E { int x; };'],
        ['struct s08b_E { }; struct s08b_D : s08b_E { };'],
        ['class s09_K { long long a; char c; }; struct s09_D : s09_K { char e; };'],
        ['struct s10_K { protected: long long a; char c; }; struct s10_D : s10_K { char e; };'],
        ['struct s10b_K { long long a; private: char c; }; struct s10b_D : s10b_K { char e; };'],
        ['struct s11_A { long long a; char c = 1; }; struct s11_D : s11_A { char e; };'],
        ['struct s12_A { long long& r; char c; }; struct s12_D : s12_A { char e; };'],
        ['struct s13_A { long long a; const char c; }; struct s13_D : s13_A { char e; };'],
        ['struct s18c_A { char c; alignas(16) char d; }; struct s18c_D : s18c_A { char e; };'],
        ['struct s21_B { int b; }; struct s21_D : virtual s21_B { int x; };'],
        ['struct s23_A { long long a; long long b; }; struct s23_D : s23_A { virtual void f(); int y; };'],
        ['struct s24_A { long long a; long long b; }; struct s24_D : s24_A { int get(); int y; };'],
        ['struct s35_E { }; struct s35_F : s35_E { int x; }; struct s35_D : s35_E, s35_F { int y; };'],
        ['struct s35r_E { }; struct s35r_F : s35r_E { int x; }; struct s35r_D : s35r_E, s35r_F { int y; };'],
        ['struct s38_A { long long a; char c; }; class s38_B : public s38_A { char d; }; struct s38_D : s38_B { char e; };'],
        ['struct s38r_A { long long a; char c; }; class s38r_B : public s38r_A { char d; }; struct s38r_D : s38r_B { char e; };'],
        ['struct s39_A { int a; }; struct s39_B { int b; }; struct s39_D : public s39_A, private s39_B { int x; };'],
        ['struct s39r_A { int a; }; struct s39r_B { int b; }; struct s39r_D : public s39r_A, private s39r_B { int x; };'],
        ['struct s40_A { long long a; char c; }; struct s40_B : s40_A { }; struct s40_D : s40_B { char e; };'],
        ['struct s41_A { char c; }; struct s41_B { long double ld; }; struct s41_D : s41_A, s41_B { char e; };'],
        ['struct s42_A { int a; }; struct s42_D final : s42_A { char e; };'],
        ['struct s46_Z { long long z; }; struct s46_A : s46_Z { char c; }; struct s46_B : s46_A { char d; }; struct s46_D : s46_B { char e; };'],
        ['struct t00_A { long long a; long long b; };', 'struct t00_D : t00_A { char e; };'],
    ]

    _DERIVED_HEAD_RE = re.compile(
        r"\b(?:struct|class)\s+(?:alignas\(\d+\)\s*|__attribute__\(\(\w+\)\)\s*)*"
        r"(\w+)(?:\s+final)?\s*:(?!:)")

    def test_declared_bases_are_never_recorded_and_always_listed(self, shared_bn):
        """#965 is split out of #989: three reviews found base layouts bn
        reconstructed wrong. Until it lands, bn records NO C++ base -- and must
        not drop one silently either. For every shape of the third review's
        matrix: no struct carries `bases`, and every struct with a base clause
        is named in that step's `unattached_bases` with the #965 reason; the
        derived struct itself is still declared (counted per type, #964)."""
        shared_bn.load(HELLO_BINARY)
        for steps in self._BASE_SHAPES:
            for step in steps:
                expected = set(self._DERIVED_HEAD_RE.findall(step))
                summary = shared_bn.json("types", "declare", step, "--summary")
                declared = summary["declared_types"]
                unattached = declared.get("unattached_bases") or {}
                assert set(unattached) == expected, (step, unattached)
                assert all("see #965" in why for why in unattached.values()), unattached
                for name in expected:
                    assert name in declared["names"], (step, declared)
                    assert "bases" not in shared_bn.json("types", "show", name), (step, name)

    def test_struct_show_renders_bases_binary_ninja_records(self, shared_bn):
        """`struct show` still renders the bases a type ALREADY carries in BN
        (debug info, a plugin, the API) -- only bn's own reconstruction is gone."""
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct Base965 { void* vtable; int x; };")
        shared_bn.json("py", "exec", "--code",
                       "import binaryninja as bn\n"
                       "s = bn.StructureBuilder.create(width=0x18)\n"
                       "s.insert(0x10, bn.Type.int(4), 'y')\n"
                       "s.base_structures = [bn.BaseStructure(bn.NamedTypeReferenceType.create("
                       "bn.NamedTypeReferenceClass.StructNamedTypeClass, None, 'Base965'), 0, 0x10)]\n"
                       "bv.define_user_type('Derived965', bn.Type.structure_type(s))\n")
        shown = shared_bn.run("struct", "show", "Derived965").stdout
        assert shown.splitlines() == [
            "struct Derived965 : Base965 // size=0x18",
            "0x0000: void* vtable  [from Base965]",
            "0x0008: int32_t x  [from Base965]",
            "0x0010: int32_t y",
        ], shown
        card = shared_bn.json("struct", "show", "Derived965")
        assert card["bases"] == [{"name": "Base965", "offset": "0x0", "width": "0x10"}], card
        assert [(m["offset"], m["name"]) for m in card["inherited_members"]] == [
            ("0x0", "vtable"), ("0x8", "x")], card

    def test_reference_only_declarations(self, shared_bn):
        """#989 review: a forward declaration of an existing type is a no-op (it
        used to apply the clobbering stub, then briefly exited 3); a misspelled
        mention is created but disclosed as `created_by_reference`."""
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct Fwd989 { long a; };")
        res = shared_bn.run("types", "declare", "struct Fwd989;", "--format", "json")
        assert res.returncode == 0, res.stdout
        assert json.loads(res.stdout)["results"][0]["status"] == "noop"
        assert "int64_t a" in shared_bn.run("struct", "show", "Fwd989").stdout
        summary = shared_bn.json("types", "declare",
                                 "struct Use989 { struct Fwd989* f; struct Fwdd989* typo; };", "--summary")
        assert summary["declared_types"]["referenced_existing"] == ["Fwd989"], summary
        assert summary["declared_types"]["created_by_reference"] == ["Fwdd989"], summary
        # The bare-mention stub is not counted as declared.
        assert summary["declared_types"]["names"] == ["Use989"], summary
        # A type only a loaded type library defines is not shadowed by an empty
        # user stub (libc's `struct timespec` on this dynamically linked fixture);
        # the library's own type is imported instead, with its dependencies, and
        # named (#993 item 2).
        lib = shared_bn.json("types", "declare", "struct Lib989 { struct timespec* ts; };", "--summary")
        # Imported, so not also listed as an existing type left unchanged.
        assert "timespec" not in lib["declared_types"].get("referenced_existing", []), lib
        assert "timespec" in lib["declared_types"]["imported_from_library"], lib
        assert shared_bn.json("types", "show", "timespec")["layout"].startswith("struct timespec // size=0x10")
        # A header of only forward declarations of library types is a noop.
        fwd = shared_bn.run("types", "declare", "struct timeval;", "--format", "json")
        assert fwd.returncode == 0, fwd.stdout
        assert json.loads(fwd.stdout)["results"][0]["status"] == "noop"
        assert shared_bn.run("types", "show", "timeval").returncode == 2

    def test_function_pointer_typedef_references_existing_struct(self, shared_bn):
        """#993 item 1: a struct mentioned only in a function-pointer parameter
        list came back from BN's parser as a named `void`; defining it redefined
        the existing struct as `void` and verification rolled the typedef back."""
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct Node993 { int a; };")
        res = shared_bn.run("types", "declare", "typedef void (*visit993_cb)(struct Node993*);",
                            "--format", "json", "--summary")
        assert res.returncode == 0, res.stdout + res.stderr
        summary = json.loads(res.stdout)
        assert summary["declared_types"]["names"] == ["visit993_cb"], summary
        assert summary["declared_types"]["referenced_existing"] == ["Node993"], summary
        assert "int32_t a" in shared_bn.run("struct", "show", "Node993").stdout
        assert shared_bn.json("types", "show", "visit993_cb")["decl"] == "void (*)(struct Node993*)"
        # A NEW name mentioned there becomes an empty struct, not a `void` typedef.
        fresh = shared_bn.json("types", "declare", "typedef void (*leaf993_cb)(struct Leaf993*);", "--summary")
        assert fresh["declared_types"]["created_by_reference"] == ["Leaf993"], fresh
        assert shared_bn.json("types", "show", "Leaf993")["layout"] == "struct Leaf993 // size=0x0"

    def test_noreturn_prototype_sets_can_return_and_previews_restore_it_966(self, shared_bn):
        """#966 (D13): `can_return` is set through `proto set ... __noreturn`
        and verified by BN's effective `can_return` after reanalysis. A preview
        and a live clear both work on a function with a user prototype, and
        the preview restores the previous value."""
        shared_bn.load(HELLO_BINARY)
        probe = "print(bool(bv.get_functions_by_name('greet')[0].can_return))"

        def can_return():
            return shared_bn.run("py", "exec", "--code", probe).stdout.strip().splitlines()[-1]

        assert can_return() == "True"
        marked = shared_bn.json("proto", "set", "greet", "int greet(char const* who) __noreturn")
        row = marked["results"][0]
        assert row["status"] == "verified", row
        assert row["observed"]["can_return"] is False, row
        assert can_return() == "False"
        preview = shared_bn.json("proto", "set", "greet", "int greet(char const* who)", "--preview")
        assert preview["results"][0]["status"] == "verified", preview
        assert preview["results"][0]["observed"]["can_return"] is True, preview
        assert can_return() == "False"
        cleared = shared_bn.json("proto", "set", "greet", "int greet(char const* who)")
        assert cleared["results"][0]["observed"]["can_return"] is True, cleared
        assert can_return() == "True"

    def test_preview_needing_a_library_type_is_refused_and_imports_nothing(self, tmp_path, shared_bn):
        """#993 item 2 / #1012 item 1: parsing a declaration that uses a
        library type makes BN import it, and undo cannot revert that. A preview
        parses with BN's own `import_dependencies=False`, so BN refuses the
        parse instead of importing; the preview is refused naming the library
        type, and the view's type list is untouched -- a single declare, and a
        batch whose later op uses the type. Once the type is in the view, the
        same batch previews."""
        shared_bn.load(HELLO_BINARY)
        probe = "print(sorted(str(n) for n in bv.type_names))"
        before = shared_bn.run("py", "exec", "--code", probe).stdout
        refused = shared_bn.run("types", "declare", "struct Stamp993 { struct timespec ts; int y; };", "--preview")
        assert refused.returncode == 3, refused.stdout + refused.stderr
        # By value: BN refuses the parse itself.
        assert "not previewed: Binary Ninja refused to parse this without importing" in refused.stdout
        assert "timespec (libc" in refused.stdout
        # Only pointed at: the parse succeeds without importing, and the
        # declaration is refused one step later, naming the parsed references
        # (#1012 review r1: this path still printed the old wording).
        pointer = shared_bn.run("types", "declare", "struct Ptr993 { struct tms* p; };", "--preview")
        assert pointer.returncode == 3, pointer.stdout + pointer.stderr
        assert ("not previewed: the declaration's types refer to type-library type(s) the view "
                "does not have yet. The live declare imports them with the library types they "
                "depend on: tms") in pointer.stdout, pointer.stdout
        assert "Nothing was imported" in pointer.stdout, pointer.stdout
        manifest = tmp_path / "batch.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "types_declare", "declaration": "struct Px993 { struct timeval tv; int64_t pad; };"},
            {"op": "struct_field_set", "struct_name": "Px993", "offset": "0x18", "field_name": "t2",
             "field_type": "struct timeval"}]}))
        batch = shared_bn.run("batch", "apply", str(manifest), "--preview")
        assert batch.returncode == 3 and "timeval" in batch.stderr + batch.stdout, batch.stderr
        assert shared_bn.run("py", "exec", "--code", probe).stdout == before
        # A platform-only declaration still previews.
        plain = shared_bn.json("types", "declare", "struct Plain993 { uint32_t a; size_t n; };",
                               "--preview", "--summary")
        assert plain["verified_count"] == 1, plain

        live = shared_bn.json("types", "declare", "struct Stamp993 { struct timespec ts; struct timeval tv; };",
                              "--summary")
        assert {"timespec", "timeval", "__time_t"} <= set(live["declared_types"]["imported_from_library"]), live
        assert "__time_t tv_sec" in shared_bn.run("types", "show", "timespec").stdout
        again = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json", "--summary")
        assert again.returncode == 0, again.stdout + again.stderr
        assert json.loads(again.stdout)["verified_count"] == 2, again.stdout

    def test_untagged_and_macro_hidden_library_uses_preview_without_importing(self, shared_bn):
        """#1012 item 1: untagged library names and a macro hiding the tag
        used to import during the preview and then fail with the imports kept.
        The preview parse now imports nothing: it is refused (exit 3) naming
        the library type, and no type id changes. A user typedef standing in
        for a library name (`sigset_t`) is the user's own type, so that
        declaration previews, again without a new id."""
        shared_bn.load(HELLO_BINARY)
        ids = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", ids).stdout
        shapes = [
            ("struct U1_993 { rusage r; };", "rusage"),
            ("struct U2_993 { timespec* p; };", "timespec"),
            ("struct U3_993 { pthread_mutex_t m; };", "pthread_mutex_t"),
            ("struct U5_993 { pid_t p; FILE* f; };", "pid_t, FILE"),
            ("#define TAGGED(x) struct x\nstruct U4_993 { TAGGED(tms) t; };", "tms"),
        ]
        for declaration, library_types in shapes:
            res = shared_bn.run("types", "declare", declaration, "--preview", "--format", "json", "--summary")
            assert res.returncode == 3, (declaration, res.stdout, res.stderr)
            summary = json.loads(res.stdout)
            assert summary["ok"] is False and summary["verified_count"] == 0, summary
            assert "not previewed: Binary Ninja refused to parse this without importing" in summary["first_error"], summary
            assert f"{library_types} (libc" in summary["first_error"], summary
            assert "Nothing was imported" in summary["first_error"], summary
            assert "kept_library_imports" not in summary, summary
        local = shared_bn.json("types", "declare", "typedef int sigset_t; struct Local993 { sigset_t s; };",
                               "--preview", "--summary")
        assert local["ok"] is True and local["verified_count"] == 1, local
        assert shared_bn.run("py", "exec", "--code", ids).stdout == before

    def test_preview_refusal_lists_name_matches_not_required_types_1012(self, shared_bn):
        """#1012 review r1: `int FILE;` is an ordinary member, yet the refusal
        said the declaration `needs pid_t, FILE`. BN's refusal names no type, so
        the preview lists by-name matches as matches and claims no dependency;
        the live run imports only what BN actually needed."""
        shared_bn.load(HELLO_BINARY)
        declaration = "struct MixedHint { pid_t p; int FILE; };"
        res = shared_bn.run("types", "declare", declaration, "--preview", "--format", "json")
        assert res.returncode == 3, res.stdout + res.stderr
        row = json.loads(res.stdout)["results"][0]
        assert "needs" not in row["message"], row
        assert "pid_t, FILE (libc" in row["message"] and "a by-name match" in row["message"], row
        assert row["observed"]["library_name_matches"] == ["pid_t", "FILE"], row
        assert "library_types" not in row["observed"], row
        live = shared_bn.json("types", "declare", declaration, "--summary")
        assert sorted(live["declared_types"]["imported_from_library"]) == ["__pid_t", "pid_t"], live

    def test_type_string_previews_import_nothing_1012(self, tmp_path, shared_bn):
        """#1012 item 2: `struct field set`, `proto set`, `local retype` and
        `data retype` previews imported the library types their type string
        named. They now parse without importing: a preview that needs one is
        refused naming it, and a prototype the preview leaves out anyway (no
        user prototype yet) is `not_previewed` without importing, so its
        sibling still previews at exit 0."""
        shared_bn.load(HELLO_BINARY)
        ids = "print(sorted(bv.type_container.type_ids))"
        shared_bn.json("types", "declare", "struct Ex1012 { int a; int b; int c; int d; int e; };")
        shared_bn.json("proto", "set", "main", "int main(int argc, char** argv)")
        before = shared_bn.run("py", "exec", "--code", ids).stdout
        local_id = next(row["local_id"] for row in shared_bn.json("local", "list", "main")["items"]
                        if row["name"] == "argc")
        data = shared_bn.run("py", "exec", "--code",
                             "print(hex(bv.sections['.data'].start))").stdout.strip().splitlines()[-1]
        main = shared_bn.run("py", "exec", "--code",
                             "print(hex(bv.get_functions_by_name('main')[0].start))").stdout.strip().splitlines()[-1]
        commands = [
            (("struct", "field", "set", "Ex1012", "0x10", "t", "struct tm"), "tm"),
            (("proto", "set", "main", "int main(pid_t argc, char** argv)"), "pid_t"),
            (("local", "retype", "main", local_id, "pid_t"), "pid_t"),
            (("data", "retype", data, "pid_t"), "pid_t"),
        ]
        for args, library_type in commands:
            res = shared_bn.run(*args, "--preview", "--format", "json", "--summary")
            assert res.returncode == 3, (args, res.stdout, res.stderr)
            summary = json.loads(res.stdout)
            assert "not previewed: Binary Ninja refused to parse this without importing" in summary["first_error"], summary
            assert f"{library_type} (libc" in summary["first_error"], summary
            assert "kept_library_imports" not in summary, summary
        manifest = tmp_path / "batch.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "set_comment", "address": main, "comment": "preview1012"},
            {"op": "set_prototype", "identifier": "puts", "prototype": "int puts(pid_t s)"}]}))
        batch = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert batch.returncode == 0, batch.stdout + batch.stderr
        rows = json.loads(batch.stdout)["results"]
        assert [row["status"] for row in rows] == ["verified", "not_previewed"], rows
        assert "which a preview does not import; name matches: pid_t" in rows[1]["message"], rows[1]
        assert shared_bn.run("py", "exec", "--code", ids).stdout == before

    def test_member_names_like_library_types_preview_normally(self, shared_bn):
        """#993 review round 4: the bare-identifier pre-check treated member
        names as type uses and refused previews that import nothing."""
        shared_bn.load(HELLO_BINARY)
        ids = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", ids).stdout
        res = shared_bn.json("types", "declare",
                             "struct Names993 { int node; int FILE; int clock_t; int pid_t; int sigset_t; };",
                             "--preview", "--summary")
        assert res["ok"] is True and res["verified_count"] == 1, res
        assert shared_bn.run("py", "exec", "--code", ids).stdout == before

    def test_live_imports_rename_collisions_and_later_imports_still_work(self, shared_bn):
        """#993 review rounds 1-2: the live path imports only through BN's own
        import, so a dependency colliding with a user type is renamed, and a
        refused preview leaves later imports working."""
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "typedef uint32_t __time_t;")
        assert shared_bn.run("types", "declare", "struct Collide993 { struct timespec ts; };",
                             "--preview").returncode == 3
        live = shared_bn.json("types", "declare", "struct Collide993 { struct timespec ts; };", "--summary")
        assert live["committed"] is True, live
        assert "__time_t_1 tv_sec" in shared_bn.run("struct", "show", "timespec").stdout
        assert "int64_t" in shared_bn.run("types", "show", "__time_t_1").stdout

        assert shared_bn.run("types", "declare", "struct Pw993 { struct passwd pw; };", "--preview").returncode == 3
        shared_bn.json("proto", "set", "main", "int main(struct passwd* a, int b)")
        probe = shared_bn.run("py", "exec", "--code", "print(bv.get_type_by_name('passwd') is not None)")
        assert probe.stdout.strip().endswith("True"), probe.stdout

    def test_every_pointer_reached_library_type_is_imported(self, shared_bn):
        """#993 review round 2: only the first of several library types
        reached through pointers was imported (the reference walk lost
        members to reused `id()`s), the rest dangled and were listed as
        referenced."""
        shared_bn.load(HELLO_BINARY)
        decl = "struct Many993 { struct tms* a; struct passwd* b; struct group* c; struct tm* d; };"
        live = shared_bn.json("types", "declare", decl, "--summary")["declared_types"]
        wanted = {"tms", "passwd", "group", "tm"}
        assert wanted <= set(live["imported_from_library"]), live
        assert not wanted & set(live.get("referenced_existing", [])), live
        probe = shared_bn.run("py", "exec", "--code",
                              "print(sorted(n for n in ('tms','passwd','group','tm') if bv.get_type_by_name(n) is None))")
        assert probe.stdout.strip().endswith("[]"), probe.stdout

    def test_failed_live_declares_name_the_imports_they_kept(self, tmp_path, shared_bn):
        """#993 review rounds 2-3: a rolled-back live batch, and a declare op
        that is itself refused after BN's parse imported, keep those imports
        (BN cannot un-import) and say so -- on the op's own row, not a
        sibling's."""
        shared_bn.load(HELLO_BINARY)
        manifest = tmp_path / "batch.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "types_declare", "declaration": "struct Ru993 { struct rusage ru; };"},
            {"op": "struct_field_set", "struct_name": "NoSuchStruct993", "offset": "0x0",
             "field_name": "x", "field_type": "int"}]}))
        failed = shared_bn.run("batch", "apply", str(manifest), "--format", "json", "--summary")
        assert failed.returncode == 3, failed.stdout + failed.stderr
        summary = json.loads(failed.stdout)
        assert summary["rolled_back"] is True, summary
        assert "rusage" in summary["kept_library_imports"], summary

        manifest.write_text(json.dumps({"ops": [
            {"op": "types_declare", "declaration": "struct S8_993 { int a; };"},
            {"op": "types_declare",
             "declaration": "struct D2_993 { struct sigaction sa; }; struct uint32_t { int x; };"}]}))
        refused = shared_bn.run("batch", "apply", str(manifest), "--format", "json")
        assert refused.returncode == 3, refused.stdout + refused.stderr
        rows = json.loads(refused.stdout)["results"]
        assert "imported_from_library" not in rows[0], rows[0]
        assert "sigaction" in rows[1]["imported_from_library"], rows[1]

    def test_typed_vptr_keeps_bn_vtable_and_renders_named_virtual_call(self, shared_bn):
        """#963: BN's parser returns a type a declaration only references as an
        EMPTY named struct; defining it overwrote BN's own `...::VTable` with a
        0-byte stub, so the typed vptr led nowhere and the virtual call in
        `svc::dispatch` stayed a raw `*(this + 0x10)`."""
        shared_bn.load(CLASSES_BINARY)
        vtable = "svc::Handler::svc::EchoHandler::VTable"
        miss = shared_bn.run("struct", "show", "svc::Handler")
        assert miss.returncode == 2
        assert "No struct named svc::Handler is defined" in miss.stderr + miss.stdout
        before = shared_bn.json("types", "show", vtable)
        declared = shared_bn.json(
            "types", "declare", f"struct `svc::Handler` {{ struct `{vtable}`* vtable; }};")
        row = declared["results"][0]
        assert list(row["defined_types"]) == ["svc::Handler"], row
        assert row["referenced_existing_types"] == [vtable], row
        assert shared_bn.json("types", "show", vtable)["layout"] == before["layout"]

        locals_ = shared_bn.json("local", "list", "svc::dispatch")
        this = next(v for v in locals_["items"] if v["is_parameter"] and v["name"] == "this")
        shared_bn.json("local", "retype", "svc::dispatch", this["local_id"], "struct `svc::Handler`*")
        text = shared_bn.json("decompile", "svc::dispatch")["text"]
        assert "this->vtable->handle(" in text, text

    def test_class_apply_types_types_the_vptr_and_this_as_one_mutation(self, shared_bn):
        """#993 item 3: `class apply-types` declares the instance struct with a
        vptr typed as BN's own `<Base>::<Class>::VTable*` and retypes `this` on
        the class's methods, previewable and verified as one mutation; a class
        BN made no table of its own for is refused, naming the derived tables."""
        shared_bn.load(CLASSES_BINARY)
        vtable = "svc::Handler::svc::EchoHandler::VTable"
        before = shared_bn.json("types", "show", vtable)["layout"]
        preview = shared_bn.json("class", "apply-types", "svc::EchoHandler", "--preview", "--summary")
        plan = preview["class_apply_types"]
        assert preview["preview"] is True and preview["failed_count"] == 0, preview
        assert plan["vtable_type"] == vtable and plan["instance_struct"] == "declared", plan
        assert {m["name"] for m in plan["retyped_methods"]} >= {
            "svc::EchoHandler::handle", "svc::EchoHandler::name"}, plan
        assert shared_bn.run("types", "show", "svc::EchoHandler").returncode == 2

        live = shared_bn.json("class", "apply-types", "svc::EchoHandler", "--summary")
        assert live["committed"] is True and live["failed_count"] == 0, live
        assert shared_bn.run("struct", "show", "svc::EchoHandler").stdout.splitlines() == [
            "struct svc::EchoHandler // size=0x8",
            f"0x0000: struct {vtable}* vtable",
        ]
        assert shared_bn.json("types", "show", vtable)["layout"] == before
        handle = shared_bn.json("local", "list", "svc::EchoHandler::handle")["items"]
        this = next(v for v in handle if v["is_parameter"] and v["name"] == "this")
        assert this["type"] == "struct svc::EchoHandler*", this

        again = shared_bn.json("class", "apply-types", "svc::EchoHandler", "--summary")
        assert again["class_apply_types"]["instance_struct"] == "unchanged", again
        assert again["changed_count"] == 0 and again["dirty_after"] is False, again

        refused = shared_bn.run("class", "apply-types", "svc::Handler")
        assert refused.returncode == 3, refused.stdout + refused.stderr
        assert "no table type of its own" in refused.stderr + refused.stdout
        assert vtable in refused.stderr + refused.stdout

    def test_class_apply_types_refuses_layout_at_offset_zero(self, shared_bn):
        """#993 review: an embedded base at offset 0 was overwritten by the
        8-byte vptr and reported `verified`; it is refused and left intact."""
        shared_bn.load(CLASSES_BINARY)
        shared_bn.json("types", "declare", "struct `svc::Handler` { void* vtable; int64_t id; int64_t flags; };")
        shared_bn.json("types", "declare", "struct `svc::EchoHandler` { struct `svc::Handler` base; int64_t extra; };")
        before = shared_bn.run("struct", "show", "svc::EchoHandler").stdout
        refused = shared_bn.run("class", "apply-types", "svc::EchoHandler")
        assert refused.returncode == 3, refused.stdout + refused.stderr
        assert "would replace it" in refused.stdout + refused.stderr
        assert shared_bn.run("struct", "show", "svc::EchoHandler").stdout == before

    def test_class_apply_types_stripped_rerun_is_a_noop(self, shared_bn):
        """#993 review: a re-run on a stripped build, where nothing is left to
        write, is a clean noop rather than an exit-3 refusal, and the vtable's
        slot functions are listed as skipped."""
        shared_bn.load(CLASSES_STRIPPED_BINARY)
        first = shared_bn.json("class", "apply-types", "svc::Codec", "--summary")
        assert first["committed"] is True and first["failed_count"] == 0, first
        assert first["class_apply_types"]["skipped_methods"], first
        again = shared_bn.run("class", "apply-types", "svc::Codec", "--format", "json", "--summary")
        assert again.returncode == 0, again.stdout + again.stderr
        summary = json.loads(again.stdout)
        assert summary["noop_count"] == 1 and summary["dirty_after"] is False, summary
        assert summary["class_apply_types"]["instance_struct"] == "unchanged", summary

    def test_qualified_prototype_preview_and_display_name_999(self, shared_bn, tmp_path):
        """#999 on BN's real parser: a `::`-qualified type in backticks and a
        `::`-qualified function name both land (items 1-2), `proto get` names the
        demangled function (item 3), and a batch preview with an AUTO-prototype op
        previews the comment beside it instead of refusing (item 5)."""
        shared_bn.load(CLASSES_BINARY)
        shared_bn.json("types", "declare", "struct `svc::Probe999` { int a; };")
        prototype = "void svc::dispatch(struct `svc::Probe999`* self)"

        manifest = tmp_path / "b.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "set_prototype", "identifier": "svc::dispatch", "prototype": prototype},
            {"op": "set_comment", "function": "svc::dispatch", "comment": "probe"},
        ]}), encoding="utf-8")
        preview = shared_bn.run("batch", "apply", str(manifest), "--preview",
                                "--format", "json", "--summary")
        assert preview.returncode == 0, (preview.stdout, preview.stderr)
        summary = json.loads(preview.stdout)
        assert summary["verified_count"] == 1, summary
        assert [row["index"] for row in summary["preview_skipped"]] == [0], summary

        committed = shared_bn.json("proto", "set", "svc::dispatch", prototype)
        assert committed["results"][0]["status"] == "verified", committed
        got = shared_bn.json("proto", "get", "svc::dispatch")
        assert got["function"]["display_name"] == "svc::dispatch", got
        assert "svc::Probe999" in got["prototype"], got


class TestPrototypeRecovery:
    def test_object_parameter_and_display_name(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        got = shared_bn.json("proto", "get", "svc::EchoHandler::handle")
        note = got["implicit_object_note"]
        assert "`this`" in note and "struct svc::EchoHandler*" in note, got
        text = shared_bn.run("proto", "get", "svc::EchoHandler::handle")
        assert text.returncode == 0, text.stderr
        assert "svc::EchoHandler::handle(" in text.stdout.splitlines()[0], text.stdout
        assert note in text.stdout
        raw = shared_bn.run("proto", "get", "svc::EchoHandler::handle", "--mangled")
        assert raw.returncode == 0, raw.stderr
        assert got["function"]["name"] + "(" in raw.stdout.splitlines()[0], raw.stdout


    def test_local_rename_keeps_object_note_observational(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        function = "svc::EchoHandler::handle"
        before = shared_bn.json("proto", "get", function)
        address = before["function"]["address"]
        flag_probe = f"result = bv.get_function_at({address}).has_user_type"
        assert shared_bn.json("py", "exec", flag_probe)["result"] is False
        assert "`this`" in before["implicit_object_note"], before
        renamed = shared_bn.json("local", "rename", function, "this", "self")
        assert renamed["results"][0]["status"] == "verified", renamed
        after = shared_bn.json("proto", "get", function)
        assert shared_bn.json("py", "exec", flag_probe)["result"] is False
        assert "* self" in after["prototype"], after
        note = after["implicit_object_note"]
        assert "`self`" in note and "struct svc::EchoHandler*" in note, after
        assert "demangled C++ signature describes a leading object parameter" in note, after
        assert "did not identify" not in note and "Do not reinterpret" not in note, after
        text = shared_bn.run("proto", "get", function)
        assert text.returncode == 0 and note in text.stdout, (text.stdout, text.stderr)


    def test_local_parameter_edits_do_not_attribute_recovery(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        function = "svc::EchoHandler::handle"
        before = shared_bn.json("proto", "get", function)
        address = before["function"]["address"]
        flag_probe = (
            f"f = bv.get_function_at({address})\n"
            "result = [f.has_user_type, f.is_var_user_defined(f.parameter_vars[0])]")
        assert shared_bn.json("py", "exec", flag_probe)["result"] == [False, False]
        retyped = shared_bn.json("local", "retype", function, "this", "int64_t")
        assert retyped["results"][0]["status"] == "verified", retyped
        assert shared_bn.json("py", "exec", flag_probe)["result"] == [False, True]
        after = shared_bn.json("proto", "get", function)
        note = after["implicit_object_note"]
        assert "`this`" in note and "int64_t" in note, after
        assert "BN recovered" not in note and "implicit object parameter" not in note, after
        renamed = shared_bn.json("local", "rename", function, "this", "__sync")
        assert renamed["results"][0]["status"] == "verified", renamed
        renamed_back = shared_bn.json("local", "rename", function, "__sync", "this")
        assert renamed_back["results"][0]["status"] == "verified", renamed_back
        again = shared_bn.json("proto", "get", function)
        assert "`this`" in again["implicit_object_note"] and "int64_t" in again["implicit_object_note"], again
        assert "BN recovered" not in again["implicit_object_note"], again


    def test_this_refusal_advice_and_nested_template(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        shared_bn.json("types", "declare", "struct `ns::Box<int, ns::Tag<char> >` { int value; };")
        prototype = "int handle(struct `ns::Box<int, ns::Tag<char> >`* this)"
        rejected = shared_bn.run("proto", "set", "svc::EchoHandler::handle",
                                 prototype, "--format", "json")
        assert rejected.returncode == 3, (rejected.stdout, rejected.stderr)
        row = json.loads(rejected.stdout)["results"][0]
        assert row["status"] == "invalid_request", row
        assert "`self`" in row["message"], row
        assert "local retype" not in row["message"] and "declare it first" not in row["message"], row
        # Apply the exact spelling correction the refusal recommends.
        applied = shared_bn.json("proto", "set", "svc::EchoHandler::handle",
                                 prototype.replace("* this", "* self"))
        assert applied["results"][0]["status"] == "verified", applied
        got = shared_bn.json("proto", "get", "svc::EchoHandler::handle")
        assert "ns::Box<int, ns::Tag<char> >* self" in got["prototype"], got
        assert "`self`" in got["implicit_object_note"], got
        assert "BN recovered" not in got["implicit_object_note"], got
        assert "user-defined prototype" not in got["implicit_object_note"], got

    def test_named_function_type_fallback(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "typedef int `ns::Callback`(int value);")
        applied = shared_bn.json("proto", "set", "main", "ns::Callback")
        assert applied["results"][0]["status"] == "verified", applied
        got = shared_bn.json("proto", "get", "main")
        assert got["prototype"] == "int32_t(int32_t value)", got

    def test_display_declarator_preserves_type_spelling_hint(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        shared_bn.json("types", "declare", "struct `ns::Cls` { int value; };")
        for declarator in ["ns::Cls::method", "method"]:
            prototype = f"void* {declarator}(struct ns::Cls* const self, void* p)"
            rejected = shared_bn.run("proto", "set", "svc::EchoHandler::handle",
                                     prototype, "--format", "json")
            assert rejected.returncode == 3, (rejected.stdout, rejected.stderr)
            row = json.loads(rejected.stdout)["results"][0]
            assert "struct `ns::Cls`" in row["message"], row
            assert "declare it first" not in row["message"], row
            applied = shared_bn.json("proto", "set", "svc::EchoHandler::handle",
                                     prototype.replace("struct ns::Cls", "struct `ns::Cls`"))
            assert applied["results"][0]["status"] == "verified", applied

    def test_this_advice_avoids_existing_parameter_names(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        shared_bn.json("types", "declare", "struct W { int value; };")
        prototype = "int32_t c_plain(struct W* this, int self, int self_1)"
        rejected = shared_bn.run("proto", "set", "main", prototype, "--format", "json")
        assert rejected.returncode == 3, (rejected.stdout, rejected.stderr)
        row = json.loads(rejected.stdout)["results"][0]
        suggestion = re.search(r"Rename that parameter to `(\w+)`", row["message"])
        assert suggestion is not None, row
        name = suggestion.group(1)
        assert name not in {"self", "self_1", "this"}, row
        # Use precisely the suggested spelling, not a separately chosen fix.
        applied = shared_bn.json("proto", "set", "main", prototype.replace("* this", f"* {name}"))
        assert applied["results"][0]["status"] == "verified", applied
        got = shared_bn.json("proto", "get", "main")
        assert f"* {name}" in got["prototype"], got

    def test_c_this_parameter_is_not_an_object(self, shared_bn, tmp_path):
        c_source = tmp_path / "cfunc.c"
        cpp_source = tmp_path / "driver.cpp"
        obj = tmp_path / "cfunc.o"
        binary = tmp_path / "c_this"
        c_source.write_text("int cfunc(int this) { return this + 13; }\n", encoding="utf-8")
        cpp_source.write_text(
            'extern "C" int cfunc(int); int main() { return cfunc(2); }\n', encoding="utf-8")
        subprocess.run(["cc", "-g", "-O0", "-c", str(c_source), "-o", str(obj)],
                       check=True, capture_output=True, text=True, timeout=30)
        subprocess.run(["c++", "-g", "-O0", str(cpp_source), str(obj), "-o", str(binary)],
                       check=True, capture_output=True, text=True, timeout=30)
        shared_bn.load(binary)
        got = shared_bn.json("proto", "get", "cfunc")
        assert "this" in got["prototype"], got
        note = got["implicit_object_note"]
        assert "implicit object parameter" not in note, got

    def test_static_member_callers_pass_their_entry_value_1013(self, shared_bn):
        """`reg::Registry::get` is static; BN still gives it a `this`. Both
        callers pass their own entry `rdi` (SSA version 0); `set_handler`'s
        callers pass the returned object."""
        shared_bn.load(STATICS_BINARY)
        got = shared_bn.json("proto", "get", "reg::Registry::get")
        fact = got["first_argument_entry_value"]
        assert fact["checked"] == 2 and fact["entry_value"] == 2, got
        assert fact["registers"] == ["rdi"] and fact["truncated"] is False, got
        assert "implicit_object_note" in got, got
        text = shared_bn.run("proto", "get", "reg::Registry::get")
        assert ("note: At 2 of 2 direct call sites the first argument is the caller's "
                "own entry value of rdi (SSA version 0)") in text.stdout, text.stdout
        member = shared_bn.json("proto", "get", "reg::Registry::set_handler")
        assert member["first_argument_entry_value"]["entry_value"] == 0, member

    def test_undefined_type_declare_command_runs_as_printed_1013(self, shared_bn):
        shared_bn.load(STATICS_BINARY)
        got = shared_bn.json("proto", "get", "reg::Registry::get")
        [row] = got["undefined_types"]
        assert row["name"] == "reg::Registry", got
        command = row["declare_command"]
        assert command == "bn types declare 'struct `reg::Registry`;'", got
        assert f"`{command}`" in shared_bn.run("proto", "get", "reg::Registry::get").stdout
        prototype = "int64_t get(struct `reg::Registry`* self)"
        refused = shared_bn.run("proto", "set", "reg::Registry::get", prototype)
        assert "Reference to unknown type" in refused.stdout + refused.stderr, refused
        argv = shlex.split(command)
        assert argv[0] == "bn", argv
        declared = shared_bn.run(*argv[1:])
        assert declared.returncode == 0, declared.stdout + declared.stderr
        committed = shared_bn.json("proto", "set", "reg::Registry::get", prototype)
        assert committed["results"][0]["status"] == "verified", committed
        assert "undefined_types" not in shared_bn.json("proto", "get", "reg::Registry::get")


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

    A missing cross-compiler or a failed cross-compile is a visible skip by
    default and a FAILURE under `BN_REQUIRE_REAL_TESTS=1` (`refuse_silent_skip`):
    every test on this probe is a real-architecture guard the mocked suite
    cannot stand in for, so a strict lane that lost the toolchain must say so
    rather than report green with the guard unexercised.
    """
    cc = shutil.which("aarch64-linux-gnu-gcc")
    if cc is None:
        refuse_silent_skip(
            "aarch64-linux-gnu-gcc not available, so the AArch64 probe lane did not "
            "run; install the cross toolchain (Debian/Ubuntu: gcc-aarch64-linux-gnu)")
    workdir = tmp_path_factory.mktemp("aarch64-probe")
    src = workdir / "probe.c"
    src.write_text("int add(int a, int b){return a + b;}\nint main(){return add(1, 2);}\n")
    probe = workdir / "probe_aarch64"
    build = subprocess.run(
        [cc, "-O0", "-no-pie", "-static", str(src), "-o", str(probe)],
        capture_output=True, text=True, timeout=120,
    )
    if build.returncode != 0:
        refuse_silent_skip(
            f"aarch64 cross-compile failed (exit {build.returncode}), so the AArch64 "
            f"probe lane did not run; fix the cross toolchain: {build.stderr.strip()}")
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


class TestOrientStringsSampleAArch64:
    """M-12 real-path guard. On a static AArch64 ELF, BN's string scan accepts
    runs of instruction bytes that happen to decode as printable ASCII, and
    `.text` sits below `.rodata` in the address space -- so the orient sample,
    the head of an address-ordered scan, was twenty rows of `@8\\t@`-shaped junk
    (plus the ELF header, which no section claims) on every such file, while
    `strings --section .rodata` had the real literals. The mocked tests prove
    the ranking; this proves BN's real section semantics drive it: no sampled
    section is `ReadOnlyCode`, the sample names its sections, and the default
    `strings` dump leads with the same sections."""

    def test_orient_sample_comes_from_data_sections(self, shared_bn, aarch64_probe):
        shared_bn.load(aarch64_probe, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        sections = json.loads(shared_bn.run("sections", "--format", "json").stdout)
        semantics = {s["name"]: s["semantics"] for s in sections["items"]}
        assert "ReadOnlyCode" in semantics.values(), semantics  # premise: BN marks code

        orient = json.loads(shared_bn.run("evidence", "orient", "--format", "json").stdout)
        sample = orient["strings_sample"]
        assert sample["items"], orient
        assert "sample_fallback" not in sample, sample
        drawn = sample["sample_sections"]
        assert drawn, sample
        assert all(semantics[name] != "ReadOnlyCode" for name in drawn), (drawn, semantics)
        # every sampled row is claimed by one of the named sections -- no header row
        starts = {name: (int(s["start"], 16), int(s["end"], 16))
                  for name, s in ((s["name"], s) for s in sections["items"])}
        for row in sample["items"]:
            addr = int(row["address"], 16)
            assert any(lo <= addr < hi for lo, hi in (starts[n] for n in drawn)), (row, drawn)

        # The default dump (at orient's min-length) leads with the same
        # data-section strings, names the code / metadata sections it ranked
        # last -- never `.rodata` -- and `.rodata` is the data section the tip
        # recommends narrowing to.
        dump = json.loads(shared_bn.run(
            "strings", "--limit", "5", "--min-length", str(orient["strings_min_length"]),
            "--format", "json").stdout)
        assert [r["value"] for r in dump["items"]] == [r["value"] for r in sample["items"][:5]]
        assert dump["noise_strings"] > 0, dump
        assert ".rodata" not in dump["noise_sections"], dump["noise_sections"]
        assert dump["data_sections"][0] == ".rodata", dump["data_sections"]
        text = shared_bn.run("strings", "--limit", "5").stdout
        assert "// tip:" in text and "--section .rodata" in text, text
        # Under a filter the caller chose, the text states where the matches
        # sit instead of recommending the flag already in use.
        narrowed = shared_bn.run("strings", "--limit", "5", "--min-length",
                                 str(orient["strings_min_length"])).stdout
        assert "// note:" in narrowed and "narrow with" not in narrowed, narrowed


def _strip_section_headers(binary: Path, out: Path) -> Path:
    """*binary* with its ELF64 section-header table unlinked (e_shoff / e_shnum /
    e_shstrndx zeroed) -- what `objcopy --strip-section-headers` produces,
    without needing a binutils new enough to have that flag. BN then recovers
    only what the dynamic segment describes (`.interp`, `.dynamic`, `.got_*`)
    and marks no code section at all."""
    data = bytearray(binary.read_bytes())
    assert data[:4] == b"\x7fELF" and data[4] == 2, "an ELF64 fixture"
    struct.pack_into("<Q", data, 0x28, 0)
    struct.pack_into("<H", data, 0x3c, 0)
    struct.pack_into("<H", data, 0x3e, 0)
    out.write_bytes(data)
    out.chmod(0o755)
    return out


class TestStringsRankingWithoutCodeSections:
    """M-12 round 2: BN 6.1 never produces a sectionless view -- every loader
    adds `.synthetic_builtins` (an ExternalSection) -- so a classifier that
    treats "has any section" as "sectioned" and "no section claims it" as
    noise ranks EVERY string of a raw image or a section-header-stripped ELF
    as `(no section)` noise: the literals printed under a tip calling them
    code bytes, and orient claiming it fell back past the data sections. The
    mocked `_RawBV` used to have no `sections` attribute at all, a shape BN
    never gives, which is why the suite could not see it; these drive the real
    loader."""

    _LITERALS = ("vxWorks boot loader", "boot: %s", "default.cfg")

    def test_raw_mapped_view_strings_are_literals_not_noise(self, shared_bn, tmp_path):
        image = bytearray(0x400)
        for offset, literal in zip((0x100, 0x200, 0x300), self._LITERALS):
            image[offset:offset + len(literal) + 1] = literal.encode() + b"\0"
        blob = tmp_path / "firmware.bin"
        blob.write_bytes(image)
        shared_bn.load(blob, copy=False)

        sections = json.loads(shared_bn.run("sections", "--format", "json").stdout)
        # premise: the raw view is NOT sectionless -- only the synthetic one
        assert {s["semantics"] for s in sections["items"]} == {"ExternalSection"}, sections

        dump = json.loads(shared_bn.run("strings", "--format", "json").stdout)
        assert [r["value"] for r in dump["items"]] == list(self._LITERALS), dump
        assert dump["noise_strings"] == 0 and dump["noise_sections"] == [], dump
        assert dump["data_sections"] == [], dump
        text = shared_bn.run("strings").stdout
        assert "tip:" not in text and "code bytes" not in text, text

        orient = json.loads(shared_bn.run("evidence", "orient", "--format", "json").stdout)
        sample = orient["strings_sample"]
        assert "sample_fallback" not in sample, sample
        assert [r["value"] for r in sample["items"]] == list(self._LITERALS), sample
        assert sample["sample_sections"] == [], sample

    def test_section_header_stripped_elf_keeps_literals_as_data(self, shared_bn, tmp_path):
        stripped = _strip_section_headers(HELLO_BINARY, tmp_path / "hello_stripped")
        shared_bn.load(stripped, copy=False)

        sections = json.loads(shared_bn.run("sections", "--format", "json").stdout)
        semantics = {s["name"]: s["semantics"] for s in sections["items"]}
        # premise: BN recovered loader metadata from the dynamic segment, but
        # marked no code section -- so it has not said where code is.
        assert "ReadOnlyCode" not in semantics.values(), semantics
        assert any(name.startswith(".interp") for name in semantics), semantics

        dump = json.loads(shared_bn.run("strings", "--format", "json").stdout)
        values = [r["value"] for r in dump["items"]]
        assert "starting up" in values and "greeting length: %d\n" in values, values
        # the unclaimed addresses (code AND literals) are data, in address
        # order; only the recovered metadata sections rank last
        assert "(no section)" not in dump["noise_sections"], dump["noise_sections"]
        assert all(name.startswith((".interp", ".dynamic", ".got")) for name in dump["noise_sections"]), dump
        unclaimed = [int(r["address"], 16) for r in dump["items"]
                     if r["value"] in ("starting up", "greeting length: %d\n")]
        data_rows = [int(r["address"], 16) for r in dump["items"]][:len(dump["items"]) - dump["noise_strings"]]
        assert data_rows == sorted(data_rows) and all(a in data_rows for a in unclaimed), dump

        orient = json.loads(shared_bn.run("evidence", "orient", "--format", "json").stdout)
        sample = orient["strings_sample"]
        assert "sample_fallback" not in sample, sample
        assert "starting up" in [r["value"] for r in sample["items"]], sample


class TestAArch64ProbeStrictGate:
    """The AArch64 probe is the only real-architecture guard the lane has; a
    missing cross-compiler or a failed cross-compile must be a visible skip by
    default and a FAILURE -- carrying the build diagnostics -- under
    `BN_REQUIRE_REAL_TESTS=1`, never a silent pass of an unexercised guard."""

    @staticmethod
    def _prime(monkeypatch, tmp_path_factory, *, which, build):
        monkeypatch.setattr(shutil, "which", which)
        monkeypatch.setattr(subprocess, "run", build)
        _build_and_prime_aarch64_probe(None, tmp_path_factory)

    def test_missing_cross_compiler_fails_in_strict_mode(self, monkeypatch, tmp_path_factory):
        from _pytest.outcomes import Failed, Skipped

        def no_build(*a, **k):
            raise AssertionError("must not try to compile without a compiler")

        monkeypatch.delenv("BN_REQUIRE_REAL_TESTS", raising=False)
        with pytest.raises(Skipped) as skipped:
            self._prime(monkeypatch, tmp_path_factory, which=lambda name: None, build=no_build)
        assert "aarch64-linux-gnu-gcc" in str(skipped.value)

        monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1")
        with pytest.raises((Failed, Skipped)) as failed:
            self._prime(monkeypatch, tmp_path_factory, which=lambda name: None, build=no_build)
        assert isinstance(failed.value, Failed), failed.value
        assert "aarch64-linux-gnu-gcc" in str(failed.value)
        assert "BN_REQUIRE_REAL_TESTS" in str(failed.value)

    def test_failed_cross_compile_fails_in_strict_mode_with_diagnostics(self, monkeypatch,
                                                                        tmp_path_factory):
        from _pytest.outcomes import Failed, Skipped

        def broken_build(*a, **k):
            return subprocess.CompletedProcess(a[0] if a else [], 1, stdout="",
                                               stderr="ld: cannot find crt1.o")

        monkeypatch.delenv("BN_REQUIRE_REAL_TESTS", raising=False)
        with pytest.raises(Skipped) as skipped:
            self._prime(monkeypatch, tmp_path_factory,
                        which=lambda name: "/usr/bin/" + name, build=broken_build)
        assert "cannot find crt1.o" in str(skipped.value)

        monkeypatch.setenv("BN_REQUIRE_REAL_TESTS", "1")
        with pytest.raises((Failed, Skipped)) as failed:
            self._prime(monkeypatch, tmp_path_factory,
                        which=lambda name: "/usr/bin/" + name, build=broken_build)
        assert isinstance(failed.value, Failed), failed.value
        assert "cannot find crt1.o" in str(failed.value) and "exit 1" in str(failed.value)
        assert "BN_REQUIRE_REAL_TESTS" in str(failed.value)


class TestSessionStopCleanupBudget:
    def test_cleanup_completes_after_full_teardown_allowance(
        self, monkeypatch, tmp_path
    ):
        # Exercise a real subprocess deadline, scaled 10:1 to avoid spending
        # 35 seconds on a stand-in. It needs the full 30s teardown allowance,
        # 2s escalation wait and 3s of outer CLI overhead before it finishes.
        completed = tmp_path / "cleanup-completed"
        program = (
            "import pathlib, sys, time\n"
            "assert sys.argv[1:] == ['session', 'stop', 'budget-edge', '--discard-unsaved']\n"
            "time.sleep(3.5)\n"
            f"pathlib.Path({str(completed)!r}).write_text('process exited')\n"
        )
        monkeypatch.setattr(
            sys.modules[__name__], "_BN_CLI", [sys.executable, "-c", program]
        )
        run = subprocess.run

        def scaled_run(*args, timeout, **kwargs):
            return run(*args, timeout=timeout / 10.0, **kwargs)

        monkeypatch.setattr(subprocess, "run", scaled_run)

        _session_stop("budget-edge")

        assert completed.read_text() == "process exited"


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
            refuse_silent_skip(
                "no creatable gap address in this fixture, so the preview/live "
                "agreement #304 fixed is unexercised; restore the gap between "
                "two adjacent analysed functions in tests/fixtures")

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

            # Reload from the still-read-only mount: the same project restores
            # the cache copy (#318), and the note says whose it is (#1036).
            info2 = _session_start(str(prog))
            inst = info2["instance_id"]
            notes2 = info2["loaded"][0]["notes"]
            assert any(n.startswith(f"restored cached database {cache_file}") for n in notes2), notes2
            search = _bn("--instance", inst, "function", "search", "RO318_MARKER", "--format", "json")
            names = [i["name"] for i in json.loads(search.stdout).get("items", [])]
            assert "RO318_MARKER" in names, f"annotation lost on reload (blank): {names}"
            _session_stop(inst)
            inst = None

            # #1036 item 1: another project's start loads the raw bytes, and the
            # command its note prints restores the entry when run as printed.
            elsewhere = tmp_path / "other-project"
            elsewhere.mkdir()
            started = subprocess.run(
                [*_BN_CLI, "session", "start", "--format", "json", str(prog)],
                capture_output=True, text=True, timeout=_SESSION_START_TIMEOUT,
                env=_env(), cwd=elsewhere)
            assert started.returncode == 0, started.stderr
            info3 = json.loads(started.stdout)
            inst = info3["instance_id"]
            loaded = info3["loaded"][0]
            assert loaded["path"] == str(prog), loaded
            [skipped] = [n for n in loaded["notes"]
                         if n.startswith(f"did not restore the cached database {cache_file}")]
            command = skipped.rsplit("`", 2)[1]
            env = _env()
            env["PATH"] = f"{Path(_BN_CLI[0]).parent}{os.pathsep}{env.get('PATH', '')}"
            restored = subprocess.run(
                ["sh", "-c", command], capture_output=True, text=True,
                timeout=_SESSION_START_TIMEOUT, env=env, cwd=elsewhere)
            assert restored.returncode == 0, f"{command}: {restored.stdout}\n{restored.stderr}"
            search = _bn("--instance", inst, "-t", cache_file.name, "function", "search",
                         "RO318_MARKER", "--format", "json")
            names = [i["name"] for i in json.loads(search.stdout).get("items", [])]
            assert "RO318_MARKER" in names, f"{search.stdout}\n{search.stderr}"
        finally:
            if inst:
                _session_stop(inst)
            ro.chmod(0o700)
            if cache_file is not None and cache_file.exists():
                cache_file.unlink()
                writer = cache_file.with_name(cache_file.name + ".writer.json")
                if writer.exists():
                    writer.unlink()


class TestSidecarContentValidation:
    """A sidecar `.bndb` is preferred by path; whether it describes the bytes
    now at that path is decided from the sha256 the bridge stamps on a raw view
    when it OPENS it (and `bn save` carries into the database). Only real BN
    can prove the stamp survives `create_database` and comes back through
    `binaryninja.load` + the analyzed-view restore, that the view's bytes stay
    what it loaded when the file is replaced underneath it, and that a legacy
    database's stored bytes (`bv.file.raw`) hash equal to its source -- the
    mocked lane fakes every one of those ends."""

    @staticmethod
    def _function_names(shared_bn) -> set[str]:
        listing = shared_bn.json("function", "list", "--limit", "500")
        return {f["name"] for f in (listing.get("items") if isinstance(listing, dict) else listing)}

    def test_replaced_bytes_load_raw_and_restored_bytes_reload_the_database(
            self, shared_bn, tmp_path):
        prog = tmp_path / "prog"
        hello = Path(HELLO_BINARY).read_bytes()
        prog.write_bytes(hello)
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")

        # Save the database for the hello bytes, then prove a plain reload
        # prefers it with its provenance VERIFIED (no unverified note).
        shared_bn.load(prog, copy=False)
        saved = shared_bn.json("save")
        assert saved["path"] == str(sidecar), saved
        assert "note" not in saved, saved  # the stamp landed silently
        assert shared_bn.run("close", str(prog)).returncode == 0
        reloaded = shared_bn.json("load", str(prog))
        assert reloaded["path"] == str(sidecar), reloaded
        notes = " ".join(reloaded["notes"])
        assert "unverified" not in notes and "NOT restoring" not in notes, notes
        assert "greet" in self._function_names(shared_bn)
        assert shared_bn.run("close", str(sidecar)).returncode == 0

        # Different bytes at the same path: the database must NOT be restored.
        prog.write_bytes(Path(ADD_BINARY).read_bytes())
        replaced = shared_bn.json("load", str(prog))
        assert replaced["path"] == str(prog), replaced
        assert any(
            n.startswith("NOT restoring ") and str(sidecar) in n and "--no-bndb" in n
            for n in replaced["notes"]), replaced["notes"]
        names = self._function_names(shared_bn)
        assert {"add", "mul"} <= names and "greet" not in names, names
        assert shared_bn.run("close", str(prog)).returncode == 0

        # The original bytes back: the database describes them again.
        prog.write_bytes(hello)
        restored = shared_bn.json("load", str(prog))
        assert restored["path"] == str(sidecar), restored
        assert not any(n.startswith("NOT restoring ") for n in restored["notes"])
        assert "greet" in self._function_names(shared_bn)
        assert shared_bn.run("close", str(sidecar)).returncode == 0

    def test_a_file_replaced_under_the_open_view_is_not_certified_by_save(
            self, shared_bn, tmp_path):
        """Review P1 (both reviewers): `bn save` hashed the file ON DISK, so a
        file replaced between load and save was stamped as the database's
        source and the stale database then matched forever. The identity is
        taken at open; a save -- and a second save after an earlier successful
        one -- certifies the bytes the view LOADED."""
        prog = tmp_path / "prog"
        hello = Path(HELLO_BINARY).read_bytes()
        prog.write_bytes(hello)
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")

        shared_bn.load(prog, copy=False)
        first = shared_bn.json("save")
        assert first["path"] == str(sidecar) and "note" not in first, first
        prog.write_bytes(Path(ADD_BINARY).read_bytes())       # replaced while open
        second = shared_bn.json("save")
        assert second["path"] == str(sidecar) and "note" not in second, second
        stamp = shared_bn.json(
            "py", "exec", "result = dict(bv.query_metadata('bn_agent_bridge'))")["result"]
        assert stamp == {"source_sha256": hashlib.sha256(hello).hexdigest(),
                         "source_size": len(hello)}, stamp
        assert shared_bn.run("close", str(prog)).returncode == 0

        # The database describes the hello bytes and says so; the file holds
        # the add bytes: the raw file loads, loudly.
        reloaded = shared_bn.json("load", str(prog))
        assert reloaded["path"] == str(prog) and reloaded.get("already_open") is None, reloaded
        assert any(
            n.startswith("NOT restoring ") and str(sidecar) in n
            for n in reloaded["notes"]), reloaded["notes"]
        names = self._function_names(shared_bn)
        assert {"add", "mul"} <= names and "greet" not in names, names
        assert shared_bn.run("close", str(prog)).returncode == 0

    def test_an_already_open_stale_database_is_judged_before_being_returned(
            self, shared_bn, tmp_path):
        """Review P1 (Astra): with the stale database explicitly open, a load
        of the raw path found it by path and returned it `already_open` with
        only dedupe notes. The open candidate is judged like a fresh one: the
        explicit database stays open, the raw bytes load beside it."""
        prog = tmp_path / "prog"
        hello = Path(HELLO_BINARY).read_bytes()
        prog.write_bytes(hello)
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")
        shared_bn.load(prog, copy=False)
        assert shared_bn.json("save")["path"] == str(sidecar)
        assert shared_bn.run("close", str(prog)).returncode == 0
        prog.write_bytes(Path(ADD_BINARY).read_bytes())

        explicit = shared_bn.json("load", str(sidecar))
        assert explicit["path"] == str(sidecar), explicit
        assert not any("NOT restoring" in n for n in explicit["notes"]), explicit["notes"]
        # Round 3 P2: kept (requested explicitly), but no longer silent.
        assert any("requested explicitly" in n for n in explicit["notes"]), explicit["notes"]

        raw = shared_bn.json("load", str(prog))
        assert raw["path"] == str(prog) and raw.get("already_open") is None, raw
        assert any(
            n.startswith("NOT restoring ") and "Loaded the raw file instead" in n
            for n in raw["notes"]), raw["notes"]
        rows = {row["filename"]: row for row in shared_bn.json("target", "list")["items"]}
        assert set(rows) == {str(sidecar), str(prog)}, rows
        assert rows[str(prog)]["source_verified"] is None
        # A raw view beside its own .bndb makes the path selector ambiguous
        # (both match `prog`); the row's target_id is the unambiguous handle.
        listing = shared_bn.json("function", "list", "--limit", "500",
                                 "-t", rows[str(prog)]["target_id"])
        names = {f["name"] for f in listing["items"]}
        assert {"add", "mul"} <= names and "greet" not in names, names

        # A third load of the raw path dedupes against the raw view, and the
        # note says the existing target was returned -- not "loaded".
        again = shared_bn.json("load", str(prog))
        assert again["already_open"] is True and again["path"] == str(prog), again
        refusal = [n for n in again["notes"] if n.startswith("NOT restoring ")]
        assert refusal and f"{prog} is already open" in refusal[0], again["notes"]
        assert "Loaded the raw file instead" not in refusal[0]
        assert shared_bn.run("close", str(prog)).returncode == 0
        assert shared_bn.run("close", str(sidecar)).returncode == 0

    def test_a_legacy_database_is_verified_from_the_bytes_it_stores_and_stamped_on_save(
            self, shared_bn, tmp_path):
        """Review P2: a database saved before content validation carries no
        record, but BN stores the original bytes in it. Real BN proves those
        hash equal to the source file: the database is VERIFIED from them, the
        verdict is on the target row, the next save persists the record, and
        the reload after that is silent."""
        prog = tmp_path / "prog"
        hello = Path(HELLO_BINARY).read_bytes()
        prog.write_bytes(hello)
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")
        shared_bn.load(prog, copy=False)
        assert shared_bn.json("save")["path"] == str(sidecar)
        assert shared_bn.run("close", str(prog)).returncode == 0
        # Make it a pre-validation database: strip the record and persist that.
        shared_bn.load(sidecar, copy=False)
        shared_bn.json("py", "exec",
                       "bv.remove_metadata('bn_agent_bridge')\n"
                       "result = bv.save_auto_snapshot()")
        assert shared_bn.run("close", str(sidecar)).returncode == 0

        legacy = shared_bn.json("load", str(prog))
        assert legacy["path"] == str(sidecar), legacy
        notes = " ".join(legacy["notes"])
        assert "unverified" not in notes and "NOT restoring" not in notes, notes
        assert any(n.startswith("database provenance verified from the bytes stored in")
                   for n in legacy["notes"]), legacy["notes"]
        assert legacy["targets"][0]["source_verified"] is True, legacy["targets"]
        info = shared_bn.json("target", "info")
        assert info["source_verified"] is True, info
        assert info["existing_annotations"]["source_verified"] is True, info
        assert "verified against the raw file's content" in info["existing_annotations"]["provenance_hint"]
        assert "greet" in self._function_names(shared_bn)
        saved = shared_bn.json("save")                      # persists the in-memory stamp
        assert saved["path"] == str(sidecar), saved
        assert shared_bn.run("close", str(sidecar)).returncode == 0

        silent = shared_bn.json("load", str(prog))
        assert silent["path"] == str(sidecar), silent
        assert not any("provenance" in n for n in silent["notes"]), silent["notes"]
        assert silent["targets"][0]["source_verified"] is True
        assert shared_bn.run("close", str(sidecar)).returncode == 0

        # ...and once stamped, other bytes at the path are refused like any
        # other mismatch -- the hazard a record-less database left as a note.
        prog.write_bytes(Path(ADD_BINARY).read_bytes())
        refused = shared_bn.json("load", str(prog))
        assert refused["path"] == str(prog), refused
        assert any(n.startswith("NOT restoring ") for n in refused["notes"]), refused["notes"]
        assert shared_bn.run("close", str(prog)).returncode == 0

    def test_an_explicitly_named_stale_database_is_kept_but_reported(
            self, shared_bn, tmp_path):
        """Review round 3 P2: `bn load <db>` never judged its database, so a
        stale one came back silently (`notes: []`, `source_verified: null`).
        It is kept -- the caller asked for it -- but the record naming other
        bytes than the sibling raw file now holds is reported, on the load
        notes, the row and both text faces."""
        prog = tmp_path / "prog"
        prog.write_bytes(Path(HELLO_BINARY).read_bytes())
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")
        shared_bn.load(prog, copy=False)
        assert shared_bn.json("save")["path"] == str(sidecar)
        assert shared_bn.run("close", str(prog)).returncode == 0
        prog.write_bytes(Path(ADD_BINARY).read_bytes())

        explicit = shared_bn.json("load", str(sidecar))
        assert explicit["path"] == str(sidecar) and explicit.get("already_open") is None, explicit
        assert not any("NOT restoring" in n for n in explicit["notes"]), explicit["notes"]
        note = next((n for n in explicit["notes"] if "requested explicitly" in n), None)
        assert note and note.startswith(f"{sidecar} was saved for different bytes than {prog}"), (
            explicit["notes"])
        assert f"`bn load {prog} --no-bndb`" in note
        assert explicit["targets"][0]["source_verified"] is False, explicit["targets"]
        assert "greet" in self._function_names(shared_bn)      # the database's analysis, kept
        info = shared_bn.json("target", "info")
        assert info["existing_annotations"]["source_verified"] is False, info
        text = shared_bn.run("target", "info")
        assert text.returncode == 0 and "source-verified=false" in text.stdout, text.stdout
        orient = shared_bn.run("evidence", "orient")
        assert orient.returncode == 0 and "source-verified=false" in orient.stdout, orient.stdout
        assert shared_bn.run("close", str(sidecar)).returncode == 0

        # The original bytes back: the same explicit load verifies, silently.
        prog.write_bytes(Path(HELLO_BINARY).read_bytes())
        back = shared_bn.json("load", str(sidecar))
        assert not any("requested explicitly" in n for n in back["notes"]), back["notes"]
        assert back["targets"][0]["source_verified"] is True, back["targets"]
        assert shared_bn.run("close", str(sidecar)).returncode == 0

    def test_an_export_reloads_paired_and_still_warns_about_its_annotations_1036(
            self, shared_bn, tmp_path):
        """#1036 items 4-5 on real BN: `save <path>` is an export that says so;
        loading it back keeps `-t prog` on the raw view and verifies the copy
        against the raw file BN recorded in it (no sibling exists). The copy is
        disclosed as a file this bridge wrote, and -- per-annotation
        attribution having been dropped after three review rounds -- its
        annotations, even this bridge's own rename, stay "may predate this
        run"; so does a reload after the source view closed."""
        prog = tmp_path / "ro" / "prog"
        prog.parent.mkdir()
        prog.write_bytes(Path(HELLO_BINARY).read_bytes())
        prog.chmod(0o755)
        export = tmp_path / "out" / "prog.bndb"
        export.parent.mkdir()
        shared_bn.load(prog, copy=False)
        renamed = shared_bn.run("rename", "greet", "greet_v2", "--format", "json")
        assert renamed.returncode == 0, renamed.stdout + renamed.stderr

        saved = shared_bn.json("save", str(export))
        assert saved["export"] is True and saved["restart_path"] == str(prog), saved

        shared_bn.load(export, copy=False)
        raw = shared_bn.json("-t", "prog", "target", "info")
        assert raw["filename"] == str(prog), raw
        assert raw["exports"] == [str(export)] and raw["database_path"] is None, raw
        db = shared_bn.json("-t", "prog.bndb", "target", "info")
        annotations = db["existing_annotations"]
        assert annotations["opened_from_database"] is True, annotations
        assert annotations["restored_from_cache"] is False, annotations
        assert annotations["source_verified"] is True, annotations
        assert annotations["written_by_this_bridge"]["kind"] == "export", annotations
        assert "own_at_open" not in annotations, annotations
        assert "may predate this run" in (annotations["provenance_hint"] or ""), annotations
        shown = shared_bn.run("-t", "prog.bndb", "target", "info")
        assert "opened from a file this bridge exported from" in shown.stdout, shown.stdout
        assert shared_bn.run("close", str(export)).returncode == 0

        second = tmp_path / "out" / "prog2.bndb"
        assert shared_bn.json("-t", "prog", "save", str(second))["export"] is True
        assert shared_bn.run("close", str(prog)).returncode == 0
        shared_bn.load(second, copy=False)
        annotations = shared_bn.json("target", "info")["existing_annotations"]
        assert annotations["written_by_this_bridge"]["kind"] == "export", annotations
        assert "may predate this run" in (annotations["provenance_hint"] or ""), annotations


class TestRestartReopensAStaleDatabaseLoudly:
    """Review round 3 P2: `session restart` reloads a saved target through its
    row's `database_path` with `prefer_bndb` off (#753) -- an explicit database
    open, and the one load path that still reopened a database whose raw file
    had changed with `notes: []`. A private session: the restart IS the subject."""

    def test_restart_reports_the_database_whose_raw_file_changed(self, tmp_path):
        from bn.commands.admin import _SHUTDOWN_TEARDOWN_TIMEOUT

        prog = tmp_path / "prog"
        hello = Path(HELLO_BINARY).read_bytes()
        prog.write_bytes(hello)
        prog.chmod(0o755)
        sidecar = prog.with_name("prog.bndb")
        inst = None
        try:
            started = _session_start(str(prog))
            inst = started["instance_id"]
            saving = _bn("--instance", inst, "-t", str(prog), "save", "--format", "json")
            assert saving.returncode == 0, f"{saving.stdout}\n{saving.stderr}"
            saved = json.loads(saving.stdout)
            assert saved["path"] == str(sidecar), saved
            prog.write_bytes(Path(ADD_BINARY).read_bytes())

            # Restart includes old-process teardown AND a full new session start;
            # don't make those phases compete for one startup-only outer budget.
            restarted = _bn(
                "session", "restart", inst, "--format", "json",
                timeout=_SHUTDOWN_TEARDOWN_TIMEOUT + 2.0 + _SESSION_START_TIMEOUT,
            )
            assert restarted.returncode == 0, f"{restarted.stdout}\n{restarted.stderr}"
            result = json.loads(restarted.stdout)
            assert result["instance_id"] == inst, result
            assert result["pid"] != started["pid"], result
            reloaded = result["loaded"][0]
            assert reloaded["path"] == str(sidecar), reloaded
            note = next((n for n in reloaded["notes"] if "requested explicitly" in n), None)
            assert note and note.startswith(f"{sidecar} was saved for different bytes than {prog}"), (
                reloaded["notes"])
            assert reloaded["targets"][0]["source_verified"] is False, reloaded["targets"]
            rows = json.loads(_bn("--instance", inst, "target", "list", "--format", "json").stdout)
            assert rows["items"][0]["source_verified"] is False, rows
        finally:
            if inst:
                _session_stop(inst)



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
            refuse_silent_skip(
                "no creatable gap address in this fixture, so the atomic batch "
                "create/revert #308 added is unexercised; restore the gap "
                "between two adjacent analysed functions in tests/fixtures")

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
            refuse_silent_skip(
                "no caller-less function that auto-analysis declines to "
                "recreate in this fixture, so the skipped-address create #360 "
                "fixed is unexercised; restore a fixture function with no code "
                "refs that disappears after remove_user_function")
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


def _function_rows(shared_bn) -> list[dict]:
    listing = shared_bn.json("function", "list")
    return listing.get("items", listing) if isinstance(listing, dict) else listing


def _function_sizes(shared_bn) -> dict:
    """`function list` sizes, keyed by name AND by `0x`-address so a test can
    ask about a function whose name BN may re-derive after a cut."""
    sizes: dict = {}
    for f in _function_rows(shared_bn):
        sizes[f["name"]] = int(f.get("size") or 0)
        sizes[f["address"]] = int(f.get("size") or 0)
    return sizes


def _function_start_and_size(shared_bn, name: str) -> tuple[int, int]:
    row = next(f for f in _function_rows(shared_bn) if f["name"] == name)
    return int(row["address"], 16), int(row.get("size") or 0)


def _py_line(shared_bn, code: str, key: str) -> str:
    probe = shared_bn.run("py", "exec", code)
    assert probe.returncode == 0, f"{probe.stdout}\n{probe.stderr}"
    line = next((l for l in probe.stdout.splitlines() if l.startswith(key + "=")), None)
    assert line is not None, f"{key!r} missing from py exec output:\n{probe.stdout}"
    return line.split("=", 1)[1].strip()


class TestFunctionCreateInteriorGuard:
    """#927 (M-03) on real BN: a create INSIDE an existing function. Every
    mocked assumption about this path has been wrong so far -- #675 assumed
    two overlapping functions, M-03 found truncation, the #927 review found
    BOTH, depending on whether the address is an instruction boundary -- so
    the guard, its disclosure and the preview's revert are pinned against
    BN itself on `parse_record` of the synthetic parser fixture."""

    @staticmethod
    def _host(shared_bn):
        return _function_start_and_size(shared_bn, "parse_record")

    def test_boundary_create_is_refused_previewed_and_truncates_on_request(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        start, size = self._host(shared_bn)
        inside = hex(start + 4)      # past `endbr64`: an instruction boundary

        refused = shared_bn.run("function", "create", inside)
        assert refused.returncode == 3, f"{refused.stdout}\n{refused.stderr}"
        assert "parse_record" in refused.stderr and hex(start) in refused.stderr
        assert "+0x4" in refused.stderr and "instruction boundary" in refused.stderr
        assert "truncates the host" in refused.stderr
        assert _function_sizes(shared_bn)["parse_record"] == size

        preview = shared_bn.run("function", "create", inside, "--allow-overlap", "--preview")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        first, second = preview.stdout.splitlines()[:2]
        assert first.startswith("mutation: preview") and "overlaps=1" in first
        assert second == f"overlapping_functions: parse_record @ {hex(start)} (+0x4, truncated)"
        assert "dirty_after=False" in first
        assert _function_sizes(shared_bn)["parse_record"] == size, "the preview did not restore the host"

        live = shared_bn.run("function", "create", inside, "--allow-overlap", "--format", "json")
        assert live.returncode == 0, f"{live.stdout}\n{live.stderr}"
        payload = json.loads(live.stdout)
        row = payload["results"][0]
        assert row["status"] == "verified"
        assert row["overlaps"] == [{"name": "parse_record", "address": hex(start), "offset": "0x4",
                                    "instruction_boundary": True, "truncated": True}]
        assert "was truncated at" in row["note"]
        # The truncated host is in the audit, not only the created function.
        assert {"address": hex(start), "before_name": "parse_record",
                "after_name": "parse_record", "changed": True} in payload["affected_functions"]
        sizes = _function_sizes(shared_bn)
        assert sizes["parse_record"] == 4, sizes
        assert sizes[hex(start + 4)] == size - 4, sizes

    def test_mid_instruction_create_reports_the_host_intact(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        start, size = self._host(shared_bn)
        inside = hex(start + 6)      # inside `mov rbp, rsp`

        refused = shared_bn.run("function", "create", inside)
        assert refused.returncode == 3
        assert "mid-instruction" in refused.stderr and "leaves the host intact" in refused.stderr

        live = shared_bn.run("function", "create", inside, "--allow-overlap")
        assert live.returncode == 0, f"{live.stdout}\n{live.stderr}"
        assert f"overlapping_functions: parse_record @ {hex(start)} (+0x6, intact)" in live.stdout
        sizes = _function_sizes(shared_bn)
        assert sizes["parse_record"] == size, "BN left the host intact; the status must not claim a cut"
        assert hex(start + 6) in sizes


def _annotate_host(shared_bn, start: int) -> str:
    """Give the function at *start* every kind of user annotation a split
    could lose -- name, prototype, doc comment, address comment, function
    and address tags, a renamed+retyped local -- and return the state
    snapshot `_host_state` reads, so a test can assert it came back."""
    inside = hex(start + 8)
    # The fixtures are this machine's compiler output, so the local to
    # annotate is picked from the listing, not named in advance: a 32-bit
    # stack local (a C `int`), retyped to its unsigned twin.
    locals_ = json.loads(shared_bn.run("local", "list", hex(start), "--format", "json").stdout)
    local = next(v for v in locals_["items"]
                 if not v["is_parameter"] and v["type"] == "int32_t" and v["name"].startswith("var_"))
    for args in (
        ("rename", hex(start), "review_host"),
        ("proto", "set", "review_host", "int review_host(int argc, char** argv)"),
        ("comment", "set", "--function", "review_host", "host doc"),
        ("comment", "set", inside, "addr note"),
        ("tag", "add", "--function", "review_host", "--type", "Bookmarks", "--data", "fn tag"),
        ("tag", "add", inside, "--type", "Bookmarks", "--data", "addr tag"),
        ("local", "rename", "review_host", local["name"], "review_local"),
        ("local", "retype", "review_host", "review_local", "uint32_t"),
    ):
        res = shared_bn.run(*args)
        assert res.returncode == 0, f"{args}\n{res.stdout}\n{res.stderr}"
    return _host_state(shared_bn, start)


def _host_state(shared_bn, start: int) -> str:
    return _py_line(shared_bn, (
        f"f = bv.get_function_at({start:#x})\n"
        f"inside = {start + 8:#x}\n"
        "state = (f.name, f.symbol.auto, str(f.type), f.has_user_type, f.comment,\n"
        "         bv.get_comment_at(inside),\n"
        "         sorted(t.type.name + ':' + t.data for t in f.get_function_tags()),\n"
        "         sorted(t.type.name + ':' + t.data for t in f.get_tags_at(inside)),\n"
        "         sorted((v.name, str(v.type), f.is_var_user_defined(v)) for v in f.vars),\n"
        "         [(hex(r.start), hex(r.end)) for r in f.address_ranges])\n"
        "print('STATE=' + repr(state))\n"), "STATE")


class TestFunctionCreateAnnotatedHost:
    """#927 round 3 on real BN: what a split host's revert must put back, and
    WHO puts it back. The repair owns only what analysis destroys and the
    undo journal cannot restore (ranges, existence, auto symbols); every
    user annotation is the journal's. The standalone preview proves the
    journal alone preserves them; the batch orderings prove the repair
    never writes a sibling's since-reverted rename back over the undo."""

    def test_standalone_preview_keeps_every_user_annotation(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        start, size = _function_start_and_size(shared_bn, "main")
        before = _annotate_host(shared_bn, start)
        assert "'review_host', False" in before and "'review_local', 'uint32_t', True" in before

        preview = shared_bn.run("function", "create", hex(start + 4), "--allow-overlap", "--preview")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        assert "dirty_after=False" in preview.stdout
        assert f"overlapping_functions: review_host @ {hex(start)} (+0x4, truncated)" in preview.stdout
        assert _host_state(shared_bn, start) == before

    def test_batch_preview_keeps_the_hosts_original_name_after_a_sibling_rename(self, shared_bn, tmp_path):
        """The reviewer's exact repro: `[rename host, create inside it]
        --preview` exited 0 with `rolled_back: true` and the host still
        carrying the preview-only name."""
        shared_bn.load(HELLO_BINARY)
        start, size = _function_start_and_size(shared_bn, "main")
        before = _annotate_host(shared_bn, start)
        mf = tmp_path / "rename_then_split.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": hex(start), "new_name": "preview_only_name"},
            {"op": "function_create", "address": hex(start + 4), "allow_overlap": True}]}))

        preview = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json", "--summary")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        summary = json.loads(preview.stdout)
        assert summary["success"] is True and summary["rolled_back"] is True, summary
        assert summary["dirty_after"] is False
        assert summary["overlapping_functions"][0]["name"] == "preview_only_name"
        after = _host_state(shared_bn, start)
        assert after == before, f"the host is not back as it was:\n{before}\n{after}"
        assert _function_sizes(shared_bn)["review_host"] == size

    def test_failed_live_batch_really_reverts_the_rename_before_the_split(self, shared_bn, tmp_path):
        """A later op fails, the batch rolls back: the rename row says
        `reverted` and the view agrees -- original name, full ranges, and
        no function at the split address."""
        shared_bn.load(HELLO_BINARY)
        start, size = _function_start_and_size(shared_bn, "main")
        before = _annotate_host(shared_bn, start)
        mf = tmp_path / "rename_split_fail.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": hex(start), "new_name": "preview_only_name"},
            {"op": "function_create", "address": hex(start + 4), "allow_overlap": True},
            {"op": "function_create", "address": "0x10"}]}))

        res = shared_bn.run("batch", "apply", str(mf), "--format", "json")
        assert res.returncode == 3, f"{res.stdout}\n{res.stderr}"
        payload = json.loads(res.stdout)
        assert payload["rolled_back"] is True and payload["success"] is False, payload
        assert [r["status"] for r in payload["results"]] == [
            "reverted", "reverted", "invalid_request"], payload["results"]
        assert _host_state(shared_bn, start) == before
        sizes = _function_sizes(shared_bn)
        assert sizes["review_host"] == size and hex(start + 4) not in sizes, sizes

    def test_batch_preview_restores_an_auto_name_a_thunk_cut_replaced_under_a_rename(self, shared_bn, tmp_path):
        """The analysis-owned half, in the ordering that hides it: host A of
        the shared-tail geometry is cut down to a jump thunk, which gives it
        an auto `j_*` symbol UNDER the sibling's user rename. The undo drops
        the user name and the thunk name would show; the repair restores the
        auto symbol set from its snapshot, so the baseline name is back."""
        shared_bn.load(PARSER_BINARY)
        base = int(_py_line(shared_bn, _SHARED_TAIL_SETUP, "BASE"), 16)
        before = _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP")
        mf = tmp_path / "rename_then_tail_split.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": hex(base), "new_name": "user_renamed_a"},
            {"op": "function_create", "address": hex(base + 0x12), "allow_overlap": True}]}))

        preview = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        payload = json.loads(preview.stdout)
        assert payload["success"] is True and payload["rolled_back"] is True, payload["message"]
        after = _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP")
        assert after == before, f"the hosts are not back as they were:\n{before}\n{after}"

    def test_batch_preview_rolls_back_clean_on_an_implicit_named_host(self, shared_bn, tmp_path):
        """#927 round 4: the common stripped-target shape. A host named only
        `sub_<addr>` has NO registered symbol -- `fn.symbol` synthesizes it
        -- so under the sibling's rename the listing held one user symbol,
        after the undo none, and reading `fn.symbol` as a stand-in for the
        empty listing made the implicit name an "added" auto symbol
        `undefine_auto_symbol` cannot remove: `success:false,
        rolled_back:false, dirty_after:true` over a view that was exactly
        as before. The auto set is the registered listing on both reads."""
        shared_bn.load(HELLO_BINARY)
        host = int(_py_line(shared_bn, _IMPLICIT_HOST_SETUP, "HOST"), 16)
        before = _py_line(shared_bn, _implicit_host_snapshot_code(host), "SNAP")
        assert f"'sub_{host:x}', []" in before, before
        mf = tmp_path / "rename_then_split_implicit.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": hex(host), "new_name": "temporary_name"},
            {"op": "function_create", "address": hex(host + 4), "allow_overlap": True}]}))

        preview = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json", "--summary")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        summary = json.loads(preview.stdout)
        assert summary["success"] is True and summary["rolled_back"] is True, summary
        assert summary["dirty_after"] is False, summary
        after = _py_line(shared_bn, _implicit_host_snapshot_code(host), "SNAP")
        assert after == before, f"the host is not back as it was:\n{before}\n{after}"


_IMPLICIT_HOST_SETUP = (
    # One function with no symbol of its own -- fifteen NOPs and a RET laid
    # over the fixture's `main` on the test's private copy, started four
    # bytes past the ELF symbol so BN names it `sub_<addr>` and registers
    # nothing in the symbol table.
    "base = [f for f in bv.functions if f.name == 'main'][0].start\n"
    "for f in list(bv.functions):\n"
    "    if base <= f.start < base + 0x14:\n"
    "        bv.remove_user_function(f)\n"
    "bv.update_analysis_and_wait()\n"
    "bv.write(base + 4, b'\\x90' * 15 + b'\\xc3')\n"
    "bv.update_analysis_and_wait()\n"
    "bv.create_user_function(base + 4); bv.update_analysis_and_wait()\n"
    "print('HOST=' + hex(base + 4))\n"
)


def _implicit_host_snapshot_code(host: int) -> str:
    return (
        f"f = bv.get_function_at({host:#x})\n"
        f"syms = [(s.name, s.auto) for s in bv.get_symbols({host:#x}, 1)]\n"
        "ranges = [(hex(r.start), hex(r.end)) for r in f.address_ranges] if f else None\n"
        f"inside = bv.get_function_at({host + 4:#x}) is not None\n"
        "print('SNAP=' + repr((f.name if f else None, syms, ranges, inside)))\n"
    )


_SHARED_TAIL_SETUP = (
    # Two user functions sharing a tail, one of them discontiguous -- the
    # geometry the #927 review reproduced on a Mapped view, rebuilt over the
    # bytes of the fixture's `main` on the test's private copy:
    #   A @ base   = [base, base+2) + [base+0x10, base+0x19)   jmp +0x0e ... ret
    #   B @ base+4 = [base+4, base+0x19)                        nops ... ret
    "base = [f for f in bv.functions if f.name == 'main'][0].start\n"
    "for f in list(bv.functions):\n"
    "    if base <= f.start < base + 0x19:\n"
    "        bv.remove_user_function(f)\n"
    "bv.update_analysis_and_wait()\n"
    "bv.write(base, b'\\xeb\\x0e' + b'\\x90' * 22 + b'\\xc3')\n"
    "bv.update_analysis_and_wait()\n"
    "bv.create_user_function(base); bv.update_analysis_and_wait()\n"
    "bv.create_user_function(base + 4); bv.update_analysis_and_wait()\n"
    "print('BASE=' + hex(base))\n"
)


def _shared_tail_snapshot_code(base: int) -> str:
    return (
        "rows = sorted((hex(f.start), f.name, [(hex(r.start), hex(r.end)) for r in f.address_ranges])\n"
        f"              for f in bv.functions if {base:#x} <= f.start < {base + 0x19:#x})\n"
        "print('SNAP=' + repr(rows))\n"
    )


class TestFunctionDeleteRepair966:
    def test_shared_tail_deletion_reports_real_observed_containment(self, shared_bn, tmp_path):
        cc = shutil.which("aarch64-linux-gnu-gcc")
        if cc is None:
            refuse_silent_skip("Install gcc-aarch64-linux-gnu for the shared-tail repair lane")
        source = tmp_path / "split.s"
        source.write_text(
            ".text\n.global _start, build_widget, shared_tail, other_user\n"
            ".type _start,%function\n_start:\n"
            " bl build_widget\n bl other_user\n mov x8,#93\n svc #0\n"
            ".type build_widget,%function\nbuild_widget:\n"
            " stp x29,x30,[sp,#-16]!\n mov x29,sp\n mov x19,x0\n mov x20,x1\n"
            ".type shared_tail,%function\nshared_tail:\n"
            " add x0,x19,x20\n ldp x29,x30,[sp],#16\n ret\n"
            ".type other_user,%function\nother_user:\n"
            " stp x29,x30,[sp,#-16]!\n b shared_tail\n"
        )
        binary = tmp_path / "split"
        build = subprocess.run([cc, "-nostdlib", "-static", str(source), "-o", str(binary)],
                               capture_output=True, text=True, timeout=120)
        assert build.returncode == 0, build.stderr
        shared_bn.load(binary, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        split = int(_py_line(shared_bn,
                            "print('TAIL=' + hex(bv.get_functions_by_name('shared_tail')[0].start))",
                            "TAIL"), 16)
        warning = shared_bn.run("decompile", "build_widget")
        assert warning.returncode == 0, f"{warning.stdout}\n{warning.stderr}"
        repair_command = f"bn function delete {hex(split)}"
        assert f"`{repair_command}`" in warning.stdout
        assert "no clean undo; --preview refuses" in warning.stdout

        preview = shared_bn.run("function", "delete", hex(split), "--preview", "--format", "json")
        assert preview.returncode == 3, f"{preview.stdout}\n{preview.stderr}"
        assert _py_line(shared_bn, f"print('ENTRY=' + str(bv.get_function_at({split}) is not None))", "ENTRY") == "True"

        deleted = shared_bn.run(*repair_command.split()[1:], "--format", "json")
        assert deleted.returncode == 0, f"{deleted.stdout}\n{deleted.stderr}"
        result = json.loads(deleted.stdout)
        observed = result["results"][0]["observed"]
        assert result["success"] is True and result["rolled_back"] is False, result
        assert observed["function_removed"] is True
        assert observed["function_at_entry"] is None
        assert observed["reanalysis_completed"] is True
        assert observed["clean_undo_available"] is False
        actual = json.loads(_py_line(shared_bn, (
            "import json\n"
            f"print('BODIES=' + json.dumps([{{'address': hex(f.start), 'name': f.name}} "
            f"for f in bv.get_functions_containing({split}) if f.start != {split}]))"
        ), "BODIES"))
        assert observed["bodies_containing_deleted_entry"] == actual
        assert {row["name"] for row in actual} == {"build_widget", "other_user"}

    @pytest.mark.parametrize("fault", [
        "commit-before", "commit-after", "analysis", "containing-readback", "entry-readback",
    ])
    @pytest.mark.parametrize("batch", [False, True], ids=["standalone", "singleton-batch"])
    @pytest.mark.parametrize("output_args", [
        (), ("--format", "json", "--summary"), ("--format", "json"),
    ], ids=["default", "compact-json", "full-json"])
    def test_post_removal_failure_reports_committed_deletion(
        self, shared_bn, tmp_path, fault, batch, output_args,
    ):
        shared_bn.load(PARSER_BINARY, timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        start, _ = _function_start_and_size(shared_bn, "parse_record")
        injected_error = f"delete966 injected {fault} failure"
        if batch:
            manifest = tmp_path / "delete.json"
            manifest.write_text(json.dumps({
                "ops": [{"op": "function_delete", "address": hex(start)}],
            }), encoding="utf-8")
            command = ("batch", "apply", str(manifest))
        else:
            command = ("function", "delete", hex(start))
        installed = _py_line(shared_bn, (
            "from binaryninja import BinaryView\n"
            f"BinaryView._delete966_fault = {fault!r}\n"
            f"BinaryView._delete966_error = {injected_error!r}\n"
            f"BinaryView._delete966_entry = {start}\n"
            "BinaryView._delete966_analysis_done = False\n"
            "BinaryView._delete966_pending_commit = None\n"
            "BinaryView._delete966_originals = {name: getattr(BinaryView, name) for name in "
            "('commit_undo_actions', 'update_analysis_and_wait', "
            "'get_functions_containing', 'get_function_at')}\n"
            "def fail_commit(self, state, *args, **kwargs):\n"
            "    if BinaryView._delete966_fault == 'commit-before':\n"
            "        BinaryView._delete966_pending_commit = state\n"
            "        raise RuntimeError(BinaryView._delete966_error)\n"
            "    result = BinaryView._delete966_originals['commit_undo_actions'](self, state, *args, **kwargs)\n"
            "    if BinaryView._delete966_fault == 'commit-after':\n"
            "        raise RuntimeError(BinaryView._delete966_error)\n"
            "    return result\n"
            "def fail_analysis(self, *args, **kwargs):\n"
            "    if BinaryView._delete966_fault == 'analysis':\n"
            "        raise RuntimeError(BinaryView._delete966_error)\n"
            "    result = BinaryView._delete966_originals['update_analysis_and_wait'](self, *args, **kwargs)\n"
            "    BinaryView._delete966_analysis_done = True\n"
            "    return result\n"
            "def fail_containing(self, address, *args, **kwargs):\n"
            "    if (BinaryView._delete966_fault == 'containing-readback' and\n"
            "            BinaryView._delete966_analysis_done and address == BinaryView._delete966_entry):\n"
            "        raise RuntimeError(BinaryView._delete966_error)\n"
            "    return BinaryView._delete966_originals['get_functions_containing'](self, address, *args, **kwargs)\n"
            "def fail_entry(self, address, *args, **kwargs):\n"
            "    if (BinaryView._delete966_fault == 'entry-readback' and\n"
            "            BinaryView._delete966_analysis_done and address == BinaryView._delete966_entry):\n"
            "        raise RuntimeError(BinaryView._delete966_error)\n"
            "    return BinaryView._delete966_originals['get_function_at'](self, address, *args, **kwargs)\n"
            "BinaryView.commit_undo_actions = fail_commit\n"
            "BinaryView.update_analysis_and_wait = fail_analysis\n"
            "BinaryView.get_functions_containing = fail_containing\n"
            "BinaryView.get_function_at = fail_entry\n"
            "print('INSTALLED=True')\n"
        ), "INSTALLED")
        try:
            assert installed == "True"
            deleted = shared_bn.run(*command, *output_args)
        finally:
            restored = _py_line(shared_bn, (
                "from binaryninja import BinaryView\n"
                "for name, original in BinaryView._delete966_originals.items():\n"
                "    setattr(BinaryView, name, original)\n"
                "print('RESTORED=True')\n"
            ), "RESTORED")
            try:
                assert restored == "True"
                gone = _py_line(shared_bn, (
                    f"print('GONE=' + str(bv.get_function_at({start:#x}) is None))\n"
                ), "GONE")
                # In particular, a journal failure must not leave the bridge
                # reporting a clean target. Read before closing a pending journal.
                target_info = shared_bn.json("target", "info")
            finally:
                cleaned = _py_line(shared_bn, (
                    "from binaryninja import BinaryView\n"
                    "try:\n"
                    "    if BinaryView._delete966_pending_commit is not None:\n"
                    "        bv.commit_undo_actions(BinaryView._delete966_pending_commit)\n"
                    "finally:\n"
                    "    for name in ('fault', 'error', 'entry', 'analysis_done', 'pending_commit', 'originals'):\n"
                    "        delattr(BinaryView, '_delete966_' + name)\n"
                    "print('CLEANED=True')\n"
                ), "CLEANED")
                assert cleaned == "True"

        assert deleted.returncode == 3, f"{deleted.stdout}\n{deleted.stderr}"
        assert gone == "True"
        assert target_info["unsaved"] is True, target_info
        assert injected_error in deleted.stdout, deleted.stdout
        if output_args:
            payload = json.loads(deleted.stdout)
            assert payload["success"] is False, payload
            assert payload["committed"] is True, payload
            assert payload["rolled_back"] is False, payload
            if "--summary" in output_args:
                assert payload["changed_count"] == 1, payload
                assert payload["dirty_after"] is True, payload
                assert payload["failed_count"] == 1, payload
                assert payload["verified_count"] == 0, payload
            else:
                assert len(payload["results"]) == 1, payload
                row = payload["results"][0]
                assert row["status"] == "verification_failed", row
                assert row["irreversible_change"] is True, row
                assert row["observed"]["error"] == injected_error, row
                assert row["observed"]["error_stage"], row
        else:
            for field in ("mutation: committed", "changed=1", "dirty_after=True",
                          "failed=1", "verified=0"):
                assert field in deleted.stdout, deleted.stdout


class TestFunctionCreateSplitRestore:
    """#927 P1 on real BN: `--allow-overlap --preview` inside two hosts that
    share a tail reported `rolled_back: true` while both hosts stayed
    shortened and an analysis-created function owned the tail -- removing
    the created function is not restoring the hosts. The revert now
    verifies the hosts against a snapshot and repairs them (dropping the
    analysis-created function, reanalyzing the hosts, putting the auto name
    BN replaced back), on both the standalone and the batch path."""

    def _build(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        base = int(_py_line(shared_bn, _SHARED_TAIL_SETUP, "BASE"), 16)
        before = _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP")
        rows = eval(before)  # noqa: S307 - our own repr of hex strings and tuples
        # The premise: two hosts share the tail and one is discontiguous.
        assert [r[0] for r in rows] == [hex(base), hex(base + 4)], rows
        assert rows[0][2] == [(hex(base), hex(base + 2)), (hex(base + 0x10), hex(base + 0x19))], rows
        assert rows[1][2] == [(hex(base + 4), hex(base + 0x19))], rows
        return base, before

    def test_standalone_preview_restores_both_hosts(self, shared_bn):
        base, before = self._build(shared_bn)

        preview = shared_bn.run("function", "create", hex(base + 0x12), "--allow-overlap",
                                "--preview", "--format", "json")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        payload = json.loads(preview.stdout)
        assert payload["success"] is True and payload["rolled_back"] is True, payload
        # THE claim under test: `rolled_back: true` means the hosts are back.
        after = _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP")
        assert after == before, f"the hosts are not back as they were:\n{before}\n{after}"
        assert payload["results"][0]["status"] == "verified"
        assert sorted((o["address"], o["truncated"]) for o in payload["results"][0]["overlaps"]) == [
            (hex(base), True), (hex(base + 4), True)]

    def test_batch_preview_restores_both_hosts(self, shared_bn, tmp_path):
        base, before = self._build(shared_bn)
        mf = tmp_path / "split.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "function_create", "address": hex(base + 0x12), "allow_overlap": True}]}))

        preview = shared_bn.run("batch", "apply", str(mf), "--preview", "--format", "json")
        assert preview.returncode == 0, f"{preview.stdout}\n{preview.stderr}"
        payload = json.loads(preview.stdout)
        assert payload["success"] is True and payload["rolled_back"] is True, payload
        after = _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP")
        assert after == before, f"the hosts are not back as they were:\n{before}\n{after}"
        assert payload["results"][0]["status"] == "verified"
        assert {o["truncated"] for o in payload["results"][0]["overlaps"]} == {True}
        # The hosts are affected functions of the batch too, with the diff the
        # cut would make -- not only entries under the overlap list.
        cut_hosts = {a["address"] for a in payload["affected_functions"] if a["changed"]}
        assert {hex(base), hex(base + 4)} <= cut_hosts, payload["affected_functions"]

        # And the live run does what the preview said: both hosts cut.
        live = shared_bn.run("batch", "apply", str(mf), "--format", "json", "--summary")
        assert live.returncode == 0, f"{live.stdout}\n{live.stderr}"
        summary = json.loads(live.stdout)
        assert sorted(o["address"] for o in summary["overlapping_functions"]) == [
            hex(base), hex(base + 4)]
        assert _py_line(shared_bn, _shared_tail_snapshot_code(base), "SNAP") != before


class TestBatchSiblingCreateGuard:
    """#927 P2 on real BN: `[create X, create X+4]` without `allow_overlap`
    exited 0 with both verified and no overlap, because the second op's
    guard ran before the first create had analyzed ranges. The batch now
    analyzes between two creates, so the second is refused and the whole
    batch -- the first create included -- is rolled back."""

    def test_second_create_inside_the_first_is_refused(self, shared_bn, tmp_path):
        shared_bn.load(HELLO_BINARY)
        start, size = _function_start_and_size(shared_bn, "main")
        gone = _py_line(shared_bn, (
            f"f = bv.get_function_at({start:#x})\n"
            "bv.remove_user_function(f); bv.update_analysis_and_wait()\n"
            f"print('GONE=' + str(bv.get_function_at({start:#x}) is None))\n"), "GONE")
        assert gone == "True"

        mf = tmp_path / "siblings.json"
        mf.write_text(json.dumps({"ops": [
            {"op": "function_create", "address": hex(start)},
            {"op": "function_create", "address": hex(start + 4)}]}))
        res = shared_bn.run("batch", "apply", str(mf), "--format", "json")
        assert res.returncode == 3, f"{res.stdout}\n{res.stderr}"
        payload = json.loads(res.stdout)
        assert [r["status"] for r in payload["results"]] == ["reverted", "invalid_request"], payload
        assert hex(start) in payload["results"][1]["message"] and "+0x4" in payload["results"][1]["message"]
        assert payload["rolled_back"] is True
        # Rolled back: the second create never persisted, and whatever sits at
        # the first address (BN may re-derive `main` from its symbol once the
        # user suppression is lifted) is not the 4-byte stub a live cut leaves.
        sizes = _function_sizes(shared_bn)
        assert hex(start + 4) not in sizes, sizes
        assert sizes.get(hex(start), size) == size, sizes

        # With the flag on the second op the batch commits, and the second
        # row names the first create as the host it truncated.
        mf.write_text(json.dumps({"ops": [
            {"op": "function_create", "address": hex(start)},
            {"op": "function_create", "address": hex(start + 4), "allow_overlap": True}]}))
        res = shared_bn.run("batch", "apply", str(mf), "--format", "json")
        assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
        payload = json.loads(res.stdout)
        assert [r["status"] for r in payload["results"]] == ["verified", "verified"]
        assert payload["results"][1]["overlaps"][0]["address"] == hex(start)
        assert payload["results"][1]["overlaps"][0]["truncated"] is True
        sizes = _function_sizes(shared_bn)
        assert sizes[hex(start)] == 4 and sizes[hex(start + 4)] == size - 4, sizes


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
            refuse_silent_skip(
                "no function in this fixture yields an empty forward-taint "
                "verdict, so the 'NOT an all-clear' caveat #310.1 added is "
                "unexercised; restore a fixture function whose taint run "
                "reaches no sink and no frontier")


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
    """Regression for #648: the HLIL label (`authoritative` then, `recovered` since
    #865) meant "HLIL produced a list", not "the list is right" -- on an
    unknown-arity callee BN assumes every argument register is live and HLIL
    renders whatever sits in them (a neighbouring call's staging, the stack
    canary). This is the negative control that keeps the fix from
    blanket-demoting everything: `memcpy` has a bundled 3-parameter prototype, so
    its arguments keep `recovered`."""

    def test_known_prototype_callee_stays_recovered(self, shared_bn):
        shared_bn.load(PARSER_BINARY)
        ev = json.loads(shared_bn.run("evidence", "function", "parse_record",
                                      "--format", "json").stdout)
        memcpy_calls = [
            c for c in ev["calls"]
            if "memcpy" in str(((c.get("target") or {}).get("function") or {}).get("name", ""))
        ]
        assert memcpy_calls
        for c in memcpy_calls:
            assert c["argument_confidence"] == "recovered", c
            # #865: the bundled libc signature is a real, independent witness
            # and it AGREES; the import's body is not in this image, so the
            # callee-body witness has nothing to read and says so.
            library = c["arity_witnesses"]["library"]
            assert library["status"] == "agreement", c
            assert library["witness_arity"] == library["declared_arity"] == 3, c
            assert c["arity_witnesses"]["callee_body"] == {
                "status": "unavailable", "reason": "imported_callee"}, c
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
        `recovered` and flag `arity_unknown`. Needs real BN + a C compiler
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
            assert c["argument_confidence"] != "recovered", c


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
                assert call["argument_confidence"] == "recovered", call
            # #865: per-witness results on a real view. No attached library
            # names a local `arity_sink`, and its -O0 body spills all three
            # incoming argument registers, which agrees with the pinned
            # 3-parameter prototype -- whatever the call-site adjustment did.
            witnesses = call["arity_witnesses"]
            assert witnesses["library"] == {
                "status": "unavailable", "reason": "no_library_entry"}, call
            assert witnesses["callee_body"]["status"] == "agreement", call
            assert witnesses["callee_body"]["witness_arity"] == 3, call
            assert witnesses["callee_body"]["declared_arity"] == 3, call

        indirect = shared_bn.json("evidence", "function", "arity_dispatch", "-t", target)
        assert len(indirect["calls"]) == 1, indirect
        call = indirect["calls"][0]
        assert call["direct"] is False, call
        assert call["indirect_call"] is True, call
        assert call["callee_unresolved"] is True, call
        assert call["argument_confidence"] == "heuristic", call
        assert "arity_mismatch" not in call, call
        assert call["arity_witnesses"] == {
            name: {"status": "unavailable", "reason": "callee_unresolved"}
            for name in ("library", "callee_body")}, call


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

    @pytest.mark.parametrize(
        "prefix",
        ["__attribute__((packed)) ", "__attribute__((aligned(8))) ", "static "],
        ids=["packed", "nested-aligned", "static"],
    )
    def test_prefixed_drop_is_refused(self, shared_bn, prefix):
        """The prefix shapes #760's harm also reproduces on: the drop used to stay
        silent behind them. `aligned(8)` nests its parens, which a single-level prefix
        match cannot cross (review of #761)."""
        shared_bn.load(HELLO_BINARY)
        res = shared_bn.run(
            "types", "declare",
            f"{prefix}struct uint32_t {{ int shadow_x; }}; struct rv_pack_t {{ int y; }};",
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


class TestStackFrameMap:
    """`local list --frame` on a real stack-protected frame: the canary slot is
    found from the guard copy BN lifts in the prologue, and every row is one
    distinct stack offset."""

    def test_frame_marks_canary_and_measures_buffer_headroom(self, shared_bn):
        shared_bn.load(CANARY_BINARY)
        frame = shared_bn.json("local", "list", "--frame", "copy_label")
        locals_ = shared_bn.json("local", "list", "copy_label")["items"]

        slots = frame["items"]
        stack_offsets = {item["storage"] for item in locals_ if item["storage"] <= 0
                         and item["source_type"] == "StackVariableSourceType"}
        assert [slot["storage"] for slot in slots] == sorted(stack_offsets)
        assert frame["slot_count"] == len(slots)

        canary = frame["canary_storage"]
        assert canary is not None, frame
        assert [slot["storage"] for slot in slots if slot["kind"] == "canary"] == [canary]
        # The 44-byte buffer sits below the canary with at least its own size of
        # headroom; `to_canary` is measured from its offset to the canary slot.
        buffer = next(slot for slot in slots if "char" in " ".join(slot["types"]))
        assert buffer["to_canary"] == canary - buffer["storage"] >= 44

    def test_renamed_argument_does_not_fabricate_tls_canary(self, shared_bn, tmp_path):
        # Optimization keeps the argument directly in the load address, rather
        # than an O0 spill/copy that the old name-keyed rule did not recognize.
        binary = tmp_path / "context_word"
        cc = shutil.which("cc") or shutil.which("gcc")
        assert cc is not None, "a C compiler is required for the frame regression"
        build = subprocess.run(
            [cc, "-O2", "-fstack-protector-all", "-o", str(binary),
             str(FIXTURES_DIR / "src" / "canary.c")],
            capture_output=True, text=True, timeout=60,
        )
        assert build.returncode == 0, build.stderr
        shared_bn.load(binary)
        function = "read_context_word"
        before = shared_bn.json("local", "list", "--frame", function)
        assert before["canary_storage"] is None, before
        parameter = next(
            item for item in shared_bn.json("local", "list", function)["items"]
            if item["is_parameter"]
        )
        shared_bn.json("local", "rename", function, parameter["local_id"], "fsbase")
        renamed = next(
            item for item in shared_bn.json("local", "list", function)["items"]
            if item["local_id"] == parameter["local_id"]
        )
        assert renamed["name"] == "fsbase"
        after = shared_bn.json("local", "list", "--frame", function)
        assert after["canary_storage"] is None, after
        assert [(s["storage"], s["kind"], s["to_canary"]) for s in after["items"]] == [
            (s["storage"], s["kind"], s["to_canary"]) for s in before["items"]
        ]


class TestClassLensRttiGuards922:
    """#922: the class lens against a real PIE C++ build -- the `.data.rel.ro`
    adjacency of vtables and `__si_class_type_info` records, the extern vptr of
    those records, an imported base class whose typeinfo is a `.got`/`.extern`
    slot, and BN demangling one template class under two spellings -- are what
    a g++ layout produces, so the mocked fakes are checked against it here."""

    def _show(self, shared_bn, name):
        res = shared_bn.run("class", "show", name, "--format", "json")
        assert res.returncode == 0, (res.stdout, res.stderr)
        return json.loads(res.stdout)

    def test_single_inheritance_chain_has_no_phantom_secondaries_or_slots(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        # Handler: ~Handler x2, handle (pure), name -> exactly four slots. The
        # word after them is the next RTTI record's vptr.
        handler = self._show(shared_bn, "svc::Handler")
        assert handler["vtable"]["total"] == 4, handler["vtable"]
        assert "secondary_vtables" not in handler, handler.get("secondary_vtables")
        assert handler["bases"] == []
        echo = self._show(shared_bn, "svc::EchoHandler")
        assert echo["vtable"]["total"] == 4, echo["vtable"]
        assert "secondary_vtables" not in echo
        assert [b["name"] for b in echo["bases"]] == ["svc::Handler"]
        # AuditHandler adds `flush()` as a fifth slot; the count is pinned so a
        # hidden slot cannot pass the text greps below (#922 review).
        audit = self._show(shared_bn, "svc::AuditHandler")
        assert audit["vtable"]["total"] == 5, audit["vtable"]
        assert audit["vtable"]["slots"][4]["method"]["name"] == "_ZN3svc12AuditHandler5flushEv"
        text = shared_bn.run("class", "show", "svc::AuditHandler").stdout
        assert "secondary vtable" not in text
        assert "<external>" not in text

    def test_bn_vtable_width_never_hides_slots(self, shared_bn):
        # #922 review (P1): GCC emits the abstract `Codec`'s two destructor
        # entries as zero words, so BN's `svc::Codec::VTable` struct is ONE slot
        # wide (it stops at the first null) while the table has five:
        # probe, null, null, __cxa_pure_virtual, version. Bounding the walk at
        # BN's width reported `total: 1, scan_truncated: false`.
        shared_bn.load(CLASSES_BINARY)
        codec = self._show(shared_bn, "svc::Codec")
        vt = codec["vtable"]
        assert vt["bn_vtable_slots"] == 1, vt
        assert vt["total"] == 5 and vt["scan_truncated"] is False, vt
        assert [s.get("null", False) for s in vt["slots"]] == [False, True, True, False, False]
        assert vt["slots"][3]["pure_virtual"] is True
        assert vt["slots"][4]["method"]["name"] == "_ZN3svc5Codec7versionEv"
        hexcodec = self._show(shared_bn, "svc::HexCodec")
        assert hexcodec["vtable"]["total"] == 5, hexcodec["vtable"]
        assert [b["name"] for b in hexcodec["bases"]] == ["svc::Codec"]

    def test_secondary_vtable_inside_the_vtable_group_is_rtti(self, shared_bn):
        # #922 review (P3): `Pipe : Sink, Source` keeps its Source subobject's
        # vtable inside the `_ZTV` object at offset-to-top -8, with `_ZThn8_`
        # thunks for slots -- the ABI's own layout, so `rtti`, not a heuristic.
        shared_bn.load(CLASSES_BINARY)
        pipe = self._show(shared_bn, "svc::Pipe")
        assert [b["name"] for b in pipe["bases"]] == ["svc::Sink", "svc::Source"]
        assert pipe["vtable"]["total"] == 4, pipe["vtable"]
        (secondary,) = pipe["secondary_vtables"]
        assert secondary["offset_to_top"] == -8
        assert secondary["confidence"] == "rtti"
        assert secondary["confidence_reason"] == "thunk_slots"
        assert all(s["method"]["name"].startswith("_ZThn8_N3svc4Pipe") for s in secondary["slots"])
        assert "rejected_secondary_candidates" not in pipe
        text = shared_bn.run("class", "show", "svc::Pipe").stdout
        assert "secondary vtable @ " in text and "[rtti]:" in text
        assert "[heuristic]" not in text

    def test_catch_table_ref_is_rejected_with_the_observed_reason(self, shared_bn):
        # #922 review (P3): `svc::ParseError` is named in a `catch`, so the
        # PIE carries a `DW.ref._ZTI` word right after `__dso_handle`, and the
        # typeinfo backwalk lands on `__dso_handle` (a self-pointer). It is
        # disclosed as what it is -- a mapped pointer -- not as an RTTI record.
        shared_bn.load(CLASSES_BINARY)
        parse_error = self._show(shared_bn, "svc::ParseError")
        assert parse_error["vtable"]["total"] == 3, parse_error["vtable"]
        (rejected,) = parse_error["rejected_secondary_candidates"]
        assert rejected["reason"].startswith(f"word[0] {rejected['address']} is a mapped pointer (into .data)")
        assert "RTTI" not in rejected["reason"]
        assert not any("RTTI record" in n for n in parse_error["notes"])

    def test_imported_base_class_reports_no_fabricated_facts(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        # std::runtime_error is defined in libstdc++: its typeinfo here is an
        # import slot, so there is no vtable to backwalk and no base to read.
        card = self._show(shared_bn, "std::runtime_error")
        assert card["typeinfo"]["import_slot"] is True
        assert card["bases"] == []
        assert card["vtable"] is None, card["vtable"]
        assert "secondary_vtables" not in card
        assert any("typeinfo is an import slot" in n for n in card["notes"])
        # ...while the LOCAL subclass still names it as a base through its own
        # (local) `__si_class_type_info` record.
        parse_error = self._show(shared_bn, "svc::ParseError")
        assert [b["name"] for b in parse_error["bases"]] == ["std::runtime_error"]
        assert "secondary_vtables" not in parse_error

    def test_template_class_spellings_are_one_card(self, shared_bn):
        shared_bn.load(CLASSES_BINARY)
        listing = json.loads(
            shared_bn.run("class", "list", "--no-stl", "--format", "json").stdout)
        boxes = [row for row in listing["items"] if "svc::Box<" in row["name"]]
        assert len(boxes) == 1, [row["name"] for row in boxes]
        assert boxes[0]["name"] == "svc::Box<std::string>"
        assert boxes[0]["has_vtable"] is True and boxes[0]["method_count"] > 0
        assert boxes[0]["aliases"] and "basic_string" in boxes[0]["aliases"][0]
        card = self._show(shared_bn, "svc::Box<std::string>")
        assert card["vtable"]["total"] == 4, card["vtable"]
        assert [b["name"] for b in card["bases"]] == ["svc::Handler"]
        by_alias = self._show(shared_bn, card["aliases"][0])
        assert by_alias["name"] == "svc::Box<std::string>"
        # #922 review (P2): `--query` by the absorbed spelling finds the one
        # canonical card, once, through the listing and the count.
        by_query = json.loads(shared_bn.run(
            "class", "list", "--no-stl", "--query", "basic_string", "--format", "json").stdout)
        assert [row["name"] for row in by_query["items"]] == ["svc::Box<std::string>"], by_query
        assert by_query["total"] == 1
        counted = json.loads(shared_bn.run(
            "class", "list", "--no-stl", "--query", "basic_string", "--count", "--format", "json").stdout)
        assert counted["count"] == 1, counted

    def test_stripped_rtti_only_list_recovers_the_same_primary_as_show_1018(self, shared_bn):
        target = shared_bn.load(CLASSES_STRIPPED_BINARY)
        listing = shared_bn.json(
            "class", "list", "--query", "svc::EchoHandler", "-t", target)
        row, = listing["items"]
        assert row["has_vtable"] is True, row
        assert row["vtable_status"] == "recovered", row
        assert listing["vtable_recovery_budget_exhausted"] is False
        card = shared_bn.json("class", "show", "svc::EchoHandler", "-t", target)
        assert card["vtable"]["typeinfo_backwalk"] is True
        assert row["method_count"] == len(card["methods"]) == 0
        assert row["vtable_slot_count"] == card["vtable"]["total"] == 4
        text = shared_bn.run(
            "class", "list", "--query", "svc::EchoHandler", "-t", target)
        assert text.returncode == 0, (text.stdout, text.stderr)
        assert "methods=0  vtable  slots=4" in text.stdout
        assert "no-vtable" not in text.stdout
        count = shared_bn.json(
            "class", "list", "--query", "svc::EchoHandler", "--count", "-t", target)
        assert count["count"] == 1
        assert "vtable_recovery_budget_exhausted" not in count
        renamed = shared_bn.run(
            "symbol", "rename", card["vtable"]["slots"][2]["address"],
            "svc::EchoHandler::handle", "-t", target)
        assert renamed.returncode == 0, (renamed.stdout, renamed.stderr)
        named_card = shared_bn.json("class", "show", "svc::EchoHandler", "-t", target)
        named_list = shared_bn.json(
            "class", "list", "--query", "svc::EchoHandler", "-t", target)
        named_row, = named_list["items"]
        assert named_row["has_vtable"] is True, named_row
        assert named_row["method_count"] == len(named_card["methods"]) == 1
        assert named_row["vtable_slot_count"] == named_card["vtable"]["total"] == 4
        assert named_card["vtable"]["address_point"] == card["vtable"]["address_point"]
        for failure in ("refs", "raw_raise", "raw_none", "raw_short"):
            unavailable = shared_bn.json(
                "py", "exec",
                "from bn_agent_bridge import read_class, seam\n"
                f"failure = {failure!r}\n"
                "class FailedRead:\n"
                "    def __getattr__(self, name): return getattr(bv, name)\n"
                "    def get_data_refs(self, address):\n"
                "        if failure == 'refs': raise RuntimeError('synthetic reference failure')\n"
                "        return bv.get_data_refs(address)\n"
                "    def read(self, address, size):\n"
                "        if failure == 'raw_raise': raise RuntimeError('synthetic raw failure')\n"
                "        if failure == 'raw_none': return None\n"
                "        if failure == 'raw_short': return b''\n"
                "        return bv.read(address, size)\n"
                "proxy = FailedRead()\n"
                "ctx = seam.BridgeContext(None)\n"
                "ctx._resolve_view = lambda selector: proxy\n"
                "result = read_class._class_list(ctx, None, query='svc::EchoHandler')\n",
                "-t", target)["result"]
            unknown_row, = unavailable["items"]
            assert unknown_row["has_vtable"] is unknown_row["vtable_slot_count"] is None, failure
            assert unknown_row["vtable_status"] == "unavailable", failure
            assert unknown_row["method_count"] == 1
        imported = shared_bn.json(
            "class", "list", "--query", "std::runtime_error", "-t", target)
        imported_row, = imported["items"]
        assert imported_row["has_vtable"] is imported_row["vtable_slot_count"] is None
        assert imported_row["vtable_status"] == "unavailable"
        imported_card = shared_bn.json("class", "show", "std::runtime_error", "-t", target)
        assert imported_card["typeinfo"]["import_slot"] is True
        imported_text = shared_bn.run(
            "class", "list", "--query", "std::runtime_error", "-t", target).stdout
        assert "vtable=? (unavailable)" in imported_text
        assert "slots=0" not in imported_text and "no-vtable (checked)" not in imported_text

    def test_stripped_slot_array_symbols_preserve_vtable_contents_935(self, shared_bn):
        # Both builds use the same source/options except -s. Compare raw slot
        # words within each view; across views only proven external symbols use
        # their identity, since BN assigns their synthetic addresses per view.
        expected_totals = {
            "svc::Codec": 5, "svc::HexCodec": 5, "svc::Handler": 4,
            "svc::EchoHandler": 4, "svc::AuditHandler": 5, "svc::Pipe": 4,
        }
        snapshots = []
        for binary in (CLASSES_BINARY, CLASSES_STRIPPED_BINARY):
            target = shared_bn.load(binary)
            cards = {name: shared_bn.json("class", "show", name, "-t", target)
                     for name in expected_totals}
            tables = {
                name: [(int(vt["address"], 16), len(vt["slots"]))
                       for vt in [card["vtable"], *card.get("secondary_vtables", [])]]
                for name, card in cards.items()
            }
            raw = shared_bn.json(
                "py", "exec",
                "import binaryninja as bn\n"
                "ptr = bv.address_size\n"
                f"tables = {tables!r}\n"
                "result = {\n"
                "    'ptr': ptr,\n"
                "    'externals': {hex(s.address): {'raw_name': s.raw_name,\n"
                "        'namespace': str(s.namespace)}\n"
                "        for s in bv.get_symbols_of_type(bn.SymbolType.ExternalSymbol)},\n"
                "    'codec_symbols': [hex(s.address) for s in bv.get_symbols()\n"
                "        if s.short_name in ('vtable for svc::Codec',\n"
                "                            '_vtable_for_svc::Codec', 'vtable_for_svc::Codec')],\n"
                "    'tables': {name: [{\n"
                "        'head': hex(head),\n"
                "        'offset_to_top': bv.read_int(head, ptr),\n"
                "        'typeinfo': hex(bv.read_int(head + ptr, ptr, False)),\n"
                "        'slots': [hex(bv.read_int(head + (2 + i) * ptr, ptr, False))\n"
                "                  for i in range(count)],\n"
                "    } for head, count in entries] for name, entries in tables.items()},\n"
                "}\n",
                "-t", target,
            )["result"]
            for name, card in cards.items():
                vt = card["vtable"]
                assert vt["total"] == expected_totals[name], card
                assert vt["scan_truncated"] is False, vt
                for layout, words in zip(
                    [vt, *card.get("secondary_vtables", [])], raw["tables"][name], strict=True,
                ):
                    assert words["typeinfo"] == card["typeinfo"]["address"]
                    assert [s["address"] for s in layout["slots"]] == words["slots"]
                assert raw["tables"][name][0]["offset_to_top"] == 0
                if name != "svc::Pipe":
                    assert "secondary_vtables" not in card, card
            codec = cards["svc::Codec"]["vtable"]
            assert [s.get("null", False) for s in codec["slots"]] == [
                False, True, True, False, False]
            assert codec["slots"][3]["pure_virtual"] is True
            assert "typeinfo_backwalk" not in codec
            # The supplied stripped repro puts this symbol on the FIRST SLOT;
            # the class card must still name the preceding object head.
            symbol_address = int(codec["address"], 16)
            if binary == CLASSES_STRIPPED_BINARY:
                symbol_address += 2 * raw["ptr"]
            assert hex(symbol_address) in raw["codec_symbols"], raw
            (secondary,) = cards["svc::Pipe"]["secondary_vtables"]
            assert secondary["offset_to_top"] == -8
            assert secondary["typeinfo_backwalk"] is True
            assert secondary["confidence"] == "rtti"
            assert secondary["confidence_reason"] in ("thunk_slots", "bn_vtable_group")
            assert raw["tables"]["svc::Pipe"][1]["offset_to_top"] == -8
            imported = shared_bn.json("class", "show", "std::runtime_error", "-t", target)
            assert imported["typeinfo"]["import_slot"] is True
            assert imported["bases"] == [] and imported["vtable"] is None
            assert "secondary_vtables" not in imported
            for name, pure_index in (("svc::Codec", 3), ("svc::Handler", 2)):
                address = raw["tables"][name][0]["slots"][pure_index]
                assert raw["externals"][address]["raw_name"] == "__cxa_pure_virtual"
            snapshots.append({
                name: [{**words, "slots": [
                    raw["externals"].get(address, address) for address in words["slots"]
                ]} for words in entries]
                for name, entries in raw["tables"].items()
            })
        assert snapshots[0] == snapshots[1]

    @pytest.mark.parametrize("stripped", [False, True])
    def test_virtual_base_construction_tables_are_not_promoted_945(self, shared_bn, tmp_path, stripped):
        source = tmp_path / "construction.cpp"
        source.write_text("""
struct V { long v = 1; virtual ~V() {} virtual int vf() { return 1; } };
struct B : virtual V {
    long b = 2; virtual int bf() { return 2; } int vf() override { return 3; }
};
struct C : virtual V { long c = 3; virtual int cf() { return 4; } };
struct D : B, C {
    long d = 4; int bf() override { return 5; } int cf() override { return 6; }
};
int main(int argc, char **) {
    D d; B b; B *p = argc > 1 ? static_cast<B *>(&d) : &b;
    return p->bf();
}
""")
        binary = tmp_path / "construction"
        build = subprocess.run(
            ["c++", "-O0", "-fno-stack-protector", *(["-s"] if stripped else []),
             "-o", str(binary), str(source)], capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr
        target = shared_bn.load(binary)
        # Check the actual native shape, not a fake population: B has both
        # complete and construction address points; C is only constructed in D.
        candidates = shared_bn.json(
            "py", "exec",
            "result = {name: [hex(s.address) for s in bv.get_symbols()\n"
            "    if s.short_name in ('vtable for ' + name, '_vtable_for_' + name)\n"
            "    and bv.read_int(s.address - 2 * bv.address_size, bv.address_size) == 0]\n"
            "    for name in ('B', 'C')}\n",
            "-t", target,
        )["result"]
        assert len(candidates["B"]) >= 2, candidates
        assert len(candidates["C"]) == 1, candidates
        for name in ("B", "C"):
            card = shared_bn.json("class", "show", name, "-t", target)
            assert card["vtable"]["address"] in candidates[name], card
            assert card["vtable"]["slots"] == [], card
            assert card["vtable"]["truncated_reason"] == "no_local_vtable_body", card

    def test_typeinfo_backwalk_excludes_only_proven_vtt_extent_945(self, shared_bn, tmp_path):
        source = tmp_path / "vtt.cpp"
        source.write_text("""
struct V { virtual int v() { return 1; } };
struct B : virtual V { virtual int b() { return 2; } };
B global_b;
int main() { return global_b.b(); }
""")
        binary = tmp_path / "vtt"
        build = subprocess.run(
            ["c++", "-O0", "-o", str(binary), str(source)],
            capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr
        target = shared_bn.load(binary)
        result = shared_bn.json(
            "py", "exec",
            "from bn_agent_bridge import read_class, seam\n"
            "ctx = seam.BridgeContext(None)\n"
            "record = read_class._build_class_registry(ctx, bv)['B']\n"
            "record['vtable'] = None\n"
            "card = read_class._enrich(ctx, bv, record)\n"
            "vtt = next(s for s in bv.get_symbols() if s.raw_name == '_ZTT1B')\n"
            "var = bv.get_data_var_at(vtt.address)\n"
            "point = int(card['vtable']['address'], 16) + 2 * bv.address_size\n"
            "import binaryninja as bn\n"
            "bv.define_user_data_var(vtt.address, bn.Type.array(\n"
            "    bn.Type.pointer(bv.arch, bn.Type.void()), 2))\n"
            "typed_record = read_class._build_class_registry(ctx, bv)['B']\n"
            "typed_record['vtable'] = None\n"
            "typed = read_class._enrich(ctx, bv, typed_record)\n"
            "result = {'card': card, 'typed': typed, 'vtt': hex(vtt.address),\n"
            "    'width': var.type.width if var else None,\n"
            "    'refs': [hex(ref) for ref in bv.get_data_refs(point)]}\n",
            "-t", target,
        )["result"]
        assert result["card"]["vtable"]["typeinfo_backwalk"] is True, result
        assert result["card"]["vtable"]["slots"], result
        assert result["vtt"] in result["refs"], result
        assert result["width"] is None, result
        assert [ref["address"] for ref in result["card"]["instances"]["stored_globals"]] == [
            ref for ref in result["refs"] if ref != result["vtt"]]
        assert any("extent is unknown" in note for note in result["card"]["notes"])
        assert result["typed"]["instances"]["stored_globals"] == [], result

    def test_untyped_vtt_preserves_native_unnamed_neighbor_945(self, shared_bn, tmp_path):
        source = tmp_path / "boundary.cpp"
        source.write_text("""
struct V { virtual int v() { return 1; } };
struct B : virtual V { virtual int b() { return 2; } };
B global_b;
struct R { long x = 42; virtual int r() { return 3; } };
R global_r;
int main() { return global_b.b() + global_r.r(); }
""")
        layout = tmp_path / "boundary.ld"
        layout.write_text(
            "SECTIONS { .boundary : { *(.data.rel.ro.local._ZTT1B) "
            "*(.data.rel.local.global_r) } } INSERT BEFORE .data;\n")
        named = tmp_path / "boundary"
        unnamed = tmp_path / "boundary-unnamed"
        build = subprocess.run(
            ["c++", "-O0", "-fdata-sections", f"-Wl,-T,{layout}",
             "-o", str(named), str(source)], capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr
        stripped = subprocess.run(
            ["objcopy", "--strip-symbol=global_r", str(named), str(unnamed)],
            capture_output=True, text=True,
        )
        assert stripped.returncode == 0, stripped.stderr
        cards = []
        for binary in (named, unnamed):
            target = shared_bn.load(binary)
            card = shared_bn.json("class", "show", "R", "-t", target)
            cards.append(card)
            raw = shared_bn.json(
                "py", "exec",
                "vtt = next(s for s in bv.get_symbols() if s.raw_name == '_ZTT1B')\n"
                "point = bv.read_int(vtt.address + 2*bv.address_size, bv.address_size, False)\n"
                "var = bv.get_data_var_at(vtt.address)\n"
                "result = {'neighbor': hex(vtt.address + 2*bv.address_size),\n"
                "    'width': var.type.width if var else None,\n"
                "    'refs': [hex(ref) for ref in bv.get_data_refs(point)]}\n",
                "-t", target,
            )["result"]
            assert raw["width"] is None, raw
            assert raw["neighbor"] in raw["refs"], raw
            assert card["instances"]["stored_globals_total"] == 1, card
            assert card["instances"]["stored_globals"][0]["address"] == raw["neighbor"], card
            assert any("extent is unknown" in note for note in card["notes"])
        assert cards[0]["vtable"] == cards[1]["vtable"]
        assert cards[0]["instances"]["stored_globals"][0]["symbol"] == "global_r"
        assert cards[1]["instances"]["stored_globals"][0]["symbol"] is None


class TestVirtualMethods998:
    """#998 on a real g++ build (`src/vcalls.cpp`, -O0): a virtual method
    reached only through its vtable slot, and the vtable's address point."""

    def _json(self, shared_bn, *args):
        res = shared_bn.run(*args, "--format", "json")
        assert res.returncode == 0, (res.stdout, res.stderr)
        return json.loads(res.stdout)

    def test_defsym_size_zero_alias_keeps_location_xrefs_fallback(self, shared_bn, tmp_path):
        import re
        import shlex

        source = tmp_path / "alias.cpp"
        source.write_text(
            "struct Short { virtual int method() { return 7; } };\n"
            "int main() { Short value; return value.method(); }\n")
        binary = tmp_path / "alias"
        build = subprocess.run(
            ["c++", "-O0", "-o", str(binary), str(source),
             "-Wl,--defsym=_ZTV5Alias=_ZTV5Short"],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        target = shared_bn.load(binary)
        symbols = shared_bn.json(
            "py", "exec",
            "result = {name: [hex(s.address) for s in bv.get_symbols_by_raw_name(name)] "
            "for name in ('_ZTV5Short', '_ZTV5Alias')}",
            "-t", target)["result"]
        assert set(symbols["_ZTV5Short"]) & set(symbols["_ZTV5Alias"]), symbols
        out = self._json(shared_bn, "callsites", "Short::method", "-t", target)
        assert out["callee_data_ref_count"] == 1, out
        assert out["callee_data_ref_vtables"] == []
        text = shared_bn.run("callsites", "Short::method", "-t", target).stdout
        assert "bn class show" not in text
        commands = re.findall(r"`(bn xrefs [^`]+)`", text)
        assert {shlex.split(command)[-1] for command in commands} == set(out["callee_data_refs"])
        for command in commands:
            result = shared_bn.run("-t", target, *shlex.split(command)[1:])
            assert result.returncode == 0, (command, result.stderr)

    def test_duplicate_anonymous_class_names_never_print_wrong_card(self, shared_bn, tmp_path):
        import re
        import shlex

        first, second, main = (tmp_path / name for name in ("first.cpp", "second.cpp", "main.cpp"))
        first.write_text(
            "namespace { struct Foo { virtual int a() { return 1; } "
            "virtual int b() { return 2; } }; }\n"
            "extern \"C\" void* make_first() { return new Foo; }\n")
        second.write_text(
            "namespace { struct Foo { virtual int c() { return 3; } "
            "virtual int d() { return 4; } virtual int e() { return 5; } }; }\n"
            "extern \"C\" void* make_second() { return new Foo; }\n")
        main.write_text(
            "extern \"C\" void* make_first(); extern \"C\" void* make_second();\n"
            "int main() { return make_first() == make_second(); }\n")
        binary = tmp_path / "locals"
        build = subprocess.run(
            ["c++", "-O0", "-o", str(binary), str(first), str(second), str(main)],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        target = shared_bn.load(binary)
        native = shared_bn.json(
            "py", "exec",
            "result = {'tables': [hex(s.address) for s in "
            "bv.get_symbols_by_raw_name('_ZTVN12_GLOBAL__N_13FooE') "
            "if s.type.name == 'DataSymbol'], "
            "'methods': [hex(f.start) for f in bv.functions if f.symbol and "
            "f.symbol.raw_name.endswith(('3Foo1aEv', '3Foo1dEv'))]}\n",
            "-t", target)["result"]
        assert len(set(native["tables"])) == len(native["methods"]) == 2, native
        for callee in native["methods"]:
            out = self._json(shared_bn, "callsites", callee, "-t", target)
            assert out["callee_data_ref_count"] == 1, out
            assert out["callee_data_ref_vtables"] == []
            text = shared_bn.run("callsites", callee, "-t", target).stdout
            assert "bn class show" not in text
            commands = re.findall(r"`(bn xrefs [^`]+)`", text)
            assert {shlex.split(command)[-1] for command in commands} == set(out["callee_data_refs"])
            for command in commands:
                result = shared_bn.run("-t", target, *shlex.split(command)[1:])
                assert result.returncode == 0, (command, result.stderr)

    def test_capped_census_never_attributes_an_overlapping_local_alias(self, shared_bn, tmp_path):
        import re
        import shlex

        header = """
.text
.globl handler
.hidden handler
.type handler,@function
handler:
 mov $7,%eax
 ret
.size handler,.-handler
.section .data.rel.ro,"aw",@progbits
.p2align 3
.globl _ZTV5Short
.type _ZTV5Short,@object
_ZTV5Short:
 .quad 0,_ZTI5Short,handler
.size _ZTV5Short,24
.local _ZTV5Alias
.type _ZTV5Alias,@object
.set _ZTV5Alias,_ZTV5Short
.size _ZTV5Alias,24
.globl _ZTI5Short
.type _ZTI5Short,@object
_ZTI5Short:
 .quad 0,_ZTS5Short
.size _ZTI5Short,16
.section .rodata
.globl _ZTS5Short
.type _ZTS5Short,@object
_ZTS5Short:
 .asciz "5Short"
.size _ZTS5Short,7
"""
        filler = r"""
.altmacro
.macro filler index
.local padding_\index
.set padding_\index,1
.endm
.set count,0
"""
        for padding in (0, 65536):
            source = tmp_path / "census.s"
            source.write_text(
                header + filler + f".rept {padding}\n"
                "filler %count\n.set count,count+1\n.endr\n"
                '.section .note.GNU-stack,"",@progbits\n')
            binary = tmp_path / f"census-{padding}.so"
            build = subprocess.run(["cc", "-shared", "-fPIC", "-o", str(binary), str(source)],
                                   capture_output=True, text=True)
            assert build.returncode == 0, build.stderr
            target = shared_bn.load(binary)
            slot = shared_bn.json(
                "py", "exec",
                "result = hex(next(s.address for s in bv.get_symbols_by_raw_name('_ZTV5Short') "
                "if s.type.name == 'DataSymbol') + 2*bv.address_size)",
                "-t", target)["result"]
            out = self._json(shared_bn, "callsites", "handler", "-t", target)
            assert out["callee_data_ref_count"] == 1, out
            assert out["callee_data_refs"] == [slot]
            assert out["callee_data_ref_vtables"] == []
            text = shared_bn.run("callsites", "handler", "-t", target).stdout
            commands = re.findall(r"`(bn [^`]+)`", text)
            assert [shlex.split(command) for command in commands] == [["bn", "xrefs", slot]]
            result = shared_bn.run("-t", target, *shlex.split(commands[0])[1:])
            assert result.returncode == 0, (result.stdout, result.stderr)

    def test_pic_vtable_hint_uses_local_definition_beside_got_alias(self, shared_bn, tmp_path):
        import re
        import shlex

        source = tmp_path / "pic.cpp"
        source.write_text(
            "struct Short { Short(); virtual ~Short(); virtual int method(); };\n"
            "Short::Short() {} Short::~Short() {} int Short::method() { return 7; }\n"
            "extern \"C\" Short* make_short() { return new Short; }\n")
        binary = tmp_path / "pic.so"
        build = subprocess.run(
            ["c++", "-O0", "-shared", "-fPIC", "-o", str(binary), str(source)],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        target = shared_bn.load(binary)
        symbols = shared_bn.json(
            "py", "exec",
            "result = [{'address': hex(s.address), 'kind': s.type.name} "
            "for s in bv.get_symbols_by_raw_name('_ZTV5Short')]",
            "-t", target)["result"]
        assert {"ImportAddressSymbol", "DataSymbol"} <= {sym["kind"] for sym in symbols}, symbols
        out = self._json(shared_bn, "callsites", "Short::method", "-t", target)
        row, = out["callee_data_ref_vtables"]
        local = next(sym for sym in symbols if sym["kind"] == "DataSymbol")
        assert row["vtable_address"] == local["address"]
        assert row["class_name"] == "Short"
        card = self._json(shared_bn, "class", "show", "Short", "-t", target)
        assert row["address_point"] == card["vtable"]["address_point"]
        text = shared_bn.run("callsites", "Short::method", "-t", target).stdout
        commands = re.findall(r"`(bn class show [^`]+)`", text)
        assert len(commands) == 1, text
        for command in commands:
            result = shared_bn.run("-t", target, *shlex.split(command)[1:])
            assert result.returncode == 0, (command, result.stderr)
            assert result.stdout.startswith("class Short ")

    def test_callsites_says_when_the_callee_is_stored_in_data(self, shared_bn):
        # `reg` is never called directly; its address sits in two named vtable
        # objects. Execute the printed class-layout hints, not callee xrefs.
        shared_bn.load(VCALLS_BINARY)
        out = self._json(shared_bn, "callsites", "vc::Dispatcher::reg")
        assert out["total"] == 0, out
        assert out["callee_data_ref_count"] == 2, out
        xrefs = self._json(shared_bn, "xrefs", "vc::Dispatcher::reg")
        assert out["callee_data_refs"] == sorted(
            r["address"] for r in xrefs["items"] if r["kind"] == "data")
        text = shared_bn.run("callsites", "vc::Dispatcher::reg").stdout
        assert "note: the callee's address is stored in 2 data location(s)" in text, text
        import re
        import shlex
        commands = re.findall(r"`(bn class show [^`]+)`", text)
        assert len(commands) == 2, text
        assert {shlex.split(command)[-1] for command in commands} == {"vc::Dispatcher", "vc::AudioDisp"}
        for command in commands:
            result = shared_bn.run(*shlex.split(command)[1:])
            assert result.returncode == 0, (command, result.stderr)
            row = next(row for row in out["callee_data_ref_vtables"]
                       if row["class_name"] == shlex.split(command)[-1])
            assert result.stdout.startswith(f"class {row['class_name']} ")
            assert f"address point {row['address_point']}" in result.stdout.splitlines()[0]
        # A function stored nowhere gets no note.
        plain = self._json(shared_bn, "callsites", "vc::pump")
        assert plain["callee_data_ref_count"] == 0
        assert "data location" not in shared_bn.run("callsites", "vc::pump").stdout

    def test_callsite_address_point_accounts_for_virtual_base_offsets(self, shared_bn, tmp_path):
        import re
        import shlex

        source = tmp_path / "virtual.cpp"
        source.write_text(
            "struct Root { virtual void f() {} virtual ~Root() {} };\n"
            "struct Leaf : virtual Root { void f() override {} };\n"
            "int main() { Leaf value; Root* root = &value; root->f(); }\n")
        binary = tmp_path / "virtual"
        build = subprocess.run(["c++", "-O0", "-o", str(binary), str(source)],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        shared_bn.load(binary)
        out = self._json(shared_bn, "callsites", "Leaf::f")
        row = next(row for row in out["callee_data_ref_vtables"] if row["class_name"] == "Leaf")
        assert row["address_point"] == row["address"]
        assert int(row["address_point"], 16) > int(row["vtable_address"], 16) + 16
        refs = self._json(shared_bn, "xrefs", row["address_point"])
        assert any(ref["kind"] == "code" for ref in refs["items"])
        text = shared_bn.run("callsites", "Leaf::f").stdout
        commands = re.findall(r"`(bn class show [^`]+)`", text)
        assert len(commands) == 1, text
        for command in commands:
            result = shared_bn.run(*shlex.split(command)[1:])
            assert result.returncode == 0, (command, result.stderr)
            assert result.stdout.startswith("class Leaf")

    def test_vtable_address_point_is_what_objects_reference(self, shared_bn):
        shared_bn.load(VCALLS_BINARY)
        card = self._json(shared_bn, "class", "show", "vc::Dispatcher")
        vt = card["vtable"]
        assert int(vt["address_point"], 16) == int(vt["address"], 16) + 16, vt
        # The ctor stores the address point, so xrefs on it finds code and on
        # the object start (what the card printed alone before) finds none.
        assert self._json(shared_bn, "xrefs", vt["address_point"])["code_ref_count"] >= 1
        assert self._json(shared_bn, "xrefs", vt["address"])["code_ref_count"] == 0
        text = shared_bn.run("class", "show", "vc::Dispatcher").stdout
        assert f"vtable @ {vt['address']}, address point {vt['address_point']}" in text

    def test_unknown_table_hints_execute_xrefs_on_the_data_location(self, shared_bn):
        import re
        import shlex

        shared_bn.load(CLASSES_STRIPPED_BINARY)
        card = self._json(shared_bn, "class", "show", "svc::EchoHandler")
        callee = card["vtable"]["slots"][0]["address"]
        out = self._json(shared_bn, "callsites", callee)
        assert out["callee_data_ref_vtables"] == []
        text = shared_bn.run("callsites", callee).stdout
        commands = re.findall(r"`(bn xrefs [^`]+)`", text)
        assert {shlex.split(command)[-1] for command in commands} == set(out["callee_data_refs"])
        assert card["vtable"]["address_point"] in out["callee_data_refs"]
        for command in commands:
            result = shared_bn.run(*shlex.split(command)[1:])
            assert result.returncode == 0, (command, result.stderr)
            assert result.stdout.startswith(f"xrefs to {shlex.split(command)[-1]} ")


class TestStructLayoutAssistance903:
    """Known-layout disclosure on the same real typed/untyped function."""

    _SOURCE = """
        #include <stdint.h>
        struct cfg { char pad[40]; uint64_t flags; uint32_t depth; };
        __attribute__((noinline)) uint64_t handle(void *c) {
            return *(uint64_t *)((char *)c + 40) + *(uint32_t *)((char *)c + 48);
        }
        __attribute__((noipa)) void sink(void) { __asm__ volatile(""); }
        /* The call keeps c in a callee-saved copy that reaches past cfg. */
        __attribute__((noinline)) uint64_t handle_alias(void *c) {
            uint64_t sum = *(uint64_t *)((char *)c + 40) + *(uint32_t *)((char *)c + 48);
            sink();
            return sum + *(uint32_t *)((char *)c + 0x80);
        }
        int main(void) {
            struct cfg c[3] = { { .flags = 7, .depth = 3 } };
            return (int)(handle(c) + handle_alias(c));
        }
    """

    def test_untyped_hint_typed_rendering_and_ambiguity(self, shared_bn, tmp_path):
        cc = shutil.which("cc") or shutil.which("gcc")
        assert cc is not None, "a C compiler is required for the issue903 regression"
        source = tmp_path / "struct_assist.c"
        source.write_text(self._SOURCE)
        binary = tmp_path / "struct_assist"
        build = subprocess.run(
            [cc, "-O2", "-fno-inline", "-fno-stack-protector", "-no-pie",
             str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=60)
        assert build.returncode == 0, f"struct fixture build failed:\n{build.stdout}\n{build.stderr}"
        shared_bn.load(binary)
        shared_bn.json("proto", "set", "handle", "uint64_t handle(void *c)")
        # A fresh ELF already has loader-provided layouts: its header can match
        # these same accesses and later make cfg ambiguous. Establish an empty
        # inventory explicitly, without blacklisting names or changing offsets.
        remaining_types = shared_bn.json(
            "py", "exec",
            "for name in list(bv.types.keys()):\n"
            "    if bv.is_type_auto_defined(name):\n"
            "        bv.undefine_type(bv.get_type_id(name))\n"
            "    else:\n"
            "        bv.undefine_user_type(name)\n"
            "bv.update_analysis_and_wait()\n"
            "result = [str(name) for name in bv.types.keys()]\n")["result"]
        assert remaining_types == [], remaining_types
        before = shared_bn.json("proto", "get", "handle")
        missing = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in missing["warnings"]), missing

        shared_bn.json(
            "types", "declare",
            "struct cfg { char pad[40]; uint64_t flags; uint32_t depth; };")
        untyped = shared_bn.json("decompile", "handle")
        hint = next((w for w in untyped["warnings"] if "struct layout candidate" in w), None)
        assert hint is not None, untyped
        for witness in ("cfg", "+0x28/8", "flags", "+0x30/4", "depth", "bn proto set"):
            assert witness in hint
        assert shared_bn.json("proto", "get", "handle") == before
        # Text prints the one-line summary; the full sentence is --explain's.
        text_result = shared_bn.run("decompile", "handle", "--explain")
        assert text_result.returncode == 0, text_result.stderr
        # --explain wraps the detail into short lines: compare word sequences.
        assert " ".join(hint.split()) in " ".join(text_result.stdout.split())

        # Accesses through a copy of the parameter must not be dropped from
        # the witness while the body visibly uses an offset cfg cannot hold.
        shared_bn.json("proto", "set", "handle_alias", "uint64_t handle_alias(void *c)")
        aliased = shared_bn.json("decompile", "handle_alias")
        assert "0x80" in aliased["text"], aliased
        assert not any("struct layout candidate" in w for w in aliased["warnings"]), aliased

        # A bitfield storage unit at a witnessed offset is potentially
        # compatible; it must block uniqueness rather than be skipped. The C
        # declaration path refuses bitfields, so build the native member.
        bit_widths = shared_bn.json(
            "py", "exec",
            "from binaryninja import StructureBuilder, Type\n"
            "s = StructureBuilder.create()\n"
            "s.insert(0, Type.array(Type.char(), 40), 'pad')\n"
            "s.insert(40, Type.int(8, False), 'flags')\n"
            "s.insert(48, Type.int(4, False), 'lo', bit_width=20)\n"
            "bv.define_user_type('bits_cfg', s)\n"
            "bv.update_analysis_and_wait()\n"
            "result = [m.bit_width for m in bv.get_type_by_name('bits_cfg').members]\n")["result"]
        assert bit_widths == [0, 0, 20], bit_widths
        blocked = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in blocked["warnings"]), blocked
        shared_bn.json(
            "py", "exec",
            "bv.undefine_user_type('bits_cfg')\n"
            "bv.update_analysis_and_wait()\n")
        assert any("struct layout candidate" in w
                   for w in shared_bn.json("decompile", "handle")["warnings"])

        # Equally concrete layouts that cfg's uniqueness must account for: a
        # base-less class, and identical structs carrying a trailing flexible
        # array or an unresolved (zero-width) named member past the accesses.
        builder = (
            "from binaryninja import NamedTypeReferenceClass, StructureBuilder, Type\n"
            "s = StructureBuilder.create()\n"
            "s.insert(0, Type.array(Type.char(), 40), 'pad')\n"
            "s.insert(40, Type.int(8, False), 'flags')\n"
            "s.insert(48, Type.int(4, False), 'depth')\n")
        competitors = {
            "K": ("ClassStructureType", None),
            "flex_cfg": ("StructStructureType",
                         builder + "s.insert(52, Type.array(Type.char(), 0), 'tail')\n"),
            "ext_cfg": ("StructStructureType", builder + (
                "s.insert(56, Type.named_type_reference(\n"
                "    NamedTypeReferenceClass.StructNamedTypeClass, 'missing_t'), 'ext')\n")),
        }
        for name, (kind, script) in competitors.items():
            if script is None:
                shared_bn.json(
                    "types", "declare",
                    f"class {name} {{ public: char pad[40]; uint64_t flags; uint32_t depth; }};")
            else:
                shared_bn.json(
                    "py", "exec",
                    script + f"bv.define_user_type({name!r}, s)\n"
                    "bv.update_analysis_and_wait()\n")
            shape = shared_bn.json(
                "py", "exec",
                f"t = bv.get_type_by_name({name!r})\n"
                "result = [t.type.name, [m.type.width for m in t.members]]\n")["result"]
            assert shape[0] == kind, (name, shape)
            if script is not None:
                assert shape[1][-1] == 0, (name, shape)
            competing = shared_bn.json("decompile", "handle")
            assert not any("struct layout candidate" in w for w in competing["warnings"]), (
                name, competing)
            if script is None:
                # A sole matching class is not suggested as `struct K *`.
                shared_bn.json(
                    "py", "exec",
                    "bv.undefine_user_type('cfg')\n"
                    "bv.update_analysis_and_wait()\n")
                sole = shared_bn.json("decompile", "handle")
                assert not any("struct layout candidate" in w for w in sole["warnings"]), sole
                shared_bn.json(
                    "types", "declare",
                    "struct cfg { char pad[40]; uint64_t flags; uint32_t depth; };")
            shared_bn.json(
                "py", "exec",
                f"bv.undefine_user_type({name!r})\n"
                "bv.update_analysis_and_wait()\n")
            restored = shared_bn.json("decompile", "handle")
            assert any("struct layout candidate" in w for w in restored["warnings"]), (
                name, restored)

        # Redeclaring an embedded type can leave the outer struct's stored
        # width stale. That unobserved tail must not drop an otherwise
        # identical competitor and so fabricate cfg's uniqueness.
        shared_bn.json(
            "types", "declare",
            "struct ext { uint32_t a; }; "
            "struct so_cfg { char pad[40]; uint64_t flags; uint32_t depth; struct ext tail; };")
        fresh = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in fresh["warnings"]), fresh
        shared_bn.json(
            "types", "declare", "struct ext { uint32_t a, b, c, d; };")
        stale = shared_bn.json(
            "py", "exec",
            "t = bv.get_type_by_name('so_cfg')\n"
            "result = [t.width, max(m.offset + m.type.width for m in t.members)]\n")["result"]
        assert stale[1] > stale[0], stale
        grown = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in grown["warnings"]), grown
        shared_bn.json(
            "py", "exec",
            "bv.undefine_user_type('so_cfg')\n"
            "bv.undefine_user_type('ext')\n"
            "bv.update_analysis_and_wait()\n")
        assert any("struct layout candidate" in w
                   for w in shared_bn.json("decompile", "handle")["warnings"])

        # Application remains an explicit mutation, never part of a read.
        shared_bn.json("proto", "set", "handle", "uint64_t handle(struct cfg *c)")
        typed = shared_bn.json("decompile", "handle")
        assert "c->flags" in typed["text"] and "c->depth" in typed["text"], typed
        assert not any("struct layout candidate" in w for w in typed["warnings"]), typed

        shared_bn.json("proto", "set", "handle", "uint64_t handle(void *c)")
        shared_bn.json(
            "types", "declare",
            "struct other_cfg { char pad[40]; uint64_t mode; uint32_t count; };")
        ambiguous = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in ambiguous["warnings"]), ambiguous

        # A typedef on a scalar member must neither hide a compatible layout
        # nor make a distinct, direct-scalar layout appear uniquely compatible.
        shared_bn.json(
            "py", "exec",
            "bv.undefine_user_type('cfg')\n"
            "bv.undefine_user_type('other_cfg')\n"
            "bv.update_analysis_and_wait()\n")
        shared_bn.json(
            "types", "declare",
            "typedef unsigned long long word_t; "
            "struct wrapped_cfg { char pad[40]; word_t flags; unsigned int depth; }; "
            "typedef struct wrapped_cfg wrapped_alias;")
        wrapped = shared_bn.json("decompile", "handle")
        hints = [w for w in wrapped["warnings"] if "struct layout candidate" in w]
        assert len(hints) == 1, wrapped
        for witness in ("struct wrapped_cfg *", "+0x28/8", "flags", "+0x30/4", "depth"):
            assert witness in hints[0]
        assert shared_bn.json("proto", "get", "handle") == before

        shared_bn.json(
            "types", "declare",
            "struct cfg { char pad[40]; unsigned long long flags; unsigned int depth; };")
        ambiguous = shared_bn.json("decompile", "handle")
        assert not any("struct layout candidate" in w for w in ambiguous["warnings"]), ambiguous
        assert shared_bn.json("proto", "get", "handle") == before

        # Crossing the resolver budget must never turn uncertainty into a
        # unique direct-layout suggestion. The 16-hop layout is a witness on
        # its own; longer chains can only block another layout's uniqueness.
        for hops in (16, 17, 64):
            shared_bn.json(
                "py", "exec",
                "bv.undefine_user_type('cfg')\n"
                "bv.undefine_user_type('wrapped_cfg')\n"
                "bv.update_analysis_and_wait()\n")
            prefix = f"word_{hops}_"
            declarations = f"typedef unsigned long long {prefix}0; " + "".join(
                f"typedef {prefix}{index - 1} {prefix}{index}; "
                for index in range(1, hops))
            shared_bn.json(
                "types", "declare",
                declarations
                + f"struct wrapped_cfg {{ char pad[40]; {prefix}{hops - 1} flags; "
                + "unsigned int depth; };")
            sole = shared_bn.json("decompile", "handle")
            hints = [w for w in sole["warnings"] if "struct layout candidate" in w]
            assert len(hints) == (1 if hops == 16 else 0), (hops, sole)
            if hints:
                assert "struct wrapped_cfg *" in hints[0]
            shared_bn.json(
                "types", "declare",
                "struct cfg { char pad[40]; unsigned long long flags; unsigned int depth; };")
            competing = shared_bn.json("decompile", "handle")
            assert not any("struct layout candidate" in w for w in competing["warnings"]), (
                hops, competing)
            assert shared_bn.json("proto", "get", "handle") == before


class TestStringsReadImportsBurndownG:
    """#946/#968/#969/#971/#972 against a real ELF: what BN's own objects hand
    the bridge (string refs, code refs, the raw parent view) is the part the
    mocked suite cannot prove."""

    def test_imports_and_target_info_name_the_libraries_969(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        info = shared_bn.json("target", "info")
        assert "libc.so.6" in info["needed_libraries"], info
        assert {"soname", "rpath", "runpath"} <= set(info), info
        rows = shared_bn.json("imports", "--limit", "500")["items"]
        by_name = {row["name"]: row["library"] for row in rows}
        # glibc versions every libc function it exports: printf@GLIBC_2.2.5.
        assert by_name.get("printf") == "libc.so.6", by_name
        assert by_name.get("strlen") == "libc.so.6", by_name

    def test_strings_with_xrefs_names_the_referencing_function_968(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        rows = shared_bn.json("strings", "--query", "greeting length", "--with-xrefs")["items"]
        assert len(rows) == 1, rows
        assert [fn["name"] for fn in rows[0]["xref_functions"]] == ["main"], rows
        assert rows[0]["xref_function_count"] == 1, rows
        # The length filter counts characters: the literal is 20 of them.
        assert shared_bn.json("strings", "--query", "greeting length", "--count",
                              "--min-length", "20", "--max-length", "20")["count"] == 1

    def test_read_positional_length_and_escaped_encoding_971_972(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        row = shared_bn.json("strings", "--query", "starting up")["items"][0]
        out = shared_bn.run("read", row["address"], "12", "--encoding", "escaped")
        assert out.returncode == 0, out.stderr
        assert out.stdout.splitlines()[0].endswith('"starting up\\0"'), out.stdout

    def test_hostile_dt_syment_does_not_kill_the_bridge_987(self, shared_bn, tmp_path):
        """#987 review blocker on real BN: DT_SYMENT is ignored by the loader,
        so a copy with DT_SYMENT = 1 << 56 still runs -- and that value, used
        as a read size, made `BinaryView.read` segfault the bridge on
        `bn imports` and `bn target info`."""
        import struct

        data = bytearray(HELLO_BINARY.read_bytes())
        phoff, = struct.unpack_from("<Q", data, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
        patched = False
        for index in range(phnum):
            p_type, _flags, p_offset, _vaddr, _paddr, p_filesz = struct.unpack_from(
                "<IIQQQQ", data, phoff + index * phentsize)
            if p_type != 2:  # PT_DYNAMIC
                continue
            for at in range(p_offset, p_offset + p_filesz, 16):
                tag, _value = struct.unpack_from("<qQ", data, at)
                if tag == 11:  # DT_SYMENT
                    struct.pack_into("<qQ", data, at, 11, 1 << 56)
                    patched = True
                    break
        assert patched, "the fixture has no DT_SYMENT to corrupt"
        hostile = tmp_path / "hello_hostile_syment"
        hostile.write_bytes(bytes(data))
        hostile.chmod(0o755)
        shared_bn.load(hostile, copy=False)

        info = shared_bn.json("target", "info")
        assert "libc.so.6" in info["needed_libraries"], info
        assert any("DT_SYMENT" in note for note in info["dynamic_section_notes"]), info
        summary = shared_bn.json("imports", "--summary")
        assert summary["total_symbols"] > 0, summary
        assert any("DT_SYMENT" in note for note in summary["dynamic_section_notes"]), summary
        # The bridge is still alive for the next command.
        assert shared_bn.run("imports", "--count").returncode == 0


# #1000 item 3 / #994 item 2: unresolved jumps out of a function (pointer tail
# calls and computed gotos), and a short literal BN renders as `&data_<addr>`.
# Built at -O2 because the shapes only exist after the compiler forms tail
# calls; AArch64 for the type library whose `setlocale(int32_t, int64_t*)`
# hides the string.
_TAIL_JUMPS_C = r"""
#include <locale.h>
typedef int (*fn_t)(int);
struct ops { long pad; fn_t cb; };
struct vtbl { fn_t f; fn_t g; };
struct obj { const struct vtbl *ops; };
fn_t gfp;
__attribute__((noinline)) int tc_param(fn_t fn, int x) { return fn(x + 1); }
__attribute__((noinline)) int tc_global(int x) { return gfp(x * 2); }
__attribute__((noinline)) int tc_field(struct ops *o, int v) { return o->cb(v + 1); }
__attribute__((noinline)) int tc_table(struct obj *o, int x) { return o->ops->g(x + 3); }
__attribute__((noinline)) int labels2(int op, int v) {
  static void *const a[] = { &&l0, &&l1 };
  static void *const b[] = { &&l1, &&l0 };
  void *const *tables[] = { a, b };
  __asm__ volatile("" : : "r"(tables) : "memory");
  goto *tables[op & 1][0];
l0: return v + 1;
l1: return v * 3;
}
__attribute__((noinline)) int labels3(int op) {
  static void *const labels[] = { &&a, &&b, &&c };
  goto *labels[op % 3];
a: return 10;
b: return 20;
c: return 30;
}
__attribute__((noinline)) int handler_a(int x) { return x + 3; }
__attribute__((noinline)) int handler_b(int x) { return x * 3; }
__attribute__((noinline)) int dispatch_call(int n) {
  fn_t fn = n ? handler_a : handler_b;
  return fn(n) + 1;
}
int idxjump(long *p);
__asm__(IDXJUMP);
__attribute__((noinline)) char *use_c_locale(void) { return setlocale(LC_ALL, "C"); }
/* Several switch cases tail-call one field pointer: on AArch64 -O2 BN shows
   one of the `br`s in two cases (#1000 review round 5, `dup.c` d1). */
__attribute__((noinline)) int d1(struct ops *o, int op, int x) {
  switch (op) {
    case 0: x += 1; break; case 1: x *= 3; break; case 2: x -= 7; break; case 3: x ^= 0x55; break;
    case 4: x <<= 2; break; case 5: x /= 3; break; case 6: x |= 9; break; case 7: x %= 11; break;
    default: return -1;
  }
  return o->cb(x);
}
static int inc(int v) { return v + 1; }
static const struct vtbl vt = { inc, inc };
int main(int argc, char **argv) {
  struct ops o = { 0, inc }; struct obj ob = { &vt }; long z = 0;
  gfp = inc;
  return tc_param(inc, argc) + tc_global(argc) + tc_field(&o, argc) + tc_table(&ob, argc)
       + labels2(argc, 2) + idxjump(&z) + !!use_c_locale() + d1(&o, argc, argc);
}
"""
# A jump through a label table indexed by a loaded value: `jmp [rax + tbl]`.
_IDXJUMP_ASM = {
    "aarch64": (
        r'".text\n.globl idxjump\n.type idxjump, %function\nidxjump:\n'
        r'  ldr x1, [x0]\n  adrp x2, tbl\n  add x2, x2, :lo12:tbl\n  ldr x3, [x2, x1]\n'
        r'  br x3\n1: mov w0, #1\n  ret\n2: mov w0, #2\n  ret\n.size idxjump, .-idxjump\n'
        r'.section .rodata\n.p2align 3\ntbl: .quad 1b, 2b\n.text\n"'),
    "x86_64": (
        r'".text\n.globl idxjump\n.type idxjump, @function\nidxjump:\n'
        r'  mov (%rdi), %rax\n  jmp *tbl(%rax)\n1: mov $1, %eax\n  ret\n2: mov $2, %eax\n'
        r'  ret\n.size idxjump, .-idxjump\n.section .rodata\n.p2align 3\n'
        r'tbl: .quad 1b, 2b\n.text\n"'),
}


@pytest.fixture(scope="session")
def tail_jump_binaries(tmp_path_factory) -> dict[str, Path]:
    workdir = tmp_path_factory.mktemp("tail-jumps")
    built: dict[str, Path] = {}
    for arch, compiler in (("aarch64", "aarch64-linux-gnu-gcc"), ("x86_64", "cc")):
        cc = shutil.which(compiler)
        if cc is None:
            refuse_silent_skip(f"{compiler} not available, so the #1000 tail-jump lane did not run")
        src = workdir / f"tail_jumps_{arch}.c"
        src.write_text(_TAIL_JUMPS_C.replace("IDXJUMP", _IDXJUMP_ASM[arch]))
        out = workdir / f"tail_jumps_{arch}"
        build = subprocess.run(
            [cc, "-O2", "-fno-stack-protector", "-no-pie", str(src), "-o", str(out)],
            capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            refuse_silent_skip(f"{compiler} failed to build the #1000 tail-jump fixture: "
                               f"{build.stderr.strip()}")
        built[arch] = out
    return built


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
class TestDecompileUnresolvedJumps1000:
    """An unresolved jump out of a function -- a pointer tail call or a
    computed goto alike -- gets exactly one neutral `jump` row and a note on
    every line that shows it, and neither ever claims which it is. The
    expected rows come from the disassembly, not from a list of fixtures:
    which ones the compiler leaves as jumps varies by gcc version. An
    indirect branch counts only when BN rendered it as an unresolved
    `jump ->` -- a jump-table branch BN resolved renders as a `switch`."""

    _INDIRECT_BRANCH = re.compile(
        r"^(0x[0-9a-f]+)\s+(?:[0-9a-f]{2} )+\s*(?:notrack\s+)?(?:br|jmp)\s+(?!0x)\S")

    @staticmethod
    def _jump_lines(text):
        return [(int(line.split()[0], 16), line) for line in text.splitlines()
                if line.startswith("0x") and "jump ->" in line]

    def test_one_neutral_row_per_indirect_branch_and_a_note_on_each_copy(
            self, shared_bn, tail_jump_binaries, arch):
        shared_bn.load(tail_jump_binaries[arch], timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        for function in ("tc_param", "tc_global", "tc_field", "tc_table", "labels2",
                         "idxjump", "d1"):
            disasm = shared_bn.run("disasm", function)
            assert disasm.returncode == 0, disasm.stderr
            branches = {int(m.group(1), 16) for line in disasm.stdout.splitlines()
                        if (m := self._INDIRECT_BRANCH.match(line))}
            result = shared_bn.json("decompile", function, "--addresses")
            jump_lines = self._jump_lines(result["text"])
            expected = branches & {address for address, _ in jump_lines}
            if function == "idxjump":
                assert len(expected) == 1, (disasm.stdout, result["text"])  # hand-written
            jumps = [row for row in result["warning_rows"] if row["code"] == "jump"]
            addresses = [int(row["address"], 16) for row in jumps]
            assert sorted(addresses) == sorted(expected), (function, jumps, disasm.stdout)
            for row in jumps:
                assert "tail call" in row["summary"] and "computed goto" in row["summary"], row
                for claim in ("virtual", "slot", "through the function pointer"):
                    assert claim not in row["summary"] and claim not in row["detail"], row
                assert row["target"] and f"`{row['target']}`" in row["detail"], row
            # Every line BN renders for one of those jumps carries the note
            # exactly once, and no other line carries it.
            for line in result["text"].splitlines():
                address = int(line.split()[0], 16) if line.startswith("0x") else None
                shows_jump = "jump ->" in line and address in expected
                assert line.count("/* bn: unresolved jump") == int(shows_jump), (function, line)


class TestDecompileRepeatedJump1000:
    """On AArch64 -O2, BN shows one of `d1`'s `br`s in two switch cases
    (#1000 review round 5): one row for it, and one note on each copy."""

    def test_a_jump_bn_repeats_is_one_row_with_a_note_on_every_copy(
            self, shared_bn, tail_jump_binaries):
        shared_bn.load(tail_jump_binaries["aarch64"], timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        result = shared_bn.json("decompile", "d1", "--addresses")
        lines = TestDecompileUnresolvedJumps1000._jump_lines(result["text"])
        copies = collections.Counter(address for address, _ in lines)
        repeated = [address for address, count in copies.items() if count > 1]
        assert repeated, ("premise: BN no longer repeats a jump in d1", result["text"])
        jumps = [int(row["address"], 16) for row in result["warning_rows"]
                 if row["code"] == "jump"]
        assert sorted(jumps) == sorted(copies), (jumps, result["text"])
        for _, line in lines:
            assert line.count("/* bn: unresolved jump") == 1, line


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_decompile_reports_native_branch_targets_and_separate_starts_1019(
        shared_bn, tail_jump_binaries, arch):
    shared_bn.load(tail_jump_binaries[arch], timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
    # Make the split deterministic across BN releases: the compiler's static
    # label table is real, while explicit function starts model the split body.
    facts = shared_bn.json("py", "exec", "--code",
        "f = bv.get_functions_by_name('labels3')[0]\n"
        "targets = sorted({b.dest_addr for b in f.indirect_branches})\n"
        "assert len(targets) == 3, f.indirect_branches\n"
        "for address in targets[1:]:\n"
        "    bv.create_user_function(address)\n"
        "bv.update_analysis_and_wait()\n"
        "result = [{'source': hex(b.source_addr), 'target': hex(b.dest_addr), "
        "'separate': bv.get_function_at(b.dest_addr) is not None} "
        "for b in f.indirect_branches]\n")["result"]
    decompiled = shared_bn.json("decompile", "labels3")
    rows = [row for row in decompiled["warning_rows"] if row["code"] == "branch"]
    actual = {(row["address"], target["address"], "function" in target)
              for row in rows for target in row["targets"]}
    expected = {(fact["source"], fact["target"], fact["separate"]) for fact in facts}
    assert actual == expected
    assert sum(separate for _, _, separate in actual) == 2
    for row in rows:
        for target in row["targets"]:
            if "function" in target:
                body = shared_bn.json("decompile", target["address"])
                assert body["function"]["address"] == target["address"]


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_values_keeps_native_call_result_separate_from_destination_947(
        shared_bn, tail_jump_binaries, arch):
    shared_bn.load(tail_jump_binaries[arch], timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
    graph = shared_bn.json("dataflow", "callgraph", "dispatch_call")
    [call] = [row for row in graph["callees"] if row["kind"] == "indirect"]
    expected = {int(row["address"], 16) for row in call["resolved"]}
    assert len(expected) == 2
    values = shared_bn.json("dataflow", "values", "dispatch_call", "--at", call["call_addr"])
    assert values["value_basis"] == "instruction"
    assert values["possible_values"]["type"] == "UndeterminedValue"
    assert set(values["call_destination"]["possible_values"]["values"]) == expected
    assert values["call_destination"]["value_basis"] == "destination_expression"
    text = shared_bn.run("dataflow", "values", "dispatch_call", "--at", call["call_addr"])
    assert text.returncode == 0, text.stderr
    assert "possible values: UndeterminedValue" in text.stdout
    assert "call destination possible values: InSetOfValues" in text.stdout


class TestShortLiteralBehindDataName994:
    def test_setlocale_literal_rendered_as_data_name_is_named(self, shared_bn, tail_jump_binaries):
        shared_bn.load(tail_jump_binaries["aarch64"], timeout=_CROSS_ARCH_ANALYSIS_TIMEOUT)
        result = shared_bn.json("decompile", "use_c_locale")
        match = re.search(r"setlocale\(6, &data_([0-9a-f]+)\)", result["text"])
        assert match, result["text"]  # the premise: BN hides the literal as data
        strings = [row for row in result["warning_rows"] if row["code"] == "string"]
        assert strings and strings[0]["strings"][0]["value"] == "C", result["warning_rows"]
        assert strings[0]["strings"][0]["address"] == hex(int(match.group(1), 16))


class TestOffsetTypeLookup1000:
    """#1000 item 4 (review round 8): a type BN keys by a multi-part qualified
    name -- every type a library or the demangler supplies -- is not found by
    `bv.get_type_by_name("std::X")`, which reads the string as ONE name part,
    so the `__offset` note called defined types missing. The lookup must
    agree with `bn types show` for a namespaced type, a missing one, and a
    templated name whose arguments contain `::` themselves."""

    _NAMES = {
        "demo::Leaf": True,
        "demo::Box<std::pair<int, demo::Leaf> >": True,
        "demo::Missing": False,
    }

    def test_type_lookup_agrees_with_types_show(self, shared_bn):
        shared_bn.load(HELLO_BINARY)
        lookup = shared_bn.json(
            "py", "exec", "--code",
            "import binaryninja as bn\n"
            "from bn_agent_bridge import read_decompile as rd\n"
            "leaf = bn.Type.structure(members=[(bn.Type.int(4), 'x')])\n"
            "bv.define_user_type(bn.QualifiedName(['demo', 'Leaf']), leaf)\n"
            "bv.define_user_type(bn.QualifiedName(['demo', 'Box<std::pair<int, demo::Leaf> >']),"
            " leaf)\n"
            f"names = {list(self._NAMES)!r}\n"
            "defined = rd._type_lookup(bv)\n"
            "result = {'naive': {n: bv.get_type_by_name(n) is not None for n in names},\n"
            "          'lookup': {n: defined(n) for n in names}}\n")["result"]
        # The premise: BN's string lookup misses the multi-part names.
        assert lookup["naive"]["demo::Leaf"] is False, lookup
        assert lookup["lookup"] == self._NAMES, lookup
        for name, exists in self._NAMES.items():
            shown = shared_bn.run("types", "show", name)
            assert (shown.returncode == 0) == exists, (name, shown.stdout, shown.stderr)


def test_unattempted_nonimporting_declaration_reports_snapshot_parse(shared_bn, tmp_path):
    shared_bn.load(ADD_BINARY)
    manifest = tmp_path / "plain-snapshot.json"
    manifest.write_text(json.dumps({"ops": [
        {"op": "set_comment", "function": "missing", "comment": "x"},
        {"op": "types_declare", "declaration": "struct PlainSnapshot { int x; };"},
    ]}))
    response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
    assert response.returncode == 3, response.stdout + response.stderr
    result = json.loads(response.stdout)
    assert result["results"][0]["status"] == "unsupported", result
    row = result["results"][1]
    assert row["status"] == "not_attempted", result
    assert row["parse_performed"] is True, result
    assert "imported_from_library" not in row, result
    assert not result.get("kept_library_imports"), result
    assert shared_bn.run("types", "show", "PlainSnapshot").returncode == 2


class TestBatchTypeTrackingPerformance1031:
    @pytest.mark.parametrize("preview", [False, True])
    def test_large_type_view_mixed_batch_stays_near_comment_only_cost(
            self, shared_bn, tmp_path, preview):
        """Exercise the reviewer's real 3000-type/300-comment CLI regression."""
        shared_bn.load(ADD_BINARY)
        setup = shared_bn.json(
            "py", "exec", "--code",
            "import binaryninja as bn\n"
            "record = bn.Type.structure(members=[(bn.Type.int(4), 'a'), "
            "(bn.Type.int(4), 'b')])\n"
            "for index in range(3000):\n"
            "    bv.define_user_type(f'Syn{index}', record)\n"
            "result = {'start': hex(bv.start), "
            "'count': sum(str(n).startswith('Syn') for n in bv.type_names)}\n")["result"]
        assert setup["count"] == 3000, setup
        start = int(setup["start"], 16)
        field = {"op": "struct_field_set", "struct_name": "Syn1", "offset": "0x0",
                 "field_name": "z", "field_type": "int32_t"}
        samples = {False: [], True: []}
        flags = ["--preview"] if preview else []
        for repeat in range(2):
            for mixed in (False, True):
                stamp = f"{repeat}-{mixed}"
                comments = [{"op": "set_comment", "address": hex(start + index),
                             "comment": f"performance note {stamp} {index}"}
                            for index in range(300)]
                operations = [*comments, field] if mixed else comments
                manifest = tmp_path / f"timing-{stamp}.json"
                manifest.write_text(json.dumps({"ops": operations}))
                begun = time.perf_counter()
                response = shared_bn.run(
                    "batch", "apply", str(manifest), *flags, "--format", "json", timeout=120)
                samples[mixed].append(time.perf_counter() - begun)
                assert response.returncode == 0, response.stdout + response.stderr
                result = json.loads(response.stdout)
                assert result["success"] is True, result
                assert len(result["results"]) == len(operations), result
                assert all(row["status"] in {"verified", "noop"} for row in result["results"]), result
                assert result["rolled_back"] is preview, result
                assert not result.get("kept_library_imports"), result
        # Best-of-two removes one-off process scheduling stalls. This is a
        # generous relative guard, not a machine-speed budget: constant scans
        # for the one parser are fine; scans for all 300 comments are not.
        base, mixed = min(samples[False]), min(samples[True])
        print(f"mixed batch preview={preview}: comments={base:.3f}s mixed={mixed:.3f}s")
        assert mixed < base * 6 + 1.0, samples
        shown = shared_bn.json("types", "show", "Syn1")
        expected_field = "int32_t a" if preview else "int32_t z"
        assert expected_field in shown["layout"], shown


class TestBatchPreviewImports1012:
    @pytest.mark.parametrize("kind", [
        "struct_field_set", "set_prototype", "local_retype", "data_retype",
    ])
    @pytest.mark.parametrize("batched", [False, True])
    def test_every_type_parser_previews_without_importing(self, shared_bn, tmp_path, kind, batched):
        """#1012 items 1-2: every type-string parser previews with
        `import_dependencies=False`: the op is refused naming the library type,
        no type id changes, and a batched sibling is reverted."""
        shared_bn.load(ADD_BINARY)
        address = shared_bn.json("function", "info", "add")["function"]["address"]
        shared_bn.json("types", "declare", "struct PreviewSlot { int a; int b; };")
        if kind == "set_prototype":
            shared_bn.json("proto", "set", "add", "int add(int a, int b)")
            op = {"op": kind, "identifier": "add", "prototype": "int add(struct passwd* p)"}
            command = ["proto", "set", "add", op["prototype"]]
        elif kind == "local_retype":
            op = {"op": kind, "function": "add", "variable": "arg1", "new_type": "struct passwd*"}
            command = ["local", "retype", "add", "arg1", op["new_type"]]
        elif kind == "data_retype":
            op = {"op": kind, "address": address, "new_type": "struct passwd*"}
            command = ["data", "retype", address, op["new_type"]]
        else:
            op = {"op": kind, "struct_name": "PreviewSlot", "offset": "0x10",
                  "field_name": "p", "field_type": "struct passwd*"}
            command = ["struct", "field", "set", "PreviewSlot", "0x10", "p", op["field_type"]]
        probe = ("import json\nprint(json.dumps(sorted((str(i), str(bv.get_type_name_by_id(i))) "
                 "for i in bv.type_container.type_ids)))")
        before = dict(json.loads(shared_bn.run("py", "exec", "--code", probe).stdout))
        if batched:
            manifest = tmp_path / "preview.json"
            manifest.write_text(json.dumps({"ops": [
                {"op": "set_comment", "function": "mul", "comment": "preview only"}, op]}))
            command = ["batch", "apply", str(manifest)]
        response = shared_bn.run(*command, "--preview", "--format", "json")
        assert response.returncode == 3, response.stdout + response.stderr
        result = json.loads(response.stdout)
        after = dict(json.loads(shared_bn.run("py", "exec", "--code", probe).stdout))
        assert after == before, result
        assert "kept_library_imports" not in result, result
        row = result["results"][-1]
        assert row["status"] == "unsupported", result
        assert "not previewed: Binary Ninja refused to parse this without importing" in row["message"], result
        assert "passwd (libc" in row["message"], result
        assert "imported_from_library" not in row, result
        if batched:
            assert result["results"][0]["status"] == "reverted", result

    def test_snapshot_parse_of_a_later_declare_imports_nothing(self, shared_bn, tmp_path):
        """The pre-apply snapshot parses a later declare; under a preview that
        parse imports nothing either, so an earlier failure keeps no types."""
        shared_bn.load(ADD_BINARY)
        probe = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", probe).stdout
        manifest = tmp_path / "late.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "set_comment", "function": "missing_function", "comment": "x"},
            {"op": "types_declare", "declaration": "struct LateImport { timespec* p; };"}]}))
        response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert response.returncode == 3, response.stdout + response.stderr
        result = json.loads(response.stdout)
        assert "kept_library_imports" not in result, result
        assert all("imported_from_library" not in row for row in result["results"]), result
        assert result["results"][1]["status"] == "not_attempted", result
        assert shared_bn.run("py", "exec", "--code", probe).stdout == before

    def test_skipped_auto_prototype_needing_a_library_type_imports_nothing(self, shared_bn, tmp_path):
        """An AUTO prototype the preview leaves out anyway (has_user_type pin)
        no longer imports while it is checked: it stays `not_previewed`, says
        it needs `passwd`, and the sibling previews at exit 0."""
        shared_bn.load(ADD_BINARY)
        probe = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", probe).stdout
        manifest = tmp_path / "auto.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "set_prototype", "identifier": "add", "prototype": "int add(struct passwd* p)"},
            {"op": "set_comment", "function": "mul", "comment": "preview only"}]}))
        response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert response.returncode == 0, response.stdout + response.stderr
        result = json.loads(response.stdout)
        assert "kept_library_imports" not in result, result
        assert result["results"][0]["status"] == "not_previewed", result
        assert "which a preview does not import; name matches: passwd" in result["results"][0]["message"], result
        assert result["results"][1]["status"] == "verified", result
        assert shared_bn.run("py", "exec", "--code", probe).stdout == before
        assert shared_bn.json("proto", "get", "add")["function"]["name"] == "add"

    def test_preflight_library_need_then_later_preflight_error_imports_nothing(self, shared_bn, tmp_path):
        shared_bn.load(ADD_BINARY)
        probe = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", probe).stdout
        manifest = tmp_path / "preflight.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "set_prototype", "identifier": "add", "prototype": "int add(struct passwd* p)"},
            {"op": "set_prototype", "identifier": "missing", "prototype": "int missing(int x)"}]}))
        response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert response.returncode == 3, response.stdout + response.stderr
        assert "Function not found: missing" in response.stdout + response.stderr
        assert "kept_library_imports" not in response.stdout, response.stdout
        assert shared_bn.run("py", "exec", "--code", probe).stdout == before

    @pytest.mark.parametrize("kind", ["function", "data"])
    @pytest.mark.parametrize("user_type", [False, True])
    def test_renamed_prototype_uses_preview_state(self, shared_bn, tmp_path, user_type, kind):
        shared_bn.load(ADD_BINARY)
        if user_type:
            shared_bn.json("proto", "set", "add", "int add(int a, int b)")
        identifier = (shared_bn.json("function", "info", "add")["function"]["address"]
                      if kind == "data" else "add")
        manifest = tmp_path / "renamed.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": identifier, "kind": kind, "new_name": "preview_sum"},
            {"op": "set_prototype", "identifier": "preview_sum",
             "prototype": "int preview_sum(long a, int b)"}]}))
        response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert response.returncode == 0, response.stdout + response.stderr
        result = json.loads(response.stdout)
        assert result["results"][0]["status"] == "verified", result
        assert result["results"][1]["status"] == ("verified" if user_type else "not_previewed"), result
        if not user_type:
            assert result["results"][1]["parse_performed"] is True, result
        assert shared_bn.json("function", "info", "add")["function"]["name"] == "add"
        assert shared_bn.run("function", "info", "preview_sum").returncode == 2

    def test_library_refusal_comes_before_any_import_1012(self, shared_bn):
        """A declaration that needs a library type and also collides with a
        builtin: the preview stops at BN's refusal to import, imports nothing,
        and names the library type."""
        shared_bn.load(ADD_BINARY)
        probe = "print(sorted(bv.type_container.type_ids))"
        before = shared_bn.run("py", "exec", "--code", probe).stdout
        response = shared_bn.run(
            "types", "declare",
            "struct ImportThenFail { rusage r; }; struct uint32_t { int x; };",
            "--preview", "--format", "json")
        assert response.returncode == 3, response.stdout + response.stderr
        result = json.loads(response.stdout)
        assert "kept_library_imports" not in result, result
        row = result["results"][0]
        assert row["status"] == "unsupported", result
        assert "rusage" in row["message"] and "Nothing was imported" in row["message"], result
        assert shared_bn.run("py", "exec", "--code", probe).stdout == before

    def test_rename_advice_runs_with_an_earlier_noop(self, shared_bn, tmp_path):
        shared_bn.load(ADD_BINARY)
        address = shared_bn.json("function", "info", "add")["function"]["address"]
        operations = [
            {"op": "rename_symbol", "identifier": "add", "new_name": "add"},
            {"op": "rename_symbol", "identifier": address, "new_name": "intermediate"},
            {"op": "rename_symbol", "identifier": "intermediate", "new_name": 'final"sum'},
        ]
        manifest = tmp_path / "rename.json"
        manifest.write_text(json.dumps({"ops": operations}))
        response = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert response.returncode == 3, response.stdout + response.stderr
        row = json.loads(response.stdout)["results"][2]
        assert row["requested"]["conflicting_op_indices"] == [1, 2], row
        message = row["message"]
        replacement = json.JSONDecoder().raw_decode(message[message.index('{"op"'):])[0]
        manifest.write_text(json.dumps({"ops": [operations[0], replacement]}))
        corrected = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json", "--summary")
        assert corrected.returncode == 0, corrected.stdout + corrected.stderr
        summary = json.loads(corrected.stdout)
        assert summary["noop_count"] == 1 and summary["verified_count"] == 1, summary
        assert shared_bn.json("function", "info", "add")["function"]["name"] == "add"


@pytest.mark.parametrize("error_location", ["included", "main"])
@pytest.mark.parametrize("filename", ["main_err.h", 'main"\\err.h'])
def test_real_declaration_included_header_error_preserves_context_1031(
        shared_bn, tmp_path, error_location, filename):
    shared_bn.load(ADD_BINARY)
    main_path = tmp_path / filename
    sub_path = tmp_path / "sub_err.h"
    sub_path.write_text(
        "typedef struct SubE { int a; nosuch_t b; } SubE;\n"
        if error_location == "included" else
        "typedef struct SubE { int a; int b; } SubE;\n",
        encoding="utf-8",
    )
    main_path.write_text(
        '#include "sub_err.h"\nstruct UsesSubE { SubE s; };\n'
        + ("struct MainError { nosuch_t value; };\n" if error_location == "main" else ""),
        encoding="utf-8",
    )

    response = shared_bn.run(
        "types", "declare", "--file", str(main_path), "--format", "json",
    )

    assert response.returncode == 3, response.stdout + response.stderr
    result = json.loads(response.stdout)
    row = result["results"][0]
    assert row["status"] == "invalid_request", result
    assert "unknown type name 'nosuch_t'" in row["message"], result
    expected_path = sub_path if error_location == "included" else main_path
    assert str(expected_path) in row["message"], result
    assert "file not found" not in row["message"], result


def test_real_declaration_library_fallback_preserves_source_context_1031(
        shared_bn, tmp_path):
    shared_bn.load(ADD_BINARY)
    main_path = tmp_path / "main_pid.h"
    main_path.write_text(
        "struct PidRecord { pid_t pid; };\nstruct UsesPid { struct PidRecord value; pid_t owner; };\n",
        encoding="utf-8",
    )

    response = shared_bn.run(
        "types", "declare", "--file", str(main_path), "--format", "json",
    )

    assert response.returncode == 0, response.stdout + response.stderr
    result = json.loads(response.stdout)
    assert result["results"][0]["status"] == "verified", result
    assert result["results"][0]["include_root"] == str(tmp_path.resolve()), result
    assert "pid" in shared_bn.json("types", "show", "PidRecord")["layout"]
    assert "value" in shared_bn.json("types", "show", "UsesPid")["layout"]
    assert "pid_t owner" in shared_bn.json("types", "show", "UsesPid")["layout"]


@pytest.mark.parametrize("summary", [False, True])
def test_importing_auto_prototype_retains_irreversible_refusal_1031(shared_bn, summary):
    shared_bn.load(ADD_BINARY)
    args = ["proto", "set", "mul", "int mul(struct passwd* p, int b)",
            "--preview", "--format", "json"]
    if summary:
        args.append("--summary")
    response = shared_bn.run(*args)
    assert response.returncode == 3, response.stdout + response.stderr
    result = json.loads(response.stdout)
    # Nothing was imported, so this is the plain AUTO-prototype refusal envelope
    # (the same shape as one that never needed a library type).
    assert result["ok"] is False and result["status"] == "unsupported", result
    assert "kept_library_imports" not in json.dumps(result), result
    message = result["error"]
    assert "has_user_type" in message and "cleanly reverted" in message, result
    assert "which a preview does not import; name matches: passwd" in message, result
    flag = shared_bn.run("py", "exec", "--code",
                         "print(bv.get_functions_by_name('mul')[0].has_user_type)")
    assert flag.returncode == 0 and flag.stdout.strip() == "False", flag.stdout + flag.stderr


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("interface", ["cli", "wire"])
def test_analysis_exception_discloses_kept_imports_without_replacing_error_1031(
        shared_bn, preview, interface):
    shared_bn.load(ADD_BINARY)
    probe = ("import json\nprint(json.dumps({str(i): str(bv.get_type_name_by_id(i)) "
             "for i in bv.type_container.type_ids}))")
    before = json.loads(shared_bn.run("py", "exec", "--code", probe).stdout)
    # The failing analysis also imports `timespec` before it raises. A preview
    # parse imports nothing any more (#1012 item 1), so the preview case gets
    # its retained import only from this analysis-side import; the live case
    # has already imported it while parsing. Either way the original exception
    # and the retained-type disclosure must both survive (#1012 review r1).
    injected = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
original = bn.BinaryView.update_analysis_and_wait
def fail_once(view, _original=original):
    bn.BinaryView.update_analysis_and_wait = _original
    library = next(lib for lib in view.type_libraries if lib.get_named_type('timespec') is not None)
    view.import_library_type('timespec', library)
    raise RuntimeError('synthetic SDK analysis failure')
fail_once._review_original = original
bn.BinaryView.update_analysis_and_wait = fail_once
""")
    assert injected.returncode == 0, injected.stdout + injected.stderr
    try:
        declaration = ("struct Exceptional1031 { int x; };" if preview
                       else "struct Exceptional1031 { timespec* p; };")
        if interface == "cli":
            args = ["types", "declare", declaration, "--format", "json"]
            if preview:
                args.append("--preview")
            response = shared_bn.run(*args)
            assert response.returncode == 2, response.stdout + response.stderr
            result = json.loads(response.stdout)
        else:
            registry = json.loads((shared_bn.cache_dir / "instances"
                                   / f"{shared_bn.instance_id}.json").read_text())
            identity = {"instance_id": registry["instance_id"], "pid": registry["pid"],
                        "token": registry["instance_token"]}
            request = {"op": "types_declare", "target": ADD_BINARY.name,
                       "_bridge_identity": identity,
                       "params": {"declaration": declaration, "preview": preview}}
            endpoint = shared_bn.cache_dir / "instances" / f"{shared_bn.instance_id}.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(30)
                connection.connect(str(endpoint))
                connection.sendall(json.dumps(request).encode() + b"\n")
                with connection.makefile("rb") as stream:
                    result = json.loads(stream.readline())
            assert "status" not in result, result
    finally:
        cleanup = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
original = getattr(bn.BinaryView.update_analysis_and_wait, '_review_original', None)
if original is not None:
    bn.BinaryView.update_analysis_and_wait = original
""")
        assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert result["ok"] is False, result
    assert "internal error: RuntimeError: synthetic SDK analysis failure" in result["error"], result
    after = json.loads(shared_bn.run("py", "exec", "--code", probe).stdout)
    imported = sorted(name for type_id, name in after.items() if type_id not in before)
    assert "timespec" in imported, (result, imported)
    assert ("kept library imports (a rollback cannot remove them): "
            + ", ".join(imported)) in result["error"], result
    if interface == "wire":
        assert result["result"]["kept_library_imports"] == imported, result
    assert shared_bn.run("types", "show", "Exceptional1031").returncode == 2


@pytest.mark.parametrize("mixed", [False, True], ids=["comment-only", "comment-and-field"])
@pytest.mark.parametrize("phase", ["write", "analysis"])
@pytest.mark.parametrize("summary", [False, True], ids=["detail", "summary"])
def test_nonparsing_import_outcome_1031(shared_bn, tmp_path, mixed, phase, summary):
    """The whole-request guard works with or without an observed operation owner."""
    shared_bn.load(ADD_BINARY)
    setup = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
bv.define_user_type('NonParsing1031', bn.Type.structure(
    members=[(bn.Type.int(4), 'original')]))
bv.set_comment_at(bv.start, 'original nonparser comment')
""")
    assert setup.returncode == 0, setup.stdout + setup.stderr
    probe = """
import json
print(json.dumps({
    'ids': {str(i): str(bv.get_type_name_by_id(i))
            for i in bv.type_container.type_ids},
    'comment': bv.get_comment_at(bv.start),
    'struct': str(bv.get_type_by_name('NonParsing1031')),
    'address': hex(bv.start),
}))
"""
    before = json.loads(shared_bn.run("py", "exec", "--code", probe).stdout)
    assert "timespec" not in before["ids"].values(), before
    original_layout = shared_bn.json("struct", "show", "NonParsing1031")["layout"]
    operations = [{"op": "set_comment", "address": before["address"],
                   "comment": "temporary nonparser comment"}]
    if mixed:
        operations.append({
            "op": "struct_field_set", "struct_name": "NonParsing1031",
            "offset": "0x0", "field_name": "changed", "field_type": "int32_t",
        })
    manifest = tmp_path / "nonparser-import.json"
    manifest.write_text(json.dumps({"ops": operations}), encoding="utf-8")
    try:
        injected = shared_bn.run("py", "exec", "--code", f"""
import binaryninja as bn
method = {'set_comment_at' if phase == 'write' else 'update_analysis_and_wait'!r}
original = getattr(bn.BinaryView, method)
def import_once(view, *args, _original=original, _method=method, **kwargs):
    setattr(bn.BinaryView, _method, _original)
    imported = view.import_library_type('timespec')
    assert imported is not None, 'synthetic fixture needs libc timespec'
    return _original(view, *args, **kwargs)
import_once._review_nonparsing_original = original
setattr(bn.BinaryView, method, import_once)
""")
        assert injected.returncode == 0, injected.stdout + injected.stderr
        response = shared_bn.run(
            "batch", "apply", str(manifest), "--preview", "--format", "json",
            *(["--summary"] if summary else []))
    finally:
        cleanup = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
for method in ('set_comment_at', 'update_analysis_and_wait'):
    original = getattr(getattr(bn.BinaryView, method), '_review_nonparsing_original', None)
    if original is not None:
        setattr(bn.BinaryView, method, original)
""")
        assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert response.returncode == 3, response.stdout + response.stderr
    result = json.loads(response.stdout)
    assert result["success"] is False, result
    assert result["preview"] is True, result
    assert result["committed"] is False, result
    after = json.loads(shared_bn.run("py", "exec", "--code", probe).stdout)
    assert {type_id: after["ids"].get(type_id) for type_id in before["ids"]} == before["ids"]
    added = {type_id: name for type_id, name in after["ids"].items()
             if type_id not in before["ids"]}
    imported = sorted(set(added.values()))
    assert "timespec" in imported, (result, added)
    assert after["ids"] == {**before["ids"], **added}
    assert result["kept_library_imports"] == imported, result
    if summary:
        assert result["ok"] is False, result
        assert result["verified_count"] == result["changed_count"] == 0, result
        if phase == "analysis":
            assert "No operation attribution is available" in result["first_error"], result
    else:
        assert result["status"] == "unsupported", result
        if phase == "write":
            owner = result["results"][0]
            assert owner["status"] == "unsupported", result
            assert sorted(owner["imported_from_library"]) == imported, result
            assert "parse_performed" not in owner, result
        else:
            assert result["unattributed_library_imports"] == imported, result
            assert "No operation attribution is available" in result["message"], result
            assert all(row["status"] == "reverted" for row in result["results"]), result
            assert all("imported_from_library" not in row for row in result["results"]), result
            for row in result["results"]:
                assert "sibling" not in row["message"].lower(), result
                assert "live-session verification" not in row["message"].lower(), result
        if mixed:
            sibling = result["results"][1]
            assert sibling["status"] == "reverted", result
            assert not sibling.get("imported_from_library"), result
    assert after["comment"] == before["comment"]
    assert after["struct"] == before["struct"]
    assert shared_bn.json("struct", "show", "NonParsing1031")["layout"] == original_layout


def test_explicit_platform_library_import_refuses_comment_preview_1031(shared_bn, tmp_path):
    """An unattached platform library can retain types without lookup provenance."""
    shared_bn.load(ADD_BINARY)
    probe = """
import json
print(json.dumps({
    'ids': {str(i): str(bv.get_type_name_by_id(i))
            for i in bv.type_container.type_ids},
    'libraries': [str(library.name) for library in bv.type_libraries],
    'lookup': bv.lookup_imported_type_library('ipc_perm') is not None,
    'comment': bv.get_comment_at(bv.start),
    'address': hex(bv.start),
}))
"""
    before_response = shared_bn.run("py", "exec", "--code", probe)
    assert before_response.returncode == 0, before_response.stdout + before_response.stderr
    before = json.loads(before_response.stdout)
    assert "ipc_perm" not in before["ids"].values(), before
    assert "SYSCALLS" not in before["libraries"], before
    assert before["lookup"] is False, before
    manifest = tmp_path / "platform-import-preview.json"
    manifest.write_text(json.dumps({"ops": [{
        "op": "set_comment", "address": before["address"],
        "comment": "temporary explicit platform import comment",
    }]}), encoding="utf-8")
    try:
        injected = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
library = next(library for library in bv.platform.type_libraries
               if library.name == 'SYSCALLS')
original = bn.BinaryView.set_comment_at
def import_once(view, *args, _original=original, _library=library, **kwargs):
    bn.BinaryView.set_comment_at = _original
    imported = view.import_library_type('ipc_perm', _library)
    assert imported is not None, 'synthetic fixture needs SYSCALLS ipc_perm'
    return _original(view, *args, **kwargs)
import_once._review_platform_original = original
bn.BinaryView.set_comment_at = import_once
""")
        assert injected.returncode == 0, injected.stdout + injected.stderr
        response = shared_bn.run(
            "batch", "apply", str(manifest), "--preview", "--format", "json")
    finally:
        cleanup = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
original = getattr(bn.BinaryView.set_comment_at, '_review_platform_original', None)
if original is not None:
    bn.BinaryView.set_comment_at = original
""")
        assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert response.returncode == 3, response.stdout + response.stderr
    result = json.loads(response.stdout)
    assert result["success"] is False, result
    assert result["preview"] is True, result
    assert result["committed"] is False, result
    assert result["status"] == "unsupported", result
    after_response = shared_bn.run("py", "exec", "--code", probe)
    assert after_response.returncode == 0, after_response.stdout + after_response.stderr
    after = json.loads(after_response.stdout)
    assert {type_id: after["ids"].get(type_id) for type_id in before["ids"]} == before["ids"]
    added = {type_id: name for type_id, name in after["ids"].items()
             if type_id not in before["ids"]}
    imported = sorted(set(added.values()))
    assert "ipc_perm" in imported, (result, added)
    assert after["ids"] == {**before["ids"], **added}
    assert result["kept_library_imports"] == imported, result
    assert result["results"][0]["status"] == "unsupported", result
    assert sorted(result["results"][0]["imported_from_library"]) == imported, result
    assert after["libraries"] == before["libraries"], after
    assert "SYSCALLS" not in after["libraries"], after
    assert after["lookup"] is False, after
    assert after["comment"] == before["comment"], after


class TestParserHeavyTrackingPerformance1031:
    """Compare real parser-heavy batches with the same build's untraced core."""

    @staticmethod
    def _sample(shared_bn, manifest, *, preview, tracked):
        # Keep the bridge, transaction, parsers, verifiers and SDK identical.
        # Only request import bookkeeping is absent from the control. Measuring
        # inside _mutation excludes CLI startup/JSON transfer from both samples.
        installed = shared_bn.run("py", "exec", "--code", f"""
import time
from bn_agent_bridge import mutation_engine as engine
original_mutation = engine._mutation
original_baseline = engine._declare_batch_type_names
def timed_mutation(ctx, selector, preview, operations, *, exclusive=None):
    started = time.perf_counter()
    try:
        if {tracked!r}:
            return original_mutation(ctx, selector, preview, operations, exclusive=exclusive)
        preview_token = engine._PREVIEW_PARSE.set(bool(preview))
        trace_token = engine._IMPORT_TRACE.set(None)
        try:
            return engine._apply_batch(ctx, selector, preview, operations, exclusive=exclusive)
        finally:
            engine._IMPORT_TRACE.reset(trace_token)
            engine._PREVIEW_PARSE.reset(preview_token)
    finally:
        timed_mutation._review_elapsed = time.perf_counter() - started
timed_mutation._review_original = original_mutation
timed_mutation._review_baseline = original_baseline
timed_mutation._review_elapsed = None
engine._mutation = timed_mutation
if not {tracked!r}:
    engine._declare_batch_type_names = lambda bv, operations: None
""")
        assert installed.returncode == 0, installed.stdout + installed.stderr
        try:
            started = time.perf_counter()
            response = shared_bn.run(
                "batch", "apply", str(manifest),
                *(["--preview"] if preview else []), "--format", "json", timeout=180)
            wall = time.perf_counter() - started
        finally:
            restored = shared_bn.json("py", "exec", "--code", """
from bn_agent_bridge import mutation_engine as engine
timed_mutation = engine._mutation
try:
    result = {'seconds': timed_mutation._review_elapsed}
finally:
    engine._mutation = timed_mutation._review_original
    engine._declare_batch_type_names = timed_mutation._review_baseline
""")
        seconds = restored["result"]["seconds"]
        assert seconds is not None and seconds > 0, restored
        return response, seconds, wall

    @pytest.mark.parametrize(("kind", "preview", "max_ratio"), [
        pytest.param("declarations", True, 4.0, id="declarations-preview"),
        pytest.param("fields", True, 2.5, id="fields-preview"),
        pytest.param("fields", False, 2.5, id="fields-live"),
    ])
    def test_parser_heavy_batch_stays_near_untraced_core(
            self, shared_bn, tmp_path, kind, preview, max_ratio):
        shared_bn.load(ADD_BINARY)
        setup = shared_bn.json("py", "exec", "--code", """
import binaryninja as bn
record = bn.Type.structure(members=[(bn.Type.int(4), 'a'), (bn.Type.int(4), 'b')])
for index in range(3000):
    bv.define_user_type(f'Syn{index}', record)
result = {'count': sum(str(name).startswith('Syn') for name in bv.type_names),
          'width': record.width,
          'members': [member.name for member in record.members]}
""")["result"]
        assert setup == {"count": 3000, "width": 8, "members": ["a", "b"]}, setup

        samples = {"control": [], "tracked": []}
        wall_samples = {"control": [], "tracked": []}
        outcomes = []
        for repeat in range(2):
            # Reverse order in the second pair to avoid always giving either
            # path the warm parser/cache or the earlier analysis scheduling.
            for tracked in ((False, True) if repeat == 0 else (True, False)):
                mode = "tracked" if tracked else "control"
                stamp = f"{mode}_{repeat}"
                if kind == "declarations":
                    names = [f"ParserTiming1031_{stamp}_{index}" for index in range(300)]
                    operations = [
                        {"op": "types_declare",
                         "declaration": f"struct {name} {{ int x; int y; }};"}
                        for name in names
                    ]
                else:
                    # Disjoint targets and fresh member names ensure neither
                    # live repeats nor the control can turn into no-op writes.
                    first = 600 * repeat + (300 if tracked else 0)
                    names = [f"Syn{first + index}" for index in range(300)]
                    operations = [
                        {"op": "struct_field_set", "struct_name": name, "offset": "0x0",
                         "field_name": f"changed_{stamp}_{index}", "field_type": "int"}
                        for index, name in enumerate(names)
                    ]
                manifest = tmp_path / f"parser-timing-{kind}-{stamp}.json"
                manifest.write_text(json.dumps({"ops": operations}))
                response, seconds, wall = self._sample(
                    shared_bn, manifest, preview=preview, tracked=tracked)
                samples[mode].append(seconds)
                wall_samples[mode].append(wall)
                assert response.returncode == 0, response.stdout + response.stderr
                result = json.loads(response.stdout)
                assert result["success"] is True, result
                assert result["committed"] is (not preview), result
                assert result["rolled_back"] is preview, result
                assert len(result["results"]) == 300, result
                assert all(row["status"] == "verified" for row in result["results"]), result
                assert len(result["affected_types"]) == 300, result
                assert not result.get("kept_library_imports"), result
                assert not any(row.get("imported_from_library")
                               for row in result["results"]), result

                # Inspect the real SDK state after every control/tracked run,
                # outside the measured interval. Preview must undo all 300
                # writes; live fields must retain all 300 fresh names.
                expected = (
                    [None] * 300 if kind == "declarations" else
                    [f"changed_{stamp}_{index}" if not preview else "a"
                     for index in range(300)]
                )
                state = shared_bn.json("py", "exec", "--code", f"""
names = {names!r}
state = []
for name in names:
    type_obj = bv.get_type_by_name(name)
    if type_obj is None:
        state.append(None)
    else:
        state.append({{'width': type_obj.width,
                      'members': [[member.name, member.offset, member.type.width]
                                  for member in type_obj.members]}})
result = state
""")["result"]
                wanted = [
                    None if field is None else
                    {"width": 8, "members": [[field, 0, 4], ["b", 4, 4]]}
                    for field in expected
                ]
                assert state == wanted, (mode, repeat, state, wanted)
                outcomes.append({
                    "success": result["success"], "committed": result["committed"],
                    "rolled_back": result["rolled_back"],
                    "statuses": dict(collections.Counter(
                        row["status"] for row in result["results"])),
                    "affected_types": len(result["affected_types"]),
                    "checked_state_count": len(state),
                })

        assert all(outcome == outcomes[0] for outcome in outcomes), outcomes
        control, tracked = min(samples["control"]), min(samples["tracked"])
        print(f"parser-heavy {kind} preview={preview}: core_samples={samples} "
              f"wall_samples={wall_samples} best_ratio={tracked / control:.3f}")
        # Best-of-two tolerates one-off scheduling stalls. Ratios are generous
        # to routine bookkeeping, but reject the review's ~6x declaration and
        # ~3x field-set regressions without a machine-specific seconds budget.
        assert tracked < control * max_ratio, (samples, wall_samples, max_ratio)


def test_live_import_from_unattached_platform_library_is_named_1012(shared_bn):
    """#1012 item 2: the live classifier knew only attached libraries and BN's
    lookup record, so an import from the platform's unattached `SYSCALLS`
    library went unreported on a committed run. An import is now recognized by
    the guid in its type id, which BN resolves to the library it names."""
    shared_bn.load(ADD_BINARY)
    address = shared_bn.run("py", "exec", "--code", "print(hex(bv.start))").stdout.strip().splitlines()[-1]
    try:
        injected = shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
library = next(library for library in bv.platform.type_libraries
               if library.name == 'SYSCALLS')
assert library not in bv.type_libraries
original = bn.BinaryView.set_comment_at
def import_once(view, *args, _original=original, _library=library, **kwargs):
    bn.BinaryView.set_comment_at = _original
    assert view.import_library_type('ipc_perm', _library) is not None
    return _original(view, *args, **kwargs)
import_once._review_platform_original = original
bn.BinaryView.set_comment_at = import_once
""")
        assert injected.returncode == 0, injected.stdout + injected.stderr
        response = shared_bn.run("comment", "set", address, "live platform import 1012", "--format", "json")
    finally:
        shared_bn.run("py", "exec", "--code", """
import binaryninja as bn
original = getattr(bn.BinaryView.set_comment_at, '_review_platform_original', None)
if original is not None:
    bn.BinaryView.set_comment_at = original
""")
    assert response.returncode == 0, response.stdout + response.stderr
    result = json.loads(response.stdout)
    assert result["committed"] is True, result
    assert "ipc_perm" in result["results"][0]["imported_from_library"], result
    assert result["library_sources"]["ipc_perm"] == "SYSCALLS", result
    assert set(result["library_sources"].values()) == {"SYSCALLS"}, result


class TestDirectPreviewRetainedTypes1031:
    """Direct mutation entry points obey the same final-ID rule as batches."""

    @staticmethod
    def _state(shared_bn):
        response = shared_bn.run("py", "exec", "--code", """
import json
print(json.dumps({
    'ids': {str(i): str(bv.get_type_name_by_id(i))
            for i in bv.type_container.type_ids},
    'functions': {hex(f.start): f.name for f in bv.functions},
}))
""")
        assert response.returncode == 0, response.stdout + response.stderr
        return json.loads(response.stdout)

    @staticmethod
    def _retained(before, after, result, names):
        assert {i: after["ids"].get(i) for i in before["ids"]} == before["ids"]
        added = {i: name for i, name in after["ids"].items() if i not in before["ids"]}
        assert sorted(set(added.values())) == sorted(names), (added, result)
        assert result["success"] is False, result
        assert result["status"] == "unsupported", result
        assert result["preview"] is True and result["committed"] is False, result
        assert result["kept_library_imports"] == sorted(names), result
        assert result["unattributed_library_imports"] == sorted(names), result

    @staticmethod
    def _load_syscall_stubs(shared_bn, tmp_path) -> int:
        """Build and load the public round-four reproducer; return the address
        of its second .text syscall stub, deliberately unreferenced and without
        a symbol after strip. The address is taken before stripping rather than
        depending on compiler layout."""
        source = tmp_path / "stubs.c"
        source.write_text("""
__attribute__((section(".text"), used, aligned(16)))
unsigned char stub_blob[] = {
  0x48,0x89,0xf8, 0xb8,0x3f,0x00,0x00,0x00, 0x0f,0x05, 0xc3,
  0x48,0x89,0xf8, 0xb8,0x63,0x00,0x00,0x00, 0x0f,0x05, 0xc3
};
int main(int c, char **v) { (void)v; return c; }
""", encoding="utf-8")
        binary = tmp_path / "stubs"
        built = subprocess.run(
            ["cc", "-O2", "-fno-pie", "-no-pie", str(source), "-o", str(binary)],
            capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        symbols = subprocess.run(["nm", "-n", str(binary)], capture_output=True, text=True)
        assert symbols.returncode == 0, symbols.stderr
        addresses = {parts[2]: int(parts[0], 16)
                     for line in symbols.stdout.splitlines()
                     if len(parts := line.split()) == 3}
        stripped = subprocess.run(["strip", str(binary)], capture_output=True, text=True)
        assert stripped.returncode == 0, stripped.stderr
        shared_bn.load(binary, copy=False)
        return addresses["stub_blob"] + 11

    def test_live_standalone_function_create_reports_its_imports_1012(self, shared_bn, tmp_path):
        """#1012 review r1: a LIVE standalone `function create` bypassed the
        request-level outcome, so the five SYSCALLS types its analysis imported
        went unreported. They are now listed, unattributed (defined during the
        reanalysis, not on the create's own thread), with their library."""
        address = self._load_syscall_stubs(shared_bn, tmp_path)
        before = self._state(shared_bn)
        assert hex(address) not in before["functions"], before
        response = shared_bn.run("function", "create", hex(address), "--format", "json")
        assert response.returncode == 0, response.stdout + response.stderr
        result = json.loads(response.stdout)
        assert result["committed"] is True, result
        after = self._state(shared_bn)
        imported = sorted(name for type_id, name in after["ids"].items()
                          if type_id not in before["ids"])
        assert imported == ["__kernel_long_t", "__kernel_ulong_t", "__u16", "__u32", "sysinfo"], imported
        reported = set(result.get("unattributed_library_imports") or [])
        for row in result["results"]:
            reported |= set(row.get("imported_from_library") or [])
        assert reported == set(imported), result
        assert result["library_sources"] == {name: "SYSCALLS" for name in imported}, result

    @pytest.mark.parametrize("syscall", [True, False], ids=["syscall-residue", "preexisting-types-clean"])
    def test_standalone_function_create_preview(self, shared_bn, tmp_path, syscall):
        address = self._load_syscall_stubs(shared_bn, tmp_path)
        if not syscall:
            # The identical fresh target is clean when its syscall types already
            # exist: pre-existing IDs must not turn a reversible preview into a failure.
            primed = shared_bn.run("py", "exec", "--code", """
library = next(x for x in bv.platform.type_libraries if x.name == 'SYSCALLS')
assert bv.import_library_type('sysinfo', library) is not None
bv.update_analysis_and_wait()
""")
            assert primed.returncode == 0, primed.stdout + primed.stderr
        before = self._state(shared_bn)
        assert hex(address) not in before["functions"], before
        response = shared_bn.run(
            "function", "create", hex(address), "--preview", "--format", "json")
        assert response.returncode == (3 if syscall else 0), response.stdout + response.stderr
        result = json.loads(response.stdout)
        after = self._state(shared_bn)
        assert hex(address) not in after["functions"], (result, after)
        assert after["functions"] == before["functions"], (result, before, after)
        assert result["rolled_back"] is True, result
        if syscall:
            self._retained(before, after, result, [
                "__kernel_long_t", "__kernel_ulong_t", "__u16", "__u32", "sysinfo"])
        else:
            assert result["success"] is True and result["committed"] is False, result
            assert result["results"][0]["status"] == "verified", result
            assert after["ids"] == before["ids"]
            assert not result.get("kept_library_imports"), result

    @pytest.mark.parametrize("retained", [True, False], ids=["retained", "clean"])
    @pytest.mark.parametrize("candidates", [True, False], ids=["rename", "no-candidates"])
    def test_go_rename_preview_guards_recovery_and_preserves_counters(
            self, shared_bn, retained, candidates):
        # No Go binary is required: replace only pcln recovery, not the actual
        # CLI/direct rename implementation, its live readback or rollback.
        shared_bn.load(ADD_BINARY)
        setup = shared_bn.run("py", "exec", "--code", """
f = next(f for f in bv.functions if f.name == 'mul')
f.name = f'sub_{f.start:x}'
""")
        assert setup.returncode == 0, setup.stdout + setup.stderr
        before = self._state(shared_bn)
        try:
            installed = shared_bn.run("py", "exec", "--code", f"""
import binaryninja as bn
from bn_agent_bridge import read_go
original = read_go._go_functions
def synthetic_recovery(ctx, selector, _original=original):
    view = ctx._resolve_view(selector)
    auto = next(f for f in view.functions if f.name == f'sub_{{f.start:x}}')
    user = next(f for f in view.functions if f.name == 'add')
    named = next(f for f in view.functions if f.name == 'main')
    if {retained!r}:
        view.define_user_type('RetainedGo1031', bn.Type.int(4))
    return {{'defined_count': 4, 'items': [
        {{'address': hex(auto.start), 'name': 'probe.Go1031' if {candidates!r} else auto.name, 'defined': True}},
        {{'address': hex(user.start), 'name': 'probe.User1031', 'defined': True}},
        {{'address': hex(named.start), 'name': named.name, 'defined': True}},
        {{'address': hex(auto.start + 1), 'name': 'probe.Interior1031', 'defined': True}},
    ]}}
synthetic_recovery._review_direct_original = original
read_go._go_functions = synthetic_recovery
""")
            assert installed.returncode == 0, installed.stdout + installed.stderr
            response = shared_bn.run("go", "rename", "--preview", "--format", "json")
        finally:
            restored = shared_bn.run("py", "exec", "--code", """
from bn_agent_bridge import read_go
original = getattr(read_go._go_functions, '_review_direct_original', None)
if original is not None:
    read_go._go_functions = original
""")
            assert restored.returncode == 0, restored.stdout + restored.stderr
        assert response.returncode == (3 if retained else 0), response.stdout + response.stderr
        result = json.loads(response.stdout)
        after = self._state(shared_bn)
        assert after["functions"] == before["functions"], (before, after, result)
        assert result["defined_count"] == 4, result
        assert result["go_renamed_candidates"] == int(candidates), result
        assert result["go_verified_count"] == int(candidates), result
        assert result["go_failed_count"] == 0 and result["go_committed_count"] == 0, result
        assert result["skipped_user_named"] == 1, result
        assert result["skipped_already_named"] == (1 if candidates else 2), result
        assert result["skipped_interior_pc"] == 1, result
        if retained:
            self._retained(before, after, result, ["RetainedGo1031"])
        else:
            assert result["success"] is True and result["committed"] is False, result
            assert after["ids"] == before["ids"]
            assert not result.get("kept_library_imports"), result

    @pytest.mark.parametrize("retained", [True, False], ids=["retained", "clean"])
    def test_class_apply_types_preview_guards_noop_planning(self, shared_bn, retained):
        shared_bn.load(CLASSES_STRIPPED_BINARY)
        first = shared_bn.json("class", "apply-types", "svc::Codec")
        assert first["success"] is True, first
        before = self._state(shared_bn)
        layout = shared_bn.json("types", "show", "svc::Codec")["layout"]
        try:
            installed = shared_bn.run("py", "exec", "--code", f"""
import binaryninja as bn
from bn_agent_bridge import mutation_engine as engine
original = engine._class_apply_types_plan
def observed_plan(ctx, selector, name, _original=original):
    operations, plan = _original(ctx, selector, name)
    assert not operations, 'fixture must exercise the direct noop return'
    if {retained!r}:
        ctx._resolve_view(selector).define_user_type('RetainedClass1031', bn.Type.int(4))
    return operations, plan
observed_plan._review_direct_original = original
engine._class_apply_types_plan = observed_plan
""")
            assert installed.returncode == 0, installed.stdout + installed.stderr
            response = shared_bn.run(
                "class", "apply-types", "svc::Codec", "--preview", "--format", "json")
        finally:
            restored = shared_bn.run("py", "exec", "--code", """
from bn_agent_bridge import mutation_engine as engine
original = getattr(engine._class_apply_types_plan, '_review_direct_original', None)
if original is not None:
    engine._class_apply_types_plan = original
""")
            assert restored.returncode == 0, restored.stdout + restored.stderr
        assert response.returncode == (3 if retained else 0), response.stdout + response.stderr
        result = json.loads(response.stdout)
        after = self._state(shared_bn)
        assert after["functions"] == before["functions"]
        assert shared_bn.json("types", "show", "svc::Codec")["layout"] == layout
        assert result["class_apply_types"]["instance_struct"] == "unchanged", result
        if retained:
            self._retained(before, after, result, ["RetainedClass1031"])
        else:
            assert result["success"] is True and result["rolled_back"] is True, result
            assert result["results"][0]["status"] == "noop", result
            assert after["ids"] == before["ids"]
            assert not result.get("kept_library_imports"), result
def test_diagnostic_recovery_and_target_commands_run_as_printed(tmp_path):
    """Transport recovery starts fresh; target inspection keeps its live routing."""
    import re
    import shlex
    from types import SimpleNamespace

    from bn.cli import _contextualize_diagnostic_hints
    from bn.transport import _bridge_recovery_advice

    def execute(command):
        argv = shlex.split(command)
        assert argv[0] == "bn"
        result = _bn(*argv[1:], timeout=_SESSION_START_TIMEOUT)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
        return result.stdout

    advice = " ".join(_bridge_recovery_advice(
        SimpleNamespace(registry_path=tmp_path / "dead.json")))
    routed = _contextualize_diagnostic_hints(
        {"error": advice}, target="my tgt", instance="X1")["error"]
    commands = re.findall(r"`(bn [^`]+)`", routed)
    start = next(c for c in commands if shlex.split(c)[1:3] == ["session", "start"])
    started = execute(start)
    match = re.search(r"^instance: (\S+)$", started, re.MULTILINE)
    assert match, started
    instance_id = match.group(1)
    try:
        loaded = _bn("-i", instance_id, "load", str(HELLO_BINARY))
        assert loaded.returncode == 0, (loaded.stdout, loaded.stderr)
        other = _bn("-i", instance_id, "load", str(ADD_BINARY))
        assert other.returncode == 0, (other.stdout, other.stderr)

        def hint(command):
            value = _contextualize_diagnostic_hints(
                {"error": f"Inspect with `{command}`."},
                target=str(HELLO_BINARY), instance=instance_id)
            return re.findall(r"`(bn [^`]+)`", value["error"])[0]

        listing = hint("bn session list --format json")
        before = json.loads(execute(listing))["items"]
        old_pid = next(row["pid"] for row in before if row["instance_id"] == instance_id)
        for command in (
            "bn target info --format json",
            f"bn -i {shlex.quote(instance_id)} target info --format json",
        ):
            info = json.loads(execute(hint(command)))
            assert "libc.so.6" in info["needed_libraries"], info
        execute(hint(f"bn -i {shlex.quote(instance_id)} session status --format json"))
        execute(hint(f"bn session restart {shlex.quote(instance_id)} --discard-unsaved"))
        after = json.loads(execute(listing))["items"]
        new_pid = next(row["pid"] for row in after if row["instance_id"] == instance_id)
        assert new_pid != old_pid
        execute(hint(f"bn session stop {shlex.quote(instance_id)} --discard-unsaved"))
        stopped = json.loads(execute(listing))["items"]
        assert instance_id not in {row["instance_id"] for row in stopped}
    finally:
        _session_stop(instance_id)


class TestAddressOperandNames1042:
    """#1042 items 1-2 on BN's own symbol table: whether a stripped view has a
    symbol behind `data_<hex>` / `sub_<hex>` is BN's call, and the mocked fakes
    can only model it. The dispatch fixture has a named data table (`table`)
    and a `.bss` buffer (`g_out`) that a stripped copy renders as `data_<hex>`."""

    def _table_address(self, shared_bn) -> str:
        rows = shared_bn.json("data", "symbols", "--query", "table")["items"]
        return next(row["a"] for row in rows if row["n"] == "table")

    def test_read_and_data_vars_take_a_data_symbol_name(self, shared_bn):
        shared_bn.load(DISPATCH_BINARY)
        address = self._table_address(shared_bn)

        by_name = shared_bn.json("read", "table", "16")
        assert by_name["resolved_symbol"] == {"name": "table", "kind": "data",
                                              "address": address}
        assert by_name["hex"] == shared_bn.json("read", address, "16")["hex"]

        at = shared_bn.json("data", "vars", "--at", "table")
        assert [row["address"] for row in at["items"]] == [address]

        text = shared_bn.run("read", "table", "8")
        assert text.returncode == 0, text.stderr
        assert text.stdout.startswith(f"note: resolved 'table' as a data symbol @ {address}\n")

    def test_stripped_placeholders_resolve_and_rename(self, shared_bn, tmp_path):
        stripped = tmp_path / "dispatch_stripped"
        build = subprocess.run(["strip", "-o", str(stripped), str(DISPATCH_BINARY)],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        nm = subprocess.run(["nm", str(DISPATCH_BINARY)], capture_output=True, text=True)
        g_out = next(int(line.split()[0], 16) for line in nm.stdout.splitlines()
                     if line.endswith(" g_out"))
        shared_bn.load(stripped, copy=False)
        placeholder = f"data_{g_out:x}"

        read = shared_bn.json("read", placeholder, "4")
        assert read["address"] == hex(g_out)
        assert read["resolved_symbol"]["kind"] == "placeholder"
        xrefs = shared_bn.json("xrefs", placeholder)
        assert xrefs["resolved_symbol"]["kind"] == "placeholder" and xrefs["total"] > 0

        # The batch that used to roll back on `Symbol not found: data_<hex>`.
        function = next(f for f in shared_bn.json("function", "list", "--limit", "500")["items"]
                        if f["name"].startswith("sub_"))
        manifest = tmp_path / "rename.json"
        manifest.write_text(json.dumps({"ops": [
            {"op": "rename_symbol", "identifier": placeholder, "new_name": "g_out",
             "kind": "data"},
            {"op": "rename_symbol", "identifier": function["name"], "new_name": "h_first"},
        ]}), encoding="utf-8")
        result = shared_bn.run("batch", "apply", str(manifest), "--preview", "--format", "json")
        assert result.returncode == 0, (result.stdout, result.stderr)
        rows = json.loads(result.stdout)["results"]
        assert [row["status"] for row in rows] == ["verified", "verified"], rows
        assert rows[0]["address"] == hex(g_out)
        assert rows[0]["resolved_symbol"]["kind"] == "placeholder"


_VTABLE_LAYOUT_ASM = r"""
.text
.macro fn name, value
.globl \name
.type \name,@function
\name:
 mov $\value,%eax
 ret
.size \name,.-\name
.endm
fn short_a, 1
fn short_b, 2
fn op_c, 3
fn op_d, 4
fn op_e, 5
fn rec_a, 6
fn rec_b, 7
fn rec_c, 8
fn impl_d1, 9
fn impl_d0, 10
fn impl_a, 11
fn impl_b, 12
.globl main
.type main,@function
main:
 xor %eax,%eax
 ret
.size main,.-main
.section .data.rel.ro,"aw",@progbits
.p2align 3
/* A vtable whose symbol covers two slots, then an anonymous function table
   with no header between them: the pointer run merges the two. */
.globl _ZTV5Short
.type _ZTV5Short,@object
_ZTV5Short:
 .quad 0,_ZTI5Short,short_a,short_b
.size _ZTV5Short,32
ops:
 .quad op_c,op_d,op_e
 .quad 0
/* A registry record with the Itanium header shape and no _ZTV symbol. */
.globl registry
.type registry,@object
registry:
 .quad 0,_ZTI5Short,rec_a,rec_b,rec_c
.size registry,40
 .quad 0
/* An all-pure abstract class's vtable, all filler, right before Impl's. */
.globl _ZTV8Abstract
.type _ZTV8Abstract,@object
_ZTV8Abstract:
 .quad 0,_ZTI8Abstract,0,0
.size _ZTV8Abstract,32
.globl _ZTV4Impl
.type _ZTV4Impl,@object
_ZTV4Impl:
 .quad 0,_ZTI4Impl,impl_d1,impl_d0,impl_a,impl_b
.size _ZTV4Impl,48
.globl _ZTI5Short
.type _ZTI5Short,@object
_ZTI5Short:
 .quad 0,_ZTS5Short
.size _ZTI5Short,16
.globl _ZTI8Abstract
.type _ZTI8Abstract,@object
_ZTI8Abstract:
 .quad 0,_ZTS8Abstract
.size _ZTI8Abstract,16
.globl _ZTI4Impl
.type _ZTI4Impl,@object
_ZTI4Impl:
 .quad 0,_ZTS4Impl
.size _ZTI4Impl,16
.section .rodata
_ZTS5Short: .asciz "5Short"
_ZTS8Abstract: .asciz "8Abstract"
_ZTS4Impl: .asciz "4Impl"
.section .note.GNU-stack,"",@progbits
"""

_VIRTUAL_BASES_CPP = """
struct V { long v = 1; virtual ~V() {} virtual int vf() { return 1; } };
struct B : virtual V { long b = 2; virtual int bf() { return 2; } int vf() override { return 3; } };
struct C : virtual V { long c = 3; virtual int cf() { return 4; } };
struct D : B, C { long d = 4; int bf() override { return 5; } int cf() override { return 6; } };
int main(int argc, char **) { B *b = new D; C *c = new D; return b->bf() + c->cf() + argc; }
"""

_HANDLER_STORE_C = """
#include <stdio.h>
#include <stdlib.h>
struct steps { long id; void (*on_open)(int); void (*on_close)(int); };
static void open_step(int x) { printf("open %d\\n", x); }
static void close_step(int x) { printf("close %d\\n", x); }
__attribute__((noinline)) void steps_init(struct steps *s) {
  s->id = 7; s->on_open = open_step; s->on_close = close_step;
}
int main(int argc, char **argv) {
  struct steps *s = malloc(sizeof *s);
  steps_init(s);
  if (argc > 3) s->on_open(argc);
  return (int)s->id;
}
"""


class TestEvidenceVtableLabels1017:
    """#1017 items 1 and 3, the #1017 `code_ref` store comment, #1018 item 1 and
    the #1033 surface footer, on real builds: vtable labels and slots come only
    from a defined ELF `_ZTV`/`_ZTC` object's `st_size`, so every #1006
    counterexample (a merged following table, a registry record, all-pure
    filler before a real vtable) is labelled by what the symbols say."""

    def _json(self, shared_bn, *args):
        res = shared_bn.run(*args, "--format", "json")
        assert res.returncode == 0, (res.stdout, res.stderr)
        return json.loads(res.stdout)

    def _build(self, tmp_path, name, source, *flags, compiler="cc", suffix=".s"):
        src = tmp_path / f"{name}{suffix}"
        src.write_text(source)
        binary = tmp_path / name
        build = subprocess.run([compiler, *flags, "-o", str(binary), str(src)],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        return binary

    def _symbols(self, shared_bn, *names):
        return shared_bn.json(
            "py", "exec",
            f"names = {list(names)!r}\n"
            "result = {n: next(s.address for s in bv.get_symbols_by_raw_name(n)\n"
            "                  if s.type.name == 'DataSymbol') for n in names}\n")["result"]

    def test_surface_labels_come_from_symbol_extents_only(self, shared_bn, tmp_path):
        binary = self._build(tmp_path, "layout", _VTABLE_LAYOUT_ASM)
        shared_bn.load(binary)
        sym = self._symbols(shared_bn, "_ZTV5Short", "registry", "_ZTV4Impl")
        out = self._json(shared_bn, "evidence", "surface")
        tables = {t["address"]: t for t in out["candidate_tables"]}
        short = tables[hex(sym["_ZTV5Short"] + 16)]
        assert short["entries"] == 5
        label, = short["elf_vtables"]
        assert (label["object"], label["entries_inside"]) == ("_ZTV5Short", 2)
        assert label["first_entry"] == hex(sym["_ZTV5Short"] + 16)
        assert short["entries_outside_elf_vtables"] == 3
        assert short["extends_past_elf_vtable"] is True
        record = tables[hex(sym["registry"] + 16)]
        assert record["elf_vtables"] == [] and record["entries_outside_elf_vtables"] == 3
        impl = tables[hex(sym["_ZTV4Impl"] + 16)]
        label, = impl["elf_vtables"]
        assert (label["label"], label["entries_inside"]) == ("vtable for Impl", 4)
        assert impl["entries_outside_elf_vtables"] == 0
        text = shared_bn.run("evidence", "surface").stdout
        assert "    in vtable for Short (_ZTV5Short @ " in text
        assert "3 entries outside every ELF vtable object -- the run continues past" in text

        hidden = self._json(shared_bn, "evidence", "surface", "--hide-vtables")
        assert hidden["summary"]["vtable_tables_hidden"] == 1
        assert {t["address"] for t in hidden["candidate_tables"]} == set(tables) - {
            hex(sym["_ZTV4Impl"] + 16)}

    def test_stripped_surface_has_no_labels_and_says_why(self, shared_bn, tmp_path):
        binary = self._build(tmp_path, "layout", _VTABLE_LAYOUT_ASM)
        stripped = tmp_path / "layout-stripped"
        result = subprocess.run(["strip", "-o", str(stripped), str(binary)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        shared_bn.load(stripped)
        out = self._json(shared_bn, "evidence", "surface", "--hide-vtables")
        assert out["candidate_tables"]
        assert all(t["elf_vtables"] == [] for t in out["candidate_tables"])
        assert out["summary"]["vtable_tables_hidden"] == 0
        assert "no .symtab (stripped)" in out["summary"]["vtable_labels"]["reason"]
        assert any(w.startswith("--hide-vtables hid nothing: no .symtab") for w in out["warnings"])
        text = shared_bn.run("evidence", "surface").stdout
        assert "vtable labels: none -- no .symtab (stripped)" in text

    def test_relocatable_object_gets_no_labels_and_says_why_1046(self, shared_bn, tmp_path):
        """#1046 review round 2: in an ET_REL object `st_value` is a section
        offset; BN placed `g_ops` where `_ZTV5Shape`'s value pointed, and the
        table was labelled and hidden as Shape's vtable."""
        source = (
            "static int a(int x) { return x + 1; } static int b(int x) { return x + 2; }\n"
            "static int c(int x) { return x + 3; } static int d(int x) { return x + 4; }\n"
            "int (*g_ops[6])(int) = { a, b, c, d, a, b };\n"
            "struct Shape { virtual ~Shape(); virtual int area(); virtual int sides(); };\n"
            "Shape::~Shape() {} int Shape::area() { return 1; } int Shape::sides() { return 0; }\n")
        obj = self._build(tmp_path, "rel.o", source, "-c", "-O1", "-fno-toplevel-reorder",
                          "-ffunction-sections", "-fdata-sections", compiler="c++",
                          suffix=".cpp")
        header = subprocess.run(["readelf", "-h", str(obj)], capture_output=True, text=True)
        assert "REL (Relocatable file)" in header.stdout
        shared_bn.load(obj)
        out = self._json(shared_bn, "evidence", "surface", "--hide-vtables")
        assert out["summary"]["vtable_labels"]["status"] == "relocatable"
        assert "ET_REL" in out["summary"]["vtable_labels"]["reason"]
        assert out["summary"]["vtable_tables_hidden"] == 0
        # No census, so no table carries a label field at all.
        assert out["candidate_tables"]
        assert all("elf_vtables" not in t for t in out["candidate_tables"])
        assert any(w.startswith("--hide-vtables hid nothing: a relocatable object")
                   for w in out["warnings"])
        callsites = self._json(shared_bn, "callsites", "_ZN5Shape4areaEv")
        assert callsites["callee_data_ref_vtables"] == []
        assert callsites["callee_data_ref_objects"] == []
        if callsites["callee_data_refs"]:
            assert callsites["callee_data_ref_census"]["status"] == "relocatable"

    def test_construction_vtables_are_labelled_by_extent(self, shared_bn, tmp_path):
        import shlex

        binary = self._build(tmp_path, "vbases", _VIRTUAL_BASES_CPP, "-O0",
                             compiler="c++", suffix=".cpp")
        shared_bn.load(binary)
        out = self._json(shared_bn, "evidence", "surface")
        # D's primary and secondary tables all lie in `_ZTV1D`'s extent.
        assert sum(1 for t in out["candidate_tables"] for obj in t["elf_vtables"]
                   if obj["object"] == "_ZTV1D") >= 2, out["candidate_tables"]
        # A function stored in B-in-D's construction vtable: its callsites note
        # names the _ZTC object, and keeps a runnable xrefs command.
        ztc = self._symbols(shared_bn, "_ZTC1D0_1B")["_ZTC1D0_1B"]
        found = shared_bn.json(
            "py", "exec",
            f"start = {ztc}\n"
            "words = [bv.read_int(start + i * 8, 8, False) for i in range(8)]\n"
            "result = [(hex(start + i * 8), hex(w)) for i, w in enumerate(words)\n"
            "          if bv.get_function_at(w) is not None]\n")["result"]
        slot_address, callee = found[0]
        callsites = self._json(shared_bn, "callsites", callee)
        row = next(r for r in callsites["callee_data_ref_objects"] if r["address"] == slot_address)
        obj, = row["objects"]
        assert obj["kind"] == "construction_vtable" and obj["object"] == "_ZTC1D0_1B"
        assert obj["label"] == "construction vtable for B-in-D"
        assert "slot" not in obj
        text = shared_bn.run("callsites", callee).stdout
        line = next(ln for ln in text.splitlines() if ln.startswith(f"  {slot_address}:"))
        assert "construction vtable for B-in-D (_ZTC1D0_1B @ " in line and "slot" not in line
        command, = re.findall(r"`(bn [^`]+)`", line)
        result = shared_bn.run(*shlex.split(command)[1:])
        assert result.returncode == 0, (command, result.stderr)

    def test_code_ref_store_and_footer_command_run_as_printed(self, shared_bn, tmp_path):
        """The footer's verify command runs exactly as printed -- with a second
        target open, so it must carry this invocation's routing (#1046 review)."""
        import shlex

        from conftest import _BN_CLI

        binary = self._build(tmp_path, "steps", _HANDLER_STORE_C, "-O1",
                             compiler="cc", suffix=".c")
        steps = shared_bn.load(binary)
        # BN made functions of both handlers; remove them so they are the
        # function-less, code-referenced addresses the surface reports.
        starts = shared_bn.json(
            "py", "exec",
            "fs = [f for f in bv.functions if f.name in ('open_step', 'close_step')]\n"
            "result = {f.name: f.start for f in fs}\n"
            "for f in fs:\n"
            "    bv.remove_user_function(f)\n"
            "bv.update_analysis_and_wait()\n")["result"]
        assert set(starts) == {"open_step", "close_step"}
        shared_bn.load(self._build(tmp_path, "layout", _VTABLE_LAYOUT_ASM))
        out = self._json(shared_bn, "evidence", "surface", "-t", steps)
        rows = {int(c["address"], 16): c for c in out["missing_function_candidates"]}
        for name, offset in (("open_step", "8"), ("close_step", "0x10")):
            stored = rows[starts[name]]["stored_to"]
            assert {(s["function"], s["destination"]) for s in stored} == {
                ("steps_init", f"[arg1 + {offset}]")}, stored
            assert rows[starts[name]]["stored_to_total"] == len(stored)
        text = shared_bn.run("evidence", "surface", "-t", steps).stdout
        assert "    stored to [arg1 + 0x10] in steps_init @ " in text
        command, = re.findall(r"verify with `(bn [^`]+)`", text)
        # Exactly as printed: no routing added by this harness.
        result = subprocess.run([*_BN_CLI, *shlex.split(command)[1:]], capture_output=True,
                                text=True, timeout=60, env=shared_bn.env())
        assert result.returncode == 0, (command, result.stderr)
        # The first listed candidate: code-likely ones lead, else address order.
        cands = out["missing_function_candidates"]
        first = int(next((c for c in cands if c["code_likely"]), cands[0])["address"], 16)
        assert result.stdout.startswith(
            f"// bn: linear disassembly of 16 units from {hex(first)} ")

    def test_evidence_function_callee_filter_1037(self, shared_bn, tmp_path):
        """#1037 item 7: `operator new` names every `_Znwm` call by its display
        name; excluding it leaves exactly the other calls, and paging follows."""
        binary = self._build(tmp_path, "vbases", _VIRTUAL_BASES_CPP, "-O0",
                             compiler="c++", suffix=".cpp")
        shared_bn.load(binary)
        full = self._json(shared_bn, "evidence", "function", "main")
        names = [((c.get("target") or {}).get("function") or {}).get("name") for c in full["calls"]]
        news = sum(1 for n in names if n == "_Znwm")
        assert news == 2, names
        kept = self._json(shared_bn, "evidence", "function", "main",
                          "--exclude-callee", "operator new")
        assert kept["total_calls"] == full["total_calls"]
        assert kept["matched_calls"] == full["total_calls"] - news
        assert kept["callee_filter"]["excluded"] == news
        assert all(((c.get("target") or {}).get("function") or {}).get("name") != "_Znwm"
                   for c in kept["calls"])
        only = self._json(shared_bn, "evidence", "function", "main", "--callee", "_Znwm",
                          "--limit", "1")
        assert only["matched_calls"] == news and only["returned"] == 1 and only["has_more"]
        text = shared_bn.run("evidence", "function", "main", "--callee", "operator new",
                             "--compact").stdout
        rows = [ln for ln in text.splitlines() if ln.startswith("- ")]
        assert len(rows) == news and all("operator new @ " in row for row in rows)
