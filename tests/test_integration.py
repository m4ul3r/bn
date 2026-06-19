"""Integration tests for multi-instance bridge sessions.

These tests require Binary Ninja to be importable. They are skipped if
the binaryninja module is not available.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HELLO_BINARY = FIXTURES_DIR / "hello_x86_64"
ADD_BINARY = FIXTURES_DIR / "add_x86_64"

try:
    # Try a cheap check: can bn-agent even start?
    # We don't import binaryninja directly since it might not be on sys.path
    # without the path-setup that headless.py does.
    _bn_python = Path("/opt/binaryninja/python")
    _has_bn = _bn_python.is_dir() and (HELLO_BINARY.exists() and ADD_BINARY.exists())
except Exception:
    _has_bn = False

pytestmark = pytest.mark.skipif(not _has_bn, reason="Binary Ninja or fixtures not available")

# Use the bn console-scripts entry point instead of -m bn.cli
# to avoid Python module shadowing issues with the 'bn' package name.
_BN_CLI = [str(Path(sys.executable).parent / "bn")]


def _bn(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_BN_CLI, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _session_start(*binaries: str, timeout: float = 30.0) -> dict:
    # session start defaults to text output; this helper parses JSON.
    cmd = [*_BN_CLI, "session", "start", "--format", "json"]
    cmd.extend(str(b) for b in binaries)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"session start failed: {result.stderr}"
    return json.loads(result.stdout)


def _session_stop(instance_id: str, timeout: float = 10.0) -> None:
    subprocess.run(
        [*_BN_CLI, "session", "stop", instance_id],
        capture_output=True,
        text=True,
        timeout=timeout,
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
                targets_a = json.loads(result_a.stdout)
                assert len(targets_a) == 1

                result_b = _bn("--instance", id_b, "target", "list", "--format", "json")
                targets_b = json.loads(result_b.stdout)
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
                sessions = data["instances"]
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


class TestProtoSetUnnamedParams:
    """Regression for #254: a `proto set` whose prototype omits parameter names
    must verify, not be reported verification_failed and reverted. BN auto-names
    unnamed params on readback (arg1, arg2, ...), so the readback text never
    matches the requested string -- only a real BN readback reproduces this, so
    the mocked suite can't cover it."""

    def _first_fn(self, inst_id):
        out = _bn("--instance", inst_id, "function", "list", "--format", "json")
        return json.loads(out.stdout)["items"][0]["name"]

    def test_unnamed_params_verify_via_preview(self):
        info = _session_start(str(HELLO_BINARY))
        inst_id = info["instance_id"]
        try:
            fn = self._first_fn(inst_id)
            res = _bn("--instance", inst_id, "proto", "set", fn,
                      f"void {fn}(int32_t, char**, char**)", "--preview", "--format", "json")
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
                      f"int32_t {fn}(int64_t argc, char** argv)", "--preview", "--format", "json")
            parsed = json.loads(res.stdout)
            assert [r.get("status") for r in parsed["results"]] == ["verified"], parsed
        finally:
            _session_stop(inst_id)
