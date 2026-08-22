"""Filesystem-layout invariants for :mod:`bn.paths`."""
from __future__ import annotations

import contextlib
import shutil
import socket as _socket
import tempfile
from pathlib import Path

import pytest

from bn import paths as _paths
from bn.transport import BridgeError, validate_instance_id


# --- sockaddr_un.sun_path bound -------------------------------------------------
#
# The deep roots below are built under a SHORT fixed base rather than pytest's
# `tmp_path`. `tmp_path` already spends ~60 bytes, and pytest-xdist adds another
# `popen-gwN/` segment per worker, so a budget computed from it goes negative
# under `-n auto` and on macOS (long /private/var base + a 103-byte budget).


@contextlib.contextmanager
def _cache_dir_of_length(monkeypatch, total: int):
    """A BN_CACHE_DIR whose `instances/<id>.sock` path is exactly *total* bytes."""
    base = tempfile.mkdtemp(prefix="s", dir="/tmp")
    try:
        # total = <base>/<pad>/instances/<id>.sock
        fixed = len(base) + 1 + len("/instances/") + len("i.sock")
        pad = total - fixed
        if pad < 1:
            pytest.skip(f"base {base!r} too long to build a {total}-byte probe")
        root = Path(base) / ("d" * pad)
        (root / "instances").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("BN_CACHE_DIR", str(root))
        yield root
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_socket_path_budget_leaves_room_for_the_nul():
    # sun_path is a fixed char array and bind() needs the terminator, so the
    # usable path is one byte shorter than the array.
    assert _paths.socket_path_budget() == _paths.SOCKET_PATH_MAX_BYTES - 1
    assert _paths.SOCKET_PATH_MAX_BYTES in (104, 108)


def test_socket_path_budget_matches_what_bind_actually_accepts(tmp_path):
    # The one test that would catch an off-by-one in the constant: probe the
    # kernel instead of restating the source. CPython rejects a non-abstract
    # path when len >= sizeof(sun_path), so budget bytes must bind and
    # budget+1 must not.
    base = tempfile.mkdtemp(prefix="s", dir="/tmp")
    try:
        budget = _paths.socket_path_budget()
        for length, should_bind in ((budget, True), (budget + 1, False)):
            name = Path(base) / ("x" * (length - len(base) - 1))
            sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            try:
                sock.bind(str(name))
                bound = True
            except OSError:
                bound = False
            finally:
                sock.close()
            assert bound is should_bind, f"{length} bytes: bind={bound}"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_socket_path_at_the_budget_is_accepted(monkeypatch):
    with _cache_dir_of_length(monkeypatch, _paths.socket_path_budget()):
        path = _paths.bridge_socket_path("i")
        assert len(str(path).encode()) == _paths.socket_path_budget()


def test_socket_path_one_byte_over_the_budget_raises(monkeypatch):
    with _cache_dir_of_length(monkeypatch, _paths.socket_path_budget() + 1):
        with pytest.raises(ValueError) as exc:
            _paths.bridge_socket_path("i")
    msg = str(exc.value)
    assert f"{_paths.socket_path_budget() + 1} bytes" in msg   # the real count
    assert f"limit {_paths.socket_path_budget()}" in msg
    assert "--instance-id" in msg and "BN_CACHE_DIR" in msg    # both escape hatches


def test_socket_basename_is_always_the_instance_id(monkeypatch):
    # `bn instance gc` reverse-maps sockets to instances by comparing a socket's
    # stem to the live registry stems, which carry the FULL id. Any scheme that
    # renames the socket (hashing, truncation) makes a live instance's socket
    # look like a registry-less orphan and gc unlinks it out from under the
    # running bridge. Keep filename == id, and fail instead of renaming.
    # A short base, so a deliberately long-but-legal id still fits the budget --
    # `tmp_path` alone spends ~60 of the 107 bytes.
    base = tempfile.mkdtemp(prefix="s", dir="/tmp")
    try:
        monkeypatch.setenv("BN_CACHE_DIR", base)
        for instance_id in ("a", "phase2", "a-descriptive-instance-id-for-parallel-runs"):
            sock = _paths.bridge_socket_path(instance_id)
            assert sock.stem == instance_id
            assert sock.stem == _paths.bridge_registry_path(instance_id).stem
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_validate_instance_id_rejects_an_unfittable_id_before_spawning(monkeypatch):
    # The whole point: fail in the CLI with a byte count, not as a bare
    # `OSError: AF_UNIX path too long` from bind() inside the spawned bridge.
    with _cache_dir_of_length(monkeypatch, _paths.socket_path_budget() + 40):
        with pytest.raises(BridgeError) as exc:
            validate_instance_id("i" * 40)
    assert "too long" in str(exc.value) and "limit" in str(exc.value)


def test_validate_instance_id_still_accepts_a_normal_id(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    assert validate_instance_id("phase2-probe") == "phase2-probe"
