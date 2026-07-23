"""Path-layout invariants (src/bn/paths.py).

Focus: the AF_UNIX socket path length ceiling. A deep ``BN_CACHE_DIR`` plus a
descriptive instance id easily exceeds the kernel's 108-byte ``sun_path`` field,
and the failure used to surface as a bare ``OSError: AF_UNIX path too long``
from ``bind()`` inside the bridge, long after the CLI had committed to the id.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest

from bn import paths


def _set_cache(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("BN_CACHE_DIR", str(root))


def _deep_root(base: Path, target_len: int) -> Path:
    """A directory path under *base* whose string length is exactly *target_len*.

    Pads with one long segment; *target_len* must exceed ``len(base) + 1``.
    """
    pad = target_len - len(str(base)) - 1
    assert pad >= 1, f"tmp_path {base} already longer than target {target_len}"
    return base / ("d" * pad)


# instances_dir() is `<root>/instances`; aim for a root that leaves a realistic
# ~18-byte budget for the socket basename (mirrors the reported campaign layout).
def _root_for_budget(base: Path, budget: int) -> Path:
    return _deep_root(base, paths.socket_path_budget() - len("/instances") - 1 - budget)


def test_sun_path_limit_matches_platform_struct():
    """The constant must reflect the real ``sockaddr_un.sun_path`` size."""
    assert paths.SOCKET_PATH_MAX_BYTES in (104, 108)
    assert paths.SOCKET_PATH_MAX_BYTES - 1 == paths.socket_path_budget()


def test_short_path_is_unchanged(monkeypatch, tmp_path):
    _set_cache(monkeypatch, tmp_path)
    sock = paths.bridge_socket_path("phase0")
    assert sock == paths.instances_dir() / "phase0.sock"


def test_long_path_is_compacted_within_limit(monkeypatch, tmp_path):
    root = _root_for_budget(tmp_path, 24)
    _set_cache(monkeypatch, root)
    long_id = "c06-btstackmain-phase0-with-a-very-descriptive-suffix"

    sock = paths.bridge_socket_path(long_id)

    assert len(str(sock).encode()) <= paths.socket_path_budget()
    assert sock.suffix == ".sock"
    assert sock.parent == paths.instances_dir()
    # Deterministic: both the CLI and the bridge derive the path independently.
    assert sock == paths.bridge_socket_path(long_id)


def test_compaction_is_collision_resistant(monkeypatch, tmp_path):
    root = _root_for_budget(tmp_path, 24)
    _set_cache(monkeypatch, root)
    prefix = "c06-btstackmain-phase0-shared-prefix-that-is-long"

    a = paths.bridge_socket_path(prefix + "-alpha")
    b = paths.bridge_socket_path(prefix + "-beta")

    assert a != b


def test_compacted_path_actually_binds(monkeypatch, tmp_path):
    """End-to-end: the compacted path must survive a real ``bind()``."""
    root = _root_for_budget(tmp_path, 24)
    _set_cache(monkeypatch, root)
    sock_path = paths.bridge_socket_path("c06-btstackmain-phase0-long-descriptive-id")
    paths.ensure_private_dir(sock_path.parent)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.bind(str(sock_path))  # raised OSError: AF_UNIX path too long before the fix
    sock_path.unlink()


def test_hopeless_cache_dir_fails_early_with_diagnostics(monkeypatch, tmp_path):
    """No basename can fit -> a clear error naming the length, not a bind OSError."""
    root = _deep_root(tmp_path, paths.SOCKET_PATH_MAX_BYTES + 40)
    _set_cache(monkeypatch, root)

    with pytest.raises(ValueError) as exc:
        paths.bridge_socket_path("phase0")

    msg = str(exc.value)
    assert "BN_CACHE_DIR" in msg
    assert str(paths.socket_path_budget()) in msg


def test_legacy_fixed_socket_fails_early_when_too_long(monkeypatch, tmp_path):
    root = _deep_root(tmp_path, paths.SOCKET_PATH_MAX_BYTES + 40)
    _set_cache(monkeypatch, root)

    with pytest.raises(ValueError) as exc:
        paths.bridge_socket_path(None)

    assert "BN_CACHE_DIR" in str(exc.value)


def test_registry_path_is_not_length_limited(monkeypatch, tmp_path):
    """Only the socket hits the kernel limit; the JSON registry is a plain file."""
    root = _deep_root(tmp_path, paths.SOCKET_PATH_MAX_BYTES + 40)
    _set_cache(monkeypatch, root)
    reg = paths.bridge_registry_path("phase0")
    assert reg == paths.instances_dir() / "phase0.json"


def test_cli_validate_rejects_unusable_id_as_bridge_error(monkeypatch, tmp_path):
    """The CLI fails early with a BridgeError instead of letting the spawned
    bridge die on ``OSError: AF_UNIX path too long``."""
    from bn import transport

    _set_cache(monkeypatch, _deep_root(tmp_path, paths.SOCKET_PATH_MAX_BYTES + 40))

    with pytest.raises(transport.BridgeError, match="too long"):
        transport.validate_instance_id("phase0")


def test_cli_validate_accepts_long_id_that_compacts(monkeypatch, tmp_path):
    from bn import transport

    _set_cache(monkeypatch, _root_for_budget(tmp_path, 24))
    long_id = "c06-btstackmain-phase0-with-a-very-descriptive-suffix"

    assert transport.validate_instance_id(long_id) == long_id


def test_tight_budget_falls_back_to_digest_only_name(monkeypatch, tmp_path):
    """When not even one readable prefix char fits, the digest alone is used."""
    _set_cache(monkeypatch, _root_for_budget(tmp_path, 13))
    sock = paths.bridge_socket_path("c06-btstackmain-phase0")

    assert len(sock.name) == 13
    assert len(str(sock).encode()) <= paths.socket_path_budget()
    assert sock.stem == hashlib.sha256(b"c06-btstackmain-phase0").hexdigest()[:8]
