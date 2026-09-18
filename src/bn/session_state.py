from __future__ import annotations

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - non-POSIX platforms only
    # #824 entry gate: the sticky-pin write serializes on an flock, so this
    # module cannot import on a platform without fcntl. Name the real constraint
    # instead of leaking `No module named 'fcntl'` out of `import bn.cli`.
    raise RuntimeError(
        "bn is POSIX-only: the sticky-pin store serializes on fcntl file locks "
        "and there is no Windows implementation. Run it under Linux or macOS."
    ) from exc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import ensure_private_dir, project_root, session_state_path, sessions_dir

KEYS = ("instance_id", "target")


def read() -> dict[str, Any]:
    """Return the current session state, or empty dict on missing/malformed."""
    path = session_state_path()
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def update(**fields: Any) -> dict[str, Any]:
    """Merge *fields* into on-disk state. ``None`` removes a key.

    The read -> merge -> write cycle runs under an exclusive flock on a lock
    file next to the state file, so concurrent updates (e.g. ``bn instance
    use`` and ``bn target use`` racing in the same project) can't lose writes.
    """
    ensure_private_dir(sessions_dir())
    lock_path = session_state_path().with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        state = read()
        for key, value in fields.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = value
        state["project_root"] = str(project_root())
        _atomic_write(state)
    return state


def _atomic_write(state: dict[str, Any]) -> None:
    path = session_state_path()
    ensure_private_dir(sessions_dir())
    # tempfile.mkstemp opens with mode 0o600 (never widened by umask, which can
    # only clear bits), so the sticky-pin file lands owner-only regardless of the
    # ambient umask; the atomic replace preserves that mode (#612).
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh)
            # close() flushes to the OS, not to the disk: a crash between the
            # rename and writeback can leave the pin pointing at a truncated or
            # empty file -- and the pin is the one piece of state a fresh process
            # trusts without re-deriving it, so a torn write is a wrong target
            # rather than a missing one (#824).
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
        _fsync_dir(path.parent)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of *path* so the rename above survives a crash.

    A directory cannot be opened on every platform, and the rename is already
    atomic within the filesystem: an ``OSError`` here is ignored rather than
    turned into a failed pin write.
    """
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
