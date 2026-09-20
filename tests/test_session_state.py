from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from bn import session_state
from bn.paths import session_state_path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Isolated cache dir + fake git project as the session-state key."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    return root


def test_read_returns_empty_dict_when_missing(project):
    assert session_state.read() == {}


def test_read_returns_empty_dict_on_malformed_file(project):
    path = session_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{", encoding="utf-8")

    assert session_state.read() == {}


def test_update_read_round_trip(project):
    state = session_state.update(instance_id="abc123")

    assert state["instance_id"] == "abc123"
    assert state["project_root"] == str(project)

    on_disk = session_state.read()
    assert on_disk["instance_id"] == "abc123"
    assert on_disk["project_root"] == str(project)


def test_update_merges_instead_of_overwriting(project):
    # Two sequential updates of different keys must both survive -- this is
    # the read-modify-write the flock protects under concurrency.
    session_state.update(instance_id="abc123")
    session_state.update(target="libfoo.so")

    state = session_state.read()
    assert state["instance_id"] == "abc123"
    assert state["target"] == "libfoo.so"


def test_update_none_removes_key(project):
    session_state.update(instance_id="abc123", target="libfoo.so")
    state = session_state.update(target=None)

    assert "target" not in state
    assert state["instance_id"] == "abc123"
    assert "target" not in session_state.read()


def test_update_takes_exclusive_flock_around_read_modify_write(project, monkeypatch):
    import fcntl

    events: list[str] = []
    real_flock = fcntl.flock
    real_read = session_state.read

    def spy_flock(fd, op):
        events.append(f"flock:{op}")
        return real_flock(fd, op)

    def spy_read():
        events.append("read")
        return real_read()

    monkeypatch.setattr(session_state.fcntl, "flock", spy_flock)
    monkeypatch.setattr(session_state, "read", spy_read)

    session_state.update(instance_id="abc123")

    assert f"flock:{fcntl.LOCK_EX}" in events
    # The lock must be taken before the read that the merge is based on.
    assert events.index(f"flock:{fcntl.LOCK_EX}") < events.index("read")


def test_update_writes_valid_json_atomically(project):
    session_state.update(instance_id="abc123")
    path = session_state_path()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["instance_id"] == "abc123"
    # No leftover temp files from the atomic write.
    leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_fsyncs_the_pin_before_the_rename(project, monkeypatch):
    """#824: close() flushes to the OS, not to the disk. The sticky pin is the one
    piece of state a fresh process trusts without re-deriving it, so a torn write
    is a wrong target rather than a missing one.

    The guarantee is an ORDER, not a count: the pin's own bytes must be on disk
    *before* the rename publishes them. A bare `assert os.fsync was called` is
    satisfied by the post-rename directory fsync alone, so it stays green with
    the pre-rename `fh.flush()`/`os.fsync(fd)` deleted -- which is exactly the
    crash window this test exists to close. Each fsync is therefore labelled by
    the kind of fd it was handed, and the rename is recorded between them.
    """
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def spy_fsync(fd):
        events.append("fsync:dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync:file")
        return real_fsync(fd)

    def spy_replace(self, target):
        events.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(Path, "replace", spy_replace)

    state = session_state.update(target="t.bndb")

    assert "fsync:file" in events, (
        "the pin's own bytes must be fsynced, not just its directory: " f"{events}"
    )
    assert "replace" in events, events
    assert events.index("fsync:file") < events.index("replace"), (
        "the fsync must precede the rename that publishes the pin: " f"{events}"
    )
    assert state["target"] == "t.bndb"
    assert json.loads(session_state_path().read_text())["target"] == "t.bndb"
