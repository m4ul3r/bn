"""`project_ops.ProjectManager` and its free functions against a fake project.

Everything here is behaviour a `bn project ...` caller can observe: selector
grammar, file rows, the first-save recipe, the ownership guard. The fake
`Project`/`ProjectFile`/`ProjectFolder` model measured BN behaviour (opaque
extensionless blobs under `data/`,
`create_file(b"")` writing nothing, `get_files_in_folder`).
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import subprocess
import sys
import types
import weakref
from pathlib import Path

import pytest

from _bridge_fakes import _load_bridge
from _project_fakes import (
    SQLITE_MAGIC, FakeProject, FakeProjectException, FakeProjectFile,
)


@pytest.fixture
def po(monkeypatch):
    """The project_ops module loaded against the shared fake `binaryninja`."""
    _load_bridge(monkeypatch)
    module = sys.modules["bn_test_bridge.project_ops"]
    fake_bn = sys.modules["binaryninja"]
    fake_bn.ProjectFile = FakeProjectFile
    fake_bn.ProjectException = FakeProjectException
    return module


@pytest.fixture
def bnpr(tmp_path):
    root = tmp_path / "demo.bnpr"
    root.mkdir()
    (root / "project.bnpm").write_bytes(SQLITE_MAGIC)
    return root


@pytest.fixture
def opened(po, bnpr, monkeypatch):
    """`(manager, project)` with the fake project already open in the manager."""
    project = FakeProject(bnpr)
    fake_bn = sys.modules["binaryninja"]
    fake_bn.Project = types.SimpleNamespace(open_project=lambda path: project)
    manager = po.ProjectManager()
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    manager.open(str(bnpr), self_pid=os.getpid())
    yield manager, project
    manager.close()


class FakeTargets:
    """What `save_project_database` tells the bridge's TargetManager."""

    def __init__(self, binding=None):
        self.binding = binding
        self.cleared: list = []
        self.databases: list = []
        self.project_databases: list = []

    def project_binding(self, bv):
        return self.binding

    def clear_dirty(self, bv):
        self.cleared.append(bv)

    def note_database(self, bv, path):
        self.databases.append((bv, path))

    def note_project_database(self, bv, file_id, path):
        self.project_databases.append((bv, file_id, path))


def _bridge(manager, targets=None):
    bridge = types.SimpleNamespace(projects=manager, targets=targets or FakeTargets())
    bridge.registry_writes = 0

    def _write():
        bridge.registry_writes += 1

    bridge._write_registry = _write
    return bridge


class SaveBV:
    """A project-backed view: `create_database` writes a SQLite-looking blob."""

    def __init__(self, pf, *, has_database=False, snapshot_ok=True, create_result=True):
        self.project_file = pf
        self.file = types.SimpleNamespace(
            filename=pf.path_on_disk, has_database=has_database,
            save_auto_snapshot=lambda: snapshot_ok)
        self._create_result = create_result
        self.created_with: str | None = None
        self.parent_existed: bool | None = None

    def create_database(self, dest: str) -> bool:
        self.created_with = dest
        self.parent_existed = Path(dest).parent.is_dir()
        if self._create_result:
            Path(dest).write_bytes(SQLITE_MAGIC + b"\x00" * 16)
        return self._create_result


# --- canonical path ---------------------------------------------------------


def test_canonical_project_path_accepts_dir_and_bnpm(po, bnpr):
    assert po.canonical_project_path(str(bnpr)) == bnpr.resolve()
    assert po.canonical_project_path(str(bnpr / "project.bnpm")) == bnpr.resolve()


# --- open -------------------------------------------------------------------


def test_open_refuses_a_directory_without_project_bnpm(po, tmp_path, monkeypatch):
    plain = tmp_path / "not_a_project"
    plain.mkdir()
    manager = po.ProjectManager()
    with pytest.raises(RuntimeError, match="Not a Binary Ninja project"):
        manager.open(str(plain), self_pid=os.getpid())
    assert manager.current is None


def test_open_twice_same_path_reports_opened_false(opened, bnpr):
    manager, project = opened
    state, again = manager.open(str(bnpr / "project.bnpm"), self_pid=os.getpid())
    assert again is False
    assert state.project is project
    assert state.path == str(bnpr.resolve())


def test_open_a_different_project_while_one_is_open_is_refused(opened, tmp_path):
    manager, _ = opened
    other = tmp_path / "other.bnpr"
    other.mkdir()
    (other / "project.bnpm").write_bytes(SQLITE_MAGIC)
    with pytest.raises(RuntimeError, match="different project is already open"):
        manager.open(str(other), self_pid=os.getpid())


def test_open_wraps_project_exception_with_the_licence_hint(po, bnpr, monkeypatch):
    def boom(path):
        raise FakeProjectException("Failed to open project")

    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=boom)
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    manager = po.ProjectManager()
    with pytest.raises(RuntimeError) as refused:
        manager.open(str(bnpr), self_pid=os.getpid())
    text = str(refused.value)
    assert "could not open project" in text and "Failed to open project" in text
    assert "Commercial or Ultimate" in text
    assert manager.current is None
    replacement = FakeProject(bnpr)
    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=lambda path: replacement)
    manager.open(str(bnpr), self_pid=os.getpid())
    manager.close()


def test_close_releases_the_handle_and_the_state(opened):
    manager, project = opened
    state = manager.close()
    assert state is not None and project.closed is True
    assert manager.current is None
    with pytest.raises(RuntimeError, match="No project is open"):
        manager.require()


def _subprocess_open(po, bnpr, tmp_path):
    """An unrelated interpreter, registry and fake native handle."""
    script = """
import importlib, json, os, sys, types
package = types.ModuleType("lease_test_bridge")
package.__path__ = [sys.argv[1]]
sys.modules[package.__name__] = package
po = importlib.import_module("lease_test_bridge.project_ops")
calls = []
def native_open(path):
    calls.append(path)
    return types.SimpleNamespace(id="fake", name="fake", close=lambda: None)
sys.modules["binaryninja"] = types.SimpleNamespace(
    Project=types.SimpleNamespace(open_project=native_open))
manager = po.ProjectManager()
try:
    manager.open(sys.argv[2], self_pid=os.getpid())
except RuntimeError as exc:
    result = {"opened": False, "error": str(exc), "native_calls": len(calls)}
else:
    result = {"opened": True, "native_calls": len(calls)}
    manager.close()
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(po.__file__).parent), str(bnpr)],
        env={**os.environ, "BN_CACHE_DIR": str(tmp_path / "independent-cache")},
        text=True, capture_output=True, timeout=15, check=True,
    )
    return json.loads(completed.stdout)


def test_lifetime_lease_blocks_independent_process_and_reopens(opened, po, bnpr, tmp_path):
    manager, _ = opened
    lock = bnpr / po._LEASE_FILENAME
    inode = lock.stat().st_ino
    refused = _subprocess_open(po, bnpr, tmp_path)
    assert refused["opened"] is False
    assert refused["native_calls"] == 0
    assert "exclusive project lease is held" in refused["error"]
    manager.close()
    assert lock.stat().st_ino == inode
    assert _subprocess_open(po, bnpr, tmp_path) == {"opened": True, "native_calls": 1}
    manager.open(str(bnpr), self_pid=os.getpid())
    assert lock.stat().st_ino == inode
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is False


def test_lease_is_acquired_before_native_open(po, bnpr, monkeypatch, tmp_path):
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    project = FakeProject(bnpr)

    def native_open(path):
        refused = _subprocess_open(po, bnpr, tmp_path)
        assert refused["opened"] is False
        assert refused["native_calls"] == 0
        return project

    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=native_open)
    manager = po.ProjectManager()
    try:
        manager.open(str(bnpr), self_pid=os.getpid())
    finally:
        manager.close()


@pytest.mark.parametrize("failure", ["exception", "none", "interrupt"])
def test_failed_open_releases_lease_and_descriptor(po, bnpr, monkeypatch, tmp_path, failure):
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    descriptors = []
    flock = fcntl.flock

    def record_flock(fd, operation):
        descriptors.append(fd)
        return flock(fd, operation)

    monkeypatch.setattr(po.fcntl, "flock", record_flock)

    def native_open(path):
        if failure == "exception":
            raise FakeProjectException("open failed")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return None

    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=native_open)
    manager = po.ProjectManager()
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        manager.open(str(bnpr), self_pid=os.getpid())
    assert manager.current is None
    assert manager._lease_fd is None
    assert manager._gate_fd is None
    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is True


def test_contended_open_closes_unused_descriptor(opened, po, bnpr, monkeypatch):
    descriptors = []
    flock = fcntl.flock

    def record_flock(fd, operation):
        descriptors.append(fd)
        return flock(fd, operation)

    monkeypatch.setattr(po.fcntl, "flock", record_flock)
    competitor = po.ProjectManager()
    with pytest.raises(RuntimeError, match="exclusive project lease"):
        competitor.open(str(bnpr), self_pid=os.getpid())
    assert competitor.current is None
    assert competitor._lease_fd is None
    assert competitor._gate_fd is None
    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("failure", ["exception", "false"])
def test_failed_close_preserves_state_cache_and_lease(opened, po, bnpr, tmp_path, failure):
    manager, project = opened
    state = manager.current
    manager.export_cache[("file", 1, 2)] = object()
    original_close = project.close

    def fail_close():
        if failure == "false":
            return False
        raise FakeProjectException("close failed")

    project.close = fail_close
    try:
        with pytest.raises(RuntimeError, match="could not close project"):
            manager.close()
        assert manager.current is state
        assert manager.export_cache
        assert _subprocess_open(po, bnpr, tmp_path)["opened"] is False
    finally:
        project.close = original_close
    manager.close()
    assert not manager.export_cache
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is True


def test_inherited_manager_cannot_unlock_parent_lease(opened, po, bnpr, tmp_path):
    manager, _ = opened
    lease_fd = manager._lease_fd
    pid = os.fork()
    if pid == 0:
        try:
            try:
                manager.close()
            except RuntimeError as exc:
                assert "after fork" in str(exc)
            else:
                raise AssertionError("inherited manager was usable")
            assert not po._LEASE_OWNERS
            try:
                os.fstat(lease_fd)
            except OSError as exc:
                assert exc.errno == errno.EBADF
            else:
                raise AssertionError("child retained the parent's lease descriptor")
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is False
    manager.close()
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is True


def test_lifetime_owner_retains_handle_when_bridge_reference_is_dropped(
    po, bnpr, tmp_path, monkeypatch,
):
    project = FakeProject(bnpr)
    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=lambda path: project)
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    manager = po.ProjectManager()
    manager.open(str(bnpr), self_pid=os.getpid())
    retained = weakref.ref(manager)
    del manager
    try:
        assert retained() is not None
        assert retained().current.project is project
        assert _subprocess_open(po, bnpr, tmp_path)["opened"] is False
    finally:
        retained().close()
    assert retained() is None
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is True


def test_set_rootfs_checks_native_result_and_readback(opened):
    manager, project = opened
    metadata = {}

    def store(key, value):
        metadata[key] = value
        return True

    project.store_metadata = store
    project.query_metadata = metadata.__getitem__
    value = {"root": "/firmware", "libs": {}}
    manager.set_rootfs(value)
    assert manager.get_rootfs() == value
    project.store_metadata = lambda key, value: False
    with pytest.raises(RuntimeError, match="returned False"):
        manager.set_rootfs(value)
    project.store_metadata = lambda key, value: True
    with pytest.raises(RuntimeError, match="did not match"):
        manager.set_rootfs({"root": "/different"})


def test_create_handoff_excludes_process_and_adopts_without_reopening(po, tmp_path):
    root = tmp_path / "created.bnpr"
    created = FakeProject(root)

    def native_create(path, name):
        assert path == str(root)
        assert name == "created"
        root.mkdir()
        (root / "project.bnpm").write_bytes(SQLITE_MAGIC)
        refused = _subprocess_open(po, root, tmp_path)
        assert refused["opened"] is False
        assert refused["native_calls"] == 0
        assert "open/create is in progress" in refused["error"]
        return created

    # There deliberately is no open_project method: creation must not reopen.
    sys.modules["binaryninja"].Project = types.SimpleNamespace(create_project=native_create)
    manager = po.ProjectManager()
    try:
        state = manager.create(str(root), "created")
        assert state.project is created
        assert not created.closed
        assert manager._gate_fd is None
        assert manager._lease_fd is not None
        assert _subprocess_open(po, root, tmp_path)["opened"] is False
    finally:
        manager.close()
    assert created.closed
    assert _subprocess_open(po, root, tmp_path)["opened"] is True


@pytest.mark.parametrize("failure", ["exception", "none"])
def test_failed_create_releases_parent_gate(po, bnpr, tmp_path, failure):
    def native_create(path, name):
        if failure == "exception":
            raise FakeProjectException("creation failed")
        return None

    sys.modules["binaryninja"].Project = types.SimpleNamespace(create_project=native_create)
    manager = po.ProjectManager()
    with pytest.raises(RuntimeError, match="could not create project"):
        manager.create(str(tmp_path / "failed.bnpr"), "failed")
    assert manager.current is None
    assert manager._gate_fd is None
    assert manager._lease_fd is None
    assert _subprocess_open(po, bnpr, tmp_path)["opened"] is True


def test_failed_creation_handoff_and_close_keeps_parent_gate(po, tmp_path, monkeypatch):
    root = tmp_path / "created.bnpr"
    created = FakeProject(root)
    original_close = created.close

    def native_create(path, name):
        root.mkdir()
        (root / "project.bnpm").write_bytes(SQLITE_MAGIC)
        return created

    def lease_failure(canonical):
        raise OSError("could not create lease file")

    def close_failure():
        raise FakeProjectException("close failed")

    created.close = close_failure
    sys.modules["binaryninja"].Project = types.SimpleNamespace(create_project=native_create)
    manager = po.ProjectManager()
    monkeypatch.setattr(manager, "_acquire_lease", lease_failure)
    try:
        with pytest.raises(RuntimeError, match="could not close project"):
            manager.create(str(root), "created")
        assert manager.current.project is created
        assert manager._gate_fd is not None
        assert manager._lease_fd is None
        refused = _subprocess_open(po, root, tmp_path)
        assert refused["opened"] is False
        assert "open/create is in progress" in refused["error"]
    finally:
        created.close = original_close
        manager.close()
    assert manager._gate_fd is None
    assert _subprocess_open(po, root, tmp_path)["opened"] is True


@pytest.mark.parametrize("close_fails", [False, True])
def test_native_property_failure_keeps_handle_protected_until_closed(
    po, bnpr, tmp_path, monkeypatch, close_fails,
):
    class BrokenProperties:
        closed = False

        @property
        def id(self):
            raise FakeProjectException("property failed")

        def close(self):
            if close_fails and not self.closed:
                raise FakeProjectException("close failed")
            self.closed = True

    project = BrokenProperties()
    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=lambda path: project)
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    manager = po.ProjectManager()
    try:
        with pytest.raises(RuntimeError if close_fails else FakeProjectException):
            manager.open(str(bnpr), self_pid=os.getpid())
        assert (manager.current is not None) is close_fails
        assert _subprocess_open(po, bnpr, tmp_path)["opened"] is not close_fails
    finally:
        project.closed = True
        manager.close()


# --- ownership guard ---------------------------------------------------------


def _registry_tree(po, monkeypatch, tmp_path, payloads: dict[str, dict]):
    instances = tmp_path / "instances"
    instances.mkdir()
    for name, payload in payloads.items():
        (instances / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(po, "instances_dir", lambda: instances)
    monkeypatch.setattr(po, "bridge_registry_path", lambda: instances / "default.json")


@pytest.mark.parametrize(
    "verdict, alive, refused",
    [
        ("proven", True, True),
        ("mismatch", True, False),
        ("unrecorded", False, False),
        ("unrecorded", True, True),
    ],
)
def test_open_ownership_guard(po, bnpr, tmp_path, monkeypatch, verdict, alive, refused):
    canonical = str(bnpr.resolve())
    _registry_tree(po, monkeypatch, tmp_path, {
        "other-bridge": {"instance_id": "other-bridge", "pid": 4242,
                         "project": {"path": canonical, "id": "p", "name": "demo"}},
    })
    monkeypatch.setattr(po, "identity_verdict", lambda payload, pid: verdict)

    def fake_kill(pid, sig):
        assert sig == 0
        if not alive:
            raise ProcessLookupError(pid)

    monkeypatch.setattr(po.os, "kill", fake_kill)
    project = FakeProject(bnpr)
    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=lambda p: project)
    manager = po.ProjectManager()
    if refused:
        with pytest.raises(RuntimeError) as exc:
            manager.open(str(bnpr), self_pid=os.getpid())
        text = str(exc.value)
        assert "already open in bridge instance other-bridge (pid 4242)" in text
        assert f"bn project attach {canonical}" in text and "-i other-bridge" in text
        assert manager.current is None
    else:
        _, opened = manager.open(str(bnpr), self_pid=os.getpid())
        assert opened is True
        manager.close()


def test_ownership_guard_skips_own_pid_and_other_projects(po, bnpr, tmp_path, monkeypatch):
    canonical = str(bnpr.resolve())
    _registry_tree(po, monkeypatch, tmp_path, {
        "me": {"instance_id": "me", "pid": os.getpid(),
               "project": {"path": canonical, "id": "p", "name": "demo"}},
        "elsewhere": {"instance_id": "elsewhere", "pid": 4343,
                      "project": {"path": "/elsewhere/x.bnpr", "id": "q", "name": "x"}},
        "noproject": {"instance_id": "noproject", "pid": 4444, "project": None},
    })
    monkeypatch.setattr(po, "identity_verdict", lambda payload, pid: "proven")
    assert po.find_other_owner(canonical, self_pid=os.getpid()) is None


# --- selectors ---------------------------------------------------------------


@pytest.fixture
def populated(opened):
    manager, project = opened
    fw = project.add_folder("fw")
    app = project.add_file("app", fw, file_id="11111111-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    app_db = project.add_file("app.bndb", fw, data=SQLITE_MAGIC + b"\x00" * 8,
                              file_id="22222222-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    loose = project.add_file("loader", None, file_id="33333333-cccc-4ccc-8ccc-cccccccccccc")
    return manager, project, types.SimpleNamespace(fw=fw, app=app, app_db=app_db, loose=loose)


def test_resolve_file_by_full_id(populated):
    manager, _, f = populated
    assert manager.resolve_file(f.app.id) is f.app


def test_resolve_file_by_path_with_and_without_leading_slash(populated):
    manager, _, f = populated
    assert manager.resolve_file("/fw/app.bndb") is f.app_db
    assert manager.resolve_file("fw/app.bndb") is f.app_db
    assert manager.resolve_file("loader") is f.loose


def test_resolve_file_by_unique_bare_name(populated):
    manager, _, f = populated
    assert manager.resolve_file("app") is f.app


def test_resolve_file_ambiguous_path_lists_both_ids(populated):
    manager, project, f = populated
    twin = project.add_file("app", f.fw, file_id="44444444-dddd-4ddd-8ddd-dddddddddddd")
    with pytest.raises(RuntimeError) as exc:
        manager.resolve_file("fw/app")
    text = str(exc.value)
    assert text.startswith("Ambiguous project file selector: 'fw/app' matches 2 files")
    assert f"  {f.app.id}  /fw/app" in text and f"  {twin.id}  /fw/app" in text


def test_resolve_file_id_prefix_needs_eight_characters(populated):
    manager, _, f = populated
    assert manager.resolve_file("11111111") is f.app
    assert manager.resolve_file("22222222-bb") is f.app_db
    with pytest.raises(RuntimeError, match="No project file matches '1111111'"):
        manager.resolve_file("1111111")


def test_resolve_file_unknown_selector_lists_every_file(populated):
    manager, _, f = populated
    with pytest.raises(RuntimeError) as exc:
        manager.resolve_file("nope")
    lines = str(exc.value).splitlines()
    assert lines[0] == "No project file matches 'nope'. Project files:"
    assert set(lines[1:]) == {
        f"  {f.app.id}  /fw/app",
        f"  {f.app_db.id}  /fw/app.bndb",
        f"  {f.loose.id}  /loader",
    }


def test_resolve_file_rejects_non_string(populated):
    manager, _, _ = populated
    with pytest.raises(RuntimeError, match="non-empty string"):
        manager.resolve_file(None)


# --- sibling database -------------------------------------------------------


def test_sibling_database_found_for_raw_file(populated):
    manager, _, f = populated
    assert manager.sibling_database(f.app) is f.app_db


def test_sibling_database_is_none_for_a_bndb_itself(populated):
    manager, _, f = populated
    assert manager.sibling_database(f.app_db) is None


def test_sibling_database_is_none_when_two_siblings_share_the_name(populated):
    manager, project, f = populated
    project.add_file("app.bndb", f.fw)
    assert manager.sibling_database(f.app) is None


def test_sibling_database_ignores_a_same_named_file_in_another_folder(populated):
    manager, project, f = populated
    project.add_file("loader.bndb", f.fw)
    assert manager.sibling_database(f.loose) is None


# --- file rows ---------------------------------------------------------------


def test_file_row_reports_kind_size_open_target_and_sibling(populated):
    manager, _, f = populated
    raw = manager.file_row(f.app, {f.app.id: "17"})
    assert raw["is_database"] is False
    assert raw["size"] == 4 and raw["exists_on_disk"] is True
    assert raw["open_target_id"] == "17"
    assert raw["sibling_database_id"] == f.app_db.id
    assert raw["path"] == "/fw/app" and raw["folder"] == "/fw" and raw["name"] == "app"

    db = manager.file_row(f.app_db)
    assert db["is_database"] is True
    assert db["size"] == len(SQLITE_MAGIC) + 8
    assert db["open_target_id"] is None and db["sibling_database_id"] is None


def test_file_row_for_a_file_with_no_blob(populated):
    manager, project, _ = populated
    ghost = project.add_file("ghost", None, data=None)
    row = manager.file_row(ghost)
    assert row["exists_on_disk"] is False
    assert row["size"] is None and row["is_database"] is False
    assert row["folder"] == ""


# --- folders -----------------------------------------------------------------


def test_ensure_folder_creates_nested_folders_once_and_reuses_them(populated):
    manager, project, f = populated
    before = len(project.folders)
    leaf = manager.ensure_folder("fw/boot/stage2")
    assert manager.folder_path(leaf) == "/fw/boot/stage2"
    assert leaf.parent.parent is f.fw
    assert len(project.folders) == before + 2
    assert manager.ensure_folder("/fw/boot/stage2/") is leaf
    assert len(project.folders) == before + 2
    assert manager.ensure_folder(None) is None and manager.ensure_folder("/") is None


# --- save (project branch) ---------------------------------------------------


def test_save_project_database_refuses_an_explicit_path(po, populated):
    manager, _, f = populated
    bridge = _bridge(manager)
    with pytest.raises(RuntimeError, match="bn save takes no output path"):
        po.save_project_database(bridge, SaveBV(f.app), "/tmp/out.bndb")


def test_save_project_database_snapshots_a_database_backed_view(po, populated):
    manager, _, f = populated
    targets = FakeTargets()
    bridge = _bridge(manager, targets)
    bv = SaveBV(f.app_db, has_database=True)
    result = po.save_project_database(bridge, bv, None)
    assert result == {
        "ok": True, "saved": True, "path": f.app_db.path_on_disk,
        "project_file_id": f.app_db.id, "project_path": "/fw/app.bndb",
        "mode": "snapshot",
    }
    assert targets.cleared == [bv] and targets.project_databases == []


def test_save_project_database_snapshot_false_is_an_error(po, populated):
    manager, _, f = populated
    bridge = _bridge(manager)
    with pytest.raises(RuntimeError, match="could not write the project database snapshot"):
        po.save_project_database(bridge, SaveBV(f.app_db, has_database=True, snapshot_ok=False), None)


def test_save_project_database_snapshot_uses_the_recorded_binding(po, populated):
    # BN keeps naming the RAW file; the bridge's binding names the database.
    manager, _, f = populated
    targets = FakeTargets(binding=(f.app_db.id, "/fw/app.bndb"))
    bridge = _bridge(manager, targets)
    bv = SaveBV(f.app, has_database=True)
    result = po.save_project_database(bridge, bv, None)
    assert result["project_file_id"] == f.app_db.id
    assert result["project_path"] == "/fw/app.bndb"


def test_save_project_database_refuses_when_a_sibling_database_exists(po, populated):
    manager, project, f = populated
    bridge = _bridge(manager)
    with pytest.raises(RuntimeError) as exc:
        po.save_project_database(bridge, SaveBV(f.app), None)
    text = str(exc.value)
    assert "/fw/app.bndb" in text and f.app_db.id in text
    assert "load that file instead" in text
    assert len(project.files) == 3


def test_save_project_database_creates_the_tracked_database(po, populated):
    manager, project, f = populated
    targets = FakeTargets()
    bridge = _bridge(manager, targets)
    bv = SaveBV(f.loose)
    result = po.save_project_database(bridge, bv, None)

    new = manager.resolve_file("/loader.bndb")
    assert new.folder is None and new is not f.loose
    assert bv.created_with == new.path_on_disk
    assert bv.parent_existed is True, "the blob's parent dir exists before create_database"
    assert Path(new.path_on_disk).exists()
    assert result == {
        "ok": True, "saved": True, "path": new.path_on_disk,
        "project_file_id": new.id, "project_path": "/loader.bndb", "mode": "created",
    }
    assert targets.cleared == [bv]
    assert targets.databases == [(bv, new.path_on_disk)]
    assert targets.project_databases == [(bv, new.id, "/loader.bndb")]
    assert bridge.registry_writes == 1
    assert manager.sibling_database(f.loose) is new


def test_save_project_database_create_false_deletes_the_new_file_and_errors(po, populated):
    manager, project, f = populated
    bridge = _bridge(manager)
    bv = SaveBV(f.loose, create_result=False)
    with pytest.raises(RuntimeError, match="Failed to save project database"):
        po.save_project_database(bridge, bv, None)
    assert [pf.name for pf in project.deleted] == ["loader.bndb"]
    assert [pf.name for pf in project.files] == ["app", "app.bndb", "loader"]
    assert bridge.registry_writes == 0


# --- delete / export / import / files ----------------------------------------


def _bridge_with_rows(manager, rows):
    bridge = _bridge(manager)
    bridge.targets.refresh = lambda: rows
    return bridge


def test_project_delete_refuses_an_open_target(po, populated):
    manager, project, f = populated
    bridge = _bridge_with_rows(manager, [
        {"target_id": "t-9", "selector": "fw/app.bndb", "project_file_id": f.app_db.id},
    ])
    with pytest.raises(RuntimeError, match=r"/fw/app.bndb is open as target fw/app.bndb; close it first"):
        po.project_delete(bridge, "fw/app.bndb")
    assert f.app_db in project.files


def test_project_delete_removes_a_closed_file(po, populated):
    manager, project, f = populated
    bridge = _bridge_with_rows(manager, [])
    result = po.project_delete(bridge, "fw/app")
    assert result == {"deleted": True, "project_file_id": f.app.id, "project_path": "/fw/app"}
    assert f.app not in project.files and bridge.registry_writes == 1


def test_project_export_refuses_an_existing_destination(po, populated, tmp_path):
    manager, _, f = populated
    bridge = _bridge(manager)
    dest = tmp_path / "already_here"
    dest.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="already exists"):
        po.project_export(bridge, "fw/app", str(dest))
    assert dest.read_bytes() == b"x"


def test_project_export_writes_the_blob(po, populated, tmp_path):
    manager, _, f = populated
    bridge = _bridge(manager)
    dest = tmp_path / "out" / "app.bin"
    dest.parent.mkdir()
    result = po.project_export(bridge, "fw/app", str(dest))
    assert result == {"exported": True, "dest": str(dest.resolve()),
                      "project_file_id": f.app.id, "project_path": "/fw/app"}
    assert dest.read_bytes() == b"\x7fELF"


def test_project_import_creates_the_folder_and_returns_a_row(po, populated, tmp_path):
    manager, project, _ = populated
    src = tmp_path / "blob.bin"
    src.write_bytes(b"\x7fELF-imported")
    bridge = _bridge_with_rows(manager, [])
    row = po.project_import(bridge, str(src), folder="fw/new", description="d")
    assert row["path"] == "/fw/new/blob.bin" and row["folder"] == "/fw/new"
    assert row["size"] == len(b"\x7fELF-imported") and row["description"] == "d"
    assert manager.resolve_file("/fw/new/blob.bin").id == row["id"]
    assert bridge.registry_writes == 1


def test_project_import_refuses_a_missing_source(po, populated, tmp_path):
    manager, _, _ = populated
    with pytest.raises(RuntimeError, match="not a file"):
        po.project_import(_bridge(manager), str(tmp_path / "missing"))


def test_project_files_filters_by_folder_and_recursion(po, populated):
    manager, project, f = populated
    deep = project.add_folder("boot", f.fw)
    project.add_file("stage2", deep)
    bridge = _bridge_with_rows(manager, [
        {"target_id": "t-1", "project_file_id": f.app.id, "selector": "fw/app"},
    ])
    everything = po.project_files(bridge)
    assert [r["path"] for r in everything["files"]] == [
        "/fw/app", "/fw/app.bndb", "/fw/boot/stage2", "/loader"]
    assert [r["path"] for r in everything["folders"]] == ["/fw", "/fw/boot"]
    assert next(r for r in everything["files"] if r["path"] == "/fw/app")["open_target_id"] == "t-1"

    shallow = po.project_files(bridge, "fw", recursive=False)
    assert [r["path"] for r in shallow["files"]] == ["/fw/app", "/fw/app.bndb"]
    root_only = po.project_files(bridge, None, recursive=False)
    assert [r["path"] for r in root_only["files"]] == ["/loader"]


def test_project_info_and_open_handler(po, bnpr, monkeypatch):
    project = FakeProject(bnpr, name="demo")
    project.add_folder("fw")
    project.add_file("app")
    sys.modules["binaryninja"].Project = types.SimpleNamespace(open_project=lambda p: project)
    monkeypatch.setattr(po, "find_other_owner", lambda path, *, self_pid: None)
    bridge = _bridge_with_rows(po.ProjectManager(), [])
    with pytest.raises(RuntimeError, match="No project is open in this bridge"):
        po.project_info(bridge)

    info = po.project_open(bridge, str(bnpr))
    assert info["opened"] is True and bridge.registry_writes == 1
    assert info["path"] == str(bnpr.resolve()) and info["id"] == project.id
    assert info["name"] == "demo" and info["file_count"] == 1 and info["folder_count"] == 1
    assert info["open_targets"] == []
    assert po.project_open(bridge, str(bnpr / "project.bnpm"))["opened"] is False
    with pytest.raises(RuntimeError, match="path must be the .bnpr directory"):
        po.project_open(bridge, "")
