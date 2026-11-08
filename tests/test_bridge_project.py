"""Project targets through `TargetManager` and `BinaryNinjaBridge`: rows,
selectors, the `save_database` dispatch, `_project_load`, `_project_close` and
the registry payload, against fake views and the fake project model in
`_project_fakes`.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
import sys
import types
from pathlib import Path

import pytest

from _bridge_fakes import _SaveBV, _load_bridge, _register_views
from _project_fakes import SQLITE_MAGIC, FakeProject, FakeProjectFile


class FakeView:
    """A loaded view. *pf* is its `ProjectFile` (None for a standalone view)."""

    _sessions = iter(range(1000, 10_000))

    def __init__(self, filename: str, pf=None, *, has_database: bool = False,
                 analysis_error: Exception | None = None):
        self.project_file = pf
        self.closed = False
        self.analyzed = False
        self.created_with: str | None = None
        self._analysis_error = analysis_error
        self.file = types.SimpleNamespace(
            session_id=str(next(self._sessions)),
            filename=filename,
            modified=False,
            has_database=has_database,
            close=lambda: setattr(self, "closed", True),
            save_auto_snapshot=lambda: True,
        )
        self.view_type = types.SimpleNamespace(name="ELF")

    def update_analysis_and_wait(self):
        if self._analysis_error is not None:
            raise self._analysis_error
        self.analyzed = True

    def create_database(self, dest: str) -> bool:
        self.created_with = dest
        Path(dest).write_bytes(SQLITE_MAGIC + b"\x00" * 8)
        return True


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A bridge instance with the fake project `demo.bnpr` open in it.

    Files: `/fw/app` (raw) and `/fw/app.bndb` (its database), `/boot/app`
    (raw, no database), `/loader` (raw, at the root).
    """
    module = _load_bridge(monkeypatch)
    fake_bn = sys.modules["binaryninja"]
    fake_bn.ProjectFile = FakeProjectFile
    root = tmp_path / "demo.bnpr"
    root.mkdir()
    (root / "project.bnpm").write_bytes(SQLITE_MAGIC)
    project = FakeProject(root, name="demo")
    fw = project.add_folder("fw")
    boot = project.add_folder("boot")
    files = types.SimpleNamespace(
        app=project.add_file("app", fw),
        app_db=project.add_file("app.bndb", fw, data=SQLITE_MAGIC + b"\x00" * 8),
        boot_app=project.add_file("app", boot),
        loader=project.add_file("loader"),
    )
    fake_bn.Project = types.SimpleNamespace(open_project=lambda path: project)
    monkeypatch.setattr(module.project_ops, "find_other_owner",
                        lambda path, *, self_pid: None)
    instance = module.BinaryNinjaBridge(instance_id="proj1")
    instance.projects.open(str(root), self_pid=os.getpid())
    _register_views(module)
    module._load_in_progress.clear()
    return types.SimpleNamespace(
        module=module, bn=fake_bn, instance=instance, project=project,
        files=files, root=root, targets=instance.targets)


def _rows(world, *views):
    _register_views(world.module, *views)
    return world.targets.refresh()


def _row_for(rows, bv_filename):
    return next(r for r in rows if r["filename"] == bv_filename)


# --- rows --------------------------------------------------------------------


def test_refresh_rows_carry_project_fields_only_for_project_views(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    plain = FakeView("/opt/firmware/plain.bin")
    rows = _rows(world, proj, plain)

    row = _row_for(rows, proj.file.filename)
    assert row["project_file_id"] == f.app_db.id
    assert row["project_path"] == "/fw/app.bndb"
    assert row["project_name"] == "app.bndb"

    other = _row_for(rows, plain.file.filename)
    assert (other["project_file_id"], other["project_path"], other["project_name"]) == ("", "", "")


def test_a_project_file_from_a_foreign_class_is_not_a_project_target(world):
    # `_project_file_of` judges by BN's class: a duck-typed attribute (a
    # permissive wrapper, a stale fake) must not divert selectors or saves.
    duck = types.SimpleNamespace(id="x", name="app.bndb", get_path_in_project=lambda: "/x")
    bv = FakeView("/opt/firmware/duck", duck)
    row = _rows(world, bv)[0]
    assert row["project_file_id"] == "" and row["project_path"] == ""


def test_note_project_database_overrides_what_bn_reports(world):
    # After the first save BN still names the RAW file; the binding wins.
    f = world.files
    bv = FakeView(f.loader.path_on_disk, f.loader)
    rows = _rows(world, bv)
    assert rows[0]["project_path"] == "/loader"

    world.targets.note_project_database(bv, "new-db-id", "/loader.bndb")
    row = world.targets.refresh()[0]
    assert row["project_file_id"] == "new-db-id"
    assert row["project_path"] == "/loader.bndb"
    assert row["project_name"] == "loader.bndb"
    assert row["selector"] == "loader.bndb"
    assert world.targets.resolve("loader.bndb") is bv
    assert world.targets.resolve("loader") is bv       # `.bndb`-stripped alias
    assert world.targets.project_binding(bv) == ("new-db-id", "/loader.bndb")


def test_binding_is_pruned_when_the_view_leaves_a_complete_snapshot(world):
    f = world.files
    bv = FakeView(f.loader.path_on_disk, f.loader)
    _rows(world, bv)
    world.targets.note_project_database(bv, "new-db-id", "/loader.bndb")

    survivor = FakeView("/opt/firmware/other.bin")
    _rows(world, survivor)                      # bv gone from a COMPLETE walk
    assert world.targets._project_bindings == {}
    # Reappearing is a NEW target: BN's own answer, not the stale binding.
    row = _row_for(_rows(world, survivor, bv), bv.file.filename)
    assert row["project_path"] == "/loader" and row["project_file_id"] == f.loader.id


def test_forget_drops_the_binding(world):
    f = world.files
    bv = FakeView(f.loader.path_on_disk, f.loader)
    _rows(world, bv)
    world.targets.note_project_database(bv, "new-db-id", "/loader.bndb")
    world.targets.forget(bv)
    assert world.targets._project_bindings == {}
    assert world.targets.project_binding(bv) is None


def test_succeed_carries_the_binding_to_the_next_manager(world):
    f = world.files
    bv = FakeView(f.loader.path_on_disk, f.loader)
    _rows(world, bv)
    world.targets.note_project_database(bv, "new-db-id", "/loader.bndb")

    successor = world.module.TargetManager()
    successor.succeed(world.targets)
    row = successor.refresh()[0]
    assert row["project_file_id"] == "new-db-id" and row["project_path"] == "/loader.bndb"
    assert successor.project_binding(bv) == ("new-db-id", "/loader.bndb")


# --- selectors ---------------------------------------------------------------


@pytest.mark.parametrize("selector_of", [
    lambda f: "fw/app.bndb",
    lambda f: "/fw/app.bndb",
    lambda f: "app.bndb",
    lambda f: "app",
    lambda f: f.app_db.id,
])
def test_project_selectors_resolve_the_project_view(world, selector_of):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    plain = FakeView("/opt/firmware/plain.bin")
    _rows(world, proj, plain)
    assert world.targets.resolve(selector_of(f)) is proj


def test_blob_filename_still_resolves_but_is_not_advertised(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    rows = _rows(world, proj)
    assert world.targets.resolve(f.app_db.path_on_disk) is proj
    assert rows[0]["selector"] == "app.bndb"


def test_pinned_target_matches_only_the_exact_target_id(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    rows = _rows(world, proj)
    pinned = world.module.PinnedTarget
    assert world.targets.resolve(pinned(rows[0]["target_id"])) is proj
    for alias in ("fw/app.bndb", "app.bndb", f.app_db.id, rows[0]["view_id"]):
        with pytest.raises(RuntimeError):
            world.targets.resolve(pinned(alias))


def test_no_id_prefix_matching_in_dash_t(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    _rows(world, proj)
    with pytest.raises(RuntimeError):
        world.targets.resolve(f.app_db.id[:8])


def test_bare_name_shared_with_a_standalone_bndb_is_ambiguous(world, tmp_path):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    standalone = FakeView(str(tmp_path / "elsewhere" / "app.bndb"))
    _rows(world, proj, standalone)
    with pytest.raises(RuntimeError, match="Ambiguous target selector: 'app' matches 2 targets"):
        world.targets.resolve("app")
    with pytest.raises(RuntimeError, match="Ambiguous target selector: 'app.bndb'"):
        world.targets.resolve("app.bndb")
    assert world.targets.resolve("fw/app.bndb") is proj
    assert world.targets.resolve("elsewhere/app.bndb") is standalone


def test_compute_selectors_advertises_the_path_in_project(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    rows = _rows(world, proj)
    assert rows[0]["selector"] == "app.bndb"
    assert "/data/" not in rows[0]["selector"]


def test_compute_selectors_disambiguates_two_project_files_with_one_name(world):
    f = world.files
    fw_app = FakeView(f.app.path_on_disk, f.app)
    boot_app = FakeView(f.boot_app.path_on_disk, f.boot_app)
    rows = _rows(world, fw_app, boot_app)
    by_path = {r["project_path"]: r["selector"] for r in rows}
    assert by_path == {"/fw/app": "fw/app", "/boot/app": "boot/app"}
    assert world.targets.resolve("boot/app") is boot_app
    with pytest.raises(RuntimeError, match="Ambiguous"):
        world.targets.resolve("app")


# --- save_database dispatch --------------------------------------------------


def test_save_database_takes_the_project_branch_for_a_project_view(world):
    f = world.files
    bv = FakeView(f.loader.path_on_disk, f.loader)
    _rows(world, bv)
    world.targets.mark_dirty(bv)

    result = world.instance._save_database("loader")

    new = world.instance.projects.resolve_file("/loader.bndb")
    assert result == {
        "ok": True, "saved": True, "path": new.path_on_disk,
        "project_file_id": new.id, "project_path": "/loader.bndb", "mode": "created",
    }
    assert bv.created_with == new.path_on_disk
    assert str(world.root) in new.path_on_disk, "the database is a blob INSIDE the project"
    assert not Path(f.loader.path_on_disk + ".bndb").exists(), "no stray untracked file"
    row = world.targets.refresh()[0]
    assert row["unsaved"] is False
    assert row["project_path"] == "/loader.bndb" and row["selector"] == "loader.bndb"
    assert row["database_path"] == new.path_on_disk


def test_save_database_refuses_a_path_for_a_project_view(world):
    f = world.files
    bv = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    _rows(world, bv)
    with pytest.raises(RuntimeError, match="bn save takes no output path"):
        world.instance._save_database("app.bndb", "/tmp/somewhere.bndb")


def test_save_database_snapshots_a_database_backed_project_view(world):
    f = world.files
    bv = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    _rows(world, bv)
    world.targets.mark_dirty(bv)
    result = world.instance._save_database("fw/app.bndb")
    assert result["mode"] == "snapshot" and result["project_file_id"] == f.app_db.id
    assert bv.created_with is None
    assert world.targets.refresh()[0]["unsaved"] is False


def test_save_database_keeps_the_standalone_path_for_a_plain_view(world, tmp_path):
    raw = tmp_path / "plain_bin"
    raw.write_bytes(b"\x7fELF")
    bv = _SaveBV(str(raw), result=True, write=True)
    _rows(world, bv)
    result = world.instance._save_database("plain_bin")
    assert bv.created_with == str(raw) + ".bndb"
    assert (tmp_path / "plain_bin.bndb").exists()
    assert result["saved"] is True and "project_file_id" not in result
    assert len(world.project.files) == 4, "a plain save never touches the project"


# --- _project_load -----------------------------------------------------------


def _install_load(world, factory):
    calls: list = []

    def fake_load(pf, update_analysis=True):
        calls.append(pf)
        return factory(pf)

    world.bn.load = fake_load
    return calls


def test_project_load_prefers_the_sibling_database(world):
    f = world.files
    calls = _install_load(world, lambda pf: FakeView(pf.path_on_disk, pf, has_database=True))
    result = world.instance._project_load("fw/app")
    assert calls == [f.app_db]
    assert result["loaded"] is True
    assert result["project_file_id"] == f.app_db.id
    assert result["project_path"] == "/fw/app.bndb"
    assert result["substituted_from"] == f.app.id
    assert result["analysis_state"] == "full"
    assert result["target"]["selector"] == "app.bndb"
    assert any("--no-bndb" in note for note in result["notes"])
    assert world.module._headless_views[0].file.filename == f.app_db.path_on_disk


def test_project_load_raw_when_prefer_bndb_is_false(world):
    f = world.files
    calls = _install_load(world, lambda pf: FakeView(pf.path_on_disk, pf))
    result = world.instance._project_load("fw/app", prefer_bndb=False)
    assert calls == [f.app]
    assert result["substituted_from"] is None
    assert result["project_path"] == "/fw/app"
    assert world.module._headless_views[0].analyzed is True
    assert result["target"]["selector"] == "app"


def test_project_load_quick_skips_analysis_for_a_raw_file(world):
    f = world.files
    _install_load(world, lambda pf: FakeView(pf.path_on_disk, pf))
    result = world.instance._project_load("loader", quick=True)
    assert result["analysis_state"] == "quick" and result["analyzed"] is False
    assert world.module._headless_views[0].analyzed is False
    assert world.module._headless_views[0] in world.module._quick_loaded_views


def test_project_load_dedups_by_file_id(world):
    f = world.files
    already = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    _rows(world, already)
    calls = _install_load(world, lambda pf: pytest.fail("must not load again"))
    for selector in ("fw/app", "fw/app.bndb", f.app_db.id):
        result = world.instance._project_load(selector)
        assert result["loaded"] is False and result["already_open"] is True
        assert result["project_file_id"] == f.app_db.id
        assert result["target"]["selector"] == "app.bndb"
    assert calls == []
    assert world.instance._project_load("fw/app")["substituted_from"] == f.app.id


@pytest.mark.parametrize("updates_live_filename", [False, True])
@pytest.mark.parametrize("selector_kind", ["raw", "database_path", "database_id"])
def test_project_reload_after_first_save_reuses_live_view(world, selector_kind, updates_live_filename):
    raw = world.files.loader
    bv = FakeView(raw.path_on_disk, raw)
    original_save = bv.create_database

    def native_save(dest):
        result = original_save(dest)
        bv.file.has_database = True
        if updates_live_filename:
            bv.file.filename = dest
        return result

    bv.create_database = native_save
    calls = _install_load(world, lambda pf: bv)
    first = world.instance._project_load(raw.id, quick=True)
    saved = world.instance._save_database("loader")
    assert bv.project_file is raw  # BN retains this stale native association.
    selector = {
        "raw": raw.id,
        "database_path": saved["project_path"],
        "database_id": saved["project_file_id"],
    }[selector_kind]
    again = world.instance._project_load(selector)
    assert again["already_open"] and not again["loaded"]
    assert again["target"]["target_id"] == first["target"]["target_id"]
    assert calls == [raw] and world.module._headless_views == [bv]


def test_project_load_refuses_ambiguous_siblings_before_native_open(world):
    duplicate = world.project.add_file("app.bndb", world.files.app.folder)
    calls = _install_load(world, lambda pf: pytest.fail("must refuse before native open"))
    with pytest.raises(RuntimeError, match="Ambiguous") as error:
        world.instance._project_load(world.files.app.id)
    assert duplicate.id in str(error.value) and world.files.app_db.id in str(error.value)
    assert not calls


def test_project_snapshot_refuses_duplicate_writer_and_names_target(world):
    pf = world.files.app_db
    first = FakeView(pf.path_on_disk, pf, has_database=True)
    second = FakeView(pf.path_on_disk, pf, has_database=True)
    rows = _rows(world, first, second)
    snapshots = []
    first.file.save_auto_snapshot = lambda: snapshots.append("saved") or True
    second.file.save_auto_snapshot = lambda: snapshots.append("duplicate") or True
    original_blob = Path(pf.path_on_disk).read_bytes()
    for view, other in ((first, second), (second, first)):
        current = world.instance._loaded_target_row(rows, view)
        duplicate = world.instance._loaded_target_row(rows, other)
        world.targets.mark_dirty(view)
        with pytest.raises(RuntimeError, match="another open target") as error:
            world.instance._save_database(world.module.PinnedTarget(current["target_id"]))
        assert duplicate["target_id"] in str(error.value)
        assert duplicate["selector"] in str(error.value)
        assert not snapshots and world.targets.is_dirty(view)
        assert Path(pf.path_on_disk).read_bytes() == original_blob
    _rows(world, first)
    assert world.instance._save_database("app.bndb")["mode"] == "snapshot"
    assert snapshots == ["saved"] and not world.targets.is_dirty(first)


def test_raw_save_names_existing_database_target(world):
    raw = FakeView(world.files.app.path_on_disk, world.files.app)
    saved = FakeView(world.files.app_db.path_on_disk, world.files.app_db, has_database=True)
    rows = _rows(world, raw, saved)
    raw_row = world.instance._loaded_target_row(rows, raw)
    saved_row = world.instance._loaded_target_row(rows, saved)
    with pytest.raises(RuntimeError, match="already has a database") as error:
        world.instance._save_database(world.module.PinnedTarget(raw_row["target_id"]))
    assert saved_row["target_id"] in str(error.value)
    assert saved_row["selector"] in str(error.value)
    assert raw.created_with is None


@pytest.mark.parametrize("alias_kind", ["direct", "file_symlink", "directory_symlink", "outward_symlink"])
def test_standalone_load_refuses_project_storage_across_cache_owners(
    world, tmp_path, monkeypatch, alias_kind,
):
    blob = Path(world.files.app_db.path_on_disk)
    if alias_kind == "file_symlink":
        path = tmp_path / "alias.bndb"
        path.symlink_to(blob)
    elif alias_kind == "directory_symlink":
        alias = tmp_path / "alias-directory"
        alias.symlink_to(blob.parent, target_is_directory=True)
        path = alias / blob.name
    elif alias_kind == "outward_symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"synthetic")
        path = world.root / "outside-link"
        path.symlink_to(outside)
    else:
        path = blob
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="bn-raw-", dir="/tmp") as cache:
        monkeypatch.setenv("BN_CACHE_DIR", cache)
        other = world.module.BinaryNinjaBridge(instance_id="other-owner")
        calls = _install_load(world, lambda pf: pytest.fail("must refuse before native open"))
        with pytest.raises(RuntimeError, match="Cannot load project storage") as error:
            other._load_binary(str(path), prefer_bndb=False)
        assert "bn project load" in str(error.value)
        assert not calls and other.projects.current is None
        assert world.instance.projects.current is not None


def test_project_load_closes_an_unpublished_view_on_failure(world):
    # #609 parity: a view BN opened but the bridge never published would leak.
    f = world.files
    opened: list[FakeView] = []

    def factory(pf):
        bv = FakeView(pf.path_on_disk, pf, analysis_error=RuntimeError("analysis blew up"))
        opened.append(bv)
        return bv

    _install_load(world, factory)
    with pytest.raises(RuntimeError, match="analysis blew up"):
        world.instance._project_load("loader")
    assert len(opened) == 1 and opened[0].closed is True
    assert world.module._headless_views == []
    assert world.targets.refresh() == []
    assert world.module._load_in_progress == {}


def test_project_load_wraps_a_bn_open_failure(world):
    def factory(pf):
        raise OSError("blob unreadable")

    _install_load(world, factory)
    with pytest.raises(RuntimeError, match="Unable to open project file /loader: blob unreadable"):
        world.instance._project_load("loader")


def test_project_load_requires_an_open_project(world):
    world.instance.projects.close()
    with pytest.raises(RuntimeError, match="No project is open"):
        world.instance._project_load("loader")



@pytest.mark.parametrize("reopen", [False, True])
def test_project_load_rejects_project_close_or_swap_during_analysis(world, monkeypatch, reopen):
    entered = threading.Event()
    resume = threading.Event()
    bv = FakeView(world.files.loader.path_on_disk, world.files.loader)
    failures = []
    linked = []

    def analyze():
        assert not world.instance._write_gate.locked()
        entered.set()
        assert resume.wait(5)

    bv.update_analysis_and_wait = analyze
    _install_load(world, lambda pf: bv)
    monkeypatch.setattr(world.module.project_rootfs, "link_view",
                        lambda *a, **kw: linked.append(True))

    def load():
        try:
            world.instance._project_load("loader")
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=load)
    worker.start()
    try:
        assert entered.wait(5)
        with world.instance._write_operation_lock():
            world.instance._project_close()
            if reopen:
                world.instance.projects.open(str(world.root), self_pid=os.getpid())
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(failures) == 1 and "Project changed while loading" in str(failures[0])
    assert bv.closed and not linked
    assert world.module._headless_views == []
    assert world.module._load_in_progress == {}


def test_project_load_revalidates_before_native_open(world, monkeypatch):
    original = world.instance._write_operation_lock
    entries = 0
    calls = _install_load(world, lambda pf: pytest.fail("must not open a stale project file"))

    @contextmanager
    def swap_before_open():
        nonlocal entries
        with original():
            entries += 1
            if entries == 3:
                world.instance.projects.close()
            yield

    monkeypatch.setattr(world.instance, "_write_operation_lock", swap_before_open)
    with pytest.raises(RuntimeError, match="Project changed while loading"):
        world.instance._project_load("loader")
    assert not calls and world.module._load_in_progress == {}


def test_project_load_does_not_dedup_foreign_project_file_with_same_id(world, tmp_path):
    foreign = FakeProject(tmp_path / "foreign.bnpr")
    foreign_pf = foreign.add_file("loader", file_id=world.files.loader.id)
    foreign_bv = FakeView(foreign_pf.path_on_disk, foreign_pf)
    _rows(world, foreign_bv)
    calls = _install_load(world, lambda pf: FakeView(pf.path_on_disk, pf))
    result = world.instance._project_load("loader")
    assert result["loaded"] and calls == [world.files.loader]
    assert not foreign_bv.closed

# --- _project_close ----------------------------------------------------------


def test_project_close_refuses_dirty_targets_unless_forced(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    raw = FakeView(f.loader.path_on_disk, f.loader)
    plain = FakeView("/opt/firmware/plain.bin")
    _rows(world, proj, raw, plain)
    world.targets.mark_dirty(proj)

    with pytest.raises(RuntimeError) as refused:
        world.instance._project_close()
    text = str(refused.value)
    assert "project demo has unsaved targets: app.bndb" in text
    assert "--force" in text
    assert world.instance.projects.current is not None
    assert not proj.closed

    result = world.instance._project_close(force=True)
    assert result["closed"] is True and result["path"] == str(world.root.resolve())
    assert sorted(r["path"] for r in result["targets_closed"]) == sorted(
        [proj.file.filename, raw.file.filename])
    assert next(r for r in result["targets_closed"] if r["path"] == proj.file.filename)["unsaved"] is True
    assert proj.closed and raw.closed and not plain.closed
    assert world.module._headless_views == [plain]
    assert world.project.closed is True
    assert world.instance.projects.current is None
    payload = json.loads(world.instance.registry_path.read_text(encoding="utf-8"))
    assert payload["project"] is None and payload["project_targets"] == []


def test_project_close_with_clean_targets_needs_no_force(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    _rows(world, proj)
    result = world.instance._project_close()
    assert [r["unsaved"] for r in result["targets_closed"]] == [False]
    assert proj.closed


# --- registry payload --------------------------------------------------------


def test_registry_payload_publishes_the_project_and_its_targets(world):
    f = world.files
    proj = FakeView(f.app_db.path_on_disk, f.app_db, has_database=True)
    plain = FakeView("/opt/firmware/plain.bin")
    rows = _rows(world, proj, plain)
    world.instance._write_registry()

    payload = json.loads(world.instance.registry_path.read_text(encoding="utf-8"))
    assert payload["project"] == {
        "path": str(world.root.resolve()), "id": world.project.id, "name": "demo"}
    proj_row = _row_for(rows, proj.file.filename)
    assert payload["project_targets"] == [{
        "target_id": proj_row["target_id"],
        "view_id": proj_row["view_id"],
        "project_file_id": f.app_db.id,
        "project_path": "/fw/app.bndb",
    }]
    assert sorted(payload["binaries"]) == sorted([proj.file.filename, plain.file.filename])


def test_registry_payload_without_a_project(monkeypatch, tmp_path):
    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge(instance_id="noproj")
    _register_views(module)
    instance._write_registry()
    payload = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert payload["project"] is None and payload["project_targets"] == []


def test_failed_first_save_dispatch_discloses_primary_error_and_residue(world):
    bv = FakeView(world.files.loader.path_on_disk, world.files.loader)
    _rows(world, bv)
    world.targets.mark_dirty(bv)

    def incomplete_save(dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("database write interrupted")

    bv.create_database = incomplete_save
    world.project.delete_file = lambda pf: False
    response = world.instance.dispatch({
        "op": "save_database", "params": {}, "target": "loader",
    })
    assert response["ok"] is False and response["status"] == "rollback_failed"
    residue = response["observed"]
    assert "database write interrupted" in residue["primary_error"]
    assert residue["rolled_back"] is False
    assert residue["row_remains"] is True and residue["blob_remains"] is True
    leftover = world.instance.projects.resolve_file(residue["project_file_id"])
    assert leftover.path_on_disk == residue["path"]
    assert Path(residue["path"]).read_bytes() == b"partial"
    assert world.targets.is_dirty(bv)


@pytest.mark.parametrize("failure", ["verification_failed", "internal_error", "rollback_failed"])
def test_project_load_retains_link_failure_and_loaded_target(world, monkeypatch, failure):
    bv = FakeView(world.files.loader.path_on_disk, world.files.loader)
    _install_load(world, lambda pf: bv)
    monkeypatch.setattr(world.instance.projects, "get_rootfs", lambda: {"libs": [{}]})

    def failing_link(*args, **kwargs):
        if failure == "internal_error":
            raise ValueError("linking interrupted")
        if failure == "rollback_failed":
            raise world.module.OperationFailure(
                failure, "link rollback failed", observed={"rolled_back": False},
            )
        return {"success": False, "status": failure, "notes": ["link readback failed"]}

    monkeypatch.setattr(world.module.project_rootfs, "link_view", failing_link)
    result = world.instance._project_load(world.files.loader.id, quick=True)
    assert result["loaded"] is True and result["target"] is not None
    assert result["link"]["success"] is False
    assert result["link"]["status"] == failure
    assert result["target"] in result["targets"]
    assert world.module._headless_views == [bv] and not bv.closed
    if failure == "rollback_failed":
        assert result["link"]["observed"]["rolled_back"] is False
    if failure == "internal_error":
        assert "linking interrupted" in result["link"]["error"]
