"""Firmware library inventory and linking against duck-typed Binary Ninja fakes."""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from _bridge_fakes import _load_bridge, _register_views
from _project_fakes import FakeProject, FakeProjectFile, SQLITE_MAGIC
from test_bridge_project import FakeView


class RootfsProject(FakeProject):
    def __init__(self, root, name="firmware"):
        super().__init__(root, name)
        self.metadata = {}
        self.writes = []
        self.bulk_entries = 0
        self.in_bulk = False

    def query_metadata(self, key):
        return self.metadata[key]

    def store_metadata(self, key, value):
        self.writes.append((key, value))
        self.metadata[key] = value
        return True

    def remove_metadata(self, key):
        # Native: True for a present and for an absent key.
        self.writes.append(("remove", key))
        self.metadata.pop(key, None)
        return True

    @contextmanager
    def bulk_operation(self):
        self.bulk_entries += 1
        self.in_bulk = True
        try:
            yield
        finally:
            self.in_bulk = False

    def create_file_from_path(self, *args, **kwargs):
        assert self.in_bulk
        self.writes.append(("import", args))
        return super().create_file_from_path(*args, **kwargs)


class ExternalLibrary:
    def __init__(self, name, writes):
        self.name = name
        self.writes = writes
        self._backing_file = None

    @property
    def backing_file(self):
        return self._backing_file

    @backing_file.setter
    def backing_file(self, value):
        self.writes.append(("backing", self.name, value.id))
        self._backing_file = value


class ExternalLocation:
    def __init__(self, name, writes, address=0x1000, library=None, target_symbol=None,
                 reject=False):
        self.source_symbol = NS(name=name, address=address)
        self._library = library
        self._target_symbol = target_symbol
        self.writes = writes
        self.reject = reject

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value):
        self.writes.append(("location", self.source_symbol.name, value.name))
        if not self.reject:
            self._library = value

    @property
    def target_symbol(self):
        return self._target_symbol

    @target_symbol.setter
    def target_symbol(self, value):
        self.writes.append(("symbol", value))
        self._target_symbol = value


class LinkView(FakeView):
    def __init__(self, pf, needed=(), symbols=()):
        super().__init__(pf.path_on_disk, pf)
        self.file.original_filename = pf.path_on_disk
        self.writes = []
        self.libraries = list(needed)
        self.externals = {name: ExternalLibrary(name, self.writes) for name in needed}
        self.locations = [ExternalLocation(name, self.writes, 0x1000 + i)
                          for i, name in enumerate(symbols)]

    def get_external_library(self, name):
        return self.externals.get(name)

    def add_external_library(self, name):
        assert name not in self.externals, "adding an existing library returns None in BN"
        self.writes.append(("add_library", name))
        lib = ExternalLibrary(name, self.writes)
        self.externals[name] = lib
        self.libraries.append(name)
        return lib

    def get_external_locations(self):
        return list(self.locations)


@pytest.fixture
def world(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    ops = bridge.project_rootfs
    root = tmp_path / "firmware.bnpr"
    root.mkdir()
    (root / "project.bnpm").write_bytes(SQLITE_MAGIC)
    project = RootfsProject(root)
    bn = sys.modules["binaryninja"]
    bn.ProjectFile = FakeProjectFile
    bn.Project = NS(open_project=lambda path: project)
    monkeypatch.setattr(bridge.project_ops, "find_other_owner", lambda *a, **kw: None)
    instance = bridge.BinaryNinjaBridge(instance_id="rootfs-test")
    instance.projects.open(str(root), self_pid=os.getpid())
    _register_views(bridge)
    bridge._load_in_progress.clear()
    infos = {}
    parsed = []

    def parse(path, *, check=None):
        if check is not None:
            check()
        parsed.append(str(path))
        return infos.get(str(path))

    monkeypatch.setattr(ops, "parse_elf_file", parse)
    world = NS(bridge=bridge, ops=ops, instance=instance, project=project, bn=bn,
               root=root, infos=infos, parsed=parsed, tmp=tmp_path)
    world.meta = {"version": 1, "root": str(tmp_path), "platform": "linux",
                  "lookup_paths": [], "libs": {}, "synced_at": None}
    instance.projects.set_rootfs(world.meta)
    return world


def elf(world, **kwargs):
    module = importlib.import_module(world.ops.__package__ + ".elf_dynamic")
    return module.ElfDynamicInfo(e_machine=kwargs.pop("e_machine", 183),
                                 elf_class=kwargs.pop("elf_class", 64),
                                 versions_unread=kwargs.pop("versions_unread", None), **kwargs)


def exports(*names, weak=False):
    return {name: NS(name=name, weak=weak, version=None, kind="func") for name in names}


def library(w, path, *, names=None, info=None, database=False):
    manager = w.instance.projects
    pf = w.project.add_file(Path(path).name, manager.ensure_folder(str(Path(path).parent)))
    info = info or elf(w)
    w.infos[pf.path_on_disk] = info
    w.meta["libs"][pf.id] = {"rel_realpath": path, "soname": info.soname,
                             "names": names or [pf.name], "sha256": "invented",
                             "e_machine": info.e_machine, "elf_class": info.elf_class}
    manager.set_rootfs(w.meta)
    if database:
        w.project.add_file(pf.name + ".bndb", pf.folder, data=SQLITE_MAGIC)
    return pf


def target(w, needed=(), symbols=(), **kwargs):
    pf = w.project.add_file("app")
    bv = LinkView(pf, needed, symbols)
    w.infos[pf.path_on_disk] = elf(w, needed=list(needed), **kwargs)
    _register_views(w.bridge, bv)
    w.instance.targets.refresh()
    return bv


def link(w, **kwargs):
    return w.ops.project_link(w.instance, "app", **kwargs)


def test_create_owns_project_seeds_metadata_and_publishes(world, monkeypatch):
    w = world
    w.instance.projects.close()
    dest = w.tmp / "new.bnpr"
    created = RootfsProject(dest, "new")
    calls = []

    def create(path, name):
        calls.append((path, name))
        dest.mkdir()
        (dest / "project.bnpm").write_bytes(SQLITE_MAGIC)
        return created

    w.bn.Project = NS(create_project=create, open_project=lambda path: created)
    result = w.ops.project_create(w.instance, str(dest), "new",
                                 {"root": str(w.tmp), "platform": "linux"})
    assert calls == [(str(dest), "new")]
    assert result["created"] and result["path"] == str(dest)
    assert w.instance.projects.get_rootfs()["libs"] == {}
    assert json.loads(created.metadata["bn.rootfs"])["version"] == 1
    assert json.loads(w.instance.registry_path.read_text())["project"]["path"] == str(dest)
    with pytest.raises(RuntimeError, match="already exists"):
        w.ops.project_create(w.instance, str(dest), "new")


def test_create_refuses_while_another_project_is_open(world):
    with pytest.raises(RuntimeError, match="different project"):
        world.ops.project_create(world.instance, str(world.tmp / "other.bnpr"), "other")
    assert not (world.tmp / "other.bnpr").exists()


def test_metadata_missing_and_json_roundtrip(world):
    manager = world.instance.projects
    world.project.metadata.clear()
    assert manager.get_rootfs() is None
    manager.set_rootfs(world.meta)
    assert manager.get_rootfs() == world.meta
    # #1071: present-but-unreadable is the third state, not "no metadata".
    world.project.metadata["bn.rootfs"] = "broken json"
    with pytest.raises(RuntimeError, match="rootfs metadata is not valid JSON"):
        manager.get_rootfs()


def _store_raw_metadata(w, document):
    """Write the `bn.rootfs` blob the way a foreign tool or another bn would:
    straight into the project, past `set_rootfs`'s readback."""
    w.project.metadata["bn.rootfs"] = json.dumps(document)


_VERSION_REFUSAL = ("this project's rootfs metadata is version 999; this bn understands "
                    "version 1; this bn cannot rebuild an inventory it cannot read: upgrade "
                    "bn, or move the project aside and re-run `bn project init <root>`")


def test_malformed_libs_metadata_is_refused_by_key_1071(world):
    """#1071: `{"libs": ["bad"]}` crashed `project libs` with
    `AttributeError: 'list' object has no attribute 'items'` (an `internal
    error:` at exit 2). It is untrusted project-level input and is refused
    naming the key, with the way back."""
    w = world
    _store_raw_metadata(w, {"version": 1, "root": str(w.tmp), "platform": "linux",
                            "lookup_paths": [], "libs": ["bad"]})
    with pytest.raises(RuntimeError) as raised:
        w.ops.project_libs(w.instance)
    message = str(raised.value)
    assert message.startswith("rootfs metadata key 'libs' must be an object, got list")
    assert "bn project init <root>" in message and "AttributeError" not in message


@pytest.mark.parametrize(("patch", "expected"), [
    ({"libs": {"11111111-aaaa-4aaa-8aaa-aaaaaaaaaaaa": "x"}},
     "rootfs metadata key 'libs['11111111-aaaa-4aaa-8aaa-aaaaaaaaaaaa']' must be an object, got str"),
    ({"libs": {"id": {"rel_realpath": 7}}},
     "rootfs metadata key 'libs['id'].rel_realpath' must be a string, got int"),
    ({"libs": {"id": {"rel_realpath": "/lib/a.so", "names": "a.so"}}},
     "rootfs metadata key 'libs['id'].names' must be an array of strings, got str"),
    ({"libs": {"id": {"rel_realpath": "/lib/a.so", "e_machine": "arm"}}},
     "rootfs metadata key 'libs['id'].e_machine' must be an integer or null, got str"),
    ({"lookup_paths": "/lib"},
     "rootfs metadata key 'lookup_paths' must be an array, got str"),
    ({"lookup_paths": [{"path": "/lib"}]},
     "rootfs metadata key 'lookup_paths[0].source' must be a string, got NoneType"),
    ({"root": 5}, "rootfs metadata key 'root' must be a string or null, got int"),
    ({"platform": 1}, "rootfs metadata key 'platform' must be a string or null, got int"),
    ({"synced_at": 5}, "rootfs metadata key 'synced_at' must be a string or null, got int"),
])
def test_every_shape_a_consumer_relies_on_is_checked_1071(world, patch, expected):
    w = world
    _store_raw_metadata(w, {**w.meta, **patch})
    for entry in (
        lambda: w.ops.project_libs(w.instance),
        lambda: w.ops.project_sync_libs(w.instance, root=str(w.tmp), dry_run=True),
        lambda: w.instance.projects.get_rootfs(),
    ):
        with pytest.raises(RuntimeError) as raised:
            entry()
        assert str(raised.value).startswith(expected), str(raised.value)


def test_missing_required_metadata_key_is_named_1071(world):
    w = world
    _store_raw_metadata(w, {"version": 1, "root": str(w.tmp), "platform": "linux"})
    with pytest.raises(RuntimeError, match=r"rootfs metadata key 'libs' is missing"):
        w.ops.project_libs(w.instance)
    _store_raw_metadata(w, ["not", "an", "object"])
    with pytest.raises(RuntimeError, match=r"rootfs metadata must be a JSON object, got list"):
        w.ops.project_libs(w.instance)


def test_future_metadata_version_is_refused_by_every_consumer_1071(world, monkeypatch):
    """#1071: `version: 999` was read as version 1. Every reader refuses it, and
    a sync must not merge into (or overwrite) a format it does not understand."""
    w = world
    target(w, needed=["libz.so"], symbols=["inflate"])
    future = {"version": 999, "root": str(w.tmp), "platform": "linux", "libs": {}}
    _store_raw_metadata(w, future)
    stored_before = w.project.metadata["bn.rootfs"]
    writes_before = list(w.project.writes)
    monkeypatch.setattr(w.ops, "lookup_paths", lambda *a: [NS(path_in_root="/lib", source="default")])

    for entry in (
        lambda: w.ops.project_libs(w.instance),
        lambda: w.ops.project_sync_libs(w.instance, root=str(w.tmp)),
        lambda: w.ops.project_sync_libs(w.instance, root=str(w.tmp), dry_run=True),
        lambda: link(w),
    ):
        with pytest.raises(RuntimeError) as raised:
            entry()
        assert str(raised.value) == _VERSION_REFUSAL, str(raised.value)

    assert w.project.metadata["bn.rootfs"] == stored_before
    assert w.project.writes == writes_before and w.project.files[1:] == []
    with pytest.raises(RuntimeError, match=r"rootfs metadata has no version"):
        _store_raw_metadata(w, {"root": str(w.tmp), "platform": "linux", "libs": {}})
        w.ops.project_libs(w.instance)


@pytest.mark.parametrize(("blob", "expected"), [
    pytest.param("null", "rootfs metadata must be a JSON object, got NoneType", id="null"),
    pytest.param("", "rootfs metadata is not valid JSON", id="empty-string"),
    pytest.param('{"version": 1, "libs": ' + "[" * 100_000 + "]" * 100_000 + "}",
                 "rootfs metadata is not valid JSON (RecursionError", id="deep-nesting"),
])
def test_a_stored_blob_is_never_read_as_no_metadata_1071(world, monkeypatch, blob, expected):
    """#1071 review: a stored `null` or empty string was read as "no metadata",
    so the next sync started from an empty inventory and imported every
    library a second time; a deeply nested blob was `internal error:
    RecursionError`. Only an ABSENT key is "none"."""
    w = world
    root = w.tmp / "rootfs"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "liba.so").write_bytes(b"\x7fELF")
    w.infos[str(root / "lib" / "liba.so")] = elf(w)
    monkeypatch.setattr(w.ops, "lookup_paths", lambda *a: [NS(path_in_root="/lib", source="default")])
    assert len(w.ops.project_sync_libs(w.instance, root=str(root))["imported"]) == 1
    w.project.metadata["bn.rootfs"] = blob
    files_before, writes_before = list(w.project.files), list(w.project.writes)

    for entry in (
        lambda: w.instance.projects.get_rootfs(),
        lambda: w.ops.project_libs(w.instance),
        lambda: w.ops.project_sync_libs(w.instance, root=str(root)),
    ):
        with pytest.raises(RuntimeError) as raised:
            entry()
        assert str(raised.value).startswith(expected), str(raised.value)[:200]
    assert w.project.files == files_before and w.project.writes == writes_before
    assert w.project.metadata["bn.rootfs"] == blob

    del w.project.metadata["bn.rootfs"]
    assert w.instance.projects.get_rootfs() is None


@pytest.mark.parametrize(("version", "shown"), [
    ("1", "version '1' (a str, not an integer)"),
    (1.0, "version 1.0 (a float, not an integer)"),
    (True, "version True (a bool, not an integer)"),
    (-1, "version -1"),
    (2, "version 2"),
])
def test_only_the_integer_current_version_is_current_1071(world, version, shown):
    w = world
    _store_raw_metadata(w, {**w.meta, "version": version})
    with pytest.raises(RuntimeError) as raised:
        w.ops.project_libs(w.instance)
    message = str(raised.value)
    assert message.startswith(f"this project's rootfs metadata is {shown}; this bn understands "
                              "version 1; "), message
    assert "move the project aside and re-run `bn project init <root>`" in message


def test_unknown_metadata_keys_are_accepted_and_refusals_are_bounded_1071(world):
    w = world
    pf = library(w, "/lib/libz.so")
    extra = {**w.meta, "added_by_a_later_bn": {"x": 1},
             "lookup_paths": [{"path": "/lib", "source": "default", "note": "extra"}]}
    extra["libs"][pf.id]["added_too"] = True
    _store_raw_metadata(w, extra)
    assert [row["project_path"] for row in w.ops.project_libs(w.instance)["libs"]] == ["/lib/libz.so"]

    # A refusal echoes what it refuses, but not a megabyte of it.
    _store_raw_metadata(w, {**w.meta, "version": "9" * 1_000_000})
    with pytest.raises(RuntimeError) as raised:
        w.ops.project_libs(w.instance)
    assert len(str(raised.value)) < 600 and "..." in str(raised.value)
    _store_raw_metadata(w, {**w.meta, "libs": {"k" * 1_000_000: "x"}})
    with pytest.raises(RuntimeError) as raised:
        w.ops.project_libs(w.instance)
    assert len(str(raised.value)) < 600 and "must be an object, got str" in str(raised.value)


def test_sync_group_skip_changed_bulk_and_metadata(world, monkeypatch):
    w = world
    root = w.tmp / "rootfs"
    (root / "lib").mkdir(parents=True)
    real = root / "lib" / "libcore-1.so"
    real.write_bytes(b"\x7fELF")
    (root / "lib" / "libcore.so.1").symlink_to(real.name)
    w.infos[str(real)] = elf(w, soname="libcore.so", exports=exports("core"))
    monkeypatch.setattr(w.ops, "lookup_paths", lambda *a: [NS(path_in_root="/lib", source="default")])
    first = w.ops.project_sync_libs(w.instance, root=str(root))
    assert len(first["imported"]) == 1 and w.project.bulk_entries == 1
    pf = w.project.files[0]
    assert pf.name == real.name and pf.get_path_in_project() == "/lib/" + real.name
    meta = w.instance.projects.get_rootfs()
    row = meta["libs"][pf.id]
    assert meta["root"] == str(root) and meta["platform"] == "linux"
    assert row["rel_realpath"] == "/lib/" + real.name
    assert row["paths"] == ["/lib/" + real.name, "/lib/libcore.so.1"]
    assert row["names"] == [real.name, "libcore.so.1", "libcore.so"]
    assert (row["e_machine"], row["elf_class"]) == (183, 64)
    assert pf.description == "rootfs library; names: " + ", ".join(row["names"])
    assert meta["lookup_paths"] == [{"path": "/lib", "source": "default"}]
    second = w.ops.project_sync_libs(w.instance)
    assert second["skipped_existing"] == 1 and not second["changed"]
    real.write_bytes(b"\x7fELF changed")
    third = w.ops.project_sync_libs(w.instance)
    assert third["changed"] == [{"rel_realpath": "/lib/" + real.name, "project_file_id": pf.id}]
    assert len(w.project.files) == 1 and w.project.bulk_entries == 1
    listing = w.ops.project_libs(w.instance)
    assert listing["libs"][0]["project_path"] == "/lib/" + real.name
    assert listing["synced_at"]


def test_sync_dry_run_no_writes_and_rejects_unparseable(world, monkeypatch):
    w = world
    a = w.tmp / "liba.so"
    b = w.tmp / "libbad.so"
    a.write_bytes(b"\x7fELF")
    b.write_bytes(b"\x7fELF")
    w.infos[str(a)] = elf(w)
    monkeypatch.setattr(w.ops, "lookup_paths", lambda *a: [NS(path_in_root="/", source="default")])
    before = list(w.project.writes)
    result = w.ops.project_sync_libs(w.instance, dry_run=True)
    assert len(result["imported"]) == 1 and result["skipped_not_elf"] == 1
    assert w.project.writes == before and not w.project.files and not w.project.bulk_entries
    w.project.metadata.clear()
    with pytest.raises(RuntimeError, match="no rootfs"):
        w.ops.project_sync_libs(w.instance)


def test_rpath_beats_config_and_machine_class_must_match(world):
    w = world
    library(w, "/config/liba.so", info=elf(w, exports=exports("run")))
    selected = library(w, "/vendor/liba.so", info=elf(w, exports=exports("run")), database=True)
    library(w, "/first/liba.so", info=elf(w, e_machine=62, exports=exports("run")))
    library(w, "/second/liba.so", info=elf(w, elf_class=32, exports=exports("run")))
    w.meta["lookup_paths"] = [{"path": "/config", "source": "ld_so_conf:/etc/ld.so.conf"}]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, ["liba.so"], ["run"], rpath="/first:/second:$ORIGIN:/vendor")
    result = link(w)
    assert result["libraries"][0]["project_file_id"] == selected.id
    assert result["libraries"][0]["backing"] == "database"
    assert bv.externals["liba.so"].backing_file.name == "liba.so.bndb"
    assert result["ambiguous_libraries"][0]["candidates"] == ["/vendor/liba.so", "/config/liba.so"]
    assert result["linked"] == 1 and result["status"] == "verified"
    assert w.instance.targets.is_dirty(bv) and not bv.file.modified


def test_runpath_after_environment_before_config(world):
    w = world
    for directory in ("env", "run", "conf"):
        library(w, f"/{directory}/liba.so")
    w.meta["lookup_paths"] = [{"path": "/env", "source": "ld_library_path:/etc/profile"},
                               {"path": "/conf", "source": "ld_so_conf:/etc/ld.so.conf"}]
    w.instance.projects.set_rootfs(w.meta)
    target(w, ["liba.so"], runpath="/run")
    result = link(w, dry_run=True)
    assert result["ambiguous_libraries"][0]["candidates"] == [
        "/env/liba.so", "/run/liba.so", "/conf/liba.so"]


def test_verneed_wins_and_transitive_bfs_scope_and_weak_order(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, needed=["libchild.so"], exports=exports("same", weak=True)))
    library(w, "/lib/libb.so", info=elf(w, exports=exports("same")))
    library(w, "/lib/libchild.so", info=elf(w, needed=["liba.so"], exports=exports("deep")))
    bv = target(w, ["liba.so", "libb.so"], ["same@V1", "deep"],
                import_libraries={"same": "libb.so"})
    result = link(w)
    assert result["scope"] == ["liba.so", "libb.so", "libchild.so"]
    assert [loc.library.name for loc in bv.locations] == ["libb.so", "libchild.so"]
    assert bv.locations[0].target_symbol == "same"
    assert bv.externals["libchild.so"].backing_file.name == "libchild.so"
    # Without verneed, an earlier weak definition wins over a later strong one.
    w.infos[bv.file.filename].import_libraries.clear()
    result = link(w, relink=True)
    assert result["relinked"] == 1 and result["already_linked"] == 1
    assert bv.locations[0].library.name == "liba.so"


def test_unresolved_reasons_and_readback_failure(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("good")))
    bv = target(w, ["liba.so", "libmissing.so"], ["good", "unknown", "versioned"],
                import_libraries={"versioned": "libmissing.so"})
    bv.locations[0].reject = True
    result = link(w)
    assert result["status"] == "verification_failed" and not result["success"]
    assert result["unresolved"] == [
        {"symbol": "unknown", "reason": "not exported by any linked library"},
        {"symbol": "versioned", "reason": "library libmissing.so not in project"}]
    assert result["notes"] and w.instance.targets.is_dirty(bv)


def test_dry_run_no_writes_then_noop_and_cache(world):
    w = world
    pf = library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    result = link(w, dry_run=True)
    assert result["linked"] == 1 and result["preview"] and not result["committed"]
    assert not bv.writes and not w.instance.targets.is_dirty(bv)
    link(w)
    count = w.parsed.count(pf.path_on_disk)
    assert count == 1
    writes = list(bv.writes)
    result = link(w)
    assert result["status"] == "noop" and result["already_linked"] == 1
    assert bv.writes == writes and w.parsed.count(pf.path_on_disk) == count
    Path(pf.path_on_disk).write_bytes(b"\x7fELF new")
    link(w)
    assert w.parsed.count(pf.path_on_disk) == count + 1
    assert len(w.instance.projects.export_cache) == 1


def test_needed_falls_back_to_target_header(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    bv.libraries.clear()
    bv.externals.clear()
    result = link(w)
    assert result["linked"] == 1 and bv.externals["liba.so"].backing_file is not None


def test_non_elf_only_binds_backing(world):
    w = world
    library(w, "/lib/liba.so")
    bv = target(w, ["/usr/lib/liba.so"], ["call"])
    bv.view_type = "Mach-O"
    result = link(w)
    assert result["linked"] == 0 and bv.locations[0].library is None
    assert bv.externals["/usr/lib/liba.so"].backing_file.name == "liba.so"


@pytest.mark.parametrize("enabled", [True, False])
def test_project_load_auto_link_and_no_link(world, enabled):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    _register_views(w.bridge)
    w.bn.load = lambda pf, **kw: bv
    result = w.instance._project_load("app", link=enabled)
    assert result["loaded"] and not w.instance.targets.is_dirty(bv)
    if enabled:
        assert result["link"]["linked"] == 1
        assert bv.locations[0].library.name == "liba.so"
    else:
        assert result["link"] is None and not bv.writes


def test_project_load_link_failure_does_not_fail_load(world, monkeypatch):
    w = world
    library(w, "/lib/liba.so")
    bv = target(w, ["liba.so"])
    _register_views(w.bridge)
    w.bn.load = lambda pf, **kw: bv

    def fail(*a, **kw):
        raise RuntimeError("invented linking failure")

    monkeypatch.setattr(w.ops, "link_view", fail)
    result = w.instance._project_load("app")
    assert result["loaded"] and any("invented linking failure" in n for n in result["notes"])


def test_binder_forwards_flags_and_target(world, monkeypatch):
    w = world
    calls = []
    monkeypatch.setattr(w.ops, "project_link", lambda *a, **kw: calls.append((a, kw)))
    w.bridge._bind_project_link(w.instance, {"dry_run": True, "relink": True}, "chosen")
    assert calls == [((w.instance, "chosen"), {"dry_run": True, "relink": True})]
    with pytest.raises(RuntimeError):
        w.bridge._bind_project_link(w.instance, {"dry_run": "yes"}, "chosen")


def test_preview_plans_missing_direct_and_transitive_libraries_without_writes(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, needed=["libchild.so"]))
    library(w, "/lib/libchild.so", info=elf(w, exports=exports("child")))
    bv = target(w, ["liba.so"], ["child"])
    bv.externals.clear()
    result = link(w, dry_run=True)
    assert result["linked"] == 1 and result["rebound"] == 2
    assert [row["name"] for row in result["libraries"]] == ["liba.so", "libchild.so"]
    assert not bv.writes and not bv.externals


def test_load_verification_failure_is_a_note_and_marks_dirty(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    bv.locations[0].reject = True
    _register_views(w.bridge)
    w.bn.load = lambda *a, **kw: bv
    result = w.instance._project_load("app")
    assert result["loaded"] and result["link"]["status"] == "verification_failed"
    assert "import linking verification failed" in result["notes"]
    assert w.instance.targets.is_dirty(bv)
    assert result["target"]["unsaved"] is True


def test_unreadable_target_header_does_not_filter_machine(world):
    w = world
    provider = library(w, "/lib/liba.so", info=elf(w, e_machine=62, exports=exports("call")))
    library(w, "/lib/libunrelated.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    w.infos.pop(bv.file.filename)
    assert link(w)["linked"] == 1
    assert bv.externals["liba.so"].backing_file is provider
    assert bv.locations[0].library is bv.externals["liba.so"]
    assert "libunrelated.so" not in bv.externals


def test_scope_depth_is_bounded(world):
    w = world
    for n in range(11):
        library(w, f"/lib/lib{n}.so", info=elf(
            w, needed=[f"lib{n + 1}.so"] if n < 10 else [], exports=exports(f"call{n}")))
    target(w, ["lib0.so"], ["call8", "call9"])
    result = link(w, dry_run=True)
    assert result["scope"] == [f"lib{n}.so" for n in range(9)]
    assert result["linked"] == 1
    assert result["unresolved"] == [
        {"symbol": "call9", "reason": "not exported by any linked library"}]


def test_load_links_under_publication_lock_and_gate_before_registry(world, monkeypatch):
    w = world
    library(w, "/lib/liba.so")
    bv = target(w, ["liba.so"])
    _register_views(w.bridge)
    w.bn.load = lambda *a, **kw: bv
    held = False
    original_lock = w.instance._target_lock.write
    original_link = w.ops.link_view

    @contextmanager
    def tracking_lock():
        nonlocal held
        with original_lock():
            held = True
            try:
                yield
            finally:
                held = False

    def linking(*args, **kwargs):
        assert held and w.instance._write_gate.locked()
        assert bv in w.bridge._headless_views
        return original_link(*args, **kwargs)

    monkeypatch.setattr(w.instance._target_lock, "write", tracking_lock)
    monkeypatch.setattr(w.ops, "link_view", linking)
    assert w.instance._project_load("app")["link"]["status"] == "verified"
    payload = json.loads(w.instance.registry_path.read_text())
    assert payload["project_targets"][0]["project_file_id"] == bv.project_file.id


@pytest.mark.parametrize("from_load", [False, True])
@pytest.mark.parametrize("failure", ["create", "backing", "location", "readback"])
def test_partial_apply_exceptions_are_structured_and_dirty(world, monkeypatch, from_load, failure):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])

    def fail(*args, **kwargs):
        raise RuntimeError(f"invented {failure} failure")

    if failure == "create":
        bv.externals.clear()
        original = bv.add_external_library

        def add_then_fail(name):
            original(name)
            fail()

        monkeypatch.setattr(bv, "add_external_library", add_then_fail)
    elif failure == "backing":
        original = ExternalLibrary.backing_file.fset

        def set_then_fail(self, value):
            original(self, value)
            fail()

        monkeypatch.setattr(ExternalLibrary, "backing_file",
                            property(ExternalLibrary.backing_file.fget, set_then_fail))
    elif failure == "location":
        original = ExternalLocation.library.fset

        def set_then_fail(self, value):
            original(self, value)
            fail()

        monkeypatch.setattr(ExternalLocation, "library",
                            property(ExternalLocation.library.fget, set_then_fail))
    else:
        monkeypatch.setattr(w.ops, "_readback", fail)
    result = w.ops.link_view(w.instance, bv, from_load=from_load)
    assert result["status"] == "internal_error"
    assert not result["success"] and not result["committed"]
    assert any(f"invented {failure} failure" in n for n in result["notes"])
    assert bv.writes and w.instance.targets.is_dirty(bv)


@pytest.mark.parametrize("field", ["backing_file", "target_symbol"])
def test_readback_verifies_backing_ids_and_filled_symbols(world, monkeypatch, field):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    cls = ExternalLibrary if field == "backing_file" else ExternalLocation
    prop = getattr(cls, field)
    monkeypatch.setattr(cls, field, property(prop.fget, lambda self, value: None))
    result = link(w)
    assert result["status"] == "verification_failed"
    assert not result["success"] and not result["committed"]
    assert any("backing file" in n if field == "backing_file" else "target symbol" in n
               for n in result["notes"])
    assert w.instance.targets.is_dirty(bv)


def test_planning_failure_never_creates_external_libraries(world, monkeypatch):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    bv.externals.clear()

    def fail():
        raise RuntimeError("cannot enumerate locations")

    monkeypatch.setattr(bv, "get_external_locations", fail)
    with pytest.raises(RuntimeError, match="cannot enumerate locations"):
        link(w)
    assert not bv.writes and not w.instance.targets.is_dirty(bv)


def test_foreign_project_target_cannot_use_owned_inventory_even_with_same_id(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    foreign = FakeProject(w.tmp / "foreign.bnpr")
    bv.project_file = foreign.add_file("app", file_id=bv.project_file.id)
    w.parsed.clear()
    with pytest.raises(RuntimeError, match="different project"):
        w.ops.link_view(w.instance, bv)
    assert not bv.writes and not w.parsed and not w.instance.projects.export_cache


def test_elf_needed_order_ignores_persisted_transitive_and_user_libraries(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, needed=["libchild.so"], exports=exports("call")))
    library(w, "/lib/libb.so", info=elf(w, exports=exports("call")))
    library(w, "/lib/libchild.so", info=elf(w, exports=exports("call")))
    library(w, "/lib/libuser.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so", "libb.so"], ["call"])
    bv.libraries[:] = ["libuser.so", "libchild.so", "libb.so", "liba.so"]
    result = link(w)
    assert result["scope"] == ["liba.so", "libb.so", "libchild.so"]
    assert bv.locations[0].library.name == "liba.so"
    assert [row["name"] for row in result["libraries"]] == ["liba.so", "libb.so"]


def test_runpath_disables_rpath_when_both_are_present(world):
    w = world
    library(w, "/rpath/liba.so")
    chosen = library(w, "/runpath/liba.so")
    target(w, ["liba.so"], rpath="/rpath", runpath="/runpath")
    result = link(w, dry_run=True)
    assert result["libraries"][0]["project_file_id"] == chosen.id


def test_missing_verneed_provider_does_not_fall_back_to_unrelated_export(world):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so", "libmissing.so"], ["call"],
                import_libraries={"call": "libmissing.so"})
    result = link(w)
    assert result["linked"] == 0 and bv.locations[0].library is None
    assert result["unresolved"] == [{"symbol": "call", "reason": "library libmissing.so not in project"}]


@pytest.mark.parametrize("needed", ["/vendor/liba.so", "relative/liba.so"])
def test_slash_needed_uses_exact_firmware_path_or_remains_unresolved(world, needed):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    selected = library(w, "/objects/actual.so", names=["actual.so", "liba.so"],
                       info=elf(w, exports=exports("call")))
    w.meta["libs"][selected.id]["paths"] = ["/vendor/liba.so"]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, [needed], ["call"])
    result = link(w)
    if needed.startswith("/"):
        assert result["libraries"][0]["project_file_id"] == selected.id
        assert bv.locations[0].library.name == needed
    else:
        assert result["scope"] == [] and result["linked"] == 0
        assert any("working directory" in n for n in result["notes"])
        assert not bv.writes


def test_loader_visible_alias_directory_controls_priority(world):
    w = world
    library(w, "/later/liba.so")
    selected = library(w, "/objects/actual.so", names=["actual.so", "liba.so"])
    w.meta["libs"][selected.id]["paths"] = ["/first/liba.so"]
    w.meta["lookup_paths"] = [{"path": "/first", "source": "default"},
                               {"path": "/later", "source": "default"}]
    w.instance.projects.set_rootfs(w.meta)
    target(w, ["liba.so"])
    assert link(w)["libraries"][0]["project_file_id"] == selected.id


def test_transitive_needed_uses_requester_runpath_not_executable_runpath(world):
    w = world
    library(w, "/lib/libparent.so", info=elf(
        w, needed=["libchild.so"], runpath="/parent"))
    library(w, "/executable/libchild.so", info=elf(w, exports=exports("child")))
    child = library(w, "/parent/libchild.so", info=elf(
        w, needed=["libgrandchild.so"], exports=exports("child")))
    library(w, "/parent/libgrandchild.so", info=elf(w, exports=exports("grand")))
    grand = library(w, "/default/libgrandchild.so", info=elf(w, exports=exports("grand")))
    w.meta["lookup_paths"] = [{"path": "/default", "source": "default"}]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, ["libparent.so"], ["child", "grand"], runpath="/executable")
    result = link(w)
    assert result["scope"] == ["libparent.so", "libchild.so", "libgrandchild.so"]
    assert bv.externals["libchild.so"].backing_file.id == child.id
    assert bv.externals["libgrandchild.so"].backing_file.id == grand.id


def test_empty_runpath_still_disables_rpath(world):
    w = world
    library(w, "/rpath/liba.so")
    chosen = library(w, "/default/liba.so")
    w.meta["lookup_paths"] = [{"path": "/default", "source": "default"}]
    w.instance.projects.set_rootfs(w.meta)
    target(w, ["liba.so"], rpath="/rpath", runpath="")
    assert link(w)["libraries"][0]["project_file_id"] == chosen.id


def test_raw_implicit_project_link_cannot_mutate_gui_focus_with_multiple_targets(world, monkeypatch):
    w = world
    bv = target(w, ["liba.so"])
    other = FakeView("/other")
    _register_views(w.bridge, bv, other)
    monkeypatch.setattr(w.ops, "project_link", lambda *a, **kw: pytest.fail("must refuse before linking"))
    result = w.instance.dispatch({"op": "project_link", "target": "active"})
    assert result["ok"] is False and not bv.writes


@pytest.mark.parametrize("operation", ["save_database", "close_binary", "project_close"])
def test_load_link_holds_gate_against_concurrent_destructive_requests(world, monkeypatch, operation):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    _register_views(w.bridge)
    w.bn.load = lambda *a, **kw: bv
    entered = threading.Event()
    resume = threading.Event()
    original = w.ops.link_view
    outcomes = []

    def linking(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(*args, **kwargs)

    def load():
        try:
            outcomes.append(w.instance._project_load("app"))
        except BaseException as exc:
            outcomes.append(exc)

    monkeypatch.setattr(w.ops, "link_view", linking)
    worker = threading.Thread(target=load)
    worker.start()
    try:
        assert entered.wait(5)
        refusal = w.instance.dispatch({"op": operation, "target": "app", "no_wait": True})
        assert refusal["ok"] is False
        assert not bv.closed and w.instance.projects.current is not None
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], dict), outcomes
    assert outcomes[0]["link"]["status"] == "verified"


def test_ancestor_rpath_is_inherited_but_requester_runpath_takes_precedence(world):
    w = world
    library(w, "/lib/libparent.so", info=elf(
        w, needed=["libchild.so", "libleaf.so"], rpath="/parent"))
    library(w, "/default/libchild.so", info=elf(w, exports=exports("child")))
    inherited = library(w, "/executable/libchild.so", info=elf(w, exports=exports("child")))
    library(w, "/default/libleaf.so", info=elf(w, exports=exports("leaf")))
    nearest = library(w, "/parent/libleaf.so", info=elf(
        w, needed=["libgrandchild.so"], runpath="/leaf", exports=exports("leaf")))
    library(w, "/executable/libgrandchild.so", info=elf(w, exports=exports("grand")))
    own = library(w, "/leaf/libgrandchild.so", info=elf(w, exports=exports("grand")))
    w.meta["lookup_paths"] = [{"path": "/default", "source": "default"}]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, ["libparent.so"], ["child", "leaf", "grand"], rpath="/executable")
    result = link(w)
    assert result["status"] == "verified"
    assert bv.externals["libchild.so"].backing_file.id == inherited.id
    assert bv.externals["libleaf.so"].backing_file.id == nearest.id
    assert bv.externals["libgrandchild.so"].backing_file.id == own.id


@pytest.mark.parametrize("exact_directory", ["/early", "/late"])
def test_exact_loader_filename_beats_competing_soname(world, exact_directory):
    w = world
    impostor = library(
        w, "/early/aaa-unrelated.so",
        names=["aaa-unrelated.so", "libwanted.so"],
        info=elf(w, soname="libwanted.so", exports=exports("call")),
    )
    exact = library(
        w, exact_directory + "/libwanted.so", info=elf(w, exports=exports("call")),
    )
    w.meta["lookup_paths"] = [
        {"path": "/early", "source": "default"},
        {"path": "/late", "source": "default"},
    ]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, ["libwanted.so"], ["call"])

    result = link(w)

    assert result["linked"] == 1
    assert bv.externals["libwanted.so"].backing_file is exact
    assert bv.externals["libwanted.so"].backing_file is not impostor
    assert result["ambiguous_libraries"] == []


def test_filename_ranking_uses_only_matching_alias_directories(world):
    w = world
    wrong_alias = library(w, "/early/real.so", names=["real.so", "libwanted.so"])
    exact = library(w, "/middle/libwanted.so")
    w.meta["libs"][wrong_alias.id]["paths"] = ["/early/real.so", "/late/libwanted.so"]
    w.meta["libs"][exact.id]["paths"] = ["/middle/libwanted.so"]
    w.meta["lookup_paths"] = [
        {"path": path, "source": "default"} for path in ("/early", "/middle", "/late")
    ]
    w.instance.projects.set_rootfs(w.meta)
    bv = target(w, ["libwanted.so"])

    link(w)

    assert bv.externals["libwanted.so"].backing_file is exact


def test_deleted_provider_disappears_from_libs_and_link_inventory(world):
    w = world
    removed = library(w, "/lib/libgone.so", database=True)
    kept = library(w, "/lib/liblive.so")
    w.project.delete_file(removed)
    bv = target(w, ["libgone.so", "liblive.so"])

    listing = w.ops.project_libs(w.instance)
    result = link(w)

    assert [row["project_file_id"] for row in listing["libs"]] == [kept.id]
    assert bv.externals["libgone.so"].backing_file is None
    assert bv.externals["liblive.so"].backing_file is kept
    assert next(row for row in result["libraries"] if row["name"] == "libgone.so")[
        "project_file_id"] is None


def test_unread_version_requirements_never_guess_unrelated_exports(world):
    w = world
    library(w, "/lib/libunrelated.so", info=elf(w, exports=exports("call")))
    bv = target(
        w, ["libunrelated.so"], ["call"],
        versions_unread="DT_VERNEED exceeds the total vernaux work limit",
    )

    result = link(w)

    assert bv.locations[0].library is None
    assert result["linked"] == 0
    assert "vernaux" in result["unresolved"][0]["reason"]


def test_sync_preserves_android_alias_for_exact_needed_resolution(world):
    w = world
    root = w.tmp / "android"
    real = root / "objects/runtime/lib64/libshared.so"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"\x7fELF")
    (root / "apex").mkdir()
    (root / "apex/a").symlink_to("/objects/runtime")
    (root / "apex/b").symlink_to("/objects/runtime")
    w.infos[str(real)] = elf(w)

    result = w.ops.project_sync_libs(w.instance, root=str(root), platform="android")
    provider_id = result["imported"][0]["id"]
    entry = w.instance.projects.get_rootfs()["libs"][provider_id]
    assert entry["paths"] == [
        "/apex/a/lib64/libshared.so", "/apex/b/lib64/libshared.so",
    ]
    needed = "/apex/b/lib64/libshared.so"
    bv = target(w, [needed])

    linked = link(w)

    assert linked["libraries"][0]["resolved"] is True
    assert bv.externals[needed].backing_file.id == provider_id


def test_link_uses_saved_database_binding_after_raw_project_file_is_deleted(world):
    w = world
    provider = library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    raw = bv.project_file
    saved = w.instance._save_database("app")
    database = w.instance.projects.resolve_file(saved["project_file_id"])
    assert bv.project_file is raw

    deleted = w.bridge.project_ops.project_delete(w.instance, raw.id)
    Path(raw.path_on_disk).unlink()
    assert deleted["deleted"] is True
    assert w.instance.targets.project_binding(bv) == (database.id, "/app.bndb")

    result = w.ops.project_link(w.instance, "app.bndb")

    assert result["success"] is True
    assert result["linked"] == 1
    assert bv.externals["liba.so"].backing_file is provider
    assert bv.locations[0].library is bv.externals["liba.so"]


@pytest.mark.parametrize("invalid_binding", ["missing-file", "wrong-path"])
def test_link_rejects_invalid_saved_binding_without_falling_back_to_raw(world, invalid_binding):
    w = world
    library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    bv = target(w, ["liba.so"], ["call"])
    saved = w.instance._save_database("app")
    database = w.instance.projects.resolve_file(saved["project_file_id"])
    if invalid_binding == "missing-file":
        w.project.delete_file(database)
    else:
        w.instance.targets.note_project_database(bv, database.id, "/not-app.bndb")
    w.parsed.clear()

    with pytest.raises(RuntimeError, match="different project"):
        w.ops.link_view(w.instance, bv)

    assert bv.writes == []
    assert w.parsed == []


def test_standalone_view_without_project_binding_can_link_owned_inventory(world):
    w = world
    provider = library(w, "/lib/liba.so", info=elf(w, exports=exports("call")))
    outside = FakeProject(w.tmp / "outside.bnpr").add_file("standalone")
    bv = LinkView(outside, ["liba.so"], ["call"])
    bv.project_file = None
    w.infos[bv.file.filename] = elf(w, needed=["liba.so"])
    _register_views(w.bridge, bv)
    w.instance.targets.refresh()
    assert w.instance.targets.project_binding(bv) is None

    result = w.ops.link_view(w.instance, bv)

    assert result["success"] is True
    assert result["linked"] == 1
    assert bv.externals["liba.so"].backing_file is provider
