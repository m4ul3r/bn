"""Library inventory remains retryable after copy or metadata failures."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

import test_project_rootfs as project_tests

elf = project_tests.elf
world = project_tests.world


def _sources(w, monkeypatch):
    root = (w.tmp / "rootfs").resolve()
    (root / "lib").mkdir(parents=True)
    for name in ("liba.so", "libb.so"):
        path = root / "lib" / name
        path.write_bytes(b"\x7fELF")
        w.infos[str(path)] = elf(w)
    monkeypatch.setattr(w.ops, "lookup_paths", lambda *a: [
        NS(path_in_root="/lib", source="default")])

    @contextmanager
    def bulk():
        # The native SDK ends bulk mode after yield, not in a finally block.
        w.project.in_bulk = True
        yield
        w.project.in_bulk = False

    monkeypatch.setattr(w.project, "bulk_operation", bulk)
    return root


def test_failed_copy_rolls_back_imports_and_retry_does_not_duplicate(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    original = w.project.create_file_from_path

    def fail_second(path, *args):
        if path.endswith("libb.so"):
            raise RuntimeError("injected copy failure")
        return original(path, *args)

    monkeypatch.setattr(w.project, "create_file_from_path", fail_second)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        w.ops.project_sync_libs(w.instance, root=str(root))
    assert not w.project.in_bulk
    assert w.project.files == []
    assert w.instance.projects.get_rootfs()["libs"] == {}

    monkeypatch.setattr(w.project, "create_file_from_path", original)
    result = w.ops.project_sync_libs(w.instance, root=str(root))
    assert {row["name"] for row in result["imported"]} == {"liba.so", "libb.so"}
    assert len(w.project.files) == 2
    assert set(w.instance.projects.get_rootfs()["libs"]) == {pf.id for pf in w.project.files}


def test_failed_metadata_commit_removes_unrecorded_imports(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    old = w.instance.projects.get_rootfs()
    store = w.project.store_metadata

    def refuse_inventory(key, value):
        import json
        if json.loads(value).get("libs"):
            return False
        return store(key, value)

    monkeypatch.setattr(w.project, "store_metadata", refuse_inventory)
    with pytest.raises(RuntimeError, match="metadata"):
        w.ops.project_sync_libs(w.instance, root=str(root))
    assert not w.project.in_bulk
    assert w.project.files == []
    assert w.instance.projects.get_rootfs() == old


def test_sync_refreshes_new_aliases_without_copying_library_again(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    w.ops.project_sync_libs(w.instance, root=str(root))
    original_ids = {pf.id for pf in w.project.files}
    (root / "lib/libalias.so").symlink_to("/lib/liba.so")
    result = w.ops.project_sync_libs(w.instance, root=str(root))
    assert result["imported"] == []
    assert {pf.id for pf in w.project.files} == original_ids
    entry = next(row for row in w.instance.projects.get_rootfs()["libs"].values()
                 if row["rel_realpath"] == "/lib/liba.so")
    assert "libalias.so" in entry["names"]
    assert "/lib/libalias.so" in entry["paths"]
