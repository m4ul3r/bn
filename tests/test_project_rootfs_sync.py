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


def test_failed_first_sync_leaves_no_metadata_blob_and_retry_works_1071(world, monkeypatch):
    """#1071 review: a sync that found NO metadata rolls back by removing the
    key. Storing `{}` instead (the old rollback) would leave a blob the
    validating reader refuses as versionless."""
    w = world
    root = _sources(w, monkeypatch)
    w.project.metadata.clear()
    assert w.instance.projects.get_rootfs() is None
    store = w.project.store_metadata

    def refuse_inventory(key, value):
        import json
        if json.loads(value).get("libs"):
            return False
        return store(key, value)

    monkeypatch.setattr(w.project, "store_metadata", refuse_inventory)
    with pytest.raises(RuntimeError, match="Binary Ninja returned False") as raised:
        w.ops.project_sync_libs(w.instance, root=str(root))
    assert "rollback incomplete" not in str(raised.value)
    assert "bn.rootfs" not in w.project.metadata
    assert w.project.files == [] and not w.project.in_bulk
    assert w.instance.projects.get_rootfs() is None

    monkeypatch.setattr(w.project, "store_metadata", store)
    retry = w.ops.project_sync_libs(w.instance, root=str(root))
    assert {row["name"] for row in retry["imported"]} == {"liba.so", "libb.so"}
    assert w.instance.projects.get_rootfs()["version"] == 1


def test_clear_rootfs_checks_the_native_result_and_the_readback_1071(world):
    manager = world.instance.projects
    world.project.remove_metadata = lambda key: False
    with pytest.raises(RuntimeError, match="could not remove the rootfs metadata"):
        manager.clear_rootfs()
    world.project.remove_metadata = lambda key: True      # claims success, removes nothing
    with pytest.raises(RuntimeError, match="still present"):
        manager.clear_rootfs()


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


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("lookup_dir", ["/.bn-findings", "/.bn-findings/nested"])
def test_sync_refuses_findings_namespace_before_any_import(world, dry_run, lookup_dir):
    w = world
    root = (w.tmp / "rootfs").resolve()
    (root / "etc").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / lookup_dir.lstrip("/")).mkdir(parents=True)
    # Put a valid library first so refusal must preflight the entire scan,
    # rather than importing it and leaving folders behind during rollback.
    (root / "etc/ld.so.conf").write_text(f"/lib\n{lookup_dir}\n")
    for directory, name in (("/lib", "libsafe.so"), (lookup_dir, "libinjected.so")):
        path = root / directory.lstrip("/") / name
        path.write_bytes(b"\x7fELF")
        w.infos[str(path)] = elf(w)
    old_metadata = w.instance.projects.get_rootfs()
    old_writes = list(w.project.writes)

    with pytest.raises(RuntimeError, match="reserved findings history"):
        w.ops.project_sync_libs(w.instance, root=str(root), dry_run=dry_run)

    assert w.project.bulk_entries == 0
    assert not w.project.in_bulk
    assert w.project.files == []
    assert w.project.folders == []
    assert w.project.writes == old_writes
    assert w.instance.projects.get_rootfs() == old_metadata


def test_sync_aggregate_discovery_refusal_never_commits_a_partial_inventory(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    monkeypatch.setenv("BN_ROOTFS_MAX_BYTES", "4")
    old_metadata = w.instance.projects.get_rootfs()
    old_writes = list(w.project.writes)

    with pytest.raises(RuntimeError, match="rootfs discovery refused: discovery byte limit"):
        w.ops.project_sync_libs(w.instance, root=str(root))

    assert w.project.files == []
    assert w.project.folders == []
    assert w.project.bulk_entries == 0
    assert w.project.writes == old_writes
    assert w.instance.projects.get_rootfs() == old_metadata


def _record_budgets(w, monkeypatch):
    budgets = []
    original = w.ops.DiscoveryBudget

    def recording(*args, **kwargs):
        budget = original(*args, **kwargs)
        budgets.append(budget)
        return budget

    monkeypatch.setattr(w.ops, "DiscoveryBudget", recording)
    return budgets


def test_sync_charges_elf_parsing_per_file_not_per_symbol_1075(world, monkeypatch):
    """#1075 item 2: a ~450-library firmware spent ~460k units of the 100k
    default because every symbol-name read charged one. Parsing now costs one
    unit per file (its bytes were already charged per MiB by the hash), so a
    10k-symbol library costs exactly what a 10-symbol one does."""
    w = world
    root = _sources(w, monkeypatch)
    monkeypatch.setenv("BN_ROOTFS_MAX_WORK", "20")
    budgets = _record_budgets(w, monkeypatch)
    original = w.ops.parse_elf_file
    cost = {}

    def parse(path, *, check=None):
        # The parser probes `check` once per symbol name it reads.
        before = budgets[-1].work_used
        for _ in range(10_000 if path.endswith("liba.so") else 10):
            check()
        info = original(path, check=check)
        cost[path.rsplit("/", 1)[1]] = budgets[-1].work_used - before
        return info

    monkeypatch.setattr(w.ops, "parse_elf_file", parse)

    result = w.ops.project_sync_libs(w.instance, root=str(root))

    assert {row["name"] for row in result["imported"]} == {"liba.so", "libb.so"}
    assert cost == {"liba.so": 0, "libb.so": 0}
    # One directory listed, two libraries hashed, two parsed: five units of 20.
    assert budgets[-1].work_used == 5


def test_sync_work_refusal_names_the_knob_and_the_restart_command_1075(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    monkeypatch.setenv("BN_ROOTFS_MAX_WORK", "2")
    old_metadata = w.instance.projects.get_rootfs()

    with pytest.raises(RuntimeError) as raised:
        w.ops.project_sync_libs(w.instance, root=str(root))

    message = str(raised.value)
    assert message.startswith("rootfs discovery refused: work limit exceeded (used more than 2 "
                              "units, BN_ROOTFS_MAX_WORK=2, charged per file opened and per MiB read)")
    assert "no inventory committed" in message
    # The bridge that owns the project read the variable when it started; the
    # CLI's environment never reaches it, so the advice is a restart.
    assert "The owning bridge reads BN_ROOTFS_MAX_WORK at start, so set it and restart" in message
    assert message.endswith(": BN_ROOTFS_MAX_WORK=4 bn session restart rootfs-test")
    assert w.project.files == []
    assert w.instance.projects.get_rootfs() == old_metadata


def test_sync_refusal_restart_command_runs_for_a_dash_leading_instance_id_1075(
    world, monkeypatch,
):
    """#1075 review: an instance id may start with `-`; printed raw, `bn session
    restart -x` is an argparse error, so the command is spelled `-- -x`."""
    w = world
    root = _sources(w, monkeypatch)
    monkeypatch.setenv("BN_ROOTFS_MAX_WORK", "2")
    w.instance.instance_id = "-x"

    with pytest.raises(RuntimeError) as raised:
        w.ops.project_sync_libs(w.instance, root=str(root))

    assert str(raised.value).endswith(": BN_ROOTFS_MAX_WORK=4 bn session restart -- -x")


def test_sync_refusal_names_the_restart_for_the_gui_plugin_with_no_instance_id_1096(
    world, monkeypatch,
):
    """#1096 A6a: the GUI plugin bridge has no instance id, so `_restart_command`
    returns None; before this fix the refusal then printed no advice at all.
    It must instead tell the user to restart the application itself."""
    w = world
    root = _sources(w, monkeypatch)
    monkeypatch.setenv("BN_ROOTFS_MAX_WORK", "2")
    w.instance.instance_id = None

    with pytest.raises(RuntimeError) as raised:
        w.ops.project_sync_libs(w.instance, root=str(root))

    message = str(raised.value)
    assert "session restart" not in message
    assert message.endswith(
        "starting point: restart Binary Ninja with BN_ROOTFS_MAX_WORK=4 set in its environment")


def test_sync_cancellation_during_elf_discovery_preserves_inventory(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    old_metadata = w.instance.projects.get_rootfs()
    cancelled = False
    original = w.ops.parse_elf_file

    def parse(path, *, check=None):
        nonlocal cancelled
        info = original(path, check=check)
        cancelled = True
        check()
        return info

    monkeypatch.setattr(w.ops, "parse_elf_file", parse)
    with pytest.raises(RuntimeError, match="request cancelled"):
        w.ops.project_sync_libs(w.instance, root=str(root), cancelled=lambda: cancelled)

    assert w.project.files == []
    assert w.project.folders == []
    assert w.project.bulk_entries == 0
    assert w.instance.projects.get_rootfs() == old_metadata


def test_sync_cancellation_during_copy_rolls_back_and_exits_bulk(world, monkeypatch):
    w = world
    root = _sources(w, monkeypatch)
    old_metadata = w.instance.projects.get_rootfs()
    original = w.project.create_file_from_path
    cancelled = False

    def copy(*args):
        nonlocal cancelled
        pf = original(*args)
        cancelled = True
        return pf

    monkeypatch.setattr(w.project, "create_file_from_path", copy)
    with pytest.raises(RuntimeError, match="request cancelled"):
        w.ops.project_sync_libs(w.instance, root=str(root), cancelled=lambda: cancelled)

    assert not w.project.in_bulk
    assert w.project.files == []
    assert w.instance.projects.get_rootfs() == old_metadata


@pytest.mark.parametrize("change", ["close", "metadata", "files"])
def test_sync_revalidates_project_after_unlocked_discovery(world, monkeypatch, change):
    w = world
    root = _sources(w, monkeypatch)
    original = w.ops.scan_libraries
    old_metadata = w.instance.projects.get_rootfs()

    def discover(*args):
        result = original(*args)
        if change == "close":
            w.instance.projects.close()
        elif change == "metadata":
            w.instance.projects.set_rootfs({**old_metadata, "platform": "android"})
        else:
            w.project.add_file("unrelated")
        return result

    monkeypatch.setattr(w.ops, "scan_libraries", discover)
    with pytest.raises(RuntimeError, match="project changed during discovery"):
        w.ops.project_sync_libs(w.instance, root=str(root))

    assert w.project.bulk_entries == 0
    assert all(pf.name == "unrelated" for pf in w.project.files)
    if change == "close":
        assert w.instance.projects.current is None
    else:
        expected = {**old_metadata, "platform": "android"} if change == "metadata" else old_metadata
        assert w.instance.projects.get_rootfs() == expected


def test_sync_discloses_skipped_nul_loader_paths(world):
    w = world
    root = (w.tmp / "rootfs").resolve()
    (root / "etc").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "etc/ld.so.conf").write_bytes(b"/bad\0path\n/lib\n")
    path = root / "lib/libvalid.so"
    path.write_bytes(b"\x7fELF")
    w.infos[str(path)] = elf(w)

    result = w.ops.project_sync_libs(w.instance, root=str(root))

    assert [row["name"] for row in result["imported"]] == ["libvalid.so"]
    assert "skipped loader path containing NUL" in result["notes"]
