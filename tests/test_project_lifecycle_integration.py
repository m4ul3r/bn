"""Native project save/reload and standalone-storage ownership regressions."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from bn.transport import BridgeError
from test_project_integration import (
    Lane,
    _cache_dir_within_socket_budget,
    _project_bridge,  # noqa: F401 - imported pytest fixture
    _start_instance,
    _stop_instance,
    lane,  # noqa: F401 - imported pytest fixture
    project_integration_fixtures,  # noqa: F401 - imported pytest fixture
)

pytestmark = [
    pytest.mark.real_bn,
    pytest.mark.usefixtures("project_integration_fixtures"),
]


def _save_raw(lane: Lane, folder: str) -> tuple[dict, dict, dict]:
    lane.ensure_open()
    raw = lane.ensure_imported(folder)
    loaded = lane.project_load(raw["id"], prefer_bndb=False)
    saved = lane.save(loaded["target"]["selector"])
    assert saved.returncode == 0, f"{saved.stderr}\n{saved.stdout}"
    result = json.loads(saved.stdout)
    assert result["mode"] == "created"
    return raw, loaded, result


def test_native_saved_reload_reuses_view_and_database(lane: Lane):
    try:
        raw, loaded, saved = _save_raw(lane, "saved-reload")
        original_id = loaded["target"]["target_id"]
        for selector in (raw["id"], saved["project_file_id"], saved["project_path"]):
            again = lane.project_load(selector)
            assert again["already_open"] and not again["loaded"]
            assert again["target"]["target_id"] == original_id
        targets = lane.target_rows()
        assert [row["target_id"] for row in targets] == [original_id]
        second_save = lane.save(saved["project_path"])
        assert second_save.returncode == 0, second_save.stderr
        assert json.loads(second_save.stdout)["mode"] == "snapshot"
        databases = [row for row in lane.project_files("saved-reload") if row["is_database"]]
        assert [row["id"] for row in databases] == [saved["project_file_id"]]
    finally:
        lane.project_close(force=True)


def test_native_ambiguous_siblings_never_create_third_database(lane: Lane, tmp_path):
    try:
        raw, _, saved = _save_raw(lane, "ambiguous-save")
        exported = tmp_path / "copy.bndb"
        lane.wire("project_export", {"selector": saved["project_file_id"], "dest": str(exported)})
        duplicate = lane.wire("project_import", {
            "path": str(exported), "folder": "ambiguous-save", "name": raw["name"] + ".bndb",
        })
        with pytest.raises(BridgeError, match="Ambiguous") as load_error:
            lane.project_load(raw["id"])
        assert saved["project_file_id"] in str(load_error.value)
        assert duplicate["id"] in str(load_error.value)
        raw_target = lane.project_load(raw["id"], prefer_bndb=False)["target"]
        refused = lane.save(raw_target["selector"])
        assert refused.returncode != 0
        assert "Ambiguous" in refused.stderr + refused.stdout
        database_ids = {
            row["id"] for row in lane.project_files("ambiguous-save") if row["is_database"]
        }
        assert database_ids == {saved["project_file_id"], duplicate["id"]}
    finally:
        lane.project_close(force=True)


def test_native_raw_blob_refusal_across_independent_caches(lane: Lane, tmp_path, tmp_path_factory):
    other_id = f"rawguard-{uuid.uuid4().hex[:8]}"
    cache, fallback = _cache_dir_within_socket_budget(tmp_path_factory, other_id)
    other = Lane(other_id, cache, lane.project_path, cli_available=True)
    started = False
    try:
        _, loaded, saved = _save_raw(lane, "raw-guard")
        _start_instance(other_id, other.env())
        started = True
        blob = Path(saved["path"])
        alias = tmp_path / "linked.bndb"
        alias.symlink_to(blob)
        for path in (blob, alias):
            refused = other.run("load", str(path), "--no-bndb", "--quick", "--format", "json")
            assert refused.returncode != 0
            assert "Cannot load project storage" in refused.stderr + refused.stdout
        assert other.target_rows() == []
        assert lane.target_info(saved["project_path"])["target_id"] == loaded["target"]["target_id"]
    finally:
        if started:
            _stop_instance(other_id, other.env())
        if fallback:
            shutil.rmtree(cache, ignore_errors=True)
        lane.project_close(force=True)
