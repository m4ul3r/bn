"""`bn project` CLI tests (BNPR_PLAN.md §5.6): op/params wiring, the
open/attach routing rule, and the documented text renderer lines."""
from __future__ import annotations

import json
import os
import types

import bn.cli
import pytest

from _cli_helpers import _fake_bridge_instance  # noqa: F401


FILE_ID_A = "4f0c2a1e-7b3d-4e8f-9a21-0c5d6e7f8a9b"
FILE_ID_B = "9b8a7f6e-5d4c-4b3a-8291-f0e1d2c3b4a5"


def _instance(instance_id, pid, project=None, **meta):
    inst = _fake_bridge_instance(instance_id, pid=pid)
    inst.meta = dict(meta)
    if project is not None:
        inst.meta["project"] = project
    return inst


def _one_instance(monkeypatch, instance_id="bridge-a", project=None):
    inst = _instance(instance_id, 111, project=project)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    return inst


def _capture_pins(monkeypatch):
    pins = []
    monkeypatch.setattr(bn.cli.session_state, "update", lambda **kw: pins.append(kw))
    return pins


def _project_info(path, **extra):
    row = {
        "path": path, "id": "p-1", "name": "fw.bnpr", "description": "",
        "file_count": 2, "folder_count": 1,
        "open_targets": [{"target_id": "123:4:1", "view_id": "4",
                          "project_file_id": FILE_ID_A, "project_path": "/fw/app.bndb"}],
    }
    row.update(extra)
    return row


# --- project open / attach ---------------------------------------------------


def test_project_open_attaches_to_the_owner_without_sending(
    fake_transport, monkeypatch, tmp_path, capsys
):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    canonical = str(proj.resolve())
    owner = _instance("owner", 11, project={"path": canonical, "id": "p-1", "name": "fw.bnpr"})
    other = _instance("other", 22)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [other, owner])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    pins = _capture_pins(monkeypatch)
    calls = fake_transport()

    # The project.bnpm spelling normalises to the directory (§5.1).
    rc = bn.cli.main(["project", "open", str(proj / "project.bnpm")])

    assert rc == 0
    assert calls == []
    assert pins == [{"instance_id": "owner"}]
    out = capsys.readouterr().out
    assert "attached to instance owner" in out


def test_project_open_sends_and_pins_the_answering_instance(
    fake_transport, monkeypatch, tmp_path, capsys
):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    canonical = str(proj.resolve())
    _one_instance(monkeypatch)
    pins = _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_open": {"ok": True, "result": _project_info(canonical, opened=True),
                         "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": [os.getcwd()]}},
    })

    rc = bn.cli.main(["project", "open", str(proj)])

    assert rc == 0
    assert [c["op"] for c in calls] == ["project_open", "associate_project_roots"]
    assert calls[0]["params"] == {"path": canonical}
    assert calls[0]["spawn_missing_named"] is True
    assert calls[0]["target"] is None
    assert calls[1]["params"] == {"roots": [os.getcwd()]}
    assert calls[1]["instance_id"] == "bridge-a"
    assert pins == [{"instance_id": "bridge-a"}]
    out = capsys.readouterr().out
    assert "project: fw.bnpr" in out
    assert f"path: {canonical}" in out
    assert "/fw/app.bndb" in out


def test_project_open_instance_id_spawns_named_bridge(fake_transport, monkeypatch, tmp_path):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [
        _instance("one", 11), _instance("two", 22),
    ])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_open": {"ok": True, "result": _project_info(str(proj.resolve()))},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    rc = bn.cli.main(["project", "open", str(proj), "--instance-id", "fresh"])

    assert rc == 0
    assert calls[0]["op"] == "project_open"
    assert calls[0]["instance_id"] == "fresh"
    assert calls[0]["spawn_missing_named"] is True


def test_project_open_refuses_ambient_routing_with_multiple_instances(
    fake_transport, monkeypatch, tmp_path, capsys
):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [
        types.SimpleNamespace(instance_id="one", pid=11, socket_path="/tmp/one.sock",
                              started_at=None, meta={}),
        types.SimpleNamespace(instance_id="two", pid=22, socket_path="/tmp/two.sock",
                              started_at=None, meta={}),
    ])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    calls = fake_transport()

    rc = bn.cli.main(["project", "open", str(proj)])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "explicit -i/--instance" in err
    assert "- one (pid=11" in err


def test_project_open_json_is_the_raw_bridge_result(fake_transport, monkeypatch, tmp_path, capsys):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    info = _project_info(str(proj.resolve()), opened=True)
    fake_transport({
        "project_open": {"ok": True, "result": dict(info)},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    rc = bn.cli.main(["project", "open", str(proj), "--format", "json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["opened"] is True
    assert data["open_targets"] == info["open_targets"]


def test_project_attach_pins_only(fake_transport, monkeypatch, tmp_path, capsys):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    canonical = str(proj.resolve())
    _one_instance(monkeypatch, "owner",
                  project={"path": canonical, "id": "p-1", "name": "fw.bnpr"})
    pins = _capture_pins(monkeypatch)
    calls = fake_transport()

    rc = bn.cli.main(["project", "attach", str(proj)])

    assert rc == 0
    assert calls == []
    assert pins == [{"instance_id": "owner"}]
    assert "attached to instance owner" in capsys.readouterr().out


def test_project_attach_errors_when_nobody_owns_it(fake_transport, monkeypatch, tmp_path, capsys):
    proj = tmp_path / "fw.bnpr"
    proj.mkdir()
    _one_instance(monkeypatch)
    pins = _capture_pins(monkeypatch)
    fake_transport()

    rc = bn.cli.main(["project", "attach", str(proj)])

    assert rc == 2
    assert pins == []
    err = capsys.readouterr().err
    assert "no running bridge owns" in err
    assert "bn project open" in err


# --- op/params wiring -----------------------------------------------------------


def test_project_info_sends_op(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_info": {"ok": True, "result": _project_info("/p/fw.bnpr")}})

    rc = bn.cli.main(["project", "info"])

    assert rc == 0
    assert [(c["op"], c["params"], c["target"]) for c in calls] == [("project_info", {}, None)]
    out = capsys.readouterr().out
    assert "project: fw.bnpr" in out
    assert "files: 2" in out
    assert "123:4:1" in out


def test_project_files_params_default_and_flat(fake_transport, monkeypatch):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_files": {"ok": True, "result": {"files": [], "folders": []}}})

    assert bn.cli.main(["project", "files"]) == 0
    assert bn.cli.main(["project", "files", "--folder", "fw", "--flat"]) == 0

    assert calls[0]["params"] == {"folder": None, "recursive": True}
    assert calls[1]["params"] == {"folder": "fw", "recursive": False}


def test_project_load_params(fake_transport, monkeypatch):
    from bn.transport import REFRESH_REQUEST_TIMEOUT
    _one_instance(monkeypatch)
    loaded = {"ok": True, "result": {"loaded": True, "path": "/p/fw.bnpr/data/ab/cd",
                                     "targets": [], "project_file_id": FILE_ID_A,
                                     "project_path": "/fw/app", "substituted_from": None}}
    calls = fake_transport({"project_load": loaded})

    assert bn.cli.main(["project", "load", "fw/app"]) == 0
    assert bn.cli.main(["project", "load", FILE_ID_A[:8], "--no-bndb", "--quick"]) == 0

    assert calls[0]["params"] == {"selector": "fw/app", "prefer_bndb": True, "quick": False,
                                  "workdir": os.getcwd()}
    assert calls[0]["default_timeout"] == REFRESH_REQUEST_TIMEOUT
    assert calls[1]["params"] == {"selector": FILE_ID_A[:8], "prefer_bndb": False,
                                  "quick": True, "workdir": os.getcwd()}


def test_project_import_params(fake_transport, monkeypatch, tmp_path, capsys):
    _one_instance(monkeypatch)
    src = tmp_path / "blob.bin"
    src.write_bytes(b"\x7fELF")
    row = {"id": FILE_ID_A, "name": "blob.bin", "path": "/fw/blob.bin", "folder": "/fw",
           "description": "", "size": 4, "exists_on_disk": True, "is_database": False,
           "open_target_id": None, "sibling_database_id": None}
    calls = fake_transport({"project_import": {"ok": True, "result": row}})

    assert bn.cli.main(["project", "import", str(src)]) == 0
    assert bn.cli.main(["project", "import", str(src), "--folder", "fw", "--name", "blob.bin",
                        "--description", "d"]) == 0

    assert calls[0]["params"] == {"path": str(src.resolve())}
    assert calls[1]["params"] == {"path": str(src.resolve()), "folder": "fw",
                                  "name": "blob.bin", "description": "d"}
    out = capsys.readouterr().out
    assert "imported:" in out and "/fw/blob.bin" in out


def test_project_export_params(fake_transport, monkeypatch, tmp_path, capsys):
    _one_instance(monkeypatch)
    dest = tmp_path / "out.bin"
    calls = fake_transport({"project_export": {
        "ok": True, "result": {"exported": True, "dest": str(dest), "project_file_id": FILE_ID_A}}})

    assert bn.cli.main(["project", "export", "fw/app", str(dest)]) == 0

    assert calls[0]["params"] == {"selector": "fw/app", "dest": str(dest.resolve())}
    assert f"exported: {dest}" in capsys.readouterr().out


def test_project_delete_params(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_delete": {
        "ok": True, "result": {"deleted": True, "project_file_id": FILE_ID_A}}})

    assert bn.cli.main(["project", "delete", "fw/app"]) == 0

    assert calls[0]["params"] == {"selector": "fw/app"}
    assert FILE_ID_A in capsys.readouterr().out


def test_project_close_params_and_text(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_close": {"ok": True, "result": {
        "closed": True, "path": "/p/fw.bnpr",
        "targets_closed": [
            {"path": "/p/fw.bnpr/data/ab/cd", "unsaved": True, "engine_modified": True},
            {"path": "/p/fw.bnpr/data/ef/01", "unsaved": False, "engine_modified": False},
        ]}}})

    assert bn.cli.main(["project", "close"]) == 0
    assert bn.cli.main(["project", "close", "--force"]) == 0

    assert calls[0]["params"] == {"force": False}
    assert calls[1]["params"] == {"force": True}
    out = capsys.readouterr().out
    assert "closed project: /p/fw.bnpr" in out
    assert "- /p/fw.bnpr/data/ab/cd  [unsaved changes discarded]" in out
    assert "- /p/fw.bnpr/data/ef/01\n" in out


def test_project_refusal_is_exit_2(monkeypatch, capsys):
    from bn.transport import BridgeError
    _one_instance(monkeypatch)

    def refuse(op, **kwargs):
        raise BridgeError("No project is open in this bridge")

    monkeypatch.setattr(bn.cli, "send_request", refuse)

    rc = bn.cli.main(["project", "info"])

    assert rc == 2
    assert "No project is open in this bridge" in capsys.readouterr().err


# --- renderers ------------------------------------------------------------------


def _files_result():
    return {
        "files": [
            {"id": FILE_ID_A, "name": "app", "path": "/fw/app", "folder": "/fw",
             "description": "", "size": 4096, "exists_on_disk": True, "is_database": False,
             "open_target_id": None, "sibling_database_id": FILE_ID_B},
            {"id": FILE_ID_B, "name": "app.bndb", "path": "/fw/app.bndb", "folder": "/fw",
             "description": "", "size": 123456, "exists_on_disk": True, "is_database": True,
             "open_target_id": "123:4:1", "sibling_database_id": None},
        ],
        "folders": [{"id": "f-1", "path": "/fw"}],
    }


def test_project_files_text_rows(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_files": {"ok": True, "result": _files_result()}})

    assert bn.cli.main(["project", "files"]) == 0

    out = capsys.readouterr().out
    raw_line = next(line for line in out.splitlines() if "/fw/app" in line and "bndb" not in line.split()[1:2])
    db_line = next(line for line in out.splitlines() if "/fw/app.bndb" in line)
    assert raw_line.startswith(FILE_ID_A[:8])
    assert raw_line.split()[1] == "raw"
    assert "4096" in raw_line
    assert f"sibling db: {FILE_ID_B[:8]}" in raw_line
    assert "[open as" not in raw_line
    assert db_line.startswith(FILE_ID_B[:8])
    assert db_line.split()[1] == "bndb"
    assert "123456" in db_line
    assert "[open as -t fw/app.bndb]" in db_line
    assert out.index("folders:") > out.index("/fw/app.bndb")
    assert "- /fw" in out


def test_project_files_json_is_raw(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_files": {"ok": True, "result": _files_result()}})

    assert bn.cli.main(["project", "files", "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out) == _files_result()


def test_project_load_text_has_project_file_line(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    # The substitution is explained by the bridge's own note (BNPR_PLAN.md
    # §5.2 `notes`); the CLI prints that note once and adds no second line.
    note = f"Loaded the project database /fw/app.bndb instead of the raw file /fw/app ({FILE_ID_A})"
    fake_transport({"project_load": {"ok": True, "result": {
        "loaded": True, "path": "/p/fw.bnpr/data/ab/cd", "targets": [{"selector": "fw/app.bndb"}],
        "notes": [note], "project_file_id": FILE_ID_B, "project_path": "/fw/app.bndb",
        "substituted_from": FILE_ID_A}}})

    assert bn.cli.main(["project", "load", "fw/app"]) == 0

    out = capsys.readouterr().out
    assert "loaded: /p/fw.bnpr/data/ab/cd" in out
    assert f"Project file: /fw/app.bndb ({FILE_ID_B})" in out
    assert out.count(FILE_ID_A) == 1
    assert "- fw/app.bndb" in out


def test_save_text_project_file_line(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"save_database": {"ok": True, "result": {
        "ok": True, "saved": True, "path": "/p/fw.bnpr/data/ab/cd",
        "project_file_id": FILE_ID_B, "project_path": "/fw/app.bndb", "mode": "created"}}})

    assert bn.cli.main(["save", "-t", "fw/app"]) == 0

    out = capsys.readouterr().out
    assert "saved: /p/fw.bnpr/data/ab/cd" in out
    assert "project file: /fw/app.bndb (created)" in out


def test_save_text_without_project_has_no_project_line(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"save_database": {"ok": True, "result": {
        "ok": True, "saved": True, "path": "/tmp/app.bndb"}}})

    assert bn.cli.main(["save", "-t", "app"]) == 0

    assert "project file:" not in capsys.readouterr().out


def test_target_list_text_project_column_only_when_present(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    rows = [
        {"selector": "fw/app.bndb", "target_id": "123:4:1", "view_id": "4", "view_name": "ELF",
         "filename": "/p/fw.bnpr/data/ab/cd", "project_file_id": FILE_ID_B,
         "project_path": "/fw/app.bndb", "project_name": "app.bndb"},
        {"selector": "plain.so", "target_id": "123:5:2", "view_id": "5", "view_name": "ELF",
         "filename": "/tmp/plain.so", "project_file_id": "", "project_path": "",
         "project_name": ""},
    ]
    fake_transport({"list_targets": {"ok": True, "result": rows}})
    assert bn.cli.main(["target", "list"]) == 0
    out = capsys.readouterr().out
    assert "project: /fw/app.bndb" in out
    assert out.count("project:") == 1

    fake_transport({"list_targets": {"ok": True, "result": [rows[1]]}})
    assert bn.cli.main(["target", "list"]) == 0
    assert "project:" not in capsys.readouterr().out


def test_instance_list_text_shows_project(monkeypatch, capsys):
    owner = _instance("owner", 11, project={"path": "/p/fw.bnpr", "id": "p-1", "name": "fw.bnpr"})
    plain = _instance("plain", 22)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [owner, plain])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    assert bn.cli.main(["instance", "list"]) == 0
    out = capsys.readouterr().out
    owner_line = next(line for line in out.splitlines() if line.startswith("owner"))
    plain_line = next(line for line in out.splitlines() if line.startswith("plain"))
    assert "project=fw.bnpr" in owner_line
    assert "project=" not in plain_line

    assert bn.cli.main(["instance", "list", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    by_id = {item["instance_id"]: item for item in data["items"]}
    assert by_id["owner"]["project"] == {"path": "/p/fw.bnpr", "id": "p-1", "name": "fw.bnpr"}
    assert "project" not in by_id["plain"]
