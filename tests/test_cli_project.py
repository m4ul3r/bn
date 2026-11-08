"""`bn project` CLI tests: op/params wiring, the open/attach routing rule,
the rootfs `init`/`libs`/`link` commands and the documented text renderer lines."""
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

    # The project.bnpm spelling normalises to its parent .bnpr directory.
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
                                  "link": True, "workdir": os.getcwd()}
    assert calls[0]["default_timeout"] == REFRESH_REQUEST_TIMEOUT
    assert calls[1]["params"] == {"selector": FILE_ID_A[:8], "prefer_bndb": False,
                                  "quick": True, "link": True, "workdir": os.getcwd()}


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
    # The substitution is explained by the bridge's own note (`notes`); the
    # CLI prints that note once and adds no second line.
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


# --- project init (rootfs) ---------------------------------------------------


def _rootfs(tmp_path, name="rootfs"):
    """A fake rootfs tree; detection itself is monkeypatched, so only the
    directory and its project locations matter here."""
    root = tmp_path / name
    root.mkdir()
    return str(root.resolve())


def _patch_layout(monkeypatch, root, *, platform="linux", lookup=None, candidates=2,
                  detect=None):
    """Fake `bn.rootfs_layout` for the CLI: detection returns *root* (or None),
    lookup paths and the candidate scan are canned."""
    from bn import rootfs_layout

    seen = {}
    info = None if detect is False else rootfs_layout.RootfsInfo(
        root, platform, ["bin", "lib", "etc/ld.so.conf"])
    paths = lookup if lookup is not None else [
        rootfs_layout.LookupPath("/opt/vendor/lib", "ld_so_conf:/etc/ld.so.conf.d/vendor.conf"),
        rootfs_layout.LookupPath("/lib", "default"),
    ]

    def detect_root(start, *, explicit=False):
        seen["detect"] = (start, explicit)
        return info

    def scan_libraries(scanned_root, scanned_paths):
        seen["scan"] = (scanned_root, list(scanned_paths))
        return [object()] * candidates, ["note: /lib/escape.so resolves outside the root"]

    monkeypatch.setattr(bn.commands.project.rootfs_layout, "detect_root", detect_root)
    monkeypatch.setattr(bn.commands.project.rootfs_layout, "lookup_paths",
                        lambda r, p: list(paths) if p != "unknown" else [])
    monkeypatch.setattr(bn.commands.project.rootfs_layout, "scan_libraries", scan_libraries)
    return seen


def _patch_projects_dir(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "projects"
    monkeypatch.setattr(bn.commands.project.paths, "projects_dir", lambda: cache)
    return cache


def _sync_result(root, imported=2, skipped=0, changed=()):
    return {
        "root": root, "platform": "linux",
        "lookup_paths": [{"path": "/opt/vendor/lib", "source": "ld_so_conf:/etc/ld.so.conf.d/vendor.conf"},
                         {"path": "/lib", "source": "default"}],
        "imported": [{"id": FILE_ID_A, "name": "libc.so", "path": "/lib/libc.so", "folder": "/lib",
                      "description": "rootfs library; names: libc.so, ld-musl-aarch64.so.1",
                      "size": 1234, "exists_on_disk": True, "is_database": False,
                      "open_target_id": None, "sibling_database_id": None}][:imported],
        "skipped_existing": skipped,
        "changed": list(changed),
        "skipped_not_elf": 1,
        "notes": [],
    }


def test_project_help_lists_rootfs_commands(capsys):
    with pytest.raises(SystemExit):
        bn.cli.main(["project", "--help"])
    out = capsys.readouterr().out
    assert "init" in out and "libs" in out and "link" in out


def test_project_init_dry_run_needs_no_bridge(monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    seen = _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(bn.cli, "send_request",
                        lambda *a, **k: pytest.fail("dry-run must not send"))
    monkeypatch.setattr(bn.cli, "list_instances",
                        lambda: pytest.fail("dry-run must not consult the registry"))
    monkeypatch.chdir(root)

    rc = bn.cli.main(["project", "init", "--dry-run"])

    assert rc == 0
    assert seen["detect"] == (os.getcwd(), False)
    assert seen["scan"][0] == root and len(seen["scan"][1]) == 2
    out = capsys.readouterr().out
    assert f"rootfs: {root}" in out
    assert "platform: linux" in out
    assert "markers: bin, lib, etc/ld.so.conf" in out
    assert f"project: {root}.bnpr  [new]" in out
    assert "/opt/vendor/lib  (ld_so_conf:/etc/ld.so.conf.d/vendor.conf)" in out
    assert "/lib  (default)" in out
    assert "candidate libraries: 2" in out
    assert "note: /lib/escape.so resolves outside the root" in out
    assert "dry-run: nothing was created" in out
    assert not os.path.exists(root + ".bnpr")


def test_project_init_dry_run_json_and_explicit_root(monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    seen = _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)

    rc = bn.cli.main(["project", "init", root, "--dry-run", "--format", "json"])

    assert rc == 0
    assert seen["detect"] == (root, True)
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True
    assert data["root"] == root and data["platform"] == "linux"
    assert data["project_path"] == root + ".bnpr" and data["project_exists"] is False
    assert data["name"] == "rootfs"
    assert data["lookup_paths"][0] == {"path": "/opt/vendor/lib",
                                       "source": "ld_so_conf:/etc/ld.so.conf.d/vendor.conf"}
    assert data["candidate_libraries"] == 2


def test_project_init_outside_a_rootfs_is_exit_2(monkeypatch, tmp_path, capsys):
    import bn.commands.project
    _patch_layout(monkeypatch, str(tmp_path), detect=False)
    monkeypatch.chdir(tmp_path)

    rc = bn.cli.main(["project", "init"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "not inside a firmware root filesystem" in err
    assert f"looked from {os.getcwd()} up to /" in err
    assert "pass the root explicitly" in err


def test_project_init_creates_syncs_and_pins(fake_transport, monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    pins = _capture_pins(monkeypatch)
    project_path = root + ".bnpr"
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(project_path, created=True,
                                                               name="rootfs"),
                           "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": [os.getcwd()]}},
        "project_sync_libs": {"ok": True, "result": _sync_result(root)},
    })

    rc = bn.cli.main(["project", "init", root])

    assert rc == 0
    assert [c["op"] for c in calls] == ["project_create", "associate_project_roots",
                                        "project_sync_libs"]
    assert calls[0]["params"] == {"path": project_path, "name": "rootfs",
                                  "rootfs": {"root": root, "platform": "linux"}}
    assert calls[0]["spawn_missing_named"] is True
    assert calls[2]["params"] == {"root": root, "platform": "linux", "dry_run": False}
    assert calls[2]["instance_id"] == "bridge-a"
    assert pins == [{"instance_id": "bridge-a"}]
    out = capsys.readouterr().out
    assert f"project: {project_path}  [created]  instance: bridge-a" in out
    assert "platform: linux" in out
    assert "/opt/vendor/lib  (ld_so_conf:" in out
    assert "libraries: imported 1 / skipped 0 / changed 0" in out
    assert "/lib/libc.so" in out
    assert "skipped (not ELF): 1" in out
    assert "next: `bn project files`" in out and "bn project load <path-in-project>" in out


def test_project_init_name_override(fake_transport, monkeypatch, tmp_path):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(root + ".bnpr", created=True)},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    assert bn.cli.main(["project", "init", root, "--name", "camera-fw", "--no-libs"]) == 0

    assert calls[0]["params"]["name"] == "camera-fw"


def test_project_init_opens_existing_preferred_project(fake_transport, monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    project_path = root + ".bnpr"
    os.mkdir(project_path)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_open": {"ok": True, "result": _project_info(project_path, opened=True),
                         "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
        "project_sync_libs": {"ok": True, "result": _sync_result(root, imported=0, skipped=2)},
    })

    rc = bn.cli.main(["project", "init", root])

    assert rc == 0
    assert [c["op"] for c in calls] == ["project_open", "associate_project_roots",
                                        "project_sync_libs"]
    assert calls[0]["params"] == {"path": project_path}
    out = capsys.readouterr().out
    assert f"project: {project_path}  [opened]" in out
    assert "libraries: imported 0 / skipped 2 / changed 0" in out


def test_project_init_prefers_existing_fallback_over_new_preferred(
    fake_transport, monkeypatch, tmp_path
):
    import bn.commands.project
    from bn import rootfs_layout
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    cache = _patch_projects_dir(monkeypatch, tmp_path)
    _preferred, fallback = rootfs_layout.project_location(root, str(cache))
    os.makedirs(fallback)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_open": {"ok": True, "result": _project_info(fallback, opened=True)},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    assert bn.cli.main(["project", "init", root, "--no-libs"]) == 0

    assert calls[0]["op"] == "project_open"
    assert calls[0]["params"] == {"path": fallback}


def test_project_init_falls_back_to_cache_when_parent_unwritable(
    fake_transport, monkeypatch, tmp_path
):
    import bn.commands.project
    from bn import rootfs_layout
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    cache = _patch_projects_dir(monkeypatch, tmp_path)
    _preferred, fallback = rootfs_layout.project_location(root, str(cache))
    monkeypatch.setattr(bn.commands.project.os, "access", lambda p, mode: False)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(fallback, created=True)},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    assert bn.cli.main(["project", "init", root, "--no-libs"]) == 0

    assert calls[0]["op"] == "project_create"
    assert calls[0]["params"]["path"] == fallback


def test_project_init_attaches_to_the_owner_then_syncs(
    fake_transport, monkeypatch, tmp_path, capsys
):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    project_path = root + ".bnpr"
    os.mkdir(project_path)
    owner = _instance("owner", 11, project={"path": project_path, "id": "p-1", "name": "rootfs"})
    other = _instance("other", 22)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [other, owner])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    pins = _capture_pins(monkeypatch)
    calls = fake_transport({"project_sync_libs": {"ok": True, "result": _sync_result(root)}})

    rc = bn.cli.main(["project", "init", root])

    assert rc == 0
    # No project_open/create: the owner is pinned and only the sync is sent to it.
    assert [c["op"] for c in calls] == ["project_sync_libs"]
    assert calls[0]["instance_id"] == "owner"
    assert pins == [{"instance_id": "owner"}]
    out = capsys.readouterr().out
    assert f"project: {project_path}  [attached]  instance: owner" in out


def test_project_init_no_libs_skips_sync(fake_transport, monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(root + ".bnpr", created=True),
                           "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    rc = bn.cli.main(["project", "init", root, "--no-libs"])

    assert rc == 0
    assert [c["op"] for c in calls] == ["project_create", "associate_project_roots"]
    out = capsys.readouterr().out
    assert "libraries: not synced (--no-libs)" in out
    assert "/opt/vendor/lib  (ld_so_conf:" in out


def test_project_init_unknown_platform_skips_sync(fake_transport, monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root, platform="unknown", candidates=0)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(root + ".bnpr", created=True),
                           "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })

    rc = bn.cli.main(["project", "init", root])

    assert rc == 0
    assert [c["op"] for c in calls] == ["project_create", "associate_project_roots"]
    assert calls[0]["params"]["rootfs"] == {"root": root, "platform": "unknown"}
    assert "libraries: not synced (platform unknown" in capsys.readouterr().out


def test_project_init_sync_failure_after_open_is_exit_1(
    fake_transport, monkeypatch, tmp_path, capsys
):
    from bn.transport import BridgeError
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    pins = _capture_pins(monkeypatch)
    project_path = root + ".bnpr"
    calls = []

    def send(op, *, params=None, target=None, timeout=None, default_timeout=None,
             connect_retries=4, instance_id=None, spawn_missing_named=False,
             resolved=False, idle_probe=False):
        calls.append(op)
        if op == "project_create":
            return {"ok": True, "result": _project_info(project_path, created=True),
                    "instance": "bridge-a"}
        if op == "associate_project_roots":
            return {"ok": True, "result": {"associated": []}}
        raise BridgeError("Binary Ninja could not import /lib/libc.so: disk full")

    monkeypatch.setattr(bn.cli, "send_request", send)

    rc = bn.cli.main(["project", "init", root])

    assert rc == 1
    assert calls == ["project_create", "associate_project_roots", "project_sync_libs"]
    # The project is open and pinned; only the import step failed.
    assert pins == [{"instance_id": "bridge-a"}]
    captured = capsys.readouterr()
    assert "error: project_sync_libs: Binary Ninja could not import" in captured.err
    assert f"project: {project_path}  [created]" in captured.out
    assert "libraries: sync FAILED -- Binary Ninja could not import" in captured.out


def test_project_init_json_summary(fake_transport, monkeypatch, tmp_path, capsys):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    _one_instance(monkeypatch)
    _capture_pins(monkeypatch)
    fake_transport({
        "project_create": {"ok": True, "result": _project_info(root + ".bnpr", created=True),
                           "instance": "bridge-a"},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
        "project_sync_libs": {"ok": True, "result": _sync_result(root)},
    })

    assert bn.cli.main(["project", "init", root, "--format", "json"]) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["created"] is True and data["attached"] is False
    assert data["instance"] == "bridge-a"
    assert data["project"]["path"] == root + ".bnpr"
    assert data["sync"]["skipped_existing"] == 0
    assert data["sync_error"] is None


# --- project libs ---------------------------------------------------------------


def _libs_result():
    return {
        "root": "/fw/rootfs", "platform": "linux", "synced_at": "2026-11-08T10:00:00Z",
        "libs": [
            {"project_file_id": FILE_ID_A, "project_path": "/lib/libc.so",
             "rel_realpath": "/lib/libc.so", "soname": "libc.so",
             "names": ["libc.so", "ld-musl-aarch64.so.1"], "e_machine": 183, "elf_class": 64,
             "sibling_database_id": None},
            {"project_file_id": FILE_ID_B, "project_path": "/opt/vendor/lib/libgreet.so.1.0",
             "rel_realpath": "/opt/vendor/lib/libgreet.so.1.0", "soname": "libgreet.so.1",
             "names": ["libgreet.so.1.0", "libgreet.so.1"], "e_machine": 183, "elf_class": 64,
             "sibling_database_id": "db-1"},
        ],
    }


def test_project_libs_table(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_libs": {"ok": True, "result": _libs_result()}})

    assert bn.cli.main(["project", "libs"]) == 0

    assert [(c["op"], c["params"], c["target"]) for c in calls] == [("project_libs", {}, None)]
    out = capsys.readouterr().out
    assert "rootfs: /fw/rootfs  (linux)  synced: 2026-11-08T10:00:00Z" in out
    assert FILE_ID_A[:8] in out and "64/183" in out and "libc.so" in out
    assert "/opt/vendor/lib/libgreet.so.1.0  libgreet.so.1.0, libgreet.so.1" in out
    assert "[db db-1]" in out


def test_project_libs_empty_hints_sync(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_libs": {"ok": True, "result": {
        "root": "/fw/rootfs", "platform": "linux", "synced_at": None, "libs": []}}})

    assert bn.cli.main(["project", "libs"]) == 0

    out = capsys.readouterr().out
    assert "(no rootfs libraries)" in out and "bn project libs --sync" in out


def test_project_libs_sync_params_and_text(fake_transport, monkeypatch, tmp_path, capsys):
    _one_instance(monkeypatch)
    root = _rootfs(tmp_path)
    calls = fake_transport({"project_sync_libs": {"ok": True, "result": _sync_result(
        root, changed=[{"rel_realpath": "/lib/libc.so", "project_file_id": FILE_ID_A}])}})

    assert bn.cli.main(["project", "libs", "--sync"]) == 0
    assert bn.cli.main(["project", "libs", "--sync", "--root", root, "--dry-run"]) == 0

    assert calls[0]["op"] == "project_sync_libs"
    assert calls[0]["params"] == {"dry_run": False}
    assert calls[1]["params"] == {"dry_run": True, "root": root}
    out = capsys.readouterr().out
    assert f"rootfs: {root}  (linux)" in out
    assert "imported 1 / skipped 0 / changed 1" in out
    assert f"/lib/libc.so  ({FILE_ID_A[:8]})" in out


def test_project_libs_root_without_sync_is_refused(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport()

    assert bn.cli.main(["project", "libs", "--root", "/fw"]) == 2

    assert calls == []
    assert "--root and --dry-run only apply with --sync" in capsys.readouterr().err


# --- project link ---------------------------------------------------------------


def _link_result(status="verified", *, dry_run=False, linked=3, relinked=0, **extra):
    failed = status == "verification_failed"
    result = {
        "status": status,
        "success": not failed,
        "preview": dry_run,
        "dry_run": dry_run,
        "committed": (not dry_run) and not failed,
        "libraries": [
            {"name": "libgreet.so.1", "resolved": True, "project_file_id": FILE_ID_B,
             "project_path": "/opt/vendor/lib/libgreet.so.1.0", "backing": "raw"},
            {"name": "libc.so", "resolved": True, "project_file_id": FILE_ID_A,
             "project_path": "/lib/libc.so", "backing": "database"},
            {"name": "libmissing.so", "resolved": False, "project_file_id": None,
             "project_path": None, "backing": None},
        ],
        "scope": ["libgreet.so.1", "libc.so"],
        "linked": linked, "already_linked": 1, "relinked": relinked,
        "unresolved": [{"symbol": "_ITM_registerTMCloneTable",
                        "reason": "not exported by any linked library"},
                       {"symbol": "vendor_hash", "reason": "library libmissing.so not in project"}],
        "ambiguous_libraries": [],
        "measured": True,
    }
    result.update(extra)
    return result


def test_project_link_params_and_compact_text(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_link": {"ok": True, "result": _link_result()}})

    rc = bn.cli.main(["project", "link", "-t", "usr/bin/app"])

    assert rc == 0
    assert calls[0]["op"] == "project_link"
    assert calls[0]["params"] == {"dry_run": False, "relink": False}
    assert calls[0]["target"] == "usr/bin/app"
    out = capsys.readouterr().out
    assert "mutation: committed  changed=3  verified=3  noop=1  failed=0  dirty_after=True" in out
    assert "libgreet.so.1            -> /opt/vendor/lib/libgreet.so.1.0  (raw)" in out
    assert "libc.so                  -> /lib/libc.so  (database)" in out
    assert "libmissing.so            -> (not in project)" in out
    assert "linked 3  relinked 0  unresolved 2" in out
    assert "_ITM_registerTMCloneTable  not exported by any linked library" in out
    assert "vendor_hash  library libmissing.so not in project" in out
    assert "next: `bn -t usr/bin/app save`" in out


def test_project_link_preview_sends_dry_run_and_relink(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    calls = fake_transport({"project_link": {"ok": True, "result": _link_result(dry_run=True)}})

    rc = bn.cli.main(["project", "link", "-t", "usr/bin/app", "--preview", "--relink"])

    assert rc == 0
    assert calls[0]["params"] == {"dry_run": True, "relink": True}
    out = capsys.readouterr().out
    assert "mutation: preview" in out
    assert "dirty_after=False" in out


def test_project_link_noop_is_exit_0_and_clean(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_link": {"ok": True, "result": _link_result("noop", linked=0)}})

    assert bn.cli.main(["project", "link", "-t", "usr/bin/app"]) == 0

    out = capsys.readouterr().out
    assert "changed=0" in out and "dirty_after=False" in out
    assert "linked 0  relinked 0  unresolved 2" in out


def test_project_link_verification_failed_is_exit_3(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_link": {"ok": True, "result": _link_result(
        "verification_failed", message="libc.so did not read back on printf")}})

    rc = bn.cli.main(["project", "link", "-t", "usr/bin/app"])

    assert rc == 3
    out = capsys.readouterr().out
    assert "mutation: FAILED" in out
    assert "first_error: libc.so did not read back on printf" in out


def test_project_link_unreadable_counter_is_exit_4(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_link": {"ok": True, "result": _link_result(linked="three")}})

    rc = bn.cli.main(["project", "link", "-t", "usr/bin/app"])

    assert rc == 4
    out = capsys.readouterr().out
    assert "warning: unmeasured" in out
    assert "this op's own counters could not be read" in out
    assert "linked ?" in out


def test_project_link_json_summary_and_verbose_detail(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    fake_transport({"project_link": {"ok": True, "result": _link_result()}})

    assert bn.cli.main(["project", "link", "-t", "usr/bin/app", "--summary",
                        "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "mutation_summary"
    assert summary["ok"] is True and summary["measured"] is True
    assert summary["changed_count"] == 3 and summary["noop_count"] == 1
    assert summary["project_link"]["linked"] == 3
    assert [u["symbol"] for u in summary["project_link"]["unresolved"]] == [
        "_ITM_registerTMCloneTable", "vendor_hash"]
    assert summary["project_link"]["libraries"][0]["name"] == "libgreet.so.1"

    assert bn.cli.main(["project", "link", "-t", "usr/bin/app", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "project link: verified" in out
    assert "scope: libgreet.so.1, libc.so" in out
    assert "linked 3  relinked 0  unresolved 2" in out


# --- project load --no-link ------------------------------------------------------


def test_project_load_no_link_and_linked_line(fake_transport, monkeypatch, capsys):
    _one_instance(monkeypatch)
    base = {"loaded": True, "path": "/p/fw.bnpr/data/ab/cd", "targets": [],
            "project_file_id": FILE_ID_A, "project_path": "/usr/bin/app",
            "substituted_from": None}
    calls = fake_transport({"project_load": {"ok": True, "result": {
        **base, "link": {"status": "verified", "linked": 7, "relinked": 0,
                         "unresolved": [{"symbol": "_ITM_registerTMCloneTable",
                                         "reason": "not exported by any linked library"}]}}}})

    assert bn.cli.main(["project", "load", "usr/bin/app"]) == 0
    assert "linked: 7 (1 unresolved)" in capsys.readouterr().out

    no_link_calls = fake_transport({
        "project_load": {"ok": True, "result": {**base, "link": None}}})
    assert bn.cli.main(["project", "load", "usr/bin/app", "--no-link"]) == 0
    assert "linked:" not in capsys.readouterr().out

    assert calls[0]["params"]["link"] is True
    assert no_link_calls[0]["params"]["link"] is False


@pytest.mark.parametrize("explicit", [False, True])
def test_project_init_multiple_bridges_need_explicit_instance(
    fake_transport, monkeypatch, tmp_path, capsys, explicit
):
    import bn.commands.project
    root = _rootfs(tmp_path)
    _patch_layout(monkeypatch, root)
    _patch_projects_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [
        _instance("one", 11), _instance("two", 22)])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    _capture_pins(monkeypatch)
    calls = fake_transport({
        "project_create": {"ok": True, "result": _project_info(root + ".bnpr", created=True),
                           "instance": "fresh"},
        "associate_project_roots": {"ok": True, "result": {"associated": []}},
    })
    argv = ["project", "init", root, "--no-libs"]
    if explicit:
        argv.extend(["--instance-id", "fresh"])

    assert bn.cli.main(argv) == (0 if explicit else 2)

    if explicit:
        assert calls[0]["instance_id"] == "fresh"
        assert calls[0]["spawn_missing_named"] is True
    else:
        assert calls == []
        assert "bn project init refuses ambient instance routing" in capsys.readouterr().err


@pytest.mark.parametrize("options", [[], ["--verbose"], ["--format", "json"]])
@pytest.mark.parametrize("counter", ["linked", "relinked", "already_linked", "unresolved"])
def test_project_link_unreadable_counts_fail_safe_in_every_output_mode(
    fake_transport, monkeypatch, capsys, options, counter
):
    _one_instance(monkeypatch)
    result = _link_result()
    result[counter] = {"unreadable": True}
    fake_transport({"project_link": {"ok": True, "result": result}})

    assert bn.cli.main(["project", "link", "-t", "usr/bin/app", *options]) == 4
