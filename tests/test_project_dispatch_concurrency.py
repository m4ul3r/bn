"""Project mutations serialize real handlers, not copies of registry policy."""
from __future__ import annotations

import copy
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from _project_fakes import FakeProject
from test_project_findings import claim
from test_project_rootfs import RootfsProject, elf, exports, library, target, world


@pytest.fixture
def dispatch_world(world, monkeypatch):
    # Model native nonempty snapshot storage; retain all production dispatch,
    # ledger, project management, import, and linking code.
    monkeypatch.setattr(
        world.project, "create_file",
        lambda contents, folder, name, description="": world.project.add_file(
            name, folder, data=contents, description=description),
    )
    monkeypatch.setattr(world.instance, "_write_registry", lambda: None)
    return world


def _dispatch(w, op, params=None, **fields):
    return w.instance.dispatch({"op": op, "params": params or {}, **fields})


def test_project_files_paging_params_reach_the_handler_through_dispatch_1080(dispatch_world):
    """The binder is what carries `offset`/`limit` off the wire; a test that
    calls the handler directly cannot see it dropped."""
    w = dispatch_world
    for name in ("a", "b", "c", "d", "e"):
        w.project.add_file(name)

    response = _dispatch(w, "project_files", {"offset": 3, "limit": 1})

    assert response["ok"] is True, response
    page = response["result"]
    assert [row["name"] for row in page["items"]] == ["d"]
    assert (page["total"], page["offset"], page["limit"], page["returned"], page["has_more"]) == (
        5, 3, 1, 1, True)
    refused = _dispatch(w, "project_files", {"limit": 0})
    assert refused["ok"] is False and "limit must be a positive integer" in refused["error"]


def _start(call, *, observed=None):
    result = {}
    done = threading.Event()

    def run():
        try:
            result["response"] = call()
        finally:
            done.set()
            if observed is not None:
                observed.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, done, result


def _finish(job):
    thread, done, result = job
    thread.join(5)
    assert done.is_set(), "dispatch worker did not finish"
    return result["response"]


def _document(w, **changes):
    source = w.project.add_file("firmware", data=b"synthetic firmware")
    return claim(NS(source=source), **changes)


def _sync_source(w):
    root = w.tmp / "rootfs"
    (root / "lib").mkdir(parents=True)
    path = root / "lib" / "libdemo.so"
    path.write_bytes(b"\x7fELF")
    w.infos[str(path)] = elf(w)
    return root


def _state(w, view):
    return (
        w.instance.projects.current,
        [(pf.id, pf.get_path_in_project(), Path(pf.path_on_disk).read_bytes())
         for pf in w.project.files],
        copy.deepcopy(w.project.metadata),
        list(view.writes) if view is not None else [],
        # Everything on disk under the scratch root: project blobs, a created
        # `.bnpr`, an exported file. A mutation that slipped past the reader
        # shows up here even when it leaves the project object alone.
        sorted(str(p.relative_to(w.tmp)) for p in w.tmp.rglob("*")),
    )


@pytest.mark.parametrize("mutation", [
    "add", "update", "review", "delete", "import", "export", "open", "close", "create",
    "link", "sync",
])
def test_project_mutation_waits_for_existing_reader(dispatch_world, monkeypatch, mutation):
    """Downgrading any mutation to shared/no locking lets it change reader state."""
    w = dispatch_world
    view = None
    if mutation in {"add", "update", "review"}:
        document = _document(w)
        op = "project_finding_add"
        params = {"document": document, "author": "analyst"}
        if mutation != "add":
            initial = _dispatch(w, op, params)
            assert initial["ok"], initial
            params = {"id": initial["result"]["id"], "expected_revision": 1}
            if mutation == "update":
                op = "project_finding_update"
                params.update(author="analyst",
                              document={**document, "claim": "Serialized correction"})
            else:
                op = "project_finding_review"
                trace = w.project.add_file("trace", data=b"independent reproduction")
                params.update(reviewer="peer", document={
                    "verdict": "accept", "rationale": "Checked independently",
                    "evidence": [{"check": "file", "file": trace.id}]})
    elif mutation == "delete":
        source = w.project.add_file("obsolete")
        op, params = "project_delete", {"selector": source.id}
    elif mutation == "import":
        # A single import is not a bulk operation; drop the rootfs fake's
        # sync-only bulk assertion for this op.
        monkeypatch.setattr(w.project, "create_file_from_path",
                            FakeProject.create_file_from_path.__get__(w.project))
        blob = w.tmp / "blob.bin"
        blob.write_bytes(b"\x7fELF-imported")
        op, params = "project_import", {"path": str(blob), "folder": "fw/new"}
    elif mutation == "export":
        source = w.project.add_file("app")
        dest = w.tmp / "exported.bin"
        op, params = "project_export", {"selector": source.id, "dest": str(dest)}
    elif mutation == "open":
        w.instance.projects.close()
        op, params = "project_open", {"path": str(w.root)}
    elif mutation == "close":
        op, params = "project_close", {}
    elif mutation == "create":
        w.instance.projects.close()
        dest = w.tmp / "new.bnpr"
        created = RootfsProject(dest, "new")

        def create_project(path, name):
            dest.mkdir()
            (dest / "project.bnpm").write_bytes(b"SQLite format 3\x00")
            return created

        w.bn.Project = NS(create_project=create_project, open_project=lambda path: created)
        op, params = "project_create", {"path": str(dest), "name": "new",
                                        "rootfs": {"root": str(w.tmp), "platform": "linux"}}
    elif mutation == "link":
        backing = library(w, "/lib/libdemo.so", info=elf(w, exports=exports("parse")))
        view = target(w, needed=["libdemo.so"], symbols=["parse"])
        op, params = "project_link", {}
    else:
        root = _sync_source(w)
        op, params = "project_sync_libs", {"root": str(root)}

    before = _state(w, view)
    observed = threading.Event()
    condition = w.instance._target_lock._condition
    wait = condition.wait

    def waiting(timeout=None):
        # Observe actual contention without replacing lock semantics. If a
        # downgraded operation runs instead, worker completion also wakes us.
        observed.set()
        return wait(timeout)

    monkeypatch.setattr(condition, "wait", waiting)
    job = None
    try:
        with w.instance._target_lock.read():
            job = _start(lambda: _dispatch(w, op, params, target="app" if view else None),
                         observed=observed)
            assert observed.wait(5), "mutation neither waited nor completed"
            assert not job[1].is_set(), "mutation completed while a reader held the view state"
            assert _state(w, view) == before
    finally:
        if job is not None:
            response = _finish(job)
    assert response["ok"], response

    if mutation in {"add", "update", "review"}:
        finding = response["result"]
        assert finding["revision"] == (1 if mutation == "add" else 2)
        if mutation == "review":
            assert finding["review_state"] == "accepted"
        history = _dispatch(w, "project_finding_history", {"id": finding["id"]})
        assert history["ok"], history
        assert [item["revision"] for item in history["result"]["items"]] == list(
            range(1, finding["revision"] + 1))
    elif mutation == "delete":
        assert source not in w.project.files
        assert source in w.project.deleted
    elif mutation == "import":
        assert response["result"]["path"] == "/fw/new/blob.bin"
        assert w.instance.projects.resolve_file(response["result"]["id"]) in w.project.files
    elif mutation == "export":
        assert response["result"]["exported"] is True
        assert dest.read_bytes() == b"\x7fELF"
    elif mutation == "open":
        assert w.instance.projects.current.project is w.project
    elif mutation == "close":
        assert response["result"]["closed"] is True
        assert w.project.closed and w.instance.projects.current is None
    elif mutation == "create":
        assert response["result"]["created"] is True
        assert w.instance.projects.current.project is created
        assert w.instance.projects.get_rootfs()["libs"] == {}
    elif mutation == "link":
        assert view.externals["libdemo.so"].backing_file is backing
        assert view.locations[0].library is view.externals["libdemo.so"]
        assert view.locations[0].target_symbol == "parse"
    else:
        assert len(response["result"]["imported"]) == 1
        assert set(w.instance.projects.get_rootfs()["libs"]) == {pf.id for pf in w.project.files}


def test_competing_finding_updates_have_one_revision_winner(dispatch_world, monkeypatch):
    w = dispatch_world
    document = _document(w)
    added = _dispatch(w, "project_finding_add", {"document": document, "author": "analyst"})
    assert added["ok"], added
    finding_id = added["result"]["id"]
    entered, release = threading.Event(), threading.Event()
    create = w.project.create_file

    def paused_create(contents, folder, name, description=""):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(5), "test did not release snapshot append"
        return create(contents, folder, name, description)

    monkeypatch.setattr(w.project, "create_file", paused_create)
    params = {"id": finding_id, "expected_revision": 1, "author": "analyst",
              "document": {**document, "claim": "First correction"}}
    job = _start(lambda: _dispatch(w, "project_finding_update", params))
    competing = {**params, "document": {**document, "claim": "Competing correction"}}
    try:
        assert entered.wait(5), "first writer did not reach native snapshot creation"
        refused = _dispatch(w, "project_finding_update", competing, no_wait=True)
        assert refused["ok"] is False
        assert "busy:" in refused["error"], refused
    finally:
        release.set()
        winner = _finish(job)
    assert winner["ok"], winner
    assert winner["result"]["revision"] == 2
    stale = _dispatch(w, "project_finding_update", competing)
    assert stale["ok"] is False
    assert "current revision 2" in stale["error"]
    history = _dispatch(w, "project_finding_history", {"id": finding_id})
    assert history["ok"], history
    assert [row["revision"] for row in history["result"]["items"]] == [1, 2]
    shown = _dispatch(w, "project_finding_show", {"id": finding_id})
    assert shown["result"]["claim"] == "First correction"


def test_sync_discovery_keeps_reads_available(dispatch_world, monkeypatch):
    w = dispatch_world
    root = _sync_source(w)
    entered, release = threading.Event(), threading.Event()
    scan = w.ops.scan_libraries

    def paused_scan(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release library discovery"
        return scan(*args, **kwargs)

    monkeypatch.setattr(w.ops, "scan_libraries", paused_scan)
    job = _start(lambda: _dispatch(w, "project_sync_libs", {"root": str(root)}))
    try:
        assert entered.wait(5), "sync did not reach library discovery"
        read = _dispatch(w, "project_libs", no_wait=True)
        assert read["ok"], read
        assert not w.project.files
        assert w.instance.projects.get_rootfs()["libs"] == {}
    finally:
        release.set()
        response = _finish(job)
    assert response["ok"], response
    assert len(response["result"]["imported"]) == 1
    assert set(w.instance.projects.get_rootfs()["libs"]) == {pf.id for pf in w.project.files}
