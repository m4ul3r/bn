"""Bridge teardown lifecycle for private cache registries (#694).

Detached load workers can outlive ``stop()``. These tests pin the teardown
latch, load-job quiescence, durable process identity, and project-association
restoration so a late worker cannot re-publish a dead bridge registry.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from _bridge_fakes import _load_bridge


@pytest.fixture
def instance(monkeypatch, tmp_path):
    """A headless bridge instance whose cache tree is this test's tmp_path."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="life1")
    assert str(tmp_path) in str(inst.registry_path)
    return inst


# --------------------------------------------------------------------------
# Durable process identity (the producer side of transport's verification)
# --------------------------------------------------------------------------


def test_registry_records_boot_id_and_process_start_time(instance):
    # Start ticks are "since boot", so they are unique only WITHIN a boot while
    # registries live in a persistent cache dir: without the boot id an old record
    # could falsely match a fresh process after a reboot (#694).
    from bn.proc_identity import boot_id, process_start_ticks

    instance._write_registry()

    payload = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert payload["pid_start_ticks"] == process_start_ticks(payload["pid"])
    assert payload["boot_id"] == boot_id()


# --------------------------------------------------------------------------
# Teardown latch: no publication after stop()
# --------------------------------------------------------------------------


def test_registry_write_after_stop_is_refused(instance):
    instance._write_registry()
    instance._server = object()  # simulate a bound socket, per #585's gate
    assert instance.registry_path.exists()

    instance.stop()
    assert not instance.registry_path.exists()

    # A detached load worker finishing here must not resurrect the registry.
    instance._write_registry()

    assert not instance.registry_path.exists()


def test_project_association_after_stop_is_refused(instance, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    instance.stop()

    assert instance._record_project_root(str(project)) is None
    result = instance._associate_project_roots([str(project)])

    assert result["associated"] == []
    assert result["skipped"] == [
        {"path": str(project), "reason": "bridge is shutting down"}
    ]
    assert not instance.registry_path.exists()


# --------------------------------------------------------------------------
# Detached load jobs are quiesced by teardown
# --------------------------------------------------------------------------


def test_stop_refuses_queued_load_jobs(instance):
    instance._load_jobs["queued-job"] = {
        "job_id": "queued-job",
        "state": "queued",
        "path": "/tmp/app.bin",
        "created_at": "2026-01-01T00:00:00Z",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
    }

    instance.stop()

    job = instance._load_jobs["queued-job"]
    assert job["state"] == "failed"
    assert job["error"] == "bridge stopped before the load started"
    assert job["finished_at"] is not None


def test_stop_leaves_terminal_load_jobs_alone(instance):
    instance._load_jobs["done"] = {"state": "complete", "result": {"loaded": True}}
    instance._load_jobs["boom"] = {"state": "failed", "error": "ValueError: nope"}

    instance.stop()

    assert instance._load_jobs["done"] == {"state": "complete", "result": {"loaded": True}}
    assert instance._load_jobs["boom"]["error"] == "ValueError: nope"


class _GatedLock:
    """Wraps a lock and holds ONE named thread at its door.

    Makes a lock-ordering interleaving deterministic: the gated thread announces
    that it has reached the lock and waits for the test to release it, so the test
    can run the competing operation first and know the gated thread acquires the
    lock strictly afterwards.
    """

    def __init__(self, inner, thread_prefix):
        self._inner = inner
        self._prefix = thread_prefix
        self.arrived = threading.Event()
        self.release = threading.Event()

    def _gated(self):
        return threading.current_thread().name.startswith(self._prefix)

    def __enter__(self):
        if self._gated():
            self.arrived.set()
            assert self.release.wait(5.0), "gated thread was never released"
        return self._inner.__enter__()

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)


class _GatedEnter:
    """Wraps a lock; parks selected threads BEFORE the underlying acquisition.

    Parking before the inner acquire is what makes an interleaving reproducible:
    the parked thread holds nothing, so the competing thread can take the lock
    while the test decides who proceeds next.
    """

    def __init__(self, inner):
        self._inner = inner
        self._gates: dict[str, tuple[threading.Event, threading.Event]] = {}
        self.acquired_by: list[str] = []

    def gate(self, prefix, arrived=None, release=None):
        arrived = arrived or threading.Event()
        release = release or threading.Event()
        self._gates[prefix] = (arrived, release)
        return arrived, release

    def __enter__(self):
        name = threading.current_thread().name
        for prefix, (arrived, release) in self._gates.items():
            if name.startswith(prefix):
                arrived.set()
                assert release.wait(5.0), f"{name} was never released"
                break
        result = self._inner.__enter__()
        self.acquired_by.append(name)
        return result

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)


def test_worker_never_starts_a_load_after_the_latch_is_set(instance, monkeypatch, tmp_path):
    # THE interleaving from the finding, driven deterministically:
    #   1. the worker has passed its latch check and is heading for the jobs lock;
    #   2. stop() has latched but is paused before acquiring the jobs lock;
    #   3. the worker then reaches the jobs lock.
    # With the latch check and the queued->running transition split across two lock
    # acquisitions (the old shape), step 3 flips the job to `running` and starts a
    # load into a process whose teardown had already latched. With both under ONE
    # teardown-lock hold, the worker cannot get in while stop() holds that lock, so
    # it observes the latch and refuses instead (#694).
    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")

    worker_ready = threading.Event()
    allow_worker = threading.Event()

    # Hook for the OLD shape: its latch check was a separate `_is_stopped()` call
    # that released the lock before the transition. Returning a stale False here is
    # exactly the observation it made before stop() latched.
    def stale_is_stopped():
        worker_ready.set()
        assert allow_worker.wait(5.0)
        return False

    monkeypatch.setattr(instance, "_is_stopped", stale_is_stopped, raising=False)

    # Hook for the CURRENT shape: the latch read lives inside the teardown lock, so
    # the worker is parked at that door instead -- holding nothing, so stop() can
    # still take the lock.
    teardown = _GatedEnter(instance._teardown_lock)
    teardown.gate("bn-load-", worker_ready, allow_worker)
    monkeypatch.setattr(instance, "_teardown_lock", teardown)

    jobs = _GatedEnter(instance._load_jobs_lock)
    stopper_at_jobs, allow_stopper = jobs.gate("stopper")
    monkeypatch.setattr(instance, "_load_jobs_lock", jobs)

    monkeypatch.setattr(
        instance,
        "_load_binary",
        lambda *a, **k: pytest.fail("a load must not start after teardown latched"),
    )

    job = instance._load_binary_async(str(binary))
    assert worker_ready.wait(5.0)              # worker past/at its latch check

    stopper = threading.Thread(target=instance.stop, name="stopper")
    stopper.start()
    assert stopper_at_jobs.wait(5.0)           # stop() latched, paused pre-jobs-lock
    assert instance._stopped is True

    allow_worker.set()                         # the worker now races for the job
    time.sleep(0.3)                            # ... and gets every chance to win
    allow_stopper.set()
    stopper.join(timeout=5.0)
    assert not stopper.is_alive()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if instance._load_jobs[job["job_id"]]["state"] != "queued":
            break
        time.sleep(0.01)

    record = instance._load_jobs[job["job_id"]]
    assert record["state"] == "failed"
    assert record["error"] == "bridge stopped before the load started"
    assert record["started_at"] is None        # it never entered `running`


def test_stop_refuses_a_queued_job_a_worker_has_not_reached(instance, monkeypatch, tmp_path):
    # The complementary path through the same lock: stop() wins outright, so the
    # worker finds its job already terminal and returns without loading.
    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    gate = _GatedLock(instance._teardown_lock, "bn-load-")
    monkeypatch.setattr(instance, "_teardown_lock", gate)
    monkeypatch.setattr(
        instance,
        "_load_binary",
        lambda *a, **k: pytest.fail("a load must not start after teardown latched"),
    )

    job = instance._load_binary_async(str(binary))
    assert gate.arrived.wait(5.0)              # worker parked at the lock
    instance.stop()
    gate.release.set()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if instance._load_jobs[job["job_id"]]["state"] != "queued":
            break
        time.sleep(0.01)

    assert instance._load_jobs[job["job_id"]]["state"] == "failed"


def test_registry_write_snapshots_project_roots_under_the_teardown_lock(
    instance, monkeypatch, tmp_path
):
    # An older concurrent write must not overwrite a root restored by restart.
    project = tmp_path / "project"
    project.mkdir()
    gate = _GatedLock(instance._teardown_lock, "regwriter")
    monkeypatch.setattr(instance, "_teardown_lock", gate)

    writer = threading.Thread(target=instance._write_registry, name="regwriter")
    writer.start()
    assert gate.arrived.wait(5.0)

    result = instance._associate_project_roots([str(project)])
    assert result["associated"] == [str(project)]

    gate.release.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive()

    payload = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert payload["project_roots"] == [str(project)]


def test_stop_joins_a_running_load_worker_and_suppresses_its_publication(
    instance, monkeypatch, tmp_path
):
    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    project = tmp_path / "project"
    project.mkdir()
    release = threading.Event()
    finished = threading.Event()

    def slow_load(*args, **kwargs):
        release.wait(5.0)
        instance._record_project_root(str(project))
        instance._write_registry()
        finished.set()
        return {"loaded": True, "path": str(binary)}

    monkeypatch.setattr(instance, "_load_binary", slow_load)
    instance._record_project_root(str(project))
    instance._write_registry()
    instance._server = object()  # simulate a bound socket, per #585's gate
    job = instance._load_binary_async(str(binary))
    assert job["job_id"] in instance._load_job_threads

    started = time.monotonic()
    stopper = threading.Thread(target=lambda: instance.stop(load_join_timeout=0.2))
    stopper.start()
    stopper.join(timeout=5.0)
    elapsed = time.monotonic() - started

    assert not stopper.is_alive()
    assert elapsed < 4.0
    assert not instance.registry_path.exists()

    release.set()
    assert finished.wait(5.0)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if instance._load_jobs[job["job_id"]]["state"] == "complete":
            break
        time.sleep(0.01)

    assert instance._load_jobs[job["job_id"]]["state"] == "complete"
    assert not instance.registry_path.exists()


# --------------------------------------------------------------------------
# associate_project_roots: session start and restart registry ownership
# --------------------------------------------------------------------------


def test_associate_project_roots_records_private_ownership(instance, tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    result = instance._associate_project_roots([str(project)])

    assert result == {
        "instance_id": "life1",
        "associated": [str(project)],
        "skipped": [],
    }
    payload = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert payload["project_roots"] == [str(project)]
    assert not list(project.glob(".bn-*"))


def test_associate_project_roots_skips_a_vanished_root(instance, tmp_path):
    missing = tmp_path / "deleted-project"

    result = instance._associate_project_roots([str(missing)])

    assert result["associated"] == []
    assert result["skipped"] == [
        {"path": str(missing), "reason": "project directory does not exist"}
    ]


@pytest.mark.parametrize("roots", ["/tmp/project", {"path": "x"}, [1], None, [None]])
def test_associate_project_roots_rejects_malformed_roots(instance, roots):
    with pytest.raises(RuntimeError, match="list of strings"):
        instance._associate_project_roots(roots)


def test_associate_project_roots_deduplicates_repeated_roots(instance, tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    result = instance._associate_project_roots([str(project), str(project)])

    assert result["associated"] == [str(project)]


def test_associate_project_roots_supports_gui_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    module = _load_bridge(monkeypatch)
    gui = module.BinaryNinjaBridge()
    project = tmp_path / "project"
    project.mkdir()

    assert gui._associate_project_roots([str(project)])["associated"] == [str(project)]


def test_associate_project_roots_op_is_read_locked_and_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="life1")
    project = tmp_path / "project"
    project.mkdir()

    spec = module.REGISTRY.spec("associate_project_roots")

    assert spec is not None and spec.lock == "read"
    assert "associate_project_roots" in module.READ_LOCKED_OPS
    assert spec.binder(inst, {"roots": [str(project)]}, None)["associated"] == [
        str(project)
    ]


# --------------------------------------------------------------------------
# stop()/start_bridge()/restart_bridge() ownership hygiene (#585)
# --------------------------------------------------------------------------


def test_stop_on_unbound_server_does_not_unlink_another_instances_files(
    monkeypatch, tmp_path
):
    # An instance whose _server is None (start() never ran, e.g. a refused
    # bind) must not unlink ANOTHER live instance's socket/registry just
    # because they happen to share the same on-disk paths (#585).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    live = module.BinaryNinjaBridge(instance_id="live1")
    live._write_registry()
    live.socket_path.parent.mkdir(parents=True, exist_ok=True)
    live.socket_path.touch()
    assert live.registry_path.exists()
    assert live.socket_path.exists()

    unbound = module.BinaryNinjaBridge(instance_id="live1")  # same paths, never start()ed
    assert unbound._server is None

    unbound.stop()

    assert live.registry_path.exists()
    assert live.socket_path.exists()


def test_stop_on_bound_server_does_unlink_its_own_files(monkeypatch, tmp_path):
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="bound1")
    inst._write_registry()
    inst.socket_path.parent.mkdir(parents=True, exist_ok=True)
    inst.socket_path.touch()
    inst._server = object()  # simulate a bound socket without a real listener

    inst.stop()

    assert not inst.registry_path.exists()
    assert not inst.socket_path.exists()


def test_start_bridge_clears_global_when_start_raises(monkeypatch):
    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module, "ui", object())
    monkeypatch.setattr(module, "_bridge", None)

    class _BoomBridge:
        def start(self):
            raise RuntimeError("bind failed")

    monkeypatch.setattr(module, "BinaryNinjaBridge", _BoomBridge)

    with pytest.raises(RuntimeError, match="bind failed"):
        module.start_bridge()

    assert module._bridge is None


def test_start_bridge_sets_global_only_after_successful_start(monkeypatch):
    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module, "ui", object())
    monkeypatch.setattr(module, "_bridge", None)

    class _FakeBridge:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(module, "BinaryNinjaBridge", _FakeBridge)

    module.start_bridge()

    assert module._bridge is not None
    assert module._bridge.started is True


def test_restart_bridge_stops_the_old_instance_before_starting_a_new_one(monkeypatch):
    # The GUI "Restart Bridge" command was bound to start_bridge(), which
    # early-returns whenever `_bridge` is already set -- so restart was a
    # no-op (#585). restart_bridge() must stop-then-start.
    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module, "ui", object())

    events: list[tuple[str, int]] = []

    class _FakeBridge:
        def start(self):
            events.append(("start", id(self)))

        def stop(self):
            events.append(("stop", id(self)))

    monkeypatch.setattr(module, "BinaryNinjaBridge", _FakeBridge)
    old = _FakeBridge()
    monkeypatch.setattr(module, "_bridge", old)

    module.restart_bridge()

    assert events[0] == ("stop", id(old))
    assert events[1][0] == "start"
    assert events[1][1] != id(old)
    assert module._bridge is not None
    assert module._bridge is not old


def test_restart_bridge_leaves_global_none_when_the_new_start_fails(monkeypatch):
    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module, "ui", object())

    class _StopOnlyBridge:
        def stop(self):
            pass

    monkeypatch.setattr(module, "_bridge", _StopOnlyBridge())

    class _BoomBridge:
        def start(self):
            raise RuntimeError("bind failed")

    monkeypatch.setattr(module, "BinaryNinjaBridge", _BoomBridge)

    module.restart_bridge()  # must not raise -- logs and returns

    assert module._bridge is None


# --------------------------------------------------------------------------
# SO_PEERCRED unsigned uid/gid unpack (#660)
# --------------------------------------------------------------------------


def test_check_peer_credentials_accepts_uid_above_signed_int32_range(monkeypatch):
    # A NIS/container-mapped uid > 2**31 must not render negative and be
    # rejected as a spurious mismatch (#660).
    import struct

    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module.os, "getuid", lambda: 3_000_000_000)

    class _FakeConn:
        def __init__(self, uid):
            self._packed = struct.pack("iII", 4242, uid, 0)

        def getsockopt(self, level, optname, buflen):
            return self._packed[:buflen]

    err = module._check_peer_credentials(_FakeConn(3_000_000_000))
    assert err is None


# --------------------------------------------------------------------------
# TargetManager.forget() -- dirty-id cleanup on close (#659)
# --------------------------------------------------------------------------


def test_target_manager_forget_discards_dirty_id(monkeypatch):
    from _bridge_fakes import _FakeFileBV, _register_views

    module = _load_bridge(monkeypatch)
    manager = module.TargetManager()
    bv = _FakeFileBV("/corpus/target.bndb")
    _register_views(module, bv)
    manager.refresh()  # assigns bv its stable view_id

    manager.mark_dirty(bv)
    assert manager.is_dirty(bv) is True

    manager.forget(bv)

    assert manager.is_dirty(bv) is False
    module._headless_views.clear()


def test_close_binary_forgets_dirty_view_on_close(monkeypatch):
    # The real close path (_close_binary) must discard the closed view's
    # dirty entry, not just the standalone TargetManager.forget() helper
    # (#659).
    from _bridge_fakes import _FakeFileBV, _register_views

    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge()
    bv = _FakeFileBV("/corpus/target.bndb")
    bv.file.close = lambda: None
    _register_views(module, bv)
    instance.targets.refresh()
    instance.targets.mark_dirty(bv)
    assert instance.targets.is_dirty(bv) is True

    instance._close_binary(path="/corpus/target.bndb")

    assert instance.targets.is_dirty(bv) is False
    module._headless_views.clear()
