"""Bridge teardown lifecycle for private cache registries (#694).

Detached load workers can outlive ``stop()``. These tests pin the teardown
latch, load-job quiescence, durable process identity, and project-association
restoration so a late worker cannot re-publish a dead bridge registry.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import types
import time
from pathlib import Path

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
    # bind) must not unlink ANOTHER live instance's socket/registry/log just
    # because they happen to share the same on-disk paths (#585).
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    live = module.BinaryNinjaBridge(instance_id="live1")
    live._write_registry()
    live.socket_path.parent.mkdir(parents=True, exist_ok=True)
    live.socket_path.touch()
    live.registry_path.with_suffix(".log").touch()
    assert live.registry_path.exists()
    assert live.socket_path.exists()

    unbound = module.BinaryNinjaBridge(instance_id="live1")  # same paths, never start()ed
    assert unbound._server is None

    unbound.stop()

    assert live.registry_path.exists()
    assert live.socket_path.exists()
    assert live.registry_path.with_suffix(".log").exists()


def test_start_headless_clears_global_when_start_raises(monkeypatch, tmp_path):
    # start_headless must not leave an unbound instance assigned to the
    # module global when start() raises -- same #585 pattern the GUI
    # start_bridge() path was fixed for in this PR. An assigned-but-unbound
    # _bridge reaches atexit -> _stop_bridge() -> stop(), which (pre-fix)
    # unlinks another live instance's discovery files.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    monkeypatch.setattr(module, "_bridge", None)

    def fake_start(self):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(module.BinaryNinjaBridge, "start", fake_start)

    with pytest.raises(RuntimeError, match="bind failed"):
        module.start_headless(instance_id="testinst")

    assert module._bridge is None


def test_start_rolls_back_a_failure_after_the_bind(monkeypatch, tmp_path):
    """A start() that fails after binding must not leave a listener behind.

    start() started the serve_forever daemon and THEN called _write_registry()
    with no rollback. A failure there (a full cache filesystem is enough) raised
    out of start() and orphaned both the bound socket and its thread -- and
    nothing could reap them: start_headless publishes the module global only
    after start() returns, and _stop_bridge() early-returns on None, so atexit
    had no handle either (#800).

    Asserted on the resources the leak consists of -- the bound socket file and
    the serving thread -- never on `_server`/`_thread`, because whether stop()
    also clears those handles is #799's contract and this test has to hold with
    or without that fix. The thread is JOINED rather than sampled: shutdown()
    returns from inside serve_forever's own finally block, before the thread
    object flips to not-alive, so is_alive() read straight after start() raises
    is a coin flip -- and a thread that exits on its own is not the leak.

    The artifacts of an earlier incarnation that shared this instance id are
    asserted to SURVIVE. The rolled-back start never published them (registering
    is what failed), and the log is the file the spawning client is holding open
    to diagnose this very failure. Asserting they are absent would only be
    measuring the stub below."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="rollback1")
    inst.registry_path.parent.mkdir(parents=True, exist_ok=True)
    inst.registry_path.write_text('{"pid": 1}')
    log_path = inst.registry_path.with_suffix(".log")
    log_path.write_text("earlier incarnation output\n")

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    def recording_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        created.append(thread)
        return thread

    monkeypatch.setattr(module.threading, "Thread", recording_thread)
    monkeypatch.setattr(
        inst, "_write_registry",
        lambda: (_ for _ in ()).throw(OSError(28, "ENOSPC")),
    )

    with pytest.raises(OSError, match="ENOSPC"):
        inst.start()

    assert not inst.socket_path.exists(), "a bound socket file was left behind"
    assert created, "the serve thread was never started"
    created[0].join(timeout=5.0)
    assert not created[0].is_alive(), "the serve_forever daemon thread leaked"
    assert log_path.read_text() == "earlier incarnation output\n"
    assert inst.registry_path.read_text() == '{"pid": 1}'


def test_start_raises_instead_of_hanging_when_the_serve_thread_will_not_start(
    monkeypatch, tmp_path
):
    """A serve thread that never comes up must fail the start, not wedge it.

    stop() is not a safe rollback for a failure this early: BaseServer.shutdown()
    blocks on an event only serve_forever() ever sets, so a rollback routed
    through it hangs start() -- and the caller with it -- instead of raising.
    start() therefore closes the listener out itself when the thread never came
    up (#800)."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    real_thread = threading.Thread

    class RefusingThread:
        """Stands in for Thread.start() failing under resource exhaustion."""

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(module.threading, "Thread", RefusingThread)
    inst = module.BinaryNinjaBridge(instance_id="refuse1")

    raised: list[BaseException] = []

    def run_start():
        try:
            inst.start()
        except BaseException as exc:  # noqa: BLE001 - the failure is the point
            raised.append(exc)

    # A REAL thread drives start(), so a wedged start() fails this assertion
    # instead of hanging the suite.
    driver = real_thread(target=run_start, daemon=True)
    driver.start()
    driver.join(timeout=5.0)

    assert not driver.is_alive(), "start() never returned"
    assert [type(exc) for exc in raised] == [RuntimeError]
    assert not inst.socket_path.exists(), "a bound socket file was left behind"


def test_start_rolls_back_a_failure_inside_the_server_constructor(
    monkeypatch, tmp_path
):
    """bind() can succeed before the server's own construction fails.

    socketserver's constructor binds and then activates (listen); when activate
    raises it closes the socket but cannot remove the AF_UNIX file bind() put on
    disk, and the server handle is never published, so no later path can unlink
    it. start() has to (#800)."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    class RefusingListen(module.ThreadedUnixServer):
        def server_activate(self):
            raise OSError(105, "ENOBUFS")

    monkeypatch.setattr(module, "ThreadedUnixServer", RefusingListen)
    inst = module.BinaryNinjaBridge(instance_id="construct1")

    with pytest.raises(OSError, match="ENOBUFS"):
        inst.start()

    assert not inst.socket_path.exists(), "a bound socket file was left behind"


# --------------------------------------------------------------------------
# The #800 rollback under #799's ownership rule
# --------------------------------------------------------------------------


def test_start_rollback_unlinks_the_socket_before_it_releases_the_bind(
    monkeypatch, tmp_path
):
    """The rollback's unlink has to be the last thing that frees the path.

    Releasing the bind is what makes the socket path takeable: the instant
    server_close() gives it up, a successor can prove the leftover stale, bind
    and register. A rollback that closes first and unlinks after therefore walks
    the very race #799 closes inside stop() -- it deletes the successor's socket
    and leaves that bridge serving on an inode no client can name. The unlink
    happens while the listener still holds the bind (shutdown() stops the loop
    without releasing it), and nothing unlinks after it.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    successor: list[socket.socket] = []

    class ReleasingServer(module.ThreadedUnixServer):
        """Hands the path to a successor the instant our bind is released."""

        def server_close(self):
            super().server_close()
            path = self.server_address
            if os.path.exists(path):
                # A successor proves the leftover stale, then binds and serves.
                os.unlink(path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(path)
            sock.listen(1)
            successor.append(sock)

    monkeypatch.setattr(module, "ThreadedUnixServer", ReleasingServer)
    inst = module.BinaryNinjaBridge(instance_id="rollbackrace1")
    monkeypatch.setattr(
        inst, "_write_registry",
        lambda: (_ for _ in ()).throw(OSError(28, "ENOSPC")),
    )

    try:
        with pytest.raises(OSError, match="ENOSPC"):
            inst.start()

        assert successor, "the rollback never released the bind"
        assert inst.socket_path.exists(), (
            "the rollback unlinked the socket of a successor that bound the path "
            "as soon as the bind was given up"
        )
        assert _socket_answers(inst.socket_path), "the successor's endpoint is dead"
    finally:
        for sock in successor:
            sock.close()


def test_start_rollback_keeps_a_successor_that_bound_when_the_constructor_failed(
    monkeypatch, tmp_path
):
    """The constructor branch has to unlink while the bind is still held, too.

    `BaseServer.__init__` takes the bind and the listen itself and, when the
    listen raises, calls `server_close()` ITSELF -- so the bind is already given
    up inside the constructor, and the path is takeable while start()'s rollback
    is still on its way to the unlink. A rollback that unlinks then deletes the
    socket of whatever bound in between: the #799 harm, inside start(). Taking
    the two steps separately (`bind_and_activate=False`) keeps the bind on
    start()'s side of the failure, so the unlink still happens under it.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    successor: list[socket.socket] = []

    class SuccessorInTheCleanup(module.ThreadedUnixServer):
        """Fails the listen, and lets a successor take the path the instant the
        constructor's own cleanup gives the bind up."""

        def server_activate(self):
            raise OSError(105, "ENOBUFS")

        def server_close(self):
            super().server_close()
            path = self.server_address
            if os.path.exists(path):
                # A successor proves the leftover stale, then binds and serves.
                os.unlink(path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(path)
            sock.listen(1)
            successor.append(sock)

    monkeypatch.setattr(module, "ThreadedUnixServer", SuccessorInTheCleanup)
    inst = module.BinaryNinjaBridge(instance_id="constructrace1")

    try:
        with pytest.raises(OSError, match="ENOBUFS"):
            inst.start()

        assert successor, "the constructor never gave the bind up"
        assert inst.socket_path.exists(), (
            "the rollback unlinked the socket of a successor that bound the path "
            "when the constructor's cleanup gave the bind up"
        )
        assert _socket_answers(inst.socket_path), "the successor's endpoint is dead"
    finally:
        for sock in successor:
            sock.close()


def test_start_rollback_never_unlinks_a_path_it_never_bound(monkeypatch, tmp_path):
    """The other direction of `holds_bind`: no bind of ours, no unlink.

    The bind and the listen are two steps now, so one of the failures between
    them is "the bind never happened" -- a successor that took the free path in
    that window makes bind() lose the race with the path already ITS. The
    rollback must remove the path only when this start() put it there; an
    unconditional unlink would delete a live successor's socket, which is the
    #799 harm reached from the other side.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    successor: list[socket.socket] = []

    class BoundBySuccessor(module.ThreadedUnixServer):
        """bind() loses the race: a successor took the path first."""

        def server_bind(self):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(self.server_address)
            sock.listen(1)
            successor.append(sock)
            raise OSError(98, "EADDRINUSE")

    monkeypatch.setattr(module, "ThreadedUnixServer", BoundBySuccessor)
    inst = module.BinaryNinjaBridge(instance_id="bindrace1")

    try:
        with pytest.raises(OSError, match="EADDRINUSE"):
            inst.start()

        assert successor, "the successor never bound the path"
        assert inst.socket_path.exists(), (
            "the rollback unlinked a socket this start() never bound"
        )
        assert _socket_answers(inst.socket_path), "the successor's endpoint is dead"
    finally:
        for sock in successor:
            sock.close()


def test_stop_after_a_rolled_back_start_owns_nothing(monkeypatch, tmp_path):
    """A stop() that reaches an instance whose start() rolled back owns no bind.

    The rollback removed the socket file its own bind() made and clears
    `_server`, so the instance has no endpoint left to take away and nothing to
    shut down. #799's release path keys ownership off `_server`, so leaving the
    closed handle set would make that stop() unlink the registry and the log
    this start() never wrote -- they belong to whoever spawned the bridge -- and
    then wait forever in BaseServer.shutdown() on a serve loop that never ran.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    real_thread = threading.Thread

    class RefusingThread:
        """Stands in for Thread.start() failing under resource exhaustion."""

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(module.threading, "Thread", RefusingThread)
    inst = module.BinaryNinjaBridge(instance_id="rollback2")
    inst.registry_path.parent.mkdir(parents=True, exist_ok=True)
    inst.registry_path.write_text('{"pid": 1}', encoding="utf-8")
    log_path = inst.registry_path.with_suffix(".log")
    log_path.write_text("earlier incarnation output\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="can't start new thread"):
        inst.start()
    assert not inst.socket_path.exists(), "a bound socket file was left behind"

    # A REAL thread drives stop(), so a wedged stop() fails this assertion
    # instead of hanging the suite.
    stopper = real_thread(target=inst.stop, daemon=True)
    stopper.start()
    stopper.join(timeout=5.0)

    # The ownership rule first: these are the files a stop() that owes this
    # instance nothing must leave exactly as it found them.
    assert inst.registry_path.exists(), "stop() deleted a registry it never wrote"
    assert log_path.exists(), "stop() deleted a log it never wrote"
    assert inst.registry_path.read_text(encoding="utf-8") == '{"pid": 1}'
    assert log_path.read_text(encoding="utf-8") == "earlier incarnation output\n"
    assert not stopper.is_alive(), "stop() never returned"


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


# --------------------------------------------------------------------------
# stop() destroys only what it still owns (#799)
# --------------------------------------------------------------------------


def _socket_answers(path) -> bool:
    """Whether a client can actually reach `path` -- the observable a bridge's
    socket file exists to make true, and the one a successor loses when a stale
    stop unlinks it."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex(str(path)) == 0
    finally:
        probe.close()


def _live_bridge(module, instance_id):
    """A bridge with a REAL bind and a REAL serve thread.

    The defect is a teardown that unlinks files a *different listener* owns, so
    the sockets have to be real: a stubbed `_server` holds no bind and no
    backlog, and the proof under test is that a successor cannot reach the same
    paths while the one before it still holds them."""
    inst = module.BinaryNinjaBridge(instance_id=instance_id)
    inst.start()
    return inst


def test_stale_stop_keeps_a_successor_that_rebound_the_same_instance(
    monkeypatch, tmp_path
):
    """A stop() on an already-torn-down bridge must not delete the successor
    that rebound its instance id.

    `_server` was assigned in start() and never cleared, and the teardown arm
    was gated on `_server is not None` alone -- so a second stop() on the same
    object walked right over whatever had taken the paths since, deleting the
    successor's socket, registry and log and leaving it serving on an inode no
    client can name (#799)."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    first = _live_bridge(module, "rebind1")
    assert _socket_answers(first.socket_path)
    first.stop()
    assert first._server is None, "stop() must clear the handle it tore down"

    successor = _live_bridge(module, "rebind1")
    # The log is opened by whatever SPAWNS a bridge, not by the bridge, so it is
    # not this process's to account for by any identity -- only by ordering.
    log_path = successor.registry_path.with_suffix(".log")
    log_path.write_text("successor serving\n", encoding="utf-8")
    assert _socket_answers(successor.socket_path)

    first.stop()  # stale: a second stop() on the torn-down object

    assert _socket_answers(successor.socket_path), "successor's endpoint was unlinked"
    assert successor.socket_path.exists()
    assert json.loads(
        successor.registry_path.read_text(encoding="utf-8")
    )["instance_token"] == successor.instance_token
    assert log_path.read_text(encoding="utf-8") == "successor serving\n"

    successor.stop()


def test_successor_started_inside_stops_join_window_survives_the_resumed_stop(
    monkeypatch, tmp_path
):
    """The same takeover, one layer earlier: inside stop() itself.

    stop() may wait up to `load_join_timeout` for a load worker that is inside
    an uninterruptible update_analysis_and_wait(). The unlinks used to happen
    AFTER that join, so a successor that started in the window could bind and
    register and then be deleted by the teardown resuming behind it (#799).
    Releasing the bind is what makes the paths takeable, so the unlinks must
    happen before it -- while we still own them -- and never after."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    first = _live_bridge(module, "join1")

    release = threading.Event()

    def slow_worker():
        release.wait(10.0)

    worker = threading.Thread(target=slow_worker, daemon=True)
    worker.start()
    first._load_job_threads["probe"] = worker

    in_join = threading.Event()
    original = first._load_worker_threads

    def observing_join():
        threads = original()
        in_join.set()  # latched, discovery files gone, now waiting on the worker
        return threads

    first._load_worker_threads = observing_join
    stopper = threading.Thread(target=first.stop)
    stopper.start()

    successor = None
    try:
        assert in_join.wait(10.0), "stop() never reached the join"
        # The successor starts while the previous owner is still inside stop():
        # at this point the path is already free, so its bind succeeds.
        successor = _live_bridge(module, "join1")
        successor.registry_path.with_suffix(".log").write_text(
            "successor serving\n", encoding="utf-8"
        )
    finally:
        release.set()
        stopper.join(10.0)

    assert not stopper.is_alive()
    assert successor is not None
    assert _socket_answers(successor.socket_path), "the resuming stop unlinked it"
    assert json.loads(
        successor.registry_path.read_text(encoding="utf-8")
    )["instance_token"] == successor.instance_token
    assert successor.registry_path.with_suffix(".log").read_text(
        encoding="utf-8"
    ) == "successor serving\n"

    successor.stop()


def test_stop_removes_only_the_files_of_the_bind_it_is_ending(monkeypatch, tmp_path):
    """Both directions of the ownership rule, in one test because each direction
    is red on a different tree.

    The one-shot that keeps a stale stop() off a successor's files is a property
    of the CURRENT bind, not of the object's lifetime: a bridge that binds the
    path again owns it again, so the stop() that ends that later bind must still
    remove the files. A flag that latches for the object's lifetime leaves the
    socket behind as exactly the clutter a clean shutdown is supposed to drop.

    Direction 2 is red on the pre-fix base (#799); direction 1 is red on a
    one-shot that never re-arms.
    """
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)

    # Direction 1: the stop() that ends a bind this object made removes it.
    reowner = _live_bridge(module, "rearm1")
    reowner.stop()
    reowner.start()  # the same object binds the path again
    assert _socket_answers(reowner.socket_path), "the rebind is not serving"
    reowner.stop()
    assert not reowner.socket_path.exists(), (
        "the stop() that ended the rebind left that bind's own socket behind"
    )

    # Direction 2: a stale stop() on an object that never rebound leaves alone
    # whatever a successor owns.
    owner = _live_bridge(module, "rearm2")
    owner.stop()
    successor = _live_bridge(module, "rearm2")
    log_path = successor.registry_path.with_suffix(".log")
    log_path.write_text("successor serving\n", encoding="utf-8")

    owner.stop()  # stale: this object did not rebind

    assert _socket_answers(successor.socket_path), "successor's endpoint was unlinked"
    assert successor.socket_path.exists()
    assert json.loads(
        successor.registry_path.read_text(encoding="utf-8")
    )["instance_token"] == successor.instance_token
    assert log_path.read_text(encoding="utf-8") == "successor serving\n"

    successor.stop()


def _start_gate_bridge(monkeypatch, tmp_path, bound, *, listing=True):
    """A bridge whose socket path already holds a file, with the kernel's
    bound-socket evidence stubbed to *bound* and the availability of that
    listing stubbed to *listing* (False = a platform with no `/proc/net/unix`,
    where every path answers None)."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="startgate")
    inst.socket_path.parent.mkdir(parents=True, exist_ok=True)
    inst.socket_path.touch()
    monkeypatch.setattr(module, "path_has_bound_socket", lambda path: bound)
    monkeypatch.setattr(module, "bound_socket_listing_available", lambda: listing)
    return module, inst


def test_start_keeps_a_socket_file_whose_status_could_not_be_proved(monkeypatch, tmp_path):
    """`connect()` cannot answer "is anything bound here": a socket that is
    BOUND but has not reached `listen` refuses exactly like a crashed bridge's
    leftover file, and every bridge passes through that state coming up. Start
    used to unlink on that negative probe, orphaning the starting bridge on an
    unlinked inode -- destruction on absence of evidence. A path the listing
    cannot represent must keep the file and say which evidence refused."""
    module, inst = _start_gate_bridge(monkeypatch, tmp_path, None)

    with pytest.raises(RuntimeError, match="nothing could be proved"):
        inst.start()

    assert inst.socket_path.exists()


def test_start_keeps_a_socket_file_the_kernel_says_is_bound(monkeypatch, tmp_path):
    """Positive evidence of a binding is a refusal whether or not that binding
    is accepting yet -- a full accept backlog answers a connect() like a dead
    bridge too."""
    module, inst = _start_gate_bridge(monkeypatch, tmp_path, True)

    with pytest.raises(RuntimeError, match="Refusing to displace"):
        inst.start()

    assert inst.socket_path.exists()


def test_start_unlinks_a_socket_file_the_kernel_says_is_unbound(monkeypatch, tmp_path):
    """The other direction, so the refusals above are not vacuous: proof that
    NOTHING is bound is what clears a crashed bridge's leftover file. Asserted
    at the bind attempt, so the test never opens a real listener."""
    module, inst = _start_gate_bridge(monkeypatch, tmp_path, False)

    def refuse_bind(*args, **kwargs):
        raise RuntimeError("bind reached")

    monkeypatch.setattr(module, "ThreadedUnixServer", refuse_bind)

    with pytest.raises(RuntimeError, match="bind reached"):
        inst.start()

    assert not inst.socket_path.exists()


def test_start_refuses_a_reachable_socket_the_listing_does_not_name(
        monkeypatch, tmp_path):
    """A successful connect() outranks a NEGATIVE listing answer.

    `/proc/net/unix` records the name `bind` was GIVEN, so a listener renamed
    onto this endpoint is fully reachable here while appearing in the listing
    under its original basename -- where the lookup's basename filter skips it
    and the answer comes back False. Base refused this on the connect alone;
    the bound-socket evidence was added to stop a FAILED connect from proving
    absence, and it must not have cost the case where the connect SUCCEEDS.

    Real socket, real rename, real connect: stubbing the evidence here would
    assert the fix against the author's model of the kernel rather than the
    kernel."""
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    module = _load_bridge(monkeypatch)
    inst = module.BinaryNinjaBridge(instance_id="renamed")
    inst.socket_path.parent.mkdir(parents=True, exist_ok=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound_as = tmp_path / "originally-bound-here.sock"
    server.bind(str(bound_as))
    server.listen(1)
    try:
        os.rename(bound_as, inst.socket_path)
        # Reachable at its new name ...
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        probe.connect(str(inst.socket_path))
        probe.close()
        # ... and invisible to the listing under that name, which is the wrong
        # negative this refusal has to survive. Stated rather than asserted: a
        # future lookup that learned to see it would only make the refusal more
        # certain, and must not red this cell.

        with pytest.raises(RuntimeError, match="already serving") as raised:
            inst.start()
        assert "Refusing to displace" in str(raised.value), raised.value
        assert inst.socket_path.exists(), (
            "a reachable endpoint was unlinked, leaving its owner serving on an "
            "inode no client can name")
    finally:
        server.close()



def test_start_falls_back_to_the_connect_probe_where_no_listing_exists(
        monkeypatch, tmp_path):
    """The one documented exception to "an unprovable answer keeps the file".

    On a platform with no `/proc/net/unix` (Darwin/BSD, both supported) EVERY
    path answers None, so the strong rule would make this bridge permanently
    unstartable on its OWN fixed socket path after a single unclean shutdown --
    and nothing in the tool could recover it, because `instance gc` retains on
    an unprovable answer too. A sweep can skip a file forever at no cost; the
    process that must BIND that exact path cannot. So it falls back to the
    weaker connect() evidence base used everywhere.

    Scoped to the platform, not to the answer: a path the LISTING cannot
    represent still keeps the file (the test above), because there the listing
    exists and the rule holds for every other path on the host."""
    module, inst = _start_gate_bridge(monkeypatch, tmp_path, None, listing=False)
    monkeypatch.setattr(module.BinaryNinjaBridge, "_socket_is_live", lambda self: False)

    def refuse_bind(*args, **kwargs):
        raise RuntimeError("bind reached")

    monkeypatch.setattr(module, "ThreadedUnixServer", refuse_bind)

    with pytest.raises(RuntimeError, match="bind reached"):
        inst.start()

    assert not inst.socket_path.exists()


def test_start_without_a_listing_still_refuses_a_socket_that_answers(
        monkeypatch, tmp_path):
    """The fallback is weaker evidence, not no evidence: where the connect()
    probe DOES answer, a serving bridge keeps its endpoint."""
    module, inst = _start_gate_bridge(monkeypatch, tmp_path, None, listing=False)
    monkeypatch.setattr(module.BinaryNinjaBridge, "_socket_is_live", lambda self: True)

    with pytest.raises(RuntimeError, match="already serving"):
        inst.start()

    assert inst.socket_path.exists()



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


# --- #869: decide by file identity, not by path spelling --------------------
#
# Driven against the REAL filesystem rather than a mock, because the whole
# defect is that `==` and `.resolve()` answer differently from the kernel:
# a mock that returns whatever the test wants cannot show that.


def test_own_database_gate_recognises_a_hard_link_to_its_own_sibling_869(
        monkeypatch, tmp_path):
    # #869 item 2: the gate resolved symlinks but NOT hard links, so an
    # explicit `bn save` through a hard link of the target's own sibling
    # recorded no `database_path` -- and the next `session restart` reopened
    # the raw bytes even though that inode held the saved analysis. That is
    # the silent-data-loss shape #753 exists to stop, reached by a spelling.
    module = _load_bridge(monkeypatch)
    binary = tmp_path / "target"
    binary.write_bytes(b"\x7fELF")
    sibling = tmp_path / "target.bndb"
    sibling.write_bytes(b"BNDB")
    hard = tmp_path / "hardlink.bndb"
    os.link(sibling, hard)
    assert os.path.samefile(sibling, hard)          # premise, from the kernel

    assert module._is_own_database_destination(str(sibling), str(binary)) is True
    assert module._is_own_database_destination(str(hard), str(binary)) is True


def test_own_database_gate_still_refuses_a_genuine_export_869(
        monkeypatch, tmp_path):
    # Must-not-fire twin, and the reason the gate exists at all: recording an
    # export would move the target's restart identity onto a copy. A distinct
    # file with its own inode is NOT the target's database, however similarly
    # it is named.
    module = _load_bridge(monkeypatch)
    binary = tmp_path / "target"
    binary.write_bytes(b"\x7fELF")
    (tmp_path / "target.bndb").write_bytes(b"BNDB")
    export = tmp_path / "target.bndb.copy"
    export.write_bytes(b"BNDB")
    assert not os.path.samefile(tmp_path / "target.bndb", export)

    assert module._is_own_database_destination(str(export), str(binary)) is False


def test_own_database_gate_answers_for_a_destination_that_does_not_exist_869(
        monkeypatch, tmp_path):
    # The normal save case: the destination has no inode yet, so `samefile`
    # RAISES rather than answering False. The spelling fast path has to carry
    # it -- this is why the string compare is kept as something that can only
    # ADD an answer, never remove one.
    module = _load_bridge(monkeypatch)
    binary = tmp_path / "fresh"
    binary.write_bytes(b"\x7fELF")
    assert not (tmp_path / "fresh.bndb").exists()

    assert module._is_own_database_destination(
        str(tmp_path / "fresh.bndb"), str(binary)) is True


def test_same_file_degrades_to_false_without_identity_evidence_869(monkeypatch):
    # Degrade-safely, the rule `socket_evidence` applies: two paths that
    # neither match as strings nor exist to be stat'd yield no evidence of
    # sameness, so the answer is False rather than a guess.
    module = _load_bridge(monkeypatch)
    assert module._same_file("/nope/a.bndb", "/nope/b.bndb") is False
    assert module._same_file("", "/nope/b.bndb") is False
    assert module._same_file(None, None) is False


# --- #867: refuse a save whose destination is another OPEN target ----------


def _collision_bridge(monkeypatch, tmp_path, collision):
    """A bridge whose single view saves for real, with the collision probe
    answering *collision*. Uses the suite's own `_SaveBV` and `targets.resolve`
    seam, i.e. the wiring every other save test uses."""
    from _bridge_fakes import _SaveBV
    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge()
    bv = _SaveBV(str(tmp_path / "target"), result=True, write=True)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "open_target_for_path",
                        lambda path, *, exclude: collision)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda _bv: None)
    monkeypatch.setattr(instance.targets, "note_database", lambda _bv, _p: None)
    return module, instance, bv


def test_save_refuses_a_destination_another_target_has_open_867(monkeypatch, tmp_path):
    # #867: the save used to LAND and then disclose. Both rows then name one
    # database, and `session restart` returns one target for both -- measured
    # 2 -> 1 at rc 0. Disclosure after an irreversible write is the weaker
    # half; refusing costs nothing the caller cannot recover.
    other = {"target_id": "t:2", "selector": "other.bndb",
             "filename": str(tmp_path / "other")}
    module, instance, bv = _collision_bridge(monkeypatch, tmp_path, other)
    dest = tmp_path / "target.bndb"

    with pytest.raises(module.OperationFailure) as excinfo:
        instance._save_database(None, str(dest))

    assert excinfo.value.status == "invalid_request"
    assert "already open as target" in excinfo.value.message
    # It must name BOTH ways out, or the refusal is a wall.
    assert "--path" in excinfo.value.message
    assert "close the other target" in excinfo.value.message
    # And nothing may have been written -- that is the whole point of moving
    # the check ahead of the write.
    assert not dest.exists()
    assert bv.created_with is None


def test_save_without_a_collision_still_writes_867(monkeypatch, tmp_path):
    # Must-not-fire twin: the ordinary save is the common path and must be
    # untouched by the new refusal.
    module, instance, bv = _collision_bridge(monkeypatch, tmp_path, None)
    dest = tmp_path / "target.bndb"

    result = instance._save_database(None, str(dest))

    assert result["saved"] is True
    assert dest.exists()


def test_save_proceeds_when_the_collision_probe_cannot_answer_867(
        monkeypatch, tmp_path):
    # Degrade-safely, the same direction the post-write disclosure already
    # takes: a probe that RAISES has no evidence of a collision, and must not
    # block a legitimate save. Only a positive match refuses.
    module, instance, bv = _collision_bridge(monkeypatch, tmp_path, None)

    def _boom(path, *, exclude):
        raise RuntimeError("target map unavailable")

    monkeypatch.setattr(instance.targets, "open_target_for_path", _boom)
    dest = tmp_path / "target.bndb"

    result = instance._save_database(None, str(dest))

    assert result["saved"] is True
    assert dest.exists()


# --- #867/#857: the post-write disclosure is the BACKSTOP, and it must fire --
#
# Round 1 blocker: when the two #857 disclosure tests were rewritten into #867
# refusal tests, the disclosure lost ALL positive coverage -- stubbing
# `_disclose_open_target_collision` to a bare `return` left the save-related
# files green (579 in this file alone), because the only surviving reference
# was the must-not-fire negative. The two destinations the PRE-WRITE refusal
# provably cannot cover are the ones asserted below.


def test_the_read_only_cache_fallback_destination_is_disclosed_867(
        monkeypatch, tmp_path):
    """The refusal tests the REQUESTED path; the fallback writes somewhere else.

    A binary on a read-only mount cannot grow an adjacent `.bndb`, so the save
    lands in the writable cache (#214/#318) -- a destination chosen only AFTER
    the primary write failed, and therefore never seen by the pre-write check.
    If that cache copy is itself open as a target (an agent that loaded it to
    resume earlier work), the two targets are one database from this moment and
    `session restart` returns one row for both.

    Extending the refusal here was considered and rejected in
    `_disclose_open_target_collision`'s own docstring: the fallback exists so
    annotations are NOT lost on a read-only mount. So disclosure is the only
    protection this destination has, and this is the test that says so.

    Drives the REAL `open_target_for_path` through `_collect_open_views_state`
    rather than a lambda, so the probe's own identity matching is exercised
    against a genuine second view instead of being assumed.
    """
    from _bridge_fakes import _SaveBV
    raw = tmp_path / "ro" / "svc"
    raw.parent.mkdir()
    raw.write_bytes(b"\x7fELF")
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / "cache"))
    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge()
    cache_dest = str(module._cache_bndb_path(str(raw)))

    class _ROSaveBV(_SaveBV):
        def create_database(self, out: str):
            self.created_with = out
            if out == str(raw) + ".bndb":
                return False          # BN's "wrote nothing" on the RO mount
            Path(out).write_text("bndb")
            return True

    bv = _ROSaveBV(str(raw), result=True, write=True)
    other = _SaveBV(cache_dest, result=True, write=True)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(
        module, "_collect_open_views_state", lambda strict=False: ([bv, other], True))
    instance.targets.refresh()

    result = instance._save_database(None, None)

    # The premise: the primary really did fail and the cache really was used,
    # so the destination under test is one the pre-write check never saw.
    assert result["fallback"] is True, result
    assert result["path"] == cache_dest
    assert not (tmp_path / "ro" / "svc.bndb").exists()

    collision = result.get("collides_with_open_target")
    assert collision, (
        "the cache fallback landed on a file another target has open and said "
        "nothing -- the one destination the #867 refusal cannot cover")
    assert collision["filename"] == cache_dest
    assert "also open as target" in result["note"]
    assert "session restart" in result["note"]


def test_a_collision_opened_after_the_pre_write_check_is_disclosed_867(
        monkeypatch, tmp_path):
    """The other uncovered destination: the check-to-write WINDOW.

    `_save_database` probes, then writes. A target opened on that file in
    between passes the check and still collapses the two targets, because the
    collapse is a property of the file after the write, not of the check. The
    refusal cannot close this -- it is a race, not a missing test -- so the
    post-write disclosure is what makes it visible, and nothing asserted that.

    The probe answers None once (the pre-write check) and then names the other
    target, which is exactly what a concurrent `bn load` looks like from here.
    """
    module, instance, bv = _collision_bridge(monkeypatch, tmp_path, None)
    other = {"target_id": "t:2", "selector": "late.bndb",
             "filename": str(tmp_path / "target.bndb")}
    answers = [None, other]

    def _probe(path, *, exclude):
        return answers.pop(0) if answers else other

    monkeypatch.setattr(instance.targets, "open_target_for_path", _probe)
    dest = tmp_path / "target.bndb"

    result = instance._save_database(None, str(dest))

    # The premise: the pre-write check passed (nothing was refused) and the
    # write really landed, so only a post-write disclosure can carry the fact.
    assert result["saved"] is True
    assert dest.exists()
    assert answers == [], "both probe points must have been reached"

    assert result.get("collides_with_open_target") == other
    assert "also open as target" in result["note"]
    assert "'late.bndb'" in result["note"]


def test_the_degraded_rehomed_save_discloses_its_collision_too_867(
        monkeypatch, tmp_path):
    """The THIRD disclosure callsite, and round 2 left it uncovered.

    A save whose `create_database` re-homes the live view and whose restore
    then fails returns a degraded `rehomed` success. Its own comment says the
    disclosure must be unconditional there, because that branch wrote a real
    file and can land on another open target exactly like the clean one --
    and deleting the call left both bridge test files fully green (322
    passed), which is the same "declared load-bearing, covered by nothing"
    shape as the original blocker.

    The note assertion is the part that matters: the degraded branch already
    SET a note, so the disclosure has to append to it. A disclosure that
    overwrote it would tell the caller their database collides while hiding
    that their live target is still homed at the copy.
    """
    from _bridge_fakes import _RestoreFailSaveBV
    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge()
    binary = tmp_path / "svc"
    binary.write_bytes(b"\x7fELF")
    dest = tmp_path / "export.bndb"
    other = {"target_id": "t:3", "selector": "export.bndb",
             "filename": str(dest)}
    answers = [None, other]

    bv = _RestoreFailSaveBV(str(binary))
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "open_target_for_path",
                        lambda path, *, exclude: answers.pop(0) if answers else other)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda _bv: None)

    result = instance._save_database(None, str(dest))

    # The premise: this is really the DEGRADED branch, not the clean one.
    assert result["rehomed"] is True
    assert result["saved"] is True
    assert answers == [], "both probe points must have been reached"

    assert result.get("collides_with_open_target") == other
    assert "also open as target" in result["note"]
    assert "could not restore" in result["note"], (
        "the disclosure must APPEND to the degradation note, not replace it")


def test_save_through_a_hard_link_records_the_database_despite_inode_replacement_869(
        monkeypatch, tmp_path):
    """#889c finding 1: the identity fix was evaluated at the wrong MOMENT.

    `create_database` REPLACES the destination's inode, so a gate that runs
    after the write compares a brand-new file against the sibling and answers
    False -- `database_path` goes unrecorded and the next `session restart`
    reopens the STALE database. That is the #753 silent-drop shape surviving
    the #869 identity fix, because the identity was gone by the time it was
    asked about.

    The double REPLACES the inode the way BN does, rather than truncating in
    place: a `write_bytes` double preserves the link and would make this test
    pass against the broken code. The producer's behaviour is the test.
    """
    from _bridge_fakes import _SaveBV
    module = _load_bridge(monkeypatch)
    instance = module.BinaryNinjaBridge()

    binary = tmp_path / "svc"
    binary.write_bytes(b"\x7fELF")
    sibling = tmp_path / "svc.bndb"
    sibling.write_bytes(b"OLD")
    hard = tmp_path / "hard.bndb"
    os.link(sibling, hard)
    assert os.path.samefile(sibling, hard)
    before = os.stat(hard).st_ino

    from pathlib import Path as _P

    class _InodeReplacingBV(_SaveBV):
        def create_database(self, out: str):
            self.created_with = out
            # unlink-then-create: a NEW inode, exactly what was measured live
            _P(out).unlink(missing_ok=True)
            _P(out).write_bytes(b"NEW")
            return True

    bv = _InodeReplacingBV(str(binary), result=True, write=True)
    noted = {}
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "open_target_for_path",
                        lambda path, *, exclude: None)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda _bv: None)
    monkeypatch.setattr(instance.targets, "note_database",
                        lambda _bv, path: noted.__setitem__("path", path))

    result = instance._save_database(None, str(hard))

    assert result["saved"] is True
    # The premise of the finding: the write really did replace the inode, so a
    # post-write identity check could not have recognised this destination.
    assert os.stat(hard).st_ino != before
    assert not os.path.samefile(hard, sibling)
    # And the database is recorded anyway, because the decision was taken
    # while the identity still existed.
    assert noted.get("path") == str(hard)
