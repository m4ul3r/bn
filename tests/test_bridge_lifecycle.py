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
