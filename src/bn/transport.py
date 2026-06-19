from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import bridge_registry_path, bridge_socket_path, instances_dir


class BridgeError(RuntimeError):
    pass


TRANSIENT_SOCKET_ERRNOS = {
    errno.ECONNREFUSED,
    errno.ENOENT,
}

# Hard ceiling on how long a request may sit with no bytes arriving before the
# CLI gives up, so a wedged bridge (e.g. a py_exec stuck under the write lock)
# can't hang every CLI invocation forever. Generous because legitimate ops
# (load/refresh with update_analysis_and_wait) can run for minutes. Override
# with BN_REQUEST_TIMEOUT=<seconds>; 0/none/off/empty disables the timeout entirely.
DEFAULT_REQUEST_TIMEOUT = 600.0


def _resolve_timeout(timeout: float | None) -> float | None:
    if timeout is not None:
        return timeout
    raw = os.environ.get("BN_REQUEST_TIMEOUT")
    if raw is None:
        return DEFAULT_REQUEST_TIMEOUT
    text = raw.strip().lower()

    def _reject() -> BridgeError:
        # Fail loud: a typo'd / out-of-range value silently falling back to the
        # 600s default (or silently disabling) left the user believing they'd set
        # a short timeout when they hadn't (#255).
        return BridgeError(
            f"BN_REQUEST_TIMEOUT={raw!r} is not a valid timeout: expected a "
            f"positive number of seconds, or one of 0/none/off/empty to disable it."
        )

    # Empty/whitespace-only and the explicit sentinels disable the timeout entirely.
    if text in ("", "none", "off"):
        return None
    try:
        value = float(text)
    except ValueError:
        raise _reject() from None
    if not math.isfinite(value):  # inf / nan parse as floats but aren't timeouts
        raise _reject()
    # Reject negatives -- including a value that underflowed to -0.0 (e.g.
    # -1e-325), where `value < 0` is False but the sign bit is still set.
    if value < 0 or math.copysign(1.0, value) < 0:
        raise _reject()
    if value == 0.0:
        # A literal 0 disables (a 0-second socket timeout would make every request
        # fail non-blocking), matching the documented sentinel. But a positive
        # magnitude that underflowed to 0.0 (e.g. 1e-325) is not a "disable"
        # request -- reject it so it can't silently turn the timeout off.
        if any(d in text for d in "123456789"):
            raise _reject()
        return None
    return value


@dataclass(slots=True)
class BridgeInstance:
    pid: int
    socket_path: Path
    registry_path: Path
    plugin_name: str
    plugin_version: str
    started_at: str | None
    meta: dict[str, Any]
    instance_id: str | None = None


def instance_selector(instance: BridgeInstance) -> str:
    return instance.instance_id or "default"


# A bridge instance id is joined directly into filesystem paths
# (instances_dir()/<id>.{json,sock,log}). Path semantics make an unvalidated id
# a traversal primitive: "../evil" escapes instances_dir() and "/abs" replaces
# it entirely, spawning a bridge whose files land outside the cache tree --
# never listed by list_instances() (which only globs instances_dir()/*.json) and
# impossible to stop normally (#84). Restrict ids to a strict basename grammar.
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_instance_id(instance_id: str) -> str:
    """Reject any instance id that isn't a safe path basename.

    Accepts letters, digits, '_', '-', '.'; rejects empty strings, '.'/'..',
    and anything containing a path separator or other character (which also
    rules out absolute paths and traversal). Raises BridgeError before any
    filesystem activity. Returns the id unchanged when valid.
    """
    if not isinstance(instance_id, str) or not instance_id:
        raise BridgeError("Instance id must be a non-empty string")
    if instance_id in (".", "..") or not _INSTANCE_ID_RE.match(instance_id):
        raise BridgeError(
            f"Invalid instance id: {instance_id!r}. Use only letters, digits, "
            "'_', '-', and '.' (no path separators, no '.'/'..', no absolute paths)."
        )
    return instance_id


def _format_instance_choices(instances: list[BridgeInstance]) -> str:
    lines = []
    for inst in instances:
        selector = instance_selector(inst)
        details = [f"pid={inst.pid}", f"socket={inst.socket_path}"]
        if inst.started_at:
            details.append(f"started={inst.started_at}")
        lines.append(f"- {selector} ({', '.join(details)})")
    return "\n".join(lines)


def _process_alive(pid: int) -> bool:
    """Best-effort check that ``pid`` still names a running process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _empty_response_error(instance: BridgeInstance, op: str | None) -> BridgeError:
    """Explain a connection that accepted the request but replied with nothing.

    This is the symptom of the bridge process dying mid-request -- the dispatch
    layer catches every Python exception, so an empty reply means the *process*
    went away (segfault or OOM during native analysis), not a handler error.
    Surface the pid, liveness, and log path instead of a bare one-liner.
    """
    op_label = f"op '{op}'" if op else "the request"
    log_path = instance.registry_path.with_suffix(".log")
    parts = [
        f"Binary Ninja bridge returned an empty response for {op_label} "
        f"(instance {instance_selector(instance)}, pid {instance.pid})."
    ]
    if _process_alive(instance.pid):
        parts.append(
            "The process is still running but closed the connection without "
            "replying -- a worker thread likely hit a native fault."
        )
    else:
        parts.append(
            "The process is no longer running -- it most likely crashed or was "
            "OOM-killed (large or complex binaries can exhaust memory during "
            "update_analysis_and_wait)."
        )
    if log_path.exists():
        parts.append(f"Check {log_path} for any crash output.")
    parts.append("Reload the target with `bn load`, or start a fresh bridge with `bn session start`.")
    return BridgeError(" ".join(parts))


def _purge_stale_registry(registry_path: Path, socket_path: Path | None = None) -> None:
    """Drop a registry whose owning process is gone, plus its orphaned socket.

    A SIGKILL or native crash leaves the unix socket file on disk (only a clean
    ``stop()`` unlinks it), so the registry is removed *and* the dead socket is
    swept here. The sibling ``.log`` is intentionally left behind: an instance
    is only purged after its socket goes dead -- frequently a crash -- and the
    log is the one breadcrumb worth keeping for the empty-response diagnostic.
    """
    with contextlib.suppress(OSError):
        registry_path.unlink()
    if socket_path is not None:
        with contextlib.suppress(OSError):
            socket_path.unlink()


def _socket_is_live(socket_path: Path, timeout: float = 0.2) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
        return True
    except OSError:
        return False


def _load_instance(path: Path) -> BridgeInstance | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        socket_path = Path(payload["socket_path"])
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if not socket_path.exists():
        _purge_stale_registry(path, socket_path)
        return None

    if not _socket_is_live(socket_path):
        _purge_stale_registry(path, socket_path)
        return None

    return BridgeInstance(
        pid=pid,
        socket_path=socket_path,
        registry_path=path,
        plugin_name=str(payload.get("plugin_name", "bn_agent_bridge")),
        plugin_version=str(payload.get("plugin_version", "0")),
        started_at=payload.get("started_at"),
        meta=payload,
        instance_id=payload.get("instance_id"),
    )


def list_instances() -> list[BridgeInstance]:
    instances: list[BridgeInstance] = []

    # Legacy fixed registry (GUI mode or old headless)
    fixed_registry = bridge_registry_path()
    if fixed_registry.exists():
        instance = _load_instance(fixed_registry)
        if instance is not None:
            instances.append(instance)

    # Per-instance registries
    inst_dir = instances_dir()
    if inst_dir.is_dir():
        for reg_file in sorted(inst_dir.glob("*.json")):
            instance = _load_instance(reg_file)
            if instance is not None:
                instances.append(instance)

    return instances


def _multiple_instances_error(instances: list[BridgeInstance]) -> BridgeError:
    return BridgeError(
        "Multiple Binary Ninja bridge instances are running; pass --instance <id> "
        "or set BN_INSTANCE.\n"
        f"Instances:\n{_format_instance_choices(instances)}"
    )


@contextlib.contextmanager
def _spawn_lock():
    """Exclusive flock serializing ALL spawns (auto and named) by this host.

    A single process-wide lock file is intentional: flock is per-open-file, so
    a named spawn and a concurrent auto-spawn must contend on the *same* file or
    they could both pass the duplicate check and fork two children (#92). Held
    only for the spawn-and-register window, never around a request.
    """
    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)
    lock_path = inst_dir / ".spawn.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _auto_spawn_locked() -> BridgeInstance:
    """Serialize auto-spawn across concurrent CLI processes.

    Without this lock, two CLIs racing on an empty registry each spawn a
    bridge, after which every bare command fails with "Multiple instances
    are running" until a human cleans up. The registry is re-checked under
    the lock because the previous holder may have spawned while we waited.
    """
    with _spawn_lock():
        instances = list_instances()
        if len(instances) == 1:
            return instances[0]
        if instances:
            raise _multiple_instances_error(instances)
        # Already under the spawn lock -- call the unlocked core directly, or a
        # second flock on the same file from this process would deadlock.
        return _spawn_instance_unlocked()


def choose_instance(
    instance_id: str | None = None,
    *,
    auto_start: bool = True,
    spawn_missing_named: bool = False,
) -> BridgeInstance:
    if instance_id is not None:
        validate_instance_id(instance_id)
    instances = list_instances()
    if instance_id is not None:
        for inst in instances:
            if inst.instance_id == instance_id or instance_selector(inst) == instance_id:
                return inst
        if spawn_missing_named:
            # `bn load --instance <new-id>` brings the named bridge up itself,
            # matching the "auto-spawn headless if needed" model. Only load opts
            # in; other commands keep failing fast so a typo'd id can't silently
            # spawn an empty process.
            return spawn_instance(instance_id)
        raise BridgeError(
            f"No bridge instance found with id: {instance_id}. "
            f"Start one with: bn session start --instance-id {instance_id}"
        )
    if len(instances) == 1:
        return instances[0]
    if instances:
        raise _multiple_instances_error(instances)
    if auto_start:
        return _auto_spawn_locked()
    raise BridgeError("No running Binary Ninja bridge instances found")


def _send_request_to_instance(
    instance: BridgeInstance,
    op: str,
    *,
    params: dict[str, Any] | None = None,
    target: str | None = None,
    timeout: float | None = None,
    connect_retries: int = 4,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "op": op,
        "params": params or {},
    }
    if target is not None:
        payload["target"] = target

    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    timeout = _resolve_timeout(timeout)

    chunks: list[bytes] = []
    last_error: OSError | None = None
    for attempt in range(connect_retries):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                if timeout is not None:
                    sock.settimeout(timeout)
                sock.connect(str(instance.socket_path))
                sock.sendall(encoded)
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_WR)
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            if exc.errno not in TRANSIENT_SOCKET_ERRNOS or attempt == connect_retries - 1:
                break
            time.sleep(0.05 * (attempt + 1))

    if last_error is not None and chunks:
        # Bytes arrived and then the connection failed: a timeout or reset
        # mid-response. Don't fall through to json.loads on the truncated
        # payload -- that misreports the failure as "invalid JSON".
        raise BridgeError(
            f"Connection to Binary Ninja bridge pid {instance.pid} failed mid-response for op '{op}' "
            f"after {len(b''.join(chunks))} bytes ({type(last_error).__name__}: {last_error})"
        ) from last_error
    if last_error is not None:
        if isinstance(last_error, TimeoutError):
            timeout_suffix = f" after {timeout:.1f}s" if timeout is not None else ""
            raise BridgeError(
                f"Timed out waiting for Binary Ninja bridge pid {instance.pid} at {instance.socket_path}"
                f"{timeout_suffix} (op '{op}'). The bridge may be busy with a long analysis; "
                "raise or disable the limit with BN_REQUEST_TIMEOUT=<seconds|0>."
            ) from last_error
        raise BridgeError(
            f"Failed to contact Binary Ninja bridge pid {instance.pid} at {instance.socket_path}: {last_error}"
        ) from last_error

    if not chunks:
        raise _empty_response_error(instance, op)

    try:
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BridgeError("Binary Ninja bridge returned invalid JSON") from exc

    if not isinstance(response, dict):
        raise BridgeError("Binary Ninja bridge returned a malformed response")

    if response.get("ok"):
        return response

    error = response.get("error") or "Unknown Binary Ninja bridge error"
    raise BridgeError(str(error))


def _find_bn_agent() -> list[str]:
    """Return the command to invoke bn-agent."""
    # Prefer the bn-agent script in the same directory as sys.executable
    exe_dir = Path(sys.executable).parent
    bn_agent = exe_dir / "bn-agent"
    if bn_agent.exists():
        return [str(bn_agent)]
    return [sys.executable, "-m", "bn.headless"]


def _log_tail(log_path: Path, lines: int = 20) -> str:
    """Return the last *lines* of the spawn log, formatted for an error message."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    tail = [line for line in text.splitlines() if line.strip()][-lines:]
    if not tail:
        return ""
    return f"\nLast output from {log_path}:\n" + "\n".join(f"  {line}" for line in tail)


def _reap_child(proc: subprocess.Popen) -> None:
    """Terminate a spawned child that won't be used, escalating to SIGKILL."""
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)


def spawn_instance(
    instance_id: str | None = None,
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
) -> BridgeInstance:
    """Spawn a new bn-agent headless process and wait for it to register.

    Serialized against all other spawns by a host-wide lock so two concurrent
    named spawns of the same id can't both fork a child (#92 Problem A).
    """
    if instance_id is not None and instance_id != "default":
        validate_instance_id(instance_id)
    with _spawn_lock():
        return _spawn_instance_unlocked(
            instance_id, timeout=timeout, poll_interval=poll_interval
        )


def _spawn_instance_unlocked(
    instance_id: str | None = None,
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
) -> BridgeInstance:
    """Spawn-and-register core. MUST run under _spawn_lock()."""
    existing = list_instances()
    if instance_id is None:
        existing_selectors = {instance_selector(inst) for inst in existing}
        while True:
            candidate = secrets.token_hex(4)
            if candidate not in existing_selectors:
                instance_id = candidate
                break
    elif instance_id == "default":
        raise BridgeError("Instance id 'default' is reserved for the fixed GUI bridge")
    elif any(inst.instance_id == instance_id or instance_selector(inst) == instance_id for inst in existing):
        raise BridgeError(f"Bridge instance already exists with id: {instance_id}")

    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    log_path = inst_dir / f"{instance_id}.log"
    log_file = open(log_path, "w")  # noqa: SIM115

    cmd = _find_bn_agent() + ["--instance-id", instance_id]
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    log_file.close()

    reg_path = bridge_registry_path(instance_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if reg_path.exists():
            inst = _load_instance(reg_path)
            if inst is not None:
                # Verify the registered process is the child WE spawned. A
                # different live pid means a stale registry slipped past the
                # liveness purge or another process raced us under this id;
                # reap our orphan rather than return someone else's bridge (#92).
                if inst.pid != proc.pid:
                    _reap_child(proc)
                    raise BridgeError(
                        f"Bridge instance id {instance_id!r} is already owned by "
                        f"another process (pid {inst.pid}); refusing to return a "
                        "bridge this call did not start."
                    )
                return inst
        exit_code = proc.poll()
        if exit_code is not None:
            raise BridgeError(
                f"Auto-started bn-agent (pid {proc.pid}, instance {instance_id}) "
                f"exited with code {exit_code} before registering."
                f"{_log_tail(log_path)}"
            )
        time.sleep(poll_interval)

    # The child is still running but never registered. Kill it so a slow
    # starter can't register later and show up as a surprise extra instance.
    _reap_child(proc)

    raise BridgeError(
        f"Auto-started bn-agent (pid {proc.pid}, instance {instance_id}) "
        f"did not register within {timeout:.0f}s and was terminated. Check {log_path}"
    )


def wait_for_teardown(
    instance: BridgeInstance,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Block until *instance* has fully torn down, or *timeout* elapses.

    `bn session stop` used to return as soon as the shutdown ACK (or a SIGTERM)
    was delivered, before the socket/registry were unlinked and the process
    exited -- so `stop X && start X` could race the dying instance and fail as a
    duplicate (#92 Problem B). Convergence here means the process is gone AND the
    registry no longer resolves (`_load_instance` returns None, which also sweeps
    a stale registry+socket left by a hard kill). Returns True on convergence.
    """
    deadline = time.monotonic() + timeout
    while True:
        gone = (
            not _process_alive(instance.pid)
            and _load_instance(instance.registry_path) is None
        )
        if gone:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval)


def send_request(
    op: str,
    *,
    params: dict[str, Any] | None = None,
    target: str | None = None,
    timeout: float | None = None,
    connect_retries: int = 4,
    instance_id: str | None = None,
    spawn_missing_named: bool = False,
) -> dict[str, Any]:
    instance = choose_instance(instance_id, spawn_missing_named=spawn_missing_named)
    return _send_request_to_instance(
        instance,
        op,
        params=params,
        target=target,
        timeout=timeout,
        connect_retries=connect_retries,
    )
