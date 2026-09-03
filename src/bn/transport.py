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
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import (
    bridge_registry_path, bridge_socket_path, ensure_private_dir, instances_dir,
    project_root,
)
from .proc_identity import PinUnavailable, identity_verdict, pin_process


class BridgeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str | None = None,
        requested: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.requested = requested
        self.observed = observed


TRANSIENT_SOCKET_ERRNOS = {
    errno.ECONNREFUSED,
    errno.ENOENT,
}

# Normal CLI requests retain the public 600-second ceiling. bn-kernel applies
# its tighter 120-second default explicitly at the Session/Client boundary.
# BN_REQUEST_TIMEOUT overrides either path; 0/none/off/empty disables it.
DEFAULT_REQUEST_TIMEOUT = 600.0
# Full analysis operations keep a larger default; BN_REQUEST_TIMEOUT still wins.
REFRESH_REQUEST_TIMEOUT = 3600.0
SPAWN_LOCK_TIMEOUT = 30.0
DEFAULT_SPAWN_TIMEOUT = 60.0
CANCEL_REQUEST_TIMEOUT = 0.25


def _resolve_timeout(
    timeout: float | None,
    *,
    default: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> float | None:
    raw = os.environ.get("BN_REQUEST_TIMEOUT")
    if raw is None:
        return timeout if timeout is not None else default
    text = raw.strip().lower()

    def _reject() -> BridgeError:
        return BridgeError(
            f"BN_REQUEST_TIMEOUT={raw!r} is not a valid timeout: expected a "
            "positive number of seconds, or one of 0/none/off/empty to disable it."
        )

    # Validate and apply the environment override even when a caller supplied a
    # timeout. This keeps CLI and native/kernel backends consistent and prevents
    # a malformed global setting from being hidden by an explicit default.
    if text in ("", "none", "off"):
        return None
    try:
        value = float(text)
    except ValueError:
        raise _reject() from None
    if not math.isfinite(value):
        raise _reject()
    if value < 0 or math.copysign(1.0, value) < 0:
        raise _reject()
    if value == 0.0:
        if any(digit in text for digit in "123456789"):
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
    instance_token: str | None = None
    # True when the registry resolved but its socket file is gone: the bridge
    # process may still be alive, yet nothing can be dispatched to it. Normal
    # discovery hides these; only the lifecycle (admin) lookup returns them (#694).
    unreachable: bool = False


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
    if instance_id in (".", "..") or not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise BridgeError(
            f"Invalid instance id: {instance_id!r}. Use only letters, digits, "
            "'_', '-', and '.' (no path separators, no '.'/'..', no absolute paths)."
        )
    # An id whose socket path cannot fit in sockaddr_un.sun_path must fail HERE,
    # in the CLI and before anything is spawned, rather than as a bare
    # `OSError: AF_UNIX path too long` from bind() inside the bridge -- by which
    # point the caller has already committed to the id and the real cause (a
    # byte count) is nowhere in the message.
    try:
        bridge_socket_path(instance_id)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
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


def _process_state(pid: int) -> str | None:
    """Return Linux /proc state, or None when unavailable."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 1 :].strip().split()
    return fields[0] if fields else None


def bridge_process_alive(instance: BridgeInstance) -> bool:
    """Whether the bridge process *instance* registered is still running.

    A recycled pid is NOT the bridge: the check runs against a pinned process
    where possible, so the answer cannot be about a pid that was reused between
    the liveness probe and the identity read (#694).
    """
    try:
        pin = pin_process(instance.pid)
    except PinUnavailable:
        # No pin: either the pid is already gone, or this platform has no pidfd.
        # Nothing is signalled from here, so a best-effort probe is acceptable --
        # it only affects how long teardown polls.
        if not _process_alive(instance.pid):
            return False
        return identity_verdict(instance.meta, instance.pid) != "mismatch"
    with pin:
        return pin.verdict(instance.meta) != "mismatch"


class BridgeProcessSignal:
    """A verified pin on a bridge process, held across a whole teardown.

    `bn session start` cleanup, `session stop` and `session restart` all fall back
    to SIGTERM, wait, then SIGKILL when the shutdown request fails. A pid read
    from a registry file is only safe to signal while it is provably still the
    bridge, so the pid is PINNED once (``os.pidfd_open``), its identity verified
    through that pin, and EVERY signal of the escalation sent through the same
    pin: the process cannot exit and have its pid recycled between the check and
    either kill, and both signals provably address one identical process (#694).

    Never raises. ``refusal`` is None only when the pin is verified; ``send()``
    returns None when the signal was delivered and a complete, user-facing
    sentence otherwise.
    """

    __slots__ = ("_instance", "_pin", "refusal")

    def __init__(self, instance: BridgeInstance) -> None:
        self._instance = instance
        self._pin = None
        selector = instance_selector(instance)
        try:
            pin = pin_process(instance.pid)
        except PinUnavailable as exc:
            self.refusal: str | None = (
                f"refusing to signal pid {instance.pid} for bridge instance "
                f"{selector!r}: {exc}"
            )
            return
        verdict = pin.verdict(instance.meta)
        if verdict == "proven":
            self._pin = pin
            self.refusal = None
            return
        pin.close()
        if verdict == "mismatch":
            self.refusal = (
                f"refusing to signal pid {instance.pid} for bridge instance "
                f"{selector!r}: the identity recorded at startup (boot id plus "
                "process start time) does not match the pinned process, so the "
                "bridge exited and its pid was reused"
            )
        else:
            self.refusal = (
                f"refusing to signal pid {instance.pid} for bridge instance "
                f"{selector!r}: it recorded no verifiable process identity (an "
                "older bridge wrote the registry, or this platform exposes no "
                "boot id / process start time), so it cannot be confirmed to "
                f"still be the bridge; confirm with `ps -p {instance.pid}` and "
                "stop it manually"
            )

    def __enter__(self) -> BridgeProcessSignal:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._pin is not None:
            self._pin.close()
            self._pin = None

    def send(self, sig: int) -> str | None:
        """Deliver *sig* through the verified pin, or return why it was not sent.

        Identity is verified once, at pin time: the pin itself guarantees every
        later signal reaches that same process, so a second signal never re-reads
        ``/proc`` (where a post-SIGTERM reap would look like "no identity").
        """
        if self.refusal is not None:
            return self.refusal
        if self._pin is None:
            return (
                f"refusing to signal pid {self._instance.pid}: the verified "
                "process pin was already released"
            )
        selector = instance_selector(self._instance)
        try:
            self._pin.send(sig)
        except ProcessLookupError:
            return f"pid {self._instance.pid} exited before the signal was delivered"
        except OSError as exc:
            return (
                f"failed to signal pid {self._instance.pid} for bridge instance "
                f"{selector!r}: {exc}"
            )
        return None


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


def _load_instance(
    path: Path,
    *,
    socket_timeout: float = 0.2,
    include_unreachable: bool = False,
) -> BridgeInstance | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        socket_path = Path(payload["socket_path"])
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    instance_id = payload.get("instance_id")
    if path.parent == instances_dir() and instance_id != path.stem:
        # The registry filename is the caller's explicit selector. Never trust a
        # payload that claims a different identity, and never unlink the socket
        # named by that foreign payload.
        with contextlib.suppress(OSError):
            path.unlink()
        return None

    process_state = _process_state(pid)
    owner_alive = _process_alive(pid) and process_state not in {"Z", "X", "x"}
    # Liveness alone cannot tell a running bridge from an unrelated process that
    # recycled its pid, so discovery consults the durable identity too (#694).
    verdict = identity_verdict(payload, pid)
    unreachable = False
    if not socket_path.exists():
        # start() binds the socket BEFORE writing the registry, so "registry with
        # no socket" is never a legitimate startup window: it is a bridge that
        # died hard, or a phantom kept listed only by whatever owns its pid now.
        # A dead or unproven owner is purged outright; a PROVEN live owner is an
        # unreachable bridge -- no request can reach it, so normal discovery and
        # `bn session list` must not advertise it, but its record is the only
        # handle `bn session stop` has on a process that is still holding memory,
        # so lifecycle lookups (include_unreachable=True) still resolve it. Once
        # that process is gone the next discovery purges the record (#694).
        if not owner_alive or verdict != "proven":
            _purge_stale_registry(path, socket_path)
            return None
        if not include_unreachable:
            return None
        unreachable = True
    elif not _socket_is_live(socket_path, timeout=socket_timeout):
        # A stopped or overloaded live bridge may not accept before the probe
        # timeout (its accept backlog can be full). Never convert temporary
        # unresponsiveness into destructive discovery cleanup: request dispatch
        # will report bridge_stopped or the real socket failure. A recorded
        # identity that MISMATCHES is not unresponsiveness -- it is proof the
        # bridge exited and its pid was reused -- so that entry is swept.
        if not owner_alive or verdict == "mismatch":
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
        instance_id=instance_id,
        instance_token=payload.get("instance_token"),
        unreachable=unreachable,
    )


def list_instances(
    *,
    timeout: float | None = None,
    include_unreachable: bool = False,
) -> list[BridgeInstance]:
    """Every resolvable bridge instance.

    ``include_unreachable`` adds registries whose socket file is gone but whose
    process is provably alive. Request routing must NEVER set it (nothing can be
    dispatched to such a bridge); the two callers that must are the lifecycle
    lookup behind `bn session stop`/`restart` and spawn collision detection,
    which has to see a hidden record before reusing its instance id (#694).
    """
    instances: list[BridgeInstance] = []
    deadline = time.monotonic() + timeout if timeout is not None else None

    def load(path: Path) -> BridgeInstance | None:
        if deadline is None:
            socket_timeout = 0.2
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeError(
                    "Timed out selecting a bridge instance while scanning registries"
                )
            socket_timeout = min(0.2, remaining)
        return _load_instance(
            path,
            socket_timeout=socket_timeout,
            include_unreachable=include_unreachable,
        )

    # Legacy fixed registry (GUI mode or old headless)
    fixed_registry = bridge_registry_path()
    if fixed_registry.exists():
        instance = load(fixed_registry)
        if instance is not None:
            instances.append(instance)

    # Per-instance registries
    inst_dir = instances_dir()
    if inst_dir.is_dir():
        for reg_file in sorted(inst_dir.glob("*.json")):
            instance = load(reg_file)
            if instance is not None:
                instances.append(instance)

    return instances


def find_lifecycle_instance(
    selector: str,
    *,
    timeout: float | None = None,
) -> BridgeInstance | None:
    """Resolve *selector* for a lifecycle command, unreachable bridges included.

    `bn session stop` / `session restart` must be able to name a bridge whose
    socket is gone -- that unreachable process is exactly the one a user needs to
    kill -- while normal discovery keeps hiding it (#694).
    """
    for inst in list_instances(timeout=timeout, include_unreachable=True):
        if inst.instance_id == selector or instance_selector(inst) == selector:
            return inst
    return None


def gc_instances() -> dict[str, Any]:
    """Reap dead instances' leftovers from ``instances_dir()``.

    ``list_instances()`` purges a dead registry + its orphaned socket lazily,
    but deliberately keeps the ``.log`` breadcrumb (the empty-response
    diagnostic). A host that spawns many short-lived bridges therefore
    accumulates hundreds of zero-byte logs for instances that are long gone
    (#80). This sweeps the logs -- and any registry-less orphan sockets -- of
    every instance that no longer has a live registry, leaving live instances
    and the shared spawn lock untouched.

    Returns a summary: ``live_instances``, ``registries_purged`` (dead
    registries the liveness sweep removed), ``logs_removed``, ``sockets_removed``,
    and ``removed`` (the list of removed paths).
    """
    inst_dir = instances_dir()
    summary: dict[str, Any] = {
        "live_instances": 0,
        "registries_purged": 0,
        "logs_removed": 0,
        "sockets_removed": 0,
        "removed": [],
    }
    # Serialize against spawns. A spawn creates ``<id>.log`` + ``<id>.sock``
    # BEFORE it writes ``<id>.json`` (the registry), so without the spawn lock gc
    # could see an in-flight spawn's live socket/log as a registry-less orphan
    # and unlink it mid-spawn -- a live file deleted (#80 review). The lock window
    # is exactly the spawn-and-register interval, so holding it makes the
    # glob/iterdir snapshot unable to straddle a registration. ``_spawn_lock()``
    # also mkdir's ``instances_dir()``, so it always exists inside this block.
    with _spawn_lock():
        registries_before = set(inst_dir.glob("*.json"))
        # Triggers the lazy liveness sweep: dead registries + their sockets are
        # unlinked as a side effect, leaving only live registries behind.
        summary["live_instances"] = len(list_instances())
        registries_after = set(inst_dir.glob("*.json"))
        summary["registries_purged"] = len(registries_before - registries_after)
        live_stems = {p.stem for p in registries_after}
        for entry in sorted(inst_dir.iterdir()):
            # Never touch the shared spawn lock or any surviving (live) registry.
            if entry.name == ".spawn.lock" or entry.suffix == ".json":
                continue
            # A .log/.sock whose registry is gone belongs to a dead/long-gone
            # instance -- the registry was purged (now or earlier) or never
            # existed (and, under the lock, is not a spawn in flight).
            if entry.suffix in (".log", ".sock") and entry.stem not in live_stems:
                with contextlib.suppress(OSError):
                    entry.unlink()
                    summary["removed"].append(str(entry))
                    if entry.suffix == ".log":
                        summary["logs_removed"] += 1
                    else:
                        summary["sockets_removed"] += 1
    return summary


def _resolve_from_project_roots(
    instances: list[BridgeInstance],
) -> BridgeInstance | None:
    """Resolve a unique live bridge associated with the caller's project.

    Associations live in each already-validated instance registry. The registry
    remains authoritative: absent or malformed metadata contributes no match,
    and multiple matches fail closed instead of choosing by ordering or age.
    """
    try:
        current_root = str(project_root())
    except OSError:
        return None

    matches: list[BridgeInstance] = []
    for instance in instances:
        roots = instance.meta.get("project_roots") if isinstance(instance.meta, dict) else None
        if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
            continue
        if current_root in roots:
            matches.append(instance)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise BridgeError(
            f"Multiple Binary Ninja bridge instances are associated with "
            f"{current_root}; pass -i/--instance <id> or set BN_INSTANCE.\n"
            f"Instances:\n{_format_instance_choices(matches)}"
        )
    return None


def _multiple_instances_error(instances: list[BridgeInstance]) -> BridgeError:
    return BridgeError(
        "Multiple Binary Ninja bridge instances are running; pass -i/--instance <id> "
        "or set BN_INSTANCE (single-agent only).\n"
        f"Instances:\n{_format_instance_choices(instances)}"
    )


@contextlib.contextmanager
def _spawn_lock(timeout: float | None = SPAWN_LOCK_TIMEOUT):
    """Exclusive, bounded flock serializing all bridge spawns on this host."""
    inst_dir = ensure_private_dir(instances_dir())
    lock_path = inst_dir / ".spawn.lock"
    with open(lock_path, "w") as lock_file:
        if timeout is None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        else:
            if timeout < 0:
                raise ValueError("spawn lock timeout must be non-negative")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise BridgeError(
                            f"Timed out waiting for the bridge spawn lock after "
                            f"{timeout:g}s; another bridge is still starting. "
                            "Retry with an explicit existing -i/--instance."
                        ) from None
                    time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _remaining_deadline(deadline: float | None, context: str) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BridgeError(f"Timed out {context}")
    return remaining


def _resolve_spawn_timeout() -> float:
    """Resolve BN_SPAWN_TIMEOUT into a positive number of seconds.

    Starting a bridge is its OWN budget, deliberately separate from the request
    budget. Handing instance selection the resolved request timeout let a child
    that never registers hold an ordinary request for 600s -- or a load/refresh
    for 3600s -- and made BN_SPAWN_TIMEOUT=1 do nothing at all (#694).
    """
    raw_timeout = os.environ.get("BN_SPAWN_TIMEOUT")
    if raw_timeout is None:
        return DEFAULT_SPAWN_TIMEOUT
    try:
        timeout = float(raw_timeout)
    except ValueError:
        raise BridgeError(
            f"BN_SPAWN_TIMEOUT={raw_timeout!r} is not a valid positive "
            "number of seconds"
        ) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise BridgeError(
            f"BN_SPAWN_TIMEOUT={raw_timeout!r} is not a valid positive "
            "number of seconds"
        )
    return timeout


def _spawn_timeout_within(deadline: float | None, context: str) -> float:
    """The spawn budget, capped by whatever is left of the request deadline.

    Both bounds are real: a spawn must not outlive the caller's end-to-end
    request budget, and it must not consume that whole budget waiting for a
    registration that is not coming.
    """
    budget = _resolve_spawn_timeout()
    remaining = _remaining_deadline(deadline, context)
    return budget if remaining is None else min(budget, remaining)


def _auto_spawn_locked(timeout: float | None = SPAWN_LOCK_TIMEOUT) -> BridgeInstance:
    """Serialize auto-spawn and keep lock, discovery, and registration bounded."""
    deadline = time.monotonic() + timeout if timeout is not None else None
    remaining = _remaining_deadline(deadline, "waiting to auto-start a bridge")
    lock_timeout = (
        min(SPAWN_LOCK_TIMEOUT, remaining)
        if remaining is not None
        else SPAWN_LOCK_TIMEOUT
    )
    with _spawn_lock(timeout=lock_timeout):
        remaining = _remaining_deadline(deadline, "auto-starting a bridge")
        instances = list_instances(timeout=remaining)
        if len(instances) == 1:
            return instances[0]
        if instances:
            associated = _resolve_from_project_roots(instances)
            if associated is not None:
                return associated
            raise _multiple_instances_error(instances)
        remaining = _remaining_deadline(deadline, "auto-starting a bridge")
        return _spawn_instance_unlocked(
            timeout=(
                remaining if remaining is not None else DEFAULT_SPAWN_TIMEOUT
            )
        )


def choose_instance(
    instance_id: str | None = None,
    *,
    auto_start: bool = True,
    spawn_missing_named: bool = False,
    timeout: float | None = None,
) -> BridgeInstance:
    if instance_id is not None:
        validate_instance_id(instance_id)
    deadline = time.monotonic() + timeout if timeout is not None else None
    instances = list_instances(
        timeout=_remaining_deadline(deadline, "selecting a bridge instance")
    )
    if instance_id is not None:
        for inst in instances:
            if inst.instance_id == instance_id or instance_selector(inst) == instance_id:
                return inst
        if spawn_missing_named:
            # A spawn is bounded by BN_SPAWN_TIMEOUT capped by the request
            # deadline -- never by the request budget alone (#694).
            return spawn_instance(
                instance_id,
                timeout=_spawn_timeout_within(
                    deadline, "starting the requested bridge instance"
                ),
            )
        raise BridgeError(
            f"No bridge instance found with id: {instance_id}. "
            f"Start one with: bn session start /path/to/binary --instance-id {instance_id}"
        )
    if len(instances) == 1:
        return instances[0]
    if instances:
        associated = _resolve_from_project_roots(instances)
        if associated is not None:
            return associated
        raise _multiple_instances_error(instances)
    if auto_start:
        return _auto_spawn_locked(
            timeout=_spawn_timeout_within(deadline, "auto-starting a bridge")
        )
    raise BridgeError("No running Binary Ninja bridge instances found")




def _instance_identity(instance: BridgeInstance) -> dict[str, Any]:
    token = instance.instance_token
    if not isinstance(token, str) or not token:
        raise BridgeError(
            f"Bridge instance {instance_selector(instance)!r} has no identity token; "
            "the bridge is stale -- restart it and retry"
        )
    return {
        "instance_id": instance.instance_id,
        "pid": instance.pid,
        "token": token,
    }


def _verify_socket_peer_pid(sock: socket.socket, instance: BridgeInstance) -> None:
    if not hasattr(socket, "SO_PEERCRED"):
        return
    getter = getattr(sock, "getsockopt", None)
    if not callable(getter):
        return
    try:
        raw = getter(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        peer_pid, _uid, _gid = struct.unpack("3i", raw)
    except (OSError, ValueError, struct.error):
        return
    if peer_pid != instance.pid:
        raise BridgeError(
            f"Bridge socket peer pid mismatch for instance {instance_selector(instance)!r}: "
            f"registry pid {instance.pid}, socket peer pid {peer_pid}; refusing to send the request"
        )


def _verify_response_identity(
    response: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual = response.get("bridge_identity")
    if actual != expected:
        raise BridgeError(
            "Binary Ninja bridge identity mismatch: "
            f"expected {expected!r}, received {actual!r}; "
            "refusing data from a different or stale bridge"
        )
def _send_cancel_request(instance: BridgeInstance, request_id: str) -> None:
    payload = {
        "id": str(uuid.uuid4()),
        "op": "cancel_request",
        "params": {"request_id": request_id},
        "_bridge_identity": _instance_identity(instance),
    }
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(CANCEL_REQUEST_TIMEOUT)
            sock.connect(str(instance.socket_path))
            _verify_socket_peer_pid(sock, instance)
            sock.sendall(encoded)
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_WR)
            with contextlib.suppress(OSError):
                while sock.recv(65536):
                    pass
    except (OSError, BridgeError):
        pass


def _send_request_to_instance(
    instance: BridgeInstance,
    op: str,
    *,
    params: dict[str, Any] | None = None,
    target: str | None = None,
    timeout: float | None = None,
    timeout_display: float | None = None,
    default_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    connect_retries: int = 4,
    resolved: bool = False,
) -> dict[str, Any]:
    process_state = _process_state(instance.pid)
    if process_state in {"T", "t"}:
        raise BridgeError(
            f"bridge_stopped: Binary Ninja bridge instance "
            f"{instance_selector(instance)!r} (pid {instance.pid}) is stopped; "
            f"resume it with `kill -CONT {instance.pid}` or restart the instance"
        )
    if process_state in {"Z", "X", "x"}:
        raise BridgeError(
            f"bridge_not_running: Binary Ninja bridge instance "
            f"{instance_selector(instance)!r} (pid {instance.pid}) is in "
            f"process state {process_state}; restart the instance"
        )
    expected_identity = _instance_identity(instance)
    payload: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "op": op,
        "params": params or {},
        "_bridge_identity": expected_identity,
    }
    if target is not None:
        payload["target"] = target

    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    # BN_REQUEST_TIMEOUT is one end-to-end budget, applied exactly once. A caller
    # that already resolved it (send_request, or a paginating Client.collect) hands
    # down the *remaining* slice; re-resolving here would restore the full env
    # value and let a multi-page collection run for a multiple of its own budget.
    if not resolved:
        timeout = _resolve_timeout(timeout, default=default_timeout)
    deadline = time.monotonic() + timeout if timeout is not None else None

    chunks: list[bytes] = []
    last_error: OSError | None = None
    for attempt in range(connect_retries):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = TimeoutError("end-to-end request deadline expired")
                break
        else:
            remaining = None
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                if remaining is not None:
                    sock.settimeout(remaining)
                sock.connect(str(instance.socket_path))
                _verify_socket_peer_pid(sock, instance)
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
            if isinstance(exc, TimeoutError):
                _send_cancel_request(instance, str(payload["id"]))
            if exc.errno not in TRANSIENT_SOCKET_ERRNOS or attempt == connect_retries - 1:
                break
            delay = 0.05 * (attempt + 1)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_error = TimeoutError("end-to-end request deadline expired")
                    break
                if delay >= remaining:
                    time.sleep(remaining)
                    last_error = TimeoutError("end-to-end request deadline expired")
                    break
            time.sleep(delay)

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
            # `:g` keeps the real value for a sub-second timeout (0.01 -> "0.01s")
            # instead of rounding to "0.0s" (#370.3), while a whole-second value
            # still reads cleanly (30.0 -> "30s").
            shown_timeout = timeout if timeout_display is None else timeout_display
            timeout_suffix = (
                f" after {shown_timeout:g}s"
                if shown_timeout is not None
                else ""
            )
            raise BridgeError(
                f"Timed out waiting for Binary Ninja bridge pid {instance.pid} at {instance.socket_path}"
                f"{timeout_suffix} (op '{op}'). The bridge may be busy with analysis; "
                f"inspect progress with `bn -i {instance_selector(instance)} target info`, "
                "then raise or disable the limit with "
                "BN_REQUEST_TIMEOUT=<seconds|0> if the operation is intentionally long."
            ) from last_error
        raise BridgeError(
            f"Failed to contact Binary Ninja bridge pid {instance.pid} at {instance.socket_path}: {last_error}"
        ) from last_error

    if not chunks:
        raise _empty_response_error(instance, op)
    try:
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BridgeError(
            f"Binary Ninja bridge returned a non-UTF-8 response for op '{op}' "
            f"(instance {instance_selector(instance)}, pid {instance.pid})"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BridgeError(
            f"Binary Ninja bridge returned invalid JSON for op '{op}' "
            f"(instance {instance_selector(instance)}, pid {instance.pid})"
        ) from exc

    if not isinstance(response, dict):
        raise BridgeError(
            f"Binary Ninja bridge returned a malformed response for op '{op}' "
            f"(instance {instance_selector(instance)}, pid {instance.pid})"
        )

    _verify_response_identity(response, expected_identity)
    if response.get("ok"):
        if "result" not in response:
            raise BridgeError(
                f"Binary Ninja bridge replied ok without a result field for op '{op}' "
                f"(instance {instance_selector(instance)}, pid {instance.pid}); "
                "the bridge may be stale -- restart it"
            )
        return response

    error = response.get("error") or "Unknown Binary Ninja bridge error"
    raise BridgeError(
        str(error),
        status=response.get("status"),
        requested=response.get("requested"),
        observed=response.get("observed"),
    )


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


def _append_spawn_diagnostic(log_path: Path, message: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[bn-cli] {message}\n")
    except OSError:
        pass


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
    timeout: float | None = None,
    poll_interval: float = 0.2,
) -> BridgeInstance:
    """Spawn a bridge within one lock-and-registration deadline."""
    if timeout is None:
        timeout = _resolve_spawn_timeout()
    if instance_id is not None and instance_id != "default":
        validate_instance_id(instance_id)
    deadline = time.monotonic() + timeout
    with _spawn_lock(timeout=timeout):
        remaining = _remaining_deadline(deadline, "starting a bridge instance")
        return _spawn_instance_unlocked(
            instance_id,
            timeout=remaining if remaining is not None else timeout,
            poll_interval=poll_interval,
        )


def _spawn_instance_unlocked(
    instance_id: str | None = None,
    *,
    timeout: float = DEFAULT_SPAWN_TIMEOUT,
    poll_interval: float = 0.2,
) -> BridgeInstance:
    """Spawn-and-register core. MUST run under _spawn_lock()."""
    deadline = time.monotonic() + timeout
    # Collision detection MUST see unreachable records too (#694): a socket-less
    # registry is hidden from normal discovery, but its process can still be
    # alive -- spawning a second bridge under that same id would bind its socket
    # path, overwrite its registry, and orphan the live process with no record
    # left to stop it.
    existing = list_instances(
        timeout=_remaining_deadline(deadline, "checking existing bridge instances"),
        include_unreachable=True,
    )
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

    inst_dir = ensure_private_dir(instances_dir())

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
    while time.monotonic() < deadline:
        if reg_path.exists():
            remaining = _remaining_deadline(deadline, "waiting for bridge registration")
            # include_unreachable so a registry that exists under this id is SEEN
            # (and reported as an ownership collision below) rather than silently
            # skipped until the spawn deadline expires (#694).
            inst = _load_instance(
                reg_path,
                socket_timeout=min(0.2, remaining or 0.2),
                include_unreachable=True,
            )
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
            message = (
                f"Auto-started bn-agent (pid {proc.pid}, instance {instance_id}) "
                f"exited with code {exit_code} before registering."
            )
            _append_spawn_diagnostic(log_path, message)
            raise BridgeError(f"{message}{_log_tail(log_path)}")
        remaining = _remaining_deadline(deadline, "waiting for bridge registration")
        time.sleep(min(poll_interval, remaining or poll_interval))

    # The child is still running but never registered. Kill it so a slow
    # starter can't register later and show up as a surprise extra instance.
    message = (
        f"Auto-started bn-agent (pid {proc.pid}, instance {instance_id}) "
        f"did not register within {timeout:g}s and was terminated. "
        f"Check {log_path}. Retry the same command; on a heavily loaded host, "
        "set BN_SPAWN_TIMEOUT=<seconds> to allow more startup time."
    )
    _append_spawn_diagnostic(log_path, message)
    _reap_child(proc)
    raise BridgeError(message)


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
    duplicate (#92 Problem B). Convergence here means the bridge process is gone
    AND the registry no longer resolves (`_load_instance` returns None, which
    also sweeps a stale registry+socket left by a hard kill). A pid that outlives
    the bridge because an unrelated process reused it counts as gone: identity,
    not the bare number, decides (#694). Returns True on convergence.
    """
    deadline = time.monotonic() + timeout
    while True:
        gone = (
            not bridge_process_alive(instance)
            # include_unreachable: convergence means the registry FILE is gone,
            # not merely hidden from normal discovery (#694).
            and _load_instance(instance.registry_path, include_unreachable=True)
            is None
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
    default_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    connect_retries: int = 4,
    instance_id: str | None = None,
    spawn_missing_named: bool = False,
    resolved: bool = False,
) -> dict[str, Any]:
    # Validate/resolve the timeout BEFORE choosing an instance: choose_instance()
    # auto-spawns a headless bridge when none is running, so a bad
    # BN_REQUEST_TIMEOUT must fail here -- not after a stray random instance has
    # already been spawned into the cache (#255 review).
    # default_timeout lets a long one-time op (load/refresh) raise the no-env
    # default without overriding an explicit BN_REQUEST_TIMEOUT (#321).
    #
    # `resolved=True` means the caller already applied BN_REQUEST_TIMEOUT once and
    # `timeout` is the *remaining* slice of that single end-to-end budget (see
    # Client.collect). Re-resolving it -- here, in choose_instance's share, or in
    # _send_request_to_instance -- would hand every page a fresh copy of the full
    # env value, so a collection could run for a multiple of its declared budget
    # and the child bridge's cancellation would be scheduled off the wrong number.
    if not resolved:
        timeout = _resolve_timeout(timeout, default=default_timeout)
    requested_timeout = timeout
    deadline = time.monotonic() + timeout if timeout is not None else None
    instance = choose_instance(
        instance_id,
        spawn_missing_named=spawn_missing_named,
        timeout=timeout,
    )
    if deadline is not None:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise BridgeError(
                f"Timed out selecting a bridge instance for op {op!r}; "
                "the end-to-end request deadline expired before connecting"
            )
    return _send_request_to_instance(
        instance,
        op,
        params=params,
        target=target,
        timeout=timeout,
        timeout_display=requested_timeout,
        default_timeout=default_timeout,
        connect_retries=connect_retries,
        resolved=True,
    )
