from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import math
import os
import secrets
import socket
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .paths import (
    bridge_registry_path, bridge_socket_path, cache_home, ensure_private_dir,
    instances_dir, project_root, validate_instance_id as _paths_validate_instance_id,
)
from .proc_identity import PinUnavailable, identity_verdict, pin_process
from .socket_evidence import path_has_bound_socket


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


def validate_instance_id(instance_id: str) -> str:
    """CLI-facing wrapper over the canonical instance-id grammar in ``.paths``.

    The grammar and its message live in ``paths.validate_instance_id``, the one
    chokepoint both the CLI and the bridge route through (#608); this function
    only translates its ``ValueError`` into the ``BridgeError`` the CLI and
    ``cli.py``'s re-export expect. Raises before any filesystem activity.
    """
    try:
        validated = _paths_validate_instance_id(instance_id)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    # An id whose socket path cannot fit in sockaddr_un.sun_path must fail HERE,
    # in the CLI and before anything is spawned, rather than as a bare
    # `OSError: AF_UNIX path too long` from bind() inside the bridge -- by which
    # point the caller has already committed to the id and the real cause (a
    # byte count) is nowhere in the message.
    try:
        bridge_socket_path(validated)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    return validated


def _format_instance_choices(instances: list[BridgeInstance]) -> str:
    lines = []
    for inst in instances:
        selector = instance_selector(inst)
        details = [f"pid={inst.pid}", f"socket={inst.socket_path}"]
        if inst.started_at:
            details.append(f"started={inst.started_at}")
        lines.append(f"- {selector} ({', '.join(details)})")
    return "\n".join(lines)


# The widest value the kernel's pid_t (a C `int`) can carry. Beyond it
# ``os.kill`` raises OverflowError -- not an OSError -- before any syscall
# happens, so `_process_alive` cannot absorb it. Registry content is checked
# against this before it is ever probed; see `_load_instance`.
_PID_MAX = 2**31 - 1


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
    """Return Linux /proc state, or None when unavailable.

    Parsed on BYTES. ``comm`` is embedded in this file exactly as the kernel
    holds it, so a neighbouring process whose executable basename carries a
    non-UTF-8 byte made ``read_text(encoding="utf-8")`` raise -- and with only
    ``OSError`` caught, a ``UnicodeDecodeError`` escaped into
    ``list_instances()`` and ``gc_instances()``. One unrelated process on the
    host then took down every discovery-backed command. The state character
    sits after the closing parenthesis of ``comm``, so that field is skipped
    without ever being decoded, and anything this parser cannot read answers
    ``None`` -- unknowable, which no arm treats as evidence (#618).
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return None
    close = stat.rfind(b")")
    if close < 0:
        return None
    fields = stat[close + 1 :].strip().split()
    if not fields:
        return None
    try:
        return fields[0].decode("ascii")
    except UnicodeDecodeError:
        return None


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


def _unlink_if_unchanged(path: Path, expected: bytes | None) -> bool:
    """Unlink *path* only while it still holds the document *expected*.

    Every destructive decision in ``_load_instance`` is taken on evidence read
    BEFORE the unlink: the registry's bytes, a pid's liveness, an errno from a
    ``connect()`` that may take the whole probe timeout. ``unlink`` acts on a
    NAME, and discovery holds no lock -- ``gc_instances`` takes ``_spawn_lock()``
    for this exact hazard, while a discovery-backed command cannot without
    blocking every spawn that calls it. So a legitimate re-spawn under that id
    replaces the registry inside that window (the bridge writes it through
    ``os.replace``), and the unlink then destroys a LIVE bridge's only handle on
    evidence about a document that is already gone (#618).

    The record is therefore re-read as late as possible and destroyed only if
    it is still the document that was judged. Comparing the BYTES rather than
    ``stat`` metadata is deliberate: inode numbers are recycled immediately on
    the filesystems the cache lives on, and a tmpfs timestamp is coarse enough
    that a replacement within the same tick can present the same
    ``(st_dev, st_ino, st_mtime_ns, st_size)`` as the file it replaced -- a
    false match measured while developing this check. A caller with no document
    to compare judged no file, and nothing is removed.
    """
    if expected is None:
        return False
    try:
        with path.open("rb") as stream:
            if stream.read() != expected:
                return False
        path.unlink()
    except (OSError, ValueError):
        # Not every path here comes from the directory scan -- the legacy fixed
        # registry is constructed by ``bridge_registry_path()`` and reaches this
        # unlink too -- and ValueError belongs here regardless: a path the
        # syscall layer cannot name (an embedded NUL, an unpaired HIGH
        # surrogate) must not take a discovery-backed command down, and
        # refusing to destroy is the safe answer for it.
        return False
    return True


def _purge_stale_registry(
    registry_path: Path,
    socket_path: Path | None = None,
    *,
    expected_record: bytes | None = None,
    socket_timeout: float = 0.2,
) -> None:
    """Drop a registry whose owning process is gone, plus its orphaned socket.

    A SIGKILL or native crash leaves the unix socket file on disk (only a clean
    ``stop()`` unlinks it), so the registry is removed *and* the dead socket is
    swept here. The sibling ``.log`` is intentionally left behind: an instance
    is only purged after its socket goes dead -- frequently a crash -- and the
    log is the one breadcrumb worth keeping for the empty-response diagnostic.

    Both halves destroy only what they can still see for themselves, because
    the caller's evidence is older than this call: the registry must still hold
    ``expected_record`` (see ``_unlink_if_unchanged``), and the socket must
    still be one nothing is bound to when asked HERE -- the last moment before
    the unlink, rather than upstream where a re-spawn could bind over it
    afterwards. A socket nothing proves unbound, and a record no caller
    identified, are kept.
    """
    _unlink_if_unchanged(registry_path, expected_record)
    if socket_path is not None and _socket_probe(socket_path, timeout=socket_timeout).nothing_bound:
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


class _SocketFacts(NamedTuple):
    """What a probe of a socket path actually established, kept apart.

    One probe was answering two questions for two different destructions, and
    the record paid for the socket's stricter evidence: on a platform with no
    ``/proc/net/unix`` a crashed bridge's record was retained forever because
    its SOCKET could not be proved unbound. These are facts about different
    objects and they need different strength, so they are returned separately
    (#618).

    ``nothing_accepting``
        Nobody can be served at this name, and that is proof rather than an
        absence: a bound and serving bridge answers ``EAGAIN`` once its accept
        backlog fills, and a timeout says nothing at all, while a refused
        connection or a missing path says nobody is accepting. This is what
        makes a RECORD litter -- with its owner gone too, it can never serve
        again, whatever is or is not at that path.
    ``nothing_bound``
        No socket holds the name at all, so the FILE is not an endpoint and may
        be unlinked. ``ECONNREFUSED`` does not establish that: a socket bound
        but not yet past ``listen`` refuses exactly like a crashed bridge's
        leftover file, and every bridge passes through that state coming up. So
        it takes the kernel's own list of bound paths, or a name that does not
        exist -- a bound AF_UNIX socket always has a directory entry, which is
        why that half needs no ``/proc``.
    """

    nothing_accepting: bool
    nothing_bound: bool


def _socket_probe(socket_path: Path, timeout: float = 0.2) -> _SocketFacts:
    """Probe *socket_path* once and report both facts a failure establishes.

    ``_socket_is_live`` answers "can this be talked to right now", which is the
    right question for routing and the wrong one for deleting. Measured on this
    kernel: a listening socket with a full backlog gives ``EAGAIN``, a missing
    path gives ``ENOENT``, and ``ECONNREFUSED`` covers THREE states -- a
    crashed bridge's leftover file, a plain file that never was a socket, and a
    socket bound but not yet listening. The first two are litter; the third is
    a bridge coming up, which is why that errno can condemn the record but
    never the socket.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
    except FileNotFoundError:
        return _SocketFacts(nothing_accepting=True, nothing_bound=True)
    except ConnectionRefusedError:
        return _SocketFacts(
            nothing_accepting=True,
            nothing_bound=path_has_bound_socket(socket_path) is False,
        )
    except OSError:
        return _SocketFacts(nothing_accepting=False, nothing_bound=False)
    return _SocketFacts(nothing_accepting=False, nothing_bound=False)


def _socket_path_is_confined(socket_path: Path) -> bool:
    """Whether a registry's ``socket_path`` lives under this user's bn cache.

    A registry is data, not a path we constructed: a corrupted or hand-edited
    entry can name any file on the host, and the loader would otherwise connect
    to it -- and, when it probes dead, unlink it through the stale sweep (#618).
    ``bridge_socket_path`` bounds what we WRITE; this bounds what we INGEST.

    Two different paths have to be inside the cache, because connect() and
    unlink() do not act on the same one. ``connect`` follows the final symlink,
    so the fully resolved target is what we would talk to. ``unlink`` never
    follows it: it removes the directory ENTRY at ``<resolved parent>/<name>``.
    Checking only the resolved target would accept an out-of-cache symlink
    whose target happens to be in-cache, and then delete that out-of-cache link.
    Any resolution failure is treated as unconfined.

    The boundary is the cache AS THE USER LAID IT OUT: both of the directories
    this code itself writes into, each measured after resolution. ``resolve()``
    follows symlinks, so measuring against ``cache_home()`` alone puts every
    socket under a symlinked ``instances/`` -- a tmpfs, a bigger disk, a
    per-project directory -- outside the cache, and the arm that refuses such a
    record used to delete it: a listening bridge lost its registry, and with it
    the only handle ``session stop`` has. An actor who can retarget those
    directories can already plant registries in them, so admitting them grants
    nothing, while refusing them costs a live bridge.
    """
    try:
        roots = (cache_home().resolve(), instances_dir().resolve())
        target = socket_path.resolve()
        entry = socket_path.parent.resolve() / socket_path.name
        return all(any(candidate.is_relative_to(root) for root in roots)
                   for candidate in (target, entry))
    except (OSError, TypeError, ValueError):
        # OSError: the path cannot be stat'd. ValueError: it cannot even be
        # interpreted as a path (an embedded NUL). TypeError: an empty final
        # component. All mean "not proven confined", and a corrupt payload must
        # not abort discovery.
        return False


def _registry_fields_are_well_formed(
    raw_socket_path: object, raw_pid: object, raw_instance_id: object
) -> bool:
    """Every payload field the adopt-vs-drop decision reads, checked in one place.

    A registry is DATA. Successive reviews found the same defect shape over and
    over -- a check that passed for the WRONG reason, so a record naming no
    live bridge was adopted as one -- because each field was guarded by a list
    of known-bad values instead of by its declared type and domain. This is the
    single admission point for those fields, and it is deliberately
    type-STRICT, mirroring ``proc_identity.recorded_start_ticks``: the only
    writer emits ``os.getpid()``, ``str(bridge_socket_path(...))`` and a
    grammar-validated id, so anything else is corruption and is DROPPED rather
    than coerced into something plausible.

    - ``socket_path`` must be a path string. An absolute one names the same
      file for everybody and is taken verbatim. A relative one names nothing
      on its own: it needs a base, and the caller's CWD is not the writer's --
      that is how a bogus record came to be believed or disbelieved depending
      on where the CLI happened to run, and, once absoluteness was demanded
      instead, how ONE cache root spelled two ways (relative when the bridge
      wrote its registry, absolute when a later CLI read it) silently dropped
      a running bridge. So a relative value decides nothing at all:
      ``_load_instance`` substitutes the socket the record is entitled to by
      construction. Judging the value further could only ever cost a live
      bridge whose own socket is listening, which is the same over-rejection
      again. Confinement still bounds the result.
    - ``pid`` must be a real ``int`` in the range ``os.kill`` accepts. ``int()``
      is lossy in exactly the direction that hurts: ``True``, ``"1"``, ``" 1 "``
      and ``1.9`` all become 1, and pid 1 always exists and answers EPERM. Zero
      and negatives address a process GROUP, so ``os.kill(0, 0)`` succeeds
      against our own group; anything wider than a C int makes ``os.kill``
      raise ``OverflowError``, which is not an ``OSError``.
    - ``instance_id`` must be absent or a valid id. The filename check below
      only covers registries under ``instances_dir()``; the legacy fixed pair
      lives in the cache root and had nothing to check its id against, so any
      value reached ``instance_selector`` and came back out as a selector.

    The identity fields (``boot_id``, ``pid_start_ticks``) are validated by
    ``proc_identity``, which already refuses a wrong type there.
    """
    if not isinstance(raw_socket_path, str):
        return False
    if isinstance(raw_pid, bool) or not isinstance(raw_pid, int):
        return False
    if not 0 < raw_pid <= _PID_MAX:
        return False
    if raw_instance_id is None:
        return True
    try:
        _paths_validate_instance_id(raw_instance_id)
    except ValueError:
        return False
    return True


def _registry_own_id(registry_path: Path) -> str:
    """The instance id a registry FILENAME names, derived one way for everyone.

    ``Path.stem`` reads a leading dot run as part of the name, so ``....json``
    -- the registry of the legal id ``...`` -- stems to the whole name. Read
    that way the record looks foreign and gets deleted while its bridge is
    still listening, and its leftovers reverse-map to no id and can never be
    reaped. Two readers need this id -- the loader and the orphan sweep -- and
    a second spelling of the derivation is a second answer waiting to disagree
    with the first.
    """
    return registry_path.name.removesuffix(".json")


def _record_socket_path(registry_path: Path, raw_socket_path: str) -> Path:
    """The socket a record can be reached through, derived one way for everyone.

    A relative ``socket_path`` decides nothing about where the socket is, so it
    names the socket this record owns by construction, in the directory
    discovery actually found the record in: `<name>.json` -> `<name>.sock`
    under ``instances_dir()``, and the legacy fixed pair's `<plugin>.json` ->
    `<plugin>.sock` in the cache root. Both are exactly what
    ``bridge_socket_path`` emits for that record, and unlike the CWD the reader
    and the writer cannot spell it differently.
    """
    socket_path = Path(raw_socket_path)
    if socket_path.is_absolute():
        return socket_path
    return registry_path.with_name(f"{_registry_own_id(registry_path)}.sock")


def _load_instance(
    path: Path,
    *,
    socket_timeout: float = 0.2,
    include_unreachable: bool = False,
) -> BridgeInstance | None:
    try:
        # The exact bytes THIS decision is taken on: every destructive arm
        # below re-reads the record immediately before unlinking and destroys
        # it only if it still holds this document, so a record replaced in the
        # window is never deleted on its predecessor's evidence (#618).
        record_document = path.read_bytes()
        payload = json.loads(record_document.decode("utf-8"))
        raw_socket_path = payload["socket_path"]
        raw_pid = payload["pid"]
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        # TypeError belongs here with the rest: subscripting a JSON document
        # that is not an object is the same class of corruption as unparseable
        # JSON or a missing key, and discovery skips a corrupt record rather
        # than taking every discovery-backed command down with a raw traceback.
        return None

    instance_id = payload.get("instance_id")
    if not _registry_fields_are_well_formed(raw_socket_path, raw_pid, instance_id):
        return None
    own_id = _registry_own_id(path)
    socket_path = _record_socket_path(path, raw_socket_path)
    pid = raw_pid

    process_state = _process_state(pid)
    owner_alive = _process_alive(pid) and process_state not in {"Z", "X", "x"}
    # Liveness alone cannot tell a running bridge from an unrelated process that
    # recycled its pid, so discovery consults the durable identity too (#694).
    verdict = identity_verdict(payload, pid)
    # One rule for every arm below: REFUSING a record costs a lookup, DELETING
    # it can cost a live bridge its only handle, so nothing here unlinks without
    # positive evidence that the record is litter -- the owner is gone, or its
    # recorded identity MISMATCHES, which is proof the bridge exited and its pid
    # was reused. "unrecorded" is the ABSENCE of evidence: it is what a pre-#694
    # bridge reports, and what every bridge reports on a platform with no
    # ``/proc`` (Darwin, which ``cache_home`` branches for), including one that
    # is serving right now. Four separate members of this defect class were
    # deletions taken on that absence (#618).
    record_is_litter = not owner_alive or verdict == "mismatch"

    if path.parent == instances_dir() and instance_id != own_id:
        # The registry filename is the caller's explicit selector. Never trust a
        # payload that claims a different identity, and never unlink the socket
        # named by that foreign payload. The DISAGREEMENT, though, can be an
        # artifact of our own reading rather than of the record: on a
        # case-insensitive filesystem `Foo.json` and `foo.json` are one file, so
        # a caller's spelling alone made a live bridge's own record read as
        # foreign -- and this arm then deleted it. Refuse always; delete only on
        # the evidence above.
        if record_is_litter:
            _purge_stale_registry(path, expected_record=record_document)
        return None

    unreachable = False
    if not _socket_path_is_confined(socket_path):
        # The payload points at a socket this code cannot place inside the
        # cache: never connect to it, and never let the stale sweep unlink it.
        # Whether to delete the RECORD is a separate question, and this arm
        # cannot tell "the payload is bogus" from "our reading of the layout is
        # wrong" -- the symlinked-cache purge proved the second happens -- so it
        # destroys only on the shared evidence above.
        #
        # A PROVEN live owner also RESOLVES for the lifecycle lookup, exactly as
        # the missing-socket arm does (#694): keeping the file without returning
        # it made it a handle for nothing -- `session stop` could not name the
        # process and spawn collision detection could not see the id, so a
        # re-spawn truncated that live bridge's log where refusing the id
        # outright had left it intact. A handle is not a connection:
        # `unreachable` is what stops dispatch (`_send_request_to_instance`
        # refuses it), and normal discovery keeps hiding it. An alive owner that
        # proves nothing gets neither the handle nor the purge: refused, and
        # left alone.
        if record_is_litter:
            _purge_stale_registry(path, expected_record=record_document)
            return None
        if not include_unreachable or verdict != "proven":
            return None
        unreachable = True
    elif not socket_path.exists():
        # start() binds the socket BEFORE writing the registry, so "registry with
        # no socket" is never a legitimate startup window: it is a bridge that
        # died hard, or a phantom kept listed only by whatever owns its pid now.
        # ROUTING and DESTROYING part company here, and keeping them together
        # cost a defect in each direction. `Path.exists()` reads an unreadable
        # directory exactly like a missing name, which is the right answer for
        # routing -- a socket this process cannot even stat is one it certainly
        # cannot serve a request through, and hiding it is what base did -- and
        # the wrong answer for deleting, because the arm below sweeps without
        # probing the socket at all. Routing keeps the broad reading; the purge
        # takes the narrow one. Making BOTH narrow adopted a record nothing can
        # reach as a healthy instance, which broke `choose_instance` on a host
        # whose other bridge was fine.
        # That absence is evidence about SERVICE, not about the handle: a bridge
        # nothing can reach is exactly the process `bn session stop` must still
        # be able to name, so normal discovery and `bn session list` hide it
        # while lifecycle lookups (include_unreachable=True) resolve it, and the
        # next discovery after that process is gone purges the record (#694).
        # Deleting it demands the same positive evidence as everywhere else:
        # purging on "unrecorded" took the only handle to a LIVE process away
        # from every pre-#694 bridge, and -- since `identity_verdict` needs
        # `/proc` for both halves of its proof -- from every bridge on a platform
        # that has none.
        if record_is_litter and _path_is_absent(socket_path):
            # No socket to sweep: this arm was reached BECAUSE nothing was at
            # that path, and `_path_is_absent` is what ESTABLISHES that rather
            # than inferring it from a probe that could not be taken. Anything
            # bound to it now arrived after the check, and unlinking it would
            # destroy an endpoint never judged here.
            _purge_stale_registry(path, expected_record=record_document)
            return None
        if not include_unreachable or verdict != "proven":
            return None
        unreachable = True
    elif not _socket_is_live(socket_path, timeout=socket_timeout):
        # A stopped or overloaded live bridge may not accept before the probe
        # timeout (its accept backlog can be full). Never convert temporary
        # unresponsiveness into destructive discovery cleanup: request dispatch
        # will report bridge_stopped or the real socket failure. A recorded
        # identity that MISMATCHES is not unresponsiveness -- it is proof the
        # bridge exited and its pid was reused -- so that entry is refused.
        #
        # This is the only arm that can unlink a socket that something may
        # still be BOUND to, and the probe that brought us here is the weakest
        # evidence in the file: a full accept backlog fails it on a bridge that
        # is serving. So the record is refused on the shared evidence, and
        # anything destroyed needs a probe that PROVES something -- both halves
        # of `record_is_litter` are inferences that can be wrong at once (a busy
        # socket plus a pid this process cannot address), and that combination
        # unlinked a live, listening socket (#618).
        if record_is_litter:
            # Two probes, two different questions. This one decides whether
            # anything here is litter at all, and it asks for the fact that fits
            # each object: nobody is ACCEPTING at this name, which -- with the
            # owner gone -- means the RECORD can never serve again whatever is at
            # that path, while the SOCKET additionally needs nothing to be bound
            # to it (decided inside the sweep). An inconclusive answer, the
            # serving bridge whose accept backlog is full, still leaves both
            # alone. Gating the record on the socket's stricter fact retained a
            # crashed bridge's record forever wherever the kernel cannot be
            # asked. The sweep re-probes immediately before it unlinks, because
            # this answer is already stale by then -- a re-spawn can bind over
            # the name inside the window, and an answer about the file it
            # replaced is not evidence about the one now bound (#618).
            if _socket_probe(socket_path, timeout=socket_timeout).nothing_accepting:
                _purge_stale_registry(
                    path,
                    socket_path,
                    expected_record=record_document,
                    socket_timeout=socket_timeout,
                )
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


def _path_is_absent(path: Path) -> bool:
    """Whether *path* PROVABLY names nothing, rather than merely not answering.

    The loader arm that sweeps a dead owner's registry reads "there is no
    socket file" as evidence, and it took that reading from ``Path.exists()``,
    which answers ``False`` for ``EACCES``, ``EIO``, ``ELOOP`` and ``ESTALE``
    exactly as it does for ``ENOENT`` -- so one unreadable directory turned a
    probe that could not be TAKEN into positive evidence, and a live owner's
    record and log went with it. The question is about the NAME this record
    carries, so what settles it is ``ENOENT``, ``ENOTDIR``, ``ENAMETOOLONG``
    -- nothing is reachable THROUGH that name, whatever the same inode may be
    reachable as under a shorter one -- and a name the syscall layer cannot
    express at all, which raises ``ValueError`` rather than ``OSError``: an
    embedded NUL, or a surrogate outside the ``surrogateescape`` range. (A
    ``\\udc80``-``\\udcff`` surrogate is NOT one of those: it is how a raw byte
    survives a decode, and ``os.fsencode`` turns it straight back into that
    byte, so such a path names a real file and is stat'ed normally.) Every
    other error is the READER's problem, not the path's, and answers
    ``False``, which costs a record that is kept (#618).
    """
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError as exc:
        return exc.errno == errno.ENAMETOOLONG
    except ValueError:
        return True
    return False


def gc_instances() -> dict[str, Any]:
    """Reap dead instances' leftovers from ``instances_dir()``.

    ``list_instances()`` purges a dead registry + its orphaned socket lazily,
    but deliberately keeps the ``.log`` breadcrumb (the empty-response
    diagnostic). A host that spawns many short-lived bridges therefore
    accumulates hundreds of zero-byte logs for instances that are long gone
    (#80). This sweeps those logs -- plus the ``.last_used`` sidecar written
    beside them and any registry-less orphan sockets -- for every instance
    that no longer has a live registry, leaving live instances and the shared
    spawn lock untouched.

    What it deliberately does NOT do is remove a registry discovery chose to
    keep. Every destructive arm in this module demands positive evidence that
    the thing it destroys is litter, and a record whose owner pid is alive but
    cannot be proven to BE this bridge -- a pre-#694 bridge, any bridge on a
    platform with no ``/proc``, or a recycled pid -- is refused on every path
    and judged litter on none. This call has no evidence the loader lacks, and
    every version that manufactured some destroyed a live bridge's handle in
    one narrowing or another. So such a record, and the ``.log`` its surviving
    registry shields here, are RETAINED; the residual that costs is measured
    and disclosed in the PR body, and pinned by
    ``test_gc_retains_the_record_discovery_cannot_judge``. Retention is
    recoverable, a destroyed handle is not (#618).

    Returns a summary: ``live_instances``, ``registries_purged`` (dead
    registries the liveness sweep removed), ``logs_removed``,
    ``sockets_removed``, ``last_used_removed``, and ``removed`` (the list of
    removed paths, the purged registries included -- reporting those as a
    count alone hid a destructive action behind a number).
    """
    inst_dir = instances_dir()
    summary: dict[str, Any] = {
        "live_instances": 0,
        "registries_purged": 0,
        "logs_removed": 0,
        "sockets_removed": 0,
        "last_used_removed": 0,
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
        # ``removed`` is documented as the list of removed paths and is the only
        # per-path account this call gives, so the registries the liveness sweep
        # took are named here, not merely counted (#618).
        purged_registries = sorted(registries_before - registries_after)
        summary["registries_purged"] = len(purged_registries)
        summary["removed"].extend(str(p) for p in purged_registries)
        # Same derivation the loader uses, for the same reason: ``Path.stem``
        # and ``Path.suffix`` read a leading dot run as part of the name, so a
        # legal all-dot id (``...`` -> ``....json`` / ``....sock``) reverse-maps
        # to nothing and its leftovers could never be reaped. One id per
        # filename, derived one way, on both sides of this sweep.
        live_ids = {_registry_own_id(p) for p in registries_after}
        for entry in sorted(inst_dir.iterdir()):
            # Never touch the shared spawn lock or any surviving (live) registry.
            if entry.name == ".spawn.lock" or entry.name.endswith(".json"):
                continue
            # A .log/.sock whose registry is gone belongs to a dead/long-gone
            # instance -- the registry was purged (now or earlier) or never
            # existed (and, under the lock, is not a spawn in flight).
            # A ``.last_used`` sidecar is reaped on the same evidence as the
            # ``.log`` beside it: nothing in this tree writes one, but a host
            # running a build that does accumulates them and `gc` is their
            # only reaper (#733 F6).
            suffix = next((s for s in (".log", ".sock", ".last_used")
                           if entry.name.endswith(s)), None)
            if suffix is None or entry.name.removesuffix(suffix) in live_ids:
                continue
            # A registry-less ``.sock`` is also what an in-flight registration
            # looks like from here: a bridge binds before it writes its registry
            # and one the GUI plugin starts takes no spawn lock, so the lock
            # above cannot order it. Unlinking that name leaves the bridge
            # serving on an unlinked inode, reachable by nobody. So this unlink
            # takes the same evidence as every other one in this module --
            # positive proof that nothing is bound -- and an unknowable answer
            # keeps the file. The earlier form refused only on a positive
            # ``True``, which meant that wherever the kernel cannot be asked it
            # reaped a bridge's own socket, and it also let a WRONG ``False``
            # (a path the listing cannot represent) do the same (#618).
            if suffix == ".sock" and path_has_bound_socket(entry) is not False:
                continue
            with contextlib.suppress(OSError):
                entry.unlink()
                summary["removed"].append(str(entry))
                summary[{".log": "logs_removed",
                         ".sock": "sockets_removed",
                         ".last_used": "last_used_removed"}[suffix]] += 1
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
    if instance.unreachable:
        # `unreachable` is a promise: this record resolved for the lifecycle
        # lookup ONLY (#694) and nothing may be dispatched to it. Enforced here
        # instead of left to connect() failing, because the confinement arm
        # resolves records whose recorded path may well have something listening
        # on it -- a socket outside this user's cache, which is exactly the file
        # discovery refused to trust (#618) -- and `session restart` dispatches
        # `list_targets` to the handle it resolves. The refusal belongs at the
        # one chokepoint every dispatch goes through.
        raise BridgeError(
            f"bridge_unreachable: Binary Ninja bridge instance "
            f"{instance_selector(instance)!r} (pid {instance.pid}) resolved for "
            f"lifecycle commands only: its recorded socket is either gone or not "
            f"one this cache can vouch for, so nothing may be dispatched to it "
            f"-- stop it with `bn session stop {instance_selector(instance)}`"
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


def unwrap_result(response: Any, op: str) -> Any:
    """The `result` an ok reply must carry.

    A reply of `{"ok": true}` with no `result` key is the same version-skew class
    as a malformed result, and `response["result"]` raised a bare `KeyError` out
    of `main()`, which catches only :class:`BridgeError`, for exit 1 and a
    traceback on every output format. The documented code for a response this
    CLI cannot read is 2.

    Stated precisely, because the earlier wording implied more than it should:
    the production transport already refuses this shape (`_send_request_to_instance`
    above raises :class:`BridgeError` for an ok reply with no `result`), so
    through that path the `KeyError` was unreachable. This is the caller-side
    half of the same guarantee, and it is not redundant -- it is what makes the
    guarantee a property of this package rather than of one caller's transport,
    which is why it lives HERE, beside that check, rather than in `cli.py`: the
    `Client` surface cannot import `cli` (`bn/__init__` imports `client`, so
    that edge closes a cycle) and ten shipped sites in `client.py`,
    `commands/admin.py` and `commands/misc.py` sat outside the rule while it
    was CLI-private -- one of them, `target list`, raising the documented
    `KeyError` and another diagnosing a valid selector as unknown off a missing
    envelope. Every place that reads a `result` off a reply goes through here:
    `cli.py`, `client.py`, `commands/admin.py` and `commands/misc.py` are swept
    by `test_no_bridge_reply_is_indexed_for_its_result_outside_the_unwrap_helper`.
    """
    if not isinstance(response, dict) or "result" not in response:
        raise BridgeError(
            f"the bridge reply to {op!r} carries no `result` -- the response was "
            f"malformed or newer than this CLI, so the outcome could not be "
            f"determined. Compare the bridge and CLI builds with `bn doctor`."
        )
    return response["result"]


def _find_bn_agent() -> list[str]:
    """Return the command to invoke bn-agent."""
    # Prefer the bn-agent script in the same directory as sys.executable
    exe_dir = Path(sys.executable).parent
    bn_agent = exe_dir / "bn-agent"
    if bn_agent.exists():
        return [str(bn_agent)]
    return [sys.executable, "-m", "bn.headless"]


def _log_tail(log_path: Path, lines: int = 20, *, start: int = 0) -> str:
    """Return the last *lines* of the spawn log, formatted for an error message.

    *start* is a byte offset: when the log was APPENDED to rather than
    truncated (see ``_spawn_instance_unlocked``), only the bytes this spawn
    produced may be quoted, or the error attributes the previous bridge's
    output to the child that just failed.
    """
    try:
        with log_path.open("rb") as stream:
            if start:
                stream.seek(start)
            text = stream.read().decode("utf-8", errors="replace")
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
    reg_path = bridge_registry_path(instance_id)
    # A registry file still on disk after the collision pass above is one
    # discovery refused without being able to prove it litter: an unparseable
    # document, a field whose type rules it out, or a record this loader keeps
    # on purpose because deleting it could cost a live bridge its only handle.
    # Every one of those is also invisible to that pass, and the bridge that
    # wrote it may be listening right now (the child then refuses to displace
    # its socket and exits). So truncating `<id>.log` here would destroy that
    # process's only recorded output on evidence that says nothing about whether
    # the id is free: append instead, and bound the diagnostic tail to what THIS
    # child wrote so the error never quotes the previous bridge's lines as its
    # own (#618).
    keep_log = reg_path.exists()
    log_start = 0
    # Nothing in this code removes a record it refuses, so the append above can
    # otherwise repeat with no statement of why the spawn keeps failing. Say what
    # is actually known -- a registry file under this id that discovery did not
    # resolve to a running bridge -- and NOT why, because this arm cannot tell an
    # uninterpretable document from a record deliberately refused and kept
    # (an unconfined socket with an alive owner that proves no identity parses
    # perfectly). "Remove it" is not the advice either: in this state the record
    # is hidden from `session list`, so the operator cannot check it through the
    # CLI, and removing a LIVE bridge's record leaves `instance gc` free to
    # unlink that bridge's socket.
    leftover_note = (
        f" A registry file for {instance_id!r} is already on disk at {reg_path} "
        f"and discovery did not resolve it to a running bridge, so this log was "
        f"appended to rather than replaced. It is kept deliberately: it may "
        f"belong to a bridge that is still running under this id. Inspect that "
        f"file (and the pid and socket it names) before removing it -- removing "
        f"it while that bridge is alive lets `bn instance gc` unlink its socket."
        if keep_log
        else ""
    )
    if keep_log:
        with contextlib.suppress(OSError):
            log_start = log_path.stat().st_size
    log_file = open(log_path, "a" if keep_log else "w")  # noqa: SIM115

    cmd = _find_bn_agent() + ["--instance-id", instance_id]
    try:
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    finally:
        # The child inherits the fd across a successful Popen; close the parent's
        # copy either way. Without this the write handle on a failed spawn lives
        # exactly as long as the traceback that holds the frame owning it: the
        # measured window is the exception propagating out of here, but anything
        # that keeps that traceback alive -- a stored exception, a debugger, a
        # caller that formats it later -- keeps the fd alive with it (#618).
        log_file.close()

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
            # The note goes to the CALLER, not into the log: the log is the file
            # the note is about, and writing it there grows the very thing this
            # arm is trying not to destroy, once per attempt.
            _append_spawn_diagnostic(log_path, message)
            raise BridgeError(
                f"{message}{leftover_note}{_log_tail(log_path, start=log_start)}"
            )
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
    raise BridgeError(f"{message}{leftover_note}")


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
