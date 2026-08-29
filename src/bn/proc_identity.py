"""Durable process identity and atomic process pinning for a bridge (#694).

A bridge registry records a pid. A pid alone is NOT an identity: the bridge can
die (crash, OOM-kill, SIGKILL) and the kernel can hand the same number to an
unrelated process, after which every liveness probe -- and worse, every
SIGTERM/SIGKILL fallback -- is talking about a stranger.

Two things make signalling that pid safe, and neither is enough alone:

* **Identity.** ``(boot id, pid, start ticks)``. Start ticks are "since boot", so
  they are unique only WITHIN one boot, while registries live in a persistent
  cache directory (``~/.cache/bn``): after a reboot an old registry's
  ``(pid, ticks)`` pair can collide with a brand-new unrelated process. The
  kernel boot id closes that hole, so a registry that predates this boot -- or
  records no boot id at all -- is never proven.
* **Atomicity.** Reading ``/proc`` and then calling ``os.kill`` is
  check-then-act: the verified process can exit and its pid be recycled in
  between. ``os.pidfd_open`` takes a reference to the kernel's ``struct pid``,
  which prevents that pid number from being recycled while the reference is held
  and makes ``signal.pidfd_send_signal`` deliver to that exact process. Identity
  is therefore validated *through an open pin* and the signal sent *through the
  same pin*. A platform without pidfd gets no fallback signalling at all --
  refusing is correct, guessing is not.

This module is symlinked into ``bn_agent_bridge`` (like ``paths.py`` and
``version.py``) so the writer and the verifier share one implementation.
"""
from __future__ import annotations

import contextlib
import os
import signal as signal_module
from pathlib import Path
from typing import Any

# /proc/<pid>/stat field 22 (1-indexed) is the process start time in clock ticks
# since boot. Fields 1 and 2 are the pid and the parenthesised comm -- which may
# itself contain spaces and parens -- so parsing starts after the LAST ')', where
# field N lands at index N - 3.
_STARTTIME_INDEX = 22 - 3

# The kernel's per-boot random id: stable for the life of the boot, different
# after every reboot -- exactly the scope start ticks are missing.
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

# The only primitives that make identity-check-then-signal atomic (Python 3.9+ on
# Linux 5.3+). Availability is a property of the INTERPRETER as well as the
# kernel: a CPython built without the wrappers reports False even on a modern
# Linux (the python-build-standalone interpreters uv installs are one such build).
# Where the flag is False there is no safe fallback signal, so every fallback
# signalling path refuses instead of degrading to a racy os.kill.
PIDFD_AVAILABLE = hasattr(os, "pidfd_open") and hasattr(
    signal_module, "pidfd_send_signal"
)


def process_start_ticks(pid: int) -> int | None:
    """*pid*'s start time in clock ticks since boot, or None when unknowable.

    None means "identity NOT proven" and must never be treated as a match: it
    covers a platform without ``/proc``, a process that is already gone, and a
    stat line this parser does not recognise.
    """
    try:
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    close = stat_text.rfind(")")
    if close < 0:
        return None
    fields = stat_text[close + 1 :].split()
    if len(fields) <= _STARTTIME_INDEX:
        return None
    try:
        ticks = int(fields[_STARTTIME_INDEX])
    except ValueError:
        return None
    return ticks if ticks >= 0 else None


def boot_id() -> str | None:
    """This boot's kernel id, or None when it cannot be read."""
    try:
        text = _BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def identity_payload() -> dict[str, Any]:
    """The identity fields the calling process records for itself.

    Empty when this platform cannot supply a COMPLETE identity: half an identity
    is unprovable anyway, so recording it would only look like proof.
    """
    ticks = process_start_ticks(os.getpid())
    current_boot = boot_id()
    if ticks is None or current_boot is None:
        return {}
    return {"pid_start_ticks": ticks, "boot_id": current_boot}


def recorded_start_ticks(payload: object) -> int | None:
    """The start ticks a registry payload recorded, or None when absent.

    A malformed value (non-int, bool, negative) is treated as absent rather than
    as a mismatch: the bridge that wrote it never proved anything either way.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("pid_start_ticks")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


def recorded_boot_id(payload: object) -> str | None:
    """The boot id a registry payload recorded, or None when absent/malformed."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("boot_id")
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def identity_verdict(payload: object, pid: int) -> str:
    """Compare a registry payload's recorded identity against the live *pid*.

    Returns:
      ``"proven"``     -- same boot, and the pid still names the process that
                          wrote the payload.
      ``"mismatch"``   -- positive evidence the record is stale: written under a
                          different boot, or the pid now reports another start
                          time (the bridge exited and its pid was reused).
      ``"unrecorded"`` -- no evidence either way: an older bridge that recorded no
                          identity (or only half of one), a platform without
                          ``/proc``, or a process that is already gone.
    """
    recorded_boot = recorded_boot_id(payload)
    recorded_ticks = recorded_start_ticks(payload)
    if recorded_boot is None or recorded_ticks is None:
        return "unrecorded"
    current_boot = boot_id()
    if current_boot is None:
        return "unrecorded"
    if current_boot != recorded_boot:
        # Start ticks mean something only within one boot and this record predates
        # this one, so whatever holds the pid now is definitively not the bridge.
        return "mismatch"
    observed = process_start_ticks(pid)
    if observed is None:
        return "unrecorded"
    return "proven" if observed == recorded_ticks else "mismatch"


class PinUnavailable(RuntimeError):
    """No atomic pin could be taken for a pid; the message says why."""


class ProcessPin:
    """A pid pinned to ONE process for as long as the pin is held.

    While the pin is open the pid number cannot be recycled and
    ``signal.pidfd_send_signal`` addresses the pinned process, so validating
    identity through :meth:`verdict` and then signalling through :meth:`send`
    cannot be split by an exit-and-reuse race. Use it as a context manager.
    """

    __slots__ = ("pid", "_fd")

    def __init__(self, pid: int, fd: int) -> None:
        self.pid = pid
        self._fd = fd

    def __enter__(self) -> ProcessPin:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._fd < 0

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)

    def verdict(self, payload: object) -> str:
        """Identity verdict for the PINNED process (its pid cannot have moved)."""
        if self.closed:
            raise RuntimeError("process pin is closed")
        return identity_verdict(payload, self.pid)

    def send(self, sig: int) -> None:
        """Deliver *sig* to the pinned process.

        Raises ``ProcessLookupError`` when the process has already been reaped and
        ``OSError`` for anything else (e.g. ``EPERM``). It can never deliver to a
        different process.
        """
        if self.closed:
            raise RuntimeError("process pin is closed")
        signal_module.pidfd_send_signal(self._fd, sig)


def pin_process(pid: int) -> ProcessPin:
    """Pin *pid*, or raise :class:`PinUnavailable` with a user-facing reason."""
    if not PIDFD_AVAILABLE:
        raise PinUnavailable(
            "this platform provides no pidfd (os.pidfd_open / "
            "signal.pidfd_send_signal), so a pid cannot be pinned and any "
            "check-then-signal would race pid reuse"
        )
    try:
        fd = os.pidfd_open(int(pid))
    except ProcessLookupError:
        raise PinUnavailable(
            f"pid {pid} is not running, so there is nothing to signal"
        ) from None
    except (OSError, ValueError, OverflowError) as exc:
        raise PinUnavailable(f"pid {pid} could not be pinned ({exc})") from exc
    return ProcessPin(int(pid), fd)
