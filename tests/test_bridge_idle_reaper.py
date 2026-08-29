"""Idle reaper: an opt-in, headless-only self-shutdown after BN_IDLE_TIMEOUT
seconds of no request activity. Default off -- unset/none/off/0 keep today's
behavior (the bridge lives until an explicit shutdown).

In-flight work is tracked by a server-generated counter (_inflight) that the
request handler raises at admission and lowers only after the response is
written -- independent of the client's request id -- so the reaper never fires
mid-request, mid-analysis, or mid-response, and the idle-to-shutdown latch is
atomic with admission (a request either registers first and blocks shutdown, or
shutdown latches first and the request is refused).

All mocked; no real BN. Loop tests use tiny real timeouts and a bounded
condition wait (never a fixed sleep) to stay fast and non-flaky.
"""
from __future__ import annotations

import time

import pytest

from _bridge_fakes import _load_bridge


def _instance(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    return bridge, bridge.BinaryNinjaBridge()


# --------------------------------------------------------------------------
# BN_IDLE_TIMEOUT parsing (mirrors BN_REQUEST_TIMEOUT grammar)
# --------------------------------------------------------------------------

def test_parse_idle_timeout_accepts_seconds(monkeypatch):
    bridge, _ = _instance(monkeypatch)
    assert bridge._parse_idle_timeout("900") == 900.0
    assert bridge._parse_idle_timeout("0.5") == 0.5


def test_parse_idle_timeout_disabled_sentinels(monkeypatch):
    bridge, _ = _instance(monkeypatch)
    for raw in (None, "", "   ", "none", "NONE", "off", "Off", "0", "0.0"):
        assert bridge._parse_idle_timeout(raw) is None, raw


def test_parse_idle_timeout_rejects_invalid(monkeypatch):
    bridge, _ = _instance(monkeypatch)
    for raw in ("abc", "-5", "nan", "inf", "1e-325"):
        with pytest.raises(RuntimeError, match="BN_IDLE_TIMEOUT"):
            bridge._parse_idle_timeout(raw)


# --------------------------------------------------------------------------
# In-flight accounting (server counter, independent of client id)
# --------------------------------------------------------------------------

def test_enter_request_admits_and_counts(monkeypatch):
    _, inst = _instance(monkeypatch)
    assert inst._inflight == 0
    assert inst._enter_request() is True
    assert inst._inflight == 1


def test_leave_request_decrements_and_stamps_activity(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._enter_request()
    inst._last_activity = 0.0
    inst._leave_request()
    assert inst._inflight == 0
    assert inst._last_activity > 0.0  # stamped AFTER the request completes


# --------------------------------------------------------------------------
# Idle decision + atomic shutdown latch
# --------------------------------------------------------------------------

def test_try_idle_shutdown_fires_when_idle_and_no_inflight(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    assert inst._try_idle_shutdown(now=131.0, timeout=30.0) is True
    assert inst._shutting_down is True  # latched


def test_try_idle_shutdown_not_before_timeout(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    assert inst._try_idle_shutdown(now=115.0, timeout=30.0) is False
    assert inst._shutting_down is False


def test_try_idle_shutdown_held_off_while_inflight(monkeypatch):
    """The hard invariant: never latch shutdown while a request is in flight (which
    includes any running update_analysis_and_wait), no matter how far past timeout."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._enter_request()  # a request is in flight
    assert inst._try_idle_shutdown(now=100.0 + 10_000.0, timeout=30.0) is False
    assert inst._shutting_down is False


@pytest.mark.parametrize("state", ["queued", "running"])
def test_try_idle_shutdown_held_off_while_load_job_nonterminal(monkeypatch, state):
    """A detached load job runs on its own worker thread outside any request's
    dispatch, so _inflight alone can't see it -- the reaper must also consult
    _load_jobs before latching shutdown."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._load_jobs["job-1"] = {"state": state}
    assert inst._try_idle_shutdown(now=100.0 + 10_000.0, timeout=30.0) is False
    assert inst._shutting_down is False


@pytest.mark.parametrize("state", ["complete", "failed"])
def test_try_idle_shutdown_ignores_terminal_load_job(monkeypatch, state):
    """complete/failed jobs are done -- they must never pin shutdown on their own."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._load_jobs["job-1"] = {"state": state}
    assert inst._try_idle_shutdown(now=131.0, timeout=30.0) is True
    assert inst._shutting_down is True


def test_idle_shutdown_waits_for_detached_load_then_full_idle_window(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._load_jobs["job"] = {"state": "running"}
    assert inst._try_idle_shutdown(now=1000.0, timeout=30.0) is False
    assert inst._last_activity == 1000.0
    inst._load_jobs["job"]["state"] = "complete"
    assert inst._try_idle_shutdown(now=1029.0, timeout=30.0) is False
    assert inst._try_idle_shutdown(now=1031.0, timeout=30.0) is True


def test_request_registered_first_blocks_shutdown(monkeypatch):
    """F1 half A: a request that registers before the reaper wins -- shutdown is
    refused because _inflight > 0."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    assert inst._enter_request() is True
    assert inst._try_idle_shutdown(now=1_000.0, timeout=30.0) is False


def test_shutdown_latched_first_refuses_new_requests(monkeypatch):
    """F1 half B: once the reaper latches shutdown, a newly arriving request is
    refused DISPATCH -- so it can never start work into a dying process. The request
    is still counted (so its error response is covered) and must be balanced by
    _leave_request()."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    assert inst._try_idle_shutdown(now=1_000.0, timeout=30.0) is True
    assert inst._enter_request() is False   # dispatch refused
    assert inst._inflight == 1              # but counted for its response
    inst._leave_request()
    assert inst._inflight == 0


# --------------------------------------------------------------------------
# Watcher thread
# --------------------------------------------------------------------------

def test_start_idle_reaper_shuts_down_when_idle(monkeypatch):
    _, inst = _instance(monkeypatch)
    t = inst._start_idle_reaper(timeout=0.01, poll_interval=0.01)
    assert t is not None
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_start_idle_reaper_holds_while_busy_then_reaps_when_idle(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._enter_request()  # busy
    t = inst._start_idle_reaper(timeout=0.01, poll_interval=0.01)
    # A request is in flight -> must not reap.
    assert inst._shutdown_event.wait(0.3) is False
    # Request completes; the bridge goes idle and the reaper fires.
    inst._leave_request()
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)


def test_start_idle_reaper_exits_on_external_shutdown(monkeypatch):
    """If shutdown is requested elsewhere (bn session stop), the reaper thread
    stops promptly instead of lingering until the next poll boundary."""
    _, inst = _instance(monkeypatch)
    inst._enter_request()  # keep it from self-reaping
    t = inst._start_idle_reaper(timeout=100.0, poll_interval=100.0)
    inst._shutdown_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


# --------------------------------------------------------------------------
# Env-gated arming (default OFF)
# --------------------------------------------------------------------------

def test_maybe_start_idle_reaper_disabled_by_default(monkeypatch):
    _, inst = _instance(monkeypatch)
    monkeypatch.delenv("BN_IDLE_TIMEOUT", raising=False)
    assert inst._maybe_start_idle_reaper() is None
    assert inst._shutdown_event.is_set() is False


def test_maybe_start_idle_reaper_armed_by_env(monkeypatch):
    _, inst = _instance(monkeypatch)
    monkeypatch.setenv("BN_IDLE_TIMEOUT", "0.01")
    t = inst._maybe_start_idle_reaper()
    assert t is not None
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)


def test_maybe_start_idle_reaper_raises_on_invalid_env(monkeypatch):
    """#637 re-review (F5): an invalid BN_IDLE_TIMEOUT must fail loud (not silently
    disable). start_headless arms the reaper inside its try/finally, so this raise
    still runs _stop_bridge() cleanup rather than orphaning the started server."""
    _, inst = _instance(monkeypatch)
    monkeypatch.setenv("BN_IDLE_TIMEOUT", "not-a-number")
    with pytest.raises(RuntimeError, match="BN_IDLE_TIMEOUT"):
        inst._maybe_start_idle_reaper()
    # No watcher armed, no shutdown latched by a failed parse.
    assert inst._shutdown_event.is_set() is False
