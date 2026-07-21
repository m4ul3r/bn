"""Idle reaper: an opt-in, headless-only self-shutdown after BN_IDLE_TIMEOUT
seconds of no request activity. Default off -- unset/none/off/0 keep today's
behavior (the bridge lives until an explicit shutdown).

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
# Idle decision (pure, injected clock)
# --------------------------------------------------------------------------

def test_should_stop_when_idle_past_timeout_and_no_requests(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._active_requests.clear()
    assert inst._idle_reaper_should_stop(now=131.0, timeout=30.0) is True


def test_should_not_stop_before_timeout(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._active_requests.clear()
    assert inst._idle_reaper_should_stop(now=115.0, timeout=30.0) is False


def test_should_not_stop_while_request_in_flight(monkeypatch):
    """The hard invariant: never reap while a request (which includes any running
    update_analysis_and_wait) is in flight, no matter how far past the timeout."""
    _, inst = _instance(monkeypatch)
    inst._last_activity = 100.0
    inst._begin_request("req-1")
    assert inst._idle_reaper_should_stop(now=100.0 + 10_000.0, timeout=30.0) is False


# --------------------------------------------------------------------------
# Activity stamp
# --------------------------------------------------------------------------

def test_end_request_stamps_activity(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._last_activity = 0.0
    inst._begin_request("r")
    inst._end_request("r")
    assert inst._last_activity > 0.0


# --------------------------------------------------------------------------
# Watcher thread
# --------------------------------------------------------------------------

def test_start_idle_reaper_shuts_down_when_idle(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._active_requests.clear()
    t = inst._start_idle_reaper(timeout=0.01, poll_interval=0.01)
    assert t is not None
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_start_idle_reaper_holds_while_busy_then_reaps_when_idle(monkeypatch):
    _, inst = _instance(monkeypatch)
    inst._begin_request("busy")
    t = inst._start_idle_reaper(timeout=0.01, poll_interval=0.01)
    # A request is in flight -> must not reap.
    assert inst._shutdown_event.wait(0.3) is False
    # Request completes; the bridge goes idle and the reaper fires.
    inst._end_request("busy")
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)


def test_start_idle_reaper_exits_on_external_shutdown(monkeypatch):
    """If shutdown is requested elsewhere (bn session stop), the reaper thread
    stops promptly instead of lingering until the next poll boundary."""
    _, inst = _instance(monkeypatch)
    inst._begin_request("busy")  # keep it from self-reaping
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
    inst._active_requests.clear()
    monkeypatch.setenv("BN_IDLE_TIMEOUT", "0.01")
    t = inst._maybe_start_idle_reaper()
    assert t is not None
    assert inst._shutdown_event.wait(2.0) is True
    t.join(timeout=2.0)
