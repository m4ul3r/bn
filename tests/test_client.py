from __future__ import annotations
import time

from typing import Any

import pytest

import bn.client as client_module
from bn import Client
from bn.transport import BridgeError


def test_request_forwards_binding_timeout_and_copies_params(monkeypatch):
    calls: list[dict[str, Any]] = []
    params = {"query": "parse"}

    def fake_send_request(op: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"op": op, **kwargs})
        kwargs["params"]["query"] = "changed"
        return {"result": {"ok": True}}

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    client = Client(instance="worker", target="sample", timeout=12.5)
    assert client.instance == "worker"
    assert client.target == "sample"
    assert client.timeout == 12.5
    assert client.request("search_functions", params) == {"ok": True}
    assert params == {"query": "parse"}
    assert calls == [
        {
            "op": "search_functions",
            "params": {"query": "changed"},
            "instance_id": "worker",
            "target": "sample",
            "timeout": 12.5,
        }
    ]


@pytest.mark.parametrize("reply", [None, [], {}, {"other": 1}])
def test_request_rejects_malformed_reply(monkeypatch, reply):
    monkeypatch.setattr(client_module, "send_request", lambda *args, **kwargs: reply)

    with pytest.raises(
        BridgeError,
        match=r"malformed bridge reply \(no result\); the bridge may be stale -- restart it and retry",
    ):
        Client().request("target_info")


def test_collect_aggregates_pages_and_preserves_first_page_metadata(monkeypatch):
    calls: list[dict[str, Any]] = []
    pages = iter(
        [
            {
                "result": {
                    "items": [{"id": 10}, {"id": 11}],
                    "offset": 10,
                    "returned": 2,
                    "has_more": True,
                    "total": 13,
                    "kind": "functions",
                }
            },
            {
                "result": {
                    "items": [{"id": 12}],
                    "offset": 12,
                    "returned": 1,
                    "has_more": False,
                    "total": 13,
                    "kind": "ignored",
                }
            },
        ]
    )

    def fake_send_request(op: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"op": op, **kwargs})
        return next(pages)

    monkeypatch.setattr(client_module, "send_request", fake_send_request)
    params = {"offset": "10", "name": "parse"}

    result = Client(instance="worker", target="sample", timeout=3).collect(
        "list_functions", params, page_size=2
    )

    assert params == {"offset": "10", "name": "parse"}
    assert result == {
        "items": [{"id": 10}, {"id": 11}, {"id": 12}],
        "offset": 10,
        "returned": 3,
        "has_more": False,
        "total": 13,
        "kind": "functions",
    }
    assert [call["params"] for call in calls] == [
        {"offset": 10, "name": "parse", "limit": 2},
        {"offset": 12, "name": "parse", "limit": 2},
    ]
    assert all(call["instance_id"] == "worker" for call in calls)
    assert all(call["target"] == "sample" for call in calls)
    timeouts = [call["timeout"] for call in calls]
    assert 0 < timeouts[1] <= timeouts[0] <= 3


def test_collect_stops_at_total_limit_and_reports_remaining_rows(monkeypatch):
    pages = iter(
        [
            {"result": {"items": [{"id": 1}, {"id": 2}], "has_more": True, "total": 5}},
            {"result": {"items": [{"id": 3}], "has_more": True, "total": 5}},
        ]
    )
    calls: list[dict[str, Any]] = []

    def fake_send_request(op: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["params"])
        return next(pages)

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    result = Client().collect("strings", page_size=2, limit=3)

    assert result == {
        "items": [{"id": 1}, {"id": 2}, {"id": 3}],
        "has_more": True,
        "total": 5,
        "offset": 0,
        "returned": 3,
    }
    assert calls == [{"offset": 0, "limit": 2}, {"offset": 2, "limit": 1}]


def test_collect_exact_limit_does_not_claim_more_when_bridge_is_finished(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {
            "result": {"items": [{"id": 1}], "has_more": False, "total": 1}
        },
    )

    result = Client().collect("imports", limit=1)

    assert result["has_more"] is False


def test_collect_zero_limit_makes_no_bridge_call(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("bridge called")

    monkeypatch.setattr(client_module, "send_request", unexpected)

    assert Client().collect("sections", {"offset": "7"}, limit=0) == {
        "items": [],
        "offset": 7,
        "returned": 0,
        "has_more": False,
        "total": None,
    }


@pytest.mark.parametrize(
    ("kwargs", "params", "message"),
    [
        ({"page_size": 0}, {}, "page_size must be at least 1"),
        ({"limit": -1}, {}, "limit must be non-negative"),
        ({}, {"limit": 5}, "params must not contain limit"),
    ],
)
def test_collect_rejects_invalid_limits(kwargs, params, message):
    with pytest.raises(ValueError, match=message):
        Client().collect("list_functions", params, **kwargs)


@pytest.mark.parametrize(
    "result",
    [
        [],
        {},
        {"items": "not-a-list", "has_more": False},
        {"items": [], "has_more": "yes"},
    ],
)
def test_collect_rejects_malformed_page(monkeypatch, result):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {"result": result},
    )

    with pytest.raises(BridgeError, match="list_functions"):
        Client().collect("list_functions")


def test_collect_rejects_zero_progress_page(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {
            "result": {"items": [], "has_more": True, "total": 2}
        },
    )

    with pytest.raises(BridgeError, match="list_functions.*empty page"):
        Client().collect("list_functions")


def test_public_exports_are_exact():
    import bn

    assert bn.__all__ == ["Client", "main", "VERSION"]


def test_collect_rejects_silent_truncation_metadata(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {
            "result": {
                "items": [],
                "returned": 0,
                "total": 3,
                "has_more": False,
            }
        },
    )

    with pytest.raises(BridgeError, match="strings.*total.*has_more"):
        Client().collect("strings")


def test_collect_accepts_empty_page_beyond_total(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {
            "result": {
                "items": [],
                "returned": 0,
                "offset": 100,
                "total": 50,
                "has_more": False,
            }
        },
    )

    assert Client().collect("strings", {"offset": 100}) == {
        "items": [],
        "returned": 0,
        "offset": 100,
        "total": 50,
        "has_more": False,
    }


def test_collect_uses_one_end_to_end_timeout_budget(monkeypatch):
    calls = []

    def fake_send_request(op, **kwargs):
        calls.append(kwargs["timeout"])
        time.sleep(0.02)
        offset = kwargs["params"]["offset"]
        return {
            "result": {
                "items": [{"i": offset}],
                "returned": 1,
                "offset": offset,
                "total": 2,
                "has_more": offset == 0,
            }
        }

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    result = Client(timeout=0.1).collect("strings", page_size=1)

    assert result["returned"] == 2
    assert len(calls) == 2
    assert 0 < calls[1] < calls[0] <= 0.1


def test_collect_honors_environment_timeout(monkeypatch):
    calls = []
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0.05")

    def fake_send_request(op, **kwargs):
        calls.append(kwargs["timeout"])
        return {
            "result": {
                "items": [],
                "returned": 0,
                "offset": 0,
                "total": 0,
                "has_more": False,
            }
        }

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    Client().collect("strings")

    assert len(calls) == 1
    assert 0 < calls[0] <= 0.05
