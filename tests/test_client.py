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
                    "limit": 2,
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
            {"result": {"items": [{"id": 1}, {"id": 2}], "offset": 0,
                        "has_more": True, "total": 5}},
            {"result": {"items": [{"id": 3}], "offset": 2, "has_more": True,
                        "total": 5}},
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
        "limit": 3,
    }
    assert calls == [{"offset": 0, "limit": 2}, {"offset": 2, "limit": 1}]


def test_collect_exact_limit_does_not_claim_more_when_bridge_is_finished(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda *args, **kwargs: {
            "result": {"items": [{"id": 1}], "offset": 0, "has_more": False,
                       "total": 1}
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
                "offset": 0,
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


def _one_page(**overrides):
    page = {
        "items": [{"id": 1}],
        "offset": 0,
        "returned": 1,
        "has_more": False,
        "total": 1,
    }
    page.update(overrides)
    return {"result": page}


def test_collect_rejects_a_page_whose_offset_does_not_echo_the_request(monkeypatch):
    """A bridge that ignores `offset` re-serves page 0 forever; the client used to
    accept the duplicates because it tracked progress with its own arithmetic."""
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda op, **kwargs: _one_page(
            items=[{"id": 1}, {"id": 2}],
            offset=0,
            returned=2,
            has_more=True,
            total=None,
        ),
    )

    with pytest.raises(BridgeError, match=r"offset 0 for requested offset 2"):
        Client().collect("list_functions", page_size=2, limit=6)


def test_collect_rejects_a_page_larger_than_the_requested_limit(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda op, **kwargs: _one_page(
            items=[{"id": index} for index in range(25)],
            returned=25,
            total=25,
        ),
    )

    with pytest.raises(BridgeError, match=r"returned 25 items for requested limit 10"):
        Client().collect("list_functions", limit=10)


def test_collect_refuses_an_intrinsically_unbounded_collection(monkeypatch):
    """No caller limit, no deadline (BN_REQUEST_TIMEOUT disabled), total=None and
    has_more=True: nothing can stop the loop, so refuse instead of spinning."""
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0")
    pages = 0

    def fake_send_request(op, **kwargs):
        nonlocal pages
        pages += 1
        assert pages < 50, "collect kept paging an unbounded collection"
        offset = kwargs["params"]["offset"]
        return _one_page(
            items=[{"id": offset}],
            offset=offset,
            has_more=True,
            total=None,
        )

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    with pytest.raises(BridgeError, match="intrinsically unbounded"):
        Client().collect("sections", page_size=1)

    assert pages == 1


def test_collect_allows_an_unbounded_shape_when_a_caller_limit_bounds_it(monkeypatch):
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "0")

    def fake_send_request(op, **kwargs):
        offset = kwargs["params"]["offset"]
        return _one_page(
            items=[{"id": offset}],
            offset=offset,
            has_more=True,
            total=None,
        )

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    result = Client().collect("callsites", page_size=1, limit=3)

    assert [row["id"] for row in result["items"]] == [0, 1, 2]
    assert result["has_more"] is True


def test_collect_spends_one_end_to_end_budget_across_pages(monkeypatch):
    """BN_REQUEST_TIMEOUT is the whole-collection budget. Each page must receive
    the *remaining* slice of it, not a fresh copy of the full value."""
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "2.0")
    budgets: list[float] = []

    def fake_send_request(op, **kwargs):
        budgets.append(kwargs["timeout"])
        time.sleep(0.15)
        offset = kwargs["params"]["offset"]
        return _one_page(
            items=[{"id": offset}],
            offset=offset,
            has_more=offset < 2,
            total=3,
        )

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    result = Client(timeout=2.0).collect("strings", page_size=1)

    assert len(result["items"]) == 3
    assert len(budgets) == 3
    assert all(budget <= 2.0 for budget in budgets)
    assert budgets[0] > budgets[1] > budgets[2], budgets


def test_collect_marks_the_budget_resolved_so_transport_cannot_re_expand_it(
    monkeypatch,
):
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "2.0")
    seen: list[dict[str, Any]] = []

    def fake_send_request(op, **kwargs):
        seen.append(kwargs)
        return _one_page()

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    Client(timeout=2.0).collect("strings")

    assert seen and seen[0]["resolved"] is True


def test_collect_with_a_preresolved_budget_does_not_reapply_the_env_override(
    monkeypatch,
):
    """A multi-phase caller (bn-kernel's literal->regex search) resolves
    BN_REQUEST_TIMEOUT once and hands `Client` the remaining slice as its whole
    budget. Re-resolving here would give each phase the full env value again."""
    monkeypatch.setenv("BN_REQUEST_TIMEOUT", "5.0")
    budgets: list[float] = []

    def fake_send_request(op, **kwargs):
        budgets.append(kwargs["timeout"])
        return _one_page()

    monkeypatch.setattr(client_module, "send_request", fake_send_request)

    Client(timeout=0.5).collect("strings", resolved=True)

    assert budgets and 0 < budgets[0] <= 0.5, budgets


def test_collect_requires_the_page_to_echo_an_integer_offset(monkeypatch):
    """A bridge repeating page 1 while omitting `offset` slipped past every other
    check when `total` was known, so a bounded collection returned duplicates."""
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda op, **kwargs: {
            "result": {
                "items": [{"id": 1}, {"id": 2}],
                "returned": 2,
                "has_more": True,
                "total": 4,
            }
        },
    )

    with pytest.raises(BridgeError, match="did not publish an integer offset"):
        Client().collect("list_functions", page_size=2, limit=4)


@pytest.mark.parametrize("echoed", ["0", True, None], ids=["str", "bool", "null"])
def test_collect_rejects_a_non_integer_echoed_offset(monkeypatch, echoed):
    monkeypatch.setattr(
        client_module,
        "send_request",
        lambda op, **kwargs: {
            "result": {
                "items": [{"id": 1}],
                "offset": echoed,
                "returned": 1,
                "has_more": False,
                "total": 1,
            }
        },
    )

    with pytest.raises(BridgeError, match="integer offset"):
        Client().collect("list_functions", page_size=2, limit=4)


@pytest.mark.parametrize(
    "label,first,second",
    [
        (
            "none to int",
            {"items": [{"id": 1}], "offset": 0, "returned": 1, "has_more": True},
            {
                "items": [{"id": 2}],
                "offset": 1,
                "returned": 1,
                "has_more": True,
                "total": 5,
            },
        ),
        (
            "int to none",
            {
                "items": [{"id": 1}],
                "offset": 0,
                "returned": 1,
                "has_more": True,
                "total": 5,
            },
            {"items": [{"id": 2}], "offset": 1, "returned": 1, "has_more": True},
        ),
    ],
)
def test_collect_rejects_any_cross_page_total_transition(
    monkeypatch, label, first, second
):
    """`aggregate.get("total")` is None both when the bridge published no total and
    when it published one that later vanished, so the old check could not see a
    None<->int transition at all."""
    pages = iter([{"result": first}, {"result": second}])
    monkeypatch.setattr(
        client_module, "send_request", lambda op, **kwargs: next(pages)
    )

    with pytest.raises(BridgeError, match="total changed across pages"):
        Client().collect("list_functions", page_size=1, limit=2)
