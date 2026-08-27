from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from .transport import BridgeError, _resolve_timeout, send_request

_MALFORMED_REPLY = (
    "malformed bridge reply (no result); the bridge may be stale -- restart it and retry"
)


# Distinguishes "no page has reported a total yet" from "a page reported None",
# which `.get("total")` on an aggregate cannot.
_UNSEEN: Any = object()


def _validate_page(
    op: str,
    page: Any,
    *,
    requested_offset: int,
    requested_limit: int,
    previous_total: Any = _UNSEEN,
) -> tuple[list[Any], bool, int | None]:
    """Reject a page that cannot be trusted to advance a collection.

    The caller tracks progress with its own arithmetic, so an unchecked page lets a
    stale or version-skewed bridge silently duplicate rows (echoing the same offset
    forever) or over-deliver past the requested limit. Both read as data, not as an
    error, which is the worst failure mode for an agent.
    """
    if not isinstance(page, dict):
        raise BridgeError(f"malformed {op} page: expected an object")
    items = page.get("items")
    if not isinstance(items, list):
        raise BridgeError(f"malformed {op} page: items must be a list")
    has_more = page.get("has_more")
    if not isinstance(has_more, bool):
        raise BridgeError(f"malformed {op} page: has_more must be a boolean")
    if has_more and not items:
        raise BridgeError(f"malformed {op} page: has_more with an empty page")
    if len(items) > requested_limit:
        raise BridgeError(
            f"malformed {op} page: returned {len(items)} items for requested "
            f"limit {requested_limit}"
        )
    # Every current bridge envelope publishes `offset`. Requiring it is the only
    # way to detect a bridge that ignores pagination: with a known `total` a
    # repeated page 1 that simply omits `offset` passes every other check and the
    # caller silently receives duplicates.
    echoed_offset = page.get("offset")
    if not isinstance(echoed_offset, int) or isinstance(echoed_offset, bool):
        raise BridgeError(
            f"malformed {op} page: did not publish an integer offset "
            f"(got {echoed_offset!r}); cannot confirm pagination advanced"
        )
    if echoed_offset != requested_offset:
        raise BridgeError(
            f"malformed {op} page: echoed offset {echoed_offset} for requested "
            f"offset {requested_offset}; the bridge is not honoring pagination"
        )
    returned = page.get("returned")
    if returned is not None and (
        not isinstance(returned, int)
        or isinstance(returned, bool)
        or returned != len(items)
    ):
        raise BridgeError(f"malformed {op} page: returned must equal items length")
    total = page.get("total")
    if total is not None and (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or (bool(items) and total < requested_offset + len(items))
    ):
        raise BridgeError(f"malformed {op} page: invalid total {total!r}")
    # Checked outside the `total is not None` guard on purpose: None->int and
    # int->None are transitions too, and `_UNSEEN` is what distinguishes "no page
    # has reported a total yet" from "a page reported None".
    if previous_total is not _UNSEEN and previous_total != total:
        raise BridgeError(
            f"malformed {op} page: total changed across pages "
            f"({previous_total!r} -> {total!r})"
        )
    if (
        total is not None
        and not has_more
        and requested_offset + len(items) < total
    ):
        raise BridgeError(
            f"malformed {op} page: total={total} requires has_more=true at "
            "this offset"
        )
    return items, has_more, total


class Client:
    """Synchronous, explicitly bound client for the Binary Ninja bridge."""

    __slots__ = ("_instance", "_target", "_timeout")

    def __init__(
        self,
        instance: str | None = None,
        target: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self._instance = instance
        self._target = target
        self._timeout = timeout

    @property
    def instance(self) -> str | None:
        return self._instance

    @property
    def target(self) -> str | None:
        return self._target

    @property
    def timeout(self) -> float | None:
        return self._timeout

    def _request(
        self,
        op: str,
        params: Mapping[str, Any] | None,
        *,
        timeout: float | None,
        resolved: bool = False,
    ) -> Any:
        # `resolved=True` says "this budget already had BN_REQUEST_TIMEOUT applied".
        # Only paginating callers set it: they resolve the override once into an
        # end-to-end deadline and then pass the shrinking remainder, which
        # send_request must forward verbatim instead of re-expanding to the full
        # env value. Keep the kwarg absent otherwise so a single request keeps the
        # documented "env override always wins" behaviour.
        extra = {"resolved": True} if resolved else {}
        response = send_request(
            op,
            params=dict(params or {}),
            instance_id=self.instance,
            target=self.target,
            timeout=timeout,
            **extra,
        )
        if not isinstance(response, dict) or "result" not in response:
            raise BridgeError(_MALFORMED_REPLY)
        return response["result"]

    def request(
        self, op: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        return self._request(op, params, timeout=self.timeout)

    def collect(
        self,
        op: str,
        params: Mapping[str, Any] | None = None,
        *,
        page_size: int = 500,
        limit: int | None = None,
        resolved: bool = False,
    ) -> dict[str, Any]:
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        base_params = dict(params or {})
        if "limit" in base_params:
            raise ValueError("params must not contain limit")

        initial_offset = int(base_params.get("offset", 0))
        if limit == 0:
            return {
                "items": [],
                "offset": initial_offset,
                "returned": 0,
                "has_more": False,
                "total": None,
            }

        offset = initial_offset
        items: list[Any] = []
        aggregate: dict[str, Any] | None = None
        bridge_has_more = False
        seen_total: Any = _UNSEEN
        # `resolved=True` says self.timeout is ALREADY the remaining slice of a
        # budget someone else resolved (bn-kernel's two-phase search). Re-applying
        # BN_REQUEST_TIMEOUT here would hand this phase the full env value again.
        effective_timeout = (
            self.timeout if resolved else _resolve_timeout(self.timeout)
        )
        deadline = (
            time.monotonic() + effective_timeout
            if effective_timeout is not None
            else None
        )

        while True:
            remaining = None if limit is None else limit - len(items)
            request_limit = page_size if remaining is None else min(page_size, remaining)
            page_params = dict(base_params)
            page_params["offset"] = offset
            page_params["limit"] = request_limit

            page_timeout = None
            if deadline is not None:
                page_timeout = deadline - time.monotonic()
                if page_timeout <= 0:
                    raise BridgeError(
                        f"Timed out collecting {op!r}; the end-to-end "
                        "collection deadline expired"
                    )
            page = self._request(
                op, page_params, timeout=page_timeout, resolved=True
            )
            page_items, bridge_has_more, total = _validate_page(
                op,
                page,
                requested_offset=offset,
                requested_limit=request_limit,
                previous_total=seen_total,
            )
            seen_total = total
            if bridge_has_more and limit is None and deadline is None and total is None:
                # Nothing left can stop this loop: no caller limit, no deadline
                # (BN_REQUEST_TIMEOUT disabled), and the bridge published no total.
                # A bridge that mis-reports pagination would page and allocate
                # forever, taking the retained kernel with it. Refuse instead of
                # substituting an arbitrary cap that silently truncates real data.
                raise BridgeError(
                    f"refusing to collect {op!r}: the collection is intrinsically "
                    "unbounded (no limit=, no request deadline because "
                    "BN_REQUEST_TIMEOUT is disabled, and the bridge reported "
                    "total=null while claiming more pages). Pass an explicit "
                    "limit= or re-enable BN_REQUEST_TIMEOUT."
                )

            if aggregate is None:
                aggregate = dict(page)
            items.extend(page_items)
            offset += len(page_items)

            if not bridge_has_more:
                break
            if limit is not None and len(items) >= limit:
                break

        assert aggregate is not None
        aggregate["items"] = items
        aggregate["offset"] = initial_offset
        aggregate["returned"] = len(items)
        aggregate["total"] = seen_total
        aggregate["has_more"] = bool(
            bridge_has_more and limit is not None and len(items) >= limit
        )
        if limit is None:
            aggregate.pop("limit", None)
        else:
            aggregate["limit"] = limit
        return aggregate
