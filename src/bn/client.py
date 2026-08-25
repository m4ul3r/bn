from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from .transport import BridgeError, _resolve_timeout, send_request

_MALFORMED_REPLY = (
    "malformed bridge reply (no result); the bridge may be stale -- restart it and retry"
)


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
    ) -> Any:
        response = send_request(
            op,
            params=dict(params or {}),
            instance_id=self.instance,
            target=self.target,
            timeout=timeout,
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
        effective_timeout = _resolve_timeout(self.timeout)
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
            page = self._request(op, page_params, timeout=page_timeout)
            if not isinstance(page, dict):
                raise BridgeError(f"malformed {op} page: expected an object")
            page_items = page.get("items")
            bridge_has_more = page.get("has_more")
            if not isinstance(page_items, list):
                raise BridgeError(f"malformed {op} page: items must be a list")
            if not isinstance(bridge_has_more, bool):
                raise BridgeError(f"malformed {op} page: has_more must be a boolean")
            if bridge_has_more and not page_items:
                raise BridgeError(f"malformed {op} page: has_more with an empty page")
            returned = page.get("returned")
            if returned is not None and (
                not isinstance(returned, int)
                or isinstance(returned, bool)
                or returned != len(page_items)
            ):
                raise BridgeError(
                    f"malformed {op} page: returned must equal items length"
                )
            total = page.get("total")
            if total is not None:
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total < 0
                    or (bool(page_items) and total < offset + len(page_items))
                ):
                    raise BridgeError(
                        f"malformed {op} page: invalid total {total!r}"
                    )
                if not bridge_has_more and offset + len(page_items) < total:
                    raise BridgeError(
                        f"malformed {op} page: total={total} requires "
                        "has_more=true at this offset"
                    )
                if (
                    aggregate is not None
                    and aggregate.get("total") is not None
                    and aggregate.get("total") != total
                ):
                    raise BridgeError(
                        f"malformed {op} page: total changed across pages"
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
        aggregate["has_more"] = bool(
            bridge_has_more and limit is not None and len(items) >= limit
        )
        if limit is None:
            aggregate.pop("limit", None)
        else:
            aggregate["limit"] = limit
        return aggregate
