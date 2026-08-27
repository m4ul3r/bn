"""Retained-kernel access to bn with native and zero-setup CLI backends."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import inspect
import re
import os
import shutil
import threading
import tempfile
import time
import warnings
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "BnError",
    "CliError",
    "BridgeError",
    "VerificationFailed",
    "Result",
    "Session",
    "session",
    "scoped",
    "run",
    "brief",
]

Backend = Literal["cli", "native"]
BackendChoice = Literal["auto", "cli", "native"]
_TEXT_KEYS = ("text", "listing", "body")
_REGEX_METACHARS = ".|()[]{}*+?^$\\"
_ENV_REQUEST_TIMEOUT = "BN_REQUEST_TIMEOUT"


def _env_timeout_override() -> tuple[bool, float | None]:
    """Read BN_REQUEST_TIMEOUT as ``(is_set, seconds_or_None_to_disable)``.

    Mirrors ``bn.transport._resolve_timeout`` deliberately instead of importing it:
    the CLI backend must stay zero-install, so this module cannot depend on the
    ``bn`` package being importable in the kernel interpreter.
    """
    raw = os.environ.get(_ENV_REQUEST_TIMEOUT)
    if raw is None:
        return False, None
    text = raw.strip().lower()

    def _reject() -> BnError:
        return BnError(
            f"{_ENV_REQUEST_TIMEOUT}={raw!r} is not a valid timeout: expected a "
            "positive number of seconds, or one of 0/none/off/empty to disable it.",
            returncode=2,
            argv=(_ENV_REQUEST_TIMEOUT,),
        )

    if text in ("", "none", "off"):
        return True, None
    try:
        value = float(text)
    except ValueError:
        raise _reject() from None
    if not math.isfinite(value) or value < 0 or math.copysign(1.0, value) < 0:
        raise _reject()
    if value == 0.0:
        if any(digit in text for digit in "123456789"):
            raise _reject()
        return True, None
    return True, value


def _resolve_budget(timeout: float | None) -> float | None:
    """The single end-to-end budget for one operation, env override applied once.

    ``None`` means "no deadline" -- the documented 0/none/off spelling. Callers
    resolve exactly once and then pass the shrinking remainder down, so a
    multi-page collection cannot silently re-expand to the full env value per page.
    """
    present, value = _env_timeout_override()
    return value if present else timeout


def _format_child_budget(budget: float | None) -> str:
    """Render a resolved budget for the child `bn` process's own env override."""
    return "0" if budget is None else f"{budget:.6g}"


def _timeout_message(operation: str, budget: float, detail: str = "") -> str:
    message = (
        f"{operation} timed out after {budget:.6g}s "
        "(requested end-to-end budget)"
    )
    guidance_marker = "The bridge may"
    if guidance_marker in detail:
        guidance = detail[detail.index(guidance_marker) :]
    else:
        guidance = (
            "The bridge may be busy with analysis; inspect progress with "
            "`bn -i NAME target info`, then raise or disable the limit with "
            "BN_REQUEST_TIMEOUT=<seconds|0> if the operation is intentionally long."
        )
    return f"{message}. {guidance}"


def _should_retry_as_regex(
    query: str,
    payload: Any,
    filters: Mapping[str, Any],
) -> bool:
    if filters.get("regex") or filters.get("exact"):
        return False
    if not isinstance(payload, dict):
        return False
    if not any(character in query for character in _REGEX_METACHARS):
        return False
    if payload.get("total") != 0 and query != ".":
        return False
    try:
        re.compile(query)
    except re.error as exc:
        raise BnError(
            f"invalid regex-like search query {query!r}: {exc}; "
            "pass exact=True to search for it literally",
            returncode=2,
            argv=("search_functions",),
        ) from exc
    return True


def _flatten_function_info(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("function"), dict):
        return payload
    return {
        **{key: value for key, value in payload.items() if key != "function"},
        **payload["function"],
    }


def _require_function_info(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("function"), dict):
        raise BnError(
            "function_info payload contract violation: expected a function object",
            returncode=0,
            argv=("function_info",),
        )
    value = _flatten_function_info(payload)
    assert isinstance(value, dict)
    size = value.get("size")
    size_known = value.get("size_known")
    valid_size = (
        isinstance(size_known, bool)
        and (
            (not size_known and size is None)
            or (
                size_known
                and isinstance(size, int)
                and not isinstance(size, bool)
                and size >= 0
            )
        )
    )
    if not valid_size or not isinstance(value.get("imported"), bool):
        raise BnError(
            "function_info payload contract violation: expected size, "
            "size_known, and imported metadata",
            returncode=0,
            argv=("function_info",),
        )
    return value


def _require_collection(
    op: str,
    value: Any,
    payload: Any,
    *,
    required_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise BnError(
            f"{op} payload contract violation: expected a paged object with items",
            returncode=0,
            argv=(op,),
        )
    if not isinstance(value, list):
        raise BnError(
            f"{op} return contract violation: expected a list",
            returncode=0,
            argv=(op,),
        )
    payload_items = payload["items"]
    if value != payload_items:
        raise BnError(
            f"{op} payload contract violation: returned rows differ from payload items",
            returncode=0,
            argv=(op,),
        )
    total = payload.get("total")
    has_more = payload.get("has_more")
    if (
        not isinstance(has_more, bool)
        or (
            total is not None
            and (
                not isinstance(total, int)
                or isinstance(total, bool)
                or total < len(value)
                or (not has_more and total > len(value) + int(payload.get("offset", 0)))
            )
        )
    ):
        raise BnError(
            f"{op} payload contract violation: inconsistent total/has_more metadata",
            returncode=0,
            argv=(op,),
        )
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise BnError(
                f"{op} row contract violation at index {index}: expected an object",
                returncode=0,
                argv=(op,),
            )
        if isinstance(row.get("items"), list) and "has_more" in row:
            raise BnError(
                f"{op} row contract violation at index {index}: nested page envelope",
                returncode=0,
                argv=(op,),
            )
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise BnError(
                f"{op} row contract violation at index {index}: "
                f"missing {', '.join(missing)}",
                returncode=0,
                argv=(op,),
            )
    return value


def _require_function_rows(op: str, value: Any, payload: Any) -> list[dict[str, Any]]:
    rows = _require_collection(
        op,
        value,
        payload,
        required_keys=("size", "size_known"),
    )
    for index, row in enumerate(rows):
        if (
            not isinstance(row["size"], int)
            or isinstance(row["size"], bool)
            or row["size"] < 0
            or not isinstance(row["size_known"], bool)
        ):
            raise BnError(
                f"{op} row contract violation at index {index}: "
                "size must be a non-negative integer and size_known a boolean",
                returncode=0,
                argv=(op,),
            )
    return rows


def _require_text(op: str, value: Any, payload: Any) -> str:
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("text"), str)
        or not payload["text"].strip()
        or not isinstance(value, str)
        or value != payload["text"]
    ):
        raise BnError(
            f"{op} text contract violation: expected a non-empty payload['text'] string",
            returncode=0,
            argv=(op,),
        )
    return value


def _require_decompile(value: Any, payload: Any) -> str:
    text = _require_text("decompile", value, payload)
    warnings_value = payload.get("warnings") if isinstance(payload, dict) else []
    warnings_text = " ".join(
        str(warning).lower() for warning in (warnings_value or [])
    )
    incomplete = bool(payload.get("analysis_skipped")) or any(
        marker in warnings_text
        for marker in (
            "incomplete stub",
            "skipped analysis",
            "taking too long to analyze",
        )
    )
    if incomplete:
        raise BnError(
            "decompile incomplete placeholder detected; retry with "
            "force_analysis=True and verify the resulting body",
            returncode=0,
            argv=("decompile",),
        )
    return text


def _require_callsites(value: Any, payload: Any) -> list[dict[str, Any]]:
    rows = _require_collection(
        "callsites",
        value,
        payload,
        required_keys=(
            "callee",
            "containing_function",
            "call_addr",
            "caller_static",
        ),
    )
    for index, row in enumerate(rows):
        callee = row["callee"]
        containing = row["containing_function"]
        nested_ok = all(
            isinstance(part, Mapping)
            and isinstance(part.get("name"), str)
            and bool(part["name"])
            and isinstance(part.get("address"), str)
            and part["address"].startswith("0x")
            for part in (callee, containing)
        )
        addresses_ok = all(
            isinstance(row.get(key), str) and row[key].startswith("0x")
            for key in ("call_addr", "caller_static")
        )
        if not nested_ok or not addresses_ok:
            raise BnError(
                f"callsites row contract violation at index {index}: "
                "callee/containing_function and addresses must be attributed "
                "objects with 0x-prefixed addresses",
                returncode=0,
                argv=("callsites",),
            )
    return rows


_ORIENT_COUNTERS = ("comments", "function_comments", "user_symbols")


def _require_orient_digest(payload: Any) -> dict[str, Any]:
    """Fail closed on any orientation digest this gate cannot actually read.

    ``assert_unannotated`` is the contamination gate for benchmark and dogfood
    inputs, so an unreadable digest must never collapse to "zero comments" and be
    certified clean. A stale or version-skewed bridge that renames, drops, or
    retypes ``existing_annotations`` is exactly the case that would otherwise
    convert contaminated data into a confident pass.
    """
    if not isinstance(payload, Mapping):
        raise BnError(
            "orient digest contract violation: expected a mapping payload, got "
            f"{type(payload).__name__}",
            returncode=0,
            argv=("orient_digest",),
        )
    annotations = payload.get("existing_annotations")
    if not isinstance(annotations, Mapping):
        raise BnError(
            "orient digest contract violation: existing_annotations must be a "
            f"mapping, got {type(annotations).__name__}",
            returncode=0,
            argv=("orient_digest",),
        )
    counts: dict[str, int] = {}
    for key in _ORIENT_COUNTERS:
        value = annotations.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise BnError(
                "orient digest contract violation: annotation counter "
                f"{key!r} must be a non-negative integer, got {value!r}",
                returncode=0,
                argv=("orient_digest",),
            )
        counts[key] = value
    return counts


_TARGET_INFO_COUNTS = (
    "function_count",
    "named_function_count",
    "unnamed_function_count",
    "imported_function_count",
)


def _require_target_info(payload: Any) -> dict[str, Any]:
    """Require the canonical `target_info` shape, not merely "a nonempty mapping".

    A nonempty-mapping check accepts any *other* bridge payload -- an `il` result,
    an evidence digest -- and then answers every documented question
    (``function_count``, ``basename``, the import counts) with a silent "absent".
    These keys are exactly what ``bridge.py`` ``_target_info`` always publishes
    (``_function_name_summary`` plus filename/basename/import_symbol_count), so
    requiring them is a real answer-or-fail contract.
    """
    if not isinstance(payload, Mapping):
        raise BnError(
            "target info contract violation: expected a mapping, got "
            f"{type(payload).__name__}",
            returncode=0,
            argv=("target_info",),
        )
    for key in ("filename", "basename"):
        if key not in payload:
            raise BnError(
                f"target info contract violation: missing {key}; this is not a "
                "target_info payload",
                returncode=0,
                argv=("target_info",),
            )
        value = payload[key]
        if value is not None and not isinstance(value, str):
            raise BnError(
                f"target info contract violation: {key} must be a string or null, "
                f"got {value!r}",
                returncode=0,
                argv=("target_info",),
            )
    for key in _TARGET_INFO_COUNTS:
        if key not in payload:
            raise BnError(
                f"target info contract violation: missing {key}; this is not a "
                "target_info payload",
                returncode=0,
                argv=("target_info",),
            )
        value = payload[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BnError(
                f"target info contract violation: {key} must be a non-negative "
                f"integer, got {value!r}",
                returncode=0,
                argv=("target_info",),
            )
    # The bridge deliberately publishes None here when the imports count raises,
    # so null is a real value for this key alone -- absent still is not.
    if "import_symbol_count" not in payload:
        raise BnError(
            "target info contract violation: missing import_symbol_count; this is "
            "not a target_info payload",
            returncode=0,
            argv=("target_info",),
        )
    import_symbol_count = payload["import_symbol_count"]
    if import_symbol_count is not None and (
        not isinstance(import_symbol_count, int)
        or isinstance(import_symbol_count, bool)
        or import_symbol_count < 0
    ):
        raise BnError(
            "target info contract violation: import_symbol_count must be a "
            f"non-negative integer or null, got {import_symbol_count!r}",
            returncode=0,
            argv=("target_info",),
        )
    return dict(payload)


# Distinguishes "no page has reported a total yet" from "a page reported None",
# which `.get("total")` on an aggregate cannot.
_UNSEEN: Any = object()


def _validate_page(
    op: str,
    payload: Any,
    *,
    requested_offset: int,
    requested_limit: int,
    previous_total: Any = _UNSEEN,
) -> tuple[list[Any], bool, int | None]:
    """Reject a page that cannot be trusted to advance a collection.

    ``all()`` tracks progress with its own arithmetic, so an unchecked page lets a
    stale bridge duplicate rows forever (echoing one offset) or over-deliver past
    the requested limit -- both of which read as data, not as an error.
    """
    if not isinstance(payload, dict):
        raise BnError(
            f"bn {op} is not a paged collection",
            returncode=0,
            argv=(op,),
        )
    items = payload.get("items")
    if not isinstance(items, list):
        raise BnError(
            f"malformed {op} page: items must be a list", returncode=0, argv=(op,)
        )
    has_more = payload.get("has_more")
    if not isinstance(has_more, bool):
        raise BnError(
            f"malformed {op} page: has_more must be a boolean",
            returncode=0,
            argv=(op,),
        )
    if has_more and not items:
        raise BnError(
            f"malformed {op} page: has_more with an empty page",
            returncode=0,
            argv=(op,),
        )
    if len(items) > requested_limit:
        raise BnError(
            f"malformed {op} page: returned {len(items)} items for requested "
            f"limit {requested_limit}",
            returncode=0,
            argv=(op,),
        )
    # Every current bridge envelope publishes `offset`. Requiring it is the only
    # way to detect a bridge that ignores pagination: with a known `total` a
    # repeated page 1 that simply omits `offset` passes every other check and the
    # caller silently receives duplicates.
    echoed_offset = payload.get("offset")
    if not isinstance(echoed_offset, int) or isinstance(echoed_offset, bool):
        raise BnError(
            f"malformed {op} page: did not publish an integer offset "
            f"(got {echoed_offset!r}); cannot confirm pagination advanced",
            returncode=0,
            argv=(op,),
        )
    if echoed_offset != requested_offset:
        raise BnError(
            f"malformed {op} page: echoed offset {echoed_offset} for requested "
            f"offset {requested_offset}; the bridge is not honoring pagination",
            returncode=0,
            argv=(op,),
        )
    returned = payload.get("returned")
    if returned is not None and (
        not isinstance(returned, int)
        or isinstance(returned, bool)
        or returned != len(items)
    ):
        raise BnError(
            f"malformed {op} page: returned must equal items length",
            returncode=0,
            argv=(op,),
        )
    total = payload.get("total")
    if total is not None and (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or (bool(items) and requested_offset + len(items) > total)
    ):
        raise BnError(
            f"malformed {op} page: invalid total {total!r}",
            returncode=0,
            argv=(op,),
        )
    # Checked outside the `total is not None` guard on purpose: None->int and
    # int->None are transitions too, and `_UNSEEN` is what distinguishes "no page
    # has reported a total yet" from "a page reported None".
    if previous_total is not _UNSEEN and previous_total != total:
        raise BnError(
            f"malformed {op} page: total changed across pages "
            f"({previous_total!r} -> {total!r})",
            returncode=0,
            argv=(op,),
        )
    if total is not None and not has_more and requested_offset + len(items) < total:
        raise BnError(
            f"malformed {op} page: total={total} requires has_more=true at "
            "this offset",
            returncode=0,
            argv=(op,),
        )
    return items, has_more, total


def _require_row_fields(op: str, payload: dict[str, Any]) -> None:
    """Require a `row_fields: list[str]` schema annotation on a probed page.

    The bridge enforces `limit >= 1`, so a caller-visible `limit=0` "give me
    the schema, not the rows" request is a one-row PROBE at the wire level.
    The probed row proves the schema; without a well-formed `row_fields` the
    caller has no way to know what columns to expect once it does ask for
    real rows.
    """
    row_fields = payload.get("row_fields")
    if not isinstance(row_fields, list) or not all(
        isinstance(field, str) for field in row_fields
    ):
        raise BnError(
            f"malformed {op} page: a limit=0 metadata page must publish "
            f"row_fields as a list of strings (got {row_fields!r})",
            returncode=0,
            argv=(op,),
        )


def _unbounded_collection_error(op: str) -> BnError:
    """Refuse a collection with nothing left that can terminate it."""
    return BnError(
        f"refusing to collect {op!r}: the collection is intrinsically unbounded "
        "(no limit=, no request deadline because BN_REQUEST_TIMEOUT is disabled, "
        "and the bridge reported total=null while claiming more pages). Pass an "
        "explicit limit= or re-enable BN_REQUEST_TIMEOUT.",
        returncode=0,
        argv=(op,),
    )


_ACTIVE_SESSIONS = weakref.WeakSet()
_ACTIVE_SCOPED_CALLBACKS: set[int] = set()
_ACTIVE_SCOPED_BINDINGS: dict[int, tuple[str | None, str | None]] = {}
_SCOPED_CALLBACK_LOCK = threading.Lock()
_SCOPED_BINDING_ATTRIBUTE = "__bn_kernel_binding__"
_FUNCTION_FILTERS = frozenset(
    {"min_address", "max_address", "min_size", "offset", "sort", "reverse", "named"}
)
_SEARCH_FILTERS = frozenset(
    {
        "regex",
        "exact",
        "word",
        "min_address",
        "max_address",
        "min_size",
        "offset",
        "sort",
        "reverse",
    }
)
_STRING_FILTERS = frozenset(
    {
        "query",
        "offset",
        "min_length",
        "max_length",
        "section",
        "no_crt",
        "regex",
        "probable_format_strings",
    }
)
_IMPORT_FILTERS = frozenset({"query", "regex", "offset", "include_got"})
_SECTION_FILTERS = frozenset({"query", "offset"})


def _validate_filters(
    helper: str,
    filters: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    unknown = sorted(set(filters) - allowed)
    if unknown:
        raise TypeError(
            f"{helper} got unexpected keyword argument(s): {', '.join(unknown)}"
        )
    return dict(filters)


_WARNED_BINDING_PAIRS: set[frozenset[tuple[str | None, str | None]]] = set()


def _register_session(session: Session) -> None:
    binding = (session.instance, session.target)
    active_bindings = {
        (active.instance, active.target)
        for active in list(_ACTIVE_SESSIONS)
        if active is not session
    }
    distinct = sorted(
        (other for other in active_bindings if other != binding),
        key=repr,
    )
    # Scan EVERY live distinct binding for the first pair not already reported,
    # not just the lowest-sorting one: with {A,B} and {A,C} recorded, re-creating C
    # while B is live would short-circuit on the known {A,C} and never disclose the
    # novel {A,C}/{B,C} overlap. Sorting only makes the choice deterministic.
    for other in distinct:
        pair = frozenset((binding, other))
        if pair in _WARNED_BINDING_PAIRS:
            continue
        _WARNED_BINDING_PAIRS.add(pair)
        warnings.warn(
            "multiple distinct bn_kernel Session bindings are live in one "
            "shared retained kernel namespace. Concurrent sibling OMP task "
            "agents can overwrite module globals such as s/rows/lines and "
            "silently read another target. Use `await bn_kernel.scoped(...)` "
            "with function-local variables, or use the direct bn CLI for "
            "parallel agents.",
            RuntimeWarning,
            stacklevel=3,
        )
        # At most one warning per registration: one disclosure is enough to send
        # the reader to scoped()/the direct CLI, and the remaining pairs stay
        # unrecorded so a later registration can still surface them.
        break
    _ACTIVE_SESSIONS.add(session)


class BnError(RuntimeError):
    """A typed failure from a bn backend."""

    def __init__(
        self, message: str, *, returncode: int, argv: Sequence[str]
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.argv = tuple(argv)


class CliError(BnError):
    """bn exit 1: CLI-side handler failure."""


class BridgeError(BnError):
    """bn exit 2: transport or bridge-side failure."""


class VerificationFailed(BnError):
    """bn exit 3: a mutation could not be verified or is unsupported."""


_EXIT_ERRORS: dict[int, type[BnError]] = {
    1: CliError,
    2: BridgeError,
    3: VerificationFailed,
}


@dataclass(frozen=True)
class Result:
    """A useful value and the complete backend payload that produced it."""

    value: Any
    payload: Any
    notes: tuple[str, ...]
    argv: tuple[str, ...]
    backend: Backend

    @property
    def total(self) -> int | None:
        if isinstance(self.payload, dict):
            total = self.payload.get("total")
            if isinstance(total, int):
                return total
        return None

    @property
    def row_fields(self) -> list[str] | None:
        """The read's own row-key declaration, or None when it published none.

        Read straight off the payload the bridge returned, so there is no second
        catalog here to drift from the rows: collection schemas differ on purpose
        (functions key on address/size, sections on start/end/length, callsites
        nest callee/containing_function) and this is how to learn which one you
        are holding without a failed `brief()` or a trip through the source.
        """
        if not isinstance(self.payload, Mapping):
            return None
        fields = self.payload.get("row_fields")
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            return None
        return list(fields)


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return items
        for key in _TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return payload


def _terminate(process: Any) -> None:
    """Kill a child that may already have exited.

    `wait_for` can time out microseconds before the child reaps itself, and a
    tighter per-page budget makes that window easy to hit; a bare `process.kill()`
    then raises ProcessLookupError and masks the real timeout.
    """
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


def _flags(flags: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for name, value in flags.items():
        if value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                argv.extend((flag, str(item)))
        else:
            argv.extend((flag, str(value)))
    return argv


def _bn_executable() -> str:
    configured = os.environ.get("BN_BIN")
    executable = configured if configured else shutil.which("bn")
    if executable:
        return executable
    raise BnError(
        "bn executable not found; set BN_BIN or install bn on PATH",
        returncode=127,
        argv=("bn",),
    )


def _load_native_client(
    instance: str | None, target: str | None, timeout: float
) -> Any:
    from bn import Client

    return Client(instance=instance, target=target, timeout=timeout)


class Session:
    """An explicit instance/target binding for retained Python kernels."""

    def __init__(
        self,
        instance: str | None = None,
        target: str | None = None,
        *,
        timeout: float = 120.0,
        backend: BackendChoice = "auto",
    ) -> None:
        configured_backend = os.environ.get("BN_BACKEND")
        if configured_backend is not None:
            env_backend = configured_backend.strip().lower()
            if env_backend not in {"auto", "cli", "native"}:
                raise ValueError(
                    f"BN_BACKEND={configured_backend!r} is invalid; "
                    "expected auto, cli, or native"
                )
            if backend == "auto":
                backend = env_backend
        if backend not in {"auto", "cli", "native"}:
            raise ValueError("backend must be 'auto', 'cli', or 'native'")
        self.instance = instance
        self.target = target
        self.timeout = timeout
        self.last: Result | None = None
        self._client: Any = None

        if backend == "cli":
            self.backend: Backend = "cli"
            _register_session(self)
            return
        try:
            self._client = _load_native_client(instance, target, timeout)
        except ImportError as exc:
            if backend == "native":
                raise BnError(
                    "native bn.Client is unavailable; use backend='cli' or install bn "
                    "into the OMP interpreter",
                    returncode=127,
                    argv=("native",),
                ) from exc
            self.backend = "cli"
        else:
            self.backend = "native"
        _register_session(self)

    def _root_argv(self) -> list[str]:
        argv: list[str] = []
        if self.instance:
            argv.extend(("-i", self.instance))
        if self.target:
            argv.extend(("-t", self.target))
        return argv

    async def run(
        self,
        *args: str,
        raw: bool = False,
        unwrap: bool = True,
        timeout: float | None = None,
        **flags: Any,
    ) -> Any:
        """Run any bn CLI command and return its complete artifact-backed result."""
        return await self._run_resolved(
            *args,
            raw=raw,
            unwrap=unwrap,
            budget=_resolve_budget(self.timeout if timeout is None else timeout),
            **flags,
        )

    async def _run_resolved(
        self,
        *args: str,
        raw: bool = False,
        unwrap: bool = True,
        budget: float | None,
        **flags: Any,
    ) -> Any:
        """Spawn bn with an ALREADY-RESOLVED end-to-end budget.

        `budget` has had BN_REQUEST_TIMEOUT applied exactly once by the caller;
        `None` means the override disabled the deadline. A paginating caller passes
        its shrinking remainder here so the env value is not re-applied per page.
        """
        self.last = None

        executable = _bn_executable()
        argv = self._root_argv()
        argv.extend(str(arg) for arg in args)
        argv.extend(_flags(flags))
        output_format = "text" if raw else "json"

        temporary = tempfile.TemporaryFile(mode="w+b")
        output_fd = temporary.fileno()
        fd_root = Path("/proc/self/fd")
        if not fd_root.is_dir():
            fd_root = Path("/dev/fd")
        output_path = fd_root / str(output_fd)

        # The child re-reads BN_REQUEST_TIMEOUT from its own environment, so it must
        # be told the REMAINING budget. Inheriting the original value would let the
        # bridge-side cancellation be scheduled off a deadline this page has already
        # spent, delaying cancellation long after we stopped waiting.
        child_env = dict(os.environ)
        child_env[_ENV_REQUEST_TIMEOUT] = _format_child_budget(budget)

        stderr_text = ""
        body = ""
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    executable,
                    *argv,
                    "--format",
                    output_format,
                    "--out",
                    str(output_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    pass_fds=(output_fd,),
                    env=child_env,
                )
            except OSError as exc:
                raise BnError(
                    f"could not execute bn: {exc}", returncode=127, argv=argv
                ) from exc

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=budget
                )
            except asyncio.TimeoutError:
                _terminate(process)
                await process.communicate()
                raise BnError(
                    _timeout_message(f"bn {' '.join(argv)}", budget or 0.0),
                    returncode=124,
                    argv=argv,
                ) from None
            except asyncio.CancelledError:
                _terminate(process)
                await process.communicate()
                raise

            stdout_text = stdout_bytes.decode(errors="replace").strip()
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            if process.returncode:
                error_type = _EXIT_ERRORS.get(process.returncode, BnError)
                raise error_type(
                    stderr_text or stdout_text or "bn failed",
                    returncode=process.returncode,
                    argv=argv,
                )
            temporary.seek(0)
            body = temporary.read().decode(encoding="utf-8", errors="replace")
        finally:
            temporary.close()

        if raw:
            payload: Any = body
        else:
            try:
                if not body.strip():
                    raise json.JSONDecodeError("empty output", body, 0)
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise BnError(
                    f"bn returned invalid JSON: {exc.msg}",
                    returncode=0,
                    argv=argv,
                ) from exc

        notes = tuple(line for line in stderr_text.splitlines() if line.strip())
        value = payload if raw or not unwrap else _unwrap(payload)
        self.last = Result(
            value=value,
            payload=payload,
            notes=notes,
            argv=tuple(argv),
            backend="cli",
        )
        return value

    async def help(
        self,
        *command: str,
        timeout: float | None = None,
        full: bool = False,
    ) -> str:
        self.last = None
        executable = _bn_executable()
        argv = self._root_argv()
        command_path = tuple(str(part) for part in command)
        aliases = {
            ("search",): ("function", "search"),
            ("function_info",): ("function", "info"),
        }
        argv.extend(aliases.get(command_path, command_path))
        argv.append("--help-full" if full else "--help")
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise BnError(
                f"could not execute bn: {exc}", returncode=127, argv=argv
            ) from exc
        deadline = self.timeout if timeout is None else timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=deadline
            )
        except asyncio.TimeoutError:
            _terminate(process)
            await process.communicate()
            raise BnError(
                _timeout_message(f"bn {' '.join(argv)}", deadline),
                returncode=124,
                argv=argv,
            ) from None

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace").strip()
        if process.returncode:
            error_lines = [
                line.strip()
                for line in stderr_text.splitlines()
                if ": error:" in line or line.lstrip().startswith("error:")
            ]
            message = (
                error_lines[-1]
                if error_lines
                else stderr_text or stdout_text.strip() or "bn failed"
            )
            error_type = _EXIT_ERRORS.get(process.returncode, BnError)
            raise error_type(
                message,
                returncode=process.returncode,
                argv=argv,
            )
        notes = tuple(line for line in stderr_text.splitlines() if line.strip())
        self.last = Result(
            value=stdout_text,
            payload=stdout_text,
            notes=notes,
            argv=tuple(argv),
            backend="cli",
        )
        return stdout_text

    async def all(
        self,
        *args: str,
        page: int = 500,
        limit: int | None = None,
        timeout: float | None = None,
        **flags: Any,
    ) -> list[dict[str, Any]]:
        """Collect a paged CLI read without printing or spilling its rows."""
        self.last = None

        if page < 1:
            raise ValueError("page must be at least 1")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        initial_offset = int(flags.pop("offset", 0))
        offset = initial_offset
        actual_argv: tuple[str, ...] = tuple(str(arg) for arg in args)
        operation = " ".join(str(arg) for arg in args)
        # One end-to-end budget for the WHOLE multi-page collection, with
        # BN_REQUEST_TIMEOUT applied exactly once here. `None` means the documented
        # 0/none/off spelling disabled the deadline entirely.
        effective_timeout = _resolve_budget(
            self.timeout if timeout is None else timeout
        )
        deadline = (
            None
            if effective_timeout is None
            else time.monotonic() + effective_timeout
        )

        if limit == 0:
            # The bridge's `minimum=1` contract stands, so the caller's zero-row
            # "just the schema" request becomes a discarded one-row probe on the
            # wire: request limit=1 at the caller's offset, validate it through
            # the normal page contract, then throw the row away. `has_more`
            # folds in whether the probe found a row at all -- that alone
            # proves more exists at this offset even if the bridge is wrong
            # about `has_more` (the `or bridge_has_more` term is defensive).
            timeout_remaining = None
            if deadline is not None:
                timeout_remaining = deadline - time.monotonic()
                if timeout_remaining <= 0:
                    self.last = None
                    raise BnError(
                        _timeout_message(operation, effective_timeout),
                        returncode=124,
                        argv=actual_argv,
                    )
            try:
                payload = await self._run_resolved(
                    *args,
                    unwrap=False,
                    budget=timeout_remaining,
                    limit=1,
                    offset=offset,
                    **flags,
                )
            except BnError as exc:
                self.last = None
                if exc.returncode == 124:
                    raise BnError(
                        _timeout_message(
                            operation, effective_timeout or 0.0, str(exc)
                        ),
                        returncode=124,
                        argv=actual_argv,
                    ) from exc
                raise
            page_result = self.last
            notes: list[str] = []
            if page_result is not None:
                notes.extend(page_result.notes)
                actual_argv = page_result.argv

            try:
                probed_items, bridge_has_more, total = _validate_page(
                    operation,
                    payload,
                    requested_offset=offset,
                    requested_limit=1,
                )
                if probed_items or "row_fields" in payload:
                    _require_row_fields(operation, payload)
            except BnError:
                self.last = None
                raise

            aggregate = dict(payload)
            aggregate["items"] = []
            aggregate["offset"] = initial_offset
            aggregate["returned"] = 0
            aggregate["total"] = total
            aggregate["has_more"] = bool(probed_items) or bridge_has_more
            aggregate["limit"] = 0
            self.last = Result(
                value=[],
                payload=aggregate,
                notes=tuple(notes),
                argv=actual_argv,
                backend="cli",
            )
            return []

        items: list[dict[str, Any]] = []
        aggregate: dict[str, Any] | None = None
        notes = []
        bridge_has_more = False
        seen_total: Any = _UNSEEN

        while True:
            remaining = None if limit is None else limit - len(items)
            page_size = page if remaining is None else min(page, remaining)
            timeout_remaining = None
            if deadline is not None:
                timeout_remaining = deadline - time.monotonic()
                if timeout_remaining <= 0:
                    self.last = None
                    raise BnError(
                        _timeout_message(operation, effective_timeout),
                        returncode=124,
                        argv=actual_argv,
                    )
            try:
                payload = await self._run_resolved(
                    *args,
                    unwrap=False,
                    budget=timeout_remaining,
                    limit=page_size,
                    offset=offset,
                    **flags,
                )
            except BnError as exc:
                self.last = None
                if exc.returncode == 124:
                    raise BnError(
                        _timeout_message(
                            operation, effective_timeout or 0.0, str(exc)
                        ),
                        returncode=124,
                        argv=actual_argv,
                    ) from exc
                raise
            page_result = self.last
            if page_result is not None:
                notes.extend(page_result.notes)
                actual_argv = page_result.argv

            try:
                page_items, bridge_has_more, total = _validate_page(
                    operation,
                    payload,
                    requested_offset=offset,
                    requested_limit=page_size,
                    previous_total=seen_total,
                )
                seen_total = total
                if (
                    bridge_has_more
                    and limit is None
                    and deadline is None
                    and total is None
                ):
                    raise _unbounded_collection_error(operation)
            except BnError:
                self.last = None
                raise
            if aggregate is None:
                aggregate = dict(payload)

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
        self.last = Result(
            value=items,
            payload=aggregate,
            notes=tuple(notes),
            argv=actual_argv,
            backend="cli",
        )
        return items

    async def _native_request(
        self,
        op: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        self.last = None
        from bn.transport import BridgeError as NativeBridgeError

        client = (
            self._client
            if timeout is None
            else _load_native_client(self.instance, self.target, timeout)
        )
        # One resolved end-to-end budget; None means BN_REQUEST_TIMEOUT disabled it.
        budget = _resolve_budget(self.timeout if timeout is None else timeout)
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(client.request, op, params),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            raise BnError(
                _timeout_message(op, budget or 0.0),
                returncode=124,
                argv=(op,),
            ) from None
        except NativeBridgeError as exc:
            if "timed out" in str(exc).lower():
                raise BnError(
                    _timeout_message(op, budget or 0.0, str(exc)),
                    returncode=124,
                    argv=(op,),
                ) from exc
            raise BridgeError(
                str(exc), returncode=2, argv=(op,)
            ) from exc
        value = _unwrap(payload)
        self.last = Result(value, payload, (), (op,), "native")
        return value

    async def _native_collect(
        self,
        op: str,
        params: dict[str, Any],
        limit: int | None,
        *,
        timeout: float | None = None,
        resolved: bool = False,
    ) -> list[dict[str, Any]]:
        self.last = None
        from bn.transport import BridgeError as NativeBridgeError

        # `resolved=True`: `timeout` is already the remaining slice of one
        # end-to-end budget (search's literal->regex phases share it). Re-resolving
        # it -- here or inside Client.collect -- would give each phase a fresh copy
        # of BN_REQUEST_TIMEOUT and let the pair spend the budget twice.
        budget = (
            timeout
            if resolved
            else _resolve_budget(self.timeout if timeout is None else timeout)
        )

        from bn import Client as NativeClient

        # A None budget means "no deadline" (BN_REQUEST_TIMEOUT disabled), so there
        # is nothing to hand down and the session's own client is right.
        client = self._client
        if budget is not None and isinstance(self._client, NativeClient):
            client = _load_native_client(self.instance, self.target, budget)
        collect_kwargs: dict[str, Any] = {"limit": limit}
        if resolved and budget is not None and isinstance(client, NativeClient):
            collect_kwargs["resolved"] = True
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.collect(op, params, **collect_kwargs)
                ),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            raise BnError(
                _timeout_message(op, budget or 0.0),
                returncode=124,
                argv=(op,),
            ) from None
        except NativeBridgeError as exc:
            if "timed out" in str(exc).lower():
                raise BnError(
                    _timeout_message(op, budget or 0.0, str(exc)),
                    returncode=124,
                    argv=(op,),
                ) from exc
            raise BridgeError(
                str(exc), returncode=2, argv=(op,)
            ) from exc
        value = _require_collection(op, _unwrap(payload), payload)
        self.last = Result(value, payload, (), (op,), "native")
        return value

    def _validated(self, validator, *args, **kwargs):
        try:
            return validator(*args, **kwargs)
        except Exception:
            self.last = None
            raise

    async def info(
        self,
        *,
        verbose: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.last = None
        if self.backend == "native":
            payload = await self._native_request(
                "target_info", {"verbose": verbose}, timeout=timeout
            )
        else:
            payload = await self.run(
                "target", "info", unwrap=False, verbose=verbose, timeout=timeout
            )
        return self._validated(_require_target_info, payload)

    async def assert_target(
        self,
        expected: str | Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.last = None
        info = await self.info(timeout=timeout)
        expected_text = str(expected)
        expected_path = Path(expected_text).expanduser()
        observed_filename = str(info.get("filename") or info.get("file") or "")
        observed_basename = str(
            info.get("basename")
            or (Path(observed_filename).name if observed_filename else "")
        )

        path_like = expected_path.is_absolute() or "/" in expected_text
        if path_like:
            expected_value = os.path.realpath(expected_path)
            observed_value = (
                os.path.realpath(Path(observed_filename))
                if observed_filename
                else ""
            )
            matches = observed_value == expected_value
        else:
            expected_value = expected_path.name
            observed_value = observed_basename
            matches = expected_value in {
                observed_basename,
                Path(observed_basename).stem,
            }

        if not matches:
            # A foreign target is a failure, not the documented contamination
            # exception: leaving its digest in `last` invites a read of another
            # binary's rows. `assert_unannotated` stays the sole exception.
            self.last = None
            raise BridgeError(
                "target identity mismatch: "
                f"expected {expected_value!r}, received {observed_value!r}; "
                "refusing to use data from a different target",
                returncode=2,
                argv=("target_info",),
            )
        return info

    async def assert_unannotated(
        self,
        *,
        allow_contaminated: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.last = None
        if self.backend == "native":
            digest = await self._native_request(
                "orient_digest", {"strings_limit": 1}, timeout=timeout
            )
        else:
            digest = await self.run(
                "evidence",
                "orient",
                unwrap=False,
                strings_limit=1,
                timeout=timeout,
            )
        # Fail CLOSED before reading counters. A digest shape this gate cannot
        # parse must never collapse to "zero comments" and certify contaminated
        # data as clean; `allow_contaminated` waives the contamination policy, not
        # the payload contract.
        counts = self._validated(_require_orient_digest, digest)
        annotations = digest["existing_annotations"]
        if counts["comments"] + counts["function_comments"] and not allow_contaminated:
            argv = self.last.argv if self.last is not None else ("orient_digest",)
            locations = [
                *annotations.get("comment_locations", []),
                *annotations.get("function_comment_locations", []),
            ]
            rendered_locations = ", ".join(
                " ".join(
                    part
                    for part in (
                        str(item.get("address", "")),
                        str(item.get("name", "")),
                    )
                    if part
                )
                for item in locations[:5]
                if isinstance(item, dict)
            )
            location_note = (
                f" First locations: {rendered_locations}."
                if rendered_locations
                else ""
            )
            raise BridgeError(
                "inherited comments detected: "
                f"comments={counts['comments']}, "
                f"function_comments={counts['function_comments']}; "
                "refusing contaminated benchmark data."
                f"{location_note} Pass allow_contaminated=True to inspect the "
                "digest and proceed explicitly. User symbols are reported but "
                "not rejected because raw binaries can legitimately carry them.",
                returncode=2,
                argv=argv,
            )
        return digest

    async def functions(
        self,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.last = None
        params = _validate_filters("functions", filters, _FUNCTION_FILTERS)
        if self.backend == "native":
            value = await self._native_collect(
                "list_functions", params, limit, timeout=timeout
            )
        else:
            value = await self.all(
                "function", "list", limit=limit, timeout=timeout, **params
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(
            _require_function_rows, "list_functions", value, payload
        )

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.last = None
        validated = _validate_filters("search", filters, _SEARCH_FILTERS)
        if self.backend == "native":
            # The two-phase literal->regex retry shares one budget; the CLI branch
            # is single-phase and lets `all()` own its own deadline.
            effective_timeout = _resolve_budget(
                self.timeout if timeout is None else timeout
            )
            params = {"query": query, **validated}
            deadline = (
                None
                if effective_timeout is None
                else time.monotonic() + effective_timeout
            )
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self.last = None
                raise BnError(
                    _timeout_message("search", effective_timeout),
                    returncode=124,
                    argv=("search_functions",),
                )
            value = await self._native_collect(
                "search_functions", params, limit, timeout=remaining, resolved=True
            )
            payload = self.last.payload if self.last is not None else None
            if self._validated(_should_retry_as_regex, query, payload, validated):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self.last = None
                    raise BnError(
                        _timeout_message("search", effective_timeout),
                        returncode=124,
                        argv=("search_functions",),
                    )
                try:
                    value = await self._native_collect(
                        "search_functions",
                        {**params, "regex": True},
                        limit,
                        timeout=remaining,
                        resolved=True,
                    )
                except BnError as exc:
                    self.last = None
                    if exc.returncode == 124:
                        raise BnError(
                            _timeout_message(
                                "search", effective_timeout, str(exc)
                            ),
                            returncode=124,
                            argv=("search_functions",),
                        ) from exc
                    raise
                retry_payload = (
                    dict(self.last.payload)
                    if self.last is not None
                    and isinstance(self.last.payload, dict)
                    else {}
                )
                retry_payload["regex_fallback"] = True
                self.last = Result(
                    value,
                    retry_payload,
                    ("0 literal matches; retried query as a regex",),
                    ("search_functions",),
                    "native",
                )
        else:
            value = await self.all(
                "function",
                "search",
                query,
                limit=limit,
                timeout=timeout,
                **validated,
            )
        payload = self.last.payload if self.last is not None else None
        rows = self._validated(
            _require_function_rows, "search_functions", value, payload
        )
        if (
            isinstance(payload, dict)
            and payload.get("total") == 0
            and "regex_fallback" not in payload
        ):
            normalized = dict(payload)
            normalized["regex_fallback"] = False
            notes = (
                (*self.last.notes, "0 literal matches; no regex fallback attempted")
                if self.last is not None
                else ("0 literal matches; no regex fallback attempted",)
            )
            self.last = Result(
                rows,
                normalized,
                notes,
                self.last.argv if self.last is not None else ("search_functions",),
                self.last.backend if self.last is not None else self.backend,
            )
        return rows

    async def function_info(
        self,
        identifier: str,
        *,
        blocks: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self.backend == "native":
            payload = await self._native_request(
                "function_info",
                {"identifier": identifier, "blocks": blocks},
                timeout=timeout,
            )
        else:
            payload = await self.run(
                "function",
                "info",
                identifier,
                unwrap=False,
                blocks=blocks,
                timeout=timeout,
            )
        value = self._validated(_require_function_info, payload)
        if self.last is not None:
            self.last = Result(
                value,
                self.last.payload,
                self.last.notes,
                self.last.argv,
                self.last.backend,
            )
        return value

    async def decompile(
        self,
        identifier: str,
        *,
        addresses: bool = False,
        force_analysis: bool = False,
        include_annotations: bool = False,
        timeout: float | None = None,
    ) -> str:
        if self.backend == "native":
            value = await self._native_request(
                "decompile",
                {
                    "identifier": identifier,
                    "addresses": addresses,
                    "force_analysis": force_analysis,
                    "include_annotations": include_annotations,
                },
                timeout=timeout,
            )
        else:
            value = await self.run(
                "decompile",
                identifier,
                addresses=addresses,
                force_analysis=force_analysis,
                include_annotations=include_annotations,
                timeout=timeout,
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_decompile, value, payload)

    async def disasm(
        self,
        identifier: str,
        *,
        linear: int | None = None,
        mode: str | None = None,
        snap_to_instruction: bool = False,
        count: int | None = None,
        lines: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> str:
        self.last = None
        if sum(option is not None for option in (linear, count, lines)) > 1:
            raise ValueError("linear, count, and lines are mutually exclusive")
        if count is not None and count < 1:
            raise ValueError("count must be at least 1")
        if lines is not None and (
            len(lines) != 2 or lines[0] < 1 or lines[1] < lines[0]
        ):
            raise ValueError("lines must be a 1-indexed inclusive (start, end) tuple")
        params = {
            "identifier": identifier,
            "linear": linear,
            "mode": mode,
            "snap_to_instruction": snap_to_instruction,
        }
        if count is not None:
            params.update({"line_start": 1, "line_end": count, "strict_range": False})
        elif lines is not None:
            params.update(
                {
                    "line_start": lines[0],
                    "line_end": lines[1],
                    "strict_range": True,
                }
            )
        if self.backend == "native":
            value = await self._native_request("disasm", params, timeout=timeout)
        else:
            value = await self.run(
                "disasm",
                identifier,
                linear=linear,
                mode=mode,
                snap_to_instruction=snap_to_instruction,
                count=count,
                lines=(f"{lines[0]}:{lines[1]}" if lines is not None else None),
                timeout=timeout,
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_text, "disasm", value, payload)

    async def il(
        self,
        identifier: str,
        *,
        view: Literal["hlil", "mlil", "llil"] = "hlil",
        ssa: bool = False,
        timeout: float | None = None,
    ) -> str:
        self.last = None
        params = {"identifier": identifier, "view": view, "ssa": ssa}
        if self.backend == "native":
            value = await self._native_request("il", params, timeout=timeout)
        else:
            value = await self.run(
                "il", identifier, view=view, ssa=ssa, timeout=timeout
            )
        # Validate AFTER the backend branch, like every other curated helper: a
        # validator attached inside one branch is a shape guarantee on one backend
        # only, and `-> str` returning a mapping is exactly the "looks like no
        # findings" shape this module exists to reject.
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_text, "il", value, payload)

    async def xrefs(
        self,
        identifier: str,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        fn_pointer_scan: bool = False,
    ) -> list[dict[str, Any]]:
        self.last = None
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                "xrefs identifier must be a non-empty function name or address"
            )
        params = {
            "identifier": identifier,
            "fn_pointer_scan": fn_pointer_scan,
        }
        if self.backend == "native":
            value = await self._native_collect(
                "xrefs", params, limit, timeout=timeout
            )
        else:
            value = await self.all(
                "xrefs",
                identifier,
                limit=limit,
                timeout=timeout,
                fn_pointer_scan=fn_pointer_scan,
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_collection, "xrefs", value, payload)

    @staticmethod
    def _within_identifiers(within: str | Sequence[str]) -> list[str]:
        identifiers = [within] if isinstance(within, str) else list(within)
        if not identifiers or any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        ):
            raise ValueError("within must contain at least one non-empty identifier")
        return identifiers

    async def callsites(
        self,
        callee: str,
        *,
        within: str | Sequence[str] | None = None,
        context: int = 3,
        limit: int | None = 100,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        self.last = None
        identifiers = [] if within is None else self._within_identifiers(within)
        if self.backend == "native":
            value = await self._native_collect(
                "callsites",
                {
                    "callee": callee,
                    "within_identifiers": identifiers,
                    "context": context,
                },
                limit,
                timeout=timeout,
            )
        elif not identifiers:
            value = await self.all(
                "callsites",
                callee,
                context=context,
                timeout=timeout,
                limit=limit,
            )
        elif len(identifiers) == 1:
            value = await self.all(
                "callsites",
                callee,
                within=identifiers[0],
                context=context,
                timeout=timeout,
                limit=limit,
            )
        else:
            scope_file = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False
            )
            scope_path = Path(scope_file.name)
            try:
                os.chmod(scope_path, 0o600)
                scope_file.write(
                    "".join(f"{identifier}\n" for identifier in identifiers)
                )
                scope_file.close()
                value = await self.all(
                    "callsites",
                    callee,
                    within_file=str(scope_path),
                    context=context,
                    timeout=timeout,
                    limit=limit,
                )
            finally:
                if not scope_file.closed:
                    scope_file.close()
                scope_path.unlink(missing_ok=True)
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_callsites, value, payload)

    async def strings(
        self,
        *,
        limit: int | None = 100,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.last = None
        params = _validate_filters("strings", filters, _STRING_FILTERS)
        if self.backend == "native":
            value = await self._native_collect(
                "strings", params, limit, timeout=timeout
            )
        else:
            value = await self.all(
                "strings", limit=limit, timeout=timeout, **params
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(
            _require_collection,
            "strings",
            value,
            payload,
            required_keys=("value",),
        )

    async def imports(
        self,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.last = None
        params = _validate_filters("imports", filters, _IMPORT_FILTERS)
        if self.backend == "native":
            value = await self._native_collect(
                "imports", params, limit, timeout=timeout
            )
        else:
            value = await self.all(
                "imports", limit=limit, timeout=timeout, **params
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_collection, "imports", value, payload)

    async def sections(
        self,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.last = None
        params = _validate_filters("sections", filters, _SECTION_FILTERS)
        if self.backend == "native":
            value = await self._native_collect(
                "sections", params, limit, timeout=timeout
            )
        else:
            value = await self.all(
                "sections", limit=limit, timeout=timeout, **params
            )
        payload = self.last.payload if self.last is not None else None
        return self._validated(_require_collection, "sections", value, payload)


async def scoped(
    callback: Any,
    *,
    instance: str | None = None,
    target: str | None = None,
    timeout: float = 120.0,
    backend: BackendChoice = "auto",
) -> Any:
    binding = (instance, target)
    callback_binding = getattr(callback, _SCOPED_BINDING_ATTRIBUTE, None)
    if callback_binding is not None and callback_binding != binding:
        raise BnError(
            "scoped callback binding mismatch: this callback was already bound "
            f"to {callback_binding!r}, not {binding!r}; define a fresh async "
            "callback for each target",
            returncode=2,
            argv=("scoped",),
        )
    try:
        setattr(callback, _SCOPED_BINDING_ATTRIBUTE, binding)
    except (AttributeError, TypeError) as exc:
        raise TypeError("scoped callback must be a mutable Python callable") from exc

    callback_id = id(callback)
    registered = False
    try:
        with _SCOPED_CALLBACK_LOCK:
            foreign_active = {
                active_binding
                for active_binding in _ACTIVE_SCOPED_BINDINGS.values()
                if active_binding != binding
            }
            if foreign_active:
                raise BnError(
                    "scoped refused a foreign active scoped binding in this shared "
                    f"interpreter: requested {binding!r}, active "
                    f"{sorted(foreign_active, key=repr)!r}",
                    returncode=2,
                    argv=("scoped",),
                )
            if callback_id in _ACTIVE_SCOPED_CALLBACKS:
                raise BnError(
                    "scoped callback is already active in this shared interpreter",
                    returncode=2,
                    argv=("scoped",),
                )
            # Arm cleanup before either registry mutation. An asynchronous
            # cancellation injected between the add/set bytecodes must not leave
            # a foreign-active binding behind for the process lifetime.
            registered = True
            _ACTIVE_SCOPED_CALLBACKS.add(callback_id)
            _ACTIVE_SCOPED_BINDINGS[callback_id] = binding

        bound = Session(instance, target, timeout=timeout, backend=backend)
        if (bound.instance, bound.target) != binding:
            raise BnError(
                "scoped Session binding changed before callback execution",
                returncode=2,
                argv=("scoped",),
            )
        value = callback(bound)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, Session):
            raise BnError(
                "scoped callback must not return its Session into shared globals",
                returncode=2,
                argv=("scoped",),
            )
        return value
    finally:
        if registered:
            with _SCOPED_CALLBACK_LOCK:
                _ACTIVE_SCOPED_CALLBACKS.discard(callback_id)
                _ACTIVE_SCOPED_BINDINGS.pop(callback_id, None)


def session(
    instance: str | None = None,
    target: str | None = None,
    *,
    timeout: float = 120.0,
    backend: BackendChoice = "auto",
) -> Session:
    return Session(instance, target, timeout=timeout, backend=backend)


async def run(
    *args: str,
    instance: str | None = None,
    target: str | None = None,
    **kwargs: Any,
) -> Any:
    return await Session(instance, target).run(*args, **kwargs)


def _missing_key_message(row: Mapping[str, Any], key: str, index: int) -> str:
    """Name what the row ACTUALLY has when a `brief()` key misses.

    Collection row schemas differ on purpose (functions key on address/size,
    sections on start/end/length, callsites nest callee/containing_function), so
    a bare KeyError plus generic dotted-path advice sent agents back to re-run a
    read -- or to the source -- just to learn the key names. List them.

    Dotted-path guidance is offered only when this row actually nests, and names
    the nesting keys instead of an unrelated example.
    """
    available = ", ".join(repr(name) for name in row) or "(row has no keys)"
    message = (
        f"brief key {key!r} is missing at row {index}; "
        f"available keys: {available}"
    )
    nested = [name for name, value in row.items() if isinstance(value, Mapping)]
    if nested:
        paths = ", ".join(f"{name}.*" for name in nested)
        message += f"; use dotted paths for the nested mappings: {paths}"
    return message


def brief(
    rows: Sequence[Mapping[str, Any]], *keys: str, n: int = 10
) -> str:
    if n < 0:
        raise ValueError("n must be non-negative")
    if (
        isinstance(rows, (str, bytes, Mapping))
        or not isinstance(rows, Sequence)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise TypeError(
            "brief expects a sequence of row mappings; for a paged payload pass "
            "payload['items'], and do not pass plain text"
        )
    if not rows:
        return "(0 rows)"

    selected_keys = keys or tuple(rows[0].keys())[:4]

    def value_at(row: Mapping[str, Any], key: str, index: int) -> Any:
        if key in row:
            return row[key]
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(_missing_key_message(row, key, index))
            value = value[part]
        return value

    lines = [
        "  ".join(
            str(value_at(row, key, index))
            for key in selected_keys
        )
        for index, row in enumerate(rows[:n])
    ]
    remaining = len(rows) - min(n, len(rows))
    if remaining:
        lines.append(f"... {remaining} more of {len(rows)}")
    return "\n".join(lines)
