"""Retained-kernel access to bn with native and zero-setup CLI backends."""

from __future__ import annotations

import asyncio
import json
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
    if distinct and not _WARNED_BINDING_PAIRS:
        pair = frozenset((binding, distinct[0]))
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
                process.kill()
                await process.communicate()
                raise BnError(
                    _timeout_message(f"bn {' '.join(argv)}", deadline),
                    returncode=124,
                    argv=argv,
                ) from None
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.kill()
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
            process.kill()
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
        if limit == 0:
            payload = {
                "items": [],
                "offset": initial_offset,
                "returned": 0,
                "has_more": False,
                "total": None,
            }
            self.last = Result(
                value=[],
                payload=payload,
                notes=(),
                argv=tuple(self._root_argv() + [str(arg) for arg in args]),
                backend="cli",
            )
            return []

        offset = initial_offset
        items: list[dict[str, Any]] = []
        aggregate: dict[str, Any] | None = None
        notes: list[str] = []
        actual_argv: tuple[str, ...] = tuple(str(arg) for arg in args)
        bridge_has_more = False
        operation = " ".join(str(arg) for arg in args)
        effective_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout

        while True:
            remaining = None if limit is None else limit - len(items)
            page_size = page if remaining is None else min(page, remaining)
            timeout_remaining = deadline - time.monotonic()
            if timeout_remaining <= 0:
                self.last = None
                raise BnError(
                    f"bn {operation} timed out after {effective_timeout:g}s",
                    returncode=124,
                    argv=actual_argv,
                )
            try:
                payload = await self.run(
                    *args,
                    unwrap=False,
                    timeout=timeout_remaining,
                    limit=page_size,
                    offset=offset,
                    **flags,
                )
            except BnError as exc:
                self.last = None
                if exc.returncode == 124:
                    raise BnError(
                        _timeout_message(operation, effective_timeout, str(exc)),
                        returncode=124,
                        argv=actual_argv,
                    ) from exc
                raise
            page_result = self.last
            if page_result is not None:
                notes.extend(page_result.notes)
                actual_argv = page_result.argv

            if not isinstance(payload, dict):
                self.last = None
                raise BnError(
                    f"bn {operation} is not a paged collection",
                    returncode=0,
                    argv=actual_argv,
                )
            page_items = payload.get("items")
            bridge_has_more = payload.get("has_more")
            if not isinstance(page_items, list):
                self.last = None
                raise BnError(
                    f"malformed {operation} page: items must be a list",
                    returncode=0,
                    argv=actual_argv,
                )
            if not isinstance(bridge_has_more, bool):
                self.last = None
                raise BnError(
                    f"malformed {operation} page: has_more must be a boolean",
                    returncode=0,
                    argv=actual_argv,
                )
            if bridge_has_more and not page_items:
                self.last = None
                raise BnError(
                    f"malformed {operation} page: has_more with an empty page",
                    returncode=0,
                    argv=actual_argv,
                )
            returned = payload.get("returned")
            if returned is not None and (
                not isinstance(returned, int)
                or isinstance(returned, bool)
                or returned != len(page_items)
            ):
                self.last = None
                raise BnError(
                    f"malformed {operation} page: returned must equal items length",
                    returncode=0,
                    argv=actual_argv,
                )
            total = payload.get("total")
            if total is not None:
                page_end = offset + len(page_items)
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total < 0
                    or (bool(page_items) and page_end > total)
                ):
                    self.last = None
                    raise BnError(
                        f"malformed {operation} page: invalid total {total!r}",
                        returncode=0,
                        argv=actual_argv,
                    )
                if (
                    aggregate is not None
                    and aggregate.get("total") is not None
                    and aggregate.get("total") != total
                ):
                    self.last = None
                    raise BnError(
                        f"malformed {operation} page: total changed across pages",
                        returncode=0,
                        argv=actual_argv,
                    )
                if not bridge_has_more and page_end < total:
                    self.last = None
                    raise BnError(
                        f"malformed {operation} page: total={total} requires "
                        "has_more=true at this offset",
                        returncode=0,
                        argv=actual_argv,
                    )
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
        budget = self.timeout if timeout is None else timeout
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(client.request, op, params),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            raise BnError(
                _timeout_message(op, budget),
                returncode=124,
                argv=(op,),
            ) from None
        except NativeBridgeError as exc:
            if "timed out" in str(exc).lower():
                raise BnError(
                    _timeout_message(op, budget, str(exc)),
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
    ) -> list[dict[str, Any]]:
        self.last = None
        from bn.transport import BridgeError as NativeBridgeError

        if timeout is None:
            client = self._client
        else:
            from bn import Client as NativeClient

            client = (
                _load_native_client(self.instance, self.target, timeout)
                if isinstance(self._client, NativeClient)
                else self._client
            )
        budget = self.timeout if timeout is None else timeout
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(
                    client.collect, op, params, limit=limit
                ),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            raise BnError(
                _timeout_message(op, budget),
                returncode=124,
                argv=(op,),
            ) from None
        except NativeBridgeError as exc:
            if "timed out" in str(exc).lower():
                raise BnError(
                    _timeout_message(op, budget, str(exc)),
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
        if self.backend == "native":
            return await self._native_request(
                "target_info", {"verbose": verbose}, timeout=timeout
            )
        return await self.run(
            "target", "info", unwrap=False, verbose=verbose, timeout=timeout
        )

    async def assert_target(
        self,
        expected: str | Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
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
        annotations = (
            digest.get("existing_annotations", {})
            if isinstance(digest, dict)
            else {}
        )
        counts = {
            key: int(annotations.get(key, 0) or 0)
            for key in ("comments", "function_comments", "user_symbols")
        }
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
        validated = _validate_filters("search", filters, _SEARCH_FILTERS)
        effective_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout
        if self.backend == "native":
            params = {"query": query, **validated}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.last = None
                raise BnError(
                    _timeout_message("search", effective_timeout),
                    returncode=124,
                    argv=("search_functions",),
                )
            value = await self._native_collect(
                "search_functions", params, limit, timeout=remaining
            )
            payload = self.last.payload if self.last is not None else None
            if self._validated(_should_retry_as_regex, query, payload, validated):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
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
        params = {"identifier": identifier, "view": view, "ssa": ssa}
        if self.backend == "native":
            return await self._native_request("il", params, timeout=timeout)
        return await self.run(
            "il", identifier, view=view, ssa=ssa, timeout=timeout
        )

    async def xrefs(
        self,
        identifier: str,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        fn_pointer_scan: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                "xrefs identifier must be a non-empty function name or address"
            )
        params = {
            "identifier": identifier,
            "fn_pointer_scan": fn_pointer_scan,
        }
        if self.backend == "native":
            return await self._native_collect(
                "xrefs", params, limit, timeout=timeout
            )
        return await self.all(
            "xrefs",
            identifier,
            limit=limit,
            timeout=timeout,
            fn_pointer_scan=fn_pointer_scan,
        )

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
        params = _validate_filters("imports", filters, _IMPORT_FILTERS)
        if self.backend == "native":
            return await self._native_collect(
                "imports", params, limit, timeout=timeout
            )
        return await self.all(
            "imports", limit=limit, timeout=timeout, **params
        )

    async def sections(
        self,
        *,
        limit: int | None = None,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        params = _validate_filters("sections", filters, _SECTION_FILTERS)
        if self.backend == "native":
            return await self._native_collect(
                "sections", params, limit, timeout=timeout
            )
        return await self.all(
            "sections", limit=limit, timeout=timeout, **params
        )


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
                raise KeyError(
                    f"brief key {key!r} is missing at row {index}; "
                    "use dotted paths such as 'callee.name' for nested fields"
                )
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
