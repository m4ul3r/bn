"""Retained-kernel access to bn with native and zero-setup CLI backends."""

from __future__ import annotations

import asyncio
import json
import inspect
import re
import os
import shutil
import tempfile
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
_REGEX_METACHARS = "|()[]{}*+?^$\\"


def _should_retry_as_regex(
    query: str,
    payload: Any,
    filters: Mapping[str, Any],
) -> bool:
    if filters.get("regex") or filters.get("exact"):
        return False
    if not isinstance(payload, dict) or payload.get("total") != 0:
        return False
    if not any(character in query for character in _REGEX_METACHARS):
        return False
    try:
        re.compile(query)
    except re.error:
        return False
    return True


def _flatten_function_info(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("function"), dict):
        return payload
    return {
        **{key: value for key, value in payload.items() if key != "function"},
        **payload["function"],
    }
_ACTIVE_SESSIONS = weakref.WeakSet()
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
    if distinct:
        pair = frozenset((binding, distinct[0]))
        if pair not in _WARNED_BINDING_PAIRS:
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
        timeout: float = 600.0,
        backend: BackendChoice = "auto",
    ) -> None:
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

        executable = _bn_executable()
        argv = self._root_argv()
        argv.extend(str(arg) for arg in args)
        argv.extend(_flags(flags))
        output_format = "text" if raw else "json"

        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        output_path = Path(temporary.name)
        temporary.close()
        os.chmod(output_path, 0o600)

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
                    f"bn {' '.join(argv)} timed out after {deadline}s",
                    returncode=124,
                    argv=argv,
                ) from None

            stdout_text = stdout_bytes.decode(errors="replace").strip()
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            if process.returncode:
                error_type = _EXIT_ERRORS.get(process.returncode, BnError)
                raise error_type(
                    stderr_text or stdout_text or "bn failed",
                    returncode=process.returncode,
                    argv=argv,
                )
            body = output_path.read_text(encoding="utf-8", errors="replace")
        finally:
            output_path.unlink(missing_ok=True)

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
    ) -> str:
        executable = _bn_executable()
        argv = self._root_argv()
        argv.extend(str(part) for part in command)
        argv.append("--help-full")
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
                f"bn {' '.join(argv)} timed out after {deadline}s",
                returncode=124,
                argv=argv,
            ) from None

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace").strip()
        if process.returncode:
            error_type = _EXIT_ERRORS.get(process.returncode, BnError)
            raise error_type(
                stderr_text or stdout_text.strip() or "bn failed",
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
        **flags: Any,
    ) -> list[dict[str, Any]]:
        """Collect a paged CLI read without printing or spilling its rows."""

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

        while True:
            remaining = None if limit is None else limit - len(items)
            page_size = page if remaining is None else min(page, remaining)
            payload = await self.run(
                *args,
                unwrap=False,
                limit=page_size,
                offset=offset,
                **flags,
            )
            page_result = self.last
            if page_result is not None:
                notes.extend(page_result.notes)
                actual_argv = page_result.argv

            if not isinstance(payload, dict):
                raise BnError(
                    f"bn {operation} is not a paged collection",
                    returncode=0,
                    argv=actual_argv,
                )
            page_items = payload.get("items")
            bridge_has_more = payload.get("has_more")
            if not isinstance(page_items, list):
                raise BnError(
                    f"malformed {operation} page: items must be a list",
                    returncode=0,
                    argv=actual_argv,
                )
            if not isinstance(bridge_has_more, bool):
                raise BnError(
                    f"malformed {operation} page: has_more must be a boolean",
                    returncode=0,
                    argv=actual_argv,
                )
            if bridge_has_more and not page_items:
                raise BnError(
                    f"malformed {operation} page: has_more with an empty page",
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
        self.last = Result(
            value=items,
            payload=aggregate,
            notes=tuple(notes),
            argv=actual_argv,
            backend="cli",
        )
        return items

    async def _native_request(self, op: str, params: dict[str, Any]) -> Any:
        from bn.transport import BridgeError as NativeBridgeError

        try:
            payload = await asyncio.to_thread(self._client.request, op, params)
        except NativeBridgeError as exc:
            raise BridgeError(
                str(exc), returncode=2, argv=(op,)
            ) from exc
        value = _unwrap(payload)
        self.last = Result(value, payload, (), (op,), "native")
        return value

    async def _native_collect(
        self, op: str, params: dict[str, Any], limit: int | None
    ) -> list[dict[str, Any]]:
        from bn.transport import BridgeError as NativeBridgeError

        try:
            payload = await asyncio.to_thread(
                self._client.collect, op, params, limit=limit
            )
        except NativeBridgeError as exc:
            raise BridgeError(
                str(exc), returncode=2, argv=(op,)
            ) from exc
        value = _unwrap(payload)
        self.last = Result(value, payload, (), (op,), "native")
        return value

    async def info(self, *, verbose: bool = False) -> dict[str, Any]:
        if self.backend == "native":
            return await self._native_request("target_info", {"verbose": verbose})
        return await self.run("target", "info", unwrap=False, verbose=verbose)

    async def assert_target(self, expected: str | Path) -> dict[str, Any]:
        info = await self.info()
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
        else:
            expected_value = expected_path.name
            observed_value = observed_basename

        if observed_value != expected_value:
            raise BridgeError(
                "target identity mismatch: "
                f"expected {expected_value!r}, received {observed_value!r}; "
                "refusing to use data from a different target",
                returncode=2,
                argv=("target_info",),
            )
        return info

    async def assert_unannotated(self) -> dict[str, Any]:
        if self.backend == "native":
            digest = await self._native_request(
                "orient_digest", {"strings_limit": 1}
            )
        else:
            digest = await self.run(
                "evidence",
                "orient",
                unwrap=False,
                strings_limit=1,
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
        if counts["comments"] + counts["function_comments"]:
            argv = self.last.argv if self.last is not None else ("orient_digest",)
            raise BridgeError(
                "inherited comments detected: "
                f"comments={counts['comments']}, "
                f"function_comments={counts['function_comments']}; "
                "refusing contaminated benchmark data. "
                "User symbols are reported but not rejected because raw binaries "
                "can legitimately carry them.",
                returncode=2,
                argv=argv,
            )
        return digest

    async def functions(
        self, *, limit: int | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        if self.backend == "native":
            return await self._native_collect("list_functions", dict(filters), limit)
        return await self.all("function", "list", limit=limit, **filters)

    async def search(
        self, query: str, *, limit: int | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        if self.backend == "native":
            params = {"query": query, **filters}
            value = await self._native_collect(
                "search_functions", params, limit
            )
            payload = self.last.payload if self.last is not None else None
            if _should_retry_as_regex(query, payload, filters):
                value = await self._native_collect(
                    "search_functions", {**params, "regex": True}, limit
                )
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
            return value
        return await self.all("function", "search", query, limit=limit, **filters)

    async def function_info(
        self, identifier: str, *, blocks: bool = False
    ) -> dict[str, Any]:
        if self.backend == "native":
            payload = await self._native_request(
                "function_info", {"identifier": identifier, "blocks": blocks}
            )
        else:
            payload = await self.run(
                "function", "info", identifier, unwrap=False, blocks=blocks
            )
        value = _flatten_function_info(payload)
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
    ) -> str:
        if self.backend == "native":
            return await self._native_request(
                "decompile",
                {
                    "identifier": identifier,
                    "addresses": addresses,
                    "force_analysis": force_analysis,
                },
            )
        return await self.run(
            "decompile",
            identifier,
            addresses=addresses,
            force_analysis=force_analysis,
        )

    async def disasm(
        self,
        identifier: str,
        *,
        linear: int | None = None,
        mode: str | None = None,
        snap_to_instruction: bool = False,
    ) -> str:
        params = {
            "identifier": identifier,
            "linear": linear,
            "mode": mode,
            "snap_to_instruction": snap_to_instruction,
        }
        if self.backend == "native":
            return await self._native_request("disasm", params)
        return await self.run(
            "disasm",
            identifier,
            linear=linear,
            mode=mode,
            snap_to_instruction=snap_to_instruction,
        )

    async def il(
        self,
        identifier: str,
        *,
        view: Literal["hlil", "mlil", "llil"] = "hlil",
        ssa: bool = False,
    ) -> str:
        params = {"identifier": identifier, "view": view, "ssa": ssa}
        if self.backend == "native":
            return await self._native_request("il", params)
        return await self.run("il", identifier, view=view, ssa=ssa)

    async def xrefs(
        self,
        identifier: str,
        *,
        limit: int | None = None,
        fn_pointer_scan: bool = False,
    ) -> list[dict[str, Any]]:
        params = {
            "identifier": identifier,
            "fn_pointer_scan": fn_pointer_scan,
        }
        if self.backend == "native":
            return await self._native_collect("xrefs", params, limit)
        return await self.all(
            "xrefs",
            identifier,
            limit=limit,
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
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        identifiers = [] if within is None else self._within_identifiers(within)
        if self.backend == "native":
            return await self._native_collect(
                "callsites",
                {
                    "callee": callee,
                    "within_identifiers": identifiers,
                    "context": context,
                },
                limit,
            )
        if not identifiers:
            return await self.all(
                "callsites",
                callee,
                context=context,
                limit=limit,
            )
        if len(identifiers) == 1:
            return await self.all(
                "callsites",
                callee,
                within=identifiers[0],
                context=context,
                limit=limit,
            )

        scope_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        scope_path = Path(scope_file.name)
        try:
            os.chmod(scope_path, 0o600)
            scope_file.write("".join(f"{identifier}\n" for identifier in identifiers))
            scope_file.close()
            return await self.all(
                "callsites",
                callee,
                within_file=str(scope_path),
                context=context,
                limit=limit,
            )
        finally:
            if not scope_file.closed:
                scope_file.close()
            scope_path.unlink(missing_ok=True)

    async def strings(
        self, *, limit: int | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        if self.backend == "native":
            return await self._native_collect("strings", dict(filters), limit)
        return await self.all("strings", limit=limit, **filters)

    async def imports(
        self, *, limit: int | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        if self.backend == "native":
            return await self._native_collect("imports", dict(filters), limit)
        return await self.all("imports", limit=limit, **filters)

    async def sections(
        self, *, limit: int | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        if self.backend == "native":
            return await self._native_collect("sections", dict(filters), limit)
        return await self.all("sections", limit=limit, **filters)


async def scoped(
    callback: Any,
    *,
    instance: str | None = None,
    target: str | None = None,
    timeout: float = 600.0,
    backend: BackendChoice = "auto",
) -> Any:
    bound = Session(instance, target, timeout=timeout, backend=backend)
    value = callback(bound)
    if inspect.isawaitable(value):
        return await value
    return value


def session(
    instance: str | None = None,
    target: str | None = None,
    *,
    timeout: float = 600.0,
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
    if not rows:
        return "(0 rows)"

    selected_keys = keys or tuple(rows[0].keys())[:4]
    lines = [
        "  ".join(str(row.get(key, "")) for key in selected_keys)
        for row in rows[:n]
    ]
    remaining = len(rows) - min(n, len(rows))
    if remaining:
        lines.append(f"... {remaining} more of {len(rows)}")
    return "\n".join(lines)
