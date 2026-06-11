from __future__ import annotations

import atexit
import contextlib
import difflib
import errno
import hashlib
import io
import json
import os
import re
import socket
import socketserver
import tempfile
import threading
import traceback
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import binaryninja as bn
from binaryninja import SSAVariable
from binaryninja.mainthread import execute_on_main_thread_and_wait, is_main_thread
from binaryninja.plugin import PluginCommand

from . import taint_engine as _taint
from .paths import PLUGIN_NAME, bridge_registry_path, bridge_socket_path, instances_dir
from .version import VERSION, build_id_for_file

try:
    import binaryninjaui as ui
except Exception:  # ImportError or UIPluginInHeadlessError
    ui = None


PLUGIN_BUILD_ID = build_id_for_file(Path(__file__).resolve())

# Upper bound on a single newline-terminated JSON request. Anything larger is
# rejected with a clean error instead of being buffered without limit.
MAX_REQUEST_BYTES = 32 * 1024 * 1024


def _json_response(*, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "result": result, "error": error}


def _format_ambiguous_function_error(identifier: Any, matches: list[Any]) -> str:
    lines = [f"Ambiguous function identifier: {identifier} matches {len(matches)} functions:"]
    for fn in sorted(matches, key=lambda f: int(f.start)):
        lines.append(f"  {int(fn.start):#010x}  {str(fn.name)}")
    lines.append("retry with one of the addresses above (e.g. `bn function info 0x…`)")
    return "\n".join(lines)


def _format_ambiguous_symbol_error(identifier: Any, matches: list[Any]) -> str:
    lines = [f"Ambiguous symbol identifier: {identifier} matches {len(matches)} symbols:"]
    for sym in sorted(matches, key=lambda s: int(s.address)):
        kind = getattr(getattr(sym, "type", None), "name", "") or str(getattr(sym, "type", ""))
        lines.append(f"  {int(sym.address):#010x}  {str(sym.name)}  [{kind}]")
    lines.append("retry with one of the addresses above")
    return "\n".join(lines)


def _format_unknown_target_error(selector: Any, targets: list[dict[str, Any]]) -> str:
    lines = [f"Unknown target selector: {selector}"]
    if not targets:
        lines.append("No BinaryView targets are open.")
        return "\n".join(lines)
    lines.append("Open targets:")
    for target in targets:
        marker = "*" if target.get("active") else " "
        lines.append(
            f"  {marker} {target.get('selector', '')}"
            f"  view_id={target.get('view_id', '')}"
            f"  target_id={target.get('target_id', '')}"
            f"  {target.get('filename', '')}"
        )
    lines.append("note: view_id / target_id are stable across `bn save`")
    return "\n".join(lines)


def _serialize_error(exc: BaseException) -> str:
    """Render an exception for the user-facing error field.

    User-facing errors (``OperationFailure`` and the plain ``RuntimeError`` /
    ``ValueError`` instances raised throughout the bridge to report bad input or
    missing targets) carry an already-actionable message, so we emit ``str(exc)``
    verbatim. Anything outside that whitelist is treated as an unexpected bug and
    prefixed with ``internal error:`` plus its class name so callers can tell the
    difference without us leaking raw Python class names into normal errors.
    """

    # OperationFailure subclasses RuntimeError, so it is already covered by the
    # tuple below; it is listed explicitly to document the intent.
    if isinstance(exc, USER_FACING_ERRORS):
        return str(exc)
    return f"internal error: {type(exc).__name__}: {exc}"


class OperationFailure(RuntimeError):
    def __init__(
        self,
        status: str,
        message: str,
        *,
        requested: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.requested = requested or {}
        self.observed = observed or {}


# Exception classes whose messages are already user-facing and actionable.
# OperationFailure is a RuntimeError subclass; RuntimeError covers the bulk of
# "Function not found"/"Type not found"/"Unknown target selector" style errors,
# and ValueError covers argument-validation errors like "Unknown operation".
USER_FACING_ERRORS: tuple[type[BaseException], ...] = (OperationFailure, RuntimeError, ValueError)


# Required request fields per mutation op kind, validated before dispatch so a
# missing field is reported as a malformed REQUEST (invalid_request, naming the
# field) rather than letting a raw KeyError surface -- and so a KeyError raised
# by BN internals inside a handler is NOT mislabeled as a missing request field.
# Optional fields (read via op.get(...)) are intentionally omitted.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "rename_symbol": ("identifier", "new_name"),
    "set_comment": ("comment",),
    "delete_comment": (),
    "set_prototype": ("identifier", "prototype"),
    "local_rename": ("function", "variable", "new_name"),
    "local_retype": ("function", "variable", "new_type"),
    "struct_field_set": ("struct_name", "field_type", "offset", "field_name"),
    "struct_field_rename": ("struct_name", "old_name", "new_name"),
    "struct_field_delete": ("struct_name", "field_name"),
    "types_declare": ("declaration",),
}

# Ops that accept one of several alternative locator fields. set_comment /
# delete_comment target EITHER a function (`function`) OR an address (`address`):
# listing `address` as unconditionally required wrongly rejected the documented
# function-only form (#67). Each group requires at least one of its fields.
REQUIRED_ONE_OF: dict[str, tuple[tuple[str, ...], ...]] = {
    "set_comment": (("function", "address"),),
    "delete_comment": (("function", "address"),),
}


def _validate_count(value: Any, *, label: str, minimum: int, allow_none: bool = False) -> int | None:
    """Coerce a limit/offset param to int and enforce *minimum*.

    The CLI argparse layer already rejects out-of-range count/limit/offset
    flags, but non-CLI callers (``bn py exec``, a raw socket client) reach the
    op handlers directly, so the bridge re-enforces the same contract: a
    negative limit must not silently drop the tail via Python slice math, and a
    zero limit must not return a degenerate empty-but-"truncated" result.
    """
    if value is None:
        if allow_none:
            return None
        raise OperationFailure("invalid_request", f"{label} is required")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise OperationFailure("invalid_request", f"{label} must be an integer, got {value!r}")
    if n < minimum:
        raise OperationFailure("invalid_request", f"{label} must be >= {minimum}, got {n}")
    return n


class _ReadWriteLock:
    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextlib.contextmanager
    def read(self):
        with self._condition:
            # Also wait while writers are queued, otherwise a steady stream of
            # readers starves a waiting writer forever.
            while self._writer or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextlib.contextmanager
    def write(self):
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


READ_LOCKED_OPS = {
    # These three read live BinaryViews (targets.refresh() dereferences each
    # view's file/session), so they must exclude write-locked close_binary /
    # load_binary. "shutdown" stays unlocked on purpose: it only sets an event
    # and must work even while a write op is wedged.
    "doctor",
    "list_targets",
    "target_info",
    "function_info",
    "get_prototype",
    "list_functions",
    "list_locals",
    "search_functions",
    "callsites",
    "decompile",
    "il",
    "structured_il",
    "defuse",
    "resolved_calls",
    "possible_values",
    "taint",
    "disasm",
    "function_evidence",
    "xrefs",
    "field_xrefs",
    "pointer_table",
    "message_lens",
    "init_arrays",
    "backward_slice",
    "types",
    "type_info",
    "strings",
    "imports",
    "bundle_function",
    "get_comment",
    "list_comments",
    "sections",
    "read",
}


WRITE_LOCKED_OPS = {
    "py_exec",
    "function_create",
    "rename_symbol",
    "set_comment",
    "delete_comment",
    "set_prototype",
    "local_rename",
    "local_retype",
    "struct_field_set",
    "struct_field_rename",
    "struct_field_delete",
    "types_declare",
    "batch_apply",
    "refresh",
    "load_binary",
    "close_binary",
    "save_database",
}


def _run_on_main_thread(func):
    if is_main_thread():
        return func()

    holder: dict[str, Any] = {}

    def wrapper():
        try:
            holder["result"] = func()
        except Exception as exc:  # pragma: no cover - exercised inside GUI
            holder["error"] = exc
            holder["traceback"] = traceback.format_exc()

    execute_on_main_thread_and_wait(wrapper)
    if "error" in holder:
        exc = holder["error"]
        if "traceback" in holder:
            bn.log_error(holder["traceback"])
        raise exc
    return holder.get("result")


_CONVENTION_RE = re.compile(r'\s*__convention\("[^"]*"\)\s*')


def _normalize_prototype(proto: str) -> str:
    """Strip ``__convention("...")`` annotations and normalize whitespace for comparison."""
    return " ".join(_CONVENTION_RE.sub("", proto).split())


def _parse_address(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    except ValueError:
        raise ValueError(
            f"{value!r} is not a valid address; expected a decimal or 0x-prefixed hex value"
        ) from None


def _artifact_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"kind": "object", "keys": sorted(value.keys())[:10], "count": len(value)}
    if isinstance(value, list):
        return {"kind": "array", "count": len(value)}
    if isinstance(value, str):
        return {"kind": "string", "chars": len(value)}
    return {"kind": type(value).__name__}


def _write_json_artifact(path_text: str | None, payload: Any) -> dict[str, Any] | None:
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return {
        "ok": True,
        "artifact_path": str(path),
        "format": "json",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "summary": _artifact_summary(payload),
    }


def _active_binary_view():
    if ui is not None:
        def resolve():
            try:
                context = ui.UIContext.activeContext()
                if context is not None:
                    frame = context.getCurrentViewFrame()
                    view = frame.getCurrentBinaryView() if frame is not None else None
                    if view is not None:
                        return view

                contexts = list(ui.UIContext.allContexts())
                if len(contexts) == 1:
                    frame = contexts[0].getCurrentViewFrame()
                    return frame.getCurrentBinaryView() if frame is not None else None
            except Exception:
                return None
            return None

        return _run_on_main_thread(resolve)

    with _headless_views_lock:
        if len(_headless_views) == 1:
            return _headless_views[0]
    return None


def _collect_open_views() -> list[Any]:
    if ui is None:
        with _headless_views_lock:
            return list(_headless_views)

    def collect():
        found: list[Any] = []
        try:
            contexts = list(ui.UIContext.allContexts())
        except Exception:
            contexts = []
        if not contexts:
            active_context = ui.UIContext.activeContext()
            if active_context is not None:
                contexts = [active_context]

        def collect_binary_view(view):
            if view is not None:
                found.append(view)

        def collect_from_frame(frame):
            if frame is None:
                return
            collect_binary_view(frame.getCurrentBinaryView())

        def collect_from_tab(context, tab):
            try:
                collect_from_frame(context.getViewFrameForTab(tab))
            except Exception:
                pass
            try:
                view = context.getViewForTab(tab)
                collect_binary_view(view.getData() if view is not None else None)
            except Exception:
                pass

        for context in contexts:
            try:
                collect_from_frame(context.getCurrentViewFrame())
            except Exception:
                pass
            try:
                tabs = list(context.getTabs())
            except Exception:
                tabs = []
            for tab in tabs:
                collect_from_tab(context, tab)

        unique: list[Any] = []
        seen: set[int] = set()
        for bv in found:
            marker = id(bv)
            if marker not in seen:
                seen.add(marker)
                unique.append(bv)
        return unique

    return _run_on_main_thread(collect)


def _path_components(path: str) -> tuple[str, ...]:
    if not path:
        return ()
    return tuple(p for p in path.split(os.sep) if p)


@dataclass(slots=True)
class TargetRecord:
    view_id: str
    ref: weakref.ReferenceType
    session_id: str
    filename: str
    basename: str
    view_name: str

    def target_id(self) -> str:
        return f"{os.getpid()}:{self.view_id}:{self.session_id}"


class TargetManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._records: dict[str, TargetRecord] = {}
        # id(bv) -> (weakref to that exact bv, stable view_id). The weakref is
        # validated on lookup because CPython recycles addresses: a new view can
        # otherwise inherit a dead view's stable view_id.
        self._ids_by_object: dict[int, tuple[weakref.ref, str]] = {}
        self._next_id = 1

    def _view_name(self, bv) -> str:
        for attr in ("view_type", "name"):
            try:
                value = getattr(bv, attr, None)
                if value:
                    return str(getattr(value, "name", value))
            except Exception:
                continue
        return type(bv).__name__

    def _compute_selectors(self, records: dict[str, TargetRecord]) -> dict[str, str]:
        components = {vid: _path_components(r.filename) for vid, r in records.items()}
        selectors: dict[str, str] = {}
        for vid, record in records.items():
            my_parts = components[vid]
            if not my_parts:
                selectors[vid] = record.target_id()
                continue
            chosen: str | None = None
            for depth in range(1, len(my_parts) + 1):
                suffix = my_parts[-depth:]
                if not any(
                    other_parts[-depth:] == suffix
                    for other_vid, other_parts in components.items()
                    if other_vid != vid and len(other_parts) >= depth
                ):
                    chosen = os.sep.join(suffix)
                    break
            selectors[vid] = chosen or record.target_id()
        return selectors

    def _matches_record(self, record: TargetRecord, selector: str | None) -> bool:
        if selector is None:
            return False
        candidate = str(selector).strip()
        if candidate in ("", "active"):
            return False
        if candidate in (
            record.target_id(),
            record.view_id,
            record.filename,
            record.basename,
        ):
            return True
        suffix = _path_components(candidate)
        if suffix:
            parts = _path_components(record.filename)
            if len(parts) >= len(suffix) and parts[-len(suffix):] == suffix:
                return True
        return False

    def matches_target(self, target_id: str, selector: str | None) -> bool:
        """Whether ``selector`` names the record with ``target_id``. Locked."""
        with self._lock:
            for record in self._records.values():
                if record.target_id() == target_id:
                    return self._matches_record(record, selector)
        return False

    def _default_view(self):
        active = _active_binary_view()
        if active is not None:
            return active

        with self._lock:
            live_views = [record.ref() for record in self._records.values()]
        live_views = [view for view in live_views if view is not None]
        if len(live_views) == 1:
            return live_views[0]
        return None

    def refresh(self) -> list[dict[str, Any]]:
        views = _collect_open_views()
        focused = _active_binary_view()

        with self._lock:
            # Prune entries whose referent is gone so the map cannot grow
            # without bound across many load/close cycles.
            self._ids_by_object = {
                key: (ref, vid)
                for key, (ref, vid) in self._ids_by_object.items()
                if ref() is not None
            }
            alive: dict[str, TargetRecord] = {}
            for bv in views:
                key = id(bv)
                view_id = None
                entry = self._ids_by_object.get(key)
                if entry is not None:
                    ref, candidate = entry
                    # Only reuse the id if the stored ref still points at this
                    # exact object; id() values get recycled by CPython.
                    if ref() is bv:
                        view_id = candidate
                if view_id is None:
                    view_id = str(self._next_id)
                    self._next_id += 1
                    self._ids_by_object[key] = (weakref.ref(bv), view_id)

                try:
                    session_id = str(bv.file.session_id)
                except Exception:
                    session_id = str(key)
                try:
                    filename = str(getattr(bv.file, "filename", "")) if bv.file else ""
                except Exception:
                    filename = ""

                alive[view_id] = TargetRecord(
                    view_id=view_id,
                    ref=weakref.ref(bv),
                    session_id=session_id,
                    filename=filename,
                    basename=os.path.basename(filename) if filename else "",
                    view_name=self._view_name(bv),
                )

            self._records = alive
            active = focused
            if active is None and len(self._records) == 1:
                active = next(iter(self._records.values())).ref()
            selectors = self._compute_selectors(self._records)

            result = []
            for view_id in sorted(self._records, key=lambda item: int(item)):
                record = self._records[view_id]
                view = record.ref()
                if view is None:
                    continue
                result.append(
                    {
                        "target_id": record.target_id(),
                        "view_id": record.view_id,
                        "session_id": record.session_id,
                        "filename": record.filename,
                        "basename": record.basename,
                        "selector": selectors[view_id],
                        "view_name": record.view_name,
                        "active": bool(view is active),
                    }
                )
            return result

    def resolve(self, selector: str | None):
        targets = self.refresh()
        if not targets:
            raise RuntimeError("No BinaryView targets are open")

        if selector in (None, "", "active"):
            active = self._default_view()
            if active is None:
                raise RuntimeError("No active BinaryView is selected and multiple targets are open")
            return active

        with self._lock:
            matches: list[tuple[TargetRecord, Any]] = []
            for record in self._records.values():
                if not self._matches_record(record, selector):
                    continue
                view = record.ref()
                if view is None:
                    continue
                matches.append((record, view))

        if not matches:
            raise RuntimeError(_format_unknown_target_error(selector, targets))
        if len(matches) > 1:
            selectors_by_view = {t["view_id"]: t["selector"] for t in targets}
            candidates = ", ".join(
                selectors_by_view.get(record.view_id, record.target_id())
                for record, _ in matches
            )
            raise RuntimeError(
                f"Ambiguous target selector: {selector!r} matches {len(matches)} targets ({candidates})"
            )
        return matches[0][1]


class BridgeHandler(socketserver.StreamRequestHandler):
    def _write_response(
        self,
        encoded: bytes,
        *,
        op: str | None = None,
        request_id: str | None = None,
    ) -> None:
        try:
            self.wfile.write(encoded)
        except OSError as exc:
            if exc.errno not in {errno.EPIPE, errno.ECONNRESET}:
                raise
            details = []
            if op:
                details.append(f"op={op}")
            if request_id:
                details.append(f"id={request_id}")
            suffix = f" ({', '.join(details)})" if details else ""
            bn.log_warn(f"BN Agent Bridge client disconnected before response could be delivered{suffix}")

    def handle(self):  # pragma: no cover - exercised from CLI
        raw = self.rfile.readline(MAX_REQUEST_BYTES)
        if not raw:
            return
        op = None
        request_id = None
        if len(raw) == MAX_REQUEST_BYTES and not raw.endswith(b"\n"):
            response = _json_response(ok=False, error="request too large")
        else:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                response = _json_response(ok=False, error="Invalid JSON request")
            else:
                if not isinstance(payload, dict):
                    response = _json_response(
                        ok=False, error="Invalid request: expected a JSON object"
                    )
                else:
                    op = payload.get("op")
                    request_id = payload.get("id")
                    response = self.server.bridge.dispatch(payload)
        encoded = json.dumps(response, sort_keys=True, default=str).encode("utf-8")
        self._write_response(encoded, op=op, request_id=request_id)


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, socket_path: str, handler, bridge):
        self.bridge = bridge
        super().__init__(socket_path, handler)


_TYPE_CLASS_NAMES: dict[int, str] = {
    0: "void",
    1: "bool",
    2: "int",
    3: "float",
    4: "struct",
    5: "enum",
    6: "pointer",
    7: "array",
    8: "function",
    9: "varargs",
    10: "value",
    11: "named_type_ref",
    12: "wide_char",
}

_STRING_TYPE_NAMES: dict[int, str] = {
    0: "ascii",
    1: "utf16",
    2: "utf32",
}

_SOURCE_TYPE_SHORT: dict[str, str] = {
    "RegisterVariableSourceType": "reg",
    "StackVariableSourceType": "stack",
    "FlagVariableSourceType": "flag",
}


class BinaryNinjaBridge:
    def __init__(self, instance_id: str | None = None):
        self.instance_id = instance_id
        self.targets = TargetManager()
        self.socket_path = bridge_socket_path(instance_id)
        self.registry_path = bridge_registry_path(instance_id)
        self._server: ThreadedUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._target_lock = _ReadWriteLock()
        self._shutdown_event = threading.Event()

    def _socket_is_live(self) -> bool:
        """Whether something is currently accepting connections on our socket path."""
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self.socket_path))
        except OSError:
            return False
        finally:
            with contextlib.suppress(OSError):
                probe.close()
        return True

    def start(self):  # pragma: no cover - requires GUI runtime
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            # A stale socket file from a crashed bridge is safe to clear, but a
            # live one belongs to another bridge instance; unlinking it would
            # silently orphan that bridge on an unlinked inode.
            if self._socket_is_live():
                raise RuntimeError(
                    f"Another bridge is already serving on {self.socket_path}; "
                    "refusing to displace it"
                )
            self.socket_path.unlink()

        self._server = ThreadedUnixServer(str(self.socket_path), BridgeHandler, self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._write_registry()
        bn.log_info(f"BN Agent Bridge listening on {self.socket_path}")

    def stop(self):  # pragma: no cover - requires GUI runtime
        if self._server is not None:
            with contextlib.suppress(Exception):
                self._server.shutdown()
            with contextlib.suppress(Exception):
                self._server.server_close()
        if self.socket_path.exists():
            with contextlib.suppress(OSError):
                self.socket_path.unlink()
        if self.registry_path.exists():
            with contextlib.suppress(OSError):
                self.registry_path.unlink()
        # On a clean shutdown there's no crash to diagnose, so drop the log
        # file too rather than leave it as clutter in the instances dir. A
        # crash skips stop() entirely (SIGKILL/segfault), so crash logs are
        # preserved for `bn`'s empty-response diagnostic to point at.
        log_path = self.registry_path.with_suffix(".log")
        if log_path.exists():
            with contextlib.suppress(OSError):
                log_path.unlink()

    def _write_registry(self):
        payload = {
            "pid": os.getpid(),
            "socket_path": str(self.socket_path),
            "plugin_name": PLUGIN_NAME,
            "plugin_version": VERSION,
            "plugin_build_id": PLUGIN_BUILD_ID,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.instance_id is not None:
            payload["instance_id"] = self.instance_id
        # Write atomically (temp file + rename) so a concurrent reader never
        # sees a half-written registry and concludes no instance exists.
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.registry_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, indent=2))
            os.replace(tmp, self.registry_path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - GUI runtime
        op = payload.get("op")
        params = payload.get("params") or {}
        target = payload.get("target")
        try:
            lock = contextlib.nullcontext()
            if op in WRITE_LOCKED_OPS:
                lock = self._target_lock.write()
            elif op == "decompile" and params.get("force_analysis"):
                # --force-analysis reanalyzes the function, mutating the view, so
                # it needs the exclusive lock even though decompile is a read op.
                lock = self._target_lock.write()
            elif op in READ_LOCKED_OPS:
                lock = self._target_lock.read()
            with lock:
                result = self._dispatch_on_main(op, params, target)
            return _json_response(ok=True, result=result)
        except Exception as exc:
            return _json_response(ok=False, error=_serialize_error(exc))

    def _dispatch_on_main(self, op: str, params: dict[str, Any], target: str | None):
        if op == "doctor":
            return self._doctor()
        if op == "list_targets":
            return self.targets.refresh()
        if op == "target_info":
            return self._target_info(params.get("selector") or target)
        if op == "refresh":
            return self._refresh(target)
        if op == "shutdown":
            self._shutdown_event.set()
            return {"shutting_down": True}

        if op == "load_binary":
            return self._load_binary(
                str(params["path"]),
                prefer_bndb=bool(params.get("prefer_bndb", True)),
                quick=bool(params.get("quick", False)),
            )
        if op == "close_binary":
            return self._close_binary(params.get("path"), target, params.get("all"))
        if op == "save_database":
            return self._save_database(target, params.get("path"))

        if op == "list_functions":
            return self._list_functions(
                target,
                min_address=params.get("min_address"),
                max_address=params.get("max_address"),
                offset=int(params.get("offset", 0)),
                limit=int(params["limit"]) if "limit" in params else None,
                count_only=bool(params.get("count_only", False)),
            )
        if op == "search_functions":
            return self._search_functions(
                target,
                str(params.get("query", "")),
                regex=bool(params.get("regex", False)),
                exact=bool(params.get("exact", False)),
                min_address=params.get("min_address"),
                max_address=params.get("max_address"),
                offset=int(params.get("offset", 0)),
                limit=int(params["limit"]) if "limit" in params else None,
            )
        if op == "callsites":
            return self._callsites(
                target,
                str(params["callee"]),
                within_identifiers=list(params.get("within_identifiers") or []),
                context=int(params.get("context", 3)),
            )
        if op == "function_info":
            return self._function_info(target, params["identifier"])
        if op == "get_prototype":
            return self._get_prototype(target, params["identifier"])
        if op == "list_locals":
            return self._list_locals_for_function(target, params["identifier"])
        if op == "decompile":
            return self._decompile(
                target,
                params["identifier"],
                addresses=bool(params.get("addresses")),
                force_analysis=bool(params.get("force_analysis")),
            )
        if op == "il":
            return self._il(target, params["identifier"], str(params.get("view", "hlil")), bool(params.get("ssa")))
        if op == "structured_il":
            return self._structured_il(
                target,
                params["identifier"],
                view=str(params.get("view", "mlil")),
                ssa=bool(params.get("ssa", True)),
            )
        if op == "defuse":
            return self._defuse(target, params["identifier"], str(params["var"]))
        if op == "resolved_calls":
            return self._resolved_calls(
                target,
                params["identifier"],
                direction=str(params.get("direction", "both")),
                resolve_indirect=bool(params.get("resolve_indirect", True)),
            )
        if op == "possible_values":
            return self._possible_values(target, params["identifier"], params["at"])
        if op == "taint":
            return self._taint(target, params)
        if op == "disasm":
            return self._disasm(target, params["identifier"])
        if op == "function_evidence":
            return self._function_evidence(
                target,
                params["identifier"],
                context=int(params.get("context", 2)),
            )
        if op == "xrefs":
            return self._xrefs(target, params["identifier"])
        if op == "field_xrefs":
            return self._field_xrefs(target, str(params["field"]))
        if op == "pointer_table":
            return self._pointer_table(
                target,
                params["address"],
                entries=int(params.get("entries", 16)),
                stride=params.get("stride"),
            )
        if op == "message_lens":
            return self._message_lens(
                target,
                str(params["query"]),
                limit=int(params.get("limit", 20)),
                table_entries=int(params.get("table_entries", 6)),
            )
        if op == "init_arrays":
            return self._init_arrays(
                target,
                limit=int(params.get("limit", 64)),
            )
        if op == "backward_slice":
            return self._backward_slice(
                target,
                str(params["identifier"]),
                str(params["address"]),
                arg_index=int(params.get("arg_index", 0)),
                view=str(params.get("view", "mlil")),
                max_depth=int(params.get("max_depth", 50)),
                interprocedural=bool(params.get("interprocedural", False)),
                ip_depth=int(params.get("ip_depth", 2)),
            )
        if op == "types":
            return self._types(
                target,
                query=params.get("query"),
                offset=int(params.get("offset", 0)),
                limit=int(params.get("limit", 100)),
            )
        if op == "type_info":
            return self._type_info(
                target,
                str(params["type_name"]),
                require_struct=bool(params.get("require_struct")),
            )
        if op == "strings":
            return self._strings(
                target,
                query=params.get("query"),
                offset=int(params.get("offset", 0)),
                limit=int(params.get("limit", 100)),
                min_length=int(params["min_length"]) if params.get("min_length") is not None else None,
                section=params.get("section"),
                no_crt=bool(params.get("no_crt", False)),
                regex=bool(params.get("regex", False)),
            )
        if op == "imports":
            return self._imports(
                target,
                summary=bool(params.get("summary", False)),
                offset=int(params.get("offset", 0)),
                limit=int(params["limit"]) if params.get("limit") is not None else None,
            )
        if op == "sections":
            return self._sections(
                target,
                query=params.get("query"),
                offset=int(params.get("offset", 0)),
                limit=int(params["limit"]) if params.get("limit") is not None else None,
            )
        if op == "read":
            return self._read(target, params["address"], int(params["length"]))
        if op == "function_create":
            return self._function_create(target, params["address"], bool(params.get("preview")))
        if op == "bundle_function":
            return self._bundle_function(target, params["identifier"], params.get("out_path"))
        if op == "py_exec":
            return self._py_exec(target, str(params["script"]))

        if op == "rename_symbol":
            return self._mutation(target, bool(params.get("preview")), [params])
        if op == "get_comment":
            return self._get_comment(target, params.get("address"), params.get("function"))
        if op == "list_comments":
            return self._list_comments(
                target,
                query=params.get("query"),
                offset=int(params.get("offset", 0)),
                limit=int(params["limit"]) if "limit" in params else None,
            )
        if op == "set_comment":
            return self._mutation(target, bool(params.get("preview")), [{"op": "set_comment", **params}])
        if op == "delete_comment":
            return self._mutation(target, bool(params.get("preview")), [{"op": "delete_comment", **params}])
        if op == "set_prototype":
            return self._mutation(target, bool(params.get("preview")), [{"op": "set_prototype", **params}])
        if op == "local_rename":
            return self._mutation(target, bool(params.get("preview")), [{"op": "local_rename", **params}])
        if op == "local_retype":
            return self._mutation(target, bool(params.get("preview")), [{"op": "local_retype", **params}])
        if op == "struct_field_set":
            return self._mutation(target, bool(params.get("preview")), [{"op": "struct_field_set", **params}])
        if op == "struct_field_rename":
            return self._mutation(target, bool(params.get("preview")), [{"op": "struct_field_rename", **params}])
        if op == "struct_field_delete":
            return self._mutation(target, bool(params.get("preview")), [{"op": "struct_field_delete", **params}])
        if op == "types_declare":
            return self._mutation(target, bool(params.get("preview")), [{"op": "types_declare", **params}])
        if op == "batch_apply":
            manifest = dict(params)
            preview = bool(manifest.get("preview"))
            # Keep None as None so the single-open-target default still applies;
            # str(None) would become the bogus selector "None".
            chosen = manifest.get("target") or target
            target = str(chosen) if chosen is not None else None
            operations = list(manifest.get("ops") or [])
            return self._mutation(target, preview, operations)

        raise ValueError(f"Unknown operation: {op}")

    def _doctor(self):
        return {
            "plugin_name": PLUGIN_NAME,
            "plugin_version": VERSION,
            "plugin_build_id": PLUGIN_BUILD_ID,
            "pid": os.getpid(),
            "socket_path": str(self.socket_path),
            "targets": self.targets.refresh(),
        }

    def _load_binary(self, path: str, *, prefer_bndb: bool = True, quick: bool = False):
        import binaryninja

        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"File not found: {resolved}")

        load_path = resolved
        notes: list[str] = []
        if prefer_bndb and resolved.suffix != ".bndb":
            sibling = Path(str(resolved) + ".bndb")
            if sibling.exists():
                load_path = sibling
                notes.append(
                    f"loaded {sibling} instead of {resolved} (use --no-bndb to skip)"
                )

        # Always open without auto-analysis, then analyze explicitly unless
        # --quick. Quick load skips update_analysis_and_wait() entirely -- the
        # expensive, occasionally-crashing/OOMing phase -- so sections, symbols,
        # imports and strings are usable in ~1s while the function set stays
        # minimal until `bn refresh` promotes it to full analysis. A .bndb
        # already carries its saved analysis, so --quick is a no-op there.
        try:
            bv = binaryninja.load(str(load_path), update_analysis=False)
        except Exception as exc:  # noqa: BLE001 - surface BN open failures cleanly
            # BN raises a bare Exception ("Unable to create new BinaryView") on a
            # corrupt/truncated .bndb; without this it reached the caller as
            # "internal error: Exception: ...". Frame it as a user-facing error.
            raise RuntimeError(
                f"Unable to open {load_path}: {exc}. The file may be corrupt, "
                "truncated, or an unsupported format."
            ) from exc
        if bv is None:
            raise RuntimeError(f"Failed to open binary: {load_path}")

        quick_effective = quick and load_path.suffix != ".bndb"
        if quick_effective:
            notes.append(
                "loaded without analysis (--quick): sections/imports/symbols are ready; "
                "strings and the full function set need `bn refresh` (or "
                "`bn decompile <fn> --force-analysis` for a single function)"
            )
            _quick_loaded_views.add(bv)
        else:
            bv.update_analysis_and_wait()
            _quick_loaded_views.discard(bv)

        with _headless_views_lock:
            _headless_views.append(bv)

        return {
            "loaded": True,
            "path": str(load_path),
            "requested_path": str(resolved),
            "analyzed": not quick_effective,
            "notes": notes,
            "targets": self.targets.refresh(),
        }

    def _close_binary(self, path: str | None = None, target: str | None = None, all_: bool = False):
        def _snapshot(bv) -> dict[str, Any]:
            return {
                "path": str(getattr(bv.file, "filename", "")),
                "unsaved": bool(getattr(bv.file, "modified", False)),
            }

        # Resolve a target selector *before* taking _headless_views_lock:
        # resolve() -> refresh() -> _collect_open_views() re-acquires that lock,
        # which is non-reentrant, so resolving while holding it deadlocks.
        target_bv = self.targets.resolve(target) if target is not None else None

        with _headless_views_lock:
            if not _headless_views:
                raise RuntimeError("No binaries are currently loaded")

            # Target-based close takes priority over path
            if target_bv is not None:
                closed = [_snapshot(target_bv)]
                target_bv.file.close()
                _headless_views[:] = [v for v in _headless_views if v is not target_bv]
                return {"closed": closed}

            # --all closes everything
            if all_ or path is None:
                closed = []
                for bv in _headless_views:
                    closed.append(_snapshot(bv))
                    bv.file.close()
                _headless_views.clear()
                return {"closed": closed}

            resolved = str(Path(path).expanduser().resolve())
            to_remove = []
            for i, bv in enumerate(_headless_views):
                filename = str(getattr(bv.file, "filename", ""))
                if filename == resolved or str(Path(filename).resolve()) == resolved:
                    to_remove.append(i)

            if not to_remove:
                raise RuntimeError(f"No loaded binary matches path: {path}")

            closed = []
            for i in reversed(to_remove):
                bv = _headless_views.pop(i)
                closed.append(_snapshot(bv))
                bv.file.close()

        return {"closed": closed}

    def _save_database(self, target: str | None, path: str | None = None):
        bv = self.targets.resolve(target)
        filename = getattr(bv.file, "filename", "")
        if path:
            out = str(Path(path).expanduser().resolve())
        elif filename.endswith(".bndb"):
            out = filename
        else:
            out = filename + ".bndb"
        out_path = Path(out)
        if not out_path.parent.exists():
            raise RuntimeError(
                f"Cannot save database to {out}: directory does not exist: {out_path.parent}"
            )
        # create_database returns a bool: False means Binary Ninja could not write
        # the file (e.g. an unwritable directory). The previous code discarded that
        # return value and unconditionally reported success, silently losing the
        # analysis. Treat a falsy result -- or a path that simply isn't there
        # afterward -- as a hard failure so callers never get a false "saved".
        try:
            created = bv.create_database(out)
        except Exception as exc:  # noqa: BLE001 - surface BN I/O errors cleanly
            raise RuntimeError(f"Failed to save database to {out}: {exc}") from exc
        if created is False or not out_path.exists():
            raise RuntimeError(
                f"Failed to save database to {out}: Binary Ninja reported no file was "
                "written (check that the directory exists and is writable)"
            )
        return {"saved": True, "path": out}

    def _target_info(self, selector: str | None):
        bv = self.targets.resolve(selector)
        record = None
        for item in self.targets.refresh():
            if item["active"] and selector in (None, "", "active"):
                record = item
                break
            if selector and self.targets.matches_target(item["target_id"], selector):
                record = item
                break
        quick = bv in _quick_loaded_views
        return {
            **(record or {}),
            "arch": str(getattr(bv, "arch", "")),
            "platform": str(getattr(bv, "platform", "")),
            "entry_point": hex(getattr(bv, "entry_point", 0)),
            # Machine-readable analysis state so callers can tell a --quick view
            # (strings/full function set pending `bn refresh`) from a real one.
            "analyzed": not quick,
            "analysis_state": "quick" if quick else "full",
        }

    def _refresh(self, selector: str | None):
        bv = self._resolve_view(selector)
        bv.update_analysis_and_wait()
        _quick_loaded_views.discard(bv)
        return {
            "refreshed": True,
            "target": self._target_info(selector),
        }

    def _resolve_view(self, selector: str | None):
        return self.targets.resolve(selector)

    def _find_function(self, bv, identifier):
        # A 0x-prefixed identifier is unambiguously an address attempt (function
        # names never start with 0x), so a parse failure or a miss should report
        # the address problem rather than silently degrading to a name search
        # that ends in a misleading "Function not found".
        looks_like_address = str(identifier).strip().lower().startswith("0x")
        addr = None
        try:
            addr = _parse_address(identifier)
        except ValueError:
            if looks_like_address:
                raise RuntimeError(
                    f"Invalid address {identifier!r}: expected a 0x-prefixed hex or decimal value"
                ) from None
        if addr is not None:
            try:
                fn = bv.get_function_at(addr)
            except Exception:
                fn = None
            if fn is not None:
                return fn
            if looks_like_address:
                raise RuntimeError(f"No function found at address {hex(addr)}")

        text = str(identifier)
        exact = self._find_functions_by_name(bv, text, case_sensitive=True)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise RuntimeError(_format_ambiguous_function_error(identifier, exact))

        folded = self._find_functions_by_name(bv, text, case_sensitive=False)
        if len(folded) == 1:
            return folded[0]
        if len(folded) > 1:
            raise RuntimeError(_format_ambiguous_function_error(identifier, folded))

        symbol = bv.get_symbol_by_raw_name(text)
        if symbol is not None:
            fn = bv.get_function_at(symbol.address)
            if fn is not None:
                return fn

        available: list[str] = []
        for fn in list(bv.functions):
            available.append(str(fn.name))
            raw = str(getattr(fn, "raw_name", fn.name))
            if raw != str(fn.name):
                available.append(raw)
        suggestions = difflib.get_close_matches(text, available, n=5, cutoff=0.5)
        if suggestions:
            raise RuntimeError(
                f"Function not found: {identifier}. Did you mean: {', '.join(suggestions)}"
            )
        raise RuntimeError(f"Function not found: {identifier}")

    def _find_functions_by_name(self, bv, text: str, *, case_sensitive: bool) -> list[Any]:
        matches = []
        needle = text if case_sensitive else text.lower()
        seen: set[int] = set()
        for fn in list(bv.functions):
            names = [str(fn.name), str(getattr(fn, "raw_name", fn.name))]
            haystacks = names if case_sensitive else [name.lower() for name in names]
            if needle not in haystacks:
                continue
            marker = int(fn.start)
            if marker in seen:
                continue
            seen.add(marker)
            matches.append(fn)
        return matches

    def _resolve_scope_functions(self, bv, identifiers: list[Any]) -> list[tuple[str, Any]]:
        if not identifiers:
            raise OperationFailure("invalid_scope", "callsites requires at least one scoped function")

        resolved = []
        seen: set[int] = set()
        for identifier in identifiers:
            fn = self._find_function(bv, identifier)
            marker = int(fn.start)
            if marker in seen:
                continue
            seen.add(marker)
            resolved.append((str(identifier), fn))
        return resolved

    def _find_symbols_by_name(self, bv, text: str, *, case_sensitive: bool) -> list[Any]:
        matches = []
        seen: set[tuple[int, str]] = set()

        if case_sensitive:
            candidates = list(bv.get_symbols_by_name(text))
            raw_match = bv.get_symbol_by_raw_name(text)
            if raw_match is not None:
                candidates.append(raw_match)
        else:
            folded = text.lower()
            candidates = []
            for symbol in list(bv.get_symbols()):
                names = [str(getattr(symbol, "name", "")), str(getattr(symbol, "raw_name", ""))]
                if folded in {name.lower() for name in names if name}:
                    candidates.append(symbol)

        for symbol in candidates:
            marker = (int(symbol.address), str(symbol.type))
            if marker in seen:
                continue
            seen.add(marker)
            matches.append(symbol)
        return matches

    def _resolve_rename_target(self, bv, identifier: Any, kind: str) -> dict[str, Any]:
        requested = {
            "kind": kind,
            "identifier": str(identifier),
        }

        try:
            address = _parse_address(identifier)
        except Exception:
            address = None

        if address is not None:
            fn = bv.get_function_at(address)
            symbol = bv.get_symbol_at(address)
            if kind == "function":
                if fn is None:
                    raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            if kind == "data":
                return {
                    "kind": "data",
                    "address": int(address),
                    "before_name": str(symbol.name) if symbol is not None else None,
                }
            if fn is not None:
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            return {
                "kind": "data",
                "address": int(address),
                "before_name": str(symbol.name) if symbol is not None else None,
            }

        if kind in {"auto", "function"}:
            exact_functions = self._find_functions_by_name(bv, str(identifier), case_sensitive=True)
            if len(exact_functions) == 1:
                fn = exact_functions[0]
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            if len(exact_functions) > 1:
                raise OperationFailure(
                    "unsupported",
                    _format_ambiguous_function_error(identifier, exact_functions),
                    requested=requested,
                )

            folded_functions = self._find_functions_by_name(bv, str(identifier), case_sensitive=False)
            if len(folded_functions) == 1:
                fn = folded_functions[0]
                return {
                    "kind": "function",
                    "address": int(fn.start),
                    "before_name": str(fn.name),
                }
            if len(folded_functions) > 1:
                raise OperationFailure(
                    "unsupported",
                    _format_ambiguous_function_error(identifier, folded_functions),
                    requested=requested,
                )

        if kind == "function":
            raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)

        exact_symbols = [
            symbol
            for symbol in self._find_symbols_by_name(bv, str(identifier), case_sensitive=True)
            if symbol.type != bn.SymbolType.FunctionSymbol
        ]
        if len(exact_symbols) == 1:
            symbol = exact_symbols[0]
            return {
                "kind": "data",
                "address": int(symbol.address),
                "before_name": str(symbol.name),
            }
        if len(exact_symbols) > 1:
            raise OperationFailure(
                "unsupported",
                _format_ambiguous_symbol_error(identifier, exact_symbols),
                requested=requested,
            )

        folded_symbols = [
            symbol
            for symbol in self._find_symbols_by_name(bv, str(identifier), case_sensitive=False)
            if symbol.type != bn.SymbolType.FunctionSymbol
        ]
        if len(folded_symbols) == 1:
            symbol = folded_symbols[0]
            return {
                "kind": "data",
                "address": int(symbol.address),
                "before_name": str(symbol.name),
            }
        if len(folded_symbols) > 1:
            raise OperationFailure(
                "unsupported",
                _format_ambiguous_symbol_error(identifier, folded_symbols),
                requested=requested,
            )

        raise OperationFailure("unsupported", f"Symbol not found: {identifier}", requested=requested)

    def _functions_containing(self, bv, address: int):
        try:
            return list(bv.get_functions_containing(address))
        except Exception:
            fn = bv.get_function_at(address)
            return [fn] if fn is not None else []

    def _sections_at(self, bv, address: int) -> list[dict[str, Any]]:
        try:
            sections = list(bv.get_sections_at(address))
        except Exception:
            sections = []
            for name, sec in getattr(bv, "sections", {}).items():
                try:
                    if int(sec.start) <= address < int(sec.end):
                        sections.append(sec)
                except Exception:
                    continue

        result = []
        for sec in sections:
            try:
                start = int(getattr(sec, "start", 0))
                end = int(getattr(sec, "end", 0))
            except Exception:
                start = end = 0
            result.append(
                {
                    "name": str(getattr(sec, "name", "")),
                    "start": hex(start),
                    "end": hex(end),
                }
            )
        return result

    def _segment_at(self, bv, address: int) -> dict[str, Any] | None:
        try:
            seg = bv.get_segment_at(address)
        except Exception:
            seg = None
        if seg is None:
            return None
        entry: dict[str, Any] = {
            "readable": bool(getattr(seg, "readable", False)),
            "writable": bool(getattr(seg, "writable", False)),
            "executable": bool(getattr(seg, "executable", False)),
        }
        for attr in ("start", "end"):
            value = getattr(seg, attr, None)
            if value is not None:
                try:
                    entry[attr] = hex(int(value))
                except Exception:
                    pass
        return entry

    def _symbol_at(self, bv, address: int) -> dict[str, Any] | None:
        try:
            symbol = bv.get_symbol_at(address)
        except Exception:
            symbol = None
        if symbol is None:
            return None
        raw_type = getattr(symbol, "type", None)
        kind = getattr(raw_type, "name", None) or str(raw_type)
        return {
            "name": str(getattr(symbol, "name", "")),
            "raw_name": str(getattr(symbol, "raw_name", getattr(symbol, "name", ""))),
            "type": kind,
        }

    def _function_entry_for_address(self, bv, address: int) -> dict[str, Any] | None:
        try:
            fn = bv.get_function_at(address)
        except Exception:
            fn = None
        if fn is None:
            functions = self._functions_containing(bv, address)
            fn = functions[0] if functions else None
        if fn is None:
            return None
        function_start = int(fn.start)
        entry = {
            "name": str(fn.name),
            "address": hex(function_start),
            "exact_start": function_start == int(address),
        }
        if function_start != int(address):
            delta = int(address) - function_start
            entry["offset"] = f"-{hex(abs(delta))}" if delta < 0 else hex(delta)
        return entry

    def _raw_sections_at(self, bv, address: int) -> list[Any]:
        try:
            return list(bv.get_sections_at(address))
        except Exception:
            result = []
            for _name, sec in getattr(bv, "sections", {}).items():
                try:
                    if int(sec.start) <= address < int(sec.end):
                        result.append(sec)
                except Exception:
                    continue
            return result

    def _section_semantics_name(self, sec) -> str:
        sem = getattr(sec, "semantics", None)
        return getattr(sem, "name", None) or str(sem)

    def _address_is_code(self, bv, address: int) -> bool:
        """True only when the address is real code.

        Keys on function membership and section *semantics* (ReadOnlyCode), not
        the segment's executable bit — firmware ELFs routinely map .rodata into
        the same r-x load segment as .text, so an executable segment is not
        evidence that an address is an instruction.
        """
        if self._functions_containing(bv, address):
            return True
        for sec in self._raw_sections_at(bv, address):
            if "Code" in self._section_semantics_name(sec):
                return True
        return False

    def _resolve_data_string(self, bv, address: int, *, max_chars: int = 96) -> dict[str, Any] | None:
        """Best-effort printable string at *address*, even when BN never
        atomized one there (e.g. single chars packed for std::string::append).

        Tries a NUL-terminated ASCII run first, then UTF-16LE. Common escaped
        text controls are allowed. Long strings are capped and marked
        truncated so evidence output stays compact. Returns None for non-string
        bytes so it can be used as a cheap "is this a string?" probe.
        """
        try:
            data = bytes(bv.read(int(address), max_chars * 2 + 2))
        except Exception:
            return None
        if not data:
            return None

        allowed_ascii = set(range(32, 127)) | {9, 10, 13}
        ascii_chars: list[str] = []
        for byte in data:
            if byte == 0:
                if ascii_chars:
                    return {
                        "value": "".join(ascii_chars),
                        "encoding": "ascii",
                        "truncated": False,
                    }
                break
            if byte not in allowed_ascii:
                ascii_chars = []
                break
            if len(ascii_chars) >= max_chars:
                return {
                    "value": "".join(ascii_chars),
                    "encoding": "ascii",
                    "truncated": True,
                }
            ascii_chars.append(chr(byte))
        else:
            if ascii_chars:
                return {
                    "value": "".join(ascii_chars),
                    "encoding": "ascii",
                    "truncated": True,
                }

        if ascii_chars:
            return {
                "value": "".join(ascii_chars),
                "encoding": "ascii",
                "truncated": True,
            }

        chars: list[str] = []
        index = 0
        terminated = False
        allowed_wide = allowed_ascii
        while index + 1 < len(data) and len(chars) < max_chars:
            lo, hi = data[index], data[index + 1]
            if lo == 0 and hi == 0:
                terminated = True
                break
            if hi != 0 or lo not in allowed_wide:
                chars = []
                break
            chars.append(chr(lo))
            index += 2
        if (
            len(chars) >= max_chars
            and index + 1 < len(data)
            and data[index] == 0
            and data[index + 1] == 0
        ):
            terminated = True
        if len(chars) >= 2:
            return {
                "value": "".join(chars),
                "encoding": "utf-16le",
                "truncated": not terminated and len(chars) >= max_chars,
            }
        return None

    def _address_context(self, bv, address: int, *, include_disasm: bool = False, arch=None,
                         assume_code: bool = False) -> dict[str, Any]:
        address = int(address)
        sections = self._sections_at(bv, address)
        segment = self._segment_at(bv, address)
        symbol = self._symbol_at(bv, address)
        function = self._function_entry_for_address(bv, address)
        context: dict[str, Any] = {
            "address": hex(address),
            "sections": sections,
            "segment": segment,
            "symbol": symbol,
            "function": function,
        }
        section_name = sections[0]["name"].lower() if sections else ""
        symbol_type = (symbol or {}).get("type") or ""
        if address == 0:
            kind = "null"
        elif segment is None and not sections:
            kind = "unmapped"
        elif assume_code or function is not None or self._address_is_code(bv, address):
            kind = "code"
        elif symbol_type == "ExternalSymbol" or "extern" in section_name:
            kind = "extern"
        else:
            resolved = self._resolve_data_string(bv, address)
            if resolved is not None:
                context["string"] = resolved
                kind = "string"
            else:
                kind = "data"
        context["kind"] = kind
        if include_disasm:
            if kind == "code":
                context["disasm"] = self._safe_disassembly(bv, address, arch)
            else:
                context["disasm"] = None
                context["notes"] = [f"target is {kind}; disassembly suppressed"]
        return context

    def _safe_disassembly(self, bv, address: int, arch=None) -> str:
        for args in ((address, arch) if arch is not None else (), (address,)):
            try:
                return bv.get_disassembly(*args) or ""
            except Exception:
                continue
        return ""

    def _pointer_size(self, bv) -> int:
        for obj in (bv, getattr(bv, "arch", None)):
            value = getattr(obj, "address_size", None)
            if value is None:
                continue
            try:
                size = int(value)
                if size > 0:
                    return size
            except Exception:
                pass
        return 4

    def _byteorder(self, bv) -> str:
        for obj in (bv, getattr(bv, "arch", None)):
            value = getattr(obj, "endianness", None)
            text = str(value)
            if "Big" in text or "big" in text:
                return "big"
        return "little"

    def _supports_thumb_pointer_tags(self, bv) -> bool:
        if self._pointer_size(bv) != 4:
            return False
        names = []
        arch = getattr(bv, "arch", None)
        for obj in (arch, getattr(bv, "platform", None)):
            if obj is None:
                continue
            for attr in ("name", "raw_name"):
                value = getattr(obj, attr, None)
                if value:
                    names.append(str(value).lower())
            names.append(str(obj).lower())
        joined = " ".join(names)
        if "aarch64" in joined or "arm64" in joined:
            return False
        return "thumb" in joined or "arm" in joined

    def _read_pointer_value(self, bv, address: int, *, size: int | None = None) -> int | None:
        pointer_size = size or self._pointer_size(bv)
        try:
            data = bytes(bv.read(address, pointer_size))
        except Exception:
            return None
        if len(data) != pointer_size:
            return None
        return int.from_bytes(data, self._byteorder(bv), signed=False)

    def _normalize_code_pointer(self, bv, value: int) -> dict[str, Any]:
        raw = int(value)
        normalized = raw
        thumb_adjusted = False
        if raw & 1 and self._supports_thumb_pointer_tags(bv):
            candidate = raw & ~1
            candidate_function = self._function_entry_for_address(bv, candidate)
            if candidate_function is not None:
                normalized = candidate
                thumb_adjusted = True
                function = candidate_function
            else:
                function = self._function_entry_for_address(bv, normalized)
        else:
            function = self._function_entry_for_address(bv, normalized)
        context = self._address_context(bv, normalized, include_disasm=bool(function))
        segment = context.get("segment")
        status = "function" if function is not None else "mapped" if segment is not None else "null" if raw == 0 else "unmapped"
        plausible = status in {"function", "mapped", "null"}
        return {
            "raw": hex(raw),
            "normalized": hex(normalized),
            "thumb_adjusted": thumb_adjusted,
            "function": function,
            "status": status,
            "plausible": plausible,
            "context": context,
        }

    def _find_variable_by_storage(self, func, storage: int, *, is_parameter: bool | None = None):
        collections = []
        if is_parameter is True:
            collections = [(func.parameter_vars, True)]
        elif is_parameter is False:
            collections = [(func.stack_layout, False)]
        else:
            collections = [(func.parameter_vars, True), (func.stack_layout, False)]

        for collection, marker in collections:
            for var in list(collection):
                if int(var.storage) == int(storage):
                    return var, marker
        raise RuntimeError(f"Variable not found at storage {storage}")

    def _variable_source_name(self, var) -> str:
        source_type = getattr(var, "source_type", None)
        if source_type is None:
            return "unknown"
        return str(getattr(source_type, "name", source_type))

    def _variable_identifier(self, var) -> int | None:
        try:
            return int(getattr(var, "identifier"))
        except Exception:
            return None

    def _local_id(self, func, var, *, is_parameter: bool) -> str:
        role = "param" if is_parameter else "local"
        storage = int(getattr(var, "storage", 0))
        index = int(getattr(var, "index", 0))
        identifier = self._variable_identifier(var)
        source_name = self._variable_source_name(var)
        short_source = _SOURCE_TYPE_SHORT.get(source_name, source_name)
        return ":".join(
            [
                hex(int(func.start)),
                role,
                short_source,
                str(storage),
                str(index),
                str(identifier if identifier is not None else "none"),
            ]
        )

    def _variable_entry(self, func, var, *, is_parameter: bool) -> dict[str, Any]:
        return {
            "name": str(var.name),
            "storage": int(var.storage),
            "type": str(var.type),
            "is_parameter": is_parameter,
            "index": int(getattr(var, "index", 0)),
            "identifier": self._variable_identifier(var),
            "source_type": self._variable_source_name(var),
            "local_id": self._local_id(func, var, is_parameter=is_parameter),
        }

    def _variable_marker(self, var) -> tuple[int | None, int]:
        return (self._variable_identifier(var), int(getattr(var, "storage", 0)))

    def _iter_canonical_variables(self, func):
        seen: set[tuple[int | None, int]] = set()

        for var in list(func.parameter_vars):
            marker = self._variable_marker(var)
            if marker in seen:
                continue
            seen.add(marker)
            yield var, True

        for var in list(func.stack_layout):
            marker = self._variable_marker(var)
            if marker in seen:
                continue
            seen.add(marker)
            yield var, False

        # Register/flag locals that HLIL renders (e.g. rsi_1, rdx_3, loop
        # counters, the success flag) are real, nameable Variables that live in
        # neither parameter_vars nor stack_layout, so without this they are
        # invisible to `local list` and unresolvable by `local rename`/`retype`
        # (-> "Variable not found", which rolls back the whole batch). Surface
        # the HLIL-visible ones; func.vars would also drag in dataflow
        # temporaries (temp0, cond intermediates) that never appear in output.
        for var in self._iter_hlil_variables(func):
            marker = self._variable_marker(var)
            if marker in seen:
                continue
            seen.add(marker)
            yield var, False

    def _iter_hlil_variables(self, func):
        """HLIL-rendered variables, or empty when HLIL is unavailable.

        Large or non-decompilable functions may have no HLIL; fall back to the
        parameter/stack set rather than failing the whole listing.
        """
        try:
            hlil = func.hlil
        except Exception:
            return []
        if hlil is None:
            return []
        try:
            return list(hlil.vars)
        except Exception:
            return []

    def _format_hlil_tree(self, ins, indent=0, *, _else_prefix=False, addresses: bool = True):
        """Recursively format HLIL tree with proper indentation."""
        lines = []
        pad = "    " * indent
        op = ins.operation.name

        BODY_INDENT = "    "
        if addresses:
            def _prefix(i):
                a = getattr(i, "address", None)
                return f"{int(a):08x}        " if a is not None else "                "

            NO_PREFIX = "                "
        else:
            def _prefix(i):
                return BODY_INDENT

            NO_PREFIX = BODY_INDENT

        if op == "HLIL_NOP":
            pass

        elif op == "HLIL_BLOCK":
            for stmt in ins:
                lines.extend(self._format_hlil_tree(stmt, indent, addresses=addresses))

        elif op == "HLIL_IF":
            if _else_prefix:
                lines.append(f"{_prefix(ins)}{pad}}} else if ({ins.condition})")
            else:
                lines.append(f"{_prefix(ins)}{pad}if ({ins.condition})")
            lines.append(f"{NO_PREFIX}{pad}{{")
            lines.extend(self._format_hlil_tree(ins.true, indent + 1, addresses=addresses))
            false_branch = ins.false
            false_op = false_branch.operation.name
            if false_op == "HLIL_NOP":
                lines.append(f"{NO_PREFIX}{pad}}}")
            elif false_op == "HLIL_IF":
                lines.extend(self._format_hlil_tree(false_branch, indent, _else_prefix=True, addresses=addresses))
            else:
                lines.append(f"{NO_PREFIX}{pad}}} else {{")
                lines.extend(self._format_hlil_tree(false_branch, indent + 1, addresses=addresses))
                lines.append(f"{NO_PREFIX}{pad}}}")

        elif op in ("HLIL_WHILE", "HLIL_WHILE_SSA"):
            lines.append(f"{_prefix(ins)}{pad}while ({ins.condition})")
            lines.append(f"{NO_PREFIX}{pad}{{")
            lines.extend(self._format_hlil_tree(ins.body, indent + 1, addresses=addresses))
            lines.append(f"{NO_PREFIX}{pad}}}")

        elif op in ("HLIL_DO_WHILE", "HLIL_DO_WHILE_SSA"):
            lines.append(f"{_prefix(ins)}{pad}do")
            lines.append(f"{NO_PREFIX}{pad}{{")
            lines.extend(self._format_hlil_tree(ins.body, indent + 1, addresses=addresses))
            lines.append(f"{NO_PREFIX}{pad}}} while ({ins.condition})")

        elif op in ("HLIL_FOR", "HLIL_FOR_SSA"):
            lines.append(f"{_prefix(ins)}{pad}for ({ins.init}; {ins.condition}; {ins.update})")
            lines.append(f"{NO_PREFIX}{pad}{{")
            lines.extend(self._format_hlil_tree(ins.body, indent + 1, addresses=addresses))
            lines.append(f"{NO_PREFIX}{pad}}}")

        elif op == "HLIL_SWITCH":
            lines.append(f"{_prefix(ins)}{pad}switch ({ins.condition})")
            lines.append(f"{NO_PREFIX}{pad}{{")
            for case in ins.cases:
                lines.extend(self._format_hlil_tree(case, indent + 1, addresses=addresses))
            default = getattr(ins, "default", None)
            if default is not None and default.operation.name != "HLIL_NOP":
                lines.append(f"{NO_PREFIX}{pad}    default:")
                lines.extend(self._format_hlil_tree(default, indent + 2, addresses=addresses))
            lines.append(f"{NO_PREFIX}{pad}}}")

        elif op == "HLIL_CASE":
            for val in ins.values:
                lines.append(f"{_prefix(ins)}{pad}case {val}:")
            lines.extend(self._format_hlil_tree(ins.body, indent + 1, addresses=addresses))

        else:
            lines.append(f"{_prefix(ins)}{pad}{ins}")

        return lines

    def _function_text(self, bv, func, *, view: str = "hlil", ssa: bool = False, addresses: bool = True) -> str:
        il_name = {"hlil": "hlil", "mlil": "mlil", "llil": "llil"}.get(view, "hlil")
        try:
            il = getattr(func, il_name)
            if ssa and hasattr(il, "ssa_form") and il.ssa_form is not None:
                il = il.ssa_form
            if il_name == "hlil" and hasattr(il, "root"):
                try:
                    lines = self._format_hlil_tree(il.root, addresses=addresses)
                    if lines:
                        return "\n".join(lines)
                except Exception:
                    pass
            lines = []
            for ins in il.instructions:
                if addresses:
                    address = getattr(ins, "address", func.start)
                    lines.append(f"{int(address):08x}        {ins}")
                else:
                    lines.append(f"    {ins}")
            if lines:
                return "\n".join(lines)
        except Exception as exc:
            # Degrade to the prototype, but say so: a silent prototype-only
            # body with ok:true reads like a successful (empty) render.
            bn.log_warn(
                f"BN Agent Bridge: {view} rendering failed for {getattr(func, 'name', func)}: "
                f"{type(exc).__name__}: {exc}"
            )
            return (
                f"// bn: IL rendering failed ({type(exc).__name__}: {exc}); "
                f"showing prototype only\n{func}"
            )
        return str(func)

    def _instruction_length(self, bv, address: int, *, arch=None) -> int:
        if arch is None:
            arch = getattr(bv, "arch", None)
        try:
            max_length = int(getattr(arch, "max_instr_length", 16) or 16)
        except Exception:
            max_length = 16

        if arch is not None and hasattr(arch, "get_instruction_info"):
            try:
                data = bv.read(address, max_length)
                info = arch.get_instruction_info(data, address)
                length = int(getattr(info, "length", 0))
                if length > 0:
                    return length
            except Exception:
                pass

        try:
            length = int(bv.get_instruction_length(address))
            if length > 0:
                return length
        except Exception:
            pass
        return 1

    def _disasm_entry(self, bv, address: int, *, arch=None) -> dict[str, Any]:
        text = ""
        if arch is not None:
            try:
                max_length = int(getattr(arch, "max_instr_length", 16) or 16)
                data = bv.read(address, max_length)
                tokens, _length = arch.get_instruction_text(data, address)
                if tokens:
                    text = "".join(str(t) for t in tokens)
            except Exception:
                pass
        if not text:
            text = bv.get_disassembly(address) or ""
        return {
            "address": hex(int(address)),
            "text": text,
        }

    def _structured_disasm_entries(self, bv, func) -> list[dict[str, Any]]:
        arch = getattr(func, "arch", None)
        entries = []
        for block in list(func.basic_blocks):
            addr = int(block.start)
            end = int(block.end)
            while addr < end:
                entry = self._disasm_entry(bv, addr, arch=arch)
                if entry["text"]:
                    entry["_address_int"] = addr
                    entries.append(entry)
                addr += max(1, self._instruction_length(bv, addr, arch=arch))
        entries.sort(key=lambda item: int(item["_address_int"]))
        return entries

    def _disasm_text(self, bv, func) -> str:
        arch = getattr(func, "arch", None)
        lines = []
        for block in list(func.basic_blocks):
            addr = block.start
            while addr < block.end:
                length = max(1, self._instruction_length(bv, int(addr), arch=arch))
                entry = self._disasm_entry(bv, addr, arch=arch)
                disasm = entry["text"]
                raw = bv.read(addr, length)
                hex_bytes = raw.hex(" ") if raw else ""
                lines.append(f"{addr:08x}  {hex_bytes:<16} {disasm}")
                addr += length
        return "\n".join(lines)

    def _sort_variable_entries(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                0 if item.get("is_parameter") else 1,
                str(item.get("source_type", "")),
                int(item.get("storage", 0)),
                int(item.get("identifier") or 0),
                str(item.get("name", "")),
            ),
        )

    def _list_locals(self, func) -> list[dict[str, Any]]:
        variables = [
            self._variable_entry(func, var, is_parameter=is_parameter)
            for var, is_parameter in self._iter_canonical_variables(func)
        ]
        return self._sort_variable_entries(variables)

    def _find_variables_by_name(self, func, name: str) -> list[tuple[Any, bool]]:
        matches = []
        for var, is_parameter in self._iter_canonical_variables(func):
            if str(var.name) == name:
                matches.append((var, is_parameter))
        return matches

    def _find_variable_selector(self, func, selector: str) -> tuple[Any, bool]:
        locals_by_id: dict[str, tuple[Any, bool]] = {}
        legacy_by_id: dict[str, tuple[Any, bool]] = {}
        for var, is_parameter in self._iter_canonical_variables(func):
            local_id = self._local_id(func, var, is_parameter=is_parameter)
            locals_by_id[local_id] = (var, is_parameter)
            # Build legacy (long-form) ID for backward compat
            role = "param" if is_parameter else "local"
            source_name = self._variable_source_name(var)
            storage = int(getattr(var, "storage", 0))
            index = int(getattr(var, "index", 0))
            identifier = self._variable_identifier(var)
            legacy_id = ":".join([
                hex(int(func.start)), role, source_name,
                str(storage), str(index),
                str(identifier if identifier is not None else "none"),
            ])
            if legacy_id != local_id:
                legacy_by_id[legacy_id] = (var, is_parameter)
        if selector in locals_by_id:
            return locals_by_id[selector]
        if selector in legacy_by_id:
            return legacy_by_id[selector]

        matches = self._find_variables_by_name(func, selector)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            lines = [f"Ambiguous variable selector: {selector} matches {len(matches)} variables:"]
            for var, is_parameter in matches:
                role = "param" if is_parameter else "local"
                source_name = self._variable_source_name(var)
                storage = int(getattr(var, "storage", 0))
                lines.append(f"  {str(getattr(var, 'name', '<unknown>'))}  [{role}; storage={storage}; source={source_name}]")
            lines.append("retry with the full local_id from `bn local list --format json`")
            raise RuntimeError("\n".join(lines))
        raise RuntimeError(f"Variable not found: {selector}")

    def _function_size(self, func) -> int | None:
        try:
            total = getattr(func, "total_bytes", None)
            if total is not None:
                return int(total)
        except Exception:
            pass
        try:
            end = max(int(block.end) for block in list(func.basic_blocks))
            return end - int(func.start)
        except Exception:
            return None

    def _function_metadata(self, func) -> dict[str, Any]:
        func_type = getattr(func, "type", None)
        calling_convention = getattr(func, "calling_convention", None)
        if calling_convention is None and func_type is not None:
            calling_convention = getattr(func_type, "calling_convention", None)
        return_type = getattr(func, "return_type", None)
        if return_type is None and func_type is not None:
            return_type = getattr(func_type, "return_value", None)
        return {
            "prototype": str(func_type),
            "return_type": str(return_type) if return_type is not None else None,
            "calling_convention": str(calling_convention) if calling_convention is not None else None,
            "size": self._function_size(func),
        }

    def _comment_map(self, bv, func) -> dict[str, str]:
        arch = getattr(func, "arch", None)
        comments: dict[str, str] = {}
        for block in list(func.basic_blocks):
            addr = block.start
            while addr < block.end:
                text = bv.get_comment_at(addr)
                if text:
                    comments[hex(addr)] = text
                addr += max(1, self._instruction_length(bv, int(addr), arch=arch))
        return comments

    def _il_op_name(self, item) -> str:
        operation = getattr(item, "operation", None)
        name = getattr(operation, "name", None)
        if name:
            return str(name)
        return str(operation)

    def _llil_constant_value(self, expr) -> int | None:
        if expr is None:
            return None
        if self._il_op_name(expr) not in {"LLIL_CONST", "LLIL_CONST_PTR"}:
            return None
        constant = getattr(expr, "constant", None)
        if constant is not None:
            return int(constant)
        value = getattr(expr, "value", None)
        if value is None:
            return None
        nested_value = getattr(value, "value", None)
        if nested_value is not None:
            return int(nested_value)
        try:
            return int(value)
        except Exception:
            return None

    def _coerce_il_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _iter_llil_instructions(self, func) -> list[Any]:
        il = getattr(func, "low_level_il", None)
        if il is None:
            il = getattr(func, "llil", None)
        if il is None:
            return []

        instructions = []
        try:
            blocks = list(il)
        except Exception:
            blocks = list(getattr(il, "basic_blocks", []) or [])
        for block in blocks:
            try:
                instructions.extend(list(block))
            except Exception:
                continue
        instructions.sort(key=lambda item: int(getattr(item, "address", 0)))
        return instructions

    def _hlil_candidates_for_llil(self, insn) -> list[Any]:
        candidates = []
        seen: set[tuple[str, int]] = set()

        def add(candidate: Any) -> None:
            if candidate is None:
                return
            expr_index = getattr(candidate, "expr_index", None)
            marker = (type(candidate).__name__, int(expr_index) if expr_index is not None else id(candidate))
            if marker in seen:
                return
            seen.add(marker)
            candidates.append(candidate)

        for attr in ("hlils", "hlil"):
            for candidate in self._coerce_il_list(getattr(insn, attr, None)):
                add(candidate)

        mapped_mlil = getattr(insn, "mapped_medium_level_il", None)
        if mapped_mlil is not None:
            for attr in ("hlils", "hlil"):
                for candidate in self._coerce_il_list(getattr(mapped_mlil, attr, None)):
                    add(candidate)

        for mlil in self._coerce_il_list(getattr(insn, "mlils", None)):
            for attr in ("hlils", "hlil"):
                for candidate in self._coerce_il_list(getattr(mlil, attr, None)):
                    add(candidate)

        return candidates

    def _il_parent(self, instruction) -> Any | None:
        for attr in ("parent", "parent_instruction"):
            parent = getattr(instruction, attr, None)
            if parent is not None and parent is not instruction:
                return parent
        return None

    def _hlil_marker(self, instruction) -> tuple[str, int]:
        expr_index = getattr(instruction, "expr_index", None)
        return (
            type(instruction).__name__,
            int(expr_index) if expr_index is not None else id(instruction),
        )

    def _hlil_type_name(self, instruction) -> str:
        return type(instruction).__name__

    def _hlil_text_is_local(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) > 240:
            return False
        if stripped.count("\n") > 1:
            return False
        return True

    def _hlil_condition_is_meaningful(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if "\n" in stripped:
            return False
        if re.search(r"\bcond:\d", stripped):
            return False
        return True

    def _is_hlil_assignment_like(self, instruction) -> bool:
        return self._hlil_type_name(instruction) in {
            "HighLevelILAssign",
            "HighLevelILVarAssign",
            "HighLevelILVarInit",
            "HighLevelILAssignMem",
            "HighLevelILAssignUnpack",
            "HighLevelILVarDeclare",
        }

    def _is_hlil_control_flow(self, instruction) -> bool:
        return self._hlil_type_name(instruction) in {
            "HighLevelILIf",
            "HighLevelILWhile",
            "HighLevelILDoWhile",
            "HighLevelILFor",
            "HighLevelILSwitch",
            "HighLevelILCase",
        }

    def _is_hlil_hard_boundary(self, instruction) -> bool:
        if self._is_hlil_assignment_like(instruction) or self._is_hlil_control_flow(instruction):
            return True
        return self._hlil_type_name(instruction) in {
            "HighLevelILRet",
            "HighLevelILBlock",
            "HighLevelILCall",
            "HighLevelILTailcall",
        }

    def _is_hlil_trivial_wrapper(self, instruction) -> bool:
        return self._hlil_type_name(instruction) in {
            "HighLevelILCall",
            "HighLevelILSx",
            "HighLevelILZx",
            "HighLevelILLowPart",
            "HighLevelILIntToFloat",
            "HighLevelILFloatToInt",
            "HighLevelILBoolToInt",
            "HighLevelILFloatConv",
            "HighLevelILAddressOf",
            "HighLevelILAddressOfField",
            "HighLevelILArrayIndex",
        }

    def _hlil_call_roots(self, insn) -> list[Any]:
        roots = []
        seen: set[tuple[str, int]] = set()
        for candidate in self._hlil_candidates_for_llil(insn):
            current = candidate
            while current is not None:
                if self._hlil_type_name(current) == "HighLevelILCall":
                    marker = self._hlil_marker(current)
                    if marker not in seen:
                        seen.add(marker)
                        roots.append(current)
                    break
                current = self._il_parent(current)
        return roots

    def _select_local_hlil_node(self, insn) -> Any | None:
        roots = self._hlil_call_roots(insn)
        if not roots:
            return None

        for root in roots:
            current = root
            best_expression = None
            assignment_candidate = None
            seen: set[tuple[str, int]] = set()
            while current is not None:
                marker = self._hlil_marker(current)
                if marker in seen:
                    break
                seen.add(marker)

                parent = self._il_parent(current)
                if parent is None:
                    break
                if self._is_hlil_control_flow(parent):
                    break
                if self._is_hlil_assignment_like(parent):
                    text = str(parent)
                    if self._hlil_text_is_local(text):
                        assignment_candidate = parent
                    break
                if self._is_hlil_hard_boundary(parent):
                    break

                parent_text = str(parent)
                if not self._is_hlil_trivial_wrapper(parent) and self._hlil_text_is_local(parent_text):
                    best_expression = parent
                current = parent

            if best_expression is not None:
                return best_expression
            if assignment_candidate is not None:
                return assignment_candidate
        return None

    def _hlil_statement_text(self, insn) -> str | None:
        node = self._select_local_hlil_node(insn)
        if node is None:
            return None
        text = str(node)
        return text if self._hlil_text_is_local(text) else None

    def _hlil_pre_branch_condition(self, insn) -> str | None:
        current = self._select_local_hlil_node(insn)
        if current is None:
            return None

        seen: set[tuple[str, int]] = set()
        while current is not None:
            marker = self._hlil_marker(current)
            if marker in seen:
                break
            seen.add(marker)
            parent = self._il_parent(current)
            if parent is None:
                break
            if self._is_hlil_control_flow(parent):
                condition = getattr(parent, "condition", None)
                if condition is None:
                    return None
                text = str(condition).strip()
                return text if self._hlil_condition_is_meaningful(text) else None
            current = parent
        return None

    def _callsites_within_function(self, bv, callee, func, *, context: int) -> list[dict[str, Any]]:
        func_arch = getattr(func, "arch", None)
        disasm_entries = self._structured_disasm_entries(bv, func)
        index_by_addr = {
            int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
        }
        callee_address = int(callee.start)
        rows = []
        for insn in self._iter_llil_instructions(func):
            op_name = self._il_op_name(insn)
            if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST"}:
                continue
            dest_value = self._llil_constant_value(getattr(insn, "dest", None))
            if dest_value != callee_address:
                continue

            call_addr = int(getattr(insn, "address", 0))
            instruction_length = self._instruction_length(bv, call_addr, arch=func_arch)
            caller_static = call_addr + instruction_length
            disasm_index = index_by_addr.get(call_addr)
            if disasm_index is None:
                continue

            previous = [
                {
                    "address": item["address"],
                    "text": item["text"],
                }
                for item in disasm_entries[max(0, disasm_index - context) : disasm_index]
            ]
            next_instructions = [
                {
                    "address": item["address"],
                    "text": item["text"],
                }
                for item in disasm_entries[disasm_index + 1 : disasm_index + 1 + context]
            ]
            call_instruction = {
                "address": disasm_entries[disasm_index]["address"],
                "text": disasm_entries[disasm_index]["text"],
            }
            rows.append(
                {
                    "callee": {
                        "name": str(callee.name),
                        "address": hex(callee_address),
                    },
                    "containing_function": {
                        "name": str(func.name),
                        "address": hex(int(func.start)),
                    },
                    "call_addr": hex(call_addr),
                    "instruction_length": instruction_length,
                    "caller_static": hex(caller_static),
                    "call_instruction": call_instruction,
                    "previous_instructions": previous,
                    "next_instructions": next_instructions,
                    "hlil_statement": self._hlil_statement_text(insn),
                    "pre_branch_condition": self._hlil_pre_branch_condition(insn),
                }
            )
        rows.sort(key=lambda item: int(item["call_addr"], 16))
        return rows

    def _callsites(
        self,
        selector: str | None,
        callee_identifier: str,
        *,
        within_identifiers: list[Any],
        context: int = 3,
    ) -> list[dict[str, Any]]:
        if context < 0:
            raise OperationFailure("invalid_context", f"Invalid callsite context size: {context}")

        bv = self._resolve_view(selector)
        callee = self._find_function(bv, callee_identifier)
        scope_functions = self._resolve_scope_functions(bv, within_identifiers)

        rows = []
        for within_query, func in scope_functions:
            function_rows = self._callsites_within_function(bv, callee, func, context=context)
            for call_index, row in enumerate(function_rows):
                row["call_index"] = call_index
                row["within_query"] = str(within_query)
            rows.extend(function_rows)
        return rows

    def _xrefs_to_address(self, bv, address: int) -> dict[str, Any]:
        code_refs = []
        data_refs = []
        get_code_refs = getattr(bv, "get_code_refs", None)
        raw_code_refs = list(get_code_refs(address)) if callable(get_code_refs) else []
        for ref in sorted(raw_code_refs, key=lambda item: int(item.address)):
            fn = getattr(ref, "function", None)
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)}
                if fn is not None
                else None
            )
            ref_addr = int(ref.address)
            ref_arch = getattr(ref, "arch", None) or getattr(fn, "arch", None)
            code_refs.append(
                {
                    "function": fn.name if fn is not None else None,
                    "address": hex(ref_addr),
                    "caller_function": caller,
                    "kind": "code",
                    "context": self._address_context(
                        bv, ref_addr, include_disasm=True, arch=ref_arch, assume_code=True
                    ),
                }
            )
        get_data_refs = getattr(bv, "get_data_refs", None)
        raw_data_refs = list(get_data_refs(address)) if callable(get_data_refs) else []
        for ref_addr in sorted(raw_data_refs):
            ref_addr = int(ref_addr)
            functions = self._functions_containing(bv, ref_addr)
            fn = functions[0] if functions else None
            caller = (
                {"address": hex(int(fn.start)), "name": str(fn.name)}
                if fn is not None
                else None
            )
            data_refs.append(
                {
                    "function": fn.name if fn is not None else None,
                    "address": hex(ref_addr),
                    "caller_function": caller,
                    "kind": "data",
                    "context": self._address_context(bv, ref_addr),
                }
            )
        return {
            "address": hex(address),
            "target_context": self._address_context(bv, address, include_disasm=True),
            "code_refs": code_refs,
            "data_refs": data_refs,
        }

    def _parse_function_address_bounds(
        self,
        min_address: Any = None,
        max_address: Any = None,
    ) -> tuple[int | None, int | None]:
        lower = _parse_address(min_address) if min_address not in (None, "") else None
        upper = _parse_address(max_address) if max_address not in (None, "") else None
        if lower is not None and upper is not None and lower > upper:
            raise OperationFailure(
                "invalid_address_range",
                f"Invalid function address range: {hex(lower)} is greater than {hex(upper)}",
            )
        return lower, upper

    def _filtered_functions(
        self,
        bv,
        *,
        min_address: Any = None,
        max_address: Any = None,
    ) -> list[Any]:
        lower, upper = self._parse_function_address_bounds(min_address, max_address)
        functions = []
        for fn in list(bv.functions):
            address = int(fn.start)
            if lower is not None and address < lower:
                continue
            if upper is not None and address > upper:
                continue
            functions.append(fn)
        functions.sort(key=lambda fn: (int(fn.start), fn.name))
        return functions

    def _list_functions(
        self,
        selector: str | None,
        *,
        min_address: Any = None,
        max_address: Any = None,
        offset: int = 0,
        limit: int | None = None,
        count_only: bool = False,
    ):
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
        bv = self._resolve_view(selector)
        functions = list(self._filtered_functions(bv, min_address=min_address, max_address=max_address))
        if count_only:
            return {"count": len(functions)}
        items = [
            {"name": fn.name, "address": hex(fn.start), "raw_name": getattr(fn, "raw_name", fn.name)}
            for fn in functions
        ]
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def _search_functions(
        self,
        selector: str | None,
        query: str,
        *,
        regex: bool = False,
        exact: bool = False,
        min_address: Any = None,
        max_address: Any = None,
        offset: int = 0,
        limit: int | None = None,
    ):
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
        bv = self._resolve_view(selector)
        items = []
        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as exc:
                raise OperationFailure("invalid_regex", f"Invalid function regex: {exc}") from exc

            def matches(name: str) -> bool:
                return bool(pattern.search(name))

        elif exact:
            needle = query.lower()

            def matches(name: str) -> bool:
                return name.lower() == needle

        else:
            needle = query.lower()

            def matches(name: str) -> bool:
                return needle in name.lower()

        for fn in self._filtered_functions(bv, min_address=min_address, max_address=max_address):
            if matches(fn.name):
                items.append({"name": fn.name, "address": hex(fn.start), "raw_name": getattr(fn, "raw_name", fn.name)})
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def _function_signature(self, func) -> str:
        """Build a C-style function signature from Binary Ninja metadata."""
        func_type = getattr(func, "type", None)
        if func_type is None:
            return func.name
        return_type = getattr(func_type, "return_value", getattr(func, "return_type", None))
        ret = str(return_type) if return_type is not None else "void"
        params = []
        for var in list(func.parameter_vars):
            params.append(f"{var.type} {var.name}")
        return f"{ret} {func.name}({', '.join(params)})"

    def _pseudo_c_text(self, func, *, addresses: bool = False) -> str:
        """Render Binary Ninja's Pseudo C for one function (GUI-equivalent).

        Walks the language-representation linear view a batch of lines at a
        time. Each line carries its own address, so the optional gutter matches
        what Binary Ninja shows in its UI. Comments are rendered inline by BN.
        """
        settings = bn.DisassemblySettings()
        # Suppress BN's built-in address column so `--addresses` controls the
        # gutter (and its format) on our side rather than doubling it. Keep the
        # explicit type casts (e.g. `*(uint8_t*)x`) that make the access width
        # legible — they are off in a default DisassemblySettings.
        settings.set_option(bn.DisassemblyOption.ShowAddress, False)
        settings.set_option(bn.DisassemblyOption.ShowTypeCasts, True)
        settings.set_option(bn.DisassemblyOption.WaitForIL, True)
        # Keep long statements (and string literals) on one line instead of
        # wrapping them into adjacent fragments — one statement per line is
        # easier to read, slice (--lines), and grep without splitting strings.
        settings.set_option(bn.DisassemblyOption.DisableLineFormatting, True)
        view_obj = bn.lineardisassembly.LinearViewObject.single_function_language_representation(func, settings)
        cursor = bn.lineardisassembly.LinearViewCursor(view_obj)
        cursor.seek_to_begin()
        out: list[str] = []
        seen_content = False
        while True:
            for line in cursor.lines:
                text = str(line.contents)
                if not text.strip():
                    # Blank separator line: keep the spacing but never emit a
                    # lone address in the gutter (decide blankness on content,
                    # not on the prefixed string).
                    out.append("")
                    continue
                if not seen_content:
                    # BN indents the function-header (signature) line by two
                    # spaces; the braces and body don't share that indent, so
                    # left-justify the header to line up with them.
                    text = text.lstrip()
                    seen_content = True
                if addresses:
                    addr = getattr(line.contents, "address", None)
                    prefix = f"{int(addr):08x}        " if addr is not None else " " * 16
                    out.append(f"{prefix}{text}")
                else:
                    out.append(text)
            if not cursor.next():
                break
        while out and not out[0]:
            out.pop(0)
        while out and not out[-1]:
            out.pop()
        return "\n".join(out)

    def _decompile_text(self, bv, func, *, addresses: bool = False) -> str:
        """Pseudo C for a function, degrading to wrapped HLIL if it is unavailable."""
        marker = ""
        try:
            text = self._pseudo_c_text(func, addresses=addresses)
        except Exception as exc:
            # Make the failure visible instead of silently returning the HLIL
            # fallback (or worse, an empty body) with ok:true.
            bn.log_warn(
                f"BN Agent Bridge: pseudo-C decompilation failed for "
                f"{getattr(func, 'name', func)}: {type(exc).__name__}: {exc}"
            )
            marker = (
                f"// bn: decompilation failed ({type(exc).__name__}: {exc}); "
                "showing HLIL fallback\n"
            )
            text = ""
        if text.strip():
            return text
        sig = self._function_signature(func)
        body = self._function_text(bv, func, view="hlil", addresses=addresses)
        if addresses:
            return f"{marker}{int(func.start):08x}        {sig}\n{body}"
        return f"{marker}{sig}\n{{\n{body}\n}}"

    def _analysis_stub_warning(self, func, text: str, *, forced: bool = False) -> str | None:
        """Warn when a decompile body is a Binary Ninja analysis stub, not a real body.

        BN skips analysis for oversized functions and renders a placeholder
        instead of a body. The authoritative signal is ``func.analysis_skipped``;
        a distinctive-phrase text match is kept as a fallback.
        """
        skipped = bool(getattr(func, "analysis_skipped", False))
        placeholder = "taking too long to analyze" in text
        if not (skipped or placeholder):
            return None
        reason = None
        try:
            raw = func.analysis_skip_reason
            # BN's AnalysisSkipReason is an IntEnum; on Python 3.11+ str() yields
            # the bare number, so prefer the member name.
            reason = getattr(raw, "name", None) or str(raw)
        except Exception:
            reason = None
        detail = f" (skip reason: {reason})" if reason else ""
        if forced:
            return (
                f"{func.name}: Binary Ninja could not complete analysis even after --force-analysis{detail}; "
                f"this decompile is still an incomplete stub, not the real function body."
            )
        return (
            f"Binary Ninja skipped analysis for {func.name}{detail}; this decompile is an incomplete stub, "
            f"not the real function body. Re-run with --force-analysis to analyze it (may be slow on large functions)."
        )

    def _force_function_analysis(self, bv, func):
        """Override a skipped function's analysis and reanalyze it in place.

        Mutates the BinaryView (skip override + reanalysis), so callers must hold
        the exclusive write lock. Returns the (possibly rebuilt) function object.
        """
        start = int(func.start)
        try:
            func.analysis_skipped = False  # setter installs NeverSkipFunctionAnalysis
        except Exception:
            pass
        try:
            func.reanalyze()
        except Exception:
            pass
        bv.update_analysis_and_wait()
        return bv.get_function_at(start) or func

    def _decompile(self, selector: str | None, identifier, *, addresses: bool = False, force_analysis: bool = False):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        forced = False
        if force_analysis and bool(getattr(func, "analysis_skipped", False)):
            func = self._force_function_analysis(bv, func)
            forced = True
        text = self._decompile_text(bv, func, addresses=addresses)
        warnings = self._render_warnings(text)
        stub = self._analysis_stub_warning(func, text, forced=forced)
        if stub:
            warnings.append(stub)
        comments = self._comment_map(bv, func)
        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "text": text,
            "comments": comments,
            "warnings": warnings,
            "analysis_skipped": bool(getattr(func, "analysis_skipped", False)),
            # `analysis_force_requested` echoes the --force-analysis flag; `analysis_forced`
            # is True only when a reanalysis actually ran this call. On a second forced
            # decompile the function is no longer skipped, so nothing reruns and
            # analysis_forced is False -- the echo tells callers the flag wasn't ignored.
            "analysis_force_requested": bool(force_analysis),
            "analysis_forced": forced,
        }

    def _function_info(self, selector: str | None, identifier):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        metadata = self._function_metadata(func)
        variables = self._list_locals(func)
        parameters = [item for item in variables if item["is_parameter"]]
        locals_only = [item for item in variables if not item["is_parameter"]]
        code_ref_count = len(list(bv.get_code_refs(func.start)))
        return {
            "function": {
                "name": func.name,
                "address": hex(func.start),
                "raw_name": getattr(func, "raw_name", func.name),
            },
            **metadata,
            "parameters": parameters,
            "locals": locals_only,
            "xref_count": code_ref_count,
        }

    def _get_prototype(self, selector: str | None, identifier):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        return {
            "function": {
                "name": func.name,
                "address": hex(func.start),
                "raw_name": getattr(func, "raw_name", func.name),
            },
            **self._function_metadata(func),
        }

    def _list_locals_for_function(self, selector: str | None, identifier):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        variables = self._list_locals(func)
        return {
            "function": {
                "name": func.name,
                "address": hex(func.start),
                "raw_name": getattr(func, "raw_name", func.name),
            },
            "locals": variables,
        }

    def _il(self, selector: str | None, identifier, view: str, ssa: bool):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        text = self._function_text(bv, func, view=view, ssa=ssa)
        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "view": view,
            "ssa": ssa,
            "text": text,
            "warnings": self._render_warnings(text),
        }

    def _disasm(self, selector: str | None, identifier):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "text": self._disasm_text(bv, func),
        }

    # ------------------------------------------------------------------
    # Structured data-flow primitives (def-use / value-set / call graph)
    # ------------------------------------------------------------------

    def _il_function_for(self, func, view: str, ssa: bool):
        attr = {"hlil": "hlil", "mlil": "mlil", "llil": "llil"}.get(view, "mlil")
        il = getattr(func, attr, None)
        if il is None:
            raise OperationFailure("unsupported", f"function has no {view.upper()}")
        if ssa:
            ssa_form = getattr(il, "ssa_form", None)
            if ssa_form is not None:
                il = ssa_form
        return il

    def _ssa_var_entry(self, v) -> dict[str, Any]:
        """Serialize an SSAVariable or plain Variable consistently.

        SSA vars expose ``.var`` (-> Variable) and ``.version``; AddressOf
        targets surface as plain Variables (no version) in ``vars_read``.
        """
        base = getattr(v, "var", v)
        version = getattr(v, "version", None)
        name = str(getattr(base, "name", base))
        entry: dict[str, Any] = {
            "name": name,
            "version": int(version) if version is not None else None,
            "ssa": f"{name}#{version}" if version is not None else name,
            "type": str(getattr(base, "type", "")) or None,
            "identifier": self._variable_identifier(base),
        }
        return entry

    def _collect_ssa_vars(self, il) -> dict[tuple[str, int], Any]:
        found: dict[tuple[str, int], Any] = {}
        try:
            items = list(il.instructions)
        except Exception:
            items = []
        for ins in items:
            for v in list(getattr(ins, "vars_read", None) or []) + list(getattr(ins, "vars_written", None) or []):
                version = getattr(v, "version", None)
                if version is None:
                    continue
                base = getattr(v, "var", v)
                found[(str(getattr(base, "name", base)), int(version))] = v
        return found

    def _resolve_ssa_variable(self, func, il, selector: str):
        index = self._collect_ssa_vars(il)
        name, sep, version = str(selector).partition("#")
        if sep and version:
            key = (name, int(version))
            if key in index:
                return index[key], None
            raise OperationFailure("unsupported", f"SSA variable not found: {selector}")
        # bare name: return the lowest-version instance plus the list of versions
        versions = sorted(v for (n, v) in index if n == name)
        if not versions:
            raise OperationFailure("unsupported", f"SSA variable not found: {selector}")
        return index[(name, versions[0])], versions

    def _structured_il(self, selector, identifier, *, view: str = "mlil", ssa: bool = True):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        il = self._il_function_for(func, view, ssa)
        instructions = []
        try:
            items = list(il.instructions)
        except Exception:
            items = []
        for ins in items:
            opn = self._il_op_name(ins)
            instructions.append({
                "il_index": int(getattr(ins, "instr_index", -1)),
                "address": hex(int(getattr(ins, "address", func.start))),
                "op": opn,
                "text": str(ins),
                "vars_read": [self._ssa_var_entry(v) for v in (getattr(ins, "vars_read", None) or [])],
                "vars_written": [self._ssa_var_entry(v) for v in (getattr(ins, "vars_written", None) or [])],
                "operands_summary": [str(o) for o in (getattr(ins, "operands", None) or [])],
                "is_call": "CALL" in opn,
            })
        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "view": view,
            "ssa": ssa,
            "instructions": instructions,
        }

    def _defuse(self, selector, identifier, var_selector: str):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        il = self._il_function_for(func, "mlil", True)
        ssa_var, other_versions = self._resolve_ssa_variable(func, il, var_selector)

        try:
            definition = il.get_ssa_var_definition(ssa_var)
        except Exception:
            definition = None
        try:
            uses = list(il.get_ssa_var_uses(ssa_var) or [])
        except Exception:
            uses = []

        def _ref(ins):
            if ins is None:
                return None
            return {
                "il_index": int(getattr(ins, "instr_index", -1)),
                "address": hex(int(getattr(ins, "address", func.start))),
                "op": self._il_op_name(ins),
                "text": str(ins),
            }

        is_phi = definition is not None and "PHI" in self._il_op_name(definition)
        phi_sources = []
        if is_phi:
            for s in (getattr(definition, "src", None) or []):
                phi_sources.append(self._ssa_var_entry(s))

        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "variable": self._ssa_var_entry(ssa_var),
            "definition": _ref(definition),
            "uses": [_ref(u) for u in uses],
            "is_phi": is_phi,
            "phi_sources": phi_sources,
            "other_versions": other_versions or [],
        }

    def _serialize_pvs(self, pvs) -> dict[str, Any] | None:
        if pvs is None:
            return None
        out: dict[str, Any] = {"raw": str(pvs)}
        t = getattr(pvs, "type", None)
        out["type"] = getattr(t, "name", None) or (str(t) if t is not None else None)

        def _coerce(v):
            try:
                return int(v)
            except Exception:
                return str(v)

        value = getattr(pvs, "value", None)
        if value is not None:
            out["value"] = _coerce(value)
        values = getattr(pvs, "values", None)
        if values:
            try:
                out["values"] = sorted(_coerce(v) for v in values)
            except Exception:
                out["values"] = [_coerce(v) for v in values]
        ranges = getattr(pvs, "ranges", None)
        if ranges:
            out["ranges"] = [
                {
                    "start": _coerce(getattr(r, "start", 0)),
                    "end": _coerce(getattr(r, "end", 0)),
                    "step": _coerce(getattr(r, "step", 1)),
                }
                for r in ranges
            ]
        return out

    def _pvs_targets(self, bv, pvs) -> list[dict[str, Any]]:
        if pvs is None:
            return []
        type_name = getattr(getattr(pvs, "type", None), "name", "") or ""
        addrs: list[int] = []
        if type_name in {"ConstantValue", "ConstantPointerValue", "ImportedAddressValue", "ExternalPointerValue"}:
            value = getattr(pvs, "value", None)
            if value is not None:
                try:
                    addrs.append(int(value))
                except Exception:
                    pass
        elif type_name == "InSetOfValues":
            for v in (getattr(pvs, "values", None) or []):
                try:
                    addrs.append(int(v))
                except Exception:
                    pass
        elif type_name == "LookupTableValue":
            mapping = getattr(pvs, "mapping", None)
            if isinstance(mapping, dict):
                for v in mapping.values():
                    try:
                        addrs.append(int(v))
                    except Exception:
                        pass
            else:
                for entry in (getattr(pvs, "table", None) or []):
                    target = getattr(entry, "to", None)
                    if target is not None:
                        try:
                            addrs.append(int(target))
                        except Exception:
                            pass
        targets = []
        for addr in addrs:
            fn = bv.get_function_at(addr) if hasattr(bv, "get_function_at") else None
            name = None
            if fn is not None:
                name = str(getattr(fn, "name", None))
            else:
                sym = bv.get_symbol_at(addr) if hasattr(bv, "get_symbol_at") else None
                name = str(getattr(sym, "name", None)) if sym is not None else None
            targets.append({"address": hex(addr), "name": name})
        return targets

    def _resolved_calls(self, selector, identifier, *, direction: str = "both", resolve_indirect: bool = True):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        result: dict[str, Any] = {"function": {"name": func.name, "address": hex(func.start)}}

        if direction in ("callees", "both"):
            il = self._il_function_for(func, "mlil", True)
            callees = []
            try:
                items = list(il.instructions)
            except Exception:
                items = []
            for ins in items:
                opn = self._il_op_name(ins)
                if "CALL" not in opn and "TAILCALL" not in opn:
                    continue
                target = _taint.const_target(getattr(ins, "dest", None))
                row = {
                    "call_addr": hex(int(getattr(ins, "address", func.start))),
                    "il_index": int(getattr(ins, "instr_index", -1)),
                }
                if target is not None:
                    fn = bv.get_function_at(target)
                    name = str(fn.name) if fn is not None else None
                    if name is None:
                        sym = bv.get_symbol_at(target) if hasattr(bv, "get_symbol_at") else None
                        name = str(sym.name) if sym is not None else None
                    row.update({"kind": "direct", "target": {"address": hex(target), "name": name}})
                else:
                    row.update({"kind": "indirect", "dest_expr": str(getattr(ins, "dest", ""))})
                    if resolve_indirect:
                        pvs = getattr(getattr(ins, "dest", None), "possible_values", None)
                        resolved = self._pvs_targets(bv, pvs)
                        row["resolved"] = resolved
                        type_name = getattr(getattr(pvs, "type", None), "name", "") or "none"
                        row["resolution"] = "possible_values" if resolved else "unresolved"
                        row["resolution_detail"] = type_name
                callees.append(row)
            result["callees"] = callees

        if direction in ("callers", "both"):
            callers = []
            seen: set[int] = set()
            for site in (list(getattr(func, "caller_sites", None) or [])):
                addr = int(getattr(site, "address", 0))
                fn = getattr(site, "function", None)
                if fn is None:
                    functions = self._functions_containing(bv, addr)
                    fn = functions[0] if functions else None
                caller = (
                    {"address": hex(int(fn.start)), "name": str(fn.name)} if fn is not None else None
                )
                callers.append({"call_addr": hex(addr), "caller": caller})
            if not callers:
                for fn in (list(getattr(func, "callers", None) or [])):
                    marker = int(getattr(fn, "start", 0))
                    if marker in seen:
                        continue
                    seen.add(marker)
                    callers.append({"caller": {"address": hex(marker), "name": str(fn.name)}})
            result["callers"] = callers

        return result

    def _possible_values(self, selector, identifier, at):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        address = _parse_address(at)
        il = self._il_function_for(func, "mlil", True)
        target_ins = None
        try:
            for ins in list(il.instructions):
                if int(getattr(ins, "address", -1)) == address:
                    target_ins = ins
                    break
        except Exception:
            target_ins = None
        instr_pvs = getattr(target_ins, "possible_values", None) if target_ins is not None else None
        src_expr = getattr(target_ins, "src", None) if target_ins is not None else None
        src_pvs = getattr(src_expr, "possible_values", None) if src_expr is not None else None
        # BN leaves a SET_VAR/STORE *instruction* value-set undetermined while the
        # SOURCE expression (the value being assigned) carries the real value-set
        # -- const / range / lookup-table. Report the source's value-set for an
        # assignment so values surface as the help promises, instead of always
        # printing UndeterminedValue; fall back to the instruction-level set when
        # there is no source or only the instruction-level set is determined (#52).
        if self._pvs_determined(src_pvs) and not self._pvs_determined(instr_pvs):
            chosen, basis = src_pvs, "source_expression"
        elif self._pvs_determined(instr_pvs):
            chosen, basis = instr_pvs, "instruction"
        elif src_pvs is not None:
            chosen, basis = src_pvs, "source_expression"
        else:
            chosen, basis = instr_pvs, "instruction"
        return {
            "function": {"name": func.name, "address": hex(func.start)},
            "at": hex(address),
            "expression": str(target_ins) if target_ins is not None else None,
            "value_basis": basis,
            "source_expression": str(src_expr) if src_expr is not None else None,
            "possible_values": self._serialize_pvs(chosen),
        }

    def _pvs_determined(self, pvs) -> bool:
        """True if a PossibleValueSet carries an actual value (not BN's
        UndeterminedValue). Used to prefer a determined source-expression
        value-set over an undetermined instruction-level one (#52)."""
        if pvs is None:
            return False
        tname = str(getattr(getattr(pvs, "type", None), "name", "") or "")
        if tname:
            return tname != "UndeterminedValue"
        return "undetermined" not in str(pvs).lower()

    def _taint(self, selector, params: dict[str, Any]):
        bv = self._resolve_view(selector)
        direction = str(params.get("direction", "forward"))
        func = self._find_function(bv, params["function"])
        models = _taint.load_models()

        def _find_variable(fn, sel):
            var, _is_param = self._find_variable_selector(fn, sel)
            return var

        engine = _taint.TaintEngine(
            bv,
            models,
            find_variable=_find_variable,
            unknown_call_policy=str(params.get("unknown_call", "conservative")),
            resolve_map=params.get("resolve_map") or {},
        )
        try:
            if direction == "forward":
                locators = [_taint.parse_locator(s) for s in (params.get("sources") or [])]
                if not locators:
                    raise _taint.TaintError("forward taint needs at least one --source")
                return engine.forward(
                    func, locators,
                    max_depth=int(params.get("max_depth", 8)),
                    enabled_sink_classes=set(params.get("enabled_sink_classes") or []),
                )
            if direction == "backward":
                locators = [_taint.parse_locator(s) for s in (params.get("sinks") or [])]
                if not locators:
                    raise _taint.TaintError("backward taint needs at least one --sink")
                return engine.backward(func, locators, max_depth=int(params.get("max_depth", 8)))
        except _taint.TaintError as exc:
            raise OperationFailure("unsupported", str(exc)) from exc
        raise OperationFailure("unsupported", f"unknown taint direction: {direction}")

    def _call_destination_value(self, insn) -> int | None:
        return self._llil_constant_value(getattr(insn, "dest", None))

    def _target_entry_for_call(self, bv, value: int | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return self._normalize_code_pointer(bv, value)

    def _il_argument_texts(self, node) -> list[str]:
        for attr in ("params", "parameters"):
            params = getattr(node, attr, None)
            if params is None:
                continue
            try:
                return [str(item) for item in list(params)]
            except Exception:
                return [str(params)]
        return []

    @staticmethod
    def _safe_int(value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    _ARG_CONSTANT_RE = re.compile(r"0x[0-9a-fA-F]+")

    def _resolve_argument_value(self, bv, text: str) -> dict[str, Any] | None:
        """Annotate a pointer-constant argument with what it points at.

        Generic: fixes std::string::append literals, log format strings, RTTI
        names, and service identifiers in one place. Returns None for arguments
        that are not a bare hex pointer or that resolve to nothing useful.
        """
        match = self._ARG_CONSTANT_RE.fullmatch(text.strip())
        if match is None:
            return None
        address = self._safe_int(int(match.group(0), 16))
        if not address:
            return None
        context = self._address_context(bv, address)
        resolved: dict[str, Any] = {"address": hex(address), "kind": context.get("kind")}
        string = context.get("string")
        if string:
            resolved["string"] = string.get("value")
            if string.get("encoding") and string.get("encoding") != "ascii":
                resolved["encoding"] = string["encoding"]
            if string.get("truncated"):
                resolved["truncated"] = True
        symbol = context.get("symbol")
        if symbol and symbol.get("name"):
            resolved["symbol"] = symbol["name"]
        function = context.get("function")
        if function and function.get("name"):
            resolved["function"] = function["name"]
        sections = context.get("sections")
        if sections:
            resolved["section"] = sections[0].get("name")
        if not any(key in resolved for key in ("string", "symbol", "function")):
            return None
        return resolved

    def _call_arguments(self, bv, insn, call_addr: int) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Pick one primary argument source and quarantine uncertain extras.

        One LLIL call can map to several HLIL call expressions (BN folds adjacent
        statements); blindly merging their params attributes another call's
        arguments to this one. Prefer the single HLIL call whose address matches
        this call site; if that is ambiguous fall back to MLIL, then LLIL. Other
        candidates are returned separately (JSON-only, not shown in text).
        """
        roots = self._hlil_call_roots(insn)
        chosen = None
        matched = [r for r in roots if self._safe_int(getattr(r, "address", None)) == int(call_addr)]
        if len(matched) == 1:
            chosen = matched[0]
        elif len(roots) == 1:
            chosen = roots[0]

        mlil = getattr(insn, "mapped_medium_level_il", None)
        if chosen is not None:
            source, texts = "hlil", self._il_argument_texts(chosen)
        elif mlil is not None:
            source, texts = "mlil", self._il_argument_texts(mlil)
        else:
            source, texts = "llil", self._il_argument_texts(insn)

        primary: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            entry: dict[str, Any] = {"index": index, "text": text}
            resolved = self._resolve_argument_value(bv, text)
            if resolved is not None:
                entry["resolved"] = resolved
            primary.append(entry)

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = {(source, e["index"], e["text"]) for e in primary}

        def add_candidates(candidate_source: str, candidate_texts: list[str]) -> None:
            for index, text in enumerate(candidate_texts):
                marker = (candidate_source, index, text)
                if marker in seen:
                    continue
                seen.add(marker)
                candidates.append({"source": candidate_source, "index": index, "text": text})

        add_candidates("llil", self._il_argument_texts(insn))
        if mlil is not None:
            add_candidates("mlil", self._il_argument_texts(mlil))
        for root in roots:
            if root is chosen:
                continue
            add_candidates("hlil", self._il_argument_texts(root))
        return source, primary, candidates

    def _function_call_evidence(self, bv, func, *, context: int) -> list[dict[str, Any]]:
        disasm_entries = self._structured_disasm_entries(bv, func)
        index_by_addr = {
            int(item["_address_int"]): index for index, item in enumerate(disasm_entries)
        }
        calls = []
        for insn in self._iter_llil_instructions(func):
            op_name = self._il_op_name(insn)
            if op_name not in {
                "LLIL_CALL",
                "LLIL_CALL_STACK_ADJUST",
                "LLIL_TAILCALL",
            }:
                continue
            call_addr = int(getattr(insn, "address", 0))
            disasm_index = index_by_addr.get(call_addr)
            previous: list[dict[str, Any]] = []
            next_instructions: list[dict[str, Any]] = []
            call_instruction = self._disasm_entry(bv, call_addr, arch=getattr(func, "arch", None))
            if disasm_index is not None:
                previous = [
                    {"address": item["address"], "text": item["text"]}
                    for item in disasm_entries[max(0, disasm_index - context) : disasm_index]
                ]
                next_instructions = [
                    {"address": item["address"], "text": item["text"]}
                    for item in disasm_entries[disasm_index + 1 : disasm_index + 1 + context]
                ]
                call_instruction = {
                    "address": disasm_entries[disasm_index]["address"],
                    "text": disasm_entries[disasm_index]["text"],
                }

            mlil = getattr(insn, "mapped_medium_level_il", None)
            dest_value = self._call_destination_value(insn)
            target = self._target_entry_for_call(bv, dest_value)
            arg_source, arguments, argument_candidates = self._call_arguments(bv, insn, call_addr)
            calls.append(
                {
                    "address": hex(call_addr),
                    "operation": op_name,
                    "direct": dest_value is not None,
                    "target": target,
                    "llil": str(insn),
                    "mlil": str(mlil) if mlil is not None else None,
                    "hlil_statement": self._hlil_statement_text(insn),
                    "pre_branch_condition": self._hlil_pre_branch_condition(insn),
                    "argument_source": arg_source,
                    "arguments": arguments,
                    "argument_candidates": argument_candidates,
                    "call_instruction": call_instruction,
                    "previous_instructions": previous,
                    "next_instructions": next_instructions,
                }
            )
        return calls

    def _function_thunk_summary(self, bv, func) -> dict[str, Any]:
        sections = self._sections_at(bv, int(func.start))
        if any("plt" in str(section.get("name", "")).lower() for section in sections):
            return {
                "is_candidate": True,
                "reason": "function starts in a PLT/import trampoline section",
                "target": None,
                "sections": sections,
            }

        llil = [
            insn
            for insn in self._iter_llil_instructions(func)
            if self._il_op_name(insn) not in {"LLIL_NOP", "LLIL_UNDEF"}
        ]
        result: dict[str, Any] = {
            "is_candidate": False,
            "reason": None,
            "target": None,
            "sections": sections,
        }
        if not llil or len(llil) > 3:
            return result
        for insn in llil:
            op_name = self._il_op_name(insn)
            if op_name not in {"LLIL_JUMP", "LLIL_TAILCALL", "LLIL_CALL", "LLIL_CALL_STACK_ADJUST"}:
                continue
            target = self._target_entry_for_call(bv, self._call_destination_value(insn))
            if target is None:
                continue
            result.update(
                {
                    "is_candidate": True,
                    "reason": f"small function with {op_name.lower()} to another address",
                    "target": target,
                }
            )
            return result

        try:
            text = self._decompile_text(bv, func)
        except Exception:
            text = ""
        if "/* tailcall */" in text and len(llil) <= 3:
            result.update(
                {
                    "is_candidate": True,
                    "reason": "small function rendered as a pseudo-C tailcall",
                }
            )
        return result

    def _function_evidence(self, selector: str | None, identifier, *, context: int = 2):
        if context < 0:
            raise OperationFailure("invalid_context", f"Invalid evidence context size: {context}")
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        text = self._decompile_text(bv, func)
        return {
            "function": {
                "name": func.name,
                "address": hex(func.start),
                "raw_name": getattr(func, "raw_name", func.name),
            },
            **self._function_metadata(func),
            "thunk": self._function_thunk_summary(bv, func),
            "calls": self._function_call_evidence(bv, func, context=context),
            "warnings": self._render_warnings(text),
        }

    def _pointer_table_for_view(
        self,
        bv,
        start: int,
        *,
        entries: int,
        stride_size: int,
        stop_after_invalid: int | None = None,
    ) -> dict[str, Any]:
        pointer_size = self._pointer_size(bv)
        rows = []
        warnings = []
        invalid_run = 0
        for index in range(entries):
            entry_address = start + index * stride_size
            value = self._read_pointer_value(bv, entry_address, size=pointer_size)
            if value is None:
                rows.append(
                    {
                        "index": index,
                        "entry_address": hex(entry_address),
                        "value": None,
                        "readable": False,
                    }
                )
                invalid_run += 1
                if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
                    warnings.append(
                        f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
                    )
                    break
                continue
            target = self._normalize_code_pointer(bv, value)
            if target["plausible"] or target["status"] == "null":
                invalid_run = 0
            else:
                invalid_run += 1
            rows.append(
                {
                    "index": index,
                    "entry_address": hex(entry_address),
                    "value": hex(value),
                    "readable": True,
                    "plausible": bool(target["plausible"]),
                    "target": target,
                }
            )
            if stop_after_invalid is not None and invalid_run >= stop_after_invalid:
                warnings.append(
                    f"stopped after {invalid_run} unreadable/implausible entries at {hex(entry_address)}"
                )
                break

        table_context = self._address_context(bv, start)
        segment = table_context.get("segment")
        section_names = [
            str(section.get("name", "")).lower()
            for section in list(table_context.get("sections") or [])
            if isinstance(section, dict)
        ]
        code_like_section = any(
            name in {".text", "__text"} or name.startswith(".plt")
            for name in section_names
        )
        data_like_section = any(
            marker in name
            for name in section_names
            for marker in ("data", "rodata", "got", "rdata", "bss", "init_array", "fini_array", "ctors", "dtors")
        )
        if isinstance(segment, dict) and segment.get("executable") and (code_like_section or not data_like_section):
            warnings.append("table start is in an executable segment; this may be code, not a pointer table")
        non_null_rows = [
            row for row in rows
            if row.get("readable") and row.get("value") not in {None, "0x0"}
        ]
        plausible_rows = [row for row in non_null_rows if row.get("plausible")]
        if non_null_rows and not plausible_rows:
            warnings.append("no non-null entries resolve to mapped addresses; low confidence pointer table")
        elif non_null_rows and len(plausible_rows) < len(non_null_rows):
            warnings.append(
                f"{len(non_null_rows) - len(plausible_rows)} non-null entries do not resolve to mapped addresses"
            )
        interior_function_rows = [
            row for row in non_null_rows
            if isinstance(row.get("target"), dict)
            and isinstance(row["target"].get("function"), dict)
            and row["target"]["function"].get("exact_start") is False
        ]
        if interior_function_rows:
            warnings.append(
                f"{len(interior_function_rows)} entries resolve inside functions but not at function starts"
            )
        return {
            "address": hex(start),
            "pointer_size": pointer_size,
            "stride": stride_size,
            "context": table_context,
            "entries": rows,
            "warnings": warnings,
        }

    def _pointer_table(self, selector: str | None, address, *, entries: int = 16, stride=None):
        if entries < 0:
            raise OperationFailure("invalid_entries", f"Invalid table entry count: {entries}")
        bv = self._resolve_view(selector)
        start = _parse_address(address)
        pointer_size = self._pointer_size(bv)
        stride_size = _parse_address(stride) if stride not in (None, "") else pointer_size
        if stride_size <= 0:
            raise OperationFailure("invalid_stride", f"Invalid table stride: {stride_size}")
        return self._pointer_table_for_view(
            bv,
            start,
            entries=entries,
            stride_size=stride_size,
        )

    def _message_lens(self, selector: str | None, query: str, *, limit: int = 20, table_entries: int = 6):
        limit = _validate_count(limit, label="limit", minimum=1)
        table_entries = _validate_count(table_entries, label="table_entries", minimum=0)
        bv = self._resolve_view(selector)
        needle = query.lower()
        matches = []
        total_matched = 0
        for item in list(getattr(bv, "strings", [])):
            value = str(getattr(item, "value", ""))
            if needle and needle not in value.lower():
                continue
            # Count every match so the reported total is honest, but only build
            # the expensive per-match evidence (xrefs + pointer tables) for the
            # first `limit` matches that are actually returned.
            total_matched += 1
            if len(matches) >= limit:
                continue
            address = int(getattr(item, "start", 0))
            xrefs = self._xrefs_to_address(bv, address)
            metadata_tables = []
            for ref in list(xrefs.get("data_refs") or [])[:3]:
                try:
                    ref_addr = _parse_address(ref["address"])
                except Exception:
                    continue
                start = max(0, ref_addr - self._pointer_size(bv) * 2)
                metadata_tables.append(
                    self._pointer_table_for_view(
                        bv,
                        start,
                        entries=table_entries,
                        stride_size=self._pointer_size(bv),
                        stop_after_invalid=1,
                    )
                )

            matches.append(
                {
                    "type_string": {
                        "address": hex(address),
                        "value": value,
                        "length": int(getattr(item, "length", len(value))),
                        "context": self._address_context(bv, address),
                    },
                    "xrefs": xrefs,
                    "metadata_table_windows": metadata_tables,
                }
            )
        return {
            "query": query,
            "matches": matches,
            "count": len(matches),
            "total": total_matched,
            "truncated": total_matched > len(matches),
        }

    def _iter_il_instructions(self, il_func):
        if il_func is None:
            return []
        instructions = []
        try:
            blocks = list(il_func)
        except Exception:
            blocks = list(getattr(il_func, "basic_blocks", []) or [])
        for block in blocks:
            try:
                instructions.extend(list(block))
            except Exception:
                continue
        return instructions

    @staticmethod
    def _ssa_vars_from(vars_list: list) -> list[SSAVariable]:
        return [v for v in vars_list if isinstance(v, SSAVariable)]

    def _build_backward_trace(
        self,
        bv,
        ssa_func,
        initial_vars: list,
        max_depth: int,
        *,
        interprocedural: bool = False,
        ip_depth: int = 2,
        view: str = "mlil",
        _call_depth: int = 0,
        base_depth: int = 0,
    ) -> list[dict[str, Any]]:
        """Recursively walk SSA use-def chains backward, optionally crossing call boundaries."""
        if _call_depth > 10:
            return []  # Safety: prevent runaway recursion
        trace: list[dict[str, Any]] = []
        # Each worklist item carries its def-use distance from the seed so the
        # reported "depth" is the real graph depth (operands of one definition
        # share a depth) rather than a sequential append index. base_depth
        # offsets a callee sub-walk so its depths continue from the call site.
        worklist: list[tuple[Any, int]] = [(v, 0) for v in initial_vars]
        visited: set[Any] = set()

        while worklist and len(trace) < max_depth:
            ssa_var, node_depth = worklist.pop(0)
            depth = base_depth + node_depth
            if not isinstance(ssa_var, SSAVariable):
                trace.append({
                    "ssa_var": str(ssa_var),
                    "depth": depth,
                    "terminates": True,
                    "reason": "undefined_or_global",
                })
                continue
            if ssa_var in visited:
                continue
            visited.add(ssa_var)

            try:
                def_insn = ssa_func.get_ssa_var_definition(ssa_var)
            except AttributeError:
                if interprocedural:
                    continue
                raise OperationFailure(
                    "no_ssa_trace",
                    f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
                )
            entry: dict[str, Any] = {
                "ssa_var": str(ssa_var),
                "depth": depth,
            }

            if def_insn is None:
                # No reaching definition: a real parameter, or an undefined
                # local / global. Only claim "function parameter" when it
                # actually is one; otherwise stay neutral (don't mislead
                # provenance slices).
                entry["terminates"] = True
                entry["reason"] = (
                    "function_parameter"
                    if self._is_parameter_ssa_var(ssa_func, ssa_var)
                    else "undefined_or_global"
                )
                trace.append(entry)
                continue

            entry["address"] = hex(int(getattr(def_insn, "address", 0)))
            entry["il_text"] = str(def_insn)
            entry["operation"] = self._il_op_name(def_insn)

            def_op = entry["operation"]
            if "CALL" in def_op or "JUMP" in def_op:
                if interprocedural and ip_depth > 0:
                    callee = self._resolve_callee(bv, def_insn)
                    if callee is not None:
                        callee_mlil = getattr(callee, "medium_level_il", None)
                        if callee_mlil is not None and callee_mlil.ssa_form is not None:
                            if hasattr(callee_mlil.ssa_form, "get_ssa_var_definition"):
                                callee_ret_vars = self._find_return_vars(callee_mlil.ssa_form, bv)
                                if callee_ret_vars:
                                    entry["cross_function"] = True
                                    entry["callee"] = callee.name
                                    entry["terminates"] = False
                                    entry["reason"] = "cross_function"
                                    trace.append(entry)
                                    callee_trace = self._build_backward_trace(
                                        bv, callee_mlil.ssa_form, callee_ret_vars,
                                        max_depth - len(trace),
                                        interprocedural=True,
                                        ip_depth=ip_depth - 1,
                                        view=view,
                                        _call_depth=_call_depth + 1,
                                        base_depth=depth + 1,
                                    )
                                    for ct in callee_trace:
                                        ct.setdefault("function_context", callee.name)
                                    trace.extend(callee_trace)
                                    continue
                entry["terminates"] = True
                entry["reason"] = "call_or_jump_boundary"
                trace.append(entry)
                continue

            if "LOAD" in def_op:
                entry["terminates"] = True
                entry["reason"] = "memory_load"
                trace.append(entry)
                for rv in self._ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
                    if rv not in visited:
                        worklist.append((rv, node_depth + 1))
                continue

            entry["terminates"] = False
            entry["reason"] = None
            trace.append(entry)

            for rv in self._ssa_vars_from(getattr(def_insn, "vars_read", []) or []):
                if rv not in visited:
                    worklist.append((rv, node_depth + 1))

        return trace

    def _is_parameter_ssa_var(self, ssa_func, ssa_var) -> bool:
        """True if *ssa_var* (an SSA variable with no reaching definition) is a
        formal parameter of the function, vs. an undefined local or a global.

        Matched against the source function's ``parameter_vars`` by identifier
        first, then by base name. Returns False whenever parameter information
        is unavailable, so the caller falls back to a neutral label rather than
        guessing 'parameter'."""
        source = getattr(ssa_func, "source_function", None)
        params = list(getattr(source, "parameter_vars", []) or [])
        if not params:
            return False
        base = getattr(ssa_var, "var", ssa_var)
        ident = getattr(base, "identifier", None)
        base_name = str(ssa_var).split("#")[0]
        for param in params:
            if ident is not None and getattr(param, "identifier", None) == ident:
                return True
            if base_name and str(getattr(param, "name", param)) == base_name:
                return True
        return False

    def _resolve_callee(self, bv, call_insn):
        """Resolve a call instruction's callee to a BN function, or None.

        Thin wrapper over the canonical resolver in ``taint_engine``; follows
        thunks/veneers (single-instruction tailcalls) to their real target so
        interprocedural tracing works through PLT stubs and GCC thunks.
        """
        return _taint.resolve_call_target(bv, call_insn, follow_thunks=True).function

    def _resolve_thunk(self, bv, fn):
        """If fn is a single-instruction tailcall thunk, return its real target
        (delegates to ``taint_engine.follow_thunk``)."""
        return _taint.follow_thunk(bv, fn)

    @staticmethod
    def _extract_dest_address(bv, dest):
        """Numeric address of a call/tailcall destination expression (delegates
        to ``taint_engine.extract_dest_address``)."""
        return _taint.extract_dest_address(bv, dest)

    def _find_return_vars(self, ssa_func, bv=None, _visited=None) -> list[SSAVariable]:
        """Find SSA variables that feed into RET instructions in a function.

        For functions that only contain a TAILCALL (PLT stubs, thunks), follows
        the tailcall to the real implementation and returns its return vars.
        """
        ret_vars: list[SSAVariable] = []
        has_ret = False
        for block in getattr(ssa_func, "basic_blocks", []) or []:
            for insn in block:
                op_name = self._il_op_name(insn)
                if op_name == "MLIL_RET":
                    has_ret = True
                    found = self._ssa_vars_from(getattr(insn, "vars_read", []) or [])
                    if not found:
                        src = getattr(insn, "src", []) or []
                        for s in src:
                            var = getattr(s, "var", None)
                            if var is not None and isinstance(var, SSAVariable):
                                found.append(var)
                    if not found:
                        dest = getattr(insn, "dest", None)
                        if dest is not None:
                            found = self._ssa_vars_from([dest] if not isinstance(dest, list) else dest)
                    if not found:
                        non_ssa = getattr(insn, "non_ssa_form", None)
                        if non_ssa is not None:
                            found = self._ssa_vars_from(getattr(non_ssa, "vars_read", []) or [])
                    ret_vars.extend(found)
        # No MLIL_RET found — try to follow TAILCALL to real implementation
        if not has_ret and bv is not None:
            if _visited is None:
                _visited = set()
            fn_key = id(ssa_func)
            if fn_key in _visited:
                return ret_vars  # Cycle detected
            _visited.add(fn_key)
            for block in getattr(ssa_func, "basic_blocks", []) or []:
                for insn in block:
                    if "TAILCALL" not in self._il_op_name(insn):
                        continue
                    dest = getattr(insn, "dest", None)
                    if dest is None:
                        break
                    fn_source = getattr(ssa_func, "source_function", None)
                    source_start = getattr(fn_source, "start", None)
                    addr = self._extract_dest_address(bv, dest)
                    if addr is not None:
                        if source_start is not None and (addr == source_start or addr & ~1 == source_start):
                            break  # Self-loop (PLT stub → itself)
                        target = bv.get_function_at(addr)
                        if target is None:
                            target = bv.get_function_at(addr & ~1)
                        if target is not None and source_start is not None and target.start == source_start:
                            break  # Self-loop via different resolution path
                        if target is not None:
                            callee_mlil = getattr(target, "medium_level_il", None)
                            if callee_mlil and callee_mlil.ssa_form:
                                return self._find_return_vars(callee_mlil.ssa_form, bv, _visited)
                    break  # Only try the first instruction
        return ret_vars

    def _backward_slice(
        self,
        selector: str | None,
        identifier: str,
        address: str,
        *,
        arg_index: int = 0,
        view: str = "mlil",
        max_depth: int = 50,
        interprocedural: bool = False,
        ip_depth: int = 2,
    ) -> dict[str, Any]:
        if max_depth < 1:
            raise OperationFailure("invalid_max_depth", f"Invalid max_depth: {max_depth}")
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        target_addr = _parse_address(address)

        il_name = {"mlil": "medium_level_il", "llil": "low_level_il", "hlil": "high_level_il"}.get(view)
        if il_name is None:
            raise OperationFailure("invalid_view", f"Unsupported IL view: {view}")

        il_func = getattr(func, il_name, None)
        if il_func is None:
            raise OperationFailure("no_il", f"Function {func.name} has no {view}")
        ssa_func = il_func.ssa_form
        if ssa_func is None:
            raise OperationFailure("no_ssa", f"Function {func.name} has no {view} SSA form")

        if not hasattr(ssa_func, "get_ssa_var_definition"):
            raise OperationFailure(
                "no_ssa_trace",
                f"{view} SSA form does not support get_ssa_var_definition (MLIL or HLIL required)",
            )

        ssa_instructions = self._iter_il_instructions(ssa_func)
        call_insn = None
        for insn in ssa_instructions:
            if int(getattr(insn, "address", 0)) != target_addr:
                continue
            # "CALL" also matches TAILCALL/SYSCALL op names.
            if "CALL" in self._il_op_name(insn):
                call_insn = insn
                break

        if call_insn is None:
            hint = ""
            if view != "mlil":
                hint = " (try --view mlil, which has the broadest call coverage)"
            raise OperationFailure(
                "instruction_not_found",
                f"No call instruction at {address} in {func.name}{hint}",
            )

        params = list(getattr(call_insn, "params", []) or [])
        if not params:
            raise OperationFailure(
                "no_params",
                f"Call at {address} has no exposed parameters in {view}",
            )
        if arg_index < 0 or arg_index >= len(params):
            raise OperationFailure(
                "invalid_arg_index",
                f"Argument index {arg_index} out of range (0..{len(params) - 1})",
            )

        param_expr = params[arg_index]
        initial_vars: list[Any] = self._ssa_vars_from(getattr(param_expr, "vars_read", []) or [])

        trace = self._build_backward_trace(
            bv, ssa_func, initial_vars, max_depth,
            interprocedural=interprocedural,
            ip_depth=ip_depth,
            view=view,
        )

        return {
            "function": func.name,
            "function_address": hex(func.start),
            "target_address": hex(target_addr),
            "arg_index": arg_index,
            "view": view,
            "interprocedural": interprocedural,
            "ip_depth": ip_depth if interprocedural else 0,
            "truncated": len(trace) >= max_depth,
            "step_count": len(trace),
            "trace": trace,
        }

    def _xrefs(self, selector: str | None, identifier):
        bv = self._resolve_view(selector)
        try:
            address = _parse_address(identifier)
        except Exception:
            try:
                address = self._find_function(bv, identifier).start
            except RuntimeError as exc:
                # An ambiguous identifier is actionable as-is; replacing it
                # with "not found / not an import symbol" would be misleading.
                # Only fall back to import-symbol lookup for genuine misses.
                if "Ambiguous" in str(exc):
                    raise
                return self._xrefs_import_symbol(bv, identifier)
        return self._xrefs_to_address(bv, address)

    @staticmethod
    def _import_symbol_name(sym) -> str:
        """Preferred display name for an import symbol."""
        return str(
            getattr(sym, "short_name", None)
            or getattr(sym, "full_name", None)
            or sym.name
        )

    def _find_import_symbol(self, bv, name: str):
        needle = name.lower()
        for attr_name, kind in self._IMPORT_SYMBOL_TYPES:
            sym_type = getattr(bn.SymbolType, attr_name, None)
            if sym_type is None:
                continue
            for sym in list(bv.get_symbols_of_type(sym_type)):
                if self._import_symbol_name(sym).lower() == needle:
                    return sym
        return None

    def _xrefs_import_symbol(self, bv, identifier: str) -> dict[str, Any]:
        sym = self._find_import_symbol(bv, identifier)
        if sym is None:
            available: list[str] = []
            for attr_name, kind in self._IMPORT_SYMBOL_TYPES:
                sym_type = getattr(bn.SymbolType, attr_name, None)
                if sym_type is None:
                    continue
                for s in list(bv.get_symbols_of_type(sym_type)):
                    available.append(self._import_symbol_name(s))
            suggestions = difflib.get_close_matches(identifier, sorted(set(available)), n=5, cutoff=0.5)
            msg = f"Function not found: {identifier}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}"
            msg += " Not found as an import symbol either. Use 'bn imports' to see available imports."
            raise RuntimeError(msg)

        sym_address = int(sym.address)
        result = self._xrefs_to_address(bv, sym_address)
        result["import_resolved"] = True
        result["import_name"] = str(identifier)

        if not result.get("code_refs"):
            manual = self._scan_for_calls_to(bv, sym_address)
            if manual:
                result["code_refs"] = manual
                result["code_refs_scanned"] = True

        return result

    def _scan_for_calls_to(self, bv, target_address: int) -> list[dict[str, Any]]:
        code_refs = []
        seen: set[int] = set()
        for fn in list(bv.functions):
            for insn in self._iter_llil_instructions(fn):
                op_name = self._il_op_name(insn)
                if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL"}:
                    continue
                dest_value = self._llil_constant_value(getattr(insn, "dest", None))
                if dest_value != target_address:
                    continue
                ref_addr = int(getattr(insn, "address", 0))
                if ref_addr in seen:
                    continue
                seen.add(ref_addr)
                fn_arch = getattr(fn, "arch", None)
                code_refs.append({
                    "function": str(fn.name),
                    "address": hex(ref_addr),
                    "caller_function": {
                        "address": hex(int(fn.start)),
                        "name": str(fn.name),
                    },
                    "kind": "code",
                    "context": self._address_context(
                        bv, ref_addr, include_disasm=True, arch=fn_arch, assume_code=True
                    ),
                })
        code_refs.sort(key=lambda item: int(item["address"], 16))
        return code_refs

    def _resolve_type_field(self, bv, field_spec: str):
        type_name, sep, field_name = str(field_spec).rpartition(".")
        if not sep or not type_name or not field_name:
            raise RuntimeError("Field selector must be in the form Struct.field")

        resolved_name, type_obj = self._find_type(bv, type_name)
        members = getattr(type_obj, "members", None)
        if members is None:
            raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")

        member_list = list(members)

        def field_info(member, index: int):
            return {
                "type_name": resolved_name,
                "field_name": str(getattr(member, "name", "")) or field_name,
                "offset": int(getattr(member, "offset", 0)),
                "member_index": index,
                "field_type": str(getattr(member, "type", "")),
            }

        for index, member in enumerate(member_list):
            if str(getattr(member, "name", "")) != field_name:
                continue
            return field_info(member, index)

        folded_matches = [
            (index, member)
            for index, member in enumerate(member_list)
            if str(getattr(member, "name", "")).lower() == field_name.lower()
        ]
        if len(folded_matches) == 1:
            index, member = folded_matches[0]
            return field_info(member, index)

        try:
            requested_offset = _parse_address(field_name)
        except Exception:
            requested_offset = None
        if requested_offset is not None:
            for index, member in enumerate(member_list):
                if int(getattr(member, "offset", 0)) != requested_offset:
                    continue
                return field_info(member, index)
            raise RuntimeError(f"Field not found: {resolved_name}.0x{requested_offset:x}")

        available = [str(getattr(member, "name", "")) for member in member_list if str(getattr(member, "name", ""))]
        suggestions = difflib.get_close_matches(field_name, available, n=5, cutoff=0.5)
        if suggestions:
            raise RuntimeError(
                f"Field not found: {resolved_name}.{field_name}. Did you mean: {', '.join(suggestions)}"
            )
        raise RuntimeError(f"Field not found: {resolved_name}.{field_name}")

    def _field_xrefs(self, selector: str | None, field_spec: str):
        bv = self._resolve_view(selector)
        field = self._resolve_type_field(bv, field_spec)

        code_refs = []
        for ref in sorted(
            list(bv.get_code_refs_for_type_field(field["type_name"], field["offset"])),
            key=lambda item: int(getattr(item, "address", 0)),
        ):
            func = getattr(ref, "func", None)
            address = int(getattr(ref, "address", 0))
            code_refs.append(
                {
                    "function": func.name if func is not None else None,
                    "address": hex(address),
                    "size": int(getattr(ref, "size", 0)),
                    "incoming_type": str(getattr(ref, "incomingType", "")) or None,
                    "disasm": bv.get_disassembly(address) or "",
                }
            )

        data_refs = []
        for address in sorted(list(bv.get_data_refs_for_type_field(field["type_name"], field["offset"]))):
            symbol = bv.get_symbol_at(address)
            # BinaryView has no get_type_at(); the data variable defined at the
            # address carries the type. The old call raised AttributeError and
            # took the whole --field query down whenever a field had data refs.
            data_var = bv.get_data_var_at(address)
            type_obj = getattr(data_var, "type", None) if data_var is not None else None
            data_refs.append(
                {
                    "address": hex(address),
                    "symbol": symbol.name if symbol is not None else None,
                    "type": str(type_obj) if type_obj is not None else None,
                }
            )

        return {
            "field": field,
            "code_refs": code_refs,
            "data_refs": data_refs,
        }

    def _types(self, selector: str | None, *, query, offset: int, limit: int):
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1)
        bv = self._resolve_view(selector)
        items = []
        needle = str(query).lower() if query else None
        for name, type_obj in list(bv.types.items()):
            entry = self._type_entry(name, type_obj)
            if needle and needle not in entry["name"].lower() and needle not in entry["decl"].lower():
                continue
            items.append(entry)
        items.sort(key=lambda item: item["name"].lower())
        return items[offset : offset + limit]

    def _find_type(self, bv, type_name: str):
        type_obj = bv.get_type_by_name(type_name)
        if type_obj is not None:
            return type_name, type_obj

        needle = str(type_name).lower()
        available: list[str] = []
        for name, candidate in list(bv.types.items()):
            if str(name).lower() == needle:
                return str(name), candidate
            available.append(str(name))

        suggestions = difflib.get_close_matches(str(type_name), available, n=5, cutoff=0.5)
        if suggestions:
            raise RuntimeError(
                f"Type not found: {type_name}. Did you mean: {', '.join(suggestions)}"
            )
        raise RuntimeError(f"Type not found: {type_name}")

    def _type_entry(self, type_name, type_obj):
        type_class = getattr(type_obj, "type_class", None)
        kind = "unknown"
        if type_class is not None:
            try:
                kind = _TYPE_CLASS_NAMES.get(int(type_class), str(type_class))
            except (TypeError, ValueError):
                kind = str(type_class)
        return {
            "name": str(type_name),
            "kind": kind,
            "decl": str(type_obj),
            "layout": self._render_type_layout(type_obj),
        }

    def _current_type_entry(self, bv, type_name: str):
        type_obj = bv.get_type_by_name(type_name)
        if type_obj is None:
            return None
        return self._type_entry(type_name, type_obj)

    def _type_info(self, selector: str | None, type_name: str, *, require_struct: bool = False):
        bv = self._resolve_view(selector)
        resolved_name, type_obj = self._find_type(bv, type_name)
        members = getattr(type_obj, "members", None)
        if require_struct and members is None:
            raise RuntimeError(f"Type is not a struct-like type: {resolved_name}")
        return self._type_entry(resolved_name, type_obj)

    _NO_CRT_PATTERNS = re.compile(
        r"^(?:"
        r"[A-Za-z]$"                                      # single letters
        r"|[a-z]{2}(?:-[A-Z]{2})?$"                        # locale codes: en, en-US
        r"|[A-Z]{2,3}$"                                    # short uppercase tokens
        r"|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)$"                # day abbreviations
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"  # month abbreviations
        r"|(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)$"
        r"|(?:January|February|March|April|June|July|August|September|October|November|December)$"
        r"|(?:AM|PM|am|pm)$"
        r"|(?:UTF-?(?:7|8|16|32)|(?:us-)?ascii|iso-\d{4}.*|euc-\w+|big5|gb\d+|shift_jis|windows-\d+|cp\d+)$"
        r")",
        re.IGNORECASE,
    )

    def _strings(self, selector: str | None, *, query, offset: int, limit: int,
                 min_length: int | None = None, section: str | None = None,
                 no_crt: bool = False, regex: bool = False):
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1)
        bv = self._resolve_view(selector)
        if bv in _quick_loaded_views:
            # In --quick mode string analysis hasn't run, so bv.strings is empty
            # and `[]` would be indistinguishable from "this binary has none".
            # Refuse with a directive instead of misleading the caller.
            raise RuntimeError(
                "Strings are not available: this target was loaded with --quick (no analysis). "
                "Run `bn refresh` to build the full string set first."
            )
        items = []
        needle = str(query) if query else None
        pattern = None
        if needle and regex:
            try:
                pattern = re.compile(needle, re.IGNORECASE)
            except re.error as exc:
                raise OperationFailure("invalid_regex", f"Invalid string regex: {exc}") from exc
        elif needle:
            needle = needle.lower()
        for item in list(getattr(bv, "strings", [])):
            value = str(getattr(item, "value", ""))
            length = int(getattr(item, "length", 0))
            address = int(getattr(item, "start", 0))
            raw_type = getattr(item, "type", "")
            try:
                string_type = _STRING_TYPE_NAMES.get(int(raw_type), str(raw_type))
            except (TypeError, ValueError):
                string_type = str(raw_type)

            if pattern is not None:
                if not pattern.search(value):
                    continue
            elif needle and needle not in value.lower():
                continue
            if min_length is not None and length < min_length:
                continue
            if section:
                secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
                if not any(getattr(s, "name", "") == section for s in secs):
                    continue
            if no_crt:
                if self._NO_CRT_PATTERNS.match(value):
                    continue
                if len(value) >= 2 and len(set(value)) == 1:
                    continue
                secs = bv.get_sections_at(address) if hasattr(bv, "get_sections_at") else []
                if any(getattr(s, "name", "") == ".text" for s in secs):
                    continue

            entry = {
                "address": hex(address),
                "length": length,
                "chars": len(value),
                "type": string_type,
                "value": value,
            }
            items.append(entry)
        items.sort(key=lambda item: (int(item["address"], 16), item["value"]))
        return items[offset : offset + limit]

    _INIT_SECTION_HINTS = (
        "init_array",
        "preinit_array",
        "fini_array",
        ".ctors",
        ".dtors",
        "__mod_init_func",
        "__mod_term_func",
    )

    def _init_arrays(self, selector: str | None, *, limit: int = 64):
        if limit < 0:
            raise OperationFailure("invalid_limit", f"Invalid init-array limit: {limit}")
        bv = self._resolve_view(selector)
        pointer_size = self._pointer_size(bv)
        sections = []
        for name, sec in getattr(bv, "sections", {}).items():
            lowered = str(name).lower()
            if not any(hint in lowered for hint in self._INIT_SECTION_HINTS):
                continue
            start = int(getattr(sec, "start", 0))
            end = int(getattr(sec, "end", 0))
            total_entries = max(0, (end - start) // pointer_size)
            shown_entries = min(total_entries, limit)
            table = self._pointer_table_for_view(
                bv,
                start,
                entries=shown_entries,
                stride_size=pointer_size,
            )
            sections.append(
                {
                    "name": str(name),
                    "start": hex(start),
                    "end": hex(end),
                    "total_entries": total_entries,
                    "shown_entries": shown_entries,
                    "truncated": total_entries > shown_entries,
                    "table": table,
                }
            )
        sections.sort(key=lambda item: int(item["start"], 16))
        return {
            "pointer_size": pointer_size,
            "sections": sections,
        }

    _IMPORT_SYMBOL_TYPES: list[tuple[str, str]] = [
        ("ImportedFunctionSymbol", "function"),
        ("ImportedDataSymbol", "data"),
        ("ImportAddressSymbol", "address"),
    ]

    # BN tags standard-ELF import symbols with these namespace sentinels rather
    # than a real shared-object name (the dynamic linker only resolves the actual
    # provider at runtime). Treat them as "no known library".
    _BN_SENTINEL_NAMESPACES: frozenset[str] = frozenset(
        {"", "BNINTERNALNAMESPACE", "BNEXTERNALNAMESPACE"}
    )

    @staticmethod
    def _needed_libraries(bv) -> list[str]:
        """DT_NEEDED shared objects this binary links against, if BN exposes them."""
        try:
            return sorted({str(lib) for lib in (getattr(bv, "libraries", None) or [])})
        except Exception:
            return []

    def _imports(self, selector: str | None, *, summary: bool = False,
                 offset: int = 0, limit: int | None = None):
        # Guard paging the same way the sibling list ops do, so a raw-socket /
        # py exec caller passing a negative offset/limit gets a clean
        # invalid_request instead of a silent Python negative-index slice (#68).
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
        bv = self._resolve_view(selector)
        needed_libraries = self._needed_libraries(bv)
        items = []
        for attr_name, kind in self._IMPORT_SYMBOL_TYPES:
            sym_type = getattr(bn.SymbolType, attr_name, None)
            if sym_type is None:
                continue
            for sym in list(bv.get_symbols_of_type(sym_type)):
                name = self._import_symbol_name(sym)
                raw_name = str(getattr(sym, "raw_name", sym.name))
                namespace = str(getattr(sym, "namespace", "") or "")
                # Only surface `library` when it's a real per-library namespace;
                # BN's sentinels become None so agents don't read them as a
                # dependency. `namespace` keeps the raw value under an honest name.
                library = namespace if namespace not in self._BN_SENTINEL_NAMESPACES else None
                items.append(
                    {
                        "name": name,
                        "address": hex(sym.address),
                        "library": library,
                        "namespace": namespace,
                        "raw_name": raw_name,
                        "kind": kind,
                    }
                )
        if summary:
            # Summary aggregates the whole import set; paging would distort the
            # counts, so it always reflects every symbol regardless of offset/limit.
            return self._imports_build_summary(items, needed_libraries)
        items.sort(key=lambda item: (item["library"] or "", item["kind"], item["name"], int(item["address"], 16)))
        if limit is not None:
            return items[offset : offset + limit]
        return items[offset:] if offset else items

    def _imports_build_summary(
        self, items: list[dict], needed_libraries: list[str] | None = None
    ) -> dict[str, Any]:
        # "namespaces" groups BN's symbol namespace (sentinels on standard ELF),
        # not a per-shared-object breakdown. The real dependency list is
        # "needed_libraries" (DT_NEEDED), which is what agents actually want.
        namespaces: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for item in items:
            ns = str(item.get("namespace", "") or "") or "(none)"
            namespaces[ns] = namespaces.get(ns, 0) + 1
            kind = str(item.get("kind", "unknown"))
            by_kind[kind] = by_kind.get(kind, 0) + 1
        return {
            "total_symbols": len(items),
            "needed_libraries": needed_libraries or [],
            "namespaces": dict(sorted(namespaces.items(), key=lambda x: -x[1])),
            "by_kind": dict(sorted(by_kind.items(), key=lambda x: -x[1])),
        }

    _SECTION_SEMANTICS_NAMES: dict[int, str] = {
        0: "DefaultSection",
        1: "ReadOnlyCode",
        2: "ReadOnlyData",
        3: "ReadWriteData",
        4: "ExternalSection",
    }

    def _sections(self, selector: str | None, *, query: str | None = None,
                  offset: int = 0, limit: int | None = None):
        bv = self._resolve_view(selector)
        items = []
        sections = getattr(bv, "sections", {})
        needle = str(query).lower() if query else None
        for name, sec in sections.items():
            if needle and needle not in name.lower():
                continue
            start = int(getattr(sec, "start", 0))
            end = int(getattr(sec, "end", 0))
            length = end - start

            raw_semantics = getattr(sec, "semantics", 0)
            try:
                semantics_int = int(raw_semantics)
            except (TypeError, ValueError):
                semantics_int = 0
            semantics = self._SECTION_SEMANTICS_NAMES.get(semantics_int, str(raw_semantics))

            entry: dict[str, Any] = {
                "name": name,
                "start": hex(start),
                "end": hex(end),
                "length": length,
                "semantics": semantics,
            }

            if hasattr(bv, "get_segment_at"):
                seg = bv.get_segment_at(start)
                if seg is not None:
                    entry["readable"] = bool(getattr(seg, "readable", None))
                    entry["writable"] = bool(getattr(seg, "writable", None))
                    entry["executable"] = bool(getattr(seg, "executable", None))

            items.append(entry)
        items.sort(key=lambda item: int(item["start"], 16))
        if limit is not None:
            return items[offset : offset + limit]
        return items[offset:] if offset else items

    @staticmethod
    def _ascii_render(data: bytes) -> str:
        return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data)

    def _read(self, selector: str | None, address, length: int):
        bv = self._resolve_view(selector)
        addr = _parse_address(address)
        if length < 0:
            raise RuntimeError(f"read length must be non-negative, got {length}")

        data = bytes(bv.read(addr, length))
        if length > 0 and not data:
            raise RuntimeError(f"Address 0x{addr:x} is not mapped (no bytes available)")

        result: dict[str, Any] = {
            "address": hex(addr),
            "length": len(data),
            "hex": data.hex(),
            "ascii": self._ascii_render(data),
        }
        if len(data) < length:
            result["requested_length"] = length
            result["short_read"] = True
            result["note"] = (
                f"short read: requested {length} bytes, only {len(data)} mapped from 0x{addr:x}"
            )
        return result

    def _is_executable_address(self, bv, addr: int) -> bool:
        is_offset_executable = getattr(bv, "is_offset_executable", None)
        if callable(is_offset_executable):
            return bool(is_offset_executable(addr))
        get_segment_at = getattr(bv, "get_segment_at", None)
        if callable(get_segment_at):
            seg = get_segment_at(addr)
            if seg is not None:
                return bool(getattr(seg, "executable", False))
        return False

    def _function_create(self, selector: str | None, address, preview: bool):
        bv = self._resolve_view(selector)
        addr = _parse_address(address)
        requested = {"op": "function_create", "address": hex(addr)}

        existing = bv.get_function_at(addr)
        if existing is not None:
            return {
                "preview": preview,
                "success": True,
                "committed": False,
                "message": "A function already starts at this address.",
                "results": [
                    {
                        "op": "function_create",
                        "status": "noop",
                        "address": hex(addr),
                        "function": str(existing.name),
                        "message": "A function already starts at this address.",
                        "requested": requested,
                    }
                ],
                "affected_functions": [],
                "affected_types": [],
            }

        # Refuse to create junk functions: the address must be mapped and live
        # inside an executable region. Auto-analysis skips exactly these handler
        # entry points (reachable only via data/function-pointer tables), so we
        # still create them on request -- but only where code can actually run.
        if len(bytes(bv.read(addr, 1))) == 0:
            raise RuntimeError(
                f"Cannot create function: address 0x{addr:x} is not mapped"
            )
        if not self._is_executable_address(bv, addr):
            raise RuntimeError(
                f"Cannot create function: address 0x{addr:x} is not inside an executable segment"
            )

        state = bv.begin_undo_actions()
        try:
            bv.add_function(addr)
            bv.update_analysis_and_wait()
            created = bv.get_function_at(addr)
            if created is None:
                reverted = self._revert_undo_safely(bv, state)
                return {
                    "preview": preview,
                    "success": False,
                    "committed": False,
                    "rolled_back": reverted,
                    "message": (
                        "Rolled back because no function was created at the address."
                        if reverted
                        else "No function was created at the address AND the rollback failed; "
                        "the view may be left partially modified."
                    ),
                    "results": [
                        {
                            "op": "function_create",
                            "status": "verification_failed",
                            "address": hex(addr),
                            "message": f"No function starts at 0x{addr:x} after analysis.",
                            "requested": requested,
                            "observed": {"address": hex(addr), "function": None},
                        }
                    ],
                    "affected_functions": [],
                    "affected_types": [],
                }

            function_name = str(created.name)
            if preview:
                bv.revert_undo_actions(state)
                message = "Preview verified and reverted."
            else:
                bv.commit_undo_actions(state)
                message = "Function created and verified in the live Binary Ninja session."
            return {
                "preview": preview,
                "success": True,
                "committed": bool(not preview),
                "message": message,
                "results": [
                    {
                        "op": "function_create",
                        "status": "verified",
                        "address": hex(addr),
                        "function": function_name,
                        "requested": requested,
                    }
                ],
                "affected_functions": [
                    {
                        "address": hex(addr),
                        "before_name": None,
                        "after_name": function_name,
                        "changed": True,
                    }
                ],
                "affected_types": [],
            }
        except Exception as exc:
            if not self._revert_undo_safely(bv, state):
                raise RuntimeError(
                    f"{exc} (additionally, rollback failed; the view may be left partially modified)"
                ) from exc
            raise

    def _get_comment(self, selector: str | None, address, function):
        bv = self._resolve_view(selector)
        if function:
            fn = self._find_function(bv, function)
            comment = bv.get_comment_at(fn.start)
            return {
                "function": fn.name,
                "address": hex(fn.start),
                "comment": comment or "",
                "has_comment": bool(comment),
            }

        if address is None:
            raise RuntimeError("comment get requires --address or --function")

        comment_address = _parse_address(address)
        comment = bv.get_comment_at(comment_address)
        return {
            "address": hex(comment_address),
            "comment": comment or "",
            "has_comment": bool(comment),
        }

    def _list_comments(
        self,
        selector: str | None,
        *,
        query: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ):
        bv = self._resolve_view(selector)
        needle = query.lower() if query else None
        items = []
        for addr in sorted(bv.address_comments):
            text = bv.address_comments[addr]
            if not text:
                continue
            if needle and needle not in text.lower():
                continue
            funcs = bv.get_functions_containing(addr)
            func_name = funcs[0].name if funcs else None
            items.append({
                "address": hex(addr),
                "function": func_name,
                "comment": text,
            })
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def _bundle_function(self, selector: str | None, identifier, out_path: str | None):
        bv = self._resolve_view(selector)
        func = self._find_function(bv, identifier)
        decompile = self._decompile_text(bv, func)
        bundle_warnings = self._render_warnings(decompile)
        stub = self._analysis_stub_warning(func, decompile)
        if stub:
            bundle_warnings.append(stub)
        bundle = {
            "target": self._target_info(selector),
            "function": {
                "name": func.name,
                "address": hex(func.start),
                "raw_name": getattr(func, "raw_name", func.name),
                "type": str(func.type),
            },
            "decompile": decompile,
            "warnings": bundle_warnings,
            "il": {
                "hlil": self._function_text(bv, func, view="hlil"),
                "mlil": self._function_text(bv, func, view="mlil"),
            },
            "disassembly": self._disasm_text(bv, func),
            "locals": self._list_locals(func),
            "comments": dict(sorted(self._comment_map(bv, func).items())),
            "xrefs": self._xrefs_to_address(bv, func.start),
        }
        artifact = _write_json_artifact(out_path, bundle)
        return artifact or bundle

    def _normalize_py_result(self, value: Any) -> tuple[Any, list[str]]:
        def normalize(item: Any) -> Any:
            if item is None or isinstance(item, (bool, int, float, str)):
                return item
            if isinstance(item, (list, tuple)):
                return [normalize(part) for part in item]
            if isinstance(item, dict):
                return {str(key): normalize(val) for key, val in item.items()}
            raise TypeError(type(item).__name__)

        try:
            return normalize(value), []
        except TypeError:
            return repr(value), ["`result` was not JSON-serializable; returned repr(result) instead."]

    def _py_exec(self, selector: str | None, script: str):
        bv = self._resolve_view(selector)
        stdout = io.StringIO()
        scope = {
            "bn": bn,
            "binaryninja": bn,
            "bv": bv,
            "result": None,
        }
        with contextlib.redirect_stdout(stdout):
            try:
                exec(script, scope, scope)
            except Exception as exc:  # noqa: BLE001 - user script errors are user-facing
                # Report every script failure the same way -- "TypeName: message".
                # Previously a ValueError surfaced as a bare message while a
                # NameError was tagged "internal error: NameError:", because only
                # some builtins are whitelisted as user-facing. The user's own
                # script raised this, so it is always a user-facing error.
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        result_value, warnings = self._normalize_py_result(scope.get("result"))
        result = {
            "stdout": stdout.getvalue(),
            "result": result_value,
            "warnings": warnings,
        }
        return result

    def _render_warnings(self, text: str) -> list[str]:
        warnings: list[str] = []
        if "__offset(" in text:
            warnings.append(
                "Decompile still contains raw __offset(...) expressions; use `bn types show` or `bn struct show` as the authoritative layout until Binary Ninja refreshes the presentation."
            )
        return warnings

    def _guess_type_affected_functions(self, bv, type_name: str, limit: int = 10):
        matches = []
        needle = type_name.lower()
        for fn in list(bv.functions):
            text = str(fn.type).lower()
            if needle in text:
                matches.append(fn)
                if len(matches) >= limit:
                    break
        return matches

    def _parse_declaration_source(self, bv, declaration: str, *, source_path: str | None = None):
        parse_result = None
        source_error: Exception | None = None
        platform = getattr(bv, "platform", None)
        if platform is not None and hasattr(platform, "parse_types_from_source"):
            kwargs: dict[str, Any] = {}
            if source_path:
                kwargs["filename"] = source_path
                kwargs["include_dirs"] = [str(Path(source_path).expanduser().resolve().parent)]
            try:
                parse_result = platform.parse_types_from_source(declaration, **kwargs)
            except Exception as exc:
                source_error = exc

        if parse_result is None:
            try:
                parse_result = bv.parse_types_from_string(declaration)
            except Exception:
                if source_error is not None:
                    raise source_error
                raise

        return {
            "types": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "types", {}).items())],
            "variables": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "variables", {}).items())],
            "functions": [(str(name), type_obj) for name, type_obj in list(getattr(parse_result, "functions", {}).items())],
        }

    def _operation_type_names(self, bv, op: dict[str, Any]) -> list[str]:
        kind = op.get("op") or "rename_symbol"
        if kind.startswith("struct_") and op.get("struct_name"):
            return [str(op["struct_name"])]
        if kind == "types_declare":
            # Tolerate a malformed op here (the pre-apply snapshot pass): a
            # missing `declaration` must surface as the precise invalid_request
            # from _apply_operation's field validation, not a raw KeyError that
            # escapes _mutation before the apply loop.
            declaration = op.get("declaration")
            if not declaration:
                return []
            return [name for name, _ in self._parse_declaration_source(
                bv,
                str(declaration),
                source_path=op.get("source_path"),
            )["types"]]
        return []

    def _guess_affected_functions(self, bv, operations: list[dict[str, Any]]):
        affected = []
        seen = set()
        for op in operations:
            kind = op.get("op") or "rename_symbol"
            functions = []
            try:
                if kind == "rename_symbol" and op.get("kind") != "data":
                    functions = [self._find_function(bv, op["identifier"])]
                elif kind in {"set_prototype", "local_rename", "local_retype"}:
                    ident = op.get("identifier") or op.get("function")
                    functions = [self._find_function(bv, ident)]
                elif kind in {"set_comment", "delete_comment"}:
                    if op.get("function"):
                        functions = [self._find_function(bv, op["function"])]
                    elif op.get("address"):
                        functions = self._functions_containing(bv, _parse_address(op["address"]))
                elif kind.startswith("struct_") or kind == "types_declare":
                    for type_name in self._operation_type_names(bv, op):
                        functions.extend(self._guess_type_affected_functions(bv, type_name))
            except Exception:
                functions = []

            for fn in functions:
                if fn is None:
                    continue
                marker = int(fn.start)
                if marker not in seen:
                    seen.add(marker)
                    affected.append(fn)
        return affected

    def _affected_type_names(self, bv, operations: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for op in operations:
            for type_name in self._operation_type_names(bv, op):
                if type_name not in seen:
                    seen.add(type_name)
                    names.append(type_name)
        return names

    def _render_type_layout(self, type_obj) -> str:
        header = str(type_obj)
        try:
            width = int(getattr(type_obj, "width", 0))
            header = f"{header} // size=0x{width:x}"
        except Exception:
            pass

        members = getattr(type_obj, "members", None)
        if members is None:
            return header

        # Enum members carry a .value (the enumerator constant) but no .offset/
        # .type, so the struct-shaped line collapses every one to
        # "0x0000: <unknown> NAME", dropping the only meaningful datum. Render the
        # value instead for enums (#54).
        tc = str(getattr(getattr(type_obj, "type_class", None), "name", "") or "")
        is_enum = "Enum" in tc

        lines = [header]
        for member in list(members):
            name = str(getattr(member, "name", "<anonymous>"))
            value = getattr(member, "value", None)
            if is_enum or (getattr(member, "offset", None) is None and value is not None):
                try:
                    ival = int(value)
                    suffix = f" (0x{ival:x})" if ival >= 0 else ""
                    lines.append(f"{name} = {ival}{suffix}")
                except Exception:
                    lines.append(f"{name} = {value}")
            else:
                try:
                    offset = int(getattr(member, "offset", 0))
                except Exception:
                    offset = 0
                member_type = str(getattr(member, "type", "<unknown>"))
                lines.append(f"0x{offset:04x}: {member_type} {name}")
        return "\n".join(lines)

    def _capture_type_snapshots(self, bv, operations: list[dict[str, Any]]):
        snapshots: dict[str, dict[str, Any]] = {}
        for type_name in self._affected_type_names(bv, operations):
            type_obj = bv.get_type_by_name(type_name)
            if type_obj is None:
                continue
            snapshots[type_name] = {
                "type_name": type_name,
                "decl": str(type_obj),
                "layout": self._render_type_layout(type_obj),
            }
        return snapshots

    def _diff_type_snapshots(self, before: dict[str, Any], after: dict[str, Any]):
        diffs = []
        for type_name in sorted(set(before) | set(after)):
            old = before.get(type_name, {"decl": "", "layout": ""})
            new = after.get(type_name, {"decl": "", "layout": ""})
            layout_diff = "\n".join(
                difflib.unified_diff(
                    old["layout"].splitlines(),
                    new["layout"].splitlines(),
                    fromfile=f"before:{type_name}",
                    tofile=f"after:{type_name}",
                    lineterm="",
                )
            )
            changed = old["decl"] != new["decl"] or old["layout"] != new["layout"]
            entry = {
                "type_name": type_name,
                "before_decl": old["decl"],
                "after_decl": new["decl"],
                "before_layout": old["layout"],
                "after_layout": new["layout"],
                "layout_diff": layout_diff,
                "changed": changed,
            }
            if not changed:
                entry["message"] = "No effective change detected"
            diffs.append(entry)
        return diffs

    def _annotate_operation_results(self, results: list[dict[str, Any]], type_diffs: list[dict[str, Any]]):
        type_changes = {item["type_name"]: item for item in type_diffs}
        annotated = []
        for result in results:
            item = dict(result)
            type_name = item.get("struct_name")
            if type_name and type_name in type_changes:
                change = type_changes[type_name]
                item["changed"] = bool(change["changed"])
                if not change["changed"]:
                    item["message"] = change["message"]
                    if item.get("status") == "verified":
                        item["status"] = "noop"
            defined_types = dict(item.get("defined_types") or {})
            if defined_types:
                changed_types = {name: bool(type_changes.get(name, {}).get("changed")) for name in defined_types}
                item["changed_types"] = changed_types
                # The authoritative change signal is the before/after layout diff
                # (changed_types), not the verify step's decl-string compare -- a
                # redeclaration of an existing NAME renders the same `struct QA`
                # decl before and after, so that compare wrongly reported `noop`
                # on a real layout change (#57). Reclassify from changed_types for
                # the success statuses only (never override a *_failed status).
                if item.get("status") in ("verified", "noop"):
                    if any(changed_types.values()):
                        item["status"] = "verified"
                    else:
                        item["status"] = "noop"
                        item["message"] = "No effective change detected"
            annotated.append(item)
        return annotated

    def _capture_function_snapshots(self, bv, functions):
        snapshots = {}
        for fn in functions:
            snapshots[int(fn.start)] = {
                "name": fn.name,
                "address": hex(fn.start),
                "text": self._function_text(bv, fn, view="hlil"),
            }
        return snapshots

    def _snippet_for_change(self, before_text: str, after_text: str, *, context_lines: int = 3, max_lines: int = 10):
        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()
        line_count = max(len(before_lines), len(after_lines))

        changed_line = None
        for index in range(line_count):
            before_line = before_lines[index] if index < len(before_lines) else None
            after_line = after_lines[index] if index < len(after_lines) else None
            if before_line != after_line:
                changed_line = index
                break

        if changed_line is None:
            return None

        start = max(0, changed_line - context_lines)
        end = min(line_count, start + max_lines)
        return {
            "start_line": start + 1,
            "before_excerpt": "\n".join(before_lines[start:end]),
            "after_excerpt": "\n".join(after_lines[start:end]),
        }

    def _diff_snapshots(self, before: dict[int, Any], after: dict[int, Any]):
        diffs = []
        snippets_added = 0
        for address in sorted(set(before) | set(after)):
            old = before.get(address, {"text": ""})
            new = after.get(address, {"text": ""})
            text_changed = old.get("text", "") != new.get("text", "")
            name_changed = old.get("name") != new.get("name")
            diff = "\n".join(
                difflib.unified_diff(
                    old["text"].splitlines(),
                    new["text"].splitlines(),
                    fromfile=f"before:{old.get('name', hex(address))}",
                    tofile=f"after:{new.get('name', hex(address))}",
                    lineterm="",
                )
            )
            if not diff and name_changed:
                diff = "\n".join(
                    [
                        f"--- before:{old.get('name', hex(address))}",
                        f"+++ after:{new.get('name', hex(address))}",
                    ]
                )
            diffs.append(
                {
                    "address": hex(address),
                    "before_name": old.get("name"),
                    "after_name": new.get("name"),
                    "changed": bool(text_changed or name_changed),
                    "diff": diff,
                }
            )
            if text_changed and snippets_added < 3:
                snippet = self._snippet_for_change(old.get("text", ""), new.get("text", ""))
                if snippet is not None:
                    diffs[-1].update(snippet)
                    snippets_added += 1
        return diffs

    def _operation_requested(self, op: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in op.items() if key != "preview"}

    def _operation_failure_result(self, op: dict[str, Any], exc: OperationFailure) -> dict[str, Any]:
        result = {
            "op": str(op.get("op") or "rename_symbol"),
            "status": exc.status,
            "message": exc.message,
            "requested": exc.requested or self._operation_requested(op),
        }
        if exc.observed:
            result["observed"] = exc.observed
        return result

    def _mark_unverified_results(self, results: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
        annotated = []
        for result in results:
            item = dict(result)
            item["status"] = "unsupported"
            item["message"] = message
            annotated.append(item)
        return annotated

    def _has_failed_results(self, results: list[dict[str, Any]]) -> bool:
        return any(item.get("status") in {"unsupported", "verification_failed"} for item in results)

    def _first_overlapping_member(self, type_obj, offset: int, width: int):
        """The first existing member whose byte range intersects the range a new
        field of *width* bytes at *offset* would occupy, else None. Width 0/unknown
        is treated as 1 byte so an exact start-offset collision is always caught.
        Used to enforce struct field set --no-overwrite (#56)."""
        members = getattr(type_obj, "members", None)
        if members is None:
            return None
        new_start = int(offset)
        new_end = new_start + max(int(width or 0), 1)
        for member in list(members):
            m_start = int(getattr(member, "offset", 0))
            try:
                m_width = int(getattr(getattr(member, "type", None), "width", 0) or 0)
            except Exception:
                m_width = 0
            m_end = m_start + max(m_width, 1)
            if new_start < m_end and m_start < new_end:
                return member
        return None

    def _find_member(self, type_obj, *, offset: int | None = None, name: str | None = None):
        members = getattr(type_obj, "members", None)
        if members is None:
            return None
        for member in list(members):
            member_offset = int(getattr(member, "offset", 0))
            member_name = str(getattr(member, "name", ""))
            if offset is not None and member_offset != int(offset):
                continue
            if name is not None and member_name != name:
                continue
            return member
        return None

    def _verify_operation(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        op = result.get("op")
        try:
            if op == "rename_symbol":
                return self._verify_rename_symbol(bv, result)
            if op == "set_comment":
                return self._verify_set_comment(bv, result)
            if op == "delete_comment":
                return self._verify_delete_comment(bv, result)
            if op == "set_prototype":
                return self._verify_set_prototype(bv, result)
            if op == "local_rename":
                return self._verify_local_rename(bv, result)
            if op == "local_retype":
                return self._verify_local_retype(bv, result)
            if op == "struct_field_set":
                return self._verify_struct_field_set(bv, result)
            if op == "struct_field_rename":
                return self._verify_struct_field_rename(bv, result)
            if op == "struct_field_delete":
                return self._verify_struct_field_delete(bv, result)
            if op == "types_declare":
                return self._verify_declared_types(bv, result)
            raise OperationFailure("unsupported", f"Unsupported verification path: {op}", requested=result.get("requested"))
        except OperationFailure as exc:
            item = dict(result)
            item["status"] = exc.status
            item["message"] = exc.message
            if exc.requested:
                item["requested"] = exc.requested
            if exc.observed:
                item["observed"] = exc.observed
            return item
        except Exception as exc:
            item = dict(result)
            item["status"] = "verification_failed"
            item["message"] = f"{type(exc).__name__}: {exc}"
            if item.get("requested") is None:
                item["requested"] = {}
            return item

    def _verify_rename_symbol(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        requested_name = str(item["new_name"])
        before_name = item.get("before_name")
        observed_name = None
        if item.get("kind") == "function":
            fn = bv.get_function_at(address)
            if fn is None:
                raise OperationFailure(
                    "verification_failed",
                    f"Function missing after rename at {item['address']}",
                    requested=item.get("requested"),
                    observed={"address": item["address"], "name": None},
                )
            observed_name = str(fn.name)
        else:
            symbol = bv.get_symbol_at(address)
            observed_name = str(symbol.name) if symbol is not None else None
        item["observed"] = {"address": item["address"], "name": observed_name}
        if observed_name != requested_name:
            raise OperationFailure(
                "verification_failed",
                f"Live rename verification failed at {item['address']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "noop" if before_name == requested_name else "verified"
        return item

    def _verify_set_comment(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        expected = str(item["requested"]["comment"])
        observed = bv.get_comment_at(address) or ""
        item["observed"] = {"address": item["address"], "comment": observed}
        if observed != expected:
            raise OperationFailure(
                "verification_failed",
                f"Live comment verification failed at {item['address']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "noop" if item.get("before_comment", "") == expected else "verified"
        return item

    def _verify_delete_comment(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        observed = bv.get_comment_at(address) or ""
        item["observed"] = {"address": item["address"], "comment": observed}
        if observed:
            raise OperationFailure(
                "verification_failed",
                f"Live comment deletion verification failed at {item['address']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "noop" if not item.get("before_comment") else "verified"
        return item

    def _verify_set_prototype(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        fn = bv.get_function_at(address)
        if fn is None:
            raise OperationFailure(
                "verification_failed",
                f"Function missing after prototype change at {item['address']}",
                requested=item.get("requested"),
                observed={"address": item["address"], "prototype": None},
            )
        observed = str(fn.type)
        item["observed"] = {"address": item["address"], "prototype": observed}
        expected = item["expected_prototype"]
        if observed != expected:
            # BN analysis may add an implicit calling convention (e.g.
            # __convention("cdecl")) that wasn't present in the parsed
            # expected type.  Normalize both before rejecting.
            if _normalize_prototype(observed) != _normalize_prototype(expected):
                raise OperationFailure(
                    "verification_failed",
                    f"Live prototype verification failed at {item['address']}",
                    requested=item.get("requested"),
                    observed=item["observed"],
                )
        item["status"] = "noop" if item.get("before_prototype") == expected else "verified"
        return item

    def _verify_local_rename(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        fn = bv.get_function_at(address)
        if fn is None:
            raise OperationFailure(
                "verification_failed",
                f"Function missing after local rename at {item['address']}",
                requested=item.get("requested"),
                observed={"address": item["address"], "variable": None},
            )
        # After analysis, variable objects may be reconstructed.  Try
        # identifier-based lookup first (stable across analysis passes),
        # then fall back to storage.  Check all variables at the storage
        # offset because BN may keep both auto and user-named entries.
        expected_name = item["new_name"]
        storage = int(item["storage"])
        identifier = item.get("identifier")
        var = None
        if identifier is not None:
            for v, _ in self._iter_canonical_variables(fn):
                if self._variable_identifier(v) == identifier:
                    var = v
                    break
        if var is None:
            var, _ = self._find_variable_by_storage(
                fn, storage, is_parameter=bool(item["is_parameter"]),
            )
        observed_name = str(var.name)
        # If the primary variable still shows the auto name, scan the
        # raw variable lists (bypassing dedup) because BN may keep both
        # auto-named and user-named entries at the same storage offset
        # after analysis. The alternate entry must still be the *same*
        # variable (matching identifier); an unrelated neighbor that
        # happens to carry the requested name must not count as success.
        if observed_name != expected_name:
            is_param = bool(item["is_parameter"])
            collections = [fn.parameter_vars] if is_param else [fn.stack_layout]
            for collection in collections:
                for v in list(collection):
                    if (
                        int(getattr(v, "storage", -1)) == storage
                        and str(v.name) == expected_name
                        and (identifier is None or self._variable_identifier(v) == identifier)
                    ):
                        observed_name = expected_name
                        var = v
                        break
                if observed_name == expected_name:
                    break
        item["observed"] = {"address": item["address"], "variable": observed_name, "storage": storage}
        if observed_name != expected_name:
            raise OperationFailure(
                "verification_failed",
                f"Live local rename verification failed at {item['address']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "noop" if item.get("before_name") == expected_name else "verified"
        return item

    def _verify_local_retype(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        address = _parse_address(item["address"])
        fn = bv.get_function_at(address)
        if fn is None:
            raise OperationFailure(
                "verification_failed",
                f"Function missing after local retype at {item['address']}",
                requested=item.get("requested"),
                observed={"address": item["address"], "type": None},
            )
        var, _ = self._find_variable_by_storage(
            fn,
            int(item["storage"]),
            is_parameter=bool(item["is_parameter"]),
        )
        observed_type = str(var.type)
        item["observed"] = {"address": item["address"], "variable": str(var.name), "type": observed_type}
        if observed_type != item["expected_type"]:
            raise OperationFailure(
                "verification_failed",
                f"Live local retype verification failed at {item['address']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "noop" if item.get("before_type") == item["expected_type"] else "verified"
        return item

    def _verify_struct_field_set(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        type_obj = bv.get_type_by_name(item["struct_name"])
        if type_obj is None:
            raise OperationFailure(
                "verification_failed",
                f"Struct missing after field set: {item['struct_name']}",
                requested=item.get("requested"),
                observed={"type_name": item["struct_name"]},
            )
        member = self._find_member(type_obj, offset=int(item["member_offset"]), name=item["field_name"])
        observed = {
            "type_name": item["struct_name"],
            "offset": item["offset"],
            "field_name": getattr(member, "name", None),
            "field_type": str(getattr(member, "type", "")) if member is not None else None,
        }
        item["observed"] = observed
        if member is None or observed["field_type"] != item["field_type"]:
            raise OperationFailure(
                "verification_failed",
                f"Live struct field verification failed for {item['struct_name']} at {item['offset']}",
                requested=item.get("requested"),
                observed=observed,
            )
        previous = item.get("before_member")
        if previous and previous.get("field_name") == item["field_name"] and previous.get("field_type") == item["field_type"]:
            item["status"] = "noop"
        else:
            item["status"] = "verified"
        return item

    def _verify_struct_field_rename(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        type_obj = bv.get_type_by_name(item["struct_name"])
        if type_obj is None:
            raise OperationFailure(
                "verification_failed",
                f"Struct missing after field rename: {item['struct_name']}",
                requested=item.get("requested"),
                observed={"type_name": item["struct_name"]},
            )
        # Verify by OFFSET, not by name: with duplicate member names a global
        # name lookup would see the OTHER same-named member and falsely report
        # failure (the #25 duplicate-name case). The member at the renamed
        # offset must now carry new_name.
        offset = int(item.get("member_offset", -1))
        member = self._find_member(type_obj, offset=offset, name=item["new_name"])
        observed = {
            "type_name": item["struct_name"],
            "offset": hex(offset) if offset >= 0 else None,
            "new_name": getattr(member, "name", None),
        }
        item["observed"] = observed
        if member is None:
            raise OperationFailure(
                "verification_failed",
                f"Live struct field rename verification failed for {item['struct_name']}",
                requested=item.get("requested"),
                observed=observed,
            )
        item["status"] = "noop" if item["old_name"] == item["new_name"] else "verified"
        return item

    def _verify_struct_field_delete(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        type_obj = bv.get_type_by_name(item["struct_name"])
        if type_obj is None:
            raise OperationFailure(
                "verification_failed",
                f"Struct missing after field delete: {item['struct_name']}",
                requested=item.get("requested"),
                observed={"type_name": item["struct_name"]},
            )
        # Verify by (offset, name), not by name alone: with duplicate member
        # names a global name lookup would see the OTHER same-named member at a
        # different offset and falsely report the delete failed (#25). The
        # specific member that was removed must be gone from its offset.
        offset = int(item.get("member_offset", -1))
        member = self._find_member(type_obj, offset=offset, name=item["field_name"])
        item["observed"] = {
            "type_name": item["struct_name"],
            "offset": hex(offset) if offset >= 0 else None,
            "field_present": member is not None,
        }
        if member is not None:
            raise OperationFailure(
                "verification_failed",
                f"Live struct field delete verification failed for {item['struct_name']}",
                requested=item.get("requested"),
                observed=item["observed"],
            )
        item["status"] = "verified"
        return item

    def _verify_declared_types(self, bv, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        defined_types = dict(item.get("defined_types") or {})
        defined_type_layouts = dict(item.get("defined_type_layouts") or {})
        if not defined_types:
            item["observed"] = {
                "defined_types": {},
                "parsed_functions": list(item.get("parsed_functions") or []),
                "parsed_variables": list(item.get("parsed_variables") or []),
            }
            item["status"] = "noop"
            item["message"] = "Parsed declarations but no named types were defined."
            return item
        observed_types: dict[str, str | None] = {}
        observed_type_layouts: dict[str, str | None] = {}
        for name, expected in defined_types.items():
            type_obj = bv.get_type_by_name(name)
            observed_types[name] = str(type_obj) if type_obj is not None else None
            observed_type_layouts[name] = self._render_type_layout(type_obj) if type_obj is not None else None
            if observed_types[name] != expected:
                if defined_type_layouts.get(name) and observed_type_layouts[name] == defined_type_layouts[name]:
                    continue
                raise OperationFailure(
                    "verification_failed",
                    f"Live type verification failed for {name}",
                    requested=item.get("requested"),
                    observed={
                        "defined_types": observed_types,
                        "defined_type_layouts": observed_type_layouts,
                    },
                )
        item["observed"] = {
            "defined_types": observed_types,
            "defined_type_layouts": observed_type_layouts,
        }
        before = dict(item.get("before_defined_types") or {})
        item["status"] = "noop" if before and all(before.get(name) == expected for name, expected in defined_types.items()) else "verified"
        return item

    def _apply_operation(self, bv, op: dict[str, Any], restores: list | None = None):
        kind = op.get("op") or "rename_symbol"
        # Validate required request fields up front so a malformed request is
        # reported precisely (invalid_request, naming the field) and a KeyError
        # raised deeper -- e.g. by BN internals inside a handler -- is no longer
        # misreported as a missing request field.
        for field in REQUIRED_FIELDS.get(kind, ()):
            if field not in op:
                raise OperationFailure(
                    "invalid_request",
                    f"operation {kind!r} is missing required field {field!r}",
                    requested=self._operation_requested(op),
                )
        for group in REQUIRED_ONE_OF.get(kind, ()):
            if not any(field in op for field in group):
                raise OperationFailure(
                    "invalid_request",
                    f"operation {kind!r} requires one of "
                    f"{' / '.join(repr(f) for f in group)}",
                    requested=self._operation_requested(op),
                )
        try:
            if kind == "rename_symbol":
                return self._op_rename_symbol(bv, op)
            if kind == "set_comment":
                return self._op_set_comment(bv, op)
            if kind == "delete_comment":
                return self._op_delete_comment(bv, op)
            if kind == "set_prototype":
                return self._op_set_prototype(bv, op, restores)
            if kind == "local_rename":
                return self._op_local_rename(bv, op, restores)
            if kind == "local_retype":
                return self._op_local_retype(bv, op, restores)
            if kind == "struct_field_set":
                return self._op_struct_field_set(bv, op)
            if kind == "struct_field_rename":
                return self._op_struct_field_rename(bv, op)
            if kind == "struct_field_delete":
                return self._op_struct_field_delete(bv, op)
            if kind == "types_declare":
                return self._op_types_declare(bv, op)
            raise OperationFailure("unsupported", f"Unsupported operation: {kind}", requested=self._operation_requested(op))
        except OperationFailure:
            raise
        except Exception as exc:
            raise OperationFailure(
                "unsupported",
                f"{type(exc).__name__}: {exc}",
                requested=self._operation_requested(op),
            ) from exc

    def _revert_undo_safely(self, bv, state) -> bool:
        """Best-effort rollback. Returns False when the revert itself failed,
        meaning partially-applied changes may still be live in the view."""
        try:
            bv.revert_undo_actions(state)
            return True
        except Exception as exc:
            bn.log_error(f"BN Agent Bridge: rollback failed, view may be partially modified: {exc!r}")
            return False

    def _find_var_for_restore(self, fn, identifier, storage, is_parameter):
        """Re-resolve a local for restore the way verification does: identifier
        first (stable across analysis passes and covers register vars, which
        stack_layout omits), then storage. Returns None if it can't be found."""
        if identifier is not None:
            for var, _ in self._iter_canonical_variables(fn):
                if self._variable_identifier(var) == identifier:
                    return var
        try:
            var, _ = self._find_variable_by_storage(fn, int(storage), is_parameter=is_parameter)
            return var
        except RuntimeError:
            return None

    def _run_local_restores(self, bv, restores) -> bool:
        """Explicitly undo changes BN's undo buffer does NOT journal -- local var
        rename/retype (Function.create_user_var) and function prototypes
        (Function.set_user_type). For these, revert_undo_actions is a silent no-op,
        so without replaying these restores --preview and rollback-on-failure would
        leave the change permanently applied. Runs in reverse apply order, then
        reanalyzes. Returns False if any restore failed."""
        if not restores:
            return True
        ok = True
        for restore in reversed(restores):
            try:
                restore()
            except Exception as exc:
                ok = False
                bn.log_error(
                    "BN Agent Bridge: failed to restore a non-journaled change "
                    "(local var create_user_var / prototype set_user_type) on revert: "
                    f"{exc!r}"
                )
        # create_user_var only materializes once analysis runs again (same as the
        # forward apply path), so settle the view before returning -- otherwise the
        # restore is queued but the old name/type is still showing.
        try:
            bv.update_analysis_and_wait()
        except Exception as exc:
            ok = False
            bn.log_error(f"BN Agent Bridge: reanalysis after local restore failed: {exc!r}")
        return ok

    def _mutation(self, selector: str | None, preview: bool, operations: list[dict[str, Any]]):
        if not operations:
            raise ValueError("Batch operation list is empty")

        bv = self._resolve_view(selector)
        affected = self._guess_affected_functions(bv, operations)
        before = self._capture_function_snapshots(bv, affected)
        type_before = self._capture_type_snapshots(bv, operations)
        state = bv.begin_undo_actions()
        results = []
        # Explicit restores for local var ops, which BN's undo buffer can't
        # revert (see _run_local_restores). Replayed on every revert path.
        restores: list = []
        try:
            for op in operations:
                results.append(self._apply_operation(bv, op, restores))
        except OperationFailure as exc:
            reverted = self._revert_undo_safely(bv, state) and self._run_local_restores(bv, restores)
            if reverted:
                message = "Rolled back before post-state verification because an operation failed to apply."
                result_note = "Rolled back before post-state verification."
            else:
                message = (
                    "An operation failed to apply AND the rollback itself failed; "
                    "the view may be left partially modified."
                )
                result_note = "Rollback failed; this operation may still be applied."
            return {
                "preview": preview,
                "success": False,
                "committed": False,
                "rolled_back": reverted,
                "message": message,
                "results": self._mark_unverified_results(results, result_note)
                + [self._operation_failure_result(operations[len(results)], exc)],
                "affected_functions": [],
                "affected_types": [],
            }

        try:
            bv.update_analysis_and_wait()
            after = self._capture_function_snapshots(bv, affected)
            type_after = self._capture_type_snapshots(bv, operations)
            diffs = self._diff_snapshots(before, after)
            type_diffs = self._diff_type_snapshots(type_before, type_after)
            verified_results = [self._verify_operation(bv, result) for result in results]
            annotated_results = self._annotate_operation_results(verified_results, type_diffs)
            failed = self._has_failed_results(annotated_results)
            restored = True
            if preview or failed:
                bv.revert_undo_actions(state)
                restored = self._run_local_restores(bv, restores)
            else:
                bv.commit_undo_actions(state)
            message = None
            if preview:
                message = "Preview verified and reverted." if restored else (
                    "Preview verified, but reverting a non-journaled change "
                    "(local variable or prototype) failed; the view may be left modified."
                )
            elif failed:
                message = "Rolled back because live-session verification failed." if restored else (
                    "Live-session verification failed AND reverting a non-journaled change "
                    "(local variable or prototype) failed; the view may be left modified."
                )
            else:
                message = "Applied and verified in the live Binary Ninja session."
            result = {
                "preview": preview,
                "success": not failed,
                "committed": bool((not preview) and (not failed)),
                "message": message,
                "results": annotated_results,
                "affected_functions": diffs,
                "affected_types": type_diffs,
            }
            if preview or failed:
                result["rolled_back"] = restored
            return result
        except Exception as exc:
            undo_ok = self._revert_undo_safely(bv, state)
            restore_ok = self._run_local_restores(bv, restores)
            if not (undo_ok and restore_ok):
                raise RuntimeError(
                    f"{exc} (additionally, rollback failed; the view may be left partially modified)"
                ) from exc
            raise

    def _op_rename_symbol(self, bv, op: dict[str, Any]):
        kind = str(op.get("kind", "auto"))
        identifier = op["identifier"]
        new_name = str(op["new_name"])
        target = self._resolve_rename_target(bv, identifier, kind)
        requested = self._operation_requested(op)
        if target["kind"] == "function":
            fn = bv.get_function_at(target["address"])
            if fn is None:
                raise OperationFailure("unsupported", f"Function not found: {identifier}", requested=requested)
            if target["before_name"] != new_name:
                fn.name = new_name
            return {
                "op": "rename_symbol",
                "kind": "function",
                "address": hex(target["address"]),
                "before_name": target["before_name"],
                "new_name": new_name,
                "requested": requested,
            }
        address = int(target["address"])
        if target["before_name"] != new_name:
            bv.define_user_symbol(bn.Symbol(bn.SymbolType.DataSymbol, address, new_name))
        return {
            "op": "rename_symbol",
            "kind": "data",
            "address": hex(address),
            "before_name": target["before_name"],
            "new_name": new_name,
            "requested": requested,
        }

    def _op_set_comment(self, bv, op: dict[str, Any]):
        comment = str(op["comment"])
        if op.get("function"):
            fn = self._find_function(bv, op["function"])
            before_comment = bv.get_comment_at(fn.start) or ""
            if before_comment != comment:
                bv.set_comment_at(fn.start, comment)
            return {
                "op": "set_comment",
                "address": hex(fn.start),
                "function": fn.name,
                "before_comment": before_comment,
                "requested": self._operation_requested(op),
            }
        address = _parse_address(op["address"])
        before_comment = bv.get_comment_at(address) or ""
        if before_comment != comment:
            bv.set_comment_at(address, comment)
        return {
            "op": "set_comment",
            "address": hex(address),
            "before_comment": before_comment,
            "requested": self._operation_requested(op),
        }

    def _op_delete_comment(self, bv, op: dict[str, Any]):
        if op.get("function"):
            fn = self._find_function(bv, op["function"])
            before_comment = bv.get_comment_at(fn.start) or ""
            if before_comment:
                bv.set_comment_at(fn.start, None)
            return {
                "op": "delete_comment",
                "address": hex(fn.start),
                "function": fn.name,
                "before_comment": before_comment,
                "requested": self._operation_requested(op),
            }
        address = _parse_address(op["address"])
        before_comment = bv.get_comment_at(address) or ""
        if before_comment:
            bv.set_comment_at(address, None)
        return {
            "op": "delete_comment",
            "address": hex(address),
            "before_comment": before_comment,
            "requested": self._operation_requested(op),
        }

    def _op_set_prototype(self, bv, op: dict[str, Any], restores: list | None = None):
        fn = self._find_function(bv, op["identifier"])
        expected_type, _ = bv.parse_type_string(str(op["prototype"]))
        before_prototype = str(fn.type)
        before_type_obj = fn.type
        expected_prototype = str(expected_type)
        if before_prototype != expected_prototype:
            # Function.set_user_type is NOT journaled by BN's undo buffer (same
            # class as create_user_var for locals), so revert_undo_actions is a
            # silent no-op for prototypes -- without an explicit restore, --preview
            # and rollback-on-failure would leave the previewed prototype committed
            # to the view (#51). Register the restore before mutating.
            self._register_prototype_restore(
                bv, restores, fn=fn,
                before_prototype=before_prototype, before_type_obj=before_type_obj,
            )
            try:
                fn.set_user_type(expected_prototype)
            except TypeError:
                fn.set_user_type(expected_type)
        return {
            "op": "set_prototype",
            "function": fn.name,
            "address": hex(fn.start),
            "before_prototype": before_prototype,
            "expected_prototype": expected_prototype,
            "requested": self._operation_requested(op),
        }

    def _register_prototype_restore(self, bv, restores, *, fn, before_prototype, before_type_obj):
        """Capture how to put a function prototype back on revert. Mirrors
        :meth:`_register_local_restore`: ``set_user_type`` is not journaled by BN's
        undo buffer, so the preview/rollback paths replay this explicitly. Re-resolves
        the function fresh at restore time by its start address."""
        if restores is None:
            return
        fn_start = int(fn.start)

        def _restore():
            rfn = bv.get_function_at(fn_start)
            if rfn is None:
                raise RuntimeError(f"function {hex(fn_start)} missing on prototype restore")
            # Restore via the .type property setter with the captured Type object:
            # it puts back the EXACT prior prototype, whereas set_user_type would
            # re-pin the calling convention explicitly (turning an implicit-default
            # auto prototype into `... __convention("cdecl") ...`), which is not a
            # true no-op. Fall back to the type string only if the object path fails.
            try:
                rfn.type = before_type_obj
            except Exception:
                rfn.set_user_type(before_prototype)

        restores.append(_restore)

    def _register_local_restore(self, bv, restores, *, fn, var, name, type_obj, is_parameter):
        """Capture how to put a local back to (name, type_obj) on revert.
        Re-resolves the variable fresh at restore time by identifier/storage,
        because the captured Variable's index can shift across reanalysis."""
        if restores is None:
            return
        fn_start = int(fn.start)
        identifier = self._variable_identifier(var)
        storage = int(var.storage)

        def _restore():
            rfn = bv.get_function_at(fn_start)
            if rfn is None:
                raise RuntimeError(f"function {hex(fn_start)} missing on restore")
            rvar = self._find_var_for_restore(rfn, identifier, storage, is_parameter)
            if rvar is None:
                raise RuntimeError(f"local at storage {storage} missing on restore in {hex(fn_start)}")
            rfn.create_user_var(rvar, type_obj, name)

        restores.append(_restore)

    def _op_local_rename(self, bv, op: dict[str, Any], restores: list | None = None):
        fn = self._find_function(bv, op["function"])
        var, is_parameter = self._find_variable_selector(fn, str(op["variable"]))
        new_name = str(op["new_name"])
        # Variable.name is a live property backed by the core: snapshot it
        # before mutating, or before_name reads back the new name and
        # verification misclassifies a real change as a noop.
        before_name = str(var.name)
        if before_name != new_name:
            # create_user_var isn't journaled by BN's undo buffer, so register
            # an explicit restore for the preview/rollback paths.
            self._register_local_restore(
                bv, restores, fn=fn, var=var, name=before_name, type_obj=var.type, is_parameter=is_parameter
            )
            fn.create_user_var(var, var.type, new_name)
        return {
            "op": "local_rename",
            "function": fn.name,
            "address": hex(fn.start),
            "variable": str(op["variable"]),
            "local_id": self._local_id(fn, var, is_parameter=is_parameter),
            "storage": int(var.storage),
            "identifier": self._variable_identifier(var),
            "source_type": self._variable_source_name(var),
            "is_parameter": is_parameter,
            "before_name": before_name,
            "new_name": new_name,
            "requested": self._operation_requested(op),
        }

    def _op_local_retype(self, bv, op: dict[str, Any], restores: list | None = None):
        fn = self._find_function(bv, op["function"])
        var, is_parameter = self._find_variable_selector(fn, str(op["variable"]))
        expected_type, _ = bv.parse_type_string(str(op["new_type"]))
        # Variable.type is a live property backed by the core: snapshot it
        # before mutating (see _op_local_rename).
        before_type_obj = var.type
        before_type = str(before_type_obj)
        if before_type != str(expected_type):
            # create_user_var isn't journaled by BN's undo buffer, so register
            # an explicit restore for the preview/rollback paths.
            self._register_local_restore(
                bv, restores, fn=fn, var=var, name=str(var.name), type_obj=before_type_obj, is_parameter=is_parameter
            )
            fn.create_user_var(var, expected_type, var.name)
        return {
            "op": "local_retype",
            "function": fn.name,
            "address": hex(fn.start),
            "variable": str(op["variable"]),
            "local_id": self._local_id(fn, var, is_parameter=is_parameter),
            "storage": int(var.storage),
            "identifier": self._variable_identifier(var),
            "source_type": self._variable_source_name(var),
            "is_parameter": is_parameter,
            "before_type": before_type,
            "expected_type": str(expected_type),
            "requested": self._operation_requested(op),
        }

    def _struct_builder(self, bv, struct_name: str):
        try:
            resolved_name, type_obj = self._find_type(bv, struct_name)
        except RuntimeError:
            raise RuntimeError(f"Struct not found: {struct_name}")
        return resolved_name, type_obj.mutable_copy()

    def _commit_struct_builder(self, bv, struct_name: str, builder):
        bv.define_user_type(struct_name, builder)

    def _op_struct_field_set(self, bv, op: dict[str, Any]):
        struct_name = str(op["struct_name"])
        resolved_name, builder = self._struct_builder(bv, struct_name)
        field_type, _ = bv.parse_type_string(str(op["field_type"]))
        offset = _parse_address(op["offset"])
        overwrite = bool(op.get("overwrite_existing", True))
        before_type = bv.get_type_by_name(resolved_name)
        before_member = None
        if before_type is not None:
            member = self._find_member(before_type, offset=offset)
            if member is not None:
                before_member = {
                    "field_name": str(getattr(member, "name", "")),
                    "field_type": str(getattr(member, "type", "")),
                    "offset": hex(int(getattr(member, "offset", offset))),
                }
        # --no-overwrite must REFUSE when the new field would clobber existing
        # data, not add an overlapping member: BN's add_member_at_offset(...,
        # overwrite=False) silently appends an overlapping member (worse than the
        # overwrite path). Guard the whole BYTE RANGE the new field occupies, not
        # just an exact start-offset collision -- an offset that lands *inside* a
        # wider member (e.g. 0x4 within an int64_t at 0x0) overlaps just as much
        # (#56).
        if not overwrite and before_type is not None:
            new_width = 0
            try:
                new_width = int(field_type.width)
            except Exception:
                new_width = 0
            clash = self._first_overlapping_member(before_type, offset, new_width)
            if clash is not None:
                raise OperationFailure(
                    "invalid_request",
                    f"a {new_width or 1}-byte field at offset {hex(offset)} in struct "
                    f"{resolved_name} would overlap existing member "
                    f"{str(getattr(clash, 'name', ''))!r} "
                    f"({str(getattr(clash, 'type', ''))}) at "
                    f"{hex(int(getattr(clash, 'offset', offset)))}; --no-overwrite refuses "
                    f"to clobber it. Re-run without --no-overwrite to replace it, or choose "
                    f"a free range.",
                    requested=self._operation_requested(op),
                )
        builder.add_member_at_offset(str(op["field_name"]), field_type, offset, overwrite)
        try:
            builder.width = max(int(builder.width), int(offset) + int(field_type.width))
        except Exception:
            pass
        self._commit_struct_builder(bv, resolved_name, builder)
        return {
            "op": "struct_field_set",
            "struct_name": resolved_name,
            "offset": hex(offset),
            "field_name": str(op["field_name"]),
            "field_type": str(field_type),
            "member_offset": int(offset),
            "before_member": before_member,
            "requested": self._operation_requested(op),
        }

    def _resolve_struct_field(self, builder, resolved_name: str, locator: Any):
        """Resolve a struct-field locator (a field NAME, or an OFFSET like 0x8 /
        8) to its ``(index, member)`` in *builder* from a SINGLE scan.

        Returning the index+member directly -- instead of a name that the caller
        re-looks-up -- is what makes ``rename``/``delete`` hit the right field
        when two members share a name at different offsets: a name round-trip
        went through ``index_by_name`` (first-match), silently mutating the
        wrong field. The offset is parsed with ``_parse_address`` so the grammar
        is identical to ``struct field set`` (a zero-padded ``0008`` resolves the
        same in all three). Raises invalid_request when nothing matches."""
        text = str(locator)
        members = list(getattr(builder, "members", []) or [])
        for index, member in enumerate(members):
            if str(getattr(member, "name", "")) == text:
                return index, member
        try:
            offset = _parse_address(text)
        except ValueError:
            offset = None
        if offset is not None:
            for index, member in enumerate(members):
                if int(getattr(member, "offset", -1)) == offset:
                    return index, member
        raise OperationFailure(
            "invalid_request",
            f"no field named or at offset {text!r} in struct {resolved_name}",
        )

    def _op_struct_field_rename(self, bv, op: dict[str, Any]):
        struct_name = str(op["struct_name"])
        resolved_name, builder = self._struct_builder(bv, struct_name)
        index, member = self._resolve_struct_field(builder, resolved_name, op["old_name"])
        old_name = str(getattr(member, "name", ""))
        member_offset = int(getattr(member, "offset", -1))
        builder.replace(index, member.type, str(op["new_name"]), True)
        self._commit_struct_builder(bv, resolved_name, builder)
        return {
            "op": "struct_field_rename",
            "struct_name": resolved_name,
            "old_name": old_name,
            "new_name": str(op["new_name"]),
            "member_offset": member_offset,
            "requested": self._operation_requested(op),
        }

    def _op_struct_field_delete(self, bv, op: dict[str, Any]):
        struct_name = str(op["struct_name"])
        resolved_name, builder = self._struct_builder(bv, struct_name)
        index, member = self._resolve_struct_field(builder, resolved_name, op["field_name"])
        field_name = str(getattr(member, "name", ""))
        member_offset = int(getattr(member, "offset", -1))
        builder.remove(index)
        self._commit_struct_builder(bv, resolved_name, builder)
        return {
            "op": "struct_field_delete",
            "struct_name": resolved_name,
            "field_name": field_name,
            "member_offset": member_offset,
            "requested": self._operation_requested(op),
        }

    def _op_types_declare(self, bv, op: dict[str, Any]):
        parsed = self._parse_declaration_source(
            bv,
            str(op["declaration"]),
            source_path=op.get("source_path"),
        )
        named_types = list(parsed["types"])
        defined_types = {}
        defined_type_layouts = {}
        before_defined_types = {}
        for name, type_obj in named_types:
            existing = self._current_type_entry(bv, str(name))
            before_defined_types[str(name)] = existing["decl"] if existing is not None else None
            bv.define_user_type(name, type_obj)
            current = self._current_type_entry(bv, str(name))
            defined_types[str(name)] = current["decl"] if current is not None else str(type_obj)
            defined_type_layouts[str(name)] = current["layout"] if current is not None else self._render_type_layout(type_obj)
        return {
            "op": "types_declare",
            "defined_types": defined_types,
            "defined_type_layouts": defined_type_layouts,
            "before_defined_types": before_defined_types,
            "count": len(defined_types),
            "parsed_functions": [name for name, _ in parsed["functions"]],
            "parsed_variables": [name for name, _ in parsed["variables"]],
            "parsed_type_count": len(named_types),
            "parsed_function_count": len(parsed["functions"]),
            "parsed_variable_count": len(parsed["variables"]),
            "requested": self._operation_requested(op),
        }

_bridge: BinaryNinjaBridge | None = None
_headless_views: list[Any] = []
_headless_views_lock = threading.Lock()
# Views loaded with --quick (analysis not run yet). Strings/full function set
# are unavailable until `bn refresh`, so commands consult this to stay honest
# instead of returning a misleading empty result. Weak so closed views drop out.
_quick_loaded_views: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _start_bridge_command(_):  # pragma: no cover - GUI runtime
    start_bridge()


def start_bridge():  # pragma: no cover - GUI runtime
    global _bridge
    if ui is None:
        return
    if _bridge is not None:
        return
    _bridge = BinaryNinjaBridge()
    _bridge.start()


def start_headless(
    binaries: list[str] | None = None,
    instance_id: str | None = None,
    quick: bool = False,
):
    """Start the bridge in headless mode (no GUI required).

    Opens any binary file paths provided, starts the socket server,
    and blocks the calling thread until shutdown is requested. With ``quick``,
    preloaded binaries skip ``update_analysis_and_wait()`` (see ``--quick``).
    """
    global _bridge
    if _bridge is not None:
        return

    if instance_id is None:
        import secrets
        instance_id = secrets.token_hex(4)

    inst_dir = instances_dir()
    inst_dir.mkdir(parents=True, exist_ok=True)

    _bridge = BinaryNinjaBridge(instance_id=instance_id)
    _bridge.start()
    bn.log_info(f"BN Agent Bridge running in headless mode (instance {instance_id})")

    # Open any preloaded binaries *after* the socket is live and the registry
    # is written, so the instance is immediately discoverable (`bn session
    # list`, `bn target list`, etc.) while the potentially multi-minute
    # update_analysis_and_wait() runs. Each view is appended only once its
    # analysis finishes, so clients never observe a half-analyzed target -- it
    # simply appears in the target list when ready. Crucially, if analysis
    # crashes here the instance has already registered, so it surfaces as a
    # dead instance instead of an invisible orphan process.
    if binaries:
        import binaryninja

        for path in binaries:
            resolved = Path(path).expanduser().resolve()
            bv = binaryninja.load(str(resolved), update_analysis=False)
            if bv is None:
                bn.log_warn(f"Failed to open binary: {resolved}")
                continue
            if not quick:
                bv.update_analysis_and_wait()
            with _headless_views_lock:
                _headless_views.append(bv)
            bn.log_info(f"Loaded {resolved}{' (no analysis)' if quick else ''}")

    try:
        _bridge._shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_bridge()


def _stop_bridge():  # pragma: no cover - GUI runtime
    global _bridge
    if _bridge is not None:
        _bridge.stop()
        _bridge = None


atexit.register(_stop_bridge)

PluginCommand.register(
    "BN Agent Bridge\\Restart Bridge",
    "Restart the bn CLI socket bridge",
    _start_bridge_command,
)
