from __future__ import annotations

import atexit
import contextlib
import errno
import hashlib
import json
import os
import socket
import socketserver
import tempfile
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import binaryninja as bn
from binaryninja.plugin import PluginCommand

from . import create_comments
from . import il_format
from . import mutation_engine
from . import read_decompile
from . import read_class
from . import read_evidence
from . import read_go
from . import read_listing
from . import read_misc
from . import read_tags
from . import read_taint_slice
from . import read_types
from . import read_xrefs
from . import vars as vars_mod
from ._shared import (
    OperationFailure,
    _artifact_summary,
    _json_response,
    _normalize_prototype,
    _parse_address,
    _residue_error_disclosure,
    _run_on_main_thread,
    _serialize_error,
    _validate_bool,
    _validate_count,
    _write_json_artifact,
)
from .op_registry import REGISTRY, op
from .paths import (
    PLUGIN_NAME, bridge_registry_path, bridge_socket_path, cache_home, instances_dir,
    marker_name, project_root,
)
from .seam import BridgeContext
from .version import VERSION, build_id_for_file, build_id_for_package

try:
    import binaryninjaui as ui
except Exception:  # ImportError or UIPluginInHeadlessError
    ui = None


PLUGIN_BUILD_ID = build_id_for_file(Path(__file__).resolve())
# Whole-package fingerprint captured at import time: the code this live process
# is actually running. `doctor` compares it to the on-disk package so an edited
# engine module (e.g. taint_engine.py) is flagged stale, not just bridge.py (#161).
ENGINE_BUILD_ID = build_id_for_package(Path(__file__).resolve().parent)

# Upper bound on a single newline-terminated JSON request. Anything larger is
# rejected with a clean error instead of being buffered without limit.
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# Same-path load dedupe for the window where a BinaryView is open or analyzing
# but not published in _headless_views yet (#400).
_load_in_progress_lock = threading.Lock()
_load_in_progress: dict[str, threading.Event] = {}

# Set by BridgeHandler while a request is executing. Long operations can poll it
# to stop work after the CLI sends a cancellation request instead of wedging a
# bridge instance with abandoned work (#365).
_request_cancel_checker = threading.local()


@contextlib.contextmanager
def _request_cancel_context(checker):
    old = getattr(_request_cancel_checker, "checker", None)
    _request_cancel_checker.checker = checker
    try:
        yield
    finally:
        if old is None:
            with contextlib.suppress(AttributeError):
                delattr(_request_cancel_checker, "checker")
        else:
            _request_cancel_checker.checker = old


def _request_cancelled() -> bool:
    checker = getattr(_request_cancel_checker, "checker", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


GO_RENAME_CHUNK_SIZE = 256


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


# REQUIRED_FIELDS / REQUIRED_ONE_OF moved to mutation_engine.py with the
# mutation cluster they validate (issue #33 Stage 4).


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

    views = _run_on_main_thread(collect)
    # `bn load` against a GUI bridge appends to _headless_views, but the UI
    # enumeration above only walks UI tabs/contexts -- so a headless-loaded view
    # would be invisible to `target list` and unselectable despite a successful
    # load (#86 Problem A). Merge in any tracked headless views the UI walk
    # missed so every loaded target is visible and resolvable.
    with _headless_views_lock:
        seen_ids = {id(bv) for bv in views}
        for bv in _headless_views:
            if id(bv) not in seen_ids:
                seen_ids.add(id(bv))
                views.append(bv)
    return views


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
        # Stable view_ids of views with a committed-but-unsaved mutation. BN's
        # bv.file.modified does NOT flip True after our verified rename/comment/
        # retype writes, so `close` could not tell an agent it was about to
        # discard annotations. We track dirtiness ourselves: set on a committed
        # change, cleared on save, surfaced by close. (L15)
        self._dirty_view_ids: set[str] = set()

    def _stable_view_id(self, bv) -> str | None:
        entry = self._ids_by_object.get(id(bv))
        if entry is not None:
            ref, vid = entry
            if ref() is bv:
                return vid
        return None

    def mark_dirty(self, bv) -> None:
        """Record that *bv* has a committed mutation not yet written to a .bndb."""
        with self._lock:
            vid = self._stable_view_id(bv)
            if vid is not None:
                self._dirty_view_ids.add(vid)

    def clear_dirty(self, bv) -> None:
        with self._lock:
            vid = self._stable_view_id(bv)
            if vid is not None:
                self._dirty_view_ids.discard(vid)

    def is_dirty(self, bv) -> bool:
        with self._lock:
            vid = self._stable_view_id(bv)
            return vid is not None and vid in self._dirty_view_ids

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
        # A .bndb corpus loads <binary>.bndb, but the obvious selector is
        # <binary>; match a candidate against the basename / path tail with a
        # trailing .bndb stripped so `-t foo` resolves `foo.bndb` (#312).
        if record.basename.endswith(".bndb"):
            core = record.basename[: -len(".bndb")]
            if candidate == core:
                return True
            # Read-only-mount targets restore the GLOBAL cache DB, named
            # `<stem>.<16-hex path digest>.bndb` by `_cache_bndb_path`. Its core
            # is `<stem>.<digest>`, so `-t <stem>` (e.g. `-t foo`, the obvious
            # base name) would miss the exact/`.bndb`-strip cases above. Strip a
            # trailing `.<16 hex>` so the cache name resolves by stem.
            stem, dot, digest = core.rpartition(".")
            if (
                dot
                and candidate == stem
                and len(digest) == 16
                and all(c in "0123456789abcdef" for c in digest)
            ):
                return True
        suffix = _path_components(candidate)
        if suffix:
            parts = _path_components(record.filename)
            if len(parts) >= len(suffix) and parts[-len(suffix):] == suffix:
                return True
            if parts and parts[-1].endswith(".bndb"):
                stripped = tuple(parts[:-1]) + (parts[-1][: -len(".bndb")],)
                if len(stripped) >= len(suffix) and tuple(stripped[-len(suffix):]) == tuple(suffix):
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
                        # Per-target analysis state so `target list` can flag a
                        # --quick or restore-failed (#458) unanalyzed view without a
                        # per-target lookup.
                        "analyzed": (
                            view not in _quick_loaded_views
                            and view not in _unanalyzed_views
                        ),
                        "analysis_state": (
                            "quick" if view in _quick_loaded_views
                            else "unanalyzed" if view in _unanalyzed_views
                            else "full"
                        ),
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

    def _encode_response(
        self,
        response: dict[str, Any],
        *,
        op: str | None = None,
        request_id: str | None = None,
    ) -> bytes:
        """Serialize a response, never letting a failure escape and silently kill
        the handler thread (#250).

        Serialization runs outside the dispatch lock and outside dispatch's
        try/except, so a response that aliases a live BN/shared mapping mutated by
        a concurrent read can raise ``RuntimeError: dictionary changed size during
        iteration`` mid-encode (``sort_keys=True`` iterates every nested dict).
        That race is transient, so retry once -- the racing mutation has usually
        completed by then -- and only if it still fails, degrade to a clean
        ``{ok: false}`` error instead of dropping the connection with no response.
        """
        for attempt in (1, 2):
            try:
                return json.dumps(response, sort_keys=True, default=str).encode("utf-8")
            except Exception as exc:  # noqa: BLE001 - any encode failure must not kill the thread
                if attempt == 1:
                    continue
                # The fallback path must itself be exception-proof: if computing the
                # detail or logging it raised, the thread would die silently -- the
                # exact failure mode this method exists to prevent (#250).
                try:
                    detail = _serialize_error(exc)
                except Exception:  # noqa: BLE001
                    detail = "unserializable error"
                try:
                    ident = ", ".join(
                        p for p in (f"op={op}" if op else "", f"id={request_id}" if request_id else "") if p
                    )
                    bn.log_error(
                        f"BN Agent Bridge could not serialize a response ({ident}): {detail}"
                    )
                except Exception:  # noqa: BLE001
                    pass
                return json.dumps(
                    _json_response(ok=False, error=f"response serialization failed: {detail}"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
        # Unreachable (the loop returns on both attempts); a static last resort.
        return b'{"error": "response serialization failed", "ok": false}'

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
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Invalid UTF-8 bytes raise UnicodeDecodeError, NOT JSONDecodeError;
                # without catching it the worker thread died with no response and
                # the client saw a misleading "empty response" (shared upstream bug).
                response = _json_response(ok=False, error="Invalid JSON request")
            else:
                if not isinstance(payload, dict):
                    response = _json_response(
                        ok=False, error="Invalid request: expected a JSON object"
                    )
                else:
                    op = payload.get("op")
                    request_id = payload.get("id")
                    bridge = self.server.bridge
                    begin_request = getattr(bridge, "_begin_request", lambda _request_id: None)
                    end_request = getattr(bridge, "_end_request", lambda _request_id: None)
                    is_cancelled = getattr(
                        bridge, "_is_request_cancelled", lambda _request_id: False,
                    )
                    begin_request(request_id)
                    try:
                        with _request_cancel_context(lambda: is_cancelled(request_id)):
                            response = bridge.dispatch(payload)
                    finally:
                        end_request(request_id)
        encoded = self._encode_response(response, op=op, request_id=request_id)
        self._write_response(encoded, op=op, request_id=request_id)


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, socket_path: str, handler, bridge):
        self.bridge = bridge
        super().__init__(socket_path, handler)


# _STRING_TYPE_NAMES moved to read_misc.py with the strings op (#33).
# _VAR_DRIFT_OPS moved to mutation_engine.py with the mutation cluster (#33).


def _is_auto_function_name(name: str) -> bool:
    """True for BN's auto-generated function names -- ``sub_<hex>`` and the
    ``j_sub_<hex>`` thunk variant. Everything else counts as a meaningful name."""
    core = name[2:] if name.startswith("j_") else name
    if not core.startswith("sub_"):
        return False
    suffix = core[4:]
    return bool(suffix) and all(c in "0123456789abcdefABCDEF" for c in suffix)


def _is_go_rename_auto_name(name: str, address: int) -> bool:
    """Whether a Go recovered name may replace the current BN name."""
    return name == f"sub_{address:x}" or name.startswith("nullsub_")


# Symbol types whose NAME comes from relocations/imports, not from analysis or a
# human. Counting these as "named" badly overstates how much real code is named
# on a stripped binary (PLT import trampolines dominate it), so they get their
# own bucket (#122). Compared by enum-member name to avoid importing the enum.
_IMPORT_SYMBOL_TYPE_NAMES = frozenset(
    {"ImportedFunctionSymbol", "ImportAddressSymbol", "ExternalSymbol"}
)


def _is_imported_function(fn) -> bool:
    sym = getattr(fn, "symbol", None)
    sym_type = getattr(sym, "type", None)
    return getattr(sym_type, "name", None) in _IMPORT_SYMBOL_TYPE_NAMES


def _segment_entries(bv) -> list[dict[str, Any]]:
    """Segment map (r/w/x ranges) for `target info --verbose` (F21)."""
    entries = []
    for seg in getattr(bv, "segments", None) or []:
        start = int(getattr(seg, "start", 0))
        end = int(getattr(seg, "end", 0))
        entries.append({
            "start": hex(start),
            "end": hex(end),
            "length": end - start,
            "readable": bool(getattr(seg, "readable", False)),
            "writable": bool(getattr(seg, "writable", False)),
            "executable": bool(getattr(seg, "executable", False)),
        })
    return entries


def _function_name_summary(bv) -> dict[str, int]:
    """Function-count breakdown for `target info` (#122): own functions split
    into named vs auto-named (sub_<hex>), with import/extern stubs (whose names
    come from relocations) in a separate bucket so they don't inflate "named".
    Reflects whatever functions analysis has discovered so far."""
    functions = list(getattr(bv, "functions", []) or [])
    total = len(functions)
    named = imported = 0
    imported_obj_names: set[str] = set()
    for fn in functions:
        if _is_imported_function(fn):
            imported += 1
            sym = getattr(fn, "symbol", None)
            raw = str(getattr(sym, "raw_name", "") or "")
            if raw:
                imported_obj_names.add(raw)
            continue
        name = str(getattr(fn, "name", "") or "")
        if name and not _is_auto_function_name(name):
            named += 1
    # #478: callable GOT slots (JUMP_SLOT-relocated) that BN never turned into a
    # PLT-stub function object still count as imported functions -- otherwise a
    # target whose PLT recovery failed reports zero imports and reads as
    # stripped/static, wrongly steering the bn-vr lane away from import-first sink
    # enumeration. Union by name so a well-analyzed binary (slot + its function
    # object both present) is not double-counted. The bv.functions partition
    # (named/unnamed) is unchanged -- those slots have no function object.
    extra_callable = read_misc._callable_import_slot_names(bv) - imported_obj_names
    return {
        "function_count": total,
        "named_function_count": named,
        "unnamed_function_count": total - named - imported,
        "imported_function_count": imported + len(extra_callable),
    }


class BinaryNinjaBridge:
    def __init__(self, instance_id: str | None = None):
        self.instance_id = instance_id
        self.targets = TargetManager()
        self.ctx = BridgeContext(self.targets)
        self.socket_path = bridge_socket_path(instance_id)
        self.registry_path = bridge_registry_path(instance_id)
        self._server: ThreadedUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._target_lock = _ReadWriteLock()
        # Serializes write operations. Long chunked writers can hold this gate
        # while releasing _target_lock between chunks so reads stay responsive.
        self._write_gate = threading.Lock()
        self._request_state_lock = threading.Lock()
        self._active_requests: set[str] = set()
        self._cancelled_requests: set[str] = set()
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

    def _write_project_marker(self, workdir: str | None, no_marker: bool,
                              refresh_only: bool = False) -> str | None:
        """Drop a `.bn-<instance_id>` pointer in the CLI's project root so a bare
        `bn` command there resolves THIS instance among many (#80). Best-effort:
        a read-only/unwritable dir is a non-fatal one-line note, never an error.
        Skipped for the GUI bridge (instance_id None keeps its legacy fixed
        registry) and when the caller opts out (--no-marker / BN_NO_MARKERS).
        Registry stays the source of truth; the marker is a validated pointer.

        ``refresh_only`` (used by `session restart`, #391) writes ONLY when a
        marker already exists in this root -- so a restart from a cwd that differs
        from the original `session start` cwd refreshes the real marker's stale
        body without dropping a stray new marker in an unintended directory."""
        if no_marker or not self.instance_id or not workdir:
            return None
        try:
            root = project_root(Path(workdir).expanduser())
        except Exception:
            return None
        marker = root / marker_name(self.instance_id)
        if refresh_only and not marker.exists():
            return None
        try:
            marker.write_text(json.dumps({
                "instance_id": self.instance_id,
                "socket_path": str(self.socket_path),
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }), encoding="utf-8")
        except OSError as exc:
            return f"could not write project marker in {root} ({exc}); resolve with -i"
        self._git_exclude_marker(root)
        return None

    def _git_exclude_marker(self, root: Path) -> None:
        """Add `.bn-*` to `.git/info/exclude` (not .gitignore -- no tracked-tree
        dirt) so markers don't show up as untracked in `git status` (#80)."""
        git_dir = root / ".git"
        if not git_dir.is_dir():
            return
        exclude = git_dir / "info" / "exclude"
        try:
            existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
            if any(line.strip() == ".bn-*" for line in existing.splitlines()):
                return
            exclude.parent.mkdir(parents=True, exist_ok=True)
            sep = "" if (not existing or existing.endswith("\n")) else "\n"
            exclude.write_text(f"{existing}{sep}.bn-*\n", encoding="utf-8")
        except OSError:
            pass  # best-effort; a missing exclude just means markers show in git status

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
        # #80: record the open binaries so `bn instance list` can show what each
        # instance holds without an N-instance `target list` round-trip -- the
        # biggest usability win for the many-bridges case. Best-effort: a registry
        # write must never fail because target enumeration hiccuped.
        binaries: list[str] = []
        try:
            seen: set[str] = set()
            for rec in self.targets.refresh():
                fn = rec.get("filename") if isinstance(rec, dict) else None
                if fn and fn not in seen:
                    seen.add(fn)
                    binaries.append(str(fn))
            binaries.sort()
        except Exception:
            binaries = []
        payload["binaries"] = binaries
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
            spec = REGISTRY.spec(op)
            lock = contextlib.nullcontext()
            if spec is not None:
                lock_class = spec.lock
                # --force-analysis reanalyzes the function, mutating the view, so
                # it needs the exclusive lock even though decompile is a read op.
                if spec.lock_escalation is not None and spec.lock_escalation(params):
                    lock_class = "write"
                if lock_class == "write":
                    lock = self._write_operation_lock()
                elif lock_class == "read":
                    lock = self._target_lock.read()
            with lock:
                result = self._dispatch_on_main(op, params, target)
            return _json_response(ok=True, result=result)
        except Exception as exc:
            # When the failure carries an unclearable has_user_type residue, the
            # caller-visible payload must DISCLOSE it (success:false /
            # rolled_back:false / prototype_user_type_residue:true + explanation),
            # not just str(exc) -- an unattended control loop reads the response
            # and must see the view is left modified (#630 round 3).
            return _json_response(
                ok=False,
                error=_serialize_error(exc),
                result=_residue_error_disclosure(exc),
            )

    @contextlib.contextmanager
    def _write_operation_lock(self):
        with self._write_gate:
            with self._target_lock.write():
                yield

    def _begin_request(self, request_id: Any) -> None:
        if not isinstance(request_id, str) or not request_id:
            return
        with self._request_state_lock:
            self._active_requests.add(request_id)

    def _end_request(self, request_id: Any) -> None:
        if not isinstance(request_id, str) or not request_id:
            return
        with self._request_state_lock:
            self._active_requests.discard(request_id)
            self._cancelled_requests.discard(request_id)

    def _is_request_cancelled(self, request_id: Any) -> bool:
        if not isinstance(request_id, str) or not request_id:
            return False
        with self._request_state_lock:
            return request_id in self._cancelled_requests

    def _cancel_request(self, request_id: Any) -> dict[str, Any]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("cancel_request requires params.request_id")
        with self._request_state_lock:
            active = request_id in self._active_requests
            if active:
                self._cancelled_requests.add(request_id)
        return {"kind": "cancel_request", "request_id": request_id, "cancelled": active}

    def _dispatch_on_main(self, op: str, params: dict[str, Any], target: str | None):
        spec = REGISTRY.spec(op)
        if spec is None:
            raise ValueError(f"Unknown operation: {op}")
        return spec.binder(self, params, target)

    def _doctor(self):
        return {
            "plugin_name": PLUGIN_NAME,
            "plugin_version": VERSION,
            "plugin_build_id": PLUGIN_BUILD_ID,
            # Whole-package fingerprint of the code this process loaded (#161).
            "engine_build_id": ENGINE_BUILD_ID,
            "pid": os.getpid(),
            "socket_path": str(self.socket_path),
            "targets": self.targets.refresh(),
        }

    def _find_open_view_for_path(self, load_path: Path):
        """Return an already-open BinaryView whose backing file resolves to
        *load_path*, or None. Keeps ``load`` idempotent: re-loading an open path
        would otherwise create a second view and make the basename/filename
        selectors ambiguous (#355). Includes GUI-open views as well as bridge-
        loaded headless views so a GUI bridge does not duplicate an already-open
        tab (#400)."""
        try:
            target = load_path.resolve()
        except Exception:
            target = load_path
        for bv in _collect_open_views():
            fname = getattr(getattr(bv, "file", None), "filename", None)
            if not fname:
                continue
            try:
                if Path(fname).resolve() == target:
                    return bv
            except Exception:
                continue
        return None

    def _load_existing_result(self, load_path: Path, resolved: Path, notes: list[str], existing):
        with self._target_lock.write():
            targets = self.targets.refresh()
        analyzed = existing not in _quick_loaded_views and existing not in _unanalyzed_views
        analysis_state = (
            "quick" if existing in _quick_loaded_views
            else "unanalyzed" if existing in _unanalyzed_views
            else "full"
        )
        notes.append(
            f"{load_path} is already open; returned the existing target "
            "instead of opening a duplicate (use `bn close` first to reload)"
        )
        return {
            "loaded": True,
            "path": str(load_path),
            "requested_path": str(resolved),
            "analyzed": analyzed,
            "analysis_state": analysis_state,
            "already_open": True,
            "notes": notes,
            "targets": targets,
        }

    def _load_binary(self, path: str, *, prefer_bndb: bool = True, quick: bool = False,
                     workdir: str | None = None, no_marker: bool = False,
                     marker_refresh_only: bool = False):
        import binaryninja

        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"File not found: {resolved}")

        load_path = resolved
        notes: list[str] = []
        substitute = _resolve_bndb_sidecar(resolved, prefer_bndb)
        if substitute is not None:
            load_path, source = substitute
            if source == "cache":
                # The binary's own mount had no writable adjacent .bndb (a
                # read-only firmware image), so the annotations were saved to the
                # writable cache. Restore them instead of silently re-analyzing
                # blank, which looked like total annotation loss (#318).
                notes.append(
                    f"restored cached database {load_path} (saved when "
                    f"{resolved.name}'s mount had no writable adjacent .bndb -- "
                    f"annotations preserved); pass --no-bndb to load the raw bytes"
                )
            else:
                notes.append(
                    f"loaded {load_path} instead of {resolved} (use --no-bndb to skip)"
                )

        # #355/#400: if a view for this (post-substitution) path is already
        # open, return its target instead of opening a duplicate. Concurrent
        # same-path loads need an in-flight marker too: full analysis runs
        # outside the target write lock, so a second loader could otherwise miss
        # the unpublished first view and open a duplicate.
        load_key = str(load_path)
        while True:
            existing = self._find_open_view_for_path(load_path)
            if existing is not None:
                return self._load_existing_result(load_path, resolved, notes, existing)
            with _load_in_progress_lock:
                waiter = _load_in_progress.get(load_key)
                if waiter is None:
                    waiter = threading.Event()
                    _load_in_progress[load_key] = waiter
                    break
            waiter.wait()

        # Always open without auto-analysis, then analyze explicitly unless
        # --quick. Quick load skips update_analysis_and_wait() entirely -- the
        # expensive, occasionally-crashing/OOMing phase -- so sections, symbols,
        # imports and strings are usable in ~1s while the function set stays
        # minimal until `bn refresh` promotes it to full analysis. A .bndb
        # already carries its saved analysis, so --quick is a no-op there.
        #
        # load_binary is NOT in WRITE_LOCKED_OPS (#99): hold the exclusive lock
        # only around the BN open (which touches global open-file state), then
        # run the multi-minute analysis UNLOCKED so doctor/target reads keep
        # responding. The view is published (and only then visible to reads)
        # under the lock at the end, so no reader ever sees a half-analyzed
        # target.
        # #609: track the view opened below so a failure between open and publish
        # can close it. Initialized to None so an open-time failure (bv never
        # assigned) doesn't trip the cleanup path with a NameError.
        bv = None
        try:
            try:
                with self._target_lock.write():
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

            # #458: a directly-named .bndb must restore its analyzed view. BN's
            # load() can default to the raw container view (0 functions) even when
            # the database holds a saved analyzed view -- recover it (under the lock,
            # since get_view_of_type touches BN state) or surface a hard restore
            # diagnostic, so an agent never silently continues on a no-symbol target.
            bndb_unanalyzed = False
            if load_path.suffix == ".bndb":
                with self._target_lock.write():
                    bv, bndb_note, bndb_unanalyzed = _restore_bndb_analyzed_view(bv, load_path)
                if bndb_note:
                    notes.append(bndb_note)

            # #369 (part 1): BN's format detection failed and fell back to a raw
            # 'Raw'/'Mapped' view -- an empty/garbage/text file opened this way has
            # no functions or symbols. Warn so an agent doesn't proceed against a
            # 0-function phantom target thinking it loaded a real binary. The .bndb
            # case is handled above with a stronger, restore-specific diagnostic.
            view_type_name = str(getattr(bv, "view_type", "") or "")
            if view_type_name in ("Raw", "Mapped") and load_path.suffix != ".bndb":
                notes.append(
                    f"{resolved.name} was opened as a raw '{view_type_name}' view -- "
                    "its format was not recognized, so it has no functions/symbols to "
                    "analyze. Confirm this is the binary you intended (or pass the "
                    "correct file)."
                )

            quick_effective = quick and load_path.suffix != ".bndb"
            if quick and not quick_effective:
                # --quick was requested but the loaded artifact is a .bndb, whose
                # saved analysis is already full -- so --quick can't apply. Say so
                # instead of silently dropping it (#316). Distinguish a sidecar
                # substitution (the surprising case: the user named the raw binary,
                # adjacent OR cache) from a directly-named .bndb.
                if substitute is not None:
                    notes.append(
                        f"--quick ignored: opened the analyzed database {load_path.name} "
                        f"(already fully analyzed) instead of {resolved.name}; pass "
                        "--no-bndb to load the raw bytes with --quick"
                    )
                else:
                    notes.append(
                        f"--quick ignored: {load_path.name} is an analyzed database, not "
                        "raw bytes; its saved analysis is already loaded"
                    )
            if quick_effective:
                notes.append(
                    "loaded without analysis (--quick): sections/imports/symbols are ready; "
                    "strings and the full function set need `bn refresh` (or "
                    "`bn decompile <fn> --force-analysis` for a single function)"
                )
                _quick_loaded_views.add(bv)
            else:
                # Slow phase, deliberately OUTSIDE the lock. bv is still unpublished
                # (not in _headless_views), so concurrent reads can't touch it.
                bv.update_analysis_and_wait()
                _quick_loaded_views.discard(bv)

            # #458: a .bndb that restored a raw container with no analyzed product
            # view is genuinely unanalyzed -- record it so target list/info report
            # analysis_state="unanalyzed" (not "full"), matching the WARNING note.
            if bndb_unanalyzed:
                _unanalyzed_views.add(bv)

            # Publish under the exclusive lock so a concurrent target read sees a
            # consistent set. Append under _headless_views_lock, then refresh
            # (which re-acquires that non-reentrant lock itself).
            with self._target_lock.write():
                with _headless_views_lock:
                    _headless_views.append(bv)
                targets = self.targets.refresh()

            # Keep the registry's open-binaries list current after a load (#80).
            self._write_registry()
            marker_note = self._write_project_marker(workdir, no_marker,
                                                     refresh_only=marker_refresh_only)
            if marker_note:
                notes.append(marker_note)
            return {
                "loaded": True,
                "path": str(load_path),
                "requested_path": str(resolved),
                "analyzed": (not quick_effective) and not bndb_unanalyzed,
                "analysis_state": (
                    "quick" if quick_effective
                    else "unanalyzed" if bndb_unanalyzed
                    else "full"
                ),
                "notes": notes,
                "targets": targets,
            }
        except BaseException:
            # #609: a view opened by binaryninja.load() above but not yet published
            # (the unlocked update_analysis_and_wait() raised/OOM'd, a .bndb restore
            # failed, or the load was cancelled) is otherwise left open, invisible to
            # `bn target list`, and uncloseable via `bn close` -- a leaked ~1.7 GB
            # view. Close it and drop it from the bookkeeping sets, then re-raise. A
            # *published* view is owned by _headless_views and closed via `bn close`,
            # so never double-close it here. BaseException so a cancel/KeyboardInterrupt
            # mid-analysis reclaims the view too.
            if bv is not None:
                with _headless_views_lock:
                    published = any(v is bv for v in _headless_views)
                if not published:
                    with contextlib.suppress(Exception):
                        _run_on_main_thread(lambda: bv.file.close())
                    _quick_loaded_views.discard(bv)
                    _unanalyzed_views.discard(bv)
            raise
        finally:
            with _load_in_progress_lock:
                current = _load_in_progress.get(load_key)
                if current is waiter:
                    _load_in_progress.pop(load_key, None)
                    waiter.set()

    def _close_binary(self, path: str | None = None, target: str | None = None, all_: bool = False):
        def _snapshot(bv) -> dict[str, Any]:
            # BN's bv.file.modified does not flip True after our verified
            # mutations, so OR in the bridge's own committed-but-unsaved tracking
            # (L15) -- otherwise close silently discards annotations.
            return {
                "path": str(getattr(bv.file, "filename", "")),
                "unsaved": bool(getattr(bv.file, "modified", False)) or self.targets.is_dirty(bv),
            }

        # A named path and all=true are mutually exclusive. The CLI already
        # rejects the combination, but a raw socket client could send both;
        # without this guard the all-branch silently wins and closes everything
        # despite a named path (#85).
        if all_ and path is not None:
            raise RuntimeError(
                "Pass either a path or all=true, not both: a named path closes "
                "only that target; all=true closes every loaded target."
            )

        # Resolve a target selector *before* taking _headless_views_lock:
        # resolve() -> refresh() -> _collect_open_views() re-acquires that lock,
        # which is non-reentrant, so resolving while holding it deadlocks.
        target_bv = self.targets.resolve(target) if target is not None else None

        # Target-based close takes priority and must succeed even when
        # _headless_views is empty: GUI-opened views resolve fine but are not
        # tracked in _headless_views, so the old "no binaries loaded" guard ran
        # before this branch and made every target-based close fail on a GUI
        # bridge (#86 Problem B). Close on the main thread (a no-op marshal when
        # already there) so closing a GUI view is safe, and only prune
        # _headless_views when the view is actually tracked there.
        if target_bv is not None:
            closed = [_snapshot(target_bv)]
            _run_on_main_thread(lambda: target_bv.file.close())
            with _headless_views_lock:
                _headless_views[:] = [v for v in _headless_views if v is not target_bv]
            # Refresh the registry's open-binaries list after a close (#80).
            # Outside _headless_views_lock: _write_registry -> refresh re-acquires
            # that non-reentrant lock, so writing inside it would deadlock.
            self._write_registry()
            return {"closed": closed}

        with _headless_views_lock:
            if not _headless_views:
                raise RuntimeError("No binaries are currently loaded")

            # --all closes everything
            if all_ or path is None:
                closed = []
                for bv in _headless_views:
                    closed.append(_snapshot(bv))
                    _run_on_main_thread(lambda v=bv: v.file.close())
                _headless_views.clear()
            else:
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
                    _run_on_main_thread(lambda v=bv: v.file.close())

        # Single post-lock registry refresh + return for both the --all and the
        # by-path close paths (must be OUTSIDE the lock; see the note above) (#80).
        self._write_registry()
        return {"closed": closed}

    def _save_database(self, target: str | None, path: str | None = None):
        bv = self.targets.resolve(target)
        filename = getattr(bv.file, "filename", "")
        explicit = path is not None
        if path:
            out = str(Path(path).expanduser().resolve())
        elif filename.endswith(".bndb"):
            out = filename
        else:
            out = filename + ".bndb"

        def _attempt(dest: str, *, make_parent: bool = False) -> str:
            dp = Path(dest)
            if make_parent:
                dp.parent.mkdir(parents=True, exist_ok=True)
            if not dp.parent.exists():
                raise RuntimeError(
                    f"Cannot save database to {dest}: directory does not exist: {dp.parent}"
                )
            # create_database returns a bool: False means Binary Ninja could not
            # write the file (e.g. an unwritable directory). A falsy result -- or
            # a path that simply isn't there afterward -- is a hard failure so
            # callers never get a false "saved".
            try:
                created = bv.create_database(dest)
            except Exception as exc:  # noqa: BLE001 - surface BN I/O errors cleanly
                raise RuntimeError(f"Failed to save database to {dest}: {exc}") from exc
            if created is False or not dp.exists():
                raise RuntimeError(
                    f"Failed to save database to {dest}: Binary Ninja reported no file was "
                    "written (check that the directory exists and is writable)"
                )
            return dest

        def _restore_filename() -> bool:
            """Undo create_database's re-home of the live view back to the original
            file. BN rebinds bv.file.filename to whatever .bndb it just wrote, and
            the target selector is derived live from that filename -- so without
            this restore a save (explicit --path #256, default <binary>.bndb, OR
            the RO->cache fallback #285) silently rekeys the -t <basename> selector
            a caller is already using. Every save is persistence, not an identity
            move, so the live target keeps its original filename. Returns True if
            the view's filename is (or already was) the original; False if the
            restore could not be applied -- the caller then surfaces a degraded
            result instead of claiming a clean save."""
            if not filename:
                return True
            try:
                bv.file.filename = filename
            except Exception:  # noqa: BLE001 - report degraded, never crash the save
                return False
            return str(getattr(bv.file, "filename", "")) == filename

        try:
            saved = _attempt(out)
        except RuntimeError as exc:
            # The common VR case is a binary on a read-only mount (firmware
            # image): the default <binary>.bndb write fails. Rather than lose the
            # annotations, retry into a writable cache dir and report where it
            # landed (#214). An EXPLICIT --path failure stays a hard error -- the
            # user chose that location, so a silent relocation would be wrong.
            if explicit:
                # create_database may have re-homed bv.file.filename to the failed
                # copy path before raising; restore the original identity so a
                # failed --path save never strands the live selector (#256 review).
                _restore_filename()
                raise
            # Shared with the load path (`_resolve_bndb_sidecar`) so a later load
            # of the same binary finds this copy and restores the annotations (#318).
            fallback = _cache_bndb_path(filename or out)
            try:
                saved = _attempt(str(fallback), make_parent=True)
            except Exception:
                # ANY fallback failure (incl. an OSError from the cache-dir mkdir
                # if it too is unwritable) re-raises the ORIGINAL default-path
                # error -- that's the actionable message, not the fallback's.
                raise exc
            self.targets.clear_dirty(bv)
            result = {"ok": True, "saved": True, "path": saved, "fallback": True, "requested_path": out}
            # The cache write re-homed the live view to the copy; the RO original
            # is intact, so restore the original filename -- otherwise the
            # -t <basename> selector silently rekeys to <basename>.<hash>.bndb (#285).
            if not _restore_filename():
                result["rehomed"] = True
                result["note"] = (
                    f"Saved to {saved}, but could not restore the live target's "
                    f"original filename ({filename!r}); the in-memory view is now "
                    f"homed at {str(getattr(bv.file, 'filename', ''))!r}. The saved "
                    f"annotations are in the cache at {saved} -- load that to resume "
                    "(the original mount is read-only, so it has no adjacent .bndb)."
                )
            return result

        # `bv.create_database` re-homes the live view to whatever .bndb it wrote,
        # so the ORIGINAL selector (and `bn close <name>`) would stop resolving --
        # for an explicit --path copy (#256) AND for a default <binary>.bndb save,
        # which equally rekeys -t <basename> to <basename>.bndb (#285). A save is
        # persistence, not an identity move, so restore the original filename in
        # both cases. (When the target was loaded from a .bndb already, out == the
        # same file and the restore is a no-op.)
        if not _restore_filename():
            # The save landed on disk, but the live view is still re-homed to the
            # copy and we could not undo it. Surface a degraded success (rehomed)
            # rather than a clean one so callers know the original identity moved and
            # `bn close <selector>` may no longer resolve it (#256 review).
            self.targets.clear_dirty(bv)  # the bytes are persisted regardless
            return {
                "ok": True,
                "saved": True,
                "path": saved,
                "rehomed": True,
                "note": (
                    f"Saved to {saved}, but could not restore the live target's "
                    f"original filename ({filename!r}); the in-memory view is now "
                    f"homed at {str(getattr(bv.file, 'filename', ''))!r}. Re-open the "
                    "original path to restore its selector."
                ),
            }
        self.targets.clear_dirty(bv)  # mutations are now persisted (L15)
        return {"ok": True, "saved": True, "path": saved}

    def _target_info(self, selector: str | None, *, verbose: bool = False):
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
        unanalyzed = bv in _unanalyzed_views
        info = {
            **(record or {}),
            "arch": str(getattr(bv, "arch", "")),
            "platform": str(getattr(bv, "platform", "")),
            # Preferred/image base BN loaded the view at (#564). For a PIE ELF this
            # is BN's chosen preferred base (commonly 0x400000 on x86_64), NOT ELF
            # VA 0 -- dynamic tools need it to rebase a BN address to runtime
            # (runtime = module_base + (bn_addr - image_base)). bv.start is the
            # authoritative source; agents previously fell back to
            # `py exec "result['value']=hex(bv.start)"` or guessed 0x400000.
            "image_base": hex(getattr(bv, "start", 0)),
            "entry_point": hex(getattr(bv, "entry_point", 0)),
            # Machine-readable analysis state so callers can tell a --quick view
            # (strings/full function set pending `bn refresh`) or a restore-failed
            # raw .bndb (#458) from a real one.
            "analyzed": not quick and not unanalyzed,
            "analysis_state": (
                "quick" if quick else "unanalyzed" if unanalyzed else "full"
            ),
            # Pollable analysis phase/progress (#321): while a `bn refresh` runs on
            # another connection, this advances (refresh no longer holds the
            # read-blocking lock) so an agent can watch a large-target analysis
            # instead of guessing whether the bridge is wedged.
            "analysis_progress": _analysis_progress(bv),
            # Function-count summary every agent reaches for (#122).
            **_function_name_summary(bv),
        }
        # --verbose surfaces the segment map (r/w/x ranges) so reaching for it on
        # target info -- the natural reflex, since function info accepts it -- is
        # rewarded with real detail instead of an "unrecognized arguments". (F21)
        if verbose:
            info["segments"] = _segment_entries(bv)
        return info

    def _refresh(self, selector: str | None):
        # #321: hold only the write GATE (which serializes other writers) around the
        # possibly-multi-minute analysis -- NOT the exclusive target lock. BN allows
        # concurrent reads while update_analysis_and_wait() runs (verified), so
        # `target info` / `function list` / analysis-progress polls stay responsive
        # on other connections and an agent can watch progress instead of guessing
        # whether the bridge is wedged. The brief exclusive window at the end only
        # flips the per-view analysis-state flags. (refresh is @op lock="none" so it
        # self-manages locking, mirroring load_binary's unlocked analysis phase.)
        #
        # #522: resolve the target INSIDE the gate, not before it. Because refresh
        # is lock="none", a concurrent close_binary/save_database (lock="write")
        # that ran between an earlier out-of-gate resolve and the gate acquisition
        # could invalidate/close the view, leaving update_analysis_and_wait() to
        # run on a dead view (TOCTOU). Acquiring the gate first serializes refresh
        # against those writers before we ever touch the view.
        with self._write_gate:
            bv = self._resolve_view(selector)
            bv.update_analysis_and_wait()
            with self._target_lock.write():
                _quick_loaded_views.discard(bv)
                # Only clear the #458 unanalyzed flag if analysis actually produced
                # functions -- a refresh on a raw-only container (no product view)
                # is a no-op and must stay "unanalyzed", not be mislabeled "full".
                if _view_function_count(bv) > 0:
                    _unanalyzed_views.discard(bv)
        # Build the response under the READ lock: the tail read touches live view
        # state (bv.functions / arch / analysis_progress), and the instant we drop
        # the write gate a queued mutation can begin writing the same view -- so
        # serialize this read against writers like every other read op does.
        with self._target_lock.read():
            target = self._target_info(selector)
        return {"refreshed": True, "target": target}

    # ---- BridgeContext seam shims (bodies live in seam.py) --------------
    # Resolution / ABI / address-context helpers were relocated into the
    # BridgeContext seam (self.ctx). These thin delegating shims keep every
    # internal caller and the direct test call sites working unchanged.

    def _resolve_view(self, *a, **k):
        return self.ctx._resolve_view(*a, **k)

    def _find_function(self, *a, **k):
        return self.ctx._find_function(*a, **k)

    def _find_functions_by_name(self, *a, **k):
        return self.ctx._find_functions_by_name(*a, **k)

    def _resolve_scope_functions(self, *a, **k):
        return self.ctx._resolve_scope_functions(*a, **k)

    def _find_symbols_by_name(self, *a, **k):
        return self.ctx._find_symbols_by_name(*a, **k)

    def _resolve_rename_target(self, *a, **k):
        return self.ctx._resolve_rename_target(*a, **k)

    def _functions_containing(self, *a, **k):
        return self.ctx._functions_containing(*a, **k)

    def _sections_at(self, *a, **k):
        return self.ctx._sections_at(*a, **k)

    def _segment_at(self, *a, **k):
        return self.ctx._segment_at(*a, **k)

    def _symbol_at(self, *a, **k):
        return self.ctx._symbol_at(*a, **k)

    def _function_entry_for_address(self, *a, **k):
        return self.ctx._function_entry_for_address(*a, **k)

    def _raw_sections_at(self, *a, **k):
        return self.ctx._raw_sections_at(*a, **k)

    def _section_semantics_name(self, *a, **k):
        return self.ctx._section_semantics_name(*a, **k)

    def _address_is_code(self, *a, **k):
        return self.ctx._address_is_code(*a, **k)

    def _resolve_data_string(self, *a, **k):
        return self.ctx._resolve_data_string(*a, **k)

    def _address_context(self, *a, **k):
        return self.ctx._address_context(*a, **k)

    def _function_object_at(self, *a, **k):
        return self.ctx._function_object_at(*a, **k)

    def _safe_disassembly(self, *a, **k):
        return self.ctx._safe_disassembly(*a, **k)

    def _pointer_size(self, *a, **k):
        return self.ctx._pointer_size(*a, **k)

    def _byteorder(self, *a, **k):
        return self.ctx._byteorder(*a, **k)

    def _supports_thumb_pointer_tags(self, *a, **k):
        return self.ctx._supports_thumb_pointer_tags(*a, **k)

    def _read_pointer_value(self, *a, **k):
        return self.ctx._read_pointer_value(*a, **k)

    def _normalize_code_pointer(self, *a, **k):
        return self.ctx._normalize_code_pointer(*a, **k)

    def _find_variable_by_storage(self, *a, **k):
        return vars_mod._find_variable_by_storage(*a, **k)

    def _variable_source_name(self, *a, **k):
        return vars_mod._variable_source_name(*a, **k)

    def _variable_identifier(self, *a, **k):
        return vars_mod._variable_identifier(*a, **k)

    def _local_id(self, *a, **k):
        return vars_mod._local_id(*a, **k)

    def _variable_entry(self, *a, **k):
        return vars_mod._variable_entry(*a, **k)

    def _variable_marker(self, *a, **k):
        return vars_mod._variable_marker(*a, **k)

    def _iter_canonical_variables(self, *a, **k):
        return vars_mod._iter_canonical_variables(*a, **k)

    def _iter_hlil_variables(self, *a, **k):
        return vars_mod._iter_hlil_variables(*a, **k)

    def _format_hlil_tree(self, *a, **k):
        return il_format._format_hlil_tree(*a, **k)

    def _function_text(self, *a, **k):
        return il_format._function_text(*a, **k)

    def _instruction_length(self, *a, **k):
        return il_format._instruction_length(*a, **k)

    def _disasm_entry(self, *a, **k):
        return il_format._disasm_entry(*a, **k)

    def _structured_disasm_entries(self, *a, **k):
        return il_format._structured_disasm_entries(*a, **k)

    def _disasm_text(self, *a, **k):
        return il_format._disasm_text(*a, **k)

    def _sort_variable_entries(self, *a, **k):
        return vars_mod._sort_variable_entries(*a, **k)

    def _list_locals(self, *a, **k):
        return vars_mod._list_locals(*a, **k)

    def _find_variables_by_name(self, *a, **k):
        return vars_mod._find_variables_by_name(*a, **k)

    def _find_variable_selector(self, *a, **k):
        return vars_mod._find_variable_selector(*a, **k)

    def _function_size(self, *a, **k):
        return il_format._function_size(*a, **k)

    def _function_metadata(self, *a, **k):
        return il_format._function_metadata(*a, **k)

    def _comment_map(self, *a, **k):
        return il_format._comment_map(*a, **k)

    def _il_op_name(self, *a, **k):
        return il_format._il_op_name(*a, **k)

    def _llil_constant_value(self, *a, **k):
        return il_format._llil_constant_value(*a, **k)

    def _coerce_il_list(self, *a, **k):
        return il_format._coerce_il_list(*a, **k)

    def _iter_llil_instructions(self, *a, **k):
        return il_format._iter_llil_instructions(*a, **k)

    def _hlil_candidates_for_llil(self, *a, **k):
        return il_format._hlil_candidates_for_llil(*a, **k)

    def _il_parent(self, *a, **k):
        return il_format._il_parent(*a, **k)

    def _hlil_marker(self, *a, **k):
        return il_format._hlil_marker(*a, **k)

    def _hlil_type_name(self, *a, **k):
        return il_format._hlil_type_name(*a, **k)

    def _hlil_text_is_local(self, *a, **k):
        return il_format._hlil_text_is_local(*a, **k)

    def _hlil_condition_is_meaningful(self, *a, **k):
        return il_format._hlil_condition_is_meaningful(*a, **k)

    def _is_hlil_assignment_like(self, *a, **k):
        return il_format._is_hlil_assignment_like(*a, **k)

    def _is_hlil_control_flow(self, *a, **k):
        return il_format._is_hlil_control_flow(*a, **k)

    def _is_hlil_hard_boundary(self, *a, **k):
        return il_format._is_hlil_hard_boundary(*a, **k)

    def _is_hlil_trivial_wrapper(self, *a, **k):
        return il_format._is_hlil_trivial_wrapper(*a, **k)

    def _hlil_call_roots(self, *a, **k):
        return il_format._hlil_call_roots(*a, **k)

    def _select_local_hlil_node(self, *a, **k):
        return il_format._select_local_hlil_node(*a, **k)

    def _hlil_statement_text(self, *a, **k):
        return il_format._hlil_statement_text(*a, **k)

    def _hlil_pre_branch_condition(self, *a, **k):
        return il_format._hlil_pre_branch_condition(*a, **k)

    def _callsites_within_function(self, *a, **k):
        return read_listing._callsites_within_function(self.ctx, *a, **k)

    def _callsites(self, *a, **k):
        return read_listing._callsites(self.ctx, *a, **k)

    def _xrefs_to_address(self, *a, **k):
        return read_xrefs._xrefs_to_address(self.ctx, *a, **k)

    def _parse_function_address_bounds(self, *a, **k):
        return read_listing._parse_function_address_bounds(self.ctx, *a, **k)

    def _filtered_functions(self, *a, **k):
        return read_listing._filtered_functions(self.ctx, *a, **k)

    def _list_functions(self, *a, **k):
        return read_listing._list_functions(self.ctx, *a, **k)

    def _paged_function_result(self, *a, **k):
        return read_listing._paged_function_result(self.ctx, *a, **k)

    def _search_functions(self, *a, **k):
        return read_listing._search_functions(self.ctx, *a, **k)

    def _function_signature(self, *a, **k):
        return il_format._function_signature(*a, **k)

    def _pseudo_c_text(self, *a, **k):
        return il_format._pseudo_c_text(*a, **k)

    def _decompile_text(self, *a, **k):
        return il_format._decompile_text(*a, **k)

    def _analysis_stub_warning(self, *a, **k):
        return il_format._analysis_stub_warning(*a, **k)

    def _force_function_analysis(self, *a, **k):
        return read_decompile._force_function_analysis(self.ctx, *a, **k)

    def _decompile(self, *a, **k):
        return read_decompile._decompile(self.ctx, *a, **k)

    def _function_info(self, *a, **k):
        return read_decompile._function_info(self.ctx, *a, **k)

    def _get_prototype(self, *a, **k):
        return read_decompile._get_prototype(self.ctx, *a, **k)

    def _list_locals_for_function(self, *a, **k):
        return read_decompile._list_locals_for_function(self.ctx, *a, **k)

    def _il(self, *a, **k):
        return read_decompile._il(self.ctx, *a, **k)

    def _disasm(self, *a, **k):
        return read_decompile._disasm(self.ctx, *a, **k)

    # ------------------------------------------------------------------
    # Structured data-flow primitives (def-use / value-set / call graph)
    # ------------------------------------------------------------------

    def _il_function_for(self, *a, **k):
        return il_format._il_function_for(*a, **k)

    def _ssa_var_entry(self, *a, **k):
        return il_format._ssa_var_entry(*a, **k)

    def _collect_ssa_vars(self, *a, **k):
        return il_format._collect_ssa_vars(*a, **k)

    def _resolve_ssa_variable(self, *a, **k):
        return il_format._resolve_ssa_variable(*a, **k)

    def _structured_il(self, *a, **k):
        return read_decompile._structured_il(self.ctx, *a, **k)

    def _defuse(self, *a, **k):
        return read_decompile._defuse(self.ctx, *a, **k)

    def _serialize_pvs(self, *a, **k):
        return il_format._serialize_pvs(*a, **k)

    def _pvs_targets(self, *a, **k):
        return read_decompile._pvs_targets(self.ctx, *a, **k)

    def _resolved_calls(self, *a, **k):
        return read_decompile._resolved_calls(self.ctx, *a, **k)

    def _possible_values(self, *a, **k):
        return read_decompile._possible_values(self.ctx, *a, **k)

    def _pvs_determined(self, *a, **k):
        return il_format._pvs_determined(*a, **k)

    def _taint(self, *a, **k):
        return read_taint_slice._taint_op(self.ctx, *a, **k)

    def _taint_models(self, *a, **k):
        return read_taint_slice._taint_models_op(self.ctx, *a, **k)

    def _call_destination_value(self, *a, **k):
        return read_evidence._call_destination_value(self.ctx, *a, **k)

    def _target_entry_for_call(self, *a, **k):
        return read_evidence._target_entry_for_call(self.ctx, *a, **k)

    def _il_argument_texts(self, *a, **k):
        return read_evidence._il_argument_texts(self.ctx, *a, **k)

    @staticmethod
    def _safe_int(*a, **k):
        return read_evidence._safe_int(*a, **k)

    def _resolve_argument_value(self, *a, **k):
        return read_evidence._resolve_argument_value(self.ctx, *a, **k)

    def _call_arguments(self, *a, **k):
        return read_evidence._call_arguments(self.ctx, *a, **k)

    def _function_call_evidence(self, *a, **k):
        return read_evidence._function_call_evidence(self.ctx, *a, **k)

    def _function_thunk_summary(self, *a, **k):
        return read_evidence._function_thunk_summary(self.ctx, *a, **k)

    def _function_evidence(self, *a, **k):
        return read_evidence._function_evidence(self.ctx, *a, **k)

    def _pointer_table_for_view(self, *a, **k):
        return read_evidence._pointer_table_for_view(self.ctx, *a, **k)

    def _pointer_table(self, *a, **k):
        return read_evidence._pointer_table(self.ctx, *a, **k)

    def _call_descriptor_evidence(self, *a, **k):
        return read_evidence._call_descriptor_evidence(self.ctx, *a, **k)

    def _resolve_virtual_call(self, *a, **k):
        return read_evidence._resolve_virtual_call(self.ctx, *a, **k)

    def _hidden_surface(self, *a, **k):
        return read_evidence._hidden_surface(self.ctx, *a, **k)

    def _message_lens(self, *a, **k):
        return read_evidence._message_lens(self.ctx, *a, **k)

    def _class_list(self, *a, **k):
        return read_class._class_list(self.ctx, *a, **k)

    def _class_show(self, *a, **k):
        return read_class._class_show(self.ctx, *a, **k)

    def _iter_il_instructions(self, *a, **k):
        return il_format._iter_il_instructions(*a, **k)

    @staticmethod
    def _ssa_vars_from(*a, **k):
        return read_taint_slice._ssa_vars_from(*a, **k)

    def _build_backward_trace(self, *a, **k):
        return read_taint_slice._build_backward_trace(self.ctx, *a, **k)

    def _is_parameter_ssa_var(self, *a, **k):
        return read_taint_slice._is_parameter_ssa_var(self.ctx, *a, **k)

    def _resolve_callee(self, *a, **k):
        return read_taint_slice._resolve_callee(self.ctx, *a, **k)

    def _resolve_thunk(self, *a, **k):
        return read_taint_slice._resolve_thunk(self.ctx, *a, **k)

    @staticmethod
    def _extract_dest_address(*a, **k):
        return read_taint_slice._extract_dest_address(*a, **k)

    def _find_return_vars(self, *a, **k):
        return read_taint_slice._find_return_vars(self.ctx, *a, **k)

    def _backward_slice(self, *a, **k):
        return read_taint_slice._backward_slice(self.ctx, *a, **k)

    def _xrefs(self, *a, **k):
        return read_xrefs._xrefs(self.ctx, *a, **k)

    def _xrefs_any(self, *a, **k):
        return read_xrefs._xrefs_any(self.ctx, *a, **k)

    def _import_symbol_name(self, *a, **k):
        return read_xrefs._import_symbol_name(*a, **k)

    def _find_import_symbol(self, *a, **k):
        return read_xrefs._find_import_symbol(self.ctx, *a, **k)

    def _xrefs_import_symbol(self, *a, **k):
        return read_xrefs._xrefs_import_symbol(self.ctx, *a, **k)

    def _scan_for_calls_to(self, *a, **k):
        return read_xrefs._scan_for_calls_to(self.ctx, *a, **k)

    def _resolve_type_field(self, *a, **k):
        return read_xrefs._resolve_type_field(self.ctx, *a, **k)

    def _field_xrefs(self, *a, **k):
        return read_xrefs._field_xrefs(self.ctx, *a, **k)

    def _types(self, *a, **k):
        return read_types._types(self.ctx, *a, **k)

    def _find_type(self, *a, **k):
        # Relocated to the BridgeContext seam (cycle-break, design spec §3.2).
        return self.ctx._find_type(*a, **k)

    def _type_entry(self, *a, **k):
        # Relocated to the BridgeContext seam (joins find_type/render_type_layout
        # so read_types + the mutation engine depend only on the seam).
        return self.ctx._type_entry(*a, **k)

    def _current_type_entry(self, *a, **k):
        # Relocated to the BridgeContext seam (see _type_entry).
        return self.ctx._current_type_entry(*a, **k)

    def _type_info(self, *a, **k):
        return read_types._type_info(self.ctx, *a, **k)

    def _strings(self, *a, **k):
        return read_misc._strings(self.ctx, *a, **k)

    def _init_arrays(self, *a, **k):
        return read_evidence._init_arrays(self.ctx, *a, **k)

    def _needed_libraries(self, *a, **k):
        return read_misc._needed_libraries(*a, **k)

    def _imports(self, *a, **k):
        return read_misc._imports(self.ctx, *a, **k)

    def _exports(self, *a, **k):
        return read_misc._exports(self.ctx, *a, **k)

    def _imports_build_summary(self, *a, **k):
        return read_misc._imports_build_summary(*a, **k)

    def _sections(self, *a, **k):
        return read_misc._sections(self.ctx, *a, **k)

    def _go_functions(self, *a, **k):
        return read_go._go_functions(self.ctx, *a, **k)

    def _go_rename(self, selector, *, preview: bool = False):
        """Apply recovered Go function names from `.gopclntab` (#217). Composes the
        go_functions recovery with the batch-rename pipeline, renaming ONLY
        auto-named (``sub_``/``nullsub_``) functions BN already has at the
        pcln-derived address -- so a user's manual names are never clobbered and a
        PIE/rebase mismatch (no BN function at the address) is skipped, not
        mis-renamed.

        This intentionally bypasses the generic mutation engine for the
        homogeneous Go auto-name case. The generic path snapshots, diffs, and
        verifies thousands of one-function mutations under one exclusive lock,
        then `_go_rename` discards most of that heavy output. Direct chunked
        renaming keeps the exact auto-name guard, verifies by readback, rolls
        back preview/failure/cancel paths, and releases the target write lock
        between chunks so same-instance reads can answer (#365).

        The anti-wedge benefit is strongest at low read concurrency: a single
        concurrent reader interleaves in the gap between chunks (~one chunk of
        write latency, not the whole rename). Under heavy read fan-out the
        writer-priority lock still convoys -- each chunk re-drains the readers
        and re-blocks the next wave -- so a busy instance serializes rather than
        truly parallelizing. It never wedges or corrupts: rename, reads, and a
        client-timeout cancellation (full rollback) all complete.
        """
        with self._write_gate:
            with self._target_lock.read():
                recovered = read_go._go_functions(self.ctx, selector)
                items = recovered.get("items") or []
                bv = self._resolve_view(selector)
                get_fn = getattr(bv, "get_function_at", None)
                candidates: list[dict[str, Any]] = []
                skipped_user_named = 0
                for it in items:
                    if not it.get("defined") or not callable(get_fn):
                        continue
                    try:
                        addr = int(it["address"], 16)
                    except (KeyError, ValueError, TypeError):
                        continue
                    fn = get_fn(addr)
                    if fn is None:
                        continue
                    current = str(getattr(fn, "name", "") or "")
                    new_name = str(it.get("name") or "")
                    if not new_name or current == new_name:
                        continue
                    if not _is_go_rename_auto_name(current, addr):
                        skipped_user_named += 1
                        continue
                    candidates.append({
                        "address": addr,
                        "before_name": current,
                        "new_name": new_name,
                    })

            if not candidates:
                return {"kind": "go_rename", "success": True, "committed": False,
                        "preview": preview, "results": [],
                        "go_renamed_candidates": 0, "skipped_user_named": skipped_user_named,
                        "defined_count": recovered.get("defined_count", 0)}

            return self._apply_go_renames_chunked(
                bv,
                candidates,
                preview=preview,
                skipped_user_named=skipped_user_named,
                defined_count=recovered.get("defined_count", 0),
            )

    def _go_rename_failure_row(
        self,
        candidate: dict[str, Any],
        status: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "op": "rename_symbol",
            "address": hex(int(candidate["address"])),
            "new_name": candidate.get("new_name"),
            "status": status,
            "message": message,
        }

    def _rollback_go_renames(self, bv, applied: list[dict[str, Any]]) -> bool:
        ok = True
        get_fn = getattr(bv, "get_function_at", None)
        if not callable(get_fn):
            return False
        for start in range(len(applied), 0, -GO_RENAME_CHUNK_SIZE):
            chunk = applied[max(0, start - GO_RENAME_CHUNK_SIZE):start]
            with self._target_lock.write():
                for candidate in reversed(chunk):
                    addr = int(candidate["address"])
                    fn = get_fn(addr)
                    if fn is None:
                        ok = False
                        continue
                    current = str(getattr(fn, "name", "") or "")
                    if current != candidate["before_name"]:
                        fn.name = candidate["before_name"]
                        current = str(getattr(fn, "name", "") or "")
                    if current != candidate["before_name"]:
                        ok = False
        return ok

    def _go_rename_cancelled(self, bv, applied: list[dict[str, Any]]) -> None:
        rolled_back = self._rollback_go_renames(bv, applied)
        suffix = "" if rolled_back else " (rollback failed; the view may be partially renamed)"
        raise RuntimeError(f"request cancelled during go rename{suffix}")

    def _apply_go_renames_chunked(
        self,
        bv,
        candidates: list[dict[str, Any]],
        *,
        preview: bool,
        skipped_user_named: int,
        defined_count: int,
    ) -> dict[str, Any]:
        get_fn = getattr(bv, "get_function_at", None)
        if not callable(get_fn):
            return {"kind": "go_rename", "success": False, "committed": False,
                    "preview": preview, "rolled_back": True,
                    "results": [{"op": "rename_symbol", "status": "unsupported",
                                 "message": "BinaryView does not support get_function_at"}],
                    "go_renamed_candidates": len(candidates),
                    "go_verified_count": 0, "go_failed_count": 1,
                    "go_committed_count": 0,
                    "skipped_user_named": skipped_user_named,
                    "defined_count": defined_count}

        applied: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        verified_count = 0
        skipped_during_apply = 0

        for start in range(0, len(candidates), GO_RENAME_CHUNK_SIZE):
            if _request_cancelled():
                self._go_rename_cancelled(bv, applied)
            chunk = candidates[start:start + GO_RENAME_CHUNK_SIZE]
            with self._target_lock.write():
                for candidate in chunk:
                    addr = int(candidate["address"])
                    fn = get_fn(addr)
                    if fn is None:
                        failed_rows.append(self._go_rename_failure_row(
                            candidate, "verification_failed", "Function missing before rename",
                        ))
                        break
                    current = str(getattr(fn, "name", "") or "")
                    if current == candidate["new_name"]:
                        continue
                    if current != candidate["before_name"]:
                        if _is_go_rename_auto_name(current, addr):
                            candidate = dict(candidate)
                            candidate["before_name"] = current
                        else:
                            skipped_during_apply += 1
                            continue
                    applied.append(candidate)
                    try:
                        fn.name = candidate["new_name"]
                    except Exception as exc:  # noqa: BLE001 - preserve all-or-nothing rename
                        failed_rows.append(self._go_rename_failure_row(
                            candidate, "verification_failed",
                            f"Live rename failed: {_serialize_error(exc)}",
                        ))
                        break
                    observed = str(getattr(fn, "name", "") or "")
                    if observed != candidate["new_name"]:
                        failed_rows.append(self._go_rename_failure_row(
                            candidate, "verification_failed", "Live rename readback disagreed",
                        ))
                        break
                    verified_count += 1
            if failed_rows:
                break

        rolled_back = True
        if preview or failed_rows:
            rolled_back = self._rollback_go_renames(bv, applied)
        committed = bool((not preview) and not failed_rows)
        success = bool(not failed_rows and (not preview or rolled_back))
        if committed and verified_count:
            self.targets.mark_dirty(bv)

        skipped_total = skipped_user_named + skipped_during_apply
        result = {
            "kind": "go_rename",
            "success": success,
            "committed": committed,
            "preview": preview,
            "results": failed_rows,
            "go_renamed_candidates": len(candidates),
            "go_verified_count": verified_count,
            "go_failed_count": len(failed_rows),
            "go_committed_count": verified_count if committed else 0,
            "skipped_user_named": skipped_total,
            "defined_count": defined_count,
            "chunk_size": GO_RENAME_CHUNK_SIZE,
        }
        if preview or failed_rows:
            result["rolled_back"] = rolled_back
        if skipped_during_apply:
            result["skipped_changed_during_apply"] = skipped_during_apply
        if failed_rows and not rolled_back:
            result["message"] = "Rollback failed; the view may be partially renamed"
        elif preview and not rolled_back:
            result["message"] = "Preview rollback failed; the view may be partially renamed"
        return result

    def _ascii_render(self, *a, **k):
        return read_misc._ascii_render(*a, **k)

    def _read(self, *a, **k):
        return read_misc._read(self.ctx, *a, **k)

    def _is_executable_address(self, *a, **k):
        return read_misc._is_executable_address(self.ctx, *a, **k)

    def _function_create(self, *a, **k):
        return create_comments._function_create(self.ctx, *a, **k)

    def _get_comment(self, *a, **k):
        return create_comments._get_comment(self.ctx, *a, **k)

    def _list_comments(self, *a, **k):
        return create_comments._list_comments(self.ctx, *a, **k)

    def _list_tag_types(self, *a, **k):
        return read_tags._list_tag_types(self.ctx, *a, **k)

    def _get_tags(self, *a, **k):
        return read_tags._get_tags(self.ctx, *a, **k)

    def _list_tags(self, *a, **k):
        return read_tags._list_tags(self.ctx, *a, **k)

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

    def _orient_digest(self, selector: str | None, *, strings_limit: int = 20):
        """One internally-consistent orientation/triage digest (#169 Layer 2):
        target + analysis state, imports summary, a bounded strings sample,
        function count, and sections -- composed under a single read lock so no
        writer interleaves between the sub-reads (the guarantee a shell loop of
        the same commands can't give). Surfaces ``analyzed`` up front and degrades
        the strings sample honestly on a --quick view rather than erroring the
        whole digest."""
        target = self._target_info(selector)
        analyzed = bool(target.get("analyzed", True))
        imports_summary = read_misc._imports(self.ctx, selector, summary=True)
        # Orient samples higher-signal strings (min-length 6) than `bn strings`
        # (BN default ~4), so its `total` is smaller; surfaced as
        # `strings_min_length` so the discrepancy is disclosed, not a mystery (#357).
        strings_min_length = 6
        if analyzed:
            try:
                strings_sample = read_misc._strings(
                    self.ctx, selector, query=None, offset=0,
                    limit=strings_limit, min_length=strings_min_length,
                )
            except RuntimeError as exc:
                strings_sample = {"unavailable": str(exc)}
        else:
            strings_sample = {
                "unavailable": "target loaded with --quick (no analysis); run "
                               "`bn refresh` for strings"
            }
        func_count = read_listing._list_functions(self.ctx, selector, count_only=True)
        sections = read_misc._sections(self.ctx, selector)
        # #561: disclose annotations ALREADY present in the view. On a cached/shared
        # BNDB, inherited comments/names bias analysis and let an agent over-credit
        # itself; surface bounded counts + a provenance hint so the inherited baseline
        # is visible up front instead of requiring a separate `bn comment list` pass.
        # Annotation counting is best-effort -- if the view can't be resolved/read it
        # degrades to an `unavailable` marker rather than erroring the whole digest.
        filename = str(target.get("filename", "") or "")
        analysis_cache_restored = filename.endswith(".bndb")
        try:
            bv = self._resolve_view(selector)
            annotations = read_listing._annotation_summary(self.ctx, bv)
            total_annotations = (
                annotations["comments"] + annotations["function_comments"]
                + annotations["user_symbols"]
            )
            hint = None
            if analysis_cache_restored or total_annotations:
                hint = (
                    f"existing BNDB annotations may predate this run: "
                    f"{annotations['comments']} comment(s), "
                    f"{annotations['function_comments']} function doc(s), "
                    f"{annotations['user_symbols']} user symbol(s) already present"
                    + (" (analysis cache restored from a .bndb)" if analysis_cache_restored else "")
                    + " -- do not over-credit current-run analysis"
                )
            existing_annotations = {
                **annotations,
                "analysis_cache_restored": analysis_cache_restored,
                "provenance_hint": hint,
            }
        except Exception as exc:
            existing_annotations = {
                "unavailable": f"annotation counts unavailable: {exc}",
                "analysis_cache_restored": analysis_cache_restored,
            }
        return {
            "kind": "orient_digest",
            "target": target,
            "analyzed": analyzed,
            "analysis_state": target.get("analysis_state"),
            "imports_summary": imports_summary,
            "strings_sample": strings_sample,
            "strings_min_length": strings_min_length,
            "function_count": func_count.get("total", func_count.get("count")),
            "sections": sections,
            "existing_annotations": existing_annotations,
        }

    def _normalize_py_result(self, *a, **k):
        return create_comments._normalize_py_result(self.ctx, *a, **k)

    def _py_exec(self, *a, **k):
        return create_comments._py_exec(self.ctx, *a, **k)

    def _render_warnings(self, *a, **k):
        return il_format._render_warnings(*a, **k)

    def _render_type_layout(self, *a, **k) -> str:
        # Relocated to the BridgeContext seam (cycle-break, design spec §3.2).
        return self.ctx._render_type_layout(*a, **k)

    # ---- mutation engine: delegating shims ----
    # Every method body moved to mutation_engine.py as a free function taking
    # the BridgeContext (self.ctx). These thin shims keep the bridge method
    # names alive so the op binders and the test suite (which call many of
    # these directly) keep working -- the existing _taint -> taint_engine
    # pattern.
    def _guess_type_affected_functions(self, *a, **k):
        return mutation_engine._guess_type_affected_functions(self.ctx, *a, **k)

    def _parse_declaration_source(self, *a, **k):
        return mutation_engine._parse_declaration_source(self.ctx, *a, **k)

    def _operation_type_names(self, *a, **k):
        return mutation_engine._operation_type_names(self.ctx, *a, **k)

    def _guess_affected_functions(self, *a, **k):
        return mutation_engine._guess_affected_functions(self.ctx, *a, **k)

    def _affected_type_names(self, *a, **k):
        return mutation_engine._affected_type_names(self.ctx, *a, **k)

    def _capture_type_snapshots(self, *a, **k):
        return mutation_engine._capture_type_snapshots(self.ctx, *a, **k)

    def _diff_type_snapshots(self, *a, **k):
        return mutation_engine._diff_type_snapshots(self.ctx, *a, **k)

    def _annotate_operation_results(self, *a, **k):
        return mutation_engine._annotate_operation_results(self.ctx, *a, **k)

    def _capture_function_snapshots(self, *a, **k):
        return mutation_engine._capture_function_snapshots(self.ctx, *a, **k)

    def _snippet_for_change(self, *a, **k):
        return mutation_engine._snippet_for_change(self.ctx, *a, **k)

    def _diff_snapshots(self, *a, **k):
        return mutation_engine._diff_snapshots(self.ctx, *a, **k)

    def _operation_requested(self, *a, **k):
        return mutation_engine._operation_requested(self.ctx, *a, **k)

    def _operation_failure_result(self, *a, **k):
        return mutation_engine._operation_failure_result(self.ctx, *a, **k)

    def _mark_unverified_results(self, *a, **k):
        return mutation_engine._mark_unverified_results(self.ctx, *a, **k)

    def _has_failed_results(self, *a, **k):
        return mutation_engine._has_failed_results(self.ctx, *a, **k)

    def _first_overlapping_member(self, *a, **k):
        return mutation_engine._first_overlapping_member(self.ctx, *a, **k)

    def _find_member(self, *a, **k):
        return mutation_engine._find_member(self.ctx, *a, **k)

    def _verify_operation(self, *a, **k):
        return mutation_engine._verify_operation(self.ctx, *a, **k)

    def _verify_rename_symbol(self, *a, **k):
        return mutation_engine._verify_rename_symbol(self.ctx, *a, **k)

    def _verify_set_comment(self, *a, **k):
        return mutation_engine._verify_set_comment(self.ctx, *a, **k)

    def _verify_delete_comment(self, *a, **k):
        return mutation_engine._verify_delete_comment(self.ctx, *a, **k)

    def _verify_set_prototype(self, *a, **k):
        return mutation_engine._verify_set_prototype(self.ctx, *a, **k)

    def _verify_local_rename(self, *a, **k):
        return mutation_engine._verify_local_rename(self.ctx, *a, **k)

    def _verify_local_retype(self, *a, **k):
        return mutation_engine._verify_local_retype(self.ctx, *a, **k)

    def _verify_struct_field_set(self, *a, **k):
        return mutation_engine._verify_struct_field_set(self.ctx, *a, **k)

    def _verify_struct_field_rename(self, *a, **k):
        return mutation_engine._verify_struct_field_rename(self.ctx, *a, **k)

    def _verify_struct_field_delete(self, *a, **k):
        return mutation_engine._verify_struct_field_delete(self.ctx, *a, **k)

    def _verify_declared_types(self, *a, **k):
        return mutation_engine._verify_declared_types(self.ctx, *a, **k)

    def _verify_tag_add(self, *a, **k):
        return mutation_engine._verify_tag_add(self.ctx, *a, **k)

    def _verify_tag_remove(self, *a, **k):
        return mutation_engine._verify_tag_remove(self.ctx, *a, **k)

    def _verify_tag_type_create(self, *a, **k):
        return mutation_engine._verify_tag_type_create(self.ctx, *a, **k)

    def _verify_tag_type_remove(self, *a, **k):
        return mutation_engine._verify_tag_type_remove(self.ctx, *a, **k)

    def _apply_operation(self, *a, **k):
        return mutation_engine._apply_operation(self.ctx, *a, **k)

    def _revert_undo_safely(self, *a, **k):
        return mutation_engine._revert_undo_safely(self.ctx, *a, **k)

    def _find_var_for_restore(self, *a, **k):
        return mutation_engine._find_var_for_restore(self.ctx, *a, **k)

    def _capture_local_var_snapshots(self, *a, **k):
        return mutation_engine._capture_local_var_snapshots(self.ctx, *a, **k)

    def _restore_local_var_drift(self, *a, **k):
        return mutation_engine._restore_local_var_drift(self.ctx, *a, **k)

    def _run_local_restores(self, *a, **k):
        return mutation_engine._run_local_restores(self.ctx, *a, **k)

    def _mutation(self, *a, **k):
        try:
            result = mutation_engine._mutation(self.ctx, *a, **k)
        except Exception as exc:
            # #630 round 2: a post-apply exception path restores the prototype
            # VALUE but cannot clear the has_user_type override an applied
            # set_prototype pinned. That residue leaves the view modified even
            # though _mutation raised (returns no result), so mark the view dirty
            # here -- otherwise `close` reads bv.file.modified == false and reports
            # no unsaved state. The original exception still propagates intact.
            if getattr(exc, "prototype_user_type_residue", False):
                try:
                    selector = a[0] if a else k.get("selector")
                    self.targets.mark_dirty(self.targets.resolve(selector))
                except Exception:
                    pass
            raise
        # A committed (non-preview) write that actually changed state leaves the
        # view dirty until saved -- mark it so `close` can warn. A pure no-op
        # (every op already in the requested state) changes nothing, so it does
        # not dirty the view. (L15)
        committed_change = (
            isinstance(result, dict)
            and result.get("committed")
            and not result.get("preview")
            and any(
                isinstance(r, dict) and r.get("status") == "verified"
                for r in (result.get("results") or [])
            )
        )
        # A FAILED rollback (preview or live) can leave partial state live in the
        # view while committed is False, so the committed check above never fires.
        # bv.file.modified does not flip for these writes, so mark dirty here or
        # `bn close` computes unsaved=false and silently discards the leftover
        # renames/types/locals (#606). Identity check, not falsy: a clean result
        # with no rolled_back key (rolled_back is None) must be unchanged, and a
        # clean rollback (rolled_back True) leaves the view as before.
        rollback_left_state = isinstance(result, dict) and result.get("rolled_back") is False
        # An unclearable has_user_type override (a proto set on an AUTO function
        # that had to be reverted) leaves the view modified even though the
        # prototype value round-tripped. It now also flips rolled_back to False,
        # but key on the residue field explicitly too so `bn close` never silently
        # discards it (#630).
        residue_left_state = isinstance(result, dict) and bool(
            result.get("prototype_user_type_residue")
        )
        if committed_change or rollback_left_state or residue_left_state:
            try:
                selector = a[0] if a else k.get("selector")
                self.targets.mark_dirty(self.targets.resolve(selector))
            except Exception:
                pass
        return result

    def _op_rename_symbol(self, *a, **k):
        return mutation_engine._op_rename_symbol(self.ctx, *a, **k)

    def _op_set_comment(self, *a, **k):
        return mutation_engine._op_set_comment(self.ctx, *a, **k)

    def _op_delete_comment(self, *a, **k):
        return mutation_engine._op_delete_comment(self.ctx, *a, **k)

    def _op_set_prototype(self, *a, **k):
        return mutation_engine._op_set_prototype(self.ctx, *a, **k)

    def _register_prototype_restore(self, *a, **k):
        return mutation_engine._register_prototype_restore(self.ctx, *a, **k)

    def _register_local_restore(self, *a, **k):
        return mutation_engine._register_local_restore(self.ctx, *a, **k)

    def _op_local_rename(self, *a, **k):
        return mutation_engine._op_local_rename(self.ctx, *a, **k)

    def _op_local_retype(self, *a, **k):
        return mutation_engine._op_local_retype(self.ctx, *a, **k)

    def _struct_builder(self, *a, **k):
        return mutation_engine._struct_builder(self.ctx, *a, **k)

    def _commit_struct_builder(self, *a, **k):
        return mutation_engine._commit_struct_builder(self.ctx, *a, **k)

    def _op_struct_field_set(self, *a, **k):
        return mutation_engine._op_struct_field_set(self.ctx, *a, **k)

    def _resolve_struct_field(self, *a, **k):
        return mutation_engine._resolve_struct_field(self.ctx, *a, **k)

    def _op_struct_field_rename(self, *a, **k):
        return mutation_engine._op_struct_field_rename(self.ctx, *a, **k)

    def _op_struct_field_delete(self, *a, **k):
        return mutation_engine._op_struct_field_delete(self.ctx, *a, **k)

    def _op_types_declare(self, *a, **k):
        return mutation_engine._op_types_declare(self.ctx, *a, **k)

    def _op_tag_add(self, *a, **k):
        return mutation_engine._op_tag_add(self.ctx, *a, **k)

    def _op_tag_remove(self, *a, **k):
        return mutation_engine._op_tag_remove(self.ctx, *a, **k)

    def _op_tag_type_create(self, *a, **k):
        return mutation_engine._op_tag_type_create(self.ctx, *a, **k)

    def _op_tag_type_remove(self, *a, **k):
        return mutation_engine._op_tag_type_remove(self.ctx, *a, **k)

_bridge: BinaryNinjaBridge | None = None
# Mutable view-tracking globals now live in bridge_state.py so read-op modules
# can read them without importing bridge. Re-imported here as the SAME objects
# (tests and handlers mutate bridge._headless_views / _quick_loaded_views in
# place, so every importer must share one object).
from .bridge_state import (  # noqa: E402
    _headless_views,
    _headless_views_lock,
    _quick_loaded_views,
    _unanalyzed_views,
)


# ---- op binders: each reproduces one former _dispatch_on_main if-arm verbatim
# (self -> bridge). dispatch() and _dispatch_on_main route through this registry
# (REGISTRY.spec(op)); READ_LOCKED_OPS/WRITE_LOCKED_OPS are derived from it. ----
# Clear first so re-executing this module (the test harness's _load_bridge re-runs
# the whole file via exec_module against this same global registry) re-registers
# the identical op set instead of tripping the duplicate-registration guard.
REGISTRY.clear()


@op("doctor", lock="read")
def _bind_doctor(bridge, params, target):
    return bridge._doctor()


@op("list_targets", lock="read")
def _bind_list_targets(bridge, params, target):
    return bridge.targets.refresh()


@op("target_info", lock="read")
def _bind_target_info(bridge, params, target):
    return bridge._target_info(params.get("selector") or target, verbose=_validate_bool(params.get("verbose"), label="verbose", default=False))


@op("refresh", lock="none")  # #321: self-manages locking -- analysis runs holding
def _bind_refresh(bridge, params, target):  # only the write gate so reads stay live
    return bridge._refresh(target)


@op("shutdown", lock="none")
def _bind_shutdown(bridge, params, target):
    bridge._shutdown_event.set()
    return {"shutting_down": True}


@op("cancel_request", lock="none")
def _bind_cancel_request(bridge, params, target):
    return bridge._cancel_request(params.get("request_id"))


@op("load_binary", lock="none")
def _bind_load_binary(bridge, params, target):
    return bridge._load_binary(
        str(params["path"]),
        prefer_bndb=_validate_bool(params.get("prefer_bndb"), label="prefer_bndb", default=True),
        quick=_validate_bool(params.get("quick"), label="quick", default=False),
        workdir=params.get("workdir"),
        no_marker=_validate_bool(params.get("no_marker"), label="no_marker", default=False),
        marker_refresh_only=_validate_bool(params.get("marker_refresh_only"),
                                           label="marker_refresh_only", default=False),
    )


@op("close_binary", lock="write")
def _bind_close_binary(bridge, params, target):
    return bridge._close_binary(
        params.get("path"),
        target,
        _validate_bool(params.get("all"), label="all", default=False),
    )


@op("save_database", lock="write")
def _bind_save_database(bridge, params, target):
    return bridge._save_database(target, params.get("path"))


@op("list_functions", lock="read")
def _bind_list_functions(bridge, params, target):
    return bridge._list_functions(
        target,
        min_address=params.get("min_address"),
        max_address=params.get("max_address"),
        min_size=params.get("min_size"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
        sort=str(params.get("sort", "address")),
        reverse=_validate_bool(params.get("reverse"), label="reverse", default=False),
    )


@op("search_functions", lock="read")
def _bind_search_functions(bridge, params, target):
    return bridge._search_functions(
        target,
        str(params.get("query", "")),
        regex=_validate_bool(params.get("regex"), label="regex", default=False),
        exact=_validate_bool(params.get("exact"), label="exact", default=False),
        word=_validate_bool(params.get("word"), label="word", default=False),
        min_address=params.get("min_address"),
        max_address=params.get("max_address"),
        min_size=params.get("min_size"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
        sort=str(params.get("sort", "address")),
        reverse=_validate_bool(params.get("reverse"), label="reverse", default=False),
    )


@op("callsites", lock="read")
def _bind_callsites(bridge, params, target):
    return bridge._callsites(
        target,
        str(params["callee"]),
        within_identifiers=list(params.get("within_identifiers") or []),
        context=int(params.get("context", 3)),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("function_info", lock="read")
def _bind_function_info(bridge, params, target):
    return bridge._function_info(target, params["identifier"])


@op("get_prototype", lock="read")
def _bind_get_prototype(bridge, params, target):
    return bridge._get_prototype(target, params["identifier"])


@op("list_locals", lock="read")
def _bind_list_locals(bridge, params, target):
    return bridge._list_locals_for_function(target, params["identifier"])


@op("decompile", lock="read",
    escalation=lambda p: _validate_bool(p.get("force_analysis"), label="force_analysis", default=False))
def _bind_decompile(bridge, params, target):
    return bridge._decompile(
        target,
        params["identifier"],
        addresses=_validate_bool(params.get("addresses"), label="addresses", default=False),
        force_analysis=_validate_bool(params.get("force_analysis"), label="force_analysis", default=False),
    )


@op("il", lock="read")
def _bind_il(bridge, params, target):
    return bridge._il(target, params["identifier"], str(params.get("view", "hlil")), _validate_bool(params.get("ssa"), label="ssa", default=False))


@op("structured_il", lock="read")
def _bind_structured_il(bridge, params, target):
    return bridge._structured_il(
        target,
        params["identifier"],
        view=str(params.get("view", "mlil")),
        ssa=_validate_bool(params.get("ssa"), label="ssa", default=True),
    )


@op("class_list", lock="read")
def _bind_class_list(bridge, params, target):
    return bridge._class_list(
        target,
        query=params.get("query"),
        include_all=_validate_bool(params.get("include_all"), label="include_all", default=False),
        no_stl=_validate_bool(params.get("no_stl"), label="no_stl", default=False),
        no_vendor=_validate_bool(params.get("no_vendor"), label="no_vendor", default=False),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("class_show", lock="read")
def _bind_class_show(bridge, params, target):
    return bridge._class_show(target, str(params["name"]))


@op("defuse", lock="read")
def _bind_defuse(bridge, params, target):
    return bridge._defuse(target, params["identifier"], str(params["var"]))


@op("resolved_calls", lock="read")
def _bind_resolved_calls(bridge, params, target):
    return bridge._resolved_calls(
        target,
        params["identifier"],
        direction=str(params.get("direction", "both")),
        resolve_indirect=_validate_bool(params.get("resolve_indirect"), label="resolve_indirect", default=True),
    )


@op("possible_values", lock="read")
def _bind_possible_values(bridge, params, target):
    return bridge._possible_values(target, params["identifier"], params["at"])


@op("taint", lock="read")
def _bind_taint(bridge, params, target):
    return bridge._taint(target, params)


@op("taint_models", lock="read")
def _bind_taint_models(bridge, params, target):
    return bridge._taint_models(target, params)


@op("disasm", lock="read")
def _bind_disasm(bridge, params, target):
    return bridge._disasm(target, params["identifier"], linear=params.get("linear"),
                          mode=params.get("mode"),
                          snap_to_instruction=params.get("snap_to_instruction", False))


@op("function_evidence", lock="read")
def _bind_function_evidence(bridge, params, target):
    aw = params.get("address_window")
    window = None
    if aw is not None:
        # "A:B" -> (int, int); accept hex or decimal, half-open [A, B).
        parts = str(aw).split(":", 1)
        if len(parts) != 2:
            raise OperationFailure("invalid_request",
                                   f"--address-window must be A:B, got {aw!r}")
        window = (_parse_address(parts[0]), _parse_address(parts[1]))
        if window[1] <= window[0]:
            raise OperationFailure("invalid_request",
                                   f"--address-window end must exceed start: {aw!r}")
    return bridge._function_evidence(
        target,
        params["identifier"],
        context=int(params.get("context", 2)),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        address_window=window,
    )


@op("xrefs", lock="read")
def _bind_xrefs(bridge, params, target):
    return bridge._xrefs(
        target,
        params["identifier"],
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        fn_pointer_scan=_validate_bool(
            params.get("fn_pointer_scan"), label="fn_pointer_scan", default=False
        ),
    )


@op("xrefs_any", lock="read")
def _bind_xrefs_any(bridge, params, target):
    return bridge._xrefs_any(target, list(params.get("symbols") or []))


@op("field_xrefs", lock="read")
def _bind_field_xrefs(bridge, params, target):
    return bridge._field_xrefs(
        target,
        str(params["field"]),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("pointer_table", lock="read")
def _bind_pointer_table(bridge, params, target):
    return bridge._pointer_table(
        target,
        params["address"],
        entries=int(params.get("entries", 16)),
        stride=params.get("stride"),
        width=params.get("width"),
        record_size=params.get("record_size"),
        ptr_fields=params.get("ptr_fields"),
        fields=params.get("fields"),
    )


@op("call_descriptors", lock="read")
def _bind_call_descriptors(bridge, params, target):
    return bridge._call_descriptor_evidence(
        target,
        params["identifier"],
        arg_index=int(params.get("arg_index", 0)),
        field_specs=params.get("fields"),
    )


@op("resolve_virtual_call", lock="read")
def _bind_resolve_virtual_call(bridge, params, target):
    return bridge._resolve_virtual_call(
        target,
        params["at"],
        providers=params.get("providers"),
    )


@op("hidden_surface", lock="read")
def _bind_hidden_surface(bridge, params, target):
    return bridge._hidden_surface(
        target,
        table_min_run=int(params.get("table_min_run", 3)),
        max_tables=int(params.get("max_tables", 64)),
        max_candidates=int(params.get("max_candidates", 128)),
        max_scan_bytes=int(params.get("max_scan_bytes", 16_000_000)),
    )


@op("message_lens", lock="read")
def _bind_message_lens(bridge, params, target):
    return bridge._message_lens(
        target,
        str(params["query"]),
        limit=int(params.get("limit", 20)),
        table_entries=int(params.get("table_entries", 6)),
    )


@op("init_arrays", lock="read")
def _bind_init_arrays(bridge, params, target):
    return bridge._init_arrays(
        target,
        limit=int(params.get("limit", 64)),
    )


@op("backward_slice", lock="read")
def _bind_backward_slice(bridge, params, target):
    return bridge._backward_slice(
        target,
        str(params["identifier"]),
        str(params["address"]),
        arg_index=int(params.get("arg_index", 0)),
        view=str(params.get("view", "mlil")),
        max_depth=int(params.get("max_depth", 50)),
        interprocedural=_validate_bool(params.get("interprocedural"), label="interprocedural", default=False),
        ip_depth=int(params.get("ip_depth", 2)),
    )


@op("types", lock="read")
def _bind_types(bridge, params, target):
    return bridge._types(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        # limit absent -> None ("no limit"), matching strings/imports/sections so
        # `types --out` exports the full body instead of capping at 100 (#165).
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("type_info", lock="read")
def _bind_type_info(bridge, params, target):
    return bridge._type_info(
        target,
        str(params["type_name"]),
        require_struct=_validate_bool(params.get("require_struct"), label="require_struct", default=False),
    )


@op("strings", lock="read")
def _bind_strings(bridge, params, target):
    return bridge._strings(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        # limit=None means "no limit" -- match the imports/sections binders so a
        # raw-socket / py exec caller that omits limit gets every string (#122).
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        min_length=int(params["min_length"]) if params.get("min_length") is not None else None,
        max_length=int(params["max_length"]) if params.get("max_length") is not None else None,
        section=params.get("section"),
        no_crt=_validate_bool(params.get("no_crt"), label="no_crt", default=False),
        regex=_validate_bool(params.get("regex"), label="regex", default=False),
        probable_format_strings=_validate_bool(
            params.get("probable_format_strings"), label="probable_format_strings", default=False
        ),
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("imports", lock="read")
def _bind_imports(bridge, params, target):
    return bridge._imports(
        target,
        summary=_validate_bool(params.get("summary"), label="summary", default=False),
        query=params.get("query"),
        regex=_validate_bool(params.get("regex"), label="regex", default=False),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
        include_got=_validate_bool(params.get("include_got"), label="include_got", default=False),
    )


@op("list_exports", lock="read")
def _bind_exports(bridge, params, target):
    return bridge._exports(
        target,
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("sections", lock="read")
def _bind_sections(bridge, params, target):
    return bridge._sections(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("go_functions", lock="read")
def _bind_go_functions(bridge, params, target):
    return bridge._go_functions(
        target,
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
        summary=_validate_bool(params.get("summary"), label="summary", default=False),
    )


@op("go_rename", lock="none")
def _bind_go_rename(bridge, params, target):
    return bridge._go_rename(
        target,
        preview=_validate_bool(params.get("preview"), label="preview", default=False),
    )


@op("read", lock="read")
def _bind_read(bridge, params, target):
    return bridge._read(target, params["address"], int(params["length"]))


@op("function_create", lock="write")
def _bind_function_create(bridge, params, target):
    return bridge._function_create(target, params["address"], _validate_bool(params.get("preview"), label="preview", default=False))


@op("bundle_function", lock="read")
def _bind_bundle_function(bridge, params, target):
    return bridge._bundle_function(target, params["identifier"], params.get("out_path"))


@op("orient_digest", lock="read")
def _bind_orient_digest(bridge, params, target):
    return bridge._orient_digest(
        target,
        strings_limit=int(params["strings_limit"]) if params.get("strings_limit") is not None else 20,
    )


@op("py_exec", lock="write")
def _bind_py_exec(bridge, params, target):
    return bridge._py_exec(target, str(params["script"]))


# #525: every single-mutation binder below builds its manifest as
# `{**params, "op": "<the_op>"}` -- the literal op is spread LAST so a
# caller-supplied `params["op"]` can never override the endpoint's own
# operation (e.g. a `set_comment` request with `params={"op":"delete_comment"}`
# must still run set_comment). Only `batch_apply` legitimately trusts
# manifest-supplied ops; these fixed-op endpoints must not.
@op("rename_symbol", lock="write")
def _bind_rename_symbol(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "rename_symbol"}])


@op("get_comment", lock="read")
def _bind_get_comment(bridge, params, target):
    return bridge._get_comment(target, params.get("address"), params.get("function"))


@op("list_comments", lock="read")
def _bind_list_comments(bridge, params, target):
    return bridge._list_comments(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        # `limit: None` ("no limit") is the CLI default and is sent as a present
        # key, so guard on the VALUE not key presence -- `"limit" in params`
        # would do int(None) and crash a bare `comment list`. Matches the
        # strings/imports/sections/types binders.
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("list_tag_types", lock="read")
def _bind_list_tag_types(bridge, params, target):
    return bridge._list_tag_types(target)


@op("get_tags", lock="read")
def _bind_get_tags(bridge, params, target):
    return bridge._get_tags(target, params.get("address"), params.get("function"))


@op("list_tags", lock="read")
def _bind_list_tags(bridge, params, target):
    return bridge._list_tags(
        target,
        function=params.get("function"),
        address=params.get("address"),
        type=params.get("type"),
        data_only=bool(params.get("data_only")),
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("tag_add", lock="write")
def _bind_tag_add(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "tag_add"}])


@op("tag_remove", lock="write")
def _bind_tag_remove(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "tag_remove"}])


@op("tag_type_create", lock="write")
def _bind_tag_type_create(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "tag_type_create"}])


@op("tag_type_remove", lock="write")
def _bind_tag_type_remove(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "tag_type_remove"}])


@op("set_comment", lock="write")
def _bind_set_comment(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "set_comment"}])


@op("delete_comment", lock="write")
def _bind_delete_comment(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "delete_comment"}])


@op("set_prototype", lock="write")
def _bind_set_prototype(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "set_prototype"}])


@op("local_rename", lock="write")
def _bind_local_rename(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "local_rename"}])


@op("local_retype", lock="write")
def _bind_local_retype(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "local_retype"}])


@op("struct_field_set", lock="write")
def _bind_struct_field_set(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "struct_field_set"}])


@op("struct_field_rename", lock="write")
def _bind_struct_field_rename(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "struct_field_rename"}])


@op("struct_field_delete", lock="write")
def _bind_struct_field_delete(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "struct_field_delete"}])


@op("types_declare", lock="write")
def _bind_types_declare(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{**params, "op": "types_declare"}])


@op("batch_apply", lock="write")
def _bind_batch_apply(bridge, params, target):
    manifest = dict(params)
    preview = _validate_bool(manifest.get("preview"), label="preview", default=False)
    # Keep None as None so the single-open-target default still applies;
    # str(None) would become the bogus selector "None".
    chosen = manifest.get("target") or target
    target = str(chosen) if chosen is not None else None
    operations = list(manifest.get("ops") or [])
    return bridge._mutation(target, preview, operations)


# Derived from the op registry -- single source of truth, replacing the former
# hand-maintained literal sets. Kept as module-level names because callers/tests
# reference bridge.READ_LOCKED_OPS / bridge.WRITE_LOCKED_OPS. Must run AFTER the
# binder block above so every op is registered before the sets are derived.
#
# Read-locked ops read live BinaryViews (targets.refresh() dereferences each
# view's file/session), so they must exclude write-locked close_binary and the
# brief exclusive sections of load_binary (its open + publish; the slow analysis
# runs unlocked, and the loading view isn't published until after, so reads
# never observe a half-analyzed target -- #99). "shutdown" stays unlocked on
# purpose (lock="none"): it only sets an event and must work even while a write
# op is wedged. load_binary is intentionally not write-locked (lock="none"): it
# does its OWN fine-grained locking (exclusive only around the BN open and the
# publish, NOT around the multi-minute update_analysis_and_wait), so doctor/
# target reads stay responsive during a large load (#99). See _load_binary.
READ_LOCKED_OPS = frozenset(REGISTRY.read_locked_ops())
WRITE_LOCKED_OPS = frozenset(REGISTRY.write_locked_ops())


def _cache_bndb_path(binary_path: str) -> Path:
    """The writable-cache .bndb location for *binary_path*.

    The single source of truth for the cache fallback: `_save_database` writes
    here when a binary's own mount is read-only (no adjacent .bndb possible), and
    `_resolve_bndb_sidecar` reads here so a later load of the same binary restores
    those annotations instead of re-analyzing blank (#214 save / #318 load). The
    digest keys on the binary's path so two binaries with the same basename don't
    collide, and so save and load agree as long as the same path is used.
    """
    stem = Path(binary_path).name or "target"
    digest = hashlib.sha256(binary_path.encode("utf-8")).hexdigest()[:16]
    return cache_home() / "bndb" / f"{stem}.{digest}.bndb"


def _resolve_bndb_sidecar(resolved: Path, prefer_bndb: bool) -> tuple[Path, str] | None:
    """The saved database to load instead of ``resolved``, as ``(path, source)``
    where source is ``"adjacent"`` or ``"cache"``; or None to load the path as given.

    Shared by the runtime ``bn load`` path (`_load_binary`) and the headless
    preload path (`_preload_binary`) so both honor a saved database -- otherwise
    `bn-agent foo` silently re-analyzes from scratch and drops the saved work that
    `bn load foo` would have picked up (#178). Prefers an adjacent
    ``<path>.bndb``; if none (e.g. the binary is on a read-only mount), falls back
    to the writable-cache copy `_save_database` wrote there (#318). A request that
    already points at a ``.bndb`` (or prefer_bndb=False) resolves to None.
    """
    if prefer_bndb and resolved.suffix != ".bndb":
        sibling = Path(str(resolved) + ".bndb")
        if sibling.exists():
            return sibling, "adjacent"
        cached = _cache_bndb_path(str(resolved))
        if cached.exists():
            return cached, "cache"
    return None


def _view_function_count(bv) -> int:
    """Cheap "does this view carry analysis" probe: the count of recovered
    functions, or 0 if the attribute is missing/raises (a raw container view)."""
    try:
        return len(bv.functions)
    except Exception:  # noqa: BLE001 - a raw/uninitialised view may not expose functions
        return 0


def _analysis_progress(bv):
    """Pollable analysis progress for `target info` (#321): the current phase plus
    item counts, so an agent watching a long `bn refresh` sees movement (and, on
    another connection, can poll it while refresh runs -- refresh no longer holds
    the read-blocking lock) instead of guessing whether the bridge is wedged.
    Returns ``{"state","count","total"}`` or None if BN does not expose it."""
    try:
        ap = bv.analysis_progress
    except Exception:  # noqa: BLE001 - not all view types expose progress
        return None
    if ap is None:
        return None
    state = getattr(ap, "state", None)
    name = getattr(state, "name", None) or (str(state) if state is not None else None)
    return {
        "state": name,
        "count": int(getattr(ap, "count", 0) or 0),
        "total": int(getattr(ap, "total", 0) or 0),
    }


# View types that are the raw byte container, never the analyzed product view.
_RAW_CONTAINER_VIEW_TYPES = frozenset({"Raw", "Hex", ""})


def _restore_bndb_analyzed_view(bv, load_path: Path):
    """For a directly-named ``.bndb``, ensure the *analyzed* view is returned (#458).

    BN's ``load()`` can default to the raw byte-container view (``Raw``/``Hex``) even
    when the database holds a saved analyzed product view -- an agent that continues
    against it sees no functions/symbols and mistakes a restore failure for an empty or
    different binary. When the loaded view is a raw container, recover the analyzed
    product view from the shared ``FileMetadata`` (``existing_views`` +
    ``get_view_of_type``); if the database has no product view at all, return a hard
    diagnostic instead of a silent no-symbol target.

    The discriminator is the *view type*, not function count alone: a product view
    (ELF/Mach-O/PE/Mapped/...) with 0 functions is a legitimately-analyzed codeless or
    data region -- firmware data blobs analyze to a 0-function ``Mapped`` view -- so it
    is accepted as-is and never warned about. Only a raw *container* view signals that
    ``load()`` failed to select the saved analyzed view.

    Returns ``(bv, note_or_None, unanalyzed)`` where ``bv`` may be the recovered view
    and ``unanalyzed`` is True only when the target is a raw container with no
    recoverable analyzed view (so callers can report ``analyzed=False`` honestly).
    """
    current_type = str(getattr(bv, "view_type", "") or "")
    # A real product view (even with 0 functions) is the analyzed view -- accept it.
    if current_type not in _RAW_CONTAINER_VIEW_TYPES or _view_function_count(bv) > 0:
        return bv, None, False

    # Loaded view is a raw container: look for a saved product view to recover.
    fm = getattr(bv, "file", None)
    existing = list(getattr(fm, "existing_views", None) or [])
    best = None  # (name, view, function_count); a product view, preferring one with code
    for name in existing:
        if name in _RAW_CONTAINER_VIEW_TYPES or name == current_type:
            continue
        try:
            candidate = fm.get_view_of_type(name)
        except Exception:  # noqa: BLE001 - a view type present on disk may still fail to open
            candidate = None
        if candidate is None:
            continue
        count = _view_function_count(candidate)
        if count > 0:
            best = (name, candidate, count)
            break  # a product view WITH functions is the clear winner
        if best is None:
            best = (name, candidate, 0)  # remember the first codeless product view

    if best is not None:
        name, candidate, count = best
        if count > 0:
            note = (
                f"restored analyzed view '{name}' ({count} functions) from "
                f"{load_path.name}; load() had defaulted to the raw "
                f"'{current_type or 'Raw'}' container view"
            )
        else:
            note = (
                f"restored analyzed view '{name}' from {load_path.name} (load() had "
                f"defaulted to the raw '{current_type or 'Raw'}' container view); it has "
                "0 functions -- an analyzed but code-free/data region"
            )
        return candidate, note, False

    return bv, (
        f"WARNING: {load_path.name} restored a raw '{current_type or 'Raw'}' view with "
        "no saved analyzed view -- the database has no product view (it may have been "
        "saved before analysis, is a raw-only image, or is corrupt). Function/symbol/"
        "xref queries will be empty; this is NOT an analyzed target. Re-create the "
        "database from the original binary, or load the raw binary and run `bn refresh`."
    ), True


def _preload_binary(path: str, quick: bool, prefer_bndb: bool = True):
    """Open one headless preload binary and register it.

    Mirrors _load_binary's sidecar + --quick bookkeeping so the preload path and
    the runtime `bn load` path agree: an adjacent <binary>.bndb is loaded
    instead of re-analyzing the raw binary (the substitution is logged so it is
    visible in the headless log), a .bndb already carries its analysis (so
    --quick is a no-op there), and a genuinely quick preload is recorded in
    _quick_loaded_views so target_info/strings stay honest ("run bn refresh")
    instead of reporting full analysis or a misleading empty string list
    (#90, #178). Returns the opened BinaryView, or None if the open failed.
    """
    import binaryninja

    resolved = Path(path).expanduser().resolve()
    substitute = _resolve_bndb_sidecar(resolved, prefer_bndb)
    if substitute is not None:
        load_path, source = substitute
        if source == "cache":
            bn.log_info(
                f"restored cached database {load_path} for {resolved} "
                "(mount has no writable adjacent .bndb); pass --no-bndb to skip"
            )
        else:
            bn.log_info(f"loaded {load_path} instead of {resolved} (use --no-bndb to skip)")
    else:
        load_path = resolved
    bv = binaryninja.load(str(load_path), update_analysis=False)
    if bv is None:
        bn.log_warn(f"Failed to open binary: {load_path}")
        return None
    try:
        # #458 parity with _load_binary: a directly-named .bndb whose load() defaults
        # to the raw container view must recover its analyzed view (or log a hard
        # restore diagnostic), so `bn-agent <file>.bndb` doesn't silently preload a
        # no-symbol target.
        bndb_unanalyzed = False
        if load_path.suffix == ".bndb":
            bv, bndb_note, bndb_unanalyzed = _restore_bndb_analyzed_view(bv, load_path)
            if bndb_note:
                (bn.log_warn if bndb_note.startswith("WARNING:") else bn.log_info)(bndb_note)
        quick_effective = quick and load_path.suffix != ".bndb"
        if quick_effective:
            _quick_loaded_views.add(bv)
        else:
            bv.update_analysis_and_wait()
            _quick_loaded_views.discard(bv)
        if bndb_unanalyzed:
            _unanalyzed_views.add(bv)
        with _headless_views_lock:
            _headless_views.append(bv)
    except BaseException:
        # #609 parity with _load_binary: a view opened above but never registered
        # (analysis raised/OOM'd, .bndb restore failed) is otherwise abandoned open.
        # Close it and drop it from the bookkeeping sets, then re-raise. `bv` may
        # have been rebound to the restored analyzed view above; close whichever
        # view is current.
        with _headless_views_lock:
            published = any(v is bv for v in _headless_views)
        if not published:
            with contextlib.suppress(Exception):
                _run_on_main_thread(lambda: bv.file.close())
            _quick_loaded_views.discard(bv)
            _unanalyzed_views.discard(bv)
        raise
    bn.log_info(f"Loaded {load_path}{' (no analysis)' if quick_effective else ''}")
    return bv


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
    prefer_bndb: bool = True,
):
    """Start the bridge in headless mode (no GUI required).

    Opens any binary file paths provided, starts the socket server,
    and blocks the calling thread until shutdown is requested. With ``quick``,
    preloaded binaries skip ``update_analysis_and_wait()`` (see ``--quick``).
    With ``prefer_bndb`` (the default), a binary with an adjacent
    ``<binary>.bndb`` loads the sidecar instead of re-analyzing -- matching
    ``bn load``; pass ``--no-bndb`` (prefer_bndb=False) to open the raw binary.
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
        for path in binaries:
            _preload_binary(path, quick, prefer_bndb)
        # #524: _preload_binary appends to _headless_views but (unlike the
        # runtime `bn load` path, which calls _write_registry) never rewrites the
        # registry, so the on-disk registry still lists zero binaries while the
        # instance holds live targets. Rewrite it once after the preload loop so
        # `bn instance list` / discovery see the loaded binaries.
        _bridge._write_registry()

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
