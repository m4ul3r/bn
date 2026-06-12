from __future__ import annotations

import atexit
import contextlib
import difflib
import errno
import io
import json
import os
import re
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
from binaryninja import SSAVariable
from binaryninja.plugin import PluginCommand

from . import il_format
from . import mutation_engine
from . import read_evidence
from . import read_xrefs
from . import taint_engine as _taint
from . import vars as vars_mod
from ._shared import (
    USER_FACING_ERRORS,
    OperationFailure,
    _artifact_summary,
    _json_response,
    _normalize_prototype,
    _parse_address,
    _run_on_main_thread,
    _serialize_error,
    _validate_bool,
    _validate_count,
    _write_json_artifact,
    _CONVENTION_RE,
)
from .op_registry import REGISTRY, op
from .paths import PLUGIN_NAME, bridge_registry_path, bridge_socket_path, instances_dir
from .seam import BridgeContext
from .version import VERSION, build_id_for_file

try:
    import binaryninjaui as ui
except Exception:  # ImportError or UIPluginInHeadlessError
    ui = None


PLUGIN_BUILD_ID = build_id_for_file(Path(__file__).resolve())

# Upper bound on a single newline-terminated JSON request. Anything larger is
# rejected with a clean error instead of being buffered without limit.
MAX_REQUEST_BYTES = 32 * 1024 * 1024


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


_STRING_TYPE_NAMES: dict[int, str] = {
    0: "ascii",
    1: "utf16",
    2: "utf32",
}

# _VAR_DRIFT_OPS moved to mutation_engine.py with the mutation cluster (#33).


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
            spec = REGISTRY.spec(op)
            lock = contextlib.nullcontext()
            if spec is not None:
                lock_class = spec.lock
                # --force-analysis reanalyzes the function, mutating the view, so
                # it needs the exclusive lock even though decompile is a read op.
                if spec.lock_escalation is not None and spec.lock_escalation(params):
                    lock_class = "write"
                if lock_class == "write":
                    lock = self._target_lock.write()
                elif lock_class == "read":
                    lock = self._target_lock.read()
            with lock:
                result = self._dispatch_on_main(op, params, target)
            return _json_response(ok=True, result=result)
        except Exception as exc:
            return _json_response(ok=False, error=_serialize_error(exc))

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
        #
        # load_binary is NOT in WRITE_LOCKED_OPS (#99): hold the exclusive lock
        # only around the BN open (which touches global open-file state), then
        # run the multi-minute analysis UNLOCKED so doctor/target reads keep
        # responding. The view is published (and only then visible to reads)
        # under the lock at the end, so no reader ever sees a half-analyzed
        # target.
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

        quick_effective = quick and load_path.suffix != ".bndb"
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

        # Publish under the exclusive lock so a concurrent target read sees a
        # consistent set. Append under _headless_views_lock, then refresh
        # (which re-acquires that non-reentrant lock itself).
        with self._target_lock.write():
            with _headless_views_lock:
                _headless_views.append(bv)
            targets = self.targets.refresh()

        return {
            "loaded": True,
            "path": str(load_path),
            "requested_path": str(resolved),
            "analyzed": not quick_effective,
            "notes": notes,
            "targets": targets,
        }

    def _close_binary(self, path: str | None = None, target: str | None = None, all_: bool = False):
        def _snapshot(bv) -> dict[str, Any]:
            return {
                "path": str(getattr(bv.file, "filename", "")),
                "unsaved": bool(getattr(bv.file, "modified", False)),
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
                _run_on_main_thread(lambda v=bv: v.file.close())

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
            # Count tail-branch references too (a `b`/branch into the sink rendered
            # as `return <addr>(...) __tailcall`), not just bl/blx -- xrefs and
            # taint backward already treat these as calls, so callsites must agree
            # or it silently misses a reachable sink during triage (#47).
            if op_name not in {"LLIL_CALL", "LLIL_CALL_STACK_ADJUST", "LLIL_TAILCALL"}:
                continue
            dest_value = self._llil_constant_value(getattr(insn, "dest", None))
            if dest_value != callee_address:
                continue
            call_kind = "tailcall" if "TAILCALL" in op_name else "call"

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
                    "call_kind": call_kind,
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

    def _xrefs_to_address(self, *a, **k):
        return read_xrefs._xrefs_to_address(self.ctx, *a, **k)

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
        return self._paged_function_result(items, offset=offset, limit=limit)

    def _paged_function_result(self, items: list[dict[str, Any]], *, offset: int,
                               limit: int | None) -> dict[str, Any]:
        """Return a function-listing page WITH paging metadata.

        The CLI can't compute the true total itself -- it fetches a bounded page
        -- so the bridge, which has the full filtered set, returns total/offset/
        limit/returned/has_more alongside the page. This lets `function list`
        state the real total + remainder (text) and expose paging in JSON, the
        same honesty convention as evidence xrefs (#59)."""
        total = len(items)
        page = items[offset:]
        if limit is not None:
            page = page[:limit]
        return {
            "functions": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "has_more": (offset + len(page)) < total,
        }

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
        return self._paged_function_result(items, offset=offset, limit=limit)

    def _function_signature(self, *a, **k):
        return il_format._function_signature(*a, **k)

    def _pseudo_c_text(self, *a, **k):
        return il_format._pseudo_c_text(*a, **k)

    def _decompile_text(self, *a, **k):
        return il_format._decompile_text(*a, **k)

    def _analysis_stub_warning(self, *a, **k):
        return il_format._analysis_stub_warning(*a, **k)

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

    def _il_function_for(self, *a, **k):
        return il_format._il_function_for(*a, **k)

    def _ssa_var_entry(self, *a, **k):
        return il_format._ssa_var_entry(*a, **k)

    def _collect_ssa_vars(self, *a, **k):
        return il_format._collect_ssa_vars(*a, **k)

    def _resolve_ssa_variable(self, *a, **k):
        return il_format._resolve_ssa_variable(*a, **k)

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

    def _serialize_pvs(self, *a, **k):
        return il_format._serialize_pvs(*a, **k)

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
            # function_at normalizes the Thumb low bit so an odd value-set target
            # resolves to its function (#89 Problem B). The raw address is kept
            # in the reported "address".
            fn = _taint.function_at(bv, addr)
            name = None
            if fn is not None:
                name = str(getattr(fn, "name", None))
            else:
                sym = bv.get_symbol_at(addr) if hasattr(bv, "get_symbol_at") else None
                if sym is None and (addr & 1) and hasattr(bv, "get_symbol_at"):
                    sym = bv.get_symbol_at(addr & ~1)
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
                # resolve_call_target (no thunk following) resolves MLIL_IMPORT
                # calls and Thumb-tagged targets that const_target/exact-address
                # lookup miss (#89). Constant direct calls resolve identically.
                resolved = _taint.resolve_call_target(bv, ins, follow_thunks=False)
                target = resolved.address
                row = {
                    "call_addr": hex(int(getattr(ins, "address", func.start))),
                    "il_index": int(getattr(ins, "instr_index", -1)),
                }
                if target is not None:
                    fn = resolved.function or (bv.get_function_at(target) if hasattr(bv, "get_function_at") else None)
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

    def _pvs_determined(self, *a, **k):
        return il_format._pvs_determined(*a, **k)

    def _taint(self, selector, params: dict[str, Any]):
        bv = self._resolve_view(selector)
        direction = str(params.get("direction", "forward"))
        func = self._find_function(bv, params["function"])
        try:
            models = _taint.load_models()
        except _taint.TaintError as exc:
            # A broken builtin DB or BN_TAINT_MODELS override is now loud instead
            # of silently producing false negatives (#97).
            raise OperationFailure("unsupported", str(exc)) from exc

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

    def _message_lens(self, *a, **k):
        return read_evidence._message_lens(self.ctx, *a, **k)

    def _iter_il_instructions(self, *a, **k):
        return il_format._iter_il_instructions(*a, **k)

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

    def _xrefs(self, *a, **k):
        return read_xrefs._xrefs(self.ctx, *a, **k)

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

    def _init_arrays(self, *a, **k):
        return read_evidence._init_arrays(self.ctx, *a, **k)

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
        # Re-enforce the count contract for raw socket / py exec callers: a
        # negative offset/limit must be a clean invalid_request, not Python
        # negative-slice behavior returning a silently-wrong subset (#100).
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
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
        if function and address is not None:
            raise RuntimeError(
                "Pass --address or --function, not both: they target different locations."
            )
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
        # Re-enforce the count contract (see _sections) so a negative offset/
        # limit is a clean invalid_request, not a silent negative-slice (#100).
        offset = _validate_count(offset, label="offset", minimum=0)
        limit = _validate_count(limit, label="limit", minimum=1, allow_none=True)
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
    # pattern. _function_create (the create-cluster op) stays in the bridge.
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
        return mutation_engine._mutation(self.ctx, *a, **k)

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

_bridge: BinaryNinjaBridge | None = None
# Mutable view-tracking globals now live in bridge_state.py so read-op modules
# can read them without importing bridge. Re-imported here as the SAME objects
# (tests and handlers mutate bridge._headless_views / _quick_loaded_views in
# place, so every importer must share one object).
from .bridge_state import (  # noqa: E402
    _headless_views,
    _headless_views_lock,
    _quick_loaded_views,
)


# ---- op binders: each reproduces one former _dispatch_on_main if-arm verbatim
# (self -> bridge). Registered at import; dispatch still uses the legacy if-chain
# until Task 1.4 wires the registry in. ----
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
    return bridge._target_info(params.get("selector") or target)


@op("refresh", lock="write")
def _bind_refresh(bridge, params, target):
    return bridge._refresh(target)


@op("shutdown", lock="none")
def _bind_shutdown(bridge, params, target):
    bridge._shutdown_event.set()
    return {"shutting_down": True}


@op("load_binary", lock="none")
def _bind_load_binary(bridge, params, target):
    return bridge._load_binary(
        str(params["path"]),
        prefer_bndb=_validate_bool(params.get("prefer_bndb"), label="prefer_bndb", default=True),
        quick=_validate_bool(params.get("quick"), label="quick", default=False),
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
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if "limit" in params else None,
        count_only=bool(params.get("count_only", False)),
    )


@op("search_functions", lock="read")
def _bind_search_functions(bridge, params, target):
    return bridge._search_functions(
        target,
        str(params.get("query", "")),
        regex=bool(params.get("regex", False)),
        exact=bool(params.get("exact", False)),
        min_address=params.get("min_address"),
        max_address=params.get("max_address"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if "limit" in params else None,
    )


@op("callsites", lock="read")
def _bind_callsites(bridge, params, target):
    return bridge._callsites(
        target,
        str(params["callee"]),
        within_identifiers=list(params.get("within_identifiers") or []),
        context=int(params.get("context", 3)),
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
        addresses=bool(params.get("addresses")),
        force_analysis=bool(params.get("force_analysis")),
    )


@op("il", lock="read")
def _bind_il(bridge, params, target):
    return bridge._il(target, params["identifier"], str(params.get("view", "hlil")), bool(params.get("ssa")))


@op("structured_il", lock="read")
def _bind_structured_il(bridge, params, target):
    return bridge._structured_il(
        target,
        params["identifier"],
        view=str(params.get("view", "mlil")),
        ssa=bool(params.get("ssa", True)),
    )


@op("defuse", lock="read")
def _bind_defuse(bridge, params, target):
    return bridge._defuse(target, params["identifier"], str(params["var"]))


@op("resolved_calls", lock="read")
def _bind_resolved_calls(bridge, params, target):
    return bridge._resolved_calls(
        target,
        params["identifier"],
        direction=str(params.get("direction", "both")),
        resolve_indirect=bool(params.get("resolve_indirect", True)),
    )


@op("possible_values", lock="read")
def _bind_possible_values(bridge, params, target):
    return bridge._possible_values(target, params["identifier"], params["at"])


@op("taint", lock="read")
def _bind_taint(bridge, params, target):
    return bridge._taint(target, params)


@op("disasm", lock="read")
def _bind_disasm(bridge, params, target):
    return bridge._disasm(target, params["identifier"])


@op("function_evidence", lock="read")
def _bind_function_evidence(bridge, params, target):
    return bridge._function_evidence(
        target,
        params["identifier"],
        context=int(params.get("context", 2)),
    )


@op("xrefs", lock="read")
def _bind_xrefs(bridge, params, target):
    return bridge._xrefs(target, params["identifier"])


@op("field_xrefs", lock="read")
def _bind_field_xrefs(bridge, params, target):
    return bridge._field_xrefs(target, str(params["field"]))


@op("pointer_table", lock="read")
def _bind_pointer_table(bridge, params, target):
    return bridge._pointer_table(
        target,
        params["address"],
        entries=int(params.get("entries", 16)),
        stride=params.get("stride"),
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
        interprocedural=bool(params.get("interprocedural", False)),
        ip_depth=int(params.get("ip_depth", 2)),
    )


@op("types", lock="read")
def _bind_types(bridge, params, target):
    return bridge._types(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params.get("limit", 100)),
    )


@op("type_info", lock="read")
def _bind_type_info(bridge, params, target):
    return bridge._type_info(
        target,
        str(params["type_name"]),
        require_struct=bool(params.get("require_struct")),
    )


@op("strings", lock="read")
def _bind_strings(bridge, params, target):
    return bridge._strings(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params.get("limit", 100)),
        min_length=int(params["min_length"]) if params.get("min_length") is not None else None,
        section=params.get("section"),
        no_crt=bool(params.get("no_crt", False)),
        regex=bool(params.get("regex", False)),
    )


@op("imports", lock="read")
def _bind_imports(bridge, params, target):
    return bridge._imports(
        target,
        summary=bool(params.get("summary", False)),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("sections", lock="read")
def _bind_sections(bridge, params, target):
    return bridge._sections(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("read", lock="read")
def _bind_read(bridge, params, target):
    return bridge._read(target, params["address"], int(params["length"]))


@op("function_create", lock="write")
def _bind_function_create(bridge, params, target):
    return bridge._function_create(target, params["address"], bool(params.get("preview")))


@op("bundle_function", lock="read")
def _bind_bundle_function(bridge, params, target):
    return bridge._bundle_function(target, params["identifier"], params.get("out_path"))


@op("py_exec", lock="write")
def _bind_py_exec(bridge, params, target):
    return bridge._py_exec(target, str(params["script"]))


@op("rename_symbol", lock="write")
def _bind_rename_symbol(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "rename_symbol", **params}])


@op("get_comment", lock="read")
def _bind_get_comment(bridge, params, target):
    return bridge._get_comment(target, params.get("address"), params.get("function"))


@op("list_comments", lock="read")
def _bind_list_comments(bridge, params, target):
    return bridge._list_comments(
        target,
        query=params.get("query"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if "limit" in params else None,
    )


@op("set_comment", lock="write")
def _bind_set_comment(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "set_comment", **params}])


@op("delete_comment", lock="write")
def _bind_delete_comment(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "delete_comment", **params}])


@op("set_prototype", lock="write")
def _bind_set_prototype(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "set_prototype", **params}])


@op("local_rename", lock="write")
def _bind_local_rename(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "local_rename", **params}])


@op("local_retype", lock="write")
def _bind_local_retype(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "local_retype", **params}])


@op("struct_field_set", lock="write")
def _bind_struct_field_set(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "struct_field_set", **params}])


@op("struct_field_rename", lock="write")
def _bind_struct_field_rename(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "struct_field_rename", **params}])


@op("struct_field_delete", lock="write")
def _bind_struct_field_delete(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "struct_field_delete", **params}])


@op("types_declare", lock="write")
def _bind_types_declare(bridge, params, target):
    return bridge._mutation(target, bool(params.get("preview")), [{"op": "types_declare", **params}])


@op("batch_apply", lock="write")
def _bind_batch_apply(bridge, params, target):
    manifest = dict(params)
    preview = bool(manifest.get("preview"))
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


def _preload_binary(path: str, quick: bool):
    """Open one headless preload binary and register it.

    Mirrors _load_binary's --quick bookkeeping so the preload path and the
    runtime `bn load` path agree: a .bndb already carries its analysis (so
    --quick is a no-op there), and a genuinely quick preload is recorded in
    _quick_loaded_views so target_info/strings stay honest ("run bn refresh")
    instead of reporting full analysis or a misleading empty string list (#90).
    Returns the opened BinaryView, or None if the open failed.
    """
    import binaryninja

    resolved = Path(path).expanduser().resolve()
    bv = binaryninja.load(str(resolved), update_analysis=False)
    if bv is None:
        bn.log_warn(f"Failed to open binary: {resolved}")
        return None
    quick_effective = quick and resolved.suffix != ".bndb"
    if quick_effective:
        _quick_loaded_views.add(bv)
    else:
        bv.update_analysis_and_wait()
        _quick_loaded_views.discard(bv)
    with _headless_views_lock:
        _headless_views.append(bv)
    bn.log_info(f"Loaded {resolved}{' (no analysis)' if quick_effective else ''}")
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
        for path in binaries:
            _preload_binary(path, quick)

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
