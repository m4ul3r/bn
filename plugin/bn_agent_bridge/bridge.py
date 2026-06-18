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
from . import read_listing
from . import read_misc
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
    _run_on_main_thread,
    _serialize_error,
    _validate_bool,
    _validate_count,
    _write_json_artifact,
)
from .op_registry import REGISTRY, op
from .paths import PLUGIN_NAME, bridge_registry_path, bridge_socket_path, cache_home, instances_dir
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
                        # Per-target analysis state so `target list` can flag a
                        # --quick (unanalyzed) view without a per-target lookup.
                        "analyzed": view not in _quick_loaded_views,
                        "analysis_state": (
                            "quick" if view in _quick_loaded_views else "full"
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
    for fn in functions:
        if _is_imported_function(fn):
            imported += 1
            continue
        name = str(getattr(fn, "name", "") or "")
        if name and not _is_auto_function_name(name):
            named += 1
    return {
        "function_count": total,
        "named_function_count": named,
        "unnamed_function_count": total - named - imported,
        "imported_function_count": imported,
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
            # Whole-package fingerprint of the code this process loaded (#161).
            "engine_build_id": ENGINE_BUILD_ID,
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
        sibling = _resolve_bndb_sidecar(resolved, prefer_bndb)
        if sibling is not None:
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

        try:
            saved = _attempt(out)
        except RuntimeError as exc:
            # The common VR case is a binary on a read-only mount (firmware
            # image): the default <binary>.bndb write fails. Rather than lose the
            # annotations, retry into a writable cache dir and report where it
            # landed (#214). An EXPLICIT --path failure stays a hard error -- the
            # user chose that location, so a silent relocation would be wrong.
            if explicit:
                raise
            stem = Path(filename or out).name or "target"
            digest = hashlib.sha256((filename or out).encode("utf-8")).hexdigest()[:16]
            fallback = cache_home() / "bndb" / f"{stem}.{digest}.bndb"
            try:
                saved = _attempt(str(fallback), make_parent=True)
            except Exception:
                # ANY fallback failure (incl. an OSError from the cache-dir mkdir
                # if it too is unwritable) re-raises the ORIGINAL default-path
                # error -- that's the actionable message, not the fallback's.
                raise exc
            self.targets.clear_dirty(bv)
            return {"saved": True, "path": saved, "fallback": True, "requested_path": out}

        self.targets.clear_dirty(bv)  # mutations are now persisted (L15)
        return {"saved": True, "path": saved}

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
        info = {
            **(record or {}),
            "arch": str(getattr(bv, "arch", "")),
            "platform": str(getattr(bv, "platform", "")),
            "entry_point": hex(getattr(bv, "entry_point", 0)),
            # Machine-readable analysis state so callers can tell a --quick view
            # (strings/full function set pending `bn refresh`) from a real one.
            "analyzed": not quick,
            "analysis_state": "quick" if quick else "full",
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
        result = mutation_engine._mutation(self.ctx, *a, **k)
        # A committed (non-preview) write that actually changed state leaves the
        # view dirty until saved -- mark it so `close` can warn. A pure no-op
        # (every op already in the requested state) changes nothing, so it does
        # not dirty the view. (L15)
        if isinstance(result, dict) and result.get("committed") and not result.get("preview"):
            changed = any(
                isinstance(r, dict) and r.get("status") == "verified"
                for r in (result.get("results") or [])
            )
            if changed:
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
        min_address=params.get("min_address"),
        max_address=params.get("max_address"),
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
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
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
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
    return bridge._xrefs(
        target,
        params["identifier"],
        offset=int(params.get("offset", 0)),
        limit=int(params["limit"]) if params.get("limit") is not None else None,
    )


@op("xrefs_any", lock="read")
def _bind_xrefs_any(bridge, params, target):
    return bridge._xrefs_any(target, list(params.get("symbols") or []))


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
        width=params.get("width"),
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
        section=params.get("section"),
        no_crt=_validate_bool(params.get("no_crt"), label="no_crt", default=False),
        regex=_validate_bool(params.get("regex"), label="regex", default=False),
        count_only=_validate_bool(params.get("count_only"), label="count_only", default=False),
    )


@op("imports", lock="read")
def _bind_imports(bridge, params, target):
    return bridge._imports(
        target,
        summary=_validate_bool(params.get("summary"), label="summary", default=False),
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


@op("read", lock="read")
def _bind_read(bridge, params, target):
    return bridge._read(target, params["address"], int(params["length"]))


@op("function_create", lock="write")
def _bind_function_create(bridge, params, target):
    return bridge._function_create(target, params["address"], _validate_bool(params.get("preview"), label="preview", default=False))


@op("bundle_function", lock="read")
def _bind_bundle_function(bridge, params, target):
    return bridge._bundle_function(target, params["identifier"], params.get("out_path"))


@op("py_exec", lock="write")
def _bind_py_exec(bridge, params, target):
    return bridge._py_exec(target, str(params["script"]))


@op("rename_symbol", lock="write")
def _bind_rename_symbol(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "rename_symbol", **params}])


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


@op("set_comment", lock="write")
def _bind_set_comment(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "set_comment", **params}])


@op("delete_comment", lock="write")
def _bind_delete_comment(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "delete_comment", **params}])


@op("set_prototype", lock="write")
def _bind_set_prototype(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "set_prototype", **params}])


@op("local_rename", lock="write")
def _bind_local_rename(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "local_rename", **params}])


@op("local_retype", lock="write")
def _bind_local_retype(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "local_retype", **params}])


@op("struct_field_set", lock="write")
def _bind_struct_field_set(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "struct_field_set", **params}])


@op("struct_field_rename", lock="write")
def _bind_struct_field_rename(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "struct_field_rename", **params}])


@op("struct_field_delete", lock="write")
def _bind_struct_field_delete(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "struct_field_delete", **params}])


@op("types_declare", lock="write")
def _bind_types_declare(bridge, params, target):
    return bridge._mutation(target, _validate_bool(params.get("preview"), label="preview", default=False), [{"op": "types_declare", **params}])


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


def _resolve_bndb_sidecar(resolved: Path, prefer_bndb: bool) -> Path | None:
    """Adjacent ``<path>.bndb`` to load instead of ``resolved``, or None.

    Shared by the runtime ``bn load`` path (`_load_binary`) and the headless
    preload path (`_preload_binary`) so both honor a saved database sitting next
    to the binary -- otherwise `bn-agent foo` silently re-analyzes from scratch
    and drops the saved work that `bn load foo` would have picked up (#178). A
    request that already points at a ``.bndb`` (or prefer_bndb=False) resolves to
    None: load the path as given.
    """
    if prefer_bndb and resolved.suffix != ".bndb":
        sibling = Path(str(resolved) + ".bndb")
        if sibling.exists():
            return sibling
    return None


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
    sibling = _resolve_bndb_sidecar(resolved, prefer_bndb)
    if sibling is not None:
        bn.log_info(f"loaded {sibling} instead of {resolved} (use --no-bndb to skip)")
        load_path = sibling
    else:
        load_path = resolved
    bv = binaryninja.load(str(load_path), update_analysis=False)
    if bv is None:
        bn.log_warn(f"Failed to open binary: {load_path}")
        return None
    quick_effective = quick and load_path.suffix != ".bndb"
    if quick_effective:
        _quick_loaded_views.add(bv)
    else:
        bv.update_analysis_and_wait()
        _quick_loaded_views.discard(bv)
    with _headless_views_lock:
        _headless_views.append(bv)
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
