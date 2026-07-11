from __future__ import annotations

import importlib
import importlib.util
import io
import json
import socket
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

from _bridge_fakes import *  # noqa: F401,F403


def test_resolve_rename_target_rejects_ambiguous_function_identifier(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV(
        functions=[
            _FakeFunction(0x401000, "duplicate_name"),
            _FakeFunction(0x402000, "duplicate_name"),
        ]
    )

    with pytest.raises(bridge.OperationFailure, match="Ambiguous function identifier"):
        instance._resolve_rename_target(bv, "duplicate_name", "function")


def test_find_function_auto_resolves_impl_over_import_stub(monkeypatch):
    """A name shared by a PLT/import stub and the real implementation resolves
    to the IMPLEMENTATION instead of erroring -- the stub is distinguishable by
    symbol.type, so the common collision Just Works with no new CLI surface
    (#122)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    stub = _FakeFunction(0x401030, "send")        # PLT/import trampoline
    stub.symbol = _FakeSymbol("ImportedFunctionSymbol")
    impl = _FakeFunction(0x401500, "send")        # real body
    impl.symbol = _FakeSymbol("FunctionSymbol")
    bv = _FakeBV(functions=[stub, impl])

    fn = instance._find_function(bv, "send")
    assert int(fn.start) == 0x401500

    # the rename resolver shares the same chokepoint -> same auto-resolution
    target = instance._resolve_rename_target(bv, "send", "function")
    assert target["address"] == 0x401500


def test_refresh_updates_analysis_and_returns_target_info(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()

    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_target_info", lambda selector: {"selector": "SnailMail_unwrapped.exe.bndb"})

    result = instance._refresh("active")

    assert result["refreshed"] is True
    assert result["target"]["selector"] == "SnailMail_unwrapped.exe.bndb"
    assert "refresh" in bv.events


def test_refresh_does_not_hold_target_lock_during_analysis(monkeypatch):
    """#321: refresh must run the (long) analysis holding only the write gate, NOT
    the exclusive target lock, so concurrent reads stay responsive and an agent can
    poll progress. Verify a reader can take the read lock while
    update_analysis_and_wait() runs -- if refresh held the write lock this would
    block."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    reader_got_lock = {"v": False}

    class _AnalysisBV(_FakeMutationBV):
        def update_analysis_and_wait(self):
            super().update_analysis_and_wait()

            def reader():
                with instance._target_lock.read():
                    reader_got_lock["v"] = True

            t = threading.Thread(target=reader)
            t.start()
            t.join(timeout=2.0)  # bounded so a regression fails instead of hanging

    bv = _AnalysisBV()
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_target_info", lambda selector: {})

    instance._refresh("active")

    assert reader_got_lock["v"] is True  # read lock was grantable mid-analysis


def test_refresh_tail_read_runs_under_read_lock(monkeypatch):
    """#321 audit P1: the response-building read at the END of _refresh (after the
    analysis, after the write gate is released) must run under the read lock, so a
    queued writer can't mutate the same view concurrently with the read."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    result_holder = {}

    def fake_target_info(selector):
        # A writer must be excluded while this tail read runs.
        acquired = {"v": None}

        def writer():
            with instance._target_lock.write():
                acquired["v"] = True

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        t.join(timeout=1.0)
        result_holder["writer_blocked"] = acquired["v"] is None
        return {}

    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_target_info", fake_target_info)

    instance._refresh("active")

    assert result_holder["writer_blocked"] is True  # write lock unavailable during tail read


def test_refresh_keeps_unanalyzed_flag_when_analysis_yields_no_functions(monkeypatch):
    """#321 audit P3: refreshing a #458 raw-restore-failed .bndb (no product view,
    still 0 functions after analysis) must stay analysis_state=unanalyzed rather than
    being mislabeled 'full' just because refresh ran."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()  # exposes no functions -> _view_function_count == 0
    monkeypatch.setattr(instance, "_resolve_view", lambda selector: bv)
    monkeypatch.setattr(instance, "_target_info", lambda selector: {})
    bridge._unanalyzed_views.add(bv)

    instance._refresh("active")

    assert bv in bridge._unanalyzed_views  # 0 functions -> stays flagged unanalyzed
    bridge._unanalyzed_views.discard(bv)


def test_refresh_resolves_target_under_write_gate(monkeypatch):
    """#522 (TOCTOU): refresh is @op lock="none", so dispatch takes no lock. It must
    acquire the write gate BEFORE resolving the view -- otherwise a concurrent
    close_binary/save_database (lock="write") could invalidate the view between the
    resolve and the gate acquisition, and update_analysis_and_wait() would run on a
    dead view. Assert the gate is already held at the moment _resolve_view runs."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeMutationBV()
    gate_held_at_resolve = {"v": None}

    def recording_resolve(selector):
        # threading.Lock.locked() is True while the `with self._write_gate:` block holds it.
        gate_held_at_resolve["v"] = instance._write_gate.locked()
        return bv

    monkeypatch.setattr(instance, "_resolve_view", recording_resolve)
    monkeypatch.setattr(instance, "_target_info", lambda selector: {})

    instance._refresh("active")

    assert gate_held_at_resolve["v"] is True  # gate acquired before the target was resolved


def test_target_info_surfaces_analysis_progress(monkeypatch):
    """#321: target info exposes pollable analysis phase/counts so a large-target
    analysis can be watched instead of guessing whether the bridge is wedged."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.analysis_progress = types.SimpleNamespace(
        state=types.SimpleNamespace(name="AnalyzeState"), count=1112, total=1939
    )
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    info = instance._target_info("active")

    assert info["analysis_progress"] == {"state": "AnalyzeState", "count": 1112, "total": 1939}


def test_target_info_surfaces_image_base(monkeypatch):
    """#564: target info exposes image_base from bv.start so dynamic tools can
    rebase a BN address to runtime instead of guessing the preferred base."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    bv.start = 0x400000
    bv.entry_point = 0x40B180
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])

    info = instance._target_info("active")

    assert info["image_base"] == "0x400000"
    assert info["entry_point"] == "0x40b180"


def test_validate_count_enforces_minimum(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    # count flags require >= 1
    with pytest.raises(bridge.OperationFailure) as e:
        bridge._validate_count(0, label="limit", minimum=1)
    assert e.value.status == "invalid_request"
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(-3, label="limit", minimum=1)
    # index flags allow 0 but reject negatives
    assert bridge._validate_count(0, label="offset", minimum=0) == 0
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(-1, label="offset", minimum=0)
    # None handling and non-integer coercion
    assert bridge._validate_count(None, label="limit", minimum=1, allow_none=True) is None
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count(None, label="limit", minimum=1)  # allow_none=False
    with pytest.raises(bridge.OperationFailure):
        bridge._validate_count("abc", label="limit", minimum=1)


def test_bridge_handler_swallows_broken_pipe(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    warnings = []

    class _BrokenWriter:
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.wfile = _BrokenWriter()
    monkeypatch.setattr(bridge.bn, "log_warn", lambda message: warnings.append(message))

    handler._write_response(b"{}", op="xrefs", request_id="req-123")

    assert warnings == [
        "BN Agent Bridge client disconnected before response could be delivered (op=xrefs, id=req-123)"
    ]


def test_bridge_handler_reraises_unrelated_write_errors(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    class _FailingWriter:
        def write(self, data):
            raise OSError(5, "Input/output error")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.wfile = _FailingWriter()

    with pytest.raises(OSError, match="Input/output error"):
        handler._write_response(b"{}", op="xrefs")


def test_cancel_request_marks_only_active_requests(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    inactive = instance._cancel_request("missing")
    assert inactive == {"kind": "cancel_request", "request_id": "missing", "cancelled": False}
    assert instance._is_request_cancelled("missing") is False

    instance._begin_request("req-1")
    active = instance._cancel_request("req-1")
    assert active == {"kind": "cancel_request", "request_id": "req-1", "cancelled": True}
    assert instance._is_request_cancelled("req-1") is True

    instance._end_request("req-1")
    assert instance._is_request_cancelled("req-1") is False


def test_py_exec_non_serializable_result_falls_back_to_repr(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._py_exec("active", "result = object()")

    assert isinstance(result["result"], str)
    assert result["warnings"]


def test_read_write_lock_blocks_reader_until_writer_releases(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    writer_ready = threading.Event()
    writer_release = threading.Event()
    reader_entered = threading.Event()

    def writer():
        with lock.write():
            writer_ready.set()
            writer_release.wait(1)

    def reader():
        writer_ready.wait(1)
        with lock.read():
            reader_entered.set()

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()

    assert writer_ready.wait(1)
    time.sleep(0.05)
    assert not reader_entered.is_set()

    writer_release.set()
    reader_thread.join(1)
    writer_thread.join(1)

    assert reader_entered.is_set()


def test_read_write_lock_allows_parallel_readers(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    entered: list[str] = []
    both_entered = threading.Event()
    release = threading.Event()

    def reader(name: str):
        with lock.read():
            entered.append(name)
            if len(entered) == 2:
                both_entered.set()
            release.wait(1)

    first = threading.Thread(target=reader, args=("first",))
    second = threading.Thread(target=reader, args=("second",))
    first.start()
    second.start()

    assert both_entered.wait(1)

    release.set()
    first.join(1)
    second.join(1)

    assert sorted(entered) == ["first", "second"]


def test_read_write_lock_waiting_writer_blocks_new_readers(monkeypatch):
    """A reader arriving while a writer is queued must not jump the queue,
    otherwise a steady reader stream starves the writer forever."""
    bridge = _load_bridge(monkeypatch)
    lock = bridge._ReadWriteLock()
    order: list[str] = []
    first_reader_in = threading.Event()
    first_reader_release = threading.Event()
    writer_done = threading.Event()
    second_reader_done = threading.Event()

    def first_reader():
        with lock.read():
            first_reader_in.set()
            first_reader_release.wait(2)

    def writer():
        with lock.write():
            order.append("writer")
        writer_done.set()

    def second_reader():
        with lock.read():
            order.append("reader")
        second_reader_done.set()

    t_first = threading.Thread(target=first_reader)
    t_first.start()
    assert first_reader_in.wait(1)

    t_writer = threading.Thread(target=writer)
    t_writer.start()
    # Wait until the writer is actually queued behind the active reader.
    deadline = time.monotonic() + 2
    while lock._writers_waiting == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert lock._writers_waiting == 1

    t_second = threading.Thread(target=second_reader)
    t_second.start()
    # The second reader must not enter while the writer is waiting.
    time.sleep(0.05)
    assert order == []

    first_reader_release.set()
    assert writer_done.wait(2)
    assert second_reader_done.wait(2)
    t_first.join(1)
    t_writer.join(1)
    t_second.join(1)

    assert order == ["writer", "reader"]


def test_collect_open_views_uses_tabs_api(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    class _View:
        def __init__(self, data):
            self._data = data

        def getData(self):
            return self._data

    class _Frame:
        def __init__(self, data):
            self._data = data

        def getCurrentBinaryView(self):
            return self._data

        def getCurrentView(self):
            return _View(self._data)

    view_a = object()
    view_b = object()
    view_c = object()

    class _Context:
        def getCurrentViewFrame(self):
            return _Frame(view_c)

        def getTabs(self):
            return ["tab-a", "tab-b", "tab-c"]

        def getViewFrameForTab(self, tab):
            mapping = {
                "tab-a": _Frame(view_a),
                "tab-b": _Frame(view_b),
                "tab-c": _Frame(view_c),
            }
            return mapping[tab]

        def getViewForTab(self, tab):
            mapping = {
                "tab-a": _View(view_a),
                "tab-b": _View(view_b),
                "tab-c": _View(view_c),
            }
            return mapping[tab]

    fake_ui = types.SimpleNamespace(
        UIContext=types.SimpleNamespace(
            allContexts=lambda: [_Context()],
            activeContext=lambda: None,
        )
    )
    monkeypatch.setattr(bridge, "ui", fake_ui)

    views = bridge._collect_open_views()

    assert len(views) == 3
    assert set(id(view) for view in views) == {id(view_a), id(view_b), id(view_c)}


def test_load_binary_prefers_sibling_bndb(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(bndb)]
    assert result["path"] == str(bndb)
    assert result["requested_path"] == str(raw)
    assert result["notes"]
    assert "foo.so.bndb" in result["notes"][0]
    assert "--no-bndb" in result["notes"][0]
    bridge._headless_views.clear()


def test_load_binary_no_bndb_opt_out(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    assert result["path"] == str(raw)
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_load_binary_no_sibling(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(raw)]
    assert result["path"] == str(raw)
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_load_binary_idempotent_returns_existing_view(monkeypatch, tmp_path):
    """Re-loading an already-open path must return the existing target, not open a
    second BinaryView that makes the basename/filename selectors ambiguous (#355)."""
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "app.bin"
    raw.write_bytes(b"")

    instance._load_binary(str(raw), prefer_bndb=False)
    assert loaded_paths == [str(raw)]
    assert len(bridge._headless_views) == 1

    # second load of the SAME path: no duplicate view, no second binaryninja.load
    result = instance._load_binary(str(raw), prefer_bndb=False)
    assert loaded_paths == [str(raw)]               # load NOT called again
    assert len(bridge._headless_views) == 1         # no duplicate view
    assert result.get("already_open") is True
    assert any("already open" in n.lower() for n in result["notes"])
    bridge._headless_views.clear()


def test_load_binary_idempotent_preserves_quick_analysis_state(monkeypatch, tmp_path):
    """Re-loading a quick-open target must not claim the existing view is fully
    analyzed just because the duplicate open was skipped (#405)."""
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "quick.bin"
    raw.write_bytes(b"")

    first = instance._load_binary(str(raw), prefer_bndb=False, quick=True)
    second = instance._load_binary(str(raw), prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    assert first["analyzed"] is False
    assert second["already_open"] is True
    assert second["analyzed"] is False
    assert second["analysis_state"] == "quick"
    assert any("already open" in n.lower() for n in second["notes"])
    bridge._headless_views.clear()
    bridge._quick_loaded_views.clear()


def test_load_bndb_recovers_analyzed_view_from_raw_default(monkeypatch, tmp_path):
    """#458: naming a .bndb whose load() defaults to the raw container view (0
    functions) must recover the analyzed view saved in the same database, not
    silently leave the agent on a no-symbol raw target."""
    bridge, instance, _ = _setup_load_test(monkeypatch)
    bndb = tmp_path / "router_image.bndb"
    bndb.write_bytes(b"")

    analyzed = _LoadBV(filename=str(bndb), view_type="ELF",
                       functions=[object(), object(), object()])
    raw = _LoadBV(filename=str(bndb), view_type="Raw", functions=[],
                  existing_views=["ELF", "Raw"], db_views={"ELF": analyzed})
    sys.modules["binaryninja"].load = lambda path, update_analysis=True: raw

    result = instance._load_binary(str(bndb))

    # The published/returned view is the analyzed one, not the raw container.
    assert bridge._headless_views == [analyzed]
    assert any("restored analyzed view 'ELF'" in n for n in result["notes"])
    assert any("3 functions" in n for n in result["notes"])
    bridge._headless_views.clear()


def test_load_bndb_raw_no_analysis_warns_hard(monkeypatch, tmp_path):
    """#458: a .bndb that restores a raw container with no saved product view must
    emit a hard restore-failure diagnostic (not the soft 'confirm this is the
    binary' note) AND report analysis_state=unanalyzed so JSON consumers aren't
    told a 0-function raw view is fully analyzed."""
    bridge, instance, _ = _setup_load_test(monkeypatch)
    bndb = tmp_path / "broken.bndb"
    bndb.write_bytes(b"")

    raw = _LoadBV(filename=str(bndb), view_type="Raw", functions=[],
                  existing_views=["Raw"], db_views={})
    sys.modules["binaryninja"].load = lambda path, update_analysis=True: raw

    result = instance._load_binary(str(bndb))

    assert bridge._headless_views == [raw]
    assert any(n.startswith("WARNING:") and "no saved analyzed view" in n
               for n in result["notes"])
    # Structured fields must match the WARNING, not claim full analysis (#458 P2).
    assert result["analyzed"] is False
    assert result["analysis_state"] == "unanalyzed"
    # Not the generic unrecognized-format note.
    assert not any("Confirm this is the binary you intended" in n for n in result["notes"])
    bridge._headless_views.clear()
    bridge._unanalyzed_views.clear()


def test_load_bndb_analyzed_view_no_spurious_warning(monkeypatch, tmp_path):
    """#458: the common case -- a .bndb whose load() already returns the analyzed
    view (functions present) -- must not emit any restore warning."""
    bridge, instance, _ = _setup_load_test(monkeypatch)
    bndb = tmp_path / "good.bndb"
    bndb.write_bytes(b"")

    analyzed = _LoadBV(filename=str(bndb), view_type="Mapped",
                       functions=[object(), object()])  # firmware Mapped-with-funcs is legit
    sys.modules["binaryninja"].load = lambda path, update_analysis=True: analyzed

    result = instance._load_binary(str(bndb))

    assert bridge._headless_views == [analyzed]
    assert result["analyzed"] is True and result["analysis_state"] == "full"
    assert not any("WARNING" in n or "restored analyzed view" in n for n in result["notes"])
    assert not any("was opened as a raw" in n for n in result["notes"])
    bridge._headless_views.clear()


def test_load_bndb_codeless_product_view_not_warned(monkeypatch, tmp_path):
    """#458 (Defect A): a legitimately-analyzed codeless/data database reloads as a
    product view (e.g. Mapped) with 0 functions. That is analyzed-but-code-free, not
    a restore failure -- it must NOT be warned about or reported unanalyzed. The
    discriminator is view type (raw container), not function count alone."""
    bridge, instance, _ = _setup_load_test(monkeypatch)
    bndb = tmp_path / "datablob.bndb"
    bndb.write_bytes(b"")

    # A saved Mapped product view with 0 functions (data-only region).
    codeless = _LoadBV(filename=str(bndb), view_type="Mapped", functions=[],
                       existing_views=["Mapped", "Raw"], db_views={})
    sys.modules["binaryninja"].load = lambda path, update_analysis=True: codeless

    result = instance._load_binary(str(bndb))

    assert bridge._headless_views == [codeless]      # accepted as-is, no swap
    assert result["analyzed"] is True                # analyzed, just code-free
    assert result["analysis_state"] == "full"
    assert not any("WARNING" in n or "restored analyzed view" in n for n in result["notes"])
    bridge._headless_views.clear()


def test_load_binary_concurrent_same_path_dedupes_inflight(monkeypatch, tmp_path):
    """Two concurrent loads for the same path should converge on one BinaryView
    even while the first full analysis has not published the view yet (#400)."""
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "race.bin"
    raw.write_bytes(b"")

    analysis_started = threading.Event()
    release_analysis = threading.Event()

    class BlockingBV(_LoadBV):
        def update_analysis_and_wait(self):
            analysis_started.set()
            assert release_analysis.wait(2)
            super().update_analysis_and_wait()

    binaryninja = sys.modules["binaryninja"]

    def fake_load(path, update_analysis=True):
        loaded_paths.append(path)
        return BlockingBV(filename=path)

    binaryninja.load = fake_load
    results: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def run(label: str):
        try:
            results[label] = instance._load_binary(str(raw), prefer_bndb=False)
        except BaseException as exc:  # noqa: BLE001 - expose thread failures
            errors[label] = exc

    leader = threading.Thread(target=run, args=("leader",))
    follower = threading.Thread(target=run, args=("follower",))
    leader.start()
    assert analysis_started.wait(2)
    follower.start()
    time.sleep(0.05)
    assert loaded_paths == [str(raw)]

    release_analysis.set()
    leader.join(2)
    follower.join(2)

    assert not leader.is_alive()
    assert not follower.is_alive()
    assert errors == {}
    assert loaded_paths == [str(raw)]
    assert len(bridge._headless_views) == 1
    assert results["leader"].get("already_open") is not True
    assert results["follower"].get("already_open") is True
    bridge._headless_views.clear()
    bridge._load_in_progress.clear()


def test_load_binary_dedupes_gui_open_view(monkeypatch, tmp_path):
    """A GUI-open view for the same path should be reused instead of adding a
    duplicate headless-loaded view (#400)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bridge._headless_views.clear()
    bridge._load_in_progress.clear()
    loaded_paths: list[str] = []
    raw = tmp_path / "gui.bin"
    raw.write_bytes(b"")
    gui_bv = _LoadBV(filename=str(raw))

    class FakeFrame:
        def getCurrentBinaryView(self):
            return gui_bv

    class FakeContext:
        def getCurrentViewFrame(self):
            return FakeFrame()

        def getTabs(self):
            return []

    fake_context = FakeContext()
    fake_ui = types.SimpleNamespace(
        UIContext=types.SimpleNamespace(
            allContexts=lambda: [fake_context],
            activeContext=lambda: fake_context,
        )
    )
    monkeypatch.setattr(bridge, "ui", fake_ui)
    binaryninja = sys.modules["binaryninja"]
    binaryninja.load = lambda path, update_analysis=True: (
        loaded_paths.append(path) or _LoadBV(filename=path)
    )

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert loaded_paths == []
    assert result["already_open"] is True
    assert result["analyzed"] is True
    assert result["analysis_state"] == "full"
    assert bridge._headless_views == []
    monkeypatch.setattr(bridge, "ui", None)


def test_load_binary_raw_mapped_view_warns(monkeypatch, tmp_path):
    """A file opened as a raw Mapped/Raw view (unrecognized format) must emit a
    warning so an agent doesn't proceed against a 0-function phantom target
    (#369 part 1)."""
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch, view_type="Mapped")
    raw = tmp_path / "garbage.elf"
    raw.write_bytes(b"not a recognized binary")

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert any(
        "not recognized" in n.lower() and "raw" in n.lower()
        for n in result["notes"]
    ), result["notes"]
    bridge._headless_views.clear()


def test_load_binary_recognized_view_does_not_warn(monkeypatch, tmp_path):
    """A normally-recognized view (ELF/PE/Mach-O) must NOT get the raw-mapped
    warning -- only the unrecognized-format fallback does (#369 part 1)."""
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch, view_type="ELF")
    raw = tmp_path / "real.elf"
    raw.write_bytes(b"\x7fELF")

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert not any(
        "not recognized" in n.lower() and "raw" in n.lower()
        for n in result["notes"]
    ), result["notes"]
    bridge._headless_views.clear()


def test_load_binary_quick_with_sibling_bndb_notes_quick_ignored(monkeypatch, tmp_path):
    # #316: `load <raw> --quick` when an adjacent .bndb is substituted opened the
    # already-analyzed database and silently dropped --quick. Say so explicitly.
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw), quick=True)

    assert loaded_paths == [str(bndb)]              # opened the .bndb
    assert result["analyzed"] is True              # --quick did NOT apply
    notes = " ".join(result["notes"])
    assert "--quick ignored" in notes
    assert "--no-bndb" in notes                    # the actionable escape hatch
    assert "foo.so.bndb" in notes
    bridge._headless_views.clear()


def test_load_binary_quick_with_no_bndb_honors_quick(monkeypatch, tmp_path):
    # The contrast: --no-bndb loads the raw bytes, so --quick is honored and there
    # is no "--quick ignored" note.
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(raw), prefer_bndb=False, quick=True)

    assert loaded_paths == [str(raw)]
    assert result["analyzed"] is False             # --quick honored
    notes = " ".join(result["notes"])
    assert "--quick ignored" not in notes
    bridge._headless_views.clear()


def test_load_binary_quick_on_named_bndb_notes_ignored(monkeypatch, tmp_path):
    # Directly naming a .bndb with --quick: distinct, clearer message (no sidecar
    # substitution happened).
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    bndb = tmp_path / "foo.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(bndb), quick=True)

    assert loaded_paths == [str(bndb)]
    assert result["analyzed"] is True
    notes = " ".join(result["notes"])
    assert "--quick ignored" in notes
    assert "analyzed database" in notes
    bridge._headless_views.clear()


def test_load_binary_restores_cache_bndb_when_no_adjacent(monkeypatch, tmp_path):
    # #318: a binary on a read-only mount has no adjacent .bndb, but `save` wrote
    # the annotations to the writable cache. A later load must restore THAT,
    # instead of silently re-analyzing blank ("total annotation loss").
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(bridge, "cache_home", lambda: cache_root)
    raw = tmp_path / "ro" / "foo.so"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")  # no adjacent foo.so.bndb
    cache_bndb = bridge._cache_bndb_path(str(raw.resolve()))
    cache_bndb.parent.mkdir(parents=True, exist_ok=True)
    cache_bndb.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(cache_bndb)]            # restored the cache copy
    notes = " ".join(result["notes"])
    assert "restored cached database" in notes
    assert "--no-bndb" in notes
    bridge._headless_views.clear()


def test_load_binary_no_bndb_skips_cache(monkeypatch, tmp_path):
    # --no-bndb must load the raw bytes even when a cache copy exists.
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(bridge, "cache_home", lambda: cache_root)
    raw = tmp_path / "ro" / "foo.so"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")
    cache_bndb = bridge._cache_bndb_path(str(raw.resolve()))
    cache_bndb.parent.mkdir(parents=True, exist_ok=True)
    cache_bndb.write_bytes(b"")

    result = instance._load_binary(str(raw), prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_load_binary_adjacent_bndb_preferred_over_cache(monkeypatch, tmp_path):
    # When BOTH an adjacent .bndb and a cache copy exist, the adjacent one wins
    # (it's next to the binary, the canonical location).
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(bridge, "cache_home", lambda: cache_root)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    adjacent = tmp_path / "foo.so.bndb"
    adjacent.write_bytes(b"")
    cache_bndb = bridge._cache_bndb_path(str(raw.resolve()))
    cache_bndb.parent.mkdir(parents=True, exist_ok=True)
    cache_bndb.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert loaded_paths == [str(adjacent)]
    assert "instead of" in " ".join(result["notes"])
    assert "restored cached" not in " ".join(result["notes"])
    bridge._headless_views.clear()


def test_cache_bndb_path_is_deterministic_and_path_keyed(monkeypatch, tmp_path):
    # The save and load sides must agree on the cache location for the same path,
    # and two binaries with the same basename must not collide.
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge, "cache_home", lambda: tmp_path)
    a1 = bridge._cache_bndb_path("/ro/dirA/foo")
    a2 = bridge._cache_bndb_path("/ro/dirA/foo")
    b = bridge._cache_bndb_path("/ro/dirB/foo")
    assert a1 == a2                       # deterministic (save==load)
    assert a1 != b                        # same basename, different path -> distinct
    assert a1.name.startswith("foo.") and a1.suffix == ".bndb"


def test_load_binary_full_runs_analysis(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert result["analyzed"] is True
    assert bridge._headless_views[-1].analysis_updated is True
    bridge._headless_views.clear()


def test_load_binary_runs_analysis_outside_write_lock(monkeypatch, tmp_path):
    # #99: load_binary must hold the exclusive lock around the BN open and the
    # publish, but NOT around the multi-minute update_analysis_and_wait -- else
    # doctor/target reads block for the whole load.
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    lock = instance._target_lock
    states: dict[str, bool] = {}
    binaryninja = sys.modules["binaryninja"]

    def fake_load(path, update_analysis=True):
        states["open_writer_held"] = lock._writer  # open is under the write lock
        bv = _LoadBV()
        original = bv.update_analysis_and_wait

        def analyze():
            states["analyze_writer_held"] = lock._writer  # analysis is unlocked
            original()

        bv.update_analysis_and_wait = analyze
        return bv

    binaryninja.load = fake_load
    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")

    result = instance._load_binary(str(raw))

    assert states["open_writer_held"] is True       # BN open held the lock
    assert states["analyze_writer_held"] is False   # analysis ran unlocked
    assert result["analyzed"] is True
    assert lock._writer is False                     # lock released at the end
    bridge._headless_views.clear()


def test_load_binary_not_in_write_locked_ops(monkeypatch):
    # The dispatcher must NOT take the exclusive lock for the whole load (#99);
    # load_binary does its own fine-grained locking instead.
    bridge = _load_bridge(monkeypatch)
    assert "load_binary" not in bridge.WRITE_LOCKED_OPS
    assert "load_binary" not in bridge.READ_LOCKED_OPS


def test_committed_mutation_marks_view_dirty_until_saved(monkeypatch):
    """A committed (non-preview) mutation that actually changed state marks the
    view dirty so `close` can warn -- BN's bv.file.modified never flips True for
    our writes. A preview or a pure no-op must NOT dirty the view, and
    clear_dirty (called on save) resets it. (L15)"""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeFileBV("/proj/svc", session_id="1")
    bridge._headless_views.clear()
    bridge._headless_views.extend([bv])
    instance.targets.refresh()  # assign the stable view_id
    tm = instance.targets

    assert tm._stable_view_id(bv) is not None
    assert tm.is_dirty(bv) is False

    # low-level mark/clear round-trips
    tm.mark_dirty(bv)
    assert tm.is_dirty(bv) is True
    tm.clear_dirty(bv)            # this is what _save_database calls
    assert tm.is_dirty(bv) is False

    me = bridge.mutation_engine
    verified = {"committed": True, "preview": False, "results": [{"status": "verified"}]}
    noop = {"committed": True, "preview": False, "results": [{"status": "noop"}]}
    previewed = {"committed": False, "preview": True, "results": [{"status": "verified"}]}

    # the facade marks dirty only on a committed write that actually changed state
    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: verified)
    instance._mutation("active", False, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is True

    tm.clear_dirty(bv)
    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: previewed)
    instance._mutation("active", True, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is False  # preview never dirties

    monkeypatch.setattr(me, "_mutation", lambda ctx, *a, **k: noop)
    instance._mutation("active", False, [{"op": "rename_symbol"}])
    assert tm.is_dirty(bv) is False  # a pure no-op changed nothing

    bridge._headless_views.clear()


def test_selector_uses_basename_when_unique(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    bv_b = _FakeFileBV("/proj/beta.bndb", session_id="22")
    _register_views(bridge, bv_a, bv_b)

    targets = bridge.TargetManager().refresh()
    selectors = {t["filename"]: t["selector"] for t in targets}

    assert selectors["/proj/alpha.bndb"] == "alpha.bndb"
    assert selectors["/proj/beta.bndb"] == "beta.bndb"
    bridge._headless_views.clear()


def test_selector_disambiguates_with_parent_dir(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    bv3 = _FakeFileBV("/work/03_layered_seal/target.bndb", session_id="3")
    _register_views(bridge, bv1, bv2, bv3)

    targets = bridge.TargetManager().refresh()
    selectors = {t["filename"]: t["selector"] for t in targets}

    assert selectors["/work/01_arithmetic_lock/target.bndb"] == "01_arithmetic_lock/target.bndb"
    assert selectors["/work/02_bytecode_vm/target.bndb"] == "02_bytecode_vm/target.bndb"
    assert selectors["/work/03_layered_seal/target.bndb"] == "03_layered_seal/target.bndb"
    bridge._headless_views.clear()


def test_selector_falls_back_to_target_id_for_identical_paths(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/dup/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/dup/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    targets = bridge.TargetManager().refresh()

    assert targets[0]["selector"] == targets[0]["target_id"]
    assert targets[1]["selector"] == targets[1]["target_id"]
    assert targets[0]["selector"] != targets[1]["selector"]
    bridge._headless_views.clear()


def test_refresh_rows_carry_analysis_state(monkeypatch):
    # target list rows must expose per-target analysis state so an agent can tell
    # a --quick view from a full one without a separate target info per target.
    bridge = _load_bridge(monkeypatch)
    bv_quick = _FakeFileBV("/proj/q.bndb", session_id="1")
    bv_full = _FakeFileBV("/proj/f.bndb", session_id="2")
    _register_views(bridge, bv_quick, bv_full)
    bridge._quick_loaded_views.add(bv_quick)

    rows = {t["filename"]: t for t in bridge.TargetManager().refresh()}
    assert rows["/proj/q.bndb"]["analysis_state"] == "quick"
    assert rows["/proj/q.bndb"]["analyzed"] is False
    assert rows["/proj/f.bndb"]["analysis_state"] == "full"
    assert rows["/proj/f.bndb"]["analyzed"] is True

    bridge._quick_loaded_views.discard(bv_quick)
    bridge._headless_views.clear()


def test_resolve_accepts_path_suffix_selector(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    resolved = manager.resolve("02_bytecode_vm/target.bndb")

    assert resolved is bv2
    bridge._headless_views.clear()


def test_resolve_strips_bndb_extension_from_selector(monkeypatch):
    # #312: a .bndb corpus loads <binary>.bndb; the obvious selector is <binary>.
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/corpus/agrep.bndb", session_id="1")
    bv2 = _FakeFileBV("/corpus/cpio.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)
    manager = bridge.TargetManager()
    assert manager.resolve("agrep") is bv1        # strips .bndb, disambiguates
    assert manager.resolve("cpio.bndb") is bv2    # exact basename still works
    bridge._headless_views.clear()


def test_resolve_strips_bndb_in_path_suffix_selector(monkeypatch):
    # The same tolerance with a disambiguating parent-dir path suffix.
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/a/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/b/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)
    manager = bridge.TargetManager()
    assert manager.resolve("b/target") is bv2     # .bndb stripped on the tail
    assert manager.resolve("b/target.bndb") is bv2
    bridge._headless_views.clear()


def test_resolve_bndb_strip_does_not_silently_pick_on_ambiguity(monkeypatch):
    # The high-risk case for the .bndb strip: a raw `foo` AND a `foo.bndb` both
    # open -> `-t foo` matches both. resolve() must raise ambiguous, never
    # silently pick one.
    bridge = _load_bridge(monkeypatch)
    raw = _FakeFileBV("/x/foo", session_id="1")
    db = _FakeFileBV("/x/foo.bndb", session_id="2")
    _register_views(bridge, raw, db)
    manager = bridge.TargetManager()
    with pytest.raises(Exception) as exc:
        manager.resolve("foo")
    assert "mbiguous" in str(exc.value)
    bridge._headless_views.clear()


def test_resolve_bndb_strip_is_exact_not_prefix(monkeypatch):
    # `-t foo` must NOT over-match `foobar.bndb` (strip is exact-equality).
    bridge = _load_bridge(monkeypatch)
    bv = _FakeFileBV("/x/foobar.bndb", session_id="1")
    other = _FakeFileBV("/x/baz.bndb", session_id="2")
    _register_views(bridge, bv, other)
    manager = bridge.TargetManager()
    with pytest.raises(Exception) as exc:
        manager.resolve("foo")
    assert "nknown target selector" in str(exc.value)  # no over-match
    bridge._headless_views.clear()


def test_resolve_matches_global_cache_bndb_stem(monkeypatch):
    # A read-only-mount target restores the GLOBAL cache DB named
    # `<stem>.<16-hex path digest>.bndb` (see _cache_bndb_path); the obvious
    # `-t <stem>` (e.g. `-t myprog`) must resolve it, not just the full name.
    bridge = _load_bridge(monkeypatch)
    cache_name = str(bridge._cache_bndb_path("/ro/usr/bin/myprog"))
    base = Path(cache_name).name  # myprog.<16 hex>.bndb
    bv = _FakeFileBV(cache_name, session_id="1")
    other = _FakeFileBV("/corpus/other.bndb", session_id="2")
    _register_views(bridge, bv, other)
    manager = bridge.TargetManager()
    assert manager.resolve("myprog") is bv                     # stem shortcut
    assert manager.resolve(base) is bv                         # exact cache basename
    assert manager.resolve(base[: -len(".bndb")]) is bv        # <stem>.<hash> core (#312)
    bridge._headless_views.clear()


def test_resolve_cache_stem_requires_16_hex_digest(monkeypatch):
    # The stem shortcut is specific to the cache scheme (a trailing `.<16 hex>`).
    # A `<name>.<not-16-hex>.bndb` must NOT resolve by `-t <name>` (no over-match).
    bridge = _load_bridge(monkeypatch)
    short = _FakeFileBV("/x/report.2024.bndb", session_id="1")            # tail not 16 chars
    nonhex = _FakeFileBV("/x/notes.deadbeefdeadbeXX.bndb", session_id="2")  # 16 chars, non-hex
    _register_views(bridge, short, nonhex)
    manager = bridge.TargetManager()
    with pytest.raises(Exception) as exc:
        manager.resolve("report")
    assert "nknown target selector" in str(exc.value)
    with pytest.raises(Exception):
        manager.resolve("notes")
    bridge._headless_views.clear()


def _close_on_watchdog(instance, *, timeout: float = 5.0, **kwargs):
    """Run _close_binary on a watchdog thread so a deadlock regression (e.g. a
    _write_registry()/resolve() call re-acquiring the non-reentrant
    _headless_views_lock) fails the test instead of hanging the suite."""
    import threading

    out: dict = {}

    def go():
        out["result"] = instance._close_binary(**kwargs)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), f"_close_binary({kwargs}) deadlocked (lock re-acquired under views lock?)"
    return out["result"]


def _hermetic_registry(instance, tmp_path):
    """Point an instance's registry/socket at a tmp dir so _write_registry (now
    called on load/close, #80) never scribbles the real ~/.cache/bn registry."""
    instance.registry_path = tmp_path / "reg.json"
    instance.socket_path = tmp_path / "reg.sock"


def test_close_binary_by_target_selector_does_not_deadlock(monkeypatch, tmp_path):
    # Regression: _close_binary used to resolve() the selector *while holding*
    # the non-reentrant _headless_views_lock, and resolve() re-acquires it ->
    # permanent deadlock. Run it on a watchdog thread so a regression fails the
    # test instead of hanging the suite.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _hermetic_registry(instance, tmp_path)
    bv_a = _ClosableBV("/proj/alpha.so", session_id="11")
    bv_b = _ClosableBV("/proj/beta.so", session_id="22")
    _register_views(bridge, bv_a, bv_b)

    _close_on_watchdog(instance, target="alpha.so")

    assert bv_a.closed and not bv_b.closed
    assert bv_a not in bridge._headless_views and bv_b in bridge._headless_views
    bridge._headless_views.clear()


def test_close_binary_all_flag_closes_everything(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _hermetic_registry(instance, tmp_path)
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    # Watchdog: the post-#80 close path writes the registry; a regression that did
    # so under _headless_views_lock would deadlock the all-branch.
    result = _close_on_watchdog(instance, all_=True)

    assert len(result["closed"]) == 2
    assert bv_a.closed and bv_b.closed
    assert bridge._headless_views == []


def test_close_binary_by_path_still_matches(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _hermetic_registry(instance, tmp_path)
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    result = _close_on_watchdog(instance, path="/proj/beta.so")

    assert [c["path"] for c in result["closed"]] == ["/proj/beta.so"]
    assert bv_b.closed and not bv_a.closed
    assert bv_b not in bridge._headless_views and bv_a in bridge._headless_views
    bridge._headless_views.clear()


def test_close_binary_rejects_path_and_all_together(monkeypatch):
    # A named path + all=true is contradictory; the all-branch used to silently
    # win and close everything. The bridge now rejects the combination so raw
    # socket clients are protected too (#85).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    bv_b = _ClosableBV("/proj/beta.so")
    _register_views(bridge, bv_a, bv_b)

    with pytest.raises(RuntimeError, match="not both"):
        instance._close_binary(path="/proj/alpha.so", all_=True)
    assert not bv_a.closed and not bv_b.closed  # nothing destroyed
    bridge._headless_views.clear()


def test_close_binary_by_target_works_when_headless_views_empty(monkeypatch):
    # A GUI-opened view resolves fine but is NOT tracked in _headless_views. The
    # old "no binaries loaded" guard ran before the target branch, so every
    # target-based close failed on a GUI bridge. Target close must succeed even
    # with an empty _headless_views (#86 Problem B).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bridge._headless_views.clear()
    gui_view = _ClosableBV("/proj/gui-opened.bndb")
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: gui_view)

    result = instance._close_binary(target="gui-opened.bndb")

    assert gui_view.closed
    assert [c["path"] for c in result["closed"]] == ["/proj/gui-opened.bndb"]


def test_collect_open_views_merges_headless_views_in_gui_mode(monkeypatch):
    # `bn load` against a GUI bridge appends to _headless_views, but the UI walk
    # only enumerates tabs/contexts -- so a headless-loaded view would be
    # invisible to target list. _collect_open_views must merge them (#86 Problem A).
    bridge = _load_bridge(monkeypatch)

    ui_view = object()
    headless_view = object()

    class _Frame:
        def getCurrentBinaryView(self):
            return ui_view

    class _Context:
        def getCurrentViewFrame(self):
            return _Frame()

        def getTabs(self):
            return []

    fake_ui = types.SimpleNamespace(
        UIContext=types.SimpleNamespace(
            allContexts=lambda: [_Context()],
            activeContext=lambda: None,
        )
    )
    monkeypatch.setattr(bridge, "ui", fake_ui)
    bridge._headless_views.clear()
    bridge._headless_views.append(headless_view)

    views = bridge._collect_open_views()

    ids = {id(v) for v in views}
    assert id(ui_view) in ids and id(headless_view) in ids  # both visible
    # No duplicate if a view is in both the UI walk and _headless_views.
    bridge._headless_views.append(ui_view)
    views2 = bridge._collect_open_views()
    assert sum(1 for v in views2 if v is ui_view) == 1
    bridge._headless_views.clear()


def test_preload_binary_prefers_sibling_bndb(monkeypatch, tmp_path):
    # Headless preload must mirror `bn load`: an adjacent <binary>.bndb carries
    # saved work (renames/comments/types) and must be loaded instead of
    # re-analyzing the raw binary from scratch (#178).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    binaryninja = sys.modules["binaryninja"]
    loaded_paths: list[str] = []
    binaryninja.load = lambda path, update_analysis=True: (
        loaded_paths.append(path) or _LoadBV()
    )

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bridge._preload_binary(str(raw), quick=False)

    assert loaded_paths == [str(bndb)]
    bridge._headless_views.clear()


def test_preload_binary_no_bndb_opt_out(monkeypatch, tmp_path):
    # `bn-agent foo --no-bndb` (prefer_bndb=False) must open the raw binary even
    # when a sidecar exists, matching `bn load --no-bndb` (#178).
    bridge = _load_bridge(monkeypatch)
    bridge._headless_views.clear()
    binaryninja = sys.modules["binaryninja"]
    loaded_paths: list[str] = []
    binaryninja.load = lambda path, update_analysis=True: (
        loaded_paths.append(path) or _LoadBV()
    )

    raw = tmp_path / "foo.so"
    raw.write_bytes(b"")
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    bridge._preload_binary(str(raw), quick=False, prefer_bndb=False)

    assert loaded_paths == [str(raw)]
    bridge._headless_views.clear()


def test_start_headless_rewrites_registry_after_preload(monkeypatch, tmp_path):
    """#524: _preload_binary appends to _headless_views but never rewrites the
    registry (unlike the runtime `bn load` path). start_headless must call
    _write_registry once AFTER the preload loop, so a `bn-agent <binary>`
    instance's on-disk registry lists its live targets instead of zero binaries."""
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path))
    bridge._headless_views.clear()

    # Neuter the socket server / GUI-only start() -- and make it release the
    # shutdown wait so start_headless returns synchronously. Crucially, start()
    # normally writes the registry itself; stubbing it means the ONLY registry
    # write under test is the post-preload one the #524 fix adds.
    def fake_start(self):
        self._shutdown_event.set()

    monkeypatch.setattr(bridge.BinaryNinjaBridge, "start", fake_start)
    monkeypatch.setattr(bridge, "_stop_bridge", lambda: None)

    # The registry's binaries list comes from targets.refresh(); model one loaded
    # target so a correctly-timed post-preload write records it.
    monkeypatch.setattr(bridge.TargetManager, "refresh",
                        lambda self: [{"filename": "/proj/foo.so"}])

    loaded: list[str] = []

    def fake_preload(path, quick, prefer_bndb=True):
        loaded.append(path)
        bridge._headless_views.append(object())
        return object()

    monkeypatch.setattr(bridge, "_preload_binary", fake_preload)

    bridge.start_headless(binaries=["/proj/foo.so"], instance_id="testinst")

    assert loaded == ["/proj/foo.so"]
    registry = tmp_path / "instances" / "testinst.json"
    assert registry.exists(), "registry not written after preload (#524)"
    payload = json.loads(registry.read_text())
    assert payload["binaries"] == ["/proj/foo.so"]
    bridge._headless_views.clear()


@pytest.mark.parametrize(
    "endpoint",
    [
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
    ],
)
def test_single_mutation_binder_ignores_injected_op(endpoint, monkeypatch):
    """#525: a single-mutation endpoint builds its manifest with the literal op
    spread LAST (`{**params, "op": endpoint}`), so a caller-supplied
    `params["op"]` can never redirect the operation (e.g. a set_comment request
    carrying `params={"op":"delete_comment"}` must still run set_comment). Only
    batch_apply legitimately trusts manifest ops."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    captured: dict = {}

    def fake_mutation(target, preview, operations):
        captured["operations"] = operations
        return {"ok": True}

    monkeypatch.setattr(instance, "_mutation", fake_mutation)

    # Inject a hostile op distinct from the endpoint's own op.
    injected = "delete_comment" if endpoint != "delete_comment" else "set_comment"
    instance._dispatch_on_main(endpoint, {"op": injected, "address": "0x1000"}, None)

    assert captured["operations"][0]["op"] == endpoint  # endpoint op wins, not the injected one


def test_dispatch_rejects_non_boolean_all(monkeypatch):
    # Raw JSON params must be real booleans: "all": "false" is truthy under
    # bool() and used to close every target. Reject it as invalid_request (#91).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv_a = _ClosableBV("/proj/alpha.so")
    _register_views(bridge, bv_a)

    with pytest.raises(bridge.OperationFailure) as exc:
        instance._dispatch_on_main("close_binary", {"all": "false"}, None)
    assert exc.value.status == "invalid_request"
    assert not bv_a.closed  # nothing closed
    bridge._headless_views.clear()


def test_dispatch_rejects_non_boolean_fn_pointer_scan(monkeypatch):
    # Raw JSON params must not coerce strings into enabling the expensive xrefs
    # function-pointer scan (#407).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    for bad in ("false", "true", 0, 1, "", "yes"):
        with pytest.raises(bridge.OperationFailure) as exc:
            instance._dispatch_on_main(
                "xrefs",
                {"identifier": "0x401000", "fn_pointer_scan": bad},
                None,
            )
        assert exc.value.status == "invalid_request"


def test_dispatch_passes_boolean_fn_pointer_scan_unchanged(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    seen: list[bool] = []

    def fake_xrefs(target, identifier, *, offset=0, limit=None, fn_pointer_scan=False):
        seen.append(fn_pointer_scan)
        return {"target": target, "identifier": identifier, "items": []}

    monkeypatch.setattr(instance, "_xrefs", fake_xrefs)

    instance._dispatch_on_main("xrefs", {"identifier": "0x401000"}, None)
    instance._dispatch_on_main(
        "xrefs",
        {"identifier": "0x401000", "fn_pointer_scan": False},
        None,
    )
    instance._dispatch_on_main(
        "xrefs",
        {"identifier": "0x401000", "fn_pointer_scan": True},
        None,
    )

    assert seen == [False, False, True]


def test_dispatch_rejects_non_boolean_go_functions_flags(monkeypatch):
    # Raw JSON params must be real booleans: "summary"/"count_only": "false" is
    # truthy under bool() and would silently return the wrong (summary/count)
    # shape. Reject string booleans as invalid_request (#413).
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()

    for flag in ("summary", "count_only"):
        for bad in ("false", "true", 0, 1, "", "yes"):
            with pytest.raises(bridge.OperationFailure) as exc:
                instance._dispatch_on_main("go_functions", {flag: bad}, None)
            assert exc.value.status == "invalid_request"


def test_dispatch_passes_boolean_go_functions_flags_unchanged(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    seen: list[tuple[bool, bool]] = []

    def fake_go_functions(target, *, offset=0, limit=None, count_only=False, summary=False):
        seen.append((count_only, summary))
        return {"target": target, "items": []}

    monkeypatch.setattr(instance, "_go_functions", fake_go_functions)

    instance._dispatch_on_main("go_functions", {}, None)
    instance._dispatch_on_main(
        "go_functions", {"count_only": True, "summary": False}, None
    )
    instance._dispatch_on_main(
        "go_functions", {"count_only": False, "summary": True}, None
    )

    assert seen == [(False, False), (True, False), (False, True)]


def test_validate_bool_accepts_real_booleans_and_default(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    assert bridge._validate_bool(None, label="quick", default=True) is True
    assert bridge._validate_bool(None, label="quick", default=False) is False
    assert bridge._validate_bool(True, label="quick", default=False) is True
    assert bridge._validate_bool(False, label="all", default=True) is False
    for bad in ("false", "true", 0, 1, "", "yes"):
        with pytest.raises(bridge.OperationFailure):
            bridge._validate_bool(bad, label="all", default=False)


def test_resolve_raises_on_ambiguous_basename(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    with pytest.raises(RuntimeError, match="Ambiguous target selector"):
        manager.resolve("target.bndb")
    bridge._headless_views.clear()


def test_resolve_unknown_selector_lists_open_targets(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv1 = _FakeFileBV("/work/01_arithmetic_lock/target.bndb", session_id="1")
    bv2 = _FakeFileBV("/work/02_bytecode_vm/target.bndb", session_id="2")
    _register_views(bridge, bv1, bv2)

    manager = bridge.TargetManager()
    with pytest.raises(RuntimeError) as exc_info:
        manager.resolve("does_not_exist")

    message = str(exc_info.value)
    assert message.startswith("Unknown target selector: does_not_exist")
    assert "Open targets:" in message
    assert "01_arithmetic_lock/target.bndb" in message
    assert "02_bytecode_vm/target.bndb" in message
    assert "view_id=" in message
    assert "target_id=" in message
    assert "view_id / target_id are stable across `bn save`" in message
    bridge._headless_views.clear()


def test_serialize_error_keeps_user_facing_messages_clean(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    runtime = RuntimeError("Function not found: foo")
    assert bridge._serialize_error(runtime) == "Function not found: foo"

    failure = bridge.OperationFailure("unsupported", "Symbol not found: bar")
    assert bridge._serialize_error(failure) == "Symbol not found: bar"

    value_error = ValueError("Unknown operation: bogus")
    assert bridge._serialize_error(value_error) == "Unknown operation: bogus"


def test_serialize_error_prefixes_unexpected_exceptions(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    assert bridge._serialize_error(KeyError("offset")) == "internal error: KeyError: 'offset'"
    assert (
        bridge._serialize_error(AttributeError("'NoneType' has no attribute 'name'"))
        == "internal error: AttributeError: 'NoneType' has no attribute 'name'"
    )


def test_load_binary_already_bndb_skips_lookup(monkeypatch, tmp_path):
    bridge, instance, loaded_paths = _setup_load_test(monkeypatch)
    bndb = tmp_path / "foo.so.bndb"
    bndb.write_bytes(b"")

    result = instance._load_binary(str(bndb))

    assert loaded_paths == [str(bndb)]
    assert result["path"] == str(bndb)
    assert result["notes"] == []
    bridge._headless_views.clear()


def test_bridge_handler_rejects_oversized_request(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    monkeypatch.setattr(bridge, "MAX_REQUEST_BYTES", 16)

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b"x" * 64)  # no newline within the cap
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert response["error"] == "request too large"


def test_bridge_handler_allows_request_exactly_at_cap_with_newline(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    line = b'{"op": "noop"      }\n'
    monkeypatch.setattr(bridge, "MAX_REQUEST_BYTES", len(line))

    dispatched = []
    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(line)
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            dispatch=lambda payload: dispatched.append(payload) or {"ok": True, "result": None, "error": None}
        )
    )
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    assert dispatched == [{"op": "noop"}]
    assert json.loads(writer.data.decode("utf-8"))["ok"] is True


def test_bridge_handler_rejects_non_dict_json(monkeypatch):
    bridge = _load_bridge(monkeypatch)

    dispatched = []
    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b"[1, 2, 3]\n")
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(dispatch=lambda payload: dispatched.append(payload))
    )
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert "JSON object" in response["error"]
    assert dispatched == []


# ---------------------------------------------------------------------------
# TargetManager: _ids_by_object pruning and id() recycling
# ---------------------------------------------------------------------------


def test_target_manager_does_not_alias_recycled_object_ids(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    _register_views(bridge, bv_a)
    manager = bridge.TargetManager()
    manager.refresh()

    # Simulate CPython id() recycling: a stale map entry whose key collides
    # with a brand-new view but whose ref points at a different object.
    bv_b = _FakeFileBV("/proj/beta.bndb", session_id="22")
    manager._ids_by_object[id(bv_b)] = (weakref.ref(bv_a), "999")
    _register_views(bridge, bv_a, bv_b)

    targets = manager.refresh()
    by_file = {t["filename"]: t["view_id"] for t in targets}

    # The new view must get a fresh id, not inherit the stale "999".
    assert by_file["/proj/beta.bndb"] != "999"
    assert by_file["/proj/alpha.bndb"] != by_file["/proj/beta.bndb"]
    bridge._headless_views.clear()


def test_target_manager_prunes_dead_id_entries_on_refresh(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    bv_a = _FakeFileBV("/proj/alpha.bndb", session_id="11")
    _register_views(bridge, bv_a)
    manager = bridge.TargetManager()

    class _Doomed:
        pass

    doomed = _Doomed()
    manager._ids_by_object[id(doomed)] = (weakref.ref(doomed), "777")
    del doomed

    manager.refresh()

    assert all(vid != "777" for _, vid in manager._ids_by_object.values())
    assert len(manager._ids_by_object) == 1
    bridge._headless_views.clear()


# ---------------------------------------------------------------------------
# Registry write atomicity
# ---------------------------------------------------------------------------


def test_write_registry_is_atomic_and_leaves_no_temp_files(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    instance.registry_path = tmp_path / "registry.json"

    instance._write_registry()

    data = json.loads(instance.registry_path.read_text(encoding="utf-8"))
    assert data["socket_path"] == str(instance.socket_path)
    assert data["plugin_name"]
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# start(): never displace a live socket
# ---------------------------------------------------------------------------


def test_start_refuses_to_displace_live_socket(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "bridge.sock"

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    instance.socket_path = sock_path
    try:
        with pytest.raises(RuntimeError, match="refusing to displace"):
            instance.start()
        # The live socket file must still be there for its owner.
        assert sock_path.exists()
    finally:
        server.close()


def test_socket_is_live_false_for_stale_socket_file(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    sock_path = tmp_path / "stale.sock"

    # Bind then close: the filesystem entry remains but nothing is listening.
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.close()
    instance.socket_path = sock_path

    assert sock_path.exists()
    assert instance._socket_is_live() is False


def test_save_database_succeeds_when_file_is_written(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SaveBV(str(tmp_path / "x.bin"), result=True, write=True)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    result = instance._save_database(None, str(out))

    assert result == {"ok": True, "saved": True, "path": str(out.resolve())}  # #364: top-level ok
    assert out.exists()
    assert bv.created_with == str(out.resolve())


def test_save_database_fails_when_create_database_returns_false(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # Unwritable-dir style failure: BN returns False and writes nothing.
    bv = _SaveBV(str(tmp_path / "x.bin"), result=False, write=False)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, str(out))
    assert not out.exists()


def test_save_database_fails_when_file_missing_despite_truthy_return(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # BN claims success but no file lands on disk -> still a hard failure.
    bv = _SaveBV(str(tmp_path / "x.bin"), result=True, write=False)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    out = tmp_path / "x.bndb"
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, str(out))


def test_save_database_errors_before_calling_bn_when_parent_dir_missing(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _SaveBV(str(tmp_path / "x.bin"))
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)

    missing = tmp_path / "nope" / "x.bndb"
    with pytest.raises(RuntimeError, match="directory does not exist"):
        instance._save_database(None, str(missing))
    # Fails fast: create_database is never attempted.
    assert bv.created_with is None


def test_list_ops_return_paged_envelope_with_true_total(monkeypatch):
    # #122: strings/imports/sections return the same {items, total, offset,
    # limit, returned, has_more} envelope as function list, so a truncating
    # limit still reports the honest total + remainder instead of a bare slice.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    fake_bn = sys.modules["binaryninja"]

    strings = [_FakeStringRef(0x1000 + i * 0x10, 8, f"tok_{i:03d}") for i in range(5)]
    secs = {f".s{i}": _FakeSection(f".s{i}", 0x2000 + i * 0x100, 0x2080 + i * 0x100)
            for i in range(5)}
    syms = []
    for i in range(5):
        s = fake_bn.Symbol(fake_bn.SymbolType.ImportedFunctionSymbol, 0x4000 + i, f"imp{i}")
        s.short_name = f"imp{i}"
        s.namespace = "lib"
        syms.append(s)
    bv = _FakeBV(strings=strings, sections=secs, symbols=syms)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # #275: the canonical envelope now carries a `kind` discriminator too.
    envelope_keys = {"kind", "items", "total", "offset", "limit", "returned", "has_more"}

    # A limit that truncates: 2 of 5 come back, but the total stays honest.
    strings_page = instance._strings(None, query=None, offset=0, limit=2)
    assert set(strings_page) == envelope_keys
    assert strings_page["kind"] == "strings"
    assert strings_page["total"] == 5
    assert strings_page["returned"] == 2
    assert len(strings_page["items"]) == 2
    assert strings_page["has_more"] is True

    imports_page = instance._imports(None, offset=0, limit=2)
    assert set(imports_page) == envelope_keys
    assert imports_page["kind"] == "imports"
    assert imports_page["total"] == 5 and imports_page["returned"] == 2
    assert imports_page["has_more"] is True

    sections_page = instance._sections(None, offset=0, limit=2)
    # sections carries the standard envelope PLUS its always-present W+X verdict
    # (#461) and, when segment perms exist, the W+X count/items (#453).
    assert envelope_keys <= set(sections_page)
    assert set(sections_page) - envelope_keys <= {
        "wx_verdict", "writable_executable_count", "writable_executable_items"}
    assert sections_page["wx_verdict"] in (
        "unknown_insufficient_metadata", "no_wx_sections_observed", "wx_sections_present")
    assert sections_page["kind"] == "sections"
    assert sections_page["total"] == 5 and sections_page["returned"] == 2
    assert sections_page["has_more"] is True

    # The last page (offset past the truncation point) reports no remainder.
    tail = instance._strings(None, query=None, offset=4, limit=2)
    assert tail["total"] == 5 and tail["returned"] == 1 and tail["has_more"] is False

    # limit=None means "no limit": every item, has_more False.
    everything = instance._strings(None, query=None, offset=0, limit=None)
    assert everything["returned"] == 5 and everything["has_more"] is False


def test_py_exec_reports_script_error_with_type_prefix(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # A NameError used to be tagged "internal error: NameError:" while a raised
    # ValueError surfaced as a bare message. Both now read "TypeName: message".
    with pytest.raises(RuntimeError, match=r"^NameError: name 'missing' is not defined$"):
        instance._py_exec("active", "missing")
    with pytest.raises(RuntimeError, match=r"^ValueError: boom$"):
        instance._py_exec("active", "raise ValueError('boom')")


def test_py_exec_systemexit_reported_not_worker_fault(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _FakeBV()
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    # SystemExit / sys.exit() are BaseException, not Exception, so they used to
    # slip the `except Exception` guard and unwind the worker thread -- the
    # client then saw a misleading "empty response / worker faulted" instead of
    # the real cause. They must be caught and reported as a clean, named error
    # that the instance survives. See issue #387.
    with pytest.raises(RuntimeError, match=r"SystemExit.*0"):
        instance._py_exec("active", "raise SystemExit(0)")
    with pytest.raises(RuntimeError, match=r"SystemExit.*3"):
        instance._py_exec("active", "import sys; sys.exit(3)")

    # The instance is still usable afterwards (the worker was protected).
    ok = instance._py_exec("active", "result = 1 + 1")
    assert ok["result"] == 2


def test_load_binary_corrupt_file_raises_clean_error(monkeypatch, tmp_path):
    bridge, instance, _ = _setup_load_test(monkeypatch)
    raw = tmp_path / "broken.bndb"
    raw.write_bytes(b"not a real database")

    def boom(path, update_analysis=True):
        raise Exception("Unable to create new BinaryView")

    sys.modules["binaryninja"].load = boom

    # A corrupt/truncated file used to escape as "internal error: Exception: ...".
    with pytest.raises(RuntimeError, match="may be corrupt"):
        instance._load_binary(str(raw))
    bridge._headless_views.clear()


def test_save_database_falls_back_to_writable_cache(monkeypatch, tmp_path):
    """A default-path save whose directory is unwritable (read-only firmware
    mount) falls back to a writable cache dir instead of losing annotations;
    an EXPLICIT --path failure stays a hard error (#214)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge, "cache_home", lambda: tmp_path / "cache")

    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_file = str(ro_dir / "firmware.bin")
    ro_bndb = ro_file + ".bndb"
    created: list[str] = []

    class _SaveBV:
        class file:
            filename = ro_file

        def create_database(self, dest):
            if str(dest) == ro_bndb:
                return False                       # simulate an unwritable default dir
            from pathlib import Path as _P
            _P(dest).parent.mkdir(parents=True, exist_ok=True)
            _P(dest).write_bytes(b"BNDB")
            created.append(str(dest))
            return True

    bv = _SaveBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda sel: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    result = instance._save_database(None)
    assert result["saved"] is True
    assert result["fallback"] is True
    assert result["requested_path"] == ro_bndb
    assert "cache" in result["path"] and result["path"].endswith(".bndb")
    assert created == [result["path"]]

    # explicit --path failure must NOT silently relocate
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database(None, path=ro_bndb)


def test_bridge_handler_serialization_failure_returns_clean_error(monkeypatch):
    """A response that fails to serialize -- the 'dictionary changed size during
    iteration' race when a response aliases a live mapping mutated by a concurrent
    read (serialization runs OUTSIDE the dispatch lock and its try/except) -- must
    return a clean {ok: false} error to the client, NOT escape handle() and
    silently kill the handler thread with no response at all (#250)."""
    bridge = _load_bridge(monkeypatch)
    errors = []
    monkeypatch.setattr(bridge.bn, "log_error", lambda message: errors.append(message))

    class _Unserializable:
        # default=str is json's fallback for non-native types; a __str__ that
        # always raises deterministically reproduces a mid-encode failure.
        def __str__(self):
            raise RuntimeError("dictionary changed size during iteration")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b'{"op": "decompile", "id": "req-9"}\n')
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            dispatch=lambda payload: {"ok": True, "result": _Unserializable(), "error": None}))
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()  # must NOT raise

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert "serializ" in response["error"].lower()
    assert errors, "the serialization failure should be logged for operators"


def test_bridge_handler_retries_transient_serialization_race(monkeypatch):
    """The dict-size race is transient: by the time a retry runs, the concurrent
    mutation has usually completed. A single re-encode preserves the REAL response
    instead of degrading every racy read to an error (#250)."""
    bridge = _load_bridge(monkeypatch)

    class _FlakyValue:
        calls = 0

        def __str__(self):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("dictionary changed size during iteration")
            return "recovered"

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b'{"op": "decompile", "id": "req-9"}\n')
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            dispatch=lambda payload: {"ok": True, "result": _FlakyValue(), "error": None}))
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is True
    assert response["result"] == "recovered"


def test_bridge_handler_serialization_fallback_survives_logging_failure(monkeypatch):
    """Even if logging the serialization failure itself raises, handle() must still
    write a clean error and never let the handler thread die silently -- the
    fallback path must not reintroduce the exact silent-death it eliminates (#250)."""
    bridge = _load_bridge(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("log sink down")

    monkeypatch.setattr(bridge.bn, "log_error", _boom)

    class _Unserializable:
        def __str__(self):
            raise RuntimeError("dictionary changed size during iteration")

    handler = bridge.BridgeHandler.__new__(bridge.BridgeHandler)
    handler.rfile = io.BytesIO(b'{"op": "decompile", "id": "req-9"}\n')
    handler.server = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            dispatch=lambda payload: {"ok": True, "result": _Unserializable(), "error": None}))
    writer = _RecordingWriter()
    handler.wfile = writer

    handler.handle()  # must NOT raise even though log_error raises

    response = json.loads(writer.data.decode("utf-8"))
    assert response["ok"] is False
    assert "serializ" in response["error"].lower()


def test_save_path_failure_restores_target_identity(monkeypatch, tmp_path):
    """#256 review: if create_database re-homes the live view and THEN the
    explicit --path save fails, the original filename must still be restored --
    otherwise a failed save silently strands the selector at a path that was
    never written."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    orig = str(tmp_path / "origbin")
    bv = _RehomingFailSaveBV(orig)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    out = tmp_path / "copy.bndb"
    with pytest.raises(RuntimeError, match="no file was written"):
        instance._save_database("origbin", str(out))
    assert bv.file.filename == orig  # identity restored despite the failure

def test_save_path_preserves_target_identity(monkeypatch, tmp_path):
    """An explicit `save --path` writes a COPY; it must NOT re-home the live target.
    BN's create_database rebinds bv.file.filename to the new .bndb -- the basis for
    selector resolution -- so the original selector (and `bn close <name>`) would
    stop resolving. The fix restores the original filename after an explicit --path
    save, so the copy is written but the live target keeps its identity (#256)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    orig = str(tmp_path / "origbin")
    bv = _RehomingSaveBV(orig)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    out = tmp_path / "copy.bndb"
    result = instance._save_database("origbin", str(out))

    assert result == {"ok": True, "saved": True, "path": str(out.resolve())}  # #364: top-level ok
    assert out.exists()
    assert bv.created_with == str(out.resolve())   # the copy WAS written
    assert bv.file.filename == orig                # ...but identity is preserved

def test_save_path_restore_failure_reports_degraded_not_clean_success(monkeypatch, tmp_path):
    """#256 review: if the post-save restore of the original filename FAILS, the
    save did land on disk but the live view is still re-homed to the copy. That
    must surface as a degraded result (rehomed=True), not a clean success that
    hides the lost identity."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    orig = str(tmp_path / "origbin")
    bv = _RestoreFailSaveBV(orig)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    out = tmp_path / "copy.bndb"
    result = instance._save_database("origbin", str(out))

    assert result["saved"] is True
    assert result.get("rehomed") is True
    assert out.exists()
    assert bv.file.filename == str(out.resolve())  # still re-homed (restore failed)


def test_save_default_preserves_target_identity(monkeypatch, tmp_path):
    """#285: a DEFAULT save writes <binary>.bndb, but BN's create_database re-homes
    the live view to it -- which rekeys the -t <basename> selector (derived live
    from bv.file.filename). A save must not change the selector a caller is using,
    so the original filename is restored afterward (same shape as #256's --path)."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    orig = str(tmp_path / "libfoo.so")
    Path(orig).write_text("elf")
    bv = _RehomingSaveBV(orig)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    result = instance._save_database(None)               # default save (no --path)

    assert result == {"ok": True, "saved": True, "path": orig + ".bndb"}  # #364: top-level ok
    assert bv.created_with == orig + ".bndb"             # the .bndb WAS written
    assert bv.file.filename == orig                       # ...selector identity preserved


def test_save_ro_fallback_preserves_target_identity(monkeypatch, tmp_path):
    """#285: the RO->cache fallback re-homes the live view to the cache copy, which
    rekeys the -t <basename> selector to <basename>.<hash>.bndb. The fallback is a
    copy (the RO original is intact), so the original filename is restored after the
    cache write -- the basename selector keeps resolving."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(bridge, "cache_home", lambda: tmp_path / "cache")
    orig = str(tmp_path / "ro" / "libfoo.so")
    Path(orig).parent.mkdir()
    Path(orig).write_text("elf")
    ro_bndb = orig + ".bndb"

    class _RoFallbackRehomeBV:
        def __init__(self):
            self.file = types.SimpleNamespace(filename=orig)
            self.created_with = None

        def create_database(self, out):
            if str(out) == ro_bndb:
                return False                              # RO default dir: write fails
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("bndb")
            self.file.filename = str(out)                # cache write re-homes the view
            self.created_with = str(out)
            return True

    bv = _RoFallbackRehomeBV()
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    result = instance._save_database(None)

    assert result["saved"] is True
    assert result["fallback"] is True
    assert "cache" in result["path"] and result["path"].endswith(".bndb")
    assert bv.file.filename == orig                       # identity preserved across the rehome


def test_save_default_restore_failure_reports_degraded(monkeypatch, tmp_path):
    """If the post-default-save restore of the original filename fails, surface a
    degraded (rehomed) result rather than a clean success that hides the rekey."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    orig = str(tmp_path / "libfoo.so")
    Path(orig).write_text("elf")
    bv = _RestoreFailSaveBV(orig)
    monkeypatch.setattr(instance.targets, "resolve", lambda target: bv)
    monkeypatch.setattr(instance.targets, "clear_dirty", lambda b: None)

    result = instance._save_database(None)

    assert result["saved"] is True
    assert result.get("rehomed") is True


def test_write_registry_records_open_binaries(monkeypatch, tmp_path):
    # #80: the registry carries the instance's open binaries (sorted, deduped,
    # blanks dropped) so listing them needs no target-list round-trip.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    instance.registry_path = tmp_path / "inst.json"
    instance.socket_path = tmp_path / "inst.sock"
    instance.instance_id = "abc123"
    monkeypatch.setattr(instance.targets, "refresh", lambda: [
        {"filename": "/fw/lib64/libfoo.so"},
        {"filename": "/fw/bin/daemon"},
        {"filename": "/fw/lib64/libfoo.so"},   # dup
        {"filename": ""},                        # blank -> dropped
    ])
    instance._write_registry()
    payload = json.loads((tmp_path / "inst.json").read_text())
    assert payload["binaries"] == ["/fw/bin/daemon", "/fw/lib64/libfoo.so"]


def test_write_registry_binaries_best_effort_on_refresh_error(monkeypatch, tmp_path):
    # A registry write must never fail because target enumeration raised.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    instance.registry_path = tmp_path / "inst.json"
    instance.socket_path = tmp_path / "inst.sock"
    instance.instance_id = "abc123"
    def boom():
        raise RuntimeError("enumeration failed")
    monkeypatch.setattr(instance.targets, "refresh", boom)
    instance._write_registry()   # must not raise
    payload = json.loads((tmp_path / "inst.json").read_text())
    assert payload["binaries"] == []   # degrades to empty, registry still written


def test_write_project_marker_gated_and_git_excludes(monkeypatch, tmp_path):
    # #80: the bridge drops a `.bn-<id>` marker in the CLI's project root, adds
    # `.bn-*` to .git/info/exclude, and is gated (GUI bridge / --no-marker write none).
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge(instance_id="zz99")

    git = tmp_path / ".git" / "info"
    git.mkdir(parents=True)  # make tmp_path look like a git work tree root
    assert inst._write_project_marker(str(tmp_path), no_marker=False) is None
    marker = tmp_path / ".bn-zz99"
    assert marker.exists()
    body = json.loads(marker.read_text())
    assert body["instance_id"] == "zz99" and "socket_path" in body
    assert ".bn-*" in (tmp_path / ".git" / "info" / "exclude").read_text()

    # opt-out writes nothing
    other = tmp_path / "other"
    other.mkdir()
    assert inst._write_project_marker(str(other), no_marker=True) is None
    assert not (other / ".bn-zz99").exists()

    # GUI bridge (instance_id None) writes nothing (keeps its legacy fixed registry)
    gui = bridge.BinaryNinjaBridge()
    assert gui._write_project_marker(str(other), no_marker=False) is None
    assert not list(other.glob(".bn-*"))


def test_write_project_marker_refresh_only_does_not_create(monkeypatch, tmp_path):
    # #391: `session restart` refreshes a marker but must NOT create a new one in
    # a restart cwd that differs from the original session-start cwd. refresh_only
    # writes ONLY when a marker already exists there.
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge(instance_id="rs42")
    git = tmp_path / ".git" / "info"
    git.mkdir(parents=True)

    # no existing marker -> refresh_only writes nothing
    assert inst._write_project_marker(str(tmp_path), no_marker=False, refresh_only=True) is None
    assert not (tmp_path / ".bn-rs42").exists()

    # create one (a prior session start), then refresh_only updates it in place
    marker = tmp_path / ".bn-rs42"
    marker.write_text(json.dumps({"instance_id": "rs42", "socket_path": "/old", "pid": 1, "created_at": "old"}))
    assert inst._write_project_marker(str(tmp_path), no_marker=False, refresh_only=True) is None
    body = json.loads(marker.read_text())
    assert body["instance_id"] == "rs42"
    assert body["created_at"] != "old"  # refreshed


def test_write_project_marker_readonly_dir_is_a_note_not_error(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge(instance_id="ro11")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        note = inst._write_project_marker(str(ro), no_marker=False)
        # best-effort: a one-line note, never a raised error (a missing/locked dir
        # must not fail the load)
        assert note is None or "marker" in note
    finally:
        ro.chmod(0o700)
