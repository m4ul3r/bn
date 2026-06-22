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

    assert result == {"saved": True, "path": str(out.resolve())}
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
    assert set(sections_page) == envelope_keys
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

    assert result == {"saved": True, "path": str(out.resolve())}
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

    assert result == {"saved": True, "path": orig + ".bndb"}
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
