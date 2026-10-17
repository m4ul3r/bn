from __future__ import annotations

import pytest

from test_taint_engine import (
    FVar, FExpr, _BufProgram, _addr_of, _cstr, _bw, _store_proof_only,
    _B_STRCPY, _B_READ, _B_MEMSET, _B_NAMES, te,
)


@pytest.mark.parametrize("writer", [_B_MEMSET, _B_STRCPY])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_loop_exit_coverage_950(reverse, initialized, writer):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    if initialized:
        pg.whole_store(buf)
    entry = pg.mem
    header = pg.phi(entry, entry + 3)
    params = ([_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
               FExpr("MLIL_CONST", "64", constant=64)] if writer == _B_MEMSET
              else [_addr_of(buf), _cstr(0x700)])
    fill = pg.call(writer, params)
    back = pg.call(_B_STRCPY, [_addr_of(FVar("other", ident=99)), _cstr(0x700)])
    back.src_memory = header.dest_memory
    arms = [fill.dest_memory, back.dest_memory]
    pg.phi(*(reversed(arms) if reverse else arms))
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    # Initialized by a store: the best a buffer reaches is the uncertified
    # store proof (#931 round 3).
    assert _store_proof_only(result) is initialized
    assert bool(leaves) is not initialized


@pytest.mark.parametrize("escape", [False, True])
def test_loop_carried_escape_951(escape):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)     # a call handed &buf would itself start an escape (#1053 r4)
    header = pg.phi(1, 4)
    # Use an existing unknown target name, with no pointer argument.
    unknown = next(k for k, v in _B_NAMES.items() if v == "compose")
    pg.call(unknown, [])
    pg.store(_cstr(0x404000), _addr_of(buf) if escape else FExpr("MLIL_CONST", "0", constant=0))
    sink = pg.sink(buf)
    sink.src_memory = header.dest_memory
    result, reasons, leaves = _bw(pg.func())
    assert _store_proof_only(result) is not escape
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert any(l.get("callee") == "compose" for l in leaves) is escape


@pytest.mark.parametrize("writer", [_B_MEMSET, _B_READ])
@pytest.mark.parametrize("length", [0, 1, 63, 64, 65])
def test_call_fill_never_proves_initialisation_931(writer, length):
    # A requested length is not the bytes a call delivered (a short read, a
    # failing call): even a full-size memset or read at offset 0 leaves the
    # buffer unproven, disclosed with the call and the consumed region.
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    n = FExpr("MLIL_CONST", str(length), constant=length)
    pg.call(writer, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0), n]
            if writer == _B_MEMSET else [FExpr("MLIL_CONST", "0", constant=0), _addr_of(buf), n])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    name = _B_NAMES[writer]
    assert any(f"written by {name}" in r for r in reasons)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    [leaf] = leaves
    assert leaf["calls"] == [name] and leaf["region"] == {"offset": 0, "size": 64}, leaf
    assert leaf["unproven"] == [[0, 64]], leaf


@pytest.mark.parametrize("budget", [1, 5, 12])
def test_fixed_point_budget_never_certifies_partial_proof(monkeypatch, budget):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_STRCPY, [_addr_of(buf), _cstr(0x700)])
    header = pg.phi(1, 4)
    pg.call(next(k for k, v in _B_NAMES.items() if v == "compose"), [])
    pg.store(_cstr(0x404000), _addr_of(buf))
    sink = pg.sink(buf)
    sink.src_memory = header.dest_memory
    monkeypatch.setattr(te, "MEM_CHAIN_BUDGET", budget)
    result, _, _ = _bw(pg.func())
    assert result["stats"]["truncated"]
    assert not result["diagnostics"]["safe_to_report_complete_slice"]


@pytest.mark.parametrize("length", [0, 1, 63])
def test_short_source_preserves_source_without_coverage(length):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_READ, [FExpr("MLIL_CONST", "0", constant=0), _addr_of(buf),
                     FExpr("MLIL_CONST", str(length), constant=length)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert result["slices"][0]["origin"] == {"kind": "source", "callee": "read"}
    assert leaves and not result["diagnostics"]["safe_to_report_complete_slice"]


def test_loop_carried_stash_is_not_a_definite_destination():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    header = pg.phi(0, 3)
    reload = FExpr("MLIL_LOAD_SSA", "[stash]", src=_cstr(0x404000),
                   src_memory=header.dest_memory)
    reload.size = 8
    pg.call(_B_STRCPY, [reload, _cstr(0x700)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    assert leaves and not result["diagnostics"]["safe_to_report_complete_slice"]


@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_nested_loop_escape_reaches_inner_call(escape, reverse):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)
    pg.phi(*([6, 1] if reverse else [1, 6]))
    pg.phi(*([5, 2] if reverse else [2, 5]))
    pg.call(next(k for k, v in _B_NAMES.items() if v == "compose"), [])
    pg.store(_cstr(0x404008), FExpr("MLIL_CONST", "0", constant=0))
    pg.store(_cstr(0x404000), _addr_of(buf) if escape else FExpr("MLIL_CONST", "0", constant=0))
    sink = pg.sink(buf)
    sink.src_memory = 2
    result, _, leaves = _bw(pg.func())
    assert _store_proof_only(result) is not escape
    assert result["diagnostics"]["safe_to_report_complete_slice"] is False
    assert any(l.get("callee") == "compose" for l in leaves) is escape
    assert not result["stats"]["truncated"]


def test_future_stash_does_not_invent_an_earlier_buffer_writer():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.whole_store(buf)     # a call handed &buf would itself start an escape (#1053 r4)
    reload = FExpr("MLIL_LOAD_SSA", "[stash]", src=_cstr(0x404000), src_memory=pg.mem)
    pg.call(_B_STRCPY, [reload, _cstr(0x700)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert not any("written by strcpy" in reason for reason in reasons)
    assert _store_proof_only(result), result["leaves"]
    assert not leaves


def test_discovery_cut_keeps_nearby_escaped_call_frontier():
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    unknown = next(k for k, v in _B_NAMES.items() if v == "compose")
    for _ in range(300):
        pg.call(unknown, [])
    pg.call(_B_MEMSET, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
                       FExpr("MLIL_CONST", "64", constant=64)])
    pg.store(_cstr(0x404000), _addr_of(buf))
    near = pg.call(unknown, [])
    pg.sink(buf)
    result, _, leaves = _bw(pg.func())
    assert result["stats"]["truncated"]
    assert not result["diagnostics"]["safe_to_report_complete_slice"]
    assert any(l.get("callee") == "compose" and l["address"] == hex(near.address)
               for l in leaves)
