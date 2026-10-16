from __future__ import annotations

import pytest

from test_taint_engine import (
    FVar, FExpr, _BufProgram, _addr_of, _cstr, _bw,
    _B_STRCPY, _B_READ, _B_MEMSET, _B_NAMES, te,
)


@pytest.mark.parametrize("writer", [_B_MEMSET, _B_STRCPY])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_loop_exit_coverage_950(reverse, initialized, writer):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    if initialized:
        pg.call(_B_STRCPY, [_addr_of(buf), _cstr(0x700)])
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
    assert result["diagnostics"]["safe_to_report_complete_slice"] is initialized
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
    assert result["diagnostics"]["safe_to_report_complete_slice"] is not escape
    assert any(l.get("callee") == "compose" for l in leaves) is escape


@pytest.mark.parametrize("length", [0, 1, 63, 64])
def test_constant_fill_extent_931(length):
    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_MEMSET, [_addr_of(buf), FExpr("MLIL_CONST", "0", constant=0),
                       FExpr("MLIL_CONST", str(length), constant=length)])
    pg.sink(buf)
    result, reasons, leaves = _bw(pg.func())
    assert any("written by memset" in r for r in reasons)
    assert result["diagnostics"]["safe_to_report_complete_slice"] is (length == 64)
    assert bool(leaves) is (length < 64)


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
    assert result["diagnostics"]["safe_to_report_complete_slice"] is not escape
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
    assert result["diagnostics"]["safe_to_report_complete_slice"]
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


@pytest.mark.parametrize("name", ["wmemcpy", "wmemmove", "__wmemcpy_chk", "__wmemmove_chk"])
@pytest.mark.parametrize("element_size", [2, 4])
@pytest.mark.parametrize("written_bytes", [0, 32, 64])
def test_wide_fill_count_uses_callee_element_width(monkeypatch, name, element_size, written_bytes):
    from types import SimpleNamespace

    buf = FVar("buf", typ="char[64]")
    pg = _BufProgram()
    pg.call(_B_READ, [_addr_of(buf), _cstr(0x700),
                     FExpr("MLIL_CONST", str(written_bytes // element_size),
                           constant=written_bytes // element_size)])
    pg.sink(buf)
    original = te.function_at
    callee = SimpleNamespace(name=name, parameter_vars=[
        SimpleNamespace(type=SimpleNamespace(target=SimpleNamespace(width=element_size)))])
    monkeypatch.setattr(te, "function_at",
                        lambda bv, addr: callee if addr == _B_READ else original(bv, addr))
    result, reasons, leaves = _bw(pg.func(), extra_names={_B_READ: name})
    assert result["diagnostics"]["safe_to_report_complete_slice"] is (written_bytes == 64)
    assert bool(leaves) is (written_bytes < 64)
