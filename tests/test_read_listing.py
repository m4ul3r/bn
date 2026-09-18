"""#814: `function list` / `function search` must build rows for the RETURNED
PAGE, not for the whole filtered population, without weakening the honest
`total`/`has_more` contract.

Two halves, both required by the issue:

* the observable contract (page contents, `total`, `has_more`, `--sort` /
  `--reverse` order) is unchanged -- pinned against an explicit expected order;
* the per-row work is O(page), not O(population) -- measured on `function
  list` by counting reads of a row-ONLY function field (`raw_name`) and on
  `function search` by counting the per-row `_function_size` projection (whose
  matcher legitimately reads every function's name forms). Both counters are
  population-wide before the fix (rows were materialized first) and page-bounded
  after.
"""
from __future__ import annotations

import pytest

from _bridge_fakes import *  # noqa: F401,F403


class _CountingFunction(_FakeFunction):
    """A Function that counts reads of `raw_name`.

    On the `function list` path `raw_name` is read exactly once per materialized
    row and never by the filtering or ordering passes, so the count IS the
    number of rows that read built. (On `function search` it is also one of the
    matcher's name forms (#196), which is why that path measures the per-row
    size projection instead.) The default `_FakeFunction` can't measure this
    because it exposes `raw_name` as a plain attribute.
    """

    def __init__(self, start: int, name: str, **kwargs):
        self.row_reads = 0
        super().__init__(start, name, **kwargs)

    @property
    def raw_name(self):
        self.row_reads += 1
        return self._raw_name

    @raw_name.setter
    def raw_name(self, value):
        self._raw_name = value


def _population(count: int, *, base: int = 0x400000, step: int = 0x10):
    """*count* distinct functions in ascending address order, each with a real
    `total_bytes` so `--sort size` has a size to order on."""
    return [
        _CountingFunction(base + i * step, f"fn_{i:05d}", total_bytes=16 + (i % 7) * 8)
        for i in range(count)
    ]


def _rows_built(functions) -> int:
    return sum(fn.row_reads for fn in functions)


def _view(monkeypatch, instance, functions):
    bv = _FakeBV(functions=functions)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)
    return bv


def _count_size_calls(monkeypatch, bridge):
    """Wrap `il_format._function_size` (the per-row size/sort projection) with a
    call counter, without changing its result."""
    real = bridge.il_format._function_size
    calls: list = []

    def counting(func):
        calls.append(func)
        return real(func)

    monkeypatch.setattr(bridge.il_format, "_function_size", counting)
    return calls


# --- the page is what gets built -------------------------------------------------

def test_list_functions_builds_rows_for_the_page_only(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(4000)
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=0, limit=20)

    # Observable contract first: a 20-row page out of 4000, exactly as before.
    assert result["total"] == 4000
    assert result["returned"] == 20
    assert result["has_more"] is True
    assert result["items"][0]["address"] == hex(0x400000)
    # ...and the work that produced it: 4000 rows was the whole point of #814.
    assert _rows_built(functions) <= 21, (
        f"{_rows_built(functions)} row(s) materialized for a 20-row page of 4000"
    )


def test_list_functions_unbounded_limit_still_builds_every_row(monkeypatch):
    # `--limit` omitted means "give me everything": the page IS the population,
    # so building every row is correct, not a regression.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(300)
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=0, limit=None)

    assert result["returned"] == 300 and result["total"] == 300
    assert result["has_more"] is False
    assert _rows_built(functions) == 300


def test_list_functions_size_sort_bounds_rows_not_the_size_read(monkeypatch):
    # `--sort size` genuinely needs one size read per filtered function for the
    # full order (that is the sort key, unchanged by #814) -- but the rows built
    # are still bounded by the page.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(2000)
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=0, limit=10, sort="size")

    assert result["total"] == 2000 and result["returned"] == 10
    assert result["items"] == sorted(result["items"], key=lambda it: it["size"])
    assert _rows_built(functions) <= 11, (
        f"{_rows_built(functions)} row(s) materialized for a 10-row size-sorted page"
    )


def test_search_functions_builds_rows_for_the_page_only(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(4000)
    _view(monkeypatch, instance, functions)
    size_calls = _count_size_calls(monkeypatch, bridge)

    # Every function matches, so the population of interest is all 4000. (The
    # matcher legitimately reads the name forms of every function -- `raw_name`
    # is one of its match keys (#196) -- but the ROW projection behind each match
    # is deferred.)
    result = instance._search_functions(None, "fn_", offset=0, limit=20)

    assert result["total"] == 4000
    assert result["returned"] == 20
    assert result["has_more"] is True
    # The per-row size projection (a `total_bytes`/basic-block read) is bounded
    # by the page; before the fix every one of the 4000 matches paid for it.
    assert len(size_calls) <= 21, (
        f"_function_size ran {len(size_calls)}x for a 20-row page of 4000 matches"
    )


def test_search_functions_min_size_filter_still_sees_the_whole_match_set(monkeypatch):
    # `--min-size` is a filter on a per-row field, so it must be evaluated for
    # every match (as before) -- the count stays exact and the page stays correct.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _CountingFunction(0x400000 + i * 0x10, f"fn_{i:05d}", total_bytes=8 if i % 2 else 64)
        for i in range(1000)
    ]
    _view(monkeypatch, instance, functions)

    result = instance._search_functions(None, "fn_", offset=0, limit=5, min_size=32)

    assert result["total"] == 500  # the odd-sized half is filtered out
    assert result["returned"] == 5 and result["has_more"] is True
    assert all(item["size"] >= 32 for item in result["items"])


# --- the contract: page contents, total, has_more, and --sort/--reverse order ---

# (start, name, total_bytes) with deliberate ties in BOTH size and name, so a
# population-level order that dropped the stable (start, name) tie-break would
# show up as a different page.
_TIE_POPULATION = [
    (0x1000, "alpha", 40),
    (0x2000, "beta", 10),
    (0x3000, "gamma", 40),
    (0x4000, "delta", 10),
    (0x5000, "zz_dup", 10),
    (0x6000, "zz_dup", 40),
]

# The full expected order per (sort, reverse) -- the documented `--sort` keys,
# stable over the (start, name) order the filtered population arrives in.
_EXPECTED_ORDER = {
    ("address", False): [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000],
    ("address", True): [0x6000, 0x5000, 0x4000, 0x3000, 0x2000, 0x1000],
    ("name", False): [0x1000, 0x2000, 0x4000, 0x3000, 0x5000, 0x6000],
    ("name", True): [0x5000, 0x6000, 0x3000, 0x4000, 0x2000, 0x1000],
    ("size", False): [0x2000, 0x4000, 0x5000, 0x1000, 0x3000, 0x6000],
    ("size", True): [0x1000, 0x3000, 0x6000, 0x2000, 0x4000, 0x5000],
}

_NAMES = {start: name for start, name, _ in _TIE_POPULATION}


@pytest.mark.parametrize("sort,reverse", sorted(_EXPECTED_ORDER))
@pytest.mark.parametrize("offset,limit", [(0, 2), (1, 3), (5, 1), (6, 2), (0, None)])
def test_list_functions_sort_order_and_paging_unchanged(monkeypatch, sort, reverse, offset, limit):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [_FakeFunction(start, name, total_bytes=size) for start, name, size in _TIE_POPULATION]
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=offset, limit=limit, sort=sort, reverse=reverse)

    whole = _EXPECTED_ORDER[(sort, reverse)]
    expected = whole[offset:] if limit is None else whole[offset:offset + limit]
    assert [item["address"] for item in result["items"]] == [hex(a) for a in expected]
    assert [item["name"] for item in result["items"]] == [_NAMES[a] for a in expected]
    assert result["total"] == len(_TIE_POPULATION)
    assert result["offset"] == offset and result["limit"] == limit
    assert result["returned"] == len(expected)
    assert result["has_more"] is (offset + len(expected) < len(_TIE_POPULATION))


def test_list_functions_offset_past_the_end_is_an_empty_page(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(6)
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=10, limit=5)

    assert result["items"] == []
    assert result["total"] == 6 and result["returned"] == 0 and result["has_more"] is False


def test_list_functions_count_only_is_unaffected_by_paging(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = _population(500)
    _view(monkeypatch, instance, functions)

    counted = instance._list_functions(None, count_only=True)

    assert counted == {"kind": "functions", "count": 500, "total": 500,
                       "analysis_state": "full", "partial": False}
    # A count builds no rows at all.
    assert _rows_built(functions) == 0


def test_list_functions_named_filter_pages_without_building_the_rest(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _CountingFunction(0x400000 + i * 0x10, f"handler_{i}" if i % 2 else f"sub_{i:x}")
        for i in range(1000)
    ]
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None, offset=0, limit=10, named=True)

    assert result["total"] == 500 and result["returned"] == 10 and result["has_more"] is True
    assert all(item["auto_named"] is False for item in result["items"])
    assert _rows_built(functions) <= 11


# --- #792: a callsite whose HLIL statement cannot be localized ---------------
#
# `hlil_statement: null` + a reason code is the honest answer for one row, but it
# left the row with no readable context on the JSON surface: `bn decompile
# <caller>` plainly renders the call, yet nothing in the row said so. The row now
# carries a bounded decompiled excerpt around the callsite.

def test_callsite_null_hlil_statement_carries_decompile_excerpt_792(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    call_addr = 0x500010
    local_addr = 0x500014
    callee = _FakeFunction(0x5a10, "send_status")
    caller = _FakeFunction(0x500000, "handle_reply")
    caller.basic_blocks = [_FakeBasicBlock(call_addr, local_addr + 4)]
    caller.arch = _FakeArch(lengths={call_addr: 4, local_addr: 4})
    block = _FakeHLILInstruction("{...}", class_name="HighLevelILBlock", address=call_addr,
                                 expr_index=9, instr_index=9)
    # A statement whose rendered text is a whole-function-sized blob: the
    # localization layer refuses it as non-local (#557) -- the shape this issue's
    # repro reports on a real AArch64 handler.
    blob = _FakeHLILInstruction("send_status(\n" + "x" * 300 + "\n)",
                                class_name="HighLevelILCall", parent=block,
                                address=call_addr, expr_index=11, instr_index=11)
    local_call = _FakeHLILInstruction("send_status(fd, &buf)", class_name="HighLevelILCall",
                                      parent=block, address=local_addr,
                                      expr_index=21, instr_index=21)
    caller.low_level_il = [[
        _FakeLLILInstruction(call_addr, _FakeConstPtr(0x5a10), hlils=[blob]),
        _FakeLLILInstruction(local_addr, _FakeConstPtr(0x5a10), hlils=[local_call]),
    ]]
    bv = _FakeBV(
        functions=[callee, caller],
        disassembly={call_addr: "bl 0x5a10", local_addr: "bl 0x5a10"},
        instruction_lengths={call_addr: 4, local_addr: 4},
    )
    _install_fake_pseudo_c(monkeypatch, bridge, caller, [[
        (0x500000, "int32_t handle_reply(int32_t fd)"),
        (0x500004, "{"),
        (0x500008, "    uint8_t buf[32];"),
        (0x50000c, "    prepare(&buf);"),
        (call_addr, "    send_status(fd, &buf);"),
        (local_addr, "    send_ack(fd);"),
        (0x500018, "    return 0;"),
        (0x50001c, "}"),
    ]])

    rows = bridge.read_listing._callsites_within_function(None, bv, callee, caller, context=1)
    by_addr = {row["call_addr"]: row for row in rows}

    failed = by_addr[hex(call_addr)]
    assert failed["hlil_statement"] is None
    assert failed["hlil_statement_reason"] == "statement_not_local"
    excerpt = failed["decompile_excerpt"]
    assert excerpt["anchor_address"] == hex(call_addr)
    assert excerpt["window"] == 3
    assert any(line.startswith(hex(call_addr)) and "send_status(fd, &buf);" in line
               for line in excerpt["lines"])
    # The excerpt is a WINDOW around the callsite, not the whole function.
    assert not any("int32_t handle_reply" in line for line in excerpt["lines"])

    # Negative control: a localized statement needs no excerpt.
    resolved = by_addr[hex(local_addr)]
    assert resolved["hlil_statement"] == "send_status(fd, &buf)"
    assert "decompile_excerpt" not in resolved
