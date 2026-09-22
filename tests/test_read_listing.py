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
#
# The render is in CONTROL-FLOW order, not address order (#792 review): a real
# pseudo-C body emits switch cases out of sequence and puts the FUNCTION START on
# the closing brace, so a body whose LINES happen to be address-sorted is a straw
# fixture -- it cannot catch an anchor rule that assumes monotonic addresses. Every
# fixture below therefore places the callsite line before an epilogue and ends with
# a brace carrying the function-entry address.

def _control_flow_batches(call_addr: int) -> list[tuple[int, str]]:
    """The repro's render shape: out-of-order addresses, a callsite line, a body
    tail, an epilogue, and a closing brace that carries the FUNCTION START."""
    return [
        (0x500000, "int32_t handle_reply(int32_t fd)"),        # header @ func start
        (0x500004, "{"),
        (0x500008, "    uint8_t buf[32];"),
        (0x500020, "    if (cmd == 1) {"),                     # control flow, not address order
        (0x500050, "        send_ack(fd);"),
        (call_addr, "    send_status(fd, &buf);"),             # the callsite, exact address
        (0x5000b8, "    if (buf == 0) {"),
        (0x5000c0, "        goto out;"),
        (0x5000c8, "    }"),
        (0x5000d0, "    last_err = 0;"),
        (0x5000e4, "    __stack_chk_fail();"),                 # epilogue
        (0x5000e8, "    no return;"),
        (0x500000, "}"),                                       # closing brace @ func start
    ]


def test_callsite_null_hlil_statement_carries_decompile_excerpt_792(monkeypatch):
    """The excerpt is a field of the callsites PAGE, so it is read through the op
    that pages, not through the per-function row builder."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    call_addr = 0x5000b0
    callee = _FakeFunction(0x5a10, "send_status")
    caller = _FakeFunction(0x500000, "handle_reply")
    caller.basic_blocks = [_FakeBasicBlock(call_addr, call_addr + 4)]
    caller.arch = _FakeArch(lengths={call_addr: 4})
    block = _FakeHLILInstruction("{...}", class_name="HighLevelILBlock", address=call_addr,
                                 expr_index=9, instr_index=9)
    # A statement whose rendered text is a whole-function-sized blob: the
    # localization layer refuses it as non-local (#557) -- the shape this issue's
    # repro reports on a real AArch64 handler.
    blob = _FakeHLILInstruction("send_status(\n" + "x" * 300 + "\n)",
                                class_name="HighLevelILCall", parent=block,
                                address=call_addr, expr_index=11, instr_index=11)
    caller.low_level_il = [[
        _FakeLLILInstruction(call_addr, _FakeConstPtr(0x5a10), hlils=[blob]),
    ]]
    bv = _FakeBV(
        functions=[callee, caller],
        disassembly={call_addr: "bl 0x5a10"},
        instruction_lengths={call_addr: 4},
    )
    _install_fake_pseudo_c(monkeypatch, bridge, caller, [_control_flow_batches(call_addr)])
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    rows = _callsites_items(instance, None, "send_status",
                            within_identifiers=["handle_reply"], context=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["hlil_statement"] is None
    assert row["hlil_statement_reason"] == "statement_not_local"
    excerpt = row["decompile_excerpt"]
    # Exact-address anchoring: the callsite's own line, NOT the trailing brace that
    # also carries a `<= call_addr` address (the pre-review scan anchored there and
    # emitted the epilogue).
    assert excerpt["anchor_address"] == hex(call_addr)
    assert excerpt["window"] == 3
    assert any(line.startswith(hex(call_addr)) and "send_status(fd, &buf);" in line
               for line in excerpt["lines"])
    assert not any("__stack_chk_fail" in line for line in excerpt["lines"])
    # The function-entry address (header line AND closing brace) stays out of the
    # window; it is not the callsite.
    assert not any(line.startswith(hex(0x500000)) for line in excerpt["lines"])


def test_callsite_excerpt_proximity_fallback_never_anchors_on_the_function_entry_792(monkeypatch):
    """No line carries the callsite address exactly: the window falls back to the
    nearest guttered line, and the function-ENTRY address is excluded from that
    contest -- a real render puts it on the closing brace, whose line is last."""
    bridge = _load_bridge(monkeypatch)
    lines, addresses = _rendered_lines(bridge, monkeypatch, [
        (0x500000, "int32_t handle_reply(int32_t fd)"),
        (0x500004, "{"),
        (0x500008, "    uint8_t buf[32];"),
        (0x500020, "    if (cmd == 1) {"),
        (0x500050, "        prep(fd);"),
        (0x5000b4, "    if (buf == 0) {"),          # nearest to the callsite below
        (0x5000c0, "        return -1;"),
        (0x5000c8, "    }"),
        (0x5000d0, "    last_err = 0;"),
        (0x5000e4, "    __stack_chk_fail();"),
        (0x500000, "}"),                            # closing brace @ func start
    ])
    excerpt = bridge.read_listing._callsite_decompile_excerpt(
        (lines, addresses), 0x5000b0, 0x500000)

    assert excerpt["anchor_address"] == "0x5000b4"
    assert any(line.startswith("0x5000b4") for line in excerpt["lines"])
    assert not any(line.startswith(hex(0x500000)) for line in excerpt["lines"])


def test_callsite_excerpt_without_an_attributable_line_says_so_792(monkeypatch):
    """Nothing in the render can be attributed to the callsite: the excerpt must
    say that instead of anchoring on the function's tail."""
    bridge = _load_bridge(monkeypatch)
    lines, addresses = _rendered_lines(bridge, monkeypatch, [
        (0x500000, "int32_t handle_reply(int32_t fd)"),
        (0x500000, "}"),
    ])
    excerpt = bridge.read_listing._callsite_decompile_excerpt(
        (lines, addresses), 0x5000b0, 0x500000)

    assert "anchor_address" not in excerpt
    assert excerpt["lines"] == []
    assert excerpt["reason"] == "statement_not_located"


def test_callsite_excerpt_caps_a_long_line_792(monkeypatch):
    """A window bounds the line COUNT, not the bytes: one measured render emits a
    583-character statement line. Each line is capped (240 chars, the same ceiling
    `_hlil_text_is_local` refuses) and the capped lines are counted, so a 7-line
    window cannot become the blob `hlil_statement` deliberately withholds."""
    bridge = _load_bridge(monkeypatch)
    text = "    send_status(fd, &buf); /* " + "z" * 583 + " */"
    lines, addresses = _rendered_lines(bridge, monkeypatch, [
        (0x500000, "int32_t handle_reply(int32_t fd)"),
        (0x5000b0, text),
        (0x500000, "}"),
    ])
    excerpt = bridge.read_listing._callsite_decompile_excerpt(
        (lines, addresses), 0x5000b0, 0x500000)

    capped = [line for line in excerpt["lines"] if "send_status" in line]
    assert len(capped) == 1
    assert capped[0].startswith(hex(0x5000b0))
    assert f"... [+{len(lines[1]) - 240} chars]" in capped[0]
    assert excerpt["truncated_lines"] == 1
    assert all(len(line) <= 240 + 40 for line in excerpt["lines"])


def _null_statement_population(count: int, callee_addr: int = 0x5A10):
    """*count* callers of one callee, each with exactly one callsite whose HLIL
    statement cannot be localized -- so every row of every caller wants the
    #792 excerpt, and the number of renders is a clean measure of the work."""
    callers, refs, disasm, lengths = [], [], {}, {}
    for index in range(count):
        start = 0x500000 + index * 0x1000
        call_addr = start + 0x10
        caller = _FakeFunction(start, f"handler_{index:02d}")
        caller.basic_blocks = [_FakeBasicBlock(call_addr, call_addr + 4)]
        caller.arch = _FakeArch(lengths={call_addr: 4})
        # No HLIL for the LLIL call -> `hlil_statement` null, reason
        # `no_hlil_mapping`: the #792 shape, on every caller.
        caller.low_level_il = [[_FakeLLILInstruction(call_addr, _FakeConstPtr(callee_addr))]]
        callers.append(caller)
        refs.append(_FakeCodeRef(call_addr, caller))
        disasm[call_addr] = "bl 0x5a10"
        lengths[call_addr] = 4
    callee = _FakeFunction(callee_addr, "send_status")
    bv = _FakeBV(functions=[callee, *callers], code_refs={callee_addr: refs},
                 disassembly=disasm, instruction_lengths=lengths)
    return bv, callers


def _counting_decompile_text(monkeypatch, bridge, rendered: list[int]):
    """Replace the pseudo-C renderer with one that records WHICH function it was
    asked to render. Counting renders is the only way to see this contract: the
    returned page is byte-identical before and after the fix."""
    def _render(bv, func, addresses=False):
        rendered.append(int(func.start))
        return f"{hex(int(func.start) + 0x10)}        send_status(fd, &buf);"

    monkeypatch.setattr(bridge.read_listing.il_format, "_decompile_text", _render)


def test_callsite_excerpt_renders_only_the_returned_page_792(monkeypatch):
    """#792 review: the excerpt costs a whole-function decompilation, so it must
    be a PAGE projection (#814), not scan-time work. A callsites read scans the
    caller set until it holds `offset + limit + 1` rows; rendering while scanning
    paid one whole-function decompile for every caller the `--offset`/`--limit`
    window then discarded."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, callers = _null_statement_population(12)
    rendered: list[int] = []
    _counting_decompile_text(monkeypatch, bridge, rendered)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._callsites(None, "send_status", within_identifiers=[], offset=8, limit=4)

    # The page itself is unchanged: the whole caller set was scanned (exact
    # `total`), and the window holds the last four callers' rows.
    assert result["total"] == 12 and result["scan_truncated"] is False
    assert [row["containing_function"]["name"] for row in result["items"]] == [
        "handler_08", "handler_09", "handler_10", "handler_11"]
    assert all(row["hlil_statement"] is None for row in result["items"])
    assert all(row["decompile_excerpt"]["lines"] for row in result["items"])
    # The WORK is page-bounded: one render per caller ON THE PAGE. Before the
    # fix this was one per caller SCANNED (all 12).
    assert rendered == [caller.start for caller in callers[8:12]], rendered
    # The transient handle never reaches a consumer.
    assert all("_excerpt_fn" not in row for row in result["items"])


def test_callsite_excerpt_page_projection_holds_on_a_truncated_scan_792(monkeypatch):
    """The early-exit page (`scan_truncated`) is a second, hand-sliced return
    path: it must project the excerpt onto its own page and drop the transient
    too, or a live Function object leaks into the row."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv, callers = _null_statement_population(12)
    rendered: list[int] = []
    _counting_decompile_text(monkeypatch, bridge, rendered)
    monkeypatch.setattr(instance.ctx, "_resolve_view", lambda selector: bv)

    result = instance._callsites(None, "send_status", within_identifiers=[], offset=8, limit=2)

    assert result["scan_truncated"] is True and result["total"] is None
    assert [row["containing_function"]["name"] for row in result["items"]] == [
        "handler_08", "handler_09"]
    assert all(row["decompile_excerpt"]["lines"] for row in result["items"])
    assert all("_excerpt_fn" not in row for row in result["items"])
    assert rendered == [caller.start for caller in callers[8:10]], rendered


def _rendered_lines(bridge, monkeypatch, batches):
    """(lines, gutter addresses) as the callsite excerpt sees them, through the
    real render + gutter parser rather than a hand-built pair."""
    func = _FakeFunction(0x500000, "handle_reply")
    _install_fake_pseudo_c(monkeypatch, bridge, func, [batches])
    return bridge.read_listing._callsite_decompile_render(_FakeBV(), func)
