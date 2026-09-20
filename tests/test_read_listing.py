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

import re
from collections.abc import Mapping
from pathlib import Path

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


def test_function_listing_collapses_duplicate_start_addresses_757(monkeypatch):
    # #757: BN can hold two Function RECORDS for one start address with
    # conflicting sizes, both asserting `size_known: true` -- and the phantom is
    # the SMALLER (stub-shaped) one, exactly what a size-sorted triage or a
    # "small function = stub" heuristic reads as fact. One address is one
    # function here: the larger extent is kept and the collapse is disclosed.
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        # the conflicting pair: same start, both size_known, 96 bytes are real
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll", total_bytes=4),
        _FakeFunction(0x401080, "widget_flush", total_bytes=16),
    ]
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None)

    assert result["total"] == 3 and result["duplicate_starts_collapsed"] == 1
    by_address = {row["address"]: row for row in result["items"]}
    # No address is listed twice, and the surviving row is the real 96-byte body
    # rather than the 4-byte phantom.
    assert len(by_address) == result["returned"] == 3
    assert by_address["0x401014"]["size"] == 96
    assert by_address["0x401014"]["size_known"] is True

    # `--count` reads the same collapsed population, so it can no longer report
    # more functions than there are distinct starts -- and says why it is lower.
    counted = instance._list_functions(None, count_only=True)
    assert counted["count"] == counted["total"] == 3
    assert counted["duplicate_starts_collapsed"] == 1

    # `function search` collapses the same population the listing does, before
    # matching, so the phantom twin cannot reach the page as a second row with
    # the conflicting size.
    searched = instance._search_functions("active", "widget_poll")
    assert searched["total"] == 1 and searched["returned"] == 1
    assert searched["items"][0]["size"] == 96
    # The address filter (`--min-address`/`--max-address`) runs on the same
    # population, so the colliding window reports one row, not two.
    windowed = instance._list_functions(None, min_address="0x401014", max_address="0x401014")
    assert windowed["returned"] == 1 and windowed["items"][0]["size"] == 96


def test_annotation_summary_snapshots_the_live_address_comment_map_861(monkeypatch):
    # #861: `_annotation_summary` walked `list(address_comments.items())` -- the
    # ITEMS VIEW of the global comment map -- and the enclosing `except
    # Exception` turned any failure of that walk into a confident `comments: 0`
    # with no locations. Those counts are what `bn_kernel.assert_unannotated`
    # reads to certify a view pristine (#733 F2), so a walk that died must not be
    # able to fabricate a clean bill of health.
    #
    # The hazard is NOT reproducible against BN 6.1, where `address_comments`
    # builds and returns a fresh local dict per access; the snapshot is cheap
    # insurance, and the MUTATING view below is a fake that forces what the real
    # accessor does not produce. What this pins is that the snapshot is taken (and
    # that the counts stay honest if a BN build ever hands back the live store).
    bridge = _load_bridge(monkeypatch)

    class _LiveCommentStore(Mapping):
        """`address_comments` as a view that IS the live store: entries added
        mid-walk, which BN 6.1's accessor never exposes."""

        def __init__(self, entries: dict[int, str]) -> None:
            self._entries = dict(entries)
            self.injected = False

        def __getitem__(self, key: int) -> str:
            return self._entries[key]

        def __iter__(self):
            return iter(self._entries)

        def __len__(self) -> int:
            return len(self._entries)

        def items(self):
            for index, (address, text) in enumerate(self._entries.items()):
                if index == 0 and not self.injected:
                    self.injected = True
                    self._entries[0x2000] = "settled mid-walk"
                yield address, text

    class _LiveAnnotationBV(_FakeBV):
        """`_FakeBV.address_comments` hands back a fresh copy, like real BN; this
        hands back the store itself, so the snapshot has something to defend."""

        def __init__(self, store) -> None:
            super().__init__()
            self._live = store

        @property
        def address_comments(self):
            return self._live

    store = _LiveCommentStore({0x1000: "a comment"})
    summary = bridge.read_listing._annotation_summary(None, _LiveAnnotationBV(store))

    # Pre-fix the live-view RuntimeError is swallowed into comments: 0 (#861);
    # post-fix the map is materialised before the walk, so the counts and the
    # sample are the entries that existed when the call started.
    assert summary["comments"] == 1
    assert summary["comment_locations"] == [{"address": "0x1000", "comment": "a comment"}]


def test_annotation_summary_snapshots_the_live_per_function_map_861(monkeypatch):
    """#861 review: the GLOBAL comment map was snapshotted, but the per-function
    map twelve lines below was still walked through its live `items()` -- the same
    shape, on the same call, one rule for one map and none for the other.

    As with the global map, the hazard is not reproducible against BN 6.1
    (`Function.comments` also returns a fresh dict per access): the snapshot is
    cheap insurance, and the fake below forces the mutating store a BN build
    would have to expose for it to matter."""
    bridge = _load_bridge(monkeypatch)

    class _LiveCommentStore(Mapping):
        """`fn.comments` as a store that IS the live map: entries added mid-walk."""

        def __init__(self, entries):
            self._entries = dict(entries)
            self.injected = False

        def __getitem__(self, key):
            return self._entries[key]

        def __iter__(self):
            return iter(self._entries)

        def __len__(self):
            return len(self._entries)

        def items(self):
            for index, (address, text) in enumerate(self._entries.items()):
                if index == 0 and not self.injected:
                    self.injected = True
                    self._entries[0x2000] = "settled mid-walk"
                yield address, text

    fn = _FakeFunction(0x401000, "widget_init", total_bytes=28)
    fn.comments = _LiveCommentStore({0x401000: "a local comment"})
    bv = _FakeBV(functions=[fn])

    summary = bridge.read_listing._annotation_summary(None, bv)

    # Pre-fix the live walk raised (and callers degraded it to the
    # `unavailable` marker); post-fix the map is materialised before the walk, so
    # the local address comment is counted and sampled.
    assert summary["comments"] == 1
    assert summary["comment_locations"][0]["comment"] == "a local comment"


def test_duplicate_start_group_with_an_unreadable_extent_is_disclosed_757(monkeypatch):
    """#757 review: "keep the larger extent" is undefined when a record's extent
    cannot be read at all. Letting the record that happens to state a size win
    would promote the stub-shaped phantom over a real body the view would not
    size, so the group is left standing and the conflict is disclosed -- the
    issue's own second answer."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll"),  # extent unreadable
    ]
    _view(monkeypatch, instance, functions)

    result = instance._list_functions(None)

    assert result["returned"] == 3  # both colliding rows survive
    assert result["duplicate_starts_unresolved"] == 1
    assert "duplicate_starts_collapsed" not in result


def test_function_count_agrees_between_target_info_and_list_count_757(monkeypatch):
    """#757 review: collapsing the listing left `target info` (and the `target`
    block inside `evidence orient`) counting RAW records -- two answers to one
    question, one of them inside a payload that carries both."""
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll", total_bytes=4),
    ]
    bv = _view(monkeypatch, instance, functions)

    summary = bridge._function_name_summary(bv)
    counted = instance._list_functions(None, count_only=True)

    assert summary["function_count"] == counted["count"] == 2
    assert summary["duplicate_starts_collapsed"] == 1


# --- #793 review: the annotation block's samples, and what it stops saying -----

def test_annotation_summary_bounds_the_exclusion_sample_without_moving_a_count_793(monkeypatch):
    """`symbol_exclusions` was the one UNCAPPED sample in this block.

    On a real 2900-function target it was ~99.5% of the command's JSON (2048+
    rows, ~180 KB), dominated by a single loader-placeholder family -- enough to
    trip the tool's own token guard on `target info`, the command every agent
    runs first, and the same block ships unconditionally on `refresh` and
    `bundle function`. It is capped like its four sibling samples now.

    The cap may not move a single COUNTER: `bn_kernel.assert_unannotated` reads
    `comments` / `function_comments` / `user_symbols` (+`analyst_symbols`) to
    certify a view, so the rows the cap drops are disclosed as a count on the
    same block instead of by shrinking `placeholder_symbols`.
    """
    import json

    from types import SimpleNamespace

    bridge = _load_bridge(monkeypatch)
    limit = bridge.read_listing._ANNOTATION_SAMPLE_LIMIT
    families = ["init", "fini", "dest", "destr_1a2b", "compar"]
    placeholders = [
        SimpleNamespace(auto=False, name=families[i % len(families)],
                        address=0x401000 + i * 0x10)
        for i in range(2048)
    ]
    analyst = SimpleNamespace(auto=False, name="parse_record", address=0x600000)
    bv = _FakeBV(functions=[], symbols=[*placeholders, analyst])

    summary = bridge.read_listing._annotation_summary(None, bv)

    # Every count is exactly what it was before the cap.
    assert summary["placeholder_symbols"] == 2048
    assert summary["user_symbols"] == 2049 and summary["analyst_symbols"] == 1
    # The sample is bounded, ordered like the walk, and its shortfall is counted.
    assert len(summary["symbol_exclusions"]) == limit == 20
    assert [row["name"] for row in summary["symbol_exclusions"]] == [
        placeholders[i].name for i in range(limit)
    ]
    assert summary["symbol_exclusions_dropped"] == 2048 - limit
    assert (len(summary["symbol_exclusions"])
            + summary["symbol_exclusions_dropped"]) == summary["placeholder_symbols"]
    # The block stayed readable: 2048 exclusion rows is ~150 KB of this payload.
    assert len(json.dumps(summary)) < 8_000, len(json.dumps(summary))


def test_annotation_summary_degrades_instead_of_fabricating_zero_comments_793(monkeypatch):
    """The last swallowing `except` on the annotation surface.

    `_annotation_summary`'s first `try` ended `except Exception: comments = 0;
    comment_locations = []`, so any OTHER failure of the global comment read --
    not just the mid-walk mutation #861 fixed -- published a confident
    `comments: 0` with no `unavailable` marker. That is the count
    `assert_unannotated` certifies a view on (#733 F2), and it was the only
    branch of the surface whose failure a caller could not tell from a pristine
    view: an unreadable `get_symbols` already degraded correctly, which is what
    made this one a silent hole rather than a pattern.

    It now propagates to `_existing_annotations`, the ONE builder `target info`
    and `evidence orient` both publish, which answers with the same marker used
    for every other unreadable annotation count.
    """
    bridge = _load_bridge(monkeypatch)

    class _UnreadableCommentStore(_FakeBV):
        @property
        def address_comments(self):
            raise RuntimeError("comment store unreadable")

    bv = _UnreadableCommentStore(functions=[_FakeFunction(0x401000, "parse_header")])

    with pytest.raises(RuntimeError, match="comment store unreadable"):
        bridge.read_listing._annotation_summary(None, bv)

    block = bridge.read_listing._existing_annotations(None, bv)

    assert block["unavailable"] == (
        "annotation counts unavailable: comment store unreadable"
    )
    # No counts are claimed: a reader must not find an absent `comments` and
    # assume a pristine zero.
    assert "comments" not in block and "analyst_symbols" not in block


def test_function_search_scopes_duplicate_starts_to_the_rows_it_returns_757(monkeypatch):
    """The two listing commands must give `duplicate_starts_collapsed` ONE meaning.

    `function search` published the count taken at the collapse, so `function
    search <no-match>` answered `total 0, items []` beside
    `duplicate_starts_collapsed: 2` -- a key whose own contract (this many
    records were dropped from the rows you got) cannot be satisfied with no
    retained row, and one a reader is left reconciling against a total it has
    nothing to do with. The count is taken against the rows the answer KEPT
    now, which is also what makes the collapse itself safe to run on the whole
    population (as `function list` does): both records of one start are in the
    same group before any row is built, so a phantom twin cannot reach the page
    -- see `test_the_two_listing_commands_disclose_one_duplicate_start_identically_757`
    for the half that needs the query NOT to be a collapse boundary.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        # the conflicting pair: same start, both size_known, 96 bytes are real
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll", total_bytes=4),
    ]
    _view(monkeypatch, instance, functions)

    listed = instance._list_functions(None, count_only=True)
    assert listed["total"] == 2 and listed["duplicate_starts_collapsed"] == 1

    # Nothing matched, so no row could have been dropped from the answer.
    missed = instance._search_functions("active", "widget_missing")
    assert missed["total"] == 0 and missed["items"] == []
    assert "duplicate_starts_collapsed" not in missed
    missed_count = instance._search_functions("active", "widget_missing", count_only=True)
    assert missed_count["total"] == 0
    assert "duplicate_starts_collapsed" not in missed_count

    # A partial match discloses only what its own rows carry: the one address
    # the answer contains was never a duplicate.
    partial = instance._search_functions("active", "widget_init")
    assert partial["total"] == 1
    assert "duplicate_starts_collapsed" not in partial

    # The colliding pair is still answered with the larger extent, once.
    twin = instance._search_functions("active", "widget_poll")
    assert twin["total"] == 1 and twin["returned"] == 1
    assert twin["duplicate_starts_collapsed"] == 1
    assert twin["items"][0]["size"] == 96


def test_duplicate_start_counts_describe_the_filtered_population_757(monkeypatch):
    """...and the SAME scoping rule has to survive the row filters.

    The collapse was scoped to the matched population one command over, then
    counted BEFORE `--min-size` / `--named` dropped rows -- so `function list
    --min-size 1000 --count` on a view with one duplicated start answered
    `count 0, total 0` beside `duplicate_starts_collapsed: 1`, and `function
    search <q> --min-size 1000 --count` reproduced it. That is the exact
    total-0-with-collapse shape the search fix removed, and the one
    `_disclose_collapsed_starts`' own docstring says cannot happen: a key whose
    contract is "this many of the rows you got were merged" cannot be satisfied
    by rows the answer does not contain. The counts are measured against the
    RETAINED population now, whichever filter shaped it.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        # the conflicting pair: same start, both size_known, 96 bytes are real
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll", total_bytes=4),
    ]
    _view(monkeypatch, instance, functions)

    # A floor above every extent: no collapsed address is in the answer.
    counted = instance._list_functions(None, min_size=1000, count_only=True)
    assert counted["count"] == counted["total"] == 0
    assert "duplicate_starts_collapsed" not in counted
    listed = instance._list_functions(None, min_size=1000)
    assert listed["total"] == 0 and listed["items"] == []
    assert "duplicate_starts_collapsed" not in listed

    # One command over, the same shape through the matched population.
    searched = instance._search_functions("active", "widget_poll", min_size=1000,
                                          count_only=True)
    assert searched["total"] == 0
    assert "duplicate_starts_collapsed" not in searched

    # A floor the RETAINED record clears keeps the disclosure: the collapse is
    # still why that address is one row rather than two.
    kept = instance._list_functions(None, min_size=64, count_only=True)
    assert kept["total"] == 1 and kept["duplicate_starts_collapsed"] == 1
    kept_search = instance._search_functions("active", "widget_poll", min_size=64)
    assert kept_search["total"] == 1 and kept_search["duplicate_starts_collapsed"] == 1


def test_duplicate_start_counts_follow_the_named_filter_757(monkeypatch):
    """The other row filter on `function list`, and the UNRESOLVED half.

    `--named` partitions the population after the collapse, so a collapse among
    the auto-named rows was disclosed on the `--named` answer that contains none
    of them. The unresolved half follows the SAME rule as its sibling: the
    address counts while a record of it is in the answer. Requiring two
    survivors instead made the disclosure structurally unreachable under
    `--min-size` (an unreadable extent reads as 0, so the floor drops the
    unsized twin of every unresolved pair) and under `--named` (the two records
    land in different partitions), leaving a row whose extent was never
    comparable rendered exactly like a resolved one -- which
    `reading.md` tells the reader means the larger extent won.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "sub_401014", total_bytes=96),
        _FakeFunction(0x401014, "sub_401014", total_bytes=4),
    ]
    _view(monkeypatch, instance, functions)

    named = instance._list_functions(None, named=True, count_only=True)
    assert named["total"] == 1                       # widget_init only
    assert "duplicate_starts_collapsed" not in named
    unnamed = instance._list_functions(None, named=False, count_only=True)
    assert unnamed["total"] == 1 and unnamed["duplicate_starts_collapsed"] == 1

    # The unresolved half: a filter that drops one record of the pair leaves a
    # row whose extent could not be ranked against the record still at that
    # address, so the conflict is disclosed for as long as the address is in
    # the answer -- and disappears only with the address itself.
    unsized = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll"),  # extent unreadable
    ]
    _view(monkeypatch, instance, unsized)
    whole = instance._list_functions(None, count_only=True)
    assert whole["duplicate_starts_unresolved"] == 1
    filtered = instance._list_functions(None, min_size=64, count_only=True)
    assert filtered["total"] == 1
    assert filtered["duplicate_starts_unresolved"] == 1
    # ...and the same through `--named`, which splits the pair the other way.
    split = [
        _FakeFunction(0x401014, "sub_401014"),      # extent unreadable
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
    ]
    _view(monkeypatch, instance, split)
    for want in (True, False):
        answer = instance._list_functions(None, named=want, count_only=True)
        assert answer["total"] == 1
        assert answer["duplicate_starts_unresolved"] == 1, want
    # The address leaves the answer entirely -> so does the disclosure.
    gone = instance._list_functions(None, min_size=1000, count_only=True)
    assert gone["total"] == 0
    assert "duplicate_starts_unresolved" not in gone


def test_the_two_listing_commands_disclose_one_duplicate_start_identically_757(monkeypatch):
    """The match filter is a ROW FILTER like any other, so it may not change
    what the disclosure means.

    `function search` collapsed the POST-match population, so the group split
    whenever a query matched only one record of a duplicated start. Two
    consequences, both the shape #757 was filed for:

    * the SMALLER record -- the stub-shaped phantom the collapse exists to
      drop -- came back as a `size_known: true` row whenever the query happened
      to name it, while `function list` answered the same address with the real
      96-byte body;
    * a query naming the sized twin of an UNRESOLVED pair returned that row with
      no disclosure at all, while `function list` returned the identical row
      beside `duplicate_starts_unresolved: 1`.

    Both commands collapse the same population now (everything the address
    filter left) and count against the rows their own answer kept, so the two
    answers agree for the same retained record.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    # One start, two records, DIFFERENT names -- so a query can name either.
    collided = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub", total_bytes=4),
    ]
    _view(monkeypatch, instance, collided)

    # The phantom is not a function, so naming it cannot conjure it back.
    phantom = instance._search_functions("active", "poll_stub")
    assert phantom["total"] == 0 and phantom["items"] == []
    assert "duplicate_starts_collapsed" not in phantom
    # ...and the retained record answers with its real extent and the collapse.
    body = instance._search_functions("active", "widget_poll")
    assert body["total"] == 1 and body["items"][0]["size"] == 96
    assert body["duplicate_starts_collapsed"] == 1
    listed = instance._list_functions(None, min_address="0x401014", max_address="0x401014")
    assert listed["items"][0]["size"] == 96
    assert listed["duplicate_starts_collapsed"] == 1

    # The unresolved half: a query naming the sized twin must disclose exactly
    # what `function list` discloses for that same row.
    unresolved = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub"),   # extent unreadable
    ]
    _view(monkeypatch, instance, unresolved)
    from_list = instance._list_functions(None, min_size=64, count_only=True)
    from_search = instance._search_functions("active", "widget_poll", count_only=True)
    assert from_list["total"] == from_search["total"] == 1
    assert from_list["duplicate_starts_unresolved"] == 1
    assert from_search["duplicate_starts_unresolved"] == 1

    # A query that matches nothing still reports no collapse: the counts are
    # taken against the rows the answer kept, not against the population.
    missed = instance._search_functions("active", "widget_missing", count_only=True)
    assert missed["total"] == 0
    assert "duplicate_starts_collapsed" not in missed
    assert "duplicate_starts_unresolved" not in missed


def test_target_info_publishes_the_unresolved_duplicate_starts_757(monkeypatch):
    """Both halves of the #757 disclosure, on both surfaces.

    `function list` disclosed `duplicate_starts_collapsed` AND
    `duplicate_starts_unresolved`; `_function_name_summary` -- the block
    `target info` spreads into its payload, and the `target` block inside
    `evidence orient` -- unpacked the unresolved count and dropped it. The two
    surfaces agreed on every number they printed while one of them omitted the
    reason, which reads as "the larger extent won" for an address where no
    extent could be compared at all and both records are still live.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    functions = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "widget_poll"),  # extent unreadable
    ]
    bv = _view(monkeypatch, instance, functions)

    summary = bridge._function_name_summary(bv)
    listed = instance._list_functions(None, count_only=True)

    assert summary["duplicate_starts_unresolved"] == 1
    assert listed["duplicate_starts_unresolved"] == 1
    assert summary["function_count"] == listed["count"] == 3
    assert "duplicate_starts_collapsed" not in summary


def test_target_info_text_discloses_the_duplicate_start_collapse_757(monkeypatch):
    """`target info` is the command every agent runs first, and its TEXT face
    printed a post-collapse function count with nothing saying so.

    This change made `_function_name_summary` collapse duplicated starts, so
    the number in `functions: N functions (...)` is no longer BN's record
    count -- and the note that explains the difference was wired only into
    `function list` and `--count`. A text reader therefore saw a silently
    smaller number on the first command they run, which is exactly the
    JSON-only disclosure the rest of this change removes.

    Measured end to end: the count and the note both come from the one
    collapse, so neither can drift from the other.
    """
    from bn.commands.binary import _render_target_info_text_with_annotations

    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    bv = _view(monkeypatch, instance, [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub", total_bytes=4),     # ranked, merged
        _FakeFunction(0x401100, "widget_tick", total_bytes=64),
        _FakeFunction(0x401100, "tick_stub"),                    # extent unreadable
    ])
    summary = bridge._function_name_summary(bv)
    # Five BN records, four functions: the number the text face prints.
    assert summary["function_count"] == 4
    text = _render_target_info_text_with_annotations(
        {"selector": "svc", "arch": "x86_64", **summary})

    assert "functions: 4 functions (4 named, 0 auto-named)" in text, text
    note = [line for line in text.splitlines() if "duplicate starts" in line]
    assert len(note) == 1, text
    # Immediately under the count it qualifies, not appended after the whole card.
    assert text.splitlines().index(note[0]) == text.splitlines().index(
        next(line for line in text.splitlines() if "functions: 4 functions" in line)) + 1, text
    assert "1 start address(es) carried duplicate function records" in note[0], note
    assert "the larger extent was kept" in note[0], note
    assert "no record was chosen there" in note[0], note
    # The denominator is the count printed above it, so the two cannot disagree.
    assert "all 4 function(s) this answer reports" in note[0], note

    # A view with no collapse renders exactly as it did before -- no standing
    # alarm on every `target info`.
    clean_bv = _view(monkeypatch, instance, [
        _FakeFunction(0x401000, "widget_init", total_bytes=28)])
    clean = _render_target_info_text_with_annotations(
        {"selector": "svc", "arch": "x86_64", **bridge._function_name_summary(clean_bv)})
    assert "duplicate" not in clean, clean


def test_annotation_summary_omits_the_dropped_key_when_nothing_was_dropped_793(monkeypatch):
    """`symbol_exclusions_dropped` follows the convention of its two siblings in
    this module (`callers_dropped`, `duplicate_starts_collapsed`): the key exists
    only when the cap actually dropped rows. A clean view used to publish a
    `0`, which made "the cap fired here" unreadable from the key set (#793
    review nit)."""
    bridge = _load_bridge(monkeypatch)
    bv = _FakeBV(functions=[_FakeFunction(0x401000, "widget_init", total_bytes=28)])

    summary = bridge.read_listing._annotation_summary(None, bv)

    assert "symbol_exclusions_dropped" not in summary



def test_records_with_an_unreadable_start_are_never_grouped_together_757(monkeypatch):
    """A record whose `start` cannot be read cannot be shown to be a duplicate
    of ANYTHING, so it must form its own group.

    The fallback keyed the record ITSELF, i.e. on its own `__hash__`/`__eq__`:
    two records that compare equal (BN wrappers around the same handle, and the
    minimal fakes a unit test builds) landed in one group and were reported as a
    collapse the view never had, while an unhashable record raised `TypeError`
    out of a read op. Keyed on a boxed `id()` now, which is the identity the
    comment always claimed and cannot collide with a start address.
    """
    bridge = _load_bridge(monkeypatch)

    class _NoStart:
        """A record whose extent AND start are unreadable, and which claims to
        equal every other record of its kind."""

        name = "ambiguous"
        raw_name = "ambiguous"

        @property
        def start(self):
            raise AttributeError("start")

        def __eq__(self, other):
            return isinstance(other, _NoStart)

        __hash__ = None          # unhashable, like a BN object with __eq__ set

    pair = [_NoStart(), _NoStart()]
    kept, collapse = bridge.read_listing._collapse_duplicate_starts(pair)

    # Both pass through untouched, and nothing is claimed about them.
    assert kept == pair
    assert collapse.counts(kept) == (0, 0)

    # ...and the same through the command: two rows, no invented collapse.
    instance = bridge.BinaryNinjaBridge()
    bv = _view(monkeypatch, instance, pair)
    summary = bridge._function_name_summary(bv)
    assert summary["function_count"] == 2
    assert "duplicate_starts_collapsed" not in summary


def test_duplicate_start_rows_carry_their_own_marker_757(monkeypatch):
    """A row of a duplicated start says so ON THE ROW, not only in a count.

    The unresolved disclosure was an address-less COUNT, and the sized twin it
    counts still publishes `size_known: true` with nothing on it: under
    `--sort size` the unsized record sorts to 0 and the sized one to its extent,
    so the two records of one start land far apart -- or on different pages --
    and a reader holding either row has no way to tell it was never ranked. The
    count says "one address somewhere in this answer", which is not locatable.

    A per-row marker was chosen over an envelope list of the affected
    addresses: it is bounded (one short string on the rows that have one, none
    otherwise), it rides with the row through every `--sort`/`--offset`, and it
    answers the question the address list cannot -- WHICH of two rows at one
    address was picked on extent.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _view(monkeypatch, instance, [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub", total_bytes=4),     # ranked, dropped
        _FakeFunction(0x401100, "widget_tick", total_bytes=64),
        _FakeFunction(0x401100, "tick_stub"),                    # extent unreadable
    ])

    listed = instance._list_functions(None)
    marks = {row["address"]: row.get("duplicate_start") for row in listed["items"]}
    # The plain address says nothing; the merged one says it won on extent; the
    # unranked pair says, on BOTH of its rows, that nothing was ranked there.
    assert marks["0x401000"] is None, marks
    assert marks["0x401014"] == "collapsed", marks
    unranked = [row for row in listed["items"] if row["address"] == "0x401100"]
    assert len(unranked) == 2, unranked
    assert {row["duplicate_start"] for row in unranked} == {"unresolved"}, unranked
    # The sized twin still reports a real size -- the marker is what says that
    # size did not win a comparison.
    sized = [row for row in unranked if row["size"] == 64]
    assert sized and sized[0]["size_known"] is True, unranked

    # The shape the count alone cannot serve: `--sort size` separates the pair
    # across pages, and each page still carries the marker on its own row.
    first = instance._list_functions(None, sort="size", limit=2)
    last = instance._list_functions(None, sort="size", offset=2, limit=2)
    assert [row.get("duplicate_start") for row in first["items"]] == ["unresolved", None], first["items"]
    assert [row.get("duplicate_start") for row in last["items"]] == ["unresolved", "collapsed"], last["items"]

    # `function search` hands back the same row, so the marker cannot be a
    # listing-only field.
    searched = instance._search_functions("active", "tick_stub")
    assert searched["items"][0]["duplicate_start"] == "unresolved", searched["items"]
    # A clean view carries no marker at all: absent, not a null column.
    _view(monkeypatch, instance, [_FakeFunction(0x401000, "widget_init", total_bytes=28)])
    assert "duplicate_start" not in instance._list_functions(None)["items"][0]


def test_duplicate_start_counts_are_scoped_to_the_total_not_the_page_757(monkeypatch):
    """ONE scoping rule, and paging is not part of it.

    The counts are taken after every ROW FILTER (`--min-size`, `--named`, the
    query) and before paging, so they describe the population `total` reports.
    `--offset`/`--limit` therefore cannot move them: a window holding none of
    the counted addresses still carries the disclosure, which is the case the
    note's "the larger extent was kept" wording read as a claim about the rows
    on screen.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    _view(monkeypatch, instance, [
        _FakeFunction(0x401000, "widget_poll", total_bytes=96),
        _FakeFunction(0x401000, "poll_stub", total_bytes=4),     # collapses
        _FakeFunction(0x401100, "widget_tick", total_bytes=64),
        _FakeFunction(0x401100, "tick_stub"),                    # unresolved
    ] + [_FakeFunction(0x402000 + i * 0x10, f"fn_{i}", total_bytes=32) for i in range(4)])

    whole = instance._list_functions(None)
    assert whole["total"] == 7
    assert whole["duplicate_starts_collapsed"] == 1
    assert whole["duplicate_starts_unresolved"] == 1

    # A window that contains NEITHER counted address: same two counts, same
    # total, and not one row of either duplicated start.
    page = instance._list_functions(None, offset=3, limit=3)
    assert [row["address"] for row in page["items"]] == ["0x402000", "0x402010", "0x402020"]
    assert page["total"] == 7
    assert page["duplicate_starts_collapsed"] == 1
    assert page["duplicate_starts_unresolved"] == 1
    assert all("duplicate_start" not in row for row in page["items"]), page["items"]

    # `function search` pages the same way, and the counts follow `total` there
    # too rather than the window.
    searched = instance._search_functions("active", "", offset=4, limit=2)
    assert [row["address"] for row in searched["items"]] == ["0x402010", "0x402020"]
    assert searched["total"] == 7
    assert searched["duplicate_starts_collapsed"] == 1
    assert searched["duplicate_starts_unresolved"] == 1


def _duplicate_starts_bullet() -> str:
    """The duplicate-start-disclosure entry of `skills/bn/reference/reading.md`.

    Everything between this list item's first line and the line that opens the
    NEXT one. Three narrower rules have already been defeated: the first line
    only (round 6), stopping at the first blank or `#`-prefixed line (round
    7), and stopping at a real ATX heading, which let `###### note: <lie>` sit
    directly under the entry and outside the compared region (round 8). A
    reader sees whatever is printed between the two bullets, so whatever is
    printed between the two bullets IS the entry.
    """
    reference = (Path(__file__).resolve().parent.parent
                 / "skills" / "bn" / "reference" / "reading.md")
    lines = reference.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines)
              if "duplicate_starts_unresolved" in line]
    assert starts, "reading.md carries no duplicate-start bullet"
    assert len(starts) == 1, (
        "more than one line of reading.md names the disclosure key, so the "
        f"entry is no longer identifiable: lines {[i + 1 for i in starts]}")
    entry = [lines[starts[0]]]
    for nxt in lines[starts[0] + 1:]:
        if nxt.startswith("- "):
            break
        entry.append(nxt)
    return "\n".join(entry)


def _publishes_duplicate_starts(payload) -> bool:
    """Does this payload disclose a duplicated start ANYWHERE in JSON?

    Recursive on purpose. `evidence orient` carries the counts twice -- at its
    own level and inside its nested `target` block -- so a top-level-only test
    reads a payload that still discloses in JSON as silent, and the entry's
    parity claim then holds vacuously on exactly the surface it was written
    for (#757 review round 7).
    """
    if isinstance(payload, Mapping):
        return any(str(k).startswith("duplicate_start")
                   or _publishes_duplicate_starts(v) for k, v in payload.items())
    if isinstance(payload, (list, tuple)):
        return any(_publishes_duplicate_starts(v) for v in payload)
    return False


def _text_discloses_duplicate_starts(text: str, count_keys) -> bool:
    """Does this rendering say anything about a duplicated start?

    Either the `// duplicate starts` note a card prints, or the count keys
    themselves -- a payload with no dedicated renderer falls back to a JSON
    dump, which prints the keys verbatim. Both are disclosure; only a face
    showing neither is silent while its payload publishes (#757 review round
    8).
    """
    return "duplicate starts" in text or any(k in text for k in count_keys)


def _duplicate_start_faces(bridge, instance, monkeypatch, population) -> dict:
    """The text faces a duplicate-start payload reaches, each paired with the
    payload behind it.

    The reference makes ONE claim about text -- parity: a face says something
    about a duplicated start exactly when the payload behind it publishes one.
    Where the line lands inside a card is a rendering detail and is
    deliberately not claimed (#757 review round 7).

    Each face is the payload the bridge really builds rendered by the text
    renderer the COMMAND really uses -- `target info` composes annotations onto
    the shared summary, so measuring the shared renderer measured a proxy.
    """
    from bn import formatters
    from bn.commands import binary as binary_cmd

    bv = _view(monkeypatch, instance, population)
    monkeypatch.setattr(instance.targets, "resolve", lambda selector: bv)
    monkeypatch.setattr(instance.targets, "refresh", lambda: [])
    listing = instance._list_functions(None)
    counted = instance._list_functions(None, count_only=True)
    searched = instance._search_functions("active", "")
    info = instance._target_info(None)
    # Only the digest's sub-reads are stubbed, and only the ones that need a
    # richer view than this population: its function count, and both counts,
    # still come from the real collapse.
    monkeypatch.setattr(bridge.read_misc, "_imports",
                        lambda ctx, sel, **k: {"kind": "imports_summary",
                                               "total_symbols": 0, "by_kind": {}})
    monkeypatch.setattr(bridge.read_misc, "_strings",
                        lambda ctx, sel, **k: {"kind": "strings", "items": [], "total": 0})
    monkeypatch.setattr(bridge.read_misc, "_sections",
                        lambda ctx, sel, **k: {"items": [{"name": ".text"}], "total": 1})
    digest = instance._orient_digest(None)
    return {
        "function list": (listing, formatters._render_function_list_text(listing)),
        "function search": (searched,
                            formatters._render_function_list_text(searched)),
        "function list --count": (counted,
                                  formatters._render_function_count_text(counted)),
        "target info": (info,
                        binary_cmd._render_target_info_text_with_annotations(info)),
        "evidence orient": (digest, formatters._render_orient_text(digest)),
        # The renderer any payload without a dedicated one falls back to: it
        # prints the keys verbatim, so it discloses too -- measured, not
        # assumed, because the entry's claim is over every text face.
        "json fallback": (info, formatters._render_fallback_text(info)),
    }


def _measure_duplicate_start_disclosure(bridge, instance, monkeypatch) -> dict:
    """The duplicate-start contract, READ OFF the bridge.

    Two things, because they are the two a reader has to get right: what a
    duplicated start does to the ROWS, and what it does to the COUNTS. The
    reference entry is RENDERED from this dict by
    `_render_duplicate_starts_bullet`, so every identifier, marker and number
    in the prose comes from a live envelope instead of from a phrase someone
    typed beside one.
    """
    # RANKED: every extent readable, so the group can be ordered by extent.
    ranked_pop = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub", total_bytes=4),
    ]
    _view(monkeypatch, instance, ranked_pop)
    ranked = instance._list_functions(None)
    ranked_rows = [r for r in ranked["items"] if r["address"] == "0x401014"]
    dropped_named = instance._search_functions("active", "poll_stub")

    # UNRANKED: one extent unreadable, so the rule cannot pick a record. THREE
    # records, so the prose may not describe the group as a pair.
    unranked_pop = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_alt", total_bytes=32),
        _FakeFunction(0x401014, "poll_stub"),          # extent unreadable
    ]
    _view(monkeypatch, instance, unranked_pop)
    unranked = instance._list_functions(None)
    unranked_rows = [r for r in unranked["items"] if r["address"] == "0x401014"]
    named_back = tuple(instance._search_functions("active", row["name"])["total"]
                       for row in unranked_rows)
    # A row filter that leaves ONE record of the address, and one that leaves
    # none of them: the counts count addresses, so only the second drops.
    kept = instance._list_functions(None, min_size=64, count_only=True)
    gone = instance._list_functions(None, min_size=1000, count_only=True)
    # ...and a row filter narrows the ROWS too, which the entry's row clauses
    # therefore may not state unconditionally: under the same filter this
    # address keeps one of its three records, and the filtered-out names stop
    # answering (#757 review round 8).
    filtered = instance._list_functions(None, min_size=64)
    filtered_rows = [r for r in filtered["items"] if r["address"] == "0x401014"]
    filtered_named_hits = tuple(
        instance._search_functions("active", fn.name, min_size=64)["total"]
        for fn in unranked_pop if fn.start == 0x401014 and fn.total_bytes is None)

    # SAME NAME: #757's own reported shape -- both records at the start carry
    # the loader-assigned name, so "naming the dropped record" answers with
    # the RETAINED one. A clause generalised off the distinct-name fixture
    # alone is false here (#757 review round 8).
    same_name_pop = [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "sub_401014", total_bytes=96),
        _FakeFunction(0x401014, "sub_401014", total_bytes=4),
    ]
    _view(monkeypatch, instance, same_name_pop)
    same_name = instance._list_functions(None)
    same_name_rows = [r for r in same_name["items"] if r["address"] == "0x401014"]
    shared_name_hits = instance._search_functions("active", "sub_401014")["total"]

    # PAGED: a window holding none of the counted addresses.
    _view(monkeypatch, instance, ranked_pop + [
        _FakeFunction(0x402000 + i * 0x10, f"fn_{i}", total_bytes=32) for i in range(4)
    ])
    paged = instance._list_functions(None, offset=3, limit=2)

    collided = _duplicate_start_faces(bridge, instance, monkeypatch, [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
        _FakeFunction(0x401014, "widget_poll", total_bytes=96),
        _FakeFunction(0x401014, "poll_stub", total_bytes=4),
        _FakeFunction(0x401100, "widget_tick", total_bytes=64),
        _FakeFunction(0x401100, "tick_stub"),
    ])
    clean_faces = _duplicate_start_faces(bridge, instance, monkeypatch, [
        _FakeFunction(0x401000, "widget_init", total_bytes=28),
    ])

    envelopes = (ranked, unranked, same_name, kept, gone, filtered, paged,
                 *(payload for payload, _ in collided.values()))
    row_keys = {k for env in (ranked, unranked) for row in env["items"]
                for k in row if k.startswith("duplicate_start")}
    return {
        # Every key any measured envelope carries, so the render's backticked
        # vocabulary can be checked against the payload rather than a list.
        "envelope_keys": frozenset(k for env in envelopes for k in env),
        "count_keys": tuple(sorted(
            {k for env in envelopes for k in env
             if k.startswith("duplicate_starts_")})),
        "row_keys": tuple(sorted(row_keys)),
        "markers": tuple(sorted(
            {row[k] for env in (ranked, unranked) for row in env["items"]
             for k in row_keys if k in row})),
        # Which marker each case puts on its rows, so the entry can say what
        # the two values MEAN without the mapping being a phrase.
        "ranked_marker": ranked_rows[0].get("duplicate_start") if ranked_rows else None,
        "unranked_marker": ({row.get("duplicate_start") for row in unranked_rows}
                            or {None}).pop(),
        "ranked_rows": len(ranked_rows),
        "ranked_kept_size": ranked_rows[0]["size"] if ranked_rows else None,
        "ranked_group_sizes": tuple(sorted(
            fn.total_bytes for fn in ranked_pop if fn.start == 0x401014)),
        "ranked_extents_all_readable": all(
            fn.total_bytes is not None for fn in ranked_pop if fn.start == 0x401014),
        "unranked_extents_all_readable": all(
            fn.total_bytes is not None for fn in unranked_pop if fn.start == 0x401014),
        "ranked_dropped_hits": dropped_named["total"],
        "same_name_rows": len(same_name_rows),
        "shared_name_hits": shared_name_hits,
        "unranked_records": sum(1 for fn in unranked_pop if fn.start == 0x401014),
        "unranked_rows": len(unranked_rows),
        "unranked_named_hits": named_back,
        "filtered_rows": len(filtered_rows),
        "filtered_named_hits": filtered_named_hits,
        "kept_total": kept["total"],
        "kept_count": kept.get("duplicate_starts_unresolved"),
        "gone_count": gone.get("duplicate_starts_unresolved"),
        "paged_count": paged.get("duplicate_starts_collapsed"),
        "paged_holds_a_counted_address": any(row["address"] == "0x401014"
                                             for row in paged["items"]),
        # The target-info face has to be the COMMAND's composed card, not the
        # shared summary renderer: only the command's adds the #793 block.
        "target_info_face_is_the_command_card": (
            "existing annotations" in collided["target info"][1]),
        "parity": tuple(
            (f"{state} {face}", _publishes_duplicate_starts(payload),
             _text_discloses_duplicate_starts(
                 text, ("duplicate_starts_collapsed", "duplicate_starts_unresolved")))
            for state, faces in (("collided", collided), ("clean", clean_faces))
            for face, (payload, text) in faces.items()
        ),
    }


#: The only backticked tokens the rendered entry may name that are NOT keys
#: an envelope carries: the two commands that collapse, and the three row /
#: page flags. A hand-written constant is itself an editable surface -- the
#: residual below names it -- so it is kept to tokens the payload cannot
#: supply, and everything else the entry backticks is checked against the
#: envelope.
_RENDER_VOCABULARY = frozenset({
    "function list", "function search", "--offset", "--limit", "--min-size",
})

#: Identifier-shaped words (snake_case) anywhere in the render, backticked or
#: not. Round 8 shipped an invented `duplicate_starts_phantom` past a
#: backtick-only vocabulary check by simply not quoting it.
_IDENTIFIER_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _render_duplicate_starts_bullet(m: dict) -> str:
    """The reference entry, RENDERED from the measured envelope.

    Six review rounds pinned this bullet against phrases -- three substring
    checks, then a paired claim inventory, then sentence completeness, then a
    residue rule, then byte equality against a hand-written constant. Each one
    was defeated by prose that sat where the current rule did not look, and the
    last one by editing the prose and its own constant together. A guard
    derived from prose cannot catch prose the author also wrote into the guard,
    so this stops deriving it from prose: the entry's operative content is
    produced HERE, out of the measurement, and the file has to match.

    The trade is stated exactly, because over-claiming is what this entry
    keeps being reviewed for and rounds 7 and 8 both over-claimed here.

    Closed: doc-side drift of any shape. Everything printed between this list
    item's first line and the next item -- a comma, a continuation line, a
    lazy `#757`-style line, an indented paragraph, a real `###### heading` --
    is inside the compared region and must EQUAL this render.

    Closed: identifiers and numbers. Every snake_case token in the render,
    quoted or not, must be a key an envelope carries or a marker a row
    carries; every backticked token must be that or one of the five command
    and flag names in `_RENDER_VOCABULARY`; every number must be one the
    measurement supplied. An invented key, a forged marker and an unmeasured
    count are unwritable here.

    NOT closed, and nothing closes it. Three editable surfaces, in descending
    order of how likely they are to be edited by accident:

    * this template plus the reference, in plain English drawn from measured
      vocabulary -- a hedge, a contradicting parenthetical, a verbless clause,
      a false claim about a command the entry names, a false count written as
      a WORD, a claim true of the fakes and false of the engine, or a
      consistent DELETION of a true clause;
    * `_RENDER_VOCABULARY` itself: adding one token re-admits the gloss it
      excludes;
    * the measured POPULATION. Every universal clause here is generalised
      from the fixtures above, so a clause is only as true as the shapes they
      cover. Round 8 found exactly this: "naming a dropped record returns 0
      rows" was true only because that fixture's records had distinct names.
      The fixtures now cover distinct names, a shared name, a three-record
      unranked group and a row-filtered answer; a shape they miss is the next
      place this entry will be wrong.

    Rounds 3-6 escalated a prose-derived guard five times trying to close the
    first of those and each escalation was defeated; the operator's ruling was
    to stop escalating and cut the claim instead. What makes the residual
    survivable is size: the entry says what a duplicated start does to the
    ROWS and to the COUNTS and stops, so this template is short enough to read
    whole and a sentence in it with no measured value beside it is visible on
    sight.
    """
    keys = " / ".join(f"`{k}`" for k in m["count_keys"])
    row_key = " / ".join(f"`{k}`" for k in m["row_keys"])
    extent = ("larger" if m["ranked_kept_size"] == max(m["ranked_group_sizes"])
              else "smaller")
    stays = ("every record of that address stays in the answer"
             if m["unranked_rows"] == m["unranked_records"]
             else f"only {m['unranked_rows']} of its {m['unranked_records']} "
                  "records stay in the answer")
    named = ("naming any of them returns it"
             if set(m["unranked_named_hits"]) == {1}
             else "naming one of them returns nothing")
    return (
        "- **Duplicate function start addresses collapse, and the collapse is "
        "disclosed (#757).** Binary Ninja can hold more than one Function record "
        "for one start address, with sizes that disagree, so `function list` and "
        "`function search` collapse them. Before any row filter narrows the "
        "answer: **Ranked** (every extent readable) — the address contributes "
        f"exactly {m['ranked_rows']} row, the one carrying the {extent} extent, "
        f"and a name only a dropped record carried returns "
        f"{m['ranked_dropped_hits']} rows, while a name the group shares still "
        f"returns the {m['shared_name_hits']} retained row. **Unranked** (any "
        f"extent unreadable) — no record is chosen, so {stays} and {named}. A "
        "row filter applies on top of that and narrows the rows as well as the "
        f"total: under `--min-size` an unranked address keeps the "
        f"{m['filtered_rows']} record that passes and a filtered-out name "
        f"answers with {m['filtered_named_hits'][0]} rows. Each returned row of "
        f"a duplicated start carries {row_key} — "
        f'`"{m["ranked_marker"]}"` on the row a ranked address kept, '
        f'`"{m["unranked_marker"]}"` on every row of an address nothing could '
        f"be ranked at. The counts {keys} count ADDRESSES, not dropped records, "
        "over the whole filtered answer reports, and are unchanged by "
        "`--offset`/`--limit` — so a page can disclose an address none of its "
        "rows holds — and are absent when zero. A payload that publishes them "
        "never reaches a text face that is silent about them, so no card reads "
        "clean while its own answer discloses."
    )


def test_reading_reference_states_the_duplicate_start_rule_the_bridge_applies_757(monkeypatch):
    """The reference entry has to be what the bridge does -- so it is rendered
    from a measurement of the bridge, and the file must match.

    Rounds 3-6 each removed one false claim from this entry and shipped its
    mirror image, because the guard was always derived from the prose: a
    substring list, then paired claims, then sentence completeness, then a
    token-residue rule, then byte equality against a hand-written pin. Round 7
    reproduced the end state of that arms race -- a plainly false, token-free
    clause shipped GREEN when the entry and its pin were edited consistently.

    The answer is not a sixth rule. It is a smaller claim: the entry states
    only what a duplicated start does to the ROWS and to the COUNTS, and the
    surface-by-surface, placement and ordering prose is gone. What is left is
    asserted against the bridge below -- across four populations, because
    round 8 showed a clause generalised off ONE fixture (records with
    distinct names) shipping false for the shape the issue actually reports
    (records sharing the loader name) -- and then RENDERED, so the entry and
    the bridge cannot disagree.
    """
    bridge = _load_bridge(monkeypatch)
    instance = bridge.BinaryNinjaBridge()
    m = _measure_duplicate_start_disclosure(bridge, instance, monkeypatch)

    # --- THE ROWS -------------------------------------------------------
    # Ranked: one row survives and it is the larger extent.
    assert m["ranked_extents_all_readable"], m
    assert m["ranked_rows"] == 1, m
    assert m["ranked_kept_size"] == max(m["ranked_group_sizes"]), m
    # A name only a dropped record carried is gone...
    assert m["ranked_dropped_hits"] == 0, m
    # ...but a name the group SHARES still answers, with the retained row.
    assert m["same_name_rows"] == 1, m
    assert m["shared_name_hits"] == 1, m
    # Unranked: nothing is chosen, so every record stays and each is reachable.
    assert not m["unranked_extents_all_readable"], m
    assert m["unranked_records"] > 2, "the group must not be a pair"
    assert m["unranked_rows"] == m["unranked_records"], m
    assert set(m["unranked_named_hits"]) == {1}, m
    # A row filter narrows the ROWS too, so the two cases above are claims
    # about an unfiltered answer and the entry has to say so.
    assert m["filtered_rows"] == 1, m
    assert set(m["filtered_named_hits"]) == {0}, m
    # Each surviving row says which case it is in, with both values in play.
    assert m["row_keys"] == ("duplicate_start",), m
    assert m["markers"] == ("collapsed", "unresolved"), m

    # --- THE COUNTS -----------------------------------------------------
    assert m["count_keys"] == ("duplicate_starts_collapsed",
                               "duplicate_starts_unresolved"), m
    # Addresses, not dropped records: one surviving record keeps the count...
    assert (m["kept_total"], m["kept_count"]) == (1, 1), m
    # ...and a filter that removes every record of the address removes it.
    assert m["gone_count"] is None, m
    # Paging does not move them: this page discloses an address it does not hold.
    assert m["paged_count"] == 1 and not m["paged_holds_a_counted_address"], m

    # --- TEXT / JSON PARITY ---------------------------------------------
    # Publication is detected ANYWHERE in the payload: the orient digest
    # carries the counts in its nested `target` block as well as at its own
    # level, and a top-level-only predicate let that face hold vacuously.
    assert _publishes_duplicate_starts({"target": {"duplicate_starts_collapsed": 1}})
    assert _publishes_duplicate_starts({"items": [{"duplicate_start": "collapsed"}]})
    assert not _publishes_duplicate_starts({"target": {"function_count": 4}})
    # The summary face is the COMMAND's composed card, not the shared renderer.
    assert m["target_info_face_is_the_command_card"], m
    # Every face -- including the generic JSON fallback a payload without its
    # own renderer reaches -- says something about a duplicated start exactly
    # when its payload publishes one. Both directions, collided and clean.
    for face, publishes, discloses in m["parity"]:
        assert discloses is publishes, (
            f"{face}: payload publishes={publishes} but its text "
            f"discloses={discloses}")
    assert {p for _, p, _ in m["parity"]} == {True, False}, (
        "parity was measured in only one direction", m["parity"])

    # --- THE REFERENCE IS THAT MEASUREMENT, RENDERED --------------------
    expected = _render_duplicate_starts_bullet(m)
    # Identifiers: every snake_case token in the render, QUOTED OR NOT, has to
    # be a key an envelope carries or a marker a row carries. A backtick-only
    # check let an invented key ship unquoted (#757 review round 8).
    measured_names = m["envelope_keys"] | set(m["row_keys"]) | set(m["markers"])
    assert set(_IDENTIFIER_TOKEN.findall(expected)) <= measured_names, (
        "the rendered entry names identifiers no measured envelope or row "
        f"carries: {sorted(set(_IDENTIFIER_TOKEN.findall(expected)) - measured_names)}"
        f"\n{expected}")
    # ...and every backticked token is one of those or a declared command/flag.
    backticked = set(re.findall(r"`([^`]+)`", expected))
    unexplained = backticked - _RENDER_VOCABULARY - measured_names - {
        f'"{v}"' for v in m["markers"]}
    assert not unexplained, (
        f"the rendered entry backticks tokens nothing produced: {sorted(unexplained)}"
        f"\n{expected}")
    # Numbers: only ones the measurement supplied, plus the issue number.
    assert set(re.findall(r"\d+", expected)) == {
        "757",                                   # the issue this entry answers
        str(m["ranked_rows"]), str(m["ranked_dropped_hits"]),
        str(m["shared_name_hits"]), str(m["filtered_rows"]),
        str(m["filtered_named_hits"][0]),
    }, expected
    # The two mappings the render turns a measurement into WORDS -- which
    # marker belongs to which case, and which extent the collapse keeps --
    # are pinned against the measurement, because inverting either is
    # invisible to every assertion above.
    assert f'`"{m["ranked_marker"]}"` on the row a ranked address kept' in expected
    assert f'`"{m["unranked_marker"]}"` on every row of an address' in expected
    assert m["ranked_marker"] != m["unranked_marker"], m
    kept_the_larger = m["ranked_kept_size"] == max(m["ranked_group_sizes"])
    assert ("carrying the larger extent" in expected) is kept_the_larger, expected
    assert ("carrying the smaller extent" in expected) is not kept_the_larger, expected
    # ...and each case's lead-in carries its own defining condition, so the
    # two cannot be swapped while the rest still reads true.
    assert ("**Ranked** (every extent readable)" in expected) is (
        m["ranked_extents_all_readable"]), expected
    assert ("**Unranked** (any extent unreadable)" in expected) is (
        not m["unranked_extents_all_readable"]), expected

    assert _duplicate_starts_bullet() == expected, (
        "skills/bn/reference/reading.md's duplicate-start entry is not what the "
        "bridge measures. This entry is GENERATED from the measurement above -- "
        "do not hand-edit it and do not add a sentence to it; replace the line "
        "with:\n\n" + expected)
