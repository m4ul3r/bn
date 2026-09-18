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

from collections.abc import Mapping

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

    # `function search` collapses the population BEFORE matching, so the phantom
    # twin cannot match as a second row with the conflicting size.
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


def test_function_search_scopes_duplicate_starts_to_the_matched_population_757(monkeypatch):
    """The two listing commands must give `duplicate_starts_collapsed` ONE meaning.

    `function list` scopes it to the population its `total` counts. `function
    search` collapsed the PRE-match population, so `function search <no-match>`
    answered `total 0, items []` beside `duplicate_starts_collapsed: 2` -- a key
    whose own contract (this many records were dropped from the rows you got)
    cannot be satisfied with no retained row, and one a reader is left
    reconciling against a total it has nothing to do with. It is scoped to the
    matched population now, which still keeps the property the collapse exists
    for: both records of one start are in the same group before any row is built,
    so a phantom twin cannot reach the page as a second row.
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

    # A partial match discloses only its own population: the one address the
    # answer contains was never a duplicate.
    partial = instance._search_functions("active", "widget_init")
    assert partial["total"] == 1
    assert "duplicate_starts_collapsed" not in partial

    # The colliding pair is still answered with the larger extent, once.
    twin = instance._search_functions("active", "widget_poll")
    assert twin["total"] == 1 and twin["returned"] == 1
    assert twin["duplicate_starts_collapsed"] == 1
    assert twin["items"][0]["size"] == 96


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
