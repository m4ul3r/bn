"""#275 conformance gate: every collection-returning read emits the canonical
`{kind, items, total, ...}` envelope and NO deprecated alias key.

Driven against an empty fake bv -- an empty result still carries the full
envelope shape, which is exactly what we assert. A new collection read that
forgets `kind`/`items` or re-introduces an alias fails here.
"""
from __future__ import annotations

from _bridge_fakes import *  # noqa: F401,F403

# Keys the unification removed; none may appear at the top level of a collection read.
ALIAS_KEYS = {"functions", "classes", "code_refs", "data_refs", "symbols", "matches", "entries", "results"}

# (expected_kind, callable(instance) -> result, paged)
COLLECTION_READS = [
    ("functions", lambda i: i._list_functions("active"), True),
    ("functions", lambda i: i._search_functions("active", ""), True),
    ("strings", lambda i: i._strings("active", query=None, offset=0, limit=None), True),
    ("sections", lambda i: i._sections("active", offset=0, limit=None), True),
    ("types", lambda i: i._types("active", query=None, offset=0, limit=None), True),
    ("imports", lambda i: i._imports("active", offset=0, limit=None), True),
    ("exports", lambda i: i._exports("active", offset=0, limit=None), True),
]

# Same ops in --count mode: {kind, count, total}, still no alias key.
COUNT_READS = [
    ("functions", lambda i: i._list_functions("active", count_only=True)),
    ("strings", lambda i: i._strings("active", query=None, offset=0, limit=None, count_only=True)),
    ("sections", lambda i: i._sections("active", offset=0, limit=None, count_only=True)),
    ("types", lambda i: i._types("active", query=None, offset=0, limit=None, count_only=True)),
    ("imports", lambda i: i._imports("active", offset=0, limit=None, count_only=True)),
    ("exports", lambda i: i._exports("active", offset=0, limit=None, count_only=True)),
]


def _instance(monkeypatch):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    monkeypatch.setattr(inst.ctx, "_resolve_view", lambda selector: _FakeBV())
    return inst


def test_collection_reads_are_canonical(monkeypatch):
    inst = _instance(monkeypatch)
    for expected_kind, call, paged in COLLECTION_READS:
        res = call(inst)
        assert isinstance(res, dict), expected_kind
        assert res.get("kind") == expected_kind, f"{expected_kind}: kind={res.get('kind')!r}"
        assert isinstance(res.get("items"), list), f"{expected_kind}: items not a list"
        assert "total" in res, f"{expected_kind}: no total"
        if paged:
            assert {"offset", "limit", "returned", "has_more"} <= res.keys(), expected_kind
        leaked = ALIAS_KEYS & res.keys()
        assert not leaked, f"{expected_kind}: leaked alias key(s) {leaked}"


def test_count_reads_are_canonical(monkeypatch):
    inst = _instance(monkeypatch)
    for expected_kind, call in COUNT_READS:
        res = call(inst)
        assert res.get("kind") == expected_kind, f"{expected_kind}: kind={res.get('kind')!r}"
        assert "count" in res and "total" in res, expected_kind
        leaked = ALIAS_KEYS & res.keys()
        assert not leaked, f"{expected_kind}: count mode leaked {leaked}"


def test_shared_builders_emit_canonical_shape_without_aliases(monkeypatch):
    # The two choke points every list/function read flows through (#275).
    bridge = _load_bridge(monkeypatch)
    fn_env = bridge.read_listing._paged_function_result(None, [], offset=0, limit=None)
    list_env = bridge.read_misc._paged_list_result([], offset=0, limit=None, kind="strings")
    for env, kind in ((fn_env, "functions"), (list_env, "strings")):
        assert env["kind"] == kind
        assert env["items"] == [] and env["total"] == 0
        assert {"offset", "limit", "returned", "has_more"} <= env.keys()
        assert not (ALIAS_KEYS & env.keys())
