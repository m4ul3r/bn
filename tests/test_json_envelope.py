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

# #819 stamped `kind` on three more reads, none of them a paged `items`
# collection: an object-shaped card (`evidence function`, rows under `calls`)
# and two scoped tag reads. They are exempt from `items` -- the envelope-tier
# list in `skills/bn/reference/reading.md` names the container each one uses --
# but NOT from the discriminator rule, which is why `tag list` is listed here
# beside them.
# (op label, expected kind, callable(instance) -> result, container key)
KINDED_READS = [
    ("evidence function", "function_evidence",
     lambda i: i._function_evidence("active", "probe", context=0), "calls"),
    ("tag types", "tag_types", lambda i: i._list_tag_types("active"), "tag_types"),
    ("tag get", "tags_at", lambda i: i._get_tags("active", "0x401000", None), "tags"),
    ("tag list", "tags", lambda i: i._list_tags("active"), "items"),
]


def _instance(monkeypatch, bv=None):
    bridge = _load_bridge(monkeypatch)
    inst = bridge.BinaryNinjaBridge()
    view = _FakeBV() if bv is None else bv
    monkeypatch.setattr(inst.ctx, "_resolve_view", lambda selector: view)
    return inst


def _kinded_instance(monkeypatch):
    """`evidence function` needs a function to read; the rest of `KINDED_READS`
    is driven against the same empty view as the collection gate above."""
    return _instance(monkeypatch, _FakeBV(functions=[_FakeFunction(0x401000, "probe")]))


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


def test_kinded_non_collection_reads_declare_their_container(monkeypatch):
    """#819/#275: a `kind` stamp is a promise about the payload's shape, so each
    read that carries one must name the container the reference documents for it
    and must not resurrect a dropped alias key."""
    inst = _kinded_instance(monkeypatch)
    for label, expected_kind, call, container in KINDED_READS:
        res = call(inst)
        assert res.get("kind") == expected_kind, f"{label}: kind={res.get('kind')!r}"
        assert isinstance(res.get(container), list), f"{label}: {container} not a list"
        leaked = (ALIAS_KEYS - {container}) & res.keys()
        assert not leaked, f"{label}: leaked alias key(s) {leaked}"


def test_one_kind_never_names_two_shapes(monkeypatch):
    """#819: `kind` is the field a consumer branches on, so it must identify the
    payload's SHAPE and not merely its subject. `tag list` (paged, rows under
    `items`) and `tag get` (unpaged, rows under `tags`) both answered to `tags`,
    so a reader that branched on `kind` and read `.items[]` got rows from one and
    null from the other -- the silent-null failure the #275 contract exists to
    stop. Derives the discriminator from the payload rather than asserting it, so
    this fails on the COLLISION and not on a renamed constant."""
    inst = _kinded_instance(monkeypatch)
    seen: dict[str, tuple[str, str]] = {}
    everything = ([(kind, kind, call, "items") for kind, call, _paged in COLLECTION_READS]
                  + KINDED_READS)
    for label, _expected_kind, call, container in everything:
        res = call(inst)
        kind = res.get("kind")
        assert isinstance(kind, str), f"{label}: no kind discriminator"
        prior_label, prior_container = seen.setdefault(kind, (label, container))
        assert prior_container == container, (
            f"kind {kind!r} names two shapes: `{prior_label}` keeps its rows under "
            f"{prior_container!r} and `{label}` under {container!r} -- a consumer "
            f"branching on kind gets a silent null from one of them")
