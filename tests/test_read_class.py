from __future__ import annotations

import importlib
import types

read_class = importlib.import_module("bn_agent_bridge.read_class")
split = read_class._split_qualified_method


def test_split_plain_method():
    assert split("net::Session::onData(int)") == ("net::Session", "onData(int)")


def test_split_nested_class():
    assert split("a::b::Outer::Inner::run()") == ("a::b::Outer::Inner", "run()")


def test_split_namespaced_free_function():
    # Name-indistinguishable from a method; still clusters under the namespace.
    assert split("net::make_session(int)") == ("net", "make_session(int)")


def test_split_template_class_args_with_scope():
    assert split("std::map<int, std::string>::insert(int)") == (
        "std::map<int, std::string>",
        "insert(int)",
    )


def test_split_template_nested_angle():
    assert split("Vec<Pair<A, B>>::push(A)") == ("Vec<Pair<A, B>>", "push(A)")


def test_split_ctor_and_dtor():
    assert split("net::Session::Session(int)") == ("net::Session", "Session(int)")
    assert split("net::Session::~Session()") == ("net::Session", "~Session()")


def test_split_operator_call():
    assert split("net::Buf::operator()(int)") == ("net::Buf", "operator()(int)")


def test_split_operator_new():
    assert split("net::Pool::operator new(unsigned long)") == (
        "net::Pool",
        "operator new(unsigned long)",
    )


def test_split_no_scope_returns_none():
    assert split("memcpy") == (None, "memcpy")
    assert split("main") == (None, "main")


class _Sym:
    def __init__(self, raw_name, short_name, address, sym_type="SymbolType.DataSymbol"):
        self.raw_name = raw_name
        self.short_name = short_name
        self.name = raw_name
        self.address = address
        self.type = sym_type


class _Fn:
    def __init__(self, start, mangled, demangled):
        self.start = start
        self.name = mangled
        self.raw_name = mangled
        self.symbol = _Sym(mangled, demangled, start)


class _RegistryBV:
    def __init__(self, functions, symbols):
        self.functions = list(functions)
        self._symbols = list(symbols)

    def get_symbols(self):
        return list(self._symbols)


def _make_registry_bv():
    fns = [
        _Fn(0x1000, "_ZN3net7SessionC1Eh", "net::Session::Session(unsigned char)"),
        _Fn(0x1100, "_ZN3net7SessionD1Ev", "net::Session::~Session()"),
        _Fn(0x1200, "_ZN3net7Session6onDataEi", "net::Session::onData(int)"),
        _Fn(0x1300, "_ZN3net4makeEv", "net::make()"),          # free fn -> name-only
        _Fn(0x1400, "_ZN3net4Pool5allocEv", "net::Pool::alloc()"),  # ctor-less; ti present
    ]
    syms = [
        _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x9000),
        _Sym("_ZTIN3net7SessionE", "typeinfo for net::Session", 0x9100),
        _Sym("_ZTIN3net4PoolE", "typeinfo for net::Pool", 0x9200),
    ]
    return _RegistryBV(fns, syms)


def test_registry_clusters_methods():
    bv = _make_registry_bv()
    reg = read_class._build_class_registry(None, bv)
    assert set(reg) >= {"net::Session", "net::Pool", "net"}
    sess = reg["net::Session"]
    kinds = {m["demangled"]: m["kind"] for m in sess["methods"]}
    assert kinds["net::Session::Session(unsigned char)"] == "ctor"
    assert kinds["net::Session::~Session()"] == "dtor"
    assert kinds["net::Session::onData(int)"] == "method"


def test_registry_confidence_levels():
    reg = read_class._build_class_registry(None, _make_registry_bv())
    assert reg["net::Session"]["confidence"] == "rtti"   # has vtable+typeinfo
    assert reg["net::Pool"]["confidence"] == "rtti"       # has typeinfo only
    assert reg["net"]["confidence"] == "name-only"        # namespace-like


def test_registry_attaches_vtable_and_typeinfo():
    reg = read_class._build_class_registry(None, _make_registry_bv())
    assert reg["net::Session"]["vtable"]["address"] == "0x9000"
    assert reg["net::Session"]["typeinfo"]["address"] == "0x9100"
    assert reg["net::Pool"]["vtable"] is None


def test_registry_handles_underscore_rtti_markers():
    # Real BN renders RTTI markers with UNDERSCORES, and -- crucially -- the
    # vtable form carries a leading underscore (_vtable_for_X) while typeinfo /
    # typeinfo-name do NOT (typeinfo_for_X, typeinfo_name_for_X). A fixed marker
    # list keyed on "typeinfo for "/"_typeinfo_for_" silently missed typeinfo, so
    # RTTI bases never resolved on real targets. Regression for that fix.
    fns = [_Fn(0x1000, "_ZN3net7SessionC1Ev", "net::Session::Session()")]
    syms = [
        _Sym("_ZTVN3net7SessionE", "_vtable_for_net::Session", 0x9000),
        _Sym("_ZTIN3net7SessionE", "typeinfo_for_net::Session", 0x9100),
        _Sym("_ZTSN3net7SessionE", "typeinfo_name_for_net::Session", 0x9200),
    ]
    reg = read_class._build_class_registry(None, _RegistryBV(fns, syms))
    rec = reg["net::Session"]
    assert rec["vtable"]["address"] == "0x9000"
    assert rec["typeinfo"]["address"] == "0x9100"        # was None before the fix
    assert rec["typeinfo_name"]["address"] == "0x9200"
    assert rec["confidence"] == "rtti"


def test_registry_matches_demangled_raw_rtti_symbols():
    # On real targets BN can set a typeinfo symbol's RAW name to the DEMANGLED
    # form (`_typeinfo_for_X`) and create no `_ZTI...` symbol at all (45 vtables
    # vs 16 `_ZTI` raws was observed). Identifying RTTI by the mangled raw prefix
    # dropped these, so typeinfo -- and every RTTI base class -- silently never
    # resolved. Identify by the demangled marker instead. (#205 dogfood regression.)
    fns = [_Fn(0x1000, "_ZN1XC1Ev", "X::X()")]
    syms = [
        _Sym("_vtable_for_X", "_vtable_for_X", 0x9000),       # raw IS the demangled form
        _Sym("_typeinfo_for_X", "_typeinfo_for_X", 0x9100),   # no `_ZTI` raw anywhere
    ]
    rec = read_class._build_class_registry(None, _RegistryBV(fns, syms))["X"]
    assert rec["vtable"]["address"] == "0x9000"
    assert rec["typeinfo"]["address"] == "0x9100"        # was None before the fix
    assert rec["confidence"] == "rtti"


class _SlotCtx:
    """ctx for _vtable_layout slot-scan tests. Supplies a valid Itanium typeinfo
    header (word[1] resolves to a typeinfo) so the layout gate passes and the
    slot rows are exercised; ``ti_ok=False`` makes the header invalid."""
    def __init__(self, rows, ti_ok=True):
        self._rows = rows
        self._ti_ok = ti_ok

    def _pointer_size(self, bv):
        return 8

    def _read_pointer_value(self, bv, addr, *, size=None):
        return 0x9100 if self._ti_ok else 0   # word[1] = typeinfo ptr (or null)

    def _typeinfo_name_at(self, bv, addr):
        return "net::Klass" if self._ti_ok else None

    def _pointer_table_layout(self, bv, start, *, entries, stride):
        # #303: the real reader returns the #275 envelope keyed on `items`.
        return {"kind": "pointer_table", "items": self._rows}


class _RecoverCtx:
    """ctx for the #354 typeinfo->vtable backwalk: _vtable_layout_for returns the
    prebuilt layout for a known vtable address (None / empty for a non-vtable ref),
    and _read_pointer_value returns the word[0] offset-to-top.

    Like the real ``_vtable_layout`` it returns a FRESH dict per call, so a layout
    the backwalk mutates (offset_to_top / typeinfo_backwalk provenance) doesn't
    alias the layout the symbolized-primary path produced for the same address."""
    def __init__(self, layouts, otts):
        self._layouts = layouts
        self._otts = otts
    def _pointer_size(self, bv):
        return 8
    def _read_pointer_value(self, bv, addr, *, size=None):
        return self._otts.get(addr, 0)
    def _vtable_layout_for(self, bv, addr):
        layout = self._layouts.get(addr)
        return dict(layout) if layout is not None else layout


class _RecoverBV:
    """The crafted refs are returned ONLY for the typeinfo address they were built
    for; any other queried address yields []. A regression that calls
    get_data_refs on the wrong address (e.g. the vtable instead of the typeinfo)
    is then caught instead of silently passing on globally-returned refs."""
    def __init__(self, ti, refs):
        self._ti = ti
        self._refs = refs
    def get_data_refs(self, addr):
        return list(self._refs) if addr == self._ti else []


def test_recover_vtables_from_typeinfo_primary_and_secondary():
    # #354/#412: stripped binary -- no _ZTV symbol. A data xref to the typeinfo
    # lands at (vtable_addr + ptr); backwalk -ptr, validate via _vtable_layout_for,
    # classify primary (offset-to-top 0) vs secondary (non-zero).
    ti = 0x9100
    bv = _RecoverBV(ti, [0x9008, 0x9028, 0x9050])
    layouts = {
        0x9000: {"address": "0x9000", "slots": [{"index": 0, "address": "0xa000"}]},
        0x9020: {"address": "0x9020", "slots": [{"index": 0, "address": "0xa100"}]},
        0x9048: {"address": "0x9048", "slots": []},   # not a vtable -> filtered
    }
    otts = {0x9000: 0, 0x9020: (1 << 64) - 16, 0x9048: 0}   # secondary offset-to-top = -16
    ctx = _RecoverCtx(layouts, otts)
    out = read_class._recover_vtables_from_typeinfo(ctx, bv, ti)
    assert out is not None
    assert out["primary"]["address"] == "0x9000"
    assert out["primary"]["offset_to_top"] == 0
    assert out["primary"]["typeinfo_backwalk"] is True
    assert len(out["secondary"]) == 1
    assert out["secondary"][0]["address"] == "0x9020"
    assert out["secondary"][0]["offset_to_top"] == -16


def test_enrich_backwalks_vtable_for_stripped_class():
    # #354/#412: a class with typeinfo but NO vtable symbol (stripped) gets its
    # primary + secondary vtables recovered via the typeinfo backwalk in _enrich.
    class _EnrichCtx(_RecoverCtx):
        def _object_size_for(self, bv, rec):
            return None
        def _bases_for(self, bv, rec):
            return []
        def _instances_for(self, bv, rec):
            return {"construction_sites": [], "stored_globals": []}

    bv = _RecoverBV(0x9100, [0x9008, 0x9028])
    layouts = {
        0x9000: {"address": "0x9000", "slots": [{"index": 0, "address": "0xa000"}]},
        0x9020: {"address": "0x9020", "slots": [{"index": 0, "address": "0xa100"}]},
    }
    otts = {0x9000: 0, 0x9020: (1 << 64) - 16}
    ctx = _EnrichCtx(layouts, otts)
    rec = {"name": "Shape", "vtable": None, "typeinfo": {"address": "0x9100"},
           "methods": [], "bases": []}
    out = read_class._enrich(ctx, bv, rec)
    assert out["vtable"]["address"] == "0x9000"          # primary recovered (#354)
    assert out["vtable"]["typeinfo_backwalk"] is True
    assert len(out["secondary_vtables"]) == 1            # secondary recovered (#412)
    assert out["secondary_vtables"][0]["address"] == "0x9020"
    assert any("typeinfo backwalk" in n for n in out["notes"])


def test_enrich_recovers_secondary_for_symbolized_primary_class():
    # #412 (codex Finding 1): the COMMON multiple-inheritance case -- the primary
    # `_ZTV` symbol SURVIVES (so the registry already carries a vtable address) but
    # the secondary-base subobject vtables are unsymbolized. _enrich must still
    # backwalk the typeinfo to recover the secondaries, WITHOUT disturbing the
    # symbolized primary layout.
    class _EnrichCtx(_RecoverCtx):
        def _object_size_for(self, bv, rec):
            return None
        def _bases_for(self, bv, rec):
            return []
        def _instances_for(self, bv, rec):
            return {"construction_sites": [], "stored_globals": []}

    bv = _RecoverBV(0x9100, [0x9008, 0x9028])
    layouts = {
        # 0x9000 is the symbolized primary (ott 0); 0x9020 is the unsymbolized
        # secondary base subobject (ott -16).
        0x9000: {"address": "0x9000", "slots": [{"index": 0, "address": "0xa000"}]},
        0x9020: {"address": "0x9020", "slots": [{"index": 0, "address": "0xa100"}]},
    }
    otts = {0x9000: 0, 0x9020: (1 << 64) - 16}
    ctx = _EnrichCtx(layouts, otts)
    rec = {"name": "Shape", "vtable": {"address": "0x9000"},
           "typeinfo": {"address": "0x9100"}, "methods": [], "bases": []}
    out = read_class._enrich(ctx, bv, rec)
    # The symbolized primary is preserved (resolved via _vtable_layout_for), NOT
    # replaced by a backwalk result.
    assert out["vtable"]["address"] == "0x9000"
    assert "typeinfo_backwalk" not in out["vtable"]
    # ...but the unsymbolized secondary is still recovered and attached, and the
    # primary's own address is NOT re-emitted as a secondary.
    assert len(out["secondary_vtables"]) == 1
    assert out["secondary_vtables"][0]["address"] == "0x9020"
    assert any("typeinfo backwalk" in n for n in out["notes"])


def test_recover_drops_zero_offset_to_top_duplicate():
    # #412 (codex Finding 2): a second data-ref to the typeinfo that resolves to a
    # DISTINCT vtable address whose offset-to-top is ALSO 0 (a construction-vtable /
    # RTTI artifact, not a real secondary base subobject) must NOT be emitted as a
    # secondary -- only a strictly non-zero offset-to-top is a real secondary.
    ti = 0x9100
    bv = _RecoverBV(ti, [0x9008, 0x9028])
    layouts = {
        0x9000: {"address": "0x9000", "slots": [{"index": 0, "address": "0xa000"}]},
        0x9020: {"address": "0x9020", "slots": [{"index": 0, "address": "0xa100"}]},
    }
    otts = {0x9000: 0, 0x9020: 0}   # both offset-to-top 0 -> the second is a dup
    ctx = _RecoverCtx(layouts, otts)
    out = read_class._recover_vtables_from_typeinfo(ctx, bv, ti)
    assert out is not None
    assert out["primary"]["address"] == "0x9000"   # first ott==0 wins as primary
    assert out["secondary"] == []                  # zero-ott duplicate dropped


def test_render_class_show_text_shows_secondary_vtables():
    # #412: a multiple-inheritance class renders its secondary (offset-to-top != 0)
    # vtable group beneath the primary.
    from bn.formatters import _render_class_show_text
    rec = {
        "name": "Shape", "confidence": "rtti", "size": None, "bases": [],
        "methods": [], "instances": {"construction_sites": [], "stored_globals": []},
        "vtable": {"address": "0x9000", "offset_to_top": 0,
                   "slots": [{"index": 0, "address": "0xa000", "method": {"name": "draw"}}]},
        "secondary_vtables": [{"address": "0x9040", "offset_to_top": -8,
                               "slots": [{"index": 0, "address": "0xb000", "method": {"name": "name"}}]}],
    }
    text = _render_class_show_text(rec)
    assert "secondary vtable @ 0x9040 (offset-to-top -8)" in text
    assert "[0] 0xb000  name" in text


def test_vtable_layout_skips_when_header_not_typeinfo():
    # The Itanium gate: if word[1] doesn't resolve to a typeinfo, this isn't a
    # real local vtable object (import/GOT slot or relocated-to-zero PIE slot) --
    # return no slots instead of decoding adjacent GOT/data as fake slots.
    rows = [{"index": 0, "value": "0x1000", "readable": True,
             "target": {"status": "function", "function": {"name": "f0"}}}]
    layout = read_class._vtable_layout(_SlotCtx(rows, ti_ok=False), object(), 0x9000)
    assert layout["slots"] == []


def test_vtable_layout_demangles_slot_name():
    # Slot labels should be demangled (symbol short_name), like the methods list,
    # not the raw mangled fn.name. The mangled name is kept alongside. (#205)
    class _Fn2:
        start = 0x41eeb0
        name = "_ZN3net7Session6onDataEv"
        symbol = _Sym("_ZN3net7Session6onDataEv", "net::Session::onData()", 0x41eeb0)

    class _BV2:
        def get_function_at(self, a):
            return _Fn2() if a == 0x41eeb0 else None

    class _Ctx:
        def _pointer_size(self, bv):
            return 8

        def _read_pointer_value(self, bv, addr, *, size=None):
            return 0x9100

        def _typeinfo_name_at(self, bv, addr):
            return "net::Session"

        def _pointer_table_layout(self, bv, start, *, entries, stride):
            return {"kind": "pointer_table", "items": [
                {"index": 0, "value": "0x41eeb0", "readable": True,
                 "target": {"status": "function",
                            "function": {"name": "_ZN3net7Session6onDataEv", "address": "0x41eeb0"}}}]}

    slot = read_class._vtable_layout(_Ctx(), _BV2(), 0x9000)["slots"][0]
    assert slot["method"]["display_name"] == "net::Session::onData()"
    assert slot["method"]["name"] == "_ZN3net7Session6onDataEv"   # mangled kept

    from bn.formatters import _render_class_show_text
    rec = {"name": "net::Session", "confidence": "rtti", "size": None, "bases": [],
           "methods": [], "instances": {"construction_sites": [], "stored_globals": []},
           "vtable": {"address": "0x9000", "slots": [slot]}}
    assert "net::Session::onData()" in _render_class_show_text(rec)
    assert "_ZN3net7Session6onDataEv" not in _render_class_show_text(rec)


class _SymBV:
    """A bv whose get_symbol_at names the __cxa_pure_virtual extern slot."""
    def __init__(self, syms=None):
        self._syms = syms or {}

    def get_symbol_at(self, addr):
        name = self._syms.get(addr)
        return _Sym(name, name, addr) if name else None

    def get_function_at(self, addr):
        return None


def test_vtable_layout_includes_null_and_external_slots_pie():
    # #441: a PIE C++ vtable interleaves null slots (pure-virtual relocated to 0)
    # and external `__cxa_pure_virtual` slots before/among the real methods. The
    # old scan broke at the first null and returned ZERO slots even though the
    # methods below resolve. Include null + external slots; terminate only at a
    # real boundary (the next object's typeinfo data pointer); trim trailing null.
    rows = [
        {"index": 0, "value": "0x0", "readable": True,
         "target": {"status": "null", "context": {"kind": "null"}}},
        {"index": 1, "value": "0x108800", "readable": True,
         "target": {"status": "mapped", "function": None, "context": {"kind": "extern"}}},
        {"index": 2, "value": "0x201040", "readable": True,
         "target": {"status": "function",
                    "function": {"name": "_ZN3net7Session6onDataEv", "address": "0x201040"}}},
        {"index": 3, "value": "0x0", "readable": True,  # trailing null -> trimmed
         "target": {"status": "null", "context": {"kind": "null"}}},
        {"index": 4, "value": "0x108000", "readable": True,  # typeinfo ptr = boundary
         "target": {"status": "mapped", "function": None, "context": {"kind": "string"}}},
    ]
    bv = _SymBV({0x108800: "__cxa_pure_virtual"})
    layout = read_class._vtable_layout(_SlotCtx(rows), bv, 0x9000)
    slots = layout["slots"]
    assert [s["index"] for s in slots] == [0, 1, 2]           # trailing null trimmed
    assert slots[0].get("null") is True
    assert slots[1].get("external") is True and slots[1]["pure_virtual"] is True
    assert slots[2]["method"]["name"] == "_ZN3net7Session6onDataEv"


def test_vtable_layout_terminates_at_unmapped_secondary_header():
    # The next sub-vtable's offset-to-top (a small negative, e.g. -8) reads as an
    # unmapped word and must end the primary scan (not be rendered as a slot).
    rows = [
        {"index": 0, "value": "0x201040", "readable": True,
         "target": {"status": "function", "function": {"name": "f0", "address": "0x201040"}}},
        {"index": 1, "value": "0xfffffffffffffff8", "readable": True,
         "target": {"status": "unmapped", "context": {"kind": "unmapped"}}},
        {"index": 2, "value": "0x201200", "readable": True,
         "target": {"status": "function", "function": {"name": "f2", "address": "0x201200"}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), _SymBV(), 0x9000)
    assert [s["index"] for s in layout["slots"]] == [0]


def test_vtable_layout_stops_at_data_slot():
    # A pointer the classifier deems data (kind != "code") is not a slot -- it
    # ends the scan rather than rendering adjacent data as fake slots (#205).
    rows = [
        {"index": 0, "value": "0x1000", "readable": True,
         "target": {"status": "function", "function": {"name": "f0"}}},
        {"index": 1, "value": "0x9999", "readable": True,
         "target": {"status": "mapped", "function": None,
                    "context": {"kind": "data", "segment": {"executable": False}}}},
        {"index": 2, "value": "0x2000", "readable": True,
         "target": {"status": "function", "function": {"name": "f2"}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), object(), 0x9000)
    assert [s["index"] for s in layout["slots"]] == [0]


def test_vtable_layout_includes_unanalyzed_code_slot():
    # A pointer the classifier deems code (kind == "code") but with no function
    # yet is code BN hasn't analyzed -- still a valid (unnamed) vtable slot.
    rows = [
        {"index": 0, "value": "0x1000", "readable": True,
         "target": {"status": "mapped", "function": None,
                    "context": {"kind": "code", "segment": {"executable": True}}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), object(), 0x9000)
    assert len(layout["slots"]) == 1 and layout["slots"][0]["unnamed"] is True


def test_vtable_layout_rejects_data_in_executable_segment():
    # The firmware case: a string/data pointer that lives in an r-x segment
    # (.rodata mapped into the code LOAD segment). The executable bit must NOT
    # make it a slot -- only kind == "code" does. Otherwise data renders as fake
    # unnamed virtual methods (#205 review; cf. seam._address_is_code).
    rows = [
        {"index": 0, "value": "0x1000", "readable": True,
         "target": {"status": "mapped", "function": None,
                    "context": {"kind": "string", "segment": {"executable": True}}}},
        {"index": 1, "value": "0x2000", "readable": True,
         "target": {"status": "function", "function": {"name": "f1"}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), object(), 0x9000)
    assert layout["slots"] == []   # stops at the data-in-r-x slot, no fake method


def test_vtable_layout_reads_items_envelope_not_legacy_entries():
    # #303 regression: the pointer-table reader returns the canonical #275
    # envelope keyed on `items`. _vtable_layout read the pre-#275 `entries` key,
    # so EVERY vtable resolved to zero slots and `class show` falsely reported a
    # recoverable dispatch table as unrecoverable ("no slots resolved"). A table
    # that supplies ONLY `items` (no `entries`) must resolve its slots.
    class _ItemsOnlyCtx(_SlotCtx):
        def _pointer_table_layout(self, bv, start, *, entries, stride):
            return {"kind": "pointer_table", "items": self._rows}  # no `entries` key

    rows = [
        {"index": 0, "value": "0x1238", "readable": True,
         "target": {"status": "function", "function": {"name": "JsonCodecD1Ev"}}},
        {"index": 1, "value": "0x1258", "readable": True,
         "target": {"status": "function", "function": {"name": "JsonCodecD0Ev"}}},
    ]
    layout = read_class._vtable_layout(_ItemsOnlyCtx(rows), object(), 0x9000)
    assert [s["index"] for s in layout["slots"]] == [0, 1]   # slots resolved from `items`


def test_resolve_class_names_template_query_not_collapsed():
    # A specific template query must match only its specialization, not every
    # Vec<...> (the old raw-"::"-substring + template-arg-dropping bug).
    reg = {"ns::Vec<std::string>": {}, "ns::Vec<float>": {}, "ns::Vec": {}}
    assert read_class._resolve_class_names(reg, "Vec<std::string>") == ["ns::Vec<std::string>"]
    assert read_class._resolve_class_names(reg, "Vec") == ["ns::Vec"]


def test_resolve_class_names_cross_namespace_leaf():
    reg = {"a::Foo": {}, "b::Foo": {}, "Bar": {}}
    assert read_class._resolve_class_names(reg, "Foo") == ["a::Foo", "b::Foo"]


def test_class_list_envelope_filters_and_pages():
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    out = read_class._class_list(_Ctx(), None, include_all=True)
    assert out["kind"] == "classes" and "classes" not in out  # #275: items-only
    assert out["returned"] == len(out["items"])
    assert out["has_more"] is False
    names = [c["name"] for c in out["items"]]
    assert "net::Session" in names and out["total"] == len(names)
    # Default (no --all) drops name-only clusters like the bare "net" namespace.
    confirmed = read_class._class_list(_Ctx(), None, include_all=False)
    assert "net" not in [c["name"] for c in confirmed["items"]]
    assert "net::Session" in [c["name"] for c in confirmed["items"]]


def test_class_list_envelope_has_more_when_paged():
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    out = read_class._class_list(_Ctx(), None, include_all=True, limit=1)
    assert out["kind"] == "classes" and "classes" not in out
    assert out["limit"] == 1
    assert out["returned"] == 1
    assert out["has_more"] is True


def test_is_library_class():
    f = read_class._is_library_class
    assert f("std::vector<int>")
    assert f("std::__cxx11::basic_string<char>")
    assert f("__gnu_cxx::__normal_iterator<char*>")
    assert f("__cxxabiv1::__si_class_type_info")
    assert f("std::_Bind<bool ()>")
    assert f("_Hashtable<std::pair<int, int>>")        # bare reserved-id internal
    assert f("_Sp_counted_ptr_inplace<Foo>")
    assert f("__detail::_Map_base")
    assert not f("alexaClientSDK::endpoints::EndpointBuilder")
    assert not f("net::Session")
    assert not f("Controller")
    assert not f("MyStd")          # 'std' must be a whole component, not a prefix
    assert not f("standard::Foo")


def test_class_list_no_stl_filters_library_classes():
    fns = [
        _Fn(0x1000, "_ZN3net7SessionC1Ev", "net::Session::Session()"),
        _Fn(0x1100, "_ZNSt6vectorIiE9push_backEi", "std::vector<int>::push_back(int)"),
        _Fn(0x1200, "_ZN9__gnu_cxx3fooEv", "__gnu_cxx::foo()"),
    ]
    syms = [_Sym("_ZTVN3net7SessionE", "_vtable_for_net::Session", 0x9000)]
    bv = _RegistryBV(fns, syms)

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    out = read_class._class_list(_Ctx(), None, include_all=True, no_stl=True)
    names = [c["name"] for c in out["items"]]
    assert "net::Session" in names
    assert "std::vector<int>" not in names
    assert "__gnu_cxx" not in names
    assert out["no_stl"] is True
    assert out["library_suppressed"] >= 2
    # Without --no-stl the library classes are present.
    full = read_class._class_list(_Ctx(), None, include_all=True)
    assert "std::vector<int>" in [c["name"] for c in full["items"]]
    assert full["library_suppressed"] == 0


def test_class_list_populates_bases_for_shown_rows():
    # `class list` must populate the per-row `bases` field (it was always [] —
    # base decode ran only in the `show` path). Bases are decoded for the
    # returned page so a single list recovers the inheritance graph. (#205 review)
    fns = [_Fn(0x1000, "_ZN3net7SessionC1Ev", "net::Session::Session()")]
    syms = [
        _Sym("_ZTVN3net7SessionE", "_vtable_for_net::Session", 0x9000),
        _Sym("_ZTIN3net7SessionE", "typeinfo_for_net::Session", 0x9100),
    ]
    bv = _RegistryBV(fns, syms)

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

        def _bases_for(self, b, rec):
            if rec["name"] == "net::Session":
                return [{"name": "net::Endpoint", "address": "0x9200", "kind": "public"}]
            return []

    out = read_class._class_list(_Ctx(), None)
    row = next(c for c in out["items"] if c["name"] == "net::Session")
    assert row["bases"] == ["net::Endpoint"]


class _VtableCtx:
    """Minimal ctx exposing the pointer-table reader over a fake slot map."""
    def __init__(self, slots, pure_addr=0xDEAD):
        # slots: list of (value_addr, function_name_or_None)
        self._slots = slots
        self._pure = pure_addr

    def _pointer_size(self, bv):
        return 8

    def _read_pointer_value(self, bv, addr, *, size=None):
        return 0x9100   # word[1] = a valid typeinfo ptr (passes the layout gate)

    def _typeinfo_name_at(self, bv, addr):
        return "net::Klass"

    def _pointer_table_layout(self, bv, start, *, entries, stride):
        rows = []
        for i, (val, fname) in enumerate(self._slots):
            target = {"status": "function", "normalized": hex(val),
                      "function": ({"name": fname, "address": hex(val)} if fname else None)}
            if val == self._pure:
                target = {"status": "function", "normalized": hex(val),
                          "function": {"name": "__cxa_pure_virtual", "address": hex(val)}}
            rows.append({"index": i, "entry_address": hex(start + i * stride),
                         "value": hex(val), "readable": True, "plausible": True, "target": target})
        return {"entries": rows}


def test_vtable_layout_skips_header_and_marks_slots():
    bv = object()
    slots = [(0x40e8b0, "onData"), (0x40e3d0, None), (0xDEAD, "__cxa_pure_virtual")]
    ctx = _VtableCtx(slots)
    layout = read_class._vtable_layout(ctx, bv, 0x9000)
    assert layout["address"] == "0x9000"
    # header words skipped: read starts at 0x9000 + 2*8
    s0, s1, s2 = layout["slots"]
    assert s0["index"] == 0 and s0["method"]["name"] == "onData"
    assert s1["unnamed"] is True            # sub_* / no symbol
    assert s2["pure_virtual"] is True


def test_vtable_layout_stops_at_unmapped_slot():
    # The scan must terminate at the first non-function/unreadable slot (the next
    # object / padding), not skip it and keep collecting later rows.
    class _Ctx:
        def _pointer_size(self, bv):
            return 8

        def _read_pointer_value(self, bv, addr, *, size=None):
            return 0x9100   # valid typeinfo header -> passes the layout gate

        def _typeinfo_name_at(self, bv, addr):
            return "net::Klass"

        def _pointer_table_layout(self, bv, start, *, entries, stride):
            return {"entries": [
                {"index": 0, "value": "0x1000", "readable": True,
                 "target": {"status": "function", "function": {"name": "f0"}}},
                {"index": 1, "value": None, "readable": False, "target": {}},
                {"index": 2, "value": "0x2000", "readable": True,
                 "target": {"status": "function", "function": {"name": "f2"}}},
            ]}

    layout = read_class._vtable_layout(_Ctx(), object(), 0x9000)
    assert [s["index"] for s in layout["slots"]] == [0]   # stopped at the gap


class _SizeCtx:
    def __init__(self, type_width=None, new_size=None):
        self._type_width = type_width
        self._new_size = new_size

    def _find_type(self, bv, name):
        if self._type_width is None:
            return None
        class _T:
            width = self._type_width
        return name, _T()

    def _operator_new_size_at_ctor(self, bv, record):
        return self._new_size  # (size, addr) or None


def test_size_prefers_bn_type_width():
    rec = {"name": "net::Session", "methods": [], "vtable": None}
    out = read_class._object_size(_SizeCtx(type_width=0xD0), object(), rec)
    assert out == {"value": "0xd0", "source": "bn_type"}


def test_size_from_operator_new_when_no_type():
    rec = {"name": "net::Session", "methods": []}
    out = read_class._object_size(_SizeCtx(new_size=(0xD0, 0x443abc)), object(), rec)
    assert out["value"] == "0xd0" and out["source"] == "operator_new"
    assert out["at"] == "0x443abc"


def test_size_none_when_nothing_resolves():
    rec = {"name": "net::Session", "methods": []}
    assert read_class._object_size(_SizeCtx(), object(), rec) is None


def test_size_survives_find_type_raising():
    # Real ctx._find_type RAISES on a missing type; _object_size must treat that
    # as "no type" and fall through, not propagate the exception.
    class _RaisingCtx(_SizeCtx):
        def _find_type(self, bv, name):
            raise RuntimeError("Type not found: " + name)
    out = read_class._object_size(_RaisingCtx(new_size=(0x40, 0x1000)), object(), {"name": "X", "methods": []})
    assert out["value"] == "0x40" and out["source"] == "operator_new"


class _RttiCtx:
    """ctx reading little-endian words from a fake memory map and resolving a
    typeinfo address back to a class name."""
    def __init__(self, words, ti_names):
        self._words = words            # {addr: int}
        self._ti_names = ti_names      # {addr: class_name}

    def _pointer_size(self, bv):
        return 8

    def _read_pointer_value(self, bv, addr, *, size=None):
        return self._words.get(addr)

    def _read_u32(self, bv, addr):
        return self._words.get(addr)

    def _typeinfo_name_at(self, bv, addr):
        return self._ti_names.get(addr)


def test_rtti_si_single_base():
    # __si_class_type_info: [vptr][name-ptr][base-ti-ptr]; base at after_name(0x9110)
    words = {0x9100: 0xAB00, 0x9108: 0x9300, 0x9110: 0x9200}
    ctx = _RttiCtx(words, {0x9200: "net::Endpoint"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="si")
    assert bases == [{"name": "net::Endpoint", "address": "0x9200", "kind": "public"}]


def test_rtti_vmi_multiple_bases():
    # __vmi_class_type_info. after_name = 0x9100 + 2*8 = 0x9110.
    #   flags  (u32) @ 0x9110, base_count (u32) @ 0x9114, base array @ 0x9118.
    words = {
        0x9100: 0xAB10, 0x9108: 0x9300,
        0x9110: 0x0,            # flags
        0x9114: 2,              # base_count
        0x9118: 0x9200, 0x9120: (2 << 8) | 0x2,   # base0: ti ptr, off_flags (public)
        0x9128: 0x9240, 0x9130: (4 << 8) | 0x2,   # base1: ti ptr, off_flags
    }
    ctx = _RttiCtx(words, {0x9200: "net::A", 0x9240: "net::B"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="vmi")
    assert [b["name"] for b in bases] == ["net::A", "net::B"]
    # off_flags 0x2 has the public bit set -> "public".
    assert bases[0]["kind"] == "public"


def test_rtti_vmi_base_access_flags():
    # __offset_flags low bits: 0x1=virtual, 0x2=public. Decode independently so a
    # private base is never mislabeled "public".
    words = {
        0x9100: 0xAB10, 0x9108: 0x9300,
        0x9110: 0x0, 0x9114: 3,                   # base_count = 3
        0x9118: 0x9200, 0x9120: 0x0,              # private non-virtual
        0x9128: 0x9240, 0x9130: 0x1,              # private virtual
        0x9138: 0x9280, 0x9140: 0x3,              # public virtual
    }
    ctx = _RttiCtx(words, {0x9200: "Priv", 0x9240: "PrivV", 0x9280: "PubV"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="vmi")
    assert [(b["name"], b["kind"]) for b in bases] == [
        ("Priv", "private"),
        ("PrivV", "private virtual"),
        ("PubV", "public virtual"),
    ]


def test_rtti_no_base():
    ctx = _RttiCtx({0x9100: 0xAB20, 0x9108: 0x9300}, {})
    assert read_class._rtti_bases(ctx, object(), 0x9100, kind_hint="base") == []


def test_rtti_infers_si_without_hint():
    # No kind_hint -> structural inference: a resolvable base ptr at after_name -> si.
    words = {0x9100: 0xAB00, 0x9108: 0x9300, 0x9110: 0x9200}
    ctx = _RttiCtx(words, {0x9200: "net::Endpoint"})
    bases = read_class._rtti_bases(ctx, object(), 0x9100)
    assert [b["name"] for b in bases] == ["net::Endpoint"]


def test_rtti_infers_no_base_without_hint():
    # after_name word doesn't resolve to a typeinfo and no plausible count -> base.
    words = {0x9100: 0xAB20, 0x9108: 0x9300, 0x9110: 0x0, 0x9114: 0}
    ctx = _RttiCtx(words, {})
    assert read_class._rtti_bases(ctx, object(), 0x9100) == []


class _InstCtx:
    def __init__(self, ctor_sites, vtable_data_refs):
        self._ctor_sites = ctor_sites          # list of dicts
        self._vtable_data_refs = vtable_data_refs

    def _ctor_construction_sites(self, bv, record):
        return list(self._ctor_sites)

    def _global_vtable_stores(self, bv, record):
        return list(self._vtable_data_refs)


def test_instances_collects_construction_and_global_stores():
    sites = [
        {"address": "0x443abc", "function": "net::open", "kind": "new", "size": "0xd0"},
        {"address": "0x4500a0", "function": "main", "kind": "stack", "size": None},
    ]
    globals_ = [{"symbol": "g_session", "address": "0x4cabcd"}]
    ctx = _InstCtx(sites, globals_)
    rec = {"name": "net::Session", "vtable": {"address": "0x9000"}, "methods": []}
    out = read_class._instances(ctx, object(), rec)
    assert out["construction_sites"][0]["kind"] == "new"
    assert out["stored_globals"][0]["symbol"] == "g_session"


def test_class_show_assembles_full_record():
    bv = _make_registry_bv()

    class _ShowCtx:
        def _resolve_view(self, sel):
            return bv
        def _pointer_size(self, b):
            return 8
        def _vtable_layout_for(self, b, addr):
            return {"address": hex(addr), "slots": [
                {"index": 0, "address": "0x40e8b0", "method": {"name": "onData"},
                 "pure_virtual": False, "unnamed": False}]}
        def _object_size_for(self, b, rec):
            return {"value": "0xd0", "source": "operator_new", "at": "0x443abc"}
        def _bases_for(self, b, rec):
            return [{"name": "net::Endpoint", "address": "0x9200", "kind": "public"}]
        def _instances_for(self, b, rec):
            return {"construction_sites": [], "stored_globals": []}

    out = read_class._class_show(_ShowCtx(), None, "net::Session")
    assert out["name"] == "net::Session"
    assert out["size"]["value"] == "0xd0"
    assert out["bases"][0]["name"] == "net::Endpoint"
    assert out["vtable"]["slots"][0]["method"]["name"] == "onData"


def test_class_show_unknown_name_errors_with_hint():
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    import pytest
    with pytest.raises(read_class.OperationFailure) as exc:
        read_class._class_show(_Ctx(), None, "net::Nope")
    assert "class list" in str(exc.value).lower()


def test_class_show_unknown_name_suggests_close_matches():
    # #413: a near-miss should point at the real class name, not just "run class list".
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    import pytest
    with pytest.raises(read_class.OperationFailure) as exc:
        read_class._class_show(_Ctx(), None, "net::Sesion")  # typo of net::Session
    msg = str(exc.value)
    assert "did you mean" in msg.lower()
    assert "net::Session" in msg


def test_class_show_ambiguous_returns_all_matches():
    # Two classes share a leaf name across namespaces.
    fns = [
        _Fn(0x1000, "_ZN1a3FooC1Ev", "a::Foo::Foo()"),
        _Fn(0x2000, "_ZN1b3FooC1Ev", "b::Foo::Foo()"),
    ]
    bv = _RegistryBV(fns, [])

    class _Ctx:
        def _resolve_view(self, sel):
            return bv
        def _vtable_layout_for(self, b, a):
            return None
        def _object_size_for(self, b, r):
            return None
        def _bases_for(self, b, r):
            return []
        def _instances_for(self, b, r):
            return {"construction_sites": [], "stored_globals": []}

    out = read_class._class_show(_Ctx(), None, "Foo")
    assert out["ambiguous"] is True
    assert {m["name"] for m in out["matches"]} == {"a::Foo", "b::Foo"}


def test_render_class_show_text_matches_mock_shape():
    from bn.formatters import _render_class_show_text
    rec = {
        "name": "net::Session", "confidence": "rtti",
        "size": {"value": "0xd0", "source": "operator_new"},
        "vtable": {"address": "0x4e0000", "slots": [
            {"index": 0, "address": "0x40e8b0", "method": {"name": "onData"}, "pure_virtual": False, "unnamed": False},
            {"index": 1, "address": "0x0", "method": None, "pure_virtual": True, "unnamed": False}]},
        "bases": [{"name": "net::Endpoint"}],
        "methods": [{"kind": "ctor", "address": "0x40abc0", "demangled": "net::Session::Session()"}],
        "instances": {"construction_sites": [{"kind": "new", "address": "0x443abc", "size": "0xd0"}],
                      "stored_globals": [{"symbol": "g_session", "address": "0x4cabcd"}]},
    }
    text = _render_class_show_text(rec)
    assert "class net::Session" in text and "size 0xd0" in text and "base: net::Endpoint" in text
    assert "vtable [0] 0x40e8b0  onData" in text
    assert "vtable [1] 0x0  __cxa_pure_virtual" in text
    assert "instances: new @ 0x443abc (size 0xd0) ; stored -> g_session @ 0x4cabcd" in text


def test_render_class_show_lists_non_virtual_methods_and_empty_vtable_note():
    # A class whose vtable symbol exists but resolves no slots (PIE/relocated)
    # and whose members are all non-virtual: the renderer must list the members
    # and explain the empty vtable rather than render a near-blank class.
    from bn.formatters import _render_class_show_text
    rec = {
        "name": "Controller", "confidence": "rtti", "size": None,
        "vtable": {"address": "0x4b6270", "slots": []},
        "bases": [], "instances": {"construction_sites": [], "stored_globals": []},
        "methods": [
            {"kind": "method", "address": "0x401000", "demangled": "Controller::setAudioFocus(int)"},
            {"kind": "method", "address": "0x401100", "demangled": "Controller::sendPingRequest()"},
        ],
    }
    text = _render_class_show_text(rec)
    assert "methods (2):" in text
    assert "0x401000  Controller::setAudioFocus(int)" in text
    assert "vtable: symbol present but no slots resolved here" in text


def test_render_class_show_no_vtable_note_when_no_vtable():
    from bn.formatters import _render_class_show_text
    rec = {"name": "GalMutex", "confidence": "ctor", "size": None,
           "vtable": None, "bases": [], "methods": [],
           "instances": {"construction_sites": [], "stored_globals": []}}
    assert "no slots resolved" not in _render_class_show_text(rec)


def test_render_construction_site_includes_function():
    from bn.formatters import _render_class_show_text
    rec = {"name": "X", "confidence": "ctor", "size": None, "vtable": None,
           "bases": [], "methods": [],
           "instances": {"construction_sites": [
               {"address": "0x443abc", "function": "AapGalifStart",
                "kind": "ctor-call", "size": None}],
               "stored_globals": []}}
    assert "ctor-call @ 0x443abc (in AapGalifStart)" in _render_class_show_text(rec)


def test_class_list_artifact_helpers_309():
    assert read_class._is_construction_vtable_artifact("C{for `A'}")
    assert read_class._is_construction_vtable_artifact("media::codec::JsonCodec{for `media::codec::Codec'}")
    assert not read_class._is_construction_vtable_artifact("media::codec::JsonCodec")
    assert read_class._is_thunk_artifact("non-virtual_thunk_to_C")
    assert read_class._is_thunk_artifact("virtual thunk to X::f()")
    assert not read_class._is_thunk_artifact("net::Session")
    # anchored to the prefix: a real class/namespace that merely CONTAINS
    # "thunk_to" must NOT be suppressed (#309 review -- substring match dropped them)
    assert not read_class._is_thunk_artifact("Thunk_to_handler")
    assert not read_class._is_thunk_artifact("thunk_to_ns::X")
    assert not read_class._is_thunk_artifact("do_thunk_to_x")
    assert not read_class._is_thunk_artifact("ThunkTo")
    assert read_class._is_vendor_class("boost::any")
    assert read_class._is_vendor_class("boost::asio::ip::tcp")
    assert not read_class._is_vendor_class("net::Session")
    assert not read_class._is_vendor_class("boostrap::Thing")   # not a `boost` component


def test_class_list_suppresses_construction_vtable_thunk_and_vendor_309(monkeypatch):
    # #309: construction-vtable artifacts hide by default (revealed by --all);
    # thunks are NEVER surfaced as classes (even --all); --no-vendor folds boost.
    def _rec(name, confidence="rtti", methods=0):
        return {"name": name, "methods": [{} for _ in range(methods)], "vtable": {"a": 1},
                "typeinfo": None, "typeinfo_name": None, "size": None, "bases": [],
                "instances": [], "confidence": confidence}
    registry = {n: _rec(n) for n in ("A", "C", "C{for `A'}", "boost::any")}
    registry["non-virtual_thunk_to_C"] = _rec("non-virtual_thunk_to_C", confidence="name-only", methods=3)
    monkeypatch.setattr(read_class, "_build_class_registry", lambda ctx, bv, query=None: registry)

    class _Ctx:
        def _resolve_view(self, sel):
            return object()
        def _bases_for(self, bv, rec):
            return []

    d = read_class._class_list(_Ctx(), None)
    names = [c["name"] for c in d["items"]]
    assert "C{for `A'}" not in names                 # construction vtable hidden by default
    assert "non-virtual_thunk_to_C" not in names     # thunk hidden
    assert {"A", "C", "boost::any"} <= set(names)
    assert d["construction_vtables_suppressed"] == 1
    assert d["thunks_suppressed"] == 1

    a = read_class._class_list(_Ctx(), None, include_all=True)
    an = [c["name"] for c in a["items"]]
    assert "C{for `A'}" in an                         # --all reveals construction vtables
    assert "non-virtual_thunk_to_C" not in an         # thunks NEVER surfaced
    assert a["thunks_suppressed"] == 1

    v = read_class._class_list(_Ctx(), None, no_vendor=True)
    assert "boost::any" not in [c["name"] for c in v["items"]]
    assert v["vendor_suppressed"] == 1


def test_render_class_list_text_shows_suppression_counts_309():
    from bn.formatters import _render_class_list_text
    value = {
        "kind": "classes",
        "items": [{"name": "A", "method_count": 5, "has_vtable": True,
                   "size": None, "bases": [], "confidence": "rtti"}],
        "total": 1, "no_stl": True, "no_vendor": True,
        "construction_vtables_suppressed": 2, "thunks_suppressed": 1,
        "library_suppressed": 3, "vendor_suppressed": 1,
    }
    out = _render_class_list_text(value)
    assert "2 construction-vtable artifacts (--all to show)" in out
    assert "1 thunk" in out
    assert "3 library/STL" in out
    assert "1 vendored" in out


def test_class_list_count_only_484():
    # #484: --count returns just the class count (respecting filters), plus the
    # non-class artifact_count, without building/paging rows.
    bv = _make_registry_bv()

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    out = read_class._class_list(_Ctx(), None, include_all=True, count_only=True)
    assert out["kind"] == "classes"
    assert "items" not in out and "count" in out
    full = read_class._class_list(_Ctx(), None, include_all=True)
    assert out["count"] == full["total"]                 # count matches the full listing
    assert out["total"] == full["total"]
    assert "artifact_count" in out and out["artifact_count"] >= 0


def test_class_list_labels_non_class_rtti_artifacts_481():
    # #481: an RTTI-confidence row with no methods and no vtable (typeinfo emitted
    # for a NON-object type -- a function signature / fundamental type) is tagged
    # `artifact`; a real class (vtable and/or clustered methods) is not.
    fns = [_Fn(0x1000, "_ZN3net7SessionC1Ev", "net::Session::Session()")]  # real class
    syms = [
        _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x9000),
        _Sym("_ZTIN3net7SessionE", "typeinfo for net::Session", 0x9100),
        # typeinfo for a function type -> a pseudo-class with no vtable, no methods.
        _Sym("_ZTIFvhE", "typeinfo for void (unsigned char)", 0x9300),
    ]
    bv = _RegistryBV(fns, syms)

    class _Ctx:
        def _resolve_view(self, sel):
            return bv

    rows = {r["name"]: r for r in read_class._class_list(_Ctx(), None, include_all=True)["items"]}
    assert rows["net::Session"]["artifact"] is False           # real class (ctor + vtable)
    artifacts = [r for r in rows.values() if r["artifact"]]
    assert artifacts                                            # the function-type typeinfo row
    for r in artifacts:
        assert r["has_vtable"] is False and r["method_count"] == 0 and r["confidence"] == "rtti"


def test_is_type_expression_name_481():
    # #481: distinguish a non-class TYPE-EXPRESSION typeinfo from a real class name,
    # incl. the anonymous-namespace / template cases that key-on-missing-vtable got
    # wrong (they tagged real classes as artifacts).
    f = read_class._is_type_expression_name
    # non-class type expressions -> artifact
    assert f("void (unsigned char)")          # function type
    assert f("char const*")                    # pointer
    assert f("Foo&")                           # reference
    assert f("int [4]")                        # array
    assert f("unsigned int")                   # multi-token fundamental
    assert f("int")                            # bare fundamental
    # real classes -> NOT artifacts (the libstdc++ false-positive class)
    assert not f("(anonymous namespace)::future_error_category")
    assert not f("net::Session")
    assert not f("std::vector<int>")           # template args stripped
    assert not f("std::__cxx11::basic_string<char, std::char_traits<char> >")
    assert not f("Outer<Foo, Bar>::Inner")     # template with comma inside <>


def test_rtti_symbol_maps_prefers_local_definition_over_alias_529():
    # #529: a GOT/import/external alias carries the SAME demangled RTTI name as the
    # real local vtable definition. When get_symbols() yields the alias first, the
    # (class, kind) slot must still keep the LOCAL definition -- otherwise class
    # recovery decodes the extern/GOT stub as the vtable object and reports
    # missing/empty slots despite a real local vtable.
    local = _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x9000,
                 sym_type="SymbolType.DataSymbol")
    alias = _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x41000,
                 sym_type="SymbolType.ExternalSymbol")

    # alias first in iteration order -- the local must still win.
    m1 = read_class._rtti_symbol_maps(_RegistryBV([], [alias, local]))
    assert m1["net::Session"]["vtable"] is local
    # local first -- an alias must never overwrite it.
    m2 = read_class._rtti_symbol_maps(_RegistryBV([], [local, alias]))
    assert m2["net::Session"]["vtable"] is local
    # import-address alias variant, alias first -> local still wins.
    imp = _Sym("_ZTVN3net7SessionE", "vtable for net::Session", 0x42000,
               sym_type="SymbolType.ImportAddressSymbol")
    m3 = read_class._rtti_symbol_maps(_RegistryBV([], [imp, local]))
    assert m3["net::Session"]["vtable"] is local
    # single symbol -- behavior unchanged (kept regardless of type).
    m4 = read_class._rtti_symbol_maps(_RegistryBV([], [alias]))
    assert m4["net::Session"]["vtable"] is alias


class _EnumType:
    """Mimics a real BN ``SymbolType`` IntEnum: ``.name`` is the member name but
    ``str()`` renders as the integer value -- so a str(sym.type) substring match
    would NEVER fire on a live BV. Guards against the regression."""
    def __init__(self, name, value):
        self.name = name
        self._value = value
    def __str__(self):
        return str(self._value)


def test_is_alias_symbol_uses_enum_name_not_str_529():
    # #529 regression guard: real BN SymbolType.__str__ is the integer, so the
    # classification must read .name. A str()-based match would misclassify BOTH.
    ext = types.SimpleNamespace(type=_EnumType("ExternalSymbol", 5))
    dat = types.SimpleNamespace(type=_EnumType("DataSymbol", 3))
    assert read_class._is_alias_symbol(ext) is True     # alias detected via .name
    assert read_class._is_alias_symbol(dat) is False    # local not misclassified
    # local BN types whose names must NOT contain an alias substring.
    for local in ("FunctionSymbol", "LibraryFunctionSymbol", "LocalLabelSymbol",
                  "SymbolicFunctionSymbol"):
        assert read_class._is_alias_symbol(
            types.SimpleNamespace(type=_EnumType(local, 0))) is False
    # and the live enum-name preference wins the slot over an alias.
    local_v = types.SimpleNamespace(name="_ZTVN3net7SessionE", address=0x9000,
                                    type=_EnumType("DataSymbol", 3))
    alias_v = types.SimpleNamespace(name="_ZTVN3net7SessionE", address=0x41000,
                                    type=_EnumType("ExternalSymbol", 5))
    def _kc(sym):
        return ("vtable", "net::Session")
    orig = read_class._rtti_kind_and_class
    read_class._rtti_kind_and_class = _kc
    try:
        m = read_class._rtti_symbol_maps(_RegistryBV([], [alias_v, local_v]))
        assert m["net::Session"]["vtable"] is local_v
    finally:
        read_class._rtti_kind_and_class = orig
