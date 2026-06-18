from __future__ import annotations

import importlib

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
        return {"entries": self._rows}


def test_vtable_layout_skips_when_header_not_typeinfo():
    # The Itanium gate: if word[1] doesn't resolve to a typeinfo, this isn't a
    # real local vtable object (import/GOT slot or relocated-to-zero PIE slot) --
    # return no slots instead of decoding adjacent GOT/data as fake slots.
    rows = [{"index": 0, "value": "0x1000", "readable": True,
             "target": {"status": "function", "function": {"name": "f0"}}}]
    layout = read_class._vtable_layout(_SlotCtx(rows, ti_ok=False), object(), 0x9000)
    assert layout["slots"] == []


def test_vtable_layout_stops_at_mapped_data_slot():
    # A mapped pointer into a NON-executable segment (GOT/.data read past the
    # table end, or an import slot misread) is not a slot -- it ends the scan,
    # rather than rendering adjacent data as fake slots (#205 review).
    rows = [
        {"index": 0, "value": "0x1000", "readable": True,
         "target": {"status": "function", "function": {"name": "f0"}}},
        {"index": 1, "value": "0x9999", "readable": True,
         "target": {"status": "mapped", "function": None,
                    "context": {"segment": {"executable": False}}}},
        {"index": 2, "value": "0x2000", "readable": True,
         "target": {"status": "function", "function": {"name": "f2"}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), object(), 0x9000)
    assert [s["index"] for s in layout["slots"]] == [0]


def test_vtable_layout_includes_executable_mapped_slot():
    # A mapped pointer into an EXECUTABLE segment is code BN hasn't analyzed into
    # a function yet -- still a valid (unnamed) vtable slot.
    rows = [
        {"index": 0, "value": "0x1000", "readable": True,
         "target": {"status": "mapped", "function": None,
                    "context": {"segment": {"executable": True}}}},
    ]
    layout = read_class._vtable_layout(_SlotCtx(rows), object(), 0x9000)
    assert len(layout["slots"]) == 1 and layout["slots"][0]["unnamed"] is True


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
    names = [c["name"] for c in out["classes"]]
    assert "net::Session" in names and out["total"] == len(names)
    # Default (no --all) drops name-only clusters like the bare "net" namespace.
    confirmed = read_class._class_list(_Ctx(), None, include_all=False)
    assert "net" not in [c["name"] for c in confirmed["classes"]]
    assert "net::Session" in [c["name"] for c in confirmed["classes"]]


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
