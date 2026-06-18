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
    def __init__(self, raw_name, short_name, address):
        self.raw_name = raw_name
        self.short_name = short_name
        self.name = raw_name
        self.address = address


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
