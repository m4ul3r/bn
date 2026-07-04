from __future__ import annotations

import pytest

from bn_agent_bridge.read_taint_models import build_catalog

_MODELS = {
    "_comment": "ignored",
    "recv": {"sources": [{"to": "*arg:1"}, {"to": "ret"}]},
    "memcpy": {"propagates": [{"from": "*arg:1", "to": "*arg:0"}],
               "sink": {"tainted_args": [2], "class": "overflow_len", "detail": "len"}},
    "system": {"sink": {"tainted_args": [0], "class": "command_injection", "detail": "cmd"}},
    "strlen": {"propagates": [{"from": "*arg:0", "to": "ret"}]},
}


def test_build_catalog_groups_by_role_and_class():
    cat = build_catalog(_MODELS)
    assert {s["symbol"] for s in cat["sources"]} == {"recv"}
    assert set(cat["sinks_by_class"]) == {"overflow_len", "command_injection"}
    assert {p["symbol"] for p in cat["propagators"]} == {"memcpy", "strlen"}
    assert "_comment" not in {s["symbol"] for s in cat["sources"]}


def test_build_catalog_role_filter():
    cat = build_catalog(_MODELS, role="sink")
    assert cat["sources"] == [] and cat["propagators"] == []
    assert set(cat["sinks_by_class"]) == {"overflow_len", "command_injection"}


def test_build_catalog_class_filter_implies_sink():
    cat = build_catalog(_MODELS, sink_class="overflow_len")
    assert set(cat["sinks_by_class"]) == {"overflow_len"}
    assert cat["sources"] == [] and cat["propagators"] == []


# --- op handler (catalog + binary-present) -----------------------------------

from bn_agent_bridge import read_taint_slice as rts


class _CtxNoView:
    def _resolve_view(self, sel):  # pragma: no cover - not hit for catalog-only
        raise AssertionError("should not resolve a view for catalog-only dump")


def test_taint_models_op_catalog_only():
    res = rts._taint_models_op(_CtxNoView(), None, {})
    assert "sinks_by_class" in res and "overlays" in res and "items" in res
    assert res["sinks_by_class"]                          # builtin DB has real sinks


def test_taint_models_op_present_without_target_errors():
    with pytest.raises(rts.OperationFailure):
        rts._taint_models_op(_CtxNoView(), None, {"present": True})


class _FakeFn:
    def __init__(self, name):
        self.name = name


class _FakeBV:
    def __init__(self, fn_names):
        self.functions = [_FakeFn(n) for n in fn_names]

    def get_symbols(self):
        return []

    def get_symbols_by_name(self, n):
        return []

    def get_code_refs(self, a):
        return []


class _CtxWithBV:
    def __init__(self, bv):
        self._bv = bv

    def _resolve_view(self, sel):
        return self._bv


def test_taint_models_op_present_intersects_binary():
    bv = _FakeBV(["memcpy", "helper_fn", "system"])
    res = rts._taint_models_op(_CtxWithBV(bv), "active", {"present": True})
    syms = {e["symbol"] for lst in res["sinks_by_class"].values() for e in lst}
    assert "memcpy" in syms and "system" in syms          # modeled + present
    assert "strcpy" not in syms                            # modeled but absent -> filtered
    for lst in res["sinks_by_class"].values():
        for e in lst:
            assert e["present"] is True


class _Ref:
    def __init__(self, a):
        self.address = a


class _Sym:
    def __init__(self, name, a):
        self.name = name
        self.address = a


class _BVSpellings:
    """memcpy and memcpy@plt both normalize to the model key 'memcpy', with
    distinct call sites, exercising the #472 aggregation path."""
    def __init__(self):
        self.functions = [_FakeFn("memcpy"), _FakeFn("memcpy@plt")]
        self._syms = {"memcpy": [_Sym("memcpy", 0x1000)],
                      "memcpy@plt": [_Sym("memcpy@plt", 0x2000)]}
        self._refs = {0x1000: [_Ref(0x400), _Ref(0x404)], 0x2000: [_Ref(0x408)]}

    def get_symbols(self):
        return []

    def get_symbols_by_name(self, n):
        return self._syms.get(n, [])

    def get_code_refs(self, a):
        return self._refs.get(a, [])


def test_present_callsites_aggregates_across_symbol_spellings():
    # #472: two symbol spellings mapping to one model key must AGGREGATE their
    # callsites, not clobber -- the old assignment let whichever spelling was seen
    # last (set iteration order) win, dropping a present sink to (0 callsites).
    res = rts._taint_models_op(_CtxWithBV(_BVSpellings()), "active",
                               {"present": True, "callsites": True})
    mc = [e for lst in res["sinks_by_class"].values() for e in lst if e["symbol"] == "memcpy"]
    assert len(mc) == 1
    assert mc[0]["callsites"] == 3                        # 2 + 1 aggregated, not clobbered
    assert set(mc[0]["addresses"]) == {"0x400", "0x404", "0x408"}


def test_present_callsites_dedups_aliased_addresses():
    # An alias spelling resolving to the SAME site must not double-count.
    class _BVDup(_BVSpellings):
        def __init__(self):
            super().__init__()
            self._refs = {0x1000: [_Ref(0x400)], 0x2000: [_Ref(0x400)]}  # same site twice
    res = rts._taint_models_op(_CtxWithBV(_BVDup()), "active",
                               {"present": True, "callsites": True})
    mc = [e for lst in res["sinks_by_class"].values() for e in lst if e["symbol"] == "memcpy"]
    assert mc[0]["callsites"] == 1
    assert mc[0]["addresses"] == ["0x400"]


def test_builtin_catalog_covers_fortify_and_exec_sinks():
    # #372 guard, relocated from the retired sink-sweep SINK_RE to the single
    # source of truth: the model DB must flag the FORTIFY (*_chk) family and bare
    # execv as sinks, so sink enumeration never silently drops them. Widened to
    # guard every dangerous-copy/exec family the retired SINK_RE matched (bcopy,
    # mempcpy, strlcpy, strlcat, execvp, dlopen) -- these were unmodeled, so the
    # model-DB-bounded `bn taint models` enumeration used to omit them.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    cat = build_catalog(models)
    sink_syms = {e["symbol"] for lst in cat["sinks_by_class"].values() for e in lst}
    for name in ("sprintf_chk", "snprintf_chk", "execv",
                 "bcopy", "mempcpy", "strlcpy", "strlcat", "execvp", "dlopen"):
        assert name in sink_syms, f"{name} must be a modeled sink"
    # fscanf is an input SOURCE (like scanf), not a sink; guard it in its own role
    # so retiring the name-regex net does not silently drop it from enumeration.
    src_syms = {s["symbol"] for s in build_catalog(models, role="source")["sources"]}
    assert "fscanf" in src_syms, "fscanf must be a modeled source"
