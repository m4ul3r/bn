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


def _addrs(entry):
    return {c["address"] for c in entry.get("callsites", [])}


def test_present_callsites_aggregates_across_symbol_spellings():
    # #472: two symbol spellings mapping to one model key must AGGREGATE their
    # callsites, not clobber -- the old assignment let whichever spelling was seen
    # last (set iteration order) win, dropping a present sink to (0 callsites).
    res = rts._taint_models_op(_CtxWithBV(_BVSpellings()), "active",
                               {"present": True, "callsites": True})
    mc = [e for lst in res["sinks_by_class"].values() for e in lst if e["symbol"] == "memcpy"]
    assert len(mc) == 1
    assert mc[0]["callsite_count"] == 3                   # 2 + 1 aggregated, not clobbered
    assert _addrs(mc[0]) == {"0x400", "0x404", "0x408"}


def test_present_callsites_dedups_aliased_addresses():
    # An alias spelling resolving to the SAME site must not double-count.
    class _BVDup(_BVSpellings):
        def __init__(self):
            super().__init__()
            self._refs = {0x1000: [_Ref(0x400)], 0x2000: [_Ref(0x400)]}  # same site twice
    res = rts._taint_models_op(_CtxWithBV(_BVDup()), "active",
                               {"present": True, "callsites": True})
    mc = [e for lst in res["sinks_by_class"].values() for e in lst if e["symbol"] == "memcpy"]
    assert mc[0]["callsite_count"] == 1
    assert _addrs(mc[0]) == {"0x400"}


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
    # #603: pread must never silently drop out of the builtin source catalog
    # (bn taint models) alongside read/recv/recvfrom.
    assert "pread" in src_syms, "pread must be a modeled source"


def test_recv_overflow_comment_names_pread_603():
    # #603-2: the recv_overflow opt-in sink family doc comment must literally
    # name pread alongside read/recv/recvfrom, not just leave it modeled with
    # no discoverable mention in the explanatory text a reader greps first.
    import json
    from bn_agent_bridge.taint_models import _BUILTIN_MODELS

    raw = json.loads(_BUILTIN_MODELS.read_text())
    comment = raw["models"]["_comment_recv_overflow"]
    assert "pread" in comment, comment


# --- #555: catalog entries marked as NON-findings --------------------------

def test_build_catalog_marks_non_findings_555():
    cat = build_catalog(_MODELS)
    # Top-level: loud, machine-readable "this is a catalog, not findings".
    assert cat["presence_catalog"] is True
    assert cat["is_finding"] is False
    assert "NOT taint findings" in cat["catalog_note"]
    # Every sink entry is a non-finding and carries conditional wording.
    memcpy = cat["sinks_by_class"]["overflow_len"][0]
    assert memcpy["is_finding"] is False
    assert memcpy["model_name"] == "memcpy"
    # Keeps the "... IF argument N is tainted" framing so a constant arg isn't a bug.
    assert "IF argument 2 is tainted" in memcpy["model_description"]


def test_build_catalog_unconditional_sink_description_555():
    # A sink with an empty tainted_args list (e.g. gets) is still a catalog entry,
    # not a finding, and its description says so without asserting a vuln.
    models = {"gets": {"sink": {"tainted_args": [], "class": "unbounded_input",
                                "detail": "always unsafe"}}}
    entry = build_catalog(models, role="sink")["sinks_by_class"]["unbounded_input"][0]
    assert entry["is_finding"] is False
    assert "not a finding" in entry["model_description"].lower()


def test_build_catalog_multi_arg_description_555():
    models = {"calloc": {"sink": {"tainted_args": [0, 1], "class": "alloc_size",
                                  "detail": "size"}}}
    entry = build_catalog(models, role="sink")["sinks_by_class"]["alloc_size"][0]
    assert "arguments 0 or 1 are tainted" in entry["model_description"]


# --- #553: containing function + context per callsite ----------------------

class _FnFull:
    def __init__(self, name, start, is_thunk=False):
        self.name = name
        self.start = start
        self.is_thunk = is_thunk


class _BVTriage:
    """Present sink ``system`` with one real application caller (parse_record) and
    one import-thunk site (the ``system`` PLT veneer). Exercises #553 (function per
    callsite) and #560 (thunk labeling / audit count)."""
    def __init__(self):
        self._app = _FnFull("parse_record", 0x5000)
        self._thunk = _FnFull("system", 0x1000, is_thunk=True)
        self.functions = [self._app, self._thunk]
        self._syms = {"system": [_Sym("system", 0x1000)]}
        self._refs = {0x1000: [_Ref(0x5010), _Ref(0x2000)]}
        self._contain = {0x5010: self._app, 0x2000: self._thunk}

    def get_symbols(self):
        return []

    def get_symbols_by_name(self, n):
        return self._syms.get(n, [])

    def get_code_refs(self, a):
        return self._refs.get(a, [])

    def get_functions_containing(self, a):
        f = self._contain.get(a)
        return [f] if f else []

    def get_function_at(self, a):
        return next((f for f in self.functions if f.start == a), None)


def _sink_entry(res, symbol):
    return next(e for lst in res["sinks_by_class"].values() for e in lst
               if e["symbol"] == symbol)


def test_present_callsites_include_function_553():
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True})
    system = _sink_entry(res, "system")
    rows = {c["address"]: c for c in system["callsites"]}
    assert rows["0x5010"]["function"] == "parse_record"
    assert rows["0x5010"]["kind"] == "app_caller"


# --- #560: label import-thunk / self-stub callsites, expose audit count -----

def test_present_callsites_label_import_thunk_560():
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True})
    system = _sink_entry(res, "system")
    rows = {c["address"]: c for c in system["callsites"]}
    assert rows["0x2000"]["kind"] == "import_thunk"
    # Raw count includes the thunk; the audit count excludes it (the real queue).
    assert system["callsite_count"] == 2
    assert system["audit_callsite_count"] == 1


def test_present_self_stub_labeled_non_audit_560():
    # A code ref located inside the modeled symbol's OWN body (a self-tailcall
    # stub) is non-audit, distinct from an import thunk.
    class _BVSelf:
        def __init__(self):
            self._body = _FnFull("memcpy", 0x1000)          # not is_thunk
            self.functions = [self._body]
            self._syms = {"memcpy": [_Sym("memcpy", 0x1000)]}
            self._refs = {0x1000: [_Ref(0x1004)]}           # self-reference
            self._contain = {0x1004: self._body}

        def get_symbols(self):
            return []

        def get_symbols_by_name(self, n):
            return self._syms.get(n, [])

        def get_code_refs(self, a):
            return self._refs.get(a, [])

        def get_functions_containing(self, a):
            f = self._contain.get(a)
            return [f] if f else []

        def get_function_at(self, a):
            return self._body if a == 0x1000 else None

    res = rts._taint_models_op(_CtxWithBV(_BVSelf()), "active",
                               {"present": True, "callsites": True})
    mc = _sink_entry(res, "memcpy")
    assert mc["callsites"][0]["kind"] == "self_stub"
    assert mc["audit_callsite_count"] == 0


# --- #556: portable / stable identifiers in model output -------------------

def test_present_exposes_portable_identifiers_556():
    res = rts._taint_models_op(_CtxWithBV(_BVSpellings()), "active",
                               {"present": True, "callsites": True})
    mc = _sink_entry(res, "memcpy")
    # model_name = normalized alias taint commands accept; raw/resolved = the
    # imported spelling xrefs/callsites need; accepted_aliases lists all spellings.
    assert mc["model_name"] == "memcpy"
    assert mc["resolved_symbol"] == "memcpy"              # exact key preferred over @plt
    assert mc["raw_symbol"] == "memcpy"
    assert set(mc["accepted_aliases"]) == {"memcpy", "memcpy@plt"}


def test_catalog_only_has_model_name_but_no_raw_symbol_556():
    # Without a target there is no binary spelling to resolve; model_name is still
    # present so a consumer always has the portable alias.
    res = rts._taint_models_op(_CtxNoView(), None, {})
    sink = next(e for e in res["items"] if e["role"] == "sink")
    assert sink["model_name"] == sink["symbol"]
    assert "raw_symbol" not in sink


# --- text rendering ---------------------------------------------------------

def test_render_taint_models_text_non_finding_banner_and_rows():
    from bn.formatters import _render_taint_models_text
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True})
    text = _render_taint_models_text(res)
    assert "NOT taint findings" in text
    assert "parse_record" in text                         # #553 function context
    assert "[import_thunk]" in text                       # #560 non-audit label
    assert "2 callsites, 1 application" in text           # raw vs audit count


def test_build_catalog_surfaces_bounded_write_sink_443():
    # #443: a bounded-write sink declares len_arg/buf_arg; the catalog surfaces them.
    models = {"app_recv": {"sink": {"class": "overflow_len", "len_arg": 1, "buf_arg": 2,
                                    "detail": "wrapped recv"}}}
    cat = build_catalog(models, sink_class="overflow_len")
    entry = cat["sinks_by_class"]["overflow_len"][0]
    assert entry["symbol"] == "app_recv"
    assert entry["len_arg"] == 1 and entry["buf_arg"] == 2


def test_validate_bounded_write_sink_schema_443():
    # #443: len_arg/buf_arg are validated as integer arg indices; a sink must be armed
    # by tainted_args OR len_arg.
    from bn_agent_bridge.taint_engine import _coerce_model_map, TaintError
    ok = {"app_recv": {"sink": {"class": "overflow_len", "len_arg": 1, "buf_arg": 2}}}
    _coerce_model_map(ok, source="test")  # no raise
    # len_arg without buf_arg is valid (armed sink, no bounded downgrade).
    _coerce_model_map({"g": {"sink": {"class": "overflow_len", "len_arg": 2}}}, source="test")
    for bad in (
        {"f": {"sink": {"class": "overflow_len", "len_arg": "x"}}},        # len_arg not int
        {"f": {"sink": {"class": "overflow_len", "len_arg": 1, "buf_arg": "y"}}},  # buf_arg not int
        {"f": {"sink": {"class": "overflow_len"}}},                        # nothing arms it
        {"f": {"sink": {"class": "overflow_len", "len_arg": -1}}},         # negative (audit D1)
        {"f": {"sink": {"class": "overflow_len", "len_arg": 1, "buf_arg": -2}}},  # negative buf_arg
    ):
        with pytest.raises(TaintError):
            _coerce_model_map(bad, source="test")
