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


def test_build_catalog_includes_underscore_prefixed_real_models_849():
    # #849: build_catalog must NOT skip models whose names start with a single
    # `_` but are NOT doc-key prefixed (_comment). Real model names like
    # ``__isoc99_scanf`` / ``__isoc99_fscanf`` / ``__isoc99_vsscanf`` start
    # with ``_`` but are genuine sources/propagators -- the engine resolves them
    # on real binaries while the catalog used to omit them entirely. The fix
    # matches only the doc-key prefix (``_comment``), not every leading ``_``.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    src_syms = {s["symbol"] for s in build_catalog(models, role="source")["sources"]}
    for name in ("__isoc99_scanf", "__isoc99_fscanf"):
        assert name in src_syms, (
            f"{name} must appear in the source catalog (starts with '_' but is "
            "a real model, not a _comment doc key)"
        )
    prop_syms = {p["symbol"] for p in build_catalog(models, role="propagator")["propagators"]}
    assert "__isoc99_vsscanf" in prop_syms, (
        "__isoc99_vsscanf must appear as a propagator in the catalog"
    )


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


class _BVDisasm(_BVTriage):
    def get_disassembly(self, a):
        return {0x5010: "call    system", 0x2000: "jmp     qword [rip+0x2f1a]"}.get(a, "")


def test_present_callsites_carry_one_line_of_context_794():
    # #794: the row answered {address, function, kind}, which says WHERE a
    # modeled sink is called but nothing about WHAT the call looks like, so
    # triaging a callsite queue cost a `bn disasm` round-trip per row. The key
    # is `disasm`, matching the sibling address-row emitters that actually use
    # it -- `read_xrefs` on its ref rows and `seam` on its call-context row --
    # rather than inventing a second spelling for one field. (`read_evidence` is
    # NOT one of them: `il_format._disasm_entry` returns {address, text} under a
    # different key. An earlier version of this comment cited it and was wrong.)
    res = rts._taint_models_op(_CtxWithBV(_BVDisasm()), "active",
                               {"present": True, "callsites": True})
    rows = {c["address"]: c for c in _sink_entry(res, "system")["callsites"]}
    assert rows["0x5010"]["disasm"] == "call    system"
    assert rows["0x2000"]["disasm"] == "jmp     qword [rip+0x2f1a]"
    # The pre-existing fields are untouched -- this is additive.
    assert rows["0x5010"]["function"] == "parse_record"
    assert rows["0x5010"]["kind"] == "app_caller"


def test_present_callsite_context_reaches_catalog_text_794():
    """The text catalog is the default read surface. A row that has disasm on
    the wire must print it beside the address and function; the existing JSON
    test alone cannot detect the renderer silently dropping the new column."""
    from bn.formatters import _render_taint_models_text

    res = rts._taint_models_op(_CtxWithBV(_BVDisasm()), "active",
                               {"present": True, "callsites": True})
    text = _render_taint_models_text(res)
    assert "0x5010  parse_record  call    system" in text
    assert "0x2000  system [import_thunk]  jmp     qword [rip+0x2f1a]" in text


def test_present_callsites_degrade_when_the_view_cannot_disassemble_794():
    # A BN shape with no `get_disassembly` (and a read that raises) must still
    # produce the row -- an unavailable context is an empty string, never a
    # failed listing. `_BVTriage` has no such method at all.
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True})
    rows = {c["address"]: c for c in _sink_entry(res, "system")["callsites"]}
    assert rows["0x5010"]["disasm"] == ""
    assert rows["0x5010"]["function"] == "parse_record"


def test_present_callsites_degrade_when_disassembly_raises_794():
    """A failed read at one address must not discard the catalog or its other
    callsites. A missing get_disassembly method exercises a different guard."""
    class _BVFailingDisasm(_BVDisasm):
        def get_disassembly(self, a):
            if a == 0x5010:
                raise RuntimeError("unmapped address")
            return super().get_disassembly(a)

    res = rts._taint_models_op(_CtxWithBV(_BVFailingDisasm()), "active",
                               {"present": True, "callsites": True})
    system = _sink_entry(res, "system")
    rows = {c["address"]: c for c in system["callsites"]}
    assert system["callsite_count"] == 2
    assert rows["0x5010"]["disasm"] == ""
    assert rows["0x5010"]["function"] == "parse_record"
    assert rows["0x2000"]["disasm"] == "jmp     qword [rip+0x2f1a]"


class _BVPlain:
    """The minimum a `_taint_op` request needs of a view: an identity the
    quick-loaded WeakSet can be asked about. The engine is recorded, not run, so
    nothing here is read."""


def test_taint_op_threads_max_iters_into_the_engine_812(monkeypatch):
    # #812: a fixpoint-truncated result's remediation string names `--max-iters`,
    # and until this handler forwarded the knob that advice pointed at nothing a
    # user could do. The threading is what makes the remediation real, so it is
    # pinned END TO END on the handler: the value in the request becomes the
    # engine's budget, and an absent value leaves the engine's own default --
    # the reason every pre-existing caller and the whole backward path are
    # unaffected by the new knob.
    #
    # The recorder SUBCLASSES the real engine rather than replacing it, so the
    # observed `max_iters` is whatever the real constructor resolved (including
    # its default), not a kwargs dict this test could read either way.
    import inspect

    seen: list[int] = []
    # Read off the REAL class before it is patched: the absent-value contract is
    # "the engine's own default", and comparing against the signature proves the
    # handler did not substitute one of its own.
    engine_default = inspect.signature(
        rts._taint.TaintEngine).parameters["max_iters"].default

    class _RecordingEngine(rts._taint.TaintEngine):
        def __init__(self, bv, models, **kw):
            super().__init__(bv, models, **kw)
            seen.append(self.max_iters)

        def forward(self, func, locators, **kw):        # never analyse anything
            return {"direction": "forward", "reached_sinks": [], "leaves": []}

    monkeypatch.setattr(rts._taint, "TaintEngine", _RecordingEngine)

    class _Ctx(_CtxWithBV):
        def _find_function(self, bv, name):
            return object()

    ctx = _Ctx(_BVPlain())
    request = {"function": "handler", "sources": ["param:0"]}
    rts._taint_op(ctx, "active", dict(request, max_iters=7))
    rts._taint_op(ctx, "active", dict(request))
    # 7 from the request; then the engine's own default, NOT a zero or a None
    # that would make the fixpoint analyse nothing. 256 is also the CLI flag's
    # default, so the two ends agree on the budget an unflagged run gets.
    assert engine_default == 256, engine_default
    assert seen == [7, engine_default], seen


def test_taint_answers_disclose_the_view_analysis_state_811(monkeypatch):
    # #811, the view-level half: both taint answers carry the SAME
    # `{analysis_state, partial}` shape every other read op attaches, imported
    # from `read_listing` rather than re-derived, so the taint surface cannot
    # fork the convention. It shipped pinned by nothing -- deleting both
    # `.update(_analysis_state_fields(bv))` calls left 711 tests green across
    # all four taint test files -- and the catalog half is the one that matters:
    # `taint models --present` computes presence by WALKING the view, so on a
    # quick-loaded view a modeled sink is reported ABSENT merely because its
    # caller was never analysed. That is a false all-clear in catalog form, and
    # the disclosure is the only thing standing between a reader and it.
    from bn_agent_bridge import read_listing as rl

    bv = _BVTriage()
    ctx = _CtxWithBV(bv)
    full = rts._taint_models_op(ctx, "active", {"present": True})
    assert full["analysis_state"] == "full", full
    assert full["partial"] is False, full

    # The same view, now quick-loaded: the catalog must say so rather than
    # answering in the same shape as a fully analysed one.
    rl._quick_loaded_views.add(bv)
    try:
        quick = rts._taint_models_op(ctx, "active", {"present": True})
    finally:
        rl._quick_loaded_views.discard(bv)
    assert quick["analysis_state"] == "quick", quick
    assert quick["partial"] is True, quick

    # The slice half. `require_analysis` refuses a quick view outright, so this
    # path can only ever report "full" today -- which is exactly why it needs a
    # test: the fields are there so the contract is uniform and a future
    # quick-tolerant taint mode cannot ship a silent partial answer, and a
    # contract kept for a future caller is the easiest kind to delete.
    class _RecordingEngine(rts._taint.TaintEngine):
        def forward(self, func, locators, **kw):        # never analyse anything
            return {"direction": "forward", "reached_sinks": [], "leaves": []}

    monkeypatch.setattr(rts._taint, "TaintEngine", _RecordingEngine)

    class _Ctx(_CtxWithBV):
        def _find_function(self, bv, name):
            return object()

    result = rts._taint_op(_Ctx(_BVPlain()), "active",
                           {"function": "handler", "sources": ["param:0"]})
    assert result["analysis_state"] == "full", result
    assert result["partial"] is False, result


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


def test_builtin_snprintf_family_declares_size_arm_808():
    # #808: snprintf/vsnprintf write AT MOST `size` bytes into the destination, so
    # the write size (arg1) is an attacker-controlled write length into arg0 -- the
    # model detail claimed size coverage while nothing armed it, so a tainted size
    # returned reached_sinks=[] plus a false all-clear. It is declared with the
    # bounded-write len_arg/buf_arg pair (the same fields the recv/read family
    # uses), NOT as an extra `tainted_args` entry, which keeps meaning "the tainted
    # FORMAT at arg2". The fortified forms shift only the format to arg4: maxlen
    # (the write length) stays arg1.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    for name, fmt_idx in (("snprintf", 2), ("vsnprintf", 2),
                          ("snprintf_chk", 4), ("vsnprintf_chk", 4)):
        sink = models[name]["sink"]
        assert sink["len_arg"] == 1, name          # the size / maxlen
        assert sink["buf_arg"] == 0, name          # the destination
        assert sink["tainted_args"] == [fmt_idx], name
    # discoverable: `bn taint models` surfaces the bounded-write indices, and its
    # condition names BOTH armed indices so the size arm is not invisible beside
    # the format arg.
    cat = build_catalog(models)
    entry = {e["symbol"]: e for lst in cat["sinks_by_class"].values() for e in lst}["snprintf"]
    assert entry["len_arg"] == 1 and entry["buf_arg"] == 0
    assert "arguments 1 (length) or 2" in entry["model_description"], entry["model_description"]


# --- #876: the fortified read family, driven through the ENGINE --------------
#
# Model-DB field values are not the claim. The claim is what a
# `--sink-class recv_overflow` query REPORTS at a fortified callsite, so every
# test below runs `TaintEngine.forward` and compares the `_chk` entry against
# its bare twin on the SAME program. Reading `sink["class"]` back out of the
# JSON cannot see the engine's two class-keyed false-positive suppressors --
# the #159 provably-bounded receive-return downgrade and the #307
# reused-aliased-slot re-headline -- and those are exactly what a fortified
# class silently opted the family out of.

# `__pread64_chk` is the FORTIFY spelling a large-file-offset build emits, and
# it needs its own row: `lookup_model`'s LFS64 allowlist rewrites `pread64` to
# `pread`, so the BARE half of the gate already covers that build, but it has
# no `pread64_chk` entry -- the fortified half only covers it if the model DB
# names the spelling itself.
_FORTIFIED_READ_PAIRS = (("read", "__read_chk"), ("recv", "__recv_chk"),
                         ("recvfrom", "__recvfrom_chk"), ("pread", "__pread_chk"),
                         ("pread", "__pread64_chk"))

_SINK_ADDR = "0x20"          # where every fixture below puts the call under test


def _fakes():
    """The synthetic MLIL-SSA fakes the engine's own tier-1 suite drives it with.

    Imported, never re-declared: a private second copy of the fakes is a second
    definition of "what the engine sees", and a test asserting against its own
    shape has stopped measuring the production seam.
    """
    import test_taint_engine
    return test_taint_engine


def _call_under_test(F, callee, index, length, reads, dest):
    """``rv#1 = callee(3, <dest>, <length>[, offset][, objsize])`` at ``_SINK_ADDR``.

    `pread` takes a trailing offset, and a FORTIFY `_chk` twin a trailing
    objsize guard -- a compile-time bound, not data flow: it appends, it does
    not shift buf/len. The byte count lands in its own SSA var, as a real
    caller writes it, so a `ret:` seed can distinguish this callsite from a
    same-named one earlier in the program.
    """
    params = [F.FExpr("MLIL_CONST", "3", constant=3), dest, length]
    if callee.lstrip("_").startswith("pread"):
        params.append(F.FExpr("MLIL_CONST", "0", constant=0))
    if callee.endswith("_chk"):
        params.append(F.FExpr("MLIL_CONST", "0x10", constant=0x10))
    return F.FInstr(index, int(_SINK_ADDR, 16), "MLIL_CALL_SSA",
                    f"rv#1 = {callee}(3, dst, len)", reads=reads,
                    writes=[F.FSSA(F.FVar("rv"), 1)],
                    dest=F.FExpr("MLIL_CONST_PTR", "0x901", constant=0x901),
                    params=params)


def _plain_length_program(callee, *, sized_destination=False):
    """``[dst = malloc(n);] callee(3, dst, n)`` for an attacker-controlled `n`.

    By default the destination is an untracked stack buffer, so nothing bounds
    the write and the finding must stand as a real overflow. With
    `sized_destination` the destination is allocated in-function with the very
    length being written -- the #443 bounded-write pair's reason for declaring
    `buf_arg`, and the only place a wrong `buf_arg` is observable.
    """
    F = _fakes()
    n = F.FVar("n", ident=1); n0 = F.FSSA(n, 0)
    length = F.FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])
    instrs = []
    if sized_destination:
        rax1 = F.FSSA(F.FVar("rax"), 1)
        instrs.append(F.FInstr(0, 0x10, "MLIL_CALL_SSA", "rax#1 = malloc(n#0)",
                               reads=[n0], writes=[rax1],
                               dest=F.FExpr("MLIL_CONST_PTR", "0x902", constant=0x902),
                               params=[F.FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])]))
        dest, reads = F.FExpr("MLIL_VAR_SSA", "rax#1", reads=[rax1]), [n0, rax1]
    else:
        dest = F.FExpr("MLIL_ADDRESS_OF", "&dst", src=F.FVar("dst", typ="char[0x10]"))
        reads = [n0]
    instrs.append(_call_under_test(F, callee, len(instrs), length, reads, dest))
    return (F.FFunc("handler", 0x10, F.FSSAFunc(instrs), params=[n]),
            F.FBV({0x901: callee, 0x902: "malloc"}))


def _bounded_receive_program(callee):
    """``n = read(3, &src, 0x40); callee(3, &dst, n)``.

    The #159 idiom `_comment_recv_overflow` itself names as the ~100%-false-
    positive one: the length is the return of a modeled receive provably
    bounded by a constant count, so the honest verdict is `bounded_len` with
    that bound, not an overflow.
    """
    F = _fakes()
    n1 = F.FSSA(F.FVar("n"), 1)
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", "n#1 = read(3, &src, 0x40)", writes=[n1],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
                 params=[F.FExpr("MLIL_CONST", "3", constant=3),
                         F.FExpr("MLIL_ADDRESS_OF", "&src", src=F.FVar("src", typ="char[0x40]")),
                         F.FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        _call_under_test(F, callee, 1, F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1]), [n1],
                         F.FExpr("MLIL_ADDRESS_OF", "&dst", src=F.FVar("dst", typ="char[0x10]"))),
    ]
    return (F.FFunc("handler", 0x10, F.FSSAFunc(instrs)),
            F.FBV({0x900: "read", 0x901: callee}))


def _ambiguous_slot_program(callee):
    """``fgets(&slot, 8, fp); slot = 0x10; n = slot @ mem; callee(3, &dst, n)``.

    The #307 shape: the length reads an address-taken slot whose taint arrived
    version-agnostically through an out-param write, and a competing in-function
    store makes the reaching definition path-ambiguous. The engine cannot stand
    behind an overflow VERDICT there, so it re-headlines to the neutral
    `tainted_len` without dropping the flow.
    """
    F = _fakes()
    slot = F.FVar("slot", ident=30); slot4 = F.FSSA(slot, 4)
    rp1 = F.FSSA(F.FVar("rp"), 1); ln1 = F.FSSA(F.FVar("ln"), 1)
    instrs = [
        F.FInstr(0, 0x04, "MLIL_SET_VAR_SSA", "rp#1 = &slot", writes=[rp1],
                 src=F.FExpr("MLIL_ADDRESS_OF", "&slot", src=slot)),
        F.FInstr(1, 0x08, "MLIL_CALL_SSA", "fgets(rp#1, 8, fp)", reads=[rp1], writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x910", constant=0x910),
                 params=[F.FExpr("MLIL_VAR_SSA", "rp#1", reads=[rp1]),
                         F.FExpr("MLIL_CONST", "8", constant=8),
                         F.FExpr("MLIL_CONST", "0", constant=0)]),
        F.FInstr(2, 0x0c, "MLIL_SET_VAR_ALIASED", "slot = 0x10", dest=slot,
                 src=F.FExpr("MLIL_CONST", "0x10", constant=0x10)),
        F.FInstr(3, 0x10, "MLIL_SET_VAR_SSA", "ln#1 = slot @ mem", reads=[slot4], writes=[ln1],
                 src=F.FExpr("MLIL_VAR_ALIASED", "slot @ mem", reads=[slot4])),
        _call_under_test(F, callee, 4, F.FExpr("MLIL_VAR_SSA", "ln#1", reads=[ln1]), [ln1],
                         F.FExpr("MLIL_ADDRESS_OF", "&dst", src=F.FVar("dst", typ="char[0x10]"))),
    ]
    return (F.FFunc("handler", 0x00, F.FSSAFunc(instrs)),
            F.FBV({0x910: "fgets", 0x901: callee}))


def _reported(program, *seeds, gate=("recv_overflow",), at=None):
    """Every sink reported at `at` (default: the call under test) for `seeds`.

    `gate=()` runs with the opt-in class disabled, which must report nothing.
    """
    from bn_agent_bridge import taint_engine as te
    func, bv = program
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator(s) for s in seeds], enabled_sink_classes=set(gate))
    return [s["sink"] for s in result["reached_sinks"]
            if s["sink"]["address"] == (at or _SINK_ADDR)]


@pytest.mark.parametrize("base,chk", _FORTIFIED_READ_PAIRS)
def test_fortified_read_sink_fires_under_the_gate_like_its_base_twin_876(base, chk):
    # #876: the read-family `_chk` entries carried `sources` and `return_bound`
    # but NO `sink`, so a tainted length into `__read_chk` answered
    # reached_sinks=[] -- a false all-clear on the FORTIFY build a hardened
    # target actually ships. Opt-in exactly like the bare twin: the ~100%-FP
    # fill-loop idiom (#499) is the same call shape, so arming one without the
    # other would make `--sink-class recv_overflow` silently partial.
    assert _reported(_plain_length_program(chk), "param:0", gate=()) == []
    bare = _reported(_plain_length_program(base), "param:0")
    fort = _reported(_plain_length_program(chk), "param:0")
    assert len(bare) == 1 and len(fort) == 1, (bare, fort)
    # arg 2 is the armed length on both -- the trailing objsize guard appends,
    # it does not shift buf/len.
    assert fort[0]["tainted_arg_index"] == bare[0]["tainted_arg_index"] == 2
    # ...and the SAME bug class. `fortified_overflow` reads as "lower severity"
    # but is what opts the entry out of the suppressors the next two tests pin;
    # the fortify nuance is carried in `detail`, which the class is not free to
    # encode without changing what the engine does with the finding.
    assert fort[0]["class"] == bare[0]["class"] == "overflow_len", (bare, fort)
    assert chk in fort[0]["detail"] and "aborts at runtime" in fort[0]["detail"]


@pytest.mark.parametrize("base,chk", _FORTIFIED_READ_PAIRS)
def test_fortified_read_sink_declares_the_write_destination_876(base, chk):
    # `buf_arg` is observable only through the #443 bounded-write downgrade:
    # with the destination allocated in-function from the very length being
    # written, the copy provably fits and the finding is relabeled. A `buf_arg`
    # naming any other argument -- the fd, or the _chk trailing guard -- leaves
    # it an overflow, so this is the test that catches a mis-declared pair.
    for callee in (base, chk):
        sinks = _reported(_plain_length_program(callee, sized_destination=True), "param:0")
        assert len(sinks) == 1, (callee, sinks)
        assert sinks[0]["class"] == "bounded_len", (callee, sinks)
        assert "attacker-derived length, but" in sinks[0]["detail"], (callee, sinks)


@pytest.mark.parametrize("base,chk", _FORTIFIED_READ_PAIRS)
def test_fortified_read_sink_takes_the_bounded_receive_downgrade_876(base, chk):
    # #159, and the reason the class may not differ from the bare twin's: the
    # length is the return of a modeled receive provably bounded by a constant
    # count -- the dominant idiom `_comment_recv_overflow` names as ~100% false
    # positive. The engine's downgrade tests `class == "overflow_len"`, so a
    # `fortified_overflow` twin reported a FALSE overflow at exactly the
    # callsite its bare twin calls bounded.
    bare = _reported(_bounded_receive_program(base), "ret:read")
    fort = _reported(_bounded_receive_program(chk), "ret:read")
    assert len(bare) == 1 and len(fort) == 1, (bare, fort)
    assert (bare[0]["class"], bare[0].get("source_bound")) == ("bounded_len", "0x40"), bare
    assert (fort[0]["class"], fort[0].get("source_bound")) \
        == (bare[0]["class"], bare[0].get("source_bound")), (bare, fort)


@pytest.mark.parametrize("base,chk", _FORTIFIED_READ_PAIRS)
def test_fortified_read_sink_neutralizes_an_ambiguous_length_876(base, chk):
    # #307, the gate's other class-keyed suppressor: a length read from a reused
    # address-taken slot with a competing in-function writer is a path-ambiguous
    # reaching definition, so the overflow VERDICT is dropped for the neutral
    # `tainted_len`. Nothing is hidden -- the flow stays in reached_sinks -- but
    # a fortified class skipped the re-headline and kept the unsound label.
    bare = _reported(_ambiguous_slot_program(base), "arg:fgets:0")
    fort = _reported(_ambiguous_slot_program(chk), "arg:fgets:0")
    assert len(bare) == 1 and len(fort) == 1, (bare, fort)
    assert (bare[0]["class"], bare[0].get("via")) == ("tainted_len", "reused_aliased_slot"), bare
    assert (fort[0]["class"], fort[0].get("via")) \
        == (bare[0]["class"], bare[0].get("via")), (bare, fort)


@pytest.mark.parametrize("base", ("fgets", "fread"))
def test_fortified_entries_whose_base_has_no_sink_stay_sinkless_876(base):
    # Must-not-fire twin. `fgets`/`fread` declare no length sink on EITHER
    # side -- they are bounded by their own size argument in a shape the
    # engine does not model as attacker-controlled -- so the #876 sweep must
    # not "fix" them into existence. The defect was an ASYMMETRY with the
    # base model, not the absence of a sink.
    for callee in (base, f"__{base}_chk"):
        assert _reported(_plain_length_program(callee), "param:0") == [], callee
    # Not vacuous: the identical harness, seed and gate DO report the armed
    # fortified read twin.
    assert len(_reported(_plain_length_program("__read_chk"), "param:0")) == 1


def _twin_as_bounding_source_program(callee):
    """``n = callee(3, &buf, 0x40); memcpy(&dst, &buf, n); system(&dst)``.

    Drives the read-family entry as a SOURCE rather than as a sink, which is
    the half its `sources` and `return_bound` fields describe and the A/B
    programs above never reach. `return_bound` makes the memcpy length provably
    bounded (#159) and the `ret` source is what taints it at all; the
    `*arg:1` source is what taints the copied buffer, so the `system` call is
    only reachable as a command_injection if the buffer half is declared too.

    The fortify guard is deliberately a DIFFERENT constant from the count:
    with both 0x40 the reported bound is the same either way and
    `max_from_arg`'s index goes unpinned, which is how a guard-indexed bound
    would downgrade a real overflow while the suite stayed green.
    """
    F = _fakes()
    n1 = F.FSSA(F.FVar("n"), 1)
    buf = F.FVar("buf", typ="char[0x40]"); dst = F.FVar("dst", typ="char[0x10]")
    params = [F.FExpr("MLIL_CONST", "3", constant=3),
              F.FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
              F.FExpr("MLIL_CONST", "0x40", constant=0x40)]
    if callee.lstrip("_").startswith("pread"):
        params.append(F.FExpr("MLIL_CONST", "0", constant=0))
    if callee.endswith("_chk"):
        params.append(F.FExpr("MLIL_CONST", "0x99", constant=0x99))
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", f"n#1 = {callee}(3, &buf, 0x40)", writes=[n1],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x901", constant=0x901), params=params),
        F.FInstr(1, 0x20, "MLIL_CALL_SSA", "memcpy(&dst, &buf, n#1)", reads=[n1], writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x2010", constant=0x2010),
                 params=[F.FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
                         F.FExpr("MLIL_ADDRESS_OF", "&buf", src=buf),
                         F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])]),
        F.FInstr(2, 0x30, "MLIL_CALL_SSA", "system(&dst)", writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x2020", constant=0x2020),
                 params=[F.FExpr("MLIL_ADDRESS_OF", "&dst", src=dst)]),
    ]
    return (F.FFunc("handler", 0x10, F.FSSAFunc(instrs)),
            F.FBV({0x901: callee, 0x2010: "memcpy", 0x2020: "system"}))


@pytest.mark.parametrize("base,chk", _FORTIFIED_READ_PAIRS)
def test_fortified_read_source_bounds_a_downstream_copy_like_its_twin_876(base, chk):
    # The twins carry `sources` and `return_bound` as well as the sink this PR
    # added, and those fields only speak when the twin is the RECEIVE, not the
    # copy: a regression dropping `return_bound` turns the bounded copy below
    # into a false unbounded overflow -- the round-1 blocker's failure mode
    # arriving through the source half instead of the class. Nothing else here
    # reaches it, because every other program uses the BARE receive to bound
    # the length and the twin only as the sink.
    for callee in (base, chk):
        copy = _reported(_twin_as_bounding_source_program(callee), f"call:{callee}", gate=())
        assert len(copy) == 1, (callee, copy)
        # `ret` + `return_bound.max_from_arg 2`: attacker-derived but provably
        # bounded by the count this call was given.
        assert (copy[0]["class"], copy[0].get("source_bound")) == ("bounded_len", "0x40"), (callee, copy)
        # `*arg:1`: the received bytes really do land in the buffer, so the
        # copy's destination is tainted and the exec sink downstream fires.
        shell = _reported(_twin_as_bounding_source_program(callee), f"call:{callee}",
                          gate=(), at="0x30")
        assert [s["class"] for s in shell] == ["command_injection"], (callee, shell)


# --- #876 sweep completeness: fortified spellings whose bare twin is armed ---
#
# The round-2/3 LFS gap was found by asking which SPELLING a real build emits
# that the DB does not name -- a question enumerating DB keys cannot answer.
# Asked of the whole fortify surface, the copy family answered the same way:
# these five are exported by the platform C library, their bare twins carry an
# armed sink, and the fortified spelling resolved to nothing.
_FORTIFIED_COPY_PAIRS = (("mempcpy", "__mempcpy_chk"), ("strlcpy", "__strlcpy_chk"),
                         ("strlcat", "__strlcat_chk"), ("wmemcpy", "__wmemcpy_chk"),
                         ("wmemmove", "__wmemmove_chk"))


def _copy_program(callee, *, tainted_length=True):
    """``read(3, &src, 0x40); callee(&dst, &src, n[, guard]); system(&dst)``.

    `n` is a parameter, not the receive's return, so nothing bounds it and the
    length arm reports the class the model declares. With
    `tainted_length=False` the length is a constant and the ONLY route to the
    destination is the model's own `propagates` source, which is what makes
    the downstream exec sink evidence for `propagates.from` rather than for
    the seed.
    """
    F = _fakes()
    n = F.FVar("n", ident=1); n0 = F.FSSA(n, 0)
    src = F.FVar("src", typ="char[0x40]"); dst = F.FVar("dst", typ="char[0x10]")
    length = (F.FExpr("MLIL_VAR_SSA", "n#0", reads=[n0]) if tainted_length
              else F.FExpr("MLIL_CONST", "0x10", constant=0x10))
    params = [F.FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
              F.FExpr("MLIL_ADDRESS_OF", "&src", src=src), length]
    if callee.endswith("_chk"):
        params.append(F.FExpr("MLIL_CONST", "0x10", constant=0x10))
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", "read(3, &src, 0x40)", writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
                 params=[F.FExpr("MLIL_CONST", "3", constant=3),
                         F.FExpr("MLIL_ADDRESS_OF", "&src", src=src),
                         F.FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        F.FInstr(1, 0x20, "MLIL_CALL_SSA", f"{callee}(&dst, &src, len)",
                 reads=([n0] if tainted_length else []), writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x901", constant=0x901), params=params),
        F.FInstr(2, 0x30, "MLIL_CALL_SSA", "system(&dst)", writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x2020", constant=0x2020),
                 params=[F.FExpr("MLIL_ADDRESS_OF", "&dst", src=dst)]),
    ]
    return (F.FFunc("handler", 0x10, F.FSSAFunc(instrs), params=[n]),
            F.FBV({0x900: "read", 0x901: callee, 0x2020: "system"}))


def _bounded_copy_program(callee):
    """``n = read(3, &src, 0x40); callee(&dst, &src, n[, guard])``.

    The #159 idiom again, aimed at the copy family: the length is a modeled
    receive return provably bounded by a constant count, so the honest verdict
    is `bounded_len`. The engine reaches that ONLY for `class == overflow_len`,
    which is why a `_chk` entry may not take a class its bare twin does not.
    """
    F = _fakes()
    n1 = F.FSSA(F.FVar("n"), 1)
    src = F.FVar("src", typ="char[0x40]"); dst = F.FVar("dst", typ="char[0x10]")
    params = [F.FExpr("MLIL_ADDRESS_OF", "&dst", src=dst),
              F.FExpr("MLIL_ADDRESS_OF", "&src", src=src),
              F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])]
    if callee.endswith("_chk"):
        params.append(F.FExpr("MLIL_CONST", "0x10", constant=0x10))
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", "n#1 = read(3, &src, 0x40)", writes=[n1],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
                 params=[F.FExpr("MLIL_CONST", "3", constant=3),
                         F.FExpr("MLIL_ADDRESS_OF", "&src", src=src),
                         F.FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        F.FInstr(1, 0x20, "MLIL_CALL_SSA", f"{callee}(&dst, &src, n#1)", reads=[n1], writes=[],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x901", constant=0x901), params=params),
    ]
    return (F.FFunc("handler", 0x10, F.FSSAFunc(instrs)),
            F.FBV({0x900: "read", 0x901: callee}))


@pytest.mark.parametrize("base,chk", _FORTIFIED_COPY_PAIRS)
def test_fortified_copy_spelling_is_modeled_like_its_bare_twin_876(base, chk):
    # A hardened build emits the `_chk` spelling; with no model at all
    # `lookup_model` returned None, so `taint forward` answered reached_sinks:[]
    # where the unfortified build reported the overflow.
    bare = _reported(_copy_program(base), "param:0", "call:read", gate=())
    fort = _reported(_copy_program(chk), "param:0", "call:read", gate=())
    assert len(bare) == 1 and len(fort) == 1, (bare, fort)
    # the trailing destlen guard appends; it does not shift dest/src/len
    assert fort[0]["tainted_arg_index"] == bare[0]["tainted_arg_index"] == 2
    # Same class as the bare twin, for the reason `_comment_fortified` states:
    # `class` is what the engine's FP suppressors test for, so a `_chk` entry
    # under a different class headlines an overflow its twin calls bounded.
    # (The engine tests the class string alone -- neither suppressor requires a
    # `len_arg`/`buf_arg` pair, which these entries do not declare.)
    assert fort[0]["class"] == bare[0]["class"] == "overflow_len", (bare, fort)
    assert chk in fort[0]["detail"] and "aborts at runtime" in fort[0]["detail"]
    # ...and that is observable, not stylistic: on the bounded-receive-return
    # idiom both spellings must reach the same downgrade.
    bare_b = _reported(_bounded_copy_program(base), "ret:read", gate=())
    fort_b = _reported(_bounded_copy_program(chk), "ret:read", gate=())
    assert len(bare_b) == 1 and len(fort_b) == 1, (bare_b, fort_b)
    assert (bare_b[0]["class"], bare_b[0].get("source_bound")) == ("bounded_len", "0x40"), bare_b
    assert (fort_b[0]["class"], fort_b[0].get("source_bound")) \
        == (bare_b[0]["class"], bare_b[0].get("source_bound")), (bare_b, fort_b)
    # The copy propagates the tainted SOURCE into the destination. Measured
    # with an UNTAINTED length, so the only route to `&dst` is
    # `propagates.from` -- with the length tainted too the destination is
    # tainted either way and a `from` naming the wrong operand goes unnoticed.
    for callee in (base, chk):
        shell = _reported(_copy_program(callee, tainted_length=False), "call:read",
                          gate=(), at="0x30")
        assert [s["class"] for s in shell] == ["command_injection"], (callee, shell)


def test_scanf_family_models_carry_arity_capped_flag_851():
    # #851: scanf/fscanf (and their isoc99 aliases) declare `arity_capped: true`
    # so the engine can emit a weak-seed note when a real call has more actual
    # params than the unrolled *arg:N run covers, rather than silently reporting
    # a clean all-clear for the unmodeled destinations.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    for name in ("scanf", "__isoc99_scanf", "fscanf", "__isoc99_fscanf"):
        assert models[name].get("arity_capped") is True, (
            f"{name} must carry arity_capped=true so the engine can detect "
            "extra destinations beyond the modeled run"
        )
    # sscanf/isoc99_sscanf reach their destinations through propagates rather
    # than sources, but their run is unrolled the same way -- so they carry the
    # flag too, and the engine discloses the residual on the propagator path (an
    # over-long sscanf's unmodeled destination is where the tainted source would
    # have landed). Pinning the OPPOSITE here is what let the first cut of the
    # #851 fix ship the scanf half only, with this test certifying the gap.
    for name in ("sscanf", "__isoc99_sscanf"):
        assert models[name].get("arity_capped") is True, (
            f"{name} propagates into its destinations and its run is unrolled "
            "like scanf's, so it must carry arity_capped=true"
        )


def test_scanf_arity_residual_marker_in_weak_seed_set_851():
    # #851: the engine emits an assumption containing "scanf_arity_residual"
    # when a call has more params than the model covers; taint_result must
    # recognise that marker to withhold the all-clear.
    from bn_agent_bridge.taint_result import _WEAK_SEED_ASSUMPTION_MARKERS
    assert "scanf_arity_residual" in _WEAK_SEED_ASSUMPTION_MARKERS, (
        "scanf_arity_residual must be in _WEAK_SEED_ASSUMPTION_MARKERS so a "
        "residual-arity assumption withholds safe_to_report_all_clear"
    )
