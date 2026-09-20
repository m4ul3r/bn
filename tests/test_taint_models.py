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


def _reported(program, seed, *, gate=("recv_overflow",)):
    """Every sink the call under test reports for `seed`.

    `gate=()` runs with the opt-in class disabled, which must report nothing.
    """
    from bn_agent_bridge import taint_engine as te
    func, bv = program
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator(seed)], enabled_sink_classes=set(gate))
    return [s["sink"] for s in result["reached_sinks"]
            if s["sink"]["address"] == _SINK_ADDR]


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
