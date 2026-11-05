from __future__ import annotations

import re

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


class _Sec:
    def __init__(self, name):
        self.name = name


class _BVSynthetic(_BVSpellings):
    """BN's per-view builtin placeholders (`__builtin_memcpy`, `__builtin_strcpy`)
    in its synthetic-builtins section, one of them referenced by a lifted inline
    builtin, plus a real `system` import elsewhere with no callers."""
    _SECTIONS = {0x9000: ".synthetic_builtins", 0x9008: ".synthetic_builtins",
                 0x5000: ".extern"}

    def __init__(self, refs=None):
        self.functions = []
        self._syms = {"__builtin_memcpy": [_Sym("__builtin_memcpy", 0x9000)],
                      "__builtin_strcpy": [_Sym("__builtin_strcpy", 0x9008)],
                      "system": [_Sym("system", 0x5000)]}
        self._refs = refs or {}

    def get_symbols(self):
        return [s for lst in self._syms.values() for s in lst]

    def get_sections_at(self, a):
        return [_Sec(self._SECTIONS[a])] if a in self._SECTIONS else []


def _present_symbols(res):
    return {e["symbol"] for e in res["items"] if e.get("present")}


def test_unreferenced_synthetic_builtins_are_not_present():
    # The placeholders exist on every view, so with no code reference they prove
    # nothing; a real import with no callers still counts.
    res = rts._taint_models_op(_CtxWithBV(_BVSynthetic()), "active",
                               {"present": True, "callsites": True})
    assert _present_symbols(res) == {"system"}
    # A placeholder BN DID reference (an inline builtin it lifted) is present.
    res = rts._taint_models_op(_CtxWithBV(_BVSynthetic({0x9000: [_Ref(0x400)]})), "active",
                               {"present": True})
    assert _present_symbols(res) == {"system", "__builtin_memcpy"}


class _BVStripped(_BVSynthetic):
    """Every function auto-named, nothing modeled but the placeholders."""
    def __init__(self):
        super().__init__()
        self.functions = [_FakeFn("sub_401000"), _FakeFn("sub_401200")]
        del self._syms["system"]


def test_presence_walk_counts_matched_models():
    stripped = rts._taint_models_op(_CtxWithBV(_BVStripped()), "active", {"present": True})
    assert stripped["presence_catalog"] is True
    assert stripped["present_model_count"] == 0
    assert stripped["items"] == []
    named = rts._taint_models_op(_CtxWithBV(_FakeBV(["memcpy", "system"])), "active",
                                 {"role": "sink"})
    assert named["present_model_count"] == 2


def test_no_target_catalog_claims_no_presence():
    res = rts._taint_models_op(_CtxNoView(), None, {})
    assert res["presence_catalog"] is False
    assert "present_model_count" not in res
    assert not any("present" in e for e in res["items"])
    # A presence walk is what switches the claim (and its note) on.
    walked = rts._taint_models_op(_CtxWithBV(_FakeBV(["memcpy"])), "active", {})
    assert walked["presence_catalog"] is True
    assert walked["catalog_note"] != res["catalog_note"]


_ROLELESS_SINK = {"app_recv": {"sink": {"class": "overflow_len"}, "bogus_key": 1}}


@pytest.mark.parametrize("via", ["$BN_TAINT_MODELS", "--models"])
def test_bad_user_model_refusal_names_the_knob_that_supplied_it(via):
    with pytest.raises(rts.OperationFailure) as err:
        rts._taint_models_op(_CtxNoView(), None, {
            "user_models": _ROLELESS_SINK, "user_models_path": "/proj/models.json",
            "user_models_via": via})
    assert f"({via} /proj/models.json)" in str(err.value)


def test_overridden_env_file_is_disclosed_as_not_loaded(monkeypatch):
    monkeypatch.delenv("BN_TAINT_MODELS", raising=False)
    res = rts._taint_models_op(_CtxNoView(), None, {
        "user_models": {"app_copy": {"propagates": [{"from": "*arg:1", "to": "*arg:0"}]}},
        "user_models_path": "/proj/flag.json", "user_models_via": "--models",
        "user_models_env_overridden": "/proj/env.json"})
    kinds = {o["kind"]: o for o in res["overlays"]}
    assert kinds["user"]["via"] == "--models"
    assert kinds["env_not_loaded"] == {"kind": "env_not_loaded", "env": "BN_TAINT_MODELS",
                                       "path": "/proj/env.json", "overridden_by": "--models"}


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


@pytest.mark.parametrize("spelling,key,role,detail", [
    ("mq_receive", "mq_receive", "source", None),
    ("mq_timedreceive", "mq_timedreceive", "source", None),
    ("__mq_timedreceive_time64", "mq_timedreceive_time64", "source", None),
    ("msgrcv", "msgrcv", "source", None),
    ("posix_spawn", "posix_spawn", "command_injection", [1, 4]),
    ("posix_spawnp", "posix_spawnp", "command_injection", [1, 4]),
    ("execvpe", "execvpe", "command_injection", [0]),
    ("execle", "execle", "command_injection", [0]),
    ("open64", "open", "path_injection", [0]),
    ("openat", "openat", "path_injection", [1]),
    ("fopen64", "fopen", "path_injection", [0]),
    ("__xstat64", "xstat", "path_injection", [1]),
    ("renameat", "renameat", "path_injection", [1, 3]),
    ("_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode",
     "ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode", "path_injection", [1]),
    ("_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode",
     "ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode", "path_injection", [1]),
])
def test_queue_spawn_and_path_spellings_resolve_to_models_957(spelling, key, role, detail):
    # #957: the binary spelling (mangled, LFS64, time64, __-prefixed) is what
    # `taint models --present` and the engine look up; each must land on a model
    # that arms the path/argv argument, or seeds the received buffer + length.
    from bn_agent_bridge.taint_engine import load_models, lookup_model
    got, model = lookup_model(load_models(), spelling)
    assert got == key
    if role == "source":
        assert [s["to"] for s in model["sources"]] == ["*arg:1", "ret"]
        assert model["return_bound"] == {"max_from_arg": 2}
    else:
        assert model["sink"]["class"] == role
        assert model["sink"]["tainted_args"] == detail
        assert not model["sink"].get("optional")      # armed by default


def _queue_to_path_and_spawn_program():
    """``mq_timedreceive(q, &msg, 0x400, 0, &ts); fopen(&msg, "r");
    posix_spawn(&pid, &msg, 0, 0, argv, envp)`` -- the dogfood shape: the
    binary's input arrives through a message queue and reaches a path and a
    spawned program."""
    F = _fakes()
    msg = F.FVar("msg", typ="char[0x400]")
    c = lambda v: F.FExpr("MLIL_CONST", hex(v), constant=v)
    at = lambda name: F.FExpr("MLIL_ADDRESS_OF", f"&{name}", src=msg if name == "msg"
                              else F.FVar(name, typ="int32_t"))
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", "mq_timedreceive(q, &msg, 0x400, 0, &ts)",
                 dest=F.FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
                 params=[c(3), at("msg"), c(0x400), c(0), at("ts")]),
        F.FInstr(1, 0x20, "MLIL_CALL_SSA", "fopen(&msg, \"r\")",
                 dest=F.FExpr("MLIL_CONST_PTR", "0x901", constant=0x901),
                 params=[at("msg"), c(0x2000)]),
        F.FInstr(2, 0x30, "MLIL_CALL_SSA", "posix_spawn(&pid, &msg, 0, 0, 0, 0)",
                 dest=F.FExpr("MLIL_CONST_PTR", "0x902", constant=0x902),
                 params=[at("pid"), at("msg"), c(0), c(0), c(0), c(0)]),
    ]
    return (F.FFunc("pump", 0x10, F.FSSAFunc(instrs)),
            F.FBV({0x900: "mq_timedreceive", 0x901: "fopen", 0x902: "posix_spawn"}))


def test_message_queue_input_reaches_path_and_spawn_sinks_957():
    from bn_agent_bridge import taint_engine as te
    func, bv = _queue_to_path_and_spawn_program()
    result = te.TaintEngine(bv, te.load_models()).forward(
        func, [te.parse_locator("call:mq_timedreceive")])
    got = sorted((s["sink"]["callee"], s["sink"]["class"], s["sink"]["tainted_arg_index"])
                 for s in result["reached_sinks"])
    assert got == [("fopen", "path_injection", 0),
                   ("posix_spawn", "command_injection", 1)], result["reached_sinks"]


def test_build_catalog_includes_underscore_prefixed_real_models_849():
    # #849: build_catalog must NOT skip models whose names start with `_` but
    # are NOT doc-key prefixed (_comment). ``__builtin_memset`` starts with
    # ``_`` but is a genuine propagator and sink the engine applies on real
    # binaries; the catalog used to omit such names entirely. The fix matches
    # only the doc-key prefix (``_comment``), not every leading ``_``.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    prop_syms = {p["symbol"] for p in build_catalog(models, role="propagator")["propagators"]}
    assert "__builtin_memset" in prop_syms
    sink_syms = {e["symbol"] for lst in build_catalog(models, role="sink")["sinks_by_class"].values()
                 for e in lst}
    assert "__builtin_memset" in sink_syms


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
    # Top-level: loud, machine-readable "this is a catalog, not findings". A bare
    # model list has not been checked against any binary, so it claims no presence.
    assert cat["presence_catalog"] is False
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
    def __init__(self, name, start, is_thunk=False, imported=False):
        self.name = name
        self.start = start
        self.is_thunk = is_thunk
        # An import stub carries BN's ImportedFunctionSymbol (#958); a plain
        # function carries FunctionSymbol.
        sym_type = "ImportedFunctionSymbol" if imported else "FunctionSymbol"
        self.symbol = type("S", (), {"type": type("T", (), {"name": sym_type})()})()


class _BVTriage:
    """Present sink ``system`` with one real application caller (parse_record) and
    one self-stub site (``system``'s own statically linked body referencing
    itself). Exercises #553 (function per callsite) and #560 (non-audit label /
    audit count). ``imported=True`` turns the body into an import stub (#958)."""
    def __init__(self, imported=False):
        self._app = _FnFull("parse_record", 0x5000)
        self._thunk = _FnFull("system", 0x1000, is_thunk=True, imported=imported)
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


# --- #560: label self-stub callsites, expose audit count --------------------

def test_present_callsites_label_self_stub_560():
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True, "all_sites": True})
    system = _sink_entry(res, "system")
    rows = {c["address"]: c for c in system["callsites"]}
    assert rows["0x2000"]["kind"] == "self_stub"
    # Raw count includes the stub; the audit count excludes it (the real queue).
    assert system["callsite_count"] == 2
    assert system["audit_callsite_count"] == 1


def test_present_callsites_hide_non_audit_rows_by_default_958():
    # #958: the non-audit rows are counted but not listed unless --all-sites, so
    # the callsite list is the work queue an agent walks.
    res = rts._taint_models_op(_CtxWithBV(_BVTriage()), "active",
                               {"present": True, "callsites": True})
    system = _sink_entry(res, "system")
    assert [c["address"] for c in system["callsites"]] == ["0x5010"]
    assert system["callsite_count"] == 2 and system["audit_callsite_count"] == 1
    assert system["hidden_callsites"] == {"self_stub": 1}


def test_import_stub_forwarding_jump_is_not_a_callsite_958():
    # The x86 `.plt.sec` stub's own `jmp [GOT]` is how the import is reached,
    # not a caller of it (AArch64's `br x17` was never counted): leaving it in
    # made the same import's count differ by architecture.
    for all_sites in (False, True):
        res = rts._taint_models_op(_CtxWithBV(_BVTriage(imported=True)), "active",
                                   {"present": True, "callsites": True, "all_sites": all_sites})
        system = _sink_entry(res, "system")
        assert [c["address"] for c in system["callsites"]] == ["0x5010"]
        assert system["callsite_count"] == 1 and system["hidden_callsites"] == {}


class _Llil:
    def __init__(self, op, dest=None, mlil=None):
        self.operation = type("Op", (), {"name": op})()
        self.dest = dest
        self.mlil = mlil


def _const(v):
    c = _Llil("LLIL_CONST_PTR")
    c.constant = v
    return c


class _ILFn(_FnFull):
    def __init__(self, name, start, llil, **kw):
        super().__init__(name, start, **kw)
        self._llil = llil

    def get_llil_at(self, a):
        return self._llil.get(a)


class _BVGotPage:
    """`memcpy` imported through a PLT stub at 0x1000 whose GOT slot is 0x9000,
    the first word of its page. Real calls: `bl 0x1000` in copy_tail and a
    `b 0x1000` tail branch in log_copy, an ordinary one-call wrapper BN flags
    is_thunk. Noise: an unrelated function's `adrp x0, 0x9000` (a code ref to
    the slot because the slot is page-aligned) and the stub's own slot load."""
    def __init__(self):
        self._stub = _ILFn("memcpy", 0x1000, {0x1004: _Llil("LLIL_SET_REG")}, imported=True)
        self._app = _ILFn("copy_tail", 0x5000, {0x5010: _Llil("LLIL_CALL", _const(0x1000))})
        self._wrap = _ILFn("log_copy", 0x6000,
                           {0x6000: _Llil("LLIL_TAILCALL", _const(0x1000))}, is_thunk=True)
        self._other = _ILFn("register_tm_clones", 0x7000, {0x7004: _Llil("LLIL_SET_REG")})
        self.functions = [self._stub, self._app, self._wrap, self._other]
        self._syms = {"memcpy": [_Sym("memcpy", 0x1000), _Sym("memcpy", 0x9000)]}
        self._refs = {0x1000: [_Ref(0x5010), _Ref(0x6000)],
                      0x9000: [_Ref(0x1004), _Ref(0x7004)]}
        self._contain = {0x1004: self._stub, 0x5010: self._app, 0x6000: self._wrap,
                         0x7004: self._other}

    def get_symbols(self):
        return [s for syms in self._syms.values() for s in syms]

    def get_symbols_by_name(self, n):
        return self._syms.get(n, [])

    def get_code_refs(self, a):
        return self._refs.get(a, [])

    def get_functions_containing(self, a):
        f = self._contain.get(a)
        return [f] if f else []

    def get_sections_at(self, a):
        return []


def test_present_callsites_count_calls_not_got_page_references_958():
    res = rts._taint_models_op(_CtxWithBV(_BVGotPage()), "active",
                               {"present": True, "callsites": True, "all_sites": True})
    mc = _sink_entry(res, "memcpy")
    rows = {c["address"]: c["kind"] for c in mc["callsites"]}
    # The adrp of the slot's page and the stub's slot load are data references.
    assert rows == {"0x5010": "app_caller", "0x6000": "app_caller"}
    # A one-call application wrapper's tail branch is an audit site, not a stub.
    assert mc["callsite_count"] == 2 and mc["audit_callsite_count"] == 2


def test_constant_inline_builtin_rows_are_non_audit_958():
    # BN lifts a `str xzr` run to `__builtin_memset(&var, 0, 0x20)`: not a call
    # in LLIL, and its flagged length is a constant, so no taint can arm it.
    zero, length = _const(0), _const(0x20)
    mlil = _Llil("MLIL_CALL", _const(0x9100))
    mlil.params = [_const(0x10), zero, length]
    fn = _ILFn("cat_cmd", 0x5000, {0x5008: _Llil("LLIL_STORE", mlil=mlil)})

    class _BV(_BVGotPage):
        def __init__(self):
            self.functions = [fn]
            self._syms = {"__builtin_memset": [_Sym("__builtin_memset", 0x9100)]}
            self._refs = {0x9100: [_Ref(0x5008)]}
            self._contain = {0x5008: fn}

    res = rts._taint_models_op(_CtxWithBV(_BV()), "active", {"present": True, "callsites": True})
    ms = _sink_entry(res, "__builtin_memset")
    assert ms["callsites"] == [] and ms["hidden_callsites"] == {"inline_builtin": 1}
    # A variable length (`rep movsb` lifted with rcx) stays an audit site.
    mlil.params = [_const(0x10), zero, _Llil("LLIL_REG")]
    res = rts._taint_models_op(_CtxWithBV(_BV()), "active", {"present": True, "callsites": True})
    assert _sink_entry(res, "__builtin_memset")["audit_callsite_count"] == 1


def test_a_rep_movs_lifted_behind_a_direction_flag_branch_is_an_audit_site():
    """V8-06: BN lifts `rep movsq` to `if (flag:d)` plus one intrinsic per arm;
    only the intrinsics carry the synthesised `__builtin_memcpy` calls. The
    catalog judged the first LLIL instruction (the branch), called the site
    data, and dropped it -- the variable-length copy no list showed."""
    def synthesised():
        call = _Llil("MLIL_CALL", _const(0x9100))
        call.params = [_Llil("MLIL_VAR"), _Llil("MLIL_VAR"), _Llil("MLIL_LSL")]
        intrinsic = _Llil("LLIL_INTRINSIC")
        intrinsic.mlils = [call]
        return intrinsic

    lifted = [_Llil("LLIL_IF", mlil=_Llil("MLIL_IF")), synthesised(), synthesised()]
    fn = _ILFn("copy_label", 0x5000, {0x5020: lifted[0]})
    fn.get_llils_at = lambda a: lifted if a == 0x5020 else []

    class _BV(_BVGotPage):
        def __init__(self):
            self.functions = [fn]
            self._syms = {"__builtin_memcpy": [_Sym("__builtin_memcpy", 0x9100)]}
            self._refs = {0x9100: [_Ref(0x5020)]}
            self._contain = {0x5020: fn}

    res = rts._taint_models_op(_CtxWithBV(_BV()), "active", {"present": True, "callsites": True})
    entry = _sink_entry(res, "__builtin_memcpy")
    assert [(c["address"], c["kind"]) for c in entry["callsites"]] == [("0x5020", "app_caller")]
    assert entry["callsite_count"] == entry["audit_callsite_count"] == 1


def test_one_catalog_request_lifts_a_function_once_and_judges_each_site_once(monkeypatch):
    """G4c P2: the catalog fetched `fn.mlil.ssa_form` -- a fresh wrapper Binary
    Ninja may regenerate -- for every synthesised call, and a `rep stos` lifts
    to two, so two fixed-size sites in one function lifted it four times and
    judged each site twice. One request now shares the engine's SSA cache and
    judges each (function, address, argument) once."""
    from types import SimpleNamespace

    lifts = []
    judged = []

    def site(addr):
        def lifted_call():
            call = _Llil("MLIL_CALL", _const(0x9100))
            call.params = [_Llil("MLIL_VAR"), _const(0), _Llil("MLIL_LSL")]
            call.ssa_form = SimpleNamespace(address=addr, params=call.params)
            intrinsic = _Llil("LLIL_INTRINSIC")
            intrinsic.mlils = [call]
            return intrinsic
        return [_Llil("LLIL_IF", mlil=_Llil("MLIL_IF")), lifted_call(), lifted_call()]

    sites = {0x5020: site(0x5020), 0x5060: site(0x5060)}
    fn = _ILFn("reset_records", 0x5000, {a: lifted[0] for a, lifted in sites.items()})
    fn.get_llils_at = lambda a: sites.get(a, [])

    class _MLIL:
        @property
        def ssa_form(self):
            lifts.append(fn.start)
            return SimpleNamespace(source_function=fn)

    fn.mlil = _MLIL()

    def fixed_size(*args):
        judged.append(args)
        return {"intrinsic": "__memfill_u64", "length_bound": "0x138",
                "length_kind": "alignment_split"}

    monkeypatch.setattr(rts._taint, "inline_expansion_length", fixed_size)

    class _BV(_BVGotPage):
        def __init__(self):
            self.functions = [fn]
            self._syms = {"__builtin_memset": [_Sym("__builtin_memset", 0x9100)]}
            self._refs = {0x9100: [_Ref(0x5020), _Ref(0x5060)]}
            self._contain = {0x5020: fn, 0x5060: fn}

    res = rts._taint_models_op(_CtxWithBV(_BV()), "active", {"present": True, "callsites": True})
    entry = _sink_entry(res, "__builtin_memset")
    assert entry["hidden_callsites"] == {"inline_builtin": 2}
    assert lifts == [0x5000]
    assert len(judged) == 2


def test_callsite_rows_name_the_caller_demangled_unless_mangled():
    # #978's rule for text: the containing function by its demangled display
    # name, the raw symbol under --mangled; JSON keeps the raw name.
    from bn.formatters import _render_taint_models_text

    class _BV(_BVGotPage):
        def __init__(self):
            app = _ILFn("_Z9copy_tailPcPKcim", 0x5000,
                        {0x5010: _Llil("LLIL_CALL", _const(0x1000))})
            app.symbol.short_name = "copy_tail"
            self.functions = [app]
            self._syms = {"memcpy": [_Sym("memcpy", 0x1000)]}
            self._refs = {0x1000: [_Ref(0x5010)]}
            self._contain = {0x5010: app}

    res = rts._taint_models_op(_CtxWithBV(_BV()), "active", {"present": True, "callsites": True})
    row = _sink_entry(res, "memcpy")["callsites"][0]
    assert row["function"] == "_Z9copy_tailPcPKcim"
    assert row["function_display_name"] == "copy_tail"
    assert "0x5010  copy_tail" in _render_taint_models_text(res)
    assert "0x5010  _Z9copy_tailPcPKcim" in _render_taint_models_text(res, mangled=True)


def test_source_callsites_and_bounded_write_args_reach_the_text_958():
    from bn.formatters import _render_taint_models_text

    class _BV(_BVGotPage):
        def __init__(self):
            app = _ILFn("main", 0x5000, {0x5010: _Llil("LLIL_CALL", _const(0x1000))})
            self.functions = [app]
            self._syms = {"read": [_Sym("read", 0x1000)]}
            self._refs = {0x1000: [_Ref(0x5010)]}
            self._contain = {0x5010: app}

    res = rts._taint_models_op(_CtxWithBV(_BV()), "active",
                               {"present": True, "callsites": True})
    text = _render_taint_models_text(res)
    src = text.split("sources")[1].split("sinks")[0]
    assert "read  ->  *arg:1, ret [present] (1 callsite)" in src
    assert "0x5010  main" in src
    # read/recv arm through len_arg/buf_arg; the line used to say `arg []`.
    assert "read (len arg 2 into arg 1) [present]" in text


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
                               {"present": True, "callsites": True, "all_sites": True})
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
                               {"present": True, "callsites": True, "all_sites": True})
    text = _render_taint_models_text(res)
    assert "0x5010  parse_record  call    system" in text
    assert "0x2000  system [self_stub]  jmp     qword [rip+0x2f1a]" in text


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
                               {"present": True, "callsites": True, "all_sites": True})
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


def test_taint_op_refuses_a_supplied_null_resolve_map_1050():
    # #1050 review round 2: `--resolve-map` naming a file that holds JSON
    # `null` reaches the bridge as a present `resolve_map: None`. That is a
    # supplied document of the wrong shape, refused with the file name -- not
    # the omitted option, which the same request without the key still is.
    import pytest

    class _Ctx(_CtxWithBV):
        def _find_function(self, bv, name):
            return object()

    request = {"function": "handler", "sources": ["param:0"]}
    with pytest.raises(rts.OperationFailure, match="--resolve-map maps/null.json"):
        rts._taint_op(_Ctx(_BVPlain()), "active",
                      dict(request, resolve_map=None, resolve_map_path="maps/null.json"))


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
                               {"present": True, "callsites": True, "all_sites": True})
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


@pytest.mark.parametrize("spellings,label", [
    (("memcpy", "memcpy@plt"), "present"),
    (("memcpy@plt",), "present as memcpy@plt"),
])
def test_a_presence_row_names_the_spelling_the_bridge_selected(spellings, label):
    """R4-03: with both `memcpy` and `memcpy@plt` in the image the bridge
    selects the exact `memcpy` (`resolved_symbol`), but the text row read
    `accepted_aliases` and printed `[present as memcpy@plt]`, sending the
    `xrefs`/`callsites` follow-up to the stub. The label follows the selection;
    JSON keeps every spelling."""
    from bn.formatters import _render_taint_models_text

    bv = _BVSpellings()
    bv.functions = [_FakeFn(name) for name in spellings]
    bv._syms = {name: syms for name, syms in bv._syms.items() if name in spellings}
    res = rts._taint_models_op(_CtxWithBV(bv), "active", {"present": True})
    assert set(_sink_entry(res, "memcpy")["accepted_aliases"]) == set(spellings)
    [row] = [line for line in _render_taint_models_text(res).splitlines()
             if line.strip().startswith("memcpy (")]
    assert re.findall(r"\[(present[^\]]*)\]", row) == [label]


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
                               {"present": True, "callsites": True, "all_sites": True})
    text = _render_taint_models_text(res)
    assert "NOT taint findings" in text
    assert "parse_record" in text                         # #553 function context
    assert "[self_stub]" in text                          # #560 non-audit label
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


def _counted(F, expr):
    """*expr* read where `if (n <= 0) return;` dominates it: BN's value set is
    ``1..0x40``, so the receive's return is a byte count, never its -1
    failure (an unchecked return bounds nothing, V7-01)."""
    expr.size = 8
    expr.possible_values = F.FPVS("SignedRangeValue")
    expr.possible_values.ranges = [F.types.SimpleNamespace(start=1, end=0x40, step=1)]
    return expr


def _bounded_receive_program(callee):
    """``n = read(3, &src, 0x40); if (n <= 0) return; callee(3, &dst, n)``.

    The #159 idiom `_comment_recv_overflow` itself names as the ~100%-false-
    positive one: the length is the checked return of a modeled receive
    provably bounded by a constant count, so the honest verdict is
    `bounded_len` with that bound, not an overflow.
    """
    F = _fakes()
    n1 = F.FSSA(F.FVar("n"), 1)
    instrs = [
        F.FInstr(0, 0x10, "MLIL_CALL_SSA", "n#1 = read(3, &src, 0x40)", writes=[n1],
                 dest=F.FExpr("MLIL_CONST_PTR", "0x900", constant=0x900),
                 params=[F.FExpr("MLIL_CONST", "3", constant=3),
                         F.FExpr("MLIL_ADDRESS_OF", "&src", src=F.FVar("src", typ="char[0x40]")),
                         F.FExpr("MLIL_CONST", "0x40", constant=0x40)]),
        _call_under_test(F, callee, 1, _counted(F, F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])),
                         [n1],
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
    """``n = callee(3, &buf, 0x40); if (n <= 0) return; memcpy(&dst, &buf, n);
    system(&dst)``.

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
                         _counted(F, F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1]))]),
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
    """``n = read(3, &src, 0x40); if (n <= 0) return; callee(&dst, &src, n[, guard])``.

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
              _counted(F, F.FExpr("MLIL_VAR_SSA", "n#1", reads=[n1]))]
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
    from bn_agent_bridge.taint_engine import load_models, lookup_model
    models = load_models()
    for name in ("scanf", "__isoc99_scanf", "fscanf", "__isoc99_fscanf"):
        assert lookup_model(models, name)[1].get("arity_capped") is True, (
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
        assert lookup_model(models, name)[1].get("arity_capped") is True, (
            f"{name} propagates into its destinations and its run is unrolled "
            "like scanf's, so it must carry arity_capped=true"
        )


@pytest.mark.parametrize("spelling,canonical", [
    (f"__isoc{std}_{base}", base)
    for std in ("99", "23") for base in ("scanf", "fscanf", "sscanf", "vsscanf")
] + [
    # The decorations lookup_model already handled must keep working through
    # the alias: a PLT-imported ISO spelling lands on the same model.
    ("__isoc23_sscanf@plt", "sscanf"),
    ("__isoc99_sscanf@plt", "sscanf"),
    ("_isoc23_sscanf", "sscanf"),
])
def test_glibc_iso_scanf_spellings_alias_the_canonical_models(spelling, canonical):
    # glibc exports __isoc99_* (C99) and __isoc23_* (C23, glibc 2.38+) for the
    # scanf family. Each is the same call -- same arguments, same destinations
    # -- so it must resolve to the canonical model by the alias table, not to a
    # hand-maintained duplicate that can drift in source, propagation, sink
    # class or arity cap.
    from bn_agent_bridge.taint_engine import load_models, lookup_model
    models = load_models()
    key, model = lookup_model(models, spelling)
    assert key == canonical
    assert model is models[canonical], (
        f"{spelling} must share the {canonical} model object, not a copy"
    )
    # The semantics the alias inherits, pinned on the model it shares.
    if canonical in ("scanf", "fscanf", "sscanf"):
        assert model.get("arity_capped") is True
        assert model["sink"]["class"] == "overflow_unbounded"
        assert model["sink"].get("format_arg") is not None
    else:
        assert model.get("destinations_unmodeled") is True


def test_iso_scanf_aliases_are_not_catalog_entries_of_their_own():
    # The catalog is a catalog of MODELS, keyed by the canonical name the
    # engine reports (and --present annotates). An alias is a way to reach
    # that model, so neither ISO spelling may appear as a second row that
    # could describe different semantics.
    from bn_agent_bridge.taint_engine import load_models
    models = load_models()
    cat = build_catalog(models)
    symbols = {e["symbol"] for e in cat["sources"]}
    symbols |= {e["symbol"] for e in cat["propagators"]}
    symbols |= {e["symbol"] for lst in cat["sinks_by_class"].values() for e in lst}
    assert not any(s.startswith(("__isoc23_", "__isoc99_")) for s in symbols), symbols
    assert {"scanf", "fscanf", "sscanf"} <= symbols


def test_c23_scanf_spellings_share_the_variadic_format_recovery():
    # The read-side format analysis normalizes the same glibc wrappers: the
    # C23 spelling must find the format argument exactly where the legacy
    # spelling does, or the model knows the callee but the call evidence and
    # listing do not know where its format is.
    from bn_agent_bridge import il_format
    for spelling, fmt_idx in (("__isoc23_scanf", 0), ("__isoc23_fscanf", 1),
                              ("__isoc23_sscanf", 1)):
        assert il_format._variadic_format_family(spelling) == (fmt_idx, True)
    assert il_format._variadic_format_family("__isoc23_vsscanf") is None  # va_list, not variadic
    assert il_format._normalize_libc_name("__isoc23_sscanf@plt") == "sscanf"


def test_scanf_arity_residual_marker_in_weak_seed_set_851():
    # #851: the engine emits an assumption containing "scanf_arity_residual"
    # when a call has more params than the model covers; taint_result must
    # recognise that marker to withhold the all-clear.
    from bn_agent_bridge.taint_result import _WEAK_SEED_ASSUMPTION_MARKERS
    assert "scanf_arity_residual" in _WEAK_SEED_ASSUMPTION_MARKERS, (
        "scanf_arity_residual must be in _WEAK_SEED_ASSUMPTION_MARKERS so a "
        "residual-arity assumption withholds safe_to_report_all_clear"
    )


# --- one argument grammar for every model role (IC04 review) ------------------

from bn_agent_bridge import taint_models as tm

_BAD_TOKENS = ["arg:-1", "arg:+1", "arg:01", "arg: 1", "arg:1 ", "arg:", "arg:x", "arg:1.0",
               "arg:256", "arg:99999999999999999999", "arg:١", "ARG:1", "argv:1", ""]


def _ptr(tok: str) -> str:
    return "*" + tok if tok.startswith("arg") else tok


@pytest.mark.parametrize("tok", _BAD_TOKENS)
@pytest.mark.parametrize("field,make", [
    ("propagates.from", lambda t: {"propagates": [{"from": t, "to": "ret"}]}),
    ("propagates.to", lambda t: {"propagates": [{"from": "arg:0", "to": t}]}),
    ("propagates.to", lambda t: {"propagates": [{"from": "arg:0", "to": _ptr(t)}]}),
    ("sources.to", lambda t: {"sources": [{"to": _ptr(t)}]}),
    ("varargs.to", lambda t: {"varargs": {"first_index": 1, "to": _ptr(t)}}),
    ("writes", lambda t: {"writes": [_ptr(t)]}),
    ("reads_dest", lambda t: {"propagates": [{"from": "*arg:1", "to": "*arg:0"}],
                              "reads_dest": [_ptr(t)]}),
])
def test_every_role_refuses_a_malformed_argument_token(tok, field, make):
    with pytest.raises(tm.TaintError) as ei:
        tm.load_models(extra={"app_fn": make(tok)})
    msg = str(ei.value)
    assert "model 'app_fn'" in msg and f"`{field}" in msg, msg
    assert "non-negative" in msg, msg


@pytest.mark.parametrize("bad", [-1, 256, 10**20, True, "1", 1.0])
@pytest.mark.parametrize("field,make", [
    ("sink.tainted_args", lambda v: {"sink": {"class": "c", "tainted_args": [v]}}),
    ("sink.len_arg", lambda v: {"sink": {"class": "c", "len_arg": v}}),
    ("sink.buf_arg", lambda v: {"sink": {"class": "c", "tainted_args": [0], "buf_arg": v}}),
    ("sink.pointer_args", lambda v: {"sink": {"class": "c", "tainted_args": [0], "pointer_args": [v]}}),
    ("sink.format_arg", lambda v: {"sink": {"class": "c", "format_arg": v, "first_dest": 300}}),
    ("varargs.first_index", lambda v: {"varargs": {"first_index": v, "to": "*arg:0"}}),
    ("continuation.null_arg", lambda v: {"propagates": [{"from": "*arg:0", "to": "ret"}],
                                         "continuation": {"null_arg": v}}),
    ("string_args", lambda v: {"sink": {"class": "c", "tainted_args": [0]}, "string_args": [v]}),
    ("append.dest", lambda v: {"propagates": [{"from": "*arg:1", "to": "*arg:0"}],
                               "append": {"dest": v, "src": 1}}),
    ("ret_offset.count_arg", lambda v: {"propagates": [{"from": "arg:0", "to": "ret"}],
                                        "ret_offset": {"count_arg": v}}),
])
def test_every_index_field_refuses_a_non_index(bad, field, make):
    model = make(bad)
    with pytest.raises(tm.TaintError) as ei:
        tm.load_models(extra={"app_fn": model})
    msg = str(ei.value)
    assert "model 'app_fn'" in msg and field.split(".")[0] in msg, msg


def test_return_bound_refuses_a_negative_index():
    with pytest.raises(tm.TaintError, match="max_from_arg"):
        tm.load_models(extra={"read": {"return_bound": {"max_from_arg": -1}}})


def test_the_grammar_accepts_every_documented_form():
    ok = tm.load_models(extra={
        "app_copy": {"propagates": [{"from": "*arg:1", "to": "*arg:0"},
                                    {"from": "arg:255", "to": "arg:0"},
                                    {"from": "arg:0", "to": "ret"},
                                    {"from": "*arg:0", "to": "*ret"}],
                     "reads_dest": ["*arg:0"], "string_args": [0, 1],
                     "append": {"dest": 0, "src": 1}, "ret_offset": {"count_arg": 2}},
        "app_recv": {"sources": [{"to": "*arg:1"}, {"to": "arg:0"}, {"to": "ret"},
                                 {"to": "*ret"}, {"to": "*iovec:1"}]},
        "app_fmt": {"varargs": {"first_index": 2, "to": "*arg:0"}},
        "app_tok": {"propagates": [{"from": "*arg:0", "to": "ret"}],
                    "continuation": {"null_arg": 0}},
        "app_zero": {"writes": ["*arg:0"]},
    })
    assert ok["app_copy"]["propagates"][1]["from"] == "arg:255"


def test_builtin_catalog_satisfies_the_grammar():
    # The packaged DB is validated on every load; the engine's own guard
    # accepts the merged map too.
    models = tm.load_models()
    tm.validate_model_arguments(models, source="builtin")
    tokens = [r[k] for m in models.values() if isinstance(m, dict)
              for role in ("sources", "propagates") for r in m.get(role) or [] for k in ("from", "to")
              if k in r]
    assert tokens and all(t in ("ret", "*ret") or tm.parse_model_token(t) for t in tokens)


def test_catalog_op_refuses_a_malformed_user_model():
    # `bn taint models --models FILE` is the documented overlay check: it
    # reports the malformed token instead of listing it.
    with pytest.raises(tm.TaintError, match=r"`propagates.to` is '\*arg:-1'"):
        tm.load_models(extra={"models": {"app_fn": {"propagates": [{"from": "arg:0", "to": "*arg:-1"}]}}})


# --- required indices are never null (round-2 review) --------------------------

@pytest.mark.parametrize("first_index", ["<missing>", None, True, False, "2", "+2", -1, 256, 2.0])
def test_varargs_first_index_is_required_and_canonical(first_index):
    va = {"to": "*arg:0"}
    if first_index != "<missing>":
        va["first_index"] = first_index
    with pytest.raises(tm.TaintError) as ei:
        tm.load_models(extra={"app_fmt": {"varargs": va}})
    msg = str(ei.value)
    assert "model 'app_fmt'" in msg and "`varargs.first_index` is required" in msg, msg


@pytest.mark.parametrize("field,model", [
    ("continuation.null_arg", {"propagates": [{"from": "*arg:0", "to": "ret"}],
                               "continuation": {"null_arg": None}}),
    ("max_from_arg", {"propagates": [{"from": "*arg:0", "to": "ret"}],
                      "return_bound": {"max_from_arg": None}}),
    ("first_dest", {"sink": {"class": "c", "format_arg": 1, "first_dest": None}}),
    ("append.dest", {"propagates": [{"from": "*arg:1", "to": "*arg:0"}],
                     "append": {"dest": None, "src": 1}}),
    ("ret_offset.count_arg", {"propagates": [{"from": "arg:0", "to": "ret"}],
                              "ret_offset": {"count_arg": None}}),
])
def test_a_present_required_index_is_never_null(field, model):
    with pytest.raises(tm.TaintError, match=field.replace(".", r"\.")):
        tm.load_models(extra={"app_fn": model})


def test_optional_sink_indices_may_be_null():
    # `len_arg`, `buf_arg`, `format_arg` and `pointer_args` are optional; the
    # engine reads a null as absent, so null stays accepted.
    ok = tm.load_models(extra={"app_send": {"sink": {
        "class": "net_write", "tainted_args": [1], "len_arg": None, "buf_arg": None,
        "format_arg": None, "pointer_args": None}}})
    assert ok["app_send"]["sink"]["len_arg"] is None


# --- P2-09: a model file is accepted only as written ----------------------------

_APP_COPY = {"propagates": [{"from": "*arg:1", "to": "*arg:0"}]}


def test_an_empty_top_level_list_is_refused_like_a_non_empty_one():
    # `[]` was falsy, so the loader skipped it and the disclosure omitted the
    # file: a wrong-shape document accepted at exit 0.
    for doc in ([], [{"app_copy": _APP_COPY}]):
        with pytest.raises(tm.TaintError, match="must be a JSON object"):
            tm.load_models(extra=doc, extra_via="--models", extra_path="/proj/m.json")


def test_an_empty_object_is_a_disclosed_file_of_no_models():
    assert tm.load_models(extra={}) == tm.load_models()
    rows = tm.model_overlay_sources({}, user_models_path="/proj/m.json")
    assert {"kind": "user", "via": "--models", "count": 0, "path": "/proj/m.json"} in rows


@pytest.mark.parametrize("op", ["forward", "backward", "models"])
@pytest.mark.parametrize("doc", [[], None])
def test_bridge_refuses_a_non_object_model_file(op, doc):
    class _Ctx(_CtxWithBV):
        def _find_function(self, bv, name):
            return object()

    params = {"user_models": doc, "user_models_path": "/proj/m.json",
              "user_models_via": "--models"}
    with pytest.raises(rts.OperationFailure, match=r"/proj/m\.json.*must be a JSON object"):
        if op == "models":
            rts._taint_models_op(_CtxNoView(), None, params)
        else:
            rts._taint_op(_Ctx(_BVPlain()), "active", dict(
                params, function="handler", direction=op,
                sources=["param:0"], sinks=["arg:memcpy:2"]))


@pytest.mark.parametrize("model, path", [
    ({"sink": {"class": "alloc_size", "tainted_args": [1]},
      "OTHER_ROLES": "COPY_FROM_THE_CURRENT_MODEL"}, "OTHER_ROLES"),
    ({"sink": {"clas": "alloc_size", "class": "alloc_size", "tainted_args": [1]}}, "sink.clas"),
    ({"propagates": [{"from": "*arg:1", "to": "*arg:0", "form": "arg:1"}]}, "propagates[0].form"),
    ({"sources": [{"to": "ret"}, {"to": "*arg:0", "kind": "x"}]}, "sources[1].kind"),
    ({"varargs": {"first_index": 1, "to": "*arg:0", "first": 1}}, "varargs.first"),
    ({**_APP_COPY, "continuation": {"null_arg": 0, "nul_arg": 0}}, "continuation.nul_arg"),
    ({**_APP_COPY, "return_bound": {"max_from_arg": 2, "min_from_arg": 0}}, "return_bound.min_from_arg"),
    ({**_APP_COPY, "append": {"dest": 0, "src": 1, "len": 2}}, "append.len"),
    ({**_APP_COPY, "ret_offset": {"count_arg": 2, "cnt_arg": 2}}, "ret_offset.cnt_arg"),
])
def test_unknown_keys_are_refused_at_every_level_naming_the_path(model, path):
    with pytest.raises(tm.TaintError) as err:
        tm.load_models(extra={"ds_resize": model})
    msg = str(err.value)
    assert "model 'ds_resize'" in msg and f"`{path}`" in msg, msg


def test_comment_and_prose_keys_stay_allowed_at_every_level():
    ok = tm.load_models(extra={"app_copy": {
        "_comment": "wrapper", "detail": "a project copy wrapper",
        "propagates": [{"from": "*arg:1", "to": "*arg:0", "_comment_rule": "src->dst"}],
        "sink": {"class": "overflow_len", "tainted_args": [2], "detail": "n", "_comment": "x",
                 "dest_size_arg": 3}}})
    assert ok["app_copy"]["sink"]["dest_size_arg"] == 3


def test_dest_size_arg_is_an_argument_index():
    with pytest.raises(tm.TaintError, match=r"`sink\.dest_size_arg`"):
        tm.load_models(extra={"app_copy": {"sink": {
            "class": "overflow_len", "tainted_args": [2], "dest_size_arg": -1}}})


def test_a_typoed_sink_class_is_refused_listing_the_valid_set():
    valid = tm.builtin_sink_classes()
    assert {"overflow_len", "alloc_size", "file_write", "net_write"} <= valid
    with pytest.raises(tm.TaintError) as err:
        tm.load_models(extra={"app_copy": {**_APP_COPY, "sink": {
            "class": "overflow_unbounde", "tainted_args": [1]}}})
    msg = str(err.value)
    assert "model 'app_copy'" in msg and "overflow_unbounde" in msg, msg
    assert all(c in msg for c in valid), msg


@pytest.mark.parametrize("target", [False, True])
def test_catalog_refuses_an_unknown_class_filter(target):
    ctx = _CtxWithBV(_FakeBV(["memcpy"])) if target else _CtxNoView()
    with pytest.raises(rts.OperationFailure) as err:
        rts._taint_models_op(ctx, "active" if target else None, {"class": "nonexistent_class"})
    msg = str(err.value)
    assert "nonexistent_class" in msg and all(c in msg for c in tm.builtin_sink_classes()), msg


# --- address-keyed models resolve against the target ---------------------------

class _FnAt:
    def __init__(self, name, start):
        self.name, self.start = name, start


class _BVWithFunctions(_FakeBV):
    """A stripped view: functions at 0x401000 (auto-named) and 0x402000."""
    def __init__(self):
        super().__init__(["sub_401000", "parse_frame"])
        self._at = {0x401000: _FnAt("sub_401000", 0x401000),
                    0x402000: _FnAt("parse_frame", 0x402000)}

    def get_function_at(self, addr):
        return self._at.get(addr)


_ADDR_MODEL = {"0x401000": {"sources": [{"to": "*arg:1"}, {"to": "ret"}]}}


def _taint_with(bv, user_models):
    seen: dict = {}

    class _RecordingEngine(rts._taint.TaintEngine):
        def __init__(self, bv, models, **kw):
            super().__init__(bv, models, **kw)
            seen["models"] = models

        def forward(self, func, locators, **kw):
            return {"direction": "forward", "reached_sinks": [], "leaves": []}

    class _Ctx(_CtxWithBV):
        def _find_function(self, bv, name):
            return object()

    import unittest.mock as um
    with um.patch.object(rts._taint, "TaintEngine", _RecordingEngine):
        result = rts._taint_op(_Ctx(bv), "active", {
            "function": "handler", "sources": ["param:0"], "user_models": user_models,
            "user_models_path": "/proj/m.json", "user_models_via": "--models"})
    return result, seen["models"]


def test_address_keyed_model_applies_under_the_functions_name_and_is_disclosed():
    result, models = _taint_with(_BVWithFunctions(), _ADDR_MODEL)
    assert models["sub_401000"] == _ADDR_MODEL["0x401000"]
    assert "0x401000" not in models
    [user] = [s for s in result["model_sources"] if s["kind"] == "user"]
    assert user["resolved"] == {"0x401000": "sub_401000"} and user["count"] == 1, user
    # A named function resolves to its current name; the envelope form too.
    _, models = _taint_with(_BVWithFunctions(), {"models": {"0x402000": _APP_COPY}})
    assert models["parse_frame"] == _APP_COPY


def test_catalog_with_a_target_resolves_address_keys():
    res = rts._taint_models_op(_CtxWithBV(_BVWithFunctions()), "active", {
        "user_models": _ADDR_MODEL, "user_models_path": "/proj/m.json",
        "user_models_via": "--models", "role": "source"})
    assert "sub_401000" in {s["symbol"] for s in res["sources"]}
    [user] = [o for o in res["overlays"] if o["kind"] == "user"]
    assert user["resolved"] == {"0x401000": "sub_401000"}, user


@pytest.mark.parametrize("doc, needle", [
    ({"0x401004": _APP_COPY}, "0x401004"),                          # not a function start
    ({"0x401000": _APP_COPY, "sub_401000": _APP_COPY}, "sub_401000"),  # name collision
    ({"0x401000": _APP_COPY, "0x0401000": _APP_COPY}, "0x0401000"),    # two spellings
])
def test_unresolvable_or_colliding_address_keys_are_refused(doc, needle):
    with pytest.raises(rts.OperationFailure) as err:
        _taint_with(_BVWithFunctions(), doc)
    assert needle in str(err.value) and "/proj/m.json" in str(err.value), str(err.value)


def test_address_keyed_model_without_a_target_is_refused():
    with pytest.raises(rts.OperationFailure, match="0x401000.*need a target"):
        rts._taint_models_op(_CtxNoView(), None, {
            "user_models": _ADDR_MODEL, "user_models_path": "/proj/m.json",
            "user_models_via": "--models"})
    with pytest.raises(tm.TaintError, match="0x401000.*need a target"):
        tm.load_models(extra=_ADDR_MODEL)
