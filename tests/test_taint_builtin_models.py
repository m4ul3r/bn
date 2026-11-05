"""Builtin-catalog papercuts (W10-01, W10-08, W02-06).

A copy sized by a regex match offset never reached its sink because no match
API was modeled; a sink a `--models` overlay declared reported `detail: null`;
the fortified `__*_chk` destination size had no role, so slicing it walked a
compile-time constant as a pointer.
"""
from __future__ import annotations

import pytest
from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, _ext_call, te

RE, MEMCPY, CHK, APP = 0x900, 0x901, 0x902, 0x903


def _match_sized_copy(api: str = "regexec", subject: int = 1, offsets: int = 3,
                      arity: int = 5):
    """get_field(s): <api>(..., s at `subject`, ..., &m at `offsets`, ...);
    n = m; memcpy(&d, s, n) -- the copy length is a match offset the match API
    wrote into `m` (regexec: `regexec(&re, s#0, 3, &m, 0)`)."""
    s, m, n, d = FVar("s"), FVar("m", typ="regmatch_t[3]"), FVar("n"), FVar("d", typ="char[0x40]")
    s0, m1, n1 = FSSA(s, 0), FSSA(m, 1), FSSA(n, 1)
    args = [FExpr("MLIL_CONST", hex(0x5000 + i), constant=0x5000 + i) for i in range(arity)]
    args[subject] = FExpr("MLIL_VAR_SSA", "s#0", reads=[s0])
    args[offsets] = FExpr("MLIL_ADDRESS_OF", "&m", src=m)
    instrs = [
        _ext_call(0, 0x10, f"{api}(.., s#0, .., &m, ..)", RE, args, reads=[s0]),
        FInstr(1, 0x14, "MLIL_SET_VAR_SSA", "n#1 = m @ mem#1", reads=[m1], writes=[n1],
               src=FExpr("MLIL_VAR_ALIASED", "m @ mem#1", reads=[m1], src=m1)),
        _ext_call(2, 0x18, "memcpy(&d, s#0, n#1)", MEMCPY,
                  [FExpr("MLIL_ADDRESS_OF", "&d", src=d), FExpr("MLIL_VAR_SSA", "s#0", reads=[s0]),
                   FExpr("MLIL_VAR_SSA", "n#1", reads=[n1])], reads=[s0, n1]),
        FInstr(3, 0x1c, "MLIL_RET", "return"),
    ]
    return FFunc("get_field", 0x10, FSSAFunc(instrs), params=[s]), FBV({RE: api,
                                                                      MEMCPY: "memcpy"})


def _sinks(result):
    return [(f["sink"]["callee"], f["sink"]["class"], f["sink"]["tainted_arg_index"])
            for f in result["reached_sinks"]]


def test_a_copy_sized_by_a_regex_match_offset_reaches_its_sink_without_an_overlay():
    func, bv = _match_sized_copy()
    builtin = te.TaintEngine(bv, te.load_models(), unknown_call_policy="stop").forward(
        func, [te.parse_locator("param:0")])
    overlay = te.TaintEngine(bv, te.load_models(
        {"regexec": {"propagates": [{"from": "*arg:1", "to": "*arg:3"}]}}),
        unknown_call_policy="stop").forward(func, [te.parse_locator("param:0")])
    assert _sinks(builtin) == [("memcpy", "overflow_len", 2)]
    # The one-line overlay the audit needed adds nothing any more.
    assert _sinks(overlay) == _sinks(builtin)


def test_every_regex_match_api_names_its_offset_array_from_its_subject():
    models = te.load_models()
    for name, subject, offsets in (("regexec", 1, 3), ("re_search", 1, 5), ("re_match", 1, 4),
                                   ("pcre_exec", 2, 6), ("pcre2_match_8", 1, 5)):
        rules = {(r["from"], r["to"]) for r in models[name]["propagates"]}
        assert (f"*arg:{subject}", f"*arg:{offsets}") in rules, name


@pytest.mark.parametrize(("api", "subject", "offsets", "arity"), [
    ("re_search", 1, 5, 6), ("re_match", 1, 4, 5), ("pcre_exec", 2, 6, 8),
    ("pcre2_match_8", 1, 5, 7),
])
def test_each_regex_match_api_carries_its_subject_into_the_offsets_it_writes(
        api, subject, offsets, arity):
    """The rule table above, run: the copy sized by an offset a non-`regexec`
    match API wrote reaches its sink from the subject alone, as the
    `regexec` run does -- and does not when that API's propagation is
    overlaid away, so the reach is the builtin rule's doing."""
    func, bv = _match_sized_copy(api, subject, offsets, arity)
    builtin = te.TaintEngine(bv, te.load_models(), unknown_call_policy="stop").forward(
        func, [te.parse_locator("param:0")])
    assert ("memcpy", "overflow_len", 2) in _sinks(builtin)
    inert = te.TaintEngine(bv, te.load_models({api: {}}), unknown_call_policy="stop").forward(
        func, [te.parse_locator("param:0")])
    assert ("memcpy", "overflow_len", 2) not in _sinks(inert)


def test_a_sink_from_a_models_overlay_carries_a_detail_like_a_builtin():
    n, d = FVar("n"), FVar("d", typ="char[0x40]")
    n0 = FSSA(n, 0)
    instrs = [_ext_call(0, 0x10, "app_copy(&d, 0x6000, n#0)", APP,
                        [FExpr("MLIL_ADDRESS_OF", "&d", src=d),
                         FExpr("MLIL_CONST_PTR", "0x6000", constant=0x6000),
                         FExpr("MLIL_VAR_SSA", "n#0", reads=[n0])], reads=[n0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    func = FFunc("relay", 0x10, FSSAFunc(instrs), params=[n])
    models = te.load_models({"app_copy": {"sink": {"class": "overflow_len", "tainted_args": [2]}}},
                            extra_via="--models", extra_path="/proj/wrappers.json")
    result = te.TaintEngine(FBV({APP: "app_copy"}), models).forward(
        func, [te.parse_locator("param:0")])
    [finding] = result["reached_sinks"]
    detail = finding["sink"]["detail"]
    assert detail and "app_copy" in detail and "/proj/wrappers.json" in detail, finding["sink"]
    # A sentence the overlay's author wrote is kept as written.
    own = te.load_models({"app_copy": {"sink": {"class": "overflow_len", "tainted_args": [2],
                                                "detail": "length into app_copy"}}})
    assert own["app_copy"]["sink"]["detail"] == "length into app_copy"


def test_a_fortified_destination_size_is_a_value_not_a_pointer_demand():
    # W02-06: `__strcpy_chk(&d, s, 0x120)` -- slicing the guard argument used
    # to walk 0x120 as a pointer the sink reads through and describe it as a
    # read-only constant address. It is the destination's size: a constant
    # there is a `constant` sink site, and no role-unknown leaf is raised.
    s, d = FVar("s"), FVar("d", typ="char[0x120]")
    s0 = FSSA(s, 0)
    instrs = [_ext_call(0, 0x10, "__strcpy_chk(&d, s#0, 0x120)", CHK,
                        [FExpr("MLIL_ADDRESS_OF", "&d", src=d),
                         FExpr("MLIL_VAR_SSA", "s#0", reads=[s0]),
                         FExpr("MLIL_CONST", "0x120", constant=0x120)], reads=[s0]),
              FInstr(1, 0x14, "MLIL_RET", "return")]
    func = FFunc("save_name", 0x10, FSSAFunc(instrs), params=[s])
    result = te.TaintEngine(FBV({CHK: "__strcpy_chk"}), te.load_models()).backward(
        func, [te.parse_locator("arg:__strcpy_chk:2")])
    [row] = result["sink_status"]
    assert row["status"] == "constant", row
    assert row["scalar"]["value"] == 0x120
    assert not [lf for lf in result["leaves"]
                if lf["kind"] in ("sink_role_unknown", "pointer_demand_not_certified")], \
        result["leaves"]
