"""Contract tests for the shared fakes' AUTO-vs-USER provenance model.

Every expectation here was probed against a live BN 5.4 install (`bn.load`
of a stock system binary) before being encoded, so the fake is faithful
rather than forgiving:

    create_user_var  -> is_var_user_defined() True, name applied
    create_auto_var  -> value applied now, but a USER override is restored
                        once analysis settles
    delete_user_var  -> provenance clears immediately, the analysis-DERIVED
                        name only re-derives once analysis settles
    Function.type=   -> parses a string prototype through the view, RENAMES the
                        function, and routes to set_user_type (has_user_type True)
    set_auto_type    -> value only, has_user_type untouched
    undo             -> restores local var value AND provenance,
                        but does NOT clear has_user_type
"""

from __future__ import annotations

import types

from _bridge_fakes import _FakeFunction, _FakeMutationBV, _FakeVariable


def _fn_with_var(name: str = "var_8", *, storage: int = -8, identifier: int = 1):
    fn = _FakeFunction(0x1000, "sub_1000")
    var = _FakeVariable(name=name, storage=storage, var_type="int32_t", identifier=identifier)
    fn.stack_layout = [var]
    return fn, var


def _mutation_bv(fn):
    bv = _FakeMutationBV(functions=[fn])
    fn.view = bv
    return bv


# --- baseline -------------------------------------------------------------

def test_fresh_function_and_vars_are_auto():
    fn, var = _fn_with_var()
    assert fn.has_user_type is False
    assert fn.is_var_user_defined(var) is False


# --- local variable provenance -------------------------------------------

def test_create_user_var_applies_value_and_marks_user():
    fn, var = _fn_with_var()
    fn.create_user_var(var, "char*", "probe_name")
    assert (var.name, var.type) == ("probe_name", "char*")
    assert fn.is_var_user_defined(var) is True


def test_create_auto_var_applies_value_without_user_provenance():
    # Negative control: the AUTO writer must not over-report as USER.
    fn, var = _fn_with_var()
    fn.create_auto_var(var, "char*", "auto_name")
    assert (var.name, var.type) == ("auto_name", "char*")
    assert fn.is_var_user_defined(var) is False


def test_delete_user_var_clears_provenance_then_reanalysis_restores_auto_value():
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)
    fn.create_user_var(var, "char*", "probe_name")

    fn.delete_user_var(var)
    # Live BN: provenance drops at once, the AUTO name is only re-derived by
    # the next analysis pass -- the fake must not restore it early.
    assert fn.is_var_user_defined(var) is False
    assert var.name == "probe_name"

    bv.update_analysis_and_wait()
    assert (var.name, var.type) == ("var_8", "int32_t")
    assert fn.is_var_user_defined(var) is False


def test_delete_user_var_on_auto_var_is_a_noop():
    # Negative control: deleting a user override that never existed must not
    # rewrite the variable or invent a baseline.
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)
    fn.create_auto_var(var, "int64_t", "renamed_by_analysis")

    fn.delete_user_var(var)
    bv.update_analysis_and_wait()

    assert (var.name, var.type) == ("renamed_by_analysis", "int64_t")
    assert fn.is_var_user_defined(var) is False


def test_var_provenance_is_keyed_per_variable():
    fn, first = _fn_with_var("var_8", storage=-8, identifier=1)
    second = _FakeVariable(name="var_10", storage=-0x10, var_type="int32_t", identifier=2)
    fn.stack_layout.append(second)

    fn.create_user_var(first, "char*", "named")

    assert fn.is_var_user_defined(first) is True
    assert fn.is_var_user_defined(second) is False


# --- prototype provenance -------------------------------------------------

def test_type_setter_routes_to_set_user_type_and_renames_the_function():
    # BN 5.4 function.py L1207-1214: the string branch parses the prototype
    # through the view and assigns `self.name = str(new_name)` BEFORE
    # set_user_type -- a prototype write is also a rename (verified live: a
    # function named `_init` came back as `renamed_fn`).
    fn, _ = _fn_with_var()
    _mutation_bv(fn)
    fn.type = "uint64_t probe_fn(int32_t a)"
    assert str(fn.type) == "uint64_t(int32_t a)"   # parsed type carries no name
    assert fn.name == "probe_fn"
    assert fn.has_user_type is True


def test_type_setter_with_anonymous_prototype_blanks_the_name():
    # Live BN assigns str(new_name) unconditionally, so an anonymous prototype
    # leaves the function with an EMPTY name (probed on BN 5.4). The fake must
    # not be kinder than that.
    fn, _ = _fn_with_var()
    _mutation_bv(fn)
    fn.type = "int32_t()"
    assert fn.name == ""
    assert fn.has_user_type is True


def test_type_setter_with_a_type_object_does_not_rename():
    # Negative control: BN's non-string branch goes straight to set_user_type,
    # with no parse and no rename.
    fn, _ = _fn_with_var()
    _mutation_bv(fn)
    parsed, _name = fn.view.parse_type_string("uint64_t other_fn(int32_t a)")

    fn.type = parsed

    assert fn.name == "sub_1000"
    assert fn.type is parsed
    assert fn.has_user_type is True


def test_set_auto_type_does_not_pin_a_user_type_or_rename():
    # Negative control: set_auto_type is a bare value write in BN -- no parse,
    # no rename, no provenance.
    fn, _ = _fn_with_var()
    fn.set_auto_type("uint64_t f(int32_t a)")
    assert fn.type == "uint64_t f(int32_t a)"
    assert fn.name == "sub_1000"
    assert fn.has_user_type is False


# --- undo journaling ------------------------------------------------------

def test_undo_restores_auto_local_value_and_provenance():
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)

    state = bv.begin_undo_actions()
    fn.create_user_var(var, "char*", "probe_name")
    bv.update_analysis_and_wait()
    assert (var.name, fn.is_var_user_defined(var)) == ("probe_name", True)

    bv.revert_undo_actions(state)
    bv.update_analysis_and_wait()
    assert (var.name, var.type) == ("var_8", "int32_t")
    assert fn.is_var_user_defined(var) is False


def test_undo_restores_a_prior_user_local_override():
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)
    fn.create_user_var(var, "int32_t", "kept_by_user")

    state = bv.begin_undo_actions()
    fn.create_user_var(var, "char*", "probe_name")
    bv.revert_undo_actions(state)
    bv.update_analysis_and_wait()

    assert (var.name, var.type) == ("kept_by_user", "int32_t")
    assert fn.is_var_user_defined(var) is True


def test_undo_restores_prototype_value_but_not_has_user_type():
    # #582's live finding: revert_undo_actions puts the prototype value back
    # yet leaves BNFunctionHasUserType set. The fake must reproduce that.
    fn, _ = _fn_with_var()
    bv = _mutation_bv(fn)
    before = fn.type

    state = bv.begin_undo_actions()
    fn.type = "uint64_t f(int32_t a)"
    assert fn.has_user_type is True

    bv.revert_undo_actions(state)
    assert fn.type == before
    assert fn.has_user_type is True


def test_commit_undo_keeps_applied_provenance():
    # Negative control: a committed transaction must not roll anything back.
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)

    state = bv.begin_undo_actions()
    fn.create_user_var(var, "char*", "probe_name")
    fn.type = "uint64_t f(int32_t a)"
    bv.commit_undo_actions(state)
    bv.update_analysis_and_wait()

    assert (var.name, fn.is_var_user_defined(var)) == ("probe_name", True)
    assert (str(fn.type), fn.has_user_type) == ("uint64_t(int32_t a)", True)


def test_delete_user_var_restores_the_analysis_derived_name_not_the_last_auto_write():
    # Live BN 5.4 probe: create_auto_var(v, t, "auto_named") -> create_user_var(
    # v, t, "user_named") -> delete_user_var(v) -> update_analysis_and_wait()
    # settles to "var_8", the value analysis DERIVES -- not the intervening AUTO
    # name. A fake that replays the last auto write would assert the wrong
    # post-rollback name.
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)
    fn.create_auto_var(var, "int32_t", "auto_named")
    fn.create_user_var(var, "char*", "user_named")

    fn.delete_user_var(var)
    bv.update_analysis_and_wait()

    assert (var.name, var.type) == ("var_8", "int32_t")
    assert fn.is_var_user_defined(var) is False


def test_auto_write_over_a_user_var_is_undone_by_analysis():
    # Live BN 5.4: an AUTO write lands immediately even over a USER override,
    # but the next analysis pass restores the user value -- BNCreateAutoVariable
    # never displaces a user override for good.
    fn, var = _fn_with_var("var_8")
    bv = _mutation_bv(fn)
    fn.create_user_var(var, "char*", "user_named")

    fn.create_auto_var(var, "int64_t", "auto_named")
    assert var.name == "auto_named"            # transiently visible...
    assert fn.is_var_user_defined(var) is True

    bv.update_analysis_and_wait()
    assert (var.name, var.type) == ("user_named", "char*")   # ...then reverted
    assert fn.is_var_user_defined(var) is True


def test_writes_reach_an_hlil_alias_sharing_the_identifier():
    # A local can live only in hlil.vars, and an hlil mirror can share an
    # identifier with a distinct stack_layout object (mutation_engine handles
    # exactly this aliasing). A write through one object must be observable
    # through the other, or a preview would report "restored" while BN drifts.
    fn, stack_var = _fn_with_var("var_8", identifier=10)
    hlil_mirror = _FakeVariable(name="var_8", storage=-8, var_type="int32_t", identifier=10)
    fn.hlil = types.SimpleNamespace(vars=[hlil_mirror])
    bv = _mutation_bv(fn)

    fn.create_user_var(stack_var, "char*", "probe_name")
    assert (hlil_mirror.name, hlil_mirror.type) == ("probe_name", "char*")

    state = bv.begin_undo_actions()
    fn.create_user_var(stack_var, "int64_t", "second_name")
    bv.revert_undo_actions(state)
    assert (hlil_mirror.name, hlil_mirror.type) == ("probe_name", "char*")


def test_hlil_only_local_is_written_and_restored():
    # Negative control for the universe fix: a var that exists ONLY in hlil.vars
    # (no stack_layout entry) must still be reached and rolled back.
    fn = _FakeFunction(0x1000, "sub_1000")
    only_hlil = _FakeVariable(name="var_c", storage=-0xC, var_type="int32_t", identifier=7)
    fn.hlil = types.SimpleNamespace(vars=[only_hlil])
    bv = _mutation_bv(fn)

    state = bv.begin_undo_actions()
    fn.create_user_var(only_hlil, "char*", "probe_name")
    assert (only_hlil.name, fn.is_var_user_defined(only_hlil)) == ("probe_name", True)

    bv.revert_undo_actions(state)
    assert (only_hlil.name, only_hlil.type) == ("var_c", "int32_t")
    assert fn.is_var_user_defined(only_hlil) is False


def test_overlapping_undo_transactions_get_distinct_handles():
    # Real BN hands back a distinct handle per transaction. With a constant
    # handle, reverting the OUTER transaction popped the inner snapshot and
    # silently left the outer mutation applied.
    fn, var = _fn_with_var("var_8")
    second = _FakeVariable(name="var_10", storage=-0x10, var_type="int32_t", identifier=2)
    fn.stack_layout.append(second)
    bv = _mutation_bv(fn)

    outer = bv.begin_undo_actions()
    fn.create_user_var(var, "char*", "first_probe")
    inner = bv.begin_undo_actions()
    fn.create_user_var(second, "char*", "second_probe")
    assert outer != inner

    bv.revert_undo_actions(outer)
    bv.update_analysis_and_wait()

    # BOTH mutations are gone, and the nested entry did not leak.
    assert (var.name, var.type) == ("var_8", "int32_t")
    assert (second.name, second.type) == ("var_10", "int32_t")
    assert fn.is_var_user_defined(var) is False
    assert bv._undo_journal == []


def test_nested_undo_transaction_reverts_only_its_own_mutation():
    # Negative control: reverting the INNER transaction must leave the outer
    # mutation applied.
    fn, var = _fn_with_var("var_8")
    second = _FakeVariable(name="var_10", storage=-0x10, var_type="int32_t", identifier=2)
    fn.stack_layout.append(second)
    bv = _mutation_bv(fn)

    bv.begin_undo_actions()
    fn.create_user_var(var, "char*", "first_probe")
    inner = bv.begin_undo_actions()
    fn.create_user_var(second, "char*", "second_probe")

    bv.revert_undo_actions(inner)
    bv.update_analysis_and_wait()

    assert var.name == "first_probe"
    assert (second.name, second.type) == ("var_10", "int32_t")
