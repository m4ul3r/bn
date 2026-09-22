"""Fidelity tests for the shared fake's `binaryninja` enum surface.

Real BN models `SymbolType` / `SymbolBinding` / `RelocationType` as `IntEnum`, so
`str()` on a member is its NUMERIC value and `.name` is the member name. A
plain-string stand-in is strictly *more forgiving*: `str(sym.type) ==
"SymbolType.ExternalSymbol"` passes the mocked suite and silently never fires
against a live view, which is how #529 shipped (#593).

The member names and values pinned below were read off a live BN 5.4 install
(`binaryninja.enums`), so a drift here means the fake stopped matching the core
the rest of the suite is standing in for.
"""

from __future__ import annotations

import enum
import sys

from _bridge_fakes import _load_bridge


def _fake_bn(monkeypatch):
    _load_bridge(monkeypatch)
    return sys.modules["binaryninja"]


def test_fake_symbol_enums_are_int_enums_with_the_live_bn_member_values(monkeypatch):
    bn = _fake_bn(monkeypatch)
    assert {m.name: m.value for m in bn.SymbolType} == {
        "FunctionSymbol": 0,
        "ImportAddressSymbol": 1,
        "ImportedFunctionSymbol": 2,
        "DataSymbol": 3,
        "ImportedDataSymbol": 4,
        "ExternalSymbol": 5,
        "LibraryFunctionSymbol": 6,
        "SymbolicFunctionSymbol": 7,
        "LocalLabelSymbol": 8,
    }
    assert {m.name: m.value for m in bn.SymbolBinding} == {
        "NoBinding": 0,
        "LocalBinding": 1,
        "GlobalBinding": 2,
        "WeakBinding": 3,
    }
    assert {m.name: m.value for m in bn.RelocationType} == {
        "ELFGlobalRelocationType": 0,
        "ELFCopyRelocationType": 1,
        "ELFJumpSlotRelocationType": 2,
        "StandardRelocationType": 3,
        "IgnoredRelocation": 4,
        "UnhandledRelocation": 5,
    }
    for name in ("SymbolType", "SymbolBinding", "RelocationType"):
        assert issubclass(getattr(bn, name), enum.IntEnum), name


def test_str_of_a_symbol_type_member_is_its_value_never_its_name(monkeypatch):
    """The #529 trap: under the fake, a `str(sym.type)` comparison must be as
    unable to match as it is against a live view."""
    bn = _fake_bn(monkeypatch)
    ext = bn.SymbolType.ExternalSymbol
    assert (str(ext), ext.name, ext.value) == ("5", "ExternalSymbol", 5)
    assert "SymbolType" not in str(ext)
    assert str(bn.RelocationType.ELFJumpSlotRelocationType) == "2"
    assert str(bn.SymbolBinding.NoBinding) == "0"
    # The hardened call-site shape (`getattr(bn.SymbolType, name)` + `==`) keeps
    # resolving to the same member, so those sites stay green under the stricter
    # fake rather than merely tolerating it.
    assert getattr(bn.SymbolType, "ExternalSymbol") is ext
    assert getattr(bn.SymbolType, "DataSymbol") == 3
