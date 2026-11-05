"""The objects a sink's pointer may read are named, never repr'd (W09-10).

`_demand_objects` describes each root a consumed pointer may point into; its
descriptions are printed verbatim in a `sink_role_unknown` /
`pointer_demand_not_certified` leaf's "It may read: ..." list. At its depth
cap it described the value with `str()`, which for a bare SSA variable is BN's
`<SSAVariable: cp version 0>` repr rather than the `cp#0` every other object
in that list uses."""
from __future__ import annotations

from test_taint_engine import FBV, FSSA, FSSAFunc, FVar, te


class _ReprSSA(FSSA):
    """An SSA variable whose str() is BN's repr, as on a live view."""

    def __str__(self):
        return f"<SSAVariable: {self.var.name} version {self.version}>"


def test_depth_capped_ssa_variable_is_named_by_its_label():
    engine = te.TaintEngine(FBV({}), te.load_models())
    [obj] = engine._demand_objects(None, FSSAFunc([]), _ReprSSA(FVar("cp"), 0), depth=17)
    assert obj == {"kind": "unknown", "object": "cp#0"}
