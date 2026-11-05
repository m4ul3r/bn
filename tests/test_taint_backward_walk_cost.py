"""The backward walk does each piece of work once (V3-04).

A depth-first walk reaches a value from ever shallower paths; re-walking its
subtree at every one of them (the depth budget cut something beneath it, so a
shallower path may reach further) made one decompressor body's backward slice
5x its base cost. Re-walks now wait in a queue and run once, from the
shallowest path. Likewise the writers of a load and the values those writers
stored are worked out once per slice, however many loads share the chain."""
from __future__ import annotations

from collections import Counter

from test_taint_engine import FBV, FExpr, FFunc, FInstr, FSSA, FSSAFunc, FVar, te


class _CountingSSA(FSSAFunc):
    def __init__(self, instrs, mem_defs=None):
        super().__init__(instrs, mem_defs)
        self.var_asks = Counter()
        self.mem_asks = Counter()

    def get_ssa_var_definition(self, ssa_var):
        self.var_asks[(ssa_var.var.name, ssa_var.version)] += 1
        return super().get_ssa_var_definition(ssa_var)

    def get_ssa_memory_definition(self, version):
        self.mem_asks[int(version)] += 1
        return super().get_ssa_memory_definition(version)


_MALLOC = 0x9000


def _add(i, dest, left, right):
    return FInstr(i, 0x1000 + 4 * i, "MLIL_SET_VAR_SSA", f"{dest} = {left} + {right}",
                  reads=[left, right], writes=[dest], dest=dest,
                  src=FExpr("MLIL_ADD", f"{left} + {right}", reads=[left, right]))


def _copy(i, dest, src):
    return FInstr(i, 0x1000 + 4 * i, "MLIL_SET_VAR_SSA", f"{dest} = {src}",
                  reads=[src], writes=[dest], dest=dest,
                  src=FExpr("MLIL_VAR_SSA", str(src), reads=[src], src=src))


def test_a_value_reached_from_ever_shallower_paths_is_rewalked_once():
    # `s = c1 + x; c1 = c2 + x; ... c5 = c6 + x`: the walk goes down the c
    # ladder first and meets x six levels deep, under a depth budget of 6 its
    # long copy chain is cut, and every rung on the way back up meets x one
    # level shallower. x's chain is re-walked from its shallowest path ONCE.
    s = FSSA(FVar("s"), 1)
    cs = [FSSA(FVar(f"c{i}"), 1) for i in range(1, 7)]
    x = FSSA(FVar("x"), 1)
    ys = [FSSA(FVar(f"y{j}"), 1) for j in range(1, 11)]
    instrs = [_add(0, s, cs[0], x)]
    instrs += [_add(i, cs[i - 1], cs[i], x) for i in range(1, 6)]
    instrs.append(_copy(6, x, ys[0]))
    instrs += [_copy(7 + j, ys[j], ys[j + 1]) for j in range(9)]
    instrs.append(FInstr(20, 0x1100, "MLIL_CALL_SSA", "malloc(s#1)", reads=[s], writes=[],
                         dest=FExpr("MLIL_CONST_PTR", hex(_MALLOC), constant=_MALLOC),
                         params=[FExpr("MLIL_VAR_SSA", "s#1", reads=[s], src=s)]))
    ssaf = _CountingSSA(instrs)
    func = FFunc("ladder", 0x1000, ssaf, params=[])
    result = te.TaintEngine(FBV({_MALLOC: "malloc"}), te.load_models(), max_depth=6).backward(
        func, [te.parse_locator("arg:malloc:0")])
    assert result["slices"], result
    # The chain below x is asked about from x's first (deepest) path and once
    # more from its shallowest -- not once per rung of the ladder.
    assert ssaf.var_asks[("y1", 1)] <= 2, ssaf.var_asks
    assert ssaf.var_asks[("x", 1)] <= 2, ssaf.var_asks


def test_loads_sharing_one_memory_chain_walk_it_once():
    # Twelve loads through twelve different pointers read after the same
    # run of stores: each store is a writer every load's walk must consider,
    # and the chain behind them is the same one each time.
    ptrs = [FSSA(FVar(f"p{i}"), 0) for i in range(12)]
    vals = [FSSA(FVar(f"v{i}"), 1) for i in range(12)]
    mem_defs, instrs = {}, []
    q = FSSA(FVar("q"), 0)
    for k in range(1, 9):
        st = FInstr(k, 0x1000 + 4 * k, "MLIL_STORE_SSA", f"[q#0 + {k}].b = {k}", reads=[q],
                    dest=FExpr("MLIL_ADD", f"q#0 + {k}", reads=[q],
                               left=FExpr("MLIL_VAR_SSA", "q#0", reads=[q], src=q),
                               right=FExpr("MLIL_CONST", str(k), constant=k)),
                    src=FExpr("MLIL_CONST", str(k), constant=k))
        st.size, st.src_memory, st.dest_memory = 1, k - 1, k
        mem_defs[k] = st
        instrs.append(st)
    total = None
    for i, (p, v) in enumerate(zip(ptrs, vals)):
        load = FExpr("MLIL_LOAD_SSA", f"[p{i}#0].b @ mem#8", reads=[p],
                     src=FExpr("MLIL_VAR_SSA", f"p{i}#0", reads=[p], src=p))
        load.size, load.src_memory = 1, 8
        instrs.append(FInstr(20 + i, 0x1100 + 4 * i, "MLIL_SET_VAR_SSA", f"{v} = {load}",
                             reads=[p], writes=[v], dest=v, src=load))
        total = v if total is None else total
    adds = []
    acc = vals[0]
    for i, v in enumerate(vals[1:], start=1):
        nxt = FSSA(FVar(f"t{i}"), 1)
        adds.append(_add(40 + i, nxt, acc, v))
        acc = nxt
    instrs += adds
    instrs.append(FInstr(60, 0x1200, "MLIL_CALL_SSA", "malloc(t)", reads=[acc], writes=[],
                         dest=FExpr("MLIL_CONST_PTR", hex(_MALLOC), constant=_MALLOC),
                         params=[FExpr("MLIL_VAR_SSA", str(acc), reads=[acc], src=acc)]))
    ssaf = _CountingSSA(instrs, mem_defs=mem_defs)
    func = FFunc("sum_fields", 0x1000, ssaf, params=[])
    result = te.TaintEngine(FBV({_MALLOC: "malloc"}), te.load_models()).backward(
        func, [te.parse_locator("arg:malloc:0")])
    assert result["slices"], result
    # Every load reads at memory version 8 through a pointer of its own, so
    # each walks the eight stores; what a STORE is -- its slot, the root of
    # its address -- is worked out once per store, not once per load.
    assert ssaf.var_asks[("q", 0)] <= 2 * 8, ssaf.var_asks
