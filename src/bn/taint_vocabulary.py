"""The taint output vocabulary: every value a reader of a taint result must interpret.

``bn capabilities`` advertises it (V4-05) so an agent can learn, before its
first run, which frontier leaf kinds block a claim, which truncation causes a
run can name, which backward origin kinds and ``sink_status`` statuses exist and
which ``run_params`` keys echo a run's configuration.

Two kinds of source, one each:

* Declared sets -- ``BLOCKING_LEAF_KINDS`` here, ``TRUNCATION_CAUSES`` in
  :mod:`taint_truncation` -- that the engine reads. Tests enumerate the engine's
  literals against them.
* Literals only the engine spells (origin kinds, ``sink_status`` statuses,
  ``run_params`` keys). Those are read off the bridge source by the AST crawler
  below, the same one the skill-reference drift tests use, so the advertised
  list and the documented tables cannot disagree with the code. The bridge
  source is located the way ``bn plugin install`` locates it
  (:func:`paths.plugin_source_dir`): ``src/bn_agent_bridge`` in a checkout or an
  editable install, the packaged ``bn_agent_bridge`` in a wheel. The source is
  read as text, never imported, so the CLI keeps no Binary Ninja dependency.

Symlinked into ``src/bn_agent_bridge/`` like ``paths.py``; stdlib-only.
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .paths import plugin_source_dir
from .taint_truncation import TRUNCATION_CAUSES

# Leaf kinds that mean "taint stopped somewhere attacker-relevant" -- an empty
# reached_sinks list carrying any of these is NEVER an all-clear. The seed-side
# honesty leaves (#562) are included so a weak/mis-anchored recv seed blocks the
# gate the same way a real frontier leaf does.
BLOCKING_LEAF_KINDS = frozenset({
    "coarse_memory_store",
    # P2-10: a sink gated behind an opt-in name (`recv_overflow`) whose
    # arming argument taint reached; the run neither reported nor ruled it
    # out, so it is not an all-clear.
    "gated_sink_reached",
    # P2-10: a `param:`/`var:` seed nothing in the function reads -- usually a
    # prototype that does not match the body -- so the empty result says
    # nothing about the value the locator meant.
    "seed_unused",
    "pointer_escape",
    "indirect_call_unresolved",
    "unmodeled_callee",
    "arg_under_recovered",
    "field_load_unresolved",
    "source_seed_misanchored",
    # #206: a tainted value reached an unlifted/unimplemented instruction, so
    # BN's lifter could not model the op and propagation through it is a silent
    # hole -- an empty reached_sinks past it is NOT an all-clear. Emitted
    # flow-sensitively (only when taint actually reaches such an instruction),
    # so a function that merely CONTAINS unlifted ops does not block the gate.
    "unlifted_instruction_reached",
    # #1041 item 4 / #1050 review finding 4: a scanf-family call whose format
    # is not a read-only constant (or uses %n$ positions) may write an
    # unbounded token; its conversions could not be checked, so an empty
    # reached_sinks past it is not an all-clear.
    "format_unchecked",
    # #1041 item 1 gate (#1050 review round 4): an indirect call whose callee
    # bn did not resolve can run after a call that fills a buffer. Which
    # pointer it receives is not worked out; what it does with the bytes is
    # unchecked, so an empty reached_sinks is not an all-clear.
    "indirect_call_unchecked",
    # #1050 review round 8 / P2-12: a scanf-family call whose fully read
    # constant format writes destinations that were never seeded, one kind per
    # cause: BN recovered fewer arguments than the format requires (forwarded
    # incoming pointers left in their argument registers), or an arity-capped
    # model's unrolled destination run ends before the recovered destinations
    # do. Either way an empty result is no all-clear.
    "format_destination_unrecovered",
    "scanf_arity_residual",
    # BN6-C009: a reached model rule declares a write into an argument BN did
    # not recover for that call. No output pointer is guessed, so what the
    # declared destination receives is unchecked -- not an all-clear.
    "model_destination_unrecovered",
    # #810: a backward ascent followed only the first N caller sites of a
    # parameter-origin slice; origins reachable only from the dropped callers are
    # absent from the result. Same class as the other frontier leaves: the walk
    # stopped with data unexamined, so the slice is NOT a complete answer.
    "caller_sites_truncated",
    # A backward slice reached `p = &local_buf` (or a folded `sink(&buf)`) and
    # the walk could not account for the buffer's CONTENTS: no memory version
    # to walk, an unmodeled call took the buffer's address or ran after it
    # escaped to memory, a path to the sink with no writer at all (a
    # conditional fill), a writer's input the walk could not interpret, or a
    # memory chain longer than the walk budget. The sink consumes the bytes,
    # not the pointer, so a slice that stops at the address-of is NOT complete.
    "address_of_local_unresolved",
    # #931 round 1: a backward slice reached a pointer with no local behind it
    # -- a call's return or a parameter -- that a consumer reads through. The
    # object's extent and earlier contents are unknown to the walk, so nothing
    # proves the bytes the consumer reads; its writers here stay in the slice.
    "pointer_object_unresolved",
    # #931/#933 round 2: stores cover a consumed region the walk cannot bound
    # -- the consumer reads outside the object or at an offset that is not
    # one constant. The stores are reported as facts, not a proof.
    "object_extent_unknown",
    # #931/#933 round 3: stores cover every consumed byte on every path. That
    # is reported (`stores_cover`) but never certifies the slice: the proof
    # rests on BN's recovery of the object's extent.
    "region_store_proof_not_certified",
    # #931 round 6, the policy: a sink that reads through a pointer is never
    # complete. Every pointer demand carries this leaf, listing each object
    # it may read.
    "pointer_demand_not_certified",
    # #931 round 7: an `arg:` sink argument no model gives a role (walked as
    # a pointer demand), and a sink a multi-sink query could not seed at all.
    "sink_role_unknown",
    "sink_unseeded",
    # IC03 / TT-PILOT-FW03-01: tainted bytes reached a keyable writable global
    # and code reachable from a call references that global. Taint is tracked
    # per function and nothing but arguments crosses a call, so what those
    # bytes reach past that boundary was never walked -- an empty
    # reached_sinks there is not an all-clear.
    "global_memory_continuation",
    # Round-3 review R3: the forward walk unwound on Python's recursion limit
    # (an unresolved cycle). Whatever it had already found is reported, but
    # the rest was never walked -- so the run is explicitly a frontier, not a
    # short clean answer.
    "analysis_truncated",
    # Review F10: the search for everything that can run before a call ran out
    # of its block budget, so what an argument register carries into that call
    # was never established. The blocks it did not read are exactly where a
    # clobbering call would be, so an empty reached_sinks past it is not an
    # all-clear.
    "abi_provenance_budget",
    # IC05 / BN6-C008: tainted arguments entered an in-binary callee whose
    # body Binary Ninja never analysed (oversized past
    # analysis.limits.maxFunctionSize, or a lift that produced no MLIL). The
    # call RESOLVED -- this is not an unresolved target -- but nothing read
    # what the body does with the bytes, so an empty reached_sinks past it is
    # an absence of evidence. Emitted whether or not a model covers the call:
    # a model is an overlay on a body, never a substitute for reading one.
    "unread_callee_body",
    # P2-02: a backward slice ended at a register's ENTRY value that no
    # parameter of the function's recovered prototype is bound to (BN's armv7
    # mapping puts a `long long`/`double` in `r2:r3` on the stack, leaving
    # `entry_r2`). The value comes from the caller through a location the
    # prototype does not map, so the caller ascent never ran: not an origin.
    "abi_mapping_unbound",
})


class VocabularyUnresolved(ValueError):
    """A producer the crawler cannot reduce to string literals. Raised rather
    than skipped: a value built some new way must not drop out of the list."""


# ``constructors(node, scopes)`` yields the dict nodes *node* builds.
Constructors = Callable[[ast.AST, list], Iterator[ast.expr]]


def _parse(path: Path) -> ast.Module:
    """*path*'s tree, parsed once per file version: one ``taint_vocabulary``
    call crawls the engine three times, and the parse is most of the cost."""
    stat = path.stat()
    return _parse_version(path.resolve(), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _parse_version(path: Path, mtime_ns: int, size: int) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def scope_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """The nodes of *fn*'s own body: nested functions, lambdas and classes are
    other scopes, so a same-named variable in one of them never resolves here."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def str_values(expr: ast.AST, scopes: list, consts: dict[str, str], tree: ast.AST) -> set[str]:
    """Every string *expr* can evaluate to: a literal, both arms of a
    conditional, a module constant, a local assigned in an enclosing scope, or
    a parameter (its default and every keyword argument a call passes it).
    Anything else raises, so a value built some new way cannot slip past."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        return (str_values(expr.body, scopes, consts, tree)
                | str_values(expr.orelse, scopes, consts, tree))
    if isinstance(expr, ast.Name):
        if expr.id in consts:
            return {consts[expr.id]}
        for depth in range(len(scopes) - 1, -1, -1):
            fn, outer = scopes[depth], scopes[:depth + 1]
            values: set[str] = set()
            found = False
            for node in scope_nodes(fn):
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == expr.id for t in node.targets):
                    values |= str_values(node.value, outer, consts, tree)
                    found = True
            args = fn.args
            positional = args.posonlyargs + args.args
            pdefaults = dict(zip([a.arg for a in positional][len(positional) - len(args.defaults):],
                                 args.defaults))
            kdefaults = {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None}
            if expr.id in {a.arg for a in positional + args.kwonlyargs}:
                found = True
                default = pdefaults.get(expr.id) or kdefaults.get(expr.id)
                if default is not None:
                    values |= str_values(default, outer, consts, tree)
                for call in ast.walk(tree):
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == fn.name):
                        for kw in call.keywords:
                            if kw.arg == expr.id:
                                values |= str_values(kw.value, outer, consts, tree)
            if found:
                return values
    raise VocabularyUnresolved(f"cannot resolve `{ast.unparse(expr)}` "
                               f"(line {getattr(expr, 'lineno', '?')}) to string literals")


def kind_rows(path: Path, constructors: Constructors, key: str = "kind") -> dict[str, set[str]]:
    """``{value of key: literal field keys}`` over every dict *constructors*
    yields from *path* -- each of which must carry *key*."""
    tree = _parse(path)
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
              for t in n.targets if isinstance(t, ast.Name)}
    rows: dict[str, set[str]] = {}

    def visit(node: ast.AST, scopes: list) -> None:
        for built in constructors(node, scopes):
            if not isinstance(built, ast.Dict):
                raise VocabularyUnresolved(
                    f"{path.name}:{node.lineno} builds one from a non-literal "
                    f"`{ast.unparse(built)}`: its `{key}` cannot be read")
            fields = {k.value: v for k, v in zip(built.keys, built.values)
                      if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if key not in fields:
                raise VocabularyUnresolved(f"{path.name}:{built.lineno} builds one without `{key}`")
            for value in str_values(fields.pop(key), scopes, consts, tree):
                rows.setdefault(value, set()).update(fields)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes = scopes + [node]
        for child in ast.iter_child_nodes(node):
            visit(child, scopes)

    visit(tree, [])
    return rows


def origin_constructors(node: ast.AST, scopes: list) -> Iterator[ast.expr]:
    """A backward origin: the argument of `_set_origin`/`_set_source_origin`,
    a dict assigned to `origin`, or the value under an `"origin"` key."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("_set_origin", "_set_source_origin")):
        yield node.args[0]
    elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
          and any(isinstance(t, ast.Name) and t.id == "origin" for t in node.targets)):
        yield node.value
    elif isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "origin" and isinstance(v, ast.Dict):
                yield v


def overlay_constructors(node: ast.AST, scopes: list) -> Iterator[ast.expr]:
    """A model overlay row: any dict with a `kind` built inside
    `model_overlay_sources`, the one function that writes `model_sources` /
    `overlays`."""
    if (isinstance(node, ast.Dict) and any(s.name == "model_overlay_sources" for s in scopes)
            and any(isinstance(k, ast.Constant) and k.value == "kind" for k in node.keys)):
        yield node


def sink_status_constructors(node: ast.AST, scopes: list) -> Iterator[ast.expr]:
    """A backward ``sink_status`` row: the argument of every
    ``sink_status.append(...)``, the one way the engine adds a row."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sink_status"):
        yield node.args[0]


def run_params_keys(path: Path) -> set[str]:
    """Every key *path* writes into a ``run_params`` echo: the keys of a dict
    literal under a ``"run_params"`` key, and each ``run_params["<key>"] = ...``
    assignment. Any other write to ``run_params`` raises."""
    keys: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and k.value == "run_params"):
                    continue
                if not isinstance(v, ast.Dict):
                    raise VocabularyUnresolved(
                        f"{path.name}:{node.lineno} builds run_params from `{ast.unparse(v)}`")
                for field in v.keys:
                    if not (isinstance(field, ast.Constant) and isinstance(field.value, str)):
                        raise VocabularyUnresolved(
                            f"{path.name}:{v.lineno} builds run_params with a non-literal key")
                    keys.add(field.value)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "run_params"):
                    if not (isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                        raise VocabularyUnresolved(
                            f"{path.name}:{node.lineno} writes run_params[{ast.unparse(t.slice)}]")
                    keys.add(t.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "run_params"):
            raise VocabularyUnresolved(
                f"{path.name}:{node.lineno} writes run_params through `{ast.unparse(node.func)}`")
    return keys


def taint_vocabulary(bridge: Path | None = None) -> dict[str, Any]:
    """The vocabulary ``bn capabilities`` advertises, each list sorted.

    *bridge* is the bridge package directory, defaulting to the one
    ``bn plugin install`` installs from."""
    bridge = plugin_source_dir() if bridge is None else bridge
    engine = bridge / "taint_engine.py"
    return {
        "leaf_kinds": sorted(BLOCKING_LEAF_KINDS),
        "truncation_causes": sorted(TRUNCATION_CAUSES),
        "origin_kinds": sorted(kind_rows(engine, origin_constructors)),
        "sink_status": sorted(kind_rows(engine, sink_status_constructors, key="status")),
        "run_params": sorted(run_params_keys(engine)
                             | run_params_keys(bridge / "read_taint_slice.py")),
    }
