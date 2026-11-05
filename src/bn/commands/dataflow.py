from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ..cli import (
    _call,
    _depth_int,
    _guarded_transform,
    _int_at_least,
    _render_result,
    _resolve_output_format,
    _resolve_target,
    _slice_hint_for_args,
    arg,
    command,
    decode_json_input,
    mangled_arg,
    read_text_input,
)
from ..formatters import (
    _render_callgraph_text,
    _render_defuse_text,
    _render_taint_models_text,
    _render_taint_text,
    _render_values_text,
    _resolution_note,
)
from ..read_taint_models import model_catalog
from ..taint_models import TaintError
from ..transport import BridgeError


# #812: the fixpoint remediation string ("raise --max-iters", taint_result.py
# _truncation_hint / the formatters truncation clause) named a flag that did not
# exist, so a truncated run told the user to do something impossible. Built here
# rather than beside `_depth_int` in cli.py so the label reads "iterations"
# instead of "depth" in the argparse error -- an iteration budget of 0 analyses
# nothing, so the floor is 1.
_iters_int = _int_at_least(1, "iterations")
_callsite_limit_int = _int_at_least(0, "callsite limit")


def _models_arg() -> tuple[tuple[str, ...], dict[str, Any]]:
    """The shared `--models FILE` option for taint commands (#317): load
    additional, project-specific taint models so taint follows flows through a
    target's own (non-libc-named) copy/format/exec wrappers."""
    return arg("--models", dest="models", default=None, metavar="FILE",
               help="JSON file of extra taint models (same schema as the builtin DB: "
                    'a {name: model} map) for project-internal copy/format/exec wrappers, '
                    "merged over the builtins so taint follows flows through your own "
                    "wrappers (#317). BN_TAINT_MODELS is honored as a fallback for the "
                    "same file, per invocation (#669); this flag wins when both are set.")


def _add_user_models(args: argparse.Namespace, params: dict[str, Any]) -> None:
    """Read `--models <file>`, else `$BN_TAINT_MODELS`, into ``params['user_models']``
    (loud on a bad file, mirroring --resolve-map and the #97 no-silent-model-failure
    rule).

    #669: the env var is resolved HERE, per invocation, and forwarded as a request
    param -- the bridge only ever captured it at spawn time, so setting it for one
    client call against a running bridge was silently ignored.

    #615 review (New finding 2 / F9 correction): the path is expanduser()'d, same
    as the bridge's own `taint_models_path()` (paths.py) -- an unexpanded ``~`` in
    a BN_TAINT_MODELS set from a shell profile/config file used to work at spawn
    time and must keep working per-invocation.

    A missing or unreadable file is refused the same way whichever knob named it:
    an env-sourced typo used to degrade silently to "builtin only" at exit 0,
    which is the no-silent-model-failure case #97 exists to refuse.

    ``--models`` wins when both are set (the option's documented contract), so
    the env file is not read; its path is forwarded as
    ``user_models_env_overridden`` so the overlay disclosure says it was not
    loaded instead of dropping it."""
    flag = getattr(args, "models", None)
    env = os.environ.get("BN_TAINT_MODELS")
    path = flag or env
    if not path:
        return
    source = "--models" if flag else "$BN_TAINT_MODELS"
    resolved = Path(path).expanduser()
    # #864: the shared reader refuses a missing, unreadable or blocking path by
    # name (a FIFO here hung with no envelope at all), and the shared decoder
    # refuses a body the parser cannot take -- malformed, nested past its
    # stack, or too large to build.
    params["user_models"] = decode_json_input(
        read_text_input(resolved, what=f"{source} file"),
        refusal=f"could not read {source} {path}")
    # #415: pass the file path through so the run's model_sources disclosure can
    # name WHICH file landed, not just a count. #669: also pass WHICH knob
    # supplied it, so the disclosure cannot label an env-sourced file `--models`.
    params["user_models_path"] = str(resolved)
    params["user_models_via"] = source
    if flag and env:
        params["user_models_env_overridden"] = str(Path(env).expanduser())


def _add_resolve_map(args: argparse.Namespace, params: dict[str, Any]) -> None:
    """Read `--resolve-map <file>` into ``params['resolve_map']`` (#824).

    Forward and backward taint each carried a byte-identical copy of this
    read/refuse policy. One copy now, mirroring the `_add_user_models`
    precedent above: a change to how the file is read or how a bad one is
    refused cannot land on one direction and silently miss the other.
    """
    if not args.resolve_map:
        return
    # #864: the shared reader refuses a directory/FIFO/device by kind -- a FIFO
    # here blocked forever with no envelope -- and the shared decoder refuses a
    # body the parser cannot take.
    params["resolve_map"] = decode_json_input(
        read_text_input(Path(args.resolve_map), what="--resolve-map file"),
        refusal=f"could not read --resolve-map {args.resolve_map}")
    # #1041 item 3: the bridge validates the entries (names resolve as -f
    # does) and names this file in any refusal.
    params["resolve_map_path"] = str(args.resolve_map)


@command("dataflow", "defuse",
         help="Show the SSA definition site and use sites of a variable "
              "(argument set-up for a call whose under-recovered callee dropped "
              "its stack-passed args is disclosed as a call-model truncation)",
         target=True,
         prefer_when="per-function SSA def/use of one variable; "
                     "use taint to follow a value across calls source->sink",
         see_also=("taint forward", "taint backward"),
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--var", dest="var", required=True,
                 help="Variable selector: name, local_id, or name#version (SSA). NOTE: "
                      "BN clamps a call's MLIL parameters to the callee's recovered "
                      "arity, so a callee auto-typed fixed-arity (a variadic declared "
                      "without `...`, a thunk with too narrow a prototype) drops its "
                      "stack-passed arguments from the call model; `uses` still lists "
                      "the argument stores, and the result carries the same "
                      "`hints` disclosure `trace` prints (`bn proto set` fixes it)"),
             mangled_arg(),
         ],
         estimable=True)
def _dataflow_defuse(args: argparse.Namespace) -> int:
    return _call(
        args,
        "defuse",
        {"identifier": args.identifier, "var": args.var},
        require_target=True,
        contextual_hints=True,
        text_renderer=lambda value: _resolution_note(value) + _render_defuse_text(
            value, mangled=bool(args.mangled)),
        stem="defuse",
    )


@command("dataflow", "callgraph", help="Resolved callees/callers (indirect targets via value-set)",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--direction", choices=("callees", "callers", "both"), default="both",
                 help="Which edges to resolve (default: both)"),
             arg("--no-resolve-indirect", dest="resolve_indirect", action="store_false", default=True,
                 help="Skip value-set resolution of indirect call targets"),
             arg("--per-site", action="store_true", default=False,
                 help="Text: one row per direct call site instead of one row per distinct "
                      "callee/caller with its count (JSON is always per site)"),
             mangled_arg(),
         ],
         estimable=True)
def _dataflow_callgraph(args: argparse.Namespace) -> int:
    return _call(
        args,
        "resolved_calls",
        {
            "identifier": args.identifier,
            "direction": args.direction,
            "resolve_indirect": bool(args.resolve_indirect),
        },
        require_target=True,
        text_renderer=lambda value: _resolution_note(value) + _render_callgraph_text(
            value, per_site=bool(args.per_site), mangled=bool(args.mangled)),
        stem="callgraph",
    )


@command("dataflow", "values", help="Instruction/source value-set and separate call-destination value-set",
         target=True,
         args=[
             arg("identifier", help="Function name or entry address (hex 0x.. or decimal)"),
             arg("--at", dest="at", required=True,
                 help="Instruction start address (hex 0x.. or decimal) within the function, "
                      "optionally ADDR:IL_INDEX to pick one of several MLIL SSA instructions "
                      "at one address (the [N] `bn function structured-il` prints); an "
                      "address where two instructions compute a value is refused with the "
                      "candidates listed. Call destinations are reported separately, not as "
                      "return values"),
             mangled_arg(),
         ],
         estimable=True)
def _dataflow_values(args: argparse.Namespace) -> int:
    return _call(
        args,
        "possible_values",
        {"identifier": args.identifier, "at": args.at},
        require_target=True,
        # #1033 item 8: the `--at` refusal prints a `bn il` command.
        contextual_hints=True,
        text_renderer=lambda value: _resolution_note(value) + _render_values_text(
            value, mangled=bool(args.mangled)),
        stem="values",
    )


# param:/arg: indices are 0-based -- for memcpy(dst, src, len) the length is
# arg:memcpy:2, not :3 (#291.4).
_LOCATOR_HELP = (
    "Locator: param:<n> | var:<selector> | ret:<callee> | arg:<callee>:<n> | "
    "call:<callee> (seeds every output the callee's model declares -- return "
    "value AND output-pointer buffers). param/arg indices are 0-based."
)

_SINK_LOCATOR_HELP = (
    "Locator: param:<n> | var:<selector> | arg:<callee>:<n> (ret: is forward-only). "
    "param/arg indices are 0-based (e.g. memcpy length is arg:memcpy:2)."
)


@command("taint", "forward", help="Forward taint: trace untrusted data from sources to sinks",
         target=True,
         prefer_when="follow untrusted data forward source->sink across calls; "
                     "use dataflow for per-function def/use, evidence for raw structure",
         see_also=("taint backward", "trace", "dataflow defuse"),
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--source", dest="sources", action="append", default=None, metavar="LOCATOR",
                 required=True,
                 help=f"Taint source (repeatable, at least one required). {_LOCATOR_HELP}"),
             arg("--max-depth", dest="max_depth", type=_depth_int, default=8,
                 help="Max interprocedural recursion depth into callees (default: 8; "
                      "0 = intraprocedural only)"),
             arg("--max-iters", dest="max_iters", type=_iters_int, default=256,
                 help="Max intra-function fixpoint iterations before the walk is "
                      "reported unconverged (default: 256). Raise when a result "
                      "reports truncation_cause fixpoint_exhausted; the configured "
                      "value is echoed back in run_params."),
             arg("--resolve-map", dest="resolve_map", default=None, metavar="FILE",
                 help="JSON file mapping indirect call addresses to target lists; a "
                      "target is an address or a function name (resolved as -f is): "
                      '{"0x4011f0": ["0x401176", "parse_frame"]}'),
             arg("--unknown-call", choices=("conservative", "stop"), default="conservative",
                 help="How to treat un-analyzed/external calls reached by taint (default: conservative)"),
             arg("--sink-class", dest="sink_classes", action="append", default=None,
                 choices=("file_write", "net_write", "recv_overflow"), metavar="CLASS",
                 help="Enable an opt-in sink class (repeatable). Off by default: "
                      "file_write (fwrite/write/fputs), net_write (send/sendto), "
                      "recv_overflow (read/recv/recvfrom length -> undersized buffer). "
                      "Use when auditing persistence / file-corruption / exfiltration, or "
                      "recv-into-fixed-buffer overflows. recv_overflow is high-FP on the "
                      "read-loop idiom (#499) -- enable it only when deliberately hunting."),
             _models_arg(),
             arg("--verbose", "-v", "--full", dest="full", action="store_true", default=False,
                 help="Show the full SSA path/slice for each flow (default: one compact line per flow)"),
             mangled_arg(),
         ],
         estimable=True,
         analysis_bounds=("--max-depth",))
def _taint_forward(args: argparse.Namespace) -> int:
    # --source is argparse-required, so an empty list cannot reach here.
    params: dict[str, Any] = {
        "direction": "forward",
        "function": args.function,
        "sources": list(args.sources),
        "max_depth": int(args.max_depth),
        "max_iters": int(args.max_iters),
        "unknown_call": args.unknown_call,
        "enabled_sink_classes": list(args.sink_classes or []),
        "mangled": args.mangled,
    }
    _add_resolve_map(args, params)
    _add_user_models(args, params)
    return _call(
        args,
        "taint",
        params,
        require_target=True,
        contextual_hints=True,
        # The frontier block prints its own `bn xrefs` follow-up (P2-04).
        route_rendered_commands=True,
        text_renderer=(lambda v: _render_taint_text(v, full=bool(getattr(args, "full", False)),
                                                    mangled=args.mangled)),
        stem="taint-forward",
    )


@command("taint", "backward", help="Backward taint: slice a sink's arguments back to their origin",
         target=True,
         prefer_when="slice a sink backward to its sources across calls; "
                     "use trace for a single argument's origin",
         see_also=("taint forward", "trace"),
         args=[
             arg("--function", "-f", dest="function", required=True,
                 help="Function to analyze (name or address)"),
             arg("--sink", dest="sinks", action="append", default=None, metavar="LOCATOR",
                 required=True,
                 help=f"Sink to slice from (repeatable, at least one required). {_SINK_LOCATOR_HELP}"),
             arg("--max-depth", dest="max_depth", type=_depth_int, default=8,
                 help="Max interprocedural depth to follow slices up into callers (default: 8; "
                      "0 = intraprocedural only); callers it stops short of are named in a "
                      "caller_sites_truncated leaf (reason max_depth) and the cut sets "
                      "stats.truncated (truncation_cause max_depth) with an assumption. The "
                      "in-function def-chain walk caps at 64 steps, and a cut there ends the "
                      "value at an `unresolved` slice origin with cause def_chain_budget."),
             arg("--resolve-map", dest="resolve_map", default=None, metavar="FILE",
                 help="JSON file mapping indirect call addresses to target lists; a "
                      "target is an address or a function name (resolved as -f is): "
                      '{"0x4011f0": ["0x401176", "parse_frame"]}'),
             _models_arg(),
             arg("--verbose", "-v", "--full", dest="full", action="store_true", default=False,
                 help="Show the full SSA path/slice for each flow (default: one compact line per flow)"),
             mangled_arg(),
         ],
         estimable=True,
         analysis_bounds=("--max-depth",))
def _taint_backward(args: argparse.Namespace) -> int:
    # --sink is argparse-required, so an empty list cannot reach here.
    params: dict[str, Any] = {
        "direction": "backward",
        "function": args.function,
        "sinks": list(args.sinks),
        "max_depth": int(args.max_depth),
        "mangled": args.mangled,
    }
    _add_resolve_map(args, params)
    _add_user_models(args, params)
    return _call(
        args,
        "taint",
        params,
        require_target=True,
        contextual_hints=True,
        route_rendered_commands=True,
        text_renderer=(lambda v: _render_taint_text(v, full=bool(getattr(args, "full", False)),
                                                    mangled=args.mangled)),
        stem="taint-backward",
    )


@command("taint", "models",
         help="List the taint model catalog (sources/sinks/propagators); with a target, which are present",
         target=True,
         prefer_when="enumerate known sinks/sources and (with a target) which appear in this binary",
         see_also=("taint forward", "taint backward"),
         args=[
             arg("--role", choices=("source", "sink", "propagator"), default=None,
                 help="Filter to one role"),
             arg("--class", dest="sink_class", default=None, metavar="CLASS",
                 help="Filter sinks to one bug class (e.g. overflow_len); implies --role sink"),
             arg("--present", action="store_true", default=False,
                 help="With a target: show only models present in the binary (errors without a target)"),
             arg("--callsites", action="store_true", default=False,
                 help="With --present: expand each present model's call sites (application callers only)"),
             arg("--all-sites", action="store_true", default=False,
                 help="With --callsites: also list the self-stub and constant-argument "
                      "inline-builtin rows that are otherwise only counted (implies --callsites)"),
             arg("--callsite-limit", dest="callsite_limit", type=_callsite_limit_int, default=10,
                 metavar="N",
                 help="Text output: print at most N callsite rows per model and count the "
                      "rest (default: 10; 0 = all). JSON always lists every row"),
             mangled_arg(),
             _models_arg(),
         ],
         estimable=True)
def _taint_models(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.role:
        params["role"] = args.role
    if args.sink_class:
        params["class"] = args.sink_class
    if args.present:
        params["present"] = True
    if args.callsites or args.all_sites:
        params["callsites"] = True
    if args.all_sites:
        params["all_sites"] = True
    # #615 review F7: `bn taint models` is the documented "confirm an overlay
    # landed" check -- it must honor --models/$BN_TAINT_MODELS identically to
    # `taint forward`/`taint backward`, not silently reflect only the bridge's
    # spawn-time env.
    _add_user_models(args, params)
    # #473: --present/--callsites are defined against a loaded binary, so require a
    # target for them -- which lets `_resolve_target` auto-select the single open
    # target like every other read command (no spurious "need a target" when one is
    # open and only --instance is given). Catalog-only mode needs no target.
    needs_target = bool(args.present or args.callsites or args.all_sites)
    render = (lambda value: _render_taint_models_text(
        value, mangled=args.mangled, callsite_limit=args.callsite_limit))
    if (not needs_target and getattr(args, "instance", None) is None
            and not getattr(args, "all_instances", False)
            and not getattr(args, "all_targets", False)
            and _resolve_target(args, require_target=False) is None):
        return _serve_catalog_locally(args, params, render)
    return _call(
        args,
        "taint_models",
        params,
        require_target=needs_target,
        text_renderer=render,
        stem="taint-models",
    )


def _serve_catalog_locally(args: argparse.Namespace, params: dict[str, Any],
                           render: Any) -> int:
    """Answer a target-free, instance-free `taint models` without a bridge.

    That catalog is the static model DB plus the user overlay, so routing it
    through a bridge only added ways to fail: with several instances registered
    and no `-i` it refused outright, and with none it spawned a headless BN to
    print a JSON file. `model_catalog` is the same function the bridge op runs
    (the module is shared by symlink, and imports no BN), so the keys, overlay
    disclosure and refusals match the bridge path. `honor_env=False`: this
    process's `$BN_TAINT_MODELS` was already resolved into ``params`` by
    `_add_user_models` (where `--models` wins) and must not load a second time
    as an override layer. An explicit `-i` still asks that bridge, whose own
    spawn-time override is part of its answer."""
    try:
        result, _models = model_catalog(params, honor_env=False)
    except TaintError as exc:
        raise BridgeError(str(exc)) from exc
    fmt = _resolve_output_format(args)
    shown = result
    if fmt == "text":
        shown = _guarded_transform(
            render, "render the taint_models result as text",
            advice="Rerun with --format json to see the raw result.")(result)
    _render_result(
        shown,
        fmt=fmt,
        out_path=args.out,
        stem="taint-models",
        spill_label="taint models",
        spill_context=result,
        provenance={"target": None, "instance": None},
        slice_hint=_slice_hint_for_args(args, fmt),
        estimate_only=bool(getattr(args, "estimate_output", False)),
    )
    return 0
