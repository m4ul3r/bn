from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from .. import cli
from ..cli import _call, _pick, arg, command
from ..transport import BridgeError, REFRESH_REQUEST_TIMEOUT, _format_instance_choices
from ..target_hint import (
    LISTED_SELECTOR_LEAD, format_multi_target_hint, positional_selector,
    respell_for_positional, target_option,
)
from ..formatters import (
    _render_close_text,
    _render_load_text,
    _render_refresh_text,
    _render_save_text,
    _render_target_info_annotations_text,
    _render_target_info_text,
)


def _render_target_info_text_with_annotations(value: Any) -> str:
    """`target info` text: the shared target summary, then the #793 annotations.

    Composed here rather than inside `formatters._render_target_info_text`
    because `target list` renders each row through the same summary and must NOT
    print a provenance block per named target -- only the single-target view has
    one view's annotations to disclose. `_render_target_info_annotations_text`
    returns "" when the payload carries no block (an older bridge), so the output
    is byte-identical to the pre-#793 one in that case.
    """
    body = _render_target_info_text(value)
    annotations = _render_target_info_annotations_text(value)
    return f"{body}\n{annotations}" if annotations else body


@command("load", help="Load a binary into headless bridge",
         args=[
             arg("path", help="Path to binary or BNDB file"),
             # `--instance-id` is an alias for the global `--instance` spawn name,
             # matching `bn session start --instance-id` so the auto-spawn flag is
             # spelled the same across both entry points (#258). SUPPRESS default
             # so an unset alias never clobbers a root --instance / BN_INSTANCE.
             arg("--instance-id", dest="instance", default=argparse.SUPPRESS,
                 metavar="INSTANCE",
                 help="ID for the bridge instance to auto-spawn (alias of --instance)"),
             arg("--no-bndb", action="store_true",
                 help="Open the raw bytes: skip both an adjacent <binary>.bndb and a "
                      "database saved to the cache. The cache is machine-wide; set "
                      "BN_CACHE_DIR per agent or project to isolate them -- it also "
                      "moves the instance registry and sticky session pins, so set it "
                      "on EVERY bn command (an instance started under it is invisible "
                      "without it)"),
             arg("--quick", "--no-analysis", action="store_true",
                 help="Load without full analysis (fast, ~1s): sections/imports/symbols are ready "
                      "immediately; strings and the full function set need `bn refresh`. Never "
                      "restores a cached database (the note names it); an adjacent "
                      "<binary>.bndb is still preferred unless --no-bndb"),
         ])
def _load(args: argparse.Namespace) -> int:
    import os
    # `bn load "$BIN"` with $BIN unset resolves "" to the cwd DIRECTORY, which
    # passes the bridge's exists() check and fails confusingly bridge-side --
    # same unset-shell-var doctrine as close/save (#690 r4).
    if not str(args.path).strip():
        raise BridgeError("path is empty: pass the binary or .bndb file to load")
    # BN_INSTANCE NAMES an instance exactly as -i does (#992): it reaches
    # args.instance as the root option's default. Only the sticky pin
    # (`bn instance use`) is ambient and stays refused below.
    explicitly_routed = bool(
        getattr(args, "_explicit_instance", False)
        or getattr(args, "_explicit_instance_id", False)
        or (getattr(args, "instance", None)
            and not getattr(args, "_sticky_instance", False))
    )
    instances = cli.list_instances() if not explicitly_routed else []
    if len(instances) > 1:
        raise BridgeError(
            "bn load refuses ambient instance routing while multiple bridges are "
            "running. Pass an explicit -i/--instance (or export BN_INSTANCE) for an "
            "existing bridge, or --instance-id to create a named bridge; the "
            "`bn instance use` pin is not used to pick where a load lands.\n"
            "Running instances:\n" + _format_instance_choices(instances)
        )
    return _call(
        args,
        "load_binary",
        {
            "path": str(Path(args.path).expanduser().resolve()),
            "prefer_bndb": not args.no_bndb,
            "quick": bool(args.quick),
            # Associate successful loads with the caller's canonical project root
            # in the private instance registry. No state is written to the checkout.
            "workdir": os.getcwd(),
        },
        require_target=False,
        text_renderer=_render_load_text,
        stem="load",
        # `bn load --instance <new-id>` auto-spawns that named bridge instead of
        # erroring, so a fresh isolated instance is one command, not two.
        spawn_missing_named=True,
        # A full (non-quick) initial analysis of a very large binary can exceed
        # the 600s read-op default; give the one-time load an hour by default
        # (BN_REQUEST_TIMEOUT still overrides), so it isn't abandoned mid-analysis
        # (#321). `--quick` returns fast and isn't affected.
        op_default_timeout=REFRESH_REQUEST_TIMEOUT,
    )


def _close_target_by_name(args: argparse.Namespace, name: str) -> str:
    """The ``target_id`` of the ONE open target a bare ``bn close <name>`` names.

    Deliberately narrower than the ``-t`` resolver (#996 review): only an exact
    basename or a full ``target_id`` counts -- never a view id (``close 5``
    must not close whichever view became view 5) and never a ``.bndb``-stripped
    alias (a second ``close hello`` must not go on to close ``hello.bndb``). It
    must match exactly one target, and the close is sent with
    ``pinned_target`` so the bridge resolves that id and nothing else.
    """
    response = cli.send_request(
        "list_targets", params={}, target=None,
        instance_id=getattr(args, "instance", None),
    )
    result = cli.unwrap_result(response, "list_targets")
    if not isinstance(result, list):
        raise BridgeError(
            "malformed bridge reply to list_targets (no target list); the "
            "bridge may be stale -- restart it or pass an explicit --target"
        )
    rows = [row for row in result if isinstance(row, dict)]
    matches = [row for row in rows
               if name in (row.get("basename"), str(row.get("target_id") or ""))]
    if len(matches) == 1 and matches[0].get("target_id"):
        return str(matches[0]["target_id"])
    if not rows:
        raise BridgeError(
            f"No targets are open, and no file {name!r} exists in this directory."
        )
    headline = (
        f"{len(matches)} open targets are named {name!r}; close one by selector."
        if matches else
        f"No open target is named {name!r}, and no file {name!r} exists in this "
        "directory. A bare name closes the open target with exactly that file "
        "name; to close by any selector use `bn close -t <selector>`."
    )
    raise BridgeError(format_multi_target_hint(headline, matches or rows))


@command("close", help="Close an open target (bn-loaded or a GUI tab)", target=True,
         args=[
             arg("path", nargs="?",
                 help="Path to close (omit to close the single open target). A bare "
                      "name (no `/`) that is no file here closes the one open target "
                      "with exactly that file name; any other selector needs -t"),
             arg("--all", action="store_true", help="Close all open targets, including GUI tabs bn did not load"),
         ])
def _close(args: argparse.Namespace) -> int:
    # Only honor an explicit -t/--target. A sticky pin must not silently pick
    # which target a bare `close` tears down, and a stale pin must not make
    # cleanup fail -- close needs to stay robust. `_call` resolves `args.target`
    # into the request target, so drop any sticky-injected value and let the
    # resolution below decide (single open target -> that one; multiple ->
    # refuse with the open-target list).
    if getattr(args, "_sticky_target", False):
        args.target = None
    explicit_target = getattr(args, "target", None)
    # Presence, not truthiness: `bn close ""` (an unset shell variable expanding
    # to an empty positional) is an operand that was GIVEN, so it must take
    # part in the conflict checks below and then be rejected as empty -- not
    # collapse into a bare close (which would tear down the sole open target),
    # and `bn close "" --all` must not slip past the path+--all guard.
    explicit_path = getattr(args, "path", None)
    # The three ways to say what to close -- `-t`, a path, `--all` -- are
    # mutually exclusive. The bridge gives a target priority over a path and
    # --all, and --all priority over a path, so any pair would silently drop one
    # operand (`bn close <path> --all` closed EVERYTHING despite naming a file,
    # #85). A destructive op must not guess which one was meant: reject every
    # combination before sending anything.
    if explicit_path is not None and args.all:
        raise BridgeError(
            "Pass a path or --all, not both: a named path closes only that "
            "target; --all closes every open target, including GUI tabs bn "
            "did not load."
        )
    if explicit_target is not None and explicit_path is not None:
        raise BridgeError(
            "Pass --target or a path, not both: --target closes that target; "
            "a path closes the open target at that path, whether bn loaded it "
            "or a GUI tab did."
        )
    if explicit_target is not None and args.all:
        raise BridgeError(
            "Pass --target or --all, not both: --target closes only that "
            "target; --all closes every open target, including GUI tabs bn "
            "did not load."
        )
    # `-t ""` is neither an explicit selector nor a bare close. It used to fall
    # into the implicit-resolution branch (`not target`) and tear down the
    # single open target; an empty selector on a destructive op is an error.
    if explicit_target is not None and not str(explicit_target).strip():
        raise BridgeError(
            "--target is empty: pass a selector from `bn target list`, a path, "
            "or --all (omit --target entirely to close the single open target)."
        )
    # Same for an empty positional: `bn close ""` must not become a bare close.
    if explicit_path is not None and not str(explicit_path).strip():
        raise BridgeError(
            "path is empty: pass a real path, a --target selector from "
            "`bn target list`, or --all (omit the path entirely to close the "
            "single open target)."
        )
    params: dict[str, Any] = {}
    if explicit_path is not None:
        resolved = Path(explicit_path).expanduser().resolve()
        name = str(explicit_path)
        if "/" in name or name.strip() == "active" or resolved.exists():
            params["path"] = str(resolved)
        else:
            # #996: every other command takes `hello` as a selector, so a bare
            # name that is no file in the cwd names an open target. Anything
            # spelled as a path, an existing file, or the volatile `active`
            # stays a path: what the operand has always meant.
            explicit_target = args.target = _close_target_by_name(args, name)
            explicit_path = None
            # #996 review: the id came from a listing, so it is sent as a PIN --
            # the bridge resolves it by exact target_id only and refuses if that
            # target is gone, rather than falling back to a file NAMED like it.
            params["pinned_target"] = True
    if args.all:
        params["all"] = True
    # #664: a bare `bn close` closes the single open target and refuses under
    # several (same hint + open-target list as every other target-required
    # command). Previously close skipped resolution and the bridge treated "no
    # target" as close-ALL, the inverse of `bn save`'s refusal.
    #
    # Round 2 (TOCTOU): the implicit resolution PINS the target_id observed in
    # the list_targets peek instead of sending the volatile literal `active`,
    # which the bridge would re-resolve at close time -- a concurrent close/load
    # between peek and close would land the destructive close on a DIFFERENT
    # binary. View ids are never reused, so any interleaving turns the pinned
    # id into a safe unknown-selector error. Round 3: the pinning lives in the
    # shared cli._implicit_target (every implicit resolution pins now, #690 R3)
    # and only the EXACT "active" literal is intercepted -- a padded ' active '
    # was a safe unknown-selector error before round 2 and must stay one, not
    # become a destructive sole-target close.
    if not (args.all or explicit_path is not None) and (
        explicit_target is None or explicit_target == "active"
    ):
        try:
            # Looked up on ``bn.cli`` at call time so tests patch one seam.
            args.target = cli._implicit_target(args)
            params["pinned_target"] = True
        except cli.MultiTargetError as exc:
            if explicit_target == "active":
                if getattr(args, "_close_spelling", None) == "target close":
                    raise BridgeError(
                        f"{exc}\nnote: `active` follows the GUI selection, "
                        "which is not honored for close: pass a concrete "
                        "selector from the list above (`bn close --all` closes "
                        "every open target, including GUI tabs bn did not load)."
                    ) from None
                raise BridgeError(
                    f"{exc}\nnote: `-t active` follows the GUI selection, "
                    "which is not honored for close: pass a concrete "
                    "selector, a path, or --all (closes every open target, "
                    "including GUI tabs bn did not load)."
                ) from None
            raise
    try:
        return _call(
            args,
            "close_binary",
            params,
            require_target=False,
            text_renderer=_render_close_text,
            stem="close",
        )
    except BridgeError as exc:
        raise _close_hint_in_run_form(args, exc) from None


def _close_hint_in_run_form(args: argparse.Namespace, exc: BridgeError) -> BridgeError:
    """Spell the bridge's "listed selector" refusal as the command that was run.

    The bridge names the selector `target list` advertises for the target (one
    that resolves to exactly it); only the CLI knows whether that goes after
    `bn close -t` or as `bn target close`'s positional (#996 review).
    """
    message = str(exc)
    if LISTED_SELECTOR_LEAD not in message:
        return exc
    head, quoted = message.rsplit(LISTED_SELECTOR_LEAD, 1)
    try:
        [selector] = shlex.split(quoted)
    except ValueError:
        return exc
    command = (f"{_target_close_prefix(args)} {positional_selector(selector)}"
               if getattr(args, "_close_spelling", None) == "target close"
               else f"bn {_routing(args)}close {target_option(selector)}")
    return BridgeError(f"{head}close it with `{command}`.")


def _routing(args: argparse.Namespace) -> str:
    instance = getattr(args, "instance", None)
    return f"-i {shlex.quote(str(instance))} " if instance else ""


def _target_close_prefix(args: argparse.Namespace) -> str:
    return f"bn {_routing(args)}target close"


@command("target", "close", help="Close one open target by selector", fmt="text",
         prefer_when="you have a selector from `bn target list` and want that one "
                     "target gone; `close` is the same operation with --target/a "
                     "path/--all",
         see_also=("close", "target list", "session stop"),
         args=[
             arg("selector",
                 help="Target selector to close (see `bn target list`)"),
         ])
def _target_close(args: argparse.Namespace) -> int:
    """Close exactly the named target.

    Discoverability, not new behavior: `target list`/`use`/`info` live under
    `target`, so agents look for the close verb there first and previously got an
    argparse error before guessing `bn close -t`. This delegates to the canonical
    `close` handler -- same unsaved warnings, same selector resolution, same
    renderer -- so the two spellings cannot drift.
    """
    selector = getattr(args, "selector", None)
    # Presence, not truthiness: `bn target close "$SEL"` with SEL unset expands
    # to an empty operand that must ERROR, never degrade into "close whatever is
    # open". The canonical handler rejects an empty selector too, but phrases the
    # recovery in terms of `--target`/a path/`--all` -- none of which this parser
    # accepts -- so the check is stated here in this command's own vocabulary.
    if selector is None or not str(selector).strip():
        raise BridgeError(
            "selector is empty: pass a target selector from `bn target list` "
            "(use `bn close --all` to close every open target, including GUI "
            "tabs bn did not load)."
        )
    args.target = selector
    # `active` follows the GUI selection and is re-resolved bridge-side, so it is
    # never a safe name for a destructive close; the canonical handler already
    # refuses to forward it and resolves a concrete target instead.
    # This spelling closes ONE named target and nothing else: neither operand
    # that could widen or redirect the close is reachable from this parser, so
    # pin them off before delegating.
    args.path = None
    args.all = False
    args._sticky_target = False
    args._close_spelling = "target close"
    try:
        return _close(args)
    except BridgeError as exc:
        # #996 re-dogfood: every refusal listing (unknown, ambiguous, several
        # open) renders `-t <sel>` rows, which this command rejects. Spell them
        # as `bn [-i ID] target close <sel>`, the form that runs here.
        respelled = respell_for_positional(str(exc), _target_close_prefix(args))
        if respelled == str(exc):
            raise
        raise BridgeError(respelled) from None


@command("save", help="Save the current analysis database (.bndb)", target=True,
         args=[
             arg("path", nargs="?",
                 help="Output path; a relative path resolves against the invoking shell's "
                      "cwd, not the bridge's. Default: <binary path>.bndb (a target opened "
                      "from a .bndb saves back to it). When that default cannot be written "
                      "(e.g. a read-only mount) the save falls back to "
                      "<cache>/bndb/<name>.<first 16 hex of the binary's sha256>.bndb -- "
                      "<cache> is $BN_CACHE_DIR, else the user cache dir (~/.cache/bn on "
                      "Linux) -- which a later `bn load` of the same bytes restores; an "
                      "explicit path never falls back, and the fallback refuses to "
                      "overwrite an existing entry this target was not loaded from (exit 2; JSON "
                      "{ok: false, status: invalid_request, observed: {cache_entry_exists, "
                      "request_sent: true, written: false}} -- exit 2, not the mutation exit 3, "
                      "because save is a file write, not a mutation). The cache is shared by every bn "
                      "instance using it: set BN_CACHE_DIR per agent or project to keep "
                      "their saved analyses apart -- it also moves the instance registry "
                      "and sticky session pins, so set it on every bn command"),
             arg("--path", dest="path_flag", default=None,
                 help="Output path (alias for the positional)"),
         ])
def _save(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    out_path = _pick(args.path, getattr(args, "path_flag", None), "save path", required=False)
    # Presence, not truthiness: `bn save "$OUT"` with $OUT unset must error,
    # not silently save to the default path (#690 r3, same doctrine as close).
    if out_path is not None and not str(out_path).strip():
        raise BridgeError(
            "path is empty: pass a real output path, or omit it to save to "
            "the default <filename>.bndb"
        )
    if out_path:
        params["path"] = str(Path(out_path).expanduser().resolve())
    return _call(
        args,
        "save_database",
        params,
        require_target=False,
        text_renderer=_render_save_text,
        stem="save",
    )


@command("refresh",
         help="Run full analysis on the selected target (e.g. after a --quick load). "
              "On a large target this can take minutes; reads stay responsive meanwhile, "
              "so poll `bn target info` on another connection to watch analysis_progress. "
              "Writes wait for it: py exec, mutations, save, close and decompile "
              "--force-analysis queue until the refresh finishes (a notice on stderr says "
              "so); set BN_NO_WAIT=1 to make them fail fast with `busy:` instead.",
         target=True)
def _refresh(args: argparse.Namespace) -> int:
    # Modern bridges advertise a reserved sole-target gate. The transport uses
    # that selected bridge's registry to retain pinned CLI preflight on legacy
    # bridges without blocking modern duplicate refusals on list_targets.
    # Keep the broken-default refusal local: the bridge cannot know its source.
    if getattr(args, "_empty_ambient_target", None):
        raise BridgeError(cli._empty_target_message(args))
    return _call(
        args,
        "refresh",
        {},
        require_target=False,
        text_renderer=_render_refresh_text,
        stem="refresh",
        # The one-time full re-analysis can run far past the 600s read-op default
        # on a very large binary; give it an hour by default (#321). Users still
        # override (or disable) via BN_REQUEST_TIMEOUT for the truly huge case.
        op_default_timeout=REFRESH_REQUEST_TIMEOUT,
    )


@command("target", "info", help="Show one target", target=True,
         fanout=True,
         args=[
             arg("--verbose", "-v", action="store_true",
                 help="Include the segment map (r/w/x address ranges)"),
         ],
         estimable=True)
def _target_info(args: argparse.Namespace) -> int:
    return _call(
        args,
        "target_info",
        {"selector": args.target, "verbose": bool(args.verbose)},
        require_target=True,
        # #793: `target info` and `evidence orient` describe the same view, so the
        # text view carries the annotation counts/provenance block too.
        text_renderer=_render_target_info_text_with_annotations,
        stem="target-info",
        stamp_instance=True,
    )
