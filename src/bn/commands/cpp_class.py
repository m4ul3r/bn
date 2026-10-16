from __future__ import annotations

import argparse
from typing import Any

from ..cli import (
    _call,
    _effective_limit,
    _mutate,
    _refuse_count_only_slices,
    arg,
    command,
    mutation_output_args,
    preview_arg,
)
from ..formatters import _render_class_list_text, _render_class_show_text


@command("class", "list", help="List C++ classes recovered from symbols/RTTI",
         fanout=True,
         target=True, paged=True,
         prefer_when="recover the C++ class hierarchy (vtables, bases, methods) from RTTI/symbols; "
                     "use evidence table to walk a raw vtable/pointer table as data",
         see_also=("class show", "evidence table"),
         args=[arg("--include-hidden", "--all", action="store_true", default=False, dest="all_clusters",
                  help="Include what the default listing hides: name-only clusters "
                       "(possible namespaces), RTTI artifacts (construction vtables, "
                       "typeinfo for non-class types), and every class/struct/union "
                       "in the view's USER type container (what `types declare` "
                       "writes -- not types Binary Ninja imported), with no filtering "
                       "by the shape of a name. It does not change paging: use "
                       "--limit/--offset for more rows. --no-stl, --no-vendor and "
                       "--local still narrow these rows; a normal listing reports "
                       "how many each hid, while --count returns only the included "
                       "count. --query reports no hidden count. BN's C parser "
                       "registers an anonymous typedef's body as a second type "
                       "(`typedef struct { ... } T;` yields `T` and `_T`), and that "
                       "half lists too. --all is the older spelling"),
               arg("--no-stl", action="store_true", default=False, dest="no_stl",
                   help="Hide standard-library / ABI-runtime classes (std::, __gnu_cxx::, __cxxabiv1::, reserved-id internals) so domain classes surface"),
               arg("--no-vendor", nargs="?", const="", default=None, dest="no_vendor",
                   metavar="NAMESPACES",
                   help="Hide vendored library classes by top-level namespace, the same way "
                        "--no-stl folds std: boost always, plus any comma-separated "
                        "namespaces given (--no-vendor Poco,acme). For classes another "
                        "module defines, whatever their namespace, see --local"),
               arg("--local", action="store_true", default=False, dest="local_only",
                   help="Hide classes with nothing defined in this binary: every RTTI "
                        "object they have is an import (or a copy relocation of one) and "
                        "every method clustered under them is an import stub"),
               arg("--vtable-budget", type=float, default=None, dest="vtable_budget",
                   metavar="SECONDS",
                   help="Time budget for recovering stripped vtables on the returned page "
                        "(default 0.05, at most 30); rows left unchecked say so"),
               arg("--query", help="Filter classes by name substring"),
               arg("--count", action="store_true", default=False, dest="count_only",
                   help="Return just the class count (respects --no-stl/--no-vendor/--local/"
                        "--query/--include-hidden), plus artifact_count -- fast "
                        "class-lens scale characterization")],
         estimable=True)
def _class_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.query:
        params["query"] = args.query
    if args.all_clusters:
        params["include_all"] = True
    if args.no_stl:
        params["no_stl"] = True
    if args.no_vendor is not None:
        params["no_vendor"] = True
        prefixes = [p.strip() for p in args.no_vendor.split(",") if p.strip()]
        if prefixes:
            params["vendor_prefixes"] = prefixes
    if args.local_only:
        params["local_only"] = True
    if args.vtable_budget is not None:
        params["vtable_budget"] = args.vtable_budget
    if args.count_only:
        # #872: `class list --count` used to FORWARD --limit alongside
        # count_only, so the bridge received a page size it cannot apply to a
        # scalar. Refuse by name like every other count surface (#767/#768)
        # rather than sending a flag that cannot mean anything.
        _refuse_count_only_slices(args, command="class list")
        params["count_only"] = True
    else:
        params["offset"] = args.offset
        limit = _effective_limit(args)
        if limit is not None:
            params["limit"] = limit
    return _call(
        args, "class_list", params,
        require_target=True,
        text_renderer=_render_class_list_text,
        paged_spill=True, page_label="classes", stem="class-list",
    )


@command("class", "show", help="Show a C++ class: methods, vtable, size, bases, instances",
         target=True,
         args=[arg("name")],
         estimable=True)
def _class_show(args: argparse.Namespace) -> int:
    return _call(
        args, "class_show", {"name": args.name},
        require_target=True,
        text_renderer=_render_class_show_text,
        # #413: an ambiguous leaf still renders its matches, but exits non-zero so
        # it can't be misread as a clean single-class success.
        result_exit_code=lambda r: 2 if isinstance(r, dict) and r.get("ambiguous") else 0,
        stem="class-show",
    )


@command("class", "apply-types",
         help="Type a C++ class for named virtual calls: its instance struct's vptr as BN's "
              "<Class>::VTable* and `this` on its methods, as one verified mutation",
         target=True, fmt="json",
         see_also=("class show", "struct show", "local retype"),
         args=[preview_arg(), *mutation_output_args(), arg("name")])
def _class_apply_types(args: argparse.Namespace) -> int:
    return _mutate(args, "class_apply_types", {"name": args.name},
                   preview=bool(args.preview), stem="class-apply-types")
