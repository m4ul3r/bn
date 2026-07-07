from __future__ import annotations

import argparse
from typing import Any

from ..cli import _call, _effective_limit, arg, command
from ..formatters import _render_class_list_text, _render_class_show_text


@command("class", "list", help="List C++ classes recovered from symbols/RTTI",
         fanout=True,
         target=True, paged=True,
         prefer_when="recover the C++ class hierarchy (vtables, bases, methods) from RTTI/symbols; "
                     "use evidence table to walk a raw vtable/pointer table as data",
         see_also=("class show", "evidence table"),
         args=[arg("--all", action="store_true", default=False, dest="all_clusters",
                   help="Include name-only clusters (possible namespaces), not just RTTI/ctor-confirmed classes"),
               arg("--no-stl", action="store_true", default=False, dest="no_stl",
                   help="Hide standard-library / ABI-runtime classes (std::, __gnu_cxx::, __cxxabiv1::, reserved-id internals) so domain classes surface"),
               arg("--no-vendor", action="store_true", default=False, dest="no_vendor",
                   help="Hide vendored/in-tree library classes (boost::) the same way --no-stl folds std"),
               arg("--query", help="Filter classes by name substring"),
               arg("--count", action="store_true", default=False, dest="count_only",
                   help="Return just the class count (respects --no-stl/--no-vendor/--query/"
                        "--all), plus artifact_count -- fast class-lens scale characterization")])
def _class_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"offset": args.offset}
    if args.query:
        params["query"] = args.query
    if args.all_clusters:
        params["include_all"] = True
    if args.no_stl:
        params["no_stl"] = True
    if args.no_vendor:
        params["no_vendor"] = True
    if args.count_only:
        params["count_only"] = True
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
         args=[arg("name")])
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
