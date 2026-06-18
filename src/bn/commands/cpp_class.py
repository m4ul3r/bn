from __future__ import annotations

import argparse
from typing import Any

from ..cli import _call, _effective_limit, arg, command
from ..formatters import _render_class_list_text, _render_class_show_text


@command("class", "list", help="List C++ classes recovered from symbols/RTTI",
         target=True, paged=True,
         args=[arg("--all", action="store_true", default=False, dest="all_clusters",
                   help="Include name-only clusters (possible namespaces), not just RTTI/ctor-confirmed classes"),
               arg("--no-stl", action="store_true", default=False, dest="no_stl",
                   help="Hide standard-library / ABI-runtime classes (std::, __gnu_cxx::, __cxxabiv1::, reserved-id internals) so domain classes surface"),
               arg("--query", help="Filter classes by name substring")])
def _class_list(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {"offset": args.offset}
    if args.query:
        params["query"] = args.query
    if args.all_clusters:
        params["include_all"] = True
    if args.no_stl:
        params["no_stl"] = True
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
        stem="class-show",
    )
