"""`bn tag` command group: tag types, reads, and mutations."""
from __future__ import annotations

import argparse

from ..cli import _call, arg, command
from ..formatters import _render_tag_types_text


@command("tag", "types", help="List tag types (name, icon, built-in)", target=True)
def _tag_types(args: argparse.Namespace) -> int:
    return _call(
        args,
        "list_tag_types",
        {},
        require_target=True,
        text_renderer=_render_tag_types_text,
        stem="tag-types",
    )
