"""The one multi-target hint grammar, shared by the CLI and the bridge (#688).

The same condition -- several targets open, no selector -- is refused in two
places: the CLI pre-flight (`_implicit_target`, for `require_target=True`
commands) and the bridge resolver (`TargetManager.resolve` /
`_resolve_sole_target_for_close`, for everything else and for every raw
socket client). Before this module each rendered its own listing, so an agent
that learned one shape misparsed the other and every message improvement had
to be made twice.

Rows render the `-t` form, shell-quoted, because they are a copy-paste
contract: for some commands (`bn save`) the positional is an OUTPUT PATH, so
echoing a bare selector invites `bn save <selector>` -- a silent wrong-file
write -- and an unquoted selector with a space splits into exactly that shape.

Symlinked into ``src/bn_agent_bridge/`` like ``paths.py``, so the process that
prints the hint pre-flight and the process that prints it at resolve time
cannot drift.
"""
from __future__ import annotations

import shlex
from typing import Any, Callable

SELECT_HINT_LINE = "Pass -t <selector> (--target) to choose one."
OPEN_TARGETS_HEADING = "Open targets:"
STABLE_ID_NOTE = "note: view_id / target_id are stable across `bn save`"


def target_row(target: dict[str, Any]) -> str:
    """One listing row, indent and active marker included."""
    marker = "*" if target.get("active") else " "
    return (
        f"  {marker} -t {shlex.quote(str(target.get('selector', '')))}"
        f"  view_id={target.get('view_id', '')}"
        f"  target_id={target.get('target_id', '')}"
        f"  {target.get('filename', '')}"
    )


def open_target_lines(
    targets: list[dict[str, Any]],
    *,
    row: Callable[[Any], str] = target_row,
) -> list[str]:
    """The `Open targets:` listing. ``row`` is injectable so the CLI, which
    renders rows out of a JSON reply rather than its own records, can keep its
    tolerance for a row that is not a dict."""
    return [OPEN_TARGETS_HEADING, *(row(target) for target in targets), STABLE_ID_NOTE]


def format_multi_target_hint(
    headline: str,
    targets: list[dict[str, Any]],
    *,
    row: Callable[[Any], str] = target_row,
) -> str:
    """``headline`` (the condition, which differs per refusal site) followed by
    the shared instruction line and listing."""
    return "\n".join([headline, SELECT_HINT_LINE, *open_target_lines(targets, row=row)])
