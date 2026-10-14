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

import re
import shlex
from typing import Any, Callable

SELECT_HINT_LINE = "Pass -t <selector> (--target) to choose one."
OPEN_TARGETS_HEADING = "Open targets:"
STABLE_ID_NOTE = "note: view_id / target_id are stable across `bn save`"
# #996 review: the tail of a destructive close's refusal when its selector named
# the target only through a `.bndb`-stripped alias. The bridge ends the message
# with this lead plus the shell-quoted LISTED selector (which resolves to
# exactly that target); the CLI replaces the tail with the spelling of the
# command that was run (`bn close -t X` / `bn target close X`).
LISTED_SELECTOR_LEAD = "close it by its listed selector instead: "


def _escape_controls(text: str) -> str:
    """*text* on one line: backslash, newline and carriage return escaped, so a
    crafted file name cannot start a line of its own in a listing."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def target_option(selector: str) -> str:
    """The `-t` argument for *selector*, shell-quoted. A selector starting with
    `-` would be read as the next option after a bare `-t`, so it is attached:
    `--target=<sel>` (#996 re-check)."""
    quoted = shlex.quote(selector)
    return f"--target={quoted}" if selector.startswith("-") else f"-t {quoted}"


def positional_selector(selector: str) -> str:
    """*selector* as a positional argument, shell-quoted, after `--` when it
    starts with `-` so argparse does not read it as an option (#996 re-check)."""
    quoted = shlex.quote(selector)
    return f"-- {quoted}" if selector.startswith("-") else quoted


def target_row(target: dict[str, Any]) -> str:
    """One listing row, indent and active marker included."""
    marker = "*" if target.get("active") else " "
    return (
        f"  {marker} {target_option(str(target.get('selector', '')))}"
        f"  view_id={target.get('view_id', '')}"
        f"  target_id={target.get('target_id', '')}"
        f"  {_escape_controls(str(target.get('filename', '')))}"
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


_ROW_START = re.compile(r"^  ([* ]) (?:-t |--target=)")


def _parse_row(text: str) -> tuple[bool, str, str] | None:
    """``(active, selector, metadata)`` of one `target_row` (which may span
    lines when the quoted selector holds a newline), or None when *text* is
    not a complete row."""
    start = _ROW_START.match(text)
    if start is None:
        return None
    rest = text[start.end():]
    lexer = shlex.shlex(rest, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        selector = lexer.get_token()
    except ValueError:              # the quoted selector continues on a later line
        return None
    if selector is None:
        return None
    metadata = rest[lexer.instream.tell():].strip()
    if not metadata.startswith("view_id="):
        return None
    return start.group(1) == "*", selector, metadata


def respell_for_positional(message: str, command: str) -> str:
    """Rewrite a refusal's `-t` rows and instruction line for a command whose
    selector is POSITIONAL (`bn target close <sel>`), where `-t` is rejected.

    The bridge and the CLI both render this module's grammar, whatever command
    the selector will go to; only the CLI knows that command. *command* is the
    full prefix to print, e.g. ``bn -i demo target close`` (#996 re-dogfood).

    Each row becomes ONE complete, shell-safe command: the selector quoted (and
    after `--` when it starts with `-`), the active marker and the metadata
    after a `#` comment with control characters escaped, so a whole pasted
    line runs exactly that command and nothing else (#996 re-check). Rows are
    parsed by their grammar, so a line inside a selector that contains a
    newline is never mistaken for a row of its own.
    """
    lines = message.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == SELECT_HINT_LINE:
            out.append(f"Run one of these to choose: {command} <selector>")
            index += 1
            continue
        parsed = None
        end = index
        if _ROW_START.match(line):
            # A quoted selector holding a newline continues the row; join
            # until it parses (bounded by the message).
            for end in range(index, len(lines)):
                parsed = _parse_row("\n".join(lines[index:end + 1]))
                if parsed is not None:
                    break
        if parsed is None:
            out.append(line)
            index += 1
            continue
        active, selector, metadata = parsed
        # The metadata is one line by construction: `target_row` escapes the
        # file name, and the parse stops at the end of the quoted selector.
        note = ("active  " if active else "") + metadata
        out.append(f"  {command} {positional_selector(selector)}  # {note}")
        index = end + 1
    return "\n".join(out)
