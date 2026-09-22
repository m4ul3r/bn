"""`bn spill` command group: host-side spill-cache maintenance (#823).

The spill root (``<cache>/spills/YYYYMMDD/``) is local cache state, not bridge
state, so this group needs no target and sends nothing over the socket -- the
same shape as `bn instance gc`, and the reason it can run on a machine with no
Binary Ninja at all.
"""
from __future__ import annotations

import argparse

from ..cli import _emit_result, _int_or_hex, arg, command
from ..formatters import _render_spill_gc_text
from ..output import gc_spills


def _parse_age_days(value: str) -> int:
    """Parse ``--older-than``: a number of DAYS, optionally with a ``d`` suffix.

    Days is the unit the spill root is NAMED in -- one directory per calendar
    day -- and the unit ``BN_SPILL_RETENTION_DAYS`` is measured in, so a
    sub-day window would have to round to a day anyway and would silently
    over-delete the current day's artifacts. The optional ``d`` accepts the
    natural spelling (``--older-than 30d``) without inventing a second
    duration grammar for it.

    Zero is refused rather than reusing the env var's ``0`` = keep-everything
    meaning: "remove what is older than nothing" reads as "remove everything"
    to anyone who has not read that convention, and a gc command must not be
    one typo away from deleting an engagement's whole cache.
    """
    text = value.strip().lower()
    if text.endswith("d"):
        text = text[:-1]
    try:
        days = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of days (e.g. 30 or 30d), got {value!r}"
        ) from None
    if days < 1:
        raise argparse.ArgumentTypeError(
            f"--older-than requires a positive number of days, got {value!r}"
        )
    return days


def _parse_max_bytes(value: str) -> int:
    """``--max-bytes``: a byte count, through the CLI's one size parser.

    ``_int_or_hex`` is the helper every other size argument uses (``read
    --length``, ``--stride``), so this flag takes the same decimal and
    ``0x``/``0o``/``0b`` forms instead of a second grammar. It only adds the
    lower bound: a negative cap is not a smaller cap, it is a request to keep
    nothing.
    """
    size = _int_or_hex(value)
    if size < 0:
        raise argparse.ArgumentTypeError(
            f"--max-bytes must be >= 0, got {value!r}"
        )
    return size


@command("spill", "gc",
         help="Reclaim spill-cache day directories (inspect with --dry-run; bound by "
              "age and/or size)",
         fmt="text",
         args=[
             arg("--dry-run", action="store_true", default=False,
                 help="Report the candidates, their sizes and the kept counts without "
                      "removing anything"),
             arg("--older-than", type=_parse_age_days, default=None, metavar="DAYS",
                 help="Remove day directories older than DAYS days (a `d` suffix is "
                      "accepted, e.g. 30d). Default: BN_SPILL_RETENTION_DAYS (14), "
                      "the window a spill prunes with"),
             arg("--max-bytes", type=_parse_max_bytes, default=None, metavar="N",
                 help="After the age pass, evict the OLDEST surviving days until the "
                      "day directories hold at most N bytes (decimal or 0x.. hex)"),
         ])
def _spill_gc(args: argparse.Namespace) -> int:
    result = gc_spills(older_than_days=args.older_than, max_bytes=args.max_bytes,
                       dry_run=bool(args.dry_run))
    _emit_result(args, result, text_renderer=_render_spill_gc_text, stem="spill-gc")
    return 0
