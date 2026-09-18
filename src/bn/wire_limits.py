"""The one place the request-size ceiling is stated, and the batch-apply
preflight derived from it.

The bridge has always capped a request at ``MAX_REQUEST_BYTES`` and answered
an oversized one with a bare ``request too large`` -- after the client had
read the whole manifest into memory, serialized it, and paid a round trip.
``bn batch apply`` had no client-side guard at all (#769), so a generated
manifest failed late with a message that named neither the limit nor a way
around it.

This module is symlinked into ``bn_agent_bridge`` exactly like ``paths.py``
and ``version.py``, so the process that REFUSES an oversized request and the
process that warns before sending one read the same number. A by-value copy
of ``32 * 1024 * 1024`` in the CLI would be the duplicated-constant drift
shape that #777 and #890 were filed for -- and a client that guessed LOW
would refuse requests the bridge would have accepted.
"""
from __future__ import annotations

import json
import os
from typing import Any

# The wire ceiling. A request line longer than this is refused by the bridge
# (see ``BridgeHandler.handle``), so it is also the honest client-side
# ceiling: sending more can only fail.
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# What the bridge caps is the SERIALIZED REQUEST, so that is what a
# client-side guard must measure. The manifest FILE is not that quantity and
# not a safe proxy for it, in either direction (#769 review):
#
#   * the file is re-serialized before it is sent, so indentation is
#     discarded -- a pretty-printed 3909-byte manifest and its 2779-byte
#     compact twin produce the SAME 3024-byte request. Judging the file
#     refuses inputs the bridge would accept, and the refusal's stated
#     reason ("sending it can only fail") is then factually false;
#   * the request also carries an envelope the file does not -- id, op,
#     target, bridge identity -- so a compact file exactly at the cap
#     produces an OVER-cap request and dies at the bridge with the bare
#     `request too large` this guard exists to pre-empt.
#
# `REQUEST_ENVELOPE_BYTES` is the FIXED non-params part -- id, op, bridge
# identity, braces -- measured at 198 and rounded up. The selector is NOT
# fixed and is not covered by it: `-t` is injected into params AND into the
# envelope after the guard runs, so both copies used to escape a flat
# reserve. Measured, the real non-params size is
# `198 + 2*(len(selector)+14) + 17 if --preview`, and with a constant
# reserve the guard let a request through at a 144-char selector -- the bare
# `request too large` this guard exists to pre-empt, reached through the
# selector instead of the file (#889c finding 2).
REQUEST_ENVELOPE_BYTES = 256
_SELECTOR_KEY_BYTES = 14          # `"target":"…"` framing, per copy
_PREVIEW_BYTES = 17               # `"preview":true,`


def request_bytes_for_params(params: Any, *, selector: str | None = None,
                             preview: bool = False) -> int:
    """Bytes the request carrying *params* will occupy on the wire.

    Serialized the same way `transport` serializes it, so the guard and the
    sender measure one quantity rather than two. *selector* and *preview*
    are the parts folded in AFTER this check runs, and are counted here
    because a caller cannot be refused for bytes and then silently grow.
    """
    total = len(json.dumps(params).encode("utf-8")) + REQUEST_ENVELOPE_BYTES
    if selector:
        total += 2 * (len(str(selector)) + _SELECTOR_KEY_BYTES)
    if preview:
        total += _PREVIEW_BYTES
    return total


# Op-count ceiling for one `batch apply` manifest. Not a wire limit -- a
# manifest well under the byte cap can still hold enough operations to hold
# the write lock for minutes, and a runaway generator is far likelier than a
# deliberate 50k-op batch. Chosen to be comfortably above any hand-written or
# tool-generated batch seen in this repo's own dogfooding while still
# catching a loop that ran away.
BATCH_APPLY_MAX_OPS = 5000

MAX_OPS_ENV = "BN_BATCH_APPLY_MAX_OPS"
MAX_BYTES_ENV = "BN_BATCH_APPLY_MAX_BYTES"


def _limit_from_env(name: str, default: int) -> int | None:
    """Resolve an override, or ``None`` to mean "no limit".

    ``0`` disables the check -- the documented escape hatch for a caller who
    really does want a 50k-op batch and has accepted the consequences. A
    malformed or negative value falls back to *default* rather than raising:
    this is a guard, and a typo'd environment variable must not turn into a
    hard failure on an otherwise valid command.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        return default
    if value < 0:
        return default
    return None if value == 0 else value


def batch_apply_max_ops() -> int | None:
    return _limit_from_env(MAX_OPS_ENV, BATCH_APPLY_MAX_OPS)


def batch_apply_max_bytes() -> int | None:
    return _limit_from_env(MAX_BYTES_ENV, MAX_REQUEST_BYTES)
