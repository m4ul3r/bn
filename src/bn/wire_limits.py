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

import os

# The wire ceiling. A request line longer than this is refused by the bridge
# (see ``BridgeHandler.handle``), so it is also the honest client-side
# ceiling: sending more can only fail.
MAX_REQUEST_BYTES = 32 * 1024 * 1024

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
