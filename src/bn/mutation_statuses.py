"""The canonical mutation failure-status set, shared by the CLI and the bridge.

This module exists because the set had TWO owners (#777). ``bn.formatters``
declared the authoritative five, and ``bn_agent_bridge.mutation_engine``
re-derived its own narrower copy inline -- so the module that PRODUCES these
status strings disagreed with the module that CLASSIFIES them, and any new
``_verify_*`` raising a status outside the narrow copy would silently skip the
revert-on-failure path while the CLI still reported the failure.

They could not simply import each other: the bridge imports stdlib plus
``binaryninja`` and never imports ``bn`` (a BN plugin must not depend on the CLI
package). The project's established answer to exactly this problem is a small
stdlib-only leaf module in ``src/bn/`` OS-symlinked into
``src/bn_agent_bridge/`` -- the same arrangement ``paths.py``, ``version.py``,
``proc_identity.py``, ``socket_evidence.py`` and ``target_hint.py`` already use,
so the two processes agree by construction instead of by review. This file is
that arrangement for the failure-status vocabulary; keep it stdlib-only or the
symlink stops being importable from the bridge.

``bn.formatters.FAILED_MUTATION_STATUSES`` remains a valid public name (it
re-exports from here), because CLAUDE.md documents it at that location and
tests pin it there.
"""
from __future__ import annotations

# The phase of the batch each failure status comes from, which is what a
# failure banner names (W07: it used to say "live verification failed" for an
# op that never got past apply, and for a preview). A semantic refusal
# (`invalid_request`), an op Binary Ninja cannot perform (`unsupported`) and an
# unexpected engine error (`internal_error`) stop the batch at or before
# apply; readback disagreement (`verification_failed`) is the post-apply
# verify; `rollback_failed` = an op succeeded but the batch revert that should
# have undone it failed, so the view may be left modified -- a real failure. A
# cleanly rolled-back sibling ("reverted") is NOT a failure and is omitted
# (#118); `internal_error` stays distinct from an unsupported request (#122).
FAILURE_PHASES = {
    "unsupported": "apply",
    "invalid_request": "apply",
    "internal_error": "apply",
    "verification_failed": "verify",
    "rollback_failed": "rollback",
}

# Keyed off the phases so a new failure status cannot exist without one.
FAILED_MUTATION_STATUSES = set(FAILURE_PHASES)
