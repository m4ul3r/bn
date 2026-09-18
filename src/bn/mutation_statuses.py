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

# "rollback_failed" = an op succeeded but the batch revert that should have
# undone it failed, so the view may be left modified -- a real failure. A
# cleanly rolled-back sibling ("reverted") is NOT a failure and is omitted (#118).
# "internal_error" = an unexpected engine bug (distinct from an unsupported
# request); still a failure, so exit codes/rendering flag it (#122).
FAILED_MUTATION_STATUSES = {
    "unsupported",
    "verification_failed",
    "invalid_request",
    "rollback_failed",
    "internal_error",
}
