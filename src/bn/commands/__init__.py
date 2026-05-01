"""Command handler submodules.

Importing this package triggers @command decorators in each submodule,
populating the central ``_COMMANDS`` registry in ``bn.cli``.
"""

from __future__ import annotations

from . import binary  # noqa: F401
from . import function  # noqa: F401
from . import misc  # noqa: F401
from . import mutation  # noqa: F401
from . import types  # noqa: F401
