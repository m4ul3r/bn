"""Positive evidence that a socket path is BOUND, shared by both processes (#618).

Both halves of this tool destroy socket files: the CLI's `gc` sweep and the
bridge's own `start()`, which clears a crashed predecessor's leftover before
binding. Both were deciding it from a failed `connect()`, which cannot answer
the question -- a socket bound but not yet past `listen` refuses identically to
a leftover file, and every bridge passes through that state coming up -- so a
wrong negative unlinks a live bridge's endpoint and leaves it serving on an
unlinked inode.

The kernel does know, and this module is the one reader of its answer. It is
SYMLINKED into ``src/bn_agent_bridge/`` like ``paths.py``, ``version.py`` and
``proc_identity.py``, so the rule cannot drift between the process that unlinks
and the process whose endpoint gets unlinked. Stdlib only, for the same reason:
the bridge's modules import stdlib plus ``binaryninja``, never ``bn``.
"""

from __future__ import annotations

import os
from pathlib import Path

_LISTING = Path("/proc/net/unix")


def bound_socket_listing_available() -> bool:
    """Can this kernel be ASKED which paths are bound?

    ``path_has_bound_socket`` answers ``None`` for two different situations,
    and one consumer has to tell them apart. "This path cannot be represented
    in the listing" is a fact about the path, and keeping the file costs
    nothing. "There is no listing on this platform" is a fact about every path
    on the host -- so a consumer that must DISPLACE the file (the bridge
    binding its own fixed socket path, as opposed to a sweep that can simply
    skip) would refuse forever on Darwin/BSD after one unclean shutdown, with
    no in-tool recovery. It falls back to the weaker connect() evidence there
    and says so; everywhere a listing exists, the strong rule stands.
    """
    try:
        _LISTING.read_bytes()
    except OSError:
        return False
    return True


def path_has_bound_socket(socket_path: str | Path) -> bool | None:
    """Whether any socket is BOUND to *socket_path*; ``None`` when unknowable.

    ``connect`` cannot answer this question. A socket that is bound but has not
    yet called ``listen`` refuses connections exactly like a crashed bridge's
    leftover file -- measured on this kernel, bound-without-listen, a leftover
    socket file and a plain file that never was a socket all give
    ``ECONNREFUSED`` -- and every bridge passes through that state on its way
    up. Treating the errno as proof that nothing is bound therefore unlinks a
    STARTING bridge's own endpoint and leaves it serving on an unlinked inode,
    which is the failure this arm exists to prevent (#618).

    The kernel does know, and lists every bound path in ``/proc/net/unix``. Off
    Linux that file does not exist, and the answer is then UNKNOWABLE rather
    than "nothing is bound": ``None`` keeps the caller from destroying on an
    absence, which is the rule the rest of this module follows.

    Unknowable also covers what the listing cannot REPRESENT, and that
    distinction is the whole safety of this function -- a wrong ``False`` is
    the sole corroboration behind every socket unlink in this module, so it
    destroys a bridge's own bound-and-listening endpoint. Four shapes:

    * The file is line-oriented, so a path containing a newline cannot appear
      in it at all. That is not evidence of nothing being bound, so it answers
      ``None``.
    * A path is bytes, not text. Decoding the listing as UTF-8 with
      ``errors="replace"`` turned any non-UTF-8 byte in a cache path into
      U+FFFD and made every comparison fail, which read as "nothing is bound"
      about a serving socket. The comparison is therefore done on BYTES, the
      form the kernel wrote and the form ``bind`` was given, so such a path is
      answered exactly rather than approximately.
    * The listing holds the string ``bind`` was given and nothing else -- not
      the binder's working directory. A RELATIVE row is therefore meaningful
      only in a cwd this reader does not know and the file cannot express, and
      resolving it against the READER's cwd is the same losing transformation
      as the other two: under a relative cache root (the supported answer to
      the AF_UNIX length limit, so the bridge really does bind a relative
      string) a CLI run from anywhere else answered "nothing is bound" about a
      live listening socket and ``gc`` unlinked it. A relative row that could
      be this path -- same basename, and ``bind`` creates the final component
      so it is never a symlink -- makes the answer ``None``. One with a
      different basename cannot be this path in any cwd, and is skipped, so an
      unrelated relative socket elsewhere on the host does not make every
      question unanswerable. An ABSTRACT-namespace row is covered by the same
      arm, and deliberately: the kernel prints it as ``@`` followed by a name
      that may itself contain slashes, and it prints a PATHNAME socket's path
      verbatim, so a relative cache root whose first component begins with
      ``@`` is byte-identical to an abstract row. Skipping every ``@`` row as
      "names no file" therefore destroyed that bridge's endpoint. The listing
      cannot tell the two apart and neither can this reader: ``None``, at the
      cost of retention -- an abstract name ending in an orphan's name keeps
      that orphan file, which is bounded by basename and reachable anyway by
      binding an absolute path and unlinking it.
    * The listing holds the name ``bind`` was given, not the name that file
      has NOW. Rename the directory of a socket that is still bound and
      listening -- an ordinary operator action -- and the row goes on naming a
      path that no longer resolves to anything, so the comparison fails on a
      SERVING socket and the sweep unlinked it. A row whose name has ceased to
      exist cannot be compared to anything, which is not evidence that nothing
      is bound, so it too answers ``None``.

    The argument is accepted as a ``str`` or a ``Path`` and normalized to bytes
    exactly once, because this module is symlinked into the bridge and both
    processes call it -- one with a ``Path`` it built, one with a registry path
    it read out of JSON (#733 F4).
    """
    wanted = os.fsencode(socket_path)
    if b"\n" in wanted:
        return None
    resolved = os.path.realpath(wanted)
    if b"\n" in resolved:
        return None
    try:
        listing = _LISTING.read_bytes()
    except OSError:
        return None
    # The basename of the bytes above, rather than a second `fsencode` of a
    # `Path`-only attribute: this function is also handed a plain string (a
    # registry path read out of JSON), and `socket_path.name` made that an
    # `AttributeError` that read as a bridge defect (#733 F4). `normpath`
    # first, because `Path` normalizes a trailing separator and a `/.` tail
    # while `posixpath.basename` does not -- `basename(b"/p/x.sock/")` is
    # `b""`, which would make the basename filter below match no row and
    # answer the one thing this module must never fabricate, a positive
    # "nothing is bound". A path that names no file at all (`/`) is
    # unknowable rather than unbound.
    name = os.path.basename(os.path.normpath(wanted))
    if not name:
        return None
    for line in listing.split(b"\n"):
        # `Num RefCount Protocol Flags Type St Inode Path`, whitespace-separated,
        # and the trailing path is present only for a BOUND socket. Splitting on
        # the first seven runs keeps a path containing spaces intact.
        fields = line.split(None, 7)
        if len(fields) < 8:
            continue
        bound_path = fields[7]
        # The record may spell the path differently from the string the bridge
        # passed to `bind` (a symlinked `instances/`), so compare what the two
        # names resolve to -- filtered by basename first, because a busy host
        # lists thousands of sockets. An exact byte match needs no resolution
        # and is taken first: it can only ever REFUSE a destruction.
        if bound_path == wanted:
            return True
        if os.path.basename(bound_path) != name:
            continue
        if not os.path.isabs(bound_path):
            # Resolvable only in the binder's cwd, which this file does not
            # carry -- and an abstract row (`@name`) is indistinguishable from
            # one. Never answer the positive "nothing is bound" from either.
            return None
        target = os.path.realpath(bound_path)
        if target == resolved:
            return True
        if not os.path.lexists(target):
            # The name this row was bound under is gone (its directory was
            # renamed or moved while the socket kept serving), so there is
            # nothing left to compare it to -- and this file may well BE it.
            return None
    return False
