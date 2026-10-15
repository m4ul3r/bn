"""ELF dynamic-section facts BN does not expose: SONAME, RPATH/RUNPATH and the
library each versioned import binds to (#969).

BN's ELF view publishes the DT_NEEDED list (``bv.libraries``) and nothing else
from ``.dynamic``: an import symbol carries a sentinel namespace, so every
`imports` row read ``library: null`` and the agent fell back to guessing from
``.dynstr``. The loader's own answer is in the file: a dynamic symbol
whose ``.gnu.version`` index names a ``.gnu.version_r`` (verneed) entry is bound
to that entry's library (``printf@GLIBC_2.17`` -> ``libc.so.6``).

The parser reads the ORIGINAL file bytes through a ``read(offset, size)``
callable (the view's raw parent), and walks program headers rather than section
headers, so a section-stripped binary answers the same way. It is pure -- no
BinaryNinja import.

Every size and count in an ELF file is attacker-controlled, and the reader is
BN's ``BinaryView.read``, which SEGFAULTS the whole process on a large enough
request (review of #987: a copy of ``ls`` with DT_SYMENT set to ``1 << 56``
killed the bridge on ``bn imports``). So no file-derived value reaches the
reader unchecked: ``_Elf.bytes_at`` is the one gate every read goes through,
and it refuses any window that is not inside the file or is larger than
``_MAX_READ``. On top of that gate each structure is validated where it is
decoded -- an entry size that is not the ABI's, a table that does not fit in
the file -- and a structure that fails is reported in ``notes`` and skipped
("not parseable"), never guessed around. A malformed or truncated file yields
``None`` or a partial answer with notes; it never raises and never asks the
reader for an out-of-range window.
"""
from __future__ import annotations

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field

_PT_LOAD = 1
_PT_DYNAMIC = 2

_DT_NULL = 0
_DT_NEEDED = 1
_DT_HASH = 4
_DT_STRTAB = 5
_DT_SYMTAB = 6
_DT_STRSZ = 10
_DT_SYMENT = 11
_DT_SONAME = 14
_DT_RPATH = 15
_DT_RUNPATH = 29
_DT_GNU_HASH = 0x6FFFFEF5
_DT_VERSYM = 0x6FFFFFF0
_DT_VERNEED = 0x6FFFFFFE
_DT_VERNEEDNUM = 0x6FFFFFFF

_SHT_DYNSYM = 11

# Sanity bounds on counts read from the file, independent of its size.
_MAX_PROGRAM_HEADERS = 4096
_MAX_SECTIONS = 1 << 16
_MAX_DYNAMIC_ENTRIES = 4096
_MAX_SYMBOLS = 1 << 20
_MAX_VERNEED = 4096
_MAX_STRING = 4096
# The largest single read the parser ever issues. The biggest legitimate one is
# the dynamic symbol table: _MAX_SYMBOLS * 24 bytes.
_MAX_READ = _MAX_SYMBOLS * 24

Reader = Callable[[int, int], bytes]


@dataclass
class ElfDynamicInfo:
    needed: list[str] = field(default_factory=list)
    soname: str | None = None
    rpath: str | None = None
    runpath: str | None = None
    # Dynamic symbol name -> the verneed library its version binds it to. Only
    # names with ONE library: a name versioned against two is not attributed.
    import_libraries: dict[str, str] = field(default_factory=dict)
    # ELF e_machine, so a library read for attribution can be checked against
    # the binary it is read for.
    machine: int | None = None
    # Names this file's dynamic symbol table DEFINES and exports to other
    # modules (global/weak/unique binding, default/protected visibility). Only
    # filled when `parse_elf_dynamic(..., with_exports=True)`; None when the
    # table could not be read, so "exports nothing" is never a guess.
    exports: frozenset[str] | None = None
    # Why part of the dynamic section was not parsed: a value out of range for
    # this file, so that part is skipped rather than guessed. Empty when the
    # whole answer was read.
    notes: list[str] = field(default_factory=list)


class _Elf:
    def __init__(self, read: Reader, file_size: int, is64: bool, endian: str):
        self._read = read
        self.file_size = file_size
        self.is64 = is64
        self.endian = endian
        self.loads: list[tuple[int, int, int]] = []  # (vaddr, filesz, offset)
        self.notes: list[str] = []

    def fits(self, offset: int, size: int) -> bool:
        """Is ``[offset, offset + size)`` a window this parser may read?"""
        return 0 <= offset and 0 <= size <= _MAX_READ and offset + size <= self.file_size

    def bytes_at(self, offset: int, size: int) -> bytes | None:
        """The ONE gate to the reader: an out-of-range window is refused here,
        so a hostile size can never reach ``BinaryView.read``."""
        if not self.fits(offset, size):
            return None
        try:
            data = bytes(self._read(offset, size) or b"")
        except Exception:  # noqa: BLE001 - a failed read is a short read
            return None
        return data if len(data) == size else None

    def unpack(self, fmt: str, offset: int) -> tuple | None:
        fmt = self.endian + fmt
        data = self.bytes_at(offset, struct.calcsize(fmt))
        return struct.unpack(fmt, data) if data is not None else None

    def file_offset(self, vaddr: int) -> int | None:
        for start, size, offset in self.loads:
            if start <= vaddr < start + size:
                at = offset + (vaddr - start)
                return at if at < self.file_size else None
        return None

    def cstring(self, offset: int, limit: int) -> str | None:
        size = min(limit, _MAX_STRING, self.file_size - offset)
        data = self.bytes_at(offset, size) if size > 0 else None
        if not data:
            return None
        end = data.find(b"\x00")
        if end < 0:
            return None
        return data[:end].decode("utf-8", errors="replace")


def parse_elf_dynamic(read: Reader, file_size: int, *,
                      with_exports: bool = False) -> ElfDynamicInfo | None:
    """The dynamic-section facts of the ELF file *read* serves (*file_size*
    bytes long), or ``None`` when it is not an ELF, carries no PT_DYNAMIC (a
    static binary), or its headers are out of range for the file.
    *with_exports* also reads the names the file defines for other modules
    (``ElfDynamicInfo.exports``), which a shared library read for import
    attribution needs (#1037 item 6)."""
    if file_size <= 0:
        return None
    try:
        return _parse(read, int(file_size), with_exports=with_exports)
    except (struct.error, ValueError, OverflowError):
        return None


def _parse(read: Reader, file_size: int, *, with_exports: bool = False) -> ElfDynamicInfo | None:
    probe = _Elf(read, file_size, False, "<")
    ident = probe.bytes_at(0, 16)
    if ident is None or ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2):
        return None
    elf = _Elf(read, file_size, ident[4] == 2, "<" if ident[5] == 1 else ">")
    machine = elf.unpack("H", 0x12)
    # e_phoff, then (past e_shoff/e_flags/e_ehsize) e_phentsize and e_phnum.
    phoff = elf.unpack("Q" if elf.is64 else "I", 0x20 if elf.is64 else 0x1C)
    counts = elf.unpack("HH", 0x36 if elf.is64 else 0x2A)
    if phoff is None or counts is None:
        return None
    phoff, (phentsize, phnum) = phoff[0], counts
    if not phoff or not phnum:
        return None
    # The entry size is fixed by the ABI; anything else is a file this parser
    # cannot read the program headers of, not one to stride through.
    if phentsize != (56 if elf.is64 else 32) or phnum > _MAX_PROGRAM_HEADERS \
            or not elf.fits(phoff, phentsize * phnum):
        return None
    dynamic: tuple[int, int] | None = None
    for index in range(phnum):
        at = phoff + index * phentsize
        if elf.is64:
            ph = elf.unpack("IIQQQQ", at)  # type, flags, offset, vaddr, paddr, filesz
            if ph is None:
                return None
            p_type, p_offset, p_vaddr, p_filesz = ph[0], ph[2], ph[3], ph[5]
        else:
            ph = elf.unpack("IIIII", at)  # type, offset, vaddr, paddr, filesz
            if ph is None:
                return None
            p_type, p_offset, p_vaddr, p_filesz = ph[0], ph[1], ph[2], ph[4]
        if p_type == _PT_LOAD:
            elf.loads.append((p_vaddr, p_filesz, p_offset))
        elif p_type == _PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        return None
    entries = _dynamic_entries(elf, *dynamic)
    if entries is None:
        return ElfDynamicInfo(notes=elf.notes)
    tags: dict[int, int] = {}
    needed_offsets: list[int] = []
    for tag, value in entries:
        if tag == _DT_NEEDED:
            needed_offsets.append(value)
        else:
            tags.setdefault(tag, value)
    info = ElfDynamicInfo(notes=elf.notes, machine=machine[0] if machine else None)
    strtab = elf.file_offset(tags[_DT_STRTAB]) if _DT_STRTAB in tags else None
    if strtab is None:
        if _DT_STRTAB in tags:
            elf.notes.append("DT_STRTAB is outside the file's loaded segments; "
                             "library names were not read")
        return info
    # A string table cannot extend past the file; one that claims to is read
    # only up to the end of the file, and said so.
    strsz = tags.get(_DT_STRSZ, _MAX_STRING)
    if strsz > file_size - strtab:
        elf.notes.append(f"DT_STRSZ {strsz} runs past the end of the file; "
                         "strings were read only up to the end of the file")
        strsz = file_size - strtab

    def dynstr(index: int) -> str | None:
        if index < 0 or index >= strsz:
            return None
        return elf.cstring(strtab + index, strsz - index)

    info.needed = [name for name in (dynstr(i) for i in needed_offsets) if name]
    info.soname = dynstr(tags[_DT_SONAME]) if _DT_SONAME in tags else None
    info.rpath = dynstr(tags[_DT_RPATH]) if _DT_RPATH in tags else None
    info.runpath = dynstr(tags[_DT_RUNPATH]) if _DT_RUNPATH in tags else None
    info.import_libraries = _versioned_import_libraries(elf, tags, dynstr)
    if with_exports:
        info.exports = _defined_exports(elf, tags, dynstr)
    return info


def _dynamic_entries(elf: _Elf, offset: int, size: int) -> list[tuple[int, int]] | None:
    entsize = 16 if elf.is64 else 8
    count = min(size // entsize, _MAX_DYNAMIC_ENTRIES)
    data = elf.bytes_at(offset, count * entsize)
    if data is None:
        elf.notes.append(f"PT_DYNAMIC ({size} bytes at file offset {offset}) is not "
                         "inside the file; the dynamic section was not parsed")
        return None
    fmt = elf.endian + ("qQ" if elf.is64 else "iI")
    entries = []
    for tag, value in struct.iter_unpack(fmt, data):
        if tag == _DT_NULL:
            break
        entries.append((tag, value))
    return entries


def _symbol_count(elf: _Elf, tags: dict[int, int]) -> int | None:
    """How many dynamic symbols there are, from the most exact source present:
    the `.dynsym` section header, DT_HASH's nchain, or the highest index the
    DT_GNU_HASH chains reach. None of them -> unknown, and nothing is attributed.

    A GNU hash table with no hashed symbol says nothing about the count (the
    undefined symbols it does not hash may sit on either side of `symoffset`),
    so an all-empty bucket array is not read as one."""
    count = _dynsym_section_count(elf)
    if count is not None:
        return count
    if _DT_HASH in tags:
        at = elf.file_offset(tags[_DT_HASH])
        header = elf.unpack("II", at) if at is not None else None
        if header is not None:
            return header[1]
    if _DT_GNU_HASH in tags:
        return _gnu_hash_symbol_count(elf, tags[_DT_GNU_HASH])
    return None


def _gnu_hash_symbol_count(elf: _Elf, vaddr: int) -> int | None:
    at = elf.file_offset(vaddr)
    header = elf.unpack("IIII", at) if at is not None else None
    if header is None:
        return None
    nbuckets, symoffset, bloom_size = header[0], header[1], header[2]
    bloom_word = 8 if elf.is64 else 4
    buckets_at = at + 16 + bloom_size * bloom_word
    if nbuckets > _MAX_SYMBOLS or bloom_size > _MAX_SYMBOLS \
            or not elf.fits(buckets_at, nbuckets * 4):
        elf.notes.append(f"DT_GNU_HASH declares {nbuckets} buckets and {bloom_size} "
                         "bloom words, which do not fit in the file; it was not "
                         "used to count symbols")
        return None
    data = elf.bytes_at(buckets_at, nbuckets * 4)
    if data is None:
        return None
    last = max(struct.unpack(f"{elf.endian}{nbuckets}I", data), default=0)
    if last < symoffset:
        return None
    chains_at = buckets_at + nbuckets * 4
    while last < _MAX_SYMBOLS:
        word = elf.unpack("I", chains_at + (last - symoffset) * 4)
        if word is None:
            return None
        if word[0] & 1:
            return last + 1
        last += 1
    return None


def _dynsym_section_count(elf: _Elf) -> int | None:
    """Entries in the SHT_DYNSYM section, or None when the file has no usable
    section header table (a section-stripped binary)."""
    shoff = elf.unpack("Q" if elf.is64 else "I", 0x28 if elf.is64 else 0x20)
    counts = elf.unpack("HH", 0x3A if elf.is64 else 0x2E)
    if shoff is None or counts is None or not shoff[0]:
        return None
    shentsize, shnum = counts
    if shentsize != (64 if elf.is64 else 40) or shnum > _MAX_SECTIONS \
            or not elf.fits(shoff[0], shentsize * shnum):
        return None
    # Elf64_Shdr / Elf32_Shdr: sh_type, sh_size and sh_entsize are fields 1, 5, 9.
    fmt = "IIQQQQIIQQ" if elf.is64 else "IIIIIIIIII"
    for index in range(shnum):
        header = elf.unpack(fmt, shoff[0] + index * shentsize)
        if header is None:
            return None
        sh_type, sh_size, sh_entsize = header[1], header[5], header[9]
        if sh_type == _SHT_DYNSYM:
            return sh_size // sh_entsize if sh_entsize == (24 if elf.is64 else 16) else None
    return None


def _verneed_files(elf: _Elf, tags: dict[int, int], dynstr) -> dict[int, str]:
    """Version index -> the library file its verneed entry names."""
    at = elf.file_offset(tags[_DT_VERNEED]) if _DT_VERNEED in tags else None
    if at is None:
        return {}
    remaining = min(tags.get(_DT_VERNEEDNUM, _MAX_VERNEED), _MAX_VERNEED)
    files: dict[int, str] = {}
    while remaining > 0:
        need = elf.unpack("HHIII", at)  # version, cnt, file, aux, next
        if need is None:
            break
        _version, cnt, file_index, aux, next_need = need
        library = dynstr(file_index)
        aux_at = at + aux
        for _ in range(min(cnt, _MAX_VERNEED)):
            entry = elf.unpack("IHHII", aux_at)  # hash, flags, other, name, next
            if entry is None:
                break
            if library:
                files[entry[2] & 0x7FFF] = library
            if not entry[4]:
                break
            aux_at += entry[4]
        remaining -= 1
        if not next_need:
            break
        at += next_need
    return files


def _dynamic_symbols(elf: _Elf, tags: dict[int, int], purpose: str) -> tuple[bytes, int, int] | None:
    """The raw dynamic symbol table as ``(bytes, count, entry size)``, or None
    when it cannot be read. Every bound is checked here, once, for both readers
    below; *purpose* names what a refusal skipped."""
    if _DT_SYMTAB not in tags:
        return None
    # DT_SYMENT is fixed by the ABI (sizeof(ElfN_Sym)); the loader ignores it,
    # so a file can carry any value and still run. Any other value is refused,
    # never used as a stride or a read size.
    expected = 24 if elf.is64 else 16
    syment = tags.get(_DT_SYMENT, expected)
    if syment != expected:
        elf.notes.append(f"DT_SYMENT {syment} is not the ABI symbol size {expected}; "
                         f"{purpose} was skipped")
        return None
    count = _symbol_count(elf, tags)
    symtab = elf.file_offset(tags[_DT_SYMTAB])
    if not count or symtab is None:
        return None
    if count > _MAX_SYMBOLS or not elf.fits(symtab, count * syment):
        elf.notes.append(f"the dynamic symbol table ({count} symbols) does not fit in "
                         f"the file; {purpose} was skipped")
        return None
    symbols = elf.bytes_at(symtab, count * syment)
    return (symbols, count, syment) if symbols is not None else None


def _versioned_import_libraries(elf: _Elf, tags: dict[int, int], dynstr) -> dict[str, str]:
    if _DT_VERSYM not in tags or _DT_SYMTAB not in tags:
        return {}
    files = _verneed_files(elf, tags, dynstr)
    if not files:
        return {}
    table = _dynamic_symbols(elf, tags, "per-import library attribution")
    versym = elf.file_offset(tags[_DT_VERSYM])
    if table is None or versym is None:
        return {}
    symbols, count, syment = table
    if not elf.fits(versym, count * 2):
        elf.notes.append(f"the dynamic symbol table ({count} symbols) does not fit in "
                         "the file; per-import library attribution was skipped")
        return {}
    versions = elf.bytes_at(versym, count * 2)
    if versions is None:
        return {}
    version_of = struct.unpack(f"{elf.endian}{count}H", versions)
    # st_name is the first field of both Elf32_Sym and Elf64_Sym. Definedness
    # is not consulted: a version index that names a verneed entry IS the
    # requirement on that library, and a copy-relocated data import (a vtable,
    # `__environ` in a non-PIE executable) is "defined" in this file's .bss
    # while still bound to the library. Exports carry verdef indices instead.
    name_fmt = elf.endian + "I"
    attributed: dict[str, str] = {}
    conflicted: set[str] = set()
    for index in range(1, count):
        library = files.get(version_of[index] & 0x7FFF)
        if library is None:
            continue
        (name_index,) = struct.unpack_from(name_fmt, symbols, index * syment)
        name = dynstr(name_index)
        if not name or name in conflicted:
            continue
        if attributed.setdefault(name, library) != library:
            del attributed[name]
            conflicted.add(name)
    return attributed


_STB_GLOBAL, _STB_WEAK, _STB_GNU_UNIQUE = 1, 2, 10
_STT_SECTION, _STT_FILE = 3, 4
_STV_DEFAULT, _STV_PROTECTED = 0, 3
_SHN_UNDEF = 0


def _defined_exports(elf: _Elf, tags: dict[int, int], dynstr) -> frozenset[str] | None:
    """Names the dynamic symbol table defines for other modules to bind to:
    a defined (``st_shndx != SHN_UNDEF``) symbol with global, weak or unique
    binding and default or protected visibility -- the symbols the dynamic
    linker can resolve another module's undefined reference against. None when
    the table could not be read."""
    table = _dynamic_symbols(elf, tags, "reading the exported symbols")
    if table is None:
        return None
    symbols, count, syment = table
    # Elf64_Sym: name I, info B, other B, shndx H, value Q, size Q.
    # Elf32_Sym: name I, value I, size I, info B, other B, shndx H.
    fmt = elf.endian + ("IBBH" if elf.is64 else "IIIBBH")
    names: set[str] = set()
    for index in range(1, count):
        fields = struct.unpack_from(fmt, symbols, index * syment)
        if elf.is64:
            name_index, info, other, shndx = fields
        else:
            name_index, _value, _size, info, other, shndx = fields
        binding, kind = info >> 4, info & 0xF
        if shndx == _SHN_UNDEF or binding not in (_STB_GLOBAL, _STB_WEAK, _STB_GNU_UNIQUE):
            continue
        if kind in (_STT_SECTION, _STT_FILE) or (other & 3) not in (_STV_DEFAULT, _STV_PROTECTED):
            continue
        name = dynstr(name_index)
        if name:
            names.add(name)
    return frozenset(names)


# ---------------------------------------------------------------------------
# #1037 item 6: attributing UNVERSIONED imports from the needed libraries'
# own export tables. A symbol version names its library (above); an import
# with no version names nothing, and on a C++ binary linked against its own
# libraries that was every symbol from them. The loader's answer is the first
# library in DT_NEEDED order that defines the name, so this reads the files
# bn can find -- never a guess from the name -- and refuses to attribute past
# a needed library it could NOT read, since that one might define the name.


@dataclass
class NeededExports:
    # Each DT_NEEDED entry, in load order, with the file read for it (None when
    # no readable ELF of the binary's machine was found in the search dirs).
    files: list[tuple[str, str | None]] = field(default_factory=list)
    # Import name -> the needed library the loader would bind it to.
    libraries: dict[str, str] = field(default_factory=dict)


def read_elf_file(path: str, *, with_exports: bool = True) -> ElfDynamicInfo | None:
    """`parse_elf_dynamic` over a file on disk; None when it cannot be opened."""
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size

            def read(offset: int, length: int) -> bytes:
                handle.seek(offset)
                return handle.read(length)

            return parse_elf_dynamic(read, size, with_exports=with_exports)
    except OSError:
        return None


def library_search_dirs(*, binary_path: str | None, runpath: str | None,
                        rpath: str | None, sysroot: str | None) -> list[str]:
    """The directories searched for a needed library, in order: the binary's
    DT_RUNPATH (else DT_RPATH) entries, the binary's own directory, then the
    sysroot's library directories.

    `$ORIGIN` is the binary's directory. An absolute entry names a path on the
    TARGET's filesystem, so it is read under *sysroot* and skipped without one:
    the same path on this machine is a host library, not the target's. A
    relative entry depends on the loader's working directory and is skipped."""
    origin = os.path.dirname(os.path.abspath(binary_path)) if binary_path else None
    dirs: list[str] = []
    for entry in (runpath if runpath is not None else rpath or "").split(":"):
        if not entry:
            continue
        if "$ORIGIN" in entry or "${ORIGIN}" in entry:
            if origin is None:
                continue
            dirs.append(entry.replace("${ORIGIN}", origin).replace("$ORIGIN", origin))
        elif entry.startswith("/") and sysroot:
            dirs.append(os.path.join(sysroot, entry.lstrip("/")))
    if origin is not None:
        dirs.append(origin)
    if sysroot:
        for base in ("lib", "lib64", "usr/lib", "usr/lib64"):
            root = os.path.join(sysroot, base)
            dirs.append(root)
            try:
                children = sorted(os.listdir(root))
            except OSError:
                continue
            dirs.extend(os.path.join(root, child) for child in children
                        if "-linux-" in child and os.path.isdir(os.path.join(root, child)))
    seen: set[str] = set()
    return [d for d in (os.path.normpath(d) for d in dirs) if not (d in seen or seen.add(d))]


def needed_export_libraries(needed: list[str], search_dirs: list[str], *,
                            machine: int | None, sysroot: str | None = None,
                            reader: Callable[[str], ElfDynamicInfo | None] = read_elf_file,
                            ) -> NeededExports:
    """Attribute import names to the needed library that defines them.

    Each DT_NEEDED name is looked up in *search_dirs*; the first file there that
    is an ELF of *machine* with a readable export table is the one read (the
    loader also skips a file of the wrong machine). A name goes to the FIRST
    library in DT_NEEDED order that exports it, and only while every library
    before it was read: past an unread one, a name stays unattributed."""
    result = NeededExports()
    blocked = False
    for library in needed:
        path, info = _find_library(library, search_dirs, machine=machine,
                                   sysroot=sysroot, reader=reader)
        if info is None or info.exports is None:
            result.files.append((library, None))
            blocked = True
            continue
        result.files.append((library, path))
        if blocked:
            continue
        for name in info.exports:
            result.libraries.setdefault(name, library)
    return result


def _find_library(library: str, search_dirs: list[str], *, machine: int | None,
                  sysroot: str | None, reader) -> tuple[str | None, ElfDynamicInfo | None]:
    if "/" in library:
        # The loader uses a DT_NEEDED with a slash as a path. Only an absolute
        # one under a sysroot names a file this machine can read for the target.
        candidates = [os.path.join(sysroot, library.lstrip("/"))] \
            if sysroot and library.startswith("/") else []
    else:
        candidates = [os.path.join(d, library) for d in search_dirs]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        info = reader(candidate)
        if info is None or (machine is not None and info.machine != machine):
            continue
        return candidate, info
    return None, None
