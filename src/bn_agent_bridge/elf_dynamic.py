"""ELF dynamic-section facts BN does not expose: SONAME, RPATH/RUNPATH, the
library each versioned import binds to (#969) and the file's own exports.

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
from typing import Literal

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
_DT_VERDEF = 0x6FFFFFFC
_DT_VERDEFNUM = 0x6FFFFFFD
_DT_VERNEED = 0x6FFFFFFE
_DT_VERNEEDNUM = 0x6FFFFFFF

_SHT_DYNSYM = 11

# Sanity bounds on counts read from the file, independent of its size.
_MAX_PROGRAM_HEADERS = 4096
_MAX_SECTIONS = 1 << 16
_MAX_DYNAMIC_ENTRIES = 4096
_MAX_SYMBOLS = 1 << 20
_MAX_VERNEED = 4096
_MAX_VERDEF = 4096
_MAX_STRING = 4096
# The largest single read the parser ever issues. The biggest legitimate one is
# the dynamic symbol table: _MAX_SYMBOLS * 24 bytes.
_MAX_READ = _MAX_SYMBOLS * 24

Reader = Callable[[int, int], bytes]


@dataclass(frozen=True)
class ExportedSymbol:
    """A dynamic symbol this file defines for other objects to bind to."""
    name: str
    # The verdef version the symbol is published under (`GLIBC_2.17`), or None
    # when the file has no verdef table or the symbol is unversioned.
    version: str | None
    kind: Literal["func", "object", "other"]
    weak: bool


@dataclass
class ElfDynamicInfo:
    # ELF header facts a linker needs to decide whether two objects can be
    # loaded together: e_machine, EI_CLASS as 32/64 and e_type (ET_DYN = 3).
    e_machine: int = 0
    elf_class: int = 0
    e_type: int = 0
    needed: list[str] = field(default_factory=list)
    soname: str | None = None
    rpath: str | None = None
    runpath: str | None = None
    # Dynamic symbol name -> the verneed library its version binds it to. Only
    # names with ONE library: a name versioned against two is not attributed.
    import_libraries: dict[str, str] = field(default_factory=dict)
    # Defined dynamic symbols (st_shndx != SHN_UNDEF, GLOBAL or WEAK binding,
    # DEFAULT or PROTECTED visibility) keyed by bare name. A name defined under
    # several versions keeps its default (non-hidden) one.
    exports: dict[str, ExportedSymbol] = field(default_factory=dict)
    # Why part of the dynamic section was not parsed: a value out of range for
    # this file, so that part is skipped rather than guessed. Empty when the
    # whole answer was read.
    notes: list[str] = field(default_factory=list)
    # Why the per-symbol version table was NOT read, or None when it was (or
    # the file has none: DT_VERSYM/DT_VERNEED absent, so no import carries a
    # version requirement). Only None lets a reader call a symbol missing from
    # `import_libraries` "unversioned" (#1037 review round 3: a refused
    # DT_SYMENT left the map empty while every import WAS versioned).
    versions_unread: str | None = "the dynamic section was not parsed"


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


def parse_elf_dynamic(read: Reader, file_size: int) -> ElfDynamicInfo | None:
    """The dynamic-section facts of the ELF file *read* serves (*file_size*
    bytes long), or ``None`` when it is not an ELF, carries no PT_DYNAMIC (a
    static binary), or its headers are out of range for the file."""
    if file_size <= 0:
        return None
    try:
        return _parse(read, int(file_size))
    except (struct.error, ValueError, OverflowError):
        return None


def parse_elf_file(path: str) -> ElfDynamicInfo | None:
    """`parse_elf_dynamic` over the file at *path*, read piecewise with
    seek+read so no window larger than `_MAX_READ` is ever in memory; None for
    an unreadable path as for a non-ELF file."""
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size

            def read(offset: int, length: int) -> bytes:
                handle.seek(offset)
                return handle.read(length)

            return parse_elf_dynamic(read, size)
    except OSError:
        return None


def _parse(read: Reader, file_size: int) -> ElfDynamicInfo | None:
    probe = _Elf(read, file_size, False, "<")
    ident = probe.bytes_at(0, 16)
    if ident is None or ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2):
        return None
    elf = _Elf(read, file_size, ident[4] == 2, "<" if ident[5] == 1 else ">")
    # e_type, e_machine; e_phoff, then (past e_shoff/e_flags/e_ehsize)
    # e_phentsize and e_phnum.
    kinds = elf.unpack("HH", 0x10)
    phoff = elf.unpack("Q" if elf.is64 else "I", 0x20 if elf.is64 else 0x1C)
    counts = elf.unpack("HH", 0x36 if elf.is64 else 0x2A)
    if kinds is None or phoff is None or counts is None:
        return None
    phoff, (phentsize, phnum) = phoff[0], counts
    info = ElfDynamicInfo(e_machine=kinds[1], elf_class=64 if elf.is64 else 32,
                          e_type=kinds[0], notes=elf.notes)
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
        return info
    tags: dict[int, int] = {}
    needed_offsets: list[int] = []
    for tag, value in entries:
        if tag == _DT_NEEDED:
            needed_offsets.append(value)
        else:
            tags.setdefault(tag, value)
    strtab = elf.file_offset(tags[_DT_STRTAB]) if _DT_STRTAB in tags else None
    if strtab is None:
        info.versions_unread = "the dynamic string table is not in the file"
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
    table = _load_symbol_table(elf, tags)
    info.import_libraries, info.versions_unread = _versioned_import_libraries(
        elf, tags, dynstr, table)
    info.exports = _exports(elf, tags, dynstr, table)
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


@dataclass
class _SymbolTable:
    """The dynamic symbol table read into memory: ``count`` entries of
    ``syment`` bytes in ``data``, and each entry's ``.gnu.version`` index when
    the file has a DT_VERSYM table."""
    count: int
    syment: int
    data: bytes
    versions: tuple[int, ...] | None


def _load_symbol_table(elf: _Elf, tags: dict[int, int]) -> _SymbolTable | str:
    """The dynamic symbol table (with its version indices when DT_VERSYM is
    declared), or the reason it was not read. Both the import attribution and
    the export list are built from this one read."""
    if _DT_SYMTAB not in tags:
        return "the file has no dynamic symbol table"
    # DT_SYMENT is fixed by the ABI (sizeof(ElfN_Sym)); the loader ignores it,
    # so a file can carry any value and still run. Any other value is refused,
    # never used as a stride or a read size.
    expected = 24 if elf.is64 else 16
    syment = tags.get(_DT_SYMENT, expected)
    if syment != expected:
        elf.notes.append(f"DT_SYMENT {syment} is not the ABI symbol size {expected}; "
                         "per-import library attribution and exports were skipped")
        return f"DT_SYMENT {syment} is not the ABI symbol size {expected}"
    count = _symbol_count(elf, tags)
    symtab = elf.file_offset(tags[_DT_SYMTAB])
    versym = elf.file_offset(tags[_DT_VERSYM]) if _DT_VERSYM in tags else None
    if not count or symtab is None or (_DT_VERSYM in tags and versym is None):
        return "the dynamic symbol table, its size or its version table could not be located"
    if count > _MAX_SYMBOLS or not elf.fits(symtab, count * syment) \
            or (versym is not None and not elf.fits(versym, count * 2)):
        elf.notes.append(f"the dynamic symbol table ({count} symbols) does not fit in "
                         "the file; per-import library attribution and exports were skipped")
        return "the dynamic symbol table does not fit in the file"
    symbols = elf.bytes_at(symtab, count * syment)
    if symbols is None:
        return "the dynamic symbol table could not be read"
    versions = None
    if versym is not None:
        data = elf.bytes_at(versym, count * 2)
        if data is None:
            return "the dynamic symbol table could not be read"
        versions = struct.unpack(f"{elf.endian}{count}H", data)
    return _SymbolTable(count, syment, symbols, versions)


def _versioned_import_libraries(elf: _Elf, tags: dict[int, int], dynstr,
                                table: _SymbolTable | str) -> tuple[dict[str, str], str | None]:
    """``(name -> library, why the version table was not read)``; the reason
    is None when it was read, or when the file declares no version table."""
    if _DT_VERSYM not in tags and _DT_VERNEED not in tags:
        return {}, None
    if _DT_VERSYM not in tags:
        return {}, "the file declares version requirements without a symbol version table"
    if _DT_SYMTAB not in tags:
        # #1037 review round 5: name the table that is actually missing.
        return {}, "the file declares symbol versions without a dynamic symbol table"
    files = _verneed_files(elf, tags, dynstr)
    if not files:
        if _DT_VERNEED in tags:
            return {}, "the version-requirement table (DT_VERNEED) could not be read"
        return {}, None
    if isinstance(table, str):
        return {}, table
    if table.versions is None:  # unreachable: DT_VERSYM is in tags, so the loader read it
        return {}, "the symbol version table could not be read"
    # st_name is the first field of both Elf32_Sym and Elf64_Sym. Definedness
    # is not consulted: a version index that names a verneed entry IS the
    # requirement on that library, and a copy-relocated data import (a vtable,
    # `__environ` in a non-PIE executable) is "defined" in this file's .bss
    # while still bound to the library. Exports carry verdef indices instead.
    name_fmt = elf.endian + "I"
    attributed: dict[str, str] = {}
    conflicted: set[str] = set()
    for index in range(1, table.count):
        library = files.get(table.versions[index] & 0x7FFF)
        if library is None:
            continue
        (name_index,) = struct.unpack_from(name_fmt, table.data, index * table.syment)
        name = dynstr(name_index)
        if not name or name in conflicted:
            continue
        if attributed.setdefault(name, library) != library:
            del attributed[name]
            conflicted.add(name)
    return attributed, None


def _verdef_names(elf: _Elf, tags: dict[int, int], dynstr) -> dict[int, str]:
    """Version index -> the name its verdef entry defines (the first verdaux;
    later ones name parent versions). Index 1, the base version that just
    restates the SONAME, is left out so such symbols read as unversioned."""
    at = elf.file_offset(tags[_DT_VERDEF]) if _DT_VERDEF in tags else None
    if at is None:
        return {}
    remaining = min(tags.get(_DT_VERDEFNUM, _MAX_VERDEF), _MAX_VERDEF)
    names: dict[int, str] = {}
    while remaining > 0:
        entry = elf.unpack("HHHHIII", at)  # version, flags, ndx, cnt, hash, aux, next
        if entry is None:
            break
        _version, _flags, ndx, cnt, _hash, aux, next_def = entry
        if cnt and (ndx & 0x7FFF) > 1:
            daux = elf.unpack("II", at + aux)  # name, next
            name = dynstr(daux[0]) if daux is not None else None
            if name:
                names[ndx & 0x7FFF] = name
        remaining -= 1
        if not next_def:
            break
        at += next_def
    return names


def _exports(elf: _Elf, tags: dict[int, int], dynstr,
             table: _SymbolTable | str) -> dict[str, ExportedSymbol]:
    """The symbols this file defines for others: a dynsym entry with a section
    (st_shndx != SHN_UNDEF), GLOBAL or WEAK binding and DEFAULT or PROTECTED
    visibility. HIDDEN and INTERNAL symbols are in the table but not bindable."""
    if isinstance(table, str):
        return {}
    names = _verdef_names(elf, tags, dynstr) if table.versions is not None else {}
    # Elf64_Sym: st_name, st_info, st_other, st_shndx; Elf32_Sym: st_name,
    # st_value, st_size, st_info, st_other, st_shndx.
    fmt = elf.endian + ("IBBH" if elf.is64 else "IIIBBH")
    exports: dict[str, ExportedSymbol] = {}
    hidden_version: set[str] = set()
    for index in range(1, table.count):
        fields = struct.unpack_from(fmt, table.data, index * table.syment)
        name_index, st_info, st_other, st_shndx = fields[0], fields[-3], fields[-2], fields[-1]
        if st_shndx == 0 or (st_info >> 4) not in (1, 2) or (st_other & 3) not in (0, 3):
            continue
        name = dynstr(name_index)
        if not name:
            continue
        version = None
        hidden = False
        if table.versions is not None:
            versym = table.versions[index]
            hidden = bool(versym & 0x8000)
            version = names.get(versym & 0x7FFF)
        # `memcpy@GLIBC_2.2.5` and `memcpy@@GLIBC_2.14` share a bare name: the
        # default (non-hidden) definition is the one an unversioned reference
        # binds to, so it wins over a hidden compatibility one.
        if name in exports and (hidden or name not in hidden_version):
            continue
        if hidden:
            hidden_version.add(name)
        else:
            hidden_version.discard(name)
        kind = {2: "func", 1: "object", 5: "object", 6: "object"}.get(st_info & 0xF, "other")
        exports[name] = ExportedSymbol(name, version, kind, (st_info >> 4) == 2)
    return exports
