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
BinaryNinja import -- and every read is bounds-checked: a malformed or truncated
file yields ``None`` / empty fields, never an exception.
"""
from __future__ import annotations

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

# Sanity bounds: a corrupt count must not turn into a multi-gigabyte read.
_MAX_DYNAMIC_ENTRIES = 4096
_MAX_SYMBOLS = 1 << 20
_MAX_VERNEED = 4096
_MAX_STRING = 4096

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


class _Elf:
    def __init__(self, read: Reader, is64: bool, endian: str):
        self._read = read
        self.is64 = is64
        self.endian = endian
        self.loads: list[tuple[int, int, int]] = []  # (vaddr, filesz, offset)

    def bytes_at(self, offset: int, size: int) -> bytes | None:
        if offset < 0 or size < 0:
            return None
        try:
            data = bytes(self._read(offset, size) or b"")
        except Exception:
            return None
        return data if len(data) == size else None

    def unpack(self, fmt: str, offset: int) -> tuple | None:
        fmt = self.endian + fmt
        data = self.bytes_at(offset, struct.calcsize(fmt))
        return struct.unpack(fmt, data) if data is not None else None

    def file_offset(self, vaddr: int) -> int | None:
        for start, size, offset in self.loads:
            if start <= vaddr < start + size:
                return offset + (vaddr - start)
        return None

    def cstring(self, offset: int, limit: int) -> str | None:
        data = self.bytes_at(offset, max(0, min(limit, _MAX_STRING)))
        if data is None:
            # Near the end of the file: shrink to whatever is readable.
            for size in (256, 64, 16):
                data = self.bytes_at(offset, min(size, limit))
                if data is not None:
                    break
        if not data:
            return None
        end = data.find(b"\x00")
        if end < 0:
            return None
        return data[:end].decode("utf-8", errors="replace")


def parse_elf_dynamic(read: Reader) -> ElfDynamicInfo | None:
    """The dynamic-section facts of the ELF file *read* serves, or ``None`` when
    it is not an ELF or carries no PT_DYNAMIC (a static binary)."""
    try:
        return _parse(read)
    except (struct.error, ValueError, OverflowError):
        return None


def _parse(read: Reader) -> ElfDynamicInfo | None:
    probe = _Elf(read, False, "<")
    ident = probe.bytes_at(0, 16)
    if ident is None or ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2):
        return None
    elf = _Elf(read, ident[4] == 2, "<" if ident[5] == 1 else ">")
    # e_phoff, then (past e_shoff/e_flags/e_ehsize) e_phentsize and e_phnum.
    phoff = elf.unpack("Q" if elf.is64 else "I", 0x20 if elf.is64 else 0x1C)
    counts = elf.unpack("HH", 0x36 if elf.is64 else 0x2A)
    if phoff is None or counts is None:
        return None
    phoff, (phentsize, phnum) = phoff[0], counts
    if not phoff or not phnum:
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
    tags: dict[int, int] = {}
    needed_offsets: list[int] = []
    for tag, value in entries:
        if tag == _DT_NEEDED:
            needed_offsets.append(value)
        else:
            tags.setdefault(tag, value)
    info = ElfDynamicInfo()
    strtab = elf.file_offset(tags[_DT_STRTAB]) if _DT_STRTAB in tags else None
    if strtab is None:
        return info
    strsz = tags.get(_DT_STRSZ, _MAX_STRING)

    def dynstr(index: int) -> str | None:
        if index < 0 or index >= strsz:
            return None
        return elf.cstring(strtab + index, strsz - index)

    info.needed = [name for name in (dynstr(i) for i in needed_offsets) if name]
    info.soname = dynstr(tags[_DT_SONAME]) if _DT_SONAME in tags else None
    info.rpath = dynstr(tags[_DT_RPATH]) if _DT_RPATH in tags else None
    info.runpath = dynstr(tags[_DT_RUNPATH]) if _DT_RUNPATH in tags else None
    info.import_libraries = _versioned_import_libraries(elf, tags, dynstr)
    return info


def _dynamic_entries(elf: _Elf, offset: int, size: int) -> list[tuple[int, int]]:
    entsize = 16 if elf.is64 else 8
    count = min(size // entsize, _MAX_DYNAMIC_ENTRIES)
    data = elf.bytes_at(offset, count * entsize)
    if data is None:
        return []
    fmt = elf.endian + ("qQ" if elf.is64 else "iI")
    entries = []
    for tag, value in struct.iter_unpack(fmt, data):
        if tag == _DT_NULL:
            break
        entries.append((tag, value))
    return entries


_SHT_DYNSYM = 11


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
        at = elf.file_offset(tags[_DT_GNU_HASH])
        header = elf.unpack("IIII", at) if at is not None else None
        if header is None:
            return None
        nbuckets, symoffset, bloom_size = header[0], header[1], header[2]
        buckets_at = at + 16 + bloom_size * (8 if elf.is64 else 4)
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
    """Entries in the SHT_DYNSYM section, or None when the file has no readable
    section header table (a section-stripped binary)."""
    shoff = elf.unpack("Q" if elf.is64 else "I", 0x28 if elf.is64 else 0x20)
    counts = elf.unpack("HH", 0x3A if elf.is64 else 0x2E)
    if shoff is None or counts is None or not shoff[0]:
        return None
    shentsize, shnum = counts
    # Elf64_Shdr / Elf32_Shdr: sh_type, sh_size and sh_entsize are fields 1, 5, 9.
    fmt = "IIQQQQIIQQ" if elf.is64 else "IIIIIIIIII"
    for index in range(min(shnum, _MAX_DYNAMIC_ENTRIES)):
        header = elf.unpack(fmt, shoff[0] + index * shentsize)
        if header is None:
            return None
        sh_type, sh_size, sh_entsize = header[1], header[5], header[9]
        if sh_type == _SHT_DYNSYM and sh_entsize:
            return sh_size // sh_entsize
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


def _versioned_import_libraries(elf: _Elf, tags: dict[int, int], dynstr) -> dict[str, str]:
    if _DT_VERSYM not in tags or _DT_SYMTAB not in tags:
        return {}
    files = _verneed_files(elf, tags, dynstr)
    if not files:
        return {}
    count = _symbol_count(elf, tags)
    symtab = elf.file_offset(tags[_DT_SYMTAB])
    versym = elf.file_offset(tags[_DT_VERSYM])
    if not count or symtab is None or versym is None:
        return {}
    count = min(count, _MAX_SYMBOLS)
    syment = tags.get(_DT_SYMENT) or (24 if elf.is64 else 16)
    symbols = elf.bytes_at(symtab, count * syment)
    versions = elf.bytes_at(versym, count * 2)
    if symbols is None or versions is None:
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
