"""`elf_dynamic.parse_elf_dynamic` against synthetic ELF images (#969).

The images are built here byte by byte -- a PT_LOAD over the whole file, a
PT_DYNAMIC, `.dynstr`, `.dynsym`, `.gnu.version`, `.gnu.version_r` and a symbol
count source -- so each test controls exactly which piece of the loader's
answer is present, in both classes and both byte orders.
"""
from __future__ import annotations

import struct

import pytest

from bn_agent_bridge.elf_dynamic import parse_elf_dynamic

_BASE = 0x400000


def _build_elf(*, is64=True, endian="<", needed=(), soname=None, rpath=None, runpath=None,
               symbols=(), count_source="hash"):
    """An ELF image whose dynamic symbols are ``symbols``: ``(name, library)``
    pairs, ``library`` None for an unversioned symbol. ``count_source`` is
    ``"hash"`` (DT_HASH), ``"section"`` (a SHT_DYNSYM header) or ``"none"``."""
    e = endian
    strtab = bytearray(b"\x00")

    def intern(text):
        offset = len(strtab)
        strtab.extend(text.encode() + b"\x00")
        return offset

    needed_offsets = [intern(lib) for lib in needed]
    soname_off = intern(soname) if soname else None
    rpath_off = intern(rpath) if rpath else None
    runpath_off = intern(runpath) if runpath else None
    # One verneed entry per library, one vernaux (version) per library.
    libraries = sorted({lib for _, lib in symbols if lib})
    version_index = {lib: 2 + i for i, lib in enumerate(libraries)}
    lib_file = {lib: intern(lib) for lib in libraries}
    lib_ver = {lib: intern(f"V_{i}") for i, lib in enumerate(libraries)}
    sym_names = [intern(name) for name, _ in symbols]

    syment = 24 if is64 else 16
    dynsym = bytearray(syment)  # index 0: the null symbol
    for name_off in sym_names:
        if is64:
            dynsym += struct.pack(e + "IBBHQQ", name_off, 0x12, 0, 0, 0, 0)
        else:
            dynsym += struct.pack(e + "IIIBBH", name_off, 0, 0, 0x12, 0, 0)
    versym = struct.pack(e + "H", 0) + b"".join(
        struct.pack(e + "H", version_index[lib] if lib else 1) for _, lib in symbols)
    verneed = bytearray()
    for i, lib in enumerate(libraries):
        next_need = 32 if i + 1 < len(libraries) else 0
        verneed += struct.pack(e + "HHIII", 1, 1, lib_file[lib], 16, next_need)
        verneed += struct.pack(e + "IHHII", 0, 0, version_index[lib], lib_ver[lib], 0)
    nsyms = len(symbols) + 1
    hashtab = struct.pack(e + "II", 1, nsyms) + struct.pack(e + "I", 0) + bytes(4 * nsyms)

    ehsize, phentsize = (64, 56) if is64 else (52, 32)
    cursor = ehsize + 2 * phentsize
    blobs = {}
    for key, blob in (("strtab", bytes(strtab)), ("dynsym", bytes(dynsym)),
                      ("versym", versym), ("verneed", bytes(verneed)), ("hash", hashtab)):
        cursor = (cursor + 7) & ~7
        blobs[key] = (cursor, blob)
        cursor += len(blob)

    def va(key):
        return _BASE + blobs[key][0]

    tags = [(1, off) for off in needed_offsets]
    for tag, off in ((14, soname_off), (15, rpath_off), (29, runpath_off)):
        if off is not None:
            tags.append((tag, off))
    tags += [(5, va("strtab")), (10, len(strtab)), (6, va("dynsym")), (11, syment),
             (0x6FFFFFF0, va("versym"))]
    if libraries:
        tags += [(0x6FFFFFFE, va("verneed")), (0x6FFFFFFF, len(libraries))]
    if count_source == "hash":
        tags.append((4, va("hash")))
    tags.append((0, 0))
    dyn = b"".join(struct.pack(e + ("qQ" if is64 else "iI"), t, v) for t, v in tags)
    cursor = (cursor + 7) & ~7
    blobs["dynamic"] = (cursor, dyn)
    cursor += len(dyn)

    shoff = shnum = 0
    shentsize = 64 if is64 else 40
    if count_source == "section":
        cursor = (cursor + 7) & ~7
        shoff, shnum = cursor, 2
        dynsym_off, dynsym_blob = blobs["dynsym"]
        null = bytes(shentsize)
        if is64:
            sec = struct.pack(e + "IIQQQQIIQQ", 0, 11, 2, va("dynsym"), dynsym_off,
                              len(dynsym_blob), 0, 1, 8, syment)
        else:
            sec = struct.pack(e + "IIIIIIIIII", 0, 11, 2, va("dynsym"), dynsym_off,
                              len(dynsym_blob), 0, 1, 4, syment)
        blobs["shdrs"] = (shoff, null + sec)
        cursor += 2 * shentsize

    total = cursor
    image = bytearray(total)
    ident = b"\x7fELF" + bytes([2 if is64 else 1, 1 if e == "<" else 2, 1]) + bytes(9)
    if is64:
        header = struct.pack(e + "HHIQQQIHHHHHH", 3, 183, 1, 0, ehsize, shoff, 0,
                             ehsize, phentsize, 2, shentsize, shnum, 0)
        load = struct.pack(e + "IIQQQQQQ", 1, 5, 0, _BASE, _BASE, total, total, 0x1000)
        dyn_ph = struct.pack(e + "IIQQQQQQ", 2, 6, blobs["dynamic"][0],
                             _BASE + blobs["dynamic"][0], 0, len(dyn), len(dyn), 8)
    else:
        header = struct.pack(e + "HHIIIIIHHHHHH", 3, 8, 1, 0, ehsize, shoff, 0,
                             ehsize, phentsize, 2, shentsize, shnum, 0)
        load = struct.pack(e + "IIIIIIII", 1, 0, _BASE, _BASE, total, total, 5, 0x1000)
        dyn_ph = struct.pack(e + "IIIIIIII", 2, blobs["dynamic"][0],
                             _BASE + blobs["dynamic"][0], 0, len(dyn), len(dyn), 6, 4)
    image[0:ehsize] = ident + header
    image[ehsize:ehsize + 2 * phentsize] = load + dyn_ph
    for offset, blob in blobs.values():
        image[offset:offset + len(blob)] = blob
    return bytes(image)


def _reader(image):
    return lambda offset, size: image[offset:offset + size]


_SYMBOLS = (
    ("printf", "libc.so.6"),
    ("_ZNSo3putEc", "libstdc++.so.6"),
    ("__gmon_start__", None),          # unversioned weak: no library
    ("_Unwind_Resume", "libgcc_s.so.1"),
)


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("count_source", ["hash", "section"])
def test_parse_reads_needed_soname_paths_and_versioned_libraries(is64, endian, count_source):
    image = _build_elf(is64=is64, endian=endian,
                       needed=("libstdc++.so.6", "libc.so.6", "libgcc_s.so.1"),
                       soname="libwidget.so.2", rpath="/opt/widget/lib", runpath="$ORIGIN/../lib",
                       symbols=_SYMBOLS, count_source=count_source)
    info = parse_elf_dynamic(_reader(image))
    assert info is not None
    assert info.needed == ["libstdc++.so.6", "libc.so.6", "libgcc_s.so.1"]
    assert (info.soname, info.rpath, info.runpath) == (
        "libwidget.so.2", "/opt/widget/lib", "$ORIGIN/../lib")
    assert info.import_libraries == {
        "printf": "libc.so.6",
        "_ZNSo3putEc": "libstdc++.so.6",
        "_Unwind_Resume": "libgcc_s.so.1",
    }


def test_absent_dynamic_strings_are_none():
    info = parse_elf_dynamic(_reader(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS)))
    assert info is not None
    assert (info.soname, info.rpath, info.runpath) == (None, None, None)


def test_no_symbol_count_source_attributes_nothing():
    # Section-stripped with no DT_HASH (and no GNU hash): the symbol count is
    # unknown, so no row is attributed rather than reading past the table.
    info = parse_elf_dynamic(_reader(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS,
                                                count_source="none")))
    assert info is not None
    assert info.needed == ["libc.so.6"]
    assert info.import_libraries == {}


def test_unversioned_target_attributes_nothing():
    # A musl/uClibc-style target: DT_NEEDED but no verneed at all.
    image = _build_elf(needed=("libc.so",), symbols=(("printf", None), ("strlen", None)))
    info = parse_elf_dynamic(_reader(image))
    assert info is not None
    assert info.needed == ["libc.so"]
    assert info.import_libraries == {}


@pytest.mark.parametrize("image", [
    b"",
    b"MZ\x90\x00" + bytes(60),                      # not an ELF
    b"\x7fELF\x03\x01\x01" + bytes(57),             # bad EI_CLASS
])
def test_non_elf_input_is_none(image):
    assert parse_elf_dynamic(_reader(image)) is None


def test_truncated_image_never_raises():
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS)
    for cut in range(0, len(image), 7):
        info = parse_elf_dynamic(_reader(image[:cut]))
        assert info is None or isinstance(info.import_libraries, dict)


def test_reader_exception_is_none():
    def boom(offset, size):
        raise OSError("read failed")
    assert parse_elf_dynamic(boom) is None
