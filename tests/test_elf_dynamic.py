"""`elf_dynamic.parse_elf_dynamic` against synthetic ELF images (#969).

The images are built here byte by byte -- a PT_LOAD over the whole file, a
PT_DYNAMIC, `.dynstr`, `.dynsym`, `.gnu.version`, `.gnu.version_r`,
`.gnu.version_d` and a symbol count source -- so each test controls exactly
which piece of the loader's answer is present, in both classes and both byte
orders.
"""
from __future__ import annotations

import multiprocessing
import os
import struct
import time
from dataclasses import dataclass

import pytest

from bn_agent_bridge.elf_dynamic import ExportedSymbol, parse_elf_dynamic, parse_elf_file

_BASE = 0x400000

_STB_GLOBAL, _STB_WEAK, _STB_LOCAL = 1, 2, 0
_STT_OBJECT, _STT_FUNC, _STT_TLS, _STT_SECTION = 1, 2, 6, 3
_STV_DEFAULT, _STV_INTERNAL, _STV_HIDDEN, _STV_PROTECTED = 0, 1, 2, 3


@dataclass(frozen=True)
class _Def:
    """A defined dynamic symbol for `_build_elf`: ``version`` names a verdef
    entry (None: the global version index 1), ``hidden`` sets the versym
    hidden bit (a non-default compatibility version)."""
    name: str
    version: str | None = None
    bind: int = _STB_GLOBAL
    vis: int = _STV_DEFAULT
    type: int = _STT_FUNC
    hidden: bool = False


def _build_elf(*, is64=True, endian="<", needed=(), soname=None, rpath=None, runpath=None,
               symbols=(), defined=(), count_source="hash"):
    """An ELF image whose dynamic symbols are the undefined ``symbols`` --
    ``(name, library)`` pairs, ``library`` None for an unversioned symbol --
    followed by the ``defined`` `_Def` rows. ``count_source`` is ``"hash"``
    (DT_HASH), ``"gnu_hash"`` (DT_GNU_HASH), ``"section"`` (a SHT_DYNSYM
    header) or ``"none"``."""
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
    # One verdef entry per distinct version a defined symbol names, after the
    # verneed indices, plus the base entry (index 1) when there are any.
    def_versions = sorted({d.version for d in defined if d.version})
    def_index = {v: 2 + len(libraries) + i for i, v in enumerate(def_versions)}
    def_name = {v: intern(v) for v in def_versions}
    base_name = intern(soname or "base") if def_versions else None
    def_names = [intern(d.name) for d in defined]

    syment = 24 if is64 else 16
    dynsym = bytearray(syment)  # index 0: the null symbol
    for name_off in sym_names:
        if is64:
            dynsym += struct.pack(e + "IBBHQQ", name_off, 0x12, 0, 0, 0, 0)
        else:
            dynsym += struct.pack(e + "IIIBBH", name_off, 0, 0, 0x12, 0, 0)
    for name_off, d in zip(def_names, defined):
        info = (d.bind << 4) | d.type
        if is64:
            dynsym += struct.pack(e + "IBBHQQ", name_off, info, d.vis, 10, 0x1000, 8)
        else:
            dynsym += struct.pack(e + "IIIBBH", name_off, 0x1000, 8, info, d.vis, 10)
    versym = struct.pack(e + "H", 0) + b"".join(
        struct.pack(e + "H", version_index[lib] if lib else 1) for _, lib in symbols)
    versym += b"".join(
        struct.pack(e + "H", (def_index[d.version] if d.version else 1) | (0x8000 if d.hidden else 0))
        for d in defined)
    verneed = bytearray()
    for i, lib in enumerate(libraries):
        next_need = 32 if i + 1 < len(libraries) else 0
        verneed += struct.pack(e + "HHIII", 1, 1, lib_file[lib], 16, next_need)
        verneed += struct.pack(e + "IHHII", 0, 0, version_index[lib], lib_ver[lib], 0)
    verdef = bytearray()
    if def_versions:
        rows = [(1, 1, base_name)] + [(def_index[v], 0, def_name[v]) for v in def_versions]
        for i, (ndx, flags, name_off) in enumerate(rows):
            next_def = 28 if i + 1 < len(rows) else 0
            verdef += struct.pack(e + "HHHHIII", 1, flags, ndx, 1, 0, 20, next_def)
            verdef += struct.pack(e + "II", name_off, 0)
    nsyms = len(symbols) + len(defined) + 1
    hashtab = struct.pack(e + "II", 1, nsyms) + struct.pack(e + "I", 0) + bytes(4 * nsyms)
    # One bucket pointing at symbol 1, one bloom word, chains over every symbol
    # with the end-of-chain bit on the last: the count is the chain's extent.
    gnu_hash = struct.pack(e + "IIII", 1, 1, 1, 0) + bytes(8 if is64 else 4) \
        + struct.pack(e + "I", 1) \
        + b"".join(struct.pack(e + "I", 1 if i == nsyms - 1 else 0) for i in range(1, nsyms))

    ehsize, phentsize = (64, 56) if is64 else (52, 32)
    cursor = ehsize + 2 * phentsize
    blobs = {}
    for key, blob in (("strtab", bytes(strtab)), ("dynsym", bytes(dynsym)),
                      ("versym", versym), ("verneed", bytes(verneed)),
                      ("verdef", bytes(verdef)), ("hash", hashtab), ("gnu_hash", gnu_hash)):
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
    if def_versions:
        tags += [(0x6FFFFFFC, va("verdef")), (0x6FFFFFFD, len(def_versions) + 1)]
    if count_source == "hash":
        tags.append((4, va("hash")))
    elif count_source == "gnu_hash":
        tags.append((0x6FFFFEF5, va("gnu_hash")))
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


def _parse(image, *, file_size=None):
    """`parse_elf_dynamic` over *image*, with a reader that FAILS the test if
    the parser ever asks for a window outside the file -- the request BN's
    `BinaryView.read` segfaults on (#987 review), so the invariant is checked
    on every read rather than inferred from the result."""
    size = len(image) if file_size is None else file_size

    def read(offset, length):
        assert 0 <= offset and 0 <= length and offset + length <= size, (
            f"parser read {length} bytes at {offset} in a {size}-byte file")
        return image[offset:offset + length]

    return parse_elf_dynamic(read, size)


def _dynamic_entries_at(image, *, is64=True, endian="<"):
    """``(file offset, tag, value)`` for each `_build_elf` dynamic entry, up to
    and including DT_NULL, plus the entry struct format."""
    ehsize, phentsize = (64, 56) if is64 else (52, 32)
    dyn_ph = ehsize + phentsize  # `_build_elf` writes PT_LOAD, then PT_DYNAMIC
    if is64:
        offset = struct.unpack_from(endian + "Q", image, dyn_ph + 8)[0]
        size = struct.unpack_from(endian + "Q", image, dyn_ph + 32)[0]
        fmt = endian + "qQ"
    else:
        offset = struct.unpack_from(endian + "I", image, dyn_ph + 4)[0]
        size = struct.unpack_from(endian + "I", image, dyn_ph + 16)[0]
        fmt = endian + "iI"
    entsize = struct.calcsize(fmt)
    rows = [(at, *struct.unpack_from(fmt, image, at))
            for at in range(offset, offset + size, entsize)]
    return rows, fmt


def _set_dynamic_tag(image, tag, value, *, is64=True, endian="<"):
    """*image* with the value of its first dynamic entry *tag* replaced (the
    entry is added over DT_NULL when absent)."""
    image = bytearray(image)
    rows, fmt = _dynamic_entries_at(image, is64=is64, endian=endian)
    for at, current, _ in rows:
        if current in (tag, 0):
            struct.pack_into(fmt, image, at, tag, value)
            if current == 0:
                image.extend(bytes(struct.calcsize(fmt)))  # keep a DT_NULL readable after it
            return bytes(image)
    raise AssertionError(f"no slot for tag {tag:#x}")


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
    info = _parse(image)
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
    info = _parse(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS))
    assert info is not None
    assert (info.soname, info.rpath, info.runpath) == (None, None, None)


def test_no_symbol_count_source_attributes_nothing():
    # Section-stripped with no DT_HASH (and no GNU hash): the symbol count is
    # unknown, so no row is attributed rather than reading past the table.
    info = _parse(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS, count_source="none"))
    assert info is not None
    assert info.needed == ["libc.so.6"]
    assert info.import_libraries == {}


def test_unversioned_target_attributes_nothing():
    # A musl/uClibc-style target: DT_NEEDED but no verneed at all.
    image = _build_elf(needed=("libc.so",), symbols=(("printf", None), ("strlen", None)))
    info = _parse(image)
    assert info is not None
    assert info.needed == ["libc.so"]
    assert info.import_libraries == {}


@pytest.mark.parametrize("image", [
    b"",
    b"MZ\x90\x00" + bytes(60),                      # not an ELF
    b"\x7fELF\x03\x01\x01" + bytes(57),             # bad EI_CLASS
])
def test_non_elf_input_is_none(image):
    assert _parse(image) is None


def test_truncated_image_never_raises_or_reads_past_the_end():
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS)
    for cut in range(0, len(image), 7):
        info = _parse(image[:cut])
        assert info is None or isinstance(info.import_libraries, dict)


def test_reader_exception_is_a_short_read():
    def boom(offset, size):
        raise OSError("read failed")
    assert parse_elf_dynamic(boom, 4096) is None


# --- hostile values (#987 review) -------------------------------------------
# Every size and count below comes from the file. The loader ignores most of
# them (DT_SYMENT included), so such a file still runs -- and before the
# review fix the first one made BN's read() segfault the bridge.


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("syment", [1 << 56, 0xFFFFFFFF, 0, 1, 25])
def test_non_abi_dt_syment_skips_attribution_with_a_note(is64, syment):
    if not is64 and syment >= 1 << 32:
        syment = 0xFFFFFFFF
    image = _set_dynamic_tag(
        _build_elf(is64=is64, needed=("libc.so.6",), soname="libw.so.1", symbols=_SYMBOLS),
        11, syment, is64=is64)
    info = _parse(image)
    assert info is not None
    assert info.import_libraries == {}
    # The rest of the dynamic section is still read.
    assert (info.needed, info.soname) == (["libc.so.6"], "libw.so.1")
    assert any("DT_SYMENT" in note for note in info.notes), info.notes
    # #1037 review round 3: the empty map is "not read", never "unversioned".
    assert info.versions_unread and "DT_SYMENT" in info.versions_unread


@pytest.mark.parametrize("symbols, expected", [
    (_SYMBOLS, None),                              # version table read
    ((("puts", None), ("free", None)), None),      # no version table at all
])
def test_versions_unread_is_none_only_when_the_versions_are_established(symbols, expected):
    info = _parse(_build_elf(needed=("libc.so.6",), symbols=symbols))
    assert info.versions_unread is expected


def test_huge_dt_strsz_reads_only_up_to_the_end_of_the_file():
    image = _set_dynamic_tag(
        _build_elf(needed=("libc.so.6",), soname="libw.so.1", symbols=_SYMBOLS), 10, 1 << 60)
    info = _parse(image)
    assert info is not None
    assert (info.needed, info.soname) == (["libc.so.6"], "libw.so.1")
    assert info.import_libraries["printf"] == "libc.so.6"
    assert any("DT_STRSZ" in note for note in info.notes), info.notes


def test_huge_symbol_count_skips_attribution_with_a_note():
    image = bytearray(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS))
    # DT_HASH's nchain is the symbol count: claim a table far past the file.
    rows, _ = _dynamic_entries_at(image)
    hash_at = next(value for _, tag, value in rows if tag == 4) - _BASE
    struct.pack_into("<I", image, hash_at + 4, 0xFFFFFF)
    info = _parse(bytes(image))
    assert info is not None and info.import_libraries == {}
    assert any("does not fit in the file" in note for note in info.notes), info.notes


@pytest.mark.parametrize("tag", [5, 6, 0x6FFFFFF0, 0x6FFFFFFE, 4])
@pytest.mark.parametrize("value", [_BASE + (1 << 40), (1 << 64) - 1, 0])
def test_addresses_past_the_file_are_never_read(tag, value):
    image = _set_dynamic_tag(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS), tag, value)
    info = _parse(image)
    assert info is not None
    assert isinstance(info.import_libraries, dict)


def test_pt_dynamic_past_the_end_is_noted():
    image = bytearray(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS))
    struct.pack_into("<Q", image, 64 + 56 + 8, 1 << 40)  # PT_DYNAMIC p_offset
    info = _parse(bytes(image))
    assert info is not None and info.needed == []
    assert any("PT_DYNAMIC" in note for note in info.notes), info.notes


@pytest.mark.parametrize("field_offset, value", [
    (0x36, 0xFFFF),   # e_phentsize
    (0x38, 0xFFFF),   # e_phnum
    (0x3A, 0xFFFF),   # e_shentsize
    (0x3C, 0xFFFF),   # e_shnum
])
def test_hostile_header_sizes_never_read_past_the_file(field_offset, value):
    image = bytearray(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS, count_source="section"))
    struct.pack_into("<H", image, field_offset, value)
    info = _parse(bytes(image))
    assert info is None or isinstance(info.import_libraries, dict)


def test_file_size_bounds_reads_even_when_the_reader_could_serve_more():
    # The parser is told the file is 200 bytes: every structure past that is
    # out of range, whatever the reader would have returned.
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS)
    info = _parse(image, file_size=200)
    assert info is None or info.import_libraries == {}


def test_versions_unread_names_the_missing_dynamic_symbol_table_1037():
    """#1037 review round 5: one condition covered a missing DT_VERSYM and a
    missing DT_SYMTAB, and always named the version table. With only the
    DT_SYMTAB declaration removed, the reason names the symbol table."""
    image = bytearray(_build_elf(needed=("libc.so.6",), symbols=_SYMBOLS))
    rows, fmt = _dynamic_entries_at(bytes(image))
    at, _tag, value = next(row for row in rows if row[1] == 6)   # DT_SYMTAB
    struct.pack_into(fmt, image, at, 21, value)                   # -> DT_DEBUG
    info = _parse(bytes(image))
    assert info.versions_unread == (
        "the file declares symbol versions without a dynamic symbol table")


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("damage", [
    "unsupported-version", "zero-version", "zero-aux", "overlapping-header-aux",
    "truncated-requirements", "overlapping-requirements", "unreadable-requirement",
    "truncated-aux", "overlapping-aux", "unreadable-aux",
    "conflicting-indices", "shared-aux", "malformed-later-requirement",
])
def test_malformed_verneed_discards_all_attribution_but_preserves_exports(is64, endian, damage):
    image = bytearray(_build_elf(
        is64=is64, endian=endian, needed=("liba.so", "libb.so"),
        symbols=(("from_a", "liba.so"), ("from_b", "libb.so")), defined=_DEFINED))
    rows, _ = _dynamic_entries_at(image, is64=is64, endian=endian)
    at = next(value for _, tag, value in rows if tag == 0x6FFFFFFE) - _BASE
    patches = {
        "unsupported-version": ("H", at, 2),
        "zero-version": ("H", at, 0),
        "zero-aux": ("I", at + 8, 0),
        "overlapping-header-aux": ("I", at + 8, 8),
        "truncated-requirements": ("I", at + 12, 0),
        "overlapping-requirements": ("I", at + 12, 8),
        "unreadable-requirement": ("I", at + 12, len(image) - at - 8),
        "truncated-aux": ("H", at + 2, 2),
        "overlapping-aux": ("H", at + 2, 2),
        "unreadable-aux": ("I", at + 8, len(image) - at - 8),
        "conflicting-indices": ("H", at + 32 + 16 + 6, 2),
        # Both requirements point at the second requirement's vernaux record.
        "shared-aux": ("I", at + 8, 48),
        "malformed-later-requirement": ("H", at + 32, 2),
    }
    fmt, offset, value = patches[damage]
    struct.pack_into(endian + fmt, image, offset, value)
    if damage == "overlapping-aux":
        struct.pack_into(endian + "I", image, at + 16 + 12, 8)
    info = _parse(bytes(image))
    assert info is not None
    assert info.import_libraries == {}
    assert info.versions_unread
    assert info.notes
    assert info.needed == ["liba.so", "libb.so"]
    assert info.exports == _EXPECTED_EXPORTS


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("other", [0, 1, 0x8000, 0x8001])
@pytest.mark.parametrize("requirement", [0, 1])
def test_reserved_vernaux_indices_cannot_claim_local_or_global_imports(
        is64, endian, other, requirement):
    image = bytearray(_build_elf(
        is64=is64, endian=endian, needed=("liba.so", "libb.so"),
        symbols=(("from_a", "liba.so"), ("from_b", "libb.so"),
                 ("local_import", None), ("global_import", None)),
        defined=_DEFINED))
    rows, _ = _dynamic_entries_at(image, is64=is64, endian=endian)
    verneed_at = next(value for _, tag, value in rows if tag == 0x6FFFFFFE) - _BASE
    versym_at = next(value for _, tag, value in rows if tag == 0x6FFFFFF0) - _BASE
    # The third import uses VER_NDX_LOCAL; the fourth retains VER_NDX_GLOBAL.
    # Neither may acquire the foreign library named by this malformed vernaux,
    # including when its reserved index is disguised with the high bit.
    struct.pack_into(endian + "H", image, versym_at + 3 * 2, 0)
    struct.pack_into(endian + "H", image, verneed_at + requirement * 32 + 16 + 6, other)

    info = _parse(bytes(image))
    assert info is not None
    # Reject the whole table, even if a valid requirement was already read.
    assert info.import_libraries == {}
    reason = "uses a reserved local/global version index"
    assert info.versions_unread and reason in info.versions_unread
    assert any(reason in note for note in info.notes), info.notes
    assert info.needed == ["liba.so", "libb.so"]
    assert info.exports == _EXPECTED_EXPORTS


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("other", [2, 0x8002])
def test_first_nonreserved_vernaux_index_attributes_only_its_versioned_import(
        is64, endian, other):
    image = bytearray(_build_elf(
        is64=is64, endian=endian, needed=("liba.so",),
        symbols=(("from_a", "liba.so"), ("local_import", None), ("global_import", None)),
        defined=_DEFINED))
    rows, _ = _dynamic_entries_at(image, is64=is64, endian=endian)
    verneed_at = next(value for _, tag, value in rows if tag == 0x6FFFFFFE) - _BASE
    versym_at = next(value for _, tag, value in rows if tag == 0x6FFFFFF0) - _BASE
    struct.pack_into(endian + "H", image, verneed_at + 16 + 6, other)
    struct.pack_into(endian + "H", image, versym_at + 2 * 2, 0)

    info = _parse(bytes(image))
    assert info is not None
    assert info.import_libraries == {"from_a": "liba.so"}
    assert info.versions_unread is None
    assert info.needed == ["liba.so"]
    assert info.exports == _EXPECTED_EXPORTS


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("total_aux", [16384, 16385])
def test_vernaux_work_limit_is_shared_across_requirements(is64, endian, total_aux):
    image = _build_elf(
        is64=is64, endian=endian, needed=("liba.so", "libb.so"),
        symbols=(("from_a", "liba.so"), ("from_b", "libb.so")), defined=_DEFINED)
    rows, _ = _dynamic_entries_at(image, is64=is64, endian=endian)
    old_at = next(value for _, tag, value in rows if tag == 0x6FFFFFFE) - _BASE
    table = bytearray()
    next_index = 16  # Leave the fixture's import/export version indices intact.
    for i, count in enumerate((8192, total_aux - 8192)):
        _, _, filename, _, _ = struct.unpack_from(endian + "HHIII", image, old_at + 32 * i)
        _, _, other, name, _ = struct.unpack_from(endian + "IHHII", image, old_at + 32 * i + 16)
        table += struct.pack(endian + "HHIII", 1, count, filename, 16,
                             16 + 16 * count if i == 0 else 0)
        for index in range(count):
            version = other if index == 0 else next_index
            next_index += 1
            table += struct.pack(endian + "IHHII", 0, 0, version, name,
                                 16 if index + 1 < count else 0)
    table_at = len(image)
    image = bytearray(_set_dynamic_tag(
        image, 0x6FFFFFFE, _BASE + table_at, is64=is64, endian=endian))
    image.extend(table)
    # Extend the fixture's whole-file PT_LOAD to include the replacement table.
    load_size_at = 64 + 32 if is64 else 52 + 16
    struct.pack_into(endian + ("QQ" if is64 else "II"),
                     image, load_size_at, len(image), len(image))
    table_bytes_read = 0
    invalid_windows = []

    def read(offset, length):
        nonlocal table_bytes_read
        if not (0 <= offset and 0 <= length and offset + length <= len(image)):
            invalid_windows.append((offset, length))
            return b""
        table_bytes_read += max(0, min(offset + length, len(image)) - max(offset, table_at))
        # Stop excess work even under a regression; assert the budget outside
        # the reader, where a parser's short-read handling cannot swallow it.
        if table_bytes_read > 16384 * 16 + 2 * 16:
            return b""
        return bytes(image[offset:offset + length])

    info = parse_elf_dynamic(read, len(image))
    assert not invalid_windows
    assert table_bytes_read <= 16384 * 16 + 2 * 16
    assert info is not None
    assert info.needed == ["liba.so", "libb.so"]
    assert info.exports == _EXPECTED_EXPORTS
    if total_aux == 16384:
        assert info.import_libraries == {"from_a": "liba.so", "from_b": "libb.so"}
        assert info.versions_unread is None
    else:
        assert info.import_libraries == {}
        assert info.versions_unread
        assert info.notes


def test_cancellation_check_precedes_every_reader_call():
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS, defined=_DEFINED)
    events = []

    def check():
        events.append("check")

    def read(offset, length):
        events.append("read")
        return image[offset:offset + length]

    info = parse_elf_dynamic(read, len(image), check=check)
    assert info is not None and info.exports == _EXPECTED_EXPORTS
    assert "read" in events
    assert all(index > 0 and events[index - 1] == "check"
               for index, event in enumerate(events) if event == "read")


@pytest.mark.parametrize("cancel_immediately", [True, False])
def test_cancellation_runtime_error_escapes_dynamic_parser(cancel_immediately):
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS, defined=_DEFINED)
    rows, _ = _dynamic_entries_at(image)
    verneed_at = next(value for _, tag, value in rows if tag == 0x6FFFFFFE) - _BASE
    cancelled = cancel_immediately
    reads_after_cancellation = []
    error = RuntimeError("cancel ELF discovery")

    def check():
        if cancelled:
            raise error

    def read(offset, length):
        nonlocal cancelled
        if cancelled:
            reads_after_cancellation.append((offset, length))
        if offset <= verneed_at < offset + length:
            cancelled = True
        return image[offset:offset + length]

    with pytest.raises(RuntimeError) as raised:
        parse_elf_dynamic(read, len(image), check=check)
    assert raised.value is error
    assert not reads_after_cancellation


# --- exports ----------------------------------------------------------------

_DEFINED = (
    _Def("widget_open", "WIDGET_1.0"),
    _Def("widget_table", "WIDGET_1.0", type=_STT_OBJECT),
    _Def("widget_tls", "WIDGET_1.1", type=_STT_TLS),
    _Def("widget_hook", bind=_STB_WEAK),                       # unversioned weak
    _Def("widget_internal_fn", "WIDGET_1.0", vis=_STV_HIDDEN),
    _Def("widget_secret", vis=_STV_INTERNAL),
    _Def("widget_protected", "WIDGET_1.1", vis=_STV_PROTECTED),
    _Def("widget_local", bind=_STB_LOCAL),
    _Def("widget_section", type=_STT_SECTION),
)

_EXPECTED_EXPORTS = {
    "widget_open": ExportedSymbol("widget_open", "WIDGET_1.0", "func", False),
    "widget_table": ExportedSymbol("widget_table", "WIDGET_1.0", "object", False),
    "widget_tls": ExportedSymbol("widget_tls", "WIDGET_1.1", "object", False),
    "widget_hook": ExportedSymbol("widget_hook", None, "func", True),
    "widget_protected": ExportedSymbol("widget_protected", "WIDGET_1.1", "func", False),
    "widget_section": ExportedSymbol("widget_section", None, "other", False),
}


@pytest.mark.parametrize("is64", [True, False])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("count_source", ["hash", "gnu_hash", "section"])
def test_exports_are_the_defined_bindable_symbols(is64, endian, count_source):
    """Defined GLOBAL/WEAK symbols with DEFAULT/PROTECTED visibility, under
    their verdef version; undefined imports, HIDDEN/INTERNAL and LOCAL rows
    are not exports. The three symbol-count sources all drive the list."""
    image = _build_elf(is64=is64, endian=endian, needed=("libc.so.6",), soname="libwidget.so.2",
                       symbols=_SYMBOLS, defined=_DEFINED, count_source=count_source)
    info = _parse(image)
    assert info is not None
    assert info.exports == _EXPECTED_EXPORTS
    assert "printf" not in info.exports and "widget_internal_fn" not in info.exports
    # The import side is untouched by the defined rows.
    assert info.import_libraries["printf"] == "libc.so.6"
    assert info.versions_unread is None


@pytest.mark.parametrize("is64, expected", [(True, (183, 64)), (False, (8, 32))])
def test_header_facts_are_reported(is64, expected):
    info = _parse(_build_elf(is64=is64, symbols=_SYMBOLS))
    assert (info.e_machine, info.elf_class, info.e_type) == (*expected, 3)


def test_exports_without_any_version_table_are_unversioned():
    # musl-style: no DT_VERSYM/DT_VERDEF at all; every export reads version None.
    image = _build_elf(symbols=(("printf", None),),
                       defined=(_Def("strdup"), _Def("environ", type=_STT_OBJECT, bind=_STB_WEAK)))
    info = _parse(image)
    assert info.exports == {
        "strdup": ExportedSymbol("strdup", None, "func", False),
        "environ": ExportedSymbol("environ", None, "object", True),
    }


def test_default_version_wins_over_a_hidden_compatibility_one():
    # `memcpy@GLIBC_2.2.5` (hidden) and `memcpy@@GLIBC_2.14` (default) share the
    # bare name: the default is the export, whichever order they appear in.
    for defined in ((_Def("memcpy", "GLIBC_2.2.5", hidden=True), _Def("memcpy", "GLIBC_2.14")),
                    (_Def("memcpy", "GLIBC_2.14"), _Def("memcpy", "GLIBC_2.2.5", hidden=True))):
        info = _parse(_build_elf(defined=defined))
        assert info.exports == {"memcpy": ExportedSymbol("memcpy", "GLIBC_2.14", "func", False)}
    # Only a hidden definition: it is still the only thing the name binds to.
    info = _parse(_build_elf(defined=(_Def("memcpy", "GLIBC_2.2.5", hidden=True),)))
    assert info.exports["memcpy"].version == "GLIBC_2.2.5"


def test_no_symbol_count_source_exports_nothing():
    info = _parse(_build_elf(symbols=_SYMBOLS, defined=_DEFINED, count_source="none"))
    assert info is not None and info.exports == {}


def test_non_abi_dt_syment_exports_nothing_with_a_note():
    image = _set_dynamic_tag(_build_elf(symbols=(("puts", None),), defined=_DEFINED), 11, 1 << 56)
    info = _parse(image)
    assert info.exports == {} and info.import_libraries == {}
    assert any("DT_SYMENT" in note for note in info.notes), info.notes


@pytest.mark.parametrize("value", [_BASE + (1 << 40), (1 << 64) - 1, 0])
def test_unreadable_verdef_leaves_exports_unversioned(value):
    image = _set_dynamic_tag(_build_elf(symbols=_SYMBOLS, defined=_DEFINED), 0x6FFFFFFC, value)
    info = _parse(image)
    assert set(info.exports) == set(_EXPECTED_EXPORTS)
    assert all(export.version is None for export in info.exports.values())


def test_truncated_image_with_exports_never_reads_past_the_end():
    image = _build_elf(needed=("libc.so.6",), symbols=_SYMBOLS, defined=_DEFINED)
    for cut in range(0, len(image), 7):
        info = _parse(image[:cut])
        assert info is None or isinstance(info.exports, dict)


# --- parse_elf_file -----------------------------------------------------------


@pytest.mark.parametrize("cancel_at", [None, "before-open", "after-read"])
def test_parse_elf_file_reads_a_file_piecewise(tmp_path, monkeypatch, cancel_at):
    image = _build_elf(needed=("libc.so.6",), soname="libwidget.so.2",
                       symbols=_SYMBOLS, defined=_DEFINED)
    path = tmp_path / "libwidget.so.2"
    path.write_bytes(image)
    # Observe the real descriptor-backed handle, including its closure when
    # cancellation interrupts parsing after a read.
    import bn_agent_bridge.elf_dynamic as elf_dynamic
    windows = []
    handles = []
    events = []
    error = RuntimeError("cancel file discovery")
    real_fdopen = os.fdopen

    def check():
        events.append("check")
        if cancel_at == "before-open" or (cancel_at == "after-read" and windows):
            raise error

    class CappedHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def fileno(self):
            return self._handle.fileno()

        def seek(self, offset):
            return self._handle.seek(offset)

        def read(self, size=-1):
            assert 0 <= size <= len(image), f"read of {size} bytes"
            events.append("read")
            windows.append(size)
            return self._handle.read(size)

    def fdopen(fd, mode):
        handle = real_fdopen(fd, mode)
        handles.append(handle)
        return CappedHandle(handle)

    monkeypatch.setattr(elf_dynamic.os, "fdopen", fdopen)
    if cancel_at is not None:
        with pytest.raises(RuntimeError) as raised:
            parse_elf_file(str(path), check=check)
        assert raised.value is error
        assert bool(windows) == (cancel_at == "after-read")
        assert all(handle.closed for handle in handles)
        return
    info = parse_elf_file(str(path), check=check)
    assert all(handle.closed for handle in handles)
    assert all(index > 0 and events[index - 1] == "check"
               for index, event in enumerate(events) if event == "read")
    assert info is not None
    assert len(windows) > 1 and max(windows) < len(image)
    assert info.soname == "libwidget.so.2" and info.needed == ["libc.so.6"]
    assert info.exports == _EXPECTED_EXPORTS
    assert info.import_libraries["printf"] == "libc.so.6"
    assert (info.e_machine, info.elf_class) == (183, 64)


def test_parse_elf_file_is_none_for_non_elf_missing_or_directory(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_bytes(b"MZ" + bytes(100))
    assert parse_elf_file(str(text)) is None
    assert parse_elf_file(str(tmp_path / "absent.so")) is None
    assert parse_elf_file(str(tmp_path)) is None
    (tmp_path / "empty.so").write_bytes(b"")
    assert parse_elf_file(str(tmp_path / "empty.so")) is None



def _parse_fifo_in_worker(path, connection):
    try:
        connection.send("ready")
        connection.send(parse_elf_file(path))
    finally:
        connection.close()


# Readiness is a fresh `spawn` interpreter importing this module: measured ~0.5s
# alone, but under two `-n 8` suites it has exceeded 10s. The deadline is
# generous because it proves nothing about `parse_elf_file`; the SHORT timeout
# below, which starts only once the worker is running, is the proof.
_FIFO_WORKER_START_DEADLINE = 120.0
_FIFO_PARSE_TIMEOUT = 5.0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_parse_elf_file_rejects_fifo_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "library.fifo"
    os.mkfifo(path)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_parse_fifo_in_worker, args=(str(path), sender))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        # Deadline-based readiness: wait for "ready", but stop as soon as the
        # worker is gone (an import error must not cost the whole deadline).
        deadline = time.monotonic() + _FIFO_WORKER_START_DEADLINE
        while not receiver.poll(0.5):
            assert process.is_alive(), (
                f"FIFO worker exited before reporting ready (exit code {process.exitcode})")
            assert time.monotonic() < deadline, (
                f"FIFO worker did not start within {_FIFO_WORKER_START_DEADLINE:.0f}s")
        assert receiver.recv() == "ready"
        # The property under test: a running worker that calls `parse_elf_file`
        # on a FIFO nobody writes to gets its answer (None) within a short,
        # fixed time instead of blocking in `open()`.
        assert receiver.poll(_FIFO_PARSE_TIMEOUT), (
            f"parse_elf_file blocked opening a FIFO for {_FIFO_PARSE_TIMEOUT:.0f}s")
        assert receiver.recv() is None
        process.join(timeout=_FIFO_PARSE_TIMEOUT)
        assert not process.is_alive(), "FIFO worker did not exit after answering"
        assert process.exitcode == 0
    finally:
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            if not process.is_alive():
                process.close()
        receiver.close()
        sender.close()
        path.unlink()
