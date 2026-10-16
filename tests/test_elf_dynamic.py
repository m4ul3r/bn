"""`elf_dynamic.parse_elf_dynamic` against synthetic ELF images (#969).

The images are built here byte by byte -- a PT_LOAD over the whole file, a
PT_DYNAMIC, `.dynstr`, `.dynsym`, `.gnu.version`, `.gnu.version_r` and a symbol
count source -- so each test controls exactly which piece of the loader's
answer is present, in both classes and both byte orders.
"""
from __future__ import annotations

import struct

import pytest

from bn_agent_bridge.elf_dynamic import (
    NONDEFAULT_ONLY,
    NOT_EXPORTED,
    SEVERAL_EXPORTERS,
    UNREAD_BEFORE,
    library_search_dirs,
    needed_export_libraries,
    parse_elf_dynamic,
    read_elf_file,
)

_BASE = 0x400000


def _build_elf(*, is64=True, endian="<", needed=(), soname=None, rpath=None, runpath=None,
               symbols=(), count_source="hash", defined=(), machine=None):
    """An ELF image whose dynamic symbols are ``symbols``: ``(name, library)``
    pairs, ``library`` None for an unversioned symbol. ``count_source`` is
    ``"hash"`` (DT_HASH), ``"section"`` (a SHT_DYNSYM header) or ``"none"``.
    ``defined`` adds DEFINED symbols as ``(name, st_info, st_other[, versym])``
    (versym 1, unversioned, by default); ``machine`` overrides e_machine."""
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
    defined_names = [(intern(d[0]), d[1], d[2]) for d in defined]

    syment = 24 if is64 else 16
    dynsym = bytearray(syment)  # index 0: the null symbol
    for name_off in sym_names:
        if is64:
            dynsym += struct.pack(e + "IBBHQQ", name_off, 0x12, 0, 0, 0, 0)
        else:
            dynsym += struct.pack(e + "IIIBBH", name_off, 0, 0, 0x12, 0, 0)
    for name_off, info, other in defined_names:
        if is64:
            dynsym += struct.pack(e + "IBBHQQ", name_off, info, other, 1, 0x1000, 8)
        else:
            dynsym += struct.pack(e + "IIIBBH", name_off, 0x1000, 8, info, other, 1)
    versym = struct.pack(e + "H", 0) + b"".join(
        struct.pack(e + "H", version_index[lib] if lib else 1) for _, lib in symbols) + b"".join(
        struct.pack(e + "H", d[3] if len(d) > 3 else 1) for d in defined)
    verneed = bytearray()
    for i, lib in enumerate(libraries):
        next_need = 32 if i + 1 < len(libraries) else 0
        verneed += struct.pack(e + "HHIII", 1, 1, lib_file[lib], 16, next_need)
        verneed += struct.pack(e + "IHHII", 0, 0, version_index[lib], lib_ver[lib], 0)
    nsyms = len(symbols) + len(defined) + 1
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
        header = struct.pack(e + "HHIQQQIHHHHHH", 3, 183 if machine is None else machine, 1, 0,
                             ehsize, shoff, 0, ehsize, phentsize, 2, shentsize, shnum, 0)
        load = struct.pack(e + "IIQQQQQQ", 1, 5, 0, _BASE, _BASE, total, total, 0x1000)
        dyn_ph = struct.pack(e + "IIQQQQQQ", 2, 6, blobs["dynamic"][0],
                             _BASE + blobs["dynamic"][0], 0, len(dyn), len(dyn), 8)
    else:
        header = struct.pack(e + "HHIIIIIHHHHHH", 3, 8 if machine is None else machine, 1, 0,
                             ehsize, shoff, 0, ehsize, phentsize, 2, shentsize, shnum, 0)
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


# --- #1037 item 6: attributing unversioned imports from needed libraries ----

_GLOBAL_FUNC, _WEAK_FUNC, _LOCAL_FUNC = 0x12, 0x22, 0x02
_HIDDEN = 2
_X86_64 = (2, 1, 62, 0)       # ELFCLASS64, ELFDATA2LSB, EM_X86_64, SYSV
_AARCH64 = (2, 1, 183, 0)


def _library(names, *, is64=True, endian="<", machine=None, hidden=(), local=(),
             needed=(), versioned=()):
    """A shared library image exporting *names* (global functions), plus
    defined-but-not-exported *hidden* and *local* symbols, *versioned*
    ``(name, versym)`` definitions, and one undefined import of its own."""
    defined = [(n, _GLOBAL_FUNC, 0) for n in names]
    defined += [(n, _GLOBAL_FUNC, _HIDDEN) for n in hidden]
    defined += [(n, _LOCAL_FUNC, 0) for n in local]
    defined += [(n, _GLOBAL_FUNC, 0, versym) for n, versym in versioned]
    return _build_elf(is64=is64, endian=endian, machine=machine, needed=needed,
                      symbols=(("malloc", None),), defined=defined)


@pytest.mark.parametrize("is64, endian", [(True, "<"), (False, ">")])
def test_exports_are_the_defined_symbols_other_modules_can_bind(is64, endian):
    image = _library(["_ZN8Registry3getEv"], is64=is64, endian=endian,
                     hidden=["internal_helper"], local=["static_helper"],
                     versioned=[("chosen", 0x8003), ("current", 2), ("both", 0x8003),
                                ("both", 3)])
    size = len(image)
    info = parse_elf_dynamic(lambda o, n: image[o:o + n], size, with_exports=True)
    # The undefined import, the hidden definition and the local one are not
    # symbols the loader resolves another module's reference against. A
    # nondefault `chosen@V` is exported but not as a default definition.
    assert info.exports == {"_ZN8Registry3getEv": True, "chosen": False, "current": True,
                            "both": True}
    assert parse_elf_dynamic(lambda o, n: image[o:o + n], size).exports is None


def _write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image)
    return str(path)


def _lookup(needed, dirs, *, abi=_AARCH64, flags=0, sysroot=None):
    return needed_export_libraries(needed, [str(d) for d in dirs], abi=abi, flags=flags,
                                   sysroot=sysroot)


def test_a_name_is_attributed_only_when_one_library_read_defines_it(tmp_path):
    _write(tmp_path / "libz_first.so", _library(["shared_name", "only_first"]))
    _write(tmp_path / "liba_second.so", _library(["shared_name", "only_second"]))
    found = _lookup(["libz_first.so", "liba_second.so"], [tmp_path])
    assert found.verdict("only_second").library == "liba_second.so"
    # Two exporters: no load-order rule picks one; both are listed, neutrally.
    shared = found.verdict("shared_name")
    assert (shared.library, shared.exporting, shared.note) == (
        None, ["libz_first.so", "liba_second.so"], SEVERAL_EXPORTERS)
    assert found.verdict("absent").note == NOT_EXPORTED
    assert [(f.library, f.path) for f in found.files] == [
        ("libz_first.so", str(tmp_path / "libz_first.so")),
        ("liba_second.so", str(tmp_path / "liba_second.so"))]


def test_a_nondefault_symbol_version_is_never_attributed_1037_review(tmp_path):
    """Review blocker 1: libold exported only `chosen@OLD_1` (hidden), libgood
    a plain `chosen`; the process bound libgood's, and bn named libold."""
    _write(tmp_path / "libold.so", _library([], versioned=[("chosen", 0x8003)]))
    _write(tmp_path / "libgood.so", _library(["chosen"]))
    both = _lookup(["libold.so", "libgood.so"], [tmp_path]).verdict("chosen")
    assert (both.library, both.exporting) == (None, ["libold.so", "libgood.so"])
    only_old = _lookup(["libold.so"], [tmp_path]).verdict("chosen")
    assert (only_old.library, only_old.note) == (None, NONDEFAULT_ONLY)


def test_no_attribution_past_a_needed_library_that_was_not_read(tmp_path):
    # libmissing.so comes first and could define `only_second`.
    _write(tmp_path / "libz_first.so", _library(["only_first"]))
    _write(tmp_path / "liba_second.so", _library(["only_second"]))
    found = _lookup(["libz_first.so", "libmissing.so", "liba_second.so"], [tmp_path])
    assert found.verdict("only_first").library == "libz_first.so"
    blocked = found.verdict("only_second")
    assert (blocked.library, blocked.exporting, blocked.note) == (
        None, ["liba_second.so"], UNREAD_BEFORE)
    assert found.files[1].path is None and found.files[1].reason


@pytest.mark.parametrize("is64, endian, machine", [
    (False, "<", 62),   # Sol: an x32 ELF32 file shares EM_X86_64 with ELF64
    (True, ">", 62),    # Opus: the other byte order on the same machine
    (True, "<", 3),     # another machine
])
def test_a_library_of_another_abi_is_skipped_like_the_loader_does(tmp_path, is64, endian,
                                                                   machine):
    """Review blocker 2: e_machine alone accepted a file of the wrong class or
    byte order, and attributed the import to the wrong file."""
    _write(tmp_path / "wrongabi" / "libabiA.so",
           _library(["foo"], is64=is64, endian=endian, machine=machine))
    _write(tmp_path / "rightabi" / "libabiA.so", _library(["a_only"], machine=62))
    _write(tmp_path / "rightabi" / "libabiB.so", _library(["foo"], machine=62))
    found = _lookup(["libabiA.so", "libabiB.so"],
                    [tmp_path / "wrongabi", tmp_path / "rightabi"], abi=_X86_64)
    assert found.verdict("foo").library == "libabiB.so"
    assert found.files[0].path == str(tmp_path / "rightabi" / "libabiA.so")


def test_a_library_with_other_e_flags_is_not_read(tmp_path):
    image = bytearray(_library(["foo"], machine=40, is64=False))
    struct.pack_into("<I", image, 0x24, 0x5000200)   # soft-float EABI5
    _write(tmp_path / "libarm.so", bytes(image))
    found = _lookup(["libarm.so"], [tmp_path], abi=(1, 1, 40, 0), flags=0x5000400)
    assert found.files[0].path is None and "e_flags" in found.files[0].reason
    assert found.verdict("foo").library is None


def test_sysroot_symlinks_resolve_inside_the_sysroot_1037_review(tmp_path):
    """Review blocker 3: `sr/usr/lib/libA.so -> /abs/host/libA.so` was read
    from the host. Under a sysroot an absolute link is re-rooted, and a link
    that climbs out of the root is refused."""
    root = tmp_path / "sr"
    host = tmp_path / "host"
    _write(host / "libA.so", _library(["host_only"]))
    _write(root / "opt" / "real" / "libA.so", _library(["target_only"]))
    (root / "usr" / "lib").mkdir(parents=True)
    (root / "usr" / "lib" / "libA.so").symlink_to("/opt/real/libA.so")
    (root / "usr" / "lib" / "libB.so").symlink_to(host / "libA.so")      # absolute, host path
    (root / "usr" / "lib" / "libC.so").symlink_to("../../../host/libA.so")  # climbs out
    dirs = library_search_dirs(binary_path=None, runpath=None, rpath=None, sysroot=str(root))
    found = _lookup(["libA.so", "libB.so", "libC.so"], dirs, sysroot=str(root))
    assert found.files[0].path == str(root / "opt" / "real" / "libA.so")
    assert found.verdict("target_only").library == "libA.so"
    assert found.verdict("host_only").library is None
    # libB's absolute target, re-rooted, does not exist inside the sysroot.
    assert found.files[1].path is None
    assert found.files[2].path is None and "outside the sysroot" in found.files[2].reason


def test_without_a_sysroot_an_absolute_library_symlink_is_not_followed(tmp_path):
    host = _write(tmp_path / "host" / "libA.so", _library(["host_only"]))
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "libA.so").symlink_to(host)
    (tmp_path / "app" / "libR.so").symlink_to("../host/libA.so")
    found = _lookup(["libA.so", "libR.so"], [tmp_path / "app"])
    assert found.files[0].path is None and "absolute symlink" in found.files[0].reason
    assert found.files[1].path == host


def test_search_dirs_expand_origin_and_read_absolute_entries_only_under_a_sysroot(tmp_path):
    binary = _write(tmp_path / "app" / "bin" / "regd", b"")
    without = library_search_dirs(binary_path=binary, runpath="$ORIGIN/../lib:/opt/vendor/lib",
                                  rpath=None, sysroot=None)
    origin = str(tmp_path / "app" / "bin")
    # `/opt/vendor/lib` on THIS machine is a host directory, not the target's.
    assert without == [str(tmp_path / "app" / "lib"), origin]
    sysroot = tmp_path / "rootfs"
    (sysroot / "usr" / "lib" / "aarch64-linux-gnu").mkdir(parents=True)
    with_root = library_search_dirs(binary_path=binary, runpath="$ORIGIN/../lib:/opt/vendor/lib",
                                    rpath="/ignored/when/runpath", sysroot=str(sysroot))
    assert with_root[:3] == [str(tmp_path / "app" / "lib"),
                             str(sysroot / "opt" / "vendor" / "lib"), origin]
    assert str(sysroot / "usr" / "lib" / "aarch64-linux-gnu") in with_root
    assert "/ignored/when/runpath" not in " ".join(with_root)


def test_read_elf_file_reports_the_abi_identity(tmp_path):
    info = read_elf_file(_write(tmp_path / "lib32.so", _library(["x"], is64=False, endian=">",
                                                                  machine=8)))
    assert (info.abi, info.flags) == ((1, 2, 8, 0), 0)
