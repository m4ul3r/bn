"""Behavioral coverage for synthetic firmware toolchain/runtime selection."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import pytest

import conftest


ROOTFS_DIR = Path(__file__).parent / "fixtures" / "rootfs"
_spec = importlib.util.spec_from_file_location("rootfs_toolchain", ROOTFS_DIR / "toolchain.py")
assert _spec is not None and _spec.loader is not None
_toolchain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toolchain)


def _elf(*, machine: int = 183, interpreter: str | None = None, musl: bool = False) -> bytes:
    data = bytearray(128)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", data, 16, 3, machine, 1)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HHH", data, 52, 64, 56, int(interpreter is not None))
    if interpreter is not None:
        value = interpreter.encode() + b"\0"
        struct.pack_into("<IIQQQQQQ", data, 64, 3, 4, 128, 0, 0, len(value), len(value), 1)
        data.extend(value)
    if musl:
        data.extend(b"musl libc (aarch64)\0")
    return bytes(data)


@pytest.fixture
def compiler(tmp_path):
    """A compiler protocol fake emits synthetic ELF; no target code is executed."""
    runtime = tmp_path / "usr/lib/aarch64-linux-musl/libc.so"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(_elf(musl=True))
    glibc = tmp_path / "host/libc.so"
    glibc.parent.mkdir()
    glibc.write_text("GROUP ( libc.so.6 AS_NEEDED ( ld-linux-x86-64.so.2 ) )")
    probe = tmp_path / "probe.elf"
    probe.write_bytes(_elf(interpreter=str(tmp_path / "lib/ld-musl-aarch64.so.1")))
    config = {
        "probe": str(probe), "sysroot": "", "libc": str(glibc),
        "search": f"libraries: ={glibc.parent}{os.pathsep}{runtime.parent}", "trace": "",
    }
    settings = tmp_path / "compiler.json"
    settings.write_text(json.dumps(config))
    executable = tmp_path / "musl-gcc"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, shutil, sys\n"
        f"config = json.loads(pathlib.Path({str(settings)!r}).read_text())\n"
        "args = sys.argv[1:]\n"
        "if '-print-sysroot' in args: print(config['sysroot'])\n"
        "elif '-print-file-name=libc.so' in args: print(config['libc'])\n"
        "elif '-print-search-dirs' in args: print(config['search'])\n"
        "else:\n"
        "    output = pathlib.Path(args[args.index('-o') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(config['probe'], output)\n"
        "    print(config['trace'])\n"
    )
    executable.chmod(0o755)
    return executable, runtime, probe, settings, config


def test_empty_sysroot_wrapper_uses_musl_search_path_and_linked_loader(compiler):
    cc, runtime, _, _, _ = compiler
    # An unrelated loader next to libc must not determine the fixture's alias.
    (runtime.parent / "ld-musl-x86_64.so.1").write_bytes(_elf(machine=62, musl=True))
    libc, loader = _toolchain.resolve(str(cc))
    assert libc == runtime.resolve()
    assert loader == "ld-musl-aarch64.so.1"


def test_linker_trace_precedes_unrelated_same_arch_runtime(compiler, tmp_path):
    cc, runtime, _, settings, config = compiler
    unrelated = tmp_path / "unrelated/libc.so"
    unrelated.parent.mkdir()
    unrelated.write_bytes(_elf(musl=True) + b"unrelated")
    config.update(libc=str(unrelated), trace=str(runtime))
    settings.write_text(json.dumps(config))
    assert _toolchain.resolve(str(cc))[0] == runtime.resolve()


def test_explicit_runtime_and_loader_are_honored(compiler, tmp_path):
    cc, _, _, _, _ = compiler
    explicit = tmp_path / "explicit-libc.so"
    explicit.write_bytes(_elf(musl=True) + b"explicit")
    assert _toolchain.resolve(str(cc), str(explicit), "ld-musl-aarch64-custom.so.1") == (
        explicit.resolve(), "ld-musl-aarch64-custom.so.1",
    )


@pytest.mark.parametrize("data", [b"GROUP ( libc.so.6 )", _elf(machine=62, musl=True), _elf()])
def test_invalid_explicit_libc_does_not_fall_back(compiler, tmp_path, data):
    cc, _, _, _, _ = compiler
    explicit = tmp_path / "invalid-libc.so"
    explicit.write_bytes(data)
    with pytest.raises(ValueError, match="no matching musl ELF libc"):
        _toolchain.resolve(str(cc), str(explicit))


def test_non_musl_compiler_is_rejected(compiler):
    cc, _, probe, _, _ = compiler
    probe.write_bytes(_elf(interpreter="/lib64/ld-linux-x86-64.so.2"))
    with pytest.raises(ValueError, match="non-musl interpreter"):
        _toolchain.resolve(str(cc))


def test_truncated_program_table_is_rejected(compiler):
    cc, _, probe, _, _ = compiler
    data = bytearray(_elf(interpreter="/lib/ld-musl-aarch64.so.1"))
    struct.pack_into("<Q", data, 32, 2**63)
    probe.write_bytes(data)
    with pytest.raises(ValueError, match="invalid ELF program header table"):
        _toolchain.resolve(str(cc))


def test_make_build_uses_environment_overrides(compiler, tmp_path, monkeypatch):
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is required for the fixture-build contract")
    cc, runtime, _, _, _ = compiler
    explicit = tmp_path / "explicit-libc.so"
    explicit.write_bytes(runtime.read_bytes() + b"explicit runtime")
    output = tmp_path / "rootfs"
    monkeypatch.setenv("CROSS_CC", str(cc))
    monkeypatch.setenv("LIBC_SRC", str(explicit))
    monkeypatch.setenv("LDMUSL", "ld-musl-aarch64-custom.so.1")
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setattr(conftest, "FIXTURES_DIR", tmp_path / "build-locks")
    assert conftest.build_rootfs_fixture(out_dir=output) == output
    assert (output / "lib/libc.so").read_bytes() == explicit.read_bytes()
    loader = output / "lib/ld-musl-aarch64-custom.so.1"
    assert loader.is_symlink() and loader.readlink() == Path("libc.so")
    assert (output / "usr/bin/app").is_file()
    assert (output / "opt/vendor/lib/libgreet.so").is_symlink()


def test_cross_sysroot_runtime_precedes_host_paths(compiler, tmp_path):
    cc, _, probe, settings, config = compiler
    sysroot = tmp_path / "cross-sysroot"
    runtime = sysroot / "lib/libc.so"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(_elf(musl=True) + b"cross runtime")
    probe.write_bytes(_elf(interpreter="/lib/ld-musl-aarch64.so.1"))
    config["sysroot"] = str(sysroot)
    settings.write_text(json.dumps(config))
    assert _toolchain.resolve(str(cc)) == (runtime.resolve(), "ld-musl-aarch64.so.1")


def test_distro_loader_resolves_when_wrapper_reports_only_host_paths(compiler, tmp_path):
    cc, runtime, probe, settings, config = compiler
    loader = tmp_path / "lib/ld-musl-aarch64.so.1"
    loader.parent.mkdir()
    loader.symlink_to(runtime)
    probe.write_bytes(_elf(interpreter=str(loader)))
    config["search"] = f"libraries: ={tmp_path / 'host'}"
    settings.write_text(json.dumps(config))
    assert _toolchain.resolve(str(cc)) == (runtime.resolve(), loader.name)
