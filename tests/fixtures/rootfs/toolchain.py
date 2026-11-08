"""Resolve a compiler's musl runtime without executing any target code."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


_MAX_ELF_SIZE = 64 * 1024 * 1024


def _elf(path: Path) -> tuple[bytes, str, tuple[int, int, int]]:
    with path.open("rb") as stream:
        data = stream.read(_MAX_ELF_SIZE + 1)
    if len(data) > _MAX_ELF_SIZE or len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError(f"not a bounded ELF file: {path}")
    if data[4] not in (1, 2) or data[5] not in (1, 2):
        raise ValueError(f"unsupported ELF encoding: {path}")
    endian = "<" if data[5] == 1 else ">"
    machine = struct.unpack_from(endian + "H", data, 18)[0]
    return data, endian, (data[4], data[5], machine)


def _interpreter(path: Path) -> tuple[str, tuple[int, int, int]]:
    data, endian, identity = _elf(path)
    wide = identity[0] == 2
    offset = struct.unpack_from(endian + ("Q" if wide else "I"), data, 32 if wide else 28)[0]
    size, count = struct.unpack_from(endian + "HH", data, 54 if wide else 42)
    if size < (56 if wide else 32) or count > 1024 or offset + size * count > len(data):
        raise ValueError("invalid ELF program header table in compiler probe")
    for index in range(count):
        start = offset + index * size
        if struct.unpack_from(endian + "I", data, start)[0] != 3:
            continue
        file_offset = struct.unpack_from(endian + ("Q" if wide else "I"), data, start + (8 if wide else 4))[0]
        length = struct.unpack_from(endian + ("Q" if wide else "I"), data, start + (32 if wide else 16))[0]
        if not 1 < length <= 4096 or file_offset + length > len(data):
            raise ValueError("invalid ELF interpreter in compiler probe")
        value = data[file_offset:file_offset + length]
        if value[-1:] != b"\0" or b"\0" in value[:-1]:
            raise ValueError("invalid ELF interpreter string in compiler probe")
        return value[:-1].decode("utf-8"), identity
    raise ValueError("compiler probe has no dynamic interpreter")


def resolve(cc: str, libc: str = "", loader: str = "") -> tuple[Path, str]:
    """Use the linked interpreter, then compiler paths, never an arbitrary loader glob."""
    def compiler(*args: str, source: str | None = None) -> str:
        result = subprocess.run(
            [cc, *args], input=source, capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise ValueError(f"{cc} {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="bn-musl-probe-") as directory:
        probe = Path(directory) / "probe"
        linked = compiler("-x", "c", "-", "-Wl,-t", "-o", str(probe),
                          source="int main(void) { return 0; }\n")
        interpreter, identity = _interpreter(probe)
    interpreter_path = Path(interpreter)
    interpreter_name = interpreter_path.name
    if not interpreter_path.is_absolute() or not (
        interpreter_name.startswith("ld-musl-") and interpreter_name.endswith(".so.1")
    ):
        raise ValueError(f"{cc} links a non-musl interpreter: {interpreter}")
    loader = loader or interpreter_name
    if Path(loader).name != loader or not (loader.startswith("ld-musl-") and loader.endswith(".so.1")):
        raise ValueError("LDMUSL must be an ld-musl-<arch>.so.1 filename")

    if libc:
        candidates = [Path(libc)]
    else:
        # GNU ld's trace includes wrapper specs paths absent from GCC's
        # generic search-directory report.
        candidates = [Path(line.strip()) for line in linked.splitlines()
                      if Path(line.strip()).name == "libc.so"]
        sysroot_text = compiler("-print-sysroot")
        sysroot = Path(sysroot_text) if sysroot_text else None
        if sysroot is not None:
            candidates.extend((sysroot / interpreter.lstrip("/"), sysroot / "lib/libc.so",
                               sysroot / "usr/lib/libc.so"))
        reported = compiler("-print-file-name=libc.so")
        if reported and reported != "libc.so":
            candidates.append(Path(reported))
        for line in compiler("-print-search-dirs").splitlines():
            if not line.startswith("libraries:"):
                continue
            for item in line.partition(":")[2].strip().removeprefix("=").split(os.pathsep):
                if not item:
                    continue
                directory = (sysroot / item[1:].lstrip("/")
                             if item.startswith("=") and sysroot else Path(item))
                candidates.extend((directory / "libc.so", directory / interpreter_name))
        # Distro musl-gcc wrappers can report their underlying glibc GCC paths.
        # Their linked loader normally points directly at the actual musl libc.
        candidates.append(interpreter_path)
        for directory in (Path("/usr/lib"), Path("/lib")):
            candidates.extend(sorted(directory.glob("*-linux-musl*/libc.so")))

    for candidate in dict.fromkeys(candidates):
        try:
            data, endian, candidate_identity = _elf(candidate)
        except (OSError, ValueError):
            continue
        if (candidate_identity == identity
                and struct.unpack_from(endian + "H", data, 16)[0] == 3
                and b"musl libc" in data):
            return candidate.resolve(), loader
    setting = f"LIBC_SRC={libc}" if libc else "the compiler's sysroot/search paths and distro musl directories"
    raise ValueError(f"no matching musl ELF libc in {setting}; set LIBC_SRC to this compiler's musl libc.so")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--libc", default="")
    parser.add_argument("--loader", default="")
    args = parser.parse_args()
    try:
        libc, loader = resolve(args.cc, args.libc, args.loader)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"musl fixture toolchain: {exc}", file=sys.stderr)
        return 1
    print(libc, loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
