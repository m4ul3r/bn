from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Event, Thread
from typing import Callable, TypeVar

import pytest

from bn import rootfs_layout as rl


REPO_ROOT = Path(__file__).resolve().parents[1]
ELF_STUB = b"\x7fELF" + b"\0" * 12


def _touch(root: Path, rel: str, data: bytes = b"") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _elf(root: Path, rel: str, data: bytes = ELF_STUB) -> Path:
    return _touch(root, rel, data)


def _link(root: Path, rel: str, target: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, p)
    return p


def _linux_rootfs(tmp_path: Path) -> Path:
    root = tmp_path / "rootfs"
    for d in ("bin", "sbin", "etc", "usr/bin", "usr/lib", "lib", "opt/vendor/lib"):
        (root / d).mkdir(parents=True, exist_ok=True)
    _elf(root, "lib/libc.so")
    _link(root, "lib/ld-musl-aarch64.so.1", "libc.so")
    _elf(root, "opt/vendor/lib/libgreet.so.1", ELF_STUB + b"greet")
    _link(root, "opt/vendor/lib/libgreet.so", "libgreet.so.1")
    _elf(root, "usr/bin/libstray.so")
    _touch(root, "etc/ld.so.conf", b"/usr/local/lib\ninclude /etc/ld.so.conf.d/*.conf\n")
    _touch(root, "etc/ld.so.conf.d/vendor.conf", b"# vendor\n/opt/vendor/lib\n")
    _touch(root, "etc/profile", b"export LD_LIBRARY_PATH=/opt/extra/lib:$LD_LIBRARY_PATH\n")
    return root


def _paths(lps: list[rl.LookupPath]) -> list[str]:
    return [lp.path_in_root for lp in lps]


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detect_linux_rootfs(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    info = rl.detect_root(str(root))
    assert info is not None
    assert info.platform == "linux"
    assert info.root == os.path.realpath(root)
    assert "etc/ld.so.conf" in info.markers
    assert "lib/ld-musl-*.so*" in info.markers
    assert {"bin", "etc", "usr"} <= set(info.markers)


def test_detect_walks_up_from_nested_dir(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    nested = root / "usr" / "share" / "deep"
    nested.mkdir(parents=True)
    info = rl.detect_root(str(nested))
    assert info is not None and info.root == os.path.realpath(root)


def test_detect_android_rootfs(tmp_path: Path) -> None:
    root = tmp_path / "android"
    (root / "system" / "lib").mkdir(parents=True)
    (root / "system" / "bin").mkdir(parents=True)
    info = rl.detect_root(str(root))
    assert info is not None and info.platform == "android"
    assert info.markers == ["system/lib", "system/bin"]

    root2 = tmp_path / "android2"
    _touch(root2, "system/build.prop", b"ro.build.id=X\n")
    info2 = rl.detect_root(str(root2))
    assert info2 is not None and info2.platform == "android"
    assert info2.markers == ["system/build.prop"]


def test_android_wins_over_linux_markers(tmp_path: Path) -> None:
    root = tmp_path / "hybrid"
    _touch(root, "system/build.prop")
    for d in ("etc", "bin"):
        (root / d).mkdir()
    _touch(root, "init")
    info = rl.detect_root(str(root))
    assert info is not None and info.platform == "android"


def test_repo_checkout_is_not_a_rootfs() -> None:
    assert rl.detect_root(str(REPO_ROOT)) is None
    assert rl.detect_root(str(REPO_ROOT / "src" / "bn")) is None


def test_layout_without_loader_marker_is_not_a_rootfs(tmp_path: Path) -> None:
    root = tmp_path / "notroot"
    (root / "usr").mkdir(parents=True)
    (root / "etc").mkdir()
    assert rl.detect_root(str(root)) is None


def test_loader_marker_alone_is_not_enough(tmp_path: Path) -> None:
    root = tmp_path / "onlylib"
    _elf(root, "lib/libc.so.6")
    assert rl.detect_root(str(root)) is None


def test_explicit_root_unknown_platform(tmp_path: Path) -> None:
    root = tmp_path / "blob"
    (root / "stuff").mkdir(parents=True)
    info = rl.detect_root(str(root), explicit=True)
    assert info == rl.RootfsInfo(os.path.realpath(root), "unknown", [])
    # explicit does not walk up
    inner = _linux_rootfs(tmp_path) / "usr"
    info2 = rl.detect_root(str(inner), explicit=True)
    assert info2 is not None and info2.platform == "unknown"


def test_detect_non_directory_is_none(tmp_path: Path) -> None:
    f = _touch(tmp_path, "file")
    assert rl.detect_root(str(f)) is None
    assert rl.detect_root(str(f), explicit=True) is None
    assert rl.detect_root(str(tmp_path / "missing")) is None


# ---------------------------------------------------------------------------
# project_location
# ---------------------------------------------------------------------------


def test_project_location(tmp_path: Path) -> None:
    root = tmp_path / "fw" / "squashfs-root"
    root.mkdir(parents=True)
    preferred, fallback = rl.project_location(str(root), "/cache/projects")
    real = os.path.realpath(root)
    assert preferred == os.path.join(os.path.dirname(real), "squashfs-root.bnpr")
    digest = hashlib.sha256(real.encode()).hexdigest()[:16]
    assert fallback == f"/cache/projects/{digest}.bnpr"


def test_project_location_canonicalises_root(tmp_path: Path) -> None:
    root = tmp_path / "fw"
    root.mkdir()
    link = tmp_path / "fw-link"
    os.symlink(root, link)
    assert rl.project_location(str(link), "/c") == rl.project_location(str(root), "/c")
    assert rl.project_location(str(root) + "/", "/c") == rl.project_location(str(root), "/c")


# ---------------------------------------------------------------------------
# lookup paths
# ---------------------------------------------------------------------------


def test_lookup_paths_linux_order(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    (root / "opt" / "extra" / "lib").mkdir(parents=True)
    (root / "usr" / "local" / "lib").mkdir(parents=True)
    lps = rl.lookup_paths(str(root), "linux")
    assert _paths(lps) == ["/opt/extra/lib", "/usr/local/lib", "/opt/vendor/lib", "/lib", "/usr/lib"]
    assert [lp.source for lp in lps] == [
        "ld_library_path:/etc/profile",
        "ld_so_conf:/etc/ld.so.conf",
        "ld_so_conf:/etc/ld.so.conf.d/vendor.conf",
        "default",
        "default",
    ]


def test_lookup_paths_drops_missing_dirs_and_dedups(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    _touch(root, "etc/ld.so.conf", b"/lib\n/lib/\n/missing\ninclude /etc/ld.so.conf.d/*.conf\n")
    lps = rl.lookup_paths(str(root), "linux")
    assert _paths(lps) == ["/lib", "/opt/vendor/lib", "/usr/lib"]
    assert lps[0].source == "ld_so_conf:/etc/ld.so.conf"


def test_lookup_paths_include_relative_and_cycle(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    (root / "opt" / "a").mkdir()
    (root / "opt" / "b").mkdir()
    _touch(root, "etc/ld.so.conf", b"include ld.so.conf.d/*.conf\ninclude /etc/ld.so.conf\n")
    _touch(root, "etc/ld.so.conf.d/a.conf", b"/opt/a\ninclude /etc/ld.so.conf.d/b.conf\n")
    _touch(root, "etc/ld.so.conf.d/b.conf", b"include ../ld.so.conf\n/opt/b\n")
    (root / "etc" / "ld.so.conf.d" / "vendor.conf").unlink()
    lps = rl.lookup_paths(str(root), "linux")
    assert _paths(lps)[:2] == ["/opt/a", "/opt/b"]
    assert lps[0].source == "ld_so_conf:/etc/ld.so.conf.d/a.conf"
    assert lps[1].source == "ld_so_conf:/etc/ld.so.conf.d/b.conf"


def test_lookup_paths_conf_line_variants(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    for d in ("opt/x", "opt/y", "opt/z"):
        (root / d).mkdir(parents=True)
    _touch(root, "etc/ld.so.conf", b"/opt/x:/opt/y   # comment\nhwcap 0 nosegneg\n  /opt/z,/lib\n")
    lps = rl.lookup_paths(str(root), "linux")
    assert _paths(lps) == ["/opt/x", "/opt/y", "/opt/z", "/lib", "/usr/lib"]


def test_lookup_paths_ld_library_path_scrape(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    for d in ("opt/extra/lib", "opt/p1", "opt/rc", "opt/rcd", "opt/env", "opt/pd"):
        (root / d).mkdir(parents=True)
    _touch(root, "etc/init.d/rcS", b"#!/bin/sh\nLD_LIBRARY_PATH='/opt/p1:relative/lib::$LD_LIBRARY_PATH'; export LD_LIBRARY_PATH\n")
    _touch(root, "etc/rc.local", b'export LD_LIBRARY_PATH="/opt/rc"\n')
    _touch(root, "etc/rc.d/init.d/S10", b"LD_LIBRARY_PATH=/opt/rcd:/opt/p1\n")
    _touch(root, "etc/environment", b"LD_LIBRARY_PATH=/opt/env\n")
    _touch(root, "etc/profile.d/x.sh", b"export LD_LIBRARY_PATH=/opt/pd\n")
    _touch(root, "etc/profile.d/ignored.txt", b"LD_LIBRARY_PATH=/opt/z\n")
    lps = rl.lookup_paths(str(root), "linux")
    ld = [(lp.path_in_root, lp.source) for lp in lps if lp.source.startswith("ld_library_path")]
    assert ld == [
        ("/opt/extra/lib", "ld_library_path:/etc/profile"),
        ("/opt/pd", "ld_library_path:/etc/profile.d/x.sh"),
        ("/opt/env", "ld_library_path:/etc/environment"),
        ("/opt/p1", "ld_library_path:/etc/init.d/rcS"),
        ("/opt/rc", "ld_library_path:/etc/rc.local"),
        ("/opt/rcd", "ld_library_path:/etc/rc.d/init.d/S10"),
    ]
    assert rl.env_hint(str(root)) == [
        ("/etc/profile", "/opt/extra/lib:$LD_LIBRARY_PATH"),
        ("/etc/profile.d/x.sh", "/opt/pd"),
        ("/etc/environment", "/opt/env"),
        ("/etc/init.d/rcS", "/opt/p1:relative/lib::$LD_LIBRARY_PATH"),
        ("/etc/rc.local", "/opt/rc"),
        ("/etc/rc.d/init.d/S10", "/opt/rcd:/opt/p1"),
    ]


def test_lookup_paths_multiarch(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    (root / "lib" / "aarch64-linux-gnu").mkdir()
    (root / "usr" / "lib" / "arm-linux-gnueabihf").mkdir()
    (root / "usr" / "lib64").mkdir()
    lps = rl.lookup_paths(str(root), "linux")
    assert _paths(lps) == [
        "/opt/vendor/lib", "/lib", "/usr/lib", "/usr/lib64",
        "/lib/aarch64-linux-gnu", "/usr/lib/arm-linux-gnueabihf",
    ]
    assert all(lp.source == "default" for lp in lps if "linux-" in lp.path_in_root)


def test_lookup_paths_android(tmp_path: Path) -> None:
    root = tmp_path / "android"
    for d in ("system/lib64", "system/lib", "system/bin", "system/lib/vndk-29",
              "apex/com.android.runtime/lib64", "vendor/lib", "usr/lib", "odm/lib"):
        (root / d).mkdir(parents=True)
    lps = rl.lookup_paths(str(root), "android")
    assert _paths(lps) == [
        "/system/lib64", "/system/lib", "/vendor/lib", "/system/lib/vndk-29",
        "/apex/com.android.runtime/lib64", "/odm/lib", "/usr/lib",
    ]
    assert [lp.source for lp in lps[:-1]] == ["android_default"] * 6
    assert lps[-1].source == "default"


def test_lookup_paths_unknown_platform(tmp_path: Path) -> None:
    assert rl.lookup_paths(str(_linux_rootfs(tmp_path)), "unknown") == []


def test_lookup_paths_symlinked_dir_keeps_spelling(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    (root / "lib64").mkdir()
    os.symlink("../lib64", root / "usr" / "lib64")
    lps = rl.lookup_paths(str(root), "linux")
    assert "/usr/lib64" in _paths(lps)


# ---------------------------------------------------------------------------
# scan_libraries
# ---------------------------------------------------------------------------


def test_scan_libraries_groups_symlinks(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    lps = rl.lookup_paths(str(root), "linux")
    cands, notes = rl.scan_libraries(str(root), lps)
    assert notes == []
    assert [c.rel_realpath for c in cands] == ["/opt/vendor/lib/libgreet.so.1", "/lib/libc.so"]
    greet, libc = cands
    assert greet.names == ["libgreet.so.1", "libgreet.so"]
    assert greet.dir_in_root == "/opt/vendor/lib"
    assert greet.realpath == os.path.realpath(root / "opt/vendor/lib/libgreet.so.1")
    assert greet.sha256 == hashlib.sha256(ELF_STUB + b"greet").hexdigest()
    assert greet.size == len(ELF_STUB) + 5
    assert greet.mtime == os.stat(root / "opt/vendor/lib/libgreet.so.1").st_mtime
    assert libc.names == ["libc.so", "ld-musl-aarch64.so.1"]
    assert libc.sha256 == hashlib.sha256(ELF_STUB).hexdigest()


def test_scan_libraries_excludes_stray_non_elf_dirs_and_escapes(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    _touch(root, "lib/libtext.so", b"not an elf\n")
    _touch(root, "lib/libfoo.so.1.2.3", ELF_STUB + b"foo")
    _touch(root, "lib/libfoo.a", ELF_STUB)
    _touch(root, "lib/modules.so.txt", b"\x7fELF?")  # name *.so.* but we still need ELF: it is
    (root / "lib" / "dir.so").mkdir()
    _elf(root, "lib/sub/libdeep.so")
    outside = _touch(tmp_path, "outside/libescape.so", ELF_STUB)
    _link(root, "lib/libescape.so", os.path.relpath(outside, root / "lib"))
    _link(root, "lib/libdangling.so", "nowhere.so")
    lps = rl.lookup_paths(str(root), "linux")
    cands, notes = rl.scan_libraries(str(root), lps)
    rels = [c.rel_realpath for c in cands]
    assert "/lib/libfoo.so.1.2.3" in rels
    assert "/lib/modules.so.txt" in rels
    assert "/lib/libtext.so" not in rels
    assert "/lib/libfoo.a" not in rels
    assert "/lib/dir.so" not in rels
    assert "/lib/sub/libdeep.so" not in rels
    assert "/usr/bin/libstray.so" not in rels
    assert not any("escape" in r or "dangling" in r for r in rels)
    assert len(notes) == 1
    assert notes[0].startswith("skipped /lib/libescape.so: resolves outside the root")


def test_scan_libraries_real_file_seen_only_through_symlink(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    _elf(root, "usr/lib/real/libz.so.1.2.11")
    _link(root, "usr/lib/libz.so.1", "real/libz.so.1.2.11")
    _link(root, "usr/lib/libz.so", "libz.so.1")
    lps = rl.lookup_paths(str(root), "linux")
    cands, _ = rl.scan_libraries(str(root), lps)
    z = [c for c in cands if c.rel_realpath == "/usr/lib/real/libz.so.1.2.11"]
    assert len(z) == 1
    assert z[0].names == ["libz.so.1.2.11", "libz.so", "libz.so.1"]
    assert z[0].dir_in_root == "/usr/lib/real"
    assert z[0].paths == ["/usr/lib/libz.so", "/usr/lib/libz.so.1"]


def test_scan_libraries_missing_dir_is_ignored(tmp_path: Path) -> None:
    root = _linux_rootfs(tmp_path)
    cands, notes = rl.scan_libraries(str(root), [rl.LookupPath("/nope", "default")])
    assert cands == [] and notes == []



def test_absolute_firmware_file_and_directory_aliases(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    real = _elf(root, "usr/lib/libsample.so.1")
    _link(root, "lib", "/usr/lib")
    _link(root, "usr/lib/libsample.so", "/lib/libsample.so.1")
    _link(root, "lib64", "lib")

    paths = rl.lookup_paths(str(root), "linux")
    assert _paths(paths) == ["/lib", "/usr/lib", "/lib64"]
    candidates, notes = rl.scan_libraries(str(root), paths + paths)
    assert notes == []
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.realpath == str(real)
    assert candidate.rel_realpath == "/usr/lib/libsample.so.1"
    assert candidate.dir_in_root == "/usr/lib"
    assert candidate.names == ["libsample.so.1", "libsample.so"]
    assert candidate.paths == [
        "/lib/libsample.so", "/lib/libsample.so.1",
        "/usr/lib/libsample.so", "/usr/lib/libsample.so.1",
        "/lib64/libsample.so", "/lib64/libsample.so.1",
    ]


def test_config_absolute_links_and_directory_globs(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    for rel in ("opt/from-conf", "opt/from-profile", "opt/from-include"):
        (root / rel).mkdir(parents=True)
    _touch(root, "config/loader.conf",
           b"/opt/from-conf\ninclude /etc/ld.so.conf.d/*.conf\n")
    _touch(root, "config/fragments/vendor.conf",
           b"/opt/from-include\ninclude /etc/ld.so.conf\n")
    _touch(root, "config/startup/profile.sh",
           b"LD_LIBRARY_PATH=/opt/from-profile\n")
    _link(root, "etc/ld.so.conf", "/config/loader.conf")
    _link(root, "etc/ld.so.conf.d", "/config/fragments")
    _link(root, "etc/profile.d", "/config/startup")

    paths = rl.lookup_paths(str(root), "linux")
    assert _paths(paths) == ["/opt/from-profile", "/opt/from-conf", "/opt/from-include"]
    assert [path.source for path in paths] == [
        "ld_library_path:/etc/profile.d/profile.sh",
        "ld_so_conf:/etc/ld.so.conf",
        "ld_so_conf:/etc/ld.so.conf.d/vendor.conf",
    ]


def test_config_and_scrape_relative_escapes_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    (root / "opt/host-only").mkdir(parents=True)
    _touch(tmp_path, "outside.conf", b"/opt/host-only\n")
    _touch(tmp_path, "startup/profile.sh", b"LD_LIBRARY_PATH=/opt/host-only\n")
    _link(root, "etc/ld.so.conf", "../../outside.conf")
    _link(root, "etc/profile.d", "../../startup")
    assert rl.lookup_paths(str(root), "linux") == []
    assert rl.env_hint(str(root)) == []

    (root / "etc/ld.so.conf").unlink()
    _touch(root, "etc/ld.so.conf",
           b"include ../../outside.conf\ninclude /etc/conf.d/*.conf\n")
    _link(root, "etc/conf.d", "../../")
    assert rl.lookup_paths(str(root), "linux") == []


def test_lookup_and_scan_reject_relative_directory_escapes(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    _elf(tmp_path, "outside/libhost.so")
    _link(root, "lib", "../outside")
    _touch(root, "etc/profile", b"LD_LIBRARY_PATH=/../outside\n")
    assert rl.lookup_paths(str(root), "linux") == []
    assert rl.scan_libraries(
        str(root), [rl.LookupPath("/lib", "default"), rl.LookupPath("/../outside", "default")]
    ) == ([], [])


def test_absolute_host_spelling_is_rebased_not_opened(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    outside = _elf(tmp_path, "outside/libhost.so")
    rebased = _elf(root, str(outside).lstrip("/"), ELF_STUB + b"firmware")
    _link(root, "lib/libalias.so", str(outside))
    candidates, notes = rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")])
    assert notes == []
    assert len(candidates) == 1
    assert candidates[0].realpath == str(rebased)
    assert candidates[0].sha256 == hashlib.sha256(ELF_STUB + b"firmware").hexdigest()
    assert candidates[0].names == ["libhost.so", "libalias.so"]


def test_symlink_cycles_and_recursive_glob_backedges(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    _touch(root, "etc/rc.d/start", b"LD_LIBRARY_PATH=/opt/good\n")
    (root / "opt/good").mkdir(parents=True)
    _link(root, "etc/rc.d/back", "/etc/rc.d")
    _link(root, "etc/rc.d/self", "self")
    _link(root, "etc/profile", "/etc/environment")
    _link(root, "etc/environment", "profile")
    _link(root, "etc/ld.so.conf", "/etc/ld.so.conf")
    _link(root, "lib", "/lib64")
    _link(root, "lib64", "lib")
    _link(root, "usr/lib/libloop.so", "/usr/lib/libloop.so.1")
    _link(root, "usr/lib/libloop.so.1", "libloop.so")
    _elf(root, "usr/lib/libgood.so")

    assert rl.env_hint(str(root)) == [("/etc/rc.d/start", "/opt/good")]
    paths = rl.lookup_paths(str(root), "linux")
    assert _paths(paths) == ["/opt/good", "/usr/lib"]
    candidates, notes = rl.scan_libraries(str(root), paths)
    assert notes == []
    assert [candidate.rel_realpath for candidate in candidates] == ["/usr/lib/libgood.so"]


def test_symlink_resolution_has_a_finite_hop_limit(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "data/real")
    for index in range(41):
        target = f"/links/{index + 1}" if index < 40 else "/data/real"
        _link(root, f"links/{index}", target)
    _link(root, "lib/liblong.so", "/links/0")
    assert rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")]) == ([], [])


def test_detector_uses_firmware_absolute_links(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    (root / "payload/lib").mkdir(parents=True)
    (root / "payload/bin").mkdir()
    _link(root, "system", "/payload")
    info = rl.detect_root(str(root))
    assert info == rl.RootfsInfo(str(root), "android", ["system/lib", "system/bin"])

    linux = tmp_path / "linux"
    (linux / "bin").mkdir(parents=True)
    (linux / "etc").mkdir()
    _elf(linux, "payload/libc.so.6")
    _link(linux, "lib", "/payload")
    linux_info = rl.detect_root(str(linux))
    assert linux_info is not None and linux_info.platform == "linux"
    assert "lib*/libc.so*" in linux_info.markers


def test_detector_skips_escaped_and_looping_markers(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    (root / "bin").mkdir(parents=True)
    (root / "etc").mkdir()
    _touch(tmp_path, "outside.conf")
    _touch(tmp_path, "outside-system/build.prop")
    _link(root, "etc/ld.so.conf", "../../outside.conf")
    _link(root, "system", "../outside-system")
    _link(root, "init", "/linuxrc")
    _link(root, "linuxrc", "init")
    assert rl.detect_root(str(root), explicit=True) == rl.RootfsInfo(str(root), "unknown", [])


def test_implicit_detection_never_scores_host_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def host_only_score(root: str) -> rl.RootfsInfo | None:
        calls.append(root)
        return rl.RootfsInfo(root, "linux", ["host"]) if root == "/" else None

    monkeypatch.setattr(rl, "_score", host_only_score)
    assert rl.detect_root(str(tmp_path)) is None
    assert "/" not in calls
    assert rl.detect_root("/") is None
    assert "/" not in calls
    assert rl.detect_root("/", explicit=True) == rl.RootfsInfo("/", "linux", ["host"])
    assert calls[-1] == "/"


@pytest.mark.parametrize(
    ("start", "candidate"),
    [("/usr", "/usr"), ("/usr/share/doc", "/usr"), ("/dev", "/dev")],
)
def test_implicit_detection_rejects_host_direct_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str, candidate: str,
) -> None:
    if not os.path.isdir(start):
        pytest.skip(f"host does not provide {start}")
    firmware = rl.detect_root(str(_linux_rootfs(tmp_path)))
    assert firmware is not None and firmware.platform == "linux"
    original_score = rl._score
    host_info = rl.RootfsInfo(candidate, firmware.platform, firmware.markers)

    def score_with_host_markers(root: str) -> rl.RootfsInfo | None:
        # Host marker availability differs across Linux and macOS installations.
        return host_info if root == candidate else original_score(root)

    monkeypatch.setattr(rl, "_score", score_with_host_markers)
    assert rl.detect_root(start) is None
    assert rl.detect_root(candidate, explicit=True) == host_info


@pytest.mark.parametrize("relative_start", ["", "share/doc"])
def test_implicit_detection_rejects_real_symlink_to_host_usr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_start: str,
) -> None:
    alias = _link(tmp_path, "host-usr", "/usr")
    start = alias / relative_start
    if not start.is_dir():
        pytest.skip(f"host does not provide /usr/{relative_start}")
    original_score = rl._score

    def score_with_host_markers(root: str) -> rl.RootfsInfo | None:
        if root == "/usr":
            return rl.RootfsInfo(root, "linux", ["bin", "lib", "lib*/libc.so*"])
        return original_score(root)

    monkeypatch.setattr(rl, "_score", score_with_host_markers)
    assert rl.detect_root(str(start)) is None


@pytest.mark.parametrize("identity", ["same", "different-device", "different-inode"])
@pytest.mark.parametrize("explicit", [False, True])
def test_host_usr_identity_requires_both_device_and_inode_unless_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str, explicit: bool,
) -> None:
    root = _linux_rootfs(tmp_path)
    nested = root / "usr/share/doc"
    nested.mkdir(parents=True)
    canonical_root = os.path.realpath(root)
    expected = rl.detect_root(canonical_root)
    assert expected is not None and expected.platform == "linux"
    original_stat = os.stat
    host_usr = original_stat("/usr")

    def stat_with_mount_identity(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == canonical_root:
            fields = list(result)
            fields[1] = host_usr.st_ino + (identity == "different-inode")
            fields[2] = host_usr.st_dev + (identity == "different-device")
            return os.stat_result(fields)
        return result

    # A bind mount retains the target's device/inode without resolving as a
    # symlink. Keep all filesystem contents and every other stat call real.
    monkeypatch.setattr(rl.os, "stat", stat_with_mount_identity)
    wanted = None if identity == "same" and not explicit else expected
    assert rl.detect_root(canonical_root, explicit=explicit) == wanted
    if not explicit:
        assert rl.detect_root(str(nested)) == wanted


def test_nested_extracted_firmware_is_detected_inside_host_usr_mount_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = _linux_rootfs(tmp_path)
    root = _linux_rootfs(alias / "extracted")
    nested = root / "usr/share/doc"
    nested.mkdir(parents=True)
    canonical_alias = os.path.realpath(alias)
    expected = rl.detect_root(str(root))
    assert expected is not None and expected.root == os.path.realpath(root)
    original_stat = os.stat
    host_usr = original_stat("/usr")

    def stat_with_mount_identity(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == canonical_alias:
            fields = list(result)
            fields[1:3] = [host_usr.st_ino, host_usr.st_dev]
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(rl.os, "stat", stat_with_mount_identity)
    assert rl.detect_root(canonical_alias) is None
    assert rl.detect_root(str(nested)) == expected


def test_recursive_startup_alias_dag_shares_work_and_deduplicates_files(
    tmp_path: Path,
) -> None:
    root = _linux_rootfs(tmp_path)
    for index in range(18):
        for alias in ("a", "b"):
            _link(root, f"startup/{index}/{alias}", f"/startup/{index + 1}")
    _touch(root, "startup/18/start", b"LD_LIBRARY_PATH=/opt/vendor/lib\n")
    _link(root, "etc/rc.d/entry", "/startup/0")
    _link(root, "etc/environment", "/startup/18/start")
    _link(root, "etc/init.d/duplicate", "/startup/18/start")

    # There are exponentially many spellings, but only nineteen real nodes.
    with rl.discovery_budget(rl.DiscoveryBudget(max_work=30_000, timeout=5)):
        assert rl.env_hint(str(root)) == [
            ("/etc/profile", "/opt/extra/lib:$LD_LIBRARY_PATH"),
            ("/etc/environment", "/opt/vendor/lib"),
        ]
        assert _paths(rl.lookup_paths(str(root), "linux")) == [
            "/opt/vendor/lib", "/lib", "/usr/lib",
        ]


def test_embedded_nul_loader_entries_preserve_valid_and_default_paths(
    tmp_path: Path,
) -> None:
    root = _linux_rootfs(tmp_path)
    for directory in ("opt/env-before", "opt/env-after", "opt/conf", "opt/included"):
        (root / directory).mkdir(parents=True)
    _touch(
        root, "etc/profile",
        b"LD_LIBRARY_PATH=/opt/env-before:/opt/bad\0path:/opt/env-after\n",
    )
    _touch(
        root, "etc/ld.so.conf",
        b"/opt/bad\0path /opt/conf\n"
        b"include /etc/bad\0/*.conf /etc/ld.so.conf.d/*.conf\n",
    )
    _touch(
        root, "etc/ld.so.conf.d/vendor.conf",
        b"include bad\0.conf\n/opt/included\n",
    )

    assert _paths(rl.lookup_paths(str(root), "linux")) == [
        "/opt/env-before", "/opt/env-after", "/opt/conf", "/opt/included",
        "/lib", "/usr/lib",
    ]


def test_discovery_byte_budget_is_shared_between_lookup_and_scan(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    profile = b"LD_LIBRARY_PATH=/lib\n"
    payload = ELF_STUB + b"library payload" * 8
    _touch(root, "etc/profile", profile)
    _elf(root, "lib/libgood.so", payload)

    # Each operation fits separately; their combined bytes must not reset.
    with rl.discovery_budget(rl.DiscoveryBudget(max_bytes=len(payload))):
        paths = rl.lookup_paths(str(root), "linux")
        assert _paths(paths) == ["/lib"]
    with rl.discovery_budget(rl.DiscoveryBudget(max_bytes=len(payload))):
        candidates, notes = rl.scan_libraries(str(root), paths)
        assert [candidate.rel_realpath for candidate in candidates] == ["/lib/libgood.so"]
        assert notes == []

    with rl.discovery_budget(rl.DiscoveryBudget(max_bytes=len(payload))):
        paths = rl.lookup_paths(str(root), "linux")
        with pytest.raises(RuntimeError, match="rootfs discovery refused"):
            rl.scan_libraries(str(root), paths)


def test_scan_byte_refusal_does_not_return_an_already_discovered_subset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/liba.so")
    _elf(root, "lib/libz.so", ELF_STUB + b"later candidate")

    with rl.discovery_budget(rl.DiscoveryBudget(max_bytes=len(ELF_STUB))):
        with pytest.raises(RuntimeError, match="rootfs discovery refused"):
            rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")])


def test_discovery_work_budget_is_shared_between_calls(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/libgood.so")
    paths = [rl.LookupPath("/lib", "default")]

    with rl.discovery_budget(rl.DiscoveryBudget(max_work=500, timeout=30)):
        candidates, notes = rl.scan_libraries(str(root), paths)
        assert [candidate.rel_realpath for candidate in candidates] == ["/lib/libgood.so"]
        assert notes == []
        # Repeated individually small calls must eventually exhaust one budget.
        with pytest.raises(RuntimeError, match="rootfs discovery refused"):
            for _ in range(501):
                rl.scan_libraries(str(root), paths)


def test_work_is_charged_per_file_and_per_mib_never_per_check_1075(tmp_path: Path) -> None:
    """#1075 item 2: `check()` is a probe (cancellation, deadline, the limits
    as charged so far); only `charge()` spends units, one per file opened and
    one per MiB read, so a file's entry/line/symbol count costs nothing."""
    budget = rl.DiscoveryBudget(max_work=3, timeout=30)
    for _ in range(10_000):
        budget.check()
    assert budget.work_used == 0
    budget.charge(files=1)
    budget.charge(bytes_read=(2 << 20) - 1)  # 1.99 MiB rounds down
    assert budget.work_used == 2
    budget.charge(bytes_read=1)  # ... and the next byte completes the MiB
    assert budget.work_used == 3
    with pytest.raises(RuntimeError) as raised:
        budget.charge(files=1)
    message = str(raised.value)
    assert message.startswith("rootfs discovery refused: work limit exceeded")
    # The budget stops at the first unit over the limit, so the number it can
    # state is the limit, not the tree's need (#1075 review).
    assert "used more than 3 units" in message and "BN_ROOTFS_MAX_WORK=3" in message
    assert "charged per file opened and per MiB read" in message
    assert "no inventory committed" in message
    assert "session restart" not in message, "no restart advice without a restart command"

    with_restart = rl.DiscoveryBudget(max_work=3, timeout=30, restart="bn session restart w1")
    with pytest.raises(RuntimeError) as raised:
        with_restart.charge(files=4)
    assert str(raised.value).endswith(
        "The owning bridge reads BN_ROOTFS_MAX_WORK at start, so set it and restart; the "
        "total needed is not known, so doubling is only a starting point: "
        "BN_ROOTFS_MAX_WORK=6 bn session restart w1")


def test_deadline_and_byte_refusals_name_their_knob_and_the_restart_1075(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1075 review: only the work refusal named its variable; a deadline or
    byte refusal left the same trap (the CLI's environment is ignored)."""
    too_many = rl.DiscoveryBudget(max_bytes=8, timeout=30, restart="bn session restart w1")
    with pytest.raises(RuntimeError) as raised:
        too_many.charge(bytes_read=9)
    message = str(raised.value)
    assert message.startswith("rootfs discovery refused: discovery byte limit exceeded "
                              "(read more than 8 bytes, BN_ROOTFS_MAX_BYTES=8)")
    assert message.endswith("BN_ROOTFS_MAX_BYTES=16 bn session restart w1")

    late = rl.DiscoveryBudget(timeout=1.5, restart="bn session restart w1")
    expired = rl.time.monotonic() + 5
    monkeypatch.setattr(rl.time, "monotonic", lambda: expired)
    with pytest.raises(RuntimeError) as raised:
        late.check()
    message = str(raised.value)
    assert message.startswith("rootfs discovery refused: discovery deadline exceeded "
                              "(BN_ROOTFS_TIMEOUT=1.5 seconds)")
    assert "reads BN_ROOTFS_TIMEOUT at start" in message
    assert message.endswith("BN_ROOTFS_TIMEOUT=3 bn session restart w1")


def test_directory_listings_of_the_startup_glob_are_charged_1075(tmp_path: Path) -> None:
    """#1075 review: the per-listing charge in the glob walk is the only
    non-deadline bound on `etc/rc.d/**` and wildcard `include` patterns. A deep
    tree of EMPTY directories holds no file to read, so every unit spent here
    is a directory listed."""
    root = _linux_rootfs(tmp_path)
    (root / "etc/profile").unlink()
    deep = root / "etc/rc.d"
    for index in range(40):
        deep = deep / f"d{index}"
    deep.mkdir(parents=True)

    budget = rl.DiscoveryBudget(max_work=1_000, timeout=30)
    with rl.discovery_budget(budget):
        assert rl.env_hint(str(root)) == []
    assert budget.work_used >= 40, budget.work_used

    with rl.discovery_budget(rl.DiscoveryBudget(max_work=20, timeout=30)):
        with pytest.raises(RuntimeError, match="work limit exceeded"):
            rl.env_hint(str(root))


def test_scan_charges_one_unit_per_file_plus_its_mib_regardless_of_entry_count_1075(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rootfs"
    for index in range(40):
        _elf(root, f"lib/lib{index:02}.so")
    big = _elf(root, "lib/libbig.so", ELF_STUB + b"\0" * (3 << 20))
    paths = [rl.LookupPath("/lib", "default")]

    budget = rl.DiscoveryBudget(max_work=100, timeout=30)
    with rl.discovery_budget(budget):
        candidates, _notes = rl.scan_libraries(str(root), paths)
    assert len(candidates) == 41
    # One directory listed, 41 libraries hashed, 3 whole MiB read from `big`.
    assert budget.work_used == 1 + 41 + (big.stat().st_size >> 20)

    # Forty small libraries cost forty units however many entries the
    # directory holds; the default budget (100k) is nowhere near.
    with rl.discovery_budget(rl.DiscoveryBudget(max_work=1 + 41 + 2, timeout=30)):
        with pytest.raises(RuntimeError, match="work limit exceeded"):
            rl.scan_libraries(str(root), paths)


@pytest.mark.parametrize("operation", ["env_hint", "lookup_paths", "scan_libraries"])
@pytest.mark.parametrize("stop", ["cancelled", "deadline"])
def test_discovery_checks_cancellation_and_deadline_between_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, stop: str,
) -> None:
    root = _linux_rootfs(tmp_path)
    cancelled = Event()
    budget = rl.DiscoveryBudget(timeout=30, cancelled=cancelled.is_set)
    expired_time = rl.time.monotonic() + 60
    operations = {
        "env_hint": lambda: rl.env_hint(str(root)),
        "lookup_paths": lambda: rl.lookup_paths(str(root), "linux"),
        "scan_libraries": lambda: rl.scan_libraries(
            str(root), [rl.LookupPath("/lib", "default")],
        ),
    }

    with rl.discovery_budget(budget):
        assert operations[operation]()
        if stop == "cancelled":
            cancelled.set()
        else:
            monkeypatch.setattr(rl.time, "monotonic", lambda: expired_time)
        with pytest.raises(RuntimeError, match=f"rootfs discovery refused: .*{stop}"):
            operations[operation]()


_DiscoveryResult = TypeVar("_DiscoveryResult")


def _run_fifo_discovery(
    fifo: Path, operation: Callable[[], _DiscoveryResult],
) -> _DiscoveryResult:
    results: list[_DiscoveryResult] = []
    errors: list[BaseException] = []

    def discover() -> None:
        try:
            results.append(operation())
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=discover, daemon=True)
    worker.start()
    worker.join(timeout=2)
    blocked = worker.is_alive()
    if blocked:
        # Release a regressed blocking read-open, then provide EOF without ever
        # blocking the test runner on a FIFO writer that has no reader.
        fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        os.close(fd)
        worker.join(timeout=2)
    assert not worker.is_alive(), "FIFO discovery worker did not stop after cleanup"
    assert not blocked, "rootfs discovery blocked on a FIFO"
    if errors:
        raise errors[0]
    return results[0]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_scan_rejects_regular_candidate_replaced_by_fifo_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "rootfs"
    candidate = _elf(root, "lib/liba-raced.so")
    _elf(root, "lib/libz-good.so")
    original_open = os.open
    replaced = Event()

    def replace_before_open(path, flags, *args, **kwargs):
        if os.fspath(path) == str(candidate) and not replaced.is_set():
            candidate.unlink()
            os.mkfifo(candidate)
            replaced.set()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rl.os, "open", replace_before_open)
    candidates, notes = _run_fifo_discovery(
        candidate,
        lambda: rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")]),
    )

    assert replaced.is_set(), "the candidate must reach the stat/open race"
    assert [item.rel_realpath for item in candidates] == ["/lib/libz-good.so"]
    assert any(note.startswith("skipped /lib/liba-raced.so:") for note in notes)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
@pytest.mark.parametrize(
    "fifo_rel", ["etc/ld.so.conf", "etc/ld.so.conf.d/pipe.conf", "etc/environment"],
)
def test_fifo_loader_configs_do_not_block_or_hide_valid_paths(
    tmp_path: Path, fifo_rel: str,
) -> None:
    root = _linux_rootfs(tmp_path)
    _touch(root, "etc/profile", b"LD_LIBRARY_PATH=/opt/vendor/lib\n")
    fifo = _touch(root, fifo_rel)
    fifo.unlink()
    os.mkfifo(fifo)

    paths = _run_fifo_discovery(
        fifo, lambda: rl.lookup_paths(str(root), "linux"),
    )
    assert _paths(paths) == ["/opt/vendor/lib", "/lib", "/usr/lib"]


@pytest.mark.parametrize("stop", ["cancelled", "deadline"])
def test_scan_propagates_mid_inventory_cancellation_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/liba.so")
    later = _elf(root, "lib/libz.so")
    cancelled = Event()
    reached_later = Event()
    budget = rl.DiscoveryBudget(timeout=30, cancelled=cancelled.is_set)
    expired_time = rl.time.monotonic() + 60
    original_open = os.open

    def interrupt_later_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if os.fspath(path) == str(later):
            reached_later.set()
            if stop == "cancelled":
                cancelled.set()
            else:
                monkeypatch.setattr(rl.time, "monotonic", lambda: expired_time)
        return fd

    monkeypatch.setattr(rl.os, "open", interrupt_later_open)
    with rl.discovery_budget(budget):
        with pytest.raises(RuntimeError, match=f"rootfs discovery refused: .*{stop}"):
            rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")])
    assert reached_later.is_set(), "refusal must interrupt an in-progress inventory"


def test_lookup_work_refusal_does_not_return_partial_or_default_paths(
    tmp_path: Path,
) -> None:
    root = _linux_rootfs(tmp_path)
    _touch(root, "etc/ld.so.conf", b"include /etc/ld.so.conf.d/*.conf\n")
    for index in range(1_000):
        _touch(root, f"etc/ld.so.conf.d/{index:04}.conf", b"/opt/vendor/lib\n")

    with rl.discovery_budget(rl.DiscoveryBudget(max_work=500, timeout=30)):
        with pytest.raises(RuntimeError, match="rootfs discovery refused"):
            rl.lookup_paths(str(root), "linux")


@pytest.mark.parametrize(
    ("name", "small", "workable"),
    [
        ("BN_ROOTFS_MAX_WORK", "1", "100000"),
        ("BN_ROOTFS_MAX_BYTES", "4", "1048576"),
        ("BN_ROOTFS_TIMEOUT", "1", "30"),
    ],
)
def test_environment_limits_can_be_raised_to_allow_complete_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    name: str, small: str, workable: str,
) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/libgood.so")
    paths = [rl.LookupPath("/lib", "default")]
    for variable in (
        "BN_ROOTFS_MAX_WORK", "BN_ROOTFS_MAX_BYTES", "BN_ROOTFS_TIMEOUT",
        "BN_ROOTFS_MAX_CONFIG_BYTES",
    ):
        monkeypatch.delenv(variable, raising=False)

    for value, refuses in ((small, True), (workable, False)):
        with monkeypatch.context() as limits:
            limits.setenv(name, value)
            budget = rl.DiscoveryBudget()
            if name == "BN_ROOTFS_TIMEOUT":
                elapsed_time = rl.time.monotonic() + 2
                limits.setattr(rl.time, "monotonic", lambda: elapsed_time)
            if refuses:
                with pytest.raises(RuntimeError, match="rootfs discovery refused"):
                    with rl.discovery_budget(budget):
                        rl.scan_libraries(str(root), paths)
            else:
                with rl.discovery_budget(budget):
                    candidates, notes = rl.scan_libraries(str(root), paths)
                assert [item.rel_realpath for item in candidates] == ["/lib/libgood.so"]
                assert notes == []


@pytest.mark.parametrize(
    "name", [
        "BN_ROOTFS_MAX_WORK", "BN_ROOTFS_MAX_BYTES", "BN_ROOTFS_TIMEOUT",
        "BN_ROOTFS_MAX_CONFIG_BYTES",
    ],
)
@pytest.mark.parametrize("value", ["invalid", "", "0", "-1", "NaN", "inf", "-inf"])
def test_invalid_environment_limits_refuse_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str,
) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/libgood.so")
    for variable in (
        "BN_ROOTFS_MAX_WORK", "BN_ROOTFS_MAX_BYTES", "BN_ROOTFS_TIMEOUT",
        "BN_ROOTFS_MAX_CONFIG_BYTES",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=f"{name} must be a positive finite"):
        rl.scan_libraries(str(root), [rl.LookupPath("/lib", "default")])


@pytest.mark.parametrize("limit", ["max_work", "max_bytes", "timeout", "max_text_bytes"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_explicit_discovery_limits_are_rejected(limit: str, value: float) -> None:
    with pytest.raises(ValueError, match="rootfs"):
        rl.DiscoveryBudget(**{limit: value})


@pytest.mark.parametrize(
    "config_rel", ["etc/profile", "etc/ld.so.conf", "etc/ld.so.conf.d/vendor.conf"],
)
@pytest.mark.parametrize("configuration", ["environment", "explicit"])
def test_config_byte_limit_refuses_truncation_and_can_be_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    config_rel: str, configuration: str,
) -> None:
    root = tmp_path / "rootfs"
    _elf(root, "lib/libgood.so")
    (root / "opt/late").mkdir(parents=True)
    if config_rel.startswith("etc/ld.so.conf.d/"):
        _touch(root, "etc/ld.so.conf", b"include /etc/ld.so.conf.d/*.conf\n")
    last_line = (
        b"LD_LIBRARY_PATH=/opt/late\n" if config_rel == "etc/profile" else b"/opt/late\n"
    )
    contents = b"# harmless padding before a required loader path\n" * 8 + last_line
    _touch(root, config_rel, contents)
    for variable in (
        "BN_ROOTFS_MAX_WORK", "BN_ROOTFS_MAX_BYTES", "BN_ROOTFS_TIMEOUT",
        "BN_ROOTFS_MAX_CONFIG_BYTES",
    ):
        monkeypatch.delenv(variable, raising=False)

    def discover(limit: int) -> list[rl.LookupPath]:
        if configuration == "environment":
            monkeypatch.setenv("BN_ROOTFS_MAX_CONFIG_BYTES", str(limit))
            budget = rl.DiscoveryBudget()
        else:
            budget = rl.DiscoveryBudget(max_text_bytes=limit)
        with rl.discovery_budget(budget):
            return rl.lookup_paths(str(root), "linux")

    with pytest.raises(
        RuntimeError,
        match="rootfs discovery refused: loader config .* exceeds the text limit; "
              "no inventory committed",
    ):
        discover(64)
    assert _paths(discover(len(contents))) == ["/opt/late", "/lib"]


def test_android_loader_globs_preserve_all_alias_spellings(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    real = _elf(root, "objects/runtime/lib64/libshared.so")
    _link(root, "apex/a", "/objects/runtime")
    _link(root, "apex/b", "/objects/runtime")

    paths = rl.lookup_paths(str(root), "android")
    assert _paths(paths) == ["/apex/a/lib64", "/apex/b/lib64"]
    candidates, notes = rl.scan_libraries(str(root), paths)

    assert notes == []
    assert len(candidates) == 1
    assert candidates[0].realpath == str(real)
    assert candidates[0].paths == [
        "/apex/a/lib64/libshared.so", "/apex/b/lib64/libshared.so",
    ]
