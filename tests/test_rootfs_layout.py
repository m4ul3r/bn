from __future__ import annotations

import hashlib
import os
from pathlib import Path

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

