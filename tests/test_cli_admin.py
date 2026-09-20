from __future__ import annotations

import json
import os
import types

import bn.cli
import pytest

from _cli_helpers import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def _isolate_omp_skill_destination(monkeypatch, tmp_path):
    config_root = tmp_path / "absent-omp-config"
    agent_dir = config_root / "agent"
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: config_root, raising=False)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: agent_dir, raising=False)
    monkeypatch.setattr(
        bn.cli, "omp_skills_dir", lambda: agent_dir / "skills", raising=False
    )


def test_skills_source_dir_prefers_repo_then_falls_back_to_prefix(monkeypatch, tmp_path):
    # #83: editable checkout uses repo skills/; a wheel install (no repo skills/)
    # falls back to the install prefix where the data files land.
    import sys as _sys

    import bn.paths as paths

    (tmp_path / "skills").mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.skills_source_dir() == tmp_path / "skills"

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: empty)  # no skills/ here
    assert paths.skills_source_dir() == paths.Path(_sys.prefix)


def test_plugin_source_dir_prefers_repo_then_falls_back_to_installed_module(monkeypatch, tmp_path):
    # #83/#406: editable checkout uses repo src/<name>; a wheel install resolves
    # the bridge packaged into site-packages via find_spec.
    import bn.paths as paths

    repo_plugin = tmp_path / "src" / paths.PLUGIN_NAME
    repo_plugin.mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    assert paths.plugin_source_dir() == repo_plugin

    empty = tmp_path / "empty"
    empty.mkdir()
    installed = tmp_path / "site" / paths.PLUGIN_NAME
    installed.mkdir(parents=True)
    fake_spec = types.SimpleNamespace(origin=str(installed / "__init__.py"))
    monkeypatch.setattr(paths, "repo_root", lambda: empty)
    monkeypatch.setattr(paths.importlib.util, "find_spec", lambda name: fake_spec)
    assert paths.plugin_source_dir() == installed


def test_plugin_install_copy_mode(tmp_path):
    destination = tmp_path / "plugin-copy"
    rc = bn.cli.main(
        [
            "plugin",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bridge.py").exists()


def test_skill_install_copy_mode(tmp_path):
    destination = tmp_path / "skill-copy"
    rc = bn.cli.main(
        [
            "skill",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(destination),
        ]
    )
    assert rc == 0
    assert (destination / "bn" / "SKILL.md").exists()
    assert (destination / "bn" / "agents" / "openai.yaml").exists()
    assert (destination / "bn-re" / "SKILL.md").exists()
    assert (destination / "bn-vr" / "SKILL.md").exists()
    assert (destination / "bn-kernel" / "SKILL.md").exists()


def test_skill_install_copy_omits_python_bytecode(tmp_path, monkeypatch):
    source_root = tmp_path / "bundled-skills"
    skill = source_root / "sample"
    cache = skill / "__pycache__"
    cache.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sample\n---\n")
    (skill / "adapter.py").write_text("VALUE = 1\n")
    (cache / "adapter.cpython-314.pyc").write_bytes(b"stale bytecode")
    (skill / "adapter.pyo").write_bytes(b"stale bytecode")
    destination = tmp_path / "installed"
    monkeypatch.setattr(bn.cli, "skills_source_dir", lambda: source_root)

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination)]
    )

    assert rc == 0
    assert (destination / "sample" / "adapter.py").exists()
    assert not (destination / "sample" / "__pycache__").exists()
    assert not (destination / "sample" / "adapter.pyo").exists()


def test_skill_install_defaults_to_claude_only_without_codex_home(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert not codex_root.exists()


def test_skill_install_defaults_to_claude_and_codex_when_codex_home_exists(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (claude_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_defaults_skip_existing_destinations(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_root = codex_home / "skills"
    codex_home.mkdir()
    (claude_root / "bn").mkdir(parents=True)
    (claude_root / "bn-re").mkdir()
    (claude_root / "bn-vr").mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (codex_root / "bn" / "SKILL.md").exists()
    assert (codex_root / "bn-re" / "SKILL.md").exists()
    assert (codex_root / "bn-vr" / "SKILL.md").exists()


def test_skill_install_default_output_is_text(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    output = capsys.readouterr().out
    assert output.startswith("Installed skills (copy):\n")
    assert "- " + str(claude_root / "bn") in output
    assert '"installed"' not in output


def test_skill_install_json_output_remains_available(tmp_path, monkeypatch, capsys):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--format", "json"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"installed":true' in output          # compact json (#215)
    assert '"installed_destinations"' in output


def test_skill_install_all_skipped_reports_installed_false(tmp_path, monkeypatch, capsys):
    # #620(b): when every destination already exists and is skipped, nothing
    # was actually installed -- the result must say so instead of always
    # claiming success.
    source_root = tmp_path / "bundled-skills"
    skill = source_root / "sample"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sample\n---\n")
    monkeypatch.setattr(bn.cli, "skills_source_dir", lambda: source_root)

    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    (claude_root / "sample").mkdir(parents=True)
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--format", "json"])

    assert rc == 0
    output = capsys.readouterr().out
    assert '"installed":false' in output
    assert '"installed_destinations":[]' in output
    assert str(claude_root / "sample") in output  # skipped_destinations explains why


def test_skill_install_custom_dest_still_fails_when_destination_exists(tmp_path):
    destination = tmp_path / "skill-copy"
    (destination / "bn").mkdir(parents=True)

    rc = bn.cli.main(["skill", "install", "--mode", "copy", "--dest", str(destination)])

    assert rc == 2


def test_plugin_install_force_refuses_an_unrelated_destination(tmp_path, capsys):
    # #766: `--dest` is caller-supplied, so --force used to shutil.rmtree
    # whatever was there. It may only replace a destination this install
    # provably owns, and must refuse with the deliberate alternative named.
    destination = tmp_path / "user-data"
    (destination / "sub").mkdir(parents=True)
    (destination / "keepme.txt").write_text("irreplaceable")
    (destination / "sub" / "notes.txt").write_text("also irreplaceable")

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    )

    assert rc == 2
    assert (destination / "keepme.txt").read_text() == "irreplaceable"
    assert (destination / "sub" / "notes.txt").read_text() == "also irreplaceable"
    assert not (destination / "bridge.py").exists()
    assert "Refusing to replace" in capsys.readouterr().err


def test_skill_install_force_refuses_an_unrelated_destination(tmp_path):
    # Same guard, reached through the other `_install_tree` caller.
    destination = tmp_path / "skill-store"
    (destination / "bn").mkdir(parents=True)
    (destination / "bn" / "keepme.txt").write_text("irreplaceable")

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    )

    assert rc == 2
    assert (destination / "bn" / "keepme.txt").read_text() == "irreplaceable"
    assert not (destination / "bn" / "SKILL.md").exists()


def test_force_refuses_a_destination_that_contains_the_cache_root(tmp_path, monkeypatch):
    # Empty, so only the confinement check can refuse it: wiping the cache
    # root's parent takes the instance registry and the sticky pins with it.
    monkeypatch.setenv("BN_CACHE_DIR", str(tmp_path / ".cache" / "bn"))
    destination = tmp_path / ".cache"
    destination.mkdir(parents=True)

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    )

    assert rc == 2
    assert destination.is_dir()
    assert not (destination / "bridge.py").exists()


def test_force_refuses_the_home_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Park every other guarded root outside the home directory, so the home
    # rule itself -- not the root-containment rule -- is what refuses here.
    elsewhere = tmp_path.parent / "guarded-roots"
    monkeypatch.setattr(bn.cli, "cache_home", lambda: elsewhere / "cache", raising=False)
    monkeypatch.setattr(
        bn.cli, "claude_skills_dir", lambda: elsewhere / "claude", raising=False
    )
    monkeypatch.setattr(
        bn.cli, "codex_skills_dir", lambda: elsewhere / "codex", raising=False
    )

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(tmp_path), "--force"]
    )

    assert rc == 2
    assert tmp_path.is_dir()
    assert not (tmp_path / "bridge.py").exists()
    assert "it is your home directory" in capsys.readouterr().err


def test_force_reinstalls_over_this_install_s_own_previous_copy(tmp_path):
    destination = tmp_path / "store" / "bn_agent_bridge"

    assert bn.cli.main(["plugin", "install", "--mode", "copy", "--dest", str(destination)]) == 0
    # Importing the installed plugin in place leaves a bytecode cache the copy
    # never wrote; the destination is still this install's own.
    (destination / "__pycache__").mkdir()
    (destination / "bridge.py").write_text("# PRIOR-INSTALL-MARKER\n")

    assert bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    ) == 0
    assert "PRIOR-INSTALL-MARKER" not in (destination / "bridge.py").read_text()


def test_force_installs_into_an_empty_destination(tmp_path):
    destination = tmp_path / "empty-destination"
    destination.mkdir()

    assert bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    ) == 0
    assert (destination / "bridge.py").exists()


def test_force_unlinks_a_symlink_destination_without_following_it(tmp_path):
    real = tmp_path / "real-directory"
    real.mkdir()
    (real / "keepme.txt").write_text("irreplaceable")
    destination = tmp_path / "linked-destination"
    destination.symlink_to(real, target_is_directory=True)

    assert bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    ) == 0
    assert (real / "keepme.txt").read_text() == "irreplaceable"
    assert not destination.is_symlink()


def test_force_refuses_a_destination_whose_entry_names_only_collide(tmp_path):
    # Matching entry NAMES is not ownership: a user's own file sitting under a
    # name the source also uses does not make the directory this install's.
    source = bn.cli.plugin_source_dir()
    colliding = next(entry.name for entry in sorted(source.iterdir()) if entry.is_file())
    destination = tmp_path / "user-data"
    destination.mkdir()
    (destination / colliding).write_text("the user's own file\n")

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    )

    assert rc == 2
    assert (destination / colliding).read_text() == "the user's own file\n"


def test_force_refuses_a_destination_holding_only_bytecode(tmp_path):
    # Bytecode is tolerated inside a verified install, never evidence of one: a
    # directory whose entries are all bytecode caches is refused, not wiped.
    destination = tmp_path / "bytecode-only"
    (destination / "__pycache__").mkdir(parents=True)
    (destination / "__pycache__" / "stale.pyc").write_bytes(b"user bytecode")
    (destination / "loose.pyc").write_bytes(b"user bytecode")

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(destination), "--force"]
    )

    assert rc == 2
    assert (destination / "loose.pyc").read_bytes() == b"user bytecode"


def test_force_refuses_a_foreign_file_inside_a_source_named_directory(tmp_path):
    # The source's own directory names are not a licence to recurse into them:
    # a file the user added inside one makes the destination not ours.
    skills_root = bn.cli.skills_source_dir()
    skill = next(entry for entry in sorted(skills_root.iterdir()) if entry.is_dir())
    nested = next(entry.name for entry in sorted(skill.iterdir()) if entry.is_dir())
    destination_root = tmp_path / "skill-store"
    destination = destination_root / skill.name
    assert bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination_root)]
    ) == 0
    (destination / nested / "mine.md").write_text("the user's own file\n")

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination_root), "--force"]
    )

    assert rc == 2
    assert (destination / nested / "mine.md").read_text() == "the user's own file\n"


def test_force_refuses_a_file_destination(tmp_path, capsys):
    # A regular file where a directory install belongs is the user's file, not
    # a previous install: --force used to unlink it and copy over the grave.
    victim = tmp_path / "notes.txt"
    victim.write_text("irreplaceable")

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(victim), "--force"]
    )

    assert rc == 2
    assert victim.read_text() == "irreplaceable"
    assert "Refusing to replace" in capsys.readouterr().err


def test_a_refused_skill_install_replaces_no_other_destination(tmp_path):
    # The refusal is decided before anything is written, so one unsafe
    # destination cannot leave the other destinations half-installed.
    skills = sorted(
        entry
        for entry in bn.cli.skills_source_dir().iterdir()
        if (entry / "SKILL.md").exists()
    )
    unsafe = skills[-1]
    destination_root = tmp_path / "skill-store"
    (destination_root / unsafe.name).mkdir(parents=True)
    (destination_root / unsafe.name / "keepme.txt").write_text("irreplaceable")

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(destination_root), "--force"]
    )

    assert rc == 2
    assert (destination_root / unsafe.name / "keepme.txt").read_text() == "irreplaceable"
    assert not (destination_root / skills[0].name).exists()


def test_a_refused_skill_install_writes_nothing_for_a_spelled_destination(tmp_path):
    # A destination spelled through a missing intermediate names the same
    # directory the validated one does, and deciding that must not create the
    # intermediate: the pre-pass used to approve "store/missing/.." (nothing
    # there yet) and the per-destination install then replaced the skills it
    # reached before refusing the unsafe one.
    skills = sorted(
        entry
        for entry in bn.cli.skills_source_dir().iterdir()
        if (entry / "SKILL.md").exists()
    )
    unsafe = skills[-1]
    store = tmp_path / "store"
    (store / unsafe.name).mkdir(parents=True)
    (store / unsafe.name / "keepme.txt").write_text("irreplaceable")

    rc = bn.cli.main(
        [
            "skill",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(store / "missing" / ".."),
            "--force",
        ]
    )

    assert rc == 2
    assert (store / unsafe.name / "keepme.txt").read_text() == "irreplaceable"
    assert not (store / skills[0].name).exists()
    assert not (store / "missing").exists()


def test_install_refuses_a_destination_inside_the_install_source(tmp_path, monkeypatch):
    # Copying the artifact into its own source tree walks the copy being
    # written, so a destination under the source is refused even though there
    # is nothing there to remove yet.
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "bridge.py").write_text("PLUGIN = 1\n")
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source)

    rc = bn.cli.main(
        ["plugin", "install", "--mode", "copy", "--dest", str(source / "nested"), "--force"]
    )

    assert rc == 2
    assert sorted(entry.name for entry in source.iterdir()) == ["bridge.py", "sub"]


def test_install_refuses_a_destination_whose_parent_is_a_file(tmp_path, capsys):
    # A regular file where a parent directory belongs cannot be a destination
    # path; the refusal is a clean exit 2, not a traceback out of main().
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory\n")

    rc = bn.cli.main(
        [
            "plugin",
            "install",
            "--mode",
            "copy",
            "--dest",
            str(blocker / "sub"),
            "--force",
        ]
    )

    assert rc == 2
    assert blocker.read_text() == "not a directory\n"
    assert "is not a directory" in capsys.readouterr().err


def test_force_replaces_a_symlink_install_that_points_at_the_source(tmp_path):
    # The default `--mode symlink` install leaves a link at the destination that
    # points at the install source. Replacing it removes the link, never the
    # source: the link's target must not be read as the thing being replaced,
    # or every re-install of a symlinked install refuses for good.
    destination = tmp_path / "store" / "bn_agent_bridge"

    assert bn.cli.main(["plugin", "install", "--dest", str(destination)]) == 0
    assert destination.is_symlink()

    assert bn.cli.main(
        ["plugin", "install", "--dest", str(destination), "--force"]
    ) == 0
    assert bn.cli.main(
        ["plugin", "install", "--dest", str(destination), "--mode", "copy", "--force"]
    ) == 0
    assert (bn.cli.plugin_source_dir() / "bridge.py").is_file()
    assert not destination.is_symlink()


def test_skill_install_force_replaces_its_own_symlink_installs(tmp_path, monkeypatch):
    claude_root = tmp_path / "claude" / "skills"
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: tmp_path / "codex")

    assert bn.cli.main(["skill", "install"]) == 0
    assert (claude_root / "bn").is_symlink()

    assert bn.cli.main(["skill", "install", "--force"]) == 0
    assert bn.cli.main(["skill", "install", "--mode", "copy", "--force"]) == 0
    assert (claude_root / "bn" / "SKILL.md").is_file()
    assert not (claude_root / "bn").is_symlink()


def test_skill_install_installs_a_shared_default_root_once(tmp_path, monkeypatch):
    # Two default roots can name one directory -- a symlinked skills root, or an
    # equal CLAUDE_HOME and CODEX_HOME. Each skill is installed there once: the
    # second entry used to install nothing, exit 2, and leave the first root
    # populated with only the skills it reached first.
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "skills").symlink_to(claude_root)
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_home / "skills")

    assert bn.cli.main(["skill", "install"]) == 0
    assert (claude_root / "bn" / "SKILL.md").is_file()
    assert (claude_root / "bn-vr" / "SKILL.md").is_file()
    assert bn.cli.main(["skill", "install", "--force"]) == 0


def test_skill_install_is_all_or_nothing_with_an_unwritable_root(tmp_path, monkeypatch):
    # A default root under a directory the caller cannot write is refused in
    # the plan, so the other root is not populated first. A caller who can
    # write there (root) installs normally; either way the first root is never
    # left half-populated.
    claude_root = tmp_path / "claude" / "skills"
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: locked)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: locked / "skills")

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc in (0, 2)
    if rc == 2:
        assert not claude_root.exists()
    else:
        assert (claude_root / "bn-vr" / "SKILL.md").is_file()


def test_a_refused_skill_install_with_an_unusable_root_writes_nothing(tmp_path, monkeypatch):
    # A default root that cannot be created -- a file where the root belongs --
    # has to be refused in the plan: refusing it in the install loop instead
    # leaves the other root already populated with the skills it reached.
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "skills").write_text("not a directory\n")
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "codex_skills_dir", lambda: codex_home / "skills")

    rc = bn.cli.main(["skill", "install"])

    assert rc == 2
    assert not claude_root.exists()
    assert (codex_home / "skills").read_text() == "not a directory\n"


def test_omp_path_resolution_follows_profiles_and_agent_override(monkeypatch, tmp_path):
    import bn.paths as paths

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "PI_CONFIG_DIR",
        "OMP_PROFILE",
        "PI_PROFILE",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    assert paths.omp_config_root() == tmp_path / ".omp"
    assert paths.omp_agent_dir() == tmp_path / ".omp" / "agent"
    assert paths.omp_skills_dir() == tmp_path / ".omp" / "agent" / "skills"

    monkeypatch.setenv("PI_CONFIG_DIR", ".custom-omp")
    monkeypatch.setenv("PI_PROFILE", "fallback")
    monkeypatch.setenv("OMP_PROFILE", "  preferred  ")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/ignored-agent")
    assert paths.omp_config_root() == tmp_path / ".custom-omp"
    assert paths.omp_agent_dir() == (
        tmp_path / ".custom-omp" / "profiles" / "preferred" / "agent"
    )

    monkeypatch.setenv("OMP_PROFILE", "  ")
    assert paths.omp_agent_dir() == tmp_path / "ignored-agent"

    monkeypatch.setenv("OMP_PROFILE", "default")
    assert paths.omp_agent_dir() == tmp_path / "ignored-agent"


def test_skill_install_adds_default_omp_profile_when_root_exists(
    tmp_path, monkeypatch
):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    omp_config = tmp_path / ".omp"
    omp_agent = omp_config / "agent"
    omp_root = omp_agent / "skills"
    omp_config.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    rc = bn.cli.main(["skill", "install", "--mode", "copy"])

    assert rc == 0
    assert (omp_root / "bn-kernel" / "SKILL.md").exists()
    assert (omp_root / "bn" / "SKILL.md").exists()


def test_skill_install_adds_omp_when_only_agent_directory_exists(
    tmp_path, monkeypatch
):
    claude_root = tmp_path / "claude" / "skills"
    codex_home = tmp_path / "codex"
    omp_config = tmp_path / "missing-config"
    omp_agent = tmp_path / "custom-agent"
    omp_root = omp_agent / "skills"
    omp_agent.mkdir()
    monkeypatch.setattr(bn.cli, "claude_skills_dir", lambda: claude_root)
    monkeypatch.setattr(bn.cli, "codex_home", lambda: codex_home)
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    assert bn.cli.main(["skill", "install", "--mode", "copy"]) == 0
    assert (omp_root / "bn-kernel" / "SKILL.md").exists()


def test_skill_install_explicit_destination_never_writes_omp(
    tmp_path, monkeypatch
):
    explicit = tmp_path / "explicit"
    omp_config = tmp_path / ".omp"
    omp_agent = omp_config / "agent"
    omp_root = omp_agent / "skills"
    omp_config.mkdir()
    monkeypatch.setattr(bn.cli, "omp_config_root", lambda: omp_config)
    monkeypatch.setattr(bn.cli, "omp_agent_dir", lambda: omp_agent)
    monkeypatch.setattr(bn.cli, "omp_skills_dir", lambda: omp_root)

    rc = bn.cli.main(
        ["skill", "install", "--mode", "copy", "--dest", str(explicit)]
    )

    assert rc == 0
    assert (explicit / "bn-kernel" / "SKILL.md").exists()
    assert not omp_root.exists()

def test_version_flag_prints_version(capsys):
    # `bn --version` is a real affordance, not "unrecognized arguments" (#49).
    from bn.version import VERSION
    with pytest.raises(SystemExit) as exc:
        bn.cli.main(["--version"])
    assert exc.value.code == 0
    out, _ = capsys.readouterr()
    assert out.strip() == f"bn {VERSION}"


def test_version_is_single_sourced_from_pyproject():
    # The version literal lives only in pyproject.toml; version.py derives it,
    # so CLI/bridge never drift and a bump touches one file (#82).
    import tomllib
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    canonical = pyproject["project"]["version"]

    assert canonical == "0.20.0"  # the reset target
    assert bn.version.VERSION == canonical
    # No stray literal: the old number must be gone from the version module.
    assert "0.12.2" not in Path(bn.version.__file__).read_text(encoding="utf-8")


def test_resolve_version_falls_back_to_dist_metadata(monkeypatch):
    # When pyproject is unreachable (installed wheel), VERSION resolves from the
    # installed distribution metadata rather than crashing (#82).
    import tomllib

    import bn.version

    monkeypatch.setattr(
        tomllib, "loads",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pyproject here")),
    )
    # Editable install metadata reports the same canonical version.
    assert bn.version._resolve_version() == bn.version.VERSION


def test_bridge_plugin_json_carries_no_version_literal():
    # plugin.json is BN-manager metadata that cannot import Python; it must not
    # duplicate the version (single-source invariant, #82).
    import json
    from pathlib import Path

    import bn.version

    repo_root = Path(bn.version.__file__).resolve().parents[2]
    manifest = json.loads(
        (repo_root / "src" / "bn_agent_bridge" / "plugin.json").read_text(encoding="utf-8")
    )
    assert "version" not in manifest


def test_doctor_reports_stale_loaded_plugin(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": "0.4.0",
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None, **_kwargs: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": "0.4.0",
                "plugin_build_id": "oldbuild123456",
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor", "--format", "json"])

    # #620(c): staleness is informational and does not affect exit code -- the
    # instance is reachable, so doctor still exits 0.
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cli_version"] == bn.cli.VERSION
    assert payload["plugin_install_build_id"]
    assert payload["instances"][0]["stale_plugin_version"] is True
    assert payload["instances"][0]["stale_plugin_code"] is True


def test_doctor_flags_stale_engine(monkeypatch, tmp_path, capsys):
    # #161: doctor reports a per-instance engine fingerprint and flags
    # stale_engine when the loaded engine package diverges from on-disk.
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    for d in (install_dir, source_dir):
        (d / "bridge.py").write_text("print('bridge')\n", encoding="utf-8")
        (d / "taint_engine.py").write_text("X = 1\n", encoding="utf-8")

    fake_instance = type("FakeInstance", (), {
        "instance_id": "abc123", "pid": 123, "socket_path": tmp_path / "bridge.sock",
        "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
    })()
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None, **_kwargs: {"ok": True, "result": {
            "plugin_name": "bn_agent_bridge", "plugin_version": bn.cli.VERSION,
            "plugin_build_id": bn.cli.build_id_for_file(install_dir / "bridge.py"),
            # Loaded engine fingerprint differs from on-disk -> stale_engine.
            "engine_build_id": "staleengine00",
            "pid": 123, "socket_path": str(tmp_path / "bridge.sock"), "targets": [],
        }},
    )

    rc = bn.cli.main(["doctor", "--format", "json"])
    # #620(c): staleness is informational and does not affect exit code -- the
    # instance is reachable, so doctor still exits 0.
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    inst = payload["instances"][0]
    assert inst["stale_engine"] is True
    assert inst["stale_plugin_code"] is False  # bridge.py matches; only the engine is stale
    assert payload["engine_install_build_id"]


def test_session_restart_respawns_and_reloads_targets(monkeypatch, capsys):
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        # Restart reads the registry payload to recover private project
        # associations; this instance owned none.
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append((op, instance_id, params))
        return {"ok": True, "result": {"path": (params or {}).get("path")}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {"ok": True, "result": [
            {"filename": "/fw/svc_a", "analysis_state": "full"},
        ]},
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    spawned = {}
    def fake_spawn(instance_id=None):
        spawned["id"] = instance_id
        return new
    monkeypatch.setattr(bn.cli, "spawn_instance", fake_spawn)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["restarted"] is True
    assert payload["instance_id"] == "keep-me"
    assert spawned["id"] == "keep-me"             # respawned under the same id
    ops = [c[0] for c in calls]
    assert "shutdown" in ops and "load_binary" in ops   # stopped, then reloaded the target
    assert any(c[0] == "load_binary" and (c[2] or {}).get("path") == "/fw/svc_a" for c in calls)


def test_session_restart_reloads_each_target_as_the_file_it_is_753(monkeypatch, capsys):
    """#753: restart hardcoded `prefer_bndb: True`, so a target opened from a raw
    file that has a `.bndb` sidecar came back as the SIDECAR -- silently changing
    which file the target IS. When that sidecar was itself open as a second
    target, both reloads resolved to the same file and the instance came back
    with fewer targets than it had (measured live: 2 -> 1, the raw selector no
    longer resolvable, every read then about a different database).

    `filename` is BN's own answer for the file each view already is, so restart
    must reopen exactly that and never re-apply the sidecar preference."""
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )
    calls = []

    # The fake MODELS the bridge instead of echoing: with the sidecar preference
    # on, a raw path resolves to its existing sibling `.bndb`, and a path already
    # open comes back as the SAME target rather than a duplicate. Without that,
    # both the path list and `loaded == 2` hold on base too (the echo answers
    # whatever it is handed), so nothing but the pinned flag distinguished the two
    # worlds and the collapse symptom went untested (#857 review).
    sidecars = {"/fw/svc_a": "/fw/svc_a.bndb"}
    opened: dict[str, str] = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append((op, instance_id, params))
        if op != "load_binary":
            return {"ok": True, "result": {}}
        path = (params or {}).get("path")
        if (params or {}).get("prefer_bndb") and path in sidecars:
            path = sidecars[path]
        target_id = opened.setdefault(path, f"999:{len(opened) + 1}:7")
        return {"ok": True, "result": {"path": path, "target_id": target_id}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))
    # The reported shape: a raw target AND its own sidecar, both open.
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {"ok": True, "result": [
            {"filename": "/fw/svc_a", "analysis_state": "full"},
            {"filename": "/fw/svc_a.bndb", "analysis_state": "full"},
        ]},
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])

    assert rc == 0
    loads = [params for op, _, params in calls if op == "load_binary"]
    # Both targets are reloaded, each as its own file: no substitution, so the
    # two cannot collapse onto one view.
    assert [p["path"] for p in loads] == ["/fw/svc_a", "/fw/svc_a.bndb"]
    assert [p["prefer_bndb"] for p in loads] == [False, False]
    assert len(json.loads(capsys.readouterr().out)["loaded"]) == 2
    # The symptom, not just the flag: two DISTINCT views. On base the raw row
    # resolves to the sidecar and both rows land on one target_id -- 7 targets
    # came back as 6 in the live repro.
    assert len(set(opened.values())) == 2
    assert set(opened) == {"/fw/svc_a", "/fw/svc_a.bndb"}


def test_session_restart_records_capture_failure_but_still_restarts(monkeypatch, capsys):
    # #620(a): a `list_targets` failure while capturing the pre-restart state
    # must not be silently swallowed. Round 2: raising BridgeError here would
    # break the documented unreachable-bridge recovery path (#694), whose
    # list_targets can never succeed -- so the restart still proceeds, but the
    # failure is now reported honestly: a stderr warning AND structured
    # reload_capture_failed/reload_capture_error result fields AND a nonzero
    # exit code, matching the project_association_error pattern.
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))

    def failing_send_to_instance(instance, op, params=None, target=None):
        raise OSError("connection refused")

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", failing_send_to_instance)
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    spawned = {}
    def fake_spawn(instance_id=None):
        spawned["id"] = instance_id
        return new
    monkeypatch.setattr(bn.cli, "spawn_instance", fake_spawn)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])

    assert rc == 1
    assert spawned["id"] == "keep-me"   # restart still happened (recovery path preserved)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["restarted"] is True
    assert payload["loaded"] == []      # nothing captured to reload
    assert payload["reload_capture_failed"] is True
    assert payload["reload_capture_error"] == "OSError: connection refused"
    assert "could not list open targets before restarting keep-me" in captured.err


def test_session_restart_with_zero_open_targets_still_restarts(monkeypatch, capsys):
    # The only legitimate way to reach `loaded == []` at rc 0: list_targets
    # succeeded and simply returned no rows.
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"),
        "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me",
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None: {"ok": True, "result": []},
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    spawned = {}
    def fake_spawn(instance_id=None):
        spawned["id"] = instance_id
        return new
    monkeypatch.setattr(bn.cli, "spawn_instance", fake_spawn)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])

    assert rc == 0
    assert spawned["id"] == "keep-me"
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["restarted"] is True
    assert payload["loaded"] == []
    assert "reload_capture_failed" not in payload
    assert captured.err == ""


def test_doctor_text_marks_healthy_instance_ok(monkeypatch, tmp_path, capsys):
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('new build')\n", encoding="utf-8")

    fake_instance = type(
        "FakeInstance",
        (),
        {
            "pid": 123,
            "socket_path": tmp_path / "bridge.sock",
            "plugin_version": bn.cli.VERSION,
            "started_at": "2026-03-09T00:00:00+00:00",
        },
    )()

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    # Genuinely healthy: the loaded plugin_build_id matches what's on disk. The
    # doctor exit code is reachable-only (staleness is informational), but a
    # stale build id would still be a lie in a fixture meant to model a clean
    # install, so keep it matching regardless.
    matching_build_id = bn.cli.build_id_for_file(install_dir / "bridge.py")
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda instance, op, params=None, target=None, **_kwargs: {
            "ok": True,
            "result": {
                "plugin_name": "bn_agent_bridge",
                "plugin_version": bn.cli.VERSION,
                "plugin_build_id": matching_build_id,
                "pid": 123,
                "socket_path": str(tmp_path / "bridge.sock"),
                "targets": [],
            },
        },
    )

    rc = bn.cli.main(["doctor"])

    assert rc == 0
    output = capsys.readouterr().out
    assert f"pid=123 plugin={bn.cli.VERSION} status=ok" in output
    assert "status=error" not in output


def test_doctor_names_engine_version(monkeypatch, tmp_path, capsys):
    """`bn doctor` must name the Binary Ninja build the bridge is driving, in both
    text and JSON. Nothing else in the tool's output does, so after a BN major
    upgrade there is no way to tell which engine produced a given result -- the
    same command can decode/analyze differently across majors."""
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")

    fake_instance = type("FakeInstance", (), {
        "pid": 7, "socket_path": tmp_path / "bridge.sock",
        "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
        "instance_id": "engine",
    })()
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [fake_instance])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None, **_kwargs: {
            "ok": True,
            "result": {
                "plugin_version": bn.cli.VERSION, "plugin_build_id": "b",
                "bn_version": "6.1.10638-dev", "bn_build_id": "256112377",
                "targets": [],
            },
        },
    )

    assert bn.cli.main(["doctor"]) == 0
    assert "binary ninja: 6.1.10638-dev (build 256112377)" in capsys.readouterr().out

    assert bn.cli.main(["doctor", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["instances"][0]["bn_version"] == "6.1.10638-dev"
    assert data["instances"][0]["bn_build_id"] == "256112377"


def test_doctor_json_carries_reachable_and_status(monkeypatch, tmp_path, capsys):
    """doctor --format json must carry the same health signal the text mode shows
    (reachable / status), so a scripted JSON health check can read it directly
    instead of re-deriving reachability from the absence of doctor.error. (L16)"""
    install_dir = tmp_path / "install"
    source_dir = tmp_path / "source"
    install_dir.mkdir()
    source_dir.mkdir()
    (install_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")
    (source_dir / "bridge.py").write_text("print('b')\n", encoding="utf-8")

    def _inst(pid, name):
        return type("FakeInstance", (), {
            "pid": pid, "socket_path": tmp_path / f"{name}.sock",
            "plugin_version": bn.cli.VERSION, "started_at": "2026-03-09T00:00:00+00:00",
            "instance_id": name,
        })()

    ok_inst, bad_inst = _inst(1, "ok"), _inst(2, "bad")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [ok_inst, bad_inst])
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: install_dir)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: source_dir)

    def fake_send(instance, op, params=None, target=None, **_kwargs):
        if instance is ok_inst:
            return {"ok": True, "result": {
                "plugin_version": bn.cli.VERSION, "plugin_build_id": "b", "targets": []}}
        raise OSError("connection refused")

    monkeypatch.setattr(bn.cli, "_send_request_to_instance", fake_send)

    rc = bn.cli.main(["doctor", "--format", "json"])
    # #620(c): an unreachable instance makes the overall doctor exit nonzero,
    # so a scripted health check can trust the exit code alone.
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    by_pid = {i["pid"]: i for i in data["instances"]}
    assert by_pid[1]["reachable"] is True and by_pid[1]["status"] == "ok"
    assert by_pid[2]["reachable"] is False and by_pid[2]["status"] == "error"


def test_instance_flag_passed_to_send_request(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "abc123", "function", "list"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_on_subcommand(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "--instance", "abc123"])

    assert "abc123" in captured_instance_ids


def test_instance_flag_from_env(monkeypatch, capsys):
    captured_instance_ids = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured_instance_ids.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1:1:1", "selector": "test.bndb"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured_instance_ids


def test_session_list_shows_instances(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_instances = [
        BridgeInstance(
            pid=111,
            socket_path=__import__("pathlib").Path("/tmp/a.sock"),
            registry_path=__import__("pathlib").Path("/tmp/a.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:00:00Z",
            meta={"project_roots": ["/workspace/project-a"]},
            instance_id="aaaa1111",
        ),
        BridgeInstance(
            pid=222,
            socket_path=__import__("pathlib").Path("/tmp/b.sock"),
            registry_path=__import__("pathlib").Path("/tmp/b.json"),
            plugin_name="bn_agent_bridge",
            plugin_version="0.1.0",
            started_at="2026-01-01T00:01:00Z",
            meta={},
            instance_id="bbbb2222",
        ),
    ]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: fake_instances)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    # #358: session/instance list now uses the {kind, items} envelope (with
    # total_rss_mb kept as an extra field).
    assert parsed["kind"] == "instances"
    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["selector"] == "aaaa1111"
    assert parsed["items"][0]["instance_id"] == "aaaa1111"
    assert parsed["items"][0]["project_roots"] == ["/workspace/project-a"]
    assert parsed["items"][1]["instance_id"] == "bbbb2222"
    assert "rss_mb" in parsed["items"][0]
    assert "total_rss_mb" in parsed

    rc = bn.cli.main(
        ["session", "list", "-i", "bbbb2222", "--format", "json"]
    )
    assert rc == 0
    filtered = json.loads(capsys.readouterr().out)
    assert [item["instance_id"] for item in filtered["items"]] == ["bbbb2222"]


def test_session_status_polls_detached_load_job(monkeypatch, capsys):
    calls = []

    def fake_send_request(op, **kwargs):
        calls.append((op, kwargs))
        return {
            "ok": True,
            "result": {
                "kind": "load_jobs",
                "items": [
                    {
                        "job_id": "job123",
                        "state": "running",
                        "path": "/tmp/sample.bndb",
                    }
                ],
                "count": 1,
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        [
            "session",
            "status",
            "job123",
            "-i",
            "worker",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    assert calls[0][0] == "load_status"
    assert calls[0][1]["params"] == {"job_id": "job123"}
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["state"] == "running"


def test_session_status_job_passes_through_machine_fields(monkeypatch, capsys):
    # The CLI must not re-derive or flatten the bridge's job verdict; whatever
    # top-level state/terminal/succeeded contract the bridge publishes is what
    # `--format json` hands the polling agent.
    def fake_send_request(op, **kwargs):
        return {
            "ok": True,
            "result": {
                "kind": "load_job",
                "job_id": "job123",
                "state": "complete",
                "terminal": True,
                "succeeded": True,
                "job": {"job_id": "job123", "state": "complete", "path": "/tmp/s.bndb"},
                "items": [{"job_id": "job123", "state": "complete", "path": "/tmp/s.bndb"}],
                "count": 1,
                "status_command": "bn -i worker session status job123",
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(
        ["session", "status", "job123", "-i", "worker", "--format", "json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job123"
    assert payload["state"] == "complete"
    assert payload["terminal"] is True
    assert payload["succeeded"] is True
    assert payload["job"]["path"] == "/tmp/s.bndb"


def test_session_status_unknown_job_fails_loudly(monkeypatch, capsys):
    # A bad job id is an error, not an empty poll result: silently returning
    # "no jobs" would make a status loop spin until its own deadline.
    def fake_send_request(op, **kwargs):
        raise bn.transport.BridgeError("Unknown load job: nope")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "status", "nope", "-i", "worker"])

    assert rc != 0
    assert "Unknown load job: nope" in capsys.readouterr().err




@pytest.mark.parametrize("flag", ["--instance", "--instance-id"])
def test_session_stop_accepts_instance_flag(monkeypatch, capsys, flag):
    # #456: after threading --instance through every command, cleanup naturally
    # tries `session stop --instance <id>`; accept it as an alias for the positional.
    seen = {}

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        seen["op"] = op
        seen["instance_id"] = instance_id
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])

    rc = bn.cli.main(["session", "stop", flag, "abc123", "--format", "json"])
    assert rc == 0
    assert seen["op"] == "shutdown" and seen["instance_id"] == "abc123"


def test_session_stop_requires_an_instance_id(monkeypatch, capsys):
    # Neither positional nor --instance -> a clean error, not a crash.
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])
    rc = bn.cli.main(["session", "stop"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "instance id" in err.lower()


def test_session_stop_sends_shutdown(monkeypatch, capsys):
    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        assert op == "shutdown"
        assert instance_id == "abc123"
        return {"ok": True, "result": {"shutting_down": True}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["stopped"] is True
    assert parsed["instance_id"] == "abc123"


def test_session_start_rejects_global_instance_selector(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "spawn_instance",
        lambda *args, **kwargs: pytest.fail("session start spawned random instance"),
    )

    rc = bn.cli.main(["-i", "named", "session", "start", "/bin/ls"])

    assert rc == 2
    error = capsys.readouterr().err
    assert "--instance-id named" in error
    assert "does not name" in error


def test_session_start_spawns_instance(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)
    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {
            "ok": True,
            "result": {
                "instance_id": "test1234",
                "associated": [os.getcwd()],
                "skipped": [],
            },
        },
    )

    rc = bn.cli.main(["session", "start", "--format", "json"])

    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["instance_id"] == "test1234"
    assert parsed["pid"] == 999


def _fake_instance(instance_id):
    from bn.transport import BridgeInstance
    import pathlib
    return BridgeInstance(
        pid=999,
        socket_path=pathlib.Path("/tmp/test.sock"),
        registry_path=pathlib.Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id=instance_id,
    )


def test_session_start_associates_project_and_forwards_workdir(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m1")
    )
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        calls.append((op, params))
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "m1",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        return {
            "ok": True,
            "result": {
                "path": params.get("path"),
                "loaded": True,
                "targets": [{"selector": "x.bndb"}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main([
        "session", "start", str(tmp_path / "x.bndb"), "--format", "json"
    ])

    assert rc == 0
    assert calls[0] == ("associate_project_roots", {"roots": [str(tmp_path)]})
    load = next(params for op, params in calls if op == "load_binary")
    assert load["workdir"] == str(tmp_path)
    assert "no_marker" not in load
    assert json.loads(capsys.readouterr().out)["project_roots"] == [str(tmp_path)]


def test_session_start_text_mode_shows_associated_projects(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m1")
    )
    monkeypatch.chdir(tmp_path)

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "m1",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        return {
            "ok": True,
            "result": {
                "path": params.get("path"),
                "loaded": True,
                "targets": [{"selector": "x.bndb"}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", str(tmp_path / "x.bndb")])

    assert rc == 0
    assert f"projects: {tmp_path}" in capsys.readouterr().out


def test_session_start_text_mode_shows_association_error(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "spawn_instance", lambda instance_id=None: _fake_instance("m2")
    )

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            raise bn.cli.BridgeError("bridge is shutting down")
        raise AssertionError(f"unexpected op {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "project association error: bridge is shutting down" in out
    assert "pass -i m2" in out


def test_session_start_partial_failure_keeps_bridge_but_exits_nonzero(monkeypatch, capsys):
    from bn.transport import BridgeInstance

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=__import__("pathlib").Path("/tmp/test.sock"),
        registry_path=__import__("pathlib").Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="half",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    ops = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        ops.append(op)
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {"instance_id": "half", "associated": [], "skipped": []},
            }
        if op == "load_binary":
            if "good" in params["path"]:
                return {"ok": True, "result": {"path": params["path"], "loaded": True, "targets": [{"selector": "good.so"}]}}
            # Malformed success (no open target) -- NOT a transport-level
            # BridgeError -- this is the branch whose error text used to
            # predict teardown the bridge never actually performs.
            return {"ok": True, "result": {"path": params["path"], "loaded": True, "targets": []}}
        raise AssertionError(f"unexpected op: {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "start", "/tmp/good.so", "/tmp/bad.so", "--format", "json"])

    # One binary loaded, so the bridge stays up, but the failure still surfaces.
    assert rc == 1
    assert "shutdown" not in ops
    parsed = json.loads(capsys.readouterr().out)
    assert "stopped" not in parsed
    bad_error = next(
        item["error"] for item in parsed["loaded"] if item["path"].endswith("bad.so")
    )
    # The bridge survives partial success -- the per-binary error text must
    # not predict a teardown that never happens.
    assert "will be stopped" not in bad_error
    assert "stopped" not in bad_error


def test_close_ignores_sticky_target_pin(fake_transport, monkeypatch, capsys):
    # A sticky pin must NOT pick which target a bare `close` tears down, and a
    # stale pin must not make cleanup fail. The pin is dropped and the bare
    # selector then resolves like any target-required command: with a single
    # open target it closes that one (#664 -- it no longer means close-all).
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [{"target_id": "123:1:7", "selector": "foo.bndb"}],
        },
        "close_binary": {"ok": True, "result": {"closed": []}},
    })
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "stale_pin"})

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 0
    assert [c["op"] for c in calls] == ["list_targets", "close_binary"]
    assert calls[-1]["target"] == "123:1:7"  # pin dropped, single target pinned by id
    assert "all" not in (calls[-1]["params"] or {})


def test_close_with_sticky_pin_under_multiple_targets_refuses(fake_transport, monkeypatch, capsys):
    # #664: the sticky pin is still dropped for `close`, but under multiple open
    # targets that bare selector now REFUSES with the --target hint instead of
    # falling through to a close-all. Nothing is closed.
    calls = fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {"target_id": "123:1:7", "selector": "alpha.so", "view_id": "1"},
                {"target_id": "123:2:9", "selector": "beta.so", "view_id": "2"},
            ],
        },
        "close_binary": {"ok": True, "result": {"closed": []}},
    })
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"target": "alpha.so"})

    rc = bn.cli.main(["close", "--format", "text"])

    assert rc == 2
    assert [c["op"] for c in calls] == ["list_targets"]
    err = capsys.readouterr().err
    assert "requires --target when multiple targets are open" in err
    assert "alpha.so" in err and "beta.so" in err


def test_instance_use_writes_state(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "abc123"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "abc123"
    assert capsys.readouterr().out.strip() == "instance: abc123"


def test_instance_use_clears_stale_target_pin_on_switch(tmp_session, monkeypatch, capsys):
    # #368 facet 3: switching to a DIFFERENT instance clears the target pin (it
    # belonged to the old instance) so a coincidentally matching selector in the
    # new instance can't silently resolve a bare command to a different target.
    bn.session_state.update(instance_id="old", target="recv_daemon")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("new1")])

    rc = bn.cli.main(["instance", "use", "new1"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "new1"
    assert state.get("target") is None          # stale pin cleared
    assert "cleared stale target pin" in capsys.readouterr().out


def test_instance_use_keeps_target_pin_on_same_instance(tmp_session, monkeypatch, capsys):
    # Re-pinning the SAME instance keeps the target pin (not a switch).
    bn.session_state.update(instance_id="same1", target="recv_daemon")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("same1")])

    rc = bn.cli.main(["instance", "use", "same1"])

    assert rc == 0
    assert bn.session_state.read().get("target") == "recv_daemon"


def test_instance_use_keeps_target_pin_when_no_prior_instance(tmp_session, monkeypatch, capsys):
    # #368 review (MED): with a target pin but NO prior instance pin (prev_instance
    # is None), pinning an instance must NOT clear the target pin -- that is a first
    # pin, not a switch FROM a different instance. (None != resolved would wrongly
    # trip the stale-pin clear.)
    bn.session_state.update(target="recv_daemon")          # target pin, no instance pin
    assert bn.session_state.read().get("instance_id") is None
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("first1")])

    rc = bn.cli.main(["instance", "use", "first1"])

    assert rc == 0
    state = bn.session_state.read()
    assert state["instance_id"] == "first1"
    assert state.get("target") == "recv_daemon"            # pin kept (not a switch)
    assert "cleared stale target pin" not in capsys.readouterr().out


def test_instance_use_default_pins_gui_bridge(tmp_session, monkeypatch, capsys):
    # The fixed GUI bridge has instance_id=None and selector "default". Storing
    # the raw None made session_state.update() DELETE the pin, so the pin
    # silently vanished. `bn instance use default` must persist "default" so
    # later bare commands resolve to the GUI bridge (#93).
    gui = _fake_bridge_instance("gui")
    object.__setattr__(gui, "instance_id", None)  # GUI bridge: id is None
    named = _fake_bridge_instance("headless1")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [gui, named])

    rc = bn.cli.main(["instance", "use", "default"])

    assert rc == 0
    state = bn.session_state.read()
    assert state.get("instance_id") == "default"  # pin persisted, not deleted
    assert capsys.readouterr().out.strip() == "instance: default"


def test_instance_use_rejects_unknown_id(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [_fake_bridge_instance("abc123")])

    rc = bn.cli.main(["instance", "use", "not-running"])

    assert rc == 2
    assert "No running bridge instance" in capsys.readouterr().err
    assert bn.session_state.read() == {}


def test_instance_clear_removes_state(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="abc123")
    assert bn.session_state.read()["instance_id"] == "abc123"

    rc = bn.cli.main(["instance", "clear"])

    assert rc == 0
    assert "instance_id" not in bn.session_state.read()
    assert capsys.readouterr().out.strip() == "cleared"


def test_sticky_instance_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "sticky_inst" in captured


def test_cli_instance_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["--instance", "explicit", "function", "list"])

    assert "explicit" in captured
    assert "sticky_inst" not in captured


def test_env_var_overrides_sticky_instance(tmp_session, monkeypatch):
    bn.session_state.update(instance_id="sticky_inst")
    monkeypatch.setenv("BN_INSTANCE", "env_inst")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(instance_id)
        if op == "list_targets":
            return {"ok": True, "result": [{"target_id": "1", "selector": "x"}]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "env_inst" in captured
    assert "sticky_inst" not in captured


def test_sticky_target_fills_when_flag_absent(tmp_session, monkeypatch):
    bn.session_state.update(target="pam_qnx.so.2")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list"])

    assert "pam_qnx.so.2" in captured


def test_cli_target_flag_overrides_sticky(tmp_session, monkeypatch):
    bn.session_state.update(target="sticky_tgt")

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        captured.append(target)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    bn.cli.main(["function", "list", "-t", "explicit_tgt"])

    assert "explicit_tgt" in captured
    assert "sticky_tgt" not in captured


def test_session_state_survives_subdir_navigation(tmp_session, monkeypatch):
    # Mark tmp_session as a project root via .git, then descend into subdirs.
    (tmp_session / ".git").mkdir()
    bn.session_state.update(target="pam_qnx.so.2")

    sub = tmp_session / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    assert bn.session_state.read()["target"] == "pam_qnx.so.2"


def test_malformed_session_state_treated_as_empty(tmp_session):
    from bn.paths import session_state_path, sessions_dir

    sessions_dir().mkdir(parents=True, exist_ok=True)
    session_state_path().write_text("{not json")

    assert bn.session_state.read() == {}


def test_session_list_marks_sticky(tmp_session, monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "list_instances",
        lambda: [_fake_bridge_instance("aaaa1111"), _fake_bridge_instance("bbbb2222", pid=222)],
    )
    bn.session_state.update(instance_id="aaaa1111")

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    by_id = {entry["instance_id"]: entry for entry in parsed["items"]}
    assert by_id["aaaa1111"].get("sticky") is True
    assert "sticky" not in by_id["bbbb2222"]


def test_target_list_marks_sticky(tmp_session, fake_transport, capsys):
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {"target_id": "1", "selector": "foo.so", "filename": "/p/foo.so"},
                {"target_id": "2", "selector": "bar.so", "filename": "/p/bar.so"},
            ],
        }
    })
    bn.session_state.update(target="foo.so")

    rc = bn.cli.main(["target", "list", "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    # #358: target list now uses the {kind, items} envelope.
    assert parsed["kind"] == "targets"
    by_sel = {entry["selector"]: entry for entry in parsed["items"]}
    assert by_sel["foo.so"].get("sticky") is True
    assert "sticky" not in by_sel["bar.so"]


def test_stale_sticky_instance_emits_hint(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="dead_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(f"No bridge instance found with id: {instance_id}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "No bridge instance found with id: dead_inst" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_failed_contact(tmp_session, monkeypatch, capsys):
    """Bridge stopped mid-flight surfaces a transport error, not a registry miss."""
    bn.session_state.update(instance_id="dying_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Failed to contact Binary Ninja bridge pid 17881 at /tmp/x.sock: "
            "[Errno 104] Connection reset by peer"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Failed to contact" in err
    assert "bn instance clear" in err


def test_sticky_hint_on_bridge_timeout(tmp_session, monkeypatch, capsys):
    bn.session_state.update(instance_id="slow_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE(
            "Timed out waiting for Binary Ninja bridge pid 9999 at /tmp/x.sock after 30.0s"
        )

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["target", "list"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Timed out" in err
    assert "bn instance clear" in err


def test_sticky_hint_skipped_for_unrelated_errors(tmp_session, monkeypatch, capsys):
    """Bridge-side analysis errors must not get the sticky-clear hint."""
    bn.session_state.update(instance_id="alive_inst")

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        from bn.transport import BridgeError as _BE
        raise _BE("Function not found: nonexistent_symbol")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["function", "info", "nonexistent_symbol"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Function not found" in err
    assert "bn instance clear" not in err


def test_session_start_no_bndb_propagates_to_each_load(monkeypatch, tmp_path):
    from bn.transport import BridgeInstance
    import pathlib

    a = tmp_path / "a"
    a.write_bytes(b"")
    b = tmp_path / "b"
    b.write_bytes(b"")

    fake_inst = BridgeInstance(
        pid=999,
        socket_path=pathlib.Path("/tmp/test.sock"),
        registry_path=pathlib.Path("/tmp/test.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="test1234",
    )
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: fake_inst)

    captured = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        if op == "associate_project_roots":
            return {
                "ok": True,
                "result": {
                    "instance_id": "test1234",
                    "associated": [str(tmp_path)],
                    "skipped": [],
                },
            }
        captured.append(dict(params or {}))
        return {
            "ok": True,
            "result": {
                "loaded": True,
                "path": params["path"],
                "notes": [],
                "targets": [{"selector": pathlib.Path(params["path"]).name}],
            },
        }

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    rc = bn.cli.main(["session", "start", "--no-bndb", str(a), str(b)])

    assert rc == 0
    assert len(captured) == 2
    assert all(item["prefer_bndb"] is False for item in captured)
    assert {item["path"] for item in captured} == {str(a), str(b)}


def test_class_list_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "list", "--all", "--query", "Session"])
    assert args.handler(args) == 0
    assert captured["op"] == "class_list"
    assert captured["params"]["include_all"] is True
    assert captured["params"]["query"] == "Session"
    assert "no_stl" not in captured["params"]   # flag absent unless passed

    args = build_parser().parse_args(["class", "list", "--no-stl"])
    assert args.handler(args) == 0
    assert captured["params"]["no_stl"] is True


def test_class_show_invokes_op(monkeypatch):
    captured = {}

    def fake_call(args, op, params, **kwargs):
        captured["op"] = op
        captured["params"] = params
        return 0

    import bn.commands.cpp_class as cpp_class
    monkeypatch.setattr(cpp_class, "_call", fake_call)
    from bn.cli import build_parser
    args = build_parser().parse_args(["class", "show", "net::Session"])
    assert args.handler(args) == 0
    assert captured["op"] == "class_show"
    assert captured["params"]["name"] == "net::Session"


def test_admin_text_renderer_failure_becomes_clean_error(monkeypatch, capsys):
    # Admin commands build their result locally and render it WITHOUT going
    # through _call, so they must apply the SAME malformed-result guard _call has
    # (#101): a text renderer that raises must surface a clean BridgeError
    # (exit 2) pointing at --format json, never a raw traceback.
    import bn.commands.admin as admin

    monkeypatch.setattr(bn.cli, "list_instances", lambda: [])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    def _boom(_value):
        raise ValueError("simulated malformed bridge result")

    monkeypatch.setattr(admin, "_render_session_list_text", _boom)

    rc = bn.cli.main(["session", "list"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--format json" in err




def test_instance_gc_json_carries_counts(monkeypatch, capsys):
    summary = {
        "live_instances": 0, "registries_purged": 0,
        "logs_removed": 0, "sockets_removed": 0, "last_used_removed": 0,
        "removed": [],
    }
    monkeypatch.setattr(bn.cli, "gc_instances", lambda: summary)

    rc = bn.cli.main(["instance", "gc", "--format", "json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["logs_removed"] == 0
    assert data["live_instances"] == 0


def test_instance_gc_reports_summary_text(monkeypatch, capsys):
    # #80: `bn instance gc` reaps dead-instance cache litter and reports counts.
    monkeypatch.setattr(bn.cli, "gc_instances", lambda: {
        "live_instances": 2, "registries_purged": 1,
        "logs_removed": 147, "sockets_removed": 3, "last_used_removed": 2,
        "removed": ["x"],
    })

    rc = bn.cli.main(["instance", "gc"])

    assert rc == 0
    out = capsys.readouterr().out
    # The WHOLE line, so a malformed fragment (a missing comma between the new
    # sidecar count and the registry count) cannot pass on a substring (#733 F6).
    assert out.strip() == (
        "gc: reaped 147 logs, 3 orphan sockets, 2 last-used sidecars, "
        "1 dead registry (2 live instances kept)"
    )
    assert "Traceback" not in out


# --- #276 Option 2: machine-readable capability index -----------------------

def test_capabilities_json_index_is_registry_derived(capsys):
    # A structured, registry-derived command->purpose->prefer-when index an
    # agent reads once to route. Local command -- no bridge/target required.
    rc = bn.cli.main(["capabilities", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert data["kind"] == "capabilities"
    items = data["items"]
    assert items and data["count"] == len(items)
    for it in items:
        assert {"command", "group", "help", "requires_target",
                "default_format", "prefer_when", "see_also"} <= set(it.keys())

    by_cmd = {it["command"]: it for it in items}
    # the overlaps the issue calls out are present and cross-linked
    assert "exact" in by_cmd["callsites"]["prefer_when"].lower()
    assert "xrefs" in by_cmd["callsites"]["see_also"]
    assert "callsites" in by_cmd["xrefs"]["see_also"]
    assert "function search" in by_cmd["function list"]["see_also"]
    assert "function list" in by_cmd["function search"]["see_also"]


def test_capabilities_see_also_references_are_valid_commands(capsys):
    # Integrity: every see_also points at a real registered command (the index
    # is registry-derived, so a stale/typo'd cross-link must fail loudly).
    rc = bn.cli.main(["capabilities", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    commands = {it["command"] for it in items}
    for it in items:
        for ref in it["see_also"]:
            assert ref in commands, f"{it['command']} see_also -> unknown command {ref!r}"


def test_capabilities_text_groups_commands_with_routing_hints(capsys):
    rc = bn.cli.main(["capabilities"])  # text is the default
    assert rc == 0
    out = capsys.readouterr().out
    assert "callsites" in out and "xrefs" in out
    assert "prefer when:" in out
    assert "see also:" in out


def test_help_command_advertises_machine_catalog(capsys):
    rc = bn.cli.main(["help", "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "help"
    assert "function" in payload["groups"]
    assert payload["capabilities_command"] == "bn capabilities --format json"


def test_help_command_filters_one_family(capsys):
    rc = bn.cli.main(["help", "function"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "function list:" in output
    assert "function search:" in output
    assert "machine-readable catalog" not in output


def _inst_with_binaries(binaries):
    from pathlib import Path as _P
    from bn.transport import BridgeInstance
    return BridgeInstance(
        pid=111, socket_path=_P("/tmp/x.sock"), registry_path=_P("/tmp/x.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={"binaries": list(binaries)}, instance_id="abc123")


def test_instance_list_shows_open_binaries(monkeypatch, capsys):
    # #80: `bn instance list` surfaces each instance's open binaries from the
    # registry, so "which instance has libfoo.so?" needs no per-instance round-trip.
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/fw/bin/daemon"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "libfoo.so" in out and "daemon" in out


def test_instance_list_json_includes_binaries(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["items"][0]["binaries"] == ["/fw/lib64/libfoo.so"]


def _one_instance(monkeypatch):
    """One registry row, stubbed like every other `session list` cell: the
    `fake_transport` fixture patches only `cli.send_request`, so without this
    the listing is empty and an `items[0]` assertion raises IndexError."""
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    inst = BridgeInstance(
        pid=111, socket_path=_P("/tmp/x.sock"), registry_path=_P("/tmp/x.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="aaaa1111")
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})


def test_session_list_probes_a_gui_bridge_by_its_selector(monkeypatch, capsys):
    """The legacy GUI pair registers no instance id, and `default` is the
    selector `choose_instance` matches on -- so keying the probe on
    `instance_id` alone reported every GUI bridge as unknown without asking it
    (#733 F1)."""
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    gui = BridgeInstance(
        pid=222, socket_path=_P("/tmp/g.sock"), registry_path=_P("/tmp/g.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id=None)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [gui])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    asked = []

    def fake_send_request(op, *, params=None, instance_id=None, **kwargs):
        asked.append(instance_id)
        return {"ok": True, "result": [{"selector": "netsvcd", "unsaved": True}]}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    assert asked == ["default"]
    assert json.loads(capsys.readouterr().out)["items"][0]["unsaved_targets"] == 1


def test_the_two_peer_probes_declare_themselves_as_probes_756(monkeypatch, capsys, tmp_path):
    """#859 review: deleting either `idle_probe=True` declaration left the suite
    green, so the CLI half of the contract was untested. This records the FULL
    kwarg set at both declaring sites -- the #787 direction -- and pins it.

    `session list` and `doctor` are the two commands that issue a real
    per-instance request purely to answer "is this bridge alive / would closing
    it discard work". That traffic belongs to whoever ran the command, never to
    the bridge's owner, so both must declare `idle_probe=True`. An ordinary
    command must NOT: `bn target list` issues the same `list_targets` op as the
    peer probe, and it is real work that keeps the bridge alive."""
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    inst = BridgeInstance(
        pid=222, socket_path=_P("/tmp/p.sock"), registry_path=_P("/tmp/p.json"),
        plugin_name="bn_agent_bridge", plugin_version=bn.cli.VERSION,
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="probe-me")
    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    monkeypatch.setattr(bn.cli, "plugin_install_dir", lambda: tmp_path)
    monkeypatch.setattr(bn.cli, "plugin_source_dir", lambda: tmp_path)
    sent: list[dict] = []

    def record_send_request(op, **kwargs):
        sent.append({"op": op, **kwargs})
        return {"ok": True, "result": [{"selector": "netsvcd", "unsaved": True}]}

    def record_to_instance(instance, op, **kwargs):
        sent.append({"op": op, **kwargs})
        return {"ok": True, "result": {
            "plugin_version": bn.cli.VERSION, "plugin_build_id": "b", "targets": []}}

    monkeypatch.setattr(bn.cli, "send_request", record_send_request)
    monkeypatch.setattr(bn.cli, "_send_request_to_instance", record_to_instance)

    assert bn.cli.main(["session", "list", "--format", "json"]) == 0
    bn.cli.main(["doctor", "--format", "json"])
    bn.cli.main(["target", "list", "--format", "json", "-i", "probe-me"])
    capsys.readouterr()

    by_op: dict[str, list[dict]] = {}
    for call in sent:
        by_op.setdefault(call["op"], []).append(call)

    # The two peer probes: declared, explicitly True (not merely truthy).
    probes = [c for c in by_op["list_targets"] if c.get("strict") or c.get("params") == {"strict": True}]
    assert probes, f"session list issued no strict list_targets probe: {sent}"
    assert all(c.get("idle_probe") is True for c in probes), probes
    assert all(c.get("idle_probe") is True for c in by_op["doctor"]), by_op["doctor"]

    # Ordinary work: `bn target list` is the SAME op and must not be exempt.
    ordinary = [c for c in by_op["list_targets"] if c not in probes]
    assert ordinary, f"target list issued no plain list_targets: {sent}"
    assert all(c.get("idle_probe") in (False, None) for c in ordinary), ordinary


def test_session_list_probes_instances_concurrently(monkeypatch, capsys):
    """A fleet triage must not pay one probe budget per bridge: a serial sweep
    of a wedged fleet is exactly the "one wedged bridge blocks the survey" the
    budget exists to prevent (#733 F1).

    Proven with a BARRIER rather than a stopwatch: every probe must be inside
    the stub at the same moment, which a serial sweep can never satisfy. A
    wall-clock threshold would answer the same question less exactly and could
    flake on a loaded box running this suite under `-n 8`.
    """
    import threading
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    insts = [
        BridgeInstance(
            pid=300 + n, socket_path=_P(f"/tmp/{n}.sock"),
            registry_path=_P(f"/tmp/{n}.json"), plugin_name="bn_agent_bridge",
            plugin_version="0.1.0", started_at="2026-01-01T00:00:00Z",
            meta={}, instance_id=f"bbbb{n}{n}{n}{n}")
        for n in range(4)
    ]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: insts)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    all_inside = threading.Barrier(len(insts))
    observed = {"concurrent": False}

    def wedged(op, *, params=None, instance_id=None, **kwargs):
        try:
            all_inside.wait(timeout=10)
            observed["concurrent"] = True
        except threading.BrokenBarrierError:
            pass          # serial: the others never arrived
        raise bn.cli.BridgeError("bridge did not answer")

    monkeypatch.setattr(bn.cli, "send_request", wedged)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    assert observed["concurrent"], "the probes never ran at the same time"
    items = json.loads(capsys.readouterr().out)["items"]
    # Order is the registry's, not completion order.
    assert [item["instance_id"] for item in items] == [i.instance_id for i in insts]
    assert all(item["unsaved_targets_unavailable"] == "bridge did not answer"
               for item in items)


def _wedged_fleet(monkeypatch, count):
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    insts = [
        BridgeInstance(
            pid=400 + n, socket_path=_P(f"/tmp/w{n}.sock"),
            registry_path=_P(f"/tmp/w{n}.json"), plugin_name="bn_agent_bridge",
            plugin_version="0.1.0", started_at="2026-01-01T00:00:00Z",
            meta={}, instance_id=f"cccc{n:04d}")
        for n in range(count)
    ]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: insts)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    return insts


def test_session_list_bounds_the_whole_probe_not_each_wave(monkeypatch, capsys):
    """The budget is ONE deadline for the command, not one per wave.

    The probes fan out `_PROBE_FAN` at a time, so a per-probe budget made a
    fleet larger than the fan cost `ceil(n / _PROBE_FAN)` budgets -- 10s for
    the fourteen-bridge fleet in #733, 125s for 200 -- and a Ctrl-C waited out
    all of it. A bridge the deadline never reached is disclosed by name, never
    counted as zero unsaved work (#733 F1 review).

    The distinguishing signal is the DISCLOSURE, not the stopwatch: under a
    per-probe budget the second wave gets a fresh one and reports "bridge did
    not answer"; under one shared deadline it reports that the budget expired
    before it was reached. That is exact and cannot flake under `-n 8`.
    """
    import threading
    import time
    import bn.commands.admin as admin

    fan = admin._PROBE_FAN
    insts = _wedged_fleet(monkeypatch, fan * 2)
    monkeypatch.setattr(admin, "_UNSAVED_PROBE_TIMEOUT", 0.3)

    first_wave = threading.Barrier(fan)

    def wedged(op, *, params=None, instance_id=None, **kwargs):
        try:
            # Only the first `fan` probes meet here; they then burn the whole
            # shared budget between them.
            first_wave.wait(timeout=10)
            time.sleep(0.4)
        except threading.BrokenBarrierError:
            pass
        raise bn.cli.BridgeError("bridge did not answer")

    monkeypatch.setattr(bn.cli, "send_request", wedged)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert [item["instance_id"] for item in items] == [i.instance_id for i in insts]
    expired = [item for item in items if item.get("unsaved_targets_unavailable")
               == "the probe budget expired before this bridge was reached"]
    assert len(expired) == fan, (
        "the second wave was given its own budget instead of the remainder: "
        f"{[item.get('unsaved_targets_unavailable') for item in items]}")
    # And no row is ever left claiming zero unsaved work.
    assert all("unsaved_targets" not in item for item in items)


def test_session_list_survives_an_exception_with_no_message(monkeypatch, capsys):
    """A listing degrades to `unknown`; it never becomes a traceback.

    A bare `TimeoutError()` has an empty `str()`, so the first-line slice was
    an `IndexError` -- not in the caught set, re-raised out of the thread pool,
    and the whole command died where the belt was supposed to absorb it. An
    empty reason is also useless, so the exception TYPE is named instead
    (#733 F1 review).
    """
    _one_instance(monkeypatch)

    def silent(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(bn.cli, "send_request", silent)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert items[0]["unsaved_targets_unavailable"] == "TimeoutError"
    assert "unsaved_targets" not in items[0]


def test_session_list_asks_strictly_and_reports_a_lossy_snapshot_as_unknown(
    monkeypatch, capsys
):
    """The probe is a SAFETY count, so it must not read one off a lossy walk.

    `list_targets` is non-strict by default -- a listing has no business
    failing because one UI query hiccuped -- but a count of unsaved targets
    derived from a snapshot that silently omits tabs is a definitive "nothing
    would be discarded" about work the reader cannot see. The probe therefore
    sends `strict: true` and discloses the bridge's refusal as unknown
    (#733 F1 review).
    """
    _one_instance(monkeypatch)
    sent = []

    def enumeration_failed(op, *, params=None, instance_id=None, **kwargs):
        sent.append((op, params))
        raise bn.cli.BridgeError(
            "Unable to enumerate every open BinaryView tab: a UI query raised "
            "mid-walk, so the open-view count cannot be trusted")

    monkeypatch.setattr(bn.cli, "send_request", enumeration_failed)

    rc = bn.cli.main(["session", "list", "--format", "json"])

    assert rc == 0
    assert sent == [("list_targets", {"strict": True})]
    items = json.loads(capsys.readouterr().out)["items"]
    assert "unsaved_targets" not in items[0]
    assert items[0]["unsaved_targets_unavailable"].startswith(
        "Unable to enumerate every open BinaryView tab")



def test_session_list_counts_unsaved_targets_per_instance(
    monkeypatch, fake_transport, capsys
):
    """#733 F1: "is it safe to stop these bridges?" answered without closing.

    The count is LIVE, not registry meta: the registry's `binaries` list is
    written on load/close only, so a registry-sourced count would report zero
    unsaved work for a bridge holding unsaved renames.
    """
    _one_instance(monkeypatch)
    fake_transport({
        "list_targets": {
            "ok": True,
            "result": [
                {"selector": "netsvcd", "unsaved": True},
                {"selector": "dnsproxy", "unsaved": False},
            ],
        }
    })

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert items[0]["unsaved_targets"] == 1

    rc = bn.cli.main(["session", "list"])
    assert rc == 0
    assert "unsaved targets: 1" in capsys.readouterr().out


def test_session_list_reports_an_unreachable_bridge_as_unknown(
    monkeypatch, capsys
):
    """A bridge that cannot answer is UNKNOWN, never zero: a fabricated 0 reads
    as "nothing would be discarded", the one wrong answer here (#733 F1)."""
    _one_instance(monkeypatch)

    def boom(*args, **kwargs):
        raise bn.cli.BridgeError("boom")

    monkeypatch.setattr(bn.cli, "send_request", boom)

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert "unsaved_targets" not in items[0]
    assert items[0]["unsaved_targets_unavailable"] == "boom"

    rc = bn.cli.main(["session", "list"])
    assert rc == 0
    assert "unsaved targets: unknown — boom" in capsys.readouterr().out


def test_session_list_reports_a_bridge_without_the_field_as_unknown(
    monkeypatch, fake_transport, capsys
):
    """An older bridge whose rows carry no `unsaved` key is disclosed by name,
    not counted as zero (#733 F1)."""
    _one_instance(monkeypatch)
    fake_transport({
        "list_targets": {"ok": True, "result": [{"selector": "netsvcd"}]}
    })

    rc = bn.cli.main(["session", "list", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert "unsaved_targets" not in items[0]
    assert items[0]["unsaved_targets_unavailable"] == (
        "this bridge does not report per-target unsaved state"
    )


def test_instance_list_stays_round_trip_free(monkeypatch, capsys):
    """#80 made `instance list` answerable from the registry alone; #733 F1's
    probe is deliberately `session list`-only. A `send_request` that explodes
    proves no round trip happens here."""
    _one_instance(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("instance list must not contact a bridge")

    monkeypatch.setattr(bn.cli, "send_request", boom)

    rc = bn.cli.main(["instance", "list", "--format", "json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert "unsaved_targets" not in items[0]
    assert "unsaved_targets_unavailable" not in items[0]


def test_instance_list_no_binaries_key_when_empty(monkeypatch, capsys):
    # An instance with nothing open (or an older registry without the field) renders
    # cleanly without a binaries line.
    inst = _inst_with_binaries([])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "binaries" not in out


def test_instance_find_locates_binary_by_basename(monkeypatch, capsys):
    # #80: `bn instance find <name>` answers "which instance has this binary?"
    # from the registry (no per-instance round-trip), matching by basename.
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/fw/bin/daemon"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo.so"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc123" in out and "libfoo.so" in out


def test_instance_find_by_exact_path(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "/fw/lib64/libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 1
    assert data["items"][0]["instance_id"] == "abc123"
    assert data["items"][0]["binary"] == "/fw/lib64/libfoo.so"


def test_instance_find_no_match(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "nope.so"])
    assert rc == 0
    assert "no instance" in capsys.readouterr().out.lower()


def test_instance_find_substring_of_basename(monkeypatch, capsys):
    # a bare query is a basename substring, so "libfoo" finds "libfoo.so.1.2.11"
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so.1.2.11"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo"])
    assert rc == 0
    assert "abc123" in capsys.readouterr().out


def test_instance_find_path_suffix_is_component_aligned(monkeypatch, capsys):
    # A path-form query matches as a component-aligned suffix: `lib64/libfoo.so`
    # matches `/fw/lib64/libfoo.so` but a mid-component byte suffix must NOT
    # (`bar/libfoo.so` must not match `/foobar/libfoo.so`) (#80 review M1).
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so", "/foobar/libqux.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "lib64/libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [i["binary"] for i in data["items"]] == ["/fw/lib64/libfoo.so"]
    # mid-component suffix does not match
    rc = bn.cli.main(["instance", "find", "bar/libqux.so", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0


def test_instance_find_empty_query_matches_nothing(monkeypatch, capsys):
    inst = _inst_with_binaries(["/fw/lib64/libfoo.so"])
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [inst])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0


def test_instance_find_across_multiple_instances_and_old_bridge(monkeypatch, capsys):
    # A query matching binaries in TWO instances lists both; an older bridge whose
    # registry has no `binaries` key is skipped without error.
    a = _inst_with_binaries(["/fw/lib64/libfoo.so"]); a.instance_id = "inst_a"
    b = _inst_with_binaries(["/other/libfoo.so"]); b.instance_id = "inst_b"
    old = _inst_with_binaries([]); old.instance_id = "old"; old.meta.pop("binaries", None)
    monkeypatch.setattr(bn.cli, "list_instances", lambda: [a, b, old])
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})
    rc = bn.cli.main(["instance", "find", "libfoo.so", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert {i["instance_id"] for i in data["items"]} == {"inst_a", "inst_b"}
    assert data["count"] == 2


def test_explicit_empty_instance_is_rejected_not_pin_filled(fake_transport, monkeypatch, capsys):
    # #690 r3: `-i "$INST"` with $INST unset must not silently route the
    # command to the pinned instance (the same unset-shell-var doctrine the
    # r2 close guards follow for -t and the path).
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["function", "list", "-i", ""])

    assert rc == 2
    assert calls == []
    assert "--instance is empty" in capsys.readouterr().err


def test_session_stop_rejects_explicit_empty_positional(fake_transport, monkeypatch, capsys):
    # #690 r4: `bn session stop "$ID"` with $ID unset must NOT fall through to
    # the sticky-pin-filled -i and shut down the PINNED bridge.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop", ""])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "instance id is empty" in err


def test_session_stop_bare_does_not_stop_sticky_pinned_instance(fake_transport, monkeypatch, capsys):
    # #588: a bare `session stop` (no positional, no -i/--instance at all) must
    # NOT fall through to a sticky-pinned instance and stop it -- only a
    # genuinely explicit -i on THIS invocation may supply the target.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop"])

    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "instance id" in err.lower()


def test_session_stop_explicit_instance_flag_still_resolves_sticky_slot(monkeypatch, capsys):
    # #588: an EXPLICIT -i/--instance on the `session stop` invocation itself
    # (as opposed to a silently-injected sticky pin) must still work, and must
    # target the FLAG's id, not whatever happens to be pinned.
    calls = []
    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                           instance_id=None, spawn_missing_named=False, **kwargs):
        calls.append({"op": op, "instance_id": instance_id})
        return {"ok": True, "result": {}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})

    rc = bn.cli.main(["session", "stop", "-i", "other-inst", "--format", "json"])

    assert rc == 0
    assert calls == [{"op": "shutdown", "instance_id": "other-inst"}]


def test_session_stop_bn_instance_env_overrides_sticky_pin(monkeypatch, capsys):
    # #588 finding 2: BN_INSTANCE is documented as "same effect as always
    # passing -i" (runtime.md:41) and must still resolve a bare `session
    # stop` even though a DIFFERENT instance is sticky-pinned.
    calls = []
    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                           instance_id=None, spawn_missing_named=False, **kwargs):
        calls.append({"op": op, "instance_id": instance_id})
        return {"ok": True, "result": {}}
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {"instance_id": "pinned-inst"})
    monkeypatch.setenv("BN_INSTANCE", "env-inst")

    rc = bn.cli.main(["session", "stop", "--format", "json"])

    assert rc == 0
    assert calls == [{"op": "shutdown", "instance_id": "env-inst"}]


def test_close_empty_instance_gets_the_actionable_message(fake_transport, monkeypatch, capsys):
    # #690 r4: close peeks list_targets BEFORE _call, so the empty-instance
    # rejection must fire on the peek path too -- with the same actionable
    # message every other command gets, not transport's generic one.
    calls = fake_transport({})
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["close", "-i", ""])

    assert rc == 2
    assert calls == []
    assert "--instance is empty" in capsys.readouterr().err


# --------------------------------------------------------------------------
# #694: stop/restart must prove the pid is still the bridge before signalling
# --------------------------------------------------------------------------


class _FakeSignaller:
    """Stand-in for transport.BridgeProcessSignal: records every send.

    Signalling is atomic now (pin the pid, verify identity through the pin, send
    through the same pin), so the CLI's remaining job is routing refusals and
    escalations. That routing is what these tests pin down; the pin itself is
    covered in tests/test_transport.py.
    """

    instances: list["_FakeSignaller"] = []

    def __init__(self, instance, refusal=None):
        self.instance = instance
        self.refusal = refusal
        self.sent: list[int] = []
        self.closed = False
        _FakeSignaller.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self.closed = True

    def send(self, sig):
        if self.refusal is not None:
            return self.refusal
        self.sent.append(sig)
        return None


def _fake_signaller(monkeypatch, refusal=None):
    """Install the fake signaller; returns the list of created signallers."""
    _FakeSignaller.instances = []
    monkeypatch.setattr(
        bn.cli,
        "BridgeProcessSignal",
        lambda instance: _FakeSignaller(instance, refusal=refusal),
    )
    return _FakeSignaller.instances


def _unreachable_fake_instance(instance_id="abc123", pid=111):
    inst = _fake_bridge_instance(instance_id, pid=pid)  # noqa: F405
    inst.unreachable = True
    return inst


def test_session_stop_sigterm_fallback_sends_through_the_verified_pin(monkeypatch, capsys):
    import signal as signal_mod

    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    signallers = _fake_signaller(monkeypatch)
    monkeypatch.setattr("os.kill", lambda pid, sig: pytest.fail("raw os.kill is banned"))

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert len(signallers) == 1                      # one pin for the whole stop
    assert signallers[0].sent == [signal_mod.SIGTERM]
    assert signallers[0].closed is True
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is True and parsed["method"] == "sigterm"


def test_session_stop_escalates_sigkill_through_the_same_pin(monkeypatch, capsys):
    import signal as signal_mod

    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {"ok": True, "result": {"shutting_down": True}},
    )
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    signallers = _fake_signaller(monkeypatch)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert len(signallers) == 1                      # SAME pin, never reopened
    assert signallers[0].sent == [signal_mod.SIGKILL]
    assert json.loads(capsys.readouterr().out)["method"] == "sigkill"


def test_session_stop_reports_a_refused_sigterm_and_signals_nothing(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    signallers = _fake_signaller(
        monkeypatch,
        refusal=(
            "refusing to signal pid 111 for bridge instance 'abc123': the identity "
            "recorded at startup (boot id plus process start time) does not match "
            "the pinned process, so the bridge exited and its pid was reused"
        ),
    )

    rc = bn.cli.main(["session", "stop", "abc123"])

    assert rc == 1
    assert signallers[0].sent == []
    err = capsys.readouterr().err
    assert "refusing to signal pid 111" in err
    assert "reused" in err


def test_session_stop_reports_a_refused_sigkill_escalation(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "send_request",
        lambda op, **kwargs: {"ok": True, "result": {"shutting_down": True}},
    )
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("abc123")  # noqa: F405
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: False)
    signallers = _fake_signaller(
        monkeypatch, refusal="refusing to signal pid 111: no verifiable process identity"
    )

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 1
    assert signallers[0].sent == []
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is False
    assert "did not fully tear down" in parsed["error"]
    assert "no verifiable process identity" in parsed["error"]


def test_session_stop_resolves_an_unreachable_bridge(monkeypatch, capsys):
    # A socket-less bridge is hidden from `session list` (nothing can be
    # dispatched to it), but stopping the live process it names is exactly what a
    # user needs -- so stop resolves it through the lifecycle lookup (#694).
    import signal as signal_mod

    from bn.transport import BridgeError

    lookups: list[str] = []

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("No bridge instance found with id: abc123")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "list_instances", lambda **kw: pytest.fail("must use the lifecycle lookup")
    )
    monkeypatch.setattr(
        bn.cli,
        "find_lifecycle_instance",
        lambda target: lookups.append(target) or _unreachable_fake_instance("abc123"),
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    signallers = _fake_signaller(monkeypatch)

    rc = bn.cli.main(["session", "stop", "abc123", "--format", "json"])

    assert rc == 0
    assert lookups == ["abc123"]
    assert signallers[0].sent == [signal_mod.SIGTERM]
    assert json.loads(capsys.readouterr().out)["method"] == "sigterm"


def test_session_restart_refuses_when_the_signal_is_not_delivered(monkeypatch, capsys):
    from bn.transport import BridgeError

    def fail_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        raise BridgeError("bridge unreachable")

    monkeypatch.setattr(bn.cli, "send_request", fail_send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("res1")  # noqa: F405
    )
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance", lambda *a, **k: {"ok": True, "result": []}
    )
    monkeypatch.setattr(
        bn.cli,
        "spawn_instance",
        lambda instance_id=None: pytest.fail("must not respawn after refusing"),
    )
    signallers = _fake_signaller(
        monkeypatch,
        refusal="refusing to signal pid 111 for bridge instance 'res1': its pid was reused",
    )

    rc = bn.cli.main(["session", "restart", "res1"])

    assert rc == 2
    assert signallers[0].sent == []
    err = capsys.readouterr().err
    assert "refusing to signal pid 111" in err
    assert "left as it is" in err


def test_session_restart_escalates_through_one_pin(monkeypatch, capsys):
    import pathlib
    import signal as signal_mod

    from bn.transport import BridgeError, BridgeInstance

    def send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
        if op == "shutdown":
            raise BridgeError("bridge unreachable")
        return {"ok": True, "result": {"loaded": True, "path": "/tmp/app.bin", "targets": []}}

    new_instance = BridgeInstance(
        pid=5151,
        socket_path=pathlib.Path("/tmp/res1.sock"),
        registry_path=pathlib.Path("/tmp/res1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="res1",
    )
    monkeypatch.setattr(bn.cli, "send_request", send_request)
    monkeypatch.setattr(
        bn.cli, "find_lifecycle_instance", lambda target: _fake_bridge_instance("res1")  # noqa: F405
    )
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance", lambda *a, **k: {"ok": True, "result": []}
    )
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new_instance)
    signallers = _fake_signaller(monkeypatch)

    assert bn.cli.main(["session", "restart", "res1", "--format", "json"]) == 0
    assert len(signallers) == 1
    assert signallers[0].sent == [signal_mod.SIGTERM, signal_mod.SIGKILL]
    assert signallers[0].closed is True


def _failing_start_instance():
    import pathlib

    from bn.transport import BridgeInstance

    return BridgeInstance(
        pid=7777,
        socket_path=pathlib.Path("/tmp/start1.sock"),
        registry_path=pathlib.Path("/tmp/start1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at=None,
        meta={},
        instance_id="start1",
    )


def _failing_start_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, **kwargs):
    from bn.transport import BridgeError

    raise BridgeError("load failed" if op.startswith("load_binary") else "shutdown refused")


def test_session_start_cleanup_signals_through_the_verified_pin(monkeypatch, capsys, tmp_path):
    # The all-preloads-failed cleanup is a fallback signalling path too: it used
    # raw os.kill and could terminate a recycled pid (#694).
    import signal as signal_mod

    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: _failing_start_instance())
    monkeypatch.setattr(bn.cli, "send_request", _failing_start_send_request)
    converged = iter([False, True])
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: next(converged))
    signallers = _fake_signaller(monkeypatch)
    monkeypatch.setattr("os.kill", lambda pid, sig: pytest.fail("raw os.kill is banned"))

    rc = bn.cli.main(["session", "start", str(binary), "--format", "json"])

    assert rc == 1
    assert len(signallers) == 1                      # one pin for TERM + KILL
    assert signallers[0].sent == [signal_mod.SIGTERM, signal_mod.SIGKILL]
    assert signallers[0].closed is True
    assert json.loads(capsys.readouterr().out)["stopped"] is True


def test_session_start_cleanup_reports_a_refused_signal(monkeypatch, capsys, tmp_path):
    binary = tmp_path / "app.bin"
    binary.write_bytes(b"\x7fELF")
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: _failing_start_instance())
    monkeypatch.setattr(bn.cli, "send_request", _failing_start_send_request)
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: False)
    signallers = _fake_signaller(
        monkeypatch, refusal="refusing to signal pid 7777: no verifiable process identity"
    )

    rc = bn.cli.main(["session", "start", str(binary), "--format", "json"])

    assert rc == 1
    assert signallers[0].sent == []
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stopped"] is False
    errors = " ".join(parsed["cleanup_errors"])
    assert "SIGTERM not sent" in errors and "SIGKILL not sent" in errors
    assert "no verifiable process identity" in errors


# --------------------------------------------------------------------------
# #694: restart preserves private project associations
# --------------------------------------------------------------------------


def _restart_bridge_stubs(
    monkeypatch, project_roots, *, restore_result=None, restore_error=None
):
    import pathlib

    from bn.transport import BridgeError, BridgeInstance

    old = _fake_bridge_instance("keep1", pid=4242)  # noqa: F405
    old.meta["project_roots"] = [str(path) for path in project_roots]
    new = BridgeInstance(
        pid=5151,
        socket_path=pathlib.Path("/tmp/keep1.sock"),
        registry_path=pathlib.Path("/tmp/keep1.json"),
        plugin_name="bn_agent_bridge",
        plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        meta={},
        instance_id="keep1",
    )
    calls: list[tuple[str, dict]] = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, **kwargs):
        calls.append((op, dict(params or {})))
        if op == "shutdown":
            return {"ok": True, "result": {"shutting_down": True}}
        if op == "load_binary":
            return {
                "ok": True,
                "result": {
                    "loaded": True,
                    "path": params["path"],
                    "notes": [],
                    "targets": [],
                },
            }
        if op == "associate_project_roots":
            if restore_error is not None:
                raise BridgeError(restore_error)
            return {
                "ok": True,
                "result": restore_result
                if restore_result is not None
                else {
                    "instance_id": "keep1",
                    "associated": list(params["roots"]),
                    "skipped": [],
                },
            }
        raise AssertionError(f"unexpected op {op}")

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(
        bn.cli,
        "_send_request_to_instance",
        lambda inst, op, **kwargs: {
            "ok": True,
            "result": [{"filename": "/tmp/app.bin", "analysis_state": "full"}],
        },
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, **kw: True)
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new)
    return calls


def test_session_restart_restores_original_project_roots(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    calls = _restart_bridge_stubs(monkeypatch, [project])

    rc = bn.cli.main(["session", "restart", "keep1", "--format", "json"])

    assert rc == 0
    restore = [params for op, params in calls if op == "associate_project_roots"]
    assert restore == [{"roots": [str(project)]}]
    load = next(params for op, params in calls if op == "load_binary")
    assert "workdir" not in load
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["project_roots"] == [str(project)]
    assert "project_association_error" not in parsed
    assert [op for op, _ in calls].index("associate_project_roots") > [
        op for op, _ in calls
    ].index("load_binary")


def test_session_restart_reports_failed_association_restore_and_fails(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch, [project], restore_error="associate operation unavailable"
    )

    rc = bn.cli.main(["session", "restart", "keep1", "--format", "json"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "project association failed" in captured.err
    parsed = json.loads(captured.out)
    assert parsed["project_association_error"] == "associate operation unavailable"
    assert "project_roots" not in parsed


def test_session_restart_reports_failed_association_restore_in_text_mode(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch, [project], restore_error="associate operation unavailable"
    )

    rc = bn.cli.main(["session", "restart", "keep1"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "project association error: associate operation unavailable" in captured.out
    assert "pass -i keep1" in captured.out


def test_session_restart_warns_about_each_skipped_association(
    monkeypatch, capsys, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    _restart_bridge_stubs(
        monkeypatch,
        [project],
        restore_result={
            "instance_id": "keep1",
            "associated": [],
            "skipped": [
                {"path": str(project), "reason": "project directory does not exist"}
            ],
        },
    )

    assert bn.cli.main(["session", "restart", "keep1", "--format", "json"]) == 0
    assert "project directory does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------
# #694: an explicit empty instance selector is an error, never "list them all"
# --------------------------------------------------------------------------


def test_session_list_rejects_an_explicit_empty_instance(monkeypatch, capsys):
    # The selector filter is truthiness-based, so `-i ''` silently disabled it and
    # listed EVERY session -- the opposite of what an explicit selector asks for.
    listed = []
    monkeypatch.setattr(
        bn.cli,
        "list_instances",
        lambda: listed.append(True) or [_fake_bridge_instance("abc123")],  # noqa: F405
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    rc = bn.cli.main(["session", "list", "-i", ""])

    assert rc == 2
    assert listed == []
    assert "--instance is empty" in capsys.readouterr().err


def test_instance_list_rejects_an_explicit_empty_instance(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli, "list_instances", lambda: pytest.fail("must not enumerate instances")
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    assert bn.cli.main(["instance", "list", "-i", ""]) == 2
    assert "--instance is empty" in capsys.readouterr().err


def test_session_list_with_an_explicit_selector_still_filters(monkeypatch, capsys):
    monkeypatch.setattr(
        bn.cli,
        "list_instances",
        lambda: [_fake_bridge_instance("abc123"), _fake_bridge_instance("zz9999")],  # noqa: F405
    )
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: {})

    assert bn.cli.main(["session", "list", "-i", "zz9999", "--format", "json"]) == 0

    items = json.loads(capsys.readouterr().out)["items"]
    assert [item["instance_id"] for item in items] == ["zz9999"]


def test_session_restart_reopens_the_saved_database_not_the_raw_file_857(monkeypatch, capsys):
    """#857 review blocker -- silent data loss. Turning the sidecar preference
    off is right for a target deliberately opened raw, but it also stopped
    restoring the SAVED DATABASE of a target whose analysis lives there: load
    raw, annotate, `bn save` (sibling, or the cache copy on a read-only mount),
    restart -> the raw bytes came back re-analysed and un-annotated, at exit 0
    with no note.

    The captured filename cannot be the discriminator: `_save_database` restores
    `bv.file.filename` to the original path after every save on purpose (a save
    is persistence, not an identity move, #256/#285), so a saved raw-loaded
    target reports the RAW file. The bridge now reports `database_path` for it
    and restart reopens THAT -- named exactly, with the sidecar preference still
    off, so nothing is guessed."""
    from bn.transport import BridgeInstance
    old = type("FakeInstance", (), {
        "instance_id": "keep-me", "pid": 500,
        "socket_path": __import__("pathlib").Path("/tmp/old.sock"), "meta": {},
    })()
    new = BridgeInstance(
        pid=999, socket_path=__import__("pathlib").Path("/tmp/new.sock"),
        registry_path=__import__("pathlib").Path("/tmp/new.json"),
        plugin_name="bn_agent_bridge", plugin_version="0.1.0",
        started_at="2026-01-01T00:00:00Z", meta={}, instance_id="keep-me")
    calls = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0, instance_id=None, spawn_missing_named=False):
        calls.append((op, params))
        return {"ok": True, "result": {"path": (params or {}).get("path")}}

    monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [old])
    monkeypatch.setattr(bn.cli, "find_lifecycle_instance", lambda target: old)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: getattr(i, "instance_id", ""))
    monkeypatch.setattr(
        bn.cli, "_send_request_to_instance",
        lambda instance, op, params=None, target=None, **kw: {"ok": True, "result": [
            # Saved: filename is the raw file, the analysis is in the sibling DB.
            {"filename": "/fw/svc_a", "analysis_state": "full",
             "database_path": "/fw/svc_a.bndb"},
            # Saved on a read-only mount: the DB is the global cache copy. This
            # row was a FICTION in round 1 -- the bridge's cache branch did not
            # record the database, so no real payload ever looked like this and
            # the blocker stayed invisible here. It is truthful now, and
            # `test_save_records_the_CACHE_database_for_restart_857` in
            # test_bridge_dispatch.py is what proves the bridge emits it.
            {"filename": "/ro/svc_b", "analysis_state": "full",
             "database_path": "/home/u/.cache/bn/bndb/svc_b.deadbeefdeadbeef.bndb"},
            # Deliberately raw and never saved: no database backs it, so it must
            # come back raw -- the #753 fix this must not undo.
            {"filename": "/fw/svc_c", "analysis_state": "quick",
             "database_path": None},
        ]},
    )
    monkeypatch.setattr(bn.cli, "wait_for_teardown", lambda inst, timeout=5.0: True)
    monkeypatch.setattr(bn.cli, "spawn_instance", lambda instance_id=None: new)
    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)

    rc = bn.cli.main(["session", "restart", "keep-me", "--format", "json"])

    assert rc == 0
    loads = [params for op, params in calls if op == "load_binary"]
    assert [p["path"] for p in loads] == [
        "/fw/svc_a.bndb",
        "/home/u/.cache/bn/bndb/svc_b.deadbeefdeadbeef.bndb",
        "/fw/svc_c",
    ]
    # Never re-enabled: the database is NAMED, not guessed at.
    assert [p["prefer_bndb"] for p in loads] == [False, False, False]
    assert [p["quick"] for p in loads] == [False, False, True]
    assert len(json.loads(capsys.readouterr().out)["loaded"]) == 3


def test_the_probe_flag_reaches_the_WIRE_envelope_756(monkeypatch, tmp_path, capsys):
    """#859 review round 2: the transport half was the one link nothing
    exercised. Deleting the emission (`payload["idle_probe"] = True`) or the
    forwarding (`idle_probe=idle_probe` into `_send_request_to_instance`) left
    711 tests green while a real wire probe showed the field never reaching the
    envelope -- so the declaration could be honoured nowhere and nothing failed.

    This drives the real transport against a real Unix socket and asserts the
    bytes: `session list`'s probe carries `idle_probe: true` in the ENVELOPE,
    and an ordinary command over the same path does not. Both mutations die
    here, because both sit on the path from the declaring site to the wire."""
    import socketserver
    import threading
    from pathlib import Path as _P
    from bn.transport import BridgeInstance

    received: list[dict] = []

    class _H(socketserver.StreamRequestHandler):
        def handle(self):
            raw = self.rfile.readline()
            if not raw:
                return
            payload = json.loads(raw.decode("utf-8"))
            received.append(payload)
            self.wfile.write(json.dumps({
                "ok": True,
                "result": [{"selector": "netsvcd", "unsaved": True}],
                "bridge_identity": payload.get("_bridge_identity"),
            }).encode("utf-8"))

    class _S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    sock_path = tmp_path / "wire.sock"
    server = _S(str(sock_path), _H)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    try:
        # pid is THIS process so the SO_PEERCRED peer-pid check passes: the
        # server above really is the peer on the other end of the socket.
        inst = BridgeInstance(
            pid=os.getpid(), socket_path=_P(str(sock_path)),
            registry_path=tmp_path / "wire.json", plugin_name="bn_agent_bridge",
            plugin_version=bn.cli.VERSION, started_at="2026-01-01T00:00:00Z",
            meta={}, instance_id="wire-probe",
            # Without a token `_instance_identity` refuses before the socket is
            # touched -- and `session list` swallows that into
            # `unsaved_targets_unavailable` at rc 0, so the test would have
            # asserted nothing while looking green.
            instance_token="wire-probe-token")
        # Both namespaces: `session list` reads `cli.list_instances`, while
        # `choose_instance` (which `-i` resolution goes through) reads the one in
        # `bn.transport`'s own module globals.
        import bn.transport as _t
        monkeypatch.setattr(bn.cli, "list_instances", lambda **kw: [inst])
        monkeypatch.setattr(_t, "list_instances", lambda **kw: [inst])
        monkeypatch.setattr(bn.cli, "session_state", types.SimpleNamespace(
            read=lambda: {}, write=lambda **kw: None))

        assert bn.cli.main(["session", "list", "--format", "json"]) == 0
        assert bn.cli.main(["target", "list", "--format", "json", "-i", "wire-probe"]) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    capsys.readouterr()

    probes = [p for p in received if (p.get("params") or {}).get("strict")]
    ordinary = [p for p in received if p not in probes]
    assert probes, f"session list sent no strict probe: {received}"
    assert all(p.get("idle_probe") is True for p in probes), probes
    assert ordinary, f"target list sent nothing: {received}"
    # Absent, not false: the envelope is byte-identical to before for real work.
    assert all("idle_probe" not in p for p in ordinary), ordinary


# ---------------------------------------------------------------------------
# #676 item 11: BN_TARGET, the per-shell target story `-i` already had
# ---------------------------------------------------------------------------



def _capture_target(monkeypatch, argv, env=None):
    """Run *argv* through `main` and return the target each request carried."""
    import bn.cli

    seen: list = []

    def fake_send_request(op, *, params=None, target=None, timeout=30.0,
                          instance_id=None, spawn_missing_named=False):
        seen.append(target)
        if op == "list_targets":
            return {"ok": True, "result": [
                {"target_id": "1:1:1", "selector": "alpha.bin"},
                {"target_id": "1:1:2", "selector": "beta.bin"},
            ]}
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    bn.cli.main(argv)
    return [t for t in seen if t is not None]


def test_bn_target_supplies_the_selector_when_no_flag_is_passed(monkeypatch):
    """The whole point: fan-out stops costing `-t` on every single command.

    The sticky pin cannot do this job -- it lives in ~/.cache and is shared by
    every shell on the machine, so two agents in one project clobber each
    other, which is why the skill tells them not to use it. An environment
    variable is per-process-tree, so each agent's selector is invisible to the
    other by construction.
    """
    assert "beta.bin" in _capture_target(
        monkeypatch, ["function", "list"], {"BN_TARGET": "beta.bin"}
    )


def test_an_explicit_target_flag_beats_the_environment(monkeypatch):
    """`-t` is the override, not a duplicate of the export.

    An agent that exports a working target and then reaches for ONE other
    binary must not have to unset its shell to do it.
    """
    assert "alpha.bin" in _capture_target(
        monkeypatch, ["-t", "alpha.bin", "function", "list"], {"BN_TARGET": "beta.bin"}
    )


def test_an_empty_bn_target_is_refused_rather_than_resolved(monkeypatch, capsys):
    """An UNSET shell variable exports as the empty string, and the bridge
    collapses an empty selector to the focused GUI view with no count check.

    So the dangerous shape is not a wrong name, it is `export BN_TARGET=$SEL`
    where SEL was never assigned: the command would then act on whichever tab
    happened to have focus while LOOKING like it had been told a target. The
    existing empty-selector refusal covers it, and this pins that the env path
    reaches that refusal rather than bypassing it.

    Round 1 blocker: the first cut asserted only `main(...) == 2`, and a 2 is
    what this argv produces at base too -- from the no-targets-open error on
    the implicit-target path, which the env default does not even reach. The
    exit code alone cannot tell the two apart, so the assertions are now the
    two things only the refusal produces: its own words, and the fact that
    NOTHING was sent. The base path sends `list_targets` before it fails, so
    the silence is what separates "refused the selector" from "asked the
    bridge and got an unrelated error".

    Round 4 review: those words now name the SOURCE rather than the `--target`
    flag nobody passed, so the assertion follows them there -- and it is a
    stronger discriminator, because only the env path can produce them.
    """
    import bn.cli

    sent: list[str] = []

    def fake_send_request(op, **kwargs):
        sent.append(op)
        return {"ok": True, "result": []}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("BN_TARGET", "   ")

    assert bn.cli.main(["function", "list"]) == 2

    assert "BN_TARGET is exported but empty" in capsys.readouterr().err
    assert sent == [], (
        "the empty selector must be refused BEFORE anything reaches the "
        f"bridge; these ops were sent instead: {sent}")


def _close_run(monkeypatch, argv, env=None, selectors=("alpha.bin", "beta.bin")):
    """Run *argv* through `main`, returning `(rc, [(op, target), ...])`.

    Pins a nonexistent instance so no probe can reach a live bridge even if a
    future change moved one of these paths onto the transport.
    """
    import bn.cli

    sent: list[tuple[str, object]] = []

    def fake_send_request(op, *, params=None, target=None, **kwargs):
        sent.append((op, target))
        if op == "list_targets":
            return {"ok": True, "result": [
                {"target_id": f"1:1:{i}", "selector": sel}
                for i, sel in enumerate(selectors, start=1)]}
        return {"ok": True, "result": {"closed": [{"selector": target}], "count": 1}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    monkeypatch.setenv("BN_INSTANCE", "prfleet-nonexistent-889")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return bn.cli.main(argv), sent


def test_a_bare_destructive_close_never_takes_the_environment_default(monkeypatch):
    """A destructive op must not be steered by an AMBIENT selector.

    `close`'s handler already nulls a sticky-injected target for exactly this
    reason: a bare `bn close` must not silently tear down whichever target
    some earlier `bn target use` happened to pin. An exported selector is the
    same ambient value from a different source, so honouring it there
    reintroduces the hazard through the other door -- the one thing #676 item
    11 must not do.

    Round 2 blocker: the guarantee was implemented as "a `required=True`
    target option never takes the env default", and NO command in the tree
    sets `required=True` -- every `--target` comes from `@command(target=True)`
    with `required=False`. So the protection covered zero commands and a bare
    `close` closed the exported selector at rc 0. This test drives the real
    command, so it cannot pass while that is true.

    Two open targets, so a bare close has to refuse rather than fall through
    to the legitimate single-open case.
    """
    rc, sent = _close_run(monkeypatch, ["close"], {"BN_TARGET": "beta.bin"})

    assert [op for op, _ in sent] == ["list_targets"], (
        "a bare destructive close must not act on the exported selector; "
        f"it sent {sent}")
    assert rc == 2


def test_the_sticky_pin_and_the_environment_default_are_refused_alike(monkeypatch):
    """The two ambient sources must behave identically at the destructive op,
    or the safer-looking one is the one that surprises you.

    Pinned as a PAIR: asserting only the env half would stay green if the
    sticky guard were deleted, and the whole argument for the env default is
    that it is the sticky pin's equal minus the cross-agent clobber.
    """
    from bn import session_state

    monkeypatch.setattr(session_state, "read", lambda: {"target": "beta.bin"})
    sticky_rc, sticky_sent = _close_run(monkeypatch, ["close"])

    monkeypatch.setattr(session_state, "read", lambda: {})
    env_rc, env_sent = _close_run(monkeypatch, ["close"], {"BN_TARGET": "beta.bin"})

    assert (sticky_rc, [op for op, _ in sticky_sent]) == (2, ["list_targets"])
    assert (env_rc, [op for op, _ in env_sent]) == (sticky_rc,
                                                    [op for op, _ in sticky_sent])


def test_an_explicit_target_still_closes_that_target(monkeypatch):
    """Must-not-fire twin: the refusal is about AMBIENCE, not about `close`.

    An explicit `-t` is the caller naming the target, and it must still work
    -- otherwise the fix above would have made `close` unusable rather than
    safe.
    """
    rc, sent = _close_run(monkeypatch, ["-t", "beta.bin", "close"],
                          {"BN_TARGET": "alpha.bin"})

    assert ("close_binary", "beta.bin") in sent, sent
    assert rc == 0


def test_an_empty_export_is_refused_by_close_rather_than_discarded(monkeypatch):
    """The two guarantees must not cancel each other out.

    Round 3 made an exported selector AMBIENT so a bare destructive `close`
    cannot be steered by it. An EMPTY export is not a selector, and marking
    it ambient too made `close` -- which DISCARDS an ambient target -- throw
    it away: with one target open the bare close then fell through to the
    single-open auto-pick and tore that target down at exit 0, where it had
    refused with nothing sent.

    That is worse than the hazard the round-3 fix removed: the empty export
    is the shape the whole refusal exists for (`export BN_TARGET=$SEL` where
    SEL was never assigned), and the destructive command is where it matters
    most. ONE target open, because that is the configuration where a
    discarded selector silently succeeds instead of hitting the multi-target
    refusal.
    """
    rc, sent = _close_run(monkeypatch, ["close"], {"BN_TARGET": "   "},
                          selectors=("only.bin",))

    assert sent == [], (
        "an empty export must reach the empty-selector refusal, not be "
        f"discarded as ambient; these ops were sent: {sent}")
    assert rc == 2


def test_an_empty_pin_is_refused_by_close_exactly_like_an_empty_export(monkeypatch):
    """Round 4 fixed the empty EXPORT. The pin is the other half of the pair.

    The pin reaches the same state by the same accident: `bn target use
    "$SEL"` with SEL unset writes an empty pin and exits 0 (measured --
    `_target_matches` answers True for "", so the pre-write validation lets it
    through and `session_state.update(target="")` runs). Reading that back as
    "no pin" made a bare destructive close fall through to the single-open
    auto-pick and tear that target down at rc 0 -- the exact behaviour round 4
    removed on the export path, still shipping on the other one, while
    runtime.md derives the empty-value guarantee from BOTH sources being
    ambient.

    Both spellings, because they failed differently: "" was never filled (so
    the command looked unpinned) and whitespace WAS filled and then marked
    ambient, which `close` discards. ONE target open -- the configuration
    where a discarded selector succeeds silently instead of hitting the
    multi-target refusal.
    """
    from bn import session_state

    monkeypatch.setattr(session_state, "read", lambda: {"target": ""})
    empty = _close_run(monkeypatch, ["close"], selectors=("only.bin",))

    monkeypatch.setattr(session_state, "read", lambda: {"target": "   "})
    blank = _close_run(monkeypatch, ["close"], selectors=("only.bin",))

    assert empty == (2, []), (
        "an empty pin must reach the empty-selector refusal, not read as no "
        f"pin at all and let a bare close take the sole target: {empty}")
    assert blank == (2, []), (
        "a whitespace pin is not a selector either, so `close` must not "
        f"discard it as ambient and auto-pick instead: {blank}")


def test_an_empty_ambient_selector_says_which_source_it_came_from(
        monkeypatch, capsys):
    """"…or omit --target" is advice the caller has already followed.

    Nobody passed a flag: the empty value arrived from the environment or
    from the pin, and the two are cleared in completely different ways. A
    refusal that blames `--target` sends the reader looking for an argument
    they never wrote, and one that names neither source leaves them
    re-running the same command -- with the pin, re-reading a file they have
    to know exists. `main` already discloses provenance this way for a sticky
    INSTANCE on a dead bridge; this is the same disclosure for the target.
    """
    from bn import session_state

    _close_run(monkeypatch, ["close"], {"BN_TARGET": "   "},
               selectors=("only.bin",))
    env_err = capsys.readouterr().err

    # The export wins over the pin, so it has to go before the pin is asked.
    monkeypatch.delenv("BN_TARGET")
    monkeypatch.setattr(session_state, "read", lambda: {"target": ""})
    _close_run(monkeypatch, ["close"], selectors=("only.bin",))
    pin_err = capsys.readouterr().err

    assert "BN_TARGET" in env_err and "--target" not in env_err, env_err
    assert "bn target clear" in pin_err and "--target" not in pin_err, pin_err


def test_a_broken_ambient_default_does_not_break_the_cleanup_verb(monkeypatch):
    """An unneeded default must not fail the command that needs no default.

    Refusing an empty ambient selector protects the resolution it corrupts.
    `bn close --all` and `bn close <path>` resolve nothing -- the caller said
    what to close -- so the broken default is never consulted, and failing
    them turns a stale shell variable into "the cleanup verb no longer runs".
    Worse, the refusal they produced came from close's operand-conflict guard
    ("Pass --target or --all, not both"), naming a flag the caller never
    typed, because the empty value had been filled into `args.target` where
    that guard counts it as a GIVEN operand.

    Both spellings and both ambient sources, because the guard that produced
    the wrong refusal is per-operand and the two sources fill the same field.
    """
    from bn import session_state

    all_env = _close_run(monkeypatch, ["close", "--all"], {"BN_TARGET": "   "},
                         selectors=("only.bin",))
    path_env = _close_run(monkeypatch, ["close", "/tmp/bn-not-a-real-target"],
                          {"BN_TARGET": "   "}, selectors=("only.bin",))

    monkeypatch.delenv("BN_TARGET")
    monkeypatch.setattr(session_state, "read", lambda: {"target": ""})
    all_pin = _close_run(monkeypatch, ["close", "--all"], selectors=("only.bin",))
    path_pin = _close_run(monkeypatch, ["close", "/tmp/bn-not-a-real-target"],
                          selectors=("only.bin",))

    for label, (rc, sent) in (("--all under an empty export", all_env),
                              ("a path under an empty export", path_env),
                              ("--all under an empty pin", all_pin),
                              ("a path under an empty pin", path_pin)):
        assert rc == 0 and [op for op, _ in sent] == ["close_binary"], (
            f"close with {label} names what to close and consults no default, "
            f"so it must still run; got rc={rc} sent={sent}")


def _fanout_pairs(monkeypatch, capsys, argv, env=None, sticky=None):
    """Run a fan-out read; return its (instance, target) pairs and auto-expansion.

    Two instances, one of them holding TWO targets: that is the only shape
    where "apply the selector to every instance" and "survey every target"
    produce different answers, so it is the shape this question has to be
    asked in.
    """
    import json as _json
    import types

    import bn.cli

    insts = [types.SimpleNamespace(instance_id="solo"),
             types.SimpleNamespace(instance_id="multi")]
    monkeypatch.setattr(bn.cli, "list_instances", lambda: insts)
    monkeypatch.setattr(bn.cli, "instance_selector", lambda i: i.instance_id)
    monkeypatch.setattr(bn.cli.session_state, "read", lambda: dict(sticky or {}))

    def fake_send_request(op, *, params=None, target=None, instance_id=None,
                          **kwargs):
        if op == "list_targets":
            rows = ([{"target_id": "m-t1"}, {"target_id": "m-t2"}]
                    if instance_id == "multi" else [{"target_id": "solo-t1"}])
            return {"ok": True, "result": rows}
        return {"ok": True, "result": {"kind": "sections", "items": [], "total": 0}}

    monkeypatch.setattr(bn.cli, "send_request", fake_send_request)
    # Each sub-case starts from NO export. `monkeypatch.setenv` lives for the
    # whole test, and the export BEATS the pin, so an earlier exported
    # sub-case would otherwise still be in the environment for a later pinned
    # one -- the pin would never be read and `pinned == exported` would be
    # comparing the export to itself.
    monkeypatch.delenv("BN_TARGET", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    assert bn.cli.main(argv) == 0
    payload = _json.loads(capsys.readouterr().out)
    return (sorted((row["instance"], row.get("target"))
                   for row in payload["instances"]),
            payload.get("auto_expanded_instances"))


def test_an_ambient_selector_does_not_narrow_an_all_instances_survey(
        monkeypatch, capsys):
    """The OTHER consumer of the ambient marker, and the one nothing pinned.

    `--all-instances` asks two different questions of a `-t`: an explicit one
    is a choice, applied to every instance, while an AMBIENT one is not -- it
    must not suppress the multi-target auto-survey (#368 facet 1, already
    pinned for the sticky pin in test_cli_core.py). Making `BN_TARGET` ambient
    in round 3 moved the export from the first answer to the second, which is
    a real change to a read surface and was pinned by nothing: the export half
    of this test passes at the round-2 head and at this one, for OPPOSITE
    reasons, if you only assert an exit code.

    So the three cases are asserted together, and the export is asserted to
    equal the SURVEY, not merely "not an error". runtime.md states exactly
    this ("a bare read under `--all-instances` still surveys every target
    rather than treating the export as a chosen one. Where you need the export
    to mean 'this exact target, no survey' ... pass `-t`"), and the pin is
    carried along because the two ambient sources must not drift apart.
    """
    survey = ([("multi", "m-t1"), ("multi", "m-t2"), ("solo", "solo-t1")],
              ["multi"])

    argv = ["sections", "--all-instances", "--format", "json"]
    exported = _fanout_pairs(monkeypatch, capsys, argv,
                             env={"BN_TARGET": "beta.bin"})
    pinned = _fanout_pairs(monkeypatch, capsys, argv,
                           sticky={"target": "beta.bin"})
    explicit = _fanout_pairs(monkeypatch, capsys,
                             ["-t", "beta.bin", *argv],
                             env={"BN_TARGET": "alpha.bin"})

    assert exported == survey, (
        "an exported BN_TARGET is ambient: --all-instances must still survey "
        f"every target rather than apply it per instance; got {exported}")
    assert pinned == exported, (
        "the sticky pin and the export are the same ambient value from two "
        f"sources and must fan out alike; pin={pinned} export={exported}")
    assert explicit == ([("multi", "beta.bin"), ("solo", "beta.bin")], None), (
        "an explicit -t IS a choice: it must apply to every instance and "
        f"suppress the survey, even with a different value exported; got {explicit}")


def test_a_broken_ambient_default_does_not_break_a_fan_out_survey(
        monkeypatch, capsys):
    """A survey consults no selector, so a broken one cannot corrupt it.

    `--all-targets` reads every open target by definition, and under
    `--all-instances` an ambient value is already not the explicit choice
    that would narrow the run. Neither resolves anything from the default,
    so refusing them because a shell variable is empty is not a safety
    guarantee -- it is a straight regression: at base an empty pin was never
    filled at all and the same survey returned its rows.

    Asserted as EQUALITY with the no-ambient run rather than "rc 0", because
    the failure this guards against is the survey silently narrowing, not
    only erroring, and both spellings of both sources are carried because
    the refusal they hit was one shared check.
    """
    for argv in (["sections", "--all-instances", "--format", "json"],
                 ["sections", "--all-targets", "--format", "json"]):
        clean = _fanout_pairs(monkeypatch, capsys, argv)
        for label, kwargs in (
                ("an empty export", {"env": {"BN_TARGET": ""}}),
                ("a whitespace export", {"env": {"BN_TARGET": "   "}}),
                ("an empty pin", {"sticky": {"target": ""}}),
                ("a whitespace pin", {"sticky": {"target": "   "}})):
            got = _fanout_pairs(monkeypatch, capsys, argv, **kwargs)
            assert got == clean, (
                f"{argv[1]} under {label} consults no selector, so it must "
                f"read exactly what it reads with none: got {got}, "
                f"expected {clean}")


def test_bn_target_is_scrubbed_from_the_test_environment():
    """An ambient BN_TARGET would redirect every test that passes no selector.

    The multi-target refusals exist precisely to fire when nothing was given;
    an exported value stops them firing and the suite goes green for the wrong
    reason. Same argument that put BN_INSTANCE on this list.
    """
    from conftest import SCRUBBED_ENV_VARS

    assert "BN_TARGET" in SCRUBBED_ENV_VARS


def test_the_runtime_reference_denies_no_environment_default_the_cli_honours(
        monkeypatch):
    """runtime.md's routing ladder is where an agent learns how `-t` resolves,
    and it said -- in bold -- that `BN_TARGET` does not exist, while the root
    parser had started defaulting `-t` from it.

    A doc that denies a shipped mechanism is worse than one that omits it: the
    agent reads the denial, keeps paying `-t` on every command, and the
    clobber hazard the variable exists to remove stays in place. The sibling
    guard `test_every_flag_the_reference_denies_really_does_not_exist` covers
    the same failure for FLAGS and cannot see this one, because its token
    pattern only matches `--spellings`.

    Both halves are asserted in one test on purpose. The doc half alone is
    satisfiable by deleting the sentence while the mechanism is reverted, and
    the behaviour half is what the sibling tests above already pin; it is
    their CONJUNCTION that is the contract.
    """
    import re
    from pathlib import Path

    from test_skill_reference_drift import _SENTENCE, bound_absence_claims

    # The mechanism is really shipped: the exported value reaches the request.
    assert "beta.bin" in _capture_target(
        monkeypatch, ["function", "list"], {"BN_TARGET": "beta.bin"})

    # The env analogue of `_DOC_FLAG`/`_ABSENCE_CLAIM`: same denial forms, an
    # env-variable token instead of a long flag. `bound_absence_claims` is
    # shared rather than reimplemented -- it is already parameterised over the
    # token and claim patterns for exactly this reason.
    env_name = re.compile(r"BN_[A-Z][A-Z0-9_]*")
    env_absence = re.compile(
        r"`(BN_[A-Z0-9_]+)`[^.`]{0,30}do(?:es)? not exist"
        r"|there is no `(BN_[A-Z0-9_]+)`"
        r"|no such `?(BN_[A-Z0-9_]+)`?"
        r"|no `(BN_[A-Z0-9_]+)` (?:variable|environment variable|default)")

    doc = Path(__file__).resolve().parents[1] / "skills/bn/reference/runtime.md"
    denied: set[str] = set()
    for sentence in _SENTENCE.split(doc.read_text(encoding="utf-8")):
        denied |= bound_absence_claims(sentence, env_name, env_absence)

    assert "BN_TARGET" not in denied, (
        "runtime.md denies BN_TARGET while the root `-t` default reads it")
