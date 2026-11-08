---
name: bn-projects
description: "Opt-in Binary Ninja Commercial/Ultimate project workflows: share one owner bridge, load and save individual databases, initialize firmware projects, discover libraries, and link imports. Install separately with bn skill install projects."
---

# bn-projects

Project APIs require **Binary Ninja Commercial or Ultimate**. This guidance is **opt-in regardless of license**: `bn skill install` excludes `bn-projects`; `bn skill install projects` installs only `bn-projects`. Installing the skill does not check or grant a license, change Binary Ninja capabilities, or start a project. Use `bn` for ordinary reads, mutations, and lifecycle, `bn-re` or `bn-vr` for methodology, and `bn-kernel` for retained-Python reads.

With this workflow opted in, when entering a supported extracted Linux or Android firmware root, automatically create or open its project with `bn project init`, then import the program explicitly before loading it. Do not substitute raw-path `bn load` for project loading.

## Command index

- **Projects** — `project info/files/load/import/export/delete/close/libs/link` → `SKILL.md`

## One owner bridge, individual databases

A project (`.bnpr`) groups files; it is not a single loaded analysis database. All agents share the **one bridge that owns the project** and load individual files through it. Never open the same project or `.bndb` in a second bridge process: simultaneous database writers can corrupt it.

`bn project open <path.bnpr>` opens or attaches to the owner and pins that instance. `bn project attach <path.bnpr>` only attaches to an already-running owner and errors if none exists. `bn project init` also opens or attaches. All three change the shared sticky instance pin: coordinate them before fan-out, not from concurrent workers. Use the owner instance ID actually returned, which may differ from a requested new name. On subsequent commands pass `-i <owner-instance-id>` explicitly, and `-t <selector>` for target commands. Do not change shared `instance use` or `target use` pins during concurrent work. Keep `BN_CACHE_DIR` consistent across every caller so they find the same owner registry.

```bash
bn project open /path/to/work.bnpr --instance-id requested-name
bn project attach /path/to/work.bnpr
bn project info
bn project files --folder usr/bin --flat
bn project load usr/bin/app
bn -i <owner-instance-id> target list
bn -i <owner-instance-id> -t <selector-from-target-list> save
```

The unqualified project examples assume the owner pin established above; concurrent workers must add `-i <owner-instance-id>`.

- `project info` reports project name, path, ID, file/folder counts, and open targets. `project files` lists project entries; `--folder` filters a path and `--flat` omits subfolders.
- `project load`, `project export`, and `project delete` select a project file by full file ID, unique ID prefix of at least eight characters, path-in-project, or unique bare name. These are **project-file selectors**, not `-t` selectors.
- Loaded target selectors come from `target list`: the shortest unique path-in-project suffix, a full project file ID, path-in-project (with or without leading `/`), or file name. A `.bndb`-stripped name is a weak alias. ID prefixes are **not** a `-t` form. The view's `filename` can be an opaque blob path; do not treat it as the source path or load blobs directly out of `.bnpr`.
- `project load` prefers the saved database; `--no-bndb` chooses raw bytes, `--quick` skips analysis, and `--no-link` skips automatic linking. Check `analysis_state` and run `bn refresh` before relying on complete function discovery from a quick load.
- `bn save -t <selector>` with no path creates the first tracked `<name>.bndb` beside the raw project file; later saves snapshot that database. Save each changed target, not merely the project container.

```bash
bn project load <selector> [--no-bndb] [--quick] [--no-link]
bn project import <path> [--folder P] [--name N] [--description D]
bn project export <selector> <dest>
bn project delete <selector>
bn project close [--force]
```

Import adds a file to the project, export copies a project file to the destination, and delete removes the selected project file. Closing refuses unsaved targets unless `--force` explicitly discards their work. Do not force-close another agent's work. Headless `session stop` / `session restart` likewise refuse unsaved targets unless `--discard-unsaved` is intentional. For rootfs projects, restart replays project targets and re-links symbols when library metadata is present.

## Firmware initialization, import, and libraries

```bash
bn project init [ROOT] [--name N] [--no-libs] [--dry-run] [--instance-id ID]
bn -i <owner-instance-id> project import /path/to/rootfs/usr/bin/app --folder usr/bin
bn -i <owner-instance-id> project load usr/bin/app
bn project libs
bn project libs --sync [--root R] [--dry-run]
```

`init` defaults to the current directory and creates or opens `<parent>/<root-name>.bnpr`, falling back to a root-hash project under the cache projects directory if the parent is unwritable. `--name` sets the project name, `--no-libs` skips library import, and `--dry-run` reports the proposed work without modifying disk. Re-running initialization opens the same project and skips already-imported libraries.

Initialization syncs libraries; **it does not import executables**. The explicit `project import` step above must precede `project load`. Loading then automatically links imports against the project's library inventory unless `--no-link` is used. `project libs` inspects that inventory; `--sync` re-scans and imports new libraries. `--root` and `--dry-run` apply only with `--sync`.

Lookup is **best-effort, not full dynamic-loader emulation**, and supports Linux and Android roots, not arbitrary firmware operating systems. Discovery uses startup `LD_LIBRARY_PATH` assignments, `ld.so.conf`, platform default directories, and Android library directories. Binary-only `RPATH`/`RUNPATH` directories are **not inventoried**: they rank matches among libraries already inventoried. Missing or ambiguous providers are not proof a dependency is absent; unsupported loader behavior, ABI mismatch, missing exports, and optional weak symbols can explain unresolved imports. Inspect results rather than assuming every runtime library or import was recovered.

## Linking and reading persisted assignments

```bash
bn project link -t <selector> --preview --format json
bn project link -t <selector>
bn project link -t <selector> --relink
bn -i <owner-instance-id> -t <selector> save
```

`project link` is a mutation. Preview the plan, inspect candidate providers, apply, read back, and save. `--preview` does not write; `--relink` permits reassigning locations already linked. Verification checks each assignment's resolved library. Inspect **`locations[]`** from `bn project link -t <selector> --preview --format json`: `action: already_linked` reports an existing assignment, while `linked` or `relinked` describes a proposed write, not proof it was applied. `bn imports` reports import namespace/version attribution, **not persisted project-library links**. Unresolved imports remain investigation leads, not evidence of a missing library by themselves.

An unreadable mutation counter produces an unmeasured summary (`op_count: null`, exit 4), not a fabricated zero; a reported mutation failure takes precedence at exit 3.

Project links do not yet make taint, callgraph, or xrefs traverse library implementations. Those analyses remain scoped to the selected binary; taint uses external-call models or its conservative unknown-call behavior at imported calls. Loading and analyzing the linked library does not remove that boundary.

## Retained Python binding

First initialize, explicitly import, and load the project file in the shell as above. Bootstrap `bn-kernel` using its installed `bootstrap.py` instructions, then bind the **returned owner instance** and loaded project's advertised target selector:

```python
s = bn_kernel.session(instance="<owner-instance-id>", target="usr/bin/app")
info = await s.info()
```

Use `target list` to choose the actual loaded selector if a database sidecar changes it. Confirm the loaded target's project identity before trusting rows; its blob filename need not equal `usr/bin/app`. The retained kernel does not inherit later shell exports. Use the same `BN_CACHE_DIR` as the owner, and do not change it in a shared kernel while other sessions use it: it is process-wide, not session-local. Use isolated kernels or explicit CLI routing for concurrent targets that cannot safely share the kernel.
