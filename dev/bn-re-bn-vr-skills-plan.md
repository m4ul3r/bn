# Plan: Add bn-re and bn-vr methodology skills

## Context

The `bn` skill is a tool interface — it teaches how to call CLI commands. Users want two methodology skills layered on top:
- **bn-re**: Reverse engineering methodology (static analysis with `bn`)
- **bn-vr**: Vulnerability research methodology (finding bugs with `bn`)

These are pure methodology guides with no new CLI tools. They reference `bn` commands in examples but focus on *what to do and why*.

---

## Changes

### 1. Create `skills/bn-re/SKILL.md`

Reverse engineering methodology skill. Content should cover:
- **Approaching an unknown binary**: Start with `bn target info` for arch/platform/entry, then `bn imports` and `bn strings` for orientation, then `bn function list` to survey scope
- **Function triage**: How to identify important functions (entry point, large functions, functions with many xrefs, string references)
- **Iterative type recovery**: Start with renames, then proto/retype locals, then struct creation — workflow for building up types incrementally
- **Call graph analysis**: Using `bn xrefs` and `bn callsites` to trace data flow and understand relationships
- **Naming conventions**: When to rename, what naming style to use, using `bn symbol rename --preview`
- **Struct reconstruction**: Reading decompilation to identify field accesses, using `bn struct show`/`bn struct field set` to build structs

Frontmatter trigger: activate when user asks to understand/reverse/analyze a binary, identify functions, recover types, or reconstruct structures.

### 2. Create `skills/bn-vr/SKILL.md`

Vulnerability research methodology skill. Content should cover:
- **Attack surface identification**: `bn imports` to find dangerous functions (strcpy, sprintf, malloc/free, system, exec*), `bn strings` for format strings
- **Input tracing**: Using `bn xrefs` to trace from input sources (read, recv, fgets, argv) to sinks (memcpy, strcpy, system)
- **Common vulnerability patterns**: What to look for in decompiled output:
  - Buffer overflows: fixed-size stack buffers with unbounded copies
  - Format strings: user-controlled format arguments
  - Integer overflows: arithmetic before allocation/bounds checks
  - Use-after-free: free/reuse patterns in callgraphs
  - Off-by-one: loop bounds, fence-post errors
- **Systematic audit workflow**: Function-by-function vs. pattern-based approaches
- **Taint analysis**: Manual taint tracking through `bn decompile` + `bn callsites` — tracing user input through function calls
- **Reporting findings**: What to capture (function, address, condition, impact)

Frontmatter trigger: activate when user asks to find vulnerabilities, audit for bugs, check security, identify exploitable conditions, or analyze attack surface.

### 3. Update `bn skill install` to install all skills

**File**: `src/bn/cli.py` (~line 1055, `_skill_install`)

Change from installing only `skills/bn` to iterating over all subdirectories in `skills/`:

```python
def _skill_install(args: argparse.Namespace) -> int:
    skills_root = repo_root() / "skills"
    results = []
    for source in sorted(skills_root.iterdir()):
        if not source.is_dir() or not (source / "SKILL.md").exists():
            continue
        dest = args.dest / source.name if args.dest else claude_skills_dir() / source.name
        _install_tree(source, dest, mode=args.mode, force=args.force)
        results.append({"skill": source.name, "source": str(source), "destination": str(dest)})
    ...
```

**File**: `src/bn/paths.py` — Add `repo_root()` import/usage and `claude_skills_dir()` to the imports in cli.py. Remove the single-skill `skill_source_dir()` / `skill_install_dir()` if no longer needed elsewhere.

### 4. Update `skills/bn/SKILL.md` — cross-reference new skills

Add a brief note to the bn skill mentioning that `bn-re` and `bn-vr` exist for methodology guidance. Keep it minimal — one or two lines.

---

## File structure after changes

```
skills/
├── bn/
│   └── SKILL.md          (existing, minor update)
├── bn-re/
│   └── SKILL.md          (new)
└── bn-vr/
    └── SKILL.md          (new)
```

## Critical files

- `skills/bn-re/SKILL.md` — new RE methodology skill
- `skills/bn-vr/SKILL.md` — new VR methodology skill
- `skills/bn/SKILL.md` — cross-reference update
- `src/bn/cli.py` — multi-skill install
- `src/bn/paths.py` — path helpers

## Verification

1. `bn skill install` installs all three skills (bn, bn-re, bn-vr) to `~/.claude/skills/`
2. `ls ~/.claude/skills/` shows `bn`, `bn-re`, `bn-vr`
3. Each SKILL.md has valid frontmatter (name, description)
4. `pytest tests/test_cli.py` passes
