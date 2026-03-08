---
name: binary-ninja-cli
description: Use the local bn CLI for Binary Ninja reversing work when a Binary Ninja GUI session is already open. Prefer this skill for decompilation, function search, IL/disassembly, xrefs, type inspection, struct field edits, previewed mutations, and inline Python execution through the bn bridge.
---

# Binary Ninja CLI

Use this skill when the user wants reverse-engineering work against an already-open Binary Ninja database and the local `bn` CLI is available.

## Workflow

1. Check bridge state first:

```bash
bn doctor
bn target list
```

2. Pick a target:
- If there is exactly one open BinaryView, target-scoped commands can omit `--target` entirely.
- Read commands also default to `--target active`.
- Otherwise prefer the `selector` field from `bn target list`.
- If multiple targets are open, pass an explicit `--target`.

3. Use `bn` directly when it is on PATH.
- If running from this repo without the global tool installed, use `uv run bn`.

## High-Value Read Commands

```bash
bn function list
bn function search attachment
bn decompile sample_track_floor_height_at_position
bn il sample_track_floor_height_at_position
bn disasm sample_track_floor_height_at_position
bn xrefs sample_track_floor_height_at_position
bn types --query Player
bn strings --query follow
bn bundle function sample_track_floor_height_at_position --out /tmp/floor.json
```

## Python Escape Hatch

Use inline Python for one-off Binary Ninja inspection that is awkward to express as a built-in command:

```bash
bn py exec --code "print(hex(bv.entry_point)); result = {'functions': len(list(bv.functions))}"
```

Use `--stdin` for larger snippets. Use `--script <file>` only for real files.

## Mutation Workflow

Prefer preview first:

```bash
bn struct field set Player 0x308 movement_flag_selector uint32_t --preview
bn symbol rename sub_401000 player_update --preview
bn proto set sub_401000 "int __cdecl player_update(Player* self)" --preview
```

For struct previews, inspect:
- `results`
- `affected_types`
- `affected_functions`

If a struct edit is already identical, preview may report `changed: false` with `No effective change detected`.

## Practical Guidance

- Prefer `bn` over MCP for shell-driven decompilation, search, bundles, and large outputs.
- Use `--out` when output may be long or when you want a stable artifact.
- Use `bn target list` again if `active` is ambiguous.
- If multiple targets are open, be explicit and pass `--target <selector>`.
- If `bn target list` is empty, the Binary Ninja plugin is not live. Check `bn plugin install`, then reload Binary Ninja plugins or restart Binary Ninja.
