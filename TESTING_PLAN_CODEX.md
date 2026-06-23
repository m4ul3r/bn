# Codex Testing Plan

Date: 2026-06-12

Scope: follow-up plan for the usability issues found in `CODEX_CODE_REVIEW_31337.md`.

## Goal

Make the `bn` CLI easier to test, easier to script, and less surprising for agent-driven reverse-engineering workflows. The priority is to turn the current audit findings into reproducible regression coverage before changing behavior.

## Plan

### 1. Restore Reproducible Stress Coverage

Fix `tests/stress/run_stress.sh` so it can run from a clean checkout.

Options:

- Restore the missing `tests/fixtures` directory and its build system.
- Or rewrite the stress script to compile temporary fixtures from sources already in the repo.

Acceptance criteria:

- `bash tests/stress/run_stress.sh` runs without missing-fixture errors.
- The script covers session lifecycle, multi-instance routing, preload, load/save/close, mutation isolation, parallel reads, and multi-target selection.
- Generated fixture outputs are ignored or placed in a temp directory.

### 2. Improve Quick-Load State Discoverability

The coarse quick/full state is already exposed by `bn target info` through
`analysis_state`. Improve discoverability around that existing signal instead
of treating it as absent.

Implementation targets:

- `bn target list`
- `bn target info`
- Docs/examples for `bn load --quick`, `bn refresh`, and `bn strings`

Acceptance criteria:

- `target list` includes the same quick/full analysis state that `target info`
  already exposes, or the docs make the `target info` check prominent.
- `strings` behavior remains clear when full analysis has not run.
- Examples explain which commands work immediately after quick load and which
  require `refresh`.

### 3. Normalize Or Document JSON Shapes

Inventory commands that return bare arrays versus object envelopes.

Start with:

- `callsites`
- `types`
- Other list-like commands found by parser/formatter review

Acceptance criteria:

- Every command documents its JSON top-level shape.
- If a shape changes, backward compatibility is considered explicitly.
- Agent-facing examples show robust parsing.

### 4. Audit Batch Operation Parity

Batch `set_comment` already supports either `function` or `address`, matching
interactive `comment set --function` / `comment set --address`. Keep this item
as a parity audit for the remaining batch operations rather than a known
`set_comment` bug.

Audit targets:

- `set_comment`
- `delete_comment`
- `set_prototype`
- `local_rename`
- `local_retype`
- `struct_field_set`
- `struct_field_rename`
- `struct_field_delete`
- `types_declare`

Known supported comment manifest shape:

```json
{
  "target": "<selector>",
  "ops": [
    {
      "op": "set_comment",
      "function": "<function identifier>",
      "comment": "..."
    }
  ]
}
```

Acceptance criteria:

- Batch operation docs match actual accepted fields.
- Invalid manifests produce clean errors and roll back.
- Unit tests cover parity and invalid mixed input for each batch operation.

### 5. Improve Type Lookup Ergonomics

Make failed type lookup more actionable.

Implementation targets:

- `types show`
- `struct show`
- README and skill examples

Acceptance criteria:

- Missing type errors suggest `bn types --query <name>`.
- Examples avoid assuming target-specific aliases such as `int32_t` exist.
- Tests cover the improved error message.

### 6. Add A Sanitized Command-Surface Smoke Audit

Create a repo-local smoke test that enumerates registered commands and exercises safe workflows without leaking target content.

Suggested script:

- `tests/stress/run_command_surface_audit.py`

Requirements:

- Enumerates `_COMMANDS`.
- Compiles or generates tiny local fixtures.
- Uses isolated `--instance` IDs.
- Uses temp dirs for plugin/skill install checks, BNDB saves, bundles, byte output, and manifests.
- Uses preview mode for mutations.
- Produces an anonymized JSON/Markdown summary.

Acceptance criteria:

- The script completes without requiring private firmware or user-specific paths.
- Every command path is marked as covered, intentionally skipped, or unsupported in the current environment.
- The output contains no target paths, raw strings, disassembly, decompiler output, hashes, or cache/socket paths.

### 7. Verify End To End

Run the verification stack after each implemented batch.

Required checks:

```bash
uv run pytest -q
bash tests/stress/run_stress.sh
uv run python tests/stress/run_command_surface_audit.py
```

Acceptance criteria:

- Unit tests pass.
- Stress tests pass from a clean checkout.
- Command-surface audit passes and produces a sanitized report.
- `CODEX_CODE_REVIEW_31337.md` is updated with before/after status.

## Execution Order

1. Stress fixture restoration or stress harness rewrite.
2. Sanitized command-surface smoke audit.
3. Quick-load state visibility.
4. JSON shape documentation or normalization.
5. Batch operation parity audit.
6. Type lookup guidance.
7. Full verification and report update.

## Notes

- Avoid modifying supplied firmware or target corpora.
- Use explicit `--instance` and `--target` in automation.
- Keep mutation tests in preview mode unless running against generated temp fixtures.
- Do not include sensitive target details in generated reports.
