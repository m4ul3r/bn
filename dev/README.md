# Development Ideas

Unsorted ideas and suggestions under consideration. Nothing here is committed to.

---

## Autosave after mutations

The bridge knows which ops are mutations (`WRITE_LOCKED_OPS`). After any successful write op (rename, retype, comment, struct edit, batch apply), auto-save the affected target's `.bndb` so a crash never loses more than the in-flight operation.

**Open questions:**
- Is `bv.create_database()` after every mutation acceptable performance-wise for large binaries?
- Alternative: autosave every N mutations, or only after bulk ops like `batch_apply`
- Should the first save auto-create the `.bndb` sibling, or require the user to have saved once manually?

## Track last-accessed time per target

Add a `last_accessed` timestamp to `TargetRecord`, updated on every request that touches a target. Show it in `bn target list` output so users/agents can see what's been sitting idle.

**Open questions:**
- Is this useful enough on its own, or only if paired with auto-close of idle targets?
- Auto-closing idle targets is risky (could surprise users who step away) -- probably not worth it

## bn-re and bn-vr methodology skills

Add two methodology skills on top of the `bn` tool skill. Full plan: [bn-re-bn-vr-skills-plan.md](bn-re-bn-vr-skills-plan.md)

- **bn-re**: Static reverse engineering methodology (approaching unknowns, function triage, type recovery, call graph analysis, struct reconstruction)
- **bn-vr**: Vulnerability research methodology (attack surface, dangerous imports, input-to-sink tracing, common vuln patterns, systematic audit)
- Update `bn skill install` to install all three skills in one command
