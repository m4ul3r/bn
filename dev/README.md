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
