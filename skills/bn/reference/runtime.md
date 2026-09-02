# bn reference — runtime

Target selection, sticky pins, instance/target resolution order, sessions/headless, output & spill, py exec, troubleshooting, quirks, install for the `bn` skill. See `../SKILL.md` for the map.

## 1. Workflow & target selection

1. Discover targets:

   ```bash
   bn target list
   ```

   The `[N]` prefix is the view id; you can pass `-t N`. If no bridge is running, any command auto-starts one.

2. Pick a target:
   - Single open BinaryView: omit `-t`.
   - Multiple open: pass `-t <selector>` from `bn target list`. Selectors match against `selector`, `target_id`, `view_id`, full filename, or basename.
   - `-t/--target` and `-i/--instance` work **before or after** the subcommand, and for two-level commands they are also accepted **between the group and the leaf**. Prefer the short forms (`-t`, `-i`). Use a pre-subcommand form to disambiguate selectors that collide with subcommand names like `session` or `pam_qnx.so.2`:

     ```bash
     bn -i myid -t pam_qnx.so.2 decompile main      # at root (preferred for agents)
     bn decompile main -i myid -t pam_qnx.so.2      # after the leaf
     bn bundle -i myid -t pam_qnx.so.2 function main  # between group and leaf (two-level)
     ```

   - Use `-t active` only when you explicitly want to follow the GUI selection.

3. (Optional) Pin sticky defaults — useful for a **single** agent/shell running many commands against the same instance/target. **Do not** use sticky pins under multi-agent fan-out (see HARD rule below).

   ```bash
   bn instance use <id>          # pin -i/--instance for this project
   bn target use <selector>       # pin -t for this project
   bn instance clear              # clear pinned instance
   bn target clear                # clear pinned target
   ```

   Resolution order:
   - **Instance:** CLI `-i/--instance` > env `BN_INSTANCE` > sticky > sole live instance > unique live instance associated with the current project > auto-spawn — except `session stop`, which never falls back to a sticky pin (a bare `bn session stop` under a pinned project errors rather than silently stopping the pinned instance; pass `-i`/`--instance-id` or rely on `BN_INSTANCE`). Associations are private registry metadata under `~/.cache/bn/instances/`; if two sessions are associated with one project, bare commands fail closed and require `-i`.
   - **Target:** CLI `-t/--target` > sticky > single-open auto-pick. **`BN_TARGET` does not exist** — target selection is the CLI flag or `bn target use`, nothing else.

   Env `BN_INSTANCE` is optional **single-agent** convenience (same effect as always passing `-i`). **Do not** rely on it for multi-agent fan-out — a shared process env is still clobberable across concurrent agents; pass `-i` on every command instead.

   State lives at `~/.cache/bn/sessions/<sha256(project_root)[:16]>.json`. Project root walks up to the nearest `.git` (cwd as fallback). `bn session list` and `bn target list` mark matching entries with `[sticky]`. When a sticky instance points at a dead bridge, errors append `Clear it with bn instance clear`.

   > **HARD rule for parallel / fan-out agents.** Sticky pins are **one shared file per git repo** — every agent rooted in the same repo reads and writes the same `instance_id` / `target`. If multiple agents run concurrently against that repo, one agent's `bn instance use` / `bn target use` / `bn instance clear` / `bn target clear` silently changes the target for *all* of them, causing cross-talk and commands hitting the wrong binary. Parallel/fan-out agents **MUST** pass **`-i/--instance` and `-t/--target`** explicitly on **every** command and **MUST NOT** call `instance use` / `target use` / `instance clear` / `target clear`. Prefer one dedicated headless instance per agent, then the short flags:
   >
   > ```bash
   > bn session start /path/to/binary --instance-id dogfood-1   # spawn naming
   > bn -i dogfood-1 -t <sel> decompile main
   > bn -i dogfood-1 -t <sel> xrefs main
   > ```
   >
   > Note: global **`-i/--instance` is routing**, not spawn naming. `bn -i foo session start …` is rejected because it previously minted a random instance while looking named. Use `bn session start /path/to/binary --instance-id foo`.
   >
   > `session start` prints the loaded target's selector (`target: <sel>   (pass -t <sel>; id …)`), so a fan-out agent does **not** need a follow-up `bn target list` just to learn what to pass to `-t` (#653).

## 2. Sessions & headless

The bridge runs as a GUI plugin or as a headless process; both speak the same protocol.

```bash
bn load /path/to/binary.bndb [--instance-id <id>]   # auto-spawns a headless bridge if none is running
bn session start /path/to/binary [--instance-id <id>]   # synchronous preload
bn session start /path/to/large.bndb --instance-id <id> --detach
bn -i <id> session status [<job-id>]     # queued/running/complete/failed
bn session list [-i <id>]                # all running instances, or filter one
bn session stop <id>                     # aliases: --instance-id <id>, -i <id>
bn close [<path>] [-t <sel>] [--all]     # close one or explicitly --all
bn target close <sel>                    # close exactly that target (alias for `close -t <sel>`)
bn exports [list]                         # public exported symbols
bn help [family]                          # concise index; advertises capabilities
bn instance gc                            # reap dead instance cache residue
```

`bn session stop <id>` deliberately refuses the sticky-pin fallback other commands use: a bare `bn session stop` with no positional id, no `-i/--instance`, and no `BN_INSTANCE` errors instead of stopping whatever instance the project happens to be pinned to. Pass the id explicitly (positional, `-i/--instance`, or `BN_INSTANCE`) to stop a specific bridge.

When multiple bridge instances exist, flagless `bn load <path>` refuses ambient project/env/sticky routing. Pass `-i/--instance` to load into an existing bridge or `--instance-id` to create a named one. This destructive lifecycle boundary never guesses among concurrent agents.

`bn instance gc` is housekeeping: a crashed/SIGKILLed bridge leaves its `.log` (and sometimes its socket) behind in `~/.cache/bn/instances/`, and the lazy liveness sweep keeps those breadcrumbs forever, so the directory accumulates dead logs over time. `bn instance gc` removes the logs and orphan sockets of instances that no longer have a live registry — it never touches a running instance or the shared spawn lock — and reports what it reaped (`--format json` for the counts).

A bare `bn close` closes the single open target; with several open it refuses with the open-target list (pass `-t <selector>`, a path, or `--all`). It never closes everything implicitly — only `--all` does (#664). Because close is destructive, it is stricter than `bn save`: on a multi-tab GUI bridge a bare close does **not** fall back to the focused tab (save does), the CLI pins the exact `target_id` it observed rather than sending `active` (so a concurrent close/load between the lookup and the close yields an unknown-selector error instead of closing a different binary), an empty `-t ""` or empty positional path `""` (e.g. an unset shell variable) is an error rather than a bare close, and `-t` cannot be combined with a path or `--all`. The bridge enforces the same rules for raw socket clients: a non-null empty `target`, an empty `path`, and any `target`+`path`/`all` pair are rejected.

`bn close` reports each closed view as `{path, unsaved, engine_modified}`. `unsaved` is the sole persistence/cleanliness signal: it means a committed bn mutation was not saved and is the only condition that triggers the discard warning. `engine_modified` is Binary Ninja's broader analysis/cache bit; it can be true after a strictly read-only session and must not be interpreted as user mutation.

**`session status <job-id>` JSON contract.** Naming a job returns the single-job
shape, so a polling agent never has to index `items[0]` or re-derive terminality
from the state string:

```json
{
  "kind": "load_job",
  "job_id": "<hex>",
  "state": "queued|running|complete|failed",
  "terminal": false,
  "succeeded": null,
  "job": { "job_id": "...", "state": "...", "path": "...", "created_at": "...",
           "started_at": null, "finished_at": null, "error": null, "result": null },
  "status_command": "bn -i <id> session status <job-id>",
  "items": [ { "...": "the same job record" } ],
  "count": 1
}
```

`terminal` is true only for `complete`/`failed`. `succeeded` is `true` for
`complete`, `false` for `failed`, and **`null` while non-terminal** — a `false`
there would read as "the load failed" and make a poll loop tear down a healthy
bridge. The poll loop is `terminal == false` → sleep → re-run `status_command`
**when it is non-null**; a `null` means this bridge cannot be named on a fresh
command line, so keep polling through the client or connection you already have
bound to it, or treat the command as unavailable. Only a bridge started with an
instance id (`--instance-id`, i.e. any headless bridge) can publish the exact
re-runnable command; a GUI-loaded bridge has no unambiguous CLI selector, so it
publishes `status_command: null` rather than a command that cannot address it.
The key is always present, so `null` and "field missing" stay distinguishable.
An unknown job id is an **error**, not an empty result, so a typo cannot spin a
status loop until its own deadline. Omitting the job id keeps the population
collection (`kind: "load_jobs"`, `items`, `count`) with no `terminal`/`succeeded`
verdict, because one verdict over many jobs would be a lie, and whose items are
the raw job records without a `status_command` of their own. Text mode prints
`<job-id>  <state>  <path>`, plus a `poll:` line naming the exact command while
the job is still running and a command is available.

**`bn target close <selector>`** closes exactly the target that selector names.
It is the discoverable spelling of `bn close -t <selector>` and delegates to the
same implementation, so unsaved warnings, selector resolution, and the refusal to
forward the volatile `active` literal are identical. It accepts no path and no
`--all`, so it can never widen into closing everything; an empty selector is an
error. Use `bn close --all` when you really do mean every loaded target.

```bash
bn save                                  # saves to <filename>.bndb
bn save /path/to/output.bndb             # explicit path (positional)
bn save --path /path/to/output.bndb      # --path is an accepted alias for the positional
```

> **Selector rebind after save.** `bn save` / `bn save <path>` rebinds the in-memory view's filename, so its basename / filename selector changes (e.g. `foo` becomes `foo.bndb`). A `-t foo` that worked before the save can stop resolving afterward. Post-save commands should target the **stable** `view_id` / `target_id` (the `[N]` prefix from `bn target list`), not the basename, to avoid `Unknown target selector` after a save.

`bn load <raw>` and `bn session start <raw> [...]` auto-prefer a sibling `<raw>.bndb` when one exists, so saved annotations come back without you having to retype the `.bndb` suffix. The CLI prints which file was actually opened:

```bash
$ bn load /path/to/foo.so
loaded: /path/to/foo.so.bndb
note: loaded /path/to/foo.so.bndb instead of /path/to/foo.so (use --no-bndb to skip)
```

Pass `--no-bndb` to force loading the raw binary even when a sibling `.bndb` exists. Passing a path that already ends in `.bndb` skips the lookup. The same `--no-bndb` flag works on `bn session start`.

> **Global BNDB cache (read-only mounts).** Auto-prefer isn't limited to a *sibling* `.bndb`. When the target's directory is not writable (a read-only firmware mount), a prior `bn save` falls back to a **global content-hash-keyed cache** at `~/.cache/bn/bndb/<stem>.<hash>.bndb` (override the cache root with `BN_CACHE_DIR`). A later `bn load <raw>` / `bn session start <raw>` **auto-restores from that cache** — carrying every prior rename/comment — and prints `note: restored cached database …; pass --no-bndb to load the raw bytes`. Two consequences for a "recover names on an unknown binary" task: (1) the view can come back **already annotated** from an earlier run, so you can't tell your recovery from a previous one — check the note (or `--no-bndb`) before claiming a clean slate; (2) a multi-MB target that "loads in seconds" is a **cache hit, not a fast cold analysis** — don't read it as a timing/perf observation. Use `--no-bndb` for a pristine, un-annotated analysis.

`bn load` blocks until analysis completes (`update_analysis_and_wait()`). Full loads can take seconds to minutes under contention; do not rely on a fixed load time. Each headless bridge can consume hundreds of MB, and a large/complex BNDB may be OOM-killed. Bound fan-out concurrency, monitor RSS with `bn session list`, and stop unused instances promptly.

**Quick load (`--quick` / `--no-analysis`).** `bn load --quick` and `bn session start --quick` skip that analysis pass (~1s instead of waiting for the full function set), at the cost of a **capability boundary** — the container is parsed but the code is not yet analyzed:

- Ready immediately: `bn sections`, `bn imports`, the symbol table, `bn target list` / `bn target info` (flagged `[not analyzed]`, JSON `analysis_state: "quick"`).
- `bn strings` **errors** until `bn refresh` (it refuses with a "Strings are not available … Run `bn refresh`" directive rather than return an empty list that reads as "no strings").
- **Partial** until `bn refresh`: `bn function list` / `bn function search` (only entry-point + symbol functions exist pre-analysis; the count grows after refresh), and `bn decompile` / `bn il` / `bn disasm` across the binary.
- **Hard-error** until `bn refresh` (they refuse rather than return a misleading empty result): `bn xrefs`, `bn callsites`, `bn function info`, `bn taint`.

Run `bn refresh` once to promote the view to full analysis (`analysis_state` flips to `"full"`), or `bn decompile <fn> --force-analysis` to analyze a single function without the full pass. Branch on `analysis_state` rather than guessing from empty results. Loading a `.bndb` ignores `--quick` (the database already carries its analysis).

**Quick-mode capability matrix.** Per-command behavior on a `--quick` / `--no-analysis` view. A **hard-error** row refuses with a `--quick` directive — that is a capability boundary, **NOT** absence of results; never read it as "nothing found." Distinguish `bn decompile <fn> --force-analysis` (analyzes one *existing* function in place — works on a quick view) from `bn function create` (materializes a *missing* function — refused on quick, see #479).

| Command | Quick-mode behavior |
|---|---|
| `sections`, `imports` | **quick-safe** — container is parsed at load |
| `target info` / `target list` | **quick-safe** — flagged `[not analyzed]` / `analysis_state:"quick"` |
| `strings` | **hard-error until `bn refresh`** (string set isn't built) |
| `function list` / `search` | **partial** — only entry-point + symbol functions exist; count grows after refresh |
| `decompile`, `il` | **partial** — render only already-analyzed functions; `bn decompile <fn> --force-analysis` analyzes one function in place (the flag is on `decompile` only), after which `il` works on it |
| `disasm <fn>` | **partial** (needs the function) · `disasm <addr> --linear N` — **quick-safe** (raw linear decode, no function required) |
| `xrefs`, `callsites`, `function info`, `taint` | **hard-error until `bn refresh`** (`require_analysis`) |
| `trace` | **function-specific** — needs the containing function's MLIL; `--force-analysis` that function first, else refresh |
| `class list` / `show` | **quick-safe** — from demangled symbols + RTTI/defined types present at load (method-body xrefs still need analysis) |
| `evidence init` / `table` | **quick-safe** — read raw memory / `.init_array` / symbols |
| `evidence function` | **partial** — reads one function's call ABI; needs that function analyzed |
| `function create --preview` | **hard-error until `bn refresh`** — refused on quick even in preview (#479); all batch mutations refuse identically |

After `bn refresh` (or `--force-analysis` on a single function) every row promotes to full behavior; branch on `analysis_state`, not on an empty or errored result.

`-i/--instance` is accepted on every subcommand (short form **`-i <id>`** preferred for agents; long form `--instance`; env `BN_INSTANCE` as single-agent convenience only). On `bn load`, `--instance-id` is an accepted alias that names the bridge instance to auto-spawn — the same spelling as `bn session start --instance-id`, so you can use `--instance-id <id>` consistently across both. Global `-i` does **not** replace `--instance-id` for spawn naming.

**Private project associations.** `bn session start` associates the new bridge with the canonical project root of the caller (nearest `.git`, otherwise cwd); each successful `bn load` does the same. Associations live only in the owner-private, atomically written instance registry under `~/.cache/bn/instances/`—`bn` never writes routing files or edits Git metadata in the checkout. With multiple live bridges, a unique registry association lets a bare command resolve from the project; two matching sessions are an explicit ambiguity and require `-i`. Explicit `-i`, `BN_INSTANCE`, and sticky pins still win. Clean stop or crash-registry cleanup removes the association with the registry, and restart preserves the original roots without associating the restart command's cwd.

**Stopping is identity-checked and atomically signalled (#694).** `session start` cleanup, `session stop` and `session restart` first ask the bridge to shut down over the socket. Only if that fails do they signal the registry's pid, and then only through a **pinned** process: the pid is pinned with `os.pidfd_open`, its identity verified through that pin, and every signal of the `SIGTERM` → wait → `SIGKILL` escalation sent through the same pin. Pinning is what makes it safe — a pidfd holds a reference to the kernel's process record, so the pid cannot be recycled while the pin is held and the signal can never land on a different process. A `/proc` check followed by `os.kill` could not offer that: the verified process can exit and its pid be reused between the two steps.

Identity is `(boot id, pid, process start time)`. Start times count from boot, so they are unique only *within* a boot while registries live in a persistent cache (`~/.cache/bn`); without the kernel boot id an old registry could falsely match a brand-new process after a reboot. A registry written under a different boot id is therefore treated as positively stale, and one that records no identity — or only half of one, e.g. written by an older bridge — is **never** proven.

When the pid cannot be proven, or the interpreter provides no pidfd (availability is a property of the CPython build, not just the kernel — the interpreters `uv` installs have no `os.pidfd_open`), the command **refuses to signal at all**, names the pid, and tells you to confirm with `ps -p <pid>` and stop it by hand. That is deliberate: there is no safe non-atomic fallback. The `SIGKILL` escalation is gated identically, and teardown convergence treats a recycled pid as gone rather than escalating against it.

**Unreachable bridges are hidden, not advertised.** The bridge binds its socket *before* writing its registry, so "registry, no socket" is never a live bridge starting up. Such an entry is dropped from normal discovery — `bn session list`, instance resolution and every request path — because nothing can be dispatched to it; if its owner is dead or unproven the record is purged outright, and it is purged as soon as a proven owner exits. While that owner is alive the record survives for **lifecycle lookups only**: `session stop` / `session restart` resolve it (marked `unreachable`) so the live process still holding memory can be stopped, and spawn collision detection consults it too, so a new bridge can never reuse that instance id, bind over its socket path and orphan the process.

An explicit-but-empty selector is always an error, never "everything": `bn session list -i ''` (e.g. an unset shell variable) is rejected instead of silently listing every session, the same doctrine `bn session stop ""` and `bn close -t ""` already follow.

**Fan-out (`--all-instances` / `--all-targets`).** Whole-target **read survey** commands (`imports`, `sections`, `strings`, `exports`, `types`, `function list`/`search`, `class list`, `go functions`, `target info`, `evidence orient`) accept `--all-instances` (run across **every running bridge instance**) and `--all-targets` (run across **every target open in an instance**); combine them for every instance × target. The result is one `{kind: "fanout", instances: […]}` aggregate (text: a section per (instance, target) via the command's own renderer; JSON for machine use). Without `--all-targets`, each instance resolves its own target by the normal rule — an explicit `-t` applies to all, otherwise the per-instance implicit single target; an instance with no/ambiguous target becomes an `ok:false` row rather than failing the whole command (the command exits non-zero only if **every** result failed). It's an explicit allow-list (the `fanout=True` command flag), **not** every text command — mutations and side-effecting commands (`save`/`close`/`refresh`/`py exec`/`load`) never get it, so a write can't be fanned. Per-function reads (`decompile`/`xrefs`/…) aren't fannable either (their identifier wouldn't resolve in another instance). Use it for cross-instance / cross-target surveys instead of a shell `for` loop. The per-(instance, target) reads run **concurrently** (bounded worker pool), so a slow instance no longer serializes the rest; each row carries `duration_ms` and the aggregate includes a `slow_rows` summary (text: a `slowest:` line) so a long survey reads as progress, not a wedge (#417).

Requests time out after 600s by default; override with `BN_REQUEST_TIMEOUT=<seconds>` (`0`/`none`/`off`/empty disable). Invalid values fail before instance selection or spawning. Full synchronous `load`/`refresh` defaults to 3600s, but genuinely large BNDBs can exceed any practical foreground budget. Prefer `session start --detach`, poll `session status`, then read the selector from `target list`; a failed detached job preserves its error in bridge state. Bridge registration has its **own** budget, separate from the request budget: 60s by default, `BN_SPAWN_TIMEOUT=<positive-seconds>` to change it, capped by whatever is left of the request deadline. That applies to an auto-started bridge too, so a child that never registers fails in its spawn budget instead of holding an ordinary request for the full 600s (or a load/refresh for 3600s) (#694). Registration failures are recorded in the instance log.

**Very large binaries (~100k+ functions).** Use detached start rather than a background shell: the bridge registers before analysis, `session status` survives the initiating CLI process, and the completed job returns target selectors or a retained error. `--quick` helps raw/container triage but cannot remove analysis already stored in a pre-analyzed BNDB. While a target is open, `target info` carries pollable `analysis_progress` and reads remain responsive during `refresh`.

## 3. Output & context

Defaults:

- Read commands → `--format text`.
- Mutations → a compact **text status line**; the full audit payload is opt-in via `--verbose`, an explicit `--format json`, or `--out` (see `reference/mutating.md`). A mutation result never spills, so `json.loads(stdout)` on a `batch apply` always works.
- Setup and export commands → `--format json`.
- `--format ndjson` is available where it makes sense.
- `--out <path>` writes the full body to disk and returns an envelope on stdout.

**Spill envelopes.** When output exceeds **10 000 estimated tokens** (~3 bytes/token heuristic), the body is written to disk and stdout carries a compact envelope; stderr carries a one-line warning. Envelope keys:

- `ok` — request status.
- `spilled` — `true` when the body was written to disk because of the threshold; `false` when `--out` was used.
- `path` (text envelope) / `artifact_path` (JSON) — location on disk: `<cache>/spills/YYYYMMDD/<stem>-HHMMSS-<pid>-<rand>.<json|ndjson|txt>` (cache dir defaults to `~/.cache/bn`, override with `BN_CACHE_DIR`).
- `format` — `json`, `ndjson`, or `text`.
- `bytes`, `tokens` (estimate), `tokenizer` (`estimate`), `sha256` — size + integrity. `sha256` is the digest of **this artifact's bytes**, not of the binary.
- `target`, `instance` — **provenance**: which target and bridge instance produced the artifact (#653). Check them before trusting a `--out` file you didn't just write: two agents sharing a scratchpad both wrote `fns.json`, and one silently read the other's list — a different target, a different binary — with nothing in the file making that detectable.
- `summary` — shape hint with `kind` and `count` / `chars` / `keys`.
- `spill_token_limit` — the threshold that tripped (so you can see how far over you went).
- `rerun` — the **command-specific slicing knob** to bound the next read (e.g. `--limit`/`--offset` for lists, `--lines` for `disasm`/`il`, `--address-window` for `evidence function`), so you re-run bounded instead of blind.

**Predicting spill (#409).** Two signals let you avoid a wasted full run:
- **Threshold override** — set `BN_SPILL_TOKENS` (e.g. `BN_SPILL_TOKENS=40000`) to raise/lower the spill point for a bigger/smaller context budget. Non-positive/garbage values fall back to the 10 000 default (spill is never silently disabled). 
- **Near-spill note** — when a read *fits* but lands within 20 % of the threshold, `bn` prints a `note:` on stderr that the next (larger) page/scope will spill — slice it pre-emptively.

**Pipe trap (correctness).** When output spills, a downstream `grep`/`jq`/`awk`/`rg` reads only the small envelope, **not** the data — so a no-match silently reads as "absent" (e.g. `bn decompile <fn> | grep memcpy` finding nothing does *not* mean there's no `memcpy`). `bn` now prints an extra `note:` on stderr when stdout is a pipe and output spilled, but don't rely on noticing it. Instead, write to a file first and process that: `bn decompile <fn> --out /tmp/f.txt && grep memcpy /tmp/f.txt`, or slice with `--lines`/`--limit` so it doesn't spill.

> **`xrefs` text is display-capped (not just spilled).** For a hot symbol with thousands of callers, `bn xrefs <sym>` text output caps the body at the first 100 caller groups per section (the on-screen page) — the total-count header line (`xrefs to 0x… (N code, M data)`) stays accurate, but the body is truncated, so `bn xrefs <sym> | grep -c` / `| wc -l` undercounts. When stdout is a pipe and the body was capped, `bn` prints a `note:` on stderr naming the true totals. To get the full set, use `--out FILE` (writes every ref), `--format json` (paged, honest `total`), or bump `--limit`.

Slicing knobs to avoid spilling in the first place:

```bash
bn decompile <fn> --lines 40:80         # 1-indexed inclusive; prints "// lines 40-80 of N"
bn xrefs <fn-or-addr> --limit 20        # cap text output
bn function info <fn>                    # compact by default
bn function info <fn> --verbose          # full params + locals
```

`--lines START:END` works on `decompile`, `il`, `disasm`, and `function structured-il` (text mode only — it errors on `--format json`). A `START` past the last line is treated as an error: the command exits non-zero with a stderr diagnostic (not a `//` comment on stdout), so a scripted consumer can tell an out-of-range slice apart from a real result.

Pagination: `--limit` / `--offset` on list commands.


## 8. Python escape hatch

Reach for `bn py exec` only when built-in commands are awkward — arbitrary BinaryView introspection or operations the bridge does not expose. Built-ins are preferred because they are verified, cache-friendly, and integrate with the preview/verify loop.

```bash
bn py exec --code "print(hex(bv.entry_point)); result = {'functions': len(list(bv.functions))}"
```

Multiline snippets via stdin with a quoted heredoc:

```bash
bn py exec --stdin <<'PY'
out = []
for f in bv.functions:
    if 0x416000 <= f.start < 0x41C000:
        out.append((f.start, f.symbol.short_name))
out.sort()
print("\n".join(f"{addr:#x} {name}" for addr, name in out))
PY
```

Shell rules:

- Quote the delimiter as `<<'PY'` so the shell does not expand `$vars`, backticks, or backslashes before Binary Ninja sees the Python.
- Keep the closing `PY` on its own line with no indentation or trailing whitespace.
- `--script <file>` for code on disk; `--code` for true one-liners.
- Materialize Binary Ninja iterators (`f.hlil.instructions`, etc.) with `list(...)` instead of assuming random-access behavior.

The exec environment includes `bn`, `binaryninja`, `bv`, and `result`.

`py exec` always returns `stdout` and `result`. `result` is JSON-serialized when possible; if not, the CLI returns `repr(result)` and a non-fatal entry in `warnings`. If your script writes a JSON artifact, it is surfaced under `artifact`.

> **Exclusive write lock + unsandboxed.** `py exec` runs under the bridge's **exclusive write lock**, so a long-running snippet blocks **every** other op (reads and writes) on a shared bridge until it returns — don't park slow scripts on a bridge other agents are using. It also runs **unsandboxed** with full `bv` / `binaryninja` access (it can mutate or write to disk). Keep snippets short on shared bridges, and for raw byte reads prefer the dedicated `bn read` (read-locked, parallel-safe) instead of a `py exec` that calls `bv.read(...)`.

## 9. Troubleshooting

Run `bn doctor` only when something is wrong — commands fail unexpectedly, targets don't appear, or the bridge seems unresponsive:

```bash
bn doctor
```

It checks CLI version, plugin staleness (`stale_plugin_version`, `stale_plugin_code`), and instance connectivity. Don't run it as part of normal workflow. Exit code is reachability-only: nonzero if any probed instance is unreachable, zero otherwise (staleness fields are informational and never affect the exit code; zero registered instances is not a failure).

## 10. Known quirks

- **`types declare` verification failures.** The source-parser path handles most declarations, but a stubborn one may roll back with `verification_failed`. Workaround: define the struct directly via `bn py exec` using `StructureBuilder`, then re-run `bn types show`:

  ```bash
  bn py exec --stdin <<'PY'
  from binaryninja import types as bntypes
  s = bntypes.StructureBuilder.create()
  s.append(bntypes.Type.pointer(bv.arch, bntypes.Type.void()), "vtable")
  s.append(bntypes.Type.array(bntypes.Type.int(1, sign=False), 0x20), "pad_04")
  s.append(bntypes.Type.int(4, sign=False), "m_bLoad")
  s.append(bntypes.Type.pointer(bv.arch, bntypes.Type.int(1, sign=False)), "m_fileBuf")
  s.append(bntypes.Type.int(4, sign=False), "m_fileBufSize")
  bv.define_user_type("MyStruct", bntypes.Type.structure_type(s))
  print("defined MyStruct")
  PY
  ```

- **Stale bridge.** If `bn doctor` reports `stale: loaded plugin code does not match installed plugin file`, restart Binary Ninja (GUI or headless) to pick up the updated bridge. Commands behave unpredictably with stale code.

- **No targets ⇒ no `py exec`.** `bn py exec` requires at least one open BinaryView. If `bn load` is still running or the target isn't ready yet, `py exec` errors with "No BinaryView targets are open".

## 11. Skill install

`bn skill install` is idempotent. It links/copies the bundled skills into `~/.claude/skills/` and, when `~/.codex/` exists, also into `~/.codex/skills/`. Honors `CLAUDE_HOME` / `CODEX_HOME`. Use `--mode copy` for standalone copies, `--dest <path>` for a single explicit destination, and `--force` to overwrite. Restart your agent to pick up renamed or newly added skills.
