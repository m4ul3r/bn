# bn reference — runtime

Use this reference for target routing, headless sessions, quick loads, output, and the Python escape hatch. Use `bn help <group>` for current flag grammar.

## Target and instance selection

```bash
bn target list
bn instance list
bn instance find <path-or-subname>
bn instance gc
bn target close <selector>
bn close -t <selector>
bn close --all
bn save
bn refresh
```

One open target can be selected implicitly. With several open targets, pass `-t <selector>` from `bn target list`; selectors include the returned selector, view or target id, and an unambiguous filename/path. Each listed `selector` resolves to its own target in that open-target snapshot, including raw/adjacent-`.bndb` views, cache-name aliases and identical paths. The bridge uses a path suffix only when the resolver finds exactly that target; otherwise it lists the full `target_id`. A live full `target_id` takes precedence over filename aliases, so even a file named after another target's ID cannot shadow that fallback. List again after opening or closing views: path aliases can become ambiguous as the open set changes. `-i <instance>` selects a bridge. Put these flags before or after a command; explicit flags take precedence over ambient defaults.

Every alias counts equally, so opening a view never moves a selector onto it: a selector that matches two views is refused with both listed. The one exception is the save round trip: after `bn save` on a raw view and `bn load` of that `.bndb`, the raw view's own name and path still mean the raw view (the database answers to its `.bndb` name) — only while that `.bndb` is still the file this bridge wrote and not loaded as unverified; a replaced database makes the name ambiguous again. A bare `bn close <name>` (no `/`, no such file in the cwd) closes only the target whose file name is exactly `<name>`; use `-t` for any other selector. `bn close -t foo` (and `bn target close foo`) refuses when `foo` names `foo.bndb` only through its `.bndb`-stripped name — including after `bn load foo` opened the adjacent `foo.bndb` — and prints the command to run with the target's listed selector.

Every listing row in a refusal is a complete command: top-level refusals,
`close`, `target close`, batched decompile errors (text and JSON/ndjson), and
the per-instance errors of `--all-instances`/`--all-targets`, which are spelled
as the single call to that instance. A value starting with `-` is attached
(`--target=<sel>`, `--instance=<id>`, `--instance-id=<id>`). The raw
`Open targets:` rows mark the active target with a trailing `(active)`.

With several bridges running, a bare command routes to the one associated with
this project root and says so on stderr. `bn instance list --mine` lists only
those (and the project's sticky pin).

When a request gets an empty or cut-off reply, bn reports what it observed --
whether the bridge process is still running, and whether its registry entry is
present, absent, or could not be read -- and names no cause: a stop, a crash, a
kill or a malformed reply can each leave either state.

Instance routing is `-i` → `BN_INSTANCE` → project sticky pin → sole/unique project-associated live instance → auto-spawn. Target routing is `-t` → `BN_TARGET` → project sticky pin → single-open auto-pick; for `batch apply`, a manifest `"target"` wins over ambient defaults but loses to explicit `-t`. `BN_TARGET` is an ambient default, not an explicit command-line selection. A bare destructive `bn close` ignores the ambient target and closes only the sole open view or refuses with a target list; use `-t`, a path, or `--all` deliberately. An empty `BN_TARGET` or empty sticky target is refused when a command would use it. A valid ambient target can affect a bare `bn save`, so pass `-t` for the database you intend to write.

```bash
bn instance use <id>
bn target use <selector>
bn instance clear
bn target clear
```

Sticky pins are shared by agents working under one project root. In concurrent work sharing a project or process environment, pass `-i` and `-t` explicitly and do not change shared pins. An agent with a genuinely private process environment can use its own `BN_INSTANCE` and `BN_TARGET`; they still lose to explicit flags. `bn session start` uses `--instance-id` to *name a new bridge*; global `-i` routes to an existing one.

## Sessions and analysis state

```bash
bn load /path/to/sample.bin
bn session start /path/to/sample.bin --instance-id analysis-1
bn session start /path/to/large.bndb --instance-id analysis-1 --detach
bn session status <job-id> -i analysis-1 --format json
bn session list
bn session restart analysis-1
bn session stop analysis-1
bn close
```

A headless start owns an instance that should be stopped when work ends. Set a positive `BN_IDLE_TIMEOUT` on an agent-owned start as a crash fallback (`BN_IDLE_TIMEOUT=3600` gives one hour); unset means no idle reaper. The command reports the loaded target selector. For long analysis, `--detach` registers a bridge and returns a load-job id. Poll that id: `terminal: false` means keep waiting; on terminal, `succeeded` is true or false. A no-id `session status` lists jobs, not one verdict. `status_command` is null when the bridge cannot name itself in a fresh CLI invocation. On a loaded host, raise `BN_SPAWN_TIMEOUT` (for example, `180` seconds) for registration and set the orchestration command timeout higher; this is separate from `BN_IDLE_TIMEOUT` and the detached load-job budget. Stop an exact owned instance even if its target close fails; a caller killed by its outer timeout may have left a child that registered later. Leave a pre-existing instance alone when start returns the exact duplicate-ID error.

`bn session stop` and `bn session restart` refuse (exit 2) while any target of that instance has unsaved mutations, listing each target's `-t` selector and the `bn -i <id> -t <selector> save` to run; `--discard-unsaved` proceeds and reports `discarded_unsaved` (`selector` + `path` per target; text: `[unsaved changes discarded]`), the same disclosure `bn close` prints. The check asks the bridge with its own short budget, independent of `BN_REQUEST_TIMEOUT`. A bridge that cannot be reached (stopped, gone, socket missing) is stopped as before; a live bridge that does not answer — busy under a write lock, or a GUI that cannot enumerate its tabs — also refuses (exit 2, "cannot verify"), and `--discard-unsaved` is the only way past it.

Process lifecycle commands (`session stop` and `session restart`) are headless-only. They refuse a GUI-owned bridge without sending a shutdown RPC or escalating to SIGTERM/SIGKILL, including with `--discard-unsaved`. The GUI `shutdown` RPC itself also refuses and leaves admission open: it has no headless event waiter to complete teardown. Use Binary Ninja's **Restart Bridge** menu command to restart its bridge, or close Binary Ninja normally to exit the GUI process.
GUI ownership is checked before target enumeration, so unsaved or unresponsive GUI views do not produce a misleading `--discard-unsaved` retry hint.

**Restart Bridge** keeps each still-open tab's target id, unsaved marker, backing database path and the ledger of comments this session wrote, so those comments still render `[this session]` rather than `<annotation redacted>`. It also keeps the locks over those tabs, so a write the old bridge admitted and is still running goes on serializing against the new bridge: the new bridge's writes queue behind it, and its reads wait while that write holds the views exclusively (not while a `refresh` analyses or `go rename` is between chunks, when reads still answer). Those waits get the usual stderr notice naming the old write, and `BN_NO_WAIT=1` refuses them with `busy:` (below). A client whose request times out still cancels it on the old bridge, which stops an op that checks for cancellation (`go rename`); any other op runs to completion. So Restart Bridge does not free a stuck write: the new bridge waits for it, and only closing Binary Ninja ends it.

Bridge shutdown closes admission to new work, stops the accept loop, and drains already received requests through response encoding and writing before removing its discovery files. The request-drain deadline is two seconds by default, shared across handlers rather than renewed per request; it does not guarantee completion of a stuck operation or a blocked response write. `BinaryNinjaBridge.stop(request_drain_timeout=...)` returns `requests_drained`, `unfinished_requests`, `request_drain_timed_out` and `caller_inflight`, and emits a BN warning when requests remain. A request calling `stop()` cannot wait for its own response: it is counted as unfinished, not silently called drained. Set `request_drain_timeout=0` and `load_join_timeout=0` for emergency teardown without request draining or load-worker joins. A concurrent emergency stop can overtake a drain; neither a late drainer nor a repeated stop removes a successor's discovery files. These are bridge-side controls, not CLI flags. The `shutdown` RPC acknowledges shutdown initiation, not the final drain verdict; `bn session stop` still checks process teardown and retains its forced-signal escalation. Expiry/forced teardown does not promise cancellation or rollback of unfinished work.

After an acknowledged shutdown, lifecycle commands allow up to 30 seconds for the process and discovery files to disappear before considering forced teardown. Signals still require a verified pidfd held for the whole escalation; a missing pidfd never falls back to `/proc` checks followed by a bare PID signal. If graceful teardown times out without that pin, the command reports a timeout and sends no process signal; restart does not spawn a replacement while the old instance remains. A failed shutdown request retains the guarded SIGTERM fallback and its five-second wait.

When `stop()` runs on the native GUI main thread, request draining and load-worker joins are skipped: those workers may themselves require that thread to complete. The verdict adds `request_drain_skipped: "gui_main_thread"` and still reports the actual `unfinished_requests`/`requests_drained`; a skipped drain is not labeled a deadline timeout. Admission and discovery-file ownership are released in the same safe order. Pending handlers can complete once the main thread returns, but teardown does not cancel them or promise their completion.

A successful fresh bind of the same bridge object restores request admission for that listener. Requests arriving over a connection retained from a retired listener remain refused; rebinding does not reopen the old listener's admission. A valid declared idle probe remains excluded from owner activity even when a retired listener refuses it.

A saved `.bndb` may be preferred over adjacent raw bytes, but only if it describes them: the bridge records the raw file's sha256 on the view when it opens it (so `bn save` and exports certify the bytes that were loaded, not whatever sits at the path by save time), and a load that prefers a sidecar or cache database — or finds that database already open — checks its record against the file now on disk. On a mismatch (re-extracted firmware, a different binary copied over the same name) the raw bytes load instead and `notes[]` says `NOT restoring <db>` with the `--no-bndb` and explicit `bn load <db>` escape hatches; a database saved before this record existed is checked against the original bytes it stores (`database provenance verified from the bytes stored in <db>`, and the record lands on its next save) or, if those differ or cannot be read, loads with a `database provenance unverified (saved before content validation)` note (stored bytes of a different SIZE cannot be a patch, so that is a `NOT restoring` refusal). A database named explicitly — `bn load <db>`, or `session restart` reopening a target through its `database_path` — is never refused, but when its record names other bytes than the sibling raw file now holds, `notes[]` says so (`<db> was saved for different bytes than <raw> now holds … kept because the database was requested explicitly`) and points at `bn load <raw> --no-bndb`. `bn target info` / `bn target list` keep the verdict as `source_verified` (`true`/`false`, `null` when no database was judged); the text `existing annotations:` line of `target info` / `evidence orient` prints it as `source-verified=true|false|unknown`, and the `existing_annotations.provenance_hint` spells it out, so it outlives the load note. On a read-only source mount, `bn save` can use a cache under `BN_CACHE_DIR` (default `~/.cache/bn`) named `<basename>.<first 16 hex of the raw file's sha256>.bndb`, so two firmware versions at one path never share a database; a later load can restore its annotations. That cache is shared per `BN_CACHE_DIR`: every agent and project using the same `BN_CACHE_DIR` restores the others' saved analyses. To isolate agents or projects, give each its own `BN_CACHE_DIR` (it also holds the instance registry, so each then sees only its own bridges). Use `--no-bndb` when the task requires raw-byte analysis without prior names/comments. A save normally preserves the live target selector; if the response reports `rehomed: true`, get the new selector from `bn target list`. For post-save commands, prefer the open view's `target_id`: a stale one fails after bridge restart, while a process-local `view_id` can be reused for a different view. After restart or close/reload, list targets again: the loaded filename and IDs can change, and a raw view beside its `.bndb` can make a bare basename ambiguous.

`--quick` skips the full analysis pass. Sections and imports are useful immediately; function listings/searches are partial, strings can refuse, and xrefs, callsites, and taint need analysis. All mutations refuse a quick-loaded target until `bn refresh`. Check `analysis_state` and run `bn refresh` before using these reads for a complete survey. `decompile --force-analysis` can analyze one existing function; it does not replace full target analysis or create a missing function. `--quick` never restores a cached database (the `<cache>/bndb/` copy, shared by every instance using that `BN_CACHE_DIR`, a save on a read-only mount writes): the note names the skipped entry and the `bn load <entry>` that restores it. On a read-only mount, a default `bn save` from a target that did not load an existing cache entry (`--quick`, `--no-bndb`) is refused rather than overwriting that entry; save with an explicit path, or `bn load` the entry to continue from it. The refusal exits 2 and returns `{"ok": false, "status": "invalid_request", "requested": {"path": ...}, "observed": {"cache_entry_exists": ..., "request_sent": true, "written": false}}`; it is exit 2, not the mutation exit 3, because `save` is a file write rather than a mutation (the same as the open-target collision refusal). Without `--quick`, the restore note names the entry, why it matched, when it was saved, and the annotation counts it already held.

`existing_annotations.provenance_hint` (in `target info` and `evidence orient`) flags inherited work only. A target this bridge opened from raw bytes gets no hint, however much you annotate or save; one opened from a database is judged by what it held when opened (`present_at_open`), never by your own later renames. The counts beside it are always current.

The load-time annotation note is omitted when inherited comments, function documentation and analyst symbols are all measured as zero. This does not remove the counts from `target info`.

While `bn refresh` runs, reads answer at once, but writes queue behind it: `py exec`, every mutation, `save`, `close` and `decompile --force-analysis` wait until the refresh finishes. After about two seconds of waiting the CLI prints `bn: <op> is waiting for the running refresh (started Ns ago) to finish` on stderr, so a queue is not a hang. Set `BN_NO_WAIT=1` to make a queued write fail fast with `busy: refresh in progress ...` (exit 2) instead.

An analysis-gated read (callsites, xrefs, `function info`, taint, strings) on a `--quick` view that a `bn refresh` is analysing right now reports the elapsed time and any available BN counters. `count` and `total` may reset or change during one analysis; they are not overall completion or a time-remaining estimate. Retry after the refresh instead of starting another. A second `bn refresh` refuses immediately, even if the first refresh is still queued behind another write.

A read waits only while a write holds the open views exclusively, or is queued to (writers go first, so a new read also waits behind a write that is itself waiting for earlier reads to finish). The writes that hold the views exclusively are: `bn load` and `session start` for most of their open (the open itself and a `.bndb` restore), the detached load `session start --detach` queues (it reports itself as `load_binary_async`), `py exec`, `save`, `close`, `batch apply`, `decompile --force-analysis`, every mutation while it applies its change, `go rename` while it renames a chunk, and `refresh` only while it updates its analysis flags at the end. Such a read gets the same notice after two seconds (`waiting for the running <op>`, or `the queued <op>` for a writer still waiting), and under `BN_NO_WAIT=1` it is refused with `busy: <op> in progress ...` or `busy: <op> queued for the open views ...`, saying `<op> needs to read the open views` (exit 2). So `BN_NO_WAIT=1 bn target list` fails, rather than waits, while another client's `bn load` is opening a file. A request that has already done its work is never refused at its reply: `refresh` waits for the read that builds its answer.

```bash
bn session start /path/to/sample.bin --instance-id triage-1 --quick
bn -i triage-1 target info
bn -i triage-1 refresh
```

## Output and artifacts

Read commands default to text. Mutations default to a compact **text status line**; use `--format json --summary` for parseable status, `--verbose` for detailed diffs, and `--out FILE` to write full detail away from stdout. Setup/export commands may default to JSON. Always parse the format requested, not the command's historical default.

`BN_SPILL_TOKENS` opt-in spills an oversized *read* to an artifact and puts an envelope on stdout. When a command is piped, the downstream `jq`/`rg`/`grep` may then see only the envelope. Without spill, a pipe receives full output, although a capturing wrapper may truncate it. Bound with `--limit`, `--lines`, or `--out`, and read stderr for slicing and spill notes. `--estimate-output` runs the read without emitting its body. Its envelope has `ok`, `estimated`, `format`, `bytes`, `tokens`, `tokenizer`, `summary`, and a command-specific `rerun` hint; `spill_token_limit` is present when a spill threshold is armed.

```bash
bn decompile parse_record --lines 20:45
bn function list --limit 100 --format json
bn strings --out /tmp/bn-strings.json
bn data symbols --offset 0 --out /tmp/bn-data-symbols.json
bn xrefs memcpy --estimate-output
bn spill gc --dry-run
```

`--out` writes an artifact and prints a small envelope. JSON envelopes identify `artifact_path`, `format`, `bytes`, `sha256`, and the source `target`/`instance`; check provenance before consuming a shared file. A text `xrefs` view may cap displayed caller groups even without a spill; use `--format json`, `--limit`, or `--out` for counts. `--lines START:END` is 1-indexed, inclusive, and text-only on decompile, IL, disasm, and structured IL. An out-of-range window errors rather than returning an empty answer. `bn spill gc` reclaims cached spill days; `--out` artifacts are caller-owned.

For paged reads, `--out` removes the default 100-row limit only when `--limit` is omitted; an explicit limit still wins, and `--offset` still applies. In particular, `bn data symbols --offset 0 --out /tmp/bn-data-symbols.json` requests the whole set in one bridge call. Each `data symbols` page otherwise rebuilds the full named-symbol population under the read lock, even though it builds rows only for that page. Prefer the one-request export when building a complete index; it avoids repeated scans, not the cost of one full scan and row build. See [Data-symbol pagination cost](reading.md#data-symbol-pagination-cost) for the large-page alternative and wire-level `limit: "all"` spelling.

## Discovery and troubleshooting

```bash
bn help evidence
bn capabilities --format json
bn doctor
bn plugin install
bn skill install
```

`bn capabilities` is the registry-derived command catalog. Use `bn doctor` when discovery, installation, or bridge connectivity is wrong; stale plugin code requires restarting the bridge or GUI. `bn plugin install` and `bn skill install` install local bridge and skill links, defaulting to text (`--format json` remains available). After moving to a new checkout, an old link is reported as not updated (`skill install` exits 1, `plugin install` refuses).

Copy installs write `.bn-copy-install.json` with `{"format":"bn-copy-install-v1","artifact":"<source directory name>"}`. `--force` replaces removable links, empty directories, or copies with a valid regular, non-symlink marker for that artifact, losing local edits and added files in the copies. Matching source bytes alone are not ownership proof. Move an unmarked older copy to a backup by hand, then re-run the same install command without `--force`; do not manufacture a marker. A linked skills root into an unmarked checkout is refused rather than rewriting the checkout.

For a refusal naming the skills root itself as a link, use its printed `Recovery command` to unlink only that skills-root link and create a new directory at that path, then rerun the same install command. Do not move directories within the target checkout. Ancestor links such as `~/.claude` may hold unrelated settings: never unlink those for recovery; choose a new `--dest` outside the linked tree or inspect the configuration manually. Copy preparation writes ownership before finalizing source directory modes, supports readable read-only sources, and cleans up a failed marker write without removing the previous destination. Ownership metadata is excluded from the engine fingerprint.

The per-destination force verdict is the forced run's own preflight with writes disabled, including current destinations that may block the run. Normal skill JSON always publishes `force_plan` and `force_ready`, even when every destination is current. Plan rows carry normalized `destination_id` values; text joins rows only when both entry identity and advice agree. Displayed roots with different remedies retain their own rows, even when they reach the same entry. An explicit `--dest` refusal publishes the whole plan with `ok: false` and `error`, exits 2, and writes no install or `--out` artifact; path-blocked rows use `state: "blocked"`. If configured spilling puts the plan in a cache artifact, this install command's stdout envelope still preserves `ok: false` and the error. Other artifact envelopes are unchanged: their `ok` describes file-write success, including mutation `--out` files. Successful installs recompute the plan against the resulting disk state. A forced run lists every preflight refusal before writing anything. Filesystem races or write errors after preflight may still leave earlier destinations installed. `bn instance gc` removes dead registry residue, not live bridges.

## Python escape hatch

```bash
bn py exec --stdin <<'PY'
print(bv.arch.name)
PY
```

Use `py exec` when a built-in read or verified mutation cannot express the task. It runs with an exclusive write lock and unsandboxed Binary Ninja/Python access, so even a read snippet blocks other clients and can mutate the view. Prefer `bn read`, `data vars`, and `data symbols` for common raw/data reads. Keep scripts bounded, verify any changes, and save intentionally.
