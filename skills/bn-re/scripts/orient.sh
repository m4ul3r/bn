#!/usr/bin/env bash
#
# orient.sh -- first-pass RE triage card for a target, wrapping `bn evidence orient`.
#
# Part of the bn-re skill's Layer 3 methodology tooling (bn #169 / IDEA_001 sec 6).
# `bn evidence orient` composes target + analysis state, imports summary, a high-signal
# strings sample, function count, and sections into ONE internally-consistent digest
# under a single read lock (a shell loop of the individual commands can't give that
# guarantee). This script renders that digest as a compact human-readable card so the
# first thing an agent does on an unknown binary is oriented, not ad hoc.
#
# Requires `bn` (uv tool install -e .) and `jq` on PATH. Any extra args are passed
# straight through to `bn evidence orient`, so target/instance selection works normally:
#
#   orient.sh                       # single open target
#   orient.sh -t libfoo.so          # by selector
#   orient.sh --instance abc -t 1   # a specific bridge instance + target
#
set -euo pipefail

command -v bn >/dev/null 2>&1 || { echo "orient.sh: 'bn' is not on PATH (run: uv tool install -e .)" >&2; exit 127; }
command -v jq >/dev/null 2>&1 || { echo "orient.sh: 'jq' is not on PATH" >&2; exit 127; }

# `--format json` is appended LAST so a stray user-supplied `--format text` in "$@" can't
# override it (argparse takes the last flag) and leave jq parsing non-JSON.
if ! json=$(bn evidence orient "$@" --format json 2>/dev/null); then
    echo "orient.sh: 'bn evidence orient' failed -- is a target open? (bn target list)" >&2
    exit 2
fi

# orient is bounded (a strings SAMPLE, not the full set) so it stays inline in practice,
# but if the digest ever exceeds the spill threshold `bn` returns a spill ENVELOPE
# (`spilled: true`, the data on disk) instead of the digest -- rendering that as a card
# would print all-fallback garbage. Detect it and point at the file (#409).
if [ "$(jq -r '.spilled // false' <<<"$json" 2>/dev/null)" = "true" ]; then
    path=$(jq -r '.artifact_path // .path // "<disk>"' <<<"$json" 2>/dev/null)
    echo "orient.sh: the orient digest spilled to $path (too large to inline)." >&2
    echo "  Read it with:  jq . $path   (or raise BN_SPILL_TOKENS)" >&2
    exit 0
fi

jq -r '
  # left-pad a count to width 6 without a negative string-repeat (jq errors on that).
  def pad6: tostring | (if length < 6 then (" " * (6 - length)) else "" end) + .;
  "== orient: \(.target.name // .target.filename // .target.path // "<target>") ==",
  "analysis : \(.analysis_state // "?")  (analyzed=\(.analyzed))",
  "functions: \(.function_count // "?")",
  "sections : \(.sections.total // "?")   W^X: \(.sections.wx_verdict // "?")",
  "imports  : \(.imports_summary.total_symbols // 0) symbols  (\(.imports_summary.by_kind.function // 0) fn, \(.imports_summary.by_kind.address // 0) addr; \(.imports_summary.got_collapsed // 0) GOT-collapsed)",
  (if ((.imports_summary.needed_libraries // []) | length) > 0
     then "needed   : " + (.imports_summary.needed_libraries | join(", "))
     else empty end),
  "namespaces (top 8 by symbol count):",
  ((.imports_summary.namespaces // {}) | to_entries | sort_by(-.value) | .[:8][]
     | "  \(.value | pad6)  \(.key)"),
  "strings sample (min \(.strings_min_length // 6) chars):",
  ((.strings_sample.items // []) | .[:12][]
     | "  \(.address // "?")  \(.value // .string // .text // "")")
' <<<"$json"
