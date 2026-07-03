#!/usr/bin/env bash
# sink-sweep.sh -- enumerate a target's dangerous-sink imports and their call
# sites, the sink-enumeration step agents tend to skip (#169 Layer 3). Wraps
# `bn imports` + `bn xrefs`; pass the usual selectors, e.g.:
#   bash scripts/sink-sweep.sh -i <instance> -t <target>
#
# The script OWNS its own output shaping: it strips any caller-supplied
# --out/-o/--format (and value) from the forwarded args so it can capture bn's
# JSON internally. Do NOT pass --out to this script -- redirect stdout instead
# (`... > file.txt`). A forwarded --out used to override the internal one and
# silently empty the capture file, producing a false "no dangerous-sink imports"
# all-clear (#438); it is now consumed here instead.
#
# Output: for each imported dangerous sink, its inbound xrefs (the call sites to
# trace back to a source). Start backward taint / trace from these.
set -euo pipefail

command -v bn  >/dev/null || { echo "sink-sweep: 'bn' not on PATH" >&2; exit 2; }
command -v jq  >/dev/null || { echo "sink-sweep: 'jq' is required" >&2; exit 2; }

# Memory-unsafe copy / format-string / command-exec / tainted-input libc sinks
# worth tracing a tainted argument back to a source. Deliberately NOT the malloc
# family -- every binary calls it, and heap bugs (UAF/double-free) aren't found
# by xref'ing the allocator anyway.
# The leading (_*) absorbs the `__` decoration and the trailing optional
# (_chk)? matches the FORTIFY_SOURCE variants (__memcpy_chk, __snprintf_chk, ...)
# -- the default on modern firmware toolchains -- so a fortified target is not a
# false all-clear (#372). `execv` is listed alongside the rest of the exec family.
SINK_RE='^(_*)(memcpy|memmove|mempcpy|strcpy|strncpy|stpcpy|strlcpy|strcat|strncat|strlcat|sprintf|vsprintf|snprintf|vsnprintf|printf|fprintf|vprintf|vfprintf|dprintf|asprintf|vasprintf|gets|scanf|sscanf|fscanf|system|popen|execve|execv|execl|execlp|execvp|dlopen|recv|recvfrom|recvmsg|read)(_chk)?(@.*)?$'

# Consume the caller's args ONCE, dropping output-shaping flags (--out/-o/--format
# and their values, in both `--flag val` and `--flag=val` forms). The script fully
# controls output for both bn calls below; a stray caller --out can no longer
# redirect bn's JSON away from our capture file and fake a false all-clear (#438).
selectors=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out=*|-o=*|--format=*) shift ;;
        --out|-o|--format)
            shift
            [ "$#" -gt 0 ] && shift ;;
        *) selectors+=("$1"); shift ;;
    esac
done

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
# --out writes the full JSON to a file (no >10k-token spill envelope to parse).
bn imports --format json --out "$tmp" ${selectors[@]+"${selectors[@]}"} >/dev/null

# Fail loud rather than "none": if bn produced no JSON (empty/invalid capture),
# a zero-sink read is not trustworthy -- never report an all-clear on it (#438).
if [ ! -s "$tmp" ] || ! jq -e . "$tmp" >/dev/null 2>&1; then
    echo "sink-sweep: 'bn imports' produced no valid JSON -- cannot enumerate sinks." >&2
    echo "sink-sweep: (do not pass --out to this script; redirect stdout instead)" >&2
    exit 2
fi

mapfile -t sinks < <(jq -r '(.items // [])[] | .name // empty' "$tmp" \
                       | grep -E "$SINK_RE" | sort -u)

if [ "${#sinks[@]}" -eq 0 ]; then
    echo "sink-sweep: no dangerous-sink imports found in this target."
    exit 0
fi

echo "sink-sweep: ${#sinks[@]} dangerous-sink import(s) -- trace each call site back to a source:"
for sink in "${sinks[@]}"; do
    echo ""
    echo "=== $sink ==="
    bn xrefs "$sink" ${selectors[@]+"${selectors[@]}"} 2>/dev/null || echo "  (no resolvable xrefs)"
done
