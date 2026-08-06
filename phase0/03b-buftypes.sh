#!/usr/bin/env bash
# Which CPU buffer type do the host-resident weights actually land in?
#
# This is the decisive check for PLAN section 1.2, and it is fast - one short run per
# configuration, no generation to speak of.
#
# cpu_buft_list is built in this order (src/llama-model.cpp:896-950):
#     ACCEL bufts -> the GPU's pinned HOST buffer type -> repack extras -> plain CPU
# and select_weight_buft returns the FIRST entry that supports the op
# (src/llama-model-loader.cpp:1046-1056). The plain, mmap-able CPU buft is therefore the
# LOWEST priority option, and no -ot override is needed to skip past it.
#
# Only the plain CPU buft gets the mmap-backed buffer, because that path requires
# is_default_buft (src/llama-model.cpp:1568-1570). Everything else means a real copy:
#   CPU_Mapped   -> mmap. Clean, evictable, never swapped.       GOOD
#   CUDA_Host    -> pinned host memory. Copy, AND page-locked.   WORST on 16 GiB
#   <repack>     -> copy plus transform. Anonymous, swappable.   BAD
set -euo pipefail
source "$(dirname "$0")/config.sh"
resolve_model

# llama-completion, not llama-cli: the TUI client forces params.verbosity =
# LOG_LEVEL_ERROR (tools/cli/cli.cpp:36) and hides these very lines.
RUNNER="$BIN_DIR/llama-completion"
[ -x "$RUNNER" ] || { echo "error: $RUNNER not found; run 00-build.sh" >&2; exit 1; }

OUT="$RESULTS_DIR/03b-buftypes"
mkdir -p "$OUT"
record_env "$OUT/env.txt"

NGL="${NGL:-17}"   # from 02-gguf-layout.py's offload-order table
CTX="${CTX:-4096}"

probe() {
    local name="$1"; shift
    echo
    echo "=== $name : $* ==="
    # -lv 4 is REQUIRED. common_get_verbosity maps GGML_LOG_LEVEL_INFO to
    # LOG_LEVEL_TRACE (4), not LOG_LEVEL_INFO (common/log.cpp:444), and the default
    # threshold is 3 - so every library INFO line, including the "model buffer size"
    # lines this whole script exists to read, is dropped unless verbosity >= 4.
    # -fit off skips the fitter, which would only abort anyway once -ngl is explicit
    # (common/fit.cpp:377) after wasting a probe load.
    local rc=0
    "$RUNNER" -m "$MODEL" -c "$CTX" -ngl "$NGL" -n 1 -no-cnv -p hi -lv 4 -fit off "$@" \
        > "$OUT/$name.log" 2>&1 < /dev/null || rc=$?
    echo "  exit code: $rc"

    if grep -q 'model buffer size' "$OUT/$name.log"; then
        grep 'model buffer size' "$OUT/$name.log"
        echo "  --- host-side total ---"
        awk '/model buffer size/ && !/CUDA[0-9]/ {s+=$(NF-1)}
             END{printf "  %.1f MiB not on a CUDA device\n", s}' "$OUT/$name.log"
    else
        # Never swallow the evidence - show what the run actually said.
        echo "  (no 'model buffer size' lines - showing the log so we can see why)"
        echo "  log: $OUT/$name.log ($(wc -l < "$OUT/$name.log") lines)"
        echo "  --- first 15 ---"
        head -15 "$OUT/$name.log" | sed 's/^/    /'
        echo "  --- last 25 ---"
        tail -25 "$OUT/$name.log" | sed 's/^/    /'
    fi
}

echo "model  : $MODEL"
echo "size   : $(du -Lh "$MODEL" | cut -f1)"
echo "runner : $RUNNER"
echo "ngl    : $NGL   ctx: $CTX"

# Sanity: does the runner produce loader output at all? If this is empty, nothing
# below will work and the problem is the binary or its logging, not the buffer types.
echo
echo "=== runner sanity check ==="
if "$RUNNER" --version > "$OUT/version.txt" 2>&1 < /dev/null; then
    head -3 "$OUT/version.txt" | sed 's/^/  /'
else
    echo "  warning: '$RUNNER --version' failed - see $OUT/version.txt"
fi

probe "default"
probe "norepack"         -nr
probe "norepack-nohost"  -nr --no-host
probe "nohost"           --no-host

echo
echo "================================================================"
echo "verdict"
echo "================================================================"
for f in "$OUT"/*.log; do
    [ "$(basename "$f")" = "version.txt" ] && continue
    n=$(basename "$f" .log)
    # The log line is "<ts> I load_tensors:   <BUFT> model buffer size = N MiB", so the
    # buffer-type name is the token immediately before "model buffer size".
    types=$(grep -o '[A-Za-z0-9_]* model buffer size' "$f" | awk '{print $1}' | sort -u | tr '\n' ' ')
    host=$(awk '/model buffer size/ && !/CUDA[0-9]/ {s+=$(NF-1)} END{printf "%.0f", s}' "$f")
    printf "  %-18s host=%6s MiB   %s\n" "$n" "$host" "${types:-<none>}"
done
echo
echo "Carry whichever configuration shows CPU_Mapped for the bulk of the host-side"
echo "weights into 04-residency.sh and 05-ngl-sweep.sh as the baseline."
echo
echo "results in $OUT"
