#!/usr/bin/env bash
# Throughput vs. how many layers sit on the GPU.
#
# This is what turns the byte layout from 02-gguf-layout.py into an actual
# recommendation: find the ngl where pp/tg peak, and see how sharply it falls off
# once the CPU-side working set stops fitting in page cache.
#
# Caveat worth knowing before you read the numbers: llama-bench does its own arg
# parsing and does NOT accept -nr/--no-repack (it has -lm and --no-host but no
# repack toggle). So every number here is with repack ENABLED. For the repack
# comparison use 04-residency.sh, which drives llama-cli.
set -euo pipefail
source "$(dirname "$0")/config.sh"
resolve_model

BENCH="$BIN_DIR/llama-bench"
[ -x "$BENCH" ] || { echo "error: $BENCH not found; run 00-build.sh" >&2; exit 1; }

OUT="$RESULTS_DIR/05-ngl-sweep"
mkdir -p "$OUT"
record_env "$OUT/env.txt"

# Set this from 02-gguf-layout.py's trailing-layer table - centre the sweep on
# the value it says fits, and bracket it.
NGL_LIST="${NGL_LIST:-0,4,8,12,16,20,24}"

echo "model    : $MODEL"
echo "ngl list : $NGL_LIST"
echo "note: llama-bench derives its context from -p/-n/-d, so N_CTX from config.sh"
echo "      does not apply here. Passing -ngl explicitly also makes the fitter bail"
echo "      out (common/fit.cpp:377), which is what we want for a controlled sweep."
echo
echo "This is slow - most of the model is on the CPU. Expect tens of minutes."
echo

drop_caches

# -r 2 keeps it tolerable; -d 0 measures at empty context. Add a second -d later
# if you want the with-history behaviour (that is where KV placement starts to
# matter for the 16 attention layers).
"$BENCH" \
    -m "$MODEL" \
    -ngl "$NGL_LIST" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep.md"

echo
echo "=== same sweep, KV forced to host (-nkvo) ==="
echo "shows how much of the win is weights vs. KV placement. On this arch only"
echo "16 of 64 layers have a KV cache at all, so the delta should be small - if it"
echo "is large, the hybrid split is not what PLAN section 0.1 assumes."
"$BENCH" \
    -m "$MODEL" \
    -ngl "$NGL_LIST" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -nkvo 1 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-nkvo.md"

echo
echo "results in $OUT"
echo
echo "next: pick the best ngl, then re-run 04-residency.sh with NGL=<that value>"
echo "to see whether the winning configuration is also the one that keeps rss_anon"
echo "and swap at zero."
