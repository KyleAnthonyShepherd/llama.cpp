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

# Run each ngl in its OWN llama-bench process. A config that does not fit calls
# ggml_abort() from inside the CUDA backend, which kills the whole process - with a
# single invocation covering the list, one OOM at the top end throws away every
# later benchmark in the script too. Isolating them costs a few reloads and makes
# the sweep self-terminating instead of fatal.
: > "$OUT/sweep.md"
for NGL in ${NGL_LIST//,/ }; do
    echo
    echo "--- ngl=$NGL ---"
    if "$BENCH" \
            -m "$MODEL" \
            -ngl "$NGL" \
            -p "$N_PROMPT" \
            -n "$N_GEN" \
            -r 2 \
            --progress \
            -o md \
            > "$OUT/sweep-ngl$NGL.md" 2>&1; then
        grep -E '^\| (qwen|model)' "$OUT/sweep-ngl$NGL.md" | tee -a "$OUT/sweep.md"
    else
        echo "  ngl=$NGL FAILED (almost certainly VRAM). Last lines:"
        tail -5 "$OUT/sweep-ngl$NGL.md" | sed 's/^/    /'
        echo "| ngl=$NGL | FAILED - out of VRAM |" >> "$OUT/sweep.md"
        echo "  -> this is the ceiling at n_ubatch=512; the -ub sweep below may lift it"
    fi
done

echo
echo "=== KV forced to host (-nkvo) at ngl=${NGL_BEST:-13} ==="
echo "shows how much of the win is weights vs. KV placement. At short context this"
echo "should be tiny; it grows with -d (KV is re-read every token)."
"$BENCH" \
    -m "$MODEL" \
    -ngl "${NGL_BEST:-13}" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -nkvo 0,1 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-nkvo.md" || true

# The 04-residency baseline reported "graph splits = 851 (with bs=512), 82 (with bs=1)"
# for a CONTIGUOUS placement, which should need only a handful. That means the scheduler
# is offloading individual large-batch ops to the GPU and streaming CPU-resident weights
# over PCIe per op. Two levers, neither in the original plan:
#   -nopo : stop offloading ops whose weights live on the host
#   -ub   : smaller ubatch shrinks the 513 MiB CUDA0 compute buffer, freeing VRAM for
#           more layers, at some prompt-processing cost
echo
echo "=== op-offload off (-nopo), best-guess ngl ==="
"$BENCH" \
    -m "$MODEL" \
    -ngl "${NGL_BEST:-13}" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -nopo 0,1 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-nopo.md" || true

echo
echo "=== thread sweep (generation is DRAM-bandwidth bound; you are at 6 of 12) ==="
"$BENCH" \
    -m "$MODEL" \
    -ngl "${NGL_BEST:-13}" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -t 4,6,8,12 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-threads.md" || true

# Depth sweep. This is the one that tests the long-context KV claim in PLAN section 2:
# KV is re-read every token, so its cost grows linearly with context while the weight
# cost stays flat. At -d 0 it is invisible; by -d 16384 it should be clearly visible,
# and it is the term -ngl CANNOT move (KV placement follows the contiguous window only,
# src/llama-kv-cache.cpp:214-219 + src/llama-model.cpp:1318-1330).
echo
echo "=== depth sweep (does KV cost grow the way the roofline predicts?) ==="
"$BENCH" \
    -m "$MODEL" \
    -ngl "${NGL_BEST:-13}" \
    -n "$N_GEN" \
    -d 0,4096,16384 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-depth.md" || true

echo
echo "=== ubatch sweep (shrinks the compute buffer, frees VRAM) ==="
"$BENCH" \
    -m "$MODEL" \
    -ngl "${NGL_BEST:-13}" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -ub 128,256,512 \
    -r 2 \
    --progress \
    -o md \
    2>&1 | tee "$OUT/sweep-ubatch.md" || true

echo
echo "results in $OUT"
echo
echo "next:"
echo "  1. pick the best ngl, then re-run 04-residency.sh with NGL=<that value>"
echo "     to check the winner also keeps rss_anon and swap at zero."
echo "  2. if -ub 256 costs little pp, it frees ~250 MiB of compute buffer, which is"
echo "     roughly one more layer AND may lift the ngl ceiling that OOM'd above."
echo "  3. remember tg gains from ngl are ~1.6%/layer on this box (DRAM roofline)."
echo "     MTP is worth ~70%. Do not spend much time here."
