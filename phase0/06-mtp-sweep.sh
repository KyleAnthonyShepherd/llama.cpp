#!/usr/bin/env bash
# Sweep speculative-decoding depth for MTP.
#
# llama-bench CANNOT do this: it parses its own arguments and has no --spec-type, so
# every llama-bench number in 05 is non-speculative. Since MTP is worth ~70% on this box
# (PLAN "MTP is the headline result") and everything in 05 is worth single digits, this
# is the more important sweep of the two.
#
# Driven through llama-cli because --spec-type and --spec-draft-n-max are registered for
# speculative/server/cli but not for completion (common/arg.cpp:4026). -lv 4 is required
# to see any of it (common/log.cpp:444).
#
# What to look for:
#   - eval t/s should peak at some n_max and then fall: deeper drafts cost a longer
#     verify batch, and acceptance decays per position.
#   - "mean len" tells you the effective tokens per pass over the weights, which is the
#     quantity that actually beats the DRAM roofline.
#
# MEASUREMENT NOTE (learned the hard way): with sampling enabled, each run generates
# DIFFERENT text, so acceptance and tg vary by ~10% run to run at n_gen=64 - enough to
# invent trends that are not there. Two things fix it, both on by default here:
#   1. Greedy sampling (--temp 0). Speculative decoding is output-equivalent to
#      non-speculative under greedy, so EVERY n_max produces the identical token
#      sequence and the only thing that varies is how it was produced. That makes the
#      comparison apples-to-apples.
#   2. A longer generation (n_gen 256 by default here, not the shared 64), so the
#      acceptance statistic averages over ~70 draft iterations instead of ~18.
# Set GREEDY=0 to sample instead, and REPS=N to repeat each point.
set -euo pipefail
source "$(dirname "$0")/config.sh"
resolve_model

CLI="$BIN_DIR/llama-cli"
[ -x "$CLI" ] || { echo "error: $CLI not found; run 00-build.sh" >&2; exit 1; }

OUT="$RESULTS_DIR/06-mtp-sweep"
mkdir -p "$OUT"
record_env "$OUT/env.txt"

# Leave NGL unset to let the fitter choose - it accounts for the MTP context correctly
# (server-context.cpp:1150-1198), and the right ngl differs per n_max because the draft
# context and the extra recurrent state both scale with it.
NGL="${NGL:-}"
NGL_ARGS=()
if [ -n "$NGL" ]; then
    NGL_ARGS=(-ngl "$NGL" -fit off)
fi

N_MAX_LIST="${N_MAX_LIST:-0 2 3 4 5 6}"
PROMPT="${PROMPT:-Explain in detail how a memory-mapped file differs from a heap allocation.}"

# Threads matter here for the same reason they matter in 05: the verify pass runs the
# host-resident weights on the CPU. Measured on the 1660 Ti box, tg rose monotonically
# 4 -> 12 threads without MTP, so the two levers should compose.
THREADS="${THREADS:-}"
THREAD_ARGS=()
if [ -n "$THREADS" ]; then
    THREAD_ARGS=(-t "$THREADS")
fi

# See the measurement note at the top. Greedy + a longer run is what makes these
# numbers comparable across n_max.
GREEDY="${GREEDY:-1}"
SAMPLE_ARGS=()
if [ "$GREEDY" = "1" ]; then
    SAMPLE_ARGS=(--temp 0 --seed 42)
fi
MTP_N_GEN="${MTP_N_GEN:-256}"
REPS="${REPS:-1}"

echo "model   : $MODEL"
echo "n_ctx   : $N_CTX   n_gen: $MTP_N_GEN"
echo "n_max   : $N_MAX_LIST   (0 = MTP disabled, the baseline)"
echo "threads : ${THREADS:-default}"
echo "ngl     : ${NGL:-fitter chooses}"
echo "sampling: $([ "$GREEDY" = 1 ] && echo 'greedy (--temp 0, comparable across n_max)' || echo 'stochastic - EXPECT ~10% run-to-run noise')"
echo "reps    : $REPS"
echo

run_one() {
    local n_max="$1"
    local rep="$2"
    local name="nmax${n_max}-r${rep}"
    local args=()

    if [ "$n_max" = "0" ]; then
        name="baseline-nospec-r${rep}"
    else
        args=(--spec-type draft-mtp --spec-draft-n-max "$n_max")
    fi

    echo "--- n_max=$n_max (rep $rep/$REPS) ---"
    drop_caches

    "$CLI" -m "$MODEL" -c "$N_CTX" -n "$MTP_N_GEN" -no-cnv -st -lv 4 \
        "${NGL_ARGS[@]}" "${THREAD_ARGS[@]}" "${SAMPLE_ARGS[@]}" "${args[@]}" \
        -p "$PROMPT" > "$OUT/$name.log" 2>&1 < /dev/null || echo "  (exited non-zero)"

    # llama-cli goes through the server path, so timings come from slot print_timing.
    # Fall back to common_perf_print for the non-speculative baseline.
    # sed, not grep -P: PCRE is a GNU extension and is unavailable in some locales.
    # Both "prompt eval time" and "eval time" match the first pattern; generation is
    # logged second, so tail -1 picks it.
    local tg ngl_used acc mlen
    tg=$(sed -n 's/.*eval time =.*, *\([0-9.]*\) tokens per second.*/\1/p' "$OUT/$name.log" | tail -1)
    ngl_used=$(sed -n 's/.*offloaded \([0-9]*\)\/.*/\1/p' "$OUT/$name.log" | tail -1)
    acc=$(sed -n 's/.*draft acceptance = \([0-9.]*\).*/\1/p' "$OUT/$name.log" | tail -1)
    mlen=$(sed -n 's/.*mean len = *\([0-9.]*\).*/\1/p' "$OUT/$name.log" | tail -1)

    printf "  ngl=%-3s tg=%-7s acceptance=%-8s mean_len=%s\n" \
        "${ngl_used:-?}" "${tg:-?}" "${acc:--}" "${mlen:--}"
    printf "%s,%s,%s,%s,%s,%s\n" "$n_max" "$rep" "${ngl_used:-}" "${tg:-}" "${acc:-}" "${mlen:-}" \
        >> "$OUT/summary.csv"
}

echo "n_max,rep,ngl,tg_tps,acceptance,mean_len" > "$OUT/summary.csv"
for n in $N_MAX_LIST; do
    for r in $(seq 1 "$REPS"); do
        run_one "$n" "$r"
    done
done

echo
echo "================================================================"
column -t -s, "$OUT/summary.csv"
echo "================================================================"
echo
echo "results in $OUT"
echo
echo "The n_max that maximises tg is the configuration to run. Note the fitter may pick"
echo "a different ngl for each n_max - that is correct behaviour, not noise: a deeper"
echo "draft needs a bigger MTP context and more recurrent-state slots, so it trades"
echo "layers for draft depth. What matters is the resulting tg."
