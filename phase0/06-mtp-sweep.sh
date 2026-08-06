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
#     verify batch, and acceptance decays per position (measured: 0.947, 0.737, 0.632).
#   - "mean len" tells you the effective tokens per pass over the weights, which is the
#     quantity that actually beats the DRAM roofline.
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

echo "model   : $MODEL"
echo "n_ctx   : $N_CTX   n_gen: $N_GEN"
echo "n_max   : $N_MAX_LIST   (0 = MTP disabled, the baseline)"
echo "threads : ${THREADS:-default}"
echo "ngl     : ${NGL:-fitter chooses}"
echo

run_one() {
    local n_max="$1"
    local name="nmax$n_max"
    local args=()

    if [ "$n_max" = "0" ]; then
        name="baseline-nospec"
    else
        args=(--spec-type draft-mtp --spec-draft-n-max "$n_max")
    fi

    echo "--- n_max=$n_max ---"
    drop_caches

    "$CLI" -m "$MODEL" -c "$N_CTX" -n "$N_GEN" -no-cnv -st -lv 4 \
        "${NGL_ARGS[@]}" "${THREAD_ARGS[@]}" "${args[@]}" \
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
    printf "%s,%s,%s,%s,%s\n" "$n_max" "${ngl_used:-}" "${tg:-}" "${acc:-}" "${mlen:-}" \
        >> "$OUT/summary.csv"
}

echo "n_max,ngl,tg_tps,acceptance,mean_len" > "$OUT/summary.csv"
for n in $N_MAX_LIST; do
    run_one "$n"
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
