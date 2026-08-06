#!/usr/bin/env bash
# The load-bearing measurement. Answers PLAN section 1.2 and 1.4:
#
#   A. Does --repack turn the CPU-side weights into anonymous (swappable) RAM?
#      Watch rss_anon vs rss_file between the "default" and "norepack" runs.
#      If the theory is right, rss_anon is many GiB in the default run and near
#      zero in the norepack run, with rss_file taking over.
#
#   B. How bad is the MAP_POPULATE startup storm? Watch the first ~60 s of the
#      csv and the wall time to first token.
#
#   C. What does an MTP run actually cost in VRAM, given llama-fit-params cannot
#      model it (PLAN section 3.2)?
#
# Each run also prints common_memory_breakdown_print() at exit (via
# common_perf_print, common/sampling.cpp:562) - that is the per-device
# model/context/compute/unaccounted table, captured in the .log files.
set -euo pipefail
source "$(dirname "$0")/config.sh"
resolve_model

# Use llama-completion, NOT llama-cli. llama-cli is the interactive TUI client: it
# forces params.verbosity = LOG_LEVEL_ERROR (tools/cli/cli.cpp:36), which hides the
# per-buffer "model buffer size" lines we need, and it ignores -no-cnv. Only the MTP
# case below falls back to llama-cli, because --spec-type is not registered for the
# completion example (common/arg.cpp:4102).
RUNNER="$BIN_DIR/llama-completion"
CLI="$BIN_DIR/llama-cli"
[ -x "$RUNNER" ] || { echo "error: $RUNNER not found; run 00-build.sh" >&2; exit 1; }

OUT="$RESULTS_DIR/04-residency"
mkdir -p "$OUT"
record_env "$OUT/env.txt"

# Leave NGL unset to let --fit decide. Do NOT default this to 99: setting -ngl
# explicitly makes the fitter bail out entirely ("n_gpu_layers already set by
# user, abort", common/fit.cpp:377-378), and llama-cli would then try to put all
# 65 layers on a 6 GB card. Once 05-ngl-sweep.sh tells you the good value, come
# back and pin it with NGL=<n>.
NGL="${NGL:-}"
NGL_ARGS=()
if [ -n "$NGL" ]; then
    # -fit off: with -ngl explicit the fitter aborts anyway (common/fit.cpp:377), so
    # skip its probe load rather than pay for it and log a confusing warning.
    NGL_ARGS=(-ngl "$NGL" -fit off)
fi

# -lv 4 is REQUIRED to see any library INFO output. common_get_verbosity maps
# GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE (4), not LOG_LEVEL_INFO (common/log.cpp:444),
# and the default threshold is 3 - so the loader's "model buffer size" lines and the
# memory breakdown table are both invisible without it.
LOG_ARGS=(-lv 4)

PROMPT="${PROMPT:-Explain in detail how a memory-mapped file differs from a heap allocation.}"

run_case() {
    local name="$1"; shift
    local bin="$1"; shift
    echo
    echo "================================================================"
    echo "case: $name   ($(basename "$bin"))"
    echo "args: $*"
    echo "================================================================"
    drop_caches

    local before_major
    before_major=$(awk '/^pgmajfault /{print $2}' /proc/vmstat)

    local t_start
    t_start=$(date +%s)

    "$bin" -m "$MODEL" -c "$N_CTX" -n "$N_GEN" -no-cnv -st "${LOG_ARGS[@]}" \
        -p "$PROMPT" "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
    local pid=$!

    "$PHASE0_DIR/sample-proc.sh" "$pid" "$OUT/$name.csv" 1 &
    local spid=$!

    wait "$pid" || echo "  (run exited non-zero - check $OUT/$name.log)"
    kill "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true

    local after_major t_end
    after_major=$(awk '/^pgmajfault /{print $2}' /proc/vmstat)
    t_end=$(date +%s)

    {
        echo "case: $name"
        echo "args: $*"
        echo "wall_seconds: $(( t_end - t_start ))"
        echo "system_pgmajfault_delta: $(( after_major - before_major ))"
        echo "peak_rss_anon_mb: $(awk -F, 'NR>1 && $2>m {m=$2} END{print m+0}' "$OUT/$name.csv")"
        echo "peak_rss_file_mb: $(awk -F, 'NR>1 && $3>m {m=$3} END{print m+0}' "$OUT/$name.csv")"
        echo "peak_vm_swap_mb:  $(awk -F, 'NR>1 && $5>m {m=$5} END{print m+0}' "$OUT/$name.csv")"
        echo "final_majflt:     $(awk -F, 'END{print $6}' "$OUT/$name.csv")"
        echo "peak_gpu_used_mb: $(awk -F, 'NR>1 && $8>m {m=$8} END{print m+0}' "$OUT/$name.csv")"
        echo "min_host_avail_mb: $(awk -F, 'NR>1 && ($9<m || m==0) {m=$9} END{print m+0}' "$OUT/$name.csv")"
        echo "--- buffer types (the PLAN 1.2 verdict) ---"
        grep 'model buffer size' "$OUT/$name.log" || echo "(none logged - wrong binary?)"
        echo "--- timings ---"
        grep -E 'eval time|total time|load time' "$OUT/$name.log" || true
        echo "--- memory breakdown ---"
        sed -n '/memory breakdown/,/^$/p' "$OUT/$name.log" || true
    } | tee "$OUT/$name.summary.txt"
}

# A. the buffer-type / residency matrix. cpu_buft_list is ordered
# ACCEL -> pinned host -> repack extras -> plain CPU (src/llama-model.cpp:896-950)
# and select_weight_buft takes the first that supports the op, so it can take three
# runs to fall all the way through to the mmap-able plain CPU buft.
run_case "default"       "$RUNNER" "${NGL_ARGS[@]}"
run_case "norepack"      "$RUNNER" "${NGL_ARGS[@]}" -nr
run_case "norepack-nohost" "$RUNNER" "${NGL_ARGS[@]}" -nr --no-host

# B. mlock is the opposite extreme - forces residency and will fail or swap on a
# 16 GiB box. A failure here is itself a clean datapoint about what must be resident.
run_case "mmap-mlock"    "$RUNNER" "${NGL_ARGS[@]}" -lm mmap+mlock || true

# C. MTP. --spec-type is registered for cli/server/speculative but NOT completion
# (common/arg.cpp:4102), so this one case has to go through llama-cli. Pass -v to undo
# the TUI's LOG_LEVEL_ERROR default (tools/cli/cli.cpp:36) or the log is empty.
MTP_ARGS=(-nr --spec-type draft-mtp)
if [ -n "$MTP_MODEL" ]; then
    MTP_ARGS+=(-md "$MTP_MODEL")
fi
run_case "mtp" "$CLI" "${NGL_ARGS[@]}" "${MTP_ARGS[@]}" || true

echo
echo "================================================================"
echo "comparison"
echo "================================================================"
printf "%-12s %12s %12s %10s %10s %10s\n" case peak_anon_mb peak_file_mb swap_mb majflt gpu_mb
for f in "$OUT"/*.summary.txt; do
    n=$(basename "$f" .summary.txt)
    a=$(awk -F': *' '/^peak_rss_anon_mb/{print $2}' "$f")
    b=$(awk -F': *' '/^peak_rss_file_mb/{print $2}' "$f")
    s=$(awk -F': *' '/^peak_vm_swap_mb/{print $2}' "$f")
    m=$(awk -F': *' '/^final_majflt/{print $2}' "$f")
    g=$(awk -F': *' '/^peak_gpu_used_mb/{print $2}' "$f")
    printf "%-12s %12s %12s %10s %10s %10s\n" "$n" "$a" "$b" "$s" "$m" "$g"
done

echo
echo "read it like this:"
echo "  peak_anon staying small (well under 1 GiB) in every case means the host-side"
echo "    weights are on the mmap path, not copied - that is the good outcome, and it is"
echo "    what was measured on 2026-08-06. A multi-GiB peak_anon would mean repack or a"
echo "    pinned-host buffer took over (PLAN section 1.2)."
echo "  swap_mb identical across cases is baseline swap from other processes, not"
echo "    something llama.cpp caused. Watch whether it GROWS with the case, not whether"
echo "    it is nonzero."
echo "  majflt is the real pressure signal - MemAvailable is not, because clean mmap'd"
echo "    weights count as reclaimable and keep it looking healthy."
echo "  the mtp case is expected to be FASTER despite fewer GPU layers, and to show a"
echo "    much larger recurrent-state allocation (n_rs_seq goes 0 -> 3)."
echo
echo "results in $OUT"
