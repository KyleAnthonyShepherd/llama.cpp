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

CLI="$BIN_DIR/llama-cli"
[ -x "$CLI" ] || { echo "error: $CLI not found; run 00-build.sh" >&2; exit 1; }

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
    NGL_ARGS=(-ngl "$NGL")
fi

PROMPT="${PROMPT:-Explain in detail how a memory-mapped file differs from a heap allocation.}"

run_case() {
    local name="$1"; shift
    echo
    echo "================================================================"
    echo "case: $name"
    echo "args: $*"
    echo "================================================================"
    drop_caches

    local before_major
    before_major=$(awk '/^pgmajfault /{print $2}' /proc/vmstat)

    local t_start
    t_start=$(date +%s)

    "$CLI" -m "$MODEL" -c "$N_CTX" -n "$N_GEN" -no-cnv -st \
        -p "$PROMPT" "$@" > "$OUT/$name.log" 2>&1 &
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
        echo "--- timings ---"
        grep -E 'eval time|total time|load time' "$OUT/$name.log" || true
        echo "--- memory breakdown ---"
        sed -n '/memory breakdown/,/^$/p' "$OUT/$name.log" || true
    } | tee "$OUT/$name.summary.txt"
}

# A. the repack A/B. Same everything else.
run_case "default"   "${NGL_ARGS[@]}"
run_case "norepack"  "${NGL_ARGS[@]}" -nr

# B. mlock is the opposite extreme - forces residency and will fail or swap on a
# 16 GB box with an 18 GB model. Included because a failure here is itself a
# clean datapoint about how much really has to be resident.
run_case "mmap-mlock" "${NGL_ARGS[@]}" -lm mmap+mlock || true

# C. MTP. --spec-type is available in the cli example (common/arg.cpp:4102).
# Skips itself cleanly if the checkpoint has no MTP layers.
if [ -n "$MTP_MODEL" ]; then
    run_case "mtp" "${NGL_ARGS[@]}" -nr --spec-type draft-mtp -md "$MTP_MODEL" || true
else
    run_case "mtp" "${NGL_ARGS[@]}" -nr --spec-type draft-mtp || true
fi

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
echo "  default vs norepack: a large peak_anon in 'default' that moves to peak_file"
echo "    in 'norepack' confirms PLAN section 1.2 - repack is what makes the CPU-side"
echo "    weights swappable."
echo "  swap_mb > 0 anywhere means we are genuinely paging, not just evicting."
echo "  the mtp case's gpu_mb minus the default case's gpu_mb is the VRAM that"
echo "    llama-fit-params cannot see (PLAN section 3.2)."
echo
echo "results in $OUT"
