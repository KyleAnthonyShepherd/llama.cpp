#!/usr/bin/env bash
# A/B two builds of the dp4a CUDA code: the server's old binaries (A) against a build with new MMQ configs (B).
#
# Why: the GTX 1660 Ti has no tensor cores, so the server builds with
# -DGGML_CUDA_FORCE_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=61-virtual (PTX only, JIT-compiled for sm_75 with
# __CUDA_ARCH__ = 610, so every kernel takes the dp4a / non-mma path). There, MMQ at 1-8 columns cost ~2.6x
# a 1-column mat-vec, which is what speculative verify batches pay. mmq-config-pascal-dp4a.cuh now uses
# smaller blocks for J <= 16.
#
# Steps (first start JIT-compiles the PTX; the cache size below keeps it):
#   cp -a build/bin bin-stock                      # keep the old binaries as A, before pulling
#   git pull && cmake --build build --config Release -j $(nproc) --target llama-server llama-bench test-backend-ops
#   MODEL=~/models/Qwen3.5-4B-Q4_K_S.gguf ./phase0/sm75-dp4a-ab.sh
#   SWEEP=1 DRAFT=~/models/qwen35-4b-dflash-Q4_K_M.gguf MODEL=... ./phase0/sm75-dp4a-ab.sh
# A's test-backend-ops has no 16-column perf case, so its n=16 cells stay empty.
# Run it with nothing else on the GPU. Env: BIN_A (default ./bin-stock), BIN_B (default ./build/bin), DEV (CUDA0).
# Results go to ./sm75-ab-<host>/ (summary.txt + raw logs).

set -euo pipefail

BIN_A="${BIN_A:-./bin-stock}"
BIN_B="${BIN_B:-./build/bin}"
DEV="${DEV:-CUDA0}"
MODEL="${MODEL:?set MODEL to the target GGUF}"
export CUDA_CACHE_MAXSIZE=4294967296

for b in "$BIN_A" "$BIN_B"; do
    for t in test-backend-ops llama-bench; do
        [ -x "$b/$t" ] || { echo "error: $b/$t not found" >&2; exit 1; }
    done
done

OUT="sm75-ab-$(hostname)"
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,power.limit --format=csv > "$SUM"

# 1. kernels: MUL_MAT 4096 x 14336 at 1-16 columns, MMVQ allowed up to 8 columns (=8) or MMQ only (=0)
FILTER="type_a=(q4_K|q5_K|q6_K|q8_0),type_b=f32,m=4096,n=(1|2|3|4|5|8|16),k=14336,"
time_mm() { # $1 = bin dir, $2 = GGML_CUDA_MMVQ_MAX_COLS, $3 = log name
    GGML_CUDA_MMVQ_MAX_COLS="$2" "$1/test-backend-ops" perf -o MUL_MAT -b "$DEV" -p "$FILTER" > "$OUT/$3.log" 2> "$OUT/$3.err" || true
    sed -n 's/.*MUL_MAT(type_a=\([A-Za-z0-9_]*\),type_b=f32,m=4096,n=\([0-9]*\),k=14336.* \([0-9.]*\) us\/run.*/\1 \2 \3/p' "$OUT/$3.log"
}
echo "-- kernels" >&2
time_mm "$BIN_A" 8 a_mvq > "$OUT/a_mvq.txt"
time_mm "$BIN_A" 0 a_mmq > "$OUT/a_mmq.txt"
time_mm "$BIN_B" 8 b_mvq > "$OUT/b_mvq.txt"
time_mm "$BIN_B" 0 b_mmq > "$OUT/b_mmq.txt"

{
    echo
    echo "== MUL_MAT us/run (A = $BIN_A, B = $BIN_B; mvq = MMVQ up to 8 columns (MMQ above), mmq = MMQ only)"
    printf "%-6s %4s %9s %9s %9s %9s\n" type n A_mvq A_mmq B_mvq B_mmq
    awk '{ k = $1 " " $2; v[FILENAME, k] = $3; keys[k] = 1 }
         END { for (k in keys) { split(k, p, " ");
               printf "%-6s %4d %9.1f %9.1f %9.1f %9.1f\n", p[1], p[2],
                   v[ARGV[1], k], v[ARGV[2], k], v[ARGV[3], k], v[ARGV[4], k] } }' \
        "$OUT/a_mvq.txt" "$OUT/a_mmq.txt" "$OUT/b_mvq.txt" "$OUT/b_mmq.txt" | sort -k1,1 -k2,2n
} >> "$SUM"

# 2. whole model, no speculation
echo "-- llama-bench" >&2
for side in A B; do
    b="$BIN_A"; [ "$side" = B ] && b="$BIN_B"
    echo -e "\n== llama-bench $side ($b)" >> "$SUM"
    "$b/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 3 2> "$OUT/bench-$side.err" | tee "$OUT/bench-$side.log" | grep -E '^\|' >> "$SUM"
done

# 3. optional: the 5-prompt sweep, ABBA (A, B, B, A)
if [ "${SWEEP:-0}" = 1 ]; then
    CFG="nospec=;mtp2=--spec-type draft-mtp --spec-draft-n-max 2 -bs;mtp3=--spec-type draft-mtp --spec-draft-n-max 3 -bs"
    if [ -n "${DRAFT:-}" ]; then
        CFG="$CFG;df3=-md $DRAFT -ngld 99 --spec-type draft-dflash --spec-draft-n-max 3 -bs;df7=-md $DRAFT -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7 -bs"
    fi
    for side in A B B A; do
        b="$BIN_A"; [ "$side" = B ] && b="$BIN_B"
        echo "-- sweep $side" >&2
        python3 phase0/qwen35-4b-mtp-sweep.py --server "$b/llama-server" --model "$MODEL" \
            --base "-ngl 99 -fa on -c 32768 -np 1 --no-webui" --configs "$CFG" \
            --out "$OUT/sweep-$side.jsonl" --logdir "$OUT/logs-$side"
    done
    echo -e "\n== sweep: per-config mean t/s (A then B; both passes of each side averaged)" >> "$SUM"
    python3 - "$OUT" >> "$SUM" <<'EOF'
import json, sys, collections
out = sys.argv[1]
for side in "AB":
    agg = collections.defaultdict(list)
    for l in open(f"{out}/sweep-{side}.jsonl"):
        r = json.loads(l)
        agg[(r["cfg"], r["sampling"])].append(r["tps"])
    for (cfg, samp), v in sorted(agg.items()):
        print(f"{side} {cfg:8s} {samp:6s} {sum(v)/len(v):7.1f}")
EOF
fi

cat "$SUM"
