#!/usr/bin/env bash
# Find, per quant type, the largest batch (columns) for which the CUDA mat-vec kernel (MMVQ) beats
# the matrix kernel (MMQ) on THIS GPU, and print the result as:
#   1. a GGML_CUDA_MMVQ_MAX_COLS value to put in the server's environment (no rebuild needed), and
#   2. a C++ entry for ggml_cuda_should_use_mmvq() in ggml/src/ggml-cuda/mmvq.cu, to bake it in.
#
# Why: MMVQ re-decodes the quantized weights for every column, so its cost grows with the batch,
# while MMQ decodes once and stays nearly flat. Speculative decoding verifies 2-6 tokens per step,
# right where the two cross. The upstream table only has entries for some GPUs; everything else
# uses MMVQ up to 8 columns. On an RTX 3060 Laptop this was worth +17% at MTP depth 4.
#
# Needs a build of this fork from the commit that added this script or later (per-type
# GGML_CUDA_MMVQ_MAX_COLS), with test-backend-ops built (part of the default build).
# Nothing else: bash, awk, nvidia-smi.
#
# Usage:
#   BIN_DIR=./build/bin ./phase0/mmvq-table.sh                 # common types
#   BIN_DIR=./build/bin ./phase0/mmvq-table.sh q4_K q6_K q8_0  # only these
#
# Run it with nothing else on the GPU, and with the machine in the power mode you serve in.
# It takes about 1-2 minutes per type. Results go to ./mmvq-table-<host>/ (raw logs + summary.txt).

set -euo pipefail

BIN_DIR="${BIN_DIR:-./build/bin}"
TBO="$BIN_DIR/test-backend-ops"
DEV="${DEV:-CUDA0}"
MARGIN="${MARGIN:-1.03}"   # MMVQ keeps a batch size unless MMQ is faster by more than this factor

if [ ! -x "$TBO" ]; then
    echo "error: $TBO not found. Build with: cmake --build build --target test-backend-ops" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    TYPES=("$@")
else
    TYPES=(q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K iq4_xs iq4_nl iq3_s iq2_xs mxfp4)
fi

OUT="mmvq-table-$(hostname)"
mkdir -p "$OUT"

TYPE_RE="$(IFS='|'; echo "${TYPES[*]}")"
# the perf set times every type at m=4096, k=14336 for 1, 2, 3, 4, 5 and 8 columns
FILTER="type_a=($TYPE_RE),type_b=f32,m=4096,n=[1-8],k=14336,"

{
    echo "== GPU"
    nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit --format=csv
    echo
} > "$OUT/summary.txt"

# sample clocks and power while the kernels run, so the result can be read against the power cap
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 500 > "$OUT/clocks.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

run_mode() { # $1 = value of GGML_CUDA_MMVQ_MAX_COLS, $2 = log name
    echo "-- timing with GGML_CUDA_MMVQ_MAX_COLS=$1 ($2)" >&2
    # stdout only: debug lines on stderr would split the result lines
    GGML_CUDA_MMVQ_MAX_COLS="$1" "$TBO" perf -o MUL_MAT -b "$DEV" -p "$FILTER" > "$OUT/$2.log" 2> "$OUT/$2.err" || true
    # -> "type n us"
    sed -n 's/.*MUL_MAT(type_a=\([A-Za-z0-9_]*\),type_b=f32,m=4096,n=\([0-9]*\),k=14336.* \([0-9.]*\) us\/run.*/\1 \2 \3/p' "$OUT/$2.log"
}

run_mode 8 mmvq > "$OUT/mmvq.txt"
run_mode 0 mmq  > "$OUT/mmq.txt"

kill $SMI_PID 2>/dev/null || true

if [ ! -s "$OUT/mmvq.txt" ]; then
    echo "error: no timings parsed; see $OUT/mmvq.log (is this build new enough?)" >&2
    exit 1
fi

# join the two runs and pick the crossover per type
awk -v margin="$MARGIN" -v out="$OUT/summary.txt" '
    FNR == NR { mvq[$1 " " $2] = $3; types[$1] = 1; ns[$2] = 1; next }
    { mmq[$1 " " $2] = $3 }
    END {
        nn = 0
        for (n in ns) nlist[++nn] = n + 0
        # sort the batch sizes
        for (i = 1; i <= nn; i++) for (j = i + 1; j <= nn; j++) if (nlist[j] < nlist[i]) { t = nlist[i]; nlist[i] = nlist[j]; nlist[j] = t }

        printf "%-8s", "type" >> out
        for (i = 1; i <= nn; i++) printf "  n=%-17d", nlist[i] >> out
        printf "  -> mvq up to\n" >> out

        env = ""
        cpp = ""
        for (ty in types) {
            printf "%-8s", ty >> out
            # mvq keeps every batch size up to the first one where mmq is clearly faster
            best = 0
            lost = 0
            for (i = 1; i <= nn; i++) {
                n = nlist[i]; k = ty " " n
                a = mvq[k]; b = mmq[k]
                if (a == "" || b == "") { printf "  %-19s", "-" >> out; continue }
                printf "  mvq %6.1f mmq %6.1f", a, b >> out
                if (!lost && a <= b * margin) {
                    best = n
                } else {
                    lost = 1
                }
            }
            # the perf set skips 6 and 7: if 5 wins and 8 loses, this stays at 5
            printf "  -> %d\n", best >> out
            env = env (env == "" ? "" : ",") ty "=" best
            cpp = cpp sprintf("            case GGML_TYPE_%s:\n                return ne11 <= %d;\n", toupper(ty), best)
        }

        printf "\n== put this in the server environment (no rebuild needed):\n" >> out
        printf "export GGML_CUDA_MMVQ_MAX_COLS=%s,default=8\n", env >> out

        printf "\n== or bake it into ggml_cuda_should_use_mmvq() (ggml/src/ggml-cuda/mmvq.cu), next to the sm_86 entry.\n" >> out
        printf "== note: cc alone cannot tell a GTX 16xx from an RTX 20xx (both 750); keep the env route if that matters.\n" >> out
        printf "    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == CC_OF_THIS_GPU) {\n        switch (type) { // tuned on THIS_GPU\n%s            default:\n                return ne11 <= MMVQ_MAX_BATCH_SIZE;\n        }\n    }\n", cpp >> out
    }
' "$OUT/mmvq.txt" "$OUT/mmq.txt"

{
    echo
    echo "== clocks while timing (sm MHz, mem MHz, W, throttle reasons; 0x4 = power cap):"
    sort "$OUT/clocks.csv" | uniq -c | sort -rn | head -8
    echo
    echo "Check the result end to end before trusting it: run phase0/qwen35-4b-mtp-sweep.py with and"
    echo "without the export above, at the draft depth you serve with. The microbenchmark shape"
    echo "(4096 x 14336) is not the model's, and small matrices may cross over at other batch sizes."
} >> "$OUT/summary.txt"

cat "$OUT/summary.txt"
