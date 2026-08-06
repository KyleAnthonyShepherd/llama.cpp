#!/usr/bin/env bash
# What does the fitter decide, and how much VRAM does it actually need?
#
# Three things come out of this:
#   1. the fitted -ngl / -ot for a range of --fit-target margins (the default is
#      1024 MiB, which is 17% of a 6 GB card - PLAN section 2.4)
#   2. the per-device model/context/compute breakdown from -fitp on
#   3. the same, with --mtp, to size the under-count described in PLAN section 3.2
set -euo pipefail
source "$(dirname "$0")/config.sh"
resolve_model

FIT="$BIN_DIR/llama-fit-params"
[ -x "$FIT" ] || { echo "error: $FIT not found; run 00-build.sh" >&2; exit 1; }

OUT="$RESULTS_DIR/03-fit"
mkdir -p "$OUT"
record_env "$OUT/env.txt"

echo "model : $MODEL"
echo "n_ctx : $N_CTX"
echo

# --- 1. free VRAM floor ------------------------------------------------------
# The fitter subtracts the margin from whatever ggml_backend_dev_memory() reports
# (common/fit.cpp:107). Record what the device reports with nothing loaded so we
# know how much of the 6 GB is already gone to the driver.
echo "=== idle GPU state ==="
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv | tee "$OUT/gpu-idle.txt"
echo

# --- 2. margin sweep ---------------------------------------------------------
echo "=== fitted args by --fit-target margin ==="
: > "$OUT/margin-sweep.txt"
for MARGIN in 1024 768 512 384 256 192 128; do
    echo "--- -fitt $MARGIN ---" | tee -a "$OUT/margin-sweep.txt"
    if "$FIT" -m "$MODEL" -c "$N_CTX" -fitt "$MARGIN" \
            > "$OUT/fit-$MARGIN.stdout" 2> "$OUT/fit-$MARGIN.stderr"; then
        cat "$OUT/fit-$MARGIN.stdout" | tee -a "$OUT/margin-sweep.txt"
    else
        echo "  FAILED (see fit-$MARGIN.stderr)" | tee -a "$OUT/margin-sweep.txt"
    fi
done
echo

# --- 3. memory breakdown -----------------------------------------------------
# -fitp on prints "device model context compute" in MiB (common/fit.cpp:960-984).
echo "=== memory breakdown (no MTP - see note below) ==="
"$FIT" -m "$MODEL" -c "$N_CTX" -fitp on 2>&1 | tee "$OUT/breakdown-nomtp.txt"
echo
echo "note: there is deliberately no --mtp run here. llama-fit-params CANNOT be told"
echo "      to model an MTP run at all: --mtp is registered for the download example"
echo "      only (common/arg.cpp:3009) and --spec-type only for speculative/server/cli"
echo "      (common/arg.cpp:4102), and options outside the current example are never"
echo "      registered, so their env vars are not consulted either (common/arg.cpp:1385)."
echo "      The MTP VRAM cost is measured for real in 04-residency.sh instead."
echo

# --- 4. the verbose fit trace ------------------------------------------------
# --verbose promotes the fitter's LOG_TRC lines, which show every bisection step
# and, critically, whether it emitted any tensor overrides at all. For a dense
# model I expect none (PLAN section 0.2) - this is where that gets confirmed.
echo "=== verbose fit trace (this is the one that confirms PLAN 0.2) ==="
"$FIT" -m "$MODEL" -c "$N_CTX" -fitt 512 --verbose > "$OUT/fit-verbose.stdout" 2> "$OUT/fit-verbose.stderr" || true
echo "wrote $OUT/fit-verbose.{stdout,stderr}"
echo
echo "quick check - number of tensor overrides the fitter produced:"
grep -c 'buffer type overridden' "$OUT/fit-verbose.stderr" 2>/dev/null || echo "0"
echo
echo "context-reduction steps (fitter shrinking n_ctx to fit):"
grep -i 'n_ctx\|reduc' "$OUT/fit-verbose.stderr" 2>/dev/null | head -20 || true

echo
echo "results in $OUT"
