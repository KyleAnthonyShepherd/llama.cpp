#!/usr/bin/env bash
# Numerical oracle for the expert tier.
#
# Comparing generated text does not work. The tier splits one mul_mat_id into a GPU hot path and
# a CPU cold path and adds them, which reassociates the sum, so the last bits differ from stock
# even when the tier is correct. At temperature 0 that eventually flips a token and the rest of
# the generation diverges for a reason that is not a bug.
#
# So compare the logprobs of the FIRST generated token, before anything can cascade. Rounding
# shows up as a tiny spread across the top-k. Duplicate-id corruption does not: it drops or
# misroutes whole expert rows, which moves logits by far more than rounding ever does.

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="${BIN:-$REPO_ROOT/build/bin/Release/llama-server.exe}"
[ -x "$BIN" ] || BIN="$REPO_ROOT/build/bin/llama-server"
MODEL="${MODEL:-$(cat "$OUT/model-path.txt")}"
TAG="${TAG:-q4ks}"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-4096}"
N_PROBS="${N_PROBS:-40}"
PROMPT="${PROMPT:-The three laws of thermodynamics state that}"

stop_server() {
    curl -s -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    taskkill //F //IM llama-server.exe >/dev/null 2>&1
    pkill -f "llama-server .*--port $PORT" >/dev/null 2>&1
    sleep 3
}

run_arm() {
    local name="$1"; shift
    local json="$OUT/oracle-$TAG-$name.json"

    stop_server
    if [ -n "${COLD_ONLY:-}" ]; then
        export LLAMA_EXPERT_TIER_COLD_ONLY=1
    else
        unset LLAMA_EXPERT_TIER_COLD_ONLY
    fi
    "$BIN" -m "$MODEL" -c "$N_CTX" -np 1 -lv 2 --host 127.0.0.1 --port "$PORT" "$@" \
        > "$OUT/oracle-$TAG-$name.log" 2>&1 &

    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 900 ]; then echo "=== $name: server never became ready ==="; stop_server; return; fi
    done

    curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":1,\"temperature\":0,\"n_probs\":$N_PROBS}" -o "$json"
    stop_server
    echo "$json"
}

mkdir -p "$OUT"
# -cmoe on the reference too. Any -ehs above 0 forces every MoE weight to the CPU
# (arg.cpp, and common.cpp for the autofit case), so a bare -ehs 0 reference would differ in
# tensor placement as well as in the tier, and the placement difference is the larger of the two.
REF="$(run_arm ref -ehs 0 -cmoe)"
for c in "${@:-cache}"; do
    case "$c" in
        cache) TEST="$(run_arm cache -ehs -1 -fitt 256)" ;;
        # does the error grow with how much traffic the hot path takes? reassociation says yes and
        # says it should be small at S=1. a big error at S=1 points at the split logic instead.
        s*)    TEST="$(run_arm "$c" -ehs "${c#s}" --expert-hyst 0 --expert-dwell 0)" ;;
        # the whole MoE through the cold kernel, hot path contributing zeros. what is left is
        # mul_mat_id_cold against stock mul_mat_id, with nothing else in the way.
        coldonly) TEST="$(COLD_ONLY=1 run_arm coldonly -ehs 1 --expert-hyst 0 --expert-dwell 0)" ;;
        *)     echo "unknown arm: $c"; continue ;;
    esac
    echo "=== $c ==="
    python "$PHASE0_DIR/oracle-logprobs.py" "$REF" "$TEST"
done
