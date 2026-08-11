#!/usr/bin/env bash
# Server-based A/B: expert cache x MTP. The server is the only harness that
# registers --spec-type (common/arg.cpp:4163) AND that drives context growth
# (llama_set_n_ctx is called only from tools/server/server-context.cpp).
#
# Long generations on purpose: the hot store needs ~300+ tokens to converge,
# so short runs report a false negative (PLAN section 15.2).

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="$REPO_ROOT/build/bin/Release/llama-server.exe"
MODEL="$(cat "$OUT/model-path.txt")"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-4096}"
N_PREDICT="${N_PREDICT:-900}"
PROMPT="Write a detailed numbered list of forty distinct facts about the history of computing, each three sentences long."

stop_server() {
    curl -s -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    taskkill //F //IM llama-server.exe >/dev/null 2>&1
    sleep 3
}

run_case() {
    local name="$1"; shift
    local log="$OUT/srv-$name.log"
    local json="$OUT/srv-$name.json"

    stop_server
    LLAMA_EXPERT_HITRATE=1 "$BIN" -m "$MODEL" -c "$N_CTX" -np 1 -lv 4 \
        --host 127.0.0.1 --port "$PORT" "$@" > "$log" 2>&1 &
    local pid=$!

    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 300 ]; then echo "=== $name: server never became ready ==="; cat "$log" | tail -5; stop_server; return; fi
    done

    curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT,\"temperature\":0}" -o "$json"

    echo "=== $name : $* ==="
    python - "$json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
t=d.get('timings',{})
print("  pp %.2f t/s (%s tok) | tg %.2f t/s (%s tok)" % (
    t.get('prompt_per_second') or 0, t.get('prompt_n'),
    t.get('predicted_per_second') or 0, t.get('predicted_n')))
PY
    grep -oE "hotstore sizing \(S=[0-9]+\)|expert tier engaged: n_tokens=[0-9]+|expert tier bypassed: n_tokens=[0-9]+ \(max [0-9]+\)|growing n_ctx: [0-9]+ -> [0-9]+" "$log" | sort -u | sed 's/^/  /'
    # mean hit rate over the run
    grep -oE "hot hit rate: [0-9]+/[0-9]+" "$log" | awk -F'[ /]' '{h+=$4;t+=$5;n++} END{if(n)printf "  mean hit rate: %.1f%% over %d samples\n",100*h/t,n}'
    grep -oE "n_draft = [0-9]+|accept rate = [0-9.]+|draft acceptance[^,]*" "$log" | sort -u | sed 's/^/  /'
    stop_server
}

mkdir -p "$OUT"
for c in "${@:-base cache mtp cachemtp}"; do
    case "$c" in
        base)     run_case base     -ehs 0 ;;
        cache)    run_case cache    -ehs -1 -fitt 256 ;;
        mtp)      run_case mtp      -ehs 0 --spec-type draft-mtp ;;
        cachemtp) run_case cachemtp -ehs -1 -fitt 256 --spec-type draft-mtp ;;
        *) echo "unknown case: $c" ;;
    esac
done
