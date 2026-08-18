#!/usr/bin/env bash
# Sweep ngram-mod settings on ordinary prompts, not copy-heavy ones.
#
# The wide config (n_min 48) only pays on long verbatim runs. On prose the table almost never has
# a 48 token continuation, so it never fires and the width just costs output buffer. This asks
# whether any setting earns its keep on a normal chat/code workload.
#
# Every arm runs the same 5 prompts through one server, with the user's sampling and a fixed seed,
# and reports aggregate tokens/second over the whole set.

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="${BIN:-$REPO_ROOT/build-cuda/bin/Release/llama-server.exe}"
MODEL="${MODEL:-$(cat "$OUT/model-path.txt")}"
TAG="${TAG:-ord}"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-8192}"
N_PREDICT="${N_PREDICT:-400}"
SEED="${SEED:-1234}"

# the user's typical sampling
SAMPLING='"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repeat_penalty":1.0'

stop_server() {
    curl -s -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    taskkill //F //IM llama-server.exe >/dev/null 2>&1
    pkill -f "llama-server .*--port $PORT" >/dev/null 2>&1
    sleep 3
}

run_case() {
    local name="$1"; shift
    local log="$OUT/ord-$TAG-$name.log"
    local out="$OUT/ord-$TAG-$name.jsonl"
    : > "$out"

    stop_server
    unset LLAMA_EXPERT_HITRATE
    "$BIN" -m "$MODEL" -c "$N_CTX" -np 1 -lv 4 --host 127.0.0.1 --port "$PORT" \
        -ehs -1 -fitt 256 --expert-tier-max-tokens 65 --flash-attn on "$@" > "$log" 2>&1 &

    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 900 ]; then echo "=== $name: server never became ready ==="; tail -5 "$log"; stop_server; return; fi
    done

    python - "$PHASE0_DIR/prompts/ordinary.json" "$N_PREDICT" "$SEED" "$PORT" "$SAMPLING" >> "$out" <<'PY'
import json, sys, urllib.request
prompts = json.load(open(sys.argv[1], encoding="utf-8"))
n_predict, seed, port, sampling = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
for p in prompts:
    body = "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    payload = json.loads("{" + sampling + "}")
    payload.update({"prompt": body, "n_predict": n_predict, "seed": seed, "cache_prompt": False})
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    print(json.dumps({"timings": d.get("timings", {}), "n": len(d.get("content",""))}))
PY

    echo "=== $name : $* ==="
    python "$PHASE0_DIR/ngram-ordinary-report.py" "$out" "$log"
    stop_server
}

mkdir -p "$OUT"
MTP='--spec-type draft-mtp --spec-draft-n-max 1'
both() { echo "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 1 --spec-ngram-mod-n-match $1 --spec-ngram-mod-n-min $2 --spec-ngram-mod-n-max $3"; }

for c in "${@:-sweep}"; do
    case "$c" in
        sweep)
            run_case mtp-only        $MTP
            run_case m24-48-64       $(both 24 48 64)
            run_case m24-04-16       $(both 24 4 16)
            run_case m16-04-16       $(both 16 4 16)
            run_case m16-02-08       $(both 16 2 8)
            run_case m08-02-08       $(both 8 2 8)
            run_case m08-01-04       $(both 8 1 4)
            run_case m32-08-32       $(both 32 8 32)
            ;;
        # the other n-gram speculators, in case ngram-mod's modulo table is simply the wrong
        # structure for prose. mtp-only interleaved as the noise reference.
        alt)
            run_case mtp-ref-a       $MTP
            run_case ngram-cache     --spec-type draft-mtp,ngram-cache --spec-draft-n-max 1
            run_case mtp-ref-b       $MTP
            run_case ngram-simple-8  --spec-type draft-mtp,ngram-simple --spec-draft-n-max 1                 --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 1
            run_case mtp-ref-c       $MTP
            run_case ngram-k4v-8     --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 1                 --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 1
            ;;
        # What does leaving ngram-mod on cost when it never fires? ABBA order, not ABAB:
        # alternating puts every A on an odd position and every B on an even one, so any
        # position effect lands entirely on the arm and looks like a result.
        idle)
            run_case idle-mtp-1  $MTP
            run_case idle-ng-1   $(both 24 48 64)
            run_case idle-ng-2   $(both 24 48 64)
            run_case idle-mtp-2  $MTP
            run_case idle-mtp-3  $MTP
            run_case idle-ng-3   $(both 24 48 64)
            run_case idle-ng-4   $(both 24 48 64)
            run_case idle-mtp-4  $MTP
            ;;
        *) echo "unknown case: $c" ;;
    esac
done
