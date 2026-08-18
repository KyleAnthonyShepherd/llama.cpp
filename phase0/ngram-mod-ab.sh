#!/usr/bin/env bash
# Step 0 of PLAN-ngram-mod-expert-cache.md: is a wide ngram-mod draft worth
# engaging the expert tier for at all?
#
# The workload is copy-heavy on purpose. ngram-mod only fires on long verbatim
# runs, so a prose prompt reports a fire rate near zero and answers nothing.
# The prompt primes two copies of a document and lets the model continue; every
# further copy is a verbatim run far longer than n_match.
#
# E1  op-offload A/B at a 49-65 token verify batch. At >= 32 tokens the CUDA
#     backend streams the whole _exps tensors over PCIe instead of running the
#     MoE on the CPU, so this asks which side of that trade the box is on.
# E2  fire rate. read from the per-impl statistics line: n_gen_drafts over
#     n_call_draft is the fraction of decode steps that drafted at all.
# E3  union of experts touched per layer per batch. needs LLAMA_EXPERT_HITRATE
#     and a hot store, and sweeps the verify batch through the tier ceiling,
#     the op-offload threshold and past it.

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="${BIN:-$REPO_ROOT/build/bin/Release/llama-server.exe}"
[ -x "$BIN" ] || BIN="$REPO_ROOT/build/bin/llama-server"
MODEL="${MODEL:-$(cat "$OUT/model-path.txt")}"
TAG="${TAG:-q4ks}"
DOC="${DOC:-$PHASE0_DIR/prompts/copy-heavy.txt}"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-8192}"
N_PREDICT="${N_PREDICT:-900}"
N_MATCH="${N_MATCH:-24}"

REQ="$OUT/.req-$TAG.json"

build_request() {
    python - "$DOC" "$N_PREDICT" > "$REQ" <<'PY'
import json, sys
doc = open(sys.argv[1], encoding="utf-8").read().rstrip("\n")
prompt = ("Copy the document below verbatim, over and over, numbering each copy.\n\n"
          "=== DOCUMENT ===\n" + doc +
          "\n\n=== COPY 1 ===\n" + doc +
          "\n\n=== COPY 2 ===\n")
json.dump({"prompt": prompt, "n_predict": int(sys.argv[2]), "temperature": 0}, sys.stdout)
PY
}

stop_server() {
    curl -s -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    taskkill //F //IM llama-server.exe >/dev/null 2>&1   # windows
    pkill -f "llama-server .*--port $PORT" >/dev/null 2>&1
    sleep 3
}

run_case() {
    local name="$1"; shift
    local log="$OUT/ng-$TAG-$name.log"
    local json="$OUT/ng-$TAG-$name.json"

    stop_server
    # the hit rate readback forces a synchronize on every decode, so it costs throughput.
    # keep it off for any case whose tg number is the result.
    if [ "${HITRATE:-1}" = "1" ]; then
        export LLAMA_EXPERT_HITRATE=1
    else
        unset LLAMA_EXPERT_HITRATE
    fi
    if [ -n "${OFFLOAD_MIN_BATCH:-}" ]; then
        export GGML_OP_OFFLOAD_MIN_BATCH="$OFFLOAD_MIN_BATCH"
    else
        unset GGML_OP_OFFLOAD_MIN_BATCH
    fi
    "$BIN" -m "$MODEL" -c "$N_CTX" -np 1 -lv 4 \
        --host 127.0.0.1 --port "$PORT" "$@" > "$log" 2>&1 &

    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 900 ]; then echo "=== $name: server never became ready ==="; tail -5 "$log"; stop_server; return; fi
    done

    curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
        -d "@$REQ" -o "$json"

    echo "=== $name : $* ==="
    python "$PHASE0_DIR/ngram-mod-report.py" "$json" "$log"
    stop_server
}

mkdir -p "$OUT"
build_request

# n_min == n_max makes the draft width exactly N, so the verify batch is exactly N+1.
ngram() { echo "--spec-type ngram-mod --spec-ngram-mod-n-match $N_MATCH --spec-ngram-mod-n-min $1 --spec-ngram-mod-n-max $1"; }

for c in "${@:-e1}"; do
    case "$c" in
        # E1/E2: the proposed config, stock ceiling, no hot store.
        e1) run_case nospec-offload-on  -ehs 0
            run_case wide-offload-on    -ehs 0 --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            run_case wide-offload-off   -ehs 0 --no-op-offload --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            ;;
        # E1b: --no-op-offload also takes prompt processing off the GPU, and a 512-token
        # prompt ubatch is exactly where op-offload pays. moving the threshold above the
        # verify batch instead keeps both.
        e1b) for t in 128 512; do
                OFFLOAD_MIN_BATCH=$t run_case "wide-offload-min$t" -ehs 0 --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            done
            ;;
        # E3: union sweep. hot store on so the hit rate is reported next to the union.
        e3) for m in 1 3 7 15 31 63; do
                run_case "u-m$((m+1))" -ehs -1 -fitt 256 $(ngram $m)
            done
            ;;
        # E4: which draft width is actually fastest, with the MoE already pinned to the CPU
        # by E1b's threshold. sets the ceiling target for a tier that has to reach it.
        e4) for m in 1 3 7 15 31 63; do
                HITRATE=0 OFFLOAD_MIN_BATCH=128 run_case "w-m$((m+1))" -ehs 0 $(ngram $m)
            done
            ;;
        # E5: does the CLI flag reach the backend before it registers? if it does, flag-min128
        # reproduces e1b's env-var number and ehs-implicit logs the raise on its own.
        e5) HITRATE=0 run_case flag-min128 -ehs 0 --op-offload-min-batch 128 --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            HITRATE=0 run_case flag-min32  -ehs 0 --op-offload-min-batch 32  --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            HITRATE=0 run_case ehs-implicit -ehs -1 -fitt 256 --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            ;;
        # E6: the ceiling raised past the verify batch. Arms interleaved so thermal drift lands
        # on both equally; this box moves ~10% between identical runs.
        e6) for rep in 1 2; do
                HITRATE=0 run_case "ceil4-r$rep"  -ehs -1 -fitt 256                     --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
                HITRATE=0 run_case "ceil65-r$rep" -ehs -1 -fitt 256 --expert-tier-max-tokens 65                     --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
            done
            ;;
        # E7: MTP n_max 1 for ordinary decode, ngram-mod for the copy runs, hot store under both.
        # ngram-mod outranks draft-mtp in the impl order and the draft loop stops at the first one
        # that returns tokens, so the stream is bimodal: 65 token batches when the copy run hits,
        # 2 token batches when MTP takes over. The tier only covers both if the limit clears 65.
        # Arms interleaved against this box's ~10% drift.
        e7) for rep in 1 2; do
                HITRATE=0 run_case "combo-cache65-r$rep" -ehs -1 -fitt 256 --expert-tier-max-tokens 65 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 1 --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
                HITRATE=0 run_case "combo-cache4-r$rep"  -ehs -1 -fitt 256 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 1 --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
                HITRATE=0 run_case "combo-nocache-r$rep" -ehs 0 --op-offload-min-batch 128 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 1 --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
                HITRATE=0 run_case "mtponly-cache-r$rep" -ehs -1 -fitt 256 --spec-type draft-mtp --spec-draft-n-max 1
            done
            ;;
        # E8: how far does the tier keep winning? Each width runs with the limit covering it
        # (tier on) and at the default 4 (tier bypassed). ABBA order so each arm gets two fast
        # and two slow slots of the position artifact in section 15.3.
        e8) w1=63; w2=127; w3=255; w4=31
            HITRATE=0 run_case "w$((w1+1))-on"  -ehs -1 -fitt 256 --expert-tier-max-tokens $((w1+1)) $(ngram $w1)
            HITRATE=0 run_case "w$((w1+1))-off" -ehs -1 -fitt 256 $(ngram $w1)
            HITRATE=0 run_case "w$((w2+1))-off" -ehs -1 -fitt 256 $(ngram $w2)
            HITRATE=0 run_case "w$((w2+1))-on"  -ehs -1 -fitt 256 --expert-tier-max-tokens $((w2+1)) $(ngram $w2)
            HITRATE=0 run_case "w$((w3+1))-on"  -ehs -1 -fitt 256 --expert-tier-max-tokens $((w3+1)) $(ngram $w3)
            HITRATE=0 run_case "w$((w3+1))-off" -ehs -1 -fitt 256 $(ngram $w3)
            HITRATE=0 run_case "w$((w4+1))-off" -ehs -1 -fitt 256 $(ngram $w4)
            HITRATE=0 run_case "w$((w4+1))-on"  -ehs -1 -fitt 256 --expert-tier-max-tokens $((w4+1)) $(ngram $w4)
            ;;
        # wiring check: one short run of the proposed config
        smoke) run_case smoke -ehs 0 --spec-type ngram-mod --spec-ngram-mod-n-match "$N_MATCH" --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 ;;
        *) echo "unknown case: $c" ;;
    esac
done
