#!/usr/bin/env bash
# Step 0 of PLAN-adaptive-draft-width.md: is there anything for a draft-width controller to win?
#
# Every arm runs the same prompt at temperature 0 through one server and streams the result, so
# the reporter has an arrival time per token and can cut the run into 100 token windows.
#
# E1  the fixed-width curve. widths 0..3 (a verify batch is 1 + width, and 4 is the tier's
#     correctness bound - see LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT). gives T(m) and the
#     acceptance per position.
# E1x the same widths with LLAMA_EXPERT_HITRATE + LLAMA_TRACE, for the hit rate, the union, and
#     the cold expert count per batch. The hit-rate readback costs throughput, so its t/s is
#     not a result - that is what E1 is for. At temperature 0 both runs emit the same tokens,
#     so the two line up by token index.
# E2  does the optimum move inside a run? the two-phase prompt generates prose and then copies
#     a document verbatim. If a different width wins the prose windows and the copy windows,
#     a controller has something to win and the gap is its size. THIS IS THE ONE THAT DECIDES.
# E3  is the hit rate the thing that moves? correlate E2's per-window winner against E1x's
#     per-window hit rate. Reporter only, no new runs.
# E4  what the always-on cold count costs. Needs two binaries: BIN_A from before the counter
#     landed, BIN_B from after. Run both at a fixed width and compare.
#
# The arms below are step 5, and only mean anything once E2 has said yes:
# verify  adaptive against fixed width 1, temperature 0, completions must be byte identical
# adapt   adaptive against every fixed width, clean timing

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="${BIN:-$REPO_ROOT/build/bin/Release/llama-server.exe}"
[ -x "$BIN" ] || BIN="$REPO_ROOT/build/bin/llama-server"
MODEL="${MODEL:-$(cat "$OUT/model-path.txt")}"
TAG="${TAG:-sw}"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-8192}"
N_PREDICT="${N_PREDICT:-800}"
SEED="${SEED:-1234}"

# widths to sweep. 3 is the ceiling: a verify batch is 1 + width and the tier serves 4 tokens
WIDTHS="${WIDTHS:-0 1 2 3}"

# every arm holds these fixed. -fitt leaves room for the KV cache the way the other phase0 runs do
COMMON="-ehs -1 -fitt 256 --flash-attn on"

mkdir -p "$OUT"

# ---------------------------------------------------------------- prompts

build_prompts() {
    python - "$PHASE0_DIR/prompts/copy-heavy.txt" "$OUT" <<'PY'
import sys
doc = open(sys.argv[1], encoding="utf-8").read().rstrip("\n")
out = sys.argv[2]

def chat(body):
    return "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n"

# prose: novel tokens, low draft acceptance, routing spread wide
open(out + "/.p-prose.txt", "w", encoding="utf-8").write(chat(
    "Explain, in flowing prose with no code and no bullet lists, how a work-stealing scheduler "
    "decides which queue to take work from, why the victim is chosen at random, and what goes "
    "wrong when every worker steals from the same place. Write at least 700 words."))

# copy: verbatim continuation, high acceptance, routing repeats
open(out + "/.p-copy.txt", "w", encoding="utf-8").write(chat(
    "Copy the document below verbatim, over and over, numbering each copy. No commentary.\n\n"
    "=== DOCUMENT ===\n" + doc + "\n\n=== COPY 1 ===\n" + doc + "\n\n=== COPY 2 ===\n"))

# two-phase: both of the above in one generation, prose first. the phase boundary is
# wherever the model starts the document, and the window table shows it as a step
open(out + "/.p-two-phase.txt", "w", encoding="utf-8").write(chat(
    "Answer in two parts, in this order, with nothing else.\n\n"
    "Part 1. Explain in flowing prose, no code and no bullet lists, how a work-stealing "
    "scheduler decides which queue to take work from. Write about 300 words.\n\n"
    "Part 2. Then reproduce the document below verbatim, twice, with no commentary.\n\n"
    "=== DOCUMENT ===\n" + doc))
PY
}

# ---------------------------------------------------------------- server

stop_server() {
    curl -s -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    taskkill //F //IM llama-server.exe >/dev/null 2>&1   # windows
    pkill -f "llama-server .*--port $PORT" >/dev/null 2>&1
    sleep 3
}

# run_case <name> <prompt-file> [server args...]
# INSTRUMENT=1 turns on the hit-rate readback and the per-step acceptance trace. It costs
# throughput, so never read t/s off an instrumented arm.
run_case() {
    local name="$1"; shift
    local prompt="$1"; shift
    local log="$OUT/$TAG-$name.log"
    local jsonl="$OUT/$TAG-$name.jsonl"
    local bin="${BIN_OVERRIDE:-$BIN}"

    stop_server

    if [ "${INSTRUMENT:-0}" = "1" ]; then
        export LLAMA_EXPERT_HITRATE=1
        export LLAMA_TRACE=1
    else
        unset LLAMA_EXPERT_HITRATE
        unset LLAMA_TRACE
    fi

    "$bin" -m "$MODEL" -c "$N_CTX" -np 1 -lv 4 $COMMON \
        --host 127.0.0.1 --port "$PORT" "$@" > "$log" 2>&1 &

    local waited=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited+5))
        if [ $waited -gt 900 ]; then
            echo "=== $name: server never became ready ==="; tail -5 "$log"; stop_server; return
        fi
    done

    python "$PHASE0_DIR/spec-width-client.py" --port "$PORT" --prompt "$prompt" \
        --n-predict "$N_PREDICT" --seed "$SEED" --out "$jsonl"

    stop_server
    python "$PHASE0_DIR/spec-width-report.py" one "$jsonl" "$log"
}

mtp() { echo "--spec-type draft-mtp --spec-draft-n-max $1"; }

build_prompts

for c in "${@:-e2}"; do
    case "$c" in

    # E1: the fixed-width curve on each workload separately. Widths interleaved across two
    # repeats so this box's ~10% thermal drift lands on every arm, not on the last one.
    e1) for rep in 1 2; do
            for w in $WIDTHS; do
                run_case "e1-prose-w$w-r$rep" "$OUT/.p-prose.txt" $(mtp $w)
            done
            for w in $(echo "$WIDTHS" | tr ' ' '\n' | tac | tr '\n' ' '); do
                run_case "e1-copy-w$w-r$rep" "$OUT/.p-copy.txt" $(mtp $w)
            done
        done
        echo "=== E1 prose ==="; python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-e1-prose-w*.jsonl
        echo "=== E1 copy  ==="; python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-e1-copy-w*.jsonl
        ;;

    # E1x: the same widths, instrumented. Answers "is cold linear in the batch", which is the
    # assumption the controller's fit rests on.
    e1x) for w in $WIDTHS; do
            INSTRUMENT=1 run_case "e1x-prose-w$w" "$OUT/.p-prose.txt" $(mtp $w)
            INSTRUMENT=1 run_case "e1x-copy-w$w"  "$OUT/.p-copy.txt"  $(mtp $w)
         done
         echo "=== E1x cold against width ==="
         python "$PHASE0_DIR/spec-width-report.py" cold "$OUT/$TAG"-e1x-*.log
         ;;

    # E2: the decision. One generation that changes phase halfway. If the window table has one
    # winner throughout, the plan says stop.
    e2) for rep in 1 2; do
            for w in $WIDTHS; do
                run_case "e2-w$w-r$rep" "$OUT/.p-two-phase.txt" $(mtp $w)
            done
        done
        echo "=== E2 per-window winner ==="
        python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-e2-w*.jsonl
        ;;

    # E2x: the same prompt instrumented once, so E3 has a per-window hit rate to correlate the
    # E2 winners against. Width 1 because that is the config in use today.
    e2x) INSTRUMENT=1 run_case "e2x-w1" "$OUT/.p-two-phase.txt" $(mtp 1) ;;

    # E3: reporter only. Read the two tables side by side - does the window where the winner
    # changes line up with the window where the hit rate changes?
    e3) echo "=== E2 windows ==="
        python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-e2-w*.jsonl
        echo
        echo "=== E2x per-window hit rate and cold count ==="
        python "$PHASE0_DIR/spec-width-report.py" one "$OUT/$TAG-e2x-w1.jsonl" "$OUT/$TAG-e2x-w1.log"
        ;;

    # E4: what the always-on cold count costs. Build the commit before it landed into another
    # tree and point BIN_A at it:
    #   BIN_A=/path/to/old/llama-server BIN_B=$BIN ./spec-width-sweep.sh e4
    e4) if [ -z "${BIN_A:-}" ]; then
            echo "e4 needs BIN_A=<binary from before the cold counter> (and optionally BIN_B)"
            continue
        fi
        for rep in 1 2; do
            BIN_OVERRIDE="$BIN_A"        run_case "e4-before-r$rep" "$OUT/.p-prose.txt" $(mtp 1)
            BIN_OVERRIDE="${BIN_B:-$BIN}" run_case "e4-after-r$rep"  "$OUT/.p-prose.txt" $(mtp 1)
        done
        python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-e4-*.jsonl
        ;;

    # step 5. Speculation is exact, so a changing width cannot change the output. If these
    # differ, the width plumbing is wrong, not the model.
    verify) run_case "v-fixed1"  "$OUT/.p-two-phase.txt" $(mtp 1)
            run_case "v-adaptive" "$OUT/.p-two-phase.txt" $(mtp 3) --spec-adaptive-width
            python - "$OUT/$TAG-v-fixed1.jsonl" "$OUT/$TAG-v-adaptive.jsonl" <<'PY'
import json, sys
def text(p):
    for line in open(p, encoding="utf-8"):
        ev = json.loads(line)
        if "tok" not in ev:
            return ev.get("content", "")
    return ""
a, b = text(sys.argv[1]), text(sys.argv[2])
print("=== completions identical ===" if a == b else "=== COMPLETIONS DIFFER ===")
if a != b:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print("  first difference at char %d: %r vs %r" % (i, a[i:i+40], b[i:i+40]))
            break
    print("  lengths %d vs %d" % (len(a), len(b)))
PY
            ;;

    # step 5. Only worth running once E2 says the optimum moves.
    adapt) for rep in 1 2; do
               for w in $WIDTHS; do
                   run_case "ad-fixed$w-r$rep" "$OUT/.p-two-phase.txt" $(mtp $w)
               done
               run_case "ad-auto-r$rep" "$OUT/.p-two-phase.txt" $(mtp 3) --spec-adaptive-width
           done
           python "$PHASE0_DIR/spec-width-report.py" compare "$OUT/$TAG"-ad-*.jsonl
           ;;

    # wiring check: one short run of each prompt at the width in use today
    smoke) N_PREDICT=120 run_case "smoke-prose" "$OUT/.p-prose.txt" $(mtp 1)
           N_PREDICT=120 INSTRUMENT=1 run_case "smoke-cold" "$OUT/.p-prose.txt" $(mtp 1)
           N_PREDICT=120 run_case "smoke-auto"  "$OUT/.p-two-phase.txt" $(mtp 3) --spec-adaptive-width
           ;;

    *) echo "unknown case: $c" ;;
    esac
done
