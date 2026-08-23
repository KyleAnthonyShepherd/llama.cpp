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
# E2  does the optimum move inside a run? THIS IS THE ONE THAT DECIDES.
#     The prompt asks for prose and then a verbatim copy, but that is not where the phase change
#     lands. Measured: the model opens a <think> block by restating the request almost word for
#     word - MTP predicts that nearly perfectly - and around token 200 it stops echoing and
#     starts reasoning. 800 tokens never reaches the copy at all. The echo phase is the better
#     experiment anyway, because every thinking model opens that way on every request.
#     Read the --baseline w0 table, not the raw one: a ratio inside one window cancels the drift.
# E3  is the hit rate the thing that moves? correlate E2's per-window winner against E1x's
#     per-window hit rate. Reporter only, no new runs.
# E4  what the always-on cold count costs. Needs two binaries: BIN_A from before the counter
#     landed, BIN_B from after. Run both at a fixed width and compare.
#
# The arms below are step 5, and only mean anything once E2 has said yes:
# verify  adaptive against fixed width 1, temperature 0, completions must be byte identical
# adapt   adaptive against every fixed width, clean timing
#
# Reading `adapt`: the ceiling is not free. The fitter sizes the compute buffer for the widest
# verify batch the config allows, so a ceiling of 3 leaves fewer hot store slots than a fixed
# width of 1 does - measured at S=26 against S=28. An adaptive arm that picks width 1 most of
# the time still runs against a smaller store than the fixed arm it is compared with.

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="${BIN:-$REPO_ROOT/build-cuda/bin/Release/llama-server.exe}"
[ -x "$BIN" ] || BIN="$REPO_ROOT/build-cuda/bin/llama-server"
MODEL="${MODEL:-$(cat "$OUT/model-path.txt")}"
TAG="${TAG:-sw}"

PORT="${PORT:-8080}"
N_CTX="${N_CTX:-8192}"
N_PREDICT="${N_PREDICT:-800}"
SEED="${SEED:-1234}"

# widths to sweep. 3 is the ceiling: a verify batch is 1 + width and the tier serves 4 tokens
WIDTHS="${WIDTHS:-0 1 2 3}"

# repeats per config. The measured spread between two identical runs on this box is ~20%, so
# anything smaller than that needs more than 2 before it is a result
REPS="${REPS:-2}"

# extra request fields, merged over the greedy default. Empty means greedy.
SAMPLING="${SAMPLING:-}"

# the sampling a real workflow runs on this box. Acceptance falls hard as the sampler gets less
# certain - measured, 0.95 at draft position 0 greedy against 0.675 here - so a draft threshold
# has to be judged under the sampling it will actually meet
SAMPLING_REAL='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":1.1}'

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

# the real one: a short question to a thinking model. The answer is short but the think block is
# not, so this is ~1300 tokens of ordinary reasoning - the shape most requests actually have
open(out + "/.p-angora.txt", "w", encoding="utf-8").write(
    "<|im_start|>user\nWhat is an angora rabbit?<|im_end|>\n<|im_start|>assistant\n<think>\n")

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
        --n-predict "$N_PREDICT" --seed "$SEED" --out "$jsonl" --sampling "$SAMPLING"

    stop_server
    python "$PHASE0_DIR/spec-width-report.py" one "$jsonl" "$log"
}

# Width 0 is no speculation at all, and --spec-draft-n-max 0 is NOT that: every draft loop pushes
# a token before it tests n_max (common/speculative.cpp:1668), so 0 still drafts one and the w0 arm
# would silently measure w1. The controller's own width 0 is a true skip - the server never calls
# the drafter when n_draft_max is 0 - so only the fixed-width arm needs this.
mtp() {
    if [ "$1" = "0" ]; then
        echo ""
    else
        echo "--spec-type draft-mtp --spec-draft-n-max $1"
    fi
}

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
            # the second repeat runs the widths backwards. Running them in the same order twice
            # aliases this box's position artifact straight onto width: measured, both w0 arms
            # ran first and came out slowest, both w3 arms ran last and came out fastest
            if [ "$rep" = "1" ]; then
                order="$WIDTHS"
            else
                order="$(echo "$WIDTHS" | tr ' ' '
' | tac | tr '
' ' ')"
            fi
            for w in $order; do
                run_case "e2-w$w-r$rep" "$OUT/.p-two-phase.txt" $(mtp $w)
            done
        done
        echo "=== E2 per-window winner ==="
        python "$PHASE0_DIR/spec-width-report.py" compare --baseline w0 "$OUT/$TAG"-e2-w*.jsonl
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
    # The adaptive arm is in the rotation like any other width, and every other repeat runs the
    # rotation backwards. Running it last every time would hand it this box's position artifact,
    # which is the one direction a result here must not be able to come from.
    adapt) rep=1
           while [ "$rep" -le "$REPS" ]; do
               arms="$WIDTHS auto"
               if [ $((rep % 2)) -eq 0 ]; then
                   arms="$(echo "$arms" | tr ' ' '
' | tac | tr '
' ' ')"
               fi
               for a in $arms; do
                   if [ "$a" = "auto" ]; then
                       run_case "ad-auto-r$rep" "$OUT/.p-two-phase.txt" $(mtp 3) --spec-adaptive-width
                   else
                       run_case "ad-fixed$a-r$rep" "$OUT/.p-two-phase.txt" $(mtp $a)
                   fi
               done
               rep=$((rep+1))
           done
           python "$PHASE0_DIR/spec-width-report.py" compare --baseline fixed0 "$OUT/$TAG"-ad-*.jsonl
           ;;

    # The question the flag has to answer: does computing p_min beat setting it, and beat picking
    # a width by hand? Real prompt, real sampling. The fixed-p_min arms are the honest baseline -
    # beating width 1 is not enough when a tuned constant is also on the table.
    pmin) rep=1
          while [ "$rep" -le "$REPS" ]; do
              arms="w1 w2 w3 p20 p40 auto"
              if [ $((rep % 2)) -eq 0 ]; then
                  arms="$(echo "$arms" | tr ' ' '\n' | tac | tr '\n' ' ')"
              fi
              for a in $arms; do
                  case "$a" in
                      w1)   SAMPLING="$SAMPLING_REAL" run_case "pm-w1-r$rep"   "$OUT/.p-angora.txt" $(mtp 1) ;;
                      w2)   SAMPLING="$SAMPLING_REAL" run_case "pm-w2-r$rep"   "$OUT/.p-angora.txt" $(mtp 2) ;;
                      w3)   SAMPLING="$SAMPLING_REAL" run_case "pm-w3-r$rep"   "$OUT/.p-angora.txt" $(mtp 3) ;;
                      p20)  SAMPLING="$SAMPLING_REAL" run_case "pm-fix20-r$rep" "$OUT/.p-angora.txt" $(mtp 3) --spec-draft-p-min 0.2 ;;
                      p40)  SAMPLING="$SAMPLING_REAL" run_case "pm-fix40-r$rep" "$OUT/.p-angora.txt" $(mtp 3) --spec-draft-p-min 0.4 ;;
                      auto) SAMPLING="$SAMPLING_REAL" run_case "pm-auto-r$rep" "$OUT/.p-angora.txt" $(mtp 3) --spec-adaptive-width ;;
                  esac
              done
              rep=$((rep+1))
          done
          python "$PHASE0_DIR/spec-width-report.py" compare --baseline w1 "$OUT/$TAG"-pm-*.jsonl
          ;;

    # The decisive one. Every other comparison of adaptive against a fixed width also compares two
    # different ceilings, and the ceiling is not free: n_rs_seq is the draft width, the recurrent
    # cache holds 1 + n_rs_seq snapshots (llama-memory-recurrent.cpp:99), and on this hybrid each
    # snapshot is ~63 MiB of VRAM taken from the expert hot store. Ceiling 3 therefore runs ~1.7
    # fewer experts resident than ceiling 1 whatever the controller then decides.
    #
    # At ceiling 1 both arms allocate the same snapshots, so this asks the mechanism alone: does
    # skipping the low-confidence single-token drafts beat always taking them?
    pmin1) rep=1
           while [ "$rep" -le "$REPS" ]; do
               arms="w1 auto1"
               if [ $((rep % 2)) -eq 0 ]; then
                   arms="auto1 w1"
               fi
               for a in $arms; do
                   case "$a" in
                       w1)    SAMPLING="$SAMPLING_REAL" run_case "p1-w1-r$rep"    "$OUT/.p-angora.txt" $(mtp 1) ;;
                       auto1) SAMPLING="$SAMPLING_REAL" run_case "p1-auto-r$rep"  "$OUT/.p-angora.txt" $(mtp 1) --spec-adaptive-width ;;
                   esac
               done
               rep=$((rep+1))
           done
           python "$PHASE0_DIR/spec-width-report.py" compare --baseline w1 "$OUT/$TAG"-p1-*.jsonl
           ;;

    # wiring check. Same prompt on all three so the numbers can be read side by side, though
    # 120 tokens is far too few to mean anything - this checks the plumbing, not the trade.
    smoke) N_PREDICT=120 run_case "smoke-prose" "$OUT/.p-prose.txt" $(mtp 1)
           N_PREDICT=120 INSTRUMENT=1 run_case "smoke-cold" "$OUT/.p-prose.txt" $(mtp 1)
           N_PREDICT=120 INSTRUMENT=1 run_case "smoke-auto" "$OUT/.p-prose.txt" $(mtp 3) --spec-adaptive-width
           grep -c "adaptive draft width" "$OUT/$TAG-smoke-auto.log" | xargs echo "  width changes logged:"
           ;;

    *) echo "unknown case: $c" ;;
    esac
done
