#!/usr/bin/env bash
# Expert-cache A/B on the dev box. llama-bench cannot drive -ehs (it parses its
# own args), so every case goes through llama-completion.
#
# Usage: ./devbox-expert-ab.sh [case ...]   (default: all)

set -u

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"
OUT="$PHASE0_DIR/results/devbox"
BIN="$REPO_ROOT/build/bin/Release/llama-completion.exe"
MODEL="$(cat "$OUT/model-path.txt")"

N_CTX="${N_CTX:-8192}"
N_GEN="${N_GEN:-128}"

# ~400 token prompt so prompt-eval is measurable, not noise.
PROMPT="Summarize the design tradeoffs in the following text, then list three risks.
A mixture-of-experts transformer routes each token to a small subset of feed-forward
experts. Only the selected experts are read from memory, so the bytes touched per token
are far smaller than the total parameter count. This makes such models attractive when
memory bandwidth, rather than compute, is the binding constraint. However, the routing
decision is made inside the layer, immediately before the expert weights are needed, so
there is no opportunity to fetch them in advance without speculating. Systems that place
experts in a fast tier must therefore predict which experts will be used, rather than
fetching them on demand. A further complication is that the router is trained with a load
balancing objective, which deliberately flattens the usage distribution across experts.
This limits how much a popularity-ranked cache can help, because a flat distribution means
that holding the hottest fraction of experts captures approximately that same fraction of
activations. The remaining lever is raw bandwidth: an expert held in device memory is read
an order of magnitude faster than one held in system memory, so the gain is proportional to
the fraction of experts that fit, regardless of how they are chosen. Prompt processing
behaves differently from token generation because many tokens are routed at once, so the
union of selected experts approaches the full set and the sparsity advantage disappears."

run_case() {
    local name="$1"; shift
    echo "=== $name : $* ==="
    local log="$OUT/ab-$name.log"
    "$BIN" -m "$MODEL" -c "$N_CTX" -n "$N_GEN" --temp 0 -lv 4 \
        "$@" -p "$PROMPT" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED rc=$rc (see $log)"
        return
    fi
    grep -E "prompt eval time|eval time =|hot store allocated|hot store: |expert hot hit rate|hot hit rate|Expert hotstore sizing|expert cache is OFF|hot store DISABLED|draft acceptance|n_draft" "$log" \
        | sed 's/^[0-9.]* I //' | sed 's/^/  /'
}

# -fitt trims the default 1024 MiB fit margin; on this box that is worth ~14 more
# hot slots per layer. LLAMA_EXPERT_HITRATE is undocumented and is the only way to
# get the hit-rate counter (src/llama-context.cpp:1523).
FITT="${FITT:-256}"
export LLAMA_EXPERT_HITRATE=1

case_off()       { run_case off       -ehs 0; }
case_cache()     { run_case cache     -ehs -1; }
case_cachefit()  { run_case cachefit  -ehs -1 -fitt "$FITT"; }
case_mtp()       { run_case mtp       -ehs 0  --spec-type draft-mtp; }
case_cachemtp()  { run_case cachemtp  -ehs -1 -fitt "$FITT" --spec-type draft-mtp; }

mkdir -p "$OUT"
CASES="${*:-off cache mtp cachemtp}"
for c in $CASES; do
    case "$c" in
        off) case_off ;; cache) case_cache ;; cachefit) case_cachefit ;; mtp) case_mtp ;; cachemtp) case_cachemtp ;;
        *) echo "unknown case: $c" ;;
    esac
done
