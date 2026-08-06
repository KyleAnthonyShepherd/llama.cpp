# Shared config for the Phase 0 measurement scripts. Source this, do not run it.
#
# Override anything by exporting it before calling a script, e.g.:
#   HF_REPO=someuser/Qwen3.6-27B-GGUF:Q4_K_M ./01-download.sh

PHASE0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PHASE0_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
BIN_DIR="${BIN_DIR:-$BUILD_DIR/bin}"
RESULTS_DIR="${RESULTS_DIR:-$PHASE0_DIR/results}"

# Hugging Face repo for the model, "<user>/<model>[:quant]".
# No default on purpose - set it to the repo you actually downloaded.
HF_REPO="${HF_REPO:-}"

# Absolute path to the .gguf. 01-download.sh writes the resolved path to
# results/model-path.txt; the other scripts read it from there if MODEL is unset.
MODEL="${MODEL:-}"

# Path to the MTP head gguf, if it ships as a separate file (may stay empty -
# on this arch the MTP layers can also live inside the main checkpoint).
MTP_MODEL="${MTP_MODEL:-}"

# Context size used for every measurement. Keep it fixed across runs or the
# numbers are not comparable.
N_CTX="${N_CTX:-8192}"

# Prompt/generation sizes for the residency and bench runs. Small on purpose -
# with ~13 GB of weights on the CPU side these runs are slow.
N_PROMPT="${N_PROMPT:-256}"
N_GEN="${N_GEN:-64}"

# Set DROP_CACHES=1 to run "sync; echo 3 > /proc/sys/vm/drop_caches" between
# runs. Needs passwordless sudo. Without it, run-to-run page cache state leaks
# and the residency comparison is much less meaningful.
DROP_CACHES="${DROP_CACHES:-0}"

mkdir -p "$RESULTS_DIR"

resolve_model() {
    if [ -n "$MODEL" ]; then
        return 0
    fi
    if [ -f "$RESULTS_DIR/model-path.txt" ]; then
        MODEL="$(cat "$RESULTS_DIR/model-path.txt")"
    fi
    if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
        echo "error: no model found. Set MODEL=/path/to/model.gguf or run 01-download.sh first." >&2
        return 1
    fi
    case "$(basename "$MODEL")" in
        *mmproj*|*MMPROJ*)
            echo "error: '$MODEL' looks like a multimodal projector, not the model." >&2
            echo "       Fix $RESULTS_DIR/model-path.txt to point at the main checkpoint." >&2
            return 1
            ;;
    esac
    # A 27B 4-bit checkpoint is ~15-20 GiB. Anything much smaller is the wrong file.
    local sz_gib
    sz_gib=$(( $(stat -c %s "$MODEL") / 1073741824 ))
    if [ "$sz_gib" -lt 8 ]; then
        echo "warning: '$MODEL' is only ${sz_gib} GiB - is that really the 27B checkpoint?" >&2
    fi
    if [ -z "$MTP_MODEL" ] && [ -f "$RESULTS_DIR/mtp-path.txt" ]; then
        MTP_MODEL="$(cat "$RESULTS_DIR/mtp-path.txt")"
    fi
}

drop_caches() {
    if [ "$DROP_CACHES" = "1" ]; then
        sync
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        echo "  (page cache dropped)"
    fi
}

# Record machine state alongside every result set so the numbers stay readable
# months later.
record_env() {
    local out="$1"
    {
        echo "date: $(date -Is)"
        echo "git: $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "kernel: $(uname -r)"
        echo "cpu: $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ *//')"
        echo "nproc: $(nproc)"
        echo "--- free -m ---"
        free -m
        echo "--- swap ---"
        swapon --show 2>/dev/null || echo "(no swap)"
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
    } > "$out"
}
