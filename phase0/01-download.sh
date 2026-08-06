#!/usr/bin/env bash
# Download the model (and the MTP head if the repo ships one) and record the
# resolved local path for the other scripts.
#
#   HF_REPO=<user>/<model>:Q4_K_M ./01-download.sh
#
# --mtp asks the downloader to also fetch the multi-token-prediction sidecar if
# the repo has one (common/arg.cpp:3006). On this architecture the MTP layers may
# instead live inside the main checkpoint, in which case nothing extra is fetched
# and that is fine - 02-gguf-layout.py will tell us which case we are in.
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ -z "$HF_REPO" ]; then
    cat >&2 <<'EOF'
error: HF_REPO is not set.

Set it to the repo you want, including the quant, for example:

  HF_REPO=someuser/Qwen3.6-27B-GGUF:Q4_K_M ./01-download.sh

I deliberately did not guess a repo name. Find the real one on Hugging Face and
prefer a Q4_K_M (or whichever 4-bit variant you settled on) - the quant matters
for the repack question in section 1.2 of PLAN-qwen36-27b.md.

If you already have the file locally, skip this script and just do:

  echo /abs/path/to/model.gguf > phase0/results/model-path.txt
EOF
    exit 1
fi

DL="$BIN_DIR/llama"
if [ ! -x "$DL" ]; then
    echo "error: $DL not found. Run 00-build.sh first." >&2
    exit 1
fi

echo "=== downloading $HF_REPO ==="
"$DL" download -hf "$HF_REPO" --mtp

# The downloader puts files in the llama.cpp HF cache. Find the newest .gguf that
# matches, and let the user correct it if the guess is wrong.
CACHE_DIR="${LLAMA_CACHE:-$HOME/.cache/llama.cpp}"
echo
echo "=== gguf files in $CACHE_DIR ==="
find "$CACHE_DIR" -name '*.gguf' -printf '%T@ %s %p\n' 2>/dev/null \
    | sort -rn | awk '{printf "  %10.2f GiB  %s\n", $2/1073741824, $3}'

NEWEST="$(find "$CACHE_DIR" -name '*.gguf' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
if [ -n "$NEWEST" ]; then
    echo "$NEWEST" > "$RESULTS_DIR/model-path.txt"
    echo
    echo "recorded model path: $NEWEST"
    echo "  -> $RESULTS_DIR/model-path.txt"
    echo
    echo "If that picked the wrong file (e.g. it grabbed the MTP sidecar instead of"
    echo "the main checkpoint), overwrite that file by hand before continuing, and"
    echo "set MTP_MODEL in config.sh to the sidecar."
else
    echo "warning: no .gguf found under $CACHE_DIR - set model-path.txt by hand." >&2
fi
