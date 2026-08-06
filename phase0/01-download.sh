#!/usr/bin/env bash
# Download the model (and the MTP head if the repo ships one) and record the
# resolved local path for the other scripts.
#
#   HF_REPO=<user>/<model>:Q4_K_M ./01-download.sh
#   HF_REPO=unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
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

# The downloader prints resolved paths to STDOUT, one per line, in a fixed order
# (app/download.cpp:62-69):
#   line 1 : the model
#   then   : mmproj, if the repo has one
#   then   : the speculative/MTP sidecar, if one was fetched
# Progress goes to stderr, so let that through to the terminal and capture only
# stdout. Do NOT try to guess the file by scanning a cache directory - the
# download may land in the HF hub cache rather than $LLAMA_CACHE.
STDOUT_LOG="$RESULTS_DIR/download-stdout.txt"
"$DL" download -hf "$HF_REPO" --mtp > "$STDOUT_LOG"

echo
echo "=== downloader reported ==="
cat "$STDOUT_LOG"
echo

MODEL_PATH="$(sed -n '1p' "$STDOUT_LOG")"
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "error: could not read a model path from the downloader output." >&2
    echo "       set it by hand: echo /abs/path/model.gguf > $RESULTS_DIR/model-path.txt" >&2
    exit 1
fi

echo "$MODEL_PATH" > "$RESULTS_DIR/model-path.txt"
echo "model      : $MODEL_PATH"
echo "size       : $(du -Lh "$MODEL_PATH" | cut -f1)"   # -L: HF cache paths are symlinks into blobs/
echo "  recorded -> $RESULTS_DIR/model-path.txt"

# Any extra line that is not the model and not an mmproj is the MTP/draft sidecar.
SIDECAR="$(tail -n +2 "$STDOUT_LOG" | grep -v -i 'mmproj' | head -1 || true)"
echo
if [ -n "$SIDECAR" ] && [ -f "$SIDECAR" ]; then
    echo "$SIDECAR" > "$RESULTS_DIR/mtp-path.txt"
    echo "MTP sidecar: $SIDECAR"
    echo "  export MTP_MODEL=\"$SIDECAR\" before running 04-residency.sh"
else
    echo "no separate MTP sidecar was fetched."
    echo "For a repo named '*-MTP-GGUF' that almost certainly means the MTP layers are"
    echo "inside the main checkpoint. 02-gguf-layout.py confirms it: look for a layer"
    echo "classified MTP (it detects blk.N.nextn.* tensors)."
fi
