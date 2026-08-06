#!/usr/bin/env bash
# Build llama.cpp with CUDA on the server. Skip this if you already have a build -
# the only thing it does beyond the standard steps is auto-detect the compute
# capability so nvcc does not fall back to a generic arch.
set -euo pipefail
source "$(dirname "$0")/config.sh"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc not on PATH. Install the CUDA toolkit first:" >&2
    echo "  sudo apt install nvidia-cuda-toolkit    # or NVIDIA's own .deb, see docs/build.md" >&2
    exit 1
fi

# Compute capability, e.g. "8.6" -> "86". Falls back to native if unavailable.
CUDA_ARCH="${CUDA_ARCH:-}"
if [ -z "$CUDA_ARCH" ]; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    if [ -n "$CC" ]; then
        CUDA_ARCH="$CC"
    fi
fi

echo "=== configuring (CUDA arch: ${CUDA_ARCH:-native}) ==="
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DGGML_CUDA=ON
    -DCMAKE_BUILD_TYPE=Release
)
if [ -n "$CUDA_ARCH" ]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

cmake "${CMAKE_ARGS[@]}"

echo "=== building ==="
cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"

echo
echo "=== binaries ==="
for b in llama-cli llama-completion llama-bench llama-fit-params llama; do
    if [ -x "$BIN_DIR/$b" ]; then
        echo "  ok   $BIN_DIR/$b"
    else
        echo "  MISSING  $BIN_DIR/$b"
    fi
done

echo
echo "Sanity check the GPU is visible to the build:"
echo "  $BIN_DIR/llama-bench --list-devices"
