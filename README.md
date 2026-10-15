# llama.cpp

> [!WARNING]
> **This is Kyle Shepherd's personal fork** ([KyleAnthonyShepherd/llama.cpp](https://github.com/KyleAnthonyShepherd/llama.cpp)), branch `Ternary`. It is tuned for one laptop (RTX 3060 6 GB, i9-11900H, 64 GB RAM) and a home server, running Qwen3.6 35B-A3B / 27B and Ternary Bonsai 27B. It is not meant for upstream as a whole. Most features below are off unless their flag is given.

## How this branch differs from upstream

Layers, oldest first:

1. **Upstream** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` as of 2026-09-17 (`972d2313b`), merged in through this fork's `master`.
2. **PrismML** [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch `prism` (prism-v7): the low-bit formats Q1_0, PQ2_0 and PTQ1_0 with CPU, CUDA, HIP and Metal kernels, the Hadamard weight fold, K-cache mean centering (`--kv-mean-center`), dspark drafter fixes, and their release CI. Their own notes are in the box below.
3. **Expert cache** from Miltos22's upstream PR [#26563](https://github.com/ggml-org/llama.cpp/pull/26563): keeps the hottest MoE experts in VRAM (`-ehs N` / `--expert-hot-s`, `-ehs -1` fits it automatically, `--ecf`, `--expert-dwell`, `--expert-hyst`, `--expert-heat-decay`, `--expert-heat-defer`, `--expert-heat-log-period`).
4. **This branch's own work**, listed next.

### Context and KV cache

| What | Flags | Notes |
|---|---|---|
| KV cache that grows with the conversation, and shrinks back between requests | `--ctx-max N`, `--ctx-grow-factor`, `--ctx-grow-headroom` | start small with `-c 512`, grow up to `--ctx-max` |
| a per-request context budget, so an agent keeps room to wrap up | `--ctx-limit N`, request field `n_ctx_limit` | stop type `ctx_limit` |
| KV of the first N layers in host memory | `--kv-cpu-layers N` | `PLAN-kv-host-streaming.md` |
| KV layers spill to host memory when growth does not fit in VRAM, and come back when idle | `--kv-spill-margin` | placement uses real free VRAM, not the WDDM budget |
| CPU flash attention decode for quantized KV, grouped by query head, AVX-512 VNNI for q8_0 | | |
| RAM slot store: named slot states in RAM, next to the disk ones | `--slot-ram-limit N`, `/slots` with `"store": "ram"` | used by the web UI's hats |
| prompt cache matches by text where the tokens differ | | a model can sample an unusual token split of a text (`.` + `\n\n` for `.\n\n`); the next request tokenizes it the usual way and missed the cache from there, a full re-prefill on hybrid models. The server now takes the cached tokens where the text is the same. Upstream issue [#11970](https://github.com/ggml-org/llama.cpp/issues/11970), unmerged PR [#12067](https://github.com/ggml-org/llama.cpp/pull/12067) |

### Speculative decoding and MoE

| What | Flags | Notes |
|---|---|---|
| draft width that follows the cost of cold experts | `--spec-adaptive-width` | `PLAN-adaptive-draft-width.md` |
| expert cache under speculative decoding: tiered path for small batches, draft-aware heatmap, shrinks when the KV cache grows | `--expert-tier-max-tokens` | `PLAN-qwen36-35b-moe.md`, `PLAN-ngram-mod-expert-cache.md` |
| op offload threshold that small host weights do not drive | `--op-offload-min-batch` | |
| MTP draft state kept in step across image and audio chunks | | |

### Images (mmproj)

| What | Flags | Notes |
|---|---|---|
| encoder weights in host memory, compute buffers only while encoding | `--mmproj-weights-host`, `--mmproj-compute-lazy` | `PLAN-mmproj-lazy-vram.md` |
| an image encode borrows the KV cache's VRAM and gives it back; fails the request, not the server, when it does not fit | `--mmproj-vram-reserve`, `--mmproj-cpu-fallback` | `PLAN-adaptive-mmproj.md` |

### Sampling and loops

| What | Flags | Notes |
|---|---|---|
| repetition penalties and DRY that skip the prompt, and pause inside tool calls | `--rep-skip-prompt`, `--rep-skip-tool-calls` | |
| stop when the output is one sequence repeated | `--repetition-stop N` | stop type `repetition` |
| **loop guard**: watches thinking and answer for a reply that goes in circles (word novelty: 6% or fewer of the last 300 words are new, after 600 words; exact lines: a line of 12+ tokens written 5 times), then closes it, adds a system note and opens a new reply. `reflect` mode lets the model judge the loop first, on a copy of its context parked in RAM | `--loop-guard off\|log\|note\|reflect`, `--loop-guard-trace DIR`, request fields `loop_guard*` | needs a chat template that renders a late system message: `models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja` for unsloth's Qwen3.6-35B-A3B-MTP GGUF, `Qwen3.6-35B-A3B-heretic-loop-guard.jinja` for llmfan46's heretic one. Clients get `loop_guard` events in the stream. `PLAN-loop-guard.md` |

### Web UI (`tools/ui`)

- thinking and response prefill, per turn and as a default
- office files (docx, pptx, ...) attached through a server tool, missing-font warning, download of the saved file
- hats: a chat can wear a saved prompt cache (system prompt, tools and documents) restored from the RAM slot store
- loop guard: holds the split reply and the system note as separate messages, shows the reflection

### Ternary Bonsai

- `examples/ternary-repack`: lossless PTQ1_0 to Q4_0 or PQ2_0 for the blocks that run on the CPU
- faster PTQ1_0 mat-vec on NVIDIA, AVX-512 PQ2_0 dot, multi-threaded CONCAT and GET_ROWS
- `PLAN-ternary-bonsai-27b.md`

### Other

- Windows shared builds export all llama symbols
- the `PLAN-*.md` files at the root record the design, measurements and decisions behind each change

Builds here use `build-cuda` (MSVC, CUDA, `-DGGML_NATIVE=OFF` with AVX-512 VNNI on). The web UI is built from source only when Node 22 is on `PATH`; otherwise CMake embeds upstream's prebuilt UI and the fork's UI changes are missing.

---

> [!IMPORTANT]
> **This is the PrismML fork of llama.cpp**, the main line behind the [Bonsai](https://huggingface.co/collections/prism-ml/bonsai) models (branch `prism`, developed as `prism-v7`). It tracks current mainline llama.cpp and adds the fork's low-bit formats and runtime features on top.
>
> **New here? Start with the [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) repo.** It downloads the right models and the correct prebuilt binaries for your hardware/backend automatically.
>
> **Which ternary model file to use:**
>
> - `*-PQ2_0.gguf` (fork group-128, ggml id 142): preferred on Metal, CUDA, HIP and CPU. About 6% smaller than group-64.
> - `*-Q2_0_g64.gguf` / 27B `*-Q2_g64.gguf` (official group-64, ggml id 42): runs on every backend here AND on mainline llama.cpp. If unsure, use this. Newer model releases name this file plain `*-Q2_0.gguf`.
> - `*-Q2_0.gguf` on OLDER model repos is the **deprecated legacy format** (group 128 stored as id 42). It does not load on these builds; the error tells you which file to get instead. If you must run it, use the frozen [`prism-v5`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-v5) line and its final release [`prism-b9601`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9601-68faa14).
>
> **Speculative decoding (dspark)** is supported via mainline's draft-dspark plus fork patches. Drafters published for older model releases need a one-time conversion with `gguf-dspark-to-dflash` (see [SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) in Bonsai-demo); newer releases ship ready-to-use drafters.
>
> Do NOT build from `prism-v6` (stale mid-migration snapshot) and do NOT mix this fork's `ggml-*` libraries with a stock llama.cpp build.

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
