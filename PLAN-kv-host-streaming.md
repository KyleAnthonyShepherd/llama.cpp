# PLAN - KV cache in host memory: spill on growth, cheaper streaming, CPU attention, CPU+GPU split

For an agent working on branch `Ternary` in `I:\Kyle_Shepherd\opt-llama`. Read `AGENTS.md`
first (commit trailer is `Assisted-by: <assistant name>`, never `Co-authored-by`; no PRs).
Background and all earlier measurements: `PLAN-ternary-bonsai-27b.md` sections 9-11.

Four work items, in this order: **A** (KV spill on growth), **B** (cheaper streaming: B3
zero-copy, B2 copy/compute overlap), **C** (fast CPU decode attention), **D** (concurrent
CPU+GPU split attention). A is independent. D builds on C (and on B2's sync ordering).
Do not use q4_0 KV to hit any target - the user rejected it on quality.

---

## 0. Ground truth you must not rediscover

Machine: Windows 11, RTX 3060 Laptop 6 GB (sm_86), i9-11900H 8C (AVX-512 incl. VNNI/VBMI), 64 GB
DDR4-3200, PCIe measured ~13 GB/s host->device for KV copies.

- **Build** (Git Bash, from repo root): `cmake --build build-cuda --config Release -j 16`.
  `build-cuda` is configured with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON`.
  MSVC native detection does NOT enable VNNI; keep these flags. A full CUDA rebuild is ~1 h;
  touching only ggml-cpu / src is minutes. A `LNK1104 cannot open ...dll` means a stray
  `llama-server.exe` holds it: `taskkill //F //IM llama-server.exe`.
- **Bash commands are truncated at ~7.7 KB** (the harness wraps them). Write long scripts and
  code with the Write tool, never as one huge heredoc. PowerShell script files (`.ps1`) are
  blocked by execution policy - run PowerShell inline.
- **CPU power limit:** after ~15 s of all-core load the CPU drops to ~2.4 GHz and stays there.
  Measure sustained runs (>= 128 generated tokens after the prompt), compare A/B runs
  back-to-back and interleaved, and repeat anything within 5%.
- **VRAM cliff:** ~370 MiB is held by the desktop. Around 5.9 GB used, Windows silently pages
  VRAM to system memory and decode collapses (~20 -> ~11 t/s) while `nvidia-smi` shows *less*
  used. Any change that grows VRAM use must be checked against this; a sudden 2x slowdown with
  flat VRAM is this, not your code.
- **Model files:** `I:/Kyle_Shepherd/models/Bonsai-2-27B-PTQ1_0-cpuNpq2.gguf` = blocks 0..N-1 as
  exact PQ2_0 (CPU), the rest PTQ1_0 (GPU); use `-ngl (65-N)`. Existing N: 2,3,4,5,6,8,9,11,12,16,17.
  Make others with `build-cuda/bin/Release/llama-ternary-repack.exe <PTQ1_0.gguf> <out> "^blk\.[0-(N-1)]\." pq2_0`
  (source: `I:/Kyle_Shepherd/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/*/Ternary-Bonsai-2-27B-PTQ1_0.gguf`).
- **Model shape:** qwen35 hybrid, 64 blocks: 16 full-attention layers (3, 7, ..., 63), 48 GDN.
  Attention: head dim 256, 24 query heads, 4 KV heads (GQA 6). q8_0 KV = ~2.1 KiB per token per
  layer (34 KiB/token for all 16).
- **Existing feature** `--kv-cpu-layers N` (commit `fbfb9fd96`): the first N attention layers'
  K/V are allocated with `ggml_backend_cpu_buffer_type()` in `llama_kv_cache` ctor
  (`src/llama-kv-cache.cpp`, look for `n_kv_host_left`). Attention stays on the GPU; the
  scheduler copies the used part of those caches to the GPU every step (split inputs, see
  `ggml_backend_sched_compute_splits`, `ggml/src/ggml-backend.cpp:~1661`).
- **Existing CPU FA** grouped-query decode path for quantized KV (commit `2f65132d6`):
  `ggml_compute_forward_flash_attn_ext_gqa_chunk` in `ggml/src/ggml-cpu/ops.cpp`, dispatched in
  `ggml_compute_forward_flash_attn_ext_f16` as `use_gqa_path`; scratch sized in
  `ggml_graph_plan` (`ggml/src/ggml-cpu/ggml-cpu.c`, case `GGML_OP_FLASH_ATTN_EXT`).
- **Existing context growth** (fork): `--ctx-max/--ctx-grow-factor/--ctx-grow-headroom`,
  `llama_context::set_n_ctx` (`src/llama-context.cpp:~1079`) -> `llama_kv_cache::resize`
  (`src/llama-kv-cache.cpp:~1345`), VRAM-fit check via `ggml_backend_set_vram_strict` and
  `ggml_cuda_fits_in_vram` (`ggml/src/ggml-cuda/ggml-cuda.cu`).

### Baselines (server, `-np 1 -c 32768 -ctk q8_0 -ctv q8_0 -ub 128 -b 512 -t 6 -fa on`)

| layout | filled | gen ms/token |
|---|---|---|
| cpu6 `-ngl 59`, `--kv-cpu-layers 16` | ~0 | 56.7 |
| same | 8.6k | 97.2 |
| same | 16.5k | 131.1 (prefill 185 t/s) |
| cpu17 `-ngl 48`, KV on GPU | 16.5k | 235 |

Host-KV cost: ~4.5 ms/token per 1k filled tokens for 16 layers (~0.28 ms per 1k per layer),
i.e. ~81 ms of the 131 at 16.5k is KV streaming. CUDA compute buffer at `-c 32768` with host KV:
235 MiB (63 MiB at 4k) - it reserves room for one layer's streamed K/V in the prefill graph.
CPU FA decode (`test-backend-ops perf -b CPU`, 16 threads, boost): q8_0 head 256 4x6, one query:
0.85 ms at 4k KV, 3.6 ms at 16k KV per layer. In-model (6 threads, base clock) CPU attention
for 16 layers at 16.5k was ~150 ms vs ~81 ms streamed.

### Harness

Save as `scratch/srvlong.sh` (any scratch dir) and a prompt file. It starts the server, sends
one long prompt, generates 128 tokens, prints VRAM and timings, kills the server.

```bash
#!/bin/bash
# usage: PROMPT=<file> srvlong.sh <model> <ngl> <kv_cpu_layers> [extra server args]
M=$1; NGL=$2; KC=$3; shift 3
L=${LOG:-/tmp/srvlong.log}
cd /i/Kyle_Shepherd/opt-llama/build-cuda/bin/Release || exit 1
./llama-server.exe -m "$M" -ngl $NGL --kv-cpu-layers $KC -fa on -c 32768 -ctk q8_0 -ctv q8_0 \
    -b 512 -ub 128 -t 6 -np 1 --port 8094 "$@" > $L 2>&1 &
for i in $(seq 1 180); do
    grep -q 'server is listening' $L && break
    tasklist | grep -qi llama-server || { echo "server died"; grep -iE 'error|fail' $L | head; exit 1; }
    sleep 1
done
echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python - <<'EOF'
import json, os, urllib.request
p = open(os.environ['PROMPT'], encoding='utf-8').read()
req = urllib.request.Request('http://127.0.0.1:8094/completion',
    data=json.dumps({'prompt': p, 'n_predict': 128, 'temperature': 0, 'ignore_eos': True}).encode(),
    headers={'Content-Type': 'application/json'})
t = json.load(urllib.request.urlopen(req, timeout=3600))['timings']
print(f"prompt {t['prompt_n']} tok at {t['prompt_per_second']:.1f} t/s, gen {t['predicted_per_second']:.2f} t/s ({t['predicted_per_token_ms']:.1f} ms/token)")
EOF
echo "VRAM after gen: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 3
```

Prompt files: build ~8.6k- and ~16.5k-token prompts (the server reports `prompt_n`). Plain
English prose tokenizes at ~2.9 chars/token for this vocab; README-like text was ~3.0. Check
`prompt_n` and trim until it is within 3% of the target.

Correctness gates for every item: `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT` (3979 cases),
`-o MUL_MAT`, `-o GET_ROWS`, `-o CPY`, `ctest -C Release -E 'test-backend-ops|test-thread-safety'`
from `build-cuda` (46 tests), and a greedy sanity run whose first ~48 tokens match the pre-change
build for "The three laws of thermodynamics are" (`llama-completion ... --temp 0 -n 48`).

---

## A. KV spill on growth

**Goal:** one model file serves any context. Start with the full KV on the GPU and a small cache;
when `resize()` cannot place a grown layer in VRAM, place it in host memory instead of failing.

1. In `llama_kv_cache::resize(n_new)`: today it reallocates each layer with the layer's current
   buffer type (`ggml_backend_buffer_get_type(layer.k->buffer)`) and copies the old cells. Change
   the per-layer choice: try the current (device) buffer type under `ggml_backend_set_vram_strict(true)`;
   if the allocation fails the fit check, allocate that layer with `ggml_backend_cpu_buffer_type()`
   and log it. Spill from the lowest attention layer upward, one layer at a time, until the
   rest fits (each layer at 32k is ~70 MiB).
2. Keep the fit margin: the WDDM cliff is ~5.9 GB total. Add `--kv-spill-margin MiB` (default
   256) and require that much free after the grow.
3. The CUDA compute buffer grows when the first layer goes to host memory (it must stage one
   layer's K/V for prefill; ~170 MiB at 32k). `llama_context::set_n_ctx` re-reserves graphs
   after a resize - make sure the reservation happens *before* deciding the spill count, or
   budget for it explicitly, otherwise the spill is right and the re-reserve then pushes VRAM
   over the cliff.
4. Shrink: when the cache shrinks (new conversation, `set_n_ctx` down, or idle), move host
   layers back to the GPU in reverse order while the fit check allows. Add hysteresis: never
   move back within 60 s of a spill, never move a layer back if that leaves less than 2x the
   margin.
5. Logging: one line per resize: `kv resize %u -> %u cells, %d/%d layers in host memory, VRAM
   free %zu MiB`.

**Accept:** with `cpu5pq2 -ngl 60 --ctx-max 32768` (no `--kv-cpu-layers`), feed prompts of
~2k, ~8.6k and ~16.5k tokens in one server session; each must succeed without reload, the log
must show the spill count rising, and gen ms/token must be within 10% of the matching static
`--kv-cpu-layers` layout at the same fill (use cpu6/59 for the spilled cases if cpu5 cannot fit
the staging buffer - record which). VRAM after load and after gen must stay below 5.9 GB.
A new chat after the 16.5k one must return to GPU KV after the hysteresis window.

---

## B. Cheaper streaming

### B3 - GPU reads host KV directly (zero-copy)

**Goal:** stop staging host K/V in the VRAM compute buffer (~170 MiB at 32k = ~2 weight blocks)
by letting the CUDA attention kernel read pinned host memory over PCIe.

1. Allocate host-KV layers with the CUDA host buffer type (`ggml_backend_cuda_host_buffer_type()`,
   `cudaMallocHost`, `ggml/src/ggml-cuda/ggml-cuda.cu:~1307`) instead of the plain CPU type, so
   the memory is pinned and, under UVA, device-addressable.
2. Teach the scheduler not to copy a split input when the destination backend can read the
   source buffer in place. Cleanest: a device capability query ("can this backend read tensors
   from buffer type X") used in `ggml_backend_sched_split_graph` when it decides split inputs;
   for CUDA return true for its own host buffer type. The FA node then receives the host tensor
   directly.
3. Check the CUDA FA kernels tolerate host pointers: the vec kernel (`fattn-vec`, used for
   ncols 1 decode) reads K/V with coalesced loads. PCIe reads from a kernel are efficient only
   for large coalesced requests; measure. If the vec kernel's access pattern halves PCIe
   throughput, keep zero-copy for VRAM savings only when it is at least as fast as the copy,
   otherwise fall back per step size.
4. Prefill (large batch) must keep working at ~185 t/s: zero-copy prefill reads each KV row once
   per query tile, which may be much slower than one bulk copy per ubatch. Likely outcome: bulk
   copy for prefill, zero-copy for decode - the compute buffer is sized from the prefill graph,
   so VRAM is only saved if prefill also avoids staging. If it cannot, record that B3 saves
   nothing and stop.

**Accept:** at 16.5k, gen ms/token no worse than 131; CUDA compute buffer at `-c 32768` with
host KV <= 80 MiB; if VRAM is freed, a layout with one more GPU block must run below the cliff
and be faster than before.

### B2 - overlap KV copies with GPU compute

**Goal:** hide host->device KV copies behind GPU work. Ceiling: the GPU does ~26 ms of compute
per token, so at most ~26 ms of the ~81 ms of copying can be hidden.

1. Today each split copies its inputs on the backend stream, then computes; copies and compute
   are serial. Give the CUDA backend a dedicated copy stream and prefetch the host-KV inputs of
   the next K splits (the next attention layers) on it, with an event per input; the consuming
   split waits on the event instead of copying.
2. Needs double buffering of the staging tensors (`sched->cur_copy` / `n_copies` already exists
   for pipeline parallelism - reuse its mechanism, it is currently only enabled for multi-GPU).
   Each extra staging buffer costs ~70 MiB of VRAM at 32k: prefetching one layer ahead is the
   likely sweet spot. Budget it against the cliff.
3. The copy source is host memory written by the previous step's KV store (set_rows on the CPU
   side or copied back): order the prefetch after the step's store for that layer. Only the
   *new* cells change between steps - a follow-up is to keep the older cells resident in the
   staging buffer and copy only the delta, but that turns staging into a GPU-resident KV again
   and is only worth it if VRAM allows; do not start there.

**Accept:** at 16.5k, gen ms/token drops by >= 15 ms vs 131; greedy output identical; no VRAM
cliff. Report the per-token copy and compute time from nsys
(`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe profile -t cuda --cuda-graph-trace=node`).

---

## C. CPU decode attention near RAM bandwidth

**Goal:** q8_0, head 256, 24 query / 4 KV heads, one query: ~1 ms per layer at 16k KV (36 MB)
in-model (6 threads, base clock), vs ~3.6 ms (16 threads, boost) today.

The current `gqa_chunk` per KV row: 6 `kq_vec_dot` calls (function pointer, q8_0 x q8_0, 256),
one scalar `dequantize_row_q8_0` of V, 6 `ggml_vec_mad_f32`. Tiling by 64 rows (done) did not
help; the per-row call overhead and conversions dominate.

1. Write a dedicated AVX-512 kernel for this case (x86, `__AVX512F__` + `__AVX512VNNI__`
   guarded; other paths unchanged):
   - Q: convert the G=6 query heads to int8 once per call with per-32 scales (like q8_0), keep
     them in registers/L1.
   - Scores for a tile of rows: for each K row block of 32 (q8_0: fp16 d + 32 int8), compute
     the 6 dot products with VNNI (`dpbusd` needs unsigned x signed: bias the query by +128 and
     subtract 128*sum(k) per block, or use the sign trick) and scale by `d_k * d_q`. Inline, no
     function pointers.
   - Online softmax per tile (as now), then V: dequantize each V row once with AVX-512
     (`cvtepi8_epi32` + `cvtepi32_ps` + mul) and FMA it into the 6 accumulators. 6 heads x 256
     dims is 96 zmm registers, more than the 32 available: keep the accumulators in an L1-resident
     buffer, or process DV in chunks of 64 (4 zmm per head, 24 total) with the tile's weights
     reused across chunks.
   - Parallelize over KV slices per thread as today and reduce with the existing partials merge.
2. Add a `test-backend-ops perf` case at 32k KV and check correctness against the reference
   (`use_ref`) path with `test-backend-ops -o FLASH_ATTN_EXT`; the CPU is the reference for the
   CUDA tests, so a wrong kernel fails thousands of cases. Also do a greedy sanity run with
   `-ngl 0` (everything on the CPU) against the pre-change build.
3. In-model check: with all 16 KV layers in host memory, pin decode attention to the CPU (a
   one-line experiment in `llm_graph_context::build_attn_mha`, `src/llama-graph.cpp`: when the
   K view's `view_src->buffer` is host and `q->ne[1] < 32`, `ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu)`
   on the flash-attention node; this was implemented and reverted in the prior session because
   CPU attention was slower). Keep the pin only if it now beats streaming.

**Accept:** perf case at 16k: <= 1.2 ms per layer (16 threads); in-model at 16.5k with
16 host KV layers and the CPU pin: gen ms/token < 131 (streaming baseline). If the kernel
reaches bandwidth but in-model loses because of the power limit, record it and make the pin a
flag (`--kv-host-attn {gpu,cpu,auto}`), default gpu.

---

## D. Concurrent CPU + GPU split attention

**Goal:** for each host-KV layer, the GPU attends over a streamed range of cells while the CPU
attends over the rest in place, then merge. Ideal at 16.5k with today's speeds:
1/(1/81 + 1/150) ~ 52 ms vs 81; better with C done.

Design:
1. Graph: in `build_attn_mha` for a host-KV layer, build two FA nodes over two cell ranges
   (views of K/V: cells [0, s) and [s, n_kv)), one pinned to the CPU, one left on the GPU, each
   returning *unnormalized* partials (M, S, VKQ) instead of the normalized output. This needs an
   op variant: add an op param to `GGML_OP_FLASH_ATTN_EXT` to return partials, or a small new op;
   the CPU already has the partials format (`ggml_flash_attn_ext_reduce_partials`), CUDA's FA
   has an internal split-K combine that can be exposed. Then a merge op (can run on the GPU; the
   CPU partials are tiny: 24 heads x (2 + 256) floats).
2. Concurrency - the hard part. `ggml_backend_sched_compute_splits` runs splits in order and the
   CPU split needs `q` from the GPU. Order the graph so that the `q` copy to the CPU and the
   CPU FA split are issued *before* the GPU FA split is launched, and make sure the CPU split
   does not synchronize the GPU stream (the copy of `q` must be event-waited, not a full
   `ggml_backend_synchronize`). The CUDA work is async, so launching the GPU FA and then running
   the CPU FA on the calling thread overlaps them. Verify with nsys that the GPU FA kernel and
   the CPU compute overlap in time.
3. Split point: s chosen so both halves finish together: s/n_kv = t_gpu/(t_gpu + t_cpu) per
   filled cell, from measured rates (start with PCIe 13 GB/s vs the C-kernel rate in-model);
   make it adaptive later (EWMA of the two measured times).
4. Watch the power limit: the CPU half runs on cores that are also running the CPU weight
   blocks' matmuls (earlier in the step, not concurrently) - fine; but sustained all-core load
   drops the clock, so measure 600-token runs.

**Accept:** at 16.5k with 16 host KV layers: gen ms/token < 131 by >= 20 ms, greedy output
identical to the non-split path for the first 48 tokens, all FA tests pass, nsys shows overlap.

---

## Reporting

After each item: update `PLAN-ternary-bonsai-27b.md` (new numbered section with a results
table), commit with a descriptive message and the `Assisted-by:` trailer, and push `Ternary`.
Record negative results as carefully as positive ones - several ideas in this area were already
measured and dropped (CPU decode pin, 4-row kernel variants, tiling alone).
