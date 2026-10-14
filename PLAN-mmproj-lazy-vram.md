# Plan: stop CUDA OOM aborts in the mmproj encode under `--mmproj-compute-lazy` + context growth

Branch: `Ternary` (written against commit `75872f9`; line numbers are approximate, function names are exact).

## 1. Symptom

`llama-server` aborts during an image encode:

```
CUDA error: out of memory
  current device: 0, in function alloc at ggml/src/ggml-cuda/ggml-cuda.cu:692
  cuMemCreate(&handle, reserve_size, &prop, 0)
...
ggml_cuda_pool_vmm::alloc
ggml_backend_sched_graph_compute
clip_encode -> clip_image_batch_encode -> mtmd_batch_encode
server_context_impl::pre_decode
```

Launch flags in use (the relevant ones):

```
--mmproj-weights-host --mmproj-compute-lazy --ctx-max 262144 -c 4096 -fitt 256 -ehs -1
--spec-type draft-mtp --flash-attn on -np 1
```

It is **sporadic**. It happened on a 1080x1512 px image (a page-thumbnail sheet), about 1.6 MP, with about 12.5k tokens of context. Images of 850x1100 px (0.94 MP) had encoded fine earlier in the same kind of run. A 900x2184 px image (about 2.0 MP) is the other large size the client sent. Whether an encode survives depends on how much VRAM the grown KV cache and the expert hot store hold at that moment.

**Goal:** no abort in any combination of image size, context size and these flags. When the VRAM can't be found, the encode should fall back to the CPU or fail the one request with a clear error. The server must keep running.

## 2. How the code works today (read this first)

### 2.1 Lazy encoder lifecycle: `tools/mtmd/clip.cpp`
- `clip_ctx::compute_acquire()` (around line 258) re-creates the **GPU backend** (`ggml_backend_dev_init`) and the scheduler (`sched_create()`).
- `clip_ctx::compute_release()` (around line 274) frees the scheduler and **the whole GPU backend**: its CUDA memory pool, its cuBLAS handle and workspace, and its streams.
- `clip_encode()` (around line 4519) wraps each encode in `compute_acquire()` and a `compute_release_guard`. The loader also calls `compute_release()` right after warmup (around line 4036).
- So **every encode starts with a brand-new CUDA backend whose memory pool is empty**. All of its scratch memory is obtained during the encode via `cuMemCreate`.
- `clip_encode()` calls `ggml_backend_sched_alloc_graph(...)` and **ignores the return value** (around line 4549).

### 2.2 Pre-encode VRAM budget: `tools/server/server-context.cpp`
This is in the media-chunk encode path, around line 960, after the comment "the encoder wants VRAM for the length of the encode…":

```
need = mtmd_batch_encode_vram_need(mbatch)        // -> clip_encode_vram_need()
kv_spilled = llama_kv_spill(ctx, need)            // move KV layers to host if free < need
hot_shrunk = ... llama_expert_hotstore_free_bytes(ctx, need)
if (llama_vram_free(ctx) < need) -> refuse the request (clean error)
else mtmd_batch_encode(...)
unspill KV, refit the hot store
```

`clip_encode_vram_need()` (`clip.cpp`, around line 4468) returns `grow + scratch`:
- `grow` is the compute-buffer size from `ggml_backend_sched_reserve_size` on a scratch scheduler, minus what the scheduler already holds. Under lazy mode that is 0, so `grow` is the full buffer.
- `scratch` is the **largest single node**: `ggml_nbytes(node)`, or `4 * nelements * sizeof(float)` for `FLASH_ATTN_EXT`.
- The function itself calls `compute_acquire()` and `compute_release()`. **A CUDA backend is created and destroyed twice per encode**: once to measure and once to encode.

The check is `vram_free >= need` **with no margin**.

### 2.3 Who fills VRAM between encodes: `src/llama-context.cpp`
- **Context growth (`--ctx-max`),** in the `set_n_ctx` growth path (around lines 1100–1230), keeps only `kv_spill_margin()` free (96 MiB by default, `llama-model.cpp`). It shrinks the hot store or spills KV layers to make room for itself.
- **`refit_expert_hotstore()`** (around line 1505) fills free VRAM up to a reserve of `16 MiB + ctx_grow_need`, and runs again after every encode (`llama_expert_hotstore_refit`).
- **`vram_free()`** (around line 1342) is the minimum of `llama_dev_free_vram(dev)` over the context's GPU backends. That goes through the CUDA proc `ggml_backend_dev_free_vram` (`ggml-cuda.cu`, around line 6111).

So by design, VRAM is nearly full between encodes, and each encode relies entirely on `need` being correct. **The pre-encode check passed, then the encode used more than `need`.** A refusal would have been logged as "the encode needs … MiB of VRAM, only … MiB are free"; an abort means the check passed.

### 2.4 Where it aborts: `ggml/src/ggml-cuda/ggml-cuda.cu`
`ggml_cuda_pool_vmm::alloc` (around line 672) grows the pool by `cuMemCreate` and `cuMemMap`:
- Growth is rounded up to `granularity`.
- Every failure goes through `CU_CHECK`, so it **aborts**.
- The pool never shrinks until the backend is destroyed. Its physical size is the high-water mark of memory the ops held at the same time.

## 3. Likely causes (confirm in Phase 0 before fixing)

| # | Hypothesis | Why it fits |
|---|---|---|
| H1 | **The estimate leaves out the cost of creating a backend.** Each encode creates a new CUDA backend: cuBLAS handle and workspace (cuBLASLt can take tens of MiB), streams and events. Kernel modules loaded lazily on first use also take device memory. | None of this is in `grow + scratch`. With a zero margin, any of it tips a tight budget over. |
| H2 | **Scratch is undercounted.** One CUDA op can hold several pool buffers at once: a cuBLAS `mul_mat` converts src0 and/or src1 to F16/BF16 and may allocate an F16 dst temp in the batched path; flash attention converts K and V and keeps partial results. Each pool growth is also rounded up to `granularity`. | The estimate takes the max over single nodes, not the sum each op holds at once. The abort is inside `ggml_cuda_pool_vmm::alloc` during compute, which is exactly pool scratch. |
| H3 | **No safety margin:** `vram_free >= need`. | The driver, fragmentation and timing differences all eat into a zero margin. |
| H4 | **`ggml_backend_sched_alloc_graph` return value ignored** in `clip_encode`. | A failed graph allocation continues into compute instead of failing cleanly. |
| H5 | **The vision tower may be running without flash attention** (unsupported ViT head size, so it falls back). | The KQ tensors are then n_patches² × heads × 4 B, which makes memory very sensitive to image size (1.6 MP vs 0.94 MP). Check that the startup log says `clip ... flash attention is enabled`. If it says disabled, that is a separate fix: FA support for that head size. |

H1 + H2 + H3 together explain "sporadic and image-size dependent" well. Phase 0 decides it.

## 4. Plan

### Phase 0: reproduce on demand, and measure the estimate against reality
1. **Make the pressure reproducible.** Add an env-gated VRAM hog for testing: `LLAMA_DEBUG_VRAM_HOG_MIB=N` allocates and holds N MiB on device 0 at server start. An external process holding VRAM works too. Then send:
   - a long text prompt (around 30–60k tokens) so `--ctx-max` grows the cache;
   - then one image per request at 850x1100, 1080x1512, 900x2184 and 2000x2000 px.

   Lower N until encodes run close to the limit.
2. **Log the budget on every encode** (debug level, not only on refusal), in `server-context.cpp`: `need`, its parts (`grow`, `scratch`, plus the overhead term from Phase 1), free VRAM before and after the KV spill and hot-store shrink, and whether each happened.
3. **Measure the encode's actual peak** in `clip_encode()`:
   - free VRAM before `compute_acquire()`: F0
   - after `ggml_backend_sched_alloc_graph()`: F1
   - just before the release guard runs, while the pool is still mapped at its high-water mark: F2

   Log `estimated X MiB, used F0-F2 MiB (backend+graph F0-F1, pool F1-F2)`. Add a small CUDA proc, `ggml_backend_cuda_pool_stats(backend, size_t * mapped, size_t * used)`, reached through `ggml_backend_reg_get_proc_address` like `ggml_backend_dev_free_vram`, so pool size is read directly rather than inferred.
4. Run the matrix: image sizes × context fill × `--mmproj-weights-host` on/off × hot store on/off × MTP on/off. Record the ratio `used / estimated`. **Deliverable:** a short table that confirms or rules out H1–H5.

### Phase 1: make the lazy encoder's footprint small and predictable
1. **Keep the CUDA backend; free only the memory.** In lazy mode, `compute_release()` should drop the scheduler and its compute buffers and **trim the pool**, but not destroy the backend. Add `ggml_backend_cuda_pool_trim(backend)`: when `pool_used == 0`, unmap and release all VMM mappings and set `pool_size = 0`, keeping the address reservation. For the legacy, non-VMM pool, free its cached buffers.

   This makes the fixed cost (cuBLAS handle and workspace, streams) a one-time cost paid at load. The fit logic and `vram_free()` then see it as ordinary used memory. The per-encode cost becomes graph buffer plus pool, and no backend is created per encode or per estimate. Keep `--mmproj-compute-lazy` as the switch and update its help text to the new meaning: "idle projector holds its weights and a small fixed context, no compute buffers or scratch".
2. **Make `clip_encode_vram_need()` conservative and complete:**
   - `scratch` = the max over ops of **the sum of the pool buffers that op holds at once**, each rounded up to the device's VMM granularity. Cover at least: cuBLAS `mul_mat`, meaning the src0 and src1 conversions plus the batched dst temp (mirror the allocations in `ggml_cuda_mul_mat_batched_cublas` / `ggml_cuda_op_mul_mat_cublas`), and `FLASH_ATTN_EXT`, meaning the K and V conversions plus partial results. If modelling every op is too invasive, use `sum of the two largest per-node scratch values` as a floor.
   - plus `overhead`: measured once at load as the free-VRAM drop across the first `compute_acquire()` on a tiny graph. If Phase 1.1 lands, this is 0 per encode.
   - plus a margin: `max(64 MiB, 10% of the total)`.
   - Stop creating a scratch backend in the estimate. Use the persistent backend from 1.1.
3. **Calibrate as it runs.** After each encode, compare actual (Phase 0.3) with estimated. Keep `vram_need_scale = max(1.0, worst observed ratio) * 1.05` in `clip_ctx`, apply it to later estimates, and log when it changes. This catches whatever the formula misses.

### Phase 2: make a short estimate survivable instead of fatal
1. **Check the graph allocation.** In `clip_encode()`, if `ggml_backend_sched_alloc_graph()` returns false, log it and return false. Never run compute on a failed allocation.
2. **Reserve the pool before any kernel runs.** Add a CUDA proc `bool ggml_backend_cuda_pool_reserve(backend, size_t bytes)` that grows the VMM pool (or the legacy pool) to `bytes`. It should use `cuMemCreate`, `cuMemMap` and `cuMemSetAccess` **with error returns, not `CU_CHECK`**, and undo partial work on failure. Call it in `clip_encode()` after the graph is allocated, with the estimated scratch. If it fails, return false: the request fails cleanly and nothing has run. During compute the pool then serves from memory already mapped, so `cuMemCreate`, and the abort, can only happen if the estimate was short. Phase 1.3 closes that gap over time.
3. **Fall back to the CPU before refusing.** If the budget can't be met even after the KV spill and hot-store shrink, or the pool reserve fails, run that one encode on the CPU backend instead of failing the request. With `--mmproj-weights-host` the weights are already in system RAM, so build the scheduler with `backend_cpu` only for that call. Add `--mmproj-cpu-fallback` (default on when `--mmproj-weights-host` is set) and log each fallback with the image size and the shortfall. Without fallback, keep today's clean refusal.

### Phase 3: optional policy between context growth and encodes
Today VRAM is filled to the brim between encodes, and each encode evicts KV or hot-store slots and refills them afterwards. That is correct, but it costs PCIe traffic on image-heavy workloads where most turns carry page images. Add `--mmproj-vram-reserve N|auto` (MiB; default 0, meaning today's behaviour):
- `auto` = the Phase 1.2 estimate for one image at `--image-max-tokens`, computed at load.
- Context growth adds it to `kv_keep`; `refit_expert_hotstore()` adds it to `vram_reserve`; `--fit` accounts for it at load.
- With the reserve held, the pre-encode spill and shrink normally never fires.

Also measure the eviction order in the pre-encode path, hot store first vs KV spill first, on the Phase 0 matrix. Pick the cheaper one.

### Phase 4: tests and acceptance
- Re-run the Phase 0 matrix, including the VRAM-hog runs at increasing pressure. Then soak test for 2 hours: alternate long text turns that grow the context with turns of 1–4 page images, as the agent client does.
- **Acceptance criteria:**
  1. Zero aborts.
  2. Every encode either succeeds on the GPU, falls back to the CPU (logged), or fails that one request with a clear message.
  3. `estimated >= used` on 100% of encodes, with average overshoot under 25%.
  4. The KV cache and hot store end up back on the GPU after each encode (`unspill_kv` succeeds).
  5. No regression in prompt or generation speed when VRAM is not under pressure.
  6. Behaviour without `--mmproj-compute-lazy` is unchanged.

## 5. Files to touch
- `tools/mtmd/clip.cpp`: `compute_acquire`/`compute_release` (keep backend, trim pool), `clip_encode` (check alloc, pool reserve, CPU fallback, peak measurement), `clip_encode_vram_need` (complete estimate, no scratch backend, calibration).
- `tools/mtmd/clip.h`, `tools/mtmd/mtmd.{h,cpp}`: pass through new params (`cpu_fallback`, reserve), and expose calibration and last-encode stats if the server wants to log them.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_pool_vmm` / legacy pool `trim()` and non-aborting `reserve()`; procs `ggml_backend_cuda_pool_trim`, `ggml_backend_cuda_pool_reserve`, `ggml_backend_cuda_pool_stats` registered alongside `ggml_backend_dev_free_vram`.
- `tools/server/server-context.cpp`: budget logging, margin, fallback handling, Phase 3 reserve hookup.
- `src/llama-context.cpp`: Phase 3 reserve in the growth path and `refit_expert_hotstore()`.
- `common/arg.cpp`, `common/common.h`: `--mmproj-cpu-fallback`, `--mmproj-vram-reserve`, updated `--mmproj-compute-lazy` help text.

## 6. Notes
- **The client side has already been reduced:** `docrender.py` now caps every image at about 1.0 MP (about 1,000 vision tokens), and `--image-max-tokens 1280` is recommended on the server. That makes crashes rarer but doesn't fix the server: any client can still send a large image, and VRAM pressure still varies with context size.
- **Workaround until this lands:** run without `--mmproj-compute-lazy`, or raise `-fitt` (to around 1024), and set `--image-max-tokens`.

## 7. What was built and measured (2026-10-13)

Measured on the laptop: RTX 3060 Laptop 6 GB, Windows, Qwen3.6-35B-A3B Q4_K_S + mmproj-BF16, the flags of section 1. **The abort itself is from the Ubuntu home server and was not reproduced here** (7.1). The failure paths were forced instead (7.4). The server run in 7.6 is still to do.

### 7.1 Why the laptop cannot reproduce it

- The old build ran all four image sizes at 4k, 55k and 115k tokens of context without an abort.
- On Windows a `cudaMalloc` buffer that does not fit goes to system memory (WDDM). That covers the weights, KV cache, hot store, compute buffers and the debug hog. A second process holding VRAM is paged out too. So `LLAMA_DEBUG_VRAM_HOG_MIB` gives no hard pressure here. It should on Linux.
- The VMM scratch pool is the one allocation that refuses: past the process budget `cuMemSetAccess` fails with "device not ready" (on Linux `cuMemCreate` fails with "out of memory", as in the trace). This is why the abort is always in `ggml_cuda_pool_vmm::alloc`.
- NVML free is more than what is usable: an encode that leaves under about 130 MiB free pages and runs 3-4x slower (2.8 s -> 10 s for 1932 tokens).

### 7.2 Estimate against actual (lazy mode, one image per encode)

| input after resize | tokens | compute buffers | scratch used | scratch estimated | estimated / used (buffers + scratch) |
|---|---|---|---|---|---|
| 448x448 | 196 | 47.4 MiB | 7.1 MiB | 87.1 MiB | 2.47 |
| 864x1088 | 941 | 124.9 MiB | 33.2 MiB | 87.1 MiB | 1.34 |
| 1088x1504 | 1626 | 217.4 MiB | 57.7 MiB | 87.1 MiB | 1.11 |
| 896x2176 | 1932 | 259.0 MiB | 68.8 MiB | 87.1 MiB | 1.06 |
| 2016x2016 | 3997 | 540.0 MiB | 143.4 MiB | 176.8 MiB | 1.05 |

- The compute buffer estimate is exact (same graph, same allocator).
- The device drop over an encode (NVML) equals compute buffers + reserved scratch, plus 18.5 MiB once on the first encode (cuBLAS). Nothing else takes VRAM.
- The 87.1 MiB floor is the largest projector weight converted to F32 (81 MiB). The 3060 computes in BF16 and does not do that. A card without BF16 does, so the floor stays.
- Scratch used on the 3060 is the largest matmul input in BF16: `2 * nelements(src1)`. FA takes none (tile kernel, head size 72, 1 parallel block).

### 7.3 Hypotheses

| # | Result |
|---|---|
| H1 | True but small here: 18.5 MiB at the first use of a backend. The old lazy mode paid it on every encode. Now once. An idle backend that has not encoded yet costs nothing measurable (lazy saves 266 MiB at idle, as before). |
| H2 | Not on the 3060: the old estimate was above actual for every size. Possible on a card without BF16: a small image had an old estimate under the 81 MiB weight conversion. Needs the server run. |
| H3 | True by reading: `vram_free >= need` with no margin, and H1 on top. |
| H4 | True by reading. Fixed. |
| H5 | No: the log says `flash attention is enabled`. Head size 72 takes the tile kernel on every card, so FA scratch is `parallel_blocks` partial outputs. 1 block on the 3060. |

### 7.4 What changed

- **CUDA pool** (`ggml-cuda.cu`): `stats`, `reserve` and `trim` on both pools, behind the procs `ggml_backend_cuda_pool_stats|reserve|trim`. The VMM growth is one function that returns false and unmaps on failure. `alloc()` still aborts when it cannot grow, and now logs the pool size and the request first.
- **Lazy lifecycle** (`clip.cpp`): the GPU backend stays. `compute_release()` frees the scheduler and trims the pool. The estimate no longer creates a backend.
- **Estimate**: compute buffer growth + scratch + margin. Scratch is the largest op: matmul `max(2 * (nelements(src1) + nelements(dst)), 4 * nelements(src0))` for a non-F32 weight, FA `fa_blocks * (output + softmax state)`. `fa_blocks` starts at 4 and an encode that used less lowers it. Scale: `max(1, worst used / formula) * 1.05`, logged when it rises. Margin: `max(64 MiB, 10%)`.
- **Encode** (`clip_encode`): gate on free VRAM (without the margin), check `ggml_backend_sched_alloc_graph`, reserve the scratch before the compute. On any refusal: CPU with `--mmproj-cpu-fallback` (default on, needs `--mmproj-weights-host`), else a clean error for that request.
- **Server**: makes room for need + `--kv-spill-margin`, lets the encode decide, and logs one line per encode: VRAM taken and asked for, free before and after making room, what moved, and the three times. The margin on top is for the Windows paging in 7.1.
- **Eviction order**: hot store first, then the KV cache. Measured: a hot store drop and refit is about 0.6 s at any size, a KV cache round trip is 0.3 s at 8k context, 1.5 s at 60k and 2.3 s at 100k.
- **The hot store is asked on every encode**, not only when NVML free is short. On Windows the store fills the driver budget (`cudaMemGetInfo`) while NVML still shows about 500 MiB free. An encode into that pages. The store sizes its drop on the budget, so asking it removes the slow encodes (7.5). The price is about 0.6 s of slot churn on most encodes at a full card. `--mmproj-vram-reserve` avoids both. On Linux the two figures agree and nothing moves without need.
- **Hot store**: `llama_expert_hotstore_free_bytes()` now also counts on `llama_vram_free()`. Before, it sized the drop on the driver budget figure, which can read higher: an encode found 627 MiB free with 1 GB of hot store still resident and fell back to the CPU.
- **Phase 3**: `--mmproj-vram-reserve MiB|auto` (default 0). KV growth and the hot store refit keep it free. `auto` is the need of the largest image the model takes (not the warmup image, which is smaller for Qwen-VL).
- **Debug**: `LLAMA_DEBUG_VRAM_HOG_MIB=N` (server, holds N MiB after the load), `MTMD_DEBUG_VRAM_FREE_MIB=N` (the encode believes N MiB are free).

Not done as planned: the scratch reserve is only made in lazy mode (a pool that is never trimmed would keep it). The per-encode overhead term is 0 because the backend stays.

### 7.5 Runs on the final code

| run | result |
|---|---|
| 4 sizes x 2, 4k-17k context | 8 GPU encodes, no eviction until 17k |
| 4 sizes x 2, 61k-68k context | 8 GPU encodes, KV cache out and back for the 4k-token image (0.9 s + 0.7 s) |
| 4 sizes x 2, 102k-119k context | 8 GPU encodes, KV cache out and back (1.2 s + 1.1 s), hot store drop and refit (0.28 s + 0.33 s) |
| same at 55k-68k with `--mmproj-vram-reserve auto` | 8 GPU encodes, nothing moved, hot store ends at 0 slots instead of 7 |
| without `--mmproj-compute-lazy` | works as before, buffers and pool stay |
| pool procs from Python | reserve past the budget returns false, pool unchanged, trim returns the memory |
| scratch reserve refused (forced), fallback on | CPU encode, same answer as the GPU |
| scratch reserve refused (forced), fallback off | request fails with a message, process stays up |
| `MTMD_DEBUG_VRAM_FREE_MIB=50`, fallback on / off | CPU encode / HTTP 500 for that request, server stays up |
| soak, 40 minutes, KV cache evicted first | 75 turns, 84 encodes, context up to 114k: all on the GPU, 0 errors, KV cache back every time. Estimated / used 1.14 on average, never under 1. KV cache moved for 45 encodes, 0.97 s per round trip on average, 2.7 s at most |
| soak, 20 minutes, hot store evicted first, only when NVML free is short | 38 encodes, context up to 90k: all on the GPU, 0 errors. Hot store round trip about 0.5 s. But 5 encodes ran 2-6x slow (up to 18 s for 1932 tokens) with about 530 MiB free and a full hot store |
| soak, 20 minutes, hot store asked on every encode (final) | 38 encodes, same sequence: all on the GPU, 0 errors, every encode at its normal time (0.9 / 2.1 / 2.8 / 10.5 s). The hot store dropped slots and took them back on 31 of them, about 0.6 s in total each |

CPU encode cost: 196 tokens 7 s, 941 tokens 100 s, 3997 tokens over 16 minutes (stopped). A fallback on a large image looks like a hang to a client. `--no-mmproj-cpu-fallback` gives the error instead.

One case misses criterion 4: at 106k context the KV cache went out for an encode and did not come back. The compute buffers are larger while the KV cache is in host memory, and the fit check for the way back counts them. The hot store took the VRAM instead (0 -> 22 slots). Above about 105k this card holds one or the other, so it is the same end state a growth step reaches.

### 7.6 To do on the server

1. Build this branch. Run the usual command. An encode logs `media encode took ... MiB of VRAM, ... MiB asked for`.
2. Run with `LLAMA_DEBUG_VRAM_HOG_MIB=N`, N raised until `nvidia-smi` shows about 300 MiB free after the load. Send images of 850x1100 up to 2000x2000. Pass: no abort. Each encode is on the GPU, on the CPU (`encoding ... on the CPU`), or fails its request.
3. With `-lv 4`, `clip_encode: ... scratch X MiB of Y MiB estimated` gives the ratio for that card. `later estimates are scaled` means the formula was short there.
