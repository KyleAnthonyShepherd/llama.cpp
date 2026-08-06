# PLAN - Qwen3.6-27B on the 6 GB VRAM / 16 GB RAM server (Ubuntu + CUDA)

Target: run a ~18 GB 4-bit GGUF of Qwen3.6-27B, optionally with an MTP head, on a box with
6 GB VRAM and 16 GB RAM, without thrashing.

Discovery pass only - no code changed. Every claim below is cited to a file:line in this
tree. Companion to `NOTES.md` (Parts 1/2/3 discovery); this plan reuses those findings
where they apply and marks where the 27B dense case diverges.

---

## Measured, 2026-08-06 - `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`

Real numbers from `phase0/02-gguf-layout.py`, replacing the structural placeholders that
sections 2 and 6 were written against. Raw output in `phase0/results/layout.json`.

| | |
|---|---|
| file size | **16.68 GiB** (not the ~18 GiB assumed - a full 1.3 GiB of slack) |
| blocks | 65 = 64 trunk + 1 MTP, `nextn_predict_layers = 1` |
| layer classes | **48 GDN + 16 full-attention**, `full_attention_interval = 4`, attention at il 3, 7, ... 63 |
| MTP | **il=64, inside this checkpoint** - 254.2 MiB, no sidecar |
| n_embd / n_head / n_head_kv | 5120 / 24 / 4 |
| key_length = value_length | 256 |
| n_ctx_train | 262144 |
| ssm | `d_conv=4, d_inner=6144, d_state=128, dt_rank=48, n_group=16` |
| quant mix | Q4_K 9.28 + Q6_K 3.78 + Q5_K 1.96 + Q8_0 1.55 GiB; **99% repack-eligible** |
| `output.weight` | **994.6 MiB** |
| `token_embd.weight` | 682.0 MiB |
| per-block size | 201.0 - 269.3 MiB (mean: ATTN 216.7, GDN 243.1) |

**Section 0.1 is confirmed exactly.** 48 GDN + 16 attention, interval 4. The KV story is
1/4 of a comparable dense model's, as predicted.

Three things the numbers add that the source reading could not give:

1. **KV is 4.0 KiB per token per attention layer** at f16 (`(256+256) * 4 heads * 2 B`).
   With all 16 attention layers on GPU that is 64 KiB/token: 512 MiB at 8k, 1 GiB at 16k.
   But only the *offloaded* attention layers cost VRAM, and a 6 GB card holds roughly the
   last 16 blocks, which contain only 4 attention layers - so the realistic figure is
   **~128 MiB at 8k**, not 512.
2. **Recurrent state is 3.12 MiB per GDN layer per sequence** (conv 120 KiB + state 3 MiB),
   ~150 MiB for all 48, and **constant in context length**. Confirms section 0.1's claim
   that growing context is free on 3/4 of the stack.
3. **`output.weight` alone is 994.6 MiB and is the first thing offloaded.** The output
   layer occupies slot `n_layer_all`, and the GPU is filled from the top down
   (`src/llama-model.cpp:1318,1333`), so it consumes ~4 blocks' worth of VRAM before any
   block lands. Conversely `token_embd.weight` (682 MiB) is **pinned to the CPU
   unconditionally** (`src/llama-model.cpp:1334-1336`) and never competes for VRAM.

Point 3 was a bug in the first version of `02-gguf-layout.py`, which counted only blocks.
Corrected, a 5000 MiB weight budget buys **`-ngl 17`** (output + the last 16 blocks,
4897 MiB), not the "20 trailing layers" the first run printed.

Resulting split at `-ngl 17`: ~4.9 GiB of weights on the GPU, ~11.8 GiB on the CPU. On
16 GiB of RAM that leaves roughly 3 GiB of headroom.

**Measured and confirmed**, `-ngl 17`, no MTP: `CUDA0` 4643.69 MiB + `CPU_Mapped`
12171.07 MiB. The prediction above was 4897.5 MiB for the GPU including the MTP block;
subtract the 254.2 MiB MTP block that is skipped without `--spec-type draft-mtp` and the
prediction is 4643.3 MiB against 4643.69 measured - **a 0.4 MiB error**. The byte model
in `02-gguf-layout.py` can be trusted for planning.

And the host side is entirely `CPU_Mapped`, i.e. clean file-backed pages. **It fits, and
it does not swap.** See the verdict box in section 1.2.

### Full run, default fitter, `-c 8192` (`phase0/04-residency.sh`, case `default`)

Hardware turned out smaller than assumed: **GTX 1660 Ti, 5748 MiB total, 5522 MiB free**
at fit time - not 6144. Turing (`ARCHS = 750`), no tensor cores.

The fitter chose **`-ngl 13`** (output layer + 12 blocks) under the default 1024 MiB
margin: `offloaded 13/66 layers`, `CPU_Mapped 13157.83` + `CUDA0 3656.93` MiB.

Final breakdown: `CUDA0 5748 = 1354 free + (4290 = 3656 model + 120 context + 513
compute) + 102 unaccounted`.

**Three predictions confirmed to the megabyte:**

| quantity | predicted | measured |
|---|---|---|
| KV total at `n_ctx=8192` | 4.0 KiB/tok/layer x 16 layers x 8192 = 512 MiB | `llama_kv_cache: size = 512.00 MiB (8192 cells, 16 layers)` |
| KV on GPU | 3 attention layers in blocks 52-63 x 32 MiB = 96 MiB | `CUDA0 KV buffer size = 96.00 MiB` (CPU 416.00) |
| recurrent state, 1 seq | 3.12 MiB x 48 GDN layers = ~150 MiB, constant in ctx | `llama_memory_recurrent: size = 149.62 MiB`, `S (f32) 144.00` + `R (f32) 5.62` |

**Section 0.2 confirmed directly from the fitter's own trace.** Every probe logs
`n_part= 0`, the header reads `filling dense layers back-to-front`, and step 4 never
runs. The fitter emitted zero tensor overrides. Whole-layer granularity, as predicted.

Measured residency, matching the section 1.2 verdict: `peak_rss_anon 722 MiB`,
`peak_rss_file 14869 MiB`, `peak_vm_swap 132 MiB`. Nothing large is anonymous.

Throughput baseline: **pp 4.07 t/s, tg 2.02 t/s.**

### Two new findings from this run

1. **The default margin is stranding ~1.3 GiB of VRAM.** The fitter stopped at 13 layers
   with `1354 MiB free`; its target was `4498 = 5522 free - 1024 margin`, and it used
   4290, so 208 MiB was stranded *inside* the target on top of the 1024 MiB margin
   itself. Layer 14 needed only 230 MiB more. On a headless box the margin is far larger
   than necessary - `-fitt` is now the highest-value zero-code knob, worth roughly four
   more layers.

2. **`graph splits = 851 (with bs=512), 82 (with bs=1)`.** Placement is *contiguous*
   (blocks 52-63 on GPU), so a naive expectation is a handful of splits. 851 means the
   scheduler is offloading individual large-batch ops to the GPU and streaming
   CPU-resident weights across PCIe per op. That is very likely what caps prompt
   processing at 4.07 t/s on a 1660 Ti. `-nopo`/`--no-op-offload` and `-ub` are the
   levers, and neither was in the original plan. Added to the sweep.

### MTP is the headline result: +70% generation, for free

`--spec-type draft-mtp`, same `-c 8192`, fitter left to choose:

| | default | `--spec-type draft-mtp` |
|---|---|---|
| chosen `-ngl` | 13 (output + 12 blocks) | **11** (output + 10 blocks) |
| GPU model weights | 3656.93 MiB | 3424.06 MiB |
| host weights | 13157.83 MiB | 13644.91 MiB |
| prompt eval | 4.07 t/s | **4.41 t/s** |
| **generation** | **2.02 t/s** | **3.43 t/s (+70%)** |

It wins *despite* having two fewer layers on the GPU. Draft acceptance was **0.800**
(44 accepted / 55 generated), mean accepted length **3.32**, per-position
`(0.947, 0.737, 0.632)`.

That makes sense for this configuration: 13 GiB of weights sit in host RAM and
generation is bandwidth-bound, so verifying ~3.3 tokens per pass over those weights is
close to a 3x reduction in the dominant cost. **`--spec-type draft-mtp` should be the
default configuration on this box**, and it moves MTP from "question 3, nice to have" to
the single largest win found so far.

Three memory effects worth knowing, all measured:

- **MTP quadruples the target context's recurrent state.** `n_rs_seq` goes 0 -> 3 (extra
  slots for draft rollback), and `llama_memory_recurrent` grows 149.62 -> **598.50 MiB**
  (S 576.00 + R 22.50). On a hybrid arch with 48 GDN layers this is the single biggest
  MTP cost - far larger than the 162 MiB draft context itself. It is fitted correctly,
  but it is why the fitter dropped from 13 layers to 11.
- **The compute buffer shrinks, partly offsetting it**: `CUDA0 compute` 513.01 ->
  **183.55 MiB**, because the server sets `n_outputs_max = 4` instead of 2048.
- The draft context is cheap and lands entirely on the GPU: `CUDA0 KV 32.00 MiB` over
  1 layer, plus 130.02 MiB compute.

### `mmap+mlock` is not viable, as expected

`failed to mlock 216412160-byte buffer (after previously locking 2064384000 bytes):
Cannot allocate memory` - `RLIMIT_MEMLOCK` caps out around 2 GiB, so only ~2 of the
13.2 GiB got locked. It also produced the worst load time (43.6 s) and worst prompt eval
(3.80 t/s). Do not use it; raising `ulimit -l` to cover 13 GiB on a 16 GiB box would be
actively harmful anyway, since locked pages cannot be reclaimed.

**Caveat on `MemAvailable`:** it read 14171 MiB at peak, which looks like plenty of
headroom but is misleading - clean mmap'd page cache counts as "available", so the metric
stays high precisely when the weights are about to be evicted. The signal to watch is
`majflt`, which hit **20060** for this run (system-wide `pgmajfault` delta 23672). That
is real re-reading from disk during generation.

---

## 0. Two premise corrections before anything else

### 0.1 Qwen3.6-27B is not a plain dense transformer - it is a hybrid

`LLM_ARCH_QWEN35` with `n_layer == 64` is the 27B (`src/models/qwen35.cpp:28-31`). Its
layers are marked recurrent by default as:

```
is_recr_impl[i] = (i < n_layer) && ((i + 1) % full_attention_interval != 0)
```

with `full_attention_interval` defaulting to 4 (`src/models/qwen35.cpp:19-25`). So the
trunk is **48 gated-delta-net (linear attention) layers + 16 full-attention layers**, not
64 attention layers. `create_memory()` builds a `llama_memory_hybrid` for it, with
`filter_attn`/`filter_recr` splitting on `hparams.is_recr(il)`
(`src/llama-model.cpp:2229-2245`).

Consequences that change the whole sizing story:

- KV cache scales with **16 layers, not 64**. Context growth is roughly 1/4 as expensive
  as a naive dense estimate.
- The other 48 layers carry a per-sequence recurrent state that is **constant in context
  length** (`n_embd_r()` / `n_embd_s()`, `src/llama-hparams.cpp:204,221`). Growing context
  costs nothing on those layers.
- The 35B-A3B you already run is *also* hybrid (`LLM_ARCH_QWEN35MOE` hits the same branch,
  `src/llama-model.cpp:2229`). So the attention/KV half of the architecture is
  **unchanged** between your two models. What actually differs is the FFN.

### 0.2 The thing that actually changes - no routed experts - is what breaks the fitter

For the 35B-A3B, `common_fit_params` has a strong lever: push `blk.N.ffn_*_exps` to system
RAM and keep everything else on GPU (`common/fit.cpp:435-438`, `LAYER_FRACTION_MOE`). That
is what makes a 35B model fit a 6 GB card at all.

For a dense model that lever does not exist, and the fitter degrades badly:

- `hp_nex == 0`, so `n_part` is never incremented - both increment sites are guarded by
  `if (hp_nex > 0)` / `if (hp_nex)` (`common/fit.cpp:596,613`).
- With `n_part == 0`, `set_ngl_tensor_split_tbo` emits **no tensor overrides at all**
  (`common/fit.cpp:479-489` only loops `il0 .. il0 + n_part`).
- Step 3 ends with an unconditional early return for dense:
  `if (hp_nex == 0 || global_surplus_cpu_moe <= 0) { set_ngl_tensor_split_tbo(...); return; }`
  - step 4 (the partial-layer refinement) is **unreachable for dense models**.

Net: for Qwen3.6-27B the fitter can only say "the last N whole layers go on the GPU." At
~250-300 MiB per layer on a 6 GB card that is a coarse quantum, and it strands up to a
full layer's worth of VRAM.

The machinery to do better already exists and is dense-compatible - the sub-layer patterns
at `common/fit.cpp:412-430` are `blk\.N\.ffn_(gate|up|gate_up|down).*`, which match dense
FFN tensor names fine. They are simply gated off. See Phase 2.

---

## 1. Question 2 first: how the model loads, and where the swap actually comes from

I am answering this before the split question because it is the hard constraint - the
split is only meaningful once you know which bytes are file-backed and which are anonymous.

### 1.1 The naive "load into RAM, then distribute" pipeline you were worried about does not exist

`llama_model_loader::load_all_data` (`src/llama-model-loader.cpp:1413`) has three paths:

| path | CPU-resident tensors | GPU-resident tensors |
|---|---|---|
| mmap (default) | `ggml_backend_tensor_alloc(buf_mmap, cur, data)` - **no copy**, tensor points into the mapping (`:1556`) | `ggml_backend_tensor_set` reading from the mapping (`:1568`) |
| no-mmap, CUDA | `file->read_raw(cur->data, ...)` into heap (`:1578`) | 4 x 64 MiB pinned staging buffers, async upload, never a full host copy (`:1587-1636`) |
| dio | as no-mmap, bypassing page cache | as no-mmap |

There is no whole-model host staging step on any path. Good.

### 1.2 The repack trap - MEASURED, and it does not fire by default

> **VERDICT (2026-08-06, measured on the target machine): this section's central
> hypothesis is refuted for the default configuration.** `phase0/03b-buftypes.sh` at
> `-ngl 17`, `-c 4096`:
>
> | config | buffer types | host-side |
> |---|---|---|
> | *(default)* | `CPU_Mapped` 12171.07 + `CUDA0` 4643.69 | **12171 MiB, all mmap** |
> | `-nr` | identical | 12171 MiB |
> | `-nr --no-host` | identical | 12171 MiB |
> | `--no-host` | `CPU_Mapped` 12171.07 + **`CPU_REPACK` 7036.88** + `CUDA0` 4643.69 | **19208 MiB** |
>
> The default already takes the mmap path for 100% of the host-side weights.
> `CPU_Mapped` + `CUDA0` = 16815 MiB against 16827 MiB of loadable weights (file total
> minus the 254 MiB MTP block, which is skipped without `--spec-type draft-mtp`) - a
> 12 MiB residual. Nothing is being copied.
>
> Consequences:
> - **Do nothing.** The default configuration is the correct one. There is no Phase 1a
>   repack tuning to do.
> - **`-nr` is a no-op here**, so the repack-vs-throughput trade this section agonised
>   over never arises.
> - **`--no-host` is actively harmful and my earlier recommendation to try it was
>   wrong.** On its own it adds 7037 MiB of `CPU_REPACK` - a real, anonymous copy - on
>   top of the unchanged 12171 MiB mapping, for 19.2 GiB of host-side demand on a 16 GiB
>   box. That is the swap scenario, manufactured by the flag meant to avoid it.
>
> **Unexplained, and left open:** why removing the pinned-host entry from `cpu_buft_list`
> causes the repack buft to win, when no `CUDA_Host` buffer is ever allocated in the
> default case either. Both runs should reach the repack entry by the same path
> (`common/fit.cpp` is not involved; `select_weight_buft` just walks the list). Since the
> default is already optimal this does not block anything, but the flag's help text
> ("bypass host buffer allowing extra buffers to be used") suggests the interaction is
> intentional and I do not yet understand the mechanism.
>
> The remaining live item in section 1 is **1.4 (`MAP_POPULATE`)**, which is untested.

The reasoning that led to the hypothesis is kept below, because the buffer-type
priority it documents is real and still governs what happens under `--no-host`, `-ot`,
and on machines where the mmap path is unavailable.

---

This is the mechanism that *could* produce a swap hit, and it is not the pipeline - it is
buffer-type selection.

> **Correction (2026-08-06):** an earlier draft of this section framed the trap as
> *override*-triggered. That was too narrow. Every CPU-resident layer uses
> `pimpl->cpu_buft_list` whether or not any override matched
> (`src/llama-model.cpp:1322-1324`), and `select_weight_buft` returns the **first**
> entry in that list which supports the op (`src/llama-model-loader.cpp:1046-1056`).
> `make_cpu_buft_list` builds it in the order: ACCEL bufts, then the GPU's **pinned host
> buffer type**, then the repack extras, and only last the plain CPU buft
> (`src/llama-model.cpp:896-950`). So the plain mmap-able CPU buft is the *lowest*
> priority option, and nothing about `-ot` is required to skip past it. Whether the
> pinned-host or the repack buft actually wins is decided by `supports_op` at runtime -
> that is an empirical question, and the load log answers it directly (see below).
> Note pinned host memory would be worse than repack: it is page-locked, so it can be
> neither swapped nor evicted.

1. When a tensor is CPU-resident, the loader does **not** default to the plain CPU buffer
   type. It calls `select_weight_buft(...)` over `cpu_buft_list`
   (`src/llama-model-loader.cpp:1198`, or `:1177` when an override matched), which
   considers the pinned-host and "extra" buffer types first - including the weight-repack
   buft (`make_cpu_buft_list`, `src/llama-model.cpp:928-942`).
2. **Q4_K is repackable on x86.** `repack_q4_K_to_q4_K_8_bl` and the
   `q4_K_8x8_q8_K` / `q4_K_8x4_q8_K` traits exist and are selected for `GGML_TYPE_Q4_K`
   (`ggml/src/ggml-cpu/repack.cpp:3231,4535-4536,4600`). So a Q4_K_M GGUF hits this.
3. The mmap-backed buffer path requires the selected buft to be the device's *default*
   buft: `if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft)`
   (`src/llama-model.cpp:1568-1570`, `is_default_buft` set at `:1567`). The repack buft is
   not the default buft, so this test fails.
4. Falling through, `ggml_backend_alloc_ctx_tensors_from_buft` allocates a **real anonymous
   buffer** (`src/llama-model.cpp:1596`), and back in `load_all_data` there is no
   `buf_mmap` for that context, so it takes `ggml_backend_tensor_set(cur, data, 0, n_size)`
   (`:1568`) - a genuine copy, plus the repack transform.

Result: your ~13 GB of CPU-side weights become anonymous, swappable memory instead of
clean page cache. On a 16 GB box that is the swap hit.

The codebase already half-knows this - there is a one-shot warning at
`src/llama-model-loader.cpp:1173-1177`: *"tensor overrides to CPU are used with mmap
enabled - consider using --no-mmap for better performance"*. That advice is aimed at
throughput, but the memory consequence is the more important one here.

**Mitigations available today, no code:** `-nr` / `--no-repack`
(`common/arg.cpp:2381-2386`, default is repack **enabled**) and `--no-host`
(`common/arg.cpp:2388-2393`), which drops the pinned-host entry from the list. Both may
be needed to fall all the way through to the plain, mmap-able CPU buft.

**One command settles which buft actually wins.** The loader logs one line per allocated
buffer with its buffer-type name (`src/llama-model.cpp:1645`). `CPU_Mapped` means the
mmap path (good); `CUDA_Host` means pinned, non-evictable host memory; a repack buft name
means a real copy plus transform. See `phase0/README.md`.

**This is a real tradeoff, not a free win.** Repack is a substantial CPU matmul speedup,
and with 13 of 18 GB on the CPU you are heavily CPU-bound. Phase 0 must measure both.

### 1.3 With mmap intact, you will not swap - you will page-cache-evict

The mapping is `MAP_SHARED | PROT_READ` (`src/llama-mmap.cpp:447,456`). Those pages are
clean and file-backed. Linux **drops** them under pressure; it never writes them to swap.

So the correct framing of your constraint is:

> The failure mode is not swap. It is re-reading weights from disk on every token once the
> CPU-side working set exceeds page cache.

That reframing matters because the fixes are different: you are optimizing for *page-cache
residency of the CPU-side weight bytes*, and the number to drive to zero is major faults
per token, not swap-in bytes.

Corollary: `--load-mode dio` is the wrong choice here. It bypasses page cache, which forces
CPU-side weights into a real host buffer - straight back to anonymous RAM.

### 1.4 Second trap: `MAP_POPULATE` prefaults the entire 18 GB file at startup

`src/llama-model.cpp:1532` calls `ml.init_mappings(true, ...)`. That `prefetch = true`
becomes `llama_mmap(file.get(), -1, is_numa)` (`src/llama-model-loader.cpp:1356`) - i.e.
`size_t(-1)`. On Linux:

```
if (prefetch) { flags |= MAP_POPULATE; }          // src/llama-mmap.cpp:454
...
posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED);   // :463
```

`MAP_POPULATE` prefaults the whole mapping inside `mmap()`. On 16 GB RAM with an 18 GB file
this does ~18 GB of sequential I/O at startup and, because it is sequential, **the tail of
the file evicts the head**. The tail is what goes to VRAM (GPU layers are the last N -
`il0 = hp_ngl + 1 - n_gpu_layers`, `common/fit.cpp:481`). The head is exactly what has to
stay resident on the CPU. So startup leaves the page cache in the worst possible state for
this configuration, and the first tokens re-read it all.

This is the single cheapest real fix in the plan. `prefetch` is already a `size_t` byte
count threaded end to end; the only reason it means "everything" is the hardcoded `true` at
`src/llama-model.cpp:1532` and the `-1` at `src/llama-model-loader.cpp:1356`.

Note `unmap_fragment` *is* functional on POSIX (`src/llama-mmap.cpp:490`) and does run after
load (`src/llama-model-loader.cpp:1678-1686`), trimming the prefix before and suffix after
the CPU-resident byte range. Since GPU layers are the tail, the tail suffix does get
unmapped. (On Windows it is a no-op stub, `src/llama-mmap.cpp:608` - not your problem here,
but relevant if you ever mirror this to the laptop.)

---

## 2. Question 1: the RAM/VRAM split

### 2.1 Where the VRAM actually goes

Four consumers, and the fitter only models three of them:

1. Weights of the offloaded layers.
2. KV cache for offloaded **full-attention** layers only. KV placement follows layer
   placement: `if (offload) { dev = model.dev_layer(il); }`
   (`src/llama-kv-cache.cpp:214-219`). There is no per-layer KV offload control -
   `--no-kv-offload` is all-or-nothing.
3. Recurrent state for offloaded GDN layers - constant in context length.
4. Compute / scheduler buffers, sized from `min(n_ctx, n_ubatch)` (`src/llama-context.cpp:556`).

Plus, if you use MTP, a whole second context - see section 3.

### 2.2 Value-per-byte ordering for this architecture

Highest value on GPU first:

1. **MTP layer(s)** (indices `>= n_layer`) - read once per drafted token, and the draft
   loop is latency-critical. Unlike the 35B-A3B case (`NOTES.md` A.6), the 27B's MTP block
   has **no routed experts**, so the whole module is high-value. The "split MTP into
   dense/shared vs routed groups" refinement does not apply here. Simpler than the MoE case.
2. **The 16 full-attention layers and their KV** - keeping the layer and its KV on the same
   device avoids a per-token host round trip.
3. **GDN layers** - their state is small and constant.
4. **Trunk FFN `down`/`gate`/`up`** - pure bandwidth, no attached state, and the natural
   overflow target.

### 2.3 The gap: the fitter cannot express any of that

`common_fit_params` fills a **contiguous slice from the back**
(`common/fit.cpp:585-640`). It cannot say "all 16 attention layers regardless of index," and
for dense models it cannot even split a layer (section 0.2). For the 27B its entire output
is one number: `n_gpu_layers`.

Two implications:

- Near term you can beat the fitter by hand with `-ot`, which feeds the exact same override
  array the fitter writes into (`src/llama-model-loader.cpp:1162-1197`). A regex over
  attention-layer indices (every 4th) is expressible today.
- Longer term, Phase 2 makes the fitter at least sub-layer-capable for dense.

### 2.4 Tune the margin

The default per-device fit margin is **1024 MiB** (`common/common.h:472`) - 17% of a 6 GB
card, held back permanently. On a headless Ubuntu server with no compositor that is far
more than needed. `-fitt` / `--fit-target` (`common/arg.cpp:2818`) sets it. Measure the real
floor rather than guessing; every 100 MiB recovered here is roughly a third of a layer.

Related: `-fitc` / `--fit-ctx` defaults to a 4096 minimum context the fitter may shrink to
(`common/common.h:469`).

---

## 3. Question 3: MTP

### 3.1 What already works

- `--mtp` (`common/arg.cpp:3006-3009`) adds `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` and sets
  `mparams.load_mtp` (`common/common.cpp:1630`).
- With `load_mtp`, the fitter counts MTP layers into `hp_ngl`
  (`common/fit.cpp:139-141` - upstream fix 9a688e51e, already merged in this tree).
- Because MTP layers have the **highest** indices and the fitter fills back-to-front, MTP
  lands on the GPU first. That is the behavior you want, for free.
- MTP on this arch gets a plain `llama_kv_cache` rather than the hybrid wrapper
  (`mtp_on_hybrid_qwen`, `src/llama-model.cpp:2225-2231`) - the MTP head is dense-attention
  only.

### 3.2 The gap: `--fit` systematically under-counts VRAM when MTP is on

`common_get_device_memory_data_impl` creates exactly **one** context -
`llama_init_from_model(model, *cparams)` (`common/fit.cpp:65`) - at the default `ctx_type`,
then reads `llama_get_memory_breakdown(ctx)` (`:75`).

The MTP draft context (`ctx_dft`) is a **separate `llama_context`** with its own KV cache
and its own compute/scheduler buffers (`NOTES.md` Part 1 item 10, confirmed independently
from both the `create_memory` and `common/speculative.cpp` sides). None of that appears in
that breakdown.

> **CORRECTION (2026-08-06, measured): this section was wrong, and Phase 3 is mostly
> already done in-tree.** The server path - which `llama-cli` uses - *does* account for
> the MTP context. `tools/server/server-context.cpp:1150-1198` runs a **second**
> `common_get_device_memory_data` probe against the draft/MTP config and adds
> `model + context + compute` per device into `params_base.fit_params_target[i]`, i.e. it
> inflates the fitter's margin by the MTP context's measured cost.
>
> Observed on the target machine: margin went `1024 -> 1186 MiB`, exactly
> `1024 + 162.02`, matching the logged `[spec] estimated memory usage of MTP context is
> 162.02 MiB`. And that estimate was accurate: the draft context actually allocated
> `KV 32.00 MiB (8192 cells, 1 layer)` + `compute 130.02 MiB` = 162 MiB.
>
> The extra recurrent state MTP forces on the *target* context is seen too, because the
> probe uses the same cparams: the initial probe reported `context = 1110 MiB` with MTP
> vs `661 MiB` without, and the measured `llama_memory_recurrent` grew from 149.62 to
> 598.50 MiB - a 449 MiB delta that matches `1110 - 661` exactly.
>
> **What remains is only a tooling gap, not a correctness gap:** the standalone
> `llama-fit-params` binary still cannot be asked to model MTP, because `--mtp` is
> registered only for `LLAMA_EXAMPLE_DOWNLOAD` (`common/arg.cpp:3009`) and `--spec-type`
> only for speculative/server/cli (`common/arg.cpp:4102`), and options outside the
> current example are never registered (`common/arg.cpp:1385`) so their env vars are not
> consulted either. Low value - registering `--spec-type` for `LLAMA_EXAMPLE_FIT_PARAMS`
> would be a one-line change, but the server already does the right thing.

The original (incorrect) reasoning follows, kept for the record.

`--fit` would over-commit the card whenever MTP is in play, if nothing compensated. On a
card with slack this is invisible; at 6 GB it is the difference between running and OOM.

### 3.3 Carried-over open item, now on the critical path

`NOTES.md` Part 1 open item 1: does `ctx_dft`'s KV cache need to grow in lockstep with
`ctx_tgt`? `begin()` compares `pos_max` from `mem_dft` against the prompt length
(`common/speculative.cpp:1336-1352`), which leans toward yes. If yes, every context-growth
event costs VRAM in **two** caches, not one. That has to be settled before sizing MTP on a
6 GB card, not after.

---

## 4. Reacting to VRAM pressure at runtime

### 4.1 Context growth is itself a VRAM spike

`llama_kv_cache::resize()` (`src/llama-kv-cache.cpp:1243`) allocates the **full new buffer
set while the old one is still live**, copies, then swaps and frees
(`:1334-1340` onward). Transient peak = old + new. Growing 8k -> 16k momentarily costs 3x
the 8k footprint, and it happens inside a `decode()` call.

On a 6 GB card this needs an explicit headroom rule. There is none today.

### 4.2 External GPU pressure (another program wants VRAM)

Nothing exists. `NOTES.md` Part 2 Phases D-G (elastic migration) are unimplemented, and both
blocking prerequisites are still open:

- **A.4** - whether a GPU-destined tensor's host-side shadow stays mmap'd and addressable,
  which the whole demote-without-copy premise depends on.
- **A.5** - whether migration invalidates `gf_res_prev` graph reuse
  (`src/llama-context.cpp:1300`), which is a second cache distinct from `sched_need_reserve`.

This is by far the largest piece of work here and I would sequence it last.

A cheap interim that is not migration but does stop the OOM: poll
`ggml_backend_dev_memory(dev, &free, &total)` (already used at `common/fit.cpp:107,116`)
between decodes and **refuse to grow** when free VRAM is below the new allocation. Grow the
CPU side instead, or stop at `ctx_max`.

---

## 5. Phased plan

### Phase 0 - measure (no code; everything below is sized off this)

Scripts are in [`phase0/`](phase0/) - see [`phase0/README.md`](phase0/README.md) for the run
order. Get the real GGUF onto the server and capture:

- `llama-fit-params` (`tools/fit-params/`) and `-fitp on --verbose` output - per-device
  `model` / `context` / `compute` breakdown.
- Per-layer byte sizes. Attention layers and GDN layers are **not** the same size, so
  "18 GB / 64" is wrong; the whole-layer quantum varies by layer index.
- Actual KV bytes per token across the 16 attention layers, and the fixed recurrent state
  cost across the 48 GDN layers.
- MTP module size at your quant.
- `-nr` on vs off: RSS (anonymous) vs page-cache split, plus prompt-processing t/s. This is
  the load-bearing measurement for section 1.2.
- Major faults per token at steady state (`/proc/<pid>/stat` majflt), with and without the
  prefetch change. This is the real objective function, not swap.
- The true VRAM margin floor on the headless box, to replace the 1024 MiB default.

### Phase 1 - cheap wins

- ~~**1a (config only):** `-nr`, tuned `-fitt`, tuned `-fitc`.~~ **Mostly dropped.**
  Measured: the default is already fully on the mmap path, `-nr` is a no-op, and
  `--no-host` is harmful (section 1.2 verdict). What survives is `-fitt`/`-fitc` tuning,
  which is about the margin, not about residency.
- **1b (~30 lines):** bounded prefetch - **now the main Phase 1 item.** Add a
  `--prefetch` / `--no-prefetch` arg and thread the byte count through `init_mappings`
  instead of the hardcoded `true` at `src/llama-model.cpp:1532`. Best version prefetches
  only the CPU-resident byte ranges, which the loader already tracks in `mmaps_used`
  (`src/llama-model-loader.cpp:1565`). With 12.17 GiB mapped on a 16 GiB box,
  `MAP_POPULATE` over the full 16.68 GiB file is exactly the wrong startup behaviour.

### Phase 2 - dense sub-layer fitting

Remove the `hp_nex > 0` guards at `common/fit.cpp:596,613` and let step 4 run for dense
models using `LAYER_FRACTION_ATTN`/`UP`/`GATE`. The patterns already match dense tensor
names. Recovers up to a full layer of stranded VRAM.

Touches shared upstream code and changes behavior for every dense model, so it needs care
and its own before/after benchmark.

### ~~Phase 3 - MTP-aware fit probe~~ - ALREADY DONE IN-TREE

Measured 2026-08-06: the server already runs a second memory probe for the MTP context and
folds it into `fit_params_target` (`tools/server/server-context.cpp:1150-1198`), and the
target context's extra recurrent state is fitted correctly too. See the correction box in
section 3.2.

All that remains is registering `--spec-type` for `LLAMA_EXAMPLE_FIT_PARAMS` so the
standalone tool can model MTP - a one-line convenience, not a correctness fix.

### Phase 4 - growth headroom guard

Refuse-to-grow when free VRAM is below the transient peak from section 4.1, and settle the
`ctx_dft` lockstep question (section 3.3) so the guard accounts for both caches.

### Phase 5 - elastic migration

`NOTES.md` Part 2 Phases D-G. Blocked on A.4 and A.5. Do not start before Phases 0-4 are
measured and landed.

---

## 6. Open items

1. Exact hparams of the Qwen3.6-27B GGUF you intend to use - `n_embd`, GQA widths,
   `full_attention_interval` (it may not be the default 4), `ssm_*`, `n_layer_nextn`.
   Everything in section 2 is structural until these are real numbers.
2. Which quant. Q4_K_M makes section 1.2's repack trap active; a non-repackable quant would
   not.
3. Whether an MTP-head GGUF for the 27B actually exists, or whether the MTP layers ship
   inside the main checkpoint (this arch supports both - see the `mtp_only` path at
   `src/models/qwen35.cpp:38-40`).
4. `NOTES.md` A.4 - the mmap-for-GPU-tensors question - is still the highest-priority
   unresolved item in the repo, and you now have the hardware to answer it. Phase 0 does
   not cover it; it needs its own experiment.

---

## 7. Tooling limitations found while building the Phase 0 scripts

Both of these are small findings in their own right, and both shape what Phase 0 can
actually measure:

1. **`llama-fit-params` cannot model an MTP run** - section 3.2 above.
2. **`llama-bench` has no repack toggle.** It parses its own arguments and exposes `-lm`,
   `--no-host`, `-nkvo` and `-ot`, but not `-nr`/`--no-repack`
   (`tools/llama-bench/llama-bench.cpp:536-1050`). Every llama-bench number is therefore
   repack-enabled; the repack A/B in section 1.2 has to be driven through
   `llama-cli`/`llama-completion`.
3. **Passing `-ngl` explicitly disables the fitter entirely** - `common_fit_params` throws
   `"n_gpu_layers already set by user, abort"` (`common/fit.cpp:377-378`), caught and
   downgraded to a warning (`common/fit.cpp:805-807`). Useful for controlled sweeps, but it
   means "`-ngl 99` plus `--fit`" is not a safe default on a 6 GB card - the fitter bails
   and all 65 layers are attempted on the GPU. Pair explicit `-ngl` with `-fit off` to skip
   the wasted probe load.
4. **Library INFO logs are invisible at default verbosity.** `common_get_verbosity` maps
   `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4), not `LOG_LEVEL_INFO`
   (`common/log.cpp:441-451`), against a default threshold of 3 (`common/common.h:523`).
   Everything the loader prints - `llama_model_loader:`, `print_info:`, `load_tensors:`,
   the per-buffer sizes, and `common_memory_breakdown_print`'s table - needs `-lv 4`.
   Only WARN and ERROR appear by default, so a run that logs nothing looks identical to a
   run that succeeded quietly.
5. **`llama-cli` is the interactive TUI client**, not the old one-shot tool. It forces
   `params.verbosity = LOG_LEVEL_ERROR` (`tools/cli/cli.cpp:36`) and does not act on
   `-no-cnv`. Use `llama-completion` for anything scripted.
