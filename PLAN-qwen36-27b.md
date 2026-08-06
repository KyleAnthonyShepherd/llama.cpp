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
16 GiB of RAM that leaves roughly 3 GiB of headroom - **this fits without swapping, as
long as the CPU-side weights stay on the mmap path.** Which is exactly what section 1.2
is about, and is now the single load-bearing unknown.

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

### 1.2 The real trap: `--repack` silently converts your CPU-side weights into anonymous RAM

This is the mechanism that produces the ~2 GB swap hit you predicted, and it is not the
pipeline - it is buffer-type selection.

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

So `--fit` will over-commit the 6 GB card whenever MTP is in play. On a card with slack this
is invisible; at 6 GB it is the difference between running and OOM.

**It is worse than an under-count: `llama-fit-params` cannot be asked about MTP at all.**
`--mtp` is registered only for `LLAMA_EXAMPLE_DOWNLOAD` (`common/arg.cpp:3009`) and
`--spec-type` only for speculative/server/cli (`common/arg.cpp:4102`). Options outside the
current example are never registered (`common/arg.cpp:1385`), and since env-var application
iterates the registered option list, `LLAMA_ARG_SPEC_TYPE` does not work around it either.
So there is no invocation of the fitting tool that models an MTP run.

Phase 3 therefore has two parts: register `--spec-type` for `LLAMA_EXAMPLE_FIT_PARAMS`, and
make the probe build the MTP context. Until then, MTP VRAM has to be measured empirically -
`phase0/04-residency.sh` does this by diffing `nvidia-smi` peak usage between a plain
`llama-cli` run and a `--spec-type draft-mtp` run.

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

- **1a (config only):** `-nr`, tuned `-fitt`, tuned `-fitc`. Measure.
- **1b (~30 lines):** bounded prefetch. Add a `--prefetch` / `--no-prefetch` arg and thread
  the byte count through `init_mappings` instead of the hardcoded `true` at
  `src/llama-model.cpp:1532`. Best version prefetches only the CPU-resident byte ranges,
  which the loader already tracks in `mmaps_used` (`src/llama-model-loader.cpp:1565`).

### Phase 2 - dense sub-layer fitting

Remove the `hp_nex > 0` guards at `common/fit.cpp:596,613` and let step 4 run for dense
models using `LAYER_FRACTION_ATTN`/`UP`/`GATE`. The patterns already match dense tensor
names. Recovers up to a full layer of stranded VRAM.

Touches shared upstream code and changes behavior for every dense model, so it needs care
and its own before/after benchmark.

### Phase 3 - MTP-aware fit probe

Have `common_get_device_memory_data_impl` also construct an MTP context when
`mparams->load_mtp` is set, and sum both memory breakdowns. Contained to `common/fit.cpp`.

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
   and all 65 layers are attempted on the GPU.
