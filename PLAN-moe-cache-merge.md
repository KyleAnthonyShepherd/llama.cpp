# Upstream MoE cache vs our expert hot store, and how to merge it

Written 2026-10-28. Upstream state: `upstream/master` 39febd2ad (2026-10-28), 935 commits past our merge base 972d2313b (2026-09-17). Our side: branch `expert-prefetch-adaptive-ub` at 03ac05aa0.

## 1. Verdict

- Take upstream's MoE cache in with the next full upstream sync. Leave its files exactly as upstream has them, keep it off by default (it already is, `--moe-cache-mib 0`), and make it mutually exclusive with our hot store (`-ehs`).
- Do not replace our hot store with it. On a 6 GB card holding about 15% of the 35B's experts, upstream's design has to upload every miss over PCIe x8. Our design reads the same bytes from DDR4 on the CPU, which is about 2.5x faster per byte. Upstream's own thread puts break-even near an 85% hit rate on PCIe 4 x8. Our store measures about 55% (section 5). This is derived, not measured: section 8 E2 measures it.
- Do not cherry-pick it ahead of the sync. Every one of its commits except the SLRU one conflicts with our tree (section 6.1), and the same conflicts come back in the sync anyway.
- Two upstream ideas are worth porting into our store, each gated on a measurement: one slot pool shared across layers (7.1), and on-device copies of hot experts during batch uploads (7.2).

## 2. What upstream built

Commits:

| commit | date | what |
|---|---|---|
| 6753a033f (#29943) | 10-06 | Moves "upload only the experts the batch uses" out of `ggml_backend_sched_compute_splits` into a user callback, `ggml_backend_sched_set_copy_callback`. llama implements it as `llama_context::sched_copy_experts`. Host weights are now copied in a second pass after all other split inputs, so the callback can read the router ids. |
| d6cf9acb2 (#29887) | 10-07 | `src/llama-moe-cache.{h,cpp}` (550 lines), `--moe-cache-mib N` / `LLAMA_ARG_MOE_CACHE_MIB`, `llama_context_params::moe_cache_size`. Port of qvac-fabric's cache. |
| c811cb8f0 (#30112) | 10-08 | Multi-GPU: the budget is split like the layers. Disables pipeline parallelism when the cache is on. |
| 33d6135cb (#30637) | 10-19 | LRU -> 50/50 segmented LRU (probation + protected). Upstream reports 1-5% better. |
| a4699400e (#31141) | 10-26 | `no_alloc` moved out of `llama_hparams` (touches the cache and about 10 of our call sites, see 6.3). |

How it works (single GPU, which is our case):

- Only layers whose experts are **all** host-resident are cached. Layers with the same expert tensor layout form a group. Each group owns one device "bank" per expert tensor (gate, up, down) of `n_slots + 1` experts. Any slot can hold any (layer, expert) pair, so the pool is shared across the layers of a group.
- Each cached layer has a host-side I32 slot map `[n_expert]` (-1 = not cached). The graph adds `slots = get_rows(slot_map, selected_experts)` and runs `mul_mat_id(bank_view, cur, slots)` on the GPU. The slot map is a host weight, so the scheduler starts a split at the lookup and calls the copy callback for it.
- The callback (`llama_moe_cache::copy`) reads the selected ids back from the GPU (`get_async` + `synchronize`, once per layer per ubatch), runs the SLRU plan, uploads every missing expert from host memory (`set_async`, merging runs of consecutive experts), then uploads the slot map.
- So for small batches **every routed expert runs on the GPU**, and misses are paid for with PCIe uploads in the critical path. The CPU computes no experts.
- Used only for ubatches of at most 32 tokens (`max_batch`), and only when one ubatch's distinct experts fit in the group's slots. Larger batches go through the normal offload path. The cache is read-only there: `copy_experts` copies cached experts device-to-device into the upload destination and uploads only the rest.
- The draft (speculative) context never gets a cache: `common_base_params_to_speculative` sets `moe_cache_size = 0`.
- Size is a fixed MiB budget given by the user. It shows up in `memory_breakdown()` as context memory, so `-fit` accounts for it but does not size it.
- Logs hits, misses and uploaded MiB at context teardown (info level).

Upstream's numbers (PR #29887, Qwen3.8-Flash-Next Q4_0, 65 GiB of experts, PCIe 4.0 x16, EPYC): about 10% of the experts cached gives a 72% hit rate and 1.57x generation on a 4090. At 29% of the experts cached, a 5090 reached an 89% hit rate and 2.2x. No prompt-processing numbers. A commenter estimated break-even against CPU decode at about an 85% hit rate on PCIe 4 x8. The PR author said hit rates below 75% were "not helpful on a slow PCIe4 box". A GLM 5.3 report shows the cache **losing** speed at a 5% hit rate (14.6 -> 13.35 tok/s).

## 3. What we have

`src/llama-expert-{hotstore,tier,heatmap}.{h,cpp}` (1207 lines, started 2026-08-03), the fit autosize in `common/fit.cpp`, and on this branch the scheduler prefetch (`ggml_backend_sched_set_prefetch`) plus the adaptive ubatch (`-ubg`).

- Fixed `hot_s` slots **per layer** (every layer gets the same count), chosen by `-fit` from leftover VRAM (`-ehs -1`). One slot is one expert's gate+up+down in one layer, about 1.42 MiB on Q4_K_M, so 55 MiB per slot across 40 layers.
- Which experts are hot: a decayed per-layer heat map (0.999 per token), re-synced every 50 tokens with a hysteresis/dwell gate. Uploads happen at re-sync time, off the per-token path.
- Graph: `llama_expert_tier_build` replaces `mul_mat_id` for registered tensors. The hot experts run on the GPU from the hot tensor via a LUT. The **cold experts run on the CPU** (`GGML_OP_MUL_MAT_ID_COLD`) from the host tensor, and a mask removes the dummy contributions. Nothing crosses PCIe per token for misses.
- Used for batches of up to 4 tokens (`LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT`): a cold draw maps to slot 0, so a token's id list holds duplicates, which the MMQ/MMF id compaction miscounts.
- Shrinks and grows with the KV cache (`resize`, `slots_that_fit`). It lends VRAM to the MTP draft context's large ubatch (`set_n_ubatch`, `ctx_other`). It is not built for MTP contexts.
- Large batches: `-epf N` uploads whole host expert tensors one split ahead on a second CUDA stream, into scheduler-owned slots, redirecting the split input's data pointer at launch.

## 4. Where they diverge

| | upstream `--moe-cache-mib` | ours `-ehs` / tier / `-epf` |
|---|---|---|
| Miss handling at decode | upload over PCIe, then GPU | CPU computes it from host RAM |
| Admission | on demand, every ubatch, SLRU | heat ranking, re-sync every 50 tokens, hysteresis |
| Slot layout | one pool per layout group, shared across layers | `hot_s` per layer, equal for every layer |
| Per-ubatch host sync | 1 readback + sync per cached layer | none for routing (re-sync reads selections on its cadence) |
| Batch range | 1..32 tokens | 1..4 tokens (duplicate-id limit) |
| Large batches | normal selective copy, cached experts copied on-device | whole-tensor prefetch on a second stream (`-epf`), no on-device reuse |
| Sizing | fixed MiB from the user | autofit (`-ehs -1`), live resize against KV growth |
| Draft context | none | none (MTP context excluded), but the draft borrows hot-store VRAM for its ubatch |
| Backends | any GPU/iGPU with the ops | CUDA only (Vulkan corrupted output, `--ecf` to force) |
| Multi-GPU | yes (c811cb8f0) | no |
| Code location | llama only (`llama-moe-cache.cpp`) plus the generic copy callback in ggml | llama plus a custom ggml op (`MUL_MAT_ID_COLD`) plus scheduler changes in ggml |

## 5. Which is faster here (derived)

Per decoded token the 35B routes 40 layers x 8 experts x 1.42 MiB = **454 MiB** of expert weights (Q4_K_M). With hit rate `h`, both designs run the hits on the GPU. They differ only in the misses, `(1-h) x 454 MiB`:

- Upstream uploads them: PCIe 4.0 x8 measured at about 9.5 GiB/s (35B prefill measurements, 2026-10-24).
- Ours reads them on the CPU: about 24 GiB/s effective (26 GB/s, measured in the 27B runs, PLAN-qwen36-35b-moe.md section 2).

At our measured `h` of about 0.55 (PLAN-qwen36-35b-moe.md 15.3, 14-16% of experts resident), the misses cost about **21 ms per token uploaded vs 8 ms on the CPU**. To tie, upstream needs `(1-h_up) <= 0.39 x (1-h_ours)`, i.e. `h_up >= ~0.82` at the same VRAM. That matches the 85% break-even quoted in upstream's thread.

Unknowns that could change this, so the answer is still a measurement (section 8 E2):

- Upstream's SLRU on our model may hit more often than our heat ranking at the same capacity (it got 72% at 10% capacity on Qwen3.8-Flash-Next). E1 answers this offline.
- Per-layer readback stalls on upstream's side, and whether our CPU cold path overlaps the GPU hot path, are not measured.
- The home server (1660 Ti, other PCIe link, other RAM) has its own ratio. The model only fits there if the host pages it. Upstream's thread notes misses then also pay page faults.

## 6. Merge

### 6.1 Trial results

`git merge-tree` of each commit onto 03ac05aa0, run 2026-10-28 (no checkout touched):

| commit | conflicting files |
|---|---|
| 6753a033f callback refactor | `ggml/src/ggml-backend.cpp` (3 hunks, real), `ggml/include/ggml-backend.h`, `src/llama-context.cpp` (1 line), `src/llama-context.h`, `tests/test-llama-archs.cpp` |
| d6cf9acb2 cache | `common/common.{h,cpp}`, `include/llama.h`, `src/llama-context.{h,cpp}`, `src/llama-graph.{h,cpp}`, `tests/test-llama-archs.cpp` |
| c811cb8f0 multi-GPU | `common/common.h`, `include/llama.h`, `src/llama-context.cpp` |
| 33d6135cb SLRU | clean |

Everything except `ggml-backend.cpp` is adjacency: both sides added a field, include or parameter on neighbouring lines. The `llama-graph.cpp` hunk is really about Hadamard: upstream merged its own Bonsai support (a9d27ac69, `build_hadamard_input`), which collides with our PrismML merge whatever we do about the cache.

### 6.2 Route

Do it as part of the regular sync, in the order already used for this fork:

1. `master` <- `upstream/master`. Master has the hot store and tier but not the prefetch. Conflicts here: params, context ctor, `build_lora_mm_id`, the selective-copy code (ours still has it inline in the scheduler, with no edits on master), and the `no_alloc` rename.
2. `Ternary` <- `master`. Adds the Hadamard collision (PrismML vs upstream Bonsai). Unrelated to the cache, but it lands in `build_lora_mm_id` too.
3. `expert-prefetch-adaptive-ub` (or wherever the prefetch lives by then) <- `Ternary`. The scheduler conflict lands here: our prefetch and async user-input copies against upstream's rewritten copy loop.

Cherry-picking the five commits now would mean resolving the same files twice.

### 6.3 Resolution rules

- **`src/llama-moe-cache.{h,cpp}`**: take upstream's files unchanged. No fork edits there, so every later sync applies cleanly.
- **Params** (`common.h`, `common.cpp`, `llama.h`, `llama-cparams.h`, `arg.cpp`, the defaults in `llama_context_default_params`): keep both sides' fields.
- **Mutual exclusion**, in the context constructor before either is built: if `moe_cache_size > 0`, resolve `-ehs -1` (auto) to 0 and turn off `-epf`. An explicit `-ehs N > 0` together with `--moe-cache-mib` is an error with a clear message. The fit autosize must see `expert_hot_s = 0` in that case, or it plans a store that is never allocated. Both hooks would claim the same `mul_mat_id`, and the cache's per-ubatch upload would fight the hot store for VRAM.
- **`build_lora_mm_id`**: upstream's order is `build_hadamard_input`, then the `slots`-or-plain `mul_mat_id`. Put our `llama_expert_tier_build` call after `build_hadamard_input` and before the `slots` branch. With mutual exclusion only one of them fires, so the order is for readability.
- **`build_moe_ffn`**: take upstream's `build_moe_cache_slots` call as is. It returns nullptr when the cache is off.
- **Scheduler creation**: upstream now creates the scheduler in one place and calls `ggml_backend_sched_set_copy_callback` right after. Every scheduler we execute graphs on needs that callback, and our `set_prefetch` call goes next to it. If one is missed, nothing breaks: offloaded batches silently upload whole expert tensors instead of the used ones. `set_n_ubatch`'s `sched_tmp` only measures and does not need it, but check whether any scheduler built there survives the swap.
- **`ggml_backend_sched_compute_splits`** (step 3): rebuild our two changes on upstream's structure instead of keeping our version of the loop:
  - Pass 1 (non-weight inputs): our async user-input batching (wait once, `set_async`, one sync after the loop, and treat a view of a user input as user data) replaces the `GGML_TENSOR_FLAG_INPUT` branch of `ggml_backend_sched_copy_input`. Give it a small per-split state for "waited" and "pending".
  - Pass 2 (host weights): check `input == pf_input` first (wait on `pf_ready`, swap `input_cpy->data` to the slot, restore after launch). Otherwise call `ggml_backend_sched_copy_input`, which reaches the callback.
  - The selective copy that used to sit inline is now `llama_context::sched_copy_experts`. Keep upstream's version.
  - Prefetch candidates (at least 2 picks per expert, about 64+ tokens) and cache batches (at most 32 tokens) never overlap, so the prefetch never bypasses a cache lookup.
- **`no_alloc`** (a4699400e): mechanical rename at our `model.hparams.no_alloc` / `hparams.no_alloc` sites (`llama-context.cpp` 716, 878, 897, 917, 920, 931, 1548, 1649, 4797; `llama-kv-cache.cpp` 304, as of 03ac05aa0), plus any in the hot store and fit code.
- **Tests**: take upstream's `test-llama-archs.cpp` MoE-cache config. It needs a GPU and covers only their path.

### 6.4 Checks after the merge

- Build CUDA on the laptop, plus the server recipe (arch 75, NO_MMA) once.
- `-ehs -1 -epf 2 -ubg 512` with Kyle's flags: same slot count at load, the same generation and prompt speed as before the sync (sustained 600-token runs), and identical output at temp 0 with `-ehs 0 -cmoe`.
- `--moe-cache-mib N -ehs 0`: loads, generates, logs its stats at exit.
- Both flags together: the error from 6.3.

## 7. Ideas to port into our store

### 7.1 One slot pool shared across layers (likely the largest win)

Today every layer gets `hot_s` slots. Measured heat says layers differ a lot: after 128 tokens layer 0 had 225 of 256 experts warm, layer 9 only 126 (PLAN-qwen36-35b-moe.md 15.3). A pool shared across the layers of a layout group, filled by global heat per byte, spends the same VRAM where it covers more activations. Upstream shows `mul_mat_id` over one bank with remapped ids works. Our LUT already remaps ids, so the tier's graph barely changes: the hot tensor becomes `{ne0, ne1, n_slots_total}` per group, and `hot_lut[layer][e]` points into it.

Touches `llama-expert-hotstore.cpp` (allocate, copy_top_s, resync_top_s, resize, slots_that_fit), the hot tensor shape in `llama-expert-tier.cpp`, and the fit autosize (a total slot count instead of a per-layer one). Q4_K_M mixes Q4_K and Q6_K `ffn_down` across layers, and unsloth UD quants mix more, so expect several groups, like upstream.

Kyle chose to build it and measure it directly, without a gate: PLAN-expert-shared-pool.md.

### 7.2 On-device copies of hot experts during uploads

When a batch uploads a host expert tensor, the experts the hot store already holds could come from VRAM instead of PCIe, as upstream's `copy_experts` does.

- Selective-copy path (offloaded batches below the prefetch threshold, about 32-63 tokens): after the merge this is about 30 lines in `llama_context::sched_copy_experts`, mirroring upstream's loop over runs. Cheap, but these batch sizes are rare in Kyle's use.
- Prefetch path (whole tensors): it saves at most hot_s/256, about 14% of the upload bytes, at the cost of splitting one copy into many. It is worth it only where prefill is upload-bound: ub 2048 and below, or the server. At ub 4096 the laptop measured 3.8 s upload vs 5.0 s kernels per 8k, which the prefetch now overlaps, so it would gain nothing there.

Gate: E3.

### 7.3 Not taken

- Demand upload per ubatch, as the replacement for the CPU cold path: section 5.
- SLRU as our admission policy: only if E1 shows it beats the heat ranking.
- A fixed MiB budget instead of autofit and resize: ours fits the KV growth and the ubatch switching. Upstream's does not.

## 8. Experiments

**E1 - policy and pool simulation (offline, one capture).** Add an env var that appends each batch's router selections (`llama_expert_sel`, already read when `LLAMA_EXPERT_HITRATE` is set) to a binary file. Capture one long real chat on the 35B. A Python replay then compares hit rates at equal VRAM for: (a) our per-layer heat ranking with decay 0.999, a 50-token re-sync and the current hysteresis; (b) global heat across layers (7.1); (c) upstream's SLRU, a re-implementation of `moe_cache_slru::plan` with one pool per layout group. This answers both 7.1 and section 5's open question without GPU time. Superseded for 7.1: Kyle chose to build and measure directly, see PLAN-expert-shared-pool.md.

**E2 - A/B after the merge (laptop, then the server).** Same VRAM for the cache as the hot store holds (its MiB from the load log): `-ehs -1 -epf 2 -ubg 512` vs `-ehs 0 -epf 0 --moe-cache-mib <same>` vs no cache. Sustained 600-token generations with MTP n_max 1, plus prompt speed at 8k. Record the hit rates (upstream logs at exit, ours with `LLAMA_EXPERT_HITRATE`). Kyle's server on 8080 locks the DLLs: use a second build dir and port.

**E3 - upload vs kernel time per ubatch at ub 2048** (nsys, as in the 2026-10-24 prefill measurements), laptop and server. If upload exceeds kernel time with `-epf`, 7.2's prefetch part is worth building.

## 9. Open questions for Kyle

- Merge timing: the cache arrives with the next upstream sync (935 commits, which also bring upstream's Bonsai Hadamard support against our PrismML merge). Is a sync planned soon, or should 7.1 go ahead on the current base after E1?
- On the mutual-exclusion error: should `--moe-cache-mib` win silently instead (warn, disable `-ehs`)? An error is safer for launch scripts that already pass `-ehs -1`.

## 10. Result of 7.1 (shared slot pool), 2026-10-28

Built as `--expert-pool shared` on branch `expert-shared-pool` (default stays `layer`) and measured on the unsloth UD-Q4_K_S. Full report: `I:\Kyle_Shepherd\toolbag\tests\expert_pool\RESULTS.md`.

- Two layout groups (37 Q4_K layers, 3 layers with Q6_K down). Correct, same VRAM, survives `-ubg` lending and resizes.
- Replayed on the same selection traces, the hit rate rises by +0.50 points at 28 slots, +0.71 at 14, +0.85 at 7. Distinct cold experts per token fall by only 0.5-0.7%.
- Live generation speed: no difference beyond run noise at either budget. Prompt speed unchanged.
- The slots leave layers 0-4 (flat routing, 12% hit at 14 slots) for layers 8-30, and the deep layers gain about as little as the early ones lose. 7.1 is not the large win it looked like.
- Follow-up the same day (RESULTS.md 7b-7c): the heat's time scale matters far more than the slot layout. Live, `--expert-heat-decay 0.995` lifts the hit rate 38.8 -> 47.2% at 28 slots and 24.2 -> 32.3% at 14. Speed: +5.3% at 14 slots, none at 28. Swaps cost ~0.73 ms each (blocking per-slice copies in resync), which eats the gain; batching them like `copy_top_s` is the next step.
- Swaps now run in the background (ac53601a5): decode blocks ~0.01 ms per swap instead of 0.34 ms. With that, a faster decay pays at both budgets: +4.4% at 28 slots (decay 0.995), +6.2% at 14 slots (decay 0.99). The `--expert-heat-decay` default is still 0.999. RESULTS.md 7d.
