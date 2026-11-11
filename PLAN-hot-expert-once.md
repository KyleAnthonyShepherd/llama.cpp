# PLAN - read each hot expert once per step (CUDA MUL_MAT_ID for the expert hot store)

Written 2026-11-06. Planning only, nothing built. Base: branch `parallel-pages` d36e62a01. Build it on the tree from PLAN-upstream-sync-2026-11.md, not on today's tree (section 6).

## 1. What Kyle wants

With hydra heads, several tokens in one decode step often pick the same expert. When that expert sits in the GPU hot store, the weights should be read from VRAM once for all those tokens, not once per token.

## 2. Facts to start from (checked in the code 2026-11-06)

- **The tier graph** (`src/llama-expert-tier.cpp:113`, `llama_expert_tier_build`): the hot path is `ggml_mul_mat_id(dst_hot, cur, ids_hot)` on the GPU. `ids_hot = hot_lut[selected]`, and a cold draw maps to slot 0. Slot 0 is a real expert, so its result is multiplied by `hot_d = 1 - cold` to zero it. The cold path is `GGML_OP_MUL_MAT_ID_COLD` on the CPU. The two are added.
- **Consequence 1: dummy work.** Every cold draw (about 55% of draws at 28 slots) runs slot 0 on the GPU and throws the result away.
- **Consequence 2: repeated ids.** A token with several cold draws has slot 0 several times in its id list.
- **The small-batch kernel** (`ggml/src/ggml-cuda/mmvq.cu:978`, `mul_mat_vec_q_moe`): grid `(rows/2, n_expert_used)`, block `(warp, n_tokens)`, one warp per token, each warp reads the rows of `ids[token][slot]` on its own. Two tokens on the same expert read it twice (L2 catches some of it). Repeated ids are harmless here, which is why the tier works on MMVQ.
- **The large-batch kernels** (MMQ, MMF) first run `mm_ids_helper` (`ggml/src/ggml-cuda/mmid.cu`), which groups rows by expert, so each weight tile is read once per column tile. That is the "read once" behaviour we want. But the helper assumes unique ids per token: with a repeated id, the generic path keeps only the last draw, the templated path has two lanes write the same slot, and `nex_prev` counts the duplicate, so every later expert's offsets shift. Rows are lost or written in the wrong place.
- **So the tier stops at the MMVQ bound.** `ggml_cuda_mul_mat_id` (`ggml-cuda.cu:2141`) uses MMVQ while `ne2 <= get_mmvq_mmid_max_batch(type, cc)` (8 for Q4_K/Q6_K on Turing+), MMQ/MMF above. `llama_expert_tier_widen` reads that bound through `ggml_backend_cuda_mmid_mmvq_max_batch` and the tier bypasses itself above it (all experts on the CPU).
- **The CPU cold op already groups by expert** (`ggml-cpu-mul-mat-id-cold.c:96`, `matrix_row_counts`): a cold expert used by 3 heads is read from RAM once per step.
- **Upstream has defined the contract we need.** 701fd25b3 (#26631, 2026-10-24): in MUL_MAT_ID "an id of -1 skips the slot and zeroes the matching row of the result", and in ADD_ID "an id of -1 adds nothing". CPU, Metal, Vulkan, SYCL, OpenCL, WebGPU, Hexagon implement it. **CUDA does not**: `test-backend-ops` skips those cases on CUDA/ROCm/MUSA behind `// TODO: [TAG_MUL_MAT_ID_SKIP] ... remove this once CUDA implements mul_mat_id with -1 = skipped`. The tests already exist (`mask_skipped_ids`: every 4th row skips all slots, other rows skip some).
- **Upstream rewrote the MoE layout** in 69ee2ecc0 (#31047, 2026-11-05): `mm_ids_helper` now emits `ggml_cuda_mm_id_tile_spec {expert, col_low, col_high}` tiles over a linear column order instead of `expert_bounds`, for both MMQ and MMF. Tiles no longer pad each expert to a full tile, which is what makes MMQ cheap when each expert has only 1-4 tokens. Also 016c1d914 (MMVQ bound from the highest compiled arch), f5222f25c (gate/up + GLU fused in MMQ for MoE).

Measured step costs (PLAN-parallel-pages.md section 8, laptop, 35B, MTP n_max 1):

| heads | tokens per step | step | where the experts run |
|---|---|---|---|
| 4 | 8 | 160 ms | hot store on GPU + cold on CPU |
| 8 | 16 | 430 ms | all on the CPU (tier bypassed) |

## 3. What it can win (derived, to be checked in M0)

Per layer, a 4-head step with MTP has 8 tokens x 8 draws = 64 draws. At a ~43% hit rate (28 slots, decay 0.995 measured 47%, 0.999 measured 39%): ~28 hot draws, ~36 cold. One slot is 1.42 MiB.

| GPU reads per step (40 layers) | today | after this plan |
|---|---|---|
| hot draws | 28 x 1.42 MiB x 40 = 1.6 GiB, ~5 ms at ~330 GB/s | distinct hot experts only: ~14-18 per layer -> 0.8-1.0 GiB, ~2.5-3 ms |
| dummy slot-0 draws | 36 x 1.42 MiB x 40 = 2.0 GiB, mostly L2 hits, ~1-2 ms | none |

So at 4 heads on the laptop **the direct saving is ~3-5 ms of a 160 ms step (2-3%)**. The step is dominated by the CPU: ~28 distinct cold experts per layer x 40 layers x ~80 us (the fit from the expert-pool work) = ~90 ms.

The larger value is what the fix unlocks: once ids can repeat or be -1 on MMQ, **the tier no longer stops at 8 tokens**. Then:

- 8 heads with MTP (16 tokens) keep their hot experts on the GPU instead of running everything on the CPU (the 430 ms row). Rough estimate: ~50 distinct cold per layer -> ~160 ms CPU, plus ~5 ms GPU, plus the rest of the step, ~230-250 ms for 16 tokens. That would beat 4 heads per token (160 ms per 8). On the 6 GB laptop and server, though, 8 heads also take ~1 GiB of state cells, which shrinks the hot store (section 8 wall 2), so the hit rate drops. M3 measures this.
- 4 heads with MTP n_max 2 (12 tokens) become possible on the hot store.
- A card with more VRAM gets the full benefit.

If M0 shows the GPU hot path is a bigger share than derived here, step B (section 4) becomes worth more. If it shows less, stop after step A.

## 4. The design

Three steps. Each is useful alone, and each has a gate.

### Step A - CUDA implements id -1 (upstream's contract)

Make every CUDA MUL_MAT_ID and ADD_ID path follow 701fd25b3:

- **MMVQ single-token** (`mul_mat_vec_q` with ids, `ncols_dst == 1`): `channel_x = ids[channel_dst]`. If it is -1, write zeros for this block's rows and return. The whole block shares one id, so the early return is safe.
- **MMVQ multi-token** (`mul_mat_vec_q_moe`): one warp per token, no shared memory, no block sync. A warp whose id is -1 writes zeros for its rows and returns.
- **Fused paths**: the gate+up+GLU fusion (`ggml_cuda_should_fuse_mul_mat_vec_q` for MUL_MAT_ID), the bias and scale fusions, and the fork's PTQ1_0 / PQ2_0 MoE variants (PrismML warp-per-row PTQ1_0 for MUL_MAT_ID, the MoE epilogue fusions). A skipped slot gives 0 after the GLU too: silu(0) * 0 = 0, and the bias must not be added for a skipped slot.
- **`mm_ids_helper`** (MMQ and MMF): a -1 never matches an expert, but `nex_prev += expert_used < expert` counts it. Exclude negatives there. Then zero the skipped rows of dst. The rows nobody writes are exactly the (token, slot) pairs with id -1, so either one extra block in the helper writes their `ids_dst` positions to a "zero list", or a small kernel zeroes dst rows from the ids tensor before the matmul. Pick whichever fits upstream's tile-spec version after the sync.
- **The generic fallback** (cuBLAS per expert, with a stream sync): skip -1 when it builds its row lists, zero those rows.
- **ADD_ID**: -1 copies src0 unchanged.
- Remove the CUDA exclusion in `test-backend-ops` (`TAG_MUL_MAT_ID_SKIP`).

Done when: `test-backend-ops -o MUL_MAT_ID` and `-o ADD_ID` pass on CUDA, including the `mask_skipped_ids` cases, on the laptop (sm_86) and in the server recipe (sm_75, `GGML_CUDA_NO_MMA`, `GGML_CUDA_FORCE_MMQ`). CUDA graphs still capture (`ggml_cuda_mul_mat_id_needs_sync` unchanged for these shapes).

This step is generic and upstream asks for it, but it stays in the fork (Kyle, 2026-11-06). Keep it in its own commit so it is easy to drop if upstream lands its own CUDA version in a later sync.

### Step B - the tier uses -1 for cold draws

In `llama_expert_tier_build`:

- `hot_lut[e] = -1` for a cold expert instead of 0. The LUT is f32, and -1.0f casts back to -1 exactly.
- Drop the `hot_d` multiply: the hot result is already 0 for a cold draw. Keep the `w_s` per-expert scale multiply.
- The hot path no longer has repeated ids, so MMQ and MMF are valid for it. Raise the tier limit above the MMVQ bound: a new default (start at 32, below the 128-token op-offload threshold), still lowered by `llama_expert_tier_widen` when a weight type has no MMQ path.
- Cold experts are no longer computed twice (once as a dummy on the GPU), so the GPU does only the hot draws.

Expected: bit-identical output for batches up to 8 (x * 1 = x, and the dummy rows were exactly 0 before too, barring inf/NaN in slot 0). Above 8 the hot path moves from "everything on the CPU" to MMQ on the GPU, so results change within the usual MMQ vs CPU rounding.

Done when: 35B greedy runs at temp 0 give the same text before and after for 1-token and 2-token (MTP) steps, `LLAMA_EXPERT_HITRATE` reports the same hits, and a 4-head hydra with MTP n_max 2 (12 tokens) logs `expert tier engaged` instead of bypassed.

### Step C - one read per hot expert in the small-batch kernel (only if M2 says so)

After step B, batches up to 8 still use the MMVQ MoE kernel: one read per draw. Two ways to read each expert once:

1. **Use MMQ (or MMF) for tier batches from 2 tokens up.** After the sync, MMQ's linear tile layout packs 1-4 tokens per expert without padding each expert to a full tile. This is a threshold change, no new kernel: a tier-only override of `get_mmvq_mmid_max_batch` (the tier knows its ids are short lists over few slots, which the generic tuning does not). It costs the q8_1 MMQ-layout quantize and the ids helper launch per matmul.
2. **A grouped MMVQ kernel.** Grid `(rows / rows_per_block, n_slots)`: each block takes one expert, reads its weight rows once, and computes the dot products for all tokens that chose it (gathered through the helper's `ids_src1`, scattered through `ids_dst`). It reuses the existing `mul_mat_vec_q<type, ncols_dst>` inner loop, which already reads a weight row once for up to 8 columns, with a runtime column count rounded up to 1, 2, 4 or 8. Blocks of experts no token chose exit at once. Keep the gate+up+GLU fusion so gate and up are each read once.

Choose by M2. Option 1 is far less code and stays close to upstream. Option 2 only if option 1 is slower than today's MMVQ at 2-8 tokens.

## 5. Measurements and gates

- **M0 - where a hydra step's time goes (no code).** nsys on the laptop, 4-head triage group with MTP: GPU time of the tier's hot MUL_MAT_ID per step, CPU time of `MUL_MAT_ID_COLD`, the rest. Plus `LLAMA_EXPERT_SEL_DUMP` on the same run: distinct hot and cold experts per layer per step, and how often two heads share one. **Gate:** if the hot MUL_MAT_ID is under ~5% of the step, steps A and B are still worth it for the token limit (M3), but skip step C.
- **M1 - kernel timings (test-backend-ops perf mode).** Add MUL_MAT_ID perf cases with the hot store's shapes: 28 slots (and 14), 8 used, n_tokens 1, 2, 4, 8, 12, 16, 32, about 55% of ids set to -1, Q4_K and Q6_K, 2048 x 512 and 512 x 2048. Compare MMVQ (today), MMQ and MMF after the sync, with and without -1. Laptop and server.
- **M2 - gate for step C.** From M1: if MMQ at 2-8 tokens is within ~10% of MMVQ, nothing is gained there. If MMQ is faster, take step C option 1. If both are well above the "distinct experts x bytes / bandwidth" floor, consider option 2.
- **M3 - live, after step B.** `toolbag/tests/hydra/pages_sweep.py` K = 4, 5, 6, 8 with MTP n_max 1, and K = 4 with n_max 2. `trans_hydra.py` at 4 heads. Compare s/page and step ms with section 8 of PLAN-parallel-pages.md. Same on the home server for one run, since its 1660 Ti has less bandwidth and a different MMVQ/MMQ balance.

## 6. Order and dependencies

1. M0 can run now (no code). It needs the laptop GPU, so it waits until Kyle's geoui run is done.
2. Steps A to C go on the synced tree (PLAN-upstream-sync-2026-11.md). 69ee2ecc0 rewrote `mm_ids_helper`, `mmq.cuh` and `mmf.cuh`. Built on today's tree, step A's MMQ/MMF half would have to be redone during the merge.
3. Build in a separate worktree and build dir, never in `build-cuda` while Kyle's server runs from it.

## 7. Risks

- **Silent row loss.** If any CUDA MUL_MAT_ID path misses the -1 handling, it reads expert -1 (out of bounds) or leaves garbage. The `mask_skipped_ids` tests cover MMVQ, MMQ and MMF only if their shapes reach each path: add cases that force each one (n_tokens 1, 2-8, >8; quantized and F16).
- **Fusion paths** are separate kernels with their own id reads (gate/up/GLU, bias, NVFP4 scales, PrismML PTQ1_0 / PQ2_0 MoE variants). Each needs the -1 branch.
- **0 x inf.** Dropping `hot_d` removes a multiply. If anything upstream of the tier ever relied on that multiply to kill a NaN in slot 0, it no longer does. -1 avoids reading slot 0 at all, so this should only make things safer.
- **The tier limit vs. the compute buffer.** A larger tier batch changes the reserved graph (more MUL_MAT_ID nodes take the tier shapes). Check the compute buffer size at load with `-ehs -1`, and that fit still plans the same slot count.
- **The home server** (sm_75, NO_MMA, FORCE_MMQ): MMQ there is the dp4a path, with a different crossover than the laptop's MMA path. M1 must run there before the tier limit's default is set.
- **VRAM, not the kernel, may cap the heads.** On 6 GB cards, 8 heads' state cells shrink the hot store (PLAN-parallel-pages.md section 8). Lifting the token limit helps only as far as the slots remain.

## 8. Files

- `ggml/src/ggml-cuda/mmvq.cu` (both kernels, fused variants), `mmid.cu` / `mmid.cuh`, `mmq.cu` / `mmq.cuh`, `mmf.cu` / `mmf.cuh`, `add-id.cu`, `ggml-cuda.cu` (dispatch, fallback path, the tier-limit proc).
- `tests/test-backend-ops.cpp`: remove the `TAG_MUL_MAT_ID_SKIP` exclusion, add path-forcing and hot-store-shape cases (no new test file).
- `src/llama-expert-tier.cpp`, `src/llama-expert-tier.h` (limit default, the MMQ id compaction TODO goes away), `src/llama-expert-hotstore.cpp` (LUT fill: -1 for cold).

## 9. Decisions and open questions

- Step A stays in the fork, no upstream PR (Kyle, 2026-11-06). In each later sync, check whether upstream has closed `TAG_MUL_MAT_ID_SKIP`; if so, take theirs and drop ours.
- Still open: run M0 before the sync, or wait and do everything after it?

## 10. Progress (2026-11-10, Claude)

- **Step A committed** (5a4b91fc8, on the synced tree): every CUDA MUL_MAT_ID path and ADD_ID take -1. Two things the plan did not foresee: MMQ/MMF `tile_spec` seam entries are written only up to the compact column count, so with skipped slots the tail held garbage (now memset to empty tiles first); and the fork's CPU MUL_MAT_ID had lost the -1 skip in its grouping pass in the sync (9edd386f7). test-backend-ops: MUL_MAT_ID 1251/1251 (170 with skipped ids), ADD_ID 72/72, on sm_86 default and with NO_MMA + FORCE_MMQ. The home server (sm_75) is not run yet.
- **Step B committed** (ea8c159b2): LUT -1 for cold, no `hot_d`, default limit 32, `llama_expert_tier_widen` asks the new proc `ggml_backend_cuda_mmid_tier_max_batch` (MMQ path -> any batch, else the MMVQ bound). Fit plans the same slot count as before step B (30). 4 heads with MTP n_max 1: 2.55 s/page (2.99 pre-sync, before steps A/B).
- The "same greedy text" gate cannot be checked: with the hot store on, two runs of one build already differ (residency depends on timing, also with `LLAMA_EXPERT_SWAP=sync`); with `-ehs 0` they are identical. With limit 4, MTP texts matched before/after, non-MTP did not (same as run-to-run).
- The 12-token gate (4 heads, n_max 2) runs since the compute buffer fixes (PLAN-upstream-sync-2026-11.md section 10): the server logs "the expert hot store serves them (up to 32)", 2.43 s/page on 4 triage pages (pre-sync, tier bypassed above 8 tokens: 3.07-3.20).
- Not done: M0-M3 measurements, path-forcing and hot-store-shape test cases, step C.
