# PLAN - resync with upstream llama.cpp (November 2026)

Written 2026-11-06. Planning only: `git fetch` and in-memory `git merge-tree` trials, no checkout or build touched. Our tip: `parallel-pages` d36e62a01. Upstream tip: `upstream/master` 18ab62a8e (2026-11-06).

## 1. Where things stand

- **Our base** is 972d2313b (upstream 2026-09-17, the last sync, `upstream-sync-2026-09`). Upstream is **1146 commits** past it.
- **Our side** is one straight line. Every fork branch is an ancestor of `parallel-pages`: master, Ternary (with the PrismML merge), smart-dry, expert-cache-vram, expert-prefetch-adaptive-ub, expert-shared-pool, qwen35-4b, hydra, lfm3, loop-guard-json-schema. The only outlier, `mmproj-compute-lazy`, is already in by patch id (`git cherry` shows `-`). So the old three-step route (master, then Ternary, then the prefetch branch, PLAN-moe-cache-merge.md 6.2) is no longer needed. **One merge into one branch covers everything.**
- **Our changes**: 278 non-merge commits, 516 files, about 98k lines added. Most of that is plans, notes and `phase0/` scripts. The code is mainly `src/` (6.5k lines), `tools/server` (4.5k), `ggml-cuda` (3.3k), `tools/ui` (3.1k), `ggml-cpu` (2.7k), `ggml-metal` (1.8k, from PrismML).
- **PrismML** (`prismml/prism` 62c418ccd, 2026-11-02) merged upstream up to 2026-10-26 and has **160 commits of its own** that we do not have yet (Hopper kernels, FWHT/GDN fusions, PTQ1_0 / PQ2_0 work, batched scheduler uploads #393, a head-72 pad #390).

## 2. Trial merge (`git merge-tree parallel-pages upstream/master`)

**94 files conflict, 305 hunks, ~9,300 lines inside conflict markers.** Largest:

| file | hunks | lines | what collides |
|---|---|---|---|
| `tools/server/server-context.cpp` | 20 | 1360 | hydra heads, loop guard, hats, ctx-limit vs ref-counted checkpoints (e152332a1), spec checkpoint restore (7df8b876b), realtime API, LFM3 audio |
| `conversion/lfm3.py` | 7 | 573 | our LFM3 port (cherry-picks with other SHAs) vs upstream's LFM3 |
| `tools/ui/.../stores/chat/index.svelte.ts` | 7 | 539 | fork UI (hats, prefill bars, hydra) vs upstream UI (compaction, search, ...) |
| `src/llama-context.cpp` | 12 | 440 | hot store, seq_reserve, n_seq_alloc, prefetch, ubatch lending vs MoE cache, MTP conv snapshots, model-member refactor (e0f96a1a1), state copy sizes |
| `src/llama-kv-cache.cpp` | 1 | 422 | `--ctx-max` growth / shrink vs `flash_attn_ext_rows` per-sequence cell bitmaps (1b6935b22), kv shift fix (6023692b1) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 7 | 417 | fork dispatch (tier proc, FWHT, MMVQ tables) vs MMQ MoE fusions, MoE layout |
| `ggml/src/ggml-backend.cpp` | 3 | 370 | our prefetch + async inputs vs the copy callback (6753a033f) and the batched copy API (418f905a4) |
| `src/llama-model.cpp` | 3 | 369 | per-model build_rope, new models |
| `src/llama-graph.cpp` | 10 | 356 | tier hooks, Hadamard (PrismML) vs upstream Bonsai + `prism.hadamard` LoRA (0682a576a), delta-net conv refactor (b74298c10), output structs (4b851fbee) |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 10 | 278 | PrismML Metal vs upstream Metal |
| `tools/mtmd/clip.cpp` | 9 | 269 | `--mtmd-pack`, lazy encoder buffers vs TTS / video / new encoders |
| `common/speculative.cpp` | 8 | 221 | per-seq MTP `pending_h`, `common_speculative_seq_cp` vs MTP hidden save/restore (0f35d6d7b), `--spec-draft-n-ubatch` |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 12 | 181 | PrismML GDN fusion vs the GDN chunked kernel (62df1815b) |

Plus `common/fit.cpp` (9 hunks), `src/llama-memory-recurrent.cpp` (4), `include/llama.h` (3), `mmq.cuh` (6), `mmvq.cu` (2), `tests/test-recurrent-state-rollback.cpp`, and about 60 small adjacency conflicts (both sides added a field or an include on neighbouring lines).

Stepping stones measured the same way: merging only up to 4413a1924 (2026-10-12, 552 commits) gives 62 files / 182 hunks; up to 7aca0d2ea (2026-10-26, 885 commits) gives 73 / 224.

## 3. What upstream brings that matters to us

- **`flash_attn_ext_rows`** (1b6935b22): with a unified KV cache, each sequence's attention reads only its own cells through a `kv_rows` list instead of a dense mask over every cell. CUDA supports it. This is aimed at exactly the hydra case: 4 heads in a 31k-54k cell unified cache, where today every head's attention scans every head's cells. It may explain part of the home server result (4 heads ~ 1 head of total speed). **Measure it first after the merge.**
- **MUL_MAT_ID id -1 = skip** (701fd25b3), CUDA still TODO: the base for PLAN-hot-expert-once.md.
- **MoE layout for MMQ/MMF** (69ee2ecc0) and gate/up + GLU fusion in MMQ for MoE (f5222f25c; c49edf096 did the same for dense FFNs with interleaved gate/up rows): faster prompt processing on the 35B's offloaded batches.
- **MoE cache** (`--moe-cache-mib`, 6753a033f, d6cf9acb2, c811cb8f0, 33d6135cb, and since then c4fb57969 LRFU/ClassLRU): take it untouched and off, per PLAN-moe-cache-merge.md.
- **Server**: reference-counted checkpoints (e152332a1, cheaper than our deep copies), restore of the spec checkpoint for long drafts (7df8b876b), logprobs with speculative decoding, realtime API.
- **Recurrent memory and MTP**: refuse rollbacks past the last ubatch's snapshots (ab3768cdc), MTP conv snapshots in one set_rows (040ce199a), MTP hidden save/restore (0f35d6d7b), delta-net conv state refactor (b74298c10), state copy size fix for quantized KV (e56f006ed).
- **LFM3 / LFM3-VL** upstream (19cce6919, d4f1b18b3 video, bc428e407, f9448cddc), which replaces our port.
- `no_alloc` moved out of `llama_hparams` (a4699400e), `n_pad` removed from memory (7a2e5aba9), `llm_graph_context` refactors (e0f96a1a1, 4b851fbee): mechanical, but they touch our code in many places.

## 4. Route

1. **Baselines first, on today's build** (section 6.3), so every regression is measurable.
2. **A new worktree and branch**: `git worktree add ../opt-llama-sync -b upstream-sync-2026-11 parallel-pages`, tag `pre-sync-2026-11` at d36e62a01. The main folder, `build-cuda` and Kyle's running server are never touched.
3. **Two merge passes**, as in the last sync:
   - **Pass 1**: `git merge 4413a1924` (upstream 2026-10-12). It brings the scheduler copy callback and the MoE cache, the hardest part of PLAN-moe-cache-merge.md, without LFM3, kv_rows, the new MoE layout or the checkpoint rewrite. 182 hunks. Build, run the quick checks (6.1), commit.
   - **Pass 2**: `git merge upstream/master`. The rest.
   - One pass would be fewer hunks in total (305 vs 182 plus the second pass), but the midpoint gives a build that can be tested, and a place to bisect from.
4. ~~PrismML as an optional third merge~~: skipped (Kyle, 2026-11-06, section 9).
5. **Resolve by area** (section 5), one area at a time, building after each area where the tree allows.
6. **Checks** (section 6), then **switch the daily build**: Kyle's server moves to a build from the new branch only after the side-by-side runs pass. Push the branch to origin, then the home server pulls and rebuilds with the full cmake line from the home-server notes.

## 5. Resolution rules by area

### 5.1 Take upstream as is (and re-add only what the fork needs)

- **LFM3 / LFM3-VL** (`conversion/lfm3.py`, `src/models/lfm3.cpp`, `lfm-base.cpp`, `lfm2.cpp`, `tools/mtmd/models/lfm3vl.cpp`, `mtmd-image.cpp` LFM parts): our port is cherry-picks of upstream commits plus two fork fixes. Take upstream's files, then check the two gaps the lfm3 port found: (1) does the fork still need `ggml_build_forward_expand` at the end of each model graph, or did upstream's central `build_logits()` change make it unneeded? (2) is the CUDA `ssm_conv` kernel size 7 in upstream (our 39da130fc)? Keep `models/templates/LFM3-VL-2B-loop-guard.jinja` and the SWA append rule (d048cadf2).
- **Metal, Hexagon, SYCL, OpenCL, Vulkan, WebGPU, CANN**: we do not run them. Where our only change came from the PrismML merge, take upstream. PrismML's Metal work comes back with a PrismML merge if it is wanted.
- **`AGENTS.md`, docs, CI, `tests/fusion/MTL.csv`**: take upstream.
- **MoE cache files** (`src/llama-moe-cache.*`): upstream unchanged.

### 5.2 Ours, ported onto upstream's new structure

- **Scheduler** (`ggml-backend.cpp`, `ggml-backend.h`): follow PLAN-moe-cache-merge.md 6.3. Keep upstream's two-pass input copy loop and the copy callback. Rebuild our `-epf` prefetch (pass 2: `input == pf_input` first) and our async user-input batching on it. Check whether upstream's batched copy API (418f905a4) or PrismML's batched uploads (#393) already does what our async input batching does. If so, drop ours.
- **MoE cache vs hot store**: mutually exclusive in the context constructor (PLAN-moe-cache-merge.md 6.3), fit must see `expert_hot_s = 0` when the cache is on, every executing scheduler gets `set_copy_callback`.
- **CUDA MUL_MAT_ID** (`mmid.cu`, `mmq.cuh`, `mmf.cuh`, `mmvq.cu`, `ggml-cuda.cu`): take upstream's tile-spec layout. Re-apply the fork's per-type `GGML_CUDA_MMVQ_MAX_COLS` sm_86 table (c48345606), the `ggml_backend_cuda_mmid_mmvq_max_batch` proc (upstream 016c1d914 now picks the bound from the highest compiled arch, so the proc must call it the same way), the tier's needs, and PrismML's PTQ1_0 / PQ2_0 paths.
- **Graph** (`llama-graph.cpp/.h`, `src/models/qwen35.cpp`): tier call in `build_lora_mm_id` after `build_hadamard_input` and before upstream's moe-cache `slots` branch. Hadamard: upstream now has Bonsai Hadamard (a9d27ac69) and `prism.hadamard` LoRA folding (0682a576a). Keep one implementation: upstream's where it covers the same tensors, PrismML's fused CUDA kernels behind it. Adapt to output structs (4b851fbee) and `model` as a member (e0f96a1a1).
- **Recurrent memory** (`llama-memory-recurrent.*`, `llama-memory.h`, hybrid wrappers): keep our tails array, `seq_cells`, `seq_reserve`, `n_seq_alloc`. Port upstream's rollback refusal (ab3768cdc), the `no_alloc` handling, the GLM5Next/Qwen4Exp MTP changes and the restore-cleanup fix into it. `test-recurrent-state-rollback.cpp`: keep our `test_seq_reserve` and take upstream's new cases.
- **KV cache** (`llama-kv-cache.*`, `llama-kv-cells.h`): the hardest single hunk. Upstream adds a per-sequence cell bitmap (the transpose of `seq`), kept in `seq_pos_inc/dec`. Our `--ctx-max` growth and the shrink resize the cell arrays: the bitmaps must be resized and rebuilt with them, or kv_rows reads past the end after the first growth. Also the hydra's head cells (`seq_cp` from the root) must show up in each head's bitmap.
- **Context** (`llama-context.*`, `include/llama.h`, `llama-cparams.h`): keep `n_seq_alloc` **last** in `llama_context_params` (after every upstream field). Any change to that struct means a full rebuild of every tool binary (old binaries crash on the layout). `no_alloc` rename at our call sites.
- **Speculative / MTP** (`common/speculative.*`, `examples/speculative-simple`): upstream now saves and restores the MTP hidden state (0f35d6d7b). Ours keeps a per-sequence `pending_h` and copies it on `seq_cp` for hydra heads. Merge them into one mechanism: upstream's save/restore, extended per sequence. Check `--spec-draft-n-ubatch` (3fb47677e) against our draft ubatch lending from the hot store.
- **Server** (`server-context.cpp/.h`, `server-common.*`, `server-task.h`, `server-queue.cpp`, `server-schema.cpp`): take upstream's `shared_ptr` checkpoints and move the fork's users onto them: loop guard light park and restore, hats (RAM slot store), checkpoints in RAM, `--ctx-limit`, hydra heads (which skip checkpoints). Keep our hydra code paths (`is_head` slots, `hydra_reserve/release/trim`, `find_mtmd_group`).
- **Fit** (`common/fit.cpp`): keep our hot-store autosize and `n_seq_ctx`, add upstream's NextN skip (2c192444c) and the router memory margin (85cc432e2).
- **mtmd** (`clip.cpp`, `clip.h`, `mtmd.cpp`, `clip-impl.h`): keep `--mtmd-pack`, the lazy encoder buffers and the head-72 pad (PrismML #390 has one too: keep one), take upstream's new encoders.
- **Web UI** (`tools/ui/...`): keep the fork's features, take upstream's. Build with portable Node 22 on PATH, or the build silently embeds upstream's prebuilt UI.

### 5.3 Small conflicts

`common/common.{h,cpp}`, `common/arg.cpp`, `src/llama-arch.*`, `src/llama-ext.h`, `gguf-py/gguf/constants.py`, `conversion/qwen.py`, `tests/CMakeLists.txt`, `tests/test-chat.cpp`, `tests/test-llama-archs.cpp`: keep both sides.

## 6. Checks

### 6.1 After each pass (quick)

- CUDA build in the worktree's own build dir (laptop flags from the fork notes: `-DGGML_NATIVE=OFF`, AVX2/FMA/F16C/AVX512/VNNI/VBMI on). Use the `JAVA_TOOL_OPTIONS` workaround if any JVM appears (not expected here).
- `test-backend-ops -b CUDA0` for MUL_MAT, MUL_MAT_ID, ADD_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, the FWHT ops.
- `test-recurrent-state-rollback` with `-c 1024` (its "dirty-ctx" check already fails at the base commit, do not chase it).
- The 35B loads with Kyle's flags (`-ehs -1 -epf 2 -ubg 512`, MTP n_max 1, `--hydra-heads 4`): same hot-store slot count at load, a coherent reply.

### 6.2 After the last pass (full)

- `test-llama-archs`, `test-chat` (static CPU build, it only builds static on Windows), the server's Python unit tests that touch slots and checkpoints.
- Server recipe build once (sm_75, `GGML_CUDA_NO_MMA`, `GGML_CUDA_FORCE_MMQ`): check `CUDA : ARCHS = 750 | NO_MMA = 1` and `UI: npm build succeeded`.
- toolbag rigs: `tests/hydra` (`pages_check.py`, `normal_bench.py`, `heads_bench.py`, `split_check.py`, `trans_hydra.py`), `tests/loop_guard` (forced trips, reflect on heads), `tests/hats` (RAM slot store save/restore), `tests/expert_pool` (replay).
- LFM3-VL: one page transcription and the loop guard rig with and without `--swa-full`.
- `--moe-cache-mib N -ehs 0` loads and logs its stats; `--moe-cache-mib` with `-ehs N` gives the error.
- geoui V3-Hydra end to end on a scratch `GEOUI_HOME` and another port, never Kyle's data folder.

### 6.3 Performance, before vs after (same box, same flags)

Record on today's build before merging, then repeat on the new one:

- 35B: sustained 600-token generation, MTP n_max 1 (tok/s, draft acceptance); prompt speed at 8k (`-ubg 512 -ub 4096`).
- Hydra: 4-page triage group (s/page, PLAN-parallel-pages.md section 8), split transcription of one Micron sheet (section 11).
- Hydra with kv_rows: the same two runs, with the KV cells of a long transcription (31k+). That shows whether `flash_attn_ext_rows` helps the heads.
- Qwen3.5-4B MTP best config (PLAN-qwen35-4b.md) and LFM3-VL Q4_0 tg, since those were tuned by kernel tables.

The GPU runs need the laptop GPU free. Kyle's geoui run is using it now, so 6.3's baselines wait until it is done, or until Kyle says to stop it.

## 7. PrismML (optional, Kyle's call)

PrismML is itself a fork of upstream at 2026-10-26, with 160 commits of its own since our last merge of it. Options:

- **Skip it this time.** Only the Ternary-Bonsai 27B and PTQ1_0 / PQ2_0 models need it. The kernels we already merged stay. Upstream now has its own Bonsai Hadamard, so the PrismML-specific part shrinks.
- **Merge it after upstream** as a separate commit (`git merge prismml/prism` on the synced branch). Since it already contains upstream to 10-26, its conflicts are mostly its own CUDA files (`fwht.cu`, `gated_delta_net.cu`, `norm.cu`, `mmvq.cu`, `vecdotq.cuh`) and Metal.

Recommendation: skip it unless the Bonsai 27B is still in use. Then merge it as step 4.

## 8. Risks

- **The KV cache bitmap vs growth/shrink** (5.2) is the place where a wrong merge gives wrong attention without a crash. Test with `--ctx-max` growth in the middle of a generation (TESTING.md sections 3-5) and with hydra heads.
- **Checkpoints rewrite vs loop guard and hats**: a lost restore gives a reply that continues from the wrong state. The loop guard rig's forced trips and the hats rig catch it.
- **Silent fallbacks**: a missed `set_copy_callback` uploads whole expert tensors (slower, still correct); a deleted build dir loses CUDA and the fork UI; a missing Node 22 embeds upstream's UI. Check the load log and `system_info` each time.
- **Struct ABI**: any binary built before the merge crashes against the new `llama_context_params`. Rebuild everything in the new build dir.
- **Size**: 305 hunks is several working sessions. The two passes and the per-area order keep each step testable.

## 9. Decisions (Kyle, 2026-11-06)

- PrismML: **skip** this time (section 7). Route step 4 is dropped.
- **Two passes**: 4413a1924 first, then `upstream/master` (section 4).

Still open: after the sync, does the daily branch become `upstream-sync-2026-11`, or should it be fast-forwarded into `parallel-pages` / `master`?

## 10. Progress (2026-11-07, Claude)

- Baselines on today's build (`toolbag/tests/sync/baseline.sh pre`, data in `toolbag/tests/sync/data/pre/`): 35B 9.7k prompt 818/892 t/s, 600-token gen 34.0/30.2 t/s (MTP acc 0.80); 4 triage pages 2.99 s/page; split log 98.5% in 72.9 s; Qwen3.5-4B MTP n_max 4 mean ~137 t/s; LFM3-VL Q4_0 tg 176.4, pp4096 6600.
- Worktree `I:\Kyle_Shepherd\opt-llama-sync`, branch `upstream-sync-2026-11`, tag `pre-sync-2026-11`.
- **Pass 1 committed** (ca94e0cf3): builds, test-backend-ops CUDA passes except upstream's new `dup=1` MUL_MAT_ID cases (CUDA mmid helper is byte-identical to upstream: the repeated-id gap of PLAN-hot-expert-once.md, then an illegal access). 35B loads with 28 hot slots, coherent text; pass1 numbers: gen 31.5/30.4 t/s, 3.08 s/page.
  - Open: the 512-token ubatch compute buffer grew 287 -> 382 MiB (two expert weight copies alive at once), which made split_check's 3-head hydra fail with 503 after a long media run. Re-measure after pass 2.
  - test_multi_seq_split_replay fails, but it also fails on the pre-sync tree (it never ran before because dirty-ctx stopped the test first; the merge fixed dirty-ctx).
- **Pass 2 committed** (447c1a806, 2026-11-10). Choices: LFM3 files from upstream; fork `llama_model_get_tok_embd_rows` renamed `_inp` (upstream has its own with another signature); hot store autosize moved into upstream's `common_fit_apply`; MMQ keeps PrismML GB10 args after upstream's `fusion` arg; GDN chunked MMA kernel only when gates are not raw and no fused cache; our seq-reserve test moved to `tests/llama-context/cases-rs.cpp` (case "reserve"); KV `n_pad` dropped as upstream did; one draft `n_ubatch` field, capped at `-ub`.
  - Checks: CUDA test-backend-ops pass for the fork's ops (upstream disabled the `dup=1` MUL_MAT_ID cases with `#if 0`); test-llama-context on Qwen3.5-0.8B (`-c 4096`, else the 262k default OOMs) 17 pass, `replay` fails as before.
  - Fixed after: 9edd386f7, the CPU MUL_MAT_ID lost upstream's -1 skip in its second (grouping) pass.
- PLAN-hot-expert-once.md steps A (5a4b91fc8) and B (ea8c159b2) are in, see section 10 there.
- **Post baselines** (`data/post/`, `data/post_35b.log`): LFM3 pp4096 6782 (6600), tg128 179.1 (176.4); 4B MTP code 151.7, prose 98.6, extract 178.4, mixed 108.5, reason 169.1 (148.1 / 98.7 / 173.6 / 107.9 / 159.3); 35B prompt 881/1223 t/s (818/892), gen 34.1/33.8 (34.0/30.2), MTP acc 0.89 (0.80); 4 triage pages 2.55 s/page, 12/12 right (2.99).
- **Both open items fixed (2026-11-10/11):**
  1. **Silent server deaths = heap overflow in the KV cells** (b9e214d17). Upstream's per-sequence cell bitmap `seq_bits` (1b6935b22, for `flash_attn_ext_rows`) is sized once, when a sequence takes its first cell; `--ctx-max` grows the cells in place, so the first cell past the old size wrote past the bitmap (512 bytes = 4096 cells). Windows ends a process with a corrupt heap at once, with no handler run. Found with ASan (`/fsanitize=address`, MSVC runtime from VS `bin/Hostx64/x64` on PATH, `ASAN_OPTIONS=detect_container_overflow=0`). New test-llama-context case `grow`; ASan reports the overflow there without the fix. 8eb2cf221 adds Windows backtraces (DbgHelp) and reports for abort, uncaught exceptions, SEH, CRT invalid parameter; `GGML_BACKTRACE_VEH=1` also reports heap corruption. The Release build in the worktree now has `/Zi` + `/DEBUG` (PDBs), configure it from PowerShell (Git Bash turns `/O2` into a path).
  2. **Compute buffers** (probe: `toolbag/tests/sync/alloc_peak_probe.diff`, `GGML_ALLOC_PEAK=1` prints the live tensors at each new peak).
     - 512-token +95 MiB was fragmentation: the GPU copy of the KQ mask (a graph input) was allocated at its first use, at 168 MiB, and stayed for the whole graph; the 210 MiB copy of a larger `ffn_down_exps` did not fit under it. d5ab88f3d flags copies of graph inputs as inputs, so they go to the start of the buffer: 382 -> 286 MiB (287 pre-sync).
     - The CUDA MoE weighted-reduction fusion kept the experts alive until its output, which blocked the in-place MUL and gave the weighted copy its own buffer that the fused kernel never writes (also pre-sync). 047836a14 keeps the last MUL instead. 35B, 9.7k prompt: main 4096-token 1474 -> 1288 MiB (1632 pre-sync), MTP 4096-token 835 -> 579 MiB (793). Perplexity identical with the old and new dependency.
     - Left: vision `reserve_compute_meta` 331 vs 314 MiB, MTP 512-token 199 vs 193 MiB, and upstream's gate/up+GLU MMQ fusion keeps both expert weight copies alive by design (alloc deps in `ggml_backend_cuda_graph_optimize`).
     - The fit's lost ~190 MiB of slack (pre-sync counted the RS buffer twice) stays lost; the buffer savings more than cover it.
  - Final (`data/post2_35b.log`): 35B prompt 790/1177, gen 33.5/33.9; 4 pages 2.71 s/page 12/12; split_check 98.5% cells in 65.2 s (pre 98.5% / 72.9 s). 4 heads with MTP n_max 2 (12-token steps, OOM before): 2.43 s/page, 728 MiB free at the image group (pre-sync 3.07-3.20 s/page, 365 MiB).
- Server recipe: `build-srv/` in the worktree is test-backend-ops with `-DGGML_CUDA_NO_MMA=ON -DGGML_CUDA_FORCE_MMQ=ON` for sm_86 (sm_75 PTX does not JIT here, the toolkit is newer than the driver). Not yet run on the home server.
