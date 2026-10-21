# PLAN - Qwen3.5-4B (unsloth MTP GGUF, Q4_K_S) on the laptop and the home server

Target: make `unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_S` as fast as possible for single-user chat and agent work. The model fits whole on the GPU on both machines, so this is a different problem from the 35B-A3B (expert cache) and Bonsai (ternary kernels): every byte is in VRAM, and what is left is (a) how many bytes a token reads, (b) how well the kernels use the bus, and (c) how much time the host adds between GPU steps.

Research and measurements 2026-10-20. Laptop numbers are measured. Server numbers are not measured yet (section 6).

---

## 1. The model

From `phase0/02-gguf-layout.py` on the real file (2.50 GiB):

| | |
|---|---|
| blocks | 33 = 32 trunk + 1 MTP (`nextn_predict_layers = 1`, inside the file) |
| layer classes | 24 GDN + 8 full attention, interval 4 (attention at il 3, 7, ... 31) |
| n_embd / n_head / n_head_kv / head dim | 2560 / 16 / 4 / 256 |
| GDN | 16 k-heads, 32 v-heads, head dim 128, conv 4 |
| vocab | 248,320, **tied embeddings**: `token_embd` is also the LM head |
| quant mix | Q4_K 1.71 GiB, **Q6_K 0.49 GiB (the one tied head tensor)**, Q5_K 0.26, Q8_0 0.01 |
| per block | GDN 62.5-67.8 MiB, ATTN 57.7-60.8 MiB, MTP 71.0 MiB |

What one token reads:

- target step: trunk 1.93 GiB + LM head 0.49 GiB = **2.42 GiB**. The head is 20% of it.
- one MTP draft token: MTP block 71 MiB + **the full LM head 497 MiB** = 568 MiB, **23% of a target step**. Nearly all of a draft is the head.

That last line is the big difference from the 27B, where the head was 6% of the bytes and a draft was nearly free. On the 4B a draft token costs about a quarter of a real one, so deep drafts stop paying early.

KV at f16: 8 attention layers x 4 heads x 256 x 2 x 2 B = **32 KiB per token** (1 GiB at 32k). GDN state is about 2 MiB per layer per sequence, about 50 MiB for all 24, constant in context length; MTP adds `n_max` rollback snapshots of it.

---

## 2. Baseline on the laptop (RTX 3060 Laptop 6 GB, sm_86)

`llama-bench -ngl 99 -fa 1 -t 6`, build c8f88eb92:

| test | t/s |
|---|---|
| pp512 | 1921 |
| pp2048 | 1915 |
| tg256 | 69.9 |
| tg256, `GGML_CUDA_DISABLE_GRAPHS=1` | 63.8 (CUDA graphs are worth ~9%) |

llama-server, same model, 400-token chat replies, no MTP: **64.9 t/s greedy, 62.7 t/s** with Qwen's non-thinking sampling (temp 0.7, top-p 0.8, top-k 20, presence 1.5).

### 2.1 The GPU is power capped during decode

`nvidia-smi` while decoding: **65 W, SM clock ~1100 MHz** (max 2100), memory clock at full 6001 MHz, utilisation 98%. The card is a 192-bit, 12 Gbps part: 288 GB/s peak. So decode runs on a power budget, not only on bandwidth, and kernels that spend more ALU per byte pay twice.

### 2.2 Where a token goes (nsys, `--cuda-graph-trace=node`, tg256)

GPU kernel time is 14.0 ms per token, which is the whole token (no launch gaps inside llama-bench):

| part | ms/token | notes |
|---|---|---|
| Q4_K + Q5_K trunk mat-vec | 8.9 | ~235 GB/s, ~80% of peak |
| LM head (Q6_K, 497 MiB, one kernel) | **2.54** | 205 GB/s, ~71% of peak |
| GDN recurrence | 0.35 | 14.6 us x 24 |
| ~20 other small kernels (rms_norm, quantize_q8_1, get_rows, concat, cpy, ssm_conv, scale, add, silu, rope, FA) | ~2.2 | 2-6 us each, hundreds per token |

Pure kernel headroom without MTP is modest: a perfect run at 288 GB/s is ~8.4 ms. Realistic kernel work (better Q6_K head, fusing small ops) is worth maybe 10-15%.

### 2.3 llama-server adds ~1.7 ms per token on the host

nsys on llama-server, no MTP: the GPU is idle **13% of the decode span**, in 476 gaps over 20 us, about 1.7 ms per token. There are ~27 `cudaStreamSynchronize` per token. The logits copy alone is 1 MB per token (248k x f32), and CPU sampling over 248k tokens is the rest. Qwen sampling costs 2.2 t/s more than greedy. `-bs` (backend sampling) removes the sampling gap: 63.3 t/s for both greedy and Qwen sampling.

---

## 3. MTP on the laptop

`phase0/qwen35-4b-mtp-sweep.py`: one llama-server per config (`-ngl 99 -fa on -c 8192 -np 1 -t 6`), four fixed prompts (code, prose, JSON extraction, reasoning), 400 tokens each, thinking off, greedy and Qwen sampling. Cell = t/s (draft acceptance).

### 3.1 Draft depth

Greedy:

| n_max | code | prose | extract | reason | **mean** |
|---|---|---|---|---|---|
| none | 65.4 | 65.1 | 64.6 | 64.6 | 64.9 |
| 1 | 84.9 (.92) | 76.0 (.71) | 85.9 (.96) | 84.6 (.93) | 82.8 |
| **2** | 90.0 (.86) | 74.8 (.62) | 93.8 (.92) | 90.3 (.87) | **87.2** |
| 3 | 88.1 (.78) | 64.1 (.48) | 94.8 (.88) | 91.5 (.83) | 84.7 |
| 4 | 87.1 (.73) | 54.3 (.35) | 93.5 (.80) | 92.3 (.78) | 81.8 |
| 6 | 73.2 (.57) | 43.0 (.27) | 88.7 (.73) | 80.9 (.65) | 71.5 |

Qwen sampling:

| n_max | code | prose | extract | reason | **mean** |
|---|---|---|---|---|---|
| none | 62.9 | 62.8 | 62.4 | 62.5 | 62.7 |
| 1 | 78.7 (.89) | 72.3 (.63) | 82.0 (.96) | 79.8 (.91) | 78.2 |
| **2** | 77.0 (.73) | 67.4 (.55) | 88.0 (.90) | 83.4 (.83) | **78.9** |
| 3 | 74.8 (.65) | 54.6 (.39) | 89.6 (.86) | 82.3 (.76) | 75.3 |
| 4 | 75.8 (.64) | 52.1 (.34) | 90.9 (.82) | 83.1 (.73) | 75.5 |
| 6 | 62.0 (.48) | 40.3 (.24) | 86.4 (.75) | 73.0 (.60) | 65.4 |

- **n_max 2 is the best fixed depth**: +34% greedy, +26% sampled. Unsloth's model card suggests 6, which is the worst MTP setting here and slower than no MTP on prose.
- The optimum depends on the text: prose wants 1, extraction wants 3-4. One fixed number cannot serve both.
- `-bs` on top of n_max 2: **89.3 greedy / 79.4 sampled**, the best measured config.

### 3.2 `--spec-draft-p-min` is broken on CUDA here, and why

| config | greedy mean | sampled mean |
|---|---|---|
| n_max 3, p_min 0.5 | 82.6 | 71.9 |
| n_max 4, p_min 0.6 | 81.2 | 51.7 |
| n_max 4, p_min 0.8 | 71.6 | 47.6 |
| n_max 6, p_min 0.75 | 55.2 | 57.2 |

p_min raises acceptance (up to 0.9-1.0) and still makes things slower, down to 25 t/s on one prompt. nsys on n_max 4 / p_min 0.6 shows **219k eager `cudaLaunchKernel` calls against 13k at fixed n_max 2, and 59 graph instantiations against 3**. Most steps ran without CUDA graphs.

Cause: the CUDA graph cache is keyed by `cgraph->nodes[0]` (`ggml/src/ggml-cuda/ggml-cuda.cu:2816`). A verify batch of a different length produces the same key, so the cache entry is overwritten. The warmup then needs two calls in a row with no property change, and while the length keeps changing it never completes, so every step launches ~600 kernels one by one (~6.7 us each under WDDM). Anything that varies the draft length hits this: p_min, adaptive depth (upstream PR #27210), early stops at EOS.

### 3.3 Where an MTP cycle goes (nsys, n_max 2, Qwen sampling, prose)

171 verify cycles, 31.7 ms wall each:

| part | ms/cycle |
|---|---|
| verify, trunk (3 tokens; mat-vec cost barely grows from 1 to 3 columns) | ~15.5 |
| verify, LM head (3 rows) | 3.1 |
| **draft, LM head, 2 x full 248k vocab** | **5.5** |
| draft, MTP block and small ops | ~1.2 |
| GDN with rollback snapshots (33.6 us vs 14.6 at 1 token) + `concat_non_cont` (41 us x 24) | ~1.8 |
| **GPU idle, host round trips** | **~4.7** |

**The LM head is 32% of all GPU time under MTP.** The draft half of it and the host gaps together are about a third of the cycle.

---

## 4. Recommended configuration today (no code)

```
llama-server -m Qwen3.5-4B-Q4_K_S.gguf -ngl 99 -fa on -c 32768 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 -bs \
  --spec-draft-vocab-dynamic 16384 --spec-draft-vocab-rank 128
```

- n_max 4 needs the sm_86 MMQ table (5.8); on a build without it, use n_max 2
- the last line needs 5.2 (commit 57cd31ea8); without it, drop the line
- no `--spec-draft-p-min`: even with 5.1 it does not beat fixed n_max 2 (see 5.1)
- `--spec-draft-n-max 1` if the work is mostly free prose
- leave `--spec-draft-backend-sampling` at its default (on)
- the 35B server and this one cannot share the 6 GB card; one at a time

---

## 5. Work items, ranked by expected gain per effort

### 5.1 CUDA graph cache per shape (small, generic, do first)

Key the graph cache by shape as well as by `nodes[0]` (for example `nodes[0]` plus `n_nodes` plus the ne of the first input), so that verify lengths 1..n_max+1 each keep their own instantiated graph. The cache already has a map and an eviction sweep (`common.cuh:1520`). Expected: p_min and variable-depth drafts stop collapsing, and EOS / short-draft steps stop costing eager launches. It also unblocks 5.4. Check upstream first. A search found no PR for this, but older graph PRs (#7302, #26802) dealt with related disable logic.

Measure: re-run the 3.2 table. Done when p_min configs never fall below fixed n_max 2 at equal acceptance.

**Result (2026-10-20, commit 62f564cc5, merged into Ternary):** the key is now a hash of `nodes[0]` and its `ne[]` (`ggml_cuda_graph_get_key`, key type `const void *` -> `uint64_t`, eviction unchanged). In Qwen3.5 the first CUDA node is the layer-0 / MTP norm, whose `ne[1]` is n_tokens, so each verify and catch-up length gets its own entry. A collision only shares an entry as before; `update_required()` still compares every node. Upstream: no merged fix. #28666 made the same change and was closed without comment; #28549 (separate arenas for MTP no-output batches) is already in our tree.

Same-day A/B, only `ggml-cuda.dll` differs, sweeps run one after the other. This was measured on a branch cut from master-era commit 2f807495f, not on Ternary; the combined sweep in 5.7 re-checks it on Ternary. The laptop ran ~7% below the 3.1 tables that day (nospec 60 vs 65):

| config | greedy before | greedy after | sampled before | sampled after |
|---|---|---|---|---|
| none | 59.3 | 60.4 | 56.8 | 58.6 |
| n_max 2 | 83.3 | 85.6 | 74.8 | 76.4 |
| n3 p0.5 | 79.1 | 82.2 | 70.3 | 73.4 |
| n4 p0.6 | 78.8 | 83.0 | 68.2 | 72.6 |
| n6 p0.75 | 74.6 | 80.1 | 67.6 | 71.7 |
| n3 p0.2 / p0.35, n4 p0.35 (after only) | | 85.4 / 82.6 / 83.3 | | 75.3 / 73.2 / 73.3 |

nsys (code + prose, greedy, n4 p0.6): eager `cudaLaunchKernel` 343k -> **58k**, instantiations 59 -> 11, captures 114 -> 46 (fixed n_max 2: 18k, 5, 14). The remaining launches are one eager run plus one capture (~600 kernels each) per recapture, most likely each time n_kv crosses a 256 step. p_min has ~11 shapes against 5, so it pays this more often (~0.4 s over two 400-token replies).

So the thrash is gone, but **p_min still does not beat fixed n_max 2**, and the reason is no longer CUDA graphs. p_min configs do get more tokens per verify cycle (greedy 2.94 for n4 p0.6 vs 2.60), yet each cycle costs more. The explanation, from the code and not measured: the token that p_min rejects has already cost a full draft decode (MTP block + head + host round trip, ~4 ms with the full head), and fixed n_max never wastes that step. Cheaper draft steps (5.2) are what would make p_min / adaptive depth (5.4) pay off. The fix still gives +2-3% at n_max 2 and +4-7% for every variable-length config.

### 5.2 Trimmed draft vocab for the MTP head (done 2026-10-20: +13-15% greedy, +12-14% sampled)

Ported upstream PR #30380 ("Add trimmed draft vocab support", open) onto Ternary, commit 57cd31ea8. The draft head scores the lowest 32,768 ids exactly, plus K more per draft token picked by a low-rank scorer (top singular vectors of the head, built at load) and read through MUL_MAT_ID. End-of-generation ids are always picked. Only the draft context uses it, so the target keeps its full head and still verifies every token. It hooks `build_lora_mm` by head tensor, so qwen35 and qwen35moe MTP need no model code.

- flags: `--spec-draft-vocab-dynamic K` (0 = full head, the default) and `--spec-draft-vocab-rank R` (fork addition, 0 = 512, the PR's fixed value)
- port: ~450 lines in 11 files. The PR's test file was left out. The CPU MUL_MAT_ID row layout change was also applied to the fork's MUL_MAT_ID_COLD op (expert cache), which shares that code and workspace size. test-backend-ops MUL_MAT_ID / TOP_K / SET_ROWS / FILL / CONCAT pass on CUDA. **The COLD op has no test case and has not been run yet (check on the 35B).**
- cost: ~0.7 s of GPU work at server start to build the scorer; +122 MiB VRAM at rank 512, ~30 MiB at rank 128
- the static frequency subset through `d2t` was not done: it needs a per-model list file and a second copy of the rows, and #30380's author measured the same speed from it

**Greedy output is identical with the trimmed head on and off**, all 4 prompts at n_max 2, 3, 4 (rank 512) and n_max 2 (rank 128). MTP itself differs from no-MTP on prose and extract with or without it (batched verify numerics, not new).

Same-run sweep, `-bs` on all, mean of 4 prompts, t/s. Acceptance is unchanged per prompt (at most -0.02 on reason at n_max 3-4). The laptop ran ~4-7% under the 3.1 tables that day, so compare within a row:

| n_max | greedy full | greedy K 16384 | sampled full | sampled K 16384 |
|---|---|---|---|---|
| none | 63.7 | | 63.1 | |
| 2 | 88.0 | **101.1 (+15%)** | 74.1 | **82.9 (+12%)** |
| 3 | 85.5 | 99.4 (+16%) | 70.7 | 81.5 (+15%) |
| 4 | 82.1 | 99.2 (+21%) | 71.9 | 84.1 (+17%) |

Second run, rank 128 (greedy / sampled): n_max 2 full 91.2 / 79.9, K 16384 rank 512 103.5 / 89.5, **K 16384 rank 128 104.5 / 90.8 (+15% / +14%)**, n_max 3 rank 128 103.2 / 86.6, n_max 4 rank 128 102.5 / 89.7. Rank 64-512 and K 8192-32768 at n_max 2 all land within 99.6-103.6 greedy, 83.0-85.1 sampled, with no acceptance change; K 32768 is the slowest.

- **Best depth stays n_max 2** on the mean. With the cheaper draft, n_max 3-4 now win on code, extract and reason (extract 117.9 at n_max 4 vs 112.5 at 2), but prose (acceptance 0.35-0.48) pulls the mean down. That makes 5.4 (adaptive depth) worth more than before.

nsys (n_max 2, Qwen sampling, prose, 378 draft tokens, same drafts both ways): **draft head 2.71 -> 1.06 ms per draft token**, or 5.4 -> 2.1 ms per verify cycle. The 1.06 ms is 0.35 ms for the 32k base rows (Q6_K), 0.15 ms for the 16k picked rows (MUL_MAT_ID), 0.46 ms for the rank-512 Q8_0 scorer and ~0.10 ms for top_k, fill, concat, set_rows and the basis product. Rank 128 should cut the scorer to ~0.12 ms (estimated, not profiled), so ~0.7 ms per draft token. What is left is mostly the 32k base rows. A smaller base (16k) would be the next step, but the PR fixes it at 32,768.

### 5.3 Host gaps (~4.7 ms per cycle, ~1.7 ms per token without MTP)

- count and remove per-token `cudaStreamSynchronize` calls (27 per token now); upstream #30196 ("sampling: read context outputs with one synchronization") is the same idea for sampling
- chain the draft steps on the GPU: with backend sampling the draft argmax is already on the device, so draft step k+1 can take its token from step k without a host read. The gemma4 path in `common_speculative_impl_draft_mtp` already runs all heads in one graph (`is_mem_shared`)
- `-bs` on the target is a stopgap: it costs 1.6 t/s greedy and wins under real sampling

Expected: up to +10-15% if most of the 4.7 ms goes. Needs a closer nsys look (osrt + nvtx) to name each gap before choosing.

### 5.4 Adaptive draft depth (after 5.1)

Both prerequisites are in (5.1 and 5.2), and 5.7 measured what is on the table. With the trimmed head the mean is flat across n_max 2-4 (101.6 / 101.2 / 101.1 greedy), but the prompts pull apart: prose falls 86 -> 67 from depth 2 to 4 while extraction rises 109 -> 116 and code 105 -> 109. **Picking the best depth per prompt is worth ~107 greedy / ~92 sampled, about +5% over any fixed depth.** Text changes kind inside one reply (prose around a code block), so tracking within a reply could add a little more. This is a ~5% item, not a 15% one.

**Confidence is the wrong signal on this model.** p_min 0.1 never fired (identical acceptance to no p_min), and at n_max 6 / p_min 0.15 prose still drafted at 28% acceptance: the MTP head is confident when it is wrong. So the controller should steer on measured acceptance and measured cost, not on draft probability. That rules out simply re-pointing the fork's `--spec-adaptive-width` (it hands back a p_min) without changing what it outputs.

**The plan: a depth controller in `server_slot::get_n_draft_max()`**, the place `PLAN-adaptive-draft-width.md` section 4 already chose, using the argmax rule from that plan with a dense cost model in place of the cold-expert term:

```
W* = argmax over W in [1, W_cap] of  G(W) / T(W)
G(W) = 1 + sum_{k=1..W} prod_{j<=k} a_j        expected tokens per cycle
T(W) = T_v(1 + W) + W * T_d                    verify of 1+W tokens + W draft steps
```

- `a_j`: per-slot decayed acceptance of draft position j given position j-1 was accepted (the server already counts per-position accepts, `n_acc_tokens_per_pos`). Half-life about 8-16 cycles, so it follows a switch from prose to code inside one reply. The 35B controller used ~35 samples because it fitted hardware cost; acceptance moves faster than hardware.
- `T_v(m)` and `T_d`: fitted from timed steps. Both are nearly constants on a dense, fully offloaded model (verify cost barely grows from 1 to 5 tokens, section 3.3), so a long-window fit is fine and can persist across requests.
- Guards carried over from the 35B run (its section 4.2 and 7.1): move W by at most one step per decision, with a margin; probe W+1 every ~16 cycles, since a_{W+1} is never observed otherwise; seed acceptance from the previous request (its warm-up cost it the fast first 100 tokens); `W_cap` 4 (5.5 notes the GDN snapshot cost grows with depth).
- flag: reuse `--spec-adaptive-width`. With no hot store it switches to this mode instead of disabling itself, and keeps the p_min mode when the hot store is on. No new flag.

**Comparison arms**, all with the trimmed head and `-bs`, 3+ repeats with rotated order (the 35B run saw 12% spread between repeats of one config): fixed n_max 2, fixed n_max 4, the controller, and upstream #27210 (`draft-mtp-adaptive`, credit-bucket controller: no cost model, reacts over tens of cycles) as a cheap alternative. Add a long mixed prompt (prose with code blocks) to the four existing ones, since mixed text is the case the controller exists for.

Done when the controller ties the best fixed depth on every prompt within noise and beats both fixed depths on the mean and on the mixed prompt. If it only ties n_max 2 on the mean, ship n_max 2 and stop.

**Result (2026-10-20, uncommitted in `tools/server/server-context.cpp`, `server_spec_depth`): built, works, does not beat the best fixed depth. Leave it opt-in.**

- `--spec-adaptive-width` without the hot store now runs the depth controller (log line `adapting the draft depth in [1, N]`; per request `adaptive depth = mean drafted, now W | T(d) a_d ...`).
- Two measurement traps found and fixed on the way: the first cycles of a request also pay for the draft context catching up on the prompt, and the first two cycles at a new verify width also capture its CUDA graph. Both made deep depths look 10-15 ms slower than they are and stuck to rarely visited depths. Now skipped. Drops go straight to the best shallower depth; climbs are one step; probes alternate up and down every 16 cycles.
- On the full-MMVQ build the cost of depth was 0.6 / 8 / 2 ms for depth 1->2 / 2->3 / 3->4, so a one-step climber never got past 2: it tied fixed 2 (85.1 vs 85.0 sampled, 96.1 vs 98.5 greedy).
- On the MMQ build (5.8), mean of 2 passes, rotated order, 5 prompts:

| config | greedy | sampled |
|---|---|---|
| fixed 2 | 102.6 | 87.5 |
| fixed 3 | 113.7 | 93.8 |
| **fixed 4** | **118.4** | **95.6** |
| fixed 5 (1 pass) | 120.2 | 96.9 |
| fixed 6 (1 pass) | 119.8 | 94.0 |
| adaptive, cap 4 | 114.2 | 94.0 |
| adaptive, cap 6 | 107.3 | 94.9 |

With MMQ the verify cost is flat from 3 to 8 tokens, so a deeper draft only costs its draft step (~1.1 ms). The per-prompt best fixed depth is then worth only ~120.4 greedy / ~98.0 sampled, +2% over fixed 4, and the controller's probes and wrong moves (it drops to depth 1 on prose) cost more than that. **Ship fixed n_max 4.** The controller is worth revisiting only if draft steps get expensive again (bigger head, no trimmed vocab) or on the server once its kernel table is known.

### 5.5 Kernels (moderate gain, hardware specific)

- **Q6_K head mat-vec at 71% of bandwidth** against ~80% for Q4_K. It is the single biggest kernel (2.5-3.1 ms). Tune rows per block / warps for sm_86 and sm_75 with `test-backend-ops perf -o MUL_MAT` on the exact shape (k=2560, m=248320, n=1..4).
- `concat_non_cont` in the verify graph: 41 us x 24 per cycle, about 1 ms. The conv-state concat runs on a non-contiguous layout; make it contiguous or fold it into `ssm_conv`.
- GDN with snapshots costs 2.3x the 1-token kernel. Each verified token writes a full 2 MiB state per layer (`gated_delta_net.cu:174`). Fine at n_max 2; it grows with depth. Replaying the accepted prefix from s0 instead of snapshotting every position is the alternative if depth goes up after 5.2.
- GDN `cols_per_warp` is tuned only for GB10 (`gated_delta_net.cu:47`). Worth a quick sweep on sm_75/sm_86.
- Small-op fusion (rms_norm + quantize_q8_1, add + rms_norm): ~2.2 ms of tiny kernels per token. Check what upstream fuses already before writing new ones.

### 5.6 Optional: a smaller head type

Re-quantizing only `token_embd` (the tied head) from Q6_K to Q4_K or Q5_K with `llama-quantize --token-embedding-type` cuts 497 MiB to ~350-410 MiB and saves ~0.5-0.8 ms on every head call (1 target + n drafts per cycle). That changes the file Kyle chose, so measure KLD against the Q6_K head first. Low priority after 5.2.

### 5.7 Combined result on Ternary (2026-10-20, merge 84e1ee584 = 5.1 + 5.2)

`build-cuda` rebuilt from merged Ternary, same sweep, `-bs` on all, trimmed head = `--spec-draft-vocab-dynamic 16384 --spec-draft-vocab-rank 128`. The laptop ran at normal speed (full-head n_max 2 at 88.0, as in 3.1). Cell = t/s (acceptance):

Greedy:

| config | code | prose | extract | reason | **mean** |
|---|---|---|---|---|---|
| none | 62.8 | 63.5 | 63.6 | 63.5 | 63.4 |
| n_max 2, full head | 91.1 (.86) | 75.2 (.62) | 94.5 (.92) | 91.3 (.87) | 88.0 |
| **n_max 2, trimmed** | 104.7 (.85) | 86.4 (.62) | 109.3 (.92) | 105.9 (.87) | **101.6** |
| n_max 3, trimmed | 106.1 (.78) | 76.8 (.48) | 113.9 (.88) | 108.0 (.81) | 101.2 |
| n_max 4, trimmed | 108.5 (.73) | 67.3 (.35) | 116.1 (.80) | 112.6 (.76) | 101.1 |
| n_max 4, trimmed, p_min 0.1 | 108.6 (.73) | 67.3 (.35) | 116.0 (.80) | 112.7 (.76) | 101.2 |
| n_max 4, trimmed, p_min 0.2 | 106.8 (.72) | 71.7 (.41) | 120.7 (.85) | 112.6 (.77) | 102.9 |
| n_max 6, trimmed, p_min 0.15 | 93.5 (.57) | 56.6 (.28) | 112.7 (.73) | 101.4 (.63) | 91.0 |

Qwen sampling:

| config | code | prose | extract | reason | **mean** |
|---|---|---|---|---|---|
| none | 63.2 | 63.0 | 62.7 | 62.7 | 62.9 |
| n_max 2, full head | 75.1 (.73) | 65.8 (.55) | 84.9 (.90) | 81.3 (.83) | 76.8 |
| n_max 2, trimmed | 86.0 (.73) | 74.8 (.55) | 96.0 (.90) | 92.2 (.83) | 87.3 |
| n_max 3, trimmed | 85.5 (.65) | 64.1 (.39) | 101.3 (.86) | 92.6 (.75) | 85.9 |
| **n_max 4, trimmed** | 90.3 (.64) | 61.4 (.34) | 105.5 (.82) | 96.8 (.72) | **88.5** |
| n_max 4, trimmed, p_min 0.1 | 89.9 (.64) | 61.1 (.34) | 105.7 (.82) | 96.7 (.72) | 88.4 |
| n_max 4, trimmed, p_min 0.2 | 83.8 (.58) | 55.5 (.29) | 105.3 (.83) | 95.9 (.71) | 85.1 |
| n_max 6, trimmed, p_min 0.15 | 74.3 (.47) | 49.5 (.24) | 104.4 (.75) | 87.2 (.58) | 78.8 |

- the two fixes stack: **+60% greedy, +41% sampled against no MTP**, and +15% over the full-head best
- no p_min config collapses any more (3.2 had 25-55 t/s), so 5.1 holds on Ternary
- n_max 2 vs 4 sampled (87.3 vs 88.5) is inside the noise of one repeat. n_max 2 stays the default because it is the robust one on prose.
- the MUL_MAT_ID_COLD change from 5.2 checked on the 35B (`-cmoe -ehs 24`, tier engaged): greedy output without MTP identical to a pre-port `ggml-cpu.dll` on all 4 prompts, speed equal over 4 interleaved runs (whichever runs first is fastest: the CPU heats up). With MTP the text varies between runs of the same build, so that is MTP, not the op.

### 5.8 MMQ from 3-4 columns on sm_86 (2026-10-20, uncommitted in `ggml/src/ggml-cuda/mmvq.cu`)

nsys at depth 2/3/4 showed the cost of depth is GPU work, not host gaps: per cycle, busy time grew ~4.5 ms per extra draft position, mostly in the verify mat-vecs. MMVQ redoes the k-quant decode for every column, and under the 65 W cap (SM ~1100-1400 MHz) that ALU work is what runs out. `test-backend-ops perf`, Q4_K 4096 x 14336:

| columns | MMVQ us | MMQ us |
|---|---|---|
| 1 | 150 | - |
| 2 | 165 | 215 |
| 3 | 216 | 216 |
| 4 | 266 | 216 |
| 5 | 279 | 217 |
| 8 | 405 | 218 |

Q5_K and Q6_K cross over at 3 columns (Q6_K at 5: 388 -> 290 us). Upstream's table has entries for Ada, Blackwell, GB10, Orin and AMD; sm_86 fell through to "MMVQ up to 8". Added `cc == 860`: Q4_K mvq up to 3, Q5_K / Q6_K up to 2, rest unchanged. 106/106 MUL_MAT tests pass. Q4_K up to 2 was tried on the model and gave nothing. Also added `GGML_CUDA_MMVQ_MAX_COLS=N` to force the threshold for all types (tuning knob; `=8` restores the old choice), which is how to find the 1660 Ti's crossovers on the server.

Effect: fixed n_max 4 went 101.1 -> **118.4 greedy**, 88.5 -> **95.6 sampled**; best depth moves from 2 to 4. The 35B (MTP n_max 2, `-cmoe -ehs 24`) is unchanged: 28.2 / 30.8 against 28.5 / 30.7 t/s with `=8`, interleaved.

### 5.9 Where a cycle goes now (nsys, n_max 4, MMQ, code prompt, 141 t/s under the profiler)

7.07 ms per generated token (8.57 before 5.8), ~3.9 tokens per cycle, ~27.6 ms per cycle:

| part | ms/cycle | share |
|---|---|---|
| verify mat-muls, MMQ, trunk + head: 2.42 GiB read once | 14.2 | 51% |
| 4 draft steps (MTP block + trimmed head, ~1.07 ms each) | 4.3 | 16% |
| small ops: GDN 1.1, concat_non_cont 0.8, F32 mul_mat_f 0.8, norms, copies, rope... | 4.7 | 17% |
| GPU idle: host round trips between the sequential draft steps | 3.8 | 14% |

While decoding: throttle reason `0x4` (SW power cap) the whole time, 61-65 W, SM 1280-1550 MHz of 2100, memory clock at max, memory controller busy ~60%. The verify reads run at ~183 GB/s, 64% of the 288 GB/s peak.

### 5.10 Host gaps and small kernels, round one (2026-10-20)

All A/Bs below are interleaved new/old/old/new (ABBA): the laptop slows ~1 t/s per run as it heats, and plain alternation credited that drift to whichever build ran first. n_max 4, trimmed head, `-bs`, 5 prompts, greedy and sampled.

| change | where | greedy | sampled | output |
|---|---|---|---|---|
| user inputs copied async on the split backend, one sync per split instead of two per input | `ggml-backend.cpp` `compute_splits` | ~+1% | ~+0.5% | identical |
| `llama_context::synchronize()` skips the backend sync when nothing was queued since the last one (the getters call it ~9 times per draft step) | `llama-context.cpp` | (in the row above) | | identical |
| thin F32 matrices (<= 512 rows, e.g. `ssm_alpha`/`ssm_beta` 2560x32) keep `mul_mat_vec_f` up to 8 columns instead of 3 | `mmvf.cu` | +1.6% | +0.7% | changes (different kernel), deterministic |
| the existing tiled `concat_dim0_transpose_u32` (conv state + transposed inputs) enabled on every CUDA device, not only GB10 | `concat.cu` | **+2.8%** | **+2.2%** | identical |
| sm_86 MMVQ/MMQ table extended to every type `phase0/mmvq-table.sh` measures (Q3_K 1; Q2_K, Q5_K, Q6_K 2; Q4_K, Q5_1, Q8_0, MXFP4 3; IQ2_XS, IQ4_XS 4; Q4_0, Q4_1, Q5_0, IQ4_NL, IQ3_S 5) | `mmvq.cu` | (4B unchanged: only Q4_K/Q5_K/Q6_K matter there) | | 407/407 MUL_MAT tests |

After these: **~122.8 greedy / ~98.2 sampled** at n_max 4 (was 118.4 / 95.6).

On the 35B-A3B (`-cmoe -ehs 24`, greedy, code + reason, ABBA against the build with only the first sm_86 table): n_max 2 +3.1% / -3.5%, n_max 3 +2.6% / +1.3%. That is inside its run-to-run spread (up to 15% between runs of one build), so no measurable change. Its experts (17 GB of Q4_K) go through MUL_MAT_ID, which the table does not touch; its dense part is 2 GB of Q8_0, which crosses at 4 columns, so only n_max 3+ can use it. In the same runs n_max 3 beat n_max 2 by ~10% on both prompts; worth a proper sweep in the usual 35B configuration.

Where the host time goes, from the server's own `DEBUG_TIMINGS` (temporarily enabled) on the code prompt, 139 t/s, 3.86 tokens per cycle:

| phase | ms per cycle | GPU part (nsys) | host part |
|---|---|---|---|
| pre-decode: the 4 draft steps | 6.25 | ~4.3 | ~2.0, ~0.4-0.5 ms per draft step |
| decode: verify + MTP catch-up | 21.4 | ~19 | ~2 |
| post-decode: sample and accept | 0.07 | | |

- the draft context reuses its graph for 3 of 5 decodes per cycle: the 5-token catch-up and the first 1-token draft step rebuild each cycle because the context keeps one previous graph. A rebuild of the MTP graph costs only ~0.11 ms (`LLAMA_GRAPH_REUSE_DISABLE=1` A/B), so a two-graph cache would buy ~1%. A rebuild of the verify graph costs ~1.1 ms, and that one is already reused every cycle.
- the rest of the ~0.4 ms per draft step is the round trip itself: wait for the GPU, read the sampled token and the 10 KB hidden row, build the next batch, upload it, launch. It is paid four times per cycle, and WDDM makes each one dearer. Only drafting on the device removes it (next item).

Still open, largest first:
1. **Device-side draft chain.** Build the n_max MTP steps as one graph: argmax on the device, `get_rows` of the drafted token's embedding, the step's `h_nextn` fed straight into the next step, KV written at pos0 + i. Removes ~1.5-2 ms of host time per cycle (~6-8%), more on WDDM than on Linux. The gemma4 path (`is_mem_shared`) already runs several heads in one graph; the Qwen MTP head writes its own KV per step, which is what makes this harder.
2. **MMQ at 5 columns**: 20-72 row tiles on 30 SMs, run as stream-k plus a fixup (~3 us per mat-mul, ~0.6 ms per cycle) at 64% of bandwidth. A smaller row tile for tiny batches would let plain tiling fill the SMs without the fixup. ~2-4%.
3. GDN with rollback snapshots (~1.1 ms per cycle) and the snapshot `cpy_scalar`s.

### 5.12 Device-side MTP draft chain (design approved by Kyle 2026-10-20, not started in code)

One decode of an n-token ubatch (pos0..pos0+n-1) on the draft context through a chain graph that unrolls the n MTP steps. Facts checked in the code:
- draft sampler is top-k 10 and takes data[0] (`common/speculative.cpp` ~1430), so in-graph `ggml_argmax` is the same draft up to ties. Use the chain only when p_min == 0.
- `build_sampling()` returns early when `res->t_logits` is null (`llama-graph.cpp` ~3957): set it null in the chain graph and expose a new `res->t_chain` (I32 [n]); `decode()` (`llama-context.cpp` ~2654-2766) copies outputs by result tensor, so add an async copy of `t_chain` there and a getter in `llama-ext.h`. Mark no batch outputs.
- per step i: K/V store through `cpy_k`/`cpy_v` with a 1-element view of `k_idxs`/`v_idxs` (I64 [n_tokens], `llama-kv-cache.cpp` ~2052); read through the full `get_k`/`get_v` view; mask = 1-row view of the causal kq_mask (flash attn only asserts contiguous, `ggml.c` ~5563). RoPE is `ggml_rope_multi` (4 sections): positions for token i are strided by n_tokens in `inp_pos`, so take a strided view + `ggml_cont` to [4].
- step 0 input: batch token 0 + `pending_h`; step i > 0: `get_rows(tok_embd, argmax_{i-1})` and `h_nextn` of step i-1. The trimmed head hooks `build_lora_mm` per call, so it works per step.
- add a `mtp_chain` field to `llm_graph_params` and to `allow_reuse`, so a chain graph never reuses the catch-up graph.
- host: in `common_speculative_impl_draft_mtp::draft()`, a fast path that builds the n-token batch, toggles the chain flag on the draft context, decodes once, reads n ids, truncates at EOS. The flag goes in `arg.cpp`, Qwen3.5/3.6 only (`src/models/qwen35.cpp` graph_mtp ~546; qwen35moe has its own copy).

Check: greedy text identical to the loop on the 5 sweep prompts; ABBA t/s; nsys GPU idle per cycle (now ~3.8 ms). Expected +5-7% at n_max 4 on the laptop, nothing at n_max 1.

**Result (2026-10-20, uncommitted): built, exact, +1%.** `--spec-draft-chain` (Qwen3.5 dense, flash attention, one sequence, p_min 0; anything else falls back to the loop).
- greedy text identical to the loop on all 5 prompts in all 4 ABBA runs; acceptance identical, so the drafts are the same tokens
- ABBA, n_max 4, trimmed head, `-bs`: chain 125.6 / 122.1, loop 123.0 / 122.2 greedy; sampled 98.9 / 94.9 against 96.1 / 95.8. **About +1.0% both ways.**
- first build: each step's views of the host inputs (a 52 B position slice, two 8 B KV indices, a 1 KB mask row) went to the GPU as separate split inputs through the scheduler's slow cross-backend path, 3 syncs each (+48 syncs per cycle). Fixed in `ggml-backend.cpp`: a view of a user input now takes the batched user-input path. That fix is general, not chain specific.
- nsys, code prompt: graph launches 626 -> 313, syncs 4800 -> 2928, GPU idle 280 -> 213 ms. But GPU busy rose ~0.15 ms per token, mostly four `argmax` over the full 248k scattered logits (+0.2 ms per cycle). Net ~0.5-0.7 ms per ~28 ms cycle.
- why less than estimated: the four draft round trips cost less than the 0.4 ms per step guessed from the phase timers, and the host work that remains (~2.8 ms idle per cycle) is around the verify decode and the catch-up (51 KB of hidden rows out and back, input upload, launch), which the chain does not touch.
- cheap follow-ups: argmax over the trimmed subset before the scatter (~0.2 ms per cycle); folding the catch-up into the chain decode (one decode and the 51 KB round trip less per cycle, ~1-2%).

**Follow-ups done (2026-10-20, uncommitted): chain now +2.0%.**
- `build_lora_mm_argmax` (`llama-graph.cpp`): with the reduced head, argmax over [lowest 32k logits, picked logits] and map the index back through [0..n_base) ++ picked ids, instead of argmax over the scattered 248k row. Falls back to the plain argmax whenever the reduced head does not apply.
- catch-up folded into the chain decode: `process()` keeps a small one-sequence generation batch (tokens, positions, the h rows shifted by one) instead of decoding it, `accept()` trims it to the accepted rows, and the next chain draft puts those rows in front: `llama_set_mtp_chain(ctx, n)` now means "chain the last n rows". The kept rows only write K/V; their attention, FFN and head are pruned from the graph, so they cost less than the old full catch-up of all rows. Kept rows are flushed with a plain catch-up whenever anything else comes first (a prompt batch, a no-draft step, a replay, the step loop), without the rows the new batch writes again.
- ABBA, n_max 4, trimmed head, `-bs`: chain 128.0 / 126.3, loop 124.8 / 124.2 greedy; sampled 101.9 / 99.5 against 100.0 / 97.6. **+2.1% greedy, +1.9% sampled.** Text identical to the loop on all 10 prompts x sampling in all 4 runs. Acceptance moves by +-0.01 (the catch-up K/V now comes from a batch of the accepted rows only, a different mat-mul shape).
- nsys, code prompt: graph launches 626 (loop) -> 313 (chain) -> 185 (chain + fold), GPU busy back to 5.92 ms per token.
- with `--spec-draft-p-min` the chain falls back to the loop after flushing the kept rows. There 2 of 5 greedy texts differ from the plain loop: the draft probabilities move by rounding, so p_min cuts at other depths, the verify widths change and the target takes other kernels. Acceptance within +-0.04. Same kind of difference as MTP against no MTP; the chain itself is exact when p_min is 0.

### 5.11 GGML_CUDA_MMVQ_MAX_COLS and phase0/mmvq-table.sh

The knob now takes per-type values: `GGML_CUDA_MMVQ_MAX_COLS=q4_K=3,q6_K=2,default=8` (a bare `N` still sets every type). `phase0/mmvq-table.sh` times every type at 1-8 columns with the knob at 8 (MMVQ) and at 0 (MMQ or whatever path is next), picks each type's crossover, and prints the export line plus a C++ table entry. On Linux: `BIN_DIR=./build/bin ./phase0/mmvq-table.sh`. For the 1660 Ti the export route is the safe one: compute capability 7.5 is shared with the RTX 20 series, which has tensor cores, so a baked-in `cc == 750` entry would also change those cards.

---

## 6. The home server (GTX 1660 Ti, Turing sm_75, no tensor cores, 16 GB RAM)

Known from `PLAN-qwen36-27b.md`: 5748 MiB visible, Turing without tensor cores, `GGML_CUDA_FORCE_MMQ=ON` gave +135% prompt processing on the 27B. Same 192-bit / 12 Gbps memory class, so ~288 GB/s like the laptop. Expect similar decode, but different power and clock behaviour, and Linux has no WDDM, so host gaps (5.3) may be smaller there.

To run there, in order:

1. `nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv`, and the same with `clocks.sm,clocks.mem,power.draw` while decoding. Is it power capped too?
2. `llama-bench -m <4B> -ngl 99 -fa 1 -p 512,2048 -n 256 -r 3` on the current build and on the `FORCE_MMQ` build.
3. `python3 phase0/qwen35-4b-mtp-sweep.py --server ./build/bin/llama-server --model <4B> --out sweep.jsonl --configs "nospec=;mtp1=--spec-type draft-mtp --spec-draft-n-max 1;mtp2=--spec-type draft-mtp --spec-draft-n-max 2;mtp3=--spec-type draft-mtp --spec-draft-n-max 3;mtp2-bs=--spec-type draft-mtp --spec-draft-n-max 2 -bs"`
4. If `nsys` is installed: `nsys profile -t cuda --cuda-graph-trace=node` around step 3 with one config, for the same breakdown as 3.3.

---

## 7. Order of work

1. ~~Config from section 4~~, now with the trimmed head.
2. Server measurements (section 6), adding the trimmed-head configs from 5.7.
3. ~~5.1 graph cache per shape~~ done, commit 62f564cc5.
4. ~~5.2 trimmed draft vocab~~ done, commit 57cd31ea8. Still open: run the MUL_MAT_ID_COLD change on the 35B with the expert cache on.
5. 5.3 host gaps, guided by a finer nsys trace (~4.7 ms per cycle was the second-largest item; now a bigger share of a shorter cycle).
6. 5.4 adaptive depth (~+5%), 5.5 kernels, 5.6 head type, in whatever order the re-measured profile says.

Measured 2026-10-20 after 5.1 + 5.2: **101.6 t/s greedy, 87-88 sampled**, against 63 without MTP and 88 / 77 with the full head. After 5.8 at n_max 4: **118.4 greedy, 95.6 sampled** (+87% / +52% against no MTP). 5.4 is done and stays opt-in.

What is left, from 5.9, largest first:
1. **Host gaps, 14%** (5.3): four sequential draft steps each wait for the host. Chaining them in one graph on the device (argmax -> get_rows -> next MTP step, the way the gemma4 path runs all heads) removes most of the 3.8 ms. No power cost at all; this is latency.
2. **Verify at 64% of bandwidth**: MMQ at 5 columns on 30 SMs. Its tile is wider than 5 columns (wasted MMA work, which costs power under the cap) and it pays a stream-k fixup. Tuning `mmq_x` / stream-k for tiny batches on sm_86, or an MMVQ that decodes once and reuses across columns, is worth up to ~3 ms of the 14.2.
3. **Small ops, 17%**: `concat_non_cont` (0.8), the F32 mat-muls that switch from `mul_mat_vec_f` to `mul_mat_f` at 4+ columns (0.8), GDN snapshots (1.1), norm/quantize fusion.
4. **Draft steps, 16%**: the fixed 32k base rows of the trimmed head are now most of a draft step; a 16k base or a smaller head type (5.6).
5. 1660 Ti: its own MMVQ/MMQ crossovers via `GGML_CUDA_MMVQ_MAX_COLS` (Turing has no tensor cores, so MMQ there is dp4a and may not win).

## 8. Sources

- unsloth model card: https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF (recommends n_max 6, measured worse here)
- Qwen3.5-4B config: https://huggingface.co/Qwen/Qwen3.5-4B
- trimmed draft vocab: https://github.com/ggml-org/llama.cpp/pull/30380
- adaptive MTP depth: https://github.com/ggml-org/llama.cpp/pull/27210
- MTP slower than baseline on Metal for Qwen3.5-9B (draft overhead > gain): https://github.com/ggml-org/llama.cpp/issues/23752
- FR-Spec, frequency-ranked draft vocab: https://arxiv.org/abs/2502.14856
- MTP vs other speculative types, prefill cost of MTP: https://victor-mtp-on-hf-endpoints.static.hf.space/index.html
- Qwen3.6-27B MTP depth and p-min on DGX Spark: https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298
