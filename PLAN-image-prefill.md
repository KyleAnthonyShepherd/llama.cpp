# PLAN - faster image ingestion on Qwen3.6-35B-A3B

Branch: `expert-prefetch-adaptive-ub` (HEAD `1c9ddcb46`). Written 2026-10-26, nothing built or measured yet (GPU busy).

Workload: many scanned PDF pages, about 1000 vision tokens each (the client caps at 1 MP, 32 px per token), plus runs of small thumbnail or header images. Laptop RTX 3060 6 GB, PCIe 4.0 x8; home server GTX 1660 Ti 6 GB.

## 0. Short answer

> Measured since: section 7. Per-decode fixed cost about 1.45 s is confirmed; the encode churn is about 0.23 s per page, not 0.6 s; the recent flags all help. The table below is the pre-measurement estimate.

The hunch is partly right. The biggest fixed cost per image is older than the recent work: **every image goes to the LLM as its own `llama_decode`**, so every image pays one full pass of expert uploads over PCIe, however few tokens it has. The recent work adds a second fixed cost: **the hot store drops and refits around every encode (about 0.6 s)**. The lazy mmproj itself and the adaptive ubatch switch cost little per image.

Costs per 1000-token page, today (M = measured in an earlier plan, E = estimate):

| step | cost | source |
|---|---|---|
| vision encode, host weights, lazy compute | 0.9 s | M, PLAN-mmproj-lazy-vram 7.5 (941 tokens) |
| of which: streaming the 861 MiB tower over PCIe | 0.12-0.23 s | M, PLAN-adaptive-mmproj 2 |
| hot store drop + refit around the encode | about 0.6 s | M, PLAN-mmproj-lazy-vram 7.4/7.5 |
| LLM prefill of the image as one 1000-token batch | about 2-2.5 s | E, from the ubatch sweep (ub512 230 tok/s, ub2048 630 tok/s) |
| short text between images (`<|vision_end|>` ... `<|vision_start|>`) | about 0.05 s each | E, its own decode, a few tokens |
| **total** | **about 3.5-4 s per page (250-290 tok/s)** | E |

The same LLM prefill packed into a 4096-token batch runs at about 1000 tok/s (M, memory note on `-epf`/`-ubg`), so about 1 s per page. Thumbnails are worse off today: a 200-token image still pays a full expert upload pass (about 1.6 s) plus the 0.6 s hot-store churn, so roughly 2.5 s each.

Target after the plan: about 1.5-2 s per page (2x), and 5-10x for runs of thumbnails. Phase 0 must confirm these numbers before any code is written.

## 1. How the code handles an image today

- `tools/server/server-context.cpp`, prompt loop (around line 5153): at the start of each slot batch, every media chunk at the current position is processed by `process_mtmd_chunk()`, then text tokens are added until the next media chunk, and the loop **breaks** (line 5207). So a prompt `[text][img][3 tokens][img][3 tokens][img]...` becomes one decode per image and one per text gap.
- `process_mtmd_chunk()` (line 1214): makes room for the encode (`llama_expert_hotstore_free_bytes`, asked on **every** encode; then `llama_kv_spill` if needed), encodes, unspills and calls `llama_expert_hotstore_refit()`, then `try_decode()`.
- `mtmd_batch_add_chunk()` (`tools/mtmd/mtmd.cpp:2058`) would group several images into one encode, but `clip_support_batch()` is false for `qwen3vl` and `batch_max_tokens` is 1024. So there is one encode per image.
- `mtmd_helper_decode_image_chunk()` (`tools/mtmd/mtmd-helper.cpp:118`) decodes one chunk's embeddings in `n_batch` slices. A 1000-token image is one 1000-token `llama_decode`.
- Each LLM ubatch uploads all CPU-resident experts (about 15 GiB, all 256 experts are hit by 1000 x 8 routings). At ub 1000 the upload (about 1.6 s at 9.5 GiB/s) is longer than the kernels, so the batch is upload-bound. Batches under 128 tokens (`op_offload_min_batch` under `-ehs`, `common/arg.cpp:973`) run the experts on the CPU instead.
- Adaptive ubatch (`src/llama-context.cpp:2799`): a batch over `-ubg` switches to the large ubatch, and only an all-outputs batch switches back. Image batches keep the large mode, so there is **one switch per turn**, not one per image. But in large mode the card is fuller, so the encode churn above happens on almost every image.
- Lazy mmproj (`clip.cpp` `compute_acquire/compute_release`): since the lazy-vram work the CUDA backend stays; each encode re-allocates the compute buffers and the pool. That is milliseconds, not a real cost.
- MTP (`common/speculative.cpp:1600`): embedding batches are skipped and the draft state is invalidated. The draft does not touch image batches.
- Vision attention: Qwen3.6's tower has head size 72 (1152 / 16). `ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu:615`) excludes 72 from the MMA (tensor core) kernel on Turing and later, so the encode uses the **tile kernel** (no tensor cores). Encode time grows faster than tokens (0.9 s at 941, 10.5 s at 3997, about N^1.7), which points at attention.

## 2. Phase 0 - measure (needs the GPU, no code)

The logs already have everything: `media encode took ... room X ms, encode Y ms, restore Z ms` (server), `image decoded (batch 1/1) in N ms` (mtmd helper), the `set_n_ubatch` switch lines, and the final `prompt eval time`.

Test request (script in the session scratchpad, one-off, as before): one chat message with 8 page images (about 1000 tokens each, from real scans) and 12 thumbnails (150-250 tokens), each image preceded by a short label (`Page 3:`). Put a random nonce at the start so the prompt cache never hits. Run each config 3 times, single client, `-c 4096` start.

| config | flags on top of `srv35.sh` |
|---|---|
| A (today) | `-b 4096 -ub 4096 -ubg 512 -epf 2 --mmproj-weights-host --mmproj-compute-lazy` |
| B | A + `--mmproj-vram-reserve auto` |
| C | A without `--mmproj-compute-lazy` |
| D | A without `--mmproj-weights-host` |
| E (old) | `-b 2048 -ub 2048`, no `-ubg`, no `-epf` |

Deliverable: a table per config of encode / room+restore / LLM decode per page and per thumbnail, and the whole-prompt time. This answers the hunch with data. Expected: B removes most of the 0.6 s; C and D change little; the LLM decode per image is the largest row in all of them.

Second measurement (5 minutes): the encode time with head size 72 against 80. Quickest way is the experiment patch of section 3.3 behind an env var, A/B on the same image.

## 3. Changes, in order of value per effort

### 3.1 Dropped: `--mmproj-vram-reserve auto`
Kyle's decision (2026-10-26): hot slots are worth more than the encode churn; the mmproj should get out of the way when it is not in use. Not tested. Section 7 shows the churn is about 0.23 s per page and zero for thumbnails, so little is lost.

### 3.2 Encode a run of images ahead, then pack them into large LLM batches (the main change)

Two steps that belong together:

**a. Burst encode.** When the prompt loop reaches a media chunk, look ahead: collect the chunks up to `n_batch` tokens ahead (images plus the short text gaps between them). Make room once, encode every image of the run one after another, keep each embedding in host RAM (`n_embd_inp x n_tokens x 4 B`, about 8 MiB per page without deepstack, 32 MiB if the tower has 3 deepstack layers; check `n_embd_inp` at load), then unspill and refit once. Fixed costs drop from once per image to once per run.

**b. Packed decode.** Decode the run as **one** embedding batch: image rows from the encoder, text rows from the token embedding table. Up to `n_batch` (4096) tokens per decode, so 4 pages or about 20 thumbnails per expert-upload pass instead of 1.

Details to get right:
- **Text rows as embeddings.** Add `llama_model_get_tok_embd_rows(model, tokens, n, out)` next to the existing `llama_model_get_tok_embd()` (`src/llama-model.cpp:3640`): dequantize only the requested rows with the type's `to_float`, the same path `ggml_get_rows` takes, so the values are identical. Pad to `n_embd_inp` with zeros (as `build_inp_embd` does with `ggml_pad`). Refuse (fall back to today's path) when the model has a Hadamard-latent embedding table or a LoRA on `tok_embd`: the embd path in `build_inp_embd` (`src/llama-graph.cpp:2478`) skips both.
- **Positions.** M-RoPE needs 4 values per row. Images use `mtmd_helper_image_get_decoder_pos`; text rows use the 1D layout of `set_position_mrope_1d`. Reuse `decode_embd_batch` from `mtmd-helper.cpp`, extended to fill rows from several chunks.
- **Recurrent layers.** Qwen3.6 is hybrid (gated delta net + attention). One sequence in one batch keeps the token order, so the recurrent state sees the same sequence. Upstream #28166 shows `find_slot` warnings on every M-RoPE image decode for qwen3_5; check the packed batch does not add a new kind.
- **MTP.** The packed batch is an embd batch, so MTP skips it as it skips images today; the few text tokens inside a run join that gap. Text after the last image stays a normal token batch, so the draft is re-primed as now. Measure acceptance on the first reply after an image prompt; upstream #30220 (DFlash) shows draft holes at image positions cost acceptance only right after the image.
- **Bookkeeping.** `slot.prompt.tokens` gets the same `push_back` / `push_back_placeholder` calls as today, in order, so the prompt cache and `server_tokens` do not change. Checkpoints already skip batches with media (`do_checkpoint && !has_mtmd`).
- **Limits.** An image larger than `n_batch` keeps today's path. Respect `n_ctx_limit` and context growth (the run's total is known before the decode, so growth can be asked for once).
- **Gate.** A server flag, `--mtmd-pack` (default off until measured), so the old path stays one switch away.

Why not upstream's way: upstream merged `llama_batch_ext` (ggml-org/llama.cpp#29385, 2026-09-28), which allows mixed token and embedding rows natively. Its server still decodes each image alone (`process_mtmd_chunk` -> `mtmd_helper_decode_image_chunk`), so the packing logic is new either way. This fork does not have `batch_ext`; porting it is a large merge. Text-as-embeddings keeps the change inside the server, mtmd helper and one small llama API. If a later upstream sync brings `batch_ext`, switch the text rows to real token rows.

Expected gain (E): pages from about 2-2.5 s to about 1 s of LLM time each, plus 0.6 s churn removed; thumbnails from about 2.5 s to under 0.4 s each. Bit-exactness check: temp 0, same prompt, packed vs unpacked, with `-ehs 0 -cmoe` (the hot store is non-deterministic). Batch size changes kernel tiling, so expect tiny logit differences, not identical tokens over long replies; compare first-token logits and a 200-token reply.

### 3.3 Vision attention on tensor cores: pad head size 72 to 80

In `clip_graph::build_attn` (`tools/mtmd/clip.cpp:851`), when FA is on and `d_head` is 72 (or 40), pad Q, K and V with zeros to 80 (`ggml_pad`), run `ggml_flash_attn_ext` with the same `kq_scale` (it is passed explicitly as 1/sqrt(72), so the zeros change nothing), and take a view of the first 72 output columns. The MMA kernel has DKQ=80 configs (`fattn-mma-f16.cuh:45`).
- Cost: 11% more attention FLOPs and three pad copies per layer; gain: tensor cores instead of the tile kernel.
- For 1000-token pages, attention is about 1.8 TFLOP of the encode (E: 3844 patches, 27 layers). If it is most of the 0.9 s, the encode could drop to about 0.4-0.5 s. Larger images gain more.
- The 1660 Ti (TU116) has no tensor cores but takes the Turing MMA path; whether that is faster there must be measured on the server. Gate on the measured kernel choice if it is slower.
- Also helps Gemma towers (head size 72); upstream #16837 left "FA sizes for head size 72" as a TODO. A small, upstreamable change.

### 3.4 Smaller items

- **Tower weights during a burst.** With host weights, each encode streams 861 MiB again (0.12-0.23 s). In a burst of N images, upload the weights to a temporary VRAM buffer once and free it after the run. Saves about 0.15 s per extra image, but needs 861 MiB free during the run (more hot-store eviction). Measure after 3.2. A Q8_0 mmproj halves the stream instead, at some risk to OCR accuracy (upstream #29251 reports Qwen3-VL embeddings in llama.cpp are already noisier than in transformers).
- **Embedding cache by image hash.** Keep the last N image embeddings in RAM (8-32 MiB each), keyed by the chunk's bitmap id. An image re-sent after the prefix changed (new system prompt, an edited earlier turn, the same PDF in a new chat) skips the encode and its churn. The LLM prefill still runs. vLLM and LMCache do the same ("encoder cache"). Low effort once 3.2a stores embeddings per chunk.
- **Client side (toolbag `servers/office/render.py`).** Crop scanned page margins before the 1 MP cap (fewer tokens for the same text size). Tile runs of thumbnails into one sheet image when they are only for overview. Do not cut page tokens much below 1000 without an accuracy check on real scans (see #29251: fewer tokens hurt more in llama.cpp than in transformers).

### 3.5 Looked at, not recommended now

- **Overlap the encode with the LLM prefill** (vLLM's encoder disaggregation idea): the encode of image i+1 could use the SMs while the 1000-token LLM batch of image i waits on PCIe. After 3.2 the LLM batches are compute-bound, so little idle time is left, and 6 GB has no room for both working sets. Skip.
- **Visual token pruning** (ET-Prune, PixelPrune, NVIDIA DART on Qwen3-VL): DART reports 25% fewer tokens near lossless but 1-2 points lost on OCR at 50%. Lossy, graph changes in the LLM, risky for OCR. Skip.
- **Batching images in one encode** (`clip_support_batch`): Qwen-VL attends over the whole input, so images would need a block mask; the encode is not the bottleneck once 3.2 and 3.3 are in. Skip.

## 4. Order of work

1. Phase 0 table (section 2), including the 72 vs 80 A/B. Done, section 7; the interleaved A / A_pad rerun is still open.
2. (3.1 dropped.)
3. 3.3 (small, one function) and measure.
4. 3.2a then 3.2b behind `--mtmd-pack`, then the Phase 0 request again, then a soak like the lazy-vram one (mixed text and image turns, context growth) on the laptop, then the home server.
5. 3.4 items by measured value.

## 5. Files

- `tools/server/server-context.cpp`: prompt loop look-ahead, burst encode, packed decode, `--mtmd-pack` gate, logging.
- `tools/mtmd/mtmd-helper.{h,cpp}`: multi-chunk embedding batch builder (text rows + image rows, M-RoPE positions).
- `src/llama-model.cpp`, `include/llama.h`: `llama_model_get_tok_embd_rows()`.
- `tools/mtmd/clip.cpp`: head-size pad in `build_attn`.
- `common/arg.cpp`, `common/common.h`: `--mtmd-pack`.

## 7. Phase 0 results (measured 2026-10-26, laptop)

Setup: unsloth UD-Q4_K_S + mmproj-BF16, `build-cuda` at `5cec762eb` plus the `MTMD_FA_PAD` experiment in `clip.cpp` (off unless the env var is set). Common flags: `--spec-type draft-mtp --spec-draft-n-max 1 --ctx-max 262144 --flash-attn on -ehs -1 -fitt 256 -c 4096 -np 1 --ctx-checkpoints 8 --checkpoint-min-step 128 -lv 4`, mmap load. Config B was not run (3.1 dropped).

Inputs: 8 pages of a real scan (`28409 clc review comments`, 880x1138 px, 1008 tokens each), 12 small images (6 header strips 832x236 and 6 page thumbnails 384x496, 187 tokens each), and 7419 tokens of text as the packed-prefill reference. One request each, a random nonce first so the cache never hits, a short label before each image, 4 output tokens, thinking off. One warm-up request, then 3 reps per config. Driver, inputs and logs: session scratchpad `p0/` (`run_p0.py`, `analyze.py`, `run1/`).

Medians of 3 reps; per-image columns are averages per image:

| config | request | prompt s | tok/s | room ms | encode ms | restore ms | LLM decode ms | text gaps etc. ms |
|---|---|---|---|---|---|---|---|---|
| **A** `-b 4096 -ub 4096 -ubg 512 -epf 2`, host weights, lazy | text 7419 | 7.2 | 1027 | | | | | |
| | 8 pages | 26.9 | 304 | 99 | 1046 | 130 | **1867** | 1778 |
| | 12 small | 23.0 | 103 | 2 | 217 | 0 | **1526** | 2079 |
| **A_pad** A + `MTMD_FA_PAD=1` | text | 7.6 | 980 | | | | | |
| | 8 pages | 25.3 | 323 | 70 | **646** | 101 | 2053 | 1816 |
| | 12 small | 24.8 | 96 | 3 | 229 | 0 | 1623 | 2493 |
| **C** A without `--mmproj-compute-lazy` | text | 9.8 | 756 | | | | | |
| | 8 pages | 29.0 | 282 | 58 | 1058 | 76 | 2116 | 2573 |
| | 12 small | 31.8 | 75 | 283 | 219 | 283 | 1636 | 2693 |
| **D** A without `--mmproj-weights-host` | text | 10.6 | 701 | | | | | |
| | 8 pages | 29.1 | 281 | 22 | 954 | 18 | 2444 | 1614 |
| | 12 small | 22.4 | 106 | 2 | 100 | 0 | 1586 | 2148 |
| **E** old `-b 2048 -ub 2048`, host weights, lazy | text | 12.7 | 584 | | | | | |
| | 8 pages | 30.5 | 268 | 5 | 1059 | 0 | 2511 | 1755 |
| | 12 small | 24.0 | 99 | 3 | 226 | 0 | 1590 | 2182 |

Hot slots planted at load: A 26, A_pad 24, C 26, D 14, E 16. In the large ubatch A holds about 10.

### 7.1 What it says

1. **The hunch is wrong about the recent work, and right that fixed costs per image dominate.** A beats C, D and E on text and pages. Host weights and lazy compute each pay for themselves: without host weights (D) the store plants 14 slots instead of 26, and the page decode is 580 ms slower; without lazy compute (C) every thumbnail pays 283 + 283 ms of churn and text drops to 756 tok/s. The adaptive ubatch (A vs E) is faster everywhere.
2. **The cost is one LLM decode per image.** A fit over A gives about **1.45 s fixed per decode + 0.415 ms per token**: 1867 ms at 1008 tokens, 1526 ms at 187 tokens. The fixed part is the expert upload pass. Images spend 70% (pages) and 80% (small images) of the prompt time there. The same tokens as text run at 1027 tok/s; pages run at 304, small images at 103.
3. **Hot-store churn around an encode is small now:** about 230 ms per page (room 99 + restore 130), zero for small images, which stay in the small ubatch. Not the 0.6 s of the older plan (the refit is async since `expert-prefetch-adaptive-ub`).
4. **Host weights cost about 90 ms per page encode and 117 ms per small encode** (A vs D). For small images that is half the encode.
5. **Text gaps** (label, `<|vision_end|>`, `<|vision_start|>`, under 128 tokens, so CPU experts) cost about 140 ms each.
6. **Pad 72 -> 80:** page encode 1046 -> 646 ms (-38%), small images no change (229 vs 217). Output identical on page01, page05 and thumb01 at temp 0 with `-ehs 0 -cmoe` in `llama-mtmd-cli` (96 tokens of transcription each). The pad takes 13 MiB more VRAM per page encode, and the store planted 24 slots instead of 26 at load. A_pad's page decodes were slower (2053 vs 1867 ms); its rep 1 matched A (23.3 s total against 27), reps 2-3 were slower. Settled by the interleaved rerun in 7.3.

### 7.3 Interleaved A / A_pad rerun (2026-10-26, GPU to itself)

Order A, A_pad, A, A_pad, 3 reps each, same driver, plus a guard that aborts if a second `llama-server` runs. SM clock 1770 MHz median for the whole run, 73 C at most: no thermal drift. Per-image medians, with the run 1 numbers for comparison:

| run | hot slots at load | page encode ms | page LLM decode ms | 8 pages, prompt s | 12 small, prompt s | text 7419, prompt s |
|---|---|---|---|---|---|---|
| A (run 1) | 26 | 1046 | 1854 | 26.8 / 27.3 / 26.9 | 22.5 / 23.0 / 23.2 | 7.8 / 7.2 / 7.2 |
| A, first | 26 | 1045 | 2214 | 28.0 / 30.1 / 35.0 | 22.8 / 25.4 / 22.3 | 8.0 / 7.1 / 7.8 |
| A, second | 26 | 1012 | 1686 | 25.1 / 25.2 / 26.0 | 21.1 / 21.3 / 23.4 | 7.5 / 6.7 / 6.7 |
| A_pad (run 1) | 24 | 646 | 2027 | 23.3 / 25.6 / 25.3 | 22.4 / 24.8 / 25.3 | 8.0 / 7.1 / 7.6 |
| A_pad, first | 24 | 620 | 1936 | 24.9 / 24.3 / 23.1 | 22.7 / 23.6 / 20.5 | 7.9 / 7.3 / 7.2 |
| A_pad, second | 24 | 622 | 1897 | 23.4 / 23.7 / 24.0 | 22.3 / 23.8 / 23.0 | 8.2 / 7.0 / 7.1 |

- **The pad's encode gain is solid:** 1012-1046 ms -> 620-646 ms per page in all three pairs (-39%).
- **The page decode difference was noise.** A's own run medians spread from 1686 to 2214 ms; A_pad's sit inside that range (1897-2027). Over all 9 reps the 8-page prompt median is 26.9 s for A against 24.0 s for A_pad: **-2.9 s per 8 pages (-11%)**, about what the encode gain alone predicts (8 x 0.4 s).
- Small images and text: no change (median 22.8 vs 23.0 s, 7.2 vs 7.3 s).
- **The cost is 2 hot slots** (24 instead of 26, in all 3 A_pad loads). The worst-case mmproj estimate rises only 26 MiB (1173.8 -> 1199.5 MiB) and the warm-up compute buffer 26 MiB (287.9 -> 313.6), but that crosses a slot boundary. The padded Q, K and V are extra F32 tensors. Generation speed with 24 vs 26 slots was not measured (each request generated 4 tokens).
- To build the pad without the slot cost: pad in the weights instead of the graph. Insert 8 zero rows per head into `qkv_w` / `qkv_b` and 8 zero columns per head into `o_w` at load, so the matmuls produce and consume 80-wide heads directly. No extra activation tensors, about 11% more weight bytes on those two tensors (a few MiB) and 11% more FLOPs on them. Open point: M-RoPE then runs on 80-wide heads with `n_dims = 36`; check that `GGML_ROPE_TYPE_VISION` rotates only the first 72 dims and passes the 8 zeros through, or the outputs change. Check the slot count and the generation speed before keeping it.

### 7.2 Projection for 3.2 (packing), from the fit

| request | today (A) | packed, estimate |
|---|---|---|
| 8 pages | 26.9 s: encode 8.4 + churn 1.8 + decode 14.9 + gaps 1.8 | about 15 s: encode 8.4 + churn 0.2 + 2 decodes 6.3; about 12 s with the pad |
| 12 small | 23.0 s: encode 2.6 + decode 18.3 + gaps 2.1 | about 5.5 s: encode 2.6 + 1 decode 2.4 + ubatch switch 0.5 |

So packing is 1.8x for pages (2.3x with the pad) and about 4x for small images. Keeping the tower weights on the GPU only for the length of a burst (3.4) would take another 1.4 s off the 12 small images.

## 8. Built and measured (2026-10-26, laptop, uncommitted)

### 8.1 Why the pad cost 2 slots: the lazy encoder's load-time reserve

The hot store's slot cap (`hot_s_max`) is set once by the load-time fit, and the server adds the projector's worst-case memory to the fit target (`mtmd_get_memory_usage`, `server-context.cpp` around line 1589). Under `--mmproj-compute-lazy` that still counted the whole encode compute buffer on CUDA (288 MiB, 314 with the pad), though lazy mode frees it between encodes and every encode makes room by dropping hot slots anyway. So that VRAM sat idle during generation and the store could never grow into it; the pad's larger buffer pushed it over a slot boundary.

Change: `clip_get_mem_usage()` (`tools/mtmd/clip.cpp`) reports at most 32 MiB of device compute for a lazy encoder (the backend context, measured 18.5 MiB). Result: **28 slots with or without the pad** (was 26 / 24). The pad no longer costs anything at idle, so the weight-padding idea of 7.3 is not needed for slots; it would only save the pad copies and 13 MiB of encode-time VRAM. Not built.

What 2 slots are worth for generation (400 tokens, temp 0, MTP, `-ehs 26` vs `-ehs 24`, reps 1/2/3): 32.8 / 35.1 / 36.1 against 32.2 / 32.8 / 33.1 tok/s, about **5-7%** in reps 2-3.

Side effect: with less VRAM held back, small-image encodes can need room again. In one unpacked run each thumbnail paid about 0.3 s room + 0.3 s restore (Windows: the store sizes on the driver budget, which is under NVML free; every refit re-plants all slots). Packing pays that once per run, so it does not show with `--mtmd-pack`.

### 8.2 `--mtmd-pack` (default off)

- `find_mtmd_run()` / `process_mtmd_run()` in `tools/server/server-context.cpp`: from an image chunk, take images and the text between them up to `n_batch` tokens (and the context limit); the text after the last image keeps the normal path, so sampling, checkpoints and MTP re-priming are unchanged. Fewer than 2 images, a non-causal model, audio or an active LoRA falls back to the old one-image path.
- One make-room / restore round for the run (sized for its largest image), one encode per image, then one `llama_decode` of the whole run as embeddings with M-RoPE positions (text rows take the same position in all 4 sections, as `llama_batch_allocr` does for tokens). One log line per run: `media run: N images + M text tokens ...`.
- Text rows: new `llama_model_get_tok_embd_rows()` (`src/llama-ext.h`, `src/llama-model.cpp`) dequantizes only the requested rows with the type's `to_float`, zero-padded to `n_embd_inp`. Refuses Hadamard-latent tables and `f_embedding_scale` models. The server also allows packing only for `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe`, because some graphs scale token input after the lookup (Gemma: sqrt(n_embd) for tokens only) and packed rows would miss it.
- Flag: `--mtmd-pack` / `--no-mtmd-pack`, env `LLAMA_ARG_MTMD_PACK` (`common/arg.cpp`, `common/common.h`).

Correctness (`-ehs 0 -cmoe`, temp 0, 3 pages + 2 headers + 2 thumbnails, 80 tokens with top-5 logprobs):

| compare | reply | chosen-token logprob diff, max / mean |
|---|---|---|
| unpacked vs packed | identical | 0.084 / 0.005 |
| unpacked `-ub 4096` vs `-ub 512` (no packing) | differs | 0.375 / 0.019 |

So packing moves the numbers less than an ordinary ubatch change does. The prompt cache reuses a packed prefix (the repeat request processed 4 tokens).

Speed, config A flags (`-b 4096 -ub 4096 -ubg 512 -epf 2`, host weights, lazy), 3 reps, median prompt seconds:

| | A | A + pack | A + pad | **A + pad + pack** |
|---|---|---|---|---|
| 8 pages (8167 tokens) | 26.1 | 16.6 | 23.2 | **13.2 (2.0x)** |
| 12 small images (2375 tokens) | 21.3 | 5.7 | 22.2 | **5.5 (3.9x)** |
| 8 pages + 300-token summary: prompt | 26.0 | 16.6 | 23.3 | **13.2** |
| same: generation tok/s | 32.2-34.5 | 31.9-33.8 | 30.6-33.8 | 32.0-33.1 |
| same: MTP draft acceptance | 0.76-0.81 | 0.78-0.84 | 0.78-0.82 | 0.76-0.81 |

All four load 28 hot slots. Packed pages run as 2 runs of 4 images (4050 tokens): LLM decode 3.3 s per run (about 1225 tok/s, was 1.75 s per page), encode 4.07 s per run (2.42 s with the pad). The 12 small images are one run: encode 2.3 s, decode 2.5 s.

### 8.3 The pad is the default where the MMA kernel runs

- New CUDA proc `ggml_backend_cuda_fattn_uses_mma(dev, op)` (`ggml-cuda.cu`), backed by `ggml_cuda_flash_attn_ext_uses_mma()` (`fattn.cu`): it runs the real FA kernel selector on an op that only describes shapes, so every vendor's rules apply (Turing+, Volta, AMD MFMA).
- `clip_ctx` asks it once at load with the padded vision op (head size 80, 1024 patches, 16 heads, F16 K/V, no mask) and sets `fa_pad_72`. `build_attn` pads only when that is set, the head size is 72, and the encode is not on the CPU fallback. `MTMD_FA_PAD=0/1` still forces it off or on. Load log: `flash attention is enabled, head size 72 padded to 80 for the MMA kernel`.
- Checked on the laptop: default pads (4-page encode 2.4 s, 8 pages 13.0-14.2 s), `MTMD_FA_PAD=0` does not (4.0 s, 16.3-17.4 s), both load 28 hot slots, a forced CPU-fallback encode works, greedy output identical pad vs no pad on page01, page05 and thumb01.
- Not measured: the GTX 1660 Ti. TU116 counts as Turing in ggml-cuda, so it takes the MMA path and gets the pad. If the encode there is slower with the pad, set `MTMD_FA_PAD=0` and report it.

### 8.4 Left to do

- Run on the home server (Linux, 1660 Ti): the smaller load reserve means encodes rely on making room; check no abort under VRAM pressure (`LLAMA_DEBUG_VRAM_HOG_MIB`).
- Small images are now encode-bound: host-weight streaming is about 0.12 s of each 0.2 s encode. Keeping the tower weights on the GPU for the length of a run (3.4) would take about 1.4 s off the 12-image request.
- A soak with mixed text and image turns and context growth, as in the lazy-vram plan.

## 9. Next steps (after commits 19aa6cff7, 4fbb4fa39, 42ddea2fc)

Where the time goes now (A + pad + pack, laptop):
- 8 pages, 13.2 s = 2 runs x (room 0.07-0.35 + encode 2.42 + restore 0.1-0.3 + LLM decode 3.3 s). LLM decode about 50%, encode about 37%.
- 12 small images, 5.5 s = encode 2.3 + LLM decode 2.5 + text gaps. Host-weight streaming is about 0.12 s of each 0.19 s small encode.

In order:

1. **Validate on the home server** (Linux, GTX 1660 Ti), before relying on it. The smaller lazy reserve means every encode relies on making room; check for no abort under `LLAMA_DEBUG_VRAM_HOG_MIB` pressure with 1-4k-token images. Time the encode with and without `MTMD_FA_PAD=0` (TU116 takes the MMA path without tensor cores). Run the p0 request set with `--mtmd-pack`.
2. **A larger ubatch for packed runs** (flags only). Every ubatch pays one expert upload pass, and packing now fills big batches. Try `-b 6144 -ub 6144` and `-b 8192 -ub 8192` with `-ubg 512`: runs of 6-8 pages and fewer upload passes per token. Before, ub 8192 overflowed VRAM (320 tok/s); the hot store now gives up slots during large batches, so measure again. Also check that generation gets its slots back after the switch. Estimate: the LLM half of a page run 20-30% shorter if it fits.
3. **Tower weights in VRAM for the length of a run** (3.4). Upload the 861 MiB once per run (about 90 ms) instead of streaming it on every encode; free it with the compute buffers. Estimate: 12 small images 5.5 -> about 4.2 s, 8 pages -0.6 s. Costs more hot-store eviction during the run only, so idle VRAM stays the hot store's.
4. **Client side** (toolbag `servers/office/render.py`): crop scanned page margins before the 1 MP cap (fewer tokens for both encode and decode, likely 10-25% on scans), and tile runs of thumbnails into one sheet when they are only for overview.
5. **Profile the padded encode** (nsys, one 1008-token page): how much is the BF16 matmuls against attention now. Decides whether an F16 or Q8_0 mmproj, or a better FA config for D=80 with 16 heads and no mask, is worth trying. Check OCR accuracy on real scans before any lossy change (#29251).
6. **Embedding cache by image hash** (3.4): skips the encode for a page re-sent after the prefix changed (edited history, new system prompt, same PDF in a new chat). The LLM prefill still runs.
7. **One encode graph per run with per-image attention**: the matmuls of all images of a run in one batch (weights read once, larger GEMMs), the attention, RoPE and merges per image. Would make item 3 unnecessary, but it is a real change to `qwen3vl.cpp` and the encode estimate. Only if 3 and 5 leave the encode as the larger half.
8. **Soak** as in the lazy-vram plan (mixed text and image turns, context growth), then consider upstreaming the pad (a small, self-contained change; #16837 left head size 72 as a TODO).

## 10. Larger ubatches, weights per run, batched encode (2026-10-26, laptop)

All with `--mtmd-pack`, lazy mmproj, host weights, `-ubg 512 -epf 2`, MTP n_max 1, `-c 4096 --ctx-max 262144`, 2-3 reps. Requests: 16 pages (16286 tokens), 8 pages, 12 small images, 7.4k text, 400-token generation.

### 10.1 `n_batch` no longer capped by the starting `-c`

`llama_context` capped `n_batch` at the starting `n_ctx`, so `-b` above `-c` was silently ignored under `--ctx-max`. decode() already grows the KV cache when a batch does not fit, so with growth on (one sequence) the cap is now `n_ctx_max` (`src/llama-context.cpp`). Checked: `-c 4096 -b 8192` gives `n_batch = 8192`, and packing builds runs of 8 (and 12) pages.

### 10.2 Ubatch sweep

| config | 16 pages s | 8 pages s | 12 small s | text s | hot slots large mode |
|---|---|---|---|---|---|
| `-b 4096 -ub 4096` | 25.4-26.6 | 12.8-13.8 | 5.1-6.3 | 6.9-8.7 | 0-15 |
| **`-b 8192 -ub 4096`** | **24.2-25.0** | **12.4-12.8** | **5.0-5.5** | 6.9-7.1 | 1-15 |
| `-ub 6144` (MTP draft too) | 30.9-31.7 | 13.5-13.8 | 5.8 | 10.5 | 0-3, draft switch fails |
| `-ub 6144 --spec-draft-ubatch 2048` | 30.9-31.4 | 12.9-13.5 | 4.9-5.0 | 9.5-9.8 | 0-14 |
| `-ub 8192 --spec-draft-ubatch 1024/2048` | 130-172 | 14.5-15.1 | 4.9-5.0 | 8.5-9.1 | 0-14 |
| `-ub 12288` | 88-89 | 16.1-16.8 | 6.2-6.5 | 9.9-10.4 | switch fails |
| MTP off, `-ub 4096 / 6144 / 8192` | 29.0 / 27.8-32.5 / 32.7-33.4 | 14.5-16.0 / 14.6-15.7 / 13.0-13.8 | | 7.8 / 8.2 / 6.6 | 13-17 / 4-8 / 0-2 |

- Target compute buffers: 1678 MiB at ub 4096, 2226 at 6144, 2582 at 8192. Above 4096 the hot store is empty during the batch, and the card is at its edge: on Windows that pages, so 8k-token decodes are bimodal (7.0 s or 10.5-11 s per 8106-token run), and some runs collapse.
- The MTP draft context switched with the target and wanted another 1131-1436 MiB at 6144 (its one layer's activations and expert staging, its outputs are already capped at 1). At 6144 that failed and the draft stayed at ub 512. New flag `--spec-draft-ubatch N` (`common/arg.cpp`, `common/speculative.cpp`) caps only the draft's large ubatch. It removes the failed switches, but 6144 is still slower than 4096 on this card.
- **On a 6 GB card the useful maximum is ub 4096. The win is `-b 8192 -ub 4096`:** runs of 8 pages, one round of making room and one switch per 8 pages instead of per 4. About 3-5% on pages; `--spec-draft-ubatch` stays for larger cards.

### 10.3 Vision weights on the device for a packed run

- `clip_ctx::weights_to_device()` / `weights_to_host()` (`tools/mtmd/clip.cpp`), exposed as `mtmd_weights_device_bytes()`, `mtmd_weights_to_device()`, `mtmd_weights_to_host()` (`tools/mtmd/mtmd.h`): one device buffer, every weight tensor re-homed into it with `ggml_backend_tensor_alloc` + a copy, pointers restored after. The scheduler places ops by the tensor's buffer, so the encodes stop streaming the weights. A CPU fallback encode moves them back first.
- `process_mtmd_run()` asks for room for the weights too, and copies them only if the smaller of NVML free and the driver's budget (`ggml_backend_dev_memory`) covers encode + weights + `--kv-spill-margin`. The first version checked NVML only: on the second run of a 16-page request (large-ubatch buffers still held) the copy oversubscribed and the encode paged (19 s) or fell to the CPU (102-110 s). With the stricter check those runs stream as before (9 of 12 runs copied in the 16/8-page tests). `MTMD_PIN_WEIGHTS=0` turns it off.
- Output: logprobs identical to the unpinned run (max difference 0.0).
- `-b 8192 -ub 4096`, pinned / not pinned: 16 pages 24.2-25.0 / 24.9-25.6 s, 8 pages 12.4-12.8 / 12.6-13.4 s, **12 small images 5.0-5.5 / 5.8-6.0 s**. Encode of the 12-image run 1.16 / 2.26 s, of an 8-page run 4.32 / 4.83 s. Generation after: 33.4-35.1 / 29.3-33.1 tok/s.

### 10.4 One encode graph per run with per-image attention: not worth it

nsys timeline of the pinned encodes (`-b 8192 -ub 4096`):

| run | span | GPU busy | gaps between images | GEMM | flash attention | elementwise and copies |
|---|---|---|---|---|---|---|
| 12 small images | 1021 ms | 89% | 11 x 7.8 ms = 86 ms | 415 ms | 99 ms | about 400 ms |
| 8 pages | 4129 ms | 97% | 7 x 17 ms = 118 ms | 1292 ms | 1326 ms | about 1370 ms |

- The GEMMs are near the card's BF16 peak already (8 pages: about 26 TFLOP in 1.29 s, 20 TFLOPS), and only 15% dearer per patch on small images (46 vs 40 us per patch).
- One graph per run would save the gaps (the per-encode graph build, allocation and sync) and part of the small-image GEMM gap: about 160 ms on the 12-image request (3%) and 120 ms on 8 pages (1%). Attention stays per image. Not worth a rewrite of `qwen3vl.cpp`.
- Where the page encode goes instead: flash attention 33% (D=80 MMA, about 62% of peak), GEMM 32%, bias and residual adds (`k_bin_bcast`) 12%, the head pad (`pad_f32`) 4.4%, RoPE 4.3%, casts 4%, norms 3.5%, GELU 2.8%, copies 3%.

### 10.5 Next

1. Commit 10.1-10.3 and use `-b 8192 -ub 4096` with `--mtmd-pack`.
2. The LLM decode is still about half of every image request (7 of 12.4 s for 8 pages). Its cost per ubatch is the expert upload pass; that is the 35B prefill plan's territory (hot store sizing during prefill, pinned host experts for faster DMA).
3. Encode: try the other MMA FA configs for D=80 with 16 heads and no mask (the encode uses ncols 64); fuse the bias adds into the GEMMs where ggml-cuda can; pad the qkv/o weights instead of the activations (removes `pad_f32`, 4.4%, but needs the RoPE check of 7.3).
4. Home server validation (section 9 item 1) still open.

### 10.6 Two more measurements

- **GPU use during the LLM decode of an 8-page run** (same nsys trace): 7.58 s window, kernels running 67% of it, host-to-device expert uploads 51%, either 89%. So the SMs wait 2.5 s per 8 pages (1.7 s on uploads alone, 0.8 s fully idle). Overlapping the next run's encode with that would save at most about 2.5 s per extra 8-page run, but needs the encode's VRAM while the large-ubatch buffers are held (not there on 6 GB) and two contexts computing at once. Not on this card.
- **Pinned host experts (`-lm none`) with `-epf 2`**: no gain. 16 pages 25.4-26.5 vs 25.7-25.8 s (mmap), 8 pages 13.3 vs 12.9-13.2 s, text 7.4-7.5 vs 7.1-7.3 s.

### 10.7 Home server: long prompts encode on the CPU; the large ubatch now steps aside (2026-10-27)

Kyle's server run (GTX 1660 Ti, Kyle's flags with `-b 8192 -ub 4096 --mtmd-pack`, a 32k prompt of text then 2528x1632 pages of about 4000 tokens each) encoded a page on the CPU: "the encode needs 855 MiB of VRAM, 776 MiB are free". With `-lv 4` the cause was plain:

| step | hot slots |
|---|---|
| load fit | 32 (2184 MiB) |
| KV cache grows 4096 -> 34304 cells (80 MiB free) | 32 |
| target to ubatch 4096: compute buffers 1818 MiB | 10 |
| MTP draft to ubatch 4096: compute buffers 792 MiB | 0, store disabled |
| encode: nothing left in the store, KV moved to host, 776 MiB free | 0 |

At 34k cells the two large-ubatch buffers hold 2.6 GB for the whole prefill, and the encode's room-making (hot store, then KV) never asked them. Fix: `llama_ubatch_release()` / `llama_ubatch_restore()` (llama-ext.h) drop a context's buffers to the small ubatch's size without a hot store refit, and take them back. The server's `ubatch_step_aside` (server-context.cpp) releases the draft then the target when free VRAM is short of the encode's need plus the keep, before the hot store and the KV are asked, and restores target then draft after the KV comes back and before the hot store refits. Both encode paths (`process_mtmd_chunk`, `process_mtmd_run`); the log lines say "large ubatch released and back".

Laptop repro (`p0/repro_long.py`: 3 copies of text.txt, then two pages upscaled to 2528x1632, 30226 prompt tokens, same flags as the server): the same state (0 hot slots after the draft grows), and with the fix both runs encode on the GPU: room 74-82 ms, encode 7.99-8.04 s, restore 170 ms, decode 11.2-11.4 s, prompt 49.0-49.1 s (616 tok/s). No CPU encode. The old binary was not rerun on the laptop; the server log is the before.

### 10.8 `GGML_CUDA_NO_MMA`: Turing code without the tensor-core kernels (2026-10-27)

The server is built with `-DCMAKE_CUDA_ARCHITECTURES="61-virtual" -DGGML_CUDA_FORCE_MMQ=ON`, ggml's advice for GTX 16xx (no tensor cores; ggml's Turing MMA kernels run but slowly there). That code has no fast FP16: `FAST_FP16_AVAILABLE` is off for `__CUDA_ARCH__ == 610`, because consumer Pascal runs FP16 at 1/64 rate. So the vision attention (tile kernel, head size 72) does its math in FP32, while TU116 runs FP16 at twice the FP32 rate. The pad was never on there (no Turing MMA compiled, the probe says no). Measured on the server: 2 pages of ~4000 tokens encode in 55.5 s with the BF16 mmproj (FP32 cuBLAS) and 83.7 s with the F16 mmproj (cuBLAS picks its own sm_75 FP16 kernels, slow on this card), so BF16 stays.

New CMake option `GGML_CUDA_NO_MMA` (`common.cuh`): leaves out the Volta/Turing/Ampere/Blackwell MMA availability macros and makes `volta/turing/ampere/blackwell_mma_available()` and `fp16/bf16_mma_hardware_available()` (NVIDIA) return false, so every kernel choice takes the non-tensor-core path. Built for 75 that keeps fast FP16. The startup hint for GTX 16xx / MX450 / MX550 now points at it, and `system_info` lists `NO_MMA = 1`.

Laptop check (`build-nomma`, arch 86, the 3060 has tensor cores so speed says nothing about TU116): test-backend-ops MUL_MAT 1391/1391, MUL_MAT_ID 1063/1063, FLASH_ATTN_EXT 3979/3979, OUT_PROD 71/71 against the CPU. Two 4000-token pages, 3 reps each (`p0/cmp_nomma.py`): identical transcription on both builds; encode 21.4-21.6 s without MMA vs 7.9-8.0 s with it (Ampere's non-tensor FP16 runs at the FP32 rate, so this is the tensor cores' worth, not a TU116 forecast), decode 9.1-10.7 vs 6.6-8.1 s.

Server build to compare against the 61 one: `cmake -B build-75 -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_NO_MMA=ON -DGGML_CUDA_FORCE_MMQ=ON`. Expected: the encode's attention up to 2x faster (attention is most of the 55 s); the LLM side gets FP16 kernels too, so check text prefill and generation speed as well.

First server run with the 75 + NO_MMA build aborted in prefill: the MTP draft's attention (tile kernel, D=256, ubatch 4096) split the KV over ~3 parallel blocks, each an F32 copy of the output (64 MiB) from the VMM pool mid-graph, 128 MiB past the reserve: `cuMemCreate ... out of memory`. With MMA that path is stream-k and needs no copies, so the reserve's formula never counted them. Fix (2d37de8ae, `fattn-common.cuh`): the non-stream-k launch keeps one block per tile when the output tiles fill 8+ waves on their own (a split adds no parallelism there). Laptop no-MMA build: FA tests 3979/3979, pool stays at the 82 MiB reserve where the old code grew it to 130 MiB, encode 21.3-21.8 s and decode 9.3-11.2 s (unchanged). Kyle's hunch from the server: generation ~33 vs ~36 tok/s with the earlier 61 build; not measured.

Still open on the server: the BF16 mmproj runs as F32 cuBLAS on Turing (no BF16 hardware; use `mmproj-F16.gguf`, which `-hf` does not pick because it ties with BF16 on bits and BF16 lists first), 4000-token pages (`--image-max-tokens`), the pad on TU116 (`MTMD_FA_PAD=0` A/B), and the prefill itself runs with 0 hot slots once both contexts hold ubatch 4096 at 34k cells (`--spec-draft-ubatch 512`, or `-ub 2048`).

## 11. Sources

- ggml-org/llama.cpp#29385 `llama_batch_ext` (mixed token/embedding batches), merged 2026-09-28.
- ggml-org/llama.cpp#16837 clip FA (head size 72 left as TODO); #30241 cast the mask once.
- ggml-org/llama.cpp#28166 M-RoPE image decode on qwen3_5 recurrent memory; #30220 draft holes after mtmd chunks; #29251 Qwen3-VL embedding noise.
- vLLM V1 encoder cache and encoder disaggregation (vllm.ai blog), LMCache encoder cache docs.
- ET-Prune (arXiv 2608.01979), PixelPrune (arXiv 2604.00886), NVIDIA TensorRT-Edge-LLM DART docs.
