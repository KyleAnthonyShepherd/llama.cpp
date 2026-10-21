# PLAN - DFlash drafter for Qwen3.5-4B (laptop RTX 3060 Laptop, server GTX 1660 Ti)

Goal: run `z-lab/Qwen3.5-4B-DFlash` as the speculative drafter for `unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_S` in this fork, measure it against today's best MTP setup, then tune it. Branch: `qwen35-4b`. Background, numbers and tools: `PLAN-qwen35-4b.md` (read sections 3, 5.7-5.12 first).

Baseline to beat (laptop, 5-prompt sweep, ABBA, see 5.12):

```
llama-server -m Qwen3.5-4B-Q4_K_S.gguf -ngl 99 -fa on -c 32768 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 -bs \
  --spec-draft-vocab-dynamic 16384 --spec-draft-vocab-rank 128 --spec-draft-chain
```
~127 t/s greedy, ~101 t/s sampled (Qwen non-thinking sampling), 3.9 tokens per verify on code.

---

## 1. What DFlash is, and why it may win here

- A separate 0.6B drafter: 6 layers (5 sliding window 4096, 1 full), hidden 2560, FFN 9216, 32 heads / 8 KV heads, head dim 128, tied embeddings (shares the target's 248,320-row LM head), block size 16, mask token 248077 (z-lab `config.json`).
- It reads the target's hidden states at 8 of its 32 layers (1, 5, 9, ..., 29), fuses them (8 x 2560 -> 2560), and denoises a whole block `[id_last, <mask> x (b-1)]` in **one forward pass**. MTP drafts one token per step.
- z-lab, B200, greedy, thinking on: accept length at block 8 is 4.7-5.7 against MTP s=7 at 4.6-5.3; at block 16, 5.9-7.7 against 5.3-6.5. MTP at our depth 4 gives ~3.4-3.9 on our prompts.
- Why it fits this laptop: the sm_86 MMQ table (5.8) makes a 9-17 token verify cost about the same as our 5-token one; one drafter pass (~435 MB at Q5_K_M + one LM head pass) is about the cost of our 4 MTP steps; no MTP catch-up decode. Rough expectation +20-40% on the laptop, less under real sampling. Treat as a hypothesis to measure, not a promise.
- Why it fits the server less well: on the GTX 1660 Ti (no tensor cores) MMQ is dp4a and costs ~2.6x a 1-column mat-vec at every width (`phase0/mmvq-table.sh` on the server, 2026-10-21: Q4_K 146 us at 1 column, 376 us for MMQ at any width). A 9-17 token verify always takes MMQ (above MMVQ's 8-column limit), so deep blocks pay more there.

## 2. What the tree already has (verified 2026-10-21)

- `--spec-type draft-dflash -md <drafter.gguf>` (upstream PR #22105 + PrismML fixes). `docs/speculative.md` "DFlash".
- `src/models/dflash.cpp`: loads `target_layers`, `dflash.block_size`, optional interleaved SWA (`attention.sliding_window` + `sliding_window_pattern`), optional `d2t` reduced vocab.
- `common/speculative.cpp` `common_speculative_impl_draft_dflash` (~line 917): reads block size and mask token from the GGUF, enables `llama_set_embeddings_layer_inp(ctx_tgt, layer, true)` for every target layer. It always drafts `block_size - 1` tokens; it does **not** honour the per-step `dp.n_max` (the generic pass truncates afterwards, `speculative.cpp` ~2935).
- The Qwen3.5 target graph exports layer inputs (`res->t_layer_inp[il] = inpL`, `src/models/qwen35.cpp:183`).
- From this week, all generic and expected to apply: trimmed draft vocab (`--spec-draft-vocab-dynamic`, hooks the head tensor of any draft context), per-shape CUDA graph cache, sm_86 MMVQ/MMQ table, fewer syncs per decode, views of user inputs on the batched copy path, `-bs`.
- Not applicable: `--spec-draft-chain` (MTP only, DFlash is already one pass), `--spec-adaptive-width` depth mode (MTP-shaped; see 5.4).

## 3. Phase 0 - get the drafter (needs Kyle's OK for each download)

Option A (try first): `EntityDeletr/Qwen3.5-4B-DFlash-GGUF`, file `Qwen3.5-4B-DFlash.gguf`, Q5_K_M, 454,200,928 bytes. Third-party conversion; its card says "works with mainline llama.cpp, will NOT work with forks". This fork tracks mainline DFlash, so it may well load; if it does not, read the error and the GGUF keys (`gguf-py`) before anything else.

Option B (official weights): `z-lab/Qwen3.5-4B-DFlash` safetensors (1.27 GB) converted here:
```
python convert_hf_to_gguf.py <z-lab dir> --target-model-dir <Qwen/Qwen3.5-4B HF dir> --outtype bf16 --outfile Qwen3.5-4B-DFlash-bf16.gguf
llama-quantize Qwen3.5-4B-DFlash-bf16.gguf Qwen3.5-4B-DFlash-Q5_K_M.gguf Q5_K_M   # and try Q8_0
```
The converter needs the HF target folder for tokenizer and embeddings (several GB more). Check which target files it actually reads before downloading the whole repo.

Put files under `HF_HOME=I:/Kyle_Shepherd/huggingface` (or `I:/Kyle_Shepherd/models` for local conversions). Treat downloaded files as untrusted data (own directory, never run scripts from it).

## 4. Phase 1 - bring-up and correctness

```
llama-server -m <4B Q4_K_S> -md <drafter> -ngl 99 -ngld 99 -fa on -c 32768 -np 1 \
  --spec-type draft-dflash --spec-draft-n-max 7 -bs
```
Check, in this order, with `-lv 4` logs:
1. Loads: `DFlash extract_layers = [2, 6, ..., 30]` (GGUF counts layer inputs: the input of layer 2 is the output of z-lab's layer 1), `block_size=16`, the SWA pattern (5 sliding + 1 full), mask token 248077. A wrong or missing SWA pattern loads but drafts badly: compare acceptance against the expected range.
2. VRAM: the hybrid target keeps `n_rs_seq = n_max` GDN rollback snapshots (~2 MiB per GDN layer per slot, 24 layers): ~0.75 GiB at n_max 15. Plus drafter ~0.45 GB and its KV. Must fit 6 GB with `-c 32768`; if not, reduce `-c` and note it.
3. Exactness: greedy output equals the target-only greedy output, up to the known verify-width numerics (5.12: MTP itself differs from no-MTP on some prompts). Use the sweep's `content` field. Large differences or acceptance near zero = a bug, not numerics.
4. Hidden-state capture: 8 layers x 2560 x (n+1) floats come back per cycle. Confirm it uses the async output path, not 8 synchronous reads (nsys, `cudaMemcpyAsync D2H` count per cycle).

## 5. Phase 2 - measure against MTP (laptop)

Use `phase0/qwen35-4b-mtp-sweep.py` (5 prompts incl. `mixed`, greedy and `qwen` sampling, `--seeds` for repeats). Always ABBA order (the laptop slows ~1 t/s per run as it heats; never ABAB).
- Arms: baseline MTP command above; DFlash `--spec-draft-n-max` 3, 7, 15 (block 4, 8, 16); each with and without `--spec-draft-vocab-dynamic 16384 --spec-draft-vocab-rank 128`; all with `-bs`.
- Record t/s, acceptance, mean accepted length (server log `mean len`), greedy text identity.
- nsys (`profile -t cuda --cuda-graph-trace=node`, then `scratchpad`-style analysis as in 5.9): per-cycle split of verify / drafter / LM head / small ops / GPU idle. Tools used before: `gaps.py`, `depth_prof.py` (recreate from 5.9/5.10 descriptions if not present).

Decision point: if no DFlash arm beats the MTP baseline by >5% on the mean, write the numbers into this plan and stop. Otherwise continue.

### 5.1 Result (2026-10-21, laptop, ABBA, drafter `Anbeeld/Qwen3.5-4B-DFlash-GGUF` Q4_K_M)

Drafter choice: unsloth has no DFlash GGUF. All current third-party files share one layout (`dflash.target_layers` = [2, 6, ..., 30], SWA 4096 x5 + 1 full, mask 248077, no embeddings or head: the target's Q6_K head is shared). Anbeeld's Q4_K_M is a plain llama-quantize of the June z-lab retrain (its `config.json` is stale, the GGUF is not). Avoid psychopenguin (old 5-layer weights).

Two fixes needed first (uncommitted at the time): the draft context took `n_outputs_max = n_batch` because `llama_model_has_encoder()` is true for DFlash (`llama-context.cpp`), and the noise-block graph ran the head on every row (`src/models/dflash.cpp`, now `get_rows` on the output ids, DFlash2 excepted). The draft compute buffer was sized for 512 rows of 248k logits: 976 MiB -> 108 MiB, total VRAM at n_max 15 5922 -> 5522 MiB. Greedy text unchanged.

`-c 32768`, `-bs` on all, trimmed = `--spec-draft-vocab-dynamic 16384 --spec-draft-vocab-rank 128`, 400 tokens. Cell = t/s (acceptance, mean accepted length).

Greedy:

| cfg | code | prose | extract | mixed | reason | **mean** |
|---|---|---|---|---|---|---|
| no spec | 62.1 | 61.8 | 62.4 | 62.7 | 62.4 | 62.3 |
| MTP n_max 4 + trimmed + chain (baseline) | 135.5 (.68, 3.7) | 89.9 (.37, 2.5) | 157.8 (.85, 4.4) | 98.3 (.43, 2.7) | 144.7 (.75, 4.0) | 125.2 |
| DFlash 3 | 132.6 (.84, 3.5) | 82.8 (.41, 2.2) | 134.6 (.87, 3.6) | 94.5 (.51, 2.5) | 131.5 (.83, 3.5) | 115.2 |
| DFlash 3 trimmed | 137.7 | 87.0 | 139.4 | 98.6 | 137.2 | 120.0 |
| DFlash 7 | 182.0 (.66, 5.5) | 81.0 (.20, 2.4) | 199.6 (.72, 6.0) | 92.5 (.25, 2.7) | 185.1 (.65, 5.5) | 148.0 |
| **DFlash 7 trimmed** | 183.7 | 81.4 | 200.1 | 93.3 | 186.1 | **148.9 (+19%)** |
| DFlash 15 | 165.0 (.37, 6.4) | 58.6 (.08, 2.2) | 195.4 (.48, 7.7) | 74.3 (.12, 2.8) | 168.8 (.36, 6.4) | 132.4 |
| DFlash 15 trimmed | 156.4 | 55.9 | 186.5 | 70.2 | 158.9 | 125.6 |

Qwen sampling:

| cfg | code | prose | extract | mixed | reason | **mean** |
|---|---|---|---|---|---|---|
| no spec | 61.0 | 61.6 | 61.4 | 61.4 | 61.4 | 61.4 |
| MTP baseline | 111.4 (.61, 3.4) | 78.1 (.33, 2.3) | 127.7 (.77, 4.1) | 78.5 (.33, 2.3) | 113.4 (.63, 3.5) | 101.8 |
| DFlash 3 trimmed | 112.7 (.75, 3.2) | 78.1 (.39, 2.2) | 118.9 (.82, 3.4) | 80.9 (.41, 2.2) | 106.3 (.69, 3.1) | 99.4 |
| DFlash 7 | 125.3 (.47, 4.3) | 67.9 (.17, 2.2) | 156.7 (.66, 5.5) | 77.9 (.21, 2.5) | 122.9 (.45, 4.1) | 110.1 |
| **DFlash 7 trimmed** | 128.6 | 69.3 | 158.2 | 79.1 | 124.8 | **112.0 (+10%)** |
| DFlash 15 | 110.2 (.25, 4.7) | 54.0 (.08, 2.2) | 165.2 (.45, 7.4) | 65.4 (.11, 2.6) | 142.5 (.35, 6.1) | 107.5 |

- passes agree within ~2 t/s (A/B means e.g. DFlash 7 trimmed 148.2/149.7 greedy)
- greedy text: DFlash 3 and 7 equal the MTP baseline on all 10 runs. All spec arms differ from no-spec (the known batched-verify numerics, MTP included). DFlash 15 differs from both at near-tie words (16-row verify kernels).
- the gain is code, extraction and reasoning (+35% / +27% / +29% greedy); on prose and mixed DFlash loses to MTP (acceptance .20-.25 at block 8).
- trimmed head: small gain at 3 and 7, a loss at 15 (one top_k + MUL_MAT_ID per row).
- passes the decision point: continue.

nsys, DFlash 7 trimmed, code prompt (179 t/s under the profiler, ~5.6 tokens per cycle, 28.9 ms per cycle):

| part | ms/cycle |
|---|---|
| MMQ (stream-k, 30 blocks): verify trunk + drafter body, 8-16 columns; + fixup | 13.3 + 0.8 |
| target LM head in the verify (8 rows, full vocab) | 2.3 |
| drafter head, trimmed: 32k base 0.3 + 8 per-row picks (`mul_mat_vec_q` 1.25 + top_k 0.5) + scorer 0.16 | ~2.2 |
| GDN with rollback snapshots + snapshot copies | 1.9 + 0.65 |
| other small ops | ~3 |
| GPU idle | 2.9 (9.9%; per cycle ~8 large D2H (feature rows) and 2-3 large H2D) |

The hidden-state capture is async (`ggml_backend_tensor_get_async` in `extract_layer_inputs`), check 4 done. The noise-block K/V are dropped before injection by the server's seq_rm on the draft context after drafting (`server-context.cpp`), so they do not pollute the cache.

## 6. Phase 3 - tune (laptop), in order of expected gain

1. **Trimmed vocab on the drafter head.** The drafter shares the target's LM head. Confirm the `vsub` hook fires for the drafter context (log line at load, nsys shows `mul_mat_vec_q g=32768` / `mul_mat_id` instead of the full 248k head). A block of 15 positions makes the head cost 15x per pass, so this should matter more than for MTP. The hook builds one top_k per output row (`llama-graph.cpp` build_lora_mm vsub branch, "one output per op"); with 15 rows that is 15 top_k + 15 MUL_MAT_ID: check it is still cheaper than the full MMQ head at 15 columns, and if not, try a shared selection per block.
2. **Block size vs verify cost.** Sweep 3/7/11/15. Watch GDN snapshot writes (2 MiB per layer per verified token) and the MMQ verify cost at 9-17 columns (the perf set stops at 8 columns: add a sweep point or time it in the model).
3. **Host round trips.** One drafter decode + one verify per cycle. Check what still waits on the host (the 8-layer feature readback and re-upload). If large, keep the features on the device between the verify and the drafter (the same idea as the MTP catch-up fold in 5.12).
4. **Drafter quant.** Q5_K_M vs Q8_0 vs Q4_K: acceptance against drafter cost.
5. **Adaptive block size.** The DFlash driver ignores `dp.n_max`, so a per-step depth controller cannot help without first making it honour it. Only worth it if 5.2 shows the best block size differs a lot between prompts.

### 6.1 Results (2026-10-21, laptop, ABBA, 5 prompts, greedy + Qwen sampling)

Item 1, drafter head: the per-row picks (one top_k + MUL_MAT_ID per block row) cost more than the 32k base rows, so fewer dynamic rows win. No code. Means, greedy / sampled:

| `--spec-draft-vocab-dynamic` (rank 128) | greedy | sampled | notes |
|---|---|---|---|
| none (full head) | 148.0 | 110.1 | |
| 16384 | 149.8 | 111.6 | |
| 4096 | 155.0 | 115.0 | |
| **1024** | **155.9** | **116.8** | acceptance and greedy text unchanged |
| 256 | 152.4 | 112.3 | second run, df7d1k there 154.1 / 112.9 |
| 64 | 151.0 | 111.3 | acceptance -0.02 on code / reason |
| 1024, rank 32 | 146.8 | 108.6 | the scorer misses: acceptance -0.03 |

A shared pick per block (one top_k for all rows) does not map onto ggml: MUL_MAT_ID with the same ids in every column takes the slow multi-column path. Not needed after the above.

Item 2, block size (all trimmed 16384): n_max 5 139.1 / 107.3, 6 147.6 / 108.5, **7 149.8 / 111.6**, 9 134.5 / 108.6, 11 132.5 / 107.6. n_max 7 = an 8-column verify, the widest that stays on the 8-column MMQ tile; 9 and 11 take the 16-column tile, cost more, and change the greedy text. Per prompt, prose and mixed prefer 5 (+5%), the rest 7: an adaptive block (item 5) would be worth ~2% on the mean. Not done.

Item 3, host round trips: nsys idle per 28.9 ms cycle: verify -> inject decode 1.00 ms, draft -> verify 0.81, inject -> draft 0.47. Built and measured a fold of the inject decode into the noise-block decode (`llama_set_dflash_inject`: the accepted rows go first in the draft ubatch, their K/V from the fused features, attention over all rows, residual/FFN on the block rows only). Exact (greedy text identical, acceptance +-0.01), but **slower: -2% greedy, -6% sampled**, and reverted (patch kept outside the repo). Why: today `process()` launches the inject decode right after the verify, and the GPU runs it while the host samples and accepts; the fold moves that work and the feature upload onto the critical path. Also found on the way: the CUDA graph key (`nodes[0]` only) collides for graphs that differ only in the middle (the fold's varying row count ran eager until the key covered every node's shape). A real gain here needs the features to stay on the device and the noise block to start without a host round trip (device-side sampling of the verify), a much larger change.

Item 4, drafter quant: not run yet (needs another download, Q5_K_M / Q8_0 from the same repo).

**Best laptop config (2026-10-21): 155.9 t/s greedy, 116.8 sampled; MTP baseline 125.2 / 101.8 (+25% / +15%).** VRAM 5258 MiB of 6144 at `-c 32768` (desktop included).

```
llama-server -m Qwen3.5-4B-Q4_K_S.gguf -md qwen35-4b-dflash-Q4_K_M.gguf -ngl 99 -ngld 99 -fa on -c 32768 -np 1 \
  --spec-type draft-dflash --spec-draft-n-max 7 -bs \
  --spec-draft-vocab-dynamic 1024 --spec-draft-vocab-rank 128
```

Where it loses: prose and mixed (block acceptance .20-.25): 85 / 97 t/s greedy against MTP's 90 / 98. On code, extraction and reasoning it is +38-42% over MTP greedy.

## 7. Phase 4 - server (GTX 1660 Ti, Ubuntu)

Kyle runs these (SSH needs his password; code arrives by `git push` + pull). Prepare exact commands for him.
- Env from his table (safe, small effect at <= 8 columns): `export GGML_CUDA_MMVQ_MAX_COLS=q5_K=8,mxfp4=8,q5_1=5,iq4_nl=5,iq2_xs=8,q8_0=5,q3_K=5,iq4_xs=5,iq3_s=8,q6_K=5,q4_0=5,q4_K=5,q4_1=5,q2_K=5,q5_0=8,default=8`
- Sweep MTP n_max 1-4 first (the server's best MTP depth is likely 2-3: 5-column MMVQ costs 2.2x a 1-column one there), then DFlash block 4 and 8. Expect a smaller block optimum than on the laptop.
- Report: t/s, acceptance, `nvidia-smi` clocks/power during decode (the table run showed 1455 MHz SM, no throttling at ~21 W).

## 8. Done when

- Phase 2 table written here (all arms, ABBA, greedy + sampled).
- Either a DFlash config that beats the MTP baseline on the mean with exact greedy text, with its command in section 4 of `PLAN-qwen35-4b.md`, or a written reason it does not.
- Code changes (if any) on `qwen35-4b`, small commits, `Assisted-by: Claude Opus 5.5` trailer, AGENTS.md style (ASCII, short comments, no new test files). Do not push.

## 9. Risks

- The third-party GGUF may not match this fork's DFlash metadata (keys, SWA pattern, block size). Fall back to option B.
- Acceptance under Qwen sampling (temp 0.7, presence 1.5) will be well below z-lab's greedy thinking-mode numbers; judge on the sampled column too.
- VRAM on 6 GB: rollback snapshots grow with block size; the 35B server cannot share the card.
- `--spec-draft-chain` must stay off with DFlash (MTP only); the code falls back, but do not mix flags in the sweep.

Sources: z-lab card and config (huggingface.co/z-lab/Qwen3.5-4B-DFlash), EntityDeletr GGUF (huggingface.co/EntityDeletr/Qwen3.5-4B-DFlash-GGUF), arXiv 2607.04244, llama.cpp PR #22105 / discussion #21569.
