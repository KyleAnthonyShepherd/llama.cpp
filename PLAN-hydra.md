# Plan: hydra heads - one shared root, N tails, cascade attention (LFM3-VL-2B first)

Rev 1, 2026-10-29. Written for an implementation agent. Read `AGENTS.md` first: ASCII only, short comments in simple English, no new files under `tests/`, reuse existing infrastructure, and **stop and ask Kyle at every gate below** (each phase ends with one). Kyle owns the design; this plan gives the shape and the open decisions.

## 1. What Kyle wants

A request with one **root** context and N **tails** (one per head). The server reads the root once, gives each head its own tail, and then decodes all N heads together.

Per decode step, for N heads:

| Data | Read from memory |
|---|---|
| model weights | once (not N times) |
| KV of the root | once (not N times) - this is the "cascade" part |
| KV of head i's tail | once, by head i only |
| result | N new tokens, one per head |

Plus:

- the llama-server web UI can call a hydra: for N heads it shows N boxes, each the tail message of one head, and shows N replies;
- a test on `W:\Kyle_Shepherd\geocrawler`: can a **2-stage swarm** of LFM3 heads transcribe a boring log quickly and accurately? (section 7). This likely needs protocol and template changes.

Target model: `LiquidAI/LFM3-VL-2B-GGUF` QAD-Q4_0 with `mmproj-LFM3-VL-2B-F16.gguf` (in the HF cache, `I:/Kyle_Shepherd/huggingface`). The design must not exclude other models, but only LFM3 has to work.

## 2. Facts to start from (do not re-derive)

Repository state: branch `expert-shared-pool` in `I:\Kyle_Shepherd\opt-llama` holds the LFM3 port (it was branch `lfm3`). Work on a new branch `hydra` from it, in its own worktree (`git worktree add -b hydra I:/Kyle_Shepherd/opt-llama-hydra`); do not build in the main folder, Kyle runs his server from `build-cuda` there and a running server locks its DLLs.

LFM3-VL-2B (`src/models/lfm3.cpp`, `tools/mtmd/models/lfm3vl.cpp`):

- 24 layers: 13 short-conv (recurrent state, `conv_L_cache` 7), 5 sliding-window attention (window 256), 6 global attention. 16 Q heads, 4 KV heads, head dim 128. Global KV is about 12 KiB per token (f16).
- Hybrid memory: `llama_memory_hybrid_iswa` (`src/llama-memory-hybrid-iswa.cpp`). `seq_pos_min` is the max of the attention and recurrent minimums; see the append rule in `server-context.cpp` (`hybrid_append`, commit `d048cadf2`) for why that matters.
- Laptop (RTX 3060 6 GB, 65 W cap): Q4_0 decode 171 t/s single stream (about 1 GB of weights per step, so weight reads dominate), prefill about 6400 t/s, page image encode 0.26-0.39 s. Server footprint about 2.4 GB at 32k context.
- LFM3 writes JSON well with thinking off; with thinking on it overthinks (see `PLAN-loop-guard.md` sections 13.x).

What the server already has:

| Piece | Where | What it does |
|---|---|---|
| `n_cmpl` (`"n"`) parent/child tasks | `server-context.cpp`: `launch_slots_with_parent_task` (~3885), child activation (~5801), `copy_state_to` (~1194); `server-task.h` `add_child` | the parent slot processes the prompt once, then copies its state (`seq_cp`) to child slots; all children have the **same** prompt |
| all slots in one decode | `update_slots` (~4672), one `llama_decode` per step (~5718) | N active slots already share one weight pass per step |
| choice index | `server-task.cpp` (`"index"`) | OpenAI `n > 1` responses carry `choices[].index` |
| prefill-only request | `n_predict = 0` (fork, "RAM slot store, prefill-only") | a slot can process a prompt and generate nothing |
| loop guard | `guard_*` in `server-context.cpp` | per slot; `reflect` parks and restores the slot's whole state |

What the KV cache already has (`src/llama-kv-cache.cpp`):

- streams: with `kv_unified = false`, `n_stream == n_seq_max` and `seq_to_stream[s] = s` (~138-156). Each sequence has its own K/V stream; `seq_cp` across streams copies data.
- with `kv_unified = true`, one stream; `seq_cp` only adds the sequence id to the cells (no copy). The root is then stored once, but every head's decode token attends over the whole pool with a mask, tails of other heads included.
- `get_k` / `get_v` (~1714, ~1734), `set_input_kq_mask` (~2387); the FA call is in `llm_graph_context::build_attn_mha` (`src/llama-graph.cpp` ~2723).

What the CUDA flash attention already has (`ggml/src/ggml-cuda/fattn-common.cuh`): split-KV partial results with per-row `(max, sum)` meta (`dst_meta`) and `flash_attn_combine_results` (~919) that merges them by log-sum-exp. A cascade merge is the same arithmetic. CPU FA is in `ggml/src/ggml-cpu/ops.cpp` (~8650 and on).

## 3. The design (cascade)

### 3.1 Memory layout

- The root is sequence R, in its own KV stream. It is written once (prefill of the root, image encodes included) and is read-only while heads run.
- Head i is sequence H_i with its own stream that holds **only its tail** (positions P, P+1, ... where P = root length).
- Each head stream knows its parent: (stream of R, P). This is new state in `llama_kv_cache` (call it the "fork" of a stream). Nothing is copied from the root's global-attention KV.

LFM3 specifics:

- conv (recurrent) layers: the root's state at P is copied into each head at fork (`llama_memory_recurrent::seq_cp`, small: 13 layers of a few KiB per head). Each head then updates its own.
- SWA layers: copy the root's last <= 256 cells into each head's SWA stream at fork (5 layers x 256 x 2 KiB, about 2.5 MiB per head). After 256 tail tokens the root is out of the window anyway. This breaks "root read once" only for these at most 256 cells of 5 layers; say so in the code and in the results. Cascade on SWA layers is a later option.
- global layers: cascade (3.2).

### 3.2 Attention for global layers

For each head token q (prefill of tails and decode alike):

```
out_q = merge( attn(q, K_root, V_root),           // pass 1: all head tokens of the batch together
               attn(q, K_tail_i, V_tail_i, mask) )  // pass 2: per stream, causal over the own tail
merge: m = max(m1, m2); out = (o1*s1*e^(m1-m) + o2*s2*e^(m2-m)) / (s1*e^(m1-m) + s2*e^(m2-m))
```

Pass 1 puts the N heads' query rows into one Q matrix against the root's K/V, so a tile of root K/V is loaded once for all heads (for N x gqa(4) rows up to the kernel's column tile; check what tile the selected kernel uses, `fattn.cu` kernel selection ~600-690). Pass 2 is the existing multi-stream FA with the tails as streams.

Proposed ggml surface (decide with Kyle at gate 3):

- `ggml_flash_attn_ext_lse(...)`: same as `ggml_flash_attn_ext`, plus a second result with the per-row (max, sum) (or log-sum-exp) of the softmax. CUDA can expose what `dst_meta` already holds; CPU computes it in the reference loop.
- `ggml_attn_merge(o1, lse1, o2, lse2)`: the merge above (CPU + CUDA). Or build it from existing elementwise ops first and fuse later.
- Alternative to consider: one FA op that takes an optional shared prefix K/V and walks prefix then own tail in the same online-softmax state (no merge op). Fewer launches, but the prefix tile is then loaded per stream unless the kernel groups streams. Measure before choosing.

Graph: `build_attn_mha` builds the two passes when the KV context says the ubatch's streams have a parent; otherwise the existing single FA. The mask input gets a second tensor (pass 1 needs none for global layers: the whole root is visible).

### 3.3 Server

- New request shape (decide the exact names with Kyle at gate 1). Proposal, as an extension of `/v1/chat/completions`:

```json
{
  "messages": [ ...the root: system, examples, the page..., ending with a closed turn... ],
  "hydra": {
    "heads": [
      { "messages": [ {"role": "user", "content": "tail of head 0"} ], "max_tokens": 400, "response_format": {...} },
      { "messages": [ {"role": "user", "content": "tail of head 1"} ] }
    ],
    "keep_root": true
  },
  "temperature": 0.1, "stream": true
}
```

  Shared fields are the defaults; a head may override sampling, `max_tokens`, `response_format`, `chat_template_kwargs.enable_thinking`. The stream carries `choices[].index` = head index (as `n > 1` does), plus a `hydra` event with root progress (prompt read %, image encode). Non-stream: `choices[i]`.
- Rendering: the root is rendered with the chat template as closed turns (no generation prompt). Head i's full prompt is rendered from root + tail i with the generation prompt. **The server must check that render(root + tail_i) starts with render(root)** for every head, token for token, and refuse the request otherwise (same spirit as the loop guard's template probe). The tail tokens are what follows.
- Tasks: a root task (prefill only, `n_predict = 0`) plus N head tasks that wait for it (like `SLOT_STATE_WAIT_OTHER`). When the root is done, each head forks from it (3.1) and prefills its own tail; then all heads decode in the same batch. This is the existing parent/child path with per-child suffixes.
- `keep_root`: the root slot keeps its context after the request, so the next hydra with the same root (2-stage swarm, stage 2) skips the image encodes and the root prefill. The root slot must be picked by prefix similarity, as normal requests are.
- Slots: `-np` must be at least N + 1. `/props` reports the largest N (`hydra_max_heads`).
- v1 limits (say them in the help text and errors): one hydra group in the server at a time, no other requests mixed into its batch; loop guard `note` works per head, `reflect` falls back to `note` for heads (reflection parks a whole slot state, which a forked head does not own).

### 3.4 Templates

- LFM3 (ChatML style): root = [system?, user: examples / page ...] closed with `<|im_end|>`; tail = a further user message. Consecutive user turns render fine in ChatML; check that the model follows a second user turn as well as one merged turn (gate 2 has a quality check for exactly this).
- If merged turns are needed (the head instruction inside the same user turn as the page), that is a template change: render the root with its last user turn left open and let the tail close it. Do it as a hydra template file, the way the loop guard templates are files (`models/templates/*-loop-guard.jinja`), with a header comment saying what changed and why. The prefix check above must still pass.
- Keep `LFM3-VL-2B-loop-guard.jinja` as the base (it is LiquidAI's template unchanged).

### 3.5 Web UI (`tools/ui`, Svelte)

- A hydra mode in the chat form (`components/app/chat/ChatForm/`): a head count (1 to `hydra_max_heads`), then N text boxes, one tail per head. The normal input box (with attachments, e.g. the page image) is the last root message.
- Send: `services/chat.service.ts` builds the `hydra` request (root = conversation so far + the input box, heads = the N boxes) and reads the stream by `choices[].index`. It already handles fork events (`loop_guard`) and the prefill bars; follow those patterns.
- Show N replies side by side (columns, horizontal scroll on narrow screens), each streaming.
- Store each head as a branch: user tail message + assistant reply as sibling branches under the root, so the existing branch controls let Kyle continue any one head as a normal chat.
- Build with Node 22 on PATH (`I:\Kyle_Shepherd\tools\node22`), or CMake silently embeds upstream's prebuilt UI. Check for `UI: npm build succeeded` (or `npm output up-to-date`).

## 4. Phases and gates

| Phase | What | Gate (ask Kyle) |
|---|---|---|
| 0 | Baseline measurements, test harness, geocrawler ground truth | numbers in section 8; is cascade still worth it? |
| 1 | Server: hydra request, per-head tails, root kept, template prefix check. No new attention yet (heads use `seq_cp` copies or the unified cache) | request shape, v1 limits |
| 2 | geocrawler 2-stage swarm test on the phase 1 server (accuracy) | does the swarm read logs better or faster than one pass? |
| 3 | Cascade attention: ggml ops (CPU + CUDA), KV fork state, two-pass graph, LFM3 fork rules | op design (3.2) before writing kernels |
| 4 | Web UI hydra mode | UI layout |
| 5 | Re-run section 8 and the geocrawler test with cascade; defaults; write-up | merge into the daily branch? |

Phase 4 can run in parallel with 3 (it only needs the phase 1 protocol).

### Phase 0 - measure first

- `n` (parent/child) on LFM3 as it is today, N = 1, 2, 4, 8, 16, root P = 2k and 10k tokens, tails 64 tokens, 256 generated per head: decode step time and total time, with `--kv-unified` and without. Record which FA kernel runs (CUDA kernel selection depends on the number of query rows; log it once).
- Expectation to confirm or refute: with the unified cache, one MMA tile may already hold all N decode rows, so the root may already be read about once per step at small N; the cost is compute on other heads' tails and interleaved cells. With split streams the root is copied and read N times. The cascade must beat both; if unified already meets the memory goal at N <= 8, report that at the gate.
- Bytes per step: weights about 1 GB (Q4_0); root KV P x 12 KiB per global-layer set (P = 10k: about 120 MiB). Use step-time scaling to infer reads (step time vs N at fixed P, and vs P at fixed N). nsys GPU traces work on this laptop (CPU sampling does not, no admin); kernel times per FA call are enough.
- Harness: `I:\Kyle_Shepherd\toolbag\tests\hydra\` (toolbag is the home for test tools; see `toolbag/tests/loop_guard/` for the style: `run.py`, plain urllib clients, one JSON per run). No new files under `opt-llama/tests/`.

### Phase 1 - server

Done when:

- a hydra of N = 8 heads with temperature 0 gives each head the same text as 8 separate requests of root + tail_i (exact, or explain any drift with a ubatch-size control as in the `--mtmd-pack` work: greedy text can flip with batch shape on this model);
- the root is processed once (log: one image encode per root image, one root prefill);
- a second hydra with the same root reuses it (prompt_n of the root = 0 or near);
- the template prefix check refuses a template that breaks it (test with one made-up template);
- `-np` too small, too many heads, mixed requests: clear errors.

### Phase 3 - cascade

Done when:

- `test-backend-ops` has cases for the new op(s) (add them to the existing file), CPU vs CUDA pass, including head dim 128, GQA 4, f16 and q8_0 K/V;
- a hydra gives the same text with cascade on and off (greedy), and the same as separate requests;
- step time at N = 8, P = 10k is close to the N = 8, P = 0 step time plus one root read (state the model you used to judge "close"), and clearly below the split-stream baseline;
- kernel trace shows the root K/V read once per step for global layers;
- the hybrid SWA and conv fork rules are covered by a test that forks at P < 256 and P > 256.

## 5. Risks and things that will bite

- **Hybrid memory and positions.** The heads' positions continue from P; the recurrent state must be the root's state at exactly P. The append rule (`hybrid_append`) and the SWA reuse threshold (`61ad80bf9` cherry-pick) both look at `seq_pos_min`; a forked head with a parent will confuse code that assumes a stream holds its own whole history. Grep every use of `seq_pos_min` / `seq_pos_max` in `tools/server` and check each.
- **Checkpoints and the RAM slot store** save and restore a whole sequence. A forked head is not a whole sequence. v1: no checkpoints or slot saves of heads (refuse or skip, and log it).
- **Loop guard** on heads: `note` appends to the head's tail stream (fine); `reflect` parks the slot (fall back to `note`, 3.3).
- **Speculative decoding / MTP**: LFM3 has none; refuse hydra with a draft model in v1.
- **Greedy drift**: on LFM3, greedy text changes with ubatch shape (seen with `--mtmd-pack`). Compare logprobs, not only text.
- **VRAM**: 6 GB laptop. N + 1 slots at large context: size `-c` for root + N tails, not N x (root + tail).
- **The big number trap**: LFM3 refuses "write 10,000 words" style requests on its own; not a hydra problem. A system message helps (`PLAN-loop-guard.md` 13.3).

## 6. Environment notes (this machine)

- Build with CMake from PowerShell; CUDA arch 86, same flags as `build-cuda/CMakeCache.txt` in the main folder (`GGML_NATIVE=OFF` + AVX2/AVX512/VNNI). Use a worktree build dir.
- `test-chat` and other internal-symbol tests only build static on Windows: a CPU-only `BUILD_SHARED_LIBS=OFF` build dir (as `opt-llama-lfm3/build-static`).
- Bash tool: commands over about 7.7 KB fail (write files with the Write tool); `\\` collapses to `\` in Bash heredocs (edit escapes with Edit/Write).
- Run test servers on other ports (8091+); Kyle may run his own on 8080. A second server without GPU: `CUDA_VISIBLE_DEVICES=-1`.
- The laptop slows about 1 t/s per run as it heats: interleave A/B runs ABBA.

## 7. The geocrawler test: 2-stage swarm on a boring log

### 7.1 What geocrawler does today

`W:\Kyle_Shepherd\geocrawler\geoui` (`geoui/jobs.py` `build_transcribe_messages` ~656) sends one transcription request per item:

- system: `prompts/transcribe.md` + the template's ABOUT, RULES, LESSONS and reply format (`templates/boring_log.md`);
- up to 3 example pages: user (image + "Transcribe this example page.") / assistant (the JSON), from `templates/boring_log/0N-*.png|md`;
- the target page(s): user (file, pages, optional engineer note, page image, optional text layer);
- one JSON object back (header fields + `samples` rows: sample, from_ft, to_ft, type, blows 0-6/6-12/12-18/18-24, n_rqd, rec, description, page).

That whole context (several page images) is the natural **root**: it is identical for every head working on the same page.

### 7.2 The 2-stage swarm (Kyle's idea as understood here - confirm at gate 2)

1. **Stage 1, one head**: root + "List the samples on this page: sample id, from, to. Also the header fields." Small JSON, thinking off. This is the skeleton.
2. **Stage 2, N heads on the same root** (kept from stage 1, so no image encode or root prefill): head 0 = the header; head k = sample k (or a small group of samples), with the skeleton in its tail ("The page has samples S-1 0.0-2.0, S-2 2.5-4.5, ... Transcribe S-3 only: one JSON object with keys ..."). Each head has its own `response_format` (one row) so the grammar fixes the shape.
3. **Merge** in the client: header + rows in skeleton order; run geoui's `checks.py` (depth order, N vs blows, recovery vs drive) on the result.

Why it may help: each head reads one row (a 2B model loses track in 12-column tables; see geocrawler's `CRAWLER.md`, "V2: one row per call"), outputs are short (less looping), and the heads run in one batch.

### 7.3 What has to change for it

- server: the hydra request (phase 1), `keep_root`, per-head `response_format`;
- template: check the second-user-turn tail on LFM3 (3.4); if merged turns read better, the hydra template;
- geoui (in `W:\Kyle_Shepherd\geocrawler`): a hydra transcription path next to `build_transcribe_messages` that reuses the same root messages exactly (same order, same image sizes), stage 1 and stage 2 calls, the merge. Keep the single-pass path; add a setting to choose.

### 7.4 Test set and scoring

- Ground truth with exact answers:
  - the 4 template example pages in `geoui/templates/boring_log/` (their `.md` is the transcription). Leave the page under test out of the root's examples (leave-one-out);
  - the synthetic logs of the loop guard rig (`toolbag/tests/loop_guard/tasks.py` `boring_log_image`: clear, blur, smudge), whose values are in the code;
  - more: boring logs from `W:\Kyle_Shepherd\geocrawler\Micron-Draft-EIS-Appendix_E_Vol_2.pdf`, with a reviewed transcription (ask Kyle; geoui's Review tab and its verified items on the work computer are the source of truth).
- Runs, same pages and settings: (a) LFM3 single pass (today's geoui prompt, thinking off); (b) LFM3 2-stage swarm, phase 1 server; (c) the same with cascade (phase 5); (d) Qwen3.6-35B-A3B single pass as the reference (`toolbag/tests/loop_guard/srv35.sh` flags).
- Scores: per-cell exact match against the truth (sample id, from, to, each blow, N, rec, type, description with whitespace normalised), rows missed or invented, `checks.py` errors, wall time per page, tokens generated, and for (b)/(c) the time of each stage.
- Write the results into this file (section 9) and a table per run under `toolbag/tests/hydra/`.

## 8. Numbers to fill in (phase 0 and 5)

Phase 0, 2026-10-29: server `n` path, hydra worktree build (= expert-shared-pool 11e2050ab + the prompt cache restore fix), LFM3 QAD-Q4_0, text only, tail 64, 256 tokens per head (ignore_eos, temp 0.8), `-np 16 -b 2048 -ub 512`, parent pinned to slot 0. Mean of two runs in ABBA order (the two runs agree within 3%). Data: `toolbag/tests/hydra/data/phase0/`.

Decode step ms (aggregate t/s = N x 1000 / step):

| Setup | P | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|---|
| `n` split streams (`-no-kvu`, `-c` 16 x 10752) | 2k | 6.5 | 7.9 | 11.7 | 17.7 | 29.9 |
| `n` split streams | 10k | 7.0 | 9.0 | 13.6 | 21.2 | 37.1 |
| `n` unified (`-kvu --swa-full -c 20480`) | 2k | 6.8 | 8.2 | 11.5 | 17.0 | 28.0 |
| `n` unified | 10k | 7.6 | 9.0 | 12.6 | 18.1 | 29.7 |
| `n` unified + `-bs` (backend sampling) | 2k | 6.3 | | | 13.5 | 20.8 |
| engine only, `llama-batched-bench -pps -kvu` | 2k | 6.1 | 7.0 | 9.5 | 12.4 | (fails, SWA) |
| unified, SWA cell fix (no `--swa-full`) (9.3) | 2k | 6.5 | 7.8 | 11.0 | 16.1 | 27.0 |
| unified, SWA cell fix | 10k | 7.0 | 8.3 | 11.6 | 16.8 | 28.6 |
| + conv projections in 2D (9.3) | 2k | 6.5 | 7.1 | 9.3 | 12.2 | 18.5 |
| + conv projections in 2D | 10k | 6.9 | 7.7 | 9.9 | 13.0 | 19.6 |
| + conv 2D + `-bs` | 2k | 6.2 | | | 9.7 | 12.7 |
| + conv 2D + `-bs` | 10k | 6.7 | | | 10.3 | 13.4 |
| engine only, conv 2D (`-kvu`, no `--swa-full`) | 2k | 6.1 | 6.4 | 8.0 | 8.8 | 11.0 |
| engine only, conv 2D | 10k | 6.6 | 6.9 | 8.6 | 9.6 | 12.0 |
| hydra phase 1 | 2k, 10k | | | | | |
| hydra cascade | 2k, 10k | | | | | |

Prefill of the root: 2k 0.34 s, 10k 1.75 s (split) / 1.89 s (unified, `--swa-full` makes the SWA layers attend over all cells). Total wall for N=16, P=10k: split 11.4 s, unified 9.8 s.

FA kernel (nsys, `llama-batched-bench` P=10k, f16 K/V, sm_86): decode N=1 runs `flash_attn_ext_f16<128,128,2,4>`, decode N=8 runs `flash_attn_ext_f16<128,128,8,4>` (ncols1 = 8, ncols2 = gqa 4: all 8 heads' rows in one tile). The vec kernel is never chosen on sm_86 with f16 K/V (gqa 4 lets the MMA path win even for one row).

## 9. Results

### 9.1 Phase 0 (2026-10-29)

Findings:

1. **Split streams read the root N times; unified reads it about once.** Step time from P = 2k to 10k: split +7.2 ms at N=16 and +3.5 ms at N=8; unified +1.7 / +1.1 ms. The split numbers match N reads of the extra root (8k x 12 KiB = 96 MiB per head) at the ~200 GB/s this card reaches. In unified, one MMA tile holds all N x 4 query rows (see the kernel above), so a root K/V tile is loaded once for all heads. This is what the cascade was meant to buy, and unified already has it at N <= 16 (the N=16 tile is the prefill instance `<128,128,16,4>`).
2. **What scales with N is not attention.** At P = 2k each extra head costs about 1.4 ms per step in the server: about 0.8 ms in the engine (dense matmuls switch from MMVQ at N=1 to `mul_mat_q<Q4_0, 8>` at N=8: 0.5 s of 3.3 s kernel time in the trace, more than all decode FA together at ~1 ms per step) and about 0.6 ms in the server (CPU sampling per slot). `-bs` takes N=16 from 27 to 21 ms per step (-23%). FA on the root is ~1 ms of the ~18 ms step at N=8, P=10k.
3. **Cascade gain is small here.** Best case vs unified at N=16, P=10k: save the ~1.7 ms the root costs (6%). Vs split: ~7 ms (20%). Unified also stores the root once (split at P=10k, N=16 keeps 16 copies = 1.9 GiB of global KV on a 6 GB card).
4. **Unified on LFM3 needs `--swa-full` today.** `llama_kv_cache::find_slot` reuses an SWA cell only when it belongs to one sequence (`seq_count(idx) == 1`). After `seq_cp` the root's SWA cells belong to all heads, so they are never freed, and the 4608-cell SWA pool (256 x 16 + 512) runs out: `failed to prepare attention swa ubatches` after ~150 decode steps at P=2k, at once at P=10k. `--swa-full` works around it but makes the SWA cache full size (VRAM; at `-c 172032` the card overflowed and WDDM paged: 2-6x slower) and makes the SWA layers attend over every cell (prefill +8%). Fix idea: reuse a cell when it is SWA-masked for every sequence it holds (small change, upstream code).
5. **Split streams need adjacent slots.** With `kv_unified = false` a ubatch only takes consecutive seq ids (`split_equal(..., sequential)`). The parent is picked by LRU (slot 14 or 15 after a warmup), the children take the free slots from 0 up, so every step was two decodes: N=2 at 16.5 ms instead of 7.9. The harness pins the parent with `id_slot: 0`. Phase 1 must give a hydra's heads consecutive slots (or use unified).
6. Other bits: `llama-batched-bench -pps` fails on LFM3 with split streams (position check after its prompt copy), so the split engine-only row is missing. Hybrid memory cannot resize, so `--ctx-max` does nothing on LFM3 (`memory->resize(...) failed`).

Recommendation for gate 0 (Kyle decides): build phase 1 on the **unified** cache (root stored once, read once per step) with the find_slot SWA fix instead of `--swa-full`, and defer phase 3 (cascade ops) until a case shows root reads matter: larger P (page images, 30k+), N > 16, or a GPU where weights are not the bottleneck. Turn `-bs` on for hydra runs where the sampler chain allows it (grammar from `response_format` still samples on the CPU).

Gate 0 answer (Kyle, 2026-10-29): make the find_slot SWA fix for unified (9.3). Phase 1 / phase 3 order still open.

### 9.3 After gate 0: SWA cell fix and the conv projection fix (2026-10-29, hydra worktree, uncommitted)

**SWA cell fix** (`src/llama-kv-cache.cpp`): `find_slot` reuses a cell when it is SWA-masked for every sequence in it (was: only cells with one sequence), and `apply_ubatch` records the purge range for every sequence of an overwritten cell. Only caches with SWA take the new path. Results: unified runs to N=16 at P=10k with the 4608-cell SWA pool, no `--swa-full`, no `failed to prepare attention swa ubatches`; 2-4% faster than `--swa-full` (SWA layers attend over the small pool again). Greedy check (`toolbag/tests/hydra/greedy_check.py`, saves top-2 logprobs): fix vs `--swa-full` gives identical tokens on every head (256 tokens, N=8, P=10k, two prompts).

Note on greedy comparisons: the LFM3 GGUF sets LiquidAI's sampling defaults (temp 0.1, `repeat_penalty` 1.1 over 64 tokens), and `temperature: 0` keeps the penalty. Logprob drift between batch shapes / cache layouts is ~0.01-0.02 nats mean (0.1 max), and the penalty turns that into token flips where the raw top-2 gap is ~1 nat. With `repeat_penalty: 1.0` the first differences between `n` heads and a single request are at real near-ties (top-2 gap 0.04). Phase 1's "same text as separate requests" test must send `repeat_penalty: 1.0` or compare logprobs.

**Why each head cost ~0.8 ms in the engine: the short-conv layers read their weights once per head.** `build_lfm_shortconv` (`src/models/lfm-base.cpp`, shared by LFM2 and LFM3, same as upstream) reshaped the input to `{n_embd, n_seq_tokens, n_seqs}` before `in_proj` and ran `out_proj` on that 3D shape. In decode `n_seq_tokens = 1`, so the CUDA mul_mat runs `mul_mat_vec_q` with one column and n_seqs channels: one pass over the weight per head. Trace at N=8: 26 such calls per step (13 conv layers x 2) at 150 us each instead of 16 us, ~3.9 ms of a ~12 ms engine step. Fix: both projections on the 2D `{n_embd, n_tokens}` tensor, reshape only around the conv. Same math; greedy output bit-identical to before (N=1 and all 8 heads, 256 tokens).

Effect: server N=16 from 27.0 to 18.5 ms (-31%), N=8 from 16.1 to 12.2 ms. Engine N=16 11.0 ms vs N=1 6.1 ms. With `-bs` the server does N=16 in 12.7 ms per step, ~1260 t/s aggregate vs ~155 t/s for one stream. The decode matmuls are now `mul_mat_q<Q4_0, 8>` (weights read once for all heads), which at N=8 move the ~1 GB of weights in ~5.5 ms, so the model is still close to bandwidth-bound; the per-head cost left is ~0.3 ms in the engine (activations, norms, attention, LM head output) and ~0.45 ms of CPU sampling in the server without `-bs`.

Open: the same 3D-input pattern may exist in other recurrent model builders (mamba, rwkv, delta-net) and could be fixed once in the CUDA mul_mat dispatch instead; not checked. This is also an upstream-worthy fix (LFM2 has the same code upstream).

### 9.4 Phase 1 (server) and phase 4 (web UI), 2026-10-29

Kyle skipped to the UI (geocrawler and cascade later); phase 1 was built first because the UI needs it. Request shape as proposed in 3.3 (not yet confirmed by Kyle), on the unified KV cache, uncommitted in the hydra worktree.

**Server** (`tools/server`):
- `POST /v1/chat/completions` with `"hydra": {"heads": [{"messages": [...tail...], ...overrides}]}`; `messages` is the root. A head may override any request field (`max_tokens`, sampling, `response_format`, `chat_template_kwargs`). `/props` has `hydra_max_heads` (= `-np` - 1, 0 with a draft model or MTP). `keep_root` is not a field: the root slot keeps the root like any cached prompt, and slot selection now prefers, on equal similarity, the slot with fewer tokens past the match (the root slot over head slots), so a hybrid model needs no rollback.
- `handle_hydra` renders the root as closed turns (no generation prompt, `prefill_assistant` off) and each head as root + tail, and refuses unless the head prompt starts with the root prompt (tested with a made-up template that puts the message count first: clear 400). Text-only roots: full tokenization per head and a token-prefix check. Roots with media: only the tail text is tokenized and appended to the root tokens, so images are not processed again per head (tails then cannot hold media).
- Tasks: the root is the `n_cmpl` parent with `n_predict = 0`, the heads are its children (`server_task::hydra`). At the existing fork point, a hydra head gets the root's state (`copy_state_to`, stats reset) and goes to `SLOT_STATE_STARTED`; the normal prompt path then finds `n_past` = root length (`hybrid_append`) and prefills only the tail. `post_tasks` gives the heads indices 0..N-1 and the root N; `handle_results` (split out of `handle_completions_impl`) hides the root's result and streams its prompt progress as `{"choices": [], "hydra": {"root_progress": ...}}`. Loop guard `reflect` becomes `note` for heads.
- Checks (`toolbag/tests/hydra/hydra_check.py`, LFM3 Q4_0 + mmproj, `-np 9 -kvu`): greedy 4 heads vs 4 separate requests, text root 1687 tokens: 3 of 4 identical, head 3 diverges at a near-tie (top-2 gap 0.14 / 0.07 nats, mean logprob drift 0.04 before it: batch-shape drift, see 9.3). Image root: same picture. Second hydra on the same root: root slot reused, root step 0.1 s. Streaming: choices 0-3 only, finish reasons per head, root progress events. Errors for too many heads, `n` with hydra, empty tail.

**Web UI** (`tools/ui`): a "Hydra" chip under the chat form (shown when `hydra_max_heads` > 0) turns on hydra mode: head count (1 to max) and one box per head. The message box (with attachments) is the last root message and may stay empty in a chat that has messages. Each head is stored as a branch under the root: tail = user message, reply = its child, both with `hydra: {group, index, count}`. The chat shows the group as columns (tail, reasoning, reply, tokens and t/s per head, live while streaming, root reading %); "Continue here" moves the chat to that head's branch, and the next message continues it as a normal chat. Not in v1: tools, hats, prefills and the agentic loop in a hydra send; regenerate or edit of a single head. `ChatService.sendHydra` and `chatStore.sendHydra`, `hydraStore`, `ChatFormHydra`, `ChatMessageHydraGroup`. Tested in the browser pane: 3 heads on a story root (title, opening, twists), ~130 t/s per head; continue head 2 + follow-up; reload keeps the group. Built into llama-server with `-DLLAMA_BUILD_UI=ON` and Node 22 (`UI: npm build succeeded`).

Kyle's feedback (16 heads), fixed: the new-chat screen lifts the form by half the window height, so 16 head boxes pushed the message box off the top: the form now stays at the bottom while the head boxes are open, the boxes sit in two columns in a panel capped at 35% of the window with its own scroll, and the panel hides after a send (the chip shows "Hydra: N heads" and opens it again; the head messages are kept for the next root). The reply group wraps into rows (cards at least 17rem wide, long replies scroll inside the card). `crypto.randomUUID` does not exist on a page opened over plain http by IP: the group id uses the app's `uuid()` helper. Checked: 16 heads on one drawn boring-log image, all 16 answers right, ~52 t/s per head (~830 t/s total), `-np 17 -kvu -c 24576`.

Kyle's 16-head run at `-c 32768` crashed: root ~4k + 16 heads x ~2.4k of thinking filled the unified pool, and a decode that finds no room fails every running slot. Fix: `handle_hydra` caps each head's `n_predict` at an equal share of what the root and tails leave of the pool (split streams: what the head's own stream leaves), and refuses a request that leaves nothing. `budget_check.py`: 16 long-story heads in an 8k pool all stop with `length` at 377 tokens, no error. Size `-c` for root + N x (tail + reply); LFM3's global KV is ~12 KiB per token, so `-c 65536` costs ~0.75 GiB.

**Loop guard on heads (built 2026-10-29).** Before: `guard_begin` only watched children (`n = 1` and not a child), so neither note nor reflect ever stepped in on a head, and `handle_hydra` also turned reflect into note. Now:
- `guard_begin` lets a hydra head intervene (a head is a branch of its own for the client, unlike the copies of an `n > 1` request); the downgrade is gone; note and reflection hops keep `id_parent` / `hydra` on the hop task.
- Reflect keeps the full-state park in RAM (Kyle's rule: no checkpoints). With a unified cache the restore removes the head's cells and writes its whole sequence back, so the root becomes the head's own copy (+ root cells per reflection; peak is the same since the reflection cells go first). `handle_hydra` keeps a quarter of the free pool for R = min(N, quarter / (root + 512 + reflect_n_predict)) reflections before splitting the rest into the per-head budgets; the heads share R (`hydra_reflections_left`, by root task id). A head that trips when R is used up gets a note (append only). R = 0 on a small pool: notes only.
- Streams: a head's first "headers only" result went out as `data: null`; now skipped.
- UI: `sendHydra` routes `reflect_start` / `reflect_delta` / `resume` / `note` by choice index. A note saves the reply so far, adds the note as a system message and a new reply under it in the head's branch (as the normal chat does, so a later turn on that head matches the cache). The group view shows each head's chain (reply, amber "Loop guard note", reply), the reflection text while it runs, and a note count badge.
- Superseded the same day, after Kyle's review (he disliked the reserve, the fallback to notes and the root copy): see "light park" below. The reserve, `hydra_reflections` and the fallback are gone.
- Checked: `loop_check.py`, 8 loop-prone rig tasks as heads, thinking on, `-np 9 -kvu -c 65536 --loop-guard reflect`: 8 reflections (park 20-36 MiB in 0.1-0.4 s, restore, note), then notes once R was used, no errors.

**Light park for hydra heads (2026-10-29, after Kyle's review).** A head on a unified cache parks only what the reflection overwrites: `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state (recurrent state; on LFM3 also the SWA window) plus its tokens, held in the slot's `parked` struct (not the checkpoint list, so nothing evicts it). The reflection only appends; unpark writes the partial state back, then `llama_memory_seq_rm(head, park position, -1)` drops the waiting token and the reflection. The global KV cells before the park position are never copied or moved, so the head keeps sharing the root. Measured: 7-9 MiB parked in ~25 ms, restored in 5-20 ms (full park: 20-36 MiB, 120-400 ms). Each head's budget keeps room for one reflection (512 + reflect_n_predict tokens, given back after); every head can reflect every time it trips. Normal (non-hydra) requests keep the full park (Kyle's rule from the loop guard review); the light park could serve them too (it rewinds only appended attention cells by position, the recurrent state comes from RAM) - Kyle's call.

**Adaptive context with hydra (`--ctx-max`).** `llama_memory_hybrid_iswa` (LFM3) did not forward `resize`, so `--ctx-max` did nothing on LFM3: it now forwards resize / resize_peak_bytes / spill / unspill to its attention caches like `llama_memory_hybrid`. A unified cache is one stream, so `llama_kv_cache::resize` already handles it with many sequences; the in-decode growth (on "no free cells") and the batch limit now also apply when `kv_unified`, and the server allows `--ctx-max` with `-np > 1` when the cache is unified. The shrink before a request is skipped while other slots run (it would undercut the heads). The hydra budget uses `--ctx-max` as the pool size. Checked: `-np 9 -kvu -c 4096 --ctx-max 65536 --loop-guard reflect`, 8 looping heads with thinking: the pool grew 4096 -> 20736 in four 1.5x steps (each logged as `init_batch: failed to prepare ...` then retried), 11 light reflections, no errors.

**UI: thinking for the heads.** The hydra panel has Thinking: Default / On / Off (sent as `chat_template_kwargs.enable_thinking`; Default follows the chat's setting); the chip shows "no thinking". Checked in the browser.

**Adaptive head count (design, not built - for Kyle).** Goal: any number of heads, no `-np` to plan for; as many heads decode at once as fit, the rest wait in RAM and resume when a head finishes. Shape:
1. Slots stay the unit of a running head (a sequence id; `n_seq_max` is fixed at context creation and sizes the recurrent and SWA state per sequence: LFM3 ~1 MiB per sequence, Qwen3.5-35B much more). `-np` becomes a ceiling of heads decoding at once, defaulted from that per-sequence cost (e.g. `-np auto` for hydra), not a limit on N.
2. Scheduler in the hydra group: heads beyond the free slots wait in a queue; when a head finishes, the next one forks from the root (pinned in its slot while the group runs).
3. Preemption when the pool cannot grow (at `--ctx-max`, or less free VRAM than a margin: on Windows WDDM pages instead of failing, so the server must check free VRAM itself): park one running head in RAM - its partial state plus its own KV cells from the root length on - free its slot and cells, and put it at the front of the queue; unpark it later into any free slot: share the root (`seq_cp`), write back its own cells, recurrent and SWA state, and its sampler / text / tokens.
4. Needs a libllama addition: state write / read of a sequence from a position on (today a sequence state is all or partial-only), so a parked head comes back without a copy of the root.
5. Stream events per head (`queued`, `parked`, `resumed`) for the UI columns.
Open decisions: which head to park (newest, longest, or least progress); the VRAM margin; whether `-np auto` is enough or `n_seq_max` must grow at runtime (much larger change: the recurrent state is allocated per sequence up front). Browser: 4 heads (count a's, knights, liar, autogram), 4 reflections + 3 notes, notes in the columns, the liar head answered after its note, the chains survive a reload.

### 9.5 Phase 2 (geocrawler), step 1 and a first look at B (2026-10-29, PLAN-hydra-phase2.md)

Harness: `toolbag/tests/hydra/geo.py` (truth = the 7 template example pages through geoui's own `templates`, root = geoui's `build_transcribe_messages` with the page under test left out, scorer, `checks.py`), `run_a.py`, `run_b.py`, `summarize.py`, `bleed.py`. Tables: `toolbag/tests/hydra/data/phase2/tables.md`. Server: hydra build, LFM3 QAD-Q4_0, `-np 17 -kvu -c 8192 --ctx-max 131072 --loop-guard note`. Score = cells of the truth tables (missed rows count as wrong cells; invented rows counted apart), header fields + title apart, numbers compared as numbers.

Layout A (geoui's request as is: 3 examples, temperature 0.3, Qwen top_k/top_p, json_schema), 7 pages x 3 seeds:

| setting | cells right | header right | rows missed / invented per page | checks flags | wall s | tokens |
|---|---|---|---|---|---|---|
| A thinking off | 54.8% | 67.9% | 3.2 / 5.6 | 2.8 | 12.2 | 1530 |
| A thinking on | 30.8% | 50.6% | 7.0 / 5.3 | 2.1 | 28.2 | 3540 |
| A off, 1 example (2 seeds) | 47.8% | 63.5% | 4.8 / 4.4 | 2.9 | 11.7 | 1480 |
| A off, 0 examples (2 seeds) | 32.1% | 59.9% | 7.0 / 7.1 | 0 | 10.4 | 1460 |

Findings:
1. **The main error is example bleed, not misreading.** On boring logs 60-90% of the wrong values are values of the same column in one of the root's examples (`bleed.py`); the model often transcribes the LAST example page (page 04, B-7) instead of the target. Thinking on makes it worse (2 of 3 seeds on page 01 wrote B-7 whole).
2. **The REPLY FORMAT block bleeds too.** geoui's system prompt holds `tpl.to_reply(tpl.example)` (the template's own example = page 01's transcription, 3000 chars). With 0 example pages the model reads the boring number right (B-12, B-7) and then copies the reply-format rows (samples 1, 2, 3 at 0.0, 2.0, 4.0 ...): 0% on boring pages 02-04. This also makes pages 01 (boring and lab) leak their own answer: score them apart.
3. Thinking on: 2.3x slower and worse. Thinking off is the baseline to beat.

First look at layout B on the 4 boring logs (1 seed, G = 1 row per head, row keys and page fixed by `const` in the grammar):

| setting | cells right | skeleton keys right | wall s (stage 1 + 2) |
|---|---|---|---|
| A thinking off, same 4 pages | 47.8% | - | 10.0 |
| B, skeleton from stage 1 | 52.0% | 68.7% | 7.8 (3.1 + 4.7) |
| B, true skeleton (stage 2 alone) | 69.0% | 100% | 6.1 |

- Skeleton: right on 02 and 03; on 01 it split sample 1 into 1A/1B (page 02's layout) and stopped at 8; on 04 it wrote page 02's keys (1A, 1B, 2 ...). Same bleed.
- Stage 2 with the true keys: 01 93%, 03 54% (A 16%), 04 56% (A 21%). Errors left: row-shift (head k reads row k+1), a column shift (Rec <- Type), examples' values on page 04's rows; the header head copies the examples' groundwater on page 02.
- Root reuse works: stage 2 root 8758 of 9298 tokens cached, 0.22 s (stage 1 cold 1.75 s). 10 heads with json_schema made ~250 t/s in total (no grammar: ~860 t/s at 16 heads): grammar on the CPU is likely the cost (step 4).

Gate 1 questions for Kyle: the truth set (7 pages; pages 01 leak through REPLY FORMAT; `*...*` in 02's truth read as italics), the metric, and whether to fix the bleed in the root (reply format without values, examples marked off from the target) before B, since every layout shares the root.

### 9.6 Gate 1 answer, the root fix, layout B on boring logs (2026-10-29)

Gate 1 (Kyle): truth set and metric fine; fix the bleed in the root first, re-run A, then B.

**Root fix (test client only, geoui unchanged)**: `geo.root_messages(fix="reply,mark")` takes geoui's messages and (1) replaces the REPLY FORMAT example with the same keys and empty values (`blank_reply`), (2) labels each example "EXAMPLE PAGE k (an example only, not the page you transcribe)" and puts "THE PAGE TO TRANSCRIBE. It is a different page from the example pages above: read every value from this page image only. Never copy a value from an example." before the target. With (1) the pages 01 no longer hold their own answer, so all 7 pages score honestly now.

A thinking off, 3 seeds (cells right):

| page | geoui root | reply fix | reply + mark | reply + mark, 1 example | reply + mark, 0 examples |
|---|---|---|---|---|---|
| boring 01 (real scan) | 82.6 (leak) | 0.0 | 6.7 | 3.5 | 9.0 |
| boring 02 | 71.6 | 75.6 | 75.0 | 62.9 | 80.6 |
| boring 03 | 16.2 | 33.3 | 43.1 | 8.8 | 8.8 |
| boring 04 | 20.8 | 28.8 | 42.9 | 11.9 | 6.2 |
| lab 01 | 81.8 (leak) | 33.5 | 26.2 | 31.2 | 45.8 |
| lab 02 | 65.2 | 58.7 | 62.3 | 57.1 | 5.2 |
| lab 03 | 45.1 | 45.1 | 45.4 | 39.5 | 26.5 |
| mean of the 5 pages without a leak | 43.8 | 48.3 | **53.7** | 36.0 | 25.5 |

Both fixes help (boring 03 and 04 x2.5), 3 examples stay best. Bleed is still most of the error (`bleed.py`), the real scan (boring 01) is read at 0-9%: two of three seeds still write the B-7 example. Root for all B runs below: reply + mark, 3 examples.

**Layout B, 4 boring logs, 3 seeds** (thinking off, temperature 0.3, row keys and page `const` in the grammar):

| setting | cells right | skeleton keys right | wall s (LFM3) |
|---|---|---|---|
| A (same root) | 41.9 | - | 9.9 |
| B, G = 1 row per head | 33.2 | 61.9 | 6.4 (2.3 + 4.1) |
| B, G = 2 | 32.1 | 61.9 | 6.1 |
| B, G = 4 | 32.4 | 61.9 | 7.2 |
| B, G = 1 + row anchors (row number, rows above and below) | 32.6 | 61.9 | 6.4 |
| B, G = 1, skeleton with its instruction merged into the page's user turn | 33.2 | 61.9 | 6.3 |
| B, G = 1, true skeleton | 55.7 | 100 | 4.9 |
| **B, G = 1, skeleton by the 35B** | **55.8** | 98.9 | 4.9 + 35B |

Per page (A / B own skeleton / B 35B skeleton): 01 6.7 / 6.9 / 44.7, 02 75.0 / 71.0 / 69.8, 03 43.1 / 54.9 / 54.9, 04 42.9 / 0.0 / 53.8.

Findings:
1. **The LFM3 skeleton is the weak part.** It is right on 02 and 03 and wrong on 01 (12 of 33 keys) and 04 (2 of 18: it writes another example's sample ids), the same in every seed. Wordings do not change it (`skel_probe.py`: boring number first, more columns, depths first, temperature 0: all 60.2% or worse); on 04 it writes "B-7" right and then copies the rows. It cannot read those sample columns, so no prompt fixes it.
2. **With a right skeleton, B beats A** on 3 of 4 pages (01 x6.6, 03 +12, 04 +11; 02 -5) and LFM3 needs 4.9 s per page instead of 9.9 s. Row groups (2, 4) and row anchors change nothing; one row per head is as good and simplest.
3. **The second-turn tail is fine.** With the grammar on, the skeleton is byte-identical with the instruction merged into the page's user turn (12 of 12 runs); the logprobs differ (without a grammar the model ignores the tail in both forms and starts a full transcription). No hydra template needed.
4. **The 35B as the skeleton writer** (Kyle's supervision idea; `toolbag/tests/loop_guard/srv35.sh`, unsloth UD-Q4_K_S, thinking off, plain request, 0 examples): 98.9% keys, no missed or invented rows (one misread From on 04), 10.7 s per page here. That launcher reads at only ~130-180 t/s (no `-epf` / `-ubg` flags from the expert-prefetch build): 46 s cold for the 3-example root, ~5 s for the 0-example root; with Kyle's ~1000 t/s prefill a page skeleton would be ~1 s of reading + ~150-300 tokens at ~35 t/s = ~5-9 s, about the same time as LFM3's whole stage 2.
5. Errors left with a right skeleton (12 runs, 1116 sample cells): Rec 63, each blow column 50-55, N 49, description 36, type 16; and the header head misses ~2 groundwater rows per page. The row heads still misread cells; stage 2 is now the limit (~56%).
6. Stage 2 throughput: 10 heads with json_schema ~250-300 t/s in total, far below the ~860 t/s of 16 heads without a grammar: grammar sampling on the CPU (step 4).

Gate 2 questions for Kyle: confirm the 2-stage reading (stage 1 skeleton, stage 2 one head per row on the kept root); which skeleton writer (LFM3 alone is not good enough on 2 of 4 layouts; 35B skeleton works); what next for the row cells: votes per row (layout E), the 35B as a checker of the merged result (reads page + JSON, writes only the wrong cells), or lab pages first (step 3).

### 9.2 Side fix: prompt cache restore into a shrunk KV cache (2026-10-29)

Kyle's 35B log during geocrawler runs: `find_slot: n_tokens = 16802 > size = 4096`, `failed to load prompt from cache`, then a full prefill. Cause: under `--ctx-max` the KV cache shrinks between requests, and `get_available_slot` restored a cached state from the prompt cache (`--cache-ram`) without growing first (the RAM and disk slot restores already call `maybe_grow_for_restore`). Fix: `server_prompt_cache::load` split into `find` + `load`, and the slot grows to the found state's size in between. Repro `toolbag/tests/hydra/restore_repro.py` on Qwen3.5-4B MTP (`-c 4096 --ctx-max 65536`): a follow-up that extends a 16.8k prompt after a short unrelated request, old build prompt_n 17700 (11.5 s), fixed 904 (1.1 s). Note: a follow-up that only shares a prefix still reprocesses on a hybrid model unless a checkpoint sits near the split point, as before.

### 9.7 Phase 2 parked (2026-10-29)

Kyle, after 9.6: ~55% of cells is not usable; base LFM3-VL-2B is too weak for this task (fine-tuning maybe later). Phase 2 stops here; effort goes to the 35B's long thinking on the same task (PLAN-loop-guard.md section 14). The harness (`toolbag/tests/hydra/geo.py`, run_a/run_b/skel_probe) stays for a fine-tuned model.

### 9.8 Many-shot examples in the root (2026-10-30)

Kyle: small models need many examples (in-context learning, ~64). Tried on LFM3 QAD-Q4_0, single pass (layout A), thinking off, root fix reply,mark.

**Truth set**: `toolbag/tests/hydra/micron_truth.py` reads every CME "TEST BORING LOG" sheet of the Micron PDF from its born-digital text layer by word position (header, groundwater, samples with Type/Rec split and blows split on "-", descriptions by geoui's rule: a description belongs to the sample it starts beside or, starting between two, to the one below; layers split at the dashed stratum lines and after a line ending in ")"). 221 sheets, 210 with samples, 1242 sample rows; every table value is found in the text layer by `checks.py`; pages 180/181 checked by eye. Out: `data/micron/pNNN.json` + 1800 px PNG. Auger probe logs (no samples) left out.

**Cost**: an example page costs LFM3 2420 image tokens at >= 1100 px, 838 at 1050 px (one tile), plus ~1360 tokens of JSON; 52 examples at 1050 px = 106k tokens, the 128k pool takes the 6 GB card to 5.8 GB, first read 61 s.

**Sweep** (`run_ms.py`, 8 test sheets from borings not in the examples; examples = other CME sheets, nested; table `mssum.py`, cells right):

| examples | text layer, 1050 px examples | no text layer | text layer, 1800 px examples |
|---|---|---|---|
| geoui's 3 template pages | 20.1 | 19.5 | - |
| 4 CME sheets | 48.8 | 44.2 | **53.9** |
| 8 | 42.0 | 32.9 | 51.8 |
| 16 | 30.8 | 33.5 | 22.6 |
| 32 | 4.3 | - | - |
| 52 | 11.7 | - | - |

Layout B on the 4-example root (1800 px): own skeleton 47.0%, true skeleton 57.8% (`data/phase2/B_ms4*`).

Findings:
1. **Same-firm examples help, many do not.** 4 CME sheets instead of the 3 template pages: 20% -> 49-54%. Past ~8 examples accuracy falls, and at 32-52 the model stops reading the target: it writes one of the examples (both 52-shot test pages came out as example 19, B-596, with its page numbers). This is the "in-context collapse" reported for small open VLMs (arXiv 2608.02830; Llama 3.2-Vision gains nothing from many shots in arXiv 2405.09798, unlike GPT-4o / Gemini).
2. The text layer adds only ~5 points; LFM3 does not use it well.
3. Errors left at 4 examples: the Page column copied from the examples (fixed by `const` in layout B), then blows and Rec misread.
4. More public log formats were not downloaded: more examples hurt, so they would only matter for a fine-tune or as more test layouts.

The truth set (210 sheets, exact) and `micron_truth.py` are the starting point for a fine-tune of LFM3 (Kyle: another day).
