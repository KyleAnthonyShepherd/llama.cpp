# PLAN - parallel pages: hydra heads with page images, sequences that exist only while used

Rev 1, 2026-11-05. Nothing built or measured for this plan yet. Belongs with the fork's plans
(copy to `I:\Kyle_Shepherd\opt-llama\PLAN-parallel-pages.md` when work starts). For the
implementation agent: read `AGENTS.md` first (ASCII only, short comments in simple English, no new
files under `tests/`, reuse what exists) and stop at every gate.

## 1. What Kyle wants

- Crawl triage (and later the moisture extract) on about 4 pages at once on one server: one GPU
  pass reads the 4 page images, and the 4 answers are generated in the same decode steps. Each page
  keeps its own context (system prompt + that page), exactly as today's single-page request.
- **No cost when not asked for.** A server that can do this must run every normal request (chat,
  transcription, a single crawl page) at the same speed as today's `-np 1`, with the same number of
  expert hot slots. Parallel work only happens when a client such as geoui asks for it in the
  request.
- Kyle's expectation to test: 4 triage pages in parallel select a similar set of experts, so
  generating them together costs much less than 4x one stream.

## 2. Facts to start from (do not re-derive)

**What an extra sequence costs today** (laptop, Qwen3.6-35B-A3B UD-Q4_K_S, from
`toolbag/tests/expert_pool/data/prompt/lv4.shared/server.log`, `-np 1`):

| item | size | note |
|---|---|---|
| recurrent state, 1 sequence | **125.6 MiB VRAM** (S 120, R 5.6) | 30 delta-net layers; 2 rows per sequence because `n_rs_seq = 1` (MTP rollback), so 62.8 MiB per row |
| KV cache | 20 KiB per token (80 MiB at 4096) | 10 attention layers |
| target compute buffer at load | 287 MiB (1630 MiB at ubatch 4096) | reserved with `n_seqs = n_seq_max` (`src/llama-context.cpp:831`) |
| MTP draft context | KV 8 MiB at 4096, compute 193 MiB | its own context, `n_seq_max = n_parallel` |
| one expert hot slot | about 68-73 MiB | 28 slots at load (PLAN-image-prefill 8.1); 2 slots are worth 5-7% generation speed |

The recurrent state is allocated at load for every sequence: `mem_size = max(1, n_seq_max)` cells
times `(1 + n_rs_seq)` rows (`src/llama-model.cpp:2920`, `src/llama-memory-recurrent.cpp:101`), and
`resize()` is a no-op (`:721`). So today `-np 5` costs 4 x 125.6 = about 500 MiB, about 7 hot
slots, forever, even when the extra slots sit idle. That is the overhead Kyle is worried about, and
it is measured, not a guess.

**Hydra already exists** (PLAN-hydra.md 9.4-9.6), uncommitted in the worktree
`I:/Kyle_Shepherd/opt-llama-hydra` (branch `hydra`, on 11e2050ab, the same commit as the main folder;
23 files, about 980 lines):

- one `/v1/chat/completions` request: `messages` is the root, `"hydra": {"heads": [...]}` holds one
  tail per head, each head may override `max_tokens`, sampling, `response_format`. Results come
  back as `choices[i]`, streaming by `choices[].index`.
- the root is the parent task (`n_predict = 0`), the heads are children that fork from it with
  `copy_state_to` (`seq_cp`) on a unified KV cache (`-kvu`): root KV cells are shared, not copied;
  the recurrent state is copied per head.
- `--ctx-max` works with `-np > 1` on a unified cache (growth in decode and before prefill); per-head
  `n_predict` is capped to a share of the pool; loop guard per head with a light park.
- checked on LFM3: greedy heads equal separate requests up to batch-shape drift; image roots work;
  a second hydra on the same root reuses it (root step 0.1 s).

Hydra's limits that this plan must remove:
1. **Heads are `-np` slots**, so their sequences are allocated at load (the cost above).
2. **A tail cannot hold an image** (roots with media: only the tail text is appended).
3. **Off with MTP**: `hydra_max_heads = spec ? 0 : n_parallel - 1` (`server-context.cpp:6686` in the
   worktree). The 35B runs MTP.
4. Even with image tails, each head's image would be decoded alone (`process_mtmd_chunk` per slot;
   `--mtmd-pack` only packs images of one slot, `find_mtmd_run` needs 2+ images in the same slot).

**Batches of several sequences need equal lengths.** The hybrid memory splits with `split_equal`
(`src/llama-batch.cpp:510`): every ubatch takes the same number of tokens from each sequence; what
is left of the longer ones goes into more ubatches, and each ubatch of 128+ tokens pays the expert
upload pass again (about 1.45 s on the laptop). So the heads' images must have the same token count.
geoui renders the longest side at `triage_px` (1024), so letter pages are about 800 tokens; other
sizes differ.

**Other code that assumes one sequence**: `llama-context.cpp:326` (n_batch may exceed n_ctx only
when `n_seq_max == 1`) and `:2884` (growth inside decode only when `n_seq_max == 1`); hydra's
worktree already extends both to unified caches. `common/fit.cpp:278` sizes an unset context per
sequence. `server-context.cpp:1806` refuses `--ctx-max` with `-np != 1` (hydra allows it for
unified).

## 3. The design

### 3.1 Sequence ids are cheap, sequence memory is made on demand (libllama)

Split "how many sequence ids may exist" from "how many sequences have memory now":

- `n_seq_max` stays the id limit (1 + max heads). With a unified cache this costs nothing in the KV
  cache: one stream, seq sets are bitsets of `LLAMA_MAX_SEQ` already.
- The recurrent memory starts with **1 cell** whatever `n_seq_max` is. New
  `llama_memory_recurrent::reserve_seqs(n)` reallocates `r_l` / `s_l` for n cells (copy the used
  rows, as `llama_kv_cache::resize` does, with a `peak_bytes` estimate first), and `release_seqs()`
  shrinks back to 1 cell once only sequence 0 is in use (move it to cell 0 first). Exposed as
  `llama_seq_reserve(ctx, n)` / `llama_seq_release(ctx)` in `src/llama-ext.h`, forwarded by
  `llama_memory_hybrid` (and `_iswa`, `_idx`), like `resize`.
- **Rollback rows only where needed.** The heads do not draft (3.3), so they need no MTP rollback
  row. Today's layout puts snapshot j of cell c at row `j * mem_size + c`. Change it to: current
  state rows `[0, mem_size)`, snapshot rows only for the first `n_rs_cells` cells (1, the normal
  slot). A head then costs 62.8 MiB, not 125.6. The graph code that reads snapshots
  (`rs_idx`, the rollback in `seq_rm`, the copy at `:990`) must follow. Gate 1 decides whether this
  is in v1 (it halves the heads' VRAM).
- **Reserve for what exists.** `sched_reserve` uses `n_seqs = cparams.n_seq_max` for the worst-case
  graph (`:831`, also `:1144`, `:1407`, `:1562`, `:1656`). Use the cells the recurrent memory has
  now; `reserve_seqs` sets `sched_need_reserve` so the next decode re-reserves for 1 + K, and
  `release_seqs` sets it again. `output_reserve(n_seq_max)` is host memory (about 1 MiB of logits
  per id): leave it.
- `common/fit.cpp` and the hot store's load-time fit must count the allocated cells, not
  `n_seq_max`, so the store plants the same slots as `-np 1`.

Done when: a context with `n_seq_max = 5`, unified, and no `reserve_seqs` call shows the same RS
buffer (125.6 MiB), the same compute buffers and the same hot slots at load as `n_seq_max = 1`.

### 3.2 Head slots that normal requests never see (server)

- New flag `--hydra-heads N` (default 0 = today, and then nothing below changes). It sets
  `n_seq_max = 1 + N`, forces `kv_unified`, and creates N **head slots** beside the normal slot.
  `-np` keeps meaning normal slots (1 here).
- Head slots are left out of `get_available_slot`, prompt-cache matching, `id_slot` and the RAM slot
  store; only a hydra group uses them. Normal requests therefore always land on slot 0 with its
  prompt cache, checkpoints, hats and MTP, as today.
- Head slots make nothing at idle: no checkpoints (heads are forked, PLAN-hydra 5), no prompt cache
  entries, sampler and grammar made per task.
- `/props`: `hydra_max_heads = N` (with MTP too, see 3.3), and `hydra_media_tails: true`.

### 3.3 MTP: the root and normal requests keep it, heads do not draft

- Heads run with `can_speculate() == false`. The root of a group generates nothing
  (`n_predict = 0`), so during a group no slot drafts, and the MTP draft context needs only
  sequence 0: keep its `n_seq_max = 1` and check that `common_speculative` accepts a target with
  more ids (`speculative.cpp:254` checks `n_seq` against the draft context for the simple draft;
  check what the MTP impl checks).
- After a group, slot 0's MTP state is stale (the decodes in between were the heads'). It is
  invalidated already after image batches; do the same at group end.
- Whether 4 heads without MTP beat 1 stream with MTP is Phase 0's main question (MTP acceptance
  0.76-0.84 at n_max 1, so one stream makes about 1.7 tokens per step).

### 3.4 Image tails and one pass for all heads (server)

- **Tails with media.** `handle_hydra` tokenizes each tail with mtmd (images allowed), appends it
  to the root's tokens per head, and keeps today's prefix check for the text. A tail's image chunks
  belong to that head's `server_tokens`.
- **Cross-head packing.** After the fork every head starts its tail in the same `update_slots`
  pass, so the group arrives together: no gather window. The text before the image
  (`<|im_start|>user\n<|vision_start|>`, the same tokens for every head) goes as one ordinary
  multi-sequence token batch. When two or more head slots of a group sit at an image chunk at the
  start of a pass, take a new path, `process_mtmd_group()`, next to `process_mtmd_run()`:
  1. group the heads by image token count (equal counts pack; a head with another count gets its
     own pass; log the groups);
  2. one round of making room (`ubatch_step_aside`, hot store, KV spill) sized for the largest
     image, plus the vision weights on the device for the run (PLAN-image-prefill 10.3), then one
     encode per image;
  3. one `llama_decode` of embeddings for all heads of a size group, each row with its head's
     `seq_id` and M-RoPE positions from that head's `n_past` (the same for all heads: same root,
     same text before the image); split into `n_batch` pieces if K x tokens is larger;
  4. `push_back` the chunks into each head's prompt tokens as today, restore room once.
  The text after the image (file, page number) goes back to the normal path as one multi-sequence
  token batch; unequal lengths there cost a few small extra ubatches (CPU experts, tens of ms), not
  a full pass.
- **Equal sizes from the client.** geoui renders the pages of a group on one canvas size (3.6), so
  the size groups are normally one group.

### 3.5 Memory for a group, and giving it back

- At group start, before the fork: `reserve_seqs(1 + K)` (K x 62.8 MiB, or x 125.6 without the
  layout change) plus the tails' KV, through the same room-making path as an encode (hot store
  slots first, never past the driver budget: on Windows WDDM pages instead of failing, see
  PLAN-image-prefill 10.3). If the room is not there: run the heads in smaller groups, or refuse
  the hydra with a clear error so geoui falls back to single pages.
- At group end: remove the heads' sequences. **Release policy** (gate 2 decides): release at once
  (the hot store refits after every group), or keep the cells while hydra groups keep coming and
  release when a normal request starts or after a few seconds idle. The crawler sends groups back
  to back, so keeping them avoids a refit per group; a chat or transcription request then gets the
  full hot store back before it starts.
- Budget: hydra's per-head `n_predict` cap stays; with `--ctx-max` the pool grows as in hydra.

What a group of 4 triage pages holds, E: heads 4 x 62.8 = 251 MiB, tails and replies 4 x about 1100
tokens x 20 KiB = 86 MiB, so about 340 MiB (5 hot slots) while it runs. Extract groups have long
replies (up to 6000 tokens: 120 MiB of KV per head), so K = 2 for extracts.

### 3.6 geoui

- `geoui/llm.py`: `Server.hydra(root_messages, heads, ...)`: one streaming request, results per
  `choices[].index` (content, finish, timings per head); the live feed labels each head's tokens
  with its page.
- `geoui/crawl.py` `run_crawl`: when `/props` reports `hydra_max_heads` and `hydra_media_tails`,
  take the job's pages in groups of K (twins taken out first, as today). Root = the triage system
  prompt; head i = one user turn with page i's image and its intro text (after the image). The root
  is what the `crawl-triage` hat holds today, so slot 0 keeps it between groups and the hat logic
  is unchanged (`hats.wear` on slot 0 only; head slots are never saved). Then the moisture pages of
  the group (or of the job) as an extract hydra with K = 2 and the `crawl-extract` root.
- Rendering: all pages of a group padded to the same pixel size (white border to the largest page
  of the group at `triage_px`), so they have the same token count. Pages of another size go into
  their own group.
- Fallback: no hydra in `/props`, or the server refuses the group: today's one-page path. A head
  whose reply does not parse is retried alone.
- Settings: per server "pages at once" (0 = off; default 4 when the server offers it). Hats and
  comparative advantage unchanged.
- geoui's own loop guard watches one stream per request today; for a hydra it watches each head's
  text and cancels the whole request only if a head loops past its `max_tokens` share (the server's
  `--loop-guard note` acts per head first).

## 4. Phases and gates

| phase | what | gate (ask Kyle) |
|---|---|---|
| 0 | measure, no code (4.1) | is parallel generation worth it next to MTP? K? |
| 1 | commit hydra; libllama sequences on demand (3.1) | idle cost equal to `-np 1`; rollback-row layout in v1? |
| 2 | server: head slots, MTP rule, image tails, cross-head packing, room and release (3.2-3.5) | same answers as single requests; speed; release policy |
| 3 | geoui (3.6) | crawl pages per hour before and after on a real job |
| 4 | soak on the laptop (mixed crawl groups, chat, transcription, context growth); home server later | defaults |

### 4.1 Phase 0 - measure

1. **What `-np` costs today**, to confirm the table in section 2: start the 35B with Kyle's flags at
   `-np 1` and at `-np 5 -kvu`, record RS buffer, every `sched_reserve` compute buffer, hot slots at
   load and after the first prefill, and text / page / generation speed (ABBA). This is the number
   phase 1 must bring back to `-np 1`.
2. **Parallel generation on an offloaded MoE**: `llama-batched-bench` with the server's model flags
   (no MTP there), `-kvu -npp 900 -ntg 64 -npl 1,2,4,8`: prefill and generation throughput per
   number of sequences. Then the same as 4 parallel server requests on `-np 5 -kvu` with real
   triage prompts (JSON schema on, so CPU grammar cost is in), against one request with MTP.
3. **Kyle's expert overlap question**: `LLAMA_EXPERT_SEL_DUMP` (07adb6a91) during (a) 4 different
   triage pages generated in parallel and (b) the same 4 pages one after another. Per layer and
   step: distinct experts selected in the batch of 4 versus 4 x 8. If the union is near 8-12 the
   heads share most expert reads; near 32 they do not.
4. Expected shape, from PLAN-image-prefill: the prompt part of a page falls from about 2.9 s alone
   to about 1.65 s in a pass of 4 (measured for 4 images in one sequence: same tokens per pass).
   Generation (about 1.5 s per triage page with MTP) is the unknown: with g = aggregate speed of 4
   heads without MTP over 1 stream with MTP, a triage page goes from about 4.5 s to 1.65 + 1.5 / g
   (g = 1: 3.2 s, g = 2: 2.4 s, g = 3: 2.2 s).

Gate 0 questions: if g < 1 (MTP alone is faster), heads could still share the prefill and then
generate one at a time with MTP; that needs per-head draft state and is a different phase 2. K = 4
fills one 4096-token ubatch with 800-token pages; K = 8 needs `-b 8192` (two ubatches, one room
round).

### 4.2 Phase 1 done when

- hydra committed on the daily branch (Kyle reviews the 23 files; the LFM3 parts and the UI are
  generic and stay);
- `--hydra-heads 4` on the 35B with no hydra request: RS buffer 125.6 MiB, compute buffers and hot
  slots equal to `-np 1`, text / page / generation timings within noise (ABBA, 3 reps);
- `reserve_seqs(5)` then `release_seqs()` in a test client: buffers grow and shrink, slot 0's
  context survives both (same next-token logits before and after), the hot store refits.

### 4.3 Phase 2 done when

- a hydra of 4 triage pages (temperature 0, `repeat_penalty` 1.0) gives each head the same answer as
  4 single requests, or differs only where single requests differ under a ubatch-size change
  (compare logprobs, as in PLAN-image-prefill 8.2 and PLAN-hydra 9.3);
- the log shows one room round, 4 encodes and one decode for the 4 images;
- a normal request after a group runs at today's speed, with the full hot store (release policy);
- refusals: too many heads, not enough room, mixed sizes beyond the limit; clear errors that geoui
  can fall back on.

## 5. Risks and things that will bite

- **Hot store churn.** Growing the recurrent memory and the large-ubatch pass both take VRAM from
  the hot store; a release after every group refits it every few seconds. Measure the refit cost
  before choosing the release policy.
- **Hybrid positions and `seq_pos_min`.** Forked heads do not hold their whole history; every use
  of `seq_pos_min` / `seq_pos_max` in the server was checked for hydra on LFM3, not on the 35B
  (`hybrid_append`, checkpoints).
- **The rollback-row layout change** touches MTP rollback for slot 0. Check MTP acceptance and
  greedy output on normal requests before and after.
- **Windows VRAM**: the driver pages instead of failing; every growth checks the budget, not NVML
  free alone (PLAN-image-prefill 10.3).
- **Grammar on the CPU**: hydra measured json_schema heads at about a third of the speed without a
  grammar on LFM3. On the 35B a step is about 30 ms, so a few ms of grammar per head matters less;
  Phase 0 item 2 includes it.
- **Home server** (GTX 1660 Ti): rarely crawls under comparative advantage; validate later, it has
  less VRAM headroom for heads.

## 6. Files

- libllama: `src/llama-memory-recurrent.{h,cpp}` (cells on demand, rollback rows), `src/llama-memory-hybrid*.cpp`
  (forward), `src/llama-context.cpp` (reserve by allocated cells, the `n_seq_max == 1` checks),
  `src/llama-ext.h` (`llama_seq_reserve` / `llama_seq_release`), `common/fit.cpp`.
- server: `tools/server/server-context.cpp` (head slots, `handle_hydra` media tails,
  `process_mtmd_group`, room and release, MTP rule, `/props`), `server-task.{h,cpp}`,
  `common/arg.cpp` + `common/common.h` (`--hydra-heads`).
- geoui: `geoui/llm.py` (hydra request), `geoui/crawl.py` (groups, padded renders, fallback),
  `geoui/docs.py` (render on a canvas), `geoui/config.py` (pages at once), `static/js/misc.js`
  (setting, feed labels).

## 7. Built and measured (2026-11-05, phases 1 and 2)

Branch `parallel-pages` (from `expert-shared-pool` 11e2050ab + the prompt cache restore fix + plans + hydra, see the
commits). Laptop, Kyle's 35B command (`toolbag/tests/hydra/srv35p.sh`, = srv35k.sh with its own log), `-lv 4`.

### 7.1 What changed against section 3

- **MTP on the heads (Kyle, 2026-11-05)**, not the 3.3 rule. The MTP drafter already keeps its state per sequence
  (`pending_h`, `valid`, `verify_h`), so the heads draft like any slot. New `common_speculative_seq_cp()` gives a forked
  sequence the draft state of its source (called from `copy_state_to`); the draft context gets the head ids too (KV only,
  so they cost nothing there). MTP n_max 1 is what is tested.
- So the heads need their rollback row: **no rollback-row layout change** (gate 1 answered by the MTP request). A head
  costs 125.6 MiB of state while a hydra runs.
- **Cells on demand without a new cell layout**: the recurrent memory used `cells[seq_id].tail` as per-sequence metadata,
  which is why it needed `n_seq_max` cells. The tails are now their own array (`tails[seq_id]`), so `size` (cells, the
  row stride the graphs use) can be smaller than the id count. No graph code changed. `seq_reserve(n)` reads the states
  to host memory, reallocates, writes the used rows back (a shrink moves used cells into free ones below n first).
  `llama_context_params::n_seq_alloc` (appended at the end), `llama_seq_reserve()` / `llama_seq_cells()` in llama-ext.h.
  The context reserves its graphs for `min(n_seq_max, cells)` sequences and the matching share of `n_outputs_max`.
- **Expert tier width**: the hot store serves batches of at most 4 tokens (MMQ id compaction bound). 4 heads x (1 + 1
  draft) = 8 tokens ran every expert on the CPU (250 ms per step). New CUDA proc `ggml_backend_cuda_mmid_mmvq_max_batch`
  and `llama_expert_tier_widen(n)`: with `--hydra-heads` (and no `--expert-tier-max-tokens`) the server widens the limit to
  heads x (1 + draft) as far as MMVQ covers every expert type on the device. 35B experts are Q4_K/Q6_K: 8 on Turing and
  later (laptop and home server).
- **Text before the image goes into the group decode** (with `--mtmd-pack`): 4 heads x 34 text tokens (the intro and
  `<|vision_start|>`) were one 136-token batch, over the 128-token op-offload threshold, so it paid a full expert upload
  pass (2.2 s). Now each head's text rows (token embedding table) go in front of its image rows. A head waits one pass
  if a sibling has not started its tail yet, so all heads of the hydra arrive together. Equal sizes are not needed for
  the text (positions are per head); the images are grouped by token count.
- **Release policy (gate 2)**: the cells stay while hydras come; a request that is not a hydra gives them back before it
  starts (`hydra_release`), so it runs with the full hot store. Nothing is released on an idle timer (the main loop does
  not wake up without a task).
- `--hydra-heads N`: N head slots with ids after the `-np` slots, unified KV. Normal requests, `id_slot` (now wraps over
  the normal slots), the LRU / similarity choice and slot save / restore never see them. Heads make no context
  checkpoints. `/props`: `hydra_max_heads` = N, `hydra_media_tails`. A media root with media tails works too (the tail is
  tokenized alone with its own files).

### 7.2 Idle cost (gate 1): same as `-np 1`

| | `-np 1` | `-np 1 --hydra-heads 4` |
|---|---|---|
| RS buffer | 125.62 MiB | 125.62 MiB (1 cell, 5 ids) |
| KV (target / MTP) | 80 / 8 MiB | 80 / 8 MiB |
| target compute | 287.00 + 12.02 host | 287.00 + 12.02 host |
| MTP compute | 193.00 + 12.02 host | 193.29 + 12.09 host |
| hot slots at load | 28 | 28 |

Normal requests (`toolbag/tests/hydra/normal_bench.py`: 3686-token text prompt + 400-token summary, one triage page;
reps 2-3, ABBA). Same tokens and draft acceptance in all four runs; the laptop slows as it heats:

| run | text prompt t/s | text gen t/s | page prompt t/s | page gen t/s |
|---|---|---|---|---|
| A1 (`-np 1`) | 907 / 921 | 32.2 / 32.4 | 203 / 198 | 24.0 / 25.2 |
| B1 (heads) | 899 / 912 | 27.7 / 29.6 | 186 / 182 | 23.2 / 21.3 |
| A2 (`-np 1`) | 869 / 852 | 28.3 / 29.0 | 188 / 190 | 22.3 / 22.9 |
| B2 (heads) | 889 / 890 | 28.6 / 29.3 | 194 / 175 | 22.5 / 20.9 |

### 7.3 Four triage pages in one hydra (gate 2)

`toolbag/tests/hydra/pages_check.py`: Micron pages p171-174 at 1024 px (800 image tokens each), geoui's triage
request (system prompt, intro text then image, json_schema, thinking off), greedy, `repeat_penalty` 1.0.

- **Same answers as 4 single requests** in both runs, max |dlogprob| of the chosen tokens 0.07-0.66 (batch shape drift);
  every boring id matches the truth files. MTP acceptance on the heads 0.94-1.0.
- First run (before the tier and text fixes): 21.2 s vs 24.8 s for the singles. Generation 8.0 s for 4 x 63 tokens.
- With both fixes, back to back with the root kept in slot 0 (as a crawl would run): **about 11 s for 4 pages**
  (2.75 s per page, single requests 6.2-6.6 s per page warm, so ~2.3x). Root 0.2 s (one token again), cells 1 -> 5
  0.3-0.4 s (first hydra only), images + text 5.0 s (4 encodes 1.74 s with the vision weights on the device, one
  decode of 4 x 834 rows), generation 5.3 s for 4 x 63 tokens = 48 t/s together (one stream: ~24 t/s).
- The log: `hydra image group: 4 heads x 800 image tokens + 136 text tokens in 1 decode(s)`, one room round.
- The next normal request: `gave back the heads' state cells, 5 -> 1 cells in 755 ms`, hot store 22 -> 28 slots.
- Streaming, two image sizes (two pages at 1024 px, one at 768 px): the two 800-token heads form one group, the
  456-token head goes alone; answers right.
- Recurrent unit test: `test-recurrent-state-rollback` has a new case (cells on demand vs all cells at load: a pending
  rollback survives the growth, a fork decodes the same, a shrink below the cells in use is refused, a shrink that moves
  seq 0's cell keeps its state).

### 7.4 Open, for Kyle

1. Phase 3 (geoui) is not started: `/props` now reports `hydra_max_heads` and `hydra_media_tails`, so geoui can take
   pages in groups of 4 (3.6). Put the intro text after the image or not: either works now.
2. Under VRAM pressure a hydra competes with the hot store: while the 5 cells (628 MiB) are held and a root needs the
   large ubatch (target 1437 MiB + MTP 792 MiB), the hot store went to 0 slots for that prefill. A crawl keeps the root in
   slot 0, so this only hits the first hydra after another request.
3. Not measured: K = 2 extract hydras with long replies, the home server (GTX 1660 Ti), the expert overlap question
   (4.1 item 3), and `--spec-draft-n-max` above 1 on the heads (the tier is 8 tokens at most).

## 8. Heads sweep for triage (2026-11-05)

`toolbag/tests/hydra/pages_sweep.py`: K Micron pages per hydra (K = 1 is one normal request per page, as geoui does
today), back to back after a warm-up hydra (root in slot 0, state cells made), 3 timed reps of distinct pages per K,
greedy. Server: Kyle's 35B command with `--hydra-heads 16` (`srv35p.sh`; `SPEC=""` drops MTP). Every boring id was right
at every K (165 + 75 pages).

| K | MTP n_max 1: s/page | pages/min | no MTP: s/page | pages/min |
|---|---|---|---|---|
| 1 | 6.52 | 9.2 | 7.32 | 8.2 |
| 2 | 3.65 | 16.5 | | |
| 3 | 3.04 | 19.7 | | |
| **4** | **2.82** | **21.3** | 3.16 | 19.0 |
| 5 | 3.65 | 16.4 | | |
| 6 | 3.86 | 15.6 | | |
| 8 | 3.80 | 15.8 | 3.05 | 19.7 |
| 12 | 6.11 | 9.8 | 3.23 | 18.6 |
| 16 | 6.04 | 9.9 | | |

**4 heads is the best on the laptop with MTP**, 2.3x one request per page. Two walls past it:

1. **Generation leaves the hot store at 5+ heads.** A step is K x (1 + 1 draft) tokens and the hot store serves at most 8
   (the MMVQ bound, 7.1). 4 heads: 5.2 s for 4 x 63 tokens (160 ms per step); 8 heads: 13.8 s (430 ms, every expert on
   the CPU); 16: 25 s. Without MTP a step is K tokens, so 8 heads still fit: that is why the no-MTP curve stays flat to 8.
2. **VRAM.** The state cells are 125.6 MiB per head and the image decode wants the large ubatch (target 1.6 GiB, MTP
   draft 0.8 GiB). At 8 heads (1.1 GiB of cells) the hot store has 0 slots during the image decode (10 s for 8 pages, vs
   3.1 s for 4); at 12+ there is no room for the large ubatch at all, the images go in 512-token ubatches with an upload
   pass each (45-60 s). Without MTP the draft's buffers are not there, so 12 heads still get the large ubatch.

For geoui: groups of 4 on this laptop. A box with more VRAM would move wall 2, but wall 1 stays until the tier serves
more than 8 tokens (the MMQ id compaction TODO in llama-expert-tier.h).

## 9. Boring log transcription with heads (2026-11-05)

`toolbag/tests/hydra/trans_hydra.py`: 4 Micron CME sheets of different borings (p397, p219, p269, p325, exact truth),
one shared root = geoui's transcription system prompt (root fix reply,mark) + 3 example sheets of other borings at
1050 px (7262 tokens, kept in slot 0), each log = one user turn (its page image and PDF text layer, ~2000 tokens).
Thinking on, temperature 0.3, reasoning budget 1500 (PLAN-loop-guard.md 14), json_schema, seed 1, `--loop-guard
reflect`, MTP n_max 1.

| mode | wall for 4 logs | s per log | generation per log (tok/s) | cells right |
|---|---|---|---|---|
| one request per log | 465 s | 116 | 29.5 | 98.8% |
| 2 hydras of 2 heads | 387 s | 97 | 17.5 each (35 together) | 99.2% |
| 1 hydra of 4 heads | 332 s | 83 | 10.4 each (42 together) | 99.2% |

- **It works and is faster, but only 1.4x at 4 heads** (triage: 2.3x). A log is ~3200 generated tokens (1500 thinking +
  the JSON), so generation is ~95% of the time and the shared root saves little: the root is cached in slot 0 for single
  requests too. The heads gain only on the generation step, which at 4 x 2 tokens and 31k cells of context (root + 4 x
  ~6k) runs ~190 ms per step (triage: 160 ms).
- Latency per log goes up: all 4 logs arrive after 5.5 min, instead of one every 2 min. For a job queue it is the
  throughput that counts.
- Accuracy and thinking length did not change (3600-4000 thinking characters per log in every mode).
- More heads would not help on this laptop: past 4 the step leaves the hot store (section 8), and the KV of 4 long
  replies already takes ~620 MiB beside the 628 MiB of state cells.

**Loop guard reflect on heads** (no trip in the runs above, the budget prevents them; forced with 3 text heads, MTP on,
thinking off): head 0 ("write buffalo 600 times") tripped at 602 tokens, light park 36 ms, reflection STUCK yes, note,
new reply; head 2 (numbers as words to 400) tripped at 1053, reflection found no loop, went on, tripped again at 1527,
note; head 1 ran untouched. Restores 27-29 ms. Fixed on the way: the light park did not cut the MTP draft context back,
so the reflection's cells stayed in the draft KV; after the fix the tripped heads keep 0.98 and 0.998 draft acceptance.

## 10. geoui V3-Hydra (phase 3, 2026-11-05)

`W:\Kyle_Shepherd\geocrawler\V3-Hydra` (a copy of V2, not a git folder; V2 untouched). Kyle: 4 heads for everything:
quick looks, moisture reads and transcriptions, waiting for 4 in the queue.

- `llm.Server.hydra()`: one streamed hydra request, per-head results, loop guard events per head, hangs up when every
  head's JSON is complete. `status()` reads `hydra_max_heads` / `hydra_media_tails` from `/props`.
- `jobs._take_group()`: up to 4 crawl jobs at once; transcriptions in groups of one template and one example set
  (`_tkey`), an app-queued group waits up to Settings › Hydra › wait_s (600 s) to fill, a hand-started one goes at
  once. `_chat_hydra()`: the root's hat as before (named from root + first head, the same name V2 used), per-head
  thinking-off retry when the loop guard gives up. `_settle()`: per job as V2; a job whose request was dropped because
  another job of it was cancelled goes back to the queue.
- `crawl.run_crawl_group()`: quick looks 4 pages per request across the jobs; a page that fails goes alone. Moisture
  pages go to a queue across files (`crawl_pages.extract_pending`, a new column V2 ignores) and are read 4 at a time
  when 4 wait, when the oldest waited wait_s, or when no crawl job is left; `kick_pending()` at start for leftovers.
  A server without hydra runs V2's path, which also reads leftovers one by one.
- Tested end to end against the 35B (`--hydra-heads 4`, test data folder, 6 PDFs / 10 pages): 4 + 2 crawl jobs
  pooled, quick looks as 4-head requests (images of two sizes grouped by size), 3 moisture pages of 3 files read in
  one 3-head request at the end, 10 moisture rows saved; 4 transcriptions in one request (259 s for 4 logs, root 9255
  tokens read once, right boring ids); 2 app-queued transcriptions waited 61 s (test wait 60) then ran together;
  cancelling one job of a running 4-head transcription group: it ended cancelled, the other 3 went back and ran as 3.
- Not tested: two servers sharing the queue, the home server, describe (left one page per request: each page's
  prompt holds the previous page's description, so a file's pages depend on each other).

## 11. One log written by several heads (split transcription, 2026-11-05)

Kyle's home server test of V3 transcriptions (4 multi-page logs per request): the 4 tails held 13 page images at
1800 px (~39k prompt tokens), the shared KV cache grew 4096 -> 54272 cells (~1 GiB on a 6 GB card, beside 628 MiB of
head states), the vision encode was ~7 s per page on the 1660 Ti, and generation stayed at the speed of one stream.
Kyle: transcribe one log at a time and split its writing across the heads, with stage 1 parallel too.

`geoui/split.py` (V3-Hydra) and `jobs.run_transcribe_split()`: root = the system prompt, the example pages and the
log's pages (read once). **Stage 1**, one hydra request of 3 heads: title + header; the small sections (groundwater,
notes, remarks); the skeleton (the main table's rows by their first 4 columns + page). **Stage 2**, one hydra
request on the same root plus one shared turn that lists the skeleton's rows: 4 heads, each writes a contiguous
group of rows in full. Each head has its part's own JSON schema. Merge in order; the transcript keeps what each head
wrote and which head wrote each row (`reply.split`). Settings › Hydra › "Split transcriptions" (off by default),
"Thinking per split head" (0). Reruns (old answer and feedback after the pages) run as one request.

Laptop, 4 Micron sheets (p397, p219, p269, p325; examples p220, p189, p339), 35B with MTP, `--hydra-heads 4`
(`toolbag/tests/hydra/split_check.py`, the same split.py):

| mode | cells right | s per log |
|---|---|---|
| one request, thinking on (budget 1500) | 98.3% | 116.6 |
| **split, thinking off** | **99.0%** | **76.0** (stage 1 ~39 s incl. the page, stage 2 ~37 s) |
| split, thinking 300 tokens per head | 96.0% | 120.5 |
| split, thinking 800 per head (first try, 1 sheet) | 98.5% | 277.8 |

- The skeleton was right on every sheet (10, 10, 8, 8 rows). Thinking in each head is the cost: every head re-reads the
  whole page in its thinking (1000-2500 characters each at budget 800), for a part that is a fraction of the reply.
- First version gave each row head the whole row list in its tail (4 x 366 tokens, over the op-offload threshold):
  7.2 s of prompt. The list is now one shared turn after the root.
- Caveat: 4 single-sheet CME logs. Section 14 of PLAN-loop-guard found thinking helps on the two-line-blow layouts
  (single request: thinking off 94.5% vs budget 1500 98.0%); not measured for split heads yet.
- geoui end to end (test data folder): an auger probe log with no samples (stage 1 only, the empty skeleton is right),
  and the template's two-line-blows log: 7 heads, 11 rows, 63 s. The review page shows the heads as chips (each opens
  what it was asked, its thinking and its reply), a "side by side" view of all heads, a badge on the header and
  sections saying which head wrote them, the head of each sample row in the # column, and a mark on a row whose first
  columns differ from the skeleton.

**Finished heads give their state back** (server, `hydra_trim`): 5 s after a head finished, while other heads still
write, the recurrent state cells shrink to the heads still running and the hot store refits (seen: 5 -> 4 cells,
hot store 16 -> 18 slots; 4 -> 2 cells, 20 -> 24). Heads that end together (crawl pages) trim once or not at all.
**TODO**: a finished head's KV cells (a long transcription tail is ~10k cells, ~200 MiB) stay until the hydra ends:
the unified cache holds every head's cells mixed together and only shrinks from the end. Giving them back needs the
cache compacted (cells moved down) while it is in use.
