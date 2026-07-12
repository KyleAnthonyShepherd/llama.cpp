# NOTES.md — Discovery log

Handoff artifact for the three companion plans supplied by the user:
- Part 1 — `llamacpp-dynamic-context-plan.md` (dynamic KV growth / `llama_set_n_ctx`)
- Part 2 — `llamacpp-auto-placement-plan.md` (throughput-aware placement / elastic VRAM)
- Part 3 — `llamacpp-mtp-mmproj-plan.md` (MTP speculative decoding × `--mmproj`)

Working agreement (Part 1 §0) applies to all three: discovery before code, verify every
symbol against the actual tree (not the plans — they were written against an older
snapshot and several of their premises no longer hold), small commits, log corrections here.

Base commit for this discovery pass: `e3546c7` on branch `claude/feature-implementation-markdown-c4fadi`.

---

## Cross-cutting correction: the codebase has moved past all three plans

Before any plan-specific discovery, a repo-wide check turned up three premise breaks:

1. **Part 2's extension point already exists in more mature form.** `common/fit.cpp` /
   `common/fit.h` (982 lines) implement `common_fit_params(...)`, taking a writable
   `llama_model_tensor_buft_override *` array and per-device `margins` — this *is*
   today's version of the "`llama_params_fit`" the plan says to search for in
   `src/llama.cpp`. There's also a standalone `tools/fit-params/fit-params.cpp` binary.
   Part 2 Phase A.1 discovery needs to restart against `common/fit.cpp`, not
   `src/llama.cpp`.
2. **MTP already exists, fully implemented, and is more mature than either plan assumes.**
   Both Part 2 §8 and Part 3 describe MTP as young/recently-merged with an
   `--spec-type mtp` flag. In this tree the flag value is **`draft-mtp`**
   (`--spec-type draft-mtp`, see `docs/speculative.md:176,317` and
   `common/speculative.cpp:35`), implemented as
   `common_speculative_impl_draft_mtp` in `common/speculative.cpp:1201-1470+` (full
   draft/verify/rollback loop, backend-offloaded sampling, per-architecture modes for
   gemma4 / step35 / qwen35(moe) — see below). No `-np 1` restriction was found anywhere
   in `common/speculative.cpp` or `common/arg.cpp`; the implementation is sized by
   `n_seq` (`= n_parallel`) throughout (e.g. `smpls.resize(n_seq)`,
   `pending_h.assign(n_seq, ...)`), i.e. multi-slot already appears to be a design
   target, not an out-of-scope restriction to lift.
3. **Part 1's core gap is real and confirmed.** No `llama_set_n_ctx` symbol exists
   anywhere in the tree (`grep -r llama_set_n_ctx` → 0 hits). This part of the work is
   still genuinely undone. Not investigated further this pass (see "Scope of this pass"
   below).

---

## Scope of this pass

Per explicit user direction, this pass covers **only Part 3 Phase 1 discovery**
(read-only: locate the guard(s), map MTP internals, map the mtmd generation path,
confirm the M-RoPE mechanism, check hidden-state capture across embedding batches,
check context-shift status, check the determinism oracle). No code was changed.
Parts 1 and 2 are untouched pending a separate pass.

---

## Part 3 Phase 1 discovery answers

### 1. Find the guard(s)

**There is no blanket startup guard rejecting `--mmproj` + `--spec-type draft-mtp`
together in this tree.** Searched `common/speculative.cpp`, `common/arg.cpp`,
`tools/server/*`, `tools/mtmd/*` for the incompatibility strings the plan expected
("not yet supported with MTP", mutual-exclusion checks, etc.) — none found.

Evidence the two are wired up independently, not mutually exclusively:
- `tools/server/server-context.cpp:1167-1225` — `has_spec` (loads `ctx_dft` /
  `spec_init`) and `has_mmproj` (loads `mctx` via `mtmd_init_from_file`) are two
  independent `if` blocks; neither checks the other.
- `tools/server/server-context.cpp:413-415` — `slot.can_speculate() { return !!spec; }`
  — purely a function of whether the speculative context initialized, not of `mctx`.
- `tools/server/server-context.cpp:1309-1310` — every slot gets both `slot.mctx` and
  `slot.spec` set unconditionally when both features are enabled.
- `tools/server/server-context.cpp:700-756` (`process_mtmd_chunk`) already threads
  `spec` through as `cb_data` into `mtmd_helper_decode_image_chunk` (line ~718-731).

**So Part 3's "central hypothesis" framing (find-and-relax a guard) is the wrong shape
for this codebase.** The plumbing already lets you turn both flags on simultaneously
and it will *run* without erroring. The real incompatibility, found below, is a **silent
correctness gap inside the MTP draft-state catch-up**, not a rejected combination.

### 2. Map the MTP implementation

Struct: `common_speculative_impl_draft_mtp` (`common/speculative.cpp:1201-`), one of
several `common_speculative_impl` subclasses selected by
`COMMON_SPECULATIVE_TYPE_DRAFT_MTP` (`draft-mtp`).

Three runtime modes, chosen once in the ctor from model introspection
(`common/speculative.cpp:1213-1219, 1294-1295`):
- `is_mem_shared` (gemma4): `ctx_dft` shares KV/memory with `ctx_tgt`
  (`llama_get_ctx_other(ctx_dft) == ctx_tgt`); no separate catch-up decode needed.
- `chain_heads` (step35): `n_mtp_layers > 1 && !is_mem_shared` — one trained head per
  draft step, driven via `llama_set_nextn_layer_offset`.
- neither (**qwen35 / qwen35moe — this is the target model's family**): single trained
  MTP head, separate `ctx_dft`/`mem_dft` from the target.

Per-draft-step state:
- `pending_h[seq_id]` — the hidden-state row (`n_embd` floats) that primes the *next*
  draft step; carried across calls.
- `verify_h[seq_id]` / `verify_h_rows[seq_id]` — hidden rows captured from the most
  recent *target* verification batch (row 0 = sampled token, row N = Nth accepted
  draft token).
- `chain_h[seq_id]` — only for `chain_heads` mode.

Three hook methods (called from `tools/server/server-context.cpp`'s generation loop,
which builds `common_speculative_draft_params_vec` / calls into `common/speculative.cpp`'s
dispatch layer around line 2260-2280):
- `begin(seq_id, prompt)` (`:1336-1352`) — sanity-checks `ctx_dft`'s KV position against
  the (text-only) prompt length; logs a "Drafts may degrade" warning on mismatch, does
  not hard-fail.
- `process(batch_in)` (`:1354-1470`) — **the catch-up step**: after the target model
  decodes a batch (prefill or an accepted draft continuation), replay it into `ctx_dft`
  so the draft model's KV + `pending_h` stay in sync with the target's actual generated
  hidden states. **This is where the vision gap lives — see item 5.**
- `draft(dparams)` (`:1472-`) — builds a token batch seeded with `pending_h` as the
  injected embedding row, decodes through `ctx_dft`, samples greedily/top-k per seq.

Divergence vs. non-MTP draft-model specs (`draft-simple`, `draft-eagle3`): those consume
plain token batches into an independent draft model; MTP instead **injects the target's
own hidden state** (`h_nextn`, captured via `cparams.embeddings_nextn`, see item 5) as an
extra input row, because the MTP head is trained to continue the target's residual
stream, not to run an independent forward pass from tokens alone.

Rollback: handled by the caller (`tools/server/server-context.cpp:3774-3841`), via
`common_sampler_sample_and_accept_n` + `llama_memory_seq_rm`-based checkpoint restore
(`slot.spec_ckpt`) when the context can't do partial KV removal
(`ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL`).

### 3. Map the mtmd generation path

- `tools/server/server-context.cpp:700-756` (`process_mtmd_chunk`) drives image-chunk
  decode via `mtmd_helper_decode_image_chunk` (`tools/mtmd/mtmd-helper.cpp`).
- Position bookkeeping is **owned by the mtmd helper, not the server**:
  `mtmd-helper.cpp:329-330` — `n_past += mtmd_input_chunk_get_n_pos(chunk); *new_n_past =
  n_past;`. Every chunk (text or image) advances a single scalar `n_past` counter by
  however many *position slots* it consumes (an image can consume many position slots
  for few actual chunks, depending on `mrope_section`/resolution). The server just
  threads this `n_past` back into its own slot bookkeeping — there is no separate
  "mrope position stream" object the server manages; the scalar `n_past` already *is*
  the correct next-position value for whatever comes next, text or image.

### 4. M-RoPE in decode

This is **not mtmd-specific plumbing** — it's a general, unconditional branch in the
core batch splitter, `src/llama-batch.cpp:781-786`:

```cpp
for (size_t j = 0; j < (size_t)n_pos_per_embd; ++j) {
    // if we are using M-RoPE
    //     if the current batch is text, we need to broadcast the same position across all RoPE sections
    //     otherwise, the input batch is image embeddings, we copy the positions as-is
    // if we are not using M-RoPE, there is only one position per token (this loop runs only once)
    ...
}
```

`n_pos_per_embd` is a property of `llama_batch_allocr` (`src/llama-batch.h:74,134`),
`= mtmd_decode_use_mrope(ctx) ? 4 : 1` at the one call site that constructs it for mtmd
image batches (`tools/mtmd/mtmd-helper.cpp:263`). For a plain **token** batch (which is
all MTP ever builds — see item 5), the public `llama_batch.pos[]` still carries one
scalar position per token (`common/common.cpp:1652-1669`,
`common_batch_add`/`batch.pos[batch.n_tokens] = pos`); `llama-batch.cpp` is what
broadcasts that scalar across all M-RoPE sections during ubatch construction.

**Conclusion: MTP's draft/verify batches (`common_batch_add(batch, id, pos, ...)` at
`common/speculative.cpp:294,353,741,808,1494,1583,1590,1593` etc.) go through the exact
same `llama_batch_allocr` as every other decode call and get the same M-RoPE broadcast
automatically, for free — as long as the scalar `pos`/`n_past` value fed in is correct.**
This matches Part 3 §0's hypothesis: there is no M-RoPE-specific code MTP needs to grow;
position handling is already architecture-generic. The open question is only whether the
`n_past` values `common/speculative.cpp` computes (from `slot.prompt.n_tokens()` /
`dp.n_past`, `server-context.cpp:2914-2932`) are correct post-image — plausible, since
they derive from the same `n_past` the mtmd helper already advances correctly (item 3),
but **not yet verified at runtime** (no GPU/model available this pass — see item 7).

### 5. Hidden-state capture across embedding batches — **this is the real gap**

Found at **`common/speculative.cpp:1354-1362`**, top of
`common_speculative_impl_draft_mtp::process()`:

```cpp
bool process(const llama_batch & batch_in) override {
    if (batch_in.n_tokens <= 0) {
        return true;
    }

    // TODO: how to make it work with vision tokens?
    if (batch_in.token == nullptr || batch_in.embd != nullptr) {
        return true;
    }
    ...
```

`process()` is the catch-up hook that, after every target-model decode, replays the
newly-decoded tokens into `ctx_dft` (draft model) so `mem_dft`'s KV cache and
`pending_h[seq_id]` (the hidden-state row that primes the next draft step) stay current
with the target. **When the batch being processed is an image/embedding batch
(`batch_in.embd != nullptr`, exactly what `mtmd_helper_decode_image_chunk` feeds
through), this hook is a silent no-op** — it returns `true` (success) without touching
`mem_dft` or `pending_h` at all. There's already a author's own `TODO` marking this as
unfinished.

**Mechanism of what actually breaks**, for the `qwen35`/`qwen35moe` and `step35`
families (i.e. everything except gemma4's `is_mem_shared` case — which is unaffected
because it has no separate catch-up step to skip):

- `mem_dft`'s KV cache stops advancing across an image chunk. It's still positioned at
  wherever it was after the last *text* catch-up. Meanwhile `ctx_tgt`'s KV cache and
  position counter (`n_past`, item 3) advance through the image normally.
- `pending_h[seq_id]` is left holding the hidden-state row from **before** the image —
  stale by construction, not merely "unset."
- When generation resumes after the image and `draft()` runs
  (`common/speculative.cpp:1472-`), it seeds the first draft token's injected embedding
  row straight from this stale `pending_h` (line 1495) and computes the draft's KV
  write position from `dp.n_past` (the *target's* correct post-image position, passed in
  by the server) — so the draft model ends up decoding at the **correct position** but
  with the **wrong injected hidden state**, and into a `mem_dft` KV cache that has a
  **gap** (no cells for the image's position span). This is not an M-RoPE bug — positions
  are fine per item 4 — it's a stale/desynced *hidden-state and KV-cell* bug, exactly the
  class of failure Part 3's own risk list calls out ("silently degraded acceptance/
  quality, not a crash").
- `begin()` (`:1336-1352`) has a guard that's aimed at exactly this
  (`pos_max < N - 1` → "Drafts may degrade" warning), but it compares against `N =
  prompt.size()` where the caller passes `slot.prompt.tokens.get_text_tokens()`
  (`server-context.cpp:3713`) — i.e. **N counts text tokens only, with images stripped
  out**, while `pos_max` reflects the position-slot-consuming scheme from item 3 where
  images consume many position slots for few chunks. These two numbers are not
  comparable in an image-containing prompt, so this warning likely **misfires or
  under/over-fires** in the exact scenario it's meant to catch — needs a runtime check,
  not just a read.

This is a precise, small, structural gap — not the "MTP graph fundamentally requires
image token embeddings" failure mode Part 3 §0 worried about as the falsifying case.
The MTP graph itself never needs to consume an image embedding (confirmed — `draft()`
never touches `batch_in`, only `pending_h` and token ids); it just needs its bookkeeping
kept current across the chunk it currently skips.

### 6. Context-shift status with mtmd

Confirmed disallowed, exactly as the plan expected:
`tools/server/server-context.cpp:1216-1219` —
```cpp
if (params_base.ctx_shift) {
    params_base.ctx_shift = false;
    SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
}
```
Also `n_cache_reuse` is force-disabled for multimodal at the same site
(`:1221-1224`), and `has_mtmd` slots skip checkpointing (`:3510-3511`) and cache-reuse
(`:3123-3140`, with a `GGML_ABORT("not supported by multimodal")` as a belt-and-suspenders
check at `:3138-3140`). This confirms the plan's noted synergy: for VLM sessions, KV
growth (Part 1) is the *only* graceful long-context mechanism, since shift/reuse are
both off unconditionally.

### 7. Determinism oracle check — **not verified this pass**

No GPU and no local model files (this is a source-only checkout) were available to run
an actual temp-0 A/B (MTP-on vs MTP-off, text-only) in this session. This item is
**open** and blocks Phase 2 (P1 milestone) of the plan, which requires it as the test
oracle. Needs to run on real hardware with a qwen35moe-family GGUF that has MTP heads.

---

## Assessment against Part 3's Phase 1 acceptance bar

> "every guard located; a one-page written mechanism for 'what exactly breaks if we just
> delete the guard' — derived from code, not from this plan's hypothesis."

- Guards located: **none exist** (item 1) — the plan's framing doesn't match this
  codebase's history; nothing needs deleting.
- Mechanism of breakage: **found and localized** to one 9-line block,
  `common/speculative.cpp:1359-1362`, plus a secondary follow-on bug in the `begin()`
  sanity check (`:1345`, text-token-count vs. position-slot-count mismatch, item 5).
  Both are narrow, well-understood gaps, not a deep architectural conflict.
- The plan's fallback ("if the hypothesis is wrong at a deep level... prime MTP state via
  a tiny recompute of the last text tokens") is **not needed** — the fix implied by item 5
  is much smaller: make `process()` handle (or explicitly invalidate-and-flag-for-reprime,
  per Part 3 Phase 3 §1's own preferred design) the embedding-batch case instead of
  silently no-op'ing, and fix the `begin()` comparison to use position counts instead of
  text-token counts.

## Open items before Phase 2 (P1) can start

1. Runtime determinism oracle (item 7) — needs GPU + a qwen35moe-family GGUF with MTP
   heads (e.g. an `unsloth/Qwen3.6-35B-A3B` quant, or any smaller model in that family
   the repo's MTP support already covers) to confirm MTP-on == MTP-off at temp 0 in
   text-only mode first, before touching the mmproj case at all.
2. Given item 1's finding (no guard exists, both flags already combine without error),
   **P1 as scoped in the plan (relax-a-guard milestone) is likely a no-op** on this
   codebase — text-only-through-mtmd-path may already work today. Worth an empirical
   check before writing any P1 code: load with both `--mmproj` and `--spec-type
   draft-mtp`, send a text-only request, see what happens now.
3. Real fix work starts at what the plan calls P2 (images in the prompt): teach
   `process()` in `common/speculative.cpp` to either (a) run its catch-up decode against
   the image's hidden-state rows too, or (b) per Part 3 Phase 3 §1's simpler v1 design,
   explicitly mark draft state invalid when an embedding batch is seen and re-prime with
   a cheap unspeculated decode step at the start of the next text generation — this
   second option requires no changes to the vision/embedding path at all, only a flag
   check in `process()` plus a one-time warm-up call site in the server's generation
   loop. **Implemented below (option (b)).**

---

## Implementation: item 5's fix (invalidate + re-prime)

Applied to `common/speculative.cpp`, `common_speculative_impl_draft_mtp`. Chose option
(b) over (a) deliberately: (a) would require trusting that `t_h_nextn`/`embd_nextn`
capture in `llama_context::decode()`/`encode()` (which is architecture-generic and
looked plausible from static reading — the capture site itself doesn't branch on
token-vs-embd input) is *also correct* for every model's graph when the input is raw
image embeddings rather than token ids, for every arch this MTP mode covers. That's
exactly the kind of thing the plan calls out as unsafe to assume without a runtime
check (§8, "silently degraded... not a crash"). Option (b) only touches token-batch
plumbing that was already proven correct, so it doesn't require that assumption.

**Changes** (all in `common_speculative_impl_draft_mtp`):
- New member `std::vector<bool> valid` (one per seq), initialized `true` in the ctor —
  matches prior behavior (text-only sessions were already fine).
- `process()`: the old unconditional skip (`// TODO: how to make it work with vision
  tokens?` → bare `return true`) for embedding batches is replaced with: mark every seq
  present in the batch as `valid[seq_id] = false`, then return. Still skips the ctx_dft
  decode exactly as before — behavior for the image chunk itself is unchanged.
- `process()`, token-batch path: computes `need_reprime` (true if any seq in this batch
  is currently invalid). The existing catch-up decode block (`if (!is_mem_shared)`) now
  also requires `!need_reprime` to run — for the whole batch, not per-seq, specifically
  to avoid slicing rows out of the "shift the tgt embeddings right by one position"
  block, which assumes tight row correspondence between `h_tgt` and the batch sent to
  `ctx_dft`; partial-row filtering there was judged too easy to get subtly wrong to do
  without a runtime test. Conservative for mixed-validity multi-seq batches (skips one
  extra seq's catch-up unnecessarily in that rare case); exact for the plan's actual
  scope (`-np 1`, single seq).
- `process()`, capture tail (unconditional, always ran before): now also sets
  `valid[seq_id] = true` after refreshing `pending_h[seq_id]` — this tail was already
  reading only from `ctx_tgt`'s `h_nextn` output, independent of whether the ctx_dft
  catch-up ran, so it's the correct place to revalidate.
- `draft()`: added a `!valid[seq_id]` skip at the top of the per-seq loop, before
  `pending_h[seq_id]` is used to seed the injected embedding row. A skipped seq
  contributes no result; the existing wrapper (`common_speculative_draft()` in the same
  file) already treats "no result produced" as "fall back to one un-speculated decode"
  for that step — which is exactly the "run the first decode step un-speculated to
  re-prime" behavior Phase 3 §1 asks for, achieved without adding a new call site.

**Net effect:** after an image (or audio) chunk, MTP stops drafting for the affected
seq(s) for exactly one generation step (an ordinary, correctness-safe decode happens
instead), then resumes drafting from a `pending_h` that was captured from the target's
real post-image hidden state — never from a stale pre-image row. No MTP draft-model KV
cell is ever written using a hidden state that doesn't correspond to its position.

**Verified:** `common/speculative.cpp` (and the rest of `common/`) compiles cleanly —
`cmake --build build --target llama-common` — no new warnings. **Not verified:** no
GPU/model available this session, so the actual runtime behavior (does drafting resume
correctly, does the temp-0 identity hold, does acceptance rate look normal) is
unconfirmed. That's item 7's determinism oracle plus a real image+MTP smoke test — still
the blocking open item before this can be called done. The `begin()` heuristic warning
mismatch noted above (item 5) was deliberately left alone: it's a non-blocking log-only
warning, and fixing it isn't needed for the correctness fix above.

---

## Part 1 Phase 1 discovery answers (dynamic context / `llama_set_n_ctx`)

Read-only pass, no code changed. Base commit unchanged (`e3546c7` + the two commits
above). Covers the plan's ten discovery questions, using current symbols — several have
moved since the plan was written (see corrections inline).

### 1. Memory class hierarchy

Abstract interface: `llama_memory_i` in `src/llama-memory.h:73-127`. No `resize()`
member exists today (confirmed repo-wide: `grep -r llama_set_n_ctx` → 0 hits;
`llama_memory_i` has no resize-shaped virtual). This part of the plan's core gap is
real.

Concrete classes — **more than the plan names**:
- `llama_kv_cache` (`src/llama-kv-cache.h/.cpp`) — plain unified KV cache.
- `llama_kv_cache_iswa` (`src/llama-kv-cache-iswa.h/.cpp`) — SWA wrapper, holds a base
  cache + a window-sized SWA cache.
- `llama_kv_cache_dsv4` (`src/llama-kv-cache-dsv4.h/.cpp`) — **not in the plan**;
  DeepSeek-V4-family variant, selected when `arch == LLM_ARCH_DEEPSEEK4`
  (`src/llama-model.cpp:2180-2196`).
- `llama_kv_cache_dsa` (`src/llama-kv-cache-dsa.h/.cpp`) — **not in the plan**; selected
  for `LLM_ARCH_DEEPSEEK32` (`src/llama-model.cpp:2050-2066`), outside the
  hybrid/recurrent branch entirely (its own top-level `switch` case).
- `llama_memory_hybrid` (`src/llama-memory-hybrid.h/.cpp`) — attention + recurrent
  children, matches the plan.
- `llama_memory_hybrid_iswa` (`src/llama-memory-hybrid-iswa.h/.cpp`) — **not in the
  plan**; hybrid models that also have SWA (attention child is itself an iSWA cache).
  Selected whenever a hybrid arch has `hparams.swa_type != LLAMA_SWA_TYPE_NONE`
  (`src/llama-model.cpp:2111-2130`). Qwen3.5/3.6 does not currently hit this branch
  (goes through plain `llama_memory_hybrid`, `:2131-2150`), but a future SWA+hybrid
  arch would need this covered by `resize()` delegation too.
- `llama_memory_recurrent` (`src/llama-memory-recurrent.h/.cpp`) — pure recurrent
  models, matches the plan.

Selection: `llama_model::create_memory()`, `src/llama-model.cpp:2027-2273`. Key facts
not anticipated by the plan:
- Selection now depends on **`params.ctx_type`**, an enum including
  `LLAMA_CONTEXT_TYPE_MTP` — directly relevant to item 10, see below.
- For Qwen3.5/3.6 (`LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE`) specifically: normal
  inference (`ctx_type != MTP`) constructs `llama_memory_hybrid` with per-layer
  `filter_attn`/`filter_recr` callbacks selecting attention vs. recurrent layers via
  `hparams.is_recr(il)` (`:2102-2109`). **When `ctx_type == LLAMA_CONTEXT_TYPE_MTP`**,
  the same arch instead takes the plain-`llama_kv_cache` branch, bypassing the hybrid
  path entirely (`mtp_on_hybrid_qwen35` flag, `:2071-2076`, consumed at `:2087` and
  `:2168-2170`) — see item 10.

### 2. KV tensor layout

Per-layer tensors created in `llama_kv_cache`'s constructor, `src/llama-kv-cache.cpp:231-232`:
```cpp
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```
Both K and V are declared with the **same** nominal shape `[n_embd_gqa, kv_size,
n_stream]` regardless of `v_trans` — `kv_size` is dim 1 for both. This looked at first
like it might simplify the plan's §4.2 V-copy story, but it doesn't:

- `n_embd_v_gqa` itself differs by `v_trans`: `!v_trans ? hparams.n_embd_v_gqa(il) :
  hparams.n_embd_v_gqa_max()` (`:208`, tagged `[TAG_V_CACHE_VARIABLE]`) — when
  transposed, V's per-layer embd width is padded to the model's max across layers, not
  the per-layer value.
- More importantly, **the physical write pattern differs by `v_trans`, not just the
  declared tensor shape**. `cpy_v()` (`src/llama-kv-cache.cpp:1330-1384`):
  - `!v_trans` branch (`:1349-1362`): reshapes to `[n_embd_gqa, n_tokens]` and does one
    `ggml_set_rows` into `v` directly — an ordinary row-major write, `kv_size` is a
    genuine contiguous-prefix-growable dimension here.
  - `v_trans` branch (`:1365-1383`): reshapes `v` itself into a **flat
    `[1, ggml_nelements(v)]`** view and scatters individual scalars via `ggml_set_rows`
    with per-element `v_idxs` (built by `build_input_v_idxs`, `:1396-1409`, which sizes
    `v_idxs` to `n_tokens * n_embd_v_gqa_max()` — one index per scalar, not per row).
  - `get_v()` (`:1263-1293`) confirms the resulting physical layout for `v_trans`: the
    view built for attention has `n_kv` (position) as **dim 0 with element-size
    stride**, i.e. positions are the fastest-varying, contiguous index for a fixed
    (head, embd-dim) pair — the reverse of `!v_trans`'s layout. This is exactly the
    "position is the row dimension" case the plan's §4.2 describes, **confirmed by
    tracing the actual index math, not assumed**: growing `kv_size` for a `v_trans`
    cache is not a contiguous-prefix copy; old data must move row-by-row (per
    head×embd-dim) to new stride-separated offsets. Plan's §4.2 V-transposed handling is
    correct and necessary as written.
- `n_pad`: cache size must be a multiple of `n_pad` (`GGML_ASSERT(kv_size % n_pad ==
  0)`, `:98`); `n_pad` is typically `1` for most callers seen in `create_memory` but the
  constructor accepts it as a parameter — a `resize()` needs to round its target up to
  the same `n_pad` the cache was constructed with (accessible via a getter — confirm
  before implementing).
- `n_stream`: `unified ? 1 : n_seq_max` (constructor init list, not directly re-quoted
  above but consistent with `src/llama-kv-cache.h:239`'s doc'd default of `1`). Per-buffer
  grouping: one `ggml_context`/buffer per distinct `(buft)` combination across all
  layers (`ctx_map`, `src/llama-kv-cache.cpp:274-293`) — i.e. already coarse-grained
  (one buffer per backend-buffer-type, not per layer), matching what Part 2's discovery
  (A.3) will separately need to know.

### 3. Cell metadata

`llama_kv_cells` in `src/llama-kv-cells.h:32-`. Per-cell state: `pos` (position, `-1` =
empty), `ext` (2D x/y position, for M-RoPE), `shift`, `seq` (bitset of owning
sequences). Bookkeeping sets: `used` (indices of non-empty cells), `seq_pos[s]` (per-seq
position multiset).

`resize(uint32_t n)` **already exists on `llama_kv_cells` itself**
(`src/llama-kv-cells.h:63-70`):
```cpp
void resize(uint32_t n) {
    pos.resize(n); ext.resize(n); shift.resize(n); seq.resize(n);
    reset();
}
```
**This is not what the plan needs** — it calls `reset()` unconditionally, wiping all
existing cell metadata. A real `llama_kv_cache::resize()` cannot call this directly; it
needs a variant that extends the vectors with empty cells *without* resetting the
existing prefix (`std::vector::resize` alone already preserves the existing prefix and
default-constructs the new tail — the bug would be the trailing `reset()` call, not the
`.resize()` calls themselves). Worth either adding a `grow(uint32_t n)` method here that
skips the `reset()`, or having `llama_kv_cache::resize()` resize+repopulate metadata for
the appended range only.

### 4. Decode failure path

`llama_context::decode()`, `src/llama-context.cpp:1750-1793`. On
`mctx->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE` (`:1768-1783`):
- **First** retries once via `memory_update(true)` (an "optimize" pass — almost
  certainly defrag; not yet traced further) — only if that *doesn't* resolve it does it
  fall through.
- Falls through to `LLAMA_LOG_WARN("failed to find a memory slot...")` and **`return
  1`** (`:1782`) — confirms the plan's guess (`return code 1`) exactly.

This is the auto-grow hook point, matching the plan, but note there's already a
pre-existing "try to fix it without failing" step (the optimize/defrag retry) that
growth would become the *second* fallback after, or would need to run before/instead of
depending on desired precedence (plan doesn't consider this pre-existing retry — should
compose with it, not replace it).

### 5. Graph reservation — simpler than the plan assumed

`graph_reserve()`: `src/llama-context.cpp:2324-`. Worst-case sizing driven by
`sched_reserve()` (`:542-`), which computes `n_tokens = std::min(cparams.n_ctx,
cparams.n_ubatch)` (`:556`) — **reads `cparams.n_ctx` fresh**, not a value cached once
at construction.

**Key finding: `sched_reserve()` is already lazy and re-triggerable**, gated by a
`sched_need_reserve` flag (`src/llama-context.h:346`, default `true`). `sched_reserve()`
itself is a no-op unless the flag is set (`:543-545`), and is called unconditionally at
the top of every `decode()`/`encode()` (`:1743`, and the analogous spot in `encode()`).
**Six existing call sites already set `sched_need_reserve = true` after mutating
something that changes graph shape**: `set_embeddings_layer_inp` (`:1138`),
`set_causal_attn` (`:1154`), `set_sampler` (`:1184,1203,1212,1222`),
`set_adapters_lora` (`:1242`), `set_adapter_cvec` (`:1281`).

**This directly answers the plan's open question in §5.2 step 3**: no new "explicit
re-reserve" plumbing is needed. `llama_set_n_ctx`'s implementation should simply set
`sched_need_reserve = true` after updating `cparams.n_ctx`/`cparams.n_ctx_seq`, exactly
like the six precedents above, and the existing machinery re-reserves automatically
(synchronously, at the very next `decode()` call — verified via `synchronize()` at
`:551` and the guarded rebuild that follows). This simplifies Part 1 §5.2 meaningfully:
step 3 becomes a one-line flag set, not new machinery, *provided* growth always happens
at a point where a decode call follows shortly after (true for both the auto-grow hook
in `decode()` itself and the server's proactive pre-decode growth).

### 6. What reads `cparams.n_ctx`

- **Constructor-only, must be replicated by `resize()`**: the normalization block at
  `src/llama-context.cpp:261-277` — pads `n_ctx` to 256, then derives `cparams.n_ctx_seq`
  (either `= n_ctx` if `kv_unified`, or `n_ctx / n_seq_max` padded to 256 otherwise,
  with a warning + `n_ctx` rounding-down if not evenly divisible). **This is the value
  actually passed to `create_memory()` as `attn_kv_size`/`kv_size`**
  (`src/llama-model.cpp:2119,2137,2191,2221,2239,2258` all pass `cparams.n_ctx_seq`, not
  `cparams.n_ctx`, to the memory constructors) — so `llama_set_n_ctx` must recompute
  `n_ctx_seq` with this exact formula before calling `memory->resize()`, not just bump
  `n_ctx`.
- **Read fresh per call, safe already**: `sched_reserve()`'s `n_tokens` (item 5, above);
  `llama_context::n_ctx()` accessor (`:712-713`, trivial passthrough).
- **Training/optimization path** (`llama_opt_*`, `:3233-3421`) also reads `n_ctx` /
  `n_ctx_train`, but this is the LoRA-training code path, not inference decode — out of
  scope for this plan, noted but not investigated further.
- Did not find any rope-scaling-init consumer of `cparams.n_ctx` itself (rope scaling
  reads `cparams.n_ctx_orig_yarn`, a separate, `params.yarn_orig_ctx`/`hparams`-derived
  value, `:115-117`, unrelated to the *current* `n_ctx` and not expected to change on
  growth).
- Server-side fixed-at-init consumer: see item 9.

### 7. State serialization

`llama_kv_cache::state_write`/`state_read`, `src/llama-kv-cache.cpp:1957-2027` (data
format) using `cells.size()` (i.e. current `kv_size`) only as a **bound to check against
on read**, not something written and restored as a target size:
- `state_read` (and the seq-scoped variant) both check `cell_count > cells.size()` and
  **error out** (`:2270`, `:2335-2336`, `"not enough cells in kv cache to restore
  state"`) rather than resizing to fit.
- Confirms the plan's assumption directly: state save/load encodes *used* cell data
  (positions + content for the used range), not `kv_size` itself. Loading a
  small-context save into a larger pre-allocated context already works (bigger
  `cells.size()` trivially satisfies the bound check) — this remains usable both as a
  test oracle and as the documented fallback copy path per the plan.

### 8. Interacting features

- **Context shift**: implemented in `tools/server/server-context.cpp`'s `pre_decode()`
  (`:2801-2834`), gated by `slot.prompt.n_tokens() + 1 >= slot.n_ctx` — **this is the
  exact spot Part 1 §5.3's growth-vs-shift precedence policy needs to hook into**: today
  it goes straight to shifting; growth needs to be tried first (per the plan's own
  policy: grow first, shift only once `ctx_max` is reached). A second, earlier gate at
  `:1860` (in `process_token()`) stops generation early
  (`STOP_TYPE_LIMIT`/`slot.truncated`) when `!params_base.ctx_shift &&
  slot.prompt.n_tokens() + 1 >= slot.n_ctx` — this is actually the **first** place
  growth needs to be tried (today it just gives up when shift is off), with `:2801`'s
  copy being effectively dead code once shift is off (comment at `:2807-2808` says as
  much: "should never get here").
- Confirmed disallowed with multimodal (already documented in the Part 3 section above):
  `ctx_shift` and `n_cache_reuse` both force-disabled when `mctx` is set
  (`server-context.cpp:1216-1224`).
- `--defrag-thold` is **deprecated** in this tree (`common/arg.cpp:2283-2288`,
  "no longer necessary to specify") — defrag now appears to run automatically
  (presumably the `memory_update(true)` "optimize" pass from item 4). Plan doesn't
  mention this; worth confirming defrag-after-growth doesn't need special handling
  (probably fine, since it's cell-metadata-level and orthogonal to buffer size).
- `--ctx-checkpoints` (`n_ctx_checkpoints`, `common/arg.cpp:1462-1466`): used by the
  server's speculative-decoding checkpoint mechanism (already seen extensively in the
  Part 3 discovery above, `slot.spec_ckpt`) — checkpoints capture/restore KV state via
  the same `state_write`/`state_read` machinery from item 7, so should be unaffected by
  growth as long as growth happens between checkpoint operations, not concurrently
  (matches Part 1's general "only at the between-decode sync point" rule).
- `--kv-unified` (`cparams.kv_unified`): changes both `n_stream` (item 2) and the
  `n_ctx_seq` formula (item 6) — `resize()` must handle both branches of that formula
  identically to how the constructor does.

### 9. Server slot model

`tools/server/server-context.cpp:1253-1312`. `slot.n_ctx` is computed **once**, at
server startup, from `llama_n_ctx_seq(ctx_tgt)` (`:1253`, capped by
`n_ctx_train` at `:1254-1256`), then copied into every slot (`:1307`) — **fixed at
init**, exactly the kind of value Part 1 §7.2 flags as needing a refresh after growth
("refresh `slot.n_ctx` and anything cached from it"). Every length-limit check found
above (item 8: `:1860`, `:2805`) reads `slot.n_ctx`, not `llama_n_ctx_seq()` fresh — so a
`llama_set_n_ctx()` call alone, without also updating `slot.n_ctx` for every slot
server-side, would grow the underlying cache but leave the server still enforcing the
old limit. This is a concrete, easy-to-miss integration point for Phase 5.

### 10. MTP speculative decoding's own KV cache — separate object, separate resize needed

Directly answered by item 1's finding: `LLAMA_CONTEXT_TYPE_MTP` is a first-class
`ctx_type` value threaded through `create_memory()`. For the plan's target arch
(qwen35/qwen35moe), an MTP-mode context (`ctx_dft` in the server/`common/speculative.cpp`
sense — see the Part 3 discovery above) gets `mtp_on_hybrid_qwen35 = true`
(`src/llama-model.cpp:2073-2075`) and is constructed as a **plain `llama_kv_cache`**
(`:2250-2266`, filtered to `il >= hparams.n_layer()` at `:2169`), entirely separate
from the main hybrid memory object used by the target context (`ctx_tgt`). This
independently confirms, from the create_memory side, what the Part 3 discovery already
established from the `common/speculative.cpp` side (`ctx_dft`/`mem_dft` are a fully
separate `llama_context`/`llama_memory_i` pair, not a view into `ctx_tgt`'s memory, for
every mode except gemma4's `is_mem_shared`).

**Consequence for Part 1**: `resize()`/`llama_set_n_ctx()` applied to `ctx_tgt` does
**not** cover `ctx_dft`'s cache. Two options, per the plan's own item 10 framing:
(a) grow `ctx_dft` in lockstep whenever `ctx_tgt` grows (the server owns both contexts,
so this is a call-site change in whatever wraps `llama_set_n_ctx`, not a `resize()`
change), or (b) leave `ctx_dft`'s cache at a fixed (smaller) size, since per the Part 3
discovery `common_speculative_impl_draft_mtp` only ever needs `mem_dft` to hold recent
positions relevant to drafting/verification, not the full conversation history — a
much smaller fixed budget than `ctx_tgt`'s grown size might already be sufficient,
**but this needs confirming**: `begin()` (`common/speculative.cpp:1336-1352`) computes
`pos_max` from `mem_dft` and compares it against the *prompt length*, implying it does
expect `mem_dft` to track roughly the same position range as `ctx_tgt`, at least once
per generation start — leans toward (a) being necessary, not just safer. **Not settled
this pass**; needs a decision before Part 1 M3 (hybrid coverage) can claim MTP sessions
are handled, and ties directly into Part 2 §8.6/Part 3's obligation to keep the MTP
cache in sync with growth (both plans already flag this as an open item; this discovery
locates exactly where the two context objects diverge).

---

## Assessment against Part 1 Phase 1's acceptance bar

> "all nine questions answered in NOTES.md with references; plan corrections listed."

(The plan numbers ten items in the text but calls the bar "nine questions" — likely a
copy-paste slip in the source plan; all ten are answered above regardless.)

Corrections vs. the plan, summarized:
- More concrete memory classes exist than named (`llama_kv_cache_dsv4`,
  `llama_kv_cache_dsa`, `llama_memory_hybrid_iswa`) — item 1.
- `llama_kv_cells::resize()` already exists but resets everything; not directly usable
  — item 3.
- Graph re-reserve is **already lazy/automatic** via `sched_need_reserve` — significant
  simplification of plan §5.2 step 3 — item 5.
- The value actually sized by `create_memory()` is `cparams.n_ctx_seq`, a *derived*
  value with its own padding/division formula, not `cparams.n_ctx` directly — item 6,
  easy to miss.
- Two separate length-limit checks exist server-side (`process_token()` and
  `pre_decode()`), not one — item 8/9.
- MTP's draft context uses a fully separate memory object even for the plan's own
  target arch, confirmed independently from both the model-construction side (this
  pass) and the speculative-decoding side (Part 3 pass) — item 10, and this is a real
  open design question, not just a "cover it too" checkbox.

## Open items before Part 1 Phase 2 (`resize()` implementation) can start

1. Whether `ctx_dft` (MTP) should grow in lockstep with `ctx_tgt` or stay fixed-size —
   item 10 above — needs a decision, likely lockstep given `begin()`'s `pos_max` vs.
   prompt-length comparison.
2. Confirm what `memory_update(true)` (the "optimize" retry before `decode()` gives up,
   item 4) actually does — presumably defrag — and decide growth's precedence relative
   to it (try optimize first as today, then grow? or grow first since it's likely
   cheaper than a defrag pass at large sizes? not measured).
3. Find or add a getter for the `n_pad` a given `llama_kv_cache` instance was
   constructed with (item 2), so `resize()` can round its target consistently.
4. No GPU available this session — none of the above was exercised at runtime (no
   `resize()` exists yet to run). Phase 0's baseline runs (build + `ctest` + reference
   token outputs) are still outstanding before Phase 2 work starts, per the plan's own
   sequencing.
