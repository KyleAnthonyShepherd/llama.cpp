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
   loop.
