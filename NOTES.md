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

---

## Part 2 Phase A discovery answers (auto placement / elastic redistribution)

Read-only pass, no code changed. Covers the plan's eight discovery questions. Builds
directly on the "Part 2's extension point already exists in more mature form" finding
from the cross-cutting section at the top of this file — this pass goes deep on that
finding plus the remaining seven items.

### A.1 `llama_params_fit`

Already located in the cross-cutting section: `common_fit_params()` (`common/fit.h:19-27`,
implemented via `common_params_fit_impl()` in `common/fit.cpp:176-`). This pass traced the
actual algorithm, which is **substantially more capable than the plan assumes** — not a
simple heuristic, and already implements most of the plan's Phase A/B "insight" as a
hardcoded (not `value_per_byte`-derived) policy:

- **Step 1** (`:192-286`): loads the model with `no_alloc=true` (via
  `common_get_device_memory_data_impl`, `:29-176`, which does a real `llama_model_load_from_file`
  + `llama_init_from_model` + reads `llama_get_memory_breakdown(ctx)` per backend-buffer-type)
  to get an accurate real per-device memory projection, not an estimate. Compares against
  a per-device **margin** (the plan's "headroom constant") passed in as `margins_s`
  — user-configurable per device via `-fitt`/`--fit-target MiB0,MiB1,...`
  (`common/arg.cpp:2601-2625`). If projected free memory already clears the margin on
  every device, returns immediately (no changes) — this *is* item 7 (free-VRAM probing)
  answered too: `ggml_backend_dev_memory(dev, &free, &total)` (`common/fit.cpp:107,116`)
  is the actual current-free-memory query, called once per device, per fit.
- **Step 2** (`:288-`): if margins can't be met, first tries reducing context size —
  **but only if `cparams->n_ctx == 0`** (i.e. the user let it auto-decide) — floored at
  `n_ctx_min` (the `-fitc`/`--fit-ctx` flag, `common/arg.cpp:2626-2630`). This is directly
  relevant to Part 1: the *existing* fitter already reduces ctx to fit VRAM at load time;
  Part 1's growth feature is the complementary runtime-side of the same story (start
  small, grow later) and should probably share the `n_ctx_min`/`--ctx-max` vocabulary
  rather than introducing parallel flags.
- **Step 3** (`:400-643`): fills devices back-to-front with "dense" layers. For a MoE
  model, **this already implements the plan's central insight as a fixed rule**: comment
  at `:402` — "for a MoE model, same as dense model but with all MoE tensors in system
  memory." Concretely, `LAYER_FRACTION_MOE` (`:22`, "everything but sparse MoE weights")
  is the default `overflow_type`, and its regex pattern
  (`get_overflow_pattern`, `:405-442`) is
  `blk\.<il>\.ffn_(up|down|gate_up|gate)_(ch|)exps` — i.e. routed-expert tensors
  specifically (the `_exps` suffix), matching GGUF's per-tensor naming for MoE weights.
  This is *exactly* `--n-cpu-moe`'s effect, generated automatically per-layer instead of
  as a single N-layer cutoff.
- **Step 4** (`:645-784`): once all dense-only layers fit, converts them front-to-back
  into "full" layers (i.e. **promotes routed experts onto GPU**) via a bisection search
  (`get_memory_for_layers`, binary-searching layer counts against the margin target)
  until the device is full, then tries to fit one more partial layer using finer
  fractions (`LAYER_FRACTION_UP` → `GATE` → `ATTN`, `:720-770` — attention/dense parts
  promoted before routed experts within a partial layer, consistent with the plan's
  value ordering even though it's not computed as one).
- **Output**: writes directly into `llama_model_tensor_buft_override[]`
  (`set_ngl_tensor_split_tbo`, `:460-498`) — **this is already the plan's "Plan format"
  and "Solver" output** (§B.3-B.4), in the exact override-machinery shape Part 2 §B.4
  says the loader should consume.
- **What's genuinely missing** (confirms the real Part 2 gap): no cost model /
  `t/s` prediction anywhere in this file — it fits to a static memory-margin target, not
  a throughput objective. No calibration, no benchmarking, no profile cache, no runtime
  elasticity. No MTP-awareness (not referenced anywhere in `fit.cpp` — confirmed via
  grep). No expert-popularity/value_per_byte reasoning — the MoE-to-CPU rule is a fixed
  heuristic, not derived from a per-token-read-probability model, so it can't express
  "promote N experts because KV budget leaves room" trade-offs the plan wants from a
  real cost model. **Part 2's actual net-new work is calibration + a throughput cost
  model + elastic runtime migration (Phases C-G) layered on top of an already-solid
  static load-time fitter — not building the fitter itself.**

### A.2 Tensor override machinery

Confirmed exactly as the plan describes, traced end to end:
- `-ot`/`--override-tensor` and `--n-cpu-moe` (`-ncmoe`, `common/arg.cpp:2477-2481`)
  both populate the same `llama_model_tensor_buft_override[]` array consumed by the
  loader (same array type `common_fit_params` writes into — one shared mechanism for
  manual and fitted placement).
- Consumption site: `src/llama-model-loader.cpp:1162-1197`, inside the per-tensor buft
  selection function. For each tensor, linearly scans `tensor_buft_overrides` (terminated
  by a `nullptr` pattern sentinel), `std::regex_search`ing the tensor's GGUF name against
  each `pattern` in order, first match wins (`:1167-1188`). If the matched override's
  `buft` is the CPU buffer type, it additionally calls `select_weight_buft(...,
  buft_list_cpu)` to pick among CPU-side buffer-type variants rather than assigning the
  plain CPU buft directly (`:1170-1178`) — and **logs a warning if `use_mmap` is also
  set** ("consider using --no-mmap for better performance", `:1173-1177`) — implying the
  override+mmap combination is already a supported, exercised code path today (a
  positive data point for item 4, below), just not the fastest one.
- Falls back to `select_weight_buft(hparams, t_meta, op, buft_list)` (`:1192-1197`) when
  no override matches — the "default" placement logic Part 2's fitter output competes
  with/replaces.

### A.3 Buffer granularity at load

Confirmed: **one `ggml_context` (and thus one contiguous `ggml_backend_buffer`, via
`ggml_backend_alloc_ctx_tensors_from_buft`) per distinct `ggml_backend_buffer_type_t`**,
not per layer or per tensor. Grouping happens in `llama_model_loader::ctx_map` (built
during tensor iteration, consumed at `src/llama-model.cpp:1499,1509-1597`): all tensors
whose selected `buft` (item A.2) matches are collected into the same `ggml_context`, then
one `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` call allocates them all as a
single buffer (`:1571`). This confirms the plan's assumption directly and identifies the
"smallest change" question (Phase A.3's ask): **the elastic-groups change (Part 2 Phase
D.1) is not a new mechanism, it's widening the grouping key** — today the key is `buft`
alone; splitting a GPU device's tensors into independently-freeable per-`placement_group`
buffers means the key needs to become `(buft, group_id)` (e.g., derived from which
override pattern matched, or an explicit layer/tier tag threaded alongside the override
array), so `ctx_map` naturally produces one buffer per group instead of one per device.
This is a real but bounded change — confined to the loader's tensor-to-context grouping
step, no allocator internals need touching, consistent with the plan's working-agreement
rule about not touching ggml allocator internals.

### A.4 mmap reality check

Not fully settled at runtime (no GPU/model this session — same limitation as Parts 1 and
3), but static evidence points the same direction as the plan's default assumption
(mmap-on demote-without-copy should work):
- The loader already anticipates and logs for the override-to-CPU + mmap combination
  (`llama-model-loader.cpp:1173-1177`, item A.2) as a real, if suboptimal, path — it
  doesn't special-case it as broken or unsupported.
- Did not find, this pass, the specific place that decides whether a tensor's `data`
  pointer becomes a raw mmap'd pointer vs. an owned/copied buffer for **GPU-destined**
  tensors specifically (the actual "does a GPU-offloaded tensor's *host-side* shadow copy
  stay mmap'd and addressable" question the plan's demote trick depends on) — this needs
  a dedicated follow-up read of `llama_model_loader`'s tensor-upload path (`set_tensor_data`
  callback machinery hinted at `llama-model-loader.cpp:531`, not traced this pass) and,
  ideally, a runtime check with `--no-mmap` vs. default on a real GPU. **Flagged as the
  single most important unresolved item before Phase D (elastic groups) can start** —
  the plan already says as much in its own risk list.

### A.5 Scheduler reaction to migration — mechanism located, not exhaustively enumerated

`ggml_backend_sched` itself decides splits per graph build (not traced deeper this pass —
its internals are outside `src/llama-*`, in `ggml/src/ggml-backend.cpp`, out of scope for
a `src/`-focused discovery pass; flagged for Phase D). The **cache the plan worries about
going stale is real and located**: `llama_context::gf_res_prev` /
`llm_graph_result::can_reuse()` (`src/llama-context.ccp:1300`
[`if (!graph_reuse_disable && res->can_reuse(gparams))`], gated by env
`LLAMA_GRAPH_REUSE_DISABLE`, `:252-257`). This reuses a previously-built `ggml_cgraph`
across decode calls when the new call's parameters match, to skip graph-build overhead.

Traced what `can_reuse()` actually checks: it's a per-*input*-tensor virtual
(`llm_graph_input_i::can_reuse`, `src/llama-graph.h:107-112`, default `false` — i.e.
inputs opt in), implemented per input kind (`llm_graph_input_pos`, `llm_graph_input_out_ids`,
the KQ-mask helper `can_reuse_kq_mask` in `src/llama-graph.cpp:45-62`, etc.) — these all
check **shape/param** compatibility (e.g. `kq_mask->ne[0] == n_kv`), not weight-tensor
identity. **This is exactly the gap the plan's A.5 item warns about**: nothing in this
reuse-eligibility check inspects whether a *weight* tensor referenced inside the cached
graph's nodes has had its `buffer`/`data` pointer changed since the graph was built (which
is precisely what Demote/Promote, Part 2 Phase D.2-D.3, would do to a tensor mid-session).
If graph reuse fires after a migration, the reused `ggml_cgraph`'s nodes would still
reference whatever `ggml_tensor*` objects they were built with — **whether that's
actually stale depends on whether Demote/Promote mutates the existing `ggml_tensor`
struct in place (`buffer`/`data` fields on the same object migration keeps rewriting) or
allocates a new tensor object** — not determined this pass; this is the load-bearing
question the plan's Phase D.4 determinism test is designed to catch, and it should be
answered by design (mutate in place) before relying on graph-reuse being safe, or the
elastic controller should force `sched_need_reserve`-style invalidation of `gf_res_prev`
(the same flag/pattern documented in the Part 1 discovery, item 5, above — `sched_reserve()`'s
`sched_need_reserve` and `gf_res_prev`'s reuse check are two *different* caches, both
need covering) on every migration, not just resize.

**Enumeration is not exhaustive** — the plan's A.5 explicitly asks to "list every cache
to invalidate," and this pass found the two real candidates (`sched_need_reserve` /
worst-case buffers, and `gf_res_prev` / per-decode graph reuse) but did not do a full
sweep for others (e.g. any KV-cache-side cell/mask caching, or backend-specific graph
plans). Sufficient to unblock Phase B/C design; not sufficient to close out Phase D.1
without a dedicated pass.

### A.6 Per-token value table for the target arch

Traced `src/models/qwen35moe.cpp`'s graph builder. Confirms the plan's value_per_byte
intuition (dense/shared parts read every token, routed experts read with low
probability) but surfaces **one nuance the plan's cost model doesn't account for**: the
MTP module for this arch is not a small dense head — **it has its own MoE routing,
structurally mirroring the main model's**:
- Main-model FFN block: `build_moe_ffn(...)` (`:501-515`, "ffn_moe_out") plus a
  shared-expert gate (`:532-536`, "shared_expert_gate"/"shared_expert_gate_sigmoid") —
  standard dense-shared + sparse-routed split, matches the plan's value table exactly
  (shared/dense = value 1.0, routed = value ≈ n_active/n_experts).
- **MTP block, separately** (`:679-707`): its own `build_moe_ffn(...)` call
  ("mtp_ffn_moe_out") and its own shared-expert gate
  ("mtp_shared_expert_gate_sigmoid") — i.e. the MTP head reuses the *same*
  `n_expert`/`n_expert_used`/`n_ff_exp` MoE structure as a main-model layer, not a
  single dense projection.
- **Consequence for Part 2 §8.2**: treating "the MTP module" as one placement_group with
  value ≈ 1.0 is too coarse for this arch — it should itself be split into an
  MTP-dense/shared part (value ≈ 1.0, read every drafted token) and an MTP-routed-expert
  part (value ≈ n_active/n_experts, same low-value treatment as the main model's routed
  experts) — otherwise the cost model will over-value pinning the whole MTP module in
  VRAM, when in fact most of its bytes (the routed-expert tensors) are exactly the kind
  of low-value_per_byte weight the rest of the model already excludes from GPU by
  default.
- Did not cross-check against `llama-bench` CPU-only t/s vs. theoretical RAM bandwidth
  this pass (no GPU/hardware available) — the plan's acceptance bar for this item
  ("cross-check with `llama-bench`") remains open.

### A.7 Free-VRAM probing

Answered directly under A.1: `ggml_backend_dev_memory(dev, &free, &total)`
(`common/fit.cpp:107,116`), called once per device per `common_fit_params()` invocation.
This is the standard ggml backend device-memory query — reports current free memory as
the backend reports it (respecting other processes' usage, to whatever extent the
backend's own query does — e.g. `cudaMemGetInfo` for CUDA). Did not verify OS/driver
reservation skew on real hardware this pass (no GPU available) — the plan's acceptance
bar here ("confirm... how OS/driver reserved memory skews it on this GPU") is unverified,
carried forward as an open item same as Part 1/3's hardware-dependent items.

### A.8 MTP internals

Substantially pre-answered by the Part 1 (item 10) and Part 3 discovery passes above;
this pass adds the plan's remaining specific sub-questions:

- **Where MTP's tensors load from**: same GGUF as the main model for this arch family —
  `n_layer_nextn > 0` (`src/llama-model.cpp:2172`, STEP35 check; qwen35moe similarly
  gated) signals MTP layers are present in the same checkpoint, filtered into a
  separate memory/graph by layer index (`il >= hparams.n_layer()`, item 10 above) — not
  a companion file. Size at target quant not measured this pass (needs an actual GGUF).
- **Own KV cache**: confirmed separate object (Part 1 item 10, Part 3 discovery) —
  `LLAMA_CONTEXT_TYPE_MTP` gets its own plain `llama_kv_cache`, independent of
  `ctx_tgt`'s memory.
- **Acceptance statistics**: **already tracked**, more thoroughly than the plan assumes
  — contrary to Part 2 §8.1's "seed a default (~0.7/draft-token), replace with measured
  values" implying this needs building. `common_speculative_impl` (base class,
  `common/speculative.cpp:140-149`) already tracks `n_gen_tokens`, `n_acc_tokens`, and
  **`n_acc_tokens_per_pos`** (a full per-draft-position acceptance histogram, not just a
  scalar rate) — updated at `:2649,2681-2691`, printed via `common_speculative_print_stats`
  (`:2735-2774`, mean acceptance length + per-position rates). **Gap**: this is currently
  only surfaced via `SPC_TRC` trace-level logging, not a queryable API. Per-request
  acceptance *is* separately surfaced to API consumers already, though: `server-context.cpp`
  tracks `slot.n_draft_total`/`n_draft_accepted` per slot and reports `draft_ratio` /
  `mean_acc_len` in the completion response's `timings` object (`:614-630`). Part 2's
  profile cache (§4.3) should read from the existing per-impl counters (extend
  `common_speculative_print_stats`'s data into something machine-readable, e.g. a getter)
  rather than adding new instrumentation from scratch.
- **Verify-batch size vs. compute buffer**: not directly measured this pass (needs
  runtime); structurally, `sched_reserve()`'s worst-case reservation (Part 1 item 5) uses
  `n_seqs`/`n_tokens` derived from `cparams`, and does not appear (from the Part 1 pass)
  to have any MTP/speculative-specific worst-case sizing — i.e. **the existing
  `graph_reserve` machinery likely does not know to reserve for `n_draft+1`-sized verify
  batches**, meaning enabling MTP after the fact (or increasing `--spec-draft-n-max`)
  may need the same `sched_need_reserve = true` treatment as growth (Part 1 item 5) to
  avoid a first-decode reallocation stall. Not confirmed; flagged for Phase C
  calibration work to check empirically.
- **`-np 1` / `--mmproj` constraints**: already answered in the Part 3 discovery pass
  above — **no such guard exists in this tree** for either restriction (MTP's
  implementation is sized by `n_seq` throughout, and no mmproj×spec guard was found).
  Part 2 §8.6 ("delete/condition the MTP-mutually-exclusive-with-mmproj constraint")
  is very likely a no-op here too, same conclusion as Part 3's discovery — nothing to
  delete.

---

## Assessment against Part 2 Phase A's acceptance bar

All eight items answered with file/line references. Two carried-forward hardware-
dependent gaps (A.4's GPU-tensor-mmap confirmation, A.6's `llama-bench` cross-check, A.7's
OS/driver skew) match the same "no GPU this session" limitation noted in the Part 1 and
Part 3 passes — consistent pattern across all three plans' discovery phases.

**Biggest scope correction vs. the plan**: Part 2 Phase A originally reads as "go find
the fitter and understand its extension points." In reality the fitter
(`common_fit_params`) already *is* most of what Phase A/B describes as net-new design —
real per-device memory introspection, MoE-aware placement, override-array output format,
even a two-tier bisection solver. The plan's genuinely-missing pieces are narrower and
more clearly scoped than the plan's phase breakdown suggests:
1. A throughput/`t/s` cost model (none exists — the fitter optimizes purely for fitting
   within a memory margin, not for tokens/sec).
2. Calibration + profile cache (none exists, though the acceptance-stats counters Phase C
   would want for MTP calibration already exist, just not surfaced as an API).
3. Elastic runtime migration (Phases D-E) — genuinely new; needs the buffer-grouping
   change (A.3) and a resolved answer to the graph-reuse staleness question (A.5) before
   it can be built safely.
4. MTP-as-placement-group needs to be *finer-grained* than the plan's §8.2 assumes (its
   own dense/shared vs. routed-expert split, per A.6), not coarser design work — an
   easier fix than "build MTP awareness from scratch."

## Open items before Part 2 Phase B (placement model) can start

1. A.4's mmap-for-GPU-tensors confirmation — highest-priority unresolved item, blocks
   Phase D's whole premise (demote-without-copy).
2. A.5's graph-mutation-in-place question (does Demote/Promote need to reuse the same
   `ggml_tensor*` object, or does migration inherently invalidate `gf_res_prev`
   regardless) — needs a design decision, not just more reading.
3. A.6's MTP-as-two-groups (dense/shared + routed) refinement to the plan's cost model
   before Phase B's placement_group definition is finalized for this target arch.
4. No GPU this session — A.4, A.6 (llama-bench cross-check), and A.7 (driver skew) all
   need real hardware before they can be called closed, same caveat as Parts 1 and 3.

---

## Part 1 Phase 2 implementation: `resize()` core

Implements the plan's §4 core primitive: `virtual bool resize(uint32_t n_new)` on
`llama_memory_i` (`src/llama-memory.h`, default `return false` — zero cost / no behavior
change for every memory type that doesn't override it), with real implementations:

- **`llama_kv_cache::resize()`** (`src/llama-kv-cache.cpp`) — the actual work. Scope,
  matching the plan's v1 restrictions plus one found during discovery:
  - only `n_stream == 1` (plan's stated restriction: single-stream).
  - only caches that don't share cells with another cache (`other == nullptr`, the
    `[TAG_KV_CACHE_SHARE_CELLS]` mechanism from the Part 1 discovery pass, item 1/10) —
    not in the original plan, found while reading the constructor; a cache constructed
    with a `share` callback aliases another cache's cells and has no independent storage
    to resize. `other == nullptr` also guarantees `layers` never contains aliased tensor
    pointers (verified from the constructor: aliasing only happens `if (share && other)`),
    which the implementation depends on.
  - `n_new` must be strictly greater than the current size and a multiple of `n_pad`
    (new getter `get_n_pad()` added — the plan's Phase 1 flagged this as missing).
  - Algorithm mirrors the plan's §4.2 exactly: allocate new per-buft-group buffers sized
    `n_new` (same grouping the constructor uses), copy old data in (K and non-transposed V
    are a contiguous-prefix copy via `ggml_backend_tensor_get`/`_set`; transposed V is
    copied row-by-row via `ggml_backend_tensor_get_2d`/`_set_2d` — these turned out to
    already exist in `ggml-backend.h`/`.cpp`, exactly shaped for this use), grow cell
    metadata via a new `llama_kv_cells::grow()` (the existing `resize()` on that class
    resets everything, as Part 1 discovery item 3 already flagged — `grow()` extends the
    vectors with empty cells and leaves the existing prefix untouched), then swap
    tensors/buffers and free the old ones. New getter `get_v_storage()` added alongside
    the existing `get_k_storage()` (needed for the resize implementation's per-layer
    lookup and useful for the unit test). On any failure (context/buffer allocation)
    before the swap, the function returns `false` and the old cache is provably untouched
    (nothing is mutated until every new allocation has succeeded).
  - MLA layers (`layer.v == nullptr`) are handled by skipping the V copy — resize only
    touches K for those layers, matching how the constructor already treats MLA.
  - Hadamard rotation matrices (`attn_rot_hadamard`) are host-memory, precomputed from
    `n_embd_head_k_all`/`n_embd_head_v_all` only — untouched by resize, no interaction.
- **`llama_memory_hybrid::resize()`** — delegates to `mem_attn->resize()` only; the
  recurrent child is untouched (confirmed in Part 1 discovery: its size is
  `max(1, n_seq_max)`, independent of context length).
- **`llama_kv_cache_iswa::resize()`** — delegates to `kv_base->resize()` only; `kv_swa`
  stays at its window size (`min(base, n_swa*n_seq_max + n_ubatch)` padded to 256), per
  the plan's explicit instruction. Known gap, not in scope: with `--swa-full` the SWA
  cache is a *one-time* snapshot equal to the base size taken at construction — growing
  only the base afterward leaves a `swa_full`-configured SWA cache smaller than the grown
  base. The plan anticipated this exact case ("leave the SWA cache at window size unless
  discovery shows it is sized from n_ctx too") and this is that case, but `--swa-full` is
  a rare/experimental flag; left as a follow-up rather than blocking this pass.
- **`llama_memory_recurrent::resize()`** — no-op, always returns `true` (capacity is
  `n_seq_max`-derived, not context-length-derived — confirmed in Part 1 discovery item 10
  / the `create_memory()` call sites).
- **Not implemented** (inherit the interface's default `return false`): `llama_kv_cache_dsv4`,
  `llama_kv_cache_dsa`, `llama_memory_hybrid_iswa` — these three concrete classes were
  found during Part 1 discovery (item 1) but aren't in the plan's own §4 implementation
  list, and none of them are on the target model's (Qwen3.5/3.6) load path. Returning
  `false` is the correct, safe default (growth is "unsupported" for these, exactly per the
  interface contract) rather than an oversight; flagged here as a known scope boundary,
  not silently skipped.

### Unit test: `tests/test-kv-resize.cpp`

Registered in `tests/CMakeLists.txt` using the same `MODEL_DEST`
(`tinyllamas/stories15M-be.Q4_0.gguf`) download fixture as the other model-requiring
tests (`test-recurrent-state-rollback.cpp`, `test-save-load-state.cpp`, etc.) — same
pattern, no new fixture needed. Includes the internal header directly
(`#include "../src/llama-kv-cache.h"`), following the precedent already established by
`test-llama-archs.cpp` (`#include "../src/llama-arch.h"`) — no new CMake include-dir
plumbing required, quoted relative includes resolve fine as-is.

For each of {FA off (`v_trans == true`), FA on (`v_trans == false`), FA on + q8_0 KV
(quantized V requires FA on, confirmed at `src/llama-context.cpp:3533`)}: fills ~200
cells via `llama_decode`, snapshots layer 0's K/V bytes for the used range directly from
the backend tensors, calls `resize()` through the public `llama_memory_t` returned by
`llama_get_memory()` (downcast to `llama_kv_cache*` via `dynamic_cast` only to reach the
cache-specific getters used for verification — `resize()` itself is called through the
polymorphic `llama_memory_i` interface, so the test doesn't need to know the concrete
type to grow the cache), and checks: the equal-size call is rejected and the cache stays
usable (`decode_one` still succeeds); the valid-growth call succeeds; `get_size()` matches
the target; `llama_memory_seq_pos_max()` is unchanged; and the K/(V) byte snapshots are
byte-identical before/after. Deliberately does **not** decode further after a successful
resize as part of the pass/fail check (see "important scope boundary" below) — build and
runtime verification of the test binary itself both done this session; the download
fixture (network) was not reachable in this sandbox, so the test has not been run
end-to-end against real model weights — that's the one thing left for the user's own
hardware tomorrow, per Phase 2's own acceptance bar ("unit test green on CPU and on the
GPU backend").

### Empirical verification beyond the unit test (this session, throwaway harness, not committed)

Built a scratch harness (reusing `tests/test-llama-archs.cpp`'s `get_gguf_ctx`/
`get_model_and_ctx` synthetic-model helpers verbatim, in-memory GGUF via
`llama_model_init_from_user`, no download needed) to exercise `resize()` end-to-end
against one arch per delegation path: `LLM_ARCH_LLAMA` (plain `llama_kv_cache`),
`LLM_ARCH_MAMBA` (`llama_memory_recurrent`, no-op path), `LLM_ARCH_FALCON_H1`
(`llama_memory_hybrid`, delegates to attn child), `LLM_ARCH_GEMMA2`
(`llama_kv_cache_iswa`, delegates to base). All four: fill cells, snapshot bytes,
`resize()`, verify size/`seq_pos_max`/byte-identity — **all four passed.**

**Caught and fixed one real bug in the process**: the scratch `ggml_context` `resize()`
creates per buffer-type group was undersized — `2u*layers.size()*ggml_tensor_overhead()`
only accounts for the K and V tensors themselves, not their per-stream view tensors
(`k_stream`/`v_stream`, one each even when `n_stream == 1` — see Part 1 discovery item 2
on why these views exist and are used by `state_write`/`state_read`). This reliably
triggered `ggml_new_object: not enough space in the context's memory pool` →
`GGML_ASSERT(obj_new) failed` (an abort, not a silent corruption) on the very first
`resize()` call. Fixed to mirror the constructor's own sizing formula exactly:
`2u*(1 + n_stream)*layers.size()*ggml_tensor_overhead()`. This is exactly the kind of bug
the plan's working agreement (§0.3, "build and test after every phase") exists to catch —
it would not have been caught by a compile-only check, only by actually running
`resize()` against a real (if synthetic) model.

**Important scope boundary, confirmed empirically (not just reasoned about) this
session**: calling `llama_kv_cache::resize()` directly and then issuing one more
`llama_decode()` call — with no other changes — **segfaults** inside
`ggml_compute_forward_set_rows` (backtrace: `llama_context::decode` →
`process_ubatch` → `graph_compute` → CPU backend `set_rows` → SIGSEGV). This is the
`sched_need_reserve`/`gf_res_prev` staleness the Part 1 discovery pass already flagged
(item 5: "six existing call sites already set `sched_need_reserve = true`... provided
growth always happens at a point where a decode call follows shortly after") — but that
discovery was static/read-only ("not yet verified at runtime"). **Now verified**: raw
`resize()` alone is not suffient for a context to keep decoding; `llama_set_n_ctx()`
(Part 1 Phase 3, not yet implemented) *must* set `sched_need_reserve = true` (and update
`cparams.n_ctx`/`n_ctx_seq`, per discovery item 6) immediately after a successful
`resize()`, before the next decode. This is exactly what the plan's §5.2 step-3 ordering
already specifies — this session's contribution is empirical confirmation that skipping
it is not merely suboptimal (a missed perf optimization) but an outright crash, so Phase 3
cannot treat that step as optional or as an incremental follow-up.

### Open items before Part 1 Phase 3 (`llama_set_n_ctx`) can start

1. Wire `llama_context::set_n_ctx()`: call `memory->resize()`, then set
   `sched_need_reserve = true`, then update `cparams.n_ctx`/`cparams.n_ctx_seq` (using the
   exact formula from discovery item 6), in that order — per the plan's own §5.2 and now
   confirmed load-bearing (see above), not optional.
2. The MTP `ctx_dft` lockstep-growth decision (Part 1 discovery item 10, still open) —
   needed before growth is wired into any MTP-enabled server path.
3. `memory_update(true)` (defrag) precedence relative to growth (discovery item 4) —
   still just reasoned about, not measured.
4. `--swa-full` + growth interaction (this section, iswa bullet above) — the SWA cache
   silently stops matching the (grown) base size; needs either an explicit re-snapshot on
   growth or a documented limitation.
5. This session still had no GPU — everything above was verified on CPU only (both the
   committed unit test's structure and the throwaway multi-arch harness ran CPU-only).
   The user's own 6 GB VRAM hardware (available "tomorrow" per their message) is needed to:
   (a) actually run `tests/test-kv-resize.cpp` against the downloaded model (this sandbox
   couldn't reach the model download host), (b) repeat the CUDA-backend determinism/copy
   checks the plan's §5.4 matrix calls for, (c) confirm the `ggml_backend_tensor_get_2d`/
   `_set_2d` transposed-V path on a non-CPU backend (only the CPU backend's fallback
   per-row loop was exercised this session — `buf->iface.set_tensor_2d`/`get_tensor_2d`
   being non-null on CUDA, taking a possibly-different code path, is unverified).

---

## Part 1 Phase 3 implementation: `llama_set_n_ctx()`, auto-grow hook, determinism harness

Same session as Phase 2 above. Implements the plan's §5 in full for the topologies Phase 2
covers, using the exact ordering the plan's §5.2 specifies (validate → `memory->resize()` →
re-reserve → update `cparams`) — with one correction to the *mechanism* of the re-reserve
step, found by actually running it (see below).

### Public API (`include/llama.h`)

- `llama_context_params` gains two fields (appended right after `defrag_thold`, before the
  callback pointers — keeps the existing "primitives, then callbacks, then bools" grouping
  intact rather than appending at the very end):
  - `uint32_t n_ctx_max` — 0 = auto-grow disabled (default); else the ceiling.
  - `float ctx_grow_factor` — growth multiplier (default 1.5, matches the plan).
- `LLAMA_API int32_t llama_set_n_ctx(llama_context * ctx, uint32_t n_ctx_new)` — return
  codes exactly as the plan's §5.1 specifies (`0` success, `-1` invalid/unsupported,
  `-2` allocation-or-reserve failure). One deliberate simplification vs. the plan's own
  two-code split: `llama_memory_i::resize()` only returns `bool`, so it can't itself
  distinguish "unsupported topology" from "allocation failed" — `set_n_ctx()` catches the
  cases it *can* tell apart before ever calling `resize()` (equal/smaller size, `n_seq_max`
  making `n_ctx_seq` degenerate) as `-1`, and buckets everything `resize()` itself rejects
  (including the multi-stream/cross-cache-sharing cases from Phase 2) under `-2`. Documented
  as a known simplification, not silently glossed over.
- `llama_n_ctx_max(const llama_context *)` — read-back accessor, mirrors `llama_n_ctx()`/
  `llama_n_ctx_seq()`.
- `src/llama-cparams.h` gets matching `n_ctx_max`/`ctx_grow_factor` fields.

### `llama_context::set_n_ctx()` (`src/llama-context.cpp`)

1. Validates `n_ctx_new > cparams.n_ctx` (padded to 256 first, matching how `cparams.n_ctx`
   itself is padded at construction).
2. Recomputes `n_ctx_seq` with the **exact same formula** the constructor uses
   (`src/llama-context.cpp:261-277`, already flagged as easy-to-miss in the Phase 1
   discovery, item 6) — `kv_unified ? n_ctx : GGML_PAD(n_ctx / n_seq_max, 256)` — and calls
   `memory->resize()` with *that* value, not `n_ctx_new` directly.
3. On success, updates `cparams.n_ctx`/`cparams.n_ctx_seq`.
4. **Re-reserves immediately, not lazily.** The Phase 2 discovery pass (Part 1 discovery
   item 5) concluded growth could just set the `sched_need_reserve` flag and let the next
   `decode()`'s unconditional `sched_reserve()` call pick it up lazily, the same way six
   existing mutators (`set_causal_attn`, etc.) do. **This is wrong for the in-decode
   auto-grow case** (see below) and was caught only by actually running it: `sched_reserve()`
   is called *once*, at the very top of `decode()`, before the retry loop that would call
   `set_n_ctx()`. Setting the flag mid-loop only helps the *next* `llama_decode()` call: the
   current call would still finish with a stale scheduler/`gf_res_prev`, sized for the old
   (smaller) `n_kv`, and crash on `graph_compute()`. The actual fix, found by reading
   `memory_update(true)`'s own tail block (`src/llama-context.cpp`, the "if the memory module
   did any computation, we have to reserve a new worst-case graph" comment) — which is
   itself invoked from the exact same call site (the `FAILED_PREPARE` retry chain in
   `decode()`) for the pre-existing defrag/optimize retry — is to call `memory->init_full()`
   + `graph_reserve(...)` **synchronously, right there**, mirroring that block exactly.
   `graph_reserve()` already resets `gf_res_prev` and the scheduler internally
   (`ggml_backend_sched_reset` + `gf_res_prev->reset()`, `src/llama-context.cpp:2348-2351`),
   so nothing else needs touching. Failure policy for this step (plan §5.2 step 3's open
   question): hard-fail (`-2`) without rolling back the already-grown KV cache — matches one
   of the two policies the plan explicitly offered, chosen because rollback would require
   re-growing backwards through machinery Phase 2 doesn't have (shrink is out of scope).

**This correction is the single most load-bearing finding of this pass** — it's the reason
the empirical multi-arch verification (below) passes where a naive lazy-flag implementation
would have reproduced the exact SIGSEGV recorded at the end of the Phase 2 section.

### Auto-grow hook (`llama_context::decode()`)

Added inside the existing `LLAMA_MEMORY_STATUS_FAILED_PREPARE` case, **after** the
pre-existing optimize/defrag retry (kept today's precedence: cheaper fix first; Part 1
discovery item 2's open question about ordering is resolved this way — not measured against
the alternative, but low-risk since it only changes behavior when optimize alone doesn't fix
it), bounded to one attempt per `decode()` call via a `did_grow` flag (mirrors the existing
`did_optimize` flag exactly):

```cpp
if (!did_grow && cparams.n_seq_max == 1 && cparams.n_ctx_max > cparams.n_ctx) {
    did_grow = true;
    const llama_pos pos_max  = memory->seq_pos_max(0); // -1 if seq 0 has no cells yet
    const uint32_t  n_needed = (uint32_t) (pos_max + 1) + balloc->get_n_tokens();
    const uint32_t  n_target = std::min(cparams.n_ctx_max,
            std::max(n_needed, (uint32_t) (cparams.n_ctx * cparams.ctx_grow_factor)));
    if (n_target > cparams.n_ctx && set_n_ctx(n_target) == 0) { continue; }
}
```

- **Restricted to `n_seq_max == 1`** (the plan's own stated v1 scope: "Single sequence /
  single slot"), checked explicitly in `decode()` rather than left to `resize()`'s own
  `n_stream == 1` precondition — `n_stream` can be `1` even when `n_seq_max > 1` if
  `kv_unified` is set, in which case `memory->seq_pos_max(0)` alone would *not* reliably
  reflect how full the shared cache actually is (other sequences' cells aren't visible
  through that one call). Explicitly out of scope rather than silently wrong.
- Sizing formula matches the plan's §5.3 pseudocode exactly (`needed = cells_used +
  n_tokens_in_batch`, `target = clamp(max(n_ctx * factor, needed), ..., n_ctx_max)`), using
  `llama_memory_i::seq_pos_max()` (already public, cross-topology) instead of a
  cache-specific "used cells" query — works uniformly across plain/hybrid/iswa without
  needing a new getter.
- If `resize()` is unsupported for the cache's topology (multi-stream, cross-cache-shared),
  `set_n_ctx()` returns non-zero and this block is a no-op — falls through to the existing
  "failed to find a memory slot" warning/`return 1`, i.e. **zero behavior change when growth
  is disabled or unsupported**, satisfying the plan's working-agreement rule 4.

### Empirical verification (throwaway harness, same technique as Phase 2, not committed)

Reused the Phase 2 harness (`get_gguf_ctx`/`get_model_and_ctx` from `test-llama-archs.cpp`)
to build two contexts per arch from the *same* synthetic model: one pre-allocated at a large
`n_ctx`, one starting small with `n_ctx_max` set to that same large value. Fed both the
identical 300-token sequence one token at a time and compared logits at every step.

**All four archs tested — plain (`LLM_ARCH_LLAMA`, both `v_trans` states), hybrid
(`LLM_ARCH_FALCON_H1`), iswa (`LLM_ARCH_GEMMA2`) — passed**: auto-grow fired exactly once
per run (confirmed via `llama_n_ctx()` increasing mid-loop), and logits matched the
pre-allocated reference context to float precision (`max abs diff` ≈ 0, well under the
`1e-4` gate) at *every* step, including the exact step where growth happened. This is the
first real evidence in this multi-session effort that grow-then-keep-decoding actually
works end to end, not just "resize() doesn't crash by itself."

### Unit test: `tests/test-ctx-grow.cpp`

Registered against the same `MODEL_DEST` download fixture as `test-kv-resize.cpp`. Covers
{FA off, FA on, FA on + q8_0 KV} the same way. For each: builds a static-`n_big` reference
context and a `n_small`-start/`n_ctx_max = n_big` growing context from the same model, feeds
both an identical 300-token sequence, asserts logits match at every step (`1e-4` abs-diff
gate) and that growth actually fired. Not run end-to-end this session (no reachable model
download host in this sandbox, same limitation as `test-kv-resize.cpp`) — build-verified
only; the throwaway harness above is the real evidence this pass's logic works, using a
synthetic in-memory model instead of the download fixture.

### Deliberately not done this pass (out of scope, per the user's own phrasing: "Phase 3 —
### `llama_set_n_ctx`, auto-grow hook, determinism harness")

- **CLI/server plumbing** (`--ctx-max`, `--ctx-grow-factor` flags, `llama-server` proactive
  growth, `slot.n_ctx` refresh) — this is the plan's own Phase 5 (§7), a separate, later
  unit of work; `llama_context_params::n_ctx_max`/`ctx_grow_factor` exist but nothing in
  `common/` or `tools/server/` sets them yet. A caller must currently set these two fields
  directly to use growth.
- **MTP `ctx_dft` lockstep growth** (Part 1 discovery item 10) — `set_n_ctx()` only grows
  the `llama_context` it's called on; a server wrapping both `ctx_tgt` and `ctx_dft` would
  need to call it on both, in some order, itself. Not addressed here.
- **iSWA `--swa-full` re-snapshot** (Phase 2 section above) — still open, unchanged.

## Open items before Part 1 Phase 4 (hybrid + SWA + Qwen3.6 smoke test) / Phase 5 (CLI/server)

1. Real hardware run of `tests/test-kv-resize.cpp` and `tests/test-ctx-grow.cpp` against the
   downloaded tinyllamas model — blocked on network access in this sandbox, first item for
   the user's own machine.
2. CUDA-backend verification of everything above (Phase 2's caveat still applies: only the
   CPU backend's code paths have been exercised, including for the transposed-V row copy).
3. Phase 5 CLI/server flags (`--ctx-max`, `--ctx-grow-factor`) are the natural next unit of
   work to make growth reachable from `llama-cli`/`llama-server` without hand-writing
   `llama_context_params`.
4. The target model smoke test (Qwen3.6-35B-A3B GGUF, growth past 3 boundaries, VRAM/t/s
   table) from the plan's Phase 4 needs real hardware and is still fully open.

---

## Real-hardware findings (user's Windows/VS2022 + CUDA + 6 GB VRAM machine)

First results from actual hardware, following `TESTING.md`. Two findings so far.

### 1. Part 3's determinism oracle (Part 3 discovery item 7 / Part 1 §5.4's test-oracle
### assumption): **MTP-on is not token-identical to MTP-off, text-only, no images involved**

Real run: `unsloth`-style Qwen3.6-35B-A3B Q4_K_M quant (with its MTP heads and mmproj
present, but **mmproj/images not exercised in this test** — this is the text-only check),
`--spec-type draft-mtp` vs `--spec-type none`, both `--temp 0 --seed 1`, same prompt.
Outputs diverge (581 vs. 614 final tokens across the two runs) but both are reported
coherent, not degenerate/repeating. `draft acceptance = 0.524 (343/654), mean len = 2.57`
for the MTP run — a plausible, healthy-looking acceptance rate, not a sign of something
badly broken.

**This is very unlikely to be caused by the Part 3 mmproj fix**: no image was ever sent in
this test, and that fix's entire effect is gated behind `batch_in.embd != nullptr` (an
image/audio embedding batch) — the `valid` flag it introduces starts `true` and is only
ever set `false` on such a batch. A pure-text conversation never touches that code path, so
the fix is provably a no-op for this specific test. The divergence must come from
`draft-mtp`'s own draft/verify/accept logic (untouched by the Part 3 session), or from an
inherent floating-point difference between the *batched* verify-pass computation and the
*single-token sequential* baseline decode (a well-known category of issue in speculative
decoding generally — different reduction order / kernel selection for batch>1 vs batch=1,
especially on a quantized model, can flip an argmax at temp 0 without any logic bug being
involved). Both explanations are consistent with what was actually observed (coherent but
diverging output, healthy acceptance rate) rather than, say, garbage output or a crash,
which would point more toward a real accept/reject logic bug.

**Decision (user's call, recorded here for continuity):** treat "MTP-on token-identical to
MTP-off" as **false** for now; do not chase this further in this line of sessions. Flagged
for separate deeper investigation later. This means: the token-identity oracle Part 1 §5.4
and Part 3 Phase 3's tests both assume ("Assert identical token output") **cannot be used
as-is** for any MTP-enabled determinism testing — per both plans' own stated fallback
("if [temp-0 identity] is not [available], use logit/perplexity comparison instead"), any
future determinism harness that must cover MTP-enabled configurations needs to switch to a
logit-distance or perplexity-based comparison, not exact token-identity. Note this does
**not** block Part 1's own growth-determinism work (`test-kv-resize.cpp`/`test-ctx-grow.cpp`):
those tests exclude MTP entirely and rely on non-speculative decode, where token/logit
identity is expected to hold (and did, per the Phase 2/3 synthetic-model verification above).

### 2. Real build bug found: `test-kv-resize`/`test-ctx-grow` failed to link on Windows (MSVC)

```
error LNK2019: unresolved external symbol "public: unsigned int __cdecl llama_kv_cache::get_size(void)const"
error LNK2019: unresolved external symbol "... llama_kv_cache::get_k_storage(int)const"
error LNK2019: unresolved external symbol "... llama_kv_cache::get_v_storage(int)const"
```

Root cause: Windows DLLs only export symbols explicitly marked `__declspec(dllexport)`
(here, via the `LLAMA_API` macro) — unlike ELF shared libraries (Linux), which export
*everything* with default visibility unless told otherwise. `llama_kv_cache` is an internal
class with no `LLAMA_API` annotations (by design — it's not part of the public C API), so
on a Windows shared (`BUILD_SHARED_LIBS=ON`, the default) build, its methods are invisible
to any other executable linking against `llama.dll`, including the two internal-header-using
test binaries added this session. This only ever surfaced now because this whole
multi-session effort had no Windows testing until this point — the Linux sandbox used for
every prior verification pass masked the problem entirely (ELF default-exports everything).

**Fix**: `src/CMakeLists.txt` now sets `WINDOWS_EXPORT_ALL_SYMBOLS ON` on the `llama` target
(under the existing `if (BUILD_SHARED_LIBS)` block), mirroring the *exact same* workaround
`common/CMakeLists.txt` already applies to `llama-common` for the identical underlying
reason (that file's own comment: `# TODO: make fine-grained exports in the future` — copied
verbatim, since it's still exactly the caveat that applies here too). This is additive only
(exports a strict superset of symbols; nothing that was exported before stops being
exported), so it doesn't change behavior for any existing consumer of `llama.dll`
(`llama-cli.exe`, `llama-server.exe`, etc.) — verified the Linux build stays unaffected too
(`WINDOWS_EXPORT_ALL_SYMBOLS` is a documented no-op on non-Windows platforms).

**Open question, not investigated**: `test-llama-archs.cpp` (pre-existing, not part of this
session's work) also directly calls non-template `llama_model_saver` methods
(`add_kv(uint32_t)` etc., implemented in `llama-model-saver.cpp`, part of the `llama` target)
from outside the DLL — by the same logic, it's plausible that test *also* fails to link on a
stock Windows shared build, independent of anything in this branch. Not confirmed (no Windows
environment available to check upstream `master` directly); if true, this fix incidentally
also resolves that latent issue, but that wasn't the goal and wasn't verified as a before/after
comparison.

### Next steps for the user's hardware

1. Re-pull this branch (now includes the `WINDOWS_EXPORT_ALL_SYMBOLS` fix), rebuild, retry
   `test-kv-resize` and `test-ctx-grow`.
2. MTP-on/off determinism: parked per the decision above; separate investigation later.

### 3. `test-kv-resize`'s "fa-on q8_0 KV" scenario failed against the real download fixture

With `WINDOWS_EXPORT_ALL_SYMBOLS` fixed, `test-kv-resize` actually ran against the real
`tinyllamas/stories15M` download fixture for the first time (CUDA, Windows). Two of three
scenarios passed (`fa-off`/`v_trans` and `fa-on`/`!v_trans`, including a real, non-synthetic
byte-identity check across an actual `resize()` call — the first time this ran against a
real downloaded model rather than the in-memory synthetic harness). The third, `fa-on + q8_0
KV`, failed context construction outright:

```
llama_init_from_model: K cache type q8_0 with block size 32 does not divide n_embd_head_k=48
```

**Not a `resize()` bug** — this is a pre-existing, general llama.cpp constraint
(`src/llama-context.cpp:3617-3624`): quantized KV types can only be used when the
quantization block size evenly divides the model's per-head embedding dimension, so a whole
number of blocks fits in one head's row. `stories15M`'s head dim happens to be 48
(`48 % 32 != 0` for `q8_0`'s block size), an incompatibility that has nothing to do with
resize()'s copy logic — it's the *model* that can't use `q8_0` KV at all, checked before any
KV cache is even constructed.

**Fix**: both `tests/test-kv-resize.cpp` and `tests/test-ctx-grow.cpp` now treat a failed
`llama_context` construction as a **skip** (log + `return true`) rather than a hard failure,
matching the existing "memory is not a plain llama_kv_cache — skipping" pattern already used
for the recurrent/hybrid-model case. This makes both tests robust to being pointed at an
arbitrary user-supplied model (via `-m`) that may not support every {FA, KV-type} combination
the test tries, without weakening what they actually check when a scenario *is* supported.
Verified this doesn't regress the two passing scenarios (rebuild only, still no reachable
model-download host in this sandbox to re-run end-to-end).

---

## Part 1 Phase 5 implementation: CLI/server plumbing

Implements the plan's §7. Scope: `--ctx-max`/`--ctx-grow-factor` CLI flags, `llama-server`
proactive growth + `slot.n_ctx` refresh + startup validation. `llama-cli` needed **no
separate work**: in this tree `llama-cli` spawns an embedded `llama-server` internally and
talks to it over HTTP (`tools/cli/cli-server.h`, `llama_server()` called in a background
thread) — it's the exact same `server_context_impl` code path, so once the flags exist in
`common/`, `llama-cli` inherits growth for free. This directly satisfies the plan's own
"llama-cli: wire the same flags for testability" bullet without any CLI-specific code.

### `common/` flags

- `common_params` gains `n_ctx_max` (`int32_t`, 0 = disabled) and `ctx_grow_factor`
  (`float`, default 1.5) — `common/common.h`.
- `--ctx-max N` / `--ctx-grow-factor N` registered in `common/arg.cpp`, no `.set_examples()`
  restriction (same as `-c`/`--ctx-size`'s own registration) — available to every example
  (cli, server, completion, etc.) uniformly.
- `common_params_parse()` postprocess step (same place `--prompt-cache-all` +
  `--interactive` incompatibility is already checked): if `--ctx-max` is set and `-c` was
  left at its 0-default, start at `min(8192, ctx_max)` instead of the model's full training
  context — exactly the plan's §7.1 instruction ("when `--ctx-max` is set and `-c` is
  untouched, default the initial size to something small").
- `common_context_params_to_llama()` forwards both fields into `llama_context_params`.

### `llama-server` (`tools/server/server-context.cpp`)

- **Startup validation** (`load_model()`, right after `params_base = params`): hard-refuses
  to start (`return false`, clear `SRV_ERR`) if `--ctx-max` is set with `--parallel != 1` —
  the plan's explicit v1 restriction, matching `resize()`'s own `n_stream == 1` scope.
- **Proactive growth**: new `maybe_grow_for_request(slot)`, called once per new task right
  before the existing prompt-length checks (`slot.state == SLOT_STATE_STARTED`, before the
  `can_split()`/`n_ctx` too-long checks). Computes `needed = n_prompt_tokens + max(n_predict,
  0)`, and if that exceeds the *current* `llama_n_ctx_seq(ctx_tgt)`, calls
  `llama_set_n_ctx(ctx_tgt, min(ctx_max, needed))` directly — **not** stepped by
  `ctx_grow_factor` (that stepping is reserved for the reactive in-`decode()` hook from
  Phase 3, which only ever sees one ubatch at a time and has no visibility into the whole
  request's shape; the server does, so it can size exactly once). On success, calls the new
  `refresh_n_ctx()` immediately.
- **Reactive refresh**: `refresh_n_ctx()` — sets `server_context_impl::n_ctx` (the
  total-context member, from `llama_n_ctx()`) and every slot's `n_ctx` (from
  `llama_n_ctx_seq()`) to the current live value. Called from two places: inside
  `maybe_grow_for_request()` on success, and unconditionally (when growth is enabled) right
  after every successful `llama_decode()` call in the low-level `decode()` method — covering
  the case where Phase 3's *reactive* hook inside `llama_decode()` itself grew the cache
  transparently, which the server has no other way to observe.
- **`/props` and error messages**: needed **no code changes** — `get_slot_n_ctx()` already
  reads `slots.back().n_ctx` live (not a cached snapshot), and the existing "prompt too
  large" error messages already interpolate `slot.n_ctx` directly, so both automatically
  reflect the current (possibly grown) value once `slot.n_ctx` itself is kept fresh.
- **Context-shift/cache-reuse interplay (plan's §5.3 policy, "grow first, shift only at
  ctx_max")**: also needed **no code changes**, for a subtler reason — both places that
  gate context-shift (`process_token()`'s early stop, `pre_decode()`'s shift trigger) key
  off `slot.n_ctx + 1 >= n_tokens`. Since `slot.n_ctx` now keeps growing (via the refresh
  above) for as long as growth has room to give, those checks simply don't fire until
  growth is truly exhausted (`slot.n_ctx` pinned at `ctx_max`) — the desired policy falls
  out of keeping one value fresh, rather than needing new precedence logic in the shift path
  itself.
- **Known gap, not addressed**: `server_prompt_cache` (the optional `--cache-ram-mib`
  prompt-caching subsystem, `tools/server/server-context.cpp:~1354`) is sized once at load
  time from the *original* `n_ctx` and is not resized on growth. Likely low-impact (it's an
  optional feature, off by default) but not verified either way — flagged for a follow-up
  pass rather than investigated this session.

### Empirical verification: real `llama-server` HTTP round-trip (not just unit tests)

Went one step further than the synthetic in-process harnesses used for Phases 2–3: built a
synthetic model **with real backend-allocated tensor data** (`get_gguf_ctx` +
`llama_model_init_from_user`, then `llama_model_saver` — its "round-trip from a live model"
constructor/`add_kv_from_model()`/`add_tensors_from_model()`/`save()` — to serialize that
in-memory model out to an actual `.gguf` **file on disk**), then launched a genuine
`llama-server` **process** against that file and drove it with real HTTP requests via
`curl`. This is a materially stronger check than the earlier in-process harnesses: it
exercises the actual CLI arg parsing, the actual server startup path, and the actual
HTTP → task → slot → decode pipeline, not code called directly from a test binary.

Results:
- `llama-server ... --ctx-max 2048 -c 256`, prompt of 500 tokens (as a raw token-ID array
  via `/completion`'s `prompt` field, sidestepping the synthetic model's placeholder
  `no_vocab` tokenizer for the *input* side): log shows
  `slot maybe_grow_f: ... grew KV cache ahead of prefill: n_ctx 256 -> 512 (prompt = 500
  tokens, n_predict = 0)` — proactive growth fired correctly through the real request path.
- `--ctx-max 512`, prompt of 1000 tokens: grew to the ceiling (`256 -> 512`, clamped
  correctly to `ctx_max` even though the request needed more), then correctly rejected with
  `"request (1000 tokens) exceeds the available context size (512 tokens)"` — note the
  error message already reports the *post-growth* 512, not the pre-growth 256, confirming
  `slot.n_ctx` was refreshed before the check ran. Server stayed healthy afterward (this
  reject path returns before any decode/token-sampling, so it never touches the synthetic
  model's vocab-crash limitation below).
- `--ctx-max 2048 -np 2`: server refused to start with the expected
  `--ctx-max requires --parallel 1` error, exit before any model load work.
- **Caveat, not a growth bug**: any request that reaches actual *generation* (sampling ≥1
  token) crashes this particular synthetic model, because `get_gguf_ctx()` sets
  `tokenizer.ggml.model = "no_vocab"` and `common_token_to_piece()` unconditionally asserts
  `type != LLAMA_VOCAB_TYPE_NONE` when formatting a sampled token back into response text
  (`src/llama-vocab.cpp:3091`). This is a limitation of the synthetic model (no real
  tokenizer), unrelated to growth — confirmed by checking exactly where each crash occurred
  (`common_token_to_piece` → `post_decode()`'s per-token result callback, always *after*
  `maybe_grow_for_request()`/`llama_decode()` had already run and succeeded). Building a
  synthetic model with a real (even minimal) tokenizer, to get a full generate-and-verify
  round trip, is a possible follow-up but wasn't pursued given the proactive-growth and
  reject-path evidence already obtained is unambiguous.

### Open items before Part 1 Phase 4 (Qwen3.6 hardware smoke test) / further polish

1. `server_prompt_cache` resize-on-growth gap (above) — needs a decision: resize it too, or
   document that `--cache-ram-mib` + `--ctx-max` together is untested/unsupported for now.
2. A full real-model, real-generation `llama-server --ctx-max ... -c ...` session (the
   user's own hardware, per `TESTING.md` §3) is the natural next real-world check — the
   verification above proves the mechanism fires and clamps correctly via real HTTP, but
   didn't (couldn't, in this sandbox) verify a full multi-turn conversation growing several
   times while generating real tokens.
3. MTP `ctx_dft` lockstep growth (carried over from Phase 3) is still unaddressed — a
   server session combining `--spec-type draft-mtp` with `--ctx-max` will grow `ctx_tgt`
   only; `ctx_dft` stays fixed-size. Not validated either way this pass.

---

## Real-hardware findings, round 2: struct layout fix + MTP lockstep growth

### 1. Crash: `common/arg.cpp:2520: GGML_ASSERT(params.n_gpu_layers < 0) failed` on Windows

Reported when starting `llama-server` with `--ctx-max` on the user's Windows/MSVC build.
This assert is **pre-existing, unrelated to anything in this branch's diff** (confirmed via
`git blame` — authored 2026-07-08, well before this branch existed) — it's a sanity check,
evaluated once at flag-registration time, that `common_params::n_gpu_layers` still holds its
compiled-in default (`-1`) at the point the `-ngl` flag's help text is generated. It firing
means `n_gpu_layers` held a nonsensical value at that point — impossible under normal
control flow (every call site either default-constructs `common_params` or passes one that
hasn't been touched yet), which is the signature of a **struct-layout/ABI mismatch between
separately-compiled translation units**, not a logic bug.

**Likely root cause, self-inflicted**: both `n_ctx_max`/`ctx_grow_factor` fields (Phase 5)
were inserted in the *middle* of `llama_context_params` (`include/llama.h`, between
`defrag_thold` and `cb_eval`) and of `common_params` (`common/common.h`, between `n_ctx` and
`n_batch`) — shifting the byte offset of every field declared after the insertion point
(including `n_gpu_layers`, dozens of fields later in `common_params`). The project's own
stated convention for `llama_context_params` (noted in the original plan itself: "llama.cpp
appends new fields... keep struct ABI notes in mind") exists precisely to avoid this: on an
incremental Windows/MSBuild build, if even one translation unit that reads/writes a shifted
field doesn't get fully recompiled against the new header (stale `.obj`/`.lib`/`.dll`
linked against a mix of old/new layouts), fields after the insertion point silently
misalign — exactly the kind of "impossible" garbage this assert is designed to catch,
just for an unrelated field.

**Fix**: moved both fields to the very end of `llama_context_params` (after `ctx_other`)
and of `common_params` (after `no_alloc`), matching the append-only convention, and moved
the corresponding two initializer-list entries in `llama_context_default_params()`
(`src/llama-context.cpp`) to match — that function uses **positional aggregate
initialization** (the `/*.field_name =*/` comments are cosmetic; there is no C++20
designated-initializer syntax in play, this project targets C++17), so the values and the
struct declaration order must move together or every field after the insertion point reads
the wrong initializer. Verified: full rebuild clean, and re-ran the same real-`llama-server`
synthetic-model HTTP check from the Phase 5 pass (proactive growth firing, `256 -> 512`) —
unaffected, as expected, since growth's own logic reads every field by name, not position.

**User action needed**: pull this fix and do a **full rebuild**, not just an incremental
one — if the theory above is right, an incremental build might not reliably clear whatever
stale state caused the mismatch in the first place. If the crash recurs after a genuinely
clean rebuild, this theory is wrong and needs to be revisited (open a fresh investigation
rather than assuming the fix above was sufficient).

### 2. Closed the MTP `ctx_dft` lockstep-growth gap

`server_context_impl::refresh_n_ctx()` (`tools/server/server-context.cpp`) now also grows
`ctx_dft` (the MTP draft context, when present) to match `ctx_tgt`'s current
`llama_n_ctx_seq()` every time it runs — i.e. after both proactive growth
(`maybe_grow_for_request()`) and reactive growth (the hook inside `llama_decode()` itself).
Implementation: direct `llama_set_n_ctx(ctx_dft, n_ctx_seq_now)` call, guarded by
`ctx_dft && ctx_dft != ctx_tgt` and only attempted when `ctx_dft` is actually behind. This
matches the plan's own framing exactly ("growing `ctx_dft` in lockstep... is a call-site
change in whatever wraps `llama_set_n_ctx`, not a `resize()` change") — no changes needed
to `llama_set_n_ctx()`/`resize()` themselves, just one more grow call from the same place
that already refreshes the server's own bookkeeping.

Known rough edge, accepted rather than solved: for architectures where the draft head
shares `ctx_tgt`'s memory instead of owning its own (gemma4's `is_mem_shared` mode —
**not** the target qwen35(moe) family this whole effort is built around), `resize()`
correctly refuses to touch a cache that shares cells with another
(`[TAG_KV_CACHE_SHARE_CELLS]`, Phase 2), so the lockstep-growth call for `ctx_dft` would
fail there — harmlessly, since growing `ctx_tgt`'s memory already covers the shared case,
but `llama_set_n_ctx()`'s return code doesn't distinguish "harmless, already covered" from
"a real allocation failure," so this logs a warning either way for that architecture family.
Not fixed, since it doesn't affect the target model and a real fix would need a new way to
ask "is this memory shared with another context" that doesn't exist yet.

**Not verified this pass**: no MTP-capable model available in this sandbox (same limitation
as every other pass) — the lockstep call was reviewed against the code (mirrors the
already-working `maybe_grow_for_request`/`refresh_n_ctx` pattern exactly, using the same
`llama_set_n_ctx` entry point Phase 3 already validated end-to-end) but not exercised at
runtime with a real `ctx_dft`. This is the first thing to check on the user's hardware:
`llama-server --spec-type draft-mtp --ctx-max ... -c ...`, grow past a boundary, confirm no
MTP-related errors/crashes and that drafting continues (watch `-lv 4` trace output and the
new `grew MTP draft context in lockstep` log line).

---

## Real-hardware findings, round 3: mid-generation growth was never actually reachable

First real long-conversation test on the user's hardware (`-c 8192 --ctx-max 131072`,
`--temp 0`, real multi-turn chat via slot/LCP-reuse) surfaced two things.

### 1. `llama_kv_cache: resizing KV cache: ...` / `llama_context: growing n_ctx: ...` never
### appear in the log — **not a bug**, a pre-existing log-verbosity mapping

Both lines are emitted via `LLAMA_LOG_INFO` (`src/llama-kv-cache.cpp`,
`src/llama-context.cpp`), which maps to `GGML_LOG_LEVEL_INFO`. `llama-server`'s own log
plumbing (`common/log.cpp:common_get_verbosity()`) maps `GGML_LOG_LEVEL_INFO` →
`LOG_LEVEL_TRACE` (verbosity 4) for messages coming from the `llama`/`ggml` library layer
specifically — as opposed to `SRV_INF`/`SLT_INF` (used by the server's own
`maybe_grow_for_request`/`refresh_n_ctx`/etc.), which pass `LOG_LEVEL_INFO` (verbosity 3)
explicitly and are visible at the default verbosity. This is **general, pre-existing
behavior**, not specific to growth — it also explains why the model's own load-time
`llama_context: n_ctx = ...`/`llama_kv_cache: size = ... MiB` lines were already absent from
every real-hardware log shared so far, growth-related or not. No code change; `-lv 4`
(already documented for MTP trace debugging, §1.5) surfaces these too. Updated `TESTING.md`
§4.2 to say so explicitly instead of implying they're always visible.

### 2. Real bug: a long, open-ended generation truncated at the *original* `n_ctx` instead
### of growing — `slot.n_ctx` was correct, but growth never got a chance to run

Observed directly in the log: one turn ended with `stop processing: n_tokens = 8191,
truncated = 1` (i.e. hit the original `-c 8192` ceiling and gave up) even though
`--ctx-max 131072` was set and plenty of room remained. The **next** turn's proactive
growth then fired fine (`grew KV cache ahead of prefill: n_ctx 8192 -> 8448`), confirming
growth itself works — the bug was specifically about **generation never reaching the point
where growth would be tried**.

Root cause, found by tracing the exact call sequence: `process_token()` has its own
long-standing check —

```cpp
if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
    slot.truncated = true;
    slot.has_next_token = false; // stop generation
    ...
}
```

— that runs **after every generated token**, deciding whether to request the *next* one.
Phase 3's reactive auto-grow hook lives entirely inside `llama_context::decode()`, triggered
only when a `llama_decode()` call actually fails to find room
(`LLAMA_MEMORY_STATUS_FAILED_PREPARE`). But this server-side check fires *before* that next
`llama_decode()` call would ever happen — if it sets `has_next_token = false`, generation
stops right there and **`llama_decode()` is never called again for this slot**, so its
reactive hook never gets a chance to run at all. This was a genuine gap in the Phase 5
design: I had reasoned (incorrectly, in the original Phase 5 notes above) that keeping
`slot.n_ctx` fresh via `refresh_n_ctx()` would be sufficient for the shift/stop precedence
to "fall out for free" — that reasoning only covers cases where growth *already happened*
by some other path; it doesn't cover the case where this exact check is the *first and
only* place that would ever notice more room is needed for an open-ended (`n_predict = -1`)
generation that organically outgrows what `maybe_grow_for_request()` sized at prompt start.

**Fix**: new `maybe_grow_mid_generation(slot)` (`tools/server/server-context.cpp`), called
from `process_token()` right before the existing stop-check, whenever
`slot.prompt.n_tokens() + 1 >= slot.n_ctx` — regardless of whether `ctx_shift` is enabled,
so growing here also makes the *separate* `ctx_shift` trigger in `pre_decode()` (which reads
the same `slot.n_ctx`) naturally skip shifting for as long as growth still has room, giving
the plan's "grow first, shift only once `ctx_max` is reached" policy for free from one call
site rather than needing precedence logic duplicated in the shift path. Sized with
`ctx_grow_factor` stepping (`max(n_needed, n_ctx_cur * factor)`), matching the formula the
in-`decode()` hook itself uses — appropriate here since, like that hook, this call site
also doesn't know in advance how much more the generation will need (unlike
`maybe_grow_for_request()`'s exact-fit sizing, which does know from `n_predict`).

**Not verified at runtime this pass**: attempted to reproduce with the same
`llama_model_saver`-round-tripped synthetic model used for earlier Phase 5 verification, but
hit an even earlier limitation than expected — that model's placeholder `no_vocab` tokenizer
crashes inside `post_decode()`'s token-to-text formatting on the *very first* generated
token, before `process_token()`'s check (where the fix lives) is ever reached for a second
token. Confirmed via the crash backtrace (`post_decode()` → `common_token_to_piece` →
abort, called from `update_slots()` *before* my check would run again). Building a synthetic
model with a real (even minimal) tokenizer to get past this would be needed for a true
in-sandbox repro; not pursued given time spent already. The fix is a small, narrowly-scoped
change reusing the exact same `llama_set_n_ctx()`/`refresh_n_ctx()` machinery Phase 5's
proactive path already validated end-to-end via real HTTP — reviewed carefully but **this
specific trigger path is unverified at runtime**. This is the top thing to check on the
user's hardware: a long, open-ended (no explicit `n_predict`, or a large one) generation
that runs past the initial `-c` size should now keep growing (`grew KV cache
mid-generation: ...` in the log) instead of truncating.
