> # RESOLVED: the logit shift under `-ehs` is device placement, not a routing error
> #
> # `-ehs -1` shifts the first token's logprobs by mean 7.0e-2 against stock. **Explained**: with
> # the tier engaged, all four MoE ops per layer - `ffn_moe_gate`, `ffn_moe_up`,
> # `ffn_moe_swiglu`, `ffn_moe_down` - move from **CPU to CUDA0, in all 40 layers**, because the
> # tier's hot output is a CUDA tensor and the scheduler pulls the rest of the MoE after it.
> # Every op is correct; they run on a different device than in the stock arm, and 40 layers of
> # CUDA-vs-CPU rounding compounds into the observed shift.
> #
> # This fits every measurement: flat in S (placement does not depend on how many experts are
> # resident), and unchanged with the hot path zeroed (the graph shape, hence the placement, is
> # the same). Measured with `GGML_SCHED_DEBUG=2`, see section 12.
> #
> # It is the same class of difference as running the model on the GPU instead of the CPU at all.
> # Perplexity over a fixed corpus is still the check worth having before shipping, but there is
> # no longer a reason to suspect the tier's routing.
# PLAN - make `ngram-mod` speculation compatible with the expert hot store (`-ehs -1`)

Goal: let `--spec-type ngram-mod` and `-ehs -1` be used together without one silently disabling
the other, on the same 6 GB VRAM / 16 GB RAM box as `PLAN-qwen36-35b-moe.md`.

Trigger: the config `--spec-type draft-mtp,ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48
--spec-ngram-mod-n-max 64` was claimed onlne to be a massive performance improvement on similar hardware for Qwen3.6-35b.

Companion to `PLAN-qwen36-35b-moe.md` (sections 1.4, 7.4, 10.1, 12.3, 15.9 are load-bearing here)
and `PLAN-qwen36-27b.md`. No code changed. Every structural claim cites a file:line in this tree.
Every number is marked **measured**, **published**, or **derived**.

---

## 0. What this document decides

The 4-token tier ceiling (`src/llama-expert-tier.h:9`) is not the interesting constraint. The
interesting constraint is that **`mul_mat_id` op-offload fires at 32 tokens**
(`ggml/src/ggml-cuda/ggml-cuda.cu:5195-5207`, threshold at `:5377`), which puts a 49-65 token
verify batch in a completely different execution regime from anything this fork has measured.

So the plan is deliberately staged:

| Step | What | Code | Decides |
|---|---|---|---|
| 0 | Three zero-code experiments | none | whether steps 2-4 are worth doing at all |
| 1 | Make the bypass non-silent | ~30 lines | nothing, do it regardless |
| 2 | Heat correctness at multi-token batches | ~60 lines | prerequisite for 3 |
| 3 | Landing-pad remap, lift the ceiling | ~150 lines | the actual compatibility fix |
| 4 | `-ehs -1` sizing and shrink floor | ~40 lines | autofit correctness |
| 5 | Validation | scripts | ship / revert |

**Step 0 can kill steps 2-5 in an afternoon and costs nothing.** Do not skip it.

---

## 1. The blocker, precisely

### 1.1 ngram-mod is all-or-nothing at `n_min`

`draft_one` walks up to `n_max` continuations and, if the modulo table runs dry before `n_min`,
**clears the result entirely** (`common/speculative.cpp:1958-1970`):

```c
        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }
```

So at `n_min 48 / n_max 64` a firing draft is 48-64 tokens and the target verify ubatch is 49-65.
There is no middle ground, and nothing truncates it: the server's per-slot cap is
`get_n_draft_max()`, a context-space limit, not a user limit
(`tools/server/server-context.cpp:3022`, `:463-481`).

ngram-mod is also **higher priority than draft-mtp** (`common/speculative.cpp:2449-2459`) and the
draft loop stops at the first impl that returns tokens (`:2645-2691`). So whenever ngram-mod
fires, MTP does not run. The stream is bimodal: 49-65 token batches when ngram-mod hits, 2-4 token
batches when it misses and MTP takes over.

### 1.2 Two independent failures above 4 tokens

1. **Tier bypass.** `llama_expert_tier_build` returns `nullptr` at `cur->ne[2] > 4`
   (`src/llama-expert-tier.cpp:85-95`), so `build_lora_mm_id` falls back to stock
   `ggml_mul_mat_id`. Correct output, no acceleration, and the hot store still holds its VRAM.
2. **Heat freeze.** `expert_heat_batch = ubatch.n_tokens <= 4` (`src/llama-context.cpp:1569`), so
   the heatmap stops updating once the store is filled and `maybe_resync` returns immediately on
   `multi_slot` (`src/llama-expert-hotstore.cpp:353-361`). This is the same failure `a02f14d05`
   fixed for `1 + n_draft` verify batches, reachable again through draft width.

### 1.3 Why the ceiling is 4

Not arbitrary. The sentinel remap maps every cold expert of a token to one slot, so a token's id
list contains duplicates. `mm_ids_helper` increments `it_compact` **once per token** via
`warp_reduce_any` (`ggml/src/ggml-cuda/mmid.cu:57`, `:75`) while `nex_prev` counts **per
occurrence** (`:47`, `:72`). With duplicates the two disagree and the compact-buffer bounds are
wrong - corruption, not just missing rows. MMVQ has no compaction and is safe; 4 is the lowest
per-type bound of `get_mmvq_mmid_max_batch()` over every arch table.

**Worth recording, because `PLAN-qwen36-35b-moe.md` section 12.3 does not say it:** the hazard is
not MMQ-specific. The host-side fallback in `ggml/src/ggml-cuda/ggml-cuda.cu:1940-1960` breaks on
the *first* match per token and then asserts:

```c
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));
```

With duplicates that is a hard abort. Any fix that removes duplicates at the source makes MMVQ,
MMQ, MMF and the fallback all correct at once; any fix that only widens the MMVQ window does not.

---

## 2. Three regimes, and where the proposed config lands

`get_op_batch_size` returns `ne[2]` (= `n_tokens`) for `MUL_MAT_ID`, and op-offload fires at
`>= 32` by default (`GGML_OP_OFFLOAD_MIN_BATCH`, `ggml-cuda.cu:5377`). With `-ehs -1` all MoE
weights are forced to host buffers (`common/common.cpp:1256-1263`), which is exactly the
condition op-offload tests (`ggml/src/ggml-backend.cpp:959`). So:

| Regime | tokens | What runs the MoE today | Tier |
|---|---|---|---|
| R1 | 1-4 | CPU, per-token matmul | **engaged** |
| R2 | 5-31 | CPU, per-token matmul | bypassed |
| R3 | >= 32 | **GPU, whole `_exps` tensors streamed over PCIe** | bypassed |

`n_min 48 / n_max 64` sits in **R3**, the one regime where the tier has never been characterised
and where a competing mechanism already owns the MoE.

That matters because the tier's cold op is **CPU-only** - `GGML_OP_MUL_MAT_ID_COLD` is implemented
in `ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.c` and appears nowhere in `ggml-cuda`. Engaging the
tier in R3 therefore does not just add a hot path, it **takes the MoE away from op-offload and
pins it to the CPU**. That is a first-order change in what the machine does, and it dwarfs the
slot-accounting arithmetic below.

### 2.1 Which side of that trade wins is not obvious, and is not measured

Derived, using this tree's measured constants: host bandwidth **~26 GB/s** (section 1.4), PCIe 3.0
**~12 GB/s** (section 1.4), one expert slab **1.42 MiB** (section 7, `= 3 * 2048 * 512 * 3.8 / 8`),
~30 host layers, 256 experts, 8 used per token (published).

Union size at `m` tokens, from section 10.1's model `U(m) ~= 8 * (1 + (m-1)(1-o))`, and from a
uniform-routing estimate `256 * (1 - (1 - 1/256)^(8m))`:

| `m` | `U` at `o=0.7` | `U` uniform | `U` at `o=0.5` |
|---|---|---|---|
| 2 | 10.4 | 15.8 | 12 |
| 4 | 15.2 | 30.7 | 20 |
| 32 | 82.4 | 165 | 132 |
| 65 | 161.6 | 222 | 256 (capped) |

At `m = 65` the union is 63-100% of all experts. Two consequences:

- **Heat ranking stops mattering.** The cold op groups draws by expert and reads each touched
  expert's weights once per batch (`ggml-cpu-mul-mat-id-cold.c:136-149`, then the per-expert chunk
  loop at `:159`). When nearly every expert is touched anyway, holding the *popular* S experts is
  worth no more than holding *any* S experts. The measured 44.7% hit rate at `m=2` (section 15.9)
  does not transfer.
- **The ceiling on the tier's saving is `S / U(m)`.** At `S = 33` and `U = 222` that is **14.9% of
  expert traffic**, versus ~45% at `m=2`.

Against that, expert traffic is a far larger share of the batch at `m=65`: dense bytes are read
once regardless, expert bytes scale with `U`. Using section 7's 68/32 dense/expert split at `m=1`,
expert traffic is ~93% of the batch at `m=65` (derived). So 14.9% of expert traffic is ~13.8%
end-to-end on those steps - real, but a quarter of what R1 delivers.

And the R3 baseline itself is in question. Derived, one forward pass at `m=65`:

| Path | Bytes | Rate | Time |
|---|---|---|---|
| CPU cold path (tier or bypass) | `222 x 1.42 MiB x 30` = 9.2 GiB | 26 GB/s | ~380 ms |
| op-offload, whole tensors | `256 x 1.42 MiB x 30` = 10.7 GiB | 12 GB/s | ~950 ms |

**If that holds, op-offload is a loss at `m=65` on this box and `--no-op-offload` is a free win
independent of everything else in this document.** It is one flag (`common/arg.cpp:2929-2933`) and
zero code. Section 1.4 already derived the same asymmetry for the per-token case; nobody has run
it at `m=65`.

---

## 3. Step 0 - three zero-code experiments (gate)

All three run on `phase0/devbox-server-ab.sh` with a copy-heavy prompt (ngram-mod only earns its
keep on long verbatim runs - code, JSON, quoted text - so use one).

**E1 - op-offload A/B.** `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
--spec-ngram-mod-n-match 24`, with and without `--no-op-offload`, at `-ehs 0`. Answers section
2.1's table. If `--no-op-offload` wins, R3 is a CPU regime and the tier is in play; if it loses,
the tier would have to beat the GPU and steps 2-5 are almost certainly dead.

**E2 - fire rate.** From the same runs, what fraction of decode steps are R3 batches versus the
R1/R2 fallback? `ngram-mod` resets its table on a low-acceptance streak (`speculative.cpp:2010-2020`),
so the mix is workload-dependent. If R3 batches are a minority of *steps*, the pad cost in step 3
is paid on every R1 step to speed up a few R3 steps, and the sign of the whole plan flips.

**E3 - measured `U(m)`.** Replace section 2.1's derived unions with real ones. Extend the
`LLAMA_EXPERT_HITRATE` reporting in `log_hit_rate` (`src/llama-context.cpp:1580`) to also count
distinct experts per layer per batch, and drop its `expert_heat_batch` gate so it reports at any
`m`. Sweep `m` in {2, 8, 16, 32, 64} using `--spec-draft-n-max` on MTP - no tier changes needed,
the readback is independent of whether the tier engaged. This also finally measures the `o` that
section 10.1 says is "the next thing to run".

**Decision rule.**

- E1 says op-offload wins at `m=65` -> stop after step 1; document `-ehs` and wide ngram-mod as
  mutually exclusive.
- E3 says `U(65)/256 > 0.9` **and** E2 says R3 steps are a minority -> stop after step 1; the pad
  costs more than it returns.
- Otherwise -> continue, and set the step 3 ceiling target from E3, not from 65.

---

## 4. Step 1 - make the incompatibility non-silent (do regardless)

Today the only signal is a one-shot `expert tier bypassed` warning guarded by a process-wide
`static bool` (`src/llama-expert-tier.cpp:87-93`). It fires once, names one tensor, and says
nothing about which speculator caused it.

- At context init, compare `1 + common_speculative_n_max(&params.speculative)`
  (`common/speculative.cpp:2288`) against the tier ceiling and warn naming both flags - the draft
  width and `-ehs`. This is the check that would have answered the original question at startup.
- Add bypassed-batch count to the server's speculative metrics next to acceptance, so an A/B shows
  *why* a config underperformed rather than just that it did.

`phase0/devbox-server-ab.sh:63` already greps for the bypass line; keep the wording stable.

---

## 5. Step 2 - heat correctness at multi-token batches (prerequisite)

If the ceiling is lifted without this, `resync_top_s` starts chasing noise.

**5.1 Rejected drafts are counted.** `update_from_graph` reads every row of `moe_sel_experts`
(`src/llama-expert-heatmap.cpp:44-55`). ngram-mod acceptance is low by construction - most of the
48-64 drafted tokens are wrong - so the ranking would be dominated by experts that were never
actually used. At `m=2` this is a rounding error; at `m=65` it is the majority of the signal.

**5.2 Decay is per call, not per token.** `decay_all()` runs once per `update_from_graph`
(`:41`) while `tokens_total += n_tokens` at `:57`. A 65-token call decays once and adds 65 counts,
so the effective half-life differs by ~65x from single-token decode. In a bimodal stream (section
1.1) the ranking is set by whichever batch shape happens to dominate, not by usage.

**Fix (this is `PLAN-qwen36-35b-moe.md` section 12.3's M3, now a hard prerequisite rather than an
optimisation):** snapshot in `process_ubatch()` but commit on the following call once
`n_accepted` is known from `common_sampler_sample_and_accept_n`
(`tools/server/server-context.cpp:3954`), counting only rows `[0, n_accepted)`, and scale decay by
the number of tokens committed. `moe_sel_experts` already holds what is needed - a deferral, not
new plumbing.

Land this **before** step 3. It is independently useful: it improves MTP at `n_max` 2 and 3 today,
which is measurable against the section 15.9 table without any other change.

---

## 6. Step 3 - landing-pad remap, and a ceiling that is a variable

Take section 12.3 M5's **option 2** (landing pad), not option 1 (`mm_ids_helper` count+rank).
Rationale: option 1 touches a hot upstream kernel with an owner, `miltos22/llama-wackMall` already
has it (section 13.1), and per section 2.1 it would buy nothing option 2 does not.

**6.1 Hot tensor layout** (`src/llama-expert-hotstore.cpp:83`). Allocate
`hot_s + n_expert_used` slices. Reserve `[0, n_expert_used)` as the pad; real experts occupy
`[n_expert_used, hot_s + n_expert_used)`. The buffer is already fully zeroed at `:121`, so pad
rows contribute zero exactly like today's single sentinel - **no mask change, and the existing
per-expert scale multiply is untouched**.

Within a token, cold draws land on `0..n_expert_used-1` (distinct by draw position) and hot draws
on `>= n_expert_used` (distinct because distinct experts get distinct slots). Duplicates become
impossible by construction, on every dispatch path.

**6.2 LUT build** (`update_luts`, `src/llama-expert-hotstore.cpp:376`). Hot expert `e` ->
`slot + n_expert_used`; cold expert -> pad base. `slot_to_expert` and `dwell_count` keep their
current indexing; only the emitted LUT value shifts.

**6.3 In-graph id remap** (`src/llama-expert-tier.cpp:45`) - the one genuinely new piece. The id
must be `lut[e]` for hot draws and the **draw position `j`** for cold ones, and `j` is not
knowable from a per-expert LUT. Two ways:

- **Arithmetic (preferred, 2 extra ops).** Store the LUT as f32, reuse the `cold_mask` gather
  already built by `remap_mask` (`:62-73`), add a constant f32 arange `[1, n_expert_used, 1]`
  (broadcasts over tokens through `ggml_can_repeat`), compute `ids = lut + m*j`, cast to I32.
  **Open item: confirm F32 -> I32 is in the CUDA cpy/cast table.** If it is not, that cast is the
  only new ggml surface this plan needs, and it is an order of magnitude smaller than touching
  `mmq`.
- **No arithmetic (fallback, ~15 extra ops per call).** `n_expert_used` LUT variants, one
  `get_rows` per draw position against a 1-row view of `ids`, then `ggml_concat`. Works with
  today's ops. If taken, hoist the remap to once per layer - `gate`/`up`/`down` share the same
  `ids` and LUT and currently recompute it three times - and check the graph node budget
  (~600 extra nodes at 40 layers).

**6.4 The ceiling becomes a variable, not a constant.** Replace `LLAMA_EXPERT_TIER_MAX_TOKENS`
with a runtime value plus a CLI knob so it can be swept. **Do not remove it.** Section 2 gives two
natural stopping points:

- **31** - the conservative target. The tier owns R1+R2 and never fights op-offload. Reachable
  today by tuning ngram-mod to `n_min 16 / n_max 30`, which still buys a 30-token verbatim run per
  forward pass. This is the recommended first landing.
- **> 32** - only if E1 showed op-offload losing at `m=65`. Then the tier pinning the MoE to the
  CPU is a feature, and the ceiling goes to whatever E3's `U(m)` curve justifies.

**6.5 The cost, stated plainly.** `n_expert_used = 8` pad slots per layer, `8 x 1.42 MiB x 40` =
**~455 MiB** (matches section 12.3's estimate; the measured `33 slots = 1803 MiB` sizing message
implies 54.6 MiB/slot, so ~437 MiB). Against a `-ehs -1` budget of ~33 slots that is **roughly a
quarter of the resident set traded away** - and it is paid on R1 steps too, where the hit rate is
what delivers the measured +30.8%. E2's fire rate is what decides whether that is affordable.

**6.6 MMQ eligibility - confirmed, no work.** All four expert types in play (IQ3_XXS, IQ4_XS,
Q4_K, Q6_K) are MMQ-supported (`ggml/src/ggml-cuda/mmq.cu:266-292`), and the 1660 Ti is Turing
with 64 KiB smem, so `ggml_cuda_should_use_mmq` returns true at `:312`. The tiered path above 4
tokens will land on MMQ, not the host-sync fallback of section 1.3. Re-check if a hot tensor type
ever falls outside that list.

---

## 7. Step 4 - `-ehs -1` sizing

- **Autofit** (`common/fit.cpp:809`): `*n_expert_hot_s = s - 1` reserves exactly one sentinel;
  with a pad it becomes `s - n_expert_used`. Needs `n_expert_used` reachable in `fit.cpp` (it has
  `hp_nex` for `n_expert`; confirm the used-count is available).
- **Shrink floor** (`src/llama-context.cpp:929-963`): context growth can drop slots. It must never
  take `hot_s` below `n_expert_used + 1` or the pad is unsatisfiable. Add an explicit floor and
  log the refusal rather than silently under-allocating.
- **Output buffer growth**: `n_outputs_per_seq = 1 + common_speculative_n_max()`
  (`tools/server/server-context.cpp:50`) is **65** with ngram-mod versus 2 with MTP `n_max 1`.
  That is ~64 x `n_vocab` x 4 B of extra logits (~39 MiB at Qwen3.6's vocab), plus embeddings when
  MTP is also enabled. Confirm the fit sees this before it computes the slot budget, or `-ehs -1`
  over-commits the card. This is section 12.4 risk 4, made ~30x worse by ngram-mod's draft width.

---

## 8. Step 5 - validation

1. **Numerical oracle first.** Temp 0, fixed prompt, `-ehs 0` versus `-ehs -1`, token-for-token
   identical output. The tier is meant to be numerically transparent; any divergence above 4
   tokens is duplicate-id corruption and nothing else. Run this before looking at throughput.
2. **A/B matrix** on `phase0/devbox-server-ab.sh`: {ngram-mod, ngram-mod + MTP} x {`n_max` 16, 30,
   64} x {`-ehs 0`, `-ehs -1`} x {op-offload on, off}. The `expert tier bypassed` line should
   disappear below the new ceiling.
3. **Regression guard.** MTP `n_max 1` must not move from **20.11 t/s** (UD-Q3_K_M) / **23.95**
   (Q4_K_M) (measured, section 15.9). The pad costs slots, so a drop here is the expected failure
   mode and the thing that decides whether step 3 ships.
4. **CUDA graphs.** `ggml-cuda.cu:2525-2527` consults `get_mmvq_mmid_max_batch` when deciding
   whether `mul_mat_id` forces a stream sync and disables graph capture. Diff `graph splits` and
   any CUDA-graph disable message before and after (section 12.4 risk 1).
5. **Per-layer attribution.** With a UD quant mix, expert types differ per layer. Log engaged /
   bypassed per layer so a regression is attributable (section 12.4 risk 2).

---

## 9. Rejected alternatives

- **Fix `mm_ids_helper` for duplicates.** Correct for everyone, but it is a hot upstream kernel
  with an owner, `wackMall` already has the count+rank version (section 13.1), and section 2.1
  says it buys nothing the landing pad does not. Track it; do not duplicate it.
- **Cap ngram-mod to `n_min`/`n_max` <= 3.** That is not compatibility, it is disabling ngram-mod.
  Its entire value is long verbatim runs. (Tuning to <= 30 for the R2 landing in 6.4 is different:
  a 30-token run is still 30x a normal decode step.)
- **Split the verify batch into 4-token ubatches.** The cold op groups by expert
  (`ggml-cpu-mul-mat-id-cold.c:136-149`), so 16 ubatches re-read the cold weights 16 times.
  Strictly worse than today's bypass.
- **Dense hot matmul over all S slots plus a gather.** Removes ids entirely, but costs `S`x the
  FLOPs on the hot path (~640 GFLOP per forward pass at `S=33`, `m=65`, derived). Not viable on a
  1660 Ti.

---

## 10. Sequencing, effort, open questions

```
Step 0 (0.5d, GATE) -> Step 1 (0.5d) -> Step 2 (1d) -> Step 3 (1-2d) -> Step 4 (0.5d) -> Step 5 (1d)
```

Steps 1 and 2 stand on their own and improve today's MTP path whatever step 0 says. Step 3 is the
only commitment, and it is gated twice: on E1/E2/E3, and on the regression guard in 8.3.

**Open items to resolve before writing code:**

1. F32 -> I32 support in the CUDA cpy/cast table (decides 6.3's variant).
   **Resolved: supported.** `cpy.cu:592` dispatches it and `ggml-cuda.cu:4958` returns true for it in
   `supports_op`. `llama-graph.cpp:2048` already round-trips I32 -> F32 -> I32 for GROVEMOE, so 6.3's
   arithmetic variant needs no new ggml surface.
2. `n_expert_used` reachability in `common/fit.cpp` and the hotstore constructor.
   **Resolved: both reachable.** The hotstore ctor already takes `const llama_model *`, so
   `model->hparams.n_expert_used` is in hand. `fit.cpp` gets `hp_nex` out of
   `common_get_device_memory_data_impl`, which reads it via `llama_model_n_expert`
   (`llama-model.cpp:2850`, declared in `llama-ext.h:86`); a sibling `llama_model_n_expert_used` is
   a two-line addition next to it.
3. Whether `--no-op-offload` should simply become implied by `-ehs` above the op-offload threshold
   - E1 answers this, and if it does, section 6.4's ceiling choice gets much simpler.
   **Resolved, but not the way it was posed.** See 11.1: `--no-op-offload` is the wrong lever
   because it also takes prompt processing off the GPU. `GGML_OP_OFFLOAD_MIN_BATCH` moves the
   threshold instead and beats both.

---

## 11. Step 0 results (measured)

Box: RTX 3060 Laptop, 6 GiB, cc 8.6, PCIe 4.0 x8, i9-11900H, 64 GiB RAM. **Not** the 1660 Ti of
section 6.6 - that is the home server, and nothing here has been run on it.

Model: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_S`. `n_layer 40`, `n_expert 256`,
`n_expert_used 8`. Expert tensors are **all Q4_K**, no i-quants, so section 6.6's MMQ conclusion
holds with room to spare. One slot is 1769472 B = 1.6875 MiB per layer, **67.5 MiB per slot** over
40 layers; autofit picks `S=32`, i.e. a 2160 MiB store.

Workload: `phase0/prompts/copy-heavy.txt` primed with two copies, model continues copying. 1018
token prompt, 900 tokens generated, temp 0. Copy-heavy on purpose - ngram-mod only fires on long
verbatim runs, so prose measures nothing. Draft statistics are bit-identical across every arm
below, which is the sanity check that only the MoE placement changed.

Harness: `phase0/ngram-mod-ab.sh` (cases `e1`, `e1b`, `e3`, `e4`), report in
`phase0/ngram-mod-report.py`, raw output in `phase0/results/devbox/ng-q4ks-*`.

### 11.1 E1 - op-offload is a loss at m=65, but `--no-op-offload` is the wrong fix

| arm | pp t/s | tg t/s |
|---|---|---|
| no speculation | 146.95 | 28.21 |
| ngram-mod 48/64/24, op-offload on (default) | 134.94 | 46.14 |
| ngram-mod, `--no-op-offload` | 73.93 | 69.15 |
| ngram-mod, `GGML_OP_OFFLOAD_MIN_BATCH=128` | 150.66 | **69.88** |
| ngram-mod, `GGML_OP_OFFLOAD_MIN_BATCH=512` | 100.02 | 68.63 |

Section 2.1 was right about the direction: taking the MoE off the GPU at m=65 is worth +51% tg.
Backing the per-step cost out of the totals, a wide verify batch goes from ~1.28 s to ~0.81 s;
section 2.1 derived 950 ms vs 380 ms, so the ratio is 1.6x, not 2.5x.

Section 2.1 was wrong that this is free. `--no-op-offload` halves prompt processing, because a
512 token prompt ubatch is exactly the batch op-offload exists for. The threshold is an env var
(`ggml-cuda.cu:5377`, upstream `9a5724dee` / PR #18535), and moving it to 128 - above the 65 token
verify batch, below the 512 token prompt ubatch - is the best pp **and** the best tg measured.
512 is worse than 128 because a 1018 token prompt splits 512 + 506 and the second chunk falls
below the line.

**Gate: E1 does not stop the plan.** R3 is a CPU regime for decode.

### 11.2 E2 - fire rate

14 of 58 draft calls fired (24.1%), mean width 63.2 tokens, 95% acceptance (841/885 tokens).

R3 batches are 14 of ~60 decode steps. That is a minority of **steps** (23%) and a majority of
everything else: they carry 841 of 900 tokens and ~90% of decode wall time. Section 3's stop rule
says "R3 steps are a minority", but its stated reason - "the pad cost is paid on every R1 step to
speed up a few R3 steps" - does not survive the time split. R1 steps are ~10% of decode time here.

### 11.3 E3 - measured `U(m)`, and a readback bug that had to be fixed first

The first E3 run returned `union` = exactly `8*m` at every m, which is impossible. Cause:
`ggml_argsort_top_k` returns a **view** of the full `[n_expert, n_tokens]` argsort
(`ggml.c:5387`), so its rows are `n_expert` apart while `ne[0]` is `n_expert_used`. A single flat
`ggml_backend_tensor_get` therefore reads the first `8*m` ints of **token 0's full 256-entry
ranking**, not the top 8 of each token.

This was not only in the new diagnostic. `llama_expert_heatmap::update_from_graph` had the same
flat read, so **every multi-token heat update since `a02f14d05` has been feeding the ranking
token 0's mid-ranked tail** - experts the router did not select - instead of the other tokens'
selections. Correct at `n_tokens == 1`, wrong for every batch above it. Fixed by
`llama_expert_read_sel_ids` (one read per row). Consequence for this document: **every hit rate
measured at m >= 2, including section 15.9's, needs re-measuring.**

With the fix, mean distinct experts per layer per batch:

| m | `U` measured | `U`/256 | derived, `o=0.7` | derived, uniform | hit rate at `S=32` |
|---|---|---|---|---|---|
| 1 | 8.0 | 3% | 8 | 8 | 18-36% |
| 2 | 14.6 | 6% | 10.4 | 15.8 | 53.6% |
| 4 | 25.5 | 10% | 15.2 | 30.7 | 53.5% |
| 8 | 42.6 | 17% | - | - | 25.9% |
| 16 | 66.6 | 26% | - | - | 26.6% |
| 32 | 94.6 | 37% | - | - | 26.6% |
| 64 | 123.5 | 48% | 161.6 | 222 | 25.3% |
| 512 | 197.9 | 77% | - | ~256 | 14.3% |

Three things fall out.

- **`U(64)/256 = 0.48`, not > 0.9.** Section 3's second stop condition does not fire under either
  reading. Both models in section 2.1 badly overestimate the union; real routing is far more
  concentrated than uniform, and even a 512 token batch touches only 77% of experts.
- **Heat ranking really does stop mattering at wide batches, exactly as 2.1 argued.** At m=64 the
  measured hit rate is 25.3% and `S/U` is `32/123.5` = 25.9%. Holding the *popular* 32 experts is
  worth what holding *any* 32 would be. At m=2, `S/U` is above 1 while the hit rate is 53.6%, so
  there the binding constraint is ranking quality, not capacity.
- The overlap `o` that section 10.1 wanted is not a constant. Fitting `U = 8(1 + (m-1)(1-o))`
  gives `o` = 0.18 at m=2, 0.27 at m=4, 0.38 at m=8: a saturating curve, not a linear one.

### 11.4 E4 - which draft width is actually fastest

`-ehs 0`, `GGML_OP_OFFLOAD_MIN_BATCH=128`, diagnostics off, `n_min == n_max` so the verify batch
is exact. (E3's tg column is depressed by the per-decode `synchronize()` the hit rate readback
forces; use this table, not that one.)

| verify batch | tg t/s | fire rate |
|---|---|---|
| 2 | 31.69 | 90.7% |
| 4 | 39.67 | 82.9% |
| 8 | 47.15 | 71.1% |
| 16 | 57.59 | 55.1% |
| 32 | 64.09 | 38.0% |
| 64 | **70.88** | 24.1% |

Monotone. Wider always wins on this workload, and it is still climbing at 64.

### 11.5 What this does to the plan's economics

The zero-code configuration - ngram-mod 48/64/24 plus `GGML_OP_OFFLOAD_MIN_BATCH=128` at
`-ehs 0` - is **70.88 t/s against a 28.21 t/s baseline, 2.5x, for no VRAM and no code.**

Against that, the tier's marginal value, derived from 11.3:

- The pad costs `n_expert_used` = 8 of 32 slots, **540 MiB**, leaving `S = 24`.
- At m=64 the tier can save at most `S/U` = `24/123.5` = 19.4% of expert reads. Expert bytes are
  ~88% of the batch at m=64 (dense read once, expert traffic scaling with `U`), so ~17% of a wide
  step, on the ~90% of decode time those steps own: **~15% end to end, ~82 t/s.**
- Landing at 31 instead (section 6.4's conservative option) starts from 64.09 rather than 70.88
  and projects to ~76 t/s, i.e. **worse than doing nothing and drafting 64 wide**.

And the tier's *existing* win is largely superseded rather than added to: +30.8% at m=2 turns
31.69 t/s into ~41 t/s, well below 70.88. The hot store was worth a lot when decode was 1-2 tokens
wide. Once a wide draft works, the wide-batch path owns the wall clock and the store's marginal
value falls to the ~15% above.

**Gate outcome: neither stop condition in section 3 fires.** E1 says op-offload loses, and E3 says
`U(64)/256` is 0.48. The plan may proceed. Whether ~15% is worth steps 2-5, ~250 lines in the
graph builder and 540 MiB of a 6 GiB card is a judgement call, not a gate, and section 6.4's
"land at 31 first" option is now clearly the wrong first landing.

---

## 12. OPEN: the tier is not numerically transparent, and nobody knows why

**This is the highest-priority open item in this document.** It predates every change made for
the ngram-mod work and it affects the whole expert cache, not just the wide-draft case.

### 12.1 What was measured

`phase0/oracle-logprobs.sh`, UD-Q4_K_S, prompt "The three laws of thermodynamics state that",
one generated token, top-40 logprobs. **Both arms carry `-cmoe`**, so every MoE weight is on the
CPU in both and only the tier differs. Getting this wrong was the first false alarm: a bare
`-ehs 0` reference leaves part of the MoE on the GPU, and that placement difference alone
accounted for roughly a third of the shift originally reported here.

| arm | max abs delta | mean abs delta | ranks moved | shared top-k |
|---|---|---|---|---|
| `-ehs 1  --expert-hyst 0 --expert-dwell 0` | 1.936e-1 | **5.469e-2** | 20 of 39 | 39 of 40 |
| `-ehs 32 --expert-hyst 0 --expert-dwell 0` | 2.478e-1 | **6.154e-2** | 24 of 38 | 38 of 40 |
| `-ehs -1 -fitt 256` | 2.363e-1 | **7.020e-2** | 19 of 38 | 38 of 40 |

`-ehs -1` is deterministic: two runs of one config are byte-identical, so none of this is noise.

### 12.2 Why this is not acceptable as-is

The tier is meant to compute the same thing as stock `ggml_mul_mat_id`. It holds bit-identical
copies of the same quantized expert slices and only splits the routed sum into a hot half and a
cold half. A mean shift of 7e-2 in logprob is orders of magnitude above what reassociating one
matmul produces.

**The decisive clue is that the error is flat in S.** At `-ehs 1` exactly one expert per layer is
resident, so at most 1 of 8 draws per token takes the GPU hot path and the other 7 stay on the
CPU. At `-ehs 32` far more traffic goes hot. If the shift came from the hot path being a
different kernel, it would grow with S. It does not move.

What *is* constant across both is that the cold half runs through `ggml_mul_mat_id_cold`
(`ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.c`) instead of stock `ggml_mul_mat_id`. That kernel
is a copy of the stock one with a `cold_mask[i02] == 0 -> continue` filter, but the stock path can
take a llamafile/tinyBLAS route for some shapes that the copy does not. Two different CPU kernels
for the same arithmetic differ by far more than last-bit rounding, and that would be benign.

That is a hypothesis, not a finding.

### 12.3 Ruled out so far

- **Not nondeterminism.** Two `-ehs -1` runs are byte-identical.
- **Not a dirty sentinel slot after a shrink.** `shrink()` re-allocates through `allocate()`,
  which calls `ggml_backend_buffer_clear(buf, 0)`.
- **Not gross id corruption.** The argmax holds and 38 of 40 top-k tokens survive.
- **Not (only) tensor placement.** Matching `-cmoe` cut the shift by about a third.
- **Not the GPU hot path.** Flat in S, and the cold-only run below settles it.
- **NOT THE HOT/COLD SPLIT.** With `LLAMA_EXPERT_TIER_COLD_ONLY=1` every expert is kept cold,
  so the hot tensor is all zeros and contributes exactly nothing, and the whole MoE goes through
  `ggml_mul_mat_id_cold`. The shift is **7.226e-2**, the same size as with the tier fully active
  (7.020e-2 at `-ehs -1`, 5.469e-2 at `-ehs 1`). The routing, the ids, the mask and the sum are
  therefore not the source. **This is the result that clears the landing-pad work in step 3**:
  step 3 changes the ids, and the ids are not what is moving the logits.
- **Not llamafile/tinyBLAS.** Neither stock `ggml_compute_forward_mul_mat_id` nor the cold copy
  calls `llamafile_sgemm`; only plain `mul_mat` does. This was the leading hypothesis and it is
  wrong.
- **Not the per-expert scale.** Both apply `w_s` after the matmul with a per-expert gather
  (`llama-graph.cpp:1534-1540` against `llama-expert-tier.cpp:135-143`). The gather is spelled
  differently, the values are the same.

Diffing the two kernels leaves only three differences: a `memset` of `dst` (needed, since the
cold op writes a subset of rows), the `cold_mask[i02] == 0 -> continue` filter, and deleted
comments. The quantization of the activations, the `vec_dot`, and the chunking are the same code.

**So two kernels with identical compute paths are producing a 7e-2 logprob difference, and that
is still unexplained.**

### 12.35 Resolved: the tier moves the MoE onto the GPU

Measured with `GGML_SCHED_DEBUG=2` on `-ehs 0 -cmoe` against `-ehs 1 -cmoe`, comparing the last
decode graph in each by node name. Identifying the right graph matters: the early dumps have 2355
nodes with `ffn_moe_down-0` a `MUL_MAT_ID` on the CPU, which is the pre-fill graph before the tier
engages; the late dumps have 3315 nodes with `ffn_moe_down-0` an `ADD` on CUDA0, which is the
tiered one.

174 shared nodes change backend. 160 of them are the whole MoE, every layer:

| node family | stock | with the tier | count |
|---|---|---|---|
| `ffn_moe_gate` | CPU, MUL_MAT_ID | CUDA0, ADD | 40 |
| `ffn_moe_up` | CPU, MUL_MAT_ID | CUDA0, ADD | 40 |
| `ffn_moe_down` | CPU, MUL_MAT_ID | CUDA0, ADD | 40 |
| `ffn_moe_swiglu` | CPU, SWIGLU | **CUDA0, SWIGLU** | 40 |

Three of the four changed identity as well as device: the node carrying the name is now the tier's
`ggml_add(hot, cold)` rather than the stock `mul_mat_id`, and the cold half necessarily stays on
the CPU because `GGML_OP_MUL_MAT_ID_COLD` has no CUDA implementation. `ffn_moe_swiglu` is the
clean case: the same op, same inputs, relocated from CPU to CUDA0 in all 40 layers. It is a
nonlinearity over the whole FFN intermediate, so a CUDA-versus-CPU difference there lands in the
residual stream of every layer.

(The remaining 14 are auto-named `node_NNNN` entries. Those names are positional and the two
graphs have different node counts, so they are name collisions, not real moves.)

The mechanism: stock keeps the MoE on the CPU because its weights are host-resident. The tier's
hot half produces a CUDA tensor, so the scheduler pulls the SwiGLU and the projections onto the
GPU with it. Every op is correct. CUDA and CPU implementations of SWIGLU and MUL_MAT_ID differ in
the last bits, and 40 layers of that compounds into a mean 7e-2 logprob shift.

This explains every earlier observation at once: the shift is flat in S because placement does not
depend on how many experts are resident, and it survives forcing the hot path to contribute zero
because the graph shape, and therefore the placement, is unchanged.

**Conclusion: benign.** The same class of difference as choosing the GPU over the CPU for those
ops in the first place. Not a routing error, not the landing pad, not the ids.

### 12.36 A methodological note worth keeping

Three comparisons in this investigation were invalid because the arms differed in more than the
one variable:

1. `-ehs 0` against `-ehs -1` without `-cmoe` on both: differed in tensor placement, worth about a
   third of the shift.
2. The `graph splits = N` line: printed by `sched_reserve` during init, before the store is
   allocated and long before the tier engages, so both arms report the same pre-tier graph.
3. Comparing the first scheduler dump instead of the last: the early dumps are pre-fill graphs
   with the tier bypassed.

On this codebase, "same flags except one" usually is not.

### 12.4 What to run next, cheapest first

1. **Force the tier cold-only.** The one diagnostic that closes 12.2: make `cold_mask` all ones
   and point `hot_lut` at the pad, so the tier computes the whole MoE through
   `ggml_mul_mat_id_cold` and the hot path contributes nothing. Compare that against stock. Any
   remaining difference is `mul_mat_id_cold` versus `mul_mat_id` and nothing else. If it accounts
   for the whole 7e-2, the tier's split is exonerated and the issue is a kernel choice.
2. **Perplexity over a fixed corpus.** Tells you whether any of this costs quality, whatever the
   cause. Needs the `llama-perplexity` target.
3. **Scale-factor audit.** `llama_expert_tier_build` applies `w_s` to both paths after the
   matmuls. Check stock `build_lora_mm_id` applies it at the same point and to the same operand.


---

## 13. Step 3 results (measured)

Landing pad in (`eed82c188`), limit settable (`cbbb8dd99`). UD-Q4_K_S, copy-heavy prompt,
ngram-mod `n_min 48 / n_max 64`, `-ehs -1 -fitt 256`, arms interleaved against thermal drift.

### 13.1 The pad is a no-op at the old limit

First-token logprobs against stock are unchanged to the last digit, before and after the pad:
5.469e-2 at `-ehs 1`, 6.154e-2 at `-ehs 32`. The arithmetic remap computes exactly what the
sentinel scheme computed.

### 13.2 The tier works above 4 tokens

| | limit 4 | limit 65 |
|---|---|---|
| bypassed batches | 17 | **2** (the prompt ubatches) |
| generated text | - | **byte-identical to limit 4**, 4238 chars, both reps |
| tg t/s | 57.70, 54.29 | **73.38, 72.75** |
| pp t/s | 118.83, 116.62 | 131.92, 134.18 |

**+30.5% on tg**, well outside this box's ~10% noise, consistent across interleaved reps. The
duplicate-id hazard of section 1.3 is closed: a 65 token batch through the tier produces the same
tokens as a 65 token batch around it.

### 13.3 But it only breaks even against not using the store at all

Same workload and the same implied `--op-offload-min-batch 128`:

| config | tg t/s |
|---|---|
| no speculation, `-ehs 0` | 28.21 |
| ngram-mod, `-ehs 0` (no hot store) | 69.88, 70.88, 74.02 -> median **70.9** |
| ngram-mod, `-ehs -1`, limit 4 | 57.70, 54.29 -> median 56.0 |
| ngram-mod, `-ehs -1`, limit 65 | 73.38, 72.75 -> median **73.1** |

73.1 against 70.9 is **inside the noise band**. Raising the limit recovers what `-ehs -1` was
losing; it does not beat leaving the hot store off.

Section 11.5 projected ~82 t/s from `S/U` arithmetic. That over-estimated, and the likely reason
is that the arithmetic counted only the expert bytes saved and ignored what the split costs: the
tier puts a GPU op and a CPU op inside every MoE, so the activations cross the bus both ways, 3
matmuls x 40 layers x 65 tokens x 2048 x 4 B per direction. At m=65 that traffic is the same
order as the expert bytes the hot slots save.

**So on this workload the hot store is not paying for itself at m=65, whatever the limit.** What
it is worth at m=2, where the store covers the whole union and `S/U` is above 1, is a different
question and is the one the MTP `n_max 1` regression guard actually measures.

---

## 14. The combined config: MTP n_max 1 + ngram-mod + `-ehs -1` (measured)

`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 1 --spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, UD-Q4_K_S, copy-heavy prompt, 900 tokens,
arms interleaved, warm page cache. Every arm produced **byte-identical output** (4238 chars), so
only speed differs.

Both speculators fire, as the impl order predicts. ngram-mod takes the copy runs and MTP covers
the rest:

| impl | fire rate | mean draft width | accepted |
|---|---|---|---|
| ngram-mod | 37.8% (14/37 calls) | 63.1 tok | 839/884 |
| draft-mtp | 100% (23/23 calls) | 1.0 tok | 23/23 |

Together they turn 900 tokens into ~39 decode steps, mean accepted length 24.3.

| config | tg t/s (median) | runs |
|---|---|---|
| no speculation, `-ehs 0` | 28.2 | 1 |
| MTP `n_max 1` + `-ehs -1` | 35.2 | 34.8, 35.3, 35.7 |
| combined, `-ehs -1`, limit 4 (today's default) | 65.4 | 63.4, 64.6, 66.2, 69.1 |
| combined, `-ehs 0` + `--op-offload-min-batch 128` | 70.0 | 69.7, 70.3 |
| ngram-mod alone, `-ehs 0` + threshold 128 | 70.9 | 69.9, 70.9, 74.0 |
| **combined, `-ehs -1`, limit 65** | **78.1** | 77.2, 77.8, 79.0 |

**This is the best configuration measured anywhere in this document: 78.1 t/s, 2.8x the
no-speculation baseline and 2.2x MTP-with-cache.**

### 14.1 It is also the first config where the hot store pays for itself

Section 13.3 found the store roughly break-even with ngram-mod alone. Here it is not:

- against the same config without the store, 78.1 vs 70.0 = **+11.6%**
- against the same config with the store but the old limit, 78.1 vs 65.4 = **+19.4%**

The within-arm spread is 2.3% for the winner and 0.8% for the no-store arm, so +11.6% is real and
not the thermal noise that swamped the earlier single-run comparisons.

The reason the store earns its keep here and not in section 13.3: with MTP filling in, *every*
decode step is a batch the tier can serve - 2 tokens when MTP drafts, 65 when ngram-mod does -
provided the limit clears 65. At limit 4 the tier serves only the MTP steps and bypasses the wide
ones, which is the 65.4 row. Raising the limit is what converts the store from dead VRAM into a
win on this workload.

### 14.2 Caveat on the first run of a batch

`combo-cache65` first measured 58.7 t/s with pp 91.5, against 128-140 pp everywhere else. That was
a cold page cache on a 21 GB file, not a property of the config; re-run warm it gives 79.0. Any
A/B on this box should discard or warm up the first run, on top of interleaving the arms.

---

## 15. n-gram speculation on ordinary prompts (measured)

Section 14's 78.1 t/s was on a deliberately copy-heavy prompt at temp 0. This section asks the
question that actually matters day to day: does any n-gram setting help on normal work?

Workload: 5 prompts (a factual question, "write a python script", a code review, a concepts
explanation, a tradeoffs summary), ChatML-wrapped, the user's sampling (temp 0.6, top-p 0.95,
top-k 20), fixed seed, n_predict 400 each, 2000 generated tokens per arm. `-ehs -1
--expert-tier-max-tokens 65 --flash-attn on` throughout, MTP `n_max 1` always on.
Harness: `phase0/ngram-ordinary.sh`.

### 15.1 Nothing fires

| speculator and setting | fire rate | draft width | accepted | tokens gained per 2000 |
|---|---|---|---|---|
| ngram-mod `n_match 24, n 48-64` | **0.0%** of 1095 calls | - | - | **0** |
| ngram-mod `n_match 24, n 4-16` | 0.0% | - | - | 0 |
| ngram-mod `n_match 16, n 4-16` | 0.1% | 16.0 | 43.8% | 7 |
| ngram-mod `n_match 16, n 2-8` | 0.1% | 8.0 | 87.5% | 7 |
| ngram-mod `n_match 8, n 2-8` | 0.6% | 7.3 | 50.0% | 22 |
| ngram-mod `n_match 8, n 1-4` | 1.0% | 4.0 | 72.7% | 32 |
| ngram-cache (defaults) | 3.9% | 1.0 | 52.3% | 23 |
| ngram-simple `n 8, m 8` | 0.5% | 8.0 | 45.0% | 18 |
| ngram-map-k4v `n 8, m 8` | 0.1% | 8.0 | 37.5% | 3 |

The user's config **never fires once in 1095 draft calls**. Loosening `n_match` and `n_min` raises
the fire rate to at most 4%, and the best any of them contributes is ~30 accepted tokens out of
2000. There is no setting that helps, because prose contains nothing to match: an n-gram
speculator needs the continuation to already exist verbatim in the context, and ordinary answers
do not repeat themselves.

This is not a tuning failure. It is the mechanism working as designed on a workload it does not
suit.

### 15.2 What it costs to leave enabled: about 1%, i.e. nothing

Measured ABBA, MTP-only against the wide ngram-mod config, stratified by the artifact in 15.3:

| state | MTP only | + ngram-mod |
|---|---|---|
| fast | 31.38, 30.25 | 30.13 |
| slow | 27.69 | 27.94, 27.53 |

Within the fast state MTP-only leads by 2.3%; within the slow state ngram-mod leads by 0.2%. The
idle cost is not resolvable and is at most ~1%.

**Recommendation: leave ngram-mod enabled.** It costs about 1% on prose and is worth 2.2x on
copy-heavy work (section 14), so the trade is strongly favourable as long as any of the session is
editing, refactoring or reproducing existing text.

### 15.3 A benchmark artifact that invalidated two sweeps, unexplained

Successive server runs alternate between two throughput states, **by run position, not by config**:

| position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pp t/s | 57.4 | 46.7 | 55.4 | 46.5 | 55.1 | 45.6 |
| tg t/s | 31.4 | 27.9 | 30.1 | 27.7 | 30.3 | 27.5 |

A ~20% swing in pp and ~10% in tg, alternating cleanly, whatever arm occupies the slot. Free VRAM
is identical (5130 MiB) in every run, the hot store is S=20 in every run, and host RAM free is flat
at ~52.5 GB, so it is none of those. `predicted_ms` is measured inside the server, so it is real
generation speed and not harness overhead. pp and tg move together, which points at a CPU power or
thermal state on this laptop rather than anything being configured. **Not explained.**

Two consequences, both learned the hard way here:

- **ABAB interleaving is worse than useless against this.** Alternating arms puts one arm on every
  odd position and the other on every even one, so the artifact maps perfectly onto the arm and
  looks like a clean result. That is exactly how sweep 1 concluded MTP-only was fastest and sweep 2
  concluded the opposite.
- Any comparison below ~10% on this box needs either ABBA-with-parity-stratification as in 15.2,
  or enough reps to average the state out.
