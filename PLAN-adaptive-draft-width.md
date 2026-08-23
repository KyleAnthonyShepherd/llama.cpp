# PLAN - adaptive speculative draft width under the expert hot store

Goal: replace the hand-tuned `--spec-type draft-mtp --spec-draft-n-max 1` with a controller that
picks the draft width per decode step, in `[0, 3]`, from what the expert hot store already measures.
Width 0 means: skip speculation this step, because verifying the extra tokens would drag more cold
experts across the host bus than the accepted tokens are worth.

**Why 3 and not 4.** A verify batch is `1 + width`, and after `448744c89` dropped the landing pad a
token id list holds duplicates again (`src/llama-expert-hotstore.h:22-31`), so the 4-token tier
limit is back to being a **correctness** bound, not a performance one - the MMQ/MMF id compaction
miscounts duplicates. `common_init_result` says so in the code that same commit rewrote
(`common/common.cpp:1269-1273`), and the auto-follow of the draft width introduced in `b153688b4`
was removed there. So the widest safe draft under `-ehs` today is 3. Section 5.2 covers what it
would take to get 4 back.

Same 6 GB VRAM / 16 GB RAM box and same model as `PLAN-qwen36-35b-moe.md`:
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_S`, `n_layer 40`, `n_expert 256`, `n_expert_used 8`,
~1.42 MiB per expert per layer. Companion to `PLAN-ngram-mod-expert-cache.md`, which covers the
*wide* (49-65 token) regime; this plan is entirely inside the tier own regime.

No code changed. Every structural claim cites a file:line in this tree. Every number is marked
**measured**, **published**, **derived** or **to measure**.

---

## 0. What this document decides

| Step | What | Code | Decides |
|---|---|---|---|
| 0 | Fixed-width sweep, plus a within-run phase split | script only | whether adaptivity has anything to win |
| 1 | Honour `dp.n_max` inside the MTP draft loop | ~4 lines | enabling fix, do it regardless |
| 2 | Expose the hot store per-decode cold count | ~50 lines | the controller only new input |
| 3 | The controller itself, in `get_n_draft_max()` | ~90 lines | the feature |
| 4 | Flags, tier-limit interaction, logging | ~30 lines | shippability |
| 5 | Validation against fixed width 1 | scripts | ship / revert |

**Step 0 can kill steps 2-5 and costs nothing but GPU hours.** If the best fixed width is the same
in every phase of a run, a controller can at best tie it. Do not skip it.

---

## 1. Why draft width costs what it costs

### 1.1 The cold path reads each expert once per batch, not once per token

`llama_expert_tier_build` splits the MoE into a GPU hot matmul and `ggml_mul_mat_id_cold`
(`src/llama-expert-tier.cpp:150-158`), a CPU op that computes **only** the cold-selected experts.
That op groups rows by expert exactly the way stock `mul_mat_id` does: it fills
`matrix_row_counts` / `matrix_rows` (`ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.c:134-149`) and
then loops `for (int cur_a = 0; cur_a < n_as; ++cur_a)`, skipping any expert with zero rows
(`:159-165`).

**Consequence, and it is the load-bearing fact of this whole plan:** the host RAM traffic of a
verify batch is proportional to the number of **distinct cold experts** the batch touches, per
layer, not to `n_tokens x n_expert_used`. Two tokens that route to the same cold expert pay for it
once.

### 1.2 So width is cheap exactly when the draws repeat, or land hot

With `n_expert 256` and `n_expert_used 8`, independent draws give a distinct-expert count of
`256 x (1 - (1 - 8/256)^m)` for a batch of `m` tokens (**derived**): 8, 15.8, 23.3, 30.5 for
`m` = 1..4, the range the tier limit allows. That is the ceiling, and real routing is more
correlated than independent, so the true curve sits under it.

Split that union into hot and cold. The hot half saturates fast - the store holds the top experts
by heat, and those are precisely the ones that repeat across adjacent tokens, so `m` tokens touch
barely more hot experts than one does. The cold half does not saturate: cold draws are spread over
the ~220 experts the store does not hold, and they rarely collide. To first order

```
cold_distinct(m)  ~=  m * n_expert_used * (1 - h)      per layer
```

where `h` is the hot hit rate. **Derived**, and it is a linear model whose slope the hit rate
controls. That is the intuition behind this request, made quantitative: a high hit rate flattens
the slope and wide drafts pay; a low hit rate steepens it and they do not.

### 1.3 What that does NOT say

It does not say wide drafts never pay. The per-batch time is

```
T(m)  ~=  T0  +  alpha * n_layer * cold_distinct(m)
```

and `T0` is not small: at `m = 1` the CPU cold op runs 40 layers of 8 skinny matmuls, which is
thread-dispatch and latency bound, nowhere near saturating host bandwidth. A wider batch amortizes
`T0` across more tokens. **The trade is real in both directions, so it has to be measured, not
assumed.** That is what step 3 fits online, and it is why step 0 comes first.

Note also that the ratio `cold_distinct(m)/cold_distinct(1) = m` is independent of `h` under the
linear model. `h` enters only through the absolute slope, that is, through how much of `T(m)` is
`T0`. A pure ratio argument would say speculation never pays; the measured fact that MTP + `-ehs`
is the fastest configuration on this box (`7d7bf8450`) says `T0` dominates at `m = 1`. Both are
consistent.

### 1.4 Drafting itself is nearly free

The Qwen3.5/3.6 MTP block is a full decoder block **with its own MoE FFN** - `ffn_gate_inp`,
`ffn_down_exps`, gate/up exps, all `n_expert` wide (`src/models/qwen35moe.cpp:126-134`). So each
draft step does route over the same 256-expert pool. But it is **one** layer against the trunk 40,
so a draft step costs roughly 1/40 of a verify token. The cost this plan controls is the verify
batch, not the draft.

---

## 2. Every signal the controller needs is already collected

### 2.1 Hit rate and union - `llama_expert_hotstore::log_hit_rate`

`src/llama-expert-hotstore.cpp:466-506` already computes, per decode, hits over total and the mean
distinct experts per layer. Today it is gated behind `getenv("LLAMA_EXPERT_HITRATE")` and it pays
for its own `synchronize()` plus a second readback (`src/llama-context.cpp:1728-1733`).

It does not have to. The deferred-heat path already reads every router selection off the graph into
host memory on every decode: `llama_expert_read_sel(res->moe_sel_experts, expert_heat_held)`
(`src/llama-context.cpp:880`), after a `synchronize()` at `:1715`. `expert_heat_held` is a
`llama_expert_sel` holding `[n_expert_used * n_tokens]` ids per layer
(`src/llama-expert-heatmap.h:17-24`). Counting cold draws out of that is a scan over
`40 x 8 x m` int32s with a `slot_of()` lookup each - **free next to the D2H copy that already
happened.**

### 2.2 Per-position acceptance - `n_acc_tokens_per_pos`

`common/speculative.cpp:150`, maintained in `common_speculative_accept` (`:2711-2717`) and printed
by `common_speculative_print_stats` (`:2784-2788`). Exactly the per-depth acceptance the controller
needs.

Caveat: it is a **cumulative, per-impl, all-sequences** counter. The controller needs a decaying
per-slot version. That is a small addition on the server side rather than a change to the impl
(section 4.3).

### 2.3 Time

`server_slot` already times decode. Nothing new.

**Nothing in this plan requires a new forward pass, a new readback, or a prediction of what the
target model will route.** That is deliberate: predicting a token routing across 40 layers before
running them is neither cheap nor reliable, and the linear model in 1.2 says it is not needed - the
quantity that matters is the *rate*, which is a property of the current region of the generation,
not of the individual token.

---

## 3. The enabling bug: `dp.n_max` is ignored by the MTP draft loop

`common_speculative_draft_params::n_max` is documented as the per-step override
(`common/speculative.h:41-43`) and the server sets it from `get_n_draft_max()`
(`tools/server/server-context.cpp:3317-3319`). The *simple* draft impl honours it inside its loop:

```c
                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
```

(`common/speculative.cpp:349-350`). The **MTP** impl does not - it tests `params.n_max` only
(`:1668`), so it runs every configured draft step and the result is truncated afterwards by the
generic pass at `:2663-2667`.

Today that costs nothing, because `--spec-draft-n-max 1` makes the two equal. Under a controller it
means every step pays for `ceiling` MTP decodes and throws `ceiling - W*` away. **Fix `:1668` to match `:349-350`**, and
check `:816` and `:1147` for the same shape in the other impls.

Worth doing on its own merits: the parameter is documented as an override and silently is not one.

---

## 4. The controller

### 4.1 Where it goes

`server_slot::get_n_draft_max()` (`tools/server/server-context.cpp:465-488`) is already the single
place that bounds the draft, and the caller already treats `0` as "do not draft at all"
(`:3293-3295`). So the controller is a `std::min` at the end of that function. No new call site, no
change to the draft dispatch.

### 4.2 The rule

Per slot, keep `T0` and `alpha` fitted by a two-parameter online least squares over the last N
decodes of `(cold_distinct_total, t_decode_us)`, and per-position acceptance `a_i` as a decayed
rate. Then choose

```
W* = argmax over W in [0, W_ceiling] of   G(W) / Tpred(1 + W)

G(W)        = 1 + sum_{i<W} a_i
Tpred(m)    = T0 + alpha * cold_rate * m
cold_rate   = measured cold_distinct_total of the last batch, divided by its own m
```

`cold_rate` is the only term that carries the hot store into the decision, and it is exactly the
quantity this request is about: when the generation moves into a region the store does not cover,
`cold_rate` rises, `Tpred` steepens, and `W*` falls to 0 on its own. No threshold to tune.

Guards, all of them necessary:
- **Warm-up.** Until the fit has N samples spanning at least two distinct widths, return the
  configured `n_max`. A degenerate fit at a single width has no slope.
- **Forced exploration.** The fit only sees widths that were used. Every K steps (K ~ 32), take
  `W* + 1` or `W* - 1` regardless, so the regression keeps a spread. Without this the controller
  locks onto whatever it tried first.
- **Width 0 has to be a sample, not a prediction.** *This paragraph was wrong in the first draft
  and the smoke run caught it.* A step that drafts nothing never reaches the accept block, so the
  obvious implementation skips it and the fit sees only `m >= 2`. Predicting `m = 1` is then an
  extrapolation below everything measured, and since `t0` trades freely against `slope` in a
  two-parameter fit, it lands on "the narrowest batch is nearly free". **Measured**: the controller
  ran width 0 on 51 of 78 decodes against a draft that was being accepted 86% of the time, walking
  down to 0, getting pushed back up by exploration, and walking down again. Sampling the width-0
  step from the plain sampling path anchors the fit and the same run then spends 27 of 44 decodes
  at width 2.
- **Hysteresis.** Move `W` by at most one step per decision, and require the predicted gain to beat
  the incumbent by a margin. Width changes the ubatch shape, which changes kernel selection;
  oscillating across that boundary is its own cost.
- **Decay per token, not per call**, for the same reason `llama_expert_heatmap::decay_all` does
  (`src/llama-expert-heatmap.h:47-49`, `ce972df20`).

### 4.3 The per-slot acceptance estimator

`n_acc_tokens_per_pos` is cumulative and shared. Add to `server_slot` a small
`std::array<float, W_ceiling>` of decayed acceptance rates, updated where the slot already knows how
many draft tokens were accepted (the same place `slot.n_draft_accepted` is maintained). Do not touch
`common_speculative_impl`.

### 4.4 The new library surface

One accessor, in the style of the existing `llama_expert_tier_n_bypassed`
(`src/llama-ext.h:90-91`):

```c
// distinct cold experts the last decode touched, summed over layers, and the batch it ran
LLAMA_API bool llama_expert_hot_last(const struct llama_context * ctx, int32_t * cold_distinct, int32_t * n_tokens);
```

Backed by two `int32_t` on `llama_context`, filled in the block that already calls
`expert_heat_update` (`src/llama-context.cpp:1714-1720`), by folding the counting loop out of
`log_hit_rate` into a small helper both call. Returns false when there is no hot store; the
controller then falls back to the configured fixed width, which is exactly today behaviour.

---

## 5. Flags and the tier-limit interaction

### 5.1 Flags

One new flag, off by default:

- `--spec-adaptive-width` (bool). When set, `--spec-draft-n-max` becomes the **ceiling** rather than
  the width, and the controller picks `[0, ceiling]`.

Resist adding a floor, a threshold, or a target hit rate. Section 4.2 has no free parameter a user
is in a position to set better than the measurement.

### 5.2 The tier limit is a hard ceiling. Do not raise it to make room.

The tier is bypassed above `llama_expert_tier_max_tokens()` (`src/llama-expert-tier.cpp:90-95`), and
the heatmap freezes on the same test (`src/llama-context.cpp:1708`). Default 4.

`b153688b4` used to auto-follow the draft width, on the reasoning that the limit was only a
performance trade. **`448744c89` removed that**, because dropping the landing pad made a token id
list non-distinct again: every cold draw names slot 0 (`src/llama-expert-hotstore.h:22-31`), MMQ/MMF
id compaction miscounts duplicates, and the host-side fallback asserts outright
(`PLAN-ngram-mod-expert-cache.md` section 1.3). The code comment that replaced the auto-follow says
it plainly - "raising the limit would be unsafe, not just slow" (`common/common.cpp:1269-1273`).

So:
- The controller clamps to `llama_expert_tier_max_tokens() - 1`, hard. That is **3** at the default.
- `--spec-adaptive-width` must **not** raise `--expert-tier-max-tokens`, and should refuse to start
  if the user has raised it past 4 while `-ehs` is on. This is a correctness gate.
- Do not let the controller step over the limit "just to try it". A batch one token over does not
  degrade gracefully in either sense: it is unsafe, and even where it is not, the whole MoE falls
  back to stock `mul_mat_id`, which reads *every* selected expert from host RAM, hot ones included,
  while the store sits resident and unread.

**Getting to width 4 and beyond** means the MMQ/MMF id compaction TODO at
`src/llama-expert-tier.h:18-26`, or the cheaper `remap_mask` route in the paragraph below it
(`:28-33`) which frees the pad slices but does **not** make the ids distinct, so it does not lift
the limit. That is `PLAN-ngram-mod-expert-cache.md` step 3 territory, not this plan.

**Stale comment to fix while in here:** the header block at `src/llama-expert-tier.h:12-15` still
claims "the landing pad removed the duplicates, so that reason is gone", which `448744c89` made
false. The last paragraph of the same block (`:28-33`) already states the correct conclusion. The
two contradict each other in the same comment.

### 5.3 Logging

One `SLT_DBG` per width change: old width, new width, `cold_rate`, `T0`, `alpha`, `G(W)`. Without
it a regression in this controller is indistinguishable from a regression in the model.

---

## 6. Step 0 - the experiments that decide whether to build any of this

New script `phase0/spec-width-sweep.sh`, same shape as `phase0/ngram-mod-ab.sh`: drive
`llama-server` over `/completion`, one arm per width, report from a companion python script.

Fixed for every arm: `-ehs -1`, `--spec-type draft-mtp`, same prompt, same seed, temp 0. Leave
`--expert-tier-max-tokens` at its default (see 5.2 - raising it is a correctness question, not a
tuning knob).

**E1 - the fixed-width curve.** `--spec-draft-n-max` in `{0, 1, 2, 3}`. Record tokens/s, the
per-position acceptance line from `common_speculative_print_stats`, and with
`LLAMA_EXPERT_HITRATE=1` the hit rate and union per batch. This gives the real `T(m)` and the real
`a_i`, and it directly tests section 1.2 - plot cold distinct against `m` and see whether it is a
line.

*Kill criterion:* if width 1 wins at every phase and every prompt by more than the noise, ship
nothing. The hand-tuned setting is already the answer.

**E2 - does the optimum move within a run?** This is the only thing that justifies a controller.
Use a prompt with two clearly different phases - a prose preamble followed by a long verbatim code
block; `phase0/prompts/copy-heavy.txt` is close to the right shape already. Report tokens/s and hit
rate **per 100-token window**, not per run. If the best width is 1 in the prose window and 3 in the
code window, adaptivity has something to win and the gap is its size. If the windows agree, stop
here.

*This is the experiment that matters. E1 without E2 only re-tunes a constant.*

### 6.1 E2 RESULT - yes, and by a lot

Run 2026-08-23, `sw-e2-*`, 8 arms (widths 0-3 x 2 repeats), 800 tokens, `-ehs -1`, temp 0.
**Measured**, speedup against width 0 in the same window, which cancels the between-run drift the
raw numbers carry:

| window | w1 | w2 | w3 |
|---|---|---|---|
| 0-100 | 1.27 | 1.44 | **1.51** |
| 100-200 | 1.14 | 1.29 | **1.49** |
| 200-300 | 1.00 | 0.96 | 1.11 |
| 300-400 | 1.06 | 0.95 | 1.03 |
| 400-500 | 1.06 | 1.09 | 0.97 |
| 500-600 | 1.04 | 0.98 | 1.04 |
| 600-700 | 1.09 | **1.31** | 1.07 |

Width 3 is worth **1.5x** for the first 200 tokens and **nothing** after - 0.97 to 1.11, and at
400-500 it is slower than not speculating at all. No fixed width has both halves. That is the
case for a controller, and it is much stronger than section 1.2 predicted.

**Where the phase change comes from, and it is not what the prompt asked for.** The model opens a
`<think>` block by restating the request nearly word for word, which MTP predicts almost
perfectly; around token 200 it stops echoing and starts reasoning, and acceptance collapses. The
prompt's Part 2 verbatim copy is never reached inside 800 tokens. **This is the better experiment
anyway**: every thinking model opens every request that way, so the fast phase is not a contrived
workload, it is the first few hundred tokens of ordinary use.

**What this run does NOT establish.** The best *fixed* width overall. Run-to-run spread between
two identical width-0 arms was **19.9%**, wider than the 16.9% best-to-worst gap across widths, and
both repeats swept the widths in the same order so a position artifact aliased onto width (both w0
arms ran first and came out slowest, both w3 arms ran last and came out fastest). The second repeat
now runs the widths backwards. That question belongs to E1 and needs more repeats either way.


**E3 - is the hit rate actually the thing that varies?** Same run as E2; correlate the per-window
optimal width against the per-window hit rate. If the correlation is weak, section 4.2 regresses
against the wrong variable, and the honest outcome is a plain measured-throughput bandit over
widths with no expert term at all - simpler, and section 4.1 is unchanged.

**E4 - cost of the counting.** Before and after step 2, tokens/s at fixed width 1. Section 2.1 says
the scan is free; confirm it, because it runs on every decode.

---

## 7. Validation (step 5)

- **Correctness is unaffected and must be shown to be.** Speculation is exact: the target verifies
  every drafted token, so a changing width cannot change the output. Show it - temp 0, fixed seed,
  adaptive against `--spec-draft-n-max 1`, byte-identical completions. If they differ, something in
  step 1 or 3 is wrong, and the device-placement shift documented at the top of
  `PLAN-ngram-mod-expert-cache.md` is *not* an excuse here: both arms run the same graph shapes.
- Throughput against fixed width 1 on at least three prompt shapes: prose, code, copy-heavy.
- Idle cost: adaptive on a prompt where speculation never helps must not be measurably slower than
  `--spec-draft-n-max 0`. Compare against the 2% idle figure recorded in `a0ca094e1`.
- A long run, to confirm the fit does not drift into a corner and stay there.

---

## 8. Risks, stated plainly

1. **The controller may be chasing noise.** Per-decode times on this box vary with whatever else
   touches the host bus. The decay window has to be long enough to average that out and short enough
   to react to a phase change; those may not both be satisfiable. E2 window analysis measures
   exactly that tension and should be read before step 3 is written.
2. **Two-parameter online least squares is a new pattern in this tree.** It is ~30 lines and it is
   defensible, but if E3 says the expert term is weak, prefer the bandit: one array of decayed
   throughput per width, pick the best, explore occasionally. Fewer moving parts, same call site.
3. **The ceiling is not free, and it makes `adapt` an unfair comparison.** Graph reservation
   happens at the ceiling width, so varying the width at run time never needs a new allocation -
   but the fitter sizes the compute buffer for that ceiling up front and the hot store gets what
   is left. **Measured**: `--spec-draft-n-max 3` leaves S=26 slots where `--spec-draft-n-max 1`
   leaves S=28. An adaptive arm that mostly picks width 1 is still running against a smaller store
   than the fixed-width-1 arm it is being compared with, so step 5 is measuring two changes at
   once. Report S for both arms.
4. **Interaction with `ngram-mod`.** ngram-mod outranks draft-mtp in the impl order
   (`common/speculative.cpp:2449-2459`) and is all-or-nothing at `n_min` (`:1958-1970`). A `dp.n_max`
   of 2 against an `n_min` of 48 means ngram-mod fires, gets truncated to 2, and MTP never runs -
   strictly worse than either alone. **Do not combine `--spec-adaptive-width` with ngram-mod** until
   `PLAN-ngram-mod-expert-cache.md` step 3 lands; refuse the combination at startup with a clear
   message.
