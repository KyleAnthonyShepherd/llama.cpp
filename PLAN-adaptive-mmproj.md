# PLAN - adaptive mmproj: where the vision tower should live

Goal: stop paying ~1.1 GB of VRAM for `--mmproj` on every text-only token, and give that VRAM
to the expert hot store instead, without making image workflows unusable.

Machine: RTX 3060 Laptop, 6144 MiB VRAM / 16 GB RAM, CUDA, Windows.
Model `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_S`, projector `mmproj-BF16.gguf` (861 MiB).

**The measurements below changed the answer.** The swap-in/swap-out design is not the best
option; a static placement change gets 95% of the VRAM back for a ~120 ms constant cost, with no
state machine at all. Sections 2 and 3 are the evidence, section 5 is the residual question.

Every number marked **measured** was produced by `llama-mtmd-cli` on this machine today.
Harness: `-t 16 -ngl 0 -c 4096 -n 1 -lv 4`, synthetic 1000x1000 PNG (grid, 60 boxes, 40 lines,
20 lines of dimension text) giving 961 image tokens. `-ngl 0` keeps the LLM off the GPU so the
VRAM reading isolates the projector.

---

## 1. What the 1.1 GB is

| part | MiB | **measured** from |
|---|---|---|
| vision tower weights | 861 | `mmproj-BF16.gguf` file size; BF16 is not requantized on load |
| clip compute buffer | 248 | `reserve_compute_meta: CUDA0 compute buffer size = 248.10 MiB` |
| **total** | **1109** | nvidia-smi peak delta 2824 - 2000 = 824 MiB when only the weights move |

The compute buffer is sized from a warmup image the arch pins at **1472 x 1472**
(`get_dummy_batch`, `tools/mtmd/clip.cpp:3362-3378`), so 248 MiB is a **ceiling**, not a
per-image figure: a 2000x2000 input reserves exactly the same buffer (**measured**, identical
`reserve_compute_meta` line in both runs). Input size drives encode *time*, not buffer size.

---

## 2. Timing: three places the tower can live

1000x1000 image, 961 tokens. Each cell is a separate process launch.

| tower weights | compute | encode | peak VRAM | vs GPU-resident |
|---|---|---|---|---|
| VRAM (today) | CUDA | **1010 / 1019 / 1027 ms** | 2824 MiB | - |
| **host RAM** | **CUDA** | **1125 / 1147 / 1153 ms** | **2000 MiB** | **+120 ms, -824 MiB** |
| host RAM | CPU (`--no-mmproj-offload`) | **69,881 ms** (`-t 16`), 100,464 ms (default `-t`) | 2000 MiB | +69 s |

2000x2000 image, 3969 tokens, clean re-run with the env var properly unset:

| tower weights | encode | peak VRAM |
|---|---|---|
| VRAM | **10,533 ms** | 3203 MiB |
| host RAM | **10,766 ms** | 2415 MiB |

**`--no-mmproj-offload` is dead.** 70 seconds for one 1000x1000 image, with every core busy.
A multi-page PDF is minutes per page. Option A from the previous revision of this document is
withdrawn.

**Host weights with GPU compute costs a near-constant ~120-230 ms**, so it amortizes with image
size: +11.6% at 961 tokens, **+2.2% at 3969 tokens**. That is the weight-streaming cost, paid
once per encode regardless of how much compute the encode then does.

### 2.1 It is numerically identical, not an approximation

Same prompt at `--temp 0`, 90 tokens, both placements: **token-for-token identical output**
(**measured**). Same CUDA kernels, same graph; only where the weights are read from changed. The
description is also correct - it names the grid, the black rectangles, the red lines and the
dimension text on the left, which is what the fixture contains.

### 2.2 Why it works, and why it was already possible

`ggml_backend_sched` offloads an op to a higher-priority backend when the op's weights sit in a
**host** buffer marked `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` and the batch is wide enough
(`ggml/src/ggml-backend.cpp:955-965`). clip already creates its sched with `op_offload = true`
and registers both CUDA and CPU (`tools/mtmd/clip.cpp:213-217`), and already marks its weight
buffer `USAGE_WEIGHTS` (`:3305`). The only thing standing in the way was that the weight buft is
hardwired to the compute backend's buft at `tools/mtmd/clip.cpp:3303`.

This is the same trade the expert hot store makes, but the vision tower is a far better
candidate for it than MoE experts are:

- The tower is **dense** - every weight is read by every image, so there is no hot subset to
  rank and nothing a cache could skip.
- Its arithmetic intensity is ~961x that of a single-token MoE expert read. 861 MiB crosses
  PCIe once and then does 961 tokens of work. That is exactly the regime `op_offload`'s
  min-batch threshold exists to select for (`--op-offload-min-batch`, 128 under `-ehs`).

### 2.3 The experiment patch is in the working tree, uncommitted

`tools/mtmd/clip.cpp:3305-3310`, ~6 lines, gated behind `MTMD_WEIGHTS_HOST=1` so a normal run is
untouched. It is a measurement hack, not a proposed shape - a real version wants a CLI flag and
should consider a pinned host buft for faster DMA (see section 6.2). **Revert or replace it
before doing anything else with the tree.**

---

## 3. What the slots are worth

- **72.7 MiB per hot slot** (**measured**, `NOTES.md:1762-1764`).
- 824 MiB recovered / 72.7 = **~11 hot slots**, permanently, for +2-12% encode time.
- The full store is worth **+18 to +35%** on an ordinary prompt (22.67 -> 30.65 t/s,
  **measured**, `NOTES.md:1943-1952`).

Still **to measure**: what those 11 specific slots are worth. Slot value is non-linear because
the heatmap fills the hottest experts first, so the marginal 11 are worth well under 11/37 of the
+18-35%. One A/B settles it: `-ehs 37` vs `-ehs 26`, no mmproj either arm, on the
`phase0/devbox-server-ab.sh` prose prompt.

---

## 4. What should be in VRAM while an image is being read

The key observation is that **the value ordering changes during the encode window, because the
LLM is idle for that second.** `NOTES.md:1912-1922` ranks residents by host bytes saved per
generated token; during an encode there are no generated tokens, so that ranking does not apply.
Re-derived for the encode window:

| resident | MiB | value during the ~1 s encode | verdict |
|---|---|---|---|
| clip compute buffer | 288 | the encode cannot run without it | **must be resident** |
| KV cache | 20 KiB/cell | none, but it is the conversation | not a budget target, it is state |
| dense LLM weights | 1389 | none (LLM idle), but needed for the prefill 1 s later, and fixed at load | leave alone |
| **vision tower weights** | **861** | **120-230 ms of encode time - measured** | **lowest value per MiB here; host RAM** |
| expert hot store | 72.7/slot | **exactly zero** - no tokens are being generated | first to evict if more is needed |

So the eviction ladder for the encode window, best to worst:

1. **Move the tower weights to host RAM, permanently.** Buys 824 MiB for ~120 ms. No state
   machine, no re-plant, no idle timer, no concurrency work.
2. **Drop hot store slots temporarily.** Worth zero during the window. `resize()` /
   `llama_expert_hotstore_refit()` already do exactly this
   (`src/llama-expert-hotstore.cpp:200`, `src/llama-context.cpp:1124`). Costs a ~600 ms re-plant
   each way (**measured**, `NOTES.md:1816`).
3. **Shrink the clip compute buffer** with `--image-max-tokens` (`common/arg.cpp:2695`) or
   `--mtmd-batch-max-tokens` (default 1024, `common/common.h:603`). Costs resolution or extra
   passes.
4. **CPU encode.** 70-100 s. Correct, and a legitimate last resort rather than failing the
   request, but never a default.

### 4.1 Near-full context: the numbers

`NOTES.md:1857-1860` gives the fixed costs on this card: dense weights **1389 MiB**, overhead
**~1100 MiB**, leaving **3655 MiB** for KV + hot store + projector. KV is 20 KiB/cell (11 of 41
blocks carry attention - this is a hybrid arch, `NOTES.md:1905-1909`). Hot store caps around 37
slots.

Hot slots that can stay resident *during an encode*:

| context | KV MiB | tower in VRAM (1109) | **tower in host RAM (288)** |
|---:|---:|---:|---:|
| 4096 | 80 | 34 | **37** |
| 8192 | 160 | 33 | **37** |
| 16384 | 320 | 31 | **37** |
| 32768 | 640 | 26 | **37** |
| 65536 | 1280 | 17 | **29** |
| 131072 | 2560 | **DOES NOT FIT** | **11** |
| 172032 | 3360 | **DOES NOT FIT** | **0** |
| 186368 | 3640 | DOES NOT FIT | DOES NOT FIT |

Read the two right-hand columns as the answer to the edge case:

- **With the tower in VRAM**, images become impossible above **~127k context** even after
  evicting the entire hot store - and they cost 4-20 slots long before that.
- **With the tower in host RAM**, nothing is evicted at all up to **~32k context**, and images
  keep working to **~172k** - which is essentially the point where the context itself stops
  fitting (`NOTES.md:1878`, "~180000 is the honest ceiling" with zero experts and no projector).

That is the substantive result for your edge case: **moving the weights to host RAM does not
just shrink the problem, it very nearly deletes it.** The projector stops being a thing that
competes for the budget and becomes a rounding error against the KV cache.

### 4.2 The Windows trap that must shape the code

`NOTES.md:1868-1871`: on WDDM a CUDA allocation that does not fit VRAM **is silently backed by
system memory instead of failing**. The tree already learned this the hard way with the KV cache
- `resize()` always "succeeded" above the physical limit and the cache quietly ended up across
PCIe, showing up only as 18.4 tok/s against 26.

So `mtmd_init_from_file()` at a full card will **not** error. It will succeed and the tower will
encode at spill speed. Any policy here must **compute the fit up front from known constants**
and pick its fallback deliberately; it must never infer "it fit" from "the allocation returned".
Host weights cut the amount exposed to this from 1109 MiB to 288 MiB.

---

## 5. Is the swap still worth building?

After host weights, the swappable amount is **288 MiB = 4 hot slots**, and section 4.1 says it
costs nothing at all below ~32k context. Against that:

- two ~600 ms hot-store re-plants per multimodal episode,
- an idle timer with hysteresis (a PDF arrives as a burst of separate requests),
- `mctx` is read from eight HTTP-thread call sites (`tools/server/server-context.cpp:4886-4891`,
  `:5471`, `:5535`, `:5594`, `:5644`, `:5806`, `:6143`, `:6238`) and the inference thread
  (`:852-906`), and every slot caches its own copy (`:1508`) - a load or free has to be gated
  like the sleeping state (`tools/server/server-queue.cpp:102-117`) or it is a use-after-free,
- the `has_mtmd` branch at `:4886` keys on `mctx != nullptr`, so a text-only request would change
  `server_tokens` shape depending on whether an image was posted two minutes ago, and prompt-cache
  entries stop matching across the boundary.

**Recommendation: do host weights first and re-measure.** If E2 (section 3) says 4 slots are
worth having, the lazy path is still available and is *cheaper* to build on top of host weights,
because the weights then stay in RAM across cycles - a reload costs no disk read, only the
288 MiB buffer. Decide it with the number, not now.

One caveat that survives regardless: the fitter adds the projector's worst case to
`params_base.fit_params_target` (`tools/server/server-context.cpp:1240-1263`) before the target
model loads, and `resize()` is capped at `hot_s_max` forever
(`src/llama-expert-hotstore.h:20-22`). **Whatever gets built, if that accounting still reserves
1109 MiB the store never sees the recovered slots** - it will run, log correctly, and buy
nothing. Host weights need it to reserve 288, not 1109.

---

## 6. Open questions

1. **E2**: `-ehs 37` vs `-ehs 26`, no mmproj. Decides whether section 5 has anything to chase.
2. **Pinned host buffer.** The experiment used the plain CPU buft. A CUDA host (pinned) buft
   would DMA faster, but `ggml-backend.cpp:959` requires `src_backend_id == n_backends - 1`, and
   a CUDA_Host buft may map to backend 0 and disqualify the offload. Worth 30 minutes to check;
   could shave part of the 120 ms.
3. **RAM cost.** The tower now occupies 861 MiB of host RAM permanently on a 16 GB box already
   mmapping a 20.3 GB checkpoint. Measure the effect on page-cache pressure and tok/s before
   calling this free.
4. **`--image-max-tokens` vs the 288 MiB.** The buffer is pinned to a 1472x1472 warmup; check
   whether capping image tokens actually lowers `reserve_compute_meta`, which would give a cheap
   third lever for the >130k case.
5. Does the +120 ms hold on the headless Linux devbox, where there is no WDDM and no desktop
   using 620 MiB? PCIe link width there may differ.
