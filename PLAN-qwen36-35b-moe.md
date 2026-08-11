# PLAN - Qwen3.6-35B-A3B expert tiering on the 6 GB VRAM / 16 GB RAM server

Goal: make the 4-bit 35B-A3B run on the box that currently runs the 27B dense at 2-3 t/s, by
placing individual routed experts across VRAM / RAM / SSD according to measured usage.

> ## REVISION - `UD-Q3_K_M` selected, the SSD tier is deleted
>
> The user's KL-divergence research puts `UD-Q3_K_M` (16.6 GB = **15.46 GiB**) inside the
> acceptable quality band. That fits in ~5.4 GiB VRAM + ~14 GiB RAM with **~3.9 GiB to spare**.
>
> **Everything about SSD offloading in this document is now moot.** Sections 1.2, 2's SSD row,
> and Phase C were the highest-value work when the target was a 20.9 GiB `UD-Q4_K_XL` with a
> 1.5 GiB overhang. With no overhang there is nothing to evict, and the question collapses to
> the one the user now asks: **what is the best use of 5748 MiB of VRAM.**
>
> Section 7 answers that and supersedes sections 2 and 3. The material below it is kept because
> the structural findings (1.1, 1.3, 1.4) are unchanged and still govern.
>
> Headline: **at Q3_K_M, 68% of the bytes read per token are dense, not expert.** Dense weights
> have a value-per-byte of 1.0; a routed expert's is 8/256 = 0.031. **A 32:1 ratio.** The first
> ~950 MiB of VRAM is worth more than the remaining ~3.8 GiB combined, and the default fitter
> already spends it correctly. Per-expert placement is now competing for the *low-value* tier.

Discovery pass only - no code changed. Companion to `PLAN-qwen36-27b.md` and `NOTES.md`;
this document reuses their measured numbers where they apply and marks where the MoE case
diverges. Every structural claim is cited to a file:line in this tree. Every number is
marked **measured**, **published**, or **derived**.

---

## 0. Geometry and the byte budget

### 0.1 Model (published, from the HF config)

| | |
|---|---|
| `num_hidden_layers` | 40 |
| `hidden_size` | 2048 |
| layer classes | 30 gated-delta-net + 10 full attention (il 3, 7, ... 39) |
| `num_experts` / `num_experts_per_tok` | **256 / 8** |
| `moe_intermediate_size` | **512** |
| `shared_expert_intermediate_size` | 512 (1 shared expert, always active) |
| attn heads / kv heads / head_dim | 16 / 2 / 256 |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144 |

Two things carry over unchanged from the 27B work: the hybrid attention split is the same
shape (`full_attention_interval = 4`, `src/models/qwen35moe.cpp:24-29`), and the arch takes
the same `llama_memory_hybrid` branch. **What differs is entirely the FFN.**

### 0.2 Derived byte model

Routed experts per layer are three matrices of `n_embd x n_ff_exp = 2048 x 512`:

```
one expert (gate + up + down) = 3 * 2048 * 512   = 3,145,728 params
                              @ Q4_K (4.5 bpw)   = 1.6875 MiB
one layer  (256 experts)                          = 432 MiB
all 40 layers                                     = 16.9 GiB
total expert count                                = 40 * 256 = 10,240
```

Routed experts are **32.2 B of the ~34 B parameters**, i.e. 80-90% of any quant's file size.
Everything else - attention, GDN, shared experts, routers, embeddings, output - is under
2 GiB at 4-bit.

Per generated token the model reads `8 experts x 40 layers = 320 expert slabs`:

```
per-token routed traffic = 320 * 1.6875 MiB = 540 MiB   (pure Q4_K; ~600-670 MiB at UD-XL mix)
```

Against the 27B dense, which reads **16.7 GiB per token**, that is a **~30x reduction in
bytes touched**. This is the entire prize, and it is delivered by the architecture, not by
any placement cleverness.

### 0.3 KV is nearly free on this model

`n_embd_k_gqa = n_embd_v_gqa = 2 kv heads * 256 head_dim = 512`, so f16 KV is
`(512 + 512) * 2 B = 2 KiB per token per attention layer`, over **10** attention layers:

| n_ctx | KV total |
|---|---|
| 8K | 160 MiB |
| 32K | 640 MiB |
| 128K | 2.5 GiB |

Compare the 27B's 4.0 KiB/tok/layer over 16 layers = 64 KiB/token. **The 35B-A3B costs 20
KiB/token, 3.2x less.** Plus a context-independent recurrent state on the 30 GDN layers.

**Consequence: `--ctx-max` growth is not what makes this model fit.** At 8K, KV is 160 MiB -
about 95 experts' worth. The fit/no-fit decision is made almost entirely by weights. Auto
context growth matters here for a different reason: see section 4.

### 0.4 The actual fit gap (published file sizes, unsloth GGUF repo)

Machine budget, using the **measured** hardware from `PLAN-qwen36-27b.md`: GTX 1660 Ti with
5748 MiB total VRAM, 16 GiB RAM. Call it ~5.4 GiB usable VRAM + ~14 GiB of usable page cache
after the OS = **~19.4 GiB**.

| quant | file size | vs 19.4 GiB |
|---|---|---|
| UD-Q3_K_XL | 16.8 GB = 15.6 GiB | fits, 3.8 GiB spare |
| **UD-IQ4_XS** | **17.7 GB = 16.5 GiB** | **fits, 2.9 GiB spare** |
| UD-IQ4_NL | 18 GB = 16.8 GiB | fits |
| UD-IQ4_NL_XL | 19.5 GB = 18.2 GiB | fits, 1.2 GiB spare |
| UD-Q4_K_S | 20.9 GB = 19.5 GiB | ~0.1 GiB over |
| UD-Q4_K_M | 22.1 GB = 20.6 GiB | **1.2 GiB over** |
| UD-Q4_K_XL | 22.4 GB = 20.9 GiB | **1.5 GiB over** |

This confirms the premise exactly: the K-quant 4-bit mixes are over by 1.2-1.5 GiB, which is
**700-900 experts, or 7-9% of the total**.

It also raises a cheap question that has to be asked before any of the work below:
**`UD-IQ4_XS` is a genuine 4-bit quant and it fits today with room to spare.** The catch is
real, not rhetorical - i-quants have materially worse CPU matmul throughput than K-quants and
are not repack-eligible (`ggml/src/ggml-cpu/repack.cpp` traits cover Q4_K, not IQ4_XS), and on
this box 70-80% of the FFN work is CPU-side. So it is a quality-vs-CPU-throughput trade, and
Phase B measures it rather than assuming. But if `UD-IQ4_XS` lands within 15% of `UD-Q4_K_XL`
on tokens/s, everything in Phases C-E is optional.

---

## 1. Three findings that reshape the request

### 1.1 Per-expert placement is not expressible today. All 256 experts are one tensor.

`src/models/qwen35moe.cpp:99-101`:

```cpp
layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
```

A layer's 256 experts live in **one 3D tensor per matrix**, expert `i` at `i02 = i`. A ggml
tensor lives in exactly one backend buffer (`src/llama-model.cpp:1499,1571` - one
`ggml_context` and one buffer per distinct `buft`). So:

- `-ot` / `--override-tensor` regexes match tensor **names**
  (`src/llama-model-loader.cpp:1162-1197`). The finest thing they can say is
  "`blk.7.ffn_up_exps.weight` goes to CPU". There is no name for expert 113 of layer 7.
- The fitter's MoE lever is the same granularity:
  `blk\.<il>\.ffn_(up|down|gate_up|gate)_(ch|)exps` (`common/fit.cpp:437`), and `-ncmoe`
  (`common/arg.cpp:2695`) is a whole-layer cutoff.

**The placement quantum available today is 432 MiB (one layer's experts), not 1.7 MiB (one
expert).** Everything in the request that involves putting *selected* experts in VRAM
requires new graph structure. Section 3, Phase E.

One saving grace for later: expert `i` is a **contiguous** `nb02`-sized slab, both in the
tensor and in the GGUF file. That is what makes Phases C and D cheap.

### 1.2 The SSD tier already exists and already works. You just cannot steer it.

Two facts compose into this:

1. **CPU-side weights are clean file-backed mmap.** `MAP_SHARED | PROT_READ`
   (`src/llama-mmap.cpp:447,456`), **measured** on this box at `CPU_Mapped 12171.07 MiB`
   with zero anonymous copies (`PLAN-qwen36-27b.md` section 1.2 verdict).
2. **`mul_mat_id` touches only the selected experts.** `ggml/src/ggml-cpu/ggml-cpu.c:1650-1654`:

```c
        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
```

Experts with no routed rows are skipped entirely, and the ones that are used are addressed by
a direct offset into the mapping. **So an unused expert's pages are never faulted in.** The
OS page cache is therefore already acting as an LRU expert cache with 4 KiB granularity,
backed by the SSD, for free, today.

This is the single most important finding in this document. The "evict cold experts to SSD"
half of the request **is not a feature to build - it is the current behavior.** What is
missing is *control*: which experts stay resident is decided by kernel LRU over 4 KiB pages,
not by measured expert value. Steering it needs `madvise` / `mlock` over byte ranges, which
is ordinary POSIX work on an existing mapping - no ggml changes at all. Section 3, Phase C.

Note the current advice is actively wrong for this workload: when prefetch is off,
`src/llama-mmap.cpp:469` sets `POSIX_MADV_RANDOM` over the whole file, which **disables
readahead**. An expert slab is 1.7 MiB of contiguous bytes that is always read in full. You
want readahead *within* a slab and none across slabs.

### 1.3 Whole-layer `-ot` already captures most of the VRAM tier's value, unless routing is skewed

VRAM budget, **derived** from the measured 5748 MiB card:

```
  5748  total
-  1300  non-expert weights that want GPU (output, attn/GDN, shared experts, routers)
-   500  compute buffer
-   160  KV at 8K
-    90  recurrent state, 30 GDN layers
-   300  margin (-fitt, tunable; see 27B plan section 2.4)
= ~3400 MiB for routed experts  ->  ~1,950 experts  ->  19% of 10,240
```

Now compare the two ways of choosing those experts:

| strategy | what it holds | activation hit rate |
|---|---|---|
| whole-layer `-ot` (**available today, zero code**) | 3400 / 432 = **7.9 layers** | 7.9/40 = **19.7%** |
| per-expert, routing uniform | 1,950 hottest experts | **19.0%** |
| per-expert, routing skewed (top 19% take 40%) | 1,950 hottest experts | **40%** |

Under uniform routing, **whole-layer placement is not worse - it is marginally better**,
because it wastes no bytes. Per-expert placement only wins by the amount routing is skewed.

**Therefore the entire case for per-expert VRAM placement rests on one number that nobody in
this repo has measured: the expert-usage distribution of Qwen3.6-35B-A3B during generation.**

Be sceptical of the number you will find quoted. Upstream issue
[#20757](https://github.com/ggml-org/llama.cpp/issues/20757) asserts "~15-20% of experts
handle ~80% of tokens", but that is an unsourced claim in a feature request, measured on a
different model (GPT-OSS-120B) via a Python prototype. MoE models are trained with a
load-balancing auxiliary loss whose explicit purpose is to flatten this distribution, and
published measurements on Mixtral show distributions much closer to uniform than 80/20. A
256-expert top-8 router with balancing is a hard case for skew.

**Decision rule for this plan:** let `H(f)` be the fraction of generation-phase activations
captured by the hottest `f` of experts. Phase A measures `H(0.19)`.

- `H(0.19) < 0.25` - skew is negligible. Do not build Phase E. Use `-ot` on whole layers.
- `0.25 <= H(0.19) < 0.40` - marginal. Phase E buys maybe 5-8% end-to-end; probably still not
  worth a new graph subsystem.
- `H(0.19) >= 0.40` - the request is justified. Build Phase E.

### 1.4 Correction to the upstream framing: nothing is streamed over PCIe at decode time

Issue #20757's premise is "every decode step copies the same ~4 hot experts per layer from RAM
to GPU". **That is not what llama.cpp does at batch size 1.** Op-offload is gated on batch
size (`ggml/src/ggml-cuda/ggml-cuda.cu:5195-5207`, and the threshold at `:5377`):

```c
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
...
    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;   // default 32
```

`ne[2]` for `mul_mat_id` is `n_tokens`. At `n_tokens = 1` the op is **never** offloaded;
CPU-resident experts are matmul'd on the CPU, and no bytes cross PCIe. (This is also why the
27B measured `graph splits = 851 (bs=512)` but only `82 (bs=1)`.)

The consequence matters for design: the win from a VRAM expert tier is **not** "avoid
redundant PCIe copies". It is "move the matmul from CPU at ~26 GB/s to GPU at ~288 GB/s". Any
design that copies experts host->device per token is fighting a 12 GB/s PCIe 3.0 link to
replace a 26 GB/s memory read - a loss unless the hit rate is very high and the copies are
amortized across many tokens.

---

## 2. What each tier is actually worth

Derived, using measured effective host bandwidth from the 27B runs (13.1 GiB CPU-side /
0.495 s per token = **~26 GB/s**) and a nominal 288 GB/s for the 1660 Ti.

Take `UD-Q4_K_XL` (20.9 GiB, the case the user actually wants) and the section 1.3 budget:

| tier | share of the 10,240 experts | per-token bytes | time |
|---|---|---|---|
| VRAM | 19% | 103 MiB | 0.4 ms |
| RAM (page cache) | ~74% | 400 MiB | 15 ms |
| SSD | **~7%** (the 1.5 GiB overhang) | 38 MiB | see below |

The SSD row is the whole game. Per token it is `0.07 * 320 = 22 expert slabs`, each a 1.7 MiB
read at effectively queue-depth 1, serialized inside the per-layer expert loop:

| storage | ~throughput at 1.7 MiB QD1 | added latency/token |
|---|---|---|
| NVMe Gen3 | 1.5 GB/s | **25 ms** |
| NVMe Gen4 | 2.5 GB/s | 15 ms |
| SATA SSD | 0.5 GB/s | **75 ms** |
| HDD | - | unusable |

So on NVMe Gen3 the SSD tier roughly **doubles** token time (15 ms -> 40 ms), and on SATA it
quintuples it. **This is where measured expert popularity pays.** If the coldest 7% of experts
account for only ~2% of activations rather than 7%, SSD reads drop from 22 to ~6 per token and
the added latency drops from 25 ms to 7 ms - an end-to-end gain of **1.4-1.8x**.

**Compare that to the VRAM tier's contribution: 0.4 ms vs 15 ms, i.e. moving 19% of experts to
VRAM saves at most ~4 ms/token.**

> **The priority ordering is the inverse of the request's emphasis.** The RAM<->SSD boundary is
> worth 2-4x more than the VRAM<->RAM boundary, and it is the one reachable without touching
> ggml. Do it first.

Sanity ceiling: at ~40 ms/token the model tops out around **25 t/s** theoretical. Real MoE
decode is latency-bound, not bandwidth-bound (8 skinny 2048x512 matmuls per layer, 320 per
token), so expect **6-12 t/s** in practice. Against the 27B's measured 2.02 t/s (3.43 with
MTP), that is a 2-4x win *before any of this plan's work*, purely from the architecture.

---

## 3. Phased plan

### Phase A - measure the expert distribution (the decision gate)

**This is the only phase that is unconditionally worth doing, and it unblocks or kills
everything else.** ~150 lines, no core changes.

The hook already exists. `build_moe_ffn` names the top-k result
(`src/llama-graph.cpp:2030-2033`):

```cpp
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);
```

`cb` becomes `ggml_format_name(cur, "%s-%d", name, il)` (`src/llama-context.cpp:2590-2596`),
and `selected_experts` is `src[2]` of the `mul_mat_id`, so it is a live graph node named
`ffn_moe_topk-<il>`, type I32, shape `[n_expert_used, n_tokens]`.

`cparams.cb_eval` (`include/llama.h:376`, plumbed at `common/common.cpp:1661-1662`) delivers
it. Copy the shape of `common_debug_cb_eval` (`common/debug.cpp:143-189`), which already
handles the non-host case via `ggml_backend_tensor_get`.

Design notes that matter:

- **Return `false` in the `ask` phase for every tensor except `ffn_moe_topk-*`.** When
  `callback_eval` is set the scheduler drops to a node-by-node loop
  (`ggml/src/ggml-backend.cpp:1730-1748`), but it batches runs of nodes the callback declines.
  `common_debug_cb_eval` returns `true` unconditionally (`common/debug.cpp:152`); do not copy
  that or the profiling run will be far slower than it needs to be.
- **Separate prompt-phase from generation-phase counts.** `n_tokens` distinguishes them. Only
  the generation-phase distribution is relevant to the bandwidth problem.
- **Layer 40 is the MTP block** and has its own router and its own `build_moe_ffn`
  (`src/models/qwen35moe.cpp:681-693`, and `NOTES.md` A.6). Count it separately; if MTP is
  enabled it is read once per drafted token.
- The profiling run does **not** require the model to fit. mmap plus section 1.2 means it will
  run at any residency, just slowly. Profile with `UD-Q4_K_XL` on day one.

Outputs, all four needed:

1. `counts[layer][expert]` for generation tokens -> the curve `H(f)`, per layer and global.
   **This is the number section 1.3's decision rule consumes.**
2. Token-to-token overlap: `|E_t & E_{t-1}| / 8` per layer. Distinguishes a static hot set
   (favours Phase C/D) from temporal locality (favours a dynamic LRU cache, Phase E).
   **Post-revision this is the most valuable single number in Phase A** - see section 7.4, it
   directly predicts what MTP is worth on a MoE model.
3. Domain sensitivity: run 4-6 distinct prompt corpora (code, prose, math, chat, the user's
   actual workload) and compare the hot sets. If the hot set is domain-dependent, a *static*
   placement baked at load time is the wrong design and Phase D needs a per-workload profile.
4. Whether skew is uniform across layers. Early layers are commonly flatter than late ones; if
   so, VRAM should be spent on the skewed layers only.

Extend `phase0/02-gguf-layout.py` at the same time to report per-expert slab size and file
offset for every `(layer, matrix, expert)`. Phase C needs those byte ranges, and the script
already parses GGUF tensor info without third-party packages.

### Phase B - the zero-code baseline (do this before believing any of section 2)

Nothing here is new code. All of it is measured on the 27B already or is a flag.

1. **Quant bake-off**: `UD-IQ4_XS` (fits) vs `UD-Q4_K_XL` (1.5 GiB over) vs `UD-Q3_K_XL`
   (fits, 3.8 GiB spare). Report t/s **and** `majflt` per token. Per `PLAN-qwen36-27b.md`
   section 1.3, major faults, not swap, is the objective function.
2. **`--spec-type draft-mtp`** if an MTP-head quant exists. **Measured +70% generation** on the
   27B. Note MTP quadrupled the 27B's recurrent state (149.62 -> 598.50 MiB); on the 35B there
   are 30 GDN layers instead of 48 and `n_embd` is 2048 not 5120, so the absolute cost should
   be smaller, but measure it.
3. **`GGML_CUDA_FORCE_MMQ=ON` with Pascal arch.** **Measured +135% prompt processing** on this
   card. Free.
4. **`-fitt`**: the 1024 MiB default margin stranded ~1.3 GiB on the 27B run. Here 1.3 GiB is
   **770 experts**.
5. **`-ot` whole-layer expert placement.** Per section 1.3 this is the honest competitor to the
   entire rest of the plan. Once Phase A says which layers are most skewed, pin those layers'
   `_exps` to CPU and everything else to GPU. Baseline to beat:
   `-ot 'blk\.(0|1|...|31)\.ffn_(up|down|gate_up|gate)_exps=CPU'`.
6. **Bounded prefetch** (`PLAN-qwen36-27b.md` Phase 1b, ~30 lines, already scoped). The
   hardcoded `ml.init_mappings(true, ...)` at `src/llama-model.cpp:1532` becomes `MAP_POPULATE`
   over a 20.9 GiB file on a 16 GiB box. That is strictly harmful here and the fix is already
   designed.

**Gate: if Phase B lands above ~8 t/s, stop.** That is 4x the current 27B and the remaining
phases are chasing a fraction of the residual.

### Phase C - steer the RAM/SSD boundary (highest value per line of code)

Per section 2 this boundary is worth 2-4x more than the VRAM one, and per section 1.2 the
mechanism already runs - it just uses kernel LRU instead of measured value.

**C.0 - prove it with zero llama.cpp changes first.**

Page cache for a `MAP_SHARED` file mapping is shared between processes. So a standalone helper
that opens the same GGUF, `mmap`s it, and `mlock`s the byte ranges of the hot experts **pins
those pages for llama.cpp's mapping too**. Ranges come from Phase A's histogram plus the
extended `02-gguf-layout.py` offsets. This tests the entire hypothesis - "does pinning the
measured hot set beat kernel LRU" - before a single line of C++ lands in the tree.

Sizing matters and the 27B run is the cautionary tale: `RLIMIT_MEMLOCK` capped out at ~2 GiB
there, and locking all 13 GiB would have been actively harmful. Here the target is different -
lock a *chosen* subset (say 10-11 GiB of hot experts) and deliberately leave the cold tail to
churn against the SSD. Raise `ulimit -l` accordingly and watch `majflt`, not `MemAvailable`
(which stays misleadingly high precisely when clean mmap pages are about to be dropped).

**C.1 - in-tree, if C.0 works.** Two small things:

- Replace the blanket `posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)`
  (`src/llama-mmap.cpp:469`) with per-expert-slab advice. `MADV_RANDOM` kills readahead across
  the whole file, but each expert slab is 1.7 MiB read in full - exactly the case where
  readahead helps.
- An optional expert-priority file (Phase A's output) consumed at load: `MADV_WILLNEED` +
  `mlock` the hot ranges, `MADV_COLD` the cold ones.

**C.2 - the one real latency trick.** The router for layer `L` cannot be known before layer
`L-1` finishes, so cross-layer prefetch is not possible without speculation.

> **CORRECTION (section 13): the speculation works better than this section assumes.** wackMall's
> pre-gate predictor runs **layer L+1's own router on layer L's hidden state**, with a norm
> reconstruction `norm_{L+1} * (x / norm_L)`, and reports top-1 ~99% / top-8 covering 81.5% of
> actual selections. A transformer block is residual, so `x_{L+1} = x_L + delta` and the router
> tolerates the delta. One-layer-ahead prefetch is therefore practical, not merely theoretical.
> The claim is self-reported and unverified here; it is measurable with Phase A's profiler. But *within* a
layer, `ffn_moe_topk-<il>` is computed before the `mul_mat_id` that consumes it, and
`mul_mat_id` then faults the 8 slabs **serially** (`ggml-cpu.c:1647-1654`). Issuing 8 parallel
`MADV_WILLNEED` calls the moment top-k is known converts 8 serial page-fault stalls into one
parallel batch. On NVMe that is worth roughly `7/8 * 22ms` on the SSD-resident fraction. It
needs a hook between the top-k node and the matmul node, which is more invasive than C.1 -
scope it only if C.0/C.1 measurements show SSD stalls dominating.

### Phase D - reorder experts by popularity (standalone tool, zero llama.cpp changes)

Routing on this arch is a plain `ffn_gate_inp` of shape `{n_embd, n_expert}` into softmax
top-k (`src/models/qwen35moe.cpp:502-510` passes `exp_probs_b = nullptr` and
`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`). **There are no expert groups** - no group-based
routing that would assume contiguous expert blocks.

That makes expert identity a pure permutation. For each layer, permuting

- the `i02` slabs of `ffn_up_exps` / `ffn_gate_exps` (or fused `ffn_gate_up_exps`) and
  `ffn_down_exps`,
- the rows of `ffn_gate_inp.weight`,
- the `{n_expert}` scale vectors `ffn_*_exps_s` if present (`src/llama-model.cpp:1410-1417`),

is **exactly output-equivalent**, and can be done as a GGUF-to-GGUF rewrite in `gguf-py`. No
inference code changes at all.

Two payoffs:

1. **Immediately**: the hot set becomes a contiguous byte range per layer, so Phase C's
   `mlock`/`madvise` becomes one call per layer instead of hundreds, and the kernel's readahead
   starts working with you instead of against you.
2. **Later**: it is the enabler for Phase E. A contiguous prefix of experts is a
   `ggml_view_3d`; an arbitrary subset is not.

Risk to check before building: verify the permutation is genuinely output-identical by
comparing logits at temp 0 on the original vs permuted GGUF. Also confirm nothing else in the
tree assumes expert-index semantics (imatrix files are keyed by tensor name, so re-quantizing
from a permuted source would need a permuted imatrix).

### Phase E - a real VRAM expert tier (only if Phase A clears the 1.3 bar)

This is the part of the request that is genuinely a new subsystem, and `AGENTS.md` is explicit
that this repo prefers "a simpler change that does 90% of the job". Read section 1.3 again
before starting: under uniform routing, Phase B item 5 already does ~100% of this job.

The obstacle is structural, not incidental. `mul_mat_id` takes one `src0` covering all 256
experts. Splitting into a GPU-resident hot tensor `[n_embd, n_ff, K]` and a CPU-resident cold
tensor `[n_embd, n_ff, 256-K]` requires **two** `mul_mat_id` nodes, and the naive versions all
fail the same way:

- Clamping ids (`min(ids, K-1)` / `max(ids-K, 0)`) and masking the outputs makes **both**
  matmuls compute all 8 slots. The CPU still does 8 experts of work per layer, so the saving is
  zero - and CPU work is the bottleneck.
- ~~The only thing that works is **compacting the ids**~~: partition the `(slot, token)` pairs
  into a hot list and a cold list, run each `mul_mat_id` over its own shorter ids tensor (the
  `cne1 == 0` skip at `ggml-cpu.c:1650` then does the right thing), and scatter-add the two
  results back into `[n_ff, n_expert_used, n_tokens]`.

> **CORRECTION (section 11): the ids-compaction claim above is wrong.** PR #26563 does it with a
> **zero sentinel slot plus a cold mask** and needs no partition/scatter op at all. The hot GPU
> tensor is `[ne0, ne1, hot_s + 1]` where slot `hot_s` is zero-filled; a per-layer LUT maps cold
> expert ids to that sentinel so their GPU contribution is exactly zero, and a `cold_mask` does
> the mirror-image job on the CPU side. The two results are summed. Read section 11 before
> treating anything below as the design.

That needs new ggml ops (a partition/compaction and a scatter) plus loader changes to split the
tensor plus `build_moe_ffn` changes. It is a multi-week change to shared upstream code that
affects every MoE architecture. **Do not start it without prior discussion upstream** - and note
issue [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) is open, unanswered by
maintainers, with no implementation attached. That is a signal about appetite, and the right
first move is to post Phase A's measured distribution there rather than arrive with a large PR.

A cheaper alternative worth evaluating first: at `n_tokens = 1` the ids tensor is 8 int32s.
Breaking the top-k out to the host and partitioning there needs no new ggml ops - but costs a
graph split and a device sync **per layer per token** (40 syncs/token, ~2-4 ms). Against a
~40 ms budget that may be acceptable, and it would additionally give host-side control for
Phase C.2's prefetch and for a dynamic LRU policy. Prototype this before the ops-based design.

---

## 4. Interaction with `--ctx-max`

Auto context growth **is** in this tree (`include/llama.h:1016`, `src/llama-context.cpp:3810`,
`--ctx-max` / `--ctx-grow-factor` at `common/arg.cpp:1598-1612`) - `NOTES.md` Part 1 is stale on
this point, it describes the pre-implementation state.

For this model the relationship is clean and worth stating explicitly, because it is different
from the 27B case:

- Growth does **not** decide whether the model fits (section 0.3: 160 MiB of KV at 8K).
- Growth **does** decide how many experts get pushed down a tier as context extends. At 20
  KiB/token, going 8K -> 128K costs 2.4 GiB = **1,370 experts**, and every one of them comes out
  of the VRAM or RAM tier.

So the correct coupling is: **each growth event should re-run the expert eviction decision**,
demoting the coldest experts by exactly the number of bytes the KV cache just claimed. Phase C's
`mlock`/`madvise` set is the natural place to express that, since it is per-byte-range and
adjustable at runtime with no reallocation.

The 27B plan's Phase 4 (refuse-to-grow when free VRAM is below the transient
old+new peak of `llama_kv_cache::resize()`) applies unchanged and is a prerequisite.

---

## 5. Open items

1. **`H(f)` for Qwen3.6-35B-A3B generation-phase routing.** Everything in Phases D and E is
   contingent on it. Phase A, item 1.
2. **Real per-expert slab size at the UD-XL mix.** Section 0.2 uses pure Q4_K (1.6875 MiB); the
   file sizes imply the real mix is heavier. `02-gguf-layout.py` extension.
3. **What storage the server actually has.** Section 2's SSD row spans 25 ms (NVMe Gen3) to
   75 ms (SATA) per token - a 3x swing in the dominant term. Measure `fio` random 1.7 MiB QD1
   before sizing anything.
4. **Does an MTP-head quant of the 35B-A3B exist?** MTP was the single largest measured win on
   the 27B (+70%). This arch supports MTP layers either in-checkpoint or as a sidecar
   (`src/models/qwen35moe.cpp:41-43`, `mtp_only` path).
5. **Is the hot set stable across domains?** Phase A item 3. If it is not, static placement
   (Phases C/D) degrades to a dynamic-cache problem and Phase E's cost/benefit changes.
6. **`NOTES.md` A.4** - whether a GPU-destined tensor's host-side shadow stays mmap'd - is still
   the highest-priority unresolved item in the repo and gates Phase E's demote-without-copy path.

---

## 7. REVISED PLAN for `UD-Q3_K_M` - what to cram into VRAM

Supersedes sections 2 and 3. Sections 1.1, 1.3 and 1.4 are unchanged and still govern.

### 7.1 Byte model at Q3_K_M

`UD-Q3_K_M` is 16.6 GB = 15.46 GiB over ~34.0 B params = **3.92 bpw average**. Splitting that
(derived; item 2 in section 5 is now the top open item - `02-gguf-layout.py` must confirm the
real per-tensor mix):

| group | params | est. bpw | resident | **read per token** |
|---|---|---|---|---|
| `output.weight` | 508.6 M | ~6.5 | 414 MiB | **414 MiB** |
| GDN weights (30 layers) | ~450 M | ~5.5 | 310 MiB | **310 MiB** |
| attention (10 layers) | 189 M | ~5.5 | 130 MiB | **130 MiB** |
| shared experts (40) | 125.8 M | ~4.5 | 70 MiB | **70 MiB** |
| routers `ffn_gate_inp` | 21 M | ~8 | 22 MiB | **22 MiB** |
| **dense subtotal** | **1.29 B** | | **946 MiB** | **946 MiB** |
| routed experts | 32.21 B | ~3.8 | 14.2 GiB | **454 MiB** (8/256) |
| `token_embd.weight` | 508.6 M | ~6.5 | 414 MiB | 8 KiB (one row) |
| **total** | **34.0 B** | | **15.5 GiB** | **1,400 MiB** |

One expert slab = `3 * 2048 * 512 * 3.8 / 8` = **1.42 MiB**. One layer = 364 MiB. All
10,240 = 14.2 GiB.

**The load-bearing line: 946 of 1,400 MiB read per token - 68% - is dense.** The 27B intuition
does not transfer, because a 256-expert top-8 router makes the sparse half genuinely small
while the 248,320-token vocabulary makes `output.weight` enormous relative to a 2048-wide model.

### 7.2 The value-per-byte table

Value = bytes read per token / bytes resident. This is the only ranking that matters when the
constraint is VRAM capacity.

| what | resident | value/byte | VRAM cost to buy 1 MiB/token of savings |
|---|---|---|---|
| `output.weight` | 414 MiB | **1.00** | 1 MiB |
| GDN / attention / shared / routers | 532 MiB | **1.00** | 1 MiB |
| one layer of routed experts | 364 MiB | **0.031** | **32 MiB** |
| `token_embd.weight` | 414 MiB | ~0.00002 | never |

**32:1.** Restating it as a budget: the first 946 MiB of VRAM removes 946 MiB/token of host
traffic. The next 946 MiB, spent on 2.6 layers of experts, removes 30 MiB/token.

`token_embd` is correctly pinned to the CPU unconditionally
(`src/llama-model.cpp:1334-1336`, confirmed measured on the 27B) - 414 MiB that must never
compete for VRAM. `output.weight` sits in slot `n_layer_all` and is the *first* thing offloaded
(`src/llama-model.cpp:1318,1333`), which is exactly right here and is free.

### 7.3 The default fitter already does the right thing. Verify, do not rebuild.

This is the good news and it deserves to be checked before any code is written.
`common_fit_params`' MoE path:

- **Step 3** (`common/fit.cpp:451`) sets `overflow_type = LAYER_FRACTION_MOE`, whose pattern is
  `blk\.<il>\.ffn_(up|down|gate_up|gate)_(ch|)exps` (`common/fit.cpp:437`). That pushes **only**
  routed experts to system RAM and keeps every dense tensor on the GPU - precisely the 7.2
  ordering, as a hardcoded rule.
- The pattern does **not** match `ffn_up_shexp` / `ffn_gate_inp` (it requires the literal
  `exps` suffix), so the shared expert and the router stay on GPU. Both are value 1.0. Correct
  by accident, but correct - **confirm empirically with `-lv 4` and the per-buffer log.**
- **Step 4** (`common/fit.cpp:645-700`) then promotes whole layers' experts back onto the GPU by
  bisection until the margin is hit.

VRAM budget at the measured 5748 MiB, 8K context, no MTP:

```
  5748  total
-  946  dense weights (section 7.1)
-  500  compute buffer      (~184 with MTP - measured on the 27B)
-  160  KV at 8K            (20 KiB/token x 8192)
-   90  recurrent state, 30 GDN layers  (~360 with MTP - MTP quadrupled it on the 27B)
-  300  margin (-fitt; default 1024 stranded ~1.3 GiB on the 27B)
= 3752 MiB for routed experts  ->  10.3 of 40 layers  ->  26%
```

Projected token time:

| | bytes/token | bandwidth | time |
|---|---|---|---|
| dense, on GPU | 946 MiB | 288 GB/s | 3.2 ms |
| experts, 26% on GPU | 118 MiB | 288 GB/s | 0.4 ms |
| experts, 74% on host | 336 MiB | ~18 GB/s (see below) | 19.5 ms |
| **total** | | | **~23 ms -> 43 t/s ceiling** |

Host bandwidth: the 27B measured 13.16 GiB of CPU-side weights in 482 ms of a 495 ms token =
**28.6 GB/s**, but that is a dense model reading large contiguous matmuls. MoE decode reads 320
scattered 1.42 MiB slabs (3 sub-reads each) into 2048x512 matvecs with poor cache reuse, so
derate to ~18 GB/s. Against the 27B's measured 2.02 t/s, expect **10-20 t/s** in practice - a
5-10x win, delivered almost entirely by the architecture and the existing fitter.

**Gate: if the stock `--fit` run lands in that band, sections 3's Phases D and E are dead.**

### 7.4 MTP is worth materially less on MoE than on dense - and Phase A predicts by how much

This is a new finding and it changes the second-biggest lever from the 27B work.

On the dense 27B, `--spec-type draft-mtp` verified a mean 3.32 tokens per pass, reading the
16.7 GiB of weights **once** instead of 3.32 times - **measured +70% generation**.

On a MoE model that does not hold. Verifying 4 tokens means `n_tokens = 4` in the `mul_mat_id`,
which needs the **union** of those 4 tokens' expert sets per layer. Let `U` be that union size:

| token-to-token expert overlap | `U` per layer | bytes for 4 tokens | vs 3.32 unspeculated |
|---|---|---|---|
| perfect (U = 8) | 8 | 1,400 MiB | 3.3x |
| 50% (U = 20) | 20 | 2,082 MiB | **2.2x** |
| none (U = 32) | 32 | 2,764 MiB | 1.7x |

The dense 946 MiB is always read once and always gets the full 3.32x. Only the expert half
degrades. So MTP here is worth somewhere between 1.7x and 3.3x fewer bytes, against 3.32x on the
27B - i.e. plausibly **+30-50% generation instead of +70%**.

**`U` is exactly Phase A item 2.** That measurement now pays twice: it sets the ceiling on
per-expert cache designs *and* it predicts MTP's value before you spend a day chasing an MTP
quant. Run it first.

Two costs to keep in view: MTP quadrupled the 27B's recurrent state (149.62 -> 598.50 MiB);
here 30 GDN layers at `n_embd` 2048 should scale to roughly 90 -> 360 MiB, about one layer of
experts. It also shrank the compute buffer (513 -> 184 MiB), which partly pays for it.

Do **not** try to exploit the `n_tokens = 4` batch by lowering `GGML_OP_OFFLOAD_MIN_BATCH`
(default 32, `ggml/src/ggml-cuda/ggml-cuda.cu:5377`). Offloading a `mul_mat_id` streams the
whole 364 MiB `_exps` tensor over a 12 GB/s PCIe link to save a 18 GB/s host read. It is a
straight loss - see section 1.4.

### 7.5 What is actually left to tune, in order

1. **Verify 7.3.** One `--fit` run at `-lv 4`. Confirm every dense tensor is on `CUDA0`, the
   shared expert and router are on `CUDA0`, `token_embd` is `CPU_Mapped`, and count how many
   layers' `_exps` got promoted. If any dense tensor landed on the host, that is the whole
   optimization and nothing else matters.
2. **`-fitt`.** The 1024 MiB default stranded ~1.3 GiB on the 27B. Here 1.3 GiB is **3.6 more
   layers of experts**, i.e. a third again of the expert-VRAM budget. Highest-value zero-code
   knob, same as before.
3. **`GGML_CUDA_FORCE_MMQ` + Pascal arch.** Measured +135% prompt processing on this card. Free.
   Unchanged.
4. **Which layers get their experts promoted.** Step 4 fills **front-to-back** - layers 0, 1, 2
   ... - which is arbitrary, not value-derived. Two candidate orderings worth an A/B, both
   expressible with `-ot` today:
   - **Align with the 10 full-attention layers** (il 3, 7, ... 39) so each layer's whole compute
     stays on one device. The 27B measured `graph splits = 82` at bs=1; fewer device crossings
     per token is worth measuring on a card this slow.
   - **Align with whatever Phase A shows is most concentrated.** Routing skew is usually not
     uniform across depth; early layers tend to be flatter. Spend VRAM where routing is
     peakiest, since that is where residency converts into hits.
5. **Spend the 3.9 GiB of slack.** Model 15.46 GiB against a ~19.4 GiB budget. Three options,
   in value order:
   - **Upgrade the dense tensors' precision, not the experts'.** Dense is only 1.29 B params
     and is read on every token, so it is where bits buy the most quality per byte *and* the
     bytes are already GPU-resident. Taking `output`/attention/GDN from ~6 to 8 bpw costs
     ~320 MiB of the slack. `llama-quantize --tensor-type` expresses this. Check what
     `UD-Q3_K_M` already does here first (open item 2) - Unsloth's UD mixes may have done it.
   - **Context.** At 20 KiB/token, 3.9 GiB is ~200K tokens. This model's 262144 `n_ctx_train` is
     genuinely reachable on this box, which was never true for the 27B (64 KiB/token).
   - **Leave it as page-cache headroom** so nothing is ever evicted. Watch `majflt`, not
     `MemAvailable` - see `PLAN-qwen36-27b.md` section 1.2's caveat.
6. **Only then** reconsider Phase A's `H(f)` for per-expert VRAM placement. The prize has
   shrunk: the entire expert-in-VRAM tier is 4.6 ms of a ~23 ms token. Going from 26% (whole
   layers) to 40% (skew-selected experts) saves ~2.5 ms - **~11% end-to-end, for the
   multi-week ggml change in Phase E.** The section 1.3 decision bars should be raised
   accordingly.

### 7.6 What changed from the pre-revision plan

| | `UD-Q4_K_XL` (old target) | `UD-Q3_K_M` (now) |
|---|---|---|
| file | 20.9 GiB | 15.46 GiB |
| overhang vs 19.4 GiB budget | **1.5 GiB (~900 experts)** | none, 3.9 GiB spare |
| SSD reads/token | ~22 slabs, 25-75 ms | **zero** |
| top lever | steer the RAM/SSD boundary (Phase C) | **verify the fitter put dense on GPU** |
| Phase C (mlock/madvise) | highest value | **dropped** |
| Phase D (expert reorder) | enabler + readahead win | **dropped** (no eviction to order) |
| Phase E (VRAM expert tier) | contingent on `H(f)` | contingent on `H(f)`, prize now ~11% |
| Phase A (profiling) | decision gate | **still worth doing, for `U` (7.4) more than `H(f)`** |

---

## 9. Why host-side sparsity is free and VRAM-side sparsity is not

Recurring question, worth writing down once. The asymmetry is not an implementation wart.

**Reading an expert from host RAM is an address. Putting an expert in VRAM is an allocation.**

1. The whole 14.2 GiB of experts is already mapped into one flat virtual address space
   (`MAP_SHARED | PROT_READ`, `src/llama-mmap.cpp:447,456`). Selecting expert 113 is one
   multiply-add - `src0->data + cur_a*nb02` (`ggml/src/ggml-cpu/ggml-cpu.c:1654`), with unused
   experts skipped outright at `:1650`. There is no load step. The CPU issues loads against a
   virtual address and the **MMU plus page cache resolve them in hardware, on demand, at 4 KiB
   granularity**. Selectivity is free because it is *lazy*: nothing was decided in advance, you
   simply never touched the bytes you did not need.

2. VRAM has no demand paging. No MMU faults a GPU load into a host page. Something must decide
   **in advance** which bytes live there, allocate a buffer, and DMA them across. Selectivity
   has to be *eager*.

3. And eager is impossible, because **routing is only known at the moment of use**. Layer L's
   router consumes layer L-1's output. By the time "expert 113" exists as a fact, you are
   microseconds from needing it: 1.42 MiB over PCIe 3.0 is ~120 us plus launch latency, against
   a whole-layer compute of a few hundred us. You would stall every layer. So GPU expert
   residency must be **predicted**, never **demanded** - and that is what turns pointer
   arithmetic into a cache-policy problem.

4. Independently: **`ggml`'s tensor is the unit of allocation, and its atomicity is what makes
   `mul_mat_id` work at all.** The op's design is "all 256 experts at a fixed stride from one
   base pointer." Split the tensor and that arithmetic breaks - you need two base pointers,
   per-slot routing between them, and a scatter to recombine. Section 1.1 and Phase E.

5. **The mechanism being imagined does exist, in two forms, and both are slower:**
   - **Unified memory.** `cudaMallocManaged` behind `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
     (`ggml/src/ggml-cuda/ggml-cuda.cu:141-142`). GPU page faults migrate host pages on demand -
     literally "request just what you need." It is an OOM escape hatch, not a performance path:
     fault latency ~10-50 us at coarse migration granularity, against 320 slabs per token.
   - **Zero-copy mapped host memory.** The GPU reads host RAM directly over PCIe with no copy.
     This is exactly the requested behavior, and it runs at **PCIe 3.0 ~12 GB/s - slower than
     the CPU's own ~28 GB/s path to the same RAM**. The GPU becomes a worse CPU. Note the CUDA
     backend reports `buffer_from_host_ptr = false` (`ggml-cuda.cu:4715`), which is also why
     `NOTES.md` A.4's demote-without-copy premise needs checking before Phase E leans on it.

**Punchline:** host-side sparsity is free because the CPU already owns all the memory - the page
cache is a demand-paged, hardware-accelerated, LRU expert cache you get for nothing. VRAM buys
10x bandwidth and pays for it by being a manually managed scratchpad with no fault handler. The
cost is not selectivity; it is the absence of an MMU.

Corollary, and the reason section 7.2 is the right frame: **because you cannot fetch on demand,
you must commit in advance, so you should commit the bytes with the highest read probability.**
Dense weights read at probability 1.0; a routed expert at 8/256 = 0.031. The 32:1 is not an
artifact of llama.cpp - it is the structural reason expert placement in VRAM is a poor trade
against dense placement.

---

## 10. Adaptive MTP - the hook exists, but not the signal that was proposed

**Measurement to explain (user, on the 35B):** MTP only wins at `n_draft = 1`. Beyond that the
cost of pulling extra experts overwhelms the acceptance gain.

### 10.1 That measurement is itself a measurement of expert overlap

> **CORRECTION (user, confirmed by testing): the 27B's MTP results do not transfer to the
> 35B-A3B.** The acceptance profile `(0.947, 0.737, 0.632)` and the +70% figure are dense-27B
> measurements and must not be used as 35B constants anywhere in this document. The table below
> is kept only to show the *shape* of the trade-off; every number in it needs re-measuring on the
> 35B before it means anything. Sections 7.4 and 11.2 inherit the same caveat.

Section 7.4's model: a verify batch of `m` tokens costs `~30 host layers x U(m) x 1.42 MiB`,
where `U(m)` is the **union** of those tokens' expert sets. First order, with token-to-token
overlap `o`: `U(m) ~= 8 * (1 + (m-1)(1-o))`. Illustrative only, using the 27B's per-position
acceptance `(0.947, 0.737, 0.632)` as a placeholder for an unmeasured 35B profile:

| `o` | k=1 (m=2) | k=2 (m=3) | k=3 (m=4) | best |
|---|---|---|---|---|
| 0.7 | 0.187 | 0.207 | 0.218 | k=3 |
| 0.5 | ~flat | ~flat | ~flat | - |
| 0.3 | 0.143 | 0.138 | 0.134 | **k=1** |

(throughput `~ accepted(k) / U(k+1)`, arbitrary units)

**So "k=1 is optimal" implies `o` is below roughly 0.5 on this model.** That is a falsifiable
prediction, checkable in an afternoon with Phase A item 2, and it should be the next thing run.

It also has a consequence bigger than MTP: **low overlap hurts a dynamic LRU expert cache
exactly as much as it hurts speculation.** If consecutive tokens share under half their experts,
an LRU VRAM cache thrashes. That argues for *static, popularity-ranked* placement over *dynamic
caching* if Phase E is ever built - a real simplification.

### 10.2 The proposed signal does not exist and would not vary

"Only draft further if the expert is already in VRAM" cannot be evaluated:

- **Per section 1.1, there is no per-expert residency.** All 256 experts of a layer are one
  tensor in one buffer. "Is expert 113 in VRAM?" resolves to "is `blk.L.ffn_up_exps` on
  `CUDA0`?", which is fixed at load time and identical for every token. The check is not merely
  cheap - it is a constant, and therefore carries zero information for a per-token decision.
- **Even with Phase E's per-expert residency it would still be unavailable**, because you cannot
  know which experts the verify pass needs until you run it. Trunk layer L's router consumes
  layer L-1's output *for that token*. The MTP head has its own separate MoE router
  (`src/models/qwen35moe.cpp:681-693`, `NOTES.md` A.6) whose expert choices are its own module's,
  not the trunk's.

### 10.3 What is implementable: observe, do not predict

**The adaptive stopping rule already exists.** `common/speculative.cpp:1652-1657`, inside the MTP
draft loop, evaluated per drafted token per sequence:

```cpp
                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }
```

capped by `params.n_max` at `:1666`. So "sometimes draft 1, sometimes draft 3" is already the
behavior; it is keyed on draft confidence alone. Three steps, cheapest first:

1. **Tune `--spec-draft-p-min` (`common/arg.cpp:4043`) with `--spec-draft-n-max 3`. Zero code.**
   Raising `p_min` makes the existing rule bail out early on exactly the low-confidence steps
   whose extra experts are not worth fetching. This may capture most of the win, and it must be
   measured before anything is written. Compare against the flat `--spec-draft-n-max 1` the user
   found optimal.
2. **Add a cost term to the same stopping rule (~50 lines).** After each verify pass both
   `n_accepted` and the realised `U` are known - `U` from the same `ffn_moe_topk-<il>` tensors
   Phase A already reads. Hill-climb `n_max` over {1, 2, 3} to maximise `n_accepted / U`,
   updated every ~32 tokens. No prediction required, only observation, and it lands next to an
   existing rule rather than adding a subsystem - which is the bar `AGENTS.md` sets. It would
   find the user's k=1 automatically, and would find k=2 or 3 on workloads where routing is
   stickier (plausibly code vs prose - Phase A item 3).
3. Only in a Phase-E world does a residency-weighted variant become expressible, and by then
   step 2's controller already subsumes it: a VRAM-resident expert contributes ~nothing to the
   measured cost, so `n_accepted / cost` self-adjusts without ever naming residency.

**Prerequisite for all three: confirm the 27B's acceptance profile transfers.** `p_min`,
`n_max`, and the controller are all sized off `(0.947, 0.737, 0.632)`, which was measured on a
dense 27B, not this model.

---

## 11. Review - PR #26563, "Expert caching", miltos22

<https://github.com/ggml-org/llama.cpp/pull/26563>. **Open, no maintainer review yet, author
mid-redesign.** Requires 2 approvals from CISC / ggerganov / JohannesGaessler / ngxson.

This is Phase E, built, and benchmarked on **Qwen3.6-35B** specifically. It changes what this
plan should do next more than anything else found so far.

### 11.1 What it does

New: `src/llama-expert-{heatmap,hotstore,tier}.{cpp,h}` (~855 lines) and
`ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.{c,h}` (~263 lines). Modified: `common/arg.cpp`,
`common/fit.cpp` (~50), `src/llama-graph.cpp` (~14), `src/llama-context.cpp` (~65), and others.

- **Hot store**: a GPU tensor per expert weight, shape `[ne0, ne1, hot_s + 1]`. Slot `hot_s` is a
  zero-filled **sentinel**.
- **Routing**, in `llama_expert_tier_build()`, a drop-in replacement inside `build_lora_mm_id()`:
  a per-layer LUT (`hot_lut`) maps real expert ids to GPU slots, and every *cold* id to the
  sentinel - so the GPU's contribution for a cold expert is exactly zero. The CPU runs a new op
  `ggml_mul_mat_id_cold` with a `cold_mask` doing the mirror-image job. The two are scaled
  per-expert and summed.
- **Heat**: `build_moe_ffn` stashes `selected_experts` per layer
  (`res->moe_sel_experts.emplace_back(il, selected_experts)`); `process_ubatch()` reads them back
  with `ggml_backend_tensor_get()` after sync and updates decay-weighted counters
  (`--expert-heat-decay`, default 0.999). `get_top_s()` ranks by heat; hysteresis gates swaps.
- **Flags**: `-ehs N` / `--expert-hot-s` (slots, `-1` = auto-fit from free VRAM), `--ecf`
  (force on non-CUDA), `--expert-heat-decay`, `--expert-heat-log-period`.

**This resolves section 3 Phase E's blocker.** The sentinel-plus-mask trick avoids the ids
partition/scatter ops that section said were unavoidable. Credit where due; that section was
wrong.

### 11.2 The disqualifying interaction: it is mutually exclusive with MTP

```cpp
if (cur->ne[2] > 1) return nullptr;
```

`ne[2]` of a `mul_mat_id` is `n_tokens` (confirmed independently: `get_op_batch_size` returns
`op->ne[2]` for `GGML_OP_MUL_MAT_ID`, `ggml/src/ggml-cuda/ggml-cuda.cu:5195-5198`). Stated
rationale: "CUDA MMQ mul_mat_id compaction assumes distinct expert ids per token; our sentinel
duplicates OOB there. Decode (n_tokens==1) is safe via mmvq."

**An MTP verify batch is `1 + n_draft` tokens, so `ne[2] >= 2`, so expert caching silently
falls back to stock for the entire verification pass** - which is where all the expert traffic
is. Prompt processing likewise gets nothing.

So on this box the two largest available levers cannot be combined:

| configuration | mechanism |
|---|---|
| stock + `--spec-type draft-mtp`, `n_max 1` | measured +70% on the 27B; user reports k=1 optimal on the 35B |
| PR #26563 + `-ehs N`, **no MTP** | author reports 1.72x-2.07x on Qwen3.6-35B |

**This is a straight A/B, and it is the single highest-value experiment now available.** Neither
number was measured on this hardware.

The restriction looks fixable in principle (it is a CUDA MMQ kernel-path constraint, not a design
one), and MTP at `n_max 1` means only `n_tokens == 2` needs to work. Worth raising in the PR
thread - the author does not appear to have considered speculative decoding.

### 11.3 It is also the cheapest possible Phase A profiler

`--expert-heat-log-period` prints the top-8 experts per layer from real decay-weighted counters.
That is section 3 Phase A's deliverable, already written, in someone else's tree.

More than that, **the reported speedups are indirect evidence on `H(f)`, the number section 1.3
makes everything contingent on.** On 8 GB VRAM with a Q5 quant the hot store holds roughly 25%
of experts. Uniform routing would move 25% of expert traffic to VRAM and yield well under 1.3x.
A measured 1.7-2.1x requires the hot 25% of slots to be capturing something like 60% of
activations.

If that reproduces here, `H(0.26) ~= 0.6` and **section 1.3's decision rule clears its top bar**
- which would make this the first real evidence that per-expert placement beats the whole-layer
`-ot` baseline on this architecture. Verify it directly from the heat log rather than inferring
it from a throughput ratio.

Caveat on transferability: their baseline is not necessarily this repo's fitter optimum, and
their host is larger (a 26 GB Q5 model on 8 GB VRAM implies >=32 GB RAM). Derived host bandwidth
from their Q5 numbers is ~32 GB/s against this box's measured 28.6 GB/s, and hot-store residency
works out at ~25% there vs ~26% here (3.75 GiB / 1.42 MiB = ~2,640 slots), so the *shape* should
transfer even if the absolute t/s does not.

### 11.4 The memory concern is real but probably benign here - under mmap only

Reviewer siganos: "Your code uses significantly more memory than vanilla llama... I run out of
memory and have to use mmap." Author acknowledged and proposed a future "sidecar system" to move
rather than copy.

Hot experts are **copies**: `ggml_backend_tensor_set()` into VRAM, originals left in place. So
every hot expert exists twice. Against the section 7.3 arrangement this looks bad - the fitter
puts ~3.75 GiB of `_exps` on the GPU and only ~11.4 GiB on the host, whereas this PR needs the
full 14.2 GiB of experts host-*resident* as the cold path's backing store, plus a 3.75 GiB VRAM
copy.

**But under default mmap it should self-correct.** Once `copy_top_s()` has uploaded a hot
expert, its host pages are never read again, so the kernel evicts them naturally. The steady-state
host working set is the *cold* experts only, ~0.74 x 14.2 = **10.5 GiB** - comfortable on 16 GiB.
The duplication only bites with `--no-mmap` (anonymous copies, nothing to evict) or under
`mlock`, which is almost certainly what siganos hit.

Two things to watch on a 16 GiB box regardless:

- **Startup.** `copy_top_s()` touches ~3.75 GiB of hot experts to upload them, on top of the
  `MAP_POPULATE` prefault of the whole 15.46 GiB file (`src/llama-model.cpp:1532`). That is the
  worst-case page-cache churn scenario, and `PLAN-qwen36-27b.md` Phase 1b (bounded prefetch,
  ~30 lines, already scoped) is now a prerequisite rather than a nice-to-have.
- **`majflt`, not `MemAvailable`.** Same caveat as always.

### 11.5 Two integration hazards specific to this fork

1. **`-ehs -1` auto-fit versus `--ctx-max`.** Auto-fit claims free VRAM at load time. This tree's
   context growth (`llama_set_n_ctx`, `include/llama.h:1016`) then allocates *more* VRAM later,
   and `llama_kv_cache::resize()` transiently holds old+new. The PR cannot know about that.
   **Use explicit `-ehs N`, not `-1`, until the refuse-to-grow guard (`PLAN-qwen36-27b.md`
   Phase 4) exists and accounts for the hot store.**
2. **It modifies `common/fit.cpp` (~50 lines).** Verify with `-lv 4` that dense tensors still all
   land on `CUDA0` - per section 7.2 the dense-first ordering is worth 32x a routed expert, and a
   hot store that steals VRAM from dense weights would be a large net loss wearing the costume of
   a win. This is the first thing to check, before looking at any t/s number.

Merge conflicts against this fork are likely in `src/llama-context.cpp` (Part 1 context growth)
and `common/fit.cpp`, and manageable.

### 11.6 Determinism

"Even at temperature 0, different cached expert selections produce minor output variations."
Expected: the hot half runs on CUDA and the cold half on CPU, with different rounding, then they
are summed. Worse, **which experts are hot changes during a session** (decay plus hysteresis
swaps), so the same token in the same context can produce different numerics at different points
in the same run.

That breaks the temp-0 A/B oracle `NOTES.md` item 7 wants and `PLAN-qwen36-27b.md` relies on.
Any evaluation of this PR needs a different quality measure - KL divergence against a stock run
over a fixed corpus, which is the tooling the `UD-Q3_K_M` decision already used.

### 11.7 Verdict

**Yes, it helps - as an experiment to run, not a base to build on.**

- Do: build the branch, run it, read `--expert-heat-log-period` to settle `H(f)` and section
  1.3's decision rule. That is a week of this plan's Phase A, free.
- Do: the 11.2 A/B against stock + MTP `n_max 1`.
- Do not: fork from it or start writing on top of it. Open, zero maintainer review, 2 approvals
  needed, author explicitly mid-redesign ("a fundamental redesign reducing RAM overhead"). Under
  `AGENTS.md`'s standards this is not a foundation yet.
- Do: raise the `ne[2] > 1` / MTP interaction in the thread once measured. It is a concrete,
  well-scoped finding the author appears not to have considered, and `n_max 1` means only
  `n_tokens == 2` needs to work.

---

## 12. PLAN - make expert caching (PR #26563) work with MTP

Goal: remove `if (cur->ne[2] > 1) return nullptr;` so the tiered hot/cold expert path stays
active during an MTP verify batch.

**The investigation below concludes this is a ~5-line guard change plus validation, not a kernel
rewrite.** The PR's stated rationale is correct about *why* a restriction is needed but wrong
about *where* the boundary is.

### 12.1 Diagnosis: the real blocker is `mm_ids_helper`, and it is not reached at MTP batch sizes

**What actually breaks.** The MMQ path compacts ids with `mm_ids_helper`
(`ggml/src/ggml-cuda/mmid.cu:28-120`), one CUDA block per expert. Its inner loop:

```c
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
```

`iex_used` is **last-writer-wins** and `it_compact` increments **once per token**, not once per
occurrence. So when one expert id appears twice in a single token's slot list, only one of the
two `dst` rows is ever written; the other is left as whatever was in the pool buffer. Separately,
`nex_prev` counts *occurrences* while `it_compact` counts *tokens*, so `expert_bounds` (`:114`,
`:120`) is inconsistent whenever duplicates exist.

The PR's sentinel design creates exactly that: within one token, every cold expert is remapped to
the single sentinel slot `hot_s`. A token with 5 cold experts produces 5 duplicate ids.

**Why it is confined.** The sentinel is the **highest** index in the hot tensor
(`[ne0, ne1, hot_s + 1]`), so `nex_prev` is still exact for every hot expert, and the mis-sized
region is the last one - it under-runs rather than overlapping. The damage is limited to
un-written `dst` rows for cold slots.

**And the path is not taken at MTP batch sizes.** `ggml_cuda_mul_mat_id`
(`ggml/src/ggml-cuda/ggml-cuda.cu:1882-1893`) tries mmvq **first**:

```cpp
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
```

`ne2` is `n_tokens` (`MMVQ_MAX_BATCH_SIZE` = 8, `ggml/src/ggml-cuda/mmvq.cuh:3`). On **Turing**,
which is what a GTX 1660 Ti is, `get_mmvq_mmid_max_batch_turing_plus`
(`ggml/src/ggml-cuda/mmvq.cu:141-151`) returns **5 for `Q3_K`** and **8 (the default) for `Q4_K`,
`Q5_K`, `Q6_K`**.

**And mmvq has no compaction at all.** The multi-column id kernel
(`ggml/src/ggml-cuda/mmvq.cu:733-769`) is a direct per-`(slot, token)` map:

```c
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
...
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
```

One block per `(slot, token)`, its own id read, its own `dst` write. **Duplicate ids are
harmless** - two slots pointing at the same expert just re-read the same weights.

### 12.2 Consequence

An MTP verify batch is `1 + n_draft` tokens. So on this hardware:

| expert tensor type | `mmvq_mmid_max` (Turing) | max safe verify batch | **max `n_draft`** |
|---|---|---|---|
| `Q3_K` | 5 | 5 | **4** |
| `Q4_K` / `Q5_K` / `Q6_K` | 8 | 8 | **7** |
| `Q2_K` | 7 | 7 | 6 |

**The `ne[2] > 1` guard is off by a factor of 5 to 8.** The correct predicate is "will this
dispatch pick mmvq", not "is this a single token".

> **CORRECTION (found while implementing M1): MMQ is not the only compacting path, and the PR
> has a latent bug in its *current* single-token form.** `ggml_cuda_mul_mat_f` also calls
> `ggml_cuda_launch_mm_ids_helper` (`ggml/src/ggml-cuda/mmf.cu:87`). For a **non-quantized**
> expert tensor on a non-AMD device, `ggml_cuda_mul_mat_id` falls past the mmvq branch (which is
> gated on `ggml_is_quantized(src0->type)`) and past MMQ, into MMF - **at any `ne2`, including
> `ne2 == 1`**. Sentinel duplicates within a single token break the same "at most one use per
> token" assumption there, so an F16/BF16 MoE with `-ehs` is already exposed today.
>
> The M1 predicate must therefore test **both** the token count and `ggml_is_quantized`, and the
> second half is a bug fix rather than a restriction. Implemented on `expert-cache-vram` as:
>
> ```cpp
>     constexpr int64_t n_tokens_max = 4;
>     if (cur->ne[2] > n_tokens_max || !ggml_is_quantized(w->type)) return nullptr;
> ```
>
> `4` is the lowest per-type bound across **all seven** arch tables in
> `get_mmvq_mmid_max_batch` (`mmvq.cu:114-247` - pascal_older, turing_plus, gcn, cdna,
> rdna1_rdna2, rdna3, rdna4); no entry in any table is below 4, so no device query is needed.
> Section 12.3's M2 test should sweep the **MMF** path too, not just MMQ.

Note `get_mmvq_mmid_max_batch` is a **performance** heuristic (its comment cites PR #20905), not
a correctness bound - above it MMQ is merely *faster* for stock inference. For the tiered path it
doubles as the correctness bound, which is a coincidence worth stating explicitly in any PR
comment so a reviewer does not assume the two are linked.

### 12.3 Phases

**M0 - establish the inputs (no code).**
- Extend `phase0/02-gguf-layout.py` to report the **ggml type of every `blk.N.ffn_*_exps`
  tensor** in `UD-Q3_K_M`. A UD mix is not uniform; the per-layer type sets the per-layer draft
  ceiling from the 12.2 table, so some layers may cap at 5 and others at 8. This is the single
  input the whole plan is sized on.
- Build the PR branch, run `-ehs N` **without** MTP, and confirm section 11's claims on this box:
  dense still all on `CUDA0` at `-lv 4`, host working set settles near 10.5 GiB, and the heat log
  gives the real `H(f)`.
- Run `-ehs N` **with** MTP and confirm the caching silently disables (hit counters go to zero
  during verify). That is the baseline the rest of this section improves on.

**M1 - the guard (the whole feature, if 12.1 holds).**

Replace the `ne[2] > 1` bail in `llama_expert_tier_build()` with a predicate mirroring the CUDA
dispatch: quantized hot-store type, and `n_tokens <= get_mmvq_mmid_max_batch(type, cc)` capped at
`MMVQ_MAX_BATCH_SIZE`. Two ways to get `cc`:
- **Conservative, no new plumbing:** hardcode the minimum across all architectures. The
  Pascal-and-older and GCN tables (`mmvq.cu:114-137`, `:154-175`) bottom out at **4** for the
  K-quants, so `n_tokens <= 4` (i.e. `n_draft <= 3`) is safe on every CUDA/HIP device without
  querying anything. **Start here.**
- **Exact:** thread the device's `cc` into the tier builder. Needed only if measurement shows
  `n_draft > 3` is worth having.

Acceptance: with MTP on, the heat log shows nonzero hits during verify batches, and output
matches `-ehs 0` + MTP within the PR's known numeric variance (section 11.6 - use KL divergence
over a fixed corpus, not token identity).

**M2 - prove the load-bearing assumption.**

Everything above rests on "mmvq is duplicate-safe". Add a `tests/test-backend-ops.cpp` case:
`MUL_MAT_ID` with deliberately duplicated ids, swept over `ne2 = 1..8` and the K-quant types.
It should pass on the mmvq path and **fail on MMQ** - which both validates M1 and documents a
genuine latent bug in `mm_ids_helper` that exists independently of this PR.

That framing matters strategically: *"fix/characterise duplicate-id handling in `mul_mat_id`"* is
a small, self-contained, obviously-correct contribution that a reviewer can evaluate on its own
merits. It is a far better first PR than *"add a hook for expert caching"*, and it de-risks
#26563 without depending on it.

**M3 - heat accounting under speculation.**

`update_from_graph(res->moe_sel_experts)` runs in `process_ubatch()` and counts **every token in
the batch, including rejected drafts**. Two distinct problems:
- Rejected tokens bias the heat toward what the MTP head predicts rather than what the model
  emits.
- Worse, the **decay rate becomes draft-length-dependent.** `--expert-heat-decay 0.999` is
  applied per update; drafting 3 and accepting 1 decays four times as fast per *accepted* token
  as no speculation does. The hot set's time constant then silently changes with `n_draft`.

Fix: snapshot the ids in `process_ubatch()` but **commit on the following call, once
`n_accepted` is known** from `common_sampler_sample_and_accept_n`
(`tools/server/server-context.cpp:3774-3841`), counting only rows `[0, n_accepted)` and decaying
once per accepted token. The `moe_sel_experts` vector already holds what is needed; this is a
deferral, not new plumbing.

Also confirm `resync_top_s()` cannot fire between a draft and its verify, and that the hot set is
fixed for the duration of a verify batch.

**M4 - re-tune draft length. This is where the payoff appears.**

Caching does not merely add to MTP - **it makes MTP better**, by lowering the marginal cost of a
larger expert union. Section 10.1's cost model has marginal cost per extra unique expert
`= 1.42 MiB / 18 GB/s = ~79 us`. With a hot set capturing a fraction `H` of activations, the
expected marginal cost falls to `(1 - H) x 79 us`. At the `H ~= 0.6` section 11.3 infers from the
author's benchmarks, that is **~32 us, a 2.5x reduction** - so the optimal `k` should move *up*,
possibly past the user's measured `k = 1`.

Re-run the `n_max` sweep {1, 2, 3, 4} with `-ehs` on and off. **The user's finding that `k = 1`
is optimal was measured without caching and should not be assumed to survive it.** If `k_opt`
stays at 1 even with caching, that is a strong signal that token-to-token expert overlap is very
low, and it kills the `n_draft > 3` half of M1 as well as the LRU-cache design space generally
(section 10.1).

Only if `k_opt > 3` does the exact-`cc` variant of M1 earn its keep.

**M5 - lift the ceiling entirely (only if M4 says `k_opt` is at the cap).**

Two options, both eliminating duplicate ids rather than working around them:

- **Fix `mm_ids_helper` for duplicates.** Emit one compact row per *occurrence*: `it_compact`
  increments by the occurrence count (`warp_reduce_sum` instead of `warp_reduce_any` at
  `mmid.cu:57,75`), which then agrees with `nex_prev`'s existing occurrence semantics. Shared-memory
  sizing for `store[]` changes accordingly. Correct for everyone, but it touches a hot,
  performance-critical kernel with an owner - discuss before writing.
- **The landing-pad remap - no kernel change at all.** Reserve slots `[0, n_expert_used)` of the
  hot store as a pad and place real hot experts at `n_expert_used ..`. Map a cold expert sitting
  at slot position `j` to hot slot `j`; hot experts keep their own slot, always `>= n_expert_used`.
  Within a token the cold positions are `0..7` (distinct by construction) and the hot ones are
  `>= 8` (distinct because distinct experts get distinct slots), so **duplicates are impossible**.
  Then multiply the hot output by a `[1, n_expert_used, n_tokens]` 0/1 mask - `ggml_mul` already
  broadcasts via `ggml_can_repeat`, and the PR **already** does a per-expert scale multiply on the
  hot path, so the mask folds into it for free.

  Cost: 8 wasted slots per layer = `8 x 1.42 MiB x 40` = **~455 MiB**, about 12% of the hot store.
  Benefit: works on *every* CUDA path including MMQ, so it removes the batch-size ceiling
  outright - and would let expert caching apply to prompt processing, which #26563 currently
  abandons. Whether that second part is a real win needs measuring against op-offload
  (section 1.4), which already streams whole `_exps` tensors at `bs >= 32`.

### 12.4 Risks

1. **CUDA graph capture.** `ggml-cuda.cu:2525-2527` consults `get_mmvq_mmid_max_batch` when
   deciding whether `mul_mat_id` forces a stream sync and therefore disables CUDA graphs. Widening
   the tier path into multi-token batches may change that decision. Check `graph splits` and any
   CUDA-graph disable message before and after.
2. **Mixed expert types across layers.** With a UD mix, one layer's `_exps` may be `Q3_K` (cap 5)
   and another `Q5_K` (cap 8). The guard is per-tensor so this is correct, but it makes caching
   active on some layers and not others within the same batch. Log per-layer so a throughput
   regression is attributable.
3. **mmvq above its heuristic is slower than MMQ for stock.** The tier path forces mmvq at
   `ne2 = 2..5` where MMQ might have been chosen. The hot-path matmuls are small so this should be
   minor, but it is a real cost and it is why M4 must A/B rather than assume.
4. **`-ehs -1` autofit versus the MTP memory probe.** `tools/server/server-context.cpp:1150-1198`
   already inflates `fit_params_target` by the measured MTP context cost; the PR separately
   reserves VRAM for the hot store in `common/fit.cpp`. These must compose or the card is
   over-committed. Section 11.5 - use explicit `-ehs N` throughout M0-M4.
5. **Upstream volatility.** The author is mid-redesign. M2 is deliberately independent of #26563
   so it survives a rewrite; M1/M3 are not.

### 12.5 Sequencing

`M0 -> M2 -> M1 -> M4` is the recommended order. M2 before M1 because it is the only step that
proves the premise, is upstreamable on its own, and costs a day. M3 can run in parallel with M4
but its result changes M4's numbers, so land it first if the heat log looks unstable. M5 is gated
entirely on M4's measured `k_opt`.

---

## 13. Review - PR #26563 updates, and the `miltos22/llama-wackMall` research branch

Reviewed 2026-08-07. **Section 12's plan is substantially overtaken: the author found the same
bug and has already fixed it on the research branch.**

### 13.1 The headline - independent confirmation of section 12.1's diagnosis

`ARCHITECTURE.md` on the research branch states the blocker in the same terms this plan derived
from the kernel source:

> "mm_ids_helper assumed at most one use of an expert per token; sentinel duplicates violate
> that."

and the fix:

> "Fixed to count+rank semantics."

That is exactly **section 12.5's M5 option 1** - make `it_compact` count occurrences rather than
tokens so it agrees with `nex_prev`'s existing occurrence semantics. Two independent derivations
landing on the same defect and the same repair is about as strong a signal as this kind of
analysis gets.

And it explains the split between the two repos:

| | PR #26563 | wackMall |
|---|---|---|
| `mm_ids_helper` fix | **no** | **yes** |
| token limit | `ne[2] > 1` bail | `LLAMA_EXPERT_TMAX`, **default 16** |
| MTP verify batch | falls back to stock | **inside the tiered path** |

**So "make expert caching work with MTP" is already done on the research branch.** The PR is
stuck at one token precisely because it declines to touch the CUDA kernel.

This also decodes the author's stated choice, which is now the gating event for everything:

- **Approach 1** - ~100 lines external, -10% best case / +60% worst case. No kernel change,
  therefore still single-token, therefore **still incompatible with MTP**.
- **Approach 2** - ~500 lines, +30% top-end / +60% low-end, **touches the CUDA kernel**. That is
  the `mm_ids_helper` count+rank fix being ported from wackMall.

If Approach 1 ships, upstream expert caching stays MTP-incompatible.

### 13.2 What survives of section 12

- **M5 is done.** Test it, do not write it.
- **M1 (relax the guard to the mmvq bound) still stands, and is now more interesting.** It is a
  strictly smaller change than Approach 2: no kernel edit at all, just a predicate mirroring the
  dispatch at `ggml/src/ggml-cuda/ggml-cuda.cu:1882-1893`. It buys `n_draft <= 3` portably (the
  Pascal/GCN tables bottom out at 4) or `<= 4` for `Q3_K` on Turing. That is a genuine middle
  path between the author's two approaches and is worth putting in front of him.
- **M2 (the duplicate-id `test-backend-ops` case) is now the single highest-value item.** Reason
  in 13.3.
- **M3 (heat accounting under speculation) is untouched by any of this and still needed.** Both
  repos count every token in the batch; neither knows about rejected drafts. With TMAX 16 this
  gets *worse*, not better, because the draft-length-dependent decay skew (section 12.3 M3)
  scales with the number of speculated tokens.
- **M4 (re-tune `k`) is unchanged and is still where the payoff is measured.**

### 13.3 The review finding: nothing can currently tell the fix from the bug

Both projects carry the same correctness disclaimer - PR: output varies at temperature 0;
wackMall README: *"Greedy output can differ from stock due to rounding differences"*, described
as "different-but-valid".

**A `mm_ids_helper` miscount does not corrupt loudly.** Per section 12.1 it leaves `dst` rows
un-written - whatever was in the pool buffer - which then gets summed with the cold path's
correct result. The symptom is *"output differs from stock"*. That is indistinguishable, by
inspection, from the rounding divergence both projects have pre-declared acceptable.

So the project's own correctness tolerance masks the failure mode of its most delicate change,
and there is **no test upstream that exercises duplicate ids in `MUL_MAT_ID` at all**.

That makes M2 the missing artifact, and it is worth doing regardless of which approach the author
picks: sweep `ne2 = 1..16` with deliberately duplicated ids across the K-quants, expect pass on
mmvq and fail on unfixed MMQ. It is small, self-contained, independently upstreamable, useful to
maintainers who have to review Approach 2, and it is the only thing that can *demonstrate* the
count+rank fix is correct rather than assert it.

### 13.4 The hardware gap - why the headline numbers will not transfer

| | wackMall's bench box | this server |
|---|---|---|
| GPU | RTX 3070, 8 GB, ~448 GB/s, Ampere | GTX 1660 Ti, 5.75 GB, ~288 GB/s, Turing TU116, **no tensor cores** |
| RAM | **31 GB** | **16 GB** |
| host bandwidth | ~40 GB/s (derived from 26.89 t/s at Q4_K_M) | **28.6 GB/s measured** |

Three consequences:

1. **`LLAMA_EXPERT_RAMPOOL` is unusable here - but should also be unnecessary.** It is a
   separately allocated buffer of expert *copies* ("per-layer contiguous blocks, one slot = one
   expert x all its tiered tensors"), i.e. anonymous memory competing with page cache that
   already holds the same bytes. This is what siganos hit ("requiring mmap even with 96 GB
   DDR5"). Its purpose is to cache experts that would otherwise come from SSD - and section 7's
   `UD-Q3_K_M` choice already removed the SSD tier. Keep it at the default 0.

   **But note the author told siganos "without the rampool argument it's basically had no
   advantage."** Read narrowly that is about one user's config; read broadly it suggests the
   headline gains may be rampool-dependent, in which case they will not reproduce here at all.
   **This is the first thing to falsify**, before any of section 12's work is scheduled.
2. **Hot fraction is smaller.** ~3.75 GiB of hot store here (section 7.3) versus ~5.5 GiB there,
   so ~26% expert residency against their ~35-40%. Gains scale roughly with that.
3. **Absolute t/s will be lower** on both the host-bandwidth and GPU-bandwidth terms.

### 13.5 Two wackMall features that matter more here than the VRAM tiering

Both are under-advertised relative to the hot store, and both target this box's actual
bottleneck.

- **`LLAMA_EXPERT_MADVISE=1` (on by default), claimed "~5.4 GiB savings on 35B models."** This is
  section 3's Phase C, already implemented and enabled. On a 16 GiB box, 5.4 GiB is the difference
  between comfortable and thrashing - plausibly worth more than the entire VRAM tier. **Test it in
  isolation** (`LLAMA_EXPERT_S=0` with madvise on) and judge it on `majflt`, not `MemAvailable`.
- **`MOE_COLD`, a fused 3-phase CPU op for cold experts.** Per section 7.1, ~74% of expert traffic
  here is CPU-side, so a fused cold kernel that avoids re-reading the gate/up/down slabs
  separately could matter more than moving 26% of experts to VRAM. Measure it separately from
  tiering.

### 13.6 Hazards specific to this fork

1. **Autofit versus `--ctx-max`.** wackMall's README: *"Current autofit is bugged to use max
   context. Set -c manually to see real benefit."* This tree has auto context growth
   (`llama_set_n_ctx`, `include/llama.h:1016`, `--ctx-max` at `common/arg.cpp:1598`). Combining a
   max-context autofit with `--ctx-max 262144` would reserve ~5 GiB of KV (20 KiB/token, section
   0.3) and leave **zero** hot store. **Always set `-c` explicitly; do not combine with
   `--ctx-max` until this is validated.**
2. **Rebase cost is the real work.** wackMall is "behind master" and "not yet confirmed stable on
   latest upstream master". This fork sits on recent master carrying the Part 1 context-growth
   work and the `common/speculative.cpp` MTP/mmproj fix - and wackMall's touch points are
   `src/llama-context.cpp`, `common/fit.cpp`, `src/llama-graph.cpp`, the same files.
3. **Unverified: does wackMall's base even have `--spec-type draft-mtp`?** If its base predates
   the MTP work, "wackMall + MTP" is not testable without doing the rebase first - i.e. the entire
   job. **Check this before anything else**; it decides whether section 12 is a test plan or a
   port.
4. **`GGML_CUDA_FORCE_MMQ` is compatible - verified.** It only gates `ggml_cuda_should_use_mmq`
   (`ggml/src/ggml-cuda/mmq.cu:320`) and a compile-time constant (`ggml-cuda.cu:5290`), and the
   mmvq branch in `ggml_cuda_mul_mat_id` returns **before** `should_use_mmq` is consulted. The
   measured +135% prompt-processing build does not push small-batch `mul_mat_id` off mmvq, so it
   composes with M1.
5. **Configuration is env-var-only** (`LLAMA_EXPERT_*`), so it does not appear in `--help` or in
   any log line. Add the whole `LLAMA_EXPERT_*` set to `phase0/config.sh`'s `record_env()` or runs
   will not be reproducible.
6. **TMAX 16 versus prompt processing.** Tiering disengages above 16 tokens, and `-cmoe`
   auto-configures batch 256 - so prompt processing keeps the stock path in both repos. Consistent
   with section 1.4: at `bs >= 32` op-offload already streams whole `_exps` tensors to the GPU.

### 13.7 Revised sequencing

1. **Check 13.6 item 3** - does wackMall's base support MTP. One `grep`. It decides everything
   after this.
2. **Falsify 13.4 item 1** - reproduce a wackMall gain on this box with `LLAMA_EXPERT_RAMPOOL=0`.
   If the gain needs a RAM pool, stop; the approach does not fit 16 GiB and section 7.3's
   fitter-verification path is the whole plan.
3. **Test 13.5 in isolation** - madvise and `MOE_COLD` without tiering. These may be the majority
   of the available win on this hardware and they carry none of the correctness risk.
4. **M2** - the duplicate-id test. Do it whichever way the rest goes.
5. **Then** section 12's M3/M4 on whichever base survives step 1.

Do **not** start M1 or M5 until step 1 resolves. If wackMall rebases onto current master, M5 is
free and M1 is pointless; if it does not, M1 on the PR branch is the only path to MTP + caching
that does not require porting a CUDA kernel change by hand.

---

## 14. Review - PR #26824 (closed) and RFC discussion #24528

Reviewed 2026-08-11. **The RFC discussion matters more than the PR.** It contains the two
measurements this whole plan has been waiting on, and one that should change the risk ordering.

### 14.1 `H(f)` is answered, and the answer is "no skew"

Discussion #24528, on the expert-coverage question section 1.3 made everything contingent on:

> Initial hypothesis that top ~1000 experts saturate value was "probably... measurement
> artifact" from under-warmed pools. Properly warmed caches show "essentially linear" returns
> across coverage percentages up to 30%+.

**Linear returns means `H(f) ~= f`.** The hottest 26% of experts capture about 26% of
activations. Section 1.3's decision rule (`H(0.19) >= 0.40` to justify per-expert placement)
**fails**, and section 11.3's inference that miltos22's 1.7-2.1x implied `H ~= 0.6` was wrong -
those gains come from elsewhere (14.2).

Consequences:
- The heat/decay/hysteresis/dwell machinery is mostly wasted motion. A static assignment would
  land within a few percent of a perfectly-ranked one.
- **But the cache is still worth having**, because its value was never skew - it is moving expert
  matmuls from ~28 GB/s host to ~288-448 GB/s VRAM. That is proportional to coverage, and
  coverage is what the VRAM budget buys.
- Section 7.3 predicted **~20%** decode gain from ~26% coverage. leloch measured **+21.6%** full
  path on Qwen3.6-35B. The byte model in section 7 is sound; the skew hypothesis was not.

Section 3's Phase A is still worth running - but now to *confirm* flatness on Qwen3.6-35B
specifically, not to discover skew.

### 14.2 The fused cold op is worth more than the VRAM cache

leloch's own decomposition on Qwen3.6-35B:

| component | decode gain |
|---|---|
| cache only | **+8%** |
| + fusion | +18% |
| + redirect | +20% |
| full path | +21.6% |

**Fusion (gate/up/SwiGLU in one pass) more than doubles the cache-only figure.** This confirms
section 13.5's guess that `MOE_COLD` matters more here than moving 26% of experts to VRAM, and it
reorders the work: the fused cold path is the high-value, low-risk piece, and it is pure CPU code
with no graph sentinel, no duplicate ids, and no batch ceiling.

It also explains the gap between leloch's +21.6% and miltos22's 1.49-2.19x: different baselines,
and miltos22's numbers bundle fusion **and** mmap pinning **and** the cache.

### 14.3 The measurement that should worry us most: old single GPUs regress

@batot1, GTX 1080 Ti, 11 GB, single GPU:

| config | tok/s |
|---|---|
| cache hard OFF | **19.32** |
| cache ON, 4096 MB | **13.25 (-31%)** |
| cache ON, 32 MB | 18.72 (-3%) |

**A 32 MiB cache still costs 3%.** That is a fixed per-token dispatch overhead independent of
cache size - the tell for an approach whose bookkeeping does not amortise on a slow GPU.

Every positive report in that thread is Ampere-or-newer, and most are multi-GPU 3090s. The one
old single-GPU datapoint is a consistent regression. **The home server's GTX 1660 Ti (Turing
TU116, no tensor cores, 5748 MiB) is architecturally much closer to the 1080 Ti than to a 3090.**

This inverts the testing order. The dev box (RTX 3060, Ampere, cc 8.6) will flatter this feature;
it is the *wrong* box to decide on. **Get a 1660 Ti number early, before investing further.**

### 14.4 MTP + cache composition is measured, and section 12.3's M4 prediction holds

> GLM-5.2 754B + MTP: 13.92 -> 29.35 tok/s (+64.9%)
> "MTP is worth +28% without cache but +51% with it"

Section 12.3 M4 argued that caching lowers the marginal cost of a larger expert union, so `k_opt`
should rise and MTP should be worth *more* with a cache than without. **That is now measured on
someone else's hardware: +28% -> +51%.** The mechanism is confirmed; the magnitudes still need
re-measuring here, since the user has confirmed 27B MTP results do not transfer to the 35B.

This strengthens the case for the M1 guard work rather than weakening it.

### 14.5 A footgun that our M1 guard shares

Discussion #24528, operational issues list:

> **Silent bypass above max batch:** If batch size exceeds cache's maximum, operations silently
> bypass cache with no diagnostic.

**Our `n_tokens_max` guard at `src/llama-expert-tier.cpp:79` is exactly this bug.** It returns
`nullptr` with no log, so a config that silently loses the tier looks identical to one that keeps
it. Cheapest possible fix: a one-shot `LLAMA_LOG_WARN` naming `n_tokens` and the cap. Should land
before any benchmarking, or half the results will be unattributable.

Two more from the same list worth pre-empting:
- **Default admission too aggressive** - "full-acceptance content regressed -30%... ADMIT_AFTER=64
  removed the cliff entirely." Do not trust `--expert-hyst 1.3` / `--expert-dwell 0` defaults;
  sweep them.
- **Budget silently capped** - requesting 20480 MiB quietly allocated 12563 MiB. Always read back
  the `GPU hot store allocated:` line rather than trusting the request.

### 14.6 A speculative-decoding failure mode to watch for in Test 3

@noonghunna: "DSpark (GPU) + cache -> 33.4 tok/s - **cache never engages**" versus "DSpark on CPU
+ cache -> 46.8 tok/s - composes." Diagnosis offered: "device/session binding doesn't survive the
reordered device list."

MTP creates a second `llama_context` (`ctx_dft`). If #26563's hot store binds to a device index or
a context identity that the draft context perturbs, **the cache can silently fail to engage with
MTP on - which would look exactly like the M1 guard not working.** Test 3's
`GGML_SCHED_DEBUG=2 | grep -c MUL_MAT_ID_COLD` distinguishes the two, because it observes the
graph directly rather than inferring from throughput. Keep that distinction in mind when reading
the result.

### 14.7 Prompt processing regresses, repeatedly, across implementations

Independent reports: -14% (@xashr), -19% (@noonghunna), -7% (@giveen) on leloch's; and on
miltos22's #26824, @kabhinara measured a **3-4x** prompt-processing regression, with Green-Sky
finding "prompt processing is ALWAYS done on cpu (300 -> 2 tps)" on BTL-4-Compact.

The 27B work already established that prompt processing is a first-class metric on this hardware
(the `GGML_CUDA_FORCE_MMQ` build was worth +135% pp). **Every A/B in sections 12-13 must report pp
alongside tg**, or a large pp regression will hide behind a modest tg gain.

Some of the reported pp loss is confounded with `-ub` sizing changes made to free VRAM for the
cache - control for that explicitly.

### 14.8 leloch's architecture is a genuine alternative, and it avoids our whole M1 problem

#24528 inverts #26563's structure: **keep `MUL_MAT_ID` on the CPU** and have the CPU kernel
dispatch cached rows to a persistent VRAM cache, computing misses locally. Versus #26563's
"hot GPU `mul_mat_id` + cold CPU op, summed".

Consequence: **no graph-level sentinel, so no duplicate expert ids, so no `mm_ids_helper` hazard
and no MMQ/MMF exposure.** Sections 12 and 13 - the entire M1/M5 line of work - exist only because
of #26563's sentinel. leloch's design does not have that problem at all.

Costs, in leloch's own accounting: it "hooks the CPU mul_mat_id hot path" (named as the largest
structural cost), carries heuristics maintainers did not choose, is hard to test in CI, and is
CUDA-only. It also still has a max-batch bypass (14.5), just for a different reason.

Not a recommendation to switch - we have #26563 merged and working. But if M1/M5 turn into a
sustained fight with the CUDA id-compaction path, this is the escape hatch, and it has a measured
MTP composition story (14.4) that #26563 does not.

### 14.9 Status of #26824 and what is worth taking from it

**Closed.** IMbackK: *"This type of pr is not acceptable as i violates multiple rules, for example
the one change per pr rule, the no changes in multiple back ends rule, vibecodeing etc."* Plus
automated flags for "multiple backend changes in one PR" and "large PR" (71 commits). Green-Sky
also objected to the title length. Constructive path offered by voidpush and Tha14: an issue
describing the architecture first, then small sequential PRs.

Read against `AGENTS.md`, this is a clean demonstration of what that file is warning about, and it
is a strong argument for keeping our branch a *private fork* rather than trying to upstream
anything except the narrow, self-contained pieces (section 13.3's M2 duplicate-id test remains the
best candidate).

Three features in #26824 are worth porting later, in this order for a 16 GiB box:

1. **`--expert-move-mode` (copy vs move).** Move mode frees the host copy of a promoted expert.
   This directly addresses section 11.4's duplication concern, which is the single biggest memory
   risk on the 16 GiB server. Copy mode heats up faster; move mode preserves RAM.
2. **`--expert-pin N`** - `MADV_WILLNEED` pinning of the top N% of *cold* experts' mmap pages,
   with a reported optimum of **30-60%**. This is section 3's Phase C, implemented, with a tuned
   range already established.
3. **Fused cold op** - per 14.2, the largest single component of the measured gain.

Everything else in #26824 (device-priority stores, throttled PCIe queue, multi-GPU handshakes) is
multi-GPU machinery with no value on a single-card box.

---

## 15. MEASURED - dev box, 2026-08-11, `UD-Q3_K_M` + PR #26563 + M1

Branch `expert-cache-vram` (`714299277`). Harness: `phase0/devbox-expert-ab.sh`,
logs in `phase0/results/devbox/`.

**Box:** RTX 3060 Laptop, 6144 MiB VRAM (**only 5066 MiB free** - 1077 MiB held by the
display), cc 8.6 Ampere, **64 GB RAM**, Windows, MSVC, CUDA 13.3.
**Model:** `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q3_K_M`, 15.93 GiB, 41 blocks
(40 trunk + MTP in-checkpoint), 256 experts, top-8, `n_ff_exp` 512, `n_ctx_train` 262144.

**Expert tensor types are i-quants, not K-quants:** IQ3_XXS x78, IQ4_XS x39, Q6_K x3,
Q4_K x1, Q3_K x2. On the `turing_plus` mmvq table (both this box and the 1660 Ti) that is
IQ3_XXS=7, IQ4_XS=8, Q3_K=5 - so `n_tokens_max = 4` is safe with headroom. Section 0.4's
worry about i-quant CPU matmul throughput applies to this quant and is unmeasured.

### 15.1 The tier engages, and only because of M1

```
llama_expert_tier_build: expert tier engaged: n_tokens=2, blk.0.ffn_up_exps.weight
llama_expert_tier_build: expert tier bypassed: n_tokens=9 (max 4), blk.0.ffn_up_exps.weight is iq3_xxs
```

**Decode runs at `n_tokens = 2`, not 1.** PR #26563's original `if (cur->ne[2] > 1)` guard
would therefore have bypassed the tier **on every single decode step** - the feature as
submitted does nothing on this configuration. Section 12's M1 change is what makes it work.
Prompt processing (`n_tokens = 9`) correctly bypasses, as designed.

> **Retracted:** an earlier run of this pass concluded "zero `MUL_MAT_ID_COLD` nodes, the tier
> never engages". That run was killed by a 10-minute timeout and its log was truncated before
> decode began. `GGML_SCHED_DEBUG=2` at `-lv 5` produces ~64 MB of output per few tokens and is
> not a practical instrument here; the one-shot logging committed in `714299277` is.

### 15.2 Throughput - warm-up dominates, and short benchmarks lie

| run | S | coverage | pp t/s | tg t/s |
|---|---|---|---|---|
| 128 tok, `-ehs 0` | - | - | 84.54 | 18.39 |
| 128 tok, `-ehs -1` | 23 | 9% | 80.59 | 18.82 |
| 128 tok, `-ehs -1 -fitt 256` | 37 | 14.5% | 80.51 | 18.32 |
| **483 tok, `-ehs 0`** | - | - | 19.10 | **15.81** |
| **483 tok, `-ehs -1 -fitt 256`** | 41 | 16% | 17.11 | **17.55 (+11%)** |

At 128 tokens the hot set has not converged and the gain is zero. At 483 tokens it is
**+11% tg / -10% pp**. **Any benchmark under ~300 generated tokens understates this feature**
and will produce a false negative. (The two 483-token rows also ran at `-c 512`, so their
absolute pp/tg are not comparable to the 128-token rows - only within-pair deltas are.)

### 15.3 Hit rate is strongly super-linear - section 14.1 was wrong for this model

With S=37-41 (**14-16% of experts resident**), the measured per-decode hit rate is **30-76%,
mean ~55%**, once warmed. Early in a run it is ~24%.

That is roughly **3.5x uniform**, and it directly contradicts discussion #24528's "essentially
linear returns" finding quoted in section 14.1. Either the effect is model-dependent, or that
thread's pools were still under-warmed (the same warm-up artifact 15.2 shows here, which is
exactly the artifact that thread claimed to have corrected for).

**Section 1.3's decision rule is therefore back in play for Qwen3.6-35B-A3B specifically**, and
section 14.1 should not be treated as settled. The 55% figure is this plan's first direct `H(f)`
measurement and it clears the top bar (`H >= 0.40`).

Per-layer heat after 128 tokens shows the depth dependence section 12.3 predicted:
layer 0 has **225 of 256 experts warm**, layer 9 only **126** - early layers flat, deep layers
concentrated. Spending VRAM preferentially on deep layers is untested and looks promising.

### 15.4 `-fitt` is worth ~40% of the hot store

Default 1024 MiB fit margin -> **S=23**. `-fitt 256` -> **S=37-41**. The fit trace shows why:
free VRAM is 5066 MiB, target = 5066 - 1024 = 4042 MiB, and `S` is derived as
`n_expert * moe_on_gpu / total_moe_bytes - 1 = 256 * 1283/13578 - 1 = 23`. The final allocation
still leaves ~1035 MiB unused, so **the autofit under-allocates even after the margin is
accounted for** - there is more headroom to reclaim than `-fitt` alone gets.

### 15.5 Two undocumented / silent behaviours

- **`LLAMA_EXPERT_HITRATE`** (env var, `src/llama-context.cpp:1523`) is the only way to get the
  hit-rate counter. Not in `--help`, not in the READMEs. It is the single most useful number the
  feature produces.
- **The heatmap freezes under speculation.** `src/llama-context.cpp:1514`:
  ```cpp
  if (expert_heatmap && (ubatch.n_tokens == 1 || !expert_hotstore || !expert_hotstore->is_filled)) {
  ```
  Once the store is filled, heat only updates at `n_tokens == 1`. An MTP verify batch is >= 2, so
  **with MTP on the hot set is frozen at whatever the first fill chose, permanently.** This is
  section 12.3's M3, and it is worse than predicted: not mis-weighted, simply dead.

### 15.6 Auto context growth does not work outside the server

`-c 512 --ctx-max 4096 -n 800` generated exactly 483 tokens and stopped at 511 = the initial
`n_ctx`. No `growing n_ctx:` line ever appeared.

- The core mechanism is present and looks correct: `llama_context::set_n_ctx()`
  (`src/llama-context.cpp:891`) and the auto-grow hook in `decode()` (`:1967-1980`), gated on
  `n_seq_max == 1 && n_ctx_max > n_ctx`, firing on a memory prepare failure.
- **`llama_set_n_ctx` is called from exactly one place in the tree:
  `tools/server/server-context.cpp` (3 sites).**
- `llama-completion` reads `n_ctx` once at startup and enforces it itself at
  `tools/completion/completion.cpp:609` (`n_past + embd.size() >= n_ctx` -> context-shift or
  stop), so `decode()` never fails and the growth hook never runs.

This is precisely the failure mode `NOTES.md` Part 1 item 9 predicted for the server
("`slot.n_ctx` computed once at startup... would grow the underlying cache but leave the server
still enforcing the old limit"), applying verbatim to the completion tool. Tools that guard the
context themselves silently opt out of growth.

### 15.7 Server matrix, 900 generated tokens (`phase0/devbox-server-ab.sh`)

The server is the only harness that registers `--spec-type` **and** drives context growth, so all
of the below runs there. `-c 4096`, `-np 1`, `n_predict 900`, temp 0.

| case | S | pp t/s | tg t/s | draft accept | mean hit rate |
|---|---|---|---|---|---|
| `-ehs 0` | - | 13.13 | **15.37** | - | - |
| `-ehs -1 -fitt 256` | 43 | 12.72 | **16.45** / 17.81 | - | 56.4% (899 samples) |
| `--spec-type draft-mtp` | - | 13.96 | **12.62** | 0.507 | - |
| both, before the 15.8 fix | 33 | 12.56 | **13.67** | 0.591 | 8.1% (**1 sample**) |
| **both, after the 15.8 fix** | 33 | 11.38 | **17.30** | 0.605 | 35.9% (322 samples) |

**MTP alone looks like a -18% regression here (12.62 vs 15.37) - but that is an artifact of the
default `n_max = 3`. See 15.9: at `n_max 1` MTP is +17.4%.** The mechanism is still sections
7.4/10.1's: the verify batch pays for the *union* of the drafted tokens' experts, so the cost
grows with draft length while acceptance falls.

**With the cache, MTP is roughly break-even** (17.30 vs 16.45-17.81) instead of -18%. That is the
section 12.3 M4 / 14.4 mechanism showing up locally: the cache lowers the marginal cost of a
larger expert union, so speculation stops being a net loss.

**Run-to-run variance is significant** - the identical `cache` case measured 16.45 and 17.81 t/s
on two runs (~8%). No conclusion below ~10% should be drawn from single runs. Everything in this
section is n=1 and should be repeated.

### 15.8 Three separate freezes broke MTP + cache, all now fixed

Committed as `a02f14d05` and `f95c5a391`.

1. **The MTP context built its own hot store and OOM'd at load.**
   `common/speculative.cpp:2373` sets `ctx_type = LLAMA_CONTEXT_TYPE_MTP` but the draft context
   still inherits `expert_hot_s`. The store is sized from `hparams.n_layer()` (40 trunk layers)
   although an MTP context only runs the nextn block, so it asked for 1803 MiB with 528 MiB free:
   ```
   allocate: not enough memory to allocate the GPU hot store of 33 slots (1803 MiB needed, 528 MiB free on CUDA0)
   common_speculative_init_result: failed to create MTP context
   ```
   Worse, `g_table` in `src/llama-expert-tier.cpp` is a **file-scope global keyed by model tensor
   pointer**, and both contexts share the same model tensors - so the draft store would have
   overwritten the target's registrations, and either destructor's `llama_expert_tier_clear()`
   would wipe both. **This global is a latent multi-context bug independent of MTP** and is worth
   reporting upstream.
2. **Heat froze.** `src/llama-context.cpp` tracked heat only at `ubatch.n_tokens == 1`; a verify
   batch is `1 + n_draft`, so after the initial fill the hot set never changed again.
3. **Resync froze.** `maybe_resync(..., ubatch.n_tokens > 1)` returns immediately on the
   multi-slot flag (`src/llama-expert-hotstore.cpp:302-306`), so even with heat updating, the
   store contents were still pinned to the first fill.

All three now key off `LLAMA_EXPERT_TIER_MAX_TOKENS` (4) - the batches the tier actually serves -
rather than `== 1`. Net effect: hit rate **8.1% (1 sample) -> 35.9% (322 samples)**, tg
**13.67 -> 17.30**.

Still open from section 12.3's M3: heat is credited to *rejected* draft tokens too, since the
update runs before acceptance is known. That biases the hot set toward what the MTP head predicts
rather than what the model emits.

### 15.9 Draft length dominates, and `n_max 1` is optimal - 15.7's "MTP regresses" was wrong

The user reported `--spec-draft-n-max 1` as optimal for 35B-A3B. **Confirmed, and it reverses
section 15.7's conclusion**: the default is `n_max = 3` (`common/common.h:325`), so every MTP run
in 15.7 was at the worst setting on the curve.

`UD-Q3_K_M`, 900 tokens, base = 15.37 t/s:

| `n_max` | verify batch | MTP alone | MTP + cache | accept | hit rate |
|---|---|---|---|---|---|
| **1** | 2 | **18.05 (+17.4%)** | **20.11 (+30.8%)** | 0.81 / 0.83 | 44.7% |
| 2 | 3 | 15.40 (+0.2%) | 19.11 (+24.3%) | 0.69 / 0.68 | 41.1% |
| 3 (default) | 4 | 12.62 (-17.9%) | 17.30 (+12.6%) | 0.51 / 0.61 | 35.9% |

Monotonic in every column. This is sections 7.4 / 10.1's union argument measured directly: each
extra drafted token adds distinct experts to the verify batch, acceptance falls, and the hit rate
falls with it (44.7 -> 41.1 -> 35.9%) because a wider union spills past the resident set.

**Two operational consequences:**
- **`n_max` must stay <= 3.** The verify batch is `1 + n_max`, and the tier bypasses above
  `LLAMA_EXPERT_TIER_MAX_TOKENS` (4), so `n_max >= 4` silently turns the expert cache off during
  verification - the exact silent-bypass footgun section 14.5 warned about, now reachable through
  an ordinary user-facing flag. The one-shot log from `714299277` is what makes it visible.
- The default `n_max = 3` is the worst usable value here. Anyone benchmarking MTP on this
  architecture without setting `n_max 1` will measure a regression.

### 15.10 `Q4_K_M` beats `UD-Q3_K_M` outright - i-quant experts are the problem

Same box, same harness, `llmfan46/...-Native-MTP-Preserved-Q4_K_M` (20.28 GiB, experts **Q4_K
x102 + Q6_K x21**) versus `unsloth UD-Q3_K_M` (15.93 GiB, experts **IQ3_XXS x78 + IQ4_XS x39**):

| case (`n_max 1`) | `UD-Q3_K_M` tg | `Q4_K_M` tg |
|---|---|---|
| base | 15.37 | **20.38 (+33%)** |
| cache | 16.45 / 17.81 | 20.16 (-1%, noise) |
| MTP alone | 18.05 | 23.78 (+16.7%) |
| **MTP + cache** | 20.11 | **23.95 (+17.5%)** |

**The 27% larger model is 33% faster at baseline.** Section 0.4 flagged exactly this risk for
i-quants ("materially worse CPU matmul throughput than K-quants and not repack-eligible") when
weighing `UD-IQ4_XS`; it is now measured, and it is decisive. With ~75% of expert matmuls running
on the CPU, i-quant expert tensors cost far more in decode time than their smaller footprint
saves.

**And the expert cache's value tracks CPU expert speed inversely:**
- on `UD-Q3_K_M` (slow i-quant CPU path) the cache is worth +7-16% alone and +11% on top of MTP
- on `Q4_K_M` (fast, repack-eligible K-quant CPU path) it is worth **nothing** (20.16 vs 20.38
  base; 23.95 vs 23.78 with MTP) despite a healthy 52.1% hit rate over 901 samples

That is a coherent mechanism, not noise: the cache moves expert matmuls from CPU to GPU, so its
benefit is proportional to how slow the CPU path was to begin with. **A high hit rate does not
imply a throughput win.**

**Caveat that decides nothing here:** `Q4_K_M` is 20.28 GiB and this box has 64 GB. The 16 GiB
server cannot hold it (section 0.4) - which is why `UD-Q3_K_M` was chosen in the first place. So
the dev box's answer ("`Q4_K_M` + MTP `n_max 1`") is not transferable, and the server's real
choice is between a K-quant that pages from SSD and an i-quant that fits but computes slowly.
**That comparison is the single most valuable thing left to measure, and only the 1660 Ti box can
measure it.** A K-quant near 16 GiB (`UD-Q3_K_XL`, 16.8 GB) is the obvious candidate to test
against `UD-Q3_K_M`.

### 15.11 Every unsloth quant at or below Q3 uses i-quant experts

`UD-Q3_K_XL` was the obvious candidate from 15.10 ("a K-quant near 16 GiB"). **It is not one.**
Scanned with `phase0/gguf-remote-types.py`, which range-requests only the GGUF header, so a quant
can be checked without downloading it. Validated against the local `UD-Q3_K_M` file first - the
remote scan reproduces its measured types exactly.

| quant | size | expert tensor types | i-quant free |
|---|---|---|---|
| UD-Q2_K_XL | 12.3 GB | IQ2_XS x78, IQ3_XXS x39, IQ4_XS x3, Q2_K x2, Q3_K x1 | no |
| UD-Q3_K_M | 16.6 GB | IQ3_XXS x78, IQ4_XS x39, Q6_K x3, Q3_K x2, Q4_K x1 | no |
| **UD-Q3_K_XL** | 16.8 GB | **IQ3_XXS x78, IQ4_XS x39, Q6_K x3, Q3_K x2, Q4_K x1** | **no** |
| **UD-Q4_K_S** | 20.9 GB | **Q4_K x120, Q6_K x3** | **yes** |
| MXFP4_MOE | 21.7 GB | MXFP4 x80, Q5_K x40, Q6_K x3 | yes (but MXFP4) |
| UD-Q4_K_M | 22.1 GB | Q4_K x102, Q6_K x21 | yes |

`UD-Q3_K_XL`'s expert tensors are **byte-for-byte the same mix as `UD-Q3_K_M`**. The entire ~200 MB
difference between the two files is one dense tensor moving Q6_K -> Q8_0 (Q8_0 x259 vs x258). The
"XL" upgrade never touches the experts, so it cannot fix the slow CPU expert path.

**Consequence: the smallest i-quant-free option is `UD-Q4_K_S` at 20.9 GB = 19.5 GiB**, which is
at or just over the server's ~19.4 GiB budget (section 0.4). There is no comfortable K-quant
choice; the server must either accept i-quant experts that fit, or a K-quant that pages from SSD.

**Warning for the constrained box:** `-ehs` (either mode) forces **all** experts to system memory
(`common/common.cpp:1242-1265`) and holds VRAM copies on top. On a box already at its memory
limit that pushes host demand *up*, which is the wrong direction - the fitter's default placement
puts several GiB of experts on the GPU instead. Section 11.4 argues mmap should make the
duplication self-correcting (hot experts' host pages stop being read and get evicted), but that
is unverified and 64 GB of dev-box RAM cannot test it. **Measure `-ehs 0` before `-ehs -1` on the
server, and judge on `majflt`, not throughput alone.**

### 15.12 Auto context growth - full tool survey

| tool | growth | why |
|---|---|---|
| `llama-server` | **works** | the only caller of `llama_set_n_ctx` (3 sites, `tools/server/server-context.cpp`) |
| `llama-cli` | **works** (inherited) | spawns `llama_server` on a thread and talks HTTP (`tools/cli/cli-server.h`, links `llama-server-impl`) |
| `llama-completion` | **broken** | caches `n_ctx` at `completion.cpp:205`, enforces it itself at `:609`, so `decode()` never fails and the auto-grow hook never fires |
| `llama-perplexity`, `batched-bench`, `imatrix` | n/a | fixed-window by design; they deliberately fill exactly `n_ctx` |

Verified on the server:
```
set_n_ctx: growing n_ctx: 512 -> 1024 (n_ctx_seq: 512 -> 1024)
```

> **CORRECTION (2026-08-11, found via the home server's llama-cli log):** those early verifications
> only exercised `maybe_grow_for_request()`, which needs an explicit `n_predict` to size against.
> The **mid-generation** hook (`maybe_grow_mid_generation()`) had an off-by-one - caller triggers
> at `n_tokens + 1 >= n_ctx`, hook bailed at `n_needed <= n_ctx` - so with `n_predict = -1` (what
> llama-cli's chat sends) nothing ever grew and generation truncated at the initial `-c`. Fixed in
> `ca7de7932`; a no-`n_predict` request now steps 512 -> 768 -> ... -> ctx_max by 1.5x. Repro and
> regression case: request with no `n_predict` against `-c 512 --ctx-max 4096`.
with `-c 512 --ctx-max 4096`, 900 predicted tokens. Growth fires from the `decode()` hook
(`src/llama-context.cpp:1967-1980`) on memory-prepare failure, gated `n_seq_max == 1`.

`llama-completion` is a genuine gap: any tool that guards the context itself silently opts out of
growth. Fixing it means re-reading `llama_n_ctx(ctx)` after each decode instead of caching it at
startup, exactly the pattern `NOTES.md` Part 1 item 9 prescribes for the server.

### 15.13 CLOSED: `--ctx-max` composes with `-ehs` on this architecture

Sections 11.5 / 13.6 flagged "hot store sized at load, never resized; growth then eats VRAM it
already claimed" as a blocking hazard and told the user not to combine the two. **Tested, and the
hazard does not bite on this model.**

`-c 512 --ctx-max 32768 -ehs -1 -fitt 256`, hot store **S=45, 2439 MiB**:

```
=== Expert hotstore sizing (S=45) ===
  GPU hot store allocated: CUDA0, 2558394368 bytes (2439 MiB) for 45+1 slots
set_n_ctx: growing n_ctx: 512 -> 1536 (n_ctx_seq: 512 -> 1536)
set_n_ctx: growing n_ctx: 1536 -> 32768 (n_ctx_seq: 1536 -> 32768)
resize:      CUDA0 KV buffer size =    30.00 MiB      (at 1536)
resize:      CUDA0 KV buffer size =   640.00 MiB      (at 32768)
```

Then real work at the grown size: a **25841-token prompt at 231.63 t/s**, no failure, no
`failed to reserve graph`. A 33301-token prompt was cleanly rejected against the ceiling with a
400 and an explicit message, which is the correct behaviour.

**Why it survives: KV on this architecture is unusually cheap.** Section 0.3's 20 KiB/token is
confirmed to the megabyte - 32768 x 20 KiB = **640.00 MiB** measured. The mechanism in 11.5 is
real, but its magnitude here is one quarter of the hot store, and the fitter's slack absorbed it.

**The boundary, stated so it can be checked elsewhere:** with `--ctx-max`, the fit sizes the hot
store against the *initial* `n_ctx`, so it over-commits VRAM by roughly the KV delta between the
initial and the maximum context. Here that is `n_ctx_max x 20 KiB` = 640 MiB at 32768, and
`-fitt 256` left enough slack to cover it. **Rule: keep `-fitt` at or above the KV cost of
`--ctx-max`.** On an architecture with expensive KV (many attention layers, weak GQA) the same
mechanism would still bite - this result does not generalise beyond Qwen3.5/3.6-style hybrids.

Also observed, and relevant to the 16 GiB box: the server creates **context checkpoints of
62.8 MiB each, up to 32** (`--ctx-checkpoints`). Only 2 were live here, but the ceiling is ~2 GiB
of host state. Worth lowering on a memory-constrained server.

### 15.14 Open

- **No repeats** - every number in 15.2/15.7 is a single run, and the same config varied ~8%
  between two runs. Repeat before trusting any delta under ~10%.
- `llama-completion` context growth is unfixed (15.9).
- Rejected draft tokens still pollute the heatmap (15.8).
- The autofit still leaves ~1035 MiB of VRAM unused beyond the margin (15.4).
- 64 GB of RAM here means host-bandwidth pressure does not reproduce; per section 14.3 the
  1660 Ti is the box that decides, and MTP's -18% may not transfer either.

---

## 8. Prior art

- [ggml-org/llama.cpp#20757](https://github.com/ggml-org/llama.cpp/issues/20757) - two-tier
  GPU+RAM expert cache, open, no maintainer response, no implementation. Its PCIe framing is
  wrong for batch-size-1 decode (section 1.4); its skew claim is unsourced (section 1.3).
- Eliseev & Mazur, *Fast Inference of Mixture-of-Experts Language Models with Offloading* -
  LRU expert cache plus speculative expert prefetch, the canonical version of Phase E.
- [FloE: On-the-Fly MoE Inference on Memory-constrained GPU](https://arxiv.org/pdf/2505.05950)
- [PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices](https://arxiv.org/pdf/2504.03664)
- [Performant local MoE CPU inference with GPU acceleration in llama.cpp](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) -
  practical `-ot` tuning, the Phase B item 5 baseline.
- MoE-Infinity (LFU eviction) and AdapMoE (LRU) for the eviction-policy design space.
