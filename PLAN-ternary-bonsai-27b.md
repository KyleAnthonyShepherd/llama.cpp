# PLAN - Ternary-Bonsai-2-27B on the laptop (RTX 3060 Laptop 6 GB, i9-11900H, 64 GB)

Goal: fastest decode (and reasonable prefill) for `prism-ml/Ternary-Bonsai-2-27B-gguf` on this
machine, from the `Ternary` branch. Everything below is sized from measurements taken on this
laptop on 2026-09-17 (plugged in, cooling pad on), unless marked *estimate*.

---

## 1. What the model is

From the PTQ1_0 GGUF header (read remotely, `phase0/gguf-remote-types.py`-style range fetch):

- `qwen35` hybrid, 64 blocks: 48 GDN (linear-attention) blocks + 16 full-attention blocks
  (`full_attention_interval = 4`), n_embd 5120, FFN 17408, 24 q heads / 4 kv heads x 256.
- Vocab 248320. Hadamard weight fold on 401 matmuls (block 1024, explicit signs), so every
  folded matmul pays an FWHT on its input activation (shared between weights with one input).
- 5.53 GiB total. Every matmul weight is PTQ1_0 (ternary g128, 1.75 bpw); `ssm_alpha/beta` BF16.
- Per block ~81 MiB (GDN) / ~78 MiB (attention). FFN gate+up+down is ~56 MiB of that.
- `token_embd` (265 MiB) stays on the CPU by default and costs one row per token.
  `output.weight` (265 MiB) is a full 1271 M-weight matmul per token.
- No MTP / nextn tensors in the GGUF.

Memory per sequence (*estimate*, verify in the load log):

| item | size |
|---|---|
| GPU weights (all but `token_embd`) | ~5395 MiB |
| recurrent state (48 GDN layers, f32) | ~150 MiB (x4 with spec rollback slots) |
| KV, 16 attn layers, per 1k ctx | 64 MiB f16 / 34 MiB q8_0 |
| CUDA compute buffer | 180-510 MiB (depends on `n_outputs_max`, `-ub`) |

Usable VRAM is ~5.6 GiB at best (6144 - 211 MiB desktop apps - CUDA context). **So the full
model does not fit with any context, and the question is where the last ~300-600 MiB go.**

## 2. Measurements (one 4096 x 14336 matmul, `test-backend-ops perf`)

GPU (CUDA0, sm_86):

| type | n=1 | n=2 | n=4 | n=8 | n=512 |
|---|---|---|---|---|---|
| q4_0 | 136 us (244 GB/s) | 143 | 212 | 300 | 23.9 TFLOPS |
| pq2_0 | 87 us (180 GB/s) | 89 | 156 | 250 | 26.7 TFLOPS |
| **ptq1_0** | **70 us (183 GB/s)** | **130** | **224** | **420** | **13.3 TFLOPS** |

CPU (8 threads, no repack - test-backend-ops uses a plain CPU buffer):

| type | native build (no VNNI) | `-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON` |
|---|---|---|
| q4_0 | 770 us | 662 us (~40 GB/s, bandwidth bound) |
| q8_0 | 1361 us | 1292 us |
| pq2_0 | 4803 us | **1621 us (3x)** |
| **ptq1_0** | **10826 us** | **9830 us (generic scalar - no x86 SIMD kernel exists)** |

Cost per million weights, decode (n=1):

| where / format | us per M weights | per 81 MiB block |
|---|---|---|
| GPU ptq1_0 | 1.2 | 0.46 ms |
| CPU q4_0 | 11.3 | 4.3 ms |
| CPU pq2_0 (VNNI) | 27.6 | 10.6 ms |
| CPU ptq1_0 | 167 | **64 ms** |

Findings:

1. **A single PTQ1_0 layer on the CPU costs more than the whole GPU part of the model.**
   Full-GPU matmul time is ~31 ms/token (*estimate*: 25.9 G weights x 1.2 us). One CPU
   PTQ1_0 block adds 64 ms, and `output.weight` on the CPU would add 212 ms. Stock `-ngl` /
   `--fit` spill of this file is therefore a trap.
2. MSVC `GGML_NATIVE` detects AVX-512 but never sets VNNI/VBMI (`ggml-cpu/cmake/FindSIMD.cmake`,
   `GGML_AVX512_VNNI:BOOL=OFF` in the cache), so the PQ2_0 VNNI kernels are compiled out.
3. On the GPU, PTQ1_0 decodes at 183 GB/s where q4_0 reaches 244 GB/s: ~25% headroom.
4. PTQ1_0 does not amortize over columns (n=2 costs 1.9x n=1, PQ2_0 costs 1.03x). **Any
   speculative decoding is a loss on PTQ1_0 until this is fixed** - with n=2 at 1.86x, a
   single extra draft token needs >86% acceptance to break even.
5. PTQ1_0 prefill is half the speed of PQ2_0 (13 vs 27 TFLOPS).
6. The desktop holds 211 MiB of VRAM on the dGPU (Signal, Edge WebView, Epic, Claude, ...).
   The driver has logged 146 s of SW thermal slowdown historically.

## 3. The split, in value order

Every MiB freed from the GPU costs decode time somewhere. Cheapest first:

1. **Free VRAM that does nothing** (zero cost): move desktop apps off the dGPU (Windows
   Graphics settings -> Power saving per app, or hybrid/MSHybrid mode), `-fitt` down from
   the 1024 MiB default margin, no `--mmproj`. ~200 + ~700 MiB.
2. **Shrink the compute buffer**: small `n_outputs_max` (the server already uses 4, the
   27B plan measured 513 -> 184 MiB) and `-ub 256`. Measure what `-ub` costs prefill.
3. **KV**: `-ctk q8_0 -ctv q8_0`, size `-c` to the job. For long contexts put the KV on the
   CPU (`--no-kv-offload`): its cost scales with the *filled* context, not the allocation,
   and on 16 layers it is ~2x cheaper per MiB freed than weights even when full.
4. **Only then weights**, as a contiguous run of the *first* blocks (fewest graph splits,
   and it is what the fitter already does), **never as PTQ1_0 on the CPU**:
   - near term: store those blocks as **Q4_0 with d = PTQ1_0 scale, q = trit + 8**. That is
     exact (every ternary block of 128 is 4 Q4_0 blocks of 32 with the same scale). Note:
     `llama-quantize` to Q4_0 is *not* exact - its signed-max scale maps +1 to 7/8 on half
     the blocks - so this needs a small converter (C, using `dequantize_row_ptq1_0`).
     4.3 ms per block instead of 64 ms.
   - prefer blocks whose weights are only FFN: move `ffn_*` of block 0..k first, keep attn/GDN
     on the GPU, only if the extra 2 splits per block measure cheaper than whole blocks.

*Estimate* for a 300 MiB shortfall after steps 1-3: ~4 blocks as Q4_0 on the CPU = +17 ms,
so ~48 ms/token (~21 t/s) versus ~31 ms (~32 t/s) if it all fit. Stock behaviour (4 PTQ1_0
blocks on the CPU) would be ~290 ms/token (~3.4 t/s).

## 4. Work items

| # | item | effort | expected gain |
|---|---|---|---|
| A | Build with `-DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON` (keep `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`) | none | 3x on PQ2_0 CPU, +15% q4_0 |
| B | Measure the real VRAM ledger and pick `-fitt`, `-ub`, `-c`, KV type (Phase 0) | none | decides everything below |
| C | Exact PTQ1_0 -> Q4_0 converter for the CPU-resident blocks (hybrid GGUF) | small | ~15x on the CPU share |
| D | Profile the PTQ1_0 mmvq kernel on sm_86 with Nsight Compute (installed) and tune (rows per block, trit-decode LUT placement) | medium | up to ~25% decode |
| E | PTQ1_0 multi-column mmvq that decodes trits once for n=2..8 | medium | prerequisite for any spec decoding |
| F | Prefill: try `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` caps (falls back from the PTQ1_0 MMQ path) | none | up to 2x pp if the fallback is faster |
| G | AVX-512 VBMI kernel for PTQ1_0 or PQ2_0 near memory bandwidth (~3 ms/block) | medium-high | ~1.4x on the CPU share over C |
| H | Per-layer KV placement (KV of some attn layers on the CPU, not all-or-nothing) | medium | long-context headroom |
| I | Overlap the CPU share with GPU work (row-split FFN, GPU rows async while the CPU computes its rows - the expert-tier hot/cold graph is the template). First check whether `ggml_backend_sched` already overlaps a CPU split with queued CUDA work | high / research | hides the CPU share entirely |

Not in scope: Vulkan (Ternary keeps upstream Vulkan, no PTQ1_0 port). MTP: the GGUF ships no
nextn head. Spec decoding in general waits on E.

## 5. Phase 0 results (2026-09-18, AVX-512 VNNI build, PQ2_0 download running in the background)

Ledger (`-v` load log): 5130 MiB free at load (desktop apps held ~409 MiB). Everything on the
GPU would be 6190 MiB. Recurrent state 149.6 MiB, KV q8_0 @ 4096 = 136 MiB. The CUDA compute
buffer is dominated by logits for a whole ubatch (vocab 248320): 623 MiB @ `-ub 512`,
133 MiB @ 128, 38-72 MiB @ 32.

Decode, `llama-completion`, greedy, all at `-c 4096 -ctk q8_0 -ctv q8_0 -fa on -t 8`:

| config | CPU blocks | tg t/s |
|---|---|---|
| stock file, fitter, `-ub 512` | 18 PTQ1_0 | 0.79 |
| stock file, fitter, `-ub 32` | 12 PTQ1_0 | 1.18 |
| hybrid `cpu12q4` (blocks 0-11 exact Q4_0), `-ngl 53` | 12 Q4_0 | 7.94 |
| same, `-lm none` (pinned host buffer) | 12 Q4_0 | 8.39 |
| same, `--no-op-offload` (38 -> 2 graph splits) | 12 Q4_0 | 8.85 |
| hybrid `cpu16q4`, `-ngl 49`, no op offload | 16 Q4_0 | 7.30 |

Fit of the last rows: **6.0 ms per CPU block** (~204 MB of Q4_0, close to RAM bandwidth) and
**~41 ms of fixed GPU time** - so the all-GPU ceiling is ~24 t/s, not 32.

`llama-bench` on `cpu12q4`, `-ngl 53 -lm none`:

| ub | op offload | pp512 | tg32 |
|---|---|---|---|
| 32 | on | 75.1 | 8.31 |
| 128 | on | **169.1** | 8.22 |
| 32 | off | 23.7 | 8.07 |
| 128 | off | 25.0 | 8.64 |

Op offload is worth ~7x on prefill, so it stays on. It also misfires on decode: CUDA's
`get_op_batch_size` falls back to `ggml_nrows(op)`, and for the GDN op (output carries the
state) and the gated-norm MUL (`[128, 48 heads, 1 token]`) that is >= 32 at one token. So each
CPU GDN layer bounces 3 times to the GPU and copies its 3 MB state across PCIe (~6 ms/token at
12 CPU blocks).

nsys, decode on `cpu12q4` (~39 ms GPU per token): ~93% of it is the PTQ1_0 `mul_mat_vec_q`.

| matmul | per call | effective BW | per token |
|---|---|---|---|
| ffn gate+up (fused) | 240 us | ~163 GB/s | ~12.4 ms |
| ffn down (K = 17408) | 163 us | **~120 GB/s** | ~8.5 ms |
| attn / GDN projections | 55 us avg | ~150-210 GB/s | ~8.6 ms |
| GDN op + state gather / copy | | | ~2-3 ms |

q4_0 reaches 244 GB/s on this card, so the PTQ1_0 mmvq has up to ~2x headroom, worst on the
down projection. `ncu` needs GPU counter access (ERR_NVGPUCTRPERM) - enable it in the NVIDIA
Control Panel (Developer -> Manage GPU Performance Counters) or run elevated.

Best measured config so far (8.2-8.6 t/s tg, 169 t/s pp512):

```
llama-server -m Bonsai-2-27B-PTQ1_0-cpu12q4.gguf -ngl 53 -fa on -c 4096 -ctk q8_0 -ctv q8_0 -b 512 -ub 128 -t 8 -lm none
```

### Next, by expected gain per effort

1. Free the ~400 MiB the desktop holds on the dGPU (apps -> Power saving in Windows Graphics
   settings). ~5 fewer CPU blocks = ~-30 ms/token (~8.5 -> ~11 t/s). Zero code.
2. PTQ1_0 mmvq on sm_86 (down projection first): up to ~2x on the ~36 ms GPU matmul time.
3. Fix the op-offload batch size for GDN / per-head ops (small ggml-cuda change): ~-6 ms/token.
4. AVX-512 VBMI kernel for a 2-bit ternary layout on the CPU: 6.0 -> ~2.5 ms per CPU block.

## 6. PTQ1_0 mmvq on sm_86 (2026-09-18)

Desktop VRAM: per-app "Power saving" GPU preference set for the 12 apps on the dGPU (backup
`I:\Kyle_Shepherd\models\UserGpuPreferences-backup-2026-09-18.reg`). After re-login they still
all attach to the dGPU (Chromium / shell open a context on every adapter): 409 -> 368 MiB only.

Kernel changes (`vecdotq.cuh`, `mmvq.cu`, CUDA only, HIP keeps its path):

1. Two lanes per 128-trit block (VDR 4 -> 2). Lane p takes qs ints {2p, 2p+1}, qs int 4+p and
   qh trit pair p. The Q8_1 block index stays compile-time (it does not depend on p) - a runtime
   index spilled `sumi` to local memory and made the first try 2x slower at n=2.
2. Dot the raw digits {0,1,2} and take the -1 off once per Q8_1 block via `ds.y` (as Q4_0
   does with its -8), instead of an emulated `__vsub4` per 4 trits.
3. Always use the small-K geometry (nwarps rows per lane) for PTQ1_0 on NVIDIA: at K = 17408
   one row per lane had too few loads in flight (142 -> 91 us). nwarps 2 or 8 were worse.

`test-backend-ops perf`, Bonsai shapes (new cases in `make_test_cases_perf`), n=1:

| m x k | before | after |
|---|---|---|
| 10240 x 5120 | 81.7 us (140 GB/s) | 59.7 us (192 GB/s) |
| 17408 x 5120 | 136.0 us (143 GB/s) | 100.7 us (194 GB/s) |
| 5120 x 17408 (ffn down) | 188.8 us (103 GB/s) | **93.0 us (210 GB/s)** |
| 5120 x 6144 | 46.2 us (149 GB/s) | 34.3 us (201 GB/s) |
| 248320 x 5120 (output) | 2507 us (111 GB/s) | 1739 us (160 GB/s) |

n=2 / n=4 also 25-35% faster, but still scale ~linearly (PQ2_0 is flat at n=2).

In the model (nsys, `cpu12q4`, `-ub 128`): GPU per token 35.4 -> 25.8 ms (-27%), token period
123.4 -> 112.1 ms. PTQ1_0 matmuls are ~18.3 ms/token for ~4.65 GB, ~254 GB/s effective (q4_0
class, ~75% of the 336 GB/s peak). Unprofiled `llama-completion` runs vary +-5 ms run to run on
the CPU side, which hides the gain in single runs.

What is left, per token: ~86 ms on the CPU (12 Q4_0 blocks + the op-offload bounces), ~26 ms
GPU. The CPU side is now >3/4 of the time, so next: op-offload fix (item 3 above), then the
AVX-512 ternary CPU kernel, and n>1 PTQ1_0 scaling only if speculative decoding comes back.

## 7. CPU side: PQ2_0 blocks, 4-row dot, op-offload fix (2026-09-18)

- CPU-resident blocks as exact PQ2_0 (`llama-ternary-repack ... pq2_0`): 1155 MiB for 12
  blocks vs 2447 MiB as Q4_0.
- `ggml_vec_dot_pq2_0_q8_0`: AVX-512 VBMI+VNNI single-row path (vpmultishiftqb decode) - 1.9x
  over the old VNNI path, but still ~40 cycles per 128 block. Measured with a standalone MSVC
  harness: every instruction is ~1/cycle, the loop is just too long per row.
- `ggml_vec_dot_pq2_0_q8_0_x4`: 4 plain-layout rows at once, same math as the
  `ggml_gemv_pq2_0_4x8_q8_0` repack GEMV (y loads, sum(y), y scale shared by 4 rows), called
  from `ggml_compute_forward_mul_mat_one_chunk` for PQ2_0. In the model: 967 -> ~570 us per
  PQ2_0 matmul, ~32 GB/s, near RAM bandwidth.
- The repack buffer itself (`--no-host`) is as fast for decode but loses op offload: pp512 33
  t/s instead of ~190. With `_x4` the plain pinned layout gets both.
- Bug fix: `--no-host` crashed on Hadamard-folded models. The rotation and sign tensors were
  allocated in the weight's buffer type, which is `CPU_REPACK` for a repacked weight and can not
  hold plain F32. They now go to the default CPU buffer type.
- Op offload: host weights under 1 MiB no longer pin an op or trigger an offload, the expand
  passes place it next to its neighbors. GATED_DELTA_NET keeps the weight rule, and CUDA now
  counts its batch as tokens (its rows include the state snapshots). Decode graph splits 38 -> 2.

`llama-bench`, `cpu12pq2`, `-ngl 53 -ub 128 -b 512 -fa 1 -ctk q8_0 -ctv q8_0 -t 8`:

| step | pp512 | tg64 |
|---|---|---|
| cpu12q4, before this section | 169-189 | 7.8-8.6 |
| cpu12pq2, `--no-host` (repack) | 33 | 11.4 |
| cpu12pq2 + `_x4` | 189 | 10.8 |
| + op-offload fix | **200** | **12.6** |

Current best:

```
llama-server -m Bonsai-2-27B-PTQ1_0-cpu12pq2.gguf -ngl 53 -fa on -c 4096 -ctk q8_0 -ctv q8_0 -b 512 -ub 128 -t 8
```

Per token now: ~26 ms GPU + ~38 ms PQ2_0 matmuls on the CPU + ~16 ms of other CPU ops and
syncs. Next candidates: fewer CPU blocks (every freed 80 MiB of VRAM is ~3 ms), and the ~16 ms
of non-matmul CPU time.

## 8. Non-matmul CPU time, and the power limit (2026-09-18)

Per-op profile over ~340 decode tokens (temporary thread-0 timer in `ggml_graph_compute_thread`,
not committed): non-matmul CPU work is ~8 ms/token, not ~16. Thread wake-up is negligible.

| op | before | after |
|---|---|---|
| CONCAT (conv state + new column, 160 KB) | 133 us | 30 us |
| GET_ROWS (recurrent state gather, one 3 MB row) | 65 us avg | 32 us avg |

Both were single-threaded at decode: CONCAT split threads over dim 2 only and GET_ROWS over
rows only, and a decode step has one sequence / one state row. CONCAT now splits over all dst
rows and copies contiguous runs with memcpy; GET_ROWS splits a long row in column chunks.
~2 ms/token saved. The rows-indexed GDN state read (no gather) only exists on the ring path
(needs `n_rs_seq > 0` and Metal), so it does not apply here.

The bigger effect is the CPU power limit: under sustained all-core load the i9-11900H drops
from ~137% of base clock to ~97% (~2.4 GHz) after ~15 s and stays there. Short benchmarks run at
boost, long generations do not (77 ms/token right after a pause, ~92 ms/token over 600 tokens).
At base clock the PQ2_0 matmuls are compute-bound, so threads trade against clock:

| threads | 600-token decode |
|---|---|
| 4 | 120.3 ms/token |
| 6 | **86.0 ms/token** |
| 8 | 94.0 ms/token |

Use `-t 6`. A higher power mode in the laptop vendor's tool (a system setting, left to the
user) would lift PL1 and help directly.

## 9. Serving: fewer CPU blocks, and the 4-row kernel limit (2026-09-18)

`llama-server` reserves logits for n_parallel outputs only (compute buffer 63 MiB at `-ub 128`
vs 133 MiB in llama-completion / llama-bench). But `-np` defaults to auto = 4 slots, and each
slot gets its own recurrent state: 598 MiB instead of 150. **`-np 1` for a single user.**

400-token generation through the server API, `-np 1 -ub 128 -c 4096 -ctk q8_0 -ctv q8_0 -t 6`:

| file | `-ngl` | VRAM used | gen |
|---|---|---|---|
| cpu12pq2 | 53 | 5238 MiB | 10.6 t/s |
| cpu9pq2 | 56 | 5490 MiB | 14.0 t/s |
| cpu8pq2 | 57 | 5576 MiB | 15.2 t/s |
| cpu6pq2 | 59 | 5746 MiB | 17.6 t/s |
| cpu5pq2 | 60 | 5830 MiB | 19.1 t/s |
| cpu4pq2 | 61 | 5916 MiB | **20.4 t/s** |
| cpu3pq2 | 62 | 5880 MiB (!) | 11.5 t/s - WDDM paged VRAM to system memory |

cpu4 is the edge with ~230 MiB spare; cpu5 leaves ~310 MiB for the desktop to grow. `-t 6` and
`-t 8` are within noise at cpu5 (54.2 vs 52.4 ms/token).

4-row PQ2_0 kernel at base clock: three variants tried in a standalone MSVC harness (whole-row
loads + vpermt2q + vpmultishiftqb decode; vpmultishiftqb crumb spread; float work per 128 block
instead of per q8_0 block). None beat the committed kernel (17-19 cycles per 128-weight row-block):
it is bound by the two 512-bit ports, with the decode shuffles on p5 and dot / float on p0.
The next step would be Q8_K activations (one scale per 256) to drop most per-32 float work, ~1.3x
on the CPU matmuls, worth ~5 ms/token at 4-5 CPU blocks. Not done.

## 10. Per-layer KV placement: `--kv-cpu-layers N` (2026-09-18)

`llama_model_params::kv_cpu_layers` / `--kv-cpu-layers N`: the KV cache of the first N layers
that have one (the 16 full-attention layers here) is allocated in host memory. Everything else is
unchanged; the scheduler streams the filled part of those caches to the GPU for attention each
step (prefill and decode), because attention stays with its query on the GPU.

Server, `-np 1 -c 32768 -ctk q8_0 -ctv q8_0 -ub 128 -t 6`, 128 generated tokens after a prompt:

| layout | filled | gen |
|---|---|---|
| cpu6, `-ngl 59`, 16 KV layers in host memory | ~0 | 56.7 ms/token |
| same | 8.6k | 97.2 ms/token |
| same | 16.5k | 131.1 ms/token (prefill 185 t/s) |
| cpu11, `-ngl 54`, 8 KV layers in host memory | 16.5k | 159 ms/token (at the VRAM cliff) |
| cpu17, `-ngl 48`, all KV on the GPU | 16.5k | 235 ms/token |

So host KV costs **~4.5 ms per 1k filled tokens** (all 16 layers, ~0.28 ms per 1k tokens per
layer; PCIe ~13 GB/s) - and it follows the *filled* context, not the allocation. At 16.5k it
beats pushing weights to the CPU by 1.8x.

Tried and not kept: running decode attention on the CPU next to the host KV. Even with a new
grouped-query CPU decode path (committed: q8_0, head 256, 4x6 heads, 16k KV 9.7 -> 3.6 ms per
layer, 2.7x) it is ~150 ms/token at 16.5k vs ~81 ms streamed. It would need a CPU attention
kernel near RAM bandwidth (~1 ms/layer at 16k) to win.

Notes:
- At `-c 32768` the CUDA compute buffer is 235 MiB (63 MiB at 4k): it reserves room for one
  layer's streamed K/V in the prefill graph. That is why cpu6, not cpu5, fits with host KV.
- Upstream's `--no-kv-offload` pins attention to the CPU only on the non-flash-attention path.

## 11. Adaptive context growth (design, not implemented)

Many workloads do not know their context up front. Today the layout (which blocks are on the
CPU, how big the KV cache is) is fixed at load. Goal: start fast (few CPU blocks, small KV) and
give ground only as the context actually fills.

### 11.1 Why not swap the whole file

The naive version - reload with the next `cpuNpq2` file when the context outgrows the current
layout - works in principle but has real costs:

1. **State is lost unless explicitly carried.** The KV cache, the recurrent state (48 GDN
   layers, 150 MiB) and the context checkpoints all live in the context being torn down. A
   hybrid model cannot recompute part of its recurrent state: without a save/restore it has to
   re-prefill the whole conversation (16k tokens = ~90 s). Carrying it means
   `llama_state_seq_get_data` / `_set_data` (or the fork's RAM slot store) around the reload -
   ~1.3 GB at 32k, fine, but it is one more path to get right, including checkpoints.
2. **Reload latency.** 6-8 GB to map and ~5.5 GB to upload to the GPU: a few seconds with a warm
   page cache, much more cold. It must happen between tokens, so streaming clients stall, and
   every slot pauses.
3. **Disk and page cache.** One ~6-8 GB file per layout. Two layouts mapped during the swap
   double the page cache footprint; with several variants the cache stops holding them.
4. **VRAM churn on Windows.** Tearing down and re-creating ~5.5 GB of allocations under WDDM,
   right at the edge of the spill cliff (section 9): free-memory reports lag, and a layout that
   fit before may spill after.
5. **Thrashing.** Contexts shrink too (new chat, cleared cache). Without hysteresis it reloads
   back and forth; with hysteresis it sits in a slow layout after the context is gone.
6. **Different numbers after the swap.** A layer that moves between GPU (PTQ1_0 kernel) and CPU
   (PQ2_0) computes the same weights with different rounding. Not wrong, but a run is no longer
   reproducible across the swap point.

Everything the swap achieves can be done inside one process, without losing state.

### 11.2 In-process levers, cheapest first

**L1 - grow the KV cache on the GPU while VRAM is free.** The fork already has this
(`--ctx-max`, `--ctx-grow-factor`, `--ctx-grow-headroom`, `llama_kv_cache::resize`). Start with
a small cache and the best weight layout (cpu4/cpu5).

**L2 - spill KV layers to host memory as the cache grows.** When a resize does not fit in VRAM,
reallocate some layers' grown K/V in host memory instead of failing (the same placement
`--kv-cpu-layers` does statically). `resize()` already reallocates per layer and copies the old
cells; it only needs to pick the buffer type per layer (host for the first k layers, k grown as
needed), using the fork's VRAM-fit check (`ggml_backend_set_vram_strict`,
`ggml_cuda_fits_in_vram`) rather than the WDDM report. Graphs are rebuilt after a resize anyway,
and the scheduler handles the streaming. Cost is known: ~0.28 ms per 1k filled tokens per
spilled layer.

**L3 - migrate weight blocks (dual residency).** Keep a host PQ2_0 copy of every block (all 64
blocks = ~6.2 GB RAM) plus PTQ1_0 in VRAM for the resident set. Evicting a block frees ~80 MiB
at once by switching that layer's graph to its host tensors; re-admitting it uploads ~70 MiB
from the mapped PTQ1_0 file (~6 ms). Needs per-layer VRAM buffers (today one buffer per buffer
type) and a loader that holds both variants (two GGUFs, or one GGUF with both tensor sets). No
state is touched. This is the dense-model version of the expert hot store.

### 11.3 Policy

Price every freed MiB of VRAM:

- a weight block on the CPU: ~4.5-6 ms/token per ~80 MiB, fixed, independent of the context;
- a KV layer in host memory: ~0.28 ms/token per 1k *filled* tokens, frees 2.2 KiB per
  *allocated* token (q8_0).

With a sized cache (allocation ~= fill, which is what L1's headroom gives) the two are close per
MiB at full fill, and KV wins whenever the cache is not full - and the measurement at 16.5k
favours KV by 1.8x. So: grow KV on the GPU (L1) -> spill KV layers (L2) -> only then move weight
blocks (L3), and prefer moving whichever is cheaper at the current fill. Shrink in reverse when
idle (the fork's refit-while-idle pattern), with hysteresis. Before a big prompt, grow once to
the known size instead of step by step (the server knows the prompt length before prefill).

### 11.4 Work items

| step | what | effort |
|---|---|---|
| a | `resize()` places grown layers in host memory when the VRAM-fit check fails (L2) | small |
| b | spill order and count from the policy above, logged per resize | small |
| c | move host layers back to the GPU on shrink / idle, with hysteresis | small |
| d | faster host-KV attention: CPU flash attention near RAM bandwidth, or reading pinned host KV straight from the GPU kernel instead of copying | medium |
| e | per-layer VRAM buffers + dual-residency loader + graph switch per layer (L3) | large |

With a, b and c, a single `cpu4pq2`/`cpu5pq2` file serves any context length: short chats run at
~19-20 t/s, long ones degrade by ~4.5 ms/token per 1k tokens instead of needing a reload.

## 12. Vision: `--mmproj-compute-lazy` (2026-09-19)

Bonsai mmproj-Q8_0: weights 600 MiB (kept in RAM with `--mmproj-weights-host`), encoder compute
buffer 288 MiB reserved by the warmup for the largest image, plus ~80 MiB for the encoder's own
CUDA backend (pool, cuBLAS). The warmup reserve does not shrink with `--image-max-tokens`.

`--mmproj-compute-lazy` (commit `148f4dd56`, cherry-picked to branch `mmproj-compute-lazy` off
master) frees the encoder scheduler and GPU backend after the warmup and after every encode and
re-creates them at the next encode, where a fresh scheduler sizes buffers to the actual image.
Encode transient above the LLM (llama-mtmd-cli, 1000x1000 image): +368 MiB non-lazy, +186 MiB
lazy (+99 at `--image-max-tokens 512`, +67 at 256).

The practical VRAM ceiling on this laptop is ~5.95 GB, and a spill does not recover: once an
encode pushes over it, text generation stays slow for the rest of the session.

Server, 1000x1000 image then text, `-np 1 -c 4096`:

| layout | text before | image turn | text after |
|---|---|---|---|
| cpu5/60, host weights | 19.6 t/s | 79 s | 8.7 t/s (spilled) |
| cpu5/60, host + lazy | 19.5 t/s | 68 s | 11.3 t/s (spilled) |
| cpu5/60, host + lazy + max 512 tokens | 19.6 t/s | 30 s | 11.3 t/s (spilled) |
| cpu6/59, host weights | 18.1 t/s | 70 s | 8.6 t/s (spilled) |
| **cpu6/59, host + lazy** | **18.1 t/s** | **21 s** | **17.7 t/s** |

Vision command (cpu6, lazy):

```
llama-server -m Bonsai-2-27B-PTQ1_0-cpu6pq2.gguf -ngl 59 --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --mmproj-weights-host --mmproj-compute-lazy -np 1 -fa on -c 4096 -ctk q8_0 -ctv q8_0 -b 512 -ub 128 -t 6
```

Qwen3.6-35B-A3B MoE on the master branch with the fitter: lazy frees ~273 MiB at idle, no
spill either way. Under the fitter that VRAM stays unused (it still reserves the projector's
worst case, which the lazy transient for a maximum-size image equals); using it needs the hot
store to take it and give it back around an encode (`PLAN-adaptive-mmproj.md`).

Open: at cpu5 even the +99 MiB transient spilled, so something else grows on an image turn
(likely the LLM compute graph for embedding input). Measure `sched_reserve` for an image batch.

## 13. KV spill on growth (2026-09-19, `PLAN-kv-host-streaming.md` item A)

With `--ctx-max`, `llama_kv_cache::resize()` places the cache before it moves it: a device keeps
all of its layers or none of them, and it keeps them while the free VRAM after the move is at
least `--kv-spill-margin MiB` (default 128). A shrink, or `llama_kv_unspill()` (the server calls
it when idle and before each prefill), moves them back, not within 60 s of a spill and only with
2x the margin free. resize() reads the cells both sizes share into host memory first, so the old
buffers go back before the new ones are taken - the device holds one copy of the cache, not two -
and a layer the device still refuses is placed in host memory. If the compute buffers cannot be
reserved after the move, the cache goes to host memory before they fall back there themselves.
One log line per resize: `kv resize 4096 -> 4864 cells, 1/16 layers in host memory, VRAM free 155 MiB`.

**The free figure has to be NVML, not `cudaMemGetInfo`.** On WDDM the CUDA figure is what is left
of the process' memory budget: it reads 0 as soon as the process is at its budget even while the
card has memory free (5130 MiB free at start, cpu6/59 at `-c 4096` asks for ~5230, so it reads 0
right after load while nvidia-smi shows 5.8 of 6.1 GB used). Every layer then looks like it does
not fit, and the first growth moved the whole cache to host memory - `--ctx-max` cost ~20% of the
decode rate at a context that used to stay on the GPU. `ggml_cuda_free_vram()` asks NVML (what
nvidia-smi reports, over all processes) and falls back to the driver figure; the strict VRAM check
and the placement both use it.

**All or nothing per device.** A mixed layout keeps the device layers *and* makes the scheduler
stage the host ones in a VRAM buffer of its own, so on a nearly full card it costs more than it
frees. At 8960 cells, 8.6k filled: 7 of 16 layers in host memory 123 ms/token, all 16 on the
device 62, all 16 in host memory 61.

Server, cpu6/59, `-c 4096 --ctx-max 32768`, one session, gen ms/token:

| prompt | cache | placement | gen |
|---|---|---|---|
| 4.7k | 4864 | GPU (1 layer refused) | 57.8 |
| 3.4k | 4096 (no growth) | GPU | 58.2 |
| 8.6k | 8960 | host | 85.9 |
| 16.5k | 16896 | host | 128.4 |

VRAM stayed at 5.87 GB or below. A fresh load at `-c 8960` with the cache on the GPU decodes at
70.9 at 8.6k filled, but *growing* into that state and keeping it there does not hold: the cache
takes the last VRAM and the compute buffers land in system memory (310 ms/token), which is what
the margin and the reserve guard are for.

Measured before the placement was rewritten (per-layer spill, `cudaMemGetInfo`, margin 256), kept
because it shows what a roomier card does - cpu11/54 has ~270 MiB free at load:

| layout | 2k | 8.6k | 16.5k | static layout at 16.5k |
|---|---|---|---|---|
| cpu5/60 | 55.0 | 96.0 (16/16 host) | **215** (compute buffer in sysmem) | - |
| cpu6/59 | 57.6 | 97.6 (16/16) | 131.2 (16/16) | 131.1 (`--kv-cpu-layers 16`) |
| cpu11/54 | 76.7 | 101.4 (11/16) | 142.8 (15/16) | - |
| cpu11/54, margin 64 | 76.3 | 90.3 (5/16) | 120.2 (10/16) | 121.1 (`--kv-cpu-layers 10`) |

Measuring this needs care: a server started within a few seconds of the previous one exiting runs
at a fraction of the rate (prefill 37 t/s against 206, decode 128 ms against 58) until the driver
has taken the old process' VRAM back. Wait for `nvidia-smi` to drop to idle between runs, and
repeat anything that looks like a cliff.

## 13b. Vision: the encode borrows the cache's VRAM (2026-09-20)

An image encode wants a few hundred MiB of VRAM while it runs (+186 MiB for 1000x1000 with
`--mmproj-compute-lazy`, section 12). Holding that open permanently is what the spill margin did:
with an mmproj loaded the cache moved to host memory one growth step earlier than without it, and
stayed there for the session. Now `llama_kv_spill()` moves the cache to host memory just before
`mtmd_batch_encode()` when the device has less than 384 MiB free, and `llama_kv_unspill(force)`
moves it straight back after - force skips the 60 s hold-off and the room-to-spare rule, since
this is the other half of a spill the server made itself.

The margin is 96 MiB, which is what separates the two outcomes on this card. Keeping a 4864-cell
cache on the GPU leaves 125 MiB free and runs; a 6656-cell one leaves 91 MiB and pages.

cpu6/59 with the mmproj, `-c 4096 --ctx-max 32768`, image first then growing text turns:

| turn | cache | placement | result |
|---|---|---|---|
| image (cold) | 4096 | spilled for the encode, back after | 8.7 s (27.5 s before) |
| text 4.7k | 4864 | GPU | 17.0 t/s (13.5 with margin 128) |
| image (same) | 4864 | spilled, back after | 2.2 s |
| text 6k | 6656 | host | 12.6 t/s (8.0 on the GPU - the cliff) |

Text without the mmproj at the same margin: 16.9 / 11.5 / 8.7 t/s at 4.7k / 8.6k / 16.5k.

## 14. Pinned host KV; zero-copy dropped (item B3)

Host KV layers (`--kv-cpu-layers` and spilled) are now allocated in the GPU's pinned host buffer
type (`CUDA_Host`) instead of plain CPU memory. The scheduler's per-step copy then DMAs from pinned
memory: **cpu6/59, 16 host layers, 16.5k: 131 -> 112 ms/token** (112.0, 111.8), prefill 185 -> 190 t/s.

Zero-copy was tried and dropped. A scheduler hook let the CUDA flash-attention node read host K/V in
place (no split input copy):
- vec kernel (decode): 918 ms/token. It has one block per Q head, so each KV row crosses PCIe 6x
  (GQA 6), with a read pattern made for VRAM.
- MMA kernel forced for host K/V (converts to F16 in one coalesced pass): 115.9 / 117.6 ms/token,
  slower than the pinned copy (112.0).
- It saves no VRAM. The CUDA compute buffer at `-c 32768` is 235 MiB **with the KV on the GPU too**:
  it is the F16 K/V conversion space of the flash-attention node (`ggml_cuda_flash_attn_ext_get_alloc_size`,
  ~4 KiB per cell), not staging. Section 10's note was wrong.

## 15. AVX-512 VNNI CPU decode attention (item C)

New `ggml_compute_forward_flash_attn_ext_gqa_chunk_q8_0` (x86, AVX-512 F/BW/DQ/VNNI; q8_0 K and V,
head sizes multiple of 64, DV <= 256): Q quantized once per call; scores as `dpbusd(q+128, k) -
128*sum(k)` with the correction shared by all heads, 16 rows reduced at once by a transposed sum;
vectorized softmax (`ggml_v_expf`); V block scales folded into the weights; DV in chunks of 64 with
named accumulators (MSVC keeps arrays of vectors in memory: the first version did a load + FMA +
store per FMA).

`test-backend-ops perf -b CPU`, head 256, 4 KV heads x 6, one query, 16 threads:

| KV | old gqa path | new |
|---|---|---|
| 4k | - | 0.37 ms |
| 16k | 4.0-4.2 ms (3.6 at boost) | **1.43-1.50 ms** |
| 32k | - | 2.7-2.9 ms |

Compute bound, not memory bound (4k in L3 has the same per-row cost; prefetch changes nothing).
Tiger Lake runs 512-bit FP on port 0 only: ~100 port-0 ops per row for scores (dpbusd, cvt, mul, fma
per head and block pair), ~112 for V. Target was 1.2 ms. Greedy `-ngl 0` output identical to the
old path.

In-model, decode attention of all 16 host layers pinned to the CPU (6 threads), cpu6/59, 16.5k:
117.4 / 119.5 ms/token vs 111.8 streamed from pinned memory. Before this kernel it was ~150.
Not kept (a 6-thread CPU at base clock does 16 layers in ~60 ms; the pinned copy takes ~55 ms).
Worth it only split with the GPU (item D).

## 16. Where a 16.5k decode goes (nsys, item B2 groundwork)

cpu6/59, 16 host KV layers (pinned), 16.5k filled, `llama-completion -n 64 --ignore-eos` under
`nsys profile -t cuda` (121 ms/token profiled, 112 unprofiled). Decode graphs appear in
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, not `..._KERNEL`.

| per token | ms |
|---|---|
| GPU compute | 32 |
| host -> device copies (561 MB, 12.0 GB/s) | 47 |
| copies overlapped with compute | 0 |
| GPU idle (CPU weight blocks, per-layer syncs) | 42 |

- Async copies on the compute stream (skip the scheduler's full sync before a host -> device input
  copy): 111.5 vs 112 ms/token. Not kept.
- The plan's prefetch cannot overlap as written: layer L's KV store (a CPU split: k_cur/v_cur go to
  the CPU, set_rows, back) runs right before layer L's attention, and the copy has to follow the store.
  Nothing computes in between. Overlap needs the old cells prefetched earlier (during layer L-1) on a
  second stream into a second staging buffer (+36 MiB at 16.5k, +70 MiB at 32k: over the cliff on
  cpu6/59, so cpu7/58), and the new row written on the GPU into that staging copy as well as into
  host memory. That is explicit staging in the graph, not the scheduler's implicit input copy.
  Ceiling ~32 ms/token.

## 17. Phase 0 - original checklist

1. System prep: NVIDIA Control Panel -> *CUDA - Sysmem Fallback Policy* = *Prefer No Sysmem
   Fallback* (otherwise an overcommit silently pages VRAM over PCIe); move the desktop apps
   off the dGPU; plugged in, Windows *Best performance*.
2. Build A. Check `system_info` shows `AVX512_VNNI = 1`.
3. Full load attempt, read the ledger:
   `llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -fa on -c 4096 -ctk q8_0 -ctv q8_0 -fitt 256 -t 8 -n 64 -p hi`
   Record `CUDA0 model buffer`, KV, `llama_memory_recurrent`, `CUDA0 compute`, graph splits,
   and the `-ngl` the fitter chose.
4. `llama-bench` tg128 / pp512 at the fitter's choice, then +/-1 block, `-ub 128/256/512`.
5. Same with the C hybrid file, once it exists.
6. Watch `nvidia-smi -q -d PERFORMANCE` during tg for thermal / power slowdown.
