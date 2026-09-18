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

## 7. Phase 0 - original checklist

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
