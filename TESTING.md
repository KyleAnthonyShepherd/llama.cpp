# TESTING.md — how to test this branch's work

This is a practical, hands-on guide for testing the two pieces of work done across the
last two sessions on branch `claude/part-1-resize-core-v43lmj`:

1. **Part 3 — MTP × `--mmproj` fix** (previous session): a correctness fix so that MTP
   speculative decoding keeps working correctly across image/audio chunks in a
   multimodal (VLM) session, instead of silently drafting from a stale hidden state.
2. **Part 1 — dynamic KV cache growth, Phases 2 & 3** (this session): `resize()` on the
   KV cache, plus `llama_set_n_ctx()` and the automatic growth hook inside
   `llama_decode()`.

Everything here was implemented and unit-tested on a network- and GPU-less Linux
sandbox. The unit tests build cleanly but were **not run end-to-end against a real
downloaded model** (the sandbox couldn't reach the model host) — that, plus anything
GPU-specific, is exactly what's left for your machine.

See `NOTES.md` at the repo root for the full discovery/implementation log this guide is
distilled from, if you want the "why," file/line references, and open items.

---

## 0. Building on Windows (Visual Studio 2022)

Reference: [`docs/build.md`](docs/build.md) — this section is that doc's Windows
instructions, condensed for your setup (6 GB VRAM NVIDIA GPU, 64 GB RAM).

### 0.1 Prerequisites

1. Install **Visual Studio 2022** (Community edition is fine). In the installer, under
   the **Workloads** tab, select **Desktop development with C++**. This pulls in MSVC,
   the Windows SDK, and CMake automatically.
2. **Always** build from a **Developer Command Prompt for VS 2022** (or the
   corresponding PowerShell variant) — search for it in the Start menu. Plain
   `cmd.exe`/PowerShell won't have `cl.exe`/`cmake` set up on `PATH` correctly.
3. Install **Git for Windows** if you don't already have it.
4. For GPU acceleration: install the **NVIDIA CUDA Toolkit**
   (<https://developer.nvidia.com/cuda-downloads>) — pick a version your driver
   supports. Check your GPU's compute capability at
   <https://developer.nvidia.com/cuda-gpus> (most current 6 GB cards — e.g. RTX 3050/4050
   class — are compute capability 8.6 or 8.9).

### 0.2 Get the code and check out this branch

```bat
git clone https://github.com/KyleAnthonyShepherd/llama.cpp
cd llama.cpp
git checkout claude/part-1-resize-core-v43lmj
```

### 0.3 CPU-only build (fastest way to a working binary, good first sanity check)

```bat
cmake -B build
cmake --build build --config Release -j 8
```

Binaries land in `build\bin\Release\` (e.g. `llama-cli.exe`, `llama-server.exe`,
`test-kv-resize.exe`).

### 0.4 CUDA build (what you actually want for the 6 GB GPU)

```bat
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

- If `nvcc` can't auto-detect your GPU (a warning like `Cannot find valid GPU for
  '-arch=native'`), pin the compute capability explicitly, e.g. for an 8.6-class card:
  ```bat
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
  ```
- First CUDA configure/build is slow (many kernel variants to compile); subsequent
  builds are much faster. Installing [ccache](https://ccache.dev/) helps a lot if you
  rebuild often.

### 0.5 Building just the tests

```bat
cmake --build build --config Release --target test-kv-resize test-ctx-grow -j 8
```

(Both targets also get built by a plain `cmake --build build --config Release` with no
`--target`, since they're wired into the normal build; this is just for a faster
incremental loop while iterating.)

---

## 1. Testing the MTP × `--mmproj` fix (Part 3)

### 1.1 What changed, in one paragraph

MTP speculative decoding keeps a small side context (`ctx_dft`) whose job is to replay
every decoded token so its hidden-state cache (`pending_h`) stays in sync with the main
model. Before the fix, that replay step **silently skipped itself** whenever the batch
being replayed was an image/audio embedding batch (`common/speculative.cpp`, look for
the old `// TODO: how to make it work with vision tokens?` — now gone). That left MTP
drafting from a stale, pre-image hidden state after any image in the conversation — not
a crash, just quietly wrong/degraded output. The fix: mark the draft state **invalid**
across an image/audio chunk, then have the very next generation step run **un-speculated**
(one ordinary decode, no drafting) to re-prime from the correct post-image hidden state,
after which drafting resumes normally. Full mechanism + code pointers: `NOTES.md`, the
"Part 3 Phase 1 discovery answers" section and "Implementation: item 5's fix."

### 1.2 What you need

- A GGUF model whose architecture has MTP heads (`qwen35`/`qwen35moe` family — i.e. your
  Qwen3.6-35B-A3B quant — is the family this fix targets) **and** a vision projector
  (`mmproj`) file for a multimodal variant of that same model family. I'm not going to
  guess a specific HuggingFace repo/quant name here since I can't verify one exists in
  exactly this shape from this sandbox — use whatever multimodal Qwen3.6-family GGUF +
  mmproj pair you already have or can source yourself. `docs/multimodal.md` has the
  general mechanics (`-hf`, `--mmproj`, `--no-mmproj-offload`, etc.) if you need a
  refresher; it doesn't (yet) list a Qwen3.6-VL entry, so you're on your own for finding
  the actual weights.
- A CUDA build (§0.4) if you want realistic speed — MTP + a 35B model on CPU-only will be
  very slow but should still be *correct*, which is all these tests actually check.

### 1.3 Test 1 — text-only determinism (does MTP still work at all after the change)

This checks nothing about images yet — it's the regression guard that the fix didn't
break ordinary MTP.

```bat
build\bin\Release\llama-server.exe -m your-model.gguf --mmproj your-mmproj.gguf ^
    --spec-type draft-mtp -c 8192 -np 1 --temp 0 --seed 1
```

Send a plain text chat request (no image) via the web UI at `http://localhost:8080` or
`curl`, twice: once with `--spec-type draft-mtp`, once with `--spec-type none` (both
`--temp 0`). In principle **the generated text should be token-for-token identical** — but
in practice, on real hardware (Qwen3.6-35B-A3B Q4_K_M, CUDA), **it wasn't**: output length
differed (581 vs. 614 tokens) even though both were coherent, with a healthy-looking draft
acceptance rate (`~0.52`, mean accepted length `2.57`). This is very unlikely to be caused
by the Part 3 `--mmproj` fix specifically — no image is involved in this test, and that
fix's code only ever runs on image/audio embedding batches — so it's most likely either a
pre-existing gap in `draft-mtp`'s own accept/verify logic, or an inherent floating-point
difference between the *batched* verify pass and *sequential* single-token decode (a known
category of issue in speculative decoding generally, independent of any logic bug). See
`NOTES.md`'s "Real-hardware findings" section for the fuller writeup. **Current status:
treat this as a known, open, separately-tracked issue** (not a blocker for anything else in
this guide) rather than a regression to chase down here — if the MTP-on output is
*coherent* but simply different from MTP-off, that matches this known state; if it's
*garbled/repeating*, that's a new, more concerning symptom worth reporting separately.

### 1.4 Test 2 — image + MTP generation (the actual fix)

```bat
build\bin\Release\llama-server.exe -m your-model.gguf --mmproj your-mmproj.gguf ^
    --spec-type draft-mtp -c 8192 -np 1 --temp 0 --seed 1
```

Send a chat request with an image followed by a text prompt (e.g. "describe this
image"), `temp 0`. Run it twice: once with `--spec-type draft-mtp`, once with
`--spec-type none`. **Same requirement: token-for-token identical output.** This is the
core correctness gate from the plan (Part 3 §4, "Phase 3 — Milestone P2"). If output
differs, or degrades into repetition/gibberish after the image, the fix has a bug —
check whether it's specific to the image being the *first* turn vs. a *later* turn, and
capture the request/response for a bug report.

Try it with:
- one image, then a follow-up question about the same image (multi-turn)
- two images in the same prompt
- an image as the very last element before generation starts (this is the case that
  most directly exercises the "invalidate → re-prime on next step" path)

### 1.5 Test 3 — is drafting actually happening, and at a reasonable rate?

Run with trace logging so you can see the MTP internals (acceptance stats, per-step
drafting):

```bat
build\bin\Release\llama-server.exe -m your-model.gguf --mmproj your-mmproj.gguf ^
    --spec-type draft-mtp -c 8192 -np 1 -lv 4
```

`-lv 4` (or `--log-verbosity 4`) enables `TRACE`-level logs, which includes the
`common_speculative_print_stats` acceptance-rate output and the `SPC_TRC` lines showing
each draft step. Things to look for:

- After an image chunk, you should see one generation step where **no draft happens**
  (the forced re-prime step), then normal drafting resumes.
- The completion response's `timings` field also reports `draft_ratio` /
  `mean_acc_len` per request (no special flag needed) — acceptance rate on an
  image-description prompt should be in the same ballpark as a comparable text-only
  prompt. If it craters specifically on image-containing prompts, that's a sign
  something's still off with post-image state (see NOTES.md's discussion of `begin()`'s
  position-counting heuristic — a known, non-blocking, log-only rough edge that was
  deliberately left alone).

### 1.6 If something's wrong

The fix lives entirely in `common/speculative.cpp`, in `common_speculative_impl_draft_mtp`
(`process()`, `draft()`, and the new `valid` member). `NOTES.md`'s "Implementation: item
5's fix" section documents exactly what changed and why, line references included — start
there before re-reading the whole file.

---

## 2. Testing the `resize()` core (Part 1, Phase 2)

This is the lower-level primitive: growing a live KV cache's capacity in place, without
touching `llama_context` or the public `llama_decode()` path yet. There's no user-facing
flag for this — it's tested via a dedicated unit test that calls the internal C++ API
directly.

> **If you hit `LNK2019: unresolved external symbol ... llama_kv_cache::...` when building
> these tests on Windows**: pull the latest commit on this branch and rebuild. Windows DLLs
> only export symbols explicitly marked for export, unlike Linux shared libraries which
> export everything by default — the internal `llama_kv_cache` methods these tests call
> directly weren't visible across the `llama.dll` boundary until this was fixed
> (`src/CMakeLists.txt` now sets `WINDOWS_EXPORT_ALL_SYMBOLS` on the `llama` target). This
> was found and fixed only after testing on a real Windows/MSVC build — the Linux sandbox
> used to write these tests masked the problem entirely. See `NOTES.md`'s "Real-hardware
> findings" section for the full story.

### 2.1 Run the unit test

```bat
cmake --build build --config Release --target test-kv-resize -j 8
ctest --test-dir build -C Release -R test-kv-resize --output-on-failure
```

This downloads a tiny reference model (`tinyllamas/stories15M-be.Q4_0.gguf`, a few MB)
the first time via the `test-download-model` CTest fixture, then runs three scenarios:
FA off (exercises the transposed-V row-by-row copy), FA on (contiguous-prefix copy), and
FA on with `q8_0` KV (quantized-type copy). Expected output ends with:

```
run_scenario: all scenarios passed
```

If `ctest` can't download the model (e.g. offline), run the built `.exe` directly against
any small local GGUF you already have:

```bat
build\bin\Release\test-kv-resize.exe -m C:\path\to\any-small-model.gguf
```

(It needs a *plain*, non-hybrid, non-recurrent architecture — e.g. a small Llama/Qwen2/
Mistral-style model. It'll print `skipping for recurrent/hybrid model` and exit 0 if you
point it at the wrong kind.)

### 2.2 What it's actually checking

For each of the three configs: decodes 400 tokens to fill part of the cache, snapshots
the raw K/V tensor bytes, then drives the internal `resize()` in both directions and
checks: a shrink that would drop cells still in use is correctly rejected (and the cache
stays usable afterward); a resize to the size the cache already has is a no-op success;
growing to a valid larger size succeeds, and shrinking back down again succeeds; the
cache's reported size updated correctly each time; the sequence's cached position range
didn't change; and the K/V bytes for the previously-decoded range are **byte-identical**
across both the grow and the shrink — i.e. no data corruption during either copy.

Note the test never decodes after a *successful* resize. A raw `resize()` leaves the
scheduler holding a graph sized for the old cache, and decoding through that crashes;
re-reserving is `llama_set_n_ctx()`'s job (§3), not `resize()`'s. Section 4 is where
decode-after-resize gets exercised.

### 2.3 GPU-specific things worth double-checking on your hardware

The transposed-V copy path uses `ggml_backend_tensor_get_2d`/`_set_2d` — these exist and
have a working CPU fallback (a per-row loop), but I couldn't confirm the CUDA backend's
own `set_tensor_2d`/`get_tensor_2d` implementation (if one exists and takes a different,
possibly-batched code path) actually gets exercised correctly, since there's no CUDA
here. `-fa off` forces the transposed-V path (`v_trans = true`) — run
`test-kv-resize.exe` after building with `-DGGML_CUDA=ON`, and separately watch for any
garbled output in a real `--flash-attn off` generation that runs across a growth
boundary (once Phase 5's CLI flags exist — see §3.4 below for how to exercise growth
without them today).

---

## 3. Testing `llama_set_n_ctx()` / auto-grow (Part 1, Phase 3)

This is the next layer up: `llama_set_n_ctx()` (the public C API to grow a live
context's KV capacity) and the hook inside `llama_decode()` that calls it automatically
when the cache runs out of room. **There are no CLI/server flags for this yet** (that's
the plan's Phase 5, not done) — right now this is only reachable through the C API
directly, which is exactly what the unit test below does.

### 3.1 Run the unit test

```bat
cmake --build build --config Release --target test-ctx-grow -j 8
ctest --test-dir build -C Release -R test-ctx-grow --output-on-failure
```

Uses the same tiny downloaded model as §2.1. For each of {FA off, FA on, FA on + `q8_0`
KV}, it builds **two** contexts from the same model:

- a "reference" context pre-allocated at a large `n_ctx` from the start, and
- a "growing" context that starts small with `n_ctx_max` set to that same large value.

It then feeds both contexts the **identical** 300-token sequence, one token at a time,
and after every single token compares the two contexts' logits. Expected output includes
a line like:

```
=== scenario: fa-off (v_trans) ===
auto-grow fired at pos 256: n_ctx 256 -> 512
run_scenario: fa-off (v_trans): OK (final n_ctx = 512)
```

followed by the same for the other two scenarios, ending with:

```
main: all scenarios passed
```

**This is the determinism gate that matters most**: if the growing context's logits ever
diverge from the static reference context's — even by a little, even just at the one
step where growth happens — the test fails loudly with the exact position and magnitude
of the divergence. That divergence is *exactly* the failure mode you'd see as garbled or
subtly-wrong generation in a real long conversation that crosses a growth boundary, so
this test failing is a "do not use this on real conversations yet" signal.

### 3.2 What could plausibly break specifically on your hardware

- **CUDA + FA off (`v_trans` path)**: same caveat as §2.3 — the row-by-row V copy on a
  non-CPU backend is unverified.
- **Larger, real growth boundaries**: the unit test only grows once, by a small amount
  (256 → 512 cells). Your actual use case (start at 8K, grow toward 131072 tokens across
  many turns) exercises this same code path repeatedly, at much larger absolute sizes —
  worth a longer manual soak (see §3.4) rather than trusting the unit test alone for that.
- **VRAM headroom during the copy**: `resize()`'s peak memory during a grow is old+new KV
  size for the layer group being copied (documented in the plan as the accepted v1
  tradeoff — no copy-free growth yet). On a 6 GB card with `--n-cpu-moe`-style CPU
  offload of most weights, the *KV cache itself* is usually only a few hundred MB even at
  large context, so this should be fine — but if you see an OOM specifically at the
  moment of a growth log line, that's the thing to suspect first.

### 3.3 Reading the growth log line in a real run

Once growth fires (either via the unit test or your own C-API test program, §3.4), watch
for these `LLAMA_LOG_INFO` lines (visible at the default log verbosity, no `-lv` needed):

```
llama_kv_cache: resizing KV cache: 8192 -> 16384 cells
llama_kv_cache:        CUDA0 KV buffer size =    64.00 MiB
llama_context: growing n_ctx: 8192 -> 16384 (n_ctx_seq: 8192 -> 16384)
```

If instead you see a decode that just fails (`llama_decode` returning `1`, "failed to
find a memory slot"), growth either isn't enabled (`n_ctx_max` wasn't set, or was set but
is still `<= n_ctx`) or isn't supported for that context's topology (`n_seq_max > 1`,
which growth is deliberately restricted away from for now).

### 3.4 What's explicitly *not* covered yet (don't be surprised)

- MTP's own side context (`ctx_dft`) now **does** grow in lockstep with `ctx_tgt` (see §4.5
  below) — reviewed against the code but not yet exercised with a real MTP-capable model in
  this sandbox, so your run is the first real check of it.
- `--cache-ram-mib` (the optional prompt-cache RAM feature) is sized once at load time and
  is not resized when the context grows. Untested combination; treat as unsupported until
  checked.

---

## 4. Testing `llama-server`/`llama-cli` with `--ctx-max` (Part 1, Phase 5)

This is the layer that actually makes growth usable from the command line — the
`--ctx-max`/`--ctx-grow-factor` flags, `llama-server`'s proactive per-request growth (grows
once, ahead of prefill, sized to fit the whole request instead of waiting to get caught out
mid-generation), and keeping the server's own bookkeeping (`slot.n_ctx`, `/props`) in sync
with whatever the KV cache actually grew to.

### 4.1 The command

```bat
build\bin\Release\llama-server.exe -m your-qwen3.6-mtp-model.gguf ^
    -c 8192 --ctx-max 131072 -np 1 --temp 0 --seed 1
```

- `-np 1` (`--parallel 1`) is **required** whenever `--ctx-max` is set — growth only
  supports a single sequence/slot for now. The server refuses to start with a clear error
  if you set `--ctx-max` with `--parallel` anything else.
- If you omit `-c` entirely, it defaults to `min(8192, --ctx-max)` automatically (logged at
  startup) rather than starting at the model's full training context — the whole point is
  to start small and grow.
- Combine with `--spec-type draft-mtp` if you want MTP + growth together — `ctx_dft` (MTP's
  own side context) now grows in lockstep with `ctx_tgt` automatically. This has not been
  exercised with a real MTP-capable model yet (see §4.5), so watch closely the first time
  a conversation grows past `ctx_dft`'s original size while MTP is active.

### 4.2 What to watch for

At the **default log verbosity**, a growth event logs one of these `slot`-tagged lines
(from the server's own logging, always visible):

```
slot maybe_grow_f: id  0 | task N | grew KV cache ahead of prefill: n_ctx 8192 -> 16384 (prompt = ... tokens, n_predict = ...)
slot maybe_grow_m: id  0 | task N | grew KV cache mid-generation: n_ctx 8192 -> 16384
```

- The first (`maybe_grow_for_request`) fires *proactively*, right before prefill, when a
  request's prompt + `n_predict` doesn't fit yet — sized exactly to what's needed.
- The second (`maybe_grow_mid_generation`) fires *reactively*, mid-generation — e.g. an
  open-ended request (no `n_predict`, or `-1`) that organically outgrows what the first one
  sized at prompt time. This one steps by `--ctx-grow-factor` (1.5x by default), so expect
  bigger jumps than the exact-fit proactive case.

**You will *not* see** `llama_kv_cache: resizing KV cache: ...` or `llama_context: growing
n_ctx: ...` at the default verbosity — those come from the `llama`/`ggml` library layer
directly and are mapped to `TRACE` level by the server's own logging, regardless of growth;
the same is true of other library-level messages like the model's own load-time `n_ctx =
...` line. Run with `-lv 4` if you want to see those too (same flag as §1.5's MTP trace
debugging).

Check `/props` after a growth event — `n_ctx` there should reflect the new, larger size,
not the value from server startup.

### 4.3 Things to actually try

- A single long multi-turn conversation that crosses at least one growth boundary — confirm
  the conversation just keeps going with no restart, no error, and (this is the important
  part) **the model doesn't seem to have "forgotten" or garbled earlier turns** right around
  the growth point. That's the real-world version of the determinism check in §3.1.
- An open-ended generation (no `n_predict`, or `n_predict: -1`) that runs long enough to
  organically outgrow the initial `-c` size *without* a large prompt — this exercises
  `maybe_grow_mid_generation` specifically (the proactive path only sizes for what the
  request declares up front). Confirm you see `grew KV cache mid-generation` and the
  conversation keeps going, rather than the turn ending early with `truncated = 1` in the
  log at the original `-c` size — that was a real bug (fixed) where generation could stop
  before growth ever got a chance to run; see `NOTES.md`'s "Real-hardware findings, round 3"
  for the full diagnosis if you still see it.
- A prompt long enough to need growth *beyond* `--ctx-max` — confirm you get the normal
  "exceeds the available context size" error (referencing the current ceiling), not a
  crash, and the server stays usable for the next request afterward.
- Try `--ctx-max` with `--parallel 2` (or any value other than 1) — confirm the server
  refuses to start with a clear error rather than silently ignoring `--ctx-max`.

### 4.4 What was verified this session, and what wasn't

I verified the mechanism end-to-end against a real `llama-server` **process** (not just a
unit test) using a synthetic model with real (if randomly-initialized) weights written to
an actual `.gguf` file, driven over real HTTP with `curl`: proactive growth fired and logged
correctly for an oversized prompt, growth correctly clamped to `--ctx-max` and then
correctly rejected a still-too-large request afterward (with the error referencing the
*post-growth* ceiling), and the server correctly refused to start with `--ctx-max` +
`--parallel 2`. What I could **not** verify in that pass: an actual multi-turn conversation
generating real text across a growth boundary — the synthetic model has a placeholder
tokenizer that crashes when formatting any *generated* token back into text (unrelated to
growth; confirmed the crash happens strictly after decode/growth already succeeded). That
means **your test on real hardware with a real tokenizer is the first time this will
actually be exercised for real generated output**, not just log lines. See `NOTES.md`'s
"Part 1 Phase 5 implementation" section for the full verification writeup.

### 4.5 Testing MTP + growth together (the lockstep fix)

Watch for this log line once a growth event fires while `--spec-type draft-mtp` is active:

```
slot maybe_grow_f: id  0 | task N | grew MTP draft context in lockstep: n_ctx 8192 -> 16384
```

If instead you see `failed to grow MTP draft context in lockstep`, that's worth reporting —
it means `ctx_dft` didn't grow, so drafting is likely to degrade (garbled/lower-quality
completions, or a possible error) once the conversation runs past `ctx_dft`'s old size.
This is the first real-hardware exercise of this code path, so treat any MTP-specific
weirdness right around a growth boundary as suspect and worth flagging.

### 4.6 If the server crashes on startup with `GGML_ASSERT(params.n_gpu_layers < 0) failed`

This was a real bug hit on the first Windows test of Phase 5 — a struct-layout issue caused
by fields added earlier in this branch, now fixed (moved to the end of the affected
structs, matching the codebase's own "append new fields" convention). If you still see this
after pulling the latest commit: do a **full clean rebuild** (not just an incremental one —
`rmdir /s build` and reconfigure, or at minimum rebuild every target, not just
`llama-server`) before assuming it's still broken. See `NOTES.md`'s "Real-hardware
findings, round 2" section for the full diagnosis.

---

## 5. Testing automatic context *shrinking*

`--ctx-max` grows the KV cache on demand. It now also gives cells back: when a request
needs far less room than the cache is holding, the server shrinks it before prefill. This
is for the agent-loop pattern where a session fills the context, writes a handoff file,
and is replaced by a fresh instance starting from a short prompt - without it, the cache
stays at the session high-water mark for the life of the server.

The shrink is deliberately lazy. It only fires when the target is at most half of what is
currently allocated (a resize copies the whole cache, so a small saving is not worth
paying for), it never goes below the `-c/--ctx-size` the server launched with, and it only
drops cells the request was going to discard anyway.

### 5.1 The command

```bat
build-cuda\bin\Release\llama-server.exe -m your-model.gguf ^
  -c 8192 --ctx-max 262144 --parallel 1 --port 8080
```

### 5.2 What to watch for

Send a long request, then a short one. The log should show both directions:

```
slot maybe_grow_f: id  0 | task 0  | grew KV cache ahead of prefill: n_ctx 8192 -> 5120 ...
slot maybe_shrink: id  0 | task 18 | shrank KV cache ahead of prefill: n_ctx 5120 -> 8192 (prompt = 5 tokens, reused = 0)
```

and, from libllama itself:

```
llama_context::set_n_ctx: n_ctx: 204800 -> 16384 (n_ctx_seq: 204800 -> 16384)
llama_kv_cache::resize: resizing KV cache: 204800 -> 16384 cells
```

Then confirm generation still works *after* the shrink - that is the part worth watching,
since it depends on the graph being re-reserved at the new size. A cycle of
long-then-short requests exercises grow and shrink repeatedly against the same context.

### 5.3 The expert hot store

On a MoE model with `-ehs`, growth already drops hot expert slots to make room for the KV
cache. Shrinking now runs that in reverse: once the cache has released its VRAM, the store
is taken back up towards the slot count it was asked for at load, bounded by what the
device reports free. The store is never taken above its original `-ehs` value - the fitter
already weighed that number against everything else on the device.

An easy way to force the whole cycle without waiting on a huge prefill: growth is sized
from `prompt + n_predict`, so a *one-token* prompt with a large `n_predict` grows the cache
at prefill time. Fire it, let the client disconnect, then send an ordinary short request to
trigger the shrink.

```bat
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Hello\",\"n_predict\":70000,\"cache_prompt\":false}" --max-time 30
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"The capital of France is\",\"n_predict\":24,\"cache_prompt\":false}"
```

### 5.4 What was verified

**Small model, mechanics only** (`stories15M-q4_0.gguf`, over HTTP): four full grow/shrink
cycles (512 -> 5120 -> 512), each followed by correct generation, plus the `resize()` unit
test's byte-identity checks in both directions.

**Real model, hot store included**: `Qwen3.6-35B-A3B-UD-Q4_K_S` (19.9 GiB) on a 6 GB
RTX 3060 Laptop, launched with `-c 2048 --ctx-max 131072 -np 1 -ehs -1 -fitt 256`. At load
the fitter took 27 slots (2389 MiB, ~88 MiB/slot) leaving 567 MiB free, with the KV at
20 KiB/cell. Two grow/shrink cycles:

```
set_n_ctx: KV needs 1370 MiB with 424 MiB free, dropping 14 hot expert slots
resize: expert hot store 13 slots, 1433 MiB
resize: resizing KV cache: 2048 -> 70144 cells
...
resize: resizing KV cache: 70144 -> 2048 cells
resize: expert hot store 27 slots, 2389 MiB
```

The store came back to the full 27 slots both times, generation after each shrink was
coherent, and the hot hit rate settled at 15-23%, i.e. the re-planted store is actually
being routed through and not merely allocated. VRAM ended at 5489 MiB used / 506 MiB free
against 5428 / 567 at load.

Worth knowing: the second cycle dropped 17 slots where the first dropped 14, because the
free-VRAM reading that drives the drop decision was lower (252 vs 424 MiB). That reading is
taken inside `set_n_ctx()` mid-request, not at idle, so it is not directly comparable to
what `nvidia-smi` shows between requests. The store regrows to the same count afterwards
either way, so this is a deeper dip mid-cycle, not a permanent loss.

**A resize does not leak VRAM.** Measured at idle across a full cycle with `-ehs 0`, so the
hot store is not a confounder (used / free MiB):

| | plain | with `--spec-type draft-mtp` |
| --- | --- | --- |
| after load (n_ctx 2048) | 4573 / 1422 | 4443 / 1552 |
| after grow to 70144 | 5800 / 195 | 5857 / 138 |
| after shrink to 2048 | 4492 / 1503 | 4475 / 1520 |

Both configs come back to their starting footprint (within ~30 MiB), so the compute buffers
are *not* ratcheting up across a grow/shrink cycle in practice, and neither is `ctx_dft`.
The `// TODO: also consider shrinking the buffer` in `llama-context.cpp` is still real, but
it is not costing anything measurable here.

**With MTP drafting** (same model and flags plus `--spec-type draft-mtp
--spec-draft-n-max 3`): `ctx_dft` is a separate context for this architecture, and it
tracks `ctx_tgt` in both directions. Two cycles, no `failed to resize` warnings:

```
srv  refresh_n_ct: resized MTP draft context in lockstep: n_ctx 2048 -> 70144
...
srv  refresh_n_ct: resized MTP draft context in lockstep: n_ctx 70144 -> 2048
```

The shrink direction is the one that could have gone wrong - a resize is refused while any
cell above the target is still live, and `ctx_dft` holds its own cells. It works because
the shrink helper's `seq_rm` runs through `common_memory`, which covers both contexts, so
`ctx_dft` is already clear by the time it is asked to shrink.

Drafting keeps working across the boundary: acceptance was 0.595 (mean len 2.79) before any
resize, then 0.771 / 3.25 and 0.537 / 2.61 after the two shrinks - i.e. inside the normal
run-to-run spread, not degraded. Generation stayed coherent throughout.

Under MTP the mid-cycle dip is deeper: the store fell to 3 slots during the second grow
before coming back to 20. `ctx_dft` holds VRAM of its own, so the ceiling is 20 rather than
27 here.

Read that "3 slots" carefully, though - it is not 3 slots' worth of VRAM. The hot tensor
holds `n_expert_used + hot_s` slices, because every cold draw position needs its own zero
slice (the landing pad, see `llama-expert-hotstore.h`). At 68.3 MiB per slice and
`n_expert_used = 8`, the pad alone is 546 MiB that can never be freed:

| hot_s | slices | buffer |
| --- | --- | --- |
| 27 | 8 + 27 = 35 | 2389 MiB |
| 20 | 8 + 20 = 28 | 1911 MiB |
| 3 | 8 + 3 = 11 | 751 MiB |

So at the bottom of the dip, 73% of the store's 751 MiB is pad and only 205 MiB is resident
experts - 3 of 256, about 1.2% coverage. That is close to the floor of what the store can
usefully be, and it is the thing to watch on a card this tight if drafting quality ever
looks off right after a growth.

---

## 6. Testing the context budget (`--ctx-limit` / `n_ctx_limit`)

Stops generation cleanly once the context reaches N tokens, wherever that lands. The point
is to let an agent loop stop *before* the context is exhausted, leaving room to write a
handoff, without having to guess from the outside where generation will end.

### 6.1 The command

```bat
build-cuda\bin\Release\llama-server.exe -m your-model.gguf ^
  -c 8192 --ctx-limit 6000 --port 8080
```

Individual requests override it with `"n_ctx_limit": N` in the request body.

### 6.2 The three cases, and what each should return

All of these end with `"stop_type": "ctx_limit"`. `tokens_cached` reports how much context
was actually used.

| case | what happens |
| --- | --- |
| limit reached during output | generation stops at the wall; `tokens_evaluated + tokens_predicted` lands on the limit |
| limit reached during reasoning | same - the check does not care what the model is in the middle of |
| prompt alone is over the limit | prefill stops at the wall, **zero** tokens are generated, `truncated` is `true` |
| cached prefix already at the limit | nothing to process; returns immediately, cache untouched, zero tokens |

The last two are the deliberate ones: the prompt's tail is simply never processed, so a
prompt longer than the limit is not an error. An agent can read part of a large file and
still be told to stop and hand off, rather than having the whole request rejected.

**Do not disambiguate by arithmetic.** `usage.prompt_tokens` counts the whole submitted
prompt even when only part of it was evaluated, so `total_tokens == n_ctx_limit` does *not*
hold for an over-limit prompt. `stop_type` is emitted next to `finish_reason` on the
OpenAI-compatible routes for exactly this reason - `finish_reason` is `length` for both a
context-budget stop and an ordinary `max_tokens` stop.

### 6.2.1 Regression: a limit at or below the cached prefix

This crashed the server before it was fixed, and it needs a *warm* cache to reproduce - the
same request against a cold slot has `n_past == 0` and takes a different path. Send any
request, then repeat it with a limit below the prompt length:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Think, then write about steel."}],
       "max_tokens":4000,"n_ctx_limit":80}'
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Think, then write about steel."}],
       "max_tokens":4000,"n_ctx_limit":10}'
```

The second must return `stop_type: ctx_limit` with `completion_tokens: 0` and the server
must still be alive. The log line to look for is:

```
slot operator (): cached prefix already reaches the context limit, nothing to process (n_past = 20, task.n_tokens = 20, n_ctx_limit = 10)
```

The original bug was cutting `n_past` back to `n_ctx_limit - 1` and letting prefill continue
from there. On a recurrent or hybrid memory that rollback is not always possible, and
`common_context_seq_rm()` answers an impossible removal with `GGML_ABORT` - the process
dies. It is worth re-testing on a **hybrid** model specifically (the Qwen3.6-35B-A3B family
is one); a plain attention model can roll back freely and never showed the fault.

### 6.3 Verified

Against `stories15M-q4_0.gguf` over HTTP with `--ctx-limit 60`:

- 10-token prompt, `n_predict 400` -> stopped after 50 generated tokens, 60 total,
  `stop_type: ctx_limit`
- 373-token prompt -> prefilled to exactly 60, 0 generated, `truncated: true`,
  `stop_type: ctx_limit`
- per-request `n_ctx_limit` above the slot's own `n_ctx` -> correctly not the binding
  constraint; the pre-existing out-of-context stop takes over
- `n_ctx_limit: 0` -> disabled, ordinary `n_predict` stop

On `Qwen3.6-35B-A3B-UD-Q4_K_S` (hybrid + MTP + `-ehs -1` + checkpoints), which is where the
rollback fault lived:

- the two-request regression above, run repeatedly - server survives, `completion_tokens: 0`
- a conversation grown across a fixed `n_ctx_limit` of 400 over seven turns: prompt grows
  18 -> 375 while the total stays pinned at ~400, and a turn that finishes early still
  reports `stop_type: eos`, so the budget does not interfere with ordinary stops
- a 550-token prompt against limits of 400, 400 (warm), and 100 (warm) -> all return
  `ctx_limit` with zero tokens; raising the limit to 4000 afterwards generates normally and
  ends on `eos`, i.e. the slot is not left in a poisoned state
