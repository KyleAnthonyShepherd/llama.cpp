# PLAN - keep context checkpoints out of VRAM

Companion to `PLAN-qwen36-35b-moe.md` (KV sizing) and `PLAN-adaptive-draft-width.md`.
Same 6 GB VRAM / 16 GB RAM box. No code changed yet. Every structural claim cites a file:line
in this tree.

---

## 0. What this document decides

Whether anything has to be built at all.

**Outcome: nothing to build.** Context checkpoints already live in host RAM and cost zero VRAM,
and their bytes are already capped and reported. Section 2 records why the byte budget this plan
was opened for is unnecessary.

| Step | What | Code | Decides |
|---|---|---|---|
| 0 | Confirm on the box that VRAM does not move with `--ctx-checkpoints` | none | whether step 1 is the right target |
| 1 | ~~Cap checkpoint RAM by bytes, not by count~~ | none | **closed, already capped by `--cache-ram`** |

---

## 1. Checkpoints are already host memory. The trace.

- The list lives on the slot: `std::list<common_prompt_checkpoint> checkpoints`
  (`tools/server/server-task.h:613`).
- The payload is three plain host vectors: `data_tgt`, `data_dft`, `data_spec`
  (`common/common.h:1133-1146`).
- `update_tgt` sizes `data_tgt` from `llama_state_seq_get_size_ext` and fills it with
  `llama_state_seq_get_data_ext` (`common/common.cpp:2191-2207`).
- That call ends in `llama_io_write_host` (`src/llama-context.cpp:2911-2960`). Plain writes
  `memcpy` into the caller's buffer. Tensor writes are queued and flushed in the destructor with
  `ggml_backend_tensor_get(winfo.tensor, winfo.ptr, ...)` - a D2H copy into that same host buffer.
  **No device buffer is created anywhere on this path.**
- The KV side only reads existing cache tensors: `state_write_data` calls `io.write_tensor` on
  `layer.k_stream` / `layer.v_stream` and creates no scratch tensor
  (`src/llama-kv-cache.cpp:2309-2405`).
- Restore is the mirror: `llama_io_read_host` queues `ggml_backend_tensor_set`
  (`src/llama-context.cpp:2962-3000`).

So `--ctx-checkpoints N` costs **host RAM plus PCIe traffic on create and restore, and zero VRAM**.

Second point worth recording: the server saves with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
(`tools/server/server-context.cpp:2444-2445`), and on a hybrid memory that flag skips the attention
KV entirely (`src/llama-memory-hybrid.cpp:191-199`). On qwen35moe a checkpoint therefore holds the
linear-attention recurrent state, not 40 layers of KV. That is why the `size = %.3f MiB` traces are
small.

### 1.1 Confirming it on the box, zero code

1. `llama_get_memory_breakdown` (`src/llama-ext.h:100`) already backs the startup report. Run the
   same prompt at `--ctx-checkpoints 0` and `--ctx-checkpoints 32`, same `--ctx-size`. The
   `context` and `compute` lines must be byte-identical.
2. `nvidia-smi --query-gpu=memory.used --format=csv -l 1` across a long generation with `-lv 1`.
   Every `created context checkpoint ... size = X MiB` trace should have **no** matching VRAM step.

If VRAM does move under checkpoints, the cause is in section 3, not here, and this plan is wrong.

---

## 2. CLOSED - the bytes are already capped, by `--cache-ram`

This plan was opened to add a byte budget. It is not needed. `server_prompt_cache_state::size()`
sums `data.size()` **plus every checkpoint on the prompt**
(`tools/server/server-task.h:645-653`), and `server_prompt_cache::update()` evicts oldest until
`size() <= limit_size` (`tools/server/server-task.cpp:1852-1857`). `limit_size` comes from
`--cache-ram`, which defaults to **8192 MiB** (`common/common.h:625`, `common/arg.cpp:1786-1791`).

The trace line that shows it, from a live run:

```
srv update:  - cache state: 1 prompts, 214.813 MiB (limits: 8192.000 MiB, 512 tokens, 51444 est)
srv update:    - prompt 00000202B5F2FB80: 1349 tokens, checkpoints:  2, 214.813 MiB
```

`checkpoints: 2` and the MiB on the same line are the same accounting the 8192 MiB limit is
applied to (`tools/server/server-task.cpp:1876-1881`). So the count cap `--ctx-checkpoints`
(default 32, `common/common.h:623`) is the *inner* bound and `--cache-ram` is the outer one, and
the outer one is in bytes.

**One residual, recorded and not acted on.** A prompt is only in that accounting once it has been
moved into the cache. The checkpoints on an *actively generating* slot are outside it until then,
bounded only by `n_slots x n_ctx_checkpoints`. At the ~107 MiB per checkpoint the trace above
implies, a slot that reached the full 32 would hold ~3.4 GB uncounted. In practice
`checkpoint_min_step` (default 8192 tokens, `common/common.h:624`) means a slot rarely holds more
than a few, and the bytes join the accounting the moment the prompt is cached. Not worth code
until a run is actually seen to hit it.

---

## 3. If the intent was "move VRAM to RAM", these are the tenants that actually move

Checkpoints are not a VRAM tenant. These are, in the order they are worth touching:

- **KV cache.** The tenant that scales with context. Already handled by `--ctx-grow-headroom` and
  `--ctx-limit` (56110ca4c) plus `shrink_expert_hotstore_for_kv`
  (`src/llama-context.cpp:1058-1107`). Next lever is `-ctk q8_0 -ctv q8_0`.
  `-nkvo` / `--no-kv-offload` (`common/arg.cpp:2485-2489`) does literally move the KV to host, but
  it puts attention on the host bus every single token. On a box that is already RAM-bandwidth
  bound through the MoE cold path, that trades the cheap tenant for the expensive one. **Do not.**
- **Expert hot store.** A deliberate VRAM tenant. `llama_expert_hotstore_refit`
  (`src/llama-ext.h:96`) and the shrink path already arbitrate it against the KV cache.
- **Compute buffers.** Scale with `n_ubatch` and with the speculative verify width. That last one
  is the subject of `PLAN-adaptive-draft-width.md`.
